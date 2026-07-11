from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, overload

import joblib
import numpy as np
import polars as pl
from joblib_progress import joblib_progress
from loguru import logger
from numpy.typing import NDArray
from pgenlib import PgenReader
from rich.progress import MofNCompleteColumn, Progress
from seqpro.rag import Ragged, lengths_to_offsets

from .._io import atomic_write_dir
from .._types import DOSAGE_TYPE, POS_TYPE, V_IDX_TYPE
from .._utils import format_memory, parse_memory
from .._vcf import VCF, Filter
from ._io import (
    _open_fmt,
    _open_genos,
    _subset_var_idxs_and_recompute_af,
    _write_dosages,
    _write_genos,
    _write_index_from_working,
)
from ._kernels import (
    _copy_chunk_dosages_helper,
    _copy_chunk_helper,
    _dense2sparse_count,
    _dense2sparse_fill,
)


@overload
def dense2sparse(
    genos: NDArray[np.int8],
    var_idxs: NDArray[V_IDX_TYPE],
    dosages: None = None,
) -> Ragged[V_IDX_TYPE]: ...
@overload
def dense2sparse(
    genos: NDArray[np.int8],
    var_idxs: NDArray[V_IDX_TYPE],
    dosages: NDArray[DOSAGE_TYPE],
) -> tuple[Ragged[V_IDX_TYPE], Ragged[DOSAGE_TYPE]]: ...
def dense2sparse(
    genos: NDArray[np.int8],
    var_idxs: NDArray[V_IDX_TYPE],
    dosages: NDArray[DOSAGE_TYPE] | None = None,
) -> Ragged[V_IDX_TYPE] | tuple[Ragged[V_IDX_TYPE], Ragged[DOSAGE_TYPE]]:
    """Convert dense genotypes (and dosages) to sparse genotypes."""
    # (s p v)
    if genos.ndim < 3:
        raise ValueError(
            "Sparse genotypes must have at least 3 dimensions, with the final three dimensions corresponding"
            + " to (samples, ploidy, variants)"
        )
    if dosages is not None:
        if dosages.ndim < 2:
            raise ValueError(
                "Sparse dosages must have at least 2 dimensions, with the final two dimensions corresponding"
                + " to (samples, variants)"
            )
        if dosages.shape[-1] != genos.shape[-1]:
            raise ValueError(
                "Sparse dosages must have the same number of variants as the genotypes"
            )
        if dosages.shape[-2] != genos.shape[-3]:
            raise ValueError(
                "Sparse dosages must have the same number of samples as the genotypes"
            )

    keep = genos == 1
    data = var_idxs[keep.nonzero()[-1]]
    lengths = keep.sum(-1)
    shape = (*lengths.shape, None)
    offsets = lengths_to_offsets(lengths)
    rag = Ragged[V_IDX_TYPE].from_offsets(data, shape, offsets)

    if dosages is not None:
        # (s v) -> (s p v)
        dosage_data = np.broadcast_to(dosages[:, None], genos.shape)[keep]
        _dosages = Ragged[DOSAGE_TYPE].from_offsets(dosage_data, shape, offsets)
        return rag, _dosages
    return rag


def _dense2sparse_with_length(
    genos: NDArray[np.integer],
    var_idxs: NDArray[V_IDX_TYPE],
    q_start: int,
    q_end: int,
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    dosages: NDArray[DOSAGE_TYPE] | None = None,
) -> Ragged[V_IDX_TYPE] | tuple[Ragged[V_IDX_TYPE], Ragged[DOSAGE_TYPE]]:
    """Convert a dense ``with_length`` window (shared, over-extended across all
    samples/haplotypes) into per-haplotype-minimal sparse output, identical to
    ``SparseVar.read_ranges_with_length`` for the same query.

    Parameters
    ----------
    genos
        Dense genotypes for the window. Shape: (samples, ploidy, variants).
    var_idxs
        Global variant indices of the window, used only to populate the sparse
        output. Shape: (variants,).
    q_start, q_end
        0-based, half-open original query span (before extension).
    v_starts
        0-based start positions of the window's variants (i.e. POS - 1).
        Window-aligned: same length as the ``genos`` variant axis and
        positionally aligned with ``var_idxs`` (NOT a global per-dataset array).
        Shape: (variants,).
    ilens
        ILEN of the window's variants (ALT - REF length). Window-aligned, like
        ``v_starts``. Shape: (variants,).
    dosages
        Optional dense dosages. Shape: (samples, variants).

    Returns
    -------
        ``Ragged[V_IDX_TYPE]`` of shape (samples, ploidy, ~variants), or a tuple
        with a matching ``Ragged[DOSAGE_TYPE]`` when ``dosages`` is given.
    """
    # single-range only: exactly (samples, ploidy, variants), no batch dimension
    if genos.ndim != 3:
        raise ValueError("Dense genotypes must have shape (samples, ploidy, variants).")
    n_samples, ploidy, _ = genos.shape

    v_starts = np.ascontiguousarray(v_starts, dtype=np.int32)
    ilens = np.ascontiguousarray(ilens, dtype=np.int32)
    var_idxs = np.ascontiguousarray(var_idxs, dtype=V_IDX_TYPE)

    # Pass 1: per-haplotype kept counts (reuses the sparse path's length walk).
    lengths = np.empty((n_samples, ploidy), dtype=np.int64)
    _dense2sparse_count(
        genos, v_starts, ilens, POS_TYPE(q_start), POS_TYPE(q_end), lengths
    )

    flat_offsets = lengths_to_offsets(lengths)
    total = int(flat_offsets[-1])
    shape = (n_samples, ploidy, None)

    # Pass 2: fill the (and optionally the dosage) output in disjoint ranges.
    out_data = np.empty(total, dtype=V_IDX_TYPE)
    has_dose = dosages is not None
    dose_in = (
        np.ascontiguousarray(dosages, dtype=DOSAGE_TYPE)
        if has_dose
        else np.empty((0, 0), dtype=DOSAGE_TYPE)
    )
    out_dose = np.empty(total if has_dose else 0, dtype=DOSAGE_TYPE)
    _dense2sparse_fill(
        genos, var_idxs, dose_in, lengths, flat_offsets, out_data, out_dose, has_dose
    )

    rag = Ragged[V_IDX_TYPE].from_offsets(out_data, shape, flat_offsets)
    if has_dose:
        drag = Ragged[DOSAGE_TYPE].from_offsets(out_dose, shape, flat_offsets)
        return rag, drag
    return rag


def _process_contig_vcf(
    path: str | Path,
    dosage_field: str | None,
    max_mem: int | str,
    contig: str,
    chunk_dir: Path,
    chunk_idx: int,
    cyvcf2_filter: Callable[..., bool] | None = None,
    pl_filter: pl.Expr | None = None,
    caller_samples: list[str] | None = None,
    keep_local: np.ndarray | None = None,
    haploid: bool = False,
) -> tuple[int, int]:
    vcf_filter: Filter | None = None
    if cyvcf2_filter is not None:
        assert pl_filter is not None, (
            "cyvcf2_filter and pl_filter must be provided together"
        )
        vcf_filter = Filter(record=cyvcf2_filter, expr=pl_filter)

    vcf = VCF(
        path,
        filter=vcf_filter,
        dosage_field=dosage_field,
        with_gvi_index=False,
    )
    if caller_samples is not None:
        vcf.set_samples(caller_samples)

    if dosage_field is not None:
        chunker = vcf.chunk(contig, max_mem=max_mem, mode=VCF.Genos8Dosages)
    else:
        chunker = vcf.chunk(contig, max_mem=max_mem, mode=VCF.Genos8)

    keep_sorted = None if keep_local is None else np.asarray(keep_local, dtype=np.int64)

    total_vars = 0
    n_chunks = 0
    contig_local_pos = 0  # running filtered-record position within this contig

    # Create a subdirectory for this contig to avoid collision
    contig_dir = chunk_dir / f"c{chunk_idx}"
    contig_dir.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(chunker):
        if isinstance(data, tuple):
            genos, dosages = data
        else:
            genos = data
            dosages = None

        n_in = genos.shape[-1]
        if keep_sorted is not None:
            # positions in [contig_local_pos, contig_local_pos + n_in) that are kept
            lo = np.searchsorted(keep_sorted, contig_local_pos)
            hi = np.searchsorted(keep_sorted, contig_local_pos + n_in)
            sel = keep_sorted[lo:hi] - contig_local_pos
            contig_local_pos += n_in
            genos = genos[..., sel]
            if dosages is not None:
                dosages = dosages[..., sel]
        else:
            contig_local_pos += n_in

        if haploid:
            # OR across haplotypes -> single haploid slot. Uses `== 1`, the same
            # predicate dense2sparse keys on, so the haploid call set equals the
            # union of the per-haplotype call sets. Dosages keep their (s, v)
            # shape and broadcast against the collapsed genos in dense2sparse.
            genos = (genos == 1).any(axis=-2, keepdims=True).astype(np.int8)

        n_vars = genos.shape[-1]
        if n_vars == 0:
            continue

        out_path = contig_dir / str(n_chunks)
        out_path.mkdir(parents=True, exist_ok=True)
        n_chunks += 1

        var_idxs = np.arange(total_vars, total_vars + n_vars, dtype=np.int32)
        if dosages is not None:
            sp_genos, sp_dosages = dense2sparse(genos, var_idxs, dosages)
            _write_genos(out_path, sp_genos)
            _write_dosages(out_path, sp_dosages.data)
        else:
            sp_genos = dense2sparse(genos, var_idxs)
            _write_genos(out_path, sp_genos)
        total_vars += n_vars
    return total_vars, n_chunks


def _process_contig_pgen(
    geno_path: str | Path,
    dosage_path: str | Path | None,
    max_mem: int,
    keep_idxs: np.ndarray,
    mem_per_var: int,
    n_samples: int,
    ploidy: int,
    chunk_dir: Path,
    chunk_idx: int,
    sample_subset: np.ndarray | None = None,
    haploid: bool = False,
) -> tuple[int, int]:
    geno_reader = PgenReader(bytes(Path(geno_path)), n_samples)
    dose_reader = (
        PgenReader(bytes(Path(dosage_path))) if dosage_path is not None else None
    )

    # --- sample subset: pgenlib requires sorted ascending indices ---
    unsorter: np.ndarray | None = None
    if sample_subset is not None:
        ss = np.ascontiguousarray(sample_subset, dtype=np.uint32)
        sorter = np.argsort(ss, kind="stable")
        sorted_ss = ss[sorter]
        unsorter = np.argsort(sorter, kind="stable")
        geno_reader.change_sample_subset(sorted_ss)
        if dose_reader is not None:
            dose_reader.change_sample_subset(sorted_ss)
        n_out = len(ss)
    else:
        n_out = n_samples

    keep_idxs = np.ascontiguousarray(keep_idxs, dtype=np.uint32)
    n_total = int(len(keep_idxs))
    vars_per_chunk = min(max_mem // mem_per_var, n_total) if n_total else 0
    if n_total and vars_per_chunk == 0:
        raise ValueError(
            f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
            + f" Memory per variant: {format_memory(mem_per_var)}."
        )

    # Create a subdirectory for this contig to avoid collision
    contig_dir = chunk_dir / f"c{chunk_idx}"
    contig_dir.mkdir(parents=True, exist_ok=True)

    total_vars = 0
    n_chunks = 0
    for i, c0 in enumerate(range(0, n_total, vars_per_chunk) if n_total else []):
        idxs = keep_idxs[c0 : c0 + vars_per_chunk]
        n_vars = int(len(idxs))
        if n_vars == 0:
            continue
        n_chunks += 1

        out_path = contig_dir / str(i)
        out_path.mkdir(parents=True, exist_ok=True)

        # Read genotypes for exactly the kept variant indices.
        # pgenlib returns (v, n_out*p) where n_out samples are in sorted_ss order.
        genos = np.empty((n_vars, n_out * ploidy), dtype=np.int32)
        geno_reader.read_alleles_list(idxs, genos)
        genos = genos.astype(np.int8)
        # (v, s, p) -> (s, p, v)
        genos = genos.reshape(n_vars, n_out, ploidy).transpose(1, 2, 0)
        genos[genos == -9] = -1
        # Restore caller-order from sorted pgenlib order
        if unsorter is not None:
            genos = genos[unsorter]

        if haploid:
            # OR across haplotypes -> single haploid slot. `ploidy` above is the
            # native read ploidy (needed to reshape pgenlib output); the collapse
            # changes only the STORED ploidy. Dosages keep (s, v) and broadcast.
            genos = (genos == 1).any(axis=-2, keepdims=True).astype(np.int8)

        dosages = None
        if dose_reader is not None:
            dosages = np.empty((n_vars, n_out), dtype=np.float32)
            dose_reader.read_dosages_list(idxs, dosages)
            # (v, s) -> (s, v)
            dosages = dosages.transpose(1, 0)
            dosages[dosages == -9] = np.nan
            if unsorter is not None:
                dosages = dosages[unsorter]

        # Convert to sparse
        var_idxs = np.arange(total_vars, total_vars + n_vars, dtype=np.int32)
        if dosages is not None:
            sp_genos, sp_dosages = dense2sparse(genos, var_idxs, dosages)
            _write_genos(out_path, sp_genos)
            _write_dosages(out_path, sp_dosages.data)
        else:
            sp_genos = dense2sparse(genos, var_idxs)
            _write_genos(out_path, sp_genos)

        total_vars += n_vars
    return total_vars, n_chunks


def _concat_data(
    out_path: Path,
    chunk_dir: Path,
    shape: tuple[int, int],
    contig_results: list[tuple[int, int]],
    with_dosages: bool = False,
):
    out_path.mkdir(parents=True, exist_ok=True)

    # Flatten chunk directories and calculate offsets
    chunk_offsets: list[int] = []
    contig_offset = 0
    global_chunk_idx = 0

    for chunk_idx, (n_vars, n_chunks) in enumerate(contig_results):
        contig_subdir = chunk_dir / f"c{chunk_idx}"

        if n_chunks > 0:
            for i in range(n_chunks):
                src = contig_subdir / str(i)
                dest = chunk_dir / str(global_chunk_idx)
                src.rename(dest)
                chunk_offsets.append(contig_offset)
                global_chunk_idx += 1

        if contig_subdir.exists():
            shutil.rmtree(contig_subdir)

        contig_offset += n_vars

    # [1, 2, 3, ...]
    chunk_dirs = [chunk_dir / str(i) for i in range(len(chunk_offsets))]

    vars_per_sp = np.zeros(shape, dtype=np.int32)

    # Pass 1: Compute lengths
    # We explicitly map only the offsets to avoid mapping the potentially large variant_idxs
    for c_dir in chunk_dirs:
        # Load offsets
        chunk_offsets_arr = np.memmap(c_dir / "offsets.npy", dtype=np.int64, mode="r")
        # Compute lengths: (n_samples * ploidy,) -> (n_samples, ploidy)
        chunk_lengths = np.diff(chunk_offsets_arr).reshape(shape)
        vars_per_sp += chunk_lengths

        # Close memmap
        del chunk_offsets_arr

    # offsets should be relatively small even for ultra-large datasets
    # scales O(n_samples * ploidy)
    offsets = lengths_to_offsets(vars_per_sp)
    offsets_memmap = np.memmap(
        out_path / "offsets.npy", dtype=offsets.dtype, mode="w+", shape=offsets.shape
    )
    offsets_memmap[:] = offsets
    offsets_memmap.flush()

    var_idxs_memmap = np.memmap(
        out_path / "variant_idxs.npy", dtype=V_IDX_TYPE, mode="w+", shape=offsets[-1]
    )

    # Use in-memory array for write offsets to avoid disk I/O
    write_offsets = offsets[:-1].copy()

    pbar = Progress(*Progress.get_default_columns(), MofNCompleteColumn())
    pbar.start()

    # Pass 2: Copy Genotypes
    for offset, c_dir in pbar.track(
        zip(chunk_offsets, chunk_dirs),
        total=len(chunk_dirs),
        description="Copying genotypes",
    ):
        # We process chunks sequentially to minimize memory usage
        sp_genos = _open_genos(c_dir, (*shape, None), mode="r")

        _copy_chunk_helper(
            var_idxs_memmap,
            write_offsets,
            sp_genos.data,
            sp_genos.offsets,
            offset,
            shape[0],
            shape[1],
        )

        # Close memmaps
        del sp_genos

    var_idxs_memmap.flush()

    if with_dosages:
        # Reset write offsets
        write_offsets = offsets[:-1].copy()

        dosages_memmap = np.memmap(
            out_path / "dosages.npy", dtype=DOSAGE_TYPE, mode="w+", shape=offsets[-1]
        )

        for c_dir in pbar.track(
            chunk_dirs, total=len(chunk_dirs), description="Copying dosages"
        ):
            sp_dosages = _open_fmt(
                "dosages", DOSAGE_TYPE, c_dir, (*shape, None), mode="r"
            )

            _copy_chunk_dosages_helper(
                dosages_memmap,
                write_offsets,
                sp_dosages.data,
                sp_dosages.offsets,
                shape[0],
                shape[1],
            )
            del sp_dosages

        dosages_memmap.flush()

    pbar.stop()


def _write_from_reader(
    *,
    out: Path,
    contigs: list[str],
    caller_samples: list[str],
    out_ploidy: int,
    with_dosages: bool,
    metadata_json: str,
    working_df: pl.DataFrame,
    kept_rows: NDArray[V_IDX_TYPE],
    alt_is_utf8: bool,
    ilen_added: bool,
    subsetting_samples: bool,
    max_mem: int | str,
    n_jobs: int,
    make_tasks: Callable[[Path, int], list[Any]],
    pre_run_check: Callable[[], None] | None = None,
) -> None:
    """Shared writer spine for :meth:`SparseVar.from_vcf`/:meth:`SparseVar.from_pgen`.

    Everything that differs between the two readers (dosage-field validation,
    reader index init, sample resolution, working-index construction,
    region/kept-row resolution, and per-contig keep-index bucketing) is done by
    the caller *before* invoking this function. This function only holds the
    parts that are byte-for-byte identical between the two writers: the atomic
    staging directory, the metadata write, the optional up-front index write
    (when not subsetting samples), job-size resolution, the parallel per-contig
    run, chunk concatenation, and the sample-subsetting MAC-drop finalize.

    Parameters
    ----------
    make_tasks
        ``(chunk_dir, job_mem) -> list[joblib.delayed task]``. Builds the
        per-contig joblib tasks (wrapping ``_process_contig_vcf`` /
        ``_process_contig_pgen``). For PGEN this callback is also responsible
        for freeing/closing the source reader right before returning, so that
        happens after task construction but before ``joblib.Parallel`` runs —
        matching ``from_pgen``'s original memory-freeing order exactly.
    pre_run_check
        Optional callback invoked inside the ``atomic_write_dir`` staging block
        immediately after the metadata write. This is where ``from_pgen``'s
        ``with_dosages and pgen._sei is None`` check lived; its placement
        (inside staging, right after the metadata write) is preserved exactly.
    """
    with atomic_write_dir(out) as staging:
        with open(staging / "metadata.json", "w") as f:
            f.write(metadata_json)

        if pre_run_check is not None:
            pre_run_check()

        # When NOT subsetting samples, write the (region-restricted) index up front.
        if not subsetting_samples:
            _write_index_from_working(
                working_df,
                kept_rows,
                staging / "index.arrow",
                alt_is_utf8,
                ilen_added,
            )

        max_mem_parsed = parse_memory(max_mem)
        n_out = len(caller_samples)
        effective_n_jobs = joblib.cpu_count() if n_jobs == -1 else n_jobs
        effective_n_jobs = min(effective_n_jobs, len(contigs))
        job_mem = max_mem_parsed // effective_n_jobs

        shape = (n_out, out_ploidy)
        with TemporaryDirectory(dir=out.parent) as chunk_dir:
            chunk_dir = Path(chunk_dir)

            tasks = make_tasks(chunk_dir, job_mem)

            with (
                joblib_progress(
                    description=f"Processing contigs using {effective_n_jobs} jobs",
                    total=len(tasks),
                ),
                joblib.Parallel(n_jobs=effective_n_jobs) as parallel,
            ):
                results: list[tuple[int, int]] = list(parallel(tasks))  # type: ignore

            logger.info("Concatenating intermediate chunks")
            _concat_data(staging, chunk_dir, shape, results, with_dosages=with_dosages)

            if subsetting_samples:
                survivors, af = _subset_var_idxs_and_recompute_af(
                    staging,
                    n_total=len(kept_rows),
                    n_out=n_out,
                    ploidy=out_ploidy,
                    with_dosages=with_dosages,
                )
                _write_index_from_working(
                    working_df,
                    kept_rows[survivors],
                    staging / "index.arrow",
                    alt_is_utf8,
                    ilen_added,
                    af=af,
                )

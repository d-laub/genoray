from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

import numba as nb
import numpy as np
import polars as pl
import pyranges as pr
from hirola import HashTable
from loguru import logger
from natsort import natsorted
from numpy.typing import ArrayLike, NDArray
from seqpro._ragged import OFFSET_TYPE, Ragged, lengths_to_offsets
from tqdm.auto import tqdm

from ._pgen import PGEN
from ._utils import ContigNormalizer
from ._vcf import VCF

SVAR_R_DTYPE = np.int64
SVAR_V_IDX = np.int32
IDX_TYPE = np.uint32
INT64_MAX = np.iinfo(SVAR_R_DTYPE).max


class SparseGenotypes(Ragged[SVAR_V_IDX]):
    """A Ragged array of variant indices with additional guarantees and methods:
    - dtype is int32
    - Ragged shape of **at least** 2 dimensions that should correspond to (samples, ploidy)
    - `from_dense` to convert dense genotypes to sparse genotypes
    """

    def __attrs_post_init__(self):
        assert self.ndim >= 2, "SparseGenotypes must have at least 2 dimensions"
        assert self.dtype.type == SVAR_V_IDX, "SparseGenotypes must be of type int32"

    @classmethod
    def from_dense(cls, genos: NDArray[np.int8], var_idxs: NDArray[SVAR_V_IDX]):
        """Convert dense genotypes to sparse genotypes.

        Parameters
        ----------
        genos
            Shape = (sample ploidy variants) Genotypes.
        var_idxs
            Shape = (variants) variant indices.
        """
        # (s p v)
        keep = genos == 1
        data = var_idxs[keep.nonzero()[-1]]
        lengths = keep.sum(-1)
        shape = genos.shape[:-1]
        offsets = lengths_to_offsets(lengths)
        return cls.from_offsets(data, shape, offsets)


class SparseVar:
    path: Path
    samples: list[str]
    ploidy: int
    contigs: list[str]
    genos: dict[str, SparseGenotypes]
    granges: pr.PyRanges
    attrs: pl.DataFrame
    _c_norm: ContigNormalizer
    _s2i: HashTable

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_variants(self) -> int:
        return len(self.granges)

    def __init__(self, path: str | Path):
        path = Path(path)
        self.path = path

        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        self.samples = metadata["samples"]
        self.ploidy = metadata["ploidy"]
        samples = np.array(self.samples)
        self._s2i = HashTable(
            len(samples) * 2,  # type: ignore
            dtype=samples.dtype,
        )
        self._s2i.add(samples)

        contigs = natsorted(p.name for p in path.iterdir() if p.is_dir())
        self.contigs = contigs
        self._c_norm = ContigNormalizer(contigs)
        self.genos = {
            c: _open_sparse_memmap(path / c, (self.n_samples, self.ploidy), "r")
            for c in contigs
        }

        self.granges, self.attrs = self._load_index()

    def var_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
    ) -> NDArray[SVAR_V_IDX]:
        """Get variant index ranges for each query range. i.e.
        For each query range, return the minimum and maximum variant that overlaps.
        Note that this means some variants within those ranges may not actually overlap with
        the query range if there is a deletion that spans the start of the query.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.

        Returns
        -------
            Shape: (ranges, 2). The first column is the start index of the variant
            and the second column is the end index of the variant.
        """
        starts = np.atleast_1d(np.asarray(starts, SVAR_R_DTYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros((n_ranges, 2), SVAR_V_IDX)

        ends = np.atleast_1d(np.asarray(ends, SVAR_R_DTYPE))
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full(n_ranges, c),
                    "Start": starts,
                    "End": ends,
                }
            )
            .with_row_index("query")
            .to_pandas(use_pyarrow_extension_array=True)
        )
        join = queries.join(self.granges)

        if len(join) == 0:
            return np.zeros((n_ranges, 2), SVAR_V_IDX)

        return (
            pl.from_pandas(join.df)
            .group_by("query")
            .agg(s=pl.col("index").min(), e=pl.col("index").max())
            .drop("query")
            .to_numpy()
            .astype(SVAR_V_IDX)
        )

    def find_starts_ends(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        samples: ArrayLike | None = None,
    ) -> NDArray[OFFSET_TYPE]:
        if samples is None:
            samples = np.atleast_1d(np.array(self.samples))
        else:
            if missing := set(samples) - set(self.samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")
            samples = np.atleast_1d(np.array(samples))

        s_idxs = cast(NDArray[np.int64], self._s2i[samples])

        starts = np.atleast_1d(np.asarray(starts, SVAR_R_DTYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros((n_ranges + 1, 2), OFFSET_TYPE)

        ends = np.atleast_1d(np.asarray(ends, SVAR_R_DTYPE))
        # (r 2)
        var_ranges = self.var_ranges(contig, starts, ends)
        genos = self.genos[c]
        # (r s p 2)
        starts_ends = _find_starts_ends(
            genos.data, genos.offsets, var_ranges, s_idxs, self.ploidy
        )
        return starts_ends

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        samples: ArrayLike | None = None,
    ) -> SparseGenotypes:
        if samples is None:
            samples = np.atleast_1d(np.array(self.samples))
        else:
            if missing := set(samples) - set(self.samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")
            samples = np.atleast_1d(np.array(samples))

        n_samples = len(samples)
        starts = np.atleast_1d(np.asarray(starts, SVAR_R_DTYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return SparseGenotypes.from_offsets(
                np.empty((0), SVAR_V_IDX),
                (n_samples, self.ploidy),
                np.zeros((0, 2), OFFSET_TYPE),
            )

        # (r s p 2)
        starts_ends = self.find_starts_ends(contig, starts, ends, samples)
        return SparseGenotypes.from_offsets(
            self.genos[c].data,
            (n_ranges, n_samples, self.ploidy),
            starts_ends.reshape(-1, 2),
        )

    @classmethod
    def from_vcf(
        cls, out: str | Path, vcf: VCF, max_mem: int | str, overwrite: bool = False
    ):
        out = Path(out)

        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.mkdir(parents=True, exist_ok=True)

        contigs = vcf.contigs

        tempdir = TemporaryDirectory()
        tdir = Path(tempdir.name)
        shape = (vcf.n_samples, vcf.ploidy)
        with open(out / "metadata.json", "w") as f:
            json.dump({"samples": vcf.available_samples, "ploidy": vcf.ploidy}, f)

        if not vcf._index_path().exists():
            logger.info("Genoray VCF index not found, creating index.")
            vcf._write_gvi_index()

        shutil.copy(vcf._index_path(), cls._index_path(out))

        c_pbar = tqdm(total=len(contigs), unit=" contig")
        for c in contigs:
            c_pbar.set_description(f"Processing contig {c}")
            v_pbar = tqdm(unit=" variant", position=1)
            v_pbar.set_description("Reading variants")
            (tdir / c).mkdir(parents=True, exist_ok=True)
            (out / c).mkdir(parents=True, exist_ok=True)
            offset = 0
            # genos: (s p v)
            with vcf.using_pbar(v_pbar) as vcf:
                for i, genos in enumerate(
                    vcf.chunk(c, max_mem=max_mem, mode=VCF.Genos8)
                ):
                    n_vars = genos.shape[-1]
                    var_idxs = np.arange(offset, offset + n_vars, dtype=np.int32)
                    sp_genos = SparseGenotypes.from_dense(genos, var_idxs)
                    _write_sparse_memmap(tdir / c / str(i), sp_genos)
                    offset += n_vars

                v_pbar.set_description("Concatenating intermediate chunks")
                _concat_sparse_memmaps(out / c, tdir / c, shape)
                v_pbar.close()
            c_pbar.update()
        c_pbar.close()

        tempdir.cleanup()

    @classmethod
    def from_pgen(
        cls, out: str | Path, pgen: PGEN, max_mem: int | str, overwrite: bool = False
    ):
        out = Path(out)

        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.mkdir(parents=True, exist_ok=True)

        contigs = pgen.contigs

        tempdir = TemporaryDirectory()
        tdir = Path(tempdir.name)
        shape = (pgen.n_samples, pgen.ploidy)
        with open(out / "metadata.json", "w") as f:
            json.dump({"samples": pgen.available_samples, "ploidy": pgen.ploidy}, f)

        shutil.copy(pgen._index_path(), cls._index_path(out))

        n_variants = len(pgen._index)
        pbar = tqdm(total=n_variants, unit=" variant")
        for c in contigs:
            (tdir / c).mkdir(parents=True, exist_ok=True)
            (out / c).mkdir(parents=True, exist_ok=True)
            offset = 0
            i = 0
            pbar.set_description(f"Contig {c}, readings variants")
            for range_ in pgen.chunk_ranges(c, max_mem=max_mem, mode=PGEN.Genos):
                if range_ is None:
                    continue
                # genos: (s p v)
                for genos in range_:
                    n_vars = genos.shape[-1]
                    var_idxs = np.arange(offset, offset + n_vars, dtype=np.int32)
                    sp_genos = SparseGenotypes.from_dense(
                        genos.astype(np.int8), var_idxs
                    )
                    _write_sparse_memmap(tdir / c / str(i), sp_genos)
                    offset += n_vars
                    i += 1
                    pbar.update(n_vars)
            pbar.set_description(f"Contig {c}, concatenating intermediate chunks")
            _concat_sparse_memmaps(out / c, tdir / c, shape)
        pbar.close()

        tempdir.cleanup()

    @classmethod
    def _index_path(cls, root: Path):
        """Path to the index file."""
        return root / "index.arrow"

    def _load_index(self) -> tuple[pr.PyRanges, pl.DataFrame]:
        """Load the index file and return the granges and attributes."""
        index = pl.scan_ipc(
            self._index_path(self.path), row_index_name="index", memory_map=False
        )

        granges = pr.PyRanges(
            index.select(
                "index",
                Chromosome="CHROM",
                Start=pl.col("POS") - 1,
                End=pl.col("POS") + pl.col("REF").str.len_bytes() - 1,
            )
            .collect()
            .to_pandas()
        )
        attrs = index.select(pl.exclude("CHROM", "POS", "index")).collect()
        return granges, attrs


def _open_sparse_memmap(path: Path, shape: tuple[int, ...], mode: Literal["r", "r+"]):
    # Load the memory-mapped files
    var_idxs = np.memmap(path / "variant_idxs.npy", dtype=np.int32, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = SparseGenotypes.from_offsets(var_idxs, shape, offsets)
    return sp_genos


def _write_sparse_memmap(path: Path, sp_genos: SparseGenotypes):
    path.mkdir(parents=True, exist_ok=True)

    var_idxs = np.memmap(
        path / "variant_idxs.npy",
        shape=sp_genos.data.shape,
        dtype=sp_genos.data.dtype,
        mode="w+",
    )
    var_idxs[:] = sp_genos.data
    var_idxs.flush()
    offsets = np.memmap(
        path / "offsets.npy",
        shape=sp_genos.offsets.shape,
        dtype=sp_genos.offsets.dtype,
        mode="w+",
    )
    offsets[:] = sp_genos.offsets
    offsets.flush()


def _concat_sparse_memmaps(out_path: Path, chunks_path: Path, shape: tuple[int, int]):
    """concat one contig"""
    out_path.mkdir(parents=True, exist_ok=True)

    # [1, 2, 3, ...]
    chunk_dirs = natsorted(chunks_path.iterdir())

    vars_per_sp = np.zeros(shape, dtype=np.int32)

    ls_sp_genos: list[SparseGenotypes] = []
    for chunk_dir in chunk_dirs:
        sp_genos = _open_sparse_memmap(chunk_dir, shape, mode="r")
        vars_per_sp += sp_genos.lengths
        ls_sp_genos.append(sp_genos)

    # this should be relatively small even for ultra-large datasets
    offsets = lengths_to_offsets(vars_per_sp)
    offsets_memmap = np.memmap(
        out_path / "offsets.npy", dtype=offsets.dtype, mode="w+", shape=offsets.shape
    )
    offsets_memmap[:] = offsets
    offsets_memmap.flush()

    var_idxs_memmap = np.memmap(
        out_path / "variant_idxs.npy", dtype=np.int32, mode="w+", shape=offsets[-1]
    )
    _concat_helper(
        var_idxs_memmap,
        offsets,
        [a.data for a in ls_sp_genos],
        [a.offsets for a in ls_sp_genos],
        shape,
    )
    var_idxs_memmap.flush()


@nb.njit(parallel=True, nogil=True, cache=True)
def _concat_helper(
    out_idxs: NDArray[np.int32],
    out_offsets: NDArray[OFFSET_TYPE],
    in_var_idxs: list[NDArray[np.int32]],
    in_offsets: list[NDArray[OFFSET_TYPE]],
    shape: tuple[int, int],
):
    n_samples, ploidy = shape
    n_chunks = len(in_var_idxs)
    assert len(in_offsets) == n_chunks
    for s in nb.prange(n_samples):
        for p in nb.prange(ploidy):
            sp = s * ploidy + p
            o_s, o_e = out_offsets[sp], out_offsets[sp + 1]
            sp_out_idxs = out_idxs[o_s:o_e]
            offset = 0
            for chunk in range(n_chunks):
                i_s, i_e = in_offsets[chunk][sp], in_offsets[chunk][sp + 1]
                chunk_len = i_e - i_s
                sp_out_idxs[offset : offset + chunk_len] = in_var_idxs[chunk][i_s:i_e]
                offset += chunk_len


@nb.njit(parallel=True, nogil=True, cache=True)
def _find_starts_ends(
    genos: NDArray[SVAR_V_IDX],
    geno_offsets: NDArray[OFFSET_TYPE],
    var_ranges: NDArray[SVAR_V_IDX],
    sample_idxs: NDArray[np.int64],
    ploidy: int,
):
    """Find the start and end offsets of the sparse genotypes for each range.

    Parameters
    ----------
    genos
        Sparse genotypes
    geno_offsets
        Genotype offsets
    var_ranges
        Shape = (ranges 2) Variant index ranges.

    Returns
    -------
        Shape: (ranges samples ploidy 2). The first column is the start index of the variant
        and the second column is the end index of the variant.
    """
    n_ranges = len(var_ranges)
    n_samples = len(sample_idxs)
    out_ranges = np.zeros((n_ranges, n_samples, ploidy, 2), dtype=OFFSET_TYPE)

    for r in nb.prange(n_ranges):
        for s in nb.prange(n_samples):
            for p in nb.prange(ploidy):
                sp = s * ploidy + p
                o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
                sp_genos = genos[o_s:o_e]
                out_ranges[r, s, p, 0] = np.searchsorted(sp_genos, var_ranges[r, 0])
                out_ranges[r, s, p, 1] = np.searchsorted(
                    sp_genos, var_ranges[r, 1], side="right"
                )

    return out_ranges

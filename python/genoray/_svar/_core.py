from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable, Sequence
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, overload

import awkward as ak
import joblib
import numpy as np
import polars as pl
from awkward.contents import Content, NumpyArray
from hirola import HashTable
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from polars._typing import IntoExpr
from pydantic import BaseModel
from rich.progress import MofNCompleteColumn, Progress
from seqpro.rag import OFFSET_TYPE, Ragged, lengths_to_offsets

from .._mutcat import MUTCAT_VERSION
from .._pgen import PGEN
from .._reference import Reference
from .._types import DOSAGE_TYPE, POS_MAX, POS_TYPE, V_IDX_TYPE
from .._utils import ContigNormalizer, atomic_write_dir
from .._var_ranges import var_ranges
from .._vcf import VCF
from ..exprs import ILEN
from ._annotate import SparseVarAnnotateMixin
from ._convert import _process_contig_pgen, _process_contig_vcf, _write_from_reader
from ._io import (
    _build_working_index,
    _open_fmt,
    _open_genos,
)
from ._kernels import (
    _find_starts_ends,
    _find_starts_ends_with_length,
    _nb_af_helper,
    _nb_count_kept,
    _nb_count_mac_per_kept,
    _nb_write_field,
    _nb_write_var_idxs,
)
from ._regions import (
    _normalize_regions,
    _normalize_samples,
    _resolve_kept_rows,
    _resolve_kept_var_idxs,
    _resolve_sample_idxs,
    _validate_fields,
)

CURRENT_VERSION = 1


class SparseVarMetadata(BaseModel):
    version: int | None = None
    samples: list[str]
    ploidy: int
    contigs: list[str]
    fields: dict[str, str] = {}  # field_name -> numpy dtype name (e.g. "float32")
    mutcat_version: int | None = None  # set when annotate_mutations has run
    mutcat_contigs: list[str] | None = None  # normalized contigs annotated; None = all


_SRT = TypeVar("_SRT")


class SparseVar(SparseVarAnnotateMixin, Generic[_SRT]):
    """Open a Sparse Variant (SVAR) directory.

    Parameters
    ----------
    path
        Path to the SVAR directory.
    attrs
        Expression of attributes to load in addition to the ALT and ILEN columns.
    fields
        Names of fields to load from the SVAR directory. Must be keys of
        :attr:`available_fields`. Only VCF FORMAT fields with ``Number=G`` are currently
        supported as custom fields.
    """

    path: Path
    version: int | None
    available_samples: list[str]
    ploidy: int
    contigs: list[str]
    """Contigs in the order they appear in the dataset. Variants are only sorted within each contig."""
    genos: Ragged[V_IDX_TYPE]
    available_fields: dict[str, np.dtype[np.number]]
    fields: dict[str, Ragged[np.number]]
    _c_norm: ContigNormalizer
    _s2i: HashTable

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.available_samples)

    @property
    def n_variants(self) -> int:
        """Number of variants in the dataset."""
        return int(self._contig_stats["n"].sum())

    @property
    def nbytes(self) -> int:
        """Total in-memory footprint, in bytes, of resident (non-mmap'd) data
        held by this reader. Only the polars variant index counts; `genos`
        and `fields` are memory-mapped and excluded.
        """
        return int(self.index.estimated_size())

    @cached_property
    # pyrefly: ignore [bad-override, missing-override-decorator]
    def index(self) -> pl.DataFrame:
        """The full variant index, materialized on first access.

        Table of variants with columns ``CHROM``, ``POS``, ``REF``, ``ALT``, ``ILEN``,
        and any additional attributes specified in ``attrs`` on construction.
        """
        return self._index_lazy.collect()

    @cached_property
    def _contig_stats(self) -> pl.DataFrame:
        """Per-contig variant count and max POS, in first-appearance order.

        Computed by a single streaming reduction over the numeric index
        columns — never materializes the full per-row index.
        """
        return (
            self._index_lazy.group_by("CHROM", maintain_order=True)
            .agg(n=pl.len(), pos_max=pl.col("POS").max())
            .collect()
        )

    @cached_property
    def _c_max_idxs(self) -> dict[str, int]:
        stats = self._contig_stats
        out = {c: int(v) - 1 for c, v in zip(stats["CHROM"], stats["n"].cum_sum())}
        out |= {c: 0 for c in self.contigs if c not in out}
        return out

    @cached_property
    def _is_biallelic(self) -> bool:
        return bool(
            self._index_lazy.select((pl.col("ALT").list.len() == 1).all())
            .collect()
            .item()
        )

    @overload
    def __init__(
        self: SparseVar[Ragged[V_IDX_TYPE]],
        path: str | Path,
        attrs: IntoExpr | None = None,
        fields: None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: SparseVar[Ragged[np.void]],
        path: str | Path,
        attrs: IntoExpr | None = None,
        fields: Sequence[str] = ...,
    ) -> None: ...
    def __init__(
        self,
        path: str | Path,
        attrs: IntoExpr | None = None,
        fields: Sequence[str] | None = None,
    ):
        path = Path(path)
        self.path = path

        if not self.path.exists():
            raise FileNotFoundError(f"SVAR directory {self.path} does not exist.")

        with open(path / "metadata.json", "rb") as f:
            metadata = SparseVarMetadata.model_validate_json(f.read())
        contigs = metadata.contigs
        self.version = metadata.version
        self.contigs = contigs
        self.available_samples = metadata.samples
        self.ploidy = metadata.ploidy
        self.available_fields = {
            name: np.dtype(dtype_str) for name, dtype_str in metadata.fields.items()
        }

        if fields is not None and (missing := set(fields) - set(self.available_fields)):
            raise ValueError(f"Fields {missing} not found in the dataset.")

        samples = np.array(self.available_samples)
        self._s2i = HashTable(
            len(samples) * 2,  # type: ignore
            dtype=samples.dtype,
        )
        self._s2i.add(samples)

        self._c_norm = ContigNormalizer(contigs)
        shape = (self.n_samples, self.ploidy, None)
        self.genos = _open_genos(path, shape, "r")
        self.fields = {
            name: _open_fmt(name, self.available_fields[name], path, shape, "r")
            for name in (fields or [])
        }
        if (
            "mutcat" in (fields or [])
            # None: svar predates mutcat versioning; no warning (treat as pre-v1).
            and metadata.mutcat_version is not None
            and metadata.mutcat_version < MUTCAT_VERSION
        ):
            logger.warning(
                "mutcat field was computed with an older version "
                f"(v{metadata.mutcat_version} < v{MUTCAT_VERSION}); "
                "recompute via annotate_mutations()."
            )
        self._index_lazy = self._scan_index(attrs)

    def var_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
    ) -> NDArray[V_IDX_TYPE]:
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
            Shape: :code:`(ranges, 2)`. The first column is the start index of the variant
            and the second column is the end index of the variant.
        """
        return var_ranges(self._c_norm, self.index, contig, starts, ends)

    def _covers_all_variants(
        self, regions: "pl.DataFrame", mode: "Literal['pos', 'record', 'variant']"
    ) -> bool:
        """True iff *regions* select every variant (one region per present contig,
        each spanning [0, pos_max]).  Lets write_view skip POS materialization.

        Only applies to ``pos``/``record`` modes (``variant`` mode is ILEN-aware
        and resolved through ``var_ranges``).
        """
        if mode == "variant":
            return False

        stats = self._contig_stats  # CHROM, n, pos_max
        present = set(stats["CHROM"])

        per_contig = regions.group_by("chrom").agg(
            start=pl.col("start").min(),
            end=pl.col("end").max(),
            k=pl.len(),
        )
        if set(per_contig["chrom"]) != present:
            return False

        pos_max = dict(zip(stats["CHROM"], stats["pos_max"]))
        end_offset = 0 if mode == "pos" else 1
        for c, s, e, k in zip(
            per_contig["chrom"], per_contig["start"], per_contig["end"], per_contig["k"]
        ):
            # POS is 1-based; 0-based p in [0, pos_max-1]. Full coverage needs a
            # single region with start <= 0 and (end + end_offset) >= pos_max.
            if k != 1 or s > 0 or (e + end_offset) < pos_max[c]:
                return False
        return True

    def _find_starts_ends(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
        samples: ArrayLike | None = None,
        out: NDArray[OFFSET_TYPE] | None = None,
    ) -> NDArray[OFFSET_TYPE]:
        """Find the start and end offsets of the sparse genotypes for each range.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.
        samples
            List of sample names to read. If None, read all samples.
        out
            Output array to write to. If None, a new array will be created.

        Returns
        -------
            Shape: (2, ranges, samples, ploidy). The first column is the start index of the variant
            and the second column is the end index of the variant.
        """
        samples, s_idxs = _resolve_sample_idxs(
            samples, self.available_samples, self._s2i
        )

        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            if out is None:
                return np.full(
                    (n_ranges, len(samples), self.ploidy, 2), -1, OFFSET_TYPE
                )
            else:
                out[:] = -1
                return out

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        # (r 2)
        var_ranges = self.var_ranges(contig, starts, ends)
        if out is None:
            # (2 r s p)
            out = np.empty((2, n_ranges, len(samples), self.ploidy), dtype=OFFSET_TYPE)
        _find_starts_ends(
            self.genos.data,
            self.genos.offsets,
            var_ranges,
            s_idxs,
            self.ploidy,
            out_offsets=out,
        )
        return out

    def _find_starts_ends_with_length(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
        samples: ArrayLike | None = None,
        out: NDArray[OFFSET_TYPE] | None = None,
    ) -> NDArray[OFFSET_TYPE]:
        """Find the start and end offsets of the sparse genotypes for each range.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.
        samples
            List of sample names to read. If None, read all samples.
        out
            Output array to write to. If None, a new array will be created.

        Returns
        -------
            Shape: (2, ranges, samples, ploidy). The first column is the start index of the variant
            and the second column is the end index of the variant.
        """
        if not self._is_biallelic:
            raise ValueError(
                "Cannot use with_length operations with multiallelic variants."
            )

        samples, s_idxs = _resolve_sample_idxs(
            samples, self.available_samples, self._s2i
        )

        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.full((n_ranges, len(samples), self.ploidy, 2), -1, OFFSET_TYPE)

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        # (r 2)
        var_ranges = self.var_ranges(contig, starts, ends)

        v_starts = (self.index["POS"] - 1).to_numpy()

        # (2 r s p)
        out = _find_starts_ends_with_length(
            self.genos.data,
            self.genos.offsets,
            starts,
            ends,
            var_ranges,
            v_starts,
            self.index["ILEN"].list.first().fill_null(0).to_numpy(),
            s_idxs,
            self.ploidy,
            self._c_max_idxs[c],
            out,
        )
        return out

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
        samples: ArrayLike | None = None,
    ) -> _SRT:
        """Read the genotypes for the given ranges.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.
        samples
            List of sample names to read. If None, read all samples.

        Returns
        -------
            When no fields are loaded: ``Ragged[V_IDX_TYPE]`` with shape
            ``(ranges, samples, ploidy, ~variants)``. When fields are loaded: an awkward
            record array of the same outer shape where ``result.genos`` is
            ``Ragged[V_IDX_TYPE]`` and each additional field (e.g. ``result.dosages``) is
            a ``Ragged`` of its respective dtype. All arrays are backed by memory-mapped
            data so only the offsets reside in RAM.
        """
        samples, _ = _resolve_sample_idxs(samples, self.available_samples, self._s2i)

        n_samples = len(samples)
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        # (2 r s p)
        starts_ends = self._find_starts_ends(contig, starts, ends, samples)
        shape = (n_ranges, n_samples, self.ploidy, None)
        flat_offsets = starts_ends.reshape(2, -1)

        genos_result = Ragged[V_IDX_TYPE].from_offsets(
            self.genos.data, shape, flat_offsets
        )

        if not self.fields:
            return genos_result  # type: ignore[return-value]

        field_results = {
            name: Ragged.from_offsets(field.data, shape, flat_offsets)
            for name, field in self.fields.items()
        }
        return Ragged.from_fields({"genos": genos_result, **field_results})  # type: ignore[return-value]

    def read_ranges_with_length(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
        samples: ArrayLike | None = None,
    ) -> _SRT:
        """Read the genotypes for the given ranges such that each entry of variants is guaranteed to have
        the minimum amount of variants to reach the query length. This can mean either fewer or more variants
        than would be returned than by :code:`read_ranges`, depending on the presence of indels.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.
        samples
            List of sample names to read. If None, read all samples.

        Returns
        -------
            Same return structure as :meth:`read_ranges`.
        """
        samples, _ = _resolve_sample_idxs(samples, self.available_samples, self._s2i)

        n_samples = len(samples)
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        # (2 r s p)
        starts_ends = self._find_starts_ends_with_length(contig, starts, ends, samples)
        shape = (n_ranges, n_samples, self.ploidy, None)
        flat_offsets = starts_ends.reshape(2, -1)

        genos_result = Ragged[V_IDX_TYPE].from_offsets(
            self.genos.data, shape, flat_offsets
        )

        if not self.fields:
            return genos_result  # type: ignore[return-value]

        field_results = {
            name: Ragged.from_offsets(field.data, shape, flat_offsets)
            for name, field in self.fields.items()
        }
        return Ragged.from_fields({"genos": genos_result, **field_results})  # type: ignore[return-value]

    @overload
    def with_fields(self, fields: Sequence[str]) -> SparseVar[Ragged[np.void]]: ...
    @overload
    def with_fields(self, fields: Literal[False]) -> SparseVar[Ragged[V_IDX_TYPE]]: ...
    @overload
    def with_fields(self, fields: None = None) -> SparseVar[_SRT]: ...
    def with_fields(
        self,
        fields: Sequence[str] | Literal[False] | None = None,
    ) -> SparseVar:
        """Return a shallow copy of this ``SparseVar`` with updated fields.

        Parameters
        ----------
        fields
            - ``None``: leave fields unchanged (returns shallow copy).
            - ``Sequence[str]``: names of fields to load from the SVAR directory.
              Must be keys of :attr:`available_fields`.
            - ``False``: drop all fields, returning a ``SparseVar[Ragged[V_IDX_TYPE]]``.
        """
        new = copy.copy(self)

        if fields is None:
            return new

        if fields is False:
            new.fields = {}
            return new

        if missing := set(fields) - set(self.available_fields):
            raise ValueError(f"Fields {missing} not found in the dataset.")
        shape = (self.n_samples, self.ploidy, None)
        new.fields = {
            name: _open_fmt(name, self.available_fields[name], self.path, shape, "r")
            for name in fields
        }
        return new

    @classmethod
    def from_vcf(
        cls,
        out: str | Path,
        vcf: VCF,
        max_mem: int | str,
        overwrite: bool = False,
        with_dosages: bool = False,
        n_jobs: int = -1,
        *,
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        samples: "str | Sequence[str] | PathLike | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: Literal["pos", "record", "variant"] = "pos",
        haploid: bool = False,
    ):
        """Create a Sparse Variant (.svar) from a VCF/BCF.

        Parameters
        ----------
        out
            Path to the output directory.
        vcf
            VCF file to write from.
        max_mem
            Maximum memory to use while writing.
        overwrite
            Whether to overwrite the output directory if it exists.
        with_dosages
            Whether to write dosages.
        n_jobs
            Number of jobs to use for parallel processing.
        regions
            Region(s) to include. Accepts the same input types as ``write_view``:
            a ``"chrom:start-end"`` string (1-based, end-inclusive), a
            ``(chrom, start, end)`` tuple (0-based, end-exclusive), a BED file
            path, or a frame-like. ``None`` (default) includes all regions.
        samples
            Sample name(s) to include (a name, a sequence of names, or a path to a
            newline-delimited file). Caller order is preserved, deduped by first
            occurrence. ``None`` (default) includes all samples. Variants whose
            minor allele count is 0 across the chosen samples are dropped from the
            output; if every variant drops, a ``ValueError`` is raised.
        merge_overlapping
            If ``False`` (default) raise on overlapping input regions; if ``True``
            dedupe via pyranges merge.
        regions_overlap
            ``"pos"`` (default), ``"record"``, or ``"variant"`` — same semantics
            as ``write_view``.
        haploid
            If ``True``, OR-collapse the ploidy axis into a single haploid call
            per sample (a variant present on any haplotype becomes one call) and
            record ``ploidy=1`` in the output metadata. Intended for unphased
            somatic data. Default ``False``.
        """
        out = Path(out)

        if with_dosages and vcf.dosage_field is None:
            raise ValueError("VCF does not have a dosage field specified.")

        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.parent.mkdir(parents=True, exist_ok=True)

        if not vcf._index_path().exists():
            logger.info("Genoray VCF index not found, creating index.")
            vcf._write_gvi_index()

        # --- resolve sample subset (None => all) ---
        if samples is None:
            caller_samples = list(vcf.available_samples)
        else:
            caller_samples = _normalize_samples(samples, vcf.available_samples)
            if not caller_samples:
                raise ValueError("from_vcf: `samples` selected no samples")

        # --- build working index (filtered) and resolve kept rows ---
        working_df, alt_is_utf8, ilen_added = _build_working_index(
            vcf._index_path(), vcf._filter.expr if vcf._filter is not None else None
        )
        if regions is None:
            kept_rows = working_df["index"].to_numpy().astype(V_IDX_TYPE)
        else:
            regions_df = _normalize_regions(regions, vcf._c_norm)
            kept_rows = _resolve_kept_rows(
                working_df, vcf._c_norm, regions_df, regions_overlap, merge_overlapping
            )
            if len(kept_rows) == 0:
                raise ValueError("no variants selected by `regions`")
            kept_rows = np.sort(kept_rows)

        # rows kept on each contig, as positions LOCAL to that contig's filtered
        # block (workers number variants per contig starting at 0).
        contigs = vcf.contigs
        # maintain_order=True relies on working_df being in contig-contiguous file
        # order — each contig must form a single contiguous block with no interleaving.
        counts = working_df.group_by("CHROM", maintain_order=True).agg(
            pl.len().alias("n")
        )
        # Verify each contig forms a single contiguous block (no interleaving):
        # the number of CHROM runs must equal the number of distinct contigs.
        # block_start offsets are only valid under this invariant.
        chrom_col = working_df["CHROM"].to_numpy()
        n_runs = (
            1 + int((chrom_col[1:] != chrom_col[:-1]).sum()) if len(chrom_col) else 0
        )
        assert n_runs == working_df["CHROM"].n_unique(), (
            "contig blocks are not contiguous — a contig appears in multiple disjoint spans"
        )
        block_start: dict[str, int] = {}
        block_n: dict[str, int] = {}
        running = 0
        for c, n in zip(counts["CHROM"].to_list(), counts["n"].to_list()):
            block_start[c] = running
            block_n[c] = int(n)
            running += n
        keep_local_by_contig: dict[str, np.ndarray] = {}
        for c in contigs:
            if c not in block_start:
                continue
            start = block_start[c]
            n = block_n[c]
            in_block = kept_rows[(kept_rows >= start) & (kept_rows < start + n)]
            keep_local_by_contig[c] = (in_block - start).astype(np.int64)

        out_ploidy = 1 if haploid else vcf.ploidy

        metadata_json = SparseVarMetadata(
            version=CURRENT_VERSION,
            contigs=contigs,
            samples=caller_samples,
            ploidy=out_ploidy,
            fields={"dosages": np.dtype(DOSAGE_TYPE).name} if with_dosages else {},
        ).model_dump_json()

        def make_tasks(chunk_dir: Path, job_mem: int) -> list[Any]:
            tasks: list[Any] = []
            for chunk_idx, c in enumerate(contigs):
                task = joblib.delayed(_process_contig_vcf)(
                    vcf.path,
                    dosage_field=vcf.dosage_field if with_dosages else None,
                    max_mem=job_mem,
                    contig=c,
                    chunk_dir=chunk_dir,
                    chunk_idx=chunk_idx,
                    cyvcf2_filter=vcf._filter.record
                    if vcf._filter is not None
                    else None,
                    pl_filter=vcf._filter.expr if vcf._filter is not None else None,
                    caller_samples=None if samples is None else caller_samples,
                    keep_local=keep_local_by_contig.get(c),
                    haploid=haploid,
                )
                tasks.append(task)
            return tasks

        return _write_from_reader(
            out=out,
            contigs=contigs,
            caller_samples=caller_samples,
            out_ploidy=out_ploidy,
            with_dosages=with_dosages,
            metadata_json=metadata_json,
            working_df=working_df,
            kept_rows=kept_rows,
            alt_is_utf8=alt_is_utf8,
            ilen_added=ilen_added,
            subsetting_samples=samples is not None,
            max_mem=max_mem,
            n_jobs=n_jobs,
            make_tasks=make_tasks,
        )

    @classmethod
    def from_pgen(
        cls,
        out: str | Path,
        pgen: PGEN,
        max_mem: int | str,
        overwrite: bool = False,
        with_dosages: bool = False,
        n_jobs: int = -1,
        *,
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        samples: "str | Sequence[str] | PathLike | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: Literal["pos", "record", "variant"] = "pos",
        haploid: bool = False,
    ):
        """Create a Sparse Variant (.svar) from a PGEN.

        Parameters
        ----------
        out
            Path to the output directory.
        pgen
            PGEN file to write from.
        max_mem
            Maximum memory to use while writing.
        overwrite
            Whether to overwrite the output directory if it exists.
        with_dosages
            Whether to write dosages.
        n_jobs
            Number of jobs to use for parallel processing.
        regions
            Region(s) to include. Accepts the same input types as ``write_view``:
            a ``"chrom:start-end"`` string (1-based, end-inclusive), a
            ``(chrom, start, end)`` tuple (0-based, end-exclusive), a BED file
            path, or a frame-like. ``None`` (default) includes all regions.
        samples
            Sample name(s) to include (a name, a sequence of names, or a path to a
            newline-delimited file). Caller order is preserved, deduped by first
            occurrence. ``None`` (default) includes all samples. Variants whose
            minor allele count is 0 across the chosen samples are dropped from the
            output; if every variant drops, a ``ValueError`` is raised.
        merge_overlapping
            If ``False`` (default) raise on overlapping input regions; if ``True``
            dedupe via pyranges merge.
        regions_overlap
            ``"pos"`` (default), ``"record"``, or ``"variant"`` — same semantics
            as ``write_view``.
        haploid
            If ``True``, OR-collapse the ploidy axis into a single haploid call
            per sample (a variant present on any haplotype becomes one call) and
            record ``ploidy=1`` in the output metadata. Intended for unphased
            somatic data. Default ``False``.
        """
        out = Path(out)

        if with_dosages and pgen.dosage_path is None:
            raise ValueError("PGEN does not have a dosage file specified.")

        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.parent.mkdir(parents=True, exist_ok=True)

        pgen._init_index()
        assert pgen.contigs is not None
        assert pgen._c_max_idxs is not None
        assert pgen._c_norm is not None

        # --- resolve sample subset ---
        if samples is None:
            caller_samples = list(pgen.available_samples)
            sample_subset: np.ndarray | None = None
        else:
            caller_samples = _normalize_samples(samples, pgen.available_samples)
            if not caller_samples:
                raise ValueError("from_pgen: `samples` selected no samples")
            sample_subset = pgen._s2i.get(np.asarray(caller_samples)).astype(np.uint32)

        # --- working index (filtered) for region resolution + output index ---
        working_df, alt_is_utf8, ilen_added = _build_working_index(
            pgen._index_path(), pgen._filter
        )
        assert pgen._index is not None
        phys = pgen._index["index"].to_numpy().astype(np.uint32)
        assert len(phys) == working_df.height, (
            f"filtered index / pgen._index misaligned: "
            f"pgen._index has {len(phys)} rows but working_df has {working_df.height}"
        )
        # Confirm ordering: pgen._index and working_df must be row-aligned.
        # Length equality alone does NOT prove alignment; this POS comparison is the
        # genuine guard that _load_index/_build_working_index produce the same ordering.
        # The assert is stripped under python -O so runtime cost is negligible.
        assert (pgen._index["POS"].to_numpy() == working_df["POS"].to_numpy()).all(), (
            "pgen._index and working_df POS columns are not aligned — ordering mismatch"
        )

        contigs = pgen.contigs

        if regions is None:
            kept_rows = working_df["index"].to_numpy().astype(V_IDX_TYPE)
        else:
            regions_df = _normalize_regions(regions, pgen._c_norm)
            kept_rows = _resolve_kept_rows(
                working_df, pgen._c_norm, regions_df, regions_overlap, merge_overlapping
            )
            if len(kept_rows) == 0:
                raise ValueError("no variants selected by `regions`")
            kept_rows = np.sort(kept_rows)

        # physical keep ids per contig, in kept-row (= output var id) order.
        # Unlike from_vcf (which passes contig-LOCAL offsets for sequential VCF
        # streaming), here we pass PHYSICAL variant indices so _process_contig_pgen
        # can call pgenlib's read_alleles_list for random access.
        kept_chrom = working_df["CHROM"].to_numpy()[kept_rows]
        kept_phys = phys[kept_rows]
        keep_by_contig: dict[str, np.ndarray] = {}
        for c in contigs:
            m = kept_chrom == c
            if m.any():
                keep_by_contig[c] = np.ascontiguousarray(kept_phys[m], dtype=np.uint32)

        out_ploidy = 1 if haploid else pgen.ploidy

        metadata_json = SparseVarMetadata(
            version=CURRENT_VERSION,
            contigs=contigs,
            samples=caller_samples,
            ploidy=out_ploidy,
            fields={"dosages": np.dtype(DOSAGE_TYPE).name} if with_dosages else {},
        ).model_dump_json()

        def pre_run_check() -> None:
            # Placement preserved: fires inside the staging block, right after the
            # metadata write and before any index/chunk work.
            if with_dosages and pgen._sei is None:
                raise ValueError("PGEN must be bi-allelic with filters applied")

        def make_tasks(chunk_dir: Path, job_mem: int) -> list[Any]:
            mem_per_var = pgen._mem_per_variant(
                pgen.GenosDosages if with_dosages else pgen.Genos  # type: ignore
            )
            tasks: list[Any] = []
            for c in contigs:
                keep_idxs = keep_by_contig.get(c)
                if keep_idxs is None or len(keep_idxs) == 0:
                    continue

                task = joblib.delayed(_process_contig_pgen)(
                    geno_path=pgen.geno_path,
                    dosage_path=pgen.dosage_path if with_dosages else None,
                    max_mem=job_mem,
                    keep_idxs=keep_idxs,
                    mem_per_var=mem_per_var,
                    n_samples=pgen.n_samples,
                    ploidy=pgen.ploidy,
                    chunk_dir=chunk_dir,
                    chunk_idx=len(tasks),
                    sample_subset=sample_subset,
                    haploid=haploid,
                )
                tasks.append(task)

            # Free parent-process memory after task construction but before the
            # parallel run (workers re-open the readers themselves).
            pgen._free_index()
            # PgenReaders can be multi-GB allocations, close them to free memory
            pgen._geno_pgen.close()
            if pgen.dosage_path is not None:
                pgen._dose_pgen.close()

            return tasks

        return _write_from_reader(
            out=out,
            contigs=contigs,
            caller_samples=caller_samples,
            out_ploidy=out_ploidy,
            with_dosages=with_dosages,
            metadata_json=metadata_json,
            working_df=working_df,
            kept_rows=kept_rows,
            alt_is_utf8=alt_is_utf8,
            ilen_added=ilen_added,
            subsetting_samples=samples is not None,
            max_mem=max_mem,
            n_jobs=n_jobs,
            make_tasks=make_tasks,
            pre_run_check=pre_run_check,
        )

    @classmethod
    def _index_path(cls, root: Path):
        """Path to the index file."""
        return root / "index.arrow"

    def _scan_index(self, attrs: IntoExpr | None = None) -> pl.LazyFrame:
        """Lazily scan the .gvi index (no collect).

        Returns a LazyFrame with columns ``index, CHROM, POS, REF, ALT, *attrs, ILEN``.
        ``index`` is the physical row index added by ``scan_ipc``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index = pl.scan_ipc(self._index_path(self.path), row_index_name="index")

        schema = index.collect_schema()

        if schema["ALT"] == pl.Utf8:
            index = index.with_columns(pl.col("ALT").cast(pl.List(pl.Utf8)))

        _attrs: set[IntoExpr] = {"ALT"}

        if attrs is not None:
            if not isinstance(attrs, str) and isinstance(attrs, Iterable):
                _attrs.update(attrs)
            else:
                _attrs.add(attrs)
            _attrs.discard("ILEN")
            user_attr_names = [a for a in _attrs - {"ALT"} if isinstance(a, str)]
            if non_numeric := [
                a for a in user_attr_names if not schema[a].is_numeric()
            ]:
                raise ValueError(f"Attrs {non_numeric} must be numeric.")

        attrs = list(_attrs)

        if "ILEN" in schema:
            attrs.append("ILEN")
        elif "ILEN" not in schema:
            attrs.append(ILEN.alias("ILEN"))

        return index.select("index", "CHROM", "POS", "REF", *attrs)

    def _to_df(self) -> pl.DataFrame:
        return self.index.drop("index")

    def _load_genos(self):
        def memmap2array(layout: Content, **kwargs: Any):
            if isinstance(layout, NumpyArray):
                data = layout.data
                if isinstance(data, np.memmap):
                    data = data[:]
                return NumpyArray(data)

        self.genos = ak.transform(memmap2array, self.genos)  # type: ignore

    def write_view(
        self,
        regions: str | tuple[str, int, int] | Path | pl.DataFrame,
        samples: str | Sequence[str] | Path,
        output: str | Path,
        fields: Sequence[str] | None = None,
        reference: "Reference | str | Path | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: Literal["pos", "record", "variant"] = "pos",
        overwrite: bool = False,
        threads: int | None = None,
        progress: bool = False,
    ) -> None:
        """Write a subset of this SparseVar to a new directory.

        Parameters
        ----------
        regions
            Region(s) to include. Accepts the same input types as
            :func:`_normalize_regions`: a ``"chrom:start-end"`` string, a
            ``(chrom, start, end)`` tuple, a BED file path, or a
            polars/pandas/pyranges frame.
        samples
            Samples to include.  Accepts a single sample name, a list, or a
            path to a file of newline-separated names.
        output
            Destination directory for the new SparseVar.
        fields
            Fields to carry over (``None`` = all available except ``"mutcat"``; ``[]`` = none).
            The derived ``mutcat`` field is **never** copied positionally by
            ``write_view`` because its mutation codes — especially DBS adjacency —
            are only valid for the full variant set; subsetting may drop a DBS
            partner and leave a stale 5' code.  Pass ``reference=`` to recompute
            ``mutcat`` on the subset instead (see below).  Explicitly including
            ``"mutcat"`` in *fields* without also providing *reference* raises a
            :class:`ValueError`.
        reference
            If provided (a :class:`~genoray._reference.Reference` instance, or a
            path to a FASTA file), :meth:`annotate_mutations` is called on the
            output view after all other data have been written, recomputing
            ``mutcat`` codes for the subset.  This is the supported way to get a
            valid ``mutcat`` field on a view.  When ``None`` (default), no
            annotation is performed and the output will not have a ``mutcat``
            field.  When provided, the FASTA is validated up front (before any
            output is written) so a bad path fails fast.
        merge_overlapping
            If ``True`` silently merge overlapping regions; if ``False``
            raise ``ValueError`` when overlaps are detected.
        regions_overlap
            How variants are matched to regions — ``"pos"``, ``"record"``, or
            ``"variant"``.  See :func:`_resolve_kept_var_idxs`.
        overwrite
            Whether to overwrite *output* if it already exists.
        threads
            Number of Numba threads to use.  ``None`` uses all available CPUs.
        progress
            If ``True``, display a phase-level :mod:`rich` progress bar while the
            view is written (one tick per major step: counting, genotypes, each
            field, index build, and mutation annotation when *reference* is
            given).  Defaults to ``False`` (no bar, no overhead).

        Notes
        -----
        Variants whose minor allele count is 0 in the chosen sample subset are
        dropped from the output. If every candidate variant drops, a
        :class:`ValueError` is raised — the same code path that fires when
        ``regions`` itself selects no variants.
        """
        from contextlib import nullcontext

        from .._utils import _resolve_threads, numba_threads

        output = Path(output)

        # --- Band A: cheap raises (fail fast; nothing on disk is touched) ---

        # mutcat cannot be positionally copied through a view.
        if fields is not None and "mutcat" in fields:
            if reference is None:
                raise ValueError(
                    "'mutcat' cannot be copied through write_view because its codes "
                    "are dataset-specific (DBS adjacency is only valid for the full "
                    "variant set; subsetting may leave stale codes). "
                    "Pass reference= to recompute mutcat on the subset, or call "
                    "annotate_mutations() on the output view yourself."
                )

        # Output existence: raise (but do NOT delete) before any heavy work.
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {output} already exists. Use overwrite=True to overwrite."
            )

        # Writing a view in place would rmtree the source under overwrite=True.
        if output.resolve() == self.path.resolve():
            raise ValueError(
                "output resolves to the same path as the source dataset; "
                "write_view cannot write a view in place"
            )

        # Normalize inputs (cheap; missing samples/fields raise here).
        regions_df = _normalize_regions(regions, self._c_norm)
        caller_samples = _normalize_samples(samples, self.available_samples)
        fields_to_write = _validate_fields(fields, self.available_fields)
        # Always exclude the derived "mutcat" field from positional copy:
        # its codes encode cross-variant DBS adjacency that is only valid for
        # the full variant set.  Subsetting can drop a DBS 3' partner, leaving
        # an orphaned 5' code that mutation_matrix would miscount.
        # Use reference= to recompute mutcat on the output view instead.
        fields_to_write = [f for f in fields_to_write if f != "mutcat"]

        if not caller_samples:
            raise ValueError("write_view requires at least one sample")

        # Validate the reference up front (existence + .fai build) and reuse the
        # built instance for the final annotate_mutations, so a bad FASTA path
        # fails now instead of after the whole output is written.
        ref_obj: "Reference | None" = None
        if reference is not None:
            ref_obj = (
                reference
                if isinstance(reference, Reference)
                else Reference.from_path(reference)
            )

        # --- Band B: heavier checks (still nothing on disk is touched) ---

        # Resolve kept variant indices.
        if self._covers_all_variants(regions_df, regions_overlap):
            # Fast path: every variant is selected; skip POS/ILEN materialization.
            kept_var_idxs = np.arange(self.n_variants, dtype=V_IDX_TYPE)
        else:
            kept_var_idxs = _resolve_kept_var_idxs(
                self, regions_df, regions_overlap, merge_overlapping
            )
        if len(kept_var_idxs) == 0:
            raise ValueError("no variants selected by `regions`")

        # --- 4. Setup ---
        n_out = len(caller_samples)
        ploidy = self.ploidy
        threads_resolved = _resolve_threads(threads)

        src_sample_idxs = self._s2i[np.array(caller_samples)].astype(np.int64)

        # --- 4.5. Pre-pass: drop variants whose MAC across kept samples is 0 ---
        mac_per_kept = np.zeros(len(kept_var_idxs), dtype=np.int64)
        with numba_threads(threads_resolved):
            _nb_count_mac_per_kept(
                self.genos.data,
                self.genos.offsets,
                src_sample_idxs,
                ploidy,
                kept_var_idxs,
                mac_per_kept,
            )
        keep_mask = mac_per_kept > 0
        n_dropped = int((~keep_mask).sum())
        if n_dropped:
            warnings.warn(
                f"write_view: dropping {n_dropped} variant(s) with MAC=0 in the output sample set",
                stacklevel=2,
            )
            kept_var_idxs = kept_var_idxs[keep_mask]
        if len(kept_var_idxs) == 0:
            raise ValueError(
                "all variants in the selected regions have MAC=0 in the "
                "chosen sample subset; nothing to write"
            )

        # --- Band C: commit. All validation passed; now (re)create the output. ---
        # Phase-level progress bar (opt-in): one tick per major write step,
        # plus one per field, plus the optional mutcat annotation. Built only
        # when progress=True so the default path constructs no Progress object.
        pbar = (
            Progress(*Progress.get_default_columns(), MofNCompleteColumn())
            if progress
            else None
        )
        n_steps = 3 + len(fields_to_write) + (1 if ref_obj is not None else 0)

        with atomic_write_dir(output) as staging, pbar or nullcontext():
            task = (
                pbar.add_task("counting entries", total=n_steps)
                if pbar is not None
                else None
            )

            def _step(desc: str) -> None:
                """Mark the current phase complete and label the next one."""
                if pbar is not None:
                    assert task is not None
                    pbar.advance(task)
                    pbar.update(task, description=desc)

            # (no rmtree/mkdir: atomic_write_dir already created `staging`)

            # --- 5. Pass 1: count kept entries per output slot ---
            out_lengths = np.zeros(n_out * ploidy, dtype=np.int64)
            with numba_threads(threads_resolved):
                _nb_count_kept(
                    self.genos.data,
                    self.genos.offsets,
                    src_sample_idxs,
                    ploidy,
                    kept_var_idxs,
                    out_lengths,
                )

            new_offsets = lengths_to_offsets(out_lengths.reshape(n_out, ploidy))

            # --- 6. Write offsets.npy ---
            offsets_mm = np.memmap(
                staging / "offsets.npy",
                dtype=np.int64,
                mode="w+",
                shape=new_offsets.shape,
            )
            offsets_mm[:] = new_offsets
            offsets_mm.flush()

            # Allocate output variant_idxs memmap
            n_entries = int(new_offsets[-1])
            out_var_idxs_mm = np.memmap(
                staging / "variant_idxs.npy",
                dtype=V_IDX_TYPE,
                mode="w+",
                shape=(n_entries,),
            )
            _step("writing genotypes")

            # --- 7. Pass 2 (genos): write remapped variant indices ---
            with numba_threads(threads_resolved):
                _nb_write_var_idxs(
                    self.genos.data,
                    self.genos.offsets,
                    src_sample_idxs,
                    ploidy,
                    kept_var_idxs,
                    new_offsets.ravel(),
                    out_var_idxs_mm,
                )
            out_var_idxs_mm.flush()

            # --- 8. Pass 2 (fields): write each field ---
            for name in fields_to_write:
                _step(f"field: {name}")
                dtype = self.available_fields[name]
                src_field_rag = _open_fmt(
                    name, dtype, self.path, (self.n_samples, ploidy, None), "r"
                )
                out_field_mm = np.memmap(
                    staging / f"{name}.npy",
                    dtype=dtype,
                    mode="w+",
                    shape=(n_entries,),
                )
                with numba_threads(threads_resolved):
                    _nb_write_field(
                        src_field_rag.data,
                        self.genos.data,
                        self.genos.offsets,
                        src_sample_idxs,
                        ploidy,
                        kept_var_idxs,
                        new_offsets.ravel(),
                        out_field_mm,
                    )
                out_field_mm.flush()
                del src_field_rag

            _step("building index")

            # --- 9. Build new index (streaming: never materialize the full index) ---
            # Compute AFs over the written genos.
            n_alleles = n_out * ploidy
            afs = np.zeros(len(kept_var_idxs), dtype=np.float32)
            _nb_af_helper(afs, out_var_idxs_mm, new_offsets.ravel(), n_alleles)

            # Small, output-sized frame keyed by the kept physical row index.
            # The row-index column produced by scan_ipc is UInt32 (see _scan_index);
            # match that dtype so the join keys align.
            idx_dtype = self._index_lazy.collect_schema()["index"]
            af_frame = pl.DataFrame(
                {
                    "index": pl.Series(kept_var_idxs).cast(idx_dtype),
                    "AF": pl.Series(afs),
                }
            )

            base = self._index_lazy
            drop_existing_af = ["AF"] if "AF" in base.collect_schema().names() else []
            out_index = (
                base.drop(drop_existing_af)
                .join(
                    af_frame.lazy(), on="index", how="inner"
                )  # filter to kept + attach AF
                .sort(
                    "index"
                )  # row order must match the ascending kept_var_idxs / written genos
                .drop("index")  # physical row index is not part of the output schema
            )
            # sink_ipc forces the streaming engine, so the inner join filters the
            # scan down to output size before the sort — peak RAM scales with the
            # selected subset, not the full input index.
            out_index.sink_ipc(SparseVar._index_path(staging))

            # --- 10. Write metadata.json ---
            with open(staging / "metadata.json", "w") as f:
                json_str = SparseVarMetadata(
                    version=CURRENT_VERSION,
                    samples=caller_samples,
                    ploidy=ploidy,
                    contigs=self.contigs,
                    fields={n: self.available_fields[n].name for n in fields_to_write},
                ).model_dump_json()
                f.write(json_str)

            # --- 11. Optionally recompute mutcat on the output view ---
            if ref_obj is not None:
                _step("annotating mutations")
                out_svar = SparseVar(staging)
                out_svar.annotate_mutations(ref_obj, write_back=True)

            # Final advance so the bar reads N/N on completion.
            if pbar is not None:
                assert task is not None
                pbar.advance(task)

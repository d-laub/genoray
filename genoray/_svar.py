from __future__ import annotations

import copy
import re
import shutil
import warnings
from collections.abc import Callable, Iterable, Sequence
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import cached_property
from typing import Any, Generic, Literal, TypeVar, cast, overload

import awkward as ak
import joblib
import numba as nb
import numpy as np
import polars as pl
import polars_bio as pb
import polars_config_meta  # noqa: F401
import seqpro as sp
from awkward.contents import Content, NumpyArray
from hirola import HashTable
from joblib_progress import joblib_progress
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from pgenlib import PgenReader
from polars._typing import IntoExpr
from pydantic import BaseModel
from rich.progress import MofNCompleteColumn, Progress
from seqpro.rag import OFFSET_TYPE, Ragged, lengths_to_offsets
from tqdm.auto import tqdm

from ._mutcat import MUTCAT_VERSION, build_entry_codes, classify_variants, count_matrix
from ._pgen import PGEN
from ._reference import Reference
from ._signatures import _load_signature_file, cosmic_signatures, fit_signatures
from ._types import DOSAGE_TYPE, DTYPE, POS_MAX, POS_TYPE, V_IDX_TYPE
from ._utils import ContigNormalizer, atomic_write_dir, format_memory, parse_memory
from ._var_ranges import var_ranges
from ._vcf import VCF
from .exprs import ILEN

NUMERIC = TypeVar("NUMERIC", bound=np.number)

_REGION_STR_RE = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def _coerce_bed_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce a BED-like frame to columns chrom (Utf8), start (Int32), end (Int32).

    Handles both the seqpro convention (chromStart/chromEnd) and the
    polars-bio convention (start/end), as well as PyRanges-style
    (Chromosome/Start/End) via sp.bed.from_pyr.
    """
    rename: dict[str, str] = {}
    cols = set(df.columns)
    for src, dst in (
        ("Chromosome", "chrom"),
        ("CHROM", "chrom"),
        ("chromStart", "start"),
        ("chromEnd", "end"),
        ("Start", "start"),
        ("End", "end"),
    ):
        if src in cols and dst not in cols:
            rename[src] = dst
    if rename:
        df = df.rename(rename)
    return df.select(
        pl.col("chrom").cast(pl.Utf8),
        pl.col("start").cast(pl.Int32),
        pl.col("end").cast(pl.Int32),
    )


def _normalize_regions(
    regions: "str | tuple[str, int, int] | PathLike | object",
    cnorm: ContigNormalizer,
) -> pl.DataFrame:
    """Normalize *regions* to a DataFrame with columns chrom (Utf8), start (Int32),
    end (Int32) using 0-based, end-exclusive coordinates.

    Accepted input types:

    * ``str`` — ``"chrom:start-end"`` (1-based inclusive, converted to 0-based half-open).
    * ``tuple[str, int, int]`` — ``(chrom, start, end)`` already 0-based half-open.
    * ``Path`` / ``PathLike`` — path to a BED3+ file; read via ``sp.bed.read``.
    * ``pl.DataFrame`` (or any frame-like) — must already have ``chrom``, ``start``,
      and ``end`` columns (or common aliases).

    Rows whose contig is not recognised by *cnorm* are dropped with a ``UserWarning``.
    """
    if isinstance(regions, str):
        m = _REGION_STR_RE.match(regions)
        if m is None:
            raise ValueError(
                f"Region string {regions!r} does not match 'chrom:start-end'"
            )
        chrom = m["chrom"]
        start = int(m["start"]) - 1  # 1-based inclusive → 0-based
        end = int(m["end"])
        df = pl.DataFrame(
            {"chrom": [chrom], "start": [start], "end": [end]},
            schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
        )
    elif (
        isinstance(regions, tuple) and len(regions) == 3 and isinstance(regions[0], str)
    ):
        chrom, start, end = regions
        df = pl.DataFrame(
            {"chrom": [chrom], "start": [int(start)], "end": [int(end)]},
            schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
        )
    elif isinstance(regions, PathLike):
        raw = sp.bed.read(Path(regions))
        df = _coerce_bed_schema(raw)
    else:
        # Frame-like
        if isinstance(regions, pl.DataFrame):
            df = regions
        else:
            # Try pandas
            try:
                import pandas as pd
            except ImportError:
                pd = None
            if pd is not None and isinstance(regions, pd.DataFrame):
                df = pl.from_pandas(regions)
            else:
                # Try pyranges (v0 or v1)
                pyr_df = None
                for mod_name in ("pyranges", "pyranges1"):
                    try:
                        pyr_mod = __import__(mod_name)
                    except ImportError:
                        continue
                    pr_cls = getattr(pyr_mod, "PyRanges", None)
                    if pr_cls is not None and isinstance(regions, pr_cls):
                        # pyranges0 exposes .df; pyranges1 exposes .to_pandas()
                        if hasattr(regions, "df"):
                            pyr_df = pl.from_pandas(regions.df)
                        else:
                            pyr_df = pl.from_pandas(regions.to_pandas())
                        break
                if pyr_df is None:
                    raise TypeError(
                        f"Unsupported regions type: {type(regions).__name__}. "
                        "Expected str, tuple, PathLike, or a polars/pandas/pyranges frame."
                    )
                df = pyr_df
        df = _coerce_bed_schema(df)

    normed = [cnorm.norm(c) for c in df["chrom"].to_list()]
    normed_chroms = [n for n in normed if n is not None]
    keep_mask = [n is not None for n in normed]
    if not all(keep_mask):
        n_dropped = sum(1 for k in keep_mask if not k)
        warnings.warn(
            f"{n_dropped} region(s) dropped: contig not in dataset.", stacklevel=2
        )
    df = df.filter(pl.Series(keep_mask))
    df = df.with_columns(pl.Series("chrom", normed_chroms))
    return df


def _normalize_samples(
    samples: "str | Sequence[str] | PathLike",
    available: Sequence[str],
) -> list[str]:
    """Normalize `samples` to a list of valid sample names, preserving caller order
    and deduping by first occurrence. Raises ValueError on unknown samples."""
    if isinstance(samples, str):
        candidates: list[str] = [samples]
    elif isinstance(samples, PathLike):
        candidates = Path(samples).read_text().splitlines()
        candidates = [s for s in candidates if s.strip()]
    else:
        candidates = list(samples)

    avail_set = set(available)
    missing = [s for s in candidates if s not in avail_set]
    if missing:
        raise ValueError(f"Samples not found in dataset: {missing}")

    seen: set[str] = set()
    deduped: list[str] = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _validate_fields(
    fields: "Sequence[str] | None",
    available: "dict[str, np.dtype]",
) -> list[str]:
    """Validate field selection. `None` returns all available fields; a sequence is
    validated as a subset of `available`. Raises ValueError on unknown fields."""
    if fields is None:
        return list(available)
    fields = list(fields)
    missing = [f for f in fields if f not in available]
    if missing:
        raise ValueError(f"Fields not found in dataset: {missing}")
    return fields


def _resolve_kept_rows(
    index_df: pl.DataFrame,
    c_norm: "ContigNormalizer",
    regions: pl.DataFrame,
    mode: Literal["pos", "record", "variant"],
    merge_overlapping: bool,
) -> "NDArray[V_IDX_TYPE]":
    """Return a sorted, deduplicated array of kept row positions (values from the
    ``index`` column) in *index_df*.

    *index_df* must have columns ``CHROM`` (Utf8), ``POS`` (Int32), ``ILEN``
    (list[int]) and ``index`` (the id returned for kept rows). Coordinates in
    *regions* are 0-based, half-open (chrom/start/end).

    Parameters
    ----------
    index_df
        Variant index DataFrame with columns ``index``, ``CHROM``, ``POS``, ``ILEN``.
    c_norm
        ContigNormalizer for the dataset.
    regions
        Normalised BED-like frame with columns ``chrom`` (Utf8), ``start`` (Int32),
        ``end`` (Int32).  Coordinates are 0-based, half-open.
    mode
        ``"variant"`` — any variant whose span (accounting for ILEN) overlaps the
        region is kept, as returned by ``var_ranges``.
        ``"pos"`` — keep only variants whose POS-1 (0-based) falls strictly inside
        ``[start, end)``.
        ``"record"`` — like ``"pos"`` but widen the end by 1, i.e. POS-1 in
        ``[start, end + 1)``.
    merge_overlapping
        If *True*, overlapping regions are silently merged before querying.
        If *False* and overlapping regions are detected, raise ``ValueError``.

    Returns
    -------
    NDArray[V_IDX_TYPE]
        Sorted, deduplicated 1-D array of ``index`` column values.
    """
    if regions.height == 0:
        return np.empty(0, dtype=V_IDX_TYPE)

    # --- overlap detection / optional merge ---
    # sp.bed.to_pyr requires chromStart/chromEnd column names.
    pyr_input = regions.rename({"start": "chromStart", "end": "chromEnd"})
    pyr = sp.bed.to_pyr(pyr_input)  # type: ignore[bad-argument-type]
    mod = type(pyr).__module__.split(".")[0]
    if mod == "pyranges":
        merged = pyr.merge()
    elif mod == "pyranges1":
        merged = pyr.merge_overlaps()
    else:
        raise RuntimeError(f"Unexpected PyRanges module: {type(pyr)!r}")

    if len(merged) != regions.height:
        if not merge_overlapping:
            raise ValueError("regions overlap; pass merge_overlapping=True to dedupe")
        regions = _coerce_bed_schema(sp.bed.from_pyr(merged))

    # --- collect candidate index values via var_ranges ---
    kept_chunks: list[NDArray[V_IDX_TYPE]] = []
    sentinel = np.iinfo(V_IDX_TYPE).max
    for contig_key, sub in regions.group_by("chrom", maintain_order=False):
        c = contig_key[0] if isinstance(contig_key, tuple) else contig_key
        starts = sub["start"].to_numpy()
        ends = sub["end"].to_numpy()
        vr = var_ranges(c_norm, index_df, c, starts, ends)  # shape (n_ranges, 2)
        valid = vr[:, 0] != sentinel
        for s, e in vr[valid]:
            kept_chunks.append(np.arange(s, e, dtype=V_IDX_TYPE))

    if not kept_chunks:
        return np.empty(0, dtype=V_IDX_TYPE)
    candidates = np.unique(np.concatenate(kept_chunks))

    # "variant" mode: var_ranges already does ILEN-aware overlap — return as-is.
    if mode == "variant":
        return candidates

    # --- pos / record mode: filter by POS membership ---
    region_by_contig: dict[str, tuple[NDArray, NDArray]] = {}
    for contig_key, sub in regions.group_by("chrom", maintain_order=False):
        c = contig_key[0] if isinstance(contig_key, tuple) else contig_key
        region_by_contig[c] = (sub["start"].to_numpy(), sub["end"].to_numpy())

    # Filter index_df to candidate rows.  Both candidates (from np.unique) and the
    # sorted filter result are ascending by index value, so the rows align without
    # any explicit reordering.
    by_id = index_df.filter(pl.col("index").is_in(candidates.tolist())).sort("index")
    cand_pos0 = by_id["POS"].to_numpy() - 1  # 1-based POS → 0-based
    cand_chrom = by_id["CHROM"].to_list()
    cand_ids = by_id["index"].to_numpy()

    end_offset = 0 if mode == "pos" else 1  # "record" widens end by 1
    keep_mask = np.zeros(len(cand_ids), dtype=bool)
    for i in range(len(cand_ids)):
        pair = region_by_contig.get(cand_chrom[i])
        if pair is None:
            continue
        r_starts, r_ends = pair
        p = cand_pos0[i]
        if np.any((r_starts <= p) & (p < r_ends + end_offset)):
            keep_mask[i] = True

    return cand_ids[keep_mask].astype(V_IDX_TYPE)


def _resolve_kept_var_idxs(
    sv: "SparseVar",
    regions: pl.DataFrame,
    mode: Literal["pos", "record", "variant"],
    merge_overlapping: bool,
) -> "NDArray[V_IDX_TYPE]":
    """Resolve kept row positions using only the numeric index columns.

    Collects ``index/CHROM/POS/ILEN`` from the lazy index (bounded RAM) instead
    of materializing the full string-bearing index.
    """
    idx_numeric = sv._index_lazy.select("index", "CHROM", "POS", "ILEN").collect()
    return _resolve_kept_rows(idx_numeric, sv._c_norm, regions, mode, merge_overlapping)


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


class SparseVar(Generic[_SRT]):
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
    index: pl.DataFrame
    """Table of variants with columns: `CHROM`, `POS`, `REF`, `ALT`, `ILEN`, and any additional
    attributes specified in `attrs` on construction."""
    _c_norm: ContigNormalizer
    _s2i: HashTable
    _c_max_idxs: dict[str, int]
    _is_biallelic: bool

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
    def index(self) -> pl.DataFrame:
        """The full variant index, materialized on first access."""
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
        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            samples = np.atleast_1d(np.array(samples))
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")

        s_idxs = cast(NDArray[np.int64], self._s2i[samples])

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

        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            samples = np.atleast_1d(np.array(samples))
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")

        s_idxs = cast(NDArray[np.int64], self._s2i[samples])

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
        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            samples = np.atleast_1d(np.array(samples))
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")

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
        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")
            samples = np.atleast_1d(np.array(samples))

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
            vcf._index_path(), vcf._pl_filter
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

        with atomic_write_dir(out) as staging:
            with open(staging / "metadata.json", "w") as f:
                json_str = SparseVarMetadata(
                    version=CURRENT_VERSION,
                    contigs=contigs,
                    samples=caller_samples,
                    ploidy=out_ploidy,
                    fields={"dosages": np.dtype(DOSAGE_TYPE).name}
                    if with_dosages
                    else {},
                ).model_dump_json()
                f.write(json_str)

            subsetting_samples = samples is not None
            # When NOT subsetting samples, write the (region-restricted) index up front.
            if not subsetting_samples:
                _write_index_from_working(
                    working_df,
                    kept_rows,
                    cls._index_path(staging),
                    alt_is_utf8,
                    ilen_added,
                )

            max_mem = parse_memory(max_mem)
            n_out = len(caller_samples)
            effective_n_jobs = joblib.cpu_count() if n_jobs == -1 else n_jobs
            effective_n_jobs = min(effective_n_jobs, len(contigs))
            job_mem = max_mem // effective_n_jobs

            with TemporaryDirectory(dir=out.parent) as chunk_dir:
                chunk_dir = Path(chunk_dir)

                shape = (n_out, out_ploidy)
                tasks = []
                for chunk_idx, c in enumerate(contigs):
                    task = joblib.delayed(_process_contig_vcf)(
                        vcf.path,
                        dosage_field=vcf.dosage_field if with_dosages else None,
                        max_mem=job_mem,
                        contig=c,
                        chunk_dir=chunk_dir,
                        chunk_idx=chunk_idx,
                        cyvcf2_filter=vcf._filter,
                        pl_filter=vcf._pl_filter,
                        caller_samples=None if samples is None else caller_samples,
                        keep_local=keep_local_by_contig.get(c),
                        haploid=haploid,
                    )
                    tasks.append(task)

                with (
                    joblib_progress(
                        description=f"Processing contigs using {effective_n_jobs} jobs",
                        total=len(tasks),
                    ),
                    joblib.Parallel(n_jobs=effective_n_jobs) as parallel,
                ):
                    results: list[tuple[int, int]] = list(parallel(tasks))  # type: ignore

                logger.info("Concatenating intermediate chunks")
                _concat_data(
                    staging, chunk_dir, shape, results, with_dosages=with_dosages
                )

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
                        cls._index_path(staging),
                        alt_is_utf8,
                        ilen_added,
                        af=af,
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
        n_out = len(caller_samples)

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

        with atomic_write_dir(out) as staging:
            with open(staging / "metadata.json", "w") as f:
                json_str = SparseVarMetadata(
                    version=CURRENT_VERSION,
                    contigs=contigs,
                    samples=caller_samples,
                    ploidy=out_ploidy,
                    fields={"dosages": np.dtype(DOSAGE_TYPE).name}
                    if with_dosages
                    else {},
                ).model_dump_json()
                f.write(json_str)

            if with_dosages and pgen._sei is None:
                raise ValueError("PGEN must be bi-allelic with filters applied")

            subsetting_samples = samples is not None
            # (mirrors from_vcf; keep in sync) metadata written + no-subset index path
            if not subsetting_samples:
                _write_index_from_working(
                    working_df,
                    kept_rows,
                    cls._index_path(staging),
                    alt_is_utf8,
                    ilen_added,
                )

            max_mem = parse_memory(max_mem)
            effective_n_jobs = joblib.cpu_count() if n_jobs == -1 else n_jobs
            effective_n_jobs = min(effective_n_jobs, len(contigs))
            job_mem = max_mem // effective_n_jobs
            mem_per_var = pgen._mem_per_variant(
                pgen.GenosDosages if with_dosages else pgen.Genos  # type: ignore
            )

            shape = (n_out, out_ploidy)
            with TemporaryDirectory(dir=out.parent) as contig_dir:
                contig_dir = Path(contig_dir)

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
                        chunk_dir=contig_dir,
                        chunk_idx=len(tasks),
                        sample_subset=sample_subset,
                        haploid=haploid,
                    )
                    tasks.append(task)

                pgen._free_index()
                # PgenReaders can be multi-GB allocations, close them to free memory
                pgen._geno_pgen.close()
                if pgen.dosage_path is not None:
                    pgen._dose_pgen.close()

                with (
                    joblib_progress(
                        description=f"Processing contigs using {effective_n_jobs} jobs",
                        total=len(tasks),
                    ),
                    joblib.Parallel(n_jobs=effective_n_jobs) as parallel,
                ):
                    results: list[tuple[int, int]] = list(parallel(tasks))  # type: ignore

                logger.info("Concatenating intermediate chunks")
                _concat_data(
                    staging, contig_dir, shape, results, with_dosages=with_dosages
                )

                # (mirrors from_vcf; keep in sync) MAC-drop + subset-sample index finalize
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
                        cls._index_path(staging),
                        alt_is_utf8,
                        ilen_added,
                        af=af,
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

    def annotate_with_gtf(
        self,
        gtf: str | pl.DataFrame,
        level_filter: int | None = 1,
        write_back: bool = True,
        *,
        strand_encoding: dict[str | None, int] | None = None,
        codon_null_token: int | None = None,
    ) -> pl.DataFrame:
        """
        Annotate variants with gene_id, strand, and codon_pos from GTF CDS features.

        Computes codon position for SNVs only; indels receive strand but null codon_pos.

        Parameters
        ----------
        gtf : str or pl.DataFrame
            Path to GTF file (.gtf or .gtf.gz) or pre-loaded Polars DataFrame.
        level_filter : int or None, default 1
            If set, keep rows with GTF 'level' <= level_filter (1 = highest quality).
        write_back : bool, default True
            If True, update self.var_table in-place and write to index.arrow file.
        strand_encoding : dict or None, optional
            Encode strand as integers. Example: {'+': 0, '-': 1, None: 2}
        codon_null_token : int or None, optional
            Replace null codon_pos with this integer for ML models.

        Returns
        -------
        pl.DataFrame
            Columns: varID (UInt32), gene_id (Utf8), strand (Utf8/Int16), codon_pos (Int8/Int16)

        Examples
        --------
        >>> svar = SparseVar("data.svar")
        >>> annot = svar.annotate_with_gtf("gencode.v45.gtf.gz")
        >>> annot.head()
        """
        # Validate inputs
        if level_filter is not None and not isinstance(level_filter, int):
            raise TypeError(
                f"level_filter must be int or None, got {type(level_filter)}"
            )
        if strand_encoding is not None and not isinstance(strand_encoding, dict):
            raise TypeError("strand_encoding must be dict or None")
        if codon_null_token is not None and not isinstance(codon_null_token, int):
            raise TypeError("codon_null_token must be int or None")

        logger.info("Loading GTF for CDS annotation")

        with tqdm(total=3, desc="GTF annotation", unit="step") as pbar:
            # Load GTF
            pbar.set_description("Loading GTF")
            gtf_df = _load_gtf(gtf)
            if level_filter is not None and "level" in gtf_df.columns:
                gtf_df = gtf_df.filter(pl.col("level").cast(pl.Int32) <= level_filter)
            pbar.update(1)

            # CDS Annotation
            pbar.set_description("CDS annotation")

            # Extract CDS features with gene_biotype
            cds_df = gtf_df.filter(pl.col("feature") == "CDS").select(
                "chrom",
                "start",
                "end",
                "strand",
                "frame",
                "gene_id",
                "transcript_id",
                "gene_biotype",
                "transcript_support_level",
                "tag",
            )

            if len(cds_df) == 0:
                annot = _empty_annot()
            else:
                annot = _get_strand_and_codon_pos(cds_df, self.index, self._c_norm)
            pbar.update()

            # Apply encoding
            pbar.set_description("Finalizing")
            if strand_encoding is not None:
                str_map = {k: v for k, v in strand_encoding.items() if k is not None}
                null_val = strand_encoding.get(None)
                strand_expr = pl.col("strand").replace_strict(str_map, default=null_val)
                annot = annot.with_columns(strand_expr.cast(pl.Int16).alias("strand"))

            if codon_null_token is not None:
                annot = annot.with_columns(
                    pl.col("codon_pos").fill_null(codon_null_token).cast(pl.Int16)
                )

            # Write back if requested
            if write_back:
                self._load_all_attrs()
                self.index = (
                    self.index.lazy()
                    .with_row_index("varID")
                    # maintain_order="left" is required: index rows are
                    # positionally aligned with the sparse genotype storage, so
                    # the join must not reorder them. The default ('none') lets
                    # polars' hash join reorder output on larger data.
                    .join(annot.lazy(), on="varID", how="left", maintain_order="left")
                    .drop("varID")
                    .collect()
                )
                df = self._to_df()
                df.write_ipc(self._index_path(self.path))
                logger.info("Wrote gene_id, strand, codon_pos to index.arrow")

            pbar.update(1)

        return annot

    def annotate_mutations(
        self,
        reference: "Reference | str | Path",
        *,
        contigs: "list[str] | None" = None,
        write_back: bool = True,
    ) -> None:
        """Classify every variant into SBS-96 / DBS-78 / ID-83 channels and store
        a per-genotype-entry ``mutcat`` field (int16, enum-encoded).

        Adjacent SNVs carried on the same haplotype are combined into DBS; the
        5' entry receives the DBS code and the 3' entry a ``DBS_PARTNER`` sentinel.

        Parameters
        ----------
        reference
            Reference genome.  A :class:`~genoray._reference.Reference` instance,
            or a path to a FASTA file (with a ``.fai`` index alongside it).
        contigs
            If given, only variants on these contigs are classified; entries on
            all other contigs are marked ``NOT_ANNOTATED`` and their contigs are
            never fetched from the reference.  Names are matched via the
            :class:`~genoray._utils.ContigNormalizer` (so ``chr1``/``1`` both
            work).  Requested contigs absent from the ``.svar`` index are skipped
            with a warning.  A listed contig present in the index but absent from
            the reference still raises (use the allowlist to exclude it instead).
            ``None`` (default) classifies all contigs.
        write_back
            If ``True`` (default), persist ``mutcat.npy`` and update
            ``metadata.json`` on disk so that subsequent ``SparseVar(...)``
            opens will see the field.  If ``False``, the ``mutcat`` field lives
            only in memory (``self.fields["mutcat"]``) and is NOT written to
            disk — reopening the file will not find it.  Note: the
            ``metadata.json`` update is not safe against concurrent writers;
            single-writer access is expected (consistent with
            ``annotate_with_gtf``).
        """
        if not isinstance(reference, Reference):
            reference = Reference.from_path(reference)

        # 0. resolve contig scope
        index_chroms = self.index["CHROM"].to_list()
        if contigs is None:
            scoped_contigs: "list[str] | None" = None
            in_scope = np.ones(self.index.height, dtype=np.bool_)
        else:
            normalized = self._c_norm.norm(list(contigs))
            unmatched = [c for c, nm in zip(contigs, normalized) if nm is None]
            if unmatched:
                logger.warning(
                    f"annotate_mutations: {len(unmatched)} requested contig(s) not "
                    f"found in the .svar index; they will be skipped: {unmatched}"
                )
            scope_set = {nm for nm in normalized if nm is not None}
            scoped_contigs = sorted(scope_set)
            in_scope = np.array([c in scope_set for c in index_chroms], dtype=np.bool_)

        # 1. intrinsic per-variant codes (scoped)
        var_code = classify_variants(self.index, reference, contigs=scoped_contigs)

        # 2. per-variant arrays needed by the adjacency kernel
        pos = self.index["POS"].to_numpy().astype(np.int64)
        ref0 = self.index["REF"].to_list()
        alt0 = self.index["ALT"].list.first().to_list()
        is_snv = np.array(
            [
                r is not None and a is not None and len(r) == 1 and len(a) == 1
                for r, a in zip(ref0, alt0)
            ],
            dtype=np.bool_,
        )
        # gate adjacency: out-of-scope variants must not be collapsed into DBS
        is_snv &= in_scope
        # contig id per variant — equality semantics only (same contig ↔ same id)
        contig_map = {c: i for i, c in enumerate(self.contigs)}
        contig_codes = np.array(
            [contig_map.get(c, -1) for c in self.index["CHROM"].to_list()],
            dtype=np.int32,
        )
        ref_b = np.array([ord(r[0]) if r else 0 for r in ref0], dtype=np.uint8)
        alt_b = np.array([ord(a[0]) if a else 0 for a in alt0], dtype=np.uint8)

        # 3. broadcast to entries + DBS adjacency override
        entry_codes = build_entry_codes(
            self.genos.data,
            self.genos.offsets,
            var_code,
            pos,
            contig_codes,
            is_snv,
            ref_b,
            alt_b,
        )

        # 4. register in-memory (mirrors how fields are opened in __init__)
        shape = (self.n_samples, self.ploidy, None)
        self.available_fields["mutcat"] = np.dtype("int16")
        self.fields["mutcat"] = Ragged.from_offsets(
            entry_codes, shape, self.genos.offsets
        )

        # 5. optionally persist
        if write_back:
            mm = np.memmap(
                self.path / "mutcat.npy",
                dtype=np.int16,
                mode="w+",
                shape=entry_codes.shape,
            )
            mm[:] = entry_codes
            mm.flush()
            del mm

            with open(self.path / "metadata.json", "rb") as f:
                meta = SparseVarMetadata.model_validate_json(f.read())
            meta.fields["mutcat"] = "int16"
            meta.mutcat_version = MUTCAT_VERSION
            meta.mutcat_contigs = scoped_contigs
            with open(self.path / "metadata.json", "w") as f:
                f.write(meta.model_dump_json())

    def mutation_matrix(
        self,
        kind: Literal["SBS96", "DBS78", "ID83"],
        *,
        count: Literal["allele", "sample"] = "allele",
    ) -> pl.DataFrame:
        """Build a per-sample mutation count matrix.

        Requires :meth:`annotate_mutations` to have been run (or the ``mutcat``
        field to be loaded). Returns a DataFrame with a ``MutationType`` column
        plus one column per sample (rows in fixed codebook order).

        The ``mutcat`` field is resolved in the following priority order:

        1. Already loaded in ``self.fields["mutcat"]`` (e.g. opened with
           ``fields=["mutcat"]``).
        2. Present on disk as ``mutcat.npy`` (written by a prior
           :meth:`annotate_mutations` call with ``write_back=True``) — opened
           lazily and cached into ``self.fields["mutcat"]`` for subsequent
           calls.
        3. Not found at all — raises :class:`ValueError`.

        Parameters
        ----------
        kind
            One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
        count
            ``"allele"`` counts every non-ref allele copy; ``"sample"`` counts
            each category at most once per sample (presence/absence).
        """
        if kind not in ("SBS96", "DBS78", "ID83"):
            raise ValueError(f"Unknown matrix kind {kind!r}.")
        if count not in ("allele", "sample"):
            raise ValueError(
                f"Unknown count mode {count!r}; choose 'allele' or 'sample'."
            )
        mut = self.fields.get("mutcat")
        if mut is None:
            if "mutcat" in self.available_fields:
                shape = (self.n_samples, self.ploidy, None)
                mut = _open_fmt(
                    "mutcat", self.available_fields["mutcat"], self.path, shape, "r"
                )
                self.fields["mutcat"] = mut
            else:
                raise ValueError(
                    "No 'mutcat' field found. Run annotate_mutations() first "
                    "(or open with fields=['mutcat'])."
                )
        return count_matrix(
            np.asarray(mut.data),
            np.asarray(self.genos.offsets),
            self.ploidy,
            self.n_samples,
            self.available_samples,
            kind,
            per_sample=(count == "sample"),
        )

    def assign_signatures(
        self,
        kind: Literal["SBS96", "DBS78", "ID83"],
        *,
        reference: "pl.DataFrame | str | Path | None" = None,
        count: Literal["allele", "sample"] = "allele",
        max_delta: float = 0.01,
        min_activity: float = 0.005,
        n_jobs: int = 1,
        backend: str = "loky",
    ) -> "pl.DataFrame":
        """Refit this object's mutation catalogue against COSMIC signatures.

        Builds the ``kind`` catalogue via :meth:`mutation_matrix` and decomposes
        it into per-sample activities via :func:`genoray.fit_signatures`.

        Parameters
        ----------
        kind
            One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
        reference
            Reference signatures as a Polars ``DataFrame`` (``MutationType`` +
            signature columns), a path to a COSMIC-style TSV, or ``None`` to fetch
            the default COSMIC set via :func:`genoray.cosmic_signatures`.
        count
            Counting unit passed to :meth:`mutation_matrix`.
        max_delta, min_activity
            Forwarded to :func:`genoray.fit_signatures`.
        n_jobs, backend
            Forwarded to :func:`genoray.fit_signatures` to control per-sample
            parallelism (``1`` (default) runs serially; ``-1`` uses all cores;
            process-based ``"loky"`` backend).

        Returns
        -------
        pl.DataFrame
            One row per sample: ``Sample``, one column per signature, and
            ``cosine_similarity``.
        """
        catalogue = self.mutation_matrix(kind, count=count)
        if reference is None:
            ref = cosmic_signatures(kind)
        elif isinstance(reference, pl.DataFrame):
            ref = reference
        else:
            ref = _load_signature_file(reference)
        return fit_signatures(
            catalogue,
            ref,
            max_delta=max_delta,
            min_activity=min_activity,
            n_jobs=n_jobs,
            backend=backend,
        )

    def cache_afs(self):
        """Cache the allele frequencies on disk. Will also load all possible attributes and add the AF column in-memory."""
        self._load_all_attrs()
        afs = self._compute_afs()
        self.index = self.index.with_columns(AF=pl.Series(afs))
        self._write_afs()

    def _load_all_attrs(self):
        idx_df = pl.scan_ipc(self._index_path(self.path))
        schema = idx_df.collect_schema()
        missing = set(schema) - set(self.index.columns)
        missing_attrs = idx_df.select(*missing).collect()
        self.index = self.index.hstack(missing_attrs)

    def _compute_afs(self) -> NDArray[np.float32]:
        n_samples, ploidy, _ = cast(tuple[int, int, None], self.genos.shape)
        max_count = n_samples * ploidy
        afs = np.zeros(self.n_variants, np.float32)
        _nb_af_helper(afs, self.genos.data, self.genos.offsets, max_count)
        return afs

    def _write_afs(self):
        df = self._to_df()
        df.write_ipc(self._index_path(self.path))

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

        from ._utils import _resolve_threads, numba_threads

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

        with pbar or nullcontext():
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

            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True)

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
                output / "offsets.npy",
                dtype=np.int64,
                mode="w+",
                shape=new_offsets.shape,
            )
            offsets_mm[:] = new_offsets
            offsets_mm.flush()

            # Allocate output variant_idxs memmap
            n_entries = int(new_offsets[-1])
            out_var_idxs_mm = np.memmap(
                output / "variant_idxs.npy",
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
                    output / f"{name}.npy",
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
            out_index.sink_ipc(SparseVar._index_path(output))

            # --- 10. Write metadata.json ---
            with open(output / "metadata.json", "w") as f:
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
                out_svar = SparseVar(output)
                out_svar.annotate_mutations(ref_obj, write_back=True)

            # Final advance so the bar reads N/N on completion.
            if pbar is not None:
                assert task is not None
                pbar.advance(task)


@nb.njit(nogil=True, cache=True)
def _nb_af_helper(
    afs: NDArray[np.float32],
    v_idxs: NDArray[np.int32],
    offsets: NDArray[np.int64],
    max_count: int,
):
    for i in range(len(offsets) - 1):
        o_s, o_e = offsets[i], offsets[i + 1]
        v_slice = v_idxs[o_s:o_e]
        afs[v_slice] += 1
    afs /= max_count


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_count_kept(
    src_data: NDArray[np.int32],
    src_offsets: NDArray[np.int64],
    src_sample_idxs: NDArray[np.int64],
    ploidy: int,
    kept_var_idxs: NDArray[np.int32],
    out_lengths: NDArray[np.int64],
):
    """Pass 1: count, per output (sample, ploidy) slot, how many source variant
    indices fall in `kept_var_idxs`."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):  # type: ignore[not-iterable]
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            count = 0
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    count += 1
            out_lengths[i * ploidy + p] = count


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_count_mac_per_kept(
    src_data: NDArray[np.int32],
    src_offsets: NDArray[np.int64],
    src_sample_idxs: NDArray[np.int64],
    ploidy: int,
    kept_var_idxs: NDArray[np.int32],
    mac_out: NDArray[np.int64],
):
    """Count, per kept variant, the number of non-ref entries across (sample, ploidy)
    in the output. Outer prange is over kept variants so each writes its own slot —
    no atomics needed."""
    n_kept = kept_var_idxs.shape[0]
    n_samples = src_sample_idxs.shape[0]
    for k in nb.prange(n_kept):  # type: ignore[not-iterable]
        v = kept_var_idxs[k]
        count = 0
        for i in range(n_samples):
            s = src_sample_idxs[i]
            for p in range(ploidy):
                src_slot = s * ploidy + p
                lo = src_offsets[src_slot]
                hi = src_offsets[src_slot + 1]
                idx = np.searchsorted(src_data[lo:hi], v)
                if idx < (hi - lo) and src_data[lo + idx] == v:
                    count += 1
        mac_out[k] = count


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_write_var_idxs(
    src_data: NDArray[np.int32],
    src_offsets: NDArray[np.int64],
    src_sample_idxs: NDArray[np.int64],
    ploidy: int,
    kept_var_idxs: NDArray[np.int32],
    new_offsets: NDArray[np.int64],
    out_var_idxs: NDArray[np.int32],
):
    """Pass 2: write remapped variant indices."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):  # type: ignore[not-iterable]
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            out_slot = i * ploidy + p
            wp = new_offsets[out_slot]
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    out_var_idxs[wp] = k
                    wp += 1


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_write_field(
    src_field: NDArray[Any],
    src_data: NDArray[np.int32],
    src_offsets: NDArray[np.int64],
    src_sample_idxs: NDArray[np.int64],
    ploidy: int,
    kept_var_idxs: NDArray[np.int32],
    new_offsets: NDArray[np.int64],
    out_field: NDArray[Any],
):
    """Pass 2 (field variant): writes src_field values at filter-kept positions."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):  # type: ignore[not-iterable]
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            out_slot = i * ploidy + p
            wp = new_offsets[out_slot]
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    out_field[wp] = src_field[j]
                    wp += 1


def _write_filtered_index(src: Path, dst: Path, pl_filter: pl.Expr | None) -> None:
    """Stream a (possibly filtered) genoray index from ``src`` to ``dst``.

    When ``pl_filter`` is None this is byte-equivalent to copying. Otherwise the
    filter is applied lazily; ALT is normalized to list[str] for the filter and
    re-joined to the on-disk comma-Utf8 form so the SVAR index format is
    unchanged. ILEN is computed on-the-fly if absent so ILEN-dependent
    expressions (e.g. ``is_snp``) work correctly, then dropped from the output
    to preserve the original on-disk schema. ILEN is always computed when absent
    from the on-disk schema — even if the filter doesn't reference it — to avoid
    introspecting the opaque Polars expression.
    """
    if pl_filter is None:
        shutil.copy(src, dst)
        return
    lf = pl.scan_ipc(src)
    schema = lf.collect_schema()
    alt_is_utf8 = schema["ALT"] == pl.Utf8
    ilen_added = "ILEN" not in schema
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").str.split(","))
    if ilen_added:
        lf = lf.with_columns(ILEN=ILEN)
    lf = lf.filter(pl_filter)
    if ilen_added:
        lf = lf.drop("ILEN")
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").list.join(","))
    lf.sink_ipc(dst, compression="zstd")


def _subset_var_idxs_and_recompute_af(
    out_path: Path,
    n_total: int,
    n_out: int,
    ploidy: int,
    with_dosages: bool,
) -> tuple[NDArray[V_IDX_TYPE], NDArray[np.float32]]:
    """After concat, drop variants whose MAC across the (already sample-subset)
    output is 0 and remap surviving variant ids to a compacted range.

    A MAC=0 variant contributes no entries to ``variant_idxs.npy`` (it is never
    non-ref in any kept sample/ploidy slot), so dropping it requires only a
    remap of the stored ids — no entries are removed and offsets are unchanged.

    ``with_dosages`` is accepted for call-site symmetry with ``from_vcf`` /
    ``from_pgen``; dosages are stored per-entry and need no remap since no
    entries are dropped.

    Returns ``(survivor_rows, af)`` where ``survivor_rows`` indexes into the
    ``n_total`` candidate rows (use it to subset the index frame) and ``af`` is
    the recomputed allele frequency for each survivor.
    """
    vi_path = out_path / "variant_idxs.npy"
    # If file is missing or empty, treat every candidate as MAC=0 (no entries
    # were ever written), so all variants will be dropped by the check below.
    if not vi_path.exists() or vi_path.stat().st_size == 0:
        mac: NDArray[np.int64] = np.zeros(n_total, dtype=np.int64)
        var_idxs = None
    else:
        var_idxs = np.memmap(vi_path, dtype=V_IDX_TYPE, mode="r+")
        mac = np.bincount(np.asarray(var_idxs, dtype=np.int64), minlength=n_total)
    survivor_mask = mac > 0
    n_surv = int(survivor_mask.sum())
    if n_surv == 0:
        raise ValueError(
            "all selected variants have MAC=0 in the chosen sample subset; "
            "nothing to write"
        )
    n_dropped = n_total - n_surv
    if n_dropped:
        warnings.warn(
            f"dropping {n_dropped} variant(s) with MAC=0 in the output sample set",
            stacklevel=2,
        )
    if var_idxs is not None:
        remap = np.empty(n_total, dtype=V_IDX_TYPE)
        remap[survivor_mask] = np.arange(n_surv, dtype=V_IDX_TYPE)
        # every referenced id survives by construction, so this never hits a gap
        var_idxs[:] = remap[np.asarray(var_idxs, dtype=np.int64)]
        var_idxs.flush()
        del var_idxs

    survivor_rows = np.flatnonzero(survivor_mask).astype(V_IDX_TYPE)
    af = (mac[survivor_mask] / (n_out * ploidy)).astype(np.float32)
    return survivor_rows, af


def _build_working_index(
    src_index_path: Path, pl_filter: pl.Expr | None
) -> tuple[pl.DataFrame, bool, bool]:
    """Load the source index, apply ``pl_filter`` (if any), and return a working
    frame with ALT as list[str], an ILEN list column, and an ``index`` column
    holding each row's position (the SVAR variant id). Also returns
    ``(alt_is_utf8, ilen_added)`` so the on-disk format can be reconstructed.
    """
    lf = pl.scan_ipc(src_index_path)
    schema = lf.collect_schema()
    alt_is_utf8 = schema["ALT"] == pl.Utf8
    ilen_added = "ILEN" not in schema
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").str.split(","))
    if ilen_added:
        lf = lf.with_columns(ILEN=ILEN)
    if pl_filter is not None:
        lf = lf.filter(pl_filter)
    df = lf.collect().with_row_index("index")
    return df, alt_is_utf8, ilen_added


def _write_index_from_working(
    working_df: "pl.DataFrame",
    rows: "NDArray[V_IDX_TYPE]",
    dst: Path,
    alt_is_utf8: bool,
    ilen_added: bool,
    af: "NDArray[np.float32] | None" = None,
) -> None:
    """Write the rows of *working_df* selected by *rows* (in the given order) to
    *dst* in the canonical SVAR on-disk index format: ALT re-joined to comma-Utf8
    if it was originally Utf8, ILEN dropped if we added it, and the helper
    ``index`` column dropped. If *af* is given, (re)sets an ``AF`` column.

    *rows* must be a numpy array of positional row offsets into *working_df*
    (i.e. values from the ``index`` column, which equals row position since
    ``_build_working_index`` calls ``with_row_index`` after any filter).
    """
    frame = working_df[rows.tolist()]
    if af is not None:
        if "AF" in frame.columns:
            frame = frame.drop("AF")
        frame = frame.with_columns(AF=pl.Series(af))
    if ilen_added and "ILEN" in frame.columns:
        frame = frame.drop("ILEN")
    if alt_is_utf8:
        frame = frame.with_columns(pl.col("ALT").list.join(","))
    frame = frame.drop("index")
    frame.write_ipc(dst, compression="zstd")


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
    vcf = VCF(
        path,
        filter=cyvcf2_filter,
        pl_filter=pl_filter,
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


def _open_genos(path: Path, shape: tuple[int | None, ...], mode: Literal["r", "r+"]):
    # Load the memory-mapped files
    var_idxs = np.memmap(path / "variant_idxs.npy", dtype=V_IDX_TYPE, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = Ragged[V_IDX_TYPE].from_offsets(var_idxs, shape, offsets)
    return sp_genos


def _open_fmt(
    name: str,
    type_: NUMERIC | np.dtype[NUMERIC] | type[NUMERIC],
    path: Path,
    shape: tuple[int | None, ...],
    mode: Literal["r", "r+"],
) -> Ragged[NUMERIC]:
    # Load the memory-mapped files
    data = np.memmap(path / f"{name}.npy", dtype=type_, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = Ragged.from_offsets(data, shape, offsets)
    return sp_genos  # type: ignore[bad-return]


def _write_genos(path: Path, sp_genos: Ragged[V_IDX_TYPE]):
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


def _write_dosages(path: Path, dosages: NDArray[DOSAGE_TYPE]):
    path.mkdir(parents=True, exist_ok=True)

    dosages_memmap = np.memmap(
        path / "dosages.npy",
        shape=dosages.shape,
        dtype=dosages.dtype,
        mode="w+",
    )
    dosages_memmap[:] = dosages
    dosages_memmap.flush()


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


@nb.njit(parallel=True, nogil=True, cache=True)
def _copy_chunk_helper(
    out_data: NDArray[DTYPE],
    write_offsets: NDArray[OFFSET_TYPE],
    in_data: NDArray[DTYPE],
    in_offsets: NDArray[OFFSET_TYPE],
    variant_offset: int,
    n_samples: int,
    ploidy: int,
):
    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in range(ploidy):
            sp = s * ploidy + p

            i_s, i_e = in_offsets[sp], in_offsets[sp + 1]
            length = i_e - i_s

            o_s = write_offsets[sp]

            # Copy and add offset
            for i in range(length):
                out_data[o_s + i] = in_data[i_s + i] + variant_offset  # type: ignore

            write_offsets[sp] += length


@nb.njit(parallel=True, nogil=True, cache=True)
def _copy_chunk_dosages_helper(
    out_data: NDArray[DOSAGE_TYPE],
    write_offsets: NDArray[OFFSET_TYPE],
    in_data: NDArray[DOSAGE_TYPE],
    in_offsets: NDArray[OFFSET_TYPE],
    n_samples: int,
    ploidy: int,
):
    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in range(ploidy):
            sp = s * ploidy + p

            i_s, i_e = in_offsets[sp], in_offsets[sp + 1]
            length = i_e - i_s

            o_s = write_offsets[sp]

            out_data[o_s : o_s + length] = in_data[i_s:i_e]

            write_offsets[sp] += length


@nb.njit(parallel=True, nogil=True, cache=True)
def _find_starts_ends(
    genos: NDArray[V_IDX_TYPE],
    geno_offsets: NDArray[OFFSET_TYPE],
    var_ranges: NDArray[V_IDX_TYPE],
    sample_idxs: NDArray[np.int64],
    ploidy: int,
    out_offsets: NDArray[OFFSET_TYPE] | None = None,
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
    sample_idxs
        Sample indices
    ploidy
        Ploidy
    out_offsets
        Output array to write to. If None, a new array will be created.

    Returns
    -------
        Shape: (ranges samples ploidy 2). The first column is the start index of the variant
        and the second column is the end index of the variant.
    """
    n_ranges = len(var_ranges)
    n_samples = len(sample_idxs)
    if out_offsets is None:
        out_offsets = np.empty((2, n_ranges, n_samples, ploidy), dtype=OFFSET_TYPE)
    sorter = np.argsort(var_ranges[:, 0])
    var_ranges = var_ranges[sorter]

    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in nb.prange(ploidy):  # type: ignore[not-iterable]
            s_idx = sample_idxs[s]
            sp = s_idx * ploidy + p
            o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
            sp_genos = genos[o_s:o_e]
            # add o_s to make indices relative to whole array
            out_offsets[..., s, p] = np.searchsorted(sp_genos, var_ranges).T + o_s

    # Ranges with no overlapping variants already get start == stop from
    # searchsorted above (an in-bounds, zero-length range). Do NOT overwrite
    # them with a sentinel: an out-of-range value (e.g. INT64_MAX) is poison for
    # downstream byte-offset math (seqpro Ragged.to_packed multiplies the offset
    # by the element size and overflows int64), even though the row is empty.

    unsorter = np.argsort(sorter)
    out_offsets[:] = out_offsets[:, unsorter]

    return out_offsets


@nb.njit(nogil=True, cache=True)
def _length_walk_n_keep(
    sp_genos: NDArray[V_IDX_TYPE],
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    start_idx: int,
    max_idx: int,
    q_start: POS_TYPE,
    q_end: POS_TYPE,
) -> int:
    """Number of leading variants in ``sp_genos[start_idx:max_idx]`` to include
    so one haplotype reaches ``q_end - q_start`` in length, extending past
    ``q_end`` only as needed. Variants strictly inside ``[q_start, q_end)`` are
    always included; the length budget only gates extension past ``q_end``.
    Returns a count in ``[0, max_idx - start_idx]``."""
    q_len = q_end - q_start
    last_v_end = q_start
    written_len = 0
    for j in range(start_idx, max_idx):
        v_idx = sp_genos[j]
        v_start = v_starts[v_idx]
        ilen = ilens[v_idx]

        maybe_add_one = POS_TYPE(v_start >= q_start)

        if v_start >= q_start:
            past_query = v_start >= q_end
            written_len += v_start - last_v_end
            if past_query and written_len >= q_len:
                return j - start_idx  # exclude this variant
            written_len += max(0, ilen) + maybe_add_one
            if past_query and written_len >= q_len:
                return j - start_idx + 1  # include this variant

        v_end = v_start - min(0, ilen) + maybe_add_one
        last_v_end = max(last_v_end, v_end)  # type: ignore[bad-specialization]

    return max_idx - start_idx


@nb.njit(parallel=True, nogil=True, cache=True)
def _dense2sparse_count(
    genos: NDArray[np.integer],
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    q_start: POS_TYPE,
    q_end: POS_TYPE,
    out_lengths: NDArray[np.int64],
) -> None:
    """Pass 1: per (sample, haplotype), count the carried ALT calls to keep.

    Gathers each haplotype's carried (``== 1``) window-local positions in order
    and routes them through :func:`_length_walk_n_keep` (the SAME walk the sparse
    path uses, so the two cannot drift). Writes the kept count to ``out_lengths``.
    """
    n_samples, ploidy, n_var = genos.shape
    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        carriers = np.empty(n_var, dtype=V_IDX_TYPE)
        for p in range(ploidy):
            nc = 0
            for v in range(n_var):
                if genos[s, p, v] == 1:
                    carriers[nc] = v
                    nc += 1
            out_lengths[s, p] = _length_walk_n_keep(
                carriers, v_starts, ilens, 0, nc, q_start, q_end
            )


@nb.njit(parallel=True, nogil=True, cache=True)
def _dense2sparse_fill(
    genos: NDArray[np.integer],
    var_idxs: NDArray[V_IDX_TYPE],
    dosages: NDArray[DOSAGE_TYPE],
    out_lengths: NDArray[np.int64],
    flat_offsets: NDArray[OFFSET_TYPE],
    out_data: NDArray[V_IDX_TYPE],
    out_dose: NDArray[DOSAGE_TYPE],
    has_dose: bool,
) -> None:
    """Pass 2: emit the first ``out_lengths[s, p]`` carried ALT calls per
    haplotype into the disjoint output range ``[flat_offsets[slot], ...)``."""
    n_samples, ploidy, n_var = genos.shape
    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in range(ploidy):
            slot = s * ploidy + p
            n_keep = out_lengths[s, p]
            w = flat_offsets[slot]
            kept = 0
            for v in range(n_var):
                if kept >= n_keep:
                    break
                if genos[s, p, v] == 1:
                    out_data[w] = var_idxs[v]
                    if has_dose:
                        out_dose[w] = dosages[s, v]
                    w += 1
                    kept += 1


@nb.njit(parallel=False, nogil=True, cache=True)
def _find_starts_ends_with_length(
    genos: NDArray[V_IDX_TYPE],
    geno_offsets: NDArray[OFFSET_TYPE],
    q_starts: NDArray[POS_TYPE],
    q_ends: NDArray[POS_TYPE],
    var_ranges: NDArray[V_IDX_TYPE],
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    sample_idxs: NDArray[np.int64],
    ploidy: int,
    contig_max_idx: int,
    out: NDArray[OFFSET_TYPE] | None = None,
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

    Notes
    -----
    Correctness requires that ``argsort(q_starts) == argsort(var_ranges[:, 0])``,
    i.e. that the per-range query positions and variant-index ranges are
    co-monotone in input order. This holds whenever ``var_ranges`` is derived
    from ``(q_starts, q_ends)`` (e.g. via ``SparseVar.var_ranges``). The
    function sorts ``var_ranges`` internally but indexes ``q_starts`` /
    ``q_ends`` by the same sorted position, so violating this invariant will
    produce results aligned to the wrong query.

    Returns
    -------
        Shape: (2 ranges samples ploidy). The first column is the start index of the variant
        and the second column is the end index of the variant.
    """
    n_ranges = len(q_starts)
    n_samples = len(sample_idxs)
    if out is None:
        out = np.empty((2, n_ranges, n_samples, ploidy), dtype=OFFSET_TYPE)

    sorter = np.argsort(var_ranges[:, 0])
    var_ranges = var_ranges[sorter]

    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in nb.prange(ploidy):  # type: ignore[not-iterable]
            s_idx = sample_idxs[s]
            sp = s_idx * ploidy + p
            o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
            sp_genos = genos[o_s:o_e]

            max_idx = np.searchsorted(sp_genos, contig_max_idx + 1)
            start_idxs = np.searchsorted(sp_genos, var_ranges[:, 0])

            for r in range(n_ranges):
                start_idx: np.intp = start_idxs[r]

                if var_ranges[r, 0] == var_ranges[r, 1]:
                    # No overlapping variants: emit an in-bounds, zero-length
                    # range (start == stop) rather than an INT64_MAX sentinel,
                    # which would overflow downstream byte-offset math even
                    # though the row is empty.
                    out[:, r, s, p] = start_idx + o_s
                    continue

                # add o_s to make indices relative to whole array
                out[0, r, s, p] = start_idx + o_s
                if start_idx == max_idx:
                    # no variants in this range
                    out[1, r, s, p] = start_idx + o_s
                    continue

                n_keep = _length_walk_n_keep(
                    sp_genos,
                    v_starts,
                    ilens,
                    int(start_idx),
                    int(max_idx),
                    q_starts[r],
                    q_ends[r],
                )
                out[1, r, s, p] = start_idx + o_s + n_keep

    unsorter = np.argsort(sorter)
    out[:] = out[:, unsorter]

    return out


def _empty_annot() -> pl.DataFrame:
    """Return an empty annotation DataFrame with the correct schema."""
    return pl.DataFrame(
        {"varID": [], "gene_id": [], "strand": [], "codon_pos": []},
        schema={
            "varID": pl.UInt32,
            "gene_id": pl.Utf8,
            "strand": pl.Utf8,
            "codon_pos": pl.Int8,
        },
    )


def _get_strand_and_codon_pos(
    cds_df: pl.DataFrame, var_table: pl.DataFrame, contig_normalizer: ContigNormalizer
) -> pl.DataFrame:
    """
    Calculate strand and codon position for variants overlapping CDS regions.

    Parameters
    ----------
    cds_df : pl.DataFrame
        CDS features from GTF with columns: chrom, start, end, strand, frame,
        gene_id, transcript_id, gene_biotype, transcript_support_level, tag
        coordinates should be 1-based
    var_table : pl.DataFrame
        Variant table with columns: index, CHROM, POS, ILEN, ...
        POS should be 1-based
    contig_normalizer : ContigNormalizer
        Normalizer to match chromosome names between CDS and granges


    Returns
    -------
    pl.DataFrame
        Annotation with varID, gene_id, strand, codon_pos
    """

    # Normalize CDS chromosome names to match granges
    # Cast to string first to avoid categorical comparison issues
    cds_df = cds_df.with_columns(
        pl.col("chrom").cast(pl.Utf8).replace(contig_normalizer.contig_map)
    )

    # Filter out CDS features with chromosomes not in granges
    cds_df = cds_df.filter(pl.col("chrom").is_in(contig_normalizer.contigs))
    cds_df.config_meta.set(coordinate_system_zero_based=False)  # type: ignore

    # Prepare var_table for pb.overlap by creating interval columns
    var_intervals = var_table.select(
        pl.col("ILEN").list.first(),
        var_id="index",
        chrom="CHROM",
        start=pl.col("POS"),
        end=pl.col("POS")
        - pl.col("ILEN").list.first().clip(upper_bound=0).fill_null(0),
    )
    var_intervals.config_meta.set(coordinate_system_zero_based=False)  # type: ignore

    # Check if CDS or var_table is empty
    if cds_df.is_empty() or var_table.is_empty():
        return _empty_annot()

    joined_cds = (
        cast(
            pl.LazyFrame,
            pb.overlap(var_intervals, cds_df, projection_pushdown=True),
        )
        .rename(
            {
                "start_1": "pos",
                "start_2": "cds_start",
                "end_2": "cds_end",
            }
        )
        .drop("end_1", "chrom_1", "chrom_2")
        .rename(lambda c: c.replace("_2", "").replace("_1", ""))
        .collect()
    )

    if joined_cds.height == 0:
        return _empty_annot()

    annot = (
        joined_cds
        # Positive strand: (rel_pos - frame) % 3
        # Negative strand: (2 * (rel_pos - frame)) % 3 (reverse complement pattern)
        .with_columns(
            pl.when(
                pl.col("frame").is_not_null()
                & (pl.col("frame") <= 2)
                & (pl.col("ILEN") == 0)
            )
            .then(
                pl.when(pl.col("strand") == "+")
                .then((pl.col("pos") - pl.col("cds_start") - pl.col("frame")) % 3)
                .otherwise(
                    (2 * (pl.col("pos") - pl.col("cds_start") - pl.col("frame"))) % 3
                )
            )
            .cast(pl.Int8)
            .alias("codon_pos")
        )
        # Get the gene_id, strand, and codon_pos.
        # If there are any duplicates, choose the one with the best rank, breaking ties by choosing the first seen.
        .with_columns(
            # Rank 0 is best, higher ranks are worse
            pl.when(pl.col("gene_biotype") == "protein_coding")
            .then(0)
            .otherwise(1)
            .alias("rank_pc"),
            pl.when(
                pl.col("tag").is_not_null()
                & pl.col("tag").str.contains(r"^(canonical|appris_principal)")
            )
            .then(0)
            .otherwise(1)
            .alias("rank_canonical"),
            pl.when(pl.col("transcript_support_level").is_not_null())
            .then(
                pl.col("transcript_support_level")
                .str.extract(r"(\d+)", 1)
                .cast(pl.Int16, strict=False)
            )
            .otherwise(9999)
            .alias("rank_tsl"),
            # Negative span so larger spans get rank 0 (best)
            -(pl.col("cds_end") - pl.col("cds_start")).alias("rank_span"),
        )
        .sort(
            [
                "var_id",
                "rank_pc",
                "rank_canonical",
                "rank_tsl",
                "rank_span",
                "transcript_id",
            ],
            descending=[
                False,
                False,
                False,
                False,
                False,
                False,
            ],  # kept this for code clarity (default also the same)
        )
        .group_by("var_id", maintain_order=True)
        .agg(pl.col("gene_id", "strand", "codon_pos").first())
        # Match the column name used by _empty_annot() and the write-back join.
        .rename({"var_id": "varID"})
    )

    return annot


def _load_gtf(gtf: str | pl.DataFrame) -> pl.DataFrame:
    """Load GTF file as a 1-based polars DataFrame."""
    if isinstance(gtf, pl.DataFrame):
        return gtf.rename({"seqname": "chrom"}, strict=False)

    return (
        sp.gtf.scan(str(gtf))
        .with_columns(
            sp.gtf.attr("gene_id"),
            sp.gtf.attr("transcript_id"),
            sp.gtf.attr("gene_name"),
            sp.gtf.attr("gene_biotype"),
            sp.gtf.attr("transcript_support_level"),
            sp.gtf.attr("level"),
            sp.gtf.attr("tag"),
        )
        .collect()
        .rename({"seqname": "chrom"}, strict=False)
    )

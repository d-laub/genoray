from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import seqpro as sp
from numpy.typing import NDArray

from .._types import V_IDX_TYPE
from .._utils import ContigNormalizer
from .._var_ranges import var_ranges

if TYPE_CHECKING:
    from ._core import SparseVar

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

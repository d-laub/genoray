from __future__ import annotations

from typing import cast

import numba as nb
import numpy as np
import polars as pl
import polars_bio as pb
import polars_config_meta  # noqa: F401
from numpy.typing import ArrayLike, NDArray
from seqpro.rag import OFFSET_TYPE, lengths_to_offsets

from ._types import POS_TYPE, V_IDX_TYPE
from ._utils import DTYPE, ContigNormalizer, np_to_pl_dtype


def var_ranges(
    contig_normalizer: ContigNormalizer,
    var_table: pl.DataFrame,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
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
    starts = np.atleast_1d(np.asarray(starts, POS_TYPE)).clip(min=0)
    n_ranges = len(starts)

    c = contig_normalizer.norm(contig)
    if c is None:
        return np.full((n_ranges, 2), np.iinfo(V_IDX_TYPE).max, V_IDX_TYPE)

    ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

    var_table = var_table.filter(pl.col("CHROM") == c)
    n_vars = var_table.height

    if n_vars == 0 or n_ranges == 0:
        return np.full((n_ranges, 2), np.iinfo(V_IDX_TYPE).max, V_IDX_TYPE)

    # 0-based
    v_starts = var_table["POS"].to_numpy() - 1
    # 0-based, exclusive end
    v_ends = (
        var_table["POS"] - var_table["ILEN"].list.first().clip(upper_bound=0)
    ).to_numpy()
    max_v_len = (v_ends - v_starts).max()

    lower_bound_s_idx = np.searchsorted(v_starts + max_v_len, starts)
    upper_bound_e_idx = np.searchsorted(v_starts, ends)

    # Find first overlapping (q_start < v_end)
    s_idx = _forward_sub_scan(v_ends, lower_bound_s_idx, upper_bound_e_idx, starts)
    # Find last overlapping (q_start < v_end), returns exclusive end
    # Needed when there are no variants after the first overlapping that should be included.
    # Example: q = [2, 3)
    # ╔═════════╦═══════╦═════════════════╦═════════════════╗
    # ║ v_start ║ v_end ║ v_start < q_end ║ q_start < v_end ║
    # ╠═════════╬═══════╬═════════════════╬═════════════════╣
    # ║    1    ║   3   ║        Y        ║        Y        ║
    # ╠═════════╬═══════╬═════════════════╬═════════════════╣
    # ║    1    ║   2   ║        Y        ║        N        ║
    # ╠═════════╬═══════╬═════════════════╬═════════════════╣
    # ║    3    ║   4   ║        N        ║        Y        ║
    # ╚═════════╩═══════╩═════════════════╩═════════════════╝
    e_idx = _backward_sub_scan(v_ends, s_idx, upper_bound_e_idx, starts)
    no_vars = s_idx == n_vars
    s_idx[no_vars] = 0
    e_idx[no_vars] = 0
    s_idx = var_table[s_idx, "index"].to_numpy()
    e_idx = var_table[e_idx, "index"].to_numpy() + 1  # exclusive end

    var_ranges = np.stack([s_idx, e_idx], axis=1, dtype=V_IDX_TYPE)
    var_ranges[no_vars] = np.iinfo(V_IDX_TYPE).max

    return var_ranges


def var_indices(
    idx_dtype: type[DTYPE],
    contig_normalizer: ContigNormalizer,
    var_table: pl.DataFrame | pl.LazyFrame,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
) -> tuple[NDArray[DTYPE], NDArray[OFFSET_TYPE]]:
    """Get variant indices for each query range."""
    starts = np.atleast_1d(np.asarray(starts, POS_TYPE)).clip(min=0)
    n_ranges = len(starts)
    if n_ranges == 0:
        return np.empty(0, idx_dtype), np.zeros(n_ranges + 1, OFFSET_TYPE)

    c = contig_normalizer.norm(contig)
    if c is None:
        return np.empty(0, idx_dtype), np.zeros(n_ranges + 1, OFFSET_TYPE)

    ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

    var_table = (
        var_table.lazy()
        .filter(pl.col("CHROM") == c)
        .select(
            pl.col("index").cast(np_to_pl_dtype(idx_dtype)),
            chrom=pl.col("CHROM").cast(pl.Utf8),
            start=pl.col("POS") - 1,
            end=pl.col("POS") - pl.col("ILEN").list.first().clip(upper_bound=0),
        )
    )
    var_table.config_meta.set(coordinate_system_zero_based=True)  # type: ignore

    queries = (
        pl.DataFrame({"chrom": c, "start": starts, "end": ends})
        .lazy()
        .with_row_index("query")
    )
    queries.config_meta.set(coordinate_system_zero_based=True)  # type: ignore

    join = (
        cast(pl.LazyFrame, pb.overlap(queries, var_table, projection_pushdown=True))
        .sort("query_1", "index_2")
        .collect()
    )

    if join.height == 0:
        return np.empty(0, idx_dtype), np.zeros(n_ranges + 1, OFFSET_TYPE)

    idxs = join["index_2"].to_numpy()
    lens = join.group_by("query_1", maintain_order=True).len()["len"].to_numpy()
    offsets = lengths_to_offsets(lens)
    return idxs, offsets


def var_counts(
    contig_normalizer: ContigNormalizer,
    var_table: pl.DataFrame | pl.LazyFrame,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
) -> NDArray[np.uint32]:
    starts = np.atleast_1d(np.asarray(starts, POS_TYPE)).clip(min=0)
    n_ranges = len(starts)
    if n_ranges == 0:
        return np.zeros(n_ranges, dtype=np.uint32)

    c = contig_normalizer.norm(contig)
    if c is None:
        return np.zeros(n_ranges, dtype=np.uint32)

    ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

    var_table = (
        var_table.lazy()
        .filter(pl.col("CHROM") == c)
        .select(
            chrom=pl.col("CHROM").cast(pl.Utf8),
            start=pl.col("POS") - 1,
            end=pl.col("POS") - pl.col("ILEN").list.first().clip(upper_bound=0),
        )
    )
    var_table.config_meta.set(coordinate_system_zero_based=True)  # type: ignore

    queries = (
        pl.DataFrame({"chrom": c, "start": starts, "end": ends})
        .lazy()
        .with_row_index("query")
    )
    queries.config_meta.set(coordinate_system_zero_based=True)  # type: ignore

    counts = (
        cast(pl.LazyFrame, pb.overlap(queries, var_table, projection_pushdown=True))
        .group_by("query_1", maintain_order=True)
        .len()
        .with_columns(pl.col("len").cast(pl.UInt32))
        .collect()
    )

    if counts.height == 0:
        return np.zeros(n_ranges, dtype=np.uint32)

    return counts["len"].to_numpy()


@nb.guvectorize(
    [(nb.int_[:], nb.int_, nb.int_, nb.int_, nb.int_[:])],
    "(n),(),(),()->()",
)
def _forward_sub_scan(
    v_ends: NDArray[np.integer],
    lower_bound: int | np.integer | NDArray[np.integer],
    upper_bound: int | np.integer | NDArray[np.integer],
    q_start: int | np.integer | NDArray[np.integer],
    indices: NDArray[np.integer] = None,  # type: ignore
) -> NDArray[np.integer]:  # type: ignore
    """Find first index where q_start < v_ends[i] (forward scan)."""
    for i in range(lower_bound, upper_bound):
        if q_start < v_ends[i]:
            indices[0] = i
            break
    else:
        indices[0] = upper_bound


@nb.guvectorize(
    [(nb.int_[:], nb.int_, nb.int_, nb.int_, nb.int_[:])],
    "(n),(),(),()->()",
)
def _backward_sub_scan(
    v_ends: NDArray[np.integer],
    lower_bound: int | np.integer | NDArray[np.integer],
    upper_bound: int | np.integer | NDArray[np.integer],
    q_start: int | np.integer | NDArray[np.integer],
    indices: NDArray[np.integer] = None,  # type: ignore
) -> NDArray[np.integer]:  # type: ignore
    """Find last index where q_start < v_ends[i] (backward scan)."""
    for i in range(upper_bound - 1, lower_bound - 1, -1):
        if q_start < v_ends[i]:
            indices[0] = i
            break
    else:
        indices[0] = lower_bound  # no overlap found

from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from ._types import INT64_MAX, POS_TYPE, V_IDX_TYPE
from ._utils import ContigNormalizer


def var_ranges(
    contig_normalizer: ContigNormalizer,
    var_table: pl.DataFrame,
    contig: str,
    starts: ArrayLike = 0,
    ends: ArrayLike = INT64_MAX,
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
    starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
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
    s_idx = _forward_sub_scan(
        v_ends, lower_bound_s_idx, upper_bound_e_idx, max_v_len, starts
    )
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
    e_idx = _backward_sub_scan(v_ends, s_idx, upper_bound_e_idx, max_v_len, starts)

    var_ranges = np.stack([s_idx, e_idx], axis=1, dtype=V_IDX_TYPE)
    var_ranges[s_idx >= e_idx] = np.iinfo(V_IDX_TYPE).max

    return var_ranges


@nb.guvectorize(
    [(nb.int_[:], nb.int_, nb.int_, nb.int_, nb.int_, nb.int_[:])],
    "(n),(),(),(),()->()",
)
def _forward_sub_scan(
    v_ends: NDArray[np.integer],
    lower_bound: int | np.integer | NDArray[np.integer],
    upper_bound: int | np.integer | NDArray[np.integer],
    max_v_len: int | np.integer | NDArray[np.integer],
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
    [(nb.int_[:], nb.int_, nb.int_, nb.int_, nb.int_, nb.int_[:])],
    "(n),(),(),(),()->()",
)
def _backward_sub_scan(
    v_ends: NDArray[np.integer],
    lower_bound: int | np.integer | NDArray[np.integer],
    upper_bound: int | np.integer | NDArray[np.integer],
    max_v_len: int | np.integer | NDArray[np.integer],
    q_start: int | np.integer | NDArray[np.integer],
    indices: NDArray[np.integer] = None,  # type: ignore
) -> NDArray[np.integer]:  # type: ignore
    """Find last index where q_start < v_ends[i] (backward scan), returns exclusive end."""
    for i in range(upper_bound - 1, lower_bound - 1, -1):
        if q_start < v_ends[i]:
            indices[0] = i + 1  # exclusive end
            break
    else:
        indices[0] = lower_bound  # no overlap found

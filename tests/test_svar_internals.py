"""Unit tests for low-level _svar numba helpers.

Regression tests for two bugs surfaced by GenVarLoader's 1KG bcftools-consensus
parity test (see BUGS-FROM-GENVARLOADER-PARITY-TEST.md):

- Bug 1: `sorter[sorter]` was used as an inverse permutation in
  `_find_starts_ends` and `_find_starts_ends_with_length`. It is only the
  inverse when `sorter` is an involution; for generic permutations it gives
  the wrong reordering.
- Bug 2: `_find_starts_ends_with_length` exited the per-range walk early when
  prior insertions filled the length budget, dropping variants strictly inside
  the query range. The length budget should only gate extension past `q_end`.
"""

from __future__ import annotations

import numpy as np
from seqpro.rag import OFFSET_TYPE

from genoray._svar import _find_starts_ends, _find_starts_ends_with_length
from genoray._types import POS_TYPE, V_IDX_TYPE


def test_argsort_is_correct_unsorter():
    """`np.argsort(sorter)` inverts `sorter`; `sorter[sorter]` generally does not."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        arr = rng.integers(0, 1_000_000, size=10)
        sorter = np.argsort(arr)
        unsorter = np.argsort(sorter)
        assert np.array_equal(arr[sorter][unsorter], arr)


def test_find_starts_ends_unsorted_ranges():
    """Result for unsorted ranges must match per-range result (Bug 1 regression)."""
    # One sample, ploidy=1. Variant indices 0..9 all assigned to this sample.
    n_vars = 10
    genos = np.arange(n_vars, dtype=V_IDX_TYPE)
    geno_offsets = np.array([0, n_vars], dtype=OFFSET_TYPE)
    sample_idxs = np.array([0], dtype=np.int64)

    # 10 variant-index ranges in an order that is NOT an involution.
    var_ranges = np.array(
        [
            [7, 9],
            [0, 2],
            [5, 7],
            [2, 3],
            [9, 10],
            [3, 5],
            [4, 6],
            [1, 4],
            [6, 8],
            [8, 10],
        ],
        dtype=V_IDX_TYPE,
    )

    out = _find_starts_ends(genos, geno_offsets, var_ranges, sample_idxs, 1)
    # out shape: (2, n_ranges, n_samples=1, ploidy=1)
    starts = out[0, :, 0, 0]
    ends = out[1, :, 0, 0]

    # Since genos == np.arange(n_vars), searchsorted gives the var indices back.
    np.testing.assert_array_equal(starts, var_ranges[:, 0])
    np.testing.assert_array_equal(ends, var_ranges[:, 1])


def test_find_starts_ends_with_length_includes_variant_at_query_edge():
    """Bug 2 regression: variants inside [q_start, q_end) must be included even
    after prior insertions saturate the alt-side length budget.
    """
    # Single sample, ploidy=1.
    # 4 variants on one contig: three +10 insertions clustered at the start,
    # then a SNP at the right edge of the query.
    v_starts = np.array([0, 1, 2, 100], dtype=np.int32)
    ilens = np.array([10, 10, 5, 0], dtype=np.int32)
    n_vars = len(v_starts)

    genos = np.arange(n_vars, dtype=V_IDX_TYPE)
    geno_offsets = np.array([0, n_vars], dtype=OFFSET_TYPE)
    sample_idxs = np.array([0], dtype=np.int64)

    # One query covering all four variants.
    q_starts = np.array([0], dtype=POS_TYPE)
    q_ends = np.array([101], dtype=POS_TYPE)
    var_ranges = np.array([[0, n_vars]], dtype=V_IDX_TYPE)

    out = _find_starts_ends_with_length(
        genos,
        geno_offsets,
        q_starts,
        q_ends,
        var_ranges,
        v_starts,
        ilens,
        sample_idxs,
        1,
        int(v_starts[-1]) + 1,
    )
    start = int(out[0, 0, 0, 0])
    end = int(out[1, 0, 0, 0])

    # All four variants must be in the returned slice.
    assert start == 0
    assert end == n_vars, (
        f"SNP at q_end-1 was dropped due to early length-budget exit: "
        f"returned slice [{start}, {end}), expected [0, {n_vars})"
    )

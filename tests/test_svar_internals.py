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

from genoray._svar._kernels import _find_starts_ends, _find_starts_ends_with_length
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


def test_with_length_large_deletion_extends_multiple_variants():
    """A large deletion at the query start forces the walk to extend across
    multiple downstream variants to satisfy the length budget.

    Verified by running _length_walk_n_keep and _find_starts_ends_with_length
    in a scratch session: n_keep=5, kept=5 (deletion + all 4 downstream SNPs).
    """
    # -10 deletion at pos 999; reference span [999, 1010).
    # Downstream SNPs are placed at pos >= 1010 so none sit inside the
    # deletion's deleted span on the same haplotype (avoids degenerate inputs).
    v_starts = np.array([999, 1010, 1012, 1014, 1016], dtype=np.int32)
    ilens = np.array([-10, 0, 0, 0, 0], dtype=np.int32)
    n_vars = len(v_starts)
    genos = np.arange(n_vars, dtype=V_IDX_TYPE)  # single hap carries all
    geno_offsets = np.array([0, n_vars], dtype=OFFSET_TYPE)
    sample_idxs = np.array([0], dtype=np.int64)

    # Query covers exactly the deletion's reference span (length 11).
    q_starts = np.array([999], dtype=POS_TYPE)
    q_ends = np.array([1010], dtype=POS_TYPE)
    # var_ranges covers only the deletion itself at the contig level;
    # the length walk extends beyond it.
    var_ranges = np.array([[0, 1]], dtype=V_IDX_TYPE)

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
        n_vars,
    )
    start = int(out[0, 0, 0, 0])
    end = int(out[1, 0, 0, 0])

    # The deletion shrinks the alternate-allele length, so the walk must pull
    # in downstream SNPs to compensate. Observed: kept=5 (all 5 variants).
    assert start == 0
    assert end - start > 1, (
        f"deletion did not force multi-variant extension: [{start}, {end})"
    )


def test_with_length_per_haplotype_divergence():
    """A deletion carried by one haplotype but not the other causes the
    deletion-carrying haplotype to keep more variants than the other.

    Setup (ploidy=2, 1 sample):
    - Global vars:
        0: pos=100, ilen=-5 (deletion; ref span [100, 106))
        1: pos=106, ilen=0  (SNP, at deletion's ref end -- no overlap)
        2: pos=108, ilen=0  (SNP)
        3: pos=110, ilen=0  (SNP, just past query end)
        4: pos=101, ilen=0  (SNP for hapB)
        5: pos=103, ilen=0  (SNP for hapB)
        6: pos=109, ilen=0  (SNP for hapB)
    - hapA (p=0) carries [0, 1, 2, 3]; hapB (p=1) carries [4, 5, 6].
    - Query [100, 110): length 10.

    Verified by scratch run:
    - hapA kept=4 (deletion + 3 SNPs needed to cover length 10)
    - hapB kept=3 (all 3 non-deletion SNPs)
    hapA > hapB because the deletion reduces the alternate-allele contribution,
    forcing the walk to extend further.
    """
    v_starts = np.array([100, 106, 108, 110, 101, 103, 109], dtype=np.int32)
    ilens = np.array([-5, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    n_vars = len(v_starts)

    # hapA carries global indices [0,1,2,3]; hapB carries [4,5,6].
    # Sparse storage slots: sample 0, p=0 -> genos[0:4]; p=1 -> genos[4:7].
    genos = np.array([0, 1, 2, 3, 4, 5, 6], dtype=V_IDX_TYPE)
    geno_offsets = np.array([0, 4, 7], dtype=OFFSET_TYPE)
    sample_idxs = np.array([0], dtype=np.int64)

    q_starts = np.array([100], dtype=POS_TYPE)
    q_ends = np.array([110], dtype=POS_TYPE)
    # var_ranges spans all global variant indices so the walk can see all vars.
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
        2,
        n_vars,
    )
    # out shape: (2, n_ranges=1, n_samples=1, ploidy=2)
    kept_A = int(out[1, 0, 0, 0]) - int(out[0, 0, 0, 0])
    kept_B = int(out[1, 0, 0, 1]) - int(out[0, 0, 0, 1])

    # Verified observed values: hapA=4, hapB=3.
    assert kept_A == 4, f"hapA expected 4 kept, got {kept_A}"
    assert kept_B == 3, f"hapB expected 3 kept, got {kept_B}"
    assert kept_A > kept_B, (
        f"deletion hap should keep more variants than non-deletion hap; "
        f"got hapA={kept_A}, hapB={kept_B}"
    )


def test_with_length_contig_end_clamp():
    """When a lone carried deletion is the only variant on the contig and the
    query length far exceeds what it can fill, the function returns exactly
    that one variant and does not crash.

    Verified by scratch run: n_keep=1, kept=1.
    """
    # Single -50 deletion at pos 1000; query [1000, 2000) length 1000 cannot
    # be satisfied by one variant, but contig_max_idx=1 prevents extension.
    v_starts = np.array([1000], dtype=np.int32)
    ilens = np.array([-50], dtype=np.int32)
    n_vars = 1

    genos = np.array([0], dtype=V_IDX_TYPE)
    geno_offsets = np.array([0, 1], dtype=OFFSET_TYPE)
    sample_idxs = np.array([0], dtype=np.int64)

    q_starts = np.array([1000], dtype=POS_TYPE)
    q_ends = np.array([2000], dtype=POS_TYPE)
    var_ranges = np.array([[0, 1]], dtype=V_IDX_TYPE)
    contig_max_idx = n_vars  # exclusive upper bound; index 0 is the only valid var

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
        contig_max_idx,
    )
    start = int(out[0, 0, 0, 0])
    end = int(out[1, 0, 0, 0])

    # Despite the large query, only the one available variant is returned.
    # Observed: kept=1.
    assert end - start == 1, (
        f"clamp should return exactly 1 variant; got [{start}, {end})"
    )

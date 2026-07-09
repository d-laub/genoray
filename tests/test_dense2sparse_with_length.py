"""Unit tests for _dense2sparse_with_length (the dense->sparse parity bridge)."""

from __future__ import annotations

import numpy as np
from seqpro.rag import OFFSET_TYPE

from genoray._svar._convert import _dense2sparse_with_length
from genoray._types import V_IDX_TYPE


def test_no_indels_keeps_all_carriers_within_query():
    # 1 sample, ploidy 1, window of 3 SNPs all inside the query, all carried.
    genos = np.array([[[1, 1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1, 2], dtype=V_IDX_TYPE)
    v_starts = np.array([10, 11, 12], dtype=np.int32)
    ilens = np.array([0, 0, 0], dtype=np.int32)
    rag = _dense2sparse_with_length(genos, var_idxs, 10, 13, v_starts, ilens)
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3], dtype=OFFSET_TYPE))
    np.testing.assert_array_equal(rag.data, np.array([0, 1, 2], dtype=V_IDX_TYPE))


def test_deletion_forces_extension_past_query():
    # 1 sample, ploidy 1. A -2 deletion at the query start spans ref [10, 13),
    # i.e. the whole query. To recover the lost length the walk must extend past
    # q_end (=13) into the variants at 13 and 14. The variant at 15 is not
    # needed. (Inputs are non-overlapping: the deletion's downstream neighbors
    # sit at/after its reference end, not inside its deleted span.)
    genos = np.array([[[1, 1, 1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1, 2, 3], dtype=V_IDX_TYPE)
    v_starts = np.array([10, 13, 14, 15], dtype=np.int32)
    ilens = np.array([-2, 0, 0, 0], dtype=np.int32)  # deletion first
    rag = _dense2sparse_with_length(genos, var_idxs, 10, 13, v_starts, ilens)
    # keeps the deletion + the two extension variants past q_end (idx 0,1,2)
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3], dtype=OFFSET_TYPE))
    np.testing.assert_array_equal(rag.data, np.array([0, 1, 2], dtype=V_IDX_TYPE))


def test_per_haplotype_independent_trim():
    # 1 sample, ploidy 2. Query [10, 15) (len 5). Window: a -2 deletion at 10
    # (spans ref [10, 13)) followed by SNPs at 13,14,15,16,17.
    # hapA carries the deletion + all SNPs -> must extend further to recover the
    # deleted length. hapB carries only the SNPs (no deletion) -> reaches length
    # with far fewer variants. The two haplotypes therefore trim independently.
    genosA = [1, 1, 1, 1, 1, 1]
    genosB = [0, 1, 1, 1, 1, 1]
    genos = np.array([[genosA, genosB]], dtype=np.int8)
    var_idxs = np.array([0, 1, 2, 3, 4, 5], dtype=V_IDX_TYPE)
    v_starts = np.array([10, 13, 14, 15, 16, 17], dtype=np.int32)
    ilens = np.array([-2, 0, 0, 0, 0, 0], dtype=np.int32)
    rag = _dense2sparse_with_length(genos, var_idxs, 10, 15, v_starts, ilens)
    # hapA keeps 5 (deletion + 4 SNPs: idx 0,1,2,3,4); hapB keeps 2 (idx 1,2)
    np.testing.assert_array_equal(rag.offsets, np.array([0, 5, 7], dtype=OFFSET_TYPE))
    np.testing.assert_array_equal(
        rag.data, np.array([0, 1, 2, 3, 4, 1, 2], dtype=V_IDX_TYPE)
    )


def test_dosages_follow_genotypes():
    genos = np.array([[[1, 1]]], dtype=np.int8)
    var_idxs = np.array([5, 6], dtype=V_IDX_TYPE)
    v_starts = np.array([10, 11], dtype=np.int32)
    ilens = np.array([0, 0], dtype=np.int32)
    dosages = np.array([[0.5, 0.9]], dtype=np.float32)  # (s v)
    rag, drag = _dense2sparse_with_length(
        genos, var_idxs, 10, 12, v_starts, ilens, dosages
    )
    np.testing.assert_array_equal(rag.data, np.array([5, 6], dtype=V_IDX_TYPE))
    np.testing.assert_allclose(drag.data, np.array([0.5, 0.9], dtype=np.float32))


def test_clamp_keeps_what_is_available():
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=V_IDX_TYPE)
    v_starts = np.array([10], dtype=np.int32)
    ilens = np.array([-5], dtype=np.int32)
    rag = _dense2sparse_with_length(genos, var_idxs, 10, 30, v_starts, ilens)
    np.testing.assert_array_equal(rag.offsets, np.array([0, 1], dtype=OFFSET_TYPE))
    np.testing.assert_array_equal(rag.data, np.array([0], dtype=V_IDX_TYPE))

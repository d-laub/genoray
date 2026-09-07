"""`SparseVar2._sample_idxs` semantics and complexity (issue #168).

The name -> column lookup used to be a pair of linear scans (`in` plus
`list.index`) per requested sample, i.e. O(S^2) for a caller that asks for the
whole cohort -- which GenVarLoader's SVAR2 writer always does. At S = 535,662
that is ~1 h of pure string comparison before a single byte is read. These tests
pin both the exact semantics (so the fix stays a pure speedup) and the
complexity (so it cannot silently regress).
"""

from __future__ import annotations

import numpy as np
import pytest

from genoray import SparseVar2


def test_none_selects_everything(svar2_store):
    assert SparseVar2(svar2_store)._sample_idxs(None) is None


def test_full_cohort_in_order(svar2_store):
    sv = SparseVar2(svar2_store)
    assert sv._sample_idxs(sv.available_samples) == [0, 1]


def test_subset(svar2_store):
    assert SparseVar2(svar2_store)._sample_idxs(["S1"]) == [1]


def test_reordered(svar2_store):
    assert SparseVar2(svar2_store)._sample_idxs(["S1", "S0"]) == [1, 0]


def test_duplicates_are_preserved_not_deduped(svar2_store):
    # Unlike `_normalize_samples`, this resolver keeps repeats: a duplicated
    # name means a duplicated output column.
    assert SparseVar2(svar2_store)._sample_idxs(["S1", "S1", "S0"]) == [1, 1, 0]


def test_scalar_string_is_one_name_not_characters(svar2_store):
    assert SparseVar2(svar2_store)._sample_idxs("S0") == [0]


def test_ndarray_input(svar2_store):
    sv = SparseVar2(svar2_store)
    assert sv._sample_idxs(np.array(["S1", "S0"])) == [1, 0]


def test_missing_sample_raises_value_error(svar2_store):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="Sample 'nope' not found in the dataset."):
        sv._sample_idxs(["S0", "nope"])


def test_missing_sample_longer_than_stored_names(svar2_store):
    # A name wider than the store's fixed-width name dtype must miss cleanly,
    # not truncate into a false match or blow up inside the hash table.
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="not found in the dataset."):
        sv._sample_idxs(["S0_but_very_much_longer"])


def test_first_occurrence_index_wins_on_duplicate_store_names(svar2_store):
    sv = SparseVar2(svar2_store)
    sv.available_samples = ["dup", "dup", "other"]
    assert sv._sample_idxs(["dup", "other"]) == [0, 2]


class _CountingList(list):
    """A sample list that records how many times it is swept end to end."""

    def __init__(self, it) -> None:
        super().__init__(it)
        self.sweeps = 0

    def __iter__(self):
        self.sweeps += 1
        return super().__iter__()


def test_cohort_is_swept_once_across_calls(svar2_store):
    """The name -> index lookup must be built by one sweep and then reused.

    This is the issue #168 regression gate, asserted structurally rather than by
    wall clock so it cannot flake on a loaded node. Two failure modes both trip
    it:

    * the original `in` + `.index()` pair -- both are C-level list operations
      that never call `__iter__`, so `sweeps` stays 0;
    * rebuilding the lookup on every query -- `sweeps` becomes 2.

    Only a lookup built once and cached leaves exactly one sweep.
    """
    n = 10_000
    names = [f"S{i}" for i in range(n)]
    sv = SparseVar2(svar2_store)
    sv.available_samples = _CountingList(names)

    assert sv._sample_idxs(names) == list(range(n))
    assert sv._sample_idxs(names[::-1]) == list(range(n - 1, -1, -1))

    assert sv.available_samples.sweeps == 1

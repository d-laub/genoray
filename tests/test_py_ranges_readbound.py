"""Task 5: `find_ranges` dict exposes the per-class dense ranges (`dense_snp_range`,
`dense_indel_range`) that the read-bound gather (Task 3) and gvl's write-side cache
(Plan 2, Task 2) consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from genoray import SparseVar2

NEW_KEYS = ("dense_snp_range", "dense_indel_range")


def test_find_ranges_exposes_dense_class_ranges(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    d = sv._find_ranges("chr1", starts, ends)

    assert "dense_snp_range" in d and "dense_indel_range" in d

    dense_range = np.asarray(d["dense_range"])
    n_regions = int(d["n_regions"])
    for k in NEW_KEYS:
        a = np.asarray(d[k])
        assert a.ndim == 2 and a.shape[1] == 2
        assert a.shape[0] == n_regions
        assert a.dtype == dense_range.dtype
        assert (a[:, 0] <= a[:, 1]).all()


def test_find_ranges_out_streaming_includes_dense_class_ranges(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    d = sv._find_ranges("chr1", starts, ends)

    out = {k: np.empty_like(np.asarray(d[k])) for k in NEW_KEYS}
    d2 = sv._find_ranges("chr1", starts, ends, out=out)

    for k in NEW_KEYS:
        np.testing.assert_array_equal(np.asarray(d2[k]), np.asarray(d[k]))
        assert np.asarray(d2[k]).base is out[k] or d2[k] is out[k]


def test_gather_ranges_roundtrips_with_dense_class_ranges(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    # Should not raise: dense_snp_range/dense_indel_range round-trip through
    # bundle_from_dict without breaking the union gather.
    sv._gather_ranges("chr1", sv._find_ranges("chr1", [0], [40]))

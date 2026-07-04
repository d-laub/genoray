from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from genoray import SparseVar2


def _assert_dicts_equal(a: dict[str, Any], b: dict[str, Any], keys: Iterable[str]):
    for k in keys:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


PAYLOAD_KEYS = [
    "vk_pos",
    "vk_key",
    "vk_off",
    "dense_pos",
    "dense_key",
    "dense_range",
    "dense_present",
    "dense_present_off",
    "lut_bytes",
    "lut_off",
]


def test_read_ranges_matches_overlap_batch(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    ob = sv.overlap_batch("chr1", list(zip(starts, ends)))
    rr = sv.read_ranges("chr1", starts, ends)
    _assert_dicts_equal(ob, rr, PAYLOAD_KEYS)
    assert int(rr["n_regions"]) == 2


def test_gather_of_find_matches_read(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    starts, ends = [0], [40]
    ranges = sv.find_ranges("chr1", starts, ends)
    gathered = sv.gather_ranges("chr1", ranges)
    read = sv.read_ranges("chr1", starts, ends)
    _assert_dicts_equal(read, gathered, PAYLOAD_KEYS)


def test_read_ranges_sample_subset(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    full = sv.overlap_batch("chr1", [(0, 40)])
    sub = sv.read_ranges("chr1", [0], [40], samples=[sv.samples[1]])
    assert int(sub["n_samples"]) == 1
    ploidy = sv.ploidy
    for p in range(ploidy):
        fh = 1 * ploidy + p
        sh = 0 * ploidy + p
        np.testing.assert_array_equal(
            full["vk_pos"][full["vk_off"][fh] : full["vk_off"][fh + 1]],
            sub["vk_pos"][sub["vk_off"][sh] : sub["vk_off"][sh + 1]],
        )


def test_find_ranges_out_streaming(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    ranges = sv.find_ranges("chr1", [0], [40])
    # Pre-allocate matching-shape buffers and stream into them.
    out = {
        k: np.empty_like(np.asarray(ranges[k]))
        for k in (
            "dense_range",
            "region_starts",
            "sample_cols",
            "vk_snp_range",
            "vk_indel_range",
        )
    }
    ranges2 = sv.find_ranges("chr1", [0], [40], out=out)
    for k in out:
        np.testing.assert_array_equal(np.asarray(ranges2[k]), np.asarray(ranges[k]))
        # out= wrote in place: returned array shares the buffer.
        assert np.asarray(ranges2[k]).base is out[k] or ranges2[k] is out[k]

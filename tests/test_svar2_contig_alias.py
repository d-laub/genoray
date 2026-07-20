"""Reader-side contig-name normalization for SparseVar2 (chr-prefix + mito aliases)."""

import numpy as np
import pytest

from genoray import SparseVar2


def test_decode_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)  # store contig is "chr1"
    native = sv.decode("chr1", [(0, 40)])
    alias = sv.decode("1", [(0, 40)])
    assert (
        native["pos"].lengths.reshape(-1).tolist()
        == alias["pos"].lengths.reshape(-1).tolist()
    )
    assert np.array_equal(np.asarray(native["pos"].data), np.asarray(alias["pos"].data))


def test_region_counts_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)
    assert np.array_equal(
        sv.region_counts("chr1", [(0, 40)]), sv.region_counts("1", [(0, 40)])
    )


def test_read_ranges_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)
    native = sv.read_ranges("chr1", [0], [40])
    alias = sv.read_ranges("1", [0], [40])
    assert np.array_equal(native["vk_pos"], alias["vk_pos"])
    assert np.array_equal(native["vk_key"], alias["vk_key"])


def test_unknown_contig_raises_valueerror(svar2_store):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="not found in store"):
        sv.decode("chrZ", [(0, 40)])

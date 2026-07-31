from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

from genoray import SparseVar2
from genoray._svar2_batch import MAX_END_SHIFT


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
    ob = sv._overlap_batch("chr1", list(zip(starts, ends)))
    rr = sv.read_ranges("chr1", starts, ends)
    _assert_dicts_equal(ob, rr, PAYLOAD_KEYS)
    assert int(rr["n_regions"]) == 2


def test_gather_of_find_matches_read(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    starts, ends = [0], [40]
    ranges = sv._find_ranges("chr1", starts, ends)
    gathered = sv._gather_ranges("chr1", ranges)
    read = sv.read_ranges("chr1", starts, ends)
    _assert_dicts_equal(read, gathered, PAYLOAD_KEYS)


def test_read_ranges_sample_subset(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    full = sv._overlap_batch("chr1", [(0, 40)])
    sub = sv.read_ranges("chr1", [0], [40], samples=[sv.available_samples[1]])
    assert int(sub["n_samples"]) == 1
    ploidy = sv.ploidy
    for p in range(ploidy):
        fh = 1 * ploidy + p
        sh = 0 * ploidy + p
        np.testing.assert_array_equal(
            full["vk_pos"][full["vk_off"][fh] : full["vk_off"][fh + 1]],
            sub["vk_pos"][sub["vk_off"][sh] : sub["vk_off"][sh + 1]],
        )


def test_read_ranges_unknown_sample_raises(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError):
        sv.read_ranges("chr1", [0], [40], samples=["NOT_A_SAMPLE"])


def test_gather_ranges_mismatched_samples_raises(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    ranges = sv._find_ranges("chr1", [0], [40], samples=[sv.available_samples[0]])
    with pytest.raises(ValueError):
        sv._gather_ranges("chr1", ranges, samples=[sv.available_samples[1]])
    # A matching subset is allowed and should not raise.
    sv._gather_ranges("chr1", ranges, samples=[sv.available_samples[0]])


def test_find_ranges_out_streaming(svar2_store: Path):
    sv = SparseVar2(svar2_store)
    ranges = sv._find_ranges("chr1", [0], [40])
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
    ranges2 = sv._find_ranges("chr1", [0], [40], out=out)
    for k in out:
        np.testing.assert_array_equal(np.asarray(ranges2[k]), np.asarray(ranges[k]))
        # out= wrote in place: returned array shares the buffer.
        assert np.asarray(ranges2[k]).base is out[k] or ranges2[k] is out[k]


def test_find_ranges_chunk_matches_find_ranges(svar2_store: Path):
    """Chunked hap slices must reassemble into the region-major bundle exactly."""
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    reg = list(zip(starts, ends))
    reader = sv._reader("chr1")
    bundle = sv._find_ranges("chr1", starts, ends)

    R = len(reg)
    P = sv.ploidy
    S = sv.n_samples
    H = S * P

    header = reader.find_ranges_header(reg, None)
    np.testing.assert_array_equal(
        np.asarray(header["dense_snp_range"]), np.asarray(bundle["dense_snp_range"])
    )
    np.testing.assert_array_equal(
        np.asarray(header["sample_cols"]), np.asarray(bundle["sample_cols"])
    )

    # One hap per call: the most adversarial chunking.
    snp = np.empty((H, R, 2), np.int64)
    indel = np.empty((H, R, 2), np.int64)
    for h in range(H):
        d = reader.find_ranges_chunk(reg, None, h, h + 1)
        snp[h] = np.asarray(d["vk_snp_range"]).reshape(1, R, 2)
        indel[h] = np.asarray(d["vk_indel_range"]).reshape(1, R, 2)

    # bundle vk ranges are region-major (R*H, 2); ours are hap-major (H, R, 2).
    np.testing.assert_array_equal(
        snp.transpose(1, 0, 2).reshape(R * H, 2),
        np.asarray(bundle["vk_snp_range"]),
    )
    np.testing.assert_array_equal(
        indel.transpose(1, 0, 2).reshape(R * H, 2),
        np.asarray(bundle["vk_indel_range"]),
    )


def _reassemble(stream) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, P, S = stream.n_regions, stream.ploidy, stream.n_samples
    snp = np.empty((S, P, R, 2), np.int64)
    indel = np.empty((S, P, R, 2), np.int64)
    keys = stream.dense_max_end_keys.copy()
    for ch in stream.chunks:
        s0, s1 = ch.sample_start, ch.sample_start + ch.n_samples
        snp[s0:s1] = ch.vk_snp_range
        indel[s0:s1] = ch.vk_indel_range
        np.maximum(keys, ch.max_end_keys, out=keys)
    return snp, indel, keys


@pytest.mark.parametrize("max_mem", [None, 1 << 30, 1])
def test_chunked_matches_find_ranges(svar2_store: Path, max_mem):
    """Every chunking, including one sample per chunk, reassembles identically."""
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    bundle = sv._find_ranges("chr1", starts, ends)
    R, P, S = 2, sv.ploidy, sv.n_samples

    if max_mem == 1:
        # 1 byte cannot fit a sample; the API must say so rather than silently
        # producing a zero-sized chunk.
        with pytest.raises(ValueError, match="max_mem"):
            sv._find_ranges_chunked("chr1", starts, ends, max_mem=max_mem)
        return

    stream = sv._find_ranges_chunked("chr1", starts, ends, max_mem=max_mem)
    snp, indel, _ = _reassemble(stream)
    np.testing.assert_array_equal(
        snp.reshape(S * P, R, 2).transpose(1, 0, 2).reshape(R * S * P, 2),
        np.asarray(bundle["vk_snp_range"]),
    )
    np.testing.assert_array_equal(
        indel.reshape(S * P, R, 2).transpose(1, 0, 2).reshape(R * S * P, 2),
        np.asarray(bundle["vk_indel_range"]),
    )


def test_chunked_max_end_keys_unpack_to_variant_ends(svar2_store: Path):
    """The reduced key unpacks to the end of the highest-position variant.

    The fixture's chr1 carries SNP@2, INS@6 and DEL@11 (ilen -2, so it ends at
    11 + 1 + 2 = 14). Region [0, 40) therefore ends at 14; region [0, 5) sees
    only SNP@2, which ends at 3.
    """
    sv = SparseVar2(svar2_store)
    stream = sv._find_ranges_chunked("chr1", [0, 0], [40, 5])
    _, _, keys = _reassemble(stream)
    mask = (1 << MAX_END_SHIFT) - 1
    ends = (keys >> MAX_END_SHIFT) + (keys & mask)
    assert keys[0] != 0 and keys[1] != 0
    assert int(ends[0]) == 14
    assert int(ends[1]) == 3


def test_chunked_sample_subset(svar2_store: Path):
    """A sample subset takes the carriage-probing dense path, not the fast path."""
    sub = [SparseVar2(svar2_store).available_samples[1]]
    sv = SparseVar2(svar2_store)
    bundle = sv._find_ranges("chr1", [0], [40], samples=sub)
    stream = sv._find_ranges_chunked("chr1", [0], [40], samples=sub)
    assert stream.n_samples == 1
    snp, _, _ = _reassemble(stream)
    np.testing.assert_array_equal(
        snp.reshape(-1, 2), np.asarray(bundle["vk_snp_range"])
    )

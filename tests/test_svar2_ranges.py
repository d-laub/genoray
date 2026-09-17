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


def test_find_ranges_header_matches_bundle(svar2_store: Path):
    """The stream header's ``O(n_regions)`` arrays must equal the bundle's.

    The chunked stream computes this header once via ``find_ranges_header``,
    while ``_find_ranges`` builds the same arrays on its own path. Every test
    below checks a stream against a bundle, so a divergence here would desync
    the oracle from the thing under test rather than failing loudly.
    """
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    reader = sv._reader("chr1")
    bundle = sv._find_ranges("chr1", starts, ends)
    header = reader.find_ranges_header(
        reader.ranges_query(list(zip(starts, ends)), None)
    )
    for k in (
        "dense_range",
        "dense_snp_range",
        "dense_indel_range",
        "region_starts",
        "sample_cols",
    ):
        np.testing.assert_array_equal(
            np.asarray(header[k]), np.asarray(bundle[k]), err_msg=k
        )


def _bundle_dense_haps(bundle, n_haps: int, n_regions: int):
    """Dense hap-major ``(n_haps, R, 2)`` ranges, unfolded from a ``_find_ranges`` bundle.

    The chunked dense stream that used to be this oracle is gone (#204), but
    ``_find_ranges`` still returns the dense bundle and still reaches it through
    the same kernel, ``find_ranges_haps`` -- just without the chunking. The
    bundle is region-major ``(R * n_haps, 2)`` with the hap axis fastest, so
    unfolding and transposing recovers exactly what the deleted path yielded.
    """
    return tuple(
        np.asarray(bundle[k]).reshape(n_regions, n_haps, 2).transpose(1, 0, 2)
        for k in ("vk_snp_range", "vk_indel_range")
    )


def _dense_to_sparse(
    snp: np.ndarray, indel: np.ndarray, sample_start: int, ploidy: int
):
    """Reference sparsifier: GVL's ``nonempty_entries``, as region-major CSR.

    ``snp``/``indel`` are hap-major ``(n_samples, ploidy, R, 2)``. Transposing to
    ``(R, n_samples, ploidy, 2)`` and running ``np.nonzero`` walks the LOGICAL
    shape in C order, so the result is region-major with ``cell_id`` ascending
    inside each region -- exactly the contract the Rust kernel must reproduce.
    """
    R = snp.shape[2]
    s = snp.transpose(2, 0, 1, 3)
    i = indel.transpose(2, 0, 1, 3)
    ne = (s[..., 1] > s[..., 0]) | (i[..., 1] > i[..., 0])
    ri, sj, pj = np.nonzero(ne)
    ptr = np.concatenate([[0], np.cumsum(np.bincount(ri, minlength=R))]).astype(
        np.int64
    )
    cell = ((sample_start + sj) * ploidy + pj).astype(np.int32)
    return (
        ptr,
        cell,
        s[ri, sj, pj, 0].astype(np.int64),
        (s[ri, sj, pj, 1] - s[ri, sj, pj, 0]).astype(np.int32),
        i[ri, sj, pj, 0].astype(np.int64),
        (i[ri, sj, pj, 1] - i[ri, sj, pj, 0]).astype(np.int32),
    )


def _assert_sparse_matches_dense(got, snp, indel, sample_start, ploidy):
    want = _dense_to_sparse(snp, indel, sample_start, ploidy)
    names = [
        "region_ptr",
        "cell_id",
        "snp_start",
        "snp_len",
        "indel_start",
        "indel_len",
    ]
    for name, g, w in zip(names, got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(g), w, err_msg=name)
        assert np.asarray(g).dtype == w.dtype, name


@pytest.mark.parametrize("n_samples", [12, 32])
def test_sparse_chunk_matches_dense_chunk(
    n_samples: int, svar2_singleton_store: Path, tmp_path: Path
):
    """Binding-level parity, per hap slice, against the np.nonzero reference.

    Parametrized over sample count so both branches of
    ``find_ranges_haps_sparse`` run: ``n_samples=12`` (24 haps) stays under
    ``PAR_COLUMN_THRESHOLD`` (64) and takes the serial path;
    ``n_samples=32`` (64 haps) meets it and takes the rayon path. The ordered
    ``collect()`` on that path is what the whole region-major /
    cell_id-ascending contract rests on, so it needs its own coverage rather
    than trusting the serial path's result.
    """
    from tests.conftest import build_svar2_singleton_store

    store = (
        svar2_singleton_store
        if n_samples == 12
        else build_svar2_singleton_store(tmp_path, n_samples=n_samples)
    )
    sv = SparseVar2(store)
    starts, ends = [0, 0], [20, 5]
    reg = list(zip(starts, ends))
    reader = sv._reader("chr1")
    P, S = sv.ploidy, sv.n_samples
    query = reader.ranges_query(reg, None)
    all_snp, all_indel = _bundle_dense_haps(
        sv._find_ranges("chr1", starts, ends), S * P, len(reg)
    )

    for hap_lo, hap_hi in [(0, S * P), (0, P), (P, 3 * P), (S * P, S * P)]:
        shape = ((hap_hi - hap_lo) // P, P, len(reg), 2)
        snp = all_snp[hap_lo:hap_hi].reshape(shape)
        indel = all_indel[hap_lo:hap_hi].reshape(shape)

        got = reader.find_ranges_chunk_sparse(query, hap_lo, hap_hi)
        _assert_sparse_matches_dense(got[:6], snp, indel, hap_lo // P, P)


def test_sparse_chunk_cell_ids_ascend_within_each_region(
    svar2_singleton_store: Path,
):
    """The counting sort's stability, observed end to end.

    Every sample carries one singleton, so region [0, 20) has one non-empty cell
    per sample and region [0, 5) has a strict subset of them. A non-stable sort
    would scramble cell_id inside a region block; a hap-major emitter would put
    region 1's entries before region 0's.
    """
    sv = SparseVar2(svar2_singleton_store)
    reg = [(0, 20), (0, 5)]
    reader = sv._reader("chr1")
    S, P = sv.n_samples, sv.ploidy
    ptr, cell, *_ = reader.find_ranges_chunk_sparse(
        reader.ranges_query(reg, None), 0, S * P
    )
    ptr = np.asarray(ptr)
    cell = np.asarray(cell)

    assert ptr.shape == (len(reg) + 1,)
    assert ptr[0] == 0 and ptr[-1] == len(cell)
    assert np.all(np.diff(ptr) >= 0)
    # Region 0 spans every singleton; region 1 only the first few.
    assert ptr[1] - ptr[0] == S
    assert 0 < ptr[2] - ptr[1] < S
    for r in range(len(reg)):
        block = cell[ptr[r] : ptr[r + 1]]
        assert np.all(np.diff(block) > 0), f"region {r}: {block}"
    # Sample i carries its singleton on hap i % 2.
    np.testing.assert_array_equal(
        cell[ptr[0] : ptr[1]],
        np.array([2 * i + (i % 2) for i in range(S)], np.int32),
    )


def test_sparse_chunk_empty_region_yields_no_entries(svar2_singleton_store: Path):
    """A region with no variants contributes an empty CSR block, not a row."""
    sv = SparseVar2(svar2_singleton_store)
    reader = sv._reader("chr1")
    S, P = sv.n_samples, sv.ploidy
    ptr, cell, snp_start, snp_len, indel_start, indel_len, keys = (
        reader.find_ranges_chunk_sparse(reader.ranges_query([(30, 40)], None), 0, S * P)
    )
    np.testing.assert_array_equal(np.asarray(ptr), np.zeros(2, np.int64))
    for arr in (cell, snp_start, snp_len, indel_start, indel_len):
        assert len(np.asarray(arr)) == 0
    np.testing.assert_array_equal(np.asarray(keys), np.zeros(1, np.int64))


def test_sparse_chunk_sample_subset(svar2_singleton_store: Path):
    """cell_id indexes the SELECTION, not the store's sample axis."""
    sv = SparseVar2(svar2_singleton_store)
    sub = [sv.available_samples[3], sv.available_samples[1]]
    reader = sv._reader("chr1")
    idxs = sv._sample_idxs(sub)
    P = sv.ploidy
    reg = [(0, 20)]
    query = reader.ranges_query(reg, idxs)
    snp, indel = (
        a.reshape(len(sub), P, 1, 2)
        for a in _bundle_dense_haps(
            sv._find_ranges("chr1", [0], [20], samples=sub), len(sub) * P, 1
        )
    )
    got = reader.find_ranges_chunk_sparse(query, 0, len(sub) * P)
    _assert_sparse_matches_dense(got[:6], snp, indel, 0, P)
    assert np.asarray(got[1]).max() < len(sub) * P


def test_sparse_chunk_rejects_out_of_bounds_hap_slice(svar2_singleton_store: Path):
    sv = SparseVar2(svar2_singleton_store)
    reader = sv._reader("chr1")
    H = sv.n_samples * sv.ploidy
    with pytest.raises(ValueError, match="out of bounds"):
        reader.find_ranges_chunk_sparse(reader.ranges_query([(0, 20)], None), 0, H + 1)
    with pytest.raises(ValueError, match="out of bounds"):
        reader.find_ranges_chunk_sparse(reader.ranges_query([(0, 20)], None), 2, 1)


def _reassemble_sparse(stream):
    """Concatenate a sparse stream into one region-major CSR table.

    Chunks partition the SAMPLE axis, so each chunk holds a full CSR over all
    regions; merging chunks means interleaving their region blocks, which is exactly
    what the consumer's per-contig merge does.
    """
    R = stream.n_regions
    keys = stream.dense_max_end_keys.copy()
    blocks = [[] for _ in range(R)]
    for ch in stream.chunks:
        ptr = np.asarray(ch.region_ptr)
        for r in range(R):
            s, e = ptr[r], ptr[r + 1]
            blocks[r].append(
                (
                    np.asarray(ch.cell_id)[s:e],
                    np.asarray(ch.snp_start)[s:e],
                    np.asarray(ch.snp_len)[s:e],
                    np.asarray(ch.indel_start)[s:e],
                    np.asarray(ch.indel_len)[s:e],
                )
            )
        np.maximum(keys, np.asarray(ch.max_end_keys), out=keys)
    cols = [np.concatenate([b[i] for r in blocks for b in r]) for i in range(5)]
    counts = [sum(len(b[0]) for b in r) for r in blocks]
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return (ptr, *cols), keys


@pytest.mark.parametrize("max_mem", [None, 1 << 30, 1 << 8])
def test_sparse_stream_matches_find_ranges(svar2_singleton_store: Path, max_mem):
    """Every chunking, down to one sample per chunk, reassembles identically.

    At two regions and ploidy 2, ``bytes_per_sample`` is 128, so ``1 << 8``
    sizes chunks at exactly one sample -- the most adversarial split, and the
    smallest value that does not raise.

    The reduced ``max_end_keys`` are checked for chunk-invariance against the
    single-chunk run rather than against a second implementation; what they
    unpack to in absolute terms is pinned to fixture constants in
    :func:`test_stream_max_end_keys_unpack_to_variant_ends`.
    """
    sv = SparseVar2(svar2_singleton_store)
    starts, ends = [0, 0], [20, 5]
    snp, indel = (
        a.reshape(sv.n_samples, sv.ploidy, 2, 2)
        for a in _bundle_dense_haps(
            sv._find_ranges("chr1", starts, ends), sv.n_samples * sv.ploidy, 2
        )
    )

    sparse = sv._find_ranges_chunked_sparse("chr1", starts, ends, max_mem=max_mem)
    got, sparse_keys = _reassemble_sparse(sparse)
    _assert_sparse_matches_dense(got, snp, indel, 0, sv.ploidy)
    _, whole_keys = _reassemble_sparse(
        sv._find_ranges_chunked_sparse("chr1", starts, ends)
    )
    np.testing.assert_array_equal(sparse_keys, whole_keys)


def test_stream_max_end_keys_unpack_to_variant_ends(svar2_store: Path):
    """The reduced key unpacks to the end of the highest-position variant.

    The fixture's chr1 carries SNP@2, INS@6 and DEL@11 (ilen -2, so it ends at
    11 + 1 + 2 = 14). Region [0, 40) therefore ends at 14; region [0, 5) sees
    only SNP@2, which ends at 3.
    """
    sv = SparseVar2(svar2_store)
    _, keys = _reassemble_sparse(
        sv._find_ranges_chunked_sparse("chr1", [0, 0], [40, 5])
    )
    mask = (1 << MAX_END_SHIFT) - 1
    ends = (keys >> MAX_END_SHIFT) + (keys & mask)
    assert keys[0] != 0 and keys[1] != 0
    assert int(ends[0]) == 14
    assert int(ends[1]) == 3


def test_stream_sample_subset_max_end_key(svar2_store: Path):
    """A sample subset takes the carriage-probing dense path, not the fast path."""
    sv = SparseVar2(svar2_store)
    sub = [sv.available_samples[1]]
    P = sv.ploidy
    snp, indel = (
        a.reshape(1, P, 1, 2)
        for a in _bundle_dense_haps(
            sv._find_ranges("chr1", [0], [40], samples=sub), P, 1
        )
    )
    stream = sv._find_ranges_chunked_sparse("chr1", [0], [40], samples=sub)
    assert stream.n_samples == 1
    got, keys = _reassemble_sparse(stream)
    _assert_sparse_matches_dense(got, snp, indel, 0, P)

    # Pins the subset path's max-end output to a fixture-derived constant, so
    # a gross regression in the carriage-probe branch (e.g. it stops firing,
    # or returns garbage) doesn't go unnoticed. This does NOT verify genuine
    # per-sample carrier filtering: on this fixture both dense-routed variants
    # (INS@6, DEL@11) are carried by BOTH samples, so a probe that ignored
    # `sample_cols` entirely would produce this exact same value. Real
    # carrier-filtering coverage -- a fixture where the excluded sample's
    # variant actually differs from the included one's -- lives in
    # tests/test_ranges_split.rs::test_dense_max_end_keys_excludes_uncarried_sample.
    expected_key = (11 << MAX_END_SHIFT) | 3  # DEL@11: ext = 1 + del_len(2) = 3, end 14
    assert int(keys[0]) == expected_key


def test_sparse_stream_cell_ids_are_absolute_across_chunks(
    svar2_singleton_store: Path,
):
    """A chunk's cell_id indexes the whole selection, not the chunk."""
    sv = SparseVar2(svar2_singleton_store)
    stream = sv._find_ranges_chunked_sparse("chr1", [0], [20], max_mem=1 << 8)
    P = stream.ploidy
    seen = []
    for ch in stream.chunks:
        cid = np.asarray(ch.cell_id)
        assert np.all(cid >= ch.sample_start * P)
        assert np.all(cid < (ch.sample_start + ch.n_samples) * P)
        seen.append(cid)
    assert stream.samples_per_chunk < stream.n_samples, "expected several chunks"
    np.testing.assert_array_equal(
        np.concatenate(seen),
        np.array([2 * i + (i % 2) for i in range(stream.n_samples)], np.int32),
    )


def test_sparse_stream_rejects_unusable_max_mem(svar2_singleton_store: Path):
    sv = SparseVar2(svar2_singleton_store)
    with pytest.raises(ValueError, match="max_mem"):
        sv._find_ranges_chunked_sparse("chr1", [0], [20], max_mem=1)


def test_sparse_stream_sample_subset(svar2_singleton_store: Path):
    sv = SparseVar2(svar2_singleton_store)
    sub = [sv.available_samples[2], sv.available_samples[5]]
    snp, indel = (
        a.reshape(2, sv.ploidy, 1, 2)
        for a in _bundle_dense_haps(
            sv._find_ranges("chr1", [0], [20], samples=sub), 2 * sv.ploidy, 1
        )
    )
    sparse = sv._find_ranges_chunked_sparse("chr1", [0], [20], samples=sub)
    assert sparse.n_samples == 2
    got, _ = _reassemble_sparse(sparse)
    _assert_sparse_matches_dense(got, snp, indel, 0, sv.ploidy)


def test_ranges_dataclasses_are_slotted():
    from dataclasses import fields

    from genoray._svar2_batch import RangesStream, SparseRangesChunk

    for cls in (RangesStream, SparseRangesChunk):
        # `cls.__dict__`, not `hasattr`: an inherited `__slots__` would pass
        # hasattr while the class itself still carried a per-instance dict.
        slots = cls.__dict__.get("__slots__")
        assert slots is not None, cls.__name__
        assert set(slots) == {f.name for f in fields(cls)}, cls.__name__


def test_ranges_query_rejects_out_of_bounds_sample(svar2_singleton_store: Path):
    """The selection is validated once, where it is marshalled.

    Left to the chunk calls, an out-of-range index would only be reached when
    the chunking happened to cover that hap -- so the same query would raise or
    not depending on ``max_mem``, and would surface as an index panic from
    inside the search rather than a ``ValueError``.
    """
    sv = SparseVar2(svar2_singleton_store)
    reader = sv._reader("chr1")
    S = sv.n_samples
    with pytest.raises(ValueError, match="out of bounds"):
        reader.ranges_query([(0, 20)], [0, S])
    # The last valid index is not off-by-one rejected.
    reader.ranges_query([(0, 20)], [S - 1])

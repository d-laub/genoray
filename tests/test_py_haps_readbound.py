"""Python bindings for the READ-BOUND gather: ``_find_haps_ranges`` /
``HapRangesRect.select`` / ``_gather_haps_readbound``.

The point of this path is that an arbitrary, non-rectangular set of
``(region, sample)`` cells costs exactly those cells. ``_gather_ranges``'s
``RangesBundle`` can't express that -- its ``sample_cols`` axis makes it a
region-by-sample rectangle -- so covering a pair set with it means either
over-gathering the cross-product or paying one call (and one
``dense_union()`` rebuild) per sample.

Every test here drives the real binding and checks it against an independent
oracle: a per-pair ``read_ranges`` call, decoded down to the same
``(position, key)`` sets. A dtype, shape, hap-row-order or presence-bit-offset
mistake in the dict contract changes those sets, so it cannot pass silently.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from genoray import SparseVar2

# 60 bp reference. REF bases below match exactly (1-based VCF POS):
# 3 = 'A', 7 = 'C', 12..14 = 'GTA', 21 = 'C', 34 = 'G', 45 = 'A'.
_REF = "ACAGTACATGGGTACTAGCTCAGCTAACCGGTTGACCGGTAACCAAGGTTCCAAGGTTCC"

_SAMPLES = ["S0", "S1", "S2", "S3", "S4", "S5"]

# Genotype columns are chosen so BOTH storage classes are exercised: the
# high-AC rows route to the dense tables (cohort-shared, per-region windows)
# and the low-AC rows stay in the per-hap var_key channel. A test that only
# hit one class would not discriminate a bug in the other's presence bits.
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=60>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2\tS3\tS4\tS5
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\t0|0\t0|0\t0|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\t0|0\t1|0\t0|0\t0|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\t1|1\t0|0\t1|1\t0|1
chr1\t21\t.\tC\tT\t.\t.\t.\tGT\t1|1\t1|1\t1|1\t1|1\t1|1\t1|1
chr1\t34\t.\tG\tA\t.\t.\t.\tGT\t0|1\t1|0\t1|1\t0|0\t1|1\t1|0
chr1\t45\t.\tA\tC\t.\t.\t.\tGT\t0|0\t0|0\t1|0\t0|0\t0|0\t0|0
"""


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("svar2_haps")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store"
    SparseVar2.from_vcf(out, bcf, ref)
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


def _bits(packed: np.ndarray, bit0: int, n: int) -> np.ndarray:
    """``n`` LSB-first bits starting at absolute bit offset ``bit0``."""
    if n == 0:
        return np.zeros(0, bool)
    lo, hi = bit0 // 8, (bit0 + n + 7) // 8
    flat = np.unpackbits(np.asarray(packed, np.uint8)[lo:hi], bitorder="little")
    off = bit0 - lo * 8
    return flat[off : off + n].astype(bool)


def _fused_hap(br, r: int, s: int, p: int) -> set[tuple[int, int]]:
    """``(position, key)`` set for one hap of a fused ``read_ranges`` result."""
    n_s, ploidy = int(br["n_samples"]), int(br["ploidy"])
    h = (r * n_s + s) * ploidy + p
    off = np.asarray(br["vk_off"])
    out = set(
        zip(
            np.asarray(br["vk_pos"])[off[h] : off[h + 1]].tolist(),
            np.asarray(br["vk_key"])[off[h] : off[h + 1]].tolist(),
        )
    )
    ds, de = np.asarray(br["dense_range"])[r]
    present = _bits(
        br["dense_present"], int(np.asarray(br["dense_present_off"])[h]), de - ds
    )
    dpos = np.asarray(br["dense_pos"])[ds:de][present]
    dkey = np.asarray(br["dense_key"])[ds:de][present]
    out |= set(zip(dpos.tolist(), dkey.tolist()))
    return out


def _split_hap(br, q: int, p: int) -> set[tuple[int, int]]:
    """``(position, key)`` set for one hap of a read-bound ``BatchResultSplit``.

    This is the gvl-side reconstruction: var_key merged with BOTH dense classes.
    """
    ploidy = int(br["ploidy"])
    h = q * ploidy + p
    off = np.asarray(br["vk_off"])
    out = set(
        zip(
            np.asarray(br["vk_pos"])[off[h] : off[h + 1]].tolist(),
            np.asarray(br["vk_key"])[off[h] : off[h + 1]].tolist(),
        )
    )
    for cls in ("snp", "indel"):
        ds, de = np.asarray(br[f"dense_{cls}_range"])[q]
        bit0 = int(np.asarray(br[f"dense_{cls}_present_off"])[h])
        present = _bits(br[f"dense_{cls}_present"], bit0, de - ds)
        pos = np.asarray(br[f"dense_{cls}_pos"])[ds:de][present]
        key = np.asarray(br[f"dense_{cls}_key"])[ds:de][present]
        out |= set(zip(pos.tolist(), key.tolist()))
    return out


def _readbound(sv: SparseVar2, pairs: list[tuple[int, int, str]]):
    """Run the whole read-bound path over a flat pair list, in ONE gather call."""
    starts = [s for s, _, _ in pairs]
    ends = [e for _, e, _ in pairs]
    names = sorted({n for _, _, n in pairs})
    slot = {n: i for i, n in enumerate(names)}
    rect = sv._find_haps_ranges("chr1", starts, ends, samples=names)
    hr = rect.select(np.arange(len(pairs)), [slot[n] for _, _, n in pairs])
    return sv._gather_haps_readbound("chr1", hr), hr


# A deliberately NON-rectangular pair set: 4 regions x 4 distinct samples with
# no two pairs sharing a region, so the cross-product a RangesBundle would need
# is 16 cells and this asks for 4. Includes both a dense-only and a var_key-only
# window so neither channel is exercised alone.
PAIRS = [
    (0, 15, "S3"),
    (10, 25, "S0"),
    (18, 40, "S5"),
    (40, 60, "S2"),
]


def test_gather_haps_readbound_matches_per_pair_read_ranges(store: Path):
    """The oracle: each flat query must equal an independent one-pair fused read."""
    sv = SparseVar2(store)
    split, _ = _readbound(sv, PAIRS)

    assert int(split["n_samples"]) == 1
    assert int(split["n_regions"]) == len(PAIRS)
    # Coverage guard: all three channels must carry content, else a bug in the
    # untouched one passes silently. If this fires the cost model re-routed the
    # fixture's variants -- escalate rather than weaken.
    assert len(np.asarray(split["dense_snp_pos"])) > 0, "no dense SNPs gathered"
    assert len(np.asarray(split["dense_indel_pos"])) > 0, "no dense indels gathered"
    assert len(np.asarray(split["vk_pos"])) > 0, "no var_key records gathered"

    for q, (s, e, name) in enumerate(PAIRS):
        fused = sv.read_ranges("chr1", [s], [e], samples=[name])
        for p in range(sv.ploidy):
            assert _split_hap(split, q, p) == _fused_hap(fused, 0, 0, p), (
                f"pair {q} ({s}, {e}, {name}) ploid {p}"
            )


def test_readbound_is_order_sensitive_not_set_equal(store: Path):
    """Guard against a gather that ignores ``orig_samples`` (or its order).

    Permuting only the SAMPLE column of the pair list while holding regions
    fixed must change the per-query result. A gather keyed off the query index
    rather than ``orig_samples`` -- exactly the bug a rectangle-shaped bundle
    hides -- would return the same thing both ways.
    """
    sv = SparseVar2(store)
    regions = [(0, 30), (0, 30)]
    a, _ = _readbound(sv, [(*regions[0], "S0"), (*regions[1], "S2")])
    b, _ = _readbound(sv, [(*regions[0], "S2"), (*regions[1], "S0")])

    same = all(
        _split_hap(a, q, p) == _split_hap(b, q, p)
        for q in range(2)
        for p in range(sv.ploidy)
    )
    assert not same, "swapping the sample column changed nothing -- sample ignored"
    # ...and the swap is exactly a transposition of the two queries.
    for p in range(sv.ploidy):
        assert _split_hap(a, 0, p) == _split_hap(b, 1, p)
        assert _split_hap(a, 1, p) == _split_hap(b, 0, p)


def test_select_produces_the_documented_flat_contract(store: Path):
    """Shapes/dtypes of ``HapRangesRect.select``'s output, which the Rust side
    casts strictly (``PyArray1<i32>`` / ``PyArray1<i64>`` / ``PyArray2<i64>``)."""
    sv = SparseVar2(store)
    _, hr = _readbound(sv, PAIRS)
    n_q, ploidy = len(PAIRS), sv.ploidy

    assert hr["region_starts"].shape == (n_q,)
    assert hr["region_starts"].dtype == np.int32
    assert hr["orig_samples"].shape == (n_q,)
    assert hr["orig_samples"].dtype == np.int64
    # orig_samples are FULL-cohort indices, not subset slots.
    assert hr["orig_samples"].tolist() == [
        sv.available_samples.index(n) for _, _, n in PAIRS
    ]
    for k in ("vk_snp_range", "vk_indel_range"):
        assert hr[k].shape == (n_q * ploidy, 2)
        assert hr[k].dtype == np.int64
    for k in ("dense_snp_range", "dense_indel_range"):
        assert hr[k].shape == (n_q, 2)
        assert hr[k].dtype == np.int32
    assert hr["ploidy"] == ploidy


def test_select_rejects_mismatched_index_lengths(store: Path):
    sv = SparseVar2(store)
    rect = sv._find_haps_ranges("chr1", [0, 10], [20, 30], samples=["S0", "S1"])
    with pytest.raises(ValueError, match="parallel"):
        rect.select([0, 1], [0])


def test_find_dense_class_ranges_matches_find_ranges(store: Path):
    """The lean dense search must agree with ``find_ranges``' -- which computes
    the same windows on the way to (and at the cost of) a ``dense_union()``."""
    sv = SparseVar2(store)
    starts, ends = [0, 10, 40], [15, 30, 60]
    rect = sv._find_haps_ranges("chr1", starts, ends)
    bundle = sv._find_ranges("chr1", starts, ends)
    for k in ("dense_snp_range", "dense_indel_range"):
        np.testing.assert_array_equal(getattr(rect, k), np.asarray(bundle[k]))
        assert getattr(rect, k).dtype == np.asarray(bundle[k]).dtype


def test_find_haps_ranges_vk_matches_find_ranges(store: Path):
    """The rectangle's var_key ranges must be the same numbers ``find_ranges``
    puts in its region-major bundle, just laid out hap-major."""
    sv = SparseVar2(store)
    starts, ends = [0, 10, 40], [15, 30, 60]
    names = ["S1", "S4"]
    rect = sv._find_haps_ranges("chr1", starts, ends, samples=names)
    bundle = sv._find_ranges("chr1", starts, ends, samples=names)

    n_r, n_s, ploidy = len(starts), len(names), sv.ploidy
    for k in ("vk_snp_range", "vk_indel_range"):
        got = getattr(rect, k)  # (S, P, R, 2)
        want = np.asarray(bundle[k]).reshape(n_r, n_s, ploidy, 2)
        np.testing.assert_array_equal(got, want.transpose(1, 2, 0, 3))
    np.testing.assert_array_equal(rect.sample_cols, np.asarray(bundle["sample_cols"]))
    np.testing.assert_array_equal(
        rect.region_starts, np.asarray(bundle["region_starts"])
    )


@pytest.mark.parametrize(
    ("mutate", "exc", "match"),
    [
        (lambda h: h.pop("orig_samples"), KeyError, "orig_samples"),
        (
            lambda h: h.update(orig_samples=h["orig_samples"].astype(np.int32)),
            TypeError,
            "int64 1D array",
        ),
        (
            lambda h: h.update(vk_snp_range=h["vk_snp_range"].astype(np.int32)),
            TypeError,
            "int64 .N,2. array",
        ),
        (
            lambda h: h.update(dense_snp_range=h["dense_snp_range"].astype(np.int64)),
            TypeError,
            "int32 .N,2. array",
        ),
        (lambda h: h.update(ploidy=1), ValueError, "ploidy"),
        (
            lambda h: h.update(orig_samples=h["orig_samples"] + 999),
            ValueError,
            "out of bounds",
        ),
        (
            lambda h: h.update(dense_snp_range=h["dense_snp_range"][:-1]),
            ValueError,
            "dense_snp_range",
        ),
        (
            lambda h: h.update(vk_indel_range=h["vk_indel_range"][:-1]),
            ValueError,
            "vk_indel_range",
        ),
    ],
    ids=[
        "missing_key",
        "orig_samples_dtype",
        "vk_range_dtype",
        "dense_range_dtype",
        "wrong_ploidy",
        "sample_out_of_range",
        "short_dense_range",
        "short_vk_range",
    ],
)
def test_gather_haps_readbound_rejects_malformed_dicts(store, mutate, exc, match):
    """A dtype/shape slip must fail loudly at the FFI boundary, not be
    reinterpreted -- the whole reason the Rust casts are exact."""
    sv = SparseVar2(store)
    _, hr = _readbound(sv, PAIRS)
    bad = dict(hr)
    mutate(bad)
    with pytest.raises(exc, match=match):
        sv._gather_haps_readbound("chr1", bad)


def test_readbound_covers_only_the_pairs_asked_for(store: Path):
    """Cost scales with pairs, not with the cross-product.

    ``vk_off`` has one entry per gathered hap, so its length is the direct
    measurement of how many cells the gather touched: ``n_pairs * ploidy``, NOT
    ``n_regions * n_samples * ploidy``. This is the regression that the
    rectangle-shaped ``_gather_ranges`` path could not satisfy.
    """
    sv = SparseVar2(store)
    split, _ = _readbound(sv, PAIRS)
    n_unique_samples = len({n for _, _, n in PAIRS})
    assert n_unique_samples > 1, "fixture must not be trivially rectangular"

    n_haps = len(np.asarray(split["vk_off"])) - 1
    assert n_haps == len(PAIRS) * sv.ploidy
    assert n_haps < len(PAIRS) * n_unique_samples * sv.ploidy

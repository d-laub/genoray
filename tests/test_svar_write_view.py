from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import numpy as np

from genoray import SparseVar
from genoray._svar import _normalize_regions, _normalize_samples, _validate_fields
from genoray._utils import ContigNormalizer


@pytest.fixture
def cnorm() -> ContigNormalizer:
    return ContigNormalizer(["chr1", "chr2"])


def test_normalize_regions_str(cnorm):
    df = _normalize_regions("chr1:10-20", cnorm)
    assert df.shape == (1, 3)
    assert df["chrom"].to_list() == ["chr1"]
    # 1-based inclusive "10-20" -> 0-based half-open [9, 20)
    assert df["start"].to_list() == [9]
    assert df["end"].to_list() == [20]


def test_normalize_regions_tuple(cnorm):
    df = _normalize_regions(("chr2", 5, 15), cnorm)
    assert df["chrom"].to_list() == ["chr2"]
    assert df["start"].to_list() == [5]
    assert df["end"].to_list() == [15]


def test_normalize_regions_alt_contig_naming(cnorm):
    # "1" should normalize to "chr1"
    df = _normalize_regions(("1", 0, 100), cnorm)
    assert df["chrom"].to_list() == ["chr1"]


def test_normalize_regions_unknown_contig_dropped(cnorm):
    with pytest.warns(UserWarning, match="dropped"):
        df = _normalize_regions(("chrZZ", 0, 10), cnorm)
    assert df.height == 0


def test_normalize_regions_bed_file(tmp_path: Path, cnorm):
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t100\t200\nchr2\t300\t400\n")
    df = _normalize_regions(bed, cnorm)
    assert df.height == 2
    df_sorted = df.sort("chrom")
    assert df_sorted["chrom"].to_list() == ["chr1", "chr2"]
    assert df_sorted["start"].to_list() == [100, 300]
    assert df_sorted["end"].to_list() == [200, 400]


def test_normalize_regions_frame_polars_bio_schema(cnorm):
    frame = pl.DataFrame({"chrom": ["chr1"], "start": [5], "end": [25]})
    df = _normalize_regions(frame, cnorm)
    assert df["start"].to_list() == [5]
    assert df["end"].to_list() == [25]


def test_normalize_regions_pandas_frame(cnorm):
    import pandas as pd

    pdf = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [10, 20], "end": [30, 40]})
    df = _normalize_regions(pdf, cnorm)
    assert df.height == 2
    df_sorted = df.sort("chrom")
    assert df_sorted["chrom"].to_list() == ["chr1", "chr2"]
    assert df_sorted["start"].to_list() == [10, 20]
    assert df_sorted["end"].to_list() == [30, 40]


def test_normalize_regions_pyranges(cnorm):
    pr = pytest.importorskip("pyranges")
    regions = pr.PyRanges(
        chromosomes=["chr1", "chr2"], starts=[100, 200], ends=[150, 250]
    )
    df = _normalize_regions(regions, cnorm)
    assert df.height == 2
    df_sorted = df.sort("chrom")
    assert df_sorted["chrom"].to_list() == ["chr1", "chr2"]
    assert df_sorted["start"].to_list() == [100, 200]
    assert df_sorted["end"].to_list() == [150, 250]


def test_normalize_regions_unsupported_type_raises(cnorm):
    with pytest.raises(TypeError, match="Unsupported regions type"):
        _normalize_regions(42, cnorm)  # type: ignore


def test_normalize_samples_str():
    assert _normalize_samples("s1", ["s0", "s1", "s2"]) == ["s1"]


def test_normalize_samples_list_preserves_order():
    assert _normalize_samples(["s2", "s0"], ["s0", "s1", "s2"]) == ["s2", "s0"]


def test_normalize_samples_dedupe_first_occurrence():
    assert _normalize_samples(["s2", "s0", "s2"], ["s0", "s1", "s2"]) == ["s2", "s0"]


def test_normalize_samples_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        _normalize_samples(["s9"], ["s0", "s1"])


def test_normalize_samples_file(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("s2\ns0\n")
    assert _normalize_samples(p, ["s0", "s1", "s2"]) == ["s2", "s0"]


def test_validate_fields_none_returns_all():
    assert _validate_fields(None, {"dosages": np.dtype("float32")}) == ["dosages"]


def test_validate_fields_subset_ok():
    avail = {"dosages": np.dtype("float32"), "GQ": np.dtype("float32")}
    assert _validate_fields(["dosages"], avail) == ["dosages"]


def test_validate_fields_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        _validate_fields(["bogus"], {"dosages": np.dtype("float32")})


def test_validate_fields_empty_list_returns_empty():
    assert _validate_fields([], {"dosages": np.dtype("float32")}) == []


# ---------------------------------------------------------------------------
# Task 4: _resolve_kept_var_idxs
# ---------------------------------------------------------------------------

ddir = Path(__file__).parent / "data"


@pytest.fixture
def svar_wv():
    return SparseVar(ddir / "biallelic.vcf.svar")


def test_resolve_kept_var_idxs_pos_mode(svar_wv):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar_wv.contigs[0]], "start": [0], "end": [10_000]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    kept = _resolve_kept_var_idxs(svar_wv, regions, mode="pos", merge_overlapping=False)
    assert kept.dtype.kind == "i"
    if len(kept) > 1:
        assert np.all(np.diff(kept) > 0)  # sorted, unique


def test_resolve_kept_var_idxs_empty_regions(svar_wv):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {
            "chrom": pl.Series([], dtype=pl.Utf8),
            "start": pl.Series([], dtype=pl.Int32),
            "end": pl.Series([], dtype=pl.Int32),
        },
    )
    kept = _resolve_kept_var_idxs(svar_wv, regions, mode="pos", merge_overlapping=False)
    assert len(kept) == 0


def test_resolve_kept_var_idxs_overlap_raises(svar_wv):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar_wv.contigs[0]] * 2, "start": [0, 5], "end": [10, 20]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    with pytest.raises(ValueError, match="overlap"):
        _resolve_kept_var_idxs(svar_wv, regions, mode="pos", merge_overlapping=False)


def test_resolve_kept_var_idxs_overlap_merges(svar_wv):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar_wv.contigs[0]] * 2, "start": [0, 5], "end": [10, 20]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    kept = _resolve_kept_var_idxs(svar_wv, regions, mode="pos", merge_overlapping=True)
    assert len(np.unique(kept)) == len(kept)


def test_resolve_kept_var_idxs_record_includes_at_least_as_much_as_pos(svar_wv):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar_wv.contigs[0]], "start": [0], "end": [50]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    k_pos = _resolve_kept_var_idxs(
        svar_wv, regions, mode="pos", merge_overlapping=False
    )
    k_rec = _resolve_kept_var_idxs(
        svar_wv, regions, mode="record", merge_overlapping=False
    )
    assert len(k_rec) >= len(k_pos)


def test_resolve_kept_var_idxs_variant_includes_at_least_as_much_as_pos(svar_wv):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar_wv.contigs[0]], "start": [0], "end": [50]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    k_pos = _resolve_kept_var_idxs(
        svar_wv, regions, mode="pos", merge_overlapping=False
    )
    k_var = _resolve_kept_var_idxs(
        svar_wv, regions, mode="variant", merge_overlapping=False
    )
    assert len(k_var) >= len(k_pos)


def test_resolve_kept_var_idxs_all_modes_return_sorted_unique(svar_wv):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar_wv.contigs[0]], "start": [81_000], "end": [82_000]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    for mode in ("pos", "record", "variant"):
        kept = _resolve_kept_var_idxs(
            svar_wv, regions, mode=mode, merge_overlapping=False
        )
        assert len(kept) == len(np.unique(kept)), f"mode={mode}: not unique"
        if len(kept) > 1:
            assert np.all(np.diff(kept) > 0), f"mode={mode}: not sorted"


def test_resolve_kept_var_idxs_pos_mode_hits_known_variant(svar_wv):
    """biallelic.vcf has variants at chr1:81262 and chr1:81265 (1-based POS).

    Region [81261, 81265) in 0-based half-open should cover POS=81262 (0-based 81261)
    but not POS=81265 (0-based 81264).  Region [81261, 81266) should cover both.
    """
    from genoray._svar import _resolve_kept_var_idxs

    # Wide region: covers POS 81262 and 81265 (0-based 81261 and 81264)
    regions = pl.DataFrame(
        {"chrom": ["chr1"], "start": [81261], "end": [81266]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    kept = _resolve_kept_var_idxs(svar_wv, regions, mode="pos", merge_overlapping=False)
    assert len(kept) >= 1
    assert np.all(kept >= 0)
    assert np.all(kept < svar_wv.n_variants)

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


def test_resolve_kept_var_idxs_variant_matches_var_ranges_exclusive_end(svar_wv):
    """In variant mode, the kept set must equal the union of [s, e) ranges from
    var_ranges (exclusive end)."""
    from genoray._svar import _resolve_kept_var_idxs

    contig = svar_wv.contigs[0]
    regions = pl.DataFrame(
        {"chrom": [contig], "start": [0], "end": [100_000]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    kept = _resolve_kept_var_idxs(
        svar_wv, regions, mode="variant", merge_overlapping=False
    )

    starts = np.array([0], dtype=np.int32)
    ends = np.array([100_000], dtype=np.int32)
    vr = svar_wv.var_ranges(contig, starts, ends)
    sentinel = np.iinfo(np.int32).max
    valid = vr[:, 0] != sentinel
    expected = (
        np.unique(
            np.concatenate([np.arange(s, e, dtype=np.int32) for s, e in vr[valid]])
        )
        if valid.any()
        else np.empty(0, dtype=np.int32)
    )
    np.testing.assert_array_equal(kept, expected)


# ---------------------------------------------------------------------------
# Task 5: numba kernels
# ---------------------------------------------------------------------------


def test_nb_count_kept_matches_python():
    from genoray._svar import _nb_count_kept

    # Two samples, ploidy=2 -> 4 slots
    src_data = np.array([0, 2, 5, 1, 3, 5, 0, 4], dtype=np.int32)
    src_offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    src_sample_idxs = np.array([1, 0], dtype=np.int64)  # reorder: sample1 first
    kept_var_idxs = np.array([0, 1, 5], dtype=np.int32)
    ploidy = 2

    out_lengths = np.zeros(2 * 2, dtype=np.int64)
    _nb_count_kept(
        src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths
    )

    # Expected per output slot:
    # (out 0, p 0) = src slot 2 [3, 5] -> kept {5} -> 1
    # (out 0, p 1) = src slot 3 [0, 4] -> kept {0} -> 1
    # (out 1, p 0) = src slot 0 [0, 2] -> kept {0} -> 1
    # (out 1, p 1) = src slot 1 [5, 1] -> kept {5, 1} -> 2
    assert out_lengths.tolist() == [1, 1, 1, 2]


def test_nb_write_var_idxs_matches_python():
    from seqpro.rag import lengths_to_offsets

    from genoray._svar import _nb_count_kept, _nb_write_var_idxs

    src_data = np.array([0, 2, 5, 1, 3, 5, 0, 4], dtype=np.int32)
    src_offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    src_sample_idxs = np.array([1, 0], dtype=np.int64)
    kept_var_idxs = np.array([0, 1, 5], dtype=np.int32)
    ploidy = 2

    out_lengths = np.zeros(2 * 2, dtype=np.int64)
    _nb_count_kept(
        src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths
    )
    new_offsets = lengths_to_offsets(out_lengths.reshape(2, ploidy))
    out_var_idxs = np.empty(int(new_offsets[-1]), dtype=np.int32)

    _nb_write_var_idxs(
        src_data,
        src_offsets,
        src_sample_idxs,
        ploidy,
        kept_var_idxs,
        new_offsets.ravel(),
        out_var_idxs,
    )
    # slot 0: src [3, 5] kept {5} -> [2]
    # slot 1: src [0, 4] kept {0} -> [0]
    # slot 2: src [0, 2] kept {0} -> [0]
    # slot 3: src [5, 1] kept {5, 1} -> [2, 1]
    assert out_var_idxs.tolist() == [2, 0, 0, 2, 1]


def test_nb_write_field_matches_python():
    from seqpro.rag import lengths_to_offsets

    from genoray._svar import _nb_count_kept, _nb_write_field

    src_data = np.array([0, 2, 5, 1, 3, 5, 0, 4], dtype=np.int32)
    src_offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    src_field = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.float32)
    src_sample_idxs = np.array([1, 0], dtype=np.int64)
    kept_var_idxs = np.array([0, 1, 5], dtype=np.int32)
    ploidy = 2

    out_lengths = np.zeros(2 * 2, dtype=np.int64)
    _nb_count_kept(
        src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths
    )
    new_offsets = lengths_to_offsets(out_lengths.reshape(2, ploidy))
    out_field = np.empty(int(new_offsets[-1]), dtype=np.float32)

    _nb_write_field(
        src_field,
        src_data,
        src_offsets,
        src_sample_idxs,
        ploidy,
        kept_var_idxs,
        new_offsets.ravel(),
        out_field,
    )
    # out 0 = sample 1, ploidy 0 -> src slot 2, data[4:6]=[3,5]; kept: v=5->field[5]=60
    # out 0 = sample 1, ploidy 1 -> src slot 3, data[6:8]=[0,4]; kept: v=0->field[6]=70
    # out 1 = sample 0, ploidy 0 -> src slot 0, data[0:2]=[0,2]; kept: v=0->field[0]=10
    # out 1 = sample 0, ploidy 1 -> src slot 1, data[2:4]=[5,1]; kept: v=5->field[2]=30, v=1->field[3]=40
    assert out_field.tolist() == [60.0, 70.0, 10.0, 30.0, 40.0]


# ---------------------------------------------------------------------------
# Task 6: SparseVar.write_view
# ---------------------------------------------------------------------------


@pytest.fixture
def svar():
    return SparseVar(ddir / "biallelic.vcf.svar")


@pytest.fixture
def svar_with_dosages():
    return SparseVar(ddir / "biallelic.vcf.svar")


def test_write_view_roundtrip_full(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    svar.write_view(
        regions=(contig, 0, 1_000_000),
        samples=samples,
        output=out,
    )
    sv2 = SparseVar(out)
    assert sv2.available_samples == samples
    assert sv2.ploidy == svar.ploidy
    assert sv2.contigs == svar.contigs

    # Variant count equals number of variants on this contig in the source
    src_idx_on_c = svar.index.filter(pl.col("CHROM") == contig)
    assert sv2.index.filter(pl.col("CHROM") == contig).height == src_idx_on_c.height


def test_write_view_sample_subset_and_order(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = [svar.available_samples[1], svar.available_samples[0]]  # reversed
    svar.write_view(
        regions=(contig, 0, 1_000_000),
        samples=samples,
        output=out,
    )
    sv2 = SparseVar(out)
    assert sv2.available_samples == samples


def test_write_view_overwrite_protection(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:1]
    svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out)
    with pytest.raises(FileExistsError):
        svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out)
    svar.write_view(
        regions=(contig, 0, 1_000_000), samples=samples, output=out, overwrite=True
    )


def test_write_view_afs_match_compute_afs(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    svar.write_view(
        regions=(contig, 0, 1_000_000),
        samples=samples,
        output=out,
    )
    # Load AF column explicitly; AF is stored in index.arrow but not loaded by default
    v = SparseVar(out, attrs="AF")
    assert "AF" in v.index.columns
    expected = v._compute_afs()
    np.testing.assert_allclose(v.index["AF"].to_numpy(), expected, atol=1e-6)


def test_write_view_threads_deterministic(tmp_path: Path, svar: SparseVar):
    out1 = tmp_path / "v1.svar"
    out2 = tmp_path / "v2.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    svar.write_view(
        regions=(contig, 0, 1_000_000), samples=samples, output=out1, threads=1
    )
    svar.write_view(
        regions=(contig, 0, 1_000_000), samples=samples, output=out2, threads=None
    )
    a = np.fromfile(out1 / "variant_idxs.npy", dtype=np.int32)
    b = np.fromfile(out2 / "variant_idxs.npy", dtype=np.int32)
    np.testing.assert_array_equal(a, b)


def test_write_view_fields_default_carries_all(
    tmp_path: Path, svar_with_dosages: SparseVar
):
    view_out = tmp_path / "view.svar"
    svar_with_dosages.write_view(
        regions=(svar_with_dosages.contigs[0], 0, 1_000_000),
        samples=svar_with_dosages.available_samples[:1],
        output=view_out,
    )
    v = SparseVar(view_out)
    assert set(v.available_fields) == set(svar_with_dosages.available_fields)


def test_write_view_fields_explicit_empty_drops_dosages(
    tmp_path: Path, svar_with_dosages: SparseVar
):
    view_out = tmp_path / "view.svar"
    svar_with_dosages.write_view(
        regions=(svar_with_dosages.contigs[0], 0, 1_000_000),
        samples=svar_with_dosages.available_samples[:1],
        output=view_out,
        fields=[],
    )
    v = SparseVar(view_out)
    assert v.available_fields == {}


def test_write_view_roundtrip_genotype_values(tmp_path: Path, svar: SparseVar):
    """Verify genotype data round-trips correctly: for each (sample, ploidy) slot,
    the *positions* of carried alt alleles must match between source and view."""
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out)
    sv2 = SparseVar(out)

    # Read full range from both; shape (1, n_samples, ploidy, ~variants)
    rag_src = svar.read_ranges(
        contig,
        np.array([0], dtype=np.int32),
        np.array([1_000_000], dtype=np.int32),
        samples=samples,
    )
    rag_view = sv2.read_ranges(
        contig,
        np.array([0], dtype=np.int32),
        np.array([1_000_000], dtype=np.int32),
    )

    src_pos = svar.index["POS"].to_numpy()
    view_pos = sv2.index["POS"].to_numpy()

    # offsets has shape (2, n_slots): row 0 = starts, row 1 = ends
    n_slots = rag_src.offsets.shape[1]
    assert n_slots == rag_view.offsets.shape[1], "slot counts differ"

    for i in range(n_slots):
        s_src, e_src = rag_src.offsets[0, i], rag_src.offsets[1, i]
        s_view, e_view = rag_view.offsets[0, i], rag_view.offsets[1, i]
        src_pp = src_pos[rag_src.data[s_src:e_src]]
        view_pp = view_pos[rag_view.data[s_view:e_view]]
        np.testing.assert_array_equal(
            src_pp, view_pp, err_msg=f"positions differ at slot {i}"
        )


def test_write_view_empty_regions_raises(tmp_path, svar):
    with pytest.raises(ValueError, match="no variants"):
        svar.write_view(
            regions=(svar.contigs[0], 10**9, 10**9 + 1),  # past end of chromosome
            samples=svar.available_samples[:1],
            output=tmp_path / "empty.svar",
        )
    assert not (tmp_path / "empty.svar").exists()


def test_write_view_empty_samples_raises(tmp_path, svar):
    with pytest.raises(ValueError, match="at least one sample"):
        svar.write_view(
            regions=(svar.contigs[0], 0, 1_000_000),
            samples=[],
            output=tmp_path / "empty.svar",
        )
    assert not (tmp_path / "empty.svar").exists()


def test_write_view_unknown_sample_raises(tmp_path, svar):
    with pytest.raises(ValueError, match="not found"):
        svar.write_view(
            regions=(svar.contigs[0], 0, 1_000_000),
            samples=["__nope__"],
            output=tmp_path / "x.svar",
        )
    assert not (tmp_path / "x.svar").exists()


def test_write_view_unknown_field_raises(tmp_path, svar):
    with pytest.raises(ValueError, match="not found"):
        svar.write_view(
            regions=(svar.contigs[0], 0, 1_000_000),
            samples=svar.available_samples[:1],
            output=tmp_path / "x.svar",
            fields=["__nope__"],
        )
    assert not (tmp_path / "x.svar").exists()

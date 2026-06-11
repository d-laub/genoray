from pathlib import Path

import numpy as np
import polars as pl
from seqpro.rag import lengths_to_offsets

import shutil
import subprocess

import pytest

from genoray import PGEN, VCF, SparseVar
from genoray._svar import (
    V_IDX_TYPE,
    _build_working_index,
    _resolve_kept_rows,
    _subset_var_idxs_and_recompute_af,
)
from genoray._utils import ContigNormalizer


def _index_df(rows, index_values=None):
    """Build a minimal working-index frame: CHROM, POS, ILEN(list[int]), index(row id).

    If *index_values* is provided those values are used for the ``index`` column
    instead of the default ``with_row_index`` sequential assignment.  This lets
    tests construct frames where the ``index`` values do NOT equal row positions,
    which is the realistic SVAR scenario (subset of a larger index).
    """
    df = pl.DataFrame(
        {
            "CHROM": [r[0] for r in rows],
            "POS": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "ILEN": [[0] for _ in rows],
        }
    )
    if index_values is None:
        return df.with_row_index("index")
    return df.with_columns(pl.Series("index", index_values, dtype=pl.UInt32))


def test_resolve_kept_rows_pos_mode():
    # variants at 1-based POS 10, 20, 30 on chr1
    df = _index_df([("chr1", 10), ("chr1", 20), ("chr1", 30)])
    cnorm = ContigNormalizer(["chr1"])
    # region 0-based [9, 21) covers POS 10 (0-based 9) and POS 20 (0-based 19)
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": pl.Series([9], dtype=pl.Int32),
            "end": pl.Series([21], dtype=pl.Int32),
        }
    )
    kept = _resolve_kept_rows(df, cnorm, regions, "pos", merge_overlapping=False)
    assert kept.tolist() == [0, 1]


def test_resolve_kept_rows_non_positional_index():
    """index column values that do NOT equal row positions must be returned.

    The frame represents three variants drawn from a larger SVAR whose global ids
    are 10, 11, 12 (not 0, 1, 2).  A query covering POS 10 and 20 (0-based [9, 21))
    must return [10, 11] — the *index column values* — not [0, 1] (row positions).

    If someone reverted _resolve_kept_rows to use positional indexing (e.g.
    ``candidates[...]`` row offsets instead of ``is_in`` + ``sort`` on the index
    column) this test would return [0, 1] instead of [10, 11] and fail.
    """
    # Variants with global SVAR ids 10, 11, 12 (not 0-based row offsets).
    df = _index_df(
        [("chr1", 10), ("chr1", 20), ("chr1", 30)],
        index_values=[10, 11, 12],
    )
    cnorm = ContigNormalizer(["chr1"])
    # region 0-based [9, 21) covers 1-based POS 10 and 20
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": pl.Series([9], dtype=pl.Int32),
            "end": pl.Series([21], dtype=pl.Int32),
        }
    )
    kept = _resolve_kept_rows(df, cnorm, regions, "pos", merge_overlapping=False)
    # Must be global ids, not row positions.
    assert kept.tolist() == [10, 11], f"expected [10, 11], got {kept.tolist()}"


def test_resolve_kept_rows_variant_mode():
    """variant mode returns every candidate that *overlaps* the region (ILEN-aware).

    A deletion at POS 5 with ILEN=-5 spans 0-based [4, 9).  Even though its POS-1=4
    is outside the query region [6, 15), the deletion's end (9) extends into it.
    pos mode would exclude it; variant mode must include it.

    The non-contiguous index values ([20, 21, 22]) confirm that the returned ids are
    index-column values, not row positions, even in variant mode.
    """
    # Deletion at POS 5, ILEN=-5 → 0-based span [4, 9); overlaps query [6, 15).
    # SNP at POS 10, ILEN=0  → 0-based span [9, 10); inside query [6, 15).
    # SNP at POS 20, ILEN=0  → 0-based span [19, 20); outside query [6, 15).
    df = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1"],
            "POS": pl.Series([5, 10, 20], dtype=pl.Int32),
            "ILEN": [[-5], [0], [0]],
            "index": pl.Series([20, 21, 22], dtype=pl.UInt32),
        }
    )
    cnorm = ContigNormalizer(["chr1"])
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": pl.Series([6], dtype=pl.Int32),
            "end": pl.Series([15], dtype=pl.Int32),
        }
    )
    # variant mode: deletion (id=20) overlaps via span; SNP at POS 10 (id=21) overlaps.
    kept_variant = _resolve_kept_rows(
        df, cnorm, regions, "variant", merge_overlapping=False
    )
    assert 20 in kept_variant.tolist(), (
        "deletion spanning into region must be kept in variant mode"
    )
    assert 21 in kept_variant.tolist(), "SNP inside region must be kept in variant mode"
    assert 22 not in kept_variant.tolist(), "SNP outside region must not be kept"

    # pos mode: deletion POS-1=4 is outside [6, 15), so id=20 must be excluded.
    kept_pos = _resolve_kept_rows(df, cnorm, regions, "pos", merge_overlapping=False)
    assert 20 not in kept_pos.tolist(), (
        "deletion POS outside region must be excluded in pos mode"
    )
    assert 21 in kept_pos.tolist(), "SNP inside region must be kept in pos mode"


def _make_svar_from_vcf(tmp_path: Path, vcf_path: str) -> Path:
    out = tmp_path / "full.svar"
    SparseVar.from_vcf(out, VCF(vcf_path), max_mem="1g", overwrite=True)
    return out


def test_build_working_index_has_required_columns(tmp_path):
    sv_path = _make_svar_from_vcf(tmp_path, "tests/data/biallelic.vcf.gz")
    df, alt_is_utf8, ilen_added = _build_working_index(
        SparseVar._index_path(sv_path), None
    )
    assert {"CHROM", "POS", "ILEN", "index"} <= set(df.columns)
    assert df["index"].to_list() == list(range(df.height))
    # ALT present as list[str] for filtering
    assert df.schema["ALT"] == pl.List(pl.Utf8)


def _read_all(sv: SparseVar):
    """Return per-(sample,ploidy) sets of (CHROM, POS) for deep comparison."""
    idx = sv.index
    chrom = idx["CHROM"].to_list()
    pos = idx["POS"].to_list()
    return sv, [(c, p) for c, p in zip(chrom, pos)]


def test_from_vcf_regions_only_matches_write_view(tmp_path):
    vcf_path = "tests/data/biallelic.vcf.gz"
    full = _make_svar_from_vcf(tmp_path, vcf_path)
    sv_full = SparseVar(full)
    contig = sv_full.contigs[0]
    region = (contig, 0, 10_000_000)

    # Convert-time subset
    direct = tmp_path / "direct.svar"
    SparseVar.from_vcf(
        direct, VCF(vcf_path), max_mem="1g", overwrite=True, regions=region
    )
    sv_direct = SparseVar(direct)

    # Post-hoc view over the same region, all samples
    view = tmp_path / "view.svar"
    sv_full.write_view(
        regions=region, samples=list(sv_full.available_samples), output=view
    )
    sv_view = SparseVar(view)

    assert sv_direct.index["POS"].to_list() == sv_view.index["POS"].to_list()
    assert sv_direct.available_samples == list(sv_full.available_samples)


def test_from_vcf_no_subset_unchanged(tmp_path):
    vcf_path = "tests/data/biallelic.vcf.gz"
    a = tmp_path / "a.svar"
    b = tmp_path / "b.svar"
    SparseVar.from_vcf(a, VCF(vcf_path), max_mem="1g", overwrite=True)
    SparseVar.from_vcf(
        b, VCF(vcf_path), max_mem="1g", overwrite=True, regions=None, samples=None
    )
    sa, sb = SparseVar(a), SparseVar(b)
    assert sa.index["POS"].to_list() == sb.index["POS"].to_list()
    assert sa.available_samples == sb.available_samples
    assert sa.n_variants == sb.n_variants


def test_from_vcf_samples_subset_matches_write_view(tmp_path):
    # biallelic.vcf.gz has 3 contigs (chr1, chr2, chr3).  write_view requires an
    # explicit regions arg, so we build a whole-genome BED frame covering all
    # contigs.  Both sides must restrict to MAC>0 in the kept-sample set, so the
    # equivalence assertion also verifies that the MAC-drop path fires.
    #
    # NOTE: the biallelic fixture has 2 samples (sample1, sample2) and 6 variants
    # total; 2 of those 6 have MAC=0 in sample1, so the MAC-drop path IS genuinely
    # exercised here (confirmed by inspection: variants at index 1 and 3 are
    # ref-hom in sample1).
    vcf_path = "tests/data/biallelic.vcf.gz"
    full = _make_svar_from_vcf(tmp_path, vcf_path)
    sv_full = SparseVar(full)
    keep_samples = list(sv_full.available_samples)[:1]

    direct = tmp_path / "direct.svar"
    SparseVar.from_vcf(
        direct, VCF(vcf_path), max_mem="1g", overwrite=True, samples=keep_samples
    )
    # Open with attrs="AF" so the on-disk AF column is loaded into .index
    sv_direct = SparseVar(direct, attrs="AF")

    # Oracle: write_view over all contigs (whole-genome BED) with same samples.
    whole_genome = pl.DataFrame(
        {
            "chrom": sv_full.contigs,
            "start": pl.Series([0] * len(sv_full.contigs), dtype=pl.Int32),
            "end": pl.Series([1_000_000_000] * len(sv_full.contigs), dtype=pl.Int32),
        }
    )
    view = tmp_path / "view.svar"
    sv_full.write_view(regions=whole_genome, samples=keep_samples, output=view)
    sv_view = SparseVar(view)

    assert sv_direct.available_samples == keep_samples
    assert sv_direct.index["CHROM"].to_list() == sv_view.index["CHROM"].to_list(), (
        f"CHROM mismatch: {sv_direct.index['CHROM'].to_list()} vs {sv_view.index['CHROM'].to_list()}"
    )
    assert sv_direct.index["POS"].to_list() == sv_view.index["POS"].to_list(), (
        f"POS mismatch: {sv_direct.index['POS'].to_list()} vs {sv_view.index['POS'].to_list()}"
    )
    # MAC>0 invariant: AF must be strictly positive for every surviving variant
    assert (sv_direct.index["AF"] > 0).all(), (
        f"AF must be >0 for all variants after MAC-drop; got {sv_direct.index['AF'].to_list()}"
    )
    # Confirm MAC-drop actually happened: direct must have fewer variants than the full SVAR
    assert sv_direct.n_variants < sv_full.n_variants, (
        "Expected MAC=0 variants to be dropped; none were"
    )


def test_from_vcf_regions_and_samples(tmp_path):
    # Combined regions + samples subset — both sides must agree on POS list.
    vcf_path = "tests/data/biallelic.vcf.gz"
    full = _make_svar_from_vcf(tmp_path, vcf_path)
    sv_full = SparseVar(full)
    contig = sv_full.contigs[0]
    keep_samples = list(sv_full.available_samples)[:1]
    region = (contig, 0, 10_000_000)

    direct = tmp_path / "d.svar"
    SparseVar.from_vcf(
        direct,
        VCF(vcf_path),
        max_mem="1g",
        overwrite=True,
        regions=region,
        samples=keep_samples,
    )
    view = tmp_path / "v.svar"
    sv_full.write_view(regions=region, samples=keep_samples, output=view)

    sv_direct = SparseVar(direct, attrs="AF")
    sv_view = SparseVar(view)
    assert sv_direct.available_samples == keep_samples
    assert sv_direct.index["POS"].to_list() == sv_view.index["POS"].to_list(), (
        f"POS mismatch: {sv_direct.index['POS'].to_list()} vs {sv_view.index['POS'].to_list()}"
    )
    assert (sv_direct.index["AF"] > 0).all()


def test_subset_var_idxs_drops_mac_zero(tmp_path):
    # 1 sample, ploidy 2, 3 candidate variants (ids 0,1,2). Variant 1 has MAC 0.
    out = tmp_path / "v.svar"
    out.mkdir()
    # slot 0 has variant 0; slot 1 has variant 2; variant 1 never appears.
    data = np.array([0, 2], dtype=V_IDX_TYPE)
    lengths = np.array([[1, 1]], dtype=np.int64)  # (n_samples=1, ploidy=2)
    offsets = lengths_to_offsets(lengths)
    np.memmap(out / "variant_idxs.npy", dtype=V_IDX_TYPE, mode="w+", shape=data.shape)[
        :
    ] = data
    np.memmap(out / "offsets.npy", dtype=offsets.dtype, mode="w+", shape=offsets.shape)[
        :
    ] = offsets

    survivors, af = _subset_var_idxs_and_recompute_af(
        out, n_total=3, n_out=1, ploidy=2, with_dosages=False
    )
    assert survivors.tolist() == [0, 2]  # variant 1 dropped
    # remapped ids: 0 -> 0, 2 -> 1
    vi = np.memmap(out / "variant_idxs.npy", dtype=V_IDX_TYPE, mode="r")
    assert sorted(vi.tolist()) == [0, 1]
    # AF over n_out*ploidy = 2 alleles: each surviving variant present once
    assert np.allclose(af, [0.5, 0.5])


# ---------------------------------------------------------------------------
# PGEN subsetting tests (guarded on plink2 availability)
# ---------------------------------------------------------------------------

VCF_FOR_PGEN = "tests/data/biallelic.vcf"


def _vcf_to_pgen(tmp_path: Path, vcf_path: str) -> Path:
    if shutil.which("plink2") is None:
        pytest.skip("plink2 not available")
    prefix = tmp_path / "conv"
    subprocess.run(
        [
            "plink2",
            "--vcf",
            str(vcf_path),
            "--make-pgen",
            "--out",
            str(prefix),
            "--allow-extra-chr",
            "--vcf-half-call",
            "haploid",
            # biallelic.vcf contains a ./1 half-call that plink2 rejects under its
            # default "error" mode; "haploid" treats the non-missing allele as
            # homozygous, making this flag load-bearing for the fixture conversion.
        ],
        check=True,
        capture_output=True,
    )
    return prefix.with_suffix(".pgen")


def test_from_pgen_regions_and_samples(tmp_path: Path):
    """from_pgen with regions+samples must produce the same POS list as write_view,
    and AF must be >0 for all survivors (MAC-drop fired)."""
    pgen_path = _vcf_to_pgen(tmp_path, VCF_FOR_PGEN)

    full = tmp_path / "full.svar"
    SparseVar.from_pgen(full, PGEN(pgen_path), max_mem="1g", overwrite=True)
    sv_full = SparseVar(full)
    contig = sv_full.contigs[0]
    keep_samples = list(sv_full.available_samples)[:1]
    region = (contig, 0, 10_000_000)

    direct = tmp_path / "d.svar"
    SparseVar.from_pgen(
        direct,
        PGEN(pgen_path),
        max_mem="1g",
        overwrite=True,
        regions=region,
        samples=keep_samples,
    )
    view = tmp_path / "v.svar"
    sv_full.write_view(regions=region, samples=keep_samples, output=view)

    sv_d = SparseVar(direct, attrs="AF")
    sv_v = SparseVar(view)
    assert sv_d.available_samples == keep_samples
    assert sv_d.index["POS"].to_list() == sv_v.index["POS"].to_list(), (
        f"POS mismatch: {sv_d.index['POS'].to_list()} vs {sv_v.index['POS'].to_list()}"
    )
    assert (sv_d.index["AF"] > 0).all(), (
        f"AF must be >0 for all variants after MAC-drop; got {sv_d.index['AF'].to_list()}"
    )


def test_from_pgen_no_subset_unchanged(tmp_path: Path):
    """from_pgen with no regions/samples must produce the same output as before."""
    pgen_path = _vcf_to_pgen(tmp_path, VCF_FOR_PGEN)

    a = tmp_path / "a.svar"
    b = tmp_path / "b.svar"
    SparseVar.from_pgen(a, PGEN(pgen_path), max_mem="1g", overwrite=True)
    SparseVar.from_pgen(
        b, PGEN(pgen_path), max_mem="1g", overwrite=True, regions=None, samples=None
    )
    sa, sb = SparseVar(a), SparseVar(b)
    assert sa.index["POS"].to_list() == sb.index["POS"].to_list()
    assert sa.available_samples == sb.available_samples
    assert sa.n_variants == sb.n_variants


def test_from_pgen_permuted_samples_genotype_order(tmp_path: Path):
    """Verify that sample-order scrambling by pgenlib's sorted requirement is
    correctly reversed.  Selects >=2 samples in REVERSED order and compares
    per-sample genotype arrays between from_pgen and write_view (the oracle).

    This test specifically guards the sort/unsort correctness fix in
    _process_contig_pgen: if change_sample_subset is called without restoring
    caller order, the genotypes for sample[0] and sample[1] will be swapped.
    """
    pgen_path = _vcf_to_pgen(tmp_path, VCF_FOR_PGEN)

    full = tmp_path / "full.svar"
    SparseVar.from_pgen(full, PGEN(pgen_path), max_mem="1g", overwrite=True)
    sv_full = SparseVar(full)

    all_samples = list(sv_full.available_samples)
    if len(all_samples) < 2:
        pytest.skip("need >=2 samples to test permuted order")

    # Reverse the sample order to guarantee a non-identity permutation
    permuted = list(reversed(all_samples))
    assert permuted != all_samples, "permuted == original (unexpected for >1 sample)"

    # --- direct conversion with permuted sample order ---
    direct = tmp_path / "d.svar"
    SparseVar.from_pgen(
        direct,
        PGEN(pgen_path),
        max_mem="1g",
        overwrite=True,
        samples=permuted,
    )

    # --- oracle: write_view over same permuted samples (whole genome) ---
    whole_genome = pl.DataFrame(
        {
            "chrom": sv_full.contigs,
            "start": pl.Series([0] * len(sv_full.contigs), dtype=pl.Int32),
            "end": pl.Series([1_000_000_000] * len(sv_full.contigs), dtype=pl.Int32),
        }
    )
    view = tmp_path / "v.svar"
    sv_full.write_view(regions=whole_genome, samples=permuted, output=view)

    sv_d = SparseVar(direct)
    sv_v = SparseVar(view)

    # Sample list must match the caller-supplied permuted order
    assert sv_d.available_samples == permuted, (
        f"available_samples mismatch: {sv_d.available_samples} != {permuted}"
    )
    assert sv_v.available_samples == permuted

    # POS lists must agree (after MAC-drop both sides apply the same filter)
    assert sv_d.index["POS"].to_list() == sv_v.index["POS"].to_list(), (
        f"POS mismatch after permuted subsetting: "
        f"{sv_d.index['POS'].to_list()} vs {sv_v.index['POS'].to_list()}"
    )

    # Per-sample genotype comparison across all contigs.
    # read_ranges returns Ragged[V_IDX_TYPE] with shape (ranges, samples, ploidy, ~variants).
    # We compare the raw variant-index arrays: they encode which variants are non-ref
    # for each (sample, ploidy) slot, so equality implies same genotype data.
    for contig in sv_d.contigs:
        r_d = sv_d.read_ranges(contig, 0, 2_000_000_000, samples=permuted)
        r_v = sv_v.read_ranges(contig, 0, 2_000_000_000, samples=permuted)
        # r_d / r_v shape: (1 range, n_samples, ploidy, ~variants)
        for s_i, sname in enumerate(permuted):
            for p_i in range(sv_d.ploidy):
                arr_d = r_d[0, s_i, p_i].to_numpy()
                arr_v = r_v[0, s_i, p_i].to_numpy()
                assert np.array_equal(arr_d, arr_v), (
                    f"Genotype mismatch for sample {sname!r} ploidy {p_i} "
                    f"on contig {contig!r}: direct={arr_d} vs view={arr_v}"
                )


# ---------------------------------------------------------------------------
# Error-path tests (Task 7.1)
# ---------------------------------------------------------------------------


def test_from_vcf_regions_no_match_raises(tmp_path):
    """A region that overlaps no variants must raise with 'no variants selected'."""
    with pytest.raises(ValueError, match="no variants selected by `regions`"):
        SparseVar.from_vcf(
            tmp_path / "x.svar",
            VCF("tests/data/biallelic.vcf.gz"),
            max_mem="1g",
            overwrite=True,
            regions=("chr1", 999_000_000, 999_000_100),
        )


def test_from_vcf_overlapping_regions_raise(tmp_path):
    """Overlapping regions without merge_overlapping=True must raise."""
    regions = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": pl.Series([0, 50], dtype=pl.Int32),
            "end": pl.Series([100, 200], dtype=pl.Int32),
        }
    )
    with pytest.raises(ValueError, match="regions overlap"):
        SparseVar.from_vcf(
            tmp_path / "x.svar",
            VCF("tests/data/biallelic.vcf.gz"),
            max_mem="1g",
            overwrite=True,
            regions=regions,
            merge_overlapping=False,
        )


def test_from_vcf_unknown_sample_raises(tmp_path):
    """A sample name not present in the VCF must raise with 'not found'."""
    with pytest.raises(ValueError, match="not found"):
        SparseVar.from_vcf(
            tmp_path / "x.svar",
            VCF("tests/data/biallelic.vcf.gz"),
            max_mem="1g",
            overwrite=True,
            samples=["NOT_A_SAMPLE"],
        )


def test_from_vcf_empty_samples_raises(tmp_path):
    """An empty samples list must raise with 'selected no samples'."""
    with pytest.raises(ValueError, match="selected no samples"):
        SparseVar.from_vcf(
            tmp_path / "x.svar",
            VCF("tests/data/biallelic.vcf.gz"),
            max_mem="1g",
            overwrite=True,
            samples=[],
        )


# ---------------------------------------------------------------------------
# Dosage subset test (Task 7.2)
# ---------------------------------------------------------------------------


def test_from_vcf_subset_with_dosages(tmp_path):
    """Conversion-time region+sample subset with dosages must match write_view.

    biallelic.vcf.gz has a DS FORMAT field (Number=A, Float).  Both direct
    (from_vcf subsetting) and oracle (write_view on a full SVAR) are built with
    with_dosages=True.  The test then compares:
      1. POS lists (variant identity)
      2. Raw dosage data arrays read via with_fields(["dosages"]) on both sides

    This exercises the MAC-drop path: sample1 is ref-hom for some variants, so
    _subset_var_idxs_and_recompute_af must drop them, and the dosage mmap must
    remain coherent after remapping.
    """
    vcf_path = "tests/data/biallelic.vcf.gz"

    # Build a full SVAR with dosages to derive oracle parameters
    full = tmp_path / "full.svar"
    SparseVar.from_vcf(
        full,
        VCF(vcf_path, dosage_field="DS"),
        max_mem="1g",
        overwrite=True,
        with_dosages=True,
    )
    sv_full = SparseVar(full)
    keep_samples = list(sv_full.available_samples)[:1]  # ["sample1"]
    contig = sv_full.contigs[0]  # "chr1"

    # Direct: convert-time subset (regions + samples + dosages)
    direct = tmp_path / "d.svar"
    SparseVar.from_vcf(
        direct,
        VCF(vcf_path, dosage_field="DS"),
        max_mem="1g",
        overwrite=True,
        with_dosages=True,
        regions=(contig, 0, 10_000_000),
        samples=keep_samples,
    )

    # Oracle: write_view on the full SVAR
    view = tmp_path / "v.svar"
    sv_full.write_view(
        regions=(contig, 0, 10_000_000), samples=keep_samples, output=view
    )

    sv_d = SparseVar(direct, fields=["dosages"])
    sv_v = SparseVar(view, fields=["dosages"])

    # 1. POS lists must match
    assert sv_d.index["POS"].to_list() == sv_v.index["POS"].to_list(), (
        f"POS mismatch: {sv_d.index['POS'].to_list()} vs {sv_v.index['POS'].to_list()}"
    )

    # 2. Both sides must have at least one surviving variant (MAC-drop may fire for
    #    sample1 on some variants, but chr1 has variants where sample1 is non-ref)
    assert sv_d.n_variants > 0, "expected at least one MAC>0 variant after subset"

    # 3. Real dosage value comparison via read_ranges across the full contig
    result_d = sv_d.read_ranges(contig, 0, 10_000_000, samples=keep_samples)
    result_v = sv_v.read_ranges(contig, 0, 10_000_000, samples=keep_samples)

    # result is an awkward record with .genos and .dosages fields
    for p_i in range(sv_d.ploidy):
        dosages_d = result_d[0, 0, p_i].dosages.to_numpy()
        dosages_v = result_v[0, 0, p_i].dosages.to_numpy()
        assert dosages_d.dtype == np.float32
        np.testing.assert_allclose(
            dosages_d,
            dosages_v,
            atol=1e-5,
            err_msg=(
                f"Dosage mismatch for sample {keep_samples[0]!r} ploidy {p_i}: "
                f"direct={dosages_d} vs view={dosages_v}"
            ),
        )

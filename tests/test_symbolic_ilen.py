from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genoray import PGEN, VCF, SparseVar, exprs
from genoray.exprs import is_imprecise, symbolic_ilen
from tests import _oracle
from tests.data.fixtures import FIXTURES as _FIXTURES


def _frame():
    # one row per scenario; SVLEN/END/IMPRECISE pre-extracted as scalars
    return pl.DataFrame(
        {
            "REF": ["G", "G", "G", "G", "A", "G"],
            "ALT": [["<DEL>"], ["<INS>"], ["<DUP>"], ["<DEL>"], ["AT"], ["<BND>"]],
            "SVLEN": [100, 50, 30, None, None, None],
            "END": [200, None, 130, None, None, None],
            "IMPRECISE": [False, False, False, True, False, False],
            "POS": [1000, 2000, 3000, 4000, 5000, 6000],
        }
    )


def test_symbolic_ilen_precise_and_unsizable():
    out = _frame().with_columns(ILEN=symbolic_ilen()).get_column("ILEN").to_list()
    assert out[0] == [-100]  # <DEL> SVLEN=100
    assert out[1] == [50]  # <INS> SVLEN=50
    assert out[2] == [30]  # <DUP> SVLEN=30
    assert out[3] == [None]  # IMPRECISE <DEL> -> null
    assert out[4] == [1]  # literal insertion AT vs A -> +1
    assert out[5] == [None]  # <BND> unsupported -> null


def test_is_imprecise_flags_only_unsizable():
    df = _frame().with_columns(ILEN=symbolic_ilen()).with_columns(imp=is_imprecise)
    assert df.get_column("imp").to_list() == [False, False, False, True, False, True]


def test_mixed_literal_symbolic_alt():
    """Regression: a single record whose ALT mixes a literal and a symbolic allele.

    REF="G", ALT=["C", "<DEL>"], SVLEN=80 -> ILEN=[0, -80].
    Locks the per-element-vector + per-row-scalar broadcast that motivates
    the map_elements design.
    """
    df = pl.DataFrame(
        {
            "REF": ["G"],
            "ALT": [["C", "<DEL>"]],
            "SVLEN": [80],
            "END": [None],
            "IMPRECISE": [False],
            "POS": [1000],
        }
    )
    out = df.with_columns(ILEN=symbolic_ilen()).get_column("ILEN").to_list()
    assert out[0] == [0, -80]


def test_is_snp_is_indel_null_ilen_excluded():
    """Regression: null-ILEN symbolic SVs must NOT be classified as SNP or indel.

    Pre-fix, list.all() ignores nulls so [null] -> is_snp=True AND is_indel=True.
    Post-fix, the null-aware predicate returns False for both.
    """
    df = pl.DataFrame(
        {
            "ILEN": [
                [0],  # SNP
                [-100],  # DEL (indel)
                [None],  # un-sizable symbolic SV
                [50],  # INS (indel)
                [0, None],  # multiallelic: SNP + un-sizable -> neither
            ]
        },
        schema={"ILEN": pl.List(pl.Int32)},
    ).with_columns(
        snp=pl.col("ILEN")
        .list.eval((pl.element() == 0) & pl.element().is_not_null())
        .list.all(),
        indel=pl.col("ILEN")
        .list.eval((pl.element() != 0) & pl.element().is_not_null())
        .list.all(),
    )
    snp = df["snp"].to_list()
    indel = df["indel"].to_list()
    # SNP row
    assert snp[0] is True and indel[0] is False
    # DEL row
    assert snp[1] is False and indel[1] is True
    # null-ILEN row: must be NEITHER snp nor indel
    assert snp[2] is False, f"null-ILEN row wrongly classified as SNP: {snp[2]}"
    assert indel[2] is False, f"null-ILEN row wrongly classified as indel: {indel[2]}"
    # INS row
    assert snp[3] is False and indel[3] is True
    # multiallelic SNP + null: null poisons the list -> neither
    assert snp[4] is False and indel[4] is False


def test_symbolic_ilen_end_fallback_no_svlen():
    """Unit test: END-based fallback when SVLEN is absent (null).

    Pins the |END - POS| convention used by symbolic_ilen() when coalesce()
    falls back from SVLEN to END.  Every fixture row that has END also has
    SVLEN, so this path was previously uncovered end-to-end.

    Convention (0-based POS):
      <DEL>  POS=1000, END=1100, SVLEN=null  -> ILEN = -(1100-1000) = -100
      <DUP>  POS=2000, END=2030, SVLEN=null  -> ILEN = +(2030-2000) = +30
    """
    df = pl.DataFrame(
        {
            "REF": ["G", "G"],
            "ALT": [["<DEL>"], ["<DUP>"]],
            "SVLEN": [None, None],
            "END": [1100, 2030],
            "IMPRECISE": [False, False],
            "POS": [1000, 2000],
        },
        schema={
            "REF": pl.Utf8,
            "ALT": pl.List(pl.Utf8),
            "SVLEN": pl.Int64,
            "END": pl.Int64,
            "IMPRECISE": pl.Boolean,
            "POS": pl.Int64,
        },
    )
    out = df.with_columns(ILEN=symbolic_ilen()).get_column("ILEN").to_list()
    assert out[0] == [-100], f"<DEL> END-fallback: expected [-100], got {out[0]}"
    assert out[1] == [30], f"<DUP> END-fallback: expected [30], got {out[1]}"


def test_symbolic_fixture_builds_and_classifies():
    from tests.data.fixtures import FIXTURES

    truth = FIXTURES["symbolic"]().truth()
    # records: 0 <DEL> precise, 1 <INS> precise, 2 <DUP> precise,
    #          3 <DEL> IMPRECISE, 4 <CNV> (unsupported), 5 <INV> (unsupported)
    assert len(truth.pos) == 6
    assert truth.alts_truth[0][0].sv_type == "DEL"
    assert truth.alts_truth[1][0].sv_type == "INS"
    assert truth.alts_truth[2][0].sv_type == "DUP"
    assert truth.alts_truth[3][0].sv_type == "DEL"  # IMPRECISE <DEL>
    assert truth.alts_truth[4][0].sv_type == "CNV"  # <CNV> unsupported type
    assert truth.alts_truth[5][0].sv_type == "INV"  # <INV> unsupported type


def test_expected_ilen_from_oracle():
    from tests.data.fixtures import FIXTURES

    truth = FIXTURES["symbolic"]().truth()
    exp = _oracle.expected_ilen(truth, slice(None))
    assert exp[0] == [-100]  # <DEL>
    assert exp[1] == [50]  # <INS>
    assert exp[2] == [30]  # <DUP>
    assert exp[3] == [None]  # IMPRECISE <DEL> -> null (mirrors symbolic_ilen)
    assert exp[4] == [None]  # <CNV> unsupported
    assert exp[5] == [None]  # <INV> unsupported


@pytest.fixture
def symbolic_vcf(tmp_path):
    path = _FIXTURES["symbolic"]().write(
        tmp_path / "symbolic.vcf.gz", bgzip=True, index=True
    )
    vcf = VCF(str(path))
    vcf._write_gvi_index()
    vcf._load_index()
    return vcf


def test_vcf_persisted_ilen_matches_oracle(symbolic_vcf):
    vcf = symbolic_vcf
    truth = _FIXTURES["symbolic"]().truth()
    exp = _oracle.expected_ilen(truth, slice(None))
    got = vcf._index.get_column("ILEN").to_list()
    # SV helper columns must not leak into the persisted index
    assert "SVLEN" not in vcf._index.columns
    assert "END" not in vcf._index.columns
    assert "IMPRECISE" not in vcf._index.columns
    # precise rows match the oracle exactly
    assert got[0] == exp[0] == [-100]
    assert got[1] == exp[1] == [50]
    assert got[2] == exp[2] == [30]
    # un-sizable rows are null
    assert got[3] == exp[3] == [None]  # IMPRECISE <DEL>
    assert got[4] == exp[4] == [None]  # <CNV>
    assert got[5] == exp[5] == [None]  # <INV>


def test_oracle_normalizes_compound_sv_type():
    """Oracle correctly handles compound SV types like DUP:TANDEM and DEL:ME.

    vcfixture stores sv_type as the full type_str (e.g. "DUP:TANDEM"), while
    symbolic_ilen normalizes to the first ':'-delimited token.  The oracle must
    apply the same normalization or subtyped SVs wrongly fall through to None.
    """
    from vcfixture import Sym, VcfBuilder, VcfVersion

    b = (
        VcfBuilder(
            samples=["s1"],
            contigs=[("chr1", 1_000_000)],
            version=VcfVersion.V4_4,
        )
        .fmt("GT")
        .info("SVLEN")
        .info("END")
        .info("SVCLAIM")
    )
    # <DUP:TANDEM> with SVLEN=75 -> +75
    b.record(
        "chr1",
        1000,
        ref="G",
        alt=[Sym.duplication("TANDEM")],
        gt=["0|1"],
        info={"SVLEN": [75], "END": [1075], "SVCLAIM": ["DJ"]},
    )
    # <DEL:ME> with SVLEN=42 -> -42
    b.record(
        "chr1",
        2000,
        ref="G",
        alt=[Sym.deletion("ME")],
        gt=["0|1"],
        info={"SVLEN": [42], "END": [2042], "SVCLAIM": ["D"]},
    )
    truth = b.build().truth()
    # Confirm vcfixture stores full type_str
    assert truth.alts_truth[0][0].sv_type == "DUP:TANDEM"
    assert truth.alts_truth[1][0].sv_type == "DEL:ME"
    exp = _oracle.expected_ilen(truth, slice(None))
    assert exp[0] == [75], f"Expected [75] for <DUP:TANDEM>, got {exp[0]}"
    assert exp[1] == [-42], f"Expected [-42] for <DEL:ME>, got {exp[1]}"


def test_var_ranges_handles_null_ilen(symbolic_vcf):
    # The eager var_ranges function (used by SparseVar) materialises ILEN to a
    # numpy array via numba ufuncs.  A null ILEN entry causes Polars to upcast
    # the column to Float64/NaN, which the numba typed ufunc cannot accept.
    # This test calls var_ranges directly so the null rows (POS=4000 IMPRECISE,
    # POS=5000 <CNV>, and POS=6000 <INV>) are always materialised.
    from genoray._var_ranges import var_ranges

    # Wide query spanning all 6 records — forces all ILEN rows through the
    # numpy path including the three nulls.  The result covers variant indices
    # [0, 6) — all six records are represented.
    result = var_ranges(symbolic_vcf._c_norm, symbolic_vcf._index, "chr1", [0], [7_000])
    assert result.shape == (1, 2)
    # All 6 variants represented: exclusive end minus start = 6
    assert result[0, 1] - result[0, 0] == 6

    # Narrow query overlapping only the precise <DEL> at POS=1000.
    # ILEN=-100 means the variant spans [999, 1100) in 0-based coords.
    # The null-ILEN rows must not corrupt the coordinate math — they should
    # be treated as point variants (ILEN=0 → end=POS).
    result2 = var_ranges(
        symbolic_vcf._c_norm, symbolic_vcf._index, "chr1", [999], [1001]
    )
    assert result2.shape == (1, 2)
    # Exactly 1 variant (the <DEL>) overlaps this range.
    assert result2[0, 1] - result2[0, 0] == 1


def test_var_counts_lazy_path_includes_null_ilen_variants(symbolic_vcf):
    # PRE-FIX (before .fill_null(0) in var_counts): null-ILEN rows upcast to
    # Float64/NaN in polars-bio overlap; the null interval end caused SILENT
    # drops, returning 3 instead of 6.  This test exercises the LAZY path
    # (var_counts → VCF.n_vars_in_ranges → VCF.read allocation) and asserts
    # the correct count.
    count = symbolic_vcf.n_vars_in_ranges("chr1", 0, 7_000)[0]
    assert count == 6, (
        f"Expected 6 variants over chr1:0-7000 (including 3 null-ILEN rows), got {count}"
    )

    # Confirm this also flows through VCF.read: the allocated output array
    # must have 6 variants on the variant axis.
    genos = symbolic_vcf.read("chr1", 0, 7_000)
    # shape is (samples, ploidy+phasing, variants) for Genos16
    assert genos.shape[-1] == 6, (
        f"VCF.read returned {genos.shape[-1]} variants, expected 6"
    )


@pytest.fixture
def symbolic_svar(tmp_path):
    path = _FIXTURES["symbolic"]().write(
        tmp_path / "symbolic.vcf.gz", bgzip=True, index=True
    )
    vcf = VCF(str(path))
    vcf._write_gvi_index()
    vcf._load_index()
    svar_path = tmp_path / "symbolic.svar"
    SparseVar.from_vcf(svar_path, vcf, max_mem="100MB", overwrite=True)
    return SparseVar(svar_path)


def test_svar_with_length_null_ilen_no_float_corruption(symbolic_svar):
    # _svar.py:720 materialises ILEN to numpy for the numba with-length kernel.
    # Without .fill_null(0), null-ILEN rows upcast to float64/NaN and the njit
    # kernel silently compiles a float64 specialisation, producing corrupt
    # coordinates.  This test asserts correct *integer* offset values.
    svar = symbolic_svar

    # Wide query over all 6 variants via the with-length read path.
    starts_ends = svar._find_starts_ends_with_length(
        "chr1",
        np.array([0], dtype=np.int32),
        np.array([7_000], dtype=np.int32),
    )
    # Shape is (2, ranges, samples, ploidy) = (2, 1, 2, 2)
    assert starts_ends.shape == (2, 1, 2, 2)

    # All offsets must be non-negative integers — no NaN/float corruption.
    # (The offsets are seqpro OFFSET_TYPE = int64.)
    assert starts_ends.dtype.kind in ("i", "u"), (
        f"starts_ends dtype should be integer, got {starts_ends.dtype}"
    )
    flat = starts_ends.ravel()
    assert np.all(flat >= 0), f"Negative offset values indicate corruption: {flat}"

    # Point-variant null-ILEN rows (ILEN→0) must not artificially shrink or
    # expand the window — the end offset must be >= the start offset for every
    # (range, sample, ploidy) slice.
    starts = starts_ends[0]  # (1, 2, 2)
    ends = starts_ends[1]  # (1, 2, 2)
    assert np.all(ends >= starts), (
        f"Some end offsets precede start offsets: starts={starts}, ends={ends}"
    )


def test_filter_parity_symbolic_vs_imprecise(tmp_path):
    builder = _FIXTURES["symbolic"]()
    path = builder.write(tmp_path / "sym.vcf.gz", bgzip=True, index=True)

    # Write the GVI index from an unfiltered VCF first.
    base_vcf = VCF(str(path))
    base_vcf._write_gvi_index()

    # ~is_symbolic drops ALL symbolic -> empty index (all 6 rows are <...> symbolic).
    # The `_index` is filtered solely by `pl_filter`; the `filter` callable is an
    # inert no-op required by the VCF constructor's filter/pl_filter pairing.
    vcf_all = VCF(
        str(path),
        filter=lambda r: True,
        pl_filter=~exprs.is_symbolic,
    )
    vcf_all._load_index()
    assert vcf_all._index.height == 0

    # ~is_imprecise keeps the 3 precise SVs, drops the 3 un-sizable ones.
    # pl_filter-only requires pairing with a no-op filter callable per the VCF API.
    vcf_precise = VCF(
        str(path),
        filter=lambda r: True,
        pl_filter=~exprs.is_imprecise,
    )
    vcf_precise._load_index()
    assert vcf_precise._index.height == 3
    assert vcf_precise._index.get_column("ILEN").to_list() == [[-100], [50], [30]]


# ---------------------------------------------------------------------------
# Task 7: PGEN symbolic ILEN
# ---------------------------------------------------------------------------

_DATA = Path(__file__).parent / "data"


@pytest.mark.skipif(
    not (_DATA / "symbolic.pgen").exists(),
    reason="run `pixi run test` to generate symbolic PGEN fixtures",
)
def test_pgen_symbolic_ilen_matches_oracle():
    # The symbolic PGEN contains 4 rows (POS 1000/2000/3000/4000).
    # POS is 1-based in the persisted index (matches PVAR convention).
    pgen = PGEN(str(_DATA / "symbolic.pgen"))
    ilen = dict(zip(pgen._index["POS"].to_list(), pgen._index["ILEN"].to_list()))
    assert ilen[1000] == [-100], f"<DEL> POS=1000: expected [-100], got {ilen[1000]}"
    assert ilen[2000] == [50], f"<INS> POS=2000: expected [50], got {ilen[2000]}"
    assert ilen[3000] == [30], f"<DUP> POS=3000: expected [30], got {ilen[3000]}"
    # IMPRECISE <DEL> at POS=4000 should have null ILEN
    assert ilen[4000] == [None], (
        f"IMPRECISE <DEL> POS=4000: expected [None], got {ilen[4000]}"
    )


# ---------------------------------------------------------------------------
# Task 8: SparseVar inherits corrected symbolic ILEN from source VCF
# ---------------------------------------------------------------------------


def test_svar_inherits_symbolic_ilen(tmp_path):
    # Build a SparseVar from the symbolic fixture filtered to the 3 precise SVs
    # (DEL/INS/DUP) by ~is_imprecise.  The SVAR passes the VCF's filtered GVI
    # index through _write_filtered_index unchanged, so the corrected ILEN must
    # be present verbatim.
    path = _FIXTURES["symbolic"]().write(
        tmp_path / "sym.vcf.gz", bgzip=True, index=True
    )
    vcf = VCF(
        str(path),
        filter=lambda r: True,
        pl_filter=~exprs.is_imprecise,
    )
    svar_path = tmp_path / "sym.svar"
    SparseVar.from_vcf(svar_path, vcf, max_mem="100MB", overwrite=True)
    svar = SparseVar(svar_path)
    assert svar.index.get_column("ILEN").to_list() == [[-100], [50], [30]]


# ---------------------------------------------------------------------------
# Task 9: Property test — symbolic ILEN matches oracle across random docs
# ---------------------------------------------------------------------------


def test_property_ilen_matches_oracle():
    """Property test: genoray's persisted VCF ILEN matches the expected_ilen oracle
    across randomly-generated symbolic VCF documents from vcfixture.

    Strategy investigation findings (empirical, 200 examples):
    - vs.symbolic_documents() generates: DEL, INS, DUP, CNV, INV, <*> (Unspecified).
    - NO BND (breakend-notation) ALTs — divergence case #2 never occurs.
    - NO END-only sized SVs (SVLEN always present when sv_end is set) — case #1 never occurs.
    - NO IMPRECISE flag in generated docs.
    - NO literal ALTs (is_sequence always False).
    - CNV / INV / <*>: both oracle and symbolic_ilen return None → they AGREE.
    Therefore the simple got == exp assertion is correct and meaningful.
    """
    import tempfile

    from hypothesis import HealthCheck, given, settings
    from vcfixture import strategies as vs

    @settings(
        max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(vs.symbolic_documents())
    def _prop(doc):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            path = doc.write(tmp / "d.vcf.gz", bgzip=True, index=True)
            vcf = VCF(str(path))
            vcf._write_gvi_index()
            vcf._load_index()
            got = vcf._index.get_column("ILEN").to_list()
            truth = doc.truth()
            exp = _oracle.expected_ilen(truth, slice(None))
            assert len(got) == len(exp), (
                f"Row count mismatch: index has {len(got)} rows, oracle has {len(exp)}"
            )
            for ri, (g_row, e_row) in enumerate(zip(got, exp)):
                assert g_row == e_row, (
                    f"ILEN mismatch at record {ri}: got {g_row}, expected {e_row} "
                    f"(variant_class={truth.variant_class[ri]})"
                )

    _prop()

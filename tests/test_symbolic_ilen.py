from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from genoray import VCF, SparseVar
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


def test_symbolic_fixture_builds_and_classifies():
    from tests.data.fixtures import FIXTURES

    truth = FIXTURES["symbolic"]().truth()
    # records: 0 <DEL> precise, 1 <INS> precise, 2 <DUP> precise,
    #          3 <DEL> IMPRECISE, 4 <CNV> (no usable SVLEN / unsupported)
    assert len(truth.pos) == 5
    assert truth.alts_truth[0][0].sv_type == "DEL"
    assert truth.alts_truth[1][0].sv_type == "INS"
    assert truth.alts_truth[2][0].sv_type == "DUP"
    assert truth.alts_truth[3][0].sv_type == "DEL"  # IMPRECISE <DEL>
    assert truth.alts_truth[4][0].sv_type == "CNV"  # <CNV> unsupported type


def test_expected_ilen_from_oracle():
    from tests.data.fixtures import FIXTURES

    truth = FIXTURES["symbolic"]().truth()
    exp = _oracle.expected_ilen(truth, slice(None))
    assert exp[0] == [-100]  # <DEL>
    assert exp[1] == [50]  # <INS>
    assert exp[2] == [30]  # <DUP>
    assert exp[3] == [None]  # IMPRECISE <DEL> -> null (mirrors symbolic_ilen)
    assert exp[4] == [None]  # <CNV> unsupported


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
    # This test calls var_ranges directly so the null rows (POS=4000 IMPRECISE
    # and POS=5000 <CNV>) are always materialised.
    from genoray._var_ranges import var_ranges

    # Wide query spanning all 5 records — forces all ILEN rows through the
    # numpy path including the two nulls.  The result covers variant indices
    # [0, 5) — all five records are represented.
    result = var_ranges(symbolic_vcf._c_norm, symbolic_vcf._index, "chr1", [0], [6_000])
    assert result.shape == (1, 2)
    # All 5 variants represented: exclusive end minus start = 5
    assert result[0, 1] - result[0, 0] == 5

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
    # drops, returning 3 instead of 5.  This test exercises the LAZY path
    # (var_counts → VCF.n_vars_in_ranges → VCF.read allocation) and asserts
    # the correct count.
    count = symbolic_vcf.n_vars_in_ranges("chr1", 0, 6_000)[0]
    assert count == 5, (
        f"Expected 5 variants over chr1:0-6000 (including 2 null-ILEN rows), got {count}"
    )

    # Confirm this also flows through VCF.read: the allocated output array
    # must have 5 variants on the variant axis.
    genos = symbolic_vcf.read("chr1", 0, 6_000)
    # shape is (samples, ploidy+phasing, variants) for Genos16
    assert genos.shape[-1] == 5, (
        f"VCF.read returned {genos.shape[-1]} variants, expected 5"
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

    # Wide query over all 5 variants via the with-length read path.
    starts_ends = svar._find_starts_ends_with_length(
        "chr1",
        np.array([0], dtype=np.int32),
        np.array([6_000], dtype=np.int32),
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

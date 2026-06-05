from __future__ import annotations

import polars as pl
import pytest

from genoray import VCF
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
    # precise rows match the oracle exactly
    assert got[0] == exp[0] == [-100]
    assert got[1] == exp[1] == [50]
    assert got[2] == exp[2] == [30]
    # un-sizable rows are null
    assert got[3] == [None]  # IMPRECISE <DEL>
    assert got[4] == [None]  # <CNV>


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

"""Tests for `genoray._svar2_fields`: `InfoField`/`FormatField` config and
VCF-header validation (`_resolve_fields`).

The fixture used here is a small, self-contained VCF built inline with
`vcfixture` (not the shared `tests/data` fixtures, which lack the INFO/Flag
fields these tests need). It declares:

- `INFO AC` (Number=A, Integer)   -> bare-str inference + int htslib_type
- `INFO DB` (Number=0, Flag)      -> flag htslib_type
- `FORMAT DS` (Number=1, Float)   -> dtype/default override + int-rejection
- `FORMAT AD` (Number=R, Integer) -> non-scalar Number rejection
"""

from __future__ import annotations

from pathlib import Path

import pytest
from vcfixture import Number, Seq, Type, VcfBuilder

from genoray._svar2_fields import FormatField, _resolve_fields


@pytest.fixture(scope="module")
def vcf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("svar2_fields")
    doc = (
        VcfBuilder(samples=["s1", "s2"], contigs=[("chr1", None)])
        .fmt("GT")
        .info("AC", Number.A, Type.INTEGER)
        .info("DB", Number.FLAG, Type.FLAG)
        .fmt("DS", Number.ONE, Type.FLOAT)
        .fmt("AD", Number.R, Type.INTEGER)
        .record(
            "chr1",
            1000,
            ref="A",
            alt=[Seq("T")],
            gt=["0|1", "1|1"],
            info={"AC": 1, "DB": True},
            DS=[[0.5], [0.9]],
            AD=[[10, 5], [2, 8]],
        )
    )
    return doc.write(tmp_path / "fix.vcf.gz", bgzip=True, index=True)


def test_resolve_bare_str_infers(vcf_path: Path):
    out = _resolve_fields(str(vcf_path), ["AC"], ["DS"])
    assert ("AC", "info", "int", None, None) in out
    assert ("DS", "format", "float", None, None) in out


def test_resolve_flag_info(vcf_path: Path):
    out = _resolve_fields(str(vcf_path), ["DB"], [])
    assert ("DB", "info", "flag", None, None) in out


def test_resolve_dtype_and_default_override(vcf_path: Path):
    out = _resolve_fields(
        str(vcf_path), [], [FormatField("DS", dtype="f16", default=0.0)]
    )
    assert ("DS", "format", "float", "f16", 0.0) in out


def test_reject_unknown_field(vcf_path: Path):
    with pytest.raises(ValueError, match="not found in the VCF header"):
        _resolve_fields(str(vcf_path), ["NOPE"], [])


def test_reject_float_stored_as_int(vcf_path: Path):
    with pytest.raises(ValueError, match="incompatible"):
        _resolve_fields(str(vcf_path), [], [FormatField("DS", dtype="i16")])


def test_reject_nonscalar_number(vcf_path: Path):
    with pytest.raises(ValueError, match="Number"):
        _resolve_fields(str(vcf_path), [], [FormatField("AD")])

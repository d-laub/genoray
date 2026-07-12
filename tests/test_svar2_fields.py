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


# --- end-to-end: from_vcf -> _core.run_conversion_pipeline -> meta.json/on-disk files ---


def test_import_field_specs_from_package():
    from genoray import FormatField as _FF
    from genoray import InfoField as _IF

    assert _IF.__name__ == "InfoField"
    assert _FF is FormatField


def test_from_vcf_writes_dosage_field(tmp_path: Path):
    import json

    from genoray import SparseVar2

    out = tmp_path / "store.svar2"
    SparseVar2.from_vcf(
        out,
        "tests/data/biallelic.vcf.gz",
        no_reference=True,
        format_fields=[FormatField("DS", default=0.0)],
    )
    meta = json.loads((out / "meta.json").read_text())
    ds = next(f for f in meta["fields"] if f["name"] == "DS")
    assert ds["category"] == "format" and ds["dtype"] == "f32"
    contig = meta["contigs"][0]
    vals = list((out / contig / "fields" / "format" / "DS").glob("*/values.bin"))
    assert vals and all(p.stat().st_size % 4 == 0 for p in vals)


def test_from_vcf_with_no_fields_writes_empty_fields_list(tmp_path: Path):
    import json

    from genoray import SparseVar2

    out = tmp_path / "store_nofields.svar2"
    SparseVar2.from_vcf(out, "tests/data/biallelic.vcf.gz", no_reference=True)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["fields"] == []


def _dup_name_vcf(tmp_path: Path) -> Path:
    """A VCF where `DP` is declared as BOTH an INFO field (site depth) and a
    FORMAT field (per-sample depth) — a supported `from_vcf` input per the
    design spec ("a name may exist in both"). Regression fixture for the
    fields/{name}/ collision bug: without a category segment in the on-disk
    path, INFO DP and FORMAT DP would merge/finalize into the same
    `values.bin`, corrupting one or both.
    """
    doc = (
        VcfBuilder(samples=["s0", "s1"], contigs=[("chr1", 100_000)])
        .info("DP", Number.ONE, Type.INTEGER)
        .fmt("GT")
        .fmt("DP", Number.ONE, Type.INTEGER)
        .record(
            "chr1",
            100,
            ref="A",
            alt=[Seq("C")],
            gt=["0|1", "1|1"],
            info={"DP": 30},
            DP=[[10], [20]],
        )
        .record(
            "chr1",
            200,
            ref="G",
            alt=[Seq("T")],
            gt=["1|0", "0|1"],
            info={"DP": 40},
            DP=[[15], [25]],
        )
    )
    return doc.write(tmp_path / "dup.vcf.gz", bgzip=True, index=True)


def test_from_vcf_same_name_info_and_format(tmp_path: Path):
    import json

    from genoray import SparseVar2

    src = _dup_name_vcf(tmp_path)
    out = tmp_path / "store_dup.svar2"
    SparseVar2.from_vcf(
        out, str(src), no_reference=True, info_fields=["DP"], format_fields=["DP"]
    )
    meta = json.loads((out / "meta.json").read_text())
    dps = [f for f in meta["fields"] if f["name"] == "DP"]
    # Both the INFO and FORMAT DP specs must survive finalize independently
    # (no clobber, no double-finalize on a shared file).
    assert {f["category"] for f in dps} == {"info", "format"}
    contig = meta["contigs"][0]
    info_vals = list((out / contig / "fields" / "info" / "DP").glob("*/values.bin"))
    format_vals = list((out / contig / "fields" / "format" / "DP").glob("*/values.bin"))
    assert info_vals
    assert format_vals
    # Auto-narrowing may resolve DP to a sub-4-byte width, and unrouted subs
    # legitimately finalize to empty (0-byte) files, so just require: every
    # file is a whole number of its own category's resolved dtype width
    # (proves each category's rewrite used its own width, undisturbed by the
    # other category sharing the name), and each category wrote SOME data.
    byte_width = {"bool": 1, "i8": 1, "u8": 1, "i16": 2, "u16": 2, "f16": 2}
    info_width = byte_width.get(
        next(f["dtype"] for f in dps if f["category"] == "info"), 4
    )
    format_width = byte_width.get(
        next(f["dtype"] for f in dps if f["category"] == "format"), 4
    )
    assert all(p.stat().st_size % info_width == 0 for p in info_vals)
    assert all(p.stat().st_size % format_width == 0 for p in format_vals)
    assert sum(p.stat().st_size for p in info_vals) > 0
    assert sum(p.stat().st_size for p in format_vals) > 0


def _rare_indel_vcf(tmp_path: Path) -> Path:
    """2 samples; a biallelic indel carried by exactly ONE call, plus a FORMAT
    float DS. With np=4 and a single carrier call, choose_representation routes
    the indel to var_key_indel (var_key ~64 bits < dense ~68), so DS must land
    in fields/format/DS/var_key_indel/values.bin."""
    doc = (
        VcfBuilder(samples=["s0", "s1"], contigs=[("chr1", 100_000)])
        .fmt("GT")
        .fmt("DS", Number.ONE, Type.FLOAT)
        .record(
            "chr1",
            100,
            ref="AT",
            alt=[Seq("A")],  # 1bp deletion (indel)
            gt=["0|1", "0|0"],  # one carrier call total
            DS=[[0.5], [0.0]],
        )
    )
    return doc.write(tmp_path / "rare_indel.vcf.gz", bgzip=True, index=True)


def test_from_vcf_varkey_indel_format_field_written(tmp_path: Path):
    from genoray import SparseVar2

    src = _rare_indel_vcf(tmp_path)
    out = tmp_path / "store_vk_indel.svar2"
    SparseVar2.from_vcf(
        out, str(src), no_reference=True, format_fields=[FormatField("DS", default=0.0)]
    )
    import json

    meta = json.loads((out / "meta.json").read_text())
    contig = meta["contigs"][0]
    vk_indel = (
        out / contig / "fields" / "format" / "DS" / "var_key_indel" / "values.bin"
    )
    assert vk_indel.is_file(), "var_key_indel DS values.bin missing"
    assert vk_indel.stat().st_size > 0, "var_key_indel DS field_calls not written"


def _multifield_vcf(tmp_path: Path) -> Path:
    """2 INFO + 2 FORMAT scalar fields with distinct values, to prove per-field
    files don't cross-contaminate through routing/merge/finalize."""
    doc = (
        VcfBuilder(samples=["s0", "s1"], contigs=[("chr1", 100_000)])
        .info("AC", Number.A, Type.INTEGER)
        .info("AN", Number.ONE, Type.INTEGER)
        .fmt("GT")
        .fmt("DP", Number.ONE, Type.INTEGER)
        .fmt("DS", Number.ONE, Type.FLOAT)
        .record(
            "chr1",
            100,
            ref="A",
            alt=[Seq("C")],
            gt=["0|1", "1|1"],
            info={"AC": 3, "AN": 4},
            DP=[[10], [20]],
            DS=[[0.25], [0.75]],
        )
    )
    return doc.write(tmp_path / "multifield.vcf.gz", bgzip=True, index=True)


def test_from_vcf_multi_field_no_cross_contamination(tmp_path: Path):
    import json

    from genoray import SparseVar2

    src = _multifield_vcf(tmp_path)
    out = tmp_path / "store_multi.svar2"
    SparseVar2.from_vcf(
        out,
        str(src),
        no_reference=True,
        info_fields=["AC", "AN"],
        format_fields=["DP", "DS"],
    )
    meta = json.loads((out / "meta.json").read_text())
    by_name = {f["name"]: f for f in meta["fields"]}
    # Each field is recorded exactly once under its own category.
    assert by_name["AC"]["category"] == "info"
    assert by_name["AN"]["category"] == "info"
    assert by_name["DP"]["category"] == "format"
    assert by_name["DS"]["category"] == "format"
    # DS is the only Float field -> f32; the ints auto-narrow to a sub-4b width.
    assert by_name["DS"]["dtype"] == "f32"
    assert by_name["DP"]["dtype"] in {"i8", "u8", "i16", "u16"}
    contig = meta["contigs"][0]
    # Every declared field has at least one non-empty values.bin, and int fields
    # stay a whole number of their own resolved width (proves disjoint files).
    width = {
        "bool": 1,
        "i8": 1,
        "u8": 1,
        "i16": 2,
        "u16": 2,
        "f16": 2,
        "f32": 4,
        "i32": 4,
        "u32": 4,
    }
    for name, spec in by_name.items():
        cat = spec["category"]
        files = list((out / contig / "fields" / cat / name).glob("*/values.bin"))
        assert files, f"{cat}/{name} has no values.bin"
        assert sum(p.stat().st_size for p in files) > 0, f"{cat}/{name} all empty"
        w = width[spec["dtype"]]
        assert all(p.stat().st_size % w == 0 for p in files), f"{cat}/{name} bad width"


def _ref_and_fields_vcf(tmp_path: Path) -> tuple[Path, Path]:
    """A reference FASTA + a bgzipped/indexed VCF whose REF bases match it,
    carrying one INFO int (AC) and one FORMAT float (DS). Reuses the 40bp
    reference convention from tests/test_svar2_from_vcf.py (_REF)."""
    import subprocess

    ref_seq = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"  # POS3='A', POS7='C'
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{ref_seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    doc = (
        VcfBuilder(samples=["s0", "s1"], contigs=[("chr1", 40)])
        .info("AC", Number.A, Type.INTEGER)
        .fmt("GT")
        .fmt("DS", Number.ONE, Type.FLOAT)
        .record(
            "chr1",
            3,
            ref="A",
            alt=[Seq("G")],
            gt=["1|0", "0|0"],
            info={"AC": 1},
            DS=[[0.5], [0.0]],
        )
        .record(
            "chr1",
            7,
            ref="C",
            alt=[Seq("CAT")],
            gt=["0|1", "1|1"],
            info={"AC": 3},
            DS=[[0.9], [0.2]],
        )
    )
    vcf = doc.write(tmp_path / "ref_fields.vcf.gz", bgzip=True, index=True)
    return vcf, ref


def test_from_vcf_signatures_and_fields_compose(tmp_path: Path):
    import json

    from genoray import SparseVar2

    vcf, ref = _ref_and_fields_vcf(tmp_path)
    out = tmp_path / "store_sig_fields.svar2"
    SparseVar2.from_vcf(
        out,
        str(vcf),
        str(ref),
        signatures=True,
        info_fields=["AC"],
        format_fields=[FormatField("DS", default=0.0)],
        threads=1,
    )
    meta = json.loads((out / "meta.json").read_text())
    names = {f["name"] for f in meta["fields"]}
    assert {"AC", "DS"} <= names, "field manifest missing AC/DS under signatures=True"
    contig = meta["contigs"][0]
    # fields written
    assert list((out / contig / "fields" / "info" / "AC").glob("*/values.bin"))
    assert list((out / contig / "fields" / "format" / "DS").glob("*/values.bin"))
    # mutcat sidecar written alongside (signatures path ran)
    sidecar = list((out / contig).rglob("*sig*")) + list(
        (out / contig).rglob("*mutcat*")
    )
    assert sidecar, "signatures=True produced no mutcat sidecar next to fields"

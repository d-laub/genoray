from pathlib import Path

import numpy as np
import pytest
from vcfixture import Number, Seq, Type, VcfBuilder

from genoray import SparseVar2
from genoray._svar2_fields import (
    FormatField,
    InfoField,
    StoredField,
    _load_field_manifest,
    _resolve_read_fields,
)


def test_canonical_keys_are_bare_when_unique():
    meta = {
        "fields": [
            {"name": "AF", "category": "info", "dtype": "f32", "default": None},
            {"name": "DS", "category": "format", "dtype": "f16", "default": 0.0},
        ]
    }
    avail = _load_field_manifest(meta)
    assert set(avail) == {"AF", "DS"}
    assert isinstance(avail["AF"], StoredField)
    assert avail["AF"].dtype == np.dtype("float32")
    assert avail["DS"].dtype == np.dtype("float16")
    assert avail["DS"].default == 0.0
    assert avail["AF"].default is None


def test_colliding_names_are_qualified_bcftools_style():
    meta = {
        "fields": [
            {"name": "DP", "category": "info", "dtype": "i32", "default": None},
            {"name": "DP", "category": "format", "dtype": "i16", "default": None},
        ]
    }
    avail = _load_field_manifest(meta)
    assert set(avail) == {"INFO/DP", "FORMAT/DP"}
    assert avail["INFO/DP"].dtype == np.dtype("int32")
    assert avail["FORMAT/DP"].dtype == np.dtype("int16")


def test_resolve_rejects_unknown_field():
    avail = _load_field_manifest(
        {
            "fields": [
                {"name": "AF", "category": "info", "dtype": "f32", "default": None}
            ]
        }
    )
    assert _resolve_read_fields(None, avail) == []
    assert _resolve_read_fields(["AF"], avail) == [avail["AF"]]
    with pytest.raises(ValueError, match="NOPE"):
        _resolve_read_fields(["NOPE"], avail)


def test_no_fields_in_meta_is_empty_not_an_error():
    assert _load_field_manifest({}) == {}
    assert _load_field_manifest({"fields": []}) == {}


def test_rejects_auto_dtype_in_meta():
    meta = {
        "fields": [{"name": "AF", "category": "info", "dtype": "auto", "default": None}]
    }
    with pytest.raises(ValueError, match="AF"):
        _load_field_manifest(meta)


def test_rejects_unknown_dtype_in_meta():
    meta = {
        "fields": [
            {"name": "DP", "category": "format", "dtype": "bogus", "default": None}
        ]
    }
    with pytest.raises(ValueError, match="DP"):
        _load_field_manifest(meta)


# --- e2e: SparseVar2(fields=...) / with_fields / decode() (Task 9) ---------
#
# DEVIATION FROM THE BRIEF: the brief's Step-1 fixture converts
# `tests/data/biallelic.vcf.gz`. That file is a GENERATED artifact (built by
# `tests/data/gen_from_vcf.sh` / `gen_svar.py`, not committed) and, as of this
# worktree, declares FORMAT DS but NO info fields at all — converting it with
# `info_fields=[InfoField("AF", ...)]` raises "field not found in the VCF
# header". Rather than hand-edit the generated corpus (out of scope for this
# task and reverted on the next `pixi run gen`), we build a small
# self-contained VCF inline with `vcfixture.VcfBuilder`, exactly like
# `tests/test_svar2_fields.py` already does for the write-path tests. This
# keeps every assertion from the brief intact (see below) while giving a
# fixture that actually has an INFO field, a FORMAT field, >=2 samples, and
# real genotypes.


@pytest.fixture
def store_with_fields(tmp_path: Path) -> SparseVar2:
    """Convert a small inline VCF carrying one INFO and one FORMAT field."""
    doc = (
        VcfBuilder(samples=["s1", "s2"], contigs=[("chr1", None)])
        .fmt("GT")
        .info("AF", Number.A, Type.FLOAT)
        .fmt("DS", Number.A, Type.FLOAT)
        .record(
            "chr1",
            1000,
            ref="A",
            alt=[Seq("T")],
            gt=["0|1", "1|1"],
            info={"AF": [0.25]},
            DS=[[0.4], [1.9]],
        )
    )
    vcf = doc.write(tmp_path / "src.vcf.gz", bgzip=True, index=True)

    out = tmp_path / "fields.svar2"
    SparseVar2.from_vcf(
        out,
        vcf,
        no_reference=True,
        info_fields=[InfoField("AF", dtype="f32")],
        format_fields=[FormatField("DS", dtype="f32", default=0.0)],
    )
    return SparseVar2(out)


def test_available_fields_reports_canonical_keys(store_with_fields):
    avail = store_with_fields.available_fields
    assert set(avail) == {"AF", "DS"}
    assert avail["AF"].category == "info"
    assert avail["DS"].category == "format"


def test_decode_without_fields_is_unchanged(store_with_fields):
    rag = store_with_fields.decode("chr1", [(0, 1_000_000)])
    assert set(rag.fields) == {"pos", "ilen", "allele"}


def test_decode_with_fields_shares_offsets_and_preserves_dtype(store_with_fields):
    sv = store_with_fields.with_fields(["AF", "DS"])
    rag = sv.decode("chr1", [(0, 1_000_000)])
    assert set(rag.fields) == {"pos", "ilen", "allele", "AF", "DS"}

    # `rag.fields` is the list of field NAMES on a record Ragged; per-field
    # data/offsets access goes through `rag["name"]` (`__getitem__`), not
    # `rag.fields["name"]`.
    pos = rag["pos"]
    af = rag["AF"]
    ds = rag["DS"]

    # Stored dtype is preserved end to end — no widening.
    assert af.data.dtype == np.dtype("float32")
    assert ds.data.dtype == np.dtype("float32")

    # One field value per decoded record, on the SAME offsets object.
    assert af.data.shape == pos.data.shape
    assert ds.data.shape == pos.data.shape
    np.testing.assert_array_equal(af.offsets, pos.offsets)
    np.testing.assert_array_equal(ds.offsets, pos.offsets)

    # `Ragged.offsets` is a property that returns the underlying offsets
    # array by reference (no copy), so object identity here is meaningful:
    # it pins the "one shared offsets object across pos/ilen/allele and
    # every decoded field" contract that GenVarLoader relies on. A
    # regression that defensively copies offsets per field would still pass
    # the `assert_array_equal` checks above but MUST fail these.
    assert af.offsets is pos.offsets
    assert ds.offsets is pos.offsets


def test_decode_rejects_unknown_field(store_with_fields):
    with pytest.raises(ValueError, match="NOPE"):
        store_with_fields.with_fields(["NOPE"])

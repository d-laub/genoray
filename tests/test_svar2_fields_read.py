import numpy as np
import pytest

from genoray._svar2_fields import (
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

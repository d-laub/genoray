from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from cyvcf2 import VCF as _CyVCF

FieldDtype = Literal["bool", "i8", "u8", "i16", "u16", "i32", "u32", "f16", "f32"]

_INT_DTYPES = {"bool", "i8", "u8", "i16", "u16", "i32", "u32"}
_FLOAT_DTYPES = {"f16", "f32"}


@dataclass(frozen=True)
class InfoField:
    """A per-variant INFO field to store in the SVAR2 output.

    ``dtype=None`` infers storage from the header (Integer→lossless auto-narrow,
    Float→f32, Flag→bool). ``default`` is the value written for VCF-missing
    entries (else a reserved sentinel/NaN).
    """

    name: str
    dtype: FieldDtype | None = None
    default: float | int | None = None


@dataclass(frozen=True)
class FormatField:
    """A per-sample FORMAT field to store in the SVAR2 output. Genotype-aligned:
    only carrier calls (var_key) or a full per-sample dense column are stored;
    non-carrier values in var_key-routed variants are dropped. Same ``dtype``/
    ``default`` semantics as :class:`InfoField`.
    """

    name: str
    dtype: FieldDtype | None = None
    default: float | int | None = None


def _htslib_type(header_type: str) -> str:
    match header_type:
        case "Integer":
            return "int"
        case "Float":
            return "float"
        case "Flag":
            return "flag"
        case other:
            raise ValueError(
                f"field Type={other!r} is unsupported (need Integer/Float/Flag)"
            )


def _check_dtype_compat(name: str, htype: str, dtype: str | None) -> None:
    if dtype is None:
        return
    if htype in ("int", "flag") and dtype not in _INT_DTYPES:
        raise ValueError(f"field {name!r} ({htype}) incompatible with dtype {dtype!r}")
    if htype == "float" and dtype not in _FLOAT_DTYPES:
        raise ValueError(f"field {name!r} (float) incompatible with dtype {dtype!r}")


def _resolve_one(
    vcf: _CyVCF, spec: str | InfoField | FormatField, category: str
) -> tuple[str, str, str, str | None, float | None]:
    field = spec if not isinstance(spec, str) else None
    name = spec if isinstance(spec, str) else spec.name
    try:
        hdr = vcf.get_header_type(name)  # {'HeaderType','Type','Number',...}
    except KeyError:
        raise ValueError(f"field {name!r} not found in the VCF header") from None
    number = str(hdr.get("Number"))
    htype = _htslib_type(hdr["Type"])
    if htype == "flag":
        if category != "info":
            raise ValueError(f"Flag field {name!r} is INFO-only")
    elif number not in ("1", "A"):
        raise ValueError(
            f"field {name!r} has Number={number}; only scalar Number=1 or "
            "biallelic-split Number=A are supported"
        )
    dtype = None if field is None else field.dtype
    default = (
        None
        if field is None
        else (None if field.default is None else float(field.default))
    )
    _check_dtype_compat(name, htype, dtype)
    return (name, category, htype, dtype, default)


def _resolve_fields(
    vcf_path: str,
    info_fields: Sequence[str | InfoField] | None,
    format_fields: Sequence[str | FormatField] | None,
) -> list[tuple[str, str, str, str | None, float | None]]:
    vcf = _CyVCF(vcf_path)
    try:
        out: list[tuple[str, str, str, str | None, float | None]] = []
        for spec in info_fields or []:
            out.append(_resolve_one(vcf, spec, "info"))
        for spec in format_fields or []:
            out.append(_resolve_one(vcf, spec, "format"))
        return out
    finally:
        vcf.close()


#: Storage dtype (meta.json `fields[].dtype`) -> numpy dtype. The 9 dtypes the
#: writer can resolve to; `bool` is stored as one byte per element.
NP_DTYPE: dict[str, np.dtype] = {
    "bool": np.dtype("bool"),
    "i8": np.dtype("int8"),
    "u8": np.dtype("uint8"),
    "i16": np.dtype("int16"),
    "u16": np.dtype("uint16"),
    "i32": np.dtype("int32"),
    "u32": np.dtype("uint32"),
    "f16": np.dtype("float16"),
    "f32": np.dtype("float32"),
}


@dataclass(frozen=True)
class StoredField:
    """A field present in a finished SVAR2 store, as declared by ``meta.json``.

    ``key`` is the canonical name callers use: the bare ``name`` when it is
    unique across categories, else bcftools-style ``INFO/<name>`` /
    ``FORMAT/<name>``. ``default`` is the fill written for VCF-missing entries;
    when it is ``None`` the store instead uses a reserved sentinel
    (``iinfo.min`` / ``iinfo.max`` / ``NaN``), which is returned as-is.
    """

    name: str
    category: Literal["info", "format"]
    dtype: np.dtype
    default: float | None
    key: str


def _load_field_manifest(meta: dict) -> dict[str, StoredField]:
    """Parse ``meta.json``'s ``fields`` array into canonical-key -> StoredField.

    A store written before fields existed simply has no ``fields`` key; that is
    an empty manifest, not an error (same for an empty ``fields`` array).
    """
    entries = meta.get("fields") or []
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["name"]] = counts.get(e["name"], 0) + 1

    out: dict[str, StoredField] = {}
    for e in entries:
        name = e["name"]
        category = e["category"]
        dtype_str = e["dtype"]
        if dtype_str not in NP_DTYPE:
            raise ValueError(
                f"field {name!r} has unsupported stored dtype {dtype_str!r}; "
                "the store may be corrupt or written by a newer genoray"
            )
        key = name if counts[name] == 1 else f"{category.upper()}/{name}"
        out[key] = StoredField(
            name=name,
            category=category,
            dtype=NP_DTYPE[dtype_str],
            default=e.get("default"),
            key=key,
        )
    return out


def _resolve_read_fields(
    requested: Sequence[str] | None, available: dict[str, StoredField]
) -> list[StoredField]:
    """Validate a field selection against the store's manifest.

    ``None`` selects nothing (fields are opt-in — decoding them costs I/O).
    """
    if requested is None:
        return []
    out: list[StoredField] = []
    for key in requested:
        if key not in available:
            raise ValueError(
                f"field {key!r} is not in this store; available fields: "
                f"{sorted(available)}"
            )
        out.append(available[key])
    return out

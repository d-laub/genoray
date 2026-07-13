from __future__ import annotations

from pathlib import Path

import pytest

from genoray import SparseVar, SparseVar2
from genoray import VCF as _V1VCF
from tests.test_svar2_from_vcf import _write_ref, _write_vcf


def _build_svar1(tmp_path: Path, *, with_dosages: bool = False) -> Path:
    """A SVAR1 store from the shared 40bp fixture VCF (2 SNP/indel biallelic vars)."""
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    v1_out = tmp_path / "in.svar"
    v1 = _V1VCF(str(vcf))
    if with_dosages:
        v1.dosage_field = "DS"
    SparseVar.from_vcf(
        v1_out, v1, max_mem="10m", overwrite=True, with_dosages=with_dosages
    )
    return v1_out


def test_from_svar1_requires_reference_or_opt_out(tmp_path: Path):
    src = _build_svar1(tmp_path)
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_svar1(tmp_path / "out", src, threads=1)


def test_from_svar1_reference_and_no_reference_conflict(tmp_path: Path):
    ref = _write_ref(tmp_path)
    src = _build_svar1(tmp_path)
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_svar1(tmp_path / "out", src, ref, no_reference=True, threads=1)


def test_from_svar1_refuses_existing_out_without_overwrite(tmp_path: Path):
    src = _build_svar1(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        SparseVar2.from_svar1(out, src, no_reference=True, threads=1)

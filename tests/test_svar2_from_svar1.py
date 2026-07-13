from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from genoray import SparseVar, SparseVar2
from genoray import VCF as _V1VCF
from genoray._svar2 import _svar1_index_arrays
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


def test_from_svar1_rejects_index_contig_order_mismatch(tmp_path: Path):
    """metadata.json's `contigs` must exactly match index.arrow's physical CHROM
    run order -- a hand-edited/foreign store that disagrees (renamed, reordered,
    or split contig) must raise loudly rather than silently mis-assign global
    variant-id ranges.
    """
    src = _build_svar1(tmp_path)
    meta_path = src / "metadata.json"
    meta = json.loads(meta_path.read_text())
    assert meta["contigs"] == ["chr1"]
    # index.arrow's only CHROM run is still "chr1"; disagree with it by renaming.
    meta["contigs"] = ["chr2"]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="index.arrow"):
        SparseVar2.from_svar1(
            tmp_path / "out", src, no_reference=True, overwrite=True, threads=1
        )


def test_from_svar1_tolerates_variant_less_declared_contig(tmp_path: Path):
    """metadata.json's `contigs` is the source's FULL header contig dictionary, so
    it routinely names a contig with zero surviving variants (no rows at all in
    index.arrow) -- e.g. a decoy/alt contig, or a full header retained in a
    per-chromosome split. That contig legitimately emits no CHROM run, so the
    guard must not require it to appear in index.arrow's run order; conversion
    must still succeed.
    """
    src = _build_svar1(tmp_path)
    meta_path = src / "metadata.json"
    meta = json.loads(meta_path.read_text())
    assert meta["contigs"] == ["chr1"]
    # "chr2" is header-declared but has no rows in index.arrow. Use
    # no_reference=True so the (nonexistent) fixture reference FASTA doesn't also
    # need a "chr2" entry -- irrelevant to the guard behavior under test.
    meta["contigs"] = ["chr1", "chr2"]
    meta_path.write_text(json.dumps(meta))

    out = tmp_path / "out"
    SparseVar2.from_svar1(out, src, no_reference=True, overwrite=True, threads=1)
    assert (out / "meta.json").exists()


def test_svar1_index_arrays_rejects_split_contig(tmp_path: Path):
    """A contig appearing in two non-adjacent physical runs (e.g. an index.arrow
    that isn't grouped by contig) must still raise, even though the zero-variant
    fix relaxes the guard elsewhere. Constructed directly against
    `_svar1_index_arrays` since `_build_svar1`'s fixture only has one contig, and
    hand-writing a tiny multi-contig index.arrow is cheaper than a real
    multi-contig SVAR1 conversion.
    """
    pl.DataFrame(
        {
            "CHROM": ["chr1", "chr2", "chr1"],
            "POS": [1, 1, 5],
            "REF": ["A", "C", "G"],
            "ALT": ["T", "G", "A"],
        }
    ).write_ipc(tmp_path / "index.arrow")

    with pytest.raises(ValueError, match="index.arrow"):
        _svar1_index_arrays(tmp_path, ["chr1", "chr2"])

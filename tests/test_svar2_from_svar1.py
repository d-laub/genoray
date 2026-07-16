from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genoray import SparseVar, SparseVar2
from genoray import VCF as _V1VCF
from genoray._svar2 import _svar1_index_arrays
from tests.test_svar2_from_pgen import _assert_ragged_equal
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


# --- Task 5: integration tests -- round-trip parity, fields, no_reference ---


def test_from_svar1_matches_from_vcf_genotypes(tmp_path: Path):
    """Headline claim: from_svar1's decoded genotype stream must be
    byte-identical to from_vcf's for the same source data (same offsets, same
    per-field payloads), mirroring test_svar2_from_pgen.py's cross-source
    parity structure.
    """
    ref = _write_ref(tmp_path)
    # Two separate subdirs: `_write_vcf` and `_build_svar1` both write a fixed
    # "in.vcf.gz" filename into whatever dir they're given, so sharing
    # `tmp_path` between them makes the second `bcftools index` collide with
    # the first's already-written .csi.
    vcf_dir = tmp_path / "vcf_src"
    vcf_dir.mkdir()
    svar1_dir = tmp_path / "svar1_src"
    svar1_dir.mkdir()
    vcf = _write_vcf(vcf_dir, symbolic=False, indexed=True)

    # SVAR2 directly from the VCF (the reference / golden path).
    v_vcf = tmp_path / "from_vcf"
    SparseVar2.from_vcf(v_vcf, vcf, ref, threads=1)

    # SVAR1, then SVAR2 from SVAR1.
    src = _build_svar1(svar1_dir)
    v_s1 = tmp_path / "from_svar1"
    dropped = SparseVar2.from_svar1(v_s1, src, ref, threads=1)
    assert dropped == 0

    a = SparseVar2(v_vcf)
    b = SparseVar2(v_s1)
    assert a.available_samples == b.available_samples
    assert a.contigs == b.contigs
    assert (v_s1 / "meta.json").exists()

    # Real decode comparison: element-for-element equal offsets + per-field
    # payloads for every contig, not just metadata parity.
    regions = [(0, 40)]
    for contig in a.contigs:
        ragged_vcf = a.decode(contig, regions)
        ragged_s1 = b.decode(contig, regions)
        _assert_ragged_equal(ragged_s1, ragged_vcf)

    # Spot-check the actual payload isn't trivially empty (both variants from
    # `_write_vcf` -- SNP@3 and INS@7 -- must have produced carrier records).
    rag = a.decode("chr1", regions)
    assert rag["pos"].lengths.reshape(-1).sum() > 0
    pos0 = np.asarray(rag["pos"].data)
    assert set(pos0.tolist()) == {2, 6}  # 0-based POS of VCF POS 3 and 7


def _write_dosage_vcf(d: Path) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DS\t1|0:1.0\t0|0:0.0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT:DS\t0|1:1.0\t1|1:2.0\n"
    )
    plain = d / "ds.vcf"
    plain.write_text(body)
    gz = d / "ds.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_svar1_carries_dosages(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_dosage_vcf(tmp_path)

    # SVAR1 with dosages.
    v1_out = tmp_path / "ds.svar"
    v1 = _V1VCF(str(vcf))
    v1.dosage_field = "DS"
    SparseVar.from_vcf(v1_out, v1, max_mem="10m", overwrite=True, with_dosages=True)

    # Convert; the SVAR1 field is named "dosages".
    out = tmp_path / "ds.svar2"
    SparseVar2.from_svar1(out, v1_out, ref, threads=1)

    sv2 = SparseVar2(out)
    assert "dosages" in sv2.available_fields
    assert sv2.available_fields["dosages"].category == "format"

    sv2f = sv2.with_fields(["dosages"])
    rag = sv2f.decode("chr1", [(0, 40)])
    assert set(rag.fields) == {"pos", "ilen", "allele", "dosages"}

    # Only ALT-carrying haplotypes emit a record (carrier-only semantics --
    # see test_svar2_fields_read.py::test_decode_field_values_match_fixture).
    # Fixture ground truth:
    #   var0 (SNP@3):  S0 = 1|0 (hap0 ALT), DS=1.0;  S1 = 0|0 (no carrier)
    #   var1 (INS@7):  S0 = 0|1 (hap1 ALT), DS=1.0;  S1 = 1|1 (both ALT), DS=2.0
    # Flat cell order is (S0,h0), (S0,h1), (S1,h0), (S1,h1); each cell carries
    # at most one record here.
    lengths = rag["pos"].lengths.reshape(-1)
    assert lengths.tolist() == [1, 1, 1, 1]  # S0h0, S0h1, S1h0, S1h1

    pos = np.asarray(rag["pos"].data)
    dosages = np.asarray(rag["dosages"].data)
    # Record order follows the flat cell order above: S0h0->var0, S0h1->var1,
    # S1h0->var1, S1h1->var1.
    assert pos.tolist() == [2, 6, 6, 6]  # 0-based POS of VCF POS 3, 7, 7, 7
    np.testing.assert_array_equal(
        dosages, np.array([1.0, 1.0, 2.0, 2.0], dtype=dosages.dtype)
    )


def test_from_svar1_no_reference_snp_indel(tmp_path: Path):
    src = _build_svar1(tmp_path)
    out = tmp_path / "noref"
    dropped = SparseVar2.from_svar1(out, src, no_reference=True, threads=1)
    assert dropped == 0
    assert (out / "meta.json").exists()
    assert SparseVar2(out).available_samples == ["S0", "S1"]


def _write_multiallelic_vcf(d: Path) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG,T\t.\t.\t.\tGT\t1|2\t0|1\n"
    )
    plain = d / "multi.vcf"
    plain.write_text(body)
    gz = d / "multi.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_svar1_rejects_multiallelic(tmp_path: Path):
    """`from_svar1` supports only SVAR1's biallelic (geno==1) model; a
    multiallelic SVAR1 store must raise rather than silently mis-convert.
    """
    ref = _write_ref(tmp_path)
    vcf = _write_multiallelic_vcf(tmp_path)

    v1_out = tmp_path / "multi.svar"
    v1 = _V1VCF(str(vcf))
    SparseVar.from_vcf(v1_out, v1, max_mem="10m", overwrite=True)
    assert not SparseVar(v1_out)._is_biallelic

    with pytest.raises(ValueError, match="biallelic"):
        SparseVar2.from_svar1(tmp_path / "out", v1_out, ref, threads=1)


def test_from_svar1_check_ref_accepts_x(tmp_path: Path):
    ref = _write_ref(tmp_path)
    src = _build_svar1(tmp_path)
    out = tmp_path / "check_ref_x"
    SparseVar2.from_svar1(out, src, ref, threads=1, check_ref="x")
    assert (out / "meta.json").exists()


def test_from_svar1_check_ref_invalid_raises(tmp_path: Path):
    ref = _write_ref(tmp_path)
    src = _build_svar1(tmp_path)
    with pytest.raises(ValueError, match="check_ref"):
        SparseVar2.from_svar1(tmp_path / "out", src, ref, threads=1, check_ref="z")

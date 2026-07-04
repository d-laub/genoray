from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genoray import SparseVar2

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _write_ref(d: Path) -> Path:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    return ref


def _write_vcf(d: Path, *, symbolic: bool, indexed: bool) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    if symbolic:
        # POS 20 (1-based) in _REF is 'T'; the anchor REF base must match the
        # FASTA or the pipeline raises RefMismatch before it can even reach
        # the skip-out-of-scope logic being tested here.
        body += "chr1\t20\t.\tT\t<DEL>\t.\t.\t.\tGT\t0|1\t0|0\n"
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    if indexed:
        subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_with_reference_roundtrips(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store"
    dropped = SparseVar2.from_vcf(out, vcf, ref, threads=1)
    assert dropped == 0
    assert (out / "meta.json").exists()
    sv = SparseVar2(out)
    assert sv.samples == ["S0", "S1"]
    assert sv.contigs == ["chr1"]


def test_from_vcf_requires_reference_or_opt_out(tmp_path: Path):
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_vcf(tmp_path / "s1", vcf, threads=1)


def test_from_vcf_reference_and_no_reference_conflict(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_vcf(tmp_path / "s2", vcf, ref, no_reference=True, threads=1)


def test_from_vcf_no_reference_snp_only(tmp_path: Path):
    # SNP-only VCF, no reference: trusts pre-normalization, converts fine.
    body_vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store_noref"
    dropped = SparseVar2.from_vcf(out, body_vcf, no_reference=True, threads=1)
    assert dropped == 0
    assert (out / "meta.json").exists()


def test_from_vcf_auto_indexes_unindexed_source(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=False)  # no .csi/.tbi
    assert not (tmp_path / "in.vcf.gz.csi").exists()
    out = tmp_path / "store_idx"
    SparseVar2.from_vcf(out, vcf, ref, threads=1)
    assert (tmp_path / "in.vcf.gz.csi").exists()
    assert (out / "meta.json").exists()


def test_from_vcf_skip_out_of_scope_counts(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=True, indexed=True)
    out = tmp_path / "store_skip"
    dropped = SparseVar2.from_vcf(out, vcf, ref, skip_out_of_scope=True, threads=1)
    assert dropped == 1


def test_from_vcf_symbolic_errors_without_skip(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=True, indexed=True)
    with pytest.raises(Exception):
        SparseVar2.from_vcf(tmp_path / "store_err", vcf, ref, threads=1)


def test_from_vcf_plain_vcf_rejected(tmp_path: Path):
    ref = _write_ref(tmp_path)
    plain = tmp_path / "in.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    with pytest.raises(ValueError, match="bgzip"):
        SparseVar2.from_vcf(tmp_path / "s3", plain, ref, threads=1)

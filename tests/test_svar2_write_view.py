"""Tests for SparseVar2.write_view (region/sample subset via re-conversion)."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from genoray import SparseVar2


def _dir_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "meta.json"
    }


def test_write_view_reroute_false_not_implemented(svar2_store, tmp_path):
    sv = SparseVar2(svar2_store)
    with pytest.raises(NotImplementedError, match="reroute"):
        sv.write_view(
            (sv.contigs[0], 0, 40),
            sv.available_samples,
            tmp_path / "v.svar2",
            reroute=False,
        )


def test_write_view_self_overwrite_guard(svar2_store):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="in place|same path"):
        sv.write_view(
            (sv.contigs[0], 0, 40),
            sv.available_samples,
            sv.path,
            overwrite=True,
        )


def test_write_view_byte_parity_with_from_vcf(tmp_path):
    """reroute=True on a full region+all samples == a fresh from_vcf on the same input.

    Self-contained oracle: builds its own single-contig VCF+ref (no fields, no
    signatures) rather than reusing the session `svar2_store` fixture, which
    exposes no vcf/ref paths. A full-region/all-sample view re-runs the same
    cost model on the same effective variants with no stored fields/signatures
    to carry through, so the routed sidecar bytes should be identical.
    """
    ref_seq = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{ref_seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = tmp_path / "in.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\n"
    )
    vcf_gz = tmp_path / "in.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {vcf_gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(vcf_gz)], check=True)

    direct = tmp_path / "direct.svar2"
    SparseVar2.from_vcf(direct, vcf_gz, ref, threads=1, overwrite=True)
    direct_sv = SparseVar2(direct)

    viewed = tmp_path / "viewed.svar2"
    direct_sv.write_view(
        (direct_sv.contigs[0], 0, len(ref_seq)),
        direct_sv.available_samples,
        viewed,
        overwrite=True,
    )
    viewed_sv = SparseVar2(viewed)

    assert viewed_sv.contigs == direct_sv.contigs
    assert viewed_sv.available_samples == direct_sv.available_samples
    for c in direct_sv.contigs:
        assert _dir_digest(direct / c) == _dir_digest(viewed / c)

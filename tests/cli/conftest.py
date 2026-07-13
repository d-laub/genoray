"""Shared fixtures for genoray-cli tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_vcf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny bgzipped+indexed VCF with 3 samples × 4 variants on chr1."""
    d = tmp_path_factory.mktemp("vcf")
    plain = d / "tiny.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\n"
        "chr1\t10\t.\tA\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/0\n"  # singleton in A
        "chr1\t20\t.\tC\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\n"  # in B and C
        "chr1\t30\t.\tG\tA\t.\t.\t.\tGT\t0/0\t0/0\t1/0\n"  # singleton in C
        "chr1\t40\t.\tT\tC\t.\t.\t.\tGT\t1/0\t0/0\t0/0\n"  # singleton in A
    )
    gz = d / "tiny.vcf.gz"
    with open(gz, "wb") as out:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=out)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


@pytest.fixture(scope="session")
def tiny_svar(tmp_path_factory: pytest.TempPathFactory, tiny_vcf: Path) -> Path:
    """A tiny SVAR built from `tiny_vcf` via `SparseVar.from_vcf`."""
    from genoray import VCF, SparseVar

    out = tmp_path_factory.mktemp("svar") / "tiny.svar"
    SparseVar.from_vcf(out, VCF(tiny_vcf), max_mem="64m", overwrite=True)
    return out


@pytest.fixture(scope="session")
def tiny_svar2(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny two-contig (chr1, chr2) SVAR2 store built via `SparseVar2.from_vcf`."""
    from genoray import SparseVar2

    d = tmp_path_factory.mktemp("svar2")
    ref = d / "ref.fa"
    ref.write_text(">chr1\nACAGT\n>chr2\nACAGT\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    plain = d / "tiny.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=PASS,Description="">\n'
        "##contig=<ID=chr1,length=5>\n"
        "##contig=<ID=chr2,length=5>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\n"
        "chr1\t1\t.\tA\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/0\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\n"
        "chr2\t1\t.\tA\tC\t.\t.\t.\tGT\t0/0\t0/0\t1/0\n"
        "chr2\t2\t.\tC\tT\t.\t.\t.\tGT\t1/0\t0/0\t0/0\n"
    )
    gz = d / "tiny.vcf.gz"
    with open(gz, "wb") as out:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=out)
    subprocess.run(["bcftools", "index", str(gz)], check=True)

    out = d / "tiny.svar2"
    SparseVar2.from_vcf(out, gz, ref, threads=1, overwrite=True)
    return out

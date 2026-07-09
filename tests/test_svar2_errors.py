"""SP-3: conversion errors surface as typed Python exceptions, not RuntimeError."""

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


def _write_vcf(d: Path, body_rows: str, *, contig_len: int = 40) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        f"##contig=<ID=chr1,length={contig_len}>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n" + body_rows
    )
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_symbolic_alt_raises_value_error(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, "chr1\t20\t.\tT\t<DEL>\t.\t.\t.\tGT\t0|1\t0|0\n")
    with pytest.raises(ValueError, match="symbolic"):
        SparseVar2.from_vcf(tmp_path / "store", vcf, ref, threads=1)


def test_ref_mismatch_raises_value_error(tmp_path: Path):
    ref = _write_ref(tmp_path)
    # POS 3 (1-based) in _REF is 'A'; claim REF='G' to force a mismatch.
    vcf = _write_vcf(tmp_path, "chr1\t3\t.\tG\tT\t.\t.\t.\tGT\t0|1\t0|0\n")
    with pytest.raises(ValueError, match="disagrees"):
        SparseVar2.from_vcf(tmp_path / "store", vcf, ref, threads=1)


def test_missing_reference_raises_file_not_found(tmp_path: Path):
    vcf = _write_vcf(tmp_path, "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n")
    with pytest.raises(FileNotFoundError):
        SparseVar2.from_vcf(tmp_path / "store", vcf, tmp_path / "nope.fa", threads=1)

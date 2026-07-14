"""Shared pytest fixtures for the SVAR2 consumer test suites (M6b/M6c)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genoray import _core

# 40 bp reference; the REF bases below match this exactly (1-based VCF POS):
# POS 3 = 'A', POS 7 = 'C', POS 12..14 = 'GTA'.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"

_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


@pytest.fixture(scope="session")
def svar2_store(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("svar2")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store"
    _core.run_conversion_pipeline(
        str(bcf),  # vcf_path (BCF + .csi, mirrors the Rust harness)
        str(ref),  # reference_path
        ["chr1"],  # chroms
        str(out),  # output_dir
        ["S0", "S1"],  # samples
        25_000,  # chunk_size
        2,  # ploidy
        1,  # max_threads
        8 * 1024 * 1024,  # long_allele_capacity
    )
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


def build_two_contig_svar2(tmp_path):
    """Build a 2-contig (chr1, chr2) svar2 store for concat/split tests."""
    import subprocess
    from pathlib import Path

    from genoray import SparseVar2

    d = Path(tmp_path)
    ref = d / "ref.fa"
    ref.write_text(">chr1\n" + _REF + "\n>chr2\n" + _REF + "\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    vcf = d / "in.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=PASS,Description="">\n'
        "##contig=<ID=chr1,length=40>\n"
        "##contig=<ID=chr2,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t9\t.\tT\tC\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr2\t5\t.\tT\tA\t.\t.\t.\tGT\t0|1\t0|1\n"
    )
    vcf_gz = d / "in.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {vcf_gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(vcf_gz)], check=True)
    out = d / "two.svar2"
    SparseVar2.from_vcf(out, vcf_gz, ref, threads=1, overwrite=True)
    return SparseVar2(out)

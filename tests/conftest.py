"""Shared pytest fixtures for the SVAR2 consumer test suites (M6b/M6c)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genoray import _core
from genoray._pipeline_args import FieldSpec, PlanSettings, RegionSpec

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
def small_vcf(tmp_path_factory) -> Path:
    """A tiny (3-record, 2-sample, single-contig) BCF+CSI store."""
    d = tmp_path_factory.mktemp("banner-vcf")
    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)
    return bcf


@pytest.fixture(scope="session")
def small_pgen(tmp_path_factory) -> Path:
    """The same 3-record, 2-sample, single-contig cohort as `small_vcf`, as a PGEN."""
    d = tmp_path_factory.mktemp("banner-pgen")
    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(vcf)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(gz),
            "--out",
            str(d / "in"),
        ],
        check=True,
    )
    return d / "in.pgen"


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
        vcf_path=str(bcf),  # BCF + .csi, mirrors the Rust harness
        reference_path=str(ref),
        output_dir=str(out),
        regions=RegionSpec(chroms=["chr1"], samples=["S0", "S1"]),
        fields=FieldSpec(),
        plan=PlanSettings(
            chunk_size=25_000,
            max_threads=1,
            long_allele_capacity=8 * 1024 * 1024,
        ),
        ploidy=2,
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


def build_svar2_singleton_store(tmp_path, n_samples: int = 12) -> Path:
    """An svar2 store whose variants all route to the var_key channel.

    One singleton SNP per sample, all inside ``[0, 20)``, so a single region
    query returns one non-empty var_key cell per sample with strictly ascending
    ``cell_id``. That is what makes the sparse emitter's ordering observable:
    the session ``svar2_store`` fixture has one non-empty var_key cell in 18,
    because at 2 samples the cost model routes its INS and DEL dense.

    Sample ``i`` carries SNP ``i`` on hap ``i % 2``, so cell ids are
    ``2*i + (i % 2)`` -- distinct, ascending in ``i``, and not simply ``0..H``.
    """
    import subprocess
    from pathlib import Path

    d = Path(tmp_path)
    assert n_samples <= 20, "keep every singleton inside [0, 20)"

    ref = d / "ref.fa"
    ref.write_text(">chr1\n" + _REF + "\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    samples = [f"S{i}" for i in range(n_samples)]
    rows = []
    for i in range(n_samples):
        pos = i + 1  # 1-based VCF POS, so REF is _REF[i]
        ref_base = _REF[i]
        alt = "A" if ref_base != "A" else "C"
        gt = ["0|0"] * n_samples
        gt[i] = "1|0" if i % 2 == 0 else "0|1"
        rows.append(f"chr1\t{pos}\t.\t{ref_base}\t{alt}\t.\t.\t.\tGT\t" + "\t".join(gt))
    vcf = d / "singletons.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples)
        + "\n"
        + "\n".join(rows)
        + "\n"
    )
    bcf = d / "singletons.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store"
    _core.run_conversion_pipeline(
        vcf_path=str(bcf),
        reference_path=str(ref),
        output_dir=str(out),
        regions=RegionSpec(chroms=["chr1"], samples=samples),
        fields=FieldSpec(),
        plan=PlanSettings(
            chunk_size=25_000,
            max_threads=1,
            long_allele_capacity=8 * 1024 * 1024,
        ),
        ploidy=2,
    )
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


@pytest.fixture(scope="session")
def svar2_singleton_store(tmp_path_factory) -> Path:
    """Session-scoped :func:`build_svar2_singleton_store`."""
    return build_svar2_singleton_store(tmp_path_factory.mktemp("svar2-singletons"))

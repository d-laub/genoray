from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genoray import SparseVar2

# 40 bp reference. 1-based POS 3 = 'A', 7 = 'C', 12..14 = 'GTA'.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"

# Phased, no half-calls, no symbolics: plink2's VCF import is lossless here, so
# from_pgen and from_vcf must agree exactly. (A half-call like './1' would NOT
# round-trip: gen_from_vcf.sh passes --vcf-half-call r, which rewrites it.)
_VCF_BODY = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=40>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
    "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
    "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    "chr1\t12\t.\tGTA\tG,GT\t.\t.\t.\tGT\t1|2\t0|1\n"
    "chr1\t20\t.\tT\tA\t.\t.\t.\tGT\t.|.\t1|0\n"
)


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> tuple[Path, Path, Path]:
    """(reference fasta, bgzipped+indexed vcf, pgen) for the same variants."""
    d = tmp_path_factory.mktemp("frompgen")

    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    plain = d / "in.vcf"
    plain.write_text(_VCF_BODY)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
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
    return ref, gz, d / "in.pgen"


def test_from_pgen_matches_from_vcf(sources, tmp_path):
    ref, vcf, pgen = sources
    from_vcf = tmp_path / "vcf.svar2"
    from_pgen = tmp_path / "pgen.svar2"

    SparseVar2.from_vcf(from_vcf, vcf, ref)
    SparseVar2.from_pgen(from_pgen, pgen, ref)

    a = SparseVar2(from_vcf)
    b = SparseVar2(from_pgen)
    assert a.n_samples == b.n_samples == 2

    regions = [(0, len(_REF))]
    ragged_vcf = a.decode("chr1", regions)
    ragged_pgen = b.decode("chr1", regions)
    assert ragged_pgen.offsets.tolist() == ragged_vcf.offsets.tolist()
    # `.data` on a record Ragged (multiple named fields sharing one ragged axis)
    # returns a dict of field -> ndarray, not a single array -- compare per field.
    assert ragged_pgen.data.keys() == ragged_vcf.data.keys()
    for field in ragged_vcf.data:
        assert ragged_pgen.data[field].tolist() == ragged_vcf.data[field].tolist()


def test_from_pgen_requires_exactly_one_of_reference_or_no_reference(sources, tmp_path):
    _, _, pgen = sources
    with pytest.raises(ValueError, match="exactly one"):
        SparseVar2.from_pgen(tmp_path / "a.svar2", pgen)
    with pytest.raises(ValueError, match="exactly one"):
        SparseVar2.from_pgen(tmp_path / "b.svar2", pgen, "ref.fa", no_reference=True)


def test_from_pgen_refuses_to_overwrite(sources, tmp_path):
    ref, _, pgen = sources
    out = tmp_path / "exists.svar2"
    SparseVar2.from_pgen(out, pgen, ref)
    with pytest.raises(FileExistsError):
        SparseVar2.from_pgen(out, pgen, ref)
    SparseVar2.from_pgen(out, pgen, ref, overwrite=True)  # no raise

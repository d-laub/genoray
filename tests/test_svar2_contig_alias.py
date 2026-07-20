"""Reader-side contig-name normalization for SparseVar2 (chr-prefix + mito aliases)."""

import subprocess

import numpy as np
import pytest

from genoray import SparseVar2


def test_decode_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)  # store contig is "chr1"
    native = sv.decode("chr1", [(0, 40)])
    alias = sv.decode("1", [(0, 40)])
    assert (
        native["pos"].lengths.reshape(-1).tolist()
        == alias["pos"].lengths.reshape(-1).tolist()
    )
    assert np.array_equal(np.asarray(native["pos"].data), np.asarray(alias["pos"].data))


def test_region_counts_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)
    assert np.array_equal(
        sv.region_counts("chr1", [(0, 40)]), sv.region_counts("1", [(0, 40)])
    )


def test_read_ranges_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)
    native = sv.read_ranges("chr1", [0], [40])
    alias = sv.read_ranges("1", [0], [40])
    assert np.array_equal(native["vk_pos"], alias["vk_pos"])
    assert np.array_equal(native["vk_key"], alias["vk_key"])


def test_unknown_contig_raises_valueerror(svar2_store):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="not found in store"):
        sv.decode("chrZ", [(0, 40)])


def test_subset_contigs_accepts_unprefixed_contig(tmp_path, svar2_store):
    sv = SparseVar2(svar2_store)  # store contig is "chr1"
    out = tmp_path / "subset.svar2"
    sv.subset_contigs(out, ["1"], overwrite=True)  # unprefixed alias
    assert SparseVar2(out).contigs == ["chr1"]  # canonical store spelling preserved


def test_subset_contigs_unknown_raises(tmp_path, svar2_store):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="not found in store"):
        sv.subset_contigs(tmp_path / "x.svar2", ["chrZ"], overwrite=True)


# 40 bp reference (matches the REF bases used below); kept local to avoid a
# fragile cross-module import of conftest internals.
_REF40 = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _bgzip_index(vcf_path):
    gz = vcf_path.with_suffix(vcf_path.suffix + ".gz")
    subprocess.run(f"bgzip -c {vcf_path} > {gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_reference_naming_mismatch(tmp_path):
    """Unprefixed VCF contigs ('1') convert against a chr-prefixed FASTA ('chr1')."""
    # chr-prefixed FASTA
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_REF40}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    # UNPREFIXED VCF (contig "1")
    vcf = tmp_path / "in.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\n"  # indel -> exercises left-align vs FASTA
    )
    vcf_gz = _bgzip_index(vcf)

    out = tmp_path / "store.svar2"
    from genoray import SparseVar2

    # Before the fix this raised: "Contig '1' not found in reference FASTA".
    SparseVar2.from_vcf(out, vcf_gz, ref, threads=1, overwrite=True)
    sv = SparseVar2(out)
    assert sv.contigs == ["1"]  # store keeps the source's spelling
    counts = sv.region_counts("1", [(0, 40)])  # non-empty => indel normalized OK
    assert int(counts.sum()) > 0

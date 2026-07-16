from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
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
    assert sv.available_samples == ["S0", "S1"]
    assert sv.contigs == ["chr1"]


def test_from_vcf_regions_restricts_conversion(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "regioned"
    dropped = SparseVar2.from_vcf(out, vcf, ref, regions="chr1:1-4", threads=1)
    assert dropped == 0

    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]
    counts = sv.region_counts("chr1", [(0, 40)])
    assert int(counts.sum()) == 1
    rag = sv.decode("chr1", [(0, 40)])
    assert np.asarray(rag["pos"].data).tolist() == [2]


def test_from_vcf_regions_accepts_multiple_specs_and_merges(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "merged"
    SparseVar2.from_vcf(
        out,
        vcf,
        ref,
        regions=["chr1:1-4", ("chr1", 3, 8)],
        merge_overlapping=True,
        threads=1,
    )

    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]
    counts = sv.region_counts("chr1", [(0, 40)])
    assert int(counts.sum()) == 4
    rag = sv.decode("chr1", [(0, 40)])
    assert sorted(set(np.asarray(rag["pos"].data).tolist())) == [2, 6]


def test_from_vcf_regions_rejects_overlaps_by_default(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    with pytest.raises(ValueError, match="regions overlap"):
        SparseVar2.from_vcf(
            tmp_path / "overlap",
            vcf,
            ref,
            regions=["chr1:1-4", ("chr1", 3, 8)],
            threads=1,
        )


def _write_vcf_with_spanning_deletion(d: Path) -> Path:
    """A dedicated (non-shared) fixture: a deletion whose POS sits outside
    ``chr1:7-12`` but whose anchor-trimmed extent reaches into it, plus the
    in-region insertion also used by ``_write_vcf``.

    ``chr1:5 TACA>T`` (0-based POS 4, REF span ``[4, 8)`` matching
    ``_REF[4:8]`` == ``"TACA"``) deletes the anchor-trimmed extent ``[5, 8)``.
    Region ``chr1:7-12`` is 0-based ``[6, 12)``: the deletion's POS (4) is
    outside it, but its trimmed extent (``[5, 8)``) overlaps it at
    ``[6, 8)``. Verified stable under the pipeline's left-alignment (POS
    stays at 0-based 4 -- see task-2-report.md) so ``regions_overlap="pos"``
    drops it while ``"variant"`` keeps it. This VCF is NOT the shared
    ``_write_vcf`` fixture -- many other tests assert exact variant
    counts/genotypes against that fixture's SNP+insertion-only contents.
    """
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t5\t.\tTACA\tT\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    plain = d / "del_in.vcf"
    plain.write_text(body)
    gz = d / "del_in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_regions_overlap_variant_keeps_spanning_deletion(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf_with_spanning_deletion(tmp_path)
    out = tmp_path / "variant_mode"
    SparseVar2.from_vcf(
        out, vcf, ref, regions="chr1:7-12", regions_overlap="variant", threads=1
    )
    sv = SparseVar2(out)
    rag = sv.decode("chr1", [(0, 40)])
    # POS 4 (0-based) is the spanning deletion; its extent overlaps the
    # region even though its POS does not.
    assert 4 in np.asarray(rag["pos"].data).tolist()


def test_from_vcf_regions_overlap_pos_excludes_spanning_deletion(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf_with_spanning_deletion(tmp_path)
    out_v = tmp_path / "v"
    out_p = tmp_path / "p"
    SparseVar2.from_vcf(
        out_v, vcf, ref, regions="chr1:7-12", regions_overlap="variant", threads=1
    )
    SparseVar2.from_vcf(
        out_p, vcf, ref, regions="chr1:7-12", regions_overlap="pos", threads=1
    )
    counts_v = int(SparseVar2(out_v).region_counts("chr1", [(0, 40)]).sum())
    counts_p = int(SparseVar2(out_p).region_counts("chr1", [(0, 40)]).sum())
    assert counts_v > counts_p


def test_from_vcf_regions_overlap_variant_rejects_multiple_regions_per_contig(
    tmp_path: Path,
):
    """Two disjoint regions on the SAME contig are unsound under
    `regions_overlap="variant"` (a deletion's anchor-trimmed extent could span
    the gap between them and be double-counted/dropped depending on the
    reader) -- must raise rather than silently misbehave.
    """
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    with pytest.raises(ValueError, match="at most one region per contig"):
        SparseVar2.from_vcf(
            tmp_path / "multi_variant",
            vcf,
            ref,
            regions=["chr1:1-4", "chr1:7-12"],
            regions_overlap="variant",
            threads=1,
        )


def test_from_vcf_regions_overlap_pos_allows_multiple_regions_per_contig(
    tmp_path: Path,
):
    """The same two disjoint regions are fine under `regions_overlap="pos"`
    (POS belongs to exactly one coalesced region) -- must NOT raise, and must
    still restrict conversion to each region's own variant.
    """
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "multi_pos"
    SparseVar2.from_vcf(
        out,
        vcf,
        ref,
        regions=["chr1:1-4", "chr1:7-12"],
        regions_overlap="pos",
        threads=1,
    )
    sv = SparseVar2(out)
    rag = sv.decode("chr1", [(0, 40)])
    assert sorted(set(np.asarray(rag["pos"].data).tolist())) == [2, 6]


def test_from_vcf_regions_overlap_variant_allows_merged_adjacent_regions(
    tmp_path: Path,
):
    """Two overlapping/adjacent region specs that `merge_overlapping=True`
    coalesces into ONE interval per contig must NOT trip the multi-region
    guard, even though the caller passed multiple region specs.
    """
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "multi_variant_merged"
    SparseVar2.from_vcf(
        out,
        vcf,
        ref,
        regions=["chr1:1-4", ("chr1", 3, 8)],
        merge_overlapping=True,
        regions_overlap="variant",
        threads=1,
    )
    sv = SparseVar2(out)
    rag = sv.decode("chr1", [(0, 40)])
    assert sorted(set(np.asarray(rag["pos"].data).tolist())) == [2, 6]


def test_from_vcf_samples_preserve_order(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "samples"
    dropped = SparseVar2.from_vcf(out, vcf, ref, samples=["S1"], threads=1)
    assert dropped == 0

    sv = SparseVar2(out)
    assert sv.available_samples == ["S1"]
    counts = sv.region_counts("chr1", [(0, 40)])
    assert counts.shape == (1, 1, 2)
    assert counts.reshape(-1).tolist() == [1, 1]


def test_from_vcf_samples_reject_unknown(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    with pytest.raises(ValueError, match="Samples not found"):
        SparseVar2.from_vcf(tmp_path / "bad_sample", vcf, ref, samples=["missing"])


def test_from_vcf_explicit_none_matches_default(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)

    default_out = tmp_path / "default"
    explicit_out = tmp_path / "explicit"
    SparseVar2.from_vcf(default_out, vcf, ref, threads=1)
    SparseVar2.from_vcf(
        explicit_out,
        vcf,
        ref,
        regions=None,
        samples=None,
        threads=1,
    )

    default = SparseVar2(default_out)
    explicit = SparseVar2(explicit_out)
    assert explicit.contigs == default.contigs
    assert explicit.available_samples == default.available_samples
    np.testing.assert_array_equal(
        explicit.region_counts("chr1", [(0, 40)]),
        default.region_counts("chr1", [(0, 40)]),
    )


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
    with pytest.raises(ValueError, match="symbolic"):
        SparseVar2.from_vcf(tmp_path / "store_err", vcf, ref, threads=1)


def test_from_vcf_plain_vcf_rejected(tmp_path: Path):
    ref = _write_ref(tmp_path)
    plain = tmp_path / "in.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    with pytest.raises(ValueError, match="bgzip"):
        SparseVar2.from_vcf(tmp_path / "s3", plain, ref, threads=1)


def _write_vcf_bad_ref(d: Path) -> Path:
    # Clean records at pos 3 (REF=A) and 7 (REF=C) match _REF; the record at
    # pos 10 declares REF=A but _REF[10] is 'G' — a REF/FASTA disagreement
    # (issue #116). Rows stay position-sorted for `bcftools index`.
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t10\t.\tA\tT\t.\t.\t.\tGT\t0|1\t0|0\n"  # REF=A, but _REF[10]='G'
    )
    plain = d / "bad.vcf"
    plain.write_text(body)
    gz = d / "bad.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_check_ref_error_aborts(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf_bad_ref(tmp_path)
    with pytest.raises(Exception):
        SparseVar2.from_vcf(tmp_path / "store", vcf, ref, check_ref="e", threads=1)


def test_from_vcf_check_ref_exclude_continues(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf_bad_ref(tmp_path)
    out = tmp_path / "store"
    SparseVar2.from_vcf(out, vcf, ref, check_ref="x", threads=1)
    assert (out / "meta.json").exists()  # completed despite the bad record
    sv = SparseVar2(out)
    # The two clean records survive; the pos-10 mismatch is excluded.
    # `region_counts` is a per-hap carrier count (see `SparseVar2.region_counts`),
    # not a distinct-variant count: pos 3 contributes 1 carrier hap (S0 "1|0"),
    # pos 7 contributes 3 (S0 "0|1", S1 hom-alt "1|1") -> 4 total.
    counts = sv.region_counts("chr1", [(0, 40)])
    assert int(counts.sum()) == 4


def test_from_vcf_check_ref_invalid_value_raises(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf_bad_ref(tmp_path)
    with pytest.raises(ValueError, match="check_ref"):
        SparseVar2.from_vcf(tmp_path / "s", vcf, ref, check_ref="z", threads=1)  # type: ignore[arg-type]

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


def _ss(d: Path, name: str, sample: str, rows: str) -> Path:
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
    )
    plain = d / f"{name}.vcf"
    plain.write_text(header + rows)
    gz = d / f"{name}.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def _ss_contig(d: Path, name: str, sample: str, contig: str, rows: str) -> Path:
    """Like `_ss`, but the contig name (both header `##contig` line and each
    data row's leading column) is a parameter -- for the mixed-naming-scheme
    test, which needs one file spelling the same logical contig `chr1` and
    another spelling it `1`."""
    header = (
        "##fileformat=VCFv4.2\n"
        f"##contig=<ID={contig},length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
    )
    plain = d / f"{name}.vcf"
    plain.write_text(header + rows)
    gz = d / f"{name}.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def _ss_fields(d: Path, name: str, sample: str, rows: str) -> Path:
    """Like `_ss`, but the header also declares `INFO AF` (Number=A, Float)
    and `FORMAT DP` (Number=1, Integer), for the field-carry-through tests."""
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
    )
    plain = d / f"{name}.vcf"
    plain.write_text(header + rows)
    gz = d / f"{name}.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_list_disjoint_sites_hom_ref_fill(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "store"
    dropped = SparseVar2.from_vcf_list(out, [a, b], ref, threads=1)
    assert dropped == 0
    sv = SparseVar2(out)
    assert sv.available_samples == ["SA", "SB"]
    # SA carries SNP@2 on hap0; SB carries INS@6 on hap1.
    counts = sv.region_counts("chr1", [(0, 40)]).reshape(
        -1
    )  # (R,S,P) -> [SA_h0,SA_h1,SB_h0,SB_h1]
    assert counts.tolist() == [1, 0, 0, 1]


def test_from_vcf_list_contig_missing_from_some_files_is_hom_ref_filled(
    tmp_path: Path,
):
    """Issue #122: a cohort-shared pipeline header declares every contig in
    every file, yet a file may have zero records on a given contig (e.g. a
    female sample has no somatic chrY). rust-htslib's seek RAISES on such a
    header-declared-but-index-absent contig (cyvcf2 instead returns empty),
    which used to abort the whole `from_vcf_list` run with an opaque OSError
    (`error seeking to "<contig>":0 in indexed file`). The conversion must
    instead skip that file for the contig and hom-ref-fill its column.

    Uses tabix (`.tbi`) indexes to match real PURPLE `*.somatic.vcf.gz` inputs
    (both `.tbi` and `bcftools index` `.csi` reproduced the crash pre-fix).
    """
    ref = tmp_path / "ref.fa"
    ref.write_text(">chr1\n" + "A" * 40 + "\n>chr2\n" + "A" * 40 + "\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        "##contig=<ID=chr2,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{s}\n"
    )

    def ss(name: str, sample: str, rows: str) -> Path:
        plain = tmp_path / f"{name}.vcf"
        plain.write_text(header.format(s=sample) + rows)
        gz = tmp_path / f"{name}.vcf.gz"
        with open(gz, "wb") as fh:
            subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
        subprocess.run(["tabix", "-p", "vcf", str(gz)], check=True)
        return gz

    # file A carries records on BOTH contigs; file B ONLY on chr1 -- its index
    # has no chr2 entry, so seeking chr2 is exactly the failing operation.
    a = ss(
        "a",
        "SA",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\nchr2\t6\t.\tA\tG\t.\t.\t.\tGT\t1|1\n",
    )
    b = ss("b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n")

    out = tmp_path / "store"
    dropped = SparseVar2.from_vcf_list(out, [a, b], ref, threads=1)
    assert dropped == 0
    sv = SparseVar2(out)
    assert sv.available_samples == ["SA", "SB"]

    # chr2: only SA carries (hom-alt, both haps); SB is hom-ref filled, NOT a
    # crash. (R,S,P) -> [SA_h0, SA_h1, SB_h0, SB_h1].
    counts_chr2 = sv.region_counts("chr2", [(0, 40)]).reshape(-1)
    assert counts_chr2.tolist() == [1, 1, 0, 0]
    # chr1 is unaffected: both files carry one hap each.
    counts_chr1 = sv.region_counts("chr1", [(0, 40)]).reshape(-1)
    assert counts_chr1.tolist() == [1, 0, 0, 1]


def test_from_vcf_list_regions_restricts(tmp_path: Path):
    """`regions` restricts the k-way merge to the requested interval, exactly
    as it restricts `from_vcf` -- exercised here through `from_vcf_list`'s
    delegation into `orchestrator::run_vcf_list`'s per-contig `SourceSpec::
    VcfList { regions, overlap }`, not the single-file `SourceSpec::Vcf` path.

    SA's SNP@0-based-pos-2 (VCF POS 3) is INSIDE `chr1:1-4` (1-based
    inclusive -> 0-based half-open [0,4)); SB's insertion@0-based-pos-6 (VCF
    POS 7) is OUTSIDE it. Both samples must still appear in the store (region
    filtering is per-variant, not per-sample), but only SA's variant should
    survive -- proven both directly (`region_counts` on the full contig span)
    and relative to an unrestricted conversion of the identical inputs.
    """
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")

    out = tmp_path / "vl_regions"
    SparseVar2.from_vcf_list(out, [a, b], ref, regions="chr1:1-4", threads=1)
    sv = SparseVar2(out)
    assert sv.available_samples == ["SA", "SB"]
    restricted_count = int(sv.region_counts("chr1", [(0, 40)]).sum())
    assert restricted_count >= 1

    full = tmp_path / "vl_full"
    SparseVar2.from_vcf_list(full, [a, b], ref, threads=1)
    full_count = int(SparseVar2(full).region_counts("chr1", [(0, 40)]).sum())
    assert restricted_count < full_count


def test_from_vcf_list_regions_no_match_raises(tmp_path: Path):
    """A region that doesn't overlap any input contig must raise, not
    silently produce an empty/nonsensical store."""
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    with pytest.raises(ValueError, match="[Nn]o requested regions"):
        SparseVar2.from_vcf_list(
            tmp_path / "s", [a], ref, regions="chr2:1-4", threads=1
        )


def test_from_vcf_list_regions_overlap_variant_rejects_multiple_regions_per_contig(
    tmp_path: Path,
):
    """`regions_overlap="variant"` with 2+ disjoint regions on the same
    contig is unsound (see `genoray._svar2._reject_multiregion_variant`) and
    must raise rather than silently double-count variants."""
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    with pytest.raises(ValueError, match="at most one region per contig"):
        SparseVar2.from_vcf_list(
            tmp_path / "vl_multi_variant",
            [a, b],
            ref,
            regions=["chr1:1-4", "chr1:7-12"],
            regions_overlap="variant",
            threads=1,
        )


def test_from_vcf_list_shared_site_one_variant(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "s"
    dropped = SparseVar2.from_vcf_list(out, [a, b], ref, threads=1)
    assert dropped == 0
    sv = SparseVar2(out)
    rag = sv.decode("chr1", [(0, 40)])
    lengths = rag["pos"].lengths.reshape(-1)
    assert lengths.tolist() == [1, 0, 0, 1]  # same site, one hap each


def test_from_vcf_list_directory_and_manifest_equivalent(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n")
    SparseVar2.from_vcf_list(tmp_path / "by_seq", [a, b], ref, threads=1)
    # directory: put both gz+csi in a subdir
    d = tmp_path / "vcfs"
    d.mkdir()
    for p in (a, b):
        (d / p.name).write_bytes(p.read_bytes())
        (d / (p.name + ".csi")).write_bytes(Path(str(p) + ".csi").read_bytes())
    SparseVar2.from_vcf_list(tmp_path / "by_dir", d, ref, threads=1)
    manifest = tmp_path / "m.txt"
    manifest.write_text(f"# comment\n{a}\n\n{b}\n")
    SparseVar2.from_vcf_list(tmp_path / "by_manifest", manifest, ref, threads=1)
    for name in ("by_dir", "by_manifest"):
        assert SparseVar2(tmp_path / name).available_samples == ["SA", "SB"]


def test_from_vcf_list_rejects_multisample(tmp_path: Path):
    ref = _write_ref(tmp_path)
    two = _ss(
        tmp_path,
        "two",
        "SA\tSB",  # header hack: two sample cols
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|1\n",
    )
    with pytest.raises(ValueError, match="single-sample"):
        SparseVar2.from_vcf_list(tmp_path / "s", [two], ref, threads=1)


def test_from_vcf_list_rejects_duplicate_samples(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "S", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "S", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    with pytest.raises(ValueError, match="duplicate|collision"):
        SparseVar2.from_vcf_list(tmp_path / "s", [a, b], ref, threads=1)


def test_from_vcf_list_requires_reference(tmp_path: Path):
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_vcf_list(tmp_path / "s", [a], threads=1)


def test_from_vcf_list_no_reference_snp_only(tmp_path):
    # SNP-only single-sample VCFs, no reference. Should convert and hom-ref fill.
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t9\t.\tG\tT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "s"
    SparseVar2.from_vcf_list(out, [a, b], no_reference=True, threads=1)
    sv = SparseVar2(out)
    assert sv.available_samples == ["SA", "SB"]
    assert sv.region_counts("chr1", [(0, 40)]).reshape(-1).tolist() == [1, 0, 0, 1]


def test_from_vcf_list_no_reference_matches_bcftools_merge_oracle(tmp_path: Path):
    """`no_reference` oracle parity, mirroring
    `test_from_vcf_list_matches_bcftools_merge_oracle` but with `no_reference=True`
    on both sides (`from_vcf` and `from_vcf_list`).

    Under `no_reference`, atoms are NOT left-aligned -- cross-file joins only
    line up when every input already agrees on how a shared site is
    represented. So this fixture is restricted to SNPs (position + REF/ALT
    bytes alone determine identity; there is no representational ambiguity
    to normalize away), unlike the reference-bearing oracle test which also
    covers indels/multiallelics.
    """
    a = _ss(
        tmp_path,
        "a",
        "SA",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n"  # shared SNP
        "chr1\t12\t.\tG\tC\t.\t.\t.\tGT\t0|1\n",  # private SNP
    )
    b = _ss(
        tmp_path,
        "b",
        "SB",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n"  # shared SNP
        "chr1\t9\t.\tG\tT\t.\t.\t.\tGT\t1|0\n",  # private SNP
    )
    paths = [a, b]

    merge = subprocess.run(
        ["bcftools", "merge", "-0", *map(str, paths)],
        check=True,
        stdout=subprocess.PIPE,
    )
    merged = tmp_path / "merged.vcf.gz"
    with open(merged, "wb") as fh:
        subprocess.run(["bgzip", "-c"], input=merge.stdout, check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(merged)], check=True)

    from_vcf_out = tmp_path / "oracle"
    SparseVar2.from_vcf(from_vcf_out, merged, no_reference=True, threads=1)
    list_out = tmp_path / "list"
    dropped = SparseVar2.from_vcf_list(list_out, paths, no_reference=True, threads=1)
    assert dropped == 0

    oracle, native = SparseVar2(from_vcf_out), SparseVar2(list_out)
    assert oracle.available_samples == native.available_samples == ["SA", "SB"]

    region = [(0, len(_REF))]
    # SA: hap0 <- shared SNP@2, hap1 <- private SNP@11
    # SB: hap0 <- private SNP@8, hap1 <- shared SNP@2
    expected_counts = np.array([[[1, 1], [1, 1]]])
    oracle_counts = oracle.region_counts("chr1", region)
    native_counts = native.region_counts("chr1", region)
    np.testing.assert_array_equal(oracle_counts, expected_counts)
    np.testing.assert_array_equal(oracle_counts, native_counts)

    ro, rl = oracle.decode("chr1", region), native.decode("chr1", region)
    # 4 ALT-carrying (sample, ploid) entries across the 3 distinct atoms (the
    # shared SNP@2 is carried by both SA and SB), matching `expected_counts`.
    assert len(np.asarray(ro["pos"].data)) == 4
    for field in ("pos", "ilen"):
        np.testing.assert_array_equal(
            np.asarray(ro[field].data), np.asarray(rl[field].data)
        )
    assert ro["allele"].to_ak().tolist() == rl["allele"].to_ak().tolist()


def test_from_vcf_list_fields_multicontig_matches_dense_oracle(tmp_path: Path):
    """Carrier-path FORMAT (`from_vcf_list`) must byte-match dense-path FORMAT
    (`from_vcf` over a `bcftools merge`), across >1 contig, for requested
    FORMAT fields. This is the end-to-end gate for route-before-densify: it
    fails if carrier-sparse FORMAT resolution diverges from the dense grid
    on any (sample, field) -- the prior tests in this file only exercised
    genotype-level (pos/ilen/allele) parity or FORMAT on a single contig.

    Restricted to SNPs on two contigs (`chr1`, `chr2`), matching the
    `no_reference` oracle's SNP restriction above (no reference FASTA is
    needed, but atoms aren't left-aligned, so only SNPs are safe to compare
    across independently-authored files).
    """
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        "##contig=<ID=chr2,length=1000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Alt frac">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{s}\n"
    )

    def ss(name: str, sample: str, rows: str) -> Path:
        plain = tmp_path / f"{name}.vcf"
        plain.write_text(header.format(s=sample) + rows)
        gz = tmp_path / f"{name}.vcf.gz"
        with open(gz, "wb") as fh:
            subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
        subprocess.run(["bcftools", "index", str(gz)], check=True)
        return gz

    a = ss(
        "a",
        "SA",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DP:VAF\t1|0:30:0.5\n"  # shared chr1 SNP
        "chr1\t12\t.\tG\tC\t.\t.\t.\tGT:DP:VAF\t0|1:22:0.9\n"  # private chr1
        "chr2\t5\t.\tT\tA\t.\t.\t.\tGT:DP:VAF\t1|0:11:0.3\n",  # private chr2
    )
    b = ss(
        "b",
        "SB",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DP:VAF\t0|1:18:0.4\n"  # shared chr1 SNP
        "chr2\t5\t.\tT\tA\t.\t.\t.\tGT:DP:VAF\t0|1:27:0.7\n",  # shared chr2 SNP
    )
    paths = [a, b]

    merge = subprocess.run(
        ["bcftools", "merge", "-0", *map(str, paths)],
        check=True,
        stdout=subprocess.PIPE,
    )
    merged = tmp_path / "merged.vcf.gz"
    with open(merged, "wb") as fh:
        subprocess.run(["bgzip", "-c"], input=merge.stdout, check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(merged)], check=True)

    fields = dict(format_fields=["DP", "VAF"])
    dense_out, list_out = tmp_path / "dense", tmp_path / "list"
    SparseVar2.from_vcf(dense_out, merged, no_reference=True, threads=1, **fields)
    dropped = SparseVar2.from_vcf_list(
        list_out, paths, no_reference=True, threads=1, **fields
    )
    assert dropped == 0

    dense = SparseVar2(dense_out).with_fields(["DP", "VAF"])
    native = SparseVar2(list_out).with_fields(["DP", "VAF"])
    assert dense.available_samples == native.available_samples == ["SA", "SB"]

    for contig, length in [("chr1", 1000), ("chr2", 1000)]:
        region = [(0, length)]
        d = dense.decode(contig, region)
        n = native.decode(contig, region)
        # Genotype-level parity (positions) AND FORMAT parity. Compare every
        # array the decode returns; the FORMAT field arrays (DP, VAF) are the
        # ones this change touches -- DP (int) and VAF (float, Number=A) are
        # both simple decimal literals (0.5, 0.9, ...) that round-trip
        # exactly through float32, so exact equality is the right bar (not
        # allclose) -- a genuine carrier-vs-grid resolution divergence should
        # not be masked by a tolerance.
        for key in ("pos", "DP", "VAF"):
            np.testing.assert_array_equal(
                np.asarray(d[key].data),
                np.asarray(n[key].data),
                err_msg=f"{contig}:{key} diverged between dense and carrier paths",
            )


def test_from_vcf_list_matches_bcftools_merge_oracle(tmp_path: Path):
    """Oracle parity: `bcftools merge -0` (missing -> hom-ref, exactly our
    semantics) -> `from_vcf` must equal the native `from_vcf_list` k-way merge.

    Mix: a shared SNP (a+b, same site), a private INS (a only), a
    multiallelic split (b only), an anchored DEL with a missing hap (c only),
    and a cross-file join onto b's NON-FIRST ALT (d only) -- and b's
    multiallelic site and c's DEL share POS 7 but differ in ILEN/ALT, so they
    must NOT be spuriously joined.

    The assertions are split into two kinds:

    - ABSOLUTE: `dropped == 0`, the exact `region_counts` array, and the
      exact total decoded-entry count. These are hand-derived from the
      fixture below and pin *reality*, not merely oracle==native agreement --
      without them, a shared-layer regression that dropped every record from
      BOTH stores (oracle and native alike) would still pass the relative
      checks vacuously (empty == empty).
    - RELATIVE (oracle vs. native): region_counts, decoded pos/ilen/allele.
      These catch anything the hand-derived numbers above wouldn't (e.g. a
      difference in *which* records/positions appear despite the same total
      counts).
    """
    ref = _write_ref(tmp_path)
    a = _ss(
        tmp_path,
        "a",
        "SA",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n"  # shared SNP
        "chr1\t12\t.\tG\tGA\t.\t.\t.\tGT\t0|1\n",  # private INS
    )
    b = _ss(
        tmp_path,
        "b",
        "SB",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n"  # shared SNP
        "chr1\t7\t.\tC\tG,T\t.\t.\t.\tGT\t1|2\n",  # multiallelic
    )
    c = _ss(
        tmp_path,
        "c",
        "SC",
        "chr1\t7\t.\tCAT\tC\t.\t.\t.\tGT\t1|.\n",  # anchored DEL + missing hap
    )
    d = _ss(
        tmp_path,
        "d",
        "SD",
        # Joins b's ALT index 2 (T), NOT index 1 (G) -- this is the only
        # place in the whole test suite (Python or Rust) where a cross-file
        # join lands on a non-first source ALT index, exercising the
        # per-file local-ALT-index -> joined-record-slot remap across files
        # rather than within one file.
        "chr1\t7\t.\tC\tT\t.\t.\t.\tGT\t1|0\n",
    )
    paths = [a, b, c, d]

    # Oracle: bcftools merge -0 (missing genotypes -> 0/0, matching our
    # hom-ref-fill semantics) -> bgzip -> index -> from_vcf.
    merge = subprocess.run(
        ["bcftools", "merge", "-0", *map(str, paths)],
        check=True,
        stdout=subprocess.PIPE,
    )
    merged = tmp_path / "merged.vcf.gz"
    with open(merged, "wb") as fh:
        subprocess.run(["bgzip", "-c"], input=merge.stdout, check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(merged)], check=True)

    from_vcf_out = tmp_path / "oracle"
    SparseVar2.from_vcf(from_vcf_out, merged, ref, threads=1)
    list_out = tmp_path / "list"
    dropped = SparseVar2.from_vcf_list(list_out, paths, ref, threads=1)
    assert dropped == 0

    oracle, native = SparseVar2(from_vcf_out), SparseVar2(list_out)
    assert (
        oracle.available_samples == native.available_samples == ["SA", "SB", "SC", "SD"]
    )

    region = [(0, len(_REF))]

    # ABSOLUTE: hand-derived expected (region, sample, ploid) counts, so a
    # shared-layer regression that zeroed BOTH stores can't pass vacuously.
    # Per-hap ALT-carrier tally from the fixture above (5 distinct atoms:
    # SNP@2, INS@11, C>G@6, C>T@6, DEL@6):
    #   SA: hap0 <- SNP@2 (shared w/ SB),        hap1 <- private INS@11
    #   SB: hap0 <- C>G@6 (ALT idx 1),            hap1 <- SNP@2 (shared w/ SA)
    #       AND C>T@6 (ALT idx 2, joins d) -> 2 entries on hap1
    #   SC: hap0 <- anchored DEL@6,               hap1 <- missing (unobservable
    #       via decode/region_counts -- see
    #       test_from_vcf_list_missing_hap_is_unobservable_in_decode)
    #   SD: hap0 <- C>T@6 (ALT idx 1 locally, joins b's ALT idx 2),
    #       hap1 <- hom-ref
    expected_counts = np.array(
        [
            [
                [1, 1],  # SA
                [1, 2],  # SB
                [1, 0],  # SC
                [1, 0],  # SD
            ]
        ]
    )
    oracle_counts = oracle.region_counts("chr1", region)
    native_counts = native.region_counts("chr1", region)
    np.testing.assert_array_equal(oracle_counts, expected_counts)
    # Sum of the per-hap tally above: 2 (SA) + 3 (SB) + 1 (SC) + 1 (SD) = 7
    # total ALT-carrying (sample, ploid) entries across the 5 atoms.
    assert int(oracle_counts.sum()) == 7

    # RELATIVE: oracle vs. native must still agree exactly.
    np.testing.assert_array_equal(oracle_counts, native_counts)

    ro, rl = oracle.decode("chr1", region), native.decode("chr1", region)
    assert len(np.asarray(ro["pos"].data)) == 7
    for field in ("pos", "ilen"):
        np.testing.assert_array_equal(
            np.asarray(ro[field].data), np.asarray(rl[field].data)
        )
    # allele: variable-length ALT bytes per (sample, ploid, variant); pure
    # deletions decode to an empty ALT (anchor base is implicit) on BOTH
    # sides, so a like-for-like comparison is still meaningful.
    assert ro["allele"].to_ak().tolist() == rl["allele"].to_ak().tolist()


def test_from_vcf_list_missing_hap_is_unobservable_in_decode(tmp_path: Path):
    """Documents a gap, doesn't close it: `_svar2.py`'s `from_vcf_list`
    docstring says a within-file `./.` (missing) call decodes to `-1`,
    *distinct* from a hom-ref fill (`0`) for a sample lacking a site
    entirely. That `-1`/`0` distinction is real in the internal per-file
    `ploid_codes`/`RawRecord.gt` representation (see
    `src/vcf_list_reader.rs`), but neither `decode()` nor `region_counts()`
    exposes it: the sparse layout only ever stores ALT-*carrying* entries, so
    a missing hap and a hom-ref hap both decode to zero entries -- there is
    no public API surface to tell them apart.

    This test pins that non-distinction (both wind up indistinguishable, and
    both equal between the bcftools-merge oracle and the native
    `from_vcf_list` merge), rather than claiming to evidence the `-1` vs `0`
    split -- no test in this suite currently observes that split through a
    public method.
    """
    ref = _write_ref(tmp_path)
    # SA: ordinary hom-ref hap (hap1) with no site at all.
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    # SB: explicit missing hap (hap1) at the SAME site as SA's hom-ref hap.
    b = _ss(tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|.\n")
    paths = [a, b]

    merge = subprocess.run(
        ["bcftools", "merge", "-0", *map(str, paths)],
        check=True,
        stdout=subprocess.PIPE,
    )
    merged = tmp_path / "merged.vcf.gz"
    with open(merged, "wb") as fh:
        subprocess.run(["bgzip", "-c"], input=merge.stdout, check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(merged)], check=True)

    from_vcf_out = tmp_path / "oracle"
    SparseVar2.from_vcf(from_vcf_out, merged, ref, threads=1)
    list_out = tmp_path / "list"
    dropped = SparseVar2.from_vcf_list(list_out, paths, ref, threads=1)
    assert dropped == 0

    oracle, native = SparseVar2(from_vcf_out), SparseVar2(list_out)
    region = [(0, len(_REF))]
    # Both SA's hom-ref hap1 and SB's missing hap1 decode to 0 entries --
    # indistinguishable from each other, and identical between oracle/native.
    expected = np.array([[[1, 0], [1, 0]]])
    np.testing.assert_array_equal(oracle.region_counts("chr1", region), expected)
    np.testing.assert_array_equal(native.region_counts("chr1", region), expected)


def test_from_vcf_list_format_field_per_sample(tmp_path: Path):
    """FORMAT fields are per-sample: each carrier contributes its OWN file's
    value at a shared site, not some other file's value or an aggregate."""
    ref = _write_ref(tmp_path)
    # SA is hom-alt (both haps carry) with DP=10; SB carries only hap1 with
    # DP=20. SC carries nothing at this site at all -- a disjoint-site
    # sample included to prove its presence doesn't perturb SA/SB's values
    # (the non-carrier-gets-default value itself is asserted directly on the
    # Rust `RawRecord` in
    # `vcf_list_reader::tests::info_first_carrier_and_format_per_sample_with_non_carrier_default`,
    # since `decode()` never emits a record for a (sample, ploid) cell that
    # doesn't carry an ALT at all -- see
    # `test_from_vcf_list_missing_hap_is_unobservable_in_decode` above for
    # the same limitation on the genotype side).
    a = _ss_fields(
        tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\tAF=0.5\tGT:DP\t1|1:10\n"
    )
    b = _ss_fields(
        tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\tAF=0.5\tGT:DP\t0|1:20\n"
    )
    # 0-based pos 5 ('A' in `_REF`), unrelated to SA/SB's shared pos2 site.
    c = _ss_fields(
        tmp_path, "c", "SC", "chr1\t6\t.\tA\tC\t.\t.\tAF=0.5\tGT:DP\t1|1:99\n"
    )
    out = tmp_path / "store"
    dropped = SparseVar2.from_vcf_list(
        out, [a, b, c], ref, threads=1, format_fields=["DP"]
    )
    assert dropped == 0

    sv = SparseVar2(out).with_fields(["DP"])
    rag = sv.decode("chr1", [(0, len(_REF))])

    # Flat (R,S,P) cell order: SA_h0, SA_h1, SB_h0, SB_h1, SC_h0, SC_h1.
    # SA carries both haps at pos2; SB only hap1; SC carries nothing there
    # (its own site is pos5, a separate decoded record).
    lengths = rag["pos"].lengths.reshape(-1)
    assert lengths.tolist() == [1, 1, 0, 1, 1, 1]

    dp = np.asarray(rag["DP"].data)
    # Order follows the carrier cells above: SA_h0, SA_h1, SB_h1, SC_h0, SC_h1.
    # Each carrier's DP is its OWN file's value -- if the merge instead took
    # e.g. the first-carrier's value for every column (an INFO/FORMAT mixup),
    # SB's and SC's DP would wrongly read back as 10.
    assert dp.tolist() == [10, 10, 20, 99, 99]


def test_from_vcf_list_info_first_carrier(tmp_path: Path):
    """INFO fields merge first-carrier-wins: at a shared site, the value
    comes from the LOWEST-numbered (earliest in `sources` order) file that
    carries the atom -- proven here by giving the two files DIFFERENT AF
    values, so taking the last carrier or e.g. their max would fail this."""
    ref = _write_ref(tmp_path)
    # A is FIRST in the list and carries AF=0.1; B is second and carries
    # AF=0.9 at the SAME shared SNP.
    a = _ss_fields(
        tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\tAF=0.1\tGT:DP\t1|0:1\n"
    )
    b = _ss_fields(
        tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\tAF=0.9\tGT:DP\t0|1:2\n"
    )
    out = tmp_path / "store"
    dropped = SparseVar2.from_vcf_list(out, [a, b], ref, threads=1, info_fields=["AF"])
    assert dropped == 0

    sv = SparseVar2(out).with_fields(["AF"])
    rag = sv.decode("chr1", [(0, len(_REF))])

    # Both SA_h0 and SB_h1 carry the ONE shared merged record, so both must
    # report the SAME first-carrier value (0.1) -- if the code instead
    # resolved each column's INFO independently (treating it like a FORMAT
    # field) SB's entry would wrongly read back as 0.9.
    lengths = rag["pos"].lengths.reshape(-1)
    assert lengths.tolist() == [1, 0, 0, 1]
    af = np.asarray(rag["AF"].data)
    np.testing.assert_allclose(af, np.array([0.1, 0.1], dtype=np.float32), atol=1e-6)


def test_from_vcf_list_rejects_mixed_contig_naming(tmp_path: Path):
    """I1 (final review): a cohort mixing UCSC-style (`chr1`) and
    Ensembl-style (`1`) contig naming across files must raise a clear error
    up front, not silently produce a half-hom-ref-filled store. The native
    merge matches contigs by an exact per-file string (`VcfListRecordSource`
    opens each file with ONE `chrom` literal), so without this check the two
    naming schemes would union into two separate "contigs" (`chr1` and `1`),
    each converting only the files using that spelling -- `decode("chr1",
    ...)` would silently read back all-zeros for every file that used the
    OTHER spelling, with no error and no warning.
    """
    a = _ss_contig(tmp_path, "a", "SA", "chr1", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss_contig(tmp_path, "b", "SB", "1", "1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n")
    with pytest.raises(ValueError, match="[Ii]nconsistent contig naming"):
        SparseVar2.from_vcf_list(tmp_path / "s", [a, b], no_reference=True, threads=1)


def test_from_vcf_list_rejects_bare_uncompressed_vcf_source(tmp_path: Path):
    """M3 (final review): a plain (uncompressed) `.vcf` handed as the single
    `sources` path must be rejected outright with the same `_ensure_bgzipped`
    message used elsewhere -- NOT silently fall into `_resolve_vcf_sources`'s
    manifest branch, where every `##`/`#CHROM` header line reads as a
    `#`-comment (skipped) and every data line is then treated as a bogus
    input *path*, producing a bewildering downstream error far from the real
    problem.
    """
    ref = _write_ref(tmp_path)
    plain = tmp_path / "cohort.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSA\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n"
    )
    with pytest.raises(ValueError, match="bgzip"):
        SparseVar2.from_vcf_list(tmp_path / "s", plain, ref, threads=1)


def test_from_vcf_list_rejects_signatures_with_no_reference(tmp_path: Path):
    """M6 (final review): pins the existing `signatures=True` +
    `no_reference=True` guard for `from_vcf_list` specifically (the guard
    itself predates this review; only the pinning test was missing)."""
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    with pytest.raises(ValueError, match="signatures"):
        SparseVar2.from_vcf_list(
            tmp_path / "s", [a], no_reference=True, signatures=True, threads=1
        )


def test_check_fd_budget_raises_actionable_error_when_limit_too_low(monkeypatch):
    """I3 (final review): at large N, opening every input file concurrently
    can exhaust the process's open-file limit; `_check_fd_budget` must catch
    this up front with a clear, actionable (`ulimit -n ...`) error instead of
    letting htslib's confusing "is there a .tbi or .csi file?" message reach
    the user for some arbitrary file near the ceiling.

    Exercised directly against `_check_fd_budget` (rather than actually
    opening 1000+ files, which would be slow and environment-dependent);
    `resource.setrlimit` is monkeypatched to fail so the "raise it ourselves"
    fast path is also forced to fall through to the error.
    """
    import resource

    from genoray._svar2 import _check_fd_budget

    monkeypatch.setattr(resource, "getrlimit", lambda which: (256, 256))

    def _raise(which, limits):
        raise OSError("simulated: cannot raise rlimit")

    monkeypatch.setattr(resource, "setrlimit", _raise)

    with pytest.raises(ValueError, match="ulimit -n"):
        _check_fd_budget(1000)


def test_check_fd_budget_raises_soft_limit_when_hard_limit_allows(monkeypatch):
    """I3 (final review): when the HARD limit already permits it, `_check_fd_
    budget` should transparently raise the soft limit itself rather than
    erroring -- this is the "and/or raise the soft limit toward the hard
    limit if that's clean" half of the fix."""
    import resource

    from genoray._svar2 import _check_fd_budget

    monkeypatch.setattr(resource, "getrlimit", lambda which: (256, 4096))
    raised: dict[str, tuple[int, int]] = {}

    def _setrlimit(which, limits):
        raised["limits"] = limits

    monkeypatch.setattr(resource, "setrlimit", _setrlimit)

    _check_fd_budget(1000)  # must not raise
    assert raised["limits"][0] >= 1000 * 2 + 64


def test_from_vcf_list_check_ref_error_aborts(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "A", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(
        tmp_path, "b", "B", "chr1\t10\t.\tA\tT\t.\t.\t.\tGT\t0|1\n"
    )  # REF=A, _REF[10]='G'
    with pytest.raises(Exception):
        SparseVar2.from_vcf_list(tmp_path / "s", [a, b], ref, check_ref="e", threads=1)


def test_from_vcf_list_check_ref_exclude_continues(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "A", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "B", "chr1\t10\t.\tA\tT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "store"
    SparseVar2.from_vcf_list(out, [a, b], ref, check_ref="x", threads=1)
    assert (out / "meta.json").exists()  # merge completed; b's bad record excluded
    sv = SparseVar2(out)
    counts = sv.region_counts("chr1", [(0, 40)])
    assert int(counts.sum()) == 1  # only a's clean pos-3 record survives


def test_from_vcf_list_auto_chunk_size(tmp_path, monkeypatch):
    """chunk_size=None (default) wires in the budget-derived _auto_chunk_size;
    an explicit int passes through unchanged (back-compat).

    Note: _auto_chunk_size(2, 2) == 25_000 == the old fixed default, so only a
    sentinel return value distinguishes "wired the budget path" from "kept the
    old constant". We patch _auto_chunk_size to a sentinel and assert the
    pipeline receives it.
    """
    import genoray._svar2 as sv2

    seen = {}
    real_pipeline = sv2._core.run_vcf_list_conversion_pipeline

    def spy(paths, ref, contigs, out, samples, contig_membership, chunk_size, *rest):
        seen["chunk_size"] = chunk_size
        return real_pipeline(
            paths, ref, contigs, out, samples, contig_membership, chunk_size, *rest
        )

    monkeypatch.setattr(sv2._core, "run_vcf_list_conversion_pipeline", spy)
    monkeypatch.setattr(
        sv2,
        "_auto_chunk_size",
        lambda n_samples, ploidy=2, n_format_fields=0, max_mem=None: 7777,
    )

    a = _ss(tmp_path, "a", "S0", "chr1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\n")
    b = _ss(tmp_path, "b", "S1", "chr1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\n")

    # default: chunk_size omitted -> budget-derived path used (not the old 25_000)
    SparseVar2.from_vcf_list(
        tmp_path / "out", [a, b], no_reference=True, overwrite=True
    )
    assert seen["chunk_size"] == 7777

    # explicit override still passes through verbatim
    seen.clear()
    SparseVar2.from_vcf_list(
        tmp_path / "out2", [a, b], no_reference=True, overwrite=True, chunk_size=321
    )
    assert seen["chunk_size"] == 321

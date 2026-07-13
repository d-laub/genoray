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


def test_from_vcf_list_matches_bcftools_merge_oracle(tmp_path: Path):
    """Oracle parity: `bcftools merge -0` (missing -> hom-ref, exactly our
    semantics) -> `from_vcf` must equal the native `from_vcf_list` k-way merge.

    Mix: a shared SNP (a+b, same site), a private INS (a only), a
    multiallelic split (b only), an anchored DEL with a missing hap (c only)
    -- and b's multiallelic site and c's DEL share POS 7 but differ in
    ILEN/ALT, so they must NOT be spuriously joined.
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
    paths = [a, b, c]

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
    SparseVar2.from_vcf_list(list_out, paths, ref, threads=1)

    oracle, native = SparseVar2(from_vcf_out), SparseVar2(list_out)
    assert oracle.available_samples == native.available_samples == ["SA", "SB", "SC"]

    region = [(0, len(_REF))]
    np.testing.assert_array_equal(
        oracle.region_counts("chr1", region), native.region_counts("chr1", region)
    )

    ro, rl = oracle.decode("chr1", region), native.decode("chr1", region)
    for field in ("pos", "ilen"):
        np.testing.assert_array_equal(
            np.asarray(ro[field].data), np.asarray(rl[field].data)
        )
    # allele: variable-length ALT bytes per (sample, ploid, variant); pure
    # deletions decode to an empty ALT (anchor base is implicit) on BOTH
    # sides, so a like-for-like comparison is still meaningful.
    assert ro["allele"].to_ak().tolist() == rl["allele"].to_ak().tolist()

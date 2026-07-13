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

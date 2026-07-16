# tests/test_svar2_from_vcf_list_parity.py
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path

from genoray import SparseVar2

_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=1000>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
)


def _ss(d: Path, i: int, rows: str) -> Path:
    plain = d / f"s{i}.vcf"
    plain.write_text(_HEADER.format(sample=f"S{i}") + rows)
    gz = d / f"s{i}.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def hash_store(store_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(store_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(store_dir).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _cohort(d: Path) -> list[str]:
    # 3 files, some shared sites (POS 100 shared across all), some private, one indel.
    a = _ss(
        d,
        0,
        "chr1\t100\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\nchr1\t200\t.\tG\tT\t.\tPASS\t.\tGT\t1/1\n",
    )
    b = _ss(
        d,
        1,
        "chr1\t100\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\nchr1\t300\t.\tC\tCA\t.\tPASS\t.\tGT\t1/1\n",
    )
    c = _ss(
        d,
        2,
        "chr1\t100\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\nchr1\t400\t.\tT\tG\t.\tPASS\t.\tGT\t1/1\n",
    )
    return [str(a), str(b), str(c)]


def test_parity_chunk_size_invariant(tmp_path: Path):
    paths = _cohort(tmp_path)
    out_small = tmp_path / "small"
    out_large = tmp_path / "large"
    SparseVar2.from_vcf_list(
        out_small, paths, no_reference=True, overwrite=True, chunk_size=1
    )
    SparseVar2.from_vcf_list(
        out_large, paths, no_reference=True, overwrite=True, chunk_size=25_000
    )
    assert hash_store(out_small) == hash_store(out_large)


def test_parity_repeatable(tmp_path: Path):
    paths = _cohort(tmp_path)
    o1, o2 = tmp_path / "a", tmp_path / "b"
    SparseVar2.from_vcf_list(o1, paths, no_reference=True, overwrite=True)
    SparseVar2.from_vcf_list(o2, paths, no_reference=True, overwrite=True)
    assert hash_store(o1) == hash_store(o2)

"""Regression tests for issue #166.

BGZF-compressed VCFs are conventionally named `.vcf.gz`, but `.vcf.bgz` names
the exact same BGZF bytes and is what several large cohorts ship (All of Us v9
phased callsets, e.g. `chr20.aou.v9.phased.filtered.vcf.bgz` + a `.tbi`).
genoray's suffix guards accepted only `.vcf.gz`/`.bcf`, so a valid bgzipped VCF
was rejected with "must be a BCF (.bcf) or bgzipped VCF (.vcf.gz); bgzip it
first." purely because of its name -- renaming the identical bytes to `.vcf.gz`
made it work.

Every test here builds ONE bgzipped VCF and copies its exact bytes to both
suffixes, so any behavioral difference is attributable to the name alone.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from genoray import VCF, SparseVar2
from genoray._utils import variant_file_type

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _write_ref(d: Path) -> Path:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    return ref


def _bgzipped_vcf(
    d: Path,
    name: str = "in",
    *,
    sample: str = "S0",
    rows: str = "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n",
    suffix: str = ".vcf.gz",
) -> Path:
    """A bgzipped, `.csi`-indexed single-sample VCF written under `suffix`."""
    plain = d / f"{name}.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n" + rows
    )
    out = d / f"{name}{suffix}"
    with open(out, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(out)], check=True)
    plain.unlink()
    return out


def _same_bytes_as_bgz(gz: Path) -> Path:
    """Copy `gz` (and its index) verbatim to the `.vcf.bgz` spelling."""
    bgz = gz.with_name(gz.name[: -len(".vcf.gz")] + ".vcf.bgz")
    shutil.copyfile(gz, bgz)
    for idx in (".csi", ".tbi"):
        if (gz.parent / (gz.name + idx)).exists():
            shutil.copyfile(gz.parent / (gz.name + idx), bgz.parent / (bgz.name + idx))
    return bgz


def test_ensure_bgzipped_accepts_vcf_bgz():
    """The suffix guard itself: `.vcf.bgz` is a bgzipped VCF, not a bare one."""
    from genoray._svar2 import _ensure_bgzipped

    _ensure_bgzipped(Path("cohort.vcf.bgz"))  # must not raise
    _ensure_bgzipped(Path("chr20.aou.v9.phased.filtered.vcf.bgz"))


def test_ensure_bgzipped_still_rejects_plain_vcf():
    """The `.bgz` alias must not weaken the guard against uncompressed VCFs."""
    from genoray._svar2 import _ensure_bgzipped

    with pytest.raises(ValueError, match="bgzip"):
        _ensure_bgzipped(Path("cohort.vcf"))


def test_resolve_vcf_sources_single_vcf_bgz(tmp_path: Path):
    """A single `.vcf.bgz` path resolves to that one file -- NOT to the
    manifest branch, which would read its (binary) contents as a path list."""
    from genoray._svar2 import _resolve_vcf_sources

    bgz = _bgzipped_vcf(tmp_path, suffix=".vcf.bgz")
    assert _resolve_vcf_sources(bgz) == [bgz]


def test_resolve_vcf_sources_directory_discovers_vcf_bgz(tmp_path: Path):
    """Directory discovery globs `.vcf.bgz` alongside `.vcf.gz`, with BCFs
    still last."""
    from genoray._svar2 import _resolve_vcf_sources

    d = tmp_path / "vcfs"
    d.mkdir()
    a = _bgzipped_vcf(d, "a", sample="SA", suffix=".vcf.bgz")
    b = _bgzipped_vcf(d, "b", sample="SB", suffix=".vcf.gz")
    assert _resolve_vcf_sources(d) == [a, b]


def test_variant_file_type_recognizes_vcf_bgz():
    assert variant_file_type("cohort.vcf.bgz") == "vcf"


def test_from_vcf_bgz_matches_vcf_gz(tmp_path: Path):
    """End-to-end: identical BGZF bytes under both suffixes convert to stores
    with identical contents."""
    ref = _write_ref(tmp_path)
    gz = _bgzipped_vcf(tmp_path, rows="chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    bgz = _same_bytes_as_bgz(gz)

    dropped_gz = SparseVar2.from_vcf(tmp_path / "gz.svar2", gz, ref)
    dropped_bgz = SparseVar2.from_vcf(tmp_path / "bgz.svar2", bgz, ref)
    assert dropped_gz == dropped_bgz

    sv_gz = SparseVar2(tmp_path / "gz.svar2")
    sv_bgz = SparseVar2(tmp_path / "bgz.svar2")
    assert sv_gz.available_samples == sv_bgz.available_samples
    assert (
        sv_gz.region_counts("chr1", [(0, len(_REF))]).tolist()
        == sv_bgz.region_counts("chr1", [(0, len(_REF))]).tolist()
    )


def test_from_vcf_list_directory_of_vcf_bgz(tmp_path: Path):
    """The vcf-list form over a directory of `.vcf.bgz` files."""
    d = tmp_path / "vcfs"
    d.mkdir()
    _bgzipped_vcf(
        d,
        "a",
        sample="SA",
        rows="chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n",
        suffix=".vcf.bgz",
    )
    _bgzipped_vcf(
        d,
        "b",
        sample="SB",
        rows="chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n",
        suffix=".vcf.bgz",
    )
    out = tmp_path / "store"
    assert SparseVar2.from_vcf_list(out, d, no_reference=True, threads=1) == 0
    assert SparseVar2(out).available_samples == ["SA", "SB"]


def test_from_vcf_bgz_auto_indexes(tmp_path: Path):
    """`_ensure_index` builds `<name>.csi` for an unindexed `.vcf.bgz` -- the
    index name is derived from the full file name, so the `.bgz` spelling is
    carried into it (`in.vcf.bgz.csi`, which is what htslib then looks for)."""
    gz = _bgzipped_vcf(tmp_path)
    bgz = _same_bytes_as_bgz(gz)
    bgz.with_name(bgz.name + ".csi").unlink()

    assert SparseVar2.from_vcf(tmp_path / "out.svar2", bgz, no_reference=True) == 0
    assert bgz.with_name(bgz.name + ".csi").exists()


def test_from_vcf_bgz_honors_existing_tbi(tmp_path: Path):
    """An existing `.vcf.bgz.tbi` (how All of Us ships its callsets) satisfies
    the index requirement -- no redundant `.csi` is built alongside it."""
    plain = tmp_path / "in.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n"
    )
    bgz = tmp_path / "in.vcf.bgz"
    with open(bgz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", "-t", str(bgz)], check=True)
    assert bgz.with_name(bgz.name + ".tbi").exists()

    assert SparseVar2.from_vcf(tmp_path / "out.svar2", bgz, no_reference=True) == 0
    assert not bgz.with_name(bgz.name + ".csi").exists()


def test_vcf_class_reads_vcf_bgz(tmp_path: Path):
    """`VCF`'s oxbow-backed record reader dispatches on the file name too."""
    gz = _bgzipped_vcf(tmp_path)
    bgz = _same_bytes_as_bgz(gz)
    assert VCF(bgz).get_record_info().equals(VCF(gz).get_record_info())

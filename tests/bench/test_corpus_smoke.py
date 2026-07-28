import re
import subprocess

import pytest

from scripts.bench_svar2 import scale_corpus
from scripts.bench_svar2.scale_corpus import generate, size_corpus

pytestmark = pytest.mark.bench


def test_size_corpus_respects_production_chunk_clamp():
    """_auto_chunk_size never exceeds 25_000; measuring above it would
    characterize a regime production cannot reach."""
    variants, chunk_size = size_corpus(samples=250, cells_budget=1_400_000_000)
    assert variants == 5_600_000
    assert chunk_size == 25_000


def test_size_corpus_gives_at_least_32_chunks_at_large_S():
    variants, chunk_size = size_corpus(samples=500_000, cells_budget=1_400_000_000)
    assert variants == 2_800
    assert chunk_size == 87
    assert variants // chunk_size >= 32


def test_size_corpus_floors_chunk_size_at_64():
    _, chunk_size = size_corpus(samples=1_000_000, cells_budget=1_000_000)
    assert chunk_size == 64


def test_generate_is_deterministic(tmp_path):
    a = generate(
        tmp_path / "a.vcf.gz",
        samples=8,
        variants=200,
        contigs=["chr22"],
        format_fields=(),
        seed=42,
        procs=2,
        bgzip_threads=1,
    )
    b = generate(
        tmp_path / "b.vcf.gz",
        samples=8,
        variants=200,
        contigs=["chr22"],
        format_fields=(),
        seed=42,
        procs=2,
        bgzip_threads=1,
    )
    assert (tmp_path / "a.vcf.gz").read_bytes() == (tmp_path / "b.vcf.gz").read_bytes()
    assert a.cells == b.cells == 8 * 200


def test_generate_record_count_matches_manifest(tmp_path):
    """A truncated corpus must not silently yield fast, bogus timings."""
    m = generate(
        tmp_path / "c.vcf.gz",
        samples=4,
        variants=500,
        contigs=["chr21", "chr22"],
        format_fields=(),
        seed=1,
        procs=2,
        bgzip_threads=1,
    )
    n = int(
        subprocess.run(
            ["bcftools", "index", "-n", str(tmp_path / "c.vcf.gz")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    assert n == m.variants


def test_generate_with_format_fields(tmp_path):
    m = generate(
        tmp_path / "d.vcf.gz",
        samples=4,
        variants=100,
        contigs=["chr22"],
        format_fields=("DP", "GQ", "AD"),
        seed=3,
        procs=1,
        bgzip_threads=1,
    )
    assert m.format_fields == ("DP", "GQ", "AD")
    hdr = subprocess.run(
        ["bcftools", "view", "-h", str(tmp_path / "d.vcf.gz")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for f in ("DP", "GQ", "AD"):
        assert f"##FORMAT=<ID={f}," in hdr


def test_generate_writes_manifest(tmp_path):
    generate(
        tmp_path / "e.vcf.gz",
        samples=4,
        variants=100,
        contigs=["chr22"],
        format_fields=(),
        seed=5,
        procs=1,
        bgzip_threads=1,
    )
    assert (tmp_path / "e.manifest.json").exists()


def test_positions_are_sorted_and_unique(tmp_path):
    generate(
        tmp_path / "f.vcf.gz",
        samples=2,
        variants=1000,
        contigs=["chr22"],
        format_fields=(),
        seed=9,
        procs=2,
        bgzip_threads=1,
    )
    out = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", str(tmp_path / "f.vcf.gz")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    pos = [int(x) for x in out]
    assert pos == sorted(pos)
    assert len(set(pos)) == len(pos)


def test_positions_are_sorted_and_unique_across_short_final_block(tmp_path):
    """2_800 variants on one contig with BLOCK_VARIANTS=2_000 gives blocks of
    2000 + 800 -- a short final block. The brief's original striping derives
    `stride` from `BLOCK_VARIANTS` and the stripe origin from the per-block `n`
    (not the per-contig total), so every block after the first lands at the
    wrong origin and/or overruns the contig: positions come out unsorted and
    exceed the declared ##contig length, and tabix indexing fails."""
    from scripts.bench_svar2.scale_corpus import DEFAULT_CONTIG_LEN

    generate(
        tmp_path / "g.vcf.gz",
        samples=2,
        variants=2_800,
        contigs=["chr22"],
        format_fields=(),
        seed=11,
        procs=2,
        bgzip_threads=1,
    )
    out = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", str(tmp_path / "g.vcf.gz")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    pos = [int(x) for x in out]
    assert len(pos) == 2_800
    assert pos == sorted(pos)
    assert len(set(pos)) == len(pos)
    assert pos[-1] <= DEFAULT_CONTIG_LEN


def test_pos_stays_within_declared_length_when_stride_floors_to_one(
    tmp_path, monkeypatch
):
    """When per_contig exceeds DEFAULT_CONTIG_LEN, `stride` floors to 1 and
    positions run essentially consecutively -- past the declared ##contig
    length -- unless the header reports the true span. Patch
    DEFAULT_CONTIG_LEN down so this regime is reachable without generating
    tens of millions of records; the invariant under test does not depend on
    the constant's actual value. Reads the declared length back out of the
    header rather than hardcoding a number, so this checks the real
    contig-truthfulness invariant, not a hand-computed expectation."""
    monkeypatch.setattr(scale_corpus, "DEFAULT_CONTIG_LEN", 2_000)
    generate(
        tmp_path / "h.vcf.gz",
        samples=2,
        variants=2_800,
        contigs=["chr22"],
        format_fields=(),
        seed=13,
        procs=2,
        bgzip_threads=1,
    )

    hdr = subprocess.run(
        ["bcftools", "view", "-h", str(tmp_path / "h.vcf.gz")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    m = re.search(r"##contig=<ID=chr22,length=(\d+)>", hdr)
    assert m is not None
    declared_len = int(m.group(1))

    out = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", str(tmp_path / "h.vcf.gz")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    pos = [int(x) for x in out]
    assert len(pos) == 2_800
    assert max(pos) <= declared_len

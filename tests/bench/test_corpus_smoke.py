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


def test_declared_contig_length_is_not_inflated_in_a_normal_regime(tmp_path):
    """In the normal regime (per_contig * stride <= DEFAULT_CONTIG_LEN, e.g.
    this repo's own variants=1000 single-contig case, same as
    test_positions_are_sorted_and_unique), the declared ##contig length must
    be *exactly* DEFAULT_CONTIG_LEN -- not merely >= it. A prior version of
    the contig_len formula multiplied by nominal block *capacity*
    (n_blocks * BLOCK_VARIANTS) rather than the actual per-contig record
    count, so any per_contig not an exact multiple of BLOCK_VARIANTS
    (variants=1000 with BLOCK_VARIANTS=2_000 gives a single short block)
    silently inflated the declared length to roughly 2x DEFAULT_CONTIG_LEN.
    Nothing crashed and no other test caught it, because the error was in
    the conservative direction and no prior test asserted the header value
    directly."""
    from scripts.bench_svar2.scale_corpus import DEFAULT_CONTIG_LEN

    generate(
        tmp_path / "i.vcf.gz",
        samples=2,
        variants=1000,
        contigs=["chr22"],
        format_fields=(),
        seed=9,
        procs=2,
        bgzip_threads=1,
    )
    hdr = subprocess.run(
        ["bcftools", "view", "-h", str(tmp_path / "i.vcf.gz")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    m = re.search(r"##contig=<ID=chr22,length=(\d+)>", hdr)
    assert m is not None
    declared_len = int(m.group(1))
    assert declared_len == DEFAULT_CONTIG_LEN


# --- plan_blocks: memory bounding must not cost determinism -----------------


def test_plan_blocks_leaves_gt_only_corpora_at_the_constant():
    """Block size sets the position striping and the per-block seed, so it is
    part of a corpus's identity. Eleven GT-only corpora are already generated
    for this sweep and the regression tier's baselines are recorded against
    one of them, so the GT-only path must keep cutting at BLOCK_VARIANTS.
    Changing it is a GENERATOR_VERSION bump, not a bug fix.

    Covers the full sample axis, including the two shapes whose blocks are
    largest in memory (S=250,000 and S=500,000).
    """
    for samples in (200, 250, 1_000, 4_000, 16_000, 64_000, 250_000, 500_000):
        variants, _ = size_corpus(samples, 1_400_000_000)
        block, procs = scale_corpus.plan_blocks(variants, 1, samples, 0, 16)
        assert block == scale_corpus.BLOCK_VARIANTS, f"S={samples} block moved"
        assert procs >= 1


def test_plan_blocks_shrinks_the_format_path_that_hung_the_sweep():
    """The hold-out (100,000 samples, 3 FORMAT fields) cut 14 blocks of 2e8
    cells and ran 14 of them concurrently, OOM-killing pool workers under the
    cgroup. Peak must now fit POOL_MEMORY_BUDGET."""
    block, procs = scale_corpus.plan_blocks(28_000, 1, 100_000, 3, 16)
    assert block < scale_corpus.BLOCK_VARIANTS
    peak = procs * block * 100_000 * scale_corpus.FMT_PEAK_BYTES_PER_CELL
    assert peak <= scale_corpus.POOL_MEMORY_BUDGET


@pytest.mark.parametrize("n_format", [0, 3])
def test_plan_blocks_block_size_is_independent_of_procs(n_format):
    """REGRESSION: block size must not depend on `procs`.

    The first cut of the memory fix derived the per-block budget as a
    `procs` share of the pool budget. That made `--procs` change the block
    partitioning, hence the position striping and per-block seeds, hence the
    output bytes -- so the same corpus request produced different files at
    different worker counts. `_format_block` seeds per block precisely so
    that cannot happen; the whole point of the pool is that it is invisible
    in the output.
    """
    sizes = {
        scale_corpus.plan_blocks(28_000, 1, 100_000, n_format, p)[0]
        for p in (1, 2, 4, 8, 16, 48)
    }
    assert len(sizes) == 1, f"block size varies with procs: {sorted(sizes)}"


def test_plan_blocks_raises_rather_than_hanging_when_one_block_cannot_fit():
    """A block too large for the budget used to be discovered as OOM-killed
    workers plus an `mp.Pool` that waited forever. Fail loudly with the
    arithmetic instead."""
    with pytest.raises(RuntimeError, match="pool budget"):
        scale_corpus.plan_blocks(10, 1, 10_000_000_000, 0, 1)


def test_sub_block_positions_stay_inside_the_declared_contig(tmp_path, monkeypatch):
    """REGRESSION: `_block_positions` striped at the BLOCK_VARIANTS constant
    rather than the actual block size.

    With smaller blocks each stripe was mostly empty, so the last block began
    at `n_blocks * BLOCK_VARIANTS * stride` -- about 20x past the declared
    contig length for the hold-out. Positions still came out sorted, so
    nothing looked wrong until `tabix` rejected the finished file, after the
    entire corpus had been written (21 minutes and 12 GB, in the real case).
    """
    monkeypatch.setattr(scale_corpus, "FMT_BLOCK_MEMORY_BUDGET", 20_000_000)
    samples, variants = 500, 2_000
    block, procs = scale_corpus.plan_blocks(variants, 1, samples, 3, 16)
    assert block < scale_corpus.BLOCK_VARIANTS, "vacuous unless blocks shrink"

    out = tmp_path / "fmt.vcf.gz"
    m = generate(
        out,
        samples,
        variants,
        ["chr22"],
        ["DP", "GQ", "AD"],
        seed=1,
        procs=procs,
        bgzip_threads=2,
    )
    assert m.variants == variants  # generate's own index check already ran

    text = subprocess.run(
        ["bcftools", "view", "-H", str(out)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pos = [int(line.split("\t")[1]) for line in text.strip().split("\n")]
    assert len(pos) == variants
    assert max(pos) <= scale_corpus.DEFAULT_CONTIG_LEN
    assert pos == sorted(pos) and len(set(pos)) == len(pos)

    # And the pool stays invisible in the output: same request, different
    # worker count, identical bytes. This is the end-to-end form of the
    # block-size-independence check above -- it is what caught the `procs`
    # coupling in the first place.
    serial = tmp_path / "fmt_serial.vcf.gz"
    generate(
        serial,
        samples,
        variants,
        ["chr22"],
        ["DP", "GQ", "AD"],
        seed=1,
        procs=1,
        bgzip_threads=2,
    )
    assert out.read_bytes() == serial.read_bytes()

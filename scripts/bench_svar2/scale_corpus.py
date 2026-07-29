"""Deterministic synthetic VCF generation for the SVAR2 scale bench.

Generation cost is linear in genotype cells (~1.2M cells/s/process measured),
which is the binding constraint on how far up the sample axis the sweep can
reach. Record blocks are formatted in a process pool and streamed IN ORDER into
a single `bgzip` stdin: parallel formatting, one compression pass, one valid
BGZF stream, no temp files and bounded memory.

Run as a module so pool workers resolve `_format_block` by name under the
forkserver start method (Python 3.14's Linux default):

    python -m scripts.bench_svar2.scale_corpus --out corpus.vcf.gz --samples 1000 ...
"""

from __future__ import annotations

import argparse
import subprocess
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from scripts.bench_svar2.records import CorpusManifest, to_json

GENERATOR_VERSION = 1
PLOIDY = 2
BASES = np.array(["A", "C", "G", "T"])
GT_TOKENS = np.array(["0|0", "0|1", "1|0", "1|1"])
GT_WEIGHTS = np.array([0.72, 0.12, 0.12, 0.04])
BLOCK_VARIANTS = 2_000
# Peak transient bytes per genotype cell inside `_format_block`, against the
# widest intermediates it holds live at once.
#
# GT-only holds one `<U3` token array: 3 chars * 4 bytes (numpy strings are
# UCS4) = 12 B/cell.
#
# The FORMAT path is an order of magnitude worse: it holds the `<U3` tokens,
# three int64 draws, four `<U3` stringified copies, and an `np.char.add` chain
# whose result widens 3 -> 6 -> 9 -> 12 -> 15 chars, with two generations live
# across each step (48 + 60 B/cell at the top). 200 B/cell is that sum rounded
# up. It assumes the `.astype("U3")` in `_format_block`: plain `.astype(str)`
# on an int64 array yields dtype `<U21` (room for the widest int64), i.e. 84
# B/cell EACH for dp/gq/ad/ad*2, which is what made a single hold-out block
# need ~70 GB and hang the sweep.
GT_PEAK_BYTES_PER_CELL = 12
FMT_PEAK_BYTES_PER_CELL = 200
# Total formatting memory the pool may use across ALL workers at once. The
# sweep runs under a 120 GB cgroup and must leave room for bgzip, the parent,
# and page cache. Budgeted at ~1/4 of the cgroup rather than 1/2 because the
# per-cell figures above are ESTIMATES of transient numpy intermediates, not
# measurements: at 32 GB the estimate can be 2x low and the job still fits,
# and the failure mode being guarded against (OOM-killed workers) previously
# manifested as a silent 3-hour hang rather than an error.
POOL_MEMORY_BUDGET = 32_000_000_000
# Memory one FORMAT-path block may use. Deliberately a PER-BLOCK constant and
# not a share of `POOL_MEMORY_BUDGET`: block size determines the position
# striping and the per-block seed, so anything it depends on becomes part of
# the corpus's identity. Deriving it from `procs` (a `--procs` share of the
# pool budget) made the SAME corpus request emit different bytes at different
# worker counts, silently breaking the determinism `_format_block`'s per-block
# seeding exists to guarantee.
FMT_BLOCK_MEMORY_BUDGET = 2_000_000_000
DEFAULT_CONTIG_LEN = 50_818_468  # GRCh38 chr22
# Production reference: `_auto_chunk_size` clamps at 25_000 and `from_vcf`
# hardcodes exactly that.
MAX_CHUNK_SIZE = 25_000
MIN_CHUNK_SIZE = 64
MIN_CHUNKS = 32


def plan_blocks(
    per_contig: int, n_contigs: int, n_samples: int, n_format: int, procs: int
) -> tuple[int, int]:
    """Choose `(block_variants, procs)` whose peak pool memory fits the budget.

    `BLOCK_VARIANTS` bounds a block by VARIANTS alone, so a block holds
    `block * n_samples` cells and its memory grows with cohort size. That is
    harmless at the sweep's small-S shapes and catastrophic at its large ones:
    the hold-out (100,000 samples, 3 FORMAT fields) splits into 14 blocks of
    2,000 variants, i.e. 2e8 cells each, and 14 of them format concurrently
    under `--procs 16`. Workers were OOM-killed by the cgroup, `mp.Pool`
    silently repopulated them, and `imap` waited forever on results that would
    never arrive -- a 3-hour silent hang with a 0-byte output file.

    Two knobs, deliberately asymmetric:

    - `block` is reduced ONLY on the FORMAT path, and ONLY as a function of
      the corpus shape (`n_samples`, `n_format`) via the per-block constant
      `FMT_BLOCK_MEMORY_BUDGET`. Block size changes output bytes -- it sets
      both the position striping and the per-block seed -- so it must not
      depend on `procs`, or the same corpus request would emit different
      bytes at different worker counts. It also stays fixed on the GT-only
      path: the 11 GT-only corpora already generated for this sweep, plus the
      committed regression baselines, must stay byte-reproducible, so
      changing GT-only block sizing is a `GENERATOR_VERSION` bump rather than
      a bug fix.
    - `procs` is capped so at most `POOL_MEMORY_BUDGET` worth of blocks are
      ever live. This is the knob that is safe to move freely, because
      concurrency alone does NOT affect output bytes: `.map` preserves order
      and `_format_block` seeds itself per block.

    The GT-only path is therefore bounded only by the `procs` cap. That is
    sufficient for every shape this harness generates (the widest, S=500,000
    at 2,800 variants, is 2 blocks of 12 GB), and if it ever stops being
    sufficient this raises with the arithmetic rather than hanging.
    """
    per_cell = FMT_PEAK_BYTES_PER_CELL if n_format else GT_PEAK_BYTES_PER_CELL
    block = BLOCK_VARIANTS
    if n_format:
        block = max(
            1,
            min(
                block,
                FMT_BLOCK_MEMORY_BUDGET // max(1, per_cell * max(n_samples, 1)),
            ),
        )

    per_block = block * max(n_samples, 1) * per_cell
    if per_block > POOL_MEMORY_BUDGET:
        raise RuntimeError(
            f"one {block}-variant block at {n_samples} samples needs "
            f"~{per_block / 1e9:.0f} GB, over the {POOL_MEMORY_BUDGET / 1e9:.0f} GB "
            "pool budget. Lower BLOCK_VARIANTS (a GENERATOR_VERSION bump: it "
            "changes output bytes) or raise POOL_MEMORY_BUDGET to match the "
            "job's cgroup."
        )
    n_tasks = n_contigs * -(-per_contig // block)
    procs = max(1, min(procs, POOL_MEMORY_BUDGET // per_block, max(n_tasks, 1)))
    return block, procs


def size_corpus(samples: int, cells_budget: int) -> tuple[int, int]:
    """Variants and chunk size for one scale point.

    Two constraints bind against each other: generation cost is linear in
    cells, and steady state needs enough chunks that fill/drain is a small
    fraction of the run. Fixing the cell budget and flooring the chunk count at
    32 resolves both. The upper clamp keeps small cohorts inside the regime
    production can actually reach.
    """
    variants = max(1, cells_budget // max(samples, 1))
    chunk_size = min(MAX_CHUNK_SIZE, max(MIN_CHUNK_SIZE, variants // MIN_CHUNKS))
    return variants, chunk_size


def _contig_key(contig: str) -> int:
    """Stable contig seed component. `hash()` is PYTHONHASHSEED-salted, so it
    would make corpora irreproducible across processes."""
    return zlib.crc32(contig.encode())


def _format_block(
    task: tuple[str, int, int],
    *,
    n_samples: int,
    n_format: int,
    seed: int,
    stride: int,
    block_variants: int,
) -> bytes:
    """Format one block of records. Module-level and keyword-bound via
    `partial` so forkserver workers can import it by name."""
    contig, block_index, n = task
    # Derive a per-block seed so output is independent of pool scheduling
    # order -- otherwise `procs` would change the bytes and break determinism.
    rng = np.random.default_rng([seed, _contig_key(contig), block_index])
    pos = _block_positions(contig, block_index, n, seed, stride, block_variants)
    ref = rng.choice(BASES, size=n)
    alt_offset = rng.integers(1, 4, size=n)
    alt = BASES[(np.searchsorted(BASES, ref) + alt_offset) % 4]

    gts = rng.choice(GT_TOKENS, size=(n, n_samples), p=GT_WEIGHTS)
    if n_format:
        dp = rng.integers(1, 100, size=(n, n_samples))
        gq = rng.integers(1, 100, size=(n, n_samples))
        ad = rng.integers(0, 50, size=(n, n_samples))
        # `.astype("U3")`, NOT `.astype(str)`. On an int64 array numpy sizes
        # the result for the widest possible int64, dtype `<U21` = 84 bytes
        # per element -- four of those over a 2e8-cell block is ~67 GB, which
        # is what OOM-killed the pool workers and hung the hold-out corpus.
        # Every value here is bounded well under 1000 (dp/gq < 100, ad < 50,
        # ad*2 < 100), so 3 characters is lossless and the emitted text is
        # byte-identical.
        cells = np.char.add(np.char.add(gts, ":"), dp.astype("U3"))
        cells = np.char.add(np.char.add(cells, ":"), gq.astype("U3"))
        cells = np.char.add(np.char.add(cells, ":"), ad.astype("U3"))
        cells = np.char.add(np.char.add(cells, ","), (ad * 2).astype("U3"))
        fmt_key = "GT:DP:GQ:AD"
    else:
        cells = gts
        fmt_key = "GT"

    lines = []
    for i in range(n):
        lines.append(
            f"{contig}\t{pos[i]}\t.\t{ref[i]}\t{alt[i]}\t.\tPASS\t.\t{fmt_key}\t"
            + "\t".join(cells[i])
        )
    return ("\n".join(lines) + "\n").encode()


def _block_positions(
    contig: str,
    block_index: int,
    n: int,
    seed: int,
    stride: int,
    block_variants: int,
) -> np.ndarray:
    """Strictly increasing positions inside this block's disjoint stripe.

    Striping keeps blocks globally sorted without a cross-block sort, which is
    what lets blocks be formatted independently and still concatenate into a
    valid tabix-indexable VCF. Gaps are drawn in [1, stride] so a block of n
    records always fits inside its own `block_variants`-wide stripe.

    The stripe width is the ACTUAL block size, not the `BLOCK_VARIANTS`
    constant. `plan_blocks` may hand out smaller blocks, and a stripe wider
    than the block it holds leaves the stripes sparse: with 280 blocks of 100
    against a 2,000-wide stripe the last block starts at
    `279 * 2000 * stride`, roughly 20x past the declared contig length, and
    `tabix` rejects the file after the whole corpus has been written. Sizing
    the stripe to the block keeps the upper bound at `per_contig * stride`,
    which is exactly what `generate` declares as `contig_len`. When
    `plan_blocks` returns `BLOCK_VARIANTS` -- every GT-only corpus -- these
    positions are identical to before.
    """
    lo = block_index * block_variants * stride + 1
    rng = np.random.default_rng([seed, _contig_key(contig), block_index, 99])
    return lo + np.cumsum(rng.integers(1, stride + 1, size=n))


def generate(
    out: Path,
    samples: int,
    variants: int,
    contigs: Sequence[str],
    format_fields: Sequence[str],
    seed: int,
    procs: int = 8,
    bgzip_threads: int = 4,
) -> CorpusManifest:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    contigs = list(contigs)
    n_format = len(format_fields)

    per_contig = variants // len(contigs)
    total = per_contig * len(contigs)
    # Run-level stride derived from the per-contig total (not BLOCK_VARIANTS)
    # so positions stay within the declared contig length regardless of how
    # many blocks a contig is split into.
    stride = max(1, DEFAULT_CONTIG_LEN // max(per_contig, 1))
    # When stride floors to 1 (per_contig > DEFAULT_CONTIG_LEN), positions run
    # past DEFAULT_CONTIG_LEN. Declare a truthful length instead of rejecting
    # the input -- a small-cohort/high-variant corpus is a legitimate ask.
    # Block b starts at b * block_variants * stride + 1 and adds at most
    # n_b * stride, where n_b is that block's record count; block sizes sum
    # to per_contig by construction (tasks below split per_contig into
    # blocks of at most block_variants). So the true upper bound on POS
    # across all blocks is per_contig * stride, independent of how per_contig
    # happens to be chunked into blocks. This collapses to exactly
    # DEFAULT_CONTIG_LEN in every regime that already worked.
    #
    # This bound requires the stripe width to BE `block_variants` -- the size
    # blocks are actually cut at -- not the `BLOCK_VARIANTS` constant. Striping
    # at the constant while cutting smaller blocks leaves every stripe mostly
    # empty and pushes the last block's start to
    # `n_blocks * BLOCK_VARIANTS * stride`, far past `contig_len`; tabix then
    # rejects the corpus only after all of it has been written. See
    # `_block_positions`.
    contig_len = max(DEFAULT_CONTIG_LEN, per_contig * stride + 1)

    header = ["##fileformat=VCFv4.2"]
    for c in contigs:
        header.append(f"##contig=<ID={c},length={contig_len}>")
    header.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    if n_format:
        header.append('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">')
        header.append(
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">'
        )
        header.append(
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic Depths">'
        )
    sample_names = [f"S{i:06d}" for i in range(samples)]
    header.append(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(sample_names)
    )

    block_variants, procs = plan_blocks(
        per_contig, len(contigs), samples, n_format, procs
    )

    tasks: list[tuple[str, int, int]] = []
    for c in contigs:
        remaining = per_contig
        bi = 0
        while remaining > 0:
            n = min(block_variants, remaining)
            tasks.append((c, bi, n))
            remaining -= n
            bi += 1

    with out.open("wb") as sink:
        bg = subprocess.Popen(
            ["bgzip", "-c", "-@", str(bgzip_threads)],
            stdin=subprocess.PIPE,
            stdout=sink,
        )
        assert bg.stdin is not None
        bg.stdin.write(("\n".join(header) + "\n").encode())
        worker = partial(
            _format_block,
            n_samples=samples,
            n_format=n_format,
            seed=seed,
            stride=stride,
            block_variants=block_variants,
        )
        if procs > 1:
            # ProcessPoolExecutor, NOT mp.Pool. When a worker dies -- and the
            # cgroup OOM killer is the realistic way that happens here --
            # `mp.Pool` silently starts a replacement and `imap` blocks
            # forever on a result no one will ever produce. That is a silent
            # infinite hang with a 0-byte output file, which cost this sweep
            # three hours. `ProcessPoolExecutor` raises `BrokenProcessPool`
            # instead, so the sbatch's `set -e` kills the job in seconds with
            # a diagnosis. `.map` is ordered like `imap`, and `_format_block`
            # is module-level so it resolves by name under forkserver.
            with ProcessPoolExecutor(max_workers=procs) as pool:
                for blob in pool.map(worker, tasks, chunksize=1):
                    bg.stdin.write(blob)
        else:
            for t in tasks:
                bg.stdin.write(worker(t))
        bg.stdin.close()
        if bg.wait() != 0:
            raise RuntimeError("bgzip failed")

    subprocess.run(["tabix", "-f", "-p", "vcf", str(out)], check=True)

    indexed = int(
        subprocess.run(
            ["bcftools", "index", "-n", str(out)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    if indexed != total:
        raise RuntimeError(
            f"corpus truncated: index reports {indexed} records, expected {total}"
        )

    manifest = CorpusManifest(
        path=str(out),
        samples=samples,
        variants=total,
        contigs=tuple(contigs),
        format_fields=tuple(format_fields),
        ploidy=PLOIDY,
        cells=samples * total,
        compressed_bytes=out.stat().st_size,
        seed=seed,
        generator_version=GENERATOR_VERSION,
    )
    out.with_suffix("").with_suffix(".manifest.json").write_text(to_json(manifest))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--variants", type=int)
    p.add_argument("--cells-budget", type=int, default=1_400_000_000)
    p.add_argument("--contigs", type=str, default="chr22")
    p.add_argument("--format-fields", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--bgzip-threads", type=int, default=4)
    a = p.parse_args()

    variants = a.variants
    if variants is None:
        variants, _ = size_corpus(a.samples, a.cells_budget)
    fields = tuple(f for f in a.format_fields.split(",") if f)
    m = generate(
        a.out,
        a.samples,
        variants,
        a.contigs.split(","),
        fields,
        a.seed,
        a.procs,
        a.bgzip_threads,
    )
    print(
        f"wrote {m.path}: {m.variants} variants x {m.samples} samples "
        f"= {m.cells} cells, {m.compressed_bytes / 1e6:.0f} MB"
    )


if __name__ == "__main__":
    main()

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
import multiprocessing as mp
import subprocess
import zlib
from collections.abc import Sequence
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
DEFAULT_CONTIG_LEN = 50_818_468  # GRCh38 chr22
# Production reference: `_auto_chunk_size` clamps at 25_000 and `from_vcf`
# hardcodes exactly that.
MAX_CHUNK_SIZE = 25_000
MIN_CHUNK_SIZE = 64
MIN_CHUNKS = 32


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
) -> bytes:
    """Format one block of records. Module-level and keyword-bound via
    `partial` so forkserver workers can import it by name."""
    contig, block_index, n = task
    # Derive a per-block seed so output is independent of pool scheduling
    # order -- otherwise `procs` would change the bytes and break determinism.
    rng = np.random.default_rng([seed, _contig_key(contig), block_index])
    pos = _block_positions(contig, block_index, n, seed, stride)
    ref = rng.choice(BASES, size=n)
    alt_offset = rng.integers(1, 4, size=n)
    alt = BASES[(np.searchsorted(BASES, ref) + alt_offset) % 4]

    gts = rng.choice(GT_TOKENS, size=(n, n_samples), p=GT_WEIGHTS)
    if n_format:
        dp = rng.integers(1, 100, size=(n, n_samples))
        gq = rng.integers(1, 100, size=(n, n_samples))
        ad = rng.integers(0, 50, size=(n, n_samples))
        cells = np.char.add(np.char.add(gts, ":"), dp.astype(str))
        cells = np.char.add(np.char.add(cells, ":"), gq.astype(str))
        cells = np.char.add(np.char.add(cells, ":"), ad.astype(str))
        cells = np.char.add(np.char.add(cells, ","), (ad * 2).astype(str))
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
    contig: str, block_index: int, n: int, seed: int, stride: int
) -> np.ndarray:
    """Strictly increasing positions inside this block's disjoint stripe.

    Striping keeps blocks globally sorted without a cross-block sort, which is
    what lets blocks be formatted independently and still concatenate into a
    valid tabix-indexable VCF. Gaps are drawn in [1, stride] so a block of n
    records always fits inside its own BLOCK_VARIANTS-wide stripe.
    """
    lo = block_index * BLOCK_VARIANTS * stride + 1
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
    # Block b starts at b * BLOCK_VARIANTS * stride + 1 and adds at most
    # n_b * stride, where n_b is that block's record count; block sizes sum
    # to per_contig by construction (tasks below split per_contig into
    # blocks of at most BLOCK_VARIANTS). So the true upper bound on POS
    # across all blocks is per_contig * stride, independent of how per_contig
    # happens to be chunked into blocks. This collapses to exactly
    # DEFAULT_CONTIG_LEN in every regime that already worked.
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

    tasks: list[tuple[str, int, int]] = []
    for c in contigs:
        remaining = per_contig
        bi = 0
        while remaining > 0:
            n = min(BLOCK_VARIANTS, remaining)
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
        )
        if procs > 1:
            with mp.Pool(procs) as pool:
                # imap (not imap_unordered) -- VCF records must stay sorted.
                for blob in pool.imap(worker, tasks, chunksize=1):
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

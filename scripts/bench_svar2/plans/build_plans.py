"""Generate the sweep plans from the spec's scale points.

Generated rather than committed so a change to `size_corpus`'s rule cannot
silently disagree with a stale JSON file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from scripts.bench_svar2.records import SweepPoint
from scripts.bench_svar2.scale_corpus import (
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    MIN_CHUNKS,
    size_corpus,
)

CELLS_BUDGET = 1_400_000_000
SCALE_SAMPLES = (250, 1_000, 4_000, 16_000, 64_000, 250_000, 500_000)
# Validate the predicted knee against a real sweep at only three points.
KNEE_VALIDATION_SAMPLES = (250, 16_000, 500_000)
KNEE_WORKERS = (1, 2, 3, 5, 7, 11)
CONTIG_COUNTS = (1, 2, 8, 22)
# Single source of truth for the hold-out corpus's shape: `sweep_scale.sbatch`
# reads this back (via `python -c 'from ... import HOLDOUT; ...'`) instead of
# repeating the numbers, so editing one cannot silently drift from the other.
HOLDOUT = {"samples": 100_000, "variants": 28_000, "format_fields": ("DP", "GQ", "AD")}
# `from_vcf` hardcodes exactly this -- the only `from_*` method that skips
# `_auto_chunk_size`. It happens to equal `size_corpus`'s own upper clamp (see
# MAX_CHUNK_SIZE's docstring in scale_corpus.py), so reuse that constant
# rather than duplicate the literal.
PROD_CHUNK_SIZE = MAX_CHUNK_SIZE
# V-linearity ladder (design spec "Variants factor out"): fixed small S, V is
# the ONLY axis varying. S=250 rather than 1_000: it is already the smallest
# point characterized elsewhere in the harness, and it is the cheapest to
# generate across the ladder -- generation cost is linear in cells = S*V, so
# even at the top of the V range that's 250 * 200_000 = 5e7 cells versus
# 1_000 * 200_000 = 2e8 cells at S=1_000.
VLINEAR_SAMPLES = 250
VLINEAR_VARIANTS = (25_000, 50_000, 100_000, 200_000)
# RLIMIT_AS installed on the points whose PURPOSE is to find out whether
# `from_vcf`'s hardcoded chunk_size survives biobank scale. Deliberately NOT on
# every point (a deviation from the design spec's "each point runs under an
# explicit RSS ceiling"): `probe.py:_build_env` sets `MALLOC_ARENA_MAX=1`
# whenever a ceiling exists, and pinning glibc to one arena while running
# `-@ 48` with up to 11 reader workers is not the production allocator regime.
# Earlier work in this repo measured `MALLOC_ARENA_MAX=1` at 73% slower in a
# multithreaded conversion (the later "safe" finding was for the SINGLE-THREADED
# `from_vcf_list` path, not this one), so applying it to the V-law ladder, both
# cost laws -- hence the H2 verdict -- and every wall time in the sweep would
# measure the allocator, not the code. The ceiling only has a job to do where an
# OOM is physically possible, which is the production-chunk_size points at large
# S; the law-fitting points run on the production allocator instead. The OOM
# deliverable is unaffected: it is exactly those points that can produce
# `oom_at_rss_mb`.
OOM_PROBE_CEILING_MB = 60_000


def _chunk_size_for(variants: int) -> int:
    """`size_corpus`'s own floor-of->=32-chunks / clamp-at-25_000 rule,
    applied directly to a known variant count instead of deriving one from a
    cell budget. Used wherever a plan point's variant count is fixed by hand,
    so its chunk size still obeys the same invariant rather than being an
    unrelated literal."""
    return min(MAX_CHUNK_SIZE, max(MIN_CHUNK_SIZE, variants // MIN_CHUNKS))


# Fixed across the whole V-ladder so V is the only thing that varies --
# re-deriving a chunk size per V (the way size_corpus does per S) would
# confound wall time with chunk size instead of isolating variant count.
# Sized off the SMALLEST V so every point in the ladder clears the
# >=32-chunks floor (larger V only adds more chunks at this same size).
VLINEAR_CHUNK_SIZE = _chunk_size_for(min(VLINEAR_VARIANTS))


def _point(
    corpus: Path,
    workers: int,
    chunk_size: int,
    threads: int,
    concurrent: int | None = None,
    rss_ceiling_mb: int | None = None,
) -> SweepPoint:
    """One plan point. `rss_ceiling_mb` defaults to None -- see
    `OOM_PROBE_CEILING_MB` for why a ceiling is opt-in per point rather than a
    sweep-wide default."""
    return SweepPoint(
        corpus=str(corpus),
        reader_workers=workers,
        concurrent_chroms=concurrent,
        shard_htslib=0,
        overshard=4,
        chunk_size=chunk_size,
        threads=threads,
        reps=3,
        rss_ceiling_mb=rss_ceiling_mb,
    )


def build(corpus_dir: Path, threads: int) -> dict[str, list[SweepPoint]]:
    scale, contig, holdout, vlinear = [], [], [], []

    for s in SCALE_SAMPLES:
        _, cs = size_corpus(s, CELLS_BUDGET)
        corpus = corpus_dir / f"s{s}.manifest.json"
        # One w=1 run per point predicts the knee; only three points get swept.
        scale.append(_point(corpus, 1, cs, threads))
        if s in KNEE_VALIDATION_SAMPLES:
            for w in KNEE_WORKERS:
                if w != 1:
                    scale.append(_point(corpus, w, cs, threads))
        # I5: every point above uses size_corpus's DERIVED chunk size, so
        # nothing ever measures from_vcf's actual hardcoded default
        # (PROD_CHUNK_SIZE). Reusing this same, already-generated corpus is
        # enough to test it -- chunk_assembler::read_next_chunk allocates the
        # packed grid at the FULL chunk_size up front and only truncates
        # after EOF, so even a corpus whose own V is far smaller than
        # PROD_CHUNK_SIZE still reserves the large allocation's ADDRESS SPACE.
        # (Address space, not RSS: `BitGrid3::zeros` is `vec![0u64; n_words]`
        # -> alloc_zeroed -> calloc, and pages the reader never writes never
        # become resident -- measured on this node, a 3 GB zeroed allocation
        # adds 0 MB to ru_maxrss. So these points probe the RLIMIT_AS ceiling,
        # and `model.py:_resident_chunk_size` bounds what they contribute to
        # the RSS law.) This is also what keeps these points cheap in wall
        # time: conversion cost tracks V, which is unchanged, not the (much
        # larger) hypothetical chunk size.
        # `OOM_PROBE_CEILING_MB` is an RLIMIT_AS the kernel enforces on the
        # child (see probe.py `_preexec`), comfortably inside
        # sweep_scale.sbatch's --mem=120G -- a genuine OOM kills only that one
        # child and is recorded as `oom_at_rss_mb`, never the node. These are
        # the ONLY points that carry it; see `OOM_PROBE_CEILING_MB`.
        # Skipped when cs already equals PROD_CHUNK_SIZE (S=250, 1_000:
        # size_corpus's own clamp already lands there), which would otherwise
        # duplicate the point above and waste a sweep slot. Those two points
        # are also the two where a 25_000-variant chunk cannot possibly
        # exhaust memory (1.5 MB of grid at S=250), so nothing is lost by
        # leaving them on the production allocator.
        if cs != PROD_CHUNK_SIZE:
            scale.append(
                _point(
                    corpus,
                    1,
                    PROD_CHUNK_SIZE,
                    threads,
                    rss_ceiling_mb=OOM_PROBE_CEILING_MB,
                )
            )

    # Contig axis at fixed cohort: hold TOTAL readers constant (12) and vary the
    # split, which is what separates "too few readers" from "wrong contig".
    # `sorted({1, min(c, 4)})` (not the tuple `(1, min(c, 4))`) so c == 1 yields
    # exactly one point instead of two identical ones: with a single contig,
    # concurrent_chroms > 1 is physically meaningless -- there is no second
    # contig to split onto -- so the "high-split" counterfactual this axis is
    # probing is inherently absent at c == 1, not something the loop can supply.
    _, cs = size_corpus(4_000, CELLS_BUDGET)
    for c in CONTIG_COUNTS:
        corpus = corpus_dir / f"s4000_c{c}.manifest.json"
        for concurrent in sorted({1, min(c, 4)}):
            workers = max(1, 12 // concurrent)
            contig.append(_point(corpus, workers, cs, threads, concurrent=concurrent))

    # Hold-out chunk size derived the same way the fitted points' is (from its
    # own variant count via the shared floor/clamp rule), not an unrelated
    # literal -- the hold-out exists to test the fitted laws OUT OF SAMPLE, so
    # it must sit on the same chunk-sizing regime they were fitted under.
    holdout.append(
        _point(
            corpus_dir / "holdout.manifest.json",
            1,
            _chunk_size_for(HOLDOUT["variants"]),
            threads,
        )
    )

    for v in VLINEAR_VARIANTS:
        corpus = corpus_dir / f"vlinear_v{v}.manifest.json"
        vlinear.append(_point(corpus, 1, VLINEAR_CHUNK_SIZE, threads))

    return {"scale": scale, "contig": contig, "holdout": holdout, "vlinear": vlinear}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=48)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, points in build(a.corpus_dir, a.threads).items():
        path = a.out_dir / f"{name}.json"
        path.write_text(json.dumps([dataclasses.asdict(pt) for pt in points], indent=2))
        print(f"{path}: {len(points)} points")


if __name__ == "__main__":
    main()

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
# Same S and V, no FORMAT fields. The gate exists to validate the S,V
# extrapolation, and it cannot do that from the F=3 corpus above: every
# law-fitting corpus is F=0 and no cost law carries an F term, so scoring
# against F=3 measures an axis nobody fitted (it read as a 63% phase-1 "model
# failure" that was really an unmodelled FORMAT-decode cost). This corpus is
# IN the fitted domain, so its error is attributable to S,V extrapolation and
# nothing else. Keep BOTH: the F=3 point still runs and is still reported, as
# the standing record of how far off the model is on data it does not cover.
HOLDOUT_F0 = {"samples": 100_000, "variants": 28_000, "format_fields": ()}
# `from_vcf` hardcodes exactly this -- the only `from_*` method that skips
# `_auto_chunk_size`. It happens to equal `size_corpus`'s own upper clamp (see
# MAX_CHUNK_SIZE's docstring in scale_corpus.py), so reuse that constant
# rather than duplicate the literal.
PROD_CHUNK_SIZE = MAX_CHUNK_SIZE
# V-linearity ladder (design spec "Variants factor out"): fixed small S, V is
# the ONLY axis varying. S=250 rather than 1_000: it is already the smallest
# point characterized elsewhere in the harness, and it is the cheapest to
# generate across the ladder -- generation cost is linear in cells = S*V, so
# even at the top of the V range that's 250 * 5_600_000 = 1.4e9 cells versus
# 1_000 * 5_600_000 = 5.6e9 cells at S=1_000.
#
# The V range is set in the ASYMPTOTIC regime, not the cheapest one. The first
# ladder (25_000 .. 200_000 at chunk_size=781) was measured and FALSIFIED by
# data already in the sweep, at the same S=250, so no cohort scaling was even
# involved: per-variant phase-1 cost fell monotonically across every rung
# (3.04e-4, 2.94e-4, 1.81e-4, 1.61e-4 s/variant) and reached 1.12e-5 at the
# scale sweep's own S=250 point (V=5_600_000) -- still falling, i.e. the ladder
# never left the fixed-cost-dominated regime. Fitting a LINE there and
# stretching it to V=1e9 predicted 740.5s at V=5_600_000 where 62.9s was
# measured: 11.8x high. Downstream that became a 2820% hold-out error and a
# 4.0e9-second projection.
#
# The mechanism was per-CHUNK cost, not per-variant work: at chunk_size=781 the
# V=200_000 rung is 257 chunks/32.1s, while the V=5_600_000 point at
# chunk_size=25_000 is 224 chunks/62.9s -- near-identical chunk counts, 28x the
# data, 2x the time. So the fitted "per-variant slope" was a per-chunk cost
# divided by 781, then applied at a 32x larger chunk size.
#
# Starting at 800_000 is forced: it is the smallest V clearing _chunk_size_for's
# >=MIN_CHUNKS floor at PROD_CHUNK_SIZE (32 * 25_000). The top rung also cuts
# the V-law's extrapolation stretch to 1e9 from ~5000x to ~179x, and matches the
# s250 corpus shape exactly, so it cross-checks against that independently
# measured point.
VLINEAR_SAMPLES = 250
VLINEAR_VARIANTS = (800_000, 1_400_000, 2_800_000, 5_600_000)
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
#
# Pinned to PROD_CHUNK_SIZE, the regime `extrapolate` actually targets, NOT
# sized off the smallest V. Deriving it from min(VLINEAR_VARIANTS) is what put
# the whole ladder at chunk_size=781, 32x below the target, and since phase-1
# cost at S=250 is dominated by per-chunk overhead rather than per-variant
# work, the fitted slope was a per-chunk artifact that over-predicted by 11.8x
# (see VLINEAR_VARIANTS). A V-law is only usable at the chunk size it was
# fitted under, so the ladder must be fitted under the one being predicted.
#
# `_chunk_size_for(min(VLINEAR_VARIANTS))` now returns this same value anyway
# (800_000 // 32 == 25_000 == MAX_CHUNK_SIZE), but stating PROD_CHUNK_SIZE
# directly makes the requirement explicit rather than an arithmetic
# coincidence that a future V change could silently break -- the assertion
# below is what keeps the >=MIN_CHUNKS floor honest if it does.
VLINEAR_CHUNK_SIZE = PROD_CHUNK_SIZE
if min(VLINEAR_VARIANTS) < MIN_CHUNKS * VLINEAR_CHUNK_SIZE:
    raise ValueError(
        f"VLINEAR_VARIANTS starts at {min(VLINEAR_VARIANTS):,}, below the "
        f"{MIN_CHUNKS * VLINEAR_CHUNK_SIZE:,} needed for >={MIN_CHUNKS} chunks at "
        f"chunk_size={VLINEAR_CHUNK_SIZE:,}. Raise the smallest rung; lowering "
        "the chunk size instead re-introduces the per-chunk-cost confound that "
        "made the V-law over-predict by 11.8x."
    )


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
    holdout.append(
        _point(
            corpus_dir / "holdout_f0.manifest.json",
            1,
            _chunk_size_for(HOLDOUT_F0["variants"]),
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

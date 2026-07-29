"""Fast regression tier: tiny corpora, committed baselines, a few minutes.

Guards the small-scale behaviour the cluster sweeps are too expensive to
re-run. Baselines are wall time and peak RSS at a handful of worker counts, and
a regression is a one-sided band -- getting faster is never a failure.

Measured (Slurm job 13332630, dedicated 8-CPU carter-compute allocation, via
`scripts/bench_svar2/regression_record.sbatch` -- the job that recorded the
committed baselines): 50 s to record from cold including corpus generation, and
50 s for a warm `pixi run bench-regression`. The same work on a contended login
node took 12+ minutes, so treat ~1 minute as the number for a dedicated
allocation and expect worse anywhere shared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.bench_svar2.records import (
    CorpusManifest,
    ProbeRecord,
    SweepPoint,
    from_json,
)
from scripts.bench_svar2.scale_corpus import GENERATOR_VERSION, generate

BASELINE_PATH = Path(__file__).parent / "baselines" / "regression.json"
# Shrunk in fix round 1 (Finding 3) from 20_000 variants to 2_000. The tier is
# still 3 points * (1 warm-up + 2 timed) = 9 conversions; only the corpus got
# smaller. Shrink the corpus, not the point list: reader_workers is the axis
# under test, so dropping a worker count would cost coverage. Shrinking it did
# not cost the maxrss_mb delta gate its signal -- that was measured, both ways,
# on dedicated allocations; see the comment on `HARD_METRICS`.
CORPUS = {"samples": 200, "variants": 2_000, "contigs": ["chr22"], "seed": 1234}
WORKERS = (1, 3, 7)
DEFAULT_TOLERANCE = 0.25
# Absolute floor on the maxrss_mb delta band, in MB, added to
# `tolerance * abs(want)` rather than replacing it.
#
# A pure percentage band is narrower than the metric's own reproducibility. Two
# dedicated 8-CPU recordings of IDENTICAL code disagree by more than 25% of the
# smaller one: worker-attributable deltas of 4.2 / 21.4 MB (job 13332630) and
# 6.73 / 27.38 MB (the recording committed to baselines/regression.json), i.e.
# gaps of 2.53 MB at w=3 and 5.98 MB at w=7. Had the 4.2 MB recording been the
# committed baseline, the run that actually produced the committed file would
# have FAILED the gate (6.73 > 4.2 * 1.25 = 5.25): a 60% false positive on
# unchanged code, on the tier whose whole job is to be trusted when it goes red.
#
# 8.0 MB clears the largest observed same-code disagreement (5.98 MB) with ~33%
# headroom. It costs little sensitivity: the signal this gate reads is ~21-27 MB
# at w=7, so an added 8 MB is still well under half of it, and any regression
# that meaningfully multiplies per-reader memory moves the delta by far more.
DELTA_FLOOR_MB = 8.0

# maxrss_mb gates on the WORKER-ATTRIBUTABLE DELTA, not the absolute value
# (fix round 2, Finding I8). The fixed interpreter + extension footprint
# dwarfs the part this tier exists to guard: the committed baselines are
# 437.8/442.0/459.2 MB for reader_workers=1/3/7, so only ~21 MB out of ~438 is
# attributable to the reader-worker axis, while 25% of the absolute baseline
# is ~110 MB. A change that quintuples the worker-attributable slice
# (21 MB -> 105 MB of extra RSS) still lands well inside a 25%-of-absolute
# band -- an absolute gate can only ever catch a regression that roughly
# doubles the WHOLE process footprint, which is not what this tier is for.
# Gating on `maxrss_mb(w) - maxrss_mb(w=1)` instead -- computed the same way
# on both the measured run and the baseline, so the fixed footprint cancels
# out on both sides -- isolates the axis under test. Absolute maxrss_mb is
# still recorded and printed every run (`INFO_METRICS`) as a trend signal,
# same treatment as wall_s.
#
# `_points` keeps `chunk_size=25_000` -- `from_vcf`'s production default, and,
# as measured, the value that MAXIMIZES the signal this gate reads. A round of
# review proposed shrinking it to 128 on the theory that a 25_000 chunk over a
# 2_000-variant corpus "never fills", so the reader axis could not move RSS.
# Both settings were then measured on dedicated 8-CPU allocations,
# worker-attributable delta (w=3 / w=7):
#     chunk_size=25_000 -> 4.2 / 21.4 MB   (job 13332630)
#     chunk_size=128    -> 3.6 /  7.8 MB   (job 13332816)
# Shrinking it roughly halved the w=7 signal, so 25_000 stays. That is the
# whole justification: the measurement. An earlier version of this comment
# explained the result by claiming every reader "pays" the full
# `BitGrid3::zeros(chunk_size, ...)` allocation that
# `ChunkAssembler::read_next_chunk` makes up front -- true of ADDRESS SPACE,
# false of RSS, which is what this gate reads: `BitGrid3::zeros` is
# `vec![0u64; n_words]` -> alloc_zeroed -> calloc, and untouched pages never
# become resident (measured on this node: a 3 GB zeroed allocation adds 0 MB to
# ru_maxrss). Do not restore that mechanism; whatever makes the larger chunk
# move RSS more here, it is not the untouched tail of the grid. Note the delta
# is small in absolute terms either way, because the tolerance is a fraction OF
# THE DELTA -- the band is ~13 MB at w=7 (25% plus DELTA_FLOOR_MB), not ~110 MB
# as it was against absolute RSS.
#
# wall_s is NOT a hard gate (fix round 1, Finding 2), and the reason is
# stronger than "the box is sometimes busy":
#
#   - On a contended login node the three points measured 68/54/119 s one day
#     and 198/250/302 s another, for a corpus 10x SMALLER the second time. A
#     25% band on top of a baseline like that absorbs a genuine 30-60%
#     slowdown whole and still prints "within tolerance" -- a false PASS.
#   - Recording on a dedicated 8-CPU allocation does not rescue it. Two such
#     recordings of IDENTICAL code, same corpus, same node type, disagree by
#     more than the band itself: job 13332508 recorded 7.2/6.2/5.5 s and job
#     13332630 recorded 5.3/5.3/5.2 s, i.e. the w=1 baseline moved -27% purely
#     between recordings. Immediate re-checks within each job swung -24%/-9%/
#     +13% and +6%/-1%/+0% respectively. At a ~140 KB corpus wall_s is mostly
#     process startup, so a 25% band is already near the noise floor even with
#     cores isolated. (Both jobs were dedicated 8-CPU cgroups on nodes at
#     loadavg ~22-31 per the job logs -- isolated cores, not idle machines. A
#     truly idle node would presumably be tighter, but nothing here has
#     measured one, so do not assume it.)
#
# So do NOT promote wall_s to HARD_METRICS on the strength of a dedicated
# allocation alone; the noise floor, not the contention, is what disqualifies
# it at this corpus size. It is still recorded and printed every run as a trend
# signal. Gating it would need a materially bigger corpus (which is the
# cluster sweep's job, not this tier's) or many reps and a median.
HARD_METRICS = ("maxrss_mb",)
INFO_METRICS = ("wall_s", "maxrss_mb")
# The reader_workers value whose measurement is subtracted off to get the
# worker-attributable delta. 1 reader worker is the floor of the axis under
# test, so its own delta against itself is trivially 0 -- the reference
# point isn't usefully self-gated, only the higher-worker points are.
DELTA_REFERENCE_WORKERS = 1


def _delta_regressed(
    got: float, want: float, tolerance: float, floor: float = 0.0
) -> bool:
    """One-sided band around a delta that may sit at or below zero.

    `got > want * (1 + tolerance)` is the band used everywhere else, but it
    inverts when `want <= 0`: multiplying a non-positive number by
    `1 + tolerance` makes it MORE negative, which tightens rather than
    relaxes the bound. Worker-attributable RSS deltas are expected to be
    small and non-negative, but baseline-recording noise (or a genuinely
    free extra reader) could land `want` at or below zero, so anchor the
    relaxation to `abs(want)` instead -- this is identical to the old
    formula whenever `want > 0`.

    `floor` widens the band to at least that many absolute units, for metrics
    whose run-to-run reproducibility is worse than `tolerance` of a small
    baseline (see `DELTA_FLOOR_MB`). It defaults to 0, i.e. off, so metrics
    without a measured noise floor keep the pure percentage band.
    """
    return got > want + max(tolerance * abs(want), floor)


def check(
    records: Sequence[ProbeRecord],
    points: Sequence[SweepPoint],
    baselines: dict[str, dict[str, float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[str]:
    """Hard regression gate.

    Only `HARD_METRICS` can fail the build. A failed run or a missing
    baseline is always reported regardless of metric -- those are never
    silently passed.

    `maxrss_mb` gates on the delta against the `reader_workers=1` point
    within this same `points`/`records` set (see the comment above
    `HARD_METRICS` for why absolute RSS is the wrong thing to gate on), with
    a `DELTA_FLOOR_MB` absolute floor under the percentage band because the
    delta's reproducibility across recordings is worse than `tolerance` of it.
    `points` supplies the `reader_workers` needed to find that reference. If
    no `reader_workers=1` point has both a measured record and a baseline,
    this falls back to the absolute comparison for every point rather than
    silently skipping the gate.
    """
    workers_by_id = {pt.point_id: pt.reader_workers for pt in points}
    ref_id = next(
        (pid for pid, w in workers_by_id.items() if w == DELTA_REFERENCE_WORKERS),
        None,
    )
    ref_record = (
        next((r for r in records if r.point_id == ref_id), None) if ref_id else None
    )
    ref_baseline = baselines.get(ref_id) if ref_id else None
    have_reference = (
        ref_record is not None and ref_record.ok and ref_baseline is not None
    )

    problems: list[str] = []
    for r in records:
        if not r.ok:
            problems.append(f"{r.point_id}: run failed ({r.error})")
            continue
        base = baselines.get(r.point_id)
        if base is None:
            problems.append(
                f"{r.point_id}: no baseline recorded -- regenerate with --record"
            )
            continue
        for metric in HARD_METRICS:
            if metric == "maxrss_mb" and have_reference:
                assert ref_record is not None and ref_baseline is not None
                got = r.maxrss_mb - ref_record.maxrss_mb
                want = base["maxrss_mb"] - ref_baseline["maxrss_mb"]
                label = f"maxrss_mb delta vs reader_workers={DELTA_REFERENCE_WORKERS}"
                # Only the delta carries a floor: DELTA_FLOOR_MB is derived
                # from two recordings OF THE DELTA, and the absolute-RSS
                # fallback below is a ~440 MB number whose 25% band already
                # dwarfs it.
                floor = DELTA_FLOOR_MB
            else:
                got = getattr(r, metric)
                want = base[metric]
                label = metric
                floor = 0.0
            if _delta_regressed(got, want, tolerance, floor):
                pct = f" ({100 * (got / want - 1):+.0f}%)" if want else ""
                problems.append(
                    f"{r.point_id}: {label} regressed {got:.1f} vs baseline "
                    f"{want:.1f}{pct}"
                )
    return problems


def _fmt_metric(metric: str, value: float) -> str:
    return f"{value:.1f}s" if metric == "wall_s" else f"{value:.1f}"


def info_deltas(
    records: Sequence[ProbeRecord], baselines: dict[str, dict[str, float]]
) -> list[str]:
    """Informational report for `INFO_METRICS`. Never fails the gate.

    Covers absolute `wall_s` and absolute `maxrss_mb` -- the latter moved
    here from `HARD_METRICS` in fix round 2 (Finding I8): it's still worth
    printing as a trend signal, it just isn't sensitive enough on its own to
    gate the build (see the comment above `HARD_METRICS`).

    Skips records that `check` already flags unconditionally (failed runs,
    missing baselines) so the two report streams don't duplicate each other.
    """
    out: list[str] = []
    for r in records:
        if not r.ok:
            continue
        base = baselines.get(r.point_id)
        if base is None:
            continue
        for metric in INFO_METRICS:
            got = getattr(r, metric)
            want = base[metric]
            pct = f"{100 * (got / want - 1):+.0f}%" if want else "n/a"
            out.append(
                f"{r.point_id}: {metric} {_fmt_metric(metric, got)} vs baseline "
                f"{_fmt_metric(metric, want)} ({pct})"
            )
    return out


def _worker_key(reader_workers: int) -> str:
    return str(reader_workers)


def _baselines_by_point_id(
    points: Sequence[SweepPoint],
    raw: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Re-key a reader_workers-keyed baseline file onto this session's point_ids.

    Fix round 1, Finding 1: `SweepPoint.point_id` hashes every field,
    including `corpus`, and `_points` derives `corpus` from `--workdir`, which
    defaults to `$CLAUDE_JOB_DIR/tmp/bench_reg`. `CLAUDE_JOB_DIR` is a
    per-session ephemeral path, so `point_id` is session-specific and cannot
    be the key a baseline file committed to git survives across sessions on
    -- `pixi run bench-regression` in any *other* session, same box, same
    cores, zero code change, would report "no baseline recorded" for every
    point. reader_workers is the actual axis this tier guards and is stable
    across sessions, so the committed file is keyed by reader_workers and
    paired back onto whatever point_ids this session's `_points()` produced.

    The pairing reads `reader_workers` off each point rather than zipping
    against a parallel `WORKERS` sequence (fix round 2, N6): a positional zip
    is a second source of truth for a fact the point already carries, and
    since the gate is one-sided, a mispaired baseline never fails loudly -- it
    just silently stops gating that point.
    """
    out: dict[str, dict[str, float]] = {}
    for pt in points:
        base = raw.get(_worker_key(pt.reader_workers))
        if base is not None:
            out[pt.point_id] = base
    return out


def corpus_is_current(manifest_path: Path) -> bool:
    """True when the corpus already on disk was generated from this `CORPUS`.

    The corpus is cached by *filename*, so without this check a change to
    `CORPUS` is silently ignored in any workdir that already holds a
    `reg.vcf.gz` -- the tier would keep measuring the old corpus. That is not a
    cosmetic staleness bug: the hard gate is one-sided (only a *higher*
    `maxrss_mb` fails), so baselines recorded against a large corpus and
    checked against a smaller one pass vacuously, for every point, forever.
    Compare the manifest against `CORPUS` rather than trusting the path.
    """
    if not manifest_path.exists():
        return False
    m = from_json(CorpusManifest, manifest_path.read_text())
    # The manifest is written last, so its presence implies generation
    # finished -- but the workdir defaults to scratch under $CLAUDE_JOB_DIR,
    # which gets cleaned. Confirm the corpus it describes is still there (N4),
    # otherwise the tier crashes inside run_point instead of regenerating.
    if not Path(m.path).exists():
        return False
    # `generate` writes floor(variants / n_contigs) * n_contigs records, so
    # comparing the request against the manifest is only valid once floored
    # (N5). Without this, any future CORPUS whose variants are not divisible
    # by the contig count regenerates on EVERY invocation, forever.
    contigs = list(CORPUS["contigs"])
    want_variants = (CORPUS["variants"] // len(contigs)) * len(contigs)
    return (
        m.samples == CORPUS["samples"]
        and m.variants == want_variants
        and list(m.contigs) == contigs
        and m.seed == CORPUS["seed"]
        and list(m.format_fields) == []
        # GENERATOR_VERSION exists to say "the generation logic changed, the
        # bytes differ" (N3). Ignoring it is the exact vacuous-pass this
        # function was written to prevent, just triggered by a code change
        # rather than a CORPUS change.
        and m.generator_version == GENERATOR_VERSION
    )


def _points(manifest_path: Path, threads: int) -> list[SweepPoint]:
    return [
        SweepPoint(
            corpus=str(manifest_path),
            reader_workers=w,
            concurrent_chroms=None,
            shard_htslib=0,
            overshard=4,
            # `from_vcf`'s production default, kept because it measured as the
            # stronger signal for the maxrss_mb delta gate -- see the numbers
            # in the comment above `HARD_METRICS`.
            chunk_size=25_000,
            threads=threads,
            reps=2,
        )
        for w in WORKERS
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--record", action="store_true", help="write baselines instead of checking"
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=Path(os.environ.get("CLAUDE_JOB_DIR", ".")) / "tmp" / "bench_reg",
    )
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    a = p.parse_args()

    from scripts.bench_svar2.probe import run_point

    a.workdir.mkdir(parents=True, exist_ok=True)
    vcf = a.workdir / "reg.vcf.gz"
    manifest_path = vcf.with_suffix("").with_suffix(".manifest.json")
    if not corpus_is_current(manifest_path):
        generate(vcf, format_fields=(), procs=4, bgzip_threads=2, **CORPUS)
    manifest = from_json(CorpusManifest, manifest_path.read_text())

    threads = len(os.sched_getaffinity(0))
    points = _points(manifest_path, threads)
    # warm=True: keep `run_point`'s untimed rep-0. Fix round 1 initially set
    # warm=False to buy back time (the corpus is freshly generated, and the
    # hard gate is maxrss_mb, which does not care). Measurement said otherwise:
    # recording without it on a dedicated 8-CPU allocation gave 13.4/13.4/3.7 s
    # for workers 1/3/7, and an immediate re-check on the same node gave a flat
    # 3.3/3.2/3.3 s. The 4x spread tracked point ORDER, not reader_workers --
    # the first points were paying one-time first-touch cost (pixi env and the
    # extension .so off NFS) that rep-0 exists to absorb. Baking that into the
    # baseline makes every later run print a spurious "-75%" improvement. Three
    # extra conversions cost ~30 s here, which is cheap for a steady-state
    # number.
    records = [run_point(pt, manifest, a.workdir, warm=True) for pt in points]

    if a.record:
        # Never record from a failed run (N1). `run_point` returns ok=False
        # records that still carry a maxrss_mb from the crashed child, and
        # recording happens unattended via sbatch, so a bad number would be
        # committed with nobody watching. Because the gate is one-sided, a
        # baseline inflated by a crash can never be exceeded again -- the tier
        # silently stops gating -- while one deflated by an early crash fails
        # every healthy run afterwards.
        failed = [
            (pt.reader_workers, r.error) for pt, r in zip(points, records) if not r.ok
        ]
        if failed:
            for w, err in failed:
                print(f"FAILED: reader_workers={w}: {err}", file=sys.stderr)
            print(
                f"{len(failed)}/{len(records)} points failed -- "
                f"{BASELINE_PATH} left unchanged",
                file=sys.stderr,
            )
            sys.exit(1)
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threads": threads,
            "points": {
                _worker_key(pt.reader_workers): {
                    "wall_s": r.wall_s,
                    "maxrss_mb": r.maxrss_mb,
                }
                for pt, r in zip(points, records)
            },
        }
        BASELINE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"recorded {len(records)} baselines to {BASELINE_PATH}")
        return

    raw = json.loads(BASELINE_PATH.read_text())
    baselines = _baselines_by_point_id(points, raw["points"])
    problems = check(records, points, baselines, a.tolerance)
    # `threads` reaches the conversion as `-@ N` and sizes a rayon pool, so it
    # moves maxrss_mb -- the metric that gates (N2). Before baselines were
    # re-keyed off point_id, a width change showed up as a loud "no baseline
    # recorded"; now it would compare silently. Refuse instead: a number taken
    # at another width is not a baseline for this run, and the one-sided gate
    # means a too-wide baseline fails open rather than loudly.
    if raw["threads"] != threads:
        problems.append(
            f"baselines were recorded at threads={raw['threads']} but this run has "
            f"threads={threads} (allocation width); maxrss_mb is not comparable "
            f"across widths -- re-record with "
            f"`sbatch scripts/bench_svar2/regression_record.sbatch`"
        )
    for msg in problems:
        print(f"REGRESSION: {msg}", file=sys.stderr)
    for msg in info_deltas(records, baselines):
        print(f"INFO: {msg}")
    if problems:
        sys.exit(1)
    print(
        f"{len(records)} points within {a.tolerance:.0%} of baseline "
        "(maxrss_mb delta-vs-reader_workers=1 hard gate; "
        "wall_s and absolute maxrss_mb informational)"
    )


if __name__ == "__main__":
    main()

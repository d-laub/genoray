"""Fast regression tier: tiny corpora, committed baselines, a few minutes.

Guards the small-scale behaviour the cluster sweeps are too expensive to
re-run. Baselines are wall time and peak RSS at a handful of worker counts, and
a regression is a one-sided band -- getting faster is never a failure.

Measured (Slurm job 13332508, dedicated 8-CPU carter-compute allocation, via
`scripts/bench_svar2/regression_record.sbatch`): 65 s to record from cold
including corpus generation, and 58 s for a warm `pixi run bench-regression`.
The same work on a contended login node took 12+ minutes, so treat ~1 minute as
the number for a dedicated allocation and expect worse anywhere shared.
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
from scripts.bench_svar2.scale_corpus import generate

BASELINE_PATH = Path(__file__).parent / "baselines" / "regression.json"
# Shrunk in fix round 1 (Finding 3) from 20_000 variants to 2_000. The tier is
# still 3 points * (1 warm-up + 2 timed) = 9 conversions; only the corpus got
# smaller. Shrink the corpus, not the point list: reader_workers is the axis
# under test, so dropping a worker count would cost coverage, whereas a corpus
# 10x smaller costs only resolution the maxrss_mb gate does not need.
CORPUS = {"samples": 200, "variants": 2_000, "contigs": ["chr22"], "seed": 1234}
WORKERS = (1, 3, 7)
DEFAULT_TOLERANCE = 0.25

# maxrss_mb is driven by corpus size and the dense-chunk cost model, so it is
# low-variance and safe as a hard gate. wall_s is NOT a hard gate (fix round 1,
# Finding 2), and the reason is stronger than "the box is sometimes busy":
#
#   - On a contended login node the three points measured 68/54/119 s one day
#     and 198/250/302 s another, for a corpus 10x SMALLER the second time. A
#     25% band on top of a baseline like that absorbs a genuine 30-60%
#     slowdown whole and still prints "within tolerance" -- a false PASS.
#   - Recording on a dedicated 8-CPU allocation does not rescue it. Job
#     13332508 recorded 7.2/6.2/5.5 s and an immediate re-check on that same
#     idle node, same code, same corpus, returned -24%/-9%/+13%. At a ~140 KB
#     corpus, wall_s is mostly process startup, so run-to-run noise already
#     fills most of a 25% band even under ideal conditions.
#
# So do NOT promote wall_s to HARD_METRICS on the strength of a dedicated
# allocation alone; the noise floor, not the contention, is what disqualifies
# it at this corpus size. It is still recorded and printed every run as a trend
# signal. Gating it would need a materially bigger corpus (which is the
# cluster sweep's job, not this tier's) or many reps and a median.
HARD_METRICS = ("maxrss_mb",)
INFO_METRICS = ("wall_s",)


def check(
    records: Sequence[ProbeRecord],
    baselines: dict[str, dict[str, float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[str]:
    """Hard regression gate.

    Only `HARD_METRICS` can fail the build. A failed run or a missing
    baseline is always reported regardless of metric -- those are never
    silently passed.
    """
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
            got = getattr(r, metric)
            want = base[metric]
            if got > want * (1 + tolerance):
                problems.append(
                    f"{r.point_id}: {metric} regressed {got:.1f} vs baseline "
                    f"{want:.1f} (+{100 * (got / want - 1):.0f}%)"
                )
    return problems


def wall_deltas(
    records: Sequence[ProbeRecord], baselines: dict[str, dict[str, float]]
) -> list[str]:
    """Informational wall_s report. Never fails the gate -- see `INFO_METRICS`.

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
            out.append(
                f"{r.point_id}: {metric} {got:.1f}s vs baseline {want:.1f}s "
                f"({100 * (got / want - 1):+.0f}%)"
            )
    return out


def _worker_key(reader_workers: int) -> str:
    return str(reader_workers)


def _baselines_by_point_id(
    points: Sequence[SweepPoint],
    workers: Sequence[int],
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
    paired back onto whatever point_ids this session's `_points()` produced,
    by position (both `WORKERS` and `_points()` iterate the same tuple in the
    same order).
    """
    out: dict[str, dict[str, float]] = {}
    for pt, w in zip(points, workers):
        key = _worker_key(w)
        if key in raw:
            out[pt.point_id] = raw[key]
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
    return (
        m.samples == CORPUS["samples"]
        and m.variants == CORPUS["variants"]
        and list(m.contigs) == list(CORPUS["contigs"])
        and m.seed == CORPUS["seed"]
        and list(m.format_fields) == []
    )


def _points(manifest_path: Path, threads: int) -> list[SweepPoint]:
    return [
        SweepPoint(
            corpus=str(manifest_path),
            reader_workers=w,
            concurrent_chroms=None,
            shard_htslib=0,
            overshard=4,
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
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        baselines = {
            _worker_key(w): {"wall_s": r.wall_s, "maxrss_mb": r.maxrss_mb}
            for w, r in zip(WORKERS, records)
        }
        BASELINE_PATH.write_text(json.dumps(baselines, indent=2, sort_keys=True))
        print(f"recorded {len(records)} baselines to {BASELINE_PATH}")
        return

    raw_baselines = json.loads(BASELINE_PATH.read_text())
    baselines = _baselines_by_point_id(points, WORKERS, raw_baselines)
    problems = check(records, baselines, a.tolerance)
    for msg in problems:
        print(f"REGRESSION: {msg}", file=sys.stderr)
    for msg in wall_deltas(records, baselines):
        print(f"INFO: {msg}")
    if problems:
        sys.exit(1)
    print(
        f"{len(records)} points within {a.tolerance:.0%} of baseline "
        "(maxrss_mb hard gate; wall_s informational)"
    )


if __name__ == "__main__":
    main()

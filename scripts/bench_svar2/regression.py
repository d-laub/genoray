"""Fast regression tier: tiny corpora, committed baselines, ~2 minutes.

Guards the small-scale behaviour the cluster sweeps are too expensive to
re-run. Baselines are wall time and peak RSS at a handful of worker counts, and
a regression is a one-sided band -- getting faster is never a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.bench_svar2.records import ProbeRecord, SweepPoint
from scripts.bench_svar2.scale_corpus import generate

BASELINE_PATH = Path(__file__).parent / "baselines" / "regression.json"
# Small enough to run in ~2 minutes on a laptop-class allocation.
CORPUS = {"samples": 200, "variants": 20_000, "contigs": ["chr22"], "seed": 1234}
WORKERS = (1, 3, 7)
DEFAULT_TOLERANCE = 0.25


def check(
    records: Sequence[ProbeRecord],
    baselines: dict[str, dict[str, float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[str]:
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
        for metric in ("wall_s", "maxrss_mb"):
            got = getattr(r, metric)
            want = base[metric]
            if got > want * (1 + tolerance):
                problems.append(
                    f"{r.point_id}: {metric} regressed {got:.1f} vs baseline "
                    f"{want:.1f} (+{100 * (got / want - 1):.0f}%)"
                )
    return problems


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
    from scripts.bench_svar2.records import CorpusManifest, from_json

    a.workdir.mkdir(parents=True, exist_ok=True)
    vcf = a.workdir / "reg.vcf.gz"
    if not vcf.exists():
        generate(vcf, format_fields=(), procs=4, bgzip_threads=2, **CORPUS)
    manifest_path = vcf.with_suffix("").with_suffix(".manifest.json")
    manifest = from_json(CorpusManifest, manifest_path.read_text())

    threads = len(os.sched_getaffinity(0))
    records = [
        run_point(pt, manifest, a.workdir) for pt in _points(manifest_path, threads)
    ]

    if a.record:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    r.point_id: {"wall_s": r.wall_s, "maxrss_mb": r.maxrss_mb}
                    for r in records
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"recorded {len(records)} baselines to {BASELINE_PATH}")
        return

    problems = check(records, json.loads(BASELINE_PATH.read_text()), a.tolerance)
    for msg in problems:
        print(f"REGRESSION: {msg}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print(f"{len(records)} points within {a.tolerance:.0%} of baseline")


if __name__ == "__main__":
    main()

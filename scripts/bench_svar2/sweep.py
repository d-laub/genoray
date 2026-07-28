"""Execute a plan of sweep points, resumably.

Holds no domain knowledge -- only execution, resumption and the oracle check.
A full sweep is an overnight job on a shared, preemptible cluster, so every
finished point is durably appended before the next one starts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from scripts.bench_svar2.records import (
    CorpusManifest,
    ProbeRecord,
    SweepPoint,
    append_ndjson,
    from_json,
    read_ndjson,
)


def load_plan(path: Path) -> list[SweepPoint]:
    raw = json.loads(Path(path).read_text())
    return [SweepPoint(**entry) for entry in raw]


def pending_points(plan: Sequence[SweepPoint], results_path: Path) -> list[SweepPoint]:
    done = {r.point_id for r in read_ndjson(results_path, ProbeRecord)}
    return [p for p in plan if p.point_id not in done]


def check_oracle(records: Sequence[ProbeRecord]) -> str | None:
    """Every successful configuration must produce a byte-identical store.
    Returns an error message, or None when all digests agree."""
    digests = {r.digest for r in records if r.ok and r.digest}
    if len(digests) > 1:
        return f"digest mismatch across configurations: {sorted(digests)}"
    return None


def run_sweep(
    plan_path: Path,
    results_path: Path,
    outdir: Path,
    runner: Callable[..., ProbeRecord] | None = None,
) -> list[ProbeRecord]:
    if runner is None:
        from scripts.bench_svar2.probe import run_point as runner  # noqa: PLW0127

    plan = load_plan(plan_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, CorpusManifest] = {}
    for point in pending_points(plan, results_path):
        if point.corpus not in manifests:
            manifests[point.corpus] = from_json(
                CorpusManifest, Path(point.corpus).read_text()
            )
        rec = runner(point, manifests[point.corpus], outdir)
        append_ndjson(results_path, rec)
        status = "OOM" if rec.oom_at_rss_mb else ("ok" if rec.ok else "FAIL")
        print(
            f"w={point.reader_workers:>3} cs={point.chunk_size:>6} "
            f"| wall {rec.wall_s:7.2f}s | phase1 {rec.phase1_s:6.2f}s "
            f"| rss {rec.maxrss_mb:7.0f}MB | pending_hw {rec.pending_highwater:>3} "
            f"| {status}",
            flush=True,
        )

    records = read_ndjson(results_path, ProbeRecord)
    problem = check_oracle(records)
    if problem:
        raise RuntimeError(problem)
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    recs = run_sweep(a.plan, a.results, a.outdir)
    print(f"{len(recs)} points recorded to {a.results}")


if __name__ == "__main__":
    main()

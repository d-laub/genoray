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


def check_oracle(
    records: Sequence[ProbeRecord], plan: Sequence[SweepPoint]
) -> str | None:
    """Every successful configuration WITHIN THE SAME CORPUS must produce a
    byte-identical store -- sharding is byte-identical, so the store hash
    must not move across configurations of one corpus. Different corpora
    hold different variant data and legitimately produce different digests,
    so the check is grouped by corpus, not pooled across the whole sweep.

    `records` alone carry no corpus (that's a `SweepPoint` field, not a
    `ProbeRecord` field), so the point_id -> corpus mapping is rebuilt from
    `plan` on every call. A record whose `point_id` is no longer in `plan`
    (the plan was edited between runs) is skipped rather than crashing --
    there is nothing to attribute it to.

    Returns an error message naming the offending corpus, or None when every
    corpus's digests agree."""
    corpus_by_point = {p.point_id: p.corpus for p in plan}
    digests_by_corpus: dict[str, set[str]] = {}
    for r in records:
        if not (r.ok and r.digest):
            continue
        corpus = corpus_by_point.get(r.point_id)
        if corpus is None:
            continue
        digests_by_corpus.setdefault(corpus, set()).add(r.digest)
    for corpus, digests in digests_by_corpus.items():
        if len(digests) > 1:
            return f"digest mismatch within corpus {corpus!r}: {sorted(digests)}"
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
        # Fail fast, not just at the end: `append_ndjson` already fsynced
        # this record, so nothing is at risk by checking now -- the results
        # file keeps exactly what it would have kept either way. A genuine
        # within-corpus digest divergence is systematic and will recur at
        # every remaining point of that corpus, so catching it here saves
        # the rest of a preemptible overnight sweep instead of burning it
        # before the failure is even reported.
        problem = check_oracle(read_ndjson(results_path, ProbeRecord), plan)
        if problem:
            raise RuntimeError(problem)

    records = read_ndjson(results_path, ProbeRecord)
    problem = check_oracle(records, plan)
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

"""Execute a plan of sweep points, resumably.

Holds no domain knowledge -- only execution, resumption and the oracle check.
A full sweep is an overnight job on a shared, preemptible cluster, so every
finished point is durably appended before the next one starts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


def build_code_id() -> str:
    """SHA256 (16 hex) of the built `_core` extension this process loaded.

    The ARTIFACT, not the git commit: `pixi run test` does not rebuild the
    extension, so a commit can advance while the measured binary does not.
    `probe.run_point` launches the child with `sys.executable`, so the child
    loads this same extension.
    """
    import genoray._core as _core

    return hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class SweepResult:
    """`measured` and `reused` are reported separately because the row count
    is the size of the output FILE, not work performed: the contaminated run
    on PR #154 printed `18 points recorded` for 6 measurements (#159)."""

    records: list[ProbeRecord]
    measured: int
    reused: int


def pending_points(
    plan: Sequence[SweepPoint], results_path: Path, code_id: str
) -> list[SweepPoint]:
    """Points still to measure AGAINST THIS BUILD.

    Keyed on `(point_id, code_id)`, not `point_id` alone: `point_id` hashes
    the configuration only, so it cannot distinguish two runs of one
    configuration against different code -- which is the one distinction a
    benchmark exists to make (issue #159).
    """
    done = {(r.point_id, r.code_id) for r in read_ndjson(results_path, ProbeRecord)}
    return [p for p in plan if (p.point_id, code_id) not in done]


def check_corpora(points: Sequence[SweepPoint]) -> None:
    """Raise naming EVERY plan point whose corpus manifest is absent.

    `run_sweep` loads manifests lazily inside the point loop, so a plan that
    names a corpus nobody generates fails hours into an overnight job -- and
    under `set -euo pipefail` that aborts the whole sbatch (issue #151, and
    #141 before it for `vlinear2`). Reporting all of them at once means one
    generation pass fixes the run, rather than one per resubmit.
    """
    missing = sorted({p.corpus for p in points if not Path(p.corpus).exists()})
    if missing:
        listed = "\n".join(f"  {m}" for m in missing)
        raise FileNotFoundError(
            f"{len(missing)} corpus manifest(s) named by the plan do not "
            f"exist:\n{listed}"
        )


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
) -> SweepResult:
    if runner is None:
        from scripts.bench_svar2.probe import run_point as runner  # noqa: PLW0127

    plan = load_plan(plan_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    code_id = build_code_id()
    run_id = os.environ.get("SLURM_JOB_ID") or uuid.uuid4().hex[:16]

    pending = pending_points(plan, results_path, code_id)
    # Pending only, not the whole plan: a fully-resumed sweep whose corpora
    # were since cleaned up has nothing left to read and must not fail.
    check_corpora(pending)
    reused = len(plan) - len(pending)
    measured = 0

    manifests: dict[str, CorpusManifest] = {}
    for point in pending:
        if point.corpus not in manifests:
            manifests[point.corpus] = from_json(
                CorpusManifest, Path(point.corpus).read_text()
            )
        rec = runner(point, manifests[point.corpus], outdir)
        # Stamped here, not in `run_point`: one site, and an injected test
        # runner does not have to know about provenance.
        rec = dataclasses.replace(rec, code_id=code_id, run_id=run_id)
        append_ndjson(results_path, rec)
        measured += 1
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
    return SweepResult(records=records, measured=measured, reused=reused)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    res = run_sweep(a.plan, a.results, a.outdir)
    print(
        f"{res.measured} measured, {res.reused} reused "
        f"({len(res.records)} rows in {a.results})"
    )


if __name__ == "__main__":
    main()

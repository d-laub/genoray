import json
from types import SimpleNamespace

import pytest

from scripts.bench_svar2.records import (
    ProbeRecord,
    SweepPoint,
    append_ndjson,
    read_ndjson,
)
from scripts.bench_svar2.sweep import check_oracle, load_plan, pending_points, run_sweep

pytestmark = pytest.mark.bench


def _plan_file(tmp_path, n=3):
    pts = [
        {
            "corpus": str(tmp_path / "c.manifest.json"),
            "reader_workers": w,
            "concurrent_chroms": None,
            "shard_htslib": 0,
            "overshard": 4,
            "chunk_size": 25_000,
            "threads": 16,
            "reps": 1,
        }
        for w in range(1, n + 1)
    ]
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(pts))
    return p


def _plan_pts(mapping: dict[str, str]):
    """Lightweight point_id -> corpus stand-ins for check_oracle's plan
    argument. check_oracle only reads `.point_id`/`.corpus`, so a real
    `SweepPoint` (whose `point_id` is a derived hash, not settable) isn't
    needed to pin specific point_ids like "a"/"b" for these tests."""
    return [
        SimpleNamespace(point_id=pid, corpus=corpus) for pid, corpus in mapping.items()
    ]


def _rec(pid: str, digest: str = "aaa", ok: bool = True) -> ProbeRecord:
    return ProbeRecord(
        point_id=pid,
        ok=ok,
        wall_s=1.0,
        phase1_s=1.0,
        cpu_s=1.0,
        maxrss_mb=1.0,
        digest=digest,
        dense_cap=6,
        dense_occupancy=(0,),
        cpu_shard_pct=(100.0,),
        cpu_exec_pct=(50.0,),
        pending_highwater=0,
        pending_bytes_highwater=0,
        shard_unit_secs=(1.0,),
    )


def test_load_plan_returns_sweep_points(tmp_path):
    pts = load_plan(_plan_file(tmp_path))
    assert len(pts) == 3
    assert all(isinstance(p, SweepPoint) for p in pts)


def test_pending_skips_already_recorded_points(tmp_path):
    plan = load_plan(_plan_file(tmp_path))
    results = tmp_path / "r.ndjson"
    append_ndjson(results, _rec(plan[0].point_id))
    remaining = pending_points(plan, results)
    assert [p.point_id for p in remaining] == [plan[1].point_id, plan[2].point_id]


def test_pending_returns_all_when_no_results_yet(tmp_path):
    plan = load_plan(_plan_file(tmp_path))
    assert len(pending_points(plan, tmp_path / "absent.ndjson")) == 3


def test_run_sweep_is_resumable(tmp_path):
    """A preempted Slurm job must resume, not restart."""
    plan_path = _plan_file(tmp_path)
    results = tmp_path / "r.ndjson"
    calls = []

    def runner(point, manifest, outdir, warm=True):
        calls.append(point.point_id)
        return _rec(point.point_id)

    (tmp_path / "c.manifest.json").write_text(
        json.dumps(
            {
                "path": str(tmp_path / "c.vcf.gz"),
                "samples": 10,
                "variants": 100,
                "contigs": ["chr22"],
                "format_fields": [],
                "ploidy": 2,
                "cells": 1000,
                "compressed_bytes": 10,
                "seed": 1,
                "generator_version": 1,
            }
        )
    )

    run_sweep(plan_path, results, tmp_path / "out", runner=runner)
    assert len(calls) == 3
    calls.clear()
    run_sweep(plan_path, results, tmp_path / "out", runner=runner)
    assert calls == []


def test_run_sweep_fails_fast_on_a_within_corpus_digest_mismatch(tmp_path):
    """append_ndjson already fsyncs every record, so nothing is lost by
    checking the oracle after each point instead of only at the end -- and a
    genuine same-corpus digest divergence is systematic, so it should
    surface immediately instead of burning the rest of a preemptible
    overnight sweep on points that would only confirm it."""
    plan_path = _plan_file(tmp_path, n=3)
    results = tmp_path / "r.ndjson"
    calls = []

    (tmp_path / "c.manifest.json").write_text(
        json.dumps(
            {
                "path": str(tmp_path / "c.vcf.gz"),
                "samples": 10,
                "variants": 100,
                "contigs": ["chr22"],
                "format_fields": [],
                "ploidy": 2,
                "cells": 1000,
                "compressed_bytes": 10,
                "seed": 1,
                "generator_version": 1,
            }
        )
    )

    digests = iter(["aaa", "bbb", "ccc"])

    def runner(point, manifest, outdir, warm=True):
        calls.append(point.point_id)
        return _rec(point.point_id, digest=next(digests))

    with pytest.raises(RuntimeError, match="digest mismatch"):
        run_sweep(plan_path, results, tmp_path / "out", runner=runner)

    # the mismatching second point tripped the check; the third never ran.
    assert len(calls) == 2
    # both attempted points are durably recorded despite the abort --
    # append_ndjson fsyncs before the oracle check runs.
    assert len(read_ndjson(results, ProbeRecord)) == 2


def test_check_oracle_flags_a_digest_mismatch():
    plan = _plan_pts({"a": "c1", "b": "c1"})
    recs = [_rec("a", "aaa"), _rec("b", "bbb")]
    assert check_oracle(recs, plan) is not None


def test_check_oracle_passes_when_all_digests_agree():
    plan = _plan_pts({"a": "c1", "b": "c1"})
    assert check_oracle([_rec("a", "aaa"), _rec("b", "aaa")], plan) is None


def test_check_oracle_ignores_failed_runs():
    """An OOM datum has no digest and must not be read as a mismatch."""
    plan = _plan_pts({"a": "c1", "b": "c1"})
    assert check_oracle([_rec("a", "aaa"), _rec("b", "", ok=False)], plan) is None


def test_check_oracle_allows_cross_corpus_divergence_but_flags_within_corpus():
    """Different corpora hold different variant data and legitimately
    produce different digests -- pooling digests across the whole sweep
    (ungrouped) would false-positive on every real multi-corpus sweep.
    Only a divergence WITHIN one corpus is a real oracle failure."""
    plan = _plan_pts({"a": "c1", "b": "c2", "c": "c1"})

    # a (c1) and b (c2) disagree, but they're different corpora: no error.
    cross_corpus = [_rec("a", "aaa"), _rec("b", "bbb")]
    assert check_oracle(cross_corpus, plan) is None

    # a and c are both c1 and disagree: a real within-corpus mismatch.
    within_corpus = [_rec("a", "aaa"), _rec("c", "zzz")]
    err = check_oracle(within_corpus, plan)
    assert err is not None
    assert "c1" in err


def test_check_oracle_ignores_records_whose_point_id_left_the_plan():
    """A plan edited between runs shouldn't crash the oracle check -- a
    stale record with no home in the current plan is simply unattributable,
    not evidence of a mismatch."""
    plan = _plan_pts({"a": "c1"})
    recs = [_rec("a", "aaa"), _rec("stale", "zzz")]
    assert check_oracle(recs, plan) is None

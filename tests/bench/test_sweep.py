import json

import pytest

from scripts.bench_svar2.records import ProbeRecord, SweepPoint, append_ndjson
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


def test_check_oracle_flags_a_digest_mismatch():
    recs = [_rec("a", "aaa"), _rec("b", "bbb")]
    assert check_oracle(recs) is not None


def test_check_oracle_passes_when_all_digests_agree():
    assert check_oracle([_rec("a", "aaa"), _rec("b", "aaa")]) is None


def test_check_oracle_ignores_failed_runs():
    """An OOM datum has no digest and must not be read as a mismatch."""
    assert check_oracle([_rec("a", "aaa"), _rec("b", "", ok=False)]) is None

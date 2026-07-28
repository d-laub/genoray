import pytest

from scripts.bench_svar2.records import ProbeRecord
from scripts.bench_svar2.regression import check

pytestmark = pytest.mark.bench


def _rec(pid: str, wall: float, rss: float = 100.0) -> ProbeRecord:
    return ProbeRecord(
        point_id=pid,
        ok=True,
        wall_s=wall,
        phase1_s=wall,
        cpu_s=wall,
        maxrss_mb=rss,
        digest="aaa",
        dense_cap=6,
        dense_occupancy=(0,),
        cpu_shard_pct=(100.0,),
        cpu_exec_pct=(50.0,),
        pending_highwater=0,
        pending_bytes_highwater=0,
        shard_unit_secs=(1.0,),
    )


BASE = {"p0": {"wall_s": 10.0, "maxrss_mb": 100.0}}


def test_within_tolerance_reports_nothing():
    assert check([_rec("p0", 11.0)], BASE, tolerance=0.25) == []


def test_wall_regression_is_reported():
    problems = check([_rec("p0", 14.0)], BASE, tolerance=0.25)
    assert len(problems) == 1
    assert "wall_s" in problems[0]


def test_rss_regression_is_reported():
    problems = check([_rec("p0", 10.0, rss=200.0)], BASE, tolerance=0.25)
    assert any("maxrss_mb" in p for p in problems)


def test_improvement_is_not_a_regression():
    assert check([_rec("p0", 4.0, rss=10.0)], BASE, tolerance=0.25) == []


def test_missing_baseline_is_reported_not_silently_passed():
    problems = check([_rec("unknown", 10.0)], BASE, tolerance=0.25)
    assert any("no baseline" in p for p in problems)


def test_failed_run_is_always_a_regression():
    bad = ProbeRecord(**{**_rec("p0", 10.0).__dict__, "ok": False, "error": "boom"})
    assert check([bad], BASE, tolerance=0.25) != []

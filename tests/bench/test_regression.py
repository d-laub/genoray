from pathlib import Path

import pytest

from scripts.bench_svar2.records import CorpusManifest, ProbeRecord, SweepPoint, to_json
from scripts.bench_svar2.regression import (
    CORPUS,
    WORKERS,
    _baselines_by_point_id,
    _points,
    check,
    corpus_is_current,
    info_deltas,
)
from scripts.bench_svar2.scale_corpus import GENERATOR_VERSION

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


def _rec_for(pt: SweepPoint, wall: float, rss: float) -> ProbeRecord:
    """Like `_rec`, but keyed to a real `SweepPoint.point_id` -- needed
    whenever a test exercises the maxrss_mb delta gate, which looks up the
    reader_workers=1 sibling by point_id via `points`."""
    return _rec(pt.point_id, wall, rss)


BASE = {"p0": {"wall_s": 10.0, "maxrss_mb": 100.0}}

# These tests use a bare "p0" point_id with no matching SweepPoint, so
# `check`'s delta gate can't find a reader_workers=1 reference and falls
# back to the absolute comparison (see `check`'s docstring). That fallback
# is exactly the pre-Finding-I8 behaviour, so passing `points=[]` here
# doubles as coverage for the fallback path itself.


def test_within_tolerance_reports_nothing():
    assert check([_rec("p0", 11.0)], [], BASE, tolerance=0.25) == []


def test_wall_regression_alone_is_not_a_hard_failure():
    # wall_s is informational only (fix round 1, Finding 2): baselines are
    # recorded under whatever contention happens to be on the box at record
    # time, so a wall-time-only regression must not fail the gate.
    assert check([_rec("p0", 14.0)], [], BASE, tolerance=0.25) == []


def test_wall_delta_is_still_reported_informationally():
    msgs = info_deltas([_rec("p0", 14.0)], BASE)
    assert any("wall_s" in m for m in msgs)


def test_absolute_maxrss_is_still_reported_informationally():
    # Finding I8: maxrss_mb moved from HARD_METRICS to INFO_METRICS (as an
    # absolute value, not a delta) -- it must still show up as a trend
    # signal even though it no longer gates.
    msgs = info_deltas([_rec("p0", 10.0, rss=150.0)], BASE)
    assert any("maxrss_mb" in m for m in msgs)


def test_rss_regression_is_reported_via_absolute_fallback():
    # No reader_workers=1 reference available (points=[]) -> falls back to
    # the absolute comparison, so this is the same shape of gate that
    # predates Finding I8, just reached through the fallback path.
    problems = check([_rec("p0", 10.0, rss=200.0)], [], BASE, tolerance=0.25)
    assert any("maxrss_mb" in p for p in problems)


def test_improvement_is_not_a_regression():
    assert check([_rec("p0", 4.0, rss=10.0)], [], BASE, tolerance=0.25) == []


def test_missing_baseline_is_reported_not_silently_passed():
    problems = check([_rec("unknown", 10.0)], [], BASE, tolerance=0.25)
    assert any("no baseline" in p for p in problems)


def test_failed_run_is_always_a_regression():
    bad = ProbeRecord(**{**_rec("p0", 10.0).__dict__, "ok": False, "error": "boom"})
    assert check([bad], [], BASE, tolerance=0.25) != []


# --- Finding I8: the maxrss_mb hard gate is a worker-attributable delta -----

# Baseline numbers straight from the finding writeup: 437.8/442.0/459.2 MB
# for reader_workers=1/3/7, i.e. only ~4.2/21.4 MB is worker-attributable
# out of a ~438 MB absolute baseline.
_DELTA_RAW_BASELINE = {
    "1": {"wall_s": 5.3, "maxrss_mb": 437.8},
    "3": {"wall_s": 5.3, "maxrss_mb": 442.0},
    "7": {"wall_s": 5.2, "maxrss_mb": 459.2},
}


def test_delta_gate_catches_a_worker_attributable_blowup_the_absolute_gate_missed():
    points = _points(Path("/job/aaaa/tmp/bench_reg/reg.manifest.json"), threads=8)
    baselines = _baselines_by_point_id(points, _DELTA_RAW_BASELINE)
    by_workers = {pt.reader_workers: pt for pt in points}

    # Quintuple the worker-attributable delta for reader_workers=7
    # (21.4 MB -> ~107 MB) while staying comfortably inside the OLD
    # 25%-of-absolute band (459.2 * 1.25 = 574.0 MB) -- proof this is a case
    # the pre-fix absolute gate would have passed.
    blown_up_w7_rss = 437.8 + 5 * (459.2 - 437.8)
    assert blown_up_w7_rss < 459.2 * 1.25

    records = [
        _rec_for(by_workers[1], wall=5.3, rss=437.8),
        _rec_for(by_workers[3], wall=5.3, rss=442.0),
        _rec_for(by_workers[7], wall=5.2, rss=blown_up_w7_rss),
    ]

    problems = check(records, points, baselines, tolerance=0.25)
    assert any("delta" in p and by_workers[7].point_id in p for p in problems), problems


def test_delta_gate_passes_a_run_that_matches_the_baseline_delta():
    points = _points(Path("/job/bbbb/tmp/bench_reg/reg.manifest.json"), threads=8)
    baselines = _baselines_by_point_id(points, _DELTA_RAW_BASELINE)
    by_workers = {pt.reader_workers: pt for pt in points}

    records = [
        _rec_for(by_workers[1], wall=5.3, rss=437.8),
        _rec_for(by_workers[3], wall=5.3, rss=442.0),
        _rec_for(by_workers[7], wall=5.2, rss=459.2),
    ]

    assert check(records, points, baselines, tolerance=0.25) == []


def test_delta_gate_does_not_fail_on_the_metrics_own_reproducibility():
    """I5: the two dedicated 8-CPU recordings of IDENTICAL code disagree by
    more than 25% of the smaller one -- 4.2/21.4 MB (job 13332630) versus the
    6.73/27.38 MB committed to baselines/regression.json. With a pure
    percentage band and the 4.2 MB recording as the baseline, the run that
    produced the committed file would have FAILED (6.73 > 4.2 * 1.25 = 5.25):
    a false positive on unchanged code. `DELTA_FLOOR_MB` is what stops that.
    """
    points = _points(Path("/job/eeee/tmp/bench_reg/reg.manifest.json"), threads=8)
    baselines = _baselines_by_point_id(points, _DELTA_RAW_BASELINE)
    by_workers = {pt.reader_workers: pt for pt in points}

    # Baseline deltas are 4.2 / 21.4; measure the other recording's 6.73 / 27.38.
    records = [
        _rec_for(by_workers[1], wall=5.3, rss=437.8),
        _rec_for(by_workers[3], wall=5.3, rss=437.8 + 6.73),
        _rec_for(by_workers[7], wall=5.2, rss=437.8 + 27.38),
    ]
    assert 6.73 > 4.2 * 1.25  # the pure percentage band this would have tripped
    assert 27.38 > 21.4 * 1.25
    assert check(records, points, baselines, tolerance=0.25) == []


def test_delta_floor_still_catches_a_real_regression():
    """The floor must not swallow the signal it sits under: the gate is there
    to catch a change that multiplies per-reader memory, which moves the w=7
    delta by tens of MB, not by the ~6 MB of recording noise."""
    points = _points(Path("/job/ffff/tmp/bench_reg/reg.manifest.json"), threads=8)
    baselines = _baselines_by_point_id(points, _DELTA_RAW_BASELINE)
    by_workers = {pt.reader_workers: pt for pt in points}

    doubled_w7 = 437.8 + 2 * 21.4
    records = [
        _rec_for(by_workers[1], wall=5.3, rss=437.8),
        _rec_for(by_workers[3], wall=5.3, rss=442.0),
        _rec_for(by_workers[7], wall=5.2, rss=doubled_w7),
    ]
    problems = check(records, points, baselines, tolerance=0.25)
    assert any("delta" in p and by_workers[7].point_id in p for p in problems), problems


def test_delta_gate_ignores_a_uniform_shift_in_the_fixed_footprint():
    # A shift in the fixed interpreter/extension footprint (e.g. a Python
    # point release) moves every point's absolute maxrss_mb by the same
    # amount but must not move the worker-attributable delta, since it
    # cancels out of both the measured and baseline subtraction.
    points = _points(Path("/job/cccc/tmp/bench_reg/reg.manifest.json"), threads=8)
    baselines = _baselines_by_point_id(points, _DELTA_RAW_BASELINE)
    by_workers = {pt.reader_workers: pt for pt in points}

    shift = 50.0
    records = [
        _rec_for(by_workers[1], wall=5.3, rss=437.8 + shift),
        _rec_for(by_workers[3], wall=5.3, rss=442.0 + shift),
        _rec_for(by_workers[7], wall=5.2, rss=459.2 + shift),
    ]

    assert check(records, points, baselines, tolerance=0.25) == []


def test_delta_gate_still_reports_a_failed_run_and_a_missing_baseline():
    points = _points(Path("/job/dddd/tmp/bench_reg/reg.manifest.json"), threads=8)
    baselines = _baselines_by_point_id(points, _DELTA_RAW_BASELINE)
    by_workers = {pt.reader_workers: pt for pt in points}

    ok_records = [
        _rec_for(by_workers[1], wall=5.3, rss=437.8),
        _rec_for(by_workers[3], wall=5.3, rss=442.0),
    ]
    bad = ProbeRecord(
        **{
            **_rec_for(by_workers[7], wall=5.2, rss=459.2).__dict__,
            "ok": False,
            "error": "boom",
        }
    )
    problems = check([*ok_records, bad], points, baselines, tolerance=0.25)
    assert any("run failed" in p for p in problems)

    missing_baseline_problems = check(ok_records, points, {}, tolerance=0.25)
    assert all("no baseline" in p for p in missing_baseline_problems)
    assert len(missing_baseline_problems) == len(ok_records)


def _manifest(tmp_path: Path, **overrides) -> Path:
    """A manifest plus the corpus file it describes, as `generate` leaves them."""
    vcf = tmp_path / "reg.vcf.gz"
    vcf.write_bytes(b"")
    fields = {
        "path": str(vcf),
        "samples": CORPUS["samples"],
        "variants": CORPUS["variants"],
        "contigs": tuple(CORPUS["contigs"]),
        "format_fields": (),
        "ploidy": 2,
        "cells": 1,
        "compressed_bytes": 1,
        "seed": CORPUS["seed"],
        "generator_version": GENERATOR_VERSION,
        **overrides,
    }
    path = tmp_path / "reg.manifest.json"
    path.write_text(to_json(CorpusManifest(**fields)))
    return path


def test_absent_corpus_is_not_current(tmp_path: Path):
    assert not corpus_is_current(tmp_path / "reg.manifest.json")


def test_matching_corpus_is_current(tmp_path: Path):
    assert corpus_is_current(_manifest(tmp_path))


def test_manifest_without_its_corpus_is_not_current(tmp_path: Path):
    # N4: the workdir defaults to scratch under $CLAUDE_JOB_DIR, which gets
    # cleaned. A manifest describing a corpus that is no longer on disk must
    # regenerate, not sail through and crash inside run_point.
    path = _manifest(tmp_path)
    (tmp_path / "reg.vcf.gz").unlink()
    assert not corpus_is_current(path)


def test_generator_version_bump_invalidates(tmp_path: Path):
    # N3: GENERATOR_VERSION exists to say "the generation logic changed, the
    # bytes differ". Ignoring it is the same vacuous pass as ignoring CORPUS.
    assert not corpus_is_current(
        _manifest(tmp_path, generator_version=GENERATOR_VERSION + 1)
    )


def test_floored_variant_count_still_counts_as_current(tmp_path: Path):
    # N5: generate() writes floor(variants / n_contigs) * n_contigs records, so
    # a manifest legitimately reports fewer variants than CORPUS requested.
    # Comparing against the unfloored request would regenerate the corpus on
    # every single invocation, forever, for any non-divisible CORPUS.
    contigs = list(CORPUS["contigs"])
    floored = (CORPUS["variants"] // len(contigs)) * len(contigs)
    assert corpus_is_current(_manifest(tmp_path, variants=floored))


@pytest.mark.parametrize(
    "override",
    [
        {"variants": CORPUS["variants"] * 10},
        {"samples": CORPUS["samples"] + 1},
        {"seed": CORPUS["seed"] + 1},
        {"contigs": ("chr1", "chr2")},
        {"format_fields": ("DP",)},
    ],
)
def test_stale_corpus_is_regenerated_not_silently_reused(tmp_path: Path, override):
    # The corpus is cached by filename, so shrinking CORPUS (as fix round 1
    # did, 20_000 -> 2_000 variants) leaves any existing workdir measuring the
    # OLD corpus. That matters because the hard gate is one-sided: baselines
    # recorded on the big corpus and checked against the small one are never
    # exceeded, so every point passes vacuously and the gate silently stops
    # gating. Each field that changes the generated bytes must invalidate.
    assert not corpus_is_current(_manifest(tmp_path, **override))


def test_baseline_keyed_by_workers_survives_a_workdir_change():
    # Fix round 1, Finding 1: SweepPoint.point_id hashes `corpus`, which is
    # derived from --workdir (defaulting to $CLAUDE_JOB_DIR/tmp/bench_reg).
    # CLAUDE_JOB_DIR is a per-session ephemeral path, so two sessions on the
    # exact same box/cores get DIFFERENT point_ids for the "same" point. A
    # baseline file keyed by point_id would be unusable outside the session
    # that recorded it. Simulate two different sessions (two different
    # manifest paths, same threads) and confirm a reader_workers-keyed raw
    # baseline dict maps onto both without loss.
    raw = {
        "1": {"wall_s": 10.0, "maxrss_mb": 100.0},
        "3": {"wall_s": 8.0, "maxrss_mb": 110.0},
        "7": {"wall_s": 6.0, "maxrss_mb": 120.0},
    }
    points_a = _points(Path("/job/aaaaaaaa/tmp/bench_reg/reg.manifest.json"), threads=8)
    points_b = _points(Path("/job/bbbbbbbb/tmp/bench_reg/reg.manifest.json"), threads=8)

    # The mechanism this finding is about: different session -> different
    # content hash, even though nothing about the code or hardware changed.
    ids_a = {pt.point_id for pt in points_a}
    ids_b = {pt.point_id for pt in points_b}
    assert ids_a.isdisjoint(ids_b)

    baselines_a = _baselines_by_point_id(points_a, raw)
    baselines_b = _baselines_by_point_id(points_b, raw)

    # Both sessions recover a baseline for every point despite the point_id
    # mismatch, and the underlying values are identical (re-keyed, not
    # re-derived).
    assert set(baselines_a) == ids_a
    assert set(baselines_b) == ids_b
    for pt, w in zip(points_a, WORKERS):
        assert baselines_a[pt.point_id] == raw[str(w)]
    for pt, w in zip(points_b, WORKERS):
        assert baselines_b[pt.point_id] == raw[str(w)]

    # And check() -- which only ever sees point_id-keyed baselines -- reports
    # no missing-baseline problems for either session once re-keyed.
    records_a = [
        _rec(pt.point_id, raw[str(w)]["wall_s"]) for pt, w in zip(points_a, WORKERS)
    ]
    problems = check(records_a, points_a, baselines_a, tolerance=0.25)
    assert not any("no baseline" in p for p in problems)


def test_pairing_follows_reader_workers_not_position():
    # N6: the pairing reads reader_workers off each point instead of zipping
    # against a parallel WORKERS sequence. Reordering the points must move the
    # baselines with them -- under the old positional zip, a reversed list
    # silently handed the w=7 point the w=1 baseline, and because the gate is
    # one-sided a mispaired baseline never fails, it just stops gating.
    raw = {
        "1": {"wall_s": 10.0, "maxrss_mb": 100.0},
        "3": {"wall_s": 8.0, "maxrss_mb": 110.0},
        "7": {"wall_s": 6.0, "maxrss_mb": 120.0},
    }
    points = _points(Path("/job/aaaa/tmp/bench_reg/reg.manifest.json"), threads=8)
    forward = _baselines_by_point_id(points, raw)
    reversed_ = _baselines_by_point_id(list(reversed(points)), raw)
    assert forward == reversed_
    for pt in points:
        assert forward[pt.point_id] == raw[str(pt.reader_workers)]


def test_unknown_worker_count_gets_no_baseline():
    # A point whose reader_workers is absent from the file must fall through
    # to check()'s "no baseline recorded" path, never to another point's row.
    points = _points(Path("/job/aaaa/tmp/bench_reg/reg.manifest.json"), threads=8)
    partial = _baselines_by_point_id(points, {"3": {"wall_s": 8.0, "maxrss_mb": 110.0}})
    assert len(partial) == 1
    records = [_rec(pt.point_id, 8.0) for pt in points]
    problems = check(records, points, partial, tolerance=0.25)
    assert sum("no baseline" in p for p in problems) == len(points) - 1

import pytest

from scripts.bench_svar2.records import (
    CorpusManifest,
    ProbeRecord,
    SweepPoint,
    Verdict,
    append_ndjson,
    from_json,
    read_ndjson,
    to_json,
)

pytestmark = pytest.mark.bench


def _manifest() -> CorpusManifest:
    return CorpusManifest(
        path="corpus.vcf.gz",
        samples=1000,
        variants=100_000,
        contigs=("chr22",),
        format_fields=(),
        ploidy=2,
        cells=100_000_000,
        compressed_bytes=30_000_000,
        seed=7,
        generator_version=1,
    )


def _record(point_id: str = "p0") -> ProbeRecord:
    return ProbeRecord(
        point_id=point_id,
        ok=True,
        wall_s=10.2,
        phase1_s=8.1,
        cpu_s=30.0,
        maxrss_mb=512.0,
        digest="abc123",
        dense_cap=6,
        dense_occupancy=(0, 1, 5),
        cpu_shard_pct=(100.0, 360.0),
        cpu_exec_pct=(60.0, 50.0),
        pending_highwater=3,
        pending_bytes_highwater=78_643_200,
        shard_unit_secs=(1.5, 2.5),
        oom_at_rss_mb=None,
        error=None,
    )


def test_manifest_round_trips():
    m = _manifest()
    assert from_json(CorpusManifest, to_json(m)) == m


def test_tuple_fields_survive_round_trip():
    """JSON has no tuples; the codec must restore them or equality breaks."""
    r = from_json(ProbeRecord, to_json(_record()))
    assert r.dense_occupancy == (0, 1, 5)
    assert isinstance(r.dense_occupancy, tuple)


def test_optional_fields_round_trip():
    r = _record()
    failed = ProbeRecord(
        **{**r.__dict__, "ok": False, "oom_at_rss_mb": 64_000.0, "error": "OOM"}
    )
    assert from_json(ProbeRecord, to_json(failed)).oom_at_rss_mb == 64_000.0


def test_ndjson_append_and_read(tmp_path):
    p = tmp_path / "results.ndjson"
    append_ndjson(p, _record("p0"))
    append_ndjson(p, _record("p1"))
    got = read_ndjson(p, ProbeRecord)
    assert [g.point_id for g in got] == ["p0", "p1"]


def test_read_ndjson_missing_file_is_empty(tmp_path):
    """Resumption reads before the first write; that must not raise."""
    assert read_ndjson(tmp_path / "nope.ndjson", ProbeRecord) == []


def test_sweep_point_id_is_deterministic():
    a = SweepPoint(
        corpus="c.json",
        reader_workers=3,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=4,
        chunk_size=25_000,
        threads=16,
        reps=3,
    )
    b = SweepPoint(
        corpus="c.json",
        reader_workers=3,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=4,
        chunk_size=25_000,
        threads=16,
        reps=3,
    )
    c = SweepPoint(
        corpus="c.json",
        reader_workers=5,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=4,
        chunk_size=25_000,
        threads=16,
        reps=3,
    )
    assert a.point_id == b.point_id
    assert a.point_id != c.point_id


# --- Verdict is print-only; its evidence does NOT survive persistence -------


def test_verdict_evidence_does_not_survive_a_json_round_trip():
    """Characterization, deliberately asserting the LOSSY behaviour.

    `Verdict` is print-only today: `model.main()` formats it to stdout and
    nothing calls `append_ndjson` with one. That is the only reason the
    codec's limits here are harmless, so pin them rather than leave the next
    author to discover them.

    `from_json` re-coerces tuples via `_tuple_fields`, which inspects
    top-level FIELD annotations. `evidence` is annotated `dict[str, Any]`, so
    its contents are opaque to that pass and every tuple inside it -- the
    beta CI, the named worst backlog row -- comes back a list. JSON object
    keys are additionally always strings, so any int-keyed dict put in
    evidence (e.g. `knees`, keyed by cohort size) returns string-keyed.

    Neither is fixable from here: `records.py` is frozen. Anyone adding
    verdict persistence must either annotate concretely or normalise the
    evidence dict on the way out -- and will fail this test when they do,
    which is the intent.
    """
    v = Verdict(
        hypothesis="H3",
        rationale="bytes, not worker count, set peak RSS",
        evidence={
            "beta_diff_ci95": (0.1, 0.2),
            "max_backlog_rss_share_row": (3, 5, 350_000_000, 6_000.0),
            "knee_points": 3,
            "knees": {250: 5, 1_000: 5},
        },
    )

    back = from_json(Verdict, to_json(v))

    assert back != v, "if this now round-trips, delete the workarounds below"
    assert back.hypothesis == v.hypothesis
    assert back.rationale == v.rationale
    # Tuples degrade to lists...
    assert back.evidence["beta_diff_ci95"] == [0.1, 0.2]
    assert back.evidence["max_backlog_rss_share_row"] == [3, 5, 350_000_000, 6_000.0]
    # ...int dict keys degrade to strings...
    assert back.evidence["knees"] == {"250": 5, "1000": 5}
    # ...while plain scalars are unaffected.
    assert back.evidence["knee_points"] == 3

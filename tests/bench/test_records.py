import pytest

from scripts.bench_svar2.records import (
    CorpusManifest,
    ProbeRecord,
    SweepPoint,
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

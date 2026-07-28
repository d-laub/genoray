import sys

import pytest

from scripts.bench_svar2.probe import parse_trace, run_point
from scripts.bench_svar2.records import CorpusManifest, SweepPoint

pytestmark = pytest.mark.bench

SAMPLE = """
2026-07-28 INFO done: 1000 kept, 0 excluded (8.15s)
TRACE genoray::monitor: pipeline sampler chrom=chr22 elapsed_s=1 dense=0 dense_cap=6 sparse=0 sparse_cap=4 long=0 long_cap=2 pending=0 pending_bytes=0 cpu_read=0% cpu_shard=100% cpu_exec=60% cpu_cw=5% cpu_lw=1%
TRACE genoray::monitor: pipeline sampler chrom=chr22 elapsed_s=2 dense=5 dense_cap=6 sparse=1 sparse_cap=4 long=0 long_cap=2 pending=3 pending_bytes=78643200 cpu_read=0% cpu_shard=360% cpu_exec=50% cpu_cw=5% cpu_lw=1%
TRACE genoray::monitor: shard unit done unit_ordinal=0 unit_secs=1.5
TRACE genoray::monitor: shard unit done unit_ordinal=1 unit_secs=2.5
2026-07-28 INFO done: 500 kept, 0 excluded (2.00s)
"""


def test_parses_phase1_as_sum_of_per_contig_spans():
    assert parse_trace(SAMPLE)["phase1_s"] == pytest.approx(10.15)


def test_parses_dense_occupancy_and_cap():
    t = parse_trace(SAMPLE)
    assert t["dense_occupancy"] == (0, 5)
    assert t["dense_cap"] == 6


def test_parses_cpu_percentages_stripping_the_sign():
    t = parse_trace(SAMPLE)
    assert t["cpu_shard_pct"] == (100.0, 360.0)
    assert t["cpu_exec_pct"] == (60.0, 50.0)


def test_parses_pending_highwater_as_max_not_last():
    t = parse_trace(SAMPLE)
    assert t["pending_highwater"] == 3
    assert t["pending_bytes_highwater"] == 78_643_200


def test_parses_shard_unit_times():
    assert parse_trace(SAMPLE)["shard_unit_secs"] == (1.5, 2.5)


def test_handles_na_cpu_columns():
    """cpu_shard reads n/a on the single-reader fallback path."""
    line = (
        "TRACE genoray::monitor: pipeline sampler chrom=chr1 elapsed_s=1 dense=0 "
        "dense_cap=6 sparse=0 sparse_cap=4 long=0 long_cap=2 pending=0 pending_bytes=0 "
        "cpu_read=0% cpu_shard=n/a cpu_exec=10% cpu_cw=0% cpu_lw=0%"
    )
    t = parse_trace(line)
    assert t["cpu_shard_pct"] == ()
    assert t["cpu_exec_pct"] == (10.0,)


def test_empty_input_yields_zeroed_trace():
    t = parse_trace("")
    assert t["phase1_s"] == 0.0
    assert t["pending_highwater"] == 0
    assert t["dense_occupancy"] == ()


def _fake_build_cmd(point, manifest, store):
    """Stand-in for the real `genoray._cli write vcf` invocation: creates the
    store directory (so `digest(store)` has something to hash) and emits well
    over a pipe's ~64 KiB kernel buffer on both stdout and stderr."""
    script = (
        "import sys, pathlib\n"
        "store = pathlib.Path(sys.argv[1])\n"
        "store.mkdir(parents=True, exist_ok=True)\n"
        "(store / 'part.bin').write_bytes(b'x')\n"
        "sys.stdout.write('o' * 200_000 + chr(10))\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('e' * 200_000 + chr(10))\n"
        "sys.stderr.flush()\n"
    )
    return [sys.executable, "-c", script, str(store)]


def test_run_point_survives_a_chatty_child_beyond_the_pipe_buffer(
    tmp_path, monkeypatch
):
    """Fix P4 regression test.

    The brief's `run_point` reads `proc.stdout`/`proc.stderr` only *after*
    `os.wait4` returns. A pipe's kernel buffer is ~64 KiB; a child that
    writes more than that before the parent drains it blocks in `write(2)`
    forever, and the parent blocks in `wait4` forever right along with it --
    deadlock. `GENORAY_LOG=genoray::monitor=trace` with
    `GENORAY_SAMPLE_INTERVAL=1` makes a multi-minute real conversion do
    exactly this. Reproduce the shape of the bug cheaply with a fake command
    that emits 200_000 bytes (well over 64 KiB) on each of stdout and
    stderr, standing in for the real `genoray._cli` invocation `run_point`
    would otherwise build via `_build_cmd`.

    This test has no manual timeout wrapper -- it relies on pytest's own
    run finishing in reasonable time. Under the pre-fix (PIPE-based)
    implementation this call hangs indefinitely.
    """
    import scripts.bench_svar2.probe as probe_mod

    monkeypatch.setattr(probe_mod, "_build_cmd", _fake_build_cmd)

    outdir = tmp_path / "out"
    manifest = CorpusManifest(
        path="unused.vcf.gz",
        samples=1,
        variants=1,
        contigs=("chr22",),
        format_fields=(),
        ploidy=2,
        cells=2,
        compressed_bytes=1,
        seed=1,
        generator_version=1,
    )
    point = SweepPoint(
        corpus="unused",
        reader_workers=1,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=1,
        chunk_size=64,
        threads=1,
        reps=1,
    )

    rec = run_point(point, manifest, outdir, warm=False)

    assert rec.ok is True
    assert rec.digest != ""

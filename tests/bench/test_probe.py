import os
import signal
import sys
from pathlib import Path

import pytest

from scripts.bench_svar2.probe import (
    _build_env,
    _is_oom_failure,
    parse_trace,
    run_point,
)
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


def _sampler_line(elapsed: int, shard: str, execp: str) -> str:
    return (
        f"TRACE genoray::monitor: pipeline sampler chrom=chr1 elapsed_s={elapsed} "
        "dense=0 dense_cap=6 sparse=0 sparse_cap=4 long=0 long_cap=2 pending=0 "
        f"pending_bytes=0 cpu_read=0% cpu_shard={shard} cpu_exec={execp}% "
        "cpu_cw=0% cpu_lw=0%"
    )


def test_handles_na_cpu_columns():
    """cpu_shard reads n/a on the single-reader fallback path. The tick is
    dropped from BOTH series, not just the one that was n/a."""
    t = parse_trace(_sampler_line(1, "n/a", "10"))
    assert t["cpu_shard_pct"] == ()
    assert t["cpu_exec_pct"] == ()


def test_na_in_one_cpu_column_does_not_misalign_the_other_series():
    """Minor 10: `model._median_costs` ZIPS cpu_shard_pct against
    cpu_exec_pct, so these tuples are only meaningful index-aligned.

    Appending each column to its own list independently means a single `n/a`
    in one column shifts every LATER sample of the other column against the
    wrong tick's value -- and the corruption is silent, because both tuples
    still look well-formed. Here the middle tick has no cpu_shard: under the
    old per-column append the pairs would come out (100, 10) and (200, 30),
    inventing a 200/30 tick that never happened and losing the real 200/40
    one. Correct behaviour drops the middle tick entirely.
    """
    text = "\n".join(
        [
            _sampler_line(1, "100%", "10"),
            _sampler_line(2, "n/a", "30"),
            _sampler_line(3, "200%", "40"),
        ]
    )
    t = parse_trace(text)
    assert t["cpu_shard_pct"] == (100.0, 200.0)
    assert t["cpu_exec_pct"] == (10.0, 40.0)
    assert len(t["cpu_shard_pct"]) == len(t["cpu_exec_pct"])


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


# --- Finding I6, bug 1: OOM attribution rule ---------------------------------


def test_is_oom_failure_detects_python_memory_error():
    assert _is_oom_failure(
        status=1,
        err="Traceback (most recent call last):\nMemoryError\n",
        maxrss_mb=100.0,
        ceiling_mb=60_000,
    )


def test_is_oom_failure_detects_rust_allocator_abort():
    # Rust's global-allocator error handler (`std::alloc::handle_alloc_error`)
    # prints this exact message before aborting.
    assert _is_oom_failure(
        status=134,
        err="memory allocation of 4096 bytes failed\n",
        maxrss_mb=100.0,
        ceiling_mb=60_000,
    )


def test_is_oom_failure_detects_sigkill_when_the_process_had_actually_grown():
    # POSIX wait-status encoding: a raw status whose low 7 bits equal the
    # signal number is exactly what os.wait4 returns for a signal-terminated
    # child. SIGKILL is how the Linux OOM killer terminates a process, but it
    # only counts as OOM here with corroborating RSS -- see the rejection
    # test below.
    status = signal.SIGKILL
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL
    assert _is_oom_failure(status=status, err="", maxrss_mb=40_000.0, ceiling_mb=60_000)


def test_is_oom_failure_rejects_sigkill_on_a_small_process():
    """Minor 9: the docstring's own list of cases to exclude names "a
    preemption signal", but an unconditional SIGKILL branch re-admitted
    exactly that.

    Slurm ends a preempted or time-limited job with SIGKILL, and so does an
    operator killing a run by hand. Under this harness's configuration the
    bare signal is near-certainly NOT memory exhaustion: `rss_ceiling_mb` is
    installed as RLIMIT_AS (60 GB) while the sweep's cgroup allows 120 GB, so
    a genuine exhaustion trips RLIMIT_AS first and dies via SIGABRT with the
    allocator message. A 100 MB process killed at a 60 GB ceiling is a job
    that got preempted, and recording it as `oom_at_rss_mb` would fabricate
    the harness's headline "OOMs at scale" finding out of cluster scheduling.
    """
    assert not _is_oom_failure(
        status=signal.SIGKILL, err="", maxrss_mb=100.0, ceiling_mb=60_000
    )


def test_is_oom_failure_detects_near_ceiling_rss_even_without_a_signature():
    assert _is_oom_failure(status=1, err="", maxrss_mb=55_000.0, ceiling_mb=60_000)


def test_is_oom_failure_rejects_a_plain_nonzero_exit_far_below_ceiling():
    """Finding I6, bug 1's core case: a bad --chunk-size, missing corpus, or
    tabix error is a plain nonzero exit with an ordinary message, nowhere
    near the ceiling and not signal-killed. Must not be attributed to OOM."""
    assert not _is_oom_failure(
        status=1,
        err="error: invalid value for '--chunk-size'\n",
        maxrss_mb=100.0,
        ceiling_mb=60_000,
    )


def _fake_build_cmd_benign_failure(point, manifest, store):
    """A plain nonzero exit with an ordinary error message -- stands in for
    a bad --chunk-size, missing corpus, or tabix error, none of which are
    memory exhaustion."""
    script = (
        "import sys\n"
        "sys.stderr.write(\"error: invalid value for '--chunk-size'\\n\")\n"
        "sys.exit(2)\n"
    )
    return [sys.executable, "-c", script]


def _fake_build_cmd_allocator_failure(point, manifest, store):
    """A failure carrying Rust's global-allocator abort message -- the
    OOM-shaped signature `_is_oom_failure` looks for."""
    script = (
        "import sys\n"
        "sys.stderr.write('memory allocation of 4096 bytes failed\\n')\n"
        "sys.exit(134)\n"
    )
    return [sys.executable, "-c", script]


def _probe_fixtures(tmp_path):
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
        rss_ceiling_mb=60_000,
    )
    return outdir, manifest, point


def test_run_point_does_not_record_oom_for_a_non_memory_failure(tmp_path, monkeypatch):
    """Finding I6, bug 1, end-to-end through `run_point`: a plain CLI/config
    failure at a configured rss_ceiling_mb must not manufacture an OOM
    datum."""
    import scripts.bench_svar2.probe as probe_mod

    monkeypatch.setattr(probe_mod, "_build_cmd", _fake_build_cmd_benign_failure)
    outdir, manifest, point = _probe_fixtures(tmp_path)

    rec = run_point(point, manifest, outdir, warm=False)

    assert rec.ok is False
    assert rec.oom_at_rss_mb is None
    assert "chunk-size" in rec.error


def test_run_point_records_oom_for_an_allocator_failure_signature(
    tmp_path, monkeypatch
):
    """Mirror of the above: a failure that genuinely looks like memory
    exhaustion at a configured ceiling IS recorded as an OOM datum."""
    import scripts.bench_svar2.probe as probe_mod

    monkeypatch.setattr(probe_mod, "_build_cmd", _fake_build_cmd_allocator_failure)
    outdir, manifest, point = _probe_fixtures(tmp_path)

    rec = run_point(point, manifest, outdir, warm=False)

    assert rec.ok is False
    assert rec.oom_at_rss_mb is not None


# --- Finding I6, bug 2: RLIMIT_AS vs RSS -------------------------------------


def test_build_env_pins_single_arena_when_ceiling_configured():
    point = SweepPoint(
        corpus="unused",
        reader_workers=1,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=1,
        chunk_size=64,
        threads=1,
        reps=1,
        rss_ceiling_mb=60_000,
    )
    assert _build_env(point)["MALLOC_ARENA_MAX"] == "1"


def test_build_env_leaves_arena_count_alone_without_a_ceiling():
    point = SweepPoint(
        corpus="unused",
        reader_workers=1,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=1,
        chunk_size=64,
        threads=1,
        reps=1,
        rss_ceiling_mb=None,
    )
    assert "MALLOC_ARENA_MAX" not in _build_env(point)


def test_build_cmd_dispatches_on_backend():
    from scripts.bench_svar2.probe import _build_cmd
    from scripts.bench_svar2.records import CorpusManifest, SweepPoint

    manifest = CorpusManifest(
        path="/tmp/corpus.pgen",
        samples=10,
        variants=100,
        contigs=("chr1",),
        format_fields=(),
        ploidy=2,
        cells=1000,
        compressed_bytes=1,
        seed=0,
        generator_version=1,
    )
    point = SweepPoint(
        corpus=manifest.path,
        reader_workers=1,
        concurrent_chroms=4,
        shard_htslib=0,
        overshard=4,
        chunk_size=1000,
        threads=8,
        reps=1,
        backend="pgen",
    )
    cmd = _build_cmd(point, manifest, Path("/tmp/store.svar"))
    assert "pgen" in cmd
    assert "vcf" not in cmd
    # Symbolic ALTs survive plink2 into the .pvar, so the PGEN arm must skip
    # them or every conversion aborts on the first <DEL>.
    assert "--skip-symbolics-and-breakends" in cmd


def test_build_cmd_defaults_to_vcf():
    from scripts.bench_svar2.probe import _build_cmd
    from scripts.bench_svar2.records import CorpusManifest, SweepPoint

    manifest = CorpusManifest(
        path="/tmp/corpus.vcf.gz",
        samples=10,
        variants=100,
        contigs=("chr1",),
        format_fields=(),
        ploidy=2,
        cells=1000,
        compressed_bytes=1,
        seed=0,
        generator_version=1,
    )
    point = SweepPoint(
        corpus=manifest.path,
        reader_workers=1,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=4,
        chunk_size=1000,
        threads=8,
        reps=1,
    )
    cmd = _build_cmd(point, manifest, Path("/tmp/s.svar"))
    assert "vcf" in cmd
    assert "--skip-symbolics-and-breakends" not in cmd

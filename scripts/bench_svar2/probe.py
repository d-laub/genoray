"""Run exactly one instrumented conversion and return one ProbeRecord.

The only module in the harness that shells out. Everything it learns comes from
two places: `os.wait4` rusage, and the `genoray::monitor` trace stream.
"""

from __future__ import annotations

import hashlib
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.bench_svar2.records import CorpusManifest, ProbeRecord, SweepPoint

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# "done: 1000 kept, 0 excluded (8.15s)" -- the per-contig phase-1 span. This is
# the ONLY span reader_workers can move; the rayon merge tail that follows is
# reader-independent, so total wall understates the reader-side effect.
RE_PHASE1 = re.compile(r"done:.*?\(([0-9.]+)s\)")
RE_SAMPLER = re.compile(r"pipeline sampler .*")
RE_UNIT = re.compile(r"shard unit done .*?unit_secs=([0-9.]+)")


def _field(line: str, key: str) -> str | None:
    m = re.search(rf"\b{key}=([^\s]+)", line)
    return m.group(1) if m else None


def parse_trace(text: str) -> dict:
    plain = ANSI.sub("", text)
    phase1 = sum(float(x) for x in RE_PHASE1.findall(plain))

    dense, shard, execp = [], [], []
    dense_cap = 0
    pending_hw = 0
    pending_bytes_hw = 0
    for line in RE_SAMPLER.findall(plain):
        d = _field(line, "dense")
        if d is not None:
            dense.append(int(d))
        c = _field(line, "dense_cap")
        if c is not None:
            dense_cap = max(dense_cap, int(c))
        p = _field(line, "pending")
        if p is not None:
            pending_hw = max(pending_hw, int(p))
        pb = _field(line, "pending_bytes")
        if pb is not None:
            pending_bytes_hw = max(pending_bytes_hw, int(pb))
        for key, sink in (("cpu_shard", shard), ("cpu_exec", execp)):
            v = _field(line, key)
            # `n/a` on the single-reader fallback path -- skip, do not zero,
            # or the median in `knee_from_probe` gets dragged down.
            if v is not None and v != "n/a":
                sink.append(float(v.rstrip("%")))

    return {
        "phase1_s": phase1,
        "dense_occupancy": tuple(dense),
        "dense_cap": dense_cap,
        "cpu_shard_pct": tuple(shard),
        "cpu_exec_pct": tuple(execp),
        "pending_highwater": pending_hw,
        "pending_bytes_highwater": pending_bytes_hw,
        "shard_unit_secs": tuple(float(x) for x in RE_UNIT.findall(plain)),
    }


def digest(store: Path) -> str:
    """Order-independent hash of every file in the .svar store -- the
    correctness oracle. Sharding is byte-identical, so this must not move
    across any configuration."""
    h = hashlib.sha256()
    for p in sorted(store.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(store).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _preexec(rss_ceiling_mb: int | None):
    if rss_ceiling_mb is None:
        return None

    def _limit() -> None:
        cap = rss_ceiling_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    return _limit


def _build_env(point: SweepPoint) -> dict[str, str]:
    env = dict(os.environ) | {
        "GENORAY_READER_WORKERS": str(point.reader_workers),
        "GENORAY_SHARD_HTSLIB": str(point.shard_htslib),
        "GENORAY_OVERSHARD": str(point.overshard),
        "GENORAY_LOG": "genoray::monitor=trace",
        "GENORAY_SAMPLE_INTERVAL": "1",
    }
    if point.concurrent_chroms is not None:
        env["GENORAY_CONCURRENT_CHROMS"] = str(point.concurrent_chroms)
    return env


def _build_cmd(point: SweepPoint, manifest: CorpusManifest, store: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "genoray._cli",
        "write",
        "vcf",
        manifest.path,
        str(store),
        "--no-reference",
        "--log-level",
        "info",
        "--overwrite",
        "-@",
        str(point.threads),
        "--chunk-size",
        str(point.chunk_size),
    ]


def _tmp_dir(outdir: Path) -> Path:
    """Scratch directory for the per-rep stdout/stderr capture files.

    Never `/tmp`: it is reaped mid-session on this cluster (see
    genoray-nfs-linker-bus-error memory). Prefer `$CLAUDE_JOB_DIR/tmp`; fall
    back to the probe's own `outdir` if that env var is unset (e.g. outside
    the Claude harness)."""
    job_dir = os.environ.get("CLAUDE_JOB_DIR")
    base = Path(job_dir) / "tmp" / "bench_probe" if job_dir else outdir / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _run_child(
    cmd: list[str],
    env: dict[str, str],
    rss_ceiling_mb: int | None,
    tmp_dir: Path,
) -> tuple[int, "resource.struct_rusage", str, str]:
    """Run one child to completion; return (exit_status, rusage, stdout, stderr).

    Fix P4: the child's stdout/stderr are redirected to FILES, not
    `subprocess.PIPE`. `run_point` sets `GENORAY_LOG=genoray::monitor=trace`
    with `GENORAY_SAMPLE_INTERVAL=1`, so a multi-minute real conversion emits
    far more than a pipe's ~64 KiB kernel buffer. A `subprocess.PIPE` is
    never drained until after `os.wait4` returns, so the child blocks in
    `write(2)`, the parent blocks in `wait4`, and the run deadlocks forever.
    Files have no such limit, and `os.wait4` is kept (not `subprocess.run`)
    because the probe needs its rusage for `maxrss_mb` and `cpu_s`.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_fd, out_name = tempfile.mkstemp(dir=tmp_dir, prefix="probe-stdout-")
    err_fd, err_name = tempfile.mkstemp(dir=tmp_dir, prefix="probe-stderr-")
    out_path, err_path = Path(out_name), Path(err_name)
    try:
        with os.fdopen(out_fd, "wb") as out_f, os.fdopen(err_fd, "wb") as err_f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=out_f,
                stderr=err_f,
                preexec_fn=_preexec(rss_ceiling_mb),
            )
            _, status, ru = os.wait4(proc.pid, 0)
        out = out_path.read_bytes().decode(errors="replace")
        err = err_path.read_bytes().decode(errors="replace")
    finally:
        out_path.unlink(missing_ok=True)
        err_path.unlink(missing_ok=True)
    return status, ru, out, err


def run_point(
    point: SweepPoint, manifest: CorpusManifest, outdir: Path, warm: bool = True
) -> ProbeRecord:
    store = outdir / "bench.svar"
    outdir.mkdir(parents=True, exist_ok=True)
    env = _build_env(point)
    cmd = _build_cmd(point, manifest, store)
    tmp_dir = _tmp_dir(outdir)

    best: ProbeRecord | None = None
    for rep in range(point.reps + (1 if warm else 0)):
        if store.exists():
            shutil.rmtree(store)
        t0 = time.perf_counter()
        status, ru, out, err = _run_child(cmd, env, point.rss_ceiling_mb, tmp_dir)
        wall = time.perf_counter() - t0
        if warm and rep == 0:
            continue  # page-cache warm-up; measure inflate+parse CPU, not first-touch IO

        maxrss_mb = ru.ru_maxrss / 1024.0
        if status != 0:
            oom = maxrss_mb if point.rss_ceiling_mb else None
            # An OOM at a known ceiling is a legitimate datum: proving the
            # current chunk_size cannot survive biobank scale is a deliverable.
            return ProbeRecord(
                point_id=point.point_id,
                ok=False,
                wall_s=wall,
                phase1_s=0.0,
                cpu_s=ru.ru_utime + ru.ru_stime,
                maxrss_mb=maxrss_mb,
                digest="",
                dense_cap=0,
                dense_occupancy=(),
                cpu_shard_pct=(),
                cpu_exec_pct=(),
                pending_highwater=0,
                pending_bytes_highwater=0,
                shard_unit_secs=(),
                oom_at_rss_mb=oom,
                error=err[-2000:],
            )

        t = parse_trace(out + err)
        rec = ProbeRecord(
            point_id=point.point_id,
            ok=True,
            wall_s=wall,
            phase1_s=t["phase1_s"],
            cpu_s=ru.ru_utime + ru.ru_stime,
            maxrss_mb=maxrss_mb,
            digest=digest(store),
            dense_cap=t["dense_cap"],
            dense_occupancy=t["dense_occupancy"],
            cpu_shard_pct=t["cpu_shard_pct"],
            cpu_exec_pct=t["cpu_exec_pct"],
            pending_highwater=t["pending_highwater"],
            pending_bytes_highwater=t["pending_bytes_highwater"],
            shard_unit_secs=t["shard_unit_secs"],
        )
        # Min-of-N on wall time; the cluster is shared, so the minimum is the
        # least contended estimate.
        if best is None or rec.wall_s < best.wall_s:
            best = rec

    assert best is not None
    return best

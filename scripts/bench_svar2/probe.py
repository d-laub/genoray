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
import signal
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
        # `n/a` on the single-reader fallback path -- skip, do not zero, or the
        # median in `knee_from_probe` gets dragged down. Append only as a
        # PAIR: `model._median_costs` ZIPS these two tuples, so appending to
        # one list without the other does not merely lose a sample, it shifts
        # every later `cpu_exec` sample against a DIFFERENT line's `cpu_shard`
        # sample -- silently corrupting `c_read / c_exec` and the knee derived
        # from it. Both fields are emitted on the same `pipeline sampler` line
        # (src/monitor.rs), and nothing downstream reads either series alone,
        # so a tick missing one is unusable rather than half-usable.
        sv, ev = _field(line, "cpu_shard"), _field(line, "cpu_exec")
        if sv is not None and ev is not None and sv != "n/a" and ev != "n/a":
            shard.append(float(sv.rstrip("%")))
            execp.append(float(ev.rstrip("%")))

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
    """Return a preexec_fn that installs `rss_ceiling_mb` as RLIMIT_AS.

    SEMANTICS (Finding I6, bug 2): RLIMIT_AS caps virtual ADDRESS SPACE, not
    resident memory. There is no portable, unprivileged way to cap RSS
    directly on Linux -- `RLIMIT_RSS` is accepted by `setrlimit` but has been
    unenforced by the kernel since 2.4.30, and a real RSS cap needs a cgroup,
    which a `Popen(preexec_fn=...)` launcher cannot set up for an
    unprivileged child without root or a pre-delegated cgroup path (both out
    of scope for a benchmark harness). `records.py` is frozen, so
    `rss_ceiling_mb` cannot be renamed to say "address space" either -- the
    field name and this docstring are the only place that fact can live.

    This matters because glibc's default multi-arena allocator reserves up
    to `8 * ncores` arenas, and each arena's VA footprint counts against
    RLIMIT_AS even before anything in it is touched. On a wide node (e.g. 48
    cores -> up to 384 arenas) a run genuinely using single-digit GB of RSS
    can trip a 60 GB nominal "ceiling" purely on unused arena headroom --
    fabricating an OOM datum in the OPPOSITE direction from the usual worry
    (a run that never came close to the real limit gets recorded as though
    it did, which would corrupt the same headline finding this harness
    exists to produce).

    Mitigation, not a full fix: `_build_env` sets `MALLOC_ARENA_MAX=1`
    whenever a ceiling is configured, pinning glibc to a single arena and
    closing almost all of that gap (the remaining slack is large individual
    mmap'd allocations, which correlate with real usage rather than thread
    count). `rss_ceiling_mb` still bounds address space, not RSS, by
    construction -- this narrows the two until they track each other
    closely, it does not make them identical. `run_point`'s
    `_is_oom_failure` further guards against what's left of the gap by
    requiring an allocation-failure signature or `maxrss_mb` actually being
    close to the ceiling before recording `oom_at_rss_mb`, rather than
    trusting the exit alone.
    """
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
    if point.rss_ceiling_mb is not None:
        # See `_preexec`: RLIMIT_AS bounds virtual address space, and
        # glibc's default per-thread arena allocator can reserve VA far
        # beyond what the process ever touches. Pin to a single arena so the
        # RLIMIT_AS ceiling tracks actual RSS closely enough to be a
        # meaningful proxy for it.
        env["MALLOC_ARENA_MAX"] = "1"
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


_OOM_STDERR_RE = re.compile(
    r"MemoryError|memory allocation of \d+ bytes failed|cannot allocate memory",
    re.IGNORECASE,
)


def _is_oom_failure(status: int, err: str, maxrss_mb: float, ceiling_mb: int) -> bool:
    """Whether a failed child's exit looks like genuine memory exhaustion.

    Finding I6, bug 1: every nonzero exit used to be recorded as
    `oom_at_rss_mb` whenever a ceiling was configured, so a bad
    `--chunk-size`, a missing corpus, a tabix error, or a preemption signal
    would fabricate an "OOMs at scale" datum -- a headline finding downstream
    -- out of an unrelated bug. Require one of three OOM-shaped signals
    before attributing the failure to memory:

    - stderr carries a known allocation-failure message: Python's
      `MemoryError`, Rust's global allocator abort message ("memory
      allocation of N bytes failed" -- what `std::alloc::handle_alloc_error`
      prints before aborting), or the OS's ENOMEM text ("Cannot allocate
      memory").
    - the child was killed by SIGKILL *and* had grown to at least half the
      ceiling. SIGKILL is how the Linux OOM killer terminates a process, but
      it is also how Slurm ends a preempted or time-limited job and how an
      operator kills a run by hand, so on its own it does not distinguish
      memory exhaustion from the very preemption case this function exists
      to exclude. Under this harness's own configuration the bare signal is
      in fact near-certainly NOT an OOM: `rss_ceiling_mb` is installed as
      RLIMIT_AS (60 GB by default, see `OOM_PROBE_CEILING_MB`) while the
      sweep's Slurm cgroup allows 120 GB, so a run that genuinely exhausts
      memory trips RLIMIT_AS first and dies via `malloc` returning NULL ->
      `handle_alloc_error` -> SIGABRT with the message the regex above
      catches. The cgroup OOM killer cannot fire until twice the ceiling,
      which RLIMIT_AS makes unreachable. Requiring corroborating RSS keeps
      the branch meaningful if that relationship is ever reconfigured
      (ceiling above the cgroup limit) without minting OOM data from job
      preemption in the meantime.
    - `maxrss_mb` is within 10% of the configured ceiling -- the process was
      genuinely close to the limit when it died even if neither signature
      above fired (e.g. it was killed by something other than SIGKILL, or
      wrote nothing recognizable to stderr).

    Anything else -- a plain nonzero exit with an ordinary error message,
    far below the ceiling, not signal-killed -- leaves `oom_at_rss_mb` unset
    and lets `ProbeRecord.error` carry the real cause.
    """
    if _OOM_STDERR_RE.search(err):
        return True
    if os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL:
        return maxrss_mb >= 0.5 * ceiling_mb
    return maxrss_mb >= 0.9 * ceiling_mb


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
            # A genuine OOM at a known ceiling is a legitimate datum: proving
            # the current chunk_size cannot survive biobank scale is a
            # deliverable. But only when the failure actually looks like
            # memory exhaustion (Finding I6, bug 1) -- see
            # `_is_oom_failure` for the rule and its rationale. Everything
            # else (bad args, missing corpus, tabix errors, ...) leaves
            # `oom_at_rss_mb` unset and surfaces via `error` instead.
            oom = (
                maxrss_mb
                if point.rss_ceiling_mb
                and _is_oom_failure(status, err, maxrss_mb, point.rss_ceiling_mb)
                else None
            )
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

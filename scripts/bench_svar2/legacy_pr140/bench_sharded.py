"""Sweep the SVAR2 sharded-VCF reader budget: wall time, CPU%, and a byte-identity
oracle across (reader_workers, per-shard HTSlib threads, overshard factor).

Requires a build carrying the BENCH-ONLY env hooks in orchestrator.rs
(GENORAY_READER_WORKERS / GENORAY_SHARD_HTSLIB / GENORAY_OVERSHARD).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def digest(out: Path) -> str:
    """Order-independent hash of every file in the .svar store — the correctness
    oracle. Sharding is documented as byte-identical, so this must not move."""
    h = hashlib.sha256()
    for p in sorted(out.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(out).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_once(src: Path, out: Path, env_over: dict[str, str], threads: int) -> dict:
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ) | env_over
    cmd = [
        sys.executable,
        "-m",
        "genoray._cli",
        "write",
        "vcf",
        str(src),
        str(out),
        "--no-reference",
        "--log-level",
        "info",
        "--overwrite",
        "-@",
        str(threads),
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    pid, status, ru = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t0
    out_txt = proc.stdout.read().decode() if proc.stdout else ""
    err = proc.stderr.read().decode()[-2000:] if proc.stderr else ""
    if status != 0:
        raise RuntimeError(f"conversion failed ({status}) cfg={env_over}\n{err}")
    # Sum the per-contig "done: N kept, M excluded (X.Ys)" phase-1 times. This is
    # the ONLY span reader_workers can move; the rayon merge tail that follows is
    # reader-independent, so total wall understates the reader-side effect.
    plain = ANSI.sub("", out_txt + err)
    p1 = sum(float(x) for x in re.findall(r"done:.*?\(([0-9.]+)s\)", plain))
    cpu = ru.ru_utime + ru.ru_stime
    return {
        "wall_s": wall,
        "phase1_s": p1,
        "cpu_s": cpu,
        "cpu_pct": 100.0 * cpu / wall,
        "maxrss_mb": ru.ru_maxrss / 1024.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=str, required=True, help="comma list")
    p.add_argument("--htslib", type=str, default="0", help="comma list")
    p.add_argument("--overshard", type=str, default="4", help="comma list")
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--label", type=str, default="")
    a = p.parse_args()

    out = a.outdir / "bench.svar"
    a.outdir.mkdir(parents=True, exist_ok=True)

    # Warm the page cache so we measure inflate+parse CPU, not first-touch disk IO.
    run_once(a.src, out, {"GENORAY_READER_WORKERS": "2"}, a.threads)

    results = []
    oracle = None
    grid = [
        (w, h, o)
        for o in [int(x) for x in a.overshard.split(",")]
        for h in [int(x) for x in a.htslib.split(",")]
        for w in [int(x) for x in a.workers.split(",")]
    ]
    for w, h, o in grid:
        env = {
            "GENORAY_READER_WORKERS": str(w),
            "GENORAY_SHARD_HTSLIB": str(h),
            "GENORAY_OVERSHARD": str(o),
        }
        reps = []
        for _ in range(a.reps):
            reps.append(run_once(a.src, out, env, a.threads))
        d = digest(out)
        if oracle is None:
            oracle = d
        ok = d == oracle
        row = {
            "label": a.label,
            "workers": w,
            "htslib": h,
            "overshard": o,
            "wall_s_min": min(r["wall_s"] for r in reps),
            "phase1_s_min": min(r["phase1_s"] for r in reps),
            "wall_s_med": statistics.median(r["wall_s"] for r in reps),
            "cpu_s_med": statistics.median(r["cpu_s"] for r in reps),
            "cpu_pct_med": statistics.median(r["cpu_pct"] for r in reps),
            "maxrss_mb": max(r["maxrss_mb"] for r in reps),
            "digest": d,
            "oracle_ok": ok,
        }
        results.append(row)
        print(
            f"w={w:>3} hts={h} os={o:>2} | wall {row['wall_s_min']:7.2f}s "
            f"| phase1 {row['phase1_s_min']:6.2f}s | cpu {row['cpu_pct_med']:6.0f}% "
            f"| rss {row['maxrss_mb']:7.0f}MB | oracle {'OK' if ok else 'MISMATCH'}",
            flush=True,
        )
    a.json_out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Compare two BUILDS of genoray end-to-end on the same corpus.

The scale sweep (`sweep.py`) characterizes one build across a plan of
configurations. This answers the other question -- "what does merging this
branch actually buy a user?" -- and so it deliberately does the opposite:
one configuration, two builds, **no `GENORAY_*` overrides at all**. Every
sweep hook (`GENORAY_READER_WORKERS`, `GENORAY_CONCURRENT_CHROMS`, ...) is
stripped from the child's environment, because a number measured with the
planner overridden is not the number a user gets.

A build is identified by its interpreter: each arm is a separate checkout with
its own environment and its own `maturin develop --release`, and `--python`
selects which one runs the conversion. Nothing else differs between arms --
same corpus, same `--chunk-size`, same `-@`, same node, interleaved in time.

Interleaving matters more than it looks: this cluster's nodes vary by 2.08x
and are shared, so running all of arm A then all of arm B attributes any drift
in machine load to the code change. `--reps` alternates arms within each shape.

Usage:

    python -m scripts.bench_svar2.ab_builds \\
        --corpus corpus.manifest.json --chunk-size 683 --threads 48 \\
        --arm base=/path/to/base/.pixi/envs/default/bin/python \\
        --arm head=/path/to/head/.pixi/envs/default/bin/python \\
        --out results.ndjson
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Sweep hooks. Cleared for every child: this harness measures DEFAULTS, and an
# inherited override from the surrounding shell would silently become part of
# the result.
BENCH_ENV_VARS = (
    "GENORAY_READER_WORKERS",
    "GENORAY_EXEC_WORKERS",
    "GENORAY_OVERSHARD",
    "GENORAY_SHARD_HTSLIB",
    "GENORAY_CONCURRENT_CHROMS",
    "GENORAY_DENSE_CAP",
    "GENORAY_MERGE_THREADS",
    "GENORAY_SAMPLE_INTERVAL",
    "GENORAY_LOG",
)


@dataclass(frozen=True)
class ArmResult:
    arm: str
    corpus: str
    samples: int
    variants: int
    contigs: int
    chunk_size: int
    threads: int
    rep: int
    ok: bool
    wall_s: float
    cpu_s: float
    maxrss_mb: float
    digest: str
    node: str
    error: str = ""

    @property
    def cores(self) -> float:
        return self.cpu_s / self.wall_s if self.wall_s else 0.0


def store_digest(store: Path) -> str:
    """Order-independent digest over every file in the store.

    Order-independent because directory iteration order is not stable across
    filesystems, and this has to compare stores written by two different
    builds on two different runs; the point is the BYTES, not the walk.
    """
    parts = []
    for p in sorted(store.rglob("*")):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            parts.append(f"{p.relative_to(store)}:{h}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def run_once(
    python: str,
    corpus: Path,
    store: Path,
    chunk_size: int,
    threads: int,
    log: Path | None = None,
) -> tuple[int, float, float, float, str]:
    """One conversion. Returns (status, wall_s, cpu_s, maxrss_mb, stderr).

    `log` keeps the child's output even on success. The conversion logs the
    plan it chose (`concurrent_chroms`, `reader_workers`) at info level, and
    a result table without the plan that produced it cannot be acted on."""
    env = {k: v for k, v in os.environ.items() if k not in BENCH_ENV_VARS}
    cmd = [
        python,
        "-m",
        "genoray._cli",
        "write",
        "vcf",
        str(corpus),
        str(store),
        "--no-reference",
        "--log-level",
        "info",
        "--overwrite",
        "-@",
        str(threads),
        "--chunk-size",
        str(chunk_size),
    ]
    if store.exists():
        shutil.rmtree(store)
    t0 = time.perf_counter()
    # `os.wait4`, not `subprocess.run`: rusage is per-wait, so this attributes
    # cpu time and peak RSS to THIS child. `getrusage(RUSAGE_CHILDREN)` would
    # accumulate across every child the driver has ever reaped and report a
    # running maximum, which silently turns arm B's peak RSS into
    # `max(A, B)` once A has run.
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, status, ru = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t0
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
    if proc.stdout:
        proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()
    if log is not None:
        log.write_text(out + err)
    return status, wall, ru.ru_utime + ru.ru_stime, ru.ru_maxrss / 1024.0, err


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True, help="corpus manifest json")
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PYTHON",
        help="an arm to measure; repeat. NAME labels it, PYTHON is that "
        "build's interpreter.",
    )
    p.add_argument("--chunk-size", type=int, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--outdir", type=Path, required=True, help="scratch for the stores")
    p.add_argument("--out", type=Path, required=True, help="NDJSON results")
    a = p.parse_args()

    manifest = json.loads(a.corpus.read_text())
    arms = [tuple(s.split("=", 1)) for s in a.arm]
    node = os.uname().nodename
    a.outdir.mkdir(parents=True, exist_ok=True)

    # Warm-up rep, discarded: the first read of a fresh corpus measures
    # first-touch page-cache fill, not the inflate+parse the conversion is
    # actually bound by. Run it on the FIRST arm only -- the cache it fills is
    # shared, so a per-arm warm-up would just be a slower way to reach the
    # same state and would bias whichever arm went second.
    warm = a.outdir / "warm.svar"
    print(f"  warm-up ({arms[0][0]}) ...", flush=True)
    run_once(arms[0][1], Path(manifest["path"]), warm, a.chunk_size, a.threads)
    if warm.exists():
        shutil.rmtree(warm)

    rows: list[ArmResult] = []
    with a.out.open("a") as sink:
        for rep in range(a.reps):
            for name, python in arms:
                store = a.outdir / f"{name}.svar"
                status, wall, cpu, rss, err = run_once(
                    python,
                    Path(manifest["path"]),
                    store,
                    a.chunk_size,
                    a.threads,
                    log=a.outdir / f"{name}_rep{rep}.log",
                )
                ok = status == 0
                row = ArmResult(
                    arm=name,
                    corpus=str(a.corpus),
                    samples=manifest["samples"],
                    variants=manifest["variants"],
                    contigs=len(manifest["contigs"]),
                    chunk_size=a.chunk_size,
                    threads=a.threads,
                    rep=rep,
                    ok=ok,
                    wall_s=wall,
                    cpu_s=cpu,
                    maxrss_mb=rss,
                    digest=store_digest(store) if ok and store.exists() else "",
                    node=node,
                    error="" if ok else err[-2000:],
                )
                rows.append(row)
                sink.write(json.dumps(asdict(row)) + "\n")
                sink.flush()
                print(
                    f"  rep{rep} {name:<6} wall={wall:7.2f}s cpu={cpu:8.1f}s "
                    f"cores={row.cores:5.2f} rss={rss:7.0f}MB "
                    f"digest={row.digest or 'FAILED'}",
                    flush=True,
                )
                if store.exists():
                    shutil.rmtree(store)

    failed = [r for r in rows if not r.ok]
    if failed:
        print(f"\n  {len(failed)} RUN(S) FAILED", file=sys.stderr)
        print(failed[0].error, file=sys.stderr)
        return 1

    digests = {r.digest for r in rows}
    if len(digests) != 1:
        print("\n  DIGEST DISAGREEMENT ACROSS ARMS -- the builds are NOT equivalent")
        for r in rows:
            print(f"    {r.arm} rep{r.rep} {r.digest}")
        return 1
    print(f"  all arms byte-identical: {digests.pop()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark the eager `Vec<RawRecord>` materialization in `Svar2Source`.

`SparseVar2.write_view` runs the coarse re-conversion seam: `Svar2Source::new`
decodes the whole contig subset up front into an in-memory
`BTreeMap<(pos,ilen,alt), Vec<bool>>` **and** a `Vec<RawRecord>` (each record's
`gt` a `Vec<i32>` of length `n_samples * ploidy`), then drains it. For a
whole-store copy (all samples, whole contig) that is O(n_variants * n_haps)
resident — potentially tens of GB at cohort scale. This benchmark measures peak
RSS and wall time as the sample count grows, to gate whether the eager path is
safe to advertise for cohort-scale whole-store copies or needs a streaming
rewrite first.

One measured point per process (so `ru_maxrss` is that run's peak). Driver mode
(`--ks`) re-execs itself once per k as a subprocess and tabulates.

    pixi run -e py310 python scripts/svar2_eager_bench.py \
        --store data/chr21.germline.svar2 --out-dir /tmp/eager --ks 100 500 1000 3202
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import genoray


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _run_one(store: Path, k: int, out: Path, end: int, threads: int | None) -> dict:
    meta = json.loads((store / "meta.json").read_text())
    samples = meta["samples"][:k]  # deterministic first-k subset
    contig = meta["contigs"][0]
    sv = genoray.SparseVar2(store)
    if out.exists():
        shutil.rmtree(out)
    t0 = time.perf_counter()
    sv.write_view(
        (contig, 0, end),
        samples,
        out,
        reroute=True,
        overwrite=True,
        threads=threads,
    )
    dt = time.perf_counter() - t0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
    out_bytes = _dir_bytes(out)
    n_haps = k * int(meta["ploidy"])
    # Lower-bound on the eager footprint: gt Vec<i32> + carrier Vec<bool> per
    # variant, over every variant in the store (whole-contig view).
    n_var = _variant_count(store, contig)
    eager_lb_gb = n_var * n_haps * (4 + 1) / 1e9
    return {
        "k": k,
        "n_haps": n_haps,
        "wall_s": round(dt, 1),
        "peak_rss_gb": round(peak_kb / 1e6, 2),
        "out_size_gb": round(out_bytes / 1e9, 3),
        "eager_lower_bound_gb": round(eager_lb_gb, 2),
    }


def _variant_count(store: Path, contig: str) -> int:
    import genoray._core as _core

    ii, _sd, _xf, _xs = _core.svar2_variant_stats(str(store), contig, [0])
    return int(ii.size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--end", type=int, default=60_000_000, help="whole-contig region end (0-based)"
    )
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ks", type=int, nargs="+", help="driver: sample counts to sweep")
    ap.add_argument("--k", type=int, help="worker: single sample count (internal)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.k is not None:  # worker
        rec = _run_one(
            args.store,
            args.k,
            args.out_dir / f"view_k{args.k}.svar2",
            args.end,
            args.threads,
        )
        print("RESULT " + json.dumps(rec))
        return

    # driver: one subprocess per k
    rows: list[dict] = []
    for k in args.ks:
        cmd = [
            sys.executable,
            __file__,
            "--store",
            str(args.store),
            "--out-dir",
            str(args.out_dir),
            "--end",
            str(args.end),
            "--k",
            str(k),
        ]
        if args.threads is not None:
            cmd += ["--threads", str(args.threads)]
        print(f"=== k={k} ===", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        sys.stderr.write(p.stderr[-2000:])
        line = next(
            (ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None
        )
        if line is None:
            print(f"  FAILED (rc={p.returncode}); stdout tail:\n{p.stdout[-1500:]}")
            rows.append(
                {
                    "k": k,
                    "wall_s": None,
                    "peak_rss_gb": None,
                    "note": f"rc={p.returncode}",
                }
            )
            continue
        rec = json.loads(line[len("RESULT ") :])
        rows.append(rec)
        print("  " + json.dumps(rec), flush=True)

    print(
        "\n| samples k | haps | wall (s) | peak RSS (GB) | out size (GB) | eager lower-bound (GB) |"
    )
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r.get("peak_rss_gb") is None:
            print(f"| {r['k']} | - | FAILED ({r.get('note', '')}) | - | - | - |")
        else:
            print(
                f"| {r['k']} | {r['n_haps']} | {r['wall_s']} | {r['peak_rss_gb']} | "
                f"{r['out_size_gb']} | {r['eager_lower_bound_gb']} |"
            )


if __name__ == "__main__":
    main()

# scripts/bench_from_vcf_list/run_bench.py
"""Drive SparseVar2.from_vcf_list under /usr/bin/time -v (MaxRSS) or memray.

Runs the conversion in a subprocess so /usr/bin/time captures the whole RSS,
including Rust threads. Sweep N by invoking with different --subset values.
"""

from __future__ import annotations
import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_RUNNER = (
    "from genoray import SparseVar2\n"
    "SparseVar2.from_vcf_list({out!r}, {paths!r}, {ref}, "
    "no_reference={noref}, overwrite=True, threads={threads})\n"
)


def _subprocess_src(
    out: str, paths: list[str], reference: str | None, threads: int | None
) -> str:
    ref = repr(reference) if reference else "None"
    noref = reference is None
    return _RUNNER.format(out=out, paths=paths, ref=ref, noref=noref, threads=threads)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--chrom",
        default="chr1",
        help="the manifest's contig (informational; the Python from_vcf_list path "
        "processes the whole manifest -- restrict at cohort-generation time)",
    )
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--subset", type=int, default=None, help="use first K files")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--profiler", choices=["time", "memray", "none"], default="time")
    ap.add_argument("--results", type=Path, default=Path("results.csv"))
    a = ap.parse_args()

    paths = [p for p in a.manifest.read_text().split() if p]
    if a.subset is not None:
        paths = paths[: a.subset]
    src = _subprocess_src(
        str(a.out), paths, str(a.reference) if a.reference else None, a.threads
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        runner = fh.name

    wall_s = maxrss_kb = None
    try:
        if a.profiler == "memray":
            subprocess.run(["memray", "run", "-o", "bench.memray", runner], check=True)
            print(
                "memray profile: bench.memray (view with `memray flamegraph bench.memray`)"
            )
        else:
            cmd = (
                ["/usr/bin/time", "-v", sys.executable, runner]
                if a.profiler == "time"
                else [sys.executable, runner]
            )
            proc = subprocess.run(cmd, capture_output=True, text=True)
            sys.stderr.write(proc.stderr)
            proc.check_returncode()
            if a.profiler == "time":
                m = re.search(
                    r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr
                )
                e = re.search(r"Elapsed \(wall clock\).*?:\s*([\d:.]+)", proc.stderr)
                # Fail loud rather than write a blank row: a silent None here would
                # look like a real zero-cost measurement in a multi-N sweep.
                if m is None or e is None:
                    raise RuntimeError(
                        "could not parse MaxRSS/Elapsed from /usr/bin/time -v output; "
                        "refusing to append a blank results row"
                    )
                maxrss_kb = int(m.group(1))
                parts = [float(x) for x in e.group(1).split(":")]
                wall_s = (
                    parts[-1]
                    + (parts[-2] * 60 if len(parts) > 1 else 0)
                    + (parts[-3] * 3600 if len(parts) > 2 else 0)
                )
    finally:
        Path(runner).unlink(missing_ok=True)

    new = not a.results.exists()
    with open(a.results, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["n_files", "wall_s", "maxrss_kb", "profiler"])
        w.writerow([len(paths), wall_s, maxrss_kb, a.profiler])
    print(f"n_files={len(paths)} wall_s={wall_s} maxrss_kb={maxrss_kb}")


if __name__ == "__main__":
    main()

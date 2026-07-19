# scripts/bench_from_vcf_list/run_bench.py
"""Drive SparseVar2.from_vcf_list under /usr/bin/time -v (MaxRSS) or memray.

Runs the conversion in a subprocess so /usr/bin/time captures the whole RSS,
including Rust threads. Sweep N by invoking with different --subset values.
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_BANNER_RE = re.compile(r"^==> Processing (\S+)", re.MULTILINE)

_RUNNER = (
    "from genoray import SparseVar2, FormatField\n"
    "SparseVar2.from_vcf_list({out!r}, {paths!r}, {ref}, "
    "no_reference={noref}, overwrite=True, threads={threads}, "
    "format_fields=[FormatField(n) for n in {fields!r}])\n"
)


def _subprocess_src(
    out: str,
    paths: list[str],
    reference: str | None,
    threads: int | None,
    fields: list[str],
) -> str:
    ref = repr(reference) if reference else "None"
    noref = reference is None
    return _RUNNER.format(
        out=out, paths=paths, ref=ref, noref=noref, threads=threads, fields=fields
    )


def parse_per_contig_highwater(
    stderr: str,
    rss_samples: list[tuple[float, int]],
    *,
    banner_times: dict[str, float],
) -> dict[str, int]:
    """Bucket RSS samples by contig, using each banner's stamped arrival time.

    ``banner_times`` maps each contig label (from an ``==> Processing {chrom}``
    banner) to the elapsed_s at which that banner arrived. Each contig's bucket
    runs from its own arrival time up to the *next* contig's arrival time (or
    to the end of the run, for the last contig). The high-water mark for a
    contig is the max RSS-in-KB observed in its window.
    """
    labels = _BANNER_RE.findall(stderr)
    # Preserve encounter order, but bucket edges come from banner_times.
    ordered = [label for label in labels if label in banner_times]
    highwater: dict[str, int] = {}
    for i, label in enumerate(ordered):
        start = banner_times[label]
        end = banner_times[ordered[i + 1]] if i + 1 < len(ordered) else float("inf")
        bucket_rss = [rss for elapsed, rss in rss_samples if start <= elapsed < end]
        if bucket_rss:
            highwater[label] = max(bucket_rss)
    return highwater


def parse_arena_heaps(status_or_smaps: str) -> int:
    """Count glibc 64 MB (65536 kB) arena heaps in a captured smaps-style string."""
    return len(re.findall(r"^Size:\s*65536\s*kB", status_or_smaps, re.MULTILINE))


class _RssSampler:
    """Background thread sampling a child's VmRSS and stdout banner lines.

    Reads ``/proc/<pid>/status`` for VmRSS and ``/proc/<pid>/smaps`` for
    arena-heap counts every ``interval`` seconds, and tails ``stdout_fh`` for
    ``==> Processing {chrom}`` banners, stamping each banner's arrival time.
    """

    def __init__(self, pid: int, stdout_fh, interval: float) -> None:
        self.pid = pid
        self.stdout_fh = stdout_fh
        self.interval = interval
        self.rss_samples: list[tuple[float, int]] = []
        self.banner_times: dict[str, float] = {}
        self.stdout_lines: list[str] = []
        self.max_arena_heaps = 0
        self._stop = threading.Event()
        self._start = time.monotonic()
        self._reader_thread = threading.Thread(target=self._tail_stdout, daemon=True)
        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)

    def start(self) -> None:
        self._reader_thread.start()
        self._sampler_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._reader_thread.join(timeout=self.interval + 1)
        self._sampler_thread.join(timeout=self.interval + 1)

    def _elapsed(self) -> float:
        return time.monotonic() - self._start

    def _tail_stdout(self) -> None:
        if self.stdout_fh is None:
            return
        for line in iter(self.stdout_fh.readline, ""):
            if not line:
                break
            self.stdout_lines.append(line)
            m = re.match(r"^==> Processing (\S+)", line)
            if m:
                self.banner_times.setdefault(m.group(1), self._elapsed())

    def _sample_loop(self) -> None:
        status_path = f"/proc/{self.pid}/status"
        smaps_path = f"/proc/{self.pid}/smaps"
        while not self._stop.is_set():
            elapsed = self._elapsed()
            try:
                status = Path(status_path).read_text()
                m = re.search(r"VmRSS:\s*(\d+)\s*kB", status)
                if m:
                    self.rss_samples.append((elapsed, int(m.group(1))))
            except OSError:
                break
            try:
                smaps = Path(smaps_path).read_text()
                self.max_arena_heaps = max(
                    self.max_arena_heaps, parse_arena_heaps(smaps)
                )
            except OSError:
                pass
            self._stop.wait(self.interval)


def _find_child_pid(parent_pid: int, timeout: float = 5.0) -> int | None:
    """Find the first child of `parent_pid` via /proc's `children` file.

    `/usr/bin/time -v <cmd>` forks the measured process rather than exec'ing
    it in place, so `parent_pid` (the `time` process itself) is not the PID
    whose RSS we want to sample -- its child is.
    """
    children_path = Path(f"/proc/{parent_pid}/task/{parent_pid}/children")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = children_path.read_text().strip()
        except OSError:
            return None
        if text:
            return int(text.split()[0])
        time.sleep(0.05)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--chrom",
        dest="chroms",
        action="append",
        default=None,
        help="the manifest's contig(s), repeatable (informational; the Python "
        "from_vcf_list path processes the whole manifest -- restrict at "
        "cohort-generation time)",
    )
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--subset", type=int, default=None, help="use first K files")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--format-field",
        dest="format_fields",
        action="append",
        default=[],
        help="repeatable FORMAT field to request from SparseVar2.from_vcf_list",
    )
    ap.add_argument("--profiler", choices=["time", "memray", "none"], default="time")
    ap.add_argument("--results", type=Path, default=Path("results.csv"))
    ap.add_argument(
        "--rss-sample-interval",
        type=float,
        default=2.0,
        help="seconds between /proc/<pid>/status VmRSS + smaps samples "
        "(--profiler time only)",
    )
    a = ap.parse_args()

    paths = [p for p in a.manifest.read_text().split() if p]
    if a.subset is not None:
        paths = paths[: a.subset]
    src = _subprocess_src(
        str(a.out),
        paths,
        str(a.reference) if a.reference else None,
        a.threads,
        a.format_fields,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        runner = fh.name

    wall_s = cpu_s = maxrss_kb = None
    per_contig_highwater: dict[str, int] = {}
    arena_heaps = 0
    try:
        if a.profiler == "memray":
            subprocess.run(["memray", "run", "-o", "bench.memray", runner], check=True)
            print(
                "memray profile: bench.memray (view with `memray flamegraph bench.memray`)"
            )
        elif a.profiler == "time":
            cmd = ["/usr/bin/time", "-v", sys.executable, runner]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            child_pid = _find_child_pid(proc.pid)
            sampler = _RssSampler(
                child_pid if child_pid is not None else proc.pid,
                proc.stdout,
                a.rss_sample_interval,
            )
            sampler.start()
            stderr_text = proc.stderr.read()
            proc.wait()
            sampler.stop()

            sys.stderr.write(stderr_text)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)

            m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr_text)
            e = re.search(r"Elapsed \(wall clock\).*?:\s*([\d:.]+)", stderr_text)
            u = re.search(r"User time \(seconds\):\s*([\d.]+)", stderr_text)
            s = re.search(r"System time \(seconds\):\s*([\d.]+)", stderr_text)
            # Fail loud rather than write a blank row: a silent None here would
            # look like a real zero-cost measurement in a multi-N sweep. Same
            # reasoning applies to CPU time -- a silently-zero cpu_s would
            # corrupt an exponent fit exactly the way a blank wall_s would.
            if m is None or e is None or u is None or s is None:
                raise RuntimeError(
                    "could not parse MaxRSS/Elapsed/User/System from "
                    "/usr/bin/time -v output; refusing to append a blank "
                    "results row"
                )
            maxrss_kb = int(m.group(1))
            parts = [float(x) for x in e.group(1).split(":")]
            wall_s = (
                parts[-1]
                + (parts[-2] * 60 if len(parts) > 1 else 0)
                + (parts[-3] * 3600 if len(parts) > 2 else 0)
            )
            cpu_s = float(u.group(1)) + float(s.group(1))
            per_contig_highwater = parse_per_contig_highwater(
                "".join(sampler.stdout_lines),
                sampler.rss_samples,
                banner_times=sampler.banner_times,
            )
            arena_heaps = sampler.max_arena_heaps
        else:
            cmd = [sys.executable, runner]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            sys.stderr.write(proc.stderr)
            proc.check_returncode()
    finally:
        Path(runner).unlink(missing_ok=True)

    n_fields = len(a.format_fields)
    per_contig_highwater_json = json.dumps(per_contig_highwater)
    new = not a.results.exists()
    with open(a.results, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(
                [
                    "n_files",
                    "fields",
                    "wall_s",
                    "cpu_s",
                    "maxrss_kb",
                    "profiler",
                    "per_contig_highwater_json",
                    "arena_heaps",
                ]
            )
        w.writerow(
            [
                len(paths),
                n_fields,
                wall_s,
                cpu_s,
                maxrss_kb,
                a.profiler,
                per_contig_highwater_json,
                arena_heaps,
            ]
        )
    print(
        f"n_files={len(paths)} fields={n_fields} wall_s={wall_s} cpu_s={cpu_s} "
        f"maxrss_kb={maxrss_kb} arena_heaps={arena_heaps} "
        f"per_contig_highwater={per_contig_highwater_json}"
    )


if __name__ == "__main__":
    main()

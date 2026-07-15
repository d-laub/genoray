"""Benchmark native region/sub-contig ``SparseVar2.from_vcf`` conversion.

The handoff for native SVAR2 regions asks for a scaling benchmark that separates
the new indexed VCF path from older post-hoc ``write_view`` measurements. This
script records one end-to-end conversion measurement per subprocess so peak RSS
is meaningful for every thread/chunk-size row.

Examples
--------

Run against an existing wide VCF/BCF:

    pixi run python scripts/svar2_region_parallel_bench.py \
        --vcf data/chr21.bcf \
        --reference /carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa \
        --regions chr21:1-50000000 \
        --out-dir /scratch/$USER/genoray-region-bench \
        --threads 1 2 4 8 16 32 \
        --chunk-sizes 25000

Generate a deterministic synthetic wide-sample VCF first:

    pixi run python scripts/svar2_region_parallel_bench.py \
        --make-synthetic \
        --synthetic-samples 5000 \
        --synthetic-variants 10000 \
        --out-dir /tmp/genoray-region-bench \
        --threads 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_THREADS = [1, 2, 4, 8, 16, 32]
DEFAULT_CHUNK_SIZES = [25_000]


def compute_scaling(rows: list[dict]) -> list[dict]:
    """Add speedup + parallel efficiency vs the fewest-threads row of the same
    (backend, chunk_size) group. Rows are returned in input order, enriched."""
    baseline: dict[tuple, float] = {}
    for r in rows:
        key = (r["backend"], r["chunk_size"])
        t = r["threads"]
        if key not in baseline or t < baseline[key][0]:
            baseline[key] = (t, r["wall_s"])
    out = []
    for r in rows:
        base_t, base_wall = baseline[(r["backend"], r["chunk_size"])]
        speedup = base_wall / r["wall_s"]
        out.append(
            {**r, "speedup": speedup, "efficiency": speedup / (r["threads"] / base_t)}
        )
    return out


def oracle_hash(store: Path) -> str:
    """Stable content hash of an SVAR2 store: sha256 over sorted (relpath, bytes)."""
    import hashlib

    h = hashlib.sha256()
    for p in sorted(store.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(store).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(rss)
    return int(rss) * 1024


def _variant_count(store: Path, contig: str) -> int:
    import genoray._core as _core

    ii, _sd, _xf, _xs = _core.svar2_variant_stats(str(store), contig, [0])
    return int(ii.size)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run_checked(cmd: list[str], *, stdout: Any | None = None) -> None:
    subprocess.run(cmd, check=True, stdout=stdout)


def _write_synthetic_input(
    root: Path,
    *,
    contig: str,
    n_samples: int,
    n_variants: int,
    carrier_stride: int,
) -> tuple[Path, Path]:
    """Create a bgzipped/indexed VCF plus matching all-A FASTA reference."""

    if n_samples < 1:
        raise ValueError("--synthetic-samples must be positive")
    if n_variants < 1:
        raise ValueError("--synthetic-variants must be positive")
    if carrier_stride < 1:
        raise ValueError("--carrier-stride must be positive")

    root.mkdir(parents=True, exist_ok=True)
    length = n_variants + 100
    ref = root / "synthetic.fa"
    _write_text(ref, f">{contig}\n{'A' * length}\n")
    _run_checked(["samtools", "faidx", str(ref)])

    sample_names = [f"S{i:06d}" for i in range(n_samples)]
    plain = root / "synthetic.vcf"
    with plain.open("w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##contig=<ID={contig},length={length}>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(sample_names)
            + "\n"
        )
        for vi in range(n_variants):
            pos = vi + 1
            genotypes = (
                "0/1" if (vi + si) % carrier_stride == 0 else "0/0"
                for si in range(n_samples)
            )
            fh.write(
                f"{contig}\t{pos}\t.\tA\tC\t.\t.\t.\tGT\t" + "\t".join(genotypes) + "\n"
            )

    gz = root / "synthetic.vcf.gz"
    with gz.open("wb") as out:
        _run_checked(["bgzip", "-f", "-c", str(plain)], stdout=out)
    _run_checked(["bcftools", "index", "-f", str(gz)])
    return gz, ref


def _sample_file_for_count(vcf: Path, root: Path, count: int) -> Path:
    import cyvcf2

    samples = list(cyvcf2.VCF(str(vcf)).samples)
    if count < 1 or count > len(samples):
        raise ValueError(f"sample count {count} is outside 1..{len(samples)} for {vcf}")
    path = root / f"samples_first_{count}.txt"
    path.write_text("\n".join(samples[:count]) + "\n")
    return path


def _pathlike_arg(value: str | None) -> str | Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.exists() else value


def _run_worker(args: argparse.Namespace) -> None:
    from genoray import SparseVar2

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    reference = None if args.no_reference else args.reference
    t0 = time.perf_counter()
    if args.backend == "pgen":
        dropped = SparseVar2.from_pgen(
            out,
            args.vcf,
            reference,
            no_reference=args.no_reference,
            skip_out_of_scope=args.skip_out_of_scope,
            chunk_size=args.chunk_size,
            threads=args.thread_count,
            overwrite=True,
        )
    else:
        dropped = SparseVar2.from_vcf(
            out,
            args.vcf,
            reference,
            regions=_pathlike_arg(args.regions),
            samples=_pathlike_arg(args.samples),
            no_reference=args.no_reference,
            skip_out_of_scope=args.skip_out_of_scope,
            chunk_size=args.chunk_size,
            threads=args.thread_count,
            overwrite=True,
        )
    wall_s = time.perf_counter() - t0

    meta = json.loads((out / "meta.json").read_text())
    contigs = list(meta.get("contigs", []))
    variants = sum(_variant_count(out, contig) for contig in contigs)
    rec = {
        "vcf": str(args.vcf),
        "reference": None if args.no_reference else str(args.reference),
        "regions": args.regions,
        "samples": args.samples,
        "backend": args.backend,
        "threads": args.thread_count,
        "thread_count": args.thread_count,
        "chunk_size": args.chunk_size,
        "repeat": args.repeat_index,
        "wall_s": round(wall_s, 3),
        "peak_rss_gb": round(_peak_rss_bytes() / 1e9, 3),
        "out_size_gb": round(_dir_bytes(out) / 1e9, 3),
        "dropped_out_of_scope": int(dropped),
        "n_samples": len(meta.get("samples", [])),
        "contigs": contigs,
        "variants": variants,
        "phase_timings": {"end_to_end_wall_s": round(wall_s, 3)},
        "phase_timing_note": (
            "Native phase timings are not exposed yet; wire this to structured "
            "progress after PR #113 lands."
        ),
    }
    print("RESULT " + json.dumps(rec, sort_keys=True), flush=True)


def _driver_command(
    args: argparse.Namespace,
    *,
    thread_count: int,
    chunk_size: int,
    repeat_index: int,
    output: Path,
    samples: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--vcf",
        str(args.vcf),
        "--output",
        str(output),
        "--backend",
        args.backend,
        "--thread-count",
        str(thread_count),
        "--chunk-size",
        str(chunk_size),
        "--repeat-index",
        str(repeat_index),
    ]
    if args.no_reference:
        cmd.append("--no-reference")
    else:
        cmd += ["--reference", str(args.reference)]
    if args.regions:
        cmd += ["--regions", args.regions]
    if samples:
        cmd += ["--samples", samples]
    if args.skip_out_of_scope:
        cmd.append("--skip-out-of-scope")
    return cmd


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    print(
        "\n| backend | threads | chunk_size | repeat | wall_s | speedup | efficiency | "
        "peak_rss_gb | out_size_gb | variants | samples | oracle_ok | note |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        if row.get("failed"):
            print(
                f"| {row.get('backend', '')} | {row['thread_count']} | {row['chunk_size']} | "
                f"{row['repeat']} | - | - | - | - | - | - | - | - | rc={row['returncode']} |"
            )
            continue
        print(
            f"| {row['backend']} | {row['thread_count']} | {row['chunk_size']} | {row['repeat']} | "
            f"{row['wall_s']} | {row.get('speedup', '')} | {row.get('efficiency', '')} | "
            f"{row['peak_rss_gb']} | {row['out_size_gb']} | "
            f"{row['variants']} | {row['n_samples']} | {row.get('oracle_ok', '')} |  |"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--regions")
    parser.add_argument(
        "--samples", help="Sample name, comma list, or sample-list file"
    )
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, nargs="+", default=DEFAULT_THREADS)
    parser.add_argument(
        "--chunk-sizes", type=int, nargs="+", default=DEFAULT_CHUNK_SIZES
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--jsonl",
        type=Path,
        help=(
            "Write one JSON object per row to this path. Raw rows are appended "
            "incrementally as each conversion finishes (crash-durable); when the "
            "full sweep completes the file is rewritten with the scaling-enriched "
            "rows (speedup/efficiency added)."
        ),
    )
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--skip-out-of-scope", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["vcf", "pgen"],
        default="vcf",
        help="Conversion backend: SparseVar2.from_vcf or SparseVar2.from_pgen.",
    )
    parser.add_argument(
        "--oracle-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify every conversion is byte-identical (sha256 over sorted store "
            "files) to the reference conversion at the fewest threads in its "
            "(backend, chunk_size) group. On by default; a mismatch aborts the run."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--thread-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)

    parser.add_argument("--make-synthetic", action="store_true")
    parser.add_argument("--synthetic-contig", default="chrBench")
    parser.add_argument("--synthetic-samples", type=int, default=2_000)
    parser.add_argument("--synthetic-variants", type=int, default=10_000)
    parser.add_argument("--carrier-stride", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.worker:
        if args.vcf is None or args.output is None:
            raise SystemExit("--worker requires --vcf and --output")
        if not args.no_reference and args.reference is None:
            raise SystemExit(
                "--worker requires --reference unless --no-reference is set"
            )
        _run_worker(args)
        return

    if args.out_dir is None:
        raise SystemExit("--out-dir is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.make_synthetic:
        args.vcf, args.reference = _write_synthetic_input(
            args.out_dir / "synthetic_input",
            contig=args.synthetic_contig,
            n_samples=args.synthetic_samples,
            n_variants=args.synthetic_variants,
            carrier_stride=args.carrier_stride,
        )

    if args.vcf is None:
        raise SystemExit("provide --vcf or --make-synthetic")
    if not args.no_reference and args.reference is None:
        raise SystemExit("provide --reference or --no-reference")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.backend == "pgen" and (args.regions or args.samples):
        raise SystemExit(
            "--backend pgen does not support --regions/--samples: "
            "SparseVar2.from_pgen ignores them, so recording them in the row "
            "would falsely imply a filter was applied. Drop those flags or use "
            "--backend vcf."
        )

    samples = args.samples
    if args.sample_count is not None:
        samples = str(_sample_file_for_count(args.vcf, args.out_dir, args.sample_count))

    # Reference oracle hash per chunk_size, established once at the fewest
    # threads in each (backend, chunk_size) group (threads are processed in
    # ascending order below so the reference always runs first).
    reference_hashes: dict[int, str] = {}

    rows: list[dict[str, Any]] = []
    for chunk_size in args.chunk_sizes:
        for thread_count in sorted(args.threads):
            for repeat_index in range(args.repeats):
                label = f"t{thread_count}_c{chunk_size}_r{repeat_index}"
                output = args.out_dir / f"from_{args.backend}_{label}.svar2"
                cmd = _driver_command(
                    args,
                    thread_count=thread_count,
                    chunk_size=chunk_size,
                    repeat_index=repeat_index,
                    output=output,
                    samples=samples,
                )
                print(
                    f"=== backend={args.backend} threads={thread_count} "
                    f"chunk={chunk_size} repeat={repeat_index} ==="
                )
                proc = subprocess.run(cmd, capture_output=True, text=True)
                sys.stderr.write(proc.stderr[-4000:])
                line = next(
                    (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")),
                    None,
                )
                if line is None:
                    row = {
                        "failed": True,
                        "backend": args.backend,
                        "thread_count": thread_count,
                        "chunk_size": chunk_size,
                        "repeat": repeat_index,
                        "returncode": proc.returncode,
                        "stdout_tail": proc.stdout[-2000:],
                    }
                    print(
                        f"  FAILED rc={proc.returncode}; stdout tail:\n{proc.stdout[-1500:]}"
                    )
                else:
                    row = json.loads(line[len("RESULT ") :])
                    if args.oracle_check:
                        actual_hash = oracle_hash(output)
                        reference = reference_hashes.get(chunk_size)
                        if reference is None:
                            reference_hashes[chunk_size] = actual_hash
                            row["oracle_ok"] = True
                        else:
                            row["oracle_ok"] = actual_hash == reference
                            if not row["oracle_ok"]:
                                raise SystemExit(
                                    "Oracle check FAILED: conversion is not "
                                    "byte-identical to the reference store for "
                                    f"backend={args.backend} chunk_size={chunk_size} "
                                    f"(row threads={thread_count} repeat={repeat_index}). "
                                    f"hash={actual_hash} reference={reference}"
                                )
                    print("  " + json.dumps(row, sort_keys=True), flush=True)
                rows.append(row)
                # Append the raw row immediately so a crash mid-sweep still
                # leaves durable per-row output; the file is rewritten with the
                # scaling-enriched rows once the full sweep completes.
                if args.jsonl is not None:
                    with args.jsonl.open("a") as fh:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                if not args.keep_outputs and output.exists():
                    shutil.rmtree(output)

    ok_rows = [r for r in rows if not r.get("failed")]
    if ok_rows:
        scaled_iter = iter(compute_scaling(ok_rows))
        rows = [r if r.get("failed") else next(scaled_iter) for r in rows]

    if args.jsonl is not None:
        with args.jsonl.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    print("\n=== scaled rows ===")
    for row in rows:
        print(json.dumps(row, sort_keys=True))

    _print_markdown(rows)


if __name__ == "__main__":
    main()

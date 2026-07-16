# SparseVar2.from_vcf_list scaling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per project convention, dispatch implementers with **Sonnet or a weaker model** (Opus reserved for review and critical-failure fixes), and run long `cargo`/`maturin` builds **in the foreground** (implementer subagents must not background them and return early).

**Goal:** Make `SparseVar2.from_vcf_list` peak RAM independent of cohort size (issue #120) and cut its multi-hour wall-time, driven by a reusable benchmark + profiling harness.

**Architecture:** Measure-first. Phase 0 builds a synthetic-cohort generator, a native Rust bench binary (the clean target for `dhat`/`perf`/`callgrind`/`cargo-show-asm`), a Python driver (`/usr/bin/time` + `memray`), and a byte-identical parity fixture. Phase 1 records a baseline. Phase 2 lands the two code-confirmed fixes (budget-derived `chunk_size`; frontier min-heap for O(log N) merge selection). Phase 3 holds the measurement-gated fixes (staged/parallel merge, `max_mem` knob, ledger trim), concretized after the Phase 1 numbers.

**Tech Stack:** Rust (rust-htslib, rayon, crossbeam-channel), pyo3, Python (cyvcf2), `dhat` crate, `memray`, `perf`, `callgrind`, `cargo-show-asm`, bgzip/tabix/bcftools (pixi env).

## Global Constraints

- **Byte-identical output** on the parity fixture for every optimization — hard gate (Task 4 defines it; Tasks 6, 7, and every Phase 3 task must leave it green).
- **Rust tests:** `pixi run test-rust` == `cargo test --no-default-features --features conversion`. Never plain `cargo test` (pyo3 `extension-module` fails to link: `undefined symbol: _Py_Dealloc`).
- **NFS build guard:** `export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$` before any cargo/commit that triggers cargo hooks (NFS `target/` bus-errors the lint-env cargo hooks).
- **prek hooks** must be installed before committing/pushing (`pixi run prek-install`).
- **Public-surface rule:** any change reachable from `import genoray` without underscores (kwargs, defaults' documented semantics, return shapes) requires the **same PR** to update `python/genoray/_svar2.py` docstrings **and** `skills/genoray-api/SKILL.md`.
- **Commit convention:** Conventional Commits (`feat:`/`fix:`/`docs:`/`chore:`/`perf:`). Never edit `CHANGELOG.md`.
- **dhat isolation:** the `#[global_allocator]` and `dhat::Profiler` live only behind the `dhat-heap` cargo feature and only in the bench binary — never compiled into the shipped `_core` cdylib.

---

## File Structure

- `scripts/bench_from_vcf_list/generate_cohort.py` — synthetic single-sample cohort generator (Task 1).
- `scripts/bench_from_vcf_list/run_bench.py` — Python driver: `/usr/bin/time -v` + `memray`, N-sweep, results CSV (Task 3).
- `scripts/bench_from_vcf_list/README.md` — how to run each profiler, incl. dhat/perf/callgrind/cargo-show-asm invocations (Task 3).
- `src/bin/bench_from_vcf_list.rs` — native bench binary calling `orchestrator::run_vcf_list` directly; dhat guard behind `dhat-heap` (Task 2).
- `Cargo.toml` — add optional `dhat` dep, `dhat-heap` feature, `[[bin]]` (Task 2).
- `tests/test_bench_generate_cohort.py` — generator unit test (Task 1).
- `tests/test_svar2_from_vcf_list_parity.py` — `hash_store` helper + golden-hash parity gate (Task 4).
- `python/genoray/_svar2.py` — `from_vcf_list` `chunk_size` budget default (Task 6); `max_mem` knob (Task 8).
- `src/vcf_list_reader.rs` — frontier min-heap (Task 7); staged/parallel merge (Task 9).
- `skills/genoray-api/SKILL.md` — public-surface updates (Tasks 6, 8).
- `docs/superpowers/plans/2026-07-15-svar2-from-vcf-list-scaling-baseline.md` — recorded baseline numbers (Task 5).

---

## Phase 0 — Harness (Tasks 1–4 are independent; dispatch in parallel)

### Task 1: Synthetic somatic-cohort generator

**Files:**
- Create: `scripts/bench_from_vcf_list/generate_cohort.py`
- Test: `tests/test_bench_generate_cohort.py`

**Interfaces:**
- Produces: `generate_cohort(out_dir: Path, n_files: int, n_variants: int, *, contig: str = "chr1", contig_len: int = 1_000_000, shared_frac: float = 0.1, indel_frac: float = 0.1, seed: int = 0) -> Path` — writes `sample_{i}.vcf.gz` (+ `.csi`) and `manifest.txt` (one path per line), returns the manifest path. Also a `main()` argparse CLI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_generate_cohort.py
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest
from cyvcf2 import VCF

_SPEC = Path(__file__).parent.parent / "scripts" / "bench_from_vcf_list" / "generate_cohort.py"


def _load():
    spec = importlib.util.spec_from_file_location("generate_cohort", _SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_cohort_shape(tmp_path: Path):
    gen = _load()
    manifest = gen.generate_cohort(
        tmp_path, n_files=4, n_variants=6, contig="chr1", contig_len=10_000,
        shared_frac=0.5, indel_frac=0.25, seed=7,
    )
    paths = [Path(p) for p in manifest.read_text().split()]
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.suffix == ".gz"
        v = VCF(str(p))
        assert len(v.samples) == 1  # single-sample
        rows = list(v("chr1"))
        assert rows == sorted(rows, key=lambda r: r.POS)  # position-sorted
    # shared sites: at least one POS appears in >= 2 files
    from collections import Counter
    pos_counts = Counter(r.POS for p in paths for r in VCF(str(p))("chr1"))
    assert max(pos_counts.values()) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_bench_generate_cohort.py -v`
Expected: FAIL (module/file not found).

- [ ] **Step 3: Write the generator**

```python
# scripts/bench_from_vcf_list/generate_cohort.py
"""Generate a synthetic single-sample somatic cohort for from_vcf_list benchmarks.

Each file gets `shared_frac` of its sites drawn from a shared pool (so the k-way
merge actually joins) and the rest private (fan-out). Reproducible by seed.
"""
from __future__ import annotations
import argparse
import random
import subprocess
from pathlib import Path

_BASES = "ACGT"
_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID={contig},length={length}>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
)


def _alt(rng: random.Random, ref: str, indel: bool) -> tuple[str, str]:
    if indel:  # simple 1bp insertion anchored on ref
        return ref, ref + rng.choice(_BASES)
    alt = rng.choice([b for b in _BASES if b != ref])
    return ref, alt


def generate_cohort(
    out_dir: Path, n_files: int, n_variants: int, *, contig: str = "chr1",
    contig_len: int = 1_000_000, shared_frac: float = 0.1, indel_frac: float = 0.1,
    seed: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n_shared = int(n_variants * shared_frac)
    shared_pos = sorted(rng.sample(range(1, contig_len), n_shared)) if n_shared else []
    manifest_paths: list[str] = []
    for i in range(n_files):
        frng = random.Random(seed * 100003 + i)
        n_priv = n_variants - n_shared
        priv_pos = frng.sample(range(1, contig_len), n_priv) if n_priv else []
        positions = sorted(set(shared_pos) | set(priv_pos))
        lines = []
        for pos in positions:
            ref = frng.choice(_BASES)
            is_indel = frng.random() < indel_frac
            r, a = _alt(frng, ref, is_indel)
            lines.append(f"{contig}\t{pos}\t.\t{r}\t{a}\t.\tPASS\t.\tGT\t1/1\n")
        sample = f"S{i:05d}"
        plain = out_dir / f"sample_{i:05d}.vcf"
        plain.write_text(_HEADER.format(contig=contig, length=contig_len, sample=sample) + "".join(lines))
        gz = out_dir / f"sample_{i:05d}.vcf.gz"
        with open(gz, "wb") as fh:
            subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
        subprocess.run(["bcftools", "index", str(gz)], check=True)
        plain.unlink()
        manifest_paths.append(str(gz))
    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(manifest_paths) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-files", type=int, required=True)
    ap.add_argument("--n-variants", type=int, required=True)
    ap.add_argument("--contig", default="chr1")
    ap.add_argument("--contig-len", type=int, default=1_000_000)
    ap.add_argument("--shared-frac", type=float, default=0.1)
    ap.add_argument("--indel-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = generate_cohort(a.out_dir, a.n_files, a.n_variants, contig=a.contig,
                        contig_len=a.contig_len, shared_frac=a.shared_frac,
                        indel_frac=a.indel_frac, seed=a.seed)
    print(m)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_bench_generate_cohort.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_from_vcf_list/generate_cohort.py tests/test_bench_generate_cohort.py
git commit -m "feat(bench): synthetic single-sample cohort generator for from_vcf_list"
```

---

### Task 2: Native Rust bench binary + dhat wiring

**Files:**
- Create: `src/bin/bench_from_vcf_list.rs`
- Modify: `Cargo.toml` (add `dhat` optional dep, `dhat-heap` feature, `[[bin]]`)

**Interfaces:**
- Consumes: `genoray_core::orchestrator::run_vcf_list` (existing pub fn), a `manifest.txt` (one VCF path per line, as Task 1 emits).
- Produces: an executable `bench_from_vcf_list` runnable as
  `cargo run --release --no-default-features --features conversion --bin bench_from_vcf_list -- <manifest> <out_dir> <chrom> [reference.fa]`, and with dhat via
  `cargo run --profile profiling --no-default-features --features conversion,dhat-heap --bin bench_from_vcf_list -- ...` (emits `dhat-heap.json`).

- [ ] **Step 1: Add the dep, feature, and bin target to `Cargo.toml`**

Under `[dependencies]`:
```toml
# Heap profiling for the from_vcf_list bench binary only (gated behind `dhat-heap`,
# never compiled into the shipped _core cdylib).
dhat = { version = "0.3", optional = true }
```
Under `[features]`:
```toml
dhat-heap = ["dep:dhat"]
```
At end of file:
```toml
[[bin]]
name = "bench_from_vcf_list"
required-features = ["conversion"]
```

- [ ] **Step 2: Write the bench binary**

```rust
// src/bin/bench_from_vcf_list.rs
//! Native driver for `orchestrator::run_vcf_list` — the clean target for dhat,
//! perf, callgrind, and cargo-show-asm (no Python in the loop). Sample names are
//! read from each file's header via rust-htslib.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use genoray_core::orchestrator::run_vcf_list;
use rust_htslib::bcf::{Read, Reader};
use std::fs;
use std::time::Instant;

fn sample_of(path: &str) -> String {
    let r = Reader::from_path(path).expect("open vcf");
    let s = r.header().samples();
    String::from_utf8_lossy(s[0]).into_owned()
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <manifest> <out_dir> <chrom> [reference.fa]", args[0]);
        std::process::exit(2);
    }
    let manifest = &args[1];
    let out_dir = &args[2];
    let chrom = &args[3];
    let reference = args.get(4).map(|s| s.as_str());

    let vcf_paths: Vec<String> = fs::read_to_string(manifest)
        .expect("read manifest")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .collect();
    let samples: Vec<String> = vcf_paths.iter().map(|p| sample_of(p)).collect();

    let t = Instant::now();
    let dropped = run_vcf_list(
        &vcf_paths,
        reference,
        &[chrom.clone()],
        out_dir,
        &samples,
        25_000, // chunk_size (overridden by later tasks' experiments via env if desired)
        2,      // ploidy
        None,   // max_threads = auto
        8 * 1024 * 1024,
        false,  // skip_out_of_scope
        false,  // signatures
        Vec::new(),
        Vec::new(),
    )
    .expect("run_vcf_list");
    eprintln!("done: dropped={dropped} elapsed={:.1}s files={}", t.elapsed().as_secs_f64(), vcf_paths.len());
}
```

- [ ] **Step 3: Build it (foreground)**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$ && cargo build --release --no-default-features --features conversion --bin bench_from_vcf_list`
Expected: compiles; binary at `$CARGO_TARGET_DIR/release/bench_from_vcf_list`.

- [ ] **Step 4: Smoke-run it against a tiny generated cohort**

Run:
```bash
python scripts/bench_from_vcf_list/generate_cohort.py /tmp/bench_smoke --n-files 5 --n-variants 20 --contig chr1 --contig-len 10000 --shared-frac 0.4 --seed 1
cargo run --release --no-default-features --features conversion --bin bench_from_vcf_list -- /tmp/bench_smoke/manifest.txt /tmp/bench_smoke_out chr1
```
Expected: prints `done: dropped=0 elapsed=...s files=5`; `/tmp/bench_smoke_out/meta.json` exists.

- [ ] **Step 5: Verify dhat build path compiles (foreground)**

Run: `cargo build --profile profiling --no-default-features --features conversion,dhat-heap --bin bench_from_vcf_list`
Expected: compiles. (Running it emits `dhat-heap.json` in cwd.)

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/bin/bench_from_vcf_list.rs
git commit -m "feat(bench): native run_vcf_list bench binary with dhat-heap feature"
```

---

### Task 3: Python bench driver (`/usr/bin/time` + memray) and README

**Files:**
- Create: `scripts/bench_from_vcf_list/run_bench.py`
- Create: `scripts/bench_from_vcf_list/README.md`

**Interfaces:**
- Consumes: a `manifest.txt` (Task 1), the reference FASTA (optional), `SparseVar2.from_vcf_list`.
- Produces: `run_bench.py --manifest M --out O --chrom C [--reference F] [--subset K] [--profiler {time,memray,none}]` → appends a row to `results.csv` (`n_files,wall_s,maxrss_kb,profiler`). Sweeping N is done by calling repeatedly with different `--subset`.

- [ ] **Step 1: Write the driver**

```python
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


def _subprocess_src(out: str, paths: list[str], reference: str | None, threads: int | None) -> str:
    ref = repr(reference) if reference else "None"
    noref = reference is None
    return _RUNNER.format(out=out, paths=paths, ref=ref, noref=noref, threads=threads)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chrom", default="chr1")
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--subset", type=int, default=None, help="use first K files")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--profiler", choices=["time", "memray", "none"], default="time")
    ap.add_argument("--results", type=Path, default=Path("results.csv"))
    a = ap.parse_args()

    paths = [p for p in a.manifest.read_text().split() if p]
    if a.subset:
        paths = paths[: a.subset]
    src = _subprocess_src(str(a.out), paths, str(a.reference) if a.reference else None, a.threads)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        runner = fh.name

    wall_s = maxrss_kb = None
    if a.profiler == "memray":
        subprocess.run(["memray", "run", "-o", "bench.memray", runner], check=True)
        print("memray profile: bench.memray (view with `memray flamegraph bench.memray`)")
    else:
        cmd = (["/usr/bin/time", "-v", sys.executable, runner]
               if a.profiler == "time" else [sys.executable, runner])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stderr.write(proc.stderr)
        if a.profiler == "time":
            m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
            e = re.search(r"Elapsed \(wall clock\).*?:\s*([\d:.]+)", proc.stderr)
            maxrss_kb = int(m.group(1)) if m else None
            if e:
                parts = [float(x) for x in e.group(1).split(":")]
                wall_s = parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)
        proc.check_returncode()

    new = not a.results.exists()
    with open(a.results, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["n_files", "wall_s", "maxrss_kb", "profiler"])
        w.writerow([len(paths), wall_s, maxrss_kb, a.profiler])
    print(f"n_files={len(paths)} wall_s={wall_s} maxrss_kb={maxrss_kb}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the README with all five profiler recipes**

```markdown
# from_vcf_list benchmark & profiling harness

## 1. Generate a synthetic cohort
python generate_cohort.py /tmp/cohort --n-files 2000 --n-variants 30000 \
    --contig chr1 --contig-len 1000000 --shared-frac 0.1 --indel-frac 0.1 --seed 0

## 2. RAM / wall-time sweep (Python entry, /usr/bin/time)
for k in 100 500 1000 2000; do
  python run_bench.py --manifest /tmp/cohort/manifest.txt --out /tmp/out_$k \
      --chrom chr1 --subset $k --profiler time --results results.csv
done

## 3. Python allocations (memray)
python run_bench.py --manifest /tmp/cohort/manifest.txt --out /tmp/out_m \
    --chrom chr1 --subset 500 --profiler memray
memray flamegraph bench.memray

## 4. Rust allocations (dhat) — native binary
cargo run --profile profiling --no-default-features --features conversion,dhat-heap \
    --bin bench_from_vcf_list -- /tmp/cohort/manifest.txt /tmp/out_dhat chr1
# open dhat-heap.json at https://nnethercote.github.io/dh_view/dh_view.html

## 5. CPU hotspots (perf) — native binary
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling \
    --no-default-features --features conversion --bin bench_from_vcf_list
perf record -g -- <target>/profiling/bench_from_vcf_list /tmp/cohort/manifest.txt /tmp/out_p chr1
perf report

## 6. Call counts (callgrind) — small N
valgrind --tool=callgrind <target>/profiling/bench_from_vcf_list /tmp/cohort_small/manifest.txt /tmp/out_c chr1
callgrind_annotate callgrind.out.*

## 7. Codegen of a hot fn (cargo-show-asm)
cargo asm --no-default-features --features conversion genoray_core::vcf_list_reader::...
```

- [ ] **Step 3: Smoke-test the driver**

Run:
```bash
python scripts/bench_from_vcf_list/generate_cohort.py /tmp/drv --n-files 6 --n-variants 15 --contig-len 5000 --seed 2
python scripts/bench_from_vcf_list/run_bench.py --manifest /tmp/drv/manifest.txt --out /tmp/drv_out --chrom chr1 --profiler time --results /tmp/drv/results.csv
```
Expected: prints `n_files=6 wall_s=... maxrss_kb=...`; `/tmp/drv/results.csv` has a header + one row. (`from_vcf_list` here uses `no_reference=True`.)

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_from_vcf_list/run_bench.py scripts/bench_from_vcf_list/README.md
git commit -m "feat(bench): from_vcf_list driver (time/memray) + profiler README"
```

---

### Task 4: Byte-identical parity fixture

**Files:**
- Create: `tests/test_svar2_from_vcf_list_parity.py`

**Interfaces:**
- Produces: `hash_store(store_dir: Path) -> str` (sha256 over sorted relative paths + file contents, skipping `meta.json`'s volatile fields is unnecessary — hash everything), and a test `test_parity_chunk_size_invariant` proving the merged store is identical across two `chunk_size` values (the invariant Task 6 relies on) and stable across repeat runs (the invariant Task 7 relies on).

- [ ] **Step 1: Write the parity test**

```python
# tests/test_svar2_from_vcf_list_parity.py
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path

from genoray import SparseVar2

_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=1000>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
)


def _ss(d: Path, i: int, rows: str) -> Path:
    plain = d / f"s{i}.vcf"
    plain.write_text(_HEADER.format(sample=f"S{i}") + rows)
    gz = d / f"s{i}.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def hash_store(store_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(store_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(store_dir).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _cohort(d: Path) -> list[str]:
    # 3 files, some shared sites (POS 100 shared across all), some private, one indel.
    a = _ss(d, 0, "chr1\t100\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\nchr1\t200\t.\tG\tT\t.\tPASS\t.\tGT\t1/1\n")
    b = _ss(d, 1, "chr1\t100\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\nchr1\t300\t.\tC\tCA\t.\tPASS\t.\tGT\t1/1\n")
    c = _ss(d, 2, "chr1\t100\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\nchr1\t400\t.\tT\tG\t.\tPASS\t.\tGT\t1/1\n")
    return [str(a), str(b), str(c)]


def test_parity_chunk_size_invariant(tmp_path: Path):
    paths = _cohort(tmp_path)
    out_small = tmp_path / "small"
    out_large = tmp_path / "large"
    SparseVar2.from_vcf_list(out_small, paths, no_reference=True, overwrite=True, chunk_size=1)
    SparseVar2.from_vcf_list(out_large, paths, no_reference=True, overwrite=True, chunk_size=25_000)
    assert hash_store(out_small) == hash_store(out_large)


def test_parity_repeatable(tmp_path: Path):
    paths = _cohort(tmp_path)
    o1, o2 = tmp_path / "a", tmp_path / "b"
    SparseVar2.from_vcf_list(o1, paths, no_reference=True, overwrite=True)
    SparseVar2.from_vcf_list(o2, paths, no_reference=True, overwrite=True)
    assert hash_store(o1) == hash_store(o2)
```

- [ ] **Step 2: Run it (verifies the current invariant holds today)**

Run: `pixi run pytest tests/test_svar2_from_vcf_list_parity.py -v`
Expected: PASS. If `test_parity_chunk_size_invariant` FAILS, the merged store is not chunk-size-invariant — STOP and report; Task 6's parity approach must change (fall back to a committed golden hash from the pre-change build instead).

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar2_from_vcf_list_parity.py
git commit -m "test(svar2): byte-identical parity fixture for from_vcf_list"
```

---

## Phase 1 — Baseline profiling gate (investigation; gates Phase 3)

### Task 5: Record the baseline

**Files:**
- Create: `docs/superpowers/plans/2026-07-15-svar2-from-vcf-list-scaling-baseline.md`

This task is investigation, not TDD. Do not change library code.

- [ ] **Step 1: Generate a mid-scale cohort** (`--n-files 2000 --n-variants 30000 --shared-frac 0.1 --indel-frac 0.1`).
- [ ] **Step 2: RAM/time sweep** over N ∈ {100, 500, 1000, 2000} with `run_bench.py --profiler time`; save `results.csv`.
- [ ] **Step 3: dhat run** at N=2000 via the native binary; capture peak heap + top allocation sites.
- [ ] **Step 4: perf run** at N=2000; capture the top self-time symbols (expect `VcfListRecordSource::next_record` / the cursor scan to dominate — S1).
- [ ] **Step 5: memray run** at N=500 on the Python entry; note the pre-flight header-scan cost.
- [ ] **Step 6: Write the baseline doc** — the RAM-vs-N slope (confirm/deny R1's linear dense-chunk term and R2's O(N) reader baseline as fractions of peak), the perf top-N (confirm S1), whether ledgers (R3) show up in dhat, and a per-fix expected-win table. **Explicitly decide** whether Phase 3's staged/parallel merge (R2+S2) is warranted by the numbers, and record the decision.
- [ ] **Step 7: Commit** the baseline doc (`docs:` commit).

**Gate:** Phase 3 tasks (9, 10) are concretized from this doc. Tasks 6 and 7 do **not** wait on it (both are code-confirmed) and may run in parallel with Task 5.

---

## Phase 2 — Code-confirmed fixes (Tasks 6 and 7 are independent; parallel)

### Task 6: Budget-derived `chunk_size` for `from_vcf_list` (R1)

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_vcf_list` signature + body, ~line 832 and ~line 979)
- Modify: `skills/genoray-api/SKILL.md`
- Test: `tests/test_svar2_from_vcf_list.py` (add one test)

**Interfaces:**
- Consumes: `_auto_chunk_size(n_samples, ploidy)` (existing, `python/genoray/_svar2.py:1296`).
- Produces: `from_vcf_list(..., chunk_size: int | None = None, ...)` — when `None`, uses `_auto_chunk_size(len(samples), ploidy)`; when set, used verbatim (back-compat). Must keep the parity fixture green (relies on chunk-size invariance).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_svar2_from_vcf_list.py
def test_from_vcf_list_auto_chunk_size(tmp_path, monkeypatch):
    import genoray._svar2 as sv2
    seen = {}
    real = sv2._core.run_vcf_list_conversion_pipeline
    def spy(paths, ref, contigs, out, samples, chunk_size, *rest):
        seen["chunk_size"] = chunk_size
        return real(paths, ref, contigs, out, samples, chunk_size, *rest)
    monkeypatch.setattr(sv2._core, "run_vcf_list_conversion_pipeline", spy)
    a = _ss(tmp_path, "a", "S0", "chr1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\n")
    b = _ss(tmp_path, "b", "S1", "chr1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\n")
    SparseVar2.from_vcf_list(tmp_path / "out", [a, b], no_reference=True, overwrite=True)
    assert seen["chunk_size"] == sv2._auto_chunk_size(2, 2)  # budget-derived, not 25_000
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py::test_from_vcf_list_auto_chunk_size -v`
Expected: FAIL (`chunk_size == 25000 != _auto_chunk_size(2,2)`).

- [ ] **Step 3: Change the default to budget-derived**

In `from_vcf_list`, change the signature `chunk_size: int = 25_000` → `chunk_size: int | None = None`, and just before the `_core.run_vcf_list_conversion_pipeline` call add:
```python
        if chunk_size is None:
            chunk_size = _auto_chunk_size(len(samples), ploidy)
```
Update the docstring: `chunk_size` now "defaults to a memory-budget-derived value (`_auto_chunk_size`), so a packed dense chunk stays ~`_DENSE_CHUNK_TARGET_BYTES` regardless of cohort size; pass an int to override."

- [ ] **Step 4: Run the new test + parity + existing suite**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py tests/test_svar2_from_vcf_list_parity.py -v`
Expected: PASS (parity green ⇒ output unchanged).

- [ ] **Step 5: Update `skills/genoray-api/SKILL.md`**

Update the `from_vcf_list` entry's `chunk_size` description to match the new default semantics (budget-derived when omitted).

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2.py tests/test_svar2_from_vcf_list.py skills/genoray-api/SKILL.md
git commit -m "perf(svar2): budget-derive from_vcf_list chunk_size so dense RAM is flat in N"
```

---

### Task 7: Frontier min-heap for O(log N) merge selection (S1)

**Files:**
- Modify: `src/vcf_list_reader.rs` (the `next_record` cursor-selection scan, ~lines 434–532, plus `FileCursor`/struct fields as needed)
- Test: `src/vcf_list_reader.rs` `#[cfg(test)]` module (add a randomized parity test vs the current scan)

**Interfaces:**
- Consumes: existing `FileCursor`, `HeapEntry`, `NormAtom`.
- Produces: identical `RawRecord` output stream (byte-identical store) with cursor selection in O(log N) via a second min-heap keyed by `(frontier, col)` with the **same first-min-wins tie-break** (unstarted `frontier == None` sorts before all `Some`, ties broken by smallest `col`).

- [ ] **Step 1: Write a randomized parity test (naive scan vs heap)**

Add a test that builds K synthetic in-memory cursors with random sorted positions, drains records via a reference `select_min_naive` (the current O(N) scan, copied into the test) and via the new heap-backed selection, and asserts the emitted `(pos, ilen, alt, cols)` sequences are identical over many seeds.

```rust
// in #[cfg(test)] mod tests
#[test]
fn frontier_heap_matches_naive_scan() {
    // Build several single-sample BCFs with overlapping + disjoint sorted
    // positions (reuse write_ss_vcf), run VcfListRecordSource::next_record to
    // exhaustion, and assert the full emitted record sequence equals a golden
    // captured from the pre-refactor scan (checked in as a Vec literal or
    // recomputed by a `select_min_naive` reference kept in this test module).
}
```

- [ ] **Step 2: Run to verify it fails / drives the design**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$ && pixi run bash -lc 'cargo test --no-default-features --features conversion --test-threads=1 frontier_heap_matches_naive_scan'`
Expected: initially FAIL or absent selection API — drives the implementation.

- [ ] **Step 3: Implement the frontier min-heap**

Add a `BinaryHeap<Reverse<FrontierKey>>` where `FrontierKey { frontier: Option<u32>, col: usize }` orders `None` before `Some` and ties by `col`. On `advance`, push the cursor's new frontier; on selection, pop the min. Keep `all_started`/`min_started`/`any_live` aggregates updated incrementally (or recompute cheaply) so the `releasable` gate is unchanged. Replace the O(N) `for (i, c) in self.cursors.iter().enumerate()` scan at `src/vcf_list_reader.rs:457` with a heap pop. Update the comment block to reflect the now-implemented heap.

- [ ] **Step 4: Run Rust suite + Python parity**

Run:
```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run bash -lc 'cargo test --no-default-features --features conversion'
pixi run maturin develop --profile profiling
pixi run pytest tests/test_svar2_from_vcf_list.py tests/test_svar2_from_vcf_list_parity.py tests/test_vcf_list_e2e.rs -v
```
Expected: all PASS (byte-identical parity green). Note: `tests/test_vcf_list_e2e.rs` runs under `cargo test`, not pytest.

- [ ] **Step 5: Confirm the win via the harness**

Run the perf recipe (README §5) at N=2000 before/after; record that `next_record` self-time dropped. Append the number to the baseline doc.

- [ ] **Step 6: Commit**

```bash
git add src/vcf_list_reader.rs
git commit -m "perf(svar2): frontier min-heap makes from_vcf_list merge selection O(log N)"
```

---

## Phase 3 — Measurement-gated fixes (concretize after Task 5)

> These tasks are **intentionally sketched, not fully specified** — the measure-first spec forbids committing final designs before the Task 5 baseline says which of R2/S2/R3 is a material fraction of peak RAM / wall-time. After Task 5, expand the warranted ones into full TDD tasks (same structure as Tasks 6–7) and drop the rest. This is a deliberate gate, not an omitted spec.

### Task 8: `max_mem` memory-budget knob (public API)

Add `max_mem: int | str | None` to `from_vcf_list` (parse via the existing memory-string helper in `_utils`), sizing `chunk_size` (and, if Task 9 lands, the merge batch size / max-open-files) from it. Update the docstring, `skills/genoray-api/SKILL.md`, and add a test asserting a small `max_mem` yields a smaller `chunk_size`. Document the resulting peak-RAM model. **Gate:** worth doing regardless, but its knobs depend on which Phase 3 fixes land.

### Task 9: Staged / parallel k-way merge (R2 + S2) — CONDITIONAL

Only if Task 5 shows the O(N) open-reader baseline (R2) or single-threaded decompression (S2) is a dominant fraction. Merge files in batches into intermediate sorted runs (caps open FDs → bounds R2) and decompress leaf batches on a rayon pool (fixes S2). Must stay byte-identical (parity fixture). Expand into full TDD tasks post-baseline; keep batch size tunable via Task 8's `max_mem`.

### Task 10: Ledger trim (R3) — CONDITIONAL

Only if dhat (Task 5) flags `var_key_ledgers`/`dense_ledgers` growth as material. Reduce per-chunk ledger retention (e.g. stream ledger rows to disk or shrink the row representation). Byte-identical output required.

---

## Self-Review

**Spec coverage:**
- Harness (spec Part A): generator = Task 1; native dhat binary = Task 2; Python driver + memray + perf/callgrind/cargo-show-asm recipes = Tasks 2–3; parity fixture = Task 4. ✓
- Baseline/measure-first = Task 5. ✓
- R1 = Task 6; S1 = Task 7; R2+S2 = Task 9; max_mem knob = Task 8; R3 = Task 10. ✓
- Success criteria (flat RAM, ~10× wall-time, byte-identical, 8 threads used) = validated by Task 5 harness + parity gate + perf checks in Tasks 5/7/9. ✓
- Non-goals (no cross-contig parallelism, no format change, no `./.` preservation) respected — no task touches them. ✓

**Placeholder scan:** Concrete code in every Phase 0–2 step. Phase 3 tasks are explicitly gated sketches (justified by measure-first), not hidden placeholders. ✓

**Type consistency:** `generate_cohort` signature identical across Task 1 test and impl; `hash_store` used consistently in Tasks 4/6/7; `chunk_size: int | None` consistent Task 6 ↔ parity; `run_vcf_list` argument order in Task 2 matches `src/orchestrator.rs:579`. ✓

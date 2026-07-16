# SVAR2 Parallel Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale single-contig VCF→SVAR2 and PGEN→SVAR2 conversion across cores behind one backend-agnostic sharding abstraction, choosing the load-balancing/architecture scheme from measurement rather than intuition.

**Architecture:** Build a benchmarking + profiling harness with a byte-identical output gate first, then re-establish the baseline. Extract PR #115's VCF sub-contig sharding into a backend-agnostic shard planner + work-stealing parallel reader/collector, add a PGEN adapter (variant-index sharding), and replace equal-basepair static splitting with over-decomposition + work stealing (Approach A, shared executor). A data gate then decides whether to promote to fully independent per-shard sub-pipelines (Approach B) only if the single executor becomes the bottleneck.

**Tech Stack:** Rust (rayon, crossbeam-channel, rust-htslib, pyo3), Python (genoray CLI, pgenlib, plink2), profiling via `perf`, `valgrind --tool=callgrind`/`cachegrind`, `cargo-show-asm`. Build/env via `pixi`.

**Design spec:** `docs/superpowers/specs/2026-07-15-svar2-parallel-conversion-design.md`

## Global Constraints

- **Byte-identical output** vs the current serial conversion is the invariant for every scheme. Gate with the `storehash` oracle on every bench row. No scheme ships that changes output bytes.
- **Coordinate convention:** 0-based, half-open `[start, end)`; left-align window is `crate::normalize::L_MAX`. Missing genotype = `-1`.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`). End every commit body with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **NFS build gotcha:** export `CARGO_TARGET_DIR=/tmp/genoray-target-$$` before any `cargo`/`maturin`/commit that triggers rust hooks (avoids the NFS linker bus error). See memory `genoray-nfs-linker-bus-error`.
- **Rust test build:** `cargo test` needs `--no-default-features --features conversion` (else the pyo3 test binary fails to link). Set `DYLD_FALLBACK_LIBRARY_PATH`/`LD_LIBRARY_PATH` to `$PWD/.pixi/envs/default/lib`. See memory `genoray-cargo-test-no-default-features`.
- **`test-rust` filters by TEST NAME, not file** — use `--test <file>` to target a file (a nonmatching name vacuously passes 0 tests). See memory `genoray-test-rust-name-filter-trap`.
- **prek hooks:** `.pre-commit-config.yaml` present — hooks must be installed (`pixi run prek-install`) before committing/pushing.
- **Public API policy:** any change to a name reachable from `import genoray` without underscores → update `skills/genoray-api/SKILL.md` in the same PR.
- **Long builds run FOREGROUND.** Never background a `cargo`/`maturin` build and return early; wait for it. See memory `sdd-implementers-background-long-builds`.
- **Benchmark environment:** `carter-compute` via `srun --overlap`, never the login node. Datasets and oracle live at `/carter/users/dlaub/svar_bench/` (`chr21.filt.bcf`, `gdc.chr21.filt.bcf`, `storehash.sh`, `oracle.chr21.hash`, `oracle.gdc.hash`). REF=`/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`. See memory `svar2-conversion-reader-bound`.
- **SVAR2 sharded decode hides missingness** — sharding must not attempt to reintroduce `-1`; a `./.` and hom-ref hap are indistinguishable in SVAR2. Do not change this. See memory `svar2-sparse-hides-missingness`.

---

## Prerequisite / starting point

This branch (`worktree-svar2-parallel-conversion`) was cut from `origin/main` and carries only the design spec commit. PR #115's perf commits are on the fork branch `bschilder/genoray:codex/svar2-subcontig-shards` and must be integrated before any implementation (Task 1).

---

## Task 1: Integrate and validate the PR #115 baseline

Bring PR #115's commits into the working branch and confirm they build, pass their own tests, and produce byte-identical output. This is the real starting point everything else extends.

**Files:**
- Modify (via merge): `src/chunk_assembler.rs`, `src/orchestrator.rs`, `src/vcf_reader.rs`, `src/vcf_list_reader.rs`, `src/budget.rs`, `src/lib.rs`, `python/genoray/_svar2.py`, `python/genoray/_cli/__main__.py`, `scripts/svar2_region_parallel_bench.py`, tests.

**Interfaces:**
- Produces: `crate::vcf_reader::VcfShard { fetch_start, fetch_end, own_start, own_end, ordinal }`, `crate::vcf_reader::plan_vcf_shards(regions: &[(u32,u32)], chrom: &str, max_shards: usize, target_bp: u32) -> Result<Vec<VcfShard>, ConversionError>`, `crate::vcf_reader::coalesce_fetch_regions`, `ChunkAssembler::with_reference(source, num_samples, ploidy, ref_seq: Arc<Vec<u8>>, has_reference, skip_out_of_scope, fields, owned_range: Option<(u32,u32)>)`, `crate::chunk_assembler::ChunkAssembler::read_next_chunk(chunk_size, chunk_id, pool: Option<&rayon::ThreadPool>)`.

- [ ] **Step 1: Fetch and merge the PR #115 fork branch**

```bash
cd /carter/users/dlaub/projects/genoray/.claude/worktrees/svar2-parallel-conversion
git fetch https://github.com/bschilder/genoray.git codex/svar2-subcontig-shards
git merge --no-edit FETCH_HEAD
```

Expected: clean merge (both branches share the `0ab5cf9` base; only the design doc is new on this side). Resolve any conflict in favor of keeping both the design doc and all PR #115 changes.

- [ ] **Step 2: Build the extension**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run maturin develop
```

Expected: builds successfully (foreground; wait for completion).

- [ ] **Step 3: Run PR #115's Rust + Python test suites**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run bash -lc 'export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion'
pixi run pytest tests/test_svar2_from_vcf.py tests/test_svar2_errors.py -q
```

Expected: all pass.

- [ ] **Step 4: Confirm byte-identical output on the real oracle (single contig, serial)**

```bash
# On a carter-compute holder via srun --overlap, NOT the login node:
cd /carter/users/dlaub/svar_bench
pixi run python run_svar2.py chr21.filt.bcf /tmp/svar2_baseline_chr21 /carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa 1
bash storehash.sh /tmp/svar2_baseline_chr21 | diff - oracle.chr21.hash
```

Expected: no diff (byte-identical to the recorded oracle).

- [ ] **Step 5: Commit the integration**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
git add -A
git commit -m "chore(svar2): integrate PR#115 sub-contig VCF sharding baseline"
```

---

## Phase 0 — Benchmarking & profiling harness

Tasks 2, 3, 4 are **parallelizable** (disjoint files: harness scaling logic vs PGEN dataset helper vs profiling docs/scripts). Task 5 depends on all three.

## Task 2: Extend the macro bench harness with scaling metrics + byte-identical gate

Add speedup/efficiency derivation, a `--backend {vcf,pgen}` switch, and a per-row byte-identical oracle check to the existing harness.

**Files:**
- Modify: `scripts/svar2_region_parallel_bench.py`
- Test: `tests/scripts/test_region_parallel_bench.py` (create)

**Interfaces:**
- Consumes: `genoray._core.run_conversion_pipeline`, `genoray._core.run_pgen_conversion_pipeline` (existing).
- Produces: `compute_scaling(rows: list[dict]) -> list[dict]` adding `speedup` and `efficiency` keys per row (relative to the same backend/chunk-size at `threads==min(threads)`); `oracle_hash(store: Path) -> str`; CLI flags `--backend`, `--oracle-check/--no-oracle-check`.

- [ ] **Step 1: Write the failing test for scaling derivation**

```python
# tests/scripts/test_region_parallel_bench.py
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "svar2_region_parallel_bench",
    Path(__file__).resolve().parents[2] / "scripts" / "svar2_region_parallel_bench.py",
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_compute_scaling_adds_speedup_and_efficiency():
    rows = [
        {"backend": "vcf", "chunk_size": 25000, "threads": 1, "wall_s": 100.0},
        {"backend": "vcf", "chunk_size": 25000, "threads": 4, "wall_s": 40.0},
    ]
    out = {(r["threads"]): r for r in bench.compute_scaling(rows)}
    assert out[1]["speedup"] == 1.0
    assert out[1]["efficiency"] == 1.0
    assert out[4]["speedup"] == 2.5
    assert abs(out[4]["efficiency"] - 0.625) < 1e-9
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pixi run pytest tests/scripts/test_region_parallel_bench.py -v
```

Expected: FAIL with `AttributeError: module ... has no attribute 'compute_scaling'`.

- [ ] **Step 3: Implement `compute_scaling` and `oracle_hash`**

```python
# scripts/svar2_region_parallel_bench.py
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
        out.append({**r, "speedup": speedup, "efficiency": speedup / (r["threads"] / base_t)})
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
```

- [ ] **Step 4: Wire `--backend` and `--oracle-check` into `main` / measurement loop**

Add `--backend` (choices `vcf`, `pgen`, default `vcf`) and `--oracle-check/--no-oracle-check` (default on) to the argparse config. In the per-measurement subprocess, after each conversion, when oracle-check is on, compute `oracle_hash(out_dir)` and compare against a reference store built once at `threads==1`; record `oracle_ok: bool` in the row and **fail the row loudly** if it differs. For `--backend pgen`, dispatch to `SparseVar2.from_pgen` instead of `from_vcf`.

- [ ] **Step 5: Run tests + a smoke conversion**

```bash
pixi run pytest tests/scripts/test_region_parallel_bench.py -v
pixi run python scripts/svar2_region_parallel_bench.py --make-synthetic \
  --synthetic-samples 4 --synthetic-variants 6 --carrier-stride 2 \
  --out-dir /tmp/genoray-bench-smoke --threads 1 2 --chunk-sizes 3 --repeats 1 --sample-count 2
pixi run ruff check scripts/svar2_region_parallel_bench.py && pixi run ruff format --check scripts/svar2_region_parallel_bench.py
```

Expected: test PASS; smoke run completes with `speedup`/`efficiency`/`oracle_ok` in the JSON output.

- [ ] **Step 6: Commit**

```bash
git add scripts/svar2_region_parallel_bench.py tests/scripts/test_region_parallel_bench.py
git commit -m "feat(bench): scaling metrics + byte-identical oracle gate in region bench"
```

## Task 3: Matched PGEN dataset generation helper

Generate a PGEN matched to a BCF via `plink2`, so VCF vs PGEN benchmarks are apples-to-apples.

**Files:**
- Create: `scripts/make_matched_pgen.py`
- Test: `tests/scripts/test_make_matched_pgen.py`

**Interfaces:**
- Produces: `plink2_cmd(bcf: Path, out_prefix: Path) -> list[str]` (pure command builder, unit-testable without plink2); `make_pgen(bcf: Path, out_prefix: Path) -> Path` (runs plink2, returns the `.pgen` path).

- [ ] **Step 1: Write the failing test for the command builder**

```python
# tests/scripts/test_make_matched_pgen.py
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_matched_pgen",
    Path(__file__).resolve().parents[2] / "scripts" / "make_matched_pgen.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_plink2_cmd_shape():
    cmd = mod.plink2_cmd(Path("/d/chr21.bcf"), Path("/out/chr21"))
    assert cmd[0] == "plink2"
    assert "--bcf" in cmd and "/d/chr21.bcf" in cmd
    assert "--make-pgen" in cmd
    assert cmd[cmd.index("--out") + 1] == "/out/chr21"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pixi run pytest tests/scripts/test_make_matched_pgen.py -v
```

Expected: FAIL (`No module named` / attribute missing).

- [ ] **Step 3: Implement**

```python
# scripts/make_matched_pgen.py
"""Build a PGEN matched to a BCF for apples-to-apples SVAR2 conversion benchmarks."""
from __future__ import annotations

import subprocess
from pathlib import Path


def plink2_cmd(bcf: Path, out_prefix: Path) -> list[str]:
    # multiallelics kept split as-is; SVAR2 atomizes downstream regardless.
    return [
        "plink2",
        "--bcf", str(bcf),
        "--make-pgen",
        "--out", str(out_prefix),
    ]


def make_pgen(bcf: Path, out_prefix: Path) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(plink2_cmd(bcf, out_prefix), check=True)
    return out_prefix.with_suffix(".pgen")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("bcf", type=Path)
    ap.add_argument("out_prefix", type=Path)
    args = ap.parse_args()
    print(make_pgen(args.bcf, args.out_prefix))
```

- [ ] **Step 4: Run tests**

```bash
pixi run pytest tests/scripts/test_make_matched_pgen.py -v
pixi run ruff check scripts/make_matched_pgen.py && pixi run ruff format --check scripts/make_matched_pgen.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_matched_pgen.py tests/scripts/test_make_matched_pgen.py
git commit -m "feat(bench): matched PGEN generation helper (plink2)"
```

## Task 4: Profiling recipes (perf / callgrind / cargo-asm)

Committed, scriptable recipes so profiling is reproducible.

**Files:**
- Create: `scripts/profile/perf_stat.sh`, `scripts/profile/perf_record.sh`, `scripts/profile/perf_sched.sh`, `scripts/profile/callgrind.sh`, `scripts/profile/cargo_asm.sh`
- Create: `docs/roadmap/svar2-conversion-profiling.md` (how to run + how to read each)

**Interfaces:**
- Each script takes `<bcf-or-pgen> <out-dir> <ref> <threads>` and wraps `run_svar2.py` (or `from_pgen`) so all profilers hit the identical workload.

- [ ] **Step 1: Write `perf_stat.sh`**

```bash
#!/usr/bin/env bash
# scripts/profile/perf_stat.sh <src> <out-dir> <ref> <threads>
# task-clock vs wall tells us whether N cores are actually busy.
set -euo pipefail
src=$1 out=$2 ref=$3 threads=$4
perf stat -e task-clock,context-switches,cpu-migrations,cache-misses,instructions,cycles \
  -- pixi run python /carter/users/dlaub/svar_bench/run_svar2.py "$src" "$out" "$ref" "$threads"
```

- [ ] **Step 2: Write `perf_record.sh` + `perf_sched.sh`**

```bash
#!/usr/bin/env bash
# scripts/profile/perf_record.sh <src> <out-dir> <ref> <threads>
# Confirms inflate/parse dominance and tracks the executor's share.
set -euo pipefail
src=$1 out=$2 ref=$3 threads=$4
perf record -g --call-graph dwarf -o perf.data \
  -- pixi run python /carter/users/dlaub/svar_bench/run_svar2.py "$src" "$out" "$ref" "$threads"
perf report -i perf.data --stdio | head -60
```

```bash
#!/usr/bin/env bash
# scripts/profile/perf_sched.sh <src> <out-dir> <ref> <threads>
# Off-CPU / scheduling: exposes collector serialization + channel stalls.
set -euo pipefail
src=$1 out=$2 ref=$3 threads=$4
perf sched record -o perf.sched.data \
  -- pixi run python /carter/users/dlaub/svar_bench/run_svar2.py "$src" "$out" "$ref" "$threads"
perf sched latency -i perf.sched.data | head -40
```

- [ ] **Step 3: Write `callgrind.sh` + `cargo_asm.sh`**

```bash
#!/usr/bin/env bash
# scripts/profile/callgrind.sh <src> <out-dir> <ref>
# Deterministic instruction-count / cache A/B on a SMALL input (serialized).
set -euo pipefail
src=$1 out=$2 ref=$3
valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
  pixi run python /carter/users/dlaub/svar_bench/run_svar2.py "$src" "$out" "$ref" 1
callgrind_annotate callgrind.out | head -60
```

```bash
#!/usr/bin/env bash
# scripts/profile/cargo_asm.sh <function-path>
# Inspect hot inner-loop codegen (inlining/vectorization). Requires cargo-show-asm.
set -euo pipefail
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
cargo asm --no-default-features --features conversion "$1"
```

- [ ] **Step 4: Write the profiling doc**

Create `docs/roadmap/svar2-conversion-profiling.md`: one section per script — what it answers, exact invocation, and how to read the output (e.g. "task-clock/wall ≈ threads means cores busy; << threads means serialization"; "watch `dense2sparse_vk` frames climb as shard count rises"; "callgrind is single-threaded — use it for per-work-item instruction cost, never scaling"). Cross-reference memory `svar2-conversion-reader-bound`.

- [ ] **Step 5: Make executable + commit**

```bash
chmod +x scripts/profile/*.sh
git add scripts/profile docs/roadmap/svar2-conversion-profiling.md
git commit -m "docs(bench): reproducible perf/callgrind/cargo-asm profiling recipes"
```

## Task 5: Re-establish the baseline profile (analysis)

Run the harness + recipes on current code (post PR #115) and record the numbers the data gate later depends on.

**Files:**
- Create: `docs/roadmap/svar2-conversion-baseline-2026-07-15.md` (results report)

- [ ] **Step 1: Macro scaling run (VCF + PGEN, both datasets)**

On a carter-compute holder via `srun --overlap`:

```bash
pixi run python scripts/make_matched_pgen.py /carter/users/dlaub/svar_bench/chr21.filt.bcf /tmp/pgen/chr21
pixi run python scripts/svar2_region_parallel_bench.py \
  --vcf /carter/users/dlaub/svar_bench/chr21.filt.bcf \
  --reference /carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa \
  --regions chr21:1-50000000 --out-dir /scratch/$USER/bench-vcf \
  --threads 1 2 4 8 16 32 --chunk-sizes 25000 --backend vcf
# repeat with --backend pgen against /tmp/pgen/chr21.pgen and the gdc dataset
```

- [ ] **Step 2: Micro profile at 1 and 16 threads**

```bash
bash scripts/profile/perf_record.sh /carter/users/dlaub/svar_bench/chr21.filt.bcf /tmp/p1 $REF 1
bash scripts/profile/perf_record.sh /carter/users/dlaub/svar_bench/chr21.filt.bcf /tmp/p16 $REF 16
bash scripts/profile/perf_sched.sh /carter/users/dlaub/svar_bench/chr21.filt.bcf /tmp/ps16 $REF 16
```

- [ ] **Step 3: Record the report**

In `docs/roadmap/svar2-conversion-baseline-2026-07-15.md`, capture: (a) speedup + efficiency tables for VCF and PGEN on both datasets; (b) confirmation of htslib-input dominance at 1 thread; (c) **the executor (`dense2sparse_vk`) share at 1 vs 16 threads** — the number the Task 11 gate uses; (d) any collector/channel stalls from `perf sched`. Note byte-identical (`oracle_ok`) held on every row.

- [ ] **Step 4: Commit**

```bash
git add docs/roadmap/svar2-conversion-baseline-2026-07-15.md
git commit -m "docs(bench): re-established SVAR2 conversion baseline profile"
```

---

## Phase 1 — Backend-agnostic shard abstraction (VCF)

## Task 6: Extract a backend-agnostic shard planner

Generalize `plan_vcf_shards` into a `WorkUnit` planner reusable by VCF (position range) and PGEN (variant-index range). VCF keeps its exact current behavior.

**Files:**
- Create: `src/shard.rs`
- Modify: `src/lib.rs` (add `mod shard;`), `src/vcf_reader.rs` (re-express `plan_vcf_shards` via `shard::plan_ranges`)
- Test: in `src/shard.rs` `#[cfg(test)]`

**Interfaces:**
- Produces:
  ```rust
  pub struct WorkUnit { pub own_start: u32, pub own_end: u32, pub fetch_start: u32, pub fetch_end: u32, pub ordinal: usize }
  /// Split coalesced, sorted, disjoint `owned` ranges into ordered units of ~`target_span`,
  /// capped at `max_shards`, padding fetch by `pad` (saturating) on each side.
  pub fn plan_ranges(owned: &[(u32,u32)], max_shards: usize, target_span: u32, pad: u32) -> Vec<WorkUnit>
  ```
- Consumes: nothing new. `crate::vcf_reader::VcfShard` becomes a thin conversion `From<WorkUnit>` (or `VcfShard` is replaced by `WorkUnit` — keep the field names identical: `fetch_start/fetch_end/own_start/own_end/ordinal`).

- [ ] **Step 1: Write failing tests (port PR #115's planner tests to `plan_ranges`)**

```rust
// src/shard.rs  (bottom)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_owned_ranges_with_padded_fetches() {
        let u = plan_ranges(&[(0, 12)], 3, 4, 5);
        assert_eq!(
            u.iter().map(|s| (s.own_start, s.own_end, s.ordinal)).collect::<Vec<_>>(),
            vec![(0, 4, 0), (4, 8, 1), (8, 12, 2)]
        );
        assert_eq!(u[0].fetch_start, 0);
        assert!(u[0].fetch_end >= u[1].own_start);
        assert!(u[1].fetch_start <= u[0].own_end);
    }

    #[test]
    fn max_shards_is_an_upper_bound() {
        let u = plan_ranges(&[(0, 100)], 4, 1, 0);
        assert_eq!(u.len(), 4);
        assert_eq!(
            u.iter().map(|s| (s.own_start, s.own_end)).collect::<Vec<_>>(),
            vec![(0, 25), (25, 50), (50, 75), (75, 100)]
        );
    }

    #[test]
    fn empty_in_empty_out() {
        assert!(plan_ranges(&[], 4, 10, 5).is_empty());
    }
}
```

- [ ] **Step 2: Run to verify fail**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run bash -lc 'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion --test-threads 1 shard 2>&1 | tail -20'
```

Expected: compile error (`shard` module / `plan_ranges` missing).

- [ ] **Step 3: Implement `plan_ranges`** (mirror PR #115's `plan_vcf_shards` span math, minus the chrom/error handling which stays in the VCF adapter)

```rust
// src/shard.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkUnit {
    pub own_start: u32,
    pub own_end: u32,
    pub fetch_start: u32,
    pub fetch_end: u32,
    pub ordinal: usize,
}

pub fn plan_ranges(owned: &[(u32, u32)], max_shards: usize, target_span: u32, pad: u32) -> Vec<WorkUnit> {
    if owned.is_empty() {
        return Vec::new();
    }
    let max_shards = max_shards.max(1);
    let total: u64 = owned.iter().map(|&(s, e)| u64::from(e - s)).sum();
    let span = total
        .div_ceil(max_shards as u64)
        .max(u64::from(target_span.max(1)))
        .min(u64::from(u32::MAX)) as u32;
    let mut out = Vec::new();
    for &(region_start, region_end) in owned {
        let mut own_start = region_start;
        while own_start < region_end {
            let own_end = own_start.saturating_add(span).min(region_end);
            out.push(WorkUnit {
                own_start,
                own_end,
                fetch_start: own_start.saturating_sub(pad),
                fetch_end: own_end.saturating_add(pad),
                ordinal: out.len(),
            });
            own_start = own_end;
        }
    }
    out
}
```

- [ ] **Step 4: Re-express `plan_vcf_shards` in terms of `plan_ranges`** — keep `coalesce_fetch_regions` + chrom error handling in `vcf_reader.rs`, then call `crate::shard::plan_ranges(&coalesced, max_shards, target_bp, crate::normalize::L_MAX)` and map to `VcfShard` (or return `WorkUnit` directly and update callers). Keep PR #115's `vcf_reader` planner tests green.

- [ ] **Step 5: Run all planner tests**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run bash -lc 'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion shard vcf_reader 2>&1 | tail -20'
```

Expected: PASS (both `shard::tests` and the ported vcf_reader planner tests).

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
git add src/shard.rs src/lib.rs src/vcf_reader.rs
git commit -m "refactor(svar2): extract backend-agnostic shard planner"
```

## Task 7: Parallel reader/collector with reorder buffer (VCF adapter)

Replace `read_vcf_shards_to_dense`'s per-shard SPSC + serial-ordinal drain with a fixed worker pool that reuses readers across a shared work queue, and a reorder buffer that emits DenseChunks in global `(ordinal, local)` order.

**Files:**
- Create: `src/shard_exec.rs` (worker pool + reorder buffer, backend-agnostic over a reader factory)
- Modify: `src/orchestrator.rs` (call `shard_exec::run` from the VCF branch), `src/lib.rs` (`mod shard_exec;`)
- Test: `src/shard_exec.rs` `#[cfg(test)]` (reorder buffer determinism) + rely on e2e byte-identical.

**Interfaces:**
- Consumes: `crate::shard::WorkUnit`, `ChunkAssembler::with_reference`.
- Produces:
  ```rust
  /// A worker converts one WorkUnit's fetch region into ordered DenseChunks.
  /// `make_assembler` builds a fresh `ChunkAssembler` (over a fresh RecordSource for the
  /// unit's fetch region) — VCF: indexed fetch; PGEN: variant-index range.
  pub fn run<F>(
      units: Vec<WorkUnit>,
      workers: usize,
      make_assembler: F,
      chunk_size: usize,
      tx_dense: &Sender<crate::types::DenseChunk>,
  ) -> Result<u64, ConversionError>
  where F: Fn(&WorkUnit) -> Result<ChunkAssembler, ConversionError> + Sync;
  ```
  Returns total `dropped_out_of_scope`. Reassigns `chunk.chunk_id` to a global monotonic counter in `(ordinal, local)` order via the reorder buffer.

- [ ] **Step 1: Write the failing reorder-buffer test**

```rust
// src/shard_exec.rs (bottom)
#[cfg(test)]
mod tests {
    use super::ReorderBuffer;

    #[test]
    fn emits_in_ordinal_order_despite_out_of_order_arrival() {
        // Two shards; shard 1 finishes its chunks before shard 0.
        let mut rb = ReorderBuffer::new(2);
        let mut emitted = Vec::new();
        // (ordinal, local, done)
        rb.push(1, 0, false, &mut |gid, tag| emitted.push((gid, tag)));
        rb.push(1, 1, false, &mut |gid, tag| emitted.push((gid, tag)));
        rb.push(1, 0, true, &mut |gid, tag| emitted.push((gid, tag))); // shard 1 done
        assert!(emitted.is_empty(), "must wait for shard 0");
        rb.push(0, 0, false, &mut |gid, tag| emitted.push((gid, tag)));
        rb.push(0, 0, true, &mut |gid, tag| emitted.push((gid, tag))); // shard 0 done -> flush 0 then 1
        assert_eq!(
            emitted,
            vec![(0, (0, 0)), (1, (1, 0)), (2, (1, 1))],
            "global ids 0,1,2 assigned in (ordinal, local) order"
        );
    }
}
```

- [ ] **Step 2: Run to verify fail**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run bash -lc 'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion shard_exec 2>&1 | tail -20'
```

Expected: compile error (`shard_exec` / `ReorderBuffer` missing).

- [ ] **Step 3: Implement `ReorderBuffer`** — buffers completed shards' chunk metadata; releases shard `k`'s chunks (assigning global ids) only once shards `0..k` are fully flushed. Then implement `run`: spawn `workers` threads, each pulling `WorkUnit`s from a shared `crossbeam::channel` work queue, building an assembler via `make_assembler`, producing local chunks, and sending `(ordinal, local, chunk, done)` to a bounded results channel drained by the collector into the `ReorderBuffer`, which forwards released chunks (with rewritten `chunk_id`) to `tx_dense`. Reader reuse: a worker reuses its `IndexedReader` by re-fetching the next unit's region (via `make_assembler` constructing over the same file — acceptable to reopen per unit for VCF given fetch cost dominates; note in a comment). On any error, set a shared cancel flag, drain, and return the first error with shard context (reuse PR #115's `with_vcf_shard_context`).

- [ ] **Step 4: Rewire the VCF branch in `orchestrator.rs`** — replace the `read_vcf_shards_to_dense` call with `shard_exec::run(units, processing_threads, make_assembler, chunk_size, &tx_dense)`, where `make_assembler` opens a `VcfRecordSource` over `unit.fetch_start..unit.fetch_end` and wraps it in `ChunkAssembler::with_reference(..., Some((unit.own_start, unit.own_end)))`. Delete `read_vcf_shards_to_dense` and `send_vcf_shard_message`/`VcfShardMessage` (superseded).

- [ ] **Step 5: Build + unit test + e2e byte-identical**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run bash -lc 'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion shard_exec vcf 2>&1 | tail -30'
# left-align boundary is a Rust e2e test (a .rs file, not pytest):
pixi run bash -lc 'export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion --test test_left_align_e2e'
pixi run maturin develop
pixi run pytest tests/test_svar2_from_vcf.py -q
```

Expected: reorder-buffer test PASS; left-align boundary e2e PASS; from_vcf e2e PASS.

- [ ] **Step 6: Byte-identical gate on the real oracle (2 and 16 threads)**

```bash
# srun --overlap holder:
for t in 2 16; do
  pixi run python /carter/users/dlaub/svar_bench/run_svar2.py \
    /carter/users/dlaub/svar_bench/chr21.filt.bcf /tmp/svar2_t$t $REF $t
  bash /carter/users/dlaub/svar_bench/storehash.sh /tmp/svar2_t$t | diff - /carter/users/dlaub/svar_bench/oracle.chr21.hash
done
```

Expected: no diff at either thread count.

- [ ] **Step 7: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
git add src/shard_exec.rs src/orchestrator.rs src/lib.rs
git commit -m "perf(svar2): work-stealing shard collector with reorder buffer (VCF)"
```

---

## Phase 2 — PGEN sharding

## Task 8: PGEN shard planner (variant-index ranges + boundary padding)

Plan PGEN work units over the contig's `var_start..var_end` variant-index range, padding each unit with neighbor variants within `L_MAX` positions of the boundary (positions from `.pvar`).

**Files:**
- Create: `src/pgen_shard.rs`
- Modify: `src/lib.rs` (`mod pgen_shard;`)
- Test: `src/pgen_shard.rs` `#[cfg(test)]`

**Interfaces:**
- Produces:
  ```rust
  pub struct PgenUnit { pub own_lo: usize, pub own_hi: usize, pub fetch_lo: usize, pub fetch_hi: usize, pub ordinal: usize }
  /// `positions[i]` is the .pvar position of global variant index (var_start + i), sorted ascending.
  /// Split [0, positions.len()) into ~equal-count units (<= max_shards), each fetch-padded to include
  /// neighbor variants whose position is within `pad` of the owned boundary positions.
  pub fn plan_pgen_units(positions: &[u32], max_shards: usize, pad: u32) -> Vec<PgenUnit>
  ```

- [ ] **Step 1: Write failing tests**

```rust
// src/pgen_shard.rs (bottom)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_count_split_no_padding_when_gaps_wide() {
        // positions far apart => pad pulls in no neighbors.
        let pos = vec![0, 1000, 2000, 3000];
        let u = plan_pgen_units(&pos, 2, 5);
        assert_eq!(u.len(), 2);
        assert_eq!((u[0].own_lo, u[0].own_hi), (0, 2));
        assert_eq!((u[1].own_lo, u[1].own_hi), (2, 4));
        assert_eq!((u[0].fetch_lo, u[0].fetch_hi), (0, 2)); // no neighbor within 5bp
        assert_eq!((u[1].fetch_lo, u[1].fetch_hi), (2, 4));
    }

    #[test]
    fn pads_fetch_across_a_close_boundary() {
        // variant index 2 (pos 101) is within 5bp of index 1 (pos 100): shard 1 must fetch index 1.
        let pos = vec![0, 100, 101, 500];
        let u = plan_pgen_units(&pos, 2, 5);
        assert_eq!((u[1].own_lo, u[1].own_hi), (2, 4));
        assert_eq!(u[1].fetch_lo, 1); // pulled in the close left neighbor for left-align context
    }

    #[test]
    fn empty_in_empty_out() {
        assert!(plan_pgen_units(&[], 4, 5).is_empty());
    }
}
```

- [ ] **Step 2: Run to verify fail**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run bash -lc 'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion pgen_shard 2>&1 | tail -20'
```

Expected: compile error.

- [ ] **Step 3: Implement `plan_pgen_units`** — equal-count split into `min(max_shards, n)` units; for each unit, extend `fetch_lo` leftward while the previous variant's position is within `pad` of `positions[own_lo]`, and `fetch_hi` rightward while the next variant's position is within `pad` of `positions[own_hi-1]`. Owned-range filtering downstream still drops atoms outside `[own_lo, own_hi)` by index.

- [ ] **Step 4: Run tests**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run bash -lc 'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion pgen_shard 2>&1 | tail -20'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
git add src/pgen_shard.rs src/lib.rs
git commit -m "feat(svar2): PGEN variant-index shard planner with boundary padding"
```

## Task 9: PGEN adapter + Python P-readers-per-contig plumbing

Wire PGEN sharding through `shard_exec::run`, and have Python pass one reader per (contig, shard).

**Files:**
- Modify: `src/pgen_reader.rs` (accept an owned variant-index range + expose an owned-range filter analogous to VCF), `src/orchestrator.rs` (`SourceSpec::Pgen` branch → shard via `shard_exec::run`), `src/lib.rs` (`run_pgen_conversion_pipeline` signature: `readers: Vec<Vec<Py<PyAny>>>` — per contig, per shard), `python/genoray/_svar2.py` (`from_pgen` builds `P` readers per contig).
- Test: `tests/test_svar2_from_pgen.py` (create — boundary + multi-thread byte-identical), reuse `tests/test_left_align_e2e.rs` pattern.

**Interfaces:**
- Consumes: `crate::pgen_shard::plan_pgen_units`, `crate::shard_exec::run`, `ChunkAssembler::with_reference`.
- Produces: `PgenRecordSource::new(reader, pvar_path, var_start, var_end, num_samples, chunk_size)` unchanged, but the orchestrator now constructs one per unit over `[fetch_lo, fetch_hi)` and passes `owned_range` as *positions* to the assembler (the assembler already filters by position, and `.pvar` gives positions).

- [ ] **Step 1: Write the failing byte-identical multi-thread test**

```python
# tests/test_svar2_from_pgen.py
import subprocess
from pathlib import Path

import genoray
from genoray import SparseVar2


def _hash(store: Path) -> bytes:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(Path(store).rglob("*")):
        if p.is_file():
            h.update(p.relative_to(store).as_posix().encode()); h.update(p.read_bytes())
    return h.digest()


def test_pgen_sharded_matches_serial(tmp_path, small_pgen_fixture, reference_fixture):
    # small_pgen_fixture / reference_fixture: existing conftest fixtures (see tests/conftest.py)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    SparseVar2.from_pgen(serial, small_pgen_fixture, reference_fixture, threads=1)
    SparseVar2.from_pgen(parallel, small_pgen_fixture, reference_fixture, threads=8)
    assert _hash(serial) == _hash(parallel)
```

- [ ] **Step 2: Run to verify fail**

```bash
pixi run pytest tests/test_svar2_from_pgen.py -q
```

Expected: FAIL — either fixture missing (add to `tests/conftest.py` from existing PGEN test-data generation) or hashes differ (sharding not wired yet).

- [ ] **Step 3: Change `run_pgen_conversion_pipeline` to take per-contig, per-shard readers** — signature `readers: Vec<Vec<Py<PyAny>>>`; the PGEN branch reads the contig's `.pvar` positions for `var_start..var_end`, calls `plan_pgen_units(&positions, processing_threads, crate::normalize::L_MAX)`, then `shard_exec::run` with a `make_assembler` that pops the next reader for that shard ordinal and builds a `PgenRecordSource` over `[fetch_lo, fetch_hi)` wrapped in `ChunkAssembler::with_reference(..., Some((own_start_pos, own_end_pos)))`. When `units.len() <= 1`, fall back to the existing single-reader path.

- [ ] **Step 4: Update `from_pgen` in `_svar2.py`** — build `readers = [[PgenReader(bytes(source), n_samples, allele_idx_offsets=aio) for _ in range(max_shards)] for _ in contigs]` where `max_shards` mirrors the Rust `processing_threads` budget (pass the same `threads`/budget hint, or over-provision to a safe cap and let Rust use `<= units.len()`). Update the SKILL.md if `from_pgen`'s signature changes (it should not — `threads` already exists).

- [ ] **Step 5: Build + e2e + boundary**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run maturin develop
pixi run pytest tests/test_svar2_from_pgen.py -q
pixi run bash -lc 'export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion --test test_left_align_e2e'
```

Expected: PASS (sharded PGEN byte-identical to serial; left-align boundary intact).

- [ ] **Step 6: Byte-identical gate on a real PGEN (matched from BCF)**

```bash
# srun --overlap holder:
pixi run python scripts/make_matched_pgen.py /carter/users/dlaub/svar_bench/chr21.filt.bcf /tmp/pgen/chr21
pixi run python -c "from genoray import SparseVar2; SparseVar2.from_pgen('/tmp/s1','/tmp/pgen/chr21.pgen','$REF',threads=1)"
pixi run python -c "from genoray import SparseVar2; SparseVar2.from_pgen('/tmp/s16','/tmp/pgen/chr21.pgen','$REF',threads=16)"
diff <(bash /carter/users/dlaub/svar_bench/storehash.sh /tmp/s1) <(bash /carter/users/dlaub/svar_bench/storehash.sh /tmp/s16)
```

Expected: no diff.

- [ ] **Step 7: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
git add src/pgen_reader.rs src/orchestrator.rs src/lib.rs python/genoray/_svar2.py tests/test_svar2_from_pgen.py tests/conftest.py
git commit -m "feat(svar2): shard PGEN conversion by variant-index range"
```

---

## Phase 3 — Load balancing (Approach A) + decision gate

## Task 10: Over-decomposition + variant-count-aware sizing

Tune the planner to emit more units than workers and size them toward equal work, so density skew is absorbed by work stealing.

**Files:**
- Modify: `src/shard.rs` / `src/pgen_shard.rs` (accept an over-decomposition factor), `src/orchestrator.rs` (pass `processing_threads * OVERSHARD_FACTOR` as `max_shards`), add `const OVERSHARD_FACTOR` with a comment tying it to Task 5's imbalance data.
- Test: `src/shard.rs` (unit count > workers), plus a bench re-run.

**Interfaces:**
- Consumes: existing planners. Produces: no new public names; `max_shards` at call sites becomes `workers * OVERSHARD_FACTOR`.

- [ ] **Step 1: Write failing test — over-decomposition emits more units than workers**

```rust
// src/shard.rs tests
#[test]
fn over_decomposes_beyond_worker_count() {
    // workers=4, factor=4 => up to 16 units over a big contig.
    let u = plan_ranges(&[(0, 16000)], 16, 1000, 5);
    assert_eq!(u.len(), 16);
}
```

- [ ] **Step 2: Run to verify fail/pass** (this may already pass given Task 6 math)

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
pixi run bash -lc 'export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"; cargo test --no-default-features --features conversion shard 2>&1 | tail -10'
```

Expected: confirms `plan_ranges` honors `max_shards` as the over-decomposed cap.

- [ ] **Step 3: Set `OVERSHARD_FACTOR` at call sites** — in `orchestrator.rs`, pass `processing_threads.saturating_mul(OVERSHARD_FACTOR)` (start at 4) as `max_shards` for both VCF and PGEN. For VCF, `target_bp` still floors unit size so tiny contigs don't over-split. For PGEN, equal-count split already balances; over-decomposition just improves stealing granularity.

- [ ] **Step 4: Re-run scaling bench + record efficiency delta**

```bash
# srun --overlap: compare efficiency at 16/32 threads vs Task 5 baseline (equal-BP, no stealing)
pixi run python scripts/svar2_region_parallel_bench.py --vcf /carter/users/dlaub/svar_bench/chr21.filt.bcf \
  --reference $REF --regions chr21:1-50000000 --out-dir /scratch/$USER/bench-a --threads 1 8 16 32 --backend vcf
```

Expected: parallel efficiency at 16/32 threads improves vs baseline on the density-skewed contig; `oracle_ok` true on every row.

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
git add src/shard.rs src/pgen_shard.rs src/orchestrator.rs
git commit -m "perf(svar2): over-decompose shards for work-stealing load balance"
```

## Task 11: Decision gate — measure executor share, decide A-final vs B

**Files:**
- Create: `docs/roadmap/svar2-conversion-decision-2026-07-15.md`

- [ ] **Step 1: Profile the executor share at high shard counts (post Approach A)**

```bash
bash scripts/profile/perf_record.sh /carter/users/dlaub/svar_bench/gdc.chr21.filt.bcf /tmp/pA16 $REF 16
bash scripts/profile/perf_record.sh /carter/users/dlaub/svar_bench/gdc.chr21.filt.bcf /tmp/pA32 $REF 32
# read the dense2sparse_vk (executor) frame share from perf report
```

- [ ] **Step 2: Record the decision**

In `docs/roadmap/svar2-conversion-decision-2026-07-15.md`: report Approach A's speedup/efficiency (VCF + PGEN, both datasets) and the executor's wall-time share at 16/32 threads. **Gate:** if executor share > 25% of wall at 32 threads → proceed to Phase 4 (Approach B). Else finalize Approach A and mark Phase 4 as documented future work. State the decision explicitly.

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap/svar2-conversion-decision-2026-07-15.md
git commit -m "docs(svar2): Approach A results + executor-ceiling decision gate"
```

---

## Phase 4 — Approach B (GATED: only if Task 11 says executor > 25%)

> Execute Phase 4 **only** if Task 11's decision record selects Approach B. Otherwise skip to Phase 5. Each shard runs its own reader→executor→writer; the existing rayon merge stitches shard-partitioned chunk files.

## Task 12: Chunk identity `(shard_ordinal, local)` through writer + merge

**Files:**
- Modify: `src/types.rs` (`DenseChunk`/`SparseChunk` carry `shard_ordinal: usize` + `local_id: usize`, or a packed global id space reserved per shard), `src/writer.rs` (write chunk files under a `(ordinal, local)` naming), `src/merge.rs` + `src/dense_merge.rs` (iterate chunk files in `(ordinal, local)` sorted order).
- Test: `src/merge.rs` `#[cfg(test)]` — merge over out-of-order-written shard chunks reproduces the serial concatenation order.

- [ ] **Step 1: Write the failing merge-order test** (synthetic chunk files named `chunk_s{ordinal}_{local}_*`; assert merge reads them in global position order). *(Full test code: mirror the existing `merge_mini_sc` test setup in `src/merge.rs`, substituting the new naming.)*
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement the `(ordinal, local)` naming + sorted iteration** in writer + both merges; keep `num_chunks` = total across shards.
- [ ] **Step 4: Run merge tests + full e2e byte-identical (Approach A path must still pass with `workers=1`).**
- [ ] **Step 5: Commit** (`refactor(svar2): shard-partitioned chunk identity for merge`).

## Task 13: Per-shard sub-pipeline (reader→executor→writer)

**Files:**
- Modify: `src/shard_exec.rs` (a worker runs the full sub-pipeline for its unit, emitting SparseChunks + writing directly, instead of forwarding DenseChunks to a shared executor), `src/orchestrator.rs` (drop the shared executor/tx_dense when sharded; keep it for the single-unit fallback).
- Test: e2e byte-identical at multiple thread counts.

- [ ] **Step 1: Write the failing e2e** (assert 16-thread sharded output byte-identical to serial — reuses `tests/test_svar2_from_vcf.py` harness with `threads=16`; will fail until the executor is per-shard).
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement per-shard `run_compute_engine`** inside each worker (each owns its own `LongAlleleTableWriter` — see Task 14 for the merge), writing SparseChunks with `(ordinal, local)` ids.
- [ ] **Step 4: Build + e2e + real-oracle byte-identical (2/16/32 threads).**
- [ ] **Step 5: Commit** (`perf(svar2): per-shard sub-pipelines (independent executor per shard)`).

## Task 14: Long-allele table sharding + rebasing at merge

**Files:**
- Modify: `src/nrvk.rs` (`LongAlleleTableWriter` per shard), `src/orchestrator.rs` / `src/merge.rs` (concatenate per-shard long-allele bins in ordinal order; rebase each shard's var_key indel offsets by the cumulative byte offset of prior shards' long-allele bins).
- Test: `tests/test_atomize_e2e.rs` / a long-allele-specific e2e — an indel whose long ALT lands in shard `k>0` resolves to the correct bytes after rebasing.

- [ ] **Step 1: Write the failing test** (multi-shard input with a long ALT in a late shard; decode and assert the reconstructed ALT bytes match serial — extend `tests/test_atomize_e2e.rs`). *(Reuse that file's fixture-build + decode assertions; place the long allele past a shard boundary.)*
- [ ] **Step 2: Run to verify fail** (offsets point into the wrong bin without rebasing).
- [ ] **Step 3: Implement per-shard long-allele bins + cumulative-offset rebase at merge.**
- [ ] **Step 4: Run long-allele e2e + full byte-identical oracle (VCF + PGEN, 32 threads).**
- [ ] **Step 5: Commit** (`fix(svar2): rebase per-shard long-allele offsets at merge`).

## Task 15: Approach B final bench + report

- [ ] **Step 1: Full scaling bench (VCF + PGEN, both datasets, threads 1..32), oracle gate on.**
- [ ] **Step 2: Update `docs/roadmap/svar2-conversion-decision-2026-07-15.md`** with B's speedup/efficiency vs A and the new bottleneck (expect merge/IO). Commit (`docs(svar2): Approach B scaling results`).

---

## Phase 5 — Documentation

## Task 16: User + roadmap docs and SKILL.md sync

**Files:**
- Modify: `docs/source/svar.md` (note per-contig sharding for VCF + PGEN; unchanged public API), `docs/roadmap/svar-2.md` (mark the parallelization milestone), `skills/genoray-api/SKILL.md` (only if any public name/kwarg/behavior changed — expected: none beyond existing `threads`).

- [ ] **Step 1: Update `docs/source/svar.md`** — document that single-contig VCF and PGEN conversion now shard within a contig, driven by the thread budget; note `from_vcf_list` does not shard (fd cost); note that sharding does not change output or reintroduce missingness (memory `svar2-sparse-hides-missingness`).
- [ ] **Step 2: Review public surface** — grep for any changed public name; if `from_pgen`/`from_vcf` signatures are unchanged, add a one-line note to `SKILL.md` that conversion is intra-contig parallel; otherwise update the affected entries.
- [ ] **Step 3: Build docs locally / lint markdown; commit** (`docs(svar2): document intra-contig conversion sharding`).

- [ ] **Step 4: Open the draft PR**

```bash
git push -u origin worktree-svar2-parallel-conversion
gh pr create --draft --base main \
  --title "perf(svar2): generalized parallel conversion (VCF + PGEN) with load balancing" \
  --body "$(cat <<'EOF'
Builds on #115. Backend-agnostic sub-contig sharding for single-file VCF and PGEN,
work-stealing collector with a reorder buffer, over-decomposition load balancing,
and a benchmarking + profiling harness with a byte-identical gate. Approach B
(per-shard sub-pipelines) included only if the executor-ceiling gate fired.

Design: docs/superpowers/specs/2026-07-15-svar2-parallel-conversion-design.md
Plan: docs/superpowers/plans/2026-07-15-svar2-parallel-conversion.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Parallelization guidance (for the executor of this plan)

- **Phase 0:** Tasks 2, 3, 4 are independent (disjoint files) → dispatch in parallel via `superpowers:dispatching-parallel-agents`. Task 5 depends on 2+3+4.
- **Phase 1 → 2:** Task 7 depends on Task 6; Tasks 8/9 depend on Task 7 (they consume `shard_exec::run`). Task 8 (pure planner) can start in parallel with Task 7.
- **Phase 3:** sequential after Phase 2.
- **Phase 4:** gated + sequential (12 → 13 → 14 → 15).
- Per user policy: implementers are **Sonnet or weaker**; use Opus only for review and critical-failure fixes. Dispatch implementers **foreground-only** for cargo/maturin builds (memory `sdd-implementers-background-long-builds`), and force `cd` into this worktree with a `rev-parse` guard (memory `subagent-cwd-in-worktrees`).

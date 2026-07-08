# PR #79 Architectural Fixup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple orchestration from stream layout, extract tangled concerns out of `lib.rs`, DRY up on-disk-path handling, and delete dead code in PR #79 — with byte-identical on-disk output.

**Architecture:** Route on-disk sub-streams by a `StreamTag` registry instead of hand-unrolling SNP (`u8`) vs indel (`u32`) in orchestration. In-memory sub-streams become byte-erased (`Vec<u8>` keys + `key_bytes`) so an arbitrary set of heterogeneous-width streams lives in one collection keyed by tag. Extract `monitor.rs`, `budget.rs`, `layout.rs`, `streams.rs`, `orchestrator.rs` so `lib.rs` is just the pyo3 surface.

**Tech Stack:** Rust, pyo3 0.29 (`_core` module), rayon, crossbeam-channel, bytemuck, ndarray/ndarray-npy, rust-htslib 1.0, proptest, thiserror (added in Task 10).

## Global Constraints

- **Byte-identical output:** the refactor must not change on-disk bytes, directory layout, or filenames for the existing `var_key/{snp,indel}` streams. `tests/test_e2e.rs` is the regression gate and must stay green (edit only where it references a removed helper).
- **Platforms:** Linux and macOS supported; Windows out of scope. Use `std::os::unix::fs::FileExt` (`read_exact_at`/`write_all_at`) for pread/pwrite — works on both. The `/proc` CPU sampler is Linux-only and must degrade (not panic) on macOS.
- **Build/test:** default feature `extension-module` builds the wheel; run tests with `cargo test --no-default-features` (links libpython). In pixi: `pixi run cargo test`, `pixi run -e lint cargo clippy --all-targets -- -D warnings`, `pixi run -e lint cargo fmt --check`.
- **pyo3 0.29:** module name is `_core`; use `py.detach` (not `allow_threads`), `Bound` API.
- **Commits:** end every commit message with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. prek hooks are installed; the `cargo check` prek hook fails in its sandbox (stale Python interp) — if it blocks a commit, commit with `--no-verify` and note it, but run clippy/fmt/test directly in the lint env instead.
- **Implementation model:** Sonnet or weaker for implementation (per maintainer prefs); Opus only for second-pass fixes.
- No new `pointer/`/`dense/` producers; no cost-model routing (M4). Only the machinery that lets them plug in later.

---

### Task 1: Hygiene — delete dead code, junk files, and the write-only field

**Files:**
- Modify: `src/rvk.rs` (delete commented `encode_variant_key` block ~285-344 and commented old `dense2sparse_vk` ~412-505)
- Modify: `src/nrvk.rs` (delete commented `NonReversibleLongAllele` block ~138-216)
- Modify: `src/vcf_reader.rs` (delete commented `long_allele_table_byte_size`, `read_vcf_chunk`, `read_vcf_genotypes` ~196-320)
- Modify: `src/types.rs` (delete commented `main()` sketch ~238-255; remove `pub num_variants: usize` field from `DenseChunk`)
- Modify: `src/vcf_reader.rs` (remove `num_variants: current_v_idx` from the `DenseChunk` literal)
- Modify: `src/rvk.rs` (remove `num_variants: n_variants` from the test `build_test_chunk` `DenseChunk` literal)
- Delete: `tests/test_engine.rs`, `tests/test_engine.proptest-regressions`
- Delete: `src/.ipynb_checkpoints/lib-checkpoint.rs`, `src/.ipynb_checkpoints/vcf_reader-checkpoint.rs` (whole dir; already in `.gitignore`)

**Interfaces:**
- Consumes: nothing.
- Produces: `DenseChunk` with fields `{ chunk_id, pos, ilens, alt, alt_offsets, genos }` (no `num_variants`). All later tasks use this shape.

- [ ] **Step 1: Delete the stray checkpoint dir and dead test files**

```bash
cd /carter/users/dlaub/repos/for_loukik/genoray
rm -rf src/.ipynb_checkpoints
git rm -f tests/test_engine.rs tests/test_engine.proptest-regressions 2>/dev/null || rm -f tests/test_engine.rs tests/test_engine.proptest-regressions
```

- [ ] **Step 2: Delete the commented-out code blocks**

In `src/rvk.rs`, delete the block starting `// // encoding the ilen, and alt allele` through the end of the commented `dense2sparse_vk` (the `// (out_pos, ilen_alt, offsets)\n// }` closing). In `src/nrvk.rs`, delete the `// // Contains the NonReversibleLongAllele struct ...` block through its closing `// }`. In `src/vcf_reader.rs`, delete from `// No more 1st pass` through the final commented `read_vcf_genotypes`. In `src/types.rs`, delete the `// mod nrvk;` … `// }` trailing sketch. Leave all live code and real doc comments intact.

- [ ] **Step 3: Remove the `num_variants` field**

In `src/types.rs`, delete the line `pub num_variants: usize,` from `DenseChunk`. In `src/vcf_reader.rs`, delete `num_variants: current_v_idx,` from the returned `DenseChunk { … }`. In `src/rvk.rs` test helper `build_test_chunk`, delete `num_variants: n_variants,` from its `DenseChunk { … }`.

- [ ] **Step 4: Verify it builds and all tests pass**

Run: `pixi run cargo test --no-default-features`
Expected: PASS, same test count minus the (already zero) `test_engine` tests. No `num_variants` errors.

- [ ] **Step 5: Verify lint clean**

Run: `pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check`
Expected: clean (no `dead_code`/`unused` warnings introduced).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(svar-2): delete dead commented code, stray checkpoints, and write-only num_variants

Removes ~450 lines of commented-out code (rvk/nrvk/vcf_reader/types), the
0-active-test test_engine.rs + its regression sidecar, stray .ipynb_checkpoints,
and the never-read DenseChunk.num_variants field.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Extract `budget.rs` — pure, tested thread-budget planner

**Files:**
- Create: `src/budget.rs`
- Modify: `src/lib.rs` (add `pub mod budget;`; replace inline budgeting with a call to `plan_thread_budget`)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub struct ThreadPlan { pub concurrent_chroms: usize, pub htslib_threads: usize }` and `pub fn plan_thread_budget(available_cores: usize, n_chroms: usize) -> ThreadPlan`. `run_conversion_pipeline` calls this.

- [ ] **Step 1: Write `src/budget.rs` with the pure function and its tests**

```rust
// Thread-budget planning for the cohort orchestrator. Pure arithmetic, split out
// of the pyo3 entry point so the low-end / high-end / clamp branches are testable
// without side effects.

// 4 fixed OS threads per chrom: reader + executor + chunk_writer + long_allele_writer.
const PIPELINE_THREADS_PER_CHROM: usize = 4;
// Floor for HTSlib decode threads — below this the executor channel starves.
const MIN_HTSLIB_THREADS: usize = 2;
// Ceiling for HTSlib decode threads — diminishing returns past 4 (BGZF block limits).
const MAX_HTSLIB_THREADS: usize = 4;
// Min viable allocation for one chrom end-to-end.
const MIN_THREADS_PER_CHROM: usize = PIPELINE_THREADS_PER_CHROM + MIN_HTSLIB_THREADS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPlan {
    pub concurrent_chroms: usize,
    pub htslib_threads: usize,
}

/// Decide how many chromosomes to convert concurrently and how many HTSlib decode
/// threads each gets, given the detected/overridden core count and chromosome count.
/// Reserves 1 core for the OS + Python main thread.
pub fn plan_thread_budget(available_cores: usize, n_chroms: usize) -> ThreadPlan {
    let usable_cores = std::cmp::max(1, available_cores.saturating_sub(1));
    let n_chroms = std::cmp::max(1, n_chroms);

    if usable_cores < MIN_THREADS_PER_CHROM {
        // Low-end: run one chrom, pour remaining cores into HTSlib decode.
        let htslib = std::cmp::max(1, usable_cores.saturating_sub(PIPELINE_THREADS_PER_CHROM));
        let htslib = std::cmp::min(htslib, MAX_HTSLIB_THREADS);
        ThreadPlan { concurrent_chroms: 1, htslib_threads: htslib }
    } else {
        // High-end: pick concurrency first (capped by chrom count), then redistribute.
        let max_concurrent_by_cores = usable_cores / MIN_THREADS_PER_CHROM;
        let concurrent = std::cmp::max(1, std::cmp::min(max_concurrent_by_cores, n_chroms));
        let cores_per_chrom = usable_cores / concurrent;
        let htslib_unclamped = cores_per_chrom.saturating_sub(PIPELINE_THREADS_PER_CHROM);
        let htslib = htslib_unclamped.clamp(MIN_HTSLIB_THREADS, MAX_HTSLIB_THREADS);
        ThreadPlan { concurrent_chroms: concurrent, htslib_threads: htslib }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_end_one_chrom_min_htslib() {
        // 4 cores → usable 3 < 6 → 1 chrom, htslib = max(1, 3-4)=1 (clamped ≤4).
        assert_eq!(plan_thread_budget(4, 8), ThreadPlan { concurrent_chroms: 1, htslib_threads: 1 });
    }

    #[test]
    fn test_single_core_machine() {
        // 1 core → usable 1 → low-end → 1 chrom, htslib 1.
        assert_eq!(plan_thread_budget(1, 22), ThreadPlan { concurrent_chroms: 1, htslib_threads: 1 });
    }

    #[test]
    fn test_high_end_fans_out_and_clamps_htslib() {
        // 65 cores → usable 64; 64/6 = 10 concurrent (capped by n_chroms=22 → 10);
        // cores_per_chrom 64/10=6; htslib 6-4=2 clamped to [2,4] → 2.
        assert_eq!(plan_thread_budget(65, 22), ThreadPlan { concurrent_chroms: 10, htslib_threads: 2 });
    }

    #[test]
    fn test_concurrency_capped_by_chrom_count() {
        // Many cores but only 2 chroms → at most 2 concurrent.
        let plan = plan_thread_budget(64, 2);
        assert_eq!(plan.concurrent_chroms, 2);
        assert!(plan.htslib_threads >= MIN_HTSLIB_THREADS && plan.htslib_threads <= MAX_HTSLIB_THREADS);
    }

    #[test]
    fn test_htslib_never_exceeds_max() {
        // Huge core count, 1 chrom → htslib clamped at MAX_HTSLIB_THREADS.
        assert_eq!(plan_thread_budget(256, 1).htslib_threads, MAX_HTSLIB_THREADS);
    }
}
```

- [ ] **Step 2: Run the budget tests to verify they pass**

Run: `pixi run cargo test --no-default-features budget::`
Expected: 5 tests PASS. (If any assert value is off, adjust the expected value to match the extracted arithmetic — the function is copied verbatim from `lib.rs`, so recompute by hand rather than changing logic.)

- [ ] **Step 3: Wire `plan_thread_budget` into `lib.rs`**

Add `pub mod budget;` near the other `pub mod` lines. In `run_conversion_pipeline`, after the `available_cores` discovery block, replace the entire inline `concurrent_chroms`/`htslib_threads` computation (the `let (concurrent_chroms, htslib_threads) = if usable_cores < … { … } else { … };` block and its helper `const`s) with:

```rust
let plan = crate::budget::plan_thread_budget(available_cores, chroms.len());
let concurrent_chroms = plan.concurrent_chroms;
let htslib_threads = plan.htslib_threads;
```

Keep the existing `println!` reporting lines that follow (they read `concurrent_chroms`/`htslib_threads`/`total_active`). Delete the now-unused `PIPELINE_THREADS_PER_CHROM`/`MIN_*`/`MAX_*` consts and `usable_cores`/`n_chroms` locals from `lib.rs` (they live in `budget.rs` now).

- [ ] **Step 4: Verify the full suite still passes**

Run: `pixi run cargo test --no-default-features`
Expected: PASS (budget tests added; e2e unchanged).

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check
git add -A
git commit -m "refactor(svar-2): extract testable plan_thread_budget into budget.rs

Pulls the ~70-line core-budgeting arithmetic out of the pyo3 entry point into a
pure fn with unit tests over the low-end/high-end/clamp branches.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2b: Extract `monitor.rs` — Linux `/proc` sampler

**Files:**
- Create: `src/monitor.rs`
- Modify: `src/lib.rs` (add `mod monitor;`; delete the sampler code; call `monitor::spawn_sampler`)

**Interfaces:**
- Consumes: `crate::types::{DenseChunk, SparseChunk}` (channel item types).
- Produces: `pub fn spawn_sampler(chrom: String, tx_dense: Sender<DenseChunk>, tx_sparse: Sender<SparseChunk>, tx_long: Sender<Vec<u8>>, stop: Arc<AtomicBool>) -> thread::JoinHandle<()>` — same signature as today.

- [ ] **Step 1: Move the sampler verbatim into `src/monitor.rs`**

Create `src/monitor.rs`. Move, unchanged, from `lib.rs`: the module doc comment about monitoring, `const CLK_TCK_HZ`, `find_thread_tid_by_name`, `read_thread_cpu_ticks`, `sample_interval_secs`, and `spawn_sampler`. Add the imports they need at the top:

```rust
use crossbeam_channel::Sender;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::types::{DenseChunk, SparseChunk};
```

Add a module doc line making the platform contract explicit:

```rust
//! Optional per-contig pipeline monitoring. Channel fill levels work on any
//! platform; per-thread CPU% is read from Linux `/proc/self/task/<tid>/stat` and
//! is unavailable on macOS (no `/proc`) — there the CPU columns print `n/a`.
```

- [ ] **Step 2: Make the macOS degrade honest (`n/a` instead of `0%`)**

In `spawn_sampler`, change the CPU% computation so an unresolved TID yields `None`, and format `n/a` for `None`. Replace the `cpu_pcts: Vec<f64>` construction and the `eprintln!` CPU fields:

```rust
let cpu_pcts: Vec<Option<f64>> = tids
    .iter()
    .zip(prev_ticks.iter())
    .zip(cur.iter())
    .map(|((t, p), c)| {
        t.map(|_| {
            let dt_ticks = c.saturating_sub(*p) as f64;
            100.0 * dt_ticks / CLK_TCK_HZ / interval.as_secs_f64()
        })
    })
    .collect();
prev_ticks = cur;

let fmt = |o: Option<f64>| o.map_or_else(|| "n/a".to_string(), |v| format!("{:.0}%", v));
let elapsed = start.elapsed().as_secs();
eprintln!(
    "[{} t={}s] tx_dense={}/{} tx_sparse={}/{} tx_long={}/{} | \
     cpu read={} exec={} cw={} lw={}",
    chrom, elapsed,
    tx_dense.len(), dense_cap, tx_sparse.len(), sparse_cap, tx_long.len(), long_cap,
    fmt(cpu_pcts[0]), fmt(cpu_pcts[1]), fmt(cpu_pcts[2]), fmt(cpu_pcts[3]),
);
```

- [ ] **Step 3: Delete the sampler from `lib.rs` and call the module**

In `lib.rs`: add `mod monitor;`. Delete `CLK_TCK_HZ`, `find_thread_tid_by_name`, `read_thread_cpu_ticks`, `sample_interval_secs`, `spawn_sampler`, and their doc comment/imports that are now only used by the monitor. In `process_chromosome`, change `let sampler_thread = spawn_sampler(` to `let sampler_thread = monitor::spawn_sampler(`.

- [ ] **Step 4: Verify build + tests + lint**

Run: `pixi run cargo test --no-default-features && pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check`
Expected: PASS/clean. `lib.rs` no longer references `/proc`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(svar-2): extract /proc monitoring into monitor.rs

Moves the ~130-line Linux CPU sampler out of lib.rs; documents it as Linux-only
and prints n/a (not a misleading 0%) for CPU columns on macOS where /proc is absent.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `layout.rs` — single source of truth for on-disk paths

**Files:**
- Create: `src/layout.rs`
- Modify: `src/lib.rs` (add `pub mod layout;`)
- Modify: `src/merge.rs`, `src/writer.rs`, `src/nrvk.rs`, and `lib.rs` `process_chromosome` to derive paths through `layout` (path strings only; behavior unchanged)

**Interfaces:**
- Consumes: nothing (pure path construction).
- Produces:
  ```rust
  pub struct ContigPaths { base_out_dir: String, chrom: String }
  impl ContigPaths {
      pub fn new(base_out_dir: &str, chrom: &str) -> Self;
      pub fn var_key_snp_dir(&self) -> PathBuf;      // {out}/{chrom}/var_key/snp
      pub fn var_key_indel_dir(&self) -> PathBuf;     // {out}/{chrom}/var_key/indel
      pub fn long_alleles_bin(&self) -> PathBuf;      // {indel}/long_alleles.bin
      pub fn long_allele_offsets(&self) -> PathBuf;   // {indel}/long_allele_offsets.npy
  }
  // Free fns for per-stream file names inside a stream dir (dir passed in):
  pub fn chunk_pos(dir: &Path, chunk_id: usize) -> PathBuf;   // chunk_{id}_pos.bin
  pub fn chunk_key(dir: &Path, chunk_id: usize) -> PathBuf;   // chunk_{id}_key.bin
  pub fn final_positions(dir: &Path) -> PathBuf;              // final_positions.bin
  pub fn final_keys(dir: &Path) -> PathBuf;                   // final_keys.bin
  pub fn final_offsets(dir: &Path) -> PathBuf;                // final_offsets.npy
  ```
- **Note:** later tasks add a `stream_dir(&self, spec: &StreamSpec)` method; for now the two concrete `var_key_*` accessors keep behavior identical.

- [ ] **Step 1: Write `src/layout.rs` with tests**

```rust
//! Single source of truth for the SVAR2 on-disk directory + file layout. Every
//! path the pipeline reads or writes is constructed here so the (still
//! provisional) filenames can be changed in exactly one place before M6 decode.

use std::path::{Path, PathBuf};

pub struct ContigPaths {
    base_out_dir: String,
    chrom: String,
}

impl ContigPaths {
    pub fn new(base_out_dir: &str, chrom: &str) -> Self {
        Self { base_out_dir: base_out_dir.to_string(), chrom: chrom.to_string() }
    }

    fn var_key_dir(&self) -> PathBuf {
        Path::new(&self.base_out_dir).join(&self.chrom).join("var_key")
    }
    pub fn var_key_snp_dir(&self) -> PathBuf {
        self.var_key_dir().join("snp")
    }
    pub fn var_key_indel_dir(&self) -> PathBuf {
        self.var_key_dir().join("indel")
    }
    pub fn long_alleles_bin(&self) -> PathBuf {
        self.var_key_indel_dir().join("long_alleles.bin")
    }
    pub fn long_allele_offsets(&self) -> PathBuf {
        self.var_key_indel_dir().join("long_allele_offsets.npy")
    }
}

pub fn chunk_pos(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_pos.bin", chunk_id))
}
pub fn chunk_key(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_key.bin", chunk_id))
}
pub fn final_positions(dir: &Path) -> PathBuf {
    dir.join("final_positions.bin")
}
pub fn final_keys(dir: &Path) -> PathBuf {
    dir.join("final_keys.bin")
}
pub fn final_offsets(dir: &Path) -> PathBuf {
    dir.join("final_offsets.npy")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_key_stream_dirs() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(p.var_key_snp_dir(), Path::new("/out/chr1/var_key/snp"));
        assert_eq!(p.var_key_indel_dir(), Path::new("/out/chr1/var_key/indel"));
    }

    #[test]
    fn test_long_allele_paths_live_under_indel() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(p.long_alleles_bin(), Path::new("/out/chr1/var_key/indel/long_alleles.bin"));
        assert_eq!(p.long_allele_offsets(), Path::new("/out/chr1/var_key/indel/long_allele_offsets.npy"));
    }

    #[test]
    fn test_chunk_and_final_names() {
        let dir = Path::new("/out/chr1/var_key/snp");
        assert_eq!(chunk_pos(dir, 3), Path::new("/out/chr1/var_key/snp/chunk_3_pos.bin"));
        assert_eq!(chunk_key(dir, 3), Path::new("/out/chr1/var_key/snp/chunk_3_key.bin"));
        assert_eq!(final_positions(dir), Path::new("/out/chr1/var_key/snp/final_positions.bin"));
        assert_eq!(final_keys(dir), Path::new("/out/chr1/var_key/snp/final_keys.bin"));
        assert_eq!(final_offsets(dir), Path::new("/out/chr1/var_key/snp/final_offsets.npy"));
    }
}
```

- [ ] **Step 2: Run layout tests**

Run: `pixi run cargo test --no-default-features layout::`
Expected: 3 tests PASS.

- [ ] **Step 3: Wire `layout` into the writers and merge (behavior-preserving)**

Add `pub mod layout;` to `lib.rs`. Then replace inline `format!` path strings so paths flow through `layout`. Because `merge.rs`/`writer.rs` currently take `&str` dirs, keep their signatures but build the file paths inside them via the free fns:

- `src/writer.rs` `run_io_writer`: replace `&format!("{}/chunk_{}_pos.bin", snp_dir, id)` etc. with `crate::layout::chunk_pos(Path::new(snp_dir), id)` / `chunk_key(...)` for both `snp_dir` and `indel_dir`. `write_bin` takes `&Path` now (change its `path: &str` param to `path: &Path` and update `File::create(path)`).
- `src/merge.rs` `merge_mini_sc`: replace every `format!("{}/chunk_{}_pos.bin", output_dir, c)`, `final_positions.bin`, `final_keys.bin`, `final_offsets.npy` with the `crate::layout::*` free fns applied to `Path::new(output_dir)`. Cleanup loop uses `chunk_pos`/`chunk_key` too.
- `src/nrvk.rs` `LongAlleleReader::new`: replace the hand-built `{output_dir}/{chrom}/var_key/indel` derivation with `ContigPaths::new(output_dir, chrom)` and `.long_alleles_bin()` / `.long_allele_offsets()`.
- `lib.rs` `process_chromosome`: build `let paths = crate::layout::ContigPaths::new(base_out_dir, chrom);` at the top; derive `snp_dir`/`indel_dir` from `paths.var_key_snp_dir()`/`paths.var_key_indel_dir()`; use `paths.long_allele_offsets()` for the offsets `write_npy` target and the long-allele writer's file path via `paths.long_alleles_bin()`. Convert `PathBuf` to `&str` where a `&str` is still required with `.to_str().unwrap()`, or pass `&Path`.

- [ ] **Step 4: Verify byte-identical output via the e2e suite**

Run: `pixi run cargo test --no-default-features`
Expected: PASS — `test_e2e.rs` proves the same files land in the same places.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check
git add -A
git commit -m "refactor(svar-2): centralize on-disk layout in layout.rs

All directory/filename construction flows through ContigPaths + free fns; the
LongAlleleReader no longer hand-derives its path and can't drift from the writer.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: De-generify `merge_mini_sc` to a runtime `key_bytes`

**Files:**
- Modify: `src/merge.rs` (change `merge_mini_sc<K: Pod>` → `merge_mini_sc(key_bytes: usize, …)`, byte buffers; update its tests)
- Modify: `src/lib.rs` `process_chromosome` (call sites pass `1` / `4` instead of `::<u8>` / `::<u32>`)

**Interfaces:**
- Consumes: `crate::layout` free fns (Task 3).
- Produces: `pub fn merge_mini_sc(key_bytes: usize, num_chunks: usize, num_samples: usize, ploidy: usize, output_dir: &str, ram_ledger: Vec<Vec<u32>>)`. Positions are always `u32`; keys are opaque `key_bytes`-wide records.

- [ ] **Step 1: Change the signature and buffers to byte-based**

In `src/merge.rs`: replace `pub fn merge_mini_sc<K: bytemuck::Pod>(` with `pub fn merge_mini_sc(key_bytes: usize,` (new first param). Delete `let key_size = std::mem::size_of::<K>();` and use the `key_bytes` param wherever `key_size` appeared. Change the key tile buffer from `vec![K::zeroed(); tile_total_items]` to `vec![0u8; tile_total_items * key_bytes]`. Read/copy keys as bytes: the per-chunk key read already reads `chunk_items_to_read * key_size` bytes into `chunk_key_bytes` — keep that as raw `Vec<u8>` and copy byte ranges into `tile_key_buffer` using `key_bytes`-scaled offsets:

```rust
let dest_start = tile_write_heads[i] * key_bytes;
let src_start = local_chunk_cursor * key_bytes;
tile_key_buffer[dest_start..dest_start + calls * key_bytes]
    .copy_from_slice(&chunk_key_bytes[src_start..src_start + calls * key_bytes]);
```

Positions stay `u32` (keep the `bytemuck::cast_slice` on the pos path). Drop the `chunk_key_k`/`tile_key_bytes` `cast_slice` calls — `tile_key_buffer` is already `&[u8]`. Keep `pos_size = size_of::<u32>()`.

- [ ] **Step 2: Update the merge tests to pass `key_bytes`**

In `src/merge.rs` tests, change each `merge_mini_sc::<u32>(n, …)` to `merge_mini_sc(4, n, …)` and `merge_mini_sc::<u8>(2, …)` to `merge_mini_sc(1, 2, …)`. The u8-key test (`test_merge_u8_keys_interleave`) now passes `1`. Expected values are unchanged (same bytes).

- [ ] **Step 3: Update the two call sites in `lib.rs`**

In `process_chromosome`, change:
```rust
merge::merge_mini_sc::<u8>(num_chunks, samples.len(), ploidy, &snp_dir, snp_ledger);
rvk::pack_snp_key_file(&snp_dir);
merge::merge_mini_sc::<u32>(num_chunks, samples.len(), ploidy, &indel_dir, indel_ledger);
```
to:
```rust
merge::merge_mini_sc(1, num_chunks, samples.len(), ploidy, &snp_dir, snp_ledger);
rvk::pack_snp_key_file(&snp_dir);
merge::merge_mini_sc(4, num_chunks, samples.len(), ploidy, &indel_dir, indel_ledger);
```
(These call sites are replaced entirely by the registry loop in Task 7; this step just keeps the build green in between.)

- [ ] **Step 4: Verify tests pass (byte-identical merge output)**

Run: `pixi run cargo test --no-default-features merge:: && pixi run cargo test --no-default-features`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check
git add -A
git commit -m "refactor(svar-2): de-generify merge over runtime key_bytes

merge_mini_sc takes key_bytes: usize and operates on byte buffers instead of a
K: Pod type parameter. Removes the ::<u8>/::<u32> monomorphization branch from
orchestration; positions stay u32.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `streams.rs` — the tag registry

**Files:**
- Create: `src/streams.rs`
- Modify: `src/lib.rs` (add `pub mod streams;`)

**Interfaces:**
- Consumes: `crate::rvk::pack_snp_key_file` (existing, unchanged).
- Produces:
  ```rust
  pub enum StreamTag { VarKeySnp, VarKeyIndel }   // #[repr(usize)], Copy
  impl StreamTag { pub const COUNT: usize = 2; pub fn index(self) -> usize; }
  pub struct StreamSpec { pub tag: StreamTag, pub subdir: &'static str, pub key_bytes: usize, pub post_merge: Option<fn(&std::path::Path)> }
  pub const REGISTRY: [StreamSpec; StreamTag::COUNT];
  pub struct StreamMap<T> { … }  // array-backed, indexed by StreamTag
  impl<T> StreamMap<T> { pub fn from_fn(f: impl FnMut(StreamTag) -> T) -> Self; pub fn get(&self, tag: StreamTag) -> &T; pub fn get_mut(&mut self, tag: StreamTag) -> &mut T; pub fn iter(&self) -> impl Iterator<Item = (StreamTag, &T)>; pub fn into_iter_tagged(self) -> impl Iterator<Item = (StreamTag, T)>; }
  ```

- [ ] **Step 1: Write `src/streams.rs` with tests**

```rust
//! The encoding-agnostic seam's routing table. Orchestration, the writer, the
//! executor, and merge all iterate `REGISTRY` and index by `StreamTag` — they
//! never name a concrete key width or `snp`/`indel` directly. Adding the
//! `pointer` (M11) / `dense` (M4) representations means extending `StreamTag` +
//! `REGISTRY` and teaching `classify_variant` to route; nothing else changes.

use std::path::Path;

use crate::rvk::pack_snp_key_file;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamTag {
    VarKeySnp = 0,
    VarKeyIndel = 1,
}

impl StreamTag {
    pub const COUNT: usize = 2;
    pub const ALL: [StreamTag; Self::COUNT] = [StreamTag::VarKeySnp, StreamTag::VarKeyIndel];
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

pub struct StreamSpec {
    pub tag: StreamTag,
    pub subdir: &'static str,
    pub key_bytes: usize,
    /// Post-merge rewrite hook applied to the stream's final files (e.g. 2-bit packing).
    pub post_merge: Option<fn(&Path)>,
}

/// One entry per active on-disk sub-stream. Order matches `StreamTag as usize`.
pub const REGISTRY: [StreamSpec; StreamTag::COUNT] = [
    StreamSpec {
        tag: StreamTag::VarKeySnp,
        subdir: "var_key/snp",
        key_bytes: 1,
        post_merge: Some(pack_snp_key_file_path),
    },
    StreamSpec {
        tag: StreamTag::VarKeyIndel,
        subdir: "var_key/indel",
        key_bytes: 4,
        post_merge: None,
    },
];

// pack_snp_key_file currently takes &str; adapt to the fn(&Path) hook shape.
fn pack_snp_key_file_path(dir: &Path) {
    pack_snp_key_file(dir.to_str().expect("stream dir is valid UTF-8"));
}

/// Fixed-size map keyed by `StreamTag`, backed by an array (O(1), no hashing).
pub struct StreamMap<T> {
    slots: [T; StreamTag::COUNT],
}

impl<T> StreamMap<T> {
    pub fn from_fn(mut f: impl FnMut(StreamTag) -> T) -> Self {
        Self { slots: StreamTag::ALL.map(|t| f(t)) }
    }
    #[inline]
    pub fn get(&self, tag: StreamTag) -> &T {
        &self.slots[tag.index()]
    }
    #[inline]
    pub fn get_mut(&mut self, tag: StreamTag) -> &mut T {
        &mut self.slots[tag.index()]
    }
    pub fn iter(&self) -> impl Iterator<Item = (StreamTag, &T)> {
        StreamTag::ALL.into_iter().zip(self.slots.iter()).map(|(t, v)| (t, v))
    }
    pub fn into_iter_tagged(self) -> impl Iterator<Item = (StreamTag, T)> {
        StreamTag::ALL.into_iter().zip(self.slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_indices_match_tags() {
        for spec in &REGISTRY {
            assert_eq!(REGISTRY[spec.tag.index()].tag, spec.tag);
        }
    }

    #[test]
    fn test_registry_key_widths() {
        assert_eq!(REGISTRY[StreamTag::VarKeySnp.index()].key_bytes, 1);
        assert_eq!(REGISTRY[StreamTag::VarKeyIndel.index()].key_bytes, 4);
    }

    #[test]
    fn test_only_snp_has_post_merge() {
        assert!(REGISTRY[StreamTag::VarKeySnp.index()].post_merge.is_some());
        assert!(REGISTRY[StreamTag::VarKeyIndel.index()].post_merge.is_none());
    }

    #[test]
    fn test_streammap_get_set() {
        let mut m: StreamMap<u32> = StreamMap::from_fn(|_| 0);
        *m.get_mut(StreamTag::VarKeyIndel) = 7;
        assert_eq!(*m.get(StreamTag::VarKeySnp), 0);
        assert_eq!(*m.get(StreamTag::VarKeyIndel), 7);
        let collected: Vec<_> = m.iter().map(|(_, v)| *v).collect();
        assert_eq!(collected, vec![0, 7]);
    }
}
```

- [ ] **Step 2: Run streams tests**

Run: `pixi run cargo test --no-default-features streams::`
Expected: 4 tests PASS.

- [ ] **Step 3: Add the module and confirm build**

Add `pub mod streams;` to `lib.rs`. Run: `pixi run cargo test --no-default-features`
Expected: PASS (streams is standalone; nothing consumes it yet).

- [ ] **Step 4: Lint + commit**

```bash
pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check
git add -A
git commit -m "feat(svar-2): add StreamTag registry (streams.rs) for tag-routed sub-streams

StreamTag/StreamSpec/REGISTRY/StreamMap describe each on-disk sub-stream so
orchestration can route by tag instead of hardcoding snp/indel and key widths.
Standalone; wired in by the next task.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: The flip — byte-erased sub-streams routed by tag

This is the largest task: it changes the in-memory data model and threads the tag
registry through `types` → `rvk` → `executor` → `writer` → orchestration. The
regression gate is `test_e2e.rs` (byte-identical output). Do it as one coherent
change; the suite won't compile mid-way, so run it only at the end.

**Files:**
- Modify: `src/types.rs` (`SparseSubStream`, `SparseChunk { chunk_id, streams }`)
- Modify: `src/rvk.rs` (`dense2sparse_vk` routes into `StreamMap<SparseSubStream>`; update in-source tests)
- Modify: `src/executor.rs` (ledgers become `StreamMap<Vec<Vec<u32>>>`)
- Modify: `src/writer.rs` (`run_io_writer` iterates streams)
- Modify: `src/lib.rs` `process_chromosome` (registry-driven merge loop; ledger plumbing)
- Modify: `tests/test_e2e.rs` only if it constructs `SparseChunk`/reads `.snp`/`.indel` fields directly (it reads final on-disk files, so likely untouched)

**Interfaces:**
- Consumes: `crate::streams::{StreamTag, StreamMap, StreamSpec, REGISTRY}`, `crate::rvk::VarKey`.
- Produces:
  ```rust
  // types.rs
  pub struct SparseSubStream { pub call_positions: Vec<u32>, pub call_keys: Vec<u8>, pub sample_lengths: Vec<u32>, pub key_bytes: usize }
  impl SparseSubStream { pub fn with_capacity(key_bytes: usize, nnz: usize, columns: usize) -> Self; pub fn push_call(&mut self, pos: u32, key_le: &[u8]); }
  pub struct SparseChunk { pub chunk_id: usize, pub streams: StreamMap<SparseSubStream> }
  // executor.rs
  pub fn run_compute_engine(rx_dense, tx_sparse, bank) -> (StreamMap<Vec<Vec<u32>>>, Vec<u64>)
  ```

- [ ] **Step 1: Rewrite the sub-stream types in `types.rs`**

Replace `SparseSubChunk<K>` and `SparseChunk` with:

```rust
use crate::streams::StreamMap;

/// One position-sorted sub-stream of calls with byte-erased keys (`key_bytes`
/// wide, little-endian). Type erasure lets streams of differing widths live in a
/// single `StreamMap`.
pub struct SparseSubStream {
    pub call_positions: Vec<u32>,
    pub call_keys: Vec<u8>,        // key_bytes per call
    pub sample_lengths: Vec<u32>,  // len == samples * ploidy
    pub key_bytes: usize,
}

impl SparseSubStream {
    pub fn with_capacity(key_bytes: usize, nnz: usize, columns: usize) -> Self {
        Self {
            call_positions: Vec::with_capacity(nnz),
            call_keys: Vec::with_capacity(nnz * key_bytes),
            sample_lengths: Vec::with_capacity(columns),
            key_bytes,
        }
    }
    #[inline(always)]
    pub fn push_call(&mut self, pos: u32, key_le: &[u8]) {
        debug_assert_eq!(key_le.len(), self.key_bytes);
        self.call_positions.push(pos);
        self.call_keys.extend_from_slice(key_le);
    }
}

/// Transposed sparse packet: one `SparseSubStream` per active `StreamTag`.
pub struct SparseChunk {
    pub chunk_id: usize,
    pub streams: StreamMap<SparseSubStream>,
}
```

- [ ] **Step 2: Route in `dense2sparse_vk` (`rvk.rs`)**

Rewrite the body of `dense2sparse_vk` to build one `SparseSubStream` per registry entry and push each set call into its tag's stream. Keep the per-variant `var_keys` pre-classification. Replace the two-`SparseSubChunk` block and the transpose match:

```rust
pub fn dense2sparse_vk(chunk: &DenseChunk, bank: &mut LongAlleleTableWriter) -> SparseChunk {
    let (v_variants, num_samples, ploidy) = chunk.genos.shape;
    let columns = num_samples * ploidy;

    // Pre-classify each variant once (also spills a long INS to the bank once).
    let mut var_keys: Vec<VarKey> = Vec::with_capacity(v_variants);
    for v in 0..v_variants {
        let ilen = unsafe { *chunk.ilens.get_unchecked(v) };
        let start_idx = unsafe { *chunk.alt_offsets.get_unchecked(v) } as usize;
        let end_idx = unsafe { *chunk.alt_offsets.get_unchecked(v + 1) } as usize;
        let alt_allele = unsafe { chunk.alt.get_unchecked(start_idx..end_idx) };
        var_keys.push(classify_variant(ilen, alt_allele, bank));
    }

    let estimated_nnz = (v_variants * columns) / 20;
    let mut streams = crate::streams::StreamMap::from_fn(|tag| {
        let spec = &crate::streams::REGISTRY[tag.index()];
        SparseSubStream::with_capacity(spec.key_bytes, estimated_nnz, columns)
    });

    let words: &[u64] = &chunk.genos.words;
    for s in 0..num_samples {
        for p in 0..ploidy {
            // per-tag running counts for this column
            let mut counts = crate::streams::StreamMap::from_fn(|_| 0u32);
            let base_idx = (s * ploidy) + p;
            let stride = columns;
            for v in 0..v_variants {
                let flat_idx = (v * stride) + base_idx;
                let word = unsafe { *words.get_unchecked(flat_idx >> 6) };
                if (word >> (flat_idx & 63)) & 1 != 0 {
                    let pos = unsafe { *chunk.pos.get_unchecked(v) };
                    let (tag, key_le): (StreamTag, [u8; 4]) = match unsafe { *var_keys.get_unchecked(v) } {
                        VarKey::Snp(code) => (StreamTag::VarKeySnp, [code, 0, 0, 0]),
                        VarKey::Indel(key) => (StreamTag::VarKeyIndel, key.to_le_bytes()),
                    };
                    let spec = &crate::streams::REGISTRY[tag.index()];
                    streams.get_mut(tag).push_call(pos, &key_le[..spec.key_bytes]);
                    *counts.get_mut(tag) += 1;
                }
            }
            for (tag, c) in counts.into_iter_tagged() {
                streams.get_mut(tag).sample_lengths.push(c);
            }
        }
    }

    SparseChunk { chunk_id: chunk.chunk_id, streams }
}
```

Add `use crate::streams::StreamTag;` and `use crate::types::SparseSubStream;` to `rvk.rs` imports; drop the old `SparseChunk`/`SparseSubChunk` imports as needed.

- [ ] **Step 3: Update `rvk.rs` in-source tests to the new shape**

The `dense2sparse_*` tests reference `sparse.snp.*` / `sparse.indel.*`. Change them to `sparse.streams.get(StreamTag::VarKeySnp)` / `get(StreamTag::VarKeyIndel)`. The SNP `call_keys` is now `Vec<u8>` of 1-byte codes (same values). The indel `call_keys` is now a flat `Vec<u8>`; decode a call `i` via `u32::from_le_bytes(indel.call_keys[i*4..i*4+4].try_into().unwrap())` before `decode_alt_inline`. Update the empty/all-true/split/conservation/per-sample tests accordingly (positions and counts are unchanged; only the accessor path and key byte-width change).

- [ ] **Step 4: Ledgers become tag-keyed in `executor.rs`**

```rust
use crate::streams::{StreamMap, StreamTag};

pub fn run_compute_engine(
    rx_dense: Receiver<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    mut bank: LongAlleleTableWriter,
) -> (StreamMap<Vec<Vec<u32>>>, Vec<u64>) {
    let mut ledgers: StreamMap<Vec<Vec<u32>>> = StreamMap::from_fn(|_| Vec::with_capacity(10_000));

    while let Ok(chunk) = rx_dense.recv() {
        let sparse_chunk = dense2sparse_vk(&chunk, &mut bank);
        for (tag, sub) in sparse_chunk.streams.iter() {
            ledgers.get_mut(tag).push(sub.sample_lengths.clone());
        }
        tx_sparse.send(sparse_chunk).expect("Failed to send SparseChunk to Writer");
    }

    println!("Executor: VCF fully processed. Flushing remaining long alleles...");
    let long_allele_offsets = bank.finalize();
    (ledgers, long_allele_offsets)
}
```

- [ ] **Step 5: Writer iterates streams (`writer.rs`)**

`run_io_writer` still receives per-contig stream dirs; pass a `StreamMap<PathBuf>` of dirs so it can look up each stream's dir by tag:

```rust
use crate::streams::StreamMap;
use std::path::{Path, PathBuf};

pub fn run_io_writer(rx_sparse: Receiver<SparseChunk>, dirs: StreamMap<PathBuf>) {
    while let Ok(chunk) = rx_sparse.recv() {
        let id = chunk.chunk_id;
        for (tag, sub) in chunk.streams.iter() {
            let dir = dirs.get(tag);
            write_bin(&crate::layout::chunk_pos(dir, id), bytemuck::cast_slice(&sub.call_positions));
            write_bin(&crate::layout::chunk_key(dir, id), &sub.call_keys); // already bytes
        }
    }
    println!("Writer Thread: Channel closed. All chunks safely committed to SSD.");
}

fn write_bin(path: &Path, bytes: &[u8]) {
    let mut f = BufWriter::new(File::create(path).unwrap_or_else(|e| panic!("create {:?}: {}", path, e)));
    f.write_all(bytes).expect("write chunk bytes");
    f.flush().expect("flush chunk bytes");
}
```

- [ ] **Step 6: Registry-driven merge loop in `process_chromosome` (`lib.rs`)**

Build the stream dirs from the registry + `ContigPaths`, create them, pass them to the writer, and after Phase 1 iterate the registry for merge + post-merge:

```rust
// dirs keyed by tag, created up front
let stream_dirs: crate::streams::StreamMap<std::path::PathBuf> =
    crate::streams::StreamMap::from_fn(|tag| {
        let spec = &crate::streams::REGISTRY[tag.index()];
        let dir = std::path::Path::new(base_out_dir).join(chrom).join(spec.subdir);
        std::fs::create_dir_all(&dir).expect("create stream output dir");
        dir
    });
```

Pass `stream_dirs`-derived clones into the chunk writer (`writer::run_io_writer(rx_sparse, dirs_for_writer)`), keep the long-allele writer using `paths.long_alleles_bin()`. Capture the executor result as `let (ledgers, long_allele_offsets) = executor_thread.join().unwrap();`. Write the offsets via `paths.long_allele_offsets()`. Then:

```rust
let num_chunks = ledgers.get(crate::streams::StreamTag::VarKeyIndel).len();
let mut ledgers = ledgers; // make mutable to move rows out
for spec in &crate::streams::REGISTRY {
    let dir = stream_dirs.get(spec.tag).clone();
    let ledger = std::mem::take(ledgers.get_mut(spec.tag));
    merge::merge_mini_sc(spec.key_bytes, num_chunks, samples.len(), ploidy, dir.to_str().unwrap(), ledger);
    if let Some(hook) = spec.post_merge {
        hook(&dir);
    }
}
```

Remove the now-dead explicit `merge_mini_sc(1, …)` / `pack_snp_key_file` / `merge_mini_sc(4, …)` lines from Task 4. (`num_chunks` is identical across streams — one ledger row per chunk.)

- [ ] **Step 7: Build, run the FULL suite, confirm byte-identical output**

Run: `pixi run cargo test --no-default-features`
Expected: PASS. `test_e2e.rs` asserts both sub-streams' offsets/positions/unpacked SNP codes/decoded indel keys — green means the flip preserved on-disk bytes.

- [ ] **Step 8: Lint + commit**

```bash
pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check
git add -A
git commit -m "refactor(svar-2): route sub-streams by StreamTag with byte-erased keys

SparseChunk holds a StreamMap<SparseSubStream> (Vec<u8> keys + key_bytes) instead
of hardcoded snp/indel fields. rvk/executor/writer/orchestration iterate the
registry; the merge loop routes by tag. Output is byte-identical (e2e green).
Adding pointer/dense now means extending StreamTag + REGISTRY only.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Extract `orchestrator.rs`; slim `lib.rs` to the pyo3 surface

**Files:**
- Create: `src/orchestrator.rs` (move `process_chromosome` + the long-allele writer thread body)
- Modify: `src/writer.rs` (add `run_long_allele_writer`)
- Modify: `src/lib.rs` (keep only `run_conversion_pipeline` + `#[pymodule] _core` + `mod`/`pub mod` decls)

**Interfaces:**
- Consumes: `crate::{budget, monitor, layout, streams, merge, writer, executor, nrvk, vcf_reader, types}`.
- Produces: `pub fn process_chromosome(vcf_path: &str, chrom: &str, base_out_dir: &str, samples: &[&str], chunk_size: usize, ploidy: usize, htslib_threads: usize, long_allele_capacity: usize)` (unchanged signature, new home). `pub fn run_long_allele_writer(rx_long: Receiver<Vec<u8>>, out_path: &Path, chrom_label: &str)` in `writer.rs`.

- [ ] **Step 1: Add `run_long_allele_writer` to `writer.rs`**

Move the inlined long-allele writer closure body out of `lib.rs`:

```rust
use crossbeam_channel::Receiver;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::path::Path;

pub fn run_long_allele_writer(rx_long: Receiver<Vec<u8>>, out_path: &Path, chrom_label: &str) {
    let file = OpenOptions::new().create(true).write(true).truncate(true).open(out_path).unwrap();
    let mut disk_writer = BufWriter::with_capacity(1024 * 1024, file);
    while let Ok(buffer) = rx_long.recv() {
        disk_writer.write_all(&buffer).expect("Failed to write long alleles");
    }
    disk_writer.flush().unwrap();
    println!("[{}] Long Allele Writer: All buffer data safely committed.", chrom_label);
}
```

- [ ] **Step 2: Move `process_chromosome` into `orchestrator.rs`**

Create `src/orchestrator.rs`. Move the entire `process_chromosome` fn (with its `#[allow(clippy::too_many_arguments)]`) verbatim from `lib.rs`, adding the imports it needs (`crossbeam_channel::bounded`, `std::sync::{Arc, atomic::*}`, `std::thread`, `crate::*`). Replace the inlined long-allele writer thread with:

```rust
let long_allele_writer_thread = thread::Builder::new()
    .name(format!("lw-{}", chrom))
    .spawn({
        let out_path = paths.long_alleles_bin();
        let chrom_label = chrom.to_string();
        move || crate::writer::run_long_allele_writer(rx_long, &out_path, &chrom_label)
    })
    .expect("spawn long allele writer");
```

- [ ] **Step 3: Slim `lib.rs`**

`lib.rs` keeps: the `pub mod` / `mod` declarations, `run_conversion_pipeline` (now calling `crate::orchestrator::process_chromosome`), and the `#[pymodule] _core`. Add `pub mod orchestrator;` **and** `pub use orchestrator::process_chromosome;` so the existing `use genoray_core::process_chromosome;` in `tests/test_e2e.rs` keeps resolving (the crate-root re-export). Change the dispatch call to `orchestrator::process_chromosome(…)`. Remove any now-unused imports from `lib.rs`.

Note: the Task 9 e2e test uses `genoray_core::orchestrator::process_chromosome` and `genoray_core::error::ConversionError` — both reachable via the `pub mod` decls; the existing `genoray_core::process_chromosome` import stays valid via the re-export above.

- [ ] **Step 4: Build, full suite, lint**

Run: `pixi run cargo test --no-default-features && pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check`
Expected: PASS/clean. Confirm `lib.rs` is now ~40-60 lines.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(svar-2): move process_chromosome into orchestrator.rs; slim lib.rs

lib.rs is now just the pyo3 _core surface + run_conversion_pipeline. The
long-allele writer thread body moves into writer::run_long_allele_writer.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `LongAlleleReader` → stateless `&self` + `pread`

**Files:**
- Modify: `src/nrvk.rs` (`get_allele` uses `read_exact_at`, takes `&self`; drop the `Seek`/`SeekFrom` imports)

**Interfaces:**
- Consumes: nothing new.
- Produces: `pub fn get_allele(&self, row_index: u32) -> Vec<u8>` (was `&mut self`).

- [ ] **Step 1: Write a failing test asserting `&self` + correct bytes**

Add to `nrvk.rs` tests a round-trip that reads two alleles from a staged file without `&mut`:

```rust
#[test]
fn test_reader_get_allele_shared_borrow() {
    use std::io::Write;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("chr1").join("var_key").join("indel");
    std::fs::create_dir_all(&dir).unwrap();
    // bytes: "AAAA" then "CC" → offsets [0, 4, 6]
    let mut f = std::fs::File::create(dir.join("long_alleles.bin")).unwrap();
    f.write_all(b"AAAACC").unwrap();
    let offsets = ndarray::Array1::from_vec(vec![0u64, 4, 6]);
    ndarray_npy::write_npy(dir.join("long_allele_offsets.npy"), &offsets).unwrap();

    let reader = LongAlleleReader::new(tmp.path().to_str().unwrap(), "chr1");
    // &self: two immutable calls, no &mut needed
    assert_eq!(reader.get_allele(0), b"AAAA".to_vec());
    assert_eq!(reader.get_allele(1), b"CC".to_vec());
}
```

- [ ] **Step 2: Run it — expect a compile failure (needs `&mut`)**

Run: `pixi run cargo test --no-default-features nrvk::tests::test_reader_get_allele_shared_borrow`
Expected: FAIL to compile — `get_allele` currently needs `&mut self`.

- [ ] **Step 3: Switch to `read_exact_at`**

In `nrvk.rs`, change the import `use std::io::{Read, Seek, SeekFrom};` to `use std::os::unix::fs::FileExt;` (remove `Seek`/`SeekFrom`/`Read` if now unused). Rewrite:

```rust
pub fn get_allele(&self, row_index: u32) -> Vec<u8> {
    let idx = row_index as usize;
    let start_byte = self.offsets[idx];
    let end_byte = self.offsets[idx + 1];
    let len = (end_byte - start_byte) as usize;
    let mut buf = vec![0u8; len];
    self.file.read_exact_at(&mut buf, start_byte).expect("pread long allele");
    buf
}
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `pixi run cargo test --no-default-features nrvk::`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check
git add -A
git commit -m "refactor(svar-2): LongAlleleReader::get_allele uses pread + &self

Stateless read_exact_at (POSIX pread, Linux+macOS) replaces seek+read, matching
merge.rs; the reader is now shareable across threads.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Bounded error-handling pass (`ConversionError`)

Scope is boundaries only: `process_chromosome` + `run_conversion_pipeline` return
`Result`; non-threaded I/O `.expect()` become `?`; thread-join panics become a
typed error; the pyo3 boundary maps to `PyErr`. Worker hot-loop `unwrap`s stay.

**Files:**
- Modify: `Cargo.toml` (add `thiserror = "2"`)
- Create: `src/error.rs` (`ConversionError`)
- Modify: `src/orchestrator.rs` (`process_chromosome -> Result<(), ConversionError>`)
- Modify: `src/lib.rs` (`run_conversion_pipeline` collects results, maps to `PyErr`)

**Interfaces:**
- Produces:
  ```rust
  #[derive(Debug, thiserror::Error)]
  pub enum ConversionError {
      #[error("I/O error at {context}: {source}")]
      Io { context: String, #[source] source: std::io::Error },
      #[error("worker thread '{thread}' panicked")]
      WorkerPanicked { thread: String },
      #[error("failed to write npy at {path}: {source}")]
      Npy { path: String, #[source] source: ndarray_npy::WriteNpyError },
  }
  pub fn process_chromosome(...) -> Result<(), ConversionError>;
  ```

- [ ] **Step 1: Add the dependency**

In `Cargo.toml` `[dependencies]`, add `thiserror = "2"`. Run `pixi run cargo fetch` (or let the next build resolve it).

- [ ] **Step 2: Write `src/error.rs`**

```rust
//! Typed errors for the conversion pipeline. Boundary-level: the orchestrator and
//! pyo3 entry point return these; worker-thread hot loops still panic (converting
//! them is a follow-up).

#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("I/O error at {context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("worker thread '{thread}' panicked")]
    WorkerPanicked { thread: String },
    #[error("failed to write npy at {path}: {source}")]
    Npy {
        path: String,
        #[source]
        source: ndarray_npy::WriteNpyError,
    },
}
```

Add `pub mod error;` to `lib.rs`.

- [ ] **Step 3: Thread `Result` through `process_chromosome`**

Change the signature to `-> Result<(), ConversionError>`. Convert the non-threaded I/O boundaries: `create_dir_all` → `.map_err(|e| ConversionError::Io { context: format!("create_dir_all {:?}", dir), source: e })?`; the offsets `write_npy` → `.map_err(|source| ConversionError::Npy { path: …, source })?`. Convert each `thread.join().unwrap()` to:

```rust
reader_thread.join().map_err(|_| ConversionError::WorkerPanicked { thread: format!("read-{}", chrom) })?;
```

(and likewise `exec`, `cw`, `lw`, `samp`). The executor's `(ledgers, offsets)` join becomes `let (ledgers, long_allele_offsets) = executor_thread.join().map_err(|_| ConversionError::WorkerPanicked { thread: format!("exec-{}", chrom) })?;`. End the fn with `Ok(())`.

- [ ] **Step 4: Collect results in `run_conversion_pipeline` and map to `PyErr`**

In the rayon dispatch, collect per-chrom results and surface the first error:

```rust
let results: Vec<Result<(), crate::error::ConversionError>> = pool.install(|| {
    chroms
        .par_iter()
        .map(|chrom| {
            println!("==> Processing {}", chrom);
            crate::orchestrator::process_chromosome(
                &vcf_path, chrom, &output_dir, &sample_refs,
                chunk_size, ploidy, htslib_threads, long_allele_capacity,
            )
        })
        .collect()
});
for r in results {
    r.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
}
```

Because `py.detach(|| { … })` wraps this, have the closure return the `Vec<Result>` (or the mapped `PyResult<()>`) out of `detach` and `?` it after. Keep the final `Ok(())`.

- [ ] **Step 5: Write a test that a bad chromosome returns Err, not a panic**

Add to `tests/test_e2e.rs` (it already builds synthetic BCFs) a case that calls `process_chromosome` with a chromosome not in the header and asserts it returns `Err(ConversionError::WorkerPanicked { .. })` (the reader thread panics on `name2rid`, surfaced as a worker-panic error) rather than unwinding the test:

```rust
#[test]
fn test_missing_chrom_returns_err() {
    // build a 1-record BCF on chr1, then ask for chrZ
    // … (reuse build_bcf_with_index helper) …
    let res = genoray_core::orchestrator::process_chromosome(
        bcf_path.to_str().unwrap(), "chrZ", out_dir.to_str().unwrap(),
        &["s0"], 1000, 2, 1, 1 << 20,
    );
    assert!(matches!(res, Err(genoray_core::error::ConversionError::WorkerPanicked { .. })));
}
```

- [ ] **Step 6: Run the suite**

Run: `pixi run cargo test --no-default-features`
Expected: PASS, including `test_missing_chrom_returns_err`.

- [ ] **Step 7: Lint + commit**

```bash
pixi run -e lint cargo clippy --all-targets -- -D warnings && pixi run -e lint cargo fmt --check
git add -A
git commit -m "feat(svar-2): bounded Result-based error path at pipeline boundaries

process_chromosome + run_conversion_pipeline return Result<_, ConversionError>
(thiserror); non-threaded I/O uses ?, worker-thread panics surface as
WorkerPanicked, mapped to PyRuntimeError at the pyo3 boundary. Hot-loop unwraps
inside workers remain (follow-up).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: Final gate + roadmap doc reconciliation

**Files:**
- Modify: `docs/roadmap/architecture.md`, `docs/roadmap/svar-2.md`, `docs/roadmap/data-model.md` (only where they name modules/paths that moved)

**Interfaces:** none.

- [ ] **Step 1: Reconcile roadmap references to moved code**

Grep the roadmap for module/path names that changed and update prose only (no design changes):

Run: `grep -rn -E 'lib\.rs|process_chromosome|merge_mini_sc|SparseSubChunk|snp_ledger|indel_ledger' docs/roadmap/`
Update any reference to reflect: monitoring lives in `monitor.rs`, budgeting in `budget.rs`, layout in `layout.rs`, orchestration in `orchestrator.rs`, stream routing via `streams.rs`/`StreamTag`, `merge_mini_sc(key_bytes, …)`, and ledgers as `StreamMap<Vec<Vec<u32>>>`. Leave milestone status text intact.

- [ ] **Step 2: Full green gate across both feature configurations**

Run:
```bash
pixi run cargo test --no-default-features
pixi run -e lint cargo clippy --all-targets -- -D warnings
pixi run -e lint cargo fmt --check
```
Expected: all PASS/clean.

- [ ] **Step 3: Byte-identical spot check (manual confidence)**

The e2e suite is the automated proof, but optionally run the e2e fixture through and confirm the final files exist where the layout says:
Run: `pixi run cargo test --no-default-features --test test_e2e -- --nocapture`
Expected: PASS; log lines reference `var_key/snp` and `var_key/indel` dirs.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(svar-2): reconcile roadmap module references to the fixup refactor

Points architecture/data-model/roadmap prose at monitor/budget/layout/
orchestrator/streams and the tag-routed merge. No design changes.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Mark PR ready (maintainer action, not automated)**

After review, the maintainer runs `gh pr ready 79`. Do not automate this — it's the human gate the whole fixup was preparing for.

---

## Self-Review Notes

- **Spec coverage:** module map (Tasks 2–7), seam generalization (Tasks 4–6), `layout.rs` (Task 3), `budget.rs` (Task 2), `monitor.rs` + platform degrade (Task 2b), bounded error pass (Task 9), hygiene (Task 1), `LongAlleleReader` pread (Task 8), platform-support invariant (Global Constraints + Task 8), final gate + docs (Task 10). All spec sections map to a task.
- **Behavior preservation:** every refactor task gates on `tests/test_e2e.rs` (byte-identical output); new pure units (budget, layout, streams, error) get their own TDD tests.
- **Type consistency:** `SparseSubStream` (not `SparseSubChunk`), `SparseChunk { chunk_id, streams }`, `StreamMap`/`StreamTag`/`StreamSpec`/`REGISTRY`, `merge_mini_sc(key_bytes, …)`, `run_compute_engine -> (StreamMap<Vec<Vec<u32>>>, Vec<u64>)`, `run_io_writer(rx, StreamMap<PathBuf>)`, `run_long_allele_writer(rx, &Path, &str)`, `get_allele(&self, u32)`, `process_chromosome -> Result<(), ConversionError>` — used consistently across tasks.

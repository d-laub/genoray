# SVAR2 Parallel-Reader Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the ~24 idle cores during single-contig VCF→SVAR2 conversion by parallelizing the reader's O(chunk·columns) genotype-presence bit-packing across a dedicated rayon pool, keeping output **byte-identical**.

**Architecture:** The reader thread stays sequential for all *ordering* work (htslib iteration, atomize, left-align, reorder heap, chunk cutting). Only the presence bit-packing in `read_next_chunk` is parallelized: variants are partitioned into **word-aligned blocks** so each rayon task writes a disjoint `&mut [u64]` slice of `BitGrid3.words` — no shared boundary words, no atomics. A second, separately-sized rayon "processing pool" is fed only the cores left idle after the pipeline + htslib threads, computed in `plan_thread_budget`.

**Tech Stack:** Rust, `rayon` (par_chunks_mut), `rust-htslib` (BCF reader), `pyo3` (extension module), `proptest` (parallel==sequential oracle), `cargo test`, `pixi`.

## Global Constraints

- **Byte-identical output is non-negotiable.** The optimized store MUST equal the oracle hash on **both** germline (`chr21.filt.bcf`) and gdc (`gdc.chr21.filt.bcf`) datasets. Oracle: `storehash.sh` + `oracle.<dataset>.hash` in `/carter/users/dlaub/svar_bench`.
- **Rust tests run with:** `pixi run bash -lc 'cargo test --no-default-features --features conversion <filter>'`. The default `extension-module` feature breaks the test binary link (`undefined symbol: _Py_Dealloc`) — always pass `--no-default-features --features conversion`.
- **Python tests:** `pixi run pytest tests -m "not network"` (525 tests today) must stay green.
- **Rust test baseline:** 185 tests today must stay green.
- **No public API change.** Everything here is internal Rust (`src/*.rs`). `skills/genoray-api/SKILL.md` needs **no** update — do not touch it.
- **Commits:** Conventional Commits (`feat:`, `perf:`, `refactor:`, `test:`, `docs:`). Branch: `svar-2`.
- **Big benchmarks run on a dedicated compute node**, never the login node (2 cores under load): `sbatch -p carter-compute -c 32 --mem=128G` holder + `srun --jobid=<id> --overlap ...`.
- **Ordering logic is frozen.** Do not change `decompose_current_record`'s atomize/left-align/heap logic, `next_atom`, chunk boundaries, or the `PendingAtom` ordering (`Ord`/`PartialOrd`). Only genotype-presence packing and thread budgeting change.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/budget.rs` | Thread-budget arithmetic (`plan_thread_budget`, `ThreadPlan`) | Modify: raise htslib cap (Task 1); add `processing_threads` field + math (Task 2) |
| `src/vcf_reader.rs` | `VcfChunkReader`, `PendingAtom`, `read_next_chunk` packing | Modify: `Rc`→`Arc` (Task 3); extract `pack_row`/`pack_presence_seq` + pool param (Task 4); add `pack_presence_par` + gating + proptest (Task 5); raw-GT fusion (Task 6, optional) |
| `src/orchestrator.rs` | Per-contig thread wiring; `process_chromosome` | Modify: accept `processing_threads`, build processing pool, hand it to the reader thread (Task 4) |
| `src/lib.rs` | pyo3 entry, rayon contig pool, dispatch | Modify: pass `plan.processing_threads` into `process_chromosome` (Task 4) |
| `tests/test_e2e.rs`, `tests/common/mod.rs`, `tests/test_convert_skip_e2e.rs` | `process_chromosome` callers | Modify: append the `processing_threads` argument (Task 4) |
| `tests/test_left_align_e2e.rs`, `tests/test_atomize_e2e.rs`, `tests/test_e2e.rs` | `read_next_chunk` callers | Modify: append the `pool` argument (Task 4) |
| `docs/roadmap/svar-2.md` | Roadmap / thread-model description | Modify: document the processing pool + parallel packing (Task 7) |

---

## Task 1: Lever 0 — raise the htslib decode-thread cap

**Rationale:** BGZF decompression is ~45% of the gdc reader; htslib already parallelizes block decompression but `plan_thread_budget` caps it at 4. With ~24 idle cores on a single-contig run, more decode threads may still pay. This is a one-constant change + a measurement, done **before** the parallel-packing refactor because it changes how much Lever 1 needs to do.

**Files:**
- Modify: `src/budget.rs:10` (the `MAX_HTSLIB_THREADS` constant) and its tests
- Test: `src/budget.rs` (inline `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `MAX_HTSLIB_THREADS` (module-private const) raised from `4` to `8`. `plan_thread_budget`'s signature and `ThreadPlan` shape are unchanged in this task.

- [ ] **Step 1: Update the failing test expectation first**

In `src/budget.rs`, the test `test_htslib_never_exceeds_max` already asserts against the constant symbolically, so it stays green. But `test_low_end_one_chrom_min_htslib` and `test_high_end_fans_out_and_clamps_htslib` pin concrete htslib numbers that the raised cap does **not** change (they clamp below 8 anyway). Add a **new** test that pins the raised ceiling to a concrete value so the bump is observable:

```rust
    #[test]
    fn test_high_end_single_chrom_uses_raised_htslib_cap() {
        // 33 cores → usable 32; 1 chrom → concurrent 1; cores_per_chrom 32;
        // htslib_unclamped = 32 - 4 = 28, clamped to [2, MAX_HTSLIB_THREADS=8] → 8.
        let plan = plan_thread_budget(33, 1);
        assert_eq!(plan.concurrent_chroms, 1);
        assert_eq!(plan.htslib_threads, 8);
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion budget::tests::test_high_end_single_chrom_uses_raised_htslib_cap'`
Expected: FAIL — `assertion failed: left: 4, right: 8` (cap is still 4).

- [ ] **Step 3: Raise the constant**

In `src/budget.rs`, change line 10:

```rust
// Ceiling for HTSlib decode threads. Bumped 4→8 for single-/few-contig
// workloads with many idle cores: gdc's 16007-sample records mean very large
// BGZF blocks where extra decode threads still pay. Multi-contig runs clamp
// well below this via cores_per_chrom, so the bump only bites when cores are idle.
const MAX_HTSLIB_THREADS: usize = 8;
```

- [ ] **Step 4: Run the budget tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion budget'`
Expected: PASS — all budget tests including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/budget.rs
git commit -m "perf(budget): raise htslib decode-thread cap 4→8 for idle-core workloads"
```

- [ ] **Step 6 (manual measurement, not a unit test): measure gdc**

On a compute node, build the profiling wheel and re-measure gdc end-to-end (see the Benchmark Harness section at the end of this plan). Record the wall-clock at htslib=8. **Optionally** try 12 and 16 by editing the constant and re-measuring; keep whichever wins and re-commit if you change it. This measurement informs — but does not block — Tasks 2–5. Note the result in the commit body or `docs/roadmap/svar-2.md`.

---

## Task 2: Add `processing_threads` to the thread budget

**Files:**
- Modify: `src/budget.rs` (`ThreadPlan` struct + `plan_thread_budget` + tests)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ThreadPlan` gains a third field: `pub processing_threads: usize`.
  - `plan_thread_budget(available_cores: usize, n_chroms: usize) -> ThreadPlan` now also populates `processing_threads = usable_cores.saturating_sub(concurrent_chroms * (PIPELINE_THREADS_PER_CHROM + htslib_threads)).max(1)` — the cores left idle after the pipeline + htslib threads, floored at 1. Consumed by `lib.rs` (Task 4) to size the processing pool.

- [ ] **Step 1: Write the failing test**

Add to `src/budget.rs`'s `mod tests`:

```rust
    #[test]
    fn test_processing_threads_absorb_idle_cores() {
        // 33 cores → usable 32; 1 chrom → concurrent 1; htslib 8 (Task 1 cap).
        // active = 1 * (PIPELINE_THREADS_PER_CHROM(4) + 8) = 12.
        // processing = max(1, 32 - 12) = 20.
        let plan = plan_thread_budget(33, 1);
        assert_eq!(plan.processing_threads, 20);
    }

    #[test]
    fn test_processing_threads_floored_at_one_when_saturated() {
        // 65 cores → usable 64; 22 chroms → concurrent 10; htslib 2.
        // active = 10 * (4 + 2) = 60. processing = max(1, 64 - 60) = 4.
        assert_eq!(plan_thread_budget(65, 22).processing_threads, 4);
        // Fully saturated: 7 cores → usable 6 → low-end, 1 chrom, htslib = min(max(1,6-4),8)=2.
        // active = 1*(4+2)=6. processing = max(1, 6-6) = 1 (floored).
        assert_eq!(plan_thread_budget(7, 1).processing_threads, 1);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion budget::tests::test_processing_threads'`
Expected: FAIL to **compile** — `ThreadPlan` has no field `processing_threads`.

- [ ] **Step 3: Add the field and compute it**

In `src/budget.rs`, extend the struct:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPlan {
    pub concurrent_chroms: usize,
    pub htslib_threads: usize,
    // Cores left idle after the pipeline + htslib threads across all concurrent
    // chroms. Sizes the reader's intra-chunk packing pool (see vcf_reader.rs).
    // Floored at 1 so the pool always builds; parallel packing self-gates off
    // when this is < 2.
    pub processing_threads: usize,
}
```

Then in `plan_thread_budget`, replace **both** `ThreadPlan { ... }` return literals so they also set `processing_threads`. Low-end branch:

```rust
    if usable_cores < MIN_THREADS_PER_CHROM {
        // Low-end: run one chrom, pour remaining cores into HTSlib decode.
        let htslib = std::cmp::max(1, usable_cores.saturating_sub(PIPELINE_THREADS_PER_CHROM));
        let htslib = std::cmp::min(htslib, MAX_HTSLIB_THREADS);
        let processing = processing_threads(usable_cores, 1, htslib);
        ThreadPlan {
            concurrent_chroms: 1,
            htslib_threads: htslib,
            processing_threads: processing,
        }
    } else {
```

High-end branch:

```rust
        let htslib = htslib_unclamped.clamp(MIN_HTSLIB_THREADS, MAX_HTSLIB_THREADS);
        let processing = processing_threads(usable_cores, concurrent, htslib);
        ThreadPlan {
            concurrent_chroms: concurrent,
            htslib_threads: htslib,
            processing_threads: processing,
        }
    }
```

Add this helper just above the `#[cfg(test)]` module:

```rust
/// Cores left idle after `concurrent` chroms each claim the pipeline threads plus
/// `htslib` decode threads. Floored at 1 so the processing pool always builds.
fn processing_threads(usable_cores: usize, concurrent: usize, htslib: usize) -> usize {
    let active = concurrent * (PIPELINE_THREADS_PER_CHROM + htslib);
    usable_cores.saturating_sub(active).max(1)
}
```

- [ ] **Step 4: Fix the other budget tests that build `ThreadPlan` literals**

Three existing tests assert against a full `ThreadPlan { concurrent_chroms, htslib_threads }` literal and will now fail to compile (missing field). Update `test_low_end_one_chrom_min_htslib`, `test_single_core_machine`, and `test_high_end_fans_out_and_clamps_htslib` to include the field. Computed values:
- `plan_thread_budget(4, 8)`: usable 3, htslib 1, active `1*(4+1)=5`, processing `max(1, 3-5)=1`.
- `plan_thread_budget(1, 22)`: usable 1, htslib 1, active `1*(4+1)=5`, processing `max(1,1-5)=1`.
- `plan_thread_budget(65, 22)`: concurrent 10, htslib 2, active 60, processing `max(1,64-60)=4`.

```rust
    #[test]
    fn test_low_end_one_chrom_min_htslib() {
        assert_eq!(
            plan_thread_budget(4, 8),
            ThreadPlan {
                concurrent_chroms: 1,
                htslib_threads: 1,
                processing_threads: 1,
            }
        );
    }

    #[test]
    fn test_single_core_machine() {
        assert_eq!(
            plan_thread_budget(1, 22),
            ThreadPlan {
                concurrent_chroms: 1,
                htslib_threads: 1,
                processing_threads: 1,
            }
        );
    }

    #[test]
    fn test_high_end_fans_out_and_clamps_htslib() {
        assert_eq!(
            plan_thread_budget(65, 22),
            ThreadPlan {
                concurrent_chroms: 10,
                htslib_threads: 2,
                processing_threads: 4,
            }
        );
    }
```

- [ ] **Step 5: Run all budget tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion budget'`
Expected: PASS — all budget tests (including Task 1's and Task 2's new ones).

- [ ] **Step 6: Commit**

```bash
git add src/budget.rs
git commit -m "feat(budget): plan a processing-thread count from idle cores"
```

---

## Task 3: Switch `PendingAtom.gt` from `Rc` to `Arc`

**Rationale:** `Rc<Vec<i32>>` is `!Send + !Sync`, so `&[PendingAtom]` cannot be shared across rayon worker threads. `Arc<Vec<i32>>` is `Send + Sync` with identical single-threaded semantics; clones happen only in `decompose_current_record` (sequential), so the atomic-refcount cost is negligible. This is a behavior-neutral prerequisite for Task 5.

**Files:**
- Modify: `src/vcf_reader.rs` (imports, `PendingAtom.gt` field, `decompose_current_record`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `PendingAtom.gt: Arc<Vec<i32>>` (was `Rc<Vec<i32>>`). No signature changes to public functions.

- [ ] **Step 1: Swap the import**

In `src/vcf_reader.rs`, change line 7:

```rust
use std::sync::Arc;
```

(Remove `use std::rc::Rc;`.)

- [ ] **Step 2: Change the field type**

In the `PendingAtom` struct, change the `gt` field:

```rust
    gt: Arc<Vec<i32>>, // len = num_samples * ploidy; allele index per column (-1 = missing)
```

- [ ] **Step 3: Change the two `Rc` call sites in `decompose_current_record`**

Line ~193:

```rust
        let gt = Arc::new(gt);
```

Line ~229 (inside the `for atom in atoms` loop):

```rust
                gt: Arc::clone(&gt),
```

- [ ] **Step 4: Run the reader-touching tests to verify no behavior change**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --test test_atomize_e2e --test test_left_align_e2e --test test_e2e'`
Expected: PASS — all existing tests unchanged (behavior-neutral swap).

- [ ] **Step 5: Commit**

```bash
git add src/vcf_reader.rs
git commit -m "refactor(vcf_reader): Rc→Arc on PendingAtom.gt for cross-thread sharing"
```

---

## Task 4: Extract the packing helpers + thread the processing pool through (still sequential)

**Rationale:** Before flipping on parallelism, restructure so the change is reviewable in two steps: (a) extract the presence-packing into a reusable `pack_row`/`pack_presence_seq` that also works on a word-offset sub-slice, and (b) plumb an optional processing pool from `lib.rs` → `process_chromosome` → the reader thread → `read_next_chunk`. This task keeps packing **sequential** (the pool is passed but unused for packing), so all tests and the byte-identical oracle must still pass unchanged.

**Files:**
- Modify: `src/vcf_reader.rs` (add `pack_row`, `pack_presence_seq`; rewrite `read_next_chunk` to take `pool` and call the helpers)
- Modify: `src/orchestrator.rs` (`process_chromosome` gains `processing_threads`, builds the pool, moves it into the reader thread, passes it to `read_next_chunk`)
- Modify: `src/lib.rs` (pass `plan.processing_threads` into `process_chromosome`)
- Modify: `tests/test_e2e.rs`, `tests/common/mod.rs`, `tests/test_convert_skip_e2e.rs` (append `processing_threads` arg)
- Modify: `tests/test_left_align_e2e.rs`, `tests/test_atomize_e2e.rs`, `tests/test_e2e.rs` (append `pool` arg to `read_next_chunk`)

**Interfaces:**
- Consumes: `ThreadPlan.processing_threads` (Task 2); `PendingAtom.gt: Arc<Vec<i32>>` (Task 3).
- Produces:
  - `fn pack_row(words: &mut [u64], word_base: usize, vi: usize, a: &PendingAtom, columns: usize)` — packs variant row `vi`'s presence bits into `words`, where `words[0]` is global word index `word_base`.
  - `fn pack_presence_seq(words: &mut [u64], atoms: &[PendingAtom], columns: usize)` — sequential full-grid pack (`word_base = 0`).
  - `VcfChunkReader::read_next_chunk(&mut self, chunk_size: usize, chunk_id: usize, pool: Option<&rayon::ThreadPool>) -> Option<DenseChunk>` — new trailing `pool` param.
  - `orchestrator::process_chromosome(..., skip_out_of_scope: bool, processing_threads: usize) -> Result<u64, ConversionError>` — new trailing `processing_threads` param.

- [ ] **Step 1: Add `pack_row` and `pack_presence_seq` to `src/vcf_reader.rs`**

Add these free functions above `impl VcfChunkReader` (after the `PendingAtom` `Ord` impls). This is the current inner packing loop, refactored to accept a `word_base` offset so it works on both the full `words` slice (`word_base = 0`) and, later, a disjoint sub-slice.

```rust
// Pack variant row `vi`'s presence bits into `words`, where `words[0]` corresponds
// to global word index `word_base`. Bit for (row vi, column col) lives at global
// flat index `vi*columns + col`; the local word index subtracts `word_base`.
// Presence is `gt[col] == source_alt_index`. Bits start zeroed and are only OR-set,
// and each word is assembled in a register and written once (identical result to a
// per-bit `or_bit` loop, far fewer stores).
#[inline]
fn pack_row(words: &mut [u64], word_base: usize, vi: usize, a: &PendingAtom, columns: usize) {
    let src = a.source_alt_index as i32;
    let gtc: &[i32] = &a.gt;
    let base = vi * columns;
    let mut col = 0usize;
    while col < columns {
        let flat = base + col;
        let w = (flat >> 6) - word_base;
        let b = flat & 63;
        let n = (64 - b).min(columns - col);
        let mut acc = 0u64;
        for k in 0..n {
            // SAFETY: col + k < columns == gtc.len().
            acc |= ((unsafe { *gtc.get_unchecked(col + k) } == src) as u64) << (b + k);
        }
        // SAFETY: w indexes a word within this row's target slice.
        unsafe {
            *words.get_unchecked_mut(w) |= acc;
        }
        col += n;
    }
}

// Sequential full-grid presence packing: one row at a time into the whole `words`
// slice (global word index == local word index, so `word_base == 0`).
fn pack_presence_seq(words: &mut [u64], atoms: &[PendingAtom], columns: usize) {
    for (vi, a) in atoms.iter().enumerate() {
        pack_row(words, 0, vi, a, columns);
    }
}
```

- [ ] **Step 2: Rewrite `read_next_chunk` to split metadata from packing and accept `pool`**

Replace the whole `read_next_chunk` method body. The metadata loop (pos/ilens/alt/alt_offsets) stays sequential; genotype packing goes through `pack_presence_seq` for now. The `pool` param is accepted but not yet used for packing (Task 5 wires it in).

```rust
    // Pull up to `chunk_size` atoms (already globally position-sorted) and pack them
    // into a variant-major DenseChunk. `pool`, when present, will host parallel
    // presence packing (Task 5); this revision still packs sequentially so output is
    // provably unchanged. Returns None once no atoms remain.
    pub fn read_next_chunk(
        &mut self,
        chunk_size: usize,
        chunk_id: usize,
        pool: Option<&rayon::ThreadPool>,
    ) -> Option<DenseChunk> {
        let _ = pool; // reserved for Task 5's parallel packing
        let mut atoms: Vec<PendingAtom> = Vec::with_capacity(chunk_size);
        while atoms.len() < chunk_size {
            match self.next_atom() {
                Some(a) => atoms.push(a),
                None => break,
            }
        }
        if atoms.is_empty() {
            return None;
        }

        let v = atoms.len();
        let columns = self.num_samples * self.ploidy;

        let mut pos = Vec::with_capacity(v);
        let mut ilens = Vec::with_capacity(v);
        let mut alt = Vec::with_capacity(v * 2);
        let mut alt_offsets = Vec::with_capacity(v + 1);
        alt_offsets.push(0u32);
        let mut genos = BitGrid3::zeros(v, self.num_samples, self.ploidy);

        // Sequential metadata pass (cheap, ordering-preserving).
        let mut off = 0u32;
        for a in atoms.iter() {
            pos.push(a.pos);
            ilens.push(a.ilen);
            alt.extend_from_slice(&a.alt);
            off += a.alt.len() as u32;
            alt_offsets.push(off);
        }

        // Presence packing (sequential for now).
        pack_presence_seq(&mut genos.words, &atoms, columns);

        Some(DenseChunk {
            chunk_id,
            pos,
            ilens,
            alt,
            alt_offsets,
            genos,
        })
    }
```

- [ ] **Step 3: Thread `processing_threads` + the pool through `process_chromosome`**

In `src/orchestrator.rs`, add `use std::sync::Arc;` is already present (line 4). Change the signature to add a trailing param:

```rust
pub fn process_chromosome(
    vcf_path: &str,
    fasta_path: Option<&str>,
    chrom: &str,
    base_out_dir: &str,
    samples: &[&str],
    chunk_size: usize,
    ploidy: usize,
    htslib_threads: usize,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    processing_threads: usize,
) -> Result<u64, ConversionError> {
```

Just before the `// Step 1 -> The Producer` block, build the processing pool (shared with the reader thread via `Arc`):

```rust
    // Dedicated rayon pool for the reader's intra-chunk presence packing. Sized to
    // the idle cores (budget::plan_thread_budget). Built even at size 1 so the
    // reader always has a handle; parallel packing self-gates off below 2 threads.
    let processing_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(processing_threads.max(1))
            .thread_name(|i| format!("pack-{}", i))
            .build()
            .expect("build processing pool"),
    );
```

In the reader thread closure, capture a clone and pass it to `read_next_chunk`. Change the `.spawn({ ... })` block: add `let pool = Arc::clone(&processing_pool);` alongside the other `let` captures, and update the read loop:

```rust
    let reader_thread = thread::Builder::new()
        .name(format!("read-{}", chrom))
        .spawn({
            let vcf = vcf_path.to_string();
            let fasta = fasta_path.map(|s| s.to_string());
            let chr = chrom.to_string();
            let s_owned: Vec<String> = samples.iter().map(|&s| s.to_string()).collect();
            let pool = Arc::clone(&processing_pool);

            move || {
                let s_refs: Vec<&str> = s_owned.iter().map(|s| s.as_str()).collect();
                let mut reader = VcfChunkReader::new(
                    &vcf,
                    fasta.as_deref(),
                    &chr,
                    &s_refs,
                    htslib_threads,
                    ploidy,
                    skip_out_of_scope,
                );
                let mut chunk_id = 0;
                while let Some(dense_chunk) =
                    reader.read_next_chunk(chunk_size, chunk_id, Some(&pool))
                {
                    tx_dense.send(dense_chunk).unwrap();
                    chunk_id += 1;
                }
                reader.dropped_out_of_scope()
            }
        })
        .expect("spawn reader");
```

- [ ] **Step 4: Pass `plan.processing_threads` from `src/lib.rs`**

In `src/lib.rs`, the `process_chromosome(...)` call inside the `chroms.par_iter().map(...)` closure needs the new trailing argument. After `skip_out_of_scope,` add `plan.processing_threads` — but `plan` is not captured into the closure. Bind it before the closure and move it in. Just after `let htslib_threads = plan.htslib_threads;` (line ~128) add:

```rust
        let processing_threads = plan.processing_threads;
```

Then in the `.map(|chrom| { ... })` call, append the argument:

```rust
                    orchestrator::process_chromosome(
                        &vcf_path,
                        fasta_ref,
                        chrom,
                        &output_dir,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        htslib_threads,
                        long_allele_capacity,
                        skip_out_of_scope,
                        processing_threads,
                    )
```

- [ ] **Step 5: Patch the test callers of `process_chromosome`**

`process_chromosome` is called from `tests/test_e2e.rs` (4 sites), `tests/common/mod.rs` (1 site), and `tests/test_convert_skip_e2e.rs` (1 site). Each call currently ends with the `skip`/`false` boolean then `)`. Append `1,` (single-threaded processing pool — these are tiny fixtures, no parallelism needed) as the new final argument. For example, in `tests/test_e2e.rs`:

```rust
    process_chromosome(
        bcf_path.to_str().unwrap(),
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,  // chunk_size
        2,    // ploidy
        1,    // htslib_threads
        4096, // long_allele_capacity
        false,
        1, // processing_threads
    )
    .expect("process_chromosome should succeed");
```

Apply the same `1, // processing_threads` insertion (as the last argument, before the closing `)`) to every `process_chromosome(...)` call in `tests/test_e2e.rs`, `tests/common/mod.rs`, and `tests/test_convert_skip_e2e.rs`.

- [ ] **Step 6: Patch the test callers of `read_next_chunk`**

`read_next_chunk` is called from `tests/test_left_align_e2e.rs` (3 sites), `tests/test_atomize_e2e.rs` (1 site), and `tests/test_e2e.rs` (1 site). Append `None` as the new final argument. Examples:

```rust
    // tests/test_left_align_e2e.rs / test_atomize_e2e.rs (loop form)
    while let Some(chunk) = reader.read_next_chunk(chunk_size, chunk_id, None) {
```

```rust
    // tests/test_e2e.rs (single-call form)
    let chunk = reader
        .read_next_chunk(100, 0, None)
        .expect("chunk should succeed");
```

Apply `None` (as the trailing arg) to all five `read_next_chunk(...)` call sites.

- [ ] **Step 7: Run the full Rust suite to verify green (still sequential, byte-identical by construction)**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion'`
Expected: PASS — all 185 tests. Packing is still sequential and structurally identical, so behavior is unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/vcf_reader.rs src/orchestrator.rs src/lib.rs tests/
git commit -m "refactor(vcf_reader): extract pack helpers, thread processing pool (still sequential)"
```

---

## Task 5: Parallelize presence packing over word-aligned variant blocks

**Rationale:** This is Lever 1. Row `vi` occupies bits `[vi*columns, (vi+1)*columns)`. A variant-block boundary at `vi` is u64-word-aligned iff `vi*columns % 64 == 0`, i.e. every `g = 64/gcd(columns,64)`-th variant. Partitioning `BitGrid3.words` into `words_per_block = columns/gcd(columns,64)` word chunks (via `par_chunks_mut`) gives each rayon task a **word-disjoint** slice — no shared boundary words, no atomics. Blocks map 1:1 to variant ranges `[c*g, (c+1)*g)`.

**Files:**
- Modify: `src/vcf_reader.rs` (add `gcd`, `pack_presence_par`, gating constant; use the pool in `read_next_chunk`; add proptest)

**Interfaces:**
- Consumes: `pack_row` (Task 4); `pool: Option<&rayon::ThreadPool>` (Task 4).
- Produces:
  - `fn gcd(a: usize, b: usize) -> usize`.
  - `fn pack_presence_par(words: &mut [u64], atoms: &[PendingAtom], columns: usize, pool: &rayon::ThreadPool)` — word-aligned-block parallel pack; result identical to `pack_presence_seq`.
  - `const PARALLEL_MIN_VARIANTS: usize` — below this, `read_next_chunk` packs sequentially.

- [ ] **Step 1: Write the failing proptest (parallel == sequential)**

Add to `src/vcf_reader.rs` a `#[cfg(test)] mod tests` block (if the file has none). This builds synthetic atoms, packs them both ways, and asserts the raw `words` are identical across shapes that cross block boundaries (including `v` not a multiple of `g`, missing values, and out-of-range allele indices).

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::sync::OnceLock;

    // One shared 4-thread pool for all proptest cases (building a pool per case is slow).
    fn test_pool() -> &'static rayon::ThreadPool {
        static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
        POOL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
        })
    }

    // Minimal PendingAtom carrying only the fields the packers read.
    fn atom(gt: Vec<i32>, src: u16) -> PendingAtom {
        PendingAtom {
            pos: 0,
            ilen: 0,
            alt: Vec::new(),
            source_alt_index: src,
            gt: std::sync::Arc::new(gt),
            seq: 0,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        // Parallel packing reproduces sequential packing bit-for-bit, for arbitrary
        // shapes (incl. v not a multiple of the word-aligned block size), allele
        // indices (incl. missing -1 and out-of-range values), and source alts.
        #[test]
        fn test_par_packing_matches_seq(
            num_samples in 1usize..40,
            ploidy in 1usize..4,
            v in 1usize..70,
            seed in any::<u64>(),
        ) {
            let columns = num_samples * ploidy;
            // xorshift64 for deterministic per-case gt/src patterns.
            let mut state = seed | 1;
            let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };

            let mut atoms = Vec::with_capacity(v);
            for _ in 0..v {
                let src = (next() % 4) as u16; // small alt index space
                let gt: Vec<i32> = (0..columns)
                    .map(|_| match next() % 5 {
                        0 => -1,            // missing
                        1 => src as i32,    // present (matches src)
                        2 => 7,             // out-of-range allele
                        _ => (next() % 4) as i32,
                    })
                    .collect();
                atoms.push(atom(gt, src));
            }

            let mut seq = BitGrid3::zeros(v, num_samples, ploidy);
            pack_presence_seq(&mut seq.words, &atoms, columns);

            let mut par = BitGrid3::zeros(v, num_samples, ploidy);
            pack_presence_par(&mut par.words, &atoms, columns, test_pool());

            prop_assert_eq!(seq.words, par.words, "columns={}, v={}", columns, v);
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion vcf_reader::tests::test_par_packing_matches_seq'`
Expected: FAIL to compile — `pack_presence_par` and `gcd` don't exist yet.

- [ ] **Step 3: Implement `gcd` and `pack_presence_par`**

In `src/vcf_reader.rs`, add the rayon prelude import near the top (after the existing `use` lines):

```rust
use rayon::prelude::*;
```

Add these below `pack_presence_seq`:

```rust
// Below this many variants in a chunk, parallel packing's per-task overhead
// outweighs the win — pack sequentially instead. Tunable; measure on gdc/germline.
const PARALLEL_MIN_VARIANTS: usize = 512;

#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

// Parallel presence packing. Variants are partitioned into word-aligned blocks:
// row `vi` occupies bits `[vi*columns, (vi+1)*columns)`, so a block boundary at a
// multiple of `g = 64/gcd(columns,64)` variants lands exactly on a u64 boundary.
// `par_chunks_mut(words_per_block)` hands each rayon task a word-DISJOINT slice, so
// there are no shared boundary words and no atomics — the result is bit-identical to
// `pack_presence_seq`. Block `c` covers variants `[c*g, min((c+1)*g, v))` and words
// `[c*words_per_block, ...)`, whose global base is `word_base = c*words_per_block`.
fn pack_presence_par(
    words: &mut [u64],
    atoms: &[PendingAtom],
    columns: usize,
    pool: &rayon::ThreadPool,
) {
    let d = gcd(columns, 64);
    let g = 64 / d; // variants per word-aligned block
    let words_per_block = columns / d; // == g * columns / 64, always an integer
    let v = atoms.len();

    pool.install(|| {
        words
            .par_chunks_mut(words_per_block)
            .enumerate()
            .for_each(|(c, wchunk)| {
                let vi_start = c * g;
                let vi_end = ((c + 1) * g).min(v);
                let word_base = c * words_per_block;
                for vi in vi_start..vi_end {
                    pack_row(wchunk, word_base, vi, &atoms[vi], columns);
                }
            });
    });
}
```

- [ ] **Step 4: Run the proptest to verify it passes**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion vcf_reader::tests::test_par_packing_matches_seq'`
Expected: PASS — parallel packing matches sequential across all generated shapes.

- [ ] **Step 5: Wire the pool into `read_next_chunk`**

In `read_next_chunk`, replace the reserved `let _ = pool;` line and the `pack_presence_seq(...)` call. Delete `let _ = pool;`, and replace the packing call with the gated dispatch:

```rust
        // Presence packing: parallel over word-aligned variant blocks when a
        // multi-thread pool is available and the chunk is large enough to amortize
        // the fan-out; identical output to the sequential path either way.
        let parallel = matches!(pool, Some(p) if p.current_num_threads() >= 2)
            && v >= PARALLEL_MIN_VARIANTS;
        if parallel {
            pack_presence_par(&mut genos.words, &atoms, columns, pool.unwrap());
        } else {
            pack_presence_seq(&mut genos.words, &atoms, columns);
        }
```

- [ ] **Step 6: Run the full Rust suite**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion'`
Expected: PASS — all tests (185 + the new proptest). E2E tests pass `None` (sequential path); the fixtures are tiny, so `v < PARALLEL_MIN_VARIANTS` would fall back anyway.

- [ ] **Step 7: Run the Python suite**

Run: `pixi run pytest tests -m "not network"`
Expected: PASS — 525 tests.

- [ ] **Step 8 (byte-identical verification — REQUIRED, on a compute node): germline oracle**

Build the release wheel and confirm the germline store hash equals the oracle. On a `carter-compute` allocation:

```bash
# Build release wheel (restores the fast default profile).
pixi run maturin develop --release
cd /carter/users/dlaub/svar_bench
# Convert germline chr21 with the parallel reader, then hash the store.
python run_svar2.py chr21.filt.bcf <out_dir> 32
./storehash.sh <out_dir> > /tmp/germline.hash
diff /tmp/germline.hash oracle.germline.hash && echo "GERMLINE BYTE-IDENTICAL"
```

Expected: `GERMLINE BYTE-IDENTICAL` (no diff).

- [ ] **Step 9 (byte-identical verification — REQUIRED, on a compute node): gdc oracle + timing**

```bash
cd /carter/users/dlaub/svar_bench
time python run_svar2.py gdc.chr21.filt.bcf <gdc_out_dir> 32
./storehash.sh <gdc_out_dir> > /tmp/gdc.hash
diff /tmp/gdc.hash oracle.gdc.hash && echo "GDC BYTE-IDENTICAL"
```

Expected: `GDC BYTE-IDENTICAL` (no diff). Record the wall-clock vs. the ~18 min baseline — this is the headline result of Lever 1.

- [ ] **Step 10: Commit**

```bash
git add src/vcf_reader.rs
git commit -m "perf(vcf_reader): parallel presence packing over word-aligned variant blocks"
```

---

## Task 6 (OPTIONAL — measure Task 5 first): fuse raw-GT decode into the parallel pass

**Rationale:** Only pursue this if Task 5's measurement shows the remaining sequential GT decode in `decompose_current_record` is a meaningful fraction. Presence needs only the **raw** BCF GT int: `present(col) = (raw_gt[col] >> 1) == src + 1` (this treats missing `0`/`1` and vector-end `i32::MIN` as absent, exactly matching `GenotypeAllele::index()`). Storing the record's flat raw GT buffer (`Arc<Vec<i32>>`) + `width` per atom deletes the per-record decode loop and defers the compare into the parallel pass. **Guard:** this touches sample-subset mapping and non-uniform ploidy — verify byte-identical on both datasets before trusting it.

**Files:**
- Modify: `src/vcf_reader.rs` (`PendingAtom` fields, `decompose_current_record`, `pack_row`; the proptest gains a raw-GT variant)

**Interfaces:**
- Consumes: `pack_presence_par`/`pack_presence_seq` structure (Task 5).
- Produces:
  - `PendingAtom` replaces `gt: Arc<Vec<i32>>` with `raw_gt: Arc<Vec<i32>>` (flat, `total_file_samples * width`) + `width: usize`.
  - `pack_row` maps column → raw index via `self`-supplied `sample_indices`/`ploidy`. Because `pack_row` now needs the subset map, it takes an extra `ctx: &PackCtx` borrow (`sample_indices: &[usize]`, `ploidy: usize`). `PackCtx` is `Send + Sync` (slices of `usize`), so it crosses into rayon fine.

> This task is deliberately left at design granularity in the plan because it is **conditional on Task 5's profiling result** and carries the sample-subset/ploidy risk called out in the spec. When you reach it, drive it with superpowers:test-driven-development: (1) add a `PackCtx { sample_indices, ploidy }`; (2) in `decompose_current_record`, read `width = self.record.format(b"GT").inner().n as usize`, take `.integer()` for `total_samples = gts.len()`, reconstruct the flat untrimmed buffer from `gts[0].as_ptr()` over `total_samples * width` and `.to_vec()` it into an `Arc` (the buffer is contiguous and htslib-allocated for the full `n*width`); (3) rewrite `pack_row` to iterate `(s_idx, p)` contiguously, computing `present = p < width && (raw[sample_indices[s_idx]*width + p] >> 1) == src + 1`, flushing each u64 word once on the word boundary; (4) extend `test_par_packing_matches_seq` with a raw-buffer generator that includes `i32::MIN` (vector-end) and `p >= width` short rows; (5) re-verify **both** oracle hashes. Do not merge unless both stay byte-identical.

---

## Task 7: Update the roadmap doc

**Files:**
- Modify: `docs/roadmap/svar-2.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Document the new thread model**

Update `docs/roadmap/svar-2.md` to describe: (a) the raised htslib decode-thread cap (Task 1); (b) `ThreadPlan.processing_threads` and how idle cores are computed (Task 2); (c) the dedicated processing pool and word-aligned-block parallel packing in the reader (Tasks 4–5); (d) the measured gdc timing from Task 5 Step 9. Keep the existing pipeline diagram but note the reader now fans its presence packing across the processing pool.

- [ ] **Step 2: Commit**

```bash
git add docs/roadmap/svar-2.md
git commit -m "docs(roadmap): document processing pool + parallel presence packing"
```

---

## Benchmark Harness (reference — used by Task 1 Step 6 and Task 5 Steps 8–9)

Everything is set up in `/carter/users/dlaub/svar_bench/`: filtered BCFs (`chr21.filt.bcf`, `gdc.chr21.filt.bcf`), `run_svar2.py`, `storehash.sh`, `oracle.germline.hash`, `oracle.gdc.hash`, and perf captures. Reference FASTA: `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`.

**Always run on a dedicated compute node** — the login node has 2 cores under heavy load:

```bash
# Grab a 32-core holder allocation, then srun into it with --overlap.
sbatch -p carter-compute -c 32 --mem=128G --wrap 'sleep 6h'   # note the JOBID
srun --jobid=<JOBID> --overlap --pty bash
```

**Profiling wheel (for perf/flamegraph):**

```bash
RUSTFLAGS="-C force-frame-pointers=yes" pixi run maturin develop --profile profiling
# ... perf record ...
pixi run maturin develop --release   # restore the fast default build afterward
```

**Known-good baselines:** germline `run_svar2.py chr21.filt.bcf … 32` ≈ 36.5 s; gdc ≈ 18 min pre-Lever-1.

---

## Self-Review

**1. Spec coverage:**
- Lever 0 (raise htslib cap, measure first) → Task 1. ✓
- Thread-budget rework (processing pool sized to idle cores) → Task 2 (arithmetic) + Task 4 (pool built and threaded through). ✓ The spec's "second, separately-sized rayon pool that the reader `install`s its packing `par_iter` into" is option (a) — chosen. ✓
- Lever 1 parallel fused decode+pack, ordering logic sequential → Task 5 (packing) + Task 6 (optional decode fusion). ✓ Ordering (atomize/left-align/heap/chunk-cut) untouched — Global Constraints + Task 4 metadata loop stays sequential. ✓
- Disjoint parallel writes via word-aligned blocks + `par_chunks_mut`/`split_at_mut` — Task 5 uses `par_chunks_mut(words_per_block)`; `words_per_block = columns/gcd(columns,64)`, block size `g = 64/gcd`. ✓ (gdc columns=32014→g=32; germline columns=6404→g=16, matching the spec's worked examples.)
- Rc→Arc Send/Sync prerequisite (implicit in spec's "trivially parallel across atoms") → Task 3, called out explicitly. ✓
- Size-threshold gate for small chunks → `PARALLEL_MIN_VARIANTS` in Task 5 Step 5. ✓
- Byte-identical verification: oracle hashes both datasets + proptest → Task 5 Steps 8–9 (oracle) + Step 1 (proptest). ✓
- rust test 185 + pytest 525 green → Task 4 Step 7 / Task 5 Steps 6–7. ✓
- Flat-vs-per-sample buffer + non-uniform ploidy risks → confined to the optional Task 6, with the reconstruction + `p < width` guard spelled out. ✓
- Roadmap update → Task 7. ✓

**2. Placeholder scan:** All code steps show complete code. Task 6 is intentionally at design granularity (conditional on Task 5's profiling result) and says so explicitly — it is optional and gated, not a hidden TODO in the required path (Tasks 1–5, 7).

**3. Type consistency:** `ThreadPlan.processing_threads: usize` (Task 2) is read as `plan.processing_threads` (Task 4). `process_chromosome(..., processing_threads: usize)` matches the `1,`/`plan.processing_threads` args passed by tests and `lib.rs`. `read_next_chunk(..., pool: Option<&rayon::ThreadPool>)` matches the `None` (tests) and `Some(&pool)` (orchestrator) call sites. `pack_row(words, word_base, vi, a, columns)` has one signature used by both `pack_presence_seq` (word_base 0) and `pack_presence_par` (word_base `c*words_per_block`). `PendingAtom.gt: Arc<Vec<i32>>` (Task 3) is what the proptest's `atom()` helper constructs (Task 5). Consistent.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-07-svar2-parallel-reader.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

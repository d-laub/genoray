# SVAR2 reader cell budget and RAM-law refit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound `SparseVar2.from_pgen`'s reader RAM — today ~8 KB per sample per concurrent contig, capped by neither `chunk_size` nor `max_mem` — then re-fit `RamLaw::PGEN` against the fixed code.

**Architecture:** `ChunkAssembler` retains presence *bitsets* instead of allele vectors (32× smaller), byte budgets replace the two variant-count constants that bound the reader's live set, `_auto_chunk_size` stops overriding its own byte budget with a 1024 floor, and the RAM law is re-fitted on a design matrix that can actually identify `kappa`.

**Tech Stack:** Rust (rayon, proptest), Python (genoray `_svar2`), pixi, Slurm/sbatch.

**Spec:** `docs/superpowers/specs/2026-08-06-reader-cell-budget-and-ram-refit-design.md`

## Global Constraints

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `docs:`, `test:`). Never edit `CHANGELOG.md`; never bump the version by hand.
- Rust tests: `CARGO_TARGET_DIR=/local/$USER/cargo-target-diag cargo test --no-default-features --features conversion`. Both flags matter — dropping `--no-default-features` fails to link the pyo3 test binary; dropping `--features conversion` silently skips the entire conversion path (341 tests vs 189).
- Also run `CARGO_TARGET_DIR=/local/$USER/cargo-target-diag cargo check --no-default-features` — the query-core build has no other CI coverage.
- `CARGO_TARGET_DIR` **must** be off NFS. An NFS `target/` bus-errors the linker.
- `pixi run test` does **not** rebuild the Rust extension. Run `pixi run maturin develop --release` before any Python-level check of Rust behaviour.
- Never detach a process (`nohup`, `setsid`, `disown`, trailing `&`). Long or compute-heavy work goes to `sbatch`, never the login node.
- Any cross-point timing comparison must pin `--nodelist`; node speed on this cluster varies 2.08×.
- Scratch goes to `/local/$USER` (visible from compute nodes; `/tmp` inside a Slurm job is a private mount). Never write bulk output under `~/.claude`.
- Public API rule: anything reachable from `import genoray` without a leading underscore requires the same PR to update `skills/genoray-api/SKILL.md`. Everything this plan touches is private — re-check against the final diff rather than assuming.
- Run all commands from the worktree root, `/carter/users/dlaub/projects/genoray/.claude/worktrees/svar2-pgen-budget-planner`.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/chunk_assembler.rs` | `PresenceMasks`, `AtomCalls`, mask packing, the two byte budgets, `PARALLEL_MIN_CELLS` | 1, 2, 3, 4 |
| `python/genoray/_svar2.py` | `_auto_chunk_size` floor | 5 |
| `tests/test_svar2_chunk_size.py` | floor tests (two currently assert the old floor) | 5 |
| `tests/test_svar2_reader_identity.py` (new) | end-to-end store byte-identity across sources | 6 |
| `scripts/bench_svar2/plans/build_plans.py` | PGEN chunk-size axis, S=128,000 rung | 8 |
| `scripts/bench_svar2/sweep_pgen.sbatch` | corpus list for the new rung | 8 |
| `src/budget.rs` | re-fitted `RamLaw::PGEN` + its tests | 9 |

**Parallelism.** Tasks **1, 5, 8** touch disjoint files and have no interdependencies — dispatch them concurrently. Tasks 2 → 3 → 4 are strictly serial (each consumes the previous one's types). Tasks 6 → 7 → 9 are serial and gated on 4, 5, and 8.

Use `superpowers:dispatching-parallel-agents` with `superpowers:subagent-driven-development`. **Implementers must be Sonnet or weaker; reserve Opus for review and for second-pass fixes after a critical implementer failure.** Dispatch each implementer with an explicit *foreground-only* rule: subagents flake by backgrounding long `cargo`/`maturin` runs and returning before they finish. Subagents also default to the main repo rather than this worktree — instruct each to `cd` to the worktree root and verify with `git rev-parse --show-toplevel` before editing.

---

### Task 1: `PresenceMasks` — the per-record presence bitset

**Files:**
- Modify: `src/chunk_assembler.rs` (add the type near `pack_row`, above `PARALLEL_MIN_VARIANTS`)
- Test: `src/chunk_assembler.rs` (`mod tests`, alongside `pack_row_dense_calls_matches_the_raw_gt_loop`)

**Interfaces:**
- Consumes: nothing.
- Produces: `struct PresenceMasks` with `fn from_dense(gt: &[i32], columns: usize, wanted: &[u16]) -> PresenceMasks` and `fn mask(&self, slot: u16) -> &[u64]`. Task 2 packs from `mask()`; Task 3 constructs via `from_dense`.

**Context the implementer needs:** `Calls::Dense(Vec<i32>)` holds one allele index per haplotype column — `0` = REF, `k` = ALT k, `-1` = missing. `source_alt_index` is 1-based over the record's ALTs and is compared directly against those values (`gt[col] == source_alt_index`), so it is never 0.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `src/chunk_assembler.rs`:

```rust
#[test]
fn presence_masks_mark_exactly_the_columns_matching_each_alt() {
    let gt = vec![0i32, 1, 2, -1, 1, 2, 0, 1];
    let m = PresenceMasks::from_dense(&gt, 8, &[1, 2]);
    // slot 0 == ALT 1 -> columns 1, 4, 7
    assert_eq!(m.mask(0)[0], (1u64 << 1) | (1 << 4) | (1 << 7));
    // slot 1 == ALT 2 -> columns 2, 5
    assert_eq!(m.mask(1)[0], (1u64 << 2) | (1 << 5));
}

#[test]
fn presence_masks_ignore_ref_missing_and_out_of_scope_alts() {
    // A record whose ALT 2 was dropped as out-of-scope (symbolic/breakend) gets
    // ONE slot. REF (0), missing (-1) and the dropped ALT must not leak into it.
    let gt = vec![0i32, 1, 2, -1, 3];
    let m = PresenceMasks::from_dense(&gt, 5, &[1]);
    assert_eq!(m.mask(0)[0], 1u64 << 1);
}

#[test]
fn presence_masks_cost_one_bit_per_column_per_slot() {
    // The whole point of the type: 200 columns cost 4 words per slot, not 200
    // i32s per record (issue #155).
    let gt = vec![0i32; 200];
    let m = PresenceMasks::from_dense(&gt, 200, &[1, 2]);
    assert_eq!(m.mask(0).len(), 4);
    assert_eq!(m.mask(1).len(), 4);
}

#[test]
fn presence_masks_set_high_columns_in_the_right_word() {
    let mut gt = vec![0i32; 200];
    for c in [0usize, 63, 64, 65, 199] {
        gt[c] = 1;
    }
    let m = PresenceMasks::from_dense(&gt, 200, &[1]);
    let w = m.mask(0);
    assert_eq!(w[0], (1u64 << 0) | (1u64 << 63));
    assert_eq!(w[1], (1u64 << 0) | (1u64 << 1));
    assert_eq!(w[3], 1u64 << (199 - 192));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion presence_masks
```
Expected: FAIL — `cannot find type PresenceMasks in this scope`.

- [ ] **Step 3: Implement `PresenceMasks`**

Insert into `src/chunk_assembler.rs` immediately above `const PARALLEL_MIN_VARIANTS`:

```rust
/// Per-record presence bitsets: for each in-scope ALT of one source record, the
/// set of haplotype columns whose call is that ALT.
///
/// This is what dense sources retain INSTEAD of `Calls::Dense(Vec<i32>)`. The
/// only things a retained `Calls::Dense` was ever used for are `pack_row`'s
/// `gt[col] == source_alt_index` test and carrier recovery -- and dense sources
/// skip carrier recovery (`flush_window` returns `None`, since recovering
/// carriers from the packed grid is cheaper). So the retained payload can be
/// that test's ANSWER: one bit per column instead of one `i32`, `columns/8`
/// bytes per record instead of `columns*4`. That 32x is what keeps the reader's
/// live set bounded at biobank cohort widths -- see issue #155.
struct PresenceMasks {
    /// Slot-major: slot `s` owns `words[s*words_per_mask .. (s+1)*words_per_mask]`.
    words: Vec<u64>,
    words_per_mask: usize,
}

impl PresenceMasks {
    /// Build one slab from a record's dense calls, in a SINGLE pass over `gt`.
    ///
    /// `wanted` is the ascending, deduplicated list of `source_alt_index` values
    /// this record's atoms actually carry; slot `i` corresponds to `wanted[i]`.
    /// Restricting to those means a record whose other ALTs were dropped as
    /// out-of-scope pays for the ALTs it kept, not for `n_alts`. Alleles outside
    /// `wanted` -- REF `0`, missing `-1`, dropped ALTs -- set no bit, which is
    /// exactly what `gt[col] == src` does for any `src` in `wanted`.
    fn from_dense(gt: &[i32], columns: usize, wanted: &[u16]) -> Self {
        debug_assert!(
            wanted.iter().all(|&a| a > 0),
            "source_alt_index is 1-based; allele 0 is REF and can never be an atom's ALT"
        );
        let words_per_mask = columns.div_ceil(64);
        let mut words = vec![0u64; words_per_mask * wanted.len()];

        // allele -> slot. `u16::MAX` means "no slot"; sized by the largest
        // wanted allele rather than by `n_alts`, which is not passed in.
        let max_allele = wanted.iter().copied().max().unwrap_or(0) as usize;
        let mut slot_of = vec![u16::MAX; max_allele + 1];
        for (slot, &allele) in wanted.iter().enumerate() {
            slot_of[allele as usize] = slot as u16;
        }

        for (col, &a) in gt.iter().take(columns).enumerate() {
            if a < 0 {
                continue; // missing
            }
            let a = a as usize;
            if a >= slot_of.len() {
                continue; // REF, or an ALT no atom kept
            }
            let slot = slot_of[a];
            if slot == u16::MAX {
                continue;
            }
            words[slot as usize * words_per_mask + (col >> 6)] |= 1u64 << (col & 63);
        }
        Self {
            words,
            words_per_mask,
        }
    }

    #[inline]
    fn mask(&self, slot: u16) -> &[u64] {
        let start = slot as usize * self.words_per_mask;
        &self.words[start..start + self.words_per_mask]
    }
}
```

Note `slot_of[0]` stays `u16::MAX` because `wanted` never contains 0, so REF columns are skipped by the `slot == u16::MAX` check.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion presence_masks
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chunk_assembler.rs
git commit -m "feat(svar2): add PresenceMasks, a per-record presence bitset

One bit per haplotype column per in-scope ALT, replacing the allele vector a
dense record's atoms retain today. 32x smaller; see issue #155."
```

---

### Task 2: Pack from a mask instead of scanning alleles

**Files:**
- Modify: `src/chunk_assembler.rs` (add `or_mask_into` next to `pack_row`)
- Test: `src/chunk_assembler.rs` (`mod tests`)

**Interfaces:**
- Consumes: `PresenceMasks::from_dense`, `PresenceMasks::mask` (Task 1).
- Produces: `fn or_mask_into(words: &mut [u64], word_base: usize, base: usize, mask: &[u64], columns: usize)`. Task 3 calls it from `pack_row`'s dense arm.

**Context the implementer needs:** row `vi` occupies bits `[vi*columns, (vi+1)*columns)` of the chunk's flat bit grid, and `words[0]` corresponds to global word `word_base`. The row's start bit is generally *not* word-aligned, so the mask must be OR-ed in shifted. `pack_presence_par` hands each rayon task a word-disjoint slice, so a write past the row's last word would corrupt a neighbouring task's words — this is the invariant the bounds check protects.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests`. For the proptest, use the same RNG import the neighbouring `test_par_packing_matches_seq` proptest already uses in this module — do not introduce a different RNG crate.

```rust
#[test]
fn or_mask_into_handles_a_word_aligned_row() {
    // vi = 0, columns = 64: s == 0, so the carry branch must not run at all
    // (`>> 64` is UB).
    let gt: Vec<i32> = (0..64).map(|c| if c % 3 == 0 { 1 } else { 0 }).collect();
    let m = PresenceMasks::from_dense(&gt, 64, &[1]);
    let mut got = vec![0u64; 1];
    or_mask_into(&mut got, 0, 0, m.mask(0), 64);
    let mut want = 0u64;
    for c in 0..64 {
        if c % 3 == 0 {
            want |= 1u64 << c;
        }
    }
    assert_eq!(got[0], want);
}

#[test]
fn or_mask_into_never_writes_past_the_rows_last_word() {
    // columns = 100, vi = 1 -> the row spans bits 100..200, i.e. words 1..3.
    // Word 3 belongs to the NEXT row and, under `pack_presence_par`, possibly to
    // another rayon task. A stray carry there is silent cross-task corruption.
    let mut gt = vec![0i32; 100];
    gt[99] = 1;
    let m = PresenceMasks::from_dense(&gt, 100, &[1]);
    let mut got = vec![0u64; 4];
    or_mask_into(&mut got, 0, 100, m.mask(0), 100);
    assert_eq!(got[3], 0, "wrote into a word outside the row");
    assert_eq!(got[(199) >> 6], 1u64 << (199 & 63));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    // The migration's safety argument: packing from a mask must reproduce, bit
    // for bit, what the per-allele scan produced. Generates `columns` that are
    // and are not multiples of 64, rows at every shift, missing calls, and
    // multiallelic records.
    #[test]
    fn or_mask_into_matches_the_allele_scan(
        columns in 1usize..300,
        vi in 0usize..40,
        src in 1u16..4,
        seed in any::<u64>(),
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let gt: Vec<i32> = (0..columns).map(|_| rng.gen_range(-1i32..4)).collect();

        // Reference: the pre-mask element scan.
        let total_words = ((vi + 1) * columns).div_ceil(64);
        let mut want = vec![0u64; total_words];
        for (col, &g) in gt.iter().enumerate() {
            if g == src as i32 {
                let flat = vi * columns + col;
                want[flat >> 6] |= 1u64 << (flat & 63);
            }
        }

        let masks = PresenceMasks::from_dense(&gt, columns, &[src]);
        let mut got = vec![0u64; total_words];
        or_mask_into(&mut got, 0, vi * columns, masks.mask(0), columns);
        prop_assert_eq!(got, want);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion or_mask_into
```
Expected: FAIL — `cannot find function or_mask_into`.

- [ ] **Step 3: Implement `or_mask_into`**

Insert immediately after `pack_row` in `src/chunk_assembler.rs`:

```rust
// OR one row's presence mask into `words` at flat bit offset `base`, where
// `words[0]` is global word `word_base`. `mask` carries `columns` meaningful
// bits; bits at or beyond `columns` are zero by construction
// (`PresenceMasks::from_dense` only ever sets `col < columns`).
//
// Replaces an O(columns) allele comparison with an O(columns/64) shifted word
// OR. `base` is generally not word-aligned -- row `vi` starts at `vi*columns` --
// so each mask word contributes to two target words.
#[inline]
fn or_mask_into(words: &mut [u64], word_base: usize, base: usize, mask: &[u64], columns: usize) {
    if columns == 0 {
        return;
    }
    let w0 = (base >> 6) - word_base;
    let s = base & 63;
    let last = ((base + columns - 1) >> 6) - word_base;

    for (j, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        // A nonzero mask word's lowest set bit is at some `col < columns`, so
        // its low half always lands inside the row's span.
        let lo = w0 + j;
        words[lo] |= m << s;
        if s > 0 {
            let hi = lo + 1;
            if hi <= last {
                words[hi] |= m >> (64 - s);
            } else {
                // Not merely an optimisation: `pack_presence_par` gives each
                // task a word-DISJOINT slice, so writing past `last` would
                // corrupt another task's words. A carry here is always zero,
                // because bits at or beyond `columns` are zero.
                debug_assert_eq!(m >> (64 - s), 0, "carry outside the row's span");
            }
        }
    }
}
```

The `s > 0` guard is required for correctness, not style: `m >> 64` is undefined behaviour in Rust and will not reliably evaluate to 0.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion or_mask_into
```
Expected: 3 passed (two unit, one proptest over 400 cases).

- [ ] **Step 5: Commit**

```bash
git add src/chunk_assembler.rs
git commit -m "feat(svar2): pack presence rows by shifted word OR from a mask

O(columns/64) word ops replacing an O(columns) allele scan. Bounds the carry
write against the row's last word: pack_presence_par's slices are word-disjoint,
so a stray carry is cross-task corruption, not a rounding error."
```

---

### Task 3: Retain masks instead of `Calls::Dense`

**Files:**
- Modify: `src/chunk_assembler.rs` — `PendingAtom` (line ~15), `pack_row` (~79), `flush_window` (~259), `decompose_raw_record` (~314), and the tests that construct `PendingAtom` (~819, ~833, ~856, ~1009, ~1053, ~1105)
- Test: `src/chunk_assembler.rs` (`mod tests`) — existing tests updated, one new

**Interfaces:**
- Consumes: `PresenceMasks` (Task 1), `or_mask_into` (Task 2).
- Produces: `enum AtomCalls { Masks { masks: Arc<PresenceMasks>, slot: u16 }, Sparse(Arc<Calls>) }` as `PendingAtom.calls`. Task 4 does not touch it.

**Context the implementer needs:** this is the task that actually frees the memory. `decompose_raw_record` currently does `Arc::new(rec.calls)` — a **move**. The `Vec<(u64, RawRecord)>` staged by `fill_normalize_batch` and the `Arc<Calls>` the heap later holds are therefore the *same* allocation. Building masks and dropping the vector at the end of `decompose_raw_record` is what breaks that hand-off.

`Calls::Sparse` must stay exactly as it is: it is already O(carriers), and `flush_window` needs its carriers. `from_vcf_list` is the only source that produces it, and it must not regress.

- [ ] **Step 1: Write the failing test**

Add to `mod tests`:

```rust
#[test]
fn decompose_retains_masks_not_the_allele_vector_for_dense_sources() {
    // The memory claim, asserted structurally rather than by measurement: after
    // decomposition a dense record's atoms must not hold anything sized like
    // `columns * 4`. If this ever reverts to `AtomCalls::Sparse` or to a
    // retained `Calls::Dense`, issue #155's ratchet is back.
    let columns = 128usize;
    let mut gt = vec![0i32; columns];
    gt[7] = 1;
    let rec = RawRecord {
        pos: 100,
        reference: b"A".to_vec(),
        alts: vec![b"C".to_vec()],
        calls: crate::record_source::Calls::Dense(gt),
        format_vals: FormatVals::Dense(Vec::new()),
        info_raw: Vec::new(),
        global_idx: -1,
    };
    let d = decompose_raw_record(rec, 0, &[], false, true, crate::normalize::CheckRef::Ignore, &[], "chrT")
        .expect("decompose");
    assert_eq!(d.atoms.len(), 1);
    match &d.atoms[0].calls {
        AtomCalls::Masks { masks, slot } => {
            assert_eq!(masks.mask(*slot)[0], 1u64 << 7);
        }
        AtomCalls::Sparse(_) => panic!("dense source must retain masks, not calls"),
    }
}
```

Construct `RawRecord` with whatever field set the struct actually declares in `src/record_source.rs` — read it first and match it exactly rather than trusting the field list above, and use the same `CheckRef` variant the neighbouring `decompose_raw_record` tests in this module already pass.

- [ ] **Step 2: Run it to verify it fails**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion decompose_retains_masks
```
Expected: FAIL — `cannot find type AtomCalls`.

- [ ] **Step 3: Introduce `AtomCalls` and rewire the four call sites**

**(a)** Add above `struct PendingAtom`:

```rust
// What an atom retains in order to pack its presence row later.
//
// Dense sources keep a bitset (`columns/8` bytes, shared across the atoms of one
// record) rather than the record's allele vector (`columns*4`). Sparse sources
// keep their calls verbatim -- already O(carriers), and `flush_window` needs the
// carriers themselves, not just their bits.
enum AtomCalls {
    Masks {
        masks: Arc<PresenceMasks>,
        /// This atom's slot within the record's slab.
        slot: u16,
    },
    Sparse(Arc<Calls>),
}
```

**(b)** In `struct PendingAtom`, replace

```rust
    calls: Arc<Calls>, // shared across the atoms decomposed from one record
```

with

```rust
    // Shared across the atoms decomposed from one record. For dense sources this
    // is a presence bitset, NOT the allele vector: retaining the vector is what
    // made the reader cost ~8 KB per sample per contig (issue #155).
    calls: AtomCalls,
```

**(c)** In `pack_row`, replace the `match a.calls.as_ref()` block's dense arm. The new body:

```rust
    let src = a.source_alt_index as i32;
    let base = vi * columns;
    match &a.calls {
        AtomCalls::Masks { masks, slot } => {
            or_mask_into(words, word_base, base, masks.mask(*slot), columns);
        }
        AtomCalls::Sparse(calls) => {
            // Only the carriers can match `src`; every other column is REF and packs 0.
            // This is the O(carriers) path that replaces the O(columns) scan.
            for (col, allele) in calls.iter_non_ref() {
                if allele == src {
                    let flat = base + col as usize;
                    let w = (flat >> 6) - word_base;
                    // SAFETY: col < columns by construction (see VcfListRecordSource).
                    unsafe {
                        *words.get_unchecked_mut(w) |= 1u64 << (flat & 63);
                    }
                }
            }
        }
    }
```

`src` is now unused in the `Masks` arm; if clippy complains, move the `let src` binding into the `Sparse` arm.

**(d)** In `flush_window`, replace the carrier match:

```rust
        let carriers = match &a.calls {
            AtomCalls::Masks { .. } => None,
            AtomCalls::Sparse(calls) => {
                let src = a.source_alt_index as i32;
                let mut c = Carriers::new();
                for (col, allele) in calls.iter_non_ref() {
                    if allele == src {
                        c.push(col, allele);
                    }
                }
                Some(c)
            }
        };
```

**(e)** In `decompose_raw_record`, reorder so the allele vector dies inside the function. Today `let calls = Arc::new(rec.calls);` runs at the top, before `atomize_record`. It must run *after*, because the slot list is derived from the atoms. Replace the top-of-function `let calls = ...` line by moving that work below `atomize_record`:

```rust
    let alt_refs: Vec<&[u8]> = rec.alts.iter().map(|a| a.as_slice()).collect();
    let mut atoms = Vec::new();
    let dropped = atomize_record(
        pos,
        &rec.reference,
        &alt_refs,
        &mut atoms,
        skip_out_of_scope,
    )?;
    // Ends the borrow of `rec.alts` so `rec.calls` can be moved out below.
    drop(alt_refs);

    // Collapse the record's calls into what its atoms will actually retain.
    // For a dense source that is one presence bitset per in-scope ALT; the
    // `Vec<i32>` is dropped at the end of this match, which is the whole point
    // -- 1.024 MB per record at S=128,000 that used to survive into the heap.
    enum RecordCalls {
        Masks {
            masks: Arc<PresenceMasks>,
            /// Indexed by `source_alt_index`; `u16::MAX` for alleles with no slot.
            slot_of: Vec<u16>,
        },
        Sparse(Arc<Calls>),
    }
    let record_calls = match rec.calls {
        Calls::Dense(gt) => {
            let columns = gt.len();
            let mut wanted: Vec<u16> = atoms.iter().map(|a| a.source_alt_index).collect();
            wanted.sort_unstable();
            wanted.dedup();
            let mut slot_of =
                vec![u16::MAX; wanted.iter().copied().max().unwrap_or(0) as usize + 1];
            for (slot, &allele) in wanted.iter().enumerate() {
                slot_of[allele as usize] = slot as u16;
            }
            RecordCalls::Masks {
                masks: Arc::new(PresenceMasks::from_dense(&gt, columns, &wanted)),
                slot_of,
            }
        }
        sparse @ Calls::Sparse(_) => RecordCalls::Sparse(Arc::new(sparse)),
    };
```

Then, in the `for (atom_ix, atom) in atoms.into_iter().enumerate()` loop, replace `calls: Arc::clone(&calls),` with:

```rust
            calls: match &record_calls {
                RecordCalls::Masks { masks, slot_of } => AtomCalls::Masks {
                    masks: Arc::clone(masks),
                    slot: slot_of[atom.source_alt_index as usize],
                },
                RecordCalls::Sparse(c) => AtomCalls::Sparse(Arc::clone(c)),
            },
```

The `CheckRef::Exclude` early return above this block is unchanged — it returns before any of it runs.

**(f)** Update every test that builds a `PendingAtom` literal. The idiom for a dense atom becomes:

```rust
        calls: {
            let m = PresenceMasks::from_dense(&gt, columns, &[src_alt]);
            AtomCalls::Masks { masks: std::sync::Arc::new(m), slot: 0 }
        },
```

and for a sparse atom `AtomCalls::Sparse(std::sync::Arc::new(Calls::Sparse(carriers)))`. Where a test's `mk` closure takes a `Calls` and builds both encodings (`pack_row_sparse_and_dense_produce_identical_bits`, `pack_row_sparse_matches_dense_across_word_boundaries`, `test_par_packing_matches_seq`), change the closure to take `AtomCalls` and build each side at the call site. **Do not weaken any of these assertions** — they are the migration's bit-identity guarantee, and `test_par_packing_matches_seq` must keep generating a *mix* of mask-backed and sparse atoms so the parallel path still sees `Sparse` at a nonzero `word_base`.

- [ ] **Step 4: Run the full Rust suite**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo check --no-default-features
```
Expected: all pass (458+ tests as of `fab677f`), no new warnings. Report the actual counts; do not assert "all pass" without the output.

- [ ] **Step 5: Commit**

```bash
git add src/chunk_assembler.rs
git commit -m "perf(svar2): retain presence masks, not allele vectors, per atom

decompose_raw_record moved rec.calls into an Arc, so the batch's staged
Vec<i32> and the heap's retained calls were the same allocation -- columns*4
bytes per record surviving all the way to the pack. It now collapses to one
bitset per in-scope ALT and the vector dies in the function: columns/8 bytes,
32x smaller (issue #155).

Calls::Sparse is untouched -- already O(carriers), and flush_window needs the
carriers themselves -- so from_vcf_list is unaffected."
```

---

### Task 4: Byte budgets for the two reader buffers

**Files:**
- Modify: `src/chunk_assembler.rs` — the constant block at ~202-224, `fill_normalize_batch` (~509), `read_next_chunk` (~645), `flush_window`'s parallel gate (~250)
- Test: `src/chunk_assembler.rs` (`mod tests`)

**Interfaces:**
- Consumes: `PresenceMasks` (Task 1), `AtomCalls` (Task 3).
- Produces: `fn batch_records(columns: usize) -> usize`, `fn pack_window(columns: usize) -> usize`, `const PARALLEL_MIN_CELLS: usize`. Task 7 tunes `PARALLEL_MIN_CELLS`; nothing else consumes these.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn batch_records_bounds_staged_bytes_at_every_cohort_width() {
    for &s in &[100usize, 2_000, 32_000, 128_000, 500_000] {
        let columns = s * 2;
        let bytes = batch_records(columns) * columns * 4;
        assert!(
            bytes <= RAW_STAGE_BYTES,
            "S={s} stages {bytes} B against a {RAW_STAGE_BYTES} B budget"
        );
    }
}

#[test]
fn batch_records_holds_todays_value_only_for_narrow_cohorts() {
    // The cap binds up to columns = RAW_STAGE_BYTES/(4*MAX_BATCH_RECORDS) =
    // 16,384, i.e. S = 8,192 at ploidy 2.
    assert_eq!(batch_records(16_384), MAX_BATCH_RECORDS);
    // Wider cohorts DO change: S=32,000 -- inside RamLaw::PGEN's fitted domain --
    // stages 262 records rather than 1,024. That is the fix working, not a
    // regression, and it is precisely why the law must be re-fitted (Task 9)
    // rather than carried over.
    assert_eq!(batch_records(64_000), 262);
}

#[test]
fn the_batch_floor_is_the_documented_limit_of_the_bound() {
    // Past columns == RAW_STAGE_BYTES/(4*MIN_BATCH_RECORDS) -- about S=1,000,000
    // at ploidy 2 -- the floor binds and staging resumes growing with S. Asserted
    // rather than hidden: this is the boundary of what the budget promises.
    assert_eq!(batch_records(4_000_000), MIN_BATCH_RECORDS);
}

#[test]
fn pack_window_bounds_retained_mask_bytes() {
    for &s in &[100usize, 2_000, 128_000, 500_000] {
        let columns = s * 2;
        let bytes = pack_window(columns) * columns.div_ceil(64) * 8;
        assert!(bytes <= MASK_STAGE_BYTES, "S={s} retains {bytes} B");
    }
}

#[test]
fn pack_window_stays_at_todays_value_until_far_past_the_fitted_domain() {
    // Masks are what keep this non-binding in the normal regime: budgeting the
    // same bytes over raw calls would give ~65 records at S=128,000.
    assert_eq!(pack_window(256_000), MAX_PACK_WINDOW); // S=128,000
    assert!(pack_window(1_000_000) < MAX_PACK_WINDOW); // S=500,000
}
```

- [ ] **Step 2: Run to verify they fail**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion _records _window
```
Expected: FAIL — `cannot find function batch_records`.

- [ ] **Step 3: Replace the two constants with budgets**

Replace the whole comment block and both `const` lines at `src/chunk_assembler.rs:202-224` with:

```rust
// *** These two budgets set the reader's peak RAM. ***
//
// They used to be fixed VARIANT counts (`PACK_WINDOW`/`NORMALIZE_BATCH_RECORDS`,
// both 1024) multiplying an O(n_samples) payload, giving a live set of
// `min(V, 2048) * n_samples * ploidy * 4` bytes -- up to 16 KB per sample,
// bounded by neither `chunk_size` nor `max_mem`, and the blocker behind
// `RamLaw::PGEN`'s conservative margin (issue #155).
//
// They are BYTE budgets now, in the units each buffer actually holds:
//
//   * `RAW_STAGE_BYTES` bounds `fill_normalize_batch`'s staged `RawRecord`s,
//     which still carry `Calls::Dense` -- `columns * 4` bytes each.
//   * `MASK_STAGE_BYTES` bounds the atoms a pack window RETAINS, which carry
//     `PresenceMasks` -- `columns/8` bytes each, 32x smaller. That 32x is why
//     the window stays large in the normal regime instead of collapsing to ~65
//     records at S=128,000.
//
// Deliberately constants rather than derived from `max_mem`: threading a budget
// into `ChunkAssembler` would add a regressor to `RamLaw::PGEN`, whereas a
// constant lands in `base_mb` where a constant belongs, and the bound stays
// checkable by arithmetic instead of only by measurement.
const RAW_STAGE_BYTES: usize = 64 << 20;
const MASK_STAGE_BYTES: usize = 64 << 20;

// Caps preserve today's value as the ceiling. They bind -- i.e. nothing changes
// -- only while a buffer's per-record cost keeps 1024 records inside the budget:
// up to S = 8,192 for the batch, and up to S = 262,144 for the window, the
// latter because masks are 32x cheaper per record. Between those, the batch
// shrinks with cohort width. That is the fix, and it is why `RamLaw::PGEN` has
// to be re-fitted rather than carried over.
const MAX_BATCH_RECORDS: usize = 1024;
const MAX_PACK_WINDOW: usize = 1024;

// Floors are small ON PURPOSE. A thread-scaled floor would defeat the budget:
// 48 threads x 4 records is 192 records, which at S=128,000 is 197 MB against a
// 64 MiB budget. At 8, the floor binds only when one record exceeds an eighth of
// the budget -- roughly S = 1,000,000 -- and even then the batch costs exactly
// the budget rather than a multiple of it. Past that width staging resumes
// growing with S and decode has only 8 tasks to spread; each is 8+ MB of work,
// so the pool is coarsely fed rather than starved. That is a stated limit of the
// bound, not a claim that it holds everywhere.
const MIN_BATCH_RECORDS: usize = 8;
const MIN_PACK_WINDOW: usize = 8;

/// Records `fill_normalize_batch` stages before decomposing them.
fn batch_records(columns: usize) -> usize {
    (RAW_STAGE_BYTES / (columns * 4).max(1)).clamp(MIN_BATCH_RECORDS, MAX_BATCH_RECORDS)
}

/// Atoms buffered before their presence bits are flushed into the chunk's grid.
///
/// The CALLER rounds this up to a multiple of the word-aligned block size
/// `g = 64/gcd(columns, 64)`, so every flush offset lands on a u64 boundary and
/// `pack_presence_par` keeps its word-disjoint invariant. That rounding can
/// exceed the budget by at most `g - 1 <= 63` records -- a few percent at the
/// widths where the budget binds, and zero whenever `columns` is a multiple of 64.
fn pack_window(columns: usize) -> usize {
    (MASK_STAGE_BYTES / (columns.div_ceil(64) * 8).max(1))
        .clamp(MIN_PACK_WINDOW, MAX_PACK_WINDOW)
}
```

- [ ] **Step 4: Wire them in**

In `fill_normalize_batch`, replace the two `NORMALIZE_BATCH_RECORDS` uses:

```rust
        let cap = batch_records(self.num_samples * self.ploidy);
        let mut records = Vec::with_capacity(cap);
        while records.len() < cap {
```

In `read_next_chunk`, replace the `window` derivation:

```rust
        let window = pack_window(columns).div_ceil(g) * g;
```

In `flush_window`, replace the parallel gate. Change the signature to take `columns` (it already does) and swap the condition:

```rust
    let parallel = matches!(pool, Some(p) if p.current_num_threads() >= 2)
        && buf.len().saturating_mul(columns) >= PARALLEL_MIN_CELLS;
```

Replace `const PARALLEL_MIN_VARIANTS: usize = 512;` and its comment with:

```rust
// Below this much packing work in a window, parallel packing's per-task overhead
// outweighs the win -- pack sequentially instead.
//
// CELLS, not variants. A variant count is a threshold on the wrong quantity: the
// work is `variants * columns`, so a cell-budgeted window at large `S` drops
// below any fixed variant count and silently disengages parallel packing.
//
// PROVISIONAL until Task 7 measures it. Seeded at the product that reproduces
// today's gate (512 variants) at a 1,024-column cohort, so narrow cohorts behave
// as they did; `or_mask_into` made packing ~64x cheaper, so the measured value
// may well be higher.
const PARALLEL_MIN_CELLS: usize = 512 * 1_024;
```

- [ ] **Step 5: Run the full suite**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo check --no-default-features
```
Expected: all pass. Report actual counts.

- [ ] **Step 6: Commit**

```bash
git add src/chunk_assembler.rs
git commit -m "perf(svar2): budget the reader's two buffers in bytes, not variants

NORMALIZE_BATCH_RECORDS and PACK_WINDOW were fixed variant counts multiplying an
O(n_samples) payload, so the reader's live set was unbounded in cohort width.
Each becomes a byte budget in the units its buffer holds, and the parallel-pack
gate becomes a cell count -- a variant threshold on a cell-sized decision would
disengage silently at large S. See issue #155."
```

---

### Task 5: `_auto_chunk_size` must not override its own budget

*Independent of Tasks 1-4 — dispatch in parallel.*

**Files:**
- Modify: `python/genoray/_svar2.py:2238-2261`
- Test: `tests/test_svar2_chunk_size.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_auto_chunk_size` with a floor of 1 instead of 1024. Task 8's plan rows and Task 9's refit both depend on this landing first, because its output is `PlanInputs.chunk_bytes` — the `kappa` regressor.

**Context the implementer needs:** two existing tests encode the old floor and must be updated, not deleted — `test_chunk_size_never_goes_below_the_floor` (line 24) and the `max(1024, ...)` in `test_chunk_size_respects_an_explicit_budget` (line 21).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_svar2_chunk_size.py`:

```python
import pytest

from genoray._svar2 import _DENSE_CHUNK_TARGET_BYTES, _STAGED_FORMAT_BYTES


@pytest.mark.parametrize("n_samples", [7089, 128_000, 500_000, 2_000_000])
@pytest.mark.parametrize("n_format_fields", [0, 7])
def test_chunk_never_exceeds_the_budget_it_was_given(
    n_samples: int, n_format_fields: int
) -> None:
    """The invariant the docstring claims. The old `max(1024, ...)` floor broke it
    exactly where it mattered: at S=2,000,000 the budget wants 536 variants and
    got 1024, i.e. a ~512 MB chunk against a 256 MiB target."""
    cs = _auto_chunk_size(n_samples, 2, n_format_fields=n_format_fields)
    per_variant = (n_samples * 2) // 8 + n_format_fields * n_samples * _STAGED_FORMAT_BYTES
    assert cs * per_variant <= _DENSE_CHUNK_TARGET_BYTES


def test_a_budget_too_small_for_one_variant_still_makes_progress() -> None:
    assert _auto_chunk_size(10_000_000, 2, n_format_fields=7, max_mem=1024) == 1
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/test_svar2_chunk_size.py -v
```
Expected: `test_chunk_never_exceeds_the_budget_it_was_given` FAILS at large `n_samples`; `test_a_budget_too_small_for_one_variant_still_makes_progress` FAILS with `1024 != 1`.

- [ ] **Step 3: Lower the floor**

In `python/genoray/_svar2.py`, replace the final line of `_auto_chunk_size`:

```python
    chunk_size = max(1, min(25_000, budget // per_variant))
    if chunk_size < 256:
        warnings.warn(
            f"budget of {budget} B affords only {chunk_size} variants per dense "
            f"chunk at n_samples={n_samples}, ploidy={ploidy}, "
            f"n_format_fields={n_format_fields}; per-chunk overhead starts to "
            "matter below ~256. Raise max_mem or request fewer FORMAT fields.",
            stacklevel=2,
        )
    return chunk_size
```

Append to the docstring:

```
    There is no floor beyond 1. A floor of 1024 shipped previously and broke
    this function's whole contract at biobank width -- at S=2,000,000 the budget
    affords 536 variants and the floor returned 1024, i.e. a ~512 MB chunk
    against a 256 MiB target. Lowering it is cheap: `plans/build_plans.py`
    records chunk-size wall-time sensitivity under 3% across a 400x range
    (S=500,000 ran 41.6 s at chunk_size=87 and 41.0 s at 25,000).
```

- [ ] **Step 4: Update the two tests that encode the old floor**

```python
def test_chunk_size_respects_an_explicit_budget() -> None:
    small = _auto_chunk_size(7089, 2, n_format_fields=7, max_mem=256 * 1024**2)
    big = _auto_chunk_size(7089, 2, n_format_fields=7, max_mem=4 * 1024**3)
    assert small < big
    # 256 MiB / (7 fields * 7089 samples * 4 B + 7089*2/8 B) per variant
    assert small == (256 * 1024**2) // (7 * 7089 * 4 + 7089 * 2 // 8)


def test_a_tiny_budget_warns_rather_than_silently_ignoring_itself() -> None:
    # Replaces test_chunk_size_never_goes_below_the_floor. The old floor did not
    # protect anything -- it silently returned a chunk 2x the budget.
    with pytest.warns(UserWarning, match="per dense chunk"):
        cs = _auto_chunk_size(10_000_000, 2, n_format_fields=7, max_mem=1024)
    assert cs == 1
```

Delete `test_chunk_size_never_goes_below_the_floor`.

- [ ] **Step 5: Run the Python suite**

```bash
pixi run pytest tests/test_svar2_chunk_size.py tests/test_memory_budget.py -v
pixi run test
```
Expected: all pass. Report actual counts. If any *other* test asserts a 1024 chunk size, update it and say which — do not loosen an assertion to make it pass.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2.py tests/test_svar2_chunk_size.py
git commit -m "fix(svar2): stop _auto_chunk_size overriding its own byte budget

The max(1024, ..) floor won exactly where the budget mattered: at S=2,000,000 it
returned 1024 variants against a budget affording 536, i.e. a ~512 MB chunk
under a 256 MiB target. Floor is 1 now, with a warning below 256. Chunk-size
wall-time sensitivity is under 3% across a 400x range, so this is cheap."
```

---

### Task 6: End-to-end store byte-identity

**Files:**
- Create: `tests/test_svar2_reader_identity.py`
- Test: itself

**Interfaces:**
- Consumes: Tasks 4 and 5 landed.
- Produces: nothing; a gate.

**Context the implementer needs:** `tests/_oracle.py` exposes `store_digest(path)` — an order-independent digest over every file in a store, already used by `tests/test_svar2_schedule_invariance.py`. Reuse it; do not write a new digest.

The test cannot compare "before and after" within one process, so it asserts the property the change must preserve: **every source produces the digest it produced at `fab677f`**. Capture those digests by running the fixtures on `fab677f` first.

- [ ] **Step 1: Capture the pre-change digests**

```bash
git stash push -u -m "reader-identity-wip-$$"
git stash list --format='%H %gs' | head -3   # capture YOUR entry's SHA
git checkout fab677f
pixi run maturin develop --release
# run the fixture conversions, print digests
git checkout worktree-svar2-pgen-budget-planner
pixi run maturin develop --release
git stash apply <sha>   # NOT pop; the stash stack is shared across worktrees
```

Record the digests in the test as literals with a comment naming the commit they came from.

- [ ] **Step 2: Write the test**

```python
"""Byte-identity across the presence-mask migration (issue #155).

The reader stopped retaining `Calls::Dense` and started retaining presence
bitsets. That is a representation change with no semantic content, so every
source must still produce exactly the store it produced before. Digests below
were captured at commit fab677f, the last commit before the migration.
"""

import pytest

from tests._oracle import store_digest

# Filled in from Step 1. One entry per source; do not relax these to
# "digests agree with each other" -- that would pass if every source broke
# identically.
EXPECTED = {
    "pgen": "...",
    "vcf": "...",
    "svar1": "...",
    "vcf_list": "...",
}


@pytest.mark.parametrize("source", sorted(EXPECTED))
def test_store_is_byte_identical_to_the_pre_mask_reader(source, tmp_path, ...):
    out = tmp_path / f"{source}.svar"
    # convert via the matching from_* entry point
    assert store_digest(out) == EXPECTED[source]
```

Build the four fixtures from the existing conftest fixtures rather than inventing corpora — read `tests/conftest.py` and `tests/test_svar2_schedule_invariance.py` for the established pattern. `vcf_list` is the control: it produces `Calls::Sparse` and must be provably untouched.

- [ ] **Step 3: Run it**

```bash
pixi run maturin develop --release
pixi run pytest tests/test_svar2_reader_identity.py -v
```
Expected: 4 passed. **A failure here means the migration changed output and must be fixed, not re-baselined.**

- [ ] **Step 4: Run everything**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag cargo check --no-default-features
pixi run test
```
Report actual counts for each.

- [ ] **Step 5: Commit**

```bash
git add tests/test_svar2_reader_identity.py
git commit -m "test(svar2): pin store byte-identity across the presence-mask migration

Digests captured at fab677f, the last commit before the reader stopped
retaining Calls::Dense. from_vcf_list is the control: it produces Calls::Sparse
and is untouched by design."
```

---

### Task 7: Measure the RSS win and set `PARALLEL_MIN_CELLS`

**Files:**
- Modify: `src/chunk_assembler.rs` (the `PARALLEL_MIN_CELLS` value and its comment)
- Create: `docs/superpowers/plans/results/2026-08-06-reader-cell-budget-measurement.md`

**Interfaces:**
- Consumes: Task 6 green.
- Produces: a measured `PARALLEL_MIN_CELLS`; the RSS numbers Task 9's refit is expected to reproduce.

**Context the implementer needs:** the `rss_mark` instrumentation shipped in `a93d1fc`. One `log_level="debug"` run emits `RSSMARK <label> rss_mb=<n>` at `contig_enter`, `reader_ready`, `reader_drained`, `pipeline_joined`, `contig_exit`. Rich injects ANSI escapes *inside* numbers, so de-ANSI before parsing: `sed -r 's/\x1b\[[0-9;]*[mK]//g'`.

Corpora at `/local/dlaub/pgen-mem` (`s2000`, `s8000`, `s32000`, `s128000`) are node-local to **carter-cn-03** and may have been reaped; regenerate with `$CLAUDE_JOB_DIR/tmp/gen_pgen.py` if missing.

- [ ] **Step 1: Run the RSS ladder under sbatch**

Pin `--nodelist`. Single contig, 1,000 variants, `chunk_size=4096`, `cc=1`, at S = 2,000 / 8,000 / 32,000 / 128,000. Baseline for comparison, from `a93d1fc`: reader delta 13 / 42 / 256 / 1,020 MB = 7.99 KB per sample.

Expected after the change: ~102 MB at S=128,000 (batch 67 + window 33 + heap ~2), i.e. ~0.8 KB per sample. **If the measured value is materially above that, stop and diagnose before proceeding** — the arithmetic in Task 4's tests is exact, so a miss means something else retains per-sample state.

- [ ] **Step 2: Sweep `PARALLEL_MIN_CELLS` for wall time**

Same pinned node, same corpora, sweeping the constant across at least `{0 (always parallel), 512*1024, 8*512*1024, usize::MAX (never parallel)}`. Record wall time per S. Pick the value that is not worse than the others at any measured width; if they are within noise, keep the seeded value and **record that the measurement showed no difference** — that is the finding, not a reason to skip it.

- [ ] **Step 3: Write the results doc**

Include the node name, the commit, both tables, and the chosen constant with its justification. Note explicitly which corpora were regenerated versus reused.

- [ ] **Step 4: Update the constant and its comment**

Replace "PROVISIONAL until Task 7 measures it" with the measured basis and a pointer to the results doc.

- [ ] **Step 5: Commit**

```bash
git add src/chunk_assembler.rs docs/superpowers/plans/results/
git commit -m "perf(svar2): measure the reader cell budget, set PARALLEL_MIN_CELLS

<one line with the measured KB/sample before and after, and the node>"
```

---

### Task 8: Give the PGEN sweep a design matrix that identifies kappa

*Independent of Tasks 1-4 and 6-7 — dispatch in parallel with Task 1 and Task 5. Do not RUN the sweep here (that is Task 9); this task only changes the plan.*

**Files:**
- Modify: `scripts/bench_svar2/plans/build_plans.py` (`PGEN_LADDERS` ~147, the PGEN loop ~353)
- Modify: `scripts/bench_svar2/sweep_pgen.sbatch` (corpus generation loop)
- Test: `tests/bench/test_model.py` or a new assertion in the plan builder

**Interfaces:**
- Consumes: nothing at build time.
- Produces: `plans/pgen.json` rows carrying the new chunk-size axis and the S=128,000 rung.

**Context the implementer needs:**

The current fit's `kappa` has SE 7.44 and a 95% CI of [-9.99, +23.68]. `_chunk_size_for(v)` depends only on V, and both ladders use the same three V values, so `chunk_size ∈ {7812, 15625, 25000}` at *both* cohort widths. `[1, S, S·cs]` is technically identifiable, but `cs` spans 3.2× against `S`'s 8× and they multiply, so the two regressors stay correlated and the SE blows up.

A corpus's `variants` is the **total** across all 22 contigs (`vcfixture bulk --records`), so V=1,000,000 is ~45,454 per contig. **Every chunk_size must leave at least one chunk per contig**: `BitGrid3::zeros` reserves the full `chunk_size` up front and truncates afterwards, so a partial chunk breaks the very linearity `kappa` measures.

- [ ] **Step 1: Write the failing assertion**

Add a test asserting the plan's PGEN rows have (a) at least three distinct `chunk_size` values at a fixed `(samples, variants)`, and (b) at least three distinct cohort widths, and (c) `variants / 22 / chunk_size >= 1.0` for every row.

- [ ] **Step 2: Run it to verify it fails**

```bash
pixi run pytest tests/bench/ -k pgen -v
```

- [ ] **Step 3: Add the chunk-size axis**

At V=1,000,000, at **both** existing cohort widths, emit rows at `chunk_size ∈ {3_125, 12_500, 25_000}` — 14.5, 3.6 and 1.8 chunks per contig, an 8× `chunk_bytes` range at constant `S`. Six extra conversion runs; existing corpora, no new generation.

- [ ] **Step 4: Add the S=128,000 rung**

`(128_000, (250_000,))` with `chunk_size ∈ {3_125, 7_812}` (3.6 and 1.5 chunks per contig, matching the existing ladder's regime). This takes `per_sample_mb`'s extrapolation to the 500,000 target from **15.6× down to 3.9×**. Generation cost ~5.5 h (3.2e10 cells at the measured ~6.15e-7 s/cell), inside the job's 72 h wall.

Add a comment at `PGEN_LADDERS` recording *why* the shape is what it is, in the style of the existing "two V-ladders at DIFFERENT cohort widths" comment — that comment exists because a bad ladder shape already produced one published-then-retracted interval in this project.

- [ ] **Step 5: Verify and commit**

```bash
pixi run pytest tests/bench/ -v
pixi run python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir /tmp/nonexistent --out-dir "$CLAUDE_JOB_DIR/tmp/plancheck" --threads 48
```
The second command only builds JSON (no corpora needed); inspect `pgen.json` and confirm the row count and the chunk-size spread by eye before committing.

```bash
git add scripts/bench_svar2/plans/build_plans.py scripts/bench_svar2/sweep_pgen.sbatch tests/bench/
git commit -m "test(bench): give the PGEN sweep a matrix that can identify kappa

_chunk_size_for depends only on V and both ladders share the same three V
values, so S and S*chunk_size stayed correlated and kappa's CI spanned zero.
Varying chunk_size at FIXED (S, V) gives an 8x chunk_bytes range at constant S,
and an S=128,000 rung cuts per_sample_mb's extrapolation from 15.6x to 3.9x.
Every chunk_size keeps >=1 chunk per contig: BitGrid3::zeros reserves the full
chunk_size, so partial chunks break the linearity kappa measures."
```

---

### Task 9: Re-fit `RamLaw::PGEN`

**Files:**
- Modify: `src/budget.rs` (the `RamLaw::PGEN` const ~120-150 and its doc comment)
- Modify: `src/budget.rs` tests `ram_law_pgen_is_a_usable_law`, `pgen_memory_bound_actually_binds`, `pgen_budget_too_small_for_one_contig_is_an_error_not_a_silent_cc_of_one`
- Create: `docs/superpowers/plans/results/2026-08-06-pgen-ram-law-refit.md`

**Interfaces:**
- Consumes: Tasks 4, 5, 7, 8 all landed and committed.
- Produces: new coefficients. Nothing downstream in this plan.

**Acceptance criteria, fixed now so the result cannot be rationalised afterwards:**
- The refit ships **only if** it over-predicts at every measured point the way `plan_sharded` evaluates it — the standard the current law met (worst-case margin +621 MB / 1.16× at cc=8, S=4,000).
- The doc comment records R², n, kappa's CI, and the validity domain, matching the existing convention in `budget.rs`.
- If kappa's CI still spans zero after decorrelation, it stays labelled a **conservative bound**, not a fitted rate. The margin trades under-utilisation and a spurious `PlanError::InsufficientMemory` against an OOM; it is not slack to be tuned away.

- [ ] **Step 1: Submit the sweep**

```bash
sbatch scripts/bench_svar2/sweep_pgen.sbatch
```
Pinned to `carter-cn-04`, 72 h wall. `/local/dlaub/pgen-sweep` is node-local to cn-04 and could not be checked from cn-03 — **if the six existing corpora were reaped, generation adds ~10.8 h**; the job caches on the full spec, so a resubmit after a wall-clock kill skips whatever survived. Do not poll on a short interval; check back on the order of the job's own timescale.

- [ ] **Step 2: Fit and check the acceptance criteria**

Run `fit_ram_law` over the new rows. Report `base_mb`, `per_sample_mb`, `kappa`, R², n, and kappa's SE/CI. Then evaluate the fitted law at every measured point the way `plan_sharded` does and confirm it over-predicts at all of them. **If it under-predicts anywhere, do not ship it** — report the point and stop.

- [ ] **Step 3: Update the constant, its doc comment, and the tests**

Update the doc comment's fitted date, R², n, kappa CI, and validity domain. The `PGEN_MAX_CONCURRENT` doc comment's caveat about `processing_threads_for` still applies and should be reviewed for whether this sweep resolves it.

`pgen_memory_bound_actually_binds` derives its budget from the real coefficients and asserts `concurrent_chroms == 2`; re-derive rather than hand-patching the expected value. `pgen_budget_too_small_for_one_contig_...` quotes a "~123,098 MB at S=1,000,000" baseline in its comment — recompute it.

- [ ] **Step 4: Write the results doc**

Node, commit, full row table, fitted coefficients with CIs, the over-prediction check at every point, and a before/after comparison of the projected host requirement at S=500,000 (currently ~79 GiB).

- [ ] **Step 5: Verify and commit**

```bash
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion budget
CARGO_TARGET_DIR=/local/$USER/cargo-target-diag \
  cargo test --no-default-features --features conversion
pixi run test
```

```bash
git add src/budget.rs docs/superpowers/plans/results/
git commit -m "perf(svar2): re-fit RamLaw::PGEN on the bounded reader

<coefficients, R^2, n, kappa CI, and the new projected host requirement at
S=500,000 against the old ~79 GiB>"
```

- [ ] **Step 6: Close out**

Update issue #155 with the measured before/after, comment on PR #154, and update the memory file `pgen-ram-scales-per-sample-per-contig.md` — its "fix is NOT a constant tweak" framing and the `PARALLEL_MIN_VARIANTS` collision note both become history once this lands.

---

## Self-Review

**Spec coverage.** Design §1 masks → Tasks 1-3. §2 batch budget, §3 window budget, §4 `PARALLEL_MIN_CELLS` → Task 4 (value measured in Task 7). §5 `_auto_chunk_size` → Task 5. §6 refit → Tasks 8-9. Verification §1 unit/property → Tasks 1, 2. §2 arithmetic → Task 4. §3 e2e byte identity → Task 6. §4 full suites → every task. §5 RSS → Task 7 Step 1. §6 wall time → Task 7 Step 2. No uncovered requirement.

**Type consistency.** `PresenceMasks::from_dense(gt, columns, wanted) -> PresenceMasks` and `mask(slot) -> &[u64]` are defined in Task 1 and used with those exact signatures in Tasks 2 and 3. `or_mask_into(words, word_base, base, mask, columns)` is defined in Task 2 and called in Task 3(c). `AtomCalls::{Masks{masks,slot}, Sparse}` is defined in Task 3(a) and matched in 3(c), 3(d), 3(e), 3(f). `batch_records`/`pack_window`/`PARALLEL_MIN_CELLS` are defined and consumed within Task 4, and `PARALLEL_MIN_CELLS` is re-tuned in Task 7.

**Known soft spots, flagged rather than papered over.** Task 6 Step 2 leaves the fixture wiring to the implementer against `tests/conftest.py` — the digests themselves are captured in Step 1, so the gate is real, but the fixture plumbing is not spelled out line by line because the existing fixture names must be read, not guessed. Task 3 Step 1's `RawRecord` literal likewise must be checked against `src/record_source.rs` rather than trusted.

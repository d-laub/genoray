# SVAR2 reader cell budget, and the RAM-law refit it forces

**Date:** 2026-08-06
**Status:** design approved, implementation plan pending

Acts on issue #155, which was opened from the measurement in `a93d1fc`. Follows
`2026-08-05-svar2-pgen-budget-planner-design.md`, whose `RamLaw::PGEN` this work
invalidates and re-fits.

## Problem

`SparseVar2.from_pgen`'s peak RSS scales at ~8 KB per sample per concurrently
processed contig, bounded by neither `chunk_size` nor `max_mem`. That is the
blocker behind `RamLaw::PGEN`'s conservative margin and its default refusal of
biobank-scale conversions on hosts under 80 GB.

Every record buffered in `ChunkAssembler` holds a `Calls::Dense(Vec<i32>)` — one
allele index per haplotype column, `n_samples * ploidy * 4` bytes, **1.024 MB per
record at S=128,000**. Both bounds on the reader's live set are fixed *variant*
counts (`src/chunk_assembler.rs`):

```rust
const PACK_WINDOW: usize = 1024;
const NORMALIZE_BATCH_RECORDS: usize = 1024;
```

giving a live set of

```
min(V, NORMALIZE_BATCH_RECORDS + PACK_WINDOW) * n_samples * ploidy * 4 bytes
```

up to **16 KB per sample**. `src/pgen_reader.rs:303` emits `Calls::Dense`
unconditionally, so PGEN always pays it; `vcf_reader.rs:699` and
`svar1_reader.rs:325` do too. A k-way-merge source producing `Calls::Sparse`
does not.

Measured, single contig, 1,000 variants, `chunk_size=4096`, `cc=1`, S=128,000:

```
contig_enter     471 MB
reader_ready     480 MB
reader_drained  1574 MB   <- +1094 MB, all of it here
pipeline_joined  564 MB
contig_exit      577 MB   (entire merge tail: +13 MB)
```

Confirmed by construction: halving *both* constants stops the ratchet and takes
peak 1,592 → 1,102 MB, a measured 0.55 against a predicted 512/1000 = 0.51.
Halving *either alone* changes nothing — the same gigabyte relocates to the
other buffer, which is why the first two attempts at this read as refutations.

### Where the bytes actually sit

`decompose_raw_record` does `Arc::new(rec.calls)` — a **move**, not a copy. So
the `Vec<(u64, RawRecord)>` that `fill_normalize_batch` accumulates and the
`Arc<Calls>` the heap later holds are the same allocation, handed over. This is
why the ratchet steps inside `fnb_records`, and it means the two constants need
*different* treatments rather than the same formula:

- `NORMALIZE_BATCH_RECORDS` bounds a transient staging vector of raw
  `Vec<i32>` calls. A byte budget over `columns * 4` bounds it directly.
- `PACK_WINDOW` bounds *retained* per-atom calls. Budgeting it in the same raw
  units would drive the window to ~65 records at S=128,000 — 63 flushes per
  4096-variant chunk, parallel packing permanently disengaged. It needs the
  payload to shrink first.

## Design

Four reader changes, one Python change, one refit. Common theme: stop sizing
`O(n_samples)` buffers by variant count.

### 1. Presence masks replace retained `Calls::Dense`

`Calls::Dense` is used downstream for exactly two things: `pack_row`'s
`gt[col] == source_alt_index` test, and `flush_window`'s carrier recovery —
which for dense sources returns `None` (recovering carriers from the grid is
correct and cheaper, see `DenseChunk::carriers`). So the retained payload can be
the *answer* to that test rather than its input: a bitset over haplotype
columns, one per in-scope ALT of the source record, shared across that record's
atoms exactly as the `Arc` is today.

```rust
struct PresenceMasks {
    words: Vec<u64>,        // slot-major: slot s occupies [s*W, (s+1)*W)
    words_per_mask: usize,  // W = columns.div_ceil(64)
    columns: usize,
}

enum AtomCalls {
    /// Dense sources. `slot` indexes this atom's mask within the record's slab.
    Masks { masks: Arc<PresenceMasks>, slot: u16 },
    /// Natively-sparse sources (`from_vcf_list`): kept verbatim, both to pack
    /// and to recover carriers.
    Sparse(Arc<Calls>),
}
```

`columns / 8` bytes instead of `columns * 4` — **32×**. At S=128,000 that is
32 KB per record instead of 1.024 MB. The `Vec<i32>` is dropped at the end of
`decompose_raw_record`.

Built in a **single** O(columns) pass per record: for each column, look up the
allele, map it through a small `allele -> slot` table, set the bit. Alleles
outside `1..=n_alts` (REF `0`, missing `-1`) set nothing, which is exactly what
`gt[col] == src` does for `src` in `1..=n_alts`. Slots are allocated only for
the `source_alt_index` values that actually appear in the record's atoms, so a
record whose other ALTs were dropped as out-of-scope (symbolic, breakend) pays
for one mask, not `n_alts`.

Cost accounting: today the O(columns) scan happens once per *atom* at pack time.
After this it happens once per *record* at decode time — strictly fewer passes
for multiallelic records, equal for biallelic — and it stays inside the existing
parallel `into_par_iter` over the batch. Pack time drops to a bit-shifted word
OR, `columns/64` word-ops instead of `columns` compares.

`Calls::Sparse` is untouched. It is already O(carriers), and its carriers must
be retained. `from_vcf_list` therefore carries no regression risk from this
change.

#### Bit-shifted OR, and the word-disjointness invariant

Row `vi` occupies bits `[vi*columns, (vi+1)*columns)`. With
`base = vi*columns`, `s = base & 63`, the dense arm becomes, per mask word `j`:

```
words[w0 + j]     |= masks[j] << s
words[w0 + j + 1] |= masks[j] >> (64 - s)     // only when s > 0
```

Two guards, both required:

- `s == 0` must skip the carry entirely — `>> 64` is UB.
- The carry word must be bounds-checked against the row's last word,
  `(base + columns - 1) >> 6`. Bits at or beyond `columns` in the final mask
  word are zero, so the carry is *value*-correct either way, but the write can
  still land outside the slice.

`pack_presence_par` hands each rayon task a word-disjoint slice: block
boundaries fall on multiples of `g = 64/gcd(columns, 64)` variants, and
`g * columns` is a multiple of 64 by construction. So a block ends exactly on a
word boundary, its final row has `s' == 0` at its end and cannot spill into the
next block. The invariant survives unchanged. A `debug_assert` records it.

### 2. `NORMALIZE_BATCH_RECORDS` becomes a byte budget

Over `columns * 4` — the raw staged calls:

```rust
fn batch_records(columns: usize) -> usize {
    (RAW_STAGE_BYTES / (columns * 4).max(1))
        .clamp(MIN_BATCH_RECORDS, MAX_BATCH_RECORDS)
}
```

`MAX_BATCH_RECORDS = 1024` preserves today's value as the cap, so no cohort
currently under the budget changes behaviour.

`MIN_BATCH_RECORDS = 8`, deliberately small rather than scaled by thread count.
A thread-scaled floor would defeat the budget: at 48 threads and 4 records per
thread the floor is 192 records, which at S=128,000 is 197 MB — three times the
64 MiB budget it is supposed to respect. Eight records instead means the floor
binds only when one record exceeds `RAW_STAGE_BYTES / 8`, i.e. above roughly
S = 1,000,000, and even then the batch costs exactly the budget rather than a
multiple of it.

The trade this accepts, stated rather than hidden: past S ≈ 1,000,000 the batch
stops shrinking, so staging RAM resumes growing with `S`, and decode has only 8
tasks to spread over the pool. Each task is 8+ MB of work at that width, so the
pool is not starved so much as coarsely fed. This is a documented limit of the
bound, not a claim that it holds everywhere.

### 3. `PACK_WINDOW` becomes a byte budget over mask bytes

Same shape, denominated in `columns.div_ceil(64) * 8`, then rounded up to a
multiple of `g` exactly as today so `flush_window`'s
`debug_assert_eq!((v0 * columns) % 64, 0)` continues to hold. Rounding up can
exceed the budget by at most `g - 1 <= 63` records — a few percent of the
budget at the widths where it binds, and zero whenever `columns` is a multiple
of 64.

Because of change (1) this budget does not bind until `columns > 524_288`, i.e.
**S > 262,144** at ploidy 2 — an order of magnitude past today's validity
domain, and it degrades gracefully above it (S=500,000 still gets a 536-record
window). Budgeting the same 64 MiB in *raw* units instead would give 65 records
at S=128,000 and 16 at S=500,000. That 32× is the whole reason masks come
first.

### 4. `PARALLEL_MIN_VARIANTS` becomes `PARALLEL_MIN_CELLS`

A variant-count threshold for what is really a cell-count decision: a
cell-budgeted window at large S falls below 512 records and silently disengages
parallel packing. Re-expressed as `buf.len() * columns`.

The value is **measured, not re-derived on paper**: change (1) makes packing
~64× cheaper, so the gate may want to move up rather than merely be restated. If
measurement across the corpus ladder shows no significant wall-time difference
over the plausible range, the value that reproduces today's behaviour is taken,
and the measurement is recorded as the reason.

### Why the reader budgets are constants, not `max_mem`-derived

Threading `max_mem` into `ChunkAssembler` — constructed per contig in the
orchestrator — would make the reader term a new regressor in the RAM law, which
is being re-fitted in the same work. A constant instead falls into `base_mb`,
where a constant belongs, and keeps the bound checkable by a pure-arithmetic
unit test rather than only by measurement. This is a deliberate scope line, not
an oversight: `max_mem` still does not cap the reader, but the reader is now
capped by something.

### 5. `_auto_chunk_size`'s floor (Python)

```python
return max(1024, min(25_000, budget // per_variant))
```

The `max(1024, ...)` wins over the budget exactly when the budget matters most.
At S=2,000,000 with `n_format_fields=0` the budget wants 536 variants and gets
1024, so one chunk is ~512 MB against a 256 MiB target — the function breaks the
invariant its own docstring states.

Replace the floor with 1, and warn below 256. Lowering it is cheap by this
repository's own measurement: `plans/build_plans.py` records chunk-size
wall-time sensitivity under 3% across a 400× range (S=500,000 ran 41.6 s at
`chunk_size=87` and 41.0 s at 25,000).

Add a test asserting `chunk_size * per_variant <= budget` across S up to
2,000,000 for several `n_format_fields` — the invariant the docstring claims and
that currently does not hold.

**Not changing:** `from_pgen` and `from_svar1` pass no `max_mem` to
`_auto_chunk_size` while `from_vcf_list` does. That asymmetry is coherent — the
first two have a concurrency planner that adapts under `max_mem`, so their chunk
stays fixed-size, while `from_vcf_list` has no planner and adapts the chunk
instead. Leaving it also keeps the refit's regime stable.

### 6. Refit `RamLaw::PGEN`

`_auto_chunk_size` feeds `PlanInputs.chunk_bytes`, which is the `kappa`
regressor, and the reader change moves `per_sample_mb`. Both coefficients are
invalidated by the above, so the refit must run on the fixed code. Ordering is
forced: 1–5 first, then 6.

Re-run `scripts/bench_svar2/sweep_pgen.sbatch` with two design-matrix changes.

**Decorrelate `kappa`.** The current fit's kappa has SE 7.44 and a 95% CI of
[-9.99, +23.68]. `_chunk_size_for(v)` depends only on V, and both ladders use
the same three V values, so `chunk_size ∈ {7812, 15625, 25000}` at both cohort
widths. The matrix `[1, S, S·cs]` is technically identifiable with 2 S-levels
and 3 cs-levels, but `cs` spans 3.2× against S's 8× and the two multiply, so
`S` and `S·cs` stay correlated enough to inflate the SE. Fix by varying
`chunk_size` at *fixed* (S, V): at V=1,000,000, at both cohort widths, add
`chunk_size ∈ {3_125, 12_500, 25_000}` — an 8× `chunk_bytes` range at constant
`S`. Six extra conversion runs; existing corpora, no new generation.

**Every chunk_size must keep at least one chunk per contig.** A corpus's
`variants` is the *total* across all 22 contigs (`vcfixture bulk --records`), so
V=1,000,000 is ~45,454 per contig and the three chunk sizes above give 14.5, 3.6
and 1.8 chunks per contig. Going lower would put several contigs under a single
partial chunk, and `BitGrid3::zeros` reserves the full `chunk_size` up front and
truncates afterwards — so a partial chunk breaks the linearity `kappa` is
supposed to measure. This constraint, not cost, sets the lower end of the
chunk-size axis.

**Cut the extrapolation.** `per_sample_mb` is currently extrapolated 15.6×
beyond the largest measured cohort (32,000) to reach the 500,000 target. Add an
S=128,000 rung at V=250,000 with `chunk_size ∈ {3_125, 7_812}` (3.6 and 1.5
chunks per contig, matching the existing ladder's regime), taking the
extrapolation to **3.9×**. Generation cost ~5.5 h (3.2e10 cells at the measured
~6.15e-7 s/cell), inside the job's 72 h wall.

**Acceptance criteria, fixed before the run:**

- The refit ships only if it *over-predicts* at every measured point the way
  `plan_sharded` evaluates it — the standard the current law met.
- The doc comment records R², n, kappa's CI, and the validity domain, per the
  convention already in `budget.rs`.
- If kappa's CI still spans zero after decorrelation, it stays labelled a
  conservative bound rather than a fitted rate. The margin is deliberate; it
  trades under-utilization and a spurious `PlanError::InsufficientMemory`
  against an OOM, and is not slack to be tuned away.

`src/budget.rs`'s tests assert on the real coefficients specifically so a refit
trips them (see the comment at the `pgen_law` test); those and
`tests/bench/test_model.py` are updated with the new numbers.

## Expected result

At S=128,000, single contig, 64 MiB budgets: reader delta **1094 MB → ~102 MB**
(batch 67 + window 33 + heap ~2), i.e. **8.0 → 0.8 KB per sample per contig**,
and bounded above by the budget constants rather than by `S`.

## Alternative rejected

Budgets alone, no masks — literally #155's proposal. It bounds RAM in ~30 lines,
but at S=128,000 it forces a ~65-variant pack window: 63 flushes per
4096-variant chunk, parallel packing permanently disengaged, and a residual
floor of `MIN_RECORDS * columns * 4` that still scales with `S`. Masks are what
make the budgets non-binding in the normal regime.

## Verification

Bit-identity is the whole safety argument for change (1).

1. **Unit / property.** Extend the existing `pack_row`-versus-reference tests
   (dense and sparse, at nonzero `word_base`) with a proptest over: `columns`
   not a multiple of 64, `s == 0` and `s != 0` rows, missing calls (`-1`),
   multiallelic records, and records with out-of-scope ALTs dropped.
2. **Arithmetic.** Unit-test the two budget functions directly — the bound is a
   claim about bytes and should not need a measurement to check.
3. **End-to-end byte identity.** Convert the same inputs before and after and
   compare output stores byte for byte, across all three dense sources (PGEN,
   VCF, SVAR1) plus `from_vcf_list` as the untouched-path control.
4. **Full suites.** `cargo test --no-default-features --features conversion`
   (with `CARGO_TARGET_DIR` off NFS), `cargo check --no-default-features`, and
   `pixi run test`. `pixi run test` does not rebuild the extension, so
   `maturin develop --release` precedes any Python-level check.
5. **RSS.** The `rss_mark` ladder at s2000/s8000/s32000/s128000, against the
   predicted ~102 MB at S=128,000.
6. **Wall time.** Pinned `--nodelist` sbatch — node speed on this cluster varies
   2.08×, so an unpinned run cannot support a comparison. This is what sets
   `PARALLEL_MIN_CELLS`.

## Out of scope

- Making the reader budgets `max_mem`-derived (see the rationale above).
- Re-enabling PGEN sub-contig sharding (`P = 1`; measured net slower).
- `PGEN_MAX_CONCURRENT`'s wall-time knee, which was measured before
  `processing_threads_for` was wired onto the PGEN path and is flagged in its
  own doc comment.

## Public API

No public name changes: `PACK_WINDOW`, `NORMALIZE_BATCH_RECORDS`,
`PARALLEL_MIN_VARIANTS`, `PresenceMasks`, and `_auto_chunk_size` are all
private. `RamLaw::PGEN`'s *values* change, and `skills/genoray-api/SKILL.md`
does not document them. No SKILL.md update is required — to be re-checked
against the final diff rather than assumed.

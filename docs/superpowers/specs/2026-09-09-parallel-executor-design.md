# A deterministic parallel executor for cohort-width conversion

Status: approved design, not yet implemented.
Issue: [#176](https://github.com/d-laub/genoray/issues/176).
Depends on: the `Tuning` API (`2026-09-08-explicit-tuning-api-design.md`), unmerged.

## Problem

At All of Us width (S = 535,662 diploid samples, 1,071,324 haplotype columns)
`SparseVar2.from_vcf` is bound by the single per-contig executor thread. From two
production conversions on genoray 4.0.1 (`reader_workers=20`, `overshard=40`,
`chunk_size=5000`, n2-highmem-64):

| conversion | elapsed | `exec-<chrom>` CPU | exec busy | each of 20 readers |
|---|---|---|---|---|
| chr12 | 58,486 s | 56,915 s | **97 %** | ~21,100 s (~36 %) |
| chr11 | 6,591 s | 6,120 s | **93 %** | ~1,400 s (~21 %) |

The monitor line confirms the shape: `dense=6 dense_cap=6` (executor's input queue
full), `pending=100` units ≈ 54 GB queued, `cpu_exec=100%`, `cpu_shard=1983%`,
`cpu_cw=0%`. Twenty readers are producing work at ~20 cores that one core cannot
consume.

This is the exact opposite of the measurement behind `754e255` ("drop the
parallel-executor prototype"), which found exec ~7 % busy and readers at 80 % of
all CPU — but at S = 1,000 and S = 16,000. Both measurements are correct; the
balance point moves with cohort width.

Executor cost per variant is also not stable across runs on the same host:
~1.6 ms/variant on an idle box (chr19) versus ~7.7 ms/variant with a co-tenant
(chr12). The executor's per-column pass over a million-column working set is
memory-bandwidth and cache bound, so saturated readers are not merely wasted —
they actively slow the stage they are feeding.

`754e255` removed the prototype for two stated reasons. Reason (a), "the executor
is not the bottleneck", is false at cohort width. Reason (b), "output was
non-deterministic at every `GENORAY_EXEC_WORKERS > 1`", was a property of *that*
design — it parallelised across whole chunks, so several executors pushed into
one long-allele bank concurrently and `merge_bank_offsets` had to reconcile them.
The design below never splits the bank, so (b) cannot recur.

## Approach

Keep **one** `exec-{chrom}` OS thread per contig. Split the work inside
`dense2sparse_vk` along the axis that actually grows with cohort width — the
haplotype column — and leave every per-variant decision serial.

Per chunk, the executor:

1. Runs `route_variants` **serially**. This is O(V) per-variant work: allele
   classification, `choose_representation`, dense table sizing, and the only
   writes to `LongAlleleTableWriter`. V is `chunk_size` (5,000), so this is cheap
   and stays a single-writer path.
2. Fans the per-column emission across a rayon pool of `exec_workers` threads
   over a **fixed** partition of `0..columns`.
3. Concatenates each slice's output in slice order.

The bank is never shared. There is no `merge_bank_offsets`. There is one dense
input queue, one `SparseChunk` per chunk, and one writer — the pipeline's shape
is unchanged outside `dense2sparse_vk`.

### Why not one executor per sample slice

Issue #176 proposes N executor threads, one per fixed sample slice, each
consuming whole `DenseChunk`s. That would give every executor its own bank (the
`754e255` failure mode, or an offset-merge to avoid it), duplicate `route_variants`
N times per chunk, and multiply the dense-queue fan-out. Slicing inside one
executor gets the same near-linear scaling on the per-sample work with none of
that.

## Determinism

Output is byte-identical to the serial path **by construction**, not by test.

| Output | Why it cannot vary |
|---|---|
| Long-allele bank (`nrvk`) | Written only by serial `route_variants`; workers never touch it. |
| Slice boundaries | A pure function of `columns` and `exec_workers`. No work stealing decides them; rayon only decides *when* a slice runs. |
| `call_positions`, `call_keys`, `field_calls`, `sample_lengths` | Each worker fills its own `SparseSubStream`; the executor concatenates in slice order. The concatenation is the serial column loop's byte sequence. |
| Dense `geno_bits` | Slice boundaries are multiples of 64 columns, so each slice's bit range is word-aligned and disjoint. Workers write into `split_at_mut` views of the single preallocated buffer — nothing to merge. |
| `Phase1Output` ledgers | Built from the concatenated streams after the fan-out joins. |

### The 64-column alignment argument

`emit_call`'s dense arm writes bit `hap * n_dense_variants + col`, where `hap` is
the haplotype column. A column slice `[lo, hi)` therefore owns the contiguous bit
range `[lo · n_dense_variants, hi · n_dense_variants)`. If `lo ≡ 0 (mod 64)` then
`lo · n_dense_variants ≡ 0 (mod 64)` for **any** `n_dense_variants` — so the
alignment holds for both dense classes simultaneously, and for every chunk,
without knowing either class's dense count in advance. The final slice's end is
`columns`, which need not be aligned; the buffer is already `div_ceil(8)`-sized
for that tail.

`exec_workers` is capped at `columns / 64` so no slice is empty by construction.

## Per-slice counting sort

The current sort is shared and cohort-width-hostile:

```rust
let mut counts_per_col = vec![0u32; columns + 1];   // 4.3 MB at S = 535k
...
let mut cursor = counts_per_col.clone();            // another 4.3 MB
let mut by_col: Vec<u32> = vec![0u32; total];       // scattered writes
```

Both passes scatter across a 4.3 MB array — larger than L2, and thrashing L3
against 20 concurrently decompressing readers.

`Carriers::cols` is documented **strictly ascending** (`record_source.rs:52`,
enforced by a `debug_assert` in `push`). So each worker can find its own window
of every variant's carrier list with `partition_point` and run a counting sort
sized to *its* columns only:

- per worker: O(V · log C) to locate windows, plus O(carriers/e + columns/e)
- no shared arrays, no cross-worker synchronisation
- working set per worker divided by `e`

The cache effect is expected to be a material part of the win, separate from the
core count. The bench in "Measurement" below separates the two.

## Scope boundaries

**In scope**

- `dense2sparse_vk` (the carrier-driven path) — the production VCF path, and the
  only one that runs at cohort width.
- `plan_sharded` core budgeting, and the `Tuning` field that overrides it.
- The monitor's `cpu_exec` attribution, which reads a `TidRegistry`
  (`monitor.rs`); worker TIDs must register or `cpu_exec` will under-report the
  stage by a factor of `exec_workers`.

**Out of scope, deliberately**

- `dense2sparse_vk_by_scan` stays serial. It is the natively-dense (PGEN) path
  and the differential-test oracle: the existing test drives a carrier-bearing
  chunk through both paths and asserts byte-identical output, so leaving it
  serial turns the parallel path's equivalence into an existing assertion.
- The dense-FORMAT second pass in `route_variants` — O(V_dense × S × F), and
  variant-major in its output layout, so slicing it is a different change. It is
  a no-op when no FORMAT fields are requested, which is the reporting case. Spun
  out as a follow-up issue.
- Reader/executor back-pressure. #174 already landed it upstream; it is unreleased
  (tag 4.0.2 predates the merge). Cutting a release is the fix for issue #176's
  item 1 and needs no code.

## Thread budget

`plan_sharded` currently hardcodes per-contig CPU demand as `1 + reader_workers`,
with a doc comment asserting the executor is "a serial recv loop, pegged at ~100 %
of one core". That comment becomes false. The change:

```
core_bound = usable_cores / (exec_workers + reader_workers)
```

and `processing_threads_for(usable, cc, w)` becomes
`usable - cc * (e + w)`, floored at 1.

`exec_workers` defaults to a **fitted** function of cohort width, not a guessed
constant. Define `r(S)` = (single-core executor cost for one chunk) / (single-core
reader cost for one chunk) at cohort width S. `r` is measured — the executor half
by the microbench below, the reader half by the existing conversion bench — and
the default is `e = ceil(r(S) · reader_workers)`, clamped to `[1, columns/64]`.
This is not circular: `r` depends only on S, `w` is already an input to
`plan_sharded`, and `core_bound` then falls out of `e + w`. Two constraints on
the fit:

1. **`e = 1` at S ≤ 16k.** That regime is what `754e255` measured, and today's
   plan is correct there. A fit that changes the plan at S = 1,000 is wrong.
2. **Never extrapolate past the measured range.** The fit is clamped at its
   largest measured S, per the RamLaw lesson (`fit-bound-laws-as-lp-envelopes`).

`Tuning.exec_workers` overrides the derived value and is honoured, not silently
shrunk — matching the tuning spec's stated contract. The only refusal path stays
the memory budget.

The `pipeline config` banner must print the resolved `exec_workers`. The
downstream cost of *not* doing this is on record: on #174 a user lost a day to a
banner that printed a `reader_workers` the run was not using.

### Pool ownership

The executor gets its **own** `rayon::ThreadPool`, built explicitly and entered
with `pool.install`. It must not use the global pool and must not inherit
`lib.rs`'s `concurrent_chroms`-sized pool: that pool is sized 1 in the common
case, and a nested `par_iter` inside `process_chromosome` silently runs
single-threaded there. An explicit pool also lets the workers be named
(`xw-{chrom}-{i}`) so `top -H` and the TID registry can both find them.

## Measurement

No 535k-sample corpus is needed, and none should be built. A standalone bench
(`src/bin/bench_dense2sparse.rs`) constructs a synthetic `DenseChunk` directly —
`chunk_size` variants, S samples, ploidy 2, carriers drawn at a given alt-call
frequency — and calls `dense2sparse_vk`. This isolates the executor exactly,
runs in seconds, and is the same seam the production path uses.

It reports, per (S, alt-freq, `exec_workers`):

1. wall time and CPU time for the stage,
2. the serial/parallel split (`route_variants` vs. fan-out),
3. a store digest, asserted equal across every `exec_workers` at fixed input.

Requirements on how it is run, from prior burns in this repo:

- Pin `--nodelist` for any cross-configuration comparison; the same job has run
  151.9 s on cn-03 and 73.2 s on cn-04.
- Run it under `sbatch`, not on the login node.
- Confirm a fresh `.so` (`maturin develop --release`) before believing any
  Python-level number; `pixi run test` does not rebuild the extension.
- Export `CARGO_TARGET_DIR` off NFS.

## Testing

1. **Equivalence, unit.** The existing `dense2sparse_vk` ↔ `dense2sparse_vk_by_scan`
   differential test, extended to run the parallel path at `exec_workers` ∈
   {1, 2, 3, 7} against the serial scan oracle. Odd and prime worker counts are
   deliberate: they force unaligned tail slices.
2. **Equivalence, property.** A proptest over (V, S, ploidy, carrier pattern,
   dense/sparse routing mix) asserting the `SparseChunk` is equal at
   `exec_workers = 1` and `exec_workers > 1`.
3. **Equivalence, end-to-end.** Store digest of a converted fixture, asserted
   identical across `exec_workers` and against the pre-change build. This is the
   claim `754e255` could not make.
4. **Alignment invariant.** A test asserting every slice boundary is a multiple
   of 64 and that slices tile `0..columns` exactly, for a spread of
   (columns, exec_workers) including `columns` not divisible by 64 and
   `exec_workers > columns / 64`.
5. **Planner.** `plan_sharded` tests for the `e + w` demand, for `e = 1` at
   S ≤ 16k, and for the clamp at the largest measured S.

## Follow-up issues to open

- **Ledger memory at cohort width.** The executor clones each stream's
  `sample_lengths` into `var_key_ledgers` per chunk, and holds them for the whole
  contig. `sample_lengths.len() == columns` for every stream, so at S = 535k that
  is 2 × 1,071,324 × 4 B ≈ 8.6 MB per chunk; a chr12-sized contig at
  `chunk_size=5000` is ~2,200 chunks ≈ 19 GB retained. The reported run showed
  `rss_mb=79983`. Worth its own investigation; not this change.
- **Dense-FORMAT second pass** — O(V_dense × S × F), serial, variant-major.
- **Release cut.** #174's back-pressure is merged and unreleased; issue #176's
  item 1 resolves on a release, not a code change.

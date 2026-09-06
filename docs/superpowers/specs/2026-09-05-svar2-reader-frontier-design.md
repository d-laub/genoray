# SVAR2 sharded-VCF reader: real worker knob, narrow frontier, bounded backlog

Date: 2026-09-05
Issue: [#169](https://github.com/d-laub/genoray/issues/169)
Scope: `SourceSpec::Vcf` (the sub-contig sharded path used by
`SparseVar2.from_vcf`). `from_vcf_list`, `from_pgen`, and `from_svar1` are out
of scope — none of them shard sub-contig at runtime.

## Problem

All of Us v9 (535,662 samples), 32 vCPU / 251 GB / HDD, genoray 4.0.1,
`from_vcf(threads=30, chunk_size=5000, regions=<MANE exon BED>)`: chr21 (60 GB
bgzf, 10.1 M records, 2,016 chunks) took **4 h 37 m**. The progress curve was
~2–3 chunks/min for 3.5 h, then ~1,470 chunks in the final ~70 min. Process
RSS swung between 40 GB and 91 GB.

Three defects, all confirmed by reading the code:

1. **`reader_workers` is not reachable from the public API.**
   `DEFAULT_READER_WORKERS = 3` (`src/lib.rs:160`) is the only source of the
   per-contig reader count on the VCF path. `threads=30` does not influence
   it. Its doc comment justifies 3 with "the executor is the bottleneck, not
   the reader" — that was the pre-parallel-executor regime; the measured knee
   has since moved to w ≈ 16–32. Meanwhile
   `budget::reader_workers(usable_cores, concurrent)` (`src/budget.rs:88`)
   already derives w from cores, is computed into `ThreadPlan.reader_workers`,
   and is **dead on the VCF path** — `lib.rs:290` passes the constant to
   `plan_sharded` instead. The only escape is the bench-only
   `GENORAY_READER_WORKERS` env hook, whose effect is invisible: the
   `pipeline config` log line prints the planner's value, not the effective
   one.

2. **The reorder frontier is too wide, so the executor is fed in bursts.**
   `shard_exec::ReorderBuffer` hands out global chunk ids in strict
   `(ordinal, local)` order. Only the head unit streams; every other in-flight
   unit buffers into `pending`. Units are planned by **base-pair span**
   (`plan_vcf_shards(.., max_shards = w × OVERSHARD_FACTOR, target_bp =
   chunk_size)`), and `OVERSHARD_FACTOR = 4`, so w=20 gives 80 units over the
   contig — **four waves**. Within a wave the executor's input rate is capped
   at one reader's rate no matter how many readers run; when the head clears,
   the wave releases in a burst. That is exactly the observed drip-then-burst
   curve. Because units are bp-spans rather than record counts, density skew
   across the contig lands directly on the head.

3. **The reorder backlog is unbounded.** `PendingBacklog` has an observing
   gauge but no ceiling. At 5,000 variants × 535k samples a pending chunk is
   hundreds of MB, and up to `w − 1` units' worth accumulate. Note
   `plan_sharded` (`src/budget.rs:463`) *already prices* this term as
   `kappa × (w + pending) × chunk_MB` with `pending = w − 1` — the planner
   models the backlog but nothing enforces it. That is why w=3 was safe and
   w=20 was 91 GB, and why "fewer readers" is currently both the safe choice
   and the slow choice.

Not in scope: the issue's own numbers show the executor drains at ≥20
chunks/min, so a perfectly-fed run of this contig is ~1.7 h. Going below that
requires parallelizing `run_compute_engine`; filed separately.

## Design

### A. Make `reader_workers` a real parameter

- Add `reader_workers: int | None = None` to `SparseVar2.from_vcf` and to the
  `genoray write` CLI. Thread it through `run_conversion_pipeline` as
  `reader_workers: Option<usize>` alongside `max_threads`.
- Delete `DEFAULT_READER_WORKERS`. When the caller passes `None`, derive w
  from the core budget by reviving `budget::reader_workers`.
- **Resolve the circularity.** Today w is a constant, so `plan_sharded` can
  take it as an input and return `concurrent_chroms`. Once w derives from
  cores-per-contig, the two are mutually dependent. Break it by planning in a
  fixed order:

  1. **Reserve the merge tail first.** `processing_threads_for` sizes the
     var_key gather pool and `dense_merge`'s bit-transpose from whatever cores
     the readers leave over. Spending *all* remaining cores on readers would
     floor that pool at 1 and give back the 2.77–2.82× merge-tiling win
     (commit `c49d1d7`). So set
     `reader_pool_cores = usable_cores − ceil(usable_cores / MERGE_RESERVE_DIV)`
     with `MERGE_RESERVE_DIV = 4`, and plan readers only inside that.
  2. **Choose `cc` preferring depth:**
     `cc = min(n_contigs, mem_bound, max(1, reader_pool_cores / (1 + W_TARGET)))`
     with `W_TARGET = 8`. Depth is preferred because each extra concurrent
     contig costs a dedicated executor core and a full `per_contig_mb` of RAM,
     while buying no extra reader cores — the readers are CPU-saturated, so
     total read throughput tracks total reader cores however they are
     partitioned. The old argument for shallow depth ("surplus readers steal
     cores from other contigs' executors") was really an argument about the
     *frontier*, which section B fixes.
  3. **Fill the depth:** `w = max(1, (reader_pool_cores / cc) − 1)` — one core
     for that contig's executor, the rest for its readers.
  4. **Re-check the memory bound** at the chosen `(cc, w)`; if it fails,
     decrement `cc` and repeat from step 3. The loop is bounded by
     `cc ≤ n_contigs`.

  Worked example, the reported machine (32 vCPU → `usable_cores = 31`, 22
  contigs): `reader_pool_cores = 23`; `cc = min(22, mem, 23/9=2) = 2`;
  `w = 23/2 − 1 = 10`; merge pool = 8. Today the same machine gives
  `cc = min(22, mem, 31/4=7)` and `w = 3`.

  `MERGE_RESERVE_DIV` and `W_TARGET` are starting values to be confirmed by
  the benchmark in section D, not fitted constants. Both must be named
  constants in `budget.rs` with the measurement that set them recorded in the
  doc comment.
- An explicit `reader_workers=` replaces step 3's derived `w` but still runs
  steps 1, 2 and 4, and raises `PlanError::InsufficientMemory` rather than
  silently shrinking the request.
- **Effective-config logging (issue proposal 4).** Move the
  `GENORAY_READER_WORKERS` / `GENORAY_OVERSHARD` / `GENORAY_SHARD_HTSLIB`
  resolution out of `process_chromosome` (`src/orchestrator.rs:518-525`) and
  into the planner in `lib.rs`, so the `pipeline config` line prints
  post-override values. Keep the env vars as overrides; they stay documented
  as bench hooks.

### B. Size shards by records, not base pairs

`plan_vcf_shards`'s `max_shards` becomes a function of the contig's record
count rather than of the worker count:

```
target_units = ceil(contig_records / (UNITS_TARGET_CHUNKS * chunk_size))
max_shards   = clamp(target_units, w, MAX_UNITS_PER_CONTIG)
```

with `UNITS_TARGET_CHUNKS = 4` (a unit is about four chunks of records) and
`MAX_UNITS_PER_CONTIG = 4096`, a guard against pathological inputs (each unit
is an independent indexed fetch, so the unit count is also a seek count).

- `contig_records` comes from `contig_cost`, which `lib.rs:255-281` already
  computes and already distinguishes by `costs.exact_counts`. When
  `exact_counts` is false the values are base-pair contig lengths, a different
  unit entirely — in that case fall back to today's `w × OVERSHARD_FACTOR`
  behaviour rather than misusing them.
- Effect on chr21 at w=20: 80 units → ~500 units, 4 waves → ~25 waves. Head
  advances at the aggregate reader rate, so the executor is fed continuously
  and the burst/idle pattern disappears.
- Cost: each unit is an independent indexed fetch whose window is padded by
  `normalize::L_MAX = 1000` bp on each side. At 4 × 5,000 records per unit on
  chr21 density (~220 records/kbp) a unit spans ~91 kbp, so padding is ~2% of
  decoded records. `UNITS_TARGET_CHUNKS` is the knob that trades this padding
  overhead against frontier width; 4 is a starting value to be confirmed by
  the benchmark in section D, not a fitted constant.
- `OVERSHARD_FACTOR` survives only as the `exact_counts == false` fallback.

### C. Bound the pending backlog, head-exempt

Add a byte ceiling to `shard_exec::run`'s collector, enforced at the producers:

- The collector publishes `head: Arc<AtomicUsize>` (the `ReorderBuffer`'s head
  ordinal) and `pending_bytes: Arc<AtomicU64>`, plus a `Condvar` it notifies
  whenever the head advances or bytes drop.
- Before `tx_res.send(Msg::Chunk { .. })`, a worker whose `unit.ordinal >
  head` waits while `pending_bytes > budget`. A worker whose
  `unit.ordinal == head` **never** waits.
- Also re-check `cancel` inside the wait so the error path still tears down.

**Why this cannot deadlock.** The work queue is a FIFO MPMC channel seeded in
ordinal order, so units are dequeued in ascending ordinal order. Let `h` be
the head. If unit `h` is still queued, no worker can hold any `j > h` (it
would have had to be dequeued before `h`), so every worker holds an ordinal
`< h` — but those are all complete by definition of the head, contradiction.
Therefore `h` is always either already done, or in flight at a worker that is
exempt from the wait. That worker runs to its `Done`, the head advances, the
condvar wakes the next holder. Progress is guaranteed.

The other two blocking edges are ordinary backpressure, not deadlock: a worker
blocked on the bounded `tx_res` is drained by the collector, and a collector
blocked on the bounded `tx_dense` is drained by the executor.

**Budget.** `pending_budget_bytes = PENDING_BUDGET_CHUNKS × chunk_bytes`, with
`PENDING_BUDGET_CHUNKS = 8` and a hard floor of 2 (a one-chunk budget would
serialize the frontier back to the head). It is deliberately **independent of
`max_mem`**: deriving it from `max_mem` would reintroduce a circularity, since
`plan_sharded` consumes the budget to pick `cc` and `max_mem` is what bounds
`cc`. Instead the budget is a fixed multiple of the chunk size, and `max_mem`
constrains `cc` through the amended law below. `chunk_bytes` here is the same
`resident_chunk_size`-narrowed value `lib.rs:255-281` already computes, not
the nominal `chunk_size × per_variant_bytes` — `BitGrid3::zeros` is a calloc,
so nominal chunk bytes are address space, not RSS.

**Planner change.** With the backlog enforced, `plan_sharded`'s per-contig
term changes from

```
per_contig_mb = per_contig_mb + kappa * (w + (w-1)) * chunk_MB
```

to

```
per_contig_mb = per_contig_mb + kappa * w * chunk_MB + pending_budget_mb
```

This is deliberately **not** a refit of `RamLaw::VCF`. The existing
coefficients stay as they are; the `(w − 1)` term is replaced by an explicit
additive budget the code now enforces, which is strictly more conservative
per unit of `w` than a fitted term would be. Removing the quadratic-in-`w`
growth is what makes a large `w` affordable at all. Any subsequent refit is
separate work — see the LP-envelope discipline in
`2026-08-11-ramlaw-vcf-envelope-design.md`.

### D. Validation

- **Correctness gate: byte-identical stores.** Every change here is a
  scheduling/ordering change; the emitted store must be unchanged. Compare
  against `main` on the existing conversion fixtures at several
  `(w, chunk_size, UNITS_TARGET_CHUNKS)` settings, including `w = 1` and a
  region-filtered run.
- **Unit tests in `shard_exec`:** head-exempt backpressure makes progress when
  every non-head worker is parked at the budget; `pending_bytes` never exceeds
  the budget plus one in-flight chunk per worker; the existing reorder-order
  tests still pass.
- **Unit tests in `budget`/`shard`:** the ordered `(cc, w)` plan is stable and
  monotone in cores; an explicit `reader_workers` that does not fit raises
  `InsufficientMemory`; record-sized `max_shards` falls back correctly when
  `exact_counts` is false.
- **Performance repro.** `scripts/bench_svar2/scale_corpus.py --samples` can
  synthesize a wide/shallow cohort — ~500k samples × ~200k variants (≈1.2 GB,
  1/50 of the reported chr21) — which reproduces the pathology's shape
  (per-chunk bytes are what matters, and they scale with samples, not
  variants). Run it pinned with `--nodelist` on one sbatch allocation.
  Report: wall time, the chunks-committed curve (flat, not drip-then-burst),
  `pending_hw`, peak RSS, and **the merge-tail wall separately from the
  pipeline wall** (the contig span log covers both — see
  `svar2-contig-span-includes-merge`), for `main` vs the branch at
  w ∈ {3, 8, 20}. The tail number is what confirms `MERGE_RESERVE_DIV`.
- **Constants set by measurement, not by this document.** `W_TARGET`,
  `MERGE_RESERVE_DIV`, `UNITS_TARGET_CHUNKS`, and `PENDING_BUDGET_CHUNKS` all
  carry starting values above so the implementation is unambiguous. Each is a
  named constant whose doc comment must record the sweep that set it. Sweep
  them one at a time on a single pinned allocation; cross-scale timing claims
  from unpinned or resumed runs are not admissible (see
  `svar2-scale-ladder-rows-contaminated`).
- **Field validation.** Hand the reporter a candidate build for the remaining
  19 chromosomes; they have the monitor and py-spy native profiles already.

## Public API impact

`skills/genoray-api/SKILL.md` must gain `reader_workers` on `from_vcf` and on
the `genoray write` CLI, per the repo's public-name rule.

## Deferred

Parallelizing `run_compute_engine` — filed as
[#170](https://github.com/d-laub/genoray/issues/170). Even a perfectly fed
pipeline leaves this contig executor-bound at ~1.7 h; #170 depends on this
work landing first, since a parallel executor cannot be measured against a
pipeline that starves it.

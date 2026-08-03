# SVAR2 tuned load balancing for conversion

**Date:** 2026-08-03
**Status:** design approved, implementation plan pending

Follows `2026-07-28-svar2-scale-bench-harness-design.md`, which produced the
measurements this spec acts on. That harness posed three hypotheses about what a
reader-budget autotuner should key on and returned **H3**: worker *count* is the
wrong invariant, in-flight *bytes* is the right one.

## Problem

`budget::plan_thread_budget` decides concurrency from two integers — core count
and contig count — and never revisits the decision. It charges every contig
`MIN_THREADS_PER_CHROM = 6` cores: four pipeline threads (reader dispatcher,
executor, chunk writer, long-allele writer) plus two HTSlib decode threads.

Both halves of that charge are wrong on the sharded VCF path.

The HTSlib pool does not exist there — `SHARDED_VCF_HTSLIB_THREADS_PER_READER`
is 0 by construction, because each shard reader decompresses inline in its own
worker thread. And of the four pipeline threads only the executor is CPU-bound;
the dispatcher and both writers spend nearly all their time blocked. Measured:
the 22-contig `cc=1, w=12` row runs 16 threads on **2.02 cores**.

So the planner reserves ~6 cores for something that consumes 0.4–1.0, and
`plan_thread_budget(48, 22)` returns `cc=7, w=2` where the machine could run
every contig at once.

### What the split is worth

The contig sweep (S=4,000, 48-core allocation, `carter-cn-04`) holds **total
reader threads fixed at 12** and varies only how they are split across contigs:

| contigs | split | wall | cores used (of 48) |
|---|---|---|---|
| 22 | `cc=1, w=12` | 35.11 s | 2.02 |
| 22 | `cc=4, w=3` | **11.74 s** | 5.24 |
| 8 | `cc=1, w=12` | 16.13 s | 3.61 |
| 8 | `cc=4, w=3` | **8.22 s** | 6.67 |
| 2 | `cc=1, w=12` | 13.80 s | 3.80 |
| 2 | `cc=2, w=6` | **9.09 s** | 5.97 |

The work is identical — both 22-contig rows sum to `phase1 = 33.0 s` of
per-contig spans. Only the concurrency differs, and it is worth **2.99×**. The
*best* of these configurations still leaves 43 of 48 cores idle.

### Why more readers cannot substitute

`executor::run_compute_engine` is a strictly serial `while let Ok(chunk) =
rx_dense.recv()` loop over `dense2sparse_vk`. One executor per contig, and it is
the per-contig throughput unit. The sampler shows two regimes:

- **Large contig** (`s4000_c1`, `s4000_c2`): executor pegged at 99–101%, dense
  channel full at its cap of 6 for the entire run, `pending_hw` 9–11. The
  readers are blocked on backpressure. This is why `w=12` buys nothing over
  `w=3` — the readers were never the constraint.
- **Many contigs** (`s4000_c8`, `s4000_c22`): executors at 36–43%, channel
  occupancy 0. *Nothing* is saturated. Wall is `Σ spans / cc`, so contig
  concurrency is the only lever that exists.

Neither regime is improved by raising `w`. Both are improved by raising `cc`.

## Goal

Make `cc` as large as cores and memory allow, order contigs so the tail is
short, and derive `w` from a measured rate rather than a hardcoded knee.

**Non-goals.** This spec does not give a contig more than one executor.
Sub-contig executors — promoting the reader's existing POS-owned shards to
independent pipelines with concatenated output — is deliberately out of scope;
see "What this does not buy".

## Design

Four units with clean seams.

### 1. `budget.rs` — plan under explicit constraints

`plan_thread_budget` is replaced by a planner over named inputs
(`usable_cores`, `n_contigs`, `n_samples`, `chunk_bytes`, `max_mem`,
`w`) returning the existing `ThreadPlan`. Per-contig core demand becomes
`1 + w` (one executor, `w` readers) rather than `PIPELINE_THREADS_PER_CHROM +
htslib_threads`. Two constraints bind:

```
cores:  cc ≤ usable_cores / (1 + w)
memory: base + per_sample·S + κ·cc·(w + pending)·chunk_bytes ≤ max_mem
cc = min(n_contigs, core_bound, mem_bound)
```

The memory coefficients are the scale-bench RAM law
(`peak_rss_mb ~ 932 + 0.01115·samples + 1.371·(w+pending)·chunk_bytes`,
R²=0.9040, n=44). They ship as named constants carrying the fit and the date
that produced them, so a later refit is a visible edit and not a silent drift.
`pending` uses the reorder buffer's structural floor `w - 1` (see the harness
README: `w-1` units ahead of the head keep everything they produce buffered even
with perfectly balanced readers).

This module stays pure arithmetic and no I/O, which is what makes the existing
unit-test style — assert a whole `ThreadPlan` for a given `(cores, contigs)` —
carry over unchanged.

The planner still computes `htslib_threads` by the current arithmetic and still
returns it, because the *monolithic* reader path is unaffected by this spec and
consumes it unchanged. The three thread-count constants
(`PIPELINE_THREADS_PER_CHROM`, `MIN_HTSLIB_THREADS`, `MAX_HTSLIB_THREADS`)
therefore survive, but their docs must say which path they bind — the current
comments read as if they bound both, which is how the sharded path came to be
charged for a decode pool it never allocates.

### 2. `contig_cost.rs` — per-contig work estimates

Only *ratios* matter; the estimates order contigs and nothing else. Fallback
chain, most to least precise, as implemented:

1. `hts_idx_get_stat` per-contig mapped-record counts from the `.tbi`/`.csi`.
2. Contig length from the header.

The middle tier from the original three-tier design — the linear index's
compressed byte extent per contig — was dropped before implementation:
reaching it means walking CSI internals through far more `unsafe` than either
surviving tier, for an estimator whose entire output is a sort key.

Tier 1's viability was an open question going in: `hts_idx_get_stat` is
documented for BAM, and whether CSI/TBI indexes over VCF populate the
mapped-record count was unconfirmed. A test built a 3-contig VCF with
deliberately unequal record counts (5/40/15) and asserted `estimate_contig_costs`
ranked them in true-count order. It passed on the first implementation —
`hts_idx_get_stat` does return real per-contig mapped counts for CSI-over-VCF —
so tier 1 shipped as designed rather than being cut down to the header-length
tier alone.

PGEN takes exact per-contig variant counts from the `.gvi` index it already
builds. Every source is metadata already on disk; none reads variant data.

A contig with no estimate sorts as if it were the largest. Guessing high on an
unknown contig costs a slightly worse order; guessing low risks starting the
longest job last, which is the exact failure the ordering exists to prevent.

### 3. `tune.rs` — the optional probe

Sizes `w` from measurement instead of a fitted knee. Two numbers, taken on a
bounded prefix of the largest contig:

- `t_read` — one shard worker's seconds per chunk (inflate, parse, densify)
- `t_exec` — `dense2sparse_vk` seconds per chunk

To keep the executor fed, `w` readers must supply at least as fast as one
executor drains: `w / t_read ≥ 1 / t_exec`, hence

```
w* = clamp(ceil(t_read / t_exec), 1, W_MAX)
```

`W_MAX` is a safety clamp, not a tuning parameter: it bounds the damage from a
probe that measured a pathological prefix (an all-reference stretch reads far
faster than it converts). The harness never observed a knee above 7, so `W_MAX`
is set well above that — 16 — and a probe returning the clamp is logged, since
hitting it means the probe, not the workload, is the thing to look at.

This is the knee, derived. It explains the harness's observation that `w*`
moves with cohort size but not core count: both rates scale with `S`, and it is
their *ratio* that sets the knee.

The probe reuses existing instrumentation (`shard_unit_secs`, and the
`exec: dense2sparse enter/exit chunk` trace points). It converts two chunks,
discards the output, and is skipped entirely when `tune=False`.

Why it is worth building even though the laws are already fitted: those laws
were fitted on synthetic corpora on a single machine, and node speed on this
cluster varies by **2.08×** (see `carter-node-speed-varies-2x`). `t_read` and
`t_exec` also move with compression ratio, field count, and ploidy — none of
which the fitted knee sees. The probe measures the ratio on the actual input,
on the actual machine.

### 4. `lib.rs` — wiring

```
estimate per-contig costs  →  [optional probe → w]  →  plan(cc)
    →  sort contigs descending by cost  →  rayon dispatch
```

Rayon's work-stealing does the dynamic balancing; descending order (LPT) is the
entire dispatch-side change. Contigs are dispatched with `with_min_len(1)` so a
single contig is stealable.

## API

```python
SparseVar2.from_vcf(..., max_mem: int | str | None = None, tune: bool = False)
```

`max_mem` parses through the existing helper in `_utils.py`. Both are public, so
`skills/genoray-api/SKILL.md` updates in the same PR.

### The `max_mem=None` default is a behavior change

`None` means *detected budget*, not unbounded. Unbounded preserves exactly the
biobank-scale OOM exposure that H3 was raised about, and the planner's whole
purpose is to have a byte budget to plan against.

Detection **must read the cgroup memory limit**, not `/proc/meminfo`. Under
Slurm those differ, and `/proc/meminfo` reports the node's memory rather than
the job's — handing the planner a budget it does not have, on precisely the
allocations where the planner matters most. Fall back to `/proc/meminfo` only
when no cgroup limit is readable.

Apply `MEM_BUDGET_FRACTION = 0.8` to whichever source is used, never the whole
limit. The RAM law predicts peak RSS with R²=0.9040 and its own hold-out error
was 3%; the headroom covers that residual plus everything the law does not model
(the Python interpreter, glibc arena fragmentation, the merge tail). Planning to
100% of a cgroup limit means the first prediction error is an OOM kill.

This changes default concurrency for existing callers. It belongs in the commit
message and the changelog-bearing commit body, not buried in a docstring.

## Correctness invariant

**Scheduling must not change output bytes.** `cc`, `w`, and contig order all
move, and each is an opportunity to perturb chunk ordinals, per-chunk ledgers,
or long-allele bank offsets.

The scale-bench harness already computes a per-run `digest`. The gate is: the
digest is identical across a permutation of `(cc, w, contig order)` on the same
corpus. This is the property the entire change rides on — if it does not hold,
nothing else in this spec matters.

## Testing

| unit | test |
|---|---|
| `budget` | Pure unit tests per binding constraint: core-bound, memory-bound, `n_contigs`-bound, and the degenerate 1-core / 1-contig cases. Assert whole `ThreadPlan` values, matching the existing style. |
| `budget` | The memory constraint actually binds: a large `S` × large `chunk_bytes` case must return a smaller `cc` than the core bound alone would. |
| `contig_cost` | Golden test against a fixture VCF: estimates ordered the same as true per-contig record counts. Each fallback tier exercised. |
| `contig_cost` | A contig missing from the index sorts first. |
| `tune` | `w*` lands in a sane range on a fixture and rises when `t_read/t_exec` rises. Bounds, not a pinned number — the probe measures a machine. |
| e2e | Digest identical across `(cc, w, order)` permutations. |
| e2e | `max_mem` too small to fit even `cc=1` fails with a clear error rather than planning `cc=0`. |
| bench | A contig-concurrency point in the scale harness, confirming the win against the 5.24-of-48-cores baseline. Must pin `--nodelist`. |

Rust tests run `--no-default-features --features conversion` with
`CARGO_TARGET_DIR` off NFS. Any Python-level timing verification needs
`maturin develop --release` first — `pixi run test` does not rebuild the
extension.

## Risks

- **Raising `cc` raises peak RSS.** Measured at S=4,000: 1193 MB at `cc=1`
  versus 1494 MB at `cc=4`, roughly 75 MB per additional concurrent contig, and
  the per-contig footprint scales with sample count. This is exactly why the
  memory constraint is in the planner rather than left to the caller. It also
  means the RAM law is now load-bearing in production, not just in the bench —
  a bad refit becomes an OOM.
- **The RAM law was fitted at F=0.** The F=3 hold-out is out of domain by 67%
  on phase 1. Until a FORMAT-field cost law exists, the planner should treat
  configured fields as a reason to be conservative with `cc`.
- **The probe adds latency to every tuned run.** Two chunks is small against a
  biobank conversion and large against a 200-variant test file. `tune=False`
  must stay the default.
- **LPT is bounded by the largest contig.** See below.

## What this does not buy

At `cc = n_contigs` the makespan floor is the largest contig's serial-executor
span. For a whole-genome VCF, chr1 is roughly 8% of the assembly against a
perfectly-balanced 4.5%, so the floor is on the order of 1.8× optimal, and no
amount of ordering or worker allocation moves it.

Removing it requires giving one contig more than one executor: promoting the
reader's existing POS-owned shards (already planned, already oversharded,
already reordered) into independent pipelines whose outputs are concatenated in
region order. That touches chunk ordinals, per-chunk ledgers, and long-allele
bank offset rebasing. It is a separate spec.

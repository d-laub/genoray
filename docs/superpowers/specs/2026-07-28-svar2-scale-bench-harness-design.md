# SVAR2 conversion scale-bench harness

**Date:** 2026-07-28
**Status:** design approved, implementation plan pending

## Problem

`SparseVar2.from_vcf`'s sub-contig sharded reader budget is tuned by a formula
(`budget.rs::reader_workers`) that was never validated against measurement.
Benchmarking PR #140 (see `scripts/bench_sharded_vcf/`, PR #141) showed the
formula overshoots single-contig by 4-11x and *regresses* multi-contig 1.5x,
because the only parallel stage is the shard readers — `executor::run_compute_engine`
is a strictly serial `recv()` loop over `dense2sparse_vk`.

Those measurements top out at 4,000 samples. The motivating workload is biobank
scale: ~500,000 samples and billions of variants across the primary assembly. We
cannot tune for a regime we have never measured, and we cannot measure that
regime directly — a 500,000 x 10^9 VCF is ~10^15 genotype cells.

This spec defines a harness that makes the biobank regime *predictable* from
measurements we can actually afford, and that guards the small-scale behaviour
against regression.

## Goal

Produce evidence that discriminates three mutually exclusive hypotheses about
what a reader-budget autotuner should key on. Each implies a different fix:

- **H1 — static cap.** `w* = c_read(S)/c_exec(S)` is effectively constant in
  cohort size `S`. The observed 250→3, 1000→5, 4000→7 is noise around ~5.
  *Implication:* no autotuner. Clamp the planner at a constant and fix the
  multi-contig divisor. A one-line change.
- **H2 — formula in S.** `w*` genuinely trends with cohort size.
  *Implication:* an autotune formula keyed on `n_samples`, with `n_contigs`
  entering only through executor parallelism.
- **H3 — closed loop.** Worker *count* is the wrong invariant; in-flight
  *bytes* is the right one, and readers should come from a global work-stealing
  pool rather than a per-contig division of `usable_cores / concurrent_chroms`.
  *Implication:* autotuning collapses into the `max_mem` budget callers already
  pass, overshoot becomes free, and the multi-contig regression disappears.

The harness is designed around the measurements that separate these, not around
a prior. For the record the prior is H3: the wall floor was ~10.2 s at 12, 16,
32 and 48 cores alike, which is a serial-executor ceiling, and the 48-core /
8-contig regression is a partitioning artifact rather than a tuning error.

**Non-goal:** this spec does not implement the autotuner. It produces the data
and a recommended policy; the implementation is a follow-up plan chosen by the
result.

## Measurement strategy

Three factorizations make biobank sample counts affordable.

### Variants factor out

`phase1_wall = fill_drain + V/cs * t_chunk(S, C, w, cs)`

Wall time is linear in variant count at fixed chunk size. The harness verifies
this once at small `S` over `V ∈ {25k, 50k, 100k, 200k}` and reports R² and the
intercept. Every subsequent extrapolation to 10^9 variants rides on that fitted
line. **Billions of variants are extrapolated, never generated.** This is the
only reason 500,000 samples is reachable.

If V-linearity fails (R² < 0.98), the whole extrapolation is invalid and the
harness must say so rather than report a number.

### The knee is predicted, then validated

`monitor` already accounts `cpu_shard` and `cpu_exec` separately, so a **single
run at `w=1`** yields per-chunk read cost `c_read` and per-chunk executor cost
`c_exec`, and therefore the predicted knee:

`w*(S) = ceil(c_read(S) / c_exec(S))`

This replaces an O(|w|) sweep at every scale point with one run. The prediction
is then validated against a real `w`-sweep at **three** points (smallest,
mid, largest reachable `S`); the harness reports predicted vs observed knee, and
a disagreement greater than ±1 invalidates the predictive shortcut for the
remaining points.

### RAM is modelled, not merely observed

Peak RSS has two terms and only one is currently visible:

- **in-flight:** `w * chunk_bytes`
- **reorder skew:** `pending_highwater * chunk_bytes` — `shard_exec.rs`'s
  `pending: HashMap<(usize, usize), DenseChunk>` is **unbounded**. A fast shard
  accumulates chunks while the reorder head waits on a slow one. This term is
  invisible today and is the load-balancing hazard H3 predicts.

where `chunk_bytes = cs * (S * ploidy / 8 + 4 * F * S)`, matching
`_auto_chunk_size`'s own cost model (packed presence grid plus staged FORMAT).

The arithmetic that motivates the exercise: `from_vcf` hardcodes
`chunk_size = 25_000` — it is the only `from_*` method that does not call
`_auto_chunk_size`. At `S = 500_000, F = 0` the packed grid alone is
**3.1 GB per chunk** against the ~2,000 variants `_auto_chunk_size` would pick.
That is an OOM at `w = 1`, before any worker multiplication. The harness must
confirm this at a scale it can actually run and bound the extrapolation.

## Corpus sizing

Two constraints bind, and they pull against each other:

1. **Generation cost is linear in cells.** Measured: ~1.2M cells/s per process,
   0.30 compressed bytes/cell.
2. **Steady state needs enough chunks.** Fill and drain must be a small fraction
   of the run, so each point needs >= 32 chunks.

The sizing rule resolves both. Fix a **cell budget per point**, then derive:

```
V  = cells_budget / S
cs = min(25_000, max(64, V // 32))     # >= 32 chunks, production clamp respected
```

Default `cells_budget = 1.4e9` — about 20 minutes of single-process generation,
~420 MB compressed. Chunk size therefore becomes an explicit, reported quantity
at every point rather than an accident, which is exactly what the RAM law needs.

The upper clamp matters: `_auto_chunk_size` never returns more than 25,000, so
measuring small cohorts at a larger chunk size would characterize a regime
production cannot reach. Small-`S` points consequently get far more than 32
chunks (at `S = 250`: 224), which is harmless — the floor is what steady state
needs, not the count.

Scale points: `S ∈ {250, 1_000, 4_000, 16_000, 64_000, 250_000, 500_000}`,
single contig, `F = 0`. Contig axis: `C ∈ {1, 2, 8, 22}` at `S = 4_000`.

**Hold-out validation point** (the one approved extra):
`S = 100_000, F = 3 (DP/GQ/AD), V = 28_000` — 2.8e9 cells. It sits off the
fitted grid on three axes at once: `S` falls between the 64,000 and 250,000
points, `F = 3` is never fitted, and the cell count is twice any fitted point's.
The harness predicts its wall time and peak RSS from the laws *before* running
it, then reports the error. A prediction error above 25% is a failure of the
model and must be reported as such.

Its size is set by generation cost, not by statistical preference. FORMAT fields
inflate the per-sample text roughly 4x (`0|1` versus `0|1:30:99:10,20`), so this
single point costs about 3 hours to generate — already the most expensive item
in the plan.

**Weakest link, stated up front:** the V-law is fitted over an 8x range at small
`S` and confirmed at only 2x on the hold-out, yet the headline extrapolation
stretches it to 10^9 variants. That is the least-supported step in the chain.
`model.py` must report the V-law's R², its fitted intercept, and the
extrapolation factor alongside every projected number, so no consumer reads the
biobank projection as better-evidenced than it is.

Generation is parallelized by formatting record blocks in a process pool and
streaming them **in order** (`Pool.imap`) into a single `bgzip -@ N` stdin.
Parallel formatting, one compression pass, one valid BGZF stream, no temp files
and bounded memory.

## Components

Existing `scripts/bench_sharded_vcf/` is generalized and renamed to
`scripts/bench_svar2/` — it now characterizes the conversion pipeline, not only
sharding. PR #141's README moves with it.

| unit | responsibility | interface |
|---|---|---|
| `scale_corpus.py` | deterministic seeded corpus generation; parallel block formatting; `F` FORMAT fields, MAF spectrum, missingness | `(S, V, C, F, seed) -> .vcf.gz + manifest.json` |
| `probe.py` | exactly one instrumented conversion run | `(corpus, config) -> record` |
| `sweep.py` | executes a plan of points; resumable | `plan.json -> results.ndjson` |
| `model.py` | fits the three laws; extrapolates to a target with residual-derived bounds; picks H1/H2/H3 | `results.ndjson -> laws + verdict` |
| `regression.py` | fast tier (~2 min, tiny corpora) against committed baselines | `pixi run bench-regression` |

Boundaries are drawn for independent testability: `model.py` is pure and unit
tested against synthetic data with known laws; `probe.py` is the only unit that
shells out; `sweep.py` carries no domain knowledge, only execution and
resumption.

Corpora are **regenerated from seeds, never committed** — a 500,000-sample point
is ~420 MB and its seed is 8 bytes. `manifest.json` records
`(S, V, C, F, cells, bytes, seed, tool_versions)` so no consumer re-derives
shape from a filename.

`probe.py` emits the whole-store SHA256 digest on every run. The byte-identity
oracle that covered the PR #140 sweeps covers the scale sweeps unchanged.

### Rust instrumentation

Trace-level only; no behaviour change on any default path.

1. **Monitor gauge for the collector's `pending` map** — length and bytes, with
   a high-water mark. Without this, H3's reorder-skew term cannot be measured
   at all.
2. **Per-shard unit completion times** at trace level, to quantify skew
   directly rather than inferring it.
3. **`GENORAY_CONCURRENT_CHROMS` bench hook** overriding
   `ThreadPlan::concurrent_chroms`. Required for the load-balancing
   counterfactual: hold *total* reader workers constant and vary how they are
   partitioned across contigs. Without it, `GENORAY_READER_WORKERS` alone
   cannot separate "too few readers" from "readers on the wrong contig".

The existing BENCH-ONLY env hooks (`GENORAY_READER_WORKERS`,
`GENORAY_SHARD_HTSLIB`, `GENORAY_OVERSHARD`) are retained unchanged.

## The three laws and the verdict

`model.py` fits:

1. **V-linearity.** `phase1_wall ~ a + b*V`; reports R² and intercept.
2. **Cost laws.** `c_read(S) = alpha_r * S^beta_r`, `c_exec(S) = alpha_e * S^beta_e`,
   fitted by least squares on logs, with 95% CIs on the exponents.
   `w*(S) = ceil(c_read/c_exec)`.
3. **RAM law.** `peak_rss = base + kappa * (w + pending_highwater) * chunk_bytes`;
   `kappa` is the fitted overhead multiple over the analytic chunk size (prior
   data suggests ~3).

It then reports a verdict against explicit, falsifiable criteria:

- **H1** if observed `w*(S)` varies by less than ±1 across the full `S` range.
- **H2** if the 95% CI of `beta_r - beta_e` excludes zero, i.e. the cost ratio
  genuinely trends with cohort size.
- **H3** if either (a) `pending_highwater >= w/2` at any point, so bytes rather
  than worker count set peak RSS; or (b) the constant-total-workers
  counterfactual shows the multi-contig regression is a partitioning artifact —
  same total readers, different split across contigs, wall times differing by
  more than 15%.

H1 and H2 are mutually exclusive by construction. H3 is independent of both: if
H3 holds it supersedes them, because a byte-bounded global pool needs no `w`
prediction. If none of the three is supported, the harness reports "no verdict"
rather than defaulting to one.

Finally it extrapolates to the target regime — `S = 500_000`, `V = 10^9`,
`C = 24`, `F = 3` — and prints predicted wall and predicted peak RSS under three
policies: current `from_vcf` defaults, `_auto_chunk_size` sizing, and the
recommended policy. This is the deliverable that answers "does the current code
survive AoU scale", and the expectation from the arithmetic above is that the
current default does not.

## Error handling

Everything long-running is checkpointed; a full sweep is an overnight Slurm job
on a shared, preemptible cluster.

- **Corpus generation** checkpoints per block. Record counts are asserted
  against the manifest after indexing, so a truncated corpus cannot silently
  produce fast, bogus timings.
- **`sweep.py`** appends NDJSON and skips already-recorded points, so a killed
  or preempted job resumes instead of restarting.
- **Digest mismatch is a hard failure**, never a warning.
- **OOM is recorded as a datum, not a crash.** Each point runs under an explicit
  RSS ceiling; exceeding it records `oom_at_rss_mb` and the sweep continues.
  Demonstrating that `chunk_size = 25_000` OOMs at biobank scale is a
  deliverable, so the harness must survive producing that result.
- Slurm hands out **non-contiguous** CPU ids. Pinning uses the job's real
  allocated ids via `os.sched_getaffinity`, never `0-(N-1)`.

## Testing

- Unit tests for `model.py` against synthetic data with known laws, including
  the degenerate cases: perfectly flat `w*` (H1), a planted trend (H2), a
  planted pending-skew (H3), and mutually inconsistent data (no verdict).
- A tiny end-to-end smoke test (`S = 50`, `V = 500`) in the normal pytest suite,
  marked `bench`, so the harness cannot rot silently.
- The whole-store digest oracle is the correctness test for every swept point.
- Rust instrumentation is covered by the existing suite;
  `cargo test --no-default-features --features conversion` must stay green.

## Out of scope

- The autotuner itself. This spec produces evidence and a recommendation.
- Real-data corpora — synthetic only, with a realism knob (MAF spectrum,
  missingness, FORMAT fields).
- Cross-machine portability of the fitted constants. The laws are characterized
  on one node class; only the *shape* of the laws is claimed to transfer.

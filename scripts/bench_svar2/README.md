# SVAR2 conversion bench harness

Characterizes `SparseVar2.from_vcf`'s conversion pipeline from small cohorts to
biobank scale, and decides what a reader-budget autotuner should key on.

Design: `docs/superpowers/specs/2026-07-28-svar2-scale-bench-harness-design.md`

## Layout

| file | purpose |
|---|---|
| `records.py` | shared frozen record schema + NDJSON codecs |
| `scale_corpus.py` | deterministic seeded corpus generation |
| `probe.py` | one instrumented conversion run |
| `sweep.py` | resumable plan execution + digest oracle |
| `model.py` | law fitting, hypothesis verdict, extrapolation |
| `regression.py` | fast tier against committed baselines |
| `plans/build_plans.py` | generates the sweep plans from the spec's scale points |
| `sweep_scale.sbatch` | the overnight cluster job |
| `regression_record.sbatch` | re-records the fast-tier baselines on a dedicated allocation |
| `legacy_pr140/` | the original PR #140 sharded-reader harness and its findings |

## Running

```bash
pixi run bench-regression            # ~1 min on a dedicated 8-CPU allocation
sbatch scripts/bench_svar2/sweep_scale.sbatch        # overnight, full scale sweep
sbatch scripts/bench_svar2/regression_record.sbatch  # re-record the baselines
```

Re-record baselines through `regression_record.sbatch`, not by running
`bench-regression-record` on a login node. The tier's corpus is ~140 KB, so its
wall time is mostly process startup: the same three points recorded 68/54/119 s
on a busy login node and 5.3/5.3/5.2 s on a dedicated allocation. Only
`maxrss_mb` gates; `wall_s` is printed as a trend signal and is deliberately not
a hard gate (see the comment on `HARD_METRICS`).

The baseline file records the `threads` (allocation width) it was taken at, and
a check run at a different width is refused rather than silently compared --
`threads` reaches the conversion as `-@ N` and sizes a rayon pool, so it moves
the `maxrss_mb` that gates. Re-record if you change `--cpus-per-task`.

Corpora and results are never committed. Corpora are seed-deterministic --
regenerating from the same seed reproduces byte-identical input (a
500,000-sample point is ~420 MB and its seed is 8 bytes). Results are runtime
measurements (`wall_s`, `maxrss_mb`, ...); they are not reproducible from a
seed, only re-obtainable by rerunning the probe.

## Gotchas

- Corpus and result files go under `$CLAUDE_JOB_DIR/tmp`. `/tmp` is reaped on
  this cluster and has destroyed corpora mid-run.
- Slurm hands out non-contiguous CPU ids; pin with `os.sched_getaffinity(0)`,
  never `taskset -c 0-15`.
- `pixi run test` does not rebuild the Rust extension. Run
  `maturin develop --release` before any Python-level verification of a Rust
  change, or you are timing stale code.
- Rust tests need `--no-default-features --features conversion`, and
  `CARGO_TARGET_DIR` must point off NFS.
- **`unit_secs` includes downstream backpressure, not just shard work.** The
  shard workers' result channel is bounded (`tx_res` at `workers * 2`, feeding
  a `tx_dense` bounded at 6), so a worker blocks inside its chunk loop
  whenever the executor is behind. As `GENORAY_READER_WORKERS` rises past the
  executor's drain rate, every unit's time converges on that drain rate and
  the per-shard spread collapses -- so a skew fitter can wrongly read a
  *narrow* `unit_secs` spread as "readers are evenly loaded" when the real
  limiter is downstream. Treat a narrow spread at high worker counts as
  suspect, not as evidence of balance.
- **`pending` / `pending_bytes` are per-contig high-water marks, not a time
  series.** They are non-decreasing within a contig (`3 -> 5 -> 5`, never
  down), so they cannot be diffed to recover instantaneous backlog, and they
  are **not summable across contigs** -- the probe takes a global maximum over
  sampler lines, never a sum.
- **A missing sampler line is missing data, not `pending=0`.** Any contig that
  converts faster than roughly the sampler's settle time plus one sample
  interval emits no sampler line at all, and the parsed record is then
  indistinguishable from a genuine zero backlog. Small-`S` sweep points are
  exactly where this bites; don't read a zero there as "no backlog observed"
  when it may mean "never measured."
- **`pending_hw` is the high-water of chunks already waiting, excluding the
  one currently arriving.** The gauge is sampled before a chunk is inserted
  into the reorder map, not after, so a chunk released the instant it arrives
  (perfectly in-order, never actually buffered) contributes 0. `pending_hw ==
  0` means no reordering backlog was ever observed for that contig; it is
  no longer floored at 1 in every sharded run.
- **`pending_hw` still grows with `w` for structural reasons, so it is not by
  itself evidence of skew.** `ReorderBuffer::push` releases a chunk on arrival
  only when its ordinal is the head, so the `w - 1` units ahead of the head
  keep everything they produce buffered until the head unit finishes:
  `(w - 1) * chunks_per_unit` chunks are resident even with perfectly balanced
  readers. A real 12-unit, `w=3`, `overshard=4` probe log sustains
  `pending=5`. `model.py:decide` therefore gates H3(a) on the backlog's BYTE
  share of measured peak RSS, not on the spec's literal `pending_hw >= w/2`,
  which every planned sweep row would trip.
- **Only the production-`chunk_size` points carry an `rss_ceiling_mb`, and
  those are the only ones that run with `MALLOC_ARENA_MAX=1`.** The ceiling is
  enforced with `RLIMIT_AS`, which bounds address space rather than RSS, and
  glibc's default multi-arena allocator reserves VA the process never touches;
  pinning to one arena is what keeps the ceiling a usable proxy (see
  `probe.py:_preexec`). It is a deviation from the production default, so those
  points' `maxrss_mb` and wall times are not the bare production configuration
  -- which is exactly why the law-fitting points (V-ladder, cost laws, contig
  counterfactual, knee validation) do NOT set a ceiling: `MALLOC_ARENA_MAX=1`
  was measured at 73% slower in an earlier multithreaded conversion regime, and
  those points feed the H2 verdict and every wall time in the sweep. See
  `build_plans.OOM_PROBE_CEILING_MB`.
- **`phase1_s` is a SUM of per-contig spans, not a wall clock.** `probe.py`'s
  `RE_PHASE1` matches the renderer's per-contig "done: N kept, M excluded
  (X.Xs)" line and sums every match. At `concurrent_chroms == 1` that sum
  equals wall time, but the contig axis also runs `concurrent_chroms =
  min(c, 4)`, where up to four contigs' spans overlap in wall time -- so
  `phase1_s` overstates by up to 4x versus the `concurrent_chroms == 1` row of
  the same pair. `phase1_s` is therefore only comparable across rows with the
  same `concurrent_chroms`; use `wall_s`, not `phase1_s`, for the contig
  counterfactual.
- **The V-law predicts phase 1, so the hold-out is scored against `phase1_s`,
  never `wall_s`.** `fit_v_law` fits `phase1_s ~ a + b*V`, which is why
  `extrapolate` returns `predicted_phase1_s` rather than a wall time.
  `ProbeRecord.wall_s` additionally carries the reader-independent rayon merge
  tail and process startup and so is always the larger number; scoring the
  projection against it adds a strictly positive term to every hold-out error,
  one-sidedly, into a 25% gate that means "the model is invalid". The V-ladder
  and hold-out corpora are both single-contig, so both sides of that
  comparison are one uncontended span. If `phase1_s` is 0 (no span in the
  trace) the time half of the gate is skipped and reported as skipped -- there
  is no correct fallback to `wall_s`.

- **The cohort exponent needs TWO V-ladders; the scale ladder alone cannot
  measure it.** The scale ladder holds `S*V = CELLS_BUDGET` at every rung, so
  the cohort law's regressand `log(phase1/V)` is identically `log(phase1) +
  log(S) - log(cells)` and its slope is `1 + dlog(phase1)/dlog(S)`. A
  constant-cells ladder is BUILT so every rung does the same total work, so
  `phase1` is flat (36-44s across a 2000x cohort range) and the slope collapses
  to 1 no matter what the underlying cost structure is. It reported
  `beta=1.0020, CI [0.9689, 1.0352]` -- tight enough to read as a solid
  measurement, from a design that could not have returned anything else.
  `model.py:cohort_beta_is_design_forced` detects this and says so.
  `vlinear`/`vlinear2` fix it: within each ladder S is fixed and V varies, so
  each fitted slope IS a per-variant cost at that cohort size, and beta is the
  log-ratio of the two (`fit_cohort_beta_from_ladders`). Measured that way
  `beta=0.9860, CI [0.9808, 0.9912]` -- outside the forced fit's own CI, so
  that fit was not merely unidentified but wrong and falsely confident.
  Refitting took the F=0 hold-out from a 40% "MODEL FAILURE" to 13%.
  - `VLINEAR2_SAMPLES` is deliberately NOT the hold-out's cohort size. A ladder
    at the hold-out's S would make the hold-out an interpolation inside the
    fitted data, so the gate would go quiet for the wrong reason. The two
    ladders (S=250 and S=250,000) BRACKET the hold-out's S=100,000.
  - `beta` is not a true constant: per-cell cost is mildly U-shaped in S
    (2.59e-8 s/cell at S=4,000-16,000 against 3.16e-8 at S=250 and 2.97e-8 at
    S=500,000), so the estimate depends on which cohorts anchor it -- the
    S=250/S=100,000 pair gives 0.9592, the S=250/S=250,000 pair 0.9860, and
    those CIs do not overlap. A single power law is an approximation; it is
    fitted across the range that brackets the hold-out, which is where it is
    used.
- **Never compare records measured on different nodes.** `point_id` hashes
  every field of `SweepPoint` but deliberately NOT the machine, so a resumed
  sweep skips work already paid for -- which also means two records can share a
  `point_id` and disagree wildly. Measured: `2ac9bbbfbe0dc691` (holdout_f0,
  w=1, chunk 875) took **151.9s on carter-cn-03 and 73.2s on carter-cn-04**, a
  2.08x spread, while same-node controls reproduced within 1.9% and a repeated
  control was identical to 0.0%. That one cross-node record was the entire
  "40% MODEL FAILURE": the true same-node error is 13%. `ProbeRecord.node`
  records the machine and `model.py` refuses to charge a cross-node gap to the
  model. Node speed varies by more than the 25% gate, so sweeps that will be
  compared must pin `--nodelist`.

## Prior findings

`legacy_pr140/README.md` records the PR #140 review: the knee sits at w≈3-7 and
moves with cohort size, not core count, because only the shard readers are
parallel — `executor::run_compute_engine` is a serial loop. That is the result
this harness extends to biobank scale.

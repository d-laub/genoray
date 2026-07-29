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
- **Any point with an `rss_ceiling_mb` runs with `MALLOC_ARENA_MAX=1`.** The
  ceiling is enforced with `RLIMIT_AS`, which bounds address space rather than
  RSS, and glibc's default multi-arena allocator reserves VA the process never
  touches; pinning to one arena is what keeps the ceiling a usable proxy (see
  `probe.py:_preexec`). It is a deviation from the production default, so the
  measured `maxrss_mb` is not strictly the bare production configuration. The
  effect was measured as ~1% on the `from_vcf_list` cross-contig peak during
  earlier work; it has NOT been measured on this `from_vcf` path, so treat the
  deviation as small-but-unquantified here rather than as established.
- **`phase1_s` is a SUM of per-contig spans, not a wall clock.** `probe.py`'s
  `RE_PHASE1` matches the renderer's per-contig "done: N kept, M excluded
  (X.Xs)" line and sums every match. At `concurrent_chroms == 1` that sum
  equals wall time, but the contig axis also runs `concurrent_chroms =
  min(c, 4)`, where up to four contigs' spans overlap in wall time -- so
  `phase1_s` overstates by up to 4x versus the `concurrent_chroms == 1` row of
  the same pair. `phase1_s` is therefore only comparable across rows with the
  same `concurrent_chroms`; use `wall_s`, not `phase1_s`, for the contig
  counterfactual.

## Prior findings

`legacy_pr140/README.md` records the PR #140 review: the knee sits at w≈3-7 and
moves with cohort size, not core count, because only the shard readers are
parallel — `executor::run_compute_engine` is a serial loop. That is the result
this harness extends to biobank scale.

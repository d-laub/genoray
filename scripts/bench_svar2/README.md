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
| `legacy_pr140/` | the original PR #140 sharded-reader harness and its findings |

## Running

```bash
pixi run bench-regression            # ~2 min, guards against regressions
sbatch scripts/bench_svar2/sweep_scale.sbatch   # overnight, full scale sweep
```

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
- **`pending_hw` has a floor of 1 in any sharded run.** Even a perfectly
  ordered chunk is inserted into the reorder map before it is drained, so the
  H3 RAM model's `(workers + pending_hw)` term always carries a constant `+1`
  that is not real backlog.

## Prior findings

`legacy_pr140/README.md` records the PR #140 review: the knee sits at w≈3-7 and
moves with cohort size, not core count, because only the shard readers are
parallel — `executor::run_compute_engine` is a serial loop. That is the result
this harness extends to biobank scale.

# PGEN RAM-law re-fit on the bounded reader (Task 9 Step 4)

Analysis-only. `src/budget.rs` is untouched by this document — the ship /
no-ship decision on new coefficients is the maintainer's, not this
document's, and it is presented here as an open question with two positions
attributed, not resolved.

This is a re-fit of `RamLaw::PGEN` against a sweep run on the code that
replaced PGEN's per-sample reader RAM with fixed byte budgets (issue #155,
PR #154, shipped). The precedent document
(`docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit.md`) fit the
*old*, unbounded reader; this document fits the same law's functional form
against the *new*, bounded one and finds the form itself does not survive
the change cleanly. This document matches that precedent's structure.

## Run record

- **Node:** `carter-cn-04`.
- **Job:** 13351684, `svar2-pgen`, `sacct`: `COMPLETED`, exit `0:0`, elapsed
  `00:18:11`.
- **Allocation:** `--cpus-per-task=48 --mem=64G`.
- **Commit the sweep ran at:** `4c9f38b` (echoed by the sbatch script). The
  branch tip this document is written against is `59665aa`, which the
  handoff states differs from `4c9f38b` only by the `probe.py:_tmp_dir`
  bench-script fallback fix below — it cannot affect these measurements.
- **Data (committed alongside this document):**
  `docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/{pgen.ndjson,pgen.json,manifests/}`.

### Failed attempts (recorded for an honest history)

1. **Job 13351680 — FAILED after 6h57m53s, exit 1.** All corpora had already
   been built (the ~7h corpus-generation cost was fully paid) when the run
   failed on its **first measurement point**. `probe.py:_tmp_dir` preferred
   `$CLAUDE_JOB_DIR/tmp` when set, and `sbatch` exports the submitting
   session's environment by default — so the job inherited a
   `$CLAUDE_JOB_DIR` scoped to node-local scratch on the *submitting* node,
   not the node the job actually landed on (or re-landed on after
   requeuing). The path dangles as a symlink off any other node, and
   `mkdir(parents=True, exist_ok=True)` under it fails
   `FileNotFoundError` then `FileExistsError` on the symlink itself, naming
   neither the symlink nor the node — this is the same trap recorded
   against a different sweep in the project memory. Fixed in `59665aa`
   (`fix(bench): fall back when $CLAUDE_JOB_DIR is another node's scratch`).
2. **Job 13351684 — the run this document reports.** Corpus generation
   served entirely from cache (the ~7h cost from the failed run's corpora),
   so all 18 sweep points completed in 18m11s total.

Net effect: corpus generation cost ~7h exactly once (the failed job); the
successful resubmission paid only the measurement time.

## Scope check — no FORMAT/dosage fields in this sweep

`pgen.json`'s 18 points carry no `dosages` key at all (verified directly:
`grep -c "dosages"` over the committed `pgen.json` returns 0), and every one
of the 7 committed manifests' `format_fields` is `[]` (verified by loading
each manifest). This sweep is entirely on the **no-FORMAT path**. It says
nothing about the dosage/FORMAT-field path, which is scoped separately by
issue #156 (`format_vals` retention sits outside both byte budgets, ~3.5 GB
at S=128,000 with one dosage field — untouched by this refit). Issue #157
(`from_svar1` under-budgeting `chunk_size` when `n_format_fields=0`) is a
different backend (`from_svar1`, not `from_pgen`) and does not apply here
either.

## Full 18-row table

`chunk_size (resident) = min(chunk_size (nominal), variants)` — the same
`_resident_chunk_size` clamp `_ram_rows` applies, since a nominal chunk
larger than the corpus's variant count is never actually resident, only
reserved address space. `chunk_bytes (MB) = manifest.chunk_bytes ×
chunk_size (resident) / 1e6` is the byte regressor `fit_ram_law` uses. `cc`
is the run's **actual** `concurrent_chroms`: the swept value for the 6
concurrency-axis rows, or — for the 12 rows that left `concurrent_chroms`
unset — the value the planner itself resolved for this sweep's fixed shape
(48 cores, 22 contigs, `max_mem` from the job's 64G cgroup budget), traced
below to **`cc=8` for all 12**, correcting the Step 2 base report's initial
`cc=7` (see "Correction: the resolved `cc` for the 12 unset rows"
immediately below).
`pending_highwater` is 0 in all 18 rows — the "pipeline sampler" trace
`pending_hw` is parsed from is VCF-sharded-reader-only instrumentation; PGEN
never emits it. `reader_workers=1` for all 18 (pinned for PGEN).

| point_id | S | V | chunk_size (nominal) | chunk_size (resident) | reader_workers | cc (actual) | pending_hw | chunk_bytes (MB) | measured maxrss_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abdf808362e218cf | 4,000 | 250,000 | 7,812 | 7,812 | 1 | 8 | 0 | 7.81 | 2690.16 |
| 072a83680d1a5e3e | 4,000 | 500,000 | 15,625 | 15,625 | 1 | 8 | 0 | 15.62 | 3086.07 |
| b2b658cfffd72e33 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 8 | 0 | 25.00 | 3364.68 |
| 7415fc17cddce4c5 | 32,000 | 250,000 | 7,812 | 7,812 | 1 | 8 | 0 | 62.50 | 7142.65 |
| 891735dd8ccab234 | 32,000 | 500,000 | 15,625 | 15,625 | 1 | 8 | 0 | 125.00 | 7225.04 |
| dd57703cd7f1a900 | 32,000 | 1,000,000 | 25,000 | 25,000 | 1 | 8 | 0 | 200.00 | 7908.43 |
| 324909c9ad1ffd56 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 1 | 0 | 25.00 | 1860.21 |
| 0e3f67d6aa854d38 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 4 | 0 | 25.00 | 2886.49 |
| 4fc094955737f5a9 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 8 | 0 | 25.00 | 3917.35 |
| 90e21de4103ddad5 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 11 | 0 | 25.00 | 3586.42 |
| 3a2cec9ee7147014 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 16 | 0 | 25.00 | 4081.68 |
| 750af98224a7ec17 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 22 | 0 | 25.00 | 4416.10 |
| f4c9de9c26bc9369 | 4,000 | 1,000,000 | 3,125 | 3,125 | 1 | 8 | 0 | 3.12 | 2611.66 |
| 0e9db5662f84c495 | 4,000 | 1,000,000 | 12,500 | 12,500 | 1 | 8 | 0 | 12.50 | 2670.59 |
| c403da8f505457ed | 32,000 | 1,000,000 | 3,125 | 3,125 | 1 | 8 | 0 | 25.00 | 4247.61 |
| 7554141f4ca9ece6 | 32,000 | 1,000,000 | 12,500 | 12,500 | 1 | 8 | 0 | 100.00 | 4815.40 |
| 04fc06d7f6d90569 | 128,000 | 250,000 | 3,125 | 3,125 | 1 | 8 | 0 | 100.00 | 4727.82 |
| 4c3c00c461c01c80 | 128,000 | 250,000 | 7,812 | 7,812 | 1 | 8 | 0 | 249.98 | 5299.50 |

### Correction: the resolved `cc` for the 12 unset rows

The Step 2 base analysis initially assumed the 12 `concurrent_chroms=null`
rows resolved to `cc=7` by reusing the precedent doc's
`plan_thread_budget(48, 22)` arithmetic. That arithmetic is the
**monolithic-VCF-reader** thread planner and is never called on the PGEN
path; `plan_sharded` is (`src/lib.rs:429-609`), and it is capped by
`PGEN_MAX_CONCURRENT`, not by `plan_thread_budget`'s 6-cores-per-contig
rule. Retracing the actual code path: `probe.py:_build_cmd` never passes
`--max-mem`, so `python/genoray/_svar2.py:1256-1266` falls back to
`detect_memory_budget()`, which prefers the cgroup limit
(`python/genoray/_utils.py:243-273`); the sbatch job's `--mem=64G` and
`MEM_BUDGET_FRACTION=0.8` give `max_mem_bytes ≈ 54,975.58 MB`. At every
corpus in this sweep, `headroom_mb` under that budget is large enough
relative to `per_contig_mb` that `mem_bound` never binds (smallest observed
`mem_bound=21`, at the largest corpus). So the pre-cap `cc` is always
`min(core_bound=23, n_contigs=22)=22`, and `src/lib.rs:589` then applies
`.min(crate::budget::PGEN_MAX_CONCURRENT)` — **`effective_cc=8` for all 12
unset rows**, robustly (insensitive to the exact detected budget over a wide
range). This is used throughout this document in place of the Step 2 base
report's original `cc=7` for those 12 rows; it does not change which
candidate passes or fails the gate, only the margin numbers (over-prediction
strengthens when `cc` rises from 7 to 8, since `plan_sharded`'s prediction
scales with `cc`).

## Fitted coefficients

Reproduced independently against the committed data in this session
(`PYTHONPATH=. pixi run python`, `_load_manifests` / `load_sweep` /
`_ram_rows` / `fit_ram_law`, none reimplemented), byte-for-byte matching
both the Step 2 report and the controller's own independent reproduction:

```
excluded: []
n rows: 18
RamLaw(base_mb=3214.5413104416407, per_sample_mb=-0.011099390799718542,
       kappa=22.090953684269625, r2=0.5352700642835617, n_points=18)
```

| Coefficient | Value |
|---|---|
| `base_mb` | 3214.5413 |
| `per_sample_mb` | −0.011099 |
| `kappa` | 22.0910 |
| `r2` | 0.5353 |
| `n` | 18 |

### Coefficient table (OLS SE from `sigma² · (XᵀX)⁻¹`, same design matrix `fit_ram_law` builds: `[1, samples, (workers+pending)·chunk_bytes/1e6]`)

dof = 15.

| Coefficient | Estimate | SE | t | p | 95% CI |
|---|---:|---:|---:|---:|---|
| `base_mb` | 3214.5413 | 391.5931 | 8.2089 | 6.26e-07 | [2379.880, 4049.202] |
| `per_sample_mb` | −0.011099 | 0.011405 | −0.9732 | 0.3459 | [−0.035409, 0.013210] |
| `kappa` | 22.0910 | 6.4170 | 3.4426 | 0.003627 | [8.4135, 35.7684] |

R² fell from the precedent's 0.8872 (n=12) to 0.5353 (n=18) despite more
data. That is not evidence the fixed code fits worse — see **Finding (a)**
below for the mechanism.

## Finding (a): `concurrent_chroms` is absent from the fit

`_ram_rows`/`fit_ram_law` (`scripts/bench_svar2/model.py:927-938`,
`:270-283`) build the chunk regressor as `(workers + pending) · chunk_bytes`
and never reference `pt.concurrent_chroms` anywhere. In this sweep
`reader_workers==1` and `pending_highwater==0` for all 18 rows, so
`(workers+pending)==1` everywhere and the fitted regressor is bare
`chunk_bytes` — `cc` never enters the design matrix at all.

`plan_sharded` (`src/budget.rs:271-289`), the consumer, does the opposite:
it charges `kappa · (w+pending) · chunk_bytes/1e6` **per contig** and
implies a total predicted peak of `base_mb + per_sample_mb·S + cc ·
kappa·(w+pending)·chunk_bytes/1e6`. The fit and the consumer are fitting
different models.

The consequence is directly visible in the 6-row `cc` ladder
(`324909c9...` through `750af982...`): all six share S=4,000,
`chunk_bytes=25.00 MB`, `workers+pending=1` — `fit_ram_row` builds the
**identical** `[1, 4000, 25.0]` design-matrix row for all six — while
`maxrss_mb` spans **1860.21 → 4416.10 MB (2.37×)** purely as a function of
the omitted `cc ∈ {1,4,8,11,16,22}`. Six replicate-`X` rows with that much
spread in `y` cap the achievable R² by construction. This is the direct
cause of the R² collapse from the precedent's 0.887 to this refit's 0.535 —
it is a property of the design (the ladder here spans multiple `cc`
values), **not** evidence that the fixed code behaves worse than the old
code. The precedent's own ladder ran at a single `cc≈7`, which hid the same
omission by making it a constant rather than a spread.

## Finding (b): the per-contig constant is now measured, not inferred

OLS on the six ladder rows alone (`maxrss_mb ~ 1 + cc`, S/V/chunk_size held
fixed by construction, n=6, dof=4):

| | Estimate | SE | t | p | 95% CI |
|---|---:|---:|---:|---:|---|
| intercept | 2351.85 | 353.75 | 6.648 | 0.0027 | [1369.69, 3334.01] |
| **slope (MB/contig)** | **107.05** | **28.23** | **3.792** | **0.0192** | **[28.67, 185.44]** |

R²=0.782. The slope is statistically significant (p=0.019, CI excludes 0),
and its 95% CI **contains** the known additive constant:
`RAW_STAGE_BYTES + MASK_STAGE_BYTES` = 64 MiB + 64 MiB = **128 MiB =
134.217728 MB** per live `ChunkAssembler` (`src/chunk_assembler.rs:391-392`),
one of which is constructed per concurrently-processed contig
(`src/orchestrator.rs:896`) — so the true additive cost scales as `128 MiB
× cc`, independent of `chunk_bytes`. Ratio of measured slope to known
constant: 107.05/134.22 = 0.798 (within the CI's noise). The final
whole-branch review that shipped the bounded-reader branch predicted this
term analytically before any refit sweep existed; this sweep measures it
directly.

## Finding (c): the 3-coefficient functional form is structurally inadequate

A single multiplicative `kappa` cannot price a chunk-independent per-contig
constant across this sweep's chunk-size range.

**Crossover where `kappa·chunk_MB` (at the as-fitted `kappa=22.0910`) falls
under the known constant:** `134.2177 / 22.0910 = 6.076` — so at any
`chunk_MB` below **≈6.08 MB**, `kappa·chunk_MB` alone under-represents the
known per-contig floor. Both of this sweep's `chunk_size=3,125` rows sit
partly in or near this zone (`f4c9de9c26bc9369` at `chunk_MB=3.12`,
literally below the floor), and it is exactly the closest call in the
acceptance table below.

**The `kappa` needed to just cover the constant at the smallest measured
chunk overshoots badly at the largest.** Solving `kappa · 3.125 =
134.2177` gives `kappa = 42.95`. Applying that same `kappa` at this sweep's
largest measured `chunk_MB=249.98` (`4c3c00c461c01c80`, S=128,000,
`chunk_size=7,812`, `cc=8`):

```
per_contig = 42.95 × 249.98 = 10,736.6 MB
total chunk term = 8 × 10,736.6 = 85,892.5 MB
vs. actual measured maxrss at that row: 5,299.5 MB
```

A **16.2× overshoot**. This is not a tuning problem — no single global
`kappa` can simultaneously cover the additive per-contig constant at small
`chunk_size` and avoid gross over-conservatism at large `chunk_size`; it is
a structural mismatch between the shipped functional form and the true
cost.

## Finding (d): the margin's true provenance

The shipped law's "conservative margin" is not a deliberate safety factor.
It is the arithmetic consequence of fitting `kappa` blind to `cc` — so
`kappa` absorbs roughly the training data's dominant `cc` — and then
`plan_sharded` multiplying by `cc` **again** at prediction time.

Refitting with `cc` explicit in the regressor
(`y ~ 1 + S + cc·chunk_MB`, i.e. the literal quantity `plan_sharded` uses,
using `effective_cc` from the Correction section above: 8 for the 12
unset rows, the swept value for the 6 ladder rows):

| Coefficient | Estimate | SE | t | p | 95% CI |
|---|---:|---:|---:|---:|---|
| `base_mb` | 3121.10 | 377.05 | 8.278 | <0.0001 | [2317.43, 3924.76] |
| `per_sample_mb` | −0.011149 | 0.010452 | −1.067 | 0.303 | [−0.033428, 0.011130] |
| `kappa` | **2.8438** | 0.7346 | 3.871 | 0.0015 | [1.2779, 4.4096] |

R²=0.5838, n=18. `kappa` drops **7.8×**, from the cc-blind fit's 22.09 to
2.84 — almost exactly the cc-blind fit's implicit dominant `cc≈8`
(`22.09/2.84 ≈ 7.78`). The cc-blind `kappa=22.09` is not a genuine
per-contig-per-MB rate; it is that rate already multiplied by roughly the
training data's dominant `cc=8`, baked in by the design-matrix omission in
Finding (a), and then `plan_sharded` multiplies by `cc` a **second** time
when the cc-blind coefficients are deployed. This matters because it means
the current law's margin cannot be reasoned about as if it were a chosen
safety factor — it is a byproduct of a fitting-vs-consuming model mismatch
that happens, in this training data, to point in the conservative
direction.

## Finding (e): the residual hazard is unreachable in production

At the smallest tested chunk (`chunk_MB=3.125`), the as-shipped cc-blind
`kappa=22.0910` charges `22.0910 × 3.125 = 69.03 MB/contig` — under the
measured ~107 MB/contig (Finding b) and well under the known 134.2
MB/contig constant. The under-charging widens with `cc` (the chunk-blind
error is per-contig and multiplied by `cc` in production). This would be
reachable at `cc=22` — but `src/lib.rs:589` clamps every production plan to
`.min(crate::budget::PGEN_MAX_CONCURRENT)` (`PGEN_MAX_CONCURRENT = 8`,
`src/budget.rs:178`). `cc > 8` is reachable only through the
`GENORAY_CONCURRENT_CHROMS` bench-only override — exactly how this sweep's
ladder rows reached 11, 16, and 22 in the first place. So **`cc ≤ 8` is
enforced by code, not merely documented**, and the under-charging regime
this analysis surfaces cannot be entered by a production user of the
current `PGEN_MAX_CONCURRENT` cap.

## Finding (f): `per_sample_mb` is negative in every specification

| Specification | `per_sample_mb` | SE | p | 95% CI |
|---|---:|---:|---:|---|
| cc-blind (as `fit_ram_law` produces) | −0.011099 | 0.011405 | 0.3459 | [−0.035409, 0.013210] |
| consumer-matching (`cc` in the regressor, Finding d) | −0.011149 | 0.010452 | 0.303 | [−0.033428, 0.011130] |
| with an explicit additive `cc` term | −0.0104 | 0.0113 | 0.383 | not computed |

Never significant; CI always spans zero. This is not a fit failure — it is
the direct, expected consequence of the branch working as designed: Tasks
1–4 (the presence-bitset byte-budget change, PR #154) removed the
per-sample reader cost this coefficient used to price, so the true
coefficient is now ≈0, and OLS across only three sample-count levels
(4,000/32,000/128,000) lands slightly under it on noise.

It is, however, unshippable as a naive point estimate. Extrapolating
`base_mb + per_sample_mb·S` to S=500,000 with the cc-blind coefficients:

| S | baseline_mb |
|---:|---:|
| 4,000 | 3170.14 |
| 32,000 | 2859.36 |
| 128,000 | 1793.82 |
| 500,000 | **−2,335.15** |
| 1,000,000 | **−7,884.85** |

A negative baseline is physically meaningless. Worse, it breaks two
existing tests in `src/budget.rs` that exist specifically as a safety
guard, independent of any judgement call:

1. `ram_law_pgen_is_a_usable_law` asserts `RamLaw::PGEN.per_sample_mb >=
   0.0` — fails outright on any of the specifications above.
2. `pgen_budget_too_small_for_one_contig_is_an_error_not_a_silent_cc_of_one`
   runs at `n_samples=1,000,000`, where the cc-blind baseline is
   `3214.54 − 0.011099×1,000,000 = **−7,884.85 MB**`. The test's
   `budget_mb < baseline_mb` assertion (`1.0 < −7884.85`) goes false, and
   `plan_sharded` would no longer return `InsufficientMemory` at that
   scale at all, because `headroom_mb = budget_mb − baseline_mb` becomes
   positive off a negative baseline. A negative `per_sample_mb` does not
   merely lose accuracy here — it inverts the baseline-dominated guard that
   exists to stop an OOM after a partial store has already been written.

## Acceptance gate — evaluated the way `plan_sharded` evaluates it, at each row's `effective_cc`

Quoting the pre-registered criterion: ship only if the fitted law
over-predicts at every measured point, evaluated the way `plan_sharded`
evaluates it. `plan_sharded` (`src/budget.rs:271-289`) computes, given
`max_mem_bytes`:

```
baseline_mb   = ram.base_mb + ram.per_sample_mb * n_samples
pending       = reader_workers.saturating_sub(1)   // = 0, pgen's pinned w=1
per_contig_mb = ram.kappa * (reader_workers + pending) * (chunk_bytes / 1e6)
predicted     = baseline_mb + cc * per_contig_mb
```

evaluated at each row's `effective_cc` (8 for the 12 unset rows, the swept
value for the 6 ladder rows — see Correction above).

### Candidate A — plain 3-coefficient refit (`fit_ram_law`'s direct output)

**PASSES**, all 18 points over-predict. Worst margin **+1,110.8 MB /
1.425×**, at `f4c9de9c26bc9369` (S=4,000, V=1,000,000, `chunk_size=3,125`,
`cc=8`) — a looser (safer) worst-case margin than the currently-shipped
law's own worst case (+621.2 MB / 1.159×). But Finding (d) shows the
mechanism by which it passes is the fit-vs-consumer model mismatch, not
correctly-priced conservatism, and Finding (f) shows it is unshippable
outright on the negative-baseline / existing-test grounds above.

### Candidate B — consumer-matching (`kappa=2.8438` from Finding d)

**FAILS.** Under-predicts at **6 of 18** points:

| point_id | S | chunk_MB | cc | actual (MB) | predicted (MB) | ratio |
|---|---:|---:|---:|---:|---:|---:|
| **7415fc17cddce4c5** | 32,000 | 62.50 | 8 | 7,142.7 | 4,186.1 | **0.586** (worst) |
| 891735dd8ccab234 | 32,000 | 125.00 | 8 | 7,225.0 | 5,608.1 | 0.776 |
| c403da8f505457ed | 32,000 | 25.00 | 8 | 4,247.6 | 3,333.1 | 0.785 |
| 04fc06d7f6d90569 | 128,000 | 100.00 | 8 | 4,727.8 | 3,969.0 | 0.840 |
| dd57703cd7f1a900 | 32,000 | 200.00 | 8 | 7,908.4 | 7,314.4 | 0.925 |
| 4fc094955737f5a9 | 4,000 | 25.00 | 8 | 3,917.3 | 3,645.3 | 0.931 |

Every under-predicting row is S=32,000 or S=128,000 — cohorts the 6-row
`cc` ladder never covered (it only ran at S=4,000). This is a **measured**
under-prediction, not a hypothetical one: once the fit is honestly
specified to match what the consumer computes, a plain OLS mean fit fails
the gate on real data.

### Candidate E1 — `per_sample_mb` pinned to 0

`base_mb=3206.7997, per_sample_mb=0.000000, kappa=17.4777` (cc-blind
convention retained, `per_sample_mb` fixed rather than fitted).
**PASSES**, worst margin **+1,032.1 MB / 1.395×**. Pinning to exactly 0
asserts the per-sample cost is zero, which the data does not establish — it
only establishes the cost is indistinguishable from zero (Finding f).

### Candidate E2 — `per_sample_mb` at the 95% upper confidence bound

`base_mb=3197.5857, per_sample_mb=0.013210, kappa=11.9870` — the cc-blind
convention retained deliberately (no bench/`model.py` code change), with
`per_sample_mb` set to the cc-blind fit's own 95% **upper** CI bound
(0.013210, from the coefficient table above) rather than its negative point
estimate. **PASSES**, all 18 points over-predict, worst margin **+938.4 MB
/ 1.359×** — better (more conservative) than the current law's own worst
margin (+621.2 MB / 1.159×). Rationale: taking the upper confidence bound
rather than asserting exactly zero is the same philosophy `kappa` already
uses in the precedent doc ("a conservative bound, not a fitted rate"); the
point estimate being negative is read as the branch working as designed
(the per-sample reader term is gone), not a fit failure to be papered over
by forcing zero.

### Current shipped law, unchanged, evaluated at each row's `effective_cc`

`base_mb=2688.5256, per_sample_mb=0.120409, kappa=6.841965` (fitted on the
*old*, unbounded-reader code and data). **PASSES**, all 18 points
over-predict, worst margin **+621.2 MB / 1.159×** at `4fc094955737f5a9`
(S=4,000, `chunk_MB=25.00`, `cc=8`) — nearly identical in both magnitude and
*location* to the currently-shipped law's own worst case on the *old*
sweep (+621 MB/1.16× at cc=8, S=4,000). The `maxrss_mb` value at this exact
configuration, `3,917.34765625`, is bit-identical across the two
independent sweeps 16 days apart, which is a deterministic-conversion
cross-check, not a coincidence.

### Candidate comparison and ledger

| Candidate | Gate (18 pts) | Mechanism | S=500,000 baseline |
|---|---|---|---|
| (A) plain 3-coef refit | PASS (worst 1.425×) | Passes via an artifact: omitting `cc` from the fit lets `kappa` absorb ≈ the training data's dominant `cc=8`, then `plan_sharded` multiplies by `cc` again | **negative, physically invalid** |
| (B) consumer-matching refit | **FAIL** (6/18 under-predict, worst 0.586×) | Properly specified; once fit, `kappa` is genuinely lower and cannot cover S=32,000/128,000 | negative |
| (E1) `per_sample_mb := 0` | PASS (worst 1.395×) | cc-blind convention, per-sample cost asserted (not established) at exactly 0 | 3,207 MB (3.1 GiB) |
| (E2) `per_sample_mb :=` 95% upper CI bound | PASS (worst 1.359×) | cc-blind convention, per-sample cost bounded conservatively rather than pinned | 9,803 MB (9.6 GiB) |
| current shipped law, unchanged | PASS (worst 1.159×) | Fitted on old code/data; independently re-clears the new bounded-code RSS with almost its original margin | 62,893 MB (61.4 GiB) |

Step 2's own candidate table (a slightly different letter scheme — its "E"
is this document's Candidate E, folding the *known* 134.2177 MB/contig
constant into the additive term and refitting the remainder rather than
taking an upper CI bound):

| Candidate | Gate (18 pts) | Worst ratio |
|---|---|---|
| A — plain refit, `kappa=22.09` | PASS | 1.425× |
| B — consumer-matching, `kappa=2.84` | **FAIL** | 0.586× (S=32,000) |
| E — known 128 MiB×cc folded in, remainder refit (`kappa=2.4659`) | **FAIL** | 0.576× |
| current shipped law | PASS | 1.159× |

Step 2's Candidate E and this document's Candidate E2 are different
constructions sharing a letter by coincidence of independent naming: Step
2's E is Candidate B's shape with the known constant subtracted before
refitting (still cc-explicit, still fails); this document's/the
controller's E2 is cc-blind like Candidate A, with only `per_sample_mb`
replaced by its upper CI bound.

## Finding (g): the open decision

Both positions are presented here attributed, unresolved, for the
maintainer to decide — this document does not pick a winner.

**Position 1 — ship candidate E2.** `base_mb=3197.5857,
per_sample_mb=0.013210` (the 95% upper confidence bound, not the point
estimate), `kappa=11.9870`. Passes the gate at all 18 points, worst margin
**+938.4 MB / 1.359×** (better than the current law's +621.2 MB / 1.159×),
and cuts the S=500,000 baseline from 62,893 MB (61.4 GiB) to 9,803 MB
(9.6 GiB) — see Before/after below. Reduces the extrapolation reach from
15.6× (32,000→500,000, the precedent's own figure) to 3.9×
(128,000→500,000). Keeps `fit_ram_law`'s cc-blind convention deliberately,
so no bench code change is required. Argument: E2 strictly dominates
keeping the current law on every measured axis — more conservative at
every point, far less wasteful at biobank scale — so "ship nothing" is
argued not to be a neutral default.

**Position 2 — ship no refit (the Step 2 agent's independent
recommendation).** Any candidate that passes the gate cc-blind (A, E1, E2)
passes via the cc-omission artifact of Finding (d) — its apparent margin
comes from a fitting-vs-consuming model mismatch, not correctly-priced
conservatism. Once the model is correctly specified (Candidate B,
consumer-matching `kappa=2.84`, or Step 2's Candidate E with the known
constant folded in), it **under-predicts real measured points** by up to
1.74× at S=32,000/128,000 — not a hypothetical extrapolation risk but a
measured failure on rows this sweep actually ran. The current law
independently re-clears the new bounded-code data at 1.159×, essentially
unchanged from its original 1.16× worst case 16 days and one branch of
changes earlier — the strongest evidence in this whole analysis for leaving
it alone: it was not tuned against this data at all, and it still holds.
Argument: a mean OLS fit can never legitimately guarantee "over-predicts at
every point" — roughly half its residuals fall below the line by
construction — so any cc-blind candidate that appears to satisfy that
criterion is doing so by a mechanism other than the one the criterion is
meant to test, and should not be trusted to generalize past this specific
sweep's `cc` mix.

Both positions agree on the diagnosis (Findings a–f); they disagree only on
what risk is acceptable to take on data this sweep did not measure (small
`chunk_size` crossed with high `cc`, S beyond 128,000). That is a judgement
call on production OOM risk appetite that this document declines to make.

## MAINTAINER DECISION (2026-08-07): SHIP E2

The open question in Finding (g) has been resolved. The maintainer chose
**Candidate E2** (`base_mb=3197.5857`, `per_sample_mb=0.013210` — the 95%
upper confidence bound — `kappa=11.9870`); these coefficients now ship as
`RamLaw::PGEN` in `src/budget.rs`.

**Why E2 over the status quo.** E2 is more conservative than the
previously shipped law at *every* measured point (worst-case margin
1.359× vs. 1.159×), *and* it is 6.4× less wasteful at biobank scale
(baseline @ S=500,000: 62,893 MB → 9,803 MB). It does not trade safety for
efficiency against the status quo — it dominates it on both axes at once,
which is why "ship nothing" was not treated as a neutral default.

**Why E2 over "ship no refit" (Position 2, not deleted — recorded above).**
Position 2's core objection is correct and is not disputed: E2 passes the
gate cc-blind, via the same cc-omission artifact as Candidate A, and its
apparent conservatism is partly a fitting-vs-consuming model mismatch
rather than deliberately priced margin. That mechanism is now written into
`RamLaw::PGEN`'s doc comment and tracked as issue #158, rather than left
implicit. But leaving the status quo in place keeps a `per_sample_mb`
(0.120409) that this sweep shows is roughly 9× too high relative to even
E2's conservative upper bound (0.013210) for a per-sample reader cost
Tasks 1–4 had already deleted from the code — carrying forward a
~79 GiB projected host requirement at S=500,000 for a cost that no longer
exists. Position 2's caution about the cc-omission artifact is valid and
now documented, but accepting it does not, by itself, justify continuing
to charge for a per-sample term the branch under test removed.

**Why E2 over E1 (`per_sample_mb := 0`).** Pinning to exactly zero asserts
the per-sample cost *is* zero; the data only shows it is indistinguishable
from zero (95% CI [-0.03540914, +0.01321036], point estimate -0.011099).
Taking the 95% upper bound is the honest conservative choice — the same
"conservative bound, not a fitted rate" philosophy this file already
applies to `kappa` — without asserting a claim (exact zero) the sweep
cannot support.

The dissent in Finding (g) stands as written above; it is not superseded,
only outvoted on this specific risk-appetite call.

## Before/after projected host requirement at S=500,000

Two figures matter here, both traceable to a named source.

**Baseline-only term** (`base_mb + per_sample_mb·500,000`), the convention
the precedent doc and `src/budget.rs`'s own test comments use:

| Law | baseline-only @ S=500,000 |
|---|---|
| Current shipped (unchanged) | 62,893.2 MB ≈ 61.4 GiB |
| Candidate A (plain refit) | **−2,335.2 MB (negative, physically meaningless)** |
| Candidate B (consumer-matching) | **−2,453.4 MB (negative)** |
| Candidate E1 (`per_sample_mb:=0`) | 3,206.8 MB ≈ 3.1 GiB |
| Candidate E2 (`per_sample_mb:=` upper CI) | **9,802.6 MB ≈ 9.6 GiB** |

**Full projected host requirement** — the figure PR #154's own body reports
as "≈79 GiB" for the current law, which is *not* the baseline term alone.
It is `(baseline + kappa·chunk_MB) / 0.8`: the minimum budget `plan_sharded`
needs to accept `cc=1` at all (`mem_bound ≥ 1`, i.e. `headroom_mb ≥
per_contig_mb`) at S=500,000's auto-selected chunk size (chunk_MB=268.4,
`_auto_chunk_size`'s value at S=500,000 per PR #154's own worked table),
divided by `MEM_BUDGET_FRACTION=0.8` to convert a minimum accepted budget
into the host RAM a detected-budget caller would need to clear it. This
document independently recomputed PR #154's own arithmetic
(`baseline=62,893.2 + chunk=6.842×268.4=1,836.4 → min_budget=64,729.6 MB →
÷0.8 = 80,912.0 MB = 79.02 GiB`) and confirms it reproduces the PR's stated
"≈79 GiB" exactly. Applying the identical methodology (same chunk_MB=268.4,
since `_auto_chunk_size` does not depend on `RamLaw`'s coefficients) to
each candidate:

| Law | min_budget (cc=1) @ S=500,000 | host RAM needed (÷0.8) |
|---|---:|---:|
| Current shipped (unchanged) | 64,729.6 MB | **79.02 GiB** |
| Candidate A (plain refit) | 3,594.1 MB | 4.39 GiB (baseline negative; unshippable regardless — Finding f) |
| Candidate B (consumer-matching) | **−1,690.1 MB (negative)** | not meaningful |
| Candidate E1 (`per_sample_mb:=0`) | 7,897.8 MB | 9.64 GiB |
| Candidate E2 (`per_sample_mb:=` upper CI) | 13,019.9 MB | **15.89 GiB** |

So the two baselines the brief asked this document to compare against —
"~79 GiB" and "61.4 GiB" — are both real and both sourced: 61.4 GiB is the
current law's baseline-only term at S=500,000, and 79 GiB is PR #154's
fuller figure once the minimum chunk term and the 80%-detection-fraction
conversion are included. Under either convention, E2 is the only refit
candidate that both (a) passes the acceptance gate cc-blind and (b) stays
physically sensible at S=500,000: it cuts the full host-RAM figure from
79.02 GiB to 15.89 GiB (5.0×) and the baseline-only figure from 61.4 GiB to
9.6 GiB (6.4×). Candidates A and B go baseline-negative (A's *min_budget*
happens to stay positive only because a large enough chunk term offsets
its negative baseline; B's does not, and goes negative even including the
chunk term) — both are unshippable regardless of the open decision in
Finding (g), independent of which position on that decision the maintainer
takes.

## Reproduction recipe

Reproduce the fit from the committed data (run from the worktree root; the
bench package is not installed, so `PYTHONPATH` must include the worktree
root):

```bash
PYTHONPATH=. pixi run python - <<'EOF'
from pathlib import Path
from scripts.bench_svar2.model import _load_manifests, _ram_rows, fit_ram_law, load_sweep

data = Path("docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data")
manifests = _load_manifests(data / "manifests")
sweep = load_sweep("pgen", data, data, manifests)  # pgen.ndjson / pgen.json both live in `data`
print("excluded:", sweep.excluded)
rows = _ram_rows(sweep)
print("n rows:", len(rows))
law = fit_ram_law(rows)
print(law)
EOF
```

Reproduce the OLS coefficient table (SE/t/p/CI, not part of `fit_ram_law`):

```python
import numpy as np
from scipy import stats

chunk = np.array([(r.workers + r.pending) * r.chunk_bytes / 1e6 for r in rows])
samples = np.array([float(r.samples) for r in rows])
y = np.array([r.peak_rss_mb for r in rows])
X = np.column_stack([np.ones(len(rows)), samples, chunk])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(rows) - X.shape[1]
sigma2 = (resid**2).sum() / dof
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
t = beta / se
p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
```

Reproduce the `cc`-explicit refit (Finding d, Candidate B) by substituting
`cc·chunk_MB` for `chunk_MB` as the third regressor column, using each
row's `effective_cc` (8 for the 12 rows with `concurrent_chroms=None` in
`pgen.json`, the swept value for the 6 ladder rows).

Reproduce the ladder-only per-contig slope (Finding b) by filtering to the
6 rows sharing S=4,000/`chunk_bytes=25.00 MB`/`workers+pending=1` and
running `maxrss_mb ~ 1 + cc` OLS on those rows alone.

## Raw data (committed for auditability)

- `docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/pgen.ndjson`
  (18 lines, one per sweep point)
- `docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/pgen.json`
  (the plan `load_sweep` joins against)
- `docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/manifests/*.manifest.json`
  (all 7 corpus manifests `_load_manifests` globs)

## Verification note

`src/budget.rs` is untouched by this document, so `pixi run test` and the
Rust `cargo test` suites were not re-run as part of writing it — there is
no code change here for them to validate. The independent Python
reproduction above (`fit_ram_law` against the committed data,
byte-for-byte matching both the Step 2 subagent's and the controller's
independent runs) is the verification this document relies on.

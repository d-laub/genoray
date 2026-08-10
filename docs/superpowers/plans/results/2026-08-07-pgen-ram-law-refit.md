# PGEN RAM-law re-fit on the bounded reader (Task 9 Step 4, re-shipped)

Fits `RamLaw::PGEN` against a sweep run on the code that replaced PGEN's
per-sample reader RAM with fixed byte budgets (issue #155, PR #154,
shipped). The precedent document
(`docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit.md`) fit the
*old*, unbounded reader; this document fits the same law's functional form
against the *new*, bounded one. This revision replaces an earlier version of
this document that was written against **contaminated** sweep data — see
the section immediately below before reading anything else here.

## Contamination, revert, and clean re-run (read this first)

A first attempt at this re-fit shipped (commit `63a6b41`) and was reverted
one commit later (`51a1a9c`) because its sweep was **12 of 18 rows stale**:
`sweep.py`'s resumable-sweep cache keys `pending_points` on `point_id` alone
and skips anything already present in the results file, and the pinned node
(`carter-cn-04`) still held the **2026-08-05** sweep's output under that
same cache path. The job that "measured" 18 points genuinely ran only 6 —
the other 12 were the 2026-08-05 sweep's rows re-emitted under a new job ID,
describing the *old, unbounded* reader while claiming to describe the new
bounded one.

**Two independent proofs, both verified directly against the files rather
than taken on a reviewer's word:**

1. `cmp`-ing the first 12 lines of the contaminated `pgen.ndjson` against
   the *entire* 2026-08-05 `pgen.ndjson`: **byte-identical**.
2. Fitting those 12 rows alone reproduces `2688.5256180212755 /
   0.12040939851127153 / 6.841965259264865`, R²=0.8872 — the pre-revert
   `RamLaw::PGEN` coefficients to the last digit.

A second, independent problem the first review pass missed: the sweep's
entire 6-row `concurrent_chroms` ladder (the points that swept `cc ∈
{1,4,8,11,16,22}` explicitly) was *also* entirely drawn from the 2026-08-05
set. Era was confounded with cohort size (S=128,000 appeared in zero stale
rows, S=4,000 in nine of them), which is what drove the contaminated
18-point fit's `per_sample_mb` negative.

**Actions taken:** the branch was reverted to `51a1a9c` (old, 2026-08-05
coefficients restored — reverted rather than patched forward, since the
branch must not carry a law whose doc claims a validation that did not
happen); the stale node-local result cache was rotated out of the way (job
13351697); the sweep was resubmitted pinned to `carter-cn-04` at `51a1a9c`
(job **13351698**, `sacct`: `COMPLETED`, exit `0:0`, elapsed **33m15s**).
Corpora were untouched by the rotation and served from cache, so this cost
33 minutes rather than the original ~7h of corpus generation. The resulting
log carries **18 measurement lines** (vs. 6 for the contaminated run), and
the 12 rows that share a configuration with the 2026-08-05 sweep no longer
reproduce that sweep's coefficients when fit alone (they now fit `2343.69 /
-0.00231 / 15.531`, distinct from `2688.5256 / 0.1204094 / 6.841965`) —
direct evidence the cache is no longer serving 2026-08-05 output.

**This document reports the clean re-run (job 13351698) only.** Every
number below — the fit, the gate, the before/after — is regenerated from
`docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/`, which
now holds that clean data (replacing the contaminated commit).

## Run record

- **Node:** `carter-cn-04` (pinned via `--nodelist`).
- **Job:** 13351698, `svar2-pgen`, `sacct`: `COMPLETED`, exit `0:0`, elapsed
  `00:33:15`.
- **Allocation:** `--cpus-per-task=48 --mem=64G`.
- **Commit the sweep ran at:** `51a1a9c` (the reverted state — old,
  2026-08-05 `RamLaw::PGEN` coefficients — so the planner's `cc` choice for
  the 12 `concurrent_chroms`-unset rows matches what actually ran; see
  "Resolved `cc` for the 12 unset rows" below for why the exact law in
  force at measurement time doesn't change the answer).
- **Data (committed alongside this document):**
  `docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/{pgen.ndjson,pgen.json,manifests/}`.

### Failed attempts (recorded for an honest history)

1. **Job 13351680 — FAILED after 6h57m53s, exit 1.** All corpora had already
   been built (the ~7h corpus-generation cost was fully paid) when the run
   failed on its first measurement point. `probe.py:_tmp_dir` preferred
   `$CLAUDE_JOB_DIR/tmp` when set, and `sbatch` exports the submitting
   session's environment by default — so the job inherited a
   `$CLAUDE_JOB_DIR` scoped to node-local scratch on the *submitting* node,
   not the node the job actually landed on. Fixed in `59665aa`
   (`fix(bench): fall back when $CLAUDE_JOB_DIR is another node's scratch`).
2. **Job 13351684 — completed, but CONTAMINATED.** See "Contamination,
   revert, and clean re-run" above. This job's data was committed
   (`c23134e`), used to write an earlier version of this document
   (`d3ae55e`), and shipped a re-fit (`63a6b41`) that was reverted
   (`51a1a9c`) once the contamination was found.
3. **Job 13351697 — cache rotation, `COMPLETED 0:0`.** Moved the stale
   `out/pgen.ndjson` on `/local/dlaub` aside to
   `out/pgen.ndjson.stale-aug05-and-13351684` without touching the 8.8G
   corpus cache.
4. **Job 13351698 — the run this document reports.** Corpora served
   entirely from cache; with the stale result cache out of the way, all 18
   points were genuinely measured in 33m15s.

## Scope check — no FORMAT/dosage fields in this sweep

`pgen.json`'s 18 points carry no `dosages` key (`grep -c "dosages"` over the
committed `pgen.json` returns 0), and every one of the 7 committed
manifests' `format_fields` is `[]`. This sweep is entirely on the
**no-FORMAT path**. It says nothing about the dosage/FORMAT-field path,
which is scoped separately by issue #156 (`format_vals` retention sits
outside both byte budgets and is untouched by this refit).

## Full 18-row table

`chunk_size (resident) = min(chunk_size (nominal), variants)` — the clamp
`_ram_rows` applies, since a nominal chunk larger than the corpus's variant
count is never actually resident. `chunk_bytes (MB) = manifest.chunk_bytes ×
chunk_size (resident) / 1e6` is the byte regressor `fit_ram_law` uses.
`pending_hw` is 0 in all 18 rows (PGEN never emits the VCF-sharded-reader
`pending_highwater` trace). `reader_workers=1` for all 18 (pinned for PGEN).

| point_id | S | V | chunk_size (nominal) | chunk_size (resident) | reader_workers | cc (plan) | pending_hw | chunk_bytes (MB) | measured maxrss_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abdf808362e218cf | 4,000 | 250,000 | 7,812 | 7,812 | 1 | unset → 8 | 0 | 7.81 | 2032.44 |
| 072a83680d1a5e3e | 4,000 | 500,000 | 15,625 | 15,625 | 1 | unset → 8 | 0 | 15.62 | 2360.57 |
| f4c9de9c26bc9369 | 4,000 | 1,000,000 | 3,125 | 3,125 | 1 | unset → 8 | 0 | 3.12 | 2617.92 |
| 0e9db5662f84c495 | 4,000 | 1,000,000 | 12,500 | 12,500 | 1 | unset → 8 | 0 | 12.50 | 2772.94 |
| b2b658cfffd72e33 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | unset → 8 | 0 | 25.00 | 2765.77 |
| 7415fc17cddce4c5 | 32,000 | 250,000 | 7,812 | 7,812 | 1 | unset → 8 | 0 | 62.50 | 3275.93 |
| 891735dd8ccab234 | 32,000 | 500,000 | 15,625 | 15,625 | 1 | unset → 8 | 0 | 125.00 | 4269.65 |
| c403da8f505457ed | 32,000 | 1,000,000 | 3,125 | 3,125 | 1 | unset → 8 | 0 | 25.00 | 4186.04 |
| 7554141f4ca9ece6 | 32,000 | 1,000,000 | 12,500 | 12,500 | 1 | unset → 8 | 0 | 100.00 | 4440.80 |
| dd57703cd7f1a900 | 32,000 | 1,000,000 | 25,000 | 25,000 | 1 | unset → 8 | 0 | 200.00 | 5281.41 |
| 04fc06d7f6d90569 | 128,000 | 250,000 | 3,125 | 3,125 | 1 | unset → 8 | 0 | 100.00 | 4746.91 |
| 4c3c00c461c01c80 | 128,000 | 250,000 | 7,812 | 7,812 | 1 | unset → 8 | 0 | 249.98 | 5698.37 |
| 324909c9ad1ffd56 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 1 | 0 | 25.00 | 1977.91 |
| 0e3f67d6aa854d38 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 4 | 0 | 25.00 | 2190.55 |
| 4fc094955737f5a9 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 8 | 0 | 25.00 | 2614.11 |
| 90e21de4103ddad5 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 11 | 0 | 25.00 | 3015.00 |
| 3a2cec9ee7147014 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 16 | 0 | 25.00 | 3300.43 |
| 750af98224a7ec17 | 4,000 | 1,000,000 | 25,000 | 25,000 | 1 | 22 | 0 | 25.00 | 3835.01 |

### Resolved `cc` for the 12 unset rows

`probe.py:_build_cmd` never passes `--max-mem`, so `detect_memory_budget()`
falls back to the cgroup limit — the sbatch job's `--mem=64G` with
`MEM_BUDGET_FRACTION=0.8` gives `max_mem_bytes ≈ 54,975.58 MB`. Under that
budget, `headroom_mb` is large enough relative to `per_contig_mb` at every
corpus in this sweep that the pre-cap `cc` (`min(core_bound=23,
n_contigs=22)=22`) never binds on memory — this holds under *either* the
old (2026-08-05) law in force when the sweep actually ran, or the new law
shipped by this document, since both yield a memory-bound `cc` well above
`PGEN_MAX_CONCURRENT`. `src/lib.rs` then applies
`.min(crate::budget::PGEN_MAX_CONCURRENT)`, giving **`effective_cc = 8`**
for all 12 unset rows, robustly.

## Fitted coefficients

Reproduced against the committed clean data (`PYTHONPATH=. pixi run
python`, `_load_manifests` / `load_sweep` / `_ram_rows` / `fit_ram_law`,
none reimplemented):

```
excluded: []
n rows: 18
RamLaw(base_mb=2570.300231003748, per_sample_mb=0.007600352463934604,
       kappa=10.793993745504235, r2=0.7697980275674173, n_points=18)
```

| Coefficient | Value |
| --- | --- |
| `base_mb` | 2570.30 |
| `per_sample_mb` | 0.007600 |
| `kappa` | 10.794 |
| `r2` | 0.7698 |
| `n` | 18 |

This is materially different from — and materially healthier than — the
withdrawn contaminated fit (`base_mb=3214.54, per_sample_mb=-0.011099,
kappa=22.09, R²=0.5353, n=18`): `per_sample_mb` is now **positive**
(+0.007600, vs. the contaminated fit's -0.011099), and R² rose from 0.5353
to 0.7698 on the same row count. The negative cohort coefficient the
reverted re-fit was built to work around was an artifact of mixing two code
eras (Finding in the contamination section above), not a property of the
bounded reader.

### Coefficient table (OLS SE from `sigma² · (XᵀX)⁻¹`, same design matrix `fit_ram_law` builds: `[1, samples, (workers+pending)·chunk_bytes/1e6]`)

dof = 15.

| Coefficient | Estimate | SE | t | p | 95% CI |
|---|---:|---:|---:|---:|---|
| `base_mb` | 2570.300 | 182.138 | 14.112 | 4.58e-10 | [2182.083, 2958.518] |
| `per_sample_mb` | 0.007600 | 0.005305 | 1.433 | 0.1724 | [-0.003707, 0.018907] |
| `kappa` | 10.794 | 2.985 | 3.617 | 0.0025 | [4.432, 17.156] |

Note the roles are **reversed** from the precedent (2026-08-05) fit: there,
`per_sample_mb` was well-determined (p=0.0057) and `kappa` was not
(p=0.382); here `kappa` is well-determined (p=0.0025) and `per_sample_mb` is
not (its CI spans zero). `R²[1,S]=0.5691` vs. `R²[1,S,chunk]=0.7698` — the
chunk/`kappa` term adds **+0.2007** to R² in this fit (vs. only +0.0106 in
the 2026-08-05 fit), so unlike the precedent, the headline R² here is
substantially carried by the chunk-size regressor, not just the
cohort-intercept split.

## Shipped construction — both slopes at their 95% CI upper bound

`RamLaw::PGEN` ships:

```
base_mb:       2570.300231003748       (fitted value, unchanged)
per_sample_mb: 0.018907303115116077    (95% CI upper bound)
kappa:         17.155662709761774      (95% CI upper bound)
```

**Construction:** the intercept is held at its fitted value; each uncertain
*slope* is independently raised to its own 95% CI upper bound. This is `>=`
the plain fit in every term, so it cannot under-predict anywhere the plain
fit doesn't. It applies the standing rule that a coefficient used as a
memory *bound* is a conservative bound, not a point estimate — the margin
this buys is not slack to be tuned away. `per_sample_mb`'s point estimate is
positive but its CI still spans zero (+0.0076, CI
[-0.0037, +0.0189]), so it stays labelled a **conservative bound**, not a
fitted rate, same as `kappa`.

**Why the intercept is not also refit.** Refitting `base_mb` on the
residual after pinning both slopes to their upper bounds pulls it *down* —
to **1900.87** — and the resulting law then **fails the gate at S=4,000**.
Pinning coefficients high in one part of an OLS refit pushes the others
down to keep fitting the same data; that can leave a "conservative"
variant less conservative than where it started. `base_mb` is held at its
own independently-fitted value specifically to avoid this trap.

## The ladder rows show a real per-contig slope — but it is out of scope for this law

The withdrawn contaminated fit claimed a "measured 107.05 MB/contig"
per-contig constant, offered as corroborating Task 4's `RAW_STAGE_BYTES +
MASK_STAGE_BYTES = 128 MiB = 134.2177 MB` staging cost per live
`ChunkAssembler`. That number came entirely from stale rows and is
**withdrawn** — it should not be treated as ever having been measured.

The clean re-run's 6 ladder rows (S=4,000, `chunk_bytes=25 MB`, `cc` swept
over `{1, 4, 8, 11, 16, 22}`, `maxrss_mb` from 1977.9 to 3835.0 MB) *are*
genuinely re-measured this time, and a bare OLS fit of `maxrss_mb ~ 1 + cc`
on those 6 rows alone is well-determined:

| | Estimate | SE | t | p | 95% CI |
|---|---:|---:|---:|---:|---|
| intercept | 1895.58 | 55.70 | — | — | — |
| **slope (MB/contig)** | **89.67** | 4.45 | 20.17 | 3.56e-05 | [77.33, 102.01] |

R²=0.9903, n=6. This is real, but two things keep it from being what the
contaminated finding claimed:

1. **It does not corroborate the 128 MiB staging guess.** Its 95% CI
   ([77.3, 102.0] MB/contig) *excludes* 134.2177 MB — unlike the
   contaminated fit's much wider CI ([28.67, 185.44]), which happened to
   contain it by coincidence of noise, not agreement.
2. **It is a separate observation from the shipped `RamLaw`.** `fit_ram_law`
   builds its regressor as `(workers+pending)·chunk_bytes` and never
   references `concurrent_chroms` — `RamLaw::PGEN` remains a 3-term,
   cc-blind fit regardless of what this direct ladder-only regression shows.
   Folding a `cc` term into the shipped law is out of scope for this task
   and would need its own acceptance-gate check across the full 18-row
   sweep, not just the 6-row ladder.

This is offered as an honest, separately-labelled finding, not as
justification for anything in the shipped coefficients above.

## The margin's provenance is a fitting artifact, not a chosen safety factor

`fit_ram_law`'s chunk regressor is `(workers+pending)·chunk_bytes` and is
never multiplied by `concurrent_chroms`. `plan_sharded` (`src/budget.rs`),
the consumer, charges `kappa·(w+pending)·chunk_bytes/1e6` **per contig** —
i.e. its implied total is `base_mb + per_sample_mb·S + cc ·
kappa·(w+pending)·chunk_bytes/1e6`. The fit and the consumer are fitting
different models: `kappa` absorbs roughly this sweep's dominant `cc`, and
`plan_sharded` then multiplies by `cc` a second time at prediction time.

Refitting with `cc` explicit in the regressor (`y ~ 1 + S + cc·chunk_MB`,
using `effective_cc=8` for the 12 unset rows and the swept value for the 6
ladder rows) on the clean 18-row data:

| Coefficient | Estimate | SE |
|---|---:|---:|
| `base_mb` | 2496.21 | 155.79 |
| `per_sample_mb` | 0.006349 | 0.004319 |
| `kappa` (cc-explicit) | **1.510** | 0.304 |

R²=0.8373, n=18. The cc-explicit `kappa` (1.510) is **~7.1–11.4× smaller**
than the cc-blind point estimate (10.794) and the shipped CI-upper `kappa`
(17.156) — close to this sweep's dominant `cc=8`, exactly the mechanism
above. As expected of a mean OLS fit, this properly-specified refit
**under-predicts at 10 of the 18 points** (closest: 0.717× at S=32,000,
chunk=25 MB, cc=8) — a cc-blind candidate that appears to "pass" the
acceptance gate at every point is doing so via this specification mismatch,
not because it is a correctly-priced conservative model. The over-charge is
real, does make the shipped bound safer, and should not be read as a
deliberately engineered margin. Tracked as issue **#158**.

## Acceptance gate — evaluated the way `plan_sharded` evaluates it, at each row's actual/resolved `cc`

```
baseline_mb   = ram.base_mb + ram.per_sample_mb * n_samples
pending       = reader_workers.saturating_sub(1)   // = 0, pgen's pinned w=1
per_contig_mb = ram.kappa * (reader_workers + pending) * (chunk_bytes / 1e6)
predicted     = baseline_mb + cc * per_contig_mb
```

**Shipped construction: PASSES.** Over-predicts at all 18 measured points.
Worst-case margin **+456.9 MB / 1.1745×** at `f4c9de9c26bc9369` (S=4,000,
chunk_bytes=3.125 MB, cc=8). Largest over-prediction **6.90×** at
`4c3c00c461c01c80` (S=128,000, chunk_bytes=249.98 MB, cc=8).

**Previously shipped (2026-08-05) law, independently re-evaluated against
this clean data:** also passes (all 18 over-predict), worst margin +723.3
MB / **1.2763×** at the same row, largest over-prediction **5.58×** at
S=128,000. The new law is tighter at the closest call (1.1745× vs. 1.2763×)
and looser at the largest measured point (6.90× vs. 5.58×) — both directions
are expected: the old law's `kappa` was fitted on the old, unbounded
reader's much larger RSS, so it over-shoots by more at large chunk sizes but
under-shot the new reader's true small-chunk floor by more.

## What actually happened: maintainer decision, invalidated, then re-applied to clean data

An earlier version of this document (written against the contaminated
18-row fit) presented an open decision between several candidates and
recorded that **Candidate E2 was the position chosen**: `per_sample_mb` set
to its (contaminated) cc-blind fit's 95% upper CI bound, `kappa` left at
that same (contaminated) fit's point estimate. That choice was made on data
later found to be 12/18 stale and was invalidated along with everything
else in that document — it never should have shipped, and the branch was
reverted before it could.

The shipped construction on this clean data (labelled **D** above,
`per_sample_mb` and `kappa` both at their CI upper bounds, `base_mb` pinned)
is **the same philosophy** as E2 — take the fitted value and, wherever its
CI does not clearly separate from zero, use the conservative bound rather
than the point estimate — applied to the clean data instead. It differs
from E2's *literal* recipe (which only pushed `per_sample_mb` to its bound
and left `kappa` at the point estimate) because on this clean data that
literal recipe would ship `kappa=10.794` (the plain point estimate) —
**below** the value Construction D ships (17.156) — which would contradict
the conservatism rule this whole exercise exists to enforce. (E2's literal
recipe, re-evaluated on the clean data for reference, still passes the gate
— all 18 over-predict, worst margin +297.9 MB / 1.1138× — but that is a
*looser* bound than Construction D's 1.1745×, which is why D is shipped
instead of E2's literal recipe.)

## Before/after: projected host requirement at S=500,000

**Baseline-only term** (`base_mb + per_sample_mb·500,000`), the convention
`src/budget.rs`'s own test comments use. (GiB below is computed correctly
as `MB × 1e6 / 1024³`, not `MB / 1024` — that shortcut, inherited from PR
#154 and this document's own precedent, overstates GiB by ~5%: 62,893.2 MB
is 58.6 GiB, not 61.4.)

| Law | baseline-only @ S=500,000 |
|---|---|
| Previously shipped (2026-08-05, fitted on the old unbounded reader) | 62,893.2 MB ≈ 58.6 GiB ≈ 62.9 GB |
| **This document's shipped law** | **12,024.0 MB ≈ 11.2 GiB ≈ 12.0 GB** |

A **5.2×** reduction in the baseline-only figure. `per_sample_mb` itself
fell **0.1204094 → 0.0189073** (≈6.4×) — expected, since the branch this
sweep measures replaced the old per-sample reader allocation with fixed
byte budgets (issue #155/PR #154).

**Full projected host requirement** — the figure PR #154's own body reports
as "≈79 GiB" for the old law: `(baseline + kappa·chunk_MB) / 0.8` at
S=500,000's auto-selected chunk size (`chunk_MB=268.4`, from
`_auto_chunk_size`, independent of `RamLaw`'s coefficients), divided by
`MEM_BUDGET_FRACTION=0.8` to convert a minimum-accepted budget into the host
RAM a detected-budget caller needs to clear it (`cc=1`, i.e.
`mem_bound >= 1`):

| Law | min_budget (cc=1) @ S=500,000 | host RAM needed (÷0.8) |
|---|---:|---:|
| Previously shipped (2026-08-05) | 64,729.6 MB | **≈75.4 GiB** |
| This document's shipped law | 16,628.5 MB | **≈19.4 GiB** |

A **3.9×** reduction in the host RAM a caller needs to clear before
`plan_sharded` will accept even `cc=1` at biobank scale — this is the
number that determines whether the planner *refuses* a host outright at
S=500,000, and it drops from requiring a >64 GiB host to one well under 32
GiB.

## Validity domain

- **Three cohort sizes**: 4,000, 32,000, 128,000 samples. `per_sample_mb` is
  extrapolated **~3.9×** beyond the largest measured cohort (128,000) to
  reach a representative S=500,000 (vs. the precedent 2026-08-05 fit's 15.6×
  extrapolation from a max of 32,000).
- **`chunk_bytes` range**: 3.125–250 MB.
- **`reader_workers == 1` and `pending == 0`** in every one of the 18 rows
  (pinned for PGEN; the sharded-VCF `pending_highwater` trace this field
  measures elsewhere is not emitted on this path).
- **22 contigs** in every corpus — no `n_contigs` axis exists in this sweep,
  so (as with the precedent fit) no `n_contigs` term can be identified or
  ruled out from this data alone.
- **One node** (`carter-cn-04`), **one profile**
  (`germline-1kgp-varskew`), **`multiallelic_rate` 0.0**, **one seed** (42).
- **No FORMAT/dosage fields** — scoped entirely to the no-FORMAT path (see
  "Scope check" above; the FORMAT/dosage path is issue #156).
- **`cc <= 8` is enforced in code, not just documented**: `src/lib.rs`
  (around line 589) clamps every planned `concurrent_chroms` to
  `PGEN_MAX_CONCURRENT` (`src/budget.rs`). `cc > 8` is reachable only
  through the bench-only `GENORAY_CONCURRENT_CHROMS` override — exactly how
  this sweep's ladder rows reached 11, 16, and 22 in the first place — not
  by a production caller.

## Reproduction recipe

Reproduce the fit from the committed clean data (run from the worktree
root; the bench package is not installed, so `PYTHONPATH` must include the
worktree root):

```bash
PYTHONPATH=. pixi run python - <<'EOF'
from pathlib import Path
from scripts.bench_svar2.model import _load_manifests, _ram_rows, fit_ram_law, load_sweep

data = Path("docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data")
manifests = _load_manifests(data / "manifests")
sweep = load_sweep("pgen", data, data, manifests)
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
tcrit = stats.t.ppf(0.975, dof)
ci = [(b - tcrit * s, b + tcrit * s) for b, s in zip(beta, se)]
```

Reproduce the ladder-only per-contig slope by filtering to the 6 rows
sharing S=4,000/`chunk_bytes=25.00 MB`/`workers+pending=1` and running
`maxrss_mb ~ 1 + cc` OLS on those rows alone (`cc` from each row's
`SweepPoint.concurrent_chroms` in `pgen.json`).

Reproduce the cc-explicit refit (margin-provenance section) by substituting
`cc·chunk_MB` for `chunk_MB` as the third regressor column, using each row's
resolved `cc` (8 for the 12 rows with `concurrent_chroms=None`, the swept
value for the 6 ladder rows).

## Raw data (committed for auditability)

- `docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/pgen.ndjson`
  (18 lines, one per sweep point — the clean, job-13351698 data; replaces
  the contaminated commit)
- `docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/pgen.json`
  (the plan `load_sweep` joins against)
- `docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit-data/manifests/*.manifest.json`
  (all 7 corpus manifests `_load_manifests` globs)

## Verification note

`src/budget.rs`'s `RamLaw::PGEN` constant and its doc comment were updated
to the coefficients and construction in this document. The three
`RamLaw::PGEN`-dependent tests (`ram_law_pgen_is_a_usable_law`,
`pgen_memory_bound_actually_binds`,
`pgen_budget_too_small_for_one_contig_is_an_error_not_a_silent_cc_of_one`)
were re-derived against the new coefficients, not hand-patched, and pass.
`cargo test --no-default-features --features conversion` and `pixi run
test` were both run in full afterward — see the commit message for exact
pass/fail/skip counts.

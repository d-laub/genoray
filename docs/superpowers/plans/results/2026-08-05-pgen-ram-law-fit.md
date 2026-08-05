# PGEN RAM-law fit and concurrency curve (Task 9)

Measurement-only deliverable for the SVAR2 PGEN budget planner. Fits
`RamLaw::PGEN`'s three coefficients from a real sweep and measures the
`concurrent_chroms` curve that decides whether Task 10 needs a
`PGEN_MAX_CONCURRENT` cap.

**Review round 1 (documentation-only fixes):** the coefficients below are
unchanged from the original measurement and independently reproduced
byte-for-byte from the committed data. This revision corrects how the fit is
*presented*: `kappa` is not statistically significant on its own (see
Identifiability below) and the doc previously implied otherwise; a
consumer-side safety-margin table, a validity-domain statement, and a
ladder-only contrast are added because they are what actually establishes
this fit is safe to ship.

## Run record

- **Node:** `carter-cn-04` (pinned via `--nodelist`, per the cluster's
  documented 2.08x node-speed variance).
- **Allocation:** `--cpus-per-task=48 --mem=64G`.
- **Commit the sweep ran at:** `80b5fd8fb5aabe6cd0e8f7827a5d95b52af4e246`
  (echoed by the sbatch script itself, `=== commit 80b5fd8... ===`, not
  reconstructed after the fact).
- **Job:** 13351539, `sacct`: `COMPLETED`, exit `0:0`, elapsed `00:15:28`.
- **Log:**
  `/carter/users/dlaub/projects/genoray/.claude/worktrees/svar2-pgen-budget-planner/svar2-pgen_13351539.log`
  (not committed — see raw-data paths below for what's durable).
- **sbatch script:** `scripts/bench_svar2/sweep_pgen.sbatch` (committed
  alongside this document, per Task 9 Step 7).

### Failed attempts (recorded for an honest history)

1. **Job 13351419 — FAILED after 4h32m.** Corpus generation completed
   (all 6 corpora + manifests, 5.9G), but the pgen plan family built its
   `SweepPoint.corpus` as the binary `.pgen` path instead of the
   `.manifest.json` path, so `sweep.py`'s manifest read hit
   `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x90 in position 3`
   at the very first sweep point. Fixed in commit `80b5fd8` (both `pgen`
   plan sites in `build_plans.py` now build
   `pgen_s{S}_v{V}.manifest.json`), with a new regression test.
2. **Job 13351538 — FAILED in 10s.** After the fix, resubmission used
   `sbatch --export=ALL`, which leaked the submitting interactive session's
   own `CLAUDE_JOB_DIR` env var into the job. `probe.py:_tmp_dir` prefers
   `$CLAUDE_JOB_DIR/tmp` when set, and that path
   (`/carter/users/dlaub/.claude/jobs/435b5bd1/tmp/bench_probe`) is scoped to
   the *submitting* Claude session, not valid for an independent sbatch job
   on a different node — `mkdir` raced a `FileNotFoundError` then a
   `FileExistsError`. Corpus generation itself cache-hit correctly in this
   run (0.6s, all 6 manifests reused), confirming the plan-family fix was
   already working. Fixed at submit time by clearing the variable:
   `sbatch --export=ALL,...,CLAUDE_JOB_DIR=`.
3. **Job 13351539 — the run this document reports.** Corpus generation
   served entirely from cache (all 6 corpora already on `/local/dlaub` from
   the first attempt), so the sweep reached its first point almost
   immediately and all 12 points completed in 15m28s total.

Net effect: corpus generation cost ~4.5h exactly once (job 1); every
subsequent resubmission paid ~1s for it via the spec+`GENERATOR_VERSION`
cache in `pgen_corpus.generate`.

## Corpus specs

`PGEN_LADDERS` (`scripts/bench_svar2/plans/build_plans.py`): two V-ladders at
different cohort sizes, deliberately — the same "two V-ladders" design that
makes the cohort exponent identifiable for the VCF law.

| samples | variants |
| --- | --- |
| 4,000 | 250,000 / 500,000 / 1,000,000 |
| 32,000 | 250,000 / 500,000 / 1,000,000 |

All six generated from the `germline-1kgp-varskew` profile via `vcfixture
bulk` + `plink2 --make-pgen`, 22 autosomes (`chr1`..`chr22`), seed 42,
`gt-only` payload (`scripts/bench_svar2/pgen_corpus.py`). Combined corpora +
intermediate `.bcf` footprint: 5.9G on `/local/dlaub/pgen-sweep/corpora`
(compresses far better than a naive worst-case bound predicted in Phase A).

**The six ladder points all left `concurrent_chroms` unset** (the planner's
own choice), and that resolves to a specific, load-bearing number:
`plan_thread_budget(available_cores=48, n_chroms=22)` computes
`usable_cores = 48 - 1 = 47`, then
`max_concurrent_by_cores = usable_cores / MIN_THREADS_PER_CHROM = 47 / 6 =
7` (`src/budget.rs`, `MIN_THREADS_PER_CHROM = PIPELINE_THREADS_PER_CHROM(4)
+ MIN_HTSLIB_THREADS(2) = 6`), capped by `n_chroms=22` → **`concurrent_chroms
= 7`**. Half the fit's rows (all six ladder points) ran at `cc=7`; this
number is used explicitly in the consumer-side prediction table below and
cannot be reconstructed from "the planner's own default" alone.

## Fitted `RamLaw::PGEN`

`peak_rss_mb = base_mb + per_sample_mb * samples + kappa * (workers + pending_hw) * chunk_bytes / 1e6`

Fitted with the existing helpers only (`_load_manifests`, `load_sweep`,
`_ram_rows`, `fit_ram_law` — none reimplemented). **These coefficients ship
unchanged from the original measurement:**

```
excluded: []
n rows: 12
RamLaw(base_mb=2688.5256180212755, per_sample_mb=0.12040939851127153,
       kappa=6.841965259264865, r2=0.8872034818397382, n_points=12)
```

| Coefficient | Value |
| --- | --- |
| `base_mb` | 2688.53 |
| `per_sample_mb` | 0.1204 |
| `kappa` | 6.842 |
| `r2` | 0.8872 |
| `n` | 12 |

`sweep.excluded` is empty — all 12 plan points resolved and joined cleanly
(no dropped/OOM/unmatched records).

### Sanity checks (brief Step 4)

1. **R² ≥ 0.8: PASS.** 0.8872 ≥ 0.8 (VCF's fit reached 0.9040 for
   comparison; PGEN's is somewhat lower but clears the bar — and per the
   Identifiability section immediately below, most of that R² comes from the
   cohort-intercept split, not the chunk term).
2. **`kappa > 0` and `per_sample_mb >= 0`: PASS as a sign check** — `kappa =
   6.842 > 0`, `per_sample_mb = 0.1204 >= 0`, and the design matrix is not
   collinear for `samples` (t=3.60, p=0.0057 — see below). **This sign check
   is not the same claim as "`kappa` is a well-determined estimate"**: its
   95% CI spans zero (see Identifiability). The coefficient still ships
   because it functions as a safe upper bound, not because it is precisely
   fitted — see "Consumer-side prediction" for why that's sufficient.
3. **Residual check for an `n_contigs` term: inconclusive by construction**
   (every corpus in this sweep spans the same 22 contigs) — full discussion
   in its own section below, including a related pattern found against
   `concurrent_chroms` instead.

### Identifiability — `kappa` is not statistically significant on its own

OLS on the same design matrix `fit_ram_law` builds (`[1, samples, chunk]`,
`chunk = (workers+pending)·chunk_bytes/1e6`), with standard errors from
`sigma² · (XᵀX)⁻¹`:

| Coefficient | Estimate | SE | t | p | 95% CI |
| --- | --- | --- | --- | --- | --- |
| `base_mb` | 2688.53 | 291.29 | 9.23 | <0.001 | [2029.58, 3347.47] |
| `per_sample_mb` | 0.1204 | 0.0334 | 3.60 | 0.0057 | [0.0448, 0.1960] |
| `kappa` | 6.842 | **7.44** | **0.92** | **0.382** | **[−9.99, +23.68]** |

`samples` is identified (t=3.60, p=0.0057) — the two-cohort-size ladder
(4,000 vs 32,000) does keep it separable from the intercept, as the previous
revision of this doc said. **`kappa` is not**: its 95% CI spans zero, so this
sweep cannot statistically distinguish `kappa=6.842` from `kappa=0` or from
`kappa=20`. Decomposing R²: the intercept+samples-only model
(`R²[1,S]=0.8766`) already explains nearly everything the full model does
(`R²[1,S,chunk]=0.8872`); the chunk/`kappa` term adds only **+0.0106**. The
headline R²=0.8872 is almost entirely the two-cohort intercept split, not
evidence that the chunk-size regressor is doing real work.

**Consequence for how `kappa` is used**: `plan_sharded` (`src/budget.rs`)
uses `kappa` as a per-concurrent-contig marginal memory rate
(`kappa · (w+pending) · chunk_bytes/1e6`, multiplied by `concurrent_chroms`)
to decide how many contigs can run at once under a memory budget. Given the
CI above, `kappa=6.842` must be read as a **conservative upper bound on that
rate for this sweep's regime**, not a precisely fitted per-MB cost — see
"Consumer-side prediction" immediately below for why shipping it anyway is
still safe.

## Consumer-side prediction — the calculation that actually decides safety

The residual table in the previous revision showed only the *fit's*
cc-agnostic prediction and left the direction of the error to Task 10. It is
fully determined by this sweep's own data. `plan_sharded` predicts, at a
given row's actual `concurrent_chroms` (`cc`):

```
peak_predicted = base_mb + per_sample_mb·samples + cc · kappa · (chunk_bytes/1e6)
```

(`pending = reader_workers - 1 = 0` for pgen's pinned `w=1`, matching the
fit's own `workers+pending=1` regressor.) Evaluated at each of the 12 rows'
actual `cc` (7 for the six ladder rows, per the corpus-specs note above; the
swept value for the six concurrency-axis rows):

| S | V | cc | chunk (MB) | actual RSS (MB) | predicted RSS (MB) | margin (MB) | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4,000 | 250,000 | 7 | 7.81 | 2690.2 | 3544.3 | +854.2 | 1.32× |
| 4,000 | 500,000 | 7 | 15.62 | 3086.1 | 3918.5 | +832.4 | 1.27× |
| 4,000 | 1,000,000 | 1 | 25.00 | 1860.2 | 3341.2 | +1481.0 | 1.80× |
| 4,000 | 1,000,000 | 4 | 25.00 | 2886.5 | 3854.4 | +967.9 | 1.34× |
| 4,000 | 1,000,000 | 7 | 25.00 | 3364.7 | 4367.5 | +1002.8 | 1.30× |
| 4,000 | 1,000,000 | 8 | 25.00 | 3917.3 | 4538.6 | **+621.2** | **1.16×** |
| 4,000 | 1,000,000 | 11 | 25.00 | 3586.4 | 5051.7 | +1465.3 | 1.41× |
| 4,000 | 1,000,000 | 16 | 25.00 | 4081.7 | 5906.9 | +1825.3 | 1.45× |
| 4,000 | 1,000,000 | 22 | 25.00 | 4416.1 | 6933.2 | +2517.1 | 1.57× |
| 32,000 | 250,000 | 7 | 62.50 | 7142.7 | 9534.8 | +2392.1 | 1.34× |
| 32,000 | 500,000 | 7 | 125.00 | 7225.0 | 12528.3 | +5303.3 | 1.73× |
| 32,000 | 1,000,000 | 7 | 200.00 | 7908.4 | 16120.4 | +8212.0 | **2.04×** |

**The consumer-side calculation over-predicts at all 12 points.** The
smallest margin — the closest call, i.e. the honest worst case for OOM
risk — is **+621 MB (1.16×) at cc=8, S=4,000**. The S=32,000 ladder rows
over-predict by **1.33–2.04× (2.4–8.2 GB)**, growing with `chunk_bytes`
exactly where a real `concurrent_chroms` term would matter most.

**Conclusion: the missing `concurrent_chroms` regressor costs
under-utilization and spurious `PlanError::InsufficientMemory`, never an
OOM.** A caller with a tight `max_mem` budget may be told fewer contigs fit
than actually would (or none at all, when the true requirement is well
within budget), but `plan_sharded`'s output can never let this sweep's
regime exceed physical memory — every single measured point had headroom
against the plan's own prediction. This is what makes shipping `kappa=6.842`
safe despite it not being statistically significant on its own: as an upper
bound it is a safe one, confirmed empirically rather than assumed.

## Validity domain

This fit and the safety argument above hold only inside what was actually
varied:

- **Two cohort sizes**: 4,000 and 32,000 samples. Nothing between or beyond
  them was measured.
- **One contig count**: every corpus spans all 22 autosomes (`chr1`..`chr22`)
  — see the `n_contigs` discussion below.
- **`concurrent_chroms=7`** for all six RAM-law ladder rows (the planner's
  own choice at 48 cores / 22 contigs); the concurrency-axis rows separately
  cover 1, 4, 8, 11, 16, 22.
- **One node** (`carter-cn-04`), **one profile**
  (`germline-1kgp-varskew`), **one chunk-size rule** (`_chunk_size_for`'s
  floor/clamp), **one seed** (42).

**Extrapolation warning.** `per_sample_mb=0.1204` is intended to scale to
biobank-size cohorts, but the largest measured cohort is 32,000 samples —
reaching a representative target of S=500,000 extrapolates the fit
**~15.6× beyond** the largest measured point. At that scale the
`base_mb + per_sample_mb·S` term alone predicts a **62.9 GB** baseline
(`2688.53 + 0.1204×500,000 ≈ 62,893 MB`), independent of any chunk cost —
**10.8×** the VCF law's per-sample coefficient (`RamLaw::VCF.per_sample_mb =
0.01115`, `0.1204/0.01115 ≈ 10.8`). Whether that ratio reflects a genuine
PGEN-vs-VCF cost difference or an artifact of a two-point cohort ladder is
not something this sweep can distinguish — flagged, not resolved.

## Ladder-only contrast

`_ram_rows` pools all 12 points into one fit. Six of those rows (the
concurrency-axis probe) are **exact duplicates in the design matrix**
(S=4,000, `chunk_bytes=25,000,000`, `workers=1`, `pending=0` — identical `X`
row) but scatter **2.56 GB** in `y` (1860.2 to 4416.1 MB) purely as a
function of the omitted `concurrent_chroms` variable. That is exactly the
kind of omitted-variable noise that inflates residual variance and drags
`kappa`'s significance down in the pooled fit above. Refitting on just the
six ladder rows (all at `cc=7`, so the omitted-variable confound is absent)
isolates the chunk-size response cleanly:

| Fit | n | base_mb | per_sample_mb | kappa | R² |
| --- | --- | --- | --- | --- | --- |
| **Shipped (pooled, all 12 points)** | 12 | 2688.53 | 0.1204 | 6.842 | 0.8872 |
| Ladder-only (6 points, all cc=7) | 6 | 2421.48 | 0.1314 | 6.197 | 0.9927 |

The ladder-only fit's R²=0.9927 is close to the VCF law's 0.9040 reference
and confirms the pooled fit's much lower chunk-term significance is an
artifact of pooling in a cc-confounded axis, not evidence that the
underlying chunk-size response is noisy. **Both fits over-predict at every
measured point** (re-running the consumer-side table above with the
ladder-only coefficients instead of the shipped ones still shows positive
margins throughout, closest call 1.069× at cc=8/S=4,000 vs. the shipped
fit's 1.16× at the same row — tighter but still safe; not included as a
second full table since the conclusion is identical). **We ship the pooled
fit, not the ladder-only one, because its
larger `kappa` (6.842 vs 6.197) is the more conservative bound** — consistent
with kappa's role as an upper bound rather than a precisely-estimated rate
(see Identifiability above).

### Within-cohort chunk response

Fitting a bare slope (`RSS ~ chunk_MB`) separately within each cohort's
three ladder points:

| Cohort | chunk range (MB) | `dRSS/d(chunk_MB)` |
| --- | --- | --- |
| S=4,000 | 7.81 – 25.00 | 38.93 |
| S=32,000 | 62.50 – 200.00 | 5.69 |

A **7× disagreement** between two slopes a single `kappa` is supposed to
hold jointly. Adding `concurrent_chroms` as a fourth regressor across all 12
points (`peak ~ 1 + samples + chunk + cc`) lifts R² to **0.9728** with a `cc`
coefficient of **107.0 MB/contig** — consistent with the direct concurrency
curve's own per-contig RSS growth (§ Concurrency curve). Both within-cohort
slopes (38.93, 5.69) sit **below** `kappa_pooled × cc=7 = 47.9`, so the
disagreement is further conservatism in the shipped fit, not additional
risk: at either cohort size, the true per-chunk-MB cost at the ladder's own
`cc=7` is smaller than what `kappa=6.842` alone (ignoring `cc`) already
implies once multiplied through by `plan_sharded`.

## Residual check for an `n_contigs` term (brief Step 4.3)

Every corpus in this sweep uses the same 22 contigs (`chr1`..`chr22`) —
`n_contigs` (the corpus's own contig count) is **constant across all 12
points**, unlike the VCF sweep's separate `contig` axis
(`CONTIG_COUNTS = (1, 2, 8, 22)`). There is no PGEN analog of that axis in
this data, so residuals cannot be correlated against contig count here — the
check as literally specified is inconclusive by construction, not passed or
failed. No `n_contigs` term is added, per the brief's "only if the residuals
say so" — there is no evidence either way from this sweep.

**The mechanism this check is looking for does exist, though.**
`python/genoray/_svar2.py:1131-1139` eagerly constructs one
`pgenlib.PgenReader` per contig up front, each sized by `n_samples`, all
alive simultaneously — so an `n_contigs × n_samples` term is real, not
hypothetical. It is currently folded entirely into `per_sample_mb` at
whatever this sweep's fixed `n_contigs=22` implies, and back-of-envelope
(one `PgenReader`'s per-sample buffers) puts it at tens of MB, i.e. low risk
at the scales measured here. The direction matters, though: on a corpus
with **more** contigs than this sweep's 22 (e.g. a full assembly with alts
and decoys), the folded-in cost would be an **under-estimate**, not the
over-conservative direction the `concurrent_chroms` gap sits in above.

## Concurrency curve (brief Step 5)

`PGEN_CONCURRENCY = (1, 4, 8, 11, 16, 22)` at `PGEN_CONCURRENCY_AT = (4_000,
1_000_000)`, plus the six ladder rows' own point at the same (S, V), which
ran at the planner's resolved default of **`cc=7`** (see corpus specs
above), included here for reference:

| `concurrent_chroms` | wall_s | phase1_s | maxrss_mb | Δwall vs. prior |
| --- | --- | --- | --- | --- |
| 7 (planner default) | 9.759 | 45.10 | 3364.7 | — |
| 1 | 31.20 | 28.40 | 1860.2 | — |
| 4 | 12.81 | 31.50 | 2886.5 | -58.9% |
| 8 | 10.18 | 50.00 | 3917.3 | -20.5% |
| 11 | 10.04 | 67.30 | 3586.4 | -1.4% |
| 16 | 9.80 | 97.10 | 4081.7 | -2.4% |
| 22 | 9.98 | 111.80 | 4416.1 | +1.8% |

(`phase1_s` is a *sum* of per-contig spans across concurrently-running
contigs, not wall-clock — it rises monotonically with `concurrent_chroms`
simply because more contigs are contributing spans simultaneously.
`wall_s`, the actual elapsed time, is the right column to judge the curve
on, per the brief.)

**The best row in this whole table is `cc=7` — not any point in the swept
`PGEN_CONCURRENCY` grid.** At 9.759s wall / 3364.7 MB RSS it beats the
grid's own cc=8 (10.183s / 3917.3 MB) on *both* axes simultaneously. `cc=7`
isn't part of the swept grid — it only appears because it's what the
planner already runs by default at this shape — so it isn't independently
confirmed across other conditions the way the six grid points are, but it
should not be reasoned about as if absent.

**Verdict: flattens before cc=22 — does not reach the core bound
monotonically.** The big win is cc=1→4 (-58.9%) and cc=4→8 (-20.5%); beyond
cc=8 every step is within ±2.4%, and cc=16→22 is actually *slightly worse*
(+1.8%, noise-level but not an improvement). The knee sits at **cc≈8**. Peak
RSS does not climb monotonically past cc=8 — it *falls* from cc=8 to cc=11
(3917 → 3586 MB) before rising again through cc=16 and cc=22 (3586 → 4082 →
4416 MB). The overall trend from cc=8 to cc=22 is upward (+499 MB / +12.7%
for zero further wall-time benefit), but the cc=8→11 dip is real in the data
and the wording should not paper over it.

One plausible contributor to that non-monotonicity: `probe.run_point`
(`probe.py:377`) keeps the record with the **minimum wall time** across 3
reps, and `maxrss_mb` rides along from whichever rep that happened to be —
it is not a max-of-reps peak. This is pre-existing harness behavior shared
with the VCF fit, not specific to this sweep, but it means each row's
`maxrss_mb` is measured on a different (the fastest) rep, which can
introduce exactly this kind of run-to-run wiggle.

### Cap decision

**Recommend `PGEN_MAX_CONCURRENT = 8`** for Task 10: it is the smallest
value in the *swept* `PGEN_CONCURRENCY` grid that reaches the wall-time
plateau, and capping there avoids paying the extra RSS that cc=11..22 add
for no wall-time benefit. (Pinning to `cc=7` instead would be defensible too
— it's the single best-observed row — but 8 is kept as the recommendation
since it comes from the grid that was actually swept across the full range,
not a single reference point.) This is not a monotonic-to-cc≈22,
no-cap-needed outcome; it is the flattens-before-cc=22 case the brief
anticipated.

**Caveat on generality: this knee was measured at exactly one shape**
(S=4,000, V=1,000,000, 48 cores, 22 contigs). The S=32,000 ladder row at the
same chunk-size regime runs in 63.1s wall vs. this shape's 9.8–31.2s range —
an entirely different reader/executor balance — so the reader-vs-executor
tradeoff that produces this particular knee, and plausibly the knee's
location itself, is not established to hold at other cohort sizes.
`PGEN_MAX_CONCURRENT=8` is being proposed as a single global constant on the
strength of one measured shape; Task 10 should treat it as a starting point
rather than a value validated across the planner's full operating range.

## Raw data (committed for auditability)

- **Committed copies** (this PR):
  - `docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit-data/pgen.ndjson`
    (12 lines, one per sweep point)
  - `docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit-data/pgen.json`
    (the plan `load_sweep` joined against)
  - `docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit-data/manifests/*.manifest.json`
    (all 6 corpus manifests `_load_manifests` globs)
- **Original node-local location** (not durable — `/local` on carter-cn-04,
  may be reclaimed): `/local/dlaub/pgen-sweep/out/pgen.ndjson`,
  `/local/dlaub/pgen-sweep/plans/pgen.json`,
  `/local/dlaub/pgen-sweep/corpora/*.manifest.json`. The full corpora
  (`*.pgen`/`*.bcf`, 5.9G) were deliberately **not** copied off — only the
  small JSON/ndjson evidence needed to reproduce the fit.

Reproduce the fit from the committed copies:

```python
from pathlib import Path
from scripts.bench_svar2.model import _load_manifests, _ram_rows, fit_ram_law, load_sweep

data = Path("docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit-data")
manifests = _load_manifests(data / "manifests")
sweep = load_sweep("pgen", data, data, manifests)  # pgen.ndjson / pgen.json both live in `data`
rows = _ram_rows(sweep)
law = fit_ram_law(rows)
```

Reproduce the identifiability table (OLS standard errors, not part of
`fit_ram_law`):

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

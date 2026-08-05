# PGEN RAM-law fit and concurrency curve (Task 9)

Measurement-only deliverable for the SVAR2 PGEN budget planner. Fits
`RamLaw::PGEN`'s three coefficients from a real sweep and measures the
`concurrent_chroms` curve that decides whether Task 10 needs a
`PGEN_MAX_CONCURRENT` cap.

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

## Fitted `RamLaw::PGEN`

`peak_rss_mb = base_mb + per_sample_mb * samples + kappa * (workers + pending_hw) * chunk_bytes / 1e6`

Fitted with the existing helpers only (`_load_manifests`, `load_sweep`,
`_ram_rows`, `fit_ram_law` — none reimplemented):

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

## Sanity checks (brief Step 4)

1. **R² ≥ 0.8: PASS.** 0.8872 ≥ 0.8 (VCF's fit reached 0.9040 for
   comparison; PGEN's is somewhat lower but clears the bar).
2. **`kappa > 0` and `per_sample_mb >= 0`: PASS.** `kappa = 6.842 > 0`,
   `per_sample_mb = 0.1204 >= 0`. The design matrix is not collinear — the
   two-cohort-size ladder (4,000 vs 32,000) keeps `samples` identifiable
   against the intercept.
3. **Residual check for an `n_contigs` term: NOT IDENTIFIABLE from this
   sweep, but a related, larger pattern found and flagged below.**

### Residual check, in detail

Every corpus in `PGEN_LADDERS` and the concurrency probe uses the same 22
contigs (`chr1`..`chr22`) — `n_contigs` (the corpus's own contig count) is
**constant across all 12 points**, unlike the VCF sweep's separate `contig`
axis (`CONTIG_COUNTS = (1, 2, 8, 22)`). There is no PGEN analog of that axis
in this data, so residuals cannot be correlated against contig count here —
the check as literally specified is inconclusive by construction, not
passed or failed. No `n_contigs` term is added, per the brief's "only if the
residuals say so" — there is no evidence either way from this sweep.

**However**, tabulating residuals against `concurrent_chroms` (the one
per-point setting that *does* vary, on the 6 points sharing
S=4,000/V=1,000,000/chunk_bytes=25,000,000) shows a real, non-noise pattern
the current 3-term law does not capture:

| `concurrent_chroms` | actual RSS (MB) | predicted RSS (MB) | residual (MB) |
| --- | --- | --- | --- |
| 1 | 1860.2 | 3341.2 | -1481.0 |
| 4 | 2886.5 | 3341.2 | -454.7 |
| 8 | 3917.3 | 3341.2 | +576.1 |
| 11 | 3586.4 | 3341.2 | +245.2 |
| 16 | 4081.7 | 3341.2 | +740.5 |
| 22 | 4416.1 | 3341.2 | +1074.9 |

Residuals trend from -1481 MB at cc=1 to +1075 MB at cc=22 — a ~2.56 GB
spread the law attributes entirely to noise because `concurrent_chroms`
isn't a regressor at all (`from_pgen` pins reader `workers=1`, so the
`(workers + pending_hw)` term never moves across this axis; `pending_hw` was
also 0 in every one of these 12 rows). This is **not** "tens of MB" the way
the spec's eager-`PgenReader`-pool prediction assumed for a contig-count
term — it's multi-GB, and it tracks the concurrency *setting*, not the
corpus's contig count. **Flagged for Task 10's attention** as a possible gap
in `RamLaw::PGEN`'s functional form (whether it needs a `concurrent_chroms`
term) rather than fixed here — changing the law's functional form is a
design decision outside this measurement task's mandate, and the
`PGEN_MAX_CONCURRENT` cap recommended below (§ Cap decision) already bounds
production to the lower, better-explained end of this range.

## Concurrency curve (brief Step 5)

`PGEN_CONCURRENCY = (1, 4, 8, 11, 16, 22)` at `PGEN_CONCURRENCY_AT = (4_000,
1_000_000)`, plus the ladder's own point at the same (S, V) with
`concurrent_chroms` left unset (the planner's own default choice) for
reference:

| `concurrent_chroms` | wall_s | phase1_s | maxrss_mb | Δwall vs. prior |
| --- | --- | --- | --- | --- |
| (planner default) | 9.76 | 45.10 | 3364.7 | — |
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

**Verdict: flattens before cc=22 — does not reach the core bound
monotonically.** The big win is cc=1→4 (-58.9%) and cc=4→8 (-20.5%); beyond
cc=8 every step is within ±2.4%, and cc=16→22 is actually *slightly worse*
(+1.8%, noise-level but not an improvement). The knee sits at **cc≈8**: past
that point additional concurrency buys no further wall-time reduction while
peak RSS keeps climbing (3917 → 3586 → 4082 → 4416 MB across cc=8..22, a
further +499 MB / +12.7% on top of what cc=8 already costs, for zero wall
gain).

### Cap decision

**Recommend `PGEN_MAX_CONCURRENT = 8`** for Task 10, backed directly by this
measurement: it is the smallest `concurrent_chroms` value in the swept range
that reaches the wall-time plateau, and capping there also avoids paying the
extra ~500 MB of RSS growth that cc=11..22 add for no wall-time benefit —
directly relevant given the unexplained concurrency-driven RSS residual
flagged above. This is not a monotonic-to-cc≈22, no-cap-needed outcome; it
is the flattens-before-cc=22 case the brief anticipated.

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

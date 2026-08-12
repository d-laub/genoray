# Crossed VCF RAM sweep — closing #158

Sweep Slurm job **13355527**, `carter-cn-03`, **48/48 points measured** (guard:
48 plan points / 48 result rows / 48 unique point ids, single node, `ok=48
failed=0`). Data: `2026-08-11-vcf-ram-law-crossed-data/`. Design and
acceptance criteria were pre-registered on issue #158 (design doc
`docs/superpowers/specs/2026-08-11-ramlaw-vcf-envelope-design.md`) before any
of it existed.

Reproduce the fit:

```bash
D=docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed-data
pixi run python -m scripts.bench_svar2.fit_ram \
  --results $D --plans $D --manifests $D/manifests --backend vcf --margin 1.25
```

`maxrss_mb` spans 170.9–3541.2 MB across the 48 points. The store-digest
oracle passed: 8 distinct digests for the 8 underlying corpora, so varying
`chunk_size` and `concurrent_chroms` changed only memory, never conversion
output.

## Why this sweep was run

`RamLaw::VCF` carried the exact defect `RamLaw::PGEN` was refitted to remove
in `c100d51`: an OLS mean fit over a sweep that never varied
`concurrent_chroms` at fixed `(S, chunk_size)`, so `per_contig_mb` shipped as
`0.0` and `kappa` absorbed the training data's dominant `cc` as an accidental
margin. `plan_sharded` then multiplies the per-contig bracket by `cc` a
second time, so the old law's apparent safety margin was a double-count
artifact, not a chosen factor — and the dangerous half of that defect is the
unpriced `~128 MiB` `ChunkAssembler` per-live-contig staging allocation,
which is the OOM direction, unlike the double-counted `kappa`.

## The gate

```
RamLaw::VCF (margin 1.25):
    base_mb:       457.25887672659735
    per_sample_mb: 0.011017408281198566
    per_contig_mb: 111.42612019477279
    kappa:         6.105786273211022

gate: n=48 passes=True under=0 worst=3.7267x mean=2.4417x min=1.2500x
```

Over-predicts **all 48 measured points**, evaluated exactly as `plan_sharded`
evaluates them at each row's actual `concurrent_chroms` (`fit_ram.predict_mb`
mirrors `budget.rs::plan_sharded`'s equation, not an approximation of it —
see issue #158's root cause). `r2 = -1.91` is printed for description only
and is explicitly **not** the shipping criterion: a law is an upper bound,
not a least-squares prediction, and a negative `r2` here just says the
envelope sits well above the mean, which is exactly the point of fitting a
bound.

**SHIPPED into `src/budget.rs`:**

```rust
pub const VCF: RamLaw = RamLaw {
    base_mb: 457.25887672659735,
    per_sample_mb: 0.011017408281198566,
    per_contig_mb: 111.42612019477279,
    kappa: 6.105786273211022,
};
```

## The interaction-term check (Phase 0) applies here unchanged

The pre-registered decision rule was tested in an earlier, zero-cluster-cost
task against the already-committed 2026-08-08 PGEN crossed data (see
`docs/superpowers/plans/results/2026-08-11-ram-law-form-check.md`):
`per_contig_per_sample_mb` (entering as `cc * samples`) improved the
worst-case ratio only **1.76%** (2.4189x -> 2.3763x) against the required
**>=20%**. **NO-GO.** That check was run once, deliberately, before the VCF
sweep so the VCF data would be fitted against the final functional form
exactly once. `RamLaw` therefore ships **four** coefficients here too;
`per_contig_per_sample_mb` stays the tested-but-dormant `0.0` default in both
the Rust struct and `scripts/bench_svar2/records.py::RamLaw`. `RamLaw::PGEN`
is unchanged by this task — a NO-GO does not trigger the refit a GO verdict
would have required.

## Margin sensitivity

The shipped margin is a **chosen** safety factor, not inherited from a
double-count. Re-run at four margins, same 48 rows:

| margin | base_mb | per_sample_mb | per_contig_mb | kappa | worst | mean | min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 365.80710138127796 | 0.008813926624958853 | 89.14089615581821 | 4.884629018568826 | 2.9814x | 1.9534x | 1.0000x |
| **1.25** | **457.25887672659735** | **0.011017408281198566** | **111.42612019477279** | **6.105786273211022** | **3.7267x** | **2.4417x** | **1.2500x** |
| 1.50 | 548.7106520719169 | 0.01322088993743828 | 133.71134423372732 | 7.326943527853232 | 4.4721x | 2.9301x | 1.5000x |
| 2.00 | 731.6142027625559 | 0.017627853249917707 | 178.28179231163642 | 9.769258037137652 | 5.9628x | 3.9068x | 2.0000x |

Every coefficient scales linearly with the margin because the LP is fit at a
single functional form and the margin only rescales the binding constraint —
`worst` and `mean` are the only non-linear column, and both grow roughly
proportionally to margin as expected of an envelope. 1.25 was chosen as the
same factor `RamLaw::PGEN` already ships, trading worst-case
over-allocation against under-utilisation and spurious
`PlanError::InsufficientMemory`, not tuned to this specific dataset.

## Scoring the sweep's own design against issue #158's pre-registered checks

**Check 4 — is `per_contig_mb` additive across the three cohort widths, or
does it vary beyond its CI?** It varies, clearly beyond noise. Regressing
`maxrss_mb` on `cc` within each `(S, chunk_size)` block (four `cc` levels:
1, 4, 8, 16) at the smallest measured chunk size for each `S`:

| S | chunk_MB | per-contig bracket (MB) | R² |
|---:|---:|---:|---:|
| 4,000 | 3.977 | 48.13 | 0.9739 |
| 32,000 | 3.976 | 89.16 | 0.9898 |
| 128,000 | 3.968 | 175.17 | 0.9978 |

and decomposing each `S`'s three crossed-grid chunk sizes
(`block_bracket = per_contig_mb + kappa * chunk_MB`) into a per-contig-only
intercept:

| S | per-contig-only (MB) | kappa (local) | R² |
|---:|---:|---:|---:|
| 4,000 | 40.65 | 1.664 | 0.987 |
| 32,000 | 90.75 | -0.026 | 0.006 |
| 128,000 | 166.95 | 2.002 | 0.999 |

The bracket clearly **grows with cohort width** (48 → 89 → 175 MB at the
smallest chunk size), the same qualitative pattern the 2026-08-08 PGEN sweep
found (83.7 → 263 → 301 MB) — not additive. The S=32,000 row's local `kappa`
decomposition is not meaningful (R² 0.006, only 3 points): the crossed grid's
three chunk sizes at that cohort width are too close together to separate
`per_contig_mb` from `kappa * chunk_MB` in isolation; only the pooled LP fit
over all 48 rows identifies them jointly. This non-additivity is exactly
what an interaction term (`per_contig_per_sample_mb`) would model — and is
exactly the term Phase 0 gated NO-GO. The envelope fit is safe regardless
(it bounds the worst block), but the single `per_contig_mb` constant does
not describe the mechanism.

**Check 3 — does an `n_chunks` term survive at constant `chunk_bytes`?**
Yes, real but noisier than PGEN's. The `VCF_NCHUNKS` family pins `chunk_size`
and varies `V` at fixed `cc`, holding `chunk_bytes` exactly constant while
chunk count moves:

| S | chunk_size | cc | MB per chunk | R² |
|---:|---:|---:|---:|---:|
| 4,000 | 7,954 | 1 | 1.539 | 0.974 |
| 4,000 | 7,954 | 8 | 2.500 | 0.715 |
| 32,000 | 994 | 1 | 2.215 | 0.817 |
| 32,000 | 994 | 8 | 4.914 | 0.993 |

The slope is positive in all four blocks (real, not noise), but R² is far
weaker than PGEN's 0.9745–1.0000 (this design's `VCF_NCHUNKS` only has 3
`V` levels per block against a 23–89 chunk-count range, so 3 points carry
less power than PGEN's design). Consistent with the 2026-08-08 PGEN finding
and the same reasoning that refused it there (Option C: real in-sample but
not licensed for a large extrapolation), this term is **not** a candidate
`fit_ram_law` currently fits (`interaction=True` only adds
`per_contig_per_sample_mb`, never an `n_chunks` term) and is not shipped
here either.

**Residual σ against the 63 MB reproducibility floor.** An unconstrained OLS
fit of the same four-term design (`base + per_sample*S + cc*(per_contig +
kappa*w*chunk_MB)`, computed for description only, never shipped) gives:

```
base_mb        -58.88   (negative -- an OLS artifact; not physical)
per_sample_mb    0.00857
per_contig_mb   91.34
kappa            1.316
RMSE           335.56 MB   (R² = 0.823)
```

335.56 / 63 = **5.33x** the reproducibility floor measured on the PGEN cc
ladder (six launches differing only in `concurrent_chroms`, R² 0.9903, RMSE
63 MB). Better than PGEN's own 7.3x on this same metric, but still an order
of magnitude above measurement noise: the four-term law does not describe
the mechanism, consistent with check 4's finding that `per_contig_mb` is not
actually constant across cohort widths. The envelope fit is safe regardless
— that is the entire point of fitting a bound rather than a mean — but #158
stays open as a research question about the missing interaction term.

## What the data cannot identify

- **The per-contig term's true functional form.** It clearly grows with `S`
  (check 4), but the pre-registered Phase 0 decision rule already rejected
  the natural next term (`per_contig_per_sample_mb`) on the PGEN data before
  this sweep ran, and that verdict was not re-tested against the VCF data —
  by design, so the VCF sweep would be fitted against the functional form
  exactly once. A future refit could re-run Phase 0's check against this
  VCF dataset specifically; nothing here rules it out or in.
- **`kappa` in isolation at S=32,000.** The three crossed-grid chunk sizes at
  that cohort width are too close together (R² 0.006 for the local
  decomposition) to identify `kappa` from `per_contig_mb` without pooling
  across all `S`.
- **The `n_chunks` term's true slope**, beyond "positive and real": each
  block has only 3 `V` levels, and R² ranges 0.72–0.99 rather than PGEN's
  near-1.0000, so this design has materially less power for that check than
  the PGEN sweep did.
- **`concurrent_chroms_used` from telemetry.** Every one of the 48 rows has
  `concurrent_chroms_used = null` — this is a known gap in
  `logging.rs`'s `FieldGrab` visitor, which forwards only `message` and
  `chrom` to Python, so the `pipeline config` tracing line's structured
  fields never reach the probe. It is **not** a data-quality problem: every
  point in this sweep *pins* `concurrent_chroms` via
  `GENORAY_CONCURRENT_CHROMS`, and `bench_concurrent_chroms` applies that
  override uncapped (unlike the PGEN path, nothing in `lib.rs` clamps VCF
  `concurrent_chroms`), so the realised `cc` equals the pinned `cc` for all
  48 rows, including the `cc=16` lever-arm points. `_ram_rows` uses the
  pinned value and never had to fall back or drop a row for this reason.
- **Production-domain `kappa` beyond 100 MB chunks.** The largest measured
  chunk (`VCF_BIGCHUNK`, 100 MB) is ~2.68x below `_auto_chunk_size`'s 256 MiB
  (268.44 MB) production target — much better than the ~16x this sweep's
  design doc calculated for a same-cells-budget-only design, but still an
  extrapolation, not a measurement.

## Validity domain

S in {4,000, 32,000, 128,000}; chunk_MB roughly 4–100 MB; `cc` in
{1, 4, 8, 16}; `reader_workers == 1` and `pending == 0` in every row; 22
contigs; one node (`carter-cn-03`); `multiallelic_rate` 0.0; no FORMAT/dosage
fields (gt-only payload, matching issue #156). `per_sample_mb` is
extrapolated ~3.9x beyond the largest measured cohort (128,000) to reach the
production target S=500,000.

`cc = 16` sits **outside the production domain**. Unlike the PGEN path
(`PGEN_MAX_CONCURRENT = 8`, enforced in `src/lib.rs`), nothing in this
repo's Rust code clamps VCF `concurrent_chroms` — `cc=16` is reachable only
through the bench-only `GENORAY_CONCURRENT_CHROMS` override, included here
purely to give the per-contig term a lever arm, exactly as the PGEN law
flags its own `cc=16` rows.

## Corpus generator note (not a byte-comparability claim)

Both `RamLaw::VCF` and `RamLaw::PGEN` now come from `vcfixture bulk` against
the same fitted `germline-1kgp-varskew` profile, but from **different CLI
versions**: this VCF sweep used vcfixture-rs v0.5.0+, while the committed
2026-08-08 PGEN corpora predate it (their manifests carry an un-migrated
`profile_hash` that v0.5.x cannot even load — verified: `generator_cli_version`
is unset/`None` on the PGEN manifests, `"0.5.0"` on the VCF ones, and the two
`profile_hash` values differ). v0.5.0 gave variant positions their own PRNG
stream, so a given seed realizes different variants under the two CLI
versions. The two corpora's **fitted distributions** (SFS, class mix, gaps)
are therefore identical, but their **realized draws** are not — the corpora
are not byte-comparable, and neither are the two laws' coefficients,
coefficient-by-coefficient. See `src/budget.rs`'s `RamLaw::PGEN` doc comment
for the same caveat stated at the source.

## What remains open

- Issue #158's substance (a measured, non-zero `per_contig_mb` fitted the
  right way, as an envelope) is closed by this refit.
- The per-contig term's growth with cohort width (check 4) and the
  `n_chunks` term (check 3) both point at the same open research question
  `RamLaw::PGEN`'s write-up already flagged: the law bounds correctly but
  does not describe the mechanism. That is explicitly out of scope for #158
  (see the design doc's scope table, item C) and should stay a separate
  issue rather than blocking this release.
- `ProbeRecord.concurrent_chroms_used` telemetry is still not populated by
  the `FieldGrab` tracing visitor — harmless for this sweep (every point
  pinned `cc`), but a point that does not pin it remains unfittable.

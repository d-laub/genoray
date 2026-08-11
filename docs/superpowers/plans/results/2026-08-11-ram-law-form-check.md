# Deciding the RAM-law functional form offline — interaction term GO/NO-GO

**Zero cluster cost.** Everything below is computed from the already-committed
`2026-08-08-pgen-ram-law-crossed-data/` (58 rows, 46 with `concurrent_chroms`
observed) — no Slurm job was submitted and no corpus was generated for this
task. The point of doing it now, against PGEN, is that the VCF sweep (#158's
remaining scope) should be fitted against whichever functional form wins this
check exactly once, not once per backend.

## Why this interaction, specifically

The 2026-08-08 crossed sweep measured the per-contig slope growing with cohort
width instead of holding constant:

| S | per-contig MB |
|---:|---:|
| 4,000 | 83.7 |
| 32,000 | 263 |
| 128,000 | 301 |

A single additive `per_contig_mb` cannot express a slope that itself scales
with `S`. The natural next term is `per_contig_per_sample_mb`, entering the
envelope as `cc * samples`, on top of the existing `cc * per_contig_mb`
bracket.

## The pre-registered decision rule

Fixed **before** either fit was run (this document did not exist until after
Step 6 numbers were in hand, but the rule itself was written into the task
brief beforehand and is reproduced here unedited):

Adopt the interaction form **only if both** hold:

1. worst-case ratio (`RamLaw.worst_ratio`, the LP's own objective `t` for that
   functional form) improves by **≥ 20%** against the current form, on the
   same 46 rows; **and**
2. **no** coefficient needs to be extrapolated more than **~2×** beyond its
   measured domain. The measured cohort domain is S ∈ {4,000, 32,000,
   128,000} against a production target of S = 500,000 — a term multiplying S
   alone is therefore already ~3.9× extrapolated (this is the existing,
   already-shipped `per_sample_mb` term, unchanged by this check), and an
   interaction term multiplying `cc * S` compounds that reach further.

This is the same lesson that sank the `n_chunks` term: it reached R² 1.0000
in-sample and was still refused, because applying it 300× beyond its 32–128
measured chunk-count range moved the S=500,000 projection from 65.3 GiB to
160.7 GiB. In-sample fit quality is not evidence about extrapolation, and
`r2` is never the shipping criterion for a law that must be an upper bound —
see `fit_ram_law`'s docstring.

## Both fits, in full

Reproduced with:

```bash
D=docs/superpowers/plans/results/2026-08-08-pgen-ram-law-crossed-data
pixi run python -m scripts.bench_svar2.fit_ram \
  --results $D --plans $D --manifests $D/manifests --backend pgen --margin 1.25
pixi run python -m scripts.bench_svar2.fit_ram \
  --results $D --plans $D --manifests $D/manifests --backend pgen --margin 1.25 --interaction
```

`n=46` cc-observed rows for both forms (`58` raw records, `12` dropped for
missing `concurrent_chroms`, same subset as the 2026-08-08 PGEN refit).

### Current form (4 coefficients, `interaction=False`)

This reproduces Task 4's baseline exactly, confirming the fitter has not
drifted:

| coefficient | value |
|---|---:|
| `base_mb` | 2696.785976670047 |
| `per_sample_mb` | 0.01575147162905773 |
| `per_contig_mb` | 209.8696589690541 |
| `kappa` | 2.3847735782388906 |
| `per_contig_per_sample_mb` | 0.0 (field not used by this form) |

Gate (`fit_ram.gate_report`, which mirrors `budget.rs::plan_sharded`'s
current four-term equation via `predict_mb`): `n=46`, `passes=True`,
`worst_ratio=2.4189x`, `mean_ratio=1.8816x`, `min_ratio=1.2500x`.

The LP's own objective (`RamLaw.worst_ratio`, computed against exactly the
design matrix that was fitted — the quantity the decision rule below actually
compares) agrees: **2.4189269852595365×**.

### Interaction form (5 coefficients, `interaction=True`)

| coefficient | value |
|---|---:|
| `base_mb` | 2649.1326867700354 |
| `per_sample_mb` | 0.015505046189608806 |
| `per_contig_mb` | 206.88460972667204 |
| `kappa` | 1.7125348772321451 |
| `per_contig_per_sample_mb` | 0.0010523669611920323 |

Gate via `fit_ram.gate_report`/`predict_mb`: `n=46`, `passes=True`,
`worst_ratio=2.3729x`, `mean_ratio=1.7813x`, `min_ratio=1.2106x`.

**Caveat on that gate number:** `predict_mb` (Task 4) is a deliberate,
exact mirror of `budget.rs::plan_sharded` *as it exists today* — a
four-term equation. It does not know about `per_contig_per_sample_mb`, so
`gate_report`'s ratios for the interaction form silently drop that term's
contribution. The number that is actually comparable to the current form's
LP objective, because it is computed against the same design matrix the LP
optimized, is `RamLaw.worst_ratio` itself:

**`worst_ratio = 2.376295448067797×`.**

(The tiny gap between 2.3729x and 2.3763x — both far smaller than any
scale relevant to the decision below — is exactly that dropped-term effect,
not a bug: it shows up because `per_contig_per_sample_mb` is a genuinely
small coefficient here.)

`test_fit_ram_law_with_interaction_is_never_looser_than_without`
(`tests/bench/test_model.py`) checks the nesting property this implies —
`inter.worst_ratio <= plain.worst_ratio` — directly, and it passes.

## Applying the decision rule

### Clause 1 — worst-case improvement ≥ 20%

```
plain.worst_ratio = 2.4189269852595365
inter.worst_ratio = 2.376295448067797
improvement       = (2.4189269852595365 - 2.376295448067797) / 2.4189269852595365
                  = 0.042631537191740 / 2.418926985259537
                  = 1.76%
```

**1.76% < 20%. Clause 1 FAILS**, decisively — not a borderline call.

### Clause 2 — no coefficient extrapolated more than ~2× beyond its measured domain

Computed for completeness, since clause 1 alone already determines the
verdict:

The new regressor is `cc * samples`. Its value across the 46 fitted rows
ranges up to 2,048,000 (at `cc=16, S=128,000` — one of the bench-only
lever-arm points swept specifically to identify `per_contig_mb`, which sits
outside PGEN's production concurrency clamp, `PGEN_MAX_CONCURRENT = 8` in
`src/budget.rs`). Applied at the production target (`S=500,000`, `cc=8`, the
largest concurrency the planner can ever request):

```
applied product = 500,000 * 8 = 4,000,000
```

- Against the full measured range (including the bench-only `cc=16` lever-arm
  points): `4,000,000 / 2,048,000 = 1.95×` — inside the ~2× bound.
- Against only the rows inside PGEN's production concurrency clamp
  (`cc <= 8`, max measured product `500,000`-scale row `cc=8, S=128,000` →
  `1,024,000`): `4,000,000 / 1,024,000 = 3.91×` — outside the ~2× bound, and
  essentially identical to the pre-existing `per_sample_mb` extrapolation
  (`500,000 / 128,000 = 3.91×`), since it is driven by the same `S` reach.

These two readings disagree about whether clause 2 passes, because the brief
does not pin down whether "measured domain" for a coefficient may include
bench-only points taken outside the value's own production range. **This
task does not need to resolve that ambiguity**: clause 1 already fails on its
own, so the rule's overall verdict is unaffected either way. It is recorded
here rather than silently resolved in whichever direction happens to support
the numbers, per the pre-registration.

## Verdict: NO-GO

Clause 1 fails (1.76% improvement, gate is ≥20%); clause 2 is failed on one
reasonable reading and passed on another, moot given clause 1. **The
interaction form is not adopted.**

**Follow-up issue sentence:** *"`per_contig_per_sample_mb` was fitted and
gated against the 2026-08-08 crossed PGEN data (worst-case ratio 2.4189x →
2.3763x, a 1.76% improvement) but rejected by the pre-registered ≥20%
worst-case-improvement gate; the per-contig slope's growth with cohort width
remains unexplained and #158 stays open for the four-coefficient envelope."*

## What this means for Task 8 / the VCF sweep

Per Step 8 of the task brief: since the verdict is NO-GO,
`interaction` stays in `fit_ram_law` as a tested, dormant option (default
`False`), and `RamLaw.per_contig_per_sample_mb` stays a `0.0`-defaulted
Python field — but neither is added to `src/budget.rs::RamLaw`. Task 8 ships
the four-coefficient form, and the upcoming VCF crossed sweep should be
fitted with `interaction=False` (the default), since this check has now
settled which functional form both backends use.

## Provenance note

The committed `2026-08-08-pgen-ram-law-crossed-data/` corpora were generated
by a **pre-v0.5.0** `vcfixture` (their manifests record the un-migrated
profile hash, and v0.5.0 gave variant positions their own PRNG stream). That
does not affect this task's validity — both fits are computed against the
exact same 46 rows, so the comparison is internally consistent regardless of
corpus generator version. It would matter if this verdict were being
transferred to the VCF backend's data (a different generator, `scale_corpus.py`,
not `vcfixture`) — it is not: this check is PGEN-only, deciding a functional
*form* to reuse, not a set of coefficients to reuse.

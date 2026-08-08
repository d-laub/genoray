# Crossed PGEN RAM sweep — resolving the RamLaw specification error

Sweep job **13351716**, `carter-cn-04`, COMPLETED `0:0`, 4h10m, **58/58 points
measured** (guard: 58 plan points / 58 result rows / 58 unique ids, single
node). Results tag `crossed`, byte-distinct from the prior file — the #159
staleness trap was avoided, not fixed.

Data: `2026-08-08-pgen-ram-law-crossed-data/`. Design and acceptance criteria
were pre-registered on issue #158 before any of it existed.

## Why this sweep was run

The 2026-08-07 refit reached R² 0.7698. That is not stochasticity. The cc
ladder — six launches differing only in `concurrent_chroms` — fits cc alone at
**R² 0.9903, RMSE 63 MB**, so peak RSS here is reproducible to a few percent.
Fitted on the block where `RamLaw`'s form has every regressor it needs, the
residual RMSE was still 463 MB, **7.3× that floor**: ~98% specification error.

## What the crossed data shows

**1. The per-contig term is real, and it is not what the analytic constant
says.** Measured at each `(S, chunk_size)` block:

| S | chunk_size | per-contig MB | 95% CI |
|---:|---:|---:|---|
| 4,000 | 3,125 | 88.83 | ± 46.67 |
| 4,000 | 12,500 | 88.32 | ± 22.27 |
| 4,000 | 25,000 | 83.73 | ± 11.62 |
| 32,000 | 25,000 | 263.43 | ± 161.74 |
| 128,000 | 7,812 | 300.52 | ± 320.39 |

The S=4,000 values reproduce the earlier ladder's 89.67 MB independently. The
growth with cohort width is visible but the CIs at large S are too wide to
call it identified.

**2. The `n_chunks` term is real and extremely well determined.** Axis A —
`chunk_size` pinned at 7,812, V varied so `chunk_bytes` is held *exactly*
constant — was the lever the old design structurally could not supply:

| S | cc | MB per chunk | R² |
|---:|---:|---:|---:|
| 4,000 | 1 | 4.372 ± 2.686 | 0.9977 |
| 4,000 | 8 | 6.294 ± 12.933 | 0.9745 |
| 32,000 | 1 | 8.272 ± 0.101 | **1.0000** |
| 32,000 | 8 | 13.135 ± 3.619 | 0.9995 |

Pre-registered check 3 is satisfied: the term survives when `n_chunks` moves
independently of `chunk_bytes`, so it is not a reparameterisation of `kappa`.

**3. But no additive form passes the gate.** All 47 linear combinations of the
candidate regressors were fitted by OLS and evaluated with slopes at their 95%
upper bounds (construction D). The best still under-predicts 3 of 46 points,
and residual σ stays 8.8–10.7× the noise floor.

Near-perfect within-block fits (R² up to 1.0000) alongside a pooled σ of 555 MB
is the signature of **interaction**, not of a missing additive term. Both new
coefficients vary with `S` and with `cc`.

## The actual defect was the fitting method

`RamLaw` is not a prediction of RSS — `plan_sharded` uses it as an upper bound,
and the gate is over-prediction everywhere. Fitting by OLS and padding to CI
upper bounds optimises squared error and then pads, so the resulting slack is
an accident of the residual spread. Stated directly as a linear program:

```
minimise  t
s.t.      y_i  <=  X_i · b  <=  t · y_i    for every measured point
          b    >=  0
```

`t` is the worst-case over-allocation, optimal by construction for that form.

| law | worst | mean |
|---|---:|---:|
| **shipped today** | **10.111×** | 3.050× |
| A: same form, envelope-fitted | 2.402× | 1.592× |
| B: + per-contig constant | 1.935× | 1.505× |
| C: + per-contig + per-chunk | 1.774× | 1.451× |

Changing only the *fitting method*, with no new fields and no API change, cuts
worst-case over-allocation **10.111× → 2.402×**.

**Option C is measurably tightest in-sample and must not be shipped.** Its
`n_chunks` term projects 40,000 chunks at V=1e9 — 300× beyond the 32–128 the
sweep measured — and the extrapolation dominates, taking the S=500,000
projection to 160.7 GiB against option B's 65.3 GiB. The ratchet must also
saturate physically, since `malloc_trim(0)` runs at each contig boundary. The
term is real; extrapolating it linearly is not justified.

## The margin becomes an explicit choice

An envelope fit removes the accidental margin but also removes *the* margin: it
touches the data at its binding points, and the law is applied 3.9× beyond the
largest measured cohort. So require every measured point to be over-predicted
by at least `m` and report the cost:

| margin | A worst | B worst |
|---:|---:|---:|
| 1.00 | 2.402× | 1.935× |
| 1.15 | 2.762× | 2.225× |
| **1.25** | **3.002×** | **2.419×** |
| 1.50 | 3.603× | 2.903× |
| 2.00 | 4.804× | 3.870× |

**Recommendation: option B at margin 1.25** —

```
base_mb        2696.8
per_sample_mb  0.015751
per_contig_mb  209.87     (new field)
kappa          2.3848
```

Worst case 2.419× against the shipped law's 10.111×: **4.2× less
over-allocation**, with a 25% safety factor that is chosen and stated rather
than inherited from a double-count. This supersedes the "margin is not slack to
be tuned away" framing of Task 9 only in *mechanism* — the margin is still not
slack, it is now simply explicit.

## What remains open

- The per-contig and per-chunk coefficients both vary with `S` and `cc`. The
  envelope fit is safe regardless, but the law still does not describe the
  mechanism, so σ stays ~9× the noise floor. #158 stays open.
- `ProbeRecord` still has no field for the realised `concurrent_chroms`; the 12
  planner's-choice rows remain unfittable and were excluded from every fit here.
- #159 (resume key omits code identity) is avoided by a fresh results tag and
  the new post-run guard, not fixed.

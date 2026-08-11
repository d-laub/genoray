# `RamLaw::VCF` envelope refit — closing #158 before the next release

**Date:** 2026-08-11
**Issue:** [#158](https://github.com/d-laub/genoray/issues/158)
**Status:** design approved, implementation plan pending

## Why now

`plan_sharded` and `RamLaw` do not exist at tag `3.4.0` — verified with
`git show 3.4.0:src/budget.rs`, which contains neither symbol. The entire
memory-budget planner sits in the unreleased range, so the next release is the
**first** one to put either law in front of users. `RamLaw::PGEN` was refitted
as an envelope in `c100d51`; `RamLaw::VCF` still carries the exact defect that
commit removed. Shipping now would introduce a known-defective law rather than
inherit one.

The VCF law's defect has one genuinely unsafe direction. The cc-blind fit makes
`kappa` absorb the training data's dominant `concurrent_chroms`, which
`plan_sharded` then multiplies by `cc` a second time — that double count
*over*-allocates, costing concurrency but never causing an OOM. The dangerous
half is `per_contig_mb = 0.0`: the ~128 MiB `ChunkAssembler` staging allocation
(`RAW_STAGE_BYTES + MASK_STAGE_BYTES`, `src/chunk_assembler.rs`) is charged per
live contig pipeline and is currently unpriced. At `cc = 8` that is roughly
1 GB unaccounted against a `base_mb` of 932 — an under-prediction, which is the
OOM direction.

## What closes #158

| | in scope | rationale |
|---|---|---|
| **A. `RamLaw::VCF` envelope refit** | yes | The substance of the issue: fit the law the way the consumer uses it, under the gate `RamLaw::PGEN` already passes. |
| **B. `ProbeRecord` records the realised `concurrent_chroms`** | yes | Named in the issue's closing comment. Today a point that does not pin `cc` is unfittable — 12 of the 58 PGEN crossed rows were dropped for this. |
| **C. "the law bounds but does not describe the mechanism"** (residual σ ≈ 9× the 63 MB reproducibility floor; coefficients interact with `S` and `cc`) | **no** — new issue | An upper-bound law does not need to describe the mechanism. Driving σ to the floor means interaction terms, which is a research question and not a release gate. |

One bounded exception to that split, taken in Phase 0 because it is free: the
committed 46-row PGEN crossed dataset can test *one* interaction offline,
before any cluster time is spent. See Phase 0.

## Key finding: the harness half is already done

`fit_ram_law` is already the envelope LP with an explicit `margin`, `RamRow`
already carries `concurrent_chroms`, and `_ram_rows` already drops rows where
`cc` was never observed. None of that machinery is PGEN-specific.

**The VCF law needs data, not new fitting code.** This design is therefore
mostly a measurement design.

## The measurement problem

VCF corpora are text, so the cell budget binds where PGEN's did not. PGEN's
crossed grid ran S=128,000 × V=250,000 = 3.2e10 cells; the VCF scale ladder
holds `S × V` at `CELLS_BUDGET = 1.4e9`, 23× smaller, which is what keeps
generation near an hour per corpus.

Per-variant chunk bytes are `samples * ploidy / 8` (`records.py:chunk_bytes`),
and the ≥1-chunk-per-contig invariant caps `chunk_size` at `V / n_contigs`.
So the largest chunk a constant-cells design can reach is

```
max chunk_MB = cells / (4 · n_contigs · 1e6)
             = 1.4e9 / (88 · 1e6)
             = 15.9 MB          ← independent of S
```

The cohort width cancels, which exposes the real constraint: **the `cc` lever
arm wants many contigs and the `chunk_MB` lever arm wants few, and at a fixed
cell budget they are in direct conflict.** A 4-contig corpus reaches 87.5 MB
but caps `cc` at 4.

This matters because production reaches far larger chunks — `_auto_chunk_size`
targets 256 MiB — so a sweep topping out at 15.9 MB would extrapolate `kappa`
~16× beyond anything measured. That is the same error the PGEN write-up
refused to commit when it declined the `n_chunks` term at 300×.

**Resolution (approved):** keep the 22-contig constant-cells grid for the `cc`
axis and add **one** oversized corpus purely to extend the `chunk_MB` lever
arm. This costs four extra points and one large corpus, and cuts `kappa`'s
extrapolation to production's 256 MiB chunks from ~16× to ~2.7×.

## Sweep design

Three plan families, all GT-only (no FORMAT/dosage fields), matching the PGEN
law's domain and issue #156.

| family | corpus | grid | points |
|---|---|---|---:|
| `VCF_CROSSED` | 22 contigs, 1.4e9 cells, S ∈ {4,000, 32,000, 128,000} | `chunk_size` ∈ {V/88, V/44, V/22} × `cc` ∈ {1, 4, 8, 16} | 36 |
| `VCF_NCHUNKS` | 22 contigs, S ∈ {4,000, 32,000}, V ∈ {½, 1, 2} × that width's crossed V | `chunk_size` pinned at `V_min/22` × `cc` ∈ {1, 8} | 12 |
| `VCF_BIGCHUNK` | 22 contigs, 1.1e10 cells, S = 32,000 (V = 343,750) | `chunk_size` ∈ {3,125, 12,500} (≈25, 100 MB) × `cc` ∈ {1, 8} | 4 |

Expressing `VCF_CROSSED`'s chunk sizes as fractions of V rather than as literal
variant counts is what makes the grid uniform: since `chunk_MB` is
`(S/4) · chunk_size` and V is `1.4e9 / S`, those three rungs land on ≈4, 8 and
16 MB at **every** cohort width, and the largest sits exactly on the
`V / n_contigs` cap. `VCF_NCHUNKS` pins `chunk_size` at the smallest rung its
own V ladder permits, so `chunk_bytes` is held exactly constant while the chunk
count moves 4×.

`VCF_CROSSED` identifies `base_mb`, `per_sample_mb`, `per_contig_mb` and
`kappa`, and tests additivity of the per-contig term across three cohort widths
(the issue's pre-registered check 4). `VCF_NCHUNKS` is the orthogonal lever
that decides whether an `n_chunks` term is real or merely a
reparameterisation of `kappa` (check 3): pinning `chunk_size` and varying V
moves the chunk count with `chunk_bytes` held exactly constant.
`VCF_BIGCHUNK` exists only to give `kappa` a lever arm out to ~100 MB.

`cc = 16` exceeds any production clamp and is reachable only through the
bench-only `GENORAY_CONCURRENT_CHROMS` override; it is included for lever arm
and must be flagged as outside the production domain, exactly as the PGEN law
flags its own `cc = 16` rows.

### Required new assertion: `chunk_size ≤ V / n_contigs`

`model.py:_resident_chunk_size` clamps `chunk_size` by **total** V, not by
per-contig V. On a 22-contig corpus a point whose `chunk_size` exceeds
`V / 22` would therefore be fitted against a chunk up to 22× larger than
anything that is ever resident, because `BitGrid3::zeros` is a `calloc` whose
untouched pages never become resident. PGEN's grid never tripped this; a
constant-cells VCF grid at S=128,000 (per-contig V ≈ 497) trips it
immediately. `build_plans.py` must assert this at plan-build time.

## Corpus generation

`scale_corpus.py` does not use vcfixture — it is numpy plus a
`ProcessPoolExecutor` and a bgzip subprocess. `pgen_corpus.py` is the module
that shells to the `vcfixture bulk` CLI. Moving RAM-law corpus generation onto
vcfixture makes the VCF corpus the PGEN pipeline stopped one step early:

```
vcfixture bulk --profile germline-1kgp-varskew.json  →  corpus.vcf.gz   ← VCF sweep
                                                     →  plink2 --make-pgen  ← PGEN sweep
```

Use **vcfixture-rs v0.5.0 or later**. v0.5.0 (2026-08-07) closed
[vcfixture-rs#22](https://github.com/d-laub/vcfixture-rs/issues/22), the
block-parallel encoder fix for `bulk` sitting at 120% CPU with ~46 cores idle.

Three consequences:

1. **The two laws become comparable for the first time.** `RamLaw::PGEN`'s doc
   comment currently ends with "NOT comparable coefficient-by-coefficient with
   `RamLaw::VCF`: the two corpora come from different generators." Sharing a
   generator retires that caveat and allows `per_contig_mb` to be cross-checked
   across backends — a free consistency check on a term neither sweep
   identifies precisely.
2. **`scale_corpus.py` stays.** It owns the hold-out, both V-linearity ladders,
   the FORMAT-field corpora and `size_corpus`'s chunk-size derivation, all of
   which the cost and V laws still use. This work adds a vcfixture path for the
   RAM-law corpora only. Migrating the rest is a follow-up, not this change.
3. **`CorpusManifest.generator_version` must record the CLI version string**,
   not a bumped integer. v0.5.0 is an explicit breaking output change —
   "generated output for a given seed differs from v0.4.0 … existing corpora
   must be regenerated" — and the PGEN corpora on `/local` predate it. A
   manifest that cannot distinguish the two invites silently pooling
   incompatible corpora.

`vcfixture bulk` is expected to emit `.vcf.gz` directly; the plan verifies this
with a `--help` check before depending on it, and falls back to
`bcftools view -Oz` if not. If the sweep ends up running on BCF instead, 2–3
crossed points must be re-measured on a bgzf-VCF copy of the same corpus,
asserting `maxrss` agrees within the 63 MB reproducibility floor — BCF input is
known to delete 23–41% of reader CPU, and RSS parity should be demonstrated
rather than assumed.

The CLI is resolved through `pgen_corpus.resolve_vcfixture` (`VCFIXTURE_BIN`,
then `PATH`), which raises an actionable error rather than failing obscurely in
CI. It is a Rust binary, separate from the PyPI `vcfixture` package pinned in
`pixi.toml` as `>=0.6.0,<0.7` — those are two independent version lines.

## Phases

### Phase 0 — functional-form check (offline, no cluster time)

The PGEN crossed dataset is committed at
`docs/superpowers/plans/results/2026-08-08-pgen-ram-law-crossed-data/`, so one
interaction can be tested before any node is booked. The data already points at
which one: the per-contig term grew 83.7 → 263 → 301 MB across S = 4,000 /
32,000 / 128,000, i.e. a bracket that scales with cohort width.

```
current:   base + per_sample·S + cc·(per_contig + kappa·(w+pending)·chunk_MB)
candidate: base + per_sample·S + cc·(per_contig + per_contig_per_sample·S
                                     + kappa·(w+pending)·chunk_MB)
```

Both fitted with the same LP at margin 1.25 and compared on worst-case `t`.

**Pre-registered decision rule** (fixed before the numbers are seen, so it
cannot be rationalised afterwards): adopt the candidate only if it

1. cuts worst-case `t` by **≥20%** on the PGEN data, **and**
2. requires **no** coefficient to be extrapolated more than ~2× beyond its
   measured domain.

Clause 2 is the `n_chunks` lesson: that term reached R² 1.0000 in-sample and
was still correctly refused, because applying it at 40,000 chunks — 300×
beyond the 32–128 measured — took the S=500,000 projection from 65.3 GiB to
160.7 GiB. A well-measured local coefficient is not licensed for a large
extrapolation.

If the candidate wins, `RamLaw` gains a fifth field, `plan_sharded`'s bracket
changes, and `RamLaw::PGEN` refits from the committed data at no measurement
cost; both laws then ship together. If it loses, the functional form is
unchanged and item C spins out to its own issue.

Phase 0 runs first specifically so the VCF sweep is fitted against the final
functional form once rather than twice.

### Phase 1 — code (no cluster time)

- `ProbeRecord.concurrent_chroms_used`, populated by parsing the child's plan
  diagnostic (the `using cores` tracing line restored on both backends in
  `1cf3d0c`). If that proves fragile, the fallback is a structured event over
  the existing monitor/tracing channel rather than a looser regex. `_ram_rows`
  prefers the pinned `SweepPoint.concurrent_chroms`, falls back to the realised
  value, and drops a row only when neither exists. The field is defaulted so
  records written before it existed still load, matching how `node` was added.
- vcfixture-backed VCF corpus generation, sharing `pgen_corpus.py`'s profile
  and CLI-resolution machinery.
- `VCF_CROSSED` / `VCF_NCHUNKS` / `VCF_BIGCHUNK` plan families, plus the
  `chunk_size ≤ V / n_contigs` assertion.
- `sweep_scale.sbatch` hygiene, all of which is currently broken or stale:
  - `JD="${CLAUDE_JOB_DIR:?}"/tmp` on line 21 is the dangling-symlink trap that
    killed job 13351680 after 6h57m with every corpus already generated.
    `sweep_pgen.sbatch` received the `unset CLAUDE_JOB_DIR` + `/local/$USER`
    fix in `59665aa`; this driver never did.
  - `WT` hardcodes `.claude/worktrees/bench-pr140-reader-workers`. Derive it
    from `git rev-parse --show-toplevel`.
  - Pin `--nodelist`. Measured node-to-node spread on identical work is 2.08×,
    and unpinned rows have already produced retracted findings.

### Phase 2 — measurement (one sbatch, pinned node)

Corpora, then sweep. A `df -h` headroom check runs first, as in
`sweep_pgen.sbatch`. Size estimate: the big-chunk corpus is roughly 2 GB of
bgzf VCF at 1.1e10 cells GT-only and the three standard corpora about 0.3 GB
each, against 392 GB free on `/local` — the check costs nothing and catches a
wrong estimate before six hours of generation rather than after.

The post-run guard asserts plan point count == result rows == unique point ids
before any fit runs. Every row records its node. Estimated 6–10 h of
measurement on top of generation, inside the partition's 72 h limit.

### Phase 3 — fit and ship

`fit_ram_law(rows, margin=1.25)` on the VCF rows, then the same gate PGEN
passed, reported in full:

- over-predicts **every** measured point, evaluated the way `plan_sharded`
  evaluates it, at each row's actual `concurrent_chroms`;
- worst / mean / min over-allocation ratios;
- additivity of `per_contig_mb` across the three cohort widths — if it varies
  beyond its CI, say so rather than pooling;
- residual σ against the 63 MB reproducibility floor.

Ship into `RamLaw::VCF` with the doc-comment contract this repo already
enforces: gate result, `n`, and validity domain recorded in the constant
itself. Add `ram_law_vcf_is_a_usable_law` mirroring the PGEN test, guarding
`per_contig_mb > 0` so a later refit that silently drops the term fails loudly.
Sweep data and a results doc land under `docs/superpowers/plans/results/`
alongside the three PGEN ones.

**The gate cannot dead-end.** The LP is feasible by construction; only the
tightness of `t` is in question. If `t` exceeds ~4×, ship it with the number
stated — a law fitted the right way and loose is strictly better than a law
fitted the wrong way — and open an issue rather than blocking the release on a
research problem.

### Phase 4 — release

`RamLaw` is Rust-internal, but the refit moves the documented `max_mem` floors,
so `skills/genoray-api/SKILL.md`'s "~932 MB VCF baseline / ~3 GB PGEN" sentence
must be updated in the same PR — this repo's rule is that public-facing text
tracks these changes. Close #158, open the follow-up issue for item C if
Phase 0 did not resolve it, then let the release workflow's `commitizen` cut
the version. Do not edit `CHANGELOG.md` or bump the version by hand.

## Verification

- `cargo test --no-default-features --features conversion` — 473 passing
  today. Bare `--no-default-features` silently skips the whole conversion path,
  and dropping `--no-default-features` fails to link the pyo3 test binary.
- `pixi run test`.
- `tests/bench/test_model.py` covers the new `cc` field and, if Phase 0 adopts
  it, the new functional form.
- `maturin develop --release` before any Python-level verification:
  `pixi run test` does **not** rebuild the Rust extension, so Python tests can
  otherwise run against a stale `.so`.
- `CARGO_TARGET_DIR` must point off NFS or the linker bus-errors.

## Risks

| risk | mitigation |
|---|---|
| Sweep rows contaminated by node contention | `--nodelist` pinned, node recorded per row, row-count guard before fitting. |
| Corpus regeneration invalidated by a vcfixture version change | CLI version string recorded in every manifest; v0.4.0 and v0.5.0 corpora never pooled. |
| `t` comes out loose | Ship with the number stated; the law is still strictly better than the current one. Open an issue. |
| Phase 0 tempts a form change that wins in-sample only | Decision rule pre-registered above, including the extrapolation clause. |
| Big-chunk corpus larger than estimated | `df -h` headroom check before generation. |

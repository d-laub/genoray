# Budget-aware conversion planning for `from_pgen`

**Date:** 2026-08-05
**Status:** design approved, implementation plan pending

Follows `2026-08-03-svar2-tuned-load-balancing-design.md` (merged as PR #141),
which gave the sharded VCF conversion path a concurrency planner bounded by both
cores and memory. That work stopped at `run_conversion_pipeline`. This spec
extends the same planner to `run_pgen_conversion_pipeline`, with PGEN's own
fitted RAM law rather than the VCF one.

## Problem

`run_pgen_conversion_pipeline` (`src/lib.rs:419`) still plans with
`budget::plan_thread_budget(available_cores, jobs.len())` — the pre-#141
planner. Four gaps follow, all confirmed against `origin/main` at `0000cb2`.

### 1. No memory bound at all

`plan_thread_budget` takes two integers, neither of which is a byte budget.
`from_pgen` has no `max_mem` parameter to supply one. This is exactly the
biobank-scale OOM exposure PR #141 removed from `from_vcf` and left in place
here: concurrency is chosen without reference to how much memory the
concurrent contigs will actually hold.

### 2. The core model charges for threads PGEN never allocates

`plan_thread_budget` charges `MIN_THREADS_PER_CHROM = 6` cores per contig:
`PIPELINE_THREADS_PER_CHROM = 4` plus `MIN_HTSLIB_THREADS = 2`. On the PGEN
path:

- There is **no HTSlib decode pool**. `SourceSpec::Pgen` reads through
  `pgenlib`; `htslib_threads` is computed and then discarded.
- Of the four pipeline threads only the executor is CPU-bound. The same
  measurement that motivated `plan_sharded` applies unchanged — a 22-contig run
  put 16 pipeline threads on 2.02 cores.

PGEN's real per-contig demand is **1 executor + 1 pgenlib reader = 2 cores**,
because `from_pgen` pins `P = 1` (one reader per contig, sub-contig sharding
disabled). On a 48-core box with 22 contigs the current planner returns
`cc = 7, processing_threads = 5`, reserving ~6 cores per contig for something
that consumes ~2.

### 3. Dispatch order is arbitrary

The VCF path needed a whole `contig_cost.rs` tier system — a tabix probe, a CSI
probe, a header-length fallback, and an FFI out-of-bounds fix — to estimate
per-contig work. PGEN needs none of it: `contig_ranges[i] = (lo, hi)` is already
an **exact** variant count, computed in Python by `_pvar_contig_ranges` before
the pipeline is entered. Longest-first dispatch is therefore free here, and
currently unused: `jobs` is dispatched in `.pvar` order.

### 4. `processing_threads` is inconsistent with the dispatched `cc`

On both paths `processing_threads` comes from `plan_thread_budget`, computed
against *that function's* `concurrent_chroms` — which, since #141, is not the
concurrency the VCF path actually dispatches. PGEN will inherit the same
mismatch the moment its `cc` starts coming from `plan_sharded`.

This matters more than it did before #141, because `processing_threads` is what
sizes the **merge tail**: `merge.rs:250`'s dedicated pool for the var_key
gathers and `dense_merge.rs`'s `threads` for the dense bit-transpose. Both
became genuinely parallel in #141 (commits `8625f70`, `e31d815`), and both run
in `process_chromosome` for every backend — so they are live on the PGEN path
today. Sizing them off a stale `cc` wastes that win.

## Goal

`from_pgen` plans contig concurrency under **both** a core bound and a fitted
memory budget, dispatches longest-first, and sizes its merge tail against the
concurrency it actually runs — with `RamLaw::PGEN` constants produced by a fit,
not inherited from the VCF fit.

Non-goal: making PGEN conversion fast by any other means. `P = 1` stays.

## Design

### `src/budget.rs` — parameterize the planner by backend

Extract the three fitted peak-RSS coefficients into a struct:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamLaw {
    pub base_mb: f64,
    pub per_sample_mb: f64,
    pub kappa: f64,
}

impl RamLaw {
    /// Fitted 2026-08-03 against the sharded VCF path (R^2 = 0.9040, n = 44).
    pub const VCF: RamLaw = RamLaw { base_mb: 932.0, per_sample_mb: 0.01115, kappa: 1.371 };
    pub const PGEN: RamLaw = RamLaw { /* three coefficients + R^2 + n from the fit */ };
}
```

The three `RamLaw::PGEN` coefficients are **an output of the Measurement phase
below, not a value this spec supplies**. They are the one thing here that cannot
be derived by reading code, and the planner must not be wired up default-on
against provisional numbers. The fit's R² and n go in the doc comment alongside
them, exactly as `RamLaw::VCF` records its own.

`PlanInputs` gains `pub ram: RamLaw`. `plan_sharded`'s body is otherwise
unchanged — it already reads the three coefficients in exactly the three places
the struct replaces.

PGEN passes `reader_workers: 1`. This is not a special case being smuggled in:
`plan_sharded`'s per-contig demand is documented as `1 + reader_workers` (one
executor plus the readers), and PGEN's single `pgenlib` reader thread *is*
`w = 1`. With `w = 1` the reorder-buffer floor `pending = w - 1` is 0, which is
also structurally correct — a single reader has nothing buffered ahead of the
head. No new concept is introduced.

Add a shared helper so both backends size the merge tail identically:

```rust
/// Cores left after the planned concurrency's executors and readers.
/// Sizes the merge tail (`merge.rs`'s gather pool, `dense_merge`'s transpose).
pub fn processing_threads_for(usable_cores: usize, cc: usize, w: usize) -> usize {
    usable_cores.saturating_sub(cc * (1 + w)).max(1)
}
```

Both `run_conversion_pipeline` and `run_pgen_conversion_pipeline` switch to it.
This **changes `from_vcf`'s `processing_threads`** — a deliberate, tested change,
not a silent side effect. It is in scope because leaving it would mean two
backends sizing the same merge tail two different ways for no reason.

`plan_thread_budget` is retained unchanged for the monolithic VCF fallback,
which still allocates a real HTSlib pool. Issue #152 (that fallback planning at
the sharded concurrency) is orthogonal and stays out of scope.

### `src/contig_cost.rs` — an exact-count constructor

```rust
impl ContigCosts {
    /// Costs known exactly without probing an index. Used by the PGEN path,
    /// where `.pvar` variant-index ranges ARE the per-contig record counts.
    pub fn exact(values: HashMap<String, u64>) -> Self {
        ContigCosts { values, exact_counts: true }
    }
}
```

`order_longest_first` is already a free function over `&HashMap<String, u64>`
and is reused unchanged. No htslib is linked or called on this path — which also
means none of `contig_cost.rs`'s index-tier hazards (tabix-vs-header id spaces,
`hts_idx_get_stat` bounds) are reachable from PGEN.

### `src/lib.rs::run_pgen_conversion_pipeline`

Add `max_mem_bytes: Option<u64>` to the signature and `#[pyo3(signature = ...)]`,
positioned to match `run_conversion_pipeline`'s convention (after
`regions_overlap`/`sample_perm`, before `log_level`).

Inside the `py.detach` block, replacing the `plan_thread_budget` call:

1. **Costs.** Build `ContigCosts::exact` from `chroms.iter().zip(&contig_ranges)`
   with value `(hi - lo) as u64`.
2. **Per-variant bytes.** `n_samples * ploidy / 8 + n_dosage * n_samples * 4`,
   with `ploidy = 2` and `n_dosage = fields.len()` — PGEN's dosage fields are
   FORMAT-category by construction, so no category filter is needed (unlike the
   VCF path, which filters INFO out).
3. **Resident chunk size.** `min(chunk_size, max contig count)`. The VCF path
   guards this with `if costs.exact_counts`, because its header-length fallback
   tier yields base pairs rather than record counts. Here counts are exact by
   construction, so the guard collapses to the narrowing branch. A comment
   records *why* the guard is absent, so a later reader does not "restore" it.
4. **Plan.** `plan_sharded(PlanInputs { usable_cores: available_cores - 1,
   n_contigs: chroms.len(), n_samples: samples.len(), chunk_bytes,
   max_mem_bytes, reader_workers: 1, ram: RamLaw::PGEN })`, mapping
   `PlanError` through `ConversionError` exactly as the VCF path does.
5. **Dispatch.** Size the rayon pool at `bench_concurrent_chroms(cc)` and sort
   **the `jobs` vector itself** longest-first — not a separate name list. Each
   job tuple carries its contig's `readers` pool and `dosage_readers` pool, so
   reordering names alone would mis-pair readers with contigs. `chroms` keeps
   its original order for `finalize_fields`/`write_meta`, since the store's
   on-disk contig order is part of its layout.
6. **Log.** Extend the existing `"pipeline config (PGEN)"` line with
   `reader_workers`, the planned `cc`, and `processing_threads`, matching the
   VCF line's fields so sweep parsing is uniform.

`results` comes back in dispatch order rather than `chroms` order. The sole
consumer sums every entry unconditionally, so this is safe — the VCF path
carries the same property and the same warning comment; the PGEN comment will
say so too.

### Python `from_pgen`

Add `max_mem: int | str | None = None`, mirroring `from_vcf` exactly:

- `None` → `detect_memory_budget()` (cgroup-first, 80% of the limit). Detection
  failure warns and degrades to core-bound planning rather than raising.
- A string like `"64GiB"` → `parse_memory`.
- Threaded to Rust as `max_mem_bytes`.

**`max_mem` here means what it means in `from_vcf`, not in `from_vcf_list`.**
`from_pgen` now has a real concurrency planner to spend the budget on, so the
budget buys concurrency and `chunk_size` keeps its independent
`_DENSE_CHUNK_TARGET_BYTES` default via `_auto_chunk_size(..., max_mem=None)`.
`from_vcf_list` derives `chunk_size` from `max_mem` *because* it has no planner
(its contigs run strictly sequentially). That divergence already required a
runtime-explaining fix once in #141 (`3cf4b92`); the `from_pgen` docstring
states which of the two it follows, in those terms.

This is a public-API change: `skills/genoray-api/SKILL.md` is updated in the
same PR, per CLAUDE.md.

## Measurement

`RamLaw::PGEN` must come from a fit. `budget.rs` states the reason in its own
comment: *"These are load-bearing in production, not just in the bench: a bad
refit becomes an OOM."* A kappa fitted too **low** under-estimates per-contig
memory, over-schedules concurrency, and reintroduces precisely the failure the
budget exists to prevent.

The VCF fit cannot be assumed to transfer. The baselines are structurally
different:

- VCF's `per_sample_mb = 0.01115` is htslib/cyvcf2 per-sample buffering, which
  the PGEN path does not have.
- PGEN instead holds `n_contigs` `PgenReader`s — plus
  `n_contigs × n_dosage_fields` dosage readers — alive eagerly, constructed in
  Python before the pipeline starts, independent of `cc`.

### Corpus: `vcfixture bulk` → BCF → `plink2` → PGEN

The PGEN arm generates its corpus with the **`vcfixture-rs` `bulk` CLI**, not
with `scale_corpus.py`. `scale_corpus.py` hand-writes VCF text in Python and
pipes it through bgzip; `vcfixture bulk` streams from a `Profile` fitted against
real data, which is what "human-genome-like" requires. The
`germline-1kgp` profile is fitted on 3,202 samples / 73.6M variants and carries
per-contig variant counts and `density_per_kb`, a gap distribution, the site
frequency spectrum, variant-class mix, ti/tv, indel lengths, and missing/phased
rates. Its site-frequency spectrum is rescaled to whatever `--samples` is asked
for, so alt-allele density stays realistic at any cohort width.

The whole recipe below was smoke-tested end-to-end at 100 samples / 20k records
/ 3 contigs while writing this spec. Every claim in this section is an
observation from that run, not an expectation.

```
vcfixture bulk --profile germline-1kgp --payload gt-only \
  --contigs chrK --records N_K --seed <per-contig> --format bcf -o chrK.bcf   # per contig
bcftools concat -O b -o corpus.bcf chr1.bcf ... chr22.bcf
plink2 --bcf corpus.bcf --make-pgen --output-chr chrM --out corpus
```

Five findings that shape it, each verified:

1. **Apportion contigs ourselves; do not use a single `--records` run.**
   `--records` splits across contigs proportional to the profile's fitted
   *`density_per_kb`*, which is nearly flat across the autosomes: a 22-contig
   run measured **max/min = 1.38×**. The profile's own fitted `n_variants` are
   skewed **6.07×** (chr2 6.09M vs chr21 1.00M). An even split is precisely what
   `scale_corpus.py`'s existing comment warns hides the makespan tail — and
   makespan skew is what longest-first dispatch exists to exploit, so a flat
   corpus would make this spec's own feature unmeasurable. Apportion `N_K` from
   the profile's `n_variants`, invoke `bulk` once per contig, and `bcftools
   concat`. Generation is fast (220k records in 1.9 s), so per-contig invocation
   costs nothing.

2. **`plink2` strips the `chr` prefix by default — pass `--output-chr chrM`.**
   Without it the emitted `.pvar` is internally inconsistent: `##contig=<ID=chr1>`
   header lines are copied verbatim while data rows say `1`. `from_pgen` reads
   contigs from the data rows, so the store's contigs would silently become
   `1..22` while the source BCF's were `chr1..chr22`. Verified both ways.

3. **The profile emits symbolic ALTs and they survive into the `.pvar`.** The
   smoke corpus contained 24 `<DEL>` records; `plink2` passed all of them
   through. Convert with `skip_out_of_scope=True` — verified to return exactly
   24 — or filter them before `plink2`. The bench must use one, consistently,
   and record which.

4. **`--seed` makes output byte-identical regardless of thread count**, so a
   corpus is deterministic and cacheable. Cache the BCF *and* the derived
   `.pgen`/`.pvar`/`.psam`, keyed on
   `(profile, samples, per-contig records, contigs, seed)`.

5. **`--payload gt-only`** for the hardcall ladders. A dosage arm, if one is
   wanted, uses `gt-vaf` plus a `plink2` dosage import; it is optional, because
   `n_dosage` enters the law only through `chunk_bytes`, which the S/V ladders
   already vary.

**Binary resolution is a CI hazard, not a detail.** `plink2` and `bcftools` are
already declared in `pixi.toml` (lines 56 and 21) and resolve inside the env.
`vcfixture` does not. The `vcfixture` pinned in
`pixi.toml:67` is the *PyPI* package (0.6.0), which ships **no console script** —
verified: no `entry_points.txt`, no `bin/vcfixture` in the env. The `bulk` CLI
is a separate Rust binary (`cargo install vcfixture --features cli`). A script
that shells out to a bare `vcfixture` passes locally and fails in CI with
`FileNotFoundError`. So: resolve via `$VCFIXTURE_BIN`, then `shutil.which`, and
fail with an actionable install line; `skipif` any test that touches it on the
binary being resolvable. Corpus generation is bench-only and never runs in CI.

**Known coverage gap:** `germline-1kgp` has `multiallelic_rate = 0.0`, so this
corpus does not exercise the multiallelic `allele_idx_offsets` path. That path
is not what the RAM law models, so it does not invalidate the fit — but the
fitted law should not be claimed to cover multiallelic-heavy cohorts. Record
the limitation next to the coefficients.

### Harness work

- **Corpus module.** A `pgen_corpus.py` in `scripts/bench_svar2` implementing
  the recipe above with caching.
- **Plan family.** A `pgen` family in `plans/build_plans.py`. The ladders must
  include **two V-ladders at different S**: a design that holds S×V constant
  forces the cohort exponent to ≈1 arithmetically, which has already produced
  one confidently-wrong published interval in this project's history.
- **Fit.** Produce `RamLaw::PGEN` from the peak-RSS regression, reporting R² and
  n alongside the coefficients in the `budget.rs` doc comment, as `RamLaw::VCF`
  does.
- **Sweep.** A `cc` axis at fixed `w = 1` to locate the wall-clock knee.

Because the two corpora come from different generators, `RamLaw::PGEN` and
`RamLaw::VCF` are **not** cross-comparable coefficient-by-coefficient. That is
acceptable — each law is fitted on, and used for, its own backend. The spec
does not claim otherwise, and neither should the doc comment.

### Protocol

Per the lessons already recorded from #141's benching: pin `--nodelist` for any
cross-point comparison (node speed varies 2.08× on this cluster), take
best-of-3 after a page-cache warm-up rep, stamp the node in results, and never
mix rows across runs. Contaminated cross-run rows produced three
published-then-retracted findings during #141.

### Question for the fit, not for the design

Whether the eager reader pool needs an explicit `n_contigs` term in the law. A
`PgenReader`'s buffers are O(n_samples) bytes, so 22 of them should be tens of
MB against a multi-GB baseline — noise. **Add the term only if the residuals
say so.** Do not add it speculatively.

## Risk: the GIL

`pgenlib` 0.91 holds the GIL through `read_alleles_range`'s decode. With `cc`
concurrent contigs there are `cc` reader threads contending on one GIL. The core
bound permits `cc = 22` on a 48-core box, which is 3× today's `cc = 7`.

Two outcomes, and the sweep decides which:

- **PGEN conversion is executor-bound per contig.** This is what the chr21c
  measurement suggests — ~33 s bound by the shared executor/writer and reference
  I/O, *not* by pgenlib decode, and flat across `OMP_NUM_THREADS` 1→32. If so
  the executors parallelize cleanly, the GIL is a small serialized fraction, and
  the concurrency win is real.
- **PGEN conversion is GIL-bound above some `cc`.** Wall clock flattens or
  degrades well before `cc = 22`, exactly as the VCF `cc` curve turned
  non-monotonic past `cc ≈ 8`.

If the second holds, the response is a `PGEN_MAX_CONCURRENT` cap **justified by
the sweep and documented with its measurement**, not a guessed constant. Nothing
ships default-on before the sweep answers this.

Note that the memory bound is valuable regardless of which outcome holds: it
removes an unbounded failure mode whether or not the core bound ever binds.

## Sequencing

Two phases, in this order, because the second consumes the first's output:

1. **Measurement.** PGEN corpus generation, the `pgen` plan family, the RAM-law
   fit, the `cc` sweep. Lands with `RamLaw::PGEN` populated.
2. **Planner.** The `budget.rs` / `contig_cost.rs` / `lib.rs` / `from_pgen`
   changes, plus tests, wired against the fitted constants.

The refactor-only parts of phase 2 — extracting `RamLaw`, adding
`ContigCosts::exact` and `processing_threads_for` — have no dependency on the
fit and can be built in parallel with phase 1. Only the `RamLaw::PGEN` values
and the decision about a `PGEN_MAX_CONCURRENT` cap are gated on it.

## Correctness invariant

The store must be **byte-identical** regardless of `(concurrent_chroms,
dispatch order)`. This is the same invariant #141 gates for VCF, and it is what
makes longest-first reordering safe to adopt at all. Contig outputs are written
to per-contig directories and the cohort-level `meta.json` is written from
`chroms` in its original order, so the invariant should hold by construction —
the test exists to prove it does, not to assume it.

## Testing

- **`tests/test_svar2_pgen_schedule_invariance.py`** — mirrors
  `test_svar2_schedule_invariance.py`: convert one multi-contig PGEN fixture at
  several fixed `cc` values and assert a single distinct
  `_oracle.store_digest`. The `bench_concurrent_chroms` override already reaches
  the PGEN path (`lib.rs:520`), so no new test seam is needed. Follow the VCF
  test's precedent of asserting the fixture actually exercises what it claims
  (there, a non-empty long-allele bank) rather than passing vacuously.
- **Rust unit tests** in `budget.rs` — `plan_sharded` under `RamLaw::PGEN` at
  `w = 1`, including the `InsufficientMemory` boundary; `processing_threads_for`
  on both backends, including the `cc * (1 + w) >= usable` floor-to-1 case.
- **Python** — a too-small `max_mem` raises the actionable
  `InsufficientMemory` message rather than writing a partial store;
  `max_mem=None` under a cgroup plans against the detected budget.
- **Regression** — existing PGEN e2e tests stay byte-identical. Run
  `cargo test --no-default-features --features conversion` (the `--features
  conversion` half is required; a bare `--no-default-features` silently skips
  the conversion path) and `pixi run pytest tests/ -q -m "not network"`.
- **Core gate** — `cargo check --no-default-features`, the query-core build
  GenVarLoader links against.

## What this does not change

- `P = 1`. Sub-contig PGEN sharding stays disabled. It was settled by
  measurement and is not revisited here.
- The `pgenlib` pin (`==0.91.*`). A 0.94 bump was evaluated and rejected —
  decode is not the bottleneck.
- `.pvar` INFO extraction. Still unimplemented, still out of scope.
- The monolithic VCF fallback's planning (issue #152).
- `scale_corpus.py` and the VCF corpus it generates. The PGEN arm adds a second,
  independent corpus generator rather than retrofitting the first — the VCF RAM
  law is already fitted against `scale_corpus.py` output, and changing that
  generator would invalidate the fit currently shipping in production.
- `from_vcf`'s concurrency. Only its `processing_threads`, via the shared
  `processing_threads_for` helper, and that change is tested.

# SVAR2 `reroute` measurement spike

**Plan:** `docs/superpowers/plans/2026-07-12-svar2-concat-split-write-view.md` (Task 1)
**Question:** Is the *separate* `reroute=False` (source-representation-preserving)
implementation of `SparseVar2.write_view` ever worth building, or is
`reroute=True` (re-run the dense/var_key cost model on the subset) a permanent
sole mode?

## TL;DR

**Keep `reroute=False` raising `NotImplementedError`; `reroute=True` is the
correct sole mode.** Sample-subsetting *does* materially change representation
for small germline subsets (up to 17 % of variants flip; preserving the source
routing would be up to **+6.6 %** larger on disk), but the size delta is
**one-directional** — `reroute=True` picks the per-variant minimum, so it is
always ≤ the source-preserving output and is *meaningfully smaller* for realistic
germline sample subsets. A `reroute=False` (source-preserving) mode would
therefore produce a **strictly equal-or-larger** store: it buys nothing on output
size or correctness, only a potential conversion-speed shortcut. The genuine
performance follow-up is **making the existing `reroute=True` path streaming**
(see the eager-materialization benchmark,
`2026-07-13-svar2-eager-materialization-benchmark.md`), not a second
representation-preserving implementation.

## Background

`write_view` writes a region/sample subset of a finished SVAR2 store. Subsetting
**samples** changes each variant's carrier count `x`, which can change its
cheapest on-disk representation under the cost model (`choose_representation`,
`src/cost_model.rs`):

```
var_key = x · (POS_BITS + key_bits)             # per-carrier-call cost
dense   = POS_BITS + key_bits + n_samples·ploidy # table row + 1 bit/hap
route to dense iff dense < var_key (ties → var_key)
```

(no-field, no-signature stores ⇒ `sidecar = info = format = 0`.)

- **`reroute=True`** (built, sole mode): recompute `x'` for the subset and
  re-route each variant — size-optimal, byte-comparable to a fresh `from_vcf` on
  the same subset.
- **`reroute=False`** (raises `NotImplementedError`): would preserve each
  variant's *source-store* representation without re-running the cost model — a
  separate array-slicing path.

This spike measures, per subset size, **how often the subset-optimal
representation differs from the source representation (a "flip")** and **the
on-disk size penalty of preserving the source representation** (`size_preserve /
size_reroute − 1`, always ≥ 0 since reroute picks the per-variant minimum).

**Decision rule (from the spec).** If flips are rare *and* the size delta is
small (≈ <1–2 %) across subset sizes, `reroute=False` is redundant with
`reroute=True` and is **not worth building** — `NotImplementedError` becomes
permanent. If flips or the size delta are material, the two modes produce
genuinely different output and a `reroute=False` follow-up is warranted.

## Method

Analytic recount — **no store is rewritten**. `_core.svar2_variant_stats(store,
chrom, subset)` (added for this spike; `src/query/reader.rs::variant_stats` +
`src/lib.rs`) walks the finished sidecars with **no gather** and returns, per
variant: `is_indel`, `src_dense` (the source store's routing = which physical
stream it lives in), `x_full` (whole-cohort carrier haps), and `x_sub` (carrier
haps among the subset). A var_key call belongs to exactly one hap-column
(`sample·ploidy + p`), so grouping the packed calls by `(pos, key)` counts
carriers directly; a dense variant's carriers are a whole-cohort popcount of its
hap-major bit row.

`scripts/svar2_reroute_spike.py` then, for each subset size `k ∈ {10, 50, 100,
500}` (deterministic `numpy.random.default_rng(0)` sample subset), recomputes
each present variant's (`x_sub > 0`) subset-optimal representation from the exact
integer cost model — asserted against the documented crossovers (np = 2000: SNP
dense at `x ≥ 60`, indel dense at `x ≥ 33`) — and accumulates flip counts and the
summed on-disk bit cost under both routings. The full cohort (`k = n_samples`) is
included as a control: it must reproduce the source routing exactly (0 flips, 0
size delta, and `src_mismatch = 0`, i.e. the analytic recount of source routing
from `x_full` matches the physical stream).

### Stores (built on SLURM, `data/` — untracked)

`data/chr21.bcf` is 1000 Genomes (germline, 3202 samples); `data/gdc.chr21.bcf`
is GDC (somatic, 16007 samples). Both required `skip_out_of_scope=True` (symbolic
`<INS>`/breakend ALTs present) — noted here because the plan's commands omitted it:

```bash
# germline (3202 samples)
sbatch -p carter-compute -c 16 --mem=64G --wrap "cd <worktree> && pixi run -e py310 python -c \"import genoray; genoray.SparseVar2.from_vcf('data/chr21.germline.svar2','data/chr21.bcf','/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa',skip_out_of_scope=True,threads=16,overwrite=True)\""
# somatic (16007 samples)
sbatch -p carter-compute -c 16 --mem=128G --wrap "cd <worktree> && pixi run -e py310 python -c \"import genoray; genoray.SparseVar2.from_vcf('data/gdc.chr21.somatic.svar2','data/gdc.chr21.bcf','/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa',skip_out_of_scope=True,threads=16,overwrite=True)\""
```

Run: `pixi run -e py310 python scripts/svar2_reroute_spike.py --germline data/chr21.germline.svar2 --somatic data/gdc.chr21.somatic.svar2`

## Results

Germline: whole chr21 (3202 samples, 1,001,385 variants). Somatic: chr21:14–29 Mb
(16007 samples, 2,421,629 variants; a representative slice — the somatic pattern
is set by allele-frequency structure, not region size, and the region is >2× the
germline variant count). Subset sizes are `numpy.random.default_rng(0)` sample
draws; the last row per store (`k = n_samples`) is the control. `src_mismatch`
(analytic source-routing recount vs. the physical stream) was **0** for every
row, confirming the recount matches the store exactly.

| store | subset k | variants in view | flips | flip % | size delta % (preserve − reroute) |
|---|---|---|---|---|---|
| germline | 10 | 182,524 | 31,749 | 17.39% | +6.633% |
| germline | 50 | 320,228 | 26,088 | 8.15% | +3.641% |
| germline | 100 | 403,010 | 21,039 | 5.22% | +2.212% |
| germline | 500 | 668,566 | 8,822 | 1.32% | +0.415% |
| germline | 3202 (control) | 1,001,385 | 0 | 0.00% | +0.000% |
| somatic | 10 | 1,135 | 0 | 0.00% | +0.000% |
| somatic | 50 | 6,751 | 0 | 0.00% | +0.000% |
| somatic | 100 | 35,631 | 2 | 0.01% | +0.012% |
| somatic | 500 | 120,453 | 0 | 0.00% | +0.000% |
| somatic | 16007 (control) | 2,421,629 | 0 | 0.00% | +0.000% |

## Verdict

**Two regimes.**

- **Germline (moderate cohort, 3202 samples).** Small sample subsets flip a
  material fraction of variants (17 % at k=10, falling to 1.3 % at k=500) and the
  size penalty of preserving the source routing rises to **+6.6 %** at k=10. The
  driver is the cohort-size term in the dense cost (`n_samples·ploidy`): a variant
  that was dense at n=3202 (needs ≥190 carrier haps) is almost always cheaper as
  var_key once the cohort shrinks to a handful of samples, and vice-versa. By k=500
  the effect has decayed below the 1–2 % "material" band (+0.42 %).
- **Somatic (large cohort, 16007 samples).** Effectively no flips (≤0.01 %) and no
  size delta (≤0.012 %) at any subset size. The dense crossover at n=16007 is ~943
  carrier haps (SNP); almost every variant is var_key at the full cohort and stays
  var_key in any subset, so re-routing changes nothing.

**Why this does *not* justify building `reroute=False`.** The size delta is always
≥ 0 and one-directional: `reroute=True` chooses the per-variant cheaper
representation, so it is by construction never larger than the source-preserving
output, and is up to 6.6 % smaller for small germline subsets. A source-preserving
`reroute=False` mode would produce a strictly equal-or-larger store with identical
decoded genotypes — no size or correctness win. Its only possible advantage is
avoiding the cost-model recount + full re-conversion (a *speed* argument this spike
does not measure).

**Decision.** `reroute=False` stays `NotImplementedError`. `reroute=True` is
validated as the size-optimal sole mode. If a future need for a faster whole-store
/ large-subset copy arises, the measured near-zero flip/delta in exactly those
large-subset regimes means a fast path there would essentially reproduce
`reroute=True`'s bytes anyway — so the higher-value performance work is the
**streaming rewrite of the `reroute=True` `Svar2Source`** (its eager whole-contig
`Vec<RawRecord>` materialization is the real scaling limit; see
`2026-07-13-svar2-eager-materialization-benchmark.md`), not a second
representation-preserving code path.

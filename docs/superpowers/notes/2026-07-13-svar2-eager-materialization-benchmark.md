# SVAR2 `write_view` eager-materialization benchmark

**Deferred call-out (PR #105):** *"Eager `Vec<RawRecord>` materialization in
`Svar2Source` (plan-mandated) — builds the whole contig view in memory; worth a
benchmark gate before advertising cohort-scale whole-store copies."*

## TL;DR

`SparseVar2.write_view` peak memory is **O(n_variants × n_haps)** and tracks the
analytic lower bound almost exactly (~**31 GB** for a whole-chr21 view of the full
3202-sample germline cohort, against a **0.2 GB** output — a ~150× blowup). It is
fine for modest cohorts/regions (≤~1 GB up to a few hundred samples whole-chr21)
but grows linearly with `samples × variants` and will OOM at biobank cohort scale
or on large chromosomes × thousands of samples. **Recommendation: do not
advertise cohort-scale whole-store copies through `write_view` until
`Svar2Source` is made streaming.**

## Why

`Svar2Source::new` (`src/svar2_source.rs`) decodes the entire contig subset up
front:

1. a `BTreeMap<(pos,ilen,alt), Vec<bool>>` — one `n_haps`-long carrier bitset per
   variant (~`n_variants · n_haps` bytes), then
2. a `Vec<RawRecord>` — each record's `gt` a `Vec<i32>` of length `n_haps`
   (~`n_variants · n_haps · 4` bytes),

both fully resident before `next_record` drains the first record into the
pipeline. So the source alone costs ≈ `n_variants · n_haps · 5` bytes, on top of
the conversion pipeline's own working set.

## Method

`scripts/svar2_eager_bench.py` runs `write_view` over the whole contig for a
deterministic first-`k`-sample subset, one k per subprocess (so `ru_maxrss` is
that run's peak), on node-local disk (the Phase-2 merge mmaps output and
SIGBUSes on NFS). Store: `data/chr21.germline.svar2` (whole chr21, 3202 samples,
1,001,385 variants). `eager lower-bound` = `n_variants · k·ploidy · 5 B`.

Run: `pixi run -e py310 python scripts/svar2_eager_bench.py --store <store> --out-dir <local> --ks 100 500 1000 3202 --threads 16`

## Results

| samples k | haps | wall (s) | peak RSS (GB) | out size (GB) | eager lower-bound (GB) |
|---|---|---|---|---|---|
| 100 | 200 | 15.2 | 0.84 | 0.005 | 1.00 |
| 500 | 1000 | 34.5 | 2.99 | 0.025 | 5.01 |
| 1000 | 2000 | 65.4 | 6.89 | 0.052 | 10.01 |
| 3202 | 6404 | 256.3 | 30.96 | 0.203 | 32.06 |

Peak RSS is linear in `k` (≈ 9.7 MB per sample here) and sits at ~0.97× the
analytic lower bound — i.e. the eager source structures dominate the footprint,
exactly as predicted. Output is ~150× smaller than peak RSS. Wall time also grows
roughly linearly-to-superlinearly (decode + full re-conversion).

## Extrapolation (why it gates cohort-scale)

Peak ≈ `n_variants · k · ploidy · 5 B`. Holding the whole-chr21 germline variant
count:

| scenario | approx peak RSS |
|---|---|
| chr21, 3202 samples | ~31 GB (measured) |
| chr21, 100k samples | ~1 TB |
| chr1 (~4–5× chr21 variants), 3202 samples | ~140 GB |
| chr1, 100k samples | multi-TB |

Any of these OOMs a normal node. The view is processed per contig, so the peak is
per-contig, not whole-genome — but a single large contig × a large cohort already
exceeds available RAM.

## Verdict / gate

- **Safe to ship now** for the advertised MVP: region/sample subsets that keep a
  modest number of samples, and small-to-moderate stores. `concat`/`split`
  (Component A, pure file ops) are unaffected — this only concerns `write_view`.
- **Do NOT advertise cohort-scale whole-store copies** (all samples, whole large
  contig) via `write_view` until `Svar2Source` streams. The fix is mechanical: the
  `BTreeMap` groups variants in ascending `(pos, …)` key order already, so a
  streaming source can advance a position cursor across the per-hap call streams
  and emit each `RawRecord` as its position is passed, instead of buffering the
  whole contig — bounding memory to the pipeline's chunk window rather than the
  whole view. This is the same follow-up the reroute spike points to
  (`2026-07-12-svar2-reroute-measurement.md`).

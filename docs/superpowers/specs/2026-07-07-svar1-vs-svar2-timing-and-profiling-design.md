# SVAR1 vs SVAR2 timing + VCF→SVAR2 profiling/optimization

**Date:** 2026-07-07 · **Branch:** `svar-2` · **Home:** `genoray`

## Goal

1. Time SVAR1 vs SVAR2 conversion for the `for_loukik` BCFs.
2. Profile VCF→SVAR2 conversion with `perf` + `pyinstrument`, then optimize the
   hot path as far as diminishing returns allow.

## Data

`/carter/users/dlaub/repos/for_loukik/`:

| File | Type | Samples | Phasing | Notes |
| --- | --- | --- | --- | --- |
| `chr21.bcf` | germline | 3202 | phased (`0\|0`) | contains `<DEL>`/`<DEL:ME>` symbolics |
| `gdc.chr21.bcf` | somatic | 16007 | unphased (`0/0`) | VAF FORMAT field |

Both single-contig (chr21).

**Reference:** `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa` (`.fai` present).
REF alleles verified concordant with the germline file, so one reference serves
both.

## Environment findings

- `perf` present (`~/.pixi/bin/perf`); `perf_event_paranoid=2` → user-space
  sampling of own processes is allowed (no kernel symbols needed).
- `cargo-flamegraph` present.
- `pyinstrument` **not installed** — add to the pixi dev env for the profiling run.
- `Cargo.toml` has **no `[profile.*]`** → default release strips debuginfo; a
  profiling profile is required for symbolized perf output.

## Structural asymmetry (report honestly)

- SVAR1 (`_svar.py from_vcf`) parallelizes **per contig**
  (`n_jobs = min(cpu, len(contigs))`) → single-threaded on a single-contig BCF,
  and needs a `.gvi` index build first.
- SVAR2 fans chunks out within a contig via rayon.
- SVAR1 does **not** self-atomize; its store is semantically un-normalized vs
  SVAR2's atomized/left-aligned output. Timing comparison only — not a semantic
  equivalence claim.

## Plan

### 1. Shared data prep (once per dataset)

Drop out-of-scope variants for both engines:

```
bcftools view -V other,bnd <raw>.bcf -Ob -o <name>.filt.bcf
bcftools index --csi <name>.filt.bcf
```

Both converters read the identical `.filt.bcf`. SVAR2 also runs with
`skip_out_of_scope=True` (belt-and-suspenders). Work dir on
`/carter/users/dlaub` (48T free), not the repo.

### 2. Objective A — timing comparison

Per dataset, wall-time:

- **SVAR2:** `SparseVar2.from_vcf(out, filt.bcf, reference=REF,
  skip_out_of_scope=True, threads=None)` — full pipeline (atomize + left-align +
  biallelic split + encode).
- **SVAR1:** `SparseVar.from_vcf(out, genoray.VCF(filt.bcf), max_mem=..., n_jobs=-1)`,
  measuring `.gvi` index build and `from_vcf` separately.

Report: dataset × {SVAR1 gvi-index, SVAR1 convert, SVAR2 convert}, store sizes,
variant counts.

**Decision:** SVAR1 gets the raw-filtered (multiallelic) BCF — a direct raw→store
measurement, not pre-normalized.

### 3. Objective B — profile & optimize VCF→SVAR2

Focus on **gdc.chr21** (16007 samples — representative NGS scale).

1. Profiling build: add `[profile.profiling]` (inherits release, `debug = true`)
   to `Cargo.toml`; build with
   `RUSTFLAGS="-C force-frame-pointers=yes" maturin develop --profile profiling`.
2. `perf record -g --call-graph fp` around a `run_conversion_pipeline` driver →
   `perf report` + flamegraph SVG.
3. `pyinstrument` run to confirm the Python side is thin (characterize wrapper
   overhead).
4. Optimize under systematic-debugging discipline: hypothesis per hotspot, apply,
   re-measure. Correctness gate after each change:
   `cargo test --no-default-features --features conversion` (the known link-flag
   gotcha), pytest e2e, **and** a byte-identical store-hash check before/after.
5. Restore the normal `maturin develop --release` build at the end.

**Stopping point:** pursue the wins the profile exposes, stop at diminishing
returns, report before/after per change.

## Out of scope

PGEN→SVAR2 (M7), SVAR1/SVAR2 store semantic equivalence, query-path perf.

---

# Results (2026-07-07)

All runs on one dedicated `carter-compute` node (carter-cn-04, 32 cores) — the
login node has 2 cores under load ~35, unusable for timing. SVAR2 numbers are on
the `profiling` build (release + debuginfo + frame pointers, ~3.5% slower than a
clean release build, used consistently for baseline vs optimized).

## Data (filtered: `bcftools view -V other,bnd`)

| Dataset | Samples | Phasing | Variants (filtered) | Input BCF |
| --- | --- | --- | --- | --- |
| germline chr21 | 3202 | phased | 1,001,385 | 177 MB |
| gdc.chr21 somatic | 16007 | unphased | 4,525,689 | 1.15 GB |

Symbolic/breakend dropped: germline had a small tail (`<DEL>`/`<DEL:ME>`); gdc had
essentially none (somatic short-read).

## Profiling (perf, frame-pointer; pyinstrument)

- **pyinstrument (germline):** 41.401s of 41.441s (**99.9%**) inside the
  GIL-released Rust `run_conversion_pipeline`; the Python `from_vcf` wrapper +
  cyvcf2 enumeration are ~0.04s. The Python side is not worth optimizing — perf
  (native) is the right tool.
- **perf, by thread:** conversion of a single contig is **reader-bound** —
  `read-chr21` = 89.8% (germline) / 93.0% (gdc); the `dense2sparse_vk` transpose
  (`exec-chr21`) only 9.6% / 7.0%. `plan_thread_budget` gives a single-contig file
  1 pipeline + 4 htslib threads (~8 of 32 cores); parallelizing the executor would
  be nearly useless.
- **Reader hot spots (baseline):** per-bit `or_bit` packing 27–30%; per-sample GT
  decode (`Vec::from_iter` + `GenotypeAllele::from`) ~17–22%; malloc/free churn
  ~16–20%; BGZF `inflate`+`crc32` ~11% (germline) / ~21% (gdc).

## Optimizations (commit `0be7bee`, byte-identical output)

1. Decode GT from the raw `record.format(b"GT").integer()` buffer instead of
   `record.genotypes().get(i)` (which allocates a per-sample `Genotype(Vec)` for
   every sample of every record). `e >= 2 → (e>>1)-1`, else -1.
2. Pack presence bits one u64 word at a time (assemble in a register, one store per
   word) instead of `BitGrid3::or_bit` per bit.

## Timing (SVAR2 baseline → optimized)

| Dataset | Baseline | Optimized | Speedup | Output |
| --- | --- | --- | --- | --- |
| germline chr21 | 101.5s | **36.5s** | **2.78×** | byte-identical (sha256) |
| gdc.chr21 | 2582.7s (43 min) | **1076.0s (18 min)** | **2.40×** | byte-identical (sha256) |

Verification: 185 cargo tests + 525 pytest tests green; store content-hash
unchanged on both datasets.

## Post-optimization profile → next bottleneck

The alloc churn and per-sample GT decode are **gone**. gdc reader is now dominated
by **BGZF decompression** (`inflate_fast` 31% + `crc32_z` 14% ≈ **45%**, htslib) +
`bcf_get_format_values` 9%; packing dropped to 16%. Further gains would need
attacking htslib decode (more decode threads — currently capped at 4 with a
"diminishing past 4" note) or a parallel reader (offload GT-decode/atomize/pack off
the htslib-read thread), both higher risk. Banked the 2.4–2.8× here.

## SVAR1 vs SVAR2 (store-to-store wall time)

SVAR1 parallelizes per contig → single-threaded on these single-contig files, and
needs a `.gvi` index build first (its store is also un-atomized — timing only).

| Dataset | SVAR1 (total) | SVAR2 (optimized) | SVAR2 faster by | SVAR1 store | SVAR2 store |
| --- | --- | --- | --- | --- | --- |
| germline chr21 | 123.8s (`from_vcf` 123.4s + gvi 0.4s) | 36.5s | **3.4×** | 814 MB | 178 MB (**4.6× smaller**) |
| gdc.chr21 | **>3000s** (50-min timeout, did not finish) | 1076s (18 min) | **>2.8×** | — | 34 MB |

SVAR1 gdc (16007 samples, single-threaded on one contig) was still inside
`from_vcf` when the 50-min cap killed it — the true factor is larger. The SVAR2
store is dramatically smaller thanks to the hybrid inline/dense encoding (SVAR1
stores full sparse pointers + an un-atomized variant table).


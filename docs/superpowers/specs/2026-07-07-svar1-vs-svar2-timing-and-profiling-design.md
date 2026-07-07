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

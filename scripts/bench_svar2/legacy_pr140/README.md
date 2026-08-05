# Sharded-VCF reader budget benchmark (PR #140 review)

Harness + measurements for `SparseVar2.from_vcf`'s sub-contig sharded reader
budget, built to evaluate [PR #140](https://github.com/d-laub/genoray/pull/140)
(`perf(svar2): rebalance sharded VCF worker budget`).

## Layout

| file | purpose |
|---|---|
| `gen_vcf.py` | synthetic multi-sample, single/multi-contig bgzipped+indexed VCF |
| `bench_sharded.py` | sweeps `(reader_workers, per-shard HTSlib threads, overshard)`; records wall / phase-1 / CPU% / MaxRSS + a byte-identity oracle |
| `sweep.sbatch` | core-count (12/16/32) and cohort-shape sweeps |
| `sweep2.sbatch` | 48-core multi-contig regression, overshard axis, multi-contig re-run |

`bench_sharded.py` drives the **BENCH-ONLY** env hooks in `orchestrator.rs`
(`GENORAY_READER_WORKERS`, `GENORAY_SHARD_HTSLIB`, `GENORAY_OVERSHARD`) so one
build sweeps the whole space. Unset, they leave the planner untouched.

Slurm hands out **non-contiguous** CPU ids, so the sbatch scripts pin with the
job's real allocated ids (`os.sched_getaffinity`), not `0-(N-1)`.

## Correctness

Every configuration is hashed over the whole `.svar` store and compared against
the first config's digest. **All 60+ configurations were byte-identical** —
sharding does not perturb output at any worker count, HTSlib setting, or
overshard factor.

## What the planner actually allocates

`processing_threads` (main, used as the per-contig shard-worker count) vs
`reader_workers` (PR #140). Reproduction validated against PR #140's own unit
tests.

| cores | 1 contig: main → PR | 22 contigs: main → PR |
|---|---|---|
| 8  | 1 → 3   | 1 → 3 |
| 12 | 1 → 7   | 1 → 7 |
| 16 | 3 → 11  | 1 → 3 |
| 32 | 19 → 27 | 1 → 2 |
| 48 | 35 → 43 | **5 → 2** |
| 96 | 83 → 91 | **5 → 2** |

Main collapses to **1 worker between 6 and 14 cores** on a single contig — a real
cliff PR #140 removes. But in the multi-contig/high-core corner the PR *reduces*
the per-contig budget.

## Measured throughput (min-of-3/5, warm page cache, carter-cn-02)

`main s` / `PR s` are the wall times at each branch's *actual* planner output.

| workload | cores | main W → PR W | main s | PR s | speedup | best W | best s | RAM ×|
|---|---|---|---|---|---|---|---|---|
| 1 contig, 1000s × 100k | 12 | 1 → 7 | 10.16 | 11.36 | **0.90×** | 1 | 10.16 | 1.39 |
| 1 contig, 1000s × 100k | 16 | 3 → 11 | 11.21 | 10.75 | 1.04× | 1 | 10.55 | 1.53 |
| 1 contig, 250s × 400k | 16 | 3 → 11 | 10.25 | 10.43 | 0.98× | 5 | 10.17 | 1.22 |
| 1 contig, 4000s × 25k | 16 | 3 → 11 | 12.12 | 10.23 | **1.19×** | 7 | 10.20 | 1.83 |
| 1 contig, 1000s × 100k | 32 | 19 → 27 | 10.37 | 10.81 | 0.96× | 3 | 10.18 | 1.07 |
| 1 contig, 1000s × 100k | 48 | 35 → 43 | 10.25 | 10.40 | 0.99× | 3 | 10.04 | 1.14 |
| 8 contigs, 1000s × 100k | 16 | 1 → 3 | 26.00 | 26.05 | 1.00× | **7** | **6.23** | 1.20 |
| 8 contigs, 1000s × 100k | 48 | 5 → 2 | 10.17 | 15.23 | **0.67×** | 5 | 10.17 | 0.67 |

## Mechanism

The pipeline is `N shard readers → collector → executor → chunk writer`. Only the
readers are parallel; `executor::run_compute_engine` is a strictly serial
`recv()` loop over `dense2sparse_vk`.

`genoray::monitor=trace` shows the handoff directly (1000s × 100k):

- `w=1`: `dense=0/6` (queue starved), `cpu_shard≈100%`, `cpu_exec` 0–66% → **reader-bound**
- `w=3`: `dense=0-1/6`, `cpu_shard≈360%`, `cpu_exec≈50%` → approaching balance
- `w=8`: `dense=5/6` (queue **full**), → **executor-bound**

So added readers pay only until the `dense` channel stops draining. That knee sits
at **w≈3–7** and moves right with cohort size (250s → w=3; 4000s → w=7), i.e. with
per-record decompress+parse cost. It does **not** move with core count: the wall
floor is ~10.2 s at 12, 16, 32 and 48 cores alike.

Multi-contig conversion parallelizes the executor (one per concurrent contig),
which is why 8 contigs at 16 cores reach 6.23 s — below the single-contig floor.

## Cost of overshooting the knee

MaxRSS grows linearly in worker count, with a slope set by cohort size:

| cohort | MB per added worker |
|---|---|
| 250 samples | 15.2 |
| 1000 samples | 33.4 |
| 4000 samples | 74.5 |

Each in-flight worker holds its own dense chunk. `from_vcf` is the **only**
`from_*` method that hardcodes `chunk_size = 25_000` — `from_pgen`,
`from_vcf_list` and `from_svar1` all size it via `_auto_chunk_size`'s memory
budget. So the one path PR #140 changes is the one without a dense-chunk RAM
guardrail, and the PR multiplies in-flight chunks by 4–11×.

## Non-levers (measured, no effect)

- **`OVERSHARD_FACTOR`** (w=7, 16 cores): 1→10.08 s, 2→10.20, 4→10.16, 8→10.20,
  16→10.61. Flat; 4 is fine, 16 mildly hurts.
- **Per-shard HTSlib threads**: at 16/32 cores `hts=0` and `hts=1` are within
  noise (10.75 vs 10.60 at w=11). The PR's `0` is defensible but is not itself a
  measured win at these cohort sizes.

## Caveats

- Synthetic biallelic-SNP, GT-only VCFs at ≤4000 samples. The motivating AoU
  chr22 run has a far larger cohort and much larger BGZF blocks; the knee moves
  right with cohort size, so at AoU scale the PR's larger budget may genuinely
  pay off. These numbers do not refute the author's 299%-CPU observation — they
  show the formula is unbounded and mis-specified for multi-contig.
- Shared cluster; min-of-N over 3–5 reps. The multi-contig curve was re-run at
  5 reps and reproduced.

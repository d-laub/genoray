# `from_vcf_list` memory + O(N²): baseline and post-fix measurements

Companion to `2026-07-16-svar2-from-vcf-list-memory.md` (the 13-task plan). This is the
closing measurement report: baseline vs. after, for every task in that plan.

## Headline: the project's stated gate was MISSED

The plan's goal was to make the k-way merge's staging cost **linear** in N ("wall vs N
goes from ~N^1.8 to ~N^1.0") by carrying carrier lists through to `rvk` instead of
widening every record to N and scanning a 99.993%-zero grid to recover it.

**That did not happen.** The measured scaling exponent on CPU time is unchanged within
noise: **N^1.756 before, N^1.747 after** (same three N, like-for-like). What the project
delivered instead is a **constant-factor speedup of ~2.3-2.6x** at every measured N, plus
substantial allocation-churn and peak-live-heap reductions (Section 3). The asymptotic
behaviour — the actual content of issue #120 — is untouched. Section 2 explains why: a
second O(V x N) term, in FORMAT staging, was left in place by the plan's own design and
now dominates where the genotype term used to.

The rest of this document lays out what was measured, what is derived/extrapolated, and
what remains.

## Environment

- Shared 8-core box (`nproc=8`), heavily and variably loaded by other tenants (load
  average ranged ~3 to ~30 across the runs in this project). **Wall-clock here is
  noise-dominated** — the same 6-file smoke workload measured 16.5 s and 144 s under
  different load at different times.
- Because of that, **`cpu_s` (user+sys CPU time) is the load-robust signal and is what
  every scaling fit in this document uses.** The workload is single-core-bound by design
  (its own logs show `read≈99% exec=0%`), so `cpu_s ≈ wall_s` on a quiet box — measured
  cpu/wall agreement is within ~1-3% on every row in Section 1, which corroborates both
  the metric choice and the single-core claim. `wall_s` is reported alongside as
  indicative only, never as the basis for a fit or a verdict.
- Synthetic cohort from `scripts/bench_from_vcf_list/generate_cohort.py`, seed 0: 3
  contigs (1, 2, 3), `--n-variants 1000` (per contig per file), `--shared-frac 0.05`,
  `--indel-frac 0.1`, FORMAT fields VAF+DP (**F=2**).
- Baseline measured at commit `a94cd54` (post-Task-4: the mechanical, byte-identical
  `Calls` migration; no perf change had landed at that point). dhat baseline measured at
  `9a2846c`. "After" measured at `208c8ba` (all 12 code tasks landed).
- **Caveat that matters:** the plan's original "Measurement anchors" table
  (`2026-07-16-svar2-from-vcf-list-memory.md`) was taken at **F=7 on real Hartwig VCFs**.
  This report is **F=2 synthetic**. The two are **not comparable** — do not read anything
  here against those anchor numbers. This report's own baseline (`a94cd54`/`9a2846c`) is
  its control throughout.

## Section 1 — Scaling (cpu_s), the headline

Fits are log-log least squares on `cpu_s`.

| N | baseline cpu_s | after cpu_s | speedup | baseline wall_s | after wall_s |
|---|---|---|---|---|---|
| 250 | 221.90 | 98.84 | 2.25x | 216.09 | 102.56 |
| 500 | 809.75 | 315.09 | 2.57x | 803.17 | 320.01 |
| 1000 | 2531.23 | 1113.78 | 2.27x | 2393.90 | 1148.60 |
| 2000 | **not collected** | 3871.31 | — | not collected | 3905.00 |

- Baseline exponent (250/500/1000, the only baseline points): **N^1.756**
- After exponent (250/500/1000/2000, all four points): **N^1.770**
- After exponent, same 3 points as baseline (**like-for-like comparison**): **N^1.747**

**GATE: "wall vs N goes from ~N^1.8 to ~N^1.0" — MISSED.** The exponent did not move
(1.756 -> 1.747, within fit noise of each other). The gain this project delivered is a
**constant factor of ~2.3-2.6x**, not a change in asymptotic behaviour. At N=2000 the
merge costs 3871 s CPU against the 791 s a linear extrapolation from the N=250 point
would predict — **4.9x worse than linear**, confirming the superlinear growth is still
present after every task in the plan landed.

Baseline N=2000 was **not collected**: the run was killed mid-flight, and at the measured
N^1.756 baseline exponent it projects to roughly 8500 s CPU, which exceeds the 90-minute
per-point cap used for this sweep. That number is stated here only as a projection — it
is not in the table above, and it is not treated as measured.

## Section 2 — Why the exponent did not move (the finding)

This is the report's central technical conclusion.

`src/chunk_assembler.rs` stages FORMAT values as, per atom:

```
for each FORMAT field j { for s in 0..num_samples { push resolve_format(...) } }
```

i.e. **F x N per atom, unconditional on how the variant will eventually route**. Routing
to dense vs. sparse storage happens downstream in `rvk`, after chunk assembly, so at
staging time the dense/sparse decision has not been made yet. The variant union V grows
~linearly with N in a private-somatic cohort (the shape Task 1's harness now models
deliberately — see the plan's "union grows with N" fix), so total staging work is
**V(N) x F x N = O(N^2)** — and this term now dominates the merge.

Concretely at N=2000: V ≈ 3 contigs x (50 shared + 2000 files x 950 private) ≈ 5.7M
variants, so staging performs roughly **22.8 billion** `f64` pushes (5.7M x 2 x 2000). At
N=500 the same term is ≈2.9B — an ~8x rise for a 4x rise in N, i.e. quadratic. (The V
estimate here is **derived** from the generator's parameters, not directly counted at
runtime.)

The shape of the miss is precise and worth stating plainly: **Task 7 made
`RawRecord`/`AtomMeta` FORMAT carrier-sparse and removed the per-atom F x N
*allocation*, but the staging loop still re-densifies to F x N when it writes into the
staged columns — the sparse representation is carried all the way to staging and then
discarded there.** That is the same round-trip pathology the project diagnosed for
genotypes at the project's outset (the merge knew the carriers, widened to N, then
scanned the 99.993%-zero grid to recover them downstream), left in place for FORMAT.
Task 8 fixed the genotype half of that round trip (`rvk`'s grid scan, replaced by
route-before-densify); the FORMAT half of the same pathology remains.

`max_mem` (Task 11) bounds this term **per chunk**, which is why it is effective against
the OOM behaviour issue #120 opened with — it does not reduce the **total** work summed
across all chunks in a run, which is what the cpu_s-vs-N exponent measures.

**Recommended next step** (stated as a recommendation, not a promise, and not attempted
in this project): carry `FormatVals` into `DenseChunk` the way Task 8 carried `Carriers`,
and let `rvk` resolve per-column only for the variants that actually route dense —
mirroring exactly what Task 8 did for genotypes. That is the change that would move the
exponent.

## Section 3 — Allocation churn (dhat)

Same cohort (500 files / 3 contigs / F=2), same binary recipe,
`--features conversion,dhat-heap`. All runs verified `run_exit=0` with a complete 18 MB
store written across 3 contigs.

| metric | baseline (`9a2846c`) | after Tasks 5+7+8 (`6a590a6`) | final, all tasks (`208c8ba`) |
|---|---|---|---|
| Total bytes | 71,250,845,776 | 21,730,780,633 | **10,608,048,301** |
| Total blocks | 68,940,183 | 71,511,090 | **71,510,096** |
| At t-gmax | 390,198,456 B / 16,208 blk | 368,301,374 B / 64,952 blk | **168,317,320 B / 64,953 blk** |
| At t-end | 73,296 B / 93 blk | 73,296 B / 93 blk | **73,296 B / 93 blk** |

- **Bytes churned: 71.25 GB -> 10.61 GB, a 6.72x reduction (-85.1%).** MET, and by a wide
  margin.
- **Peak live heap: 390.2 MB -> 168.3 MB (-57%).**
- **At t-end unchanged at 73,296 B across every run** — there was never a leak in the
  baseline, and there is none now. This corroborates the project's founding diagnosis
  (allocator churn plus an O(V x N) round trip, not a leak).
- **GATE: "total blocks down >=10x" — MISSED.** Blocks went slightly *up* (+3.7%,
  68.94M -> 71.51M). Explain honestly: that gate was calibrated on the F=7 real-cohort
  profile, where the two named churn sites (`vcf_list_reader.rs:485`,
  `chunk_assembler.rs:338`) dominated block count as well as bytes. On this F=2 cohort
  those two sites were ~63% of *bytes* but only ~12% of *blocks* — so eliminating them
  moved bytes far more than it moved block count. The residual ~71.5M blocks are
  dominated by `vcf_reader::decode_format_raw` and `VcfRecordSource` (~1.5M blocks each
  across several call sites) — per-record FORMAT decode inside the per-file readers,
  which this project never targeted. Additionally, `Carriers`/`CarrierFormat` now
  allocate two small `Vec`s per record where one large `Vec` used to be, so bytes fall
  while block count does not.
- **GATE: "neither `vcf_list_reader::next_record` nor `decompose_raw_record` in the
  top-5 sites" — MET.** Baseline top-3 by bytes were `VcfListRecordSource::next_record`
  (33.37 GB / 2.78M blk), `chunk_assembler::decompose_raw_record` (11.12 GB / 1.39M blk),
  and `VcfListRecordSource` (5.56 GB / 1.39M blk). All three are gone from the profile
  after. Task 9's target (`types::StagedColumn::with_capacity`, which became the #1 byte
  site at 2.78 GB x 6 ≈ 16.7 GB in only 57 blocks after Tasks 5+7) accounts for most of
  the further 21.73 GB -> 10.61 GB drop.

## Section 4 — Peak RSS

| N | baseline MaxRSS | after MaxRSS | change |
|---|---|---|---|
| 250 | 1.24 GB | 1.19 GB | -4.4% |
| 500 | 2.07 GB | 1.85 GB | -10.6% |
| 1000 | 3.58 GB | 3.27 GB | -8.6% |
| 2000 | not collected | 5.99 GB | — |

- Baseline RSS exponent **N^0.765**; after **N^0.783** (all 4 points) / **N^0.732** (same
  3 points as baseline).
- **GOAL: "peak RAM independent of cohort size" — NOT ACHIEVED.** RSS still grows with N
  (1.19 GB at N=250 -> 5.99 GB at N=2000). The improvement is a modest 4-11% at matched N,
  not a change in scaling.
- Note for interpretation: Task 9's over-reservation fix targets **address space**
  (VmPeak / `ulimit -v`), not RSS — untouched reserved pages are never resident. So its
  large effect on dhat's byte total (which counts requested bytes, Section 3) correctly
  does **not** show up as a large RSS win here. This is expected, not a discrepancy
  between the two measurements.

## Section 5 — Correctness

- Store is **byte-identical** at every step: Python parity, including a bcftools-merge
  byte-parity oracle, stayed at 42 passed / 2 skipped across all ten code tasks.
- Rust: 281 lib unittests green; `check-core` (the `--no-default-features` query-core
  build) clean; clippy `-D warnings` and `cargo fmt --check` enforced on every commit.
- Python: 734 passed / 2 skipped / 16 xfailed.

## Section 6 — Summary

| gate | verdict |
|---|---|
| cpu_s exponent (~N^1.8 -> ~N^1.0) | **MISSED** (1.756 -> 1.747) |
| byte churn (dhat total bytes) | **MET** (6.72x reduction) |
| top-2 churn site elimination | **MET** |
| block count (>=10x reduction) | **MISSED** (+3.7%) |
| no leak at t-end | **MET** (unchanged, 73,296 B, before and after) |
| peak RSS independent of N | **NOT ACHIEVED** (still grows with N; 4-11% lower at matched N) |
| byte-identical store output | **MET** |

This branch is worth merging for a real ~2.3-2.6x constant-factor CPU speedup, a 6.72x
reduction in allocation churn, a 57% lower peak live heap, byte-identical output across
every task, and the `Carriers`/`Calls` plus route-before-densify infrastructure that a
genuine asymptotic fix would build directly on (Section 2's recommended next step reuses
Task 8's pattern rather than inventing a new one).

It explicitly does **not** close the O(N^2): the cpu_s-vs-N exponent is unchanged, peak
RSS still grows with cohort size, and issue #120's scaling wall — the reason the project
was opened — is not resolved by this branch.

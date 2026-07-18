# `from_vcf_list` scaling — measured baseline

Companion to `2026-07-15-svar2-from-vcf-list-scaling.md` (Task 5). Records what was
actually measured, what it says about each hypothesised cost (R1/R2/R3/S1/S2), and the
**explicit Phase 3 decision** the plan gates on this doc.

## Environment caveats (read before trusting any wall-clock number)

- Shared cluster box, **load average ~31** throughout (many unrelated jobs from other
  projects). Wall-clock is noisy; the same 6-file smoke measured 16.5 s and 144 s at
  different times. Treat `perf` **self-time percentages** as the load-independent signal
  and wall-clock as indicative only.
- Measurements below are **post-Task-6 and (where noted) pre/post-Task-7**.

## Cohort

`scripts/bench_from_vcf_list/generate_cohort.py`, 500 single-sample bgzipped+indexed VCFs
x 10,000 variants, `--contig chr1 --contig-len 1000000 --shared-frac 0.1 --indel-frac 0.1
--seed 0`. Reduced from the plan's 2000 x 30,000: generation shells out to
`bgzip`+`bcftools index` per file, which at this load made 2000 files uneconomical. N=500
is enough to fit the slope; extrapolations to 2000 are labelled as such.

> **Generator bug found and fixed while doing this (commit `3518c38`).** At 500 files the
> generator's per-file-random private positions collided *across* files (birthday paradox
> in a finite contig) with independently drawn REF/ALT, so `from_vcf_list` aborted with
> `cross-file REF disagreement at pos 0 ... file sample_00076 (column 76) has REF "A" but
> file sample_00115 (column 115) has REF "C"`. REF/ALT is now a deterministic function of
> position, so any collision agrees (mirroring a real cohort called against one
> reference). **Cohorts generated before `3518c38` are stale — regenerate them.**

## RAM vs N (Python entry, `/usr/bin/time -v`, post Task 6+7)

| N (files) | wall_s | MaxRSS (KB) | MaxRSS (MB) |
|---:|---:|---:|---:|
| 50  | 12.13 | 376,440 | 367.6 |
| 100 | 16.16 | 396,528 | 387.2 |
| 250 | 20.68 | 452,692 | 442.1 |
| 500 | 35.96 | 552,796 | 539.8 |

Per-interval slope: **0.392, 0.366, 0.391 MB/file** — strikingly linear.

**Fit: `RAM(N) ≈ 348 MB + 0.383 MB × N`** (~392 KB per input file).

Extrapolated: **N=2000 → ~1.1 GB**, N=10,000 → ~4.2 GB, N=43,000 → ~17 GB.

### R1 (linear dense-chunk term) — **NOT the driver at realistic N**

A packed dense chunk costs `chunk_size * n_samples * ploidy / 8`. Here `n_samples == N`
(one sample per file), so at the default `chunk_size=25_000` it is `25_000 * N * 2 / 8`
= **6.25 KB per file** — i.e. 3.1 MB at N=500, **12.5 MB at N=2000**.

That is **~1.6% of the measured ~392 KB/file slope**, and ~1% of the extrapolated 1.1 GB
peak at N=2000. **R1 is real but immaterial at realistic cohort sizes.**

Consequence for **Task 6** (shipped, `cc24f1d`): `_auto_chunk_size` only reduces
`chunk_size` below 25,000 once `N * ploidy > 256MiB*8 / 25_000`, i.e. **N > ~42,950
files**. Below that it returns exactly 25,000 — the old constant. So Task 6 is a
**large-N guardrail, not a fix for the 2000-file case**; it is correct and worth keeping
(it bounds the dense term at absurd N), but it does **not** move peak RAM at N ≤ ~43k.
The plan's framing of R1 as *the* linear RAM term does not survive measurement.

### R2 (O(N) open readers) — **IS the driver**

The unexplained ~386 KB/file (392 minus R1's 6.25) tracks the one thing that obviously
scales with file count: `from_vcf_list` opens **all N files concurrently, per contig**
(each with its own htslib/BGZF reader + buffers).

**Caveat — this is inference by elimination on a residual, not an isolated
measurement.** R3 (ledgers, below) also scales with N and was **not** measured, and
per-sample FORMAT buffers/sample metadata scale with N too; any of them could account for
part of that ~386 KB/file. A `dhat` run at N≥500 (recipe: `scripts/bench_from_vcf_list/
README.md` §4) would separate R2 from R3 cheaply — **do that before committing to Task 9's
design**, since Task 9 is expensive and is sequenced on this attribution.

### R3 (ledger growth) — **UNDECIDED, not measured**

The plan gates Task 10 on a `dhat` run flagging `var_key_ledgers`/`dense_ledgers`. **That
run was not performed** (the dhat build path is wired and verified to compile/run — Task 2
— but the profiling run itself was not done at this load). R3 remains an open question;
do not build Task 10 on speculation.

## CPU hotspots (`perf`, native bench binary, N=500)

Profiling build with `-C force-frame-pointers=yes`, flat sampling `-F 499`, self-time.

| symbol | BEFORE Task 7 | AFTER Task 7 |
|---|---:|---:|
| `VcfListRecordSource::next_record` | **34.72%** (#1) | **3.09%** (#11) |
| `rvk::dense2sparse_vk` | 6.08% | **10.75%** (#1) |
| `vcf_parse` | 5.06% | 6.67% |
| `__memmove_avx_unaligned_erms` | 4.34% | 6.54% |
| `BinaryHeap::pop` | 3.28% | 4.48% |
| `next_line` | 3.26% | 4.27% |
| `chunk_assembler::flush_window` | 2.46% | 4.07% |
| release wall-clock (500 files) | 74.3 s | 35.6 s |

### S1 (merge selection) — **CONFIRMED, and now fixed**

`next_record` was the #1 self-time symbol at 34.72%. Its O(N) cursor scan ran once per
atom read *plus* once per record emitted, i.e. O(N²·variants) total against O(N·variants)
downstream — so its **share grows linearly in N** and would be worse at 2000 files.

**Task 7** (frontier min-heap, `d3c4efa`) took it to **3.09%** and moved the bottleneck
downstream to `dense2sparse_vk`. Byte-identical parity held. Wall-clock 74.3 s → 35.6 s
(~2.1x) — larger than removing 34.7% of CPU alone predicts, so some of that is load
variance; the self-time drop is the trustworthy number.

### S2 (single-threaded decompression) — **partially indicated, not isolated**

Post-Task-7, htslib input parsing is spread across several symbols — `vcf_parse` 6.67%,
`next_line` 4.27%, `bcf_hdr_id2int` 3.57%, `__strcmp_avx2` 3.19%, `vcf_parse_format`
2.85%, plus `hts_itr_next`/`bcf_unpack` — totalling **~20%+**, consistent with the
existing "htslib-INPUT-bound (inflate+parse)" characterisation. Thread utilisation was
**not** measured, so whether parallel leaf decompression pays is not established here.

## Per-fix scorecard

| # | Fix | Status | Measured verdict |
|---|---|---|---|
| R1 | Budget-derive `chunk_size` (Task 6) | **shipped** `cc24f1d` | Correct, but only bites at **N > ~43k**. Dense term is ~1% of peak at N=2000. Keep it: it converges `from_vcf_list` with `from_pgen`/`from_svar1` (which already default this way) and guards absurd N — but it is **not** the #120 fix. |
| S1 | Frontier min-heap (Task 7) | **shipped** `d3c4efa` | **Big win.** 34.72% → 3.09% self-time; wall 74.3 s → 35.6 s at N=500; win grows with N. |
| R2 | Cap open readers (staged merge, Task 9) | **not built** | **Inferred** RAM driver (~386 KB/file *residual*, not isolated — R3 is unmeasured and could account for part of it). Most likely blocker to "RAM independent of cohort size"; confirm with dhat first. |
| S2 | Parallel leaf decompression (Task 9) | **not built** | htslib parse ~20%+; thread utilisation unmeasured. Plausible, unproven. |
| R3 | Ledger trim (Task 10) | **not built** | **Unknown — dhat not run.** |
| — | `max_mem` knob (Task 8) | **not built** | Worth doing, but see decision below. |

## Phase 3 decision (the gate this doc exists for)

- **Task 9 (staged / batched merge) — WARRANTED, and it is now the highest-value
  remaining work — but run dhat first (one cheap step).** The linear RAM term is real and
  measured (~383 KB/file → ~1.1 GB at N=2000, ~4.2 GB at N=10k); attributing it to R2 is
  inference by elimination (see the R2 caveat above), and R3 is an unmeasured claimant on
  the same residual. A dhat run at N≥500 confirms the split before spending the expensive
  batching work on it. Assuming it confirms: batching files into intermediate sorted runs
  caps concurrent open readers and is the only identified change that makes peak RAM
  independent of cohort size — the stated goal of issue #120. It would also remove the
  `ulimit -n` ceiling `_check_fd_budget` currently enforces.
  - The **S2 half** (rayon-parallel leaf decompression) is *not* independently justified
    by these numbers — bundle it only if the batching work makes it near-free, otherwise
    split it out and measure thread utilisation first.
- **Task 8 (`max_mem` knob) — worth doing, but re-aim it.** The plan has it size
  `chunk_size`; measurement says `chunk_size` is ~1% of peak at realistic N, so a
  `max_mem` that only tunes `chunk_size` would be a knob that does nothing. It should
  primarily size **Task 9's batch width / max-open-files** (the actual RAM lever), with
  `chunk_size` secondary. **Sequence it after Task 9.**
- **Task 10 (ledger trim) — DEFER, undecided.** Its gate (a dhat run) was not executed.
  Run dhat at N≥500 (recipe: `scripts/bench_from_vcf_list/README.md` §4) before deciding.

## Reproducing

```bash
# cohort (regenerate — anything older than 3518c38 is stale)
python scripts/bench_from_vcf_list/generate_cohort.py /tmp/cohort500 \
    --n-files 500 --n-variants 10000 --contig chr1 --contig-len 1000000 \
    --shared-frac 0.1 --indel-frac 0.1 --seed 0

# RAM/time sweep
for k in 50 100 250 500; do
  python scripts/bench_from_vcf_list/run_bench.py --manifest /tmp/cohort500/manifest.txt \
      --out /tmp/sweep_$k --chrom chr1 --subset $k --profiler time --results results.csv
done

# perf self-time (native, no Python in the loop)
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling \
    --no-default-features --features conversion --bin bench_from_vcf_list
perf record -F 499 -o perf.data -- target/profiling/bench_from_vcf_list \
    /tmp/cohort500/manifest.txt /tmp/out chr1
perf report -i perf.data --stdio --sort=overhead,symbol -g none
```

Note: `cargo test`/debug builds must set `CARGO_TARGET_DIR` to local disk — on the NFS
`target/` they fail with `failed to map object file: memory map must have a non-zero
length` (proptest/hts-sys).

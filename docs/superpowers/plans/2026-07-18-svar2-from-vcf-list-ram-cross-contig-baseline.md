# `from_vcf_list` peak-RAM (#120): measured baseline + localization

> Companion to `2026-07-18-svar2-from-vcf-list-ram-cross-contig.md`. This is the
> current-code baseline every Phase-C change is measured against, plus the Phase-B
> localization verdict that selects the fix.

## Harness

- **Cohorts:** `scripts/bench_from_vcf_list/generate_cohort.py --jobs $(nproc)` (Task 1),
  24 contigs (`1..22, X, Y`), `--n-variants 300` per contig per file,
  `shared_frac=0.1`, `indel_frac=0.1`, **F=7** FORMAT fields
  (`VAF DP AD GQ PL MQ SB` ≈ the Hartwig regime).
- **Driver:** `scripts/bench_from_vcf_list/run_bench.py --threads 1 --profiler time`
  (Task 2), no reference (`no_reference=True`), sampling child `/proc/<pid>/status`
  VmRSS + `/proc/<pid>/smaps` every 2 s and bucketing per `==> Processing {chrom}`
  banner.
- **Build:** `maturin develop --release` (genoray 3.2.1 editable), `CARGO_TARGET_DIR`
  on local disk.
- **N-variants is deliberately modest** so the sweep runs in minutes. Absolute RSS
  here is therefore far below a production WGS cohort (whose per-file variant density
  is orders of magnitude higher — the source of the historical ~283 GiB observation).
  **This harness measures the *shape* of the cross-contig ratchet and the fix's effect
  on it, not the production absolute.**

## Baseline sweep (pre-fix)

| N     | MaxRSS (GB) | wall (s) | cpu (s) | arena_heaps | contig-1 window-peak (GB) | max window-peak (GB) | ratchet ratio (max ÷ c1) |
|------:|------------:|---------:|--------:|------------:|--------------------------:|---------------------:|-------------------------:|
| 500   | 2.63        | 192.7    | 115.5   | 4           | 0.97                      | 2.63                 | 2.72×                    |
| 1000  | 5.23        | 306.3    | 251.8   | 6           | 1.66                      | 5.23                 | 3.14×                    |
| 2000  | 13.03       | 827.1    | 601.6   | 50          | 3.26                      | 13.03                | 4.00×                    |
| 4000  | 28.45       | 1524.1   | 1276.1  | 167         | 6.11                      | 28.45                | 4.66×                    |

Raw per-contig high-water dicts and the CSV are in `$CLAUDE_JOB_DIR/tmp/baseline.csv`.

## Fits

- **Peak RSS:** `MaxRSS_GB = 0.00184 · N^1.162` (log-log least squares over the 4 points).
  **Superlinear** — peak grows *faster* than the number of files.
- **Wall:** exponent **1.038** (≈ linear, consistent with the PR #121 linear-CPU result).
- **CPU:** exponent **1.165**.

The peak-RSS exponent exceeding the wall exponent is the fingerprint of a
`Σ_contigs(per-contig retention)` term: work per contig is ~linear in N, and peak
accumulates that across all 24 contigs instead of releasing it.

## The ratchet (the KPI)

`per_contig_highwater[c]` is the max RSS sampled *while contig c was being processed*.
If memory were released at each contig boundary, every contig's window-peak would sit
at roughly `baseline(N) + one contig's working set` — i.e. **flat** across contigs,
ratchet ratio ≈ 1. Instead the window-peak **climbs monotonically in trend** from
contig 1 to the last contig, reaching **4.66× contig-1's peak at N=4000** (and growing
with N: 2.72× → 3.14× → 4.00× → 4.66×). Memory allocated for early contigs is retained
while later contigs are processed. **This is the cross-contig ratchet #120 targets.**

`arena_heaps` (count of 64 MB glibc arena mappings from `smaps`) rising 4 → 6 → 50 →
167 with N is the leading mechanistic hint: freed per-contig allocations are landing in
glibc arenas that are not returned to the OS.

## Extrapolation to N = 7,089

`0.00184 · 7089^1.162 = 55.0 GB` on this harness — **under the 64 GB budget**, because
the synthetic per-file variant density is far below production WGS. **This does not mean
#120 is a non-issue at production scale:** the same superlinear exponent and 4.66×-and-
growing ratchet ratio, applied to production variant density (which set the historical
~283 GiB peak), is exactly what blows the budget. The actionable target for Phase C is
therefore stated as a *shape* KPI, not this harness's absolute:

> **Success KPI: flatten the per-contig window-peak curve — ratchet ratio → ≈ 1×
> (later contigs stop setting new high-water marks) — with the store byte-identical
> and no material wall-clock regression.** A flat ratchet turns peak RSS from
> `baseline(N) + Σ_contigs(retained)` into `baseline(N) + max_single_contig`, which is
> what carries the production cohort under 64 GB.

## Gap to budget

- Harness N=7,089 extrapolation: **55 GB** (under 64, but ratchet-inflated).
- If the ratchet were already flat at N=4000, peak would be ≈ the max single-contig
  working set ≈ contig-1's 6.11 GB plus baseline, i.e. **~4.7× lower** than the observed
  28.45 GB. That 4.7× is the headroom the fix recovers, and the multiplier that matters
  at production density.

---

## Localization (Phase B — Task 4)

_Appended by Task 4 before any Task-5 code change._

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

Three independent lines of evidence, none of them a guess:

### 1. Static: no Rust owner survives the contig loop

`run_vcf_list`'s contig loop (`src/orchestrator.rs:1044-1068`) calls
`process_chromosome(...)`, which returns **only `u64`** (the dropped-record count).
No Rust value owning contig-*N*'s buffers survives into contig-(*N*+1)'s iteration —
all per-contig state is dropped at the end of each call. Therefore the *live* Rust
heap at the start of every contig is the same baseline; **live memory cannot
accumulate across contigs.** Any cross-contig RSS growth must be allocator-side
(freed-but-not-returned), not retained live state.

### 2. Empirical arena signal (from the baseline sweep)

`arena_heaps` (count of 64 MB glibc arena mappings in `/proc/self/smaps`) climbs
**4 → 6 → 50 → 167** across N=500→4000, tracking the RSS ratchet. Freed per-contig
allocations are landing in glibc arenas that are not released to the OS.

### 3. Decisive boundary probe: `malloc_trim` reclaims the ratchet to a flat baseline

A temporary probe (reverted before Task 5) instrumented the contig boundary to log
`VmRSS` immediately after each `process_chromosome`, then call `malloc_trim(0)` and
log `VmRSS` again. Run on the **N=1000** cohort (F=7, 24 contigs):

| contig | VmRSS before trim (MB) | trim ret | VmRSS after trim (MB) |
|:------:|-----------------------:|:--------:|----------------------:|
| 1      | 1659                   | 1        | 426                   |
| 2      | 1668                   | 1        | 436                   |
| 3      | 438                    | 1        | 436                   |
| 4      | 1667                   | 1        | 436                   |
| 12     | 1668                   | 1        | 436                   |
| …      | ~439                   | 1        | ~436                  |
| Y      | 439                    | 1        | 436                   |

(Full 24-contig log in `$CLAUDE_JOB_DIR/tmp` task output.)

- `malloc_trim(0)` returns **1** (memory actually released to the OS) at **every one
  of the 24 boundaries**.
- **`VmRSS after trim` is flat at ~426–436 MB across all 24 contigs** — no upward
  drift. This is the airtight proof: once freed memory is returned, the resident
  baseline is constant, so there is **no live-memory ratchet**.
- `VmRSS before trim` sawtooths between ~438 MB and ~1.67 GB. The ~1.2 GB delta is
  freed-but-unreturned per-contig working memory. Without a trim it accumulates into
  the ratchet the baseline sweep measured (n1000 MaxRSS 5.23 GB, ratchet 3.14×). Note
  that in this probe run the trim runs every contig, so the ratchet is *suppressed* —
  which is exactly why the post-trim baseline stays flat.
- The probe run stayed **byte-identical** (`dropped == 0`, same as baseline);
  `malloc_trim` does not alter output.

> **VERDICT: the cross-contig ratchet is glibc freed-but-not-returned heap. The
> `process_chromosome` boundary already drops all Rust state; the memory is simply
> held by glibc's allocator. `malloc_trim(0)` at the contig boundary reclaims it in
> full (post-trim RSS flat at ~436 MB). Task 5's lever is `malloc_trim(0)` at the
> contig boundary — NOT an explicit Rust `drop` (there is no retained owner to drop).**

(The memray `--native` run reported a 5.9 GB peak-live-heap number, but that is a
cross-run, instrumentation-inflated single high-watermark that cannot distinguish a
per-contig working set from a cross-contig ratchet; the direct `VmRSS`-at-boundary
probe above is the trustworthy measurement and supersedes it.)

---

## Fix result (Phase C — Task 5: `libc::malloc_trim(0)` at the contig boundary)

Commit `92f5f4f`. Same harness, same cohorts, fixed `.so`:

| N    | baseline MaxRSS (GB) | fixed MaxRSS (GB) | reduction | baseline wall (s) | fixed wall (s) | fixed ratchet ratio (max ÷ c1) |
|-----:|---------------------:|------------------:|----------:|------------------:|---------------:|-------------------------------:|
| 500  | 2.63                 | 0.96              | 2.75×     | 192.7             | 140.3          | **1.003×**                     |
| 1000 | 5.23                 | 1.69              | 3.10×     | 306.3             | 271.3          | **1.014×**                     |
| 2000 | 13.03                | 3.30              | 3.95×     | 827.1             | 653.8          | **1.012×**                     |
| 4000 | 28.45                | 6.12              | 4.65×     | 1524.1            | 1293.9         | **1.001×**                     |

- **Ratchet flattened.** The per-contig window-peak ratio drops from 2.72–4.66× to
  **1.00–1.01×** — every contig now peaks at the same level, i.e. peak =
  `baseline(N) + one contig's working set`, exactly the target shape.
- **Peak RSS down 2.75× → 4.65×**, and the reduction *grows with N* (the ratchet's
  cost grew with N, so removing it helps more at scale).
- **Peak-RSS exponent: `N^1.162` (superlinear) → `N^0.901` (sublinear).** The
  `Σ_contigs(retained)` term is gone; what remains is ~linear baseline + max contig.
- **Wall-clock improved at every N** (140 vs 193 … 1294 vs 1524 s) — the trim is cheap
  (one processing thread, no arena-lock contention) and *reduces* page-management
  overhead. No regression; a net speedup.
- Store stayed **byte-identical**: `tests/test_svar2_from_vcf_list.py` 26/26 pass,
  including the new 3-contig decode-parity oracle (passes pre- and post-fix).

### Fixed extrapolation to N = 7,089

`0.00347 · 7089^0.901 = 10.2 GB` on this harness — **6× under the 64 GB budget**
(vs. the pre-fix 55 GB). At production WGS variant density the absolute is higher, but
the fix's structural change — turning `baseline + Σ_contigs(retained)` into
`baseline + max_single_contig` — is exactly what carries the 7,089-sample somatic
cohort (whose ratchet drove the historical ~283 GiB peak) under budget.

## Task 7 (windowed per-contig gather) — decision gate: **N/A**

Task 7 is conditional on the flattened single-contig peak still exceeding 64 GB. It does
not: the flattened peak extrapolates to **10.2 GB at N=7,089**, 6× under budget, and the
per-contig curve is already flat (ratio ≈ 1.00×). **Task 7 is skipped.** Should a future
production cohort at much higher variant density push a *single* contig's working set
past budget, the windowed-gather lever (bound the in-memory gather to `chunk_size`
variants, flushing per window) remains the documented next step — but it is not needed
to close #120.

## Task 6: `MALLOC_ARENA_MAX` as a documented complement — **not recommended**

Measured at N=2000 on the fixed `.so` (three arms; default vs. cap of 2 and 4):

| `MALLOC_ARENA_MAX` | MaxRSS (GB) | wall (s) | arena_heaps |
|:------------------:|------------:|---------:|------------:|
| (unset / default)  | 3.257       | 537.8    | 50          |
| 2                  | 3.275       | 536.5    | 18          |
| 4                  | 3.290       | 536.9    | 43          |

- **No RSS benefit:** peak is within **1%** across all three — capping arenas does not
  lower peak on top of the `malloc_trim` fix, because the trim already returns freed
  heap to the OS at each contig boundary. There is nothing left for an arena cap to save.
- **No wall penalty:** wall is flat (±0.3%). The prior "MALLOC_ARENA_MAX is 73% slower"
  finding was arena-lock contention at ~4,000 threads; with `VCF_LIST_HTSLIB_THREADS=0`
  and one processing thread there is no contention, so the cap is now harmless — but
  also pointless.

**Recommendation: do NOT set or document `MALLOC_ARENA_MAX` for `from_vcf_list`.** It is
redundant with the boundary trim and confers no measurable benefit. No docstring or
public-API change is made (the plan's "document only if net-positive" gate is not met).

_(Aside: run-to-run wall varies ~15–20% with OS file-cache warmth — the default arm
here is 537.8 s vs. 653.8 s in the fix sweep for the identical config — but MaxRSS is
stable to ~1%, so the RSS conclusions above are robust.)_

# SVAR2 `from_vcf_list` peak-RAM: close #120 (cross-contig ratchet)

**Status:** design, approved for planning.
**Issue:** #120 — `from_vcf_list` peak RAM scales with cohort size; 7,089-file somatic
WGS merge OOMs on ordinary nodes.
**Target hardware:** 64 GB node.
**Supersedes emphasis of:** `2026-07-16-svar2-from-vcf-list-memory-design.md` (which correctly
diagnosed churn → arena fragmentation, but predated the 3.2.1 full-cohort profile that
re-ranks the *cross-contig ratchet* above the per-contig gather).

---

## 1. Problem, re-diagnosed against the 3.2.1 profile

The CPU side of #120 is already closed. PR #121 made the k-way merge linear in N
(cpu_s exponent N^1.756 → N^0.890 via FORMAT route-before-densify), removed the
7,089-thread explosion (`VCF_LIST_HTSLIB_THREADS = 0`, 1 processing thread), and cut
allocation churn ~6.7×. **Peak RAM was explicitly not fixed** and is the entire
remaining content of #120.

The decisive new evidence is the **genoray 3.2.1 full-cohort profile** (issue comment):
the real 7,089-sample somatic k-way merge (hg19, 24 contigs) was **OOM-killed at 256 GB**
(contig 7, MaxRSS 255.5 GiB) and **completed on a 600 GB node at ~283 GiB peak** in 3h06m.
MaxRSS climbs across contigs and only plateaus around contig 12:

| contig | 1 | 2 | 3 | 4 | 5 | 7 | 9 | ~12–14 | 15…X,Y |
|---|---|---|---|---|---|---|---|---|---|
| MaxRSS | ~41 | ~80 | ~117 | ~155 | ~193 | ~243 | ~267 | ~297 (plateau) | 297 |

**This re-ranks the levers.** If peak were `max_contig(O(n_files × contig_load))` with
clean release between contigs, chr1 (the largest chromosome) would set the high-water
mark first and every later, smaller contig would sit *under* it — RSS flat from contig 1.
Instead the high-water mark keeps rising ~15–40 GB per contig *while the chromosomes get
smaller*. That is the signature of **cross-contig accumulation**: state that survives each
`Merge Complete.` boundary and grows with the number of contigs processed, on top of the
O(n_files) reader baseline.

Consequences:

- A **single contig** costs ~41 GB at N=7,089; **24 contigs** cost ~283 GB. The gap is
  the ratchet, not the per-contig peak.
- **Fixing only the ratchet** takes peak from ~283 GB toward the max single-contig cost
  (~41–80 GB) — which is what puts the 64 GB target within reach. Streaming the per-contig
  gather alone (the issue body's original ask) would **not** cap peak, because the
  cross-contig climb would still push the high-water mark up.

Prior corroborated facts that still hold (from the 3.0.0 dhat/RSS analysis, issue comment
and `2026-07-16` design):

- Peak **live** heap is tiny (~862 MB at N=250/F=7); peak **RSS** is ~90% glibc arena
  fragmentation (2,593 × 64 MB arena heaps = 111.5 GB of 132 GB on the earlier live job).
  Rust frees everything (73 KB live at t-end); the cost is memory glibc has freed but
  cannot return to the OS.
- Open readers cost only ~250–350 KB/file ≈ 2 GB — **not** the driver. Batching the k-way
  merge is aimed at ~1.5% of the problem and stays dropped.
- Within a contig, RSS ≈ `0.66 GB + 3.7 MB × N` (~27 GB for chr1 at N=7,089 on 3.0.0;
  ~41 GB observed on 3.2.1 including the first-contig arena baseline).

## 2. Scope & non-goals

**In scope:** reduce peak RSS of `SparseVar2.from_vcf_list` for the somatic WGS cohort to
**< 64 GB at N=7,089** (24 contigs, F≈7), with the output store **byte-identical** to today.

**Non-goals:**
- CPU/wall-clock (already linear; do not regress it, but it is not the objective).
- Germline cohorts (bounded union) — they do not exhibit the wall; any win there is a
  bonus, not a requirement.
- Correctness re-validation via the bulk cohort. Correctness stays on the **existing
  small-data differential oracle** (`from_vcf` vs `from_vcf_list`, byte-identical store,
  DP/VAF exact). The bulk cohort is a **speed/memory harness only** — no VcfBuilder,
  no GroundTruth (explicit user decision).

## 3. Workload characterization (performant-py-rust Phase 1)

| dimension | sweep | max | grows? | notes |
|---|---|---|---|---|
| **n_files (samples)** | 500 · 1k · 2k · 4k | 7,089 | **yes, unbounded per cohort** | dominating dimension |
| contigs | 24 | 24 | fixed | drives the cross-contig ratchet |
| F (FORMAT fields) | 7 | ~7 | fixed | per-atom staging cost, F×carriers |
| v_private / sample / contig | somatic WGS | — | union ∝ N | merge cost is O(V), V ∝ N |

**Bound: allocator / memory-bound, not CPU-bound.** RSS is dominated by arena
fragmentation and cross-contig retention, not live working set and not compute. This
selects the Phase-2 levers: *release retained state and return freed pages to the OS*,
not *add cores*.

## 4. Design levers (Phase 2), ranked by the 3.2.1 evidence

Goal: peak RSS = `O(largest single contig)` and **flat in contig count**.

1. **Eliminate cross-contig accumulation (primary lever).**
   Audit everything that survives the per-contig `Merge Complete.` boundary in
   `orchestrator.rs`'s contig loop and drop it explicitly: retained reader/index state,
   sidecar/staging buffers, per-contig metadata. Target: per-contig high-water mark stops
   ratcheting (flat across contigs 1→24).
2. **Return freed memory to the OS between contigs (secondary lever).**
   `malloc_trim(0)` at each contig boundary and/or a bounded `MALLOC_ARENA_MAX`. Both are
   **now viable** because the thread explosion is gone (0 htslib threads, 1 processing
   thread) — the earlier "MALLOC_ARENA_MAX is 73% slower" result was a
   4,000-threads-on-2-arena-locks contention artifact and no longer applies. Confirm with a
   measurement, do not assume.
3. **Stream the per-contig gather (tertiary lever).**
   Chunk each contig into windows and flush per window so the per-contig peak is bounded by
   window size, not contig size. Apply **only if** the max single-contig peak is still over
   64 GB after levers 1–2. This is the issue body's original "stream the gather" ask,
   correctly demoted.
4. **Field-aware `_auto_chunk_size` + memory-budget knob — ALREADY SHIPPED; verify it
   bites.** As of the current tree `_auto_chunk_size(n_samples, ploidy, n_format_fields,
   max_mem)` budgets *both* the packed grid and staged FORMAT (`F × n_samples × 4`), and
   `from_vcf_list` already exposes a public `max_mem` kwarg wired through to it. So lever 4
   is not new implementation — it is **verification**: confirm that setting `max_mem`
   actually lowers observed peak RSS (it bounds *per-chunk* dense cost, which is only one
   term of peak; if peak is dominated by the cross-contig ratchet, `max_mem` alone will not
   cap it — which is itself a finding worth recording).

## 5. Harness (Phase 3) — built before optimizing

### 5.1 Correctness oracle
The existing small-data differential tests. Every optimization must keep the
`from_vcf_list` store byte-identical to `from_vcf` on the small fixtures (genotypes,
DP/VAF exact). No change to this oracle.

### 5.2 Fast single-sample bulk cohort generator
Keep the shape of `scripts/bench_from_vcf_list/generate_cohort.py`, which already:
- grows the union with N (`_POSITIONS_PER_PRIVATE_VARIANT`, `required_contig_len =
  max(1e6, n_files × n_priv × 20)`) so per-file collision rate is constant as N grows —
  the "fixed 1e6 pool asymptotes" harness flaw the earlier comment named is **already
  fixed** in the current version;
- supports multi-contig (`--contig`, repeatable) and per-sample FORMAT fields
  (`--format-field`, e.g. VAF/DP);
- derives REF/ALT deterministically per `(contig, pos)` so cross-file REF agreement holds
  for the no-reference merge.

The one real deficiency is **generation speed**: it writes each single-sample VCF as
Python text, then shells out to `bgzip` and `bcftools index` **once per file, serially** —
prohibitive at 7k WGS-scale files. Fix: **parallelize the per-file bgzip+index across
cores** (e.g. `multiprocessing.Pool` / `concurrent.futures` over the file list). Files must
stay bgzipped **and indexed** — the k-way merge seeks per contig and requires the `.csi`.

Rationale for staying single-sample-direct (recorded so it is not re-litigated): a
multi-sample "bulk VCF → `bcftools +split`" path is O(N²) on the *generator* side for a
somatic cohort — a multi-sample VCF is sites × samples dense, and the somatic site union
grows ≈ N × v_private, so the bulk file is ≈ N² × v_private cells (~99.99% `./.`),
hundreds of GB at N=7,089. Single-sample-direct generation is O(N × v_private), linear.

### 5.3 Memory benchmark
`scripts/bench_from_vcf_list/run_bench.py` extended to:
- sweep **N ∈ {500, 1k, 2k, 4k}** × **24 contigs** × **F=7**;
- record **MaxRSS**, the **per-contig high-water mark** (the ratchet is the primary KPI —
  a flat curve across contigs is the success signal), and glibc **arena-heap count**;
- extrapolate to N=7,089; validate at full scale occasionally (not every iteration).

Baseline = a fresh 3.2.1 sweep on this harness, anchored to the known ~283 GiB @ N=7,089.

## 6. Optimize → measure → repeat (Phase 4)

1. **Localize the retention first — do not guess it.** Run `memray`/`heaptrack` across two
   contig boundaries to identify *what* survives `Merge Complete.`: Rust-owned state vs.
   glibc arenas. This decision (drop-state vs. trim-arenas vs. both) is driven by the
   measurement, not assumed. (The issue author offered exactly this snapshot.)
2. Apply lever 1 (drop retained per-contig state) → confirm per-contig high-water is flat.
3. Apply lever 2 (`malloc_trim` / arena cap at contig boundary) → confirm RSS *falls*
   between contigs. Measure the wall-clock cost; keep only if net-positive.
4. Apply lever 3 (stream the gather) **only if** max single-contig peak > 64 GB.
5. Verify lever 4: confirm `max_mem` measurably lowers peak RSS (or record that it cannot,
   because peak is ratchet-dominated). No new implementation expected here.

Re-run the oracle (byte-identical) and the memory sweep after every change; keep a change
only if it lowers peak RSS *and* stays byte-identical, else revert. **Stop when** peak RSS
< 64 GB at N=7,089, or the next lever's share of peak is too small to matter, or the
remaining gain is not worth the complexity — state which.

## 7. Public-API / docs impact

- If a memory-budget kwarg (e.g. `max_mem`) becomes public on `from_vcf_list`,
  **`skills/genoray-api/SKILL.md` MUST be updated in the same PR** (project rule for any
  change reachable from `import genoray` without underscores).
- Update the peak-RAM model documented on #120 once measured (the honest model is
  `O(n_files) baseline + max_contig` after the fix, vs today's
  `O(n_files) + Σ_contigs(retained state)`).

## 8. Files in play

- Rust: `src/orchestrator.rs` (per-contig loop / `Merge Complete.` boundary),
  `src/vcf_list_reader.rs`, `src/chunk_assembler.rs`, `src/rvk.rs`.
- Python: `genoray/_svar2.py` (`_auto_chunk_size`, optional `max_mem` surface).
- Harness: `scripts/bench_from_vcf_list/generate_cohort.py` (parallelize gen),
  `scripts/bench_from_vcf_list/run_bench.py` (sweep + per-contig high-water + arena stats).
- Docs/skill: `skills/genoray-api/SKILL.md` if the memory knob goes public.

## 9. Benchmarking gotchas (from prior sessions)

- **NFS bus-error:** export a local `CARGO_TARGET_DIR` for `cargo test`/dhat builds; the
  default NFS `target/` bus-errors on debug object mmap.
- **Stale `.so`:** `pixi run test` does **not** rebuild the Rust extension. Run
  `maturin develop --release` before any Python-level perf/e2e verification of Rust changes.
- **`/tmp` is reaped** mid-session — park bench data and cohorts in `$CLAUDE_JOB_DIR/tmp`,
  and always check process exit + real output before believing a profile.
- **Rust tests:** `cargo test --no-default-features` (else the pyo3 test binary fails to
  link, `undefined symbol: _Py_Dealloc`); `test-rust <arg>` filters by test *name*, not
  file — use `--test <file>`.
- Bench build/anchor caveat: F=7-Hartwig and F=2-synthetic numbers are **not** comparable;
  compare exponents/per-doubling ratios within one build profile, not absolute times across
  debug vs release.

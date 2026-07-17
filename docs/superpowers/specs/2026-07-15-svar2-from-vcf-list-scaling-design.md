# SparseVar2.from_vcf_list — memory & speed scaling

**Issue:** #120 (peak RAM scales with cohort size → OOM)
**Also targets:** multi-hour wall-time on large cohorts
**Date:** 2026-07-15

## Problem

`SparseVar2.from_vcf_list` builds one SVAR2 store from many single-sample VCFs
via a native k-way merge. On a real cohort of **7,089 single-sample somatic
VCFs** (each ~3 MB), against hg19:

- **Peak RAM scales with cohort size.** The job was SIGKILLed by the cgroup OOM
  at **63.8 GB RSS** (64 GB limit) during the **phase-1 read of contig 2**,
  before anything was gathered or written. Bumping to 256 GB only moves the
  ceiling; a few thousand more samples (or WGS germline) hits it again.
- **Wall-time is multi-hour.** chr1 alone took **~3h37m with 8 threads**, and
  the 8 threads bought almost nothing because the merge is single-threaded.

## Root-cause hypotheses (from static reading of the pipeline)

`from_vcf_list` → `_core.run_vcf_list_conversion_pipeline` →
`orchestrator::run_vcf_list`, which loops contigs **sequentially**, calling
`process_chromosome` with a `VcfList` source. That source
(`VcfListRecordSource`, `src/vcf_list_reader.rs`) opens **all N files at once**
and runs a single-threaded streaming k-way merge on one `read-{chrom}` thread.
Dense chunks flow through bounded channels (`tx_dense=6`) to an executor + writers.

### Speed

- **S1 — O(atoms·N) merge selection (confirmed in code).**
  `VcfListRecordSource::next_record` (`src/vcf_list_reader.rs:457`) scans **all N
  cursors on every atom** to pick the min-frontier cursor. The code comment
  itself notes "A frontier heap would make selection O(log N) — deferred." At
  N≈7089 × somatic-scale atom counts this quadratic-ish term is the prime
  suspect for the multi-hour cost.
- **S2 — the merge is single-threaded (confirmed in code).** The entire k-way
  merge (decompress + parse + normalize N files) runs on one reader thread with
  `htslib_threads=1` per file; the "processing threads" only feed a packing pool
  that prior profiling showed is <0.05% of reader time. So 8 cores are wasted
  during the merge.

### RAM (peak scales with N)

- **R1 — dense chunk size is fixed, not budgeted (confirmed in code).**
  `from_vcf_list` hardcodes `chunk_size=25_000` and **never calls
  `_auto_chunk_size`** (unlike `from_svar1`/`from_pgen`). A packed dense chunk is
  `chunk_size · N · ploidy / 8` bytes → **~443 MB at N=7089** (vs the 256 MB
  `_DENSE_CHUNK_TARGET_BYTES`), times channel depth 6 + in-flight copies. Grows
  linearly with N.
- **R2 — O(N) open htslib readers.** All N `IndexedReader`s are held open with
  their BGZF decompression buffers for the whole contig — an O(N) RAM baseline
  before any variant data (the "7089 files open at once" banner).
- **R3 — ledger accumulation (to be confirmed).** `var_key_ledgers` /
  `dense_ledgers` (`src/executor.rs`) grow with the chunk count over a contig.
  Whether this is a material fraction is left to the dhat profile rather than
  guessed at.

## Approach: measure-first

Per "measure, don't guess," we build a **benchmark + profiling harness first**,
then apply a **prioritized, measurement-gated optimization sequence** where each
fix is validated (and re-profiled) before the next.

### Part A — Benchmark & profiling harness

Location: `scripts/bench_from_vcf_list/` (Python driver + generator) plus a
native Rust bench binary.

1. **Synthetic somatic-cohort generator.** Emits M single-sample,
   bgzipped + tabix/CSI-indexed VCFs, each with ~K mostly-private SNVs/indels on
   a configurable contig, seeded and reproducible, no PHI. A tunable fraction of
   **shared sites** (to exercise the k-way join, not only private-variant fan-out)
   and a tunable indel rate (to exercise the long-allele path). Sweeps
   **N ∈ {100, 500, 2000, …}** so RAM-vs-N and time-vs-N **slopes are measured**,
   not asserted.
2. **Optional real-manifest mode.** The same driver accepts a manifest/dir of
   real VCFs (point it at the 7089-cohort) to reproduce the exact 63.8 GB / 3h37m
   profile when the data is reachable.
3. **Profiling drivers, one flag each.**
   - **Rust allocations → `dhat`.** A **feature-gated (`dhat-heap`) native
     benchmark binary** `src/bin/bench_from_vcf_list.rs` calls
     `orchestrator::run_vcf_list` **directly** (no Python), with a
     `#[global_allocator] dhat::Alloc` and a `dhat::Profiler::new_heap()` guard,
     both behind the `dhat-heap` cargo feature so the shipped `_core` wheel is
     unaffected. Emits `dhat-heap.json` (peak heap + per-site backtraces) to
     localize R1/R2/R3. The same binary is the clean target for the next two.
   - **CPU hotspots → `perf record`/`report`** on the native binary (should
     finger S1).
   - **Call counts → `callgrind`** on the native binary (confirms the per-atom
     N-scan count for S1).
   - **Codegen → `cargo-show-asm`** on the confirmed hot function.
   - **Python allocations → `memray`** on the real `from_vcf_list` entry, to
     catch the pre-flight cyvcf2 header scan over N files and `_resolve_fields`.
4. **Parity fixture.** A small fixed cohort whose SVAR2 output is snapshotted, so
   every optimization asserts **byte-identical output**. This is the hard
   correctness gate for the merge rewrite (S1) especially.
5. **Results artifact.** The driver writes a small table/plot of MaxRSS and
   wall-time vs N per revision, so each optimization lands with a before/after
   number.

### Part B — Optimization sequence (each gated on re-profiling)

Ordered by confidence and cost; re-measure after each.

1. **R1 (cheap, high value).** Make `from_vcf_list` derive `chunk_size` from
   `_auto_chunk_size(n_samples, ploidy)` when the caller does not set it (mirror
   `from_svar1`/`from_pgen`), and reconsider the `tx_dense` depth. *Expected:*
   dense-pipeline RAM flat in N. *Public-surface note:* `chunk_size`'s default
   semantics change (fixed 25k → budget-derived); update the docstring and
   `skills/genoray-api/SKILL.md`.
2. **S1 (high value).** Add a **frontier min-heap** to `VcfListRecordSource` so
   cursor selection is O(log N) instead of O(N) per atom. Must be **byte-identical**
   under the parity fixture (same tie-break/first-min-wins ordering). *Expected:*
   the multi-hour selection term collapses.
3. **R2 + S2 (bigger, conditional on the dhat/perf numbers).** A
   **staged/hierarchical k-way merge** — merge files in batches into intermediate
   sorted runs, then merge the runs — which simultaneously **caps open FDs**
   (bounds R2) and unlocks **parallel leaf decompression** across cores (fixes
   S2). Built **only if** dhat shows R2's reader baseline, or perf shows S2's
   serialization, is a dominant fraction. If R2 turns out small, a lighter fix
   (raise `htslib_threads`, or a bounded reader pool) may suffice.
4. **Memory-budget knob.** Expose a `max_mem` (and/or `max_open_files`) parameter
   on `from_vcf_list` per the issue, sizing chunk/batch/tile counts from it, and
   document the resulting peak-RAM model. Public surface → update the docstring
   and `skills/genoray-api/SKILL.md`.
5. **R3.** Only if dhat flags ledger growth as material.

Each step is its own commit carrying a before/after harness number.

## Success criteria

- **Peak RAM flat in N** — validated on the synthetic sweep to ≥10k files on a
  64 GB node (no linear RAM-vs-N slope through the read phase). Exact target
  refined once the harness gives a baseline.
- **chr1 wall-time cut by roughly an order of magnitude** on the reproduction
  profile.
- **Byte-identical output** on the parity fixture for every optimization (hard
  constraint).
- **8 threads actually used** during the merge (S2), measurable as >1 core busy.

## Non-goals / out of scope

- Cross-contig parallelism for `from_vcf_list` (contigs stay sequential; that is
  what bounds open FDs today and is orthogonal).
- Changing the on-disk SVAR2 format or the merge's join-on-atom semantics.
- Preserving in-file `./.` missingness through the merge (SVAR2 is sparse; a
  missing hap and a hom-ref hap are indistinguishable by design).

## Risks & mitigations

- **Merge rewrite (S1) changing output.** Mitigated by the byte-identical parity
  fixture, run in CI-scale before/after; the frontier heap must reproduce the
  exact `min_by_key` first-min-wins tie-break.
- **dhat global-allocator leaking into the wheel.** Mitigated by gating the
  allocator + profiler behind the `dhat-heap` feature and a separate bench
  binary — never compiled into `_core`.
- **Staged merge adding disk-I/O passes** that trade RAM for time. Mitigated by
  making it conditional on measurement and keeping batch size tunable via the
  memory knob.
- **Synthetic data not matching the real cohort's pathology.** Mitigated by the
  optional real-manifest mode to confirm the fix on the actual 7089 VCFs.

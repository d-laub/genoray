# SVAR2 conversion: generalized parallelization & load balancing

**Date:** 2026-07-15
**Status:** Design (approved for planning)
**Builds on:** PR #115 (`perf(svar2): parallelize and shard VCF normalization`, draft)

## 1. Goal

Make single-contig VCF→SVAR2 and PGEN→SVAR2 conversion scale with available
cores, and pick the parallelization/load-balancing scheme from measurement
rather than intuition. Concretely:

1. A benchmarking + profiling harness that produces scaling curves, localizes
   the bottleneck, and gates every change on byte-identical output.
2. Sharded conversion generalized behind one backend-agnostic abstraction
   covering **single-file VCF** (harden PR #115) and **PGEN** (new).
   `from_vcf_list` is explicitly **out of scope** this iteration.
3. Better load balancing than PR #115's equal-basepair static split +
   serial-by-ordinal drain.
4. A data gate that decides whether to stop at the shared-executor architecture
   (Approach A) or promote to fully independent per-shard sub-pipelines
   (Approach B).

Non-goals: cross-contig scheduling changes, dosage support, `from_vcf_list`
sharding, changes to the query/read path.

## 2. The measured reality this design must respect

Prior profiling (32 cores, single contig, byte-identical germline + gdc
oracles; recorded in project memory `svar2-conversion-reader-bound`):

- Within a contig, conversion is a 4-thread pipeline (reader ∥ executor ∥
  chunk-writer ∥ long-writer). It is **reader-bound (~78% of wall)**. The
  executor (`dense2sparse_vk` transpose) is ~10%; writers are small; the
  Phase-2 merge is already rayon-parallel.
- After the earlier packing / GT-decode wins, the reader's self-time is
  **htslib *input*: BGZF `inflate` ~32% + `vcf_parse` ~9% + name tokenization
  ~14%.** GT-decode and presence-packing no longer register (<0.05%).
- **Raising htslib decode threads 4→8 was inert (−0.5%, noise).** The reader
  inflates/parses on its own serial path; the indexed-fetch read path may not
  even engage decode threads.

Consequences that shape the whole design:

- **The unit of useful parallelism is "an independent reader over a region"**,
  each with its own `inflate`+`parse` serial path — not more decode threads on
  one reader. This answers "4 htslib threads per shard vs a channel": prefer
  **more shards** over more decode threads per shard. (The harness re-confirms
  this on current code before we rely on it.)
- Normalization (atomize + left-align + field resolution) is **not** the
  bottleneck. PR #115's batched-parallel normalization is marginal for
  shardable inputs; it stays only as the intra-contig parallelism for
  **unshardable** paths.
- Once readers are parallelized ~P×, the **single shared executor (~10% today)
  becomes the next ceiling**. Whether we hit it is the central measurement.

> These numbers are point-in-time. Phase 0 re-establishes them on current code
> (post PR #115) before any decision depends on them.

## 3. What PR #115 provides (the foundation we extend)

Two independent mechanisms land in PR #115:

- **Batched parallel normalization** — `ChunkAssembler::fill_normalize_batch`
  refills the reorder heap in `NORMALIZE_BATCH_RECORDS = 1024` batches,
  decomposed via a rayon pool (`decompose_raw_record` in parallel), then pushed
  onto the min-heap serially. Kept, but scoped to unshardable inputs.
- **Sub-contig VCF sharding** — `plan_vcf_shards` splits a contig's owned
  region into up to `processing_threads` shards. `read_vcf_shards_to_dense`
  spawns one thread per shard; each opens its own `IndexedReader`, fetches a
  **padded** interval `[own_start − L_MAX, own_end + L_MAX)`, normalizes, and
  (via `ChunkAssembler::with_reference`'s `owned_range`) keeps only atoms whose
  left-aligned position lands in `[own_start, own_end)` — so an indel that
  left-aligns across a boundary is emitted exactly once. The collector drains
  per-shard bounded SPSC channels **serially in ordinal order**, reassigns a
  global `chunk_id`, and forwards to the single `tx_dense` → single executor.

Limitations we address:

- **VCF-only.** PGEN and VcfList do not shard.
- **Load balance.** Equal-basepair split ≠ equal variant count ≠ equal work.
  Density-skewed contigs (gene deserts, index gaps) leave shards idle. The
  serial-by-ordinal drain with `bounded(2)` channels means later shards stall
  at 2 buffered chunks while an early/dense shard is still draining; tail
  latency = slowest shard, with no work stealing.
- **Single executor** is a hard scaling ceiling.

## 4. Architecture

### 4.1 Phase 0 — Benchmarking & profiling harness (built first)

Extend `scripts/svar2_region_parallel_bench.py` into a three-layer harness.

**Macro (scaling).** Per-subprocess end-to-end measurement (so peak RSS is
meaningful) recording wall time, peak RSS, output bytes, and throughput
(variants/s and samples×variants/s). Sweep `threads ∈ {1,2,4,8,16,32}` ×
backend {VCF, PGEN} × dataset {germline chr21, somatic gdc (16007 samples)}.
Derived: **speedup vs 1-thread and parallel efficiency**. Matched PGEN is
generated once from the BCFs via `plink2` so VCF vs PGEN is apples-to-apples.

**Micro (localize).** Documented, scriptable recipes committed alongside the
harness:

- `perf stat` — task-clock vs wall (are N cores actually busy?),
  context-switches, CPU-migrations, IPC.
- `perf record` / `perf report` — confirm `inflate`/`parse` dominance and
  **track the executor's share as readers parallelize**.
- `perf sched` / off-CPU profiling (or `perf lock`) — measure collector
  serialization and channel stalls directly (the empirical SPSC-vs-MPSC
  question).
- `valgrind --tool=callgrind` + `cachegrind` — deterministic instruction-count
  and cache A/B of a single change on a fixed small input (serialized; measures
  per-work-item cost, not scaling).
- `cargo-show-asm` — inspect hot inner-loop codegen (packing / normalize) to
  confirm inlining and vectorization survive refactors.

**Correctness gate.** Every configuration is checked **byte-identical** against
the current serial conversion using the established `storehash` oracle, wired
into the harness as a hard per-row pass/fail. No scheme ships that changes
output bytes.

Environment: run on `carter-compute` via `srun --overlap` (never the login
node), with the `LD_LIBRARY_PATH`/`DYLD_FALLBACK_LIBRARY_PATH` setup already
documented for the build.

**Deliverable:** a re-established baseline profile for current `main` + PR #115,
answering: (a) is htslib-input still dominant? (b) what is the executor's share
at 1 thread, and where does it land at 8/16/32 shards?

### 4.2 Backend-agnostic shard abstraction

Extract PR #115's VCF-specific sharding into one module with a clear seam:

- **Shard planner** — produces ordered, disjoint work units. Each unit owns a
  contiguous range and, in ordinal order, covers the contig in position order.
  - *VCF*: unit = padded position interval (existing `VcfShard`).
  - *PGEN*: unit = variant-index sub-range of `var_start..var_end`, with
    boundary indels handled by including neighbor variants whose `.pvar`
    position is within `L_MAX` of the boundary. Variant-count balance is free
    (exact counts from the index range).
- **Parallel reader/collector** — a fixed pool of worker threads (sized to the
  thread budget), each owning one backend reader that it re-seeks to
  successive work units pulled from a shared queue. Output is collected in a
  **reorder buffer keyed by (ordinal, local_chunk_id)** and released in global
  order, then handed to the downstream pipeline. This bounds open readers/file
  descriptors to the worker count while allowing work stealing.

PGEN plumbing: Python already builds one `PgenReader` per contig and passes a
list to `run_pgen_conversion_pipeline`. It builds **P readers per contig**
instead (cheap: each is a file handle plus the shared, file-wide
`allele_idx_offsets`). Reconstructing readers in Rust is avoided — the
multiallelic `allele_idx_offsets` requirement (project memory
`pgenlib-multiallelic-gotchas`) makes Python-side construction the safe seam.

### 4.3 Load balancing — Approach A (this iteration)

Replace equal-basepair static split + serial-ordinal drain with:

- **Over-decomposition**: plan more shards than workers (e.g. a small multiple),
  sized toward equal *variant count* where the index allows (exact for PGEN;
  estimated from CSI/TBI or a cheap scan for VCF).
- **Work-stealing pool**: workers pull the next unit when free — naturally
  balances density skew. A shared MPSC work queue *of shard assignments*
  (not of undifferentiated record streams) plus per-worker reader reuse.
- **Reorder buffer**: preserves the position-sorted `chunk_id` order the
  Phase-2 merge relies on, even when shards finish out of order.

All shards still funnel into the **single existing executor** (Approach A). This
is the low-risk, clear win and also directly resolves the SPSC concern (the
serial-ordinal drain is gone).

### 4.4 Data gate → Approach B (conditional)

After Phase 0 + Approach A are measured, if the **executor's share of wall time
exceeds a pre-committed threshold (target: >25%)** at high shard counts, the
single executor is the ceiling and we promote to:

- **Approach B — fully independent per-shard sub-pipelines.** Each shard runs
  reader→executor→writer over its region and writes chunk files tagged
  `(ordinal, local)`; the existing rayon merge stitches them with no inter-shard
  channels. Requires: `chunk_id` → `(shard, local)` threaded through the writer
  and merge iteration order; and the **per-contig long-allele table sharded
  per worker then rebased at merge** (var_key indel offsets shifted by each
  shard's cumulative long-allele byte offset). The §4.2 abstraction is designed
  so a shard can emit either DenseChunks (A) or its own SparseChunks + partial
  long-allele table (B) without reworking the planner/collector seam.

If the executor share stays below threshold, Approach A ships as the final
architecture and B is left documented as future work.

## 5. Components & boundaries

| Unit | Responsibility | Depends on |
|------|----------------|-----------|
| Harness (`scripts/…bench.py` + profiling recipes) | Scaling numbers, profiles, byte-identical gate | genoray CLI, plink2, perf, valgrind, cargo-show-asm |
| Shard planner (Rust) | Ordered disjoint work units per backend | index metadata (VCF CSI/TBI, PGEN var range + `.pvar`) |
| Parallel reader/collector (Rust) | Worker pool, reader reuse, reorder buffer | planner, `ChunkAssembler` |
| VCF backend adapter | Padded position fetch + owned-range filter | `vcf_reader` |
| PGEN backend adapter | Variant-index sub-range + boundary padding | `pgen_reader`, `pvar` |
| Python `from_pgen` | Build P readers/contig, pass through | `pgenlib` |

Each is testable in isolation: the planner is pure arithmetic over ranges
(unit-tested like PR #115's `plan_vcf_shards` tests); the collector's ordering
is testable with synthetic out-of-order completions; adapters are exercised by
the existing e2e + boundary fixtures; the harness's correctness gate is the
integration check.

## 6. Correctness & determinism

- **Byte-identical output** vs current serial conversion is the invariant. The
  boundary fixture from PR #115 (indel left-aligning across a shard boundary,
  emitted exactly once) is extended to PGEN and to the over-decomposed /
  work-stealing collector.
- **Deterministic order**: the reorder buffer emits `(ordinal, local)` in global
  position order regardless of completion order, preserving the merge invariant.
- **Error propagation & cancellation**: a failing shard cancels siblings and
  surfaces its work-unit coordinates (as PR #115 already does for VCF);
  generalized to PGEN units.

## 7. Public API impact

Conversion entry points (`SparseVar2.from_vcf`, `from_pgen`) keep their
signatures; parallelism stays internal and budget-driven. If any user-visible
name, kwarg, or behavior changes (e.g. a new threads/shard knob), the same PR
updates `skills/genoray-api/SKILL.md` per repo policy. Expected: no public API
change beyond possibly surfacing existing thread controls.

## 8. Risks

- **Executor ceiling** — mitigated by the §4.4 data gate.
- **PGEN boundary correctness** — variant-index sharding must include L_MAX
  neighbors; covered by an extended boundary fixture and the byte-identical
  gate.
- **File-descriptor / reader count** — bounded by worker count via reader reuse,
  not by shard count.
- **PR #115 is a draft** — Phase 0 begins by rebasing/validating it (its own
  validation checklist) so the baseline is real.
- **Long-allele table rebasing (B only)** — the trickiest part of B; isolated to
  the merge and only reached if the data gate fires.

## 9. Phasing

0. Harness + profiling recipes + byte-identical gate; re-establish baseline on
   `main` + PR #115.
1. Extract backend-agnostic shard planner + parallel reader/collector; VCF
   adapter (harden PR #115) behind it.
2. PGEN adapter (variant-index sharding, boundary padding, P readers/contig).
3. Load balancing (Approach A): over-decomposition + work-stealing + reorder
   buffer; measure speedup/efficiency and executor share.
4. **Data gate.** If executor share > threshold → Approach B (per-shard
   sub-pipelines + long-allele rebasing). Else finalize A.
5. Docs (`docs/source/svar.md`, roadmap) + `SKILL.md` if any public surface
   moved.

Phases 1–3 have parallelizable sub-tasks (planner vs harness extensions vs PGEN
Python plumbing) suitable for parallel implementer agents.

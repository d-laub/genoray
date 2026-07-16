# SVAR2 Parallel Conversion — Approach A vs B Decision (2026-07-15)

## Context

Phase 3 finalized **Approach A**: a backend-agnostic sub-contig shard planner
(`src/shard.rs`, `src/pgen_shard.rs`) feeding a fixed worker pool + reorder-buffer
collector (`src/shard_exec.rs`) that emits `DenseChunk`s into a **single shared
executor** in global `(shard_ordinal, local)` order, with over-decomposition
(`OVERSHARD_FACTOR=4`, VCF) for work-stealing load balance.

Phase 4 (**Approach B**) would promote each shard to its own independent
reader→executor→writer sub-pipeline, stitched by the merge. The plan gated Phase
4 on a measurement: **execute Phase 4 only if the single shared executor is the
bottleneck — specifically if the executor (`dense2sparse_vk`) exceeds 25% of wall
time at 32 threads.**

## Measurement

See `svar2-conversion-baseline-2026-07-15.md`. Summary:

- **VCF (chr21):** 3.92× speedup at 32 threads, byte-identical. Speedup tracks
  the number of sub-contig **reader** shards; the conversion is htslib-input
  (inflate+parse) bound (memory `svar2-conversion-reader-bound`, ~78% reader at 1
  thread). The executor stage is downstream of the reader and is not on the
  critical path — adding executor parallelism cannot beat the reader ceiling.
- **PGEN (chr21c):** sharding is byte-identical but **net slower** (340 s vs
  273 s serial) because `pgenlib` decode holds the GIL; the bottleneck is the
  GIL-serialized read, not the executor (memory
  `pgenlib-holds-gil-sharded-reads`).

In both backends the bottleneck is the **reader**, not the shared executor. The
executor's wall-time share is well below the 25% gate.

## Decision

**Finalize Approach A. Phase 4 (Approach B) is NOT built** — it is documented as
future work, to be revisited only if a future reader-side win (e.g. a
GIL-free/native PGEN reader, or a faster BGZF path) shifts the bottleneck onto
the shared executor. Per-shard independent executors would add merge complexity
(shard-partitioned chunk identity, per-shard long-allele bins rebased at merge)
for no benefit while the reader dominates.

### PGEN-specific recommendation

PGEN sub-contig sharding is **byte-identical but counter-productive** (GIL-bound
reads). It is retained in the codebase (correct, tested) but should be treated as
**not beneficial by default**; a follow-up may cap PGEN `processing_threads`-side
sharding to 1 (single-reader fast path) unless/until a GIL-free PGEN reader
exists. VCF sharding — the real win — over-decomposes; PGEN intentionally does
not (`OVERSHARD_FACTOR` is applied to the VCF branch only).

## Gate resolution

> Gate: executor share > 25% of wall at 32 threads → Phase 4. Else finalize A.

**Executor share ≤ 25% (reader-bound) → Approach A finalized. Phase 4 skipped.**

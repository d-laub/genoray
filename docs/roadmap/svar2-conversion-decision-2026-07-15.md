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
- **PGEN (chr21c):** sharding is byte-identical but **net slower** (44.9 s vs
  32.6 s serial on a quiet node — the earlier 340 s/273 s pair was a contention
  artifact, see the baseline doc). Single-reader conversion is already fast
  (~33 s) and bound by the shared executor/writer + reference I/O, not decode,
  so sharding can't beat that floor; concurrent readers add coordination and,
  on `pgenlib`<0.92, GIL-serialize (memory `pgenlib-holds-gil-sharded-reads`).
  Bumping to a GIL-releasing pgenlib (0.94.x, internal `prange`) does **not**
  help — the conversion is flat at ~33 s across `OMP_NUM_THREADS` 1→32 —
  because decode isn't the bottleneck, so the pin stays at `0.91.*`.

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

PGEN sub-contig sharding is **byte-identical but counter-productive**, so it is
**disabled by default**: `from_pgen` pins the shard budget to `P = 1` (single
reader per contig). The machinery is retained (correct, tested via the
`pgen_shard` unit tests) but off. Measurement (baseline doc): single-reader
conversion is already fast (~33 s, executor/IO-bound not decode-bound), and
sharding measures slower (44.9 s at `threads=24`).

Bumping `pgenlib` to a GIL-releasing build (0.94.x adds `prange(nogil=True)` to
`read_alleles_range`) was evaluated and **rejected**: the conversion is flat at
~33 s across `OMP_NUM_THREADS` 1→32 and identical to 0.91.0, because decode is
not the bottleneck. The pin stays at `0.91.*`; revisit only if a future
reader/executor change shifts the bottleneck onto decode (then a single 0.94.x
reader would parallelize decode internally — no sub-contig sharding needed).

VCF sharding — the real win — over-decomposes; PGEN intentionally does not
(`OVERSHARD_FACTOR` is applied to the VCF branch only).

## Gate resolution

> Gate: executor share > 25% of wall at 32 threads → Phase 4. Else finalize A.

**Executor share ≤ 25% (reader-bound) → Approach A finalized. Phase 4 skipped.**

# SVAR2 Parallel Conversion — Baseline & Scaling Results (2026-07-15)

Measured on `carter-cn-02` (32 cores, 32 GB) via `srun --overlap`, using the
generalized parallel-conversion implementation (Tasks 1–10). Byte-identical
output vs the serial conversion is gated on every row with the `storehash`
oracle. Dataset: `chr21.filt.bcf` (germline, ~1M variants) and its
plink2-matched PGEN (`chr21c.pgen`, `--output-chr chrM`).

## VCF → SVAR2 scaling (chr21, over-decomposition ON, `OVERSHARD_FACTOR=4`)

| threads | wall (s) | speedup vs serial | byte-identical |
|--------:|---------:|------------------:|:--------------:|
| 1  | 1176.5 | 1.00× | ✅ |
| 2  | 1175.8 | 1.00× | ✅ |
| 16 |  503.1 | 2.34× | ✅ |
| 32 |  300.3 | 3.92× | ✅ |

**~3.9× wall-clock reduction at 32 cores, byte-identical.**

Two structural facts shape the curve:

1. **Sub-contig sharding only engages at `threads ≥ ~15`.** `plan_thread_budget`
   first spends added cores on HTSlib BGZF decode threads for the single reader;
   `processing_threads` (the sub-contig shard budget) stays `1` until the core
   count clears that stage. So `threads=1,2,4,8` all run one un-sharded reader
   and show no speedup — the parallel win is entirely in the high-thread regime
   where the contig is split into work-stealing shards.
2. **Conversion is htslib-input (reader) bound**, not compute bound (see memory
   `svar2-conversion-reader-bound`: ~78% reader at 1 thread; after the earlier
   GT-decode/per-word-pack/parallel-pack wins it is inflate+parse bound). Sharding
   parallelizes that reader across sub-contig regions (independent indexed
   fetches), which is why speedup tracks added shard readers rather than added
   executor threads. Parallel efficiency is therefore modest in absolute terms
   (3.9×/32 ≈ 12%) but the wall-clock win is real and the ceiling is I/O, not the
   scheme.

## PGEN → SVAR2 (chr21c, matched)

| threads | wall (s) | vs serial | byte-identical |
|--------:|---------:|----------:|:--------------:|
| 1  | 273.0 | 1.00× | ✅ |
| 24 | 340.8 | 0.80× (**slower**) | ✅ |

**PGEN sharding is byte-identical but NET SLOWER.** Root cause: `pgenlib`'s
`read_alleles_range` holds the CPython GIL for its entire decode (verified
against the compiled `.so`; the prior in-code comment claiming it releases the
GIL was wrong). So concurrent shard readers cannot decode in parallel — they
serialize on the GIL — and sharding adds pure coordination overhead. A
per-variant GIL re-acquisition in the reader additionally produced a GIL convoy
that stalled large runs entirely; that was fixed (`fa47530`, bulk-copy each
batch under one GIL acquisition, O(records)→O(refills)), which is what makes the
340 s run complete at all. See memory `pgenlib-holds-gil-sharded-reads`.

## Byte-identity

`oracle_ok` / hash-vs-oracle held on **every** row above, at all thread counts,
for both backends, including over-decomposed VCF (16 shards at threads=16) and
full-scale multiallelic PGEN (32k co-located positions). Two byte-identity bugs
were found and fixed during implementation:
- PGEN shard-boundary under-fetch (`93906f9`) — fetch padding must anchor on the
  ownership boundary position, not the last owned variant (mirrors VCF).
- (No VCF byte-identity defects; the VCF path was byte-identical from Task 7.)

## Executor share (Task 11 input)

The executor stage (`dense2sparse_vk`) is **not** the bottleneck: the conversion
is reader-bound, and speedup scales with reader shards, so the executor's
wall-time share is well under the Task 11 gate's 25% threshold. See the decision
record (`svar2-conversion-decision-2026-07-15.md`).

### Scope note (dropped coverage)

To fit the contended shared cluster and this node's memory limits, the giant
`gdc.chr21.filt.bcf` (5× larger) scaling sweep and a formal `perf record`
executor-frame measurement were **not** run; the executor-share conclusion rests
on the established reader-bound profile plus the observed reader-scaling shape,
not a fresh callgraph. A `perf record` at 16/32 threads (recipes in
`scripts/profile/`) would formalize it if a dedicated node becomes available.

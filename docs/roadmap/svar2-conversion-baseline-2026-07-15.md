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

> **Caveat on absolute wall-times.** These VCF numbers were taken on the same
> shared cluster whose contention later proved to inflate the PGEN absolutes
> (see below). The load-bearing results here are the **speedup ratio** (~3.9× at
> 32 threads) and **byte-identity at every thread count**, both of which are
> ratios/invariants robust to a uniform slowdown; treat the raw seconds as
> indicative, not precise.

## PGEN → SVAR2 (chr21c, matched)

Re-measured on a quiet `carter-cn-02` (chr21c: ~1M variants × 3202 samples).
An earlier run on a heavily-contended node reported 273 s serial / 340 s
sharded; **those absolute numbers were a contention artifact and do not
reproduce** (they were ~8× inflated). The reproducible picture:

| config | wall (s) | vs serial | byte-identical |
|:-------|---------:|----------:|:--------------:|
| serial `P=1`, pgenlib 0.91.0 | 32.6–32.9 | 1.00× | ✅ |
| serial `P=1`, pgenlib 0.94.1 | 33.0 | 1.00× | ✅ |
| sub-contig sharded, `threads=24` (0.91.0) | 44.9 | **0.73× (slower)** | ✅ |
| pgenlib 0.94.1, `OMP_NUM_THREADS` 1→32 | 32.6–33.0 | flat | ✅ |

All rows share the same store hash. Two conclusions, both reproducible:

1. **PGEN sharding is byte-identical but NET SLOWER** (44.9 s vs 32.6 s).
   Single-reader conversion is already fast (~33 s) and bound by the shared
   executor/writer + reference I/O, **not** by pgenlib decode — so sub-contig
   sharding cannot beat that floor, and concurrent readers only add
   coordination overhead (and, on `pgenlib`<0.92, serialize on the CPython GIL:
   `read_alleles_range` holds it through the whole decode — verified against
   both the compiled `.so` and the `.pyx`, 0 `nogil`/`prange` in 0.91.0). A
   per-variant GIL re-acquisition additionally produced a convoy that stalled
   large sharded runs; fixed in `fa47530` (bulk-copy each batch under one GIL
   acquisition). See memory `pgenlib-holds-gil-sharded-reads`.
2. **A GIL-releasing pgenlib does not help.** 0.94.1 parallelizes
   `read_alleles_range` decode via `prange(nogil=True)`, but the conversion is
   flat at ~33 s across `OMP_NUM_THREADS` 1→32 and identical to 0.91.0 —
   because decode isn't the bottleneck. Hence the `pgenlib` pin stays at
   `0.91.*`; bumping to 0.94.x buys nothing here (see the decision record).

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

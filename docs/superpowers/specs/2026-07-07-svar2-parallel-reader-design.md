# SVAR2 parallel-reader prototype — design

**Date:** 2026-07-07 · **Branch:** `svar-2` · **Home:** `genoray` · **Status:** design, not yet started

> Follow-on to the profiling/optimization work in
> [`2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md`](2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md)
> (commits `0be7bee` reader micro-opts, `af8a2b0` results). Read that first —
> especially the perf findings — before starting here.

## Problem recap

VCF→SVAR2 conversion of a **single contig** is single-thread **reader-bound** and
uses only ~8 of 32 cores. After the `0be7bee` reader micro-optimizations
(raw-GT-buffer decode + per-word bit packing, 2.4–2.8× with byte-identical
output), the per-thread split is still ~83–93% reader / ~7–17% executor, and the
reader's remaining cost (gdc.chr21, 16007 samples) is:

- **BGZF decompression** `inflate_fast` ~31% + `crc32_z` ~14% ≈ **45%** — htslib,
  already multi-threaded but the decode threads inherit the `read-chr21` name so
  they show under that comm.
- **`bcf_get_format_values`** ~9% — htslib copying the raw GT FORMAT array out.
- **`read_next_chunk`** (presence bit packing) ~16% — the fused O(chunk·columns)
  loop, still on the single reader thread.
- **`Vec::from_iter` (nested)** ~7% — `record.format(b"GT").integer()` building the
  per-sample `Vec<&[i32]>` slice array once per record.
- Executor `dense2sparse_vk` transpose ~17%.

~24 cores sit idle. Goal: use them, keep output **byte-identical**, keep the
default (wheel) build behavior-correct.

## Current architecture (as of `af8a2b0`)

Per contig, `orchestrator::process_chromosome` spawns a 4-thread pipeline joined by
bounded channels, and `lib.rs` fans contigs across a rayon pool sized to
`concurrent_chroms`:

```
htslib ─▶ [reader thread] ─DenseChunk─▶ [executor thread] ─SparseChunk─▶ [chunk writer]
           read+decode+                  dense2sparse_vk                  + [long-allele writer]
           atomize+leftalign+            (transpose + cost-model          then rayon tile-merge
           pack BitGrid3                  routing + BANK spill)
```

Key files: `src/vcf_reader.rs` (`VcfChunkReader`, `decompose_current_record`,
`read_next_chunk`), `src/orchestrator.rs` (thread wiring), `src/executor.rs`
(`run_compute_engine`), `src/rvk.rs` (`dense2sparse_vk`, `classify_variant`,
`pack_variant` → `LongAlleleTableWriter`), `src/budget.rs` (`plan_thread_budget`),
`src/lib.rs` (rayon pool + dispatch), `src/types.rs` (`BitGrid3`, `DenseChunk`).

**Two facts that make the reader safe to parallelize:**

1. **The long-allele bank is executor-side.** `LongAlleleTableWriter` (sequential
   row-index assignment, which determines the on-disk LUT and the 32-bit keys) is
   touched only in `dense2sparse_vk` on the single executor thread. Parallelizing
   the *reader* does not touch bank determinism — the byte-identical hazard that
   would exist for a parallel *executor* does not apply here.
2. **The reader's heavy work is O(columns) per record/atom**, and the presence bit
   for `(variant vi, column col)` depends only on that atom and its own raw GT.

## Reader hot work, and the insight that fuses decode+pack

Today `decompose_current_record` decodes a full `gt: Vec<i32>` (one allele index
per column, 25 KB/record for gdc, Rc-shared across a record's atoms), and
`read_next_chunk` later packs presence bits as `gt[col] == source_alt_index`.

The gt vector is unnecessary. Presence collapses to a single compare on the **raw**
BCF GT int:

```
present(col) = decode(raw_gt[col]) == src            (src = atom.source_alt_index)
             = (raw_gt[col] >= 2 && (raw_gt[col] >> 1) - 1 == src)
             = (raw_gt[col] >> 1) == (src + 1)        // one shift + compare, no branch
```

So an atom needs only `Rc<Vec<i32>>` of the record's **raw** GT buffer plus its
`src`. This deletes the 25 KB gt-vector alloc/decode entirely and turns packing
into a per-atom, per-column shift-compare that is trivially parallel across atoms.

## Approach

Two independent levers. Do them in this order; measure after each.

### Lever 0 (cheap, orthogonal, do first): more htslib decode threads

BGZF is ~45% of the gdc reader and htslib already parallelizes block
decompression, but `plan_thread_budget` caps `htslib_threads` at
`MAX_HTSLIB_THREADS = 4` (comment: "diminishing past 4"). That ceiling was
presumably tuned on multi-contig, typical-width data; gdc's 16007-sample records
mean very large BGZF blocks where more decode threads may still pay. **Experiment
before the refactor:** bump the cap (8, 12, 16) for few-contig / many-idle-core
workloads and re-measure gdc. One-line change; an ~18-min run per trial. If it
closes a big fraction of the 45%, it changes how much Lever 1 needs to do. Keep
whatever wins behind the budget logic (only widen when idle cores exist).

### Lever 1 (the prototype): parallel fused decode+pack within the chunk

Keep **all ordering logic sequential** — htslib record iteration, `atomize_record`,
`left_align`, the reorder `BinaryHeap`, and chunk cutting stay exactly as they are,
so the atom set and chunk boundaries (hence output) are unchanged. Parallelize only
the O(chunk·columns) fused decode+pack.

Restructure:

- **`decompose_current_record`:** extract the raw GT buffer once per record as
  `Rc<Vec<i32>>` (a single flat `Vec<i32>` of `n_samples * width`, from the FORMAT
  buffer — ideally the flat buffer, not the per-sample `Vec<&[i32]>`; see risks).
  Atomize + left-align as today. Each `PendingAtom` carries `Rc<raw_gt>` + `src`
  instead of the decoded `gt`. **No per-column work here.**
- **`read_next_chunk`:** gather the chunk's atoms (sequential, ordered), allocate
  `DenseChunk.genos` (`BitGrid3`), then **rayon-parallel over variant-index blocks**
  compute each atom's row: for row `vi`, for each `col`, set bit iff
  `(raw_gt[col] >> 1) == src + 1`. Reuse the committed per-u64-word accumulation
  inside each row.

**Disjoint parallel writes into `BitGrid3.words: Vec<u64>`.** Row `vi` occupies bits
`[vi*columns, (vi+1)*columns)`. Partition variants into blocks whose bit-boundaries
are u64-aligned — a block boundary at variant `v_b` is word-aligned iff
`v_b*columns % 64 == 0`, i.e. `v_b` a multiple of `64/gcd(columns,64)`
(`columns = n_samples*ploidy` is even; for gdc `columns=32014` → every 32nd
variant; for germline `columns=6404` → every 16th). Give each rayon task a
word-disjoint slice via `split_at_mut`/`chunks_mut` so there are no shared boundary
words and no atomics. (Alternative: pad each row to a byte/word boundary in
`BitGrid3` so every row is independent — simpler partitioning but changes the
`BitGrid3` layout and the `dense2sparse_vk` stride/`popcount_plane`; more blast
radius. Prefer the word-aligned-block partition first.)

Optionally also fuse the GT decode that currently happens per-record into this same
parallel pass (it already is, since presence reads raw GT directly).

### Thread budget (blocking prerequisite for Lever 1)

`lib.rs` builds the rayon pool with `num_threads(concurrent_chroms)` — **1 for a
single contig** — and `process_chromosome` runs inside `pool.install`. A nested
`par_iter` from `read_next_chunk` would therefore run on a 1-thread pool and get
**zero** parallelism. `budget.rs` must be reworked to hand a single-/few-contig
workload a **processing pool** sized to the idle cores
(`usable - pipeline - htslib*concurrent`). Options: (a) a second, separately-sized
rayon pool that the reader `install`s its packing `par_iter` into; (b) fold packing
parallelism into the existing pool with a smarter split between contig-level and
intra-contig parallelism. Encode the new split in `plan_thread_budget` with tests
(mirror the existing low-end/high-end/clamp cases), so the multi-contig path stays
unchanged and only spare cores feed the processing pool.

## Amdahl / expected ceiling

If Lever 1 parallelizes the ~16% packing (+ folds in the ~7% GT-buffer build and
the O(columns) decode) across many cores, and Lever 0 shrinks the ~45% BGZF, the
serial floor becomes htslib record iteration + `bcf_get_format_values` (~10–15%) +
the executor transpose (~17%, still single-threaded — a *separate* future lever:
the executor could be parallelized across chunks, but that DOES hit bank
determinism and is out of scope here). Plausible target: another ~2× on gdc
(18 min → ~8–10 min) from Lever 1+0, before the executor becomes the new ceiling.
Measure — do not assume.

## Correctness & verification (non-negotiable: byte-identical output)

- Oracle: the store content-hash script `storehash.sh` +
  `oracle.<dataset>.hash` in `/carter/users/dlaub/svar_bench` (sha256 of every
  store file). Optimized store MUST equal the oracle on **both** germline and gdc.
- `cargo test --no-default-features --features conversion` (the link-flag gotcha —
  default `extension-module` breaks the test binary; see
  [[genoray-cargo-test-no-default-features]]) — 185 tests today.
- `pytest tests -m "not network"` — 525 tests today.
- Add a targeted proptest: parallel-packed `BitGrid3` == sequential-packed for
  random (n_samples, ploidy, atoms, GT patterns incl. missing/vector-end), across
  block-partition boundaries.

## Benchmark harness (already set up)

`/carter/users/dlaub/svar_bench/`: filtered BCFs (`chr21.filt.bcf`,
`gdc.chr21.filt.bcf`), `run_svar2.py`, `storehash.sh`, oracle hashes, and perf
captures (`perf.germline.data` pre-opt, `perf.gdc.opt.data` post-opt). Reference:
`/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`. **Run on a dedicated
`carter-compute` node** (`sbatch -p carter-compute -c 32 --mem=128G` holder +
`srun --jobid=<id> --overlap`) — the login node has 2 cores under heavy load.
Build the profiling wheel for perf:
`RUSTFLAGS="-C force-frame-pointers=yes" maturin develop --profile profiling`;
restore with `maturin develop --release` after.

## Risks / open questions

- **Flat vs per-sample GT buffer.** `record.format(b"GT").integer()` returns
  `Vec<&[i32]>` (per-sample slices, ~7% `from_iter`). For the raw-buffer Rc we want
  the flat `n_samples*width` `Vec<i32>` without the per-sample split — likely needs
  the lower-level `bcf_get_format_int32` via `rust_htslib::htslib` sys bindings and
  a `width` (max ploidy) from the buffer length. Verify the missing/vector-end
  sentinels survive (`(raw>>1)==src+1` already treats 0/1/`i32::MIN` as absent).
- **Non-uniform ploidy / buffer width > `self.ploidy`.** The flat index is
  `s*width + p`; column `s*ploidy + p` must map correctly and clamp `p < width`.
  Preserve the current `raw.get(p)`-style bounds behavior.
- **rayon nesting / pool starvation** (see Thread budget) — the main structural
  work; get this right or Lever 1 shows no speedup.
- **Small chunks / few columns**: parallel overhead can exceed the win. Gate the
  `par_iter` behind a size threshold; fall back to the committed sequential
  per-word packing.
- **Does Lever 0 alone suffice?** If bumping htslib threads recovers most of the
  45% BGZF, Lever 1's ROI drops. Measure Lever 0 first.

## Suggested first steps (for the fresh session)

1. Reproduce the baseline on a compute node: build profiling wheel, confirm
   germline oracle hash + `run_svar2.py chr21.filt.bcf … 32` ≈ 36.5s.
2. **Lever 0 experiment**: raise `MAX_HTSLIB_THREADS`, re-measure gdc; record.
3. Prototype **Lever 1** behind the thread-budget rework; verify byte-identical +
   tests; measure germline (fast) then gdc.
4. Update the roadmap `svar-2.md` if the pipeline/thread-model description changes.

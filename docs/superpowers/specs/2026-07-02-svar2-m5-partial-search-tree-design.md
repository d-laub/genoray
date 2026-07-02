# SVAR 2.0 — M5 (partial): format-independent overlap search core (design)

> Partial spec for roadmap milestone **M5** in
> [`docs/roadmap/svar-2.md`](../../roadmap/svar-2.md).
> Supplements: [`architecture.md`](../../roadmap/architecture.md#query-path),
> [`data-model.md`](../../roadmap/data-model.md#overlap-queries-and-deletions).
> Branch off `svar-2`, own worktree; can proceed in parallel with M3.

## Context

M5 is `(range, sample)` overlap queries. It has two separable halves:

1. **A format-independent algorithmic core** — a static search tree over a sorted `u32`
   position array plus the overlap lower-bound/upper-bound resolution that handles
   deletions spanning the query start. This depends only on in-memory arrays.
2. **Disk integration** — reading a real SVAR2 contig (positions/offsets/`max_del` across
   the `var_key`/`dense` × `snp`/`indel` sub-streams), merging results in position order,
   and gathering per-sample genotypes.

This spec covers **only half 1**. It is deliberately decoupled so it can be built and
proptested against synthetic arrays *before* M3 finalizes the on-disk format and *before*
M5's `max_del.npy` producer exists — the two halves touch disjoint files.

The existing SVAR 1.0 reader already implements this shape in Python
([`_var_ranges.py::var_ranges`](../../../python/genoray/_var_ranges.py): `np.searchsorted`
for LB/UB, then numba `_forward_sub_scan` / `_backward_sub_scan` to trim to true
overlaps). This spec generalizes that into the Rust core, swapping `searchsorted` for the
cache-friendly [left-tree static search tree](https://curiouscoding.nl/posts/static-search-tree/#left-tree).

## Scope

**In:**

- A **static search tree** built over a sorted, ascending `u32` array, exposing
  `lower_bound(x)` (first index with `arr[i] >= x`) and `upper_bound(x)` (first index with
  `arr[i] > x`), following the left-tree layout for cache-friendly probes.
- An **overlap-range resolver**: given sorted `v_starts: &[u32]`, a `max_region_length:
  u32` bound (the max-deletion span, passed as a **parameter** — not read from disk), and a
  query `[q_start, q_end)`, return the `[s_idx, e_idx)` index range of variants that truly
  overlap, mirroring `var_ranges`:
  - LB = search on `v_starts + max_region_length` for `q_start`;
  - UB = search on `v_starts` for `q_end`;
  - forward sub-scan for the first `q_start < v_end`, backward sub-scan for the last, where
    `v_end` is reconstructed from `v_start` and the deletion length.
- Correctness validated by **proptest against a brute-force linear-scan oracle** over
  random sorted arrays, random queries, and random deletion lengths (including the SNP case
  `max_region_length == 0`, empty arrays, and no-overlap queries).

**Out (deferred to full M5 / M6):**

- Reading `positions.bin` / `offsets.npy` / `max_del.npy` from a SVAR2 directory (needs
  the M3 format contract and the M5 `max_del` producer).
- Producing `max_del.npy` at conversion time (the write-side other half of M5).
- Combining results across the up-to-six sub-streams (`var_key`/`dense` × `snp`/`indel`)
  and the sorted-union merge (that union is M6 decode territory).
- Per-sample genotype gather and the `SparseVar2` Python class.
- Batched/multi-query vectorization (SVAR 1.0 does many ranges at once; the core here is
  single-query — batching is a thin wrapper added during integration).

## Design

### Module

A self-contained Rust module (e.g. `src/search.rs`) with no dependency on `layout.rs`,
`merge.rs`, or any on-disk type. Inputs are borrowed slices; outputs are index ranges.

### `v_end` reconstruction

A SNP spans one base (`v_end = v_start + 1`); an indel deletion of length `d` spans
`v_end = v_start + 1 + d` (0-based, exclusive), matching the SVAR 1.0 formula
`v_end = POS - min(ILEN, 0)` after the 0-based shift. The resolver takes `v_ends` (or the
data to compute them) alongside `v_starts`; how deletion length reaches the core (explicit
`v_ends` slice vs. `(v_starts, ilens)`) is an implementation detail to settle in the plan.

### Why a search tree over `searchsorted`

The static search tree amortizes branchy binary search into a flat, prefetch-friendly
layout — the win the [reference post](https://curiouscoding.nl/posts/static-search-tree/#left-tree)
documents for repeated queries against a fixed sorted array, which is exactly the
per-contig position sidecar accessed across many range queries. The tree is **built once
per sorted array** and reused across queries.

### Overlap algorithm (single query)

Mirrors `var_ranges` in [`_var_ranges.py`](../../../python/genoray/_var_ranges.py):

1. `lb = tree_over(v_starts + max_region_length).lower_bound(q_start)` — leftmost variant
   whose start could still reach `q_start` given the max deletion span.
2. `ub = tree_over(v_starts).lower_bound(q_end)` — one past the last variant starting
   before `q_end`.
3. Forward sub-scan `[lb, ub)` for the first `i` with `q_start < v_ends[i]` → `s_idx`.
4. Backward sub-scan `(lb, ub]` for the last such `i` → `e_idx` (exclusive).
5. Empty result when `s_idx == ub` (no overlap).

Note the two searches are over two different key arrays (`v_starts + max_region_length` and
`v_starts`); whether that is two trees or one tree plus an offset probe is a plan-time
decision.

## Testing

- **Oracle proptest:** brute-force `O(n)` overlap over random `(sorted v_starts, deletion
  lengths, query)` must match the tree+scan result exactly. This is the primary
  correctness gate.
- **Edge cases:** empty array; single element; `max_region_length == 0` (pure SNP stream,
  reduces to half-open `[lb, ub)`); query entirely left/right of all variants; deletion
  that starts before the query and spans into it; adjacent-but-non-overlapping
  (`v_end == q_start`).
- **Search-tree unit tests:** `lower_bound`/`upper_bound` against `slice::partition_point`
  as an independent reference, across sizes that exercise the tree's block boundaries.

## Worktree & dependencies

- Worktree: `.claude/worktrees/svar-2-m5-search-core`, branch off `svar-2`, PR into
  `svar-2`. Install prek hooks before committing.
- **No dependency on M3.** Operates on in-memory arrays; touches only the new `search.rs`
  (+ its tests). Disk integration — reading M3's finalized sidecars and M5's own
  `max_del.npy` — is a follow-up once both this core and M3 have merged.
- When the write-side of M5 (the `max_del.npy` producer) is specced, it pairs with this
  core: this module *consumes* `max_region_length`; the producer *emits* it per
  `(contig, sample, ploid)`.
</content>
</invoke>

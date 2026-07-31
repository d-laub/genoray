# SearchTree construction audit (closes #148)

**Date:** 2026-07-31
**Issue:** [#148](https://github.com/d-laub/genoray/issues/148)
**Predecessors:** #144 (`find_ranges` var_key hoist), #145 (dense union + dense
class tables + `svar2_slice` hoist)

## Problem

`SearchTree::new` is `O(n)` in a channel's variant count and allocates two
`Vec<u32>`s. It is region-independent: the tree over a channel's positions does
not depend on the query. Building one per region therefore multiplies a fixed
cost by the region count for no benefit — the `O(regions x channels)` shape that
#144 and #145 removed from `find_ranges` and `svar2_slice`.

This spec audits every remaining construction site and closes the one #145 left
behind, plus two adjacent instances of the same defect class (a
region-independent structure rebuilt when it could be built once or not at all).

## Audit

Every non-test construction site in `src/`, and its verdict:

| Site | Shape | Verdict |
| --- | --- | --- |
| `overlap_batch_impl` -> `vk_slice` -> `spine::gather_keys` (`gather.rs:246`) | `2 x R x S x P` builds | **DEFECT — #148** |
| `ContigReader::max_deletion_len` (`reader.rs:347`) | builds a whole `DenseUnion` to read a scalar field | **DEFECT — adjacent** |
| `read_ranges` (`gather.rs:951`) | builds `DenseUnion` twice per query | **DEFECT — adjacent** |
| `find_ranges` / `find_ranges_haps` (`gather.rs:365`, `466`) | one index per column/contig, above the region loop | clean (#144/#145) |
| `svar2_slice::gather_var_key` / `gather_dense` (`svar2_slice.rs:554-592`) | one index per column/class, above the region loop | clean (#145) |
| `py_query_ranges::find_ranges_header` (`py_query_ranges.rs:281-296`) | one index per dense channel per call | clean (#145) |
| `svar1_query::var_ranges` (`svar1_query.rs:60`) | one tree per contig batch | clean |
| `DenseUnion::index` (`union.rs:38`) | built only by callers that search it | clean |
| `oracle::overlap_sample` (`oracle.rs:61`) | `2 x P` builds on ONE region | not a defect (see below) |

### Why `overlap_sample` is not in scope

It is the documented tree-per-query reference implementation, takes a single
region, and its per-ploid builds are each over a distinct column — none are
redundant. After Fix 1 it becomes the sole caller of
`vk_slice`/`spine::gather_keys`, which is the correct end state: the oracle
keeps an implementation independent of the optimized path it cross-checks.

### The `#148` defect in detail

`overlap_batch_impl` loops region -> sample -> ploid around
`reader.vk_slice(col, s, p, qs, qe)`. `vk_slice` calls `spine::gather_keys`
twice (snp + indel sub-streams), and `gather_keys` builds a fresh `SearchTree`
**and** a fresh `v_ends` vector on every call (`spine.rs:136-141`). Cost is
`2 x R x S x P` tree builds where `2 x S x P` suffice.

The #145 design doc excluded the spine paths as "not region-looped here"
(`2026-07-31-svar2-overlap-index-hoist-design.md:224`). That rationale is
inaccurate for `overlap_batch_impl`, which is region-looped. The exclusion was
still right for #145's scope; the reason was not.

## Design

### Fix 1 — route `overlap_batch_impl` through the search/gather split

`gather.rs:944` already documents `read_ranges(reader, regions, None)` as
byte-identical to `overlap_batch`, and `tests/test_ranges_split.rs:450` asserts
it at the Python-binding layer. So the batch path does not need a second
hoisting mechanism — it needs to stop reimplementing the one that exists.

Make `gather_ranges` generic over `T: DenseSrcElem`, following the
`gather_haps_readbound` / `gather_haps_readbound_impl` pattern already used
twice in this file:

```rust
fn gather_ranges_impl<T: DenseSrcElem>(
    reader: &ContigReader,
    rb: &RangesBundle,
    dense: &DenseUnion,
) -> BatchResult
```

- `pub fn gather_ranges` = `gather_ranges_impl::<KeyRef>` with a freshly built
  union (signature and behaviour unchanged).
- `overlap_batch` = `find_ranges_with(.., None, &dense)` +
  `gather_ranges_impl::<KeyRef>(.., &dense)`.
- `overlap_batch_src` = the same with `SrcKeyRef`, which is what restores
  `vk_src` (via `T::split`) and `dense_src` (via `T::CARRIES_SRC`).

The duplicated inner loop in `overlap_batch_impl` is deleted, not rewritten.
Tree builds drop to `O(S x P)`.

Two accepted consequences:

- **Threading.** The batch path inherits `find_ranges_haps`'s rayon fan-out
  above `PAR_COLUMN_THRESHOLD` (64) haps. Deterministic: haps write disjoint
  `par_chunks_mut` slices and the only reduction (`max_end_keys`) is a max that
  `overlap_batch` discards. `overlap_batch`'s "Single-threaded" doc comment is
  stale and must be updated. Consequence for tests: the
  `search_tree_build_count` thread-local is only observable below the
  threshold, so guards must use a small fixture.
- **Memory.** A `RangesBundle` (~32 B/hap) is materialized where the old loop
  streamed. `_overlap_batch` is a gvl test-oracle entry point, not a hot read
  path, so this is acceptable.

The rejected alternative — hand `gather_keys` a prebuilt `OverlapIndex`, per
the issue's suggested approach — requires inverting `overlap_batch_impl` to
column-outer (its `vk_off` CSR is region-major) plus a region-major
reassembly pass, i.e. a *third* var_key gather implementation alongside
`gather_vk` and the tuned inline merge in `gather_haps_readbound_impl`.
Hoisting all `S x P` column indices while staying region-outer is also
rejected: peak memory becomes `O(total var_key calls on the contig)`.

### Fix 2 — build `DenseUnion` once per query

`find_ranges` and `gather_ranges` each build their own union, so `read_ranges`
(public API) builds it twice — an `O(dense variants log dense variants)` sort
paid twice per query. Thread the union explicitly:

- private `find_ranges_with(reader, regions, samples, dense: &DenseUnion)`
- private `gather_ranges_impl(reader, rb, dense: &DenseUnion)`
- public `find_ranges` / `gather_ranges` build their own and delegate
  (signatures unchanged)
- `read_ranges` / `overlap_batch` / `overlap_batch_src` build ONE and pass it to
  both halves

Explicit plumbing rather than a `OnceLock` cache on `ContigReader`: caching
would hold an `O(dense variants)` derived structure for the reader's whole
lifetime, which is a memory regression on the population-scale write path that
never queries it.

### Fix 3 — `max_deletion_len` stops building a union

`DenseUnion::max_del` is unconditionally `self.dense_indel_max_del`
(`union.rs:100`), so

```rust
vk.max(self.dense_union().max_del())
```

is exactly

```rust
vk.max(self.dense_indel_max_del)
```

with an `O(n log n)` sort over every dense variant on the contig in between.
Read the field. This removes a whole union build from
`py_query_ranges::find_ranges_header` — the chunked path gvl's writer uses —
which currently builds one union for the overflow preflight and a second one
nine lines later for the real work. Update `union.rs:33-37`, which names
`max_deletion_len` as a union-building caller.

## Testing

Complexity guards, in the established style: exact zero growth in
`search::search_tree_build_count()` between a 1-region and a 16-region query,
with a `cost_one > 0` positive control so the assertion cannot pass vacuously.
Fixture is `synth_reader_wide` (2 samples x 2 ploidy = 4 columns, well under
`PAR_COLUMN_THRESHOLD`) so the serial path runs on the test thread and the
thread-local stays observable.

1. `overlap_batch_tree_builds_do_not_scale_with_regions`
2. `overlap_batch_src_tree_builds_do_not_scale_with_regions` — a separate
   monomorphization, so it needs its own guard
3. `assert_eq!(overlap_batch(&reader, &regions), read_ranges(&reader, &regions, None))`
   — `BatchResult: PartialEq`, so this is a direct byte-identity assertion at
   the Rust level, complementing the existing Python-binding dict comparison
4. A value test for `max_deletion_len` on a fixture carrying both a long
   var_key indel and a long dense deletion, pinning the max across both
   channels

Existing coverage that must keep passing unchanged:
`tests/test_batch.rs` (`test_overlap_batch_matches_overlap_sample`,
`prop_overlap_batch_matches_overlap_sample`),
`tests/test_ranges_split.rs`, `tests/test_field_provenance.rs`
(`overlap_batch_src` provenance).

## Non-goals

- No change to `oracle::overlap_sample`, `vk_slice`, or `spine::gather_keys`
  beyond doc comments recording that the oracle is now their only caller.
- No removal of `overlap_batch` in favour of `read_ranges`. It keeps a distinct
  Python binding (`_overlap_batch`) that gvl's test oracles call, and
  `overlap_batch_src` has no `read_ranges` counterpart.
- No `DenseUnion` build counter analogous to `search_tree_build_count`. Fix 2's
  correctness is structural (one binding passed to both halves), not something
  worth a new observability hook.

## Public API impact

None. No name reachable from `import genoray` without an underscore changes in
signature, return shape, dtype, or semantics — `read_ranges` returns the same
bytes, just faster. `skills/genoray-api/SKILL.md` therefore needs no update.

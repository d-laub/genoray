# Hoist every per-region search-tree build behind one `OverlapIndex`

Date: 2026-07-31
Issue: [#145](https://github.com/d-laub/genoray/issues/145)
Follow-up to: #144

## Problem

#144 removed the O(regions x columns) `SearchTree` rebuild from
`query::find_ranges` by hoisting per-column state into `VkColumnIndex`. Three
sibling sites still rebuild a search tree once per region.

| Site | Rebuilt per call | Callers that loop over regions |
| --- | --- | --- |
| `VkColumnIndex` via `vk_snp_index` / `vk_indel_index` (`src/query/reader.rs:271`, `:292`) | `SearchTree` + a `v_ends` `Vec` sized to the column | `svar2_slice::gather_var_key` -> `region_hits` (`src/svar2_slice.rs:707`, `:737`) |
| `DenseUnion::overlap` (`src/query/union.rs:31`) | `SearchTree` (`v_ends` / `max_del` already cached on the struct) | `find_ranges` (`src/query/gather.rs:479`), `overlap_batch_impl` (`src/query/gather.rs:233`), py `find_ranges` (`src/py_query_ranges.rs:281`), `oracle::overlap_sample` (single region) |
| `ContigReader::dense_snp_overlap` / `dense_indel_overlap` (`src/query/reader.rs:333`, `:350`) | `SearchTree` + a `v_ends` `Vec` sized to the class table | `src/query/gather.rs:488`, `:492`; `src/py_query_ranges.rs:288`, `:292`; `svar2_slice::gather_dense` (`src/svar2_slice.rs:584`, `:595`) |

The var_key row is #145 as filed: `gather_var_key` is column-outer, but the
`overlap_range` closure it hands `region_hits` is
`|qsw, qew| reader.vk_snp_index(col_src).overlap(qsw, qew)`, so a fresh index —
including `SearchTree::new` — is constructed inside the per-region loop, per
column. The dense rows are the same defect on a channel #144 did not touch, and
they are shared with the read path: `find_ranges` still pays three tree builds
per region (union, dense SNP, dense indel). #144's complexity guard budgets for
exactly that (`allowed growth = 2 * Δregions`), so the leak is currently
*asserted* rather than fixed.

None of this is a regression. The pre-#144 methods rebuilt the same state on
every call; #144 simply did not extend the fix past `find_ranges_haps`.

Impact scales with regions per request, so it matters most for batch and
read-bound callers.

### Constraint that shapes the fix

`dense_union()` has two classes of caller. `find_ranges`, `overlap_batch_impl`,
py `find_ranges` and `oracle::overlap_sample` call `overlap` on the result.
`gather_ranges` (`src/query/gather.rs:549`) and `ContigReader::max_deletion_len`
(`src/query/reader.rs:324`) do not — `gather_ranges` advertises "Contains NO
`SearchTree::new`" in its docstring, and `max_deletion_len` only wants
`max_del`. Making the union's tree eager inside `dense_union()` would charge
both of those a build they never use. The tree must therefore be built by an
explicit, separately-constructed index, not by the union constructor.

## Design

### 1. One index type

`VkColumnIndex` already models "region-independent search state over
`(positions, v_ends, max_del)` at base offset `o0`". The dense class channels
are the same thing with `o0 = 0`; the dense union is the same thing with
`v_ends` already materialized on `DenseUnion`. Rename it `OverlapIndex` and make
`v_ends` a `Cow`:

```rust
pub(crate) struct OverlapIndex<'a> {
    /// Absolute base offset of this channel in the packed arrays (`0` for the
    /// dense class tables and the union).
    o0: usize,
    /// `None` for an empty channel — `SearchTree` / `overlap_range` are not
    /// defined over an empty position array.
    inner: Option<(SearchTree, Cow<'a, [u32]>)>,
    max_del: u32,
}

impl OverlapIndex<'_> {
    pub(crate) fn overlap(&self, q_start: u32, q_end: u32) -> Range<usize>;
}
```

Constructors:

- `ContigReader::vk_snp_index(col) -> OverlapIndex<'static>` — body unchanged,
  computed `v_ends` becomes `Cow::Owned`.
- `ContigReader::vk_indel_index(sample, p) -> OverlapIndex<'static>` — same.
- `ContigReader::dense_snp_index() -> OverlapIndex<'static>` — the body of the
  deleted `dense_snp_overlap`, `o0 = 0`, `max_del = 0`.
- `ContigReader::dense_indel_index() -> OverlapIndex<'static>` — the body of the
  deleted `dense_indel_overlap`, `o0 = 0`, `max_del = self.dense_indel_max_del`.
- `DenseUnion::index(&self) -> OverlapIndex<'_>` — borrows the union's existing
  `v_ends` as `Cow::Borrowed`, so no per-query copy. `dense_union()` itself stays
  tree-free.

`dense_snp_overlap` and `dense_indel_overlap` are deleted; `DenseUnion::overlap`
moves onto the index. The `Cow` is what lets one type serve both the channels
that compute `v_ends` on the fly and the union that already stores it, without
either a per-query `O(dense variants)` copy or a second near-identical struct.

Rejected alternative: leave `VkColumnIndex` untouched and add `DenseClassIndex`
plus `DenseUnionIndex`. That is three copies of the same six-line `overlap` body
for no benefit.

### 2. Hoist at every call site

Every site becomes "build the index once, then map over regions".

- `src/query/gather.rs:479-493` (`find_ranges`) — `dense.index()`,
  `reader.dense_snp_index()`, `reader.dense_indel_index()` built once, above the
  three `regions.iter().map(...).collect()` calls.
- `src/query/gather.rs:233` (`overlap_batch_impl`) — union index built once
  above the per-region `map`.
- `src/py_query_ranges.rs:281-293` — same three indices as `find_ranges`.
- `src/query/oracle.rs:54` — `dense.index().overlap(q_start, q_end)`. Single
  region, so cost is unchanged; the edit is to keep one call convention.
- `src/svar2_slice.rs` — `gather_var_key`'s injection point changes from a range
  to an index:

  ```rust
  // was: overlap_range: impl Fn(usize, usize, usize, u32, u32) -> Range<usize>
  column_index: impl Fn(usize, usize) -> OverlapIndex<'static>,   // (s_orig, p)
  ```

  built once inside the column loop, before `region_hits`:

  ```rust
  let ix = column_index(s_orig, p);
  let hits = region_hits(
      positions, regions, query_regions, overlap,
      |qsw, qew| ix.overlap(qsw, qew), &v_end_of,
  );
  ```

  Call sites in `slice_genos_inner` become
  `|s_orig, p| reader.vk_snp_index(s_orig * ploidy + p)` and
  `|s_orig, p| reader.vk_indel_index(s_orig, p)`, mirroring `find_ranges_haps`
  (`src/query/gather.rs:393-394`). The `col_src` parameter is dropped — it is
  derivable from `(s_orig, p)`, and both ignored-argument placeholders go with
  it.

- `src/svar2_slice.rs:576-597` — the two `gather_dense` calls get
  `reader.dense_snp_index()` / `reader.dense_indel_index()` built once at the
  call site in `slice_genos_inner`, passed as `|qsw, qew| ix.overlap(qsw, qew)`.
  `gather_dense` calls `region_hits` exactly once per class, so hoisting to the
  call site is sufficient; its signature does not change.

`region_hits` itself is unchanged.

### 3. Documentation corrections in the same pass

- `src/query/union.rs:30-31` — "Builds a fresh search tree over `positions`
  (cheap; one per region in a batch)" is no longer true.
- `src/query/gather.rs:478` — "Region-independent union; `overlap` builds one
  SearchTree per region".
- `src/svar2_slice.rs` — `region_hits` and `gather_var_key` docstrings name the
  old `overlap_range` parameter.
- `src/query/reader.rs:243-246` — `VkColumnIndex`'s doc comment, which describes
  only the var_key channel, generalizes to the renamed type.

## Correctness

Behavior is identical by construction: the same `(positions, v_ends, max_del,
q_start, q_end)` reach `overlap_range`; only the point at which the tree is
built moves. Outputs are byte-identical.

Existing coverage is the net:

- `tests/test_svar2_slice.rs` byte-parity tests
  (`slice_full_coverage_is_byte_identical_genos`,
  `preserve_identity_slice_is_byte_parity`,
  `slice_one_sample_subset_decodes_equivalently_to_the_source`,
  `the_three_overlap_modes_are_distinguishable`).
- `tests/test_ranges_split.rs`, `tests/test_batch.rs`,
  `tests/test_readbound_gather.rs` for the read paths, plus
  `query::oracle::overlap_sample` parity.

## Tests

Both guards use `search::search_tree_build_count()` — the thread-local
`SearchTree::new` counter already used by
`test_find_ranges_tree_builds_do_not_scale_with_regions`.

**Tighten the existing guard.**
`tests/test_ranges_split.rs:443` currently allows `2 * Δregions` growth for the
two legitimately-per-region dense trees. With all three sites hoisted, no
channel builds a tree per region, so the assertion becomes exact equality:
`cost_many == cost_one`. The allowance is removed, not widened.

**New slice guard.** `tests/test_svar2_slice.rs` gets a sibling asserting the
same exact equality across `slice_contig_genos` with 1 region vs. 16.
`src/svar2_slice.rs` contains no rayon, so the counter stays observable on the
test thread.

**New wide fixture for that guard.** The existing `fixture_records()` populates
only 1 of 4 var_key SNP columns and 1 of 4 indel columns. Following the lesson
of commit 47be396 — a barely-discriminating guard guards little — the new test
gets its own fixture:

- one single-carrier SNP and one single-carrier indel per hap column (single
  carrier so the cost model routes them VarKey, not Dense), so all four var_key
  columns of both classes are populated; **and**
- one multi-carrier (x=2) SNP and one multi-carrier indel, so both dense class
  tables exist. Without these the slice guard would not exercise the dense
  hoist at all — the dense channels would early-return on an absent table.

Against unfixed code that fixture grows by 4 var_key SNP + 4 var_key indel +
1 dense SNP + 1 dense indel = 10 tree builds per region, against an allowance
of 0. It is a separate fixture so the byte-parity tests' expectations are
untouched.

The read-path guard in `tests/test_ranges_split.rs` already exercises the union
and dense-indel channels on `synth_reader_wide` (its current allowance of 2 is
exactly those two builds per region), so tightening it to 0 covers the union and
dense-class hoists on that side.

Both guards are verified red-then-green by stashing the fix, per the procedure
recorded in 47be396's commit message.

No separate benchmark. The build counters are the direct measurement of the
complexity claim; a wall-clock benchmark would measure the same thing less
precisely.

## Build verification

- `cargo test --no-default-features` (the pyo3 test binary fails to link
  otherwise) with `CARGO_TARGET_DIR` off NFS.
- `cargo check --no-default-features` specifically, because
  `src/py_query_ranges.rs` is pyo3-gated and the query-core build has a known CI
  coverage gap for changes that cross `lib.rs` module gating.
- `pixi run test` for the Python suite.

## Out of scope

- `src/spine.rs:141` and `src/svar1_query.rs:60` build trees on the SVAR1 /
  spine paths, which have a different call shape and are not region-looped here.

  **Correction (#148, 2026-07-31):** the second half of that sentence is wrong
  for `spine.rs:141`. `overlap_batch_impl` *was* region-looped around
  `vk_slice` -> `spine::gather_keys`, so it kept an `O(regions x columns)` tree
  build after this PR. Excluding it was still right for this PR's scope; the
  stated reason was not. Fixed in
  `2026-07-31-searchtree-construction-audit-design.md`. (`svar1_query.rs:60` is
  correctly described — it builds one tree per contig batch.)
- No public Python surface changes — `OverlapIndex` and every constructor are
  `pub(crate)`. `skills/genoray-api/SKILL.md` needs no update.

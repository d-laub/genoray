# OverlapIndex Hoist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build every SVAR2 range-search tree once per channel instead of once per (region, channel), by unifying all five search-state constructors behind a single `OverlapIndex` type and hoisting its construction out of the per-region loops.

**Architecture:** #144 introduced `VkColumnIndex` — region-independent search state (a `SearchTree` + parallel `v_ends` + a deletion bound) built once per var_key column. Three sibling channels still rebuild that state per region: the dense union (`DenseUnion::overlap`), the two dense class tables (`ContigReader::dense_snp_overlap` / `dense_indel_overlap`), and the var_key columns as consumed by `svar2_slice::gather_var_key`. This plan renames `VkColumnIndex` to `OverlapIndex<'a>` with a `Cow<'a, [u32]>` for `v_ends` (so the union can borrow the extents it already stores while the other channels own the ones they compute), adds the two dense class constructors plus `DenseUnion::index`, and hoists every construction above its region loop.

**Tech Stack:** Rust, `pixi` for the toolchain, `cargo test` for the Rust suite. No Python surface changes.

**Spec:** `docs/superpowers/specs/2026-07-31-svar2-overlap-index-hoist-design.md`

## Global Constraints

- **`CARGO_TARGET_DIR` must point off NFS for every cargo invocation.** The repo's `./target` is NFS-backed and the linker bus-errors on it in debug builds. Prefix every cargo/pixi-cargo command with `CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"` (or any node-local path).
- **Rust tests run via `pixi run test-rust`**, which is `cargo test --no-default-features --features conversion`. A bare `cargo test` fails to link the pyo3 test binary (`undefined symbol: _Py_Dealloc`); dropping `--features conversion` fails to build the tests in this plan, because `tests/common::build_contig` calls `orchestrator::process_chromosome`.
- **`pixi run test-rust <arg>` filters by TEST NAME, not file.** A non-matching argument vacuously passes 0 tests. Always select a file with `--test <file>`, e.g. `pixi run test-rust --test test_ranges_split`.
- **`pixi run check-core` (`cargo check --no-default-features`) must pass** before each commit that touches `src/`. That is the query-only build gvl links as a path dependency, and CI's Rust job otherwise only builds with `conversion` on.
- **Commits follow Conventional Commits** (`feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`). Never edit `CHANGELOG.md` or bump the version by hand.
- **Never change observable behavior.** Every task in this plan is a pure hoist: the same `(positions, v_ends, max_del, q_start, q_end)` reach `overlap_range`; only the moment the tree is built moves. Any diff in query output or written bytes is a bug in the change, not an accepted consequence.
- All new/changed types and constructors stay `pub(crate)`. No public Python surface changes, so `skills/genoray-api/SKILL.md` is not touched.

---

## File Structure

| File | Responsibility after this plan |
| --- | --- |
| `src/query/reader.rs` | Owns `OverlapIndex<'a>` (the one search-state type) and four of its five constructors: `vk_snp_index`, `vk_indel_index`, `dense_snp_index`, `dense_indel_index`. `dense_snp_overlap` / `dense_indel_overlap` are deleted. |
| `src/query/union.rs` | `DenseUnion` stays a tree-free data struct; gains `index()` — the fifth `OverlapIndex` constructor, borrowing the union's stored `v_ends`. `DenseUnion::overlap` is deleted. |
| `src/query/gather.rs` | `find_ranges` and `overlap_batch_impl` build each index once above their region loops. |
| `src/py_query_ranges.rs` | Same hoist for the Python `find_ranges` binding. |
| `src/query/oracle.rs` | Single-region caller; adapted to the new call convention, cost unchanged. |
| `src/svar2_slice.rs` | `gather_var_key` takes a column-index factory instead of a range closure and builds one index per column; the two `gather_dense` calls get their dense index built once at the call site. |
| `tests/test_ranges_split.rs` | Read-path complexity guard, tightened from "≤ 2 tree builds per extra region" to "zero growth". |
| `tests/test_svar2_slice.rs` | New wide fixture + new slice-path complexity guard asserting zero growth. |

**Task order is strictly sequential.** Task 2 needs the type from Task 1; Task 3 needs Task 2's dense hoist to have landed before its guard can assert *zero* growth. Tasks 2 and 3 also both edit `src/svar2_slice.rs`. There is no independent work to dispatch in parallel here.

---

### Task 1: Unify search state behind `OverlapIndex`

Pure refactor. `VkColumnIndex` becomes `OverlapIndex<'a>` with a `Cow` for `v_ends` and two shared constructors (`new`, `empty`) that Tasks 2 and 3 build on. No behavior changes, so there is no new test — **the existing Rust suite is the gate**, and it must stay green. Do not add a test that asserts a rename.

**Files:**
- Modify: `src/query/reader.rs:240-315` (type + `vk_snp_index` + `vk_indel_index`)
- Modify: `src/query/gather.rs:358-359`, `src/query/gather.rs:409` (doc comments naming the old type)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub(crate) struct OverlapIndex<'a>` with
  - `pub(crate) fn new(o0: usize, positions: &[u32], v_ends: Cow<'a, [u32]>, max_del: u32) -> Self`
  - `pub(crate) fn empty(o0: usize) -> Self`
  - `pub(crate) fn overlap(&self, q_start: u32, q_end: u32) -> Range<usize>`
  - `pub(crate) o0: usize` field
  and the retargeted `ContigReader::vk_snp_index(col) -> OverlapIndex<'static>` /
  `ContigReader::vk_indel_index(sample, p) -> OverlapIndex<'static>`.

- [ ] **Step 1: Add the `Cow` import**

In `src/query/reader.rs`, alongside the existing `use std::ops::Range;`:

```rust
use std::borrow::Cow;
use std::ops::Range;
use std::path::Path;
```

- [ ] **Step 2: Replace the `VkColumnIndex` type with `OverlapIndex`**

In `src/query/reader.rs`, replace the whole `VkColumnIndex` struct and its `impl` block (currently lines 240-266) with:

```rust
/// Region-independent search state for one channel: a `SearchTree` over its
/// positions, the parallel `v_ends` extents, and the deletion bound
/// `overlap_range` needs.
///
/// Built ONCE per channel, then queried per region. Hoisting this out of the
/// old per-region `vk_*_overlap` / `dense_*_overlap` methods is what turns
/// `find_ranges` and `svar2_slice`'s gather from O(regions x channels) tree
/// builds into O(channels): each packed store is swept once instead of once
/// per region.
///
/// `v_ends` is a `Cow` because the two kinds of channel differ in who owns the
/// extents: the var_key columns and the dense class tables COMPUTE theirs
/// (`pos + 1`, `pos + 1 + deletion_len(key)`) and must own the result, while
/// `DenseUnion` already stores its own and can lend them — which is what keeps
/// `DenseUnion::index` from copying an O(dense variants) array per query.
pub(crate) struct OverlapIndex<'a> {
    /// Absolute base offset of this channel in the packed arrays. `0` for the
    /// dense class tables and the dense union, whose ranges are already
    /// absolute; the var_key columns pass their column's start offset.
    pub(crate) o0: usize,
    /// `None` for an empty channel — `SearchTree`/`overlap_range` are not
    /// defined over an empty position array, matching the old early return.
    inner: Option<(SearchTree, Cow<'a, [u32]>)>,
    max_del: u32,
}

impl<'a> OverlapIndex<'a> {
    /// Build the search state for a NON-EMPTY channel. `positions` must be
    /// ascending and `v_ends[i]` must be the right extent of `positions[i]`.
    pub(crate) fn new(
        o0: usize,
        positions: &[u32],
        v_ends: Cow<'a, [u32]>,
        max_del: u32,
    ) -> Self {
        debug_assert!(!positions.is_empty(), "use OverlapIndex::empty instead");
        debug_assert_eq!(positions.len(), v_ends.len());
        OverlapIndex {
            o0,
            inner: Some((SearchTree::new(positions), v_ends)),
            max_del,
        }
    }

    /// An index over an empty channel: `overlap` always returns `o0..o0`, and
    /// no `SearchTree` is built.
    pub(crate) fn empty(o0: usize) -> Self {
        OverlapIndex {
            o0,
            inner: None,
            max_del: 0,
        }
    }

    /// Absolute `[start, end)` into the channel's packed positions/keys for one
    /// region. Every element of the returned range truly overlaps
    /// `[q_start, q_end)` — `overlap_range` does the left-overlap sub-scan.
    pub(crate) fn overlap(&self, q_start: u32, q_end: u32) -> Range<usize> {
        let Some((tree, v_ends)) = &self.inner else {
            return self.o0..self.o0;
        };
        let (s, e) = overlap_range(tree, v_ends, self.max_del, q_start, q_end);
        (self.o0 + s)..(self.o0 + e)
    }
}
```

- [ ] **Step 3: Retarget the two var_key constructors**

In `src/query/reader.rs`, replace the bodies of `vk_snp_index` and `vk_indel_index` (currently lines 268-315) with:

```rust
impl ContigReader {
    /// SNP-channel column index. SNP `v_end = pos + 1` and `max_region_length =
    /// 0`, since a SNP spans exactly one base.
    pub(crate) fn vk_snp_index(&self, col: usize) -> OverlapIndex<'static> {
        let vk_range = self.vk_snp.column(col);
        let (o0, o1) = (vk_range.start, vk_range.end);
        let positions = &self.vk_snp.positions()[o0..o1];
        if positions.is_empty() {
            return OverlapIndex::empty(o0);
        }
        let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        OverlapIndex::new(o0, positions, Cow::Owned(v_ends), 0)
    }

    /// Indel-channel column index for `(sample, p)`. `v_end = pos + 1 +
    /// deletion_len(key)`; the search bound is this column's `max_del`.
    pub(crate) fn vk_indel_index(&self, sample: usize, p: usize) -> OverlapIndex<'static> {
        let col = sample * self.ploidy + p;
        let vk_range = self.vk_indel.column(col);
        let (o0, o1) = (vk_range.start, vk_range.end);
        let positions = &self.vk_indel.positions()[o0..o1];
        if positions.is_empty() {
            return OverlapIndex::empty(o0);
        }
        let keys = &as_u32(&self.vk_indel.keys)[o0..o1];
        let v_ends: Vec<u32> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| pos + 1 + rvk::deletion_len(keys[i]))
            .collect();
        OverlapIndex::new(
            o0,
            positions,
            Cow::Owned(v_ends),
            self.vk_indel_max_del[[sample, p]],
        )
    }
}
```

Note the behavior-preserving detail: the old empty-column path returned `max_del: 0`, and `OverlapIndex::empty` does the same. `max_del` is unused when `inner` is `None`.

- [ ] **Step 4: Fix the doc comments that name the old type**

In `src/query/gather.rs`, in the `find_ranges_haps` doc comment (~line 358), change:

```
/// Column-outer / region-inner, so each column's `VkColumnIndex` is built
```

to:

```
/// Column-outer / region-inner, so each column's `OverlapIndex` is built
```

and in the inline comment inside `fill` (~line 409), change `VkColumnIndex::last_overlapping` to `OverlapIndex::last_overlapping`.

- [ ] **Step 5: Verify nothing else names the old type**

Run:

```bash
rg -n 'VkColumnIndex' src/ tests/
```

Expected: no output.

- [ ] **Step 6: Run the full Rust suite**

Run:

```bash
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" pixi run test-rust
```

Expected: PASS, same test count as before the change. This is a rename plus two shared constructors; a single failure here means the refactor changed behavior and must be corrected, not accommodated.

- [ ] **Step 7: Verify the query-core build**

Run:

```bash
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" pixi run check-core
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/query/reader.rs src/query/gather.rs
git commit -m "refactor(query): generalize VkColumnIndex into a Cow-backed OverlapIndex

One search-state type for every channel: shared new/empty constructors and
a Cow v_ends so a channel that already stores its extents can lend them
instead of copying. Pure rename plus constructors; no behavior change.

Relates to #145"
```

---

### Task 2: Hoist the dense union and dense class indices

TDD proper: tighten the existing read-path complexity guard first (it goes red because the dense channels still rebuild per region), then make it green by adding the three dense constructors and hoisting all their call sites. `src/svar2_slice.rs`'s two `gather_dense` calls are updated here too, since this task deletes the methods they call.

**Files:**
- Modify: `tests/test_ranges_split.rs:443-470` (`test_find_ranges_tree_builds_do_not_scale_with_regions`)
- Modify: `src/query/reader.rs:326-370` (delete `dense_snp_overlap` / `dense_indel_overlap`, add `dense_snp_index` / `dense_indel_index`)
- Modify: `src/query/union.rs:17-43` (doc comment + `overlap` → `index`)
- Modify: `src/query/gather.rs:233-238`, `src/query/gather.rs:477-493`
- Modify: `src/py_query_ranges.rs:281-293`
- Modify: `src/query/oracle.rs:54-55`
- Modify: `src/svar2_slice.rs:576-597`

**Interfaces:**
- Consumes: `OverlapIndex::{new, empty, overlap}` from Task 1.
- Produces: `ContigReader::dense_snp_index(&self) -> OverlapIndex<'static>`,
  `ContigReader::dense_indel_index(&self) -> OverlapIndex<'static>`,
  `DenseUnion::index(&self) -> OverlapIndex<'_>`.
  `ContigReader::dense_snp_overlap`, `ContigReader::dense_indel_overlap` and
  `DenseUnion::overlap` no longer exist.

- [ ] **Step 1: Tighten the read-path guard to zero growth**

In `tests/test_ranges_split.rs`, replace the assertion block at the end of `test_find_ranges_tree_builds_do_not_scale_with_regions` (currently the `let dense_growth = ...;` line and the `assert!` that follows) with:

```rust
    // After the dense hoist (#145) NO channel builds a tree per region: the
    // var_key columns, the dense union and the two dense class tables are each
    // swept once per call. So this is exact equality, not an allowance — a
    // budget here is what let the dense leak survive #144.
    assert_eq!(
        cost_many, cost_one,
        "tree builds grew with region count: {cost_one} -> {cost_many}"
    );
```

Also update the test's doc comment (the paragraph above `#[test]` that explains the `3 * Δregions` / `2 * Δregions` allowance) to state that the expectation is now zero growth across all channels.

- [ ] **Step 2: Run the guard to verify it fails**

Run:

```bash
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" \
  pixi run test-rust --test test_ranges_split test_find_ranges_tree_builds_do_not_scale_with_regions
```

Expected: FAIL, with `cost_many` exceeding `cost_one` by `2 * (16 - 1) = 30` — one dense-union tree and one dense-indel tree per extra region. (`synth_reader_wide` routes nothing Dense-SNP, so there is no third term.) If it passes, stop: the fixture is not exercising the dense channels and the guard is worthless.

- [ ] **Step 3: Replace the dense class overlap methods with index constructors**

In `src/query/reader.rs`, replace the entire `impl ContigReader` block containing `dense_snp_overlap` and `dense_indel_overlap` (currently lines 330-370) with:

```rust
impl ContigReader {
    /// Dense SNP class index, over `dense/snp`'s positions/keys. SNP `v_end =
    /// pos + 1` (`max_region_length = 0`). Empty if there is no snp table.
    pub(crate) fn dense_snp_index(&self) -> OverlapIndex<'static> {
        let Some(d) = &self.dense_snp else {
            return OverlapIndex::empty(0);
        };
        let positions = d.positions();
        if positions.is_empty() {
            return OverlapIndex::empty(0);
        }
        let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        OverlapIndex::new(0, positions, Cow::Owned(v_ends), 0)
    }

    /// Dense indel class index, over `dense/indel`'s positions/keys. `v_end =
    /// pos + 1 + deletion_len(key)`; the search bound is the per-contig dense
    /// max_del. Empty if there is no indel table.
    pub(crate) fn dense_indel_index(&self) -> OverlapIndex<'static> {
        let Some(d) = &self.dense_indel else {
            return OverlapIndex::empty(0);
        };
        let positions = d.positions();
        if positions.is_empty() {
            return OverlapIndex::empty(0);
        }
        let keys = as_u32(&d.keys);
        // Fail fast on a corrupt sidecar rather than silently truncating.
        debug_assert_eq!(positions.len(), keys.len());
        let v_ends: Vec<u32> = positions
            .iter()
            .zip(keys.iter())
            .map(|(&pos, &key)| pos + 1 + rvk::deletion_len(key))
            .collect();
        OverlapIndex::new(0, positions, Cow::Owned(v_ends), self.dense_indel_max_del)
    }
}
```

`o0` is `0` for both because the dense class ranges are already absolute — the old methods returned `s..e` unshifted.

- [ ] **Step 4: Replace `DenseUnion::overlap` with `DenseUnion::index`**

In `src/query/union.rs`, add `OverlapIndex` to the reader import and swap the method. The import line becomes:

```rust
use super::reader::{ContigReader, OverlapIndex};
```

and `std::borrow::Cow` joins the `std` imports:

```rust
use std::borrow::Cow;
use std::ops::Range;
```

Replace the `impl DenseUnion` block's `overlap` method (currently lines 29-38) with:

```rust
    /// Region-independent search state over the union, built once by the
    /// callers that actually search it.
    ///
    /// This is deliberately NOT built inside `dense_union()`: `gather_ranges`,
    /// `dense_max_end_keys` and `ContigReader::max_deletion_len` all construct
    /// a `DenseUnion` without ever overlapping it, and must not be charged a
    /// `SearchTree::new` they never use. Borrows `v_ends`, so building an index
    /// copies nothing.
    pub(crate) fn index(&self) -> OverlapIndex<'_> {
        if self.refs.is_empty() {
            return OverlapIndex::empty(0);
        }
        OverlapIndex::new(
            0,
            &self.positions,
            Cow::Borrowed(self.v_ends.as_slice()),
            self.max_del,
        )
    }
```

`SearchTree` and `overlap_range` are no longer used in this file — drop them from `use crate::search::{SearchTree, overlap_range};` (the whole `use` goes if nothing else in the file needs it; let `cargo check` tell you).

Also fix the now-false struct doc comment at `src/query/union.rs:17-20`: `overlap` no longer exists, so the sentence "Region-independent — built once per query; `overlap` derives each region's index range from it" becomes "Region-independent and tree-free — built once per query; `index()` adds the search state for callers that range-query it."

- [ ] **Step 5: Hoist in `overlap_batch_impl`**

In `src/query/gather.rs`, replace the union block in `overlap_batch_impl` (currently lines 233-238):

```rust
    let dense = reader.dense_union();
    // Per-region dense index ranges — shared across all samples in the region.
    // One index for the whole batch: `SearchTree::new` runs once, not per region.
    let dense_ix = dense.index();
    let ranges: Vec<Range<usize>> = regions
        .iter()
        .map(|&(qs, qe)| dense_ix.overlap(qs, qe))
        .collect();
```

- [ ] **Step 6: Hoist in `find_ranges`**

In `src/query/gather.rs`, replace the three per-region `map` blocks in `find_ranges` (currently lines 477-493):

```rust
    // Region-independent union and dense class indices, each built ONCE for the
    // whole batch — this is the O(regions x channels) -> O(channels) fix.
    let dense = reader.dense_union();
    let dense_ix = dense.index();
    let dense_range: Vec<Range<usize>> = regions
        .iter()
        .map(|&(qs, qe)| dense_ix.overlap(qs, qe))
        .collect();
    let region_starts: Vec<u32> = regions.iter().map(|&(qs, _)| qs).collect();

    let dense_snp_ix = reader.dense_snp_index();
    let dense_snp_range: Vec<Range<usize>> = regions
        .iter()
        .map(|&(qs, qe)| dense_snp_ix.overlap(qs, qe))
        .collect();
    let dense_indel_ix = reader.dense_indel_index();
    let dense_indel_range: Vec<Range<usize>> = regions
        .iter()
        .map(|&(qs, qe)| dense_indel_ix.overlap(qs, qe))
        .collect();
```

- [ ] **Step 7: Hoist in the Python binding**

In `src/py_query_ranges.rs`, replace the three per-region `map` blocks (currently lines 281-293):

```rust
        let dense = self.inner.dense_union();
        let dense_ix = dense.index();
        let dense_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| dense_ix.overlap(qs, qe))
            .collect();
        let dense_snp_ix = self.inner.dense_snp_index();
        let dense_snp_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| dense_snp_ix.overlap(qs, qe))
            .collect();
        let dense_indel_ix = self.inner.dense_indel_index();
        let dense_indel_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| dense_indel_ix.overlap(qs, qe))
            .collect();
```

- [ ] **Step 8: Adapt the single-region oracle caller**

In `src/query/oracle.rs`, replace line 55:

```rust
    let d_range = dense.index().overlap(q_start, q_end);
```

One region, so this builds exactly the one tree it built before — the edit exists only to keep a single call convention.

- [ ] **Step 9: Hoist the two dense calls in the slice path**

In `src/svar2_slice.rs`, in `slice_genos_inner`, build each dense index once above its `gather_dense` call. The two calls (currently lines 576-597) become:

```rust
    // One index per dense class for the whole request — `gather_dense` calls
    // `region_hits` once per class, so building here is what keeps the
    // per-region loop inside it tree-free.
    let d_snp_ix = reader.dense_snp_index();
    let d_snp_g = gather_dense(
        reader.dense_snp.as_ref(),
        true,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
        |qsw, qew| d_snp_ix.overlap(qsw, qew),
    );

    let d_indel_ix = reader.dense_indel_index();
    let d_indel_g = gather_dense(
        reader.dense_indel.as_ref(),
        false,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
        |qsw, qew| d_indel_ix.overlap(qsw, qew),
    );
```

- [ ] **Step 10: Run the guard to verify it passes**

Run:

```bash
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" \
  pixi run test-rust --test test_ranges_split test_find_ranges_tree_builds_do_not_scale_with_regions
```

Expected: PASS.

- [ ] **Step 11: Run the full Rust suite and the core build**

Run:

```bash
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" pixi run test-rust
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" pixi run check-core
```

Expected: both PASS. `test_gather_ranges_builds_no_search_tree` in the same file is the load-bearing one to watch — it asserts `gather_ranges` builds zero trees, which is exactly the invariant that would break if `dense_union()` had been made eager instead of adding `index()`.

- [ ] **Step 12: Verify the deleted methods have no callers left**

Run:

```bash
rg -n 'dense_snp_overlap|dense_indel_overlap|dense\.overlap|\.dense_union\(\)\.overlap' src/ tests/
```

Expected: no output.

- [ ] **Step 13: Commit**

```bash
git add src/query/reader.rs src/query/union.rs src/query/gather.rs \
        src/py_query_ranges.rs src/query/oracle.rs src/svar2_slice.rs \
        tests/test_ranges_split.rs
git commit -m "perf(query): build each dense channel's search tree once per request

DenseUnion::overlap and the dense_snp/indel_overlap methods each rebuilt a
SearchTree per region, so find_ranges, overlap_batch, the py binding and the
slice gather all paid O(regions x channels) builds. Replace them with
OverlapIndex constructors hoisted above every region loop; dense_union()
stays tree-free for gather_ranges / dense_max_end_keys / max_deletion_len,
which never search it.

The find_ranges complexity guard drops its 2-builds-per-region allowance and
now asserts exact zero growth.

Relates to #145"
```

---

### Task 3: Hoist the var_key column index in the slice path

This is #145 as filed. `gather_var_key` is already column-outer, but the closure it hands `region_hits` constructs the index inside the per-region loop. Change the injection point from a range closure to an index factory. TDD: the new guard goes red on var_key growth alone, because Task 2 already took the dense channels to zero.

**Files:**
- Modify: `tests/test_svar2_slice.rs` (new fixture + new test, appended near the existing fixture helpers)
- Modify: `src/svar2_slice.rs:707-770` (`region_hits` doc, `gather_var_key` signature/body)
- Modify: `src/svar2_slice.rs:543-573` (the two `gather_var_key` call sites)

**Interfaces:**
- Consumes: `OverlapIndex` and `ContigReader::vk_snp_index` / `vk_indel_index` from Task 1.
- Produces: `gather_var_key`'s `column_index: impl Fn(usize, usize) -> OverlapIndex<'static>` parameter, replacing `overlap_range: impl Fn(usize, usize, usize, u32, u32) -> Range<usize>`. `region_hits`'s signature is unchanged.

- [ ] **Step 1: Add the wide fixture**

In `tests/test_svar2_slice.rs`, add after `build_fixture_store`:

```rust
/// Ten records at n=2, ploidy=2 (np=4, flat gt order `[s0p0, s0p1, s1p0,
/// s1p1]`) chosen so the tree-build guard below is actually discriminating:
///
/// * four single-carrier SNPs and four single-carrier indels, one per hap, so
///   ALL FOUR columns of BOTH var_key channels are non-empty. An empty column
///   early-returns without building a tree, so a fixture that populates only
///   one column (as `fixture_records` does) barely moves the counter.
/// * one x=2 SNP and one x=2 indel so both DENSE class tables exist. Without
///   them the dense channels early-return and the guard would silently stop
///   covering the dense hoist.
///
/// Single carrier vs. x=2 is what picks the route: at np=4 with no sidecar or
/// field bits, `cost_model::choose_representation` keeps x=1 in VarKey and
/// flips x=2 to Dense.
///
/// Used ONLY by `slice_tree_builds_do_not_scale_with_regions` — do not extend
/// `fixture_records` for this, it is pinned by the byte-parity tests.
fn wide_fixture_records() -> Vec<SynthRecord<'static>> {
    let snp = |pos: i64, gt: Vec<i32>| SynthRecord {
        pos,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt,
    };
    let del = |pos: i64, gt: Vec<i32>| SynthRecord {
        pos,
        ref_allele: b"ATG",
        alts: vec![&b"A"[..]], // pure DEL, ilen = -2
        gt,
    };
    vec![
        // one single-carrier (VarKey) SNP per hap column
        snp(10, vec![1, 0, 0, 0]),
        snp(20, vec![0, 1, 0, 0]),
        snp(30, vec![0, 0, 1, 0]),
        snp(40, vec![0, 0, 0, 1]),
        // one single-carrier (VarKey) indel per hap column
        del(50, vec![1, 0, 0, 0]),
        del(60, vec![0, 1, 0, 0]),
        del(70, vec![0, 0, 1, 0]),
        del(80, vec![0, 0, 0, 1]),
        // x=2 => Dense, one per class, so both dense tables exist
        snp(90, vec![1, 1, 0, 0]),
        del(100, vec![0, 0, 1, 1]),
    ]
}

fn build_wide_fixture_store(dir: &Path, samples: &[&str]) {
    std::fs::create_dir_all(dir).unwrap();
    let records = wide_fixture_records();
    build_contig(dir, "chr1", samples, 2, &records);
    // Same reason as `build_fixture_store`: `build_contig` overwrites max_del
    // with a conservative fixture, so recompute the real routing-aware value.
    genoray_core::max_del::write_max_del(&dir.join("chr1"), samples.len(), 2).unwrap();
    write_meta(
        dir,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();
}
```

- [ ] **Step 2: Write the failing guard test**

In `tests/test_svar2_slice.rs`, add the `search` import to the existing `use genoray_core::...` block:

```rust
use genoray_core::search;
```

and add the test:

```rust
/// `gather_var_key` is column-outer, but until #145 the closure it handed
/// `region_hits` built a fresh `OverlapIndex` — `SearchTree::new` and all —
/// INSIDE the per-region loop, for every column. That is the same
/// O(regions x columns) shape #144 removed from `find_ranges`.
///
/// After the fix no channel in the slice path builds a tree per region: eight
/// var_key columns (four per class, all populated by `wide_fixture_records`)
/// and two dense classes are each swept once per request. So this asserts
/// EXACT equality, not an allowance.
///
/// `src/svar2_slice.rs` contains no rayon, so the whole slice runs on this
/// thread and `search::search_tree_build_count` — a thread-local — stays
/// observable. Each call gets its own output directory because a slice writes
/// `{out}/chr1` and would otherwise overwrite the previous run.
#[test]
fn slice_tree_builds_do_not_scale_with_regions() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_wide_fixture_store(&src, &samples);

    let slice_with = |out: &Path, regions: &[(u32, u32)]| {
        std::fs::create_dir_all(out).unwrap();
        slice_contig_genos(
            src.to_str().unwrap(),
            out.to_str().unwrap(),
            "chr1",
            &(0..samples.len()).collect::<Vec<_>>(),
            2,
            regions,
            OverlapMode::Variant,
            Routing::Preserve,
        )
        .unwrap()
    };

    let one = vec![(0u32, 1_000u32)];
    let many: Vec<(u32, u32)> = (0..16).map(|i| (i * 5, i * 5 + 1_000)).collect();

    let out_one = tmp.path().join("out_one");
    let b0 = search::search_tree_build_count();
    let n_one = slice_with(&out_one, &one);
    let cost_one = search::search_tree_build_count() - b0;

    let out_many = tmp.path().join("out_many");
    let b1 = search::search_tree_build_count();
    let n_many = slice_with(&out_many, &many);
    let cost_many = search::search_tree_build_count() - b1;

    // Both region sets cover every fixture variant, so this is the same slice
    // twice — any difference in tree builds is pure region-count overhead.
    assert_eq!(n_one, 10, "the 1-region slice must keep all 10 variants");
    assert_eq!(n_many, 10, "the 16-region slice must keep all 10 variants");
    assert!(cost_one > 0, "fixture must build trees at all");
    assert_eq!(
        cost_many, cost_one,
        "tree builds grew with region count: {cost_one} -> {cost_many}"
    );
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run:

```bash
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" \
  pixi run test-rust --test test_svar2_slice slice_tree_builds_do_not_scale_with_regions
```

Expected: FAIL on the final `assert_eq!`, with `cost_many` around `8 * 16 + 2 = 130` against `cost_one` around `8 + 2 = 10` — four var_key SNP columns plus four var_key indel columns rebuilt per region, over a constant 2 dense-class builds that Task 2 already hoisted. If instead it fails on `cost_one > 0` or on a variant count, the fixture is wrong; fix the fixture before touching `src/`.

- [ ] **Step 4: Change `gather_var_key`'s injection point**

In `src/svar2_slice.rs`, change the `overlap_range` parameter to a column-index factory and hoist the construction. Replace the parameter line and the `region_hits` call inside the column loop:

```rust
#[allow(clippy::too_many_arguments)]
fn gather_var_key(
    positions: &[u32],
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
    column_index: impl Fn(usize, usize) -> OverlapIndex<'static>,
    key_of: impl Fn(usize) -> u32,
    v_end_of: impl Fn(usize) -> u32,
) -> Vec<GatheredCall> {
    let mut out = Vec::new();
    for (s_out, &s_orig) in sample_orig_idx.iter().enumerate() {
        for p in 0..ploidy {
            let col_out = s_out * ploidy + p;
            // ONE index per column, built above the region loop inside
            // `region_hits` — the #145 fix. Mirrors `find_ranges_haps`.
            let ix = column_index(s_orig, p);
            let hits = region_hits(
                positions,
                regions,
                query_regions,
                overlap,
                |qsw, qew| ix.overlap(qsw, qew),
                &v_end_of,
            );
            for i in hits {
                out.push(GatheredCall {
                    src: i,
                    col_out,
                    pos: positions[i],
                    key: key_of(i),
                });
            }
        }
    }
    out
}
```

The old `col_src` local is gone — it was only ever passed to the closure, and both callers can derive it from `(s_orig, p)`. Remove the now-unused `let col_src = s_orig * ploidy + p;` line. Add `OverlapIndex` to the reader import at the top of the file:

```rust
use crate::query::reader::OverlapIndex;
```

Then update the doc comment above `gather_var_key`, whose last paragraph describes the old five-argument `overlap_range`:

```
/// `column_index(s_orig, p)` builds that column's search state ONCE, before the
/// per-region loop; the SNP caller derives its flat column as `s_orig * ploidy
/// + p`, the indel caller passes `(s_orig, p)` straight through because its
/// `max_del` bound is per-(sample, ploid).
```

Also update `region_hits`'s doc comment, which names `overlap_range` as "the tree-based windowed search": it now receives an already-built index's `overlap`, so reword to "narrow via `overlap_range` (one region's window from the column's prebuilt `OverlapIndex` — identical to what `find_ranges` uses …)". Leave the rest of that comment, especially the load-bearing paragraph about the extent re-check, untouched.

- [ ] **Step 5: Update the two call sites**

In `src/svar2_slice.rs`, in `slice_genos_inner`, replace the two closures (currently lines 553 and 571):

```rust
        |s_orig, p| reader.vk_snp_index(s_orig * ploidy + p),
```

and

```rust
        // The indel channel's tree search needs a per-(sample, ploid) max_del
        // bound — `s_orig` here must be the ORIGINAL column (`vk_indel_max_del`
        // is indexed by the source cohort, not the subset), mirroring
        // `find_ranges`'s `orig_s` usage.
        |s_orig, p| reader.vk_indel_index(s_orig, p),
```

- [ ] **Step 6: Run the guard to verify it passes**

Run:

```bash
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" \
  pixi run test-rust --test test_svar2_slice slice_tree_builds_do_not_scale_with_regions
```

Expected: PASS.

- [ ] **Step 7: Run the slice suite, then everything**

Run:

```bash
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" pixi run test-rust --test test_svar2_slice
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" pixi run test-rust
CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target" pixi run check-core
```

Expected: all PASS. The byte-parity tests (`slice_full_coverage_is_byte_identical_genos`, `preserve_identity_slice_is_byte_parity`) are the correctness net for this task — the slice must write identical bytes.

- [ ] **Step 8: Run the Python suite**

Run:

```bash
pixi run test
```

Expected: PASS. (`pixi run test` does NOT rebuild the Rust extension; that is fine here because no Python-visible behavior changed and the Python tests are only a regression net against accidental surface changes.)

- [ ] **Step 9: Commit**

```bash
git add src/svar2_slice.rs tests/test_svar2_slice.rs
git commit -m "perf(svar2): build each var_key column's search tree once per slice

gather_var_key was column-outer but built a fresh OverlapIndex inside the
per-region loop, so a slice paid O(regions x columns) SearchTree builds —
the shape #144 removed from find_ranges. Take a column-index factory instead
of a range closure and build once per column.

Guarded by a new slice-path tree-build test on a fixture wide enough to
populate all four columns of both var_key channels plus both dense tables.

Closes #145"
```

---

## Verification Before Completion

- [ ] `CARGO_TARGET_DIR=... pixi run test-rust` — full Rust suite green, test count unchanged except the one added test.
- [ ] `CARGO_TARGET_DIR=... pixi run check-core` — query-core build green.
- [ ] `pixi run test` — Python suite green.
- [ ] `rg -n 'VkColumnIndex|dense_snp_overlap|dense_indel_overlap' src/ tests/` — no output.
- [ ] Both complexity guards assert exact equality, and both were observed red before their fix landed.

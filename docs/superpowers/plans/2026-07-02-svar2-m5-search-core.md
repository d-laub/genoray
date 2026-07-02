# SVAR 2.0 — M5 (partial): format-independent overlap search core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Rust module `src/search.rs` providing a static search tree over a sorted `u32` array (`lower_bound`/`upper_bound`) plus an overlap-range resolver that, given variant starts/ends and a max-deletion bound, returns the `[s_idx, e_idx)` index range of variants truly overlapping a query `[q_start, q_end)`.

**Architecture:** A flat, cache-friendly `(B+1)`-ary static search tree (Algorithmica-style static B-tree; the block-structured, breadth-first layout the [left-tree post](https://curiouscoding.nl/posts/static-search-tree/#left-tree) describes) built once per sorted array and reused across queries. Each node is a `B`-key block searched with a scalar linear scan (autovectorizes); descent uses `child(k, j) = k*(B+1) + j + 1`. The overlap resolver mirrors SVAR 1.0's `var_ranges` (`python/genoray/_var_ranges.py`): a lower-bound / upper-bound pair over `v_starts` followed by forward/backward sub-scans over `v_ends`. Crucially, the two searches (`v_starts + max_region_length` for the LB, `v_starts` for the UB) collapse to **one tree** by shifting the *query* with a saturating subtraction instead of building a second tree.

**Tech Stack:** Rust (edition 2024), `proptest` (dev-dep, already present), `pixi` lint env for building/testing.

## Global Constraints

- **Rust edition 2024.** Plain `u32` throughout — no dependency on `types.rs`, `layout.rs`, `merge.rs`, or any on-disk type. Inputs are borrowed slices; outputs are index ranges (`usize`).
- **Stored positions must be `< u32::MAX`.** `u32::MAX` is reserved as the tree's padding sentinel. Genomic positions are far below this; document the invariant, do not runtime-check it on the hot path.
- **Block size `const B: usize = 16`** (64-byte cache line of `u32`). Child index formula: `child(k, j) = k * (B + 1) + j + 1`.
- **Sorted-ascending precondition** on `SearchTree::new` input and on `v_starts`. Document it; a `debug_assert!` is acceptable but no release-mode runtime validation.
- **All tests run via:** `pixi run -e lint cargo test --no-default-features <filter>` (the lint env carries the rust + build toolchain; `--no-default-features` drops the `extension-module` feature so the Rust test binary links libpython). Run from the worktree root.
- **Tests live in-source** in a `#[cfg(test)] mod tests` block inside `src/search.rs`, matching the codebase convention (see `src/bits.rs`).
- **Commit with prek hooks active** (already installed). End commit messages with the `Co-Authored-By` trailer if your workflow adds one; otherwise plain Conventional Commits (`feat(svar-2): ...`) as in the branch history.

---

## File Structure

- **Create `src/search.rs`** — the entire module: `const B`, `child()`, `block_lower()`, `block_upper()`, `struct SearchTree` (`new`, `lower_bound`, `upper_bound`, `len`, `is_empty`), `overlap_range()`, and the `#[cfg(test)] mod tests`. One file, one responsibility (format-independent overlap search).
- **Modify `src/lib.rs`** — add `pub mod search;` to the module list (insert after `pub mod rvk;`, before `pub mod streams;`).

No other files change. Disk integration (reading M3 sidecars, producing `max_del.npy`, the sub-stream union, genotype gather) is explicitly out of scope per the spec.

---

## Reference: the SVAR 1.0 algorithm being ported

From `python/genoray/_var_ranges.py::var_ranges` (0-based, exclusive ends):

- `v_starts[i]` = `POS - 1`; `v_ends[i]` = `POS - min(ILEN, 0)` → a SNP spans one base (`v_end = v_start + 1`), a deletion of `d` bases spans `v_end = v_start + 1 + d`.
- `lb = searchsorted(v_starts + max_v_len, q_start)` — leftmost variant whose start could still reach `q_start`.
- `ub = searchsorted(v_starts, q_end)` — one past the last variant starting before `q_end`.
- forward sub-scan `[lb, ub)` for the first `i` with `q_start < v_ends[i]` → `s_idx`.
- backward sub-scan for the last such `i` → `e_idx` (exclusive).

**Key generalization decisions locked in by this plan (spec left them open):**

1. **`max_region_length` = an upper bound on deletion length `d`** (0 for a SNP-only stream), *not* the span. The exact safe LB threshold is `q_start - max_region_length`: potential overlap needs `v_end > q_start` ⟺ `v_start + 1 + d > q_start` ⟺ `v_start >= q_start - d`, and `d <= max_region_length`, so `v_start >= q_start - max_region_length` is a conservative (never-misses) bound that the forward sub-scan then tightens.
2. **One tree, shifted query.** `searchsorted(v_starts + max_region_length, q_start)` = first `i` with `v_starts[i] >= q_start - max_region_length` = `tree.lower_bound(q_start.saturating_sub(max_region_length))`. Saturating subtraction handles `q_start < max_region_length` (threshold clamps to 0 → index 0, all elements qualify). No second tree needed.
3. **`v_ends` passed explicitly** as `&[u32]`. The core stays purely positional; reconstructing `v_end` from `ILEN` is the (future) caller's job.
4. **Empty result convention:** return `(s, e)` with `s == e`. The resolver returns `(ub, ub)` on no-overlap; tests treat any `s == e` as empty.

Note the resolver is **more correct than the Python reference** in one edge case: when `lb == ub` (or the forward scan finds nothing) it returns an empty range, whereas the Python `s_idx == n_vars` sentinel can mislabel a non-overlapping query. Test against the brute-force oracle, **not** against the Python.

---

### Task 1: Module scaffold + block-scan primitives

**Files:**
- Create: `src/search.rs`
- Modify: `src/lib.rs` (add `pub mod search;`)
- Test: in-source `#[cfg(test)] mod tests` in `src/search.rs`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `const B: usize = 16;`
  - `fn child(k: usize, j: usize) -> usize` — `k * (B + 1) + j + 1`
  - `fn block_lower(block: &[u32], x: u32) -> usize` — first `i` with `block[i] >= x`, else `block.len()`
  - `fn block_upper(block: &[u32], x: u32) -> usize` — first `i` with `block[i] > x`, else `block.len()`

- [ ] **Step 1: Create `src/search.rs` with the module doc, constants, and block primitives**

```rust
//! Format-independent overlap search core (SVAR 2.0 milestone M5, part 1).
//!
//! A static, cache-friendly search tree over a sorted ascending `u32` array
//! (`lower_bound`/`upper_bound`) plus an overlap-range resolver that mirrors the
//! SVAR 1.0 `var_ranges` shape (see `python/genoray/_var_ranges.py`). Depends only
//! on in-memory slices — no `layout.rs`/`merge.rs`/on-disk types. Disk integration
//! (M3 sidecars, `max_del.npy` producer, sub-stream union, genotype gather) is a
//! separate follow-up.
//!
//! Layout: a `(B+1)`-ary static B-tree, the block-structured breadth-first layout
//! of the "left-tree" (<https://curiouscoding.nl/posts/static-search-tree/>). Each
//! node is a `B`-key block scanned linearly; descent uses
//! `child(k, j) = k*(B+1) + j + 1`. `u32::MAX` is reserved as the padding sentinel,
//! so stored positions must be `< u32::MAX` (genomic positions always are).

/// Keys per node — a 64-byte cache line of `u32`.
const B: usize = 16;

/// Index of the `j`-th child (`0..=B`) of node `k` in the flat tree array.
#[inline]
fn child(k: usize, j: usize) -> usize {
    k * (B + 1) + j + 1
}

/// First index `i` in `[0, block.len()]` with `block[i] >= x` (i.e. `block.len()`
/// when every element is `< x`). Scalar scan; the fixed-width block loop
/// autovectorizes.
#[inline]
fn block_lower(block: &[u32], x: u32) -> usize {
    let mut i = 0;
    while i < block.len() && block[i] < x {
        i += 1;
    }
    i
}

/// First index `i` with `block[i] > x` (`block.len()` when every element is `<= x`).
#[inline]
fn block_upper(block: &[u32], x: u32) -> usize {
    let mut i = 0;
    while i < block.len() && block[i] <= x {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_lower_finds_first_ge() {
        let b = [1u32, 3, 5, 7];
        assert_eq!(block_lower(&b, 0), 0);
        assert_eq!(block_lower(&b, 1), 0);
        assert_eq!(block_lower(&b, 2), 1);
        assert_eq!(block_lower(&b, 5), 2);
        assert_eq!(block_lower(&b, 7), 3);
        assert_eq!(block_lower(&b, 8), 4); // none >= x → len
    }

    #[test]
    fn block_upper_finds_first_gt() {
        let b = [1u32, 3, 5, 7];
        assert_eq!(block_upper(&b, 0), 0);
        assert_eq!(block_upper(&b, 1), 1);
        assert_eq!(block_upper(&b, 5), 3);
        assert_eq!(block_upper(&b, 7), 4); // none > x → len
    }

    #[test]
    fn child_formula() {
        assert_eq!(child(0, 0), 1);
        assert_eq!(child(0, B), B + 1);
        assert_eq!(child(1, 0), B + 2);
    }
}
```

- [ ] **Step 2: Wire the module into `src/lib.rs`**

In `src/lib.rs`, add the module declaration after `pub mod rvk;`:

```rust
pub mod rvk;
pub mod search;
pub mod streams;
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: PASS — `test result: ok. 3 passed` (the three `search::tests` unit tests), and the crate compiles with the new module.

- [ ] **Step 4: Commit**

```bash
git add src/search.rs src/lib.rs
git commit -m "feat(svar-2): search.rs scaffold with block-scan primitives"
```

---

### Task 2: `SearchTree` construction + `lower_bound`

**Files:**
- Modify: `src/search.rs`
- Test: in-source `#[cfg(test)] mod tests` in `src/search.rs`

**Interfaces:**
- Consumes: `B`, `child`, `block_lower` from Task 1.
- Produces:
  - `pub struct SearchTree { keys: Vec<u32>, idx: Vec<u32>, nblocks: usize, n: usize }`
  - `pub fn SearchTree::new(sorted: &[u32]) -> SearchTree`
  - `pub fn SearchTree::len(&self) -> usize`
  - `pub fn SearchTree::is_empty(&self) -> bool`
  - `pub fn SearchTree::lower_bound(&self, x: u32) -> usize` — first `i` with `sorted[i] >= x`, else `len()`

- [ ] **Step 1: Write the failing tests**

Add these tests inside the existing `#[cfg(test)] mod tests` block in `src/search.rs`. The reference is `slice::partition_point` — an independent, obviously-correct lower bound.

```rust
    /// Independent reference: first index with `sorted[i] >= x`.
    fn lb_ref(sorted: &[u32], x: u32) -> usize {
        sorted.partition_point(|&v| v < x)
    }

    #[test]
    fn lower_bound_empty() {
        let t = SearchTree::new(&[]);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert_eq!(t.lower_bound(0), 0);
        assert_eq!(t.lower_bound(u32::MAX - 1), 0);
    }

    #[test]
    fn lower_bound_single() {
        let a = [42u32];
        let t = SearchTree::new(&a);
        assert_eq!(t.lower_bound(0), 0);
        assert_eq!(t.lower_bound(42), 0);
        assert_eq!(t.lower_bound(43), 1); // past end
    }

    #[test]
    fn lower_bound_matches_partition_point_across_block_boundaries() {
        // Sizes that straddle single-block, block+1, and multi-level tree shapes.
        for &n in &[1usize, 15, 16, 17, 31, 32, 33, 271, 272, 273, 1000] {
            // strictly increasing evens: 0, 2, 4, ... so odd queries land between keys
            let a: Vec<u32> = (0..n as u32).map(|v| v * 2).collect();
            let t = SearchTree::new(&a);
            // probe every gap: below, on, and just above each key, plus past-end
            for k in 0..=(n as u32 * 2 + 1) {
                assert_eq!(
                    t.lower_bound(k),
                    lb_ref(&a, k),
                    "n={n} x={k}"
                );
            }
        }
    }

    #[test]
    fn lower_bound_with_duplicates() {
        // Duplicates: lower_bound must return the FIRST occurrence.
        let a = [5u32, 5, 5, 8, 8, 9];
        let t = SearchTree::new(&a);
        assert_eq!(t.lower_bound(5), 0);
        assert_eq!(t.lower_bound(6), 3);
        assert_eq!(t.lower_bound(8), 3);
        assert_eq!(t.lower_bound(9), 5);
        assert_eq!(t.lower_bound(10), 6);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: FAIL to compile — `cannot find type SearchTree in this scope`.

- [ ] **Step 3: Implement `SearchTree` construction and `lower_bound`**

Add this above the `#[cfg(test)]` block in `src/search.rs`:

```rust
/// A static search tree over a sorted ascending `u32` array. Built once, queried
/// many times. See the module docs for the layout.
pub struct SearchTree {
    /// `nblocks * B` keys, filled in-order from the sorted input; unused trailing
    /// slots hold `u32::MAX`.
    keys: Vec<u32>,
    /// Parallel to `keys`: the original input index of each key; padding slots
    /// hold `n` (one-past-the-end).
    idx: Vec<u32>,
    nblocks: usize,
    n: usize,
}

impl SearchTree {
    /// Build a tree over `sorted` (ascending). Values must be `< u32::MAX`.
    /// `O(n)` construction.
    pub fn new(sorted: &[u32]) -> Self {
        let n = sorted.len();
        let nblocks = n.div_ceil(B);
        let mut tree = SearchTree {
            keys: vec![u32::MAX; nblocks * B],
            idx: vec![n as u32; nblocks * B],
            nblocks,
            n,
        };
        let mut cursor = 0usize;
        tree.build(0, sorted, &mut cursor);
        tree
    }

    /// In-order fill: recurse into each child subtree, writing the next sorted
    /// value between children so stored keys are globally ascending in in-order.
    /// `cursor` walks `sorted` once.
    fn build(&mut self, k: usize, sorted: &[u32], cursor: &mut usize) {
        if k >= self.nblocks {
            return;
        }
        for j in 0..B {
            self.build(child(k, j), sorted, cursor);
            if *cursor < sorted.len() {
                self.keys[k * B + j] = sorted[*cursor];
                self.idx[k * B + j] = *cursor as u32;
                *cursor += 1;
            }
        }
        self.build(child(k, B), sorted, cursor);
    }

    /// Number of real elements in the tree.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// First index `i` with `sorted[i] >= x`, or `len()` if every element is `< x`.
    pub fn lower_bound(&self, x: u32) -> usize {
        let mut k = 0;
        let mut res = self.n as u32; // default: past end
        while k < self.nblocks {
            let block = &self.keys[k * B..k * B + B];
            let j = block_lower(block, x);
            if j < B {
                res = self.idx[k * B + j];
            }
            k = child(k, j);
        }
        res as usize
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: PASS — all Task 1 + Task 2 unit tests green.

- [ ] **Step 5: Commit**

```bash
git add src/search.rs
git commit -m "feat(svar-2): SearchTree construction and lower_bound"
```

---

### Task 3: `upper_bound`

**Files:**
- Modify: `src/search.rs`
- Test: in-source `#[cfg(test)] mod tests` in `src/search.rs`

**Interfaces:**
- Consumes: `SearchTree`, `child`, `block_upper` from Tasks 1–2.
- Produces: `pub fn SearchTree::upper_bound(&self, x: u32) -> usize` — first `i` with `sorted[i] > x`, else `len()`.

- [ ] **Step 1: Write the failing tests**

Add inside `#[cfg(test)] mod tests`:

```rust
    /// Independent reference: first index with `sorted[i] > x`.
    fn ub_ref(sorted: &[u32], x: u32) -> usize {
        sorted.partition_point(|&v| v <= x)
    }

    #[test]
    fn upper_bound_empty_and_single() {
        let t = SearchTree::new(&[]);
        assert_eq!(t.upper_bound(0), 0);

        let t = SearchTree::new(&[42u32]);
        assert_eq!(t.upper_bound(41), 0);
        assert_eq!(t.upper_bound(42), 1); // strictly greater → past this element
        assert_eq!(t.upper_bound(43), 1);
    }

    #[test]
    fn upper_bound_with_duplicates() {
        let a = [5u32, 5, 5, 8, 8, 9];
        let t = SearchTree::new(&a);
        assert_eq!(t.upper_bound(5), 3); // past the last 5
        assert_eq!(t.upper_bound(7), 3);
        assert_eq!(t.upper_bound(8), 5);
        assert_eq!(t.upper_bound(9), 6);
    }

    #[test]
    fn upper_bound_matches_partition_point_across_block_boundaries() {
        for &n in &[1usize, 15, 16, 17, 31, 32, 33, 271, 272, 273, 1000] {
            let a: Vec<u32> = (0..n as u32).map(|v| v * 2).collect();
            let t = SearchTree::new(&a);
            for k in 0..=(n as u32 * 2 + 1) {
                assert_eq!(t.upper_bound(k), ub_ref(&a, k), "n={n} x={k}");
            }
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: FAIL to compile — `no method named upper_bound found for struct SearchTree`.

- [ ] **Step 3: Implement `upper_bound`**

Add this method to the `impl SearchTree` block in `src/search.rs`, directly after `lower_bound`:

```rust
    /// First index `i` with `sorted[i] > x`, or `len()` if every element is `<= x`.
    pub fn upper_bound(&self, x: u32) -> usize {
        let mut k = 0;
        let mut res = self.n as u32;
        while k < self.nblocks {
            let block = &self.keys[k * B..k * B + B];
            let j = block_upper(block, x);
            if j < B {
                res = self.idx[k * B + j];
            }
            k = child(k, j);
        }
        res as usize
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: PASS — all unit tests green.

- [ ] **Step 5: Commit**

```bash
git add src/search.rs
git commit -m "feat(svar-2): SearchTree upper_bound"
```

---

### Task 4: Proptest `lower_bound`/`upper_bound` against `partition_point`

**Files:**
- Modify: `src/search.rs`
- Test: in-source `#[cfg(test)] mod tests` in `src/search.rs`

**Interfaces:**
- Consumes: `SearchTree` (`new`, `lower_bound`, `upper_bound`).
- Produces: no new public API — property-test coverage only.

- [ ] **Step 1: Write the failing proptests**

Add a `proptest!` block inside `#[cfg(test)] mod tests`. Add `use proptest::prelude::*;` at the top of the test module if not already present (Task 5/6 also use it — add it once).

```rust
    proptest! {
        // lower_bound/upper_bound match slice::partition_point for random sorted
        // arrays and random queries. Sizes range across the tree's block/level
        // boundaries; the query range extends just past the max value so past-end
        // results are exercised.
        #[test]
        fn prop_bounds_match_partition_point(
            mut vals in proptest::collection::vec(0u32..2000, 0..600),
            queries in proptest::collection::vec(0u32..2100, 1..40),
        ) {
            vals.sort_unstable();
            let t = SearchTree::new(&vals);
            prop_assert_eq!(t.len(), vals.len());
            for &q in &queries {
                prop_assert_eq!(t.lower_bound(q), vals.partition_point(|&v| v < q), "lb q={}", q);
                prop_assert_eq!(t.upper_bound(q), vals.partition_point(|&v| v <= q), "ub q={}", q);
            }
        }
    }
```

- [ ] **Step 2: Run to verify it passes (this exercises existing code)**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: PASS. (This proptest validates the Task 2/3 implementation; it should pass immediately. If it fails, a real bug in `lower_bound`/`upper_bound` was surfaced — fix that before proceeding, and add the failing seed as a regression via the `proptest-regressions/` file proptest writes automatically.)

- [ ] **Step 3: Commit**

```bash
git add src/search.rs proptest-regressions 2>/dev/null; git add src/search.rs
git commit -m "test(svar-2): proptest search-tree bounds vs partition_point"
```

---

### Task 5: `overlap_range` resolver + edge-case unit tests

**Files:**
- Modify: `src/search.rs`
- Test: in-source `#[cfg(test)] mod tests` in `src/search.rs`

**Interfaces:**
- Consumes: `SearchTree` (`new`, `lower_bound`, `len`).
- Produces:
  - `pub fn overlap_range(tree: &SearchTree, v_ends: &[u32], max_region_length: u32, q_start: u32, q_end: u32) -> (usize, usize)` — the half-open variant index range overlapping `[q_start, q_end)`; `s == e` means empty.

- [ ] **Step 1: Write the failing edge-case tests**

Add inside `#[cfg(test)] mod tests`. Helper `v_ends_from` builds ends from starts + per-variant deletion lengths (`v_end = v_start + 1 + d`).

```rust
    /// Build exclusive ends: SNP (`d = 0`) spans one base; a deletion of `d` bases
    /// spans `v_start + 1 + d`.
    fn v_ends_from(v_starts: &[u32], dels: &[u32]) -> Vec<u32> {
        v_starts.iter().zip(dels).map(|(&s, &d)| s + 1 + d).collect()
    }

    #[test]
    fn overlap_empty_array() {
        let t = SearchTree::new(&[]);
        let (s, e) = overlap_range(&t, &[], 0, 10, 20);
        assert_eq!(s, e); // empty
    }

    #[test]
    fn overlap_snp_only_reduces_to_half_open() {
        // Pure-SNP stream: max_region_length == 0, v_end = v_start + 1.
        let v_starts = [2u32, 4, 6, 8];
        let v_ends = v_ends_from(&v_starts, &[0, 0, 0, 0]);
        let t = SearchTree::new(&v_starts);
        // query [4, 7): SNPs at 4 and 6 overlap (indices 1, 2).
        assert_eq!(overlap_range(&t, &v_ends, 0, 4, 7), (1, 3));
        // query [3, 4): nothing overlaps (SNP at 2 ends at 3, SNP at 4 starts at 4).
        let (s, e) = overlap_range(&t, &v_ends, 0, 3, 4);
        assert_eq!(s, e);
    }

    #[test]
    fn overlap_query_entirely_left_and_right() {
        let v_starts = [10u32, 20, 30];
        let v_ends = v_ends_from(&v_starts, &[0, 0, 0]);
        let t = SearchTree::new(&v_starts);
        // entirely left of all variants
        let (s, e) = overlap_range(&t, &v_ends, 0, 0, 5);
        assert_eq!(s, e);
        // entirely right of all variants
        let (s, e) = overlap_range(&t, &v_ends, 0, 40, 50);
        assert_eq!(s, e);
    }

    #[test]
    fn overlap_adjacent_non_overlap_v_end_eq_q_start() {
        // SNP at start 5 → v_end = 6. Query [6, 8): v_end == q_start ⟹ no overlap.
        let v_starts = [5u32];
        let v_ends = v_ends_from(&v_starts, &[0]);
        let t = SearchTree::new(&v_starts);
        let (s, e) = overlap_range(&t, &v_ends, 0, 6, 8);
        assert_eq!(s, e);
        // Query [5, 8): v_start=5 < 8 and q_start=5 < v_end=6 ⟹ overlaps.
        assert_eq!(overlap_range(&t, &v_ends, 0, 5, 8), (0, 1));
    }

    #[test]
    fn overlap_deletion_spans_query_start() {
        // A deletion starting BEFORE the query that reaches INTO it must be found,
        // even though its start is left of q_start. max_region_length must cover it.
        // variant 0: start 2, deletion of 6 bases → v_end = 2 + 1 + 6 = 9.
        // variant 1: SNP at start 10 → v_end 11.
        let v_starts = [2u32, 10];
        let dels = [6u32, 0];
        let v_ends = v_ends_from(&v_starts, &dels);
        let t = SearchTree::new(&v_starts);
        let max_rl = 6; // = max deletion length
        // query [5, 7): variant 0 (2..9) spans it though it starts at 2 < 5.
        assert_eq!(overlap_range(&t, &v_ends, max_rl, 5, 7), (0, 1));
    }

    #[test]
    fn overlap_interior_non_overlap_is_kept_in_range() {
        // Endpoints overlap; an interior variant that does NOT overlap stays inside
        // the returned [s_idx, e_idx) (the resolver reports min/max, not a filter).
        // v0: start 0, del 8  → v_end 9   (spans query [3,4)? 3 < 9 and 0 < 4 → YES)
        // v1: SNP start 1     → v_end 2   (1 < 4 but 3 < 2 false → NO overlap)
        // v2: SNP start 3     → v_end 4   (3 < 4 and 3 < 4 → YES)
        let v_starts = [0u32, 1, 3];
        let dels = [8u32, 0, 0];
        let v_ends = v_ends_from(&v_starts, &dels);
        let t = SearchTree::new(&v_starts);
        // s_idx = 0 (v0 overlaps), e_idx = 3 (v2 overlaps); v1 interior non-overlap.
        assert_eq!(overlap_range(&t, &v_ends, 8, 3, 4), (0, 3));
    }

    #[test]
    fn overlap_max_region_length_overshoot_is_safe() {
        // Passing max_region_length LARGER than the true max deletion still yields
        // the correct result — the forward sub-scan tightens the loose LB.
        let v_starts = [2u32, 10];
        let v_ends = v_ends_from(&v_starts, &[6, 0]);
        let t = SearchTree::new(&v_starts);
        assert_eq!(overlap_range(&t, &v_ends, 100, 5, 7), (0, 1)); // maxrl=100 ≫ 6
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: FAIL to compile — `cannot find function overlap_range in this scope`.

- [ ] **Step 3: Implement `overlap_range`**

Add this free function to `src/search.rs`, after the `impl SearchTree` block and before `#[cfg(test)]`:

```rust
/// Resolve the half-open variant index range `[s_idx, e_idx)` overlapping the query
/// `[q_start, q_end)`.
///
/// * `tree` — a [`SearchTree`] over the ascending variant starts `v_starts`.
/// * `v_ends` — parallel exclusive ends: `v_starts[i] + 1` for a SNP,
///   `v_starts[i] + 1 + d` for a deletion of `d` bases. `v_ends.len() == tree.len()`.
/// * `max_region_length` — an upper bound on any deletion length
///   (`>= max_i (v_ends[i] - v_starts[i] - 1)`); pass `0` for a SNP-only stream.
///
/// Returns `(s_idx, e_idx)` with `s_idx == e_idx` when nothing overlaps. Only the
/// endpoints are guaranteed to overlap: an interior index can be a deletion-spanned
/// non-overlap (the resolver reports the min/max overlapping index, matching SVAR
/// 1.0 `var_ranges`).
pub fn overlap_range(
    tree: &SearchTree,
    v_ends: &[u32],
    max_region_length: u32,
    q_start: u32,
    q_end: u32,
) -> (usize, usize) {
    debug_assert_eq!(v_ends.len(), tree.len());

    // LB: leftmost variant whose start could still reach q_start given the max
    // deletion span. first i with v_starts[i] + max_region_length >= q_start
    //   <=> v_starts[i] >= q_start - max_region_length (saturating at 0).
    let lb = tree.lower_bound(q_start.saturating_sub(max_region_length));
    // UB: one past the last variant starting before q_end.
    let ub = tree.lower_bound(q_end);

    // Forward sub-scan for the first true overlap (q_start < v_end).
    let mut s_idx = ub;
    for i in lb..ub {
        if q_start < v_ends[i] {
            s_idx = i;
            break;
        }
    }
    if s_idx == ub {
        return (ub, ub); // no overlap
    }
    // Backward sub-scan for the last true overlap; e_idx is exclusive.
    let mut e_idx = s_idx + 1;
    for i in (s_idx..ub).rev() {
        if q_start < v_ends[i] {
            e_idx = i + 1;
            break;
        }
    }
    (s_idx, e_idx)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: PASS — all edge-case tests green.

- [ ] **Step 5: Commit**

```bash
git add src/search.rs
git commit -m "feat(svar-2): overlap_range resolver with edge-case tests"
```

---

### Task 6: Brute-force oracle proptest for `overlap_range`

**Files:**
- Modify: `src/search.rs`
- Test: in-source `#[cfg(test)] mod tests` in `src/search.rs`

**Interfaces:**
- Consumes: `SearchTree::new`, `overlap_range`.
- Produces: no new public API — the primary correctness gate.

- [ ] **Step 1: Write the failing oracle proptest**

Add inside `#[cfg(test)] mod tests`. The oracle is an `O(n)` linear scan: variant `i` overlaps iff `v_starts[i] < q_end && q_start < v_ends[i]`.

```rust
    /// Brute-force `O(n)` overlap: returns `(min_overlap_idx, max_overlap_idx + 1)`,
    /// or an empty `(0, 0)` when nothing overlaps.
    fn overlap_oracle(
        v_starts: &[u32],
        v_ends: &[u32],
        q_start: u32,
        q_end: u32,
    ) -> (usize, usize) {
        let mut lo: Option<usize> = None;
        let mut hi = 0usize;
        for i in 0..v_starts.len() {
            if v_starts[i] < q_end && q_start < v_ends[i] {
                lo.get_or_insert(i);
                hi = i + 1;
            }
        }
        match lo {
            Some(s) => (s, hi),
            None => (0, 0),
        }
    }

    proptest! {
        // The tree+scan resolver must agree with the brute-force oracle for random
        // sorted starts, random deletion lengths (including the all-SNP case), and
        // random non-empty queries. This is the primary correctness gate.
        #[test]
        fn prop_overlap_matches_oracle(
            raw_starts in proptest::collection::vec(0u32..1000, 0..300),
            del_seeds in proptest::collection::vec(0u32..30, 0..300),
            q_start in 0u32..1000,
            q_len in 1u32..60,
        ) {
            // sorted ascending starts
            let mut v_starts = raw_starts;
            v_starts.sort_unstable();
            let n = v_starts.len();

            // per-variant deletion lengths, sized to n (0 when del_seeds is short)
            let dels: Vec<u32> = (0..n).map(|i| del_seeds.get(i).copied().unwrap_or(0)).collect();
            let v_ends: Vec<u32> = v_starts.iter().zip(&dels).map(|(&s, &d)| s + 1 + d).collect();
            let max_region_length = dels.iter().copied().max().unwrap_or(0);

            let q_end = q_start + q_len;

            let tree = SearchTree::new(&v_starts);
            let (rs, re) = overlap_range(&tree, &v_ends, max_region_length, q_start, q_end);
            let (os, oe) = overlap_oracle(&v_starts, &v_ends, q_start, q_end);

            if os == oe {
                // oracle empty → resolver must also be empty (s == e)
                prop_assert_eq!(rs, re, "expected empty; got ({rs},{re})");
            } else {
                prop_assert_eq!((rs, re), (os, oe));
            }
        }

        // Same, but with max_region_length deliberately OVERSHOT: a loose LB bound
        // must still yield the exact oracle result (forward sub-scan tightens it).
        #[test]
        fn prop_overlap_matches_oracle_overshot_maxrl(
            raw_starts in proptest::collection::vec(0u32..1000, 0..300),
            del_seeds in proptest::collection::vec(0u32..30, 0..300),
            q_start in 0u32..1000,
            q_len in 1u32..60,
        ) {
            let mut v_starts = raw_starts;
            v_starts.sort_unstable();
            let n = v_starts.len();
            let dels: Vec<u32> = (0..n).map(|i| del_seeds.get(i).copied().unwrap_or(0)).collect();
            let v_ends: Vec<u32> = v_starts.iter().zip(&dels).map(|(&s, &d)| s + 1 + d).collect();
            let true_max = dels.iter().copied().max().unwrap_or(0);
            let max_region_length = true_max + 500; // overshoot

            let q_end = q_start + q_len;
            let tree = SearchTree::new(&v_starts);
            let (rs, re) = overlap_range(&tree, &v_ends, max_region_length, q_start, q_end);
            let (os, oe) = overlap_oracle(&v_starts, &v_ends, q_start, q_end);

            if os == oe {
                prop_assert_eq!(rs, re);
            } else {
                prop_assert_eq!((rs, re), (os, oe));
            }
        }
    }
```

- [ ] **Step 2: Run to verify it passes (validates Task 5)**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: PASS. (If it fails, proptest shrinks a minimal counterexample and writes it to `proptest-regressions/`; treat that as a real bug in `overlap_range` or the bounds and fix before proceeding — do not weaken the oracle.)

- [ ] **Step 3: Run the full search module suite once more and confirm clean**

Run: `pixi run -e lint cargo test --no-default-features search::`
Expected: PASS — every `search::tests::*` test green (unit + proptests).

- [ ] **Step 4: Run the full crate test suite to confirm no regressions**

Run: `pixi run -e lint cargo test --no-default-features 2>&1 | grep "test result:"`
Expected: every line `test result: ok. ... 0 failed` — the baseline 94 tests plus the new `search` tests, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add src/search.rs proptest-regressions 2>/dev/null; git add src/search.rs
git commit -m "test(svar-2): brute-force oracle proptest for overlap_range"
```

---

## Final verification

- [ ] Run `pixi run -e lint cargo test --no-default-features 2>&1 | grep "test result:"` — all green, 0 failed.
- [ ] Run `pixi run -e lint cargo clippy --no-default-features 2>&1 | tail -20` — no warnings in `search.rs` (the codebase gates on clippy).
- [ ] Run `pixi run -e lint prek run --files src/search.rs src/lib.rs` — formatting/lint hooks pass.
- [ ] Open a PR into `svar-2` (not `main`), per the spec's worktree note.

## Notes for the integrator (future full-M5 work, not this plan)

- This core *consumes* `max_region_length`; the write-side M5 producer *emits* it per `(contig, sample, ploid)` as `max_del.npy`. The `+1` (SNP span) is applied when reconstructing `v_ends`, so the producer stores raw deletion length `d`, and `max_region_length = max_i d_i`.
- Batching (SVAR 1.0 queries many ranges at once) is a thin wrapper over `overlap_range` — out of scope here.
- SIMD `block_lower`/`block_upper` and explicit prefetch are the documented left-tree throughput wins; the scalar versions here are correct and autovectorize. Optimize only if a benchmark justifies it (measure, don't guess).

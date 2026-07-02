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
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
                assert_eq!(t.lower_bound(k), lb_ref(&a, k), "n={n} x={k}");
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
}

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

// Scaffold-only: these primitives have no callers yet — `SearchTree`/`overlap_range`
// land in a follow-up task and will use them. Remove once wired up.
#![allow(dead_code)]

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

//! PGEN variant-index shard planner.
//!
//! Splits a contig's `[0, positions.len())` variant-index range into
//! `<= max_shards` equal-count units by INDEX (`own_lo..own_hi`), then widens
//! each unit's FETCH range by POSITION to include close neighbor variants
//! (within `pad` bp of the owned boundary) so downstream left-align/context
//! logic sees them. Ownership itself is never widened -- callers filter
//! decoded atoms back down to `[own_lo, own_hi)` by index; padding only grows
//! what gets fetched from the `.pgen`, never what a unit reports as owned.

/// One PGEN work unit: owns variant indices `[own_lo, own_hi)`, but should
/// fetch the (possibly wider) `[fetch_lo, fetch_hi)` range so boundary-
/// adjacent context (e.g. left-alignment) is available. `ordinal` is the
/// unit's position among its siblings (0-based), stable regardless of how
/// units are later reordered/dispatched.
pub struct PgenUnit {
    pub own_lo: usize,
    pub own_hi: usize,
    pub fetch_lo: usize,
    pub fetch_hi: usize,
    pub ordinal: usize,
}

/// Plan PGEN work units over `positions` (the `.pvar` position of global
/// variant index `var_start + i`, sorted ascending).
///
/// Splits `[0, positions.len())` into `min(max_shards.max(1), n)` contiguous,
/// equal(-ish)-count units by INDEX -- the first `n % k` units get one extra
/// element so the split is deterministic and covers `[0, n)` with no gaps or
/// overlaps. Each unit's fetch range is then padded by POSITION: it extends
/// left/right one variant at a time while the neighbor's position is within
/// `pad` of the unit's owned boundary position, stopping at the first gap
/// wider than `pad` (or the array edge).
pub fn plan_pgen_units(positions: &[u32], max_shards: usize, pad: u32) -> Vec<PgenUnit> {
    let n = positions.len();
    if n == 0 {
        return Vec::new();
    }
    let k = max_shards.max(1).min(n);

    let base = n / k;
    let extra = n % k; // first `extra` units get one additional element

    let mut units = Vec::with_capacity(k);
    let mut own_lo = 0usize;
    for ordinal in 0..k {
        let width = base + if ordinal < extra { 1 } else { 0 };
        let own_hi = own_lo + width;

        // Extend fetch_lo leftward while the previous variant is within `pad`
        // of positions[own_lo] (one variant at a time).
        let mut fetch_lo = own_lo;
        while fetch_lo > 0 && positions[own_lo] - positions[fetch_lo - 1] <= pad {
            fetch_lo -= 1;
        }

        // Extend fetch_hi to include every variant within `pad` of the OWNERSHIP
        // BOUNDARY position (positions[own_hi]) -- a variant at/after the boundary
        // can left-align by up to `pad` INTO this shard's owned [.., own_end)
        // range, so it must be fetched even across a wide inter-variant gap.
        // Anchoring on positions[own_hi-1] (last owned) under-fetches when
        // gap(own_hi-1, own_hi) > pad, dropping a boundary indel from BOTH shards.
        // Mirrors VCF shard::plan_ranges (fetch_end = own_end + pad).
        let mut fetch_hi = own_hi;
        if own_hi < n {
            let boundary = positions[own_hi];
            while fetch_hi < n && positions[fetch_hi] - boundary <= pad {
                fetch_hi += 1;
            }
        }
        // own_hi == n (last shard): owns to u32::MAX; fetch_hi stays n.

        units.push(PgenUnit {
            own_lo,
            own_hi,
            fetch_lo,
            fetch_hi,
            ordinal,
        });
        own_lo = own_hi;
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetches_boundary_variant_even_across_wide_gap() {
        // positions far apart => left/right *interior* pads pull in no neighbors,
        // BUT unit0 must still fetch the ownership-boundary variant at index 2
        // (pos 2000): a boundary indel there can left-align up to 5bp into
        // unit0's owned [0, 2000) range, so under-fetching it would drop it from
        // both shards. Anchoring fetch_hi on the boundary position (not the last
        // owned variant) makes unit0 fetch index 2.
        let pos = vec![0, 1000, 2000, 3000];
        let u = plan_pgen_units(&pos, 2, 5);
        assert_eq!(u.len(), 2);
        assert_eq!((u[0].own_lo, u[0].own_hi), (0, 2));
        assert_eq!((u[1].own_lo, u[1].own_hi), (2, 4));
        assert_eq!((u[0].fetch_lo, u[0].fetch_hi), (0, 3)); // fetches boundary idx 2
        assert_eq!((u[1].fetch_lo, u[1].fetch_hi), (2, 4)); // last shard: owns to the end
    }

    #[test]
    fn pads_fetch_across_a_close_boundary() {
        // variant index 2 (pos 101) is within 5bp of index 1 (pos 100): shard 1 must fetch index 1.
        let pos = vec![0, 100, 101, 500];
        let u = plan_pgen_units(&pos, 2, 5);
        assert_eq!((u[1].own_lo, u[1].own_hi), (2, 4));
        assert_eq!(u[1].fetch_lo, 1); // pulled in the close left neighbor for left-align context
    }

    #[test]
    fn fetches_boundary_indel_across_gap_wider_than_pad() {
        // idx1(pos 10) and idx2(pos 2000) are 1990bp apart (> pad); unit0 must STILL
        // fetch idx2 because a boundary indel there can left-align into unit0's range.
        // This test FAILS under the old anchor (positions[own_hi-1]=pos 10, so
        // 2000-10=1990 > 5 => old fetch_hi stays 2); it passes only when fetch_hi
        // anchors on the ownership boundary positions[own_hi]=pos 2000.
        let pos = vec![0, 10, 2000, 2010];
        let u = plan_pgen_units(&pos, 2, 5);
        assert_eq!((u[0].own_lo, u[0].own_hi), (0, 2));
        assert_eq!(u[0].fetch_hi, 3); // fetches the boundary variant idx2 despite the wide gap
    }

    #[test]
    fn empty_in_empty_out() {
        assert!(plan_pgen_units(&[], 4, 5).is_empty());
    }
}

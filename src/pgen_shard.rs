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

        // Extend fetch_hi rightward while the next variant is within `pad`
        // of positions[own_hi - 1] (one variant at a time).
        let mut fetch_hi = own_hi;
        while fetch_hi < n && positions[fetch_hi] - positions[own_hi - 1] <= pad {
            fetch_hi += 1;
        }

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
    fn equal_count_split_no_padding_when_gaps_wide() {
        // positions far apart => pad pulls in no neighbors.
        let pos = vec![0, 1000, 2000, 3000];
        let u = plan_pgen_units(&pos, 2, 5);
        assert_eq!(u.len(), 2);
        assert_eq!((u[0].own_lo, u[0].own_hi), (0, 2));
        assert_eq!((u[1].own_lo, u[1].own_hi), (2, 4));
        assert_eq!((u[0].fetch_lo, u[0].fetch_hi), (0, 2)); // no neighbor within 5bp
        assert_eq!((u[1].fetch_lo, u[1].fetch_hi), (2, 4));
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
    fn empty_in_empty_out() {
        assert!(plan_pgen_units(&[], 4, 5).is_empty());
    }
}

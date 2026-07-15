//! Backend-agnostic work-unit shard planner.
//!
//! `plan_ranges` splits coalesced, sorted, disjoint `owned` ranges into ordered
//! [`WorkUnit`]s of roughly `target_span`, capped at `max_shards` total units,
//! padding each unit's fetch bounds by `pad` (saturating) on either side. This
//! is the backend-agnostic core of what was `vcf_reader::plan_vcf_shards`'s
//! span math -- VCF (byte-position ranges) and PGEN (variant-index ranges)
//! both reduce to "split disjoint `u32` ranges into padded chunks", so that
//! math lives here once and each backend supplies only its own
//! coalescing/validation and unit-to-domain error handling.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkUnit {
    pub own_start: u32,
    pub own_end: u32,
    pub fetch_start: u32,
    pub fetch_end: u32,
    pub ordinal: usize,
}

/// Split coalesced, sorted, disjoint `owned` ranges into ordered units of
/// ~`target_span`, capped at `max_shards`, padding fetch by `pad` (saturating)
/// on each side.
pub fn plan_ranges(
    owned: &[(u32, u32)],
    max_shards: usize,
    target_span: u32,
    pad: u32,
) -> Vec<WorkUnit> {
    if owned.is_empty() {
        return Vec::new();
    }
    let max_shards = max_shards.max(1);
    let total: u64 = owned.iter().map(|&(s, e)| u64::from(e - s)).sum();
    let span = total
        .div_ceil(max_shards as u64)
        .max(u64::from(target_span.max(1)))
        .min(u64::from(u32::MAX)) as u32;
    let mut out = Vec::new();
    for &(region_start, region_end) in owned {
        let mut own_start = region_start;
        while own_start < region_end {
            let own_end = own_start.saturating_add(span).min(region_end);
            out.push(WorkUnit {
                own_start,
                own_end,
                fetch_start: own_start.saturating_sub(pad),
                fetch_end: own_end.saturating_add(pad),
                ordinal: out.len(),
            });
            own_start = own_end;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_owned_ranges_with_padded_fetches() {
        let u = plan_ranges(&[(0, 12)], 3, 4, 5);
        assert_eq!(
            u.iter()
                .map(|s| (s.own_start, s.own_end, s.ordinal))
                .collect::<Vec<_>>(),
            vec![(0, 4, 0), (4, 8, 1), (8, 12, 2)]
        );
        assert_eq!(u[0].fetch_start, 0);
        assert!(u[0].fetch_end >= u[1].own_start);
        assert!(u[1].fetch_start <= u[0].own_end);
    }

    #[test]
    fn max_shards_is_an_upper_bound() {
        let u = plan_ranges(&[(0, 100)], 4, 1, 0);
        assert_eq!(u.len(), 4);
        assert_eq!(
            u.iter()
                .map(|s| (s.own_start, s.own_end))
                .collect::<Vec<_>>(),
            vec![(0, 25), (25, 50), (50, 75), (75, 100)]
        );
    }

    #[test]
    fn empty_in_empty_out() {
        assert!(plan_ranges(&[], 4, 10, 5).is_empty());
    }
}

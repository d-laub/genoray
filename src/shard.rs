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

/// How many `chunk_size`-record chunks one work unit should cover.
///
/// This is the knob that trades indexed-fetch padding against reorder-frontier
/// width. Every unit re-decodes `normalize::L_MAX` (1000) bp of padding on
/// each side, so smaller units cost redundant decode; larger units make the
/// reorder head advance in coarser jumps, which is what starved the executor
/// in issue #169. At 4 chunks of 5,000 records on chr21 density (~220
/// records/kbp) a unit spans ~91 kbp, so padding is ~2% of decoded records.
///
/// STARTING VALUE, not a fitted constant -- set it from the sweep in
/// `docs/superpowers/specs/2026-09-05-svar2-reader-frontier-design.md`
/// section D and record the measurement here when you do.
pub const UNITS_TARGET_CHUNKS: usize = 4;

/// Hard cap on work units per contig. Each unit is an independent indexed
/// fetch, so this is also a per-contig seek cap -- the guard against a
/// pathological `chunk_size`/record-count combination asking for millions of
/// seeks. The cap wins over the worker floor.
pub const MAX_UNITS_PER_CONTIG: usize = 4096;

/// How many work units to split one contig into.
///
/// `contig_records` is `Some` ONLY when the caller holds EXACT per-contig
/// record counts (`contig_cost::ContigCosts::exact_counts`). The
/// header-length fallback tier's values are base pairs, a different unit
/// entirely, and feeding them here would mis-size the frontier by orders of
/// magnitude -- pass `None` and take the pre-#169 `workers * overshard_factor`
/// shape instead.
pub fn plan_unit_count(
    contig_records: Option<u64>,
    workers: usize,
    chunk_size: usize,
    overshard_factor: usize,
) -> usize {
    let workers = workers.max(1);
    match contig_records {
        None => workers
            .saturating_mul(overshard_factor.max(1))
            .clamp(1, MAX_UNITS_PER_CONTIG),
        Some(records) => {
            let per_unit = (UNITS_TARGET_CHUNKS as u64).saturating_mul(chunk_size.max(1) as u64);
            let target = records.div_ceil(per_unit.max(1));
            // Order matters: the worker floor keeps every worker fed on a
            // small contig, then the seek cap overrides it -- a worker count
            // above the cap gets the cap, not a runaway unit count.
            usize::try_from(target)
                .unwrap_or(MAX_UNITS_PER_CONTIG)
                .max(workers)
                .clamp(1, MAX_UNITS_PER_CONTIG)
        }
    }
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

    #[test]
    fn over_decomposes_beyond_worker_count() {
        // workers=4, factor=4 => up to 16 units over a big contig.
        let u = plan_ranges(&[(0, 16000)], 16, 1000, 5);
        assert_eq!(u.len(), 16);
    }

    #[test]
    fn plan_unit_count_falls_back_to_overshard_without_exact_counts() {
        // No exact record count -> the pre-#169 shape, so a header-length
        // fallback tier's base-pair values can never be mistaken for records.
        assert_eq!(plan_unit_count(None, 20, 5_000, 4), 80);
        assert_eq!(plan_unit_count(None, 1, 5_000, 4), 4);
    }

    #[test]
    fn plan_unit_count_targets_four_chunks_of_records_per_unit() {
        // chr21 of the #169 report: 10.1M records at chunk_size 5,000.
        // 4 chunks/unit -> 20,000 records/unit -> 505 units, versus the 80
        // the old `workers * OVERSHARD_FACTOR` shape produced at w=20.
        assert_eq!(plan_unit_count(Some(10_100_000), 20, 5_000, 4), 505);
        // Independent of the worker count once the record floor dominates.
        assert_eq!(plan_unit_count(Some(10_100_000), 3, 5_000, 4), 505);
    }

    #[test]
    fn plan_unit_count_floors_at_the_worker_count() {
        // A tiny contig must still give every worker something to steal.
        assert_eq!(plan_unit_count(Some(100), 20, 5_000, 4), 20);
    }

    #[test]
    fn plan_unit_count_caps_at_max_units_per_contig() {
        // Each unit is an independent indexed fetch, so the unit count is
        // also a seek count -- the cap wins over both the record target and
        // the worker floor.
        assert_eq!(
            plan_unit_count(Some(u64::MAX), 20, 1, 4),
            MAX_UNITS_PER_CONTIG
        );
        assert_eq!(
            plan_unit_count(Some(100), 100_000, 5_000, 4),
            MAX_UNITS_PER_CONTIG
        );
    }

    #[test]
    fn plan_unit_count_is_never_zero() {
        assert_eq!(plan_unit_count(Some(0), 1, 5_000, 4), 1);
        assert_eq!(plan_unit_count(None, 0, 0, 0), 1);
    }
}

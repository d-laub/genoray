//! Ungated SVAR1 range-query core: the query counterpart to the conversion-gated
//! `svar1_reader::Svar1RecordSource`.
//!
//! Two independent stages, mirroring `python/genoray/_var_ranges.py` +
//! `python/genoray/_svar/_kernels.py::_find_starts_ends`:
//!
//! * [`var_ranges`] — POS ranges -> global variant-id ranges. Pure; a thin wrapper
//!   over `search::overlap_range`, which already ports the Python algorithm.
//! * [`find_ranges`] — variant-id ranges -> absolute CSR index pairs into the
//!   `variant_idxs` mmap, via two `partition_point`s per haplotype.
//!
//! There is deliberately **no `gather_ranges`**: SVAR2 needs one because it merges
//! two channels and decodes keys, but SVAR1's on-disk layout is already the target
//! representation, so consumers build a zero-copy view straight from the index pairs
//! (cf. `SparseVar.read_ranges` -> `Ragged.from_offsets`).

use std::ops::Range;

use crate::search::{SearchTree, overlap_range};

/// POS ranges -> **global** half-open variant-id ranges, one per region, in
/// `regions` order.
///
/// * `v_starts` / `v_ends` — this contig's LOCAL 0-based variant starts (ascending)
///   and exclusive ends (`v_end = POS - min(ILEN, 0)`; a SNP at `s` has `v_end == s+1`).
/// * `max_v_len` — `max(v_ends - v_starts)` over the contig, i.e. **Python's
///   `var_ranges` convention** (`_var_ranges.py:78`). `overlap_range` only requires a
///   `>=` bound on the deletion span, so this over-estimates by exactly 1 and is
///   provably overshoot-safe (it merely widens the candidate window). Do NOT subtract
///   1 to "tighten" it — under-estimating IS a correctness bug.
/// * `contig_start` — this contig's first variant's GLOBAL id. Contigs are contiguous
///   in global-id space.
///
/// Nothing overlapping yields an **in-bounds zero-length** range (`start == end`),
/// never a sentinel: an out-of-range offset is poison for downstream byte math
/// (seqpro `Ragged.to_packed` overflows int64 even for an empty row). This
/// deliberately differs from Python `var_ranges`, which returns `INT32_MAX`.
///
/// Only the endpoints are guaranteed to overlap — an interior id can be a
/// deletion-spanned non-overlap. Same contract as `search::overlap_range` and SVAR 1.0
/// `var_ranges`.
pub fn var_ranges(
    v_starts: &[u32],
    v_ends: &[u32],
    max_v_len: u32,
    contig_start: u32,
    regions: &[(u32, u32)],
) -> Vec<Range<u32>> {
    debug_assert_eq!(v_starts.len(), v_ends.len());
    // An empty contig has no tree to build and no ends to scan.
    if v_starts.is_empty() {
        return regions.iter().map(|_| contig_start..contig_start).collect();
    }
    // One tree for the whole batch: `overlap_range` is called per region but the
    // tree build is hoisted, mirroring the SVAR2 search/gather split's intent.
    let tree = SearchTree::new(v_starts);
    regions
        .iter()
        .map(|&(q_start, q_end)| {
            let (s, e) = overlap_range(&tree, v_ends, max_v_len, q_start, q_end);
            (contig_start + s as u32)..(contig_start + e as u32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Three variants on a contig whose global ids start at 100.
    // local 0: SNP  at 10 -> v_end 11
    // local 1: DEL  at 20, ILEN -3 -> v_end 23
    // local 2: SNP  at 30 -> v_end 31
    // max_v_len (Python convention) = max(v_ends - v_starts) = max(1, 3, 1) = 3
    fn fixture() -> (Vec<u32>, Vec<u32>, u32) {
        (vec![10, 20, 30], vec![11, 23, 31], 3)
    }

    #[test]
    fn var_ranges_maps_local_overlap_to_global_ids() {
        let (vs, ve, mvl) = fixture();
        // [10, 21) overlaps local 0 (SNP@10) and local 1 (DEL@20) -> global 100..102
        let got = var_ranges(&vs, &ve, mvl, 100, &[(10, 21)]);
        assert_eq!(got, vec![100..102]);
    }

    #[test]
    fn var_ranges_deletion_spanning_query_start_is_included() {
        // The whole point of the sub-scan: a DEL starting BEFORE the query still
        // deletes bases inside it. Query [21, 22) starts after the DEL's POS (20)
        // but before its end (23), so local 1 must be included.
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 100, &[(21, 22)]);
        assert_eq!(got, vec![101..102]);
    }

    #[test]
    fn var_ranges_no_overlap_is_zero_length_not_sentinel() {
        // A zero-length in-bounds range -- NEVER a sentinel like u32::MAX. An
        // out-of-range offset overflows int64 in seqpro's Ragged.to_packed.
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 100, &[(50, 60)]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].start, got[0].end, "no-overlap must be zero-length");
    }

    #[test]
    fn var_ranges_empty_contig_yields_zero_length_ranges() {
        // n_local == 0: must not panic (a .max() over an empty slice would).
        let got = var_ranges(&[], &[], 0, 42, &[(0, 100), (5, 6)]);
        assert_eq!(got, vec![42..42, 42..42]);
    }

    #[test]
    fn var_ranges_batches_regions_in_order() {
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 0, &[(30, 31), (10, 11)]);
        assert_eq!(got, vec![2..3, 0..1], "output must be in `regions` order");
    }
}

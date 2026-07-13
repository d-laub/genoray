//! SVAR1 record source: reconstruct variant-major `RawRecord`s from SVAR1's
//! sample-major sparse store, so `from_svar1` reuses the shared conversion spine
//! (`chunk_assembler` onward) exactly as VCF/PGEN do. See
//! `docs/superpowers/specs/2026-07-13-svar1-to-svar2-conversion-design.md`.

/// Invert SVAR1's sample-major CSR (`variant_idxs`/`offsets`) into a variant-major
/// carrier list for ONE contig. Returns, per local variant `0..n_local`, the
/// `(haplotype column, flat entry index)` pairs of the haplotypes carrying it.
///
/// `variant_idxs` holds each haplotype's sorted global non-ref variant ids;
/// `offsets` is the CSR over `num_haps = num_samples * ploidy` haplotypes
/// (`offsets.len() == num_haps + 1`). Contigs are contiguous in global-id space,
/// so this contig owns global ids `[contig_start, contig_start + n_local)`; per
/// hap we binary-search that sub-range (ids are sorted) rather than scanning all
/// entries. `flat entry index` indexes both `variant_idxs` and any per-entry
/// field array (they share `offsets`).
pub fn build_variant_major(
    variant_idxs: &[i32],
    offsets: &[i64],
    num_haps: usize,
    contig_start: i32,
    n_local: usize,
) -> Vec<Vec<(u32, u64)>> {
    let contig_end = contig_start + n_local as i32;
    let mut buckets: Vec<Vec<(u32, u64)>> = vec![Vec::new(); n_local];
    for h in 0..num_haps {
        let lo = offsets[h] as usize;
        let hi = offsets[h + 1] as usize;
        let hap = &variant_idxs[lo..hi];
        let s = hap.partition_point(|&g| g < contig_start);
        let e = hap.partition_point(|&g| g < contig_end);
        for (k, &g) in hap.iter().enumerate().take(e).skip(s) {
            let local = (g - contig_start) as usize;
            buckets[local].push((h as u32, (lo + k) as u64));
        }
    }
    buckets
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2 samples × ploidy 2 = 4 haplotypes. Global variant ids 0..5.
    // Contig under test starts at global id 2, has 3 local variants (ids 2,3,4).
    // Per-hap sorted global ids (offsets CSR):
    //   hap0: [0, 2, 4]   hap1: [3]   hap2: [2]   hap3: []
    #[test]
    fn transpose_buckets_carriers_by_local_variant() {
        let variant_idxs: Vec<i32> = vec![0, 2, 4, /*h0*/ 3, /*h1*/ 2 /*h2*/];
        let offsets: Vec<i64> = vec![0, 3, 4, 5, 5]; // len num_haps+1 = 5
        let got = build_variant_major(&variant_idxs, &offsets, 4, 2, 3);

        // local 0 (gid 2): hap0 at entry 1, hap2 at entry 4
        assert_eq!(got[0], vec![(0u32, 1u64), (2u32, 4u64)]);
        // local 1 (gid 3): hap1 at entry 3
        assert_eq!(got[1], vec![(1u32, 3u64)]);
        // local 2 (gid 4): hap0 at entry 2
        assert_eq!(got[2], vec![(0u32, 2u64)]);
    }

    #[test]
    fn transpose_empty_contig_is_all_empty() {
        let got = build_variant_major(&[0, 1], &[0, 1, 2, 2, 2], 4, 100, 2);
        assert_eq!(got, vec![Vec::new(), Vec::new()]);
    }
}

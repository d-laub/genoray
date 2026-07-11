//! Streaming mutation-catalogue counting with on-the-fly DBS pairing.

use ndarray::Array2;

use crate::dense::DenseClass;
use crate::layout::{ContigPaths, MutcatSub};
use crate::mutcat::classify::dbs78_code;
use crate::mutcat::sidecar::{MutcatView, open_sidecar};
use crate::mutcat::{DBS78_OFFSET, ID83_OFFSET, N_CODES, SBS96_OFFSET, UNCLASSIFIED};
use crate::query::ContigReader;
use crate::query::sidecar::as_bytes;
use svar2_codec::{decode_snp_2bit, unpack_snp_key_at};

/// One SNV on a single haplotype (already position-sorted & deduped).
#[derive(Debug, Clone, Copy)]
pub struct SnvCall {
    pub pos: u32,
    pub sbs: u8,   // SBS96 class-local index or sentinel
    pub ref_i: u8, // 2-bit ref base code
    pub alt_i: u8, // 2-bit alt base code
}

/// Emit one unified mutation code per SNV (SBS, or DBS once for an isolated
/// adjacent pair — the pair contributes a single DBS call, not one per
/// member). Sentinels are skipped. Mirrors v1 isolation logic.
pub fn emit_snv_codes(snvs: &[SnvCall], out: &mut impl FnMut(usize)) {
    let n = snvs.len();
    let mut j = 0;
    while j < n {
        let v = snvs[j];
        // try to pair v with the next SNV
        if j + 1 < n && snvs[j + 1].pos == v.pos + 1 {
            let w = snvs[j + 1];
            // isolation: no adjacent SNV before v, none after w.
            let before = j > 0 && snvs[j - 1].pos + 1 == v.pos;
            let after = j + 2 < n && snvs[j + 2].pos == w.pos + 1;
            if !before && !after {
                let code = dbs78_code(
                    decode_snp_2bit(v.ref_i),
                    decode_snp_2bit(v.alt_i),
                    decode_snp_2bit(w.ref_i),
                    decode_snp_2bit(w.alt_i),
                );
                if code != UNCLASSIFIED {
                    out(DBS78_OFFSET + code as usize);
                    j += 2;
                    continue;
                }
            }
        }
        if (v.sbs as usize) < 96 {
            out(SBS96_OFFSET + v.sbs as usize);
        }
        j += 1;
    }
    #[allow(clippy::assertions_on_constants)]
    {
        debug_assert!(N_CODES == 257);
    }
}

/// The four mutcat sidecar views for one contig, bundled so `count_contig`
/// takes a single handle instead of four.
pub struct Sidecars {
    pub vk_snp: MutcatView,
    pub vk_indel: MutcatView,
    pub dense_snp: MutcatView,
    pub dense_indel: MutcatView,
}

impl Sidecars {
    /// Open all four mutcat sub-stream sidecars for `paths`'s contig. Any
    /// sub-stream with no sidecar on disk opens as an empty `MutcatView`
    /// (`open_sidecar`'s existing missing-file tolerance).
    pub fn open(paths: &ContigPaths) -> std::io::Result<Sidecars> {
        Ok(Sidecars {
            vk_snp: open_sidecar(paths, MutcatSub::VkSnp)?,
            vk_indel: open_sidecar(paths, MutcatSub::VkIndel)?,
            dense_snp: open_sidecar(paths, MutcatSub::DenseSnp)?,
            dense_indel: open_sidecar(paths, MutcatSub::DenseIndel)?,
        })
    }
}

/// Accumulate one contig's full mutation-count matrix into `acc`
/// (`(n_samples, N_CODES)`).
///
/// Walks every flat `(sample, ploid)` column (`col = sample * ploidy +
/// ploid`, matching `ContigReader::vk_slice`/`dense_carried`'s convention).
/// For each column:
/// 1. Gathers that hap's SNVs from `var_key/snp` (per-call, sidecar-indexed
///    by the absolute call index) and `dense/snp` (per dense variant the hap
///    carries, sidecar-indexed by the dense variant column), merges by
///    position, dedups, and runs `emit_snv_codes` for on-the-fly DBS pairing.
/// 2. Gathers that hap's indels from `var_key/indel` + `dense/indel`; each
///    emits `ID83_OFFSET + code` directly (no pairing). A sentinel code
///    (`UNCLASSIFIED`/`NOT_ANNOTATED`, both `>= N_ID83`) pushes the unified
///    code past `N_CODES`, so the single `unified < N_CODES` bound filters
///    both sentinels without naming them.
/// 3. Accumulates into `acc[[sample, unified_code]]`: `per_sample` marks each
///    code present at most once per sample (across all its ploids), else
///    every hit increments by one.
///
/// Single-threaded (rayon parallelization is a later task).
pub fn count_contig(
    reader: &ContigReader,
    sidecars: &Sidecars,
    per_sample: bool,
    acc: &mut Array2<i64>,
) {
    let ploidy = reader.ploidy;
    for sample in 0..reader.n_samples {
        // Presence-mode bookkeeping: at most one mark per (sample, code),
        // even if the sample carries the same code on multiple ploids.
        let mut seen = per_sample.then(|| vec![false; N_CODES]);
        let mut bump = |code: usize| {
            if let Some(seen) = seen.as_mut() {
                if seen[code] {
                    return;
                }
                seen[code] = true;
            }
            acc[[sample, code]] += 1;
        };

        for p in 0..ploidy {
            let col = sample * ploidy + p;

            // --- SNVs: var_key/snp + dense/snp, merged by position ---
            let mut snvs: Vec<SnvCall> = Vec::new();
            {
                let vk_range = reader.vk_snp.column(col);
                let (o0, o1) = (vk_range.start, vk_range.end);
                let positions = &reader.vk_snp.positions()[o0..o1];
                let keys = as_bytes(&reader.vk_snp.keys);
                for (i, &pos) in positions.iter().enumerate() {
                    let abs_i = o0 + i;
                    snvs.push(SnvCall {
                        pos,
                        sbs: sidecars.vk_snp.code_at(abs_i),
                        ref_i: sidecars.vk_snp.ref_at(abs_i),
                        alt_i: unpack_snp_key_at(keys, abs_i),
                    });
                }
            }
            if let Some(dense) = reader.dense_view(DenseClass::Snp) {
                let keys = as_bytes(&dense.keys);
                for (dcol, &pos) in dense.positions().iter().enumerate() {
                    if dense.carried(col, dcol) {
                        snvs.push(SnvCall {
                            pos,
                            sbs: sidecars.dense_snp.code_at(dcol),
                            ref_i: sidecars.dense_snp.ref_at(dcol),
                            alt_i: unpack_snp_key_at(keys, dcol),
                        });
                    }
                }
            }
            snvs.sort_by_key(|s| s.pos);
            snvs.dedup_by_key(|s| s.pos);
            emit_snv_codes(&snvs, &mut bump);

            // --- indels: var_key/indel + dense/indel, each direct (no pairing) ---
            {
                let vk_range = reader.vk_indel.column(col);
                for abs_i in vk_range {
                    let unified = ID83_OFFSET + sidecars.vk_indel.code_at(abs_i) as usize;
                    if unified < N_CODES {
                        bump(unified);
                    }
                }
            }
            if let Some(dense) = reader.dense_view(DenseClass::Indel) {
                for dcol in 0..dense.n_dense_variants {
                    if dense.carried(col, dcol) {
                        let unified = ID83_OFFSET + sidecars.dense_indel.code_at(dcol) as usize;
                        if unified < N_CODES {
                            bump(unified);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snv(pos: u32, ref_b: u8, alt_b: u8) -> SnvCall {
        use crate::mutcat::classify::sbs96_code;
        SnvCall {
            pos,
            sbs: sbs96_code(b'A', ref_b, alt_b, b'A'), // dummy flanks for the test
            ref_i: svar2_codec::encode_snp_2bit(ref_b),
            alt_i: svar2_codec::encode_snp_2bit(alt_b),
        }
    }

    #[test]
    fn isolated_pair_emits_one_dbs() {
        let snvs = [snv(10, b'A', b'C'), snv(11, b'C', b'A')];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        // AC>CA is DBS index 0 → unified 96; the doublet contributes exactly once.
        assert_eq!(codes, vec![96]);
    }

    #[test]
    fn run_of_three_stays_sbs() {
        let snvs = [
            snv(10, b'C', b'A'),
            snv(11, b'C', b'A'),
            snv(12, b'C', b'A'),
        ];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        assert_eq!(
            codes.len(),
            3,
            "all three SNVs in the run must emit an SBS code"
        );
        assert!(
            codes.iter().all(|&c| c < DBS78_OFFSET),
            "runs of 3 must stay SBS"
        );
    }

    #[test]
    fn non_adjacent_snvs_are_sbs() {
        let snvs = [snv(10, b'C', b'A'), snv(20, b'C', b'A')];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        assert!(codes.iter().all(|&c| c < DBS78_OFFSET));
        assert_eq!(codes.len(), 2);
    }
}

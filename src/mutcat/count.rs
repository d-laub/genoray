//! Streaming mutation-catalogue counting with on-the-fly DBS pairing.

use crate::mutcat::classify::dbs78_code;
use crate::mutcat::{DBS78_OFFSET, N_CODES, SBS96_OFFSET, UNCLASSIFIED};
use svar2_codec::decode_snp_2bit;

/// One SNV on a single haplotype (already position-sorted & deduped).
#[derive(Debug, Clone, Copy)]
pub struct SnvCall {
    pub pos: u32,
    pub sbs: u8,   // SBS96 class-local index or sentinel
    pub ref_i: u8, // 2-bit ref base code
    pub alt_i: u8, // 2-bit alt base code
}

/// Emit one unified mutation code per SNV (SBS or, for isolated adjacent pairs,
/// DBS for both members). Sentinels are skipped. Mirrors v1 isolation logic.
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
    fn isolated_pair_emits_two_dbs() {
        let snvs = [snv(10, b'A', b'C'), snv(11, b'C', b'A')];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        // AC>CA is DBS index 0 → unified 96; both members contribute.
        assert_eq!(codes, vec![96, 96]);
    }

    #[test]
    fn run_of_three_stays_sbs() {
        let snvs = [
            snv(10, b'C', b'A'),
            snv(11, b'C', b'A'),
            snv(12, b'C', b'A'),
        ];
        let mut n_dbs = 0;
        emit_snv_codes(&snvs, &mut |c| {
            if c >= DBS78_OFFSET {
                n_dbs += 1;
            }
        });
        assert_eq!(n_dbs, 0, "runs of 3 must stay SBS");
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

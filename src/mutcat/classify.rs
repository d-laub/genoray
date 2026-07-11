//! Pure mutation classifiers (SBS96, ID83, DBS78). Port of
//! python/genoray/_mutcat/classify.py. No I/O; operates on ASCII bytes +
//! reference slices.

use crate::mutcat::UNCLASSIFIED;

/// A=0 C=1 G=2 T=3, else -1 (N or non-ACGT).
#[inline]
pub fn base_index(b: u8) -> i8 {
    match b {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => -1,
    }
}

// (ref_idx, alt_idx) -> SBS substitution index 0..5 for pyrimidine-folded refs.
// Order: C>A, C>G, C>T, T>A, T>C, T>G  (A=0,C=1,G=2,T=3).
const SUB_LUT: [[i8; 4]; 4] = {
    let mut lut = [[-1i8; 4]; 4];
    // C(1)>A(0),C>G(2),C>T(3)
    lut[1][0] = 0;
    lut[1][2] = 1;
    lut[1][3] = 2;
    // T(3)>A(0),T>C(1),T>G(2)
    lut[3][0] = 3;
    lut[3][1] = 4;
    lut[3][2] = 5;
    lut
};

/// SBS96 class-local index (0..=95) or `UNCLASSIFIED`.
/// `five`/`three`: immediate reference flanks; `refb`/`altb`: REF/ALT bases.
pub fn sbs96_code(five: u8, refb: u8, altb: u8, three: u8) -> u8 {
    let r = base_index(refb);
    let a = base_index(altb);
    let f = base_index(five);
    let t = base_index(three);
    if r < 0 || a < 0 || f < 0 || t < 0 || r == a {
        return UNCLASSIFIED;
    }
    let purine = r == 0 || r == 2; // A or G -> fold
    let (rr, aa, ff, tt) = if purine {
        // fold: complement ref/alt, and flanks swap+complement
        (3 - r, 3 - a, 3 - t, 3 - f)
    } else {
        (r, a, f, t)
    };
    let sub = SUB_LUT[rr as usize][aa as usize];
    if sub < 0 {
        return UNCLASSIFIED;
    }
    (sub as u8) * 16 + (ff as u8) * 4 + (tt as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutcat::UNCLASSIFIED;

    #[test]
    fn sbs96_pyrimidine_ref_no_fold() {
        // A[C>A]G : sub C>A = 0, five=A=0, three=G=2 -> 0*16 + 0*4 + 2 = 2
        assert_eq!(sbs96_code(b'A', b'C', b'A', b'G'), 2);
    }

    #[test]
    fn sbs96_purine_ref_folds() {
        // Purine ref folds to its pyrimidine partner with flanks swapped+complemented.
        // G>T at flanks A..C  == (fold) C>A at flanks G..T.
        // Direct fold: r=G(2)->rr=1(C), a=T(3)->aa=0(A) => sub C>A=0.
        // five=A(0)->ff=comp(three=C(1))=2(G); three=C(1)->tt=comp(five=A(0))=3(T).
        // code = 0*16 + 2*4 + 3 = 11.
        assert_eq!(sbs96_code(b'A', b'G', b'T', b'C'), 11);
    }

    #[test]
    fn sbs96_rejects_non_acgt_and_ref_eq_alt() {
        assert_eq!(sbs96_code(b'N', b'C', b'A', b'G'), UNCLASSIFIED);
        assert_eq!(sbs96_code(b'A', b'C', b'C', b'G'), UNCLASSIFIED); // ref==alt
    }
}

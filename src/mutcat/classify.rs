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

// ID83 lookup tables, built to match codebook.py ID83 order.
// ID83 label order (see _build_id83): 24 single-base (Del/Ins x C/T x rep0..5),
// then 48 repeat (Del/Ins x size2..5 x rep0..5), then 11 microhomology-Del.
struct Id83Luts {
    id1: [[[u8; 6]; 2]; 2], // [kind(Del=0,Ins=1)][base(C=0,T=1)][rep0..5]
    idr: [[[u8; 6]; 4]; 2], // [kind][size_bucket(2..5)][rep0..5]
    idm: [[u8; 6]; 4],      // [size_bucket][mh]
}

const fn build_id83_luts() -> Id83Luts {
    // Index formula mirrors codebook order:
    //   single: base = (kind*2 + base_ct)*6 + rep                          [0..24)
    //   repeat: 24 + (kind*4 + size_bucket)*6 + rep                        [24..72)
    //   mh    : 72 + cumulative(mh caps 1,2,3,5) offset + (m-1)            [72..83)
    let u = UNCLASSIFIED;
    let mut id1 = [[[u; 6]; 2]; 2];
    let mut idr = [[[u; 6]; 4]; 2];
    let mut idm = [[u; 6]; 4];
    let mut kind = 0;
    while kind < 2 {
        let mut b = 0;
        while b < 2 {
            let mut r = 0;
            while r < 6 {
                id1[kind][b][r] = ((kind * 2 + b) * 6 + r) as u8;
                r += 1;
            }
            b += 1;
        }
        let mut s = 0;
        while s < 4 {
            let mut r = 0;
            while r < 6 {
                idr[kind][s][r] = (24 + (kind * 4 + s) * 6 + r) as u8;
                r += 1;
            }
            s += 1;
        }
        kind += 1;
    }
    // microhomology (Del only): caps per size bucket = [1,2,3,5]; base offset 72.
    let caps = [1usize, 2, 3, 5];
    let mut s = 0;
    let mut base = 72usize;
    while s < 4 {
        let mut m = 1;
        while m <= caps[s] {
            idm[s][m] = (base + (m - 1)) as u8;
            m += 1;
        }
        base += caps[s];
        s += 1;
    }
    Id83Luts { id1, idr, idm }
}

const ID83_LUTS: Id83Luts = build_id83_luts();
const MH_CAP: [usize; 4] = [1, 2, 3, 5];

/// ID83 class-local index (0..=82) or `UNCLASSIFIED`. Port of `_id83_kernel`.
pub fn id83_code(seq: &[u8], pos0: usize, refa: &[u8], alta: &[u8]) -> u8 {
    let n = seq.len();
    let rl = refa.len();
    let al = alta.len();
    // atomized indels always share an anchor base; require it and equal anchors.
    if rl == 0 || al == 0 || refa[0] != alta[0] {
        return UNCLASSIFIED;
    }
    let is_del = rl > al;
    // deleted/inserted unit = allele[1..]
    let (buf, ilen): (&[u8], usize) = if is_del {
        (&refa[1..], rl - 1)
    } else {
        (&alta[1..], al - 1)
    };
    if ilen == 0 {
        return UNCLASSIFIED;
    }
    for &c in buf {
        if base_index(c) < 0 {
            return UNCLASSIFIED;
        }
    }
    // count tandem repeats of the unit downstream from pos0+1
    let scan = pos0 + 1;
    let mut n_rep = 0usize;
    let mut i = 0usize;
    while scan + i + ilen <= n {
        let mut m = true;
        let mut j = 0;
        while j < ilen {
            if seq[scan + i + j] != buf[j] {
                m = false;
                break;
            }
            j += 1;
        }
        if !m {
            break;
        }
        n_rep += 1;
        i += ilen;
    }
    if ilen == 1 {
        let mut bi = base_index(buf[0]);
        if bi == 0 || bi == 2 {
            bi = 3 - bi; // A/G -> pyrimidine partner
        }
        let base_ct = if bi == 1 { 0 } else { 1 }; // C->0, T->1
        if is_del && n_rep == 0 {
            return UNCLASSIFIED; // v1 _REF_MISMATCH -> UNCLASSIFIED
        }
        let mut rep = if is_del { n_rep - 1 } else { n_rep };
        if rep > 5 {
            rep = 5;
        }
        return ID83_LUTS.id1[if is_del { 0 } else { 1 }][base_ct][rep];
    }
    let sb = if ilen < 5 { ilen } else { 5 };
    let si = sb - 2; // 0..3
    let rep;
    if is_del {
        // microhomology: longest prefix of the unit matching downstream
        let mut mh = 0usize;
        let mut kk = 1;
        while kk < ilen {
            let mut eq = true;
            let mut j = 0;
            while j < kk {
                if scan + j >= n || seq[scan + j] != buf[j] {
                    eq = false;
                    break;
                }
                j += 1;
            }
            if eq {
                mh = kk;
            }
            kk += 1;
        }
        if mh > 0 && n_rep <= 1 {
            let cap = MH_CAP[si];
            let m = if mh < cap { mh } else { cap };
            return ID83_LUTS.idm[si][m];
        }
        if n_rep == 0 {
            return UNCLASSIFIED; // _REF_MISMATCH
        }
        rep = n_rep - 1;
    } else {
        rep = n_rep;
    }
    let rep = if rep > 5 { 5 } else { rep };
    ID83_LUTS.idr[if is_del { 0 } else { 1 }][si][rep]
}

// Canonical DBS78 doublet labels, codebook order (must match codebook.py DBS78).
const DBS78_LABELS: [&str; 78] = [
    "AC>CA", "AC>CG", "AC>CT", "AC>GA", "AC>GG", "AC>GT", "AC>TA", "AC>TG", "AC>TT", "AT>CA",
    "AT>CC", "AT>CG", "AT>GA", "AT>GC", "AT>TA", "CC>AA", "CC>AG", "CC>AT", "CC>GA", "CC>GG",
    "CC>GT", "CC>TA", "CC>TG", "CC>TT", "CG>AT", "CG>GC", "CG>GT", "CG>TA", "CG>TC", "CG>TT",
    "CT>AA", "CT>AC", "CT>AG", "CT>GA", "CT>GC", "CT>GG", "CT>TA", "CT>TC", "CT>TG", "GC>AA",
    "GC>AG", "GC>AT", "GC>CA", "GC>CG", "GC>TA", "TA>AT", "TA>CG", "TA>CT", "TA>GC", "TA>GG",
    "TA>GT", "TC>AA", "TC>AG", "TC>AT", "TC>CA", "TC>CG", "TC>CT", "TC>GA", "TC>GG", "TC>GT",
    "TG>AA", "TG>AC", "TG>AT", "TG>CA", "TG>CC", "TG>CT", "TG>GA", "TG>GC", "TG>GT", "TT>AA",
    "TT>AC", "TT>AG", "TT>CA", "TT>CC", "TT>CG", "TT>GA", "TT>GC", "TT>GG",
];

// Only exercised by the exhaustive test below; not otherwise called from
// production code, hence the narrow dead_code allow for non-test builds.
#[allow(dead_code)]
const BASES_ACGT: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// ACGT base -> 2-bit index (A=0 C=1 G=2 T=3), for indexing `DBS78_LUT`.
/// Labels contain only ACGT, so the fallthrough is unreachable.
const fn dbs_bidx(b: u8) -> usize {
    match b {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 0,
    }
}

/// Compile-time `(r0,a0,r1,a1) -> code` table, indexed by
/// `(r0<<6)|(a0<<4)|(r1<<2)|a1` over 2-bit base codes. Built from `DBS78_LABELS`
/// (source of truth): each label fills its literal key, then every remaining
/// key folds to its reverse-complement's literal — reproducing the old
/// literal-then-revcomp search with a single array load and zero allocation.
const DBS78_LUT: [u8; 256] = {
    let mut lut = [UNCLASSIFIED; 256];
    let mut i = 0;
    while i < DBS78_LABELS.len() {
        // Label "R0R1>A0A1": bytes [0,1] = ref doublet, [3,4] = alt doublet.
        let s = DBS78_LABELS[i].as_bytes();
        let key =
            (dbs_bidx(s[0]) << 6) | (dbs_bidx(s[3]) << 4) | (dbs_bidx(s[1]) << 2) | dbs_bidx(s[4]);
        lut[key] = i as u8;
        i += 1;
    }
    // Snapshot the literal-only table so the revcomp fold never reads a value
    // written by this second pass (RC is involutive, so a self-referential read
    // would otherwise be order-dependent).
    let literal = lut;
    let mut idx = 0;
    while idx < 256 {
        if lut[idx] == UNCLASSIFIED {
            let r0 = (idx >> 6) & 3;
            let a0 = (idx >> 4) & 3;
            let r1 = (idx >> 2) & 3;
            let a1 = idx & 3;
            // revcomp: complement (3 - b) each base and swap the two positions.
            let rc = ((3 - r1) << 6) | ((3 - a1) << 4) | ((3 - r0) << 2) | (3 - a0);
            lut[idx] = literal[rc];
        }
        idx += 1;
    }
    lut
};

/// DBS78 class-local index (0..=77) or `UNCLASSIFIED`. Literal-then-revcomp
/// canonicalization folded into `DBS78_LUT` at compile time.
#[inline]
pub fn dbs78_code(r0: u8, a0: u8, r1: u8, a1: u8) -> u8 {
    let (i0, i1, i2, i3) = (
        base_index(r0),
        base_index(a0),
        base_index(r1),
        base_index(a1),
    );
    if i0 < 0 || i1 < 0 || i2 < 0 || i3 < 0 {
        return UNCLASSIFIED;
    }
    DBS78_LUT[((i0 as usize) << 6) | ((i1 as usize) << 4) | ((i2 as usize) << 2) | (i3 as usize)]
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

    #[test]
    fn id83_1bp_del_in_repeat() {
        // seq: ...A C C C C... delete one C at a run of 4 C's.
        // REF="AC" ALT="A" (anchor A, deleted unit "C"), pos0 at the 'A'.
        // downstream run of C's from pos0+1: 4 -> is_del, rep=n_rep-1=3.
        // base C -> base_ct=0, kind Del=0 -> id1[0][0][3] = (0*2+0)*6+3 = 3
        let seq = b"AACCCCG"; // pos0 = 1 ('A'), then CCCC
        assert_eq!(id83_code(seq, 1, b"AC", b"A"), 3);
    }

    #[test]
    #[allow(clippy::identity_op)] // spelled-out kind*2+base_ct to mirror the LUT formula
    fn id83_1bp_ins() {
        // REF="A" ALT="AC" insert one C; downstream C run from pos0+1.
        // seq A C C C: n_rep counts inserted-unit "C" copies present downstream.
        // kind Ins=1, base C base_ct=0, rep=n_rep (no -1 for ins).
        let seq = b"ACCCG"; // pos0=0 ('A'); downstream from index1: C C C -> n_rep=3
        assert_eq!(id83_code(seq, 0, b"A", b"AC"), (1 * 2 + 0) * 6 + 3);
    }

    #[test]
    fn id83_rejects_non_acgt_unit() {
        let seq = b"ANNNG";
        assert_eq!(id83_code(seq, 0, b"AN", b"A"), UNCLASSIFIED);
    }

    #[test]
    fn dbs78_literal_hit() {
        // "AC>CA" is index 0.
        assert_eq!(dbs78_code(b'A', b'C', b'C', b'A'), 0);
    }

    #[test]
    fn dbs78_revcomp_fold() {
        // "GT>TG" is not literal; revcomp = "AC>CA" (index 0).
        assert_eq!(dbs78_code(b'G', b'T', b'T', b'G'), 0);
    }

    #[test]
    fn dbs78_all_16_doublets_map_or_uncl() {
        // Every ACGT^4 combo with r!=a on both is either a code < 78 or UNCLASSIFIED.
        for &r0 in &BASES_ACGT {
            for &r1 in &BASES_ACGT {
                for &a0 in &BASES_ACGT {
                    for &a1 in &BASES_ACGT {
                        let c = dbs78_code(r0, a0, r1, a1);
                        assert!(c < 78 || c == UNCLASSIFIED);
                    }
                }
            }
        }
    }
}

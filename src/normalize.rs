//! Variant normalization: biallelic split + atomization (no left-alignment; roadmap M2).
//! Pure functions over (pos, REF, ALTs) → atomic biallelic primitives. This is the only
//! module that knows atomization rules; the encode seam downstream stays sealed.

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum NormalizeError {
    #[error("symbolic/breakend ALT '{alt}' at pos {pos} is out of scope (short-read only)")]
    SymbolicAllele { pos: u32, alt: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom {
    /// 0-based start position of the atom.
    pub pos: u32,
    /// len(alt) - len(ref): 0 = SNP, > 0 = INS, < 0 = DEL.
    pub ilen: i32,
    /// SNP: the single ALT base. INS: anchor + inserted bases. DEL: the anchor base.
    pub alt: Vec<u8>,
    /// 1-based index into the record's original ALTs, for genotype remapping.
    pub source_alt_index: u16,
}

/// Decompose one VCF record into atomic biallelic primitives, appended to `out`.
/// `alts` are the ALT alleles only (REF excluded). `*` / `.` alleles are skipped;
/// symbolic/breakend alleles return an error.
pub fn atomize_record(
    pos: u32,
    ref_allele: &[u8],
    alts: &[&[u8]],
    out: &mut Vec<Atom>,
) -> Result<(), NormalizeError> {
    for (j, &alt) in alts.iter().enumerate() {
        let src = (j + 1) as u16;
        if alt == b"*" || alt == b"." {
            continue;
        }
        if is_symbolic(alt) {
            return Err(NormalizeError::SymbolicAllele {
                pos,
                alt: String::from_utf8_lossy(alt).into_owned(),
            });
        }
        atomize_biallelic(pos, ref_allele, alt, src, out);
    }
    Ok(())
}

#[inline]
fn is_symbolic(alt: &[u8]) -> bool {
    alt.first() == Some(&b'<') || alt.contains(&b'[') || alt.contains(&b']')
}

/// Decompose a single REF/ALT pair. Mirrors bcftools `_atomize_allele`: trim the
/// shared suffix, emit a SNV per interior mismatch, and attach any length change as a
/// single left-anchored indel at the last aligned index. The one deviation is the
/// substituted-deletion-anchor case (see below), forced by the pure-DEL encoding.
fn atomize_biallelic(pos: u32, ref_allele: &[u8], alt: &[u8], src: u16, out: &mut Vec<Atom>) {
    // 1. Trim shared suffix, keeping >= 1 base on each side.
    let mut rlen = ref_allele.len();
    let mut alen = alt.len();
    while rlen > 1 && alen > 1 && ref_allele[rlen - 1] == alt[alen - 1] {
        rlen -= 1;
        alen -= 1;
    }
    let r = &ref_allele[..rlen];
    let a = &alt[..alen];

    let n = rlen.min(alen);
    let k = n - 1; // last aligned index; the indel (if any) anchors here

    // 2. Interior aligned positions [0, k): one SNV per mismatch.
    for i in 0..k {
        if r[i] != a[i] {
            out.push(Atom {
                pos: pos + i as u32,
                ilen: 0,
                alt: vec![a[i]],
                source_alt_index: src,
            });
        }
    }

    // 3. Boundary at k.
    let apos = pos + k as u32;
    if rlen == alen {
        // Pure substitution tail.
        if r[k] != a[k] {
            out.push(Atom {
                pos: apos,
                ilen: 0,
                alt: vec![a[k]],
                source_alt_index: src,
            });
        }
    } else if alen > rlen {
        // Insertion anchored at k. ref[k..] is a single base; alt[k..] carries the
        // (possibly substituted) anchor + inserted bases — the full alt is stored, so a
        // substituted anchor is faithfully represented.
        let ins_alt = a[k..alen].to_vec();
        let ilen = ins_alt.len() as i32 - 1; // ref_len == 1
        out.push(Atom {
            pos: apos,
            ilen,
            alt: ins_alt,
            source_alt_index: src,
        });
    } else {
        // Deletion anchored at k. alt[k..] is a single base.
        let del_ref_len = (rlen - k) as i32; // >= 2
        let ilen = 1 - del_ref_len; // alt_len(1) - ref_len
        if r[k] == a[k] {
            // Clean anchor → pure DEL (alt = the anchor base).
            out.push(Atom {
                pos: apos,
                ilen,
                alt: vec![a[k]],
                source_alt_index: src,
            });
        } else {
            // Substituted anchor: the pure-DEL encoding reconstructs the anchor from
            // the reference, so it cannot carry a substitution. Split into a SNV plus a
            // clean DEL whose anchor is the unchanged reference base.
            out.push(Atom {
                pos: apos,
                ilen: 0,
                alt: vec![a[k]],
                source_alt_index: src,
            });
            out.push(Atom {
                pos: apos,
                ilen,
                alt: vec![r[k]],
                source_alt_index: src,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn atoms(pos: u32, r: &[u8], alts: &[&[u8]]) -> Vec<Atom> {
        let mut out = Vec::new();
        atomize_record(pos, r, alts, &mut out).unwrap();
        out
    }

    fn atom(pos: u32, ilen: i32, alt: &[u8], src: u16) -> Atom {
        Atom {
            pos,
            ilen,
            alt: alt.to_vec(),
            source_alt_index: src,
        }
    }

    #[test]
    fn snp_passthrough() {
        assert_eq!(atoms(100, b"A", &[b"C"]), vec![atom(100, 0, b"C", 1)]);
    }

    #[test]
    fn biallelic_split_tags_source_index() {
        // A>C,G → two independent SNPs carrying source indices 1 and 2.
        assert_eq!(
            atoms(100, b"A", &[b"C", b"G"]),
            vec![atom(100, 0, b"C", 1), atom(100, 0, b"G", 2)]
        );
    }

    #[test]
    fn mnp_becomes_per_position_snps() {
        // AC>GT → A>G@100, C>T@101 (both differ).
        assert_eq!(
            atoms(100, b"AC", &[b"GT"]),
            vec![atom(100, 0, b"G", 1), atom(101, 0, b"T", 1)]
        );
    }

    #[test]
    fn mnp_skips_matching_bases() {
        // ACA>ATA → only the middle base changes.
        assert_eq!(atoms(100, b"ACA", &[b"ATA"]), vec![atom(101, 0, b"T", 1)]);
    }

    #[test]
    fn simple_insertion_anchored() {
        // A>AT → anchored INS, ilen=+1, full alt stored.
        assert_eq!(atoms(100, b"A", &[b"AT"]), vec![atom(100, 1, b"AT", 1)]);
    }

    #[test]
    fn simple_deletion_anchored() {
        // ATG>A → pure DEL, ilen=-2, alt = anchor base.
        assert_eq!(atoms(100, b"ATG", &[b"A"]), vec![atom(100, -2, b"A", 1)]);
    }

    #[test]
    fn shared_suffix_is_trimmed() {
        // ATG>CG shares suffix G → AT>C: anchor substitutes (A→C), so split into
        // SNV(A>C)@100 plus a clean DEL(anchor=ref base A, delete T)@100.
        assert_eq!(
            atoms(100, b"ATG", &[b"CG"]),
            vec![atom(100, 0, b"C", 1), atom(100, -1, b"A", 1)]
        );
    }

    #[test]
    fn complex_snv_plus_insertion() {
        // GCG>GTGA (the bcftools abuf.c example): interior SNV C>T@101, then an
        // anchored INS G>GA@102.
        assert_eq!(
            atoms(100, b"GCG", &[b"GTGA"]),
            vec![atom(101, 0, b"T", 1), atom(102, 1, b"GA", 1)]
        );
    }

    #[test]
    fn star_and_missing_alleles_skipped() {
        assert_eq!(atoms(100, b"A", &[b"*"]), vec![]);
        assert_eq!(atoms(100, b"A", &[b"."]), vec![]);
        // still processes the real ALT alongside a skipped one
        assert_eq!(atoms(100, b"A", &[b"*", b"C"]), vec![atom(100, 0, b"C", 2)]);
    }

    #[test]
    fn symbolic_allele_errors() {
        let mut out = Vec::new();
        assert!(matches!(
            atomize_record(100, b"A", &[b"<DEL>"], &mut out),
            Err(NormalizeError::SymbolicAllele { .. })
        ));
        assert!(matches!(
            atomize_record(100, b"A", &[b"A[chr2:321[".as_slice()], &mut out),
            Err(NormalizeError::SymbolicAllele { .. })
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        // Every atom is encoder-valid and starts within the record's reference span.
        #[test]
        fn atoms_are_encoder_valid(
            pos in 0u32..1_000_000,
            r in proptest::collection::vec(prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T')], 1..12),
            a in proptest::collection::vec(prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T')], 1..12),
        ) {
            let mut out = Vec::new();
            let ref_bytes: Vec<u8> = r;
            let alt_bytes: Vec<u8> = a;
            atomize_record(pos, &ref_bytes, &[alt_bytes.as_slice()], &mut out).unwrap();
            for at in &out {
                let alt_len = at.alt.len() as i32;
                let valid = (at.ilen == 0 && alt_len == 1)
                    || (at.ilen > 0 && alt_len == at.ilen + 1)  // INS: ref_len 1
                    || (at.ilen < 0 && alt_len == 1);           // DEL: alt_len 1
                prop_assert!(valid, "atom not encoder-valid: {:?}", at);
                prop_assert!(at.pos >= pos, "atom before record start");
                prop_assert!(at.pos < pos + ref_bytes.len() as u32, "atom past reference span");
            }
        }
    }
}

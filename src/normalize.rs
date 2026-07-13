//! Variant normalization: biallelic split + atomization (no left-alignment; roadmap M2).
//! Pure functions over (pos, REF, ALTs) → atomic biallelic primitives. This is the only
//! module that knows atomization rules; the encode seam downstream stays sealed.

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum NormalizeError {
    #[error("symbolic/breakend ALT '{alt}' at pos {pos} is out of scope (short-read only)")]
    SymbolicAllele { pos: u32, alt: String },
    #[error("REF '{expected}' at pos {pos} disagrees with reference FASTA ('{found}')")]
    RefMismatch {
        pos: u32,
        expected: String,
        found: String,
    },
    #[error(
        "REF at pos {pos} needs reference bases up to {needed_end} but contig is only \
         {contig_len} long"
    )]
    RefOutOfContig {
        pos: u32,
        needed_end: usize,
        contig_len: usize,
    },
}

/// Maximum leftward shift (in reference bases) applied during left-alignment, and the
/// matching width by which the reorder buffer's emit bound is relaxed (see
/// `crate::vcf_reader::VcfChunkReader::next_atom`). Left-alignment never moves an atom
/// more than `L_MAX` bases; an indel inside a repeat longer than `L_MAX` is left
/// *partially* aligned — the same truncation `bcftools` applies at its `--buffer-size`
/// limit. Conservative default pending measurement on representative short-read VCFs (the
/// M2b spec's open question); real short-read indel/STR shifts are far below this.
pub const L_MAX: u32 = 1000;

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
/// `alts` are the ALT alleles only (REF excluded). `*` / `.` alleles are always
/// skipped. Symbolic/breakend alleles are skipped and counted when
/// `skip_out_of_scope` is set; otherwise they return an error. Returns the number
/// of out-of-scope ALTs dropped (always 0 unless `skip_out_of_scope`).
pub fn atomize_record(
    pos: u32,
    ref_allele: &[u8],
    alts: &[&[u8]],
    out: &mut Vec<Atom>,
    skip_out_of_scope: bool,
) -> Result<u32, NormalizeError> {
    let mut dropped = 0u32;
    for (j, &alt) in alts.iter().enumerate() {
        let src = (j + 1) as u16;
        if alt == b"*" || alt == b"." {
            continue;
        }
        if is_symbolic(alt) {
            if skip_out_of_scope {
                dropped += 1;
                continue;
            }
            return Err(NormalizeError::SymbolicAllele {
                pos,
                alt: String::from_utf8_lossy(alt).into_owned(),
            });
        }
        atomize_biallelic(pos, ref_allele, alt, src, out);
    }
    Ok(dropped)
}

/// Fail-fast check that a record's REF allele matches the reference FASTA. Left-alignment
/// rolls an indel against the reference, so a REF that disagrees with the FASTA would
/// silently corrupt positions; reject it instead (a mismatch is an error, never silently
/// "corrected"). Comparison is ASCII-case-insensitive so soft-masked (lowercase)
/// reference bases match uppercase VCF alleles. `ref_seq` is the full 0-based contig.
pub fn validate_ref(pos: u32, ref_allele: &[u8], ref_seq: &[u8]) -> Result<(), NormalizeError> {
    let start = pos as usize;
    let needed_end = start + ref_allele.len();
    if needed_end > ref_seq.len() {
        return Err(NormalizeError::RefOutOfContig {
            pos,
            needed_end,
            contig_len: ref_seq.len(),
        });
    }
    if !ref_seq[start..needed_end].eq_ignore_ascii_case(ref_allele) {
        return Err(NormalizeError::RefMismatch {
            pos,
            expected: String::from_utf8_lossy(ref_allele).into_owned(),
            found: String::from_utf8_lossy(&ref_seq[start..needed_end]).into_owned(),
        });
    }
    Ok(())
}

/// Shift an anchored indel atom to its leftmost reference-equivalent position (classic VCF
/// left-alignment / repeat roll), capped at `l_max` leftward bases. SNP atoms and
/// substituted-anchor insertions never move. `ref_seq` is the full 0-based contig
/// (uppercased); the caller must have run `validate_ref` on the source record first, so
/// every reference read here is in-bounds. Only `pos` and the anchor base(s) in `alt`
/// change — `ilen` (and thus deletion length) and `source_alt_index` are invariant.
pub fn left_align(mut atom: Atom, ref_seq: &[u8], l_max: u32) -> Atom {
    use std::cmp::Ordering;
    match atom.ilen.cmp(&0) {
        Ordering::Equal => atom, // SNP: never moves.
        Ordering::Less => {
            // Deletion: anchor at `pos`, `ndel` deleted bases at ref_seq[pos+1 ..= pos+ndel].
            // Roll while the anchor base equals the last deleted base.
            let ndel = (-atom.ilen) as usize;
            let mut pos = atom.pos as usize;
            let mut shifts = 0u32;
            while pos > 0 && shifts < l_max && ref_seq[pos] == ref_seq[pos + ndel] {
                pos -= 1;
                shifts += 1;
            }
            atom.pos = pos as u32;
            atom.alt = vec![ref_seq[pos]]; // pure DEL: alt is the (new) anchor base.
            atom
        }
        Ordering::Greater => {
            // Insertion: alt = [anchor] ++ inserted. Roll only a clean anchor; a
            // substituted anchor is SNP+INS and stays put.
            let mut pos = atom.pos as usize;
            if ref_seq[pos] != atom.alt[0] {
                return atom;
            }
            let mut inserted: Vec<u8> = atom.alt[1..].to_vec();
            let mut shifts = 0u32;
            while pos > 0 && shifts < l_max && ref_seq[pos] == *inserted.last().unwrap() {
                inserted.rotate_right(1);
                pos -= 1;
                shifts += 1;
            }
            let mut alt = Vec::with_capacity(1 + inserted.len());
            alt.push(ref_seq[pos]);
            alt.extend_from_slice(&inserted);
            atom.pos = pos as u32;
            atom.alt = alt;
            atom
        }
    }
}

/// One biallelic atom in ordinary VCF REF/ALT form (not the internal anchor-only
/// [`Atom`] encoding), ready to place in a `RawRecord`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BiallelicRecord {
    /// 0-based start position of the record.
    pub pos: u32,
    /// REF bytes, reconstructed from the reference (or the source record's REF).
    pub reference: Vec<u8>,
    /// ALT bytes.
    pub alt: Vec<u8>,
    /// 1-based index into the ORIGINAL record's ALTs, for genotype remapping.
    pub source_alt_index: u16,
}

/// Atomize `(pos, ref_allele, alts)` into biallelic VCF-form records.
///
/// When `ref_seq` is `Some`, each atom is left-aligned and its REF bytes are read from
/// `ref_seq[pos .. pos + ref_len]`. When `None` (no-reference mode), no left-alignment
/// happens and REF bytes come from the record's own REF slice at the atom's offset.
/// `ref_len = alt.len() as i32 - ilen` (so a DEL, with `ilen < 0`, grows the REF).
///
/// Returns `(records, dropped_out_of_scope)`, matching [`atomize_record`]'s drop count.
pub fn atomize_to_vcf_biallelic(
    pos: u32,
    ref_allele: &[u8],
    alts: &[&[u8]],
    ref_seq: Option<&[u8]>,
    skip_out_of_scope: bool,
) -> Result<(Vec<BiallelicRecord>, u32), NormalizeError> {
    let mut atoms = Vec::new();
    let dropped = atomize_record(pos, ref_allele, alts, &mut atoms, skip_out_of_scope)?;
    let mut out = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let atom = match ref_seq {
            Some(rs) => left_align(atom, rs, L_MAX),
            None => atom,
        };
        let ref_len = (atom.alt.len() as i32 - atom.ilen) as usize;
        let reference = match ref_seq {
            Some(rs) => rs[atom.pos as usize..atom.pos as usize + ref_len].to_vec(),
            None => {
                // No left-align, so atom.pos >= pos and the atom's REF span lies within
                // the record's own REF.
                let off = (atom.pos - pos) as usize;
                ref_allele[off..off + ref_len].to_vec()
            }
        };
        out.push(BiallelicRecord {
            pos: atom.pos,
            reference,
            alt: atom.alt,
            source_alt_index: atom.source_alt_index,
        });
    }
    Ok((out, dropped))
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
    // Valid VCF guarantees non-empty REF/ALT; a `*`/`.`/symbolic ALT is filtered by the
    // caller. Fail fast on the precondition — otherwise `k = n - 1` below underflows.
    debug_assert!(
        !ref_allele.is_empty() && !alt.is_empty(),
        "atomize_biallelic requires non-empty REF and ALT"
    );
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
        atomize_record(pos, r, alts, &mut out, false).unwrap();
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
            atomize_record(100, b"A", &[b"<DEL>"], &mut out, false),
            Err(NormalizeError::SymbolicAllele { .. })
        ));
        assert!(matches!(
            atomize_record(100, b"A", &[b"A[chr2:321[".as_slice()], &mut out, false),
            Err(NormalizeError::SymbolicAllele { .. })
        ));
    }

    #[test]
    fn symbolic_allele_skipped_and_counted_when_opted_in() {
        // Multiallelic: one real SNP + one symbolic ALT. With skip on, the SNP
        // survives and the symbolic ALT is dropped and counted.
        let mut out = Vec::new();
        let dropped = atomize_record(100, b"A", &[b"C", b"<DEL>"], &mut out, true).unwrap();
        assert_eq!(dropped, 1);
        assert_eq!(out, vec![atom(100, 0, b"C", 1)]);
    }

    #[test]
    fn breakend_allele_skipped_when_opted_in() {
        let mut out = Vec::new();
        let dropped =
            atomize_record(100, b"A", &[b"A[chr2:321[".as_slice()], &mut out, true).unwrap();
        assert_eq!(dropped, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn out_of_scope_errors_when_skip_disabled() {
        let mut out = Vec::new();
        assert!(matches!(
            atomize_record(100, b"A", &[b"<DEL>"], &mut out, false),
            Err(NormalizeError::SymbolicAllele { .. })
        ));
    }

    #[test]
    fn validate_ref_accepts_matching_ref() {
        // ref_seq (0-based): A C G T A C
        let ref_seq = b"ACGTAC";
        assert_eq!(validate_ref(2, b"GT", ref_seq), Ok(()));
        assert_eq!(validate_ref(0, b"A", ref_seq), Ok(()));
    }

    #[test]
    fn validate_ref_is_case_insensitive() {
        // Soft-masked (lowercase) reference bases must still match uppercase VCF REF.
        let ref_seq = b"acgtac";
        assert_eq!(validate_ref(2, b"GT", ref_seq), Ok(()));
    }

    #[test]
    fn validate_ref_rejects_mismatch() {
        let ref_seq = b"ACGTAC";
        assert!(matches!(
            validate_ref(2, b"AA", ref_seq),
            Err(NormalizeError::RefMismatch { pos: 2, .. })
        ));
    }

    #[test]
    fn validate_ref_rejects_out_of_contig() {
        let ref_seq = b"ACGT";
        assert!(matches!(
            validate_ref(3, b"TAC", ref_seq),
            Err(NormalizeError::RefOutOfContig {
                pos: 3,
                needed_end: 6,
                contig_len: 4
            })
        ));
    }

    // Reference used across left_align tests (0-based indices):
    //  0:C 1:A 2:A 3:A 4:A 5:T 6:C 7:A 8:G 9:A 10:G 11:T
    const LA_REF: &[u8] = b"CAAAATCAGAGT";

    #[test]
    fn left_align_deletion_rolls_through_homopolymer() {
        // AA>A anchored at pos 3 (delete one A) → bcftools: pos 0, REF "CA" ALT "C".
        let a = atom(3, -1, b"A", 1);
        assert_eq!(left_align(a, LA_REF, L_MAX), atom(0, -1, b"C", 1));
    }

    #[test]
    fn left_align_deletion_stops_at_contig_start() {
        // The homopolymer roll above lands exactly on pos 0 (contig start) and stops
        // there without reading ref_seq[-1].
        let a = atom(3, -1, b"A", 1);
        assert_eq!(left_align(a, LA_REF, L_MAX).pos, 0);
    }

    #[test]
    fn left_align_insertion_rolls_through_homopolymer() {
        // A>AA anchored at pos 3 (insert one A) → bcftools: pos 0, REF "C" ALT "CA".
        let a = atom(3, 1, b"AA", 1);
        assert_eq!(left_align(a, LA_REF, L_MAX), atom(0, 1, b"CA", 1));
    }

    #[test]
    fn left_align_dinucleotide_deletion_rolls_repeat() {
        // GAG>G anchored at pos 8 (delete "AG" from the AGAG repeat) → bcftools:
        // pos 6, REF "CAG" ALT "C".
        let a = atom(8, -2, b"G", 1);
        assert_eq!(left_align(a, LA_REF, L_MAX), atom(6, -2, b"C", 1));
    }

    #[test]
    fn left_align_snp_never_moves() {
        let a = atom(3, 0, b"T", 1);
        assert_eq!(left_align(a, LA_REF, L_MAX), atom(3, 0, b"T", 1));
    }

    #[test]
    fn left_align_substituted_anchor_insertion_stays_put() {
        // alt[0]='G' but ref_seq[3]='A' → substituted anchor (SNP+INS), not a pure
        // insertion. Must not roll.
        let a = atom(3, 1, b"GT", 1);
        assert_eq!(left_align(a, LA_REF, L_MAX), atom(3, 1, b"GT", 1));
    }

    #[test]
    fn left_align_respects_l_max_partial_align() {
        // Same homopolymer deletion, but l_max = 2 caps the roll: pos 3 → 1 (two shifts),
        // not all the way to 0.
        let a = atom(3, -1, b"A", 1);
        let out = left_align(a, LA_REF, 2);
        assert_eq!(out.pos, 1);
        assert_eq!(out.alt, b"A".to_vec()); // ref_seq[1] == 'A'
    }

    // Apply a (clean) indel atom to a copy of the reference, returning the alt haplotype.
    // DEL removes ref_seq[pos+1 ..= pos+ndel]; INS splices alt[1..] in after `pos`.
    fn apply_edit(atom: &Atom, ref_seq: &[u8]) -> Vec<u8> {
        let pos = atom.pos as usize;
        let mut out = ref_seq[..=pos].to_vec();
        if atom.ilen > 0 {
            out.extend_from_slice(&atom.alt[1..]);
            out.extend_from_slice(&ref_seq[pos + 1..]);
        } else {
            let ndel = (-atom.ilen) as usize;
            out.extend_from_slice(&ref_seq[pos + 1 + ndel..]);
        }
        out
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        // Left-alignment preserves the alt haplotype (reference-equivalent) and produces a
        // representation that cannot shift further left within L_MAX.
        #[test]
        fn left_align_is_reference_equivalent_and_leftmost(
            reference in proptest::collection::vec(
                prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T')], 2..40),
            anchor in 0usize..38,
            is_del in any::<bool>(),
            unit in proptest::collection::vec(
                prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T')], 1..4),
        ) {
            let n = reference.len();
            let pos = anchor % (n - 1); // leave room for >= 1 base to the right

            let atom = if is_del {
                // Clean DEL: delete up to `unit.len()` bases starting at pos+1.
                let ndel = unit.len().min(n - pos - 1);
                if ndel == 0 { return Ok(()); }
                Atom { pos: pos as u32, ilen: -(ndel as i32),
                       alt: vec![reference[pos]], source_alt_index: 1 }
            } else {
                // Clean INS: anchor is the true reference base, insert `unit`.
                let mut alt = vec![reference[pos]];
                alt.extend_from_slice(&unit);
                Atom { pos: pos as u32, ilen: unit.len() as i32,
                       alt, source_alt_index: 1 }
            };

            let aligned = left_align(atom.clone(), &reference, L_MAX);

            // 1. Reference-equivalent: same alt haplotype.
            prop_assert_eq!(apply_edit(&atom, &reference), apply_edit(&aligned, &reference));

            // 2. Length invariant.
            prop_assert_eq!(aligned.ilen, atom.ilen);

            // 3. Leftmost within L_MAX: applying left_align again is a fixpoint.
            let again = left_align(aligned.clone(), &reference, L_MAX);
            prop_assert_eq!(&again.pos, &aligned.pos);
            prop_assert_eq!(&again.alt, &aligned.alt);
        }
    }

    #[test]
    fn biallelic_snp_from_ref() {
        // ref_seq "ACAGT..."; POS 2 (0-based) is 'A'->'G' SNP
        let ref_seq = b"ACAGTACATG";
        let (recs, dropped) =
            atomize_to_vcf_biallelic(2, b"A", &[b"G".as_ref()], Some(ref_seq), false).unwrap();
        assert_eq!(dropped, 0);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].pos, 2);
        assert_eq!(recs[0].reference, b"A");
        assert_eq!(recs[0].alt, b"G");
        assert_eq!(recs[0].source_alt_index, 1);
    }

    #[test]
    fn biallelic_deletion_ref_bytes_reconstructed() {
        // Anchored DEL: REF "CAT" ALT "C" at POS 6 (0-based). ref_len = 1 - (-2) = 3.
        let ref_seq = b"ACAGTACATGGG";
        let (recs, _) =
            atomize_to_vcf_biallelic(6, b"CAT", &[b"C".as_ref()], Some(ref_seq), false).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].reference, b"CAT"); // anchor + deleted bases
        assert_eq!(recs[0].alt, b"C");
    }

    #[test]
    fn biallelic_multiallelic_splits_with_alt_indices() {
        let ref_seq = b"ACAGTACATG";
        let (recs, _) = atomize_to_vcf_biallelic(
            2,
            b"A",
            &[b"G".as_ref(), b"T".as_ref()],
            Some(ref_seq),
            false,
        )
        .unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].source_alt_index, 1);
        assert_eq!(recs[1].source_alt_index, 2);
    }

    #[test]
    fn biallelic_no_reference_uses_record_ref() {
        // No ref_seq: REF bytes come from the record's own REF, no left-align.
        let (recs, _) = atomize_to_vcf_biallelic(6, b"CAT", &[b"C".as_ref()], None, false).unwrap();
        assert_eq!(recs[0].pos, 6);
        assert_eq!(recs[0].reference, b"CAT");
        assert_eq!(recs[0].alt, b"C");
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
            atomize_record(pos, &ref_bytes, &[alt_bytes.as_slice()], &mut out, false).unwrap();
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

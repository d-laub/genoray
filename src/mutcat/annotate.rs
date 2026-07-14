//! Classify a finished contig's records and write the mutcat sidecar. Runs at
//! write time (FASTA already open) or post-hoc.

use crate::layout::{ContigPaths, MutcatSub};
use crate::mutcat::NOT_ANNOTATED;
use crate::mutcat::classify::{id83_code, sbs96_code};
use crate::mutcat::sidecar::write_sidecar;
use crate::query::ContigReader;
use svar2_codec::{DecodedKey, decode_key, decode_snp_2bit, encode_snp_2bit};

/// Classify every record of `reader`'s four sub-streams against `ref_seq` (the
/// whole contig, 0-based ASCII) and write the sidecar under `paths`.
pub fn annotate_contig(
    reader: &ContigReader,
    paths: &ContigPaths,
    ref_seq: &[u8],
) -> std::io::Result<()> {
    // Classifiers match uppercase ACGT only; normalize here so BOTH entry points
    // (write-time from_vcf(signatures=True) and post-hoc annotate_mutations, which
    // reads a possibly soft-masked reference) classify identically. base_index()
    // treats lowercase as non-ACGT -> UNCLASSIFIED, so a soft-masked reference would
    // otherwise silently drop masked-region variants from SBS96/ID83.
    let ref_seq: Vec<u8> = ref_seq.to_ascii_uppercase();
    let ref_seq: &[u8] = &ref_seq;
    // --- var_key/snp (per call) ---
    {
        let (positions, keys) = reader.vk_snp_records(); // &[u32], packed 2-bit keys
        let mut codes = Vec::with_capacity(positions.len());
        let mut refs = Vec::with_capacity(positions.len());
        for (i, &pos) in positions.iter().enumerate() {
            let (code, refc) = snp_record(ref_seq, pos, snp_alt(keys, i));
            codes.push(code);
            refs.push(refc);
        }
        write_sidecar(paths, MutcatSub::VkSnp, &codes, Some(&refs), None)?;
    }
    // --- dense/snp (per variant) ---
    if let Some((positions, keys)) = reader.dense_snp_records() {
        // Dense SNP keys are 2-bit-packed (4 variants/byte), same as var_key/snp:
        // decode via `snp_alt` (unpack_snp_key_at + decode_snp_2bit), never `keys[i]`.
        let mut codes = Vec::with_capacity(positions.len());
        let mut refs = Vec::with_capacity(positions.len());
        for (i, &pos) in positions.iter().enumerate() {
            let (code, refc) = snp_record(ref_seq, pos, snp_alt(keys, i));
            codes.push(code);
            refs.push(refc);
        }
        write_sidecar(paths, MutcatSub::DenseSnp, &codes, Some(&refs), None)?;
    }
    // --- var_key/indel + dense/indel (per record) ---
    annotate_indel(reader, paths, ref_seq, MutcatSub::VkIndel)?;
    annotate_indel(reader, paths, ref_seq, MutcatSub::DenseIndel)?;
    Ok(())
}

/// SBS96 code + 2-bit ref base for a SNP at 0-based `pos` with ASCII `alt`.
fn snp_record(ref_seq: &[u8], pos: u32, alt: u8) -> (u8, u8) {
    let p = pos as usize;
    let n = ref_seq.len();
    if p == 0 || p + 1 >= n {
        return (NOT_ANNOTATED, 0); // contig-end flank missing → not classified
    }
    let refb = ref_seq[p];
    let code = sbs96_code(ref_seq[p - 1], refb, alt, ref_seq[p + 1]);
    (code, encode_snp_2bit(refb))
}

/// ALT base for record `i` of a 2-bit-packed SNP key buffer (var_key or dense).
fn snp_alt(packed_keys: &[u8], i: usize) -> u8 {
    decode_snp_2bit(svar2_codec::unpack_snp_key_at(packed_keys, i))
}

fn annotate_indel(
    reader: &ContigReader,
    paths: &ContigPaths,
    ref_seq: &[u8],
    sub: MutcatSub,
) -> std::io::Result<()> {
    let recs = reader.indel_records(sub); // Option<(&[u32] positions, &[u32] keys)>
    let (positions, keys) = match recs {
        Some(r) => r,
        None => return write_sidecar(paths, sub, &[], None, None),
    };
    let mut codes = Vec::with_capacity(positions.len());
    for (i, &pos) in positions.iter().enumerate() {
        codes.push(indel_code(reader, ref_seq, pos, keys[i]));
    }
    write_sidecar(paths, sub, &codes, None, None)
}

/// ID83 code for one indel record, reconstructing REF/ALT from key + reference.
fn indel_code(reader: &ContigReader, ref_seq: &[u8], pos: u32, key: u32) -> u8 {
    let p = pos as usize;
    if p >= ref_seq.len() {
        return NOT_ANNOTATED;
    }
    let anchor = ref_seq[p];
    let (refa, alta): (Vec<u8>, Vec<u8>) = match decode_key(key) {
        DecodedKey::Inline { alt } => {
            // atomized INS/anchor: REF = single anchor base; ALT = alt bytes.
            (vec![anchor], alt)
        }
        DecodedKey::PureDel { ilen } => {
            let dl = (-ilen) as usize;
            if p + dl >= ref_seq.len() {
                return NOT_ANNOTATED;
            }
            // REF = anchor + dl deleted bases; ALT = anchor only.
            (ref_seq[p..p + dl + 1].to_vec(), vec![anchor])
        }
        DecodedKey::Lookup { row } => {
            let alt = reader.lut_allele(row); // Vec<u8>
            (vec![anchor], alt)
        }
    };
    id83_code(ref_seq, p, &refa, &alta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutcat::classify::sbs96_code;

    // Minimal check: a single var_key/snp record's sidecar code equals the
    // direct sbs96_code of its reference context.
    #[test]
    fn snp_record_matches_direct_classify() {
        let ref_seq = b"TACGT"; // pos 2 = 'C', 5'=A(1), 3'=G(3)
        let (code, refc) = snp_record(ref_seq, 2, b'A');
        assert_eq!(code, sbs96_code(b'A', b'C', b'A', b'G'));
        assert_eq!(refc, svar2_codec::encode_snp_2bit(b'C'));
    }

    #[test]
    fn contig_end_snp_is_not_annotated() {
        let ref_seq = b"CG";
        assert_eq!(snp_record(ref_seq, 0, b'A').0, NOT_ANNOTATED);
    }
}

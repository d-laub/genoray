// Shared BCF test harness: synthetic record builder, binary I/O helpers, and key decoders
// for E2E pipeline testing.

use genoray_core::rvk::decode_alt_inline;

use rust_htslib::bcf::record::GenotypeAllele;
use rust_htslib::bcf::{Format, Header, Writer};

use std::path::Path;

/// One synthetic VCF record: position, ref allele, list of alt alleles, and a
/// flat genotype vector laid out as [s0_p0, s0_p1, s1_p0, s1_p1, ...] holding
/// allele indices (0 = ref, 1 = first alt, …).
pub struct SynthRecord<'a> {
    pub pos: i64,
    pub ref_allele: &'a [u8],
    pub alts: Vec<&'a [u8]>,
    pub gt: Vec<i32>,
}

pub fn build_bcf_with_index(
    bcf_path: &Path,
    chrom: &str,
    chrom_len: u64,
    samples: &[&str],
    records: &[SynthRecord],
) {
    let mut header = Header::new();
    let contig = format!("##contig=<ID={},length={}>", chrom, chrom_len);
    header.push_record(contig.as_bytes());
    header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
    for s in samples {
        header.push_sample(s.as_bytes());
    }

    {
        let mut writer =
            Writer::from_path(bcf_path, &header, false, Format::Bcf).expect("open BCF writer");

        for rec in records {
            let mut record = writer.empty_record();
            record.set_rid(Some(0));
            record.set_pos(rec.pos);

            // set_alleles wants &[&[u8]] starting with ref then alts
            let mut alleles: Vec<&[u8]> = Vec::with_capacity(1 + rec.alts.len());
            alleles.push(rec.ref_allele);
            for a in &rec.alts {
                alleles.push(a);
            }
            record.set_alleles(&alleles).expect("set alleles");

            let gt_alleles: Vec<GenotypeAllele> =
                rec.gt.iter().map(|&i| GenotypeAllele::Phased(i)).collect();
            record.push_genotypes(&gt_alleles).expect("push genotypes");

            writer.write(&record).expect("write record");
        }
        // writer drops here, finalizing the BCF file
    }

    // Build CSI index alongside the BCF so IndexedReader can fetch by chrom.
    // BCF only supports CSI (TBI is text-VCF-only). min_shift=14 is the htslib
    // standard for genomic data; n_threads=0 means synchronous build.
    rust_htslib::bcf::index::build(bcf_path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
        .expect("build BCF index");
}

pub fn read_u32_bin(path: &Path) -> Vec<u32> {
    // Alignment-agnostic decode: std::fs::read can hand back a Vec<u8> whose
    // pointer isn't 4-byte aligned (especially when empty), which would trip
    // bytemuck::cast_slice's TargetAlignmentGreater check.
    let bytes = std::fs::read(path).expect("read u32 bin");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn read_offsets_npy(path: &Path) -> Vec<u64> {
    let arr: ndarray::Array1<u64> = ndarray_npy::read_npy(path).expect("read offsets npy");
    arr.to_vec()
}

/// Decode a single packed key. Discriminator:
///   bit 0 = 1                 → lookup (long allele bank row index)
///   bit 0 = 0 AND bit 31 = 1  → pure DEL (signed ilen in upper 31 bits)
///   bit 0 = 0 AND bit 31 = 0  → inline ALT (top 5 bits hold the length field)
#[derive(Debug, PartialEq)]
pub enum DecodedKey {
    Inline { alt: Vec<u8> },
    PureDel { ilen: i32 },
    Lookup { row: u32 },
}

pub fn decode_key(key: u32) -> DecodedKey {
    if key & 1 == 1 {
        DecodedKey::Lookup { row: key >> 1 }
    } else if (key as i32) < 0 {
        DecodedKey::PureDel {
            ilen: (key as i32) >> 1,
        }
    } else {
        DecodedKey::Inline {
            alt: decode_alt_inline(key),
        }
    }
}

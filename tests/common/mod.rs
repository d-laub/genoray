// Shared BCF test harness: synthetic record builder and binary I/O helpers
// for E2E pipeline testing.

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

/// Build a FASTA (+ `.fai`) whose single contig is `chrom_len` bases of 'N' filler with
/// each record's REF allele stamped at its 0-based `pos`, so it agrees with the records
/// pushed into the companion BCF. 'N' filler never satisfies the left-align repeat
/// condition, so records that should not move don't. Overlapping records must agree on
/// their shared bases (real VCFs are reference-consistent).
#[allow(dead_code)]
pub fn build_fasta_with_index(
    fasta_path: &Path,
    chrom: &str,
    chrom_len: usize,
    records: &[SynthRecord],
) {
    use std::io::Write;
    let mut seq = vec![b'N'; chrom_len];
    for rec in records {
        let start = rec.pos as usize;
        seq[start..start + rec.ref_allele.len()].copy_from_slice(rec.ref_allele);
    }
    {
        let mut f = std::fs::File::create(fasta_path).expect("create fasta");
        writeln!(f, ">{}", chrom).expect("write header");
        f.write_all(&seq).expect("write seq");
        writeln!(f).expect("write newline");
    }
    rust_htslib::faidx::build(fasta_path).expect("build .fai");
}

// Not every integration-test binary that pulls in this shared `mod common;` calls
// every helper (e.g. test_atomize_e2e.rs never reads on-disk binary output). Each
// test file is its own compilation unit, so clippy's dead-code pass is per-binary;
// allow(dead_code) keeps `cargo clippy --tests -D warnings` clean across all of them.
#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn read_offsets_npy(path: &Path) -> Vec<u64> {
    let arr: ndarray::Array1<u64> = ndarray_npy::read_npy(path).expect("read offsets npy");
    arr.to_vec()
}

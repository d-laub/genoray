// Shared BCF test harness: synthetic record builder and binary I/O helpers
// for E2E pipeline testing.

use genoray_core::process_chromosome;
use ndarray::{Array1, Array2};
use proptest::prelude::*;
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

/// Deletion length implied by a record's ref/alt lengths (`max(0, -ilen)`).
#[allow(dead_code)]
pub fn del_len_of(rec: &SynthRecord) -> u32 {
    let ilen = rec.alts[0].len() as i32 - rec.ref_allele.len() as i32;
    if ilen < 0 { (-ilen) as u32 } else { 0 }
}

/// Write the `max_del` sidecars for a finished contig. Conservative per-column
/// bound: each `(sample, ploid)` column's max over ALL deletions it carries (an
/// over-estimate vs. the var_key-only contract, but `overlap_range`'s overshoot
/// is proven safe — see `search.rs` `overlap_max_region_length_overshoot_is_safe`),
/// and the global max for `dense/max_del`. This exercises per-column indexing in
/// the consumer while remaining independent of the producer's tight per-column bound.
#[allow(dead_code)]
pub fn write_max_del_fixture(
    contig_dir: &Path,
    n_samples: usize,
    ploidy: usize,
    records: &[SynthRecord],
) {
    let columns = n_samples * ploidy;
    let mut per_col = vec![0u32; columns];
    let mut global = 0u32;
    for rec in records {
        let d = del_len_of(rec);
        global = global.max(d);
        for (hap, &g) in rec.gt.iter().enumerate() {
            if g == 1 {
                per_col[hap] = per_col[hap].max(d);
            }
        }
    }
    let arr = Array2::from_shape_vec((n_samples, ploidy), per_col).unwrap();
    ndarray_npy::write_npy(contig_dir.join("max_del.npy"), &arr).unwrap();

    std::fs::create_dir_all(contig_dir.join("dense")).unwrap();
    let dense = Array1::from_vec(vec![global]);
    ndarray_npy::write_npy(contig_dir.join("dense").join("max_del.npy"), &dense).unwrap();
}

/// Convert `records` to a finished SVAR2 contig under `out/{chrom}` and write the
/// `max_del` fixture. `out` must already exist.
#[allow(dead_code)]
pub fn build_contig(
    out: &Path,
    chrom: &str,
    samples: &[&str],
    ploidy: usize,
    records: &[SynthRecord],
) {
    let bcf = out.join("in.bcf");
    let fasta = out.join("in.fa");
    build_bcf_with_index(&bcf, chrom, 1_000_000, samples, records);
    // M2b: the reader validates REF and left-aligns against a reference FASTA.
    // Stamp each record's REF into an 'N'-filler contig; 'N' never satisfies the
    // left-align repeat condition, so positions stay put and the oracle holds.
    build_fasta_with_index(&fasta, chrom, 1_000_000, records);
    process_chromosome(
        bcf.to_str().unwrap(),
        Some(fasta.to_str().unwrap()),
        chrom,
        out.to_str().unwrap(),
        samples,
        1000, // chunk_size
        ploidy,
        1,    // htslib_threads
        4096, // long_allele_capacity
        false,
        1, // processing_threads
    )
    .expect("process_chromosome should succeed");
    write_max_del_fixture(&out.join(chrom), samples.len(), ploidy, records);
}

/// Owned analogue of `SynthRecord` (proptest needs values that outlive the
/// borrow). One atomized bi-allelic variant.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct OwnedRecord {
    pub pos: i64,
    pub ref_allele: Vec<u8>,
    pub alt: Vec<u8>,
    pub gt: Vec<i32>, // len == n_haps
}

/// Random atomized contig: strictly increasing positions, each variant a SNP,
/// INS (alt = anchor + tail), or DEL (ref = anchor + tail), with random per-hap
/// genotypes. INS tails reach 15 bases so some insertions spill to the LUT.
#[allow(dead_code)]
pub fn arb_records(n_haps: usize) -> impl Strategy<Value = Vec<OwnedRecord>> {
    proptest::collection::vec(
        (
            0u8..3u8,                                            // kind: 0 SNP, 1 INS, 2 DEL
            0usize..4,                                           // anchor base index
            0usize..4,                                           // SNP alt base index
            proptest::collection::vec(0usize..4, 1..16),         // INS/DEL tail (>= 1 base)
            proptest::collection::vec(0i32..2, n_haps..=n_haps), // genotypes
            17u32..40u32, // position gap: >= 17 keeps an 'N' between records' REF
                          // stamps (max REF span is 16), so no overlap and M2b never left-shifts
        ),
        1..8, // 1..7 records; empty contigs are covered by the degenerate tests
    )
    .prop_map(move |specs| {
        const BASES: [u8; 4] = [b'A', b'C', b'T', b'G'];
        let mut pos: i64 = 100;
        let mut out = Vec::new();
        for (kind, anchor, snp_alt, tail, gt, gap) in specs {
            pos += gap as i64;
            let b0 = BASES[anchor];
            // M2b left-aligns any indel whose anchor base repeats at the allele's
            // end (roll condition: anchor == tail's last base). Force them distinct
            // so generated records are already left-canonical and the converter
            // leaves positions put — keeping the brute-force oracle valid.
            let mut tail = tail;
            if tail.last() == Some(&anchor) {
                *tail.last_mut().unwrap() = (anchor + 1) % 4;
            }
            let (ref_allele, alt) = match kind {
                0 => {
                    let alt_idx = if snp_alt == anchor {
                        (anchor + 1) % 4
                    } else {
                        snp_alt
                    };
                    (vec![b0], vec![BASES[alt_idx]])
                }
                1 => {
                    let mut a = vec![b0];
                    a.extend(tail.iter().map(|&x| BASES[x]));
                    (vec![b0], a)
                }
                _ => {
                    let mut r = vec![b0];
                    r.extend(tail.iter().map(|&x| BASES[x]));
                    (r, vec![b0])
                }
            };
            out.push(OwnedRecord {
                pos,
                ref_allele,
                alt,
                gt,
            });
        }
        out
    })
}

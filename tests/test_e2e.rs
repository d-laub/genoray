// End-to-end pipeline test.
//
// Builds a tiny synthetic BCF using rust-htslib::bcf::Writer, indexes it,
// pushes it through the full process_chromosome → merge pipeline, and
// validates the final sample-major sparse outputs against hand-computed
// ground truth.
//
// Also covers the negative side of the atomization contract: multi-allelic
// records and complex variants must panic at the reader, not silently corrupt
// downstream data.

use genoray_core::process_chromosome;
use genoray_core::rvk::decode_alt_inline;
use genoray_core::vcf_reader::VcfChunkReader;

use rust_htslib::bcf::record::GenotypeAllele;
use rust_htslib::bcf::{Format, Header, Writer};

use std::path::Path;
use tempfile::tempdir;

// One synthetic VCF record: position, ref allele, list of alt alleles, and a
// flat genotype vector laid out as [s0_p0, s0_p1, s1_p0, s1_p1, ...] holding
// allele indices (0 = ref, 1 = first alt, …).
struct SynthRecord<'a> {
    pos: i64,
    ref_allele: &'a [u8],
    alts: Vec<&'a [u8]>,
    gt: Vec<i32>,
}

fn build_bcf_with_index(
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

fn read_u32_bin(path: &Path) -> Vec<u32> {
    // Alignment-agnostic decode: std::fs::read can hand back a Vec<u8> whose
    // pointer isn't 4-byte aligned (especially when empty), which would trip
    // bytemuck::cast_slice's TargetAlignmentGreater check.
    let bytes = std::fs::read(path).expect("read u32 bin");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_offsets_npy(path: &Path) -> Vec<u64> {
    let arr: ndarray::Array1<u64> = ndarray_npy::read_npy(path).expect("read offsets npy");
    arr.to_vec()
}

// Decode a single packed key. Discriminator:
//   bit 0 = 1                 → lookup (long allele bank row index)
//   bit 0 = 0 AND bit 31 = 1  → pure DEL (signed ilen in upper 31 bits)
//   bit 0 = 0 AND bit 31 = 0  → inline ALT (top 5 bits hold the length field)
#[derive(Debug, PartialEq)]
enum DecodedKey {
    Inline { alt: Vec<u8> },
    PureDel { ilen: i32 },
    Lookup { row: u32 },
}

fn decode_key(key: u32) -> DecodedKey {
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

// Positive E2E: 3 records spanning SNP / INS / pure DEL across 2 samples diploid.
//   pos=100  A  → C    (SNP)        gt = [1, 0, 1, 1]   → haps 0, 2, 3 carry it
//   pos=200  A  → AT   (INS)        gt = [0, 1, 0, 0]   → only hap 1
//   pos=300  AT → A    (pure DEL)   gt = [1, 1, 0, 0]   → haps 0, 1
//
// Expected per-hap (sample-major) call positions:
//   hap 0 (S0_p0): [100, 300]
//   hap 1 (S0_p1): [200, 300]
//   hap 2 (S1_p0): [100]
//   hap 3 (S1_p1): [100]
// → final_offsets = [0, 2, 4, 5, 6]
// → final_positions = [100, 300,  200, 300,  100,  100]
#[test]
fn test_e2e_normalized_bcf_pipeline() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("test.bcf");

    let samples = vec!["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 1, 1],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![0, 1, 0, 0],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 0, 0],
        },
    ];

    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    process_chromosome(
        bcf_path.to_str().unwrap(),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,  // chunk_size
        2,    // ploidy
        1,    // htslib_threads
        4096, // long_allele_capacity
    );

    let chrom_dir = out_dir.join("chr1/var_key");
    let final_pos = read_u32_bin(&chrom_dir.join("final_positions.bin"));
    let final_key = read_u32_bin(&chrom_dir.join("final_keys.bin"));
    let final_off = read_offsets_npy(&chrom_dir.join("final_offsets.npy"));

    // 4 haps + 1 sentinel
    assert_eq!(final_off.len(), 5);
    assert_eq!(final_off, vec![0u64, 2, 4, 5, 6]);

    // hap 0: SNP@100, DEL@300
    assert_eq!(&final_pos[0..2], &[100, 300]);
    // hap 1: INS@200, DEL@300
    assert_eq!(&final_pos[2..4], &[200, 300]);
    // hap 2: SNP@100
    assert_eq!(&final_pos[4..5], &[100]);
    // hap 3: SNP@100
    assert_eq!(&final_pos[5..6], &[100]);

    // Decode the keys to confirm payload semantics.
    // hap 0 calls: [SNP A→C, DEL AT→A]
    match decode_key(final_key[0]) {
        DecodedKey::Inline { alt } => assert_eq!(alt, b"C".to_vec()),
        other => panic!("expected inline SNP, got {:?}", other),
    }
    match decode_key(final_key[1]) {
        DecodedKey::PureDel { ilen } => assert_eq!(ilen, -1),
        other => panic!("expected PureDel ilen=-1, got {:?}", other),
    }
    // hap 1 calls: [INS A→AT, DEL AT→A]
    match decode_key(final_key[2]) {
        DecodedKey::Inline { alt } => assert_eq!(alt, b"AT".to_vec()),
        other => panic!("expected inline INS, got {:?}", other),
    }
    match decode_key(final_key[3]) {
        DecodedKey::PureDel { ilen } => assert_eq!(ilen, -1),
        other => panic!("expected PureDel, got {:?}", other),
    }
}

// Mutation conservation across the full pipeline: the total length of
// final_positions equals the total number of true genotype calls in the input.
// Catches drops anywhere — reader, transpose, writer, merge.
#[test]
fn test_e2e_mutation_conservation() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("conserve.bcf");

    let samples = vec!["S0", "S1", "S2"];
    let records = vec![
        SynthRecord {
            pos: 50,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![1, 1, 0, 0, 1, 0],
        },
        SynthRecord {
            pos: 150,
            ref_allele: b"C",
            alts: vec![&b"T"[..]],
            gt: vec![0, 1, 1, 1, 0, 0],
        },
        SynthRecord {
            pos: 250,
            ref_allele: b"G",
            alts: vec![&b"A"[..]],
            gt: vec![1, 0, 0, 1, 1, 1],
        },
    ];
    let expected_calls: usize = records
        .iter()
        .map(|r| r.gt.iter().filter(|&&g| g == 1).count())
        .sum();

    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    process_chromosome(
        bcf_path.to_str().unwrap(),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,
        2,
        1,
        4096,
    );

    let chrom_dir = out_dir.join("chr1/var_key");
    let final_pos = read_u32_bin(&chrom_dir.join("final_positions.bin"));
    let final_off = read_offsets_npy(&chrom_dir.join("final_offsets.npy"));

    assert_eq!(
        final_pos.len(),
        expected_calls,
        "mutation conservation across pipeline"
    );
    assert_eq!(*final_off.last().unwrap() as usize, expected_calls);
    assert_eq!(final_off.len(), samples.len() * 2 + 1);
}

// ─────────────────────────────────────────────────────────────────────────
// Negative tests — the reader must panic on inputs that violate the
// `bcftools norm -m -any --atomize` contract. We invoke VcfChunkReader
// directly so the panic surfaces in the test thread (process_chromosome
// would route it through JoinHandle::unwrap, masking the message).
// ─────────────────────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "must be normalized")]
fn test_reader_panics_on_multi_allelic() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("multi.bcf");

    let samples = vec!["S0"];
    let records = vec![
        // 3-allele record: REF + 2 ALTs → forbidden by `bcftools norm -m -any`
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..], &b"G"[..]],
            gt: vec![1, 2],
        },
    ];
    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);

    let mut reader = VcfChunkReader::new(bcf_path.to_str().unwrap(), "chr1", &samples, 1, 2);
    let _ = reader.read_next_chunk(100, 0); // panics here
}

#[test]
#[should_panic(expected = "must be atomized")]
fn test_reader_panics_on_complex_variant() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("complex.bcf");

    let samples = vec!["S0"];
    let records = vec![
        // Complex variant: ref=ATG (3bp), alt=CG (2bp).
        // alt_len=2 > 1 AND alt_len=2 < ref_len=3 → forbidden by `bcftools norm --atomize`
        SynthRecord {
            pos: 100,
            ref_allele: b"ATG",
            alts: vec![&b"CG"[..]],
            gt: vec![1, 1],
        },
    ];
    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);

    let mut reader = VcfChunkReader::new(bcf_path.to_str().unwrap(), "chr1", &samples, 1, 2);
    let _ = reader.read_next_chunk(100, 0); // panics here
}

// Sanity: a clean, properly atomized DEL (alt_len=1) does NOT trip the panic.
// Acts as a regression guard: we don't want the atomization assert to false-
// positive on legitimate pure deletions.
#[test]
fn test_reader_accepts_pure_del() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("puredel.bcf");

    let samples = vec!["S0"];
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"ATG",
        alts: vec![&b"A"[..]],
        gt: vec![1, 1],
    }];
    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);

    let mut reader = VcfChunkReader::new(bcf_path.to_str().unwrap(), "chr1", &samples, 1, 2);
    let chunk = reader
        .read_next_chunk(100, 0)
        .expect("chunk should succeed");
    assert_eq!(chunk.num_variants, 1);
    assert_eq!(chunk.ilens, vec![-2]);
}

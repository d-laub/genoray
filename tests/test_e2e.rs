// End-to-end pipeline test.
//
// Builds a tiny synthetic BCF using rust-htslib::bcf::Writer, indexes it,
// pushes it through the full process_chromosome → merge pipeline, and
// validates the final sample-major sparse outputs against hand-computed
// ground truth.

mod common;

use common::{SynthRecord, build_bcf_with_index, read_offsets_npy, read_u32_bin};
use genoray_core::process_chromosome;
use genoray_core::rvk::{decode_alt_inline, unpack_snp_keys};
use genoray_core::vcf_reader::VcfChunkReader;

use tempfile::tempdir;

/// Decode a single packed key. Discriminator:
///   bit 0 = 1                 → lookup (long allele bank row index)
///   bit 0 = 0 AND bit 31 = 1  → pure DEL (signed ilen in upper 31 bits)
///   bit 0 = 0 AND bit 31 = 0  → inline ALT (top 5 bits hold the length field)
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
    )
    .expect("process_chromosome should succeed");

    let snp_dir = out_dir.join("chr1/var_key/snp");
    let indel_dir = out_dir.join("chr1/var_key/indel");

    // ---- SNP stream ----
    let snp_pos = read_u32_bin(&snp_dir.join("final_positions.bin"));
    let snp_off = read_offsets_npy(&snp_dir.join("final_offsets.npy"));
    let snp_key_packed = std::fs::read(snp_dir.join("final_keys.bin")).unwrap();

    // 4 haps + sentinel; SNP calls per hap: [1, 0, 1, 1]
    assert_eq!(snp_off, vec![0u64, 1, 1, 2, 3]);
    assert_eq!(snp_pos, vec![100, 100, 100]);
    // 'C' → code 1; three carriers → codes [1, 1, 1]
    let snp_codes = unpack_snp_keys(&snp_key_packed, snp_pos.len());
    assert_eq!(snp_codes, vec![1u8, 1, 1]);

    // ---- Indel stream ----
    let indel_pos = read_u32_bin(&indel_dir.join("final_positions.bin"));
    let indel_key = read_u32_bin(&indel_dir.join("final_keys.bin"));
    let indel_off = read_offsets_npy(&indel_dir.join("final_offsets.npy"));

    // indel calls per hap: [1, 2, 0, 0]
    assert_eq!(indel_off, vec![0u64, 1, 3, 3, 3]);
    assert_eq!(indel_pos, vec![300, 200, 300]);

    // hap0: DEL@300
    match decode_key(indel_key[0]) {
        DecodedKey::PureDel { ilen } => assert_eq!(ilen, -1),
        other => panic!("expected PureDel ilen=-1, got {:?}", other),
    }
    // hap1: INS@200 then DEL@300
    match decode_key(indel_key[1]) {
        DecodedKey::Inline { alt } => assert_eq!(alt, b"AT".to_vec()),
        other => panic!("expected inline INS AT, got {:?}", other),
    }
    match decode_key(indel_key[2]) {
        DecodedKey::PureDel { ilen } => assert_eq!(ilen, -1),
        other => panic!("expected PureDel ilen=-1, got {:?}", other),
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
    )
    .expect("process_chromosome should succeed");

    let snp_dir = out_dir.join("chr1/var_key/snp");
    let indel_dir = out_dir.join("chr1/var_key/indel");

    let snp_pos = read_u32_bin(&snp_dir.join("final_positions.bin"));
    let snp_off = read_offsets_npy(&snp_dir.join("final_offsets.npy"));
    let indel_pos = read_u32_bin(&indel_dir.join("final_positions.bin"));

    // Every SNP call is conserved; the indel stream is empty.
    assert_eq!(
        snp_pos.len(),
        expected_calls,
        "mutation conservation across pipeline"
    );
    assert_eq!(*snp_off.last().unwrap() as usize, expected_calls);
    assert_eq!(snp_off.len(), samples.len() * 2 + 1);
    assert_eq!(indel_pos.len(), 0);
}

// Sanity: a clean, properly atomized DEL (alt_len=1) passes the reader.
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
    assert_eq!(chunk.ilens.len(), 1);
    assert_eq!(chunk.ilens, vec![-2]);
}

// Boundary error-handling: a chromosome absent from the VCF header must surface
// as a typed `ConversionError::WorkerPanicked` (the reader thread panics on
// `name2rid`, and `process_chromosome` converts the join panic into an Err)
// rather than unwinding the calling thread.
#[test]
fn test_missing_chrom_returns_err() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("missing_chrom.bcf");

    let samples = vec!["s0"];
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt: vec![1, 0],
    }];
    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    let res = genoray_core::orchestrator::process_chromosome(
        bcf_path.to_str().unwrap(),
        "chrZ",
        out_dir.to_str().unwrap(),
        &["s0"],
        1000,
        2,
        1,
        1 << 20,
    );

    assert!(matches!(
        res,
        Err(genoray_core::error::ConversionError::WorkerPanicked { .. })
    ));
}

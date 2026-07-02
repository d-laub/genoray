// End-to-end pipeline test.
//
// Builds a tiny synthetic BCF using rust-htslib::bcf::Writer, indexes it,
// pushes it through the full process_chromosome → merge pipeline, and
// validates the final sample-major sparse outputs against hand-computed
// ground truth.

mod common;

use common::{SynthRecord, build_bcf_with_index, read_offsets_npy, read_u32_bin};
use genoray_core::process_chromosome;
use genoray_core::rvk::decode_alt_inline;
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
// → offsets = [0, 2, 4, 5, 6]
// → positions = [100, 300,  200, 300,  100,  100]
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

    // Routing (np=4): common SNP@100 (x=3) → dense/snp; INS@200 (x=1) →
    // var_key/indel; DEL@300 (x=2) → dense/indel.
    //
    // The orchestrator drives the dense merge (Task 11), so per-chunk dense
    // output is merged into `final_*` files (and the per-chunk temp files
    // are removed) just like the var_key streams.
    let dsnp = out_dir.join("chr1/dense/snp");
    let dindel = out_dir.join("chr1/dense/indel");
    let vk_indel = out_dir.join("chr1/var_key/indel");

    // dense/snp: 1 variant @100, geno bits for haps 0,2,3.
    let dsnp_pos = read_u32_bin(&dsnp.join("positions.bin"));
    assert_eq!(dsnp_pos, vec![100]);
    let dsnp_geno = std::fs::read(dsnp.join("genotypes.bin")).unwrap();
    // np=4, v_dense=1 → bit h*1+0. Carriers = haps 0,2,3 (gt [1,0,1,1]).
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 0));
    assert!(!genoray_core::bits::get_bit(&dsnp_geno, 1));
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 2));
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 3));

    // var_key/indel: only INS@200 for hap 1 (this stream IS merged, since
    // orchestrator already drives var_key merge via `REGISTRY`). Also decode
    // the key to confirm it's the inline "AT" insertion, not a pure DEL or
    // lookup row.
    let vki_pos = read_u32_bin(&vk_indel.join("positions.bin"));
    let vki_off = read_offsets_npy(&vk_indel.join("offsets.npy"));
    let vki_key = read_u32_bin(&vk_indel.join("alleles.bin"));
    assert_eq!(vki_off, vec![0u64, 0, 1, 1, 1]); // hap1 has the single INS call
    assert_eq!(vki_pos, vec![200]);
    match decode_key(vki_key[0]) {
        DecodedKey::Inline { alt } => assert_eq!(alt, b"AT".to_vec()),
        other => panic!("expected inline INS AT, got {:?}", other),
    }

    // dense/indel: DEL@300 (x=2, haps 0,1).
    let dindel_pos = read_u32_bin(&dindel.join("positions.bin"));
    assert_eq!(dindel_pos, vec![300]);
    let dindel_geno = std::fs::read(dindel.join("genotypes.bin")).unwrap();
    assert!(genoray_core::bits::get_bit(&dindel_geno, 0)); // hap0
    assert!(genoray_core::bits::get_bit(&dindel_geno, 1)); // hap1
    assert!(!genoray_core::bits::get_bit(&dindel_geno, 2));
    assert!(!genoray_core::bits::get_bit(&dindel_geno, 3));
}

// Dense round-trip: a SNP carried by most of a small cohort must be routed to
// dense/snp and its genotype bits reconstructable from genotypes.bin.
#[test]
fn test_e2e_dense_snp_roundtrip() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("dense.bcf");
    let samples = vec!["S0", "S1", "S2"]; // np = 6

    // One SNP A→C carried by haps 0,1,2,3,4 (5 of 6). x=5.
    // dense = 32+2+6 = 40 < var_key 5*34 = 170 → dense.
    let records = vec![SynthRecord {
        pos: 500,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt: vec![1, 1, 1, 1, 1, 0], // S0(1,1) S1(1,1) S2(1,0)
    }];
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
    .expect("conversion");

    let dsnp = out_dir.join("chr1/dense/snp");
    let pos = read_u32_bin(&dsnp.join("positions.bin"));
    assert_eq!(pos, vec![500]);
    // packed key: 'C' code 1, 1 variant → pack_snp_keys([1]) == [0x01]
    assert_eq!(
        std::fs::read(dsnp.join("alleles.bin")).unwrap(),
        vec![0x01u8]
    );

    // genotypes: np=6, v_dense=1 → bit h. Haps 0..5 carriers = [1,1,1,1,1,0].
    let geno = std::fs::read(dsnp.join("genotypes.bin")).unwrap();
    for h in 0..5 {
        assert!(genoray_core::bits::get_bit(&geno, h), "hap {}", h);
    }
    assert!(!genoray_core::bits::get_bit(&geno, 5));
}

// Mutation conservation across the full pipeline: the total length of
// positions equals the total number of true genotype calls in the input.
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

    // Routing (np=6): all three SNPs have x=3,3,4 carriers — every one clears
    // the dense crossover (34*x > 34+np=40), so they ALL route to dense/snp
    // and var_key/snp is empty. Conservation must therefore be checked
    // across every bucket: var_key positions lengths + dense genotype
    // popcounts — both merged by the orchestrator (Task 11 wires the dense
    // merge alongside the pre-existing var_key merge via `REGISTRY`).
    let mut total = 0usize;
    for sub in ["var_key/snp", "var_key/indel"] {
        let p = out_dir.join(format!("chr1/{sub}/positions.bin"));
        if p.exists() {
            total += read_u32_bin(&p).len();
        }
    }
    for sub in ["dense/snp", "dense/indel"] {
        let g = out_dir.join(format!("chr1/{sub}/genotypes.bin"));
        if g.exists() {
            total += std::fs::read(&g)
                .unwrap()
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum::<usize>();
        }
    }
    assert_eq!(
        total, expected_calls,
        "mutation conservation across pipeline"
    );
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

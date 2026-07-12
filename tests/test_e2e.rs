// End-to-end pipeline test.
//
// Builds a tiny synthetic BCF using rust-htslib::bcf::Writer, indexes it,
// pushes it through the full process_chromosome → merge pipeline, and
// validates the final sample-major sparse outputs against hand-computed
// ground truth.

mod common;

use common::{
    SynthRecord, build_bcf_with_index, build_fasta_with_index, read_offsets_npy, read_u32_bin,
};
use genoray_core::process_chromosome;
use genoray_core::rvk::decode_alt_inline;
use genoray_core::search::{SearchTree, overlap_range};
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
    build_fasta_with_index(&bcf_path.with_extension("fa"), "chr1", 10_000, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    process_chromosome(
        bcf_path.to_str().unwrap(),
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,  // chunk_size
        2,    // ploidy
        1,    // htslib_threads
        4096, // long_allele_capacity
        false,
        1,     // processing_threads
        false, // signatures
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

// M5 max_del post-pass, end-to-end. Two deletions with known lengths:
//   pos=100  ATTT → A  (d=3)  gt=[1,0,0,0]  x=1 → rare → var_key/indel (hap 0)
//   pos=200  ATT  → A  (d=2)  gt=[1,1,1,0]  x=3 → common → dense/indel (shared)
// Asserts the produced artifacts, then feeds max_del into overlap_range to prove
// a deletion that starts left of the query but spans into it is recoverable.
#[test]
fn test_e2e_max_del_postpass() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("maxdel.bcf");
    let samples = vec!["S0", "S1"]; // np = 4

    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"ATTT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 0, 0, 0], // hap 0 only → var_key/indel
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"ATT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 1, 0], // haps 0,1,2 → dense/indel
        },
    ];
    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);
    build_fasta_with_index(&bcf_path.with_extension("fa"), "chr1", 10_000, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    process_chromosome(
        bcf_path.to_str().unwrap(),
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,
        2,
        1,
        4096,
        false,
        1,     // processing_threads
        false, // signatures
    )
    .expect("conversion");

    let contig_dir = out_dir.join("chr1");

    // var_key max_del: shape (2, 2); only col 0 (S0_p0) carries the d=3 deletion.
    let m: ndarray::Array2<u32> = ndarray_npy::read_npy(contig_dir.join("max_del.npy")).unwrap();
    assert_eq!(m.shape(), &[2, 2]);
    assert_eq!(m[[0, 0]], 3);
    assert_eq!(m[[0, 1]], 0);
    assert_eq!(m[[1, 0]], 0);
    assert_eq!(m[[1, 1]], 0);

    // dense max_del: shape (1,); single max over the shared indel table = 2.
    let dm: ndarray::Array1<u32> =
        ndarray_npy::read_npy(contig_dir.join("dense/max_del.npy")).unwrap();
    assert_eq!(dm, ndarray::Array1::from_vec(vec![2u32]));

    // Close the producer -> consumer loop: the var_key/indel deletion for hap 0
    // starts at pos 100 and spans 3 bases (v_end = 100 + 1 + 3 = 104). A query
    // strictly right of the start but inside the deletion must still find it,
    // using the produced max_del as max_region_length.
    let vk_indel = contig_dir.join("var_key/indel");
    let off = read_offsets_npy(&vk_indel.join("offsets.npy"));
    let pos = read_u32_bin(&vk_indel.join("positions.bin"));
    // hap 0 owns exactly the [off[0], off[1]) slice = one call at pos 100.
    let col0 = &pos[off[0] as usize..off[1] as usize];
    assert_eq!(col0, &[100u32]);

    let v_starts: Vec<u32> = col0.to_vec();
    let v_ends: Vec<u32> = vec![100 + 1 + 3]; // start + 1 + d, d = max_del[0,0]
    let tree = SearchTree::new(&v_starts);
    let max_region_length = m[[0, 0]];
    // query [102, 103): starts right of variant start 100 but inside the deletion.
    assert_eq!(
        overlap_range(&tree, &v_ends, max_region_length, 102, 103),
        (0, 1)
    );
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
    build_fasta_with_index(&bcf_path.with_extension("fa"), "chr1", 10_000, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    process_chromosome(
        bcf_path.to_str().unwrap(),
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,
        2,
        1,
        4096,
        false,
        1,     // processing_threads
        false, // signatures
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
    build_fasta_with_index(&bcf_path.with_extension("fa"), "chr1", 10_000, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    process_chromosome(
        bcf_path.to_str().unwrap(),
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,
        2,
        1,
        4096,
        false,
        1,     // processing_threads
        false, // signatures
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
    build_fasta_with_index(&bcf_path.with_extension("fa"), "chr1", 10_000, &records);

    let mut reader = VcfChunkReader::new(
        bcf_path.to_str().unwrap(),
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        &samples,
        1,
        2,
        false,
    )
    .unwrap();
    let chunk = reader
        .read_next_chunk(100, 0, None)
        .expect("chunk should succeed")
        .expect("chunk should succeed");
    assert_eq!(chunk.ilens.len(), 1);
    assert_eq!(chunk.ilens, vec![-2]);
}

// Boundary error-handling: a chromosome absent from the VCF header must surface
// as a typed `ConversionError::Input` (SP-3: `VcfChunkReader::new` now returns
// `Result` instead of panicking on `name2rid`, so the reader thread's closure
// propagates it and the join surfaces the real error, not a swallowed
// `WorkerPanicked`).
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
    build_fasta_with_index(&bcf_path.with_extension("fa"), "chr1", 10_000, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    let res = genoray_core::orchestrator::process_chromosome(
        bcf_path.to_str().unwrap(),
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chrZ",
        out_dir.to_str().unwrap(),
        &["s0"],
        1000,
        2,
        1,
        1 << 20,
        false,
        1,     // processing_threads
        false, // signatures
    );

    assert!(matches!(
        res,
        Err(genoray_core::error::ConversionError::Input(_))
    ));
}

// Boundary error-handling: a VCF/BCF path that does not exist on disk must
// surface as a typed `ConversionError::MissingFile`, symmetric with the FASTA
// missing-file check (SP-3 final review, M2).
#[test]
fn test_missing_vcf_returns_missing_file() {
    let tmp = tempdir().unwrap();
    let missing_path = tmp.path().join("does_not_exist.vcf.gz");

    let res = VcfChunkReader::new(
        missing_path.to_str().unwrap(),
        None,
        "chr1",
        &["s0"],
        1,
        2,
        false,
    );

    assert!(matches!(
        res,
        Err(genoray_core::error::ConversionError::MissingFile { .. })
    ));
}

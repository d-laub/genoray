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
    let dsnp_pos = read_u32_bin(&dsnp.join("final_positions.bin"));
    assert_eq!(dsnp_pos, vec![100]);
    let dsnp_geno = std::fs::read(dsnp.join("final_genotypes.bin")).unwrap();
    // np=4, v_dense=1 → bit h*1+0. Carriers = haps 0,2,3 (gt [1,0,1,1]).
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 0));
    assert!(!genoray_core::bits::get_bit(&dsnp_geno, 1));
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 2));
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 3));

    // var_key/indel: only INS@200 for hap 1 (this stream IS merged, since
    // orchestrator already drives var_key merge via `REGISTRY`). Also decode
    // the key to confirm it's the inline "AT" insertion, not a pure DEL or
    // lookup row.
    let vki_pos = read_u32_bin(&vk_indel.join("final_positions.bin"));
    let vki_off = read_offsets_npy(&vk_indel.join("final_offsets.npy"));
    let vki_key = read_u32_bin(&vk_indel.join("final_keys.bin"));
    assert_eq!(vki_off, vec![0u64, 0, 1, 1, 1]); // hap1 has the single INS call
    assert_eq!(vki_pos, vec![200]);
    match decode_key(vki_key[0]) {
        DecodedKey::Inline { alt } => assert_eq!(alt, b"AT".to_vec()),
        other => panic!("expected inline INS AT, got {:?}", other),
    }

    // dense/indel: DEL@300 (x=2, haps 0,1).
    let dindel_pos = read_u32_bin(&dindel.join("final_positions.bin"));
    assert_eq!(dindel_pos, vec![300]);
    let dindel_geno = std::fs::read(dindel.join("final_genotypes.bin")).unwrap();
    assert!(genoray_core::bits::get_bit(&dindel_geno, 0)); // hap0
    assert!(genoray_core::bits::get_bit(&dindel_geno, 1)); // hap1
    assert!(!genoray_core::bits::get_bit(&dindel_geno, 2));
    assert!(!genoray_core::bits::get_bit(&dindel_geno, 3));
}

// Dense round-trip: a SNP carried by most of a small cohort must be routed to
// dense/snp and its genotype bits reconstructable from final_genotypes.bin.
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
    let pos = read_u32_bin(&dsnp.join("final_positions.bin"));
    assert_eq!(pos, vec![500]);
    // packed key: 'C' code 1, 1 variant → pack_snp_keys([1]) == [0x01]
    assert_eq!(
        std::fs::read(dsnp.join("final_keys.bin")).unwrap(),
        vec![0x01u8]
    );

    // genotypes: np=6, v_dense=1 → bit h. Haps 0..5 carriers = [1,1,1,1,1,0].
    let geno = std::fs::read(dsnp.join("final_genotypes.bin")).unwrap();
    for h in 0..5 {
        assert!(genoray_core::bits::get_bit(&geno, h), "hap {}", h);
    }
    assert!(!genoray_core::bits::get_bit(&geno, 5));
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

    // Routing (np=6): all three SNPs have x=3,3,4 carriers — every one clears
    // the dense crossover (34*x > 34+np=40), so they ALL route to dense/snp
    // and var_key/snp is empty. Conservation must therefore be checked
    // across every bucket: var_key final_positions lengths + dense genotype
    // popcounts — both merged by the orchestrator (Task 11 wires the dense
    // merge alongside the pre-existing var_key merge via `REGISTRY`).
    let mut total = 0usize;
    for sub in ["var_key/snp", "var_key/indel"] {
        let p = out_dir.join(format!("chr1/{sub}/final_positions.bin"));
        if p.exists() {
            total += read_u32_bin(&p).len();
        }
    }
    for sub in ["dense/snp", "dense/indel"] {
        let g = out_dir.join(format!("chr1/{sub}/final_genotypes.bin"));
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

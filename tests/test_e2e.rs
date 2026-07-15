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
use genoray_core::chunk_assembler::ChunkAssembler;
use genoray_core::process_chromosome;
use genoray_core::rvk::decode_alt_inline;
use genoray_core::search::{SearchTree, overlap_range};
use genoray_core::vcf_reader::VcfRecordSource;

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
        genoray_core::orchestrator::SourceSpec::Vcf {
            vcf_path: bcf_path.to_str().unwrap().to_string(),
            htslib_threads: 1,
        },
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,  // chunk_size
        2,    // ploidy
        4096, // long_allele_capacity
        false,
        genoray_core::normalize::CheckRef::Error,
        1,     // processing_threads
        false, // signatures
        &[],   // fields
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
        genoray_core::orchestrator::SourceSpec::Vcf {
            vcf_path: bcf_path.to_str().unwrap().to_string(),
            htslib_threads: 1,
        },
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,
        2,
        4096,
        false,
        genoray_core::normalize::CheckRef::Error,
        1,     // processing_threads
        false, // signatures
        &[],   // fields
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
        genoray_core::orchestrator::SourceSpec::Vcf {
            vcf_path: bcf_path.to_str().unwrap().to_string(),
            htslib_threads: 1,
        },
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,
        2,
        4096,
        false,
        genoray_core::normalize::CheckRef::Error,
        1,     // processing_threads
        false, // signatures
        &[],   // fields
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
        genoray_core::orchestrator::SourceSpec::Vcf {
            vcf_path: bcf_path.to_str().unwrap().to_string(),
            htslib_threads: 1,
        },
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,
        2,
        4096,
        false,
        genoray_core::normalize::CheckRef::Error,
        1,     // processing_threads
        false, // signatures
        &[],   // fields
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

// Task 4: INFO/FORMAT field extraction into DenseChunk. Builds a DEDICATED
// small BCF (not the shared SynthRecord/build_bcf_with_index helper, which
// has no INFO/FORMAT beyond GT) with an INFO/AC (Number=1, Integer) and a
// FORMAT/DS (Number=1, Float) field over 2 samples, 3 strictly-increasing
// biallelic SNP positions (no indels ⇒ atom order == record order, so
// per-variant assertions stay stable). AC is left unset on the last record to
// exercise the missing → sentinel path (no `default` set on the FieldSpec).
#[test]
fn test_reader_extracts_info_format_fields() {
    use genoray_core::field::{FieldCategory, FieldSpec, HtslibType, StorageDtype};
    use genoray_core::types::StagedColumn;
    use rust_htslib::bcf::record::GenotypeAllele;
    use rust_htslib::bcf::{Format as BcfFormat, Header, Writer};

    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("fields.bcf");

    let samples = ["S0", "S1"];
    let num_samples = samples.len();
    let ploidy = 2;

    // (pos, ref, alt, ac (None = leave INFO/AC unset), ds per sample, gt per hap)
    #[allow(clippy::type_complexity)]
    let records: Vec<(i64, &[u8], &[u8], Option<i32>, [f32; 2], [i32; 4])> = vec![
        (100, b"A", b"C", Some(5), [0.1, 0.2], [1, 0, 0, 1]),
        (200, b"G", b"T", Some(7), [0.3, 0.4], [0, 1, 1, 0]),
        (300, b"C", b"A", None, [0.5, 0.6], [1, 1, 0, 0]),
    ];

    {
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chr1,length=10000>");
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_record(b"##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Allele count\">");
        header.push_record(b"##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Dosage\">");
        for s in &samples {
            header.push_sample(s.as_bytes());
        }

        let mut writer =
            Writer::from_path(&bcf_path, &header, false, BcfFormat::Bcf).expect("open BCF writer");
        for (pos, r, a, ac, ds, gt) in &records {
            let mut record = writer.empty_record();
            record.set_rid(Some(0));
            record.set_pos(*pos);
            record.set_alleles(&[r, a]).expect("set alleles");
            let gt_alleles: Vec<GenotypeAllele> =
                gt.iter().map(|&i| GenotypeAllele::Phased(i)).collect();
            record.push_genotypes(&gt_alleles).expect("push genotypes");
            if let Some(ac) = ac {
                record.push_info_integer(b"AC", &[*ac]).expect("push AC");
            }
            record.push_format_float(b"DS", ds).expect("push DS");
            writer.write(&record).expect("write record");
        }
    }
    rust_htslib::bcf::index::build(&bcf_path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
        .expect("build BCF index");

    let info = vec![FieldSpec {
        name: "AC".to_string(),
        category: FieldCategory::Info,
        htype: HtslibType::Int,
        dtype: StorageDtype::Auto,
        default: None,
    }];
    let fmt = vec![FieldSpec {
        name: "DS".to_string(),
        category: FieldCategory::Format,
        htype: HtslibType::Float,
        dtype: StorageDtype::Auto,
        default: None,
    }];
    let mut all = info.clone();
    all.extend(fmt.clone());

    let sample_refs: Vec<&str> = samples.to_vec();
    // fasta_path = None: skip validate_ref/left_align, so positions stay put
    // and atom order == record order (all SNPs, no atomize splitting).
    let source = VcfRecordSource::new(
        bcf_path.to_str().unwrap(),
        "chr1",
        &sample_refs,
        1,
        ploidy,
        &all,
    )
    .unwrap();
    let mut reader = ChunkAssembler::new(
        Box::new(source),
        sample_refs.len(),
        ploidy,
        None,
        "chr1",
        false,
        genoray_core::normalize::CheckRef::Error,
        &all,
    )
    .unwrap();
    let chunk = reader
        .read_next_chunk(100, 0, None)
        .expect("chunk should succeed")
        .expect("chunk should succeed");

    assert_eq!(chunk.pos, vec![100, 200, 300]);

    let ds = match &chunk.format_staged[0] {
        StagedColumn::Float(v) => v,
        _ => panic!("DS should stage as Float"),
    };
    assert_eq!(ds.len(), chunk.pos.len() * num_samples);
    // variant 0 (pos 100), sample 1 (S1) → index 0*2 + 1
    assert!((ds[1] - 0.2).abs() < 1e-6);
    // variant 1 (pos 200), sample 0 (S0) → index 1*2 + 0
    assert!((ds[2] - 0.3).abs() < 1e-6);
    // variant 2 (pos 300), sample 1 (S1) → index 2*2 + 1
    assert!((ds[5] - 0.6).abs() < 1e-6);

    let ac = match &chunk.info_staged[0] {
        StagedColumn::Int(v) => v,
        _ => panic!("AC should stage as Int"),
    };
    assert_eq!(ac.len(), chunk.pos.len());
    assert_eq!(ac[0], 5);
    assert_eq!(ac[1], 7);
    assert_eq!(ac[2], i32::MIN, "missing AC (no default) → sentinel");
}

// Regression (Task 4 code review, Critical): `resolve_scalar` must index a
// Number=A htslib buffer by `source_alt_index - 1` (0-based per-ALT), not
// `source_alt_index` directly (1-based). Biallelic records can't catch this —
// a single-ALT Number=A buffer has length 1 and hits the `vals.len() == 1`
// scalar branch regardless of index. Only a MULTIALLELIC record exercises the
// `vals.get(idx)` arm: on unpatched code ALT1 (source_alt_index=1) reads
// ALT2's slot and ALT2 (source_alt_index=2) reads out of bounds → sentinel.
//
// Builds one low-pos biallelic SNP (ordering sanity) then a 2-ALT record
// ref="A", alt=["C","G"] with INFO/AC=[10,20] (Number=A Integer) and
// FORMAT/DS Number=A Float, sample-major per htslib layout:
// sample0=[0.1(ALT_C), 0.2(ALT_G)], sample1=[0.3(ALT_C), 0.4(ALT_G)] →
// push_format_float(&[0.1, 0.2, 0.3, 0.4]). `atomize_record` pushes ALTs in
// order (C=source_alt_index 1, G=source_alt_index 2) and the reorder heap
// tiebreaks equal positions by insertion sequence, so the multiallelic
// record's two atoms come out ALT-C then ALT-G. fasta_path=None skips
// left-align, so positions/ordering stay exactly as written.
#[test]
fn test_reader_extracts_multiallelic_number_a_fields() {
    use genoray_core::field::{FieldCategory, FieldSpec, HtslibType, StorageDtype};
    use genoray_core::types::StagedColumn;
    use rust_htslib::bcf::record::GenotypeAllele;
    use rust_htslib::bcf::{Format as BcfFormat, Header, Writer};

    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("multiallelic_fields.bcf");

    let samples = ["S0", "S1"];
    let num_samples = samples.len();
    let ploidy = 2;

    {
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chr1,length=10000>");
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_record(b"##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count\">");
        header.push_record(b"##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Dosage\">");
        for s in &samples {
            header.push_sample(s.as_bytes());
        }

        let mut writer =
            Writer::from_path(&bcf_path, &header, false, BcfFormat::Bcf).expect("open BCF writer");

        // Record 0: biallelic SNP at a lower pos, ordering sanity only.
        {
            let mut record = writer.empty_record();
            record.set_rid(Some(0));
            record.set_pos(50);
            record.set_alleles(&[b"T", b"A"]).expect("set alleles");
            let gt_alleles = vec![
                GenotypeAllele::Phased(1),
                GenotypeAllele::Phased(0),
                GenotypeAllele::Phased(0),
                GenotypeAllele::Phased(0),
            ];
            record.push_genotypes(&gt_alleles).expect("push genotypes");
            record.push_info_integer(b"AC", &[1]).expect("push AC");
            record
                .push_format_float(b"DS", &[0.9, 0.9])
                .expect("push DS");
            writer.write(&record).expect("write record");
        }

        // Record 1: MULTIALLELIC SNP, ref="A", alt=["C","G"].
        {
            let mut record = writer.empty_record();
            record.set_rid(Some(0));
            record.set_pos(200);
            record
                .set_alleles(&[b"A", b"C", b"G"])
                .expect("set alleles");
            // S0: hap0=ALT1(C), hap1=ALT2(G); S1: hom ref.
            let gt_alleles = vec![
                GenotypeAllele::Phased(1),
                GenotypeAllele::Phased(2),
                GenotypeAllele::Phased(0),
                GenotypeAllele::Phased(0),
            ];
            record.push_genotypes(&gt_alleles).expect("push genotypes");
            record.push_info_integer(b"AC", &[10, 20]).expect("push AC");
            // Number=A, 2 samples, sample-major: [s0_altC, s0_altG, s1_altC, s1_altG].
            record
                .push_format_float(b"DS", &[0.1, 0.2, 0.3, 0.4])
                .expect("push DS");
            writer.write(&record).expect("write record");
        }
    }
    rust_htslib::bcf::index::build(&bcf_path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
        .expect("build BCF index");

    let info = vec![FieldSpec {
        name: "AC".to_string(),
        category: FieldCategory::Info,
        htype: HtslibType::Int,
        dtype: StorageDtype::Auto,
        default: None,
    }];
    let fmt = vec![FieldSpec {
        name: "DS".to_string(),
        category: FieldCategory::Format,
        htype: HtslibType::Float,
        dtype: StorageDtype::Auto,
        default: None,
    }];
    let mut all = info.clone();
    all.extend(fmt.clone());

    let sample_refs: Vec<&str> = samples.to_vec();
    let source = VcfRecordSource::new(
        bcf_path.to_str().unwrap(),
        "chr1",
        &sample_refs,
        1,
        ploidy,
        &all,
    )
    .unwrap();
    let mut reader = ChunkAssembler::new(
        Box::new(source),
        sample_refs.len(),
        ploidy,
        None,
        "chr1",
        false,
        genoray_core::normalize::CheckRef::Error,
        &all,
    )
    .unwrap();
    let chunk = reader
        .read_next_chunk(100, 0, None)
        .expect("chunk should succeed")
        .expect("chunk should succeed");

    // 3 atoms total: 1 (pos 50) + 2 (pos 200, the multiallelic split).
    assert_eq!(chunk.pos, vec![50, 200, 200]);
    assert_eq!(chunk.ilens, vec![0, 0, 0], "all SNPs, no ilen shift");

    // Identify the two multiallelic atoms by pos, then confirm which is which
    // via their actual ALT bytes (don't just assume emission order) — this is
    // the "non-vacuous" check: it fails loudly if atomize/heap ordering ever
    // changes instead of silently comparing the wrong atom.
    let multi_idxs: Vec<usize> = chunk
        .pos
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p == 200)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(multi_idxs.len(), 2, "expected exactly 2 split atoms @200");
    let atom_alt = |i: usize| -> Vec<u8> {
        let start = chunk.alt_offsets[i] as usize;
        let end = chunk.alt_offsets[i + 1] as usize;
        chunk.alt[start..end].to_vec()
    };
    let idx_c = *multi_idxs
        .iter()
        .find(|&&i| atom_alt(i) == b"C")
        .expect("an atom with ALT=C must exist");
    let idx_g = *multi_idxs
        .iter()
        .find(|&&i| atom_alt(i) == b"G")
        .expect("an atom with ALT=G must exist");
    assert_ne!(idx_c, idx_g);

    let ac = match &chunk.info_staged[0] {
        StagedColumn::Int(v) => v,
        _ => panic!("AC should stage as Int"),
    };
    assert_eq!(ac.len(), chunk.pos.len());
    assert_eq!(
        ac[idx_c], 10,
        "ALT=C (source_alt_index=1) must read AC[0]=10, not AC[1]=20"
    );
    assert_eq!(
        ac[idx_g], 20,
        "ALT=G (source_alt_index=2) must read AC[1]=20, not out-of-bounds/sentinel"
    );

    let ds = match &chunk.format_staged[0] {
        StagedColumn::Float(v) => v,
        _ => panic!("DS should stage as Float"),
    };
    assert_eq!(ds.len(), chunk.pos.len() * num_samples);
    assert!(
        (ds[idx_c * num_samples] - 0.1).abs() < 1e-6,
        "ALT=C sample0 DS should be 0.1, got {}",
        ds[idx_c * num_samples]
    );
    assert!(
        (ds[idx_c * num_samples + 1] - 0.3).abs() < 1e-6,
        "ALT=C sample1 DS should be 0.3, got {}",
        ds[idx_c * num_samples + 1]
    );
    assert!(
        (ds[idx_g * num_samples] - 0.2).abs() < 1e-6,
        "ALT=G sample0 DS should be 0.2, got {}",
        ds[idx_g * num_samples]
    );
    assert!(
        (ds[idx_g * num_samples + 1] - 0.4).abs() < 1e-6,
        "ALT=G sample1 DS should be 0.4, got {}",
        ds[idx_g * num_samples + 1]
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

    let source =
        VcfRecordSource::new(bcf_path.to_str().unwrap(), "chr1", &samples, 1, 2, &[]).unwrap();
    let mut reader = ChunkAssembler::new(
        Box::new(source),
        samples.len(),
        2,
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chr1",
        false,
        genoray_core::normalize::CheckRef::Error,
        &[],
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
// as a typed `ConversionError` (SP-3: `VcfRecordSource::new` now returns
// `Result` instead of panicking on `name2rid`, so the reader thread's closure
// propagates it and the join surfaces the real error, not a swallowed
// `WorkerPanicked`; SP-4 final review (M1) narrowed the variant from a generic
// `Input(String)` to a structurally-matchable `ContigNotInHeader`).
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
        genoray_core::orchestrator::SourceSpec::Vcf {
            vcf_path: bcf_path.to_str().unwrap().to_string(),
            htslib_threads: 1,
        },
        Some(bcf_path.with_extension("fa").to_str().unwrap()),
        "chrZ",
        out_dir.to_str().unwrap(),
        &["s0"],
        1000,
        2,
        1 << 20,
        false,
        genoray_core::normalize::CheckRef::Error,
        1,     // processing_threads
        false, // signatures
        &[],   // fields
    );

    // SP-4 final review (M1): a chromosome absent from the VCF header is a
    // distinct, structurally-typed `ContigNotInHeader` -- not a generic
    // `Input` string -- so callers (e.g. `vcf_list_reader`'s per-file
    // skip-and-hom-ref-fill branch) can match on the variant instead of
    // grepping the message.
    assert!(matches!(
        res,
        Err(genoray_core::error::ConversionError::ContigNotInHeader { .. })
    ));
}

// Boundary error-handling: a VCF/BCF path that does not exist on disk must
// surface as a typed `ConversionError::MissingFile`, symmetric with the FASTA
// missing-file check (SP-3 final review, M2).
#[test]
fn test_missing_vcf_returns_missing_file() {
    let tmp = tempdir().unwrap();
    let missing_path = tmp.path().join("does_not_exist.vcf.gz");

    let res = VcfRecordSource::new(missing_path.to_str().unwrap(), "chr1", &["s0"], 1, 2, &[]);

    assert!(matches!(
        res,
        Err(genoray_core::error::ConversionError::MissingFile { .. })
    ));
}

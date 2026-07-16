// End-to-end test for `SparseVar2.from_vcf_list`'s merge core: two
// single-sample VCFs with DISJOINT site lists, merged into ONE SVAR2 store via
// `orchestrator::run_vcf_list` (the library entrypoint the
// `_core.run_vcf_list_conversion_pipeline` pyfunction wraps).

mod common;

use common::{SynthRecord, build_bcf_with_index, build_fasta_with_index, read_u32_bin};
use genoray_core::normalize::CheckRef;
use genoray_core::orchestrator::run_vcf_list;
use genoray_core::rvk::{DecodedKey, decode_key, decode_snp_2bit, unpack_snp_keys};

use tempfile::tempdir;

#[test]
fn vcf_list_e2e_two_samples_one_store() {
    let tmp = tempdir().unwrap();

    // SA: chr1 POS3 (0-based pos 2) A>G, homozygous.
    let sa_records = vec![SynthRecord {
        pos: 2,
        ref_allele: b"A",
        alts: vec![&b"G"[..]],
        gt: vec![1, 1],
    }];
    // SB: chr1 POS7 (0-based pos 6) C>CAT, homozygous. A DISJOINT site from
    // SA's -- SB carries nothing at pos 2 and SA carries nothing at pos 6, so
    // the merged store must hom-ref-fill each sample's missing column.
    let sb_records = vec![SynthRecord {
        pos: 6,
        ref_allele: b"C",
        alts: vec![&b"CAT"[..]],
        gt: vec![1, 1],
    }];

    let a_path = tmp.path().join("a.bcf");
    let b_path = tmp.path().join("b.bcf");
    build_bcf_with_index(&a_path, "chr1", 1000, &["SA"], &sa_records);
    build_bcf_with_index(&b_path, "chr1", 1000, &["SB"], &sb_records);

    // Shared reference: both files' REF alleles stamped into one fasta so
    // `validate_ref` succeeds for both single-sample sources.
    let ref_path = tmp.path().join("ref.fa");
    let mut all_records = sa_records;
    all_records.extend(sb_records);
    build_fasta_with_index(&ref_path, "chr1", 1000, &all_records);

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let vcf_paths = vec![
        a_path.to_str().unwrap().to_string(),
        b_path.to_str().unwrap().to_string(),
    ];
    let samples = vec!["SA".to_string(), "SB".to_string()];
    let chroms = vec!["chr1".to_string()];

    let dropped = run_vcf_list(
        &vcf_paths,
        Some(ref_path.to_str().unwrap()),
        &chroms,
        out.to_str().unwrap(),
        &samples,
        25_000,
        2,
        Some(1),
        8_388_608,
        false,
        CheckRef::Error,
        false,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        genoray_core::svar2_view::OverlapMode::Pos,
    )
    .expect("run_vcf_list should succeed");

    assert_eq!(dropped, 0);

    let meta_path = out.join("meta.json");
    assert!(meta_path.exists(), "meta.json should exist");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["samples"], serde_json::json!(["SA", "SB"]));
    assert_eq!(meta["contigs"], serde_json::json!(["chr1"]));
    assert_eq!(meta["ploidy"], serde_json::json!(2));

    // The assertions above only prove `run_vcf_list` returned `Ok` with the
    // arguments it was CALLED with -- `dropped`/`meta.json` are echoed
    // straight back from the function's own inputs (`meta::write_meta` writes
    // `samples`/`contigs`/`ploidy` verbatim; see `src/meta.rs`), so they would
    // still look "correct" even if file B's records never reached the merged
    // stream at all. Read the ON-DISK per-contig artifacts instead: they can
    // only contain what the merge core actually emitted.
    //
    // Cohort layout: samples = [SA, SB], ploidy 2 -> 4 haplotype columns,
    // sample-major-then-ploidy: hap0/hap1 = SA_p0/SA_p1, hap2/hap3 =
    // SB_p0/SB_p1 (see `src/rvk.rs`'s `hap = s * ploidy + p` transpose).
    //
    // Routing (cost_model::choose_representation, np=4, x=2 carrier haps,
    // no sidecar/INFO/FORMAT): both variants are hom-alt in ONE sample (x=2
    // carrier haps out of np=4), so BOTH clear the dense crossover (SNP:
    // dense_bits=38 < var_key_bits=68; INS: dense_bits=68 < var_key_bits=128)
    // and route to `dense/snp` and `dense/indel` respectively -- neither
    // `var_key/snp` nor `var_key/indel` should exist.
    let contig_dir = out.join("chr1");
    for empty_sub in ["var_key/snp", "var_key/indel"] {
        let p = contig_dir.join(empty_sub).join("positions.bin");
        // The stream's directory/file is created unconditionally by the
        // pipeline regardless of whether any variant ever routed there (see
        // `test_e2e_mutation_conservation`'s `if p.exists()` guard for the
        // same reason) -- an empty var_key bucket is a 0-byte file, not a
        // missing one, so check length rather than existence.
        let len = if p.exists() {
            read_u32_bin(&p).len()
        } else {
            0
        };
        assert_eq!(
            len, 0,
            "{empty_sub}/positions.bin should be empty -- both variants are \
             expected to route dense, not var_key"
        );
    }

    // dense/snp: exactly 1 variant, the SA SNP at 0-based pos 2 (VCF POS 3).
    let dsnp = contig_dir.join("dense/snp");
    let dsnp_pos = read_u32_bin(&dsnp.join("positions.bin"));
    assert_eq!(
        dsnp_pos,
        vec![2u32],
        "dense/snp should hold exactly the SNP@pos2"
    );

    // Decode the packed 2-bit ALT code back to confirm it's really the 'G'
    // ALT from file A's record, not some other/default byte pattern.
    let dsnp_keys = std::fs::read(dsnp.join("alleles.bin")).unwrap();
    let dsnp_codes = unpack_snp_keys(&dsnp_keys, dsnp_pos.len());
    assert_eq!(
        decode_snp_2bit(dsnp_codes[0]),
        b'G',
        "dense/snp ALT should decode to file A's 'G'"
    );

    // dense/snp genotype bits: hap ordering is sample-major-then-ploidy, one
    // dense variant -> bit index == hap index directly. SA (haps 0,1) is
    // hom-alt for the SNP; SB (haps 2,3) never carries it in file B, so it
    // must be hom-REF (bit clear), NOT missing -- SVAR2 has no separate
    // missing encoding for a dense bit, but this is exactly the check that
    // would catch file B's disjoint site silently reusing file A's calls
    // instead of being independently hom-ref-filled.
    let dsnp_geno = std::fs::read(dsnp.join("genotypes.bin")).unwrap();
    assert!(
        genoray_core::bits::get_bit(&dsnp_geno, 0),
        "SA hap0 must carry the SNP"
    );
    assert!(
        genoray_core::bits::get_bit(&dsnp_geno, 1),
        "SA hap1 must carry the SNP"
    );
    assert!(
        !genoray_core::bits::get_bit(&dsnp_geno, 2),
        "SB hap0 must be hom-ref at SA's SNP site (SB never carries it)"
    );
    assert!(
        !genoray_core::bits::get_bit(&dsnp_geno, 3),
        "SB hap1 must be hom-ref at SA's SNP site (SB never carries it)"
    );

    // dense/indel: exactly 1 variant, the SB insertion at 0-based pos 6
    // (VCF POS 7).
    let dindel = contig_dir.join("dense/indel");
    let dindel_pos = read_u32_bin(&dindel.join("positions.bin"));
    assert_eq!(
        dindel_pos,
        vec![6u32],
        "dense/indel should hold exactly the INS@pos6"
    );

    // Decode the packed indel key to confirm it's really file B's "CAT"
    // insertion ALT, not file A's site or a default/zeroed key -- this is the
    // assertion that most directly catches "file B was silently ignored":
    // if the merge only ever saw file A, this variant (and this whole
    // dense/indel bucket) simply would not exist.
    let dindel_keys = read_u32_bin(&dindel.join("alleles.bin"));
    assert_eq!(dindel_keys.len(), 1);
    match decode_key(dindel_keys[0]) {
        DecodedKey::Inline { alt } => assert_eq!(alt, b"CAT".to_vec()),
        other => panic!("expected inline INS CAT for dense/indel, got {:?}", other),
    }

    // dense/indel genotype bits: SB (haps 2,3) is hom-alt for the insertion;
    // SA (haps 0,1) never carries it in file A, so it must be hom-REF, not
    // missing. Combined with the dense/snp check above, this proves BOTH
    // input files' per-sample columns landed correctly in the merged store.
    let dindel_geno = std::fs::read(dindel.join("genotypes.bin")).unwrap();
    assert!(
        !genoray_core::bits::get_bit(&dindel_geno, 0),
        "SA hap0 must be hom-ref at SB's INS site (SA never carries it)"
    );
    assert!(
        !genoray_core::bits::get_bit(&dindel_geno, 1),
        "SA hap1 must be hom-ref at SB's INS site (SA never carries it)"
    );
    assert!(
        genoray_core::bits::get_bit(&dindel_geno, 2),
        "SB hap0 must carry the INS"
    );
    assert!(
        genoray_core::bits::get_bit(&dindel_geno, 3),
        "SB hap1 must carry the INS"
    );
}

/// `regions`/`overlap` must restrict the k-way merge to in-region variants,
/// mirroring the single-VCF `SourceSpec::Vcf` path
/// (`test_e2e_normalized_bcf_pipeline` in `tests/test_e2e.rs`) but exercised
/// through `run_vcf_list`'s `SourceSpec::VcfList` delegation instead. Two
/// files, disjoint sites (same layout as
/// `vcf_list_e2e_two_samples_one_store` above): SA's SNP at 0-based pos 2 is
/// IN the requested region `[0, 5)`, SB's insertion at 0-based pos 6 is OUT --
/// only the SNP should survive.
#[test]
fn vcf_list_e2e_regions_restricts_merge() {
    let tmp = tempdir().unwrap();

    let sa_records = vec![SynthRecord {
        pos: 2,
        ref_allele: b"A",
        alts: vec![&b"G"[..]],
        gt: vec![1, 1],
    }];
    let sb_records = vec![SynthRecord {
        pos: 6,
        ref_allele: b"C",
        alts: vec![&b"CAT"[..]],
        gt: vec![1, 1],
    }];

    let a_path = tmp.path().join("a.bcf");
    let b_path = tmp.path().join("b.bcf");
    build_bcf_with_index(&a_path, "chr1", 1000, &["SA"], &sa_records);
    build_bcf_with_index(&b_path, "chr1", 1000, &["SB"], &sb_records);

    let ref_path = tmp.path().join("ref.fa");
    let mut all_records = sa_records;
    all_records.extend(sb_records);
    build_fasta_with_index(&ref_path, "chr1", 1000, &all_records);

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let vcf_paths = vec![
        a_path.to_str().unwrap().to_string(),
        b_path.to_str().unwrap().to_string(),
    ];
    let samples = vec!["SA".to_string(), "SB".to_string()];
    let chroms = vec!["chr1".to_string()];

    let dropped = run_vcf_list(
        &vcf_paths,
        Some(ref_path.to_str().unwrap()),
        &chroms,
        out.to_str().unwrap(),
        &samples,
        25_000,
        2,
        Some(1),
        8_388_608,
        false,
        CheckRef::Error,
        false,
        Vec::new(),
        Vec::new(),
        vec![("chr1".to_string(), 0u32, 5u32)],
        genoray_core::svar2_view::OverlapMode::Pos,
    )
    .expect("run_vcf_list should succeed");

    // Region filtering happens per-record inside `VcfRecordSource`, upstream
    // of atomization -- an out-of-region record never reaches
    // `atomize_to_vcf_biallelic`, so it contributes 0 to the out-of-scope
    // drop count (that count is for symbolic/breakend ALTs, not region
    // exclusions).
    assert_eq!(dropped, 0);

    let contig_dir = out.join("chr1");

    // dense/snp: SA's in-region SNP@pos2 must be present.
    let dsnp = contig_dir.join("dense/snp");
    let dsnp_pos = read_u32_bin(&dsnp.join("positions.bin"));
    assert_eq!(
        dsnp_pos,
        vec![2u32],
        "dense/snp should hold the in-region SNP@pos2"
    );

    // dense/indel: SB's out-of-region insertion@pos6 must be ABSENT -- the
    // directory is still created unconditionally (see the sibling test's
    // comment), so check the file is empty, not merely that it exists.
    let dindel = contig_dir.join("dense/indel");
    let p = dindel.join("positions.bin");
    let len = if p.exists() {
        read_u32_bin(&p).len()
    } else {
        0
    };
    assert_eq!(
        len, 0,
        "dense/indel should be empty -- SB's INS@pos6 lies outside chr1:0-5"
    );
}

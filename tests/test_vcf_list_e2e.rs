// End-to-end test for `SparseVar2.from_vcf_list`'s merge core: two
// single-sample VCFs with DISJOINT site lists, merged into ONE SVAR2 store via
// `orchestrator::run_vcf_list` (the library entrypoint the
// `_core.run_vcf_list_conversion_pipeline` pyfunction wraps).

mod common;

use common::{SynthRecord, build_bcf_with_index, build_fasta_with_index};
use genoray_core::orchestrator::run_vcf_list;

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
        false,
        Vec::new(),
        Vec::new(),
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
}

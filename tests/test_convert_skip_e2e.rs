mod common;

use common::{SynthRecord, build_bcf_with_index, build_fasta_with_index};
use genoray_core::orchestrator::process_chromosome;
use tempfile::TempDir;

// Two SNP records plus one symbolic <DEL> record on chr1.
fn records() -> Vec<SynthRecord<'static>> {
    vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"G",
            alts: vec![&b"<DEL>"[..]],
            gt: vec![1, 1, 0, 0],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"T",
            alts: vec![&b"A"[..]],
            gt: vec![0, 0, 1, 0],
        },
    ]
}

fn convert(
    out: &std::path::Path,
    fasta: Option<&str>,
    skip: bool,
) -> Result<u64, genoray_core::error::ConversionError> {
    let tmp = out.parent().unwrap();
    let bcf = tmp.join("in.bcf");
    let samples = ["S0", "S1"];
    let recs = records();
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &recs);
    build_fasta_with_index(&bcf.with_extension("fa"), "chr1", 1000, &recs);
    let sample_refs: Vec<&str> = samples.to_vec();
    process_chromosome(
        genoray_core::orchestrator::SourceSpec::Vcf {
            vcf_path: bcf.to_str().unwrap().to_string(),
            htslib_threads: 1,
            regions: Vec::new(),
        },
        fasta,
        "chr1",
        out.to_str().unwrap(),
        &sample_refs,
        25_000,
        2,
        8 * 1024 * 1024,
        skip,
        genoray_core::normalize::CheckRef::Error,
        1,     // processing_threads
        false, // signatures
        &[],   // fields
    )
}

#[test]
fn symbolic_record_errors_by_default() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    let fasta = tmp.path().join("in.fa");
    let res = convert(&out, Some(fasta.to_str().unwrap()), false);
    assert!(
        res.is_err(),
        "default (skip=false) must fail on a symbolic record"
    );
}

#[test]
fn symbolic_record_skipped_and_counted() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    let fasta = tmp.path().join("in.fa");
    let dropped = convert(&out, Some(fasta.to_str().unwrap()), true).unwrap();
    assert_eq!(dropped, 1, "the single <DEL> ALT should be dropped");
    assert!(
        out.join("chr1/var_key/snp/positions.bin").exists(),
        "SNPs still converted"
    );
}

#[test]
fn converts_without_a_reference() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    // No FASTA passed; skip the symbolic record so only SNPs remain.
    let dropped = convert(&out, None, true).unwrap();
    assert_eq!(dropped, 1);
    assert!(out.join("chr1/var_key/snp/positions.bin").exists());
}

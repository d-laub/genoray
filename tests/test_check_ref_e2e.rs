mod common;

use common::{SynthRecord, build_bcf_with_index, build_fasta_with_index};
use genoray_core::normalize::CheckRef;
use genoray_core::orchestrator::{SourceSpec, process_chromosome};
use tempfile::TempDir;

// Three records; the middle one's REF ("G") will disagree with the FASTA
// (which carries "T" at that position), reproducing issue #116.
fn bcf_records() -> Vec<SynthRecord<'static>> {
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
            alts: vec![&b"GTTT"[..]],
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

// Same loci, but the FASTA truth at pos 200 is "T" (not the record's "G").
fn fasta_records() -> Vec<SynthRecord<'static>> {
    vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![],
            gt: vec![],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"T",
            alts: vec![],
            gt: vec![],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"T",
            alts: vec![],
            gt: vec![],
        },
    ]
}

fn convert(
    out: &std::path::Path,
    check_ref: CheckRef,
) -> Result<u64, genoray_core::error::ConversionError> {
    let tmp = out.parent().unwrap();
    let bcf = tmp.join("in.bcf");
    let fasta = tmp.join("in.fa");
    let samples = ["S0", "S1"];
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &bcf_records());
    build_fasta_with_index(&fasta, "chr1", 1000, &fasta_records());
    let sample_refs: Vec<&str> = samples.to_vec();
    process_chromosome(
        SourceSpec::Vcf {
            vcf_path: bcf.to_str().unwrap().to_string(),
            htslib_threads: 1,
        },
        Some(fasta.to_str().unwrap()),
        "chr1",
        out.to_str().unwrap(),
        &sample_refs,
        25_000,
        2,
        8 * 1024 * 1024,
        false,     // skip_out_of_scope
        check_ref, // NEW
        1,         // processing_threads
        false,     // signatures
        &[],       // fields
    )
}

#[test]
fn ref_mismatch_errors_under_e() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    assert!(
        convert(&out, CheckRef::Error).is_err(),
        "check_ref=e must abort on a REF mismatch"
    );
}

#[test]
fn ref_mismatch_excluded_under_x() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    // Succeeds; the two clean SNPs (pos 100, 300) are still written.
    convert(&out, CheckRef::Exclude).unwrap();
    assert!(
        out.join("chr1/var_key/snp/positions.bin").exists(),
        "clean SNPs still converted"
    );
}

use common::build_bcf_with_index as build_single;

// Two single-sample files; file B carries a REF at pos 200 that disagrees with
// the FASTA (which says "T"). Under x, B's bad record is dropped; the merge
// still produces a store.
#[test]
fn vcf_list_ref_mismatch_excluded_under_x() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    let fasta = tmp.path().join("in.fa");
    build_fasta_with_index(&fasta, "chr1", 1000, &fasta_records());

    let a = tmp.path().join("a.bcf");
    let b = tmp.path().join("b.bcf");
    build_single(
        &a,
        "chr1",
        1000,
        &["A"],
        &[SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0],
        }],
    );
    build_single(
        &b,
        "chr1",
        1000,
        &["B"],
        &[SynthRecord {
            pos: 200,
            ref_allele: b"G",
            alts: vec![&b"GTTT"[..]],
            gt: vec![1, 0],
        }],
    );

    let dropped = genoray_core::orchestrator::run_vcf_list(
        &[
            a.to_str().unwrap().to_string(),
            b.to_str().unwrap().to_string(),
        ],
        Some(fasta.to_str().unwrap()),
        &["chr1".to_string()],
        out.to_str().unwrap(),
        &["A".to_string(), "B".to_string()],
        25_000,
        2,
        Some(1),
        8 * 1024 * 1024,
        false,             // skip_out_of_scope
        CheckRef::Exclude, // NEW
        false,             // signatures
        Vec::new(),        // info_fields
        Vec::new(),        // format_fields
    )
    .unwrap();
    assert_eq!(
        dropped, 0,
        "no out-of-scope ALTs; the REF-mismatch is excluded, not counted here"
    );
    assert!(
        out.join("meta.json").exists(),
        "merge completed despite the bad record"
    );
}

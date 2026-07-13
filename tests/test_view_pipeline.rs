// Disk-integration test for `run_view_pipeline` + `SourceSpec::Svar2`: builds a
// small 3-sample SVAR2 store via the real conversion pipeline (VCF -> svar2,
// same as `build_contig` uses everywhere else), runs `run_view_pipeline` over a
// region + sample subset, reopens the output with `ContigReader`, and checks
// the decoded per-hap calls against a direct `overlap_sample` query of the
// SOURCE store restricted to the same region/samples.
//
// Fixture (chr1, samples S0/S1/S2, ploidy 2):
//   pos=10  SNP A->C   gt: S0/hap0, S2/hap1          (inside the view region)
//   pos=20  INS A->AGG gt: S1/hap0, S1/hap1           (S1-only -> MAC=0 once S1
//                                                       is dropped from the subset)
//   pos=30  SNP A->G   gt: S0/hap1, S2/hap0          (inside the view region)
//   pos=500 SNP A->T   gt: S0/hap0, S0/hap1          (outside the view region [0,100))
//
// View: samples [S0, S2] (drops S1), region chr1:[0, 100), regions_overlap="pos".
// Expected surviving calls: pos 10 and pos 30 only (pos 20 MAC=0, pos 500 out of
// region). Every variant here is a SNP inside the region interior (no boundary
// deletions), so `regions_overlap="pos"` and `overlap_sample`'s extent-overlap
// semantics agree -- `overlap_sample` on the SOURCE (restricted to the same
// [0, 100) window and the same original sample columns) is a valid independent
// oracle for what the view should contain.

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::meta::{FORMAT_VERSION, write_meta};
use genoray_core::query::ContigReader;
use genoray_core::query::oracle::overlap_sample;
use genoray_core::run_view_pipeline;
use pyo3::Python;
use tempfile::tempdir;

#[test]
fn view_region_sample_subset_matches_source() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();

    let samples = ["S0", "S1", "S2"];
    let records = vec![
        SynthRecord {
            pos: 10,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0, 0, 1], // S0/hap0, S2/hap1
        },
        SynthRecord {
            pos: 20,
            ref_allele: b"A",
            alts: vec![&b"AGG"[..]],
            gt: vec![0, 0, 1, 1, 0, 0], // S1 only
        },
        SynthRecord {
            pos: 30,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![0, 1, 0, 0, 1, 0], // S0/hap1, S2/hap0
        },
        SynthRecord {
            pos: 500,
            ref_allele: b"A",
            alts: vec![&b"T"[..]],
            gt: vec![1, 1, 0, 0, 0, 0], // S0, both haps -- outside [0, 100)
        },
    ];
    build_contig(&src, "chr1", &samples, 2, &records);
    write_meta(
        &src,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();

    let out = tmp.path().join("view");

    Python::attach(|py| {
        run_view_pipeline(
            py,
            src.to_str().unwrap().to_string(),
            out.to_str().unwrap().to_string(),
            vec!["chr1".to_string()],
            vec!["S0".to_string(), "S2".to_string()],
            vec![("chr1".to_string(), 0u32, 100u32)],
            "pos".to_string(),
            false,
            Vec::new(),
            None,
            Some(1),
            false,
        )
    })
    .expect("run_view_pipeline should succeed");

    // Reopen the output store: 2 samples (S0, S2), ploidy 2.
    let out_reader =
        ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).expect("output contig should open");

    // Independent oracle: query the SOURCE directly for S0 (orig idx 0) and S2
    // (orig idx 2), restricted to the same [0, 100) window.
    let src_reader =
        ContigReader::open(src.to_str().unwrap(), "chr1", 3, 2).expect("source contig should open");
    let expect_s0 = overlap_sample(&src_reader, 0, 0, 100);
    let expect_s2 = overlap_sample(&src_reader, 2, 0, 100);

    let got_s0 = overlap_sample(&out_reader, 0, 0, 1_000); // output sample 0 == S0
    let got_s2 = overlap_sample(&out_reader, 1, 0, 1_000); // output sample 1 == S2

    assert_eq!(
        got_s0, expect_s0,
        "S0's calls in the view must match the source"
    );
    assert_eq!(
        got_s2, expect_s2,
        "S2's calls in the view must match the source"
    );

    // Make the oracle comparison concrete, not just "equal to itself":
    // hap0 (S0/p0): SNP@10 only (pos 20 dropped -- MAC=0 once S1 leaves the
    // subset; pos 500 dropped -- outside the region).
    assert_eq!(got_s0.per_hap[0].positions, vec![10]);
    assert_eq!(got_s0.per_hap[0].ilens, vec![0]);
    assert_eq!(got_s0.per_hap[0].alts, vec![b"C".to_vec()]);
    // hap1 (S0/p1): SNP@30 only.
    assert_eq!(got_s0.per_hap[1].positions, vec![30]);
    assert_eq!(got_s0.per_hap[1].ilens, vec![0]);
    assert_eq!(got_s0.per_hap[1].alts, vec![b"G".to_vec()]);

    // hap0 (S2/p0): SNP@30 only.
    assert_eq!(got_s2.per_hap[0].positions, vec![30]);
    assert_eq!(got_s2.per_hap[0].alts, vec![b"G".to_vec()]);
    // hap1 (S2/p1): SNP@10 only.
    assert_eq!(got_s2.per_hap[1].positions, vec![10]);
    assert_eq!(got_s2.per_hap[1].alts, vec![b"C".to_vec()]);

    // The output store's meta.json must record the SUBSET samples and the
    // contigs actually written.
    let meta_text = std::fs::read_to_string(out.join("meta.json")).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&meta_text).unwrap();
    assert_eq!(meta["samples"], serde_json::json!(["S0", "S2"]));
    assert_eq!(meta["contigs"], serde_json::json!(["chr1"]));
    assert_eq!(meta["ploidy"], 2);
}

#[test]
fn non_empty_fields_fail_fast() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();

    let samples = ["S0"];
    let records = vec![SynthRecord {
        pos: 10,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt: vec![1, 0],
    }];
    build_contig(&src, "chr1", &samples, 2, &records);
    write_meta(
        &src,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();

    let out = tmp.path().join("view");
    let err = Python::attach(|py| {
        run_view_pipeline(
            py,
            src.to_str().unwrap().to_string(),
            out.to_str().unwrap().to_string(),
            vec!["chr1".to_string()],
            vec!["S0".to_string()],
            vec![("chr1".to_string(), 0u32, 100u32)],
            "pos".to_string(),
            false,
            vec!["DP".to_string()],
            None,
            Some(1),
            false,
        )
    })
    .unwrap_err();
    Python::attach(|py| {
        assert!(
            err.is_instance_of::<pyo3::exceptions::PyValueError>(py),
            "expected ValueError, got {err:?}"
        );
    });
    assert!(!out.exists(), "must fail before creating the output dir");
}

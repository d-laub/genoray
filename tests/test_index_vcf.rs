mod common;

use common::{SynthRecord, build_bcf_with_index};
use tempfile::TempDir;

#[test]
fn build_csi_makes_a_usable_index() {
    let tmp = TempDir::new().unwrap();
    let bcf = tmp.path().join("in.bcf");
    let samples = ["S0"];
    let recs = vec![SynthRecord {
        pos: 10,
        ref_allele: b"A",
        alts: vec![b"C"],
        gt: vec![1, 0],
    }];
    // build_bcf_with_index writes an index too; delete it so we can rebuild.
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &recs);
    let csi = tmp.path().join("in.bcf.csi");
    std::fs::remove_file(&csi).ok();
    assert!(!csi.exists());

    genoray_core::index_bcf_csi(bcf.to_str().unwrap()).expect("index build");
    assert!(
        csi.exists(),
        "a .csi index should be written next to the BCF"
    );
}

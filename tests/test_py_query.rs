//! Boundary test for the M6a PyO3 seam: `PyContigReader` opens a finished contig
//! built through the real conversion pipeline, and tolerates an empty contig dir
//! (mirroring `ContigReader::open`'s missing-sub-stream contract).

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::py_query::PyContigReader;
use pyo3::Python;
use tempfile::tempdir;

#[test]
fn test_py_contig_reader_opens_built_contig() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt: vec![1, 0, 1, 1],
    }];
    build_contig(&out, "chr1", &samples, 2, &records);

    Python::attach(|_py| {
        let r = PyContigReader::new(out.to_str().unwrap(), "chr1", 2, 2);
        assert!(r.is_ok(), "PyContigReader should open a built contig");
    });
}

#[test]
fn test_py_contig_reader_empty_dir_tolerated() {
    let tmp = tempdir().unwrap();
    Python::attach(|_py| {
        // No contig dir / sub-streams: opens as all-empty, no error.
        let r = PyContigReader::new(tmp.path().to_str().unwrap(), "chrX", 1, 2);
        assert!(r.is_ok(), "empty contig should still open");
    });
}

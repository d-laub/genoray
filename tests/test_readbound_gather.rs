//! Read-bound per-class gather: find_ranges emits per-class dense ranges and
//! gather_ranges_readbound replays them into BatchResultSplit without building
//! the contig-wide DenseUnion.
mod common;

use common::{SynthRecord, build_contig};
// TODO(Task 3): restore `gather_ranges, gather_ranges_readbound, overlap_batch`
// (and `use genoray_core::search;`) once gather_ranges_readbound exists and
// Task 4 extends this file with tests that use them. Left out for now to
// avoid unused-import warnings on an isolated Task 2 build.
use genoray_core::query::{ContigReader, find_ranges};
use tempfile::tempdir;

fn synth_reader(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![0, 1, 1, 1],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 0, 1],
        },
    ];
    build_contig(out, "chr1", &samples, 2, &records);
    ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap()
}

#[test]
fn test_find_ranges_emits_per_class_dense_ranges() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let rb = find_ranges(&reader, &regions, None);
    // Both per-class range vectors are per-region (dense is cohort-shared).
    assert_eq!(rb.dense_snp_range.len(), regions.len());
    assert_eq!(rb.dense_indel_range.len(), regions.len());
    // Each per-class window is a subset of that class's table; ranges are valid.
    for &(s, e) in rb.dense_snp_range.iter().chain(rb.dense_indel_range.iter()) {
        assert!(s <= e);
    }
    // Region 0 spans the whole contig: it must see the one dense SNP (pos 100 is
    // var_key here, but the SNP class table is nonempty iff any SNP is dense) and
    // the dense indels. The union window must equal snp∪indel counts.
    let (us0, ue0) = rb.dense_range[0];
    let snp0 = rb.dense_snp_range[0].1 - rb.dense_snp_range[0].0;
    let indel0 = rb.dense_indel_range[0].1 - rb.dense_indel_range[0].0;
    assert_eq!(
        ue0 - us0,
        snp0 + indel0,
        "union window size must equal sum of per-class window sizes"
    );
}

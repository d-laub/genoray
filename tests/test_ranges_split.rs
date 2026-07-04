//! SVAR2 search/gather split: find_ranges produces the index ranges that
//! gather_ranges replays into the same BatchResult overlap_batch returns.

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::query::{ContigReader, find_ranges, gather_ranges, overlap_batch, read_ranges};
use genoray_core::search;
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
fn test_find_ranges_dense_range_matches_overlap_batch() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let br = overlap_batch(&reader, &regions);
    let rb = find_ranges(&reader, &regions, None);

    // Same per-region dense index ranges; H+1 vk_off implies R*H vk sub-ranges.
    assert_eq!(rb.dense_range, br.dense_range);
    assert_eq!(rb.n_regions, br.n_regions);
    assert_eq!(rb.n_samples, br.n_samples);
    assert_eq!(rb.ploidy, br.ploidy);
    assert_eq!(
        rb.vk_snp_range.len(),
        regions.len() * br.n_samples * br.ploidy
    );
    assert_eq!(
        rb.vk_indel_range.len(),
        regions.len() * br.n_samples * br.ploidy
    );
    assert_eq!(rb.region_starts, vec![0u32, 250u32]);
}

#[test]
fn test_gather_ranges_reproduces_overlap_batch_field_for_field() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let oracle = overlap_batch(&reader, &regions);
    let got = gather_ranges(&reader, &find_ranges(&reader, &regions, None));

    assert_eq!(got.n_regions, oracle.n_regions);
    assert_eq!(got.n_samples, oracle.n_samples);
    assert_eq!(got.ploidy, oracle.ploidy);
    assert_eq!(got.vk, oracle.vk);
    assert_eq!(got.vk_off, oracle.vk_off);
    assert_eq!(got.dense, oracle.dense);
    assert_eq!(got.dense_range, oracle.dense_range);
    assert_eq!(got.dense_present, oracle.dense_present);
    assert_eq!(got.dense_present_off, oracle.dense_present_off);
}

#[test]
fn test_read_ranges_equals_overlap_batch() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let oracle = overlap_batch(&reader, &regions);
    let got = read_ranges(&reader, &regions, None);
    assert_eq!(got.vk, oracle.vk);
    assert_eq!(got.vk_off, oracle.vk_off);
    assert_eq!(got.dense_present, oracle.dense_present);
    assert_eq!(got.dense_present_off, oracle.dense_present_off);
    assert_eq!(got.dense_range, oracle.dense_range);
}

// Subset parity: read_ranges over a sample subset equals the corresponding
// hap-rows of the full overlap_batch. For samples=[1] (original index 1),
// region r's hap rows are r*H + [ploidy .. 2*ploidy) of the full result.
#[test]
fn test_read_ranges_sample_subset_matches_full() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 400u32)];

    let full = overlap_batch(&reader, &regions);
    let sub = read_ranges(&reader, &regions, Some(&[1]));
    assert_eq!(sub.n_samples, 1);
    // hap rows for sample 1 in the full result: h in [1*ploidy, 2*ploidy).
    let ploidy = full.ploidy;
    for p in 0..ploidy {
        let full_h = ploidy + p;
        let sub_h = p;
        assert_eq!(
            &sub.vk[sub.vk_off[sub_h]..sub.vk_off[sub_h + 1]],
            &full.vk[full.vk_off[full_h]..full.vk_off[full_h + 1]],
        );
    }
}

#[test]
fn test_gather_ranges_builds_no_search_tree() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let rb = find_ranges(&reader, &regions, None);
    let before = search::search_tree_build_count();
    let _ = gather_ranges(&reader, &rb);
    assert_eq!(
        search::search_tree_build_count(),
        before,
        "gather_ranges must build zero SearchTrees"
    );
    // positive control: find_ranges DID build trees
    let b2 = search::search_tree_build_count();
    let _ = find_ranges(&reader, &regions, None);
    assert!(
        search::search_tree_build_count() > b2,
        "find_ranges should build trees"
    );
}

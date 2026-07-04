//! Read-bound per-class gather: find_ranges emits per-class dense ranges and
//! gather_ranges_readbound replays them into BatchResultSplit without building
//! the contig-wide DenseUnion.
mod common;

use common::{SynthRecord, build_contig};
use genoray_core::query::{
    BatchResultSplit, ContigReader, HapCalls, decode_keyref_pub, find_ranges,
    gather_haps_readbound, gather_ranges_readbound, overlap_batch,
};
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

/// Merge vk ⋈ dense_snp ⋈ dense_indel for one hap and decode — the gvl-side
/// reconstruction, expressed as a test oracle.
fn readbound_decode_hap(
    br: &BatchResultSplit,
    _reader: &ContigReader,
    r: usize,
    s: usize,
    p: usize,
) -> Vec<(u32, i32)> {
    use genoray_core::query::KeyRef;
    let h = (r * br.n_samples + s) * br.ploidy + p;
    let mut merged: Vec<KeyRef> = br.vk[br.vk_off[h]..br.vk_off[h + 1]].to_vec();

    let (ss, se) = br.dense_snp_range[r];
    let bit0 = br.dense_snp_present_off[h];
    for (k, j) in (ss..se).enumerate() {
        if genoray_core::bits_get_bit(&br.dense_snp_present, bit0 + k) {
            merged.push(br.dense_snp[j]);
        }
    }
    let (is_, ie_) = br.dense_indel_range[r];
    let bit0 = br.dense_indel_present_off[h];
    for (k, j) in (is_..ie_).enumerate() {
        if genoray_core::bits_get_bit(&br.dense_indel_present, bit0 + k) {
            merged.push(br.dense_indel[j]);
        }
    }
    // Stable position sort (var_key already ahead of dense within its own run).
    merged.sort_by_key(|kr| kr.position);
    merged
        .into_iter()
        .map(|kr| (kr.position, kr.key as i32))
        .collect()
}

#[test]
fn test_readbound_reconstructs_union_per_hap() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32), (150u32, 250u32)];

    let oracle = overlap_batch(&reader, &regions);
    let rb = find_ranges(&reader, &regions, None);
    let got = gather_ranges_readbound(&reader, &rb);

    assert_eq!(got.n_regions, oracle.n_regions);
    assert_eq!(got.n_samples, oracle.n_samples);
    assert_eq!(got.ploidy, oracle.ploidy);

    for r in 0..oracle.n_regions {
        for s in 0..oracle.n_samples {
            for p in 0..oracle.ploidy {
                // Oracle: decode via the shipped union decode_hap, keep (pos, key).
                let hc: HapCalls = oracle.decode_hap(&reader, r, s, p);
                // decode_hap returns decoded alts, not raw keys — compare on the
                // (position, ilen) projection that survives decode instead.
                let want: Vec<(u32, i32)> = hc
                    .positions
                    .iter()
                    .zip(hc.ilens.iter())
                    .map(|(&a, &b)| (a, b))
                    .collect();
                let got_keys = readbound_decode_hap(&got, &reader, r, s, p);
                // Decode the read-bound raw keys the same way to get ilens.
                let got_dec: Vec<(u32, i32)> = got_keys
                    .iter()
                    .map(|&(pos, key)| (pos, decode_keyref_pub(pos, key as u32, &reader)))
                    .collect();
                assert_eq!(got_dec, want, "hap (r={r}, s={s}, p={p})");
            }
        }
    }
}

#[test]
fn test_readbound_gather_builds_no_search_tree() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let rb = find_ranges(&reader, &regions, None);
    let before = search::search_tree_build_count();
    let _ = gather_ranges_readbound(&reader, &rb);
    assert_eq!(
        search::search_tree_build_count(),
        before,
        "gather_ranges_readbound must build zero SearchTrees (no dense_union)"
    );
}

#[test]
fn test_readbound_subset_matches_full_selected_haps() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out); // 2 samples, ploidy 2
    let regions = vec![(0u32, 1_000_000u32)];

    let full = gather_ranges_readbound(&reader, &find_ranges(&reader, &regions, None));
    // Select only sample 1.
    let sub = gather_ranges_readbound(&reader, &find_ranges(&reader, &regions, Some(&[1])));
    assert_eq!(sub.n_samples, 1);
    for p in 0..reader_ploidy(&reader) {
        let a = readbound_decode_hap(&sub, &reader, 0, 0, p); // selected slot 0 == orig sample 1
        let b = readbound_decode_hap(&full, &reader, 0, 1, p); // orig sample 1
        assert_eq!(a, b, "subset ploid {p}");
    }
}

fn reader_ploidy(_r: &ContigReader) -> usize {
    2
}

#[test]
fn test_flat_gather_matches_cartesian_full_cohort() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out); // 2 samples, ploidy 2
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];
    let ploidy = 2usize;

    let rb = find_ranges(&reader, &regions, None);
    let cart = gather_ranges_readbound(&reader, &rb);

    // Enumerate flat queries in the SAME order cart lays out haps:
    // region-major, samples 0..S, so query q = r*S + s, orig sample = s.
    let s_n = rb.n_samples;
    let mut region_starts = Vec::new();
    let mut orig_samples = Vec::new();
    let mut vk_snp_range = Vec::new();
    let mut vk_indel_range = Vec::new();
    let mut dsr = Vec::new();
    let mut dir_ = Vec::new();
    for r in 0..regions.len() {
        for s in 0..s_n {
            region_starts.push(rb.region_starts[r]);
            orig_samples.push(rb.sample_cols[s]);
            dsr.push(rb.dense_snp_range[r]);
            dir_.push(rb.dense_indel_range[r]);
            for p in 0..ploidy {
                let row = r * (s_n * ploidy) + s * ploidy + p;
                vk_snp_range.push(rb.vk_snp_range[row]);
                vk_indel_range.push(rb.vk_indel_range[row]);
            }
        }
    }
    let flat = gather_haps_readbound(
        &reader,
        &region_starts,
        &orig_samples,
        &vk_snp_range,
        &vk_indel_range,
        &dsr,
        &dir_,
        ploidy,
    );

    // Compare decoded per-hap. cart hap (r,s,p) == flat query q=r*S+s, ploid p.
    for r in 0..regions.len() {
        for s in 0..s_n {
            for p in 0..ploidy {
                let a = readbound_decode_hap(&cart, &reader, r, s, p);
                let b = readbound_decode_hap(&flat, &reader, r * s_n + s, 0, p);
                assert_eq!(a, b, "flat vs cartesian (r={r}, s={s}, p={p})");
            }
        }
    }
}

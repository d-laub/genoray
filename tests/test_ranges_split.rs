//! SVAR2 search/gather split: find_ranges produces the index ranges that
//! gather_ranges replays into the same BatchResult overlap_batch returns.

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::py_query::PyContigReader;
use genoray_core::query::{
    ContigReader, MAX_END_SHIFT, PAR_COLUMN_THRESHOLD, dense_max_end_keys, find_ranges,
    find_ranges_haps, gather_ranges, overlap_batch, read_ranges,
};
use genoray_core::search;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use tempfile::tempdir;

fn i32_slice<'py>(d: &Bound<'py, PyDict>, k: &str) -> Vec<i32> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.cast::<PyArray1<i32>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}
fn i64_slice<'py>(d: &Bound<'py, PyDict>, k: &str) -> Vec<i64> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.cast::<PyArray1<i64>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}
fn u8_slice<'py>(d: &Bound<'py, PyDict>, k: &str) -> Vec<u8> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.cast::<PyArray1<u8>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}
fn i32_2d_flat<'py>(d: &Bound<'py, PyDict>, k: &str) -> Vec<i32> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.cast::<PyArray2<i32>>().unwrap().readonly();
    arr.as_array().iter().copied().collect()
}

fn assert_payload_dicts_eq<'py>(a: &Bound<'py, PyDict>, b: &Bound<'py, PyDict>) {
    for k in ["vk_pos", "vk_key", "dense_pos", "dense_key"] {
        assert_eq!(i32_slice(a, k), i32_slice(b, k), "key {k}");
    }
    for k in ["vk_off", "dense_present_off"] {
        assert_eq!(i64_slice(a, k), i64_slice(b, k), "key {k}");
    }
    assert_eq!(
        u8_slice(a, "dense_present"),
        u8_slice(b, "dense_present"),
        "key dense_present"
    );
    assert_eq!(
        i32_2d_flat(a, "dense_range"),
        i32_2d_flat(b, "dense_range"),
        "key dense_range"
    );
}

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

/// Like `synth_reader`, but with three extra single-carrier SNPs (positions
/// 10, 20, 30 — each carried by exactly one distinct hap) so that all 4
/// `vk_snp` columns are non-empty instead of just column 0.
///
/// `synth_reader` alone can't distinguish the region-outer bug from the fix:
/// with only column 0 populated, the old code's var_key tree-build growth is
/// 1/region, which happens to exactly match this test's dense-tree allowance
/// (a tie passes `<=`). Widening carrier coverage across all columns while
/// keeping each new record single-carrier (so the cost model still routes it
/// `VarKey`, not `Dense` — 3 carriers flips the routing, see
/// `cost_model::choose_representation`) makes the var_key term actually show
/// up in the old code's growth.
///
/// Positions 10/20/30 are deliberately below 100/200/300 and outside
/// `[90, 110)` / `[900, 950)`, so a reader who later reuses this fixture
/// won't disturb `synth_reader`'s own max-end / empty-region expectations
/// (Task 2 depends on those). Used ONLY by
/// `test_find_ranges_tree_builds_do_not_scale_with_regions` — do not extend
/// `synth_reader` itself for this purpose, it's shared by other tests here.
fn synth_reader_wide(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 10,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![0, 1, 0, 0],
        },
        SynthRecord {
            pos: 20,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![0, 0, 1, 0],
        },
        SynthRecord {
            pos: 30,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![0, 0, 0, 1],
        },
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

/// Built specifically to distinguish the SVAR1 tie-break rule (max by
/// POSITION first, end second) from a buggy "max by absolute end" rule that
/// `test_max_end_keys_pick_highest_position_variant` cannot catch — on
/// `synth_reader`, position and end rise together across all three variants
/// (100/101, 200/201, 300/302), so both rules agree there.
///
/// Here a DEL@200 with a long deletion (ext = 1 + 150 = 151, end = 351) sits
/// at a LOWER position than a SNP@300 (ext = 1, end = 301) whose own end is
/// smaller. The correct packed-key ordering `(pos << SHIFT) | ext` picks the
/// SNP (higher position) — unpacked end 301; a "max by absolute end" bug
/// would instead return 351 (the DEL's larger end, at a lower position).
///
/// The DEL's ref allele is 151 'A's with the LAST byte forced to 'T'
/// (`del_ref[150] = b'T'`), not a pure homopolymer: `normalize::left_align`
/// rolls a deletion left while `ref_seq[pos] == ref_seq[pos + ndel]`, and a
/// pure 151 x 'A' run trivially satisfies that against itself (position 350,
/// the deleted span's last base, is also 'A'), silently shifting the variant
/// to pos 199 instead of the intended 200. Breaking the last byte makes
/// `ref_seq[200] != ref_seq[350]`, so the deletion stays put. (Found by this
/// test failing with an off-by-one before the fix — see the fix commit.)
///
/// Each variant is single-carrier (`x_calls = 1`), so per `cost_model`'s
/// `np = n_samples * ploidy = 4`: indel dense_bits = 68 > var_key_bits = 64,
/// SNP dense_bits = 38 > var_key_bits = 34 — both stay `VarKey`, so this
/// fixture only exercises `find_ranges_haps`, like `synth_reader`.
///
/// Used ONLY by the tie-break test below — do not extend `synth_reader`
/// itself for this purpose.
fn synth_reader_tiebreak(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let mut del_ref = vec![b'A'; 151];
    del_ref[150] = b'T'; // break the homopolymer so left-align doesn't roll it
    let records = vec![
        SynthRecord {
            pos: 200,
            ref_allele: &del_ref,
            alts: vec![&b"A"[..]],
            gt: vec![1, 0, 0, 0],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![0, 1, 0, 0],
        },
    ];
    build_contig(out, "chr1", &samples, 2, &records);
    ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap()
}

/// Two indel variants at the SAME position (200), both routed to `Dense`
/// (`x_calls = 2` each, `np = 4`: indel dense_bits = 68 < var_key_bits = 128),
/// with different deletion lengths — for exercising
/// `dense_max_end_keys`'s claim that it scans the WHOLE tied run at a winning
/// position rather than stopping at the first carried hit.
///
/// File order (and therefore dense-table column order: `rvk`'s per-chunk
/// classify loop pushes dense variants in encounter order, and
/// `DenseUnion`'s position sort is stable, so same-position ties keep that
/// order) is: the LONGER deletion (ref "ATCG" -> alt "A", deletion_len 3,
/// ext 4, end 204) FIRST, then the SHORTER one (ref "AT" -> alt "A",
/// deletion_len 1, ext 2, end 202) SECOND. `dense_max_end_keys`'s backward
/// walk visits the HIGHER table index first, i.e. the shorter/later one —
/// so a scan that stopped at that first hit would report 202 instead of the
/// correct 204.
///
/// Deliberately NOT a homopolymer run (unlike a first draft of this fixture,
/// which used all-'A' refs): `normalize::left_align` rolls a deletion left
/// while `ref_seq[pos] == ref_seq[pos + ndel]`, which a homopolymer run
/// trivially satisfies against its own last base, silently shifting both
/// variants' positions by 1 (see `synth_reader_tiebreak`'s doc comment for
/// the same gotcha). "ATCG" and "AT" both start with 'A' but each ends on a
/// distinct base different from the anchor, so neither rolls. Both records
/// still agree on the reference at every position they share ("AT" is a
/// prefix of "ATCG"), so `build_fasta_with_index`'s last-record-wins
/// stamping never conflicts.
fn synth_reader_dense_tie(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 200,
            ref_allele: b"ATCG",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 0, 0],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: vec![0, 0, 1, 1],
        },
    ];
    build_contig(out, "chr1", &samples, 2, &records);
    ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap()
}

/// `n_samples` samples at `ploidy` (flat gt layout `[s0_p0, s0_p1, ...]`),
/// carrying three variants scattered across the hap axis (first hap, an
/// interior hap, last hap) so per-hap max-end keys actually vary. Built for
/// `test_max_end_keys_parallel_matches_serial_reduction`, which needs
/// `n_samples * ploidy >= PAR_COLUMN_THRESHOLD` to force
/// `find_ranges_haps`'s rayon `fold`/`reduce` branch.
fn synth_reader_many_haps(out: &std::path::Path, n_samples: usize, ploidy: usize) -> ContigReader {
    let sample_names: Vec<String> = (0..n_samples).map(|i| format!("S{i}")).collect();
    let samples: Vec<&str> = sample_names.iter().map(String::as_str).collect();
    let h = n_samples * ploidy;

    let mut gt_snp = vec![0i32; h];
    gt_snp[0] = 1; // hap 0
    let mut gt_ins = vec![0i32; h];
    gt_ins[h / 2] = 1; // an interior hap
    let mut gt_del = vec![0i32; h];
    gt_del[h - 1] = 1; // last hap

    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: gt_snp,
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: gt_ins,
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: gt_del,
        },
    ];
    build_contig(out, "chr1", &samples, ploidy, &records);
    ContigReader::open(out.to_str().unwrap(), "chr1", n_samples, ploidy).unwrap()
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

#[test]
fn test_py_read_ranges_dict_matches_overlap_batch_dict() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let _reader = synth_reader(&out);
    let base = out.to_str().unwrap().to_string();
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    Python::attach(|py| {
        let pr = PyContigReader::new(&base, "chr1", 2, 2).unwrap();
        let d_ob = pr.overlap_batch(py, regions.clone()).unwrap();
        let d_rr = pr.read_ranges(py, regions.clone(), None).unwrap();
        assert_payload_dicts_eq(&d_ob, &d_rr);
    });
}

/// `find_ranges` must build a bounded number of search trees regardless of how
/// many regions are queried. Before the column-outer rewrite this was
/// O(regions x columns): each `vk_*_overlap` call rebuilt the column's tree.
///
/// Uses `synth_reader_wide`, NOT `synth_reader`: `synth_reader` populates only
/// one of the four `vk_snp` columns (the other three are empty and early-
/// return without ever calling `SearchTree::new`), so on that fixture the old
/// code's growth is exactly 1 (dense union) + 1 (dense indel) + 1 (the one
/// populated `vk_snp` column) = 3/region — which exactly *ties* a `3 *
/// Δregions` allowance instead of exceeding it, so the guard would pass even
/// against the unfixed region-outer code. `synth_reader_wide` gives all 4
/// `vk_snp` columns a variant, so the old code's growth is 4 (var_key columns)
/// plus 1 (dense union) plus 1 (dense indel) = 6/region (this fixture builds
/// no dense-snp tree at all — nothing routes Dense-SNP — so there are only 2
/// legitimate per-region dense trees, not 3). Over the 15 extra regions below
/// that's 90 actual tree builds against an allowance of 30: a real ~3x
/// separation, not a tie.
///
/// After the dense hoist (#145) NO channel builds a tree per region: the
/// var_key columns, the dense union and the two dense class tables are each
/// swept once per call, so `cost_many` must equal `cost_one` exactly — zero
/// growth across all channels, not just the var_key ones.
///
/// The fixture is deliberately small (2 samples x 2 ploidy = 4 columns, well
/// under `PAR_COLUMN_THRESHOLD`) so the serial path runs on this thread and
/// `search::search_tree_build_count` — a thread-local — stays observable.
#[test]
fn test_find_ranges_tree_builds_do_not_scale_with_regions() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader_wide(&out);

    let one = vec![(0u32, 1_000_000u32)];
    let many: Vec<(u32, u32)> = (0..16).map(|i| (i * 20, i * 20 + 1_000_000)).collect();

    let b0 = search::search_tree_build_count();
    let _ = find_ranges(&reader, &one, None);
    let cost_one = search::search_tree_build_count() - b0;

    let b1 = search::search_tree_build_count();
    let _ = find_ranges(&reader, &many, None);
    let cost_many = search::search_tree_build_count() - b1;

    // After the dense hoist (#145) NO channel builds a tree per region: the
    // var_key columns, the dense union and the two dense class tables are each
    // swept once per call. So this is exact equality, not an allowance — a
    // budget here is what let the dense leak survive #144.
    assert_eq!(
        cost_many, cost_one,
        "tree builds grew with region count: {cost_one} -> {cost_many}"
    );
}

#[test]
fn test_py_gather_of_find_matches_read_dict() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let _reader = synth_reader(&out);
    let base = out.to_str().unwrap().to_string();
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    Python::attach(|py| {
        let pr = PyContigReader::new(&base, "chr1", 2, 2).unwrap();
        let bundle = pr.find_ranges(py, regions.clone(), None).unwrap();
        let d_gather = pr.gather_ranges(py, bundle).unwrap();
        let d_read = pr.read_ranges(py, regions.clone(), None).unwrap();
        assert_payload_dicts_eq(&d_gather, &d_read);
    });
}

fn unpack_end(key: u64) -> u32 {
    ((key >> MAX_END_SHIFT) + (key & ((1 << MAX_END_SHIFT) - 1))) as u32
}

/// The per-region max end must be the end of the HIGHEST-POSITION overlapping
/// variant (ties broken by the larger end), not the largest end overall — this
/// is the SVAR1-parity rule GenVarLoader's `_svar2_region_max_ends` implements.
#[test]
fn test_max_end_keys_pick_highest_position_variant() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);

    // Region covering all three variants; the DEL at 300 is highest-position.
    let regions = vec![(0u32, 1_000u32)];
    let sample_cols: Vec<usize> = (0..2).collect();
    let h = 2 * reader.ploidy();
    let mut snp = vec![0i64; h * 2];
    let mut indel = vec![0i64; h * 2];
    let vk_keys = find_ranges_haps(&reader, &regions, &sample_cols, 0, h, &mut snp, &mut indel);

    // `dense_union`/`DenseUnion::index` are pub(crate); this integration
    // test crate is external to `genoray_core`, so the per-region dense range
    // is obtained via `find_ranges`'s public `dense_range` field instead —
    // it's computed the same way (`reader.dense_union().index().overlap(qs, qe)`),
    // independent of the sample subset.
    let dense_range = find_ranges(&reader, &regions, None).dense_range;
    let dense_keys = dense_max_end_keys(&reader, &regions, &dense_range, &sample_cols, true);

    let key = vk_keys[0].max(dense_keys[0]);
    assert_ne!(key, 0, "region has variants, so the key must be non-zero");
    assert_eq!(
        unpack_end(key),
        302,
        "DEL@300 with deletion_len 1 ends at 302"
    );
}

/// A region containing only the SNP must report that SNP's end, and an empty
/// region must report the 0 sentinel so the caller keeps its original chromEnd.
#[test]
fn test_max_end_keys_snp_only_and_empty_region() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);

    let regions = vec![(90u32, 110u32), (900u32, 950u32)];
    let sample_cols: Vec<usize> = (0..2).collect();
    let h = 2 * reader.ploidy();
    let mut snp = vec![0i64; h * regions.len() * 2];
    let mut indel = vec![0i64; h * regions.len() * 2];
    let vk_keys = find_ranges_haps(&reader, &regions, &sample_cols, 0, h, &mut snp, &mut indel);

    let dense_range = find_ranges(&reader, &regions, None).dense_range;
    let dense_keys = dense_max_end_keys(&reader, &regions, &dense_range, &sample_cols, true);

    let k0 = vk_keys[0].max(dense_keys[0]);
    assert_eq!(unpack_end(k0), 101, "SNP@100 ends at 101");
    assert_eq!(
        vk_keys[1].max(dense_keys[1]),
        0,
        "no variants in [900, 950)"
    );
}

/// Splitting the hap axis must not change the reduced result — the writer
/// reduces per-chunk keys with an elementwise max.
#[test]
fn test_max_end_keys_reduce_across_hap_slices() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);

    let regions = vec![(0u32, 1_000u32)];
    let sample_cols: Vec<usize> = (0..2).collect();
    let h = 2 * reader.ploidy();

    let mut snp = vec![0i64; h * 2];
    let mut indel = vec![0i64; h * 2];
    let whole = find_ranges_haps(&reader, &regions, &sample_cols, 0, h, &mut snp, &mut indel);

    let mut reduced = vec![0u64; 1];
    for lo in (0..h).step_by(1) {
        let mut s = vec![0i64; 2];
        let mut i = vec![0i64; 2];
        let part = find_ranges_haps(&reader, &regions, &sample_cols, lo, lo + 1, &mut s, &mut i);
        reduced[0] = reduced[0].max(part[0]);
    }
    assert_eq!(whole, reduced);
}

/// The per-region max end must order by POSITION first, end second — not by
/// absolute end. `synth_reader_tiebreak`'s DEL@200 has a larger end (351)
/// than its SNP@300 (301), but the SNP is at the higher position, so it must
/// win. A "max by absolute end" bug would report 351 here; this fixture (see
/// its doc comment) is specifically built so that bug and the correct rule
/// disagree, unlike `test_max_end_keys_pick_highest_position_variant`'s
/// `synth_reader`, where they happen to coincide.
#[test]
fn test_max_end_keys_position_beats_larger_end_at_lower_position() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader_tiebreak(&out);

    let regions = vec![(0u32, 1_000u32)];
    let sample_cols: Vec<usize> = (0..2).collect();
    let h = 2 * reader.ploidy();
    let mut snp = vec![0i64; h * 2];
    let mut indel = vec![0i64; h * 2];
    let vk_keys = find_ranges_haps(&reader, &regions, &sample_cols, 0, h, &mut snp, &mut indel);

    let dense_range = find_ranges(&reader, &regions, None).dense_range;
    let dense_keys = dense_max_end_keys(&reader, &regions, &dense_range, &sample_cols, true);

    let key = vk_keys[0].max(dense_keys[0]);
    assert_eq!(
        unpack_end(key),
        301,
        "SNP@300 (higher position) must win over DEL@200's larger end (351)"
    );
}

/// `dense_max_end_keys` must scan the WHOLE tied run at a winning position,
/// not stop at the first carried hit it finds — see `synth_reader_dense_tie`'s
/// doc comment for the exact table-order argument. A scan that stopped early
/// would report 202 (the shorter deletion, hit first by the backward walk);
/// the correct answer is 204 (the longer one, filed earlier in the table but
/// visited second).
#[test]
fn test_dense_max_end_keys_scans_full_tied_run() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader_dense_tie(&out);

    let regions = vec![(0u32, 1_000u32)];
    let sample_cols: Vec<usize> = (0..2).collect();
    let dense_range = find_ranges(&reader, &regions, None).dense_range;
    let dense_keys = dense_max_end_keys(&reader, &regions, &dense_range, &sample_cols, true);

    assert_eq!(
        unpack_end(dense_keys[0]),
        204,
        "must pick the longer deletion (ext=4, end=204), not the shorter tied \
         one (ext=2, end=202) the backward walk hits first"
    );
}

/// The rayon `fold`/`reduce` accumulation of per-region max-end keys (taken
/// when `n_haps >= PAR_COLUMN_THRESHOLD`) must agree with the serial
/// single-hap-slice reduction the smaller fixtures above exercise. 32 samples
/// x ploidy 2 = 64 haps forces the parallel branch for the whole-slice call;
/// each single-hap call (`hap_hi = hap_lo + 1`) has `n_haps = 1`, well under
/// the threshold, so it takes the serial path — this compares the two
/// branches directly instead of trusting the fold/reduce closures by
/// inspection.
#[test]
fn test_max_end_keys_parallel_matches_serial_reduction() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let n_samples = 32;
    let ploidy = 2;
    let reader = synth_reader_many_haps(&out, n_samples, ploidy);

    let regions = vec![(0u32, 1_000u32)];
    let sample_cols: Vec<usize> = (0..n_samples).collect();
    let h = n_samples * ploidy;
    assert!(
        h >= PAR_COLUMN_THRESHOLD,
        "fixture must exercise the parallel branch"
    );

    let mut snp = vec![0i64; h * 2];
    let mut indel = vec![0i64; h * 2];
    let whole = find_ranges_haps(&reader, &regions, &sample_cols, 0, h, &mut snp, &mut indel);

    let mut reduced = vec![0u64; 1];
    for lo in 0..h {
        let mut s = vec![0i64; 2];
        let mut i = vec![0i64; 2];
        let part = find_ranges_haps(&reader, &regions, &sample_cols, lo, lo + 1, &mut s, &mut i);
        reduced[0] = reduced[0].max(part[0]);
    }
    assert_eq!(whole, reduced);
    // Sanity: the fixture actually carries a variant in [0, 1000), so this
    // isn't vacuously comparing two zero vectors.
    assert_ne!(whole[0], 0);
}

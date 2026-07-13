//! Task 5: `vk_src` provenance must be aligned 1:1 with the merged `vk`
//! channel, and every entry must resolve back to the record that actually
//! produced it — through the snp/indel position-merge, through multiple
//! var_key columns with different absolute-index bases, and through the
//! per-element overlap filter that drops candidates a narrower query excludes.
mod common;

use common::{SynthRecord, build_contig};
use genoray_core::query::gather::gather_haps_readbound_src;
use genoray_core::query::{
    BatchResultSplit, ContigReader, HapRanges, find_ranges, gather_haps_readbound,
};
use genoray_core::spine;
use tempfile::tempdir;

/// A store with BOTH var_key/snp and var_key/indel records, spread across all
/// four (sample, ploid) columns of a 2-sample/ploidy-2 cohort, plus a
/// same-position SNP+indel tie on column (S0, p0) — modeled on
/// `tie_break_reader` in `tests/test_readbound_gather.rs`. Every record here
/// is AC=1, so `cost_model::choose_representation` always keeps it in
/// `var_key` (never `dense`) — see that reader's doc comment.
fn provenance_reader(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let records = vec![
        // col0 (S0,p0): SNP only.
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0],
        },
        // col2 (S1,p0): indel only.
        SynthRecord {
            pos: 150,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![0, 0, 1, 0],
        },
        // col1 (S0,p1): SNP only. This is the record that pins absolute (not
        // column-local) indexing: col0 already contributed 2 snp entries
        // ahead of it in the packed buffer, so its absolute index is 2 even
        // though it is the FIRST snp entry in its own column.
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![0, 1, 0, 0],
        },
        // col3 (S1,p1): indel only.
        SynthRecord {
            pos: 250,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![0, 0, 0, 1],
        },
        // col0 (S0,p0) again: SNP+indel at the SAME position 400, same hap as
        // the pos-100 SNP above -> exercises the position-merge tie-break
        // (snp must win) on a column whose absolute base is already nonzero.
        SynthRecord {
            pos: 400,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0],
        },
        SynthRecord {
            pos: 400,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![1, 0, 0, 0],
        },
    ];
    build_contig(out, "chr1", &samples, 2, &records);
    ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap()
}

/// `vk_src` must be aligned 1:1 with `vk`, and each entry must point at the
/// record that actually produced it — including through the position-merge
/// of snp+indel and through the overlap FILTER that drops records (the
/// filter is why the index can't be recovered from run position afterwards).
#[test]
fn vk_src_is_aligned_with_vk_and_survives_the_merge() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = provenance_reader(&out);
    let n_samples = 2usize;
    let ploidy = 2usize;

    // Region 0: full coverage, sees every record. Region 1: q_start = 300
    // drops the pos-100/150/200/250 records via the left-overlap re-check
    // (`q_start < v_end`) while keeping the pos-400 tie (v_end 401/402 > 300)
    // — this is the "overlap filter drops records" case the provenance must
    // survive: the surviving vk_src entries must still resolve to their true
    // absolute indices, not renumbered ones.
    let regions = [(0u32, 1_000_000u32), (300u32, 1_000_000u32)];

    let rb = find_ranges(&reader, &regions, None);

    // Flatten into HapRanges in the same region-major, sample-major order
    // `find_ranges`/`gather_ranges_readbound` use (see
    // `test_flat_gather_matches_cartesian_full_cohort` in
    // tests/test_readbound_gather.rs, which this mirrors for the full cohort).
    let s_n = rb.n_samples;
    assert_eq!(s_n, n_samples);
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
            dsr.push(rb.dense_snp_range[r].clone());
            dir_.push(rb.dense_indel_range[r].clone());
            for p in 0..ploidy {
                let row = r * (s_n * ploidy) + s * ploidy + p;
                vk_snp_range.push(rb.vk_snp_range[row].clone());
                vk_indel_range.push(rb.vk_indel_range[row].clone());
            }
        }
    }
    let hr = HapRanges::new(
        &region_starts,
        &orig_samples,
        &vk_snp_range,
        &vk_indel_range,
        &dsr,
        &dir_,
        ploidy,
    );

    let plain: BatchResultSplit = gather_haps_readbound(&reader, &hr);
    let with_src: BatchResultSplit = gather_haps_readbound_src(&reader, &hr);

    // The no-provenance path is unchanged, and provenance is purely additive.
    assert!(
        plain.vk_src.is_empty(),
        "plain path must not populate vk_src"
    );
    assert_eq!(
        plain.vk, with_src.vk,
        "adding provenance must not change vk"
    );
    assert_eq!(plain.vk_off, with_src.vk_off);
    assert_eq!(with_src.vk_src.len(), with_src.vk.len());

    // Coverage guard: fail loudly if the fixture stopped exercising both
    // sub-streams (rather than passing vacuously on an all-snp or all-indel
    // vk).
    let (mut any_snp, mut any_indel) = (false, false);

    // Every vk_src must resolve to the record that produced its key,
    // regardless of which hap/region it came from or whether the query
    // narrowed the candidate window.
    let snp_pos = reader.vk_snp_positions();
    let indel_pos = reader.vk_indel_positions();
    for (i, kr) in with_src.vk.iter().enumerate() {
        let (is_indel, idx) = spine::unpack_vk_src(with_src.vk_src[i]);
        any_snp |= !is_indel;
        any_indel |= is_indel;
        let pos = if is_indel {
            indel_pos[idx]
        } else {
            snp_pos[idx]
        };
        assert_eq!(
            pos, kr.position,
            "vk_src[{i}] points at position {pos} but vk[{i}] is at {}",
            kr.position
        );
    }
    assert!(any_snp, "fixture must exercise a var_key/snp vk_src entry");
    assert!(
        any_indel,
        "fixture must exercise a var_key/indel vk_src entry"
    );

    // Pin the tie-break explicitly: hap (region 0, sample S0, ploid 0) must
    // merge to [snp@100, snp@400, indel@400] — snp wins the position-400 tie.
    let s0_slot = rb
        .sample_cols
        .iter()
        .position(|&orig| orig == 0)
        .expect("S0 must be present");
    let row0 = s0_slot * ploidy; // region 0, S0, p0
    let (vs, ve) = (with_src.vk_off[row0], with_src.vk_off[row0 + 1]);
    let got: Vec<(u32, bool)> = with_src.vk[vs..ve]
        .iter()
        .zip(&with_src.vk_src[vs..ve])
        .map(|(kr, &src)| (kr.position, spine::unpack_vk_src(src).0))
        .collect();
    assert_eq!(
        got,
        vec![(100, false), (400, false), (400, true)],
        "hap (region0, S0, p0) must merge to [snp@100, snp@400, indel@400] with snp winning the tie"
    );

    // Also pin the absolute-index recovery is correct on this same hap: the
    // pos-400 snp entry's absolute index must land in vk_snp_positions/keys
    // at exactly `pos == 400`, distinguishing it from the pos-100 entry.
    let (is_indel0, idx0) = spine::unpack_vk_src(with_src.vk_src[vs]);
    assert!(!is_indel0);
    assert_eq!(snp_pos[idx0], 100);
    let (is_indel1, idx1) = spine::unpack_vk_src(with_src.vk_src[vs + 1]);
    assert!(!is_indel1);
    assert_eq!(snp_pos[idx1], 400);
    let (is_indel2, idx2) = spine::unpack_vk_src(with_src.vk_src[vs + 2]);
    assert!(is_indel2);
    assert_eq!(indel_pos[idx2], 400);

    // n_samples()/ploidy() accessors (needed by gvl regardless of fields):
    // confirm they report what the reader was opened with.
    assert_eq!(reader.n_samples(), n_samples);
    assert_eq!(reader.ploidy(), ploidy);
}

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

// --- Task 8: `decode_hap_src` record-order + per-record provenance --------

use genoray_core::query::gather::overlap_batch_src;
use genoray_core::query::overlap_batch;

/// A store with all four provenance classes present: a var_key SNP, a var_key
/// indel, a dense SNP table (one fully-carried + one partially-carried
/// record), and a dense indel table (two records). Modeled on `synth_reader`
/// in `tests/test_readbound_gather.rs`, which already routes an AC=4 and an
/// AC=2 SNP to Dense and two AC=3 indels to Dense; this fixture additionally
/// adds an AC=1 indel so `var_key/indel` is exercised too — `synth_reader` has
/// none. Per `cost_model::choose_representation` at n_samples=2/ploidy=2
/// (np=4): AC=1 always stays `VarKey` for both classes; AC>=2 routes SNPs to
/// `Dense` and AC>=2 routes indels to `Dense` too.
fn field_provenance_reader(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let records = vec![
        // var_key/snp: AC=1 (S0,p0).
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0],
        },
        // var_key/indel: AC=1 (S1,p0). Insertion (ref "A" -> alt "AC"), ilen +1.
        SynthRecord {
            pos: 130,
            ref_allele: b"A",
            alts: vec![&b"AC"[..]],
            gt: vec![0, 0, 1, 0],
        },
        // dense/snp: AC=4, all haps carry -> exercises an all-true presence row.
        SynthRecord {
            pos: 150,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 1, 1, 1],
        },
        // var_key/snp tie-break probe: AC=1, at the SAME position (150) as the
        // dense/snp record above, carried by the SAME hap (S0,p0 — the dense
        // record's gt is all-1s, so it always carries too). A flipped
        // `decode_hap_src` tie-break (`<=` -> `<` at the var_key/dense merge)
        // would swap this pair's order — undetectable by
        // `decode_hap_src_matches_decode_hap_order` without this collision,
        // since every other position in this fixture is unique. Different ALT
        // (T vs C) makes the two records distinguishable in `hc.alts`.
        SynthRecord {
            pos: 150,
            ref_allele: b"A",
            alts: vec![&b"T"[..]],
            gt: vec![1, 0, 0, 0],
        },
        // dense/snp: AC=2, partial carriage -> exercises a mixed presence row.
        // Different ALT than the pos-150 record so a wrong-allele bug would
        // be caught by a decode-level comparison.
        SynthRecord {
            pos: 175,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![1, 1, 0, 0],
        },
        // dense/indel: AC=3, insertion.
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![0, 1, 1, 1],
        },
        // dense/indel: AC=3, deletion.
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

/// `decode_hap_src` must emit exactly one provenance entry per decoded
/// record, in the same order as `decode_hap`, and each entry must resolve
/// back to the record that actually produced it — for var_key entries via the
/// public `vk_snp_positions`/`vk_indel_positions` accessors, and for dense
/// entries via this fixture's known (and asserted-monotonic-by-construction)
/// per-class row order. A coverage guard fails loudly if any of the four
/// provenance classes (var_key snp/indel, dense snp/indel) stops appearing.
#[test]
fn decode_hap_src_is_aligned_with_decoded_records() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = field_provenance_reader(&out);
    let n_samples = 2usize;
    let ploidy = 2usize;
    let regions = [(0u32, 1_000_000u32)];
    let br = overlap_batch_src(&reader, &regions);

    let snp_pos = reader.vk_snp_positions();
    let indel_pos = reader.vk_indel_positions();

    // Known dense per-class row -> position mapping for THIS fixture: both
    // dense tables are built from records already given in increasing
    // position order, and `SearchTree`/`overlap_range` require monotonic
    // per-class positions, so row i is exactly the i-th dense record of that
    // class in insertion order.
    let dense_snp_expected = [150u32, 175u32];
    let dense_indel_expected = [200u32, 300u32];

    let (mut n_vk_snp, mut n_vk_indel, mut n_dense_snp, mut n_dense_indel) =
        (0u32, 0u32, 0u32, 0u32);

    for s in 0..n_samples {
        for p in 0..ploidy {
            let (hc, srcs) = br.decode_hap_src(&reader, 0, s, p);
            assert_eq!(hc.positions.len(), srcs.len(), "one src per decoded record");
            for (i, src) in srcs.iter().enumerate() {
                let want_pos = hc.positions[i];
                if !src.is_dense {
                    let pos = if src.is_indel {
                        n_vk_indel += 1;
                        indel_pos[src.idx]
                    } else {
                        n_vk_snp += 1;
                        snp_pos[src.idx]
                    };
                    assert_eq!(
                        pos, want_pos,
                        "var_key src[{i}] resolves to the wrong position"
                    );
                } else if src.is_indel {
                    n_dense_indel += 1;
                    assert_eq!(
                        dense_indel_expected[src.idx], want_pos,
                        "dense/indel src[{i}] resolves to the wrong row"
                    );
                } else {
                    n_dense_snp += 1;
                    assert_eq!(
                        dense_snp_expected[src.idx], want_pos,
                        "dense/snp src[{i}] resolves to the wrong row"
                    );
                }
            }
        }
    }

    assert!(n_vk_snp > 0, "fixture must exercise a var_key/snp record");
    assert!(
        n_vk_indel > 0,
        "fixture must exercise a var_key/indel record"
    );
    assert!(n_dense_snp > 0, "fixture must exercise a dense/snp record");
    assert!(
        n_dense_indel > 0,
        "fixture must exercise a dense/indel record"
    );
}

/// `decode_hap_src` must return records in EXACTLY the order `decode_hap`
/// does — otherwise field values would be attached to the wrong variants.
#[test]
fn decode_hap_src_matches_decode_hap_order() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = field_provenance_reader(&out);
    let regions = [(0u32, 1_000_000u32)];
    let plain = overlap_batch(&reader, &regions);
    let with_src = overlap_batch_src(&reader, &regions);
    for s in 0..2 {
        for p in 0..2 {
            let a = plain.decode_hap(&reader, 0, s, p);
            let (b, srcs) = with_src.decode_hap_src(&reader, 0, s, p);
            assert_eq!(a.positions, b.positions);
            assert_eq!(a.ilens, b.ilens);
            assert_eq!(a.alts, b.alts);
            assert_eq!(srcs.len(), b.positions.len());
        }
    }
}

/// Explicit pin for the var_key/dense position tie at hap (S0, p0): the
/// var_key/snp AC=1 record (ALT `T`) and the dense/snp AC=4 record (ALT `C`)
/// both sit at position 150 and are both carried by this hap (AC=4's gt is
/// all-1s). Per the merge's documented contract (`decode_hap`'s
/// `spine::merge_keys`, mirrored by `decode_hap_src`'s two-pointer merge),
/// var_key must win the tie and come first.
///
/// `decode_hap_src_matches_decode_hap_order` above already catches a flipped
/// tie-break generically (its `assert_eq!(a.alts, b.alts)` would fail, since
/// `decode_hap` — untouched — still orders var_key first), but this test
/// pins the expected order directly and verifies the fixture actually routed
/// each AC to the class this test assumes, rather than assuming it.
#[test]
fn decode_hap_src_pins_var_key_dense_position_tie() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = field_provenance_reader(&out);
    let regions = [(0u32, 1_000_000u32)];
    let br = overlap_batch_src(&reader, &regions);

    // Class-routing guard: verify, don't assume.
    assert!(
        reader.vk_snp_positions().contains(&150),
        "the AC=1 record at 150 must have routed to var_key/snp"
    );

    let (hc, srcs) = br.decode_hap_src(&reader, 0, 0, 0);
    let at_150: Vec<usize> = hc
        .positions
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p == 150)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        at_150.len(),
        2,
        "both records at position 150 must be carried by hap (S0,p0)"
    );
    let (i0, i1) = (at_150[0], at_150[1]);

    assert!(!srcs[i0].is_dense, "var_key must win the position tie");
    assert_eq!(hc.alts[i0], b"T", "var_key entry's ALT");
    assert!(srcs[i1].is_dense, "dense must come second on the tie");
    assert_eq!(hc.alts[i1], b"C", "dense entry's ALT");
}

// --- Task 8 fix pass (Finding 3): pin `decode_batch_fields`'s emitted BYTES,
// not just that it builds. `gather_field_bytes`/`OpenField` are plain Rust
// (no `Python<'py>`), so this needs no GIL. -----------------------------

use genoray_core::layout::{ContigPaths, FieldSub};
use genoray_core::py_query_decode::{OpenField, gather_field_bytes};

fn write_field_i32(paths: &ContigPaths, category: &str, name: &str, sub: FieldSub, vals: &[i32]) {
    let p = paths.field_values(category, name, sub);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&p, bytes).unwrap();
}

/// `decode_batch_fields`'s Rust core (`OpenField::open` + `gather_field_bytes`)
/// must gather the right byte for the right record — through all four
/// provenance classes (var_key snp/indel, dense snp/indel) and through the
/// dense-FORMAT `(row, orig_sample)` stride, which is the one bug class a
/// build-only smoke test cannot catch. Every written value is distinct (per
/// field, class-offset + index) so a wrong index reads a wrong-but-plausible
/// value instead of accidentally still matching.
#[test]
fn gather_field_bytes_pins_bytes_per_class_and_dense_format_stride() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = field_provenance_reader(&out);
    let n_samples = reader.n_samples();
    assert_eq!(n_samples, 2, "fixture is a fixed 2-sample cohort");

    // Per-class call/row counts — verified via the reader's own accessors for
    // var_key, and via the already-pinned dense row layout
    // (`decode_hap_src_is_aligned_with_decoded_records`'s
    // `dense_snp_expected`/`dense_indel_expected`, unaffected by the Finding-2
    // var_key addition above) for dense.
    let n_vk_snp = reader.vk_snp_positions().len();
    let n_vk_indel = reader.vk_indel_positions().len();
    assert_eq!(n_vk_snp, 2, "var_key/snp: pos-100 and pos-150 AC=1 records");
    assert_eq!(n_vk_indel, 1, "var_key/indel: the pos-130 AC=1 record");
    let n_dense_snp = 2usize; // rows: pos150 (AC=4), pos175 (AC=2)
    let n_dense_indel = 2usize; // rows: pos200 (AC=3 ins), pos300 (AC=3 del)

    // --- INFO field "IV" (i32): one value per (class, call/row), disjoint
    // ranges per class so a cross-class index bug reads an out-of-range value.
    let vk_snp_iv: Vec<i32> = (0..n_vk_snp as i32).map(|i| 100 + i).collect();
    let vk_indel_iv: Vec<i32> = (0..n_vk_indel as i32).map(|i| 200 + i).collect();
    let dense_snp_iv: Vec<i32> = (0..n_dense_snp as i32).map(|i| 300 + i).collect();
    let dense_indel_iv: Vec<i32> = (0..n_dense_indel as i32).map(|i| 400 + i).collect();

    let paths = ContigPaths::new(out.to_str().unwrap(), "chr1");
    write_field_i32(&paths, "info", "IV", FieldSub::VkSnp, &vk_snp_iv);
    write_field_i32(&paths, "info", "IV", FieldSub::VkIndel, &vk_indel_iv);
    write_field_i32(&paths, "info", "IV", FieldSub::DenseSnp, &dense_snp_iv);
    write_field_i32(&paths, "info", "IV", FieldSub::DenseIndel, &dense_indel_iv);

    // --- FORMAT field "DP" (i32): var_key is per-call (no sample stride, per
    // the cost model: FORMAT is paid per carrier call in var_key); dense is
    // per (row, orig_sample) — `value = 5000 + row*10 + sample`, so samples
    // DIFFER at the same row. This is the assertion that catches a missing or
    // wrong dense-FORMAT stride.
    let vk_snp_dp: Vec<i32> = (0..n_vk_snp as i32).map(|i| 600 + i).collect();
    let vk_indel_dp: Vec<i32> = (0..n_vk_indel as i32).map(|i| 700 + i).collect();
    let mut dense_snp_dp: Vec<i32> = Vec::with_capacity(n_dense_snp * n_samples);
    for row in 0..n_dense_snp {
        for s in 0..n_samples {
            dense_snp_dp.push((5000 + row * 10 + s) as i32);
        }
    }
    let mut dense_indel_dp: Vec<i32> = Vec::with_capacity(n_dense_indel * n_samples);
    for row in 0..n_dense_indel {
        for s in 0..n_samples {
            dense_indel_dp.push((8000 + row * 10 + s) as i32);
        }
    }
    write_field_i32(&paths, "format", "DP", FieldSub::VkSnp, &vk_snp_dp);
    write_field_i32(&paths, "format", "DP", FieldSub::VkIndel, &vk_indel_dp);
    write_field_i32(&paths, "format", "DP", FieldSub::DenseSnp, &dense_snp_dp);
    write_field_i32(
        &paths,
        "format",
        "DP",
        FieldSub::DenseIndel,
        &dense_indel_dp,
    );

    let regions = [(0u32, 1_000_000u32)];
    let br = overlap_batch_src(&reader, &regions);

    let iv = OpenField::open(&paths, "info", "IV", "i32", n_samples).unwrap();
    let dp = OpenField::open(&paths, "format", "DP", "i32", n_samples).unwrap();
    let fields = vec![iv, dp];
    let fbytes = gather_field_bytes(&reader, &br, &fields);
    assert_eq!(fbytes.len(), 2);

    // Independently replicate the expected bytes via the SAME provenance
    // (`decode_hap_src`) `gather_field_bytes` itself uses, but computing the
    // expected value here rather than calling into the code under test.
    let mut expect_iv: Vec<u8> = Vec::new();
    let mut expect_dp: Vec<u8> = Vec::new();
    let mut n_records = 0usize;
    for r in 0..br.n_regions {
        for s in 0..br.n_samples {
            for p in 0..br.ploidy {
                let (_, srcs) = br.decode_hap_src(&reader, r, s, p);
                for src in &srcs {
                    n_records += 1;
                    let iv_val = match (src.is_dense, src.is_indel) {
                        (false, false) => vk_snp_iv[src.idx],
                        (false, true) => vk_indel_iv[src.idx],
                        (true, false) => dense_snp_iv[src.idx],
                        (true, true) => dense_indel_iv[src.idx],
                    };
                    expect_iv.extend_from_slice(&iv_val.to_le_bytes());

                    let dp_val = match (src.is_dense, src.is_indel) {
                        (false, false) => vk_snp_dp[src.idx],
                        (false, true) => vk_indel_dp[src.idx],
                        // Dense: format_at(row, orig_sample) == row*n_samples + orig_sample.
                        (true, false) => dense_snp_dp[src.idx * n_samples + s],
                        (true, true) => dense_indel_dp[src.idx * n_samples + s],
                    };
                    expect_dp.extend_from_slice(&dp_val.to_le_bytes());
                }
            }
        }
    }

    assert_eq!(fbytes[0], expect_iv, "INFO field bytes mismatch");
    assert_eq!(fbytes[1], expect_dp, "FORMAT field bytes mismatch");
    assert_eq!(
        fbytes[0].len(),
        n_records * 4,
        "field_bytes.len() must equal n_records * itemsize"
    );
    assert_eq!(fbytes[1].len(), n_records * 4);

    // Sanity: this fixture must actually exercise a dense record carried by
    // BOTH samples with DIFFERING per-sample DP values (else the dense-FORMAT
    // stride assertion above would pass vacuously). The AC=4 dense/snp record
    // at row 0 (pos 150) is carried by every hap, so both s=0 and s=1 read it.
    let row0 = 0usize;
    assert_ne!(
        dense_snp_dp[row0 * n_samples],
        dense_snp_dp[row0 * n_samples + 1],
        "row-0 DP must differ across samples for the stride test to be meaningful"
    );
}

// Disk-integration test for `Svar2Source`: builds a tiny 2-sample contig via the
// real conversion pipeline (plus the top-level `meta.json` `Svar2Source::new`
// needs to resolve the original cohort size from), then drives `Svar2Source`
// over the full region + all samples and checks the emitted `RawRecord`s are
// position-sorted with the right carrier bits.

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::meta::{FORMAT_VERSION, write_meta};
use genoray_core::record_source::RecordSource;
use genoray_core::svar2_source::{OverlapMode, Svar2Source};
use tempfile::tempdir;

#[test]
fn emits_position_sorted_records_with_carrier_bits() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 2,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0], // sample0/hap0 carrier
        },
        SynthRecord {
            pos: 5,
            ref_allele: b"ATG",
            alts: vec![&b"A"[..]],
            gt: vec![0, 0, 1, 1], // sample1, both haps
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    // `Svar2Source::new` reads the original cohort size from the top-level
    // `meta.json` (there is no per-contig sample count sidecar); `build_contig`
    // only runs `process_chromosome` (mirrors `run_conversion_pipeline`'s
    // per-contig half), so write the cohort-level file it doesn't cover.
    write_meta(
        &out,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();

    let mut src = Svar2Source::new(
        out.to_str().unwrap(),
        "chr1",
        &[0, 1],
        2,
        &[(0, 40)],
        OverlapMode::Pos,
    )
    .unwrap();

    let r0 = src.next_record().unwrap().unwrap();
    assert_eq!(r0.pos, 2);
    assert_eq!(r0.reference.len(), 1); // SNP: 1-byte dummy REF
    assert_eq!(r0.alts.len(), 1);
    assert_eq!(r0.alts[0], b"C".to_vec());
    assert_eq!(r0.gt, vec![1, 0, 0, 0]); // sample0/hap0 carrier

    let r1 = src.next_record().unwrap().unwrap();
    assert_eq!(r1.pos, 5);
    assert!(r1.reference.len() > 1); // pure DEL: REF length == 1 - ilen
    assert_eq!(r1.alts.len(), 1);
    assert_eq!(r1.gt, vec![0, 0, 1, 1]); // sample1, both haps

    assert!(src.next_record().unwrap().is_none());
}

/// Drain a source into `(pos, gt)` pairs.
fn drain(mut src: Svar2Source) -> Vec<(u32, Vec<i32>)> {
    let mut out = Vec::new();
    while let Some(r) = src.next_record().unwrap() {
        out.push((r.pos, r.gt));
    }
    out
}

#[test]
fn the_three_overlap_modes_are_distinguishable() {
    // genoray's published contract (`python/genoray/_svar/_regions.py`):
    //   pos     -> keep iff  q_start <= POS <  q_end
    //   record  -> keep iff  q_start <= POS <  q_end + 1   (POS rule, end + 1)
    //   variant -> keep iff the variant's *extent* overlaps [q_start, q_end)
    //
    // Fixture, queried over [7, 20):
    //   DEL @5, REF len 4 (ilen -3) -> extent [5, 9): extent overlaps, POS does
    //                                  not  => `Variant` only.
    //   SNP @10                     -> inside on every rule => all three modes.
    //   SNP @20  (POS == q_end)     -> POS hits only the widened `record`
    //                                  window; its extent [20, 21) does not
    //                                  touch [7, 20)  => `Record` only.
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0"];
    let records = vec![
        SynthRecord {
            pos: 5,
            ref_allele: b"ATGC",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1],
        },
        SynthRecord {
            pos: 10,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0],
        },
        SynthRecord {
            pos: 20,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![0, 1],
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    write_meta(
        &out,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();

    let open =
        |mode| Svar2Source::new(out.to_str().unwrap(), "chr1", &[0], 2, &[(7, 20)], mode).unwrap();

    assert_eq!(
        drain(open(OverlapMode::Pos)),
        vec![(10, vec![1, 0])],
        "`pos`: POS in [7, 20) only"
    );
    assert_eq!(
        drain(open(OverlapMode::Record)),
        vec![(10, vec![1, 0]), (20, vec![0, 1])],
        "`record`: POS in [7, 21) — the POS == q_end variant is kept, the \
         extent-only deletion is not"
    );
    assert_eq!(
        drain(open(OverlapMode::Variant)),
        vec![(5, vec![1, 1]), (10, vec![1, 0])],
        "`variant`: extent overlap — the deletion is kept, the POS == q_end \
         variant is not"
    );
}

#[test]
fn sample_subset_is_reordered_and_mac0_variants_drop_out() {
    // Guards the `BatchResult::decode_hap` index contract: `s` indexes into the
    // *subset* passed to `read_ranges`, not the original cohort. Selecting
    // `[2, 0]` (reordered, and skipping S1) would scramble carriers under a
    // subset/original mix-up.
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1", "S2"];
    let records = vec![
        SynthRecord {
            pos: 10,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0, 0, 1], // S0/hap0 and S2/hap1
        },
        SynthRecord {
            pos: 12,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![0, 0, 1, 1, 0, 0], // S1 only -> MAC 0 in the subset
        },
        SynthRecord {
            pos: 14,
            ref_allele: b"A",
            alts: vec![&b"T"[..]],
            gt: vec![0, 1, 1, 0, 0, 0], // S0/hap1 (+ S1/hap0, not selected)
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    write_meta(
        &out,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();

    // Output hap order: [S2/hap0, S2/hap1, S0/hap0, S0/hap1].
    let src = Svar2Source::new(
        out.to_str().unwrap(),
        "chr1",
        &[2, 0],
        2,
        &[(0, 40)],
        OverlapMode::Pos,
    )
    .unwrap();
    assert_eq!(
        drain(src),
        vec![(10, vec![0, 1, 1, 0]), (14, vec![0, 0, 0, 1])],
        "pos 12 is S1-only -> MAC 0 in the subset -> dropped"
    );
}

#[test]
fn carrier_bits_or_across_overlapping_regions() {
    // A variant seen through two overlapping regions is emitted once, with the
    // carrier bits OR-ed (not duplicated, not last-write-wins).
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![SynthRecord {
        pos: 10,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt: vec![1, 0, 0, 1],
    }];
    build_contig(&out, "chr1", &samples, 2, &records);
    write_meta(
        &out,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();

    let src = Svar2Source::new(
        out.to_str().unwrap(),
        "chr1",
        &[0, 1],
        2,
        &[(5, 15), (8, 20)],
        OverlapMode::Pos,
    )
    .unwrap();
    assert_eq!(drain(src), vec![(10, vec![1, 0, 0, 1])]);
}

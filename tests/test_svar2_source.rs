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

#[test]
fn overlap_mode_pos_prunes_deletion_extent_hits_before_the_region() {
    // A DEL@pos=5, ref len 4 (ilen=-3), spans [5, 9). Under `Pos` mode a query
    // region starting after `pos` but still inside the deletion's extent (e.g.
    // [7, 20)) must NOT see the call: `pos` (5) is outside `[7, 20)` even
    // though the extent overlaps. Under `Record`/`Variant` mode it must.
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0"];
    let records = vec![SynthRecord {
        pos: 5,
        ref_allele: b"ATGC",
        alts: vec![&b"A"[..]],
        gt: vec![1, 1],
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

    let mut pos_mode = Svar2Source::new(
        out.to_str().unwrap(),
        "chr1",
        &[0],
        2,
        &[(7, 20)],
        OverlapMode::Pos,
    )
    .unwrap();
    assert!(pos_mode.next_record().unwrap().is_none());

    let mut record_mode = Svar2Source::new(
        out.to_str().unwrap(),
        "chr1",
        &[0],
        2,
        &[(7, 20)],
        OverlapMode::Record,
    )
    .unwrap();
    let r = record_mode.next_record().unwrap().unwrap();
    assert_eq!(r.pos, 5);
    assert_eq!(r.gt, vec![1, 1]);
    assert!(record_mode.next_record().unwrap().is_none());
}

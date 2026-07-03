//! Disk-integration tests for the batched M6.1 spine (`overlap_batch`). Builds
//! finished SVAR2 contigs via the real converter, then cross-checks the two-
//! channel batch result against the M5 single-sample `overlap_sample` — the
//! already-oracle-tested reference — for every (region, sample, ploid), plus a
//! decode-free dense-count sanity check.

mod common;

use common::{OwnedRecord, SynthRecord, arb_records, build_contig};
use genoray_core::query::{ContigReader, overlap_batch, overlap_sample};
use proptest::prelude::*;
use tempfile::tempdir;

#[test]
fn test_overlap_batch_matches_overlap_sample() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 1, 1],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![0, 1, 0, 0],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 0, 0],
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap();

    // Two regions: one whole-contig, one that a deletion spans into.
    let regions = [(0u32, 1000u32), (301u32, 302u32)];
    let batch = overlap_batch(&reader, &regions);

    assert_eq!(batch.n_regions, 2);
    assert_eq!(batch.n_samples, 2);
    assert_eq!(batch.ploidy, 2);

    for (r, &(qs, qe)) in regions.iter().enumerate() {
        for s in 0..2usize {
            let single = overlap_sample(&reader, s, qs, qe);
            for p in 0..2usize {
                let got = batch.decode_hap(&reader, r, s, p);
                assert_eq!(got, single.per_hap[p], "r={r} s={s} p={p}");
            }
        }
    }
}

/// popcount of a hap's dense presence bitmask == the dense variants that hap
/// carries in that region (the decode-free count M6c is built on).
fn dense_popcount(batch: &genoray_core::query::BatchResult, r: usize, s: usize, p: usize) -> usize {
    let h = (r * batch.n_samples + s) * batch.ploidy + p;
    let (bit0, bit1) = (batch.dense_present_off[h], batch.dense_present_off[h + 1]);
    (bit0..bit1)
        .filter(|&b| (batch.dense_present[b >> 3] >> (b & 7)) & 1 != 0)
        .count()
}

#[test]
fn test_dense_present_popcount_counts_dense_carriers() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    // 6 samples; a 4-base DEL carried by nearly everyone -> dense/indel.
    let samples = ["S0", "S1", "S2", "S3", "S4", "S5"];
    let records = vec![SynthRecord {
        pos: 500,
        ref_allele: b"ATATC",
        alts: vec![&b"A"[..]],
        gt: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], // S5 carries neither hap
    }];
    build_contig(&out, "chr1", &samples, 2, &records);
    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 6, 2).unwrap();

    let regions = [(0u32, 1000u32)];
    let batch = overlap_batch(&reader, &regions);

    // The DEL is dense, so it sits in the dense channel: carriers see popcount 1,
    // S5 (gt 0,0) sees 0. Cross-check against decode_hap's length.
    for s in 0..6usize {
        for p in 0..2usize {
            let want = batch.decode_hap(&reader, 0, s, p).positions.len();
            assert_eq!(dense_popcount(&batch, 0, s, p), want, "s={s} p={p}");
        }
    }
    // Explicit: S5 carries nothing here.
    assert_eq!(dense_popcount(&batch, 0, 5, 0), 0);
    assert_eq!(dense_popcount(&batch, 0, 5, 1), 0);
}

proptest! {
    // Heavy: each case runs the full converter. Low case count on purpose.
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn prop_overlap_batch_matches_overlap_sample(
        records in arb_records(6), // 3 samples, diploid
        starts in proptest::collection::vec(0u32..1200, 1..4),
        q_len in 1u32..300,
    ) {
        let n_samples = 3usize;
        let ploidy = 2usize;
        let sample_names = ["S0", "S1", "S2"];

        let synth: Vec<SynthRecord> = records
            .iter()
            .map(|r: &OwnedRecord| SynthRecord {
                pos: r.pos,
                ref_allele: &r.ref_allele,
                alts: vec![&r.alt[..]],
                gt: r.gt.clone(),
            })
            .collect();

        let tmp = tempdir().unwrap();
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        build_contig(&out, "chr1", &sample_names, ploidy, &synth);
        let reader = ContigReader::open(out.to_str().unwrap(), "chr1", n_samples, ploidy).unwrap();

        let regions: Vec<(u32, u32)> = starts.iter().map(|&s| (s, s + q_len)).collect();
        let batch = overlap_batch(&reader, &regions);

        for (r, &(qs, qe)) in regions.iter().enumerate() {
            for s in 0..n_samples {
                let single = overlap_sample(&reader, s, qs, qe);
                for p in 0..ploidy {
                    prop_assert_eq!(
                        batch.decode_hap(&reader, r, s, p),
                        single.per_hap[p].clone(),
                        "r={} s={} p={}", r, s, p
                    );
                }
            }
        }
    }
}

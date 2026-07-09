//! Disk-integration tests for the batched M6.1 spine (`overlap_batch`). Builds
//! finished SVAR2 contigs via the real converter, then cross-checks the two-
//! channel batch result against the M5 single-sample `overlap_sample` — the
//! already-oracle-tested reference — for every (region, sample, ploid), plus a
//! decode-free dense-count sanity check.

mod common;

use common::{OwnedRecord, SynthRecord, arb_records, build_contig};
use genoray_core::query::oracle::overlap_sample;
use genoray_core::query::{ContigReader, overlap_batch};
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

/// Regression for the spanning-deletion overlap bug: `overlap_range`'s window
/// endpoints truly overlap the query, but a spanning deletion earlier in the
/// run can widen the window's left bound and pull in a later variant whose
/// `v_end <= q_start` (it ends before the query starts). This builds a contig
/// where a 20-base DEL@100 (v_end 121) spans a SNP@105 (v_end 106, ends before
/// q_start=110 — must be EXCLUDED) and a SNP@112 (v_end 113, truly overlaps
/// [110, 115) — must be INCLUDED), all common enough to route to the dense
/// channel, and checks both `overlap_sample` and `overlap_batch`/`decode_hap`
/// exclude the spurious SNP@105.
#[test]
fn test_dense_channel_excludes_interior_non_overlap_behind_spanning_deletion() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    // DEL@100: anchor 'A' + 20-base tail, alt = anchor only -> ilen -20, v_end
    // 121. Non-repetitive tail bases (all 'C' except two positions that must
    // agree with the SNPs' REF below) so the converter's left-alignment never
    // rolls the deletion: `left_align` only rolls while
    // `ref_seq[pos] == ref_seq[pos + ndel]`, and anchor 'A' != tail's last
    // base 'C' (pos 120), so the roll check fails immediately regardless of
    // the interior bytes.
    let mut tail = vec![b'C'; 20];
    tail[4] = b'G'; // offset 5 -> pos 105, must match SNP_A's REF below
    tail[11] = b'T'; // offset 12 -> pos 112, must match SNP_B's REF below
    let mut del_ref = vec![b'A'];
    del_ref.extend_from_slice(&tail);

    // All four haps (2 samples, diploid) carry every variant here: x=4 makes
    // both SNP (dense 34+4=38 < var_key 34*4=136) and DEL (dense
    // 32+32+4=68 < var_key 64*4=256) strictly cheaper as dense
    // (`cost_model::choose_representation`), so all three route to the dense
    // channel and this exercises the `dense_carried`/`overlap_batch`
    // presence-bit fix, not the var_key fix (that's covered by
    // `spine::tests::test_gather_keys_excludes_interior_non_overlap_behind_spanning_deletion`).
    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: &del_ref,
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 1, 1],
        },
        SynthRecord {
            pos: 105,
            ref_allele: b"G",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 1, 1],
        },
        SynthRecord {
            pos: 112,
            ref_allele: b"T",
            alts: vec![&b"C"[..]],
            gt: vec![1, 1, 1, 1],
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);

    // Confirm all three actually routed to dense (not var_key) so this test
    // exercises the channel it claims to.
    let dense_snp_positions = common::read_u32_bin(
        &out.join("chr1")
            .join("dense")
            .join("snp")
            .join("positions.bin"),
    );
    assert_eq!(
        dense_snp_positions,
        vec![105, 112],
        "expected both SNPs to route to dense/snp"
    );
    let dense_indel_positions = common::read_u32_bin(
        &out.join("chr1")
            .join("dense")
            .join("indel")
            .join("positions.bin"),
    );
    assert_eq!(
        dense_indel_positions,
        vec![100],
        "expected the DEL to route to dense/indel"
    );

    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap();

    // overlap_sample: query [110, 115) spans SNP@105 (excluded) and truly
    // overlaps DEL@100 and SNP@112 (included).
    let r = overlap_sample(&reader, 0, 110, 115);
    for (p, hc) in r.per_hap.iter().enumerate() {
        assert_eq!(hc.positions, vec![100, 112], "hap {p} positions");
        assert_eq!(hc.ilens, vec![-20, 0], "hap {p} ilens");
        assert_eq!(
            hc.alts,
            vec![Vec::<u8>::new(), b"C".to_vec()],
            "hap {p} alts"
        );
    }

    // overlap_batch/decode_hap must agree.
    let batch = genoray_core::query::overlap_batch(&reader, &[(110u32, 115u32)]);
    for s in 0..2usize {
        for p in 0..2usize {
            let got = batch.decode_hap(&reader, 0, s, p);
            assert_eq!(got.positions, vec![100, 112], "s={s} p={p} positions");
            assert_eq!(got.ilens, vec![-20, 0], "s={s} p={p} ilens");
            assert_eq!(
                got.alts,
                vec![Vec::<u8>::new(), b"C".to_vec()],
                "s={s} p={p} alts"
            );
        }
    }
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

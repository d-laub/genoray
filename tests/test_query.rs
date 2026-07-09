//! Disk-integration tests for the `(range, sample)` query. Builds finished SVAR2
//! contigs via the real conversion pipeline, then overwrites `max_del.npy` with a
//! deliberately conservative fixture: an over-estimate is provably overshoot-safe
//! (see `search.rs`), so the consumer's answers are identical to the tight bound
//! the wired-in producer writes, while the fixture keeps the test independent of it.

mod common;

use proptest::prelude::*;

use common::{OwnedRecord, SynthRecord, arb_records, build_contig};
use genoray_core::query::ContigReader;
use genoray_core::query::oracle::overlap_sample;
use tempfile::tempdir;

#[test]
fn test_open_ok_and_missing_dirs_tolerated() {
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
            pos: 300,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 0, 0],
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    assert!(ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).is_ok());

    // A bare contig dir with no sub-streams at all still opens (everything empty).
    let empty = tmp.path().join("empty_out");
    std::fs::create_dir_all(empty.join("chrX")).unwrap();
    assert!(ContigReader::open(empty.to_str().unwrap(), "chrX", 1, 2).is_ok());
}

// Mirrors the shape of `test_e2e_normalized_bcf_pipeline`: SNP@100 (-> dense/snp,
// x=3), INS@200 (-> var_key/indel, x=1), DEL@300 (-> dense/indel, x=2). The query
// unions across those three sub-streams.
#[test]
fn test_overlap_sample_known_contig() {
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

    // Sample 0, whole contig.
    let r = overlap_sample(&reader, 0, 0, 1000);
    assert_eq!(r.per_hap.len(), 2);
    // hap 0 (S0_p0): SNP@100 (gt 1) + DEL@300 (gt 1); INS@200 gt 0.
    assert_eq!(r.per_hap[0].positions, vec![100, 300]);
    assert_eq!(r.per_hap[0].ilens, vec![0, -1]);
    assert_eq!(r.per_hap[0].alts, vec![b"C".to_vec(), Vec::<u8>::new()]);
    // hap 1 (S0_p1): INS@200 (gt 1) + DEL@300 (gt 1); SNP@100 gt 0.
    assert_eq!(r.per_hap[1].positions, vec![200, 300]);
    assert_eq!(r.per_hap[1].ilens, vec![1, -1]);
    assert_eq!(r.per_hap[1].alts, vec![b"AT".to_vec(), Vec::<u8>::new()]);

    // Sample 1: only the SNP@100 (gt for haps 2,3 = 1,1).
    let r1 = overlap_sample(&reader, 1, 0, 1000);
    assert_eq!(r1.per_hap[0].positions, vec![100]);
    assert_eq!(r1.per_hap[1].positions, vec![100]);

    // Deletion spanning the query start: [301, 302) still returns DEL@300
    // (v_end = 300 + 1 + 1 = 302). Exercises the dense/indel max_del path.
    let r2 = overlap_sample(&reader, 0, 301, 302);
    assert_eq!(r2.per_hap[0].positions, vec![300]);
    assert_eq!(r2.per_hap[1].positions, vec![300]);

    // No overlap: gap between variants.
    let r3 = overlap_sample(&reader, 0, 150, 160);
    assert!(r3.per_hap[0].positions.is_empty());
    assert!(r3.per_hap[1].positions.is_empty());
}

// Regression guard for the var_key/snp decode path: `overlap_sample` must unpack
// the 2-bit SNP code at the ABSOLUTE index `o0 + i` into the whole-stream packed
// keys buffer, not the local (per-column) index `i`. A fixture where every
// var_key/snp column happens to start at CSR offset 0 (as in
// `test_overlap_sample_known_contig`) can't catch a regression to local `i` --
// this test forces a nonzero offset for the second column.
//
// Two singleton-carrier (x=1) SNPs always route to var_key/snp regardless of
// cohort size (see `cost_model::choose_representation`: var_key_bits = 34 <=
// dense_bits = 34+np for every np >= 0). SNP_A@100 (ALT C, code 1) is carried
// only by S0's hap p0 (flat col 0); SNP_B@200 (ALT G, code 3) only by S0's hap
// p1 (flat col 1). The sample-major transpose fills var_key/snp column-by-
// column, so col 0's CSR range is [0,1) and col 1's is [1,2): querying hap p1
// unpacks the packed-key buffer at absolute index `o0+i = 1+0 = 1`. Under a
// regression to local `i` it would instead read index 0 -- col 0's code (`C`)
// -- and decode SNP_B's ALT as `C` instead of `G`.
#[test]
fn test_overlap_sample_var_key_snp_nonzero_csr_offset() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0], // hap p0 only
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![0, 1], // hap p1 only
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);

    // Confirm both SNPs actually routed to var_key/snp (not dense/snp): the
    // var_key/snp positions sidecar must hold both calls, and dense/snp must be
    // absent/empty.
    let vk_snp_positions = out
        .join("chr1")
        .join("var_key")
        .join("snp")
        .join("positions.bin");
    let vk_positions = common::read_u32_bin(&vk_snp_positions);
    assert_eq!(
        vk_positions,
        vec![100, 200],
        "expected both SNPs to route to var_key/snp"
    );
    let dense_snp_positions = out
        .join("chr1")
        .join("dense")
        .join("snp")
        .join("positions.bin");
    assert!(
        !dense_snp_positions.exists()
            || std::fs::metadata(&dense_snp_positions).unwrap().len() == 0,
        "expected dense/snp to be empty"
    );

    // Sanity-check the CSR offsets directly: col 0 -> [0,1), col 1 -> [1,2).
    let offsets = common::read_offsets_npy(
        &out.join("chr1")
            .join("var_key")
            .join("snp")
            .join("offsets.npy"),
    );
    assert_eq!(offsets, vec![0, 1, 2]);

    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 1, 2).unwrap();
    let r = overlap_sample(&reader, 0, 0, 1000);
    assert_eq!(r.per_hap.len(), 2);

    // hap p0 (col 0, o0=0): SNP_A@100, ALT C.
    assert_eq!(r.per_hap[0].positions, vec![100]);
    assert_eq!(r.per_hap[0].alts, vec![b"C".to_vec()]);

    // hap p1 (col 1, o0=1): SNP_B@200, ALT G. This is the assertion the
    // local-`i` regression breaks: it would decode col 0's code and return `C`.
    assert_eq!(r.per_hap[1].positions, vec![200]);
    assert_eq!(r.per_hap[1].alts, vec![b"G".to_vec()]);
}

// A variant routed to dense vs. var_key must give identical query results for the
// carrying sample — routing is an internal storage choice, invisible to queries.
// A deletion carried by 1 sample (rare -> var_key/indel) vs. by many samples
// (common -> dense/indel) must both come back with the same (pos, ilen, alt).
#[test]
fn test_routing_invariant_dense_vs_var_key() {
    // Rare: 6 samples, only S0_p0 carries the DEL -> var_key/indel.
    let rare = {
        let tmp = tempdir().unwrap();
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let samples = ["S0", "S1", "S2", "S3", "S4", "S5"];
        let records = vec![SynthRecord {
            pos: 500,
            ref_allele: b"ATATC", // 4-base deletion
            alts: vec![&b"A"[..]],
            gt: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }];
        build_contig(&out, "chr1", &samples, 2, &records);
        let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 6, 2).unwrap();
        let r = overlap_sample(&reader, 0, 0, 1000);
        (
            r.per_hap[0].positions.clone(),
            r.per_hap[0].ilens.clone(),
            r.per_hap[0].alts.clone(),
        )
    };

    // Common: same DEL carried by nearly everyone -> dense/indel.
    let common = {
        let tmp = tempdir().unwrap();
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let samples = ["S0", "S1", "S2", "S3", "S4", "S5"];
        let records = vec![SynthRecord {
            pos: 500,
            ref_allele: b"ATATC",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        }];
        build_contig(&out, "chr1", &samples, 2, &records);
        let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 6, 2).unwrap();
        let r = overlap_sample(&reader, 0, 0, 1000);
        (
            r.per_hap[0].positions.clone(),
            r.per_hap[0].ilens.clone(),
            r.per_hap[0].alts.clone(),
        )
    };

    assert_eq!(rare, common);
    assert_eq!(rare.0, vec![500]);
    assert_eq!(rare.1, vec![-4]); // 5-base ref, 1-base alt
    assert_eq!(rare.2, vec![Vec::<u8>::new()]);
}

// A pure-SNP contig (all-zero max_del, possibly no dense/indel or var_key/indel
// dir) queries correctly, and a query outside all variants is empty.
#[test]
fn test_pure_snp_contig_and_no_overlap() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 1],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"G",
            alts: vec![&b"T"[..]],
            gt: vec![1, 0],
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 1, 2).unwrap();

    let r = overlap_sample(&reader, 0, 0, 1000);
    assert_eq!(r.per_hap[0].positions, vec![100, 200]);
    assert_eq!(r.per_hap[0].ilens, vec![0, 0]);
    assert_eq!(r.per_hap[1].positions, vec![100]); // hap 1 carries only the SNP@100

    // Entirely-left and entirely-right queries are empty.
    let left = overlap_sample(&reader, 0, 0, 50);
    assert!(left.per_hap[0].positions.is_empty());
    let right = overlap_sample(&reader, 0, 900, 1000);
    assert!(right.per_hap[0].positions.is_empty());
}

/// Brute-force reference: for `sample`, per hap, the carried variants overlapping
/// `[q_start, q_end)` in position order. `alt` matches the query contract — the
/// ALT bases for SNP/INS, empty for a pure DEL.
#[allow(clippy::type_complexity)]
fn oracle(
    records: &[OwnedRecord],
    sample: usize,
    ploidy: usize,
    q_start: u32,
    q_end: u32,
) -> Vec<(Vec<u32>, Vec<i32>, Vec<Vec<u8>>)> {
    let mut per_hap: Vec<(Vec<u32>, Vec<i32>, Vec<Vec<u8>>)> =
        vec![(Vec::new(), Vec::new(), Vec::new()); ploidy];
    for rec in records {
        let pos = rec.pos as u32;
        let ilen = rec.alt.len() as i32 - rec.ref_allele.len() as i32;
        let del = if ilen < 0 { (-ilen) as u32 } else { 0 };
        let v_end = pos + 1 + del;
        if !(pos < q_end && q_start < v_end) {
            continue;
        }
        let alt = if ilen < 0 {
            Vec::new()
        } else {
            rec.alt.clone()
        };
        #[allow(clippy::needless_range_loop)]
        for p in 0..ploidy {
            let hap = sample * ploidy + p;
            if rec.gt[hap] == 1 {
                per_hap[p].0.push(pos);
                per_hap[p].1.push(ilen);
                per_hap[p].2.push(alt.clone());
            }
        }
    }
    per_hap // records are position-sorted, so each hap's lists already are too
}

proptest! {
    // Heavy: each case runs the full converter (BCF write + index + pipeline), so
    // the case count is deliberately low. Not a silent cap — 24 random contigs,
    // each queried for every sample, is the primary correctness gate for the
    // dense/var_key union.
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn prop_overlap_sample_matches_oracle(
        records in arb_records(6), // 3 samples, diploid -> 6 haps
        q_start in 0u32..1200,
        q_len in 1u32..300,
    ) {
        let n_samples = 3;
        let ploidy = 2;
        let sample_names = ["S0", "S1", "S2"];

        let synth: Vec<SynthRecord> = records
            .iter()
            .map(|r| SynthRecord {
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

        let q_end = q_start + q_len;
        for s in 0..n_samples {
            let got = overlap_sample(&reader, s, q_start, q_end);
            let want = oracle(&records, s, ploidy, q_start, q_end);
            #[allow(clippy::needless_range_loop)]
            for p in 0..ploidy {
                prop_assert_eq!(&got.per_hap[p].positions, &want[p].0, "s={} p={} positions", s, p);
                prop_assert_eq!(&got.per_hap[p].ilens, &want[p].1, "s={} p={} ilens", s, p);
                prop_assert_eq!(&got.per_hap[p].alts, &want[p].2, "s={} p={} alts", s, p);
            }
        }
    }
}

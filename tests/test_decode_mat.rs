//! M6c: decode_batch per-hap slices equal the M5 overlap_sample oracle, and
//! region_counts equals the decoded per-hap lengths (the count shortcut).

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::py_query::PyContigReader;
use genoray_core::query::ContigReader;
use genoray_core::query::oracle::overlap_sample;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDictMethods;
use tempfile::tempdir;

#[test]
fn test_decode_batch_matches_overlap_sample_and_counts() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
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
    build_contig(&out, "chr1", &samples, 2, &records);
    let base = out.to_str().unwrap();
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];
    let (ns, pl) = (2usize, 2usize);

    let cr = ContigReader::open(base, "chr1", ns, pl).unwrap();

    Python::attach(|py| {
        let reader = PyContigReader::new(base, "chr1", ns, pl).unwrap();
        let d = reader.decode_batch(py, regions.clone()).unwrap();

        let get_i32 = |k: &str| {
            let o = d.get_item(k).unwrap().unwrap();
            o.cast::<PyArray1<i32>>()
                .unwrap()
                .readonly()
                .as_slice()
                .unwrap()
                .to_vec()
        };
        let get_i64 = |k: &str| {
            let o = d.get_item(k).unwrap().unwrap();
            o.cast::<PyArray1<i64>>()
                .unwrap()
                .readonly()
                .as_slice()
                .unwrap()
                .to_vec()
        };
        let pos = get_i32("pos");
        let ilen = get_i32("ilen");
        let off = get_i64("off");

        // Per-hap slice equals overlap_sample oracle for that (region, sample, ploid).
        for (r, &(qs, qe)) in regions.iter().enumerate() {
            for s in 0..ns {
                let qr = overlap_sample(&cr, s, qs, qe);
                for p in 0..pl {
                    let h = (r * ns + s) * pl + p;
                    let (a, b) = (off[h] as usize, off[h + 1] as usize);
                    let hc = &qr.per_hap[p];
                    let want_pos: Vec<i32> = hc.positions.iter().map(|&x| x as i32).collect();
                    assert_eq!(&pos[a..b], &want_pos[..], "pos hap {h}");
                    assert_eq!(&ilen[a..b], &hc.ilens[..], "ilen hap {h}");
                }
            }
        }

        // Count shortcut: region_counts == per-hap decoded lengths.
        let counts_obj = reader.region_counts(py, regions.clone()).unwrap();
        let counts = counts_obj.readonly();
        let counts = counts.as_slice().unwrap();
        for h in 0..(regions.len() * ns * pl) {
            let decoded_len = off[h + 1] - off[h];
            assert_eq!(counts[h], decoded_len, "count hap {h}");
        }
    });
}

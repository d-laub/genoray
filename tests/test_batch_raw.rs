//! M6b: the `PyContigReader.overlap_batch` numpy dict is a faithful image of the
//! pure `query::overlap_batch` BatchResult (the already-oracle-tested reference).

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::py_query::PyContigReader;
use genoray_core::query::{ContigReader, overlap_batch};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDictMethods;
use tempfile::tempdir;

fn i32_slice<'py>(d: &Bound<'py, pyo3::types::PyDict>, k: &str) -> Vec<i32> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.cast::<PyArray1<i32>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}
fn i64_slice<'py>(d: &Bound<'py, pyo3::types::PyDict>, k: &str) -> Vec<i64> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.cast::<PyArray1<i64>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}
fn u8_slice<'py>(d: &Bound<'py, pyo3::types::PyDict>, k: &str) -> Vec<u8> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.cast::<PyArray1<u8>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}

#[test]
fn test_overlap_batch_dict_matches_oracle() {
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

    // Oracle: pure Rust BatchResult.
    let cr = ContigReader::open(base, "chr1", 2, 2).unwrap();
    let br = overlap_batch(&cr, &regions);

    Python::attach(|py| {
        let reader = PyContigReader::new(base, "chr1", 2, 2).unwrap();
        let d = reader.overlap_batch(py, regions.clone()).unwrap();

        assert_eq!(
            i32_slice(&d, "vk_pos"),
            br.vk.iter().map(|k| k.position as i32).collect::<Vec<_>>()
        );
        assert_eq!(
            i32_slice(&d, "vk_key"),
            br.vk.iter().map(|k| k.key as i32).collect::<Vec<_>>()
        );
        assert_eq!(
            i64_slice(&d, "vk_off"),
            br.vk_off.iter().map(|&x| x as i64).collect::<Vec<_>>()
        );
        assert_eq!(
            i32_slice(&d, "dense_pos"),
            br.dense
                .iter()
                .map(|k| k.position as i32)
                .collect::<Vec<_>>()
        );
        assert_eq!(u8_slice(&d, "dense_present"), br.dense_present);
        assert_eq!(
            i64_slice(&d, "dense_present_off"),
            br.dense_present_off
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>()
        );
    });
}

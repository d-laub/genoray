//! SVAR2 search/gather split: numpy-dict bindings on `PyContigReader` for the
//! `query::find_ranges`/`gather_ranges`/`read_ranges` core (Tasks 1-3). Separate
//! `#[pymethods]` block (multiple-pymethods) so the M6b `overlap_batch` binding
//! in `py_query_batch.rs` is untouched.
//!
//! `read_ranges`/`gather_ranges` produce the exact same dict contract as
//! `overlap_batch` (see `py_query_batch.rs`): `vk_pos, vk_key, vk_off, dense_pos,
//! dense_key, dense_range, dense_present, dense_present_off, lut_bytes, lut_off,
//! n_regions, n_samples, ploidy`. `find_ranges` returns the compact `RangesBundle`
//! dict instead; `gather_ranges` consumes one to replay a `BatchResult`.
//!
//! `out=` streaming (writing directly into a caller-provided memmap) is
//! deferred to the Python layer (Task 5) — `find_ranges` here always returns
//! freshly-allocated numpy arrays.

use std::ops::Range;

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::py_convert::{u8_to_pyarray, u32_to_i32_pyarray, usize_to_i64_pyarray};
use crate::py_query::PyContigReader;
use crate::query::{BatchResult, RangesBundle, find_ranges, gather_ranges, read_ranges};

/// Identical to `py_query_batch.rs::overlap_batch`'s dict assembly — the whole
/// point of the search/gather split is that `read_ranges`/`gather_ranges`
/// produce the same numpy contract as `overlap_batch`.
fn batch_result_to_dict<'py>(
    py: Python<'py>,
    reader_lut: (Vec<u8>, Vec<u64>),
    br: &BatchResult,
) -> PyResult<Bound<'py, PyDict>> {
    let vk_pos: Vec<u32> = br.vk.iter().map(|k| k.position).collect();
    let vk_key: Vec<u32> = br.vk.iter().map(|k| k.key).collect();
    let dense_pos: Vec<u32> = br.dense.iter().map(|k| k.position).collect();
    let dense_key: Vec<u32> = br.dense.iter().map(|k| k.key).collect();

    // dense_range as [R, 2] i32.
    let r = br.dense_range.len();
    let mut dr: Vec<i32> = Vec::with_capacity(r * 2);
    for range in &br.dense_range {
        dr.push(range.start as i32);
        dr.push(range.end as i32);
    }
    let dense_range = Array2::from_shape_vec((r, 2), dr)
        .expect("dense_range shape")
        .to_pyarray(py);

    let (lut_bytes, lut_off_u64) = reader_lut;
    let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();

    let d = PyDict::new(py);
    d.set_item("vk_pos", u32_to_i32_pyarray(py, &vk_pos))?;
    d.set_item("vk_key", u32_to_i32_pyarray(py, &vk_key))?;
    d.set_item("vk_off", usize_to_i64_pyarray(py, &br.vk_off))?;
    d.set_item("dense_pos", u32_to_i32_pyarray(py, &dense_pos))?;
    d.set_item("dense_key", u32_to_i32_pyarray(py, &dense_key))?;
    d.set_item("dense_range", dense_range)?;
    d.set_item("dense_present", u8_to_pyarray(py, &br.dense_present))?;
    d.set_item(
        "dense_present_off",
        usize_to_i64_pyarray(py, &br.dense_present_off),
    )?;
    d.set_item("lut_bytes", u8_to_pyarray(py, &lut_bytes))?;
    d.set_item("lut_off", PyArray1::from_slice(py, &lut_off))?;
    d.set_item("n_regions", br.n_regions)?;
    d.set_item("n_samples", br.n_samples)?;
    d.set_item("ploidy", br.ploidy)?;
    Ok(d)
}

/// Compact `RangesBundle` -> numpy dict: `dense_range (R,2) i32`, `region_starts
/// (R) i32`, `sample_cols (n_samples) i64`, `vk_snp_range (R*H,2) i64`,
/// `vk_indel_range (R*H,2) i64`, `dense_snp_range (R,2) i32`, `dense_indel_range
/// (R,2) i32`, plus `n_regions`/`n_samples`/`ploidy` scalars (H =
/// n_samples*ploidy).
fn bundle_to_dict<'py>(py: Python<'py>, rb: &RangesBundle) -> PyResult<Bound<'py, PyDict>> {
    let pairs_to_i32_flat = |v: &[Range<usize>]| -> Vec<i32> {
        let mut o = Vec::with_capacity(v.len() * 2);
        for r in v {
            o.push(r.start as i32);
            o.push(r.end as i32);
        }
        o
    };
    let pairs_to_i64_flat = |v: &[Range<usize>]| -> Vec<i64> {
        let mut o = Vec::with_capacity(v.len() * 2);
        for r in v {
            o.push(r.start as i64);
            o.push(r.end as i64);
        }
        o
    };

    let dr = pairs_to_i32_flat(&rb.dense_range);
    let dense_range = Array2::from_shape_vec((rb.n_regions, 2), dr)
        .expect("dense_range shape")
        .to_pyarray(py);

    let dsr = pairs_to_i32_flat(&rb.dense_snp_range);
    let dense_snp_range = Array2::from_shape_vec((rb.n_regions, 2), dsr)
        .expect("dense_snp_range shape")
        .to_pyarray(py);
    let dir_ = pairs_to_i32_flat(&rb.dense_indel_range);
    let dense_indel_range = Array2::from_shape_vec((rb.n_regions, 2), dir_)
        .expect("dense_indel_range shape")
        .to_pyarray(py);

    let h = rb.n_samples * rb.ploidy;
    let vk_snp = pairs_to_i64_flat(&rb.vk_snp_range);
    let vk_snp_range = Array2::from_shape_vec((rb.n_regions * h, 2), vk_snp)
        .expect("vk_snp_range shape")
        .to_pyarray(py);
    let vk_indel = pairs_to_i64_flat(&rb.vk_indel_range);
    let vk_indel_range = Array2::from_shape_vec((rb.n_regions * h, 2), vk_indel)
        .expect("vk_indel_range shape")
        .to_pyarray(py);

    let sample_cols: Vec<i64> = rb.sample_cols.iter().map(|&x| x as i64).collect();

    let d = PyDict::new(py);
    d.set_item("dense_range", dense_range)?;
    d.set_item("region_starts", u32_to_i32_pyarray(py, &rb.region_starts))?;
    d.set_item("sample_cols", PyArray1::from_slice(py, &sample_cols))?;
    d.set_item("vk_snp_range", vk_snp_range)?;
    d.set_item("vk_indel_range", vk_indel_range)?;
    d.set_item("dense_snp_range", dense_snp_range)?;
    d.set_item("dense_indel_range", dense_indel_range)?;
    d.set_item("n_regions", rb.n_regions)?;
    d.set_item("n_samples", rb.n_samples)?;
    d.set_item("ploidy", rb.ploidy)?;
    Ok(d)
}

/// Inverse of `bundle_to_dict`: read a `find_ranges` dict back into a
/// `RangesBundle` for `gather_ranges`. Fallible: a missing key or wrong
/// dtype/shape becomes a Python KeyError/TypeError rather than a Rust panic.
fn bundle_from_dict(d: &Bound<'_, PyDict>) -> PyResult<RangesBundle> {
    let require = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(k)?
            .ok_or_else(|| PyKeyError::new_err(format!("bundle missing key '{k}'")))
    };
    let get_i32 = |k: &str| -> PyResult<Vec<i32>> {
        let obj = require(k)?;
        let arr = obj.cast::<PyArray1<i32>>().map_err(|_| {
            PyTypeError::new_err(format!("bundle key '{k}' must be an int32 1D array"))
        })?;
        Ok(arr.readonly().as_slice()?.to_vec())
    };
    let get_i64 = |k: &str| -> PyResult<Vec<i64>> {
        let obj = require(k)?;
        let arr = obj.cast::<PyArray1<i64>>().map_err(|_| {
            PyTypeError::new_err(format!("bundle key '{k}' must be an int64 1D array"))
        })?;
        Ok(arr.readonly().as_slice()?.to_vec())
    };
    let get_i32_pairs = |k: &str| -> PyResult<Vec<Range<usize>>> {
        let obj = require(k)?;
        let arr = obj
            .cast::<PyArray2<i32>>()
            .map_err(|_| {
                PyTypeError::new_err(format!("bundle key '{k}' must be an int32 (N,2) array"))
            })?
            .readonly();
        Ok(arr
            .as_array()
            .rows()
            .into_iter()
            .map(|row| (row[0] as usize)..(row[1] as usize))
            .collect())
    };
    let get_i64_pairs = |k: &str| -> PyResult<Vec<Range<usize>>> {
        let obj = require(k)?;
        let arr = obj
            .cast::<PyArray2<i64>>()
            .map_err(|_| {
                PyTypeError::new_err(format!("bundle key '{k}' must be an int64 (N,2) array"))
            })?
            .readonly();
        Ok(arr
            .as_array()
            .rows()
            .into_iter()
            .map(|row| (row[0] as usize)..(row[1] as usize))
            .collect())
    };
    let get_usize = |k: &str| -> PyResult<usize> { require(k)?.extract() };

    Ok(RangesBundle {
        n_regions: get_usize("n_regions")?,
        n_samples: get_usize("n_samples")?,
        ploidy: get_usize("ploidy")?,
        region_starts: get_i32("region_starts")?
            .into_iter()
            .map(|x| x as u32)
            .collect(),
        dense_range: get_i32_pairs("dense_range")?,
        sample_cols: get_i64("sample_cols")?
            .into_iter()
            .map(|x| x as usize)
            .collect(),
        vk_snp_range: get_i64_pairs("vk_snp_range")?,
        vk_indel_range: get_i64_pairs("vk_indel_range")?,
        dense_snp_range: get_i32_pairs("dense_snp_range")?,
        dense_indel_range: get_i32_pairs("dense_indel_range")?,
    })
}

#[pymethods]
impl PyContigReader {
    /// Search + gather in one call, returning the same dict contract as
    /// `overlap_batch` (see `py_query_batch.rs`). `samples`, if given, selects
    /// (and reorders) a sample subset by original index.
    pub fn read_ranges<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let br = read_ranges(&self.inner, &regions, samples.as_deref());
        batch_result_to_dict(py, self.inner.lut_arrays(), &br)
    }

    /// Search-only: returns a compact `RangesBundle` dict (index ranges into the
    /// dense union and var_key sub-streams) with no per-element gather. No `out=`
    /// param — streaming into a caller buffer is a Python-layer (Task 5) concern;
    /// this always returns fresh arrays.
    pub fn find_ranges<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rb = find_ranges(&self.inner, &regions, samples.as_deref());
        bundle_to_dict(py, &rb)
    }

    /// Tree-free gather: replay a `find_ranges` bundle dict into the same dict
    /// contract as `overlap_batch`/`read_ranges`. No `samples` param — the bundle
    /// already fixes the sample subset via `find_ranges`.
    pub fn gather_ranges<'py>(
        &self,
        py: Python<'py>,
        bundle: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rb = bundle_from_dict(&bundle)?;
        let br = gather_ranges(&self.inner, &rb);
        batch_result_to_dict(py, self.inner.lut_arrays(), &br)
    }
}

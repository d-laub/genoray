//! M6b: raw two-channel `BatchResult` → numpy exposure on `PyContigReader`.
//! Owned by the `svar-2-m6b` worktree; separate `#[pymethods]` block
//! (multiple-pymethods) so M6b and M6c never touch the same file.

use ndarray::Array2;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_convert::{u8_to_pyarray, u32_to_i32_pyarray, usize_to_i64_pyarray};
use crate::py_query::PyContigReader;
use crate::query::overlap_batch;

#[pymethods]
impl PyContigReader {
    /// Batched two-channel query over `regions` (half-open `[q_start, q_end)`)
    /// within this contig. Returns the frozen `BatchResult → numpy` contract as a
    /// dict keyed by the contract array names, plus `n_regions`/`n_samples`/`ploidy`.
    ///
    /// `pub` so the external integration-test crate can call it as a plain Rust
    /// method (same reason `PyContigReader::new` is `pub`); pyo3 still exposes it
    /// to Python via `#[pymethods]`.
    pub fn overlap_batch<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let br = overlap_batch(&self.inner, &regions);

        let vk_pos: Vec<u32> = br.vk.iter().map(|k| k.position).collect();
        let vk_key: Vec<u32> = br.vk.iter().map(|k| k.key).collect();
        let dense_pos: Vec<u32> = br.dense.iter().map(|k| k.position).collect();
        let dense_key: Vec<u32> = br.dense.iter().map(|k| k.key).collect();

        // dense_range as [R, 2] i32.
        let r = br.dense_range.len();
        let mut dr: Vec<i32> = Vec::with_capacity(r * 2);
        for &(s, e) in &br.dense_range {
            dr.push(s as i32);
            dr.push(e as i32);
        }
        let dense_range = Array2::from_shape_vec((r, 2), dr)
            .expect("dense_range shape")
            .to_pyarray(py);

        let (lut_bytes, lut_off_u64) = self.inner.lut_arrays();
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
}

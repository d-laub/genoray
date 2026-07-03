//! M6c: decoded `seqpro.rag.Ragged` materialization + decode-free counts on
//! `PyContigReader`. Owned by the `svar-2-m6c` worktree; separate `#[pymethods]`
//! block (multiple-pymethods) so M6b and M6c never touch the same file.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_convert::{i32_to_pyarray, u8_to_pyarray};
use crate::py_query::PyContigReader;
use crate::query::overlap_batch;

#[pymethods]
impl PyContigReader {
    /// Decode every hap's overlapping variants across `regions` into flat, shared
    /// variant-axis buffers (the M6c materialization primitive). Hap order is
    /// region-major `h = (r*n_samples + s)*ploidy + p`.
    fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let br = overlap_batch(&self.inner, &regions);
        let mut pos: Vec<i32> = Vec::new();
        let mut ilen: Vec<i32> = Vec::new();
        let mut allele: Vec<u8> = Vec::new();
        let mut str_off: Vec<i64> = vec![0];
        let mut off: Vec<i64> = vec![0];

        for r in 0..br.n_regions {
            for s in 0..br.n_samples {
                for p in 0..br.ploidy {
                    let hc = br.decode_hap(&self.inner, r, s, p);
                    for i in 0..hc.positions.len() {
                        pos.push(hc.positions[i] as i32);
                        ilen.push(hc.ilens[i]);
                        allele.extend_from_slice(&hc.alts[i]);
                        str_off.push(allele.len() as i64);
                    }
                    off.push(pos.len() as i64);
                }
            }
        }

        let d = PyDict::new(py);
        d.set_item("pos", i32_to_pyarray(py, &pos))?;
        d.set_item("ilen", i32_to_pyarray(py, &ilen))?;
        d.set_item("allele", u8_to_pyarray(py, &allele))?;
        d.set_item("str_off", PyArray1::from_slice(py, &str_off))?;
        d.set_item("off", PyArray1::from_slice(py, &off))?;
        d.set_item("n_regions", br.n_regions)?;
        d.set_item("n_samples", br.n_samples)?;
        d.set_item("ploidy", br.ploidy)?;
        Ok(d)
    }
}

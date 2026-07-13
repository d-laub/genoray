//! M6c: decoded `seqpro.rag.Ragged` materialization + decode-free counts on
//! `PyContigReader`. Owned by the `svar-2-m6c` worktree; separate `#[pymethods]`
//! block (multiple-pymethods) so M6b and M6c never touch the same file.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::bits;
use crate::py_convert::{i32_to_pyarray, u8_to_pyarray};
use crate::py_query::PyContigReader;
use crate::query::overlap_batch;

#[pymethods]
impl PyContigReader {
    /// Decode every hap's overlapping variants across `regions` into flat, shared
    /// variant-axis buffers (the M6c materialization primitive). Hap order is
    /// region-major `h = (r*n_samples + s)*ploidy + p`.
    // pub: called directly by the external integration-test crate (see PyContigReader::new)
    pub fn decode_batch<'py>(
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

    /// As `decode_batch`, plus one raw-bytes buffer per requested field, aligned
    /// 1:1 with the decoded records. `fields` is `(category, name, dtype_str)`.
    /// Values are returned as little-endian bytes + an itemsize; Python applies
    /// the numpy dtype (same trick as `allele`), so Rust does no dtype dispatch.
    #[pyo3(signature = (regions, fields, base_dir, contig))]
    pub fn decode_batch_fields<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        fields: Vec<(String, String, String)>,
        base_dir: &str,
        contig: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        use crate::field::StorageDtype;
        use crate::layout::{ContigPaths, FieldSub};
        use crate::query::{FieldView, gather::overlap_batch_src};

        let br = overlap_batch_src(&self.inner, &regions);
        let paths = ContigPaths::new(base_dir, contig);
        let n_samples_cohort = self.inner.n_samples();

        // Open the four sub-stream views per field up front.
        struct OpenField {
            key: String,
            is_format: bool,
            width: usize,
            views: [FieldView; 4], // indexed by FieldSub::all() order
        }
        let mut open: Vec<OpenField> = Vec::with_capacity(fields.len());
        for (category, name, dtype_str) in &fields {
            let dtype = StorageDtype::from_meta_str(dtype_str).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "field {name:?} has unresolved/unknown dtype {dtype_str:?}"
                ))
            })?;
            let width = dtype
                .width_bytes()
                .expect("from_meta_str rejects unresolved dtypes");
            let mut views = Vec::with_capacity(4);
            for sub in FieldSub::all() {
                views.push(
                    FieldView::open(&paths, category, name, sub, dtype, n_samples_cohort)
                        .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?,
                );
            }
            open.push(OpenField {
                key: format!("{category}/{name}"),
                is_format: category == "format",
                width,
                views: views.try_into().map_err(|_| ()).expect("exactly 4 subs"),
            });
        }

        let mut pos: Vec<i32> = Vec::new();
        let mut ilen: Vec<i32> = Vec::new();
        let mut allele: Vec<u8> = Vec::new();
        let mut str_off: Vec<i64> = vec![0];
        let mut off: Vec<i64> = vec![0];
        let mut fbytes: Vec<Vec<u8>> = vec![Vec::new(); open.len()];

        for r in 0..br.n_regions {
            for s in 0..br.n_samples {
                for p in 0..br.ploidy {
                    let (hc, srcs) = br.decode_hap_src(&self.inner, r, s, p);
                    for (i, &src) in srcs.iter().enumerate() {
                        pos.push(hc.positions[i] as i32);
                        ilen.push(hc.ilens[i]);
                        allele.extend_from_slice(&hc.alts[i]);
                        str_off.push(allele.len() as i64);

                        // FieldSub::all() order: VkSnp, VkIndel, DenseSnp, DenseIndel
                        let sub_ix = match (src.is_dense, src.is_indel) {
                            (false, false) => 0,
                            (false, true) => 1,
                            (true, false) => 2,
                            (true, true) => 3,
                        };
                        for (fi, f) in open.iter().enumerate() {
                            let view = &f.views[sub_ix];
                            // Dense FORMAT strides by the ORIGINAL cohort sample
                            // index. `s` here is already the cohort index because
                            // `overlap_batch_src` runs over the whole cohort.
                            let elem = if src.is_dense && f.is_format {
                                view.bytes_at(src.idx * n_samples_cohort + s)
                            } else {
                                view.bytes_at(src.idx)
                            };
                            debug_assert_eq!(elem.len(), f.width);
                            fbytes[fi].extend_from_slice(elem);
                        }
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
        for (fi, f) in open.iter().enumerate() {
            d.set_item(format!("field_{}", f.key), u8_to_pyarray(py, &fbytes[fi]))?;
            d.set_item(format!("field_itemsize_{}", f.key), f.width)?;
        }
        Ok(d)
    }

    /// Per-hap variant count over `regions` WITHOUT decoding: var_key slice length
    /// (`vk_off` diff) + dense-present popcount. Flat length `H`, region-major hap
    /// order; the caller reshapes to `(n_regions, n_samples, ploidy)`. The simplified
    /// `SparseVar.var_ranges` replacement.
    // pub: called directly by the external integration-test crate (see PyContigReader::new)
    pub fn region_counts<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let br = overlap_batch(&self.inner, &regions);
        let h = br.n_regions * br.n_samples * br.ploidy;
        let mut counts: Vec<i64> = Vec::with_capacity(h);
        for i in 0..h {
            let vk_n = br.vk_off[i + 1] - br.vk_off[i];
            let (b0, b1) = (br.dense_present_off[i], br.dense_present_off[i + 1]);
            let dn = (b0..b1)
                .filter(|&b| bits::get_bit(&br.dense_present, b))
                .count();
            counts.push((vk_n + dn) as i64);
        }
        Ok(PyArray1::from_slice(py, &counts))
    }
}

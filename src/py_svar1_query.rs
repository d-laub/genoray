//! Python-facing handle over an SVAR1 store's range-query core. Wraps the pure-Rust
//! `svar1_query::Svar1Reader`. Ungated, like the module it wraps.
//!
//! Mirrors `py_query.rs`/`py_query_ranges.rs`: a thin pyclass plus a numpy-dict data
//! contract (not a pyclass graph). There is no `gather_ranges` counterpart — SVAR1
//! stops at index pairs, which the caller turns into a zero-copy `Ragged` view.

use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::svar1_query::{Svar1RangesBundle, Svar1Reader, find_ranges, var_ranges};

/// An SVAR1 store opened for querying. Constructed from Python as
/// `PySvar1Reader(svar1_dir, n_samples, ploidy)`.
#[pyclass]
pub struct PySvar1Reader {
    pub(crate) inner: Svar1Reader,
}

/// `Svar1RangesBundle` -> numpy dict: `starts`/`stops` (each `(r*s*p,)` int64,
/// C-order `(range, sample, ploid)`), `sample_cols` `(s,)` int64, plus
/// `n_ranges`/`n_samples`/`ploidy` scalars.
fn bundle_to_dict<'py>(py: Python<'py>, b: &Svar1RangesBundle) -> PyResult<Bound<'py, PyDict>> {
    let sample_cols: Vec<i64> = b.sample_cols.iter().map(|&x| x as i64).collect();
    let d = PyDict::new(py);
    d.set_item("starts", PyArray1::from_slice(py, &b.starts))?;
    d.set_item("stops", PyArray1::from_slice(py, &b.stops))?;
    d.set_item("sample_cols", PyArray1::from_slice(py, &sample_cols))?;
    d.set_item("n_ranges", b.n_ranges)?;
    d.set_item("n_samples", b.n_samples)?;
    d.set_item("ploidy", b.ploidy)?;
    Ok(d)
}

#[pymethods]
impl PySvar1Reader {
    // `pub` so integration tests (an external crate) can call it directly as a plain
    // Rust constructor; pyo3 keeps `#[new]` methods callable from Rust.
    #[new]
    pub fn new(svar1_dir: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        let inner = Svar1Reader::open(svar1_dir, n_samples, ploidy)?;
        Ok(Self { inner })
    }

    pub fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    pub fn ploidy(&self) -> usize {
        self.inner.ploidy()
    }

    /// Stage A: POS ranges -> GLOBAL variant-id ranges, `(n_regions, 2)` int64.
    ///
    /// `max_v_len` uses Python `var_ranges`'s convention (`max(v_ends - v_starts)`).
    /// No-overlap yields an in-bounds zero-length row, NOT Python's `INT32_MAX`
    /// sentinel.
    pub fn var_ranges<'py>(
        &self,
        py: Python<'py>,
        v_starts: Vec<u32>,
        v_ends: Vec<u32>,
        max_v_len: u32,
        contig_start: u32,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let rs = var_ranges(&v_starts, &v_ends, max_v_len, contig_start, &regions);
        let mut flat: Vec<i64> = Vec::with_capacity(rs.len() * 2);
        for r in &rs {
            flat.push(r.start as i64);
            flat.push(r.end as i64);
        }
        let arr = ndarray::Array2::from_shape_vec((rs.len(), 2), flat).expect("var_ranges shape");
        Ok(arr.to_pyarray(py))
    }

    /// Stage B: `(n_ranges, 2)` GLOBAL variant-id ranges -> the index-pair dict.
    pub fn find_ranges<'py>(
        &self,
        py: Python<'py>,
        ranges: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rs: Vec<std::ops::Range<u32>> = ranges.iter().map(|&(s, e)| s..e).collect();
        let b = find_ranges(&self.inner, &rs, samples.as_deref());
        bundle_to_dict(py, &b)
    }
}

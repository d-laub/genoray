//! Python-facing handle over a finished SVAR2 contig (M6a). Wraps the pure-Rust
//! `query::ContigReader` so Python can open a contig; query methods (raw
//! two-channel M6b, decoded M6c) are added to this class in their milestones.

use pyo3::prelude::*;

use crate::query::ContigReader;

/// A finished SVAR2 contig opened for querying. Constructed from Python as
/// `PyContigReader(base_out_dir, chrom, n_samples, ploidy)`.
#[pyclass]
pub struct PyContigReader {
    pub(crate) inner: ContigReader,
}

#[pymethods]
impl PyContigReader {
    // `pub` so the integration test (an external crate) can call it directly as a
    // plain Rust constructor; pyo3 keeps `#[new]` methods callable from Rust.
    #[new]
    pub fn new(base_out_dir: &str, chrom: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        let inner = ContigReader::open(base_out_dir, chrom, n_samples, ploidy)?;
        Ok(Self { inner })
    }
}

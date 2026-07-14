//! Python-facing mutcat: post-hoc annotation + count-matrix on PyContigReader.

use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use crate::layout::ContigPaths;
use crate::mutcat::N_CODES;
use crate::mutcat::annotate::annotate_contig;
use crate::mutcat::count::{Sidecars, count_contig};
use crate::py_query::PyContigReader;

#[pymethods]
impl PyContigReader {
    /// Classify this contig's records against `ref_seq` and write the sidecar.
    fn annotate_mutations(
        &self,
        base_out_dir: &str,
        chrom: &str,
        ref_seq: PyReadonlyArray1<u8>,
    ) -> PyResult<()> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        let seq = ref_seq.as_slice()?;
        annotate_contig(&self.inner, &paths, seq, None)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("annotate {chrom}: {e}")))?;
        Ok(())
    }

    /// Build the `(n_samples, N_CODES)` count matrix for this contig.
    fn count_matrix<'py>(
        &self,
        py: Python<'py>,
        base_out_dir: &str,
        chrom: &str,
        per_sample: bool,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        let sidecars = Sidecars::open(&paths).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("open sidecar {chrom}: {e}"))
        })?;
        let mut acc = ndarray::Array2::<i64>::zeros((self.inner.n_samples, N_CODES));
        count_contig(&self.inner, &sidecars, per_sample, &mut acc);
        Ok(acc.to_pyarray(py))
    }
}

//! Python-facing mutcat: post-hoc annotation + count-matrix on PyContigReader.

use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use crate::layout::ContigPaths;
use crate::mutcat::N_CODES;
use crate::mutcat::annotate::{StrandIntervals, annotate_contig};
use crate::mutcat::count::{Sidecars, count_contig};
use crate::py_query::PyContigReader;

#[pymethods]
impl PyContigReader {
    /// Classify this contig's records against `ref_seq` and write the sidecar.
    /// If all three `strand_*` arrays are given (a sorted, disjoint gene-strand
    /// interval partition), also write the 2-bit `strand.bin` enabling SBS192/384.
    #[pyo3(signature = (base_out_dir, chrom, ref_seq, strand_starts=None, strand_stops=None, strand_values=None))]
    fn annotate_mutations(
        &self,
        base_out_dir: &str,
        chrom: &str,
        ref_seq: PyReadonlyArray1<u8>,
        strand_starts: Option<PyReadonlyArray1<i32>>,
        strand_stops: Option<PyReadonlyArray1<i32>>,
        strand_values: Option<PyReadonlyArray1<u8>>,
    ) -> PyResult<()> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        let seq = ref_seq.as_slice()?;
        // Keep the numpy arrays alive for the duration of the borrow.
        let strand_arrays = match (strand_starts, strand_stops, strand_values) {
            (Some(a), Some(b), Some(c)) => Some((a, b, c)),
            _ => None,
        };
        let strand = match &strand_arrays {
            Some((a, b, c)) => Some(StrandIntervals {
                starts: a.as_slice()?,
                stops: b.as_slice()?,
                values: c.as_slice()?,
            }),
            None => None,
        };
        annotate_contig(&self.inner, &paths, seq, strand)
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

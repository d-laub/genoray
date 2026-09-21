//! Python-facing cluster labels: annotate + fill on `PyContigReader`.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::cluster::write::{ClusterError, annotate_contig, fill_contig};
use crate::field::StorageDtype;
use crate::layout::ContigPaths;
use crate::py_query::PyContigReader;

fn cluster_err(context: &str, e: ClusterError) -> PyErr {
    match e {
        ClusterError::Invalid(msg) => {
            pyo3::exceptions::PyValueError::new_err(format!("{context}: {msg}"))
        }
        ClusterError::Io(err) => pyo3::exceptions::PyIOError::new_err(format!("{context}: {err}")),
    }
}

#[pymethods]
impl PyContigReader {
    /// Classify this contig's samples against `cutoffs` (one per cohort
    /// sample) and write all four `cluster_class` value streams.
    #[pyo3(signature = (base_out_dir, chrom, cutoffs, vaf=None, vaf_cut=0.1))]
    fn annotate_clusters(
        &self,
        base_out_dir: &str,
        chrom: &str,
        cutoffs: PyReadonlyArray1<f64>,
        vaf: Option<(String, String)>,
        vaf_cut: f64,
    ) -> PyResult<()> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        let cutoffs = cutoffs.as_slice()?;
        let vaf = match &vaf {
            Some((name, dtype)) => {
                let dtype = StorageDtype::from_meta_str(dtype).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown VAF storage dtype {dtype:?}"
                    ))
                })?;
                Some((name.as_str(), dtype))
            }
            None => None,
        };
        annotate_contig(&self.inner, &paths, cutoffs, vaf, vaf_cut)
            .map_err(|e| cluster_err(&format!("annotate clusters {chrom}"), e))
    }

    /// Write 255-filled `cluster_class` streams for this contig (out of scope).
    fn fill_cluster_labels(&self, base_out_dir: &str, chrom: &str) -> PyResult<()> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        fill_contig(&self.inner, &paths)
            .map_err(|e| cluster_err(&format!("fill cluster labels {chrom}"), e))
    }
}

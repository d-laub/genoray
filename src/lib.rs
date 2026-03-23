use pyo3::prelude::*;
use crossbeam_channel::bounded;
use std::thread;
use std::sync::Arc;

// Declare the modules so the Rust compiler knows to look for these files
mod types;
mod vcf_reader;
mod rvk;
mod writer;
mod nrvk;
mod merge;

// Import the specific structs and functions from our modules
use types::{DenseChunk, SparseChunk};
use nrvk::SharedArena;
use vcf_reader::read_vcf_chunk;
use rvk::convert_dense_to_sparse;
use writer::write_sparse_chunk;
use merge::merge_mini_cscs;

// The Python entry point. Defines #[pymodule] and sets up the crossbeam threads.
#[pymodule]
#[pyo3(name = "_core")]
fn core(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

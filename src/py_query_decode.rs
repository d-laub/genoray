//! M6c: decoded `seqpro.rag.Ragged` materialization + decode-free counts on
//! `PyContigReader`. Owned by the `svar-2-m6c` worktree. Separate `#[pymethods]`
//! block so M6b and M6c never touch the same file.

use crate::py_query::PyContigReader;
use pyo3::prelude::*;

#[pymethods]
impl PyContigReader {}

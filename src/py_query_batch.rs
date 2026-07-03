//! M6b: raw two-channel `BatchResult` → numpy exposure on `PyContigReader`.
//! Owned by the `svar-2-m6b` worktree; the query method lands there. Kept in a
//! separate `#[pymethods]` block (multiple-pymethods) so M6b and M6c never touch
//! the same file.

use crate::py_query::PyContigReader;
use pyo3::prelude::*;

#[pymethods]
impl PyContigReader {}

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_core")]
fn core(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

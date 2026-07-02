//! Typed errors for the conversion pipeline. Boundary-level: the orchestrator and
//! pyo3 entry point return these; worker-thread hot loops still panic (converting
//! them is a follow-up).

#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("I/O error at {context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("worker thread '{thread}' panicked")]
    WorkerPanicked { thread: String },
    #[error("failed to write npy at {path}: {source}")]
    Npy {
        path: String,
        #[source]
        source: ndarray_npy::WriteNpyError,
    },
}

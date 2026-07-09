//! Boundary-level typed errors for the conversion pipeline. Categories map to
//! Python builtins via `impl From<ConversionError> for PyErr`.

#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// User-recoverable *content* error: bad contig/sample name, symbolic ALT,
    /// REF/FASTA mismatch, missing index. The message is the whole error.
    #[error("{0}")]
    Input(String),
    /// A required input file (index, FASTA) is absent.
    #[error("required file not found: {path}")]
    MissingFile { path: String },
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
    #[error("failed to read npy at {path}: {source}")]
    ReadNpy {
        path: String,
        #[source]
        source: ndarray_npy::ReadNpyError,
    },
}

#[cfg(feature = "conversion")]
impl From<crate::normalize::NormalizeError> for ConversionError {
    fn from(e: crate::normalize::NormalizeError) -> Self {
        ConversionError::Input(e.to_string())
    }
}

impl From<ConversionError> for pyo3::PyErr {
    fn from(e: ConversionError) -> Self {
        use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyRuntimeError, PyValueError};
        let msg = e.to_string();
        match e {
            ConversionError::Input(_) => PyValueError::new_err(msg),
            ConversionError::MissingFile { .. } => PyFileNotFoundError::new_err(msg),
            ConversionError::Io { .. }
            | ConversionError::Npy { .. }
            | ConversionError::ReadNpy { .. } => PyOSError::new_err(msg),
            ConversionError::WorkerPanicked { .. } => PyRuntimeError::new_err(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "conversion")]
    use crate::normalize::NormalizeError;

    #[cfg(feature = "conversion")]
    #[test]
    fn normalize_error_maps_to_input_with_message() {
        let ne = NormalizeError::RefMismatch {
            pos: 7,
            expected: "A".into(),
            found: "C".into(),
        };
        let expected_msg = ne.to_string();
        let ce: ConversionError = ne.into();
        match ce {
            ConversionError::Input(msg) => {
                assert_eq!(msg, expected_msg);
                assert!(msg.contains("disagrees"));
            }
            other => panic!("expected Input, got {other:?}"),
        }
    }

    #[test]
    fn missing_file_message_includes_path() {
        let ce = ConversionError::MissingFile {
            path: "/x/in.vcf.gz.tbi".into(),
        };
        assert!(ce.to_string().contains("/x/in.vcf.gz.tbi"));
    }
}

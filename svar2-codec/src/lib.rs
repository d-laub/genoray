//! Encode/decode for the SVAR 2.0 variant-key bit layout.
//!
//! This crate is the **single source of truth** for the on-disk key layouts:
//! the 2-bit SNP code stream and the 32-bit indel key (inline INS/SNP, pure DEL,
//! and long-allele-bank lookup lanes). Both the pure encode primitives and the
//! decode primitives live here, so the two halves of the layout can never drift.
//!
//! Pure and std-only: no I/O, no pyo3, no long-allele bank. Callers that need
//! those (file packing, bank spill) live in the `genoray` crate and call in here.

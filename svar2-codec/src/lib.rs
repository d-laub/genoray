//! Encode/decode for the SVAR 2.0 variant-key bit layout.
//!
//! This crate is the **single source of truth** for the on-disk key layouts:
//! the 2-bit SNP code stream and the 32-bit indel key (inline INS/SNP, pure DEL,
//! and long-allele-bank lookup lanes). Both the pure encode primitives and the
//! decode primitives live here, so the two halves of the layout can never drift.
//!
//! Pure and std-only: no I/O, no pyo3, no long-allele bank. Callers that need
//! those (file packing, bank spill) live in the `genoray` crate and call in here.

/// Minimum signed `ilen` representable inline as a pure DEL (i31 two's complement).
/// Real data won't approach this — atomized DELs span at most chromosome length
/// (~250 Mbp).
pub const MIN_I31: i32 = -(1 << 30);

/// Maximum ALT byte length that fits the inline encoding (26 bits ÷ 2 bits/base =
/// 13). Beyond this, a pure-INS variant spills to the long-allele bank.
pub const MAX_INLINE_ALT_LEN: usize = 13;

/// 2-bit code → ALT base. `A=00 C=01 T=10 G=11`. `T`/`G` are swapped vs. the
/// obvious alphabetical order — the values are an implementation detail of this
/// crate and carry no meaning outside it.
pub const BASES: [u8; 4] = [b'A', b'C', b'T', b'G'];

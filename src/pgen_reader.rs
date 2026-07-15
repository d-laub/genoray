//! PGEN record source.
//!
//! Genotypes come from the `pgenlib` PyPI wheel (LGPL-3.0) via its **public Python
//! API** -- the same way `genoray/_pgen.py` already uses it. genoray links no
//! plink-ng code and vendors none; `_core` stays MIT. Do not change this without
//! reading `docs/superpowers/specs/2026-07-12-pgen-to-svar2-design.md`.
//!
//! `PgenReader.read_alleles_range` does NOT release the GIL for its decode loop
//! (its compiled module contains no `PyEval_SaveThread`), so it holds the GIL
//! for the whole decode. To keep GIL contention across concurrent shards to the
//! bare minimum, each `refill` acquires the GIL exactly once -- for the
//! `read_alleles_range` call AND a bulk copy of the whole batch into a
//! Rust-owned buffer -- so that `next_record` (called per variant) touches no
//! GIL at all. A previous design acquired the GIL once per row inside
//! `next_record`; at 11-way intra-contig concurrency that per-row GIL churn
//! collapsed throughput to ~zero (a GIL convoy that looked like a deadlock).

use crate::error::ConversionError;
use crate::pvar::PvarReader;
use crate::record_source::{RawRecord, RecordSource};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Byte budget for the `(batch, 2 * n_samples)` int32 allele-code buffer. Sized so
/// small cohorts get thousands of variants per Python call (amortizing dispatch)
/// while biobank cohorts stay memory-bounded.
pub const PGEN_BATCH_BYTES: usize = 32 * 1024 * 1024;

pub struct PgenRecordSource {
    /// A `pgenlib.PgenReader`, constructed in Python and owned here.
    reader: Py<PyAny>,
    /// `(batch, 2 * num_samples)` int32 allele-code scratch buffer, reused per refill.
    buf: Py<PyArray2<i32>>,
    /// Rust-owned mirror of `buf`, refilled in bulk under a single GIL acquisition
    /// per `refill` so `next_record` can serve rows without touching the GIL.
    /// pgenlib's `-9` missing sentinel is already normalized to `-1` here.
    host_buf: Vec<i32>,
    batch: usize,
    num_samples: usize,
    /// Next global variant index to fetch from the .pgen.
    var_next: usize,
    /// Exclusive end of this contig's variant index range.
    var_end: usize,
    /// Rows currently valid in `buf`, and the next one to serve.
    filled: usize,
    row: usize,
    pvar: PvarReader,
}

impl PgenRecordSource {
    pub fn new(
        reader: Py<PyAny>,
        pvar_path: &str,
        var_start: usize,
        var_end: usize,
        num_samples: usize,
        chunk_size: usize,
    ) -> Result<Self, ConversionError> {
        let batch = (PGEN_BATCH_BYTES / (2 * num_samples * 4)).clamp(1, chunk_size.max(1));
        let pvar = PvarReader::open(pvar_path, var_start)?;
        let buf = Python::attach(|py| {
            PyArray2::<i32>::zeros(py, [batch, 2 * num_samples], false).unbind()
        });
        Ok(Self {
            reader,
            buf,
            host_buf: Vec::with_capacity(batch * 2 * num_samples),
            batch,
            num_samples,
            var_next: var_start,
            var_end,
            filled: 0,
            row: 0,
            pvar,
        })
    }

    /// Refill `buf` with the next `min(batch, var_end - var_next)` variants.
    /// Returns the number of rows filled (0 => this contig is exhausted).
    fn refill(&mut self) -> Result<usize, ConversionError> {
        let lo = self.var_next;
        if lo >= self.var_end {
            return Ok(0);
        }
        let hi = (lo + self.batch).min(self.var_end);
        let columns = 2 * self.num_samples;
        // One GIL acquisition for BOTH the pgenlib decode (which holds the GIL)
        // AND the bulk copy into `host_buf` -- see the module doc for why
        // per-row GIL churn here collapses under concurrent shards.
        Python::attach(|py| -> Result<(), ConversionError> {
            self.reader
                .bind(py)
                .call_method1(
                    "read_alleles_range",
                    (lo as u32, hi as u32, self.buf.bind(py)),
                )
                .map_err(|e| {
                    ConversionError::Input(format!(
                        "pgenlib read_alleles_range({lo}, {hi}) failed: {e}"
                    ))
                })?;
            let arr = self.buf.bind(py).readonly();
            let flat = arr.as_slice().expect("pgen buffer is C-contiguous");
            let n = (hi - lo) * columns;
            self.host_buf.clear();
            // pgenlib encodes missing as -9; the conversion spine uses -1.
            self.host_buf
                .extend(flat[..n].iter().map(|&c| if c < 0 { -1 } else { c }));
            Ok(())
        })?;
        self.var_next = hi;
        self.filled = hi - lo;
        self.row = 0;
        Ok(self.filled)
    }
}

impl RecordSource for PgenRecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        if self.row == self.filled && self.refill()? == 0 {
            return Ok(None);
        }

        // .pvar and .pgen advance in lockstep -- one metadata row per genotype row.
        let Some(meta) = self.pvar.next_variant()? else {
            return Err(ConversionError::Input(
                "pvar ran out of variants before the .pgen did; \
                 the .pvar and .pgen disagree on variant count"
                    .to_string(),
            ));
        };

        // Served straight from `host_buf` -- already normalized and GIL-free.
        let columns = 2 * self.num_samples;
        let base = self.row * columns;
        let gt = self.host_buf[base..base + columns].to_vec();
        self.row += 1;

        Ok(Some(RawRecord {
            pos: meta.pos,
            reference: meta.reference,
            alts: meta.alts,
            gt,
            // PGEN has no FORMAT, and .pvar INFO extraction is out of scope for v1.
            info_raw: Vec::new(),
            format_raw: Vec::new(),
        }))
    }
}

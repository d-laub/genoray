//! The seam between "where variant records come from" (VCF via htslib, PGEN via
//! pgenlib) and the source-agnostic conversion spine (`chunk_assembler`).
//!
//! `RawRecord` is deliberately OWNED rather than borrowed: the assembler mutates
//! its heap while consuming a record, and the VCF path already allocates every one
//! of these fields per record anyway, so owning costs nothing.

use crate::error::ConversionError;

/// One variant record, decoded to the minimum the conversion spine needs.
pub struct RawRecord {
    /// 0-based start position (VCF/pvar POS minus 1).
    pub pos: u32,
    /// REF allele bases, uppercase ASCII.
    pub reference: Vec<u8>,
    /// ALT alleles in file order. ALT1 is `alts[0]`; note `PendingAtom::
    /// source_alt_index` is 1-based (ALT1 => 1), matching BCF GT allele codes.
    pub alts: Vec<Vec<u8>>,
    /// Allele index per haplotype column, length `num_samples * ploidy`,
    /// sample-major then ploidy-minor. `0` = REF, `k` = ALT k, `-1` = missing.
    pub gt: Vec<i32>,
    /// Raw INFO buffers, widened to f64, one entry per requested INFO `FieldSpec`
    /// in spec order. `None` = the field is absent from this record. Empty when no
    /// INFO fields were requested.
    pub info_raw: Vec<Option<Vec<f64>>>,
    /// Raw FORMAT buffers, widened to f64: outer index = requested FORMAT
    /// `FieldSpec` in spec order, inner index = **selected** sample
    /// (`0..num_samples`, already remapped from the source's own sample order).
    /// Outer `None` = the field is absent from this record for all samples. Empty
    /// when no FORMAT fields were requested.
    pub format_raw: Vec<Option<Vec<Vec<f64>>>>,
}

/// A cursor over one contig's variant records, in file order.
pub trait RecordSource {
    /// Next record, or `None` at end of contig.
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError>;
}

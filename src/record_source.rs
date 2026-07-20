//! The seam between "where variant records come from" (VCF via htslib, PGEN via
//! pgenlib) and the source-agnostic conversion spine (`chunk_assembler`).
//!
//! `RawRecord` is deliberately OWNED rather than borrowed: the assembler mutates
//! its heap while consuming a record, and the VCF path already allocates every one
//! of these fields per record anyway, so owning costs nothing.

use crate::error::ConversionError;
use crate::field::FieldSpec;

/// One variant record, decoded to the minimum the conversion spine needs.
#[derive(Debug, Clone, PartialEq)]
pub struct RawRecord {
    /// 0-based start position (VCF/pvar POS minus 1).
    pub pos: u32,
    /// REF allele bases, uppercase ASCII.
    pub reference: Vec<u8>,
    /// ALT alleles in file order. ALT1 is `alts[0]`; note `PendingAtom::
    /// source_alt_index` is 1-based (ALT1 => 1), matching BCF GT allele codes.
    pub alts: Vec<Vec<u8>>,
    /// Per-haplotype allele calls. `Calls::Dense` carries one entry per column;
    /// `Calls::Sparse` carries only the non-REF columns, which is what a k-way merge
    /// of single-sample files naturally produces.
    pub calls: Calls,
    /// Raw INFO buffers, widened to f64, one entry per requested INFO `FieldSpec`
    /// in spec order. `None` = the field is absent from this record. Empty when no
    /// INFO fields were requested.
    pub info_raw: Vec<Option<Vec<f64>>>,
    /// Raw FORMAT buffers. `Dense` for sources whose records physically carry
    /// every sample (a multi-sample VCF, PGEN); `ByCarrier` for a k-way merge of
    /// single-sample files, where only the carrying samples have anything to say.
    pub format_vals: FormatVals,
    /// Dataset-global variant id, threaded verbatim onto every atom this record
    /// decomposes into. `-1` for sources that don't have one yet (everything but
    /// PGEN, for now -- see `pgen_reader.rs`).
    pub global_idx: i32,
}

/// A cursor over one contig's variant records, in file order.
pub trait RecordSource {
    /// Next record, or `None` at end of contig.
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError>;
}

/// The non-REF calls of one record: which haplotype columns carry which allele.
///
/// Length is the carrier count, never `num_samples * ploidy`. For a k-way merge of
/// single-sample VCFs a record typically has exactly one carrier out of N, so this is
/// the difference between O(1) and O(N) work per record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Carriers {
    /// Haplotype column index per carrier, strictly ascending.
    cols: Vec<u32>,
    /// Allele index per carrier, parallel to `cols`: `k` = ALT k (1-based),
    /// `-1` = missing. Never `0` — a REF call is represented by absence.
    alleles: Vec<i32>,
}

impl Carriers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a non-REF call. `col` must be strictly greater than the previous push
    /// (callers walk columns in ascending order), and `allele` must not be 0.
    pub fn push(&mut self, col: u32, allele: i32) {
        debug_assert!(allele != 0, "REF is represented by absence, not a 0 entry");
        debug_assert!(
            self.cols.last().is_none_or(|&p| p < col),
            "carrier columns must be pushed strictly ascending"
        );
        self.cols.push(col);
        self.alleles.push(allele);
    }

    pub fn len(&self) -> usize {
        self.cols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }

    /// `(column, allele)` per carrier, ascending.
    pub fn iter(&self) -> impl Iterator<Item = (u32, i32)> + '_ {
        self.cols.iter().copied().zip(self.alleles.iter().copied())
    }
}

/// A record's per-haplotype allele calls.
///
/// Sources whose records physically carry every sample (one multi-sample VCF, PGEN)
/// produce `Dense` — for them the widening is the real data, not waste. A k-way merge
/// of single-sample files produces `Sparse`. Both must be indistinguishable through
/// the accessors below, since they feed the same store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Calls {
    /// Allele index per haplotype column, length `num_samples * ploidy`, sample-major
    /// then ploidy-minor. `0` = REF, `k` = ALT k, `-1` = missing.
    Dense(Vec<i32>),
    /// Only the non-REF columns.
    Sparse(Carriers),
}

impl Calls {
    /// Allele at `col`: `0` REF, `k` ALT k, `-1` missing.
    ///
    /// O(1) for `Dense`, O(log n_carriers) for `Sparse`. Prefer `iter_non_ref` in hot
    /// loops — a scan of every column is exactly the O(V x N) this design removes.
    ///
    /// `col` must be less than the cohort's column count (`num_samples * ploidy`); the
    /// caller owns that bound, not this type. `Sparse` has no width to check it
    /// against — carrying one per record would defeat the point of the representation
    /// — so an out-of-range `col` silently reads back as REF (`0`), the same as any
    /// other absent column. `Dense` panics instead, since it indexes a `Vec` directly.
    /// Do not rely on that panic to catch indexing bugs: the same bug against a
    /// `Sparse` record fails silently rather than loudly.
    pub fn allele_at(&self, col: usize) -> i32 {
        match self {
            Calls::Dense(gt) => gt[col],
            Calls::Sparse(c) => match c.cols.binary_search(&(col as u32)) {
                Ok(i) => c.alleles[i],
                Err(_) => 0,
            },
        }
    }

    /// `(column, allele)` for every non-REF column, ascending.
    pub fn iter_non_ref(&self) -> impl Iterator<Item = (u32, i32)> + '_ {
        // Boxed so both arms share one return type; the per-record cost is one
        // allocation against the O(N) buffer this replaces.
        let it: Box<dyn Iterator<Item = (u32, i32)> + '_> = match self {
            Calls::Dense(gt) => Box::new(
                gt.iter()
                    .enumerate()
                    .filter(|&(_, &a)| a != 0)
                    .map(|(i, &a)| (i as u32, a)),
            ),
            Calls::Sparse(c) => Box::new(c.iter()),
        };
        it
    }

    pub fn n_non_ref(&self) -> usize {
        match self {
            Calls::Dense(gt) => gt.iter().filter(|&&a| a != 0).count(),
            Calls::Sparse(c) => c.len(),
        }
    }

    /// Column count for `Dense`; `None` for `Sparse`, whose width is implied by the
    /// cohort rather than carried per record.
    pub fn len_dense(&self) -> Option<usize> {
        match self {
            Calls::Dense(gt) => Some(gt.len()),
            Calls::Sparse(_) => None,
        }
    }
}

/// FORMAT values for the samples that actually called a record.
///
/// `vals[i * n_fields + j]` is field `j` for the sample at `samples[i]`. A sample
/// absent from `samples` resolves to the field's default downstream -- the same
/// contract an empty per-sample buffer has today.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CarrierFormat {
    /// Selected-sample index per carrier, strictly ascending.
    samples: Vec<u32>,
    /// Row-major `[carrier][field]`.
    vals: Vec<f64>,
    n_fields: usize,
}

impl CarrierFormat {
    pub fn new(n_fields: usize) -> Self {
        Self {
            samples: Vec::new(),
            vals: Vec::new(),
            n_fields,
        }
    }

    /// `vals` must have exactly `n_fields` entries, in FORMAT `FieldSpec` order.
    /// `sample` must be strictly greater than the previous push (callers walk
    /// samples in ascending order).
    pub fn push_sample(&mut self, sample: u32, vals: &[f64]) {
        debug_assert_eq!(vals.len(), self.n_fields);
        debug_assert!(
            self.samples.last().is_none_or(|&p| p < sample),
            "carrier samples must be pushed strictly ascending"
        );
        self.samples.push(sample);
        self.vals.extend_from_slice(vals);
    }

    /// Field `field` for `sample`, or `None` if that sample called nothing here.
    pub fn value(&self, sample: usize, field: usize) -> Option<f64> {
        if self.n_fields == 0 || field >= self.n_fields {
            return None;
        }
        let i = self.samples.binary_search(&(sample as u32)).ok()?;
        Some(self.vals[i * self.n_fields + field])
    }
}

/// Raw FORMAT buffers for one record.
#[derive(Debug, Clone, PartialEq)]
pub enum FormatVals {
    /// Outer index = requested FORMAT `FieldSpec` in spec order, inner index =
    /// selected sample. Outer `None` = field absent from this record for all
    /// samples. Natural for sources whose records carry every sample.
    Dense(Vec<Option<Vec<Vec<f64>>>>),
    /// Only the calling samples. Natural for a k-way merge of single-sample files.
    ByCarrier(CarrierFormat),
}

// Delegates to `FieldSpec::missing_sentinel` — single source of truth shared
// with `dense2sparse_vk`'s genotype-aligned non-carrier fill (src/rvk.rs).
fn sentinel_default(spec: &FieldSpec) -> f64 {
    spec.missing_sentinel()
}

// htslib missing-value detection on an already-widened f64: floats use NaN
// (covers both htslib's MISSING and VECTOR_END float encodings, which are
// distinct NaN bit patterns); ints use MISSING (i32::MIN) or VECTOR_END
// (i32::MIN + 1) — the round-trip through f64 is exact for both since they're
// in i32 range.
fn is_htslib_missing(raw_val: f64, is_float: bool) -> bool {
    if is_float {
        raw_val.is_nan()
    } else {
        let iv = raw_val as i32;
        iv == i32::MIN || iv == i32::MIN + 1
    }
}

// Resolve one atom's value from a record-level raw buffer (already widened to
// f64): `None` (field absent from the record/sample) or an out-of-range
// Number=A index falls back to the spec's sentinel/default; a length-1 buffer
// is a Number=1 scalar, otherwise indexed by `source_alt_index - 1` (Number=A):
// `source_alt_index` is 1-based (ALT1 → 1, matching BCF GT allele codes; see
// `normalize::atomize_record`), but htslib's Number=A buffer is 0-based
// per-ALT (ALT1's value lives at `vals[0]`).
pub(crate) fn resolve_scalar(vals: Option<&[f64]>, source_alt_index: u16, spec: &FieldSpec) -> f64 {
    let default_val = sentinel_default(spec);
    let Some(vals) = vals else {
        return default_val;
    };
    if vals.is_empty() {
        return default_val;
    }
    let raw_val = if vals.len() == 1 {
        vals[0]
    } else {
        // source_alt_index is 1-based; htslib Number=A buffers are 0-based per-ALT.
        match vals.get(source_alt_index.saturating_sub(1) as usize) {
            Some(&v) => v,
            None => return default_val,
        }
    };
    if is_htslib_missing(raw_val, spec.stage_is_float()) {
        default_val
    } else {
        raw_val
    }
}

// Field `j` (`spec`) for sample `s`, from a record-level `FormatVals` -- the
// lazy counterpart to the eager per-atom resolution `decompose_raw_record` used
// to do. `source_alt_index` is THIS ATOM's own index (atoms sharing one source
// record can each carry a different ALT after `atomize_record` decomposes a
// multiallelic record), not the record's.
//
// **The ALT-index asymmetry between the two arms is deliberate, not a bug:**
// - `Dense` buffers are still record-raw (Number=A, one entry per source ALT,
//   0-based) exactly as `decode_format_raw`/`RawRecord::format_vals` produced
//   them -- so this arm must re-apply `source_alt_index` via `resolve_scalar`,
//   the same call `decompose_raw_record` made before this change.
// - `ByCarrier` values are already fully resolved, per carrier, against THAT
//   FILE's OWN `source_alt_index` at merge time (`vcf_list_reader.rs`'s
//   `FileCursor::advance`, before the merge heap ever sees them) -- a k-way
//   merge of single-sample files only ever emits single-ALT `RawRecord`s
//   (`alts: vec![alt]`), so re-applying an index here would double-resolve
//   against the WRONG (always-1) index and silently corrupt a non-carrier's
//   fallback path. `cf.value` already returns `None` for a non-carrier, and
//   `resolve_scalar(None, 0, spec)` resolves that to the spec default
//   regardless of the index passed, so `0` there is inert, not a real choice.
pub(crate) fn resolve_format(
    fv: &FormatVals,
    spec: &FieldSpec,
    source_alt_index: u16,
    s: usize,
    j: usize,
) -> f64 {
    match fv {
        FormatVals::ByCarrier(cf) => cf
            .value(s, j)
            .unwrap_or_else(|| resolve_scalar(None, 0, spec)),
        FormatVals::Dense(raw) => {
            let sample_vals = raw[j].as_ref().map(|v| v[s].as_slice());
            resolve_scalar(sample_vals, source_alt_index, spec)
        }
    }
}

#[cfg(test)]
mod format_vals_tests {
    use super::*;

    // FORMAT is per *sample* (not per haplotype column). A non-carrier must resolve to
    // the field default -- the same contract `resolve_scalar` gives an empty buffer
    // today -- so Dense and ByCarrier stay interchangeable.
    #[test]
    fn by_carrier_returns_none_for_non_carriers() {
        let mut cf = CarrierFormat::new(2); // 2 FORMAT fields
        cf.push_sample(1, &[0.5, 30.0]);
        cf.push_sample(3, &[0.25, 12.0]);

        assert_eq!(cf.value(1, 0), Some(0.5));
        assert_eq!(cf.value(1, 1), Some(30.0));
        assert_eq!(cf.value(3, 0), Some(0.25));
        assert_eq!(cf.value(0, 0), None, "sample 0 carries nothing");
        assert_eq!(cf.value(2, 1), None, "sample 2 carries nothing");
    }

    #[test]
    fn by_carrier_is_empty_when_no_fields_requested() {
        let cf = CarrierFormat::new(0);
        assert_eq!(cf.value(0, 0), None);
    }
}

#[cfg(test)]
mod calls_tests {
    use super::*;

    // A GT column is 0 = REF, k = ALT k, -1 = missing. `Carriers` stores only the
    // non-REF columns; anything absent reads back as 0. Dense and Sparse must be
    // indistinguishable through the accessors, because the k-way merge and a
    // multi-sample VCF must produce identical stores from identical calls.
    fn dense_fixture() -> Vec<i32> {
        // 4 samples x ploidy 2 = 8 columns; sample 1 hom-ALT1, sample 3 het ALT2/missing.
        vec![0, 0, 1, 1, 0, 0, 2, -1]
    }

    fn sparse_fixture() -> Carriers {
        let mut c = Carriers::new();
        c.push(2, 1);
        c.push(3, 1);
        c.push(6, 2);
        c.push(7, -1);
        c
    }

    #[test]
    fn allele_at_agrees_between_dense_and_sparse() {
        let d = Calls::Dense(dense_fixture());
        let s = Calls::Sparse(sparse_fixture());
        for col in 0..8 {
            assert_eq!(
                d.allele_at(col),
                s.allele_at(col),
                "disagreement at column {col}"
            );
        }
    }

    #[test]
    fn iter_non_ref_yields_only_non_ref_ascending() {
        let s = Calls::Sparse(sparse_fixture());
        let got: Vec<(u32, i32)> = s.iter_non_ref().collect();
        assert_eq!(got, vec![(2, 1), (3, 1), (6, 2), (7, -1)]);

        let d = Calls::Dense(dense_fixture());
        let got_d: Vec<(u32, i32)> = d.iter_non_ref().collect();
        assert_eq!(
            got_d, got,
            "dense and sparse must agree on the carrier list"
        );
    }

    #[test]
    fn absent_column_reads_as_ref() {
        let s = Calls::Sparse(sparse_fixture());
        assert_eq!(s.allele_at(0), 0);
        assert_eq!(s.allele_at(5), 0);
    }

    #[test]
    fn n_non_ref_counts_missing_as_non_ref() {
        // -1 is not REF; it must not be dropped, or a ./. hap would silently become 0/0.
        assert_eq!(Calls::Sparse(sparse_fixture()).n_non_ref(), 4);
        assert_eq!(Calls::Dense(dense_fixture()).n_non_ref(), 4);
    }

    #[test]
    fn empty_carriers_is_all_ref() {
        let s = Calls::Sparse(Carriers::new());
        assert_eq!(s.n_non_ref(), 0);
        assert_eq!(s.allele_at(3), 0);
        assert_eq!(s.iter_non_ref().count(), 0);
    }

    /// Pins the documented asymmetry from `Calls::allele_at`'s doc comment: `Sparse`
    /// has no width to bounds-check against, so a column past the cohort's actual
    /// width (8, per the fixtures above) silently reads as REF instead of panicking
    /// like `Dense` would. This is deliberate — see the doc comment — but must stay a
    /// pinned property, not an accident a later refactor could change unknowingly.
    #[test]
    fn out_of_range_column_reads_as_ref_on_sparse() {
        let s = Calls::Sparse(sparse_fixture());
        assert_eq!(s.allele_at(8), 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "REF is represented by absence, not a 0 entry")]
    fn push_panics_on_ref_allele() {
        let mut c = Carriers::new();
        c.push(0, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "carrier columns must be pushed strictly ascending")]
    fn push_panics_on_non_ascending_columns() {
        let mut c = Carriers::new();
        c.push(2, 1);
        c.push(1, 1);
    }
}

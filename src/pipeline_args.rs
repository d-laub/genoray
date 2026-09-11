//! Grouped arguments for the conversion pipeline entry points.
//!
//! `run_conversion_pipeline` reached 20 user-facing parameters one addition at a
//! time, and the cost was that most of them were positionally interchangeable by
//! type: six strings, three `usize`, four bool-ish flags, and two identically
//! typed field manifests. Transposing `info_fields` and `format_fields` at a call
//! site compiled cleanly and routed every field to the wrong section (#153).
//!
//! These structs follow the pattern [`crate::tuning::TuningIn`] already
//! established: a `FromPyObject` struct on this side, a frozen `kw_only`
//! dataclass on the Python side. Two properties fall out of that pairing, and
//! both are the point:
//!
//! - fields are extracted **by name**, so no argument order exists to get wrong;
//! - `kw_only=True` means the dataclasses cannot be built positionally either,
//!   so the transposition is unrepresentable at the construction site too, not
//!   merely moved one level down.
//!
//! They are deliberately plain data. Parsing and validation stay where they were
//! -- in the pipeline function, before any output byte is written -- so that an
//! invalid `check_ref` or `regions_overlap` still raises at the same moment.

use pyo3::FromPyObject;

/// One entry of a field manifest: `(name, category, dtype, default, fill)` as
/// `_svar2_fields.py` serializes it for [`crate::field::parse_manifest`].
pub type FieldTuple = (String, String, String, Option<String>, Option<f64>);

/// Which INFO and FORMAT fields to extract during the write.
///
/// The two manifests have the same type, which is exactly why they belong in a
/// struct: named fields make swapping them a compile error rather than a silent
/// mis-routing of every field.
#[derive(FromPyObject, Debug, Clone, Default)]
pub struct FieldSpec {
    pub info: Vec<FieldTuple>,
    pub format: Vec<FieldTuple>,
}

/// What to read: which contigs, which samples, and which coordinate ranges.
#[derive(FromPyObject, Debug, Clone, Default)]
pub struct RegionSpec {
    pub chroms: Vec<String>,
    pub samples: Vec<String>,
    /// `(chrom, start, end)` triples; grouped by chrom inside the pipeline.
    pub region_ranges: Vec<(String, u32, u32)>,
    /// Parsed by `svar2_view::parse_overlap_mode`, which rejects bad values up
    /// front. Kept as the raw string here so that error stays one place.
    pub regions_overlap: String,
}

/// The budget the planner works within.
///
/// Distinct from [`crate::tuning::TuningIn`], which overrides the planner's
/// *decisions*: these are the constraints it plans against. `tuning` also
/// applies to every pipeline, while these settings are shaped by this one --
/// `max_mem_bytes` in particular is only honoured by the sharded VCF path.
#[derive(FromPyObject, Debug, Clone)]
pub struct PlanSettings {
    pub chunk_size: usize,
    pub max_threads: Option<usize>,
    pub long_allele_capacity: usize,
    pub max_mem_bytes: Option<u64>,
}

//! Global finalize pass: resolves each field's on-disk dtype from data
//! observed across every contig, then rewrites the staged `values.bin` files
//! in place to that resolved width.
//!
//! Staging (Tasks 4/5/8a/8b) always writes fields as native-width `i32`
//! (Int/Flag) or `f32` (Float), 4 bytes/element, little-endian. A staged
//! missing sentinel (`i32::MIN` for int, `NaN` for float) is present ONLY
//! when the field has no configured `default` — when a `default` is set,
//! missing was already replaced by that default value at staging time, so no
//! sentinel is ever written for that field. This pass:
//!
//!   1. Enumerates every `values.bin` file staged for a field, across all
//!      contigs and the four genotype-aligned sub-streams
//!      (`var_key_snp`/`var_key_indel`/`dense_snp`/`dense_indel`).
//!   2. Scans them to find the field's global `[min, max]` and whether a
//!      missing sentinel was observed.
//!   3. Resolves the field's concrete `StorageDtype` — validating an
//!      explicit user request against the observed range, or (for `Auto`)
//!      choosing the smallest lossless width, reserving a sentinel slot iff
//!      missing was observed.
//!   4. Rewrites every file to the resolved width, remapping the staged
//!      sentinel (if any) to the resolved dtype's own sentinel encoding.

use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec, HtslibType, StorageDtype};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The four genotype-aligned sub-stream directory names a field's values may
/// be staged under (mirrors `orchestrator.rs`'s `sub_label` computation:
/// `spec.subdir.replace('/', "_")` over `"var_key/snp"`, `"var_key/indel"`,
/// `"dense/snp"`, `"dense/indel"`).
const SUB_LABELS: [&str; 4] = ["var_key_snp", "var_key_indel", "dense_snp", "dense_indel"];

/// A field's globally-resolved on-disk representation, produced once all
/// contigs have been staged and rewritten. `dtype` is always concrete
/// (never `StorageDtype::Auto`).
#[derive(Debug, Clone)]
pub struct ResolvedField {
    pub name: String,
    pub category: FieldCategory,
    pub dtype: StorageDtype,
    pub default: Option<f64>,
}

/// Global `[min, max]` (kept as `f64`; exact for both the `i32` and `f32`
/// staged domains) plus whether a missing sentinel was observed, across
/// every file staged for one field.
#[derive(Debug, Clone, Copy)]
struct ScanStats {
    min: f64,
    max: f64,
    has_missing: bool,
    any_values: bool,
}

impl ScanStats {
    fn empty() -> Self {
        ScanStats {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            has_missing: false,
            any_values: false,
        }
    }

    fn observe(&mut self, v: f64) {
        self.any_values = true;
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
    }
}

fn bad(msg: impl Into<String>) -> ConversionError {
    ConversionError::Input(msg.into())
}

/// Resolve and rewrite every field's `values.bin` files under `output_dir`,
/// returning one `ResolvedField` per input `FieldSpec` (same order).
pub fn finalize_fields(
    output_dir: &Path,
    contigs: &[String],
    fields: &[FieldSpec],
) -> Result<Vec<ResolvedField>, ConversionError> {
    fields
        .iter()
        .map(|field| finalize_one(output_dir, contigs, field))
        .collect()
}

fn finalize_one(
    output_dir: &Path,
    contigs: &[String],
    field: &FieldSpec,
) -> Result<ResolvedField, ConversionError> {
    let files = field_files(output_dir, contigs, field.category.as_str(), &field.name);
    let stats = scan(&files, field)?;
    let dtype = resolve_dtype(field, &stats)?;
    for path in &files {
        rewrite_file(path, field, dtype)?;
    }
    Ok(ResolvedField {
        name: field.name.clone(),
        category: field.category,
        dtype,
        default: field.default,
    })
}

/// Enumerate the `values.bin` files that exist for `name`, across every
/// contig and all four sub labels. Iterating the known sub labels directly
/// (rather than globbing) is simpler and matches the fixed staging layout.
fn field_files(output_dir: &Path, contigs: &[String], category: &str, name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for contig in contigs {
        for sub in SUB_LABELS {
            let path = output_dir
                .join(contig)
                .join("fields")
                .join(category)
                .join(name)
                .join(sub)
                .join("values.bin");
            if path.is_file() {
                out.push(path);
            }
        }
    }
    out
}

/// Read a staged `values.bin` as `f64` (exact for both `i32` and `f32`
/// staged domains), decoding each 4-byte little-endian element per
/// `is_float`.
fn read_staged(path: &Path, is_float: bool) -> Result<Vec<f64>, ConversionError> {
    let bytes = std::fs::read(path).map_err(|e| ConversionError::Io {
        context: format!("reading {}", path.display()),
        source: e,
    })?;
    if bytes.len() % 4 != 0 {
        return Err(bad(format!(
            "{}: staged values.bin length {} is not a multiple of 4 bytes",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| {
            let arr: [u8; 4] = c.try_into().unwrap();
            if is_float {
                f32::from_le_bytes(arr) as f64
            } else {
                i32::from_le_bytes(arr) as f64
            }
        })
        .collect())
}

/// `true` iff a staged element `v` represents the staged missing sentinel.
/// Only meaningful (and only ever actually observed) when the field has no
/// `default` — a `default` means missing was already substituted at staging
/// time, so no sentinel exists in the data regardless of what this returns.
fn is_staged_missing(v: f64, is_float: bool) -> bool {
    if is_float {
        v.is_nan()
    } else {
        v as i64 == i32::MIN as i64
    }
}

fn scan(files: &[PathBuf], field: &FieldSpec) -> Result<ScanStats, ConversionError> {
    let is_float = field.stage_is_float();
    // A `default` was already substituted at staging time, so nothing in
    // the data can be the staged sentinel; treat nothing as missing (the
    // default value itself still flows into min/max via `observe`).
    let treat_missing = field.default.is_none();
    let mut stats = ScanStats::empty();
    for path in files {
        for v in read_staged(path, is_float)? {
            if treat_missing && is_staged_missing(v, is_float) {
                stats.has_missing = true;
            } else {
                stats.observe(v);
            }
        }
    }
    Ok(stats)
}

fn unsigned_bound(width: u32) -> u64 {
    match width {
        1 => u8::MAX as u64,
        2 => u16::MAX as u64,
        4 => u32::MAX as u64,
        _ => unreachable!("width must be 1, 2, or 4 bytes"),
    }
}

fn signed_bounds(width: u32) -> (i64, i64) {
    match width {
        1 => (i8::MIN as i64, i8::MAX as i64),
        2 => (i16::MIN as i64, i16::MAX as i64),
        4 => (i32::MIN as i64, i32::MAX as i64),
        _ => unreachable!("width must be 1, 2, or 4 bytes"),
    }
}

fn unsigned_fits(width: u32, min: i64, max: i64, has_missing: bool) -> bool {
    let reserved = if has_missing { 1 } else { 0 };
    min >= 0 && max <= (unsigned_bound(width) as i64 - reserved)
}

fn signed_fits(width: u32, min: i64, max: i64, has_missing: bool) -> bool {
    let reserved = if has_missing { 1 } else { 0 };
    let (smin, smax) = signed_bounds(width);
    min >= smin + reserved && max <= smax
}

#[derive(Debug, Clone, Copy)]
enum IntChoice {
    Unsigned(u32),
    Signed(u32),
}

/// Smallest lossless integer width for `[min, max]`, reserving a sentinel
/// slot iff `has_missing`. Prefers unsigned over signed at a given width
/// whenever `min >= 0` (checked first below). Widths are tried smallest
/// first; `i32` staging (with `i32::MIN` carved out as the sentinel, so real
/// values are always `> i32::MIN`) guarantees the 4-byte signed case always
/// fits as a last resort.
fn choose_auto_int(min: i64, max: i64, has_missing: bool) -> IntChoice {
    for width in [1u32, 2, 4] {
        if unsigned_fits(width, min, max, has_missing) {
            return IntChoice::Unsigned(width);
        }
        if signed_fits(width, min, max, has_missing) {
            return IntChoice::Signed(width);
        }
    }
    IntChoice::Signed(4)
}

fn int_choice_to_dtype(choice: IntChoice) -> StorageDtype {
    match choice {
        IntChoice::Unsigned(1) => StorageDtype::U8,
        IntChoice::Unsigned(2) => StorageDtype::U16,
        IntChoice::Unsigned(4) => StorageDtype::U32,
        IntChoice::Signed(1) => StorageDtype::I8,
        IntChoice::Signed(2) => StorageDtype::I16,
        IntChoice::Signed(4) => StorageDtype::I32,
        _ => unreachable!("choose_auto_int only emits width 1/2/4"),
    }
}

/// Resolve a field's concrete `StorageDtype`: `Auto` chooses the narrowest
/// lossless representation; an explicit request is range-checked against the
/// observed data and returned as-is (or rejected with `ConversionError`).
fn resolve_dtype(field: &FieldSpec, stats: &ScanStats) -> Result<StorageDtype, ConversionError> {
    match field.dtype {
        StorageDtype::Auto => {
            if field.htype == HtslibType::Flag {
                return Ok(StorageDtype::Bool);
            }
            if field.stage_is_float() {
                return Ok(StorageDtype::F32);
            }
            if !stats.any_values {
                // No data staged anywhere for this field; nothing to narrow.
                return Ok(StorageDtype::U8);
            }
            let choice = choose_auto_int(stats.min as i64, stats.max as i64, stats.has_missing);
            Ok(int_choice_to_dtype(choice))
        }
        explicit => {
            validate_explicit(explicit, stats, &field.name)?;
            Ok(explicit)
        }
    }
}

fn validate_explicit(
    dtype: StorageDtype,
    stats: &ScanStats,
    field_name: &str,
) -> Result<(), ConversionError> {
    if !stats.any_values {
        // No real values to range-check; a lone missing-only bool field has
        // nowhere to put a sentinel, but that's an existing-data edge case
        // we don't need to reject here (nothing conflicts with anything).
        return Ok(());
    }
    let min_i = stats.min as i64;
    let max_i = stats.max as i64;
    match dtype {
        StorageDtype::Bool => {
            if stats.has_missing {
                return Err(bad(format!(
                    "field {field_name:?}: explicit dtype bool has no spare value to encode a missing sentinel"
                )));
            }
            if min_i < 0 || max_i > 1 {
                return Err(bad(format!(
                    "field {field_name:?}: explicit dtype bool requires values in {{0,1}}, observed range [{}, {}]",
                    stats.min, stats.max
                )));
            }
        }
        StorageDtype::I8 | StorageDtype::I16 | StorageDtype::I32 => {
            let width = dtype.width_bytes().unwrap() as u32;
            if !signed_fits(width, min_i, max_i, stats.has_missing) {
                return Err(bad(format!(
                    "field {field_name:?}: explicit dtype {:?} cannot hold observed range [{}, {}]{}",
                    dtype,
                    stats.min,
                    stats.max,
                    if stats.has_missing {
                        " plus a reserved missing sentinel"
                    } else {
                        ""
                    }
                )));
            }
        }
        StorageDtype::U8 | StorageDtype::U16 | StorageDtype::U32 => {
            let width = dtype.width_bytes().unwrap() as u32;
            if !unsigned_fits(width, min_i, max_i, stats.has_missing) {
                return Err(bad(format!(
                    "field {field_name:?}: explicit dtype {:?} cannot hold observed range [{}, {}]{}",
                    dtype,
                    stats.min,
                    stats.max,
                    if stats.has_missing {
                        " plus a reserved missing sentinel"
                    } else {
                        ""
                    }
                )));
            }
        }
        StorageDtype::F16 => {
            if stats.max > 65504.0 || stats.min < -65504.0 {
                return Err(bad(format!(
                    "field {field_name:?}: explicit dtype f16 cannot hold observed range [{}, {}] (max magnitude 65504)",
                    stats.min, stats.max
                )));
            }
        }
        StorageDtype::F32 => {}
        StorageDtype::Auto => unreachable!("Auto is resolved separately, never validated here"),
    }
    Ok(())
}

fn rewrite_file(
    path: &Path,
    field: &FieldSpec,
    dtype: StorageDtype,
) -> Result<(), ConversionError> {
    let is_float = field.stage_is_float();
    let treat_missing = field.default.is_none();
    let values = read_staged(path, is_float)?;

    let width = dtype.width_bytes().unwrap_or(4);
    let mut out = Vec::with_capacity(values.len() * width);
    for v in values {
        let is_missing = treat_missing && is_staged_missing(v, is_float);
        encode(&mut out, dtype, v, is_missing);
    }

    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| ConversionError::Io {
            context: format!("rewriting {}", path.display()),
            source: e,
        })?;
    f.write_all(&out).map_err(|e| ConversionError::Io {
        context: format!("writing {}", path.display()),
        source: e,
    })?;
    f.flush().map_err(|e| ConversionError::Io {
        context: format!("flushing {}", path.display()),
        source: e,
    })?;
    Ok(())
}

/// Encode one resolved-dtype element into `out`. `v` is the staged value
/// (already validated to fit `dtype`); `is_missing` selects the resolved
/// sentinel encoding instead (`u*::MAX` for unsigned, `i*::MIN` for signed,
/// `NaN` for float/`f16`) — only ever `true` when the field has no
/// `default`, per the staging invariant.
fn encode(out: &mut Vec<u8>, dtype: StorageDtype, v: f64, is_missing: bool) {
    match dtype {
        StorageDtype::Bool => {
            out.push(if v != 0.0 { 1 } else { 0 });
        }
        StorageDtype::I8 => {
            let x: i8 = if is_missing { i8::MIN } else { v as i64 as i8 };
            out.push(x as u8);
        }
        StorageDtype::U8 => {
            let x: u8 = if is_missing { u8::MAX } else { v as i64 as u8 };
            out.push(x);
        }
        StorageDtype::I16 => {
            let x: i16 = if is_missing {
                i16::MIN
            } else {
                v as i64 as i16
            };
            out.extend_from_slice(&x.to_le_bytes());
        }
        StorageDtype::U16 => {
            let x: u16 = if is_missing {
                u16::MAX
            } else {
                v as i64 as u16
            };
            out.extend_from_slice(&x.to_le_bytes());
        }
        StorageDtype::I32 => {
            let x: i32 = if is_missing {
                i32::MIN
            } else {
                v as i64 as i32
            };
            out.extend_from_slice(&x.to_le_bytes());
        }
        StorageDtype::U32 => {
            let x: u32 = if is_missing {
                u32::MAX
            } else {
                v as i64 as u32
            };
            out.extend_from_slice(&x.to_le_bytes());
        }
        StorageDtype::F16 => {
            let x = if is_missing {
                half::f16::NAN
            } else {
                half::f16::from_f64(v)
            };
            out.extend_from_slice(&x.to_le_bytes());
        }
        StorageDtype::F32 => {
            let x: f32 = if is_missing { f32::NAN } else { v as f32 };
            out.extend_from_slice(&x.to_le_bytes());
        }
        StorageDtype::Auto => unreachable!("dtype is resolved (concrete) before rewrite"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    // Test fields below always pair `write_i32_field` with `int_field`
    // (category "info") and `write_f32_field` with `float_field` (category
    // "format"), so the category segment is hardcoded here to match.
    fn write_i32_field(root: &Path, contig: &str, name: &str, sub: &str, values: &[i32]) {
        let dir = root
            .join(contig)
            .join("fields")
            .join("info")
            .join(name)
            .join(sub);
        fs::create_dir_all(&dir).unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(dir.join("values.bin"), bytes).unwrap();
    }

    fn write_f32_field(root: &Path, contig: &str, name: &str, sub: &str, values: &[f32]) {
        let dir = root
            .join(contig)
            .join("fields")
            .join("format")
            .join(name)
            .join(sub);
        fs::create_dir_all(&dir).unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(dir.join("values.bin"), bytes).unwrap();
    }

    fn read_values_bin(path: &Path) -> Vec<u8> {
        fs::read(path).unwrap()
    }

    fn int_field(name: &str, dtype: StorageDtype, default: Option<f64>) -> FieldSpec {
        FieldSpec {
            name: name.to_string(),
            category: FieldCategory::Info,
            htype: HtslibType::Int,
            dtype,
            default,
        }
    }

    fn float_field(name: &str, dtype: StorageDtype, default: Option<f64>) -> FieldSpec {
        FieldSpec {
            name: name.to_string(),
            category: FieldCategory::Format,
            htype: HtslibType::Float,
            dtype,
            default,
        }
    }

    #[test]
    fn auto_int_narrows_to_u8_no_missing() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        write_i32_field(root, "chr1", "AC", "var_key_snp", &[0, 50, 200]);
        write_i32_field(root, "chr2", "AC", "var_key_snp", &[0, 5, 10]);

        let contigs = vec!["chr1".to_string(), "chr2".to_string()];
        let field = int_field("AC", StorageDtype::Auto, None);
        let resolved = finalize_fields(root, &contigs, &[field]).unwrap();

        assert_eq!(resolved.len(), 1);
        assert!(matches!(resolved[0].dtype, StorageDtype::U8));

        let path1 = root.join("chr1/fields/info/AC/var_key_snp/values.bin");
        let bytes1 = read_values_bin(&path1);
        assert_eq!(bytes1, vec![0u8, 50, 200]);

        let path2 = root.join("chr2/fields/info/AC/var_key_snp/values.bin");
        let bytes2 = read_values_bin(&path2);
        assert_eq!(bytes2, vec![0u8, 5, 10]);
    }

    #[test]
    fn auto_int_with_missing_reserves_sentinel_and_may_bump_width() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        // Values 0..=255 with a missing sentinel present: u8 can only carry
        // 255 distinct non-sentinel values (0..=254), so the real value 255
        // forces a bump to u16.
        write_i32_field(root, "chr1", "DP", "var_key_snp", &[0, 254, 255, i32::MIN]);

        let contigs = vec!["chr1".to_string()];
        let field = int_field("DP", StorageDtype::Auto, None);
        let resolved = finalize_fields(root, &contigs, &[field]).unwrap();

        assert!(matches!(resolved[0].dtype, StorageDtype::U16));

        let path = root.join("chr1/fields/info/DP/var_key_snp/values.bin");
        let bytes = read_values_bin(&path);
        let words: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(words, vec![0u16, 254, 255, u16::MAX]);
    }

    #[test]
    fn auto_int_narrows_to_u8_with_missing_when_range_leaves_room() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        // max 200 with missing: u8 needs only 201 slots (0..=200) plus one
        // sentinel slot (255) — fits in u8 (max usable non-sentinel is 254).
        write_i32_field(root, "chr1", "AF", "var_key_snp", &[0, 200, i32::MIN]);

        let contigs = vec!["chr1".to_string()];
        let field = int_field("AF", StorageDtype::Auto, None);
        let resolved = finalize_fields(root, &contigs, &[field]).unwrap();
        assert!(matches!(resolved[0].dtype, StorageDtype::U8));

        let path = root.join("chr1/fields/info/AF/var_key_snp/values.bin");
        let bytes = read_values_bin(&path);
        assert_eq!(bytes, vec![0u8, 200, u8::MAX]);
    }

    #[test]
    fn explicit_f16_in_range_rewrites_to_two_bytes() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        write_f32_field(root, "chr1", "DS", "dense_snp", &[0.0, 65504.0, -100.5]);

        let contigs = vec!["chr1".to_string()];
        let field = float_field("DS", StorageDtype::F16, None);
        let resolved = finalize_fields(root, &contigs, &[field]).unwrap();
        assert!(matches!(resolved[0].dtype, StorageDtype::F16));

        let path = root.join("chr1/fields/format/DS/dense_snp/values.bin");
        let bytes = read_values_bin(&path);
        assert_eq!(bytes.len(), 3 * 2);
        let words: Vec<half::f16> = bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(words[0].to_f32(), 0.0);
        assert_eq!(words[1].to_f32(), 65504.0);
        assert_eq!(words[2].to_f32(), -100.5);
    }

    #[test]
    fn explicit_f16_overflow_errors() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        write_f32_field(root, "chr1", "DS", "dense_snp", &[0.0, 70000.0]);

        let contigs = vec!["chr1".to_string()];
        let field = float_field("DS", StorageDtype::F16, None);
        let err = finalize_fields(root, &contigs, &[field]).unwrap_err();
        assert!(matches!(err, ConversionError::Input(_)));
    }

    #[test]
    fn explicit_int_overflow_errors() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        write_i32_field(root, "chr1", "XX", "var_key_snp", &[0, 300]);

        let contigs = vec!["chr1".to_string()];
        let field = int_field("XX", StorageDtype::I8, None);
        let err = finalize_fields(root, &contigs, &[field]).unwrap_err();
        assert!(matches!(err, ConversionError::Input(_)));
    }

    #[test]
    fn auto_int_negative_min_narrows_to_signed_i8() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        // min=-5, max=100, no missing: -5..100 fits i8's -128..=127 range, and
        // a negative min forces the Signed branch (unsigned_fits requires
        // min >= 0).
        write_i32_field(root, "chr1", "SC", "var_key_snp", &[-5, 0, 100]);

        let contigs = vec!["chr1".to_string()];
        let field = int_field("SC", StorageDtype::Auto, None);
        let resolved = finalize_fields(root, &contigs, &[field]).unwrap();

        assert!(
            matches!(resolved[0].dtype, StorageDtype::I8),
            "expected I8 (signed), got {:?}",
            resolved[0].dtype
        );

        let path = root.join("chr1/fields/info/SC/var_key_snp/values.bin");
        let bytes = read_values_bin(&path);
        let decoded: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(decoded, vec![-5i8, 0, 100]);
    }

    #[test]
    fn auto_int_negative_min_out_of_i8_range_bumps_to_signed_i16() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        // min=-200 doesn't fit i8 (-128..=127), forcing a bump to i16.
        write_i32_field(root, "chr1", "SC2", "var_key_snp", &[-200, 0, 100]);

        let contigs = vec!["chr1".to_string()];
        let field = int_field("SC2", StorageDtype::Auto, None);
        let resolved = finalize_fields(root, &contigs, &[field]).unwrap();

        assert!(
            matches!(resolved[0].dtype, StorageDtype::I16),
            "expected I16 (signed), got {:?}",
            resolved[0].dtype
        );

        let path = root.join("chr1/fields/info/SC2/var_key_snp/values.bin");
        let bytes = read_values_bin(&path);
        let decoded: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(decoded, vec![-200i16, 0, 100]);
    }

    #[test]
    fn default_set_values_treated_as_non_missing() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        // No i32::MIN sentinel is ever staged when `default` is set — every
        // element (including substituted defaults) is a real value.
        write_i32_field(root, "chr1", "GQ", "var_key_snp", &[0, 5, 0]);

        let contigs = vec!["chr1".to_string()];
        let field = int_field("GQ", StorageDtype::Auto, Some(0.0));
        let resolved = finalize_fields(root, &contigs, &[field]).unwrap();
        // No missing observed (by construction), so u8 needs no reserved
        // sentinel slot: max 5 fits trivially.
        assert!(matches!(resolved[0].dtype, StorageDtype::U8));
        assert_eq!(resolved[0].default, Some(0.0));
    }

    #[test]
    fn finalize_auto_narrow_byte_golden() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // i32 staged values incl. one MISSING sentinel; auto -> narrow.
        write_i32_field(root, "chr1", "AC", "var_key_snp", &[0, 5, 127, i32::MIN, 3]);
        let field = int_field("AC", StorageDtype::Auto, None);
        let resolved = finalize_fields(root, &[String::from("chr1")], &[field]).unwrap();
        let path = root
            .join("chr1")
            .join("fields")
            .join("info")
            .join("AC")
            .join("var_key_snp")
            .join("values.bin");
        let got = read_values_bin(&path);
        // Captured from a run on pre-refactor code (`cargo test --no-default-features
        // --features conversion --lib field_finalize::tests::finalize_auto_narrow_byte_golden
        // -- --nocapture`): values [0, 5, 127, i32::MIN, 3], no default => MIN is
        // the staged-missing sentinel. max=127 with has_missing=true still fits
        // u8 (0..=254 usable, sentinel reserved at 255) => resolves to U8, with
        // the sentinel remapped to u8::MAX=255.
        assert_eq!(resolved[0].dtype, StorageDtype::U8);
        assert_eq!(got, vec![0u8, 5, 127, 255, 3]);
    }

    #[test]
    fn explicit_f32_rewrites_in_place_byte_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let contig = "chr1";
        let vals: [f32; 4] = [0.5, -1.25, 0.0, 3.5];
        // stage under one var_key sub label
        write_f32_field(root, contig, "DS", "var_key_snp", &vals);
        let path = root
            .join(contig)
            .join("fields")
            .join("format")
            .join("DS")
            .join("var_key_snp")
            .join("values.bin");
        let before = read_values_bin(&path);

        let field = float_field("DS", StorageDtype::F32, None);
        let resolved = finalize_fields(root, &[contig.to_string()], &[field]).unwrap();

        assert_eq!(resolved[0].dtype, StorageDtype::F32);
        let after = read_values_bin(&path);
        // f32 staged -> f32 resolved: rewrite must reproduce the same 16 bytes.
        assert_eq!(before, after, "f32->f32 finalize must be byte-identical");
    }
}

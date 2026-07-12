//! Field-spec typing core for SVAR2 INFO/FORMAT field write support.
//!
//! Pure typing/validation over a manifest of `(name, category, htype, dtype,
//! default)` tuples produced on the Python side. No I/O here — this module
//! only maps user-facing strings to typed enums and validates them.

use crate::error::ConversionError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldCategory {
    Info,
    Format,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtslibType {
    Int,
    Float,
    Flag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDtype {
    Auto,
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F16,
    F32,
}

impl StorageDtype {
    /// Final on-disk width in bytes; `Auto` is undecided until finalize.
    pub fn width_bytes(self) -> Option<usize> {
        Some(match self {
            StorageDtype::Auto => return None,
            StorageDtype::Bool | StorageDtype::I8 | StorageDtype::U8 => 1,
            StorageDtype::I16 | StorageDtype::U16 | StorageDtype::F16 => 2,
            StorageDtype::I32 | StorageDtype::U32 | StorageDtype::F32 => 4,
        })
    }

    /// `true` iff the given htslib type should be staged as floating point.
    pub fn stage_is_float(htype: HtslibType) -> bool {
        htype == HtslibType::Float
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bool" => Self::Bool,
            "i8" => Self::I8,
            "u8" => Self::U8,
            "i16" => Self::I16,
            "u16" => Self::U16,
            "i32" => Self::I32,
            "u32" => Self::U32,
            "f16" => Self::F16,
            "f32" => Self::F32,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub name: String,
    pub category: FieldCategory,
    pub htype: HtslibType,
    pub dtype: StorageDtype,
    pub default: Option<f64>,
}

impl FieldSpec {
    pub fn stage_is_float(&self) -> bool {
        self.htype == HtslibType::Float
    }

    /// The sentinel a resolved field value falls back to when htslib reports
    /// it missing and the spec has no explicit `default`: htslib's own
    /// missing-int encoding for Int/Flag columns, NaN for Float columns
    /// (mirrors htslib's float-missing encoding, which is itself a NaN bit
    /// pattern). Also used as the non-carrier fill for genotype-aligned
    /// dense FORMAT columns.
    pub fn missing_sentinel(&self) -> f64 {
        self.default.unwrap_or(if self.stage_is_float() {
            f64::from(f32::NAN)
        } else {
            i32::MIN as f64
        })
    }
}

fn bad(ctx: &str) -> ConversionError {
    ConversionError::Input(ctx.to_string())
}

#[allow(clippy::type_complexity)]
pub fn parse_manifest(
    raw: Vec<(String, String, String, Option<String>, Option<f64>)>,
) -> Result<Vec<FieldSpec>, ConversionError> {
    raw.into_iter()
        .map(|(name, category, htype, dtype, default)| {
            let category = match category.as_str() {
                "info" => FieldCategory::Info,
                "format" => FieldCategory::Format,
                other => return Err(bad(&format!("bad field category {other:?}"))),
            };
            let htype = match htype.as_str() {
                "int" => HtslibType::Int,
                "float" => HtslibType::Float,
                "flag" => HtslibType::Flag,
                other => return Err(bad(&format!("bad htslib type {other:?}"))),
            };
            let dtype = match dtype {
                None => StorageDtype::Auto,
                Some(s) => StorageDtype::parse(&s)
                    .ok_or_else(|| bad(&format!("bad storage dtype {s:?}")))?,
            };
            Ok(FieldSpec {
                name,
                category,
                htype,
                dtype,
                default,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_manifest_maps_tuples() {
        let raw = vec![
            (
                "DS".into(),
                "format".into(),
                "float".into(),
                Some("f16".into()),
                Some(0.0),
            ),
            ("AC".into(), "info".into(), "int".into(), None, None),
        ];
        let specs = parse_manifest(raw).unwrap();
        assert_eq!(specs.len(), 2);
        assert!(matches!(specs[0].category, FieldCategory::Format));
        assert!(matches!(specs[0].dtype, StorageDtype::F16));
        assert!(specs[0].stage_is_float());
        assert!(matches!(specs[1].dtype, StorageDtype::Auto));
        assert!(!specs[1].stage_is_float());
    }

    #[test]
    fn parse_manifest_rejects_bad_category() {
        let raw = vec![("X".into(), "bogus".into(), "int".into(), None, None)];
        assert!(parse_manifest(raw).is_err());
    }
}

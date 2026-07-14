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

    /// Parse a finalized `meta.json` `fields[].dtype` string. Returns `None` for
    /// `"auto"` — finalize always resolves `Auto` to a concrete dtype, so `auto`
    /// on disk means the store is corrupt or was never finalized.
    pub fn from_meta_str(s: &str) -> Option<Self> {
        match Self::parse(s) {
            Some(Self::Auto) | None => None,
            Some(d) => Some(d),
        }
    }

    /// Lowercase on-disk/`meta.json` name. `Auto` never reaches here in
    /// practice (finalize always resolves it to a concrete dtype first), but
    /// maps to `"auto"` rather than panicking so this stays a total function.
    pub fn as_str(self) -> &'static str {
        match self {
            StorageDtype::Auto => "auto",
            StorageDtype::Bool => "bool",
            StorageDtype::I8 => "i8",
            StorageDtype::U8 => "u8",
            StorageDtype::I16 => "i16",
            StorageDtype::U16 => "u16",
            StorageDtype::I32 => "i32",
            StorageDtype::U32 => "u32",
            StorageDtype::F16 => "f16",
            StorageDtype::F32 => "f32",
        }
    }
}

impl FieldCategory {
    /// Lowercase on-disk/`meta.json` name.
    pub fn as_str(self) -> &'static str {
        match self {
            FieldCategory::Info => "info",
            FieldCategory::Format => "format",
        }
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

    /// Encode one scalar `v` into this (finalized) field's on-disk dtype bytes,
    /// applying the EXACT same per-element narrowing and sentinel-remapping
    /// `finalize_fields` uses — this delegates to `field_finalize::encode`
    /// rather than duplicating the dtype match, so a var_key -> dense
    /// non-carrier fill is byte-identical to what `rvk.rs`'s dense push (then
    /// finalize) writes for a genuine dense non-carrier.
    ///
    /// `self.dtype` must be concrete (post-finalize); `StorageDtype::Auto`
    /// panics inside `encode`, matching the rewrite path.
    ///
    /// A `v` equal to this field's staged missing sentinel is emitted as the
    /// resolved dtype's own sentinel (`u*::MAX` / `i*::MIN` / `NaN`) IFF the
    /// field has no `default` — identical to the finalize per-element rule
    /// (`treat_missing = default.is_none()`). With a `default` set, `v` (the
    /// default value) is encoded as an ordinary value.
    pub fn encode_scalar(&self, v: f64) -> Vec<u8> {
        let is_missing = self.default.is_none()
            && crate::field_finalize::is_staged_missing(v, self.stage_is_float());
        let mut out = Vec::with_capacity(self.dtype.width_bytes().unwrap_or(4));
        crate::field_finalize::encode(&mut out, self.dtype, v, is_missing);
        out
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
    fn encode_scalar_missing_matches_resolved_sentinel() {
        // Integer FORMAT field, no default: encoding its missing sentinel must
        // reproduce the resolved dtype's reserved sentinel (u16::MAX) — the
        // EXACT byte pattern `field_finalize::encode` writes for a dense
        // non-carrier (and hence what `rvk.rs`'s dense push -> finalize emits).
        let dp = FieldSpec {
            name: "DP".into(),
            category: FieldCategory::Format,
            htype: HtslibType::Int,
            dtype: StorageDtype::U16,
            default: None,
        };
        assert_eq!(
            dp.encode_scalar(dp.missing_sentinel()),
            u16::MAX.to_le_bytes().to_vec()
        );

        // Signed integer variant: i16::MIN.
        let sc = FieldSpec {
            name: "SC".into(),
            category: FieldCategory::Info,
            htype: HtslibType::Int,
            dtype: StorageDtype::I16,
            default: None,
        };
        assert_eq!(
            sc.encode_scalar(sc.missing_sentinel()),
            i16::MIN.to_le_bytes().to_vec()
        );

        // Float field, no default: the sentinel is NaN (bit-identity, so
        // compare via `is_nan()` not `==`).
        let af = FieldSpec {
            name: "AF".into(),
            category: FieldCategory::Info,
            htype: HtslibType::Float,
            dtype: StorageDtype::F32,
            default: None,
        };
        let bytes = af.encode_scalar(af.missing_sentinel());
        assert_eq!(bytes.len(), 4);
        let x = f32::from_le_bytes(bytes.try_into().unwrap());
        assert!(x.is_nan(), "float missing sentinel must be NaN");

        // With a `default`, the value is encoded ordinarily (not as a sentinel).
        let gq = FieldSpec {
            name: "GQ".into(),
            category: FieldCategory::Format,
            htype: HtslibType::Int,
            dtype: StorageDtype::U8,
            default: Some(0.0),
        };
        assert_eq!(gq.encode_scalar(gq.missing_sentinel()), vec![0u8]);
    }

    #[test]
    fn parse_manifest_rejects_bad_category() {
        let raw = vec![("X".into(), "bogus".into(), "int".into(), None, None)];
        assert!(parse_manifest(raw).is_err());
    }

    #[test]
    fn test_from_meta_str_rejects_auto() {
        assert_eq!(StorageDtype::from_meta_str("f16"), Some(StorageDtype::F16));
        assert_eq!(StorageDtype::from_meta_str("i8"), Some(StorageDtype::I8));
        assert_eq!(
            StorageDtype::from_meta_str("bool"),
            Some(StorageDtype::Bool)
        );
        // `auto` is never a finalized on-disk dtype.
        assert_eq!(StorageDtype::from_meta_str("auto"), None);
        assert_eq!(StorageDtype::from_meta_str("nonsense"), None);
    }
}

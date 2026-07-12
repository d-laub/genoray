//! Top-level `meta.json` writer. Written once by the conversion pipeline after
//! every contig succeeds — the only scope that knows the full samples / contigs
//! / ploidy. Rust never reads this back; Python does (via `json.load`).

use serde_json::json;
use std::path::Path;

/// Integer schema version for the on-disk SVAR2 layout, used by `SparseVar2` to
/// negotiate. Bump on any breaking layout/dtype change. Array dtypes are keyed
/// to this version (see `docs/roadmap/data-model.md`), not duplicated in the JSON.
pub const FORMAT_VERSION: u32 = 1;

/// Write the top-level `{output_dir}/meta.json`. Called once, after all contigs
/// convert successfully.
pub fn write_meta(
    output_dir: &Path,
    format_version: u32,
    samples: &[String],
    contigs: &[String],
    ploidy: usize,
    fields: &[crate::field_finalize::ResolvedField],
) -> std::io::Result<()> {
    let fields: Vec<_> = fields
        .iter()
        .map(|f| {
            json!({
                "name": f.name,
                "category": f.category.as_str(),
                "dtype": f.dtype.as_str(),
                "default": f.default,
            })
        })
        .collect();
    let meta = json!({
        "format_version": format_version,
        "samples": samples,
        "contigs": contigs,
        "ploidy": ploidy,
        "fields": fields,
    });
    // Serializing a plain serde_json::Value effectively cannot fail (no custom
    // Serialize), but propagate rather than panic to keep the io::Result contract
    // uniform; the real error is almost always the filesystem write below.
    let bytes = serde_json::to_vec_pretty(&meta).map_err(std::io::Error::other)?;
    std::fs::write(output_dir.join("meta.json"), bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{FieldCategory, StorageDtype};
    use crate::field_finalize::ResolvedField;
    use serde_json::Value;
    use tempfile::tempdir;

    #[test]
    fn test_write_meta_round_trip() {
        let tmp = tempdir().unwrap();
        let samples = vec!["s1".to_string(), "s2".to_string()];
        let contigs = vec!["chr1".to_string(), "chr2".to_string()];

        write_meta(tmp.path(), FORMAT_VERSION, &samples, &contigs, 2, &[]).unwrap();

        let text = std::fs::read_to_string(tmp.path().join("meta.json")).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["format_version"], 1);
        assert_eq!(v["samples"], serde_json::json!(["s1", "s2"]));
        assert_eq!(v["contigs"], serde_json::json!(["chr1", "chr2"]));
        assert_eq!(v["ploidy"], 2);
        assert_eq!(v["fields"], serde_json::json!([]));
    }

    #[test]
    fn test_write_meta_fields_manifest_round_trip() {
        let tmp = tempdir().unwrap();
        let samples = vec!["s1".to_string()];
        let contigs = vec!["chr1".to_string()];
        let fields = vec![
            ResolvedField {
                name: "DS".to_string(),
                category: FieldCategory::Format,
                dtype: StorageDtype::F32,
                default: Some(0.0),
            },
            ResolvedField {
                name: "AC".to_string(),
                category: FieldCategory::Info,
                dtype: StorageDtype::U8,
                default: None,
            },
        ];

        write_meta(tmp.path(), FORMAT_VERSION, &samples, &contigs, 2, &fields).unwrap();

        let text = std::fs::read_to_string(tmp.path().join("meta.json")).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            v["fields"][0],
            serde_json::json!({"name": "DS", "category": "format", "dtype": "f32", "default": 0.0})
        );
        assert_eq!(
            v["fields"][1],
            serde_json::json!({"name": "AC", "category": "info", "dtype": "u8", "default": null})
        );
        assert!(v["fields"][1]["default"].is_null());
    }
}

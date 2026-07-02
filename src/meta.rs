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
) -> std::io::Result<()> {
    let meta = json!({
        "format_version": format_version,
        "samples": samples,
        "contigs": contigs,
        "ploidy": ploidy,
    });
    // Serializing a serde_json::Value cannot fail (no custom Serialize), so the
    // only real error is the filesystem write.
    let bytes = serde_json::to_vec_pretty(&meta).expect("serialize meta.json value");
    std::fs::write(output_dir.join("meta.json"), bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tempfile::tempdir;

    #[test]
    fn test_write_meta_round_trip() {
        let tmp = tempdir().unwrap();
        let samples = vec!["s1".to_string(), "s2".to_string()];
        let contigs = vec!["chr1".to_string(), "chr2".to_string()];

        write_meta(tmp.path(), FORMAT_VERSION, &samples, &contigs, 2).unwrap();

        let text = std::fs::read_to_string(tmp.path().join("meta.json")).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["format_version"], 1);
        assert_eq!(v["samples"], serde_json::json!(["s1", "s2"]));
        assert_eq!(v["contigs"], serde_json::json!(["chr1", "chr2"]));
        assert_eq!(v["ploidy"], 2);
    }
}

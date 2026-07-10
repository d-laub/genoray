//! The encoding-agnostic seam's routing table. Orchestration, the writer, the
//! executor, and merge all iterate `REGISTRY` and index by `StreamTag` — they
//! never name a concrete key width or `snp`/`indel` directly. Adding the
//! `pointer` (M11) / `dense` (M4) representations means extending `StreamTag` +
//! `REGISTRY` and teaching `classify_variant` to route; nothing else changes.

use std::path::Path;

use crate::enum_map::{EnumKey, EnumMap};
use crate::rvk::pack_snp_key_file;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamTag {
    VarKeySnp = 0,
    VarKeyIndel = 1,
}

impl EnumKey for StreamTag {
    const COUNT: usize = 2;
    const ALL: &'static [StreamTag] = &[StreamTag::VarKeySnp, StreamTag::VarKeyIndel];
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

/// Post-merge rewrite hook applied to a stream's final files (e.g. 2-bit packing).
pub type PostMergeHook = fn(&Path) -> Result<(), crate::error::ConversionError>;

pub struct StreamSpec {
    pub tag: StreamTag,
    pub subdir: &'static str,
    pub key_bytes: usize,
    /// Post-merge rewrite hook applied to the stream's final files (e.g. 2-bit packing).
    pub post_merge: Option<PostMergeHook>,
}

/// One entry per active on-disk sub-stream. Order matches `StreamTag as usize`.
pub const REGISTRY: [StreamSpec; StreamTag::COUNT] = [
    StreamSpec {
        tag: StreamTag::VarKeySnp,
        subdir: "var_key/snp",
        key_bytes: 1,
        post_merge: Some(pack_snp_key_file),
    },
    StreamSpec {
        tag: StreamTag::VarKeyIndel,
        subdir: "var_key/indel",
        key_bytes: 4,
        post_merge: None,
    },
];

/// Fixed-size map keyed by `StreamTag`, backed by an array (O(1), no hashing).
pub type StreamMap<T> = EnumMap<StreamTag, T, { StreamTag::COUNT }>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_indices_match_tags() {
        for spec in &REGISTRY {
            assert_eq!(REGISTRY[spec.tag.index()].tag, spec.tag);
        }
    }

    #[test]
    fn test_registry_key_widths() {
        assert_eq!(REGISTRY[StreamTag::VarKeySnp.index()].key_bytes, 1);
        assert_eq!(REGISTRY[StreamTag::VarKeyIndel.index()].key_bytes, 4);
    }

    #[test]
    fn test_only_snp_has_post_merge() {
        assert!(REGISTRY[StreamTag::VarKeySnp.index()].post_merge.is_some());
        assert!(
            REGISTRY[StreamTag::VarKeyIndel.index()]
                .post_merge
                .is_none()
        );
    }

    #[test]
    fn test_streammap_get_set() {
        let mut m: StreamMap<u32> = StreamMap::from_fn(|_| 0);
        *m.get_mut(StreamTag::VarKeyIndel) = 7;
        assert_eq!(*m.get(StreamTag::VarKeySnp), 0);
        assert_eq!(*m.get(StreamTag::VarKeyIndel), 7);
        let collected: Vec<_> = m.iter().map(|(_, v)| *v).collect();
        assert_eq!(collected, vec![0, 7]);
    }
}

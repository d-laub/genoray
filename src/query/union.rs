//! `DenseUnion`: the per-contig dense table unioned across `snp`+`indel`,
//! position-sorted, used by the union-based (non-read-bound) query paths
//! (`oracle::overlap_sample`, `gather::overlap_batch`, `gather::gather_ranges`).

use crate::rvk;
use crate::search::{SearchTree, overlap_range};
use crate::spine::KeyRef;

use super::reader::ContigReader;
use super::sidecar::{as_bytes, as_u32};

/// The per-contig dense table unioned across `snp`+`indel`, position-sorted,
/// carrying uniform keys plus the `(is_indel, col)` needed to test carriage.
/// Region-independent — built once per query; `overlap` derives each region's
/// index range from it. `src[i] = (is_indel, col)` addresses the original dense
/// class table for the genotype-bit test.
pub(crate) struct DenseUnion {
    pub(crate) refs: Vec<KeyRef>,
    pub(crate) src: Vec<(bool, usize)>,
    positions: Vec<u32>,
    pub(crate) v_ends: Vec<u32>,
    max_del: u32,
}

impl DenseUnion {
    /// `[s, e)` into `refs`/`src` for `[q_start, q_end)`, deletion-aware. Builds a
    /// fresh search tree over `positions` (cheap; one per region in a batch).
    pub(crate) fn overlap(&self, q_start: u32, q_end: u32) -> (usize, usize) {
        if self.refs.is_empty() {
            return (0, 0);
        }
        let tree = SearchTree::new(&self.positions);
        overlap_range(&tree, &self.v_ends, self.max_del, q_start, q_end)
    }
}

impl ContigReader {
    /// Build the region-independent dense `snp`+`indel` union (see `DenseUnion`).
    /// SNP codes re-expand to uniform keys; the max_region_length bound is the
    /// per-contig dense/indel max (SNP contributes 0).
    pub(crate) fn dense_union(&self) -> DenseUnion {
        // (position, key, del_len, is_indel, col), snp pushed before indel so a
        // stable sort keeps snp-before-indel on any shared position.
        let mut items: Vec<(u32, u32, u32, bool, usize)> = Vec::new();
        if let Some(d) = &self.dense_snp {
            let positions = d.positions();
            let keys = as_bytes(&d.keys);
            for (col, &pos) in positions.iter().enumerate() {
                let key = rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, col));
                items.push((pos, key, 0, false, col));
            }
        }
        if let Some(d) = &self.dense_indel {
            let positions = d.positions();
            let keys = as_u32(&d.keys);
            // Fail fast on a corrupt sidecar: `zip` would otherwise silently
            // truncate to the shorter of the two instead of panicking like the
            // pre-refactor indexed loop did.
            debug_assert_eq!(positions.len(), keys.len());
            for (col, (&pos, &key)) in positions.iter().zip(keys.iter()).enumerate() {
                items.push((pos, key, rvk::deletion_len(key), true, col));
            }
        }
        items.sort_by_key(|it| it.0);

        let refs = items
            .iter()
            .map(|it| KeyRef {
                position: it.0,
                key: it.1,
            })
            .collect();
        let positions = items.iter().map(|it| it.0).collect();
        let v_ends = items.iter().map(|it| it.0 + 1 + it.2).collect();
        let src = items.iter().map(|it| (it.3, it.4)).collect();
        DenseUnion {
            refs,
            src,
            positions,
            v_ends,
            max_del: self.dense_indel_max_del,
        }
    }
}

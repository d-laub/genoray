//! `DenseUnion`: the per-contig dense table unioned across `snp`+`indel`,
//! position-sorted, used by the union-based (non-read-bound) query paths
//! (`oracle::overlap_sample`, `gather::overlap_batch`, `gather::gather_ranges`).

use std::borrow::Cow;
use std::ops::Range;

use crate::dense::DenseClass;
use crate::query::gather::MAX_END_SHIFT;
use crate::rvk;
use crate::spine::KeyRef;

use super::reader::{ContigReader, OverlapIndex};
use super::sidecar::{as_bytes, as_u32};

/// The per-contig dense table unioned across `snp`+`indel`, position-sorted,
/// carrying uniform keys plus the `(DenseClass, col)` needed to test carriage.
/// Region-independent and tree-free — built once per query; `index()` adds
/// the search state for callers that range-query it. `src[i] = (DenseClass,
/// col)` addresses the original dense class table for the genotype-bit test.
pub(crate) struct DenseUnion {
    pub(crate) refs: Vec<KeyRef>,
    pub(crate) src: Vec<(DenseClass, usize)>,
    positions: Vec<u32>,
    pub(crate) v_ends: Vec<u32>,
    max_del: u32,
}

impl DenseUnion {
    /// Region-independent search state over the union, built once by the
    /// callers that actually search it.
    ///
    /// This is deliberately NOT built inside `dense_union()`: `gather_ranges`,
    /// `dense_max_end_keys` and `ContigReader::max_deletion_len` all construct
    /// a `DenseUnion` without ever overlapping it, and must not be charged a
    /// `SearchTree::new` they never use. Borrows `v_ends`, so building an index
    /// copies nothing.
    pub(crate) fn index(&self) -> OverlapIndex<'_> {
        if self.refs.is_empty() {
            return OverlapIndex::empty(0);
        }
        OverlapIndex::new(
            0,
            &self.positions,
            Cow::Borrowed(self.v_ends.as_slice()),
            self.max_del,
        )
    }

    /// The per-contig dense deletion bound, for the caller's overflow preflight.
    pub(crate) fn max_del(&self) -> u32 {
        self.max_del
    }
}

impl ContigReader {
    /// Build the region-independent dense `snp`+`indel` union (see `DenseUnion`).
    /// SNP codes re-expand to uniform keys; the max_region_length bound is the
    /// per-contig dense/indel max (SNP contributes 0).
    pub(crate) fn dense_union(&self) -> DenseUnion {
        // (position, key, del_len, class, col), snp pushed before indel so a
        // stable sort keeps snp-before-indel on any shared position.
        let mut items: Vec<(u32, u32, u32, DenseClass, usize)> = Vec::new();
        if let Some(d) = &self.dense_snp {
            let positions = d.positions();
            let keys = as_bytes(&d.keys);
            for (col, &pos) in positions.iter().enumerate() {
                let key = rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, col));
                items.push((pos, key, 0, DenseClass::Snp, col));
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
                items.push((pos, key, rvk::deletion_len(key), DenseClass::Indel, col));
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

/// Per-region max `(pos << MAX_END_SHIFT) | ext` over the DENSE channel,
/// restricted to variants carried by at least one selected hap. `0` when the
/// region has no such variant.
///
/// The dense genotype matrix is hap-major (`hap * n_dense_variants + col`), so a
/// "is this variant carried by anyone selected?" probe is strided across haps.
/// Two things keep that cheap:
///
/// * The walk runs BACKWARD from the end of the region's dense window and stops
///   once it drops below the position of the first carried variant it found.
///   Dense variants are common by construction, so this almost always terminates
///   on the first index.
/// * `all_samples` skips the carriage probe entirely: every dense variant in the
///   store has at least one carrier among all samples, so the last truly
///   overlapping variant is the answer. This is the path `gvl.write` takes.
///
/// The whole tied run at the winning position is scanned rather than stopping at
/// the first hit: within a class the table's order is not by `ext`, so a later
/// same-position variant can carry a longer deletion.
pub fn dense_max_end_keys(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    dense_range: &[Range<usize>],
    sample_cols: &[usize],
    all_samples: bool,
) -> Vec<u64> {
    let ploidy = reader.ploidy;
    let dense = reader.dense_union();
    let mut out = vec![0u64; regions.len()];

    for (ri, &(qs, _)) in regions.iter().enumerate() {
        let (ds, de) = (dense_range[ri].start, dense_range[ri].end);
        let mut best = 0u64;
        let mut best_pos: Option<u32> = None;
        let mut j = de;
        while j > ds {
            j -= 1;
            let pos = dense.refs[j].position;
            if let Some(bp) = best_pos
                && pos < bp
            {
                break; // every remaining index has a lower position
            }
            if dense.v_ends[j] <= qs {
                continue; // no true left-overlap
            }
            let carried = all_samples || {
                let (class, dcol) = dense.src[j];
                let view = reader.dense_view(class).expect("dense src implies table");
                sample_cols
                    .iter()
                    .any(|&s| (0..ploidy).any(|p| view.carried(s * ploidy + p, dcol)))
            };
            if !carried {
                continue;
            }
            let ext = (dense.v_ends[j] - pos) as u64;
            best = best.max(((pos as u64) << MAX_END_SHIFT) | ext);
            best_pos = Some(pos);
        }
        out[ri] = best;
    }
    out
}

//! `ContigReader`: opens a finished SVAR2 contig's mmap'd sidecars and
//! provides the per-hap var_key slice + search-range helpers that back both
//! the union (`union.rs`) and gather (`gather.rs`) query paths.

use std::ops::Range;
use std::path::Path;

use ndarray::Array2;

use crate::dense::DenseClass;
use crate::layout::{self, ContigPaths};
use crate::nrvk::LongAlleleReader;
use crate::rvk;
use crate::search::{SearchTree, overlap_range};
use crate::spine::{self, KeyRef};

use super::sidecar::{
    DenseView, SubStreamView, as_bytes, as_u32, load_dense_max_del, load_max_del, load_offsets,
    mmap_file, open_dense,
};
use super::union::DenseUnion;

/// Opens a finished SVAR2 contig directory and holds its sidecars mmap'd for the
/// lifetime of queries against it.
pub struct ContigReader {
    pub(crate) ploidy: usize,
    pub(crate) n_samples: usize,
    pub(crate) vk_snp: SubStreamView,
    pub(crate) vk_indel: SubStreamView,
    pub(crate) dense_snp: Option<DenseView>,
    pub(crate) dense_indel: Option<DenseView>,
    /// `(n_samples, ploidy)` per-column max deletion length for var_key/indel.
    vk_indel_max_del: Array2<u32>,
    /// Per-contig max deletion length over the shared dense/indel table.
    pub(crate) dense_indel_max_del: u32,
    /// Long-allele bank reader; present iff the shared indel LUT exists.
    pub(crate) lut: Option<LongAlleleReader>,
}

impl ContigReader {
    /// Open the contig `{base_out_dir}/{chrom}` for a cohort of `n_samples`
    /// samples at `ploidy`. Missing sub-streams (pure-SNP contigs, absent dense
    /// dirs) are tolerated as empty.
    pub fn open(
        base_out_dir: &str,
        chrom: &str,
        n_samples: usize,
        ploidy: usize,
    ) -> std::io::Result<Self> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        let contig_dir = Path::new(base_out_dir).join(chrom);
        let columns = n_samples * ploidy;

        let vk_snp_dir = paths.var_key_snp_dir();
        let vk_indel_dir = paths.var_key_indel_dir();

        let vk_snp = SubStreamView {
            positions: mmap_file(&layout::positions(&vk_snp_dir))?,
            keys: mmap_file(&layout::alleles(&vk_snp_dir))?,
            offsets: load_offsets(&layout::offsets(&vk_snp_dir), columns)?,
        };
        let vk_indel = SubStreamView {
            positions: mmap_file(&layout::positions(&vk_indel_dir))?,
            keys: mmap_file(&layout::alleles(&vk_indel_dir))?,
            offsets: load_offsets(&layout::offsets(&vk_indel_dir), columns)?,
        };

        let dense_snp = open_dense(&paths.dense_snp_dir())?;
        let dense_indel = open_dense(&paths.dense_indel_dir())?;

        let vk_indel_max_del = load_max_del(&layout::max_del(&contig_dir), n_samples, ploidy)?;
        let dense_indel_max_del = load_dense_max_del(&layout::dense_max_del(&contig_dir))?;

        let lut = if paths.long_alleles_bin().exists() {
            Some(LongAlleleReader::new(base_out_dir, chrom)?)
        } else {
            None
        };

        Ok(Self {
            ploidy,
            n_samples,
            vk_snp,
            vk_indel,
            dense_snp,
            dense_indel,
            vk_indel_max_del,
            dense_indel_max_del,
            lut,
        })
    }
}

impl ContigReader {
    /// Raw long-allele LUT for the M6b contract: all bytes + CSR row offsets.
    /// A contig with no LUT returns empty bytes and a single `[0]` offset (an
    /// empty CSR), so the numpy side never special-cases a missing file.
    pub fn lut_arrays(&self) -> (Vec<u8>, Vec<u64>) {
        match &self.lut {
            Some(l) => (l.all_bytes(), l.offsets().to_vec()),
            None => (Vec::new(), vec![0u64]),
        }
    }

    /// The dense view backing `class`, or `None` if this contig has no table of
    /// that class. Replaces the `if is_indel { &self.dense_indel } else { ... }`
    /// dispatch that a bool-and-col src pair forced at every carriage test.
    pub(crate) fn dense_view(&self, class: DenseClass) -> Option<&DenseView> {
        match class {
            DenseClass::Snp => self.dense_snp.as_ref(),
            DenseClass::Indel => self.dense_indel.as_ref(),
        }
    }
}

impl ContigReader {
    /// `var_key` channel for one flat hap-column over one region: `snp`+`indel`
    /// unioned, uniform keys (SNP re-expanded via `snp_code_to_key`), sorted.
    pub(crate) fn vk_slice(
        &self,
        col: usize,
        sample: usize,
        p: usize,
        q_start: u32,
        q_end: u32,
    ) -> Vec<KeyRef> {
        let mut runs: Vec<Vec<KeyRef>> = Vec::with_capacity(2);

        // var_key/snp: 2-bit codes, absolute index o0 + i into the packed buffer.
        {
            let vk_range = self.vk_snp.column(col);
            let (o0, o1) = (vk_range.start, vk_range.end);
            let positions = &self.vk_snp.positions()[o0..o1];
            let keys = as_bytes(&self.vk_snp.keys);
            let mut run = Vec::new();
            spine::gather_keys(
                positions,
                0,
                q_start,
                q_end,
                |_| 0,
                |_| true,
                |i| rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, o0 + i)),
                &mut run,
            );
            runs.push(run);
        }

        // var_key/indel: uniform u32 keys, per-column max_del bound.
        {
            let vk_range = self.vk_indel.column(col);
            let (o0, o1) = (vk_range.start, vk_range.end);
            let positions = &self.vk_indel.positions()[o0..o1];
            let keys = &as_u32(&self.vk_indel.keys)[o0..o1];
            let max_del = self.vk_indel_max_del[[sample, p]];
            let mut run = Vec::new();
            spine::gather_keys(
                positions,
                max_del,
                q_start,
                q_end,
                |i| rvk::deletion_len(keys[i]),
                |_| true,
                |i| keys[i],
                &mut run,
            );
            runs.push(run);
        }

        spine::merge_keys(runs)
    }

    /// The dense variants in `union[s..e]` that `hap` carries and that TRULY
    /// overlap `[q_start, q_end)` (exact left-overlap `q_start < v_end`; the
    /// right half is guaranteed by `union.overlap`'s window — see
    /// `spine::gather_keys`'s doc comment for why the per-element check is
    /// still needed within a carried window).
    pub(crate) fn dense_carried(
        &self,
        union: &DenseUnion,
        hap: usize,
        s: usize,
        e: usize,
        q_start: u32,
    ) -> Vec<KeyRef> {
        let mut out = Vec::new();
        for j in s..e {
            if union.v_ends[j] <= q_start {
                continue;
            }
            let (class, col) = union.src[j];
            let carried = self
                .dense_view(class)
                .expect("dense src implies table")
                .carried(hap, col);
            if carried {
                out.push(union.refs[j]);
            }
        }
        out
    }
}

impl ContigReader {
    /// Absolute `[start, end)` into `vk_snp`'s packed positions/keys for
    /// `(col, region)` — the SNP channel's search half (`max_region_length =
    /// 0`, since a SNP always spans exactly one base). No gather.
    pub(crate) fn vk_snp_overlap(&self, col: usize, q_start: u32, q_end: u32) -> Range<usize> {
        let vk_range = self.vk_snp.column(col);
        let (o0, o1) = (vk_range.start, vk_range.end);
        let positions = &self.vk_snp.positions()[o0..o1];
        if positions.is_empty() {
            return o0..o0;
        }
        let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        let tree = SearchTree::new(positions);
        let (s, e) = overlap_range(&tree, &v_ends, 0, q_start, q_end);
        (o0 + s)..(o0 + e)
    }

    /// Absolute `[start, end)` into `vk_indel`'s packed positions/keys for
    /// `(col, region)` — the indel channel's search half (per-column
    /// `max_del` bound). No gather.
    pub(crate) fn vk_indel_overlap(
        &self,
        col: usize,
        sample: usize,
        p: usize,
        q_start: u32,
        q_end: u32,
    ) -> Range<usize> {
        let vk_range = self.vk_indel.column(col);
        let (o0, o1) = (vk_range.start, vk_range.end);
        let positions = &self.vk_indel.positions()[o0..o1];
        if positions.is_empty() {
            return o0..o0;
        }
        let keys = &as_u32(&self.vk_indel.keys)[o0..o1];
        let max_del = self.vk_indel_max_del[[sample, p]];
        let v_ends: Vec<u32> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| pos + 1 + rvk::deletion_len(keys[i]))
            .collect();
        let tree = SearchTree::new(positions);
        let (s, e) = overlap_range(&tree, &v_ends, max_del, q_start, q_end);
        (o0 + s)..(o0 + e)
    }

    /// Absolute `[s, e)` into `dense/snp`'s positions/keys for one region.
    /// SNP v_end = pos + 1 (max_region_length = 0). `(0, 0)` if no snp table.
    pub(crate) fn dense_snp_overlap(&self, q_start: u32, q_end: u32) -> Range<usize> {
        let d = match &self.dense_snp {
            Some(d) => d,
            None => return 0..0,
        };
        let positions = d.positions();
        if positions.is_empty() {
            return 0..0;
        }
        let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        let tree = SearchTree::new(positions);
        let (s, e) = overlap_range(&tree, &v_ends, 0, q_start, q_end);
        s..e
    }

    /// Absolute `[s, e)` into `dense/indel`'s positions/keys for one region.
    /// Indel v_end = pos + 1 + deletion_len(key); per-contig dense max_del bound.
    pub(crate) fn dense_indel_overlap(&self, q_start: u32, q_end: u32) -> Range<usize> {
        let d = match &self.dense_indel {
            Some(d) => d,
            None => return 0..0,
        };
        let positions = d.positions();
        if positions.is_empty() {
            return 0..0;
        }
        let keys = as_u32(&d.keys);
        debug_assert_eq!(positions.len(), keys.len());
        let v_ends: Vec<u32> = positions
            .iter()
            .zip(keys.iter())
            .map(|(&pos, &key)| pos + 1 + rvk::deletion_len(key))
            .collect();
        let tree = SearchTree::new(positions);
        let (s, e) = overlap_range(&tree, &v_ends, self.dense_indel_max_del, q_start, q_end);
        s..e
    }
}

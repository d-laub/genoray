//! Batched multi-region × whole-cohort/subset query paths: the two
//! `BatchResult` variants (union-dense `BatchResult`, split-dense
//! `BatchResultSplit`), the search/gather split (`find_ranges` +
//! `gather_ranges`/`gather_haps_readbound`), and `BatchResult::decode_hap`'s
//! per-hap decode.

use std::ops::Range;

use crate::bits;
use crate::rvk;
use crate::spine::{self, KeyRef};

use super::decode::{HapCalls, decode_keyref};
use super::reader::ContigReader;
use super::sidecar::{as_bytes, as_u32};

/// CSR presence-bitmask accumulator: owns the `(bits, offsets)` pair, one row of
/// `nbits` bits appended per hap. `offsets` starts `[0]`; after each `push_hap`,
/// `offsets.last()` is the total bit count.
pub(crate) struct PresenceBitWriter {
    bits: Vec<u8>,
    offsets: Vec<usize>,
}

impl PresenceBitWriter {
    pub(crate) fn new() -> Self {
        Self {
            bits: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Reserve one hap row of `nbits` bits: grow `bits` to fit, record the new
    /// end offset, and return this row's starting bit offset.
    fn reserve_row(&mut self, nbits: usize) -> usize {
        let base = *self.offsets.last().unwrap();
        let need_bytes = (base + nbits).div_ceil(8);
        if self.bits.len() < need_bytes {
            self.bits.resize(need_bytes, 0);
        }
        self.offsets.push(base + nbits);
        base
    }

    /// Append one hap row of `nbits` bits; `set` is called with each in-row
    /// index `k in 0..nbits` and must return whether bit `k` is present.
    pub(crate) fn push_hap(&mut self, nbits: usize, mut set: impl FnMut(usize) -> bool) {
        let base = self.reserve_row(nbits);
        for k in 0..nbits {
            if set(k) {
                bits::set_bit(&mut self.bits, base + k);
            }
        }
    }

    /// Append one hap row of `nbits` bits, filled in bulk: `fill(bits, base)`
    /// receives the (resized) bit buffer and this row's starting bit offset
    /// `base`, and is responsible for setting whichever bits are present
    /// (e.g. via a `copy_bits` block copy). For the per-bit case use `push_hap`.
    pub(crate) fn push_hap_bulk(&mut self, nbits: usize, fill: impl FnOnce(&mut [u8], usize)) {
        let base = self.reserve_row(nbits);
        fill(&mut self.bits, base);
    }

    /// Consume into the `(present, present_off)` fields the result structs expect.
    pub(crate) fn into_parts(self) -> (Vec<u8>, Vec<usize>) {
        (self.bits, self.offsets)
    }
}

/// The batched, two-channel query spine result for one contig (M6.1). Carries
/// undecoded uniform keys; consumers (gvl M6b / Python M6c) do the final
/// `var_key ⋈ dense` merge and allele decode. `H = n_regions * n_samples *
/// ploidy` hap-slices in region-major order `h = (r * n_samples + s) * ploidy + p`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchResult {
    pub n_regions: usize,
    pub n_samples: usize,
    pub ploidy: usize,
    /// Flat var_key channel; `vk_off` (len H+1, CSR) slices it per hap.
    pub vk: Vec<KeyRef>,
    pub vk_off: Vec<usize>,
    /// Shared dense union (uniform keys, position-sorted); decode-once.
    pub dense: Vec<KeyRef>,
    /// `[s, e)` into `dense` per region (len n_regions).
    pub dense_range: Vec<Range<usize>>,
    /// Per-hap dense presence bitmask over that region's `dense[s..e]`, LSB-first,
    /// concatenated; `dense_present_off` (len H+1) holds BIT offsets.
    pub dense_present: Vec<u8>,
    pub dense_present_off: Vec<usize>,
}

/// Read-bound analog of `BatchResult`: the var_key channel merged per hap (as
/// today), but the dense channel **split per class** so no contig-wide
/// `DenseUnion` is built. gvl merges `var_key ⋈ dense_snp ⋈ dense_indel` by
/// position downstream. `H = n_regions * n_samples * ploidy`, hap index
/// `(r*n_samples + s)*ploidy + p` over the *selected* samples.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchResultSplit {
    pub n_regions: usize,
    pub n_samples: usize,
    pub ploidy: usize,
    /// Flat merged var_key channel (snp+indel per hap); `vk_off` (len H+1) slices it.
    pub vk: Vec<KeyRef>,
    pub vk_off: Vec<usize>,
    /// Per-region `dense/snp` windows (uniform keys), concatenated.
    pub dense_snp: Vec<KeyRef>,
    /// `[s, e)` into `dense_snp` per region (len n_regions).
    pub dense_snp_range: Vec<Range<usize>>,
    /// Per-hap presence bitmask over that region's `dense_snp[s..e]`, LSB-first;
    /// `dense_snp_present_off` (len H+1) holds BIT offsets.
    pub dense_snp_present: Vec<u8>,
    pub dense_snp_present_off: Vec<usize>,
    /// Per-region `dense/indel` windows (uniform u32 keys), concatenated.
    pub dense_indel: Vec<KeyRef>,
    pub dense_indel_range: Vec<Range<usize>>,
    pub dense_indel_present: Vec<u8>,
    pub dense_indel_present_off: Vec<usize>,
}

/// The var_key channel for one flat hap-column over one region window: SNP and
/// indel packed slices decoded to uniform `KeyRef`s and merged position-sorted.
/// This is the shared body the batch/ranges/read-bound gathers previously
/// hand-inlined. `vk_slice` already performs the union+merge for a column; this
/// wrapper is the seam the read-bound path replays from precomputed ranges.
pub(crate) fn gather_vk(
    reader: &ContigReader,
    vk_snp_range: Range<usize>,
    vk_indel_range: Range<usize>,
    q_start: u32,
) -> Vec<KeyRef> {
    let snp_positions = reader.vk_snp.positions();
    let snp_keys = as_bytes(&reader.vk_snp.keys);
    let indel_positions = reader.vk_indel.positions();
    let indel_keys = as_u32(&reader.vk_indel.keys);

    let (ss, se) = (vk_snp_range.start, vk_snp_range.end);
    let mut snp_run: Vec<KeyRef> = Vec::new();
    for (j, &pos) in snp_positions.iter().enumerate().take(se).skip(ss) {
        if q_start < pos + 1 {
            // snp v_end = pos + 1
            snp_run.push(KeyRef {
                position: pos,
                key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
            });
        }
    }

    let (is_, ie_) = (vk_indel_range.start, vk_indel_range.end);
    let mut indel_run: Vec<KeyRef> = Vec::new();
    for j in is_..ie_ {
        let pos = indel_positions[j];
        let v_end = pos + 1 + rvk::deletion_len(indel_keys[j]);
        if q_start < v_end {
            indel_run.push(KeyRef {
                position: pos,
                key: indel_keys[j],
            });
        }
    }

    spine::merge_keys(vec![snp_run, indel_run])
}

/// Batched multi-region × whole-cohort query. `regions` is a list of half-open
/// `[q_start, q_end)`. Single-threaded; `rayon` over the H hap-slices and dense-
/// window subsetting are M6b concerns (see the design spec's open questions).
pub fn overlap_batch(reader: &ContigReader, regions: &[(u32, u32)]) -> BatchResult {
    let ploidy = reader.ploidy;
    let n_samples = reader.n_samples;
    let n_regions = regions.len();

    let dense = reader.dense_union();
    // Per-region dense index ranges — shared across all samples in the region.
    let ranges: Vec<Range<usize>> = regions
        .iter()
        .map(|&(qs, qe)| dense.overlap(qs, qe))
        .collect();

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut presence = PresenceBitWriter::new();

    for (r, &(qs, qe)) in regions.iter().enumerate() {
        let (ds, de) = (ranges[r].start, ranges[r].end);
        for s in 0..n_samples {
            for p in 0..ploidy {
                let col = s * ploidy + p;

                // var_key channel slice for (region r, sample s, ploid p).
                let slice = reader.vk_slice(col, s, p, qs, qe);
                vk.extend_from_slice(&slice);
                vk_off.push(vk.len());

                // dense presence bits over dense[ds..de].
                let nbits = de - ds;
                presence.push_hap(nbits, |k| {
                    let j = ds + k;
                    let (class, dcol) = dense.src[j];
                    let carried = reader
                        .dense_view(class)
                        .expect("dense src implies table")
                        .carried(col, dcol);
                    carried && dense.v_ends[j] > qs
                });
            }
        }
    }

    let (dense_present, dense_present_off) = presence.into_parts();

    BatchResult {
        n_regions,
        n_samples,
        ploidy,
        vk,
        vk_off,
        dense: dense.refs,
        dense_range: ranges,
        dense_present,
        dense_present_off,
    }
}

/// Search-only half of the batch query: every `SearchTree::new` runs here, and
/// the result is a compact bundle of index ranges that `gather_ranges` replays
/// with no further search.
///
/// `H = n_samples * ploidy` (subset-aware: `n_samples` counts only the
/// *selected* samples); row index into `vk_snp_range`/`vk_indel_range` is
/// `r * H + si * ploidy + p`, where `si` is the selected slot (not the
/// original sample index — see `sample_cols`).
pub struct RangesBundle {
    pub n_regions: usize,
    pub n_samples: usize,
    pub ploidy: usize,
    /// `q_start` per region — needed by `gather_ranges`'s left-overlap re-check.
    pub region_starts: Vec<u32>,
    /// `[s, e)` into the shared dense union, per region.
    pub dense_range: Vec<Range<usize>>,
    /// Selected slot -> original sample index.
    pub sample_cols: Vec<usize>,
    /// Absolute `[start, end)` into `vk_snp`'s packed positions/keys, per
    /// `(region, selected sample, ploid)`.
    pub vk_snp_range: Vec<Range<usize>>,
    /// Absolute `[start, end)` into `vk_indel`'s packed positions/keys, per
    /// `(region, selected sample, ploid)`.
    pub vk_indel_range: Vec<Range<usize>>,
    /// `[s, e)` into `dense/snp`'s on-disk positions/keys, per region (dense is
    /// cohort-shared, so one window per region, not per hap). Read-bound path.
    pub dense_snp_range: Vec<Range<usize>>,
    /// `[s, e)` into `dense/indel`'s on-disk positions/keys, per region.
    pub dense_indel_range: Vec<Range<usize>>,
}

/// Search-only pass: run every `SearchTree::new` up front and record the
/// resulting index ranges. `find_ranges` owns ALL tree builds for the batch
/// query (region-level dense union overlap + per-hap var_key overlap);
/// `gather_ranges` replays the ranges it produces with zero further search.
pub fn find_ranges(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    samples: Option<&[usize]>,
) -> RangesBundle {
    let ploidy = reader.ploidy;
    let sample_cols: Vec<usize> = match samples {
        Some(s) => s.to_vec(),
        None => (0..reader.n_samples).collect(),
    };
    let n_samples = sample_cols.len();
    let n_regions = regions.len();
    let h = n_samples * ploidy;

    // Region-independent union; `overlap` builds one SearchTree per region.
    let dense = reader.dense_union();
    let dense_range: Vec<Range<usize>> = regions
        .iter()
        .map(|&(qs, qe)| dense.overlap(qs, qe))
        .collect();
    let region_starts: Vec<u32> = regions.iter().map(|&(qs, _)| qs).collect();

    let dense_snp_range: Vec<Range<usize>> = regions
        .iter()
        .map(|&(qs, qe)| reader.dense_snp_overlap(qs, qe))
        .collect();
    let dense_indel_range: Vec<Range<usize>> = regions
        .iter()
        .map(|&(qs, qe)| reader.dense_indel_overlap(qs, qe))
        .collect();

    let mut vk_snp_range: Vec<Range<usize>> = Vec::with_capacity(n_regions * h);
    let mut vk_indel_range: Vec<Range<usize>> = Vec::with_capacity(n_regions * h);
    for &(qs, qe) in regions {
        for &orig_s in &sample_cols {
            for p in 0..ploidy {
                let col = orig_s * ploidy + p;
                vk_snp_range.push(reader.vk_snp_overlap(col, qs, qe));
                vk_indel_range.push(reader.vk_indel_overlap(col, orig_s, p, qs, qe));
            }
        }
    }

    RangesBundle {
        n_regions,
        n_samples,
        ploidy,
        region_starts,
        dense_range,
        sample_cols,
        vk_snp_range,
        vk_indel_range,
        dense_snp_range,
        dense_indel_range,
    }
}

/// Tree-free gather: replay a `RangesBundle` into the same `BatchResult` that
/// `overlap_batch` produces. Contains NO `SearchTree::new` — the search
/// already happened in `find_ranges`. Mirrors `overlap_batch`'s inner loop
/// exactly, except the var_key channel is replayed from the precomputed
/// ranges (no per-column `SearchTree` rebuild, no per-element `carried` test
/// — `vk_slice`'s `carried` closure is `|_| true` for both channels, so only
/// the `q_start < v_end` left-overlap re-check remains) and the loop runs
/// over the *selected* sample slots.
pub fn gather_ranges(reader: &ContigReader, rb: &RangesBundle) -> BatchResult {
    let ploidy = rb.ploidy;
    let n_samples = rb.n_samples;
    let n_regions = rb.n_regions;
    let hpr = n_samples * ploidy; // haps per region

    let dense = reader.dense_union();

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut presence = PresenceBitWriter::new();

    for r in 0..n_regions {
        let qs = rb.region_starts[r];
        let dr = &rb.dense_range[r];
        let (ds, de) = (dr.start, dr.end);
        for si in 0..n_samples {
            let orig_s = rb.sample_cols[si];
            for p in 0..ploidy {
                let col = orig_s * ploidy + p;
                let row = r * hpr + si * ploidy + p;

                // --- var_key gather (no search) ---
                let merged = gather_vk(
                    reader,
                    rb.vk_snp_range[row].clone(),
                    rb.vk_indel_range[row].clone(),
                    qs,
                );
                vk.extend_from_slice(&merged);
                vk_off.push(vk.len());

                // --- dense presence bits (verbatim from overlap_batch) ---
                let nbits = de - ds;
                presence.push_hap(nbits, |k| {
                    let j = ds + k;
                    let (class, dcol) = dense.src[j];
                    let carried = reader
                        .dense_view(class)
                        .expect("dense src implies table")
                        .carried(col, dcol);
                    carried && dense.v_ends[j] > qs
                });
            }
        }
    }

    let (dense_present, dense_present_off) = presence.into_parts();

    BatchResult {
        n_regions,
        n_samples,
        ploidy,
        vk,
        vk_off,
        dense: dense.refs,
        dense_range: rb.dense_range.clone(),
        dense_present,
        dense_present_off,
    }
}

/// Borrowed per-query range slices for `gather_haps_readbound`, bundling the six
/// parallel slices + ploidy so the parallel-length contract is validated once at
/// construction instead of via five `assert_eq!`s at every call. Row layout is
/// unchanged: `dense_*_range` are per-query (len `n_q`); `vk_*_range` are
/// per-(query,ploid) (len `n_q * ploidy`, row `q*ploidy + p`).
pub struct HapRanges<'a> {
    pub region_starts: &'a [u32],
    pub orig_samples: &'a [usize],
    pub vk_snp_range: &'a [Range<usize>],
    pub vk_indel_range: &'a [Range<usize>],
    pub dense_snp_range: &'a [Range<usize>],
    pub dense_indel_range: &'a [Range<usize>],
    pub ploidy: usize,
}

impl<'a> HapRanges<'a> {
    /// Validate the parallel-slice length contract (panics on mismatch, matching
    /// the invariants `gather_haps_readbound` previously asserted inline).
    pub fn new(
        region_starts: &'a [u32],
        orig_samples: &'a [usize],
        vk_snp_range: &'a [Range<usize>],
        vk_indel_range: &'a [Range<usize>],
        dense_snp_range: &'a [Range<usize>],
        dense_indel_range: &'a [Range<usize>],
        ploidy: usize,
    ) -> Self {
        let n_q = region_starts.len();
        assert_eq!(orig_samples.len(), n_q, "orig_samples len must equal n_q");
        assert_eq!(
            dense_snp_range.len(),
            n_q,
            "dense_snp_range len must equal n_q"
        );
        assert_eq!(
            dense_indel_range.len(),
            n_q,
            "dense_indel_range len must equal n_q"
        );
        assert_eq!(
            vk_snp_range.len(),
            n_q * ploidy,
            "vk_snp_range len must equal n_q*ploidy"
        );
        assert_eq!(
            vk_indel_range.len(),
            n_q * ploidy,
            "vk_indel_range len must equal n_q*ploidy"
        );
        Self {
            region_starts,
            orig_samples,
            vk_snp_range,
            vk_indel_range,
            dense_snp_range,
            dense_indel_range,
            ploidy,
        }
    }
}

/// Flat per-query read-bound gather for gvl's arbitrary-(region,sample) reads.
/// Each of `n_q = region_starts.len()` queries is one (region, sample) pair
/// reconstructing `ploidy` haps. Range arrays are per-query (`dense_*_range`,
/// length n_q) or per-(query,ploid) (`vk_*_range`, length n_q*ploidy, row =
/// q*ploidy + p). Builds zero SearchTrees and never calls `dense_union()`.
/// Returns a `BatchResultSplit` with `n_samples = 1`, hap index `q*ploidy + p`.
#[allow(clippy::needless_range_loop)]
pub fn gather_haps_readbound(reader: &ContigReader, rb: &HapRanges<'_>) -> BatchResultSplit {
    let region_starts = rb.region_starts;
    let orig_samples = rb.orig_samples;
    let vk_snp_range = rb.vk_snp_range;
    let vk_indel_range = rb.vk_indel_range;
    let dense_snp_range = rb.dense_snp_range;
    let dense_indel_range = rb.dense_indel_range;
    let ploidy = rb.ploidy;
    let n_q = region_starts.len();

    let snp_positions = reader.vk_snp.positions();
    let snp_keys = as_bytes(&reader.vk_snp.keys);
    let indel_positions = reader.vk_indel.positions();
    let indel_keys = as_u32(&reader.vk_indel.keys);
    let d_snp = reader.dense_snp.as_ref();
    let d_indel = reader.dense_indel.as_ref();
    let d_snp_pos: &[u32] = d_snp.map(|d| d.positions()).unwrap_or(&[]);
    let d_indel_pos: &[u32] = d_indel.map(|d| d.positions()).unwrap_or(&[]);

    // Dense windows per query (uniform keys), decoded once.
    let mut dense_snp: Vec<KeyRef> = Vec::new();
    let mut dense_snp_range_out: Vec<Range<usize>> = Vec::with_capacity(n_q);
    let mut dense_indel: Vec<KeyRef> = Vec::new();
    let mut dense_indel_range_out: Vec<Range<usize>> = Vec::with_capacity(n_q);
    for q in 0..n_q {
        let (ss, se) = (dense_snp_range[q].start, dense_snp_range[q].end);
        let base = dense_snp.len();
        if let Some(d) = d_snp {
            let keys = as_bytes(&d.keys);
            // Slice once so the loop body indexes `d_snp_pos` via the
            // iterator (no per-iteration bounds check); `keys` stays
            // unsliced since it's 2-bit-packed and `unpack_snp_key_at`
            // needs the true absolute call index `j`, not a local one.
            for (k, &pos) in d_snp_pos[ss..se].iter().enumerate() {
                let j = ss + k;
                dense_snp.push(KeyRef {
                    position: pos,
                    key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, j)),
                });
            }
        }
        dense_snp_range_out.push(base..dense_snp.len());
        let (is_, ie_) = (dense_indel_range[q].start, dense_indel_range[q].end);
        let base = dense_indel.len();
        if let Some(d) = d_indel {
            let keys = as_u32(&d.keys);
            // Both `d_indel_pos` and `keys` are plain `&[u32]` in the same
            // absolute index space, so a paired slice iterator hoists the
            // bounds check to the slicing op instead of once per element.
            for (&pos, &key) in d_indel_pos[is_..ie_].iter().zip(&keys[is_..ie_]) {
                dense_indel.push(KeyRef { position: pos, key });
            }
        }
        dense_indel_range_out.push(base..dense_indel.len());
    }

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut snp_presence = PresenceBitWriter::new();
    let mut indel_presence = PresenceBitWriter::new();

    for q in 0..n_q {
        let qs = region_starts[q];
        let orig_s = orig_samples[q];
        let (ss, se) = (dense_snp_range[q].start, dense_snp_range[q].end);
        let (is_r, ie_r) = (dense_indel_range[q].start, dense_indel_range[q].end);
        // The dense-SNP position filter `qs < pos + 1` (i.e. `pos >= qs`) is
        // hap-independent, and dense positions are sorted ascending, so the
        // columns that pass form the suffix [c0_snp..se). Compute the threshold
        // ONCE per query (was re-tested per hap per column) so each hap's
        // presence gather becomes a contiguous block copy of that suffix.
        let c0_snp = match d_snp {
            Some(_) => ss + d_snp_pos[ss..se].partition_point(|&p| p < qs),
            None => se,
        };
        for p in 0..ploidy {
            let hap = orig_s * ploidy + p;
            let row = q * ploidy + p;

            // var_key gather.
            let (vs, ve) = (vk_snp_range[row].start, vk_snp_range[row].end);
            // Was `snp_positions.iter().enumerate().take(ve).skip(vs)`:
            // `Skip::next()` drains and discards `vs` items from the front
            // on first call, so every hap re-walked `snp_positions[0..vs]`
            // from scratch. Slicing to `[vs..ve]` up front makes the walk
            // start at `vs` directly (one bounds check for the slice op,
            // not O(vs) wasted iterator steps per hap); `j = vs + k`
            // recovers the absolute call index `unpack_snp_key_at` needs
            // against the (unsliced, 2-bit-packed) `snp_keys` buffer.
            // Capacity is pre-sized so the filter below never reallocs.
            let mut snp_run: Vec<KeyRef> = Vec::with_capacity(ve.saturating_sub(vs));
            for (k, &pos) in snp_positions[vs..ve].iter().enumerate() {
                let j = vs + k;
                if qs < pos + 1 {
                    snp_run.push(KeyRef {
                        position: pos,
                        key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                    });
                }
            }
            let (vis, vie) = (vk_indel_range[row].start, vk_indel_range[row].end);
            // Same slicing treatment: `indel_positions`/`indel_keys` are
            // both plain `&[u32]` in the same absolute index space, so a
            // paired slice iterator drops the per-element bounds checks
            // that `indel_positions[j]`/`indel_keys[j]` incurred.
            let mut indel_run: Vec<KeyRef> = Vec::with_capacity(vie.saturating_sub(vis));
            for (&pos, &key) in indel_positions[vis..vie].iter().zip(&indel_keys[vis..vie]) {
                let v_end = pos + 1 + rvk::deletion_len(key);
                if qs < v_end {
                    indel_run.push(KeyRef { position: pos, key });
                }
            }
            // Specialized 2-way merge, provably byte-identical to
            // `spine::merge_keys(vec![snp_run, indel_run])`: that generic
            // k-way merge picks `best = Some(0)` (snp_run) on the first
            // scan and only switches to run 1 (indel_run) when
            // `!(snp_run[h0].position <= indel_run[h1].position)`, i.e.
            // exactly the `<=` two-pointer comparison below (ties still
            // favor snp_run).
            //
            // Task 3 (gather_vk extraction) benched routing this hap-major
            // loop through the shared `gather_vk` helper (which rebuilds
            // `snp_run`/`indel_run` via `merge_keys` per call, like
            // `gather_ranges`/`gather_ranges_readbound` do). The readbound
            // test fixtures are too small (single-digit variant counts) for
            // a stable wall-clock signal, so the decision was made on
            // algorithmic grounds instead: unifying would reintroduce the
            // slicing/allocation costs this block was written to remove
            // (re-walking `snp_positions[0..vs]` per hap via `skip`, plus
            // three extra allocations per hap from `merge_keys`'s
            // `vec![...]`/`heads`/`out`). This tuned merge is retained.
            let (mut si, mut ii) = (0usize, 0usize);
            while si < snp_run.len() && ii < indel_run.len() {
                if snp_run[si].position <= indel_run[ii].position {
                    vk.push(snp_run[si]);
                    si += 1;
                } else {
                    vk.push(indel_run[ii]);
                    ii += 1;
                }
            }
            vk.extend_from_slice(&snp_run[si..]);
            vk.extend_from_slice(&indel_run[ii..]);
            vk_off.push(vk.len());

            // dense/snp presence over [ss..se). Columns [ss..c0_snp) fail the
            // position filter (bit stays 0); [c0_snp..se) all pass, so their
            // presence == this hap's genotype bits (hap-major, contiguous) — a
            // single block copy, byte-identical to the per-column test.
            let nbits = se - ss;
            snp_presence.push_hap_bulk(nbits, |bits, base| {
                if let Some(d) = d_snp
                    && c0_snp < se
                {
                    let gt = as_bytes(&d.genotypes);
                    bits::copy_bits(
                        bits,
                        base + (c0_snp - ss),
                        gt,
                        hap * d.n_dense_variants + c0_snp,
                        se - c0_snp,
                    );
                }
            });

            // dense/indel presence over [is_r..ie_r).
            let nbits = ie_r - is_r;
            indel_presence.push_hap(nbits, |k| {
                let j = is_r + k;
                match d_indel {
                    Some(d) => {
                        let keys = as_u32(&d.keys);
                        let v_end = d_indel_pos[j] + 1 + rvk::deletion_len(keys[j]);
                        d.carried(hap, j) && qs < v_end
                    }
                    None => false,
                }
            });
        }
    }

    let (dense_snp_present, dense_snp_present_off) = snp_presence.into_parts();
    let (dense_indel_present, dense_indel_present_off) = indel_presence.into_parts();

    BatchResultSplit {
        n_regions: n_q,
        n_samples: 1,
        ploidy,
        vk,
        vk_off,
        dense_snp,
        dense_snp_range: dense_snp_range_out,
        dense_snp_present,
        dense_snp_present_off,
        dense_indel,
        dense_indel_range: dense_indel_range_out,
        dense_indel_present,
        dense_indel_present_off,
    }
}

/// Fused search+gather: `find_ranges` then `gather_ranges` in one call. The
/// public/live-query analog of the split; byte-identical to `overlap_batch`
/// when `samples = None`, and the parity oracle for sample subsetting.
pub fn read_ranges(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    samples: Option<&[usize]>,
) -> BatchResult {
    gather_ranges(reader, &find_ranges(reader, regions, samples))
}

impl BatchResult {
    /// Decode the merged `var_key ⋈ dense` variants that `(sample s, ploid p)`
    /// carries in region `r` — position-sorted `(position, ilen, alt)`, identical
    /// to `overlap_sample(sample=s, region=r).per_hap[p]`. The M6c materialization
    /// primitive; here also the Task 5 cross-check oracle. Needs `reader` for the
    /// LUT (long-INS allele bytes).
    pub fn decode_hap(&self, reader: &ContigReader, r: usize, s: usize, p: usize) -> HapCalls {
        let h = (r * self.n_samples + s) * self.ploidy + p;
        let vk_slice = self.vk[self.vk_off[h]..self.vk_off[h + 1]].to_vec();

        let (ds, de) = (self.dense_range[r].start, self.dense_range[r].end);
        let bit0 = self.dense_present_off[h];
        let mut dn: Vec<KeyRef> = Vec::new();
        for (k, j) in (ds..de).enumerate() {
            if bits::get_bit(&self.dense_present, bit0 + k) {
                dn.push(self.dense[j]);
            }
        }

        let merged = spine::merge_keys(vec![vk_slice, dn]);
        let lut = reader.lut.as_ref();
        let mut hc = HapCalls::default();
        for kr in merged {
            let c = decode_keyref(kr, lut);
            hc.positions.push(c.position);
            hc.ilens.push(c.ilen);
            hc.alts.push(c.alt);
        }
        hc
    }
}

#[cfg(test)]
mod tests {
    use super::super::sidecar::mmap_file;
    use super::*;
    use tempfile::tempdir;

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn test_batch_result_offsets_are_consistent() {
        // A hand-built BatchResult: 1 region, 2 samples, ploidy 2 -> H = 4.
        // vk_off / dense_present_off must be non-decreasing, length H+1, and
        // bound the flat buffers.
        let br = BatchResult {
            n_regions: 1,
            n_samples: 2,
            ploidy: 2,
            vk: vec![KeyRef {
                position: 10,
                key: 1 << 25,
            }],
            vk_off: vec![0, 1, 1, 1, 1],
            dense: vec![],
            dense_range: vec![0..0],
            dense_present: vec![],
            dense_present_off: vec![0, 0, 0, 0, 0],
        };
        let h = br.n_regions * br.n_samples * br.ploidy;
        assert_eq!(br.vk_off.len(), h + 1);
        assert_eq!(br.dense_present_off.len(), h + 1);
        assert_eq!(*br.vk_off.last().unwrap(), br.vk.len());
        assert!(br.vk_off.windows(2).all(|w| w[0] <= w[1]));
        assert!(br.dense_present_off.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_mmap_u32_roundtrip() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("positions.bin");
        let vals: Vec<u32> = vec![10, 20, 30, 40];
        std::fs::write(&p, bytemuck::cast_slice(&vals)).unwrap();
        let m = mmap_file(&p).unwrap();
        assert_eq!(as_u32(&m), &vals[..]);
    }

    #[test]
    fn test_mmap_missing_and_empty_are_none() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope.bin");
        assert!(mmap_file(&missing).unwrap().is_none());

        let empty = dir.path().join("empty.bin");
        std::fs::File::create(&empty).unwrap();
        assert!(mmap_file(&empty).unwrap().is_none());

        assert_eq!(as_u32(&None), &[] as &[u32]);
        assert_eq!(as_bytes(&None), &[] as &[u8]);
    }
}

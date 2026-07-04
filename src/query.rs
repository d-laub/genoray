//! Disk-facing `(range, sample)` query for a finished SVAR2 contig (M5 part 2b).
//! Wires the pure `search.rs` overlap core to the on-disk sidecars: for a contig,
//! region `[q_start, q_end)`, and sample, return that sample's overlapping
//! variants per haplotype. `search.rs` is untouched.

use std::fs::File;
use std::io::ErrorKind;
use std::path::Path;

use memmap2::Mmap;
use ndarray::{Array1, Array2};

use crate::bits;
use crate::layout::{self, ContigPaths};
use crate::nrvk::LongAlleleReader;
use crate::rvk::{self, DecodedKey};
use crate::search::{SearchTree, overlap_range};
use crate::spine::{self, KeyRef};

/// mmap a file into memory, returning `None` for a missing or zero-length file
/// (memmap2 rejects empty maps; an absent sidecar means an empty sub-stream).
fn mmap_file(path: &Path) -> std::io::Result<Option<Mmap>> {
    match File::open(path) {
        Ok(f) => {
            let len = f.metadata()?.len();
            if len == 0 {
                Ok(None)
            } else {
                // SAFETY: the sidecar is a finished, read-only artifact; we never
                // mutate the file while it is mapped.
                Ok(Some(unsafe { Mmap::map(&f)? }))
            }
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// View a raw little-endian `u32` sidecar (`positions.bin`, indel `alleles.bin`)
/// as a `&[u32]`. mmap pages are page-aligned, so `bytemuck`'s alignment check
/// always passes; `None` (missing/empty) yields an empty slice.
fn as_u32(m: &Option<Mmap>) -> &[u32] {
    match m {
        Some(mm) => bytemuck::cast_slice(&mm[..]),
        None => &[],
    }
}

/// Raw bytes of a mmap'd sidecar (packed SNP `alleles.bin`, `genotypes.bin`),
/// or an empty slice when missing/empty.
fn as_bytes(m: &Option<Mmap>) -> &[u8] {
    match m {
        Some(mm) => &mm[..],
        None => &[],
    }
}

/// One overlapping variant call, decoded. The per-element intermediate before
/// the k-way merge (the columnar `HapCalls` is assembled from these).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub position: u32,
    /// Length delta (ALT − REF): `0` SNP, `> 0` insertion, `< 0` deletion.
    pub ilen: i32,
    /// Decoded ALT bytes for SNP/INS; empty for a pure DEL (the anchor base is
    /// not stored in the key — recovered from the reference downstream).
    pub alt: Vec<u8>,
}

/// Decode one uniform `KeyRef` into a `Call`, resolving long-INS lookups through
/// the LUT. The single place a query result touches alleles: SNP/INS decode
/// inline (`Inline`), a pure DEL yields an empty ALT (`PureDel`), a long INS
/// resolves via the bank (`Lookup`). `ilen = alt.len() - 1` for the inline lanes
/// matches M5's `decode_snp_2bit`/`decode_indel_hit` contract exactly.
fn decode_keyref(kr: KeyRef, lut: Option<&LongAlleleReader>) -> Call {
    match rvk::decode_key(kr.key) {
        DecodedKey::Inline { alt } => Call {
            position: kr.position,
            ilen: alt.len() as i32 - 1,
            alt,
        },
        DecodedKey::PureDel { ilen } => Call {
            position: kr.position,
            ilen,
            alt: Vec::new(),
        },
        DecodedKey::Lookup { row } => {
            let alt = lut
                .expect("indel lookup key requires a long-allele LUT")
                .get_allele(row);
            Call {
                position: kr.position,
                ilen: alt.len() as i32 - 1,
                alt,
            }
        }
    }
}

/// A var_key sub-stream (snp or indel): mmap'd `positions.bin` + `alleles.bin`
/// with the CSR `offsets.npy` giving per-`(sample, ploid)` column bounds.
struct SubStreamView {
    positions: Option<Mmap>, // raw u32 LE, one per call
    keys: Option<Mmap>,      // packed 2-bit codes (snp) or u32 LE keys (indel)
    offsets: Vec<u64>,       // CSR prefix-sum, len == columns + 1
}

impl SubStreamView {
    fn positions(&self) -> &[u32] {
        as_u32(&self.positions)
    }
    /// Half-open `[start, end)` call range for flat column `c`.
    fn column(&self, c: usize) -> (usize, usize) {
        (self.offsets[c] as usize, self.offsets[c + 1] as usize)
    }
}

/// A dense class table (snp or indel): shared per-contig `positions.bin` +
/// `alleles.bin` + hap-major `genotypes.bin` 1-bit matrix.
struct DenseView {
    positions: Option<Mmap>,
    keys: Option<Mmap>,
    genotypes: Option<Mmap>,
    n_dense_variants: usize,
}

impl DenseView {
    fn positions(&self) -> &[u32] {
        as_u32(&self.positions)
    }
    /// Whether haplotype `hap` carries dense variant `col` (hap-major bit
    /// `hap * n_dense_variants + col`).
    fn carried(&self, hap: usize, col: usize) -> bool {
        bits::get_bit(as_bytes(&self.genotypes), hap * self.n_dense_variants + col)
    }
}

/// Load a CSR `offsets.npy` (len `columns + 1`); a missing file means an empty
/// stream — return an all-zero prefix-sum so every column is empty.
fn load_offsets(path: &Path, columns: usize) -> Vec<u64> {
    if path.exists() {
        let a: Array1<u64> = ndarray_npy::read_npy(path).expect("read offsets.npy");
        a.to_vec()
    } else {
        vec![0u64; columns + 1]
    }
}

/// Load `max_del.npy` (`u32`, shape `(n_samples, ploidy)`); a missing file
/// (pure-SNP contig, or predating the post-pass) defaults to all-zero.
fn load_max_del(path: &Path, n_samples: usize, ploidy: usize) -> Array2<u32> {
    if path.exists() {
        ndarray_npy::read_npy(path).expect("read max_del.npy")
    } else {
        Array2::zeros((n_samples, ploidy))
    }
}

/// Load `dense/max_del.npy` (`u32`, shape `(1,)`); missing defaults to `0`.
fn load_dense_max_del(path: &Path) -> u32 {
    if path.exists() {
        let a: Array1<u32> = ndarray_npy::read_npy(path).expect("read dense/max_del.npy");
        a.into_iter().next().unwrap_or(0)
    } else {
        0
    }
}

/// Open a dense class table, or `None` when the class has no variants (absent
/// dir / empty `positions.bin`).
fn open_dense(dir: &Path) -> std::io::Result<Option<DenseView>> {
    let positions = mmap_file(&layout::positions(dir))?;
    let n_dense_variants = as_u32(&positions).len();
    if n_dense_variants == 0 {
        return Ok(None);
    }
    Ok(Some(DenseView {
        keys: mmap_file(&layout::alleles(dir))?,
        genotypes: mmap_file(&layout::genotypes(dir))?,
        positions,
        n_dense_variants,
    }))
}

/// Opens a finished SVAR2 contig directory and holds its sidecars mmap'd for the
/// lifetime of queries against it.
pub struct ContigReader {
    ploidy: usize,
    n_samples: usize,
    vk_snp: SubStreamView,
    vk_indel: SubStreamView,
    dense_snp: Option<DenseView>,
    dense_indel: Option<DenseView>,
    /// `(n_samples, ploidy)` per-column max deletion length for var_key/indel.
    vk_indel_max_del: Array2<u32>,
    /// Per-contig max deletion length over the shared dense/indel table.
    dense_indel_max_del: u32,
    /// Long-allele bank reader; present iff the shared indel LUT exists.
    lut: Option<LongAlleleReader>,
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
            offsets: load_offsets(&layout::offsets(&vk_snp_dir), columns),
        };
        let vk_indel = SubStreamView {
            positions: mmap_file(&layout::positions(&vk_indel_dir))?,
            keys: mmap_file(&layout::alleles(&vk_indel_dir))?,
            offsets: load_offsets(&layout::offsets(&vk_indel_dir), columns),
        };

        let dense_snp = open_dense(&paths.dense_snp_dir())?;
        let dense_indel = open_dense(&paths.dense_indel_dir())?;

        let vk_indel_max_del = load_max_del(&layout::max_del(&contig_dir), n_samples, ploidy);
        let dense_indel_max_del = load_dense_max_del(&layout::dense_max_del(&contig_dir));

        let lut = if paths.long_alleles_bin().exists() {
            Some(LongAlleleReader::new(base_out_dir, chrom))
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
}

/// The per-contig dense table unioned across `snp`+`indel`, position-sorted,
/// carrying uniform keys plus the `(is_indel, col)` needed to test carriage.
/// Region-independent — built once per query; `overlap` derives each region's
/// index range from it. `src[i] = (is_indel, col)` addresses the original dense
/// class table for the genotype-bit test.
struct DenseUnion {
    refs: Vec<KeyRef>,
    src: Vec<(bool, usize)>,
    positions: Vec<u32>,
    v_ends: Vec<u32>,
    max_del: u32,
}

impl DenseUnion {
    /// `[s, e)` into `refs`/`src` for `[q_start, q_end)`, deletion-aware. Builds a
    /// fresh search tree over `positions` (cheap; one per region in a batch).
    fn overlap(&self, q_start: u32, q_end: u32) -> (usize, usize) {
        if self.refs.is_empty() {
            return (0, 0);
        }
        let tree = SearchTree::new(&self.positions);
        overlap_range(&tree, &self.v_ends, self.max_del, q_start, q_end)
    }
}

impl ContigReader {
    /// `var_key` channel for one flat hap-column over one region: `snp`+`indel`
    /// unioned, uniform keys (SNP re-expanded via `snp_code_to_key`), sorted.
    fn vk_slice(
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
            let (o0, o1) = self.vk_snp.column(col);
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
            let (o0, o1) = self.vk_indel.column(col);
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

    /// Build the region-independent dense `snp`+`indel` union (see `DenseUnion`).
    /// SNP codes re-expand to uniform keys; the max_region_length bound is the
    /// per-contig dense/indel max (SNP contributes 0).
    fn dense_union(&self) -> DenseUnion {
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

    /// The dense variants in `union[s..e]` that `hap` carries and that TRULY
    /// overlap `[q_start, q_end)` (exact left-overlap `q_start < v_end`; the
    /// right half is guaranteed by `union.overlap`'s window — see
    /// `spine::gather_keys`'s doc comment for why the per-element check is
    /// still needed within a carried window).
    fn dense_carried(
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
            let (is_indel, col) = union.src[j];
            let carried = if is_indel {
                self.dense_indel
                    .as_ref()
                    .expect("indel src implies table")
                    .carried(hap, col)
            } else {
                self.dense_snp
                    .as_ref()
                    .expect("snp src implies table")
                    .carried(hap, col)
            };
            if carried {
                out.push(union.refs[j]);
            }
        }
        out
    }
}

/// Per-haplotype overlapping calls, position-sorted. Struct-of-arrays for a
/// numpy-friendly M6 hand-off.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HapCalls {
    pub positions: Vec<u32>,
    pub ilens: Vec<i32>,
    pub alts: Vec<Vec<u8>>,
}

/// Result of an `overlap_sample` query: one `HapCalls` per haplotype
/// (`per_hap.len() == ploidy`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryResult {
    pub per_hap: Vec<HapCalls>,
}

/// Return every variant that `sample` carries overlapping `[q_start, q_end)`, per
/// haplotype, position-sorted, unioning the var_key and dense sub-streams. M5's
/// public contract, re-expressed on the M6.1 spine: gather uniform KeyRefs, do
/// the final `var_key ⋈ dense` 2-way merge, then decode.
pub fn overlap_sample(
    reader: &ContigReader,
    sample: usize,
    q_start: u32,
    q_end: u32,
) -> QueryResult {
    let ploidy = reader.ploidy;
    let lut = reader.lut.as_ref();
    let dense = reader.dense_union();
    let (ds, de) = dense.overlap(q_start, q_end);

    let mut per_hap = Vec::with_capacity(ploidy);
    for p in 0..ploidy {
        let col = sample * ploidy + p; // flat column
        let hap = col; // sample-major hap index == flat column
        let vk = reader.vk_slice(col, sample, p, q_start, q_end);
        let dn = reader.dense_carried(&dense, hap, ds, de, q_start);
        let merged = spine::merge_keys(vec![vk, dn]);

        let mut hc = HapCalls::default();
        for kr in merged {
            let c = decode_keyref(kr, lut);
            hc.positions.push(c.position);
            hc.ilens.push(c.ilen);
            hc.alts.push(c.alt);
        }
        per_hap.push(hc);
    }
    QueryResult { per_hap }
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
    pub dense_range: Vec<(usize, usize)>,
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
    pub dense_snp_range: Vec<(usize, usize)>,
    /// Per-hap presence bitmask over that region's `dense_snp[s..e]`, LSB-first;
    /// `dense_snp_present_off` (len H+1) holds BIT offsets.
    pub dense_snp_present: Vec<u8>,
    pub dense_snp_present_off: Vec<usize>,
    /// Per-region `dense/indel` windows (uniform u32 keys), concatenated.
    pub dense_indel: Vec<KeyRef>,
    pub dense_indel_range: Vec<(usize, usize)>,
    pub dense_indel_present: Vec<u8>,
    pub dense_indel_present_off: Vec<usize>,
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
    let ranges: Vec<(usize, usize)> = regions
        .iter()
        .map(|&(qs, qe)| dense.overlap(qs, qe))
        .collect();

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut dense_present: Vec<u8> = Vec::new();
    let mut dense_present_off: Vec<usize> = vec![0];

    for (r, &(qs, qe)) in regions.iter().enumerate() {
        let (ds, de) = ranges[r];
        for s in 0..n_samples {
            for p in 0..ploidy {
                let col = s * ploidy + p;
                let hap = col; // sample-major hap index == flat column

                // var_key channel slice for (region r, sample s, ploid p).
                let slice = reader.vk_slice(col, s, p, qs, qe);
                vk.extend_from_slice(&slice);
                vk_off.push(vk.len());

                // dense presence bits over dense[ds..de].
                let nbits = de - ds;
                let bit_base = *dense_present_off.last().unwrap();
                let need_bytes = (bit_base + nbits).div_ceil(8);
                if dense_present.len() < need_bytes {
                    dense_present.resize(need_bytes, 0);
                }
                for (k, j) in (ds..de).enumerate() {
                    let (is_indel, dcol) = dense.src[j];
                    let carried = if is_indel {
                        reader
                            .dense_indel
                            .as_ref()
                            .expect("indel src implies table")
                            .carried(hap, dcol)
                    } else {
                        reader
                            .dense_snp
                            .as_ref()
                            .expect("snp src implies table")
                            .carried(hap, dcol)
                    };
                    if carried && dense.v_ends[j] > qs {
                        bits::set_bit(&mut dense_present, bit_base + k);
                    }
                }
                dense_present_off.push(bit_base + nbits);
            }
        }
    }

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
    pub dense_range: Vec<(usize, usize)>,
    /// Selected slot -> original sample index.
    pub sample_cols: Vec<usize>,
    /// Absolute `[start, end)` into `vk_snp`'s packed positions/keys, per
    /// `(region, selected sample, ploid)`.
    pub vk_snp_range: Vec<(usize, usize)>,
    /// Absolute `[start, end)` into `vk_indel`'s packed positions/keys, per
    /// `(region, selected sample, ploid)`.
    pub vk_indel_range: Vec<(usize, usize)>,
    /// `[s, e)` into `dense/snp`'s on-disk positions/keys, per region (dense is
    /// cohort-shared, so one window per region, not per hap). Read-bound path.
    pub dense_snp_range: Vec<(usize, usize)>,
    /// `[s, e)` into `dense/indel`'s on-disk positions/keys, per region.
    pub dense_indel_range: Vec<(usize, usize)>,
}

impl ContigReader {
    /// Absolute `[start, end)` into `vk_snp`'s packed positions/keys for
    /// `(col, region)` — the SNP channel's search half (`max_region_length =
    /// 0`, since a SNP always spans exactly one base). No gather.
    fn vk_snp_overlap(&self, col: usize, q_start: u32, q_end: u32) -> (usize, usize) {
        let (o0, o1) = self.vk_snp.column(col);
        let positions = &self.vk_snp.positions()[o0..o1];
        if positions.is_empty() {
            return (o0, o0);
        }
        let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        let tree = SearchTree::new(positions);
        let (s, e) = overlap_range(&tree, &v_ends, 0, q_start, q_end);
        (o0 + s, o0 + e)
    }

    /// Absolute `[start, end)` into `vk_indel`'s packed positions/keys for
    /// `(col, region)` — the indel channel's search half (per-column
    /// `max_del` bound). No gather.
    fn vk_indel_overlap(
        &self,
        col: usize,
        sample: usize,
        p: usize,
        q_start: u32,
        q_end: u32,
    ) -> (usize, usize) {
        let (o0, o1) = self.vk_indel.column(col);
        let positions = &self.vk_indel.positions()[o0..o1];
        if positions.is_empty() {
            return (o0, o0);
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
        (o0 + s, o0 + e)
    }

    /// Absolute `[s, e)` into `dense/snp`'s positions/keys for one region.
    /// SNP v_end = pos + 1 (max_region_length = 0). `(0, 0)` if no snp table.
    fn dense_snp_overlap(&self, q_start: u32, q_end: u32) -> (usize, usize) {
        let d = match &self.dense_snp {
            Some(d) => d,
            None => return (0, 0),
        };
        let positions = d.positions();
        if positions.is_empty() {
            return (0, 0);
        }
        let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        let tree = SearchTree::new(positions);
        overlap_range(&tree, &v_ends, 0, q_start, q_end)
    }

    /// Absolute `[s, e)` into `dense/indel`'s positions/keys for one region.
    /// Indel v_end = pos + 1 + deletion_len(key); per-contig dense max_del bound.
    fn dense_indel_overlap(&self, q_start: u32, q_end: u32) -> (usize, usize) {
        let d = match &self.dense_indel {
            Some(d) => d,
            None => return (0, 0),
        };
        let positions = d.positions();
        if positions.is_empty() {
            return (0, 0);
        }
        let keys = as_u32(&d.keys);
        debug_assert_eq!(positions.len(), keys.len());
        let v_ends: Vec<u32> = positions
            .iter()
            .zip(keys.iter())
            .map(|(&pos, &key)| pos + 1 + rvk::deletion_len(key))
            .collect();
        let tree = SearchTree::new(positions);
        overlap_range(&tree, &v_ends, self.dense_indel_max_del, q_start, q_end)
    }
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
    let dense_range: Vec<(usize, usize)> = regions
        .iter()
        .map(|&(qs, qe)| dense.overlap(qs, qe))
        .collect();
    let region_starts: Vec<u32> = regions.iter().map(|&(qs, _)| qs).collect();

    let dense_snp_range: Vec<(usize, usize)> = regions
        .iter()
        .map(|&(qs, qe)| reader.dense_snp_overlap(qs, qe))
        .collect();
    let dense_indel_range: Vec<(usize, usize)> = regions
        .iter()
        .map(|&(qs, qe)| reader.dense_indel_overlap(qs, qe))
        .collect();

    let mut vk_snp_range = Vec::with_capacity(n_regions * h);
    let mut vk_indel_range = Vec::with_capacity(n_regions * h);
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
    let mut dense_present: Vec<u8> = Vec::new();
    let mut dense_present_off: Vec<usize> = vec![0];

    let snp_positions = reader.vk_snp.positions();
    let snp_keys = as_bytes(&reader.vk_snp.keys);
    let indel_positions = reader.vk_indel.positions();
    let indel_keys = as_u32(&reader.vk_indel.keys);

    for r in 0..n_regions {
        let qs = rb.region_starts[r];
        let (ds, de) = rb.dense_range[r];
        for si in 0..n_samples {
            let orig_s = rb.sample_cols[si];
            for p in 0..ploidy {
                let col = orig_s * ploidy + p;
                let hap = col; // sample-major hap index == flat column
                let row = r * hpr + si * ploidy + p;

                // --- var_key gather (no search) ---
                let (ss, se) = rb.vk_snp_range[row];
                let mut snp_run: Vec<KeyRef> = Vec::new();
                for (j, &pos) in snp_positions.iter().enumerate().take(se).skip(ss) {
                    if qs < pos + 1 {
                        // snp v_end = pos + 1
                        snp_run.push(KeyRef {
                            position: pos,
                            key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                        });
                    }
                }

                let (is_, ie_) = rb.vk_indel_range[row];
                let mut indel_run: Vec<KeyRef> = Vec::new();
                for j in is_..ie_ {
                    let pos = indel_positions[j];
                    let v_end = pos + 1 + rvk::deletion_len(indel_keys[j]);
                    if qs < v_end {
                        indel_run.push(KeyRef {
                            position: pos,
                            key: indel_keys[j],
                        });
                    }
                }

                let merged = spine::merge_keys(vec![snp_run, indel_run]);
                vk.extend_from_slice(&merged);
                vk_off.push(vk.len());

                // --- dense presence bits (verbatim from overlap_batch) ---
                let nbits = de - ds;
                let bit_base = *dense_present_off.last().unwrap();
                let need_bytes = (bit_base + nbits).div_ceil(8);
                if dense_present.len() < need_bytes {
                    dense_present.resize(need_bytes, 0);
                }
                for (k, j) in (ds..de).enumerate() {
                    let (is_indel, dcol) = dense.src[j];
                    let carried = if is_indel {
                        reader
                            .dense_indel
                            .as_ref()
                            .expect("indel src implies table")
                            .carried(hap, dcol)
                    } else {
                        reader
                            .dense_snp
                            .as_ref()
                            .expect("snp src implies table")
                            .carried(hap, dcol)
                    };
                    if carried && dense.v_ends[j] > qs {
                        bits::set_bit(&mut dense_present, bit_base + k);
                    }
                }
                dense_present_off.push(bit_base + nbits);
            }
        }
    }

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

/// Tree-free, union-free gather: replay a `RangesBundle` into a split-dense
/// `BatchResultSplit`. Builds NO `SearchTree` and never calls `dense_union()` —
/// each region's dense windows come from the per-class `dense_snp_range` /
/// `dense_indel_range` computed in `find_ranges`. The var_key channel is
/// identical to `gather_ranges`; only the dense side is split per class.
#[allow(clippy::needless_range_loop)]
pub fn gather_ranges_readbound(reader: &ContigReader, rb: &RangesBundle) -> BatchResultSplit {
    let ploidy = rb.ploidy;
    let n_samples = rb.n_samples;
    let n_regions = rb.n_regions;
    let hpr = n_samples * ploidy;

    let snp_positions = reader.vk_snp.positions();
    let snp_keys = as_bytes(&reader.vk_snp.keys);
    let indel_positions = reader.vk_indel.positions();
    let indel_keys = as_u32(&reader.vk_indel.keys);

    // Dense class tables (may be absent).
    let d_snp = reader.dense_snp.as_ref();
    let d_indel = reader.dense_indel.as_ref();
    let d_snp_pos: &[u32] = d_snp.map(|d| d.positions()).unwrap_or(&[]);
    let d_indel_pos: &[u32] = d_indel.map(|d| d.positions()).unwrap_or(&[]);

    // --- dense channel windows (per region), decoded to uniform keys once ---
    let mut dense_snp: Vec<KeyRef> = Vec::new();
    let mut dense_snp_range: Vec<(usize, usize)> = Vec::with_capacity(n_regions);
    let mut dense_indel: Vec<KeyRef> = Vec::new();
    let mut dense_indel_range: Vec<(usize, usize)> = Vec::with_capacity(n_regions);
    for r in 0..n_regions {
        let (ss, se) = rb.dense_snp_range[r];
        let base = dense_snp.len();
        if let Some(d) = d_snp {
            let keys = as_bytes(&d.keys);
            for j in ss..se {
                dense_snp.push(KeyRef {
                    position: d_snp_pos[j],
                    key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, j)),
                });
            }
        }
        dense_snp_range.push((base, dense_snp.len()));

        let (is_, ie_) = rb.dense_indel_range[r];
        let base = dense_indel.len();
        if let Some(d) = d_indel {
            let keys = as_u32(&d.keys);
            for j in is_..ie_ {
                dense_indel.push(KeyRef {
                    position: d_indel_pos[j],
                    key: keys[j],
                });
            }
        }
        dense_indel_range.push((base, dense_indel.len()));
    }

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut dense_snp_present: Vec<u8> = Vec::new();
    let mut dense_snp_present_off: Vec<usize> = vec![0];
    let mut dense_indel_present: Vec<u8> = Vec::new();
    let mut dense_indel_present_off: Vec<usize> = vec![0];

    for r in 0..n_regions {
        let qs = rb.region_starts[r];
        let (ss, se) = rb.dense_snp_range[r];
        let (is_r, ie_r) = rb.dense_indel_range[r];
        for si in 0..n_samples {
            let orig_s = rb.sample_cols[si];
            for p in 0..ploidy {
                let col = orig_s * ploidy + p;
                let hap = col;
                let row = r * hpr + si * ploidy + p;

                // --- var_key gather (identical to gather_ranges) ---
                let (vs, ve) = rb.vk_snp_range[row];
                let mut snp_run: Vec<KeyRef> = Vec::new();
                for (j, &pos) in snp_positions.iter().enumerate().take(ve).skip(vs) {
                    if qs < pos + 1 {
                        snp_run.push(KeyRef {
                            position: pos,
                            key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                        });
                    }
                }
                let (vis, vie) = rb.vk_indel_range[row];
                let mut indel_run: Vec<KeyRef> = Vec::new();
                for j in vis..vie {
                    let pos = indel_positions[j];
                    let v_end = pos + 1 + rvk::deletion_len(indel_keys[j]);
                    if qs < v_end {
                        indel_run.push(KeyRef {
                            position: pos,
                            key: indel_keys[j],
                        });
                    }
                }
                let merged = spine::merge_keys(vec![snp_run, indel_run]);
                vk.extend_from_slice(&merged);
                vk_off.push(vk.len());

                // --- dense/snp presence bits over [ss..se) ---
                let nbits = se - ss;
                let bit_base = *dense_snp_present_off.last().unwrap();
                let need = (bit_base + nbits).div_ceil(8);
                if dense_snp_present.len() < need {
                    dense_snp_present.resize(need, 0);
                }
                if let Some(d) = d_snp {
                    for (k, j) in (ss..se).enumerate() {
                        // snp v_end = pos + 1; left-overlap re-check qs < v_end.
                        if d.carried(hap, j) && qs < d_snp_pos[j] + 1 {
                            bits::set_bit(&mut dense_snp_present, bit_base + k);
                        }
                    }
                }
                dense_snp_present_off.push(bit_base + nbits);

                // --- dense/indel presence bits over [is_r..ie_r) ---
                let nbits = ie_r - is_r;
                let bit_base = *dense_indel_present_off.last().unwrap();
                let need = (bit_base + nbits).div_ceil(8);
                if dense_indel_present.len() < need {
                    dense_indel_present.resize(need, 0);
                }
                if let Some(d) = d_indel {
                    let keys = as_u32(&d.keys);
                    for (k, j) in (is_r..ie_r).enumerate() {
                        let v_end = d_indel_pos[j] + 1 + rvk::deletion_len(keys[j]);
                        if d.carried(hap, j) && qs < v_end {
                            bits::set_bit(&mut dense_indel_present, bit_base + k);
                        }
                    }
                }
                dense_indel_present_off.push(bit_base + nbits);
            }
        }
    }

    BatchResultSplit {
        n_regions,
        n_samples,
        ploidy,
        vk,
        vk_off,
        dense_snp,
        dense_snp_range,
        dense_snp_present,
        dense_snp_present_off,
        dense_indel,
        dense_indel_range,
        dense_indel_present,
        dense_indel_present_off,
    }
}

/// Flat per-query read-bound gather for gvl's arbitrary-(region,sample) reads.
/// Each of `n_q = region_starts.len()` queries is one (region, sample) pair
/// reconstructing `ploidy` haps. Range arrays are per-query (`dense_*_range`,
/// length n_q) or per-(query,ploid) (`vk_*_range`, length n_q*ploidy, row =
/// q*ploidy + p). Builds zero SearchTrees and never calls `dense_union()`.
/// Returns a `BatchResultSplit` with `n_samples = 1`, hap index `q*ploidy + p`.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn gather_haps_readbound(
    reader: &ContigReader,
    region_starts: &[u32],
    orig_samples: &[usize],
    vk_snp_range: &[(usize, usize)],
    vk_indel_range: &[(usize, usize)],
    dense_snp_range: &[(usize, usize)],
    dense_indel_range: &[(usize, usize)],
    ploidy: usize,
) -> BatchResultSplit {
    let n_q = region_starts.len();
    assert_eq!(orig_samples.len(), n_q);
    assert_eq!(dense_snp_range.len(), n_q);
    assert_eq!(dense_indel_range.len(), n_q);
    assert_eq!(vk_snp_range.len(), n_q * ploidy);
    assert_eq!(vk_indel_range.len(), n_q * ploidy);

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
    let mut dense_snp_range_out: Vec<(usize, usize)> = Vec::with_capacity(n_q);
    let mut dense_indel: Vec<KeyRef> = Vec::new();
    let mut dense_indel_range_out: Vec<(usize, usize)> = Vec::with_capacity(n_q);
    for q in 0..n_q {
        let (ss, se) = dense_snp_range[q];
        let base = dense_snp.len();
        if let Some(d) = d_snp {
            let keys = as_bytes(&d.keys);
            for j in ss..se {
                dense_snp.push(KeyRef {
                    position: d_snp_pos[j],
                    key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, j)),
                });
            }
        }
        dense_snp_range_out.push((base, dense_snp.len()));
        let (is_, ie_) = dense_indel_range[q];
        let base = dense_indel.len();
        if let Some(d) = d_indel {
            let keys = as_u32(&d.keys);
            for j in is_..ie_ {
                dense_indel.push(KeyRef {
                    position: d_indel_pos[j],
                    key: keys[j],
                });
            }
        }
        dense_indel_range_out.push((base, dense_indel.len()));
    }

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut dense_snp_present: Vec<u8> = Vec::new();
    let mut dense_snp_present_off: Vec<usize> = vec![0];
    let mut dense_indel_present: Vec<u8> = Vec::new();
    let mut dense_indel_present_off: Vec<usize> = vec![0];

    for q in 0..n_q {
        let qs = region_starts[q];
        let orig_s = orig_samples[q];
        let (ss, se) = dense_snp_range[q];
        let (is_r, ie_r) = dense_indel_range[q];
        for p in 0..ploidy {
            let hap = orig_s * ploidy + p;
            let row = q * ploidy + p;

            // var_key gather.
            let (vs, ve) = vk_snp_range[row];
            let mut snp_run: Vec<KeyRef> = Vec::new();
            for (j, &pos) in snp_positions.iter().enumerate().take(ve).skip(vs) {
                if qs < pos + 1 {
                    snp_run.push(KeyRef {
                        position: pos,
                        key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                    });
                }
            }
            let (vis, vie) = vk_indel_range[row];
            let mut indel_run: Vec<KeyRef> = Vec::new();
            for j in vis..vie {
                let pos = indel_positions[j];
                let v_end = pos + 1 + rvk::deletion_len(indel_keys[j]);
                if qs < v_end {
                    indel_run.push(KeyRef {
                        position: pos,
                        key: indel_keys[j],
                    });
                }
            }
            vk.extend_from_slice(&spine::merge_keys(vec![snp_run, indel_run]));
            vk_off.push(vk.len());

            // dense/snp presence over [ss..se).
            let nbits = se - ss;
            let bit_base = *dense_snp_present_off.last().unwrap();
            let need = (bit_base + nbits).div_ceil(8);
            if dense_snp_present.len() < need {
                dense_snp_present.resize(need, 0);
            }
            if let Some(d) = d_snp {
                for (k, j) in (ss..se).enumerate() {
                    if d.carried(hap, j) && qs < d_snp_pos[j] + 1 {
                        bits::set_bit(&mut dense_snp_present, bit_base + k);
                    }
                }
            }
            dense_snp_present_off.push(bit_base + nbits);

            // dense/indel presence over [is_r..ie_r).
            let nbits = ie_r - is_r;
            let bit_base = *dense_indel_present_off.last().unwrap();
            let need = (bit_base + nbits).div_ceil(8);
            if dense_indel_present.len() < need {
                dense_indel_present.resize(need, 0);
            }
            if let Some(d) = d_indel {
                let keys = as_u32(&d.keys);
                for (k, j) in (is_r..ie_r).enumerate() {
                    let v_end = d_indel_pos[j] + 1 + rvk::deletion_len(keys[j]);
                    if d.carried(hap, j) && qs < v_end {
                        bits::set_bit(&mut dense_indel_present, bit_base + k);
                    }
                }
            }
            dense_indel_present_off.push(bit_base + nbits);
        }
    }

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

        let (ds, de) = self.dense_range[r];
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
    use super::*;
    use tempfile::tempdir;

    #[test]
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
            dense_range: vec![(0, 0)],
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

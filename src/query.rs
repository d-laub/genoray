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

    /// The dense variants in `union[s..e]` that `hap` carries, as uniform KeyRefs.
    fn dense_carried(&self, union: &DenseUnion, hap: usize, s: usize, e: usize) -> Vec<KeyRef> {
        let mut out = Vec::new();
        for j in s..e {
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
        let dn = reader.dense_carried(&dense, hap, ds, de);
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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

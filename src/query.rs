//! Disk-facing `(range, sample)` query for a finished SVAR2 contig (M5 part 2b).
//! Wires the pure `search.rs` overlap core to the on-disk sidecars: for a contig,
//! region `[q_start, q_end)`, and sample, return that sample's overlapping
//! variants per haplotype. `search.rs` is untouched.

// The query internals (loaders, `gather_run`, `kway_merge`, the reader's private
// views) are exercised only by tests until `overlap_sample` ties them together in
// Task 6. This keeps `cargo clippy -D warnings` green in the interim.
// REMOVE this `#![allow(dead_code)]` in Task 6, once `overlap_sample` uses them all.
#![allow(dead_code)]

use std::fs::File;
use std::io::ErrorKind;
use std::path::Path;

use memmap2::Mmap;
use ndarray::{Array1, Array2};

use crate::bits;
use crate::layout::{self, ContigPaths};
use crate::nrvk::LongAlleleReader;
use crate::search::{SearchTree, overlap_range};

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

/// Gather one sub-stream's overlapping, sample-carried calls into `out`.
///
/// * `positions` — ascending variant starts for this run (a var_key column slice,
///   or a whole dense-class table).
/// * `max_region_length` — the run's max deletion bound (`max_del`), `0` for SNP.
/// * `del_len(i)` — deletion length of run element `i` (for the exclusive end
///   `positions[i] + 1 + del_len(i)`); `0` for SNP runs.
/// * `carried(i)` — whether the queried `(sample, ploid)` carries element `i`
///   (always `true` for var_key columns; a genotype-bit test for dense).
/// * `decode_hit(i)` — `(ilen, alt)` for a carried, overlapping element `i`.
// The 8-argument shape is the task's locked interface (per-sub-stream decode
// callbacks passed positionally); bundling them into a struct would obscure
// the call sites more than it helps.
#[allow(clippy::too_many_arguments)]
fn gather_run(
    positions: &[u32],
    max_region_length: u32,
    q_start: u32,
    q_end: u32,
    del_len: impl Fn(usize) -> u32,
    carried: impl Fn(usize) -> bool,
    decode_hit: impl Fn(usize) -> (i32, Vec<u8>),
    out: &mut Vec<Call>,
) {
    if positions.is_empty() {
        return;
    }
    let v_ends: Vec<u32> = positions
        .iter()
        .enumerate()
        .map(|(i, &p)| p + 1 + del_len(i))
        .collect();
    let tree = SearchTree::new(positions);
    let (s_idx, e_idx) = overlap_range(&tree, &v_ends, max_region_length, q_start, q_end);
    for (i, &position) in positions.iter().enumerate().take(e_idx).skip(s_idx) {
        if carried(i) {
            let (ilen, alt) = decode_hit(i);
            out.push(Call {
                position,
                ilen,
                alt,
            });
        }
    }
}

/// K-way merge of already position-sorted runs into one position-sorted list.
/// The union-shaped dual of sorted intersection; a 5th/6th sub-stream (M11
/// `pointer/*`) is one more entry. Stable across ties (earlier run wins).
/// `O(total_calls × n_runs)` with `n_runs ≤ 4`.
fn kway_merge(runs: Vec<Vec<Call>>) -> Vec<Call> {
    let total: usize = runs.iter().map(|r| r.len()).sum();
    let mut heads = vec![0usize; runs.len()];
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        let mut best: Option<usize> = None;
        for r in 0..runs.len() {
            if heads[r] >= runs[r].len() {
                continue;
            }
            match best {
                // Keep `best` on ties so the earlier run emits first (stable).
                Some(b) if runs[b][heads[b]].position <= runs[r][heads[r]].position => {}
                _ => best = Some(r),
            }
        }
        let b = best.expect("total accounts for every remaining element");
        out.push(runs[b][heads[b]].clone());
        heads[b] += 1;
    }
    out
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

    fn call(position: u32, ilen: i32, alt: &[u8]) -> Call {
        Call {
            position,
            ilen,
            alt: alt.to_vec(),
        }
    }

    #[test]
    fn test_gather_run_snp_half_open() {
        // Pure-SNP run: positions [10, 20, 30], v_end = pos + 1, max_del 0.
        let positions = [10u32, 20, 30];
        let mut out = Vec::new();
        gather_run(
            &positions,
            0,
            15,
            25, // query [15, 25): only 20 overlaps
            |_| 0,
            |_| true,
            |i| (0, vec![b'A' + i as u8]),
            &mut out,
        );
        assert_eq!(out, vec![call(20, 0, b"B")]);
    }

    #[test]
    fn test_gather_run_deletion_spans_query_start() {
        // v0 start 2 deletes 6 bases -> v_end 9; v1 SNP at 10.
        let positions = [2u32, 10];
        let dels = [6u32, 0];
        let mut out = Vec::new();
        gather_run(
            &positions,
            6, // max_region_length covers the 6-base deletion
            5,
            7, // query [5, 7): only v0 (2..9) spans it
            |i| dels[i],
            |_| true,
            |_| (-6, Vec::new()),
            &mut out,
        );
        assert_eq!(out, vec![call(2, -6, b"")]);
    }

    #[test]
    fn test_gather_run_carried_filter() {
        // Dense-style: only even indices are carried by the sample.
        let positions = [10u32, 20, 30, 40];
        let mut out = Vec::new();
        gather_run(
            &positions,
            0,
            0,
            100,
            |_| 0,
            |i| i % 2 == 0,
            |i| (0, vec![b'A' + i as u8]),
            &mut out,
        );
        assert_eq!(out, vec![call(10, 0, b"A"), call(30, 0, b"C")]);
    }

    #[test]
    fn test_gather_run_empty_positions() {
        let mut out = Vec::new();
        gather_run(&[], 0, 0, 100, |_| 0, |_| true, |_| (0, vec![]), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_kway_merge_orders_by_position() {
        let runs = vec![
            vec![call(10, 0, b"A"), call(30, 0, b"C")],
            vec![call(20, 1, b"AT")],
            vec![],
            vec![call(25, -2, b"")],
        ];
        let merged = kway_merge(runs);
        let positions: Vec<u32> = merged.iter().map(|c| c.position).collect();
        assert_eq!(positions, vec![10, 20, 25, 30]);
    }

    #[test]
    fn test_kway_merge_ties_keep_earlier_run_first() {
        let runs = vec![
            vec![call(50, 0, b"A")],  // run 0
            vec![call(50, 1, b"AT")], // run 1, same position
        ];
        let merged = kway_merge(runs);
        assert_eq!(merged, vec![call(50, 0, b"A"), call(50, 1, b"AT")]);
    }
}

//! Mmap'd sidecar file access: raw file I/O helpers, `offsets.npy`/`max_del.npy`
//! loaders, and the `SubStreamView`/`DenseView` structs that hold a contig's
//! mmap'd var_key and dense class tables.

use std::fs::File;
use std::io::ErrorKind;
use std::ops::Range;
use std::path::Path;

use memmap2::Mmap;
use ndarray::{Array1, Array2};

use crate::bits;
use crate::layout;

/// mmap a file into memory, returning `None` for a missing or zero-length file
/// (memmap2 rejects empty maps; an absent sidecar means an empty sub-stream).
pub(crate) fn mmap_file(path: &Path) -> std::io::Result<Option<Mmap>> {
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
pub(crate) fn as_u32(m: &Option<Mmap>) -> &[u32] {
    match m {
        Some(mm) => bytemuck::cast_slice(&mm[..]),
        None => &[],
    }
}

/// Raw bytes of a mmap'd sidecar (packed SNP `alleles.bin`, `genotypes.bin`),
/// or an empty slice when missing/empty.
pub(crate) fn as_bytes(m: &Option<Mmap>) -> &[u8] {
    match m {
        Some(mm) => &mm[..],
        None => &[],
    }
}

/// A var_key sub-stream (snp or indel): mmap'd `positions.bin` + `alleles.bin`
/// with the CSR `offsets.npy` giving per-`(sample, ploid)` column bounds.
pub(crate) struct SubStreamView {
    pub(crate) positions: Option<Mmap>, // raw u32 LE, one per call
    pub(crate) keys: Option<Mmap>,      // packed 2-bit codes (snp) or u32 LE keys (indel)
    pub(crate) offsets: Vec<u64>,       // CSR prefix-sum, len == columns + 1
}

impl SubStreamView {
    pub(crate) fn positions(&self) -> &[u32] {
        as_u32(&self.positions)
    }
    /// Half-open `[start, end)` call range for flat column `c`.
    pub(crate) fn column(&self, c: usize) -> Range<usize> {
        (self.offsets[c] as usize)..(self.offsets[c + 1] as usize)
    }
}

/// A dense class table (snp or indel): shared per-contig `positions.bin` +
/// `alleles.bin` + hap-major `genotypes.bin` 1-bit matrix.
pub(crate) struct DenseView {
    pub(crate) positions: Option<Mmap>,
    pub(crate) keys: Option<Mmap>,
    pub(crate) genotypes: Option<Mmap>,
    pub(crate) n_dense_variants: usize,
}

impl DenseView {
    pub(crate) fn positions(&self) -> &[u32] {
        as_u32(&self.positions)
    }
    /// Whether haplotype `hap` carries dense variant `col` (hap-major bit
    /// `hap * n_dense_variants + col`).
    pub(crate) fn carried(&self, hap: usize, col: usize) -> bool {
        bits::get_bit(as_bytes(&self.genotypes), hap * self.n_dense_variants + col)
    }
}

/// Load a CSR `offsets.npy` (len `columns + 1`); a missing file means an empty
/// stream — return an all-zero prefix-sum so every column is empty.
pub(crate) fn load_offsets(path: &Path, columns: usize) -> std::io::Result<Vec<u64>> {
    if path.exists() {
        let a: Array1<u64> = ndarray_npy::read_npy(path)
            .map_err(|e| std::io::Error::other(format!("read {}: {e}", path.display())))?;
        Ok(a.to_vec())
    } else {
        Ok(vec![0u64; columns + 1])
    }
}

/// Load `max_del.npy` (`u32`, shape `(n_samples, ploidy)`); a missing file
/// (pure-SNP contig, or predating the post-pass) defaults to all-zero.
pub(crate) fn load_max_del(
    path: &Path,
    n_samples: usize,
    ploidy: usize,
) -> std::io::Result<Array2<u32>> {
    if path.exists() {
        ndarray_npy::read_npy(path)
            .map_err(|e| std::io::Error::other(format!("read {}: {e}", path.display())))
    } else {
        Ok(Array2::zeros((n_samples, ploidy)))
    }
}

/// Load `dense/max_del.npy` (`u32`, shape `(1,)`); missing defaults to `0`.
pub(crate) fn load_dense_max_del(path: &Path) -> std::io::Result<u32> {
    if path.exists() {
        let a: Array1<u32> = ndarray_npy::read_npy(path)
            .map_err(|e| std::io::Error::other(format!("read {}: {e}", path.display())))?;
        Ok(a.into_iter().next().unwrap_or(0))
    } else {
        Ok(0)
    }
}

/// Open a dense class table, or `None` when the class has no variants (absent
/// dir / empty `positions.bin`).
pub(crate) fn open_dense(dir: &Path) -> std::io::Result<Option<DenseView>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_offsets_missing_file_is_empty_prefix_sum() {
        let p = std::path::Path::new("/nonexistent-sp3/offsets.npy");
        let v = load_offsets(p, 3).unwrap();
        assert_eq!(v, vec![0u64; 4]);
    }

    #[test]
    fn load_offsets_corrupt_file_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("offsets.npy");
        std::fs::write(&p, b"not a valid npy header").unwrap();
        assert!(load_offsets(&p, 3).is_err());
    }
}

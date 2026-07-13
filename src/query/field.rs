//! Read the per-contig INFO/FORMAT field sidecars written by the conversion
//! pipeline. Mirrors `crate::mutcat::sidecar`: an mmap per (field, sub-stream)
//! plus indexed accessors.
//!
//! Two things this reader must get right, both of which the file itself cannot
//! tell you:
//!  * The element width comes ONLY from `meta.json`'s `fields[].dtype` — values
//!    are staged as 4-byte i32/f32 and rewritten in place at finalize.
//!  * Dense FORMAT is indexed by the ORIGINAL cohort sample index, never a
//!    selected subset slot.

use half::f16;
use memmap2::Mmap;

use crate::field::StorageDtype;
use crate::layout::{ContigPaths, FieldSub};
use crate::query::sidecar::mmap_file;

/// One field element, in the dtype it is stored as. Never widened or converted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldValue {
    Bool(bool),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    F16(f16),
    F32(f32),
}

/// An mmap'd `values.bin` for one (field, sub-stream), plus the dtype needed to
/// interpret it. A missing/empty file is a legal empty sub-stream.
pub struct FieldView {
    values: Option<Mmap>,
    dtype: StorageDtype,
    /// Cohort sample count — the stride for dense FORMAT columns.
    n_samples: usize,
    /// Element count (bytes / width).
    n: usize,
}

impl FieldView {
    pub fn open(
        paths: &ContigPaths,
        category: &str,
        name: &str,
        sub: FieldSub,
        dtype: StorageDtype,
        n_samples: usize,
    ) -> std::io::Result<Self> {
        let values = mmap_file(&paths.field_values(category, name, sub))?;
        let width = dtype.width_bytes().ok_or_else(|| {
            std::io::Error::other(format!(
                "field {name:?} has unresolved dtype {:?}; the store was never finalized",
                dtype.as_str()
            ))
        })?;
        let n = values.as_ref().map(|m| m.len() / width).unwrap_or(0);
        Ok(Self {
            values,
            dtype,
            n_samples,
            n,
        })
    }

    #[inline]
    pub fn dtype(&self) -> StorageDtype {
        self.dtype
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Zero-copy typed slice. `None` if the stored dtype's width does not match
    /// `T`, the sub-stream is empty, or the mapped length is not an exact
    /// multiple of `T`'s width (a truncated/corrupt store) — `bytemuck::cast_slice`
    /// panics on a remainder, so we validate first rather than let a bad file on
    /// disk crash the reader.
    pub fn as_slice<T: bytemuck::Pod>(&self) -> Option<&[T]> {
        let m = self.values.as_ref()?;
        if self.dtype.width_bytes()? != std::mem::size_of::<T>() {
            return None;
        }
        if m.len() % std::mem::size_of::<T>() != 0 {
            return None;
        }
        Some(bytemuck::cast_slice(&m[..]))
    }

    /// Element `i`: a var_key **call** index, or a dense INFO **variant row**.
    /// Panics if `i >= len()` (an out-of-range index means the caller's
    /// provenance disagrees with the store — a bug, not bad input).
    #[inline]
    pub fn value_at(&self, i: usize) -> FieldValue {
        let m = self
            .values
            .as_ref()
            .expect("value_at on an empty field sub-stream");
        let w = self
            .dtype
            .width_bytes()
            .expect("open() rejected unresolved dtypes");
        let b = &m[i * w..(i + 1) * w];
        match self.dtype {
            StorageDtype::Bool => FieldValue::Bool(b[0] != 0),
            StorageDtype::I8 => FieldValue::I8(b[0] as i8),
            StorageDtype::U8 => FieldValue::U8(b[0]),
            StorageDtype::I16 => FieldValue::I16(i16::from_le_bytes([b[0], b[1]])),
            StorageDtype::U16 => FieldValue::U16(u16::from_le_bytes([b[0], b[1]])),
            StorageDtype::I32 => FieldValue::I32(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            StorageDtype::U32 => FieldValue::U32(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            StorageDtype::F16 => FieldValue::F16(f16::from_le_bytes([b[0], b[1]])),
            StorageDtype::F32 => FieldValue::F32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            StorageDtype::Auto => unreachable!("open() rejects Auto"),
        }
    }

    /// Dense FORMAT element for `(dense_row, orig_sample)`.
    /// `orig_sample` MUST be the original cohort sample index.
    #[inline]
    pub fn format_at(&self, dense_row: usize, orig_sample: usize) -> FieldValue {
        debug_assert!(
            orig_sample < self.n_samples,
            "orig_sample {orig_sample} >= cohort n_samples {}",
            self.n_samples
        );
        self.value_at(dense_row * self.n_samples + orig_sample)
    }

    /// Raw little-endian bytes for element `i` (width = `dtype().width_bytes()`).
    /// Used by the Python decode path, which applies the dtype numpy-side.
    #[inline]
    pub fn bytes_at(&self, i: usize) -> &[u8] {
        let m = self
            .values
            .as_ref()
            .expect("bytes_at on an empty field sub-stream");
        let w = self
            .dtype
            .width_bytes()
            .expect("open() rejected unresolved dtypes");
        &m[i * w..(i + 1) * w]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::StorageDtype;
    use crate::layout::{ContigPaths, FieldSub};
    use std::fs;

    fn write_field(paths: &ContigPaths, cat: &str, name: &str, sub: FieldSub, bytes: &[u8]) {
        let p = paths.field_values(cat, name, sub);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, bytes).unwrap();
    }

    #[test]
    fn value_at_reads_each_dtype() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");

        let vals: [i16; 3] = [-5, 0, 300];
        write_field(
            &paths,
            "info",
            "AC",
            FieldSub::VkSnp,
            bytemuck::cast_slice(&vals),
        );
        let v =
            FieldView::open(&paths, "info", "AC", FieldSub::VkSnp, StorageDtype::I16, 2).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v.value_at(0), FieldValue::I16(-5));
        assert_eq!(v.value_at(2), FieldValue::I16(300));

        let f: [f32; 2] = [0.5, 1.25];
        write_field(
            &paths,
            "info",
            "AF",
            FieldSub::DenseSnp,
            bytemuck::cast_slice(&f),
        );
        let v = FieldView::open(
            &paths,
            "info",
            "AF",
            FieldSub::DenseSnp,
            StorageDtype::F32,
            2,
        )
        .unwrap();
        assert_eq!(v.value_at(1), FieldValue::F32(1.25));
    }

    #[test]
    fn format_at_strides_by_original_cohort_sample() {
        // 3 dense variants x 4 cohort samples, variant-major.
        // value = row*10 + sample
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let n_samples = 4usize;
        let mut vals: Vec<i32> = Vec::new();
        for row in 0..3i32 {
            for s in 0..4i32 {
                vals.push(row * 10 + s);
            }
        }
        write_field(
            &paths,
            "format",
            "DP",
            FieldSub::DenseSnp,
            bytemuck::cast_slice(&vals),
        );
        let v = FieldView::open(
            &paths,
            "format",
            "DP",
            FieldSub::DenseSnp,
            StorageDtype::I32,
            n_samples,
        )
        .unwrap();
        // Must use the ORIGINAL cohort sample index, not a selected slot.
        assert_eq!(v.format_at(0, 0), FieldValue::I32(0));
        assert_eq!(v.format_at(2, 3), FieldValue::I32(23));
        assert_eq!(v.format_at(1, 2), FieldValue::I32(12));
    }

    #[test]
    fn missing_file_opens_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let v = FieldView::open(
            &paths,
            "info",
            "NOPE",
            FieldSub::VkIndel,
            StorageDtype::F32,
            2,
        )
        .unwrap();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }
}

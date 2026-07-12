# SVAR2 Field Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let consumers read the INFO/FORMAT fields SVAR2 already writes — GenVarLoader via a public Rust `FieldView` + record provenance, and Python via `SparseVar2.decode(fields=…)`.

**Architecture:** gvl runs its *own* var_key⋈dense merge and already holds the source channel + index when it emits a record, so genoray never gathers or materializes field values. It exposes (a) `FieldView`, an mmap over `values.bin` whose element width comes from `meta.json`, and (b) `vk_src`, the one index destroyed when var_key snp+indel are merged by position. Dense provenance is free (absolute row = arithmetic on the retained window ranges). Provenance is a monomorphized generic so the no-fields path stays byte-identical and zero-cost.

**Tech Stack:** Rust (pyo3, memmap2, bytemuck, half), Python (numpy, seqpro.rag.Ragged), pixi, pytest, cargo.

**Spec:** `docs/superpowers/specs/2026-07-12-svar2-fields-read-design.md`

## Global Constraints

- **Rust tests MUST use `--no-default-features`**, else the pyo3 test binary fails to link with `undefined symbol: _Py_Dealloc`. Always: `pixi run bash -lc 'cargo test --no-default-features <args>'`.
- **Element width is NOT in `values.bin`.** It comes only from `meta.json`'s `fields[].dtype`. Never infer width from file length.
- **Dense FORMAT is indexed by the ORIGINAL cohort sample index**, never the selected subset slot. Index is `dense_row * n_samples + orig_sample`, where `n_samples` is the cohort width.
- **On-disk field path:** `{contig}/fields/{info|format}/{name}/{var_key_snp|var_key_indel|dense_snp|dense_indel}/values.bin`.
- **Missing sentinels:** `i*::MIN` / `u*::MAX` / `NaN`. A sentinel exists **only** when the field's `default` is `None`. Never translate or widen — preserve the stored dtype end to end.
- **`vk_src` packing:** bit 31 = sub-stream tag (0 = snp, 1 = indel), bits 0..=30 = absolute call index. Ceiling 2^31, enforced by a **hard assert** that lives ONLY in the provenance-carrying code path.
- **Zero-cost rule:** a no-fields gather must produce byte-identical output to today. Provenance is added via monomorphized generics, not runtime branches in the hot loop.
- Commits follow Conventional Commits (`feat:`, `test:`, `refactor:`, `docs:`).
- **Public-API rule (CLAUDE.md):** any change reachable from `import genoray` without an underscore MUST update `skills/genoray-api/SKILL.md` in the same PR.

## File Structure

**Milestone 1 — Rust read primitive + provenance (unblocks gvl)**

| File | Responsibility |
| --- | --- |
| `src/layout.rs` (modify) | Add `FieldSub` enum + `ContigPaths::field_values(cat, name, sub)`. Single source of truth for field paths (today they are built inline in `orchestrator.rs`/`field_finalize.rs`). |
| `src/field.rs` (modify) | Make `StorageDtype::parse` public; add `StorageDtype::from_meta_str`. |
| `src/query/field.rs` (create) | `FieldView` — mmap + dtype; `value_at`, `format_at`, `as_slice`. Mirrors `src/mutcat/sidecar.rs`. |
| `src/spine.rs` (modify) | `VkElem` trait, `SrcKeyRef`, `pack_vk_src`/`unpack_vk_src` (+ assert), generic `merge_by_position` and `gather_keys`. `merge_keys` becomes a thin alias. |
| `src/query/reader.rs` (modify) | `vk_slice` becomes generic over `VkElem`; add `open_field`. |
| `src/query/gather.rs` (modify) | `gather_vk` generic; `BatchResultSplit.vk_src`; `gather_haps_readbound_src`. |
| `src/query/mod.rs` (modify) | Export `FieldView`, `SrcKeyRef`, `pack_vk_src`, `unpack_vk_src`, `gather_haps_readbound_src`. |

**Milestone 2 — Python decode surface**

| File | Responsibility |
| --- | --- |
| `src/py_query_decode.rs` (modify) | `decode_batch_fields` — gather field bytes per decoded record; return raw bytes + itemsize (Python applies dtype, as it already does for `allele`). |
| `python/genoray/_svar2_fields.py` (modify) | Read the `meta.json` manifest; canonical field keys; dtype mapping. |
| `python/genoray/_svar2.py` (modify) | `SparseVar2(fields=…)`, `with_fields`, `available_fields`. |
| `python/genoray/_svar2_decode.py` (modify) | `decode()` attaches field arrays sharing the genotype offsets. |
| `tests/test_svar2_fields_read.py` (create) | e2e read tests. |

---

## Task 1: Field paths in `layout.rs`

**Files:**
- Modify: `src/layout.rs`
- Test: `src/layout.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub enum FieldSub { VkSnp, VkIndel, DenseSnp, DenseIndel }` with `pub fn dir_name(self) -> &'static str`; `ContigPaths::field_values(&self, category: &str, name: &str, sub: FieldSub) -> PathBuf`.

- [ ] **Step 1: Write the failing test**

Append inside the existing `#[cfg(test)] mod tests` in `src/layout.rs`:

```rust
    #[test]
    fn test_field_values_paths() {
        let paths = ContigPaths::new("/out", "chr1");
        assert_eq!(
            paths.field_values("format", "DS", FieldSub::VkSnp),
            Path::new("/out/chr1/fields/format/DS/var_key_snp/values.bin")
        );
        assert_eq!(
            paths.field_values("info", "AF", FieldSub::DenseIndel),
            Path::new("/out/chr1/fields/info/AF/dense_indel/values.bin")
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features test_field_values_paths'`
Expected: FAIL — `cannot find type FieldSub in this scope`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/layout.rs` (after the `MutcatSub` block):

```rust
/// The four sub-streams a field sidecar mirrors. Same four directories as
/// `MutcatSub`, but kept separate: field dirs live under `fields/{category}/`
/// and gain no `has_ref` notion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSub {
    VkSnp,
    VkIndel,
    DenseSnp,
    DenseIndel,
}

impl FieldSub {
    pub fn dir_name(self) -> &'static str {
        match self {
            FieldSub::VkSnp => "var_key_snp",
            FieldSub::VkIndel => "var_key_indel",
            FieldSub::DenseSnp => "dense_snp",
            FieldSub::DenseIndel => "dense_indel",
        }
    }
    /// Every sub-stream, in a fixed order (for iteration at open/finalize time).
    pub fn all() -> [FieldSub; 4] {
        [
            FieldSub::VkSnp,
            FieldSub::VkIndel,
            FieldSub::DenseSnp,
            FieldSub::DenseIndel,
        ]
    }
}
```

And inside `impl ContigPaths`:

```rust
    /// `{out}/{contig}/fields/{category}/{name}/{sub}/values.bin`.
    /// `category` is `"info"` or `"format"` (see `FieldCategory::as_str`).
    pub fn field_values(&self, category: &str, name: &str, sub: FieldSub) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("fields")
            .join(category)
            .join(name)
            .join(sub.dir_name())
            .join("values.bin")
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run bash -lc 'cargo test --no-default-features test_field_values_paths'`
Expected: PASS.

- [ ] **Step 5: Verify no regression in the write path**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: all existing tests PASS (this task only adds; it does not yet rewire `orchestrator.rs`).

- [ ] **Step 6: Commit**

```bash
git add src/layout.rs
git commit -m "feat(svar2): field values.bin paths in layout"
```

---

## Task 2: Public `StorageDtype` parsing

**Files:**
- Modify: `src/field.rs`
- Test: `src/field.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Consumes: existing `StorageDtype` (`src/field.rs`) with its private `fn parse(s: &str) -> Option<Self>` and `pub fn width_bytes(self) -> Option<usize>` / `pub fn as_str(self) -> &'static str`.
- Produces: `pub fn StorageDtype::from_meta_str(s: &str) -> Option<StorageDtype>` — parses a `meta.json` `fields[].dtype` string. `Auto` is rejected (returns `None`): finalize always resolves it, so `Auto` on disk means a corrupt store.

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` in `src/field.rs` (create the module if absent):

```rust
    #[test]
    fn test_from_meta_str_rejects_auto() {
        assert_eq!(StorageDtype::from_meta_str("f16"), Some(StorageDtype::F16));
        assert_eq!(StorageDtype::from_meta_str("i8"), Some(StorageDtype::I8));
        assert_eq!(StorageDtype::from_meta_str("bool"), Some(StorageDtype::Bool));
        // `auto` is never a finalized on-disk dtype.
        assert_eq!(StorageDtype::from_meta_str("auto"), None);
        assert_eq!(StorageDtype::from_meta_str("nonsense"), None);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features test_from_meta_str_rejects_auto'`
Expected: FAIL — `no function or associated item named from_meta_str`.

- [ ] **Step 3: Write minimal implementation**

Add inside `impl StorageDtype` in `src/field.rs`:

```rust
    /// Parse a finalized `meta.json` `fields[].dtype` string. Returns `None` for
    /// `"auto"` — finalize always resolves `Auto` to a concrete dtype, so `auto`
    /// on disk means the store is corrupt or was never finalized.
    pub fn from_meta_str(s: &str) -> Option<Self> {
        match Self::parse(s) {
            Some(Self::Auto) | None => None,
            Some(d) => Some(d),
        }
    }
```

Note: `Self::parse` never returns `Auto` today (it has no `"auto"` arm), but the
match is written to be total so a future `parse` change cannot leak `Auto`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run bash -lc 'cargo test --no-default-features test_from_meta_str_rejects_auto'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/field.rs
git commit -m "feat(svar2): public StorageDtype::from_meta_str"
```

---

## Task 3: `FieldView` — the mmap read primitive

**Files:**
- Create: `src/query/field.rs`
- Modify: `src/query/mod.rs` (add `pub mod field;` and re-export)
- Test: `src/query/field.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Consumes: `layout::{ContigPaths, FieldSub}` (Task 1), `StorageDtype::from_meta_str` (Task 2), `crate::query::sidecar::mmap_file` (existing, `pub(crate)`), `memmap2::Mmap`, `bytemuck`, `half::f16`.
- Produces:
  - `pub enum FieldValue { Bool(bool), I8(i8), U8(u8), I16(i16), U16(u16), I32(i32), U32(u32), F16(f16), F32(f32) }`
  - `pub struct FieldView` with `pub fn open(paths: &ContigPaths, category: &str, name: &str, sub: FieldSub, dtype: StorageDtype, n_samples: usize) -> std::io::Result<FieldView>`
  - `pub fn value_at(&self, i: usize) -> FieldValue`
  - `pub fn format_at(&self, dense_row: usize, orig_sample: usize) -> FieldValue`
  - `pub fn as_slice<T: bytemuck::Pod>(&self) -> Option<&[T]>`
  - `pub fn dtype(&self) -> StorageDtype`, `pub fn len(&self) -> usize`, `pub fn is_empty(&self) -> bool`

- [ ] **Step 1: Write the failing test**

Create `src/query/field.rs` with ONLY the test module for now:

```rust
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
        write_field(&paths, "info", "AC", FieldSub::VkSnp, bytemuck::cast_slice(&vals));
        let v = FieldView::open(&paths, "info", "AC", FieldSub::VkSnp, StorageDtype::I16, 2).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v.value_at(0), FieldValue::I16(-5));
        assert_eq!(v.value_at(2), FieldValue::I16(300));

        let f: [f32; 2] = [0.5, 1.25];
        write_field(&paths, "info", "AF", FieldSub::DenseSnp, bytemuck::cast_slice(&f));
        let v = FieldView::open(&paths, "info", "AF", FieldSub::DenseSnp, StorageDtype::F32, 2).unwrap();
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
        write_field(&paths, "format", "DP", FieldSub::DenseSnp, bytemuck::cast_slice(&vals));
        let v = FieldView::open(
            &paths, "format", "DP", FieldSub::DenseSnp, StorageDtype::I32, n_samples,
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
        let v =
            FieldView::open(&paths, "info", "NOPE", FieldSub::VkIndel, StorageDtype::F32, 2).unwrap();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `pub mod field;` to `src/query/mod.rs` (next to the existing `pub mod decode;` etc.), then:

Run: `pixi run bash -lc 'cargo test --no-default-features query::field'`
Expected: FAIL — `cannot find type FieldView in this scope`.

- [ ] **Step 3: Write minimal implementation**

Prepend to `src/query/field.rs` (above the test module):

```rust
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
    /// `T`, or the sub-stream is empty. mmap pages are page-aligned, so
    /// `bytemuck`'s alignment check always passes.
    pub fn as_slice<T: bytemuck::Pod>(&self) -> Option<&[T]> {
        let m = self.values.as_ref()?;
        if self.dtype.width_bytes()? != std::mem::size_of::<T>() {
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features query::field'`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/query/field.rs src/query/mod.rs
git commit -m "feat(svar2): FieldView mmap reader for INFO/FORMAT sidecars"
```

---

## Task 4: `vk_src` provenance — packing + the generic merge

**Files:**
- Modify: `src/spine.rs`
- Test: `src/spine.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Consumes: existing `KeyRef { position: u32, key: u32 }`, existing `gather_keys`, existing `merge_keys`.
- Produces:
  - `pub struct SrcKeyRef { pub key: KeyRef, pub src: u32 }`
  - `pub trait VkElem: Copy { fn make(position: u32, key: u32, is_indel: bool, call_idx: usize) -> Self; fn position(&self) -> u32; }` — implemented for `KeyRef` (ignores provenance entirely; **no assert**) and `SrcKeyRef` (packs + asserts).
  - `pub const VK_SRC_INDEL_BIT: u32 = 1 << 31;`
  - `pub fn pack_vk_src(is_indel: bool, call_idx: usize) -> u32`
  - `pub fn unpack_vk_src(src: u32) -> (bool, usize)`
  - `pub fn merge_by_position<T: VkElem>(runs: Vec<Vec<T>>) -> Vec<T>`
  - `pub fn merge_keys(runs: Vec<Vec<KeyRef>>) -> Vec<KeyRef>` (now an alias — signature unchanged, all existing callers keep working)
  - `gather_keys` gains two params: `is_indel: bool`, `abs_base: usize`, and is generic over `T: VkElem`.

**Why `make` takes `is_indel`/`call_idx` rather than a pre-packed `u32`:** the 2^31
assert must not run on the no-fields path. `KeyRef::make` ignores both args, so
the packing and its assert are dead code and compile away. Passing a pre-packed
`u32` would force the assert to execute even when provenance is discarded.

- [ ] **Step 1: Write the failing tests**

Append to `#[cfg(test)] mod tests` in `src/spine.rs`:

```rust
    #[test]
    fn pack_unpack_vk_src_round_trips() {
        assert_eq!(unpack_vk_src(pack_vk_src(false, 0)), (false, 0));
        assert_eq!(unpack_vk_src(pack_vk_src(true, 0)), (true, 0));
        assert_eq!(unpack_vk_src(pack_vk_src(false, 12345)), (false, 12345));
        assert_eq!(
            unpack_vk_src(pack_vk_src(true, (1 << 31) - 1)),
            (true, (1 << 31) - 1)
        );
    }

    #[test]
    #[should_panic(expected = "exceeds the 2^31 vk_src ceiling")]
    fn pack_vk_src_asserts_on_overflow() {
        pack_vk_src(false, 1 << 31);
    }

    #[test]
    fn merge_by_position_carries_src_and_keeps_snp_tie_break() {
        // snp run at positions 10, 20 (calls 0, 1); indel run at 10, 15 (calls 7, 8).
        // Ties at position 10 must keep the SNP first (run 0 wins).
        let snp: Vec<SrcKeyRef> = vec![
            SrcKeyRef { key: KeyRef { position: 10, key: 100 }, src: pack_vk_src(false, 0) },
            SrcKeyRef { key: KeyRef { position: 20, key: 200 }, src: pack_vk_src(false, 1) },
        ];
        let indel: Vec<SrcKeyRef> = vec![
            SrcKeyRef { key: KeyRef { position: 10, key: 300 }, src: pack_vk_src(true, 7) },
            SrcKeyRef { key: KeyRef { position: 15, key: 400 }, src: pack_vk_src(true, 8) },
        ];
        let out = merge_by_position(vec![snp, indel]);
        let got: Vec<(u32, bool, usize)> = out
            .iter()
            .map(|e| {
                let (is_indel, idx) = unpack_vk_src(e.src);
                (e.key.position, is_indel, idx)
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (10, false, 0), // snp wins the tie
                (10, true, 7),
                (15, true, 8),
                (20, false, 1),
            ]
        );
    }

    #[test]
    fn merge_keys_is_unchanged_for_bare_keyrefs() {
        let a = vec![KeyRef { position: 1, key: 10 }, KeyRef { position: 5, key: 50 }];
        let b = vec![KeyRef { position: 1, key: 99 }, KeyRef { position: 3, key: 30 }];
        let out = merge_keys(vec![a, b]);
        assert_eq!(
            out,
            vec![
                KeyRef { position: 1, key: 10 }, // stable: earlier run first
                KeyRef { position: 1, key: 99 },
                KeyRef { position: 3, key: 30 },
                KeyRef { position: 5, key: 50 },
            ]
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run bash -lc 'cargo test --no-default-features spine::'`
Expected: FAIL — `cannot find function pack_vk_src`, `cannot find type SrcKeyRef`.

- [ ] **Step 3: Write the implementation**

In `src/spine.rs`, add after the `KeyRef` definition:

```rust
/// Bit 31 of a packed `vk_src`: 0 = `var_key/snp`, 1 = `var_key/indel`.
pub const VK_SRC_INDEL_BIT: u32 = 1 << 31;

/// Pack a var_key record's provenance: sub-stream tag in bit 31, absolute call
/// index in bits 0..=30.
///
/// The assert is deliberate and load-bearing: a silent overflow here would
/// return values from the WRONG record, which is far worse than a panic. It is
/// unreachable on any real store (2^31 calls in a single contig sub-stream), and
/// it never runs on the no-provenance path (`KeyRef::make` ignores its args).
#[inline]
pub fn pack_vk_src(is_indel: bool, call_idx: usize) -> u32 {
    assert!(
        call_idx < (1usize << 31),
        "var_key call index {call_idx} exceeds the 2^31 vk_src ceiling"
    );
    (call_idx as u32) | if is_indel { VK_SRC_INDEL_BIT } else { 0 }
}

/// Inverse of [`pack_vk_src`]: `(is_indel, call_idx)`.
#[inline]
pub fn unpack_vk_src(src: u32) -> (bool, usize) {
    (
        src & VK_SRC_INDEL_BIT != 0,
        (src & !VK_SRC_INDEL_BIT) as usize,
    )
}

/// A `KeyRef` plus the provenance needed to index a field sidecar: which
/// var_key sub-stream it came from and its absolute call index there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SrcKeyRef {
    pub key: KeyRef,
    /// Packed via [`pack_vk_src`].
    pub src: u32,
}

/// The element type a var_key gather emits. Monomorphizing over this is what
/// keeps the no-fields path zero-cost: `KeyRef::make` discards the provenance
/// args, so the packing (and its assert) is dead code and compiles away.
pub trait VkElem: Copy {
    fn make(position: u32, key: u32, is_indel: bool, call_idx: usize) -> Self;
    fn position(&self) -> u32;
}

impl VkElem for KeyRef {
    #[inline]
    fn make(position: u32, key: u32, _is_indel: bool, _call_idx: usize) -> Self {
        KeyRef { position, key }
    }
    #[inline]
    fn position(&self) -> u32 {
        self.position
    }
}

impl VkElem for SrcKeyRef {
    #[inline]
    fn make(position: u32, key: u32, is_indel: bool, call_idx: usize) -> Self {
        SrcKeyRef {
            key: KeyRef { position, key },
            src: pack_vk_src(is_indel, call_idx),
        }
    }
    #[inline]
    fn position(&self) -> u32 {
        self.key.position
    }
}
```

Replace `merge_keys` with the generic merge plus a compatibility alias. **The body
is byte-for-byte the old one with `.position` → `.position()`:**

```rust
/// K-way merge of already position-sorted runs into one position-sorted list.
/// Stable across ties (earlier run wins). `O(total × n_runs)` with `n_runs`
/// small (2 within a channel; more only if M11 adds a `pointer` sub-stream).
///
/// `query::gather_haps_readbound` hand-inlines a 2-run (snp_run, indel_run)
/// equivalent of this merge for allocation reasons — any change to this
/// function's ordering or tie-break MUST be mirrored there. That twin now
/// carries the same `VkElem` payload, so the mirroring covers `src` too.
pub fn merge_by_position<T: VkElem>(runs: Vec<Vec<T>>) -> Vec<T> {
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
                Some(b) if runs[b][heads[b]].position() <= runs[r][heads[r]].position() => {}
                _ => best = Some(r),
            }
        }
        let b = best.expect("total accounts for every remaining element");
        out.push(runs[b][heads[b]]);
        heads[b] += 1;
    }
    out
}

/// Bare-`KeyRef` merge — the pre-existing signature, now a thin alias so every
/// existing caller is untouched and provably unchanged.
pub fn merge_keys(runs: Vec<Vec<KeyRef>>) -> Vec<KeyRef> {
    merge_by_position(runs)
}
```

Make `gather_keys` generic. Replace its signature and push site:

```rust
#[allow(clippy::too_many_arguments)]
pub fn gather_keys<T: VkElem>(
    positions: &[u32],
    max_region_length: u32,
    q_start: u32,
    q_end: u32,
    del_len: impl Fn(usize) -> u32,
    carried: impl Fn(usize) -> bool,
    to_key: impl Fn(usize) -> u32,
    /// Sub-stream tag for provenance (`false` = snp, `true` = indel).
    is_indel: bool,
    /// Absolute index of `positions[0]` in the sub-stream's packed buffer, so
    /// the absolute call index of element `i` is `abs_base + i`.
    abs_base: usize,
    out: &mut Vec<T>,
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
        if carried(i) && q_start < v_ends[i] {
            out.push(T::make(position, to_key(i), is_indel, abs_base + i));
        }
    }
}
```

- [ ] **Step 4: Fix the two existing `gather_keys` call sites**

`src/query/reader.rs::vk_slice` calls `gather_keys` twice. Add the two new args
(the snp block already has `o0` in scope; so does the indel block):

- snp block: after the `to_key` closure argument, insert `false, o0,`
- indel block: after the `to_key` closure argument, insert `true, o0,`

(Task 5 makes `vk_slice` itself generic; for now it still builds `Vec<KeyRef>`, so
`T` infers to `KeyRef` and the provenance args are discarded.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: PASS — the 4 new spine tests plus every existing test (the whole suite
must be green: `merge_keys`'s behaviour is unchanged by construction).

- [ ] **Step 6: Commit**

```bash
git add src/spine.rs src/query/reader.rs
git commit -m "feat(svar2): vk_src provenance packing + generic merge_by_position"
```

---

## Task 5: Thread provenance through all three var_key gathers

There are **three** producers of the `vk` channel and every one merges snp+indel,
destroying provenance. All three must become generic over `VkElem`:

1. `ContigReader::vk_slice` (`src/query/reader.rs:119`) → used by `overlap_batch`
2. `gather_vk` (`src/query/gather.rs:126`) → used by `gather_ranges`
3. the hand-inlined merge in `gather_haps_readbound` (`src/query/gather.rs:545-598`)

**Files:**
- Modify: `src/query/reader.rs`, `src/query/gather.rs`
- Test: `src/query/gather.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Consumes: `spine::{VkElem, SrcKeyRef, KeyRef, merge_by_position, pack_vk_src, unpack_vk_src}` (Task 4).
- Produces:
  - `ContigReader::vk_slice<T: VkElem>(...) -> Vec<T>`
  - `gather_vk<T: VkElem>(reader, vk_snp_range, vk_indel_range, q_start) -> Vec<T>`
  - `BatchResultSplit` gains `pub vk_src: Vec<u32>` (empty on the no-provenance path, one entry per `vk` record otherwise).
  - `pub fn gather_haps_readbound_src(reader: &ContigReader, rb: &HapRanges<'_>) -> BatchResultSplit` — fills `vk_src`.
  - `gather_haps_readbound` keeps its exact signature and leaves `vk_src` empty.

- [ ] **Step 1: Write the failing test**

Append to `#[cfg(test)] mod tests` in `src/query/gather.rs`:

```rust
    /// `vk_src` must be aligned 1:1 with `vk`, and each entry must point at the
    /// record that actually produced it — including through the position-merge
    /// of snp+indel and through the overlap FILTER that drops records (the
    /// filter is why the index can't be recovered from run position afterwards).
    #[test]
    fn vk_src_is_aligned_with_vk_and_survives_the_merge() {
        let (dir, n_samples, ploidy) = super::tests_support::tiny_store_with_snps_and_indels();
        let reader =
            ContigReader::open(dir.path().to_str().unwrap(), "chr1", n_samples, ploidy).unwrap();

        let regions = [(0u32, 1_000u32)];
        let rb = find_ranges(&reader, &regions, None);

        // Build the equivalent HapRanges for the readbound path (1 region x all samples).
        let region_starts: Vec<u32> = (0..n_samples).map(|_| regions[0].0).collect();
        let orig_samples: Vec<usize> = (0..n_samples).collect();
        let hr = HapRanges::new(
            &region_starts,
            &orig_samples,
            &rb.vk_snp_range,
            &rb.vk_indel_range,
            &rb.dense_snp_range.repeat(n_samples),
            &rb.dense_indel_range.repeat(n_samples),
            ploidy,
        );

        let plain = gather_haps_readbound(&reader, &hr);
        let with_src = gather_haps_readbound_src(&reader, &hr);

        // The no-provenance path is unchanged, and provenance is purely additive.
        assert!(plain.vk_src.is_empty(), "plain path must not populate vk_src");
        assert_eq!(plain.vk, with_src.vk, "adding provenance must not change vk");
        assert_eq!(plain.vk_off, with_src.vk_off);
        assert_eq!(with_src.vk_src.len(), with_src.vk.len());

        // Every vk_src must resolve to the record that produced its key.
        let snp_pos = reader.vk_snp.positions();
        let indel_pos = reader.vk_indel.positions();
        for (i, kr) in with_src.vk.iter().enumerate() {
            let (is_indel, idx) = spine::unpack_vk_src(with_src.vk_src[i]);
            let pos = if is_indel { indel_pos[idx] } else { snp_pos[idx] };
            assert_eq!(
                pos, kr.position,
                "vk_src[{i}] points at position {pos} but vk[{i}] is at {}",
                kr.position
            );
        }
    }
```

Add a small fixture helper module in the same `#[cfg(test)]` block. It builds a
store with BOTH snp and indel var_key records so the merge and the tie-break are
actually exercised:

```rust
    pub(super) mod tests_support {
        use tempfile::TempDir;

        /// Build a minimal on-disk contig with var_key snp + indel records for
        /// 2 samples at ploidy 2. Reuses the crate's existing conversion test
        /// helper so the store is a real, finalized one.
        pub fn tiny_store_with_snps_and_indels() -> (TempDir, usize, usize) {
            crate::testutil::tiny_svar2_store()
        }
    }
```

> **If `crate::testutil::tiny_svar2_store()` does not exist**, find the helper the
> existing gather/readbound tests already use to build a store (grep the
> `#[cfg(test)]` blocks in `src/query/gather.rs` and `tests/test_readbound_gather.rs`
> for `ContigReader::open`) and call that instead. Do **not** hand-roll a new
> fixture — reuse the existing one so this test exercises a real finalized store.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features vk_src_is_aligned'`
Expected: FAIL — `cannot find function gather_haps_readbound_src`; `no field vk_src on BatchResultSplit`.

- [ ] **Step 3: Make `vk_slice` and `gather_vk` generic**

In `src/query/reader.rs`, change the `vk_slice` signature and its final merge:

```rust
    pub(crate) fn vk_slice<T: spine::VkElem>(
        &self,
        col: usize,
        sample: usize,
        p: usize,
        q_start: u32,
        q_end: u32,
    ) -> Vec<T> {
        let mut runs: Vec<Vec<T>> = Vec::with_capacity(2);
```

…leaving both `gather_keys` blocks as edited in Task 4 (they now push `T`), and
ending with:

```rust
        spine::merge_by_position(runs)
    }
```

In `src/query/gather.rs`, make `gather_vk` generic. Body is unchanged except the
element construction and the merge:

```rust
pub(crate) fn gather_vk<T: spine::VkElem>(
    reader: &ContigReader,
    vk_snp_range: Range<usize>,
    vk_indel_range: Range<usize>,
    q_start: u32,
) -> Vec<T> {
    let snp_positions = reader.vk_snp.positions();
    let snp_keys = as_bytes(&reader.vk_snp.keys);
    let indel_positions = reader.vk_indel.positions();
    let indel_keys = as_u32(&reader.vk_indel.keys);

    let (ss, se) = (vk_snp_range.start, vk_snp_range.end);
    let mut snp_run: Vec<T> = Vec::new();
    for (j, &pos) in snp_positions.iter().enumerate().take(se).skip(ss) {
        if q_start < pos + 1 {
            // snp v_end = pos + 1
            snp_run.push(T::make(
                pos,
                rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                false,
                j,
            ));
        }
    }

    let (is_, ie_) = (vk_indel_range.start, vk_indel_range.end);
    let mut indel_run: Vec<T> = Vec::new();
    for j in is_..ie_ {
        let pos = indel_positions[j];
        let v_end = pos + 1 + rvk::deletion_len(indel_keys[j]);
        if q_start < v_end {
            indel_run.push(T::make(pos, indel_keys[j], true, j));
        }
    }

    spine::merge_by_position(vec![snp_run, indel_run])
}
```

`overlap_batch` and `gather_ranges` call these with no turbofish; `T` infers to
`KeyRef` from the `Vec<KeyRef>` they extend. No behaviour change.

- [ ] **Step 4: Add `vk_src` to `BatchResultSplit` and split the readbound gather**

In `src/query/gather.rs`, add the field to `BatchResultSplit` (after `vk_off`):

```rust
    /// Absolute var_key provenance per `vk` record, packed by
    /// `spine::pack_vk_src` (bit 31 = sub-stream, bits 0..=30 = call index).
    /// **Empty** unless produced by `gather_haps_readbound_src` — the plain
    /// gather does not pay for it.
    pub vk_src: Vec<u32>,
```

Add `vk_src: Vec::new(),` to every existing `BatchResultSplit { … }` literal (the
one in `gather_haps_readbound` and the one in `src/query/oracle.rs:188`).

Now make the readbound gather generic. Rename the existing
`pub fn gather_haps_readbound` body to a private generic
`fn gather_haps_readbound_impl<T: spine::VkElem>(reader, rb) -> (Vec<T>, Vec<usize>, /* rest of BatchResultSplit fields */)`.
The mechanical change inside the hot loop is exactly three edits:

```rust
            // was: let mut snp_run: Vec<KeyRef> = Vec::with_capacity(...)
            let mut snp_run: Vec<T> = Vec::with_capacity(ve.saturating_sub(vs));
            for (k, &pos) in snp_positions[vs..ve].iter().enumerate() {
                let j = vs + k;
                if qs < pos + 1 {
                    // was: snp_run.push(KeyRef { position: pos, key: ... });
                    snp_run.push(T::make(
                        pos,
                        rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                        false,
                        j,
                    ));
                }
            }

            // indel run — RE-ADD the absolute index the zip previously dropped:
            let mut indel_run: Vec<T> = Vec::with_capacity(vie.saturating_sub(vis));
            for (k, (&pos, &key)) in indel_positions[vis..vie]
                .iter()
                .zip(&indel_keys[vis..vie])
                .enumerate()
            {
                let j = vis + k;
                let v_end = pos + 1 + rvk::deletion_len(key);
                if qs < v_end {
                    // was: indel_run.push(KeyRef { position: pos, key });
                    indel_run.push(T::make(pos, key, true, j));
                }
            }
```

and the hand-inlined 2-way merge, which now compares via the trait:

```rust
            // Specialized 2-way merge, provably byte-identical to
            // `spine::merge_by_position(vec![snp_run, indel_run])`: that generic
            // k-way merge picks `best = Some(0)` (snp_run) on the first scan and
            // only switches to run 1 (indel_run) when
            // `!(snp_run[h0].position() <= indel_run[h1].position())`, i.e.
            // exactly the `<=` two-pointer comparison below (ties still favor
            // snp_run). The `VkElem` payload rides along unchanged, so this stays
            // a mirror of the generic merge for `SrcKeyRef` as well as `KeyRef`.
            let (mut si, mut ii) = (0usize, 0usize);
            while si < snp_run.len() && ii < indel_run.len() {
                if snp_run[si].position() <= indel_run[ii].position() {
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
```

(`vk` is `Vec<T>`. Keep the existing long comment block above the merge — extend
it with the `VkElem` sentence as shown. `use crate::spine::VkElem;` must be in
scope for `.position()`.)

Then the two public entry points, which differ only in `T` and in how they
project `vk`:

```rust
/// Flat per-query read-bound gather (see `gather_haps_readbound_impl`). Does NOT
/// populate `vk_src` — the no-provenance path, byte-identical to pre-field
/// behaviour and zero-cost (`KeyRef::make` discards the provenance args).
pub fn gather_haps_readbound(reader: &ContigReader, rb: &HapRanges<'_>) -> BatchResultSplit {
    gather_haps_readbound_impl::<KeyRef>(reader, rb)
}

/// As `gather_haps_readbound`, but also populates `vk_src` so a consumer can map
/// each merged var_key record back to its `(sub-stream, absolute call index)` and
/// index a `FieldView`. This is the entry point gvl uses to read fields.
pub fn gather_haps_readbound_src(reader: &ContigReader, rb: &HapRanges<'_>) -> BatchResultSplit {
    gather_haps_readbound_impl::<SrcKeyRef>(reader, rb)
}
```

`gather_haps_readbound_impl<T>` builds `vk: Vec<T>` and, at the end, splits it:

```rust
    // Project the generic element back into the frozen (vk, vk_src) contract.
    // For T = KeyRef, `src_of` is `None` and vk_src stays empty.
    let (vk, vk_src) = T::split(vk);
```

Add to the `VkElem` trait in `src/spine.rs` (and its two impls):

```rust
    /// Split a merged run into the frozen `(keys, src)` contract. `KeyRef`
    /// yields an empty `src`; `SrcKeyRef` yields one entry per key.
    fn split(v: Vec<Self>) -> (Vec<KeyRef>, Vec<u32>);
```

```rust
// in impl VkElem for KeyRef
    #[inline]
    fn split(v: Vec<Self>) -> (Vec<KeyRef>, Vec<u32>) {
        (v, Vec::new())
    }

// in impl VkElem for SrcKeyRef
    #[inline]
    fn split(v: Vec<Self>) -> (Vec<KeyRef>, Vec<u32>) {
        let mut keys = Vec::with_capacity(v.len());
        let mut src = Vec::with_capacity(v.len());
        for e in v {
            keys.push(e.key);
            src.push(e.src);
        }
        (keys, src)
    }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run bash -lc 'cargo test --no-default-features vk_src_is_aligned'`
Expected: PASS.

- [ ] **Step 6: Run the full suite — the zero-cost / no-regression gate**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: ALL PASS. In particular `tests/test_readbound_gather.rs` (the
readbound-vs-union oracle) must still pass unchanged — that is the proof that
`gather_haps_readbound`'s output is byte-identical to before.

- [ ] **Step 7: Commit**

```bash
git add src/spine.rs src/query/reader.rs src/query/gather.rs src/query/oracle.rs
git commit -m "feat(svar2): thread vk_src provenance through the var_key gathers"
```

---

## Task 6: Export the public Rust API + dense-provenance helper

**Files:**
- Modify: `src/query/mod.rs`, `src/lib.rs`
- Test: `tests/test_field_read.rs` (create — an integration test, proving the API is reachable from *outside* the crate exactly as gvl reaches it)

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces (the public surface gvl consumes):
  - `genoray::query::{FieldView, FieldValue}`
  - `genoray::query::{SrcKeyRef, pack_vk_src, unpack_vk_src, VK_SRC_INDEL_BIT}`
  - `genoray::query::gather_haps_readbound_src`
  - `genoray::layout::FieldSub`, `genoray::field::StorageDtype`
  - `pub fn dense_abs_row(window_range: &Range<usize>, out_range: &Range<usize>, i: usize) -> usize` — documents the dense arithmetic so consumers don't re-derive it.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_field_read.rs`:

```rust
//! Proves the field-read API is reachable from OUTSIDE the crate — the same way
//! GenVarLoader consumes it (Cargo path-dep, `default-features = false`).

use genoray::field::StorageDtype;
use genoray::layout::{ContigPaths, FieldSub};
use genoray::query::{FieldValue, FieldView, pack_vk_src, unpack_vk_src};

#[test]
fn field_view_is_publicly_constructible() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
    let p = paths.field_values("format", "DS", FieldSub::VkSnp);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    let vals: [f32; 3] = [0.0, 1.5, 2.0];
    std::fs::write(&p, bytemuck::cast_slice(&vals)).unwrap();

    let v = FieldView::open(&paths, "format", "DS", FieldSub::VkSnp, StorageDtype::F32, 1).unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(v.value_at(1), FieldValue::F32(1.5));
    assert_eq!(v.as_slice::<f32>().unwrap(), &vals[..]);
}

#[test]
fn vk_src_helpers_are_public() {
    let s = pack_vk_src(true, 42);
    assert_eq!(unpack_vk_src(s), (true, 42));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features --test test_field_read'`
Expected: FAIL — `module query is private` / unresolved imports.

- [ ] **Step 3: Add the exports**

In `src/query/mod.rs`, alongside the existing `pub use` block:

```rust
pub mod field;

pub use field::{FieldValue, FieldView};
pub use crate::spine::{SrcKeyRef, VK_SRC_INDEL_BIT, VkElem, pack_vk_src, unpack_vk_src};
pub use gather::gather_haps_readbound_src;
```

Add `dense_abs_row` to `src/query/gather.rs` and export it:

```rust
/// Absolute dense **variant row** for entry `i` of a `BatchResultSplit` dense
/// window — the index a `dense_snp`/`dense_indel` field `values.bin` is aligned
/// to (INFO: `value_at(row)`; FORMAT: `format_at(row, orig_sample)`).
///
/// Dense provenance needs no extra state: each per-region window is a contiguous
/// copy of the on-disk slice, so the absolute row is pure arithmetic.
/// `on_disk` is `HapRanges::dense_*_range[q]`; `out` is
/// `BatchResultSplit::dense_*_range[q]`; `i` indexes into `out`.
#[inline]
pub fn dense_abs_row(on_disk: &Range<usize>, out: &Range<usize>, i: usize) -> usize {
    debug_assert!(out.contains(&i), "i must index into the output window");
    on_disk.start + (i - out.start)
}
```

Ensure `src/lib.rs` exposes the modules the test imports (`pub mod query;`,
`pub mod layout;`, `pub mod field;` — add whichever are not already `pub`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run bash -lc 'cargo test --no-default-features --test test_field_read'`
Expected: PASS — 2 tests.

- [ ] **Step 5: Run the full suite**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add src/query/mod.rs src/query/gather.rs src/lib.rs tests/test_field_read.rs
git commit -m "feat(svar2): export FieldView + vk_src provenance as public Rust API"
```

**MILESTONE 1 COMPLETE — gvl is unblocked.** gvl can now call
`gather_haps_readbound_src`, index a `FieldView` by `unpack_vk_src(vk_src[i])`
for var_key records and by `dense_abs_row(...)` for dense records, and push the
value next to `pos`/`ilen` in its own merge. Spec #3 (SVAR1→SVAR2) is unblocked.

---

## Task 7: Python — read the `meta.json` field manifest

**Files:**
- Modify: `python/genoray/_svar2_fields.py`
- Test: `tests/test_svar2_fields_read.py` (create)

**Interfaces:**
- Consumes: `meta.json`'s `fields` array — `[{"name","category","dtype","default"}]` (written by `src/meta.rs:23-33`).
- Produces:
  - `NP_DTYPE: dict[str, np.dtype]` mapping the 9 storage dtypes to numpy.
  - `@dataclass(frozen=True) class StoredField: name: str; category: Literal["info","format"]; dtype: np.dtype; default: float | None; key: str`
  - `def _load_field_manifest(meta: dict) -> dict[str, StoredField]` — keyed by **canonical key**: the bare name when unique across categories, else bcftools-style `INFO/<name>` / `FORMAT/<name>`.
  - `def _resolve_read_fields(requested: Sequence[str] | None, available: dict[str, StoredField]) -> list[StoredField]` — `None` → `[]` (opt in explicitly); unknown key → `ValueError` naming the available keys.

- [ ] **Step 1: Write the failing test**

Create `tests/test_svar2_fields_read.py`:

```python
import numpy as np
import pytest

from genoray._svar2_fields import (
    StoredField,
    _load_field_manifest,
    _resolve_read_fields,
)


def test_canonical_keys_are_bare_when_unique():
    meta = {
        "fields": [
            {"name": "AF", "category": "info", "dtype": "f32", "default": None},
            {"name": "DS", "category": "format", "dtype": "f16", "default": 0.0},
        ]
    }
    avail = _load_field_manifest(meta)
    assert set(avail) == {"AF", "DS"}
    assert avail["AF"].dtype == np.dtype("float32")
    assert avail["DS"].dtype == np.dtype("float16")
    assert avail["DS"].default == 0.0
    assert avail["AF"].default is None


def test_colliding_names_are_qualified_bcftools_style():
    meta = {
        "fields": [
            {"name": "DP", "category": "info", "dtype": "i32", "default": None},
            {"name": "DP", "category": "format", "dtype": "i16", "default": None},
        ]
    }
    avail = _load_field_manifest(meta)
    assert set(avail) == {"INFO/DP", "FORMAT/DP"}
    assert avail["INFO/DP"].dtype == np.dtype("int32")
    assert avail["FORMAT/DP"].dtype == np.dtype("int16")


def test_resolve_rejects_unknown_field():
    avail = _load_field_manifest(
        {"fields": [{"name": "AF", "category": "info", "dtype": "f32", "default": None}]}
    )
    assert _resolve_read_fields(None, avail) == []
    assert _resolve_read_fields(["AF"], avail) == [avail["AF"]]
    with pytest.raises(ValueError, match="NOPE"):
        _resolve_read_fields(["NOPE"], avail)


def test_no_fields_in_meta_is_empty_not_an_error():
    assert _load_field_manifest({}) == {}
    assert _load_field_manifest({"fields": []}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar2_fields_read.py -v`
Expected: FAIL — `ImportError: cannot import name 'StoredField'`.

- [ ] **Step 3: Write the implementation**

Append to `python/genoray/_svar2_fields.py`:

```python
import numpy as np

#: Storage dtype (meta.json `fields[].dtype`) -> numpy dtype. The 9 dtypes the
#: writer can resolve to; `bool` is stored as one byte per element.
NP_DTYPE: dict[str, np.dtype] = {
    "bool": np.dtype("bool"),
    "i8": np.dtype("int8"),
    "u8": np.dtype("uint8"),
    "i16": np.dtype("int16"),
    "u16": np.dtype("uint16"),
    "i32": np.dtype("int32"),
    "u32": np.dtype("uint32"),
    "f16": np.dtype("float16"),
    "f32": np.dtype("float32"),
}


@dataclass(frozen=True)
class StoredField:
    """A field present in a finished SVAR2 store, as declared by ``meta.json``.

    ``key`` is the canonical name callers use: the bare ``name`` when it is
    unique across categories, else bcftools-style ``INFO/<name>`` /
    ``FORMAT/<name>``. ``default`` is the fill written for VCF-missing entries;
    when it is ``None`` the store instead uses a reserved sentinel
    (``iinfo.min`` / ``iinfo.max`` / ``NaN``), which is returned as-is.
    """

    name: str
    category: Literal["info", "format"]
    dtype: np.dtype
    default: float | None
    key: str


def _load_field_manifest(meta: dict) -> dict[str, StoredField]:
    """Parse ``meta.json``'s ``fields`` array into canonical-key -> StoredField.

    A store written before fields existed simply has no ``fields`` key; that is
    an empty manifest, not an error.
    """
    entries = meta.get("fields") or []
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["name"]] = counts.get(e["name"], 0) + 1

    out: dict[str, StoredField] = {}
    for e in entries:
        name = e["name"]
        category = e["category"]
        dtype_str = e["dtype"]
        if dtype_str not in NP_DTYPE:
            raise ValueError(
                f"field {name!r} has unsupported stored dtype {dtype_str!r}; "
                "the store may be corrupt or written by a newer genoray"
            )
        key = name if counts[name] == 1 else f"{category.upper()}/{name}"
        out[key] = StoredField(
            name=name,
            category=category,
            dtype=NP_DTYPE[dtype_str],
            default=e.get("default"),
            key=key,
        )
    return out


def _resolve_read_fields(
    requested: "Sequence[str] | None", available: dict[str, StoredField]
) -> list[StoredField]:
    """Validate a field selection against the store's manifest.

    ``None`` selects nothing (fields are opt-in — decoding them costs I/O).
    """
    if requested is None:
        return []
    out: list[StoredField] = []
    for key in requested:
        if key not in available:
            raise ValueError(
                f"field {key!r} is not in this store; available fields: "
                f"{sorted(available)}"
            )
        out.append(available[key])
    return out
```

Ensure the module's imports include `from typing import Literal` (already
present) and `from dataclasses import dataclass` (already present).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_svar2_fields_read.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_svar2_fields.py tests/test_svar2_fields_read.py
git commit -m "feat(svar2): parse the meta.json field manifest for reads"
```

---

## Task 8: Rust — gather field bytes per decoded record

`decode_batch` (`src/py_query_decode.rs`) emits one record per `(region, sample,
ploid, variant)`. This task makes it also emit, per requested field, one **raw
little-endian element** per record. Python applies the dtype — exactly the trick
already used for `allele` (`d["allele"].view("S1")`), so Rust needs no numpy
dtype dispatch.

**Files:**
- Modify: `src/query/gather.rs` (`BatchResult` gains `vk_src`; `decode_hap` gains a provenance-emitting twin), `src/py_query_decode.rs`
- Test: `src/query/gather.rs` (inline)

**Interfaces:**
- Consumes: `FieldView` (Task 3), `SrcKeyRef`/`unpack_vk_src` (Task 4), generic `vk_slice` (Task 5).
- Produces:
  - `BatchResult` gains `pub vk_src: Vec<u32>` (empty unless requested), by the same `VkElem` split as Task 5.
  - `pub struct RecordSrc { pub is_dense: bool, pub is_indel: bool, pub idx: usize }` — provenance of one decoded record. For var_key, `idx` is the absolute call index; for dense, the absolute dense variant row.
  - `BatchResult::decode_hap_src(&self, reader, r, s, p) -> (HapCalls, Vec<RecordSrc>)` — as `decode_hap`, but also returns per-record provenance in the same order.
  - `PyContigReader::decode_batch_fields(regions, fields) -> PyDict` where `fields: Vec<(String, String, String)>` is `(category, name, dtype_str)`. Adds one key per field: `field_<key>` → `PyArray1<u8>` of `n_records * itemsize` raw bytes, plus `field_itemsize_<key>` → `usize`.

- [ ] **Step 1: Write the failing test**

Append to `#[cfg(test)] mod tests` in `src/query/gather.rs`:

```rust
    /// `decode_hap_src` must emit exactly one provenance entry per decoded
    /// record, in the same order, and each must resolve to the record's position.
    #[test]
    fn decode_hap_src_is_aligned_with_decoded_records() {
        let (dir, n_samples, ploidy) = super::tests_support::tiny_store_with_snps_and_indels();
        let reader =
            ContigReader::open(dir.path().to_str().unwrap(), "chr1", n_samples, ploidy).unwrap();
        let regions = [(0u32, 1_000u32)];
        let br = overlap_batch_src(&reader, &regions);

        let snp_pos = reader.vk_snp.positions();
        let indel_pos = reader.vk_indel.positions();

        for s in 0..n_samples {
            for p in 0..ploidy {
                let (hc, srcs) = br.decode_hap_src(&reader, 0, s, p);
                assert_eq!(hc.positions.len(), srcs.len(), "one src per decoded record");
                for (i, src) in srcs.iter().enumerate() {
                    if !src.is_dense {
                        let pos = if src.is_indel {
                            indel_pos[src.idx]
                        } else {
                            snp_pos[src.idx]
                        };
                        assert_eq!(pos, hc.positions[i]);
                    }
                }
            }
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features decode_hap_src_is_aligned'`
Expected: FAIL — `cannot find function overlap_batch_src`.

- [ ] **Step 3: Implement `vk_src` on `BatchResult` + `decode_hap_src`**

In `src/query/gather.rs`:

Add to `BatchResult` (after `vk_off`):

```rust
    /// As `BatchResultSplit::vk_src`. Empty unless produced by `overlap_batch_src`.
    pub vk_src: Vec<u32>,
```

Add `vk_src: Vec::new(),` to the existing `BatchResult { … }` literals in
`overlap_batch` and `gather_ranges`, then generalize `overlap_batch` the same way
as Task 5 (private `overlap_batch_impl<T: VkElem>`, with `overlap_batch` =
`::<KeyRef>` and a new `pub fn overlap_batch_src` = `::<SrcKeyRef>`; the only
change inside is `reader.vk_slice::<T>(...)` and the final `T::split(vk)`).

Add `RecordSrc` and `decode_hap_src`:

```rust
/// Where one decoded record came from, so a consumer can index a `FieldView`.
/// var_key: `idx` is the absolute call index in `var_key/{snp,indel}`.
/// dense: `idx` is the absolute dense variant row in `dense/{snp,indel}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordSrc {
    pub is_dense: bool,
    pub is_indel: bool,
    pub idx: usize,
}

impl BatchResult {
    /// As `decode_hap`, but also returns one `RecordSrc` per decoded record, in
    /// the same order. Requires a `BatchResult` built by `overlap_batch_src`
    /// (i.e. with `vk_src` populated); panics otherwise, since silently
    /// returning wrong provenance would return wrong field values.
    pub fn decode_hap_src(
        &self,
        reader: &ContigReader,
        r: usize,
        s: usize,
        p: usize,
    ) -> (HapCalls, Vec<RecordSrc>) {
        assert_eq!(
            self.vk_src.len(),
            self.vk.len(),
            "decode_hap_src requires a BatchResult from overlap_batch_src"
        );
        let h = (r * self.n_samples + s) * self.ploidy + p;

        // var_key run for this hap, carrying provenance.
        let (lo, hi) = (self.vk_off[h], self.vk_off[h + 1]);
        let vk_run: Vec<(KeyRef, RecordSrc)> = (lo..hi)
            .map(|i| {
                let (is_indel, idx) = spine::unpack_vk_src(self.vk_src[i]);
                (
                    self.vk[i],
                    RecordSrc {
                        is_dense: false,
                        is_indel,
                        idx,
                    },
                )
            })
            .collect();

        // Dense run: present entries of this region's window. `dense.src[j]`
        // already holds `(class, per-class row)` — the index dense field
        // values.bin is aligned to.
        let dense = reader.dense_union();
        let (ds, de) = (self.dense_range[r].start, self.dense_range[r].end);
        let bit0 = self.dense_present_off[h];
        let mut dn_run: Vec<(KeyRef, RecordSrc)> = Vec::new();
        for (k, j) in (ds..de).enumerate() {
            if bits::get_bit(&self.dense_present, bit0 + k) {
                let (class, dcol) = dense.src[j];
                dn_run.push((
                    self.dense[j],
                    RecordSrc {
                        is_dense: true,
                        is_indel: matches!(class, crate::dense::DenseClass::Indel),
                        idx: dcol,
                    },
                ));
            }
        }

        // Merge by position with the SAME stable tie-break as `decode_hap`
        // (var_key run first on ties).
        let mut merged: Vec<(KeyRef, RecordSrc)> = Vec::with_capacity(vk_run.len() + dn_run.len());
        let (mut a, mut b) = (0usize, 0usize);
        while a < vk_run.len() && b < dn_run.len() {
            if vk_run[a].0.position <= dn_run[b].0.position {
                merged.push(vk_run[a]);
                a += 1;
            } else {
                merged.push(dn_run[b]);
                b += 1;
            }
        }
        merged.extend_from_slice(&vk_run[a..]);
        merged.extend_from_slice(&dn_run[b..]);

        let lut = reader.lut.as_ref();
        let mut hc = HapCalls::default();
        let mut srcs = Vec::with_capacity(merged.len());
        for (kr, src) in merged {
            let c = decode_keyref(kr, lut);
            hc.positions.push(c.position);
            hc.ilens.push(c.ilen);
            hc.alts.push(c.alt);
            srcs.push(src);
        }
        (hc, srcs)
    }
}
```

> **Ordering invariant:** `decode_hap` merges via
> `spine::merge_keys(vec![vk_slice, dn])`, which is stable with the earlier run
> (var_key) winning ties. The two-pointer merge above reproduces exactly that.
> A test in Step 5 pins `decode_hap_src`'s record order to `decode_hap`'s.

- [ ] **Step 4: Add the pyo3 binding**

In `src/py_query_decode.rs`, add to the `#[pymethods] impl PyContigReader` block:

```rust
    /// As `decode_batch`, plus one raw-bytes buffer per requested field, aligned
    /// 1:1 with the decoded records. `fields` is `(category, name, dtype_str)`.
    /// Values are returned as little-endian bytes + an itemsize; Python applies
    /// the numpy dtype (same trick as `allele`), so Rust does no dtype dispatch.
    #[pyo3(signature = (regions, fields, base_dir, contig))]
    pub fn decode_batch_fields<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        fields: Vec<(String, String, String)>,
        base_dir: &str,
        contig: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        use crate::field::StorageDtype;
        use crate::layout::{ContigPaths, FieldSub};
        use crate::query::{FieldView, gather::overlap_batch_src};

        let br = overlap_batch_src(&self.inner, &regions);
        let paths = ContigPaths::new(base_dir, contig);
        let n_samples_cohort = self.inner.n_samples();

        // Open the four sub-stream views per field up front.
        struct OpenField {
            key: String,
            is_format: bool,
            width: usize,
            views: [FieldView; 4], // indexed by FieldSub::all() order
        }
        let mut open: Vec<OpenField> = Vec::with_capacity(fields.len());
        for (category, name, dtype_str) in &fields {
            let dtype = StorageDtype::from_meta_str(dtype_str).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "field {name:?} has unresolved/unknown dtype {dtype_str:?}"
                ))
            })?;
            let width = dtype
                .width_bytes()
                .expect("from_meta_str rejects unresolved dtypes");
            let mut views = Vec::with_capacity(4);
            for sub in FieldSub::all() {
                views.push(
                    FieldView::open(&paths, category, name, sub, dtype, n_samples_cohort)
                        .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?,
                );
            }
            open.push(OpenField {
                key: format!("{category}/{name}"),
                is_format: category == "format",
                width,
                views: views.try_into().map_err(|_| ()).expect("exactly 4 subs"),
            });
        }

        let mut pos: Vec<i32> = Vec::new();
        let mut ilen: Vec<i32> = Vec::new();
        let mut allele: Vec<u8> = Vec::new();
        let mut str_off: Vec<i64> = vec![0];
        let mut off: Vec<i64> = vec![0];
        let mut fbytes: Vec<Vec<u8>> = vec![Vec::new(); open.len()];

        for r in 0..br.n_regions {
            for s in 0..br.n_samples {
                for p in 0..br.ploidy {
                    let (hc, srcs) = br.decode_hap_src(&self.inner, r, s, p);
                    for i in 0..hc.positions.len() {
                        pos.push(hc.positions[i] as i32);
                        ilen.push(hc.ilens[i]);
                        allele.extend_from_slice(&hc.alts[i]);
                        str_off.push(allele.len() as i64);

                        let src = srcs[i];
                        // FieldSub::all() order: VkSnp, VkIndel, DenseSnp, DenseIndel
                        let sub_ix = match (src.is_dense, src.is_indel) {
                            (false, false) => 0,
                            (false, true) => 1,
                            (true, false) => 2,
                            (true, true) => 3,
                        };
                        for (fi, f) in open.iter().enumerate() {
                            let view = &f.views[sub_ix];
                            // Dense FORMAT strides by the ORIGINAL cohort sample
                            // index. `s` here is already the cohort index because
                            // `overlap_batch_src` runs over the whole cohort.
                            let elem = if src.is_dense && f.is_format {
                                view.bytes_at(src.idx * n_samples_cohort + s)
                            } else {
                                view.bytes_at(src.idx)
                            };
                            debug_assert_eq!(elem.len(), f.width);
                            fbytes[fi].extend_from_slice(elem);
                        }
                    }
                    off.push(pos.len() as i64);
                }
            }
        }

        let d = PyDict::new(py);
        d.set_item("pos", i32_to_pyarray(py, &pos))?;
        d.set_item("ilen", i32_to_pyarray(py, &ilen))?;
        d.set_item("allele", u8_to_pyarray(py, &allele))?;
        d.set_item("str_off", PyArray1::from_slice(py, &str_off))?;
        d.set_item("off", PyArray1::from_slice(py, &off))?;
        d.set_item("n_regions", br.n_regions)?;
        d.set_item("n_samples", br.n_samples)?;
        d.set_item("ploidy", br.ploidy)?;
        for (fi, f) in open.iter().enumerate() {
            d.set_item(format!("field_{}", f.key), u8_to_pyarray(py, &fbytes[fi]))?;
            d.set_item(format!("field_itemsize_{}", f.key), f.width)?;
        }
        Ok(d)
    }
```

If `ContigReader::n_samples()` is not a public accessor, add one:

```rust
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
```

- [ ] **Step 5: Add the ordering-parity test**

Append to `#[cfg(test)] mod tests` in `src/query/gather.rs`:

```rust
    /// `decode_hap_src` must return records in EXACTLY the order `decode_hap`
    /// does — otherwise field values would be attached to the wrong variants.
    #[test]
    fn decode_hap_src_matches_decode_hap_order() {
        let (dir, n_samples, ploidy) = super::tests_support::tiny_store_with_snps_and_indels();
        let reader =
            ContigReader::open(dir.path().to_str().unwrap(), "chr1", n_samples, ploidy).unwrap();
        let regions = [(0u32, 1_000u32)];
        let plain = overlap_batch(&reader, &regions);
        let with_src = overlap_batch_src(&reader, &regions);
        for s in 0..n_samples {
            for p in 0..ploidy {
                let a = plain.decode_hap(&reader, 0, s, p);
                let (b, srcs) = with_src.decode_hap_src(&reader, 0, s, p);
                assert_eq!(a.positions, b.positions);
                assert_eq!(a.ilens, b.ilens);
                assert_eq!(a.alts, b.alts);
                assert_eq!(srcs.len(), b.positions.len());
            }
        }
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: ALL PASS, including the two new decode tests.

- [ ] **Step 7: Build the extension so Python can call it**

Run: `pixi run maturin develop --release`
Expected: build succeeds. (This is a long foreground build — do NOT background it.)

- [ ] **Step 8: Commit**

```bash
git add src/query/gather.rs src/query/reader.rs src/py_query_decode.rs
git commit -m "feat(svar2): decode_batch_fields — field bytes per decoded record"
```

---

## Task 9: Python — `SparseVar2(fields=…)` and `decode()`

**Files:**
- Modify: `python/genoray/_svar2.py`, `python/genoray/_svar2_decode.py`
- Test: `tests/test_svar2_fields_read.py` (extend)

**Interfaces:**
- Consumes: `_load_field_manifest`, `_resolve_read_fields`, `StoredField` (Task 7); `decode_batch_fields` (Task 8).
- Produces:
  - `SparseVar2.__init__(path, *, fields: Sequence[str] | None = None)`
  - `SparseVar2.available_fields -> dict[str, StoredField]`
  - `SparseVar2.with_fields(fields: Sequence[str]) -> SparseVar2`
  - `SparseVar2.decode(contig, regions)` — the `Ragged` gains one entry per selected field, sharing the variant-axis offsets with `pos`/`ilen`/`allele`.

- [ ] **Step 1: Write the failing e2e test**

Append to `tests/test_svar2_fields_read.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from genoray import SparseVar2
from genoray._svar2_fields import FormatField, InfoField


@pytest.fixture
def store_with_fields(tmp_path: Path) -> SparseVar2:
    """Convert the shared test VCF carrying one INFO and one FORMAT field."""
    vcf = Path(__file__).parent / "data" / "biallelic.vcf.gz"
    out = tmp_path / "fields.svar2"
    SparseVar2.from_vcf(
        out,
        vcf,
        no_reference=True,
        info_fields=[InfoField("AF", dtype="f32")],
        format_fields=[FormatField("DS", dtype="f32", default=0.0)],
    )
    return SparseVar2(out)


def test_available_fields_reports_canonical_keys(store_with_fields):
    avail = store_with_fields.available_fields
    assert set(avail) == {"AF", "DS"}
    assert avail["AF"].category == "info"
    assert avail["DS"].category == "format"


def test_decode_without_fields_is_unchanged(store_with_fields):
    rag = store_with_fields.decode("chr1", [(0, 1_000_000)])
    assert set(rag.fields) == {"pos", "ilen", "allele"}


def test_decode_with_fields_shares_offsets_and_preserves_dtype(store_with_fields):
    sv = store_with_fields.with_fields(["AF", "DS"])
    rag = sv.decode("chr1", [(0, 1_000_000)])
    assert set(rag.fields) == {"pos", "ilen", "allele", "AF", "DS"}

    pos = rag.fields["pos"]
    af = rag.fields["AF"]
    ds = rag.fields["DS"]

    # Stored dtype is preserved end to end — no widening.
    assert af.data.dtype == np.dtype("float32")
    assert ds.data.dtype == np.dtype("float32")

    # One field value per decoded record, on the SAME offsets object.
    assert af.data.shape == pos.data.shape
    assert ds.data.shape == pos.data.shape
    np.testing.assert_array_equal(af.offsets, pos.offsets)
    np.testing.assert_array_equal(ds.offsets, pos.offsets)


def test_decode_rejects_unknown_field(store_with_fields):
    with pytest.raises(ValueError, match="NOPE"):
        store_with_fields.with_fields(["NOPE"])
```

> The fixture uses `tests/data/biallelic.vcf.gz`. If that file has no `AF`/`DS`
> declared in its header, the conversion will raise "field not found in the VCF
> header" — in that case, generate a fixture that HAS them by extending
> `tests/data/gen_from_vcf.sh` / `tests/data/gen_svar.py` (which already build
> the test corpus), and use that path instead. Do not weaken the assertions.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar2_fields_read.py -v -k "decode or available"`
Expected: FAIL — `SparseVar2.__init__() got an unexpected keyword argument 'fields'`.

- [ ] **Step 3: Wire fields into `SparseVar2`**

In `python/genoray/_svar2.py`, replace `__init__` and add the two members:

```python
    def __init__(
        self, path: str | Path, *, fields: "Sequence[str] | None" = None
    ) -> None:
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        self.format_version: int = meta["format_version"]
        self.available_samples: list[str] = list(meta["samples"])
        self.contigs: list[str] = list(meta["contigs"])
        self.ploidy: int = meta["ploidy"]
        self.available_fields: dict[str, StoredField] = _load_field_manifest(meta)
        #: The fields this reader decodes. Empty unless opted into via
        #: ``fields=`` / :meth:`with_fields` — decoding a field costs extra I/O.
        self._fields: list[StoredField] = _resolve_read_fields(
            fields, self.available_fields
        )
        self._readers: dict[str, _core.PyContigReader] = {
            contig: _core.PyContigReader(
                str(self.path), contig, len(self.available_samples), self.ploidy
            )
            for contig in self.contigs
        }

    def with_fields(self, fields: "Sequence[str]") -> "SparseVar2":
        """A new reader over the same store that also decodes ``fields``.

        Keys are those of :attr:`available_fields`: the bare field name when it
        is unique across INFO/FORMAT, else bcftools-style ``INFO/DP`` /
        ``FORMAT/DP``.
        """
        return SparseVar2(self.path, fields=fields)
```

Update the import at the top of `_svar2.py`. It currently reads
`from genoray._svar2_fields import _resolve_fields` (the write-path helper —
keep it). Replace with:

```python
from genoray._svar2_fields import (
    StoredField,
    _load_field_manifest,
    _resolve_fields,
    _resolve_read_fields,
)
```

`Sequence` is already imported under `TYPE_CHECKING` in this module.

- [ ] **Step 4: Attach fields in `decode()`**

In `python/genoray/_svar2_decode.py`, replace `decode`:

```python
    # Provided by the concrete SparseVar2 host class.
    _fields: list[Any]
    path: Any

    def decode(self, contig: str, regions: Iterable[tuple[int, int]]) -> "Ragged":
        """Materialize overlapping variants for ``contig`` into a record ``Ragged``.

        Fields ``pos`` (i32), ``ilen`` (i32), ``allele`` (opaque-string ALT bytes),
        plus one entry per selected INFO/FORMAT field (see
        :meth:`SparseVar2.with_fields`) — every one sharing a single variant-axis
        offsets object, shape ``(R, S, P, None)``, the same layout as gvl's
        ``RaggedVariants``. Pure-deletion ALT is empty.

        Field values come back in the dtype they are STORED as (SVAR2
        losslessly auto-narrows integers), and VCF-missing entries carry the
        store's ``default`` or its reserved sentinel (``NaN`` for floats,
        ``iinfo.min``/``iinfo.max`` for ints) — neither is translated.
        """
        import numpy as np
        from seqpro.rag import Ragged

        reg = [(int(s), int(e)) for s, e in regions]
        reader = self._readers[contig]
        if not self._fields:
            d = reader.decode_batch(reg)
        else:
            d = reader.decode_batch_fields(
                reg,
                [(f.category, f.name, _META_DTYPE[f.dtype]) for f in self._fields],
                str(self.path),
                contig,
            )

        shape = (d["n_regions"], d["n_samples"], d["ploidy"], None)
        off = d["off"]
        pos = Ragged.from_offsets(d["pos"], shape, off)
        ilen = Ragged.from_offsets(d["ilen"], shape, off)
        allele = Ragged.from_offsets(
            d["allele"].view("S1"), shape, off, str_offsets=d["str_off"]
        )
        # If a consumer hits an error reading `.lengths` on a (2, N) offsets
        # layout, call `.to_packed()` first — a known seqpro slicing quirk.
        rec: dict[str, Ragged] = {"pos": pos, "ilen": ilen, "allele": allele}
        for f in self._fields:
            # Rust hands back raw little-endian bytes + an itemsize; the dtype is
            # applied here (same trick as `allele`), so Rust does no dtype
            # dispatch. `.view` is zero-copy.
            raw: np.ndarray = d[f"field_{f.category}/{f.name}"]
            itemsize = d[f"field_itemsize_{f.category}/{f.name}"]
            if itemsize != f.dtype.itemsize:
                # A width/dtype disagreement would silently reinterpret the
                # bytes into wrong values — fail loudly instead.
                raise ValueError(
                    f"field {f.key!r}: store wrote {itemsize}-byte elements but "
                    f"meta.json declares {f.dtype} ({f.dtype.itemsize} bytes)"
                )
            vals = raw.view(f.dtype)
            rec[f.key] = Ragged.from_offsets(vals, shape, off)
        return Ragged.from_fields(rec)
```

Add the reverse dtype map to `python/genoray/_svar2_fields.py`:

```python
#: numpy dtype -> the `meta.json` storage-dtype string the Rust side expects.
_META_DTYPE: dict[np.dtype, str] = {v: k for k, v in NP_DTYPE.items()}
```

…and import it in `_svar2_decode.py`:

```python
from genoray._svar2_fields import _META_DTYPE
```

- [ ] **Step 5: Rebuild and run the tests**

Run: `pixi run maturin develop --release`
Then: `pixi run pytest tests/test_svar2_fields_read.py -v`
Expected: ALL PASS.

- [ ] **Step 6: Run the full Python suite (no-regression gate)**

Run: `pixi run pytest -m "not network"`
Expected: ALL PASS — in particular the existing `tests/test_svar2_decode.py`,
which pins the no-fields `decode()` output.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_svar2.py python/genoray/_svar2_decode.py python/genoray/_svar2_fields.py tests/test_svar2_fields_read.py
git commit -m "feat(svar2): SparseVar2.decode(fields=...) returns fields on shared offsets"
```

---

## Task 10: Public-API docs + changelog

**Files:**
- Modify: `skills/genoray-api/SKILL.md`, `CHANGELOG.md`

This is **mandatory**, not optional — `CLAUDE.md` requires the same PR that changes
a public name to update the skill doc. New public names: `SparseVar2(fields=…)`,
`SparseVar2.available_fields`, `SparseVar2.with_fields`, `SparseVar2.decode`'s
field entries, and `StoredField`.

- [ ] **Step 1: Update the skill doc**

In `skills/genoray-api/SKILL.md`, in the `SparseVar2` section, add:

````markdown
### Reading INFO/FORMAT fields (SVAR2)

Fields written by `from_vcf(info_fields=…, format_fields=…)` are read back by
opting in. They are **not** decoded by default (each one costs extra I/O).

```python
sv = SparseVar2(path)
sv.available_fields          # {"AF": StoredField(...), "DS": StoredField(...)}

sv = sv.with_fields(["AF", "DS"])      # or SparseVar2(path, fields=["AF", "DS"])
rag = sv.decode("chr1", [(0, 10_000)])
rag.fields["DS"]             # Ragged, sharing offsets with pos/ilen/allele
```

**Field keys** are the bare field name when it is unique across INFO and FORMAT,
and bcftools-style `INFO/DP` / `FORMAT/DP` when a name appears in both. Use
whatever `available_fields` reports.

**Dtype is preserved as stored.** SVAR2 losslessly auto-narrows integer fields,
so an `AC` field may come back as `int8`. Nothing is widened.

**Missing values** are the field's `default` if one was set at write time, else a
reserved sentinel — `NaN` for floats, `iinfo.min`/`iinfo.max` for ints. They are
returned as-is, never translated.

**FORMAT caveat (inherited from the write path):** FORMAT values are
genotype-aligned, so values at non-carrier samples are not stored. `decode()`
only ever emits carrier records, so this is invisible on this surface.
````

- [ ] **Step 2: Update the changelog**

Under `## Unreleased` in `CHANGELOG.md`:

```markdown
### Added

- **SVAR2: read INFO/FORMAT fields.** `SparseVar2(fields=…)` /
  `SparseVar2.with_fields(…)` / `SparseVar2.available_fields`, and
  `SparseVar2.decode()` now returns each selected field as a `Ragged` sharing the
  variant-axis offsets with `pos`/`ilen`/`allele`. Values come back in the dtype
  they are stored as; missing entries keep the store's `default` or sentinel.
  Field keys are bare names, or bcftools-style `INFO/DP` / `FORMAT/DP` when a
  name is used by both categories.
- **Public Rust read API** for consumers that do their own channel merge (gvl):
  `query::FieldView` (mmap over a field's `values.bin`), `query::vk_src`
  provenance on `BatchResult`/`BatchResultSplit` via
  `query::gather_haps_readbound_src` / `overlap_batch_src`, plus
  `query::dense_abs_row`.
```

- [ ] **Step 3: Verify the docs match the code**

Run: `pixi run pytest tests/test_svar2_fields_read.py -v`
Expected: PASS (the snippets in the doc mirror these tests).

- [ ] **Step 4: Commit**

```bash
git add skills/genoray-api/SKILL.md CHANGELOG.md
git commit -m "docs(svar2): document the field-read API"
```

---

## Task 11: Final verification

- [ ] **Step 1: Full Rust suite**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: ALL PASS.

- [ ] **Step 2: Full Python suite**

Run: `pixi run test`
Expected: ALL PASS.

- [ ] **Step 3: Lint**

Run: `pixi run bash -lc 'cargo clippy --no-default-features -- -D warnings && cargo fmt --check'`
Then: `ruff check genoray tests && ruff format --check genoray tests`
Expected: clean.

- [ ] **Step 4: Confirm the zero-cost claim held**

The no-fields paths must be untouched. Confirm `tests/test_readbound_gather.rs`
and `tests/test_svar2_decode.py` pass **without modification** — if either needed
editing, the provenance work changed existing behaviour and must be revisited.

Run: `git diff main --stat -- tests/test_readbound_gather.rs tests/test_svar2_decode.py`
Expected: **no output** (neither file changed).

- [ ] **Step 5: Push**

```bash
git push
```

---

## Spec Coverage Check

| Spec section | Task |
| --- | --- |
| §1 `FieldView` (public, mmap, dtype from meta.json, `value_at`/`format_at`/`as_slice`) | 1, 2, 3, 6 |
| §2 `vk_src` packing + hard 2^31 assert | 4 |
| §3 One generic merge (`merge_by_position`), zero-cost when unused | 4, 5, 11 (step 4) |
| §4 Dtype preserved, never converted; sentinels untranslated | 3, 7, 9, 10 |
| §5 Python `decode(fields=…)`, shared offsets, canonical `INFO/x` keys | 7, 8, 9 |
| §6 No field arrays on the result structs; adding a field costs nothing | 5, 8 (per-field cost is one mmap + one `bytes_at` per record) |
| §7 gvl-side dependents | out of scope by design; noted in the spec, not this plan |
| Dense provenance is free (arithmetic) | 6 (`dense_abs_row`), 8 (`dense.src[j]` in `decode_hap_src`) |
| Testing strategy (dtype coverage, subset/orig-sample, assert, merge tie-break, oracle parity, e2e, no-regression) | 3, 4, 5, 8, 9, 11 |
| Docs (SKILL.md mandatory, CHANGELOG) | 10 |

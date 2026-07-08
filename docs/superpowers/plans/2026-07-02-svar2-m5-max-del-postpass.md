# SVAR 2.0 — M5 (part 2a): `max_del` standalone post-pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce, as a standalone post-pass over a finished SVAR2 contig directory, the per-`(sample, ploid)` and dense max-deletion-length artifacts (`max_del.npy`, `dense/max_del.npy`) that the `(range, sample)` overlap query will consume.

**Architecture:** A deletion's reference-base length is fully recoverable from the already-written indel key streams — a pure DEL packs its signed `ilen` inline (`bit 31 = 1`, `bits[31:1] = signed ilen`), so no LUT, reference genome, or conversion-pipeline coupling is needed. The pass reads `var_key/indel/{offsets.npy, alleles.bin}` and `dense/indel/alleles.bin`, decodes each key's deletion length via a decoder exposed from `rvk.rs` (the single source of the bit layout), takes per-column and dense maxima, and writes two `.npy` files. It runs after a contig's merge completes, callable per-contig.

**Tech Stack:** Rust 2024, `ndarray` + `ndarray-npy` (npy I/O, already used by `merge.rs`), `bytemuck` (byte↔u32 casting, already a dependency), `thiserror` (typed errors), `proptest` + `tempfile` (dev-only, already used by `search.rs`/`test_e2e.rs`).

## Global Constraints

- **Crate lib name is `genoray_core`** — integration tests under `tests/` import `genoray_core::...` (the package is `genoray`, but `[lib] name = "genoray_core"`).
- **Run Rust tests with `--no-default-features`** so the test binary links libpython instead of building the pyo3 extension module. Canonical command in this repo: `pixi run cargo test --no-default-features <filter>`. Unit tests (in-crate `#[cfg(test)]`) use `--lib`; integration tests (under `tests/`) use `--test <name>`.
- **Lint gate:** `pixi run -e lint cargo clippy --all-targets -- -D warnings` must pass (pre-commit hook). Run `pixi run -e lint cargo fmt --all` before committing. **prek git hooks must be installed** before committing/pushing (`prek install` if not already).
- **Column convention is sample-major:** the flat column index is `c = s * ploidy + p` (see `src/rvk.rs` `dense2sparse_vk`, `base_idx = (s * ploidy) + p`; `total_columns = num_samples * ploidy` in `src/merge.rs`). A flat `(total_columns,)` array reshapes **row-major (C-order)** to `(n_samples, ploidy)` so that `arr[[s, p]] == flat[s*ploidy + p]`.
- **On-disk key format:** `alleles.bin` in an indel stream dir is a raw little-endian `u32` array (4 bytes/key). `offsets.npy` is a 1-D `u64` array of length `total_columns + 1` (prefix sum). Read `alleles.bin` alignment-agnostically via `chunks_exact(4)` (a plain `std::fs::read` may hand back an unaligned buffer — the established pattern in `tests/common/mod.rs::read_u32_bin`).
- **Frozen contract decisions (shared with the `(range, sample)` query spec):**
  - `max_del.npy` — dtype `u32`, shape `(n_samples, ploidy)`, at `{contig_dir}/max_del.npy`. A pure-SNP / no-deletion contig still emits an all-zero array (consumer never special-cases a missing file).
  - `dense/max_del.npy` — dtype `u32`, shape **`(1,)`** (a length-1 `Array1`, chosen over 0-d for round-trip simplicity with `ndarray_npy::read_npy`), at `{contig_dir}/dense/max_del.npy`. **Always written** (creating the `dense/` dir if absent), value `0` when there is no dense indel stream — symmetric with `max_del.npy`, so the consumer never special-cases absence.
  - SNP sub-streams (`var_key/snp`, `dense/snp`) get **no** file — their overlap is a plain half-open range (`max_region_length == 0`).

---

## File Structure

- **`src/rvk.rs`** (modify) — expose `pub fn deletion_len(key: u32) -> u32`, the single decoder for a deletion's reference-base length from a 32-bit indel key. Keeps the bit layout co-located with `pack_variant`.
- **`src/layout.rs`** (modify) — add four contig-dir-relative path helpers used by the post-pass: `var_key_indel_dir`, `dense_indel_dir`, `max_del`, `dense_max_del`. Centralizes all paths the pass touches (layout.rs is the single source of on-disk paths).
- **`src/error.rs`** (modify) — add a `ReadNpy` variant to `ConversionError` (reading `offsets.npy` can fail; the existing `Npy` variant is write-only).
- **`src/max_del.rs`** (create) — the post-pass: `pub fn write_max_del(contig_dir, n_samples, ploidy) -> Result<(), ConversionError>` plus private `var_key_max_del` / `dense_max_del_scalar` / `read_keys` helpers. Owns all producer logic and its tests.
- **`src/lib.rs`** (modify) — `pub mod max_del;`.
- **`src/orchestrator.rs`** (modify) — invoke `write_max_del` per contig after the dense merge, so a normal conversion emits the artifacts end-to-end.
- **`tests/test_e2e.rs`** (modify) — a new end-to-end test that runs a conversion with known deletions (one rare → `var_key/indel`, one common → `dense/indel`), asserts both artifacts' shapes/values, and feeds the produced `max_del` into `overlap_range` to close the producer↔consumer loop.

---

## Task 1: `rvk::deletion_len` decoder

**Files:**
- Modify: `src/rvk.rs` (add `deletion_len` near `pack_variant` at `src/rvk.rs:242-261`; add tests in the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: the key layout defined by `pack_variant` (`src/rvk.rs:243`) — inline lane has `bit 0 == 0`; a pure DEL sets `bit 31` and stores signed `ilen` in `bits[31:1]`; long-INS lookup keys set `bit 0 == 1`.
- Produces: `pub fn deletion_len(key: u32) -> u32` — reference-base deletion length (`-ilen` for a pure DEL; `0` for SNP/INS inline keys and long-INS lookup keys). Consumed by Task 3's `max_del.rs`.

- [ ] **Step 1: Write the failing tests**

Add to the `mod tests` block in `src/rvk.rs` (alongside `test_pack_variant_pure_del` at line 620). These build keys via the real `pack_variant` encoder so the decoder is validated against the encoder, not against a re-derived layout:

```rust
    #[test]
    fn test_deletion_len_snp_and_ins_are_zero() {
        let mut bank = make_bank();
        // SNP (ilen 0) and INS (ilen > 0) carry no deletion.
        let snp = pack_variant(0, b"C", &mut bank);
        let ins = pack_variant(2, b"ACG", &mut bank);
        assert_eq!(deletion_len(snp), 0);
        assert_eq!(deletion_len(ins), 0);
    }

    #[test]
    fn test_deletion_len_small_and_large_del() {
        let mut bank = make_bank();
        // ref=ACGT alt=A → ilen=-3, deletion of 3 reference bases.
        let small = pack_variant(-3, b"A", &mut bank);
        assert_eq!(deletion_len(small), 3);
        // Largest atomizable deletion: ilen == MIN_I31 → d == 1 << 30.
        let large = pack_variant(crate::types::MIN_I31, b"A", &mut bank);
        assert_eq!(deletion_len(large), (1u32 << 30));
    }

    #[test]
    fn test_deletion_len_lookup_key_is_zero() {
        // A long-INS lookup key sets the LSB. Even with the top bit set
        // (a large row index), the LSB check must win → not a deletion.
        assert_eq!(deletion_len(0x0000_0003), 0); // LSB set
        assert_eq!(deletion_len(0xFFFF_FFFF), 0); // LSB set AND top bit set
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run cargo test --no-default-features --lib deletion_len`
Expected: FAIL — `cannot find function 'deletion_len' in this scope`.

- [ ] **Step 3: Implement `deletion_len`**

Insert immediately after `pack_variant` (after `src/rvk.rs:261`):

```rust
/// Reference-base deletion length encoded in a 32-bit indel `var_key`/`dense` key.
///
/// Inverse of the DEL lane of [`pack_variant`]: a pure DEL clears the lookup flag
/// (`bit 0 == 0`) and sets the top bit (`bit 31 == 1`), storing its signed `ilen`
/// in `bits[31:1]`; the deletion length is `-ilen`. Inline INS/SNP keys clear the
/// top bit, and long-INS lookup keys set `bit 0`, so both yield `0`. This is the
/// single decode site for the deletion length — the bit layout stays co-located
/// with the encoder (respecting the encoding-agnostic seam).
#[inline]
pub fn deletion_len(key: u32) -> u32 {
    // Long-INS lookup keys set the LSB; they are never deletions. Checked first
    // because such a key can also have its top bit set (a large row index).
    if key & 1 == 1 {
        return 0;
    }
    // Inline lane. Pure DEL sets the top bit; INS/SNP clear it.
    if key & (1 << 31) == 0 {
        return 0;
    }
    let ilen = (key as i32) >> 1; // arithmetic shift recovers the signed i31
    debug_assert!(ilen < 0, "top-bit-set inline key must be a negative-ilen DEL");
    ilen.unsigned_abs()
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run cargo test --no-default-features --lib deletion_len`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rvk.rs
git commit -m "feat(svar-2): expose rvk::deletion_len decoder for max_del post-pass"
```

---

## Task 2: `layout` path helpers for the post-pass

**Files:**
- Modify: `src/layout.rs` (add four free functions after the existing `offsets`/`genotypes` free functions at `src/layout.rs:71-79`; add tests in the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing new (mirrors the on-disk layout already defined by `ContigPaths` and `merge.rs`).
- Produces (all take `contig_dir: &Path` = `{out}/{contig}`):
  - `pub fn var_key_indel_dir(contig_dir: &Path) -> PathBuf` → `{contig_dir}/var_key/indel`
  - `pub fn dense_indel_dir(contig_dir: &Path) -> PathBuf` → `{contig_dir}/dense/indel`
  - `pub fn max_del(contig_dir: &Path) -> PathBuf` → `{contig_dir}/max_del.npy`
  - `pub fn dense_max_del(contig_dir: &Path) -> PathBuf` → `{contig_dir}/dense/max_del.npy`

  Consumed by Task 3 (`max_del.rs`) and Task 5 (`orchestrator.rs`).

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/layout.rs` (after `test_dense_dirs` at line 110):

```rust
    #[test]
    fn test_max_del_postpass_paths() {
        let c = Path::new("/out/chr1");
        assert_eq!(var_key_indel_dir(c), Path::new("/out/chr1/var_key/indel"));
        assert_eq!(dense_indel_dir(c), Path::new("/out/chr1/dense/indel"));
        assert_eq!(max_del(c), Path::new("/out/chr1/max_del.npy"));
        assert_eq!(dense_max_del(c), Path::new("/out/chr1/dense/max_del.npy"));
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run cargo test --no-default-features --lib test_max_del_postpass_paths`
Expected: FAIL — `cannot find function 'var_key_indel_dir' in this scope`.

- [ ] **Step 3: Implement the helpers**

Insert after the `genotypes` free function (after `src/layout.rs:79`), before the `#[cfg(test)]` block:

```rust
/// Contig-dir-relative path helpers for the standalone `max_del` post-pass. These
/// take the contig directory (`{out}/{contig}`) directly, unlike the `ContigPaths`
/// methods which build from `base_out_dir` + `chrom`. Keeping them here preserves
/// layout.rs as the single source of on-disk paths.
pub fn var_key_indel_dir(contig_dir: &Path) -> PathBuf {
    contig_dir.join("var_key").join("indel")
}
pub fn dense_indel_dir(contig_dir: &Path) -> PathBuf {
    contig_dir.join("dense").join("indel")
}
pub fn max_del(contig_dir: &Path) -> PathBuf {
    contig_dir.join("max_del.npy")
}
pub fn dense_max_del(contig_dir: &Path) -> PathBuf {
    contig_dir.join("dense").join("max_del.npy")
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run cargo test --no-default-features --lib test_max_del_postpass_paths`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/layout.rs
git commit -m "feat(svar-2): add max_del post-pass path helpers to layout"
```

---

## Task 3: `max_del` module — the post-pass producer

**Files:**
- Modify: `src/error.rs` (add `ReadNpy` variant)
- Create: `src/max_del.rs`
- Modify: `src/lib.rs` (add `pub mod max_del;`)

**Interfaces:**
- Consumes: `rvk::deletion_len` (Task 1); `layout::{var_key_indel_dir, dense_indel_dir, max_del, dense_max_del, offsets, alleles}` (Task 2 + existing `src/layout.rs:68-73`); `error::ConversionError` (`src/error.rs`).
- Produces: `pub fn write_max_del(contig_dir: &Path, n_samples: usize, ploidy: usize) -> Result<(), ConversionError>` — writes `{contig_dir}/max_del.npy` (shape `(n_samples, ploidy)`, `u32`) and `{contig_dir}/dense/max_del.npy` (shape `(1,)`, `u32`). Consumed by Task 4 (proptest) and Task 5 (orchestrator + e2e).

- [ ] **Step 1: Add the `ReadNpy` error variant**

In `src/error.rs`, add inside the `ConversionError` enum (after the `Npy` variant at `src/error.rs:15-20`):

```rust
    #[error("failed to read npy at {path}: {source}")]
    ReadNpy {
        path: String,
        #[source]
        source: ndarray_npy::ReadNpyError,
    },
```

- [ ] **Step 2: Wire the module**

In `src/lib.rs`, add the module declaration in alphabetical position (after `pub mod layout;` at line 12):

```rust
pub mod max_del;
```

- [ ] **Step 3: Write the failing integration tests**

Create `src/max_del.rs` with ONLY the test module first (the `write_max_del` body comes in Step 5). This compiles against the not-yet-written `write_max_del`, so it fails to build — that is the intended red state.

```rust
//! Standalone `max_del` post-pass (SVAR 2.0, M5 part 2a).
//!
//! Scans a finished contig's indel key streams and emits the max-deletion-length
//! artifacts consumed by the `(range, sample)` overlap query. A deletion's length
//! is recoverable from the inline pure-DEL key alone (see [`crate::rvk::deletion_len`]),
//! so this is a pure scan of `alleles.bin` — no LUT reads, no reference genome, no
//! coupling to the conversion/merge write path. Runs after a contig's merge
//! completes; callable per-contig or as a batch sweep over an existing directory.

use crate::error::ConversionError;
use crate::layout;
use crate::rvk::deletion_len;
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use std::path::Path;

#[cfg(test)]
mod tests {
    // `use super::*` already brings `Array1`, `Array2`, `Path`, `layout`,
    // `write_npy`, `deletion_len`, and `write_max_del` into scope.
    use super::*;
    use tempfile::tempdir;

    /// Pure-DEL key for a deletion of `d` reference bases (`ilen = -d`), matching
    /// the pure-DEL lane of `rvk::pack_variant`. `d == 0` yields a non-deletion
    /// inline key (deletion_len 0). The encoder↔decoder faithfulness itself is
    /// proven in `rvk`'s `deletion_len` tests; here we only exercise the scan.
    fn del_key(d: u32) -> u32 {
        if d == 0 {
            0 // inline lane, bit 31 clear → not a deletion
        } else {
            ((-(d as i32)) << 1) as u32
        }
    }

    /// Write a synthetic `var_key/indel` stream: `offsets.npy` (u64 prefix sum,
    /// len total_columns+1) and `alleles.bin` (raw le u32 keys).
    fn write_var_key_indel(contig_dir: &Path, offsets: &[u64], keys: &[u32]) {
        let dir = layout::var_key_indel_dir(contig_dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_npy(layout::offsets(&dir), &Array1::from_vec(offsets.to_vec())).unwrap();
        std::fs::write(layout::alleles(&dir), bytemuck::cast_slice(keys)).unwrap();
    }

    /// Write a synthetic `dense/indel` stream: just `alleles.bin` (raw le u32 keys).
    fn write_dense_indel(contig_dir: &Path, keys: &[u32]) {
        let dir = layout::dense_indel_dir(contig_dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(layout::alleles(&dir), bytemuck::cast_slice(keys)).unwrap();
    }

    fn read_max_del(contig_dir: &Path) -> Array2<u32> {
        ndarray_npy::read_npy(layout::max_del(contig_dir)).unwrap()
    }

    fn read_dense_max_del(contig_dir: &Path) -> Array1<u32> {
        ndarray_npy::read_npy(layout::dense_max_del(contig_dir)).unwrap()
    }

    #[test]
    fn absent_streams_emit_all_zero_artifacts() {
        // No var_key/indel, no dense/indel dirs at all. Contract still holds:
        // an all-zero (n_samples, ploidy) file and a [0] dense scalar.
        let tmp = tempdir().unwrap();
        let c = tmp.path();
        write_max_del(c, 2, 2).unwrap();
        assert_eq!(read_max_del(c), Array2::<u32>::zeros((2, 2)));
        assert_eq!(read_dense_max_del(c), Array1::from_vec(vec![0u32]));
    }

    #[test]
    fn per_column_max_over_var_key_indel() {
        // np = 4 columns (2 samples x ploidy 2), sample-major c = s*ploidy + p.
        //   col 0: dels {3, 1}      → 3
        //   col 1: {} (empty)       → 0
        //   col 2: SNP/INS only     → 0
        //   col 3: dels {2}         → 2
        let tmp = tempdir().unwrap();
        let c = tmp.path();
        let keys = vec![
            del_key(3), del_key(1), // col 0
            // col 1 empty
            del_key(0), del_key(0), // col 2: non-deletion inline keys
            del_key(2), // col 3
        ];
        let offsets = vec![0u64, 2, 2, 4, 5];
        write_var_key_indel(c, &offsets, &keys);

        write_max_del(c, 2, 2).unwrap();
        let m = read_max_del(c);
        // row-major reshape: [[col0, col1], [col2, col3]]
        assert_eq!(m[[0, 0]], 3);
        assert_eq!(m[[0, 1]], 0);
        assert_eq!(m[[1, 0]], 0);
        assert_eq!(m[[1, 1]], 2);
        // No dense stream → dense scalar is 0.
        assert_eq!(read_dense_max_del(c), Array1::from_vec(vec![0u32]));
    }

    #[test]
    fn dense_scalar_is_single_max_over_shared_keys() {
        let tmp = tempdir().unwrap();
        let c = tmp.path();
        // A pure-SNP var_key side (no deletions) + a dense indel table with dels.
        write_var_key_indel(c, &[0u64, 0, 0, 0, 0], &[]);
        write_dense_indel(c, &[del_key(5), del_key(2), del_key(0)]);

        write_max_del(c, 2, 2).unwrap();
        assert_eq!(read_max_del(c), Array2::<u32>::zeros((2, 2)));
        assert_eq!(read_dense_max_del(c), Array1::from_vec(vec![5u32]));
    }
}
```

- [ ] **Step 4: Run the tests to verify they fail (build error)**

Run: `pixi run cargo test --no-default-features --lib max_del::`
Expected: FAIL — `cannot find function 'write_max_del' in this scope` (function not yet defined).

- [ ] **Step 5: Implement the producer**

Insert the implementation into `src/max_del.rs` between the `use` block and the `#[cfg(test)] mod tests` block:

```rust
/// Emit `{contig_dir}/max_del.npy` (shape `(n_samples, ploidy)`, `u32`) and
/// `{contig_dir}/dense/max_del.npy` (shape `(1,)`, `u32`). Both are always written
/// — a contig with no deletions emits all-zero artifacts so the consumer never
/// special-cases a missing file. `O(total indel calls)`; a single serial pass
/// (measure before parallelizing — this is I/O-bound).
pub fn write_max_del(
    contig_dir: &Path,
    n_samples: usize,
    ploidy: usize,
) -> Result<(), ConversionError> {
    let var_key = var_key_max_del(contig_dir, n_samples, ploidy)?;
    let out = layout::max_del(contig_dir);
    write_npy(&out, &var_key).map_err(|source| ConversionError::Npy {
        path: out.display().to_string(),
        source,
    })?;

    let dense = dense_max_del_scalar(contig_dir)?;
    let dense_out = layout::dense_max_del(contig_dir);
    // dense/max_del.npy lives under {contig_dir}/dense, which may not exist for a
    // contig without any dense stream. Create it so the write (and the contract)
    // always succeeds.
    if let Some(parent) = dense_out.parent() {
        std::fs::create_dir_all(parent).map_err(|source| ConversionError::Io {
            context: format!("creating {}", parent.display()),
            source,
        })?;
    }
    let dense_arr = Array1::from_vec(vec![dense]); // shape (1,)
    write_npy(&dense_out, &dense_arr).map_err(|source| ConversionError::Npy {
        path: dense_out.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Per-column max deletion length over the `var_key/indel` stream, reshaped to
/// `(n_samples, ploidy)`. Absent stream ⇒ all-zero (pure-SNP / no-indel contig).
fn var_key_max_del(
    contig_dir: &Path,
    n_samples: usize,
    ploidy: usize,
) -> Result<Array2<u32>, ConversionError> {
    let total_columns = n_samples * ploidy;
    let indel_dir = layout::var_key_indel_dir(contig_dir);
    let offsets_path = layout::offsets(&indel_dir);

    // No offsets file ⇒ the stream was never written ⇒ zero output.
    if !offsets_path.exists() {
        return Ok(Array2::zeros((n_samples, ploidy)));
    }

    let offsets: Array1<u64> =
        ndarray_npy::read_npy(&offsets_path).map_err(|source| ConversionError::ReadNpy {
            path: offsets_path.display().to_string(),
            source,
        })?;
    debug_assert_eq!(
        offsets.len(),
        total_columns + 1,
        "offsets.npy length must be total_columns + 1"
    );

    let keys = read_keys(&layout::alleles(&indel_dir))?;

    let mut per_col = vec![0u32; total_columns];
    for (c, slot) in per_col.iter_mut().enumerate() {
        let lo = offsets[c] as usize;
        let hi = offsets[c + 1] as usize;
        *slot = keys[lo..hi].iter().copied().map(deletion_len).max().unwrap_or(0);
    }

    // Sample-major columns (c = s*ploidy + p) ⇒ row-major reshape to (n_samples, ploidy).
    Ok(Array2::from_shape_vec((n_samples, ploidy), per_col)
        .expect("per_col length == n_samples * ploidy"))
}

/// Single max deletion length over the shared `dense/indel` key table. Absent
/// stream ⇒ 0.
fn dense_max_del_scalar(contig_dir: &Path) -> Result<u32, ConversionError> {
    let keys = read_keys(&layout::alleles(&layout::dense_indel_dir(contig_dir)))?;
    Ok(keys.iter().copied().map(deletion_len).max().unwrap_or(0))
}

/// Read a raw little-endian `u32` key file into a `Vec<u32>`. A missing file is
/// treated as empty (an absent stream). Alignment-agnostic: `std::fs::read` may
/// return an unaligned buffer, so decode via `chunks_exact(4)` rather than
/// `bytemuck::cast_slice`.
fn read_keys(path: &Path) -> Result<Vec<u32>, ConversionError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(source) => Err(ConversionError::Io {
            context: format!("reading {}", path.display()),
            source,
        }),
    }
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pixi run cargo test --no-default-features --lib max_del::`
Expected: PASS (3 tests: `absent_streams_emit_all_zero_artifacts`, `per_column_max_over_var_key_indel`, `dense_scalar_is_single_max_over_shared_keys`).

- [ ] **Step 7: Lint**

Run: `pixi run -e lint cargo fmt --all && pixi run -e lint cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 8: Commit**

```bash
git add src/error.rs src/lib.rs src/max_del.rs
git commit -m "feat(svar-2): max_del post-pass producer (var_key per-column + dense scalar)"
```

---

## Task 4: Proptest oracle for `write_max_del`

**Files:**
- Modify: `src/max_del.rs` (add a `proptest!` block inside the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `write_max_del` (Task 3) and the test helpers `del_key` / `write_var_key_indel` / `read_max_del` already defined in Task 3's test module.
- Produces: nothing (test-only).

- [ ] **Step 1: Write the failing proptest**

Add inside `mod tests` in `src/max_del.rs` (after the three unit tests). It builds a random synthetic `var_key/indel` contig with known per-call deletion lengths across random columns, runs the pass, and asserts each `max_del[s, p]` equals the brute-force per-column max — the oracle style of `search.rs`:

```rust
    use proptest::prelude::*;

    proptest! {
        // Random per-column deletion lengths → the produced (n_samples, ploidy)
        // max_del must equal the brute-force per-column maximum. `0` marks a
        // non-deletion call (SNP/INS). This is the primary correctness gate for
        // the scan + column slicing + reshape.
        #[test]
        fn prop_var_key_max_del_matches_oracle(
            n_samples in 1usize..4,
            ploidy in 1usize..3,
            // per-column deletion lengths, flattened; sized/chunked below.
            col_lens in proptest::collection::vec(0usize..6, 0..12),
            del_seeds in proptest::collection::vec(0u32..1000, 0..64),
        ) {
            let total_columns = n_samples * ploidy;

            // Deterministic per-column call counts (0..=5) derived from col_lens.
            let counts: Vec<usize> =
                (0..total_columns).map(|c| col_lens.get(c).copied().unwrap_or(0)).collect();

            // Assign deletion lengths to calls from del_seeds (cycled), building
            // both the key stream and the per-column oracle max in lockstep.
            let mut keys: Vec<u32> = Vec::new();
            let mut offsets: Vec<u64> = vec![0u64; total_columns + 1];
            let mut oracle = vec![0u32; total_columns];
            let mut seed_i = 0usize;
            for c in 0..total_columns {
                let mut col_max = 0u32;
                for _ in 0..counts[c] {
                    let d = del_seeds.get(seed_i % del_seeds.len().max(1)).copied().unwrap_or(0);
                    seed_i += 1;
                    keys.push(del_key(d));
                    col_max = col_max.max(d);
                }
                offsets[c + 1] = offsets[c] + counts[c] as u64;
                oracle[c] = col_max;
            }

            let tmp = tempdir().unwrap();
            let cdir = tmp.path();
            write_var_key_indel(cdir, &offsets, &keys);
            write_max_del(cdir, n_samples, ploidy).unwrap();

            let m = read_max_del(cdir);
            prop_assert_eq!(m.shape(), &[n_samples, ploidy]);
            for c in 0..total_columns {
                let s = c / ploidy;
                let p = c % ploidy;
                prop_assert_eq!(m[[s, p]], oracle[c], "col {}", c);
            }
        }
    }
```

- [ ] **Step 2: Run the proptest to verify it is exercised**

Run: `pixi run cargo test --no-default-features --lib max_del::tests::prop_var_key_max_del_matches_oracle`
Expected: PASS (the implementation from Task 3 already satisfies it; if it fails, the Task 3 scan/reshape has a bug — fix there). If `del_seeds` is empty the `% .max(1)` guard keeps indexing safe; those calls get `d = 0`.

- [ ] **Step 3: Commit**

```bash
git add src/max_del.rs
git commit -m "test(svar-2): brute-force oracle proptest for max_del per-column producer"
```

---

## Task 5: Orchestrator wiring + end-to-end round-trip

**Files:**
- Modify: `src/orchestrator.rs` (call `write_max_del` after the dense merge loop at `src/orchestrator.rs:252`)
- Modify: `tests/test_e2e.rs` (add a new e2e test)

**Interfaces:**
- Consumes: `write_max_del` (Task 3); `process_chromosome` (`src/orchestrator.rs:44`); `layout::{max_del, dense_max_del}` (Task 2); `search::{SearchTree, overlap_range}` (`src/search.rs:63,148`); test helpers `SynthRecord`, `build_bcf_with_index`, `read_u32_bin`, `read_offsets_npy` (`tests/common/mod.rs`).
- Produces: a normal conversion now emits `max_del.npy` + `dense/max_del.npy` per contig.

- [ ] **Step 1: Wire `write_max_del` into the orchestrator**

In `src/orchestrator.rs`, insert after the dense-merge `for` loop (after `src/orchestrator.rs:252`, before the `println!("[{}] Pipeline Execution Finished..."` line):

```rust
    // M5 post-pass: emit max-deletion-length artifacts for the overlap query.
    // A pure scan of the finished indel key streams — decoupled from the merge.
    let contig_dir = std::path::Path::new(base_out_dir).join(chrom);
    crate::max_del::write_max_del(&contig_dir, samples.len(), ploidy)?;
```

- [ ] **Step 2: Write the failing e2e test**

Add to `tests/test_e2e.rs` (after `test_e2e_normalized_bcf_pipeline`, and add the search import at the top of the file). At the top, alongside the existing `use genoray_core::...` lines (near `tests/test_e2e.rs:11-13`), add:

```rust
use genoray_core::search::{overlap_range, SearchTree};
```

Then append the test:

```rust
// M5 max_del post-pass, end-to-end. Two deletions with known lengths:
//   pos=100  ATTT → A  (d=3)  gt=[1,0,0,0]  x=1 → rare → var_key/indel (hap 0)
//   pos=200  ATT  → A  (d=2)  gt=[1,1,1,0]  x=3 → common → dense/indel (shared)
// Asserts the produced artifacts, then feeds max_del into overlap_range to prove
// a deletion that starts left of the query but spans into it is recoverable.
#[test]
fn test_e2e_max_del_postpass() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("maxdel.bcf");
    let samples = vec!["S0", "S1"]; // np = 4

    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"ATTT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 0, 0, 0], // hap 0 only → var_key/indel
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"ATT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 1, 0], // haps 0,1,2 → dense/indel
        },
    ];
    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    process_chromosome(
        bcf_path.to_str().unwrap(),
        "chr1",
        out_dir.to_str().unwrap(),
        &samples,
        100,
        2,
        1,
        4096,
    )
    .expect("conversion");

    let contig_dir = out_dir.join("chr1");

    // var_key max_del: shape (2, 2); only col 0 (S0_p0) carries the d=3 deletion.
    let m: ndarray::Array2<u32> =
        ndarray_npy::read_npy(contig_dir.join("max_del.npy")).unwrap();
    assert_eq!(m.shape(), &[2, 2]);
    assert_eq!(m[[0, 0]], 3);
    assert_eq!(m[[0, 1]], 0);
    assert_eq!(m[[1, 0]], 0);
    assert_eq!(m[[1, 1]], 0);

    // dense max_del: shape (1,); single max over the shared indel table = 2.
    let dm: ndarray::Array1<u32> =
        ndarray_npy::read_npy(contig_dir.join("dense/max_del.npy")).unwrap();
    assert_eq!(dm, ndarray::Array1::from_vec(vec![2u32]));

    // Close the producer -> consumer loop: the var_key/indel deletion for hap 0
    // starts at pos 100 and spans 3 bases (v_end = 100 + 1 + 3 = 104). A query
    // strictly right of the start but inside the deletion must still find it,
    // using the produced max_del as max_region_length.
    let vk_indel = contig_dir.join("var_key/indel");
    let off = read_offsets_npy(&vk_indel.join("offsets.npy"));
    let pos = read_u32_bin(&vk_indel.join("positions.bin"));
    // hap 0 owns exactly the [off[0], off[1]) slice = one call at pos 100.
    let col0 = &pos[off[0] as usize..off[1] as usize];
    assert_eq!(col0, &[100u32]);

    let v_starts: Vec<u32> = col0.to_vec();
    let v_ends: Vec<u32> = vec![100 + 1 + 3]; // start + 1 + d, d = max_del[0,0]
    let tree = SearchTree::new(&v_starts);
    let max_region_length = m[[0, 0]];
    // query [102, 103): starts right of variant start 100 but inside the deletion.
    assert_eq!(overlap_range(&tree, &v_ends, max_region_length, 102, 103), (0, 1));
}
```

- [ ] **Step 3: Run the e2e test to verify it fails (before wiring is proven)**

If Step 1 is already applied, this passes; to see the red state first, the test build itself proves the wiring — run:

Run: `pixi run cargo test --no-default-features --test test_e2e test_e2e_max_del_postpass`
Expected: PASS (Step 1 wiring produces the files; the assertions confirm shapes/values and the overlap loop). If it fails with a missing-file error, Step 1's orchestrator wiring was not applied — apply it.

- [ ] **Step 4: Run the full test suite to confirm no regressions**

Run: `pixi run cargo test --no-default-features`
Expected: PASS — all existing lib + integration tests plus the new ones. (`test_e2e_normalized_bcf_pipeline`, `test_e2e_mutation_conservation`, etc. are unaffected — the post-pass only adds files.)

- [ ] **Step 5: Lint**

Run: `pixi run -e lint cargo fmt --all && pixi run -e lint cargo clippy --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator.rs tests/test_e2e.rs
git commit -m "feat(svar-2): wire max_del post-pass into orchestrator + e2e round-trip"
```

---

## Notes on deferred scope (from the design)

- **Consuming `max_del.npy` in the disk query** — the `(range, sample)` spec. Task 5's e2e demonstrates the loop with `overlap_range` directly, but the full on-disk `(range, sample)` gather is out of scope here.
- **No change to the conversion/merge write path** — this is strictly a post-pass; it does not piggyback on the merge, keeping it decoupled from the M2b left-alignment spec.
- **`pointer/indel` (M11)** — no such stream exists yet; when it lands, its indel keys join the same per-`(sample, ploid)` max (extend `var_key_max_del` to fold that stream in).
- **Batch-sweep entrypoint** — `write_max_del` is already usable standalone against any finished contig directory (a sweep is just a loop over contig dirs); a CLI/py entrypoint can be added later if needed.

## Frozen-contract sync reminder

The `(range, sample)` query spec MUST match these choices (kept in sync per the design's open questions):
- `dense/max_del.npy` shape is **`(1,)`** `u32` (not 0-d), always present, `0` when no dense indels.
- `max_del.npy` shape is `(n_samples, ploidy)` `u32`, row-major (`arr[[s, p]] == flat[s*ploidy + p]`), always present, all-zero for a no-deletion contig.
- SNP sub-streams get no file.

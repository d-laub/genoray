# SVAR 2.0 — M5 (part 2b): `(range, sample)` query — disk integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the pure `search.rs` overlap core to a finished SVAR2 contig on disk so that, given a contig, a `[q_start, q_end)` region, and a sample, we return that sample's variants (both haplotypes) overlapping the region, in position order.

**Architecture:** A new `src/query.rs` module. `ContigReader` mmaps a contig's sidecars (`positions.bin`, `alleles.bin`, `offsets.npy`, `genotypes.bin`, `max_del.npy`, `dense/max_del.npy`) and the long-allele LUT. `overlap_sample` runs `search::overlap_range` over each of ≤4 already-sorted sub-streams (`var_key/{snp,indel}` per-column, `dense/{snp,indel}` shared + genotype-filtered), decodes each hit through the `rvk` seam, and k-way merges the runs into one position-sorted `QueryResult` per haplotype. `search.rs` is not touched.

**Tech Stack:** Rust (edition 2024, lib crate `genoray_core`), `memmap2`, `ndarray` + `ndarray-npy`, `bytemuck`, `proptest` (dev), `rust-htslib` (test fixtures via `process_chromosome`).

## Global Constraints

- **Do not modify `src/search.rs`.** Its proptests are the M5 part-1 gate; treat `SearchTree` and `overlap_range` as a frozen dependency.
- **All key/deletion bit math goes through `src/rvk.rs`** (the "encoding seam"). No inline `>> 1` / `& 1` / `>> 27` decoding outside `rvk.rs`. `v_end` is always `pos + 1 + rvk::deletion_len(key)`.
- **Frozen `max_del.npy` contract** (shared with the parallel max_del post-pass spec — reproduce exactly): per-contig `{contig}/max_del.npy` is dtype `u32`, shape `(n_samples, ploidy)`; `max_del[s, p]` = max deletion length over the `var_key/indel` column for flat column `s*ploidy + p`, `0` if none; a pure-SNP contig still emits an all-zero array. `{contig}/dense/max_del.npy` is dtype `u32`, shape `(1,)` — the single max over the shared `dense/indel` table, `0` if none. SNP sub-streams get **no** file (`max_region_length == 0`). **This spec fixes the open `dense/max_del` shape question at `(1,)`.**
- **Flat column ↔ `(sample, ploid)` mapping is `col = sample * ploidy + p`** (sample-major, matches `merge.rs` and `dense2sparse_vk`). The dense genotype matrix is hap-major: bit `(hap, dense_col)` at flat index `hap * n_dense_variants + dense_col`, `hap == col`.
- **Test command:** `pixi run -e sbox cargo test --no-default-features [filter]`. The `--no-default-features` flag drops `pyo3/extension-module` so the Rust test binaries link libpython (see `Cargo.toml`). Plain `cargo test` fails to link.
- **Pre-commit hooks** (`cargo fmt`, `cargo clippy -D warnings`, `commitizen`) run on commit. Run `pixi run prek-install` once before the first commit if hooks are not yet installed. Commit messages must be Conventional Commits (`feat:`, `test:`, `refactor:`) — commitizen enforces this.
- **This spec is a consumer.** It develops against a hand-written `max_del.npy` fixture until the post-pass producer lands. `src/rvk.rs::deletion_len` and `src/layout.rs::{max_del,dense_max_del}` may also be added by that parallel spec — **if they already exist when you reach the relevant task, reuse them and skip the addition** (the bodies below are the canonical form; they will merge identically).
- **Result shape is provisional (finalized in M6):** `QueryResult` is struct-of-arrays (columnar) per haplotype. Keep it a plain owned Rust struct — no PyO3 types.

---

## File Structure

- **New:** `src/query.rs` — `ContigReader`, `overlap_sample`, `QueryResult`, `HapCalls`, `Call`, the mmap loaders, `gather_run`, `kway_merge`, hit decoders. In-source `#[cfg(test)] mod tests` for the pure-slice logic (mmap, `gather_run`, `kway_merge`).
- **New:** `tests/test_query.rs` — disk-integration tests (`ContigReader::open`, `overlap_sample` known-VCF cases, oracle proptest, dense/var_key cross-check, degenerate cases) plus the `max_del` fixture builder. Uses the existing `tests/common/mod.rs` harness (`build_bcf_with_index`, `SynthRecord`).
- **Modify:** `src/rvk.rs` — add `DecodedKey`, `decode_key`, `deletion_len`, `decode_snp_2bit`, `unpack_snp_key_at` (the query-side decode seam).
- **Modify:** `src/layout.rs` — add `max_del(contig_dir)` and `dense_max_del(contig_dir)` path helpers.
- **Modify:** `src/lib.rs` — register `pub mod query;`.

Task order builds bottom-up: decode seam → paths → module scaffold + mmap → pure query logic → reader → end-to-end query → proptest → degenerate/cross-check.

---

### Task 1: `rvk` query-side decode seam

**Files:**
- Modify: `src/rvk.rs` (add public items after `decode_alt_inline`, ~`src/rvk.rs:227`, inside the non-test region; add tests to the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: existing `rvk::decode_alt_inline(payload: u32) -> Vec<u8>`, `rvk::encode_snp_2bit(base: u8) -> u8`, `rvk::pack_variant(ilen: i32, alt: &[u8], bank: &mut LongAlleleTableWriter) -> u32`, `rvk::pack_snp_keys(codes: &[u8]) -> Vec<u8>`, the test helper `make_bank()` (already in `rvk`'s test module), constants `MIN_I31`, `MAX_INLINE_ALT_LEN`.
- Produces:
  - `pub enum DecodedKey { Inline { alt: Vec<u8> }, PureDel { ilen: i32 }, Lookup { row: u32 } }`
  - `pub fn decode_key(key: u32) -> DecodedKey`
  - `pub fn deletion_len(key: u32) -> u32`
  - `pub fn decode_snp_2bit(code: u8) -> u8`
  - `pub fn unpack_snp_key_at(packed: &[u8], i: usize) -> u8`

- [ ] **Step 1: Write the failing tests**

Add to the existing `#[cfg(test)] mod tests` in `src/rvk.rs` (it already has `use super::*;` and `make_bank()`):

```rust
    #[test]
    fn test_decode_key_roundtrips_pack_variant() {
        let mut bank = make_bank();
        // SNP A->C: ilen 0, alt "C" inline.
        let snp = pack_variant(0, b"C", &mut bank);
        assert_eq!(decode_key(snp), DecodedKey::Inline { alt: b"C".to_vec() });
        assert_eq!(deletion_len(snp), 0);
        // INS A->AT: ilen 1, alt "AT" inline.
        let ins = pack_variant(1, b"AT", &mut bank);
        assert_eq!(decode_key(ins), DecodedKey::Inline { alt: b"AT".to_vec() });
        assert_eq!(deletion_len(ins), 0);
        // Pure DEL of 2 bases: ilen -2, alt "A".
        let del = pack_variant(-2, b"A", &mut bank);
        assert_eq!(decode_key(del), DecodedKey::PureDel { ilen: -2 });
        assert_eq!(deletion_len(del), 2);
        // Long INS (> 13 bases) spills to the bank -> Lookup, deletion_len 0.
        let long = pack_variant(19, b"ACGTACGTACGTACGTACGT", &mut bank);
        assert!(matches!(decode_key(long), DecodedKey::Lookup { .. }));
        assert_eq!(deletion_len(long), 0);
    }

    #[test]
    fn test_deletion_len_near_min_i31() {
        let mut bank = make_bank();
        let big = pack_variant(MIN_I31, b"A", &mut bank);
        assert_eq!(deletion_len(big), (-(MIN_I31 as i64)) as u32);
    }

    #[test]
    fn test_decode_snp_2bit_inverts_encode() {
        for &b in &[b'A', b'C', b'T', b'G'] {
            assert_eq!(decode_snp_2bit(encode_snp_2bit(b)), b);
        }
    }

    #[test]
    fn test_unpack_snp_key_at_matches_unpack_all() {
        let codes = [1u8, 2, 3, 0, 2, 1, 3];
        let packed = pack_snp_keys(&codes);
        let all = unpack_snp_keys(&packed, codes.len());
        for i in 0..codes.len() {
            assert_eq!(unpack_snp_key_at(&packed, i), all[i], "code {}", i);
            assert_eq!(unpack_snp_key_at(&packed, i), codes[i]);
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e sbox cargo test --no-default-features rvk::tests::test_decode_key_roundtrips_pack_variant`
Expected: FAIL — compile error, `cannot find function `decode_key` in this scope` (and the other new names).

- [ ] **Step 3: Write the implementation**

Insert into the non-test region of `src/rvk.rs`, immediately after `decode_alt_inline` (after `src/rvk.rs:227`):

```rust
/// Discriminated form of a 32-bit indel key. Mirrors the `pack_variant` layout:
/// bit 0 = lookup flag, bit 31 (of a non-lookup key) = pure-DEL flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedKey {
    /// Inline INS/SNP; `alt` is the decoded ALT bases (`alt.len() == ilen + 1`).
    Inline { alt: Vec<u8> },
    /// Pure deletion of `-ilen` reference bases (`ilen < 0`). The ALT (anchor)
    /// base is not stored in the key; recover it from the reference downstream.
    PureDel { ilen: i32 },
    /// Long insertion spilled to the long-allele bank at row `row`.
    Lookup { row: u32 },
}

/// Decode a packed 32-bit indel key into its discriminated form. Single entry
/// point for the query path — no caller re-derives the bit layout.
pub fn decode_key(key: u32) -> DecodedKey {
    if key & 1 == 1 {
        DecodedKey::Lookup { row: key >> 1 }
    } else if (key as i32) < 0 {
        DecodedKey::PureDel {
            ilen: (key as i32) >> 1,
        }
    } else {
        DecodedKey::Inline {
            alt: decode_alt_inline(key),
        }
    }
}

/// Deletion length (in reference bases) encoded by an indel `key`, or `0` for an
/// insertion / SNP / lookup key. Used to reconstruct exclusive variant ends
/// (`v_end = pos + 1 + deletion_len(key)`) for `overlap_range`.
pub fn deletion_len(key: u32) -> u32 {
    if key & 1 == 0 && (key as i32) < 0 {
        // Pure DEL: ilen = (key as i32) >> 1 (negative); length = -ilen.
        (-((key as i32) >> 1)) as u32
    } else {
        0
    }
}

/// Recover the ALT base for a 2-bit SNP code (`A=00 C=01 T=10 G=11`). Inverse of
/// [`encode_snp_2bit`].
#[inline]
pub fn decode_snp_2bit(code: u8) -> u8 {
    const BASES: [u8; 4] = [b'A', b'C', b'T', b'G'];
    BASES[(code & 3) as usize]
}

/// Read the 2-bit SNP code at call index `i` from a 2-bit-packed key buffer
/// (4 codes/byte; see [`pack_snp_keys`]) without materializing the whole array.
#[inline]
pub fn unpack_snp_key_at(packed: &[u8], i: usize) -> u8 {
    (packed[i >> 2] >> ((i & 3) * 2)) & 3
}
```

Note: if the parallel max_del post-pass spec already added `deletion_len` to `rvk.rs`, keep its version and add only the other four items.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e sbox cargo test --no-default-features rvk::tests::test_decode`
Then: `pixi run -e sbox cargo test --no-default-features rvk::tests::test_unpack_snp_key_at_matches_unpack_all rvk::tests::test_deletion_len_near_min_i31`
Expected: PASS (4 new tests).

- [ ] **Step 5: Commit**

```bash
git add src/rvk.rs
git commit -m "feat: rvk query-side decode seam (decode_key, deletion_len, snp helpers)"
```

---

### Task 2: `layout` `max_del` path helpers

**Files:**
- Modify: `src/layout.rs` (add two free functions after `genotypes`, ~`src/layout.rs:79`; add a test to the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `std::path::{Path, PathBuf}` (already imported at `src/layout.rs:5`).
- Produces:
  - `pub fn max_del(contig_dir: &Path) -> PathBuf` → `{contig_dir}/max_del.npy`
  - `pub fn dense_max_del(contig_dir: &Path) -> PathBuf` → `{contig_dir}/dense/max_del.npy`

- [ ] **Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` in `src/layout.rs`:

```rust
    #[test]
    fn test_max_del_paths() {
        let contig = Path::new("/out/chr1");
        assert_eq!(max_del(contig), Path::new("/out/chr1/max_del.npy"));
        assert_eq!(
            dense_max_del(contig),
            Path::new("/out/chr1/dense/max_del.npy")
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e sbox cargo test --no-default-features layout::tests::test_max_del_paths`
Expected: FAIL — `cannot find function `max_del` in this scope`.

- [ ] **Step 3: Write the implementation**

Insert into the non-test region of `src/layout.rs`, after the `genotypes` free function (after `src/layout.rs:79`):

```rust
/// Per-contig `max_del.npy` (var_key/indel per-`(sample, ploid)` max deletion
/// length; `u32`, shape `(n_samples, ploidy)`). See the M5 `max_del` contract.
pub fn max_del(contig_dir: &Path) -> PathBuf {
    contig_dir.join("max_del.npy")
}

/// Per-contig `dense/max_del.npy` (single scalar max deletion length over the
/// shared dense/indel table; `u32`, shape `(1,)`).
pub fn dense_max_del(contig_dir: &Path) -> PathBuf {
    contig_dir.join("dense").join("max_del.npy")
}
```

Note: if the parallel max_del post-pass spec already added these, reuse them and skip this step (the bodies are identical).

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e sbox cargo test --no-default-features layout::tests::test_max_del_paths`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/layout.rs
git commit -m "feat: layout path helpers for max_del.npy and dense/max_del.npy"
```

---

### Task 3: `query` module scaffold + mmap loaders

**Files:**
- Create: `src/query.rs`
- Modify: `src/lib.rs` (register the module)
- Test: in-source `#[cfg(test)] mod tests` in `src/query.rs`

**Interfaces:**
- Consumes: `memmap2::Mmap`, `bytemuck::cast_slice`, `std::fs::File`, `tempfile` (dev).
- Produces (private module internals, used by later tasks):
  - `fn mmap_file(path: &Path) -> std::io::Result<Option<Mmap>>` — `None` for a missing or zero-length file.
  - `fn as_u32(m: &Option<Mmap>) -> &[u32]` — LE `u32` view, empty slice when `None`.
  - `fn as_bytes(m: &Option<Mmap>) -> &[u8]` — raw byte view, empty slice when `None`.

- [ ] **Step 1: Register the module**

In `src/lib.rs`, add `pub mod query;` in alphabetical position between `pub mod orchestrator;` (`src/lib.rs:18`) and `pub mod rvk;` (`src/lib.rs:19`):

```rust
pub mod orchestrator;
pub mod query;
pub mod rvk;
```

- [ ] **Step 2: Write the failing tests**

Create `src/query.rs` with the module header, imports, and a test module (implementation stubs come next):

```rust
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
```

The module-level `#![allow(dead_code)]` (included in the block above) keeps clippy green while the query internals are test-only; Task 6 removes it once `overlap_sample` consumes them all.

- [ ] **Step 3: Run tests to verify they pass** (this task's implementation *is* the loaders — the test is written against the real code)

Run: `pixi run -e sbox cargo test --no-default-features query::tests`
Expected: PASS (2 tests).

- [ ] **Step 4: Verify the whole suite still builds**

Run: `pixi run -e sbox cargo test --no-default-features --lib`
Expected: PASS (108 lib tests: 106 baseline + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/query.rs
git commit -m "feat: query module scaffold with mmap sidecar loaders"
```

---

### Task 4: `Call`, `gather_run`, and `kway_merge` (pure query logic)

**Files:**
- Modify: `src/query.rs` (add types + functions before the `#[cfg(test)] mod tests`; add tests inside it)

**Interfaces:**
- Consumes: `search::{SearchTree, overlap_range}`.
- Produces:
  - `pub struct Call { pub position: u32, pub ilen: i32, pub alt: Vec<u8> }`
  - `fn gather_run(positions: &[u32], max_region_length: u32, q_start: u32, q_end: u32, del_len: impl Fn(usize) -> u32, carried: impl Fn(usize) -> bool, decode_hit: impl Fn(usize) -> (i32, Vec<u8>), out: &mut Vec<Call>)`
  - `fn kway_merge(runs: Vec<Vec<Call>>) -> Vec<Call>`

- [ ] **Step 1: Write the failing tests**

Add to `src/query.rs`'s `#[cfg(test)] mod tests`:

```rust
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
            vec![call(50, 0, b"A")], // run 0
            vec![call(50, 1, b"AT")], // run 1, same position
        ];
        let merged = kway_merge(runs);
        assert_eq!(merged, vec![call(50, 0, b"A"), call(50, 1, b"AT")]);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e sbox cargo test --no-default-features query::tests::test_gather_run_snp_half_open`
Expected: FAIL — `cannot find function `gather_run``.

- [ ] **Step 3: Write the implementation**

In `src/query.rs`, add the `search` import to the top `use` block:

```rust
use crate::search::{SearchTree, overlap_range};
```

Then, before the `#[cfg(test)] mod tests`, add:

```rust
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
    for i in s_idx..e_idx {
        if carried(i) {
            let (ilen, alt) = decode_hit(i);
            out.push(Call {
                position: positions[i],
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e sbox cargo test --no-default-features query::tests`
Expected: PASS (8 tests: 2 from Task 3 + 6 new).

- [ ] **Step 5: Commit**

```bash
git add src/query.rs
git commit -m "feat: gather_run and kway_merge query primitives"
```

---

### Task 5: `ContigReader::open` — load sidecars

**Files:**
- Modify: `src/query.rs` (add `SubStreamView`, `DenseView`, `ContigReader`, loaders)
- Create: `tests/test_query.rs` (fixture builder + open test)

**Interfaces:**
- Consumes: `layout::{ContigPaths, positions, alleles, offsets, genotypes, max_del, dense_max_del}`, `nrvk::LongAlleleReader`, `bits::get_bit`, `ndarray::{Array1, Array2}`, `ndarray_npy::read_npy`. From `tests/common/mod.rs`: `SynthRecord`, `build_bcf_with_index`. From the crate: `process_chromosome`, `query::ContigReader`.
- Produces:
  - `pub struct ContigReader { /* private */ }`
  - `pub fn ContigReader::open(base_out_dir: &str, chrom: &str, n_samples: usize, ploidy: usize) -> std::io::Result<ContigReader>`
  - Private `SubStreamView` (fields `positions: Option<Mmap>`, `keys: Option<Mmap>`, `offsets: Vec<u64>`) with `positions(&self) -> &[u32]` and `column(&self, c: usize) -> (usize, usize)`.
  - Private `DenseView` (fields `positions/keys/genotypes: Option<Mmap>`, `n_dense_variants: usize`) with `positions(&self) -> &[u32]` and `carried(&self, hap: usize, col: usize) -> bool`.
  - Test-only fixture builder (in `tests/test_query.rs`): `build_contig(dir, chrom, samples, ploidy, records)`, `write_max_del_fixture(contig_dir, n_samples, ploidy, records)`, `del_len_of(record) -> u32`.

- [ ] **Step 1: Write the failing test (fixture builder + open)**

Create `tests/test_query.rs`:

```rust
//! Disk-integration tests for the `(range, sample)` query. Builds finished SVAR2
//! contigs via the real conversion pipeline and a hand-written `max_del.npy`
//! fixture (until the max_del post-pass producer lands).

mod common;

use common::{SynthRecord, build_bcf_with_index};
use genoray_core::process_chromosome;
use genoray_core::query::ContigReader;
use ndarray::{Array1, Array2};
use std::path::Path;
use tempfile::tempdir;

/// Deletion length implied by a record's ref/alt lengths (`max(0, -ilen)`).
fn del_len_of(rec: &SynthRecord) -> u32 {
    let ilen = rec.alts[0].len() as i32 - rec.ref_allele.len() as i32;
    if ilen < 0 { (-ilen) as u32 } else { 0 }
}

/// Write the `max_del` sidecars for a finished contig. Conservative per-column
/// bound: each `(sample, ploid)` column's max over ALL deletions it carries (an
/// over-estimate vs. the var_key-only contract, but `overlap_range`'s overshoot
/// is proven safe — see `search.rs` `overlap_max_region_length_overshoot_is_safe`),
/// and the global max for `dense/max_del`. This exercises per-column indexing in
/// the consumer while remaining independent of the not-yet-landed producer.
fn write_max_del_fixture(
    contig_dir: &Path,
    n_samples: usize,
    ploidy: usize,
    records: &[SynthRecord],
) {
    let columns = n_samples * ploidy;
    let mut per_col = vec![0u32; columns];
    let mut global = 0u32;
    for rec in records {
        let d = del_len_of(rec);
        global = global.max(d);
        for (hap, &g) in rec.gt.iter().enumerate() {
            if g == 1 {
                per_col[hap] = per_col[hap].max(d);
            }
        }
    }
    let arr = Array2::from_shape_vec((n_samples, ploidy), per_col).unwrap();
    ndarray_npy::write_npy(contig_dir.join("max_del.npy"), &arr).unwrap();

    std::fs::create_dir_all(contig_dir.join("dense")).unwrap();
    let dense = Array1::from_vec(vec![global]);
    ndarray_npy::write_npy(contig_dir.join("dense").join("max_del.npy"), &dense).unwrap();
}

/// Convert `records` to a finished SVAR2 contig under `out/{chrom}` and write the
/// `max_del` fixture. `out` must already exist.
fn build_contig(
    out: &Path,
    chrom: &str,
    samples: &[&str],
    ploidy: usize,
    records: &[SynthRecord],
) {
    let bcf = out.join("in.bcf");
    build_bcf_with_index(&bcf, chrom, 1_000_000, samples, records);
    process_chromosome(
        bcf.to_str().unwrap(),
        chrom,
        out.to_str().unwrap(),
        samples,
        1000, // chunk_size
        ploidy,
        1,    // htslib_threads
        4096, // long_allele_capacity
    )
    .expect("process_chromosome should succeed");
    write_max_del_fixture(&out.join(chrom), samples.len(), ploidy, records);
}

#[test]
fn test_open_ok_and_missing_dirs_tolerated() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord { pos: 100, ref_allele: b"A", alts: vec![&b"C"[..]], gt: vec![1, 0, 1, 1] },
        SynthRecord { pos: 300, ref_allele: b"AT", alts: vec![&b"A"[..]], gt: vec![1, 1, 0, 0] },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    assert!(ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).is_ok());

    // A bare contig dir with no sub-streams at all still opens (everything empty).
    let empty = tmp.path().join("empty_out");
    std::fs::create_dir_all(empty.join("chrX")).unwrap();
    assert!(ContigReader::open(empty.to_str().unwrap(), "chrX", 1, 2).is_ok());
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e sbox cargo test --no-default-features --test test_query test_open_ok_and_missing_dirs_tolerated`
Expected: FAIL — `no function or associated item named `open` found for struct `ContigReader`` (the struct/method don't exist yet).

- [ ] **Step 3: Write the implementation**

In `src/query.rs`, extend the top `use` block to:

```rust
use std::fs::File;
use std::io::ErrorKind;
use std::path::Path;

use memmap2::Mmap;
use ndarray::{Array1, Array2};

use crate::bits;
use crate::layout::{self, ContigPaths};
use crate::nrvk::LongAlleleReader;
use crate::search::{SearchTree, overlap_range};
```

Then add, before the `#[cfg(test)] mod tests`:

```rust
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
```

Leave the module-level `#![allow(dead_code)]` from Task 3 in place for now — `gather_run`, `kway_merge`, and `DenseView::carried` are still only reached from tests until `overlap_sample` (Task 6). It is removed in Task 6.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e sbox cargo test --no-default-features --test test_query test_open_ok_and_missing_dirs_tolerated`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/query.rs tests/test_query.rs
git commit -m "feat: ContigReader opens and mmaps SVAR2 contig sidecars"
```

---

### Task 6: `overlap_sample` + `QueryResult` (end-to-end query)

**Files:**
- Modify: `src/query.rs` (add `HapCalls`, `QueryResult`, `decode_indel_hit`, `overlap_sample`)
- Modify: `tests/test_query.rs` (add known-VCF query assertions)

**Interfaces:**
- Consumes: `ContigReader`, `Call`, `gather_run`, `kway_merge`, `rvk::{decode_key, DecodedKey, deletion_len, decode_snp_2bit, unpack_snp_key_at}`, `LongAlleleReader::get_allele`.
- Produces:
  - `pub struct HapCalls { pub positions: Vec<u32>, pub ilens: Vec<i32>, pub alts: Vec<Vec<u8>> }` (derives `Default`)
  - `pub struct QueryResult { pub per_hap: Vec<HapCalls> }` (`per_hap.len() == ploidy`)
  - `pub fn overlap_sample(reader: &ContigReader, sample: usize, q_start: u32, q_end: u32) -> QueryResult`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_query.rs`:

```rust
use genoray_core::query::overlap_sample;

// Mirrors the shape of `test_e2e_normalized_bcf_pipeline`: SNP@100 (-> dense/snp,
// x=3), INS@200 (-> var_key/indel, x=1), DEL@300 (-> dense/indel, x=2). The query
// unions across those three sub-streams.
#[test]
fn test_overlap_sample_known_contig() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord { pos: 100, ref_allele: b"A", alts: vec![&b"C"[..]], gt: vec![1, 0, 1, 1] },
        SynthRecord { pos: 200, ref_allele: b"A", alts: vec![&b"AT"[..]], gt: vec![0, 1, 0, 0] },
        SynthRecord { pos: 300, ref_allele: b"AT", alts: vec![&b"A"[..]], gt: vec![1, 1, 0, 0] },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap();

    // Sample 0, whole contig.
    let r = overlap_sample(&reader, 0, 0, 1000);
    assert_eq!(r.per_hap.len(), 2);
    // hap 0 (S0_p0): SNP@100 (gt 1) + DEL@300 (gt 1); INS@200 gt 0.
    assert_eq!(r.per_hap[0].positions, vec![100, 300]);
    assert_eq!(r.per_hap[0].ilens, vec![0, -1]);
    assert_eq!(r.per_hap[0].alts, vec![b"C".to_vec(), Vec::<u8>::new()]);
    // hap 1 (S0_p1): INS@200 (gt 1) + DEL@300 (gt 1); SNP@100 gt 0.
    assert_eq!(r.per_hap[1].positions, vec![200, 300]);
    assert_eq!(r.per_hap[1].ilens, vec![1, -1]);
    assert_eq!(r.per_hap[1].alts, vec![b"AT".to_vec(), Vec::<u8>::new()]);

    // Sample 1: only the SNP@100 (gt for haps 2,3 = 1,1).
    let r1 = overlap_sample(&reader, 1, 0, 1000);
    assert_eq!(r1.per_hap[0].positions, vec![100]);
    assert_eq!(r1.per_hap[1].positions, vec![100]);

    // Deletion spanning the query start: [301, 302) still returns DEL@300
    // (v_end = 300 + 1 + 1 = 302). Exercises the dense/indel max_del path.
    let r2 = overlap_sample(&reader, 0, 301, 302);
    assert_eq!(r2.per_hap[0].positions, vec![300]);
    assert_eq!(r2.per_hap[1].positions, vec![300]);

    // No overlap: gap between variants.
    let r3 = overlap_sample(&reader, 0, 150, 160);
    assert!(r3.per_hap[0].positions.is_empty());
    assert!(r3.per_hap[1].positions.is_empty());
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e sbox cargo test --no-default-features --test test_query test_overlap_sample_known_contig`
Expected: FAIL — `cannot find function `overlap_sample` in module `genoray_core::query``.

- [ ] **Step 3: Write the implementation**

First, **remove the module-level `#![allow(dead_code)]`** added in Task 3 (near the top of `src/query.rs`) — `overlap_sample` below makes every internal live, so the blanket allow is no longer needed and would mask a real regression.

Then, in `src/query.rs`, extend the top `use` block's crate imports to include the decode seam:

```rust
use crate::rvk::{self, DecodedKey};
```

Then add, before the `#[cfg(test)] mod tests`:

```rust
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

/// Decode an indel key into `(ilen, alt)` for a hit, resolving long-INS lookups
/// through the LUT. `alt` is empty for a pure DEL.
fn decode_indel_hit(key: u32, lut: Option<&LongAlleleReader>) -> (i32, Vec<u8>) {
    match rvk::decode_key(key) {
        DecodedKey::Inline { alt } => (alt.len() as i32 - 1, alt),
        DecodedKey::PureDel { ilen } => (ilen, Vec::new()),
        DecodedKey::Lookup { row } => {
            let alt = lut
                .expect("indel lookup key requires a long-allele LUT")
                .get_allele(row);
            (alt.len() as i32 - 1, alt)
        }
    }
}

/// Return every variant that `sample` carries overlapping `[q_start, q_end)`, per
/// haplotype, position-sorted, unioning the var_key and dense sub-streams.
pub fn overlap_sample(
    reader: &ContigReader,
    sample: usize,
    q_start: u32,
    q_end: u32,
) -> QueryResult {
    let ploidy = reader.ploidy;
    let mut per_hap = Vec::with_capacity(ploidy);

    for p in 0..ploidy {
        let col = sample * ploidy + p; // flat column
        let hap = col; // sample-major hap index == flat column
        let mut runs: Vec<Vec<Call>> = Vec::with_capacity(4);

        // --- var_key/snp column ---
        {
            let (o0, o1) = reader.vk_snp.column(col);
            let positions = &reader.vk_snp.positions()[o0..o1];
            let keys = as_bytes(&reader.vk_snp.keys);
            let mut run = Vec::new();
            gather_run(
                positions,
                0,
                q_start,
                q_end,
                |_| 0,
                |_| true,
                |i| (0, vec![rvk::decode_snp_2bit(rvk::unpack_snp_key_at(keys, o0 + i))]),
                &mut run,
            );
            runs.push(run);
        }

        // --- var_key/indel column ---
        {
            let (o0, o1) = reader.vk_indel.column(col);
            let positions = &reader.vk_indel.positions()[o0..o1];
            let all_keys = as_u32(&reader.vk_indel.keys);
            let keys = &all_keys[o0..o1];
            let max_del = reader.vk_indel_max_del[[sample, p]];
            let lut = reader.lut.as_ref();
            let mut run = Vec::new();
            gather_run(
                positions,
                max_del,
                q_start,
                q_end,
                |i| rvk::deletion_len(keys[i]),
                |_| true,
                |i| decode_indel_hit(keys[i], lut),
                &mut run,
            );
            runs.push(run);
        }

        // --- dense/snp (shared table, genotype-filtered) ---
        if let Some(dense) = &reader.dense_snp {
            let positions = dense.positions();
            let keys = as_bytes(&dense.keys);
            let mut run = Vec::new();
            gather_run(
                positions,
                0,
                q_start,
                q_end,
                |_| 0,
                |col_d| dense.carried(hap, col_d),
                |col_d| (0, vec![rvk::decode_snp_2bit(rvk::unpack_snp_key_at(keys, col_d))]),
                &mut run,
            );
            runs.push(run);
        }

        // --- dense/indel (shared table, genotype-filtered) ---
        if let Some(dense) = &reader.dense_indel {
            let positions = dense.positions();
            let keys = as_u32(&dense.keys);
            let max_del = reader.dense_indel_max_del;
            let lut = reader.lut.as_ref();
            let mut run = Vec::new();
            gather_run(
                positions,
                max_del,
                q_start,
                q_end,
                |i| rvk::deletion_len(keys[i]),
                |col_d| dense.carried(hap, col_d),
                |i| decode_indel_hit(keys[i], lut),
                &mut run,
            );
            runs.push(run);
        }

        let merged = kway_merge(runs);
        let mut hc = HapCalls::default();
        for c in merged {
            hc.positions.push(c.position);
            hc.ilens.push(c.ilen);
            hc.alts.push(c.alt);
        }
        per_hap.push(hc);
    }

    QueryResult { per_hap }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e sbox cargo test --no-default-features --test test_query test_overlap_sample_known_contig`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/query.rs tests/test_query.rs
git commit -m "feat: overlap_sample unions sub-streams into per-hap QueryResult"
```

---

### Task 7: Oracle proptest against a brute-force reference

**Files:**
- Modify: `tests/test_query.rs` (add owned-record model, arbitrary strategy, oracle, proptest)

**Interfaces:**
- Consumes: everything from Tasks 5–6 plus `proptest`.
- Produces (test-local): `struct OwnedRecord`, `fn arb_records(n_haps) -> impl Strategy`, `fn oracle(...) -> Vec<(Vec<u32>, Vec<i32>, Vec<Vec<u8>>)>`, `#[test] prop_overlap_sample_matches_oracle`.

- [ ] **Step 1: Write the failing proptest**

Add to `tests/test_query.rs`:

```rust
use proptest::prelude::*;

/// Owned analogue of `SynthRecord` (proptest needs values that outlive the
/// borrow). One atomized bi-allelic variant.
#[derive(Clone, Debug)]
struct OwnedRecord {
    pos: i64,
    ref_allele: Vec<u8>,
    alt: Vec<u8>,
    gt: Vec<i32>, // len == n_haps
}

/// Random atomized contig: strictly increasing positions, each variant a SNP,
/// INS (alt = anchor + tail), or DEL (ref = anchor + tail), with random per-hap
/// genotypes. INS tails reach 15 bases so some insertions spill to the LUT.
fn arb_records(n_haps: usize) -> impl Strategy<Value = Vec<OwnedRecord>> {
    proptest::collection::vec(
        (
            0u8..3u8,                                     // kind: 0 SNP, 1 INS, 2 DEL
            0usize..4,                                    // anchor base index
            0usize..4,                                    // SNP alt base index
            proptest::collection::vec(0usize..4, 1..16), // INS/DEL tail (>= 1 base)
            proptest::collection::vec(0i32..2, n_haps..=n_haps), // genotypes
            1u32..40u32,                                  // position gap
        ),
        1..8, // 1..7 records; empty contigs are covered by the degenerate tests
    )
    .prop_map(move |specs| {
        const BASES: [u8; 4] = [b'A', b'C', b'T', b'G'];
        let mut pos: i64 = 100;
        let mut out = Vec::new();
        for (kind, anchor, snp_alt, tail, gt, gap) in specs {
            pos += gap as i64;
            let b0 = BASES[anchor];
            let (ref_allele, alt) = match kind {
                0 => {
                    let alt_idx = if snp_alt == anchor { (anchor + 1) % 4 } else { snp_alt };
                    (vec![b0], vec![BASES[alt_idx]])
                }
                1 => {
                    let mut a = vec![b0];
                    a.extend(tail.iter().map(|&x| BASES[x]));
                    (vec![b0], a)
                }
                _ => {
                    let mut r = vec![b0];
                    r.extend(tail.iter().map(|&x| BASES[x]));
                    (r, vec![b0])
                }
            };
            out.push(OwnedRecord { pos, ref_allele, alt, gt });
        }
        out
    })
}

/// Brute-force reference: for `sample`, per hap, the carried variants overlapping
/// `[q_start, q_end)` in position order. `alt` matches the query contract — the
/// ALT bases for SNP/INS, empty for a pure DEL.
fn oracle(
    records: &[OwnedRecord],
    sample: usize,
    ploidy: usize,
    q_start: u32,
    q_end: u32,
) -> Vec<(Vec<u32>, Vec<i32>, Vec<Vec<u8>>)> {
    let mut per_hap: Vec<(Vec<u32>, Vec<i32>, Vec<Vec<u8>>)> =
        vec![(Vec::new(), Vec::new(), Vec::new()); ploidy];
    for rec in records {
        let pos = rec.pos as u32;
        let ilen = rec.alt.len() as i32 - rec.ref_allele.len() as i32;
        let del = if ilen < 0 { (-ilen) as u32 } else { 0 };
        let v_end = pos + 1 + del;
        if !(pos < q_end && q_start < v_end) {
            continue;
        }
        let alt = if ilen < 0 { Vec::new() } else { rec.alt.clone() };
        for p in 0..ploidy {
            let hap = sample * ploidy + p;
            if rec.gt[hap] == 1 {
                per_hap[p].0.push(pos);
                per_hap[p].1.push(ilen);
                per_hap[p].2.push(alt.clone());
            }
        }
    }
    per_hap // records are position-sorted, so each hap's lists already are too
}

proptest! {
    // Heavy: each case runs the full converter (BCF write + index + pipeline), so
    // the case count is deliberately low. Not a silent cap — 24 random contigs,
    // each queried for every sample, is the primary correctness gate for the
    // dense/var_key union.
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn prop_overlap_sample_matches_oracle(
        records in arb_records(6), // 3 samples, diploid -> 6 haps
        q_start in 0u32..1200,
        q_len in 1u32..300,
    ) {
        let n_samples = 3;
        let ploidy = 2;
        let sample_names = ["S0", "S1", "S2"];

        let synth: Vec<SynthRecord> = records
            .iter()
            .map(|r| SynthRecord {
                pos: r.pos,
                ref_allele: &r.ref_allele,
                alts: vec![&r.alt[..]],
                gt: r.gt.clone(),
            })
            .collect();

        let tmp = tempdir().unwrap();
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        build_contig(&out, "chr1", &sample_names, ploidy, &synth);
        let reader = ContigReader::open(out.to_str().unwrap(), "chr1", n_samples, ploidy).unwrap();

        let q_end = q_start + q_len;
        for s in 0..n_samples {
            let got = overlap_sample(&reader, s, q_start, q_end);
            let want = oracle(&records, s, ploidy, q_start, q_end);
            for p in 0..ploidy {
                prop_assert_eq!(&got.per_hap[p].positions, &want[p].0, "s={} p={} positions", s, p);
                prop_assert_eq!(&got.per_hap[p].ilens, &want[p].1, "s={} p={} ilens", s, p);
                prop_assert_eq!(&got.per_hap[p].alts, &want[p].2, "s={} p={} alts", s, p);
            }
        }
    }
}
```

- [ ] **Step 2: Run the proptest to verify it fails (or errors) first**

Temporarily break the assertion to confirm the harness runs the real code (optional sanity), or simply run it — since all called functions exist, this step verifies the proptest *compiles and executes*. If it passes immediately, that is the expected green state; there is no separate "red" for an oracle test whose production code already exists. Confirm it compiles:

Run: `pixi run -e sbox cargo test --no-default-features --test test_query prop_overlap_sample_matches_oracle`
Expected: the test runs 24 cases. It must PASS. If it FAILS, proptest prints a minimized `records`/query counterexample — treat that as a real bug (use superpowers:systematic-debugging), not a flaky test.

- [ ] **Step 3: (If red) fix the implementation, not the test**

If the proptest surfaces a mismatch, the bug is in Task 4–6 code (or the `max_del` fixture), not the oracle. Do not weaken the oracle. Common suspects: column vs. absolute indexing in `overlap_sample`, `hap` vs `col` in `carried`, `v_end` off-by-one, SNP unpack offset (`o0 + i`).

- [ ] **Step 4: Run to verify green**

Run: `pixi run -e sbox cargo test --no-default-features --test test_query prop_overlap_sample_matches_oracle`
Expected: PASS (24 cases).

- [ ] **Step 5: Commit**

```bash
git add tests/test_query.rs
git commit -m "test: oracle proptest for overlap_sample across random contigs"
```

---

### Task 8: Dense/var_key routing cross-check + degenerate cases

**Files:**
- Modify: `tests/test_query.rs` (add targeted `#[test]`s)

**Interfaces:**
- Consumes: Tasks 5–6 API + the fixture builder.
- Produces (test-local): `#[test] test_routing_invariant_dense_vs_var_key`, `#[test] test_pure_snp_contig_and_no_dense_dir`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_query.rs`:

```rust
// A variant routed to dense vs. var_key must give identical query results for the
// carrying sample — routing is an internal storage choice, invisible to queries.
// A deletion carried by 1 sample (rare -> var_key/indel) vs. by many samples
// (common -> dense/indel) must both come back with the same (pos, ilen, alt).
#[test]
fn test_routing_invariant_dense_vs_var_key() {
    // Rare: 6 samples, only S0_p0 carries the DEL -> var_key/indel.
    let rare = {
        let tmp = tempdir().unwrap();
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let samples = ["S0", "S1", "S2", "S3", "S4", "S5"];
        let records = vec![SynthRecord {
            pos: 500,
            ref_allele: b"ATATA", // 4-base deletion
            alts: vec![&b"A"[..]],
            gt: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }];
        build_contig(&out, "chr1", &samples, 2, &records);
        let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 6, 2).unwrap();
        let r = overlap_sample(&reader, 0, 0, 1000);
        (r.per_hap[0].positions.clone(), r.per_hap[0].ilens.clone(), r.per_hap[0].alts.clone())
    };

    // Common: same DEL carried by nearly everyone -> dense/indel.
    let common = {
        let tmp = tempdir().unwrap();
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let samples = ["S0", "S1", "S2", "S3", "S4", "S5"];
        let records = vec![SynthRecord {
            pos: 500,
            ref_allele: b"ATATA",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        }];
        build_contig(&out, "chr1", &samples, 2, &records);
        let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 6, 2).unwrap();
        let r = overlap_sample(&reader, 0, 0, 1000);
        (r.per_hap[0].positions.clone(), r.per_hap[0].ilens.clone(), r.per_hap[0].alts.clone())
    };

    assert_eq!(rare, common);
    assert_eq!(rare.0, vec![500]);
    assert_eq!(rare.1, vec![-4]); // 5-base ref, 1-base alt
    assert_eq!(rare.2, vec![Vec::<u8>::new()]);
}

// A pure-SNP contig (all-zero max_del, possibly no dense/indel or var_key/indel
// dir) queries correctly, and a query outside all variants is empty.
#[test]
fn test_pure_snp_contig_and_no_overlap() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0"];
    let records = vec![
        SynthRecord { pos: 100, ref_allele: b"A", alts: vec![&b"C"[..]], gt: vec![1, 1] },
        SynthRecord { pos: 200, ref_allele: b"G", alts: vec![&b"T"[..]], gt: vec![1, 0] },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 1, 2).unwrap();

    let r = overlap_sample(&reader, 0, 0, 1000);
    assert_eq!(r.per_hap[0].positions, vec![100, 200]);
    assert_eq!(r.per_hap[0].ilens, vec![0, 0]);
    assert_eq!(r.per_hap[1].positions, vec![100]); // hap 1 carries only the SNP@100

    // Entirely-left and entirely-right queries are empty.
    let left = overlap_sample(&reader, 0, 0, 50);
    assert!(left.per_hap[0].positions.is_empty());
    let right = overlap_sample(&reader, 0, 900, 1000);
    assert!(right.per_hap[0].positions.is_empty());
}
```

- [ ] **Step 2: Run to verify they fail (or exercise a real gap)**

Run: `pixi run -e sbox cargo test --no-default-features --test test_query test_routing_invariant_dense_vs_var_key test_pure_snp_contig_and_no_overlap`
Expected: These call only existing API, so they should compile and PASS. If either FAILS, it has found a real routing/empty-handling bug — debug the implementation (superpowers:systematic-debugging), do not adjust the assertions.

(If a genotype vector length looks off: for 6 samples diploid the `gt` vector is 12 long, sample-major `[s0_p0, s0_p1, s1_p0, ...]`.)

- [ ] **Step 3: (If red) fix implementation**

Address any bug in Task 5/6 code (dense presence handling, `col`/`hap` indexing).

- [ ] **Step 4: Run the full query test file + whole suite**

Run: `pixi run -e sbox cargo test --no-default-features --test test_query`
Expected: PASS (all `test_query` tests).
Run: `pixi run -e sbox cargo test --no-default-features`
Expected: PASS (whole Rust suite: 106 baseline lib + 8 new lib + all integration tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_query.rs
git commit -m "test: dense/var_key routing invariance and degenerate query cases"
```

---

## Final verification

- [ ] **Run the whole Rust test suite:** `pixi run -e sbox cargo test --no-default-features`
  Expected: all pass, 0 failures.
- [ ] **Lint clean:** `pixi run -e sbox cargo clippy --no-default-features --tests -- -D warnings` and `pixi run -e sbox cargo fmt --check`
  Expected: no warnings, no diffs. (These also run as pre-commit hooks.)
- [ ] **Confirm `search.rs` is untouched:** `git diff --stat main -- src/search.rs`
  Expected: empty (no changes to the frozen core).

---

## Notes for the implementer (context you lack)

- **Why mmap and not `std::fs::read`.** Sidecars can be large; the query touches only a slice. mmap gives a page-aligned, zero-copy `&[u8]`, and page alignment is what makes `bytemuck::cast_slice::<u8, u32>` safe here (a plain `Vec<u8>` from `fs::read` can be under-aligned and panic — see the comment in `tests/common/mod.rs:73`). This is the first mmap use in the crate; that is intentional and matches the max_del post-pass spec's access model.
- **Why `v_end = pos + 1 + deletion_len`.** `overlap_range` (see its doc comment, `src/search.rs:135`) defines the exclusive end as `v_start + 1` for a SNP and `v_start + 1 + d` for a `d`-base deletion. Only pure DELs contribute `d > 0`; SNP and INS have `d == 0`. `rvk::deletion_len` returns exactly `d`.
- **Why a pure DEL's `alt` is empty.** The inline DEL key stores only the signed `ilen` — the ALT (anchor) base is a reference base, not in the key. Recovering it needs the reference genome (M2b territory). The query returns `ilen` (which fully specifies the deletion length) and an empty `alt`; the oracle matches this contract.
- **Why the `max_del` fixture is conservative.** Until the post-pass producer lands, the fixture over-estimates each column's bound (per-column max over *all* carried deletions, not just var_key-routed ones; global max for dense). `overlap_range`'s `overlap_max_region_length_overshoot_is_safe` proptest proves an over-estimate is always correct — so when the real (tight) `max_del.npy` replaces the fixture, query results are identical. Do not tighten the fixture to "match" the producer; the point is that both yield the same answer.
- **Rebuild times.** The first `cargo test` in this worktree compiles `hts-sys`/`rust-htslib` (vendored htslib via the conda toolchain) — several minutes. Subsequent runs are incremental. Always go through `pixi run -e sbox` so `libclang`/`zlib.h` are on the path.
- **Single-query only.** A batched (many-ranges) wrapper is a deliberate follow-up (see the spec's Open questions). Do not add one here.

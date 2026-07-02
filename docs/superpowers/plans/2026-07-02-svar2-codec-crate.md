# `svar2-codec` Crate Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the SVAR2 variant-key bit-layout seam (encode + decode) from `src/rvk.rs` / `src/types.rs` into a new dependency-light `svar2-codec` workspace crate that can be published to crates.io and linked by GenVarLoader's Rust core.

**Architecture:** Convert the single-crate repo into a two-member Cargo workspace: the existing `genoray` pyo3 cdylib (root) plus a new pure-`rlib` `svar2-codec` (std-only, no `pyo3`/`htslib`/`nrvk`). Relocate every function and constant that knows the 2-bit SNP key layout or the 32-bit indel key layout — both the pure *encode* primitives and the *decode* primitives — into `svar2-codec`, resolving the documented "keep encode/decode co-located" hazard by making the crate the single source of layout truth. Only the parts that need I/O (`pack_snp_key_file`), the long-allele bank writer (`pack_variant`'s spill, `classify_variant`), or routing stay in `genoray` and call into the crate. Intra-genoray call sites keep compiling via thin `pub use` re-export shims in `rvk`/`types`.

**Tech Stack:** Rust (edition 2024), Cargo workspaces, `proptest` + `tempfile` (dev), pixi (`lint` env carries the Rust + clang toolchain), maturin (root wheel build, must stay green).

## Global Constraints

- **`svar2-codec` is pure and leaf.** No `pyo3`, no `rust-htslib`, no `memmap2`, no dependency on any `genoray` module. Std-only; **zero** non-std runtime dependencies. It must remain independently publishable to crates.io (`cargo publish -p svar2-codec` with no path deps to unpublished crates).
- **Edition:** `edition = "2024"` for both crates (matches the existing root).
- **Moves are relocations, not rewrites.** Copy function bodies verbatim; behavior must stay byte-identical. The migrated proptests (PEXT-vs-SWAR byte-identical; `deletion_len` oracle) are the safety net — they must stay green at every step.
- **Layout co-location:** every bit-position of both key layouts lives in `svar2-codec`. `genoray` never re-derives a shift or mask; it calls the crate.
- **Backward-compatible internal API:** after each move, `crate::rvk::<name>` and `crate::types::<name>` still resolve (via `pub use svar2_codec::<name>;`), so existing call sites in `query.rs`, `max_del.rs`, `dense_merge.rs`, `streams.rs`, and `tests/test_e2e.rs` compile unchanged.
- **Rust test commands** (run in the pixi `lint` env, which has `rust` + `clangdev`):
  - Codec only (fast, no htslib): `pixi run -e lint cargo test -p svar2-codec`
  - Full genoray Rust suite (needs `--no-default-features` so pyo3 links libpython instead of building the extension module — per the comment in `Cargo.toml`): `pixi run -e lint cargo test -p genoray --no-default-features`
- **Publishing is manual and out of scope here.** This plan makes `svar2-codec` publish-*ready* (`cargo publish --dry-run -p svar2-codec` passes). Do **not** edit `.github/workflows/publish.yaml`; the crates.io release + version gate is managed by the maintainer.
- **Commits:** conventional-commit style (`cz` / commitizen check runs in prek). prek hooks are already installed in this worktree. Commit only files you created/modified in that task; leave the untracked `docs/superpowers/plans/2026-07-02-svar2-m3-format-finalization.md` alone.
- All commits land in this genoray worktree (branch `svar-2`).

---

### Task 1: Create the workspace + empty `svar2-codec` crate

**Files:**
- Modify: `Cargo.toml` (add `[workspace]` table)
- Create: `svar2-codec/Cargo.toml`
- Create: `svar2-codec/src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: a compiling, testable `svar2-codec` workspace member (empty public surface). Later tasks add items to `svar2-codec/src/lib.rs`.

- [ ] **Step 1: Add the `[workspace]` table to the root `Cargo.toml`**

Append after the existing `[features]` block (root stays the `genoray` package; adding `[workspace]` here makes it the workspace root with `svar2-codec` as a second member):

```toml
[workspace]
members = ["svar2-codec"]
```

- [ ] **Step 2: Create `svar2-codec/Cargo.toml`**

```toml
[package]
name = "svar2-codec"
version = "0.1.0"
edition = "2024"
description = "Encode/decode for the SVAR 2.0 variant-key bit layout (SNP 2-bit + indel 32-bit inline keys)."
license = "MIT OR Apache-2.0"
repository = "https://github.com/d-laub/genoray"
keywords = ["bioinformatics", "genomics", "variant", "vcf"]
categories = ["science", "encoding"]

[dependencies]

[dev-dependencies]
proptest = "1.11.0"
```

- [ ] **Step 3: Create an empty `svar2-codec/src/lib.rs`**

```rust
//! Encode/decode for the SVAR 2.0 variant-key bit layout.
//!
//! This crate is the **single source of truth** for the on-disk key layouts:
//! the 2-bit SNP code stream and the 32-bit indel key (inline INS/SNP, pure DEL,
//! and long-allele-bank lookup lanes). Both the pure encode primitives and the
//! decode primitives live here, so the two halves of the layout can never drift.
//!
//! Pure and std-only: no I/O, no pyo3, no long-allele bank. Callers that need
//! those (file packing, bank spill) live in the `genoray` crate and call in here.
```

- [ ] **Step 4: Verify the workspace builds and both members test**

Run: `pixi run -e lint cargo test -p svar2-codec`
Expected: compiles; `running 0 tests` (no tests yet), `test result: ok. 0 passed`.

Run: `pixi run -e lint cargo build -p genoray --no-default-features`
Expected: the root crate still builds under the new workspace manifest (maturin's target is unaffected).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock svar2-codec/Cargo.toml svar2-codec/src/lib.rs
git commit -m "build(svar2-codec): scaffold workspace crate for the key-layout seam"
```

---

### Task 2: Move the layout constants + unified `BASES` table

**Files:**
- Modify: `svar2-codec/src/lib.rs` (add constants)
- Modify: `Cargo.toml` (add `svar2-codec` path dependency to `genoray`)
- Modify: `src/types.rs:3-9` (replace the two `const` defs with a re-export shim)
- Modify: `src/rvk.rs:6-8` (import path unchanged via the `types` shim; no edit needed if it still imports from `crate::types`)

**Interfaces:**
- Consumes: nothing.
- Produces: `svar2_codec::MIN_I31: i32`, `svar2_codec::MAX_INLINE_ALT_LEN: usize`, `svar2_codec::BASES: [u8; 4]`. `crate::types::MIN_I31` / `crate::types::MAX_INLINE_ALT_LEN` remain valid (re-exported).

- [ ] **Step 1: Add the constants to `svar2-codec/src/lib.rs`**

```rust
/// Minimum signed `ilen` representable inline as a pure DEL (i31 two's complement).
/// Real data won't approach this — atomized DELs span at most chromosome length
/// (~250 Mbp).
pub const MIN_I31: i32 = -(1 << 30);

/// Maximum ALT byte length that fits the inline encoding (26 bits ÷ 2 bits/base =
/// 13). Beyond this, a pure-INS variant spills to the long-allele bank.
pub const MAX_INLINE_ALT_LEN: usize = 13;

/// 2-bit code → ALT base. `A=00 C=01 T=10 G=11`. `T`/`G` are swapped vs. the
/// obvious alphabetical order — the values are an implementation detail of this
/// crate and carry no meaning outside it.
pub const BASES: [u8; 4] = [b'A', b'C', b'T', b'G'];
```

- [ ] **Step 2: Add the path dependency from `genoray` to `svar2-codec`**

In the root `Cargo.toml` `[dependencies]` (keep the list alphabetical-ish; place after `serde_json`):

```toml
svar2-codec = { path = "svar2-codec" }
```

- [ ] **Step 3: Replace the constant definitions in `src/types.rs` with a shim**

Replace lines 3-9 (the `MIN_I31` and `MAX_INLINE_ALT_LEN` doc-comments + `pub const`s) with:

```rust
// Key-layout constants now live in the `svar2-codec` crate (single source of
// layout truth). Re-exported so existing `crate::types::{MIN_I31,
// MAX_INLINE_ALT_LEN}` call sites keep resolving.
pub use svar2_codec::{MAX_INLINE_ALT_LEN, MIN_I31};
```

(Leave the `use crate::streams::StreamMap;` line and `BitGrid3` below untouched.)

- [ ] **Step 4: Verify the full Rust suite still passes**

Run: `pixi run -e lint cargo test -p genoray --no-default-features`
Expected: PASS — same test count as before this task; `rvk.rs` still reads `MIN_I31`/`MAX_INLINE_ALT_LEN` through `crate::types`.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock svar2-codec/src/lib.rs src/types.rs
git commit -m "refactor(svar2-codec): move key-layout constants + BASES into the crate"
```

---

### Task 3: Move the SNP 2-bit code + pack/unpack primitives

**Files:**
- Modify: `svar2-codec/src/lib.rs` (add SNP fns + their unit tests)
- Modify: `src/rvk.rs` (delete `encode_snp_2bit` `:126`, `pack_snp_keys` `:134`, `unpack_snp_keys` `:143`, `decode_snp_2bit` `:261`, `unpack_snp_key_at` `:269`; add `pub use` shims; update the local `BASES` in `decode_snp_2bit` — now gone)

**Interfaces:**
- Consumes: `svar2_codec::BASES`.
- Produces: `svar2_codec::{encode_snp_2bit, decode_snp_2bit, pack_snp_keys, unpack_snp_keys, unpack_snp_key_at}`. All keep resolving as `crate::rvk::<name>` via re-export, so `query.rs`, `dense_merge.rs`, and `tests/test_e2e.rs` are untouched.

- [ ] **Step 1: Add the SNP primitives to `svar2-codec/src/lib.rs`**

```rust
/// Bare 2-bit ALT code for the SNP stream: `A=00 C=01 T=10 G=11`. Branchless
/// `(base >> 1) & 3` — no lookup, no match.
#[inline(always)]
pub fn encode_snp_2bit(base: u8) -> u8 {
    (base >> 1) & 3
}

/// Recover the ALT base for a 2-bit SNP code. Inverse of [`encode_snp_2bit`].
#[inline]
pub fn decode_snp_2bit(code: u8) -> u8 {
    BASES[(code & 3) as usize]
}

/// Pack 2-bit SNP codes 4-per-byte, little-pair-first: code `i` occupies bits
/// `[(i&3)*2 + 1 : (i&3)*2]` of byte `i >> 2`. The final byte is zero-padded when
/// `codes.len()` is not a multiple of 4. Offsets index CALLS, not bytes.
pub fn pack_snp_keys(codes: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; codes.len().div_ceil(4)];
    for (i, &c) in codes.iter().enumerate() {
        out[i >> 2] |= (c & 3) << ((i & 3) * 2);
    }
    out
}

/// Inverse of [`pack_snp_keys`]. Returns the first `n` codes.
pub fn unpack_snp_keys(packed: &[u8], n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (packed[i >> 2] >> ((i & 3) * 2)) & 3)
        .collect()
}

/// Read the 2-bit SNP code at call index `i` from a 2-bit-packed buffer
/// (4 codes/byte; see [`pack_snp_keys`]) without materializing the whole array.
#[inline]
pub fn unpack_snp_key_at(packed: &[u8], i: usize) -> u8 {
    (packed[i >> 2] >> ((i & 3) * 2)) & 3
}
```

- [ ] **Step 2: Add SNP round-trip tests to `svar2-codec/src/lib.rs`**

Append a test module (create it if this is the first test in the file):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn snp_code_round_trip_all_bases() {
        for &b in &[b'A', b'C', b'G', b'T'] {
            let code = encode_snp_2bit(b);
            assert_eq!(decode_snp_2bit(code), b, "base {} round-trips", b as char);
        }
    }

    proptest! {
        #[test]
        fn snp_pack_unpack_round_trips(codes in proptest::collection::vec(0u8..4, 0..64)) {
            let packed = pack_snp_keys(&codes);
            prop_assert_eq!(unpack_snp_keys(&packed, codes.len()), codes.clone());
            for (i, &c) in codes.iter().enumerate() {
                prop_assert_eq!(unpack_snp_key_at(&packed, i), c);
            }
        }
    }
}
```

- [ ] **Step 3: Delete the five functions from `src/rvk.rs` and add re-export shims**

Remove the bodies of `encode_snp_2bit` (`:123-128`), `pack_snp_keys` (`:130-140`), `unpack_snp_keys` (`:142-147`), `decode_snp_2bit` (`:258-264`), and `unpack_snp_key_at` (`:266-271`). Near the top of `src/rvk.rs` (just below the existing `use` block), add:

```rust
// SNP 2-bit key primitives now live in `svar2-codec`. Re-exported so existing
// `crate::rvk::<name>` call sites (query.rs, dense_merge.rs, tests) resolve.
pub use svar2_codec::{
    decode_snp_2bit, encode_snp_2bit, pack_snp_keys, unpack_snp_key_at, unpack_snp_keys,
};
```

(Any `#[cfg(test)]` tests in `rvk.rs` that exercised these — e.g. SNP pack round-trips — move to the codec module in Step 2 or are deleted if now redundant. Keep tests that cover *other* `rvk` functions.)

- [ ] **Step 4: Verify both suites pass**

Run: `pixi run -e lint cargo test -p svar2-codec`
Expected: PASS including `snp_code_round_trip_all_bases` and `snp_pack_unpack_round_trips`.

Run: `pixi run -e lint cargo test -p genoray --no-default-features`
Expected: PASS — `pack_snp_key_file`, `dense_merge`, `query.rs`, and `tests/test_e2e.rs` still resolve the SNP fns through the `rvk` re-export.

- [ ] **Step 5: Commit**

```bash
git add svar2-codec/src/lib.rs src/rvk.rs
git commit -m "refactor(svar2-codec): move SNP 2-bit code + pack/unpack into the crate"
```

---

### Task 4: Move the inline indel encoder (PEXT/SWAR)

**Files:**
- Modify: `svar2-codec/src/lib.rs` (add the inline encoder + the PEXT-vs-SWAR proptest)
- Modify: `src/rvk.rs` (delete `load_padded_reversed` `:22`, `pext_reduce` `:54`, `swar_reduce_portable` `:75`, `encode_bases` `:104`, `encode_snp` `:118`, `encode_alt_inline` `:195`; make `pack_variant` call `svar2_codec::encode_alt_inline`)

**Interfaces:**
- Consumes: nothing (self-contained bit twiddling).
- Produces: `svar2_codec::encode_alt_inline(alt_allele: &[u8], ilen: u32) -> u32` — packs ≤13 DNA bases into the inline INS/SNP lane (top 5 bits `ilen`, bases at `[26:25]` downward, LSB 0). Panics if `alt_allele.len() > 13`.

- [ ] **Step 1: Add the inline encoder to `svar2-codec/src/lib.rs`**

Paste verbatim from the current `src/rvk.rs` (lines 21-121 and 188-209), making `encode_alt_inline` `pub` and keeping the helpers private:

```rust
#[inline(always)]
fn load_padded_reversed(alt_allele: &[u8], n: usize) -> (u64, u64) {
    debug_assert!(n <= 13);
    let mut padded = [0u8; 16];
    let n1 = n.min(8);
    for i in 0..n1 {
        padded[7 - i] = alt_allele[i];
    }
    let n2 = n.saturating_sub(8);
    for i in 0..n2 {
        padded[15 - i] = alt_allele[8 + i];
    }
    let block1 = u64::from_le_bytes(padded[0..8].try_into().unwrap());
    let block2 = u64::from_le_bytes(padded[8..16].try_into().unwrap());
    (block1, block2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
unsafe fn pext_reduce(block1: u64, block2: u64, n_bases: usize) -> u32 {
    use std::arch::x86_64::_pext_u64;
    const SWAR_MASK: u64 = 0x0303030303030303;
    let bits1 = (block1 >> 1) & SWAR_MASK;
    let extracted1 = unsafe { _pext_u64(bits1, SWAR_MASK) } as u32;
    let part1 = extracted1 << 11;
    if n_bases <= 8 {
        return part1;
    }
    let bits2 = (block2 >> 1) & SWAR_MASK;
    let extracted2 = unsafe { _pext_u64(bits2, SWAR_MASK) } as u32;
    let part2 = extracted2 >> 5;
    part1 | part2
}

#[inline(always)]
fn swar_reduce_portable(block1: u64, block2: u64, n_bases: usize) -> u32 {
    const SWAR_MASK: u64 = 0x0303030303030303;
    let bits1 = (block1 >> 1) & SWAR_MASK;
    let bits2 = (block2 >> 1) & SWAR_MASK;
    let mut payload: u32 = 0;
    let mut shift: i32 = 25;
    let n1 = n_bases.min(8);
    for i in 0..n1 {
        let byte_shift = (7 - i) * 8;
        payload |= (((bits1 >> byte_shift) & 3) as u32) << shift;
        shift -= 2;
    }
    if n_bases > 8 {
        let n2 = n_bases - 8;
        for i in 0..n2 {
            let byte_shift = (7 - i) * 8;
            payload |= (((bits2 >> byte_shift) & 3) as u32) << shift;
            shift -= 2;
        }
    }
    payload
}

#[inline(always)]
fn encode_bases(alt_allele: &[u8], n_bases: usize) -> u32 {
    let (block1, block2) = load_padded_reversed(alt_allele, n_bases);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            return unsafe { pext_reduce(block1, block2, n_bases) };
        }
    }
    swar_reduce_portable(block1, block2, n_bases)
}

#[inline(always)]
fn encode_snp(alt_base: u8) -> u32 {
    ((alt_base as u32 >> 1) & 3) << 25
}

/// Pack ≤13 DNA bases into the inline INS/SNP lane of a 32-bit indel key:
/// `[ ilen:5 | base[0..alt_len] × 2bit | _ | flag=0 ]`. `ilen` (top 5 bits) is
/// `alt_len - 1` for atomized INS/SNP; decode reconstructs `alt_len = ilen + 1`.
/// Panics if `alt_allele.len() > 13`.
#[inline(always)]
pub fn encode_alt_inline(alt_allele: &[u8], ilen: u32) -> u32 {
    let alt_len = alt_allele.len();
    if alt_len > 13 {
        panic!("Inline ALT must be 13 bases or fewer");
    }
    debug_assert!(ilen <= 12, "inline ilen must be ≤ 12 (alt_len ≤ 13)");
    if alt_len == 1 {
        return encode_snp(alt_allele[0]);
    }
    let payload = encode_bases(alt_allele, alt_len);
    payload | (ilen << 27)
}
```

> Note: `_pext_u64` is `unsafe` and, under edition 2024, must be wrapped in an explicit `unsafe { }` block even inside an `unsafe fn` — the two call sites above already show the wrapping. If `cargo` reports the outer `unsafe fn` as now having an unnecessary block, that is fine; keep the inner wrapping.

- [ ] **Step 2: Move the PEXT-vs-SWAR byte-identical proptest into the codec test module**

Locate the existing proptest in `src/rvk.rs`'s `#[cfg(test)]` module that asserts `pext_reduce` and `swar_reduce_portable` produce identical output over random ALT alleles (the "proven byte-identical by proptest" test). Move it into `svar2-codec/src/lib.rs`'s `mod tests`. If it referenced private helpers, they are now siblings in the codec module. Representative shape (adapt to the real test's name/strategy):

```rust
    proptest! {
        #[test]
        fn pext_and_swar_agree(bases in proptest::collection::vec(
            prop::sample::select(vec![b'A', b'C', b'G', b'T']), 1..=13usize)) {
            let (b1, b2) = load_padded_reversed(&bases, bases.len());
            let swar = swar_reduce_portable(b1, b2, bases.len());
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("bmi2") {
                let pext = unsafe { pext_reduce(b1, b2, bases.len()) };
                prop_assert_eq!(pext, swar);
            }
        }
    }
```

- [ ] **Step 3: Delete the moved fns from `src/rvk.rs` and rewire `pack_variant`**

Remove `load_padded_reversed`, `pext_reduce`, `swar_reduce_portable`, `encode_bases`, `encode_snp`, and `encode_alt_inline` from `src/rvk.rs`. In `pack_variant` (`:287`), change the inline call:

```rust
        if alt_allele.len() <= MAX_INLINE_ALT_LEN {
            return svar2_codec::encode_alt_inline(alt_allele, ilen as u32);
        }
```

(`MAX_INLINE_ALT_LEN` still resolves via `crate::types`, which re-exports it.)

- [ ] **Step 4: Verify both suites pass**

Run: `pixi run -e lint cargo test -p svar2-codec`
Expected: PASS including `pext_and_swar_agree`.

Run: `pixi run -e lint cargo test -p genoray --no-default-features`
Expected: PASS — the encoder round-trip e2e tests and the `dense2sparse_vk` path (via `classify_variant` → `pack_variant`) are unchanged in behavior.

- [ ] **Step 5: Commit**

```bash
git add svar2-codec/src/lib.rs src/rvk.rs
git commit -m "refactor(svar2-codec): move inline indel encoder (PEXT/SWAR) into the crate"
```

---

### Task 5: Move the indel decode primitives + the remaining encode lanes

**Files:**
- Modify: `svar2-codec/src/lib.rs` (add `DecodedKey`, `decode_key`, `decode_alt_inline`, `deletion_len`, `encode_pure_del`, `encode_lookup` + tests)
- Modify: `src/rvk.rs` (delete `decode_alt_inline` `:212`, `DecodedKey` `:232`, `decode_key` `:244`, `deletion_len` `:316`; add re-export shims; rewire `pack_variant`'s DEL + spill lanes)

**Interfaces:**
- Consumes: `svar2_codec::{BASES, MIN_I31}`.
- Produces:
  - `svar2_codec::DecodedKey` (`Inline { alt: Vec<u8> }` / `PureDel { ilen: i32 }` / `Lookup { row: u32 }`)
  - `svar2_codec::decode_key(key: u32) -> DecodedKey`
  - `svar2_codec::decode_alt_inline(payload: u32) -> Vec<u8>`
  - `svar2_codec::deletion_len(key: u32) -> u32`
  - `svar2_codec::encode_pure_del(ilen: i32) -> u32` (the `(ilen as u32) << 1` DEL lane)
  - `svar2_codec::encode_lookup(row: u32) -> u32` (the `(row << 1) | 1` bank-pointer lane)
  - `crate::rvk::{DecodedKey, decode_key, decode_alt_inline, deletion_len}` remain valid (re-exported), so `query.rs`, `max_del.rs`, and `tests/test_e2e.rs` are untouched.

- [ ] **Step 1: Add the decode primitives + remaining encode lanes to `svar2-codec/src/lib.rs`**

```rust
/// Decode the inline INS/SNP lane's ALT bases. Top 5 bits hold `ilen`;
/// `alt_len = ilen + 1` (atomized invariant). Bases read MSB-first from `[26:25]`.
#[inline(always)]
pub fn decode_alt_inline(payload: u32) -> Vec<u8> {
    let ilen = (payload >> 27) as usize;
    let alt_len = ilen + 1;
    let mut decoded = Vec::with_capacity(alt_len);
    for i in 0..alt_len {
        let shift = 25 - (i * 2);
        let bit_val = ((payload >> shift) & 3) as usize;
        decoded.push(BASES[bit_val]);
    }
    decoded
}

/// Discriminated form of a 32-bit indel key. Mirrors the encode lanes:
/// bit 0 = lookup flag, bit 31 (of a non-lookup key) = pure-DEL flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedKey {
    /// Inline INS/SNP; `alt` is the decoded ALT bases (`alt.len() == ilen + 1`).
    Inline { alt: Vec<u8> },
    /// Pure deletion of `-ilen` reference bases (`ilen < 0`). The anchor base is
    /// recovered from the reference downstream, not stored in the key.
    PureDel { ilen: i32 },
    /// Long insertion spilled to the long-allele bank at row `row`.
    Lookup { row: u32 },
}

/// Decode a packed 32-bit indel key into its discriminated form. Single decode
/// entry point — no caller re-derives the bit layout.
pub fn decode_key(key: u32) -> DecodedKey {
    if key & 1 == 1 {
        DecodedKey::Lookup { row: key >> 1 }
    } else if (key as i32) < 0 {
        DecodedKey::PureDel { ilen: (key as i32) >> 1 }
    } else {
        DecodedKey::Inline { alt: decode_alt_inline(key) }
    }
}

/// Reference-base deletion length encoded in a 32-bit indel key. Inverse of the
/// DEL lane of [`encode_pure_del`]: a pure DEL clears the lookup flag (bit 0) and
/// sets bit 31, storing signed `ilen` in `[31:1]`; the length is `-ilen`. Inline
/// INS/SNP keys (bit 31 clear) and lookup keys (bit 0 set) both yield `0`.
#[inline]
pub fn deletion_len(key: u32) -> u32 {
    if key & 1 == 1 {
        return 0;
    }
    if key & (1 << 31) == 0 {
        return 0;
    }
    let ilen = (key as i32) >> 1;
    debug_assert!(ilen < 0, "top-bit-set inline key must be a negative-ilen DEL");
    ilen.unsigned_abs()
}

/// Encode a pure deletion of `-ilen` reference bases into the DEL lane
/// (`bit 0 = 0`, `bit 31 = 1`, signed `ilen` in `[31:1]`). `ilen` must be `< 0`
/// and `>= MIN_I31`.
#[inline]
pub fn encode_pure_del(ilen: i32) -> u32 {
    debug_assert!(ilen < 0, "encode_pure_del expects a negative ilen");
    debug_assert!(ilen >= MIN_I31, "pure DEL ilen below MIN_I31 aliases the inline lane");
    (ilen as u32) << 1
}

/// Encode a long-allele-bank row index into the lookup lane (`(row << 1) | 1`).
/// `row` must fit in 31 bits.
#[inline]
pub fn encode_lookup(row: u32) -> u32 {
    debug_assert!(row < (1 << 31), "bank row index must fit in 31 bits");
    (row << 1) | 1
}
```

- [ ] **Step 2: Add decode/encode-lane tests to the codec `mod tests`**

```rust
    #[test]
    fn deletion_len_snp_ins_lookup_are_zero() {
        // inline SNP (ilen 0), inline INS, and a lookup key all decode to 0.
        assert_eq!(deletion_len(encode_alt_inline(b"A", 0)), 0);
        assert_eq!(deletion_len(encode_alt_inline(b"ACG", 2)), 0);
        assert_eq!(deletion_len(encode_lookup(5)), 0);
    }

    #[test]
    fn pure_del_round_trips() {
        for d in [1i32, 3, 100, 1 << 20] {
            let key = encode_pure_del(-d);
            assert_eq!(deletion_len(key), d as u32);
            assert_eq!(decode_key(key), DecodedKey::PureDel { ilen: -d });
        }
    }

    #[test]
    fn lookup_round_trips() {
        assert_eq!(decode_key(encode_lookup(42)), DecodedKey::Lookup { row: 42 });
    }

    proptest! {
        #[test]
        fn inline_ins_snp_round_trips(
            bases in proptest::collection::vec(
                prop::sample::select(vec![b'A', b'C', b'G', b'T']), 1..=13usize)) {
            let ilen = (bases.len() - 1) as u32;
            let key = encode_alt_inline(&bases, ilen);
            prop_assert_eq!(decode_key(key), DecodedKey::Inline { alt: bases.clone() });
        }
    }
```

- [ ] **Step 3: Delete the decode fns from `src/rvk.rs`, add shims, rewire `pack_variant`**

Remove `decode_alt_inline` (`:211-227`), `DecodedKey` (`:229-240`), `decode_key` (`:242-256`), and `deletion_len` (`:307-329`) from `src/rvk.rs`. Extend the re-export shim near the top of `rvk.rs` to include them:

```rust
pub use svar2_codec::{
    DecodedKey, decode_alt_inline, decode_key, decode_snp_2bit, deletion_len, encode_snp_2bit,
    pack_snp_keys, unpack_snp_key_at, unpack_snp_keys,
};
```

Rewire the DEL and spill lanes of `pack_variant` to use the crate (so no bit-position stays in `rvk`):

```rust
#[inline(always)]
pub fn pack_variant(ilen: i32, alt_allele: &[u8], bank: &mut LongAlleleTableWriter) -> u32 {
    if ilen >= 0 {
        if alt_allele.len() <= MAX_INLINE_ALT_LEN {
            return svar2_codec::encode_alt_inline(alt_allele, ilen as u32);
        }
        let row_index = bank.push_long_allele(alt_allele);
        return svar2_codec::encode_lookup(row_index);
    }
    svar2_codec::encode_pure_del(ilen)
}
```

Delete any now-duplicated `rvk` `#[cfg(test)]` tests for `deletion_len` / `decode_key` (they are covered in the codec module now); keep tests that assert `pack_variant` routing/spill behavior (those exercise the bank + the crate together and belong in `rvk`).

- [ ] **Step 4: Verify both suites pass**

Run: `pixi run -e lint cargo test -p svar2-codec`
Expected: PASS including `pure_del_round_trips`, `lookup_round_trips`, `inline_ins_snp_round_trips`.

Run: `pixi run -e lint cargo test -p genoray --no-default-features`
Expected: PASS — `query.rs` (`decode_key`/`decode_indel_hit`), `max_del.rs` (`deletion_len`), and `tests/test_e2e.rs` (`decode_key`/`decode_alt_inline`/`DecodedKey`) all resolve through the `rvk` re-export.

- [ ] **Step 5: Commit**

```bash
git add svar2-codec/src/lib.rs src/rvk.rs
git commit -m "refactor(svar2-codec): move indel decode + DEL/lookup encode lanes into the crate"
```

---

### Task 6: Publish-readiness + a crate-owned round-trip contract test

**Files:**
- Modify: `svar2-codec/src/lib.rs` (add a top-level module doc example + a combined round-trip proptest, if not already covered)
- Create: `svar2-codec/README.md`
- Modify: `svar2-codec/Cargo.toml` (add `readme`, `rust-version`)

**Interfaces:**
- Consumes: the full `svar2_codec` public surface.
- Produces: a crate that passes `cargo publish --dry-run -p svar2-codec`.

- [ ] **Step 1: Add a `README.md` for `svar2-codec`**

```markdown
# svar2-codec

Encode/decode for the [SVAR 2.0](https://github.com/d-laub/genoray) variant-key bit
layout: the 2-bit SNP code stream and the 32-bit indel key (inline INS/SNP, pure
DEL, and long-allele-bank lookup lanes).

Pure and `std`-only — no I/O, no Python bindings. It is the single source of truth
for the on-disk key layouts, shared by the `genoray` converter and by downstream
Rust consumers (e.g. GenVarLoader) that decode SVAR2 query results in-process.

## License

Licensed under either of Apache-2.0 or MIT at your option.
```

- [ ] **Step 2: Add `readme` + `rust-version` to `svar2-codec/Cargo.toml`**

In `[package]`:

```toml
readme = "README.md"
rust-version = "1.85"
```

(`1.85` is the first stable release supporting `edition = "2024"`; the pixi `lint` toolchain is `1.93.x`, comfortably above it.)

- [ ] **Step 3: Add a combined encode→decode round-trip proptest**

Append to `svar2-codec/src/lib.rs`'s `mod tests` — one property that exercises every non-spill lane through the public API (spill is `genoray`'s concern since it needs the bank):

```rust
    proptest! {
        #[test]
        fn key_round_trip_all_inline_lanes(kind in 0u8..3, ilen in 1i32..=1_000_000) {
            match kind {
                // SNP
                0 => {
                    let base = b'C';
                    let key = encode_alt_inline(&[base], 0);
                    prop_assert_eq!(decode_key(key), DecodedKey::Inline { alt: vec![base] });
                }
                // pure DEL
                1 => {
                    let key = encode_pure_del(-ilen);
                    prop_assert_eq!(deletion_len(key), ilen as u32);
                    prop_assert_eq!(decode_key(key), DecodedKey::PureDel { ilen: -ilen });
                }
                // lookup
                _ => {
                    let row = (ilen as u32) & ((1 << 31) - 1);
                    prop_assert_eq!(decode_key(encode_lookup(row)), DecodedKey::Lookup { row });
                }
            }
        }
    }
```

- [ ] **Step 4: Verify tests and the publish dry-run**

Run: `pixi run -e lint cargo test -p svar2-codec`
Expected: PASS.

Run: `pixi run -e lint cargo publish --dry-run -p svar2-codec`
Expected: `Packaging svar2-codec v0.1.0`, `Verifying svar2-codec v0.1.0`, compiles clean, no path-dependency error (the crate has none). If it warns about uncommitted changes, that is expected mid-task and resolved by the commit below.

- [ ] **Step 5: Commit**

```bash
git add svar2-codec/Cargo.toml svar2-codec/README.md svar2-codec/src/lib.rs
git commit -m "build(svar2-codec): add crates.io metadata, README, and round-trip contract test"
```

---

### Task 7: Reconcile roadmap + supplements

**Files:**
- Modify: `docs/roadmap/svar-2.md` (M5 → done; M6 core reflects the codec landing)
- Modify: `docs/roadmap/architecture.md` (the encoding-seam section names `svar2-codec`)
- Modify: `docs/roadmap/data-model.md` (the inline-encoding section names `svar2-codec`)

**Interfaces:**
- Consumes: nothing.
- Produces: docs consistent with the shipped code (the roadmap's working agreement requires this in the same PR).

- [ ] **Step 1: Mark M5 done and update M6 in `docs/roadmap/svar-2.md`**

Change the M5 checkbox from `[~]` to `[x]` and update its *Remaining* note to *Done*, reflecting that `src/query.rs` (`ContigReader`/`overlap_sample`) shipped the disk-integrated `(range, sample)` query — `max_del` consumer, the sorted sub-stream union (`kway_merge`), per-`(contig, sample, ploid)` wiring, and the genotype gather.

In the M6 bullet, replace the "Finish M5's disk integration" clause with a note that M5's query landed, and that M6's decode core now consists of: the `svar2-codec` crate extraction (this plan, done), then the two-channel/materialized consumer interfaces (M6b/M6c). Add a one-line pointer to this plan:
`docs/superpowers/plans/2026-07-02-svar2-codec-crate.md`.

- [ ] **Step 2: Name `svar2-codec` in `architecture.md`**

In "The encoding-agnostic seam" section, add a sentence: the encode/decode layer is realized as the standalone **`svar2-codec`** crate (crates.io-published, std-only), so the single place that knows the bit layout is a linkable unit shared by the converter and by downstream Rust consumers.

- [ ] **Step 3: Name `svar2-codec` in `data-model.md`**

In "Inline variant encoding (`var_key`)", add a sentence noting the encode/decode of these layouts lives in `svar2-codec` (the co-located single source of truth for both halves).

- [ ] **Step 4: Verify docs render / links resolve**

Run: `git diff --stat docs/`
Expected: three files changed. Eyeball that the M5 checkbox is `[x]` and the plan link path is correct.

- [ ] **Step 5: Commit**

```bash
git add docs/roadmap/svar-2.md docs/roadmap/architecture.md docs/roadmap/data-model.md
git commit -m "docs(svar-2): reconcile M5 done + svar2-codec seam crate"
```

---

## Self-Review

**Spec coverage:**
- svar2-codec crate created, pure/leaf, crates.io-ready → Tasks 1, 6.
- Decode seam (`decode_key`, `decode_snp_2bit`, `unpack_snp_key_at`, `unpack_snp_keys`, `decode_alt_inline`, `deletion_len`, `DecodedKey`) moved → Tasks 3, 5.
- Encode co-location (`encode_snp_2bit`, `pack_snp_keys`, inline encoder, `encode_pure_del`, `encode_lookup`) moved so no bit-position stays in `genoray` → Tasks 3, 4, 5.
- Constants + unified `BASES` moved → Task 2.
- All call sites keep compiling via `rvk`/`types` re-export shims → Tasks 2, 3, 5 (verified by the full-suite runs).
- Parity guarded by migrated PEXT/SWAR + `deletion_len` proptests and new round-trip tests → Tasks 4, 5, 6.
- Publish-readiness dry-run → Task 6.
- Roadmap/supplement reconciliation (required by the working agreement) → Task 7.
- Out of scope, called out: crates.io/PyPI publishing automation, the uniform-key re-expansion (M6b), the two-channel + Python interfaces (M6b/M6c).

**Placeholder scan:** none — every step carries real code or a concrete command. The two "adapt to the real test's name/strategy" notes (Task 4 Step 2) are because the exact proptest identifier lives in `rvk.rs` and is relocated, not authored; the shape is given.

**Type consistency:** `DecodedKey` variants (`Inline{alt}`, `PureDel{ilen}`, `Lookup{row}`) match across Tasks 5-6 and the existing `query.rs`/`max_del.rs` consumers. `encode_alt_inline(&[u8], u32) -> u32`, `encode_pure_del(i32) -> u32`, `encode_lookup(u32) -> u32`, `deletion_len(u32) -> u32`, `decode_key(u32) -> DecodedKey` are used identically wherever referenced. Re-export lists in `rvk.rs` are additive and consistent between Task 3 and Task 5.

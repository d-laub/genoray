# SVAR 2.0 M4 — Dense Representation + Cost-Model Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the 1-bit dense genotype representation and a deterministic per-variant cost model that routes each variant to `var_key` or `dense`, whichever is cheaper on disk.

**Architecture:** A pure `cost_model` leaf decides `VarKey` vs `Dense` from the variant's allele count (popcount of its plane, known within one chunk). The executor's transpose routes each set bit into either the existing per-call `var_key` streams or a new per-class hap-major dense bit block + variant table. A rectangular `dense_merge` bit-concatenates per-hap rows across chunks into `dense/{snp,indel}/{final_positions,final_keys,final_genotypes}.bin`. The long-allele LUT becomes a single shared per-contig indel table.

**Tech Stack:** Rust 2024 edition, `crossbeam-channel`, `rayon`, `ndarray`/`ndarray-npy`, `bytemuck`, `proptest`+`tempfile` (dev). Design spec: `docs/superpowers/specs/2026-07-01-svar2-m4-dense-cost-model-design.md`.

## Global Constraints

- **Encoding-agnostic seam:** orchestration/writer/merge must not branch on key bit layout; only `rvk.rs` (encode/decode) knows it. Route by class/representation tag, treat keys as opaque fixed-width bytes.
- **`layout.rs` is the single source of truth** for every on-disk path. No stringly-typed paths elsewhere.
- **Cost is computed in bits** (integer arithmetic, no floats): `POS_BITS = 32`; `key_bits(Snp) = 2`, `key_bits(Indel) = 32`; dense bitmask = `np` bits.
- **Tie-break:** equal cost → `VarKey` (deterministic).
- **Platform:** Linux/macOS (`std::os::unix::fs::FileExt` `pread`/`pwrite` already used in `merge.rs`).
- **Bit convention (dense geno blocks):** bit `i` lives at byte `i >> 3`, position `i & 7`, **LSB-first** — the same convention `bits::set_bit`/`copy_bits` use everywhere.
- **Test command:** `pixi run -e lint cargo test --no-default-features <filter>` (the `lint` env carries the rust toolchain + `LIBCLANG_PATH`; default features pull `pyo3/extension-module`, which won't link a test binary). Lint gate before commit: `pixi run -e lint cargo clippy --all-targets -- -D warnings` and `pixi run -e lint cargo fmt --all`.
- **Doc working agreement:** the doc-reconciliation task (Task 12) is REQUIRED, not optional.

---

## File structure

| File | Responsibility | Task |
| --- | --- | --- |
| `src/cost_model.rs` (new) | pure `choose_representation` + `Class`/`Representation` + constants | 1 |
| `src/types.rs` (mod) | `BitGrid3::popcount_plane`; `DenseSubChunk`; `SparseChunk.dense` field | 2, 5 |
| `src/bits.rs` (new) | `set_bit`, `get_bit`, `copy_bits` bit-slice helpers | 4 |
| `src/dense.rs` (new) | `DenseClass`, `DENSE_REGISTRY`, `DenseMap<T>` | 5 |
| `src/layout.rs` (mod) | dense dirs + `chunk_geno`/`final_genotypes`; relocate shared LUT | 3, 6-prep |
| `src/nrvk.rs` (mod) | `LongAlleleReader` reads shared LUT path | 3 |
| `src/rvk.rs` (mod) | routing pre-pass + dense payload build in `dense2sparse_vk` | 10 |
| `src/writer.rs` (mod) | write dense per-chunk `pos`/`key`/`geno` files | 7 |
| `src/executor.rs` (mod) | accumulate dense ledgers; return `Phase1Output` | 8 |
| `src/dense_merge.rs` (new) | rectangular dense merge (concat + bit-transpose + snp pack) | 9 |
| `src/orchestrator.rs` (mod) | create dense dirs; drive dense merge after Phase 1 | 11 |
| `tests/test_e2e.rs` (mod) | update var_key expectations + dense round-trip e2e | 11 |
| `src/lib.rs` (mod) | `pub mod cost_model/bits/dense/dense_merge;` | 1, 4, 5, 9 |
| `docs/roadmap/{data-model,architecture,svar-2}.md` | reconcile with implemented decisions | 12 |

**Sequencing rationale (green at every task):** Tasks 1–9 add code paths that operate on *empty* dense payloads (routing is still off), so all existing tests keep passing. Task 10 flips routing on and updates the existing e2e assertions in the same task (routing legitimately changes output). Task 11 adds new dense round-trip coverage. Task 12 reconciles docs.

---

### Task 1: Cost model (`cost_model.rs`)

**Files:**
- Create: `src/cost_model.rs`
- Modify: `src/lib.rs` (add `pub mod cost_model;` after `pub mod budget;`)
- Test: in-source `#[cfg(test)]` in `src/cost_model.rs`

**Interfaces:**
- Produces:
  - `pub enum Class { Snp, Indel }` (derives `Debug, Clone, Copy, PartialEq, Eq`)
  - `pub enum Representation { VarKey, Dense }` (same derives)
  - `pub fn key_bits(class: Class) -> u64`
  - `pub fn choose_representation(class: Class, n_samples: usize, ploidy: usize, x_calls: usize) -> Representation`
  - `pub const POS_BITS: u64 = 32;`

- [ ] **Step 1: Write the failing tests**

Create `src/cost_model.rs`:

```rust
//! Per-variant dense-vs-sparse routing. Pure leaf: no I/O, all integer bit
//! arithmetic (no floats, so the crossover is exact and reproducible).
//! Costs are the *actual on-disk bits* one variant occupies in each
//! representation (see docs/roadmap/data-model.md#dense-vs-sparse-cost-model).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Snp,
    Indel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    VarKey,
    Dense,
}

/// Per-call u32 position, in bits. Paid once per call in `var_key`, once per
/// variant in `dense`.
pub const POS_BITS: u64 = 32;

/// Key width in bits by class: SNP is a 2-bit ALT code, indel a 32-bit key.
#[inline]
pub fn key_bits(class: Class) -> u64 {
    match class {
        Class::Snp => 2,
        Class::Indel => 32,
    }
}

/// Choose the cheaper representation for one variant with `x_calls` carriers.
///
/// var_key = x · (POS_BITS + key_bits)     (position + key inlined per call)
/// dense   = POS_BITS + key_bits + np      (table row once + 1-bit-per-hap mask)
///
/// Route to `Dense` iff strictly cheaper; ties break to `VarKey`.
#[inline]
pub fn choose_representation(
    class: Class,
    n_samples: usize,
    ploidy: usize,
    x_calls: usize,
) -> Representation {
    let np = (n_samples as u64) * (ploidy as u64);
    let per_call = POS_BITS + key_bits(class);
    let var_key_bits = (x_calls as u64) * per_call;
    let dense_bits = POS_BITS + key_bits(class) + np;
    if dense_bits < var_key_bits {
        Representation::Dense
    } else {
        Representation::VarKey
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x_zero_is_var_key() {
        // No carriers: var_key costs 0, dense costs a full row — var_key wins.
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 0),
            Representation::VarKey
        );
    }

    #[test]
    fn test_snp_crossover_np2000() {
        // np=2000: dense=32+2+2000=2034; var_key=34x. Dense wins when 34x > 2034
        // → x >= 60 (34*59=2006 < 2034 → var_key; 34*60=2040 > 2034 → dense).
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 59),
            Representation::VarKey
        );
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 60),
            Representation::Dense
        );
    }

    #[test]
    fn test_indel_crossover_np2000() {
        // np=2000: dense=32+32+2000=2064; var_key=64x. Dense when 64x > 2064
        // → x >= 33 (64*32=2048 < 2064; 64*33=2112 > 2064).
        assert_eq!(
            choose_representation(Class::Indel, 1000, 2, 32),
            Representation::VarKey
        );
        assert_eq!(
            choose_representation(Class::Indel, 1000, 2, 33),
            Representation::Dense
        );
    }

    #[test]
    fn test_tie_breaks_to_var_key() {
        // Construct an exact tie: dense_bits == var_key_bits → VarKey.
        // Snp np=32: dense=32+2+32=66; per_call=34. No integer x makes 34x==66,
        // so use a case where dense==var_key: pick class/np so equality holds.
        // Indel np=32: dense=32+32+32=96; per_call=64; 64*x==96 has no int soln.
        // Use np such that dense is a multiple of per_call:
        // Snp per_call=34, choose np so 32+2+np = 34*k. np=34*2-34=34 → dense=34+34=... 
        // Simpler: assert the boundary just-below stays VarKey (covered above) and
        // that a hand-built equal case resolves to VarKey via direct formula check.
        let np = 34u64 * 3 - 34; // = 68 → dense_bits = 34 + 68 = 102 = 34*3
        let n = np as usize; // ploidy 1
        assert_eq!(
            choose_representation(Class::Snp, n, 1, 3), // var_key = 34*3 = 102 == dense
            Representation::VarKey,
            "exact tie must resolve to VarKey"
        );
    }

    proptest::proptest! {
        // Monotonic: once Dense wins at x, it wins for every larger x.
        #[test]
        fn test_monotonic_in_x(
            n in 1usize..5000,
            ploidy in 1usize..3,
            x in 0usize..20000,
        ) {
            let here = choose_representation(Class::Snp, n, ploidy, x);
            if here == Representation::Dense {
                let more = choose_representation(Class::Snp, n, ploidy, x + 1);
                prop_assert_eq!(more, Representation::Dense);
            }
        }
    }
}
```

Add to `src/lib.rs` after line `pub mod budget;`:

```rust
pub mod cost_model;
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features cost_model`
Expected: FAIL — `src/cost_model.rs` does not yet compile against `lib.rs` until the `pub mod` line is added; once added, tests compile and pass (this module is written complete in Step 1).

> Note: this module is small and self-contained, so Step 1 already contains the full implementation. If you prefer strict red-green, temporarily stub `choose_representation` to `Representation::VarKey`, watch `test_snp_crossover_np2000` fail, then restore.

- [ ] **Step 3: Confirm implementation present** — the body in Step 1 is the implementation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features cost_model`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/cost_model.rs src/lib.rs
git commit -m "feat(svar-2): cost model for per-variant dense/sparse routing"
```

---

### Task 2: `BitGrid3::popcount_plane` (`types.rs`)

**Files:**
- Modify: `src/types.rs` (add method in `impl BitGrid3`, after `truncate_v`)
- Test: in-source `#[cfg(test)]` in `src/types.rs`

**Interfaces:**
- Consumes: `BitGrid3 { words: Vec<u64>, shape: (usize,usize,usize) }`
- Produces: `pub fn popcount_plane(&self, v: usize) -> usize` — number of set bits in variant `v`'s `(S,P)` plane (its allele count `x`).

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/types.rs`:

```rust
#[test]
fn test_popcount_plane_counts_set_bits_in_variant() {
    // shape (3, 2, 2): plane size = S*P = 4 bits per variant.
    let mut g = BitGrid3::zeros(3, 2, 2);
    // variant 0: set 2 bits (flat 0..4)
    g.or_bit(0, true);
    g.or_bit(3, true);
    // variant 1: set 4 bits (flat 4..8)
    for i in 4..8 {
        g.or_bit(i, true);
    }
    // variant 2: set 0 bits (flat 8..12)
    assert_eq!(g.popcount_plane(0), 2);
    assert_eq!(g.popcount_plane(1), 4);
    assert_eq!(g.popcount_plane(2), 0);
}

proptest! {
    // popcount_plane equals a naive per-bit count for arbitrary shapes/patterns.
    #[test]
    fn test_popcount_plane_matches_naive(
        v in 1usize..12,
        s in 1usize..8,
        p in 1usize..3,
        seed in any::<u64>(),
    ) {
        let mut bg = BitGrid3::zeros(v, s, p);
        let plane = s * p;
        let mut expected = vec![0usize; v];
        let mut state = seed | 1;
        for idx in 0..(v * plane) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let val = state & 1 != 0;
            bg.or_bit(idx, val);
            if val { expected[idx / plane] += 1; }
        }
        for vi in 0..v {
            prop_assert_eq!(bg.popcount_plane(vi), expected[vi], "variant {}", vi);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e lint cargo test --no-default-features popcount_plane`
Expected: FAIL — `no method named popcount_plane`.

- [ ] **Step 3: Implement the method**

Add inside `impl BitGrid3` in `src/types.rs`, after `truncate_v`:

```rust
/// Number of set bits in variant `v`'s (S, P) plane — its allele count `x`.
/// The plane is the contiguous bit range `[v*S*P, (v+1)*S*P)`; counted
/// word-wise (masked head/tail) so it is O(plane_bits / 64), not per-bit.
pub fn popcount_plane(&self, v: usize) -> usize {
    let (_, s, p) = self.shape;
    let plane = s * p;
    let start = v * plane;
    let end = start + plane;
    debug_assert!(end <= self.shape.0 * plane);

    let mut count = 0u32;
    let mut bit = start;
    while bit < end {
        let word_idx = bit >> 6;
        let bit_in_word = bit & 63;
        // bits available in this word from `bit_in_word` to end of word (64)
        let word_end_bit = (word_idx + 1) << 6;
        let span_end = end.min(word_end_bit);
        let take = span_end - bit; // 1..=64 bits from this word
        let word = self.words[word_idx];
        // mask the [bit_in_word, bit_in_word+take) window
        let mask = if take == 64 {
            u64::MAX
        } else {
            ((1u64 << take) - 1) << bit_in_word
        };
        count += (word & mask).count_ones();
        bit = span_end;
    }
    count as usize
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features popcount_plane`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/types.rs
git commit -m "feat(svar-2): BitGrid3::popcount_plane for allele-count routing"
```

---

### Task 3: Relocate long-allele LUT to a shared per-contig path (`layout.rs`, `nrvk.rs`)

**Files:**
- Modify: `src/layout.rs` (`ContigPaths::long_alleles_bin` / `long_allele_offsets` → shared `{contig}/indel/` dir; update tests)
- Modify: `src/nrvk.rs` (doc comment + test path)
- Test: existing `src/layout.rs` and `src/nrvk.rs` in-source tests (updated)

**Interfaces:**
- Produces: `ContigPaths::long_alleles_bin()` → `{out}/{chrom}/indel/long_alleles.bin`; `long_allele_offsets()` → `{out}/{chrom}/indel/long_allele_offsets.npy`. (Both var_key and dense indel reference this single table.)

- [ ] **Step 1: Update the failing tests to the new path**

In `src/layout.rs`, add a shared-indel dir helper and repoint the LUT. Replace the `long_alleles_bin`/`long_allele_offsets` methods:

```rust
    /// Shared per-contig indel long-allele LUT dir. Both var_key/indel and
    /// dense/indel reference this single table (spilled keys are
    /// representation-portable).
    pub fn shared_indel_dir(&self) -> PathBuf {
        Path::new(&self.base_out_dir).join(&self.chrom).join("indel")
    }
    pub fn long_alleles_bin(&self) -> PathBuf {
        self.shared_indel_dir().join("long_alleles.bin")
    }
    pub fn long_allele_offsets(&self) -> PathBuf {
        self.shared_indel_dir().join("long_allele_offsets.npy")
    }
```

Update `src/layout.rs` test `test_long_allele_paths_live_under_indel` expectations:

```rust
    #[test]
    fn test_long_allele_paths_live_under_shared_indel() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(
            p.long_alleles_bin(),
            Path::new("/out/chr1/indel/long_alleles.bin")
        );
        assert_eq!(
            p.long_allele_offsets(),
            Path::new("/out/chr1/indel/long_allele_offsets.npy")
        );
    }
```

Update `src/nrvk.rs`:
- Doc comment on `LongAlleleReader::new` (line ~107): change the layout note to `{output_dir}/{chrom}/indel/{long_alleles.bin, long_allele_offsets.npy}`.
- Test `test_reader_get_allele_shared_borrow`: change `let dir = tmp.path().join("chr1").join("var_key").join("indel");` to `let dir = tmp.path().join("chr1").join("indel");`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features -- layout nrvk`
Expected: FAIL on the two updated tests until the impl (Step 1's method edits) is in — since Step 1 edits both impl and tests together, run to confirm they now reflect the shared path.

> The executor already creates the LUT via `ContigPaths::long_alleles_bin()`, so no orchestrator change is needed — the writer follows `paths.long_alleles_bin()`. Only the physical directory moves.

- [ ] **Step 3: Ensure the shared dir is created**

In `src/orchestrator.rs`, the long-allele writer opens `paths.long_alleles_bin()`. That parent dir must exist. The existing `stream_dirs` loop creates `var_key/{snp,indel}` but not `{contig}/indel`. Add, right after the `for (_, dir) in stream_dirs.iter()` creation loop (around line 72):

```rust
    // Shared per-contig indel LUT dir (long alleles for var_key + dense indels).
    fs::create_dir_all(paths.shared_indel_dir()).map_err(|e| ConversionError::Io {
        context: format!("create_dir_all {:?}", paths.shared_indel_dir()),
        source: e,
    })?;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: PASS — all suites, including `test_e2e` (the e2e doesn't assert the LUT path, and the LUT dir now exists).

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/layout.rs src/nrvk.rs src/orchestrator.rs
git commit -m "refactor(svar-2): relocate long-allele LUT to shared per-contig indel dir"
```

---

### Task 4: Bit-slice helpers (`bits.rs`)

**Files:**
- Create: `src/bits.rs`
- Modify: `src/lib.rs` (add `pub mod bits;`)
- Test: in-source `#[cfg(test)]` in `src/bits.rs`

**Interfaces:**
- Produces:
  - `pub fn set_bit(buf: &mut [u8], idx: usize)` — set bit `idx` (LSB-first byte convention).
  - `pub fn get_bit(buf: &[u8], idx: usize) -> bool`
  - `pub fn copy_bits(dst: &mut [u8], dst_bit: usize, src: &[u8], src_bit: usize, n: usize)` — copy `n` bits.

- [ ] **Step 1: Write the failing tests**

Create `src/bits.rs`:

```rust
//! LSB-first bit-slice helpers for the dense genotype matrix. Bit `i` lives at
//! byte `i >> 3`, position `i & 7`. `copy_bits` is the single hot spot for the
//! dense merge's per-hap bit-concatenation (see dense_merge.rs); it is kept
//! isolated so it can be optimized in place later (see M4 open questions).

#[inline(always)]
pub fn set_bit(buf: &mut [u8], idx: usize) {
    buf[idx >> 3] |= 1u8 << (idx & 7);
}

#[inline(always)]
pub fn get_bit(buf: &[u8], idx: usize) -> bool {
    (buf[idx >> 3] >> (idx & 7)) & 1 != 0
}

/// Copy `n` bits from `src` (starting at bit `src_bit`) into `dst` (starting at
/// bit `dst_bit`). Bits already set in the untouched region of `dst` are
/// preserved; the target window is OR-written from a zeroed assumption
/// (callers pass a zeroed destination window).
pub fn copy_bits(dst: &mut [u8], dst_bit: usize, src: &[u8], src_bit: usize, n: usize) {
    for i in 0..n {
        if get_bit(src, src_bit + i) {
            set_bit(dst, dst_bit + i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_set_get_roundtrip_across_byte_boundary() {
        let mut b = vec![0u8; 2];
        set_bit(&mut b, 0);
        set_bit(&mut b, 7);
        set_bit(&mut b, 8);
        set_bit(&mut b, 15);
        assert!(get_bit(&b, 0) && get_bit(&b, 7) && get_bit(&b, 8) && get_bit(&b, 15));
        assert!(!get_bit(&b, 1));
        assert_eq!(b, vec![0b1000_0001u8, 0b1000_0001u8]);
    }

    #[test]
    fn test_copy_bits_unaligned() {
        // src bits [3..3+5) = pattern; copy to dst starting at bit 6.
        let mut src = vec![0u8; 2];
        // set src bits 3,4,6,7 (within the 5-bit window [3,8))
        for &i in &[3usize, 4, 6, 7] {
            set_bit(&mut src, i);
        }
        let mut dst = vec![0u8; 2];
        copy_bits(&mut dst, 6, &src, 3, 5); // copies src[3..8] → dst[6..11]
        // src window bits (offset j in 0..5): j=0(src3)=1, j=1(src4)=1, j=2(src5)=0,
        // j=3(src6)=1, j=4(src7)=1 → dst bits 6,7,9,10 set; dst 8 clear.
        for &(i, want) in &[(6usize, true), (7, true), (8, false), (9, true), (10, true)] {
            assert_eq!(get_bit(&dst, i), want, "dst bit {}", i);
        }
    }

    proptest! {
        // copy_bits matches a naive reference for arbitrary offsets/lengths.
        #[test]
        fn test_copy_bits_matches_reference(
            src_bytes in proptest::collection::vec(any::<u8>(), 1..8),
            dst_bit in 0usize..40,
            src_bit in 0usize..40,
            n in 0usize..40,
        ) {
            // ensure buffers are big enough
            let need_src = (src_bit + n).div_ceil(8).max(src_bytes.len());
            let mut src = src_bytes.clone();
            src.resize(need_src, 0);
            let dst_len = (dst_bit + n).div_ceil(8).max(1);
            let mut dst = vec![0u8; dst_len];
            let mut reference = vec![0u8; dst_len];

            copy_bits(&mut dst, dst_bit, &src, src_bit, n);
            for i in 0..n {
                if get_bit(&src, src_bit + i) {
                    set_bit(&mut reference, dst_bit + i);
                }
            }
            prop_assert_eq!(dst, reference);
        }
    }
}
```

Add to `src/lib.rs` (after `pub mod budget;` block):

```rust
pub mod bits;
```

- [ ] **Step 2: Run tests to verify they fail, then pass**

Run: `pixi run -e lint cargo test --no-default-features bits`
Expected: PASS once `pub mod bits;` is added (module is complete in Step 1). To see red first, stub `copy_bits` body to `{}` and watch `test_copy_bits_unaligned` fail, then restore.

- [ ] **Step 3: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/bits.rs src/lib.rs
git commit -m "feat(svar-2): LSB-first bit-slice helpers for dense matrix"
```

---

### Task 5: Dense class registry + chunk payload types (`dense.rs`, `types.rs`)

**Files:**
- Create: `src/dense.rs`
- Modify: `src/types.rs` (add `DenseSubChunk`; add `dense` field to `SparseChunk`)
- Modify: `src/rvk.rs` (`dense2sparse_vk` initializes the new `dense` field empty — no routing yet)
- Modify: `src/lib.rs` (`pub mod dense;`)
- Test: in-source tests in `src/dense.rs`

**Interfaces:**
- Produces:
  - `dense::DenseClass { Snp, Indel }` with `COUNT: usize = 2`, `ALL: [DenseClass; 2]`, `index(self) -> usize`, `key_bytes(self) -> usize` (Snp→1, Indel→4), `cost_class(self) -> crate::cost_model::Class`.
  - `dense::DenseSpec { class, subdir: &'static str, key_bytes: usize, pack_snp: bool }` and `dense::DENSE_REGISTRY: [DenseSpec; 2]`.
  - `dense::DenseMap<T>` with `from_fn`, `get`, `get_mut`, `iter`, `into_iter_tagged` (mirrors `StreamMap`).
  - `types::DenseSubChunk { key_bytes: usize, n_dense_variants: usize, positions: Vec<u32>, keys: Vec<u8>, geno_bits: Vec<u8> }` + `DenseSubChunk::empty(key_bytes: usize) -> Self`.
  - `types::SparseChunk` gains `pub dense: crate::dense::DenseMap<DenseSubChunk>`.

- [ ] **Step 1: Write `dense.rs` with tests**

Create `src/dense.rs`:

```rust
//! Dense-representation class registry. Mirrors `streams.rs` but for the
//! *per-variant* dense matrix (not per-call streams). Two classes: 2-bit SNP
//! (packed post-merge, no LUT) and 32-bit indel (shares the per-contig LUT).

use crate::cost_model::Class;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseClass {
    Snp = 0,
    Indel = 1,
}

impl DenseClass {
    pub const COUNT: usize = 2;
    pub const ALL: [DenseClass; Self::COUNT] = [DenseClass::Snp, DenseClass::Indel];
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
    #[inline]
    pub fn key_bytes(self) -> usize {
        match self {
            DenseClass::Snp => 1,   // one raw 2-bit code per variant (packed at merge)
            DenseClass::Indel => 4, // u32 key per variant
        }
    }
    #[inline]
    pub fn cost_class(self) -> Class {
        match self {
            DenseClass::Snp => Class::Snp,
            DenseClass::Indel => Class::Indel,
        }
    }
}

pub struct DenseSpec {
    pub class: DenseClass,
    pub subdir: &'static str,
    pub key_bytes: usize,
    /// 2-bit-pack the merged key file (SNP only).
    pub pack_snp: bool,
}

pub const DENSE_REGISTRY: [DenseSpec; DenseClass::COUNT] = [
    DenseSpec {
        class: DenseClass::Snp,
        subdir: "dense/snp",
        key_bytes: 1,
        pack_snp: true,
    },
    DenseSpec {
        class: DenseClass::Indel,
        subdir: "dense/indel",
        key_bytes: 4,
        pack_snp: false,
    },
];

/// Fixed-size map keyed by `DenseClass`, array-backed (O(1), no hashing).
pub struct DenseMap<T> {
    slots: [T; DenseClass::COUNT],
}

impl<T> DenseMap<T> {
    pub fn from_fn(f: impl FnMut(DenseClass) -> T) -> Self {
        Self {
            slots: DenseClass::ALL.map(f),
        }
    }
    #[inline]
    pub fn get(&self, c: DenseClass) -> &T {
        &self.slots[c.index()]
    }
    #[inline]
    pub fn get_mut(&mut self, c: DenseClass) -> &mut T {
        &mut self.slots[c.index()]
    }
    pub fn iter(&self) -> impl Iterator<Item = (DenseClass, &T)> {
        DenseClass::ALL.into_iter().zip(self.slots.iter())
    }
    pub fn into_iter_tagged(self) -> impl Iterator<Item = (DenseClass, T)> {
        DenseClass::ALL.into_iter().zip(self.slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_indices_match_classes() {
        for spec in &DENSE_REGISTRY {
            assert_eq!(DENSE_REGISTRY[spec.class.index()].class, spec.class);
        }
    }

    #[test]
    fn test_key_bytes() {
        assert_eq!(DenseClass::Snp.key_bytes(), 1);
        assert_eq!(DenseClass::Indel.key_bytes(), 4);
    }

    #[test]
    fn test_densemap_get_set() {
        let mut m: DenseMap<u32> = DenseMap::from_fn(|_| 0);
        *m.get_mut(DenseClass::Indel) = 9;
        assert_eq!(*m.get(DenseClass::Snp), 0);
        assert_eq!(*m.get(DenseClass::Indel), 9);
    }
}
```

- [ ] **Step 2: Add `DenseSubChunk` + `SparseChunk.dense` to `types.rs`**

In `src/types.rs`, after the `SparseSubStream` impl block (before `SparseChunk`), add:

```rust
// Per-class dense payload for one chunk: the hap-major 1-bit genotype block
// plus the per-variant table (positions + keys). Built by `dense2sparse_vk`,
// consumed by the writer, transposed/concatenated by dense_merge.
pub struct DenseSubChunk {
    pub key_bytes: usize,
    pub n_dense_variants: usize,
    pub positions: Vec<u32>, // len == n_dense_variants
    pub keys: Vec<u8>,       // key_bytes * n_dense_variants
    // Hap-major bit block, shape (S, P, n_dense_variants), variant fastest.
    // Bit (hap h, dense col d) at flat index h*n_dense_variants + d.
    pub geno_bits: Vec<u8>,
}

impl DenseSubChunk {
    pub fn empty(key_bytes: usize) -> Self {
        Self {
            key_bytes,
            n_dense_variants: 0,
            positions: Vec::new(),
            keys: Vec::new(),
            geno_bits: Vec::new(),
        }
    }
}
```

Change the `SparseChunk` struct to add the `dense` field:

```rust
pub struct SparseChunk {
    pub chunk_id: usize,
    pub streams: StreamMap<SparseSubStream>,
    pub dense: crate::dense::DenseMap<DenseSubChunk>,
}
```

- [ ] **Step 3: Initialize `dense` empty in `dense2sparse_vk`**

In `src/rvk.rs`, at the end of `dense2sparse_vk` where `SparseChunk { chunk_id, streams }` is constructed, change to:

```rust
    SparseChunk {
        chunk_id: chunk.chunk_id,
        streams,
        dense: crate::dense::DenseMap::from_fn(|c| {
            crate::types::DenseSubChunk::empty(c.key_bytes())
        }),
    }
```

Add `pub mod dense;` to `src/lib.rs` (after `pub mod cost_model;`).

- [ ] **Step 4: Run tests to verify everything compiles + passes**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: PASS — `dense` tests pass; all existing `rvk.rs`/e2e tests still pass because the `dense` field is empty and untouched by the transpose. (The existing `dense2sparse` tests read `sparse.streams` only.)

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/dense.rs src/types.rs src/rvk.rs src/lib.rs
git commit -m "feat(svar-2): dense class registry + empty dense chunk payload"
```

---

### Task 6: Dense on-disk paths (`layout.rs`)

**Files:**
- Modify: `src/layout.rs` (dense dirs on `ContigPaths`; `chunk_geno`, `final_genotypes`)
- Test: in-source tests in `src/layout.rs`

**Interfaces:**
- Produces:
  - `ContigPaths::dense_snp_dir()` → `{out}/{chrom}/dense/snp`; `dense_indel_dir()` → `{out}/{chrom}/dense/indel`.
  - `layout::chunk_geno(dir, id)` → `{dir}/chunk_{id}_geno.bin`.
  - `layout::final_genotypes(dir)` → `{dir}/final_genotypes.bin`.
  - (Reuses existing `chunk_pos`, `chunk_key`, `final_positions`, `final_keys` for dense pos/key files.)

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/layout.rs`:

```rust
    #[test]
    fn test_dense_dirs() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(p.dense_snp_dir(), Path::new("/out/chr1/dense/snp"));
        assert_eq!(p.dense_indel_dir(), Path::new("/out/chr1/dense/indel"));
    }

    #[test]
    fn test_dense_chunk_and_final_names() {
        let dir = Path::new("/out/chr1/dense/snp");
        assert_eq!(chunk_geno(dir, 2), Path::new("/out/chr1/dense/snp/chunk_2_geno.bin"));
        assert_eq!(
            final_genotypes(dir),
            Path::new("/out/chr1/dense/snp/final_genotypes.bin")
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features -- layout`
Expected: FAIL — `dense_snp_dir`/`chunk_geno`/`final_genotypes` undefined.

- [ ] **Step 3: Implement**

In `src/layout.rs`, add methods to `impl ContigPaths` (near `var_key_snp_dir`):

```rust
    pub fn dense_snp_dir(&self) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("dense")
            .join("snp")
    }
    pub fn dense_indel_dir(&self) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("dense")
            .join("indel")
    }
```

Add free functions (near `chunk_key`/`final_keys`):

```rust
pub fn chunk_geno(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_geno.bin", chunk_id))
}
pub fn final_genotypes(dir: &Path) -> PathBuf {
    dir.join("final_genotypes.bin")
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features -- layout`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/layout.rs
git commit -m "feat(svar-2): dense on-disk path helpers (dirs, geno, final)"
```

---

### Task 7: Writer emits dense per-chunk files (`writer.rs`)

**Files:**
- Modify: `src/writer.rs` (`run_io_writer` signature + dense writing)
- Modify: `src/orchestrator.rs` (build + pass dense dirs to the writer)
- Test: in-source `#[cfg(test)]` in `src/writer.rs`

**Interfaces:**
- Consumes: `SparseChunk.dense: DenseMap<DenseSubChunk>`, `dense::DENSE_REGISTRY`, `layout::{chunk_pos,chunk_key,chunk_geno}`.
- Produces: `pub fn run_io_writer(rx_sparse: Receiver<SparseChunk>, dirs: StreamMap<PathBuf>, dense_dirs: DenseMap<PathBuf>)` — now writes var_key (as before) **and** dense per-chunk `pos`/`key`/`geno` files for every class with `n_dense_variants > 0`.

- [ ] **Step 1: Write the failing test**

Add a `tests` module to `src/writer.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::{DenseClass, DenseMap};
    use crate::streams::{StreamMap, StreamTag};
    use crate::types::{DenseSubChunk, SparseChunk, SparseSubStream};
    use crossbeam_channel::bounded;
    use tempfile::tempdir;

    #[test]
    fn test_writer_persists_dense_chunk_files() {
        let tmp = tempdir().unwrap();
        // dense/snp dir with 2 dense variants, np=2 → geno = ceil(2*2/8)=1 byte.
        let snp_dir = tmp.path().join("dense/snp");
        let indel_dir = tmp.path().join("dense/indel");
        std::fs::create_dir_all(&snp_dir).unwrap();
        std::fs::create_dir_all(&indel_dir).unwrap();

        // var_key dirs (empty streams, still iterated by the writer)
        let vk_snp = tmp.path().join("var_key/snp");
        let vk_indel = tmp.path().join("var_key/indel");
        std::fs::create_dir_all(&vk_snp).unwrap();
        std::fs::create_dir_all(&vk_indel).unwrap();

        let mut dense = DenseMap::from_fn(|c| DenseSubChunk::empty(c.key_bytes()));
        let snp = dense.get_mut(DenseClass::Snp);
        snp.n_dense_variants = 2;
        snp.positions = vec![100, 200];
        snp.keys = vec![1u8, 2u8]; // 2 raw codes
        snp.geno_bits = vec![0b0000_1011u8]; // arbitrary

        let streams = StreamMap::from_fn(|tag| {
            let kb = crate::streams::REGISTRY[tag.index()].key_bytes;
            SparseSubStream::with_capacity(kb, 0, 0)
        });
        let _ = StreamTag::VarKeySnp; // keep import used

        let chunk = SparseChunk { chunk_id: 0, streams, dense };

        let (tx, rx) = bounded(1);
        tx.send(chunk).unwrap();
        drop(tx);

        let dirs = StreamMap::from_fn(|tag| match tag {
            StreamTag::VarKeySnp => vk_snp.clone(),
            StreamTag::VarKeyIndel => vk_indel.clone(),
        });
        let dense_dirs = DenseMap::from_fn(|c| match c {
            DenseClass::Snp => snp_dir.clone(),
            DenseClass::Indel => indel_dir.clone(),
        });

        run_io_writer(rx, dirs, dense_dirs);

        // dense snp chunk files exist with the right bytes
        let pos = std::fs::read(snp_dir.join("chunk_0_pos.bin")).unwrap();
        assert_eq!(pos.len(), 2 * 4); // 2 u32 positions
        let geno = std::fs::read(snp_dir.join("chunk_0_geno.bin")).unwrap();
        assert_eq!(geno, vec![0b0000_1011u8]);
        // indel had 0 dense variants → no files written
        assert!(!indel_dir.join("chunk_0_geno.bin").exists());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e lint cargo test --no-default-features writer`
Expected: FAIL — `run_io_writer` takes 2 args, not 3.

- [ ] **Step 3: Implement the dense write path**

In `src/writer.rs`, update imports and the function:

```rust
use crate::dense::{DenseMap, DENSE_REGISTRY};
```

Replace `run_io_writer`:

```rust
pub fn run_io_writer(
    rx_sparse: Receiver<SparseChunk>,
    dirs: StreamMap<PathBuf>,
    dense_dirs: DenseMap<PathBuf>,
) {
    while let Ok(chunk) = rx_sparse.recv() {
        let id = chunk.chunk_id;

        // var_key per-call streams (unchanged)
        for (tag, sub) in chunk.streams.iter() {
            let dir = dirs.get(tag);
            write_bin(
                &layout::chunk_pos(dir, id),
                bytemuck::cast_slice(&sub.call_positions),
            );
            write_bin(&layout::chunk_key(dir, id), &sub.call_keys);
        }

        // dense per-class matrix + table (only classes with dense variants)
        for spec in &DENSE_REGISTRY {
            let sub = chunk.dense.get(spec.class);
            if sub.n_dense_variants == 0 {
                continue;
            }
            let dir = dense_dirs.get(spec.class);
            write_bin(
                &layout::chunk_pos(dir, id),
                bytemuck::cast_slice(&sub.positions),
            );
            write_bin(&layout::chunk_key(dir, id), &sub.keys);
            write_bin(&layout::chunk_geno(dir, id), &sub.geno_bits);
        }
    }

    println!("Writer Thread: Channel closed. All chunks safely committed to SSD.");
}
```

In `src/orchestrator.rs`, build dense dirs and pass them. Near where `dirs_for_writer` is built (line ~132):

```rust
    let dirs_for_writer = StreamMap::from_fn(|tag| stream_dirs.get(tag).clone());
    let dense_dirs_for_writer = crate::dense::DenseMap::from_fn(|c| {
        let spec = &crate::dense::DENSE_REGISTRY[c.index()];
        std::path::Path::new(base_out_dir).join(chrom).join(spec.subdir)
    });
```

Update the writer spawn to pass the third arg:

```rust
        .spawn(move || writer::run_io_writer(rx_sparse, dirs_for_writer, dense_dirs_for_writer))
```

Also create the dense dirs up front. In the dir-creation loop area (after the shared-indel `create_dir_all` from Task 3), add:

```rust
    for spec in &crate::dense::DENSE_REGISTRY {
        let dir = std::path::Path::new(base_out_dir).join(chrom).join(spec.subdir);
        fs::create_dir_all(&dir).map_err(|e| ConversionError::Io {
            context: format!("create_dir_all {:?}", dir),
            source: e,
        })?;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: PASS — new writer test passes; e2e still green (dense payloads are empty, so no dense chunk files are written and no dense assertions exist yet).

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/writer.rs src/orchestrator.rs
git commit -m "feat(svar-2): writer emits dense per-chunk pos/key/geno files"
```

---

### Task 8: Executor accumulates dense ledgers (`executor.rs`)

**Files:**
- Modify: `src/executor.rs` (`run_compute_engine` return type)
- Modify: `src/orchestrator.rs` (destructure the new return)
- Test: in-source `#[cfg(test)]` in `src/executor.rs`

**Interfaces:**
- Produces:
  - `pub struct Phase1Output { pub var_key_ledgers: StreamMap<Vec<Vec<u32>>>, pub dense_ledgers: DenseMap<Vec<u32>>, pub long_allele_offsets: Vec<u64> }`
  - `pub fn run_compute_engine(rx_dense, tx_sparse, bank) -> Phase1Output` — `dense_ledgers.get(class)[chunk_id]` = `n_dense_variants` of that class in that chunk (rectangular: uniform across haps, so a scalar per chunk).

- [ ] **Step 1: Write the failing test**

Add to `src/executor.rs` a `tests` module. It drives the engine with a hand-built `DenseChunk`, but since routing is not yet wired (Task 10), the dense ledger will be all-zero here; the test asserts the *shape* and that var_key ledgers are unchanged:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DenseClass;
    use crate::streams::StreamTag;
    use crate::types::BitGrid3;
    use crate::types::DenseChunk;
    use crossbeam_channel::bounded;

    fn one_snp_chunk() -> DenseChunk {
        // 1 variant, 1 sample, 2 ploidy, both haps carry it (SNP A→C).
        let mut genos = BitGrid3::zeros(1, 1, 2);
        genos.or_bit(0, true);
        genos.or_bit(1, true);
        DenseChunk {
            chunk_id: 0,
            pos: vec![100],
            ilens: vec![0],
            alt: b"C".to_vec(),
            alt_offsets: vec![0, 1],
            genos,
        }
    }

    #[test]
    fn test_phase1_output_shapes() {
        let (tx_d, rx_d) = bounded(4);
        let (tx_s, rx_s) = bounded(4);
        let (tx_l, _rx_l) = bounded(4);
        tx_d.send(one_snp_chunk()).unwrap();
        drop(tx_d);

        let bank = crate::nrvk::LongAlleleTableWriter::new(tx_l, 1 << 16);
        let out = run_compute_engine(rx_d, tx_s, bank);

        // one chunk processed → one ledger row per stream and per dense class
        assert_eq!(out.var_key_ledgers.get(StreamTag::VarKeySnp).len(), 1);
        assert_eq!(out.dense_ledgers.get(DenseClass::Snp).len(), 1);
        assert_eq!(out.dense_ledgers.get(DenseClass::Indel).len(), 1);
        // drain sparse so the channel doesn't leak
        while rx_s.recv().is_ok() {}
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e lint cargo test --no-default-features -- executor`
Expected: FAIL — `run_compute_engine` returns a tuple, `Phase1Output`/`dense_ledgers` don't exist.

- [ ] **Step 3: Implement**

Rewrite `src/executor.rs`:

```rust
use crate::dense::DenseMap;
use crate::nrvk::LongAlleleTableWriter;
use crate::rvk::dense2sparse_vk;
use crate::streams::StreamMap;
use crate::types::{DenseChunk, SparseChunk};
use crossbeam_channel::{Receiver, Sender};

/// Phase-1 outputs consumed by the merge stage.
pub struct Phase1Output {
    /// One row per chunk of per-column call counts, per var_key stream.
    pub var_key_ledgers: StreamMap<Vec<Vec<u32>>>,
    /// One scalar per chunk (n_dense_variants), per dense class. Rectangular:
    /// every hap contributes the same count, so no per-column matrix.
    pub dense_ledgers: DenseMap<Vec<u32>>,
    pub long_allele_offsets: Vec<u64>,
}

pub fn run_compute_engine(
    rx_dense: Receiver<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    mut bank: LongAlleleTableWriter,
) -> Phase1Output {
    let mut var_key_ledgers: StreamMap<Vec<Vec<u32>>> =
        StreamMap::from_fn(|_| Vec::with_capacity(10_000));
    let mut dense_ledgers: DenseMap<Vec<u32>> = DenseMap::from_fn(|_| Vec::with_capacity(10_000));

    while let Ok(chunk) = rx_dense.recv() {
        let sparse_chunk = dense2sparse_vk(&chunk, &mut bank);

        for (tag, sub) in sparse_chunk.streams.iter() {
            var_key_ledgers.get_mut(tag).push(sub.sample_lengths.clone());
        }
        for (class, sub) in sparse_chunk.dense.iter() {
            dense_ledgers.get_mut(class).push(sub.n_dense_variants as u32);
        }

        tx_sparse
            .send(sparse_chunk)
            .expect("Failed to send SparseChunk to Writer");
    }

    println!("Executor: VCF fully processed. Flushing remaining long alleles...");
    let long_allele_offsets: Vec<u64> = bank.finalize();

    Phase1Output {
        var_key_ledgers,
        dense_ledgers,
        long_allele_offsets,
    }
}
```

In `src/orchestrator.rs`, update the executor-result destructure. Replace:

```rust
    let (ledgers, long_allele_offsets) =
        executor_res.map_err(|_| ConversionError::WorkerPanicked {
            thread: format!("exec-{}", chrom),
        })?;
```

with:

```rust
    let phase1 = executor_res.map_err(|_| ConversionError::WorkerPanicked {
        thread: format!("exec-{}", chrom),
    })?;
    let crate::executor::Phase1Output {
        var_key_ledgers: ledgers,
        dense_ledgers,
        long_allele_offsets,
    } = phase1;
```

(The existing var_key merge loop keeps using `ledgers`; `dense_ledgers` is consumed in Task 11.) To avoid an unused-variable warning until Task 11, prefix with underscore for now: bind as `dense_ledgers: _dense_ledgers` **only if** Task 11 is not done in the same session. If executing sequentially into Task 11, leave as `dense_ledgers`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: PASS. (If clippy flags `dense_ledgers` unused, temporarily rename to `_dense_ledgers`; Task 11 uses it.)

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/executor.rs src/orchestrator.rs
git commit -m "feat(svar-2): executor returns Phase1Output with dense ledgers"
```

---

### Task 9: Dense merge (`dense_merge.rs`)

**Files:**
- Create: `src/dense_merge.rs`
- Modify: `src/lib.rs` (`pub mod dense_merge;`)
- Test: in-source `#[cfg(test)]` in `src/dense_merge.rs`

**Interfaces:**
- Consumes: `layout::{chunk_pos,chunk_key,chunk_geno,final_positions,final_keys,final_genotypes}`, `bits::copy_bits`, `rvk::pack_snp_keys`.
- Produces: `pub fn merge_dense_class(num_chunks: usize, num_samples: usize, ploidy: usize, key_bytes: usize, pack_snp: bool, output_dir: &str, dense_ledger: Vec<u32>)` — writes `final_positions.bin` (concat u32), `final_keys.bin` (concat keys; 2-bit-packed if `pack_snp`), `final_genotypes.bin` (hap-major `(S,P,V_dense)` bit matrix), and removes per-chunk temp files. No-op-safe when `V_dense == 0` (writes empty finals).

- [ ] **Step 1: Write the failing tests**

Create `src/dense_merge.rs`:

```rust
//! Rectangular dense merge: concatenate per-chunk dense variant tables and
//! bit-transpose per-chunk hap-major geno blocks into one (S, P, V_dense)
//! matrix. Unlike the ragged var_key tile merge (merge.rs), every hap
//! contributes the SAME per-chunk count, so offsets are uniform — the only
//! non-trivial step is the per-hap bit concatenation across chunks.

use crate::bits::copy_bits;
use crate::layout;
use crate::rvk::pack_snp_keys;
use std::fs;
use std::io::Write;
use std::path::Path;

pub fn merge_dense_class(
    num_chunks: usize,
    num_samples: usize,
    ploidy: usize,
    key_bytes: usize,
    pack_snp: bool,
    output_dir: &str,
    dense_ledger: Vec<u32>,
) {
    let dir = Path::new(output_dir);
    let np = num_samples * ploidy;
    let v_total: usize = dense_ledger.iter().map(|&c| c as usize).sum();

    // ---- positions + keys: sequential concat in chunk order ----
    let mut positions: Vec<u8> = Vec::new();
    let mut keys: Vec<u8> = Vec::new();
    for c in 0..num_chunks {
        if dense_ledger[c] == 0 {
            continue;
        }
        positions.extend_from_slice(&fs::read(layout::chunk_pos(dir, c)).expect("read dense pos"));
        keys.extend_from_slice(&fs::read(layout::chunk_key(dir, c)).expect("read dense key"));
    }
    write_all(&layout::final_positions(dir), &positions);
    let final_key_bytes = if pack_snp {
        pack_snp_keys(&keys) // keys are one raw 2-bit code per variant
    } else {
        keys
    };
    write_all(&layout::final_keys(dir), &final_key_bytes);

    // ---- genotypes: per-hap bit concatenation across chunks ----
    // output bit (hap h, global col g) at flat index h * v_total + g.
    let out_bits_len = (np * v_total).div_ceil(8);
    let mut out = vec![0u8; out_bits_len];

    // prefix sum of dense variants per chunk = global column offset per chunk
    let mut col_prefix = vec![0usize; num_chunks + 1];
    for c in 0..num_chunks {
        col_prefix[c + 1] = col_prefix[c] + dense_ledger[c] as usize;
    }

    for c in 0..num_chunks {
        let v_c = dense_ledger[c] as usize;
        if v_c == 0 {
            continue;
        }
        let block = fs::read(layout::chunk_geno(dir, c)).expect("read dense geno");
        // block bit (hap h, local col d) at h*v_c + d.
        for h in 0..np {
            let src_bit = h * v_c;
            let dst_bit = h * v_total + col_prefix[c];
            copy_bits(&mut out, dst_bit, &block, src_bit, v_c);
        }
    }
    write_all(&layout::final_genotypes(dir), &out);

    // ---- cleanup per-chunk temp files ----
    for c in 0..num_chunks {
        let _ = fs::remove_file(layout::chunk_pos(dir, c));
        let _ = fs::remove_file(layout::chunk_key(dir, c));
        let _ = fs::remove_file(layout::chunk_geno(dir, c));
    }
}

fn write_all(path: &Path, bytes: &[u8]) {
    let mut f = fs::File::create(path).unwrap_or_else(|e| panic!("create {}: {}", path.display(), e));
    f.write_all(bytes).expect("write dense final");
    f.flush().expect("flush dense final");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{get_bit, set_bit};
    use tempfile::tempdir;

    // Build a hap-major block (np rows × v_c cols) from a bool matrix
    // indexed [hap][col], and stage it as chunk `c`'s geno + pos + key files.
    fn stage_chunk(dir: &Path, c: usize, positions: &[u32], keys: &[u8], mat: &[Vec<bool>]) {
        let np = mat.len();
        let v_c = if np > 0 { mat[0].len() } else { 0 };
        let mut block = vec![0u8; (np * v_c).div_ceil(8)];
        for h in 0..np {
            for d in 0..v_c {
                if mat[h][d] {
                    set_bit(&mut block, h * v_c + d);
                }
            }
        }
        write_all(&layout::chunk_pos(dir, c), bytemuck::cast_slice(positions));
        write_all(&layout::chunk_key(dir, c), keys);
        write_all(&layout::chunk_geno(dir, c), &block);
    }

    fn read_u32(path: &Path) -> Vec<u32> {
        let b = fs::read(path).unwrap();
        b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn test_merge_dense_two_chunks_transpose() {
        // np=2 haps. chunk0: 2 variants, chunk1: 1 variant → v_total=3.
        // hap0: [1,0 | 1] ; hap1: [0,1 | 0]  (col order = chunk0 cols then chunk1)
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        stage_chunk(dir, 0, &[100, 200], &[1u8, 2u8], &[vec![true, false], vec![false, true]]);
        stage_chunk(dir, 1, &[300], &[3u8], &[vec![true], vec![false]]);

        merge_dense_class(2, 1, 2, 1, /*pack_snp=*/ false, dir.to_str().unwrap(), vec![2, 1]);

        // positions concat in chunk order
        assert_eq!(read_u32(&layout::final_positions(dir)), vec![100, 200, 300]);
        // keys concat (pack_snp=false → raw)
        assert_eq!(fs::read(layout::final_keys(dir)).unwrap(), vec![1u8, 2, 3]);

        // genotypes: hap0 row = [1,0,1], hap1 row = [0,1,0], flat h*v_total+g.
        let geno = fs::read(layout::final_genotypes(dir)).unwrap();
        let expect_bits = [
            (0usize, true), (1, false), (2, true),   // hap0 cols 0,1,2
            (3, false), (4, true), (5, false),       // hap1 cols 0,1,2
        ];
        for (idx, want) in expect_bits {
            assert_eq!(get_bit(&geno, idx), want, "geno bit {}", idx);
        }
        // temp files removed
        assert!(!layout::chunk_geno(dir, 0).exists());
    }

    #[test]
    fn test_merge_dense_empty() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        merge_dense_class(1, 2, 2, 1, true, dir.to_str().unwrap(), vec![0]);
        assert_eq!(fs::read(layout::final_positions(dir)).unwrap().len(), 0);
        assert_eq!(fs::read(layout::final_genotypes(dir)).unwrap().len(), 0);
    }

    #[test]
    fn test_merge_dense_snp_packs_keys() {
        // pack_snp=true: 5 raw codes → packed into ceil(5/4)=2 bytes.
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        // single chunk, np=1, 5 dense variants, one hap all-set.
        stage_chunk(
            dir, 0, &[1, 2, 3, 4, 5],
            &[1u8, 2, 3, 0, 1],
            &[vec![true, true, true, true, true]],
        );
        merge_dense_class(1, 1, 1, 1, true, dir.to_str().unwrap(), vec![5]);
        // pack_snp_keys([1,2,3,0,1]) == [0x39, 0x01] (see rvk.rs test)
        assert_eq!(fs::read(layout::final_keys(dir)).unwrap(), vec![0x39u8, 0x01]);
    }
}
```

Add `pub mod dense_merge;` to `src/lib.rs`.

- [ ] **Step 2: Run tests to verify they fail, then pass**

Run: `pixi run -e lint cargo test --no-default-features dense_merge`
Expected: PASS once `pub mod dense_merge;` is added (module + tests complete in Step 1). To see red first, stub `merge_dense_class` body to `{}` and watch `test_merge_dense_two_chunks_transpose` fail on the missing final files, then restore.

- [ ] **Step 3: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/dense_merge.rs src/lib.rs
git commit -m "feat(svar-2): rectangular dense merge (concat + bit-transpose + snp pack)"
```

---

### Task 10: Wire routing into `dense2sparse_vk` (`rvk.rs`) — the behavior flip

**Files:**
- Modify: `src/rvk.rs` (`dense2sparse_vk`: routing pre-pass + dense payload build)
- Test: in-source `#[cfg(test)]` in `src/rvk.rs`
- Modify: `tests/test_e2e.rs` (update existing var_key expectations for routed variants)

**Interfaces:**
- Consumes: `cost_model::{Class, Representation, choose_representation}`, `BitGrid3::popcount_plane`, `bits::set_bit`, `dense::{DenseClass, DenseMap}`, `types::DenseSubChunk`, existing `classify_variant`/`VarKey`.
- Produces: `dense2sparse_vk` now fills `SparseChunk.dense` for variants routed to `Dense`, and **excludes** those variants from the var_key streams.

- [ ] **Step 1: Write the failing tests (in-memory routing + conservation)**

Add to the `tests` module in `src/rvk.rs`:

```rust
    use crate::cost_model::Representation;
    use crate::dense::DenseClass;

    // A SNP carried by every hap in a small cohort routes to DENSE, leaving the
    // var_key snp stream empty; its bit lands in the dense geno block.
    #[test]
    fn test_common_snp_routes_to_dense() {
        // n=2 diploid, np=4. One SNP A→C, all 4 haps carry it.
        // dense = 32+2+4 = 38 bits; var_key = 4*(32+2)=136 → dense wins.
        let refs: Vec<&[u8]> = vec![b"A"];
        let alts: Vec<&[u8]> = vec![b"C"];
        let bits = vec![true; 1 * 2 * 2];
        let chunk = build_test_chunk(1, 2, 2, &refs, &alts, &bits);

        let mut bank = make_bank();
        let sparse = dense2sparse_vk(&chunk, &mut bank);

        // var_key snp stream is empty (variant went dense)
        let snp = sparse.streams.get(StreamTag::VarKeySnp);
        assert_eq!(snp.call_positions.len(), 0);

        // dense snp table has 1 variant, all 4 haps' bits set
        let d = sparse.dense.get(DenseClass::Snp);
        assert_eq!(d.n_dense_variants, 1);
        assert_eq!(d.positions, vec![100]);
        assert_eq!(d.keys, vec![1u8]); // 'C' code == 1
        // np=4 haps, v_dense=1 → 4 bits, one per hap at flat idx h*1+0
        for h in 0..4 {
            assert!(crate::bits::get_bit(&d.geno_bits, h), "hap {} bit", h);
        }
    }

    // A rare SNP (single carrier) stays var_key; dense table empty.
    #[test]
    fn test_rare_snp_stays_var_key() {
        // n=100 diploid np=200. One SNP, single carrier → var_key wins
        // (34 < dense 32+2+200=234).
        let n = 100;
        let refs: Vec<&[u8]> = vec![b"A"];
        let alts: Vec<&[u8]> = vec![b"C"];
        let mut bits = vec![false; 1 * n * 2];
        bits[0] = true; // one hap carries it
        let chunk = build_test_chunk(1, n, 2, &refs, &alts, &bits);

        let mut bank = make_bank();
        let sparse = dense2sparse_vk(&chunk, &mut bank);
        assert_eq!(sparse.streams.get(StreamTag::VarKeySnp).call_positions.len(), 1);
        assert_eq!(sparse.dense.get(DenseClass::Snp).n_dense_variants, 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        // Conservation across routing: every set genotype bit is stored exactly
        // once — either as a var_key call or a dense set bit.
        #[test]
        fn test_routing_conserves_all_calls(
            n_variants in 1usize..20,
            n_samples in 1usize..6,
            ploidy in 1usize..3,
            seed in any::<u64>(),
        ) {
            let bits = random_bits(n_variants * n_samples * ploidy, seed);
            let true_count = bits.iter().filter(|&&b| b).count();
            let refs: Vec<&[u8]> = vec![&b"A"[..]; n_variants];
            let alts: Vec<&[u8]> = vec![&b"C"[..]; n_variants]; // all SNPs
            let chunk = build_test_chunk(n_variants, n_samples, ploidy, &refs, &alts, &bits);

            let mut bank = make_bank();
            let sparse = dense2sparse_vk(&chunk, &mut bank);

            let vk_calls: usize = sparse.streams.get(StreamTag::VarKeySnp)
                .sample_lengths.iter().map(|&c| c as usize).sum::<usize>()
                + sparse.streams.get(StreamTag::VarKeyIndel)
                .sample_lengths.iter().map(|&c| c as usize).sum::<usize>();

            let mut dense_calls = 0usize;
            for (_c, sub) in sparse.dense.iter() {
                dense_calls += sub.geno_bits.iter().map(|b| b.count_ones() as usize).sum::<usize>();
            }
            prop_assert_eq!(vk_calls + dense_calls, true_count, "lost or double-counted a call");
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features -- rvk`
Expected: FAIL — dense payload is empty (routing not wired), so `test_common_snp_routes_to_dense` fails (`n_dense_variants == 0`).

- [ ] **Step 3: Implement routing in `dense2sparse_vk`**

In `src/rvk.rs`, add imports at top:

```rust
use crate::cost_model::{choose_representation, Class, Representation};
use crate::dense::{DenseClass, DenseMap};
use crate::types::DenseSubChunk;
```

Replace the body of `dense2sparse_vk` (keep the signature). New logic:

```rust
pub fn dense2sparse_vk(chunk: &DenseChunk, bank: &mut LongAlleleTableWriter) -> SparseChunk {
    let (v_variants, num_samples, ploidy) = chunk.genos.shape;
    let columns = num_samples * ploidy;

    // Per-variant routing record built in one pre-pass.
    enum Route {
        VarKey(VarKey),
        // dense variant: which class + its column index within that class's table
        Dense { class: DenseClass, col: u32 },
    }

    // Dense per-class tables accumulate positions + keys as we discover dense
    // variants; geno_bits is sized after the pre-pass (n_dense known).
    let mut dense = DenseMap::from_fn(|c| DenseSubChunk::empty(c.key_bytes()));
    let mut routes: Vec<Route> = Vec::with_capacity(v_variants);

    for v in 0..v_variants {
        let ilen = unsafe { *chunk.ilens.get_unchecked(v) };
        let start_idx = unsafe { *chunk.alt_offsets.get_unchecked(v) } as usize;
        let end_idx = unsafe { *chunk.alt_offsets.get_unchecked(v + 1) } as usize;
        let alt_allele = unsafe { chunk.alt.get_unchecked(start_idx..end_idx) };

        let vk = classify_variant(ilen, alt_allele, bank);
        let (class, dclass) = match vk {
            VarKey::Snp(_) => (Class::Snp, DenseClass::Snp),
            VarKey::Indel(_) => (Class::Indel, DenseClass::Indel),
        };
        let x = chunk.genos.popcount_plane(v);

        match choose_representation(class, num_samples, ploidy, x) {
            Representation::VarKey => routes.push(Route::VarKey(vk)),
            Representation::Dense => {
                let sub = dense.get_mut(dclass);
                let col = sub.n_dense_variants as u32;
                let pos = unsafe { *chunk.pos.get_unchecked(v) };
                sub.positions.push(pos);
                match vk {
                    VarKey::Snp(code) => sub.keys.push(code),
                    VarKey::Indel(key) => sub.keys.extend_from_slice(&key.to_le_bytes()),
                }
                sub.n_dense_variants += 1;
                routes.push(Route::Dense { class: dclass, col });
            }
        }
    }

    // Size each dense class's hap-major geno block now that n_dense is known.
    for (_c, sub) in dense.iter_mut_hack() {
        let bits = columns * sub.n_dense_variants;
        sub.geno_bits = vec![0u8; bits.div_ceil(8)];
    }

    let estimated_nnz = (v_variants * columns) / 20;
    let mut streams = crate::streams::StreamMap::from_fn(|tag| {
        let spec = &crate::streams::REGISTRY[tag.index()];
        SparseSubStream::with_capacity(spec.key_bytes, estimated_nnz, columns)
    });

    let words: &[u64] = &chunk.genos.words;

    for s in 0..num_samples {
        for p in 0..ploidy {
            let mut counts = crate::streams::StreamMap::from_fn(|_| 0u32);
            let base_idx = (s * ploidy) + p;
            let hap = base_idx; // hap index in sample-major order
            let stride = columns;

            for v in 0..v_variants {
                let flat_idx = (v * stride) + base_idx;
                let word = unsafe { *words.get_unchecked(flat_idx >> 6) };
                if (word >> (flat_idx & 63)) & 1 == 0 {
                    continue;
                }
                match unsafe { routes.get_unchecked(v) } {
                    Route::VarKey(vk) => {
                        let pos = unsafe { *chunk.pos.get_unchecked(v) };
                        let (tag, key_le): (StreamTag, [u8; 4]) = match vk {
                            VarKey::Snp(code) => (StreamTag::VarKeySnp, [*code, 0, 0, 0]),
                            VarKey::Indel(key) => (StreamTag::VarKeyIndel, key.to_le_bytes()),
                        };
                        let spec = &crate::streams::REGISTRY[tag.index()];
                        streams.get_mut(tag).push_call(pos, &key_le[..spec.key_bytes]);
                        *counts.get_mut(tag) += 1;
                    }
                    Route::Dense { class, col } => {
                        let sub = dense.get_mut(*class);
                        let bit = hap * sub.n_dense_variants + (*col as usize);
                        crate::bits::set_bit(&mut sub.geno_bits, bit);
                    }
                }
            }
            for (tag, c) in counts.into_iter_tagged() {
                streams.get_mut(tag).sample_lengths.push(c);
            }
        }
    }

    SparseChunk {
        chunk_id: chunk.chunk_id,
        streams,
        dense,
    }
}
```

`DenseMap` needs a mutable iterator for the sizing loop. Add to `src/dense.rs` `impl<T> DenseMap<T>`:

```rust
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (DenseClass, &mut T)> {
        DenseClass::ALL.into_iter().zip(self.slots.iter_mut())
    }
```

and in `rvk.rs` replace `dense.iter_mut_hack()` with `dense.iter_mut()`.

> **Borrow note:** the transpose loop calls `dense.get_mut(*class)` inside the same scope that reads `routes`. `routes` and `dense` are distinct locals, so there is no aliasing. `sub.n_dense_variants` is read to compute the bit index — it is fixed after the pre-pass, so reading it during the transpose is correct.

- [ ] **Step 4: Update the existing e2e expectations**

The routing flip changes `tests/test_e2e.rs::test_e2e_normalized_bcf_pipeline` (np=4, so common variants now go dense). Update it to reflect routing. With `n=2` diploid (np=4):
- pos100 SNP, carriers = 3 → dense (34·3=102 > 32+2+4=38) → **dense/snp**.
- pos200 INS, carriers = 1 → var_key (64 < 32+32+4=68) → **var_key/indel**.
- pos300 DEL, carriers = 2 → var_key (128 > 68 → dense!). 2·64=128 > 68 → **dense/indel**.

Rather than hand-verify every count, retarget this test to assert **routing outcomes + genotype reconstruction** rather than the old pure-var_key layout. Replace the body's post-conditions with:

```rust
    // Routing (np=4): common SNP@100 (x=3) → dense/snp; INS@200 (x=1) → var_key/indel;
    // DEL@300 (x=2) → dense/indel.
    let dsnp = out_dir.join("chr1/dense/snp");
    let dindel = out_dir.join("chr1/dense/indel");
    let vk_indel = out_dir.join("chr1/var_key/indel");

    // dense/snp: 1 variant @100, packed keys, geno bits for haps 0,2,3.
    let dsnp_pos = read_u32_bin(&dsnp.join("final_positions.bin"));
    assert_eq!(dsnp_pos, vec![100]);
    let dsnp_geno = std::fs::read(dsnp.join("final_genotypes.bin")).unwrap();
    // np=4, v_dense=1 → bit h*1+0. Carriers = haps 0,2,3 (gt [1,0,1,1]).
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 0));
    assert!(!genoray_core::bits::get_bit(&dsnp_geno, 1));
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 2));
    assert!(genoray_core::bits::get_bit(&dsnp_geno, 3));

    // var_key/indel: only INS@200 for hap 1.
    let vki_pos = read_u32_bin(&vk_indel.join("final_positions.bin"));
    let vki_off = read_offsets_npy(&vk_indel.join("final_offsets.npy"));
    assert_eq!(vki_off, vec![0u64, 0, 1, 1, 1]); // hap1 has the single INS call
    assert_eq!(vki_pos, vec![200]);

    // dense/indel: DEL@300 (x=2, haps 0,1).
    let dindel_pos = read_u32_bin(&dindel.join("final_positions.bin"));
    assert_eq!(dindel_pos, vec![300]);
    let dindel_geno = std::fs::read(dindel.join("final_genotypes.bin")).unwrap();
    assert!(genoray_core::bits::get_bit(&dindel_geno, 0)); // hap0
    assert!(genoray_core::bits::get_bit(&dindel_geno, 1)); // hap1
    assert!(!genoray_core::bits::get_bit(&dindel_geno, 2));
    assert!(!genoray_core::bits::get_bit(&dindel_geno, 3));
```

Delete the old `snp_dir`/`indel_dir` var_key assertions in this test that assumed everything was var_key. Keep the imports; add `use genoray_core::bits;` if preferred over the fully-qualified path.

For the **conservation** e2e test (`test_e2e_...` conservation, lines ~217+), generalize it to sum calls across **all four** buckets (var_key pos-file lengths + dense geno popcounts) instead of only var_key positions. Concretely, after conversion, compute:

```rust
    // total calls = var_key final_positions lengths + dense genotypes popcounts
    let mut total = 0usize;
    for sub in ["var_key/snp", "var_key/indel"] {
        let p = out_dir.join(format!("chr1/{sub}/final_positions.bin"));
        if p.exists() {
            total += read_u32_bin(&p).len();
        }
    }
    for sub in ["dense/snp", "dense/indel"] {
        let g = out_dir.join(format!("chr1/{sub}/final_genotypes.bin"));
        if g.exists() {
            total += std::fs::read(&g).unwrap().iter().map(|b| b.count_ones() as usize).sum::<usize>();
        }
    }
    assert_eq!(total, expected_true_calls);
```

(Adapt `expected_true_calls` to the existing ground-truth variable in that test.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: PASS — routing unit tests, conservation proptest, and the updated e2e all green.

- [ ] **Step 6: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/rvk.rs src/dense.rs tests/test_e2e.rs
git commit -m "feat(svar-2): route variants dense vs var_key by cost model"
```

---

### Task 11: Orchestrator drives the dense merge + dense round-trip e2e (`orchestrator.rs`, `tests/test_e2e.rs`)

**Files:**
- Modify: `src/orchestrator.rs` (run `merge_dense_class` per dense class after Phase 1)
- Modify: `tests/test_e2e.rs` (add a dedicated dense round-trip test)

**Interfaces:**
- Consumes: `executor::Phase1Output.dense_ledgers`, `dense::DENSE_REGISTRY`, `dense_merge::merge_dense_class`.
- Produces: after `process_chromosome`, each `dense/{snp,indel}/` dir holds `final_positions.bin`, `final_keys.bin`, `final_genotypes.bin`; per-chunk temp files removed.

- [ ] **Step 1: Write the failing test (dense round-trip)**

Add to `tests/test_e2e.rs` a test that forces a common variant to dense and reconstructs its genotypes:

```rust
// Dense round-trip: a SNP carried by most of a small cohort must be routed to
// dense/snp and its genotype bits reconstructable from final_genotypes.bin.
#[test]
fn test_e2e_dense_snp_roundtrip() {
    let tmp = tempdir().unwrap();
    let bcf_path = tmp.path().join("dense.bcf");
    let samples = vec!["S0", "S1", "S2"]; // np = 6

    // One SNP A→C carried by haps 0,1,2,3,4 (5 of 6). x=5.
    // dense = 32+2+6 = 40 < var_key 5*34 = 170 → dense.
    let records = vec![SynthRecord {
        pos: 500,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt: vec![1, 1, 1, 1, 1, 0], // S0(1,1) S1(1,1) S2(1,0)
    }];
    build_bcf_with_index(&bcf_path, "chr1", 10_000, &samples, &records);

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    process_chromosome(
        bcf_path.to_str().unwrap(), "chr1", out_dir.to_str().unwrap(),
        &samples, 100, 2, 1, 4096,
    )
    .expect("conversion");

    let dsnp = out_dir.join("chr1/dense/snp");
    let pos = read_u32_bin(&dsnp.join("final_positions.bin"));
    assert_eq!(pos, vec![500]);
    // packed key: 'C' code 1, 1 variant → pack_snp_keys([1]) == [0x01]
    assert_eq!(std::fs::read(dsnp.join("final_keys.bin")).unwrap(), vec![0x01u8]);

    // genotypes: np=6, v_dense=1 → bit h. Haps 0..5 carriers = [1,1,1,1,1,0].
    let geno = std::fs::read(dsnp.join("final_genotypes.bin")).unwrap();
    for h in 0..5 {
        assert!(genoray_core::bits::get_bit(&geno, h), "hap {}", h);
    }
    assert!(!genoray_core::bits::get_bit(&geno, 5));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e lint cargo test --no-default-features test_e2e_dense_snp_roundtrip`
Expected: FAIL — `dense/snp/final_*` don't exist (orchestrator doesn't run the dense merge yet).

- [ ] **Step 3: Implement the dense merge drive in the orchestrator**

In `src/orchestrator.rs`, after the existing var_key merge loop (the `for spec in &REGISTRY { … merge::merge_mini_sc(…) … }` block), add:

```rust
    // Dense merge: one rectangular merge per dense class (no-op-safe when empty).
    let mut dense_ledgers = dense_ledgers; // make mutable to move rows out
    for spec in &crate::dense::DENSE_REGISTRY {
        let dir = std::path::Path::new(base_out_dir).join(chrom).join(spec.subdir);
        let ledger = std::mem::take(dense_ledgers.get_mut(spec.class));
        crate::dense_merge::merge_dense_class(
            num_chunks,
            samples.len(),
            ploidy,
            spec.key_bytes,
            spec.pack_snp,
            dir.to_str().unwrap(),
            ledger,
        );
    }
```

(`num_chunks` is already computed above as `ledgers.get(StreamTag::VarKeyIndel).len()`; it is identical for dense classes — one ledger row per chunk. If Task 8 renamed the binding to `_dense_ledgers`, rename it back to `dense_ledgers` here.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: PASS — dense round-trip + all prior tests.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
git add src/orchestrator.rs tests/test_e2e.rs
git commit -m "feat(svar-2): orchestrator drives dense merge; dense e2e round-trip"
```

---

### Task 12: Documentation reconciliation (`docs/roadmap/*.md`)

**Files:**
- Modify: `docs/roadmap/data-model.md`
- Modify: `docs/roadmap/architecture.md`
- Modify: `docs/roadmap/svar-2.md`

**Interfaces:** none (docs only). Required by the SVAR 2.0 working agreement.

- [ ] **Step 1: `data-model.md` — cost model.** In the "Dense vs. sparse cost model" section, state that `s` **includes** the per-call `u32` position (`POS_BITS = 32`), give the concrete per-representation bit costs (`var_key = x·(32 + key_bits)`, `dense = 32 + key_bits + np`, `key_bits`: SNP 2, indel 32), note the model is evaluated in **bits** (exact, no fractions), tie-break to `var_key`. In the "Open questions" list, mark **"`s` for `var_key` / whether packed-position bytes count"** resolved (yes, positions count).

- [ ] **Step 2: `data-model.md` — LUT + layout.** State the indel long-allele LUT is a **single shared per-contig table** at `{contig}/indel/long_alleles.{bin,offsets}`, referenced by both `var_key/indel` and `dense/indel`; remove the per-representation LUT placement from the on-disk tree. In the dense-representation section, state `genotypes` is a **raw bit-packed** matrix (LSB-first, hap-major `(S,P,V_dense)`, variant fastest) whose shape is derived from `len(positions)` + `(n, ploidy)` (no shape sidecar). Note the provisional dense final filenames (`final_positions.bin`/`final_keys.bin`/`final_genotypes.bin`) mirror var_key and will be unified by M3.

- [ ] **Step 3: `architecture.md` — routing granularity.** In "Open questions", resolve **"Routing granularity"**: representation is chosen **strictly per variant**, decided locally within a chunk from the variant's plane popcount (a variant is fully contained in one chunk).

- [ ] **Step 4: `svar-2.md` — milestones.** Flip **M4** from `[ ]` to `[x]` (or `[~]` if any sub-item is intentionally deferred) and rewrite its status line to describe: cost model (`cost_model.rs`), per-variant routing in the transpose, dense per-class matrix + table, rectangular dense merge, and the shared per-contig indel LUT. Note explicitly that `max_del.npy`/overlap (M5), `meta.json` (M3), and pointer (M11) remain out of scope.

- [ ] **Step 5: Commit**

```bash
git add docs/roadmap/data-model.md docs/roadmap/architecture.md docs/roadmap/svar-2.md
git commit -m "docs(svar-2): reconcile roadmap with M4 dense + cost-model implementation"
```

---

## Self-Review

**Spec coverage:**
- Cost model (spec §2) → Task 1. ✓
- Routing in executor + conservation (spec §3) → Tasks 2, 10. ✓
- Dense chunk output + writer (spec §4) → Tasks 5, 7. ✓
- Dense merge (spec §5) → Tasks 4, 9. ✓
- Shared LUT (spec §6) → Task 3. ✓
- Layout additions (spec §7) → Tasks 3, 6. ✓
- Component boundaries (spec §8) → file structure table. ✓
- Doc reconciliation (spec §9) → Task 12. ✓
- Testing strategy (spec §10) → tests in Tasks 1,2,4,9,10 + e2e in 10,11. ✓
- Out-of-scope items (max_del/M5, meta.json/M3, pointer/M11) → excluded, noted in Task 12. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step shows full code. The one judgment call (Task 8 `dense_ledgers` vs `_dense_ledgers` to dodge an unused-var warning if executed non-sequentially) is spelled out with the exact rename.

**Type consistency:**
- `choose_representation(class, n_samples, ploidy, x_calls)` — same signature in Tasks 1 and 10. ✓
- `Class`/`Representation` from `cost_model`; `DenseClass`/`DenseMap`/`DenseSpec`/`DENSE_REGISTRY` from `dense` — consistent across Tasks 5, 7, 8, 9, 10, 11. ✓
- `DenseSubChunk { key_bytes, n_dense_variants, positions, keys, geno_bits }` — defined Task 5, populated Task 10, consumed Tasks 7 (writer) and 9 (merge, via files). ✓
- `Phase1Output { var_key_ledgers, dense_ledgers, long_allele_offsets }` — defined Task 8, consumed Task 11. ✓
- `merge_dense_class(num_chunks, num_samples, ploidy, key_bytes, pack_snp, output_dir, dense_ledger)` — defined Task 9, called Task 11. ✓
- `copy_bits(dst, dst_bit, src, src_bit, n)`, `set_bit`, `get_bit` — defined Task 4, used Tasks 9, 10, 11. ✓
- `popcount_plane(v)` — defined Task 2, used Task 10. ✓
- Layout fns `chunk_geno`/`final_genotypes`/`dense_snp_dir`/`dense_indel_dir` and relocated `long_alleles_bin`/`shared_indel_dir` — defined Tasks 3, 6, used Tasks 7, 9, 11. ✓

One inconsistency found and fixed inline: Task 10 Step 3 initially referenced a nonexistent `dense.iter_mut_hack()`; corrected to add `DenseMap::iter_mut` in `dense.rs` and call `dense.iter_mut()`.
</content>

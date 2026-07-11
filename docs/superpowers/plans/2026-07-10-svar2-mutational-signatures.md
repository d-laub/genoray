# SVAR2 Mutational Signatures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add COSMIC mutational-signature support (SBS96, ID83, DBS78-via-adjacency) to `SparseVar2`, implemented in Rust, with v1 feature parity and far lower memory.

**Architecture:** A per-contig `mutcat/` sidecar stores one `u8` mutation code per record (aligned to each existing sub-stream's `positions.bin`) plus a 2-bit reference base for SNPs. A single Rust classification pass writes it (at write time or post-hoc). `mutation_matrix` streams the mmap'd genotype sub-streams column-by-column, reclassifying adjacent same-haplotype SNV pairs into DBS78 on the fly — never materializing a per-entry code array. The existing numpy/scipy `fit_signatures` refit is reused unchanged.

**Tech Stack:** Rust (pyo3, memmap2, ndarray, ndarray-npy, rayon), `svar2-codec` crate (2-bit codec reuse), Python (polars, numpy), pixi.

## Global Constraints

- **Coordinate convention:** all ranges 0-based half-open `[start, end)`; POS in the Python index is 1-based and converted to 0-based internally.
- **Codebook is single-source-of-truth:** SBS96/DBS78/ID83 label order, `N_CODES=257`, offsets (`SBS96_OFFSET=0`, `DBS78_OFFSET=96`, `ID83_OFFSET=174`), and `MUTCAT_VERSION=3` live in `python/genoray/_mutcat/codebook.py`. Rust must emit **identical** SBS96 (`0–95`) and ID83 (`0–82`) indices. Do **not** duplicate the label lists in Rust.
- **Sidecar `u8` code encoding:** SNP record = SBS96 index `0–95`; indel record = ID83 index `0–82`; `254 = UNCLASSIFIED`, `255 = NOT_ANNOTATED`. DBS78 is never stored. The counter skips any `code >= N_CODES`.
- **Sidecar `u8` values are class-local indices** (`0–95` / `0–82`), NOT the unified code-space offsets. The counter adds the class offset when accumulating.
- **2-bit ref base uses the existing codec:** `svar2_codec::{pack_snp_keys, unpack_snp_key_at}`, encoding `A=00 C=01 T=10 G=11` (`encode_snp_2bit`). Reuse it; write no new 2-bit packer.
- **REF from FASTA:** SNP records store no REF; indels are left-aligned to the FASTA. The classifier gets REF from the reference sequence. The counter must NOT need the FASTA.
- **Public-API rule:** any change reachable from `import genoray` without underscores requires updating `skills/genoray-api/SKILL.md` in the same change (see `CLAUDE.md`).
- **Rust test command:** `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion <filter>'` (the bare `cargo test` fails to link pyo3 — `undefined symbol: _Py_Dealloc`).
- **Python extension rebuild:** after any Rust change, run `pixi run maturin develop` before Python tests.
- **Commits:** Conventional Commits. Never run long `cargo`/`maturin` builds in the background — foreground only, or they get abandoned mid-build.
- **Branch:** work on `feat/svar2-mutational-signatures` (already created).

## File Structure

**New Rust (`src/mutcat/`):**
- `src/mutcat/mod.rs` — module root + re-exports; `pub const` code-space offsets/sentinels mirrored from the Python codebook (with a parity test).
- `src/mutcat/classify.rs` — pure classifiers: `sbs96_code`, `id83_code`, `dbs78_code` (+ the `(4,4,4,4)` DBS table).
- `src/mutcat/sidecar.rs` — sidecar file paths + typed read/write (`u8` code arrays, 2-bit ref).
- `src/mutcat/annotate.rs` — the classification pass over finalized records → sidecar.
- `src/mutcat/count.rs` — streaming count matrix with DBS pairing.

**Modified Rust:**
- `src/lib.rs` — `mod mutcat;`, register pyfunctions, thread `signatures` into `run_conversion_pipeline`.
- `src/layout.rs` — `mutcat/` path helpers.
- `src/cost_model.rs` — `choose_representation` gains a `sidecar_bits` term.
- `src/rvk.rs` — pass `sidecar_bits` (0 unless signatures) into `choose_representation` in `dense2sparse_vk`.
- `src/py_query.rs` (or new `src/py_mutcat.rs`) — `PyContigReader` methods `annotate_mutations` + `count_matrix`.

**New/modified Python:**
- `python/genoray/_svar2_mutcat.py` (new) — `_MutcatMixin` with `annotate_mutations`, `mutation_matrix`, `assign_signatures`.
- `python/genoray/_svar2.py` — inherit `_MutcatMixin`; add `signatures: bool` to `from_vcf`.
- `python/genoray/_mutcat/codebook.py`, `_signatures.py` — reused unchanged.

**Tests:**
- Rust unit tests colocated in each `src/mutcat/*.rs`.
- `tests/test_svar2_mutcat.py` — Python round-trip + v1 parity.

---

## Phase 0 — Module scaffold + code-space constants

### Task 0: `mutcat` module with codebook constants

**Files:**
- Create: `src/mutcat/mod.rs`
- Modify: `src/lib.rs` (add `mod mutcat;` near the other `mod` declarations)

**Interfaces:**
- Produces: `mutcat::{SBS96_OFFSET, DBS78_OFFSET, ID83_OFFSET, N_CODES, UNCLASSIFIED, NOT_ANNOTATED}` (all `usize`/`u8`), and `mutcat::Kind { Sbs96, Dbs78, Id83 }` with `code_range(self) -> Range<usize>`.

- [ ] **Step 1: Write the failing test** — in `src/mutcat/mod.rs`:

```rust
//! COSMIC mutation-catalogue code space and classifiers for SVAR2.
//! Code-space layout MUST match python/genoray/_mutcat/codebook.py.

pub mod classify;

use std::ops::Range;

/// Class-local index counts (COSMIC codebook sizes).
pub const N_SBS96: usize = 96;
pub const N_DBS78: usize = 78;
pub const N_ID83: usize = 83;

/// Unified code-space offsets — mirror codebook.py exactly.
pub const SBS96_OFFSET: usize = 0;
pub const DBS78_OFFSET: usize = SBS96_OFFSET + N_SBS96; // 96
pub const ID83_OFFSET: usize = DBS78_OFFSET + N_DBS78; // 174
pub const N_CODES: usize = ID83_OFFSET + N_ID83; // 257

/// Sidecar `u8` sentinels (unsigned; v1's negative int16 sentinels don't survive).
pub const UNCLASSIFIED: u8 = 254;
pub const NOT_ANNOTATED: u8 = 255;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Sbs96,
    Dbs78,
    Id83,
}

impl Kind {
    /// Half-open unified-code range for this kind.
    pub fn code_range(self) -> Range<usize> {
        match self {
            Kind::Sbs96 => SBS96_OFFSET..DBS78_OFFSET,
            Kind::Dbs78 => DBS78_OFFSET..ID83_OFFSET,
            Kind::Id83 => ID83_OFFSET..N_CODES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_space_matches_codebook() {
        assert_eq!(SBS96_OFFSET, 0);
        assert_eq!(DBS78_OFFSET, 96);
        assert_eq!(ID83_OFFSET, 174);
        assert_eq!(N_CODES, 257);
    }

    #[test]
    fn kind_ranges_are_contiguous_and_sized() {
        assert_eq!(Kind::Sbs96.code_range(), 0..96);
        assert_eq!(Kind::Dbs78.code_range(), 96..174);
        assert_eq!(Kind::Id83.code_range(), 174..257);
    }
}
```

- [ ] **Step 2: Add `mod mutcat;` to `src/lib.rs`** next to the other module declarations (e.g. after `mod merge;`).

- [ ] **Step 3: Run tests to verify they pass**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::tests'`
Expected: PASS (2 tests).

- [ ] **Step 4: Commit**

```bash
git add src/mutcat/mod.rs src/lib.rs
git commit -m "feat(mutcat): scaffold mutcat module + code-space constants"
```

---

## Phase 1 — Classifiers (pure functions)

### Task 1: SBS96 classifier

**Files:**
- Create: `src/mutcat/classify.rs`

**Interfaces:**
- Consumes: `crate::mutcat::UNCLASSIFIED`.
- Produces:
  - `pub fn base_index(b: u8) -> i8` — `A=0 C=1 G=2 T=3`, else `-1`.
  - `pub fn sbs96_code(five: u8, refb: u8, altb: u8, three: u8) -> u8` — returns SBS96 class-local index `0–95` or `UNCLASSIFIED`. `five`/`three` are the immediate 5′/3′ reference flanks (ASCII), `refb`/`altb` the ASCII REF/ALT bases.

This ports `python/genoray/_mutcat/classify.py:_sbs96_codes` (COSMIC order: `sub*16 + five*4 + three`, substitutions `C>A,C>G,C>T,T>A,T>C,T>G`, pyrimidine folding).

- [ ] **Step 1: Write the failing test** — append to `src/mutcat/classify.rs`:

```rust
//! Pure mutation classifiers (SBS96, ID83, DBS78). Port of
//! python/genoray/_mutcat/classify.py. No I/O; operates on ASCII bytes +
//! reference slices.

use crate::mutcat::UNCLASSIFIED;

/// A=0 C=1 G=2 T=3, else -1 (N or non-ACGT).
#[inline]
pub fn base_index(b: u8) -> i8 {
    match b {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => -1,
    }
}

// (ref_idx, alt_idx) -> SBS substitution index 0..5 for pyrimidine-folded refs.
// Order: C>A, C>G, C>T, T>A, T>C, T>G  (A=0,C=1,G=2,T=3).
const SUB_LUT: [[i8; 4]; 4] = {
    let mut lut = [[-1i8; 4]; 4];
    // C(1)>A(0),C>G(2),C>T(3)
    lut[1][0] = 0;
    lut[1][2] = 1;
    lut[1][3] = 2;
    // T(3)>A(0),T>C(1),T>G(2)
    lut[3][0] = 3;
    lut[3][1] = 4;
    lut[3][2] = 5;
    lut
};

/// SBS96 class-local index (0..=95) or `UNCLASSIFIED`.
/// `five`/`three`: immediate reference flanks; `refb`/`altb`: REF/ALT bases.
pub fn sbs96_code(five: u8, refb: u8, altb: u8, three: u8) -> u8 {
    let r = base_index(refb);
    let a = base_index(altb);
    let f = base_index(five);
    let t = base_index(three);
    if r < 0 || a < 0 || f < 0 || t < 0 || r == a {
        return UNCLASSIFIED;
    }
    let purine = r == 0 || r == 2; // A or G -> fold
    let (rr, aa, ff, tt) = if purine {
        // fold: complement ref/alt, and flanks swap+complement
        (3 - r, 3 - a, 3 - t, 3 - f)
    } else {
        (r, a, f, t)
    };
    let sub = SUB_LUT[rr as usize][aa as usize];
    if sub < 0 {
        return UNCLASSIFIED;
    }
    (sub as u8) * 16 + (ff as u8) * 4 + (tt as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutcat::UNCLASSIFIED;

    #[test]
    fn sbs96_pyrimidine_ref_no_fold() {
        // A[C>A]G : sub C>A = 0, five=A=0, three=G=2 -> 0*16 + 0*4 + 2 = 2
        assert_eq!(sbs96_code(b'A', b'C', b'A', b'G'), 2);
    }

    #[test]
    fn sbs96_purine_ref_folds() {
        // Purine ref folds to its pyrimidine partner with flanks swapped+complemented.
        // G>T at flanks A..C  == (fold) C>A at flanks G..T.
        // Direct fold: r=G(2)->rr=1(C), a=T(3)->aa=0(A) => sub C>A=0.
        // five=A(0)->ff=comp(three=C(1))=2(G); three=C(1)->tt=comp(five=A(0))=3(T).
        // code = 0*16 + 2*4 + 3 = 11.
        assert_eq!(sbs96_code(b'A', b'G', b'T', b'C'), 11);
    }

    #[test]
    fn sbs96_rejects_non_acgt_and_ref_eq_alt() {
        assert_eq!(sbs96_code(b'N', b'C', b'A', b'G'), UNCLASSIFIED);
        assert_eq!(sbs96_code(b'A', b'C', b'C', b'G'), UNCLASSIFIED); // ref==alt
    }
}
```

- [ ] **Step 2: Add `pub mod classify;` is already in `mod.rs` (Task 0). Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::classify'`
Expected: PASS (3 tests).

- [ ] **Step 3: Cross-check against v1 on 20 random SNVs (one-off, not committed).** Write a throwaway Python snippet in the scratchpad that calls `genoray._mutcat.classify._sbs96_codes` on 20 random `(five, ref, alt, three)` contexts and compares to a hand-run of the Rust logic; confirm the COSMIC index matches for at least the two asserted cases. (This is a sanity gate; the authoritative check is the Python parity test in Task 14.)

- [ ] **Step 4: Commit**

```bash
git add src/mutcat/classify.rs
git commit -m "feat(mutcat): SBS96 classifier (pyrimidine-folded trinucleotide)"
```

### Task 2: ID83 classifier

**Files:**
- Modify: `src/mutcat/classify.rs`

**Interfaces:**
- Produces: `pub fn id83_code(seq: &[u8], pos0: usize, refa: &[u8], alta: &[u8]) -> u8` — ID83 class-local index `0–82` or `UNCLASSIFIED`. `seq` is the whole contig (ASCII uint8), `pos0` the 0-based REF start, `refa`/`alta` the full REF/ALT allele bytes. Ports `_id83_kernel` in `classify.py:134-225`. A ref/reference disagreement (v1's internal `_REF_MISMATCH`) maps to `UNCLASSIFIED`.

- [ ] **Step 1: Write the failing test** — append to `src/mutcat/classify.rs` (inside the module body, above `#[cfg(test)]`):

```rust
// ID83 lookup tables, built to match codebook.py ID83 order.
// ID83 label order (see _build_id83): 24 single-base (Del/Ins x C/T x rep0..5),
// then 48 repeat (Del/Ins x size2..5 x rep0..5), then 11 microhomology-Del.
struct Id83Luts {
    id1: [[[u8; 6]; 2]; 2],  // [kind(Del=0,Ins=1)][base(C=0,T=1)][rep0..5]
    idr: [[[u8; 6]; 4]; 2],  // [kind][size_bucket(2..5)][rep0..5]
    idm: [[u8; 6]; 4],       // [size_bucket][mh]
}

const fn build_id83_luts() -> Id83Luts {
    // Index formula mirrors codebook order:
    //   single: base = (kind*2 + base_ct)*6 + rep                          [0..24)
    //   repeat: 24 + (kind*4 + size_bucket)*6 + rep                        [24..72)
    //   mh    : 72 + cumulative(mh caps 1,2,3,5) offset + (m-1)            [72..83)
    let u = UNCLASSIFIED;
    let mut id1 = [[[u; 6]; 2]; 2];
    let mut idr = [[[u; 6]; 4]; 2];
    let mut idm = [[u; 6]; 4];
    let mut kind = 0;
    while kind < 2 {
        let mut b = 0;
        while b < 2 {
            let mut r = 0;
            while r < 6 {
                id1[kind][b][r] = ((kind * 2 + b) * 6 + r) as u8;
                r += 1;
            }
            b += 1;
        }
        let mut s = 0;
        while s < 4 {
            let mut r = 0;
            while r < 6 {
                idr[kind][s][r] = (24 + (kind * 4 + s) * 6 + r) as u8;
                r += 1;
            }
            s += 1;
        }
        kind += 1;
    }
    // microhomology (Del only): caps per size bucket = [1,2,3,5]; base offset 72.
    let caps = [1usize, 2, 3, 5];
    let mut s = 0;
    let mut base = 72usize;
    while s < 4 {
        let mut m = 1;
        while m <= caps[s] {
            idm[s][m] = (base + (m - 1)) as u8;
            m += 1;
        }
        base += caps[s];
        s += 1;
    }
    Id83Luts { id1, idr, idm }
}

const ID83_LUTS: Id83Luts = build_id83_luts();
const MH_CAP: [usize; 4] = [1, 2, 3, 5];

/// ID83 class-local index (0..=82) or `UNCLASSIFIED`. Port of `_id83_kernel`.
pub fn id83_code(seq: &[u8], pos0: usize, refa: &[u8], alta: &[u8]) -> u8 {
    let n = seq.len();
    let rl = refa.len();
    let al = alta.len();
    // atomized indels always share an anchor base; require it and equal anchors.
    if rl == 0 || al == 0 || refa[0] != alta[0] {
        return UNCLASSIFIED;
    }
    let is_del = rl > al;
    // deleted/inserted unit = allele[1..]
    let (buf, ilen): (&[u8], usize) = if is_del {
        (&refa[1..], rl - 1)
    } else {
        (&alta[1..], al - 1)
    };
    if ilen == 0 {
        return UNCLASSIFIED;
    }
    for &c in buf {
        if base_index(c) < 0 {
            return UNCLASSIFIED;
        }
    }
    // count tandem repeats of the unit downstream from pos0+1
    let scan = pos0 + 1;
    let mut n_rep = 0usize;
    let mut i = 0usize;
    while scan + i + ilen <= n {
        let mut m = true;
        let mut j = 0;
        while j < ilen {
            if seq[scan + i + j] != buf[j] {
                m = false;
                break;
            }
            j += 1;
        }
        if !m {
            break;
        }
        n_rep += 1;
        i += ilen;
    }
    if ilen == 1 {
        let mut bi = base_index(buf[0]);
        if bi == 0 || bi == 2 {
            bi = 3 - bi; // A/G -> pyrimidine partner
        }
        let base_ct = if bi == 1 { 0 } else { 1 }; // C->0, T->1
        if is_del && n_rep == 0 {
            return UNCLASSIFIED; // v1 _REF_MISMATCH -> UNCLASSIFIED
        }
        let mut rep = if is_del { n_rep - 1 } else { n_rep };
        if rep > 5 {
            rep = 5;
        }
        return ID83_LUTS.id1[if is_del { 0 } else { 1 }][base_ct][rep];
    }
    let sb = if ilen < 5 { ilen } else { 5 };
    let si = sb - 2; // 0..3
    let rep;
    if is_del {
        // microhomology: longest prefix of the unit matching downstream
        let mut mh = 0usize;
        let mut kk = 1;
        while kk < ilen {
            let mut eq = true;
            let mut j = 0;
            while j < kk {
                if scan + j >= n || seq[scan + j] != buf[j] {
                    eq = false;
                    break;
                }
                j += 1;
            }
            if eq {
                mh = kk;
            }
            kk += 1;
        }
        if mh > 0 && n_rep <= 1 {
            let cap = MH_CAP[si];
            let m = if mh < cap { mh } else { cap };
            return ID83_LUTS.idm[si][m];
        }
        if n_rep == 0 {
            return UNCLASSIFIED; // _REF_MISMATCH
        }
        rep = n_rep - 1;
    } else {
        rep = n_rep;
    }
    let rep = if rep > 5 { 5 } else { rep };
    ID83_LUTS.idr[if is_del { 0 } else { 1 }][si][rep]
}
```

Add tests inside the existing `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn id83_1bp_del_in_repeat() {
        // seq: ...A C C C C... delete one C at a run of 4 C's.
        // REF="AC" ALT="A" (anchor A, deleted unit "C"), pos0 at the 'A'.
        // downstream run of C's from pos0+1: 4 -> is_del, rep=n_rep-1=3.
        // base C -> base_ct=0, kind Del=0 -> id1[0][0][3] = (0*2+0)*6+3 = 3
        let seq = b"AACCCCG"; // pos0 = 1 ('A'), then CCCC
        assert_eq!(id83_code(seq, 1, b"AC", b"A"), 3);
    }

    #[test]
    fn id83_1bp_ins() {
        // REF="A" ALT="AC" insert one C; downstream C run from pos0+1.
        // seq A C C C: n_rep counts inserted-unit "C" copies present downstream.
        // kind Ins=1, base C base_ct=0, rep=n_rep (no -1 for ins).
        let seq = b"ACCCG"; // pos0=0 ('A'); downstream from index1: C C C -> n_rep=3
        assert_eq!(id83_code(seq, 0, b"A", b"AC"), (1 * 2 + 0) * 6 + 3);
    }

    #[test]
    fn id83_rejects_non_acgt_unit() {
        let seq = b"ANNNG";
        assert_eq!(id83_code(seq, 0, b"AN", b"A"), UNCLASSIFIED);
    }
```

- [ ] **Step 2: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::classify'`
Expected: PASS (6 tests). If the microhomology or repeat index math is off, compare against `python/genoray/_mutcat/classify.py:134-259` line-by-line — the control flow here mirrors it exactly.

- [ ] **Step 3: Commit**

```bash
git add src/mutcat/classify.rs
git commit -m "feat(mutcat): ID83 indel classifier (port of _id83_kernel)"
```

### Task 3: DBS78 doublet table

**Files:**
- Modify: `src/mutcat/classify.rs`

**Interfaces:**
- Produces: `pub fn dbs78_code(r0: u8, a0: u8, r1: u8, a1: u8) -> u8` — DBS78 class-local index `0–77` or `UNCLASSIFIED`. Inputs are ASCII bases of the 2bp REF (`r0 r1`) and 2bp ALT (`a0 a1`). Ports `classify_dbs78` + `_build_dbs_table` (`classify.py:35-51,262-277`): try literal `REF>ALT`, else reverse-complement, against the canonical DBS78 list.

- [ ] **Step 1: Write the failing test** — append to `classify.rs`:

```rust
// Canonical DBS78 doublet labels, codebook order (must match codebook.py DBS78).
const DBS78_LABELS: [&str; 78] = [
    "AC>CA", "AC>CG", "AC>CT", "AC>GA", "AC>GG", "AC>GT", "AC>TA", "AC>TG", "AC>TT",
    "AT>CA", "AT>CC", "AT>CG", "AT>GA", "AT>GC", "AT>TA",
    "CC>AA", "CC>AG", "CC>AT", "CC>GA", "CC>GG", "CC>GT", "CC>TA", "CC>TG", "CC>TT",
    "CG>AT", "CG>GC", "CG>GT", "CG>TA", "CG>TC", "CG>TT",
    "CT>AA", "CT>AC", "CT>AG", "CT>GA", "CT>GC", "CT>GG", "CT>TA", "CT>TC", "CT>TG",
    "GC>AA", "GC>AG", "GC>AT", "GC>CA", "GC>CG", "GC>TA",
    "TA>AT", "TA>CG", "TA>CT", "TA>GC", "TA>GG", "TA>GT",
    "TC>AA", "TC>AG", "TC>AT", "TC>CA", "TC>CG", "TC>CT", "TC>GA", "TC>GG", "TC>GT",
    "TG>AA", "TG>AC", "TG>AT", "TG>CA", "TG>CC", "TG>CT", "TG>GA", "TG>GC", "TG>GT",
    "TT>AA", "TT>AC", "TT>AG", "TT>CA", "TT>CC", "TT>CG", "TT>GA", "TT>GC", "TT>GG",
];

#[inline]
fn comp(b: u8) -> u8 {
    match b {
        b'A' => b'T',
        b'T' => b'A',
        b'C' => b'G',
        b'G' => b'C',
        _ => b,
    }
}

// Build the (r0,a0,r1,a1) -> code table once. Bases A=0 C=1 G=2 T=3.
const BASES_ACGT: [u8; 4] = [b'A', b'C', b'G', b'T'];

fn dbs_index_of(key: &str) -> Option<u8> {
    DBS78_LABELS.iter().position(|&l| l == key).map(|i| i as u8)
}

/// DBS78 class-local index (0..=77) or `UNCLASSIFIED`. Try literal then
/// reverse-complement of the doublet REF>ALT.
pub fn dbs78_code(r0: u8, a0: u8, r1: u8, a1: u8) -> u8 {
    if base_index(r0) < 0 || base_index(r1) < 0 || base_index(a0) < 0 || base_index(a1) < 0 {
        return UNCLASSIFIED;
    }
    let literal = format!(
        "{}{}>{}{}",
        r0 as char, r1 as char, a0 as char, a1 as char
    );
    if let Some(c) = dbs_index_of(&literal) {
        return c;
    }
    // reverse-complement both sides
    let rc = format!(
        "{}{}>{}{}",
        comp(r1) as char,
        comp(r0) as char,
        comp(a1) as char,
        comp(a0) as char
    );
    dbs_index_of(&rc).unwrap_or(UNCLASSIFIED)
}
```

Tests in the `tests` module:

```rust
    #[test]
    fn dbs78_literal_hit() {
        // "AC>CA" is index 0.
        assert_eq!(dbs78_code(b'A', b'C', b'C', b'A'), 0);
    }

    #[test]
    fn dbs78_revcomp_fold() {
        // "GT>TG" is not literal; revcomp = "AC>CA" (index 0).
        assert_eq!(dbs78_code(b'G', b'T', b'T', b'G'), 0);
    }

    #[test]
    fn dbs78_all_16_doublets_map_or_uncl() {
        // Every ACGT^4 combo with r!=a on both is either a code < 78 or UNCLASSIFIED.
        for &r0 in &BASES_ACGT {
            for &r1 in &BASES_ACGT {
                for &a0 in &BASES_ACGT {
                    for &a1 in &BASES_ACGT {
                        let c = dbs78_code(r0, a0, r1, a1);
                        assert!(c < 78 || c == UNCLASSIFIED);
                    }
                }
            }
        }
    }
```

- [ ] **Step 2: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::classify'`
Expected: PASS (9 tests).

- [ ] **Step 3: Commit**

```bash
git add src/mutcat/classify.rs
git commit -m "feat(mutcat): DBS78 doublet classifier (literal + revcomp fold)"
```

---

## Phase 2 — Sidecar layout + I/O

### Task 4: `mutcat/` path helpers

**Files:**
- Modify: `src/layout.rs`

**Interfaces:**
- Produces on `ContigPaths`: `mutcat_dir(&self, sub: MutcatSub) -> PathBuf`, `mutcat_code(&self, sub) -> PathBuf`, `mutcat_ref(&self, sub) -> PathBuf`, where `MutcatSub { VkSnp, VkIndel, DenseSnp, DenseIndel }`.

- [ ] **Step 1: Write the failing test** — add to `src/layout.rs`:

```rust
/// The four sub-streams a mutcat sidecar mirrors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutcatSub {
    VkSnp,
    VkIndel,
    DenseSnp,
    DenseIndel,
}

impl MutcatSub {
    fn dir_name(self) -> &'static str {
        match self {
            MutcatSub::VkSnp => "var_key_snp",
            MutcatSub::VkIndel => "var_key_indel",
            MutcatSub::DenseSnp => "dense_snp",
            MutcatSub::DenseIndel => "dense_indel",
        }
    }
    /// Whether this sub-stream carries a 2-bit ref-base stream (snp only).
    pub fn has_ref(self) -> bool {
        matches!(self, MutcatSub::VkSnp | MutcatSub::DenseSnp)
    }
}
```

Add methods inside `impl ContigPaths`:

```rust
    fn mutcat_dir(&self, sub: MutcatSub) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("mutcat")
            .join(sub.dir_name())
    }
    pub fn mutcat_code(&self, sub: MutcatSub) -> PathBuf {
        self.mutcat_dir(sub).join("code.bin")
    }
    pub fn mutcat_ref(&self, sub: MutcatSub) -> PathBuf {
        self.mutcat_dir(sub).join("ref.bin")
    }
    /// Directory created before writing a sidecar sub-stream.
    pub fn mutcat_sub_dir(&self, sub: MutcatSub) -> PathBuf {
        self.mutcat_dir(sub)
    }
```

Test in `layout.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn mutcat_paths() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(
            p.mutcat_code(MutcatSub::VkSnp),
            Path::new("/out/chr1/mutcat/var_key_snp/code.bin")
        );
        assert_eq!(
            p.mutcat_ref(MutcatSub::DenseSnp),
            Path::new("/out/chr1/mutcat/dense_snp/ref.bin")
        );
        assert!(MutcatSub::VkSnp.has_ref());
        assert!(!MutcatSub::VkIndel.has_ref());
    }
```

- [ ] **Step 2: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion layout::tests::mutcat_paths'`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/layout.rs
git commit -m "feat(mutcat): mutcat/ sidecar path helpers"
```

### Task 5: Sidecar writer + reader

**Files:**
- Create: `src/mutcat/sidecar.rs`
- Modify: `src/mutcat/mod.rs` (add `pub mod sidecar;`)

**Interfaces:**
- Consumes: `crate::layout::{ContigPaths, MutcatSub}`, `svar2_codec::{pack_snp_keys, unpack_snp_key_at}`, `crate::query::sidecar::mmap_file`.
- Produces:
  - `pub fn write_sidecar(paths: &ContigPaths, sub: MutcatSub, codes: &[u8], ref_codes: Option<&[u8]>) -> std::io::Result<()>` — writes `code.bin` (raw `u8`) and, for snp subs, `ref.bin` (2-bit packed via `pack_snp_keys`). `ref_codes` are 2-bit values `0–3`.
  - `pub struct MutcatView { code: Option<Mmap>, ref_packed: Option<Mmap>, n: usize }` with `code_at(i) -> u8` and `ref_at(i) -> u8` (2-bit).
  - `pub fn open_sidecar(paths: &ContigPaths, sub: MutcatSub) -> std::io::Result<MutcatView>` — empty view if files absent.

- [ ] **Step 1: Write the failing test** — `src/mutcat/sidecar.rs`:

```rust
//! Read/write the per-contig mutcat sidecar. `code.bin` is raw u8 (one per
//! record); `ref.bin` (snp subs only) is 2-bit packed A/C/T/G ref bases.

use std::fs;
use std::io::Write;
use std::path::Path;

use memmap2::Mmap;

use crate::layout::{ContigPaths, MutcatSub};
use crate::query::sidecar::mmap_file;
use svar2_codec::{pack_snp_keys, unpack_snp_key_at};

pub fn write_sidecar(
    paths: &ContigPaths,
    sub: MutcatSub,
    codes: &[u8],
    ref_codes: Option<&[u8]>,
) -> std::io::Result<()> {
    let dir = paths.mutcat_sub_dir(sub);
    fs::create_dir_all(&dir)?;
    write_bytes(&paths.mutcat_code(sub), codes)?;
    if sub.has_ref() {
        let refs = ref_codes.expect("snp sub-stream requires ref_codes");
        debug_assert_eq!(refs.len(), codes.len());
        let packed = pack_snp_keys(refs);
        write_bytes(&paths.mutcat_ref(sub), &packed)?;
    }
    Ok(())
}

fn write_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(bytes)?;
    f.flush()
}

pub struct MutcatView {
    code: Option<Mmap>,
    ref_packed: Option<Mmap>,
    pub n: usize,
}

impl MutcatView {
    /// Class-local mutation code at record `i` (u8; may be a sentinel).
    #[inline]
    pub fn code_at(&self, i: usize) -> u8 {
        match &self.code {
            Some(m) => m[i],
            None => crate::mutcat::NOT_ANNOTATED,
        }
    }
    /// 2-bit reference-base code at snp record `i` (`0–3`). Panics if no ref stream.
    #[inline]
    pub fn ref_at(&self, i: usize) -> u8 {
        let m = self.ref_packed.as_ref().expect("ref_at on a stream with no ref.bin");
        unpack_snp_key_at(&m[..], i)
    }
}

pub fn open_sidecar(paths: &ContigPaths, sub: MutcatSub) -> std::io::Result<MutcatView> {
    let code = mmap_file(&paths.mutcat_code(sub))?;
    let n = code.as_ref().map(|m| m.len()).unwrap_or(0);
    let ref_packed = if sub.has_ref() {
        mmap_file(&paths.mutcat_ref(sub))?
    } else {
        None
    };
    Ok(MutcatView { code, ref_packed, n })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snp_sidecar_round_trips_code_and_ref() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let codes = [5u8, 95, 254, 0];
        let refs = [1u8, 3, 0, 2]; // C,G(→ codec 3),A,T
        write_sidecar(&paths, MutcatSub::VkSnp, &codes, Some(&refs)).unwrap();
        let v = open_sidecar(&paths, MutcatSub::VkSnp).unwrap();
        assert_eq!(v.n, 4);
        for i in 0..4 {
            assert_eq!(v.code_at(i), codes[i]);
            assert_eq!(v.ref_at(i), refs[i]);
        }
    }

    #[test]
    fn indel_sidecar_has_no_ref() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let codes = [10u8, 82, 255];
        write_sidecar(&paths, MutcatSub::VkIndel, &codes, None).unwrap();
        let v = open_sidecar(&paths, MutcatSub::VkIndel).unwrap();
        assert_eq!(v.n, 3);
        assert_eq!(v.code_at(1), 82);
    }

    #[test]
    fn missing_sidecar_opens_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let v = open_sidecar(&paths, MutcatSub::DenseSnp).unwrap();
        assert_eq!(v.n, 0);
        assert_eq!(v.code_at(0), crate::mutcat::NOT_ANNOTATED);
    }
}
```

Add `pub mod sidecar;` to `src/mutcat/mod.rs`. Ensure `tempfile` is a dev-dependency (it is — used across `src/`).

- [ ] **Step 2: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::sidecar'`
Expected: PASS (3 tests).

- [ ] **Step 3: Commit**

```bash
git add src/mutcat/sidecar.rs src/mutcat/mod.rs
git commit -m "feat(mutcat): sidecar reader/writer (u8 code + 2-bit ref)"
```

---

## Phase 3 — Classification pass

### Task 6: Classify one contig's finalized records → sidecar

**Files:**
- Create: `src/mutcat/annotate.rs`
- Modify: `src/mutcat/mod.rs` (add `pub mod annotate;`)

**Interfaces:**
- Consumes: `crate::query::ContigReader` (opened contig), the classifiers (Task 1–3), `write_sidecar` (Task 5), a reference sequence slice provider.
- Produces: `pub fn annotate_contig(reader: &ContigReader, paths: &ContigPaths, ref_seq: &[u8]) -> std::io::Result<()>`. `ref_seq` is the whole contig's ASCII sequence (0-based). It classifies each of the four sub-streams' records and writes the sidecar. For snp subs it also emits the per-record 2-bit ref base (`encode_snp_2bit(ref_seq[pos])`).

**Notes for the implementer (record reconstruction):**
- The `ContigReader` exposes `vk_snp`/`vk_indel` (`SubStreamView`, per-call `positions()` + packed keys) and `dense_snp`/`dense_indel` (`Option<DenseView>`, per-variant `positions()` + keys). Add small `pub(crate)` accessors on `ContigReader`/`SubStreamView`/`DenseView` if needed (e.g. `vk_snp_len()`, a way to read the whole packed positions/keys) — mirror the existing `vk_slice` access patterns in `src/query/reader.rs:119`.
- SNP ALT: `decode_snp_2bit(unpack_snp_key_at(keys, i))`. SNP REF/flanks: `ref_seq[pos-1], ref_seq[pos], ref_seq[pos+1]` (guard bounds → `UNCLASSIFIED` at contig ends).
- Indel key: `decode_key(u32_key)` → `Inline{alt}` (ALT bytes), `PureDel{ilen}` (REF = `ref_seq[pos..pos + (-ilen) + 1]`, ALT = `ref_seq[pos..pos+1]`, per the DEL empty-allele anchor convention), or `Lookup{row}` (ALT from `reader.lut`). REF for an inline INS = `ref_seq[pos..pos+1]` (single anchor). Build `refa`/`alta` byte vecs then call `id83_code(ref_seq, pos as usize, &refa, &alta)`.

- [ ] **Step 1: Write the failing test** — `src/mutcat/annotate.rs`. Build a tiny synthetic contig with a helper that hand-writes the four sub-streams (or reuse an existing test fixture builder if `tests/common/mod.rs` has one). Minimal viable test: one var_key/snp record and assert the sidecar code equals the direct `sbs96_code` of its context.

```rust
//! Classify a finished contig's records and write the mutcat sidecar. Runs at
//! write time (FASTA already open) or post-hoc.

use crate::layout::{ContigPaths, MutcatSub};
use crate::mutcat::classify::{id83_code, sbs96_code};
use crate::mutcat::sidecar::write_sidecar;
use crate::mutcat::{NOT_ANNOTATED, UNCLASSIFIED};
use crate::query::ContigReader;
use svar2_codec::{decode_key, decode_snp_2bit, encode_snp_2bit, DecodedKey};

/// Classify every record of `reader`'s four sub-streams against `ref_seq` (the
/// whole contig, 0-based ASCII) and write the sidecar under `paths`.
pub fn annotate_contig(
    reader: &ContigReader,
    paths: &ContigPaths,
    ref_seq: &[u8],
) -> std::io::Result<()> {
    // --- var_key/snp (per call) ---
    {
        let (positions, keys) = reader.vk_snp_records(); // &[u32], packed 2-bit keys
        let mut codes = Vec::with_capacity(positions.len());
        let mut refs = Vec::with_capacity(positions.len());
        for (i, &pos) in positions.iter().enumerate() {
            let (code, refc) = snp_record(ref_seq, pos, snp_alt(keys, i));
            codes.push(code);
            refs.push(refc);
        }
        write_sidecar(paths, MutcatSub::VkSnp, &codes, Some(&refs))?;
    }
    // --- dense/snp (per variant) ---
    if let Some((positions, keys)) = reader.dense_snp_records() {
        let mut codes = Vec::with_capacity(positions.len());
        let mut refs = Vec::with_capacity(positions.len());
        for (i, &pos) in positions.iter().enumerate() {
            let (code, refc) = snp_record(ref_seq, pos, decode_snp_2bit(keys[i]));
            codes.push(code);
            refs.push(refc);
        }
        write_sidecar(paths, MutcatSub::DenseSnp, &codes, Some(&refs))?;
    }
    // --- var_key/indel + dense/indel (per record) ---
    annotate_indel(reader, paths, ref_seq, MutcatSub::VkIndel)?;
    annotate_indel(reader, paths, ref_seq, MutcatSub::DenseIndel)?;
    Ok(())
}

/// SBS96 code + 2-bit ref base for a SNP at 0-based `pos` with ASCII `alt`.
fn snp_record(ref_seq: &[u8], pos: u32, alt: u8) -> (u8, u8) {
    let p = pos as usize;
    let n = ref_seq.len();
    if p == 0 || p + 1 >= n {
        return (NOT_ANNOTATED, 0); // contig-end flank missing → not classified
    }
    let refb = ref_seq[p];
    let code = sbs96_code(ref_seq[p - 1], refb, alt, ref_seq[p + 1]);
    (code, encode_snp_2bit(refb))
}

fn snp_alt(packed_keys: &[u8], i: usize) -> u8 {
    decode_snp_2bit(svar2_codec::unpack_snp_key_at(packed_keys, i))
}

fn annotate_indel(
    reader: &ContigReader,
    paths: &ContigPaths,
    ref_seq: &[u8],
    sub: MutcatSub,
) -> std::io::Result<()> {
    let recs = reader.indel_records(sub); // Option<(&[u32] positions, &[u32] keys)>
    let (positions, keys) = match recs {
        Some(r) => r,
        None => return write_sidecar(paths, sub, &[], None),
    };
    let mut codes = Vec::with_capacity(positions.len());
    for (i, &pos) in positions.iter().enumerate() {
        codes.push(indel_code(reader, ref_seq, pos, keys[i]));
    }
    write_sidecar(paths, sub, &codes, None)
}

/// ID83 code for one indel record, reconstructing REF/ALT from key + reference.
fn indel_code(reader: &ContigReader, ref_seq: &[u8], pos: u32, key: u32) -> u8 {
    let p = pos as usize;
    if p >= ref_seq.len() {
        return NOT_ANNOTATED;
    }
    let anchor = ref_seq[p];
    let (refa, alta): (Vec<u8>, Vec<u8>) = match decode_key(key) {
        DecodedKey::Inline { alt } => {
            // atomized INS/anchor: REF = single anchor base; ALT = alt bytes.
            (vec![anchor], alt)
        }
        DecodedKey::PureDel { ilen } => {
            let dl = (-ilen) as usize;
            if p + dl >= ref_seq.len() {
                return NOT_ANNOTATED;
            }
            // REF = anchor + dl deleted bases; ALT = anchor only.
            (ref_seq[p..p + dl + 1].to_vec(), vec![anchor])
        }
        DecodedKey::Lookup { row } => {
            let alt = reader.lut_allele(row); // Vec<u8>
            (vec![anchor], alt)
        }
    };
    id83_code(ref_seq, p, &refa, &alta)
}
```

Also add the `pub(crate)` accessors this task needs on `ContigReader` (in `src/query/reader.rs`): `vk_snp_records()`, `dense_snp_records()`, `indel_records(sub)`, `lut_allele(row)`. Each is a thin wrapper over the existing mmap fields (`self.vk_snp.positions()`, `as_bytes(&self.vk_snp.keys)`, `self.dense_snp.as_ref().map(...)`, `self.lut.as_ref().unwrap().get_allele(row)`).

Test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutcat::classify::sbs96_code;

    // Uses the shared fixture builder (tests/common) or a hand-built contig dir.
    // Minimal check: a single var_key/snp record's sidecar code equals the
    // direct sbs96_code of its reference context.
    #[test]
    fn snp_record_matches_direct_classify() {
        let ref_seq = b"TACGT"; // pos 2 = 'C', 5'=A(1), 3'=G(3)
        let (code, refc) = snp_record(ref_seq, 2, b'A');
        assert_eq!(code, sbs96_code(b'A', b'C', b'A', b'G'));
        assert_eq!(refc, svar2_codec::encode_snp_2bit(b'C'));
    }

    #[test]
    fn contig_end_snp_is_not_annotated() {
        let ref_seq = b"CG";
        assert_eq!(snp_record(ref_seq, 0, b'A').0, NOT_ANNOTATED);
    }
}
```

- [ ] **Step 2: Implement the `ContigReader` accessors** referenced above; run `cargo build` to confirm they compile.

Run: `pixi run -e lint bash -lc 'cargo build --no-default-features --features conversion'`
Expected: builds clean.

- [ ] **Step 3: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::annotate'`
Expected: PASS (2 tests).

- [ ] **Step 4: Commit**

```bash
git add src/mutcat/annotate.rs src/mutcat/mod.rs src/query/reader.rs
git commit -m "feat(mutcat): classify finalized contig records into the sidecar"
```

---

## Phase 4 — Cost-model integration (write-time)

### Task 7: `choose_representation` with sidecar bits

**Files:**
- Modify: `src/cost_model.rs`
- Modify: `src/rvk.rs` (`dense2sparse_vk` call site)

**Interfaces:**
- Produces: `choose_representation(class, n_samples, ploidy, x_calls, sidecar_bits)` — new trailing `sidecar_bits: u64` parameter. `var_key_bits += x_calls * sidecar_bits`; `dense_bits += sidecar_bits`. Existing callers pass `0`.
- Add `pub const SIDECAR_BITS_SNP: u64 = 10;` and `pub const SIDECAR_BITS_INDEL: u64 = 8;` (8 code + 2 ref for snp; 8 code for indel).

- [ ] **Step 1: Update the existing tests + add new ones** in `src/cost_model.rs`. Change the signature and thread `sidecar_bits` through:

```rust
pub const SIDECAR_BITS_SNP: u64 = 10; // 8-bit code + 2-bit ref
pub const SIDECAR_BITS_INDEL: u64 = 8; // 8-bit code

#[inline]
pub fn choose_representation(
    class: Class,
    n_samples: usize,
    ploidy: usize,
    x_calls: usize,
    sidecar_bits: u64,
) -> Representation {
    let np = (n_samples as u64) * (ploidy as u64);
    let per_call = POS_BITS + key_bits(class) + sidecar_bits;
    let var_key_bits = (x_calls as u64) * per_call;
    let dense_bits = POS_BITS + key_bits(class) + np + sidecar_bits;
    if dense_bits < var_key_bits {
        Representation::Dense
    } else {
        Representation::VarKey
    }
}
```

Update every existing test in `cost_model.rs` to pass `0` as the final arg, and add:

```rust
    #[test]
    fn sidecar_bits_shift_snp_crossover_toward_dense() {
        // Without sidecar: np=2000 → dense wins at x>=60 (see test_snp_crossover_np2000).
        // With +10 sidecar/call on var_key and +10 once on dense:
        // dense = 32+2+2000+10 = 2044; per_call = 34+10 = 44.
        // dense < 44x → x > 46.45 → dense at x>=47 (vs 60 without).
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 46, SIDECAR_BITS_SNP),
            Representation::VarKey
        );
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 47, SIDECAR_BITS_SNP),
            Representation::Dense
        );
    }

    #[test]
    fn zero_sidecar_matches_legacy() {
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 60, 0),
            Representation::Dense
        );
    }
```

- [ ] **Step 2: Thread through `rvk.rs`.** In `dense2sparse_vk` (`src/rvk.rs:177`), add a `sidecar_bits` parameter to the function (default `0`) and select per-class:

```rust
// at the call site (rvk.rs:177), replace choose_representation(class, num_samples, ploidy, x)
let bits = if sidecar_bits_enabled {
    match class {
        Class::Snp => crate::cost_model::SIDECAR_BITS_SNP,
        Class::Indel => crate::cost_model::SIDECAR_BITS_INDEL,
    }
} else {
    0
};
match choose_representation(class, num_samples, ploidy, x, bits) {
```

Add `sidecar_bits_enabled: bool` to `dense2sparse_vk`'s signature and thread it from the orchestrator (Task 12 wires the `signatures` flag; until then pass `false`). Update `dense2sparse_vk`'s existing callers (grep `dense2sparse_vk`) to pass `false`.

- [ ] **Step 3: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion cost_model rvk'`
Expected: PASS (all existing + 2 new).

- [ ] **Step 4: Commit**

```bash
git add src/cost_model.rs src/rvk.rs
git commit -m "feat(mutcat): cost model accounts for sidecar bits when signatures on"
```

---

## Phase 5 — Streaming count matrix

### Task 8: Per-column SNV merge + DBS pairing core

**Files:**
- Create: `src/mutcat/count.rs`
- Modify: `src/mutcat/mod.rs` (add `pub mod count;`)

**Interfaces:**
- Produces the pure pairing routine, decoupled from I/O so it is unit-testable:
  `pub fn emit_snv_codes(snvs: &[SnvCall], out: &mut impl FnMut(usize))` where
  `pub struct SnvCall { pub pos: u32, pub sbs: u8, pub ref_i: u8, pub alt_i: u8 }`
  (`sbs` = SBS96 class-local index or sentinel; `ref_i`/`alt_i` = 2-bit base codes for DBS). `snvs` is one haplotype's SNVs **sorted by pos, deduped**. `out(unified_code)` is called once per emitted unified code (`SBS96_OFFSET + sbs` or `DBS78_OFFSET + dbs`), skipping sentinels. Implements v1's isolation rule (`classify.py:_entry_codes_kernel`): an adjacent `Δpos==1` pair becomes a DBS **iff** neither neighbor has another adjacent SNV (runs of ≥3 stay SBS).

- [ ] **Step 1: Write the failing test** — `src/mutcat/count.rs`:

```rust
//! Streaming mutation-catalogue counting with on-the-fly DBS pairing.

use crate::mutcat::classify::dbs78_code;
use crate::mutcat::{DBS78_OFFSET, N_CODES, SBS96_OFFSET, UNCLASSIFIED};
use svar2_codec::decode_snp_2bit;

/// One SNV on a single haplotype (already position-sorted & deduped).
#[derive(Debug, Clone, Copy)]
pub struct SnvCall {
    pub pos: u32,
    pub sbs: u8,   // SBS96 class-local index or sentinel
    pub ref_i: u8, // 2-bit ref base code
    pub alt_i: u8, // 2-bit alt base code
}

/// Emit one unified mutation code per SNV (SBS or, for isolated adjacent pairs,
/// DBS for both members). Sentinels are skipped. Mirrors v1 isolation logic.
pub fn emit_snv_codes(snvs: &[SnvCall], out: &mut impl FnMut(usize)) {
    let n = snvs.len();
    let mut j = 0;
    while j < n {
        let v = snvs[j];
        // try to pair v with the next SNV
        if j + 1 < n && snvs[j + 1].pos == v.pos + 1 {
            let w = snvs[j + 1];
            // isolation: no adjacent SNV before v, none after w.
            let before = j > 0 && snvs[j - 1].pos + 1 == v.pos;
            let after = j + 2 < n && snvs[j + 2].pos == w.pos + 1;
            if !before && !after {
                let code = dbs78_code(
                    decode_snp_2bit(v.ref_i),
                    decode_snp_2bit(v.alt_i),
                    decode_snp_2bit(w.ref_i),
                    decode_snp_2bit(w.alt_i),
                );
                if code != UNCLASSIFIED {
                    out(DBS78_OFFSET + code as usize);
                    out(DBS78_OFFSET + code as usize);
                    j += 2;
                    continue;
                }
            }
        }
        if (v.sbs as usize) < 96 {
            out(SBS96_OFFSET + v.sbs as usize);
        }
        j += 1;
    }
    debug_assert!(N_CODES == 257);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snv(pos: u32, ref_b: u8, alt_b: u8) -> SnvCall {
        use crate::mutcat::classify::sbs96_code;
        SnvCall {
            pos,
            sbs: sbs96_code(b'A', ref_b, alt_b, b'A'), // dummy flanks for the test
            ref_i: svar2_codec::encode_snp_2bit(ref_b),
            alt_i: svar2_codec::encode_snp_2bit(alt_b),
        }
    }

    #[test]
    fn isolated_pair_emits_two_dbs() {
        let snvs = [snv(10, b'A', b'C'), snv(11, b'C', b'A')];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        // AC>CA is DBS index 0 → unified 96; both members contribute.
        assert_eq!(codes, vec![96, 96]);
    }

    #[test]
    fn run_of_three_stays_sbs() {
        let snvs = [snv(10, b'C', b'A'), snv(11, b'C', b'A'), snv(12, b'C', b'A')];
        let mut n_dbs = 0;
        emit_snv_codes(&snvs, &mut |c| {
            if c >= DBS78_OFFSET {
                n_dbs += 1;
            }
        });
        assert_eq!(n_dbs, 0, "runs of 3 must stay SBS");
    }

    #[test]
    fn non_adjacent_snvs_are_sbs() {
        let snvs = [snv(10, b'C', b'A'), snv(20, b'C', b'A')];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        assert!(codes.iter().all(|&c| c < DBS78_OFFSET));
        assert_eq!(codes.len(), 2);
    }
}
```

Add `pub mod count;` to `src/mutcat/mod.rs`.

- [ ] **Step 2: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::count'`
Expected: PASS (3 tests).

- [ ] **Step 3: Commit**

```bash
git add src/mutcat/count.rs src/mutcat/mod.rs
git commit -m "feat(mutcat): SNV->SBS/DBS emission core with v1 isolation rule"
```

### Task 9: Full count matrix over a contig

**Files:**
- Modify: `src/mutcat/count.rs`

**Interfaces:**
- Consumes: `ContigReader` + the four `MutcatView`s (Task 5), `emit_snv_codes` (Task 8).
- Produces: `pub fn count_contig(reader: &ContigReader, sidecars: &Sidecars, per_sample: bool, acc: &mut Array2<i64>)` where `acc` is `(n_samples, N_CODES)` and `Sidecars` bundles the four `MutcatView`s. For each column `(sample, ploid)`:
  1. Gather the column's SNVs from `var_key/snp` (per-call: positions[o0..o1], code/ref from `VkSnp` view at absolute index; alt from the packed key stream) and `dense/snp` (per-variant carried bit set for this hap; code/ref from `DenseSnp` view at the variant col). Merge-sort by pos, dedup, build `Vec<SnvCall>`, run `emit_snv_codes`.
  2. Gather the column's indels from `var_key/indel` + `dense/indel`; each emits `ID83_OFFSET + code` directly (no pairing).
  3. Accumulate: `per_sample` marks each code once per sample (set to 1), else increments.

**Notes:** the var_key column bounds come from `SubStreamView::column(col)` (`offsets[col]..offsets[col+1]`); the sidecar `VkSnp` code/ref arrays are aligned to the same absolute per-call index, so `view.code_at(abs_i)` / `view.ref_at(abs_i)` line up. For dense, iterate the dense variant table and test `DenseView::carried(hap, col)`; the `DenseSnp` sidecar index is the dense variant `col`.

- [ ] **Step 1: Write the failing test.** Reuse the fixture from Task 6 (a small built contig with a known genotype) so `count_contig` produces a known matrix. Minimal assertion: a single sample carrying two isolated adjacent SNVs yields exactly one DBS channel = 2 (allele mode) and the corresponding SBS channels = 0.

```rust
    #[test]
    fn count_contig_pairs_adjacent_snvs_for_one_sample() {
        // Build a 1-sample, ploidy-1 contig with two var_key SNVs at pos 10,11
        // both carried by the sample. Expect DBS channel == 2, no SBS.
        // (fixture builder lives in tests/common or is hand-written here)
        // ... assemble reader + sidecars ...
        // let mut acc = Array2::<i64>::zeros((1, N_CODES));
        // count_contig(&reader, &sidecars, false, &mut acc);
        // assert_eq!(acc[[0, DBS78_OFFSET + 0]], 2);
        // assert_eq!(acc.slice(s![0, 0..96]).sum(), 0);
    }
```

Flesh this out against whatever fixture helper exists (see `tests/common/mod.rs`); if none writes a full contig dir, hand-write the four sub-stream files + sidecars in the test using `write_sidecar` and the existing `layout`/`ndarray_npy` writers.

- [ ] **Step 2: Implement `count_contig` + `Sidecars`.** Follow the column-merge structure from `ContigReader::vk_slice` (`src/query/reader.rs:119`) for how to walk var_key columns and dense carriage.

- [ ] **Step 3: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::count'`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/mutcat/count.rs
git commit -m "feat(mutcat): streaming per-contig count matrix with DBS pairing"
```

### Task 10: Parallelize over columns

**Files:**
- Modify: `src/mutcat/count.rs`

**Interfaces:**
- Produces: `count_contig` uses `rayon` to process columns in parallel, each thread owning a private `(N_CODES,)` row accumulator keyed by sample (fold + reduce). Deterministic regardless of thread count.

- [ ] **Step 1: Write the failing test** — determinism: run `count_contig` on the Task 9 fixture with `rayon` thread pool sizes 1 and 4 (via `rayon::ThreadPoolBuilder`), assert identical matrices.

```rust
    #[test]
    fn count_is_thread_count_invariant() {
        // build fixture once; run under 1 and 4 threads; assert equal.
    }
```

- [ ] **Step 2: Implement** with `rayon` (already a dependency — see `Cargo.toml`). Fold per-column results into per-sample rows; since multiple columns map to the same sample (ploidy>1), reduce with `+=` (allele) or `max`/`or` (sample-mode presence collapses per sample after reduce).

- [ ] **Step 3: Run tests**

Run: `pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion mutcat::count'`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/mutcat/count.rs
git commit -m "perf(mutcat): parallelize count matrix over sample-columns"
```

---

## Phase 6 — PyO3 bindings + Python API

### Task 11: PyO3 methods on `PyContigReader`

**Files:**
- Create: `src/py_mutcat.rs`
- Modify: `src/lib.rs` (`mod py_mutcat;`)

**Interfaces:**
- Produces two `#[pymethods]` on `PyContigReader`:
  - `fn annotate_mutations(&self, base_out_dir: &str, chrom: &str, ref_seq: PyReadonlyArray1<u8>) -> PyResult<()>` — calls `mutcat::annotate::annotate_contig` with a fresh `ContigPaths`. `ref_seq` is the contig's ASCII bytes passed from Python's `Reference.contig_array`.
  - `fn count_matrix<'py>(&self, py: Python<'py>, base_out_dir: &str, chrom: &str, per_sample: bool) -> PyResult<Bound<'py, PyArray2<i64>>>` — opens the four sidecars, runs `count_contig`, returns the `(n_samples, N_CODES)` matrix.

- [ ] **Step 1: Write the binding** (mirror the numpy-array patterns in `src/py_query_batch.rs` / `src/py_query_ranges.rs`):

```rust
//! Python-facing mutcat: post-hoc annotation + count-matrix on PyContigReader.

use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use crate::layout::ContigPaths;
use crate::mutcat::annotate::annotate_contig;
use crate::mutcat::count::{count_contig, Sidecars};
use crate::mutcat::N_CODES;
use crate::py_query::PyContigReader;

#[pymethods]
impl PyContigReader {
    /// Classify this contig's records against `ref_seq` and write the sidecar.
    fn annotate_mutations(
        &self,
        base_out_dir: &str,
        chrom: &str,
        ref_seq: PyReadonlyArray1<u8>,
    ) -> PyResult<()> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        let seq = ref_seq.as_slice()?;
        annotate_contig(&self.inner, &paths, seq).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("annotate {chrom}: {e}"))
        })?;
        Ok(())
    }

    /// Build the `(n_samples, N_CODES)` count matrix for this contig.
    fn count_matrix<'py>(
        &self,
        py: Python<'py>,
        base_out_dir: &str,
        chrom: &str,
        per_sample: bool,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        let sidecars = Sidecars::open(&paths).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("open sidecar {chrom}: {e}"))
        })?;
        let mut acc = ndarray::Array2::<i64>::zeros((self.inner.n_samples, N_CODES));
        count_contig(&self.inner, &sidecars, per_sample, &mut acc);
        Ok(acc.to_pyarray(py))
    }
}
```

Add `Sidecars::open(paths)` to `count.rs` if not already present (opens the four `MutcatView`s). Expose `ContigReader::n_samples` (already a field; add a `pub(crate) fn n_samples()` or make the field visible to `py_mutcat`).

- [ ] **Step 2: Build the extension**

Run: `pixi run maturin develop`
Expected: builds and installs `genoray._core`.

- [ ] **Step 3: Smoke test from Python** (scratchpad, not committed): open an existing test SVAR2 store's `PyContigReader`, call `count_matrix(...)`, confirm it returns an `(n_samples, 257)` int64 array.

- [ ] **Step 4: Commit**

```bash
git add src/py_mutcat.rs src/lib.rs src/mutcat/count.rs src/query/reader.rs
git commit -m "feat(mutcat): PyContigReader.annotate_mutations + count_matrix"
```

### Task 12: Wire `signatures` into the conversion pipeline

**Files:**
- Modify: `src/lib.rs` (`run_conversion_pipeline` signature + orchestrator call)
- Modify: `src/orchestrator.rs` (thread `signatures` → `dense2sparse_vk` `sidecar_bits_enabled`, and run `annotate_contig` per contig after finalization using the reference already loaded)
- Modify: `python/genoray/_svar2.py` (`from_vcf(..., signatures=False)`)

**Interfaces:**
- `run_conversion_pipeline(..., signatures: bool)` — new trailing bool. When true: (a) `dense2sparse_vk` uses sidecar-aware cost model; (b) after each contig's sub-streams are finalized, open a `ContigReader` for it and call `annotate_contig` with the contig's reference sequence (already mmap'd for normalization/left-align).

- [ ] **Step 1: Add the Python kwarg + pass-through** in `_svar2.py`:

```python
    @classmethod
    def from_vcf(
        cls,
        out: str | Path,
        source: str | Path,
        reference: str | Path | None = None,
        *,
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        ploidy: int = 2,
        chunk_size: int = 25_000,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
        signatures: bool = False,
    ) -> int:
```

Add to the docstring: *"signatures: if True, classify SBS96/ID83 codes during the write and store the mutcat sidecar (factored into the dense/var_key cost model). Requires a reference; raises if `no_reference=True`."* Validate:

```python
        if signatures and no_reference:
            raise ValueError("signatures=True requires a reference (not no_reference).")
```

Pass `signatures` as the final arg to `_core.run_conversion_pipeline(...)`.

- [ ] **Step 2: Thread through Rust** `run_conversion_pipeline` → orchestrator. In the orchestrator's per-contig finalization, after sub-streams + `max_del` are written, if `signatures`:

```rust
let reader = crate::query::ContigReader::open(out_dir, chrom, n_samples, ploidy)?;
let paths = crate::layout::ContigPaths::new(out_dir, chrom);
let ref_seq = /* the contig's ASCII sequence already loaded for normalization */;
crate::mutcat::annotate::annotate_contig(&reader, &paths, &ref_seq)?;
```

(If the reference bytes aren't retained past normalization, re-fetch via the existing reference reader used in `normalize.rs` — reuse that loader, don't add a new FASTA dependency.)

- [ ] **Step 3: Build + integration test**

Run: `pixi run maturin develop`
Then a scratchpad script: `SparseVar2.from_vcf(out, test_vcf, reference=fasta, signatures=True, overwrite=True)` on an existing fixture; confirm `out/<contig>/mutcat/var_key_snp/code.bin` exists and `PyContigReader(...).count_matrix(...)` is non-trivial.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/orchestrator.rs python/genoray/_svar2.py
git commit -m "feat(mutcat): signatures=True annotation during from_vcf write"
```

### Task 13: Python `_MutcatMixin` (annotate/matrix/assign)

**Files:**
- Create: `python/genoray/_svar2_mutcat.py`
- Modify: `python/genoray/_svar2.py` (inherit the mixin)

**Interfaces:**
- `_MutcatMixin.annotate_mutations(self, reference, *, contigs=None)` — post-hoc. For each in-scope contig, open its `PyContigReader` and call `.annotate_mutations(str(self.path), contig, reference.contig_array(contig))`. Stamp `meta.json` with `mutcat_version` + `mutcat_contigs`.
- `_MutcatMixin.mutation_matrix(self, kind, *, count="allele")` — sum each contig's `count_matrix(per_sample=(count=="sample"))`, slice the `kind` code range via `genoray._mutcat.codebook.code_ranges()`, and build the codebook-ordered `pl.DataFrame` (`MutationType` + per-sample columns), matching v1's `count_matrix` output shape. For `count="sample"`, presence must be OR-combined across contigs (clip to 1 after summing).
- `_MutcatMixin.assign_signatures(self, kind, *, reference=None, count="allele", max_delta=0.01, min_activity=0.005, n_jobs=1, backend="loky")` — build the matrix, then call `genoray._signatures.fit_signatures` (delegating `reference=None` → `cosmic_signatures(kind)`), identical to v1's `assign_signatures`.

- [ ] **Step 1: Write the failing test** in `tests/test_svar2_mutcat.py`:

```python
import numpy as np
import polars as pl
import pytest
import genoray
from genoray import SparseVar2


def test_mutation_matrix_shape_and_labels(tmp_path, small_vcf, small_fasta):
    out = tmp_path / "store.svar2"
    SparseVar2.from_vcf(out, small_vcf, reference=small_fasta, overwrite=True)
    sv2 = SparseVar2(out)
    sv2.annotate_mutations(small_fasta)
    mm = sv2.mutation_matrix("SBS96", count="allele")
    assert mm.columns[0] == "MutationType"
    assert mm.height == 96
    assert set(mm.columns[1:]) == set(sv2.available_samples)
    assert mm.select(pl.exclude("MutationType")).to_numpy().dtype.kind in "iu"
```

(`small_vcf`/`small_fasta` fixtures: reuse whatever `tests/conftest.py` or `tests/data/` already provides for SVAR2 e2e tests — check `tests/test_*` for the existing fixture names before writing new ones.)

- [ ] **Step 2: Write the mixin.**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

import genoray._core as _core
from genoray._mutcat import MUTCAT_VERSION, N_CODES, code_ranges, labels
from genoray._reference import Reference
from genoray._signatures import cosmic_signatures, fit_signatures, _load_signature_file


class _MutcatMixin:
    def annotate_mutations(
        self, reference: "Reference | str | Path", *, contigs: "list[str] | None" = None
    ) -> None:
        if not isinstance(reference, Reference):
            reference = Reference.from_path(reference)
        scope = self.contigs if contigs is None else [
            c for c in self.contigs if c in set(contigs)
        ]
        for contig in scope:
            seq = reference.contig_array(contig).astype(np.uint8, copy=False)
            self._readers[contig].annotate_mutations(str(self.path), contig, seq)
        meta_path = self.path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["mutcat_version"] = MUTCAT_VERSION
        meta["mutcat_contigs"] = scope
        meta_path.write_text(json.dumps(meta))

    def mutation_matrix(
        self, kind: Literal["SBS96", "DBS78", "ID83"], *, count: Literal["allele", "sample"] = "allele"
    ) -> pl.DataFrame:
        if kind not in ("SBS96", "DBS78", "ID83"):
            raise ValueError(f"Unknown matrix kind {kind!r}.")
        if count not in ("allele", "sample"):
            raise ValueError(f"Unknown count mode {count!r}.")
        per_sample = count == "sample"
        total = np.zeros((self.n_samples, N_CODES), dtype=np.int64)
        for contig in self.contigs:
            total += self._readers[contig].count_matrix(str(self.path), contig, per_sample)
        if per_sample:
            np.clip(total, 0, 1, out=total)  # presence OR across contigs
        lo, hi = code_ranges()[kind]
        block = total[:, lo:hi]
        out: dict[str, object] = {"MutationType": labels(kind)}
        for si, name in enumerate(self.available_samples):
            out[name] = block[si]
        return pl.DataFrame(out)

    def assign_signatures(
        self,
        kind: Literal["SBS96", "DBS78", "ID83"],
        *,
        reference: "pl.DataFrame | str | Path | None" = None,
        count: Literal["allele", "sample"] = "allele",
        max_delta: float = 0.01,
        min_activity: float = 0.005,
        n_jobs: int = 1,
        backend: str = "loky",
    ) -> pl.DataFrame:
        catalogue = self.mutation_matrix(kind, count=count)
        if reference is None:
            ref = cosmic_signatures(kind)
        elif isinstance(reference, pl.DataFrame):
            ref = reference
        else:
            ref = _load_signature_file(reference)
        return fit_signatures(
            catalogue, ref, max_delta=max_delta, min_activity=min_activity,
            n_jobs=n_jobs, backend=backend,
        )
```

Make `SparseVar2` inherit `_MutcatMixin` in `_svar2.py`:

```python
from genoray._svar2_mutcat import _MutcatMixin
class SparseVar2(_BatchQueryMixin, _DecodeMixin, _MutcatMixin):
```

- [ ] **Step 3: Build + run**

Run: `pixi run maturin develop && pixi run pytest tests/test_svar2_mutcat.py::test_mutation_matrix_shape_and_labels -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add python/genoray/_svar2_mutcat.py python/genoray/_svar2.py tests/test_svar2_mutcat.py
git commit -m "feat(mutcat): SparseVar2 annotate_mutations/mutation_matrix/assign_signatures"
```

---

## Phase 7 — Parity, docs, housekeeping

### Task 14: v1 parity + round-trip tests

**Files:**
- Modify: `tests/test_svar2_mutcat.py`

**Interfaces:** consumes v1 `SparseVar` and `SparseVar2` built from the same VCF.

- [ ] **Step 1: Write the parity test** (biallelic-SNV + simple-indel fixture, where v1 and SVAR2 atomization agree):

```python
def test_matrix_parity_with_v1(tmp_path, small_vcf, small_fasta):
    # v1
    from genoray import SparseVar
    v1_out = tmp_path / "v1.svar"
    SparseVar.from_vcf(v1_out, small_vcf, reference=small_fasta)  # match existing v1 API
    v1 = SparseVar(v1_out)
    v1.annotate_mutations(small_fasta)

    # v2
    v2_out = tmp_path / "v2.svar2"
    SparseVar2.from_vcf(v2_out, small_vcf, reference=small_fasta, overwrite=True)
    v2 = SparseVar2(v2_out)
    v2.annotate_mutations(small_fasta)

    for kind in ("SBS96", "ID83", "DBS78"):
        for count in ("allele", "sample"):
            a = v1.mutation_matrix(kind, count=count)
            b = v2.mutation_matrix(kind, count=count)
            # align sample column order, compare counts
            cols = ["MutationType", *v2.available_samples]
            assert a.select(cols).equals(b.select(cols)), f"{kind}/{count} mismatch"
```

If exact equality fails, first confirm both stores atomize identically on the fixture (inspect a few variants); if atomization genuinely differs on complex records, narrow the fixture to biallelic SNVs + pure indels and document the limitation in the test docstring. SVAR2's classification of the normalized/atomized record is the authoritative behavior.

- [ ] **Step 2: Write the write-time round-trip test:**

```python
def test_write_time_signatures_match_posthoc(tmp_path, small_vcf, small_fasta):
    a = tmp_path / "wtime.svar2"
    SparseVar2.from_vcf(a, small_vcf, reference=small_fasta, signatures=True, overwrite=True)
    b = tmp_path / "posthoc.svar2"
    SparseVar2.from_vcf(b, small_vcf, reference=small_fasta, overwrite=True)
    sb = SparseVar2(b); sb.annotate_mutations(small_fasta)
    sa = SparseVar2(a)
    for kind in ("SBS96", "ID83"):
        assert sa.mutation_matrix(kind).equals(sb.mutation_matrix(kind))
```

Note: the two stores may differ in var_key/dense *routing* (write-time cost model shifts with signature bits), but `mutation_matrix` is routing-invariant, so the matrices must match.

- [ ] **Step 3: Run**

Run: `pixi run maturin develop && pixi run pytest tests/test_svar2_mutcat.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_svar2_mutcat.py
git commit -m "test(mutcat): v1 parity + write-time/post-hoc round-trip"
```

### Task 15: Skill, CHANGELOG, docs

**Files:**
- Modify: `skills/genoray-api/SKILL.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/source/svar.md` (if it documents SVAR2 methods)

- [ ] **Step 1: Update `skills/genoray-api/SKILL.md`** — add the three `SparseVar2` methods and the `from_vcf(signatures=...)` kwarg, mirroring how v1 `SparseVar.annotate_mutations`/`mutation_matrix`/`assign_signatures` are documented. State: matrices are `MutationType` + per-sample columns; `kind ∈ {SBS96, DBS78, ID83}`; `count ∈ {allele, sample}`; annotation is post-hoc **or** via `from_vcf(signatures=True)`; DBS78 arises only from adjacent same-haplotype SNV pairs (MNVs are atomized).

- [ ] **Step 2: CHANGELOG entry** under the unreleased/next section:

```markdown
### Added
- `SparseVar2.annotate_mutations`, `mutation_matrix`, and `assign_signatures`
  for COSMIC mutational-signature workflows (SBS96/ID83/DBS78), implemented in
  Rust with a per-record sidecar and streaming count matrix. `from_vcf` gains a
  `signatures=` flag to classify during the write (factored into the cost model).
```

- [ ] **Step 3: Verify docs build (if configured)**

Run: `pixi run -e doc bash -lc 'sphinx-build -b html docs/source docs/build/html -q'` (skip if the doc env is heavy; at minimum confirm no autodoc reference errors for the new methods).

- [ ] **Step 4: Commit**

```bash
git add skills/genoray-api/SKILL.md CHANGELOG.md docs/source/svar.md
git commit -m "docs(mutcat): document SparseVar2 signature API + changelog"
```

### Task 16: Full suite + lint gate

- [ ] **Step 1: Rust suite**

Run: `pixi run -e lint test-rust`
Expected: all pass.

- [ ] **Step 2: Python suite**

Run: `pixi run maturin develop && pixi run test`
Expected: all pass (network-tagged COSMIC tests may be skipped with `-m "not network"`).

- [ ] **Step 3: Lint/format/typecheck**

Run: `pixi run -e lint lint && pixi run typecheck`
Expected: clean. Fix any `ruff`/`pyrefly`/`clippy`/`cargo fmt` findings.

- [ ] **Step 4: Final commit (if lint made changes)**

```bash
git add -A
git commit -m "chore(mutcat): lint + format pass"
```

---

## Self-Review

**Spec coverage:**
- §1 Public API → Tasks 11, 13 (methods), 12 (`from_vcf` flag). ✓
- §2 Sidecar data model + encoding → Tasks 4, 5. ✓
- §3 Classification pass (two entry points) → Tasks 6, 12 (write-time), 13 (post-hoc). ✓
- §4 Cost-model integration → Task 7 (+ wired in 12). ✓
- §5 Streaming count + DBS pairing → Tasks 8, 9, 10. ✓
- §6 Refit reuse → Task 13 (`assign_signatures` delegates to `fit_signatures`). ✓
- Testing strategy (classifier units, parity, cost-model, round-trip) → Tasks 1–3, 7, 14. ✓
- Docs/skill/changelog → Task 15. ✓

**Placeholder scan:** Tasks 9 and 10's fixture-dependent test bodies are sketched, not literal, because the exact fixture helper must be read from `tests/common/mod.rs` first — flagged inline as the implementer's first action for those tasks, with the surrounding assertions concrete. No `TODO`/`TBD` left in shipped code.

**Type consistency:** `choose_representation(..., sidecar_bits: u64)` used consistently (Tasks 7, 12). `SnvCall{pos,sbs,ref_i,alt_i}` and `emit_snv_codes` signatures match between Tasks 8 and 9. `MutcatSub` variants (`VkSnp/VkIndel/DenseSnp/DenseIndel`) consistent across Tasks 4, 5, 6. Sidecar `u8` codes are class-local everywhere; the unified offset is added only in `count.rs` (Tasks 8, 9) and sliced in Python via `code_ranges()` (Task 13).

**Known risk:** v1-vs-SVAR2 exact parity (Task 14) depends on identical atomization of complex variants; mitigation documented in the task (narrow fixture + treat SVAR2's normalized classification as authoritative).

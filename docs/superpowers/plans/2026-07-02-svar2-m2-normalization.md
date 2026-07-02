# SVAR 2.0 M2 — Variant normalization (split + atomize) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept un-normalized VCFs during SVAR2 conversion by decomposing multi-allelic / MNP / complex records into atomized biallelic primitives inline as they stream through the reader.

**Architecture:** A new pure `normalize.rs` module turns one VCF record into a list of atomic `Atom`s (SNP / anchored INS / anchored DEL), mirroring bcftools `_atomize_allele`. The reader is restructured around a position-keyed min-heap reorder buffer so atomization (which spreads atom positions rightward) still emits globally position-sorted `DenseChunk`s, preserving the invariant the interleaving merge depends on. Genotypes are remapped by comparing each haplotype's integer allele index to the atom's source ALT index. The encode seam, executor, and merge are untouched.

**Tech Stack:** Rust 2024, rust-htslib (BCF/VCF), proptest, tempfile. Left-alignment (the only reference-genome-dependent piece) is explicitly deferred to a follow-up (M2b).

## Global Constraints

- Every emitted `Atom` MUST be one of exactly three encoder-valid shapes: SNP (`ref_len==1, alt_len==1, ilen==0`); anchored INS (`ref_len==1, alt_len>=2, ilen>0`, `alt.len()==ilen+1`); anchored DEL (`ref_len>=2, alt_len==1, ilen<0`). The encode seam (`rvk.rs`, `streams.rs`) is NOT modified.
- No change to the Python API signature `run_conversion_pipeline`, the on-disk layout, or `meta.json`.
- `*` (spanning-deletion) and `.` (no-ALT) alleles are skipped (no atom). Symbolic/breakend alleles (`<...>`, containing `[` or `]`) are rejected.
- No left-alignment. Consequently every atom's `pos >= its source record's start pos`; the reorder buffer relies on this.
- Run the Rust test suite with: `pixi run -e lint cargo test --no-default-features`. (Default features enable pyo3 `extension-module`, which doesn't link libpython for test binaries; the `lint` env provides rust + libclang + the htslib build deps.)
- Match existing code style; keep files focused. `normalize.rs` owns all atomization logic — no SNP/indel branching leaks into the reader beyond genotype presence.

---

### Task 1: `normalize.rs` — pure atomization core

**Files:**
- Create: `src/normalize.rs`
- Modify: `src/lib.rs` (add `pub mod normalize;`)
- Test: in-source `#[cfg(test)]` in `src/normalize.rs`

**Interfaces:**
- Consumes: nothing (pure module).
- Produces:
  - `genoray_core::normalize::Atom { pub pos: u32, pub ilen: i32, pub alt: Vec<u8>, pub source_alt_index: u16 }`
  - `genoray_core::normalize::NormalizeError` (thiserror enum; variant `SymbolicAllele { pos: u32, alt: String }`)
  - `pub fn atomize_record(pos: u32, ref_allele: &[u8], alts: &[&[u8]], out: &mut Vec<Atom>) -> Result<(), NormalizeError>`

- [ ] **Step 1: Add the module declaration**

In `src/lib.rs`, add alongside the other `pub mod` lines (keep alphabetical grouping near `pub mod nrvk;`):

```rust
pub mod normalize;
```

- [ ] **Step 2: Write the failing tests**

Create `src/normalize.rs` with ONLY the test module first (the types/fn don't exist yet, so it won't compile — that's the "fail"):

```rust
//! Variant normalization: biallelic split + atomization (no left-alignment; roadmap M2).
//! Pure functions over (pos, REF, ALTs) → atomic biallelic primitives. This is the only
//! module that knows atomization rules; the encode seam downstream stays sealed.

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn atoms(pos: u32, r: &[u8], alts: &[&[u8]]) -> Vec<Atom> {
        let mut out = Vec::new();
        atomize_record(pos, r, alts, &mut out).unwrap();
        out
    }

    fn atom(pos: u32, ilen: i32, alt: &[u8], src: u16) -> Atom {
        Atom { pos, ilen, alt: alt.to_vec(), source_alt_index: src }
    }

    #[test]
    fn snp_passthrough() {
        assert_eq!(atoms(100, b"A", &[b"C"]), vec![atom(100, 0, b"C", 1)]);
    }

    #[test]
    fn biallelic_split_tags_source_index() {
        // A>C,G → two independent SNPs carrying source indices 1 and 2.
        assert_eq!(
            atoms(100, b"A", &[b"C", b"G"]),
            vec![atom(100, 0, b"C", 1), atom(100, 0, b"G", 2)]
        );
    }

    #[test]
    fn mnp_becomes_per_position_snps() {
        // AC>GT → A>G@100, C>T@101 (both differ).
        assert_eq!(
            atoms(100, b"AC", &[b"GT"]),
            vec![atom(100, 0, b"G", 1), atom(101, 0, b"T", 1)]
        );
    }

    #[test]
    fn mnp_skips_matching_bases() {
        // ACA>ATA → only the middle base changes.
        assert_eq!(atoms(100, b"ACA", &[b"ATA"]), vec![atom(101, 0, b"T", 1)]);
    }

    #[test]
    fn simple_insertion_anchored() {
        // A>AT → anchored INS, ilen=+1, full alt stored.
        assert_eq!(atoms(100, b"A", &[b"AT"]), vec![atom(100, 1, b"AT", 1)]);
    }

    #[test]
    fn simple_deletion_anchored() {
        // ATG>A → pure DEL, ilen=-2, alt = anchor base.
        assert_eq!(atoms(100, b"ATG", &[b"A"]), vec![atom(100, -2, b"A", 1)]);
    }

    #[test]
    fn shared_suffix_is_trimmed() {
        // ATG>CG shares suffix G → AT>C: anchor substitutes (A→C), so split into
        // SNV(A>C)@100 plus a clean DEL(anchor=ref base A, delete T)@100.
        assert_eq!(
            atoms(100, b"ATG", &[b"CG"]),
            vec![atom(100, 0, b"C", 1), atom(100, -1, b"A", 1)]
        );
    }

    #[test]
    fn complex_snv_plus_insertion() {
        // GCG>GTGA (the bcftools abuf.c example): interior SNV C>T@101, then an
        // anchored INS G>GA@102.
        assert_eq!(
            atoms(100, b"GCG", &[b"GTGA"]),
            vec![atom(101, 0, b"T", 1), atom(102, 1, b"GA", 1)]
        );
    }

    #[test]
    fn star_and_missing_alleles_skipped() {
        assert_eq!(atoms(100, b"A", &[b"*"]), vec![]);
        assert_eq!(atoms(100, b"A", &[b"."]), vec![]);
        // still processes the real ALT alongside a skipped one
        assert_eq!(atoms(100, b"A", &[b"*", b"C"]), vec![atom(100, 0, b"C", 2)]);
    }

    #[test]
    fn symbolic_allele_errors() {
        let mut out = Vec::new();
        assert!(matches!(
            atomize_record(100, b"A", &[b"<DEL>"], &mut out),
            Err(NormalizeError::SymbolicAllele { .. })
        ));
        assert!(matches!(
            atomize_record(100, b"A", &[b"A[chr2:321[".as_slice()], &mut out),
            Err(NormalizeError::SymbolicAllele { .. })
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        // Every atom is encoder-valid and starts within the record's reference span.
        #[test]
        fn atoms_are_encoder_valid(
            pos in 0u32..1_000_000,
            r in proptest::collection::vec(prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T')], 1..12),
            a in proptest::collection::vec(prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T')], 1..12),
        ) {
            let mut out = Vec::new();
            let ref_bytes: Vec<u8> = r;
            let alt_bytes: Vec<u8> = a;
            atomize_record(pos, &ref_bytes, &[alt_bytes.as_slice()], &mut out).unwrap();
            for at in &out {
                let alt_len = at.alt.len() as i32;
                let valid = (at.ilen == 0 && alt_len == 1)
                    || (at.ilen > 0 && alt_len == at.ilen + 1)  // INS: ref_len 1
                    || (at.ilen < 0 && alt_len == 1);           // DEL: alt_len 1
                prop_assert!(valid, "atom not encoder-valid: {:?}", at);
                prop_assert!(at.pos >= pos, "atom before record start");
                prop_assert!(at.pos < pos + ref_bytes.len() as u32, "atom past reference span");
            }
        }
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features normalize`
Expected: FAIL — compile error, `cannot find type Atom` / `atomize_record` not found.

- [ ] **Step 4: Write the implementation**

Prepend to `src/normalize.rs` (above the `#[cfg(test)]` module, below the top doc comment):

```rust
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum NormalizeError {
    #[error("symbolic/breakend ALT '{alt}' at pos {pos} is out of scope (short-read only)")]
    SymbolicAllele { pos: u32, alt: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atom {
    /// 0-based start position of the atom.
    pub pos: u32,
    /// len(alt) - len(ref): 0 = SNP, > 0 = INS, < 0 = DEL.
    pub ilen: i32,
    /// SNP: the single ALT base. INS: anchor + inserted bases. DEL: the anchor base.
    pub alt: Vec<u8>,
    /// 1-based index into the record's original ALTs, for genotype remapping.
    pub source_alt_index: u16,
}

/// Decompose one VCF record into atomic biallelic primitives, appended to `out`.
/// `alts` are the ALT alleles only (REF excluded). `*` / `.` alleles are skipped;
/// symbolic/breakend alleles return an error.
pub fn atomize_record(
    pos: u32,
    ref_allele: &[u8],
    alts: &[&[u8]],
    out: &mut Vec<Atom>,
) -> Result<(), NormalizeError> {
    for (j, &alt) in alts.iter().enumerate() {
        let src = (j + 1) as u16;
        if alt == b"*" || alt == b"." {
            continue;
        }
        if is_symbolic(alt) {
            return Err(NormalizeError::SymbolicAllele {
                pos,
                alt: String::from_utf8_lossy(alt).into_owned(),
            });
        }
        atomize_biallelic(pos, ref_allele, alt, src, out);
    }
    Ok(())
}

#[inline]
fn is_symbolic(alt: &[u8]) -> bool {
    alt.first() == Some(&b'<') || alt.contains(&b'[') || alt.contains(&b']')
}

/// Decompose a single REF/ALT pair. Mirrors bcftools `_atomize_allele`: trim the
/// shared suffix, emit a SNV per interior mismatch, and attach any length change as a
/// single left-anchored indel at the last aligned index. The one deviation is the
/// substituted-deletion-anchor case (see below), forced by the pure-DEL encoding.
fn atomize_biallelic(pos: u32, ref_allele: &[u8], alt: &[u8], src: u16, out: &mut Vec<Atom>) {
    // 1. Trim shared suffix, keeping >= 1 base on each side.
    let mut rlen = ref_allele.len();
    let mut alen = alt.len();
    while rlen > 1 && alen > 1 && ref_allele[rlen - 1] == alt[alen - 1] {
        rlen -= 1;
        alen -= 1;
    }
    let r = &ref_allele[..rlen];
    let a = &alt[..alen];

    let n = rlen.min(alen);
    let k = n - 1; // last aligned index; the indel (if any) anchors here

    // 2. Interior aligned positions [0, k): one SNV per mismatch.
    for i in 0..k {
        if r[i] != a[i] {
            out.push(Atom { pos: pos + i as u32, ilen: 0, alt: vec![a[i]], source_alt_index: src });
        }
    }

    // 3. Boundary at k.
    let apos = pos + k as u32;
    if rlen == alen {
        // Pure substitution tail.
        if r[k] != a[k] {
            out.push(Atom { pos: apos, ilen: 0, alt: vec![a[k]], source_alt_index: src });
        }
    } else if alen > rlen {
        // Insertion anchored at k. ref[k..] is a single base; alt[k..] carries the
        // (possibly substituted) anchor + inserted bases — the full alt is stored, so a
        // substituted anchor is faithfully represented.
        let ins_alt = a[k..alen].to_vec();
        let ilen = ins_alt.len() as i32 - 1; // ref_len == 1
        out.push(Atom { pos: apos, ilen, alt: ins_alt, source_alt_index: src });
    } else {
        // Deletion anchored at k. alt[k..] is a single base.
        let del_ref_len = (rlen - k) as i32; // >= 2
        let ilen = 1 - del_ref_len; // alt_len(1) - ref_len
        if r[k] == a[k] {
            // Clean anchor → pure DEL (alt = the anchor base).
            out.push(Atom { pos: apos, ilen, alt: vec![a[k]], source_alt_index: src });
        } else {
            // Substituted anchor: the pure-DEL encoding reconstructs the anchor from
            // the reference, so it cannot carry a substitution. Split into a SNV plus a
            // clean DEL whose anchor is the unchanged reference base.
            out.push(Atom { pos: apos, ilen: 0, alt: vec![a[k]], source_alt_index: src });
            out.push(Atom { pos: apos, ilen, alt: vec![r[k]], source_alt_index: src });
        }
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features normalize`
Expected: PASS (all unit tests + the proptest).

- [ ] **Step 6: Lint**

Run: `pixi run -e lint cargo clippy --no-default-features -- -D warnings`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/normalize.rs src/lib.rs
git commit -m "feat(svar-2): normalize.rs — biallelic split + atomization core"
```

---

### Task 2: Extract shared test harness; drop obsolete negative e2e tests

**Files:**
- Create: `tests/common/mod.rs`
- Modify: `tests/test_e2e.rs`
- Test: `tests/test_e2e.rs` (still passes with existing positive coverage)

**Interfaces:**
- Consumes: `genoray_core::process_chromosome`, `genoray_core::vcf_reader::VcfChunkReader`.
- Produces (in `tests/common/mod.rs`, `pub`):
  - `struct SynthRecord<'a> { pub pos: i64, pub ref_allele: &'a [u8], pub alts: Vec<&'a [u8]>, pub gt: Vec<i32> }`
  - `fn build_bcf_with_index(bcf_path: &Path, chrom: &str, chrom_len: u64, samples: &[&str], records: &[SynthRecord])`
  - `fn read_u32_bin(path: &Path) -> Vec<u32>`
  - `fn read_offsets_npy(path: &Path) -> Vec<u64>`

- [ ] **Step 1: Create the shared harness module**

Create `tests/common/mod.rs` by moving the harness out of `tests/test_e2e.rs` and making the items `pub`:

```rust
// Shared synthetic-BCF harness for the e2e / atomization integration tests.
use rust_htslib::bcf::record::GenotypeAllele;
use rust_htslib::bcf::{Format, Header, Writer};
use std::path::Path;

// One synthetic VCF record: position, ref allele, list of alt alleles, and a flat
// genotype vector laid out as [s0_p0, s0_p1, s1_p0, s1_p1, ...] holding allele indices
// (0 = ref, 1 = first alt, ...). Use a negative value for a missing allele.
pub struct SynthRecord<'a> {
    pub pos: i64,
    pub ref_allele: &'a [u8],
    pub alts: Vec<&'a [u8]>,
    pub gt: Vec<i32>,
}

pub fn build_bcf_with_index(
    bcf_path: &Path,
    chrom: &str,
    chrom_len: u64,
    samples: &[&str],
    records: &[SynthRecord],
) {
    let mut header = Header::new();
    let contig = format!("##contig=<ID={},length={}>", chrom, chrom_len);
    header.push_record(contig.as_bytes());
    header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
    for s in samples {
        header.push_sample(s.as_bytes());
    }

    {
        let mut writer =
            Writer::from_path(bcf_path, &header, false, Format::Bcf).expect("open BCF writer");

        for rec in records {
            let mut record = writer.empty_record();
            record.set_rid(Some(0));
            record.set_pos(rec.pos);

            let mut alleles: Vec<&[u8]> = Vec::with_capacity(1 + rec.alts.len());
            alleles.push(rec.ref_allele);
            for a in &rec.alts {
                alleles.push(a);
            }
            record.set_alleles(&alleles).expect("set alleles");

            let gt_alleles: Vec<GenotypeAllele> =
                rec.gt.iter().map(|&i| GenotypeAllele::Phased(i)).collect();
            record.push_genotypes(&gt_alleles).expect("push genotypes");

            writer.write(&record).expect("write record");
        }
    }

    rust_htslib::bcf::index::build(bcf_path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
        .expect("build BCF index");
}

pub fn read_u32_bin(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read u32 bin");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn read_offsets_npy(path: &Path) -> Vec<u64> {
    let arr: ndarray::Array1<u64> = ndarray_npy::read_npy(path).expect("read offsets npy");
    arr.to_vec()
}
```

- [ ] **Step 2: Rewrite `tests/test_e2e.rs` to use the harness and drop the obsolete negative tests**

Replace the top of `tests/test_e2e.rs` (the imports + the moved harness `SynthRecord` / `build_bcf_with_index` / `read_u32_bin` / `read_offsets_npy` definitions) with a `mod common;` include, and **delete** the two `#[should_panic]` tests `test_reader_panics_on_multi_allelic` and `test_reader_panics_on_complex_variant` (that behavior is being removed). Keep `test_e2e_normalized_bcf_pipeline`, `test_e2e_mutation_conservation`, `test_reader_accepts_pure_del`, and `test_missing_chrom_returns_err`, changing their references to the harness items to `common::...`.

Concretely, the new header of `tests/test_e2e.rs`:

```rust
// End-to-end pipeline test: builds tiny synthetic BCFs and validates the final
// sample-major sparse outputs against hand-computed ground truth.
mod common;

use common::{SynthRecord, build_bcf_with_index, read_offsets_npy, read_u32_bin};

use genoray_core::process_chromosome;
use genoray_core::rvk::{decode_alt_inline, unpack_snp_keys};
use genoray_core::vcf_reader::VcfChunkReader;

use std::path::Path;
use tempfile::tempdir;
```

Leave the four retained test functions and the `DecodedKey` / `decode_key` helpers as-is (they already refer to `SynthRecord` etc., now imported from `common`). Update the doc comment on `test_reader_accepts_pure_del` to drop the "assert-panic contract" framing.

- [ ] **Step 3: Run the e2e tests to verify they still pass**

Run: `pixi run -e lint cargo test --no-default-features --test test_e2e`
Expected: PASS — 4 tests (`test_e2e_normalized_bcf_pipeline`, `test_e2e_mutation_conservation`, `test_reader_accepts_pure_del`, `test_missing_chrom_returns_err`).

- [ ] **Step 4: Commit**

```bash
git add tests/common/mod.rs tests/test_e2e.rs
git commit -m "test(svar-2): extract shared BCF harness; drop obsolete reader-panic tests"
```

---

### Task 3: Reader restructure — reorder buffer, genotype remapping, atomized chunks

**Files:**
- Modify: `src/vcf_reader.rs` (full restructure of `VcfChunkReader`)
- Create: `tests/test_atomize_e2e.rs`
- Test: `tests/test_atomize_e2e.rs`

**Interfaces:**
- Consumes: `genoray_core::normalize::{Atom, atomize_record}` (Task 1); `common::*` harness (Task 2); `crate::types::{BitGrid3, DenseChunk}`.
- Produces: unchanged public surface — `VcfChunkReader::new(vcf_path: &str, chrom: &str, samples: &[&str], htslib_threads: usize, ploidy: usize) -> Self` and `read_next_chunk(&mut self, chunk_size: usize, chunk_id: usize) -> Option<DenseChunk>`. `DenseChunk` now holds one row per **atom** (globally position-sorted across chunks) instead of one row per input record.

- [ ] **Step 1: Write the failing integration tests**

Create `tests/test_atomize_e2e.rs`:

```rust
// End-to-end atomization/normalization: feed un-normalized records and assert the
// reader emits correctly split, atomized, and globally position-sorted DenseChunks.
mod common;

use common::{SynthRecord, build_bcf_with_index};
use genoray_core::vcf_reader::VcfChunkReader;
use std::path::Path;
use tempfile::tempdir;

// Collect every atom the reader emits across all chunks, as (pos, ilen) plus the
// per-column presence bits, in emission order.
fn drain_reader(
    bcf_path: &Path,
    chrom: &str,
    samples: &[&str],
    ploidy: usize,
    chunk_size: usize,
) -> Vec<(u32, i32, Vec<bool>)> {
    let mut reader = VcfChunkReader::new(bcf_path.to_str().unwrap(), chrom, samples, 1, ploidy);
    let columns = samples.len() * ploidy;
    let mut out = Vec::new();
    let mut chunk_id = 0;
    while let Some(chunk) = reader.read_next_chunk(chunk_size, chunk_id) {
        let v = chunk.pos.len();
        for i in 0..v {
            let mut presence = Vec::with_capacity(columns);
            for col in 0..columns {
                presence.push(chunk.genos.get_bit(i * columns + col));
            }
            out.push((chunk.pos[i], chunk.ilens[i], presence));
        }
        chunk_id += 1;
    }
    out
}

#[test]
fn multiallelic_site_splits_and_remaps_genotypes() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("multi.bcf");
    let samples = vec!["S0"];
    // One diploid sample, genotype 1|2 at a 2-ALT site A>C,G.
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"A",
        alts: vec![&b"C"[..], &b"G"[..]],
        gt: vec![1, 2],
    }];
    build_bcf_with_index(&bcf, "chr1", 10_000, &samples, &records);

    let atoms = drain_reader(&bcf, "chr1", &samples, 2, 100);
    // Two SNP atoms at pos 100: ALT C carried on hap0 only, ALT G on hap1 only.
    assert_eq!(atoms.len(), 2);
    assert_eq!((atoms[0].0, atoms[0].1), (100, 0));
    assert_eq!((atoms[1].0, atoms[1].1), (100, 0));
    assert_eq!(atoms[0].2, vec![true, false]); // source ALT 1 (C) → hap0
    assert_eq!(atoms[1].2, vec![false, true]); // source ALT 2 (G) → hap1
}

#[test]
fn mnp_atomizes_to_snps_shared_presence() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("mnp.bcf");
    let samples = vec!["S0"];
    // AC>GT MNP, sample homozygous for the ALT on both haps.
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"AC",
        alts: vec![&b"GT"[..]],
        gt: vec![1, 1],
    }];
    build_bcf_with_index(&bcf, "chr1", 10_000, &samples, &records);

    let atoms = drain_reader(&bcf, "chr1", &samples, 2, 100);
    // Two SNP atoms (A>G@100, C>T@101), both carried on both haps.
    assert_eq!(atoms.len(), 2);
    assert_eq!((atoms[0].0, atoms[0].1), (100, 0));
    assert_eq!((atoms[1].0, atoms[1].1), (101, 0));
    assert_eq!(atoms[0].2, vec![true, true]);
    assert_eq!(atoms[1].2, vec![true, true]);
}

#[test]
fn atoms_are_globally_position_sorted_across_records() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("sorted.bcf");
    let samples = vec!["S0"];
    // An MNP at 100 spans to 104; a SNP record starts at 102 — its atom would land
    // between the MNP's atoms unless the reader reorders. Also force small chunks so
    // the ordering must hold ACROSS chunk boundaries.
    let records = vec![
        SynthRecord { pos: 100, ref_allele: b"ACGTA", alts: vec![&b"GCGTG"[..]], gt: vec![1, 1] },
        SynthRecord { pos: 102, ref_allele: b"A", alts: vec![&b"T"[..]], gt: vec![1, 0] },
    ];
    build_bcf_with_index(&bcf, "chr1", 10_000, &samples, &records);

    // chunk_size = 1 → every atom lands in its own chunk; emission order must still
    // be globally sorted.
    let atoms = drain_reader(&bcf, "chr1", &samples, 2, 1);
    let positions: Vec<u32> = atoms.iter().map(|a| a.0).collect();
    // MNP → A>G@100, A>G@104; SNP → T@102. Sorted: 100, 102, 104.
    assert_eq!(positions, vec![100, 102, 104]);
    let mut sorted = positions.clone();
    sorted.sort();
    assert_eq!(positions, sorted, "emitted positions must be globally sorted");
}

#[test]
fn complex_deletion_with_substituted_anchor() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("complex.bcf");
    let samples = vec!["S0"];
    // ATG>CG (previously rejected as "complex"): → SNV(A>C)@100 + DEL(ilen=-1)@100.
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"ATG",
        alts: vec![&b"CG"[..]],
        gt: vec![1, 1],
    }];
    build_bcf_with_index(&bcf, "chr1", 10_000, &samples, &records);

    let atoms = drain_reader(&bcf, "chr1", &samples, 2, 100);
    assert_eq!(atoms.len(), 2);
    assert_eq!((atoms[0].0, atoms[0].1), (100, 0));  // SNV A>C
    assert_eq!((atoms[1].0, atoms[1].1), (100, -1)); // DEL
    assert_eq!(atoms[0].2, vec![true, true]);
    assert_eq!(atoms[1].2, vec![true, true]);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e lint cargo test --no-default-features --test test_atomize_e2e`
Expected: FAIL — the current reader panics on the multi-allelic / complex inputs (`must be normalized` / `must be atomized`) and does not atomize MNPs.

- [ ] **Step 3: Restructure `src/vcf_reader.rs`**

Replace the entire contents of `src/vcf_reader.rs` with:

```rust
use crate::normalize::atomize_record;
use crate::types::{BitGrid3, DenseChunk};
use rust_htslib::bcf::record::Record;
use rust_htslib::bcf::{IndexedReader, Read};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::rc::Rc;

// A decomposed atom awaiting emission. Carries a shared handle to its source record's
// per-column allele indices so genotype presence is computed at chunk-build time.
struct PendingAtom {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    source_alt_index: u16,
    gt: Rc<Vec<i32>>, // len = num_samples * ploidy; allele index per column (-1 = missing)
    seq: u64,         // stable tiebreak for equal positions
}

impl PartialEq for PendingAtom {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.seq == other.seq
    }
}
impl Eq for PendingAtom {}
impl PartialOrd for PendingAtom {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PendingAtom {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pos.cmp(&other.pos).then(self.seq.cmp(&other.seq))
    }
}

pub struct VcfChunkReader {
    inner_reader: IndexedReader,
    num_samples: usize,
    ploidy: usize,
    sample_indices: Vec<usize>,

    // Reorder state, persisted across read_next_chunk calls.
    record: Record,
    heap: BinaryHeap<Reverse<PendingAtom>>,
    frontier: u32, // start pos of the most recently read record: all future atoms have pos >= this
    eof: bool,
    next_seq: u64,
}

impl VcfChunkReader {
    // opens the file, applies the sample filter at the C-level, and jumps to the chromosome.
    pub fn new(
        vcf_path: &str,
        chrom: &str,
        samples: &[&str],
        htslib_threads: usize,
        ploidy: usize,
    ) -> Self {
        let mut reader = IndexedReader::from_path(vcf_path)
            .expect("Failed to open VCF/BCF index. Is there a .tbi or .csi file?");

        reader
            .set_threads(htslib_threads)
            .expect("Failed to allocate HTSlib background threads");

        let header = reader.header().clone();

        let rid = header
            .name2rid(chrom.as_bytes())
            .expect("Chromosome not found in VCF header");

        reader
            .fetch(rid, 0, None)
            .expect("Failed to fetch chromosome region");

        let sample_indices: Vec<usize> = samples
            .iter()
            .map(|name| {
                header
                    .sample_id(name.as_bytes())
                    .unwrap_or_else(|| panic!("Sample {} not found in VCF", name))
            })
            .collect();

        let record = reader.empty_record();

        Self {
            inner_reader: reader,
            num_samples: samples.len(),
            ploidy,
            sample_indices,
            record,
            heap: BinaryHeap::new(),
            frontier: 0,
            eof: false,
            next_seq: 0,
        }
    }

    // Decompose `self.record` into atoms and push them onto the reorder heap, sharing
    // one decoded genotype vector across all atoms of the record.
    fn decompose_current_record(&mut self) {
        let pos = self.record.pos() as u32;

        // Own the alleles so the record borrow is released before we mutate self.
        let ref_allele: Vec<u8>;
        let alts_owned: Vec<Vec<u8>>;
        {
            let alleles = self.record.alleles();
            ref_allele = alleles[0].to_vec();
            alts_owned = alleles[1..].iter().map(|a| a.to_vec()).collect();
        }

        // Decode per-column allele indices (-1 = missing).
        let columns = self.num_samples * self.ploidy;
        let mut gt = vec![-1i32; columns];
        {
            let genotypes = self.record.genotypes().expect("Failed to read genotypes");
            for (s_idx, &vcf_idx) in self.sample_indices.iter().enumerate() {
                let sample_gt = genotypes.get(vcf_idx);
                for p in 0..self.ploidy {
                    let idx = if p < sample_gt.len() {
                        sample_gt[p].index().map(|v| v as i32).unwrap_or(-1)
                    } else {
                        -1
                    };
                    gt[s_idx * self.ploidy + p] = idx;
                }
            }
        }
        let gt = Rc::new(gt);

        let alt_refs: Vec<&[u8]> = alts_owned.iter().map(|a| a.as_slice()).collect();
        let mut atoms = Vec::new();
        atomize_record(pos, &ref_allele, &alt_refs, &mut atoms)
            .expect("symbolic/breakend ALT is out of scope for SVAR2 (short-read only)");

        for atom in atoms {
            let seq = self.next_seq;
            self.next_seq += 1;
            self.heap.push(Reverse(PendingAtom {
                pos: atom.pos,
                ilen: atom.ilen,
                alt: atom.alt,
                source_alt_index: atom.source_alt_index,
                gt: Rc::clone(&gt),
                seq,
            }));
        }
    }

    // Yield the next atom in global position order, reading and decomposing more
    // records as needed. An atom is safe to emit once its position is strictly below
    // the read frontier (no future atom can precede the frontier, since there is no
    // left-alignment) or once the input is exhausted.
    fn next_atom(&mut self) -> Option<PendingAtom> {
        loop {
            if let Some(Reverse(top)) = self.heap.peek() {
                if self.eof || top.pos < self.frontier {
                    return Some(self.heap.pop().unwrap().0);
                }
            } else if self.eof {
                return None;
            }

            match self.inner_reader.read(&mut self.record) {
                Some(Ok(())) => {
                    self.frontier = self.record.pos() as u32;
                    self.decompose_current_record();
                }
                Some(Err(e)) => panic!("VCF Read Error: {}", e),
                None => self.eof = true,
            }
        }
    }

    // Pull up to `chunk_size` atoms (already globally position-sorted) and pack them
    // into a variant-major DenseChunk. Returns None once no atoms remain.
    pub fn read_next_chunk(&mut self, chunk_size: usize, chunk_id: usize) -> Option<DenseChunk> {
        let mut atoms: Vec<PendingAtom> = Vec::with_capacity(chunk_size);
        while atoms.len() < chunk_size {
            match self.next_atom() {
                Some(a) => atoms.push(a),
                None => break,
            }
        }
        if atoms.is_empty() {
            return None;
        }

        let v = atoms.len();
        let columns = self.num_samples * self.ploidy;

        let mut pos = Vec::with_capacity(v);
        let mut ilens = Vec::with_capacity(v);
        let mut alt = Vec::with_capacity(v * 2);
        let mut alt_offsets = Vec::with_capacity(v + 1);
        alt_offsets.push(0u32);
        let mut genos = BitGrid3::zeros(v, self.num_samples, self.ploidy);

        let mut off = 0u32;
        for (vi, a) in atoms.iter().enumerate() {
            pos.push(a.pos);
            ilens.push(a.ilen);
            alt.extend_from_slice(&a.alt);
            off += a.alt.len() as u32;
            alt_offsets.push(off);

            let src = a.source_alt_index as i32;
            let base = vi * columns;
            for col in 0..columns {
                genos.or_bit(base + col, a.gt[col] == src);
            }
        }

        Some(DenseChunk {
            chunk_id,
            pos,
            ilens,
            alt,
            alt_offsets,
            genos,
        })
    }
}
```

- [ ] **Step 4: Run the new integration tests to verify they pass**

Run: `pixi run -e lint cargo test --no-default-features --test test_atomize_e2e`
Expected: PASS — all 4 tests.

- [ ] **Step 5: Run the full Rust test suite (regression check)**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: PASS — including the retained `test_e2e` tests (already-atomized inputs round-trip unchanged) and all in-source unit/proptests.

- [ ] **Step 6: Lint**

Run: `pixi run -e lint cargo clippy --no-default-features -- -D warnings`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/vcf_reader.rs tests/test_atomize_e2e.rs
git commit -m "feat(svar-2): atomize + reorder in the reader; accept un-normalized VCFs"
```

---

### Task 4: Reconcile roadmap + data-model docs

**Files:**
- Modify: `docs/roadmap/svar-2.md`
- Modify: `docs/roadmap/data-model.md`

**Interfaces:**
- Consumes: nothing (docs).
- Produces: nothing (docs).

- [ ] **Step 1: Update the M2 milestone in `docs/roadmap/svar-2.md`**

Change the M2 checkbox from `[ ]` to `[~]` and rewrite its body to reflect the shipped scope (split + atomize) and the deferral. Then add a new `M2b` entry directly after it. Replace the existing M2 bullet with:

```markdown
- [~] **M2. Variant normalization during conversion.** Atomization and biallelic
  splitting (split multi-allelic sites) applied inline as variants stream through. See
  [`data-model.md`](data-model.md#variant-normalization). *Shipped:* the reader accepts
  un-normalized VCFs — a pure `normalize.rs` decomposes each record into atomic
  biallelic primitives (SNP / anchored INS / anchored DEL) mirroring bcftools
  `_atomize_allele`, and a position-keyed reorder buffer preserves the merge's
  sorted-position invariant across chunk boundaries. Genotypes are remapped by comparing
  each haplotype's integer allele index to the atom's source ALT index. The former
  "input must be normalized" asserts are gone; symbolic/breakend ALTs are rejected and
  `*`/`.` alleles are skipped.
- [ ] **M2b. Left-alignment during conversion.** Shift indels to their leftmost
  equivalent position. Deferred from M2 because it is the only normalization step that
  needs a reference genome (FASTA/faidx) and a new required conversion argument, and it
  widens the reorder-buffer bound (leftward shifts). See
  [`data-model.md`](data-model.md#variant-normalization).
```

- [ ] **Step 2: Update the normalization section in `docs/roadmap/data-model.md`**

Replace the `## Variant normalization` section body so left-alignment is flagged as a separate milestone:

```markdown
## Variant normalization

Conversion normalizes variants inline as they stream through, so the on-disk model is
always normalized:

- **Atomization (M2)** — break complex/MNV records into atomic primitives (SNP /
  anchored INS / anchored DEL), mirroring bcftools `_atomize_allele`.
- **Biallelic split (M2)** — split multi-allelic sites into separate biallelic records,
  remapping genotypes by original ALT index.
- **Left-alignment (M2b, deferred)** — shift indels to their leftmost equivalent
  position. Deferred because it is the only step requiring a reference genome.

This keeps `ILEN`/ALT semantics simple and makes the inline encoding well-defined.
Because atomization spreads atom positions rightward (and, once M2b lands,
left-alignment shifts them leftward), the reader emits atoms through a position-keyed
reorder buffer so each per-`(sample, ploid)` stream stays position-sorted for the
interleaving merge.
```

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap/svar-2.md docs/roadmap/data-model.md
git commit -m "docs(svar-2): reconcile M2 (split+atomize shipped; left-align → M2b)"
```

---

## Self-Review

**Spec coverage:**
- Biallelic split → Task 1 (`atomize_record` loops ALTs) + Task 3 test `multiallelic_site_splits_and_remaps_genotypes`. ✓
- Atomization (MNP + complex) → Task 1 impl + tests; Task 3 e2e. ✓
- Left-alignment deferred → Task 4 (M2b). ✓
- `*`/`.` skipped, symbolic rejected → Task 1 (`atomize_record`, `is_symbolic`) + tests. ✓
- Genotype remapping via integer allele index → Task 3 `decompose_current_record` + tests. ✓
- Reorder buffer preserves sorted invariant → Task 3 `next_atom` + `atoms_are_globally_position_sorted_across_records`. ✓
- Encode seam / executor / merge / Python API unchanged → no tasks touch them; enforced by Global Constraints and the passing `test_e2e` regression. ✓
- Error handling (symbolic → typed error at boundary) → Task 1 `NormalizeError`; the reader `.expect()`s it, surfacing via the existing thread-panic→`ConversionError::WorkerPanicked` path (consistent with `error.rs`'s "worker hot loops panic" note and `test_missing_chrom_returns_err`). ✓
- e2e vs bcftools oracle → covered structurally by hand-built fixtures in Task 3 (`bcftools`-on-PATH oracle noted as optional in the spec; hand fixtures chosen to avoid a hard external-tool dependency in CI).

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `Atom { pos: u32, ilen: i32, alt: Vec<u8>, source_alt_index: u16 }` and `atomize_record(pos, ref_allele, alts, out) -> Result<(), NormalizeError>` are identical in Task 1 definition and Task 3 use. `VcfChunkReader::{new, read_next_chunk}` signatures match the pre-existing public API the tests call. `common::{SynthRecord, build_bcf_with_index, read_u32_bin, read_offsets_npy}` defined in Task 2 and consumed in Tasks 2–3. ✓

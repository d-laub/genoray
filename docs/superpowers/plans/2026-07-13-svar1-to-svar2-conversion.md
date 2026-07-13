# `SparseVar2.from_svar1` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SparseVar2.from_svar1(out, source, reference=...)` — a native SVAR1→SVAR2 migration that skips htslib and re-reads no VCF.

**Architecture:** Reuse the `RecordSource` seam that `from_pgen` (PR #102) established. `from_svar1` adds one new `SourceSpec::Svar1` variant to `process_chromosome`, one new `Svar1RecordSource` that mmaps SVAR1's sparse arrays and yields variant-major `RawRecord`s (transposing SVAR1's sample-major layout), and one new `run_svar1_conversion_pipeline` pyfunction. Everything downstream (`ChunkAssembler` → `dense2sparse_vk` cost-routing → codec → merge → writer → `field_finalize`) is reused unchanged, so output is byte-identical to `from_vcf` for genotype streams under matching normalization. A thin Python classmethod reads SVAR1's `metadata.json` (samples/ploidy/contigs/fields) and `index.arrow` (via polars), then marshals per-contig `POS/REF/ALT` + contig ranges + a field manifest to Rust.

**Tech Stack:** Rust (pyo3, memmap2, bytemuck, rayon — all existing deps; **no new crates**), Python (polars — existing dep), pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-svar1-to-svar2-conversion-design.md`

## Global Constraints

- **No new Rust dependencies.** `index.arrow` is read in Python via polars; SVAR1 `.npy` arrays are mmap'd via the existing `memmap2` crate. Do not add `arrow-rs`.
- **SVAR1 `.npy` files are header-less raw binary** (written via `np.memmap`, not `np.save`) — mmap and cast via `bytemuck`; there is no `\x93NUMPY` header to skip.
- **SVAR1 `index.arrow` `POS` column is 1-based.** `RawRecord.pos` is 0-based ⇒ `pos = POS - 1`.
- **Coordinate/missing conventions unchanged:** 0-based half-open; SVAR1 has no missing genotypes (sparse stores non-ref only), so reconstructed `gt` is `0` (REF) or `1` (ALT1) — biallelic only.
- **Public-API rule (CLAUDE.md):** any change reachable from `import genoray` without underscores MUST update `skills/genoray-api/SKILL.md` in the same PR. `SparseVar2.from_svar1` is public.
- **Conventional Commits** for every commit (`feat:`, `test:`, `docs:`).
- **Rust build/test gotchas (from prior sessions):** export `CARGO_TARGET_DIR=/tmp/genoray-cargo-$$` before any cargo/commit to avoid an NFS linker bus error; run Rust tests via the `test-rust` task (`cargo test --no-default-features --features conversion`) or the equivalent — the default feature set fails to link the pyo3 test binary (`undefined symbol: _Py_Dealloc`).
- **Rebuild the extension after Rust edits:** `pixi run maturin develop` (the editable install does NOT auto-recompile on import) before running any Python test that exercises new Rust.

---

## File Structure

- **Create** `src/svar1_reader.rs` — `Svar1RecordSource` + the pure `build_variant_major` transpose + raw-`.npy` mmap helpers. One responsibility: turn SVAR1's on-disk sparse arrays into `RawRecord`s.
- **Modify** `src/orchestrator.rs` — add `SourceSpec::Svar1 {…}` variant + the reader-thread `match` arm constructing `Svar1RecordSource`.
- **Modify** `src/lib.rs` — `pub mod svar1_reader;`, the `run_svar1_conversion_pipeline` pyfunction, and its `m.add_function(...)` registration.
- **Modify** `python/genoray/_svar2.py` — the `from_svar1` classmethod + private helpers (`_read_svar1_metadata`, `_svar1_index_arrays`, `_svar1_fields_manifest`).
- **Create** `tests/test_svar2_from_svar1.py` — validation/guard tests + round-trip parity + genotype-membership + field round-trip + `no_reference`.
- **Modify** `skills/genoray-api/SKILL.md`, `CHANGELOG.md` — docs.

---

## Task 1: Rust — `build_variant_major` transpose (pure, unit-tested)

The one genuinely new algorithm: invert SVAR1's sample-major CSR into a variant-major carrier list for one contig. Pure function over slices so it is unit-testable without files.

**Files:**
- Create: `src/svar1_reader.rs`
- Modify: `src/lib.rs` (add `pub mod svar1_reader;` near the other `pub mod` lines, e.g. next to `pub mod pgen_reader;`)

**Interfaces:**
- Produces: `pub fn build_variant_major(variant_idxs: &[i32], offsets: &[i64], num_haps: usize, contig_start: i32, n_local: usize) -> Vec<Vec<(u32, u64)>>` — index `local_vid` → `Vec<(hap_column, flat_entry_index)>`. `flat_entry_index` indexes both `variant_idxs` and any per-entry field array.

- [ ] **Step 1: Write the failing test**

Add to the bottom of `src/svar1_reader.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // 2 samples × ploidy 2 = 4 haplotypes. Global variant ids 0..5.
    // Contig under test starts at global id 2, has 3 local variants (ids 2,3,4).
    // Per-hap sorted global ids (offsets CSR):
    //   hap0: [0, 2, 4]   hap1: [3]   hap2: [2]   hap3: []
    #[test]
    fn transpose_buckets_carriers_by_local_variant() {
        let variant_idxs: Vec<i32> = vec![0, 2, 4, /*h0*/ 3, /*h1*/ 2 /*h2*/];
        let offsets: Vec<i64> = vec![0, 3, 4, 5, 5]; // len num_haps+1 = 5
        let got = build_variant_major(&variant_idxs, &offsets, 4, 2, 3);

        // local 0 (gid 2): hap0 at entry 1, hap2 at entry 4
        assert_eq!(got[0], vec![(0u32, 1u64), (2u32, 4u64)]);
        // local 1 (gid 3): hap1 at entry 3
        assert_eq!(got[1], vec![(1u32, 3u64)]);
        // local 2 (gid 4): hap0 at entry 2
        assert_eq!(got[2], vec![(0u32, 2u64)]);
    }

    #[test]
    fn transpose_empty_contig_is_all_empty() {
        let got = build_variant_major(&[0, 1], &[0, 1, 2, 2, 2], 4, 100, 2);
        assert_eq!(got, vec![Vec::new(), Vec::new()]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run cargo test --no-default-features --features conversion build_variant_major transpose 2>&1 | tail -20
```
Expected: FAIL — `cannot find function build_variant_major` (module/function not defined).

- [ ] **Step 3: Write minimal implementation**

Put at the top of `src/svar1_reader.rs` (above the test module):

```rust
//! SVAR1 record source: reconstruct variant-major `RawRecord`s from SVAR1's
//! sample-major sparse store, so `from_svar1` reuses the shared conversion spine
//! (`chunk_assembler` onward) exactly as VCF/PGEN do. See
//! `docs/superpowers/specs/2026-07-13-svar1-to-svar2-conversion-design.md`.

/// Invert SVAR1's sample-major CSR (`variant_idxs`/`offsets`) into a variant-major
/// carrier list for ONE contig. Returns, per local variant `0..n_local`, the
/// `(haplotype column, flat entry index)` pairs of the haplotypes carrying it.
///
/// `variant_idxs` holds each haplotype's sorted global non-ref variant ids;
/// `offsets` is the CSR over `num_haps = num_samples * ploidy` haplotypes
/// (`offsets.len() == num_haps + 1`). Contigs are contiguous in global-id space,
/// so this contig owns global ids `[contig_start, contig_start + n_local)`; per
/// hap we binary-search that sub-range (ids are sorted) rather than scanning all
/// entries. `flat entry index` indexes both `variant_idxs` and any per-entry
/// field array (they share `offsets`).
pub fn build_variant_major(
    variant_idxs: &[i32],
    offsets: &[i64],
    num_haps: usize,
    contig_start: i32,
    n_local: usize,
) -> Vec<Vec<(u32, u64)>> {
    let contig_end = contig_start + n_local as i32;
    let mut buckets: Vec<Vec<(u32, u64)>> = vec![Vec::new(); n_local];
    for h in 0..num_haps {
        let lo = offsets[h] as usize;
        let hi = offsets[h + 1] as usize;
        let hap = &variant_idxs[lo..hi];
        let s = hap.partition_point(|&g| g < contig_start);
        let e = hap.partition_point(|&g| g < contig_end);
        for k in s..e {
            let local = (hap[k] - contig_start) as usize;
            buckets[local].push((h as u32, (lo + k) as u64));
        }
    }
    buckets
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run cargo test --no-default-features --features conversion build_variant_major transpose 2>&1 | tail -20
```
Expected: PASS (both `transpose_*` tests).

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add src/svar1_reader.rs src/lib.rs
git commit -m "feat(svar2): SVAR1 sample-major->variant-major transpose"
```

---

## Task 2: Rust — `Svar1RecordSource` + `SourceSpec::Svar1` wiring

Wrap the transpose in a `RecordSource` that mmaps the raw arrays and serves records, and wire it into `process_chromosome`.

**Files:**
- Modify: `src/svar1_reader.rs`
- Modify: `src/orchestrator.rs` (the `SourceSpec` enum ~line 44, and the reader-thread `match source { … }` ~line 168)

**Interfaces:**
- Consumes: `build_variant_major` (Task 1); `crate::record_source::{RawRecord, RecordSource}`; `crate::field::FieldSpec`; `crate::error::ConversionError`.
- Produces:
  - `pub struct Svar1RecordSource` with `pub fn new(svar1_dir: &str, contig_start: usize, n_local: usize, num_samples: usize, ploidy: usize, pos: Vec<u32>, ref_bytes: Vec<u8>, ref_offsets: Vec<i64>, alt_bytes: Vec<u8>, alt_offsets: Vec<i64>, format_fields: &[FieldSpec], format_src_dtypes: &[String]) -> Result<Self, ConversionError>`
  - `SourceSpec::Svar1 { svar1_dir: String, contig_start: usize, n_local: usize, pos: Vec<u32>, ref_bytes: Vec<u8>, ref_offsets: Vec<i64>, alt_bytes: Vec<u8>, alt_offsets: Vec<i64>, format_fields: Vec<FieldSpec>, format_src_dtypes: Vec<String> }`

Notes for the implementer:
- `pos` here is already 0-based (`POS-1`); the pyfunction (Task 3) subtracts 1 before building it — do NOT subtract again.
- `ref_bytes`/`ref_offsets` (and `alt_*`) are a packed-string layout: allele `i`'s bytes are `ref_bytes[ref_offsets[i]..ref_offsets[i+1]]`. `ref_offsets.len() == n_local + 1`. Biallelic ⇒ exactly one ALT per variant, so `alt_offsets` is also length `n_local + 1`.
- `format_src_dtypes[j]` is the numpy dtype string of `format_fields[j]`'s on-disk array (e.g. `"float32"`, `"int16"`), read from SVAR1 `metadata.fields`. Used to interpret the raw `{name}.npy` bytes; values are widened to `f64` for `RawRecord.format_raw`.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` module in `src/svar1_reader.rs`:

```rust
    use std::io::Write;

    fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(bytemuck::cast_slice(data)).unwrap();
    }

    #[test]
    fn record_source_yields_variant_major_records_with_dosage() {
        // 2 samples, ploidy 2 (4 haps). One contig, global ids == local ids 0..2.
        //   var0 (gid0): carried by hap0 (S0 hap0) and hap2 (S1 hap0)
        //   var1 (gid1): carried by hap3 (S1 hap1)
        // Per-hap CSR: hap0:[0] hap1:[] hap2:[0] hap3:[1]
        let tmp = std::env::temp_dir().join(format!("svar1_rs_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        write_raw::<i32>(&tmp, "variant_idxs.npy", &[0, 0, 1]);
        write_raw::<i64>(&tmp, "offsets.npy", &[0, 1, 1, 2, 3]);
        // dosages aligned 1:1 with variant_idxs entries
        write_raw::<f32>(&tmp, "dosages.npy", &[0.5, 1.5, 2.5]);

        let ds = crate::field::FieldSpec {
            name: "dosages".into(),
            category: crate::field::FieldCategory::Format,
            htype: crate::field::HtslibType::Float,
            dtype: crate::field::StorageDtype::F32,
            default: None,
        };
        let mut src = Svar1RecordSource::new(
            tmp.to_str().unwrap(), 0, 2, 2, 2,
            vec![10, 20],                 // pos (0-based) for var0, var1
            b"AC".to_vec(), vec![0, 1, 2],  // REF: "A","C"
            b"GT".to_vec(), vec![0, 1, 2],  // ALT: "G","T"
            std::slice::from_ref(&ds), &["float32".to_string()],
        ).unwrap();

        let r0 = src.next_record().unwrap().unwrap();
        assert_eq!(r0.pos, 10);
        assert_eq!(r0.reference, b"A");
        assert_eq!(r0.alts, vec![b"G".to_vec()]);
        assert_eq!(r0.gt, vec![1, 0, 1, 0]); // hap0 & hap2 carry ALT1
        // format_raw[0] = Some(per-sample). S0 carried on hap0 -> 0.5; S1 on hap2 -> 2.5
        let ds0 = r0.format_raw[0].as_ref().unwrap();
        assert_eq!(ds0[0], vec![0.5]);
        assert_eq!(ds0[1], vec![2.5]);

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.gt, vec![0, 0, 0, 1]); // only S1 hap1
        let ds1 = r1.format_raw[0].as_ref().unwrap();
        assert!(ds1[0][0].is_nan());          // S0 non-carrier -> missing sentinel (NaN)
        assert_eq!(ds1[1], vec![1.5]);        // S1 carrier

        assert!(src.next_record().unwrap().is_none());
        std::fs::remove_dir_all(&tmp).ok();
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run cargo test --no-default-features --features conversion record_source_yields 2>&1 | tail -20
```
Expected: FAIL — `cannot find struct Svar1RecordSource` / `new`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/svar1_reader.rs` (above the test module). Uses `memmap2` + `bytemuck` (both existing deps):

```rust
use crate::error::ConversionError;
use crate::field::FieldSpec;
use crate::record_source::{RawRecord, RecordSource};
use memmap2::Mmap;
use std::fs::File;

/// A per-entry field array, mmap'd raw and read as f64 on demand.
enum FieldArray {
    F32(Mmap),
    F16(Mmap),
    I8(Mmap),
    I16(Mmap),
    I32(Mmap),
    U8(Mmap),
    U16(Mmap),
    U32(Mmap),
}

impl FieldArray {
    fn open(path: &std::path::Path, np_dtype: &str) -> Result<Self, ConversionError> {
        let mmap = mmap_ro(path)?;
        Ok(match np_dtype {
            "float32" => FieldArray::F32(mmap),
            "float16" => FieldArray::F16(mmap),
            "int8" => FieldArray::I8(mmap),
            "int16" => FieldArray::I16(mmap),
            "int32" => FieldArray::I32(mmap),
            "uint8" => FieldArray::U8(mmap),
            "uint16" => FieldArray::U16(mmap),
            "uint32" => FieldArray::U32(mmap),
            other => {
                return Err(ConversionError::Input(format!(
                    "SVAR1 field dtype {other:?} is unsupported for conversion"
                )))
            }
        })
    }

    fn value_f64(&self, entry: usize) -> f64 {
        match self {
            FieldArray::F32(m) => bytemuck::cast_slice::<u8, f32>(m)[entry] as f64,
            FieldArray::F16(m) => f64::from(bytemuck::cast_slice::<u8, half::f16>(m)[entry]),
            FieldArray::I8(m) => bytemuck::cast_slice::<u8, i8>(m)[entry] as f64,
            FieldArray::I16(m) => bytemuck::cast_slice::<u8, i16>(m)[entry] as f64,
            FieldArray::I32(m) => bytemuck::cast_slice::<u8, i32>(m)[entry] as f64,
            FieldArray::U8(m) => m[entry] as f64,
            FieldArray::U16(m) => bytemuck::cast_slice::<u8, u16>(m)[entry] as f64,
            FieldArray::U32(m) => bytemuck::cast_slice::<u8, u32>(m)[entry] as f64,
        }
    }
}

fn mmap_ro(path: &std::path::Path) -> Result<Mmap, ConversionError> {
    let f = File::open(path).map_err(|e| ConversionError::Io {
        context: format!("open {path:?}"),
        source: e,
    })?;
    // SAFETY: read-only map of a file we do not mutate for the source's lifetime.
    unsafe { Mmap::map(&f) }.map_err(|e| ConversionError::Io {
        context: format!("mmap {path:?}"),
        source: e,
    })
}

pub struct Svar1RecordSource {
    num_haps: usize,
    ploidy: usize,
    num_samples: usize,
    pos: Vec<u32>,
    ref_bytes: Vec<u8>,
    ref_offsets: Vec<i64>,
    alt_bytes: Vec<u8>,
    alt_offsets: Vec<i64>,
    buckets: Vec<Vec<(u32, u64)>>, // per local variant -> (hap col, entry idx)
    fields: Vec<(FieldSpec, FieldArray)>,
    cursor: usize,
    n_local: usize,
}

impl Svar1RecordSource {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        svar1_dir: &str,
        contig_start: usize,
        n_local: usize,
        num_samples: usize,
        ploidy: usize,
        pos: Vec<u32>,
        ref_bytes: Vec<u8>,
        ref_offsets: Vec<i64>,
        alt_bytes: Vec<u8>,
        alt_offsets: Vec<i64>,
        format_fields: &[FieldSpec],
        format_src_dtypes: &[String],
    ) -> Result<Self, ConversionError> {
        let dir = std::path::Path::new(svar1_dir);
        let num_haps = num_samples * ploidy;

        let vi_mmap = mmap_ro(&dir.join("variant_idxs.npy"))?;
        let off_mmap = mmap_ro(&dir.join("offsets.npy"))?;
        let variant_idxs: &[i32] = bytemuck::cast_slice(&vi_mmap);
        let offsets: &[i64] = bytemuck::cast_slice(&off_mmap);
        if offsets.len() != num_haps + 1 {
            return Err(ConversionError::Input(format!(
                "SVAR1 offsets.npy has {} entries; expected num_samples*ploidy+1 = {}",
                offsets.len(),
                num_haps + 1
            )));
        }
        let buckets =
            build_variant_major(variant_idxs, offsets, num_haps, contig_start as i32, n_local);

        let mut fields = Vec::with_capacity(format_fields.len());
        for (spec, np_dtype) in format_fields.iter().zip(format_src_dtypes) {
            let arr = FieldArray::open(&dir.join(format!("{}.npy", spec.name)), np_dtype)?;
            fields.push((spec.clone(), arr));
        }

        Ok(Self {
            num_haps,
            ploidy,
            num_samples,
            pos,
            ref_bytes,
            ref_offsets,
            alt_bytes,
            alt_offsets,
            buckets,
            fields,
            cursor: 0,
            n_local,
        })
    }
}

impl RecordSource for Svar1RecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        let v = self.cursor;
        if v >= self.n_local {
            return Ok(None);
        }
        self.cursor += 1;

        let reference =
            self.ref_bytes[self.ref_offsets[v] as usize..self.ref_offsets[v + 1] as usize].to_vec();
        let alt =
            self.alt_bytes[self.alt_offsets[v] as usize..self.alt_offsets[v + 1] as usize].to_vec();

        let mut gt = vec![0i32; self.num_haps];
        for &(col, _e) in &self.buckets[v] {
            gt[col as usize] = 1; // biallelic: ALT1
        }

        let format_raw = self
            .fields
            .iter()
            .map(|(spec, arr)| {
                let sent = spec.missing_sentinel();
                let mut per_sample: Vec<Vec<f64>> = vec![vec![sent]; self.num_samples];
                for &(col, e) in &self.buckets[v] {
                    let s = col as usize / self.ploidy;
                    per_sample[s] = vec![arr.value_f64(e as usize)];
                }
                Some(per_sample)
            })
            .collect();

        Ok(Some(RawRecord {
            pos: self.pos[v],
            reference,
            alts: vec![alt],
            gt,
            info_raw: Vec::new(), // SVAR1 has no INFO fields
            format_raw,
        }))
    }
}
```

- [ ] **Step 4: Wire `SourceSpec::Svar1` into the orchestrator**

In `src/orchestrator.rs`, add to the `SourceSpec` enum (after the `Pgen { … }` variant):

```rust
    Svar1 {
        svar1_dir: String,
        contig_start: usize,
        n_local: usize,
        pos: Vec<u32>,
        ref_bytes: Vec<u8>,
        ref_offsets: Vec<i64>,
        alt_bytes: Vec<u8>,
        alt_offsets: Vec<i64>,
        format_fields: Vec<crate::field::FieldSpec>,
        format_src_dtypes: Vec<String>,
    },
```

In the reader-thread `match source { … }` (where `SourceSpec::Pgen => …` is), add:

```rust
                    SourceSpec::Svar1 {
                        svar1_dir,
                        contig_start,
                        n_local,
                        pos,
                        ref_bytes,
                        ref_offsets,
                        alt_bytes,
                        alt_offsets,
                        format_fields,
                        format_src_dtypes,
                    } => Box::new(crate::svar1_reader::Svar1RecordSource::new(
                        &svar1_dir,
                        contig_start,
                        n_local,
                        s_refs.len(),
                        ploidy,
                        pos,
                        ref_bytes,
                        ref_offsets,
                        alt_bytes,
                        alt_offsets,
                        &format_fields,
                        &format_src_dtypes,
                    )?),
```

- [ ] **Step 5: Run test + build to verify pass**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run cargo test --no-default-features --features conversion record_source_yields 2>&1 | tail -20
pixi run cargo check --no-default-features --features conversion 2>&1 | tail -5
```
Expected: `record_source_yields_variant_major_records_with_dosage` PASSES; `cargo check` succeeds (orchestrator match is exhaustive).

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add src/svar1_reader.rs src/orchestrator.rs
git commit -m "feat(svar2): Svar1RecordSource + SourceSpec::Svar1 wiring"
```

---

## Task 3: Rust — `run_svar1_conversion_pipeline` pyfunction

Mirror `run_pgen_conversion_pipeline`: fan contigs over `process_chromosome`, then `finalize_fields` + `write_meta`.

**Files:**
- Modify: `src/lib.rs` (add the pyfunction after `run_pgen_conversion_pipeline`; register it in the `#[pymodule]` block alongside `m.add_function(wrap_pyfunction!(run_pgen_conversion_pipeline, m)?)?;`)

**Interfaces:**
- Consumes: `SourceSpec::Svar1` (Task 2); `crate::field::parse_manifest`; `crate::field_finalize::finalize_fields`; `crate::meta::write_meta`.
- Produces (called from Python in Task 4):
  `run_svar1_conversion_pipeline(svar1_dir, reference_path, chroms, contig_starts, contig_lens, output_dir, samples, ploidy, chunk_size, max_threads, long_allele_capacity, skip_out_of_scope, signatures, pos_per_contig, ref_bytes_per_contig, ref_offsets_per_contig, alt_bytes_per_contig, alt_offsets_per_contig, format_fields, format_src_dtypes) -> int`

- [ ] **Step 1: Add the pyfunction**

In `src/lib.rs`, after `run_pgen_conversion_pipeline`:

```rust
/// Convert a SVAR1 (`SparseVar`) store to an SVAR2 store natively (no htslib).
///
/// Per-contig `POS`/`REF`/`ALT` come from Python (it reads `index.arrow` via
/// polars); the big sample-major sparse arrays and per-entry field arrays are
/// mmap'd in Rust from `svar1_dir`. `contig_starts[i]`/`contig_lens[i]` give
/// contig `i`'s global variant-id start and length. `pos_per_contig[i]` is 0-based
/// (`POS-1`). `format_fields` is the SVAR1 field manifest (all FORMAT);
/// `format_src_dtypes[j]` is the numpy dtype of `format_fields[j]`'s on-disk array.
#[cfg(feature = "conversion")]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (svar1_dir, reference_path, chroms, contig_starts, contig_lens, output_dir, samples, ploidy, chunk_size, max_threads, long_allele_capacity, skip_out_of_scope, signatures, pos_per_contig, ref_bytes_per_contig, ref_offsets_per_contig, alt_bytes_per_contig, alt_offsets_per_contig, format_fields, format_src_dtypes))]
fn run_svar1_conversion_pipeline(
    py: Python,
    svar1_dir: String,
    reference_path: Option<String>,
    chroms: Vec<String>,
    contig_starts: Vec<usize>,
    contig_lens: Vec<usize>,
    output_dir: String,
    samples: Vec<String>,
    ploidy: usize,
    chunk_size: usize,
    max_threads: Option<usize>,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    signatures: bool,
    pos_per_contig: Vec<Vec<u32>>,
    ref_bytes_per_contig: Vec<Vec<u8>>,
    ref_offsets_per_contig: Vec<Vec<i64>>,
    alt_bytes_per_contig: Vec<Vec<u8>>,
    alt_offsets_per_contig: Vec<Vec<i64>>,
    format_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
    format_src_dtypes: Vec<String>,
) -> PyResult<usize> {
    let n = chroms.len();
    if [
        contig_starts.len(),
        contig_lens.len(),
        pos_per_contig.len(),
        ref_bytes_per_contig.len(),
        ref_offsets_per_contig.len(),
        alt_bytes_per_contig.len(),
        alt_offsets_per_contig.len(),
    ]
    .iter()
    .any(|&l| l != n)
    {
        return Err(PyValueError::new_err(
            "all per-contig inputs must have the same length as `chroms`",
        ));
    }
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
    let fields = crate::field::parse_manifest(format_fields)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Move per-contig owned data into jobs before detaching.
    let mut jobs: Vec<_> = Vec::with_capacity(n);
    for i in 0..n {
        jobs.push((
            chroms[i].clone(),
            contig_starts[i],
            contig_lens[i],
            pos_per_contig[i].clone(),
            ref_bytes_per_contig[i].clone(),
            ref_offsets_per_contig[i].clone(),
            alt_bytes_per_contig[i].clone(),
            alt_offsets_per_contig[i].clone(),
        ));
    }

    let results: Vec<Result<u64, crate::error::ConversionError>> = py.detach(|| {
        let available_cores = match max_threads {
            Some(t) if t > 0 => t,
            _ => std::thread::available_parallelism().unwrap().get(),
        };
        let plan = crate::budget::plan_thread_budget(available_cores, jobs.len());
        let concurrent_chroms = plan.concurrent_chroms;
        let processing_threads = plan.processing_threads;
        println!(
            "Pipeline Config (SVAR1): {} concurrent chromosomes | {} processing threads each.",
            concurrent_chroms, processing_threads
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrent_chroms)
            .thread_name(|i| format!("chrom-{}", i))
            .build()
            .expect("build chrom pool");

        pool.install(|| {
            jobs.into_par_iter()
                .map(|(chrom, start, len, pos, rb, ro, ab, ao)| {
                    orchestrator::process_chromosome(
                        orchestrator::SourceSpec::Svar1 {
                            svar1_dir: svar1_dir.clone(),
                            contig_start: start,
                            n_local: len,
                            pos,
                            ref_bytes: rb,
                            ref_offsets: ro,
                            alt_bytes: ab,
                            alt_offsets: ao,
                            format_fields: fields.clone(),
                            format_src_dtypes: format_src_dtypes.clone(),
                        },
                        reference_path.as_deref(),
                        &chrom,
                        &output_dir,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        long_allele_capacity,
                        skip_out_of_scope,
                        processing_threads,
                        signatures,
                        &fields,
                    )
                })
                .collect()
        })
    });

    let mut dropped = 0u64;
    for r in results {
        dropped += r?;
    }

    let resolved_fields = crate::field_finalize::finalize_fields(
        std::path::Path::new(&output_dir),
        &chroms,
        &fields,
    )?;
    crate::meta::write_meta(
        std::path::Path::new(&output_dir),
        crate::meta::FORMAT_VERSION,
        &samples,
        &chroms,
        ploidy,
        &resolved_fields,
    )
    .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("failed to write meta.json: {e}")))?;

    Ok(dropped as usize)
}
```

Register it in the `#[pymodule]` block (next to the other `add_function` calls, keeping the `#[cfg(feature = "conversion")]` gating pattern used for `run_pgen_conversion_pipeline`):

```rust
    m.add_function(wrap_pyfunction!(run_svar1_conversion_pipeline, m)?)?;
```

- [ ] **Step 2: Build to verify it compiles**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run cargo check --no-default-features --features conversion 2>&1 | tail -8
```
Expected: compiles clean (only pre-existing warnings, if any).

- [ ] **Step 3: Rebuild the extension so Python can import it**

```bash
pixi run maturin develop 2>&1 | tail -5
pixi run python -c "from genoray import _core; assert hasattr(_core, 'run_svar1_conversion_pipeline'); print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add src/lib.rs
git commit -m "feat(svar2): run_svar1_conversion_pipeline pyfunction"
```

---

## Task 4: Python — `SparseVar2.from_svar1` classmethod + helpers

The thin shim: read metadata + index, build the field manifest, marshal to Rust. Validation-only paths are unit-tested here (they raise before touching Rust).

**Files:**
- Modify: `python/genoray/_svar2.py`
- Test: `tests/test_svar2_from_svar1.py` (create)

**Interfaces:**
- Consumes: `_core.run_svar1_conversion_pipeline` (Task 3); `genoray._svar._core.SparseVar` (`_is_biallelic`, metadata); `_auto_chunk_size` (existing in `_svar2.py`).
- Produces:
  - `SparseVar2.from_svar1(out, source, reference=None, *, no_reference=False, skip_out_of_scope=False, chunk_size=None, threads=None, overwrite=False, long_allele_capacity=8*1024*1024, signatures=False) -> int`
  - `_read_svar1_metadata(source: Path) -> tuple[list[str], int, list[str], dict[str, str]]` → `(samples, ploidy, contigs, fields)`
  - `_svar1_index_arrays(source, contigs) -> tuple[list[int], list[int], list[np.ndarray], list[bytes], list[np.ndarray], list[bytes], list[np.ndarray]]` → per-contig `(starts, lens, pos0_arrays, ref_bytes, ref_offsets, alt_bytes, alt_offsets)`
  - `_svar1_fields_manifest(fields: dict[str, str]) -> tuple[list[tuple], list[str]]` → `(format_field_tuples, src_dtypes)`; drops `mutcat`.

- [ ] **Step 1: Write the failing validation tests**

Create `tests/test_svar2_from_svar1.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from genoray import SparseVar, SparseVar2
from genoray import VCF as _V1VCF
from tests.test_svar2_from_vcf import _write_ref, _write_vcf


def _build_svar1(tmp_path: Path, *, with_dosages: bool = False) -> Path:
    """A SVAR1 store from the shared 40bp fixture VCF (2 SNP/indel biallelic vars)."""
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    v1_out = tmp_path / "in.svar"
    v1 = _V1VCF(str(vcf))
    if with_dosages:
        v1.dosage_field = "DS"
    SparseVar.from_vcf(
        v1_out, v1, max_mem="10m", overwrite=True, with_dosages=with_dosages
    )
    return v1_out


def test_from_svar1_requires_reference_or_opt_out(tmp_path: Path):
    src = _build_svar1(tmp_path)
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_svar1(tmp_path / "out", src, threads=1)


def test_from_svar1_reference_and_no_reference_conflict(tmp_path: Path):
    ref = _write_ref(tmp_path)
    src = _build_svar1(tmp_path)
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_svar1(
            tmp_path / "out", src, ref, no_reference=True, threads=1
        )


def test_from_svar1_refuses_existing_out_without_overwrite(tmp_path: Path):
    src = _build_svar1(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        SparseVar2.from_svar1(out, src, no_reference=True, threads=1)
```

Note: `_write_vcf` declares only `GT`, so `with_dosages=True` here is exercised in Task 5 with a dosage-carrying VCF. The `_build_svar1` default (`with_dosages=False`) is what these validation tests use.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_svar2_from_svar1.py -x -q 2>&1 | tail -20
```
Expected: FAIL — `AttributeError: type object 'SparseVar2' has no attribute 'from_svar1'`.

- [ ] **Step 3: Implement `from_svar1` + helpers**

Add to `python/genoray/_svar2.py`. Put the classmethod next to `from_pgen`, and the helpers near `_pvar_contig_ranges`. Imports needed at top: `import json`, `import numpy as np`, `import polars as pl` (add if absent — `pl` is already used elsewhere in the module).

```python
    @classmethod
    def from_svar1(
        cls,
        out: str | Path,
        source: str | Path,
        reference: str | Path | None = None,
        *,
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        chunk_size: int | None = None,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
        signatures: bool = False,
    ) -> int:
        """Convert a SVAR1 (``SparseVar``) store to an SVAR2 store natively.

        Reads no VCF and no htslib: SVAR1 is already sparse, so this reconstructs
        variant records from SVAR1's arrays and reuses the same conversion spine
        as :meth:`from_vcf`.

        Exactly one of `reference` or `no_reference=True` is required, same meaning
        as :meth:`from_vcf`. `ploidy` is read from SVAR1's metadata. Returns the
        number of out-of-scope (symbolic/breakend) ALTs dropped.

        Only **biallelic** SVAR1 stores are supported (SVAR1's ``geno==1`` model);
        multiallelic input raises. All SVAR1 FORMAT fields (e.g. ``dosages``) are
        carried through; ``mutcat`` is dropped (pass `signatures=True` to recompute
        signatures from the reference). Because SVAR1 discarded non-carrier FORMAT
        values, a dense-routed variant's non-carrier cells are filled with the
        field's default/missing sentinel — field output is byte-identical to
        :meth:`from_vcf` only for var_key (carrier-only) routing.
        """
        out = Path(out)
        source = Path(source)

        if (reference is None) == (not no_reference):
            raise ValueError(
                "provide exactly one of `reference` (a FASTA path) or "
                "`no_reference=True`"
            )
        if signatures and no_reference:
            raise ValueError("signatures=True requires a reference (not no_reference).")
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        if not source.exists():
            raise FileNotFoundError(source)

        sv1 = SparseVar(source)
        if not sv1._is_biallelic:
            raise ValueError(
                "from_svar1 supports only biallelic SVAR1 stores; this store has "
                "multiallelic variants. Re-create it biallelically first."
            )
        samples, ploidy, contigs, fields = _read_svar1_metadata(source)
        n_samples = len(samples)
        if n_samples == 0:
            raise ValueError(f"No samples found in {source}.")

        (
            starts,
            lens,
            pos_pc,
            ref_bytes_pc,
            ref_off_pc,
            alt_bytes_pc,
            alt_off_pc,
        ) = _svar1_index_arrays(source, contigs)
        format_tuples, src_dtypes = _svar1_fields_manifest(fields)

        if chunk_size is None:
            chunk_size = _auto_chunk_size(n_samples, ploidy)

        out.parent.mkdir(parents=True, exist_ok=True)
        return _core.run_svar1_conversion_pipeline(
            str(source),
            None if no_reference else str(reference),
            contigs,
            starts,
            lens,
            str(out),
            samples,
            ploidy,
            chunk_size,
            threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            pos_pc,
            ref_bytes_pc,
            ref_off_pc,
            alt_bytes_pc,
            alt_off_pc,
            format_tuples,
            src_dtypes,
        )
```

Helpers (module-level):

```python
def _read_svar1_metadata(
    source: Path,
) -> tuple[list[str], int, list[str], dict[str, str]]:
    """(samples, ploidy, contigs, fields) from a SVAR1 metadata.json."""
    meta = json.loads((source / "metadata.json").read_text())
    return (
        list(meta["samples"]),
        int(meta["ploidy"]),
        list(meta["contigs"]),
        dict(meta.get("fields", {})),
    )


def _pack_strings(values: list[str]) -> tuple[bytes, "np.ndarray"]:
    """Pack a list of ASCII allele strings into (concatenated bytes, i64 offsets)
    with offsets length len(values)+1."""
    encoded = [v.encode("ascii") for v in values]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(b) for b in encoded], out=offsets[1:])
    return b"".join(encoded), offsets


def _svar1_index_arrays(source: Path, contigs: list[str]):
    """Per-contig POS(0-based)/REF/ALT arrays + global contig start/len ranges.

    SVAR1's index.arrow is variant-major and contig-contiguous; POS is 1-based.
    """
    df = pl.read_ipc(source / "index.arrow", columns=["CHROM", "POS", "REF", "ALT"])
    # ALT is comma-Utf8 on disk; biallelic => a single token per row.
    starts: list[int] = []
    lens: list[int] = []
    pos_pc, ref_b, ref_o, alt_b, alt_o = [], [], [], [], []
    cursor = 0
    for c in contigs:
        sub = df.filter(pl.col("CHROM") == c)
        n = sub.height
        starts.append(cursor)
        lens.append(n)
        cursor += n
        pos_pc.append((sub["POS"].to_numpy().astype(np.int64) - 1).astype(np.uint32))
        rb, ro = _pack_strings(sub["REF"].to_list())
        ab, ao = _pack_strings(sub["ALT"].to_list())
        ref_b.append(rb)
        ref_o.append(ro)
        alt_b.append(ab)
        alt_o.append(ao)
    return starts, lens, pos_pc, ref_b, ref_o, alt_b, alt_o


def _svar1_fields_manifest(
    fields: dict[str, str],
) -> tuple[list[tuple[str, str, str, None, None]], list[str]]:
    """Map SVAR1 metadata.fields -> (FORMAT FieldSpec tuples, source numpy dtypes).

    Every SVAR1 custom field is FORMAT. `mutcat` is dropped (signature machinery).
    htype is inferred from the numpy dtype; storage dtype is left None (Auto).
    """
    tuples: list[tuple[str, str, str, None, None]] = []
    src_dtypes: list[str] = []
    for name, np_dtype in fields.items():
        if name == "mutcat":
            continue
        htype = "float" if np.dtype(np_dtype).kind == "f" else "int"
        tuples.append((name, "format", htype, None, None))
        src_dtypes.append(np_dtype)
    return tuples, src_dtypes
```

- [ ] **Step 4: Run validation tests to verify they pass**

```bash
pixi run pytest tests/test_svar2_from_svar1.py -x -q 2>&1 | tail -20
```
Expected: the three validation tests PASS.

- [ ] **Step 5: Typecheck + commit**

```bash
pixi run typecheck 2>&1 | tail -5
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add python/genoray/_svar2.py tests/test_svar2_from_svar1.py
git commit -m "feat(svar2): SparseVar2.from_svar1 Python shim + validation"
```

---

## Task 5: Integration tests — round-trip parity, genotype membership, fields, no_reference

The correctness proof: `from_svar1` must match `from_vcf` on genotypes and preserve SVAR1's carrier dosages through the read API (`with_fields`).

**Files:**
- Modify: `tests/test_svar2_from_svar1.py`

**Interfaces:**
- Consumes: `SparseVar2.from_svar1` (Task 4); `SparseVar2.from_vcf`; `SparseVar2.decode`/`with_fields` (existing read API from PR #101); the `_write_ref`/`_write_vcf` fixture helpers.

- [ ] **Step 1: Write the round-trip + membership test**

Append to `tests/test_svar2_from_svar1.py`. First add a helper that decodes every haplotype's carried variants for a store, then compare v-from-svar1 vs v-from-vcf:

```python
import numpy as np


def _all_hap_variant_keys(sv: SparseVar2) -> dict:
    """Map (contig) -> per-hap set of (pos, ref?, alt) decoded records, so two
    stores can be compared for identical genotype membership.

    Uses the whole-contig decode path. Adjust the exact decode call to the store's
    public API (see tests/test_svar2_decode.py for the current signature)."""
    result = {}
    for contig in sv.contigs:
        # Decode the full contig for all samples; compare the ragged offsets +
        # decoded (pos, allele) payloads. See test_svar2_decode.py for the exact
        # method + return shape to assert against.
        result[contig] = sv  # placeholder: replace with real decode below
    return result


def test_from_svar1_matches_from_vcf_genotypes(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)

    # SVAR2 directly from the VCF (the reference / golden path).
    v_vcf = tmp_path / "from_vcf"
    SparseVar2.from_vcf(v_vcf, vcf, ref, threads=1)

    # SVAR1, then SVAR2 from SVAR1.
    src = _build_svar1(tmp_path)
    v_s1 = tmp_path / "from_svar1"
    dropped = SparseVar2.from_svar1(v_s1, src, ref, threads=1)
    assert dropped == 0

    a = SparseVar2(v_vcf)
    b = SparseVar2(v_s1)
    assert a.available_samples == b.available_samples
    assert a.contigs == b.contigs
    # meta.json ploidy/samples/contigs parity:
    assert (v_s1 / "meta.json").exists()
```

> **Implementer note (do this, don't skip):** replace `_all_hap_variant_keys`'s
> placeholder and this test's tail with a real decode comparison. Open
> `tests/test_svar2_decode.py` and `tests/test_svar2_from_pgen.py` (its
> `_assert_ragged_equal` helper) for the exact current decode call and ragged
> return shape, then assert the two stores' decoded records are element-for-element
> equal for every contig/sample. `from_pgen`'s test is the closest precedent —
> mirror its structure: decode both stores and compare offsets + per-field data.

- [ ] **Step 2: Run it, then fix the decode comparison to be real**

```bash
pixi run pytest tests/test_svar2_from_svar1.py::test_from_svar1_matches_from_vcf_genotypes -x -q 2>&1 | tail -30
```
Expected: PASS after you wire the real decode comparison (mirroring `test_svar2_from_pgen.py::_assert_ragged_equal`). The genotype streams must be byte-identical under matching normalization.

- [ ] **Step 3: Write the dosage field round-trip test**

Requires a VCF with a `DS` FORMAT field so SVAR1 can store dosages. Add a local fixture VCF (do not reuse `_write_vcf`, which is GT-only):

```python
def _write_dosage_vcf(d: Path) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DS\t1|0:1.0\t0|0:0.0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT:DS\t0|1:1.0\t1|1:2.0\n"
    )
    plain = d / "ds.vcf"
    plain.write_text(body)
    gz = d / "ds.vcf.gz"
    import subprocess

    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_svar1_carries_dosages(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_dosage_vcf(tmp_path)

    # SVAR1 with dosages.
    v1_out = tmp_path / "ds.svar"
    v1 = _V1VCF(str(vcf))
    v1.dosage_field = "DS"
    SparseVar.from_vcf(v1_out, v1, max_mem="10m", overwrite=True, with_dosages=True)

    # Convert; the SVAR1 field is named "dosages".
    out = tmp_path / "ds.svar2"
    SparseVar2.from_svar1(out, v1_out, ref, threads=1)

    sv2 = SparseVar2(out)
    assert "dosages" in sv2.available_fields
    assert sv2.available_fields["dosages"].category == "format"
    sv2f = sv2.with_fields(["dosages"])
    # Decode and assert carrier dosages equal SVAR1's stored values.
    # (Wire the exact decode assertion from test_svar2_fields_read.py — that file
    # is the precedent for reading a FORMAT field back and checking per-sample
    # carrier values. Carrier S0@var0 -> 1.0, S1@var1(indel) -> 2.0, etc.)
```

> **Implementer note:** finish the dosage assertion using
> `tests/test_svar2_fields_read.py` as the precedent for the exact `with_fields` +
> decode call and how per-sample FORMAT values surface. Assert carrier values match
> the VCF's `DS` (1.0/2.0), and do NOT assert on dense-routed non-carrier cells
> (documented caveat — they are sentinel/NaN).

- [ ] **Step 4: Write the no_reference + scope-guard tests**

```python
def test_from_svar1_no_reference_snp_indel(tmp_path: Path):
    src = _build_svar1(tmp_path)
    out = tmp_path / "noref"
    dropped = SparseVar2.from_svar1(out, src, no_reference=True, threads=1)
    assert dropped == 0
    assert (out / "meta.json").exists()
    assert SparseVar2(out).available_samples == ["S0", "S1"]
```

(Multiallelic rejection is covered by `_is_biallelic`; if a cheap multiallelic SVAR1
fixture is available, add a `pytest.raises(ValueError, match="biallelic")` case.
Otherwise the guard is exercised by the unit path in Task 4.)

- [ ] **Step 5: Run the full new test module**

```bash
pixi run pytest tests/test_svar2_from_svar1.py -q 2>&1 | tail -30
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add tests/test_svar2_from_svar1.py
git commit -m "test(svar2): from_svar1 round-trip parity + dosage carry-through"
```

---

## Task 6: Docs — SKILL.md + CHANGELOG

**Files:**
- Modify: `skills/genoray-api/SKILL.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Document `from_svar1` in SKILL.md**

Find the `SparseVar2.from_vcf` / `from_pgen` section and add a sibling entry:

```markdown
### `SparseVar2.from_svar1(out, source, reference=None, *, no_reference=False, skip_out_of_scope=False, chunk_size=None, threads=None, overwrite=False, long_allele_capacity=8*1024*1024, signatures=False) -> int`

Convert an existing SVAR1 (`SparseVar`) store to SVAR2 natively — no VCF re-read,
no htslib. Exactly one of `reference` (validate + left-align indels) or
`no_reference=True`. `ploidy` is taken from SVAR1 metadata (no kwarg). Returns the
count of out-of-scope ALTs dropped.

- **Biallelic SVAR1 only** (raises on multiallelic input).
- **Fields:** all SVAR1 FORMAT fields (e.g. `dosages`) are carried through, keyed
  by their SVAR1 name; `mutcat` is dropped — pass `signatures=True` to recompute
  signatures from the reference instead. Field values are byte-identical to
  `from_vcf` for var_key-routed variants; for dense-routed variants, non-carrier
  cells are the default/missing sentinel (SVAR1 stored no non-carrier values).
```

- [ ] **Step 2: Add the CHANGELOG entry**

Under `## Unreleased` → `### Added` in `CHANGELOG.md`:

```markdown
- `SparseVar2.from_svar1` converts an existing SVAR1 (`SparseVar`) store to SVAR2
  natively (no VCF re-read, no htslib), reusing the same conversion spine as
  `from_vcf`/`from_pgen` via a new `Svar1RecordSource`. Biallelic-only; `ploidy`
  is read from SVAR1 metadata. All SVAR1 FORMAT fields (e.g. `dosages`) are carried
  through keyed by their SVAR1 name; `mutcat` is dropped (`signatures=True`
  recomputes signatures from the reference). Genotype streams are byte-identical
  to `from_vcf` under matching normalization; carried FORMAT field values match
  for var_key-routed variants, while dense-routed non-carrier cells are the
  default/missing sentinel (SVAR1 discarded non-carrier FORMAT values).
```

- [ ] **Step 3: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add skills/genoray-api/SKILL.md CHANGELOG.md
git commit -m "docs(svar2): document SparseVar2.from_svar1"
```

---

## Task 7: Full verification pass

- [ ] **Step 1: Run the whole test suite (Rust + Python)**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run test-rust 2>&1 | tail -15
pixi run maturin develop 2>&1 | tail -3
pixi run test 2>&1 | tail -30
```
Expected: all Rust tests pass; full pytest suite passes (including the new `test_svar2_from_svar1.py`). Investigate any failure before proceeding — do not paper over it.

- [ ] **Step 2: Confirm no accidental public-API drift**

```bash
pixi run typecheck 2>&1 | tail -5
```
Expected: clean.

---

## Self-Review (completed during planning)

- **Spec coverage:** API shape (Task 4) ✓; `RecordSource`-seam architecture (Tasks 1–3) ✓; Python/Rust split + polars index read (Task 4) ✓; transpose (Task 1) ✓; field auto-carry + drop-mutcat + dense caveat (Tasks 2/4/6) ✓; biallelic guard (Task 4) ✓; normalization contract (Task 4) ✓; testing strategy incl. round-trip/membership/field/no_reference (Task 5) ✓; docs (Task 6) ✓; obsoleted arrow/SparseChunk/prepass — none reintroduced ✓.
- **Placeholder scan:** the only intentional "finish this" markers are the two decode-assertion notes in Task 5 Steps 1/3, which point at the exact precedent files (`test_svar2_from_pgen.py`, `test_svar2_fields_read.py`) because the current decode return shape must be read from live code rather than guessed — this is a deliberate instruction, not a vague TODO.
- **Type consistency:** `build_variant_major` signature identical across Tasks 1–2; `Svar1RecordSource::new` arg order matches the `SourceSpec::Svar1` destructuring in Task 2 and the pyfunction marshalling in Task 3; `run_svar1_conversion_pipeline` Python call args (Task 4) match the Rust `#[pyo3(signature=…)]` order (Task 3); `_svar1_fields_manifest` returns `(tuples, src_dtypes)` consumed positionally by `from_svar1`.

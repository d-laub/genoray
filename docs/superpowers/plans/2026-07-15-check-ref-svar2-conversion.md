# check_ref for SVAR2 conversion methods — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `check_ref` policy (`"e"` = error, current default; `"x"` = exclude offending record and continue) to all four SVAR2 conversion entry points (`from_vcf`, `from_pgen`, `from_svar1`, `from_vcf_list`), so a single REF-vs-FASTA disagreement no longer aborts the whole build.

**Architecture:** One shared decision helper (`apply_check_ref`) in `src/normalize.rs` replaces the two unconditional `validate_ref(...)?` call sites (`chunk_assembler.rs` for vcf/pgen/svar1; `vcf_list_reader.rs` for vcf_list). A `CheckRef` mode is threaded from the four pyo3 pipeline functions → `process_chromosome` / `run_vcf_list` → the assemblers. Excluded-record counts are logged (per-contig count + first offending locus); the Python return type (int = out-of-scope drops) is unchanged.

**Tech Stack:** Rust (pyo3, rust-htslib), Python (cyclopts CLI). Build via `pixi`. Rust extension is `genoray_core` / `_core`.

## Global Constraints

- **Conventional Commits** for every commit (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`).
- **Never edit `CHANGELOG.md`** — commitizen owns it.
- **Public-API changes MUST update `skills/genoray-api/SKILL.md`** in the same PR (Task 6).
- **Default is `"e"`** — behavior for existing callers must be byte-identical (non-breaking).
- **Rust tests** run with `pixi run bash -lc 'cargo test --no-default-features --features conversion <name>'` — the bare pyo3 test binary fails to link without `--no-default-features` (undefined `_Py_Dealloc`). The conversion pipeline code is behind `#[cfg(feature = "conversion")]`, so the `conversion` feature is required for these tests.
- **Guard against the NFS linker bus error**: `export CARGO_TARGET_DIR=/tmp/genoray-checkref-target` before any `cargo`/commit that triggers Rust hooks.
- **prek hooks** must be installed before committing (`pixi run prek-install`); they run `cargo fmt`/`clippy`/`ruff`/`pyrefly` on staged files.
- **`test-rust <arg>` filters by TEST NAME, not file** — a nonmatching name vacuously passes 0 tests. Prefer `cargo test --test <file_stem>` to run a whole integration-test file.
- **REF comparison is already case-insensitive** in `validate_ref` — do not add case handling.

---

## File Structure

- `src/normalize.rs` — **modify**: add `CheckRef` enum, `RefDecision` enum, `apply_check_ref`, `FromStr for CheckRef`, unit tests. (Task 1)
- `src/chunk_assembler.rs` — **modify**: `ChunkAssembler` gains a `check_ref` field + `ref_excluded` counter; `new` gains a param; `decompose_record` uses `apply_check_ref`; add `ref_excluded()` getter. (Task 2)
- `src/orchestrator.rs` — **modify**: `process_chromosome` gains a `check_ref: CheckRef` param, passed to `ChunkAssembler::new` and (Task 4) `VcfListRecordSource::new`; reader thread logs the per-contig excluded count; `run_vcf_list` threads the mode; `VcfListDroppedProxy` mirrors a `ref_excluded` atomic (Task 4). (Tasks 2, 4)
- `src/lib.rs` — **modify**: the four `#[pyo3]` pipeline fns gain a `check_ref: String` param, parsed to `CheckRef`. (Tasks 2, 3, 4)
- `src/vcf_list_reader.rs` — **modify**: `FileCursor` + `VcfListRecordSource` gain check_ref awareness + `ref_excluded` counters. (Task 4)
- `python/genoray/_svar2.py` — **modify**: `check_ref: Literal["e","x"] = "e"` kwarg on all four `from_*` methods + a `_validate_check_ref` helper + docstrings. (Tasks 2, 3, 4)
- `python/genoray/_cli/__main__.py` — **modify**: `--check-ref` flag on `genoray write`. (Task 5)
- `skills/genoray-api/SKILL.md` — **modify**: document the new kwarg. (Task 6)
- `tests/test_check_ref_e2e.rs` — **create**: Rust e2e for the shared chunk_assembler path (Task 2) + a vcf_list variant (Task 4).
- `tests/test_svar2_from_vcf.py` — **modify**: Python behavior test (Task 2).
- `tests/test_svar2_from_vcf_list.py` — **modify**: Python behavior test (Task 4).

---

## Task 1: `CheckRef` primitives in `normalize.rs`

**Files:**
- Modify: `src/normalize.rs` (add after `validate_ref`, ~line 103)
- Test: `src/normalize.rs` `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: existing `validate_ref(pos: u32, ref_allele: &[u8], ref_seq: &[u8]) -> Result<(), NormalizeError>`, `NormalizeError`.
- Produces:
  - `pub enum CheckRef { Error, Exclude }` (derives `Debug, Clone, Copy, PartialEq, Eq`)
  - `pub enum RefDecision { Keep, Exclude(NormalizeError) }`
  - `pub fn apply_check_ref(mode: CheckRef, pos: u32, ref_allele: &[u8], ref_seq: &[u8]) -> Result<RefDecision, NormalizeError>`
  - `impl std::str::FromStr for CheckRef { type Err = String; ... }`

- [ ] **Step 1: Write the failing unit tests**

Add to the `mod tests` block in `src/normalize.rs`:

```rust
    #[test]
    fn apply_check_ref_error_mode_propagates_mismatch() {
        let ref_seq = b"ACGTAC";
        assert!(matches!(
            apply_check_ref(CheckRef::Error, 2, b"AA", ref_seq),
            Err(NormalizeError::RefMismatch { pos: 2, .. })
        ));
    }

    #[test]
    fn apply_check_ref_error_mode_propagates_out_of_contig() {
        let ref_seq = b"ACGT";
        assert!(matches!(
            apply_check_ref(CheckRef::Error, 3, b"TAC", ref_seq),
            Err(NormalizeError::RefOutOfContig { .. })
        ));
    }

    #[test]
    fn apply_check_ref_exclude_mode_excludes_mismatch() {
        let ref_seq = b"ACGTAC";
        assert!(matches!(
            apply_check_ref(CheckRef::Exclude, 2, b"AA", ref_seq),
            Ok(RefDecision::Exclude(NormalizeError::RefMismatch { pos: 2, .. }))
        ));
    }

    #[test]
    fn apply_check_ref_exclude_mode_excludes_out_of_contig() {
        let ref_seq = b"ACGT";
        assert!(matches!(
            apply_check_ref(CheckRef::Exclude, 3, b"TAC", ref_seq),
            Ok(RefDecision::Exclude(NormalizeError::RefOutOfContig { .. }))
        ));
    }

    #[test]
    fn apply_check_ref_keeps_matching_ref_in_both_modes() {
        let ref_seq = b"ACGTAC";
        assert!(matches!(
            apply_check_ref(CheckRef::Error, 2, b"GT", ref_seq),
            Ok(RefDecision::Keep)
        ));
        assert!(matches!(
            apply_check_ref(CheckRef::Exclude, 2, b"GT", ref_seq),
            Ok(RefDecision::Keep)
        ));
    }

    #[test]
    fn check_ref_from_str_parses_e_and_x() {
        assert_eq!("e".parse::<CheckRef>(), Ok(CheckRef::Error));
        assert_eq!("x".parse::<CheckRef>(), Ok(CheckRef::Exclude));
        assert!("z".parse::<CheckRef>().is_err());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-checkref-target && pixi run bash -lc 'cargo test --no-default-features --features conversion --lib apply_check_ref check_ref_from_str'`
Expected: FAIL — `cannot find function apply_check_ref` / `cannot find type CheckRef`.

- [ ] **Step 3: Implement the primitives**

Insert after `validate_ref` (after line 103) in `src/normalize.rs`:

```rust
/// Policy for how a REF/FASTA disagreement is handled during conversion,
/// mirroring the `e`/`x` subset of `bcftools norm --check-ref`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckRef {
    /// Abort the build on the first disagreement (the historical behavior).
    Error,
    /// Drop the offending record and continue.
    Exclude,
}

impl std::str::FromStr for CheckRef {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "e" => Ok(CheckRef::Error),
            "x" => Ok(CheckRef::Exclude),
            other => Err(format!(
                "invalid check_ref {other:?}; expected \"e\" (error) or \"x\" (exclude)"
            )),
        }
    }
}

/// Per-record outcome of applying a [`CheckRef`] policy. `Exclude` carries the
/// originating [`NormalizeError`] so the caller can log the first offending
/// locus without re-running validation.
#[derive(Debug)]
pub enum RefDecision {
    /// REF matches the reference — process the record normally.
    Keep,
    /// REF disagrees under [`CheckRef::Exclude`] — skip this whole record.
    Exclude(NormalizeError),
}

/// Apply the [`CheckRef`] policy to one record's REF. Returns `Err` only under
/// [`CheckRef::Error`] (propagating `RefMismatch` / `RefOutOfContig`); under
/// [`CheckRef::Exclude`] a disagreement (either kind) becomes
/// `Ok(RefDecision::Exclude(_))`. A matching REF is always `Ok(RefDecision::Keep)`.
pub fn apply_check_ref(
    mode: CheckRef,
    pos: u32,
    ref_allele: &[u8],
    ref_seq: &[u8],
) -> Result<RefDecision, NormalizeError> {
    match validate_ref(pos, ref_allele, ref_seq) {
        Ok(()) => Ok(RefDecision::Keep),
        Err(e) => match mode {
            CheckRef::Error => Err(e),
            CheckRef::Exclude => Ok(RefDecision::Exclude(e)),
        },
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-checkref-target && pixi run bash -lc 'cargo test --no-default-features --features conversion --lib apply_check_ref check_ref_from_str'`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-checkref-target
git add src/normalize.rs
git commit -m "feat: add CheckRef policy and apply_check_ref helper"
```

---

## Task 2: Thread `check_ref` through the single-source engine + `from_vcf`

This task wires the shared `ChunkAssembler` path end-to-end and exposes the kwarg on `from_vcf`. `process_chromosome` gains a param, so **all four** of its callers must pass it — the three not yet exposed to Python (`run_pgen`/`run_svar1`/`run_vcf_list` call sites) pass `CheckRef::Error` as a placeholder, flipped to the real mode in Tasks 3–4.

**Files:**
- Modify: `src/chunk_assembler.rs` (struct ~249-264, `new` ~266-303, `decompose_record` ~313-321, add getter)
- Modify: `src/orchestrator.rs` (`process_chromosome` sig ~120-133, `ChunkAssembler::new` call ~305-313, reader-thread tail ~321, `run_vcf_list`'s `process_chromosome` call ~633-649)
- Modify: `src/lib.rs` (`run_conversion_pipeline` pyo3 ~123-215; and the `process_chromosome` calls inside `run_pgen_conversion_pipeline` ~320 and `run_svar1_conversion_pipeline` ~910+ — pass `CheckRef::Error` placeholder)
- Modify: `python/genoray/_svar2.py` (`from_vcf` ~474-558, add `_validate_check_ref` helper)
- Create: `tests/test_check_ref_e2e.rs`
- Modify: `tests/test_svar2_from_vcf.py`

**Interfaces:**
- Consumes: `normalize::{CheckRef, RefDecision, apply_check_ref}` (Task 1).
- Produces:
  - `ChunkAssembler::new(..., skip_out_of_scope: bool, check_ref: CheckRef, fields: &[FieldSpec])` — **new `check_ref` param inserted before `fields`**.
  - `ChunkAssembler::ref_excluded(&self) -> u64`
  - `process_chromosome(..., skip_out_of_scope: bool, check_ref: CheckRef, processing_threads: usize, signatures: bool, fields: &[FieldSpec])` — **new `check_ref` param inserted after `skip_out_of_scope`**.
  - `run_conversion_pipeline(...)` pyo3 gains trailing `check_ref: String` param.
  - Python `SparseVar2.from_vcf(..., check_ref: Literal["e","x"] = "e")`.
  - `_validate_check_ref(check_ref: str) -> str` module helper in `_svar2.py`.

- [ ] **Step 1: Write the failing Rust e2e test** (`tests/test_check_ref_e2e.rs`)

```rust
mod common;

use common::{SynthRecord, build_bcf_with_index, build_fasta_with_index};
use genoray_core::normalize::CheckRef;
use genoray_core::orchestrator::{SourceSpec, process_chromosome};
use tempfile::TempDir;

// Three records; the middle one's REF ("G") will disagree with the FASTA
// (which carries "T" at that position), reproducing issue #116.
fn bcf_records() -> Vec<SynthRecord<'static>> {
    vec![
        SynthRecord { pos: 100, ref_allele: b"A", alts: vec![&b"C"[..]], gt: vec![1, 0, 0, 0] },
        SynthRecord { pos: 200, ref_allele: b"G", alts: vec![&b"GTTT"[..]], gt: vec![1, 1, 0, 0] },
        SynthRecord { pos: 300, ref_allele: b"T", alts: vec![&b"A"[..]], gt: vec![0, 0, 1, 0] },
    ]
}

// Same loci, but the FASTA truth at pos 200 is "T" (not the record's "G").
fn fasta_records() -> Vec<SynthRecord<'static>> {
    vec![
        SynthRecord { pos: 100, ref_allele: b"A", alts: vec![], gt: vec![] },
        SynthRecord { pos: 200, ref_allele: b"T", alts: vec![], gt: vec![] },
        SynthRecord { pos: 300, ref_allele: b"T", alts: vec![], gt: vec![] },
    ]
}

fn convert(out: &std::path::Path, check_ref: CheckRef) -> Result<u64, genoray_core::error::ConversionError> {
    let tmp = out.parent().unwrap();
    let bcf = tmp.join("in.bcf");
    let fasta = tmp.join("in.fa");
    let samples = ["S0", "S1"];
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &bcf_records());
    build_fasta_with_index(&fasta, "chr1", 1000, &fasta_records());
    let sample_refs: Vec<&str> = samples.to_vec();
    process_chromosome(
        SourceSpec::Vcf { vcf_path: bcf.to_str().unwrap().to_string(), htslib_threads: 1 },
        Some(fasta.to_str().unwrap()),
        "chr1",
        out.to_str().unwrap(),
        &sample_refs,
        25_000,
        2,
        8 * 1024 * 1024,
        false,     // skip_out_of_scope
        check_ref, // NEW
        1,         // processing_threads
        false,     // signatures
        &[],       // fields
    )
}

#[test]
fn ref_mismatch_errors_under_e() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    assert!(convert(&out, CheckRef::Error).is_err(), "check_ref=e must abort on a REF mismatch");
}

#[test]
fn ref_mismatch_excluded_under_x() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    // Succeeds; the two clean SNPs (pos 100, 300) are still written.
    convert(&out, CheckRef::Exclude).unwrap();
    assert!(out.join("chr1/var_key/snp/positions.bin").exists(), "clean SNPs still converted");
}
```

- [ ] **Step 2: Run it to verify it fails to compile**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-checkref-target && pixi run bash -lc 'cargo test --no-default-features --features conversion --test test_check_ref_e2e'`
Expected: FAIL — `process_chromosome` takes 12 args, not 13 (the `check_ref` arg does not exist yet).

- [ ] **Step 3: Add the `check_ref` field + counter + getter to `ChunkAssembler`**

In `src/chunk_assembler.rs`, add two fields to the `ChunkAssembler` struct (after `skip_out_of_scope: bool,`):

```rust
    check_ref: crate::normalize::CheckRef,
    ref_excluded: u64,
```

In `ChunkAssembler::new`, add a `check_ref` parameter **before** `fields`:

```rust
    pub fn new(
        source: Box<dyn RecordSource + Send>,
        num_samples: usize,
        ploidy: usize,
        fasta_path: Option<&str>,
        chrom: &str,
        skip_out_of_scope: bool,
        check_ref: crate::normalize::CheckRef,
        fields: &[FieldSpec],
    ) -> Result<Self, ConversionError> {
```

and set both new fields in the returned `Self { ... }` (after `skip_out_of_scope,`):

```rust
            check_ref,
            ref_excluded: 0,
```

Add the getter next to `dropped_out_of_scope`:

```rust
    /// Records excluded because their REF disagreed with the reference under
    /// `CheckRef::Exclude`. Valid after the read loop drains.
    pub fn ref_excluded(&self) -> u64 {
        self.ref_excluded
    }
```

Replace the `validate_ref` block in `decompose_record` (currently lines 317-321):

```rust
        // Fail fast only when a reference is available; without one we trust the
        // input is already normalized/left-aligned.
        if self.has_reference {
            match crate::normalize::apply_check_ref(
                self.check_ref,
                pos,
                &rec.reference,
                &self.ref_seq,
            )? {
                crate::normalize::RefDecision::Keep => {}
                crate::normalize::RefDecision::Exclude(e) => {
                    self.ref_excluded += 1;
                    if self.ref_excluded == 1 {
                        println!(
                            "Notice: check_ref=x excluding record(s) whose REF disagrees \
                             with the reference (first: {e}); further exclusions on this \
                             contig are counted, not printed."
                        );
                    }
                    return Ok(());
                }
            }
        }
```

- [ ] **Step 4: Thread `check_ref` through `process_chromosome` and its callers**

In `src/orchestrator.rs`, add the param to `process_chromosome` (after `skip_out_of_scope: bool,` ~line 129):

```rust
    skip_out_of_scope: bool,
    check_ref: crate::normalize::CheckRef,
    processing_threads: usize,
```

Pass it to `ChunkAssembler::new` (~line 305-313), inserting before `&fields_owned`:

```rust
                let mut reader = crate::chunk_assembler::ChunkAssembler::new(
                    src,
                    s_refs.len(),
                    ploidy,
                    fasta.as_deref(),
                    &chr,
                    skip_out_of_scope,
                    check_ref,
                    &fields_owned,
                )?;
```

`check_ref` is `Copy`, so capturing it in the reader-thread closure needs no clone; add `let check_ref = check_ref;` is unnecessary — it moves in as a `Copy` value. (If the borrow checker complains about the move into the `spawn` closure, add `let check_ref = check_ref;` just above the closure alongside the other `let` bindings at ~line 217.)

Log the per-contig excluded count: after the `while let Some(dense_chunk)` loop (right before `Ok(reader.dropped_out_of_scope() + ...)` at ~line 321):

```rust
                let ref_excluded = reader.ref_excluded();
                if ref_excluded > 0 {
                    println!(
                        "[{chr}] check_ref=x: excluded {ref_excluded} record(s) whose REF \
                         disagreed with the reference FASTA."
                    );
                }
                Ok(reader.dropped_out_of_scope() + vcf_list_dropped.load(Ordering::Relaxed))
```

Update `run_vcf_list`'s `process_chromosome` call (~line 633-649), inserting `crate::normalize::CheckRef::Error` after `skip_out_of_scope,` (placeholder; Task 4 replaces it):

```rust
            skip_out_of_scope,
            crate::normalize::CheckRef::Error,
            processing_threads,
```

- [ ] **Step 5: Thread `check_ref` through the pyo3 fns**

In `src/lib.rs`, `run_conversion_pipeline`:
- Add `check_ref="e".to_string()` to the end of the `#[pyo3(signature = (...))]` list (after `format_fields=Vec::new()`).
- Add the param `check_ref: String,` at the end of the fn args (after `format_fields`).
- Parse it right after the `parse_manifest` call (~line 145):

```rust
    let check_ref: crate::normalize::CheckRef =
        check_ref.parse().map_err(PyValueError::new_err)?;
```

- Pass `check_ref` in the `process_chromosome` call (~line 199-215), inserting after `skip_out_of_scope,` (line 211). `check_ref` is `Copy`; it can be captured by the `par_iter().map()` closure directly.

In `run_pgen_conversion_pipeline` (~line 320) and `run_svar1_conversion_pipeline` (~line 910+), insert `crate::normalize::CheckRef::Error` after `skip_out_of_scope,` in their `process_chromosome` calls (placeholders; Task 3 replaces them). Do **not** add a pyo3 param to these two yet.

- [ ] **Step 6: Run the Rust e2e test to verify it passes**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-checkref-target && pixi run bash -lc 'cargo test --no-default-features --features conversion --test test_check_ref_e2e'`
Expected: PASS (2 tests). Also confirm the whole crate still builds: `pixi run bash -lc 'cargo test --no-default-features --features conversion --test test_convert_skip_e2e'` → PASS (unchanged).

- [ ] **Step 7: Add the Python `check_ref` kwarg + helper + docstring to `from_vcf`**

In `python/genoray/_svar2.py`, add a module-level helper (near the other module-level helpers, e.g. above the `SparseVar2` class):

```python
def _validate_check_ref(check_ref: str) -> str:
    """Validate a `check_ref` mode string. Returns it unchanged on success."""
    if check_ref not in ("e", "x"):
        raise ValueError(
            f'check_ref must be "e" (error) or "x" (exclude), got {check_ref!r}'
        )
    return check_ref
```

Ensure `Literal` is imported (`from typing import Literal` — add if absent).

Add the kwarg to `from_vcf`'s signature (after `format_fields=...`):

```python
        format_fields: Sequence[str | FormatField] | None = None,
        check_ref: Literal["e", "x"] = "e",
```

Add to the docstring (after the `info_fields, format_fields:` paragraph):

```
        check_ref: policy for a record whose REF disagrees with the reference
        FASTA (ignored when `no_reference=True`). `"e"` (default) raises and
        aborts the build — matching `bcftools norm --check-ref e`. `"x"` drops
        the offending record (including a REF that runs past the contig end)
        and continues, logging a per-contig count. Comparison is
        case-insensitive, so soft-masked (lowercase) reference bases match.
```

Validate and pass it in the `_core.run_conversion_pipeline(...)` call (append as the last argument):

```python
        _validate_check_ref(check_ref)
        return _core.run_conversion_pipeline(
            str(source),
            reference_path,
            contigs,
            str(out),
            samples,
            chunk_size,
            ploidy,
            threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            info,
            format_,
            check_ref,
        )
```

- [ ] **Step 8: Write the failing Python behavior test**

`tests/test_svar2_from_vcf.py` already defines `_write_ref(d)` (writes `_REF = "ACAGTACATGGGTAC..."` as `chr1` + `.fai`). In that reference, 1-based position 10 is `G`. Add a bad-REF VCF writer and three tests to the file:

```python
def _write_vcf_bad_ref(d: Path) -> Path:
    # Clean records at pos 3 (REF=A) and 7 (REF=C) match _REF; the record at
    # pos 10 declares REF=A but _REF[10] is 'G' — a REF/FASTA disagreement
    # (issue #116). Rows stay position-sorted for `bcftools index`.
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t10\t.\tA\tT\t.\t.\t.\tGT\t0|1\t0|0\n"  # REF=A, but _REF[10]='G'
    )
    plain = d / "bad.vcf"
    plain.write_text(body)
    gz = d / "bad.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_check_ref_error_aborts(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf_bad_ref(tmp_path)
    with pytest.raises(Exception):
        SparseVar2.from_vcf(tmp_path / "store", vcf, ref, check_ref="e", threads=1)


def test_from_vcf_check_ref_exclude_continues(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf_bad_ref(tmp_path)
    out = tmp_path / "store"
    SparseVar2.from_vcf(out, vcf, ref, check_ref="x", threads=1)
    assert (out / "meta.json").exists()  # completed despite the bad record
    sv = SparseVar2(out)
    # The two clean records survive; the pos-10 mismatch is excluded.
    counts = sv.region_counts("chr1", [(0, 40)])
    assert int(counts.sum()) == 2


def test_from_vcf_check_ref_invalid_value_raises(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf_bad_ref(tmp_path)
    with pytest.raises(ValueError, match="check_ref"):
        SparseVar2.from_vcf(tmp_path / "s", vcf, ref, check_ref="z", threads=1)  # type: ignore[arg-type]
```

(`region_counts` is used elsewhere in this file — confirm the exact signature there if the count assertion needs adjusting.)

- [ ] **Step 9: Run the Python test to verify it fails then passes**

Run (before building): `pixi run pytest tests/test_svar2_from_vcf.py -k check_ref -v` → FAIL (kwarg not yet in the installed extension).
Rebuild the extension so the new pyo3 signature is importable: `pixi run bash -lc 'maturin develop --features conversion'` (foreground; do not background — long builds that get backgrounded flake).
Run again: `pixi run pytest tests/test_svar2_from_vcf.py -k check_ref -v` → PASS (3 tests).

- [ ] **Step 10: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-checkref-target
git add src/chunk_assembler.rs src/orchestrator.rs src/lib.rs python/genoray/_svar2.py tests/test_check_ref_e2e.rs tests/test_svar2_from_vcf.py tests/conftest.py
git commit -m "feat: check_ref option for from_vcf (single-source engine)"
```

---

## Task 3: Expose `check_ref` on `from_pgen` and `from_svar1`

The engine already honors `check_ref` (Task 2); this flips the two placeholder `CheckRef::Error` calls to real parsed modes and adds the Python kwargs. PGEN/SVAR1 use the same `ChunkAssembler`, so behavior is already proven by the Rust e2e — these tests are lighter (threading + validation smoke).

**Files:**
- Modify: `src/lib.rs` (`run_pgen_conversion_pipeline` ~257-343, `run_svar1_conversion_pipeline` ~832-960)
- Modify: `python/genoray/_svar2.py` (`from_pgen` ~560-671, `from_svar1` ~841-930)

**Interfaces:**
- Consumes: `process_chromosome`'s `check_ref` param (Task 2), `_validate_check_ref` (Task 2).
- Produces: `run_pgen_conversion_pipeline`/`run_svar1_conversion_pipeline` pyo3 fns gain a trailing `check_ref: String`; Python `from_pgen`/`from_svar1` gain `check_ref: Literal["e","x"] = "e"`.

- [ ] **Step 1: Thread `check_ref` through `run_pgen_conversion_pipeline`**

In `src/lib.rs`:
- Append `check_ref` to the `#[pyo3(signature = (...))]` of `run_pgen_conversion_pipeline` (it has no defaults — add bare `check_ref` at the end of the tuple).
- Add `check_ref: String,` as the last fn arg (after `pgen_readers`).
- Parse it near the top (after the length checks, ~line 282):

```rust
    let check_ref: crate::normalize::CheckRef =
        check_ref.parse().map_err(PyValueError::new_err)?;
```

- Replace the placeholder `crate::normalize::CheckRef::Error` in its `process_chromosome` call with `check_ref`.

- [ ] **Step 2: Thread `check_ref` through `run_svar1_conversion_pipeline`**

Same three edits in `run_svar1_conversion_pipeline`: append `check_ref` to the signature tuple, add `check_ref: String,` (after `format_src_dtypes`), parse it after `parse_manifest` (~line 879), and replace its placeholder `CheckRef::Error` with `check_ref`.

- [ ] **Step 3: Add the Python kwargs + docstrings**

In `python/genoray/_svar2.py`, add `check_ref: Literal["e", "x"] = "e"` to both `from_pgen` and `from_svar1` signatures (as the last keyword arg), add the same docstring paragraph as `from_vcf` (Task 2 Step 7), call `_validate_check_ref(check_ref)` before the `_core` call, and append `check_ref` as the last positional argument to `_core.run_pgen_conversion_pipeline(...)` and `_core.run_svar1_conversion_pipeline(...)` respectively.

- [ ] **Step 4: Rebuild and write smoke tests**

Rebuild: `pixi run bash -lc 'maturin develop --features conversion'` (foreground).

The `check_ref` behavior for these methods runs through the same `ChunkAssembler` already proven by the Rust e2e (Task 2) and the `from_vcf` Python test — so these are **smoke tests** for the threading + validation, not full mismatch fixtures (constructing a PGEN/SVAR1 whose stored REF disagrees with a FASTA is not worth the fixture cost here). Two per method:

1. an existing clean PGEN/SVAR1 conversion test in the file, re-run with `check_ref="x"` added, still succeeds (asserts `(out / "meta.json").exists()`);
2. the same call with `check_ref="z"` raises `ValueError` matching `"check_ref"`.

Concretely: open `tests/test_svar2_from_pgen.py`, find the simplest existing passing conversion test (one that builds a PGEN + reference and calls `SparseVar2.from_pgen`), and copy its fixture setup into two new tests named `test_from_pgen_check_ref_accepts_x` and `test_from_pgen_check_ref_invalid_raises`, adding the `check_ref=` kwarg as above. Do the same in `tests/test_svar2_from_svar1.py` for `from_svar1` (`test_from_svar1_check_ref_accepts_x` / `..._invalid_raises`). Reuse the file's own fixture/helper functions verbatim — do not invent new ones.

- [ ] **Step 5: Run the tests**

Run: `pixi run pytest tests/test_svar2_from_pgen.py tests/test_svar2_from_svar1.py -k check_ref -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-checkref-target
git add src/lib.rs python/genoray/_svar2.py tests/test_svar2_from_pgen.py tests/test_svar2_from_svar1.py
git commit -m "feat: check_ref option for from_pgen and from_svar1"
```

---

## Task 4: `check_ref` for `from_vcf_list` (k-way merge path)

The merge validates each per-file record in `FileCursor::advance` (`vcf_list_reader.rs`). This threads `check_ref` there, counts exclusions per cursor, exposes the total via the existing proxy pattern, and wires the mode from Python.

**Files:**
- Modify: `src/vcf_list_reader.rs` (`FileCursor` struct ~68-79, `advance` ~88-186, `VcfListRecordSource` struct ~228-237 + `new` ~245-316 + add `ref_excluded()` + call site ~508-514)
- Modify: `src/orchestrator.rs` (`VcfListDroppedProxy` ~90-116, `VcfListRecordSource::new` call ~264-273, reader-thread proxy wiring ~229/274-278, `run_vcf_list` sig ~576-590 + `process_chromosome` call ~633-649)
- Modify: `src/lib.rs` (`run_vcf_list_conversion_pipeline` ~787-822)
- Modify: `python/genoray/_svar2.py` (`from_vcf_list` ~673-838)
- Modify: `tests/test_check_ref_e2e.rs` (add a vcf_list variant)
- Modify: `tests/test_svar2_from_vcf_list.py`

**Interfaces:**
- Consumes: `normalize::{CheckRef, RefDecision, apply_check_ref}`, `run_vcf_list`, `process_chromosome`.
- Produces:
  - `FileCursor` gains `ref_excluded: u64`.
  - `FileCursor::advance(..., skip_out_of_scope: bool, check_ref: CheckRef, info_specs, format_specs)` — new `check_ref` param.
  - `VcfListRecordSource::new(..., skip_out_of_scope: bool, check_ref: CheckRef, fields)` — new `check_ref` param; stored as a field; `ref_excluded(&self) -> u64` getter.
  - `run_vcf_list(..., skip_out_of_scope: bool, check_ref: CheckRef, signatures, ...)` — new param.
  - `run_vcf_list_conversion_pipeline` pyo3 gains trailing `check_ref: String`.
  - Python `from_vcf_list(..., check_ref: Literal["e","x"] = "e")`.

- [ ] **Step 1: Write the failing Rust e2e vcf_list variant**

Append to `tests/test_check_ref_e2e.rs`:

```rust
use common::build_bcf_with_index as build_single;

// Two single-sample files; file B carries a REF at pos 200 that disagrees with
// the FASTA (which says "T"). Under x, B's bad record is dropped; the merge
// still produces a store.
#[test]
fn vcf_list_ref_mismatch_excluded_under_x() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    let fasta = tmp.path().join("in.fa");
    build_fasta_with_index(&fasta, "chr1", 1000, &fasta_records());

    let a = tmp.path().join("a.bcf");
    let b = tmp.path().join("b.bcf");
    build_single(&a, "chr1", 1000,
        &["A"],
        &[SynthRecord { pos: 100, ref_allele: b"A", alts: vec![&b"C"[..]], gt: vec![1, 0] }]);
    build_single(&b, "chr1", 1000,
        &["B"],
        &[SynthRecord { pos: 200, ref_allele: b"G", alts: vec![&b"GTTT"[..]], gt: vec![1, 0] }]);

    let dropped = genoray_core::orchestrator::run_vcf_list(
        &[a.to_str().unwrap().to_string(), b.to_str().unwrap().to_string()],
        Some(fasta.to_str().unwrap()),
        &["chr1".to_string()],
        out.to_str().unwrap(),
        &["A".to_string(), "B".to_string()],
        25_000,
        2,
        Some(1),
        8 * 1024 * 1024,
        false,               // skip_out_of_scope
        CheckRef::Exclude,   // NEW
        false,               // signatures
        Vec::new(),          // info_fields
        Vec::new(),          // format_fields
    )
    .unwrap();
    assert_eq!(dropped, 0, "no out-of-scope ALTs; the REF-mismatch is excluded, not counted here");
    assert!(out.join("meta.json").exists(), "merge completed despite the bad record");
}
```

(Note the single-sample builder call: `build_bcf_with_index` accepts a `samples` slice of length 1 for these files.)

- [ ] **Step 2: Run it to verify it fails to compile**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-checkref-target && pixi run bash -lc 'cargo test --no-default-features --features conversion --test test_check_ref_e2e'`
Expected: FAIL — `run_vcf_list` takes 13 args, not 14.

- [ ] **Step 3: Make `FileCursor` check_ref-aware**

In `src/vcf_list_reader.rs`:
- Add `ref_excluded: u64,` to the `FileCursor` struct (after `dropped: u64,`).
- Initialize `ref_excluded: 0,` at both `FileCursor { ... }` construction sites in `VcfListRecordSource::new` (the `Ok(vcf)` arm and the `ContigNotInHeader` arm).
- Add `check_ref: crate::normalize::CheckRef,` to `advance`'s params (after `skip_out_of_scope: bool,`).
- Replace the `validate_ref` block (currently lines 114-116) with:

```rust
                    if let Some(rs) = ref_seq {
                        match crate::normalize::apply_check_ref(check_ref, rec.pos, &rec.reference, rs)? {
                            crate::normalize::RefDecision::Keep => {}
                            crate::normalize::RefDecision::Exclude(e) => {
                                self.ref_excluded += 1;
                                if self.ref_excluded == 1 {
                                    println!(
                                        "Notice: check_ref=x excluding record(s) in {} \
                                         whose REF disagrees with the reference (first: {e}).",
                                        self.path
                                    );
                                }
                                // This record contributes no atoms; fall through to
                                // the buffer pop, exactly like an all-`*`/dropped ALT
                                // record — the caller re-advances.
                                return Ok(self.buf.pop_front());
                            }
                        }
                    }
```

- [ ] **Step 4: Make `VcfListRecordSource` check_ref-aware**

In `src/vcf_list_reader.rs`:
- Add `check_ref: crate::normalize::CheckRef,` to the `VcfListRecordSource` struct (after `skip_out_of_scope: bool,`).
- Add `check_ref: crate::normalize::CheckRef,` to `new`'s params (after `skip_out_of_scope: bool,`) and set `check_ref,` in the returned `Self { ... }`.
- Add the getter after `dropped_out_of_scope`:

```rust
    /// Total REF-mismatch records excluded across every file so far
    /// (`CheckRef::Exclude`). Valid after the read loop drains.
    pub fn ref_excluded(&self) -> u64 {
        self.cursors.iter().map(|c| c.ref_excluded).sum()
    }
```

- Pass `self.check_ref` in the `advance` call (~line 508-514), after `self.skip_out_of_scope,`:

```rust
                    if let Some(atom) = self.cursors[i].advance(
                        ref_seq,
                        self.ploidy,
                        self.skip_out_of_scope,
                        self.check_ref,
                        &self.info_specs,
                        &self.format_specs,
                    )? {
```

- [ ] **Step 5: Wire the proxy + `process_chromosome` in `orchestrator.rs`**

In `src/orchestrator.rs`:
- Add a field to `VcfListDroppedProxy`: `ref_excluded_out: Arc<AtomicU64>,` (next to `dropped_out`).
- In its `read_next_chunk` (or wherever the EOF `store` happens, ~line 110-113), mirror the dropped store on the EOF transition:

```rust
        if rec.is_none() {
            self.dropped_out.store(self.inner.dropped_out_of_scope(), Ordering::Relaxed);
            self.ref_excluded_out.store(self.inner.ref_excluded(), Ordering::Relaxed);
        }
```

- Add `check_ref: crate::normalize::CheckRef,` to `process_chromosome`'s params — **already added in Task 2**; no change here.
- In the `SourceSpec::VcfList` arm of the reader thread (~256-278): create a second atomic alongside `vcf_list_dropped`:

```rust
                let vcf_list_dropped = Arc::new(AtomicU64::new(0));
                let vcf_list_ref_excluded = Arc::new(AtomicU64::new(0));
```

  Pass `check_ref` to `VcfListRecordSource::new` (after `skip_out_of_scope,`), and set `ref_excluded_out: Arc::clone(&vcf_list_ref_excluded),` in the `VcfListDroppedProxy { ... }` literal.

- Extend the per-contig log (added in Task 2 at ~line 321) so it also reports the vcf_list count. Replace that block with:

```rust
                let ref_excluded = reader.ref_excluded() + vcf_list_ref_excluded.load(Ordering::Relaxed);
                if ref_excluded > 0 {
                    println!(
                        "[{chr}] check_ref=x: excluded {ref_excluded} record(s) whose REF \
                         disagreed with the reference FASTA."
                    );
                }
                Ok(reader.dropped_out_of_scope() + vcf_list_dropped.load(Ordering::Relaxed))
```

- In `run_vcf_list` (~576-590): add `check_ref: crate::normalize::CheckRef,` after `skip_out_of_scope: bool,`, and replace the placeholder `crate::normalize::CheckRef::Error` in its `process_chromosome` call (Task 2 Step 4) with `check_ref`.

- [ ] **Step 6: Wire the pyo3 fn**

In `src/lib.rs`, `run_vcf_list_conversion_pipeline`:
- Append `check_ref="e".to_string()` to the `#[pyo3(signature = (...))]`.
- Add `check_ref: String,` as the last fn arg.
- Parse it (before the `py.detach`): `let check_ref: crate::normalize::CheckRef = check_ref.parse().map_err(PyValueError::new_err)?;`
- Pass `check_ref` into the `orchestrator::run_vcf_list(...)` call, after `skip_out_of_scope,`.

- [ ] **Step 7: Run the Rust e2e test**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-checkref-target && pixi run bash -lc 'cargo test --no-default-features --features conversion --test test_check_ref_e2e'`
Expected: PASS (3 tests total). Also run the existing merge test to confirm no regression: `pixi run bash -lc 'cargo test --no-default-features --features conversion --test test_vcf_list_e2e'` → PASS.

- [ ] **Step 8: Add the Python kwarg + test**

In `python/genoray/_svar2.py`, add `check_ref: Literal["e", "x"] = "e"` to `from_vcf_list` (last kwarg), the same docstring paragraph, `_validate_check_ref(check_ref)` before the `_core` call, and append `check_ref` as the last positional arg to `_core.run_vcf_list_conversion_pipeline(...)` (after `format_,`, ~line 837).

Rebuild: `pixi run bash -lc 'maturin develop --features conversion'` (foreground).

`tests/test_svar2_from_vcf_list.py` already defines `_write_ref(d)` (same `_REF`, pos 10 = `G`) and `_ss(d, name, sample, rows)` (writes one bgzipped+indexed single-sample VCF). Append:

```python
def test_from_vcf_list_check_ref_error_aborts(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "A", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "B", "chr1\t10\t.\tA\tT\t.\t.\t.\tGT\t0|1\n")  # REF=A, _REF[10]='G'
    with pytest.raises(Exception):
        SparseVar2.from_vcf_list(tmp_path / "s", [a, b], ref, check_ref="e", threads=1)


def test_from_vcf_list_check_ref_exclude_continues(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "A", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "B", "chr1\t10\t.\tA\tT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "store"
    SparseVar2.from_vcf_list(out, [a, b], ref, check_ref="x", threads=1)
    assert (out / "meta.json").exists()  # merge completed; b's bad record excluded
    sv = SparseVar2(out)
    counts = sv.region_counts("chr1", [(0, 40)])
    assert int(counts.sum()) == 1  # only a's clean pos-3 record survives
```

- [ ] **Step 9: Run the Python test**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py -k check_ref -v`
Expected: PASS (2 tests).

- [ ] **Step 10: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-checkref-target
git add src/vcf_list_reader.rs src/orchestrator.rs src/lib.rs python/genoray/_svar2.py tests/test_check_ref_e2e.rs tests/test_svar2_from_vcf_list.py
git commit -m "feat: check_ref option for from_vcf_list (k-way merge)"
```

---

## Task 5: CLI `--check-ref` flag

**Files:**
- Modify: `python/genoray/_cli/__main__.py` (`write_svar2` ~54-142)

**Interfaces:**
- Consumes: `SparseVar2.from_vcf` / `from_pgen` `check_ref` kwarg (Tasks 2-3).
- Produces: `genoray write ... --check-ref {e,x}` (default `e`).

- [ ] **Step 1: Add the `--check-ref` parameter**

In `write_svar2`, add after `skip_symbolics_and_breakends` (keep it keyword-only, in the existing `*,` block):

```python
    check_ref: Annotated[
        Literal["e", "x"], Parameter(name="--check-ref")
    ] = "e",
```

Ensure `Literal` is imported at the top of the file.

Add to the docstring's Parameters:

```
    check_ref
        REF-vs-reference policy (ignored with ``--no-reference``). ``e``
        (default) aborts on the first REF/FASTA disagreement; ``x`` drops the
        offending record and continues. Mirrors ``bcftools norm --check-ref``.
```

- [ ] **Step 2: Pass it through both branches**

Add `check_ref=check_ref,` to both the `SparseVar2.from_pgen(...)` call (~line 117) and the `SparseVar2.from_vcf(...)` call (~line 129).

- [ ] **Step 3: Verify the CLI wires through**

Run: `pixi run bash -lc 'genoray write --help'` → shows `--check-ref`.
Run a smoke conversion on an existing test fixture with `--check-ref x` and confirm it exits 0 (use any VCF+FASTA fixture already under `tests/data/`):

```bash
pixi run bash -lc 'genoray write <fixture>.vcf.gz /tmp/cli_checkref_store --reference <fixture>.fa --check-ref x --overwrite && echo OK'
```
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add python/genoray/_cli/__main__.py
git commit -m "feat: --check-ref flag on genoray write"
```

---

## Task 6: Update `skills/genoray-api/SKILL.md`

**Files:**
- Modify: `skills/genoray-api/SKILL.md`

**Interfaces:**
- Consumes: the finished public API (Tasks 2-5).

- [ ] **Step 1: Find where the conversion methods are documented**

Run: `grep -n "from_vcf\|from_pgen\|from_svar1\|from_vcf_list\|skip_out_of_scope\|no_reference" skills/genoray-api/SKILL.md`

- [ ] **Step 2: Document `check_ref`**

For each of the four `from_*` methods documented in `SKILL.md`, add the `check_ref` keyword alongside the existing `no_reference` / `skip_out_of_scope` documentation, with this wording (adapt to the file's format):

```
- `check_ref` (`"e"` | `"x"`, default `"e"`): policy when a record's REF
  disagrees with the reference FASTA (ignored with `no_reference=True`).
  `"e"` raises and aborts (like `bcftools norm --check-ref e`); `"x"` drops the
  offending record — including a REF running past the contig end — and
  continues. Case-insensitive, so soft-masked reference bases match.
```

If `SKILL.md` documents these methods with a shared "reference handling" note rather than per-method, add the `check_ref` note once in that shared section and reference it from each method.

- [ ] **Step 3: Verify no stale claims**

Re-read the edited sections; confirm the return-value description still says the methods return the out-of-scope drop count (unchanged) and that nothing claims REF mismatches always abort.

- [ ] **Step 4: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs: document check_ref in genoray-api skill"
```

---

## Task 7: Full-suite verification

- [ ] **Step 1: Run the full Rust test suite**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-checkref-target && pixi run bash -lc 'cargo test --no-default-features --features conversion'`
Expected: all pass (including the pre-existing `test_convert_skip_e2e`, `test_vcf_list_e2e`, `test_left_align_e2e`, and `normalize` unit tests).

- [ ] **Step 2: Confirm the no-conversion query-core build still compiles**

Run: `pixi run bash -lc 'cargo check --no-default-features'`
Expected: clean (the `check_ref` additions live behind `#[cfg(feature = "conversion")]` in `lib.rs`; `normalize.rs` additions must not reference conversion-only symbols).

- [ ] **Step 3: Rebuild and run the full Python suite**

Run: `pixi run bash -lc 'maturin develop --features conversion'` (foreground) then `pixi run test`.
Expected: all pass. Confirm the default path is unchanged by spot-checking that existing from_vcf/from_pgen/from_vcf_list/from_svar1 tests (without `check_ref`) still pass.

- [ ] **Step 4: Lint**

Run: `pixi run bash -lc 'ruff check genoray tests && ruff format --check genoray tests'` and confirm prek hooks pass on a dry run: `pixi run bash -lc 'prek run --all-files'`.
Expected: clean.

- [ ] **Step 5: Push and open the draft PR**

```bash
git push -u origin worktree-check-ref-svar2
gh pr create --draft --title "feat: check_ref (e/x) for all SVAR2 conversion methods (#116)" --body "$(cat <<'EOF'
Closes #116.

Adds a `check_ref` policy to `from_vcf`, `from_pgen`, `from_svar1`, and
`from_vcf_list` (plus `genoray write --check-ref`):

- `"e"` (default) — abort on a REF/FASTA disagreement (current behavior).
- `"x"` — drop the offending record and continue, logging a per-contig count
  and the first offending locus.

One shared `apply_check_ref` helper in `normalize.rs` replaces the two
`validate_ref(...)?` call sites. Return types and default behavior are
unchanged (non-breaking). `skills/genoray-api/SKILL.md` updated.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** `CheckRef`/`apply_check_ref` (Task 1) ↔ spec "Architecture"; chunk_assembler + vcf_list_reader call-site replacement (Tasks 2, 4) ↔ spec "one shared decision point"; threading through 4 pyo3 fns + Python kwargs (Tasks 2-4) ↔ spec "Threading" + "Python surface"; CLI (Task 5) + SKILL.md (Task 6) ↔ spec "Python / CLI / docs surface"; per-contig count + first-locus logging (Tasks 2, 4) ↔ spec "Reporting" (grand total explicitly dropped, spec updated); `RefOutOfContig` handled under `x` (Task 1 tests + apply_check_ref) ↔ spec note; `w`/`s` absent ↔ spec "Out of scope".
- **Type consistency:** `check_ref` param is inserted **after `skip_out_of_scope`** in every Rust fn (`ChunkAssembler::new`, `process_chromosome`, `run_vcf_list`, `FileCursor::advance`, `VcfListRecordSource::new`) and **before `fields`** where a `fields`/`format_specs` trailing arg exists; the pyo3 boundary always takes `check_ref: String` as the **trailing** param and parses via `FromStr`. Getters are `ref_excluded()` on both `ChunkAssembler` and `VcfListRecordSource`. Python kwarg is `check_ref: Literal["e","x"] = "e"` on all four methods, validated by `_validate_check_ref`.
- **Non-breaking:** every new Rust param has a placeholder (`CheckRef::Error`) until its surface is wired; pyo3 signatures default `check_ref="e"`; Python kwargs default `"e"`. Existing tests pass unchanged (verified in Task 7).

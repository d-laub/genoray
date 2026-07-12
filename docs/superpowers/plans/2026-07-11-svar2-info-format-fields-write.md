# SVAR2 INFO/FORMAT field write — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `SparseVar2.from_vcf` extract scalar-numeric INFO (per-variant) and FORMAT (per-call) fields from a VCF/BCF and store them alongside the genotype streams in the SVAR2 store.

**Architecture:** Field values ride the existing conversion pipeline (reader → `DenseChunk` → `dense2sparse_vk` → `SparseChunk` → per-chunk write → per-contig merge), staged at a wide type (`i32` for int/flag, `f32` for float) and positionally aligned to the genotype records (Option A, genotype-aligned). A final **global** pass converts every field's staged `values.bin` to its resolved storage dtype (explicit, range-checked; or auto-narrowed losslessly for ints), records the resolved dtypes in `meta.json`, and is the single place that sees all values across contigs.

**Tech Stack:** Rust (pyo3, rust-htslib, rayon, bytemuck, memmap2), the `half` crate (new, for `f16`), Python (cyvcf2 for header introspection), pixi.

## Global Constraints

- Conventional Commits for every commit (`feat:`, `test:`, `refactor:`, `docs:`).
- Rust tests run via `pixi run test-rust` (= `cargo test --no-default-features --features conversion`). Never plain `cargo test` (pyo3 `extension-module` fails to link a test binary: `undefined symbol: _Py_Dealloc`).
- Python tests run via `pixi run test` (regenerates fixtures via `gen_from_vcf.sh` first) or `pixi run pytest tests/<file>` for a single file.
- Public-API rule (`CLAUDE.md`): any new name reachable from `import genoray` without an underscore prefix (here: `InfoField`, `FormatField`, new `from_vcf` kwargs) MUST be reflected in `skills/genoray-api/SKILL.md` in this same effort.
- Do **not** bump the version or edit `CHANGELOG.md`'s versioned sections; accumulate entries under `## Unreleased` only.
- Scope: scalar-numeric only — header `Number` ∈ {`0` (Flag, INFO-only), `1`, `A`}; header `Type` ∈ {`Integer`, `Float`, `Flag`}. Reject everything else at config time.
- Storage model is **genotype-aligned (Option A)**: FORMAT values exist only where the genotype has a call (var_key) or in a full per-sample dense column (dense, non-carrier slots = `default`). Non-carrier FORMAT values in var_key-routed variants are dropped by design.
- After all Rust changes, rebuild the extension so Python sees the new pyfunction signature: `pixi run develop` (or the repo's maturin-develop task — confirm via `grep -n develop pixi.toml`).

**Spec correction (record in the spec before starting):** spec §2 says auto-narrowing happens "at merge time." Merge is per-contig, but a field's on-disk dtype must be identical across all contigs (the reader keys dtype off the global `meta.json`). Auto-narrowing therefore moves to a **global finalize pass** after all contigs merge (Task 11). Update `docs/superpowers/specs/2026-07-11-svar2-info-format-fields-write-design.md` §2 accordingly as part of Task 1's commit.

---

## File structure

**Python**
- Create `python/genoray/_svar2_fields.py` — `FieldDtype`, `InfoField`, `FormatField`, and `_resolve_fields(vcf_path, info_fields, format_fields) -> list[tuple]` (header validation + dtype resolution). One responsibility: the public field-config surface + validation.
- Modify `python/genoray/_svar2.py` — `from_vcf` gains `info_fields`/`format_fields` kwargs, calls `_resolve_fields`, passes the manifest into `_core.run_conversion_pipeline`.
- Modify `python/genoray/__init__.py` — export `InfoField`, `FormatField`.

**Rust**
- Create `src/field.rs` — `FieldCategory`, `HtslibType`, `StorageDtype`, `FieldSpec` + `FieldSpec::from_ffi` and `parse_manifest`. One responsibility: field typing/validation core (no I/O).
- Create `src/field_finalize.rs` — the global staged→final conversion pass (`finalize_fields`).
- Modify `src/types.rs` — staged field columns on `DenseChunk`, `SparseSubStream`, `DenseSubChunk`.
- Modify `src/vcf_reader.rs` — extract declared fields per record into `DenseChunk`.
- Modify `src/rvk.rs` — route field values in `dense2sparse_vk`.
- Modify `src/cost_model.rs` — `info_bits`/`format_bits` in `choose_representation`.
- Modify `src/writer.rs` + `src/layout.rs` — per-chunk field file write + path helpers.
- Modify `src/merge.rs` + `src/dense_merge.rs` — thread field arrays through the finalized per-contig `values.bin`.
- Modify `src/orchestrator.rs` — thread `&[FieldSpec]` through `process_chromosome`, call the field merges, and run per-contig field staging.
- Modify `src/lib.rs` + `src/meta.rs` — FFI params, global finalize call, `fields` manifest in `meta.json`.
- Modify `Cargo.toml` — add `half`.

---

## Phase 1 — Python field-config API + validation

### Task 1: `InfoField`/`FormatField` config + header validation

**Files:**
- Create: `python/genoray/_svar2_fields.py`
- Test: `tests/test_svar2_fields.py`
- Modify (docs): `docs/superpowers/specs/2026-07-11-svar2-info-format-fields-write-design.md` (§2 finalize-pass correction)

**Interfaces:**
- Produces:
  - `FieldDtype = Literal["bool","i8","u8","i16","u16","i32","u32","f16","f32"]`
  - `@dataclass(frozen=True) class InfoField: name: str; dtype: FieldDtype | None = None; default: float | int | None = None`
  - `@dataclass(frozen=True) class FormatField: name: str; dtype: FieldDtype | None = None; default: float | int | None = None`
  - `_resolve_fields(vcf_path: str, info_fields, format_fields) -> list[tuple[str, str, str, str | None, float | None]]` — each tuple is `(name, category, htslib_type, dtype_or_none, default_or_none)` with `category ∈ {"info","format"}`, `htslib_type ∈ {"int","float","flag"}`. Raises `ValueError` on any invalid field.

- [ ] **Step 1: Write failing tests**

Use an existing fixture BCF (run `pixi run gen` once so `tests/data/*.bcf` exist; pick one with known INFO/FORMAT fields — inspect with `bcftools view -h`). If no fixture has a usable Float FORMAT + Integer INFO + Flag, add a tiny header to `tests/data/gen_from_vcf.sh` or build one in the test with `cyvcf2`. Assume `tests/data/biallelic.bcf` with `##FORMAT=<ID=DS,Number=1,Type=Float>`, `##INFO=<ID=AC,Number=1,Type=Integer>`, `##INFO=<ID=DB,Number=0,Type=Flag>` (adjust names to the real fixture).

```python
import pytest
from genoray._svar2_fields import InfoField, FormatField, _resolve_fields

BCF = "tests/data/biallelic.bcf"  # adjust to a real fixture with DS/AC/DB

def test_resolve_bare_str_infers():
    out = _resolve_fields(BCF, ["AC"], ["DS"])
    assert ("AC", "info", "int", None, None) in out
    assert ("DS", "format", "float", None, None) in out

def test_resolve_flag_info():
    out = _resolve_fields(BCF, ["DB"], [])
    assert ("DB", "info", "flag", None, None) in out

def test_resolve_dtype_and_default_override():
    out = _resolve_fields(BCF, [], [FormatField("DS", dtype="f16", default=0.0)])
    assert ("DS", "format", "float", "f16", 0.0) in out

def test_reject_unknown_field():
    with pytest.raises(ValueError, match="not found in the VCF header"):
        _resolve_fields(BCF, ["NOPE"], [])

def test_reject_float_stored_as_int():
    with pytest.raises(ValueError, match="incompatible"):
        _resolve_fields(BCF, [], [FormatField("DS", dtype="i16")])

def test_reject_nonscalar_number():
    # a Number=R/G/. field must be rejected; use a real one from the fixture,
    # else add ##FORMAT=<ID=AD,Number=R,Type=Integer> to the fixture.
    with pytest.raises(ValueError, match="Number"):
        _resolve_fields(BCF, [], [FormatField("AD")])
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/test_svar2_fields.py -v`
Expected: FAIL (`ModuleNotFoundError: genoray._svar2_fields`).

- [ ] **Step 3: Implement `_svar2_fields.py`**

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from cyvcf2 import VCF as _CyVCF

FieldDtype = Literal["bool", "i8", "u8", "i16", "u16", "i32", "u32", "f16", "f32"]

_INT_DTYPES = {"bool", "i8", "u8", "i16", "u16", "i32", "u32"}
_FLOAT_DTYPES = {"f16", "f32"}


@dataclass(frozen=True)
class InfoField:
    """A per-variant INFO field to store in the SVAR2 output.

    ``dtype=None`` infers storage from the header (Integer→lossless auto-narrow,
    Float→f32, Flag→bool). ``default`` is the value written for VCF-missing
    entries (else a reserved sentinel/NaN).
    """

    name: str
    dtype: FieldDtype | None = None
    default: float | int | None = None


@dataclass(frozen=True)
class FormatField:
    """A per-sample FORMAT field to store in the SVAR2 output. Genotype-aligned:
    only carrier calls (var_key) or a full per-sample dense column are stored;
    non-carrier values in var_key-routed variants are dropped. Same ``dtype``/
    ``default`` semantics as :class:`InfoField`.
    """

    name: str
    dtype: FieldDtype | None = None
    default: float | int | None = None


def _htslib_type(header_type: str) -> str:
    match header_type:
        case "Integer":
            return "int"
        case "Float":
            return "float"
        case "Flag":
            return "flag"
        case other:
            raise ValueError(f"field Type={other!r} is unsupported (need Integer/Float/Flag)")


def _check_dtype_compat(name: str, htype: str, dtype: str | None) -> None:
    if dtype is None:
        return
    if htype in ("int", "flag") and dtype not in _INT_DTYPES:
        raise ValueError(f"field {name!r} ({htype}) incompatible with dtype {dtype!r}")
    if htype == "float" and dtype not in _FLOAT_DTYPES:
        raise ValueError(f"field {name!r} (float) incompatible with dtype {dtype!r}")


def _resolve_one(vcf: _CyVCF, spec, category: str) -> tuple[str, str, str, str | None, float | None]:
    field = spec if not isinstance(spec, str) else None
    name = spec if isinstance(spec, str) else spec.name
    kind = "INFO" if category == "info" else "FORMAT"
    try:
        hdr = vcf.get_header_type(name)  # {'HeaderType','Type','Number',...}
    except KeyError:
        raise ValueError(f"field {name!r} not found in the VCF header") from None
    number = str(hdr.get("Number"))
    htype = _htslib_type(hdr["Type"])
    if htype == "flag":
        if category != "info":
            raise ValueError(f"Flag field {name!r} is INFO-only")
    elif number not in ("1", "A"):
        raise ValueError(
            f"field {name!r} has Number={number}; only scalar Number=1 or "
            "biallelic-split Number=A are supported"
        )
    dtype = None if field is None else field.dtype
    default = None if field is None else (None if field.default is None else float(field.default))
    _check_dtype_compat(name, htype, dtype)
    return (name, category, htype, dtype, default)


def _resolve_fields(
    vcf_path: str,
    info_fields: Sequence[str | InfoField] | None,
    format_fields: Sequence[str | FormatField] | None,
) -> list[tuple[str, str, str, str | None, float | None]]:
    vcf = _CyVCF(vcf_path)
    try:
        out: list[tuple[str, str, str, str | None, float | None]] = []
        for spec in info_fields or []:
            out.append(_resolve_one(vcf, spec, "info"))
        for spec in format_fields or []:
            out.append(_resolve_one(vcf, spec, "format"))
        return out
    finally:
        vcf.close()
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `pixi run pytest tests/test_svar2_fields.py -v`
Expected: PASS. If a fixture lacks a needed field, add its header line to `tests/data/gen_from_vcf.sh`, re-run `pixi run gen`, and adjust the test's field names.

- [ ] **Step 5: Correct the spec's finalize-pass wording**

Edit `docs/superpowers/specs/2026-07-11-svar2-info-format-fields-write-design.md` §2: replace "the merge observes the field's global `[min,max]` … and writes the smallest width" with "a **global finalize pass** (after all per-contig merges) observes the field's global `[min,max]` across all contigs and writes the smallest width; per-contig storage is uniform because dtype is global in `meta.json`."

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2_fields.py tests/test_svar2_fields.py docs/superpowers/specs/2026-07-11-svar2-info-format-fields-write-design.md
git commit -m "feat(svar2): InfoField/FormatField config + header validation"
```

---

## Phase 2 — Rust field-spec types + FFI plumbing

### Task 2: `src/field.rs` typing core

**Files:**
- Create: `src/field.rs`
- Modify: `src/lib.rs` (add `mod field;`)

**Interfaces:**
- Produces:
  - `enum FieldCategory { Info, Format }`
  - `enum HtslibType { Int, Float, Flag }`
  - `enum StorageDtype { Auto, Bool, I8, U8, I16, U16, I32, U32, F16, F32 }` with `fn width_bytes(self) -> Option<usize>` (`Auto → None`) and `fn stage_is_float(htype) -> bool`.
  - `struct FieldSpec { name: String, category: FieldCategory, htype: HtslibType, dtype: StorageDtype, default: Option<f64> }`
  - `fn parse_manifest(raw: Vec<(String,String,String,Option<String>,Option<f64>)>) -> Result<Vec<FieldSpec>, ConversionError>`
  - `impl FieldSpec { fn stage_is_float(&self) -> bool }` — `true` iff `htype == Float`.

- [ ] **Step 1: Write failing unit tests** (inline `#[cfg(test)]` in `src/field.rs`)

```rust
#[test]
fn parse_manifest_maps_tuples() {
    let raw = vec![
        ("DS".into(), "format".into(), "float".into(), Some("f16".into()), Some(0.0)),
        ("AC".into(), "info".into(), "int".into(), None, None),
    ];
    let specs = parse_manifest(raw).unwrap();
    assert_eq!(specs.len(), 2);
    assert!(matches!(specs[0].category, FieldCategory::Format));
    assert!(matches!(specs[0].dtype, StorageDtype::F16));
    assert!(specs[0].stage_is_float());
    assert!(matches!(specs[1].dtype, StorageDtype::Auto));
    assert!(!specs[1].stage_is_float());
}

#[test]
fn parse_manifest_rejects_bad_category() {
    let raw = vec![("X".into(), "bogus".into(), "int".into(), None, None)];
    assert!(parse_manifest(raw).is_err());
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion field::tests'`
Expected: FAIL (module/functions missing).

- [ ] **Step 3: Implement `src/field.rs`**

```rust
use crate::error::ConversionError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldCategory { Info, Format }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtslibType { Int, Float, Flag }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDtype { Auto, Bool, I8, U8, I16, U16, I32, U32, F16, F32 }

impl StorageDtype {
    /// Final on-disk width in bytes; `Auto` is undecided until finalize.
    pub fn width_bytes(self) -> Option<usize> {
        Some(match self {
            StorageDtype::Auto => return None,
            StorageDtype::Bool | StorageDtype::I8 | StorageDtype::U8 => 1,
            StorageDtype::I16 | StorageDtype::U16 | StorageDtype::F16 => 2,
            StorageDtype::I32 | StorageDtype::U32 | StorageDtype::F32 => 4,
        })
    }
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bool" => Self::Bool, "i8" => Self::I8, "u8" => Self::U8,
            "i16" => Self::I16, "u16" => Self::U16, "i32" => Self::I32,
            "u32" => Self::U32, "f16" => Self::F16, "f32" => Self::F32,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub name: String,
    pub category: FieldCategory,
    pub htype: HtslibType,
    pub dtype: StorageDtype,
    pub default: Option<f64>,
}

impl FieldSpec {
    pub fn stage_is_float(&self) -> bool { self.htype == HtslibType::Float }
}

fn bad(ctx: &str) -> ConversionError {
    ConversionError::Config { message: ctx.to_string() } // add this variant to error.rs if absent
}

pub fn parse_manifest(
    raw: Vec<(String, String, String, Option<String>, Option<f64>)>,
) -> Result<Vec<FieldSpec>, ConversionError> {
    raw.into_iter()
        .map(|(name, category, htype, dtype, default)| {
            let category = match category.as_str() {
                "info" => FieldCategory::Info,
                "format" => FieldCategory::Format,
                other => return Err(bad(&format!("bad field category {other:?}"))),
            };
            let htype = match htype.as_str() {
                "int" => HtslibType::Int,
                "float" => HtslibType::Float,
                "flag" => HtslibType::Flag,
                other => return Err(bad(&format!("bad htslib type {other:?}"))),
            };
            let dtype = match dtype {
                None => StorageDtype::Auto,
                Some(s) => StorageDtype::parse(&s)
                    .ok_or_else(|| bad(&format!("bad storage dtype {s:?}")))?,
            };
            Ok(FieldSpec { name, category, htype, dtype, default })
        })
        .collect()
}
```

Add a `Config { message: String }` variant to `ConversionError` in `src/error.rs` if one does not already exist (mirror the existing variant style; check `src/error.rs` first and reuse an existing catch-all if present).

- [ ] **Step 4: Register module + run tests**

Add `mod field;` near the other `mod` declarations in `src/lib.rs`. Run:
`pixi run bash -lc 'cargo test --no-default-features --features conversion field::tests'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/field.rs src/lib.rs src/error.rs
git commit -m "feat(svar2): field-spec typing core (Rust)"
```

### Task 3: FFI + `process_chromosome` signature threading (no storage yet)

**Files:**
- Modify: `src/lib.rs:93-181`, `src/orchestrator.rs:44-58`
- Test: `tests/test_e2e.rs` (existing tests must still pass — they call `process_chromosome` directly)

**Interfaces:**
- Consumes: `field::parse_manifest`.
- Produces: `run_conversion_pipeline(..., info_fields, format_fields)` and `process_chromosome(..., fields: &[FieldSpec])`.

- [ ] **Step 1: Extend the pyfunction signature** (`src/lib.rs`)

Add two params before/after `signatures` and extend the `#[pyo3(signature=...)]` defaults:

```rust
#[pyo3(signature = (vcf_path, reference_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608, skip_out_of_scope=false, signatures=false, info_fields=Vec::new(), format_fields=Vec::new()))]
fn run_conversion_pipeline(
    py: Python,
    // …existing params…
    signatures: bool,
    info_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
    format_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
) -> PyResult<usize> {
```

Near the top of the body, build the manifest (info then format; category already encoded in the tuples from Python, so just concatenate):

```rust
let mut raw = info_fields;
raw.extend(format_fields);
let fields = crate::field::parse_manifest(raw)
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
```

Pass `&fields` into each `process_chromosome` call (add as the final arg at `src/lib.rs:177`).

- [ ] **Step 2: Extend `process_chromosome`** (`src/orchestrator.rs`)

Add `fields: &[crate::field::FieldSpec]` as the final parameter. Do not use it yet (add `let _ = fields;` to silence the unused warning, removed in Task 5). This keeps the compile green while later tasks fill it in.

- [ ] **Step 3: Build + run the existing Rust suite**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion'`
Expected: PASS (existing e2e tests unchanged — they pass `&[]` for fields; update their `process_chromosome` call sites to add the new `&[]` arg).

- [ ] **Step 4: Rebuild the extension + smoke-test Python**

Run: `pixi run develop` then
`pixi run python -c "import genoray._core as c; print('ok')"`
Expected: prints `ok` (new pyfunction signature loads).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/orchestrator.rs tests/test_e2e.rs
git commit -m "feat(svar2): thread field manifest through FFI + process_chromosome"
```

---

## Phase 3 — Reader extraction

### Task 4: Stage field columns on `DenseChunk` + extract in the reader

**Files:**
- Modify: `src/types.rs:98-111` (`DenseChunk`), `src/vcf_reader.rs` (`read_next_chunk` + `VcfChunkReader::new`)
- Test: `tests/test_e2e.rs` (extend the synthetic-BCF builder to include a Float FORMAT + Integer INFO field, assert extraction)

**Interfaces:**
- Produces on `DenseChunk`: `pub info_staged: Vec<Vec<StagedVal>>` (one inner vec per INFO field, `len == v`), `pub format_staged: Vec<Vec<StagedVal>>` (one per FORMAT field, `len == v * num_samples`, sample-major). `StagedVal` is a plain `f32`-or-`i32` union staged as `enum StagedVal { I(i32), F(f32) }`? — no: to keep the hot path monomorphic, store each staged column as `StagedColumn { Int(Vec<i32>), Float(Vec<f32>) }`. So `DenseChunk.info_staged: Vec<StagedColumn>` and `.format_staged: Vec<StagedColumn>`.

Define in `src/types.rs`:
```rust
pub enum StagedColumn { Int(Vec<i32>), Float(Vec<f32>) }
impl StagedColumn {
    pub fn with_capacity(is_float: bool, n: usize) -> Self {
        if is_float { StagedColumn::Float(Vec::with_capacity(n)) }
        else { StagedColumn::Int(Vec::with_capacity(n)) }
    }
    pub fn push_f64(&mut self, v: f64) {
        match self { StagedColumn::Int(x) => x.push(v as i32), StagedColumn::Float(x) => x.push(v as f32) }
    }
}
```
Staging sentinels: missing → `default` if set, else `i32::MIN` (int) / `f32::NAN` (float). Flag stored in an `Int` column as `0`/`1`.

- [ ] **Step 1: Write failing extraction test** (`tests/test_e2e.rs`)

Extend the synthetic BCF writer to add `##INFO=<ID=AC,Number=1,Type=Integer>` and `##FORMAT=<ID=DS,Number=1,Type=Float>`, populate known values on a few records (incl. one missing), build `FieldSpec`s for them, and call `VcfChunkReader::read_next_chunk`. Assert:
```rust
// after reading the one chunk:
let ds = match &chunk.format_staged[0] { StagedColumn::Float(v) => v, _ => panic!() };
assert_eq!(ds.len(), v_variants * num_samples);
assert!((ds[/* variant 0, sample 1 */] - 0.5).abs() < 1e-6);
let ac = match &chunk.info_staged[0] { StagedColumn::Int(v) => v, _ => panic!() };
assert_eq!(ac[/* variant 2 */], i32::MIN); // missing → sentinel (no default set)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion test_e2e'`
Expected: FAIL (fields not populated / new struct fields absent).

- [ ] **Step 3: Add the struct fields + reader extraction**

- Add `info_staged`/`format_staged` to `DenseChunk` (`src/types.rs`) and initialize them empty in `read_next_chunk` when no fields are requested.
- Store `fields: Vec<FieldSpec>` (or split info/format index lists) on `VcfChunkReader` (`new` gains a `fields: &[FieldSpec]` param; `process_chromosome` passes its `fields` down where it constructs the reader at `src/orchestrator.rs:151`).
- In `read_next_chunk`, for each buffered record (the metadata pass at `src/vcf_reader.rs:426`), for each INFO field pull the scalar via rust-htslib (`record.info(name).integer()?` / `.float()?` / flag presence), applying biallelic-split allele index for `Number=A`; push into the per-field `StagedColumn` (one per variant). For each FORMAT field pull `record.format(name).integer()?`/`.float()?` → `(n_samples, k)` slice; push the per-sample scalar for every sample (sample-major). Missing → `default`/sentinel.

Reference rust-htslib access patterns already used for GT in `src/vcf_reader.rs` (the existing genotype decode shows the `record.format(...)` idiom). Keep the extraction in the same sequential metadata pass so ordering matches `pos`/`ilens`.

- [ ] **Step 4: Run to confirm pass**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion test_e2e'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/types.rs src/vcf_reader.rs src/orchestrator.rs tests/test_e2e.rs
git commit -m "feat(svar2): extract INFO/FORMAT fields into DenseChunk"
```

---

## Phase 4 — Routing

### Task 5: Route field values in `dense2sparse_vk`

**Files:**
- Modify: `src/types.rs` (`SparseSubStream`, `DenseSubChunk`), `src/rvk.rs:138-274`
- Test: inline `#[cfg(test)]` in `src/rvk.rs`

**Interfaces:**
- Produces on `SparseSubStream`: `pub field_calls: Vec<StagedColumn>` (one per field, pushed in lockstep with `push_call`). On `DenseSubChunk`: `pub field_info: Vec<StagedColumn>` (one per INFO field, `len == n_dense_variants`) and `pub field_format: Vec<StagedColumn>` (one per FORMAT field, `len == n_dense_variants * num_samples`, per-sample column).
- Consumes: `DenseChunk.info_staged`/`format_staged`, `field::FieldSpec` list.
- `dense2sparse_vk` gains a `fields: &[FieldSpec]` param.

- [ ] **Step 1: Write failing routing test** (`src/rvk.rs` tests)

Build a tiny `DenseChunk` with 2 variants (force one var_key via 1 carrier, one dense via many carriers by choosing `n_samples`), one INFO field and one FORMAT field staged with known values, run `dense2sparse_vk(&chunk, &mut bank, false, &fields)`, and assert:
```rust
// var_key variant: field value duplicated per call, aligned to call order
let vk = sc.streams.get(StreamTag::VarKeySnp);
match &vk.field_calls[0] { StagedColumn::Float(v) => assert_eq!(v.len(), vk.call_positions.len()), _=>panic!() }
// dense variant: INFO one-per-variant, FORMAT full per-sample column
let d = sc.dense.get(DenseClass::Snp);
match &d.field_info[0] { StagedColumn::Int(v) => assert_eq!(v.len(), d.n_dense_variants), _=>panic!() }
match &d.field_format[0] { StagedColumn::Float(v) => assert_eq!(v.len(), d.n_dense_variants * num_samples), _=>panic!() }
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion rvk::'`
Expected: FAIL.

- [ ] **Step 3: Implement routing**

- Add the new vecs to `SparseSubStream`/`DenseSubChunk` (`src/types.rs`), sized from the field list in `with_capacity`/`empty` (pass field metadata or size lazily on first push).
- In the **pre-pass** (`src/rvk.rs:160-207`), when a variant routes to `Dense`, push each INFO field's `info_staged[v]` once into `sub.field_info[i]`, and push the full per-sample slice `format_staged[i][v*num_samples .. (v+1)*num_samples]` into `sub.field_format[i]` (non-carrier slots carry whatever the reader stored — for genotype-aligned semantics, overwrite non-carrier samples with the field `default`; a sample is a carrier iff any of its ploidy bits are set in `chunk.genos` plane `v`).
- In the **transpose loop** (`src/rvk.rs:224-268`), immediately after `push_call` for a var_key call, push the per-call field value: for INFO use `info_staged[i][v]`; for FORMAT use `format_staged[i][v*num_samples + s]`. Keep a parallel `Vec<StagedColumn>` on the stream (`field_calls`).

Show the exact insert (var_key call site):
```rust
streams.get_mut(tag).push_call(pos, &key_le[..spec.key_bytes]);
let st = streams.get_mut(tag);
for (i, f) in fields.iter().enumerate() {
    let val = match f.category {
        FieldCategory::Info   => staged_f64(&chunk.info_staged[info_ix[i]], v),
        FieldCategory::Format => staged_f64(&chunk.format_staged[fmt_ix[i]], v * num_samples + s),
    };
    st.field_calls[i].push_f64(val);
}
```
(`info_ix`/`fmt_ix` map the flat field index to its per-category staged column; `staged_f64` reads an `i32`/`f32` back as `f64`.)

- [ ] **Step 4: Run to confirm pass**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion rvk::'`
Expected: PASS.

- [ ] **Step 5: Wire the call site + full suite**

Update `executor::run_compute_engine` and its `dense2sparse_vk` call (`src/executor.rs:33`) to pass `&fields` (thread `fields: &[FieldSpec]` from `process_chromosome` → `run_compute_engine`). Run:
`pixi run bash -lc 'cargo test --no-default-features --features conversion'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/types.rs src/rvk.rs src/executor.rs src/orchestrator.rs
git commit -m "feat(svar2): route INFO/FORMAT field values in dense2sparse_vk"
```

---

## Phase 5 — Cost model

### Task 6: `info_bits`/`format_bits` in `choose_representation`

**Files:**
- Modify: `src/cost_model.rs:51-67`, `src/rvk.rs:181-189` (pass the new terms)
- Test: inline `#[cfg(test)]` in `src/cost_model.rs`

**Interfaces:**
- Produces: `choose_representation(class, n_samples, ploidy, x_calls, sidecar_bits, info_bits, format_bits) -> Representation` where `info_bits`/`format_bits` are the summed per-record widths (in bits) of the active INFO/FORMAT fields. Formula:
  `var_key = x·(POS+key+sidecar_bits) + info_bits·x + format_bits·x`;
  `dense = POS+key+np+sidecar_bits + info_bits + format_bits·n_samples`.

- [ ] **Step 1: Write failing tests** (extend `src/cost_model.rs` tests)

```rust
#[test]
fn info_bits_shift_toward_dense() {
    // With an INFO field, dense amortizes (once) while var_key pays per call.
    let base = choose_representation(Class::Snp, 100, 2, 30, 0, 0, 0);
    let with_info = choose_representation(Class::Snp, 100, 2, 30, 0, 32, 0);
    // a case that is VarKey without info but Dense with it
    assert_eq!(base, Representation::VarKey);
    assert_eq!(with_info, Representation::Dense);
}

#[test]
fn format_bits_do_not_cancel() {
    // FORMAT is width·x (var_key) vs width·n_samples (dense): reinforces dense
    // for high carrier counts.
    let r = choose_representation(Class::Snp, 100, 2, 150, 0, 0, 16);
    assert_eq!(r, Representation::Dense);
}
```
(Pick constants so the assertions hold given `POS_BITS=32`, `key_bits(Snp)=2`, `np=n_samples*ploidy`.)

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion cost_model::'`
Expected: FAIL (arity mismatch).

- [ ] **Step 3: Implement**

```rust
#[inline]
pub fn choose_representation(
    class: Class, n_samples: usize, ploidy: usize, x_calls: usize,
    sidecar_bits: u64, info_bits: u64, format_bits: u64,
) -> Representation {
    let np = (n_samples as u64) * (ploidy as u64);
    let x = x_calls as u64;
    let per_call = POS_BITS + key_bits(class) + sidecar_bits;
    let var_key_bits = x * (per_call + info_bits + format_bits);
    let dense_bits = POS_BITS + key_bits(class) + np + sidecar_bits
        + info_bits + format_bits * (n_samples as u64);
    if dense_bits < var_key_bits { Representation::Dense } else { Representation::VarKey }
}
```

- [ ] **Step 4: Thread the new args from `dense2sparse_vk`**

In `src/rvk.rs` pre-pass, compute `info_bits`/`format_bits` once from `fields` (sum of `f.dtype.width_bytes().unwrap_or(4) * 8` per category; `Auto → 4 bytes` staging estimate) and pass into `choose_representation` at line 189. Update all other `choose_representation` call sites (grep) to pass `0, 0` when no fields.

- [ ] **Step 5: Run tests to confirm pass**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion cost_model:: rvk::'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cost_model.rs src/rvk.rs
git commit -m "feat(svar2): factor INFO/FORMAT field bits into cost model"
```

---

## Phase 6 — Persist: per-chunk write + per-contig merge

### Task 7: Per-chunk field file write

**Files:**
- Modify: `src/layout.rs` (path helpers), `src/writer.rs:15-51`
- Test: inline test in `src/writer.rs` (write a `SparseChunk` with field arrays, assert files exist with expected byte lengths)

**Interfaces:**
- Produces layout helpers: `pub fn chunk_field(dir: &Path, chunk_id: usize, field_ix: usize) -> PathBuf` → `dir.join(format!("chunk_{}_field{}.bin", chunk_id, field_ix))`; analogous `dense` variant if the dense sub-dir differs.
- Writer writes each `field_calls[i]` (var_key) and `field_info[i]`/`field_format[i]` (dense) as raw staged bytes (`bytemuck::cast_slice` for `i32`/`f32`).

- [ ] **Step 1: Write failing test** — construct a `SparseChunk` with one var_key stream carrying a 3-element `StagedColumn::Float`, call `run_io_writer` (or a smaller extracted `write_stream_fields` helper), assert `chunk_0_field0.bin` is 12 bytes.
- [ ] **Step 2: Run** `pixi run bash -lc 'cargo test --no-default-features --features conversion writer::'` → FAIL.
- [ ] **Step 3: Implement** the path helpers + write loop additions in `run_io_writer` (parallel to the existing `chunk_key`/`chunk_geno` writes at `src/writer.rs:20-46`). Use `write_bin` (`src/writer.rs:87`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(svar2): write per-chunk field files`.

### Task 8: Merge field arrays into per-contig staged `values.bin`

**Files:**
- Modify: `src/merge.rs:25-266` (var_key), `src/dense_merge.rs` (dense), `src/orchestrator.rs:267-302` (call sites), `src/layout.rs` (final `values.bin` path)
- Test: `tests/test_e2e.rs` — after a full `process_chromosome` with fields, assert each `{contig}/fields/{name}/{sub}/values.bin` exists with the staged dtype width and length matching the sub-stream record count.

**Interfaces:**
- Produces: finalized per-contig staged files `{contig}/fields/{name}/var_key_snp/values.bin` etc. (staged `i32`/`f32`; dtype not yet narrowed).
- `merge_mini_sc` gains a `field_widths: &[usize]` (staged bytes per field) + reads each chunk's `chunk_{id}_field{ix}.bin` alongside pos/key, gathering into a third buffer and `write_all_at` into a finalized `values.bin` per field (mirror the pos/key gather at `src/merge.rs:147-255`).

- [ ] **Step 1: Write failing e2e assertion** (extend the Task 4 e2e test): after conversion, `std::fs::metadata(contig_dir.join("fields/DS/var_key_snp/values.bin"))` exists and its len == `n_var_key_snp_calls * 4` (staged f32).
- [ ] **Step 2: Run** `test_e2e` → FAIL.
- [ ] **Step 3: Implement** — var_key: extend `merge_mini_sc` to gather+write field values parallel to `alleles.bin` (each field its own `values.bin` under a per-field sub-dir; use the ledger's per-column offsets identically). Dense: in `dense_merge::merge_dense_class`, concatenate `field_info` (per-variant) and `field_format` (per-sample column) across chunks in variant order into the dense `values.bin`. Wire the field list into both merge calls in `orchestrator.rs`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(svar2): merge field values into per-contig staged values.bin`.

---

## Phase 7 — Finalize, manifest, Python wiring, docs

### Task 9: Global finalize pass (staged → resolved dtype)

**Files:**
- Create: `src/field_finalize.rs`
- Modify: `Cargo.toml` (add `half = "2"`), `src/lib.rs` (call after all contigs; collect resolved dtypes)
- Test: inline test in `src/field_finalize.rs`

**Interfaces:**
- Produces: `fn finalize_fields(output_dir: &Path, contigs: &[String], fields: &[FieldSpec]) -> Result<Vec<ResolvedField>, ConversionError>` where `ResolvedField { name, category, dtype: StorageDtype (concrete), default }`. For each field: scan every contig's staged `values.bin` for global `[min,max]`/missing; resolve `Auto` → smallest lossless int width (reserving a sentinel iff missing observed) or keep `f32`; for explicit dtypes range-check and error on overflow (`f16` overflow > 65504, int overflow, nonzero-fraction float→int); rewrite each staged file to the resolved width (map staged sentinel → resolved sentinel or `default`). `f16` via the `half::f16` crate.

- [ ] **Step 1: Write failing test** — write two fake per-contig staged `i32` `values.bin` (one contig max 200, other max 10), run `finalize_fields` with an `Auto` int field, assert resolved dtype is `U8`/`I16` per the sentinel rule and the rewritten bytes decode back to the originals.
- [ ] **Step 2: Run** `pixi run bash -lc 'cargo test --no-default-features --features conversion field_finalize::'` → FAIL.
- [ ] **Step 3: Implement** `finalize_fields` + add `half` to `Cargo.toml`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(svar2): global field finalize + lossless int auto-narrow`.

### Task 10: `meta.json` `fields` manifest

**Files:**
- Modify: `src/meta.rs` (`write_meta` gains `fields: &[ResolvedField]`), `src/lib.rs:193-200` (pass finalize output)
- Test: `src/meta.rs` unit test (round-trip the JSON, assert the `fields` array)

- [ ] **Step 1: Write failing test** — `write_meta` with one `ResolvedField`, read back the JSON, assert `fields[0] == {"name":"DS","category":"format","dtype":"f32","default":0.0}`.
- [ ] **Step 2: Run** `pixi run bash -lc 'cargo test --no-default-features --features conversion meta::'` → FAIL.
- [ ] **Step 3: Implement** — extend the `json!` value in `write_meta` (`src/meta.rs`) with a `"fields"` array; serialize `StorageDtype` to its lowercase string (`"f32"` etc.) and `category` to `"info"`/`"format"`. In `src/lib.rs`, call `finalize_fields` before `write_meta` and pass its result.
- [ ] **Step 4: Run** → PASS. Then full suite `pixi run bash -lc 'cargo test --no-default-features --features conversion'` → PASS.
- [ ] **Step 5: Commit** `feat(svar2): record field manifest in meta.json`.

### Task 11: Wire `from_vcf` end-to-end + Python round-trip test

**Files:**
- Modify: `python/genoray/_svar2.py:59-128`, `python/genoray/__init__.py`
- Test: `tests/test_svar2_fields.py` (add an integration test)

**Interfaces:**
- Consumes: `_resolve_fields`, `_core.run_conversion_pipeline(..., info_fields, format_fields)`.
- Produces: `SparseVar2.from_vcf(..., info_fields=None, format_fields=None)`.

- [ ] **Step 1: Write failing integration test**

```python
import json, numpy as np
from pathlib import Path
from genoray import SparseVar2, FormatField

def test_from_vcf_writes_dosage_field(tmp_path):
    out = tmp_path / "store.svar2"
    SparseVar2.from_vcf(out, "tests/data/biallelic.bcf", reference="tests/data/ref.fa",
                        format_fields=[FormatField("DS", default=0.0)])
    meta = json.loads((out / "meta.json").read_text())
    ds = next(f for f in meta["fields"] if f["name"] == "DS")
    assert ds["category"] == "format" and ds["dtype"] == "f32"
    # read-path is spec #2; here just confirm the raw staged/finalized file exists
    contig = meta["contigs"][0]
    vals = list((out / contig / "fields" / "DS").glob("*/values.bin"))
    assert vals and all(p.stat().st_size % 4 == 0 for p in vals)
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_svar2_fields.py::test_from_vcf_writes_dosage_field -v` → FAIL.
- [ ] **Step 3: Implement** — add `info_fields`/`format_fields` kwargs to `from_vcf`; after building `contigs`, call `flds = _resolve_fields(str(source), info_fields, format_fields)`, split into `info=[t for t in flds if t[1]=="info"]` / `format=[...]`, and pass as the last two args to `_core.run_conversion_pipeline`. Export `InfoField`/`FormatField` in `__init__.py` (`__all__` + lazy map + `TYPE_CHECKING` re-export, mirroring `SparseVar2` at `__init__.py:35,22,53-61`).
- [ ] **Step 4: Run** `pixi run develop` then the test → PASS. Run `pixi run test` (full suite) → PASS.
- [ ] **Step 5: Commit** `feat(svar2): from_vcf info_fields/format_fields kwargs`.

### Task 12: Docs + changelog

**Files:**
- Modify: `skills/genoray-api/SKILL.md`, `CHANGELOG.md`, SVAR2 roadmap/data-model docs (`docs/roadmap/*`)

- [ ] **Step 1:** Add `InfoField`/`FormatField` and the `from_vcf(info_fields=, format_fields=)` kwargs to `skills/genoray-api/SKILL.md` (document scalar-numeric scope, dtype auto-narrow, `default`, and the genotype-aligned non-carrier-drop caveat).
- [ ] **Step 2:** Add a `## Unreleased` `feat:` entry to `CHANGELOG.md`.
- [ ] **Step 3:** Update the SVAR2 on-disk-layout doc with the `fields/{name}/{sub}/values.bin` tree + `meta.json` `fields` schema.
- [ ] **Step 4: Commit** `docs(svar2): document INFO/FORMAT field write API + layout`.

---

## Self-review notes (for the executor)

- **Read-path is out of scope** (spec #2): every test here inspects `meta.json` and raw `values.bin` byte sizes, never a decode API. Do not add reader methods.
- **Staging vs final dtype:** everything upstream of Task 9 operates at staged width (`i32`/`f32`); only `finalize_fields` produces the resolved on-disk dtype. Keep the two straight — chunk/merge files are staged, `meta.json` records resolved.
- **Type-name consistency:** `StagedColumn` (types.rs), `FieldSpec`/`StorageDtype`/`FieldCategory` (field.rs), `ResolvedField` (field_finalize.rs) must match verbatim across tasks.
- **`signatures` composition:** both `signatures=True` and fields add cost-model bits and per-record data. When both are on, `choose_representation` receives `sidecar_bits` AND `info_bits`/`format_bits` — verify the e2e path with both enabled once (add an assertion to the Task 11 test if time permits).

# SVAR2 Conversion Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry arbitrary auxiliary fields through X→SVAR2 conversion — PGEN dosages (from same or separate `.pgen`), SVAR1 field selection — and split the `write` CLI into per-source subcommands that expose them.

**Architecture:** PGEN dosages are stored as ordinary FORMAT fields, reusing the entire existing VCF FORMAT machinery (`parse_manifest` → `resolve_format` → var_key/dense routing → `finalize_fields`) unchanged. The only new Rust code is (a) threading a non-empty `Vec<FieldSpec>` + dosage reader pools into the PGEN pipeline, and (b) `PgenRecordSource::refill` reading dosages via pgenlib and emitting `FormatVals::Dense`. SVAR1 selection filters the existing field manifest before it reaches Rust. The CLI split is pure argument plumbing over the existing `from_*` classmethods.

**Tech Stack:** Rust (pyo3, rayon), Python 3.10–3.14, cyclopts (CLI), pgenlib, cyvcf2, polars, maturin, pytest.

## Global Constraints

- **Conventional Commits** for every commit (`feat:`, `test:`, `docs:`, etc.). This whole feature is a **non-breaking `feat:`** — no 4.0 bump (the SVAR2 write CLI has no production consumers).
- **Rust tests must run with `--no-default-features`** (else the pyo3 test binary fails to link: `undefined symbol: _Py_Dealloc`): `pixi run bash -lc 'cargo test --no-default-features ...'`.
- **`pixi run test` does NOT rebuild the Rust `.so`.** Before ANY Python-level e2e/perf verification of Rust changes, run `pixi run bash -lc 'maturin develop --release'` (release `.so` ≈4MB; debug ≈79MB).
- **NFS `target/` breaks cargo linking** (bus error). Export `CARGO_TARGET_DIR=/tmp/genoray-target-$USER` for `cargo test` AND for commits (pre-commit runs `cargo check`/`clippy`). Do NOT park build artifacts on NFS.
- **`cargo test-rust <arg>` filters by TEST NAME, not file** — a nonmatching arg vacuously passes 0 tests. Use `--test <file>`.
- **Field 5-tuple layout** (Python→Rust, everywhere): `(name: str, category: "info"|"format", htype: "int"|"float"|"flag", dtype: str|None, default: float|None)`. A per-sample dosage field is `(name, "format", "float", "f16"|"f32", default)`.
- **Coordinate convention:** 0-based half-open `[start, end)`. Missing genotype `-1`; missing dosage `np.nan`.
- **Public API rule:** any change to a name reachable from `import genoray` without an underscore — including CLI surface — MUST update `skills/genoray-api/SKILL.md` in the same PR (Task 7).
- **Package root:** `python/genoray/`. Rust: `src/`. Work in the worktree `/carter/users/dlaub/projects/genoray/.claude/worktrees/svar2-conversion-fields`.

---

## File Structure

- `python/genoray/_svar2_fields.py` — **modify**: add `DosageField` dataclass, `_dosage_field_to_tuple`, `_parse_cli_field_specs` (bcftools-string → info/format lists).
- `python/genoray/__init__.py` — **modify**: export `DosageField` (`__all__`, `_LAZY`, `TYPE_CHECKING`).
- `python/genoray/_svar2.py` — **modify**: `from_pgen(dosages=...)` plumbing + dosage-reader construction/validation; `from_svar1(fields=...)` selection.
- `src/lib.rs` — **modify**: `run_pgen_conversion_pipeline` grows `dosage_fields` + `dosage_readers` params; replace hardcoded empty `fields` (line 327) with `parse_manifest(dosage_fields)`.
- `src/orchestrator.rs` — **modify**: `SourceSpec::Pgen` carries per-shard dosage readers; thread through `process_chromosome`.
- `src/pgen_reader.rs` — **modify**: `PgenRecordSource` reads dosages in `refill`, emits `FormatVals::Dense` in `next_record`.
- `python/genoray/_cli/__main__.py` — **modify**: replace `@write.default` auto-detect with `write vcf` / `write pgen` / `write svar1` subcommands; move legacy VCF/PGEN→SVAR1 writer to top-level `write-svar1`.
- `skills/genoray-api/SKILL.md` — **modify**: document `DosageField`, `from_pgen(dosages=)`, `from_svar1(fields=)`, new CLI surface.
- Tests: `tests/test_svar2_fields.py` (new, unit), `tests/test_svar2_pgen_dosages.py` (new, e2e), `tests/test_svar2.py` or existing svar1 test (from_svar1 selection), `tests/test_cli.py` (CLI).

---

## Task 1: `DosageField` spec + CLI field-string parser (pure Python)

**Files:**
- Modify: `python/genoray/_svar2_fields.py`
- Modify: `python/genoray/__init__.py`
- Test: `tests/test_svar2_fields.py` (create)

**Interfaces:**
- Produces:
  - `DosageField(name: str = "dosage", source: str | Path = "self", dtype: Literal["f16","f32"] = "f32", default: float | None = None)` — frozen dataclass, exported as `genoray.DosageField`.
  - `_dosage_field_to_tuple(df: DosageField) -> tuple[str, str, str, str, float | None]` → `(df.name, "format", "float", df.dtype, df.default)`.
  - `_parse_cli_field_specs(specs: Sequence[str]) -> tuple[list[str], list[str]]` → `(info_names, format_names)`, parsing bcftools-style `"INFO/x"` / `"FORMAT/x"` / `"FMT/x"`, de-duplicating (first occurrence wins, across both lists).

- [ ] **Step 1: Write failing tests**

Create `tests/test_svar2_fields.py`:

```python
import pytest

from genoray import DosageField
from genoray._svar2_fields import _dosage_field_to_tuple, _parse_cli_field_specs


def test_dosage_field_defaults():
    df = DosageField()
    assert df.name == "dosage"
    assert df.source == "self"
    assert df.dtype == "f32"
    assert df.default is None


def test_dosage_field_to_tuple():
    df = DosageField(name="VAF", source="vaf.pgen", dtype="f16", default=0.0)
    assert _dosage_field_to_tuple(df) == ("VAF", "format", "float", "f16", 0.0)


def test_dosage_field_rejects_non_float_dtype():
    with pytest.raises(ValueError, match="dtype"):
        _dosage_field_to_tuple(DosageField(name="x", dtype="i8"))  # type: ignore[arg-type]


def test_parse_cli_field_specs_splits_and_dedups():
    info, fmt = _parse_cli_field_specs(
        ["INFO/AF", "FORMAT/AD", "FMT/AD", "INFO/AF", "INFO/AC"]
    )
    assert info == ["AF", "AC"]
    assert fmt == ["AD"]


def test_parse_cli_field_specs_rejects_bad_prefix():
    with pytest.raises(ValueError, match="INFO/ or FORMAT/"):
        _parse_cli_field_specs(["AF"])
    with pytest.raises(ValueError, match="INFO/ or FORMAT/"):
        _parse_cli_field_specs(["GT/x"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_svar2_fields.py -v`
Expected: FAIL — `ImportError: cannot import name 'DosageField'`.

- [ ] **Step 3: Add `DosageField` + helpers to `_svar2_fields.py`**

At the top of `python/genoray/_svar2_fields.py`, ensure `from pathlib import Path` is imported (add if absent). After the `FormatField` dataclass (around line 42), add:

```python
@dataclass(frozen=True)
class DosageField:
    """A per-sample dosage track to store in the SVAR2 output as a FORMAT field.

    Dosages come from a PLINK2 ``.pgen`` — either the hardcall file itself
    (``source="self"``) or a separate ``.pgen`` (e.g. VAF/CCF stored as dosage,
    kept separate because pgenlib derives hard calls from dosage when present).
    Stored genotype-aligned like any FORMAT field: under var_key routing a
    non-carrier's dosage is dropped (fine for VAF/CCF, ~0 for non-carriers).
    """

    name: str = "dosage"
    source: str | Path = "self"
    dtype: Literal["f16", "f32"] = "f32"
    default: float | None = None


def _dosage_field_to_tuple(
    df: DosageField,
) -> tuple[str, str, str, str, float | None]:
    if df.dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"DosageField {df.name!r} dtype must be 'f16' or 'f32', got {df.dtype!r}"
        )
    return (df.name, "format", "float", df.dtype, df.default)


def _parse_cli_field_specs(specs: Sequence[str]) -> tuple[list[str], list[str]]:
    """Parse bcftools-style ``INFO/x`` / ``FORMAT/x`` / ``FMT/x`` field strings.

    Returns ``(info_names, format_names)``, de-duplicated with first occurrence
    winning across both categories.
    """
    info: list[str] = []
    fmt: list[str] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        prefix, sep, name = spec.partition("/")
        if not sep or not name:
            raise ValueError(
                f"field {spec!r} must be prefixed with INFO/ or FORMAT/ (or FMT/)"
            )
        upper = prefix.upper()
        if upper == "INFO":
            key = ("info", name)
            bucket = info
        elif upper in ("FORMAT", "FMT"):
            key = ("format", name)
            bucket = fmt
        else:
            raise ValueError(
                f"field {spec!r} must be prefixed with INFO/ or FORMAT/ (or FMT/)"
            )
        if key not in seen:
            seen.add(key)
            bucket.append(name)
    return info, fmt
```

Confirm `Literal` and `Sequence` are already imported at the top of the file (they are, per lines 3–5). `_FLOAT_DTYPES` already exists (line 13).

- [ ] **Step 4: Export `DosageField` from the package**

In `python/genoray/__init__.py`:
- Add `"DosageField",` to `__all__` (after `"FormatField",`, line 24).
- Add to `_LAZY`: `"DosageField": ("genoray._svar2_fields", "DosageField"),` (after the `FormatField` entry, line 39).
- Add under `TYPE_CHECKING` (after line 64): `from ._svar2_fields import DosageField as DosageField`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_svar2_fields.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2_fields.py python/genoray/__init__.py tests/test_svar2_fields.py
git commit -m "feat(svar2): add DosageField spec and CLI field-string parser"
```

---

## Task 2: Rust — thread dosage fields + reader pools into the PGEN pipeline

This wires the new parameters through the Rust pipeline WITHOUT reading dosages yet (`PgenRecordSource` still emits empty FORMAT). Because Python (unchanged until Task 3) passes empty dosage args, behavior is identical and existing tests stay green — so this task is independently reviewable and testable via `cargo test`.

**Files:**
- Modify: `src/lib.rs` (`run_pgen_conversion_pipeline`, ~296–427)
- Modify: `src/orchestrator.rs` (`SourceSpec::Pgen` ~61–82; `process_chromosome` ~254–268; construction sites ~515–593)
- Modify: `src/pgen_reader.rs` (`PgenRecordSource` struct ~30–68; `new` ~82–104) — add fields, default-empty, no read logic yet

**Interfaces:**
- Consumes (from Python, Task 3 will supply): `dosage_fields: Vec<(String, String, String, Option<String>, Option<f64>)>` (the same 5-tuple `parse_manifest` consumes) and `dosage_readers: Vec<Vec<Vec<Py<PyAny>>>>` indexed `[contig][shard][dosage_field]`.
- Produces: a `run_pgen_conversion_pipeline` whose `fields` vec is `parse_manifest(dosage_fields)` and whose per-shard `SourceSpec::Pgen` carries `dosage_readers: Vec<Py<PyAny>>` (one per dosage field, in field order).

- [ ] **Step 1: Add the two pyo3 params to `run_pgen_conversion_pipeline`**

In `src/lib.rs`, in the `run_pgen_conversion_pipeline` signature (ends at `sample_perm: Vec<usize>,`), add two params (place `dosage_fields` right after `signatures: bool,` to mirror the VCF pipeline's field placement, and `dosage_readers` right after `readers: Vec<Vec<Py<PyAny>>>,`):

```rust
    dosage_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
    // ... existing params up to and including `readers: Vec<Vec<Py<PyAny>>>,`
    dosage_readers: Vec<Vec<Vec<pyo3::Py<pyo3::PyAny>>>>,
```

Update the `#[pyo3(signature = (...))]` attribute (if present on this fn) to include both new names in the same positions.

- [ ] **Step 2: Replace the hardcoded empty `fields`**

Replace line 327 (`let fields: Vec<crate::field::FieldSpec> = Vec::new();` and its comment at 326) with:

```rust
    let fields = crate::field::parse_manifest(dosage_fields)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
```

- [ ] **Step 3: Validate and thread `dosage_readers` into jobs**

Near the existing length check (`chroms.len() != readers.len()`, ~317), also require `dosage_readers.len() == chroms.len()` and, for each contig `i`, `dosage_readers[i].len() == readers[i].len()` (same shard count) and every `dosage_readers[i][s].len() == fields.len()`. Raise `PyValueError` on mismatch. Then extend the `jobs` zip (~341–347) so each job carries its contig's `Vec<Vec<Py<PyAny>>>` dosage readers alongside the hardcall `Vec<Py<PyAny>>`.

- [ ] **Step 4: Carry dosage readers in `SourceSpec::Pgen` and into the record source**

In `src/orchestrator.rs`, add a field to the `SourceSpec::Pgen` variant (~61–82):

```rust
        dosage_readers: Vec<pyo3::Py<pyo3::PyAny>>,  // one per dosage FieldSpec, in order
```

At the two construction sites (single-reader ~515–529 and sharded ~578–593), pass the per-shard dosage readers (from the job). Thread them into `PgenRecordSource::new(...)`.

In `src/pgen_reader.rs`, add to the `PgenRecordSource` struct (~30–68):

```rust
    dosage_readers: Vec<pyo3::Py<pyo3::PyAny>>,
    dosage_bufs: Vec<pyo3::Py<numpy::PyArray2<f32>>>,   // shape (batch, num_samples), one per dosage field
    host_dosages: Vec<Vec<f32>>,                        // per field, filled in refill
```

In `new` (~82–104), accept `dosage_readers`, allocate one `(batch, num_samples)` `f32` buffer per reader (mirror the `buf` allocation at ~85–87), and initialize `host_dosages` as `vec![Vec::new(); dosage_readers.len()]`. **Do not read dosages yet** — `next_record` still emits `FormatVals::Dense(Vec::new())`.

- [ ] **Step 5: Build with default-features off and run Rust tests**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$USER
pixi run bash -lc 'cargo build --no-default-features'
pixi run bash -lc 'cargo test --no-default-features'
```
Expected: builds; existing tests PASS (no behavior change — Python still passes empty vecs, verified in Task 3).

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/orchestrator.rs src/pgen_reader.rs
git commit -m "feat(svar2): thread dosage field specs and reader pools into PGEN pipeline"
```

---

## Task 3: Rust dosage read + Python `from_pgen(dosages=)` (end-to-end)

This is the first task with a real dosage deliverable: `PgenRecordSource` reads dosages and emits them, and `from_pgen` builds the readers/specs and calls the pipeline. Tested via a Python e2e round-trip after `maturin develop`.

**Files:**
- Modify: `src/pgen_reader.rs` (`refill` ~108–143; `next_record` ~210–233)
- Modify: `python/genoray/_svar2.py` (`from_pgen` ~768–995)
- Test: `tests/test_svar2_pgen_dosages.py` (create)

**Interfaces:**
- Consumes: `DosageField`, `_dosage_field_to_tuple` (Task 1); the Rust pipeline params (Task 2).
- Produces: `from_pgen(..., dosages: Sequence[DosageField] | None = None)`. Stored dosages are readable via `SparseVar2(path, fields=[name]).decode(...)` — one `Ragged` per dosage field.

- [ ] **Step 1: Write the failing e2e test**

Create `tests/test_svar2_pgen_dosages.py`. Use an existing PGEN fixture that carries a dosage track (search `tests/` for `.pgen` fixtures and how `from_pgen` tests build them — reuse the same fixture/reference the existing `test_svar2` PGEN tests use). Structure:

```python
import numpy as np
import pytest

from genoray import DosageField, SparseVar2


def test_from_pgen_self_dosages_round_trip(tmp_path, pgen_with_dosages, reference_fasta):
    # pgen_with_dosages: fixture path to a .pgen carrying a dosage track,
    # with its .pvar/.psam siblings. reference_fasta: matching FASTA.
    out = tmp_path / "out.svar2"
    SparseVar2.from_pgen(
        out, pgen_with_dosages, reference_fasta,
        dosages=[DosageField(name="DS", source="self")],
    )
    sv = SparseVar2(out, fields=["DS"])
    assert "DS" in sv.available_fields
    res = sv.decode(sv.contigs[0], 0, 2**30, samples=list(sv.available_samples))
    # carrier dosages match the source PGEN's dosages for carrier calls
    # (assert against genoray.PGEN(pgen_with_dosages).read(...) dosages)
    ds = res.DS  # Ragged of float32
    assert ds is not None


def test_from_pgen_separate_dosage_file(tmp_path, hardcall_pgen, dosage_pgen, reference_fasta):
    out = tmp_path / "out.svar2"
    SparseVar2.from_pgen(
        out, hardcall_pgen, reference_fasta,
        dosages=[DosageField(name="VAF", source=dosage_pgen)],
    )
    sv = SparseVar2(out, fields=["VAF"])
    assert "VAF" in sv.available_fields


def test_from_pgen_dosage_psam_mismatch_raises(tmp_path, hardcall_pgen, mismatched_dosage_pgen, reference_fasta):
    with pytest.raises(ValueError, match="samples"):
        SparseVar2.from_pgen(
            tmp_path / "out.svar2", hardcall_pgen, reference_fasta,
            dosages=[DosageField(name="VAF", source=mismatched_dosage_pgen)],
        )


def test_from_pgen_no_dosages_unchanged(tmp_path, hardcall_pgen, reference_fasta):
    out = tmp_path / "out.svar2"
    SparseVar2.from_pgen(out, hardcall_pgen, reference_fasta)  # dosages=None default
    sv = SparseVar2(out)
    assert sv.available_fields == {}
```

Fixtures (`pgen_with_dosages`, `hardcall_pgen`, `dosage_pgen`, `mismatched_dosage_pgen`, `reference_fasta`): add to `tests/conftest.py` or the nearest existing PGEN-fixture module. Build dosage PGENs with `plink2 --import-dosage` or reuse the SVAR1 dosage-PGEN test fixtures already in the repo (grep `dosage_path` in tests). If no dosage fixture exists, generate one in the fixture from a small VCF with a `DS` FORMAT field via `plink2 --vcf ... dosage=DS --make-pgen`.

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pixi run pytest tests/test_svar2_pgen_dosages.py -v`
Expected: FAIL — `from_pgen() got an unexpected keyword argument 'dosages'`.

- [ ] **Step 3: Read dosages in `PgenRecordSource::refill`**

In `src/pgen_reader.rs::refill` (~108–143), immediately after the existing `read_alleles_range` call + host copy, for each `i` in `0..self.dosage_readers.len()` call (under the same GIL scope):

```rust
self.dosage_readers[i].bind(py).call_method1(
    "read_dosages_range",
    (lo as u32, hi as u32, self.dosage_bufs[i].bind(py)),
)?;
```

Then bulk-copy each dosage buffer into `self.host_dosages[i]` (mirror the `host_buf` copy at ~130–137, but `f32`), normalizing pgenlib's missing sentinel to `f32::NAN` — **match `genoray/_pgen.py::_read_dosages` (lines 773–778): values equal to `-9` become NaN**. Confirm `PgenReader.read_dosages_range(start, end, out)` exists in the pinned pgenlib (0.91.x); if not, fall back to `read_dosages_list` with an explicit `arange(lo, hi)` index array (mirror `_read_dosages`, which uses `read_dosages_list`).

- [ ] **Step 4: Emit dosages in `next_record`**

In `next_record` (~210–233), replace `format_vals: FormatVals::Dense(Vec::new())` (line ~232) with a `FormatVals::Dense` whose length equals `self.dosage_readers.len()`, each entry `Some(per_sample_dosages_for_this_variant)` — the row of `host_dosages[i]` for the current variant. **Apply the SAME per-sample reordering the hardcall path applies** (the `sample_perm` gather used for `host_buf`): the dosage buffer returns samples in pgenlib's subset-sorted order, so the per-sample dosage vector must be permuted identically to hardcalls before being emitted, so FORMAT values align with the output sample order. Reference the hardcall row-serving logic at ~210–223 for the exact indexing.

- [ ] **Step 5: Add `dosages=` to `from_pgen` and build the readers**

In `python/genoray/_svar2.py::from_pgen` (~768), add param `dosages: "Sequence[DosageField] | None" = None` (after `signatures: bool = False,`). Import `DosageField`/`_dosage_field_to_tuple` from `_svar2_fields`. After the hardcall `readers` are built and `subset_idx` applied (~960–974), add:

```python
        from genoray._pgen import _read_psam

        dosage_specs = list(dosages) if dosages is not None else []
        # Reject duplicate / reserved names.
        seen_names: set[str] = set()
        for df in dosage_specs:
            if df.name == "mutcat":
                raise ValueError("dosage field name 'mutcat' is reserved")
            if df.name in seen_names:
                raise ValueError(f"duplicate dosage field name {df.name!r}")
            seen_names.add(df.name)

        dosage_field_tuples = [_dosage_field_to_tuple(df) for df in dosage_specs]

        # One dosage reader per (contig, shard, dosage_field), parallel to `readers`.
        # `readers` is [contig][shard]; dosage readers are [contig][shard][field].
        dosage_readers: list[list[list[object]]] = []
        for _contig in contigs:
            per_contig: list[list[object]] = []
            for _shard in range(P):
                per_shard: list[object] = []
                for df in dosage_specs:
                    if df.source == "self":
                        dpath = source
                        d_offsets = allele_idx_offsets
                    else:
                        dpath = Path(df.source)
                        if dpath.suffix != ".pgen":
                            raise ValueError(
                                f"dosage source for {df.name!r} must be a .pgen, got {dpath}"
                            )
                        if not dpath.exists():
                            raise FileNotFoundError(dpath)
                        # Separate dosage file: samples must match `source`'s .psam,
                        # and variant count must align 1:1 with `source`'s .pvar.
                        dose_psam = dpath.with_suffix(".psam")
                        if not dose_psam.exists():
                            raise FileNotFoundError(dose_psam)
                        dose_samples = _read_psam(dose_psam).tolist()
                        if dose_samples != all_psam_samples:
                            raise ValueError(
                                f"dosage file {dpath} samples do not match {source} .psam"
                            )
                        _dc, _dr, d_offsets = _pvar_contig_ranges(_find_pvar(dpath))
                        # 1:1 variant alignment: offset arrays are length n_variants+1.
                        if len(d_offsets) != len(allele_idx_offsets):
                            raise ValueError(
                                f"dosage file {dpath} has "
                                f"{len(d_offsets) - 1} variants; {source} has "
                                f"{len(allele_idx_offsets) - 1} — .pvar must align 1:1"
                            )
                        # (position/allele identity beyond count is a caller precondition)
                    r = pgenlib.PgenReader(
                        bytes(dpath), n_samples, allele_idx_offsets=d_offsets
                    )
                    if subset_idx is not None:
                        r.change_sample_subset(subset_idx)
                    per_shard.append(r)
                per_contig.append(per_shard)
            dosage_readers.append(per_contig)
```

Replace the `chunk_size is None` block to account for dosage FORMAT fields:

```python
        if chunk_size is None:
            chunk_size = _auto_chunk_size(
                len(selected_samples), 2, n_format_fields=len(dosage_specs)
            )
```

Add both new args to the `_core.run_pgen_conversion_pipeline(...)` call in the positions defined in Task 2 (`dosage_field_tuples` after `signatures`, `dosage_readers` after `readers`).

Update the `from_pgen` docstring: replace the "Dosages … ignored" bullet (~826) with the `dosages=` description (same-file `"self"` vs separate `.pgen`; stored as FORMAT fields; carrier-only under var_key routing; separate file must match `.psam` and align 1:1 on `.pvar`).

- [ ] **Step 6: Rebuild the extension and run the e2e test**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$USER
pixi run bash -lc 'maturin develop --release'
pixi run pytest tests/test_svar2_pgen_dosages.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add src/pgen_reader.rs python/genoray/_svar2.py tests/test_svar2_pgen_dosages.py
git commit -m "feat(svar2): read PGEN dosages as FORMAT fields in from_pgen"
```

---

## Task 4: SVAR1 field selection

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_svar1` ~1281; `_svar1_fields_manifest` ~1719)
- Test: add to the existing SVAR1→SVAR2 test module (grep `from_svar1` under `tests/`)

**Interfaces:**
- Produces: `from_svar1(..., fields: Sequence[str] | None = None)`. `None` = carry all (current behavior); `[...]` = subset by SVAR1 field name; `[]` = none. Unknown name raises `ValueError` listing available names.

- [ ] **Step 1: Write failing tests**

In the existing from_svar1 test file (e.g. `tests/test_svar2.py`), add — reuse the fixture that builds a SVAR1 store with a `dosages` field:

```python
def test_from_svar1_carries_all_fields_by_default(tmp_path, svar1_with_dosages, reference_fasta):
    out = tmp_path / "out.svar2"
    SparseVar2.from_svar1(out, svar1_with_dosages, reference_fasta)
    assert "dosages" in SparseVar2(out).available_fields


def test_from_svar1_fields_subset_empty(tmp_path, svar1_with_dosages, reference_fasta):
    out = tmp_path / "out.svar2"
    SparseVar2.from_svar1(out, svar1_with_dosages, reference_fasta, fields=[])
    assert SparseVar2(out).available_fields == {}


def test_from_svar1_fields_unknown_raises(tmp_path, svar1_with_dosages, reference_fasta):
    with pytest.raises(ValueError, match="nope"):
        SparseVar2.from_svar1(
            tmp_path / "out.svar2", svar1_with_dosages, reference_fasta, fields=["nope"]
        )
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pixi run pytest tests/test_svar2.py -k from_svar1_fields -v`
Expected: FAIL — `from_svar1() got an unexpected keyword argument 'fields'`.

- [ ] **Step 3: Add `fields=` and filter the manifest**

In `from_svar1` add param `fields: "Sequence[str] | None" = None` (after `signatures: bool = False,`). After `meta_samples, ploidy, contigs, fields_meta = _read_svar1_metadata(source)` (rename the unpacked `fields` local to `fields_meta` to avoid shadowing the new param), and before `_svar1_fields_manifest(...)`, select:

```python
        if fields is None:
            selected_fields = fields_meta  # carry all (lossless migration)
        else:
            available = [n for n in fields_meta if n != "mutcat"]
            unknown = [n for n in fields if n not in available]
            if unknown:
                raise ValueError(
                    f"unknown SVAR1 field(s) {unknown}; available: {sorted(available)}"
                )
            selected_fields = {n: fields_meta[n] for n in fields}
```

Change the manifest call to `format_tuples, src_dtypes = _svar1_fields_manifest(selected_fields)`. Update the `from_svar1` docstring (the "All SVAR1 FORMAT fields … carried through" paragraph ~1312) to document `fields=` (None=all, `[]`=none, subset by name).

- [ ] **Step 4: Run tests to confirm pass**

Run: `pixi run pytest tests/test_svar2.py -k from_svar1 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_svar2.py tests/test_svar2.py
git commit -m "feat(svar2): add field selection to from_svar1 (default carries all)"
```

---

## Task 5: CLI restructure — `write vcf` / `write pgen` / `write svar1` + legacy `write-svar1`

**Files:**
- Modify: `python/genoray/_cli/__main__.py`
- Test: `tests/test_cli.py` (grep for existing CLI test module; create if none)

**Interfaces:**
- Consumes: `_parse_cli_field_specs`, `DosageField` (Task 1); `from_pgen(dosages=)` (Task 3); `from_svar1(fields=)` (Task 4).
- Produces CLI commands:
  - `genoray write vcf SOURCE OUT [--fields "INFO/AF" ...]` → `from_vcf` / `from_vcf_list`.
  - `genoray write pgen SOURCE OUT [--dosages NAME=self|PATH ...]` → `from_pgen`.
  - `genoray write svar1 SOURCE OUT [--fields NAME ...] [--empty-fields]` → `from_svar1` (SVAR1→SVAR2).
  - `genoray write-svar1 SOURCE OUT ...` → legacy VCF/PGEN→SVAR1 (the previous `write svar1`).

- [ ] **Step 1: Write failing CLI tests**

In `tests/test_cli.py`, drive the cyclopts `app` in-process (mirror existing CLI tests' invocation pattern — grep `from genoray._cli` in tests):

```python
from genoray._cli.__main__ import app


def test_write_bare_is_removed():
    # bare `write SOURCE OUT` no longer resolves to a converter
    with pytest.raises(SystemExit):
        app(["write", "x.vcf.gz", "out.svar2", "--no-reference"], exit_on_error=False)


def test_write_vcf_fields_parsed(tmp_path, small_vcf, reference_fasta, monkeypatch):
    captured = {}
    import genoray
    def fake_from_vcf(out, source, ref, **kw):
        captured.update(kw); return 0
    monkeypatch.setattr(genoray.SparseVar2, "from_vcf", staticmethod(fake_from_vcf))
    app(["write", "vcf", str(small_vcf), str(tmp_path/"o.svar2"),
         "--reference", str(reference_fasta), "--fields", "INFO/AF", "--fields", "FMT/AD"])
    assert captured["info_fields"] == ["AF"]
    assert captured["format_fields"] == ["AD"]


def test_write_pgen_dosages_parsed(tmp_path, small_pgen, reference_fasta, monkeypatch):
    captured = {}
    import genoray
    def fake_from_pgen(out, source, ref, **kw):
        captured.update(kw); return 0
    monkeypatch.setattr(genoray.SparseVar2, "from_pgen", staticmethod(fake_from_pgen))
    app(["write", "pgen", str(small_pgen), str(tmp_path/"o.svar2"),
         "--reference", str(reference_fasta),
         "--dosages", "DS=self", "--dosages", "VAF=/x/vaf.pgen"])
    ds = {d.name: d.source for d in captured["dosages"]}
    assert ds == {"DS": "self", "VAF": "/x/vaf.pgen"}


def test_write_svar1_legacy_still_exists(tmp_path, small_vcf, monkeypatch):
    import genoray
    called = {}
    monkeypatch.setattr(genoray.SparseVar, "from_vcf", staticmethod(lambda *a, **k: called.setdefault("hit", True)))
    app(["write-svar1", str(small_vcf), str(tmp_path/"o.svar")])
    assert called.get("hit")
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/test_cli.py -v`
Expected: FAIL — subcommands `write vcf` / `write pgen` / `write-svar1` don't exist; bare `write` still resolves.

- [ ] **Step 3: Replace `@write.default` with source subcommands**

In `python/genoray/_cli/__main__.py`:
- **Delete** the `@write.default def write_svar2(...)` function (lines 54–249) and split its body into three `@write.command`-decorated functions: `write_vcf` (name `"vcf"`), `write_pgen` (name `"pgen"`), `write_from_svar1` (name `"svar1"`). Each keeps only the shared flags relevant to its source (drop `--ploidy` from `pgen`; keep it for `vcf`). Move the per-source dispatch body from the old if/elif into the matching subcommand (VCF single vs vcf-list detection stays inside `write vcf`).
- `write_vcf` gains `fields: Annotated[list[str] | None, Parameter(name=["--fields", "-f"])] = None`; parse via `from genoray._svar2_fields import _parse_cli_field_specs` → `info_fields, format_fields` passed to `from_vcf`/`from_vcf_list`. (Reject `--fields` for the vcf-list form only if FORMAT semantics differ — they don't; pass through.)
- `write_pgen` gains `dosages: Annotated[list[str] | None, Parameter(name="--dosages")] = None`; parse each `NAME=SRC` into a `DosageField`, pass list to `from_pgen(dosages=...)`:

```python
from genoray import DosageField

dosage_specs: list[DosageField] | None = None
if dosages:
    dosage_specs = []
    for entry in dosages:
        name, sep, src = entry.partition("=")
        if not sep or not name or not src:
            raise ValueError(
                f"--dosages must be NAME=self or NAME=/path.pgen, got {entry!r}"
            )
        dosage_specs.append(DosageField(name=name, source=src))
```

- `write_from_svar1` gains `fields: list[str] | None = None` and `empty_fields: Annotated[bool, Parameter(name="--empty-fields", negative="")] = False`; pass `fields=[] if empty_fields else fields` to `from_svar1`.
- **Rename** the existing `@write.command(name="svar1") def write_svar1(...)` (legacy VCF/PGEN→SVAR1, lines 252–358): promote it to a top-level `@app.command(name="write-svar1")` (unchanged body/flags). Remove its `@write.command` registration.

- [ ] **Step 4: Run tests to confirm pass**

Run: `pixi run pytest tests/test_cli.py -v`
Expected: PASS. Also smoke-check help: `pixi run genoray write --help` lists `vcf`, `pgen`, `svar1`; `pixi run genoray write-svar1 --help` works.

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_cli/__main__.py tests/test_cli.py
git commit -m "feat(cli): split write into per-source subcommands; rename legacy writer to write-svar1"
```

---

## Task 6: Update the API skill doc

**Files:**
- Modify: `skills/genoray-api/SKILL.md`

- [ ] **Step 1: Update the public-surface + details lines**

Edit `skills/genoray-api/SKILL.md`:
- Line 20 (`SparseVar2` bullet) and line 39 (`_svar2.py` details): change `from_pgen` — no longer "no `info_fields`/`format_fields`"; now "diploid-only; `dosages=Sequence[DosageField]` stores per-sample dosage tracks (from the hardcall `.pgen` via `source="self"` or a separate `.pgen`) as FORMAT fields". Change `from_svar1` — "`fields=` selects which SVAR1 fields carry (default `None` = all, `[]` = none)".
- Line 21: add `genoray.DosageField` — frozen dataclass (`name="dosage"`, `source="self"|Path`, `dtype="f32"`, `default=None`) configuring a PGEN dosage FORMAT field for `SparseVar2.from_pgen`.
- Line 40 (`_svar2_fields.py`): add `DosageField` and `_parse_cli_field_specs`.
- Line 41 (CLI): replace `write / write svar1` with `write vcf / write pgen / write svar1` (all → SVAR2) and top-level `write-svar1` (legacy → SVAR1).
- The `from_pgen` signature block (~362) and the "Conversion from PGEN" section (~351): add the `dosages=` kwarg and a short example; delete/adjust the "No dosages … dropped by design" note (~345) to describe the new dosage support.

- [ ] **Step 2: Verify no stale claims**

Run: `grep -n "no dosages\|No dosages\|info_fields=`/`format_fields=`\|write svar1" skills/genoray-api/SKILL.md`
Confirm every hit reflects the new reality (PGEN dosages supported; CLI is `write vcf/pgen/svar1` + `write-svar1`).

- [ ] **Step 3: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(skill): document DosageField, from_pgen(dosages=), from_svar1(fields=), new write CLI"
```

---

## Task 7: Full suite + final verification

- [ ] **Step 1: Rebuild and run the whole suite**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-$USER
pixi run bash -lc 'maturin develop --release'
pixi run bash -lc 'cargo test --no-default-features'
pixi run test
```
Expected: all green. If `pixi run test` regenerates fixtures via `gen_from_vcf.sh`, ensure new dosage fixtures are produced deterministically there (not `Math.random`-style).

- [ ] **Step 2: Lint**

```bash
pixi run bash -lc 'ruff check genoray tests && ruff format --check genoray tests'
```

- [ ] **Step 3: Open the PR** (per background-job shipping rule)

```bash
git push -u origin worktree-svar2-conversion-fields
gh pr create --draft --title "feat(svar2): arbitrary fields in X→SVAR2 conversion" --body "..."
```

---

## Execution Strategy (parallelism)

Per the repo's SDD conventions, dispatch implementers with **subagent-driven-development** using **Sonnet or weaker** models (Opus only for review / critical-failure fixes), and give each an explicit **foreground-only** rule for `cargo`/`maturin` (implementers otherwise background long builds and return early). Force `cd` into the worktree + a `git rev-parse` guard.

Dependency graph / parallel groups:
- **Group A (parallel):** Task 1 (Python spec/parser) ‖ Task 2 (Rust pipeline seam) ‖ Task 4 (SVAR1 selection, Python). No shared files.
- **Then:** Task 3 (needs Task 1 + Task 2; touches `pgen_reader.rs` after Task 2's edits there — must follow Task 2, not parallel).
- **Then:** Task 5 (CLI; needs Tasks 1, 3, 4).
- **Then:** Task 6 (docs; needs all code tasks) → Task 7 (verify/PR).

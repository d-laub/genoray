# SVAR2 M6a — PyO3 Query Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared PyO3 seam both M6 consumers stand on — a `numpy` dependency, reusable Rust→numpy array-conversion helpers, a `PyContigReader` pyclass, and a Python `SparseVar2` skeleton that reads `meta.json` and opens one native reader per contig.

**Architecture:** The SVAR2 query spine already exists in Rust (`overlap_batch`/`BatchResult`/`decode_hap` in `src/query.rs`), but nothing query-related crosses into Python yet. This plan adds the thin, shared boundary layer: a `src/py_convert.rs` helper module (the array plumbing both M6b and M6c reuse), a `src/py_query.rs` `PyContigReader` wrapping the existing `ContigReader::open`, registration in the `_core` pymodule, and a Python `SparseVar2` class. No query method is exposed here — those land in M6b (raw two-channel) and M6c (decoded Ragged), each against the frozen contract in the spec.

**Tech Stack:** Rust (edition 2024), `pyo3 0.29`, `rust-numpy` (matching pyo3), `maturin` (module `genoray._core`), Python 3.10+, `pytest`, `pixi` for envs.

## Global Constraints

- **pyo3 version:** `0.29` (already pinned in `Cargo.toml`). `rust-numpy` must resolve to the release whose `pyo3` dep matches `0.29` — it re-exports pyo3, so a mismatch is a compile error, not a silent bug.
- **GIL API:** pyo3 0.29 uses attach/detach terminology. `src/lib.rs` already uses `py.detach(|| …)`; acquire the GIL in tests with `Python::attach(|py| { … })` (the counterpart), **not** the older `Python::with_gil`.
- **Rust tests link libpython:** run them with `pixi run cargo test --no-default-features` — the `extension-module` feature is default-on for the wheel but must be **off** so the test binary links libpython (see the `Cargo.toml` comment on the feature).
- **Python module name:** `genoray._core` (set by `[tool.maturin] module-name`). Rebuild the extension with `pixi run maturin develop` before running pytest.
- **`meta.json` is Python-read** (its established convention — Rust only writes it, `src/meta.rs`). The schema is `{format_version: u32, samples: [str], contigs: [str], ploidy: usize}`.
- **Frozen FFI dtypes** (from the M6 spec, for the helpers): positions/keys/ranges → `i32`; offsets → `i64`; masks/LUT/alleles → `u8`. `u32` keys cross as their `i32` bit-pattern (`x as i32` in Rust is bit-preserving).
- **Paths:** all on-disk paths come from `src/layout.rs` / `ContigPaths` — do not hand-build paths.
- **Commits:** Conventional Commits (commitizen-checked). End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **prek hooks:** ensure installed (`pixi run prek-install`) before committing; the fmt/clippy/check hooks gate the commit.

---

### Task 1: `numpy` dependency + Rust→numpy conversion helpers

**Files:**
- Modify: `Cargo.toml` (add `numpy` dependency)
- Create: `src/py_convert.rs`
- Modify: `src/lib.rs` (add `pub mod py_convert;`)
- Test: in-source `#[cfg(test)]` in `src/py_convert.rs`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces (used by M6b/M6c, not by later M6a tasks):
  - `crate::py_convert::u32_to_i32_pyarray<'py>(py: Python<'py>, xs: &[u32]) -> Bound<'py, numpy::PyArray1<i32>>` — bit-preserving `u32`→`i32` (positions + 32-bit keys).
  - `crate::py_convert::i32_to_pyarray<'py>(py: Python<'py>, xs: &[i32]) -> Bound<'py, numpy::PyArray1<i32>>` — ilens / flattened ranges.
  - `crate::py_convert::usize_to_i64_pyarray<'py>(py: Python<'py>, xs: &[usize]) -> Bound<'py, numpy::PyArray1<i64>>` — CSR / bit offsets.
  - `crate::py_convert::u8_to_pyarray<'py>(py: Python<'py>, xs: &[u8]) -> Bound<'py, numpy::PyArray1<u8>>` — presence masks / LUT / alleles.

- [ ] **Step 1: Add the `numpy` dependency and confirm it resolves against pyo3 0.29**

Run:
```bash
pixi run cargo add numpy
```
Then open `Cargo.toml` and confirm the new line sits alongside `pyo3 = { version = "0.29" }` in `[dependencies]`, e.g.:
```toml
numpy = "0.29"
```
Confirm the pyo3 versions unify:
```bash
pixi run cargo tree -i pyo3 --no-default-features
```
Expected: a single `pyo3 v0.29.x` node (no duplicate pyo3 versions). If `cargo add` picked a `numpy` whose pyo3 differs, pin the `numpy` release whose `pyo3` dep is `0.29` and re-run.

- [ ] **Step 2: Write the failing test module**

Create `src/py_convert.rs`:
```rust
//! Rust→numpy array-conversion helpers shared by the M6 consumers (M6b raw
//! two-channel exposure, M6c decoded materialization). One place for the
//! frozen-contract dtype conversions so the plumbing is not duplicated.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32_to_i32_preserves_bit_pattern() {
        Python::attach(|py| {
            let xs: Vec<u32> = vec![0, 100, 0x8000_0001, u32::MAX];
            let arr = u32_to_i32_pyarray(py, &xs);
            let ro = arr.readonly();
            let back = ro.as_slice().unwrap();
            let expect: Vec<i32> = xs.iter().map(|&x| x as i32).collect();
            assert_eq!(back, &expect[..]);
        });
    }

    #[test]
    fn test_i32_roundtrip() {
        Python::attach(|py| {
            let xs: Vec<i32> = vec![-3, 0, 1, 42];
            let arr = i32_to_pyarray(py, &xs);
            let ro = arr.readonly();
            assert_eq!(ro.as_slice().unwrap(), &xs[..]);
        });
    }

    #[test]
    fn test_usize_to_i64() {
        Python::attach(|py| {
            let xs: Vec<usize> = vec![0, 1, 1, 5];
            let arr = usize_to_i64_pyarray(py, &xs);
            let ro = arr.readonly();
            assert_eq!(ro.as_slice().unwrap(), &[0i64, 1, 1, 5]);
        });
    }

    #[test]
    fn test_u8_roundtrip() {
        Python::attach(|py| {
            let xs: Vec<u8> = vec![0b1010_0101, 0x00, 0xFF];
            let arr = u8_to_pyarray(py, &xs);
            let ro = arr.readonly();
            assert_eq!(ro.as_slice().unwrap(), &xs[..]);
        });
    }
}
```
Add the module to `src/lib.rs` — insert `pub mod py_convert;` in the alphabetized `pub mod` block (after `pub mod orchestrator;`, before `pub mod query;`):
```rust
pub mod orchestrator;
pub mod py_convert;
pub mod query;
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
pixi run cargo test --no-default-features py_convert
```
Expected: FAIL to compile — `cannot find function u32_to_i32_pyarray` (and the three siblings) in this scope.

- [ ] **Step 4: Write the minimal implementation**

In `src/py_convert.rs`, insert the four helpers above the `#[cfg(test)]` block:
```rust
/// Bit-preserving `u32` → `i32` numpy array. Positions and 32-bit keys both cross
/// the FFI as their `i32` bit-pattern; `x as i32` in Rust reinterprets the bits
/// (no value clamping), so the numpy side recovers the original `u32` via
/// `.view(np.uint32)`.
pub fn u32_to_i32_pyarray<'py>(py: Python<'py>, xs: &[u32]) -> Bound<'py, PyArray1<i32>> {
    let v: Vec<i32> = xs.iter().map(|&x| x as i32).collect();
    PyArray1::from_slice(py, &v)
}

/// `i32` slice → numpy array (ilens, flattened index ranges).
pub fn i32_to_pyarray<'py>(py: Python<'py>, xs: &[i32]) -> Bound<'py, PyArray1<i32>> {
    PyArray1::from_slice(py, xs)
}

/// `usize` slice → `i64` numpy array (CSR offsets, bitmask bit-offsets).
pub fn usize_to_i64_pyarray<'py>(py: Python<'py>, xs: &[usize]) -> Bound<'py, PyArray1<i64>> {
    let v: Vec<i64> = xs.iter().map(|&x| x as i64).collect();
    PyArray1::from_slice(py, &v)
}

/// `u8` slice → numpy array (presence bitmasks, LUT bytes, packed alleles).
pub fn u8_to_pyarray<'py>(py: Python<'py>, xs: &[u8]) -> Bound<'py, PyArray1<u8>> {
    PyArray1::from_slice(py, xs)
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
pixi run cargo test --no-default-features py_convert
```
Expected: PASS — `test_u32_to_i32_preserves_bit_pattern`, `test_i32_roundtrip`, `test_usize_to_i64`, `test_u8_roundtrip` all green.

- [ ] **Step 6: Commit**

```bash
pixi run prek-install
git add Cargo.toml Cargo.lock src/py_convert.rs src/lib.rs
git commit -m "feat(svar2): add rust-numpy dep and Rust->numpy conversion helpers (M6a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `PyContigReader` pyclass + `_core` registration

**Files:**
- Create: `src/py_query.rs`
- Modify: `src/lib.rs` (add `pub mod py_query;` and register the class in `_core`)
- Test: `tests/test_py_query.rs` (reuses the `tests/common` fixture harness)

**Interfaces:**
- Consumes: `crate::query::ContigReader::open(base_out_dir: &str, chrom: &str, n_samples: usize, ploidy: usize) -> std::io::Result<ContigReader>` (existing).
- Produces:
  - `crate::py_query::PyContigReader` — a `#[pyclass]` exposed to Python as `genoray._core.PyContigReader`, holding `pub(crate) inner: ContigReader`.
  - Python constructor `PyContigReader(base_out_dir: str, chrom: str, n_samples: int, ploidy: int)` raising `OSError` on failure. Rust-callable as `PyContigReader::new(base_out_dir, chrom, n_samples, ploidy) -> PyResult<Self>`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_py_query.rs`:
```rust
//! Boundary test for the M6a PyO3 seam: `PyContigReader` opens a finished contig
//! built through the real conversion pipeline, and tolerates an empty contig dir
//! (mirroring `ContigReader::open`'s missing-sub-stream contract).

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::py_query::PyContigReader;
use pyo3::Python;
use tempfile::tempdir;

#[test]
fn test_py_contig_reader_opens_built_contig() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt: vec![1, 0, 1, 1],
    }];
    build_contig(&out, "chr1", &samples, 2, &records);

    Python::attach(|_py| {
        let r = PyContigReader::new(out.to_str().unwrap(), "chr1", 2, 2);
        assert!(r.is_ok(), "PyContigReader should open a built contig");
    });
}

#[test]
fn test_py_contig_reader_empty_dir_tolerated() {
    let tmp = tempdir().unwrap();
    Python::attach(|_py| {
        // No contig dir / sub-streams: opens as all-empty, no error.
        let r = PyContigReader::new(tmp.path().to_str().unwrap(), "chrX", 1, 2);
        assert!(r.is_ok(), "empty contig should still open");
    });
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pixi run cargo test --no-default-features --test test_py_query
```
Expected: FAIL to compile — `unresolved import genoray_core::py_query` / `PyContigReader` not found.

- [ ] **Step 3: Write the minimal implementation**

Create `src/py_query.rs`:
```rust
//! Python-facing handle over a finished SVAR2 contig (M6a). Wraps the pure-Rust
//! `query::ContigReader` so Python can open a contig; query methods (raw
//! two-channel M6b, decoded M6c) are added to this class in their milestones.

use pyo3::prelude::*;

use crate::query::ContigReader;

/// A finished SVAR2 contig opened for querying. Constructed from Python as
/// `PyContigReader(base_out_dir, chrom, n_samples, ploidy)`.
#[pyclass]
pub struct PyContigReader {
    pub(crate) inner: ContigReader,
}

#[pymethods]
impl PyContigReader {
    // `pub` so the integration test (an external crate) can call it directly as a
    // plain Rust constructor; pyo3 keeps `#[new]` methods callable from Rust.
    #[new]
    pub fn new(base_out_dir: &str, chrom: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        let inner = ContigReader::open(base_out_dir, chrom, n_samples, ploidy)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}
```
In `src/lib.rs`, add the module in the alphabetized block (after `pub mod py_convert;`):
```rust
pub mod py_convert;
pub mod py_query;
pub mod query;
```
And register the class inside the `_core` pymodule (extend the existing `_core` fn):
```rust
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_conversion_pipeline, m)?)?;
    m.add_class::<crate::py_query::PyContigReader>()?;
    Ok(())
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
pixi run cargo test --no-default-features --test test_py_query
```
Expected: PASS — `test_py_contig_reader_opens_built_contig` and `test_py_contig_reader_empty_dir_tolerated` green.

- [ ] **Step 5: Guard against the `#[allow(dead_code)]` regression on `inner`**

`inner` is unused until M6b/M6c add query methods. Confirm clippy stays clean (the field is read by pyo3's generated code, so it should not warn):
```bash
pixi run cargo clippy --no-default-features --all-targets
```
Expected: no `field inner is never read` warning, no errors. If clippy flags `inner`, add `#[allow(dead_code)]` on the field with a comment noting M6b/M6c consume it — do **not** remove it.

- [ ] **Step 6: Commit**

```bash
git add src/py_query.rs src/lib.rs tests/test_py_query.rs
git commit -m "feat(svar2): expose PyContigReader over ContigReader::open (M6a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Python `SparseVar2` skeleton

**Files:**
- Create: `python/genoray/_svar2.py`
- Modify: `python/genoray/__init__.py` (export `SparseVar2`)
- Test: `tests/test_svar2.py`

**Interfaces:**
- Consumes: `genoray._core.PyContigReader(base_out_dir: str, chrom: str, n_samples: int, ploidy: int)` (Task 2).
- Produces:
  - `genoray.SparseVar2(path: str | Path)` with attributes `.path: Path`, `.format_version: int`, `.samples: list[str]`, `.contigs: list[str]`, `.ploidy: int`, `.n_samples: int` (property), and `._readers: dict[str, _core.PyContigReader]` (one per contig).

- [ ] **Step 1: Write the failing test**

Create `tests/test_svar2.py`:
```python
import json
from pathlib import Path

from genoray import SparseVar2, _core


def _write_fixture(root: Path) -> None:
    """A finished-store fixture: meta.json + empty per-contig dirs. Empty
    sub-streams are tolerated by ContigReader::open, so no pipeline run is
    needed to exercise the skeleton."""
    meta = {
        "format_version": 1,
        "samples": ["S0", "S1"],
        "contigs": ["chr1", "chr2"],
        "ploidy": 2,
    }
    (root / "meta.json").write_text(json.dumps(meta))
    for contig in meta["contigs"]:
        (root / contig).mkdir()


def test_sparsevar2_reads_meta(tmp_path):
    _write_fixture(tmp_path)
    sv = SparseVar2(tmp_path)
    assert sv.samples == ["S0", "S1"]
    assert sv.contigs == ["chr1", "chr2"]
    assert sv.ploidy == 2
    assert sv.n_samples == 2
    assert sv.format_version == 1


def test_sparsevar2_opens_a_reader_per_contig(tmp_path):
    _write_fixture(tmp_path)
    sv = SparseVar2(tmp_path)
    assert set(sv._readers) == {"chr1", "chr2"}
    assert all(isinstance(r, _core.PyContigReader) for r in sv._readers.values())
```

- [ ] **Step 2: Build the extension and run the test to verify it fails**

Run:
```bash
pixi run maturin develop
pixi run pytest tests/test_svar2.py -v
```
Expected: FAIL — `ImportError: cannot import name 'SparseVar2' from 'genoray'`.

- [ ] **Step 3: Write the minimal implementation**

Create `python/genoray/_svar2.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

from genoray import _core


class SparseVar2:
    """Reader for a finished SVAR2 store (M6a skeleton).

    Loads the top-level ``meta.json`` and opens one native
    :class:`genoray._core.PyContigReader` per contig. Query methods land in M6b
    (raw two-channel result) and M6c (decoded ``seqpro.rag.Ragged``).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        self.format_version: int = meta["format_version"]
        self.samples: list[str] = list(meta["samples"])
        self.contigs: list[str] = list(meta["contigs"])
        self.ploidy: int = meta["ploidy"]
        self._readers: dict[str, _core.PyContigReader] = {
            contig: _core.PyContigReader(
                str(self.path), contig, len(self.samples), self.ploidy
            )
            for contig in self.contigs
        }

    @property
    def n_samples(self) -> int:
        return len(self.samples)
```
Edit `python/genoray/__init__.py` to export it — add the import next to the other reader imports and extend `__all__`:
```python
from ._svar import SparseGenotypes, SparseVar
from ._svar2 import SparseVar2
from ._vcf import VCF
```
```python
__all__ = ["Reader", "VCF", "PGEN", "SparseVar", "SparseVar2", "SparseGenotypes", "exprs"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
pixi run pytest tests/test_svar2.py -v
```
Expected: PASS — `test_sparsevar2_reads_meta` and `test_sparsevar2_opens_a_reader_per_contig` green.

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_svar2.py python/genoray/__init__.py tests/test_svar2.py
git commit -m "feat(svar2): add SparseVar2 Python skeleton reading meta.json (M6a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Manual verification (after all tasks)

- [ ] Full Rust suite (extension-module off) is green:
  ```bash
  pixi run cargo test --no-default-features
  ```
- [ ] Extension rebuilds and the new Python test passes from a clean build:
  ```bash
  pixi run maturin develop && pixi run pytest tests/test_svar2.py -v
  ```
- [ ] Lint gate is clean:
  ```bash
  pixi run -e lint lint
  ```
- [ ] Update the roadmap: tick **M6a** to `[x]` in `docs/roadmap/svar-2.md` with a one-line "shipped" note (per the SVAR2 working agreement — any PR touching the effort updates the roadmap in the same PR). Do this in a final docs commit.

## Notes for M6b/M6c (out of scope here, do not implement)

- M6b adds a `PyContigReader.overlap_batch_raw(regions)` method returning the frozen-contract arrays via the Task 1 helpers, plus LUT exposure and dense-window packing.
- M6c adds `PyContigReader.decode_batch(regions)` (loops `BatchResult::decode_hap`) and `count_batch(regions)`, then `SparseVar2` assembles `seqpro.rag.Ragged.from_fields`.
- Both add methods in their **own** `#[pymethods] impl PyContigReader` block (pyo3 0.29 allows multiple), so they land without conflict on the shared class.

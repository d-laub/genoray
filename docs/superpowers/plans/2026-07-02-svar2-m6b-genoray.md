# SVAR2 M6b (genoray side) — Raw Two-Channel numpy Exposure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the batched two-channel `BatchResult` from a finished SVAR2 contig to Python as the frozen numpy contract, so the gvl two-source kernel (and the M6c counts path) can consume it.

**Architecture:** A new `#[pymethods] impl PyContigReader` block in the M6b-owned `src/py_query_batch.rs` adds `overlap_batch(regions)`, which calls the existing pure-Rust `query::overlap_batch` and converts `BatchResult`'s fields into the frozen dtype/shape table (positions/keys as `i32` bit-patterns, offsets as `i64`, masks/LUT as `u8`). The shared long-allele LUT is exposed via a new additive `ContigReader::lut_arrays` accessor. A thin `_BatchQueryMixin.overlap_batch` wraps it per contig.

**Tech Stack:** Rust (pyo3 0.29, rust-numpy 0.29, ndarray), Python 3.10, pixi/maturin.

## Global Constraints

- Runs in worktree `/carter/users/dlaub/projects/genoray/.claude/worktrees/svar-2-m6b` on branch `svar-2-m6b`. Verify with `git rev-parse --show-toplevel` before any git/build step.
- Depends on the prep commit (Plan `2026-07-02-svar2-m6-prep.md`): `multiple-pymethods` enabled; `src/py_query_batch.rs` + `python/genoray/_svar2_batch.py` stubs exist; `tests/conftest.py` `svar2_store` fixture exists.
- **Owned files only:** `src/py_query_batch.rs`, `python/genoray/_svar2_batch.py`, `tests/test_svar2_batch.py`, `tests/test_batch_raw.rs`. Additive edits to `src/query.rs` and `src/nrvk.rs` (new accessors only — M6c touches neither, so no conflict). Do NOT edit `src/py_query.rs`, `src/py_query_decode.rs`, `python/genoray/_svar2.py`, `python/genoray/_svar2_decode.py`, `src/py_convert.rs`, or `tests/conftest.py`.
- **Frozen `BatchResult → numpy` contract** (M6a; code against these exactly). `H = n_regions·n_samples·ploidy`; hap index `h = (r·n_samples + s)·ploidy + p`:

  | key | dtype | shape | source |
  | --- | --- | --- | --- |
  | `vk_pos` | i32 | `[·]` | `br.vk[i].position` |
  | `vk_key` | i32 (u32 bits) | `[·]` | `br.vk[i].key` |
  | `vk_off` | i64 | `[H+1]` | `br.vk_off` |
  | `dense_pos` | i32 | `[D]` | `br.dense[i].position` |
  | `dense_key` | i32 (u32 bits) | `[D]` | `br.dense[i].key` |
  | `dense_range` | i32 | `[R,2]` | `br.dense_range` |
  | `dense_present` | u8 | `[·]` | `br.dense_present` |
  | `dense_present_off` | i64 | `[H+1]` | `br.dense_present_off` (BIT offsets) |
  | `lut_bytes` | u8 | `[·]` | `ContigReader::lut_arrays().0` |
  | `lut_off` | i64 | `[·]` | `ContigReader::lut_arrays().1` (u64→i64) |
- pre-commit hooks run on commit; fix `cargo fmt`/`clippy`/`ruff` failures, do not bypass.

---

### Task 1: Raw LUT accessors

**Files:**
- Modify: `src/nrvk.rs` (add two methods to `impl LongAlleleReader`, ~after line 132; add a unit test)
- Modify: `src/query.rs` (add `ContigReader::lut_arrays`, in an `impl ContigReader` block)

**Interfaces:**
- Produces: `LongAlleleReader::offsets(&self) -> &[u64]`, `LongAlleleReader::all_bytes(&self) -> Vec<u8>`, and `ContigReader::lut_arrays(&self) -> (Vec<u8>, Vec<u64>)`. Consumed by Task 2's pymethod.

- [ ] **Step 1: Write the failing LUT-reader test**

In `src/nrvk.rs`, inside the existing `#[cfg(test)] mod tests`, add:

```rust
    #[test]
    fn test_reader_all_bytes_and_offsets() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("chr1").join("indel");
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::File::create(dir.join("long_alleles.bin")).unwrap();
        f.write_all(b"AAAACC").unwrap();
        let offsets = ndarray::Array1::from_vec(vec![0u64, 4, 6]);
        ndarray_npy::write_npy(dir.join("long_allele_offsets.npy"), &offsets).unwrap();

        let reader = LongAlleleReader::new(tmp.path().to_str().unwrap(), "chr1");
        assert_eq!(reader.offsets(), &[0u64, 4, 6]);
        assert_eq!(reader.all_bytes(), b"AAAACC".to_vec());
    }
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd .claude/worktrees/svar-2-m6b && pixi run cargo test -p genoray nrvk::tests::test_reader_all_bytes_and_offsets`
Expected: FAIL — `no method named offsets`/`all_bytes`.

- [ ] **Step 3: Add the accessors**

In `src/nrvk.rs`, inside `impl LongAlleleReader` (after `get_allele`, ~line 132):

```rust
    /// CSR row offsets (`len == n_long_alleles + 1`); byte range of row `i` is
    /// `offsets[i]..offsets[i+1]`.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// The entire long-allele byte bank (M6b raw-LUT exposure). Reads the whole
    /// file once; the LUT holds only long-INS spills (SNPs never spill, most
    /// indels are inline), so it is typically small.
    pub fn all_bytes(&self) -> Vec<u8> {
        let total = *self.offsets.last().unwrap_or(&0) as usize;
        let mut buf = vec![0u8; total];
        if total > 0 {
            self.file
                .read_exact_at(&mut buf, 0)
                .expect("pread long_alleles.bin");
        }
        buf
    }
```

- [ ] **Step 4: Run the reader test to confirm it passes**

Run: `pixi run cargo test -p genoray nrvk::tests::test_reader_all_bytes_and_offsets`
Expected: PASS.

- [ ] **Step 5: Add `ContigReader::lut_arrays`**

In `src/query.rs`, add a new `impl ContigReader` block (place it right after the existing `impl ContigReader { pub fn open(...) }`, ~line 254):

```rust
impl ContigReader {
    /// Raw long-allele LUT for the M6b contract: all bytes + CSR row offsets.
    /// A contig with no LUT returns empty bytes and a single `[0]` offset (an
    /// empty CSR), so the numpy side never special-cases a missing file.
    pub fn lut_arrays(&self) -> (Vec<u8>, Vec<u64>) {
        match &self.lut {
            Some(l) => (l.all_bytes(), l.offsets().to_vec()),
            None => (Vec::new(), vec![0u64]),
        }
    }
}
```

- [ ] **Step 6: Verify the crate builds**

Run: `pixi run cargo check -p genoray`
Expected: compiles.

- [ ] **Step 7: Commit**

```bash
git add src/nrvk.rs src/query.rs
git commit -m "feat(svar2): raw LUT accessors for M6b two-channel exposure"
```

---

### Task 2: `PyContigReader.overlap_batch` pymethod

**Files:**
- Modify: `src/py_query_batch.rs` (replace the empty stub block)

**Interfaces:**
- Consumes: `query::overlap_batch`, `BatchResult` fields, `ContigReader::lut_arrays` (Task 1), the `py_convert` helpers, `PyContigReader { inner }`.
- Produces: `PyContigReader.overlap_batch(regions: list[(int,int)]) -> dict[str, ndarray]` returning the frozen contract keys plus `n_regions`/`n_samples`/`ploidy` ints.

- [ ] **Step 1: Replace the stub with the method**

Overwrite `src/py_query_batch.rs`:

```rust
//! M6b: raw two-channel `BatchResult` → numpy exposure on `PyContigReader`.
//! Owned by the `svar-2-m6b` worktree; separate `#[pymethods]` block
//! (multiple-pymethods) so M6b and M6c never touch the same file.

use ndarray::Array2;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_convert::{u32_to_i32_pyarray, u8_to_pyarray, usize_to_i64_pyarray};
use crate::py_query::PyContigReader;
use crate::query::overlap_batch;

#[pymethods]
impl PyContigReader {
    /// Batched two-channel query over `regions` (half-open `[q_start, q_end)`)
    /// within this contig. Returns the frozen `BatchResult → numpy` contract as a
    /// dict keyed by the contract array names, plus `n_regions`/`n_samples`/`ploidy`.
    fn overlap_batch<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let br = overlap_batch(&self.inner, &regions);

        let vk_pos: Vec<u32> = br.vk.iter().map(|k| k.position).collect();
        let vk_key: Vec<u32> = br.vk.iter().map(|k| k.key).collect();
        let dense_pos: Vec<u32> = br.dense.iter().map(|k| k.position).collect();
        let dense_key: Vec<u32> = br.dense.iter().map(|k| k.key).collect();

        // dense_range as [R, 2] i32.
        let r = br.dense_range.len();
        let mut dr: Vec<i32> = Vec::with_capacity(r * 2);
        for &(s, e) in &br.dense_range {
            dr.push(s as i32);
            dr.push(e as i32);
        }
        let dense_range = Array2::from_shape_vec((r, 2), dr)
            .expect("dense_range shape")
            .to_pyarray(py);

        let (lut_bytes, lut_off_u64) = self.inner.lut_arrays();
        let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();

        let d = PyDict::new(py);
        d.set_item("vk_pos", u32_to_i32_pyarray(py, &vk_pos))?;
        d.set_item("vk_key", u32_to_i32_pyarray(py, &vk_key))?;
        d.set_item("vk_off", usize_to_i64_pyarray(py, &br.vk_off))?;
        d.set_item("dense_pos", u32_to_i32_pyarray(py, &dense_pos))?;
        d.set_item("dense_key", u32_to_i32_pyarray(py, &dense_key))?;
        d.set_item("dense_range", dense_range)?;
        d.set_item("dense_present", u8_to_pyarray(py, &br.dense_present))?;
        d.set_item("dense_present_off", usize_to_i64_pyarray(py, &br.dense_present_off))?;
        d.set_item("lut_bytes", u8_to_pyarray(py, &lut_bytes))?;
        d.set_item("lut_off", PyArray1::from_slice(py, &lut_off))?;
        d.set_item("n_regions", br.n_regions)?;
        d.set_item("n_samples", br.n_samples)?;
        d.set_item("ploidy", br.ploidy)?;
        Ok(d)
    }
}
```

- [ ] **Step 2: Build the extension**

Run: `pixi run cargo check -p genoray`
Expected: compiles (warnings about unused imports mean a helper name is off — reconcile with `src/py_convert.rs`).

- [ ] **Step 3: Commit**

```bash
git add src/py_query_batch.rs
git commit -m "feat(svar2): PyContigReader.overlap_batch raw two-channel numpy method"
```

---

### Task 3: Rust integration cross-check vs the oracle `BatchResult`

**Files:**
- Create: `tests/test_batch_raw.rs`

**Interfaces:**
- Consumes: `tests/common` (`SynthRecord`, `build_contig`), `query::{ContigReader, overlap_batch}`, `py_query::PyContigReader`.
- Produces: proof the dict arrays are a faithful, dtype-correct image of `overlap_batch`'s `BatchResult`.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_batch_raw.rs`:

```rust
//! M6b: the `PyContigReader.overlap_batch` numpy dict is a faithful image of the
//! pure `query::overlap_batch` BatchResult (the already-oracle-tested reference).

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::py_query::PyContigReader;
use genoray_core::query::{ContigReader, overlap_batch};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDictMethods;
use tempfile::tempdir;

fn i32_slice<'py>(d: &Bound<'py, pyo3::types::PyDict>, k: &str) -> Vec<i32> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.downcast::<PyArray1<i32>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}
fn i64_slice<'py>(d: &Bound<'py, pyo3::types::PyDict>, k: &str) -> Vec<i64> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.downcast::<PyArray1<i64>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}
fn u8_slice<'py>(d: &Bound<'py, pyo3::types::PyDict>, k: &str) -> Vec<u8> {
    let obj = d.get_item(k).unwrap().unwrap();
    let arr = obj.downcast::<PyArray1<u8>>().unwrap().readonly();
    arr.as_slice().unwrap().to_vec()
}

#[test]
fn test_overlap_batch_dict_matches_oracle() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord { pos: 100, ref_allele: b"A", alts: vec![&b"C"[..]], gt: vec![1, 0, 0, 0] },
        SynthRecord { pos: 200, ref_allele: b"A", alts: vec![&b"AT"[..]], gt: vec![0, 1, 1, 1] },
        SynthRecord { pos: 300, ref_allele: b"AT", alts: vec![&b"A"[..]], gt: vec![1, 1, 0, 1] },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    let base = out.to_str().unwrap();
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    // Oracle: pure Rust BatchResult.
    let cr = ContigReader::open(base, "chr1", 2, 2).unwrap();
    let br = overlap_batch(&cr, &regions);

    Python::attach(|py| {
        let reader = PyContigReader::new(base, "chr1", 2, 2).unwrap();
        let d = reader.overlap_batch(py, regions.clone()).unwrap();

        assert_eq!(
            i32_slice(&d, "vk_pos"),
            br.vk.iter().map(|k| k.position as i32).collect::<Vec<_>>()
        );
        assert_eq!(
            i32_slice(&d, "vk_key"),
            br.vk.iter().map(|k| k.key as i32).collect::<Vec<_>>()
        );
        assert_eq!(
            i64_slice(&d, "vk_off"),
            br.vk_off.iter().map(|&x| x as i64).collect::<Vec<_>>()
        );
        assert_eq!(
            i32_slice(&d, "dense_pos"),
            br.dense.iter().map(|k| k.position as i32).collect::<Vec<_>>()
        );
        assert_eq!(u8_slice(&d, "dense_present"), br.dense_present);
        assert_eq!(
            i64_slice(&d, "dense_present_off"),
            br.dense_present_off.iter().map(|&x| x as i64).collect::<Vec<_>>()
        );
    });
}
```

- [ ] **Step 2: Run it to confirm it passes (the impl already exists)**

Run: `pixi run cargo test -p genoray --test test_batch_raw`
Expected: PASS. (If `downcast`/`readonly` names differ under rust-numpy 0.29, reconcile against `src/py_convert.rs`'s test imports, which use the same API.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_batch_raw.rs
git commit -m "test(svar2): cross-check overlap_batch dict against oracle BatchResult"
```

---

### Task 4: Python `_BatchQueryMixin.overlap_batch` + count-invariant test

**Files:**
- Modify: `python/genoray/_svar2_batch.py` (fill `_BatchQueryMixin`)
- Create: `tests/test_svar2_batch.py`

**Interfaces:**
- Consumes: `SparseVar2._readers[contig]` (a `PyContigReader`), the `svar2_store` fixture.
- Produces: `SparseVar2.overlap_batch(contig: str, regions) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write the failing Python test**

Create `tests/test_svar2_batch.py`:

```python
import numpy as np

from genoray import SparseVar2


def _carried_counts(res: dict, n_samples: int, ploidy: int) -> list[int]:
    """Per-hap carried-variant count for region 0 = var_key slice length +
    dense-present popcount. Invariant to left-alignment position shifts."""
    vk_off = res["vk_off"]
    dp = np.unpackbits(res["dense_present"], bitorder="little")
    dpo = res["dense_present_off"]
    counts = []
    for h in range(n_samples * ploidy):  # region 0 → h = (0*S + s)*P + p = s*P + p
        vk_n = int(vk_off[h + 1] - vk_off[h])
        dn = int(dp[dpo[h] : dpo[h + 1]].sum())
        counts.append(vk_n + dn)
    return counts


def test_overlap_batch_counts_and_dtypes(svar2_store):
    sv = SparseVar2(svar2_store)
    res = sv.overlap_batch("chr1", [(0, 40)])

    # Frozen-contract dtypes.
    for k in ("vk_pos", "vk_key", "dense_pos", "dense_key"):
        assert res[k].dtype == np.int32
    for k in ("vk_off", "dense_present_off", "lut_off"):
        assert res[k].dtype == np.int64
    assert res["dense_present"].dtype == np.uint8
    assert res["dense_range"].shape == (1, 2)

    # H + 1 offsets for 2 samples, ploidy 2, 1 region.
    assert len(res["vk_off"]) == 2 * 2 * 1 + 1
    assert int(res["vk_off"][-1]) == len(res["vk_pos"])

    # Known carriers: SNP@POS3 (S0h0), INS@POS7 (S0h1,S1h0,S1h1),
    # DEL@POS12 (S0h0,S0h1,S1h1) → per-hap [2, 2, 1, 2], total 7.
    assert _carried_counts(res, sv.n_samples, sv.ploidy) == [2, 2, 1, 2]

    # All positions within the reference.
    assert np.all((res["vk_pos"] >= 0) & (res["vk_pos"] < 40))
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd .claude/worktrees/svar-2-m6b && pixi run pytest tests/test_svar2_batch.py -q`
Expected: FAIL — `SparseVar2 has no attribute overlap_batch` (mixin still a stub).

- [ ] **Step 3: Fill the mixin**

Replace the body of `_BatchQueryMixin` in `python/genoray/_svar2_batch.py`:

```python
"""M6b: raw two-channel batch-query methods for :class:`SparseVar2`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class _BatchQueryMixin:
    """Raw ``BatchResult`` → numpy query methods."""

    def overlap_batch(
        self, contig: str, regions: Iterable[tuple[int, int]]
    ) -> dict[str, "np.ndarray"]:
        """Batched two-channel query for one ``contig``.

        ``regions`` is an iterable of half-open ``(q_start, q_end)`` pairs. Returns
        the frozen ``BatchResult`` → numpy contract as a dict of arrays (see the M6b
        plan). Cross-contig batching is the caller's job (query each contig).
        """
        reg = [(int(s), int(e)) for s, e in regions]
        return self._readers[contig].overlap_batch(reg)
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pixi run pytest tests/test_svar2_batch.py -q`
Expected: PASS. (If it fails on the count assertion, a left-align shift changed nothing about counts — so a real failure means a genotype-filter or offset bug; debug there, not in the expected values.)

- [ ] **Step 5: Run the full suite + clippy**

Run: `pixi run cargo test -p genoray && pixi run pytest tests/test_svar2_batch.py tests/test_svar2.py -q && pixi run cargo clippy -p genoray -- -D warnings`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2_batch.py tests/test_svar2_batch.py
git commit -m "feat(svar2): SparseVar2.overlap_batch raw two-channel query (M6b)"
```

---

## Self-Review

- **Spec coverage:** M6b genoray side = "raw two-channel numpy exposure + dense-window subsetting." `overlap_batch` returns the full frozen contract including per-region `dense_range` and per-hap `dense_present` window bits (the subsetting) — Task 2/4. LUT exposure — Task 1. Cross-checked exactly (Task 3) and structurally (Task 4).
- **Placeholder scan:** none; every step has full code/commands.
- **Type consistency:** helper names (`u32_to_i32_pyarray`, `usize_to_i64_pyarray`, `u8_to_pyarray`) match `src/py_convert.rs`; `overlap_batch`/`ContigReader`/`BatchResult` field names match `src/query.rs`; `lut_arrays`/`offsets`/`all_bytes` consistent across Tasks 1–2.
- **Note for the two-source consumer (gvl):** `lut_bytes`/`lut_off` are returned per call; the LUT is contig-level and usually tiny, but if profiling shows it dominates, add a cached `PyContigReader.lut()` method and drop it from the batch dict — a contract refinement to coordinate with the gvl worktree, out of scope here.

# SP-0 Quick Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** De-risk the clean-code refactor by deleting verified-dead code, fixing four latent bugs (with regression tests), and repairing wrong documentation — all behavior-preserving except the tested bug fixes.

**Architecture:** Ten independent tasks, each a self-contained commit: four pure deletions, four TDD bug fixes, two documentation/string-hygiene passes. No module restructuring (that is SP-1/SP-2/SP-7). No public-API change.

**Tech Stack:** Python (numpy, polars, cyvcf2, pgenlib, seqpro, phantom-types), Rust (PyO3 extension `genoray._core`, crossbeam-channel), Pixi for env/tasks, pytest + pytest_cases, cargo for Rust tests, ruff/clippy/pyrefly via pre-commit hooks.

**Spec:** [`docs/superpowers/specs/2026-07-08-sp0-quick-wins-design.md`](../specs/2026-07-08-sp0-quick-wins-design.md)

## Global Constraints

- **Working directory:** the worktree at `.claude/worktrees/clean-code-audit` on branch `clean-code-audit`. Confirm with `git rev-parse --abbrev-ref HEAD` → `clean-code-audit` before any commit.
- **Rust tests MUST use `--no-default-features`** or the pyo3 test binary fails to link (`undefined symbol: _Py_Dealloc`). Always: `pixi run bash -lc 'cargo test --no-default-features <args>'`.
- **All Python commands run inside Pixi:** `pixi run pytest ...`, `pixi run ruff ...`.
- **Before any deletion, re-grep `python/`, `tests/`, and `src/` to confirm zero references.** Do not trust prior analysis. This rule already corrected the spec once.
- **No public-API/`SKILL.md` change.** Nothing here renames/removes/alters a name reachable via `import genoray` without underscores. Confirm at the end.
- **Coordinate/missing-value conventions (unchanged, do not break):** ranges are 0-based half-open `[start, end)`; missing genotypes `-1`, missing dosages `np.nan`.
- **Conventional Commits.** Each commit message ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Record the one user-visible behavior change (Task 5, B1) in `CHANGELOG.md`.**

---

### Task 1: Delete dead file `src/utils.rs`

**Files:**
- Delete: `src/utils.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

`src/utils.rs` is never declared with `mod utils`, so it does not compile into the crate; its `ravel!`/`unravel!` macros have zero call sites.

- [ ] **Step 1: Re-confirm the file is dead**

Run:
```bash
grep -rn "mod utils" src/
grep -rn "ravel!\|unravel!" src/ | grep -v "macro_rules"
```
Expected: **no output** from either (first proves it is not a module; second proves the macros are unused).

- [ ] **Step 2: Delete the file**

```bash
git rm src/utils.rs
```

- [ ] **Step 3: Verify the crate still compiles**

Run: `pixi run bash -lc 'cargo check --no-default-features'`
Expected: compiles with no new errors (the file was never part of the build).

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove dead src/utils.rs (never compiled, unused macros)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Delete dead Python names `is_dtype` and `POLARS_V_IDX_TYPE`

**Files:**
- Modify: `python/genoray/_utils.py` (remove `is_dtype`, lines ~80-95)
- Modify: `python/genoray/_types.py` (remove `POLARS_V_IDX_TYPE`, line 9)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

Both are underscore-free but live in underscore-prefixed modules (not reachable via `import genoray`), and both have zero callers.

- [ ] **Step 1: Re-confirm both are dead**

Run:
```bash
grep -rn "is_dtype" python/ tests/
grep -rn "POLARS_V_IDX_TYPE" python/ tests/
```
Expected: each prints only its single definition line (`_utils.py:80` / `_types.py:9`) and nothing else.

- [ ] **Step 2: Delete `is_dtype`**

In `python/genoray/_utils.py`, delete the entire `is_dtype` function (the `def is_dtype(obj: Any, dtype: type[DTYPE]) -> TypeGuard[NDArray[DTYPE]]:` block through its `return isinstance(...)` line, plus the now-doubled blank line).

- [ ] **Step 3: Delete `POLARS_V_IDX_TYPE`**

In `python/genoray/_types.py`, delete the line:
```python
POLARS_V_IDX_TYPE = pl.Int32
```

- [ ] **Step 4: Drop imports that just went unused**

Run: `pixi run ruff check --fix python/genoray/_utils.py python/genoray/_types.py`
Expected: ruff removes any import left unused by the deletions (e.g. `import polars as pl` in `_types.py` if `pl.Int32` was its only use; `TypeGuard`/`Any` in `_utils.py` if now unused). Re-run `pixi run ruff check python/genoray/_utils.py python/genoray/_types.py` → no errors.

- [ ] **Step 5: Verify the suite still passes**

Run: `pixi run pytest -q -m "not network"`
Expected: PASS (no collection or import errors).

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_utils.py python/genoray/_types.py
git commit -m "refactor: remove unused is_dtype and POLARS_V_IDX_TYPE

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Remove stale `#[allow(dead_code)]` on `PyContigReader::inner`

**Files:**
- Modify: `src/py_query.rs:12-17`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (`inner` field unchanged, still `pub(crate)`).

`inner` is now read by `py_query_batch.rs`, `py_query_decode.rs`, and `py_query_ranges.rs`, so the `#[allow(dead_code)]` and its "not yet read" comment are obsolete.

- [ ] **Step 1: Confirm `inner` is actually read**

Run: `grep -rn "self.inner\|\.inner" src/py_query_batch.rs src/py_query_decode.rs src/py_query_ranges.rs`
Expected: several hits (the field is live).

- [ ] **Step 2: Edit the struct**

Replace this block in `src/py_query.rs`:
```rust
#[pyclass]
pub struct PyContigReader {
    // Not yet read outside this module — M6b (raw two-channel) and M6c (decoded)
    // query methods land on this class and will consume `inner`.
    #[allow(dead_code)]
    pub(crate) inner: ContigReader,
}
```
with:
```rust
#[pyclass]
pub struct PyContigReader {
    pub(crate) inner: ContigReader,
}
```

- [ ] **Step 3: Verify compile + no dead-code warning**

Run: `pixi run bash -lc 'cargo check --no-default-features 2>&1 | grep -i "dead_code\|never read" || echo NO-DEAD-CODE-WARNING'`
Expected: `NO-DEAD-CODE-WARNING` (removing the allow did not surface a warning, because the field is genuinely used).

- [ ] **Step 4: Commit**

```bash
git add src/py_query.rs
git commit -m "refactor: drop obsolete allow(dead_code) on PyContigReader::inner

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Remove bare annotations shadowing `cached_property` in `SparseVar`

**Files:**
- Modify: `python/genoray/_svar.py` (class-body annotations ~503-509 and the `index` `cached_property` ~529-532)

**Interfaces:**
- Consumes: nothing.
- Produces: `SparseVar.index`, `._c_max_idxs`, `._is_biallelic` unchanged in behavior (still `cached_property`).

Three names (`index`, `_c_max_idxs`, `_is_biallelic`) are declared both as bare class-level annotations AND as `@cached_property`. The bare annotations are redundant (annotation-only, create no attribute) and misleading. Keep the other bare annotations (`genos`, `available_fields`, `fields`, `_c_norm`, `_s2i`) — those are real `__init__`-set attributes.

- [ ] **Step 1: Delete the three shadowing annotations**

In `python/genoray/_svar.py`, in the `SparseVar` class body, delete these three annotation lines (and the docstring attached to `index`):
```python
    index: pl.DataFrame
    """Table of variants with columns: `CHROM`, `POS`, `REF`, `ALT`, `ILEN`, and any additional
    attributes specified in `attrs` on construction."""
```
```python
    _c_max_idxs: dict[str, int]
```
```python
    _is_biallelic: bool
```
Leave `genos`, `available_fields`, `fields`, `_c_norm`, `_s2i` in place.

- [ ] **Step 2: Preserve the richer docstring on the `index` property**

Update the `index` `cached_property` so its docstring carries the column description that was on the deleted annotation. Replace:
```python
    @cached_property
    def index(self) -> pl.DataFrame:
        """The full variant index, materialized on first access."""
        return self._index_lazy.collect()
```
with:
```python
    @cached_property
    def index(self) -> pl.DataFrame:
        """The full variant index, materialized on first access.

        Table of variants with columns ``CHROM``, ``POS``, ``REF``, ``ALT``, ``ILEN``,
        and any additional attributes specified in ``attrs`` on construction.
        """
        return self._index_lazy.collect()
```

- [ ] **Step 3: Verify SparseVar still imports, type-checks, and behaves**

Run:
```bash
pixi run python -c "from genoray import SparseVar; print(SparseVar.index.__doc__.splitlines()[0])"
pixi run pytest -q tests/test_svar.py
```
Expected: prints `The full variant index, materialized on first access.`; `test_svar.py` passes.

- [ ] **Step 4: Commit**

```bash
git add python/genoray/_svar.py
git commit -m "refactor: drop bare annotations shadowing SparseVar cached_property

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Fix VCF filter ignored on the no-index Genos*Dosages path (bug B1)

**Files:**
- Test: `tests/test_vcf.py` (add one test)
- Modify: `python/genoray/_vcf.py` — `_fill_genos_and_dosages` (~1416-1472)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `VCF(path, phasing, dosage_field, with_gvi_index, filter, pl_filter)`, `VCF.read(contig, start, end, mode)`, `VCF.Genos8`, `VCF.Genos8Dosages`, `genoray.exprs.is_snp`.
- Produces: nothing new; corrects behavior of `read(..., mode=Genos*Dosages)` under a filter with no index.

`_fill_genos` and `_fill_dosages` apply `self._filter` at the top before the `out is None` / `out is not None` split. `_fill_genos_and_dosages` applies it only inside the `out is not None` branch, so a filtered `Genos*Dosages` read with **no `.gvi` index** (→ `out is None`) silently returns unfiltered data.

- [ ] **Step 1: Write the failing regression test**

Add to `tests/test_vcf.py` (the module already has `ddir = tdir / "data"` and imports `VCF`; add `from genoray import exprs` if not present):
```python
def test_filter_applied_to_genos_dosages_without_index():
    # A record filter + NO .gvi index forces the out-is-None path of
    # _fill_genos_and_dosages, where the filter was previously dropped.
    vcf = VCF(
        ddir / "biallelic.vcf.gz",
        phasing=False,
        dosage_field="DS",
        with_gvi_index=False,
        filter=lambda rec: len(rec.REF) == 1 and all(len(a) == 1 for a in rec.ALT),
        pl_filter=exprs.is_snp,
    )
    cse = ("chr1", 81261, 81266)  # covers the GAT>A indel and two SNPs
    genos_only = vcf.read(*cse, VCF.Genos8)          # filters correctly today
    genos, _dosages = vcf.read(*cse, VCF.Genos8Dosages)
    # The genotypes from the combined mode must match the genos-only mode:
    # both must have the SNP filter applied (indel dropped).
    np.testing.assert_array_equal(genos, genos_only)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest -q "tests/test_vcf.py::test_filter_applied_to_genos_dosages_without_index"`
Expected: FAIL — shapes differ on the last (variants) axis (combined mode returns 3 variants, genos-only returns 2), raising a mismatch in `assert_array_equal`.

- [ ] **Step 3: Apply the fix**

In `python/genoray/_vcf.py`, in `_fill_genos_and_dosages`, hoist the filter to the top of the method so it runs before the `out is None` branch. Change the opening from:
```python
    ) -> tuple[Genos8Dosages | Genos16Dosages, int]:
        if out is None:
```
to:
```python
    ) -> tuple[Genos8Dosages | Genos16Dosages, int]:
        if self._filter is not None:
            vcf = filter(self._filter, vcf)

        if out is None:
```
Then delete the now-duplicate filter block that currently sits inside the `out is not None` branch (just after `n_variants = out[0].shape[-1]`):
```python
        if self._filter is not None:
            vcf = filter(self._filter, vcf)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run pytest -q "tests/test_vcf.py::test_filter_applied_to_genos_dosages_without_index"`
Expected: PASS.

- [ ] **Step 5: Guard against regressions in the other two modes**

Run: `pixi run pytest -q tests/test_vcf.py`
Expected: PASS (genos-only and dosages-only paths unchanged).

- [ ] **Step 6: Note the behavior change in the changelog**

Add an entry under the current unreleased/next-version section of `CHANGELOG.md`:
```markdown
### Fixed
- `VCF.read(..., mode=Genos*Dosages)` now applies a configured `filter` when no
  `.gvi` index is loaded, matching the genotype-only and dosage-only modes
  (previously the filter was silently ignored on this path).
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_vcf.py python/genoray/_vcf.py CHANGELOG.md
git commit -m "fix(vcf): apply filter on no-index Genos*Dosages read path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Fix scalar-string sample crash in `read_ranges_with_length` (bug B2)

**Files:**
- Test: `tests/test_svar.py` (add one test)
- Modify: `python/genoray/_svar.py` — `read_ranges_with_length` (~911-916)

**Interfaces:**
- Consumes: `SparseVar(path, "AF")`, `SparseVar.read_ranges_with_length(contig, starts, ends, samples)`.
- Produces: nothing new; corrects `samples=<bare str>` handling.

At lines ~911-916 the `set(samples)` membership check runs on the **raw** argument before `np.atleast_1d(np.array(...))`, so a single sample passed as a bare `str` (`"sample1"`) is iterated character-by-character and raises a spurious "not found". The sibling methods (`_find_starts_ends`, `_find_starts_ends_with_length`, `read_ranges`) coerce to an array first. This is a one-site fix — do **not** refactor to `_normalize_samples` (that is SP-1's job).

- [ ] **Step 1: Write the failing regression test**

Add to `tests/test_svar.py` (module has `ddir` and `from genoray import SparseVar`):
```python
def test_read_ranges_with_length_accepts_scalar_sample_string():
    svar = SparseVar(ddir / "biallelic.vcf.svar", "AF")
    # A single sample name as a bare str must not be iterated char-by-char.
    out_str = svar.read_ranges_with_length("chr1", 81261, 81266, samples="sample1")
    out_list = svar.read_ranges_with_length("chr1", 81261, 81266, samples=["sample1"])
    np.testing.assert_array_equal(out_str.data, out_list.data)
    np.testing.assert_array_equal(out_str.offsets, out_list.offsets)
```
(If `tests/test_svar.py` does not already `import numpy as np`, add it.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest -q "tests/test_svar.py::test_read_ranges_with_length_accepts_scalar_sample_string"`
Expected: FAIL — `ValueError: Samples {...} not found in the dataset.` (the bare string was split into characters).

- [ ] **Step 3: Apply the fix**

In `python/genoray/_svar.py`, in `read_ranges_with_length`, reorder the `else` branch so coercion precedes the membership check (matching `_find_starts_ends` at ~721-723). Change:
```python
        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")
            samples = np.atleast_1d(np.array(samples))
```
to:
```python
        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            samples = np.atleast_1d(np.array(samples))
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run pytest -q "tests/test_svar.py::test_read_ranges_with_length_accepts_scalar_sample_string"`
Expected: PASS.

- [ ] **Step 5: Verify the sample-subset behavior is unchanged elsewhere**

Run: `pixi run pytest -q tests/test_svar.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_svar.py python/genoray/_svar.py
git commit -m "fix(svar): accept bare-str sample in read_ranges_with_length

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Fix PGEN `__del__` double-close (bug B3)

**Files:**
- Test: `tests/test_pgen.py` (add one test)
- Modify: `python/genoray/_pgen.py` — `__del__` (~389-393)

**Interfaces:**
- Consumes: `PGEN(path)`.
- Produces: nothing new; corrects `__del__` to close the shared reader once.

When `dosage_path is None`, `self._dose_pgen = self._geno_pgen` (the same object). `__del__` calls `.close()` on `_geno_pgen` then on `_dose_pgen` — closing the same handle twice.

- [ ] **Step 1: Write the failing regression test**

Add to `tests/test_pgen.py` (module has `ddir` and `from genoray._pgen import PGEN` or `from genoray import PGEN` — match the existing import):
```python
def test_del_does_not_double_close_shared_reader():
    pgen = PGEN(ddir / "biallelic.pgen")  # no separate dosage path
    assert pgen._dose_pgen is pgen._geno_pgen  # same underlying reader

    class _CloseCounter:
        def __init__(self):
            self.n = 0

        def close(self):
            self.n += 1

    counter = _CloseCounter()
    pgen._geno_pgen = counter
    pgen._dose_pgen = counter
    pgen.__del__()
    assert counter.n == 1  # closed exactly once, not twice
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest -q "tests/test_pgen.py::test_del_does_not_double_close_shared_reader"`
Expected: FAIL — `assert 2 == 1` (the shared reader was closed twice).

- [ ] **Step 3: Apply the fix**

In `python/genoray/_pgen.py`, change `__del__` from:
```python
    def __del__(self):
        if hasattr(self, "_geno_pgen"):
            self._geno_pgen.close()
        if hasattr(self, "_dose_pgen") and self._dose_pgen is not None:
            self._dose_pgen.close()
```
to:
```python
    def __del__(self):
        if hasattr(self, "_geno_pgen"):
            self._geno_pgen.close()
        if (
            hasattr(self, "_dose_pgen")
            and self._dose_pgen is not None
            and self._dose_pgen is not self._geno_pgen
        ):
            self._dose_pgen.close()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run pytest -q "tests/test_pgen.py::test_del_does_not_double_close_shared_reader"`
Expected: PASS.

- [ ] **Step 5: Verify PGEN suite still passes**

Run: `pixi run pytest -q tests/test_pgen.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_pgen.py python/genoray/_pgen.py
git commit -m "fix(pgen): guard __del__ against double-closing shared reader

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Fix nrvk capacity-assert message + drop redundant mask (bug B4)

**Files:**
- Modify: `src/nrvk.rs` — `push_long_allele` (lines ~40-62)
- Test: `src/nrvk.rs` — add a `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: `LongAlleleTableWriter::new(tx_long: Sender<Vec<u8>>, buffer_capacity: usize)`, `push_long_allele(&mut self, alt_bytes: &[u8]) -> u32`.
- Produces: nothing new. `push_long_allele` returns the same values (the mask was a proven no-op given the assert).

The assert bounds `row_index <= 0x7FFFFFFF` (2³¹−1 = 2,147,483,647) but the message prints `4,294,967,295` (2³²−1 = `u32::MAX`, the wrong bound). The trailing `current_index & 0x7FFFFFFF` is dead — the assert already guarantees the high bit is clear. **This is a message/comment correctness fix + dead-code removal; there is no behavioral change, so no red→green test is possible.** The added unit test is a regression guard.

- [ ] **Step 1: Fix the assert message**

In `src/nrvk.rs`, change:
```rust
        assert!(
            self.row_index <= 0x7FFFFFFF,
            "Exceeded 31-bit (4,294,967,295) index capacity! Cannot proceed with this many long alleles!"
        );
```
to:
```rust
        assert!(
            self.row_index <= 0x7FFFFFFF,
            "Exceeded 31-bit (2,147,483,647) index capacity! Cannot proceed with this many long alleles!"
        );
```

- [ ] **Step 2: Drop the redundant mask**

Change the return (and its comment) from:
```rust
        // return strictly the 31 LSBs (masking out the 32nd bit just in case)
        current_index & 0x7FFFFFFF
```
to:
```rust
        // the assert above guarantees the high bit is clear, so return as-is
        current_index
    }
```
(Note: `current_index` is captured before `self.row_index += 1`, so it is the pre-increment row index — unchanged.)

- [ ] **Step 3: Add a regression-guard unit test**

Append to `src/nrvk.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    #[test]
    fn push_long_allele_returns_sequential_pre_increment_indices() {
        let (tx, _rx) = unbounded::<Vec<u8>>();
        let mut w = LongAlleleTableWriter::new(tx, 1024);
        assert_eq!(w.push_long_allele(b"ACGT"), 0);
        assert_eq!(w.push_long_allele(b"TTTT"), 1);
        assert_eq!(w.push_long_allele(b"GG"), 2);
    }
}
```

- [ ] **Step 4: Run the Rust test**

Run: `pixi run bash -lc 'cargo test --no-default-features nrvk'`
Expected: the new `push_long_allele_returns_sequential_pre_increment_indices` test PASSES; no other nrvk tests regress.

- [ ] **Step 5: Confirm clippy is clean**

Run: `pixi run bash -lc 'cargo clippy --no-default-features -- -D warnings 2>&1 | tail -5'`
Expected: no warnings/errors introduced by the change.

- [ ] **Step 6: Commit**

```bash
git add src/nrvk.rs
git commit -m "fix(nrvk): correct 31-bit capacity message; drop redundant mask

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Rewrite the broken PGEN-filter docs (C1)

**Files:**
- Modify: `README.md` (Filtering section, ~124-148)
- Modify: `docs/source/index.md` (Filtering section — the PGEN block, ~135-160)

**Interfaces:**
- Consumes: real `.gvi` schema (`CHROM`, `POS`, `REF`, `ALT` List[Utf8], `ILEN` List[Int32]) and `genoray.exprs` names (`is_snp`, `is_indel`, `is_biallelic`, `is_symbolic`, `is_breakend`, `is_imprecise`, `ILEN`).
- Produces: runnable examples.

Both docs describe a nonexistent PGEN-filter schema (`Chromosome`/`Start`/`End`/`ilen`/`kind`) and give an example (`pl.col("kind")...`) that raises at query time. The real index columns are `CHROM`/`POS`/`REF`/`ALT`/`ILEN`, and `is_snp`/`is_indel` are derived from `ILEN`.

- [ ] **Step 1: Rewrite the PGEN paragraph + example in `README.md`**

Replace the block in `README.md` that starts at `For PGENs, the expression will operate on a polars DataFrame ...` and ends after the ` ```python ... ``` ` SNP example (the `Chromosome`/`Start`/`End`/`ilen`/`kind` list and the `pl.col("kind")...` example) with:
```markdown
For PGENs, the expression operates on the `.gvi` index — a polars DataFrame with columns:
- `CHROM` — contig name
- `POS` — 1-based position
- `REF` — reference allele
- `ALT` — list of alternate alleles
- `ILEN` — list of indel lengths (one per ALT: `len(ALT) - len(REF)`, or a signed size for symbolic SVs; `null` for un-sizable symbolic/breakend alleles)

Prefer the ready-made expressions in `genoray.exprs` — `is_snp`, `is_indel`, `is_biallelic`, `is_symbolic`, `is_breakend`, `is_imprecise`, and `ILEN` — and combine them with polars operators. For custom predicates, use `pl.col("CHROM"/"POS"/"REF"/"ALT"/"ILEN")` directly.

```python
import genoray
from genoray import PGEN

# only include SNPs
pgen = PGEN("file.pgen", filter=genoray.exprs.is_snp)

# exclude symbolic alleles and breakends
pgen = PGEN("file.pgen", filter=~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend)
```
```

- [ ] **Step 2: Rewrite the matching PGEN paragraph + example in `docs/source/index.md`**

Apply the identical replacement to `docs/source/index.md` (same stale `Chromosome`/`Start`/`End`/`ilen`/`kind` list and `pl.col("kind")...` example → the same corrected markdown block from Step 1).

- [ ] **Step 3: Verify the new example actually runs**

Run:
```bash
pixi run python -c "
import genoray
from genoray import PGEN
from pathlib import Path
ddir = Path('tests/data')
pgen = PGEN(ddir / 'biallelic.pgen', filter=genoray.exprs.is_snp)
print('is_snp filter OK; n samples =', pgen.n_samples)
pgen = PGEN(ddir / 'biallelic.pgen', filter=~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend)
print('symbolic/breakend filter OK')
"
```
Expected: prints both OK lines with no exception. (If `biallelic.pgen` is absent, first run `pixi run bash -lc 'cd tests/data && ./gen_from_vcf.sh'` to regenerate test data.)

- [ ] **Step 4: Commit**

```bash
git add README.md docs/source/index.md
git commit -m "docs: fix stale PGEN filter schema in README and index

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: String hygiene — TypeGuard typo, wrong-backend logs, orphaned docstring (C2, C3, C4)

**Files:**
- Modify: `python/genoray/_vcf.py:140` (TypeGuard typo) and `:36-37` (orphaned docstring)
- Modify: `python/genoray/_pgen.py:566,652,735,862` (log strings)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (internal type annotation + log/doc text only; no public name touched).

- [ ] **Step 1: Fix the `TypeGuard` copy-paste typo (C2)**

In `python/genoray/_vcf.py`, the `_is_genos16_dosages` predicate is annotated with `Genos8`; it should be `Genos16`. Change:
```python
def _is_genos16_dosages(obj: object) -> TypeGuard[tuple[Genos8, Dosages]]:
```
to:
```python
def _is_genos16_dosages(obj: object) -> TypeGuard[tuple[Genos16, Dosages]]:
```
Confirm the function **body** already validates the correct dtype (this is a type-annotation-only fix — no runtime change).

- [ ] **Step 2: Fix the wrong-backend log messages in PGEN (C3)**

In `python/genoray/_pgen.py`, four methods log `"...not found in VCF file..."` inside the PGEN backend (lines ~566, 652, 735, 862). Replace all four occurrences of:
```python
                f"Query contig {contig} not found in VCF file, even after normalizing for UCSC/Ensembl nomenclature."
```
with:
```python
                f"Query contig {contig} not found in PGEN file, even after normalizing for UCSC/Ensembl nomenclature."
```
Verify: `grep -c "not found in VCF file" python/genoray/_pgen.py` → `0`; `grep -c "not found in PGEN file" python/genoray/_pgen.py` → `4`.

- [ ] **Step 3: Remove the orphaned floating docstring (C4)**

In `python/genoray/_vcf.py`, delete the orphaned triple-quoted string sitting above `V_IDX_TYPE` (it describes `int64`/CSI but the real constant is `uint32`, which already has its own docstring):
```python
"""Dtype for VCF range indices. This determines the maximum size of a contig in genoray.
We have to use int64 because this is what htslib uses for CSI indexes."""
```
Leave the real `V_IDX_TYPE = np.uint32` line and its own docstring intact.

- [ ] **Step 4: Verify type-check + suite**

Run:
```bash
pixi run ruff check python/genoray/_vcf.py python/genoray/_pgen.py
pixi run bash -lc 'pyrefly check python/genoray/_vcf.py' 2>&1 | tail -3
pixi run pytest -q tests/test_vcf.py tests/test_pgen.py
```
Expected: ruff clean; pyrefly reports no new errors on the `_is_genos16_dosages` annotation; tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_vcf.py python/genoray/_pgen.py
git commit -m "fix: correct Genos16 TypeGuard, PGEN log backend name, orphaned docstring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification (after all tasks)

- [ ] **Full Python suite:** `pixi run pytest -q -m "not network"` → PASS.
- [ ] **Full Rust suite:** `pixi run bash -lc 'cargo test --no-default-features'` → PASS.
- [ ] **Lint/format/type gates:** `pixi run bash -lc 'cargo fmt --check && cargo clippy --no-default-features -- -D warnings'` and `pixi run ruff check python tests` and `pixi run ruff format --check python tests` → all clean.
- [ ] **No public-API change:** `git diff 59fd540..HEAD -- python/genoray/__init__.py python/genoray/exprs.py` → no removed/renamed public (non-underscore) names; confirm no `SKILL.md` edit was needed.
- [ ] **Dead names gone:** `grep -rn "is_dtype\|POLARS_V_IDX_TYPE\|not found in VCF file" python/` and `test -f src/utils.rs` → no hits / file absent.
- [ ] **Roadmap sync:** confirm `docs/roadmap/clean-code-audit.md` marks SP-0 done and the scalar-oracle relocation reassigned to SP-7 (already recorded).

## Self-review notes (author)

- **Spec coverage:** every SP-0 spec item maps to a task — A1→T1, A2/A3→T2, A4→T3, A5→T4, B1→T5, B2→T6, B3→T7, B4→T8, C1→T9, C2/C3/C4→T10. The scalar-classifier item is intentionally absent (reassigned to SP-7 per the spec correction).
- **No placeholders:** every code/edit step shows the exact before/after; every run step gives the exact command and expected result.
- **Type/name consistency:** test helpers reuse the real APIs verified in-repo (`VCF.read(contig,start,end,mode)`, `VCF.Genos8`/`Genos8Dosages`, `SparseVar(path,"AF")`, `read_ranges_with_length`, `LongAlleleTableWriter::new`); the record-filter lambda mirrors `tests/test_svar_filtering.py`.

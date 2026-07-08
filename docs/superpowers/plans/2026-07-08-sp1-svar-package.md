# SP-1 `_svar.py` → `_svar/` Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 3,283-line `python/genoray/_svar.py` into a focused `genoray/_svar/` package, collapse the two near-duplicate `from_vcf`/`from_pgen` writers into one shared body, and converge four hand-rolled sample-validation sites onto a single behavior-preserving helper — with zero public-API change.

**Architecture:** Move cohesive symbol groups into `_svar/{_regions,_convert,_io,_kernels,_annotate,_core}.py` along lifecycle seams; `_svar/__init__.py` re-exports `SparseVar`, `SparseVarMetadata`, `dense2sparse` so both `genoray.SparseVar` (via the `_LAZY` map) and the internal `genoray._svar.SparseVar` path keep resolving with a zero diff to `genoray/__init__.py`. Because this is code-motion, the safety net is the **existing** test suite staying green (characterization), not new red-green tests — except the one new helper (Task 3), which gets its own targeted tests.

**Tech Stack:** Python 3.10+, NumPy, Polars, Numba (`@nb.njit(cache=True)`), pydantic, hirola `HashTable`, joblib; Pixi for env/tasks; pytest; ruff + pyrefly + cargo hooks via pre-commit/pre-push.

## Global Constraints

- **Behavior-preserving.** No change to any public method's inputs, outputs, shapes, dtypes, or raised errors. Copy `ValueError` message text verbatim.
- **Zero public-API diff → no SKILL update.** `git diff python/genoray/__init__.py` must be empty; `skills/genoray-api/SKILL.md` untouched. Verify at the end.
- **Leaf modules (`_regions`, `_convert`, `_io`, `_kernels`, `_annotate`) must NOT import `_core`** (no import cycles). `_core` imports from the leaves.
- **Line numbers in this plan are current-tree orientation refs, not contracts.** Re-locate every symbol by name before moving it (earlier moves shift the file).
- **Commit convention:** Conventional Commits. End each commit message with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Test commands (run inside pixi):**
  - Full Python suite: `pixi run pytest -q`
  - Single file/test: `pixi run pytest tests/test_svar.py -q`
  - Rust (only needed for the final sweep): `pixi run bash -lc 'cargo test --no-default-features --features conversion'`
- **Pre-push hooks must be installed:** `pixi run prek-install` (one-time; `.pre-commit-config.yaml` is present, so prek hooks are mandatory before committing/pushing).

---

## File Structure

New package `python/genoray/_svar/` replacing the module `python/genoray/_svar.py`:

- `_svar/__init__.py` — re-export shim: `SparseVar`, `SparseVarMetadata`, `dense2sparse` + `__all__`.
- `_svar/_regions.py` — region/sample/field normalization + row/var-idx resolution: `_coerce_bed_schema`, `_normalize_regions`, `_normalize_samples`, `_validate_fields`, `_resolve_kept_rows`, `_resolve_kept_var_idxs`, and (added in Task 3) `_resolve_sample_idxs`. Also owns `_REGION_STR_RE`.
- `_svar/_convert.py` — dense↔sparse + writers: `dense2sparse` (+ overloads), `_dense2sparse_with_length`, `_process_contig_vcf`, `_process_contig_pgen`, `_concat_data`, and (added in Task 2) `_write_from_reader`.
- `_svar/_io.py` — index + memmap I/O: `_write_filtered_index`, `_subset_var_idxs_and_recompute_af`, `_build_working_index`, `_write_index_from_working`, `_open_genos`, `_open_fmt`, `_write_genos`, `_write_dosages`.
- `_svar/_kernels.py` — all `@nb.njit` kernels: `_nb_af_helper`, `_nb_count_kept`, `_nb_count_mac_per_kept`, `_nb_write_var_idxs`, `_nb_write_field`, `_copy_chunk_helper`, `_copy_chunk_dosages_helper`, `_find_starts_ends`, `_length_walk_n_keep`, `_dense2sparse_count`, `_dense2sparse_fill`, `_find_starts_ends_with_length`.
- `_svar/_annotate.py` — `SparseVarAnnotateMixin` (methods `annotate_with_gtf`, `annotate_mutations`, `mutation_matrix`, `assign_signatures`, `cache_afs`, `_load_all_attrs`, `_compute_afs`, `_write_afs`) + module helpers `_empty_annot`, `_get_strand_and_codon_pos`, `_load_gtf`.
- `_svar/_core.py` — `SparseVarMetadata` (pydantic `BaseModel`), `SparseVar(Generic[_SRT], SparseVarAnnotateMixin)` (open + query + `write_view`).

Unchanged: `python/genoray/__init__.py`, `_svar2*.py`, `skills/genoray-api/SKILL.md`.

---

## Task 1: Split `_svar.py` into the `_svar/` package (pure code motion)

Move every symbol into its target file and delete the old module. One atomic commit — the package is only importable once all groups are in place. The gate is the full existing suite staying green plus an explicit import check.

**Files:**
- Create: `python/genoray/_svar/__init__.py`, `_svar/_regions.py`, `_svar/_convert.py`, `_svar/_io.py`, `_svar/_kernels.py`, `_svar/_annotate.py`, `_svar/_core.py`
- Delete: `python/genoray/_svar.py`
- Reference (do not edit): `python/genoray/__init__.py` (lines 33, 44, 65 reach `genoray._svar.SparseVar`)

**Interfaces:**
- Produces: package `genoray._svar` exporting `SparseVar`, `SparseVarMetadata`, `dense2sparse` (identical objects/signatures to today). `_core.SparseVar` inherits `_annotate.SparseVarAnnotateMixin`. Leaf modules expose the private helpers listed in File Structure for Tasks 2–3 to import.

- [ ] **Step 1: Snapshot green baseline**

Run: `pixi run pytest -q 2>&1 | tail -5`
Expected: all pass/xfail/skip as on `main` (record the pass count, e.g. `528 passed`). This is the characterization baseline every later step must match.

- [ ] **Step 2: Create the package skeleton and move leaf groups first**

Create `python/genoray/_svar/` and move symbols **bottom-up** (leaves before `_core`) so each file's imports resolve. For each new leaf file, copy the exact source of its symbols out of `_svar.py`, then add only the imports that file actually uses (subset the header at `_svar.py:1-45`).

`_svar/_kernels.py` — move the two `@nb.njit` blocks (currently `~2175-2303` and `~2808-3107`). Needs: `numba as nb`, `numpy as np`, and the dtype/`Ragged` imports each kernel references (`from ..._types import ...` → note the package is now one level deeper, so relative imports become `from .._types import ...`, `from .._utils import ...`, etc.).

`_svar/_io.py` — move `_write_filtered_index`, `_subset_var_idxs_and_recompute_af`, `_build_working_index`, `_write_index_from_working`, `_open_genos`, `_open_fmt`, `_write_genos`, `_write_dosages`. `_io` may import from `_kernels` (e.g. `_nb_*` used by AF recompute) — allowed (leaf→leaf, no cycle back to `_core`).

`_svar/_regions.py` — move `_REGION_STR_RE`, `_coerce_bed_schema`, `_normalize_regions`, `_normalize_samples`, `_validate_fields`, `_resolve_kept_rows`, `_resolve_kept_var_idxs`.

`_svar/_convert.py` — move `dense2sparse` (+ its `@overload`s), `_dense2sparse_with_length`, `_process_contig_vcf`, `_process_contig_pgen`, `_concat_data`. May import from `_kernels`, `_io`, `_regions`.

`_svar/_annotate.py` — move `_empty_annot`, `_get_strand_and_codon_pos`, `_load_gtf` as module functions, and wrap the four public annotate methods + `cache_afs`/`_load_all_attrs`/`_compute_afs`/`_write_afs` into `class SparseVarAnnotateMixin:` (copy the method bodies verbatim, keeping `self.` references — they resolve against the concrete `SparseVar` at runtime). Do not add a Protocol/host type (SP-6 concern).

- [ ] **Step 3: Move `_core` and wire inheritance**

Create `_svar/_core.py`: move `SparseVarMetadata` and the `SparseVar` class. Change the class declaration to inherit the mixin:

```python
from ._annotate import SparseVarAnnotateMixin

class SparseVar(SparseVarAnnotateMixin, Generic[_SRT]):
    ...
```

Add `_core`'s imports: pull the moved helpers from their new homes, e.g.

```python
from ._regions import (
    _coerce_bed_schema, _normalize_regions, _normalize_samples,
    _validate_fields, _resolve_kept_rows, _resolve_kept_var_idxs,
)
from ._convert import dense2sparse, _process_contig_vcf, _process_contig_pgen, _concat_data
from ._io import (
    _build_working_index, _write_index_from_working, _write_filtered_index,
    _subset_var_idxs_and_recompute_af, _open_genos, _open_fmt, _write_genos, _write_dosages,
)
from ._kernels import _find_starts_ends, _find_starts_ends_with_length  # + any others used by methods
```

Keep the third-party imports `_core` still needs (`joblib`, `joblib_progress`, `rich.progress`, `tqdm`, `HashTable`, `PGEN`, `VCF`, `Reference`, mutcat/signature imports used by the class or mixin). If a symbol ends up used only by the mixin, import it in `_annotate.py` instead.

- [ ] **Step 4: Write the re-export shim**

Create `python/genoray/_svar/__init__.py`:

```python
from ._convert import dense2sparse
from ._core import SparseVar, SparseVarMetadata

__all__ = ["SparseVar", "SparseVarMetadata", "dense2sparse"]
```

- [ ] **Step 5: Delete the old module**

Run: `git rm python/genoray/_svar.py`
Then remove any stale bytecode/numba cache: `rm -rf python/genoray/__pycache__`

- [ ] **Step 6: Import guard**

Run:
```bash
pixi run python -c "import genoray; from genoray._svar import SparseVar, SparseVarMetadata, dense2sparse; from genoray import SparseVar as S; assert S is SparseVar; print('import OK', SparseVar)"
```
Expected: `import OK <class 'genoray._svar._core.SparseVar'>` with no ImportError/circular-import error.

- [ ] **Step 7: Run the full suite (characterization gate)**

Run: `pixi run pytest -q 2>&1 | tail -8`
Expected: identical pass/skip/xfail counts to Step 1. Numba kernels recompile once on first run (cache moved) — this only affects wall-time, not results. If anything fails, it is an import/motion error — fix the moved imports; do not change logic.

- [ ] **Step 8: Confirm zero public-API diff**

Run: `git diff --stat python/genoray/__init__.py skills/genoray-api/SKILL.md`
Expected: empty (both untouched).

- [ ] **Step 9: Lint/type and commit**

Run: `pixi run ruff check python/genoray/_svar && pixi run ruff format python/genoray/_svar`
Then:
```bash
git add python/genoray/_svar python/genoray/_svar.py
git commit -m "refactor: split _svar.py into _svar/ package

Pure code motion along lifecycle seams (_regions, _convert, _io, _kernels,
_annotate mixin, _core) with a re-export shim. No public-API or behavior
change; existing suite is the characterization gate.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(The pre-commit hooks — ruff/pyrefly — run here; they must pass. `git add python/genoray/_svar.py` stages the deletion.)

---

## Task 2: Extract `_write_from_reader` (collapse the duplicate writers)

`from_vcf` (`~978-1188`) and `from_pgen` (`~1189-1417`) share an identical spine; the divergent parts are labeled `# (mirrors from_vcf; keep in sync)` at `~1332` and `~1395`. Extract the spine into `_convert._write_from_reader`; both classmethods become thin wrappers. The existing write/parity tests are the gate. This is the one med-risk task — commit it alone.

**Files:**
- Modify: `python/genoray/_svar/_convert.py` (add `_write_from_reader`)
- Modify: `python/genoray/_svar/_core.py` (`from_vcf`, `from_pgen` → wrappers)
- Test (existing gate): `tests/test_svar.py` (write/round-trip + subset tests) and any `test_svar_*` covering `from_vcf`/`from_pgen`

**Interfaces:**
- Produces: `_write_from_reader(*, out, reader, samples, max_mem, overwrite, n_jobs, make_contig_task, ploidy, ...) -> None` in `_convert.py`. The reader-specific behavior is injected via `make_contig_task`, a callable the wrapper supplies that returns the per-contig joblib task (wrapping `_process_contig_vcf` / `_process_contig_pgen`). Exact kwarg list is dictated by reading the two current bodies — keep every value both writers compute.
- Consumes: `_process_contig_vcf`, `_process_contig_pgen`, `_concat_data` (from `_convert`); `_build_working_index`, `_write_index_from_working`, `_subset_var_idxs_and_recompute_af` (from `_io`); `_resolve_kept_rows`, `_normalize_samples` (from `_regions`).

- [ ] **Step 1: Establish the writer baseline**

Run: `pixi run pytest tests/test_svar.py -q 2>&1 | tail -5`
Expected: all green. Note which tests exercise `from_vcf`/`from_pgen` (grep: `pixi run pytest tests/ -q --collect-only 2>/dev/null | grep -iE 'from_vcf|from_pgen|write|round' | head`). These are the regression gate for this task.

- [ ] **Step 2: Read both writer bodies and diff them mentally**

Read `_core.py` `from_vcf` and `from_pgen` in full. Identify, line by line, the 11 shared spine steps (overwrite check; sample resolution; `_build_working_index`; `_resolve_kept_rows` + sort; per-contig keep-index bucketing; metadata write; up-front index write when not subsetting; `parse_memory`/job sizing; `TemporaryDirectory` + `joblib.Parallel`; `_concat_data`; subsetting MAC-drop finalize) and the exact points where they diverge (the per-contig task construction and reader-derived `samples`/`ploidy`). Write the divergent parts down before extracting.

- [ ] **Step 3: Write `_write_from_reader` in `_convert.py`**

Move the shared spine verbatim into a new module-level `_write_from_reader(...)`. Replace the two divergent points with calls to the injected `make_contig_task` callback and the passed-in reader-derived values. Do not alter control flow, the subsetting/non-subsetting index branch, or the MAC-drop finalize. Preserve the `joblib_progress` usage exactly (progress-bar unification is out of scope).

- [ ] **Step 4: Rewrite `from_vcf`/`from_pgen` as wrappers**

Each classmethod: keep its **exact current signature and return type** (both return `None`), build its reader + `make_contig_task` closure over `_process_contig_vcf`/`_process_contig_pgen`, then `return _write_from_reader(...)`. No signature change → no SKILL update.

- [ ] **Step 5: Run the writer gate**

Run: `pixi run pytest tests/test_svar.py -q 2>&1 | tail -5`
Expected: identical to Step 1 (all green). If a subset/MAC or contig-block test fails, the extraction dropped a spine step — re-diff against Step 2, do not patch the test.

- [ ] **Step 6: Full suite + lint**

Run: `pixi run pytest -q 2>&1 | tail -5 && pixi run ruff check python/genoray/_svar`
Expected: full suite matches Task 1 baseline; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_svar/_convert.py python/genoray/_svar/_core.py
git commit -m "refactor(svar): extract _write_from_reader from from_vcf/from_pgen

Collapse the two ~200-line near-duplicate writers (previously kept in sync by
hand) into one shared spine parameterized by a per-contig task callback.
Behavior-preserving; from_vcf/from_pgen signatures and None return unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Converge sample validation onto `_resolve_sample_idxs`

Four methods (`_find_starts_ends` `~689`, `_find_starts_ends_with_length` `~755`, `read_ranges` `~826`, `read_ranges_with_length` `~884`) hand-roll the same validate-then-index block. Replace with one helper that preserves order **and duplicates** (behavior-preserving — deliberately NOT `_normalize_samples`, which dedups). New tests guard the two invariants that matter: bare-`str` works (SP-0 bug) and duplicates are preserved.

**Files:**
- Modify: `python/genoray/_svar/_regions.py` (add `_resolve_sample_idxs`)
- Modify: `python/genoray/_svar/_core.py` (four call sites; import the helper)
- Test: `tests/test_svar.py` (add two tests)

**Interfaces:**
- Produces: `_resolve_sample_idxs(samples, available, s2i) -> tuple[NDArray, NDArray[np.int64]]` in `_regions.py`. `samples: ArrayLike | None`; `available: Sequence[str]`; `s2i: HashTable`. Returns `(names, s_idxs)` where `names` is the coerced name array (order + duplicates preserved) and `s_idxs = s2i[names]` cast to `int64`. `samples=None` → all `available`.
- Consumes: nothing new.

- [ ] **Step 1: Write the two failing/guard tests**

Add to `tests/test_svar.py` (adapt the fixture/sample name to the file's existing `SparseVar` fixture — grep an existing `read_ranges_with_length` test for the pattern and real sample IDs):

```python
def test_read_ranges_with_length_bare_str_sample(small_svar):
    # SP-0 regression: a single sample as a bare str must be one sample, not
    # iterated character-by-character.
    sample = small_svar.available_samples[0]
    out = small_svar.read_ranges_with_length("chr1", 0, 1_000_000, samples=sample)
    assert out.shape[2] == 1  # (..., samples, ...) axis == 1

def test_read_ranges_preserves_duplicate_samples(small_svar):
    # Behavior-preserving invariant: duplicates are NOT deduped on the read path.
    s = small_svar.available_samples[0]
    out = small_svar.read_ranges("chr1", 0, 1_000_000, samples=[s, s])
    assert out.shape[2] == 2  # two identical sample columns preserved
```

(If `test_read_ranges_with_length_bare_str_sample` already exists from SP-0, keep the existing one and add only the duplicate-samples test.)

- [ ] **Step 2: Run the tests against current (pre-refactor) code**

Run: `pixi run pytest tests/test_svar.py -k "bare_str_sample or preserves_duplicate" -q`
Expected: both PASS on the current code (current behavior already preserves duplicates and, post-SP-0, accepts bare str). These are **characterization/guard** tests — they lock the behavior the refactor must not break. (Adjust axis index / contig / coordinates if a shape assertion is off, until they pass on unchanged code.)

- [ ] **Step 3: Add `_resolve_sample_idxs` to `_regions.py`**

```python
def _resolve_sample_idxs(
    samples: "ArrayLike | None",
    available: Sequence[str],
    s2i: HashTable,
) -> tuple[NDArray, NDArray[np.int64]]:
    """Validate `samples` against `available`, preserving caller order AND
    duplicates, and return (name_array, integer_sample_indices).

    `None` selects all `available`. A bare `str` is coerced to a single-element
    array first, so it is treated as one sample name (not iterated per-character).
    """
    if samples is None:
        names = np.atleast_1d(np.asarray(available))
    else:
        names = np.atleast_1d(np.asarray(samples))
        if missing := set(names.tolist()) - set(available):
            raise ValueError(f"Samples {missing} not found in the dataset.")
    s_idxs = cast(NDArray[np.int64], s2i[names])
    return names, s_idxs
```

Add imports to `_regions.py` as needed: `from hirola import HashTable`, `from numpy.typing import ArrayLike, NDArray`, `from typing import cast`, `numpy as np`, `Sequence`. **Match the exact error message** used today (`f"Samples {missing} not found in the dataset."`).

- [ ] **Step 4: Replace the four inline blocks in `_core.py`**

At each of the four sites, replace the block

```python
if samples is None:
    samples = np.atleast_1d(np.array(self.available_samples))
else:
    samples = np.atleast_1d(np.array(samples))
    if missing := set(samples) - set(self.available_samples):  # type: ignore
        raise ValueError(f"Samples {missing} not found in the dataset.")
s_idxs = cast(NDArray[np.int64], self._s2i[samples])
```

with

```python
from ._regions import _resolve_sample_idxs  # top-of-module import, not inline
...
samples, s_idxs = _resolve_sample_idxs(samples, self.available_samples, self._s2i)
```

Note `_find_starts_ends_with_length` (`~858`) and `read_ranges_with_length` (`~913`) currently lack the `s_idxs = self._s2i[...]` line inline in the same place — verify each site keeps producing the same local variables (`samples`, `s_idxs`) it used downstream before you delete the old lines. Keep `read_ranges`'s subsequent call into `_find_starts_ends` as-is (its harmless re-validation is pre-existing; removing the double work is out of scope).

- [ ] **Step 5: Re-run the guard tests + full suite**

Run: `pixi run pytest tests/test_svar.py -k "bare_str_sample or preserves_duplicate" -q && pixi run pytest -q 2>&1 | tail -5`
Expected: the two guard tests PASS and the full suite matches the Task 1 baseline.

- [ ] **Step 6: Lint and commit**

Run: `pixi run ruff check python/genoray/_svar tests/test_svar.py`
```bash
git add python/genoray/_svar/_regions.py python/genoray/_svar/_core.py tests/test_svar.py
git commit -m "refactor(svar): converge sample validation onto _resolve_sample_idxs

Replace four hand-rolled validate-then-index blocks with one helper that
preserves order and duplicates (behavior-preserving; not _normalize_samples,
which dedups). Adds guard tests for bare-str and duplicate-sample inputs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Whole-branch verification sweep

No code changes — the final gate before PR. Confirms the full stack (including Rust and all hooks) and the public-surface invariant.

**Files:** none (verification only).

- [ ] **Step 1: Fix stale `self.var_table` docstrings (opportunistic, doc-only)**

In `_svar/_annotate.py`, the moved `annotate_with_gtf` docstring says "update self.var_table in-place" / describes a `varID` column, and `_get_strand_and_codon_pos`'s param is named `var_table`; the real attribute is `self.index`. Update the docstring references to `self.index` (doc-only, no runtime change). Grep to confirm none remain: `grep -rn "var_table" python/genoray/_svar` → only the helper param name if you choose to leave it; rename it to `index` for consistency if trivial.

- [ ] **Step 2: Full Python + Rust suites**

Run:
```bash
pixi run pytest -q 2>&1 | tail -5
pixi run bash -lc 'cargo test --no-default-features --features conversion' 2>&1 | tail -15
```
Expected: Python matches the Task 1 baseline; Rust green. (`--no-default-features` avoids the `_Py_Dealloc` link error; `--features conversion` is required or the integration tests fail to compile against `rust-htslib`.)

- [ ] **Step 3: All pre-commit/pre-push hooks**

Run: `pixi run bash -lc 'prek run --all-files'` (or `pre-commit run --all-files` if that is the configured runner)
Expected: ruff check, ruff format, pyrefly type-check, cargo fmt/check/clippy, commitizen — all pass.

- [ ] **Step 4: Public-surface guard**

Run:
```bash
git diff --stat origin/main -- python/genoray/__init__.py skills/genoray-api/SKILL.md
pixi run python -c "import genoray; from genoray._svar import SparseVar, SparseVarMetadata, dense2sparse; print('OK')"
```
Expected: empty diff on both files; `OK` printed. If either file shows a diff, a public name changed — stop and reconcile (SKILL update would be required, contradicting the spec's zero-diff promise).

- [ ] **Step 5: Line-count sanity**

Run: `wc -l python/genoray/_svar/*.py`
Expected: no single file near the old 3,283; total (minus the writer-dedup savings) roughly tracks the original. Confirms the decomposition actually happened.

- [ ] **Step 6: Commit the docstring fix (if Step 1 made changes) and open the PR**

```bash
git add python/genoray/_svar/_annotate.py
git commit -m "docs(svar): fix stale self.var_table references to self.index

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Then push and open a PR against `main` summarizing the three refactor commits + verification (Python suite counts, Rust green, empty public-API diff).

---

## Self-Review

**Spec coverage:**
- Package split along the six named seams → Task 1. ✓
- Re-export shim / zero `__init__.py` diff → Task 1 Steps 4, 8; Task 4 Step 4. ✓
- `_write_from_reader` extraction (med-risk, own commit) → Task 2. ✓
- Sample-validation convergence with behavior-preserving `_resolve_sample_idxs` (not dedup) → Task 3. ✓
- SP-0 bare-str regression preserved + duplicate-sample guard → Task 3 Steps 1–2. ✓
- Opportunistic `self.var_table` docstring fix → Task 4 Step 1. ✓
- No-import-cycle constraint → Global Constraints + Task 1 Step 2 ordering. ✓
- Numba cache recompile note → Task 1 Step 7. ✓
- Verification (pytest, cargo `--no-default-features --features conversion`, hooks, public-surface guard) → Task 4. ✓
- svar2 untouched / deferred items not in any task → confirmed absent. ✓

**Placeholder scan:** No TBD/TODO; every code step shows concrete code or an exact command. The writer extraction (Task 2) intentionally references the existing bodies to lift rather than reproducing ~400 lines — Step 2 requires reading them first, which is the correct move for a motion-extraction.

**Type consistency:** `_resolve_sample_idxs(samples, available, s2i) -> (names, s_idxs)` is named identically in the interface block, the implementation (Step 3), and the call site (Step 4). `_write_from_reader` and `make_contig_task` are used consistently across Task 2. `SparseVarAnnotateMixin` is named identically in File Structure, Task 1 Step 2/3.

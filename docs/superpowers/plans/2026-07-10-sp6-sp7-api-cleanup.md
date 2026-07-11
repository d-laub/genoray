# SP-6 + SP-7 API Cleanup & Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile genoray's public API (rename `SparseVar2.samples`, privatize the SVAR2 raw-dict FFI methods, drop the `Reader` alias, privatize exprs internals, rename a CLI flag), fix doc drift, split the two remaining god-modules (`_mutcat.py`, `_utils.py`), and land the coordinated genvarloader update — all as genoray **3.0.0**.

**Architecture:** Two Python source packages under `python/genoray/` (`_mutcat/`, plus new `_io.py`/`_contigs.py`/`_genos.py` split out of `_utils.py`), a lazy public `__init__.py`, a `cyclopts` CLI in `_cli/__main__.py`, a `pyo3`/`maturin` Rust extension (`_core`) that is **not** touched here, and a downstream `genvarloader` (gvl) that consumes genoray as a frozen wheel. Changes are behavior-preserving except three intentional breaks (sample-accessor rename, method privatization, CLI flag rename) that ride the 3.0.0 major bump.

**Tech Stack:** Python 3.10+, Pixi environment, `polars`, `numpy`, `numba`, `phantom-types`, `cyclopts` (CLI), `pytest`, `ruff`, `pyrefly` (type check via pre-commit), `commitizen` (version bump). gvl uses `pixi` + a frozen genoray wheel.

## Global Constraints

- **Package root is `python/genoray/`** (not `genoray/`). All paths below are repo-relative.
- **Run everything inside Pixi:** `pixi run pytest ...`, `pixi run ruff ...`. Never bare `pytest`.
- **Conventional Commits** required (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:`); commitizen checks it in a pre-commit hook. Breaking changes use `!` (e.g. `feat(api)!: ...`) or a `BREAKING CHANGE:` footer.
- **prek git hooks must be installed** before committing (`pixi run prek-install`). Hooks run ruff + pyrefly + cargo fmt/clippy + commitizen.
- **SKILL is mandatory:** any change to a name reachable via `import genoray` without underscores MUST update `skills/genoray-api/SKILL.md` in the same commit (per `CLAUDE.md`).
- **Test gate for every task:** the full genoray suite (`pixi run test`) stays green. `pixi run test` regenerates fixtures via `gen_from_vcf.sh` then runs pytest; for fast inner loops use `pixi run pytest <file>`.
- **No Rust changes.** The `_core` extension (`src/**/*.rs`) is out of scope; only Python wrappers, docs, and gvl change.
- **This branch:** `sp6-sp7-api-cleanup` (already created; the spec commit is on it).
- **Version bump is the LAST task**, after all code/docs/gvl are green: `2.15.0` → `3.0.0`.

---

## Phase 1 — SP-7 internal cleanup (no public break)

Behavior-preserving code motion + micro-refactors. These land first because they need no cross-repo coordination and de-risk later work. For pure code-motion tasks the regression gate is the existing suite, not a new unit test.

### Task 1: Relocate the scalar mutation-catalogue oracles to a test module

**Files:**
- Create: `tests/_mutcat_oracle.py`
- Modify: `python/genoray/_mutcat.py` (remove `_classify_variants_scalar` @689, `classify_sbs96` @191, `classify_id83` @248, `_microhomology_len` @305)
- Modify: `tests/test_mutcat.py` (imports @23-27), `tests/test_svar_mutations.py` (import @475)

**Interfaces:**
- Produces: `tests/_mutcat_oracle.py` exporting `_classify_variants_scalar(index: pl.DataFrame, reference: Reference) -> np.ndarray`, `classify_sbs96(five, ref, alt, three) -> int`, `classify_id83(pos, ref, alt, fetch) -> int`, `_microhomology_len(indel, downstream, ilen) -> int`.
- Consumes: `classify_dbs78`, `Sentinel`, `_REF_MISMATCH`, codebook constants — imported back from `genoray._mutcat` (they stay shipped).

- [ ] **Step 1: Verify the relocation set is genuinely test-only**

Run: `pixi run bash -lc 'grep -rn "classify_sbs96\|classify_id83\|_microhomology_len\|_classify_variants_scalar" python/ | grep -v "def "'`
Expected: the ONLY non-`def` hits inside `python/` are calls from *within* `_classify_variants_scalar` (`_mutcat.py:718,727` and its `_microhomology_len` use via `classify_id83`). No other production caller. `classify_dbs78` must NOT be in this set (it is called at `_mutcat.py:534` by `_build_dbs_table` and stays).

- [ ] **Step 2: Create the oracle module**

Move the four functions verbatim from `_mutcat.py` into `tests/_mutcat_oracle.py`. Preserve their bodies exactly (they are the ground-truth oracle). Add the imports they need at the top:

```python
"""Scalar, per-record mutation-catalogue classifiers — the slow ground-truth
oracle used only by tests to validate the shipped vectorized path
(``genoray._mutcat.classify_variants``). Relocated out of shipped code in SP-7.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from genoray._mutcat import (
    Sentinel,
    _REF_MISMATCH,
    classify_dbs78,
    # plus any codebook constants the moved bodies reference, e.g. SBS96_INDEX,
    # DBS78_INDEX, ID83 offsets — copy the exact import list the functions use.
)
from genoray._reference import Reference
```

Keep the four function bodies unchanged. `_classify_variants_scalar` keeps calling `classify_sbs96`/`classify_id83`/`_microhomology_len` (now local) and `classify_dbs78` (imported).

- [ ] **Step 3: Delete the four functions from `_mutcat.py`**

Remove `classify_sbs96`, `classify_id83`, `_microhomology_len`, and `_classify_variants_scalar` from `python/genoray/_mutcat.py`. Leave `classify_dbs78` and everything else intact. If any import in `_mutcat.py` becomes unused after removal, delete it (ruff will flag).

- [ ] **Step 4: Repoint test imports**

In `tests/test_mutcat.py`, change the imports at lines ~23-27 that pull `_classify_variants_scalar`, `classify_sbs96`, `classify_id83`, `_microhomology_len` from `genoray._mutcat` to import them from `_mutcat_oracle` instead. Keep importing `classify_dbs78`, `Sentinel`, `_REF_MISMATCH`, `DBS78_INDEX`, etc. from `genoray._mutcat`. In `tests/test_svar_mutations.py:475`, change `from genoray._mutcat import classify_sbs96, classify_variants` to import `classify_sbs96` from `_mutcat_oracle` and `classify_variants` from `genoray._mutcat`.

Note on test import resolution: tests already import sibling helpers (confirm `tests/` is on `sys.path` via `conftest.py` or `pytest` rootdir — genoray's existing tests import fixtures as top-level modules, so `import _mutcat_oracle` / `from _mutcat_oracle import ...` resolves the same way).

- [ ] **Step 5: Run the affected tests**

Run: `pixi run pytest tests/test_mutcat.py tests/test_svar_mutations.py -q`
Expected: PASS (same assertions, oracle now imported from the test module).

- [ ] **Step 6: Run the full suite**

Run: `pixi run test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_mutcat.py tests/_mutcat_oracle.py tests/test_mutcat.py tests/test_svar_mutations.py
git commit -m "refactor(mutcat): relocate scalar classifier oracles to tests"
```

### Task 2: Split `_mutcat.py` into a `_mutcat/` package

**Files:**
- Create: `python/genoray/_mutcat/__init__.py`, `python/genoray/_mutcat/codebook.py`, `python/genoray/_mutcat/classify.py`, `python/genoray/_mutcat/count.py`
- Delete: `python/genoray/_mutcat.py`
- Verify importers unaffected: `python/genoray/_svar/_annotate.py`, `python/genoray/_signatures.py`, `tests/*`, `tests/_mutcat_oracle.py`

**Interfaces:**
- Produces: `genoray._mutcat` package whose `__init__.py` re-exports the **exact same public-to-package names** the old module exposed, so every existing `from genoray._mutcat import X` keeps working: `Sentinel`, `Kind`, `MUTCAT_VERSION`, `SBS96`, `DBS78`, `ID83`, `SBS96_INDEX`, `DBS78_INDEX`, `N_CODES`, `code_ranges`, `labels`, `classify_dbs78`, `classify_variants`, `build_entry_codes`, `count_matrix`, `_REF_MISMATCH`, `_build_dbs_table`, and any other name currently imported elsewhere.

- [ ] **Step 1: Inventory the names other modules import from `_mutcat`**

Run: `pixi run bash -lc 'grep -rn "from .*_mutcat import\|from genoray._mutcat import\|_mutcat\." python/ tests/ | grep -v "def "'`
Expected: a list of every symbol imported from `_mutcat`. This is the exact set `_mutcat/__init__.py` must re-export. Record it.

- [ ] **Step 2: Create the package skeleton with three modules**

Split the surviving contents of `_mutcat.py` (post-Task-1) by responsibility, per audit finding 03:
- `codebook.py`: label lists, offsets, `code_ranges`, `labels`, `Sentinel`, `Kind`, `MUTCAT_VERSION`, `SBS96`/`DBS78`/`ID83` and their `*_INDEX` maps, `N_CODES`.
- `classify.py`: numpy/numba LUT builders + vectorized kernels + `classify_variants` + `build_entry_codes` + `classify_dbs78` + `_build_dbs_table` + `_REF_MISMATCH`.
- `count.py`: `count_matrix` and DataFrame assembly.

Within-package imports use relative form (`from .codebook import Sentinel`). Move code verbatim; do not change logic.

- [ ] **Step 3: Write `__init__.py` re-export shim**

```python
"""Mutation-catalogue codebooks, vectorized classifiers, and count matrices.

Package-internal API (no ``import genoray`` reach). Re-exported flat so existing
``from genoray._mutcat import X`` sites are unaffected by the SP-7 split.
"""
from .codebook import (
    DBS78, DBS78_INDEX, ID83, Kind, MUTCAT_VERSION, N_CODES, SBS96,
    SBS96_INDEX, Sentinel, code_ranges, labels,
)
from .classify import (
    _REF_MISMATCH, _build_dbs_table, build_entry_codes, classify_dbs78,
    classify_variants,
)
from .count import count_matrix

__all__ = [  # match the Step 1 inventory exactly
    "DBS78", "DBS78_INDEX", "ID83", "Kind", "MUTCAT_VERSION", "N_CODES",
    "SBS96", "SBS96_INDEX", "Sentinel", "code_ranges", "labels",
    "_REF_MISMATCH", "_build_dbs_table", "build_entry_codes", "classify_dbs78",
    "classify_variants", "count_matrix",
]
```

Adjust the lists to the Step 1 inventory (add any name that was imported elsewhere; drop any that wasn't and isn't needed cross-module).

- [ ] **Step 4: Delete the old module**

Run: `git rm python/genoray/_mutcat.py`

- [ ] **Step 5: Verify imports resolve**

Run: `pixi run python -c "import genoray._mutcat as m; import genoray._signatures; from genoray._svar import _annotate; import _mutcat_oracle" 2>&1 | tail -5`
(run the `_mutcat_oracle` import from the `tests/` dir or add it to the path)
Expected: no ImportError.

- [ ] **Step 6: Run the full suite**

Run: `pixi run test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_mutcat/ python/genoray/_mutcat.py
git commit -m "refactor(mutcat): split _mutcat.py into codebook/classify/count package"
```

### Task 3: Split `_utils.py` into `_io.py`, `_contigs.py`, `_genos.py`

**Files:**
- Create: `python/genoray/_io.py`, `python/genoray/_contigs.py`, `python/genoray/_genos.py`
- Modify: `python/genoray/_utils.py` (keep memory/dtype/thread helpers only)
- Modify: every importer of the moved names (`_vcf.py`, `_pgen.py`, `_svar/*.py`, `_var_ranges.py`, `_reference.py`, `_svar2*.py`, CLI, tests)

**Interfaces:**
- Produces:
  - `genoray._io`: `atomic_write_path`, `atomic_write_dir`, `_unique_sibling`.
  - `genoray._contigs`: `ContigNormalizer`.
  - `genoray._genos`: `hap_ilens`.
  - `genoray._utils` (residual): memory parsing/formatting, `np_to_pl_dtype`, `variant_file_type`, thread resolution, the numba-thread contextmanager.
- Decision: `hap_ilens` is consumed by both `_vcf.py` and `_pgen.py`; there is no single genotype module, so it lands in a dedicated `_genos.py` (genotype-domain indel-length math) rather than being duplicated or left in the junk drawer.

- [ ] **Step 1: Inventory importers of the moving names**

Run: `pixi run bash -lc 'grep -rn "ContigNormalizer\|hap_ilens\|atomic_write_path\|atomic_write_dir\|_unique_sibling" python/ tests/ | grep import'`
Expected: the full list of import sites to repoint. Record it.

- [ ] **Step 2: Move `ContigNormalizer` → `_contigs.py`**

Move the class verbatim (with its imports). While here, apply finding 03's precompute (this is Task 6 territory — leave the O(n²) code as-is for now to keep this task pure motion; Task 6 optimizes it).

- [ ] **Step 3: Move the atomic-write group → `_io.py`**

Move `_unique_sibling` (@231), `atomic_write_path` (@247), `atomic_write_dir` (@271) verbatim with their imports.

- [ ] **Step 4: Move `hap_ilens` → `_genos.py`**

Move `hap_ilens` (@117) verbatim with its imports (numba, numpy).

- [ ] **Step 5: Repoint every importer**

For each site from Step 1, change the import module:
- `from ._utils import ContigNormalizer` → `from ._contigs import ContigNormalizer`
- `from ._utils import hap_ilens` → `from ._genos import hap_ilens` (in `_vcf.py:31`, `_pgen.py:27`)
- `from ._utils import atomic_write_...` → `from ._io import atomic_write_...`

Leave memory/dtype/thread-helper imports pointing at `_utils`.

- [ ] **Step 6: Verify imports + run suite**

Run: `pixi run python -c "import genoray; from genoray import VCF, PGEN, SparseVar, SparseVar2"` then `pixi run test`
Expected: no ImportError; suite PASS.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_utils.py python/genoray/_io.py python/genoray/_contigs.py python/genoray/_genos.py python/genoray/*.py python/genoray/_svar/*.py
git commit -m "refactor(utils): split _utils.py into _io/_contigs/_genos by domain"
```

### Task 4: Extract `_var_end_expr()` and fix the shadowed local in `_var_ranges.py`

**Files:**
- Modify: `python/genoray/_var_ranges.py` (dup expr @67-70, 131-132, 186-187; shadow @97)

**Interfaces:**
- Produces: module-level `_var_end_expr() -> pl.Expr` returning `pl.col("POS") - pl.col("ILEN").list.first().clip(upper_bound=0).fill_null(0)`, reused by `var_ranges`, `var_indices`, `var_counts`.

- [ ] **Step 1: Add the helper**

```python
def _var_end_expr() -> pl.Expr:
    """0-based exclusive end for a variant. A null/absent ILEN (symbolic SV with
    no computable length) is treated as a point variant (end == POS); a negative
    ILEN (deletion) extends the span leftward. Shared by var_ranges/var_indices/
    var_counts so the symbolic-SV handling stays consistent across all three.
    """
    return pl.col("POS") - pl.col("ILEN").list.first().clip(upper_bound=0).fill_null(0)
```

- [ ] **Step 2: Replace the three Polars duplications**

At `_var_ranges.py:131-132` (`var_indices`) and `:186-187` (`var_counts`), replace the inlined expression with `_var_end_expr()`. For the numpy variant in `var_ranges` (@67-70), either express it via `_var_end_expr()` where a Polars expr is usable, or add a comment `# same rule as _var_end_expr()` if it must stay numpy — keep behavior identical.

- [ ] **Step 3: Rename the shadowing local**

At `_var_ranges.py:97`, rename the local `var_ranges = np.stack(...)` to `ranges = np.stack(...)` and update its later uses in the function body.

- [ ] **Step 4: Run the range tests then full suite**

Run: `pixi run pytest tests/ -k "var_range or var_count or var_idx or ranges" -q` then `pixi run test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_var_ranges.py
git commit -m "refactor(var-ranges): extract shared _var_end_expr, drop name shadowing"
```

### Task 5: `np_to_pl_dtype` lookup table + `variant_file_type` annotation

**Files:**
- Modify: `python/genoray/_utils.py` (`np_to_pl_dtype` @157, `variant_file_type` @145)

**Interfaces:**
- Produces: `variant_file_type(path) -> Literal["vcf", "pgen"] | None` (explicit `None` on unknown); `np_to_pl_dtype(dtype) -> type[pl.DataType]` backed by a module-level dict.

- [ ] **Step 1: Convert `np_to_pl_dtype` to a lookup**

Replace the if/elif ladder with a module-level table and a single miss error:

```python
_NP_TO_PL: dict[type[np.generic], type[pl.DataType]] = {
    np.int8: pl.Int8, np.int16: pl.Int16, np.int32: pl.Int32, np.int64: pl.Int64,
    np.uint8: pl.UInt8, np.uint16: pl.UInt16, np.uint32: pl.UInt32, np.uint64: pl.UInt64,
    np.float32: pl.Float32, np.float64: pl.Float64, np.bool_: pl.Boolean,
    # copy the EXACT set of branches the current ladder covers — do not add/drop mappings
}


def np_to_pl_dtype(dtype: DTypeLike) -> type[pl.DataType]:
    key = np.dtype(dtype).type
    try:
        return _NP_TO_PL[key]
    except KeyError:
        raise ValueError(f"Unsupported numpy dtype for polars conversion: {dtype!r}")
```

Mirror the ladder's exact coverage — read `_utils.py:157` first and transcribe every branch.

- [ ] **Step 2: Annotate `variant_file_type`**

Add `-> Literal["vcf", "pgen"] | None` to the signature and make the fall-through `return None` explicit. Add `from typing import Literal` if absent.

- [ ] **Step 3: Run the suite**

Run: `pixi run test`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add python/genoray/_utils.py
git commit -m "refactor(utils): table-driven np_to_pl_dtype, typed variant_file_type"
```

### Task 6: Precompute `ContigNormalizer.name_to_index`

**Files:**
- Modify: `python/genoray/_contigs.py` (the O(n²) remapper, was `_utils.py:42`)

**Interfaces:**
- Produces: `ContigNormalizer` builds `name_to_index = {c: i for i, c in enumerate(self.contigs)}` once in `__init__` and indexes into it instead of calling `self.contigs.index(c)` per entry. Public behavior unchanged.

- [ ] **Step 1: Replace the per-entry `list.index`**

In `__init__`, add `self._name_to_index = {c: i for i, c in enumerate(self.contigs)}` and rewrite the remapper dict comprehension to `{k: self._name_to_index[c] for k, c in self.contig_map.items()}`.

- [ ] **Step 2: Run the contig-normalization tests then full suite**

Run: `pixi run pytest tests/ -k "contig or normal" -q` then `pixi run test`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add python/genoray/_contigs.py
git commit -m "perf(contigs): precompute name_to_index, drop O(n^2) remapper build"
```

---

## Phase 2 — SP-6 public API breaks (genoray side)

Each task here is a public break riding the 3.0.0 bump. Every task that touches a public name updates `skills/genoray-api/SKILL.md` in the same commit.

### Task 7: Rename `SparseVar2.samples` → `available_samples`

**Files:**
- Modify: `python/genoray/_svar2.py` (`self.samples` @44; `n_samples` @55-56; `PyContigReader` construction @49), `python/genoray/_svar2_batch.py` (mixin attr decl @53; `_sample_idxs` @78,80)
- Modify: `tests/test_svar2_ranges.py` (`sv.samples` @74,76), any other test referencing `sv.samples`
- Modify: `skills/genoray-api/SKILL.md`

**Interfaces:**
- Produces: `SparseVar2.available_samples: list[str]` (was `samples`). `n_samples` unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_svar2.py` (or the nearest SVAR2 test module):

```python
def test_svar2_available_samples(svar2_store):
    sv = SparseVar2(str(svar2_store))
    assert isinstance(sv.available_samples, list)
    assert sv.n_samples == len(sv.available_samples)
    assert not hasattr(sv, "samples")  # renamed in 3.0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run pytest tests/test_svar2.py::test_svar2_available_samples -v`
Expected: FAIL (`AttributeError: 'SparseVar2' object has no attribute 'available_samples'`).

- [ ] **Step 3: Rename in the source**

In `_svar2.py`: `self.samples` → `self.available_samples` (@44); `len(self.samples)` → `len(self.available_samples)` in `n_samples` (@56) and `PyContigReader(... len(self.available_samples) ...)` (@49). In `_svar2_batch.py`: change the mixin attribute declaration `samples: list[str]` → `available_samples: list[str]` (@53) and both `self.samples` uses in `_sample_idxs` → `self.available_samples` (@78,80).

- [ ] **Step 4: Repoint tests referencing `sv.samples`**

Run: `pixi run bash -lc 'grep -rn "\.samples\b" tests/ | grep -i svar2'` and change each `sv.samples` → `sv.available_samples` (e.g. `tests/test_svar2_ranges.py:74,76`).

- [ ] **Step 5: Run tests to verify pass**

Run: `pixi run pytest tests/test_svar2.py tests/test_svar2_ranges.py -v`
Expected: PASS.

- [ ] **Step 6: Update SKILL**

In `skills/genoray-api/SKILL.md`, change SVAR2's sample accessor from `.samples` to `.available_samples` and note it is the canonical name shared with VCF/PGEN/SparseVar.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_svar2.py python/genoray/_svar2_batch.py tests/ skills/genoray-api/SKILL.md
git commit -m "feat(svar2)!: rename SparseVar2.samples to available_samples"
```

### Task 8: Privatize the SVAR2 raw-dict FFI methods

**Files:**
- Modify: `python/genoray/_svar2_batch.py` (`overlap_batch`→`_overlap_batch` @55; `find_ranges`→`_find_ranges` @104; `gather_ranges`→`_gather_ranges` @141; fix cross-references in docstrings @33,113,147,151)
- Modify: `tests/test_svar2_ranges.py`, `tests/test_svar2_errors.py`, `tests/test_py_ranges_readbound.py`, `tests/test_ranges_split.rs`-adjacent Python callers
- Modify: `skills/genoray-api/SKILL.md`

**Interfaces:**
- Produces: `SparseVar2._overlap_batch`, `SparseVar2._find_ranges`, `SparseVar2._gather_ranges` (underscored). Public SVAR2 query surface: `decode`, `region_counts`, `read_ranges`. The Rust `PyContigReader.overlap_batch/find_ranges/gather_ranges` calls **inside** these wrappers keep their names (Rust untouched).

- [ ] **Step 1: Write the failing test**

Add to the SVAR2 test module:

```python
def test_svar2_raw_methods_are_private(svar2_store):
    sv = SparseVar2(str(svar2_store))
    for public in ("overlap_batch", "find_ranges", "gather_ranges"):
        assert not hasattr(sv, public), f"{public} should be privatized in 3.0.0"
    for private in ("_overlap_batch", "_find_ranges", "_gather_ranges"):
        assert callable(getattr(sv, private))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run pytest tests/test_svar2.py::test_svar2_raw_methods_are_private -v`
Expected: FAIL (public names still present).

- [ ] **Step 3: Rename the three methods**

In `_svar2_batch.py`, rename the three `def`s to `_overlap_batch`/`_find_ranges`/`_gather_ranges`. Keep bodies and the internal `self._readers[contig].<name>(...)` calls unchanged. Update the docstring cross-references that name these methods (`RangesBundle` docstring @33 "replayed by `gather_ranges`", `find_ranges` docstring @113 "replayed by `gather_ranges`", `gather_ranges` docstring @147,151) to the underscored names.

- [ ] **Step 4: Repoint genoray's own test callers**

Run: `pixi run bash -lc 'grep -rln "\.overlap_batch\|\.find_ranges\|\.gather_ranges" tests/'` and, in each Python test, change `sv.overlap_batch`→`sv._overlap_batch`, `sv.find_ranges`→`sv._find_ranges`, `sv.gather_ranges`→`sv._gather_ranges`. (Rust test files call the Rust `PyContigReader`/`query` functions directly — leave `.rs` files unchanged unless they call the Python wrapper `pr.gather_ranges` on a `PyContigReader`, which is the Rust binding, also unchanged.)

- [ ] **Step 5: Run the SVAR2 range tests then full suite**

Run: `pixi run pytest tests/test_svar2_ranges.py tests/test_svar2_errors.py tests/test_py_ranges_readbound.py -v` then `pixi run test`
Expected: PASS.

- [ ] **Step 6: Update SKILL**

In `skills/genoray-api/SKILL.md`, remove `overlap_batch`/`find_ranges`/`gather_ranges` from the public SVAR2 surface. State the SVAR2 user-facing query API is `decode`/`region_counts`/`read_ranges`, and that the underscored `_find_ranges`/`_overlap_batch`/`_gather_ranges` are an internal, gvl-only numpy-dict wire contract not covered by semver.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_svar2_batch.py tests/ skills/genoray-api/SKILL.md
git commit -m "feat(svar2)!: privatize raw-dict FFI methods (overlap_batch/find_ranges/gather_ranges)"
```

### Task 9: Drop the `Reader` type alias

**Files:**
- Modify: `python/genoray/__init__.py` (`__all__` @21; `__getattr__` branch @44-49; `TYPE_CHECKING` alias @72)
- Modify: `skills/genoray-api/SKILL.md`

**Interfaces:**
- Produces: `genoray.Reader` no longer exists. `import genoray; genoray.Reader` raises `AttributeError`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_public_api.py` (create if absent):

```python
import pytest
import genoray


def test_reader_alias_removed():
    with pytest.raises(AttributeError):
        genoray.Reader
    assert "Reader" not in genoray.__all__
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run pytest tests/test_public_api.py::test_reader_alias_removed -v`
Expected: FAIL (`Reader` still resolvable).

- [ ] **Step 3: Remove `Reader`**

In `python/genoray/__init__.py`: delete `"Reader",` from `__all__`; delete the entire `if name == "Reader": ...` branch in `__getattr__` (lines 44-49); delete the `Reader = VCF | PGEN | SparseVar` line in the `TYPE_CHECKING` block (line 72).

- [ ] **Step 4: Run tests + import smoke**

Run: `pixi run pytest tests/test_public_api.py -v` then `pixi run python -c "import genoray; print(genoray.__all__)"`
Expected: PASS; `Reader` absent from the printed `__all__`.

- [ ] **Step 5: Update SKILL**

Remove every `Reader` mention from `skills/genoray-api/SKILL.md` (notably the reader-union line ~:18).

- [ ] **Step 6: Commit**

```bash
git add python/genoray/__init__.py tests/test_public_api.py skills/genoray-api/SKILL.md
git commit -m "feat(api)!: drop the Reader type alias (no shared protocol)"
```

### Task 10: Privatize `symbolic_ilen` and `IndexSchema` in `exprs`

**Files:**
- Modify: `python/genoray/exprs.py` (`symbolic_ilen`→`_symbolic_ilen` @148; `IndexSchema`→`_IndexSchema` @29; `:func:` xref @112)
- Modify: `python/genoray/_vcf.py:35`, `python/genoray/_pgen.py:31` (imports); their call sites (`_vcf.py:1086`, `_pgen.py:1077`)
- Modify: `docs/source/api.md:17` (remove the moot `:exclude-members: IndexSchema`)
- Modify: `skills/genoray-api/SKILL.md`

**Interfaces:**
- Produces: `genoray.exprs._symbolic_ilen(...)`, `genoray.exprs._IndexSchema` (underscored). Public exprs surface is the 7 documented predicates + `ILEN`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_public_api.py`:

```python
from genoray import exprs


def test_exprs_internals_are_private():
    assert not hasattr(exprs, "symbolic_ilen")
    assert not hasattr(exprs, "IndexSchema")
    assert hasattr(exprs, "_symbolic_ilen")
    assert hasattr(exprs, "_IndexSchema")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run pytest tests/test_public_api.py::test_exprs_internals_are_private -v`
Expected: FAIL.

- [ ] **Step 3: Rename in `exprs.py`**

Rename `def symbolic_ilen(` → `def _symbolic_ilen(` (@148) and `IndexSchema = {` → `_IndexSchema = {` (@29). Update the `:func:`symbolic_ilen`` cross-reference in the module docstring/comment (@112) to `:func:`_symbolic_ilen``.

- [ ] **Step 4: Repoint internal imports and calls**

`_vcf.py:35` `from .exprs import ILEN, symbolic_ilen` → `from .exprs import ILEN, _symbolic_ilen`; update the call `index.with_columns(ILEN=symbolic_ilen())` (@1086) → `_symbolic_ilen()`. Same in `_pgen.py:31` (`from .exprs import ILEN, is_biallelic, symbolic_ilen` → `... _symbolic_ilen`) and its call (@1077).

- [ ] **Step 5: Remove the moot autodoc exclusion**

In `docs/source/api.md:17`, delete the `:exclude-members: IndexSchema` line (the name is now underscore-private, so `automodule` won't render it).

- [ ] **Step 6: Run tests + full suite**

Run: `pixi run pytest tests/test_public_api.py -v` then `pixi run test`
Expected: PASS.

- [ ] **Step 7: Update SKILL**

In `skills/genoray-api/SKILL.md`, fix the "complete set (currently 7)" exprs claim so it no longer implies `symbolic_ilen`/`IndexSchema` are public; list exactly the public predicates + `ILEN`.

- [ ] **Step 8: Commit**

```bash
git add python/genoray/exprs.py python/genoray/_vcf.py python/genoray/_pgen.py docs/source/api.md tests/test_public_api.py skills/genoray-api/SKILL.md
git commit -m "refactor(exprs)!: privatize symbolic_ilen and IndexSchema"
```

### Task 11: Collapse the SVAR2 CLI flags to `--skip-symbolics-and-breakends`

**Files:**
- Modify: `python/genoray/_cli/__main__.py` (`write_svar2` signature @66-67; body @108; docstring @95-104)
- Modify: `skills/genoray-api/SKILL.md`
- Test: `tests/test_cli.py` (or the nearest CLI test module)

**Interfaces:**
- Produces: `genoray write` accepts a single `--skip-symbolics-and-breakends` (bool) mapping to `SparseVar2.from_vcf(skip_out_of_scope=...)`. The `--no-symbolic`/`--no-breakend` flags no longer exist on `write` (they remain on `write svar1`, unchanged).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
from genoray._cli.__main__ import app  # cyclopts App


def test_write_svar2_has_single_skip_flag(capsys):
    # --help lists the new flag and neither old flag
    with pytest.raises(SystemExit):
        app(["write", "--help"])
    out = capsys.readouterr().out
    assert "--skip-symbolics-and-breakends" in out
    assert "--no-symbolic" not in out
    assert "--no-breakend" not in out
```

(If the existing CLI tests drive commands differently — e.g. via `subprocess` on the `genoray` entrypoint — mirror that harness instead; read `tests/test_cli.py` first and match its invocation style.)

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run pytest tests/test_cli.py::test_write_svar2_has_single_skip_flag -v`
Expected: FAIL (old flags still present).

- [ ] **Step 3: Change the signature and body**

Replace the two params (@66-67):

```python
    skip_symbolics_and_breakends: Annotated[
        bool, Parameter(name="--skip-symbolics-and-breakends", negative="")
    ] = False,
```

and the body (@108):

```python
    skip_out_of_scope = skip_symbolics_and_breakends
```

- [ ] **Step 4: Rewrite the docstring**

Replace the `no_symbolic`/`no_breakend` parameter docs (@95-104) with a single entry:

```
    skip_symbolics_and_breakends
        Drop records whose ALT is symbolic (``<DEL>``, ``<INS>``, …) or a
        breakend, instead of erroring. The SVAR2 core cannot expand either
        class into nucleotides, so they are dropped together. (On
        ``genoray write svar1`` the two classes are filtered independently
        via ``--no-symbolic`` / ``--no-breakend``.)
```

- [ ] **Step 5: Run the CLI tests then full suite**

Run: `pixi run pytest tests/test_cli.py -v` then `pixi run test`
Expected: PASS.

- [ ] **Step 6: Update SKILL**

In `skills/genoray-api/SKILL.md`, update the CLI section: `genoray write` uses `--skip-symbolics-and-breakends`; keep the independent `--no-symbolic`/`--no-breakend` documented only under `write svar1`.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_cli/__main__.py tests/test_cli.py skills/genoray-api/SKILL.md
git commit -m "feat(cli)!: collapse svar2 write flags to --skip-symbolics-and-breakends"
```

---

## Phase 3 — SP-6 documentation & SKILL (no code change)

### Task 12: Fix the stale PGEN-filter example in README and docs

**Files:**
- Modify: `README.md:135-148`, `docs/source/index.md:144-157`

**Interfaces:** none (prose/example only).

- [ ] **Step 1: Confirm the real schema and a working predicate**

Run: `pixi run python -c "from genoray import exprs; print([n for n in dir(exprs) if not n.startswith('_')])"`
Expected: prints `ILEN`, `is_snp`, `is_indel`, `is_biallelic`, `is_symbolic`, `is_breakend`, `is_imprecise` (the public set). The `.gvi` columns are `CHROM`/`POS`/`REF`/`ALT`/`ILEN`.

- [ ] **Step 2: Rewrite both examples**

Replace the stale `Chromosome/Start/End/ilen/kind` + `pl.col("kind")...` example in `README.md:135-148` and `docs/source/index.md:144-157` with the current idiom, e.g.:

````markdown
```python
import genoray
from genoray import PGEN, exprs, Filter

# Keep only biallelic SNPs:
pgen = PGEN("file.pgen", filter=Filter(expr=exprs.is_snp & exprs.is_biallelic))

# Custom predicate over the .gvi index columns (CHROM, POS, REF, ALT, ILEN):
import polars as pl
pgen = PGEN("file.pgen", filter=Filter(expr=pl.col("ILEN").list.first().abs() <= 50))
```
````

Verify the `Filter` construction matches the current constructor (read `python/genoray/_vcf.py` `Filter` definition and the SKILL Filtering section; adjust the snippet to the real signature).

- [ ] **Step 3: Sanity-check the snippet runs**

Run the rewritten snippet against a test PGEN fixture in a scratch `python -c`/REPL to confirm it does not raise.
Expected: constructs without error.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/source/index.md
git commit -m "docs: fix stale PGEN filter example to current IndexSchema + exprs"
```

### Task 13: Broaden README summary + add missing api.md autodoc + fix exprs docstring

**Files:**
- Modify: `README.md:5-16` (summary), `docs/source/api.md` (autodoc), `python/genoray/exprs.py:10` (module docstring)

**Interfaces:** none.

- [ ] **Step 1: Broaden the README summary**

Under the `README.md:5-16` "two classes / five methods" framing, add a short "Also included" list: `SparseVar`/`SparseVar2` (sparse variant stores), `Reference`, mutation catalogues + signature refitting (`cosmic_signatures`/`fit_signatures`), and the `genoray index|write|view` CLI — pointing at the skill/docs for detail. Do not rewrite the whole README.

- [ ] **Step 2: Add the missing autodoc entries**

In `docs/source/api.md`, add:

```rst
.. autoclass:: SparseVar2
   :members:

.. autoclass:: Reference
   :members:

.. autofunction:: cosmic_signatures

.. autofunction:: fit_signatures
```

(match the file's existing autodoc directive style/indentation).

- [ ] **Step 3: Reword the exprs module docstring**

In `python/genoray/exprs.py:10`, change "Applicable for PGEN files and the experimental :meth:`VCF._load_index` method." to describe the capability without the private name, e.g. "Applicable to PGEN indexes, and to VCF indexes when one has been built."

- [ ] **Step 4: Build the docs (if a docs build task exists) or import-check**

Run: `pixi run python -c "import genoray; genoray.SparseVar2; genoray.Reference; genoray.cosmic_signatures; genoray.fit_signatures"`
Expected: all resolve (autodoc targets exist). If the repo has a docs-build task, run it and confirm no Sphinx warnings for the new entries.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/source/api.md python/genoray/exprs.py
git commit -m "docs: broaden README surface, add SVAR2/Reference/signatures autodoc, fix exprs docstring"
```

### Task 14: Close SKILL inventory + subset-idiom rationale

**Files:**
- Modify: `skills/genoray-api/SKILL.md`

**Interfaces:** none (doc), but this is the SP-6 SKILL reconciliation not already folded into Tasks 7-11.

- [ ] **Step 1: Add the missing method inventory**

Add to the relevant "where to look" lines: `VCF.get_record_info` (record-level INFO/annotation DataFrame), `SparseVar.annotate_with_gtf` (GTF annotation entry point with `level_filter`/`write_back`), `SparseVar.read_ranges_with_length` (length-guaranteed range read), `SparseVar.cache_afs` (allele-frequency cache). One line each, with return type.

- [ ] **Step 2: Document the canonical sample accessor + the subset-idiom rationale**

Add a short subsection stating:
- `available_samples` is the canonical "all samples in the file" accessor on all four readers.
- VCF/PGEN additionally expose `current_samples` (the selected subset) and a stateful `set_samples(...) -> Self`.
- **Why the two idioms differ (performance):** subsetting samples on VCF/PGEN is a costly operation (backend reader re-initialization), so it is a deliberate, stateful `set_samples()` call; on `SparseVar`/`SparseVar2` it is ~free, so subsetting is a per-call `samples=` kwarg on the read methods. This divergence is intentional, not an inconsistency.

- [ ] **Step 3: Verify SKILL has no remaining stale public names**

Run: `pixi run bash -lc 'grep -n "Reader\|symbolic_ilen\|IndexSchema\|overlap_batch\|find_ranges\|gather_ranges\|--no-symbolic\|\.samples\b" skills/genoray-api/SKILL.md'`
Expected: no hits except (a) `--no-symbolic`/`--no-breakend` under the `write svar1` section, and (b) `_find_ranges`/`_overlap_batch`/`_gather_ranges` if mentioned as private. Fix any stray public reference.

- [ ] **Step 4: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(skill): close inventory gaps, document sample-subset perf rationale"
```

---

## Phase 4 — genvarloader coordination + release

### Task 15: Update genvarloader to the new genoray surface

**Files (in the gvl repo/worktree `~/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel`):**
- Modify: `python/genvarloader/_dataset/_svar2_store_py.py`, `python/genvarloader/_dataset/_write.py`, `python/genvarloader/_dataset/_svar2_source.py`
- Modify: gvl test/generate scripts using `sv.samples`
- Modify: `pyproject.toml`/`pixi.toml` genoray pin

**Interfaces:**
- Consumes: genoray 3.0.0 — `SparseVar2._find_ranges`/`_overlap_batch`, `SparseVar2.available_samples`, no `genoray.Reader`.

- [ ] **Step 1: Build the genoray 3.0.0 wheel (or shadow) for gvl to test against**

genoray is a frozen wheel dep in gvl's Python env (per project memory), so an editable path-dep is not in play. For iteration, expose the genoray Python edits to gvl via a `PYTHONPATH` shadow:

Run (from the gvl worktree): `PYTHONPATH=/carter/users/dlaub/projects/genoray/python pixi run python -c "import genoray; print(genoray.__file__, genoray.__version__)"`
Expected: resolves to the genoray source tree. (Rust `_core` still comes from the installed wheel — fine, since no Rust changed.)

- [ ] **Step 2: Repoint the privatized method calls**

In `python/genvarloader/_dataset/_svar2_store_py.py` (the `svar2.find_ranges(...)` calls @60,141,210,306), `_write.py` (@1198), and `_svar2_source.py` (`svar2.overlap_batch(...)` @41): `svar2.find_ranges` → `svar2._find_ranges`, `svar2.overlap_batch` → `svar2._overlap_batch`. `gather_ranges` is unused in gvl — no change.

- [ ] **Step 3: Rename the sample accessor**

Run (gvl worktree): `pixi run bash -lc 'grep -rn "\.samples\b" python/ tests/ | grep -i svar2'` and change any `sv.samples`/`svar2.samples` → `.available_samples`.

- [ ] **Step 4: Drop the `Reader` import and fix `write`'s signature**

In `python/genvarloader/_dataset/_write.py`: change the import (@21) `from genoray import PGEN, VCF, Reader, SparseVar, SparseVar2` → `from genoray import PGEN, VCF, SparseVar, SparseVar2`. Change the `variants` annotation (@107) `str | Path | Reader | None` → `str | Path | VCF | PGEN | SparseVar | SparseVar2 | None`. If `Reader` is used elsewhere in gvl, define a gvl-local alias next to that import instead.

Run (gvl worktree): `pixi run bash -lc 'grep -rn "\bReader\b" python/ | grep genoray'`
Expected: no remaining genoray-`Reader` import after the fix.

- [ ] **Step 5: Run the gvl SVAR2 tests against the shadowed genoray**

Run (gvl worktree): `PYTHONPATH=/carter/users/dlaub/projects/genoray/python pixi run pytest -k "svar2" -q`
Expected: PASS.

Note (from project memory): a pyrefly pre-commit/pre-push hook can spuriously block commits from gvl `.claude/worktrees/*`. If it fires on unrelated files, that is the known false positive — do not chase it; commit the scoped gvl change.

- [ ] **Step 6: Bump the genoray pin (do this once genoray 3.0.0 is tagged — see Task 16)**

Update gvl's genoray dependency constraint to `>=3.0.0` in `pyproject.toml` (and `pixi.toml` if pinned there). Regenerate the lock if the workflow requires it.

- [ ] **Step 7: Commit (in the gvl repo)**

```bash
# in the gvl worktree
git add python/genvarloader/_dataset/ python/... pyproject.toml
git commit -m "feat!: adopt genoray 3.0.0 API (privatized SVAR2 FFI, available_samples, no Reader)"
```

### Task 16: Version bump genoray to 3.0.0 and final verification

**Files:**
- Modify: version managed by `commitizen` (`pyproject.toml` `version`, tag).

**Interfaces:** none.

- [ ] **Step 1: Full green gate on the genoray branch**

Run: `pixi run test`
Expected: PASS (entire suite).

- [ ] **Step 2: Lint/type gate**

Run: `pixi run ruff check genoray tests && pixi run ruff format --check genoray tests`
Expected: clean. (pyrefly runs via the pre-commit hook; ensure `pixi run prek-install` has been done.)

- [ ] **Step 3: Preview the bump**

Run: `pixi run bump-dry`
Expected: shows a **3.0.0** major bump (driven by the `feat!:`/`BREAKING CHANGE` commits). If it does not propose 3.0.0, add a `BREAKING CHANGE:` footer to one of the feat! commits or bump explicitly.

- [ ] **Step 4: Perform the bump**

Run: `pixi run bash -lc 'cz bump'` (or the repo's documented bump task)
Expected: `pyproject.toml` version → `3.0.0`, changelog updated, tag created.

- [ ] **Step 5: Confirm the public surface**

Run: `pixi run python -c "import genoray; print(genoray.__version__); print(sorted(genoray.__all__)); assert not hasattr(genoray, 'Reader'); from genoray import SparseVar2; import inspect; assert not hasattr(SparseVar2, 'find_ranges'); print('OK')"`
Expected: prints `3.0.0`, the `__all__` list without `Reader`, and `OK`.

- [ ] **Step 6: Final commit / push coordination**

The genoray 3.0.0 tag and the gvl adoption commit (Task 15) merge together. Push both branches / open both PRs referencing each other so CI builds the wheel and gvl validates against it.

---

## Self-Review

**Spec coverage** (spec §A–F → tasks):
- A1 sample rename → Task 7. A2 privatize FFI → Task 8. A3 drop Reader → Task 9. A4 privatize exprs internals → Task 10. A5 CLI flag → Task 11.
- B1 stale filter docs → Task 12. B2 README summary, B3 api.md autodoc, B4 exprs docstring → Task 13.
- C SKILL: sample rename/FFI/exprs/CLI folded into Tasks 7/8/10/11; inventory + subset rationale + canonical accessor → Task 14.
- D1 split `_mutcat` → Task 2. D2 relocate oracles → Task 1. D3 split `_utils` (+`hap_ilens`) → Task 3. D4 `_var_end_expr` → Task 4. D5 micro-refactors → Tasks 5 (np_to_pl_dtype, variant_file_type) + 6 (name_to_index).
- E gvl → Task 15. F testing/sequencing/version → gates in every task + Task 16.

All spec items map to a task. No gaps.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Code-motion tasks (1-3) intentionally give the function→file mapping + re-export contract + suite gate rather than fabricated line diffs, because the deliverable is verbatim motion validated by the existing oracle suite; the exact bodies already exist in the repo and must not be altered.

**Type/name consistency:** `available_samples` (Task 7) used consistently in Tasks 8/14/15. Underscored `_find_ranges`/`_overlap_batch`/`_gather_ranges` (Task 8) match Task 15's gvl repoint. `_symbolic_ilen`/`_IndexSchema` (Task 10) consistent. `_var_end_expr` (Task 4) single definition. `skip_symbolics_and_breakends` → `skip_out_of_scope` mapping (Task 11) matches `SparseVar2.from_vcf`'s existing kwarg.

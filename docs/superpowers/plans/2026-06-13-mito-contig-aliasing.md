# Mitochondrial Contig Aliasing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ContigNormalizer` treat the four mitochondrial contig spellings `{M, MT, chrM, chrMT}` as mutually equivalent, so an Ensembl-named `MT` variant resolves against a UCSC-named `chrM` reference instead of crashing.

**Architecture:** Add a module-level `_MITO_ALIASES` tuple and, in `ContigNormalizer.__init__`, detect the reference's mito contig and merge a `{alias: mito_contig}` map into the existing `contig_map`. All downstream structures (`remapper`, `_c2dup`, `dup2i`) derive from `contig_map`, so `norm()` and `c_idxs()` pick up the aliases automatically.

**Tech Stack:** Python, `pytest`, `pytest_cases`, run via `pixi run pytest`.

Reference spec: `docs/superpowers/specs/2026-06-13-mito-contig-aliasing-design.md`

---

### Task 1: Mito aliasing in `ContigNormalizer`

**Files:**
- Modify: `genoray/_utils.py:16-33` (add `_MITO_ALIASES`, update `contig_map`, fix docstring at line 21)
- Test: `tests/test_utils.py:38-49` (add `contig_*` cases consumed by existing `test_normalize_contig_name`)

- [ ] **Step 1: Write the failing tests**

In `tests/test_utils.py`, add these three case functions after `contig_no_match` (around line 42). They follow the existing `contig_*` / `parametrize_with_cases` pattern and need no new test function:

```python
def contig_mito_mt_to_chrm():
    unnormed = "MT"
    source = ContigNormalizer(["chr1", "chrM"])
    desired = "chrM"
    return unnormed, source, desired


def contig_mito_chrmt_to_chrm():
    unnormed = "chrMT"
    source = ContigNormalizer(["chr1", "chrM"])
    desired = "chrM"
    return unnormed, source, desired


def contig_mito_chrm_to_mt():
    unnormed = "chrM"
    source = ContigNormalizer(["1", "MT"])
    desired = "MT"
    return unnormed, source, desired


def contig_mito_absent():
    unnormed = "MT"
    source = ContigNormalizer(["chr1", "chr2"])
    desired = None
    return unnormed, source, desired
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_utils.py::test_normalize_contig_name -v`
Expected: the three mito-resolving cases FAIL (`assert None == 'chrM'` / `'MT'`); `contig_mito_absent` PASSES (already `None`).

- [ ] **Step 3: Add `_MITO_ALIASES` constant**

In `genoray/_utils.py`, add after line 16 (`DTYPE = TypeVar(...)`):

```python
_MITO_ALIASES = ("M", "MT", "chrM", "chrMT")
```

- [ ] **Step 4: Update `contig_map` and the docstring**

In `genoray/_utils.py`, replace the docstring note (line 21) and the `contig_map` construction (lines 29-33).

Change the docstring line 21 from:

```python
    Note: this does not handle the special case of equivalence between M and MT.
```

to:

```python
    Mitochondrial aliases {M, MT, chrM, chrMT} are treated as mutually equivalent and
    resolve to whichever spelling the reference actually contains.
```

Replace the `contig_map` assignment:

```python
        self.contig_map = (
            {f"{c[3:]}": c for c in contigs if c.startswith("chr")}
            | {f"chr{c}": c for c in contigs if not c.startswith("chr")}
            | {c: c for c in contigs}
        )
```

with:

```python
        mito = next((c for c in self.contigs if c in _MITO_ALIASES), None)
        mito_map = {a: mito for a in _MITO_ALIASES} if mito is not None else {}
        self.contig_map = (
            {f"{c[3:]}": c for c in contigs if c.startswith("chr")}
            | {f"chr{c}": c for c in contigs if not c.startswith("chr")}
            | {c: c for c in contigs}
            | mito_map
        )
```

Note: `self.contigs` is already assigned on line 28 before this block, so `next(...)` over it is safe.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_utils.py::test_normalize_contig_name -v`
Expected: all cases PASS, including the three mito-resolving cases and `contig_mito_absent`.

- [ ] **Step 6: Run the full utils test file to check for regressions**

Run: `pixi run pytest tests/test_utils.py -v`
Expected: all PASS (existing `chr`-prefix cases unaffected).

- [ ] **Step 7: Commit**

```bash
git add genoray/_utils.py tests/test_utils.py
git commit -m "fix(utils): treat mitochondrial aliases M/MT/chrM/chrMT as equivalent (#61)"
```

---

## Self-Review Notes

- **Spec coverage:** Mito aliasing (Task 1, Steps 3-4), behavior table incl. symmetric MT case + no-mito fallback (Step 1 cases), degenerate multi-mito → `next()` first-in-order (Step 4 code), docstring fix (Step 4). No graceful-skip work — out of scope per spec.
- **Public API:** None changed (`_utils.py` is internal); no `SKILL.md` update needed, per spec.
- **Placeholders:** None — all code and commands are concrete.

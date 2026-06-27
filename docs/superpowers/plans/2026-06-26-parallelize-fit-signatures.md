# Parallelize fit_signatures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize `fit_signatures` over samples with joblib, defaulting to all cores, with identical results.

**Architecture:** The per-sample refit is already a pure function `_fit_one(W, m, ...)`. Replace the serial sample loop in `fit_signatures` with a `joblib.Parallel` dispatch over `delayed(_fit_one)` calls; results return in input order so determinism and `activities[j]` alignment are preserved. Thread the new `n_jobs`/`backend` knobs through the public `SparseVar.assign_signatures` wrapper.

**Tech Stack:** Python, joblib (already a dependency: `joblib>=1.4.2,<2`), scipy NNLS, polars, numpy.

## Global Constraints

- `joblib` is already declared in `pyproject.toml` (`joblib>=1.4.2,<2`) — no new dependency.
- Conventional Commits for every commit (`feat:`, `docs:`, `test:`, etc.).
- Public API change: `fit_signatures` and `assign_signatures` gain kwargs → `skills/genoray-api/SKILL.md` MUST be updated in this same work (per `CLAUDE.md`).
- New parameters, identical on both functions: `n_jobs: int = -1`, `backend: str = "loky"`.
- Results MUST be bit-identical between `n_jobs=1` and `n_jobs=-1` (the fit is deterministic).
- Run tests with `pixi run pytest`.

---

### Task 1: Parallelize `fit_signatures`

**Files:**
- Modify: `genoray/_signatures.py` (imports near line 17; `fit_signatures` signature at lines 108-114; per-sample loop at lines 175-186)
- Test: `tests/test_signatures.py`

**Interfaces:**
- Consumes: existing `_fit_one(W, m, *, max_delta, min_activity) -> tuple[NDArray, float]` (unchanged).
- Produces: `fit_signatures(catalogue, reference, *, max_delta=0.01, min_activity=0.005, n_jobs=-1, backend="loky") -> pl.DataFrame` — same return shape/columns as before (`Sample`, one column per signature, `cosine_similarity`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_signatures.py` (the `_catalogue_and_reference()` helper already exists at lines 88-109):

```python
def test_fit_signatures_parallel_matches_serial():
    cat, ref = _catalogue_and_reference()
    serial = fit_signatures(cat, ref, n_jobs=1)
    parallel = fit_signatures(cat, ref, n_jobs=-1)
    assert serial.columns == parallel.columns
    for col in serial.columns:
        if col == "Sample":
            assert serial[col].to_list() == parallel[col].to_list()
        else:
            assert serial[col].to_numpy() == pytest.approx(
                parallel[col].to_numpy(), abs=0.0, rel=0.0
            )


def test_fit_signatures_accepts_backend_kwarg():
    cat, ref = _catalogue_and_reference()
    act = fit_signatures(cat, ref, n_jobs=2, backend="loky")
    assert act["Sample"].to_list() == ["s0", "s1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_signatures.py::test_fit_signatures_parallel_matches_serial tests/test_signatures.py::test_fit_signatures_accepts_backend_kwarg -v`
Expected: FAIL with `TypeError: fit_signatures() got an unexpected keyword argument 'n_jobs'`

- [ ] **Step 3: Add the joblib import**

In `genoray/_signatures.py`, with the other imports (after the scipy import at line 17), add:

```python
from joblib import Parallel, delayed
```

- [ ] **Step 4: Add the new parameters to the signature**

Change `fit_signatures`' signature (lines 108-114) from:

```python
def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    max_delta: float = 0.01,
    min_activity: float = 0.005,
) -> pl.DataFrame:
```

to:

```python
def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    max_delta: float = 0.01,
    min_activity: float = 0.005,
    n_jobs: int = -1,
    backend: str = "loky",
) -> pl.DataFrame:
```

- [ ] **Step 5: Replace the serial loop with a joblib dispatch**

Replace the loop (current lines 175-180):

```python
    activities = np.zeros((len(sample_cols), len(sig_cols)), dtype=np.float64)
    cosines = np.zeros(len(sample_cols), dtype=np.float64)
    for j in range(len(sample_cols)):
        h, cos = _fit_one(W, M[:, j], max_delta=max_delta, min_activity=min_activity)
        activities[j] = h
        cosines[j] = cos
```

with:

```python
    activities = np.zeros((len(sample_cols), len(sig_cols)), dtype=np.float64)
    cosines = np.zeros(len(sample_cols), dtype=np.float64)
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_fit_one)(W, M[:, j], max_delta=max_delta, min_activity=min_activity)
        for j in range(len(sample_cols))
    )
    for j, (h, cos) in enumerate(results):
        activities[j] = h
        cosines[j] = cos
```

- [ ] **Step 6: Update the docstring**

In the `fit_signatures` docstring (Parameters section, after the `min_activity` entry around line 130), add:

```
    n_jobs
        Number of parallel workers for the per-sample refit (passed to
        ``joblib.Parallel``). ``-1`` (default) uses all cores; ``1`` runs
        serially. Results are identical regardless of ``n_jobs``.
    backend
        ``joblib`` backend (default ``"loky"``, process-based). Samples are
        refit independently, so a process backend avoids GIL contention from
        the forward-selection orchestration.
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `pixi run pytest tests/test_signatures.py::test_fit_signatures_parallel_matches_serial tests/test_signatures.py::test_fit_signatures_accepts_backend_kwarg -v`
Expected: PASS

- [ ] **Step 8: Run the full signatures test file to verify no regression**

Run: `pixi run pytest tests/test_signatures.py -v`
Expected: PASS (all pre-existing tests still green; network-marked tests may be skipped/deselected)

- [ ] **Step 9: Commit**

```bash
git add genoray/_signatures.py tests/test_signatures.py
git commit -m "feat: parallelize fit_signatures over samples with joblib"
```

---

### Task 2: Thread `n_jobs`/`backend` through `SparseVar.assign_signatures`

**Files:**
- Modify: `genoray/_svar.py` (`assign_signatures` signature at lines 1674-1682; `fit_signatures` call at lines 1714-1716)
- Test: `tests/test_svar_mutations.py`

**Interfaces:**
- Consumes: `fit_signatures(..., n_jobs, backend)` from Task 1.
- Produces: `SparseVar.assign_signatures(kind, *, reference=None, count="allele", max_delta=0.01, min_activity=0.005, n_jobs=-1, backend="loky") -> pl.DataFrame` — same return shape as before.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_svar_mutations.py` (the `_toy_sbs_reference()` helper and `annotated_svar` fixture already exist; see `test_assign_signatures_with_explicit_reference` at line 383):

```python
def test_assign_signatures_forwards_n_jobs(annotated_svar):
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    ref = _toy_sbs_reference()
    serial = svar.assign_signatures("SBS96", reference=ref, n_jobs=1)
    parallel = svar.assign_signatures("SBS96", reference=ref, n_jobs=2, backend="loky")
    assert serial.columns == parallel.columns
    for col in ("SBS_A", "SBS_B", "cosine_similarity"):
        assert serial[col].to_numpy() == pytest.approx(
            parallel[col].to_numpy(), abs=0.0, rel=0.0
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/test_svar_mutations.py::test_assign_signatures_forwards_n_jobs -v`
Expected: FAIL with `TypeError: assign_signatures() got an unexpected keyword argument 'n_jobs'`

- [ ] **Step 3: Add the parameters to the signature**

Change `assign_signatures`' signature (lines 1674-1682) from:

```python
    def assign_signatures(
        self,
        kind: Literal["SBS96", "DBS78", "ID83"],
        *,
        reference: "pl.DataFrame | str | Path | None" = None,
        count: Literal["allele", "sample"] = "allele",
        max_delta: float = 0.01,
        min_activity: float = 0.005,
    ) -> "pl.DataFrame":
```

to:

```python
    def assign_signatures(
        self,
        kind: Literal["SBS96", "DBS78", "ID83"],
        *,
        reference: "pl.DataFrame | str | Path | None" = None,
        count: Literal["allele", "sample"] = "allele",
        max_delta: float = 0.01,
        min_activity: float = 0.005,
        n_jobs: int = -1,
        backend: str = "loky",
    ) -> "pl.DataFrame":
```

- [ ] **Step 4: Forward the parameters to `fit_signatures`**

Change the return call (lines 1714-1716) from:

```python
        return fit_signatures(
            catalogue, ref, max_delta=max_delta, min_activity=min_activity
        )
```

to:

```python
        return fit_signatures(
            catalogue,
            ref,
            max_delta=max_delta,
            min_activity=min_activity,
            n_jobs=n_jobs,
            backend=backend,
        )
```

- [ ] **Step 5: Update the docstring**

In the `assign_signatures` docstring, after the `max_delta, min_activity` Parameters entry (around line 1699), add:

```
        n_jobs, backend
            Forwarded to :func:`genoray.fit_signatures` to control per-sample
            parallelism (default ``-1`` = all cores, process-based ``"loky"``
            backend).
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pixi run pytest tests/test_svar_mutations.py::test_assign_signatures_forwards_n_jobs -v`
Expected: PASS

- [ ] **Step 7: Run the related test file to verify no regression**

Run: `pixi run pytest tests/test_svar_mutations.py -v`
Expected: PASS (including the pre-existing `test_assign_signatures_with_explicit_reference`)

- [ ] **Step 8: Commit**

```bash
git add genoray/_svar.py tests/test_svar_mutations.py
git commit -m "feat: forward n_jobs/backend through SparseVar.assign_signatures"
```

---

### Task 3: Document the new kwargs in SKILL.md

**Files:**
- Modify: `skills/genoray-api/SKILL.md` (signature-refit section around lines 415-429, where `fit_signatures`, `assign_signatures`, and `cosmic_signatures` are documented)

**Interfaces:**
- Consumes: the final public signatures from Tasks 1 and 2.
- Produces: documentation only (no code).

- [ ] **Step 1: Locate the documented signatures**

Run: `grep -n "fit_signatures\|assign_signatures" skills/genoray-api/SKILL.md`
Expected: lines listing the `assign_signatures(...)` examples (around 420-422) and the `fit_signatures` / `cosmic_signatures` reference block (around 426-429).

- [ ] **Step 2: Update the `fit_signatures` reference line**

Find the `fit_signatures` documentation line and update its signature to include the new kwargs, e.g. change a line like:

```
- `fit_signatures(catalogue, reference, *, max_delta=0.01, min_activity=0.005) -> pl.DataFrame`
```

to:

```
- `fit_signatures(catalogue, reference, *, max_delta=0.01, min_activity=0.005, n_jobs=-1, backend="loky") -> pl.DataFrame`
  — refits per sample in parallel (joblib). `n_jobs=-1` uses all cores; `n_jobs=1`
  is serial. Results are identical regardless of `n_jobs`/`backend`.
```

(If the documented signature text differs slightly, preserve the existing wording and append the two new kwargs plus the explanatory sentence.)

- [ ] **Step 3: Update the `assign_signatures` documentation**

Find the `assign_signatures` reference/signature in the same section and append `, n_jobs=-1, backend="loky"` to its keyword list, with a one-line note: "forwards `n_jobs`/`backend` to `fit_signatures` for per-sample parallelism."

- [ ] **Step 4: Verify both functions now mention the kwargs**

Run: `grep -n "n_jobs" skills/genoray-api/SKILL.md`
Expected: at least two matches — one in the `fit_signatures` line, one in the `assign_signatures` line.

- [ ] **Step 5: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs: document n_jobs/backend kwargs on signature refit API"
```

---

### Task 4: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full signature + mutation test surface**

Run: `pixi run pytest tests/test_signatures.py tests/test_svar_mutations.py -v -m "not network"`
Expected: PASS

- [ ] **Step 2: Lint/format check**

Run: `ruff check genoray tests && ruff format --check genoray tests`
Expected: no errors (run `ruff format genoray tests` to fix formatting if needed, then re-stage/commit)

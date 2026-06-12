# COSMIC Signature Refitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose a `SparseVar`'s SBS-96 / DBS-78 / ID-83 mutation catalogue into per-sample COSMIC signature activities, via a lean numpy/scipy port of SigProfilerAssignment's sparse forward-selection refit.

**Architecture:** A new pure module `genoray/_signatures.py` holds the refit core (`fit_signatures`) and a `pooch`-backed reference loader (`cosmic_signatures`). `SparseVar.assign_signatures` is a thin convenience wiring `mutation_matrix` → `fit_signatures`. No heavyweight SigProfiler deps; reference signatures are fetched/cached on demand.

**Tech Stack:** Python, numpy, scipy (`scipy.optimize.nnls`), polars, pooch, pytest + pytest-cases. All dev commands run under Pixi (`pixi run pytest ...`). Cross-check runs in the `sigprofiler` pixi env.

**Spec:** `docs/superpowers/specs/2026-06-11-cosmic-signature-refitting-design.md`

---

## Key codebase facts (read before starting)

- `SparseVar.mutation_matrix(kind, *, count="allele"|"sample")` (`genoray/_svar.py:1550`) returns a `pl.DataFrame`: a `MutationType` string column + one `Int64` column per sample, rows in fixed codebook order. This is the catalogue input to refitting.
- `genoray/_mutcat.py` defines the canonical row labels and order:
  - `labels(kind)` → ordered `list[str]` for `kind ∈ {"SBS96","DBS78","ID83"}`.
  - SBS96 labels look like `A[C>A]A`; DBS78 like `AC>CA`; ID83 like `1:Del:C:0`. **These match the `Type` column in COSMIC reference signature files exactly**, so reference rows align by string join.
- `genoray/__init__.py` uses a lazy PEP-562 `__getattr__` with a `_LAZY` dict and a `TYPE_CHECKING` block. New public names must be added in **three** places: `__all__`, `_LAZY`, and the `TYPE_CHECKING` block.
- `pyproject.toml` `[project].dependencies` is the runtime dep list. `pixi.toml` has `pooch = "*"` already in `[pypi-dependencies]` (dev), and a `sigprofiler` environment (`[feature.sigprofiler.*]`) used for SigProfiler cross-checks.
- Tests live in `tests/`, use `pytest`/`pytest_cases`, import private helpers directly. Markers `network` and `sigprofiler` already exist in `pyproject.toml [tool.pytest.ini_options].markers`.
- Run tests: `pixi run pytest tests/<file> -v`. Cross-check: `pixi run -e sigprofiler pytest tests/<file> -v`.
- Commit convention: Conventional Commits (`feat:`, `test:`, `docs:`, `chore:`).

---

## File structure

- **Create** `genoray/_signatures.py` — `fit_signatures`, `cosmic_signatures`, and private refit helpers (`_cosine`, `_nnls`, `_fit_one`).
- **Modify** `genoray/_svar.py` — add `SparseVar.assign_signatures`.
- **Modify** `genoray/__init__.py` — export `fit_signatures`, `cosmic_signatures`.
- **Modify** `pyproject.toml` — add `scipy` and `pooch` to runtime `dependencies`.
- **Modify** `pixi.toml` — add `sigprofilerassignment` to the `sigprofiler` feature for the cross-check test.
- **Modify** `skills/genoray-api/SKILL.md` — document the new public surface.
- **Create** `tests/test_signatures.py` — unit tests for the refit core and loader.
- **Create** `tests/test_signatures_calibration.py` — SigProfilerAssignment cross-check (`sigprofiler` env).
- **Create** `tests/data/cosmic_mini.txt` — tiny local reference-signature fixture for loader/parse tests.

---

## Task 1: Dependencies

**Files:**
- Modify: `pyproject.toml:9-36` (`[project].dependencies`)
- Modify: `pixi.toml:106-107` (`[feature.sigprofiler.pypi-dependencies]`)

- [ ] **Step 1: Add runtime deps to `pyproject.toml`**

In `pyproject.toml`, inside the `dependencies = [ ... ]` list (ends at line 36), add two entries just before the closing `]`:

```toml
    "filelock>3,<4",
    "scipy>=1.10",
    "pooch>=1.7",
]
```

(The `"filelock>3,<4",` line already exists — add the two new lines after it.)

- [ ] **Step 2: Add the cross-check dep to the sigprofiler pixi env**

In `pixi.toml`, under `[feature.sigprofiler.pypi-dependencies]` (currently only `sigprofilermatrixgenerator`):

```toml
[feature.sigprofiler.pypi-dependencies]
sigprofilermatrixgenerator = ">=1.3, <2"
sigprofilerassignment = "*"
```

- [ ] **Step 3: Install and verify imports**

Run:
```bash
pixi install
pixi run python -c "import scipy.optimize, pooch; print('ok')"
```
Expected: prints `ok` with no ImportError.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.toml
git commit -m "chore(signatures): add scipy + pooch runtime deps and SPA cross-check env"
```

---

## Task 2: Cosine + NNLS primitives

Create the module with the two numeric primitives the refit is built on.

**Files:**
- Create: `genoray/_signatures.py`
- Test: `tests/test_signatures.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_signatures.py`:

```python
import numpy as np
import pytest

from genoray._signatures import _cosine, _nnls


def test_cosine_identical_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert _cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert _cosine(a, b) == 0.0


def test_nnls_recovers_nonnegative_solution():
    # W h = m with a known nonnegative h
    W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    h_true = np.array([3.0, 5.0])
    m = W @ h_true
    h = _nnls(W, m)
    assert np.allclose(h, h_true, atol=1e-6)
    assert (h >= 0).all()
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_signatures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'genoray._signatures'`.

- [ ] **Step 3: Create the module with the primitives**

Create `genoray/_signatures.py`:

```python
"""COSMIC mutational-signature refitting.

Ports the core of SigProfilerAssignment: a sparse forward-selection refit that
decomposes a mutation catalogue into per-sample activities against a set of
reference signatures. Pure numpy/scipy/polars; no SigProfiler dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import nnls

from ._mutcat import labels

Kind = Literal["SBS96", "DBS78", "ID83"]


def _cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Cosine similarity of two vectors; 0.0 if either has zero norm."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _nnls(W: NDArray[np.floating], m: NDArray[np.floating]) -> NDArray[np.float64]:
    """Non-negative least squares: argmin_{h>=0} ||W h - m||."""
    h, _ = nnls(W.astype(np.float64), m.astype(np.float64))
    return h
```

- [ ] **Step 4: Run it to verify pass**

Run: `pixi run pytest tests/test_signatures.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add genoray/_signatures.py tests/test_signatures.py
git commit -m "feat(signatures): add cosine + NNLS primitives"
```

---

## Task 3: Single-sample forward-selection refit (`_fit_one`)

Greedy forward selection by cosine improvement, then prune low-activity signatures.

**Files:**
- Modify: `genoray/_signatures.py`
- Test: `tests/test_signatures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signatures.py`:

```python
from genoray._signatures import _fit_one


def _toy_reference():
    # 4 mutation types, 3 signatures (columns sum to 1).
    W = np.array(
        [
            [0.7, 0.1, 0.25],
            [0.1, 0.7, 0.25],
            [0.1, 0.1, 0.25],
            [0.1, 0.1, 0.25],
        ]
    )
    return W


def test_fit_one_recovers_sparse_truth():
    W = _toy_reference()
    # Only signatures 0 and 1 are active; signature 2 absent.
    h_true = np.array([30.0, 70.0, 0.0])
    m = W @ h_true
    h, cos = _fit_one(W, m, max_delta=0.01, min_activity=0.005)
    assert cos == pytest.approx(1.0, abs=1e-6)
    assert h[2] == 0.0                      # unused signature stays out
    assert h[0] == pytest.approx(30.0, rel=1e-3)
    assert h[1] == pytest.approx(70.0, rel=1e-3)


def test_fit_one_zero_sample():
    W = _toy_reference()
    h, cos = _fit_one(W, np.zeros(4), max_delta=0.01, min_activity=0.005)
    assert (h == 0).all()
    assert cos == 0.0


def test_fit_one_prunes_below_min_activity():
    W = _toy_reference()
    # Mostly signature 0, a tiny sliver of signature 1 (< 0.5% of total).
    h_true = np.array([100.0, 0.2, 0.0])
    m = W @ h_true
    h, cos = _fit_one(W, m, max_delta=0.001, min_activity=0.005)
    # signature 1 sliver is below min_activity -> pruned to 0
    assert h[1] == 0.0
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_signatures.py -k fit_one -v`
Expected: FAIL — `ImportError: cannot import name '_fit_one'`.

- [ ] **Step 3: Implement `_fit_one`**

Append to `genoray/_signatures.py`:

```python
def _fit_one(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    *,
    max_delta: float,
    min_activity: float,
) -> tuple[NDArray[np.float64], float]:
    """Refit one sample by sparse forward selection.

    Returns ``(activities, cosine)`` where ``activities`` has one entry per
    reference signature (column of ``W``), zero for unselected signatures, and
    ``cosine`` is the reconstruction cosine similarity of the final fit.
    """
    n_sigs = W.shape[1]
    full = np.zeros(n_sigs, dtype=np.float64)
    if float(np.sum(m)) == 0.0:
        return full, 0.0

    active: list[int] = []
    remaining = list(range(n_sigs))
    best_cos = 0.0

    # Forward selection: add the signature that most improves cosine, until the
    # improvement falls below max_delta.
    while remaining:
        best = None  # (cos, sig_index, h_subvector)
        for c in remaining:
            cand = active + [c]
            h_sub = _nnls(W[:, cand], m)
            recon = W[:, cand] @ h_sub
            cos = _cosine(m, recon)
            if best is None or cos > best[0]:
                best = (cos, c, h_sub)
        assert best is not None
        cos, c, _ = best
        if cos - best_cos < max_delta:
            break
        best_cos = cos
        active.append(c)
        remaining.remove(c)

    if not active:
        return full, 0.0

    # Prune signatures below min_activity (as a fraction of total), re-fitting
    # survivors until the active set is stable.
    while True:
        h_sub = _nnls(W[:, active], m)
        total = float(h_sub.sum())
        if total == 0.0:
            return full, 0.0
        keep = [active[i] for i in range(len(active)) if h_sub[i] / total >= min_activity]
        if len(keep) == len(active):
            break
        if not keep:
            # everything pruned: keep the single largest contributor
            keep = [active[int(np.argmax(h_sub))]]
        active = keep

    h_sub = _nnls(W[:, active], m)
    recon = W[:, active] @ h_sub
    cos = _cosine(m, recon)
    for i, sig in enumerate(active):
        full[sig] = h_sub[i]
    return full, cos
```

- [ ] **Step 4: Run it to verify pass**

Run: `pixi run pytest tests/test_signatures.py -k fit_one -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add genoray/_signatures.py tests/test_signatures.py
git commit -m "feat(signatures): add single-sample forward-selection refit"
```

---

## Task 4: `fit_signatures` — DataFrame orchestration, row alignment, errors

Aligns catalogue rows to reference rows by `MutationType`, normalizes reference columns to sum 1 (so activities are in count units), fits every sample, and returns the activities DataFrame.

**Files:**
- Modify: `genoray/_signatures.py`
- Test: `tests/test_signatures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signatures.py`:

```python
import polars as pl

from genoray._signatures import fit_signatures


def _catalogue_and_reference():
    types = ["A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T"]
    ref = pl.DataFrame(
        {
            "MutationType": types,
            "SBS_X": [0.7, 0.1, 0.1, 0.1],
            "SBS_Y": [0.1, 0.7, 0.1, 0.1],
            "SBS_Z": [0.25, 0.25, 0.25, 0.25],
        }
    )
    W = ref.select(["SBS_X", "SBS_Y", "SBS_Z"]).to_numpy()
    # sample s0 = 30*X + 70*Y ; sample s1 = 100*Z
    m0 = W @ np.array([30.0, 70.0, 0.0])
    m1 = W @ np.array([0.0, 0.0, 50.0])
    cat = pl.DataFrame(
        {
            "MutationType": types,
            "s0": np.rint(m0).astype(np.int64),
            "s1": np.rint(m1).astype(np.int64),
        }
    )
    return cat, ref


def test_fit_signatures_shape_and_columns():
    cat, ref = _catalogue_and_reference()
    act = fit_signatures(cat, ref)
    assert act.columns == ["Sample", "SBS_X", "SBS_Y", "SBS_Z", "cosine_similarity"]
    assert act["Sample"].to_list() == ["s0", "s1"]
    assert act.height == 2


def test_fit_signatures_recovers_activities():
    cat, ref = _catalogue_and_reference()
    act = fit_signatures(cat, ref)
    row0 = act.filter(pl.col("Sample") == "s0")
    assert row0["SBS_X"].item() == pytest.approx(30.0, rel=0.02)
    assert row0["SBS_Y"].item() == pytest.approx(70.0, rel=0.02)
    assert row0["SBS_Z"].item() == 0.0
    assert row0["cosine_similarity"].item() == pytest.approx(1.0, abs=1e-3)


def test_fit_signatures_aligns_rows_by_join_not_position():
    cat, ref = _catalogue_and_reference()
    ref_shuffled = ref.sort("MutationType", descending=True)  # reorder rows
    act = fit_signatures(cat, ref_shuffled)
    row0 = act.filter(pl.col("Sample") == "s0")
    assert row0["SBS_X"].item() == pytest.approx(30.0, rel=0.02)


def test_fit_signatures_missing_type_raises():
    cat, ref = _catalogue_and_reference()
    ref_missing = ref.head(3)  # drop a row present in the catalogue
    with pytest.raises(ValueError, match="MutationType"):
        fit_signatures(cat, ref_missing)
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_signatures.py -k fit_signatures -v`
Expected: FAIL — `ImportError: cannot import name 'fit_signatures'`.

- [ ] **Step 3: Implement `fit_signatures`**

Append to `genoray/_signatures.py`:

```python
def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    max_delta: float = 0.01,
    min_activity: float = 0.005,
) -> pl.DataFrame:
    """Refit a mutation catalogue against reference signatures.

    Parameters
    ----------
    catalogue
        A ``mutation_matrix``-shaped frame: a ``MutationType`` column followed by
        one numeric column per sample.
    reference
        A ``MutationType`` column followed by one column per reference signature.
        Columns need not be pre-normalized; each is scaled to sum 1 so reported
        activities are in mutation-count units.
    max_delta
        Minimum cosine-similarity improvement to keep adding a signature
        (forward-selection stop criterion).
    min_activity
        Minimum fractional contribution; signatures below this are pruned.

    Returns
    -------
    pl.DataFrame
        One row per sample: a ``Sample`` column, one Float column per reference
        signature (activities, 0.0 if unselected), and a ``cosine_similarity``
        column for the final reconstruction.

    Raises
    ------
    ValueError
        If a ``MutationType`` present in the catalogue is missing from the
        reference (rows cannot be aligned).
    """
    if "MutationType" not in catalogue.columns:
        raise ValueError("catalogue must have a 'MutationType' column.")
    if "MutationType" not in reference.columns:
        raise ValueError("reference must have a 'MutationType' column.")

    sample_cols = [c for c in catalogue.columns if c != "MutationType"]
    sig_cols = [c for c in reference.columns if c != "MutationType"]

    # Align reference rows to the catalogue's row order by joining on MutationType.
    aligned = catalogue.select("MutationType").join(
        reference, on="MutationType", how="left"
    )
    missing = aligned.filter(pl.col(sig_cols[0]).is_null())
    if missing.height > 0:
        bad = missing["MutationType"].to_list()
        raise ValueError(
            f"reference is missing MutationType rows present in the catalogue: {bad}"
        )

    W = aligned.select(sig_cols).to_numpy().astype(np.float64)  # (n_types, n_sigs)
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0.0] = 1.0  # avoid div-by-zero for empty signatures
    W = W / col_sums  # normalize each signature column to sum 1

    M = catalogue.select(sample_cols).to_numpy().astype(np.float64)  # (n_types, n_samples)

    activities = np.zeros((len(sample_cols), len(sig_cols)), dtype=np.float64)
    cosines = np.zeros(len(sample_cols), dtype=np.float64)
    for j in range(len(sample_cols)):
        h, cos = _fit_one(W, M[:, j], max_delta=max_delta, min_activity=min_activity)
        activities[j] = h
        cosines[j] = cos

    out: dict[str, object] = {"Sample": sample_cols}
    for i, sig in enumerate(sig_cols):
        out[sig] = activities[:, i]
    out["cosine_similarity"] = cosines
    return pl.DataFrame(out)
```

- [ ] **Step 4: Run it to verify pass**

Run: `pixi run pytest tests/test_signatures.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add genoray/_signatures.py tests/test_signatures.py
git commit -m "feat(signatures): add fit_signatures DataFrame orchestration + row alignment"
```

---

## Task 5: `cosmic_signatures` — pooch-backed reference loader

Fetches/caches the official COSMIC reference signature TSV and returns it as a
validated DataFrame. Parsing is tested against a tiny local fixture; the real
network fetch is a `network`-marked smoke test.

**Files:**
- Modify: `genoray/_signatures.py`
- Create: `tests/data/cosmic_mini.txt`
- Test: `tests/test_signatures.py`

- [ ] **Step 1: Create the local fixture**

Create `tests/data/cosmic_mini.txt` (tab-separated; COSMIC files use a `Type`
header and signature columns). Include exactly the first four SBS96 labels so it
parses without needing all 96 rows:

```
Type	SBS1	SBS5
A[C>A]A	0.0001	0.0002
A[C>A]C	0.0003	0.0004
A[C>A]G	0.0005	0.0006
A[C>A]T	0.0007	0.0008
```

(Use literal tab characters between fields.)

- [ ] **Step 2: Write the failing test**

Append to `tests/test_signatures.py`:

```python
from pathlib import Path

from genoray._signatures import _load_signature_file, cosmic_signatures

DATA = Path(__file__).parent / "data"


def test_load_signature_file_renames_type_column():
    df = _load_signature_file(DATA / "cosmic_mini.txt")
    assert df.columns[0] == "MutationType"
    assert "SBS1" in df.columns and "SBS5" in df.columns
    assert df["MutationType"].to_list()[0] == "A[C>A]A"


def test_load_signature_file_path_as_str():
    df = _load_signature_file(str(DATA / "cosmic_mini.txt"))
    assert df.height == 4


@pytest.mark.network
def test_cosmic_signatures_sbs96_row_order():
    from genoray._mutcat import labels

    df = cosmic_signatures("SBS96")
    assert df.columns[0] == "MutationType"
    assert df["MutationType"].to_list() == labels("SBS96")  # canonical 96 rows
    # signature columns are the COSMIC SBS set
    assert any(c.startswith("SBS") for c in df.columns[1:])
```

- [ ] **Step 3: Run it to verify failure**

Run: `pixi run pytest tests/test_signatures.py -k "load_signature or cosmic" -v`
Expected: FAIL — `ImportError: cannot import name '_load_signature_file'`.

- [ ] **Step 4: Implement the loader**

Append to `genoray/_signatures.py`:

```python
import pooch  # noqa: E402  (kept near use for clarity)

# COSMIC reference signatures (v3.4). The filename convention is
# COSMIC_v{ver}_{SBS,DBS,ID}_{genome}.txt with a `Type` header column.
# NOTE: URLs MUST be verified against the current COSMIC release (Step 5 below);
# the catalogue host occasionally changes document paths.
_COSMIC_BASE = "https://cancer.sanger.ac.uk/signatures/documents"

# Map (kind, version, genome) -> (url, known_hash). known_hash is None until
# pinned (Step 5); pooch will warn but still download when None.
_COSMIC_REGISTRY: dict[tuple[str, str, str], tuple[str, str | None]] = {
    # filled in Step 5 after verifying URLs and recording sha256 hashes
}

_KIND_TOKEN = {"SBS96": "SBS", "DBS78": "DBS", "ID83": "ID"}


def _load_signature_file(path: str | Path) -> pl.DataFrame:
    """Parse a COSMIC-style signature TSV into a `MutationType`-first frame."""
    df = pl.read_csv(Path(path), separator="\t")
    first = df.columns[0]
    if first != "MutationType":
        df = df.rename({first: "MutationType"})
    return df


def cosmic_signatures(
    kind: Kind,
    *,
    version: str = "3.4",
    genome: str = "GRCh38",
) -> pl.DataFrame:
    """Fetch (and cache) the COSMIC reference signatures for ``kind``.

    Parameters
    ----------
    kind
        One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
    version
        COSMIC signature release (default ``"3.4"``).
    genome
        Reference build for SBS/DBS (``"GRCh37"`` or ``"GRCh38"``). Ignored for
        ID83 (indel signatures are build-independent in the COSMIC release).

    Returns
    -------
    pl.DataFrame
        A ``MutationType`` column (in genoray's canonical codebook order for
        ``kind``) followed by one column per COSMIC signature, ready to pass to
        :func:`fit_signatures`.
    """
    if kind not in _KIND_TOKEN:
        raise ValueError(f"Unknown kind {kind!r}; choose from {list(_KIND_TOKEN)}.")
    eff_genome = "GRCh37" if kind == "ID83" else genome
    key = (kind, version, eff_genome)
    if key not in _COSMIC_REGISTRY:
        raise ValueError(
            f"No COSMIC URL registered for {key}. Register it in "
            "genoray/_signatures.py:_COSMIC_REGISTRY."
        )
    url, known_hash = _COSMIC_REGISTRY[key]
    local = pooch.retrieve(url=url, known_hash=known_hash)
    df = _load_signature_file(local)

    # Reindex to genoray's canonical row order so it aligns with mutation_matrix.
    order = labels(kind)
    df = (
        pl.DataFrame({"MutationType": order})
        .join(df, on="MutationType", how="left")
    )
    return df
```

- [ ] **Step 5: Populate and verify the COSMIC registry**

This step records real URLs and hashes (cannot be fabricated). For each `kind`:

1. Find the current download URL on https://cancer.sanger.ac.uk/signatures/downloads/
   for COSMIC v3.4 (files named like `COSMIC_v3.4_SBS_GRCh38.txt`,
   `COSMIC_v3.4_DBS_GRCh38.txt`, `COSMIC_v3.4_ID_GRCh37.txt`).
2. Fetch once with `known_hash=None` to download and print the sha256:

```bash
pixi run python - <<'PY'
import pooch
# replace each URL with the verified COSMIC v3.4 link
urls = {
  ("SBS96","3.4","GRCh38"): "<SBS_GRCh38_url>",
  ("DBS78","3.4","GRCh38"): "<DBS_GRCh38_url>",
  ("ID83","3.4","GRCh37"):  "<ID_GRCh37_url>",
}
for key, url in urls.items():
    path = pooch.retrieve(url=url, known_hash=None)
    print(key, "sha256:" + pooch.file_hash(path))
PY
```

3. Paste the verified `(url, "sha256:...")` pairs into `_COSMIC_REGISTRY` in
   `genoray/_signatures.py`, e.g.:

```python
_COSMIC_REGISTRY = {
    ("SBS96", "3.4", "GRCh38"): ("https://.../COSMIC_v3.4_SBS_GRCh38.txt", "sha256:abc..."),
    ("DBS78", "3.4", "GRCh38"): ("https://.../COSMIC_v3.4_DBS_GRCh38.txt", "sha256:def..."),
    ("ID83",  "3.4", "GRCh37"): ("https://.../COSMIC_v3.4_ID_GRCh37.txt",  "sha256:0a1..."),
}
```

- [ ] **Step 6: Run the parse tests (no network) and the network smoke test**

Run (offline parse tests must pass without network):
```bash
pixi run pytest tests/test_signatures.py -k "load_signature" -v
```
Expected: PASS.

Then verify the network path once the registry is populated:
```bash
pixi run pytest tests/test_signatures.py -k "cosmic" -m network -v
```
Expected: PASS (downloads, caches, 96 rows in canonical order).

- [ ] **Step 7: Commit**

```bash
git add genoray/_signatures.py tests/test_signatures.py tests/data/cosmic_mini.txt
git commit -m "feat(signatures): add pooch-backed cosmic_signatures loader"
```

---

## Task 6: `SparseVar.assign_signatures`

Thin convenience: `mutation_matrix(kind)` → `fit_signatures(...)`, defaulting the
reference to `cosmic_signatures(kind)` and accepting a DataFrame or a TSV path.

**Files:**
- Modify: `genoray/_svar.py` (add method after `mutation_matrix`, ends `:1607`; add import)
- Test: `tests/test_svar_mutations.py` (reuses the `annotated_svar` fixture there)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar_mutations.py`:

```python
import numpy as np
import polars as pl

from genoray import SparseVar


def _toy_sbs_reference():
    # cover all 96 SBS rows with two signatures so any catalogue aligns
    from genoray._mutcat import labels

    rows = labels("SBS96")
    n = len(rows)
    return pl.DataFrame(
        {
            "MutationType": rows,
            "SBS_A": np.linspace(1, 2, n) / np.linspace(1, 2, n).sum(),
            "SBS_B": np.linspace(2, 1, n) / np.linspace(2, 1, n).sum(),
        }
    )


def test_assign_signatures_with_explicit_reference(annotated_svar):
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    ref = _toy_sbs_reference()
    act = svar.assign_signatures("SBS96", reference=ref)
    assert act.columns[0] == "Sample"
    assert "cosine_similarity" in act.columns
    assert set(act["Sample"].to_list()).issubset(set(svar.available_samples))
    # activities are nonnegative
    assert (act.select(["SBS_A", "SBS_B"]).to_numpy() >= 0).all()
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_svar_mutations.py -k assign_signatures -v`
Expected: FAIL — `AttributeError: 'SparseVar' object has no attribute 'assign_signatures'`.

- [ ] **Step 3: Add the import in `genoray/_svar.py`**

In `genoray/_svar.py`, after the existing `from ._reference import Reference` (line 35), add:

```python
from ._signatures import cosmic_signatures, fit_signatures
```

- [ ] **Step 4: Implement the method**

In `genoray/_svar.py`, immediately after the `mutation_matrix` method (which ends at line 1607, just after its `return count_matrix(...)`), add this method to the `SparseVar` class (same indentation as `mutation_matrix`):

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
        """Refit this object's mutation catalogue against COSMIC signatures.

        Builds the ``kind`` catalogue via :meth:`mutation_matrix` and decomposes
        it into per-sample activities via :func:`genoray.fit_signatures`.

        Parameters
        ----------
        kind
            One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
        reference
            Reference signatures as a Polars ``DataFrame`` (``MutationType`` +
            signature columns), a path to a COSMIC-style TSV, or ``None`` to fetch
            the default COSMIC set via :func:`genoray.cosmic_signatures`.
        count
            Counting unit passed to :meth:`mutation_matrix`.
        max_delta, min_activity
            Forwarded to :func:`genoray.fit_signatures`.

        Returns
        -------
        pl.DataFrame
            One row per sample: ``Sample``, one column per signature, and
            ``cosine_similarity``.
        """
        catalogue = self.mutation_matrix(kind, count=count)
        if reference is None:
            ref = cosmic_signatures(kind)
        elif isinstance(reference, pl.DataFrame):
            ref = reference
        else:
            from ._signatures import _load_signature_file

            ref = _load_signature_file(reference)
        return fit_signatures(
            catalogue, ref, max_delta=max_delta, min_activity=min_activity
        )
```

Note: `Path` is already imported in `_svar.py` (used throughout); `Literal` and
`pl` are already imported. No new top-level imports beyond Step 3.

- [ ] **Step 5: Run it to verify pass**

Run: `pixi run pytest tests/test_svar_mutations.py -k assign_signatures -v`
Expected: PASS.

- [ ] **Step 6: Run the full svar-mutations suite (regression)**

Run: `pixi run pytest tests/test_svar_mutations.py -v`
Expected: PASS (existing tests + the new one).

- [ ] **Step 7: Commit**

```bash
git add genoray/_svar.py tests/test_svar_mutations.py
git commit -m "feat(svar): add SparseVar.assign_signatures convenience"
```

---

## Task 7: Public exports

**Files:**
- Modify: `genoray/__init__.py`
- Test: `tests/test_signatures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signatures.py`:

```python
def test_public_exports():
    import genoray

    assert hasattr(genoray, "fit_signatures")
    assert hasattr(genoray, "cosmic_signatures")
    assert "fit_signatures" in genoray.__all__
    assert "cosmic_signatures" in genoray.__all__
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_signatures.py -k public_exports -v`
Expected: FAIL — `AssertionError` (attributes/`__all__` missing).

- [ ] **Step 3: Add exports in three places**

In `genoray/__init__.py`:

(a) Extend `__all__` (line 16):

```python
__all__ = [
    "PGEN",
    "Reference",
    "VCF",
    "Reader",
    "SparseVar",
    "exprs",
    "fit_signatures",
    "cosmic_signatures",
]
```

(b) Add to the `_LAZY` dict (after the `"exprs"` entry, line 24):

```python
    "exprs": ("genoray.exprs", None),
    "fit_signatures": ("genoray._signatures", "fit_signatures"),
    "cosmic_signatures": ("genoray._signatures", "cosmic_signatures"),
}
```

(c) Add to the `TYPE_CHECKING` block (after line 51):

```python
    from ._signatures import cosmic_signatures as cosmic_signatures
    from ._signatures import fit_signatures as fit_signatures
```

- [ ] **Step 4: Run it to verify pass**

Run: `pixi run pytest tests/test_signatures.py -k public_exports -v`
Expected: PASS.

- [ ] **Step 5: Verify import stays lazy (no eager scipy/pooch import cost)**

Run: `pixi run python -c "import genoray; print('fit_signatures' in genoray.__all__)"`
Expected: prints `True` with no error.

- [ ] **Step 6: Commit**

```bash
git add genoray/__init__.py tests/test_signatures.py
git commit -m "feat(signatures): export fit_signatures and cosmic_signatures"
```

---

## Task 8: Document the public surface in SKILL.md

**Files:**
- Modify: `skills/genoray-api/SKILL.md` (mutation-catalogue section ends ~`:388`)

- [ ] **Step 1: Add a "Signature refitting" subsection**

In `skills/genoray-api/SKILL.md`, after the existing `mutation_matrix`
documentation block (around line 388), insert:

````markdown
### Signature refitting (COSMIC)

Decompose a catalogue into per-sample COSMIC signature activities.

```python
import genoray

ref = genoray.cosmic_signatures("SBS96")        # pooch-fetched + cached
cat = svar.mutation_matrix("SBS96")              # MutationType + sample cols
act = genoray.fit_signatures(cat, ref)           # activities + cosine_similarity

# convenience: mutation_matrix -> fit_signatures in one call
act = svar.assign_signatures("SBS96")                       # default COSMIC ref
act = svar.assign_signatures("SBS96", reference=ref, min_activity=0.01)
act = svar.assign_signatures("SBS96", reference="my_sigs.txt")  # TSV path
```

Signatures:
- `cosmic_signatures(kind, *, version="3.4", genome="GRCh38") -> pl.DataFrame`
  — fetches/caches the COSMIC reference set for `kind ∈ {"SBS96","DBS78","ID83"}`.
  Returns a `MutationType` column (canonical codebook order) + one column per
  signature. `genome` is ignored for `ID83`.
- `fit_signatures(catalogue, reference, *, max_delta=0.01, min_activity=0.005) -> pl.DataFrame`
  — sparse forward-selection refit (NNLS + cosine-guided add + min-activity
  prune). Aligns rows by joining on `MutationType` (raises `ValueError` if the
  catalogue has a type missing from the reference). Returns one row per sample:
  `Sample`, one Float column per signature (counts; `0.0` if unselected), and
  `cosine_similarity`.
- `SparseVar.assign_signatures(kind, *, reference=None, count="allele", max_delta=0.01, min_activity=0.005) -> pl.DataFrame`
  — `mutation_matrix(kind, count=...)` then `fit_signatures(...)`. `reference`
  accepts a `pl.DataFrame`, a TSV path, or `None` (defaults to `cosmic_signatures(kind)`).

Out of scope (v1): de novo extraction, opportunity normalization, bootstrap CIs,
plotting.
````

- [ ] **Step 2: Verify the doc renders sensibly**

Run: `pixi run python -c "print(open('skills/genoray-api/SKILL.md').read().count('assign_signatures'))"`
Expected: prints a number ≥ 2.

- [ ] **Step 3: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(signatures): document refitting API in SKILL.md"
```

---

## Task 9: SigProfilerAssignment cross-check (calibration)

Mirrors `tests/test_mutcat_calibration.py`: skip unless SigProfilerAssignment is
installed; run in the `sigprofiler` pixi env. Asserts genoray's `fit_signatures`
agrees with SPA on a small synthetic catalogue built from known COSMIC signatures.

**Files:**
- Create: `tests/test_signatures_calibration.py`

- [ ] **Step 1: Write the calibration test**

Create `tests/test_signatures_calibration.py`:

```python
"""Calibration: cross-check genoray.fit_signatures against SigProfilerAssignment.

Skips automatically when SigProfilerAssignment is not installed. Run in the
dedicated env:
    pixi run -e sigprofiler pytest tests/test_signatures_calibration.py -v
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("SigProfilerAssignment")

from genoray import cosmic_signatures, fit_signatures  # noqa: E402

pytestmark = [pytest.mark.sigprofiler, pytest.mark.network]


def test_fit_matches_spa_on_synthetic_sbs96(tmp_path):
    # Build a catalogue from a known mix of two real COSMIC signatures.
    ref = cosmic_signatures("SBS96")
    sig_cols = [c for c in ref.columns if c != "MutationType"]
    # pick two well-known signatures present in COSMIC v3.4
    a, b = "SBS1", "SBS5"
    assert a in sig_cols and b in sig_cols
    W = ref.select([a, b]).to_numpy()
    h_true = np.array([300.0, 700.0])
    counts = np.rint(W @ h_true).astype(np.int64)

    catalogue = pl.DataFrame(
        {"MutationType": ref["MutationType"], "sample1": counts}
    )

    # genoray refit
    act = fit_signatures(catalogue, ref)
    g_a = act.filter(pl.col("Sample") == "sample1")[a].item()
    g_b = act.filter(pl.col("Sample") == "sample1")[b].item()

    # SigProfilerAssignment refit on the same matrix
    from SigProfilerAssignment import Analyzer as spa  # noqa: N813

    matrix_path = tmp_path / "samples.txt"
    catalogue.rename({"MutationType": "MutationType"}).write_csv(
        matrix_path, separator="\t"
    )
    out_dir = tmp_path / "spa_out"
    spa.cosmic_fit(
        samples=str(matrix_path),
        output=str(out_dir),
        input_type="matrix",
        cosmic_version=3.4,
        genome_build="GRCh38",
        collapse_to_SBS96=True,
        verbose=False,
    )
    spa_act = pl.read_csv(
        next(out_dir.rglob("*Activities*.txt")), separator="\t"
    )
    # SPA activities: rows = samples, columns = signatures
    spa_row = spa_act.filter(pl.col(spa_act.columns[0]) == "sample1")
    spa_a = float(spa_row[a].item()) if a in spa_act.columns else 0.0
    spa_b = float(spa_row[b].item()) if b in spa_act.columns else 0.0

    # Activities should agree within a modest tolerance (both fit the same data).
    total = g_a + g_b
    assert g_a / total == pytest.approx(spa_a / (spa_a + spa_b), abs=0.1)
    assert g_b / total == pytest.approx(spa_b / (spa_a + spa_b), abs=0.1)
```

- [ ] **Step 2: Verify it skips in the default env**

Run: `pixi run pytest tests/test_signatures_calibration.py -v`
Expected: SKIPPED (SigProfilerAssignment not installed in default env).

- [ ] **Step 3: Run the cross-check in the sigprofiler env**

Run: `pixi run -e sigprofiler pytest tests/test_signatures_calibration.py -v`
Expected: PASS. If SPA's `cosmic_fit` API name/args differ in the installed
version, adjust the call to match its current signature (the assertion logic
stays the same: compare normalized activity fractions for SBS1/SBS5).

- [ ] **Step 4: Commit**

```bash
git add tests/test_signatures_calibration.py
git commit -m "test(signatures): cross-check fit_signatures against SigProfilerAssignment"
```

---

## Task 10: Full regression + lint

**Files:** none (verification only)

- [ ] **Step 1: Run the full default-env suite (skip network)**

Run: `pixi run pytest -m "not network" -v`
Expected: PASS (calibration + network tests skipped/deselected).

- [ ] **Step 2: Lint and format**

Run:
```bash
ruff check genoray tests
ruff format --check genoray tests
```
Expected: no errors. If `ruff format --check` reports changes, run
`ruff format genoray tests` and re-stage.

- [ ] **Step 3: Typecheck**

Run: `pixi run typecheck`
Expected: no new errors in `genoray/_signatures.py` or `genoray/_svar.py`.

- [ ] **Step 4: Commit any lint/format fixups**

```bash
git add -A
git commit -m "chore(signatures): lint/format fixups" || echo "nothing to commit"
```

---

## Self-review notes

- **Spec coverage:**
  - `fit_signatures` pure function — Tasks 2–4.
  - `cosmic_signatures` pooch loader — Task 5.
  - `SparseVar.assign_signatures` convenience — Task 6.
  - Faithful forward-selection refit (NNLS + cosine add + min-activity prune) — Task 3 (`_fit_one`).
  - Activities + `cosine_similarity`, rows=sample orientation — Task 4.
  - Reference fetched on demand via pooch; no bundling — Task 5.
  - Empty/all-zero sample → zeros + cosine 0.0 — Task 3 (`test_fit_one_zero_sample`).
  - Row-alignment by join + `ValueError` on missing type — Task 4.
  - Exports + lazy import — Task 7.
  - SKILL.md update (required by CLAUDE.md) — Task 8.
  - SPA cross-check mirroring mutcat calibration — Task 9.
  - Out-of-scope items documented in spec + SKILL.md — Task 8.
- **Type consistency:** `Kind = Literal["SBS96","DBS78","ID83"]`; `fit_signatures(catalogue, reference, *, max_delta, min_activity)`, `_fit_one(W, m, *, max_delta, min_activity) -> (ndarray, float)`, `cosmic_signatures(kind, *, version, genome)`, `_load_signature_file(path) -> pl.DataFrame`, `SparseVar.assign_signatures(kind, *, reference, count, max_delta, min_activity)` — names and signatures are used identically across Tasks 4–9.
- **Known live action:** Task 5 Step 5 requires fetching real COSMIC URLs/hashes (cannot be fabricated offline). Parsing is tested against `tests/data/cosmic_mini.txt` so the non-network path is fully covered; the real fetch is `network`-marked.
- **API name surfaced in mutcat plan:** `mutation_matrix` returns `MutationType` + sample columns — consumed verbatim by `fit_signatures`.

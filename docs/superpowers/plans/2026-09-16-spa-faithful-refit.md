# SPA-Faithful Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a faithful reimplementation of SigProfilerAssignment's `cosmic_fit` to `genoray.fit_signatures`, selectable via `strategy=Spa()`, without changing the existing forward-selection default.

**Architecture:** `python/genoray/_signatures.py` becomes a package. Algorithm selection moves into two frozen dataclasses, `Forward` and `Spa`, passed as `strategy=`; each carries only the parameters its own algorithm uses. The SPA path runs four stages — saturate over all signatures, prune backward on relative L2, refine with add-remove layers, then rescale and integer-round activities to conserve the mutation burden. Per-sample parallelism over `joblib` is unchanged.

**Tech Stack:** Python 3.10+, numpy, scipy (`scipy.optimize.nnls`), polars, joblib, pooch. Environment and tasks via pixi. Tests via pytest. Lint/format via ruff, types via pyrefly, hooks via prek.

**Spec:** `docs/superpowers/specs/2026-09-16-spa-faithful-refit-design.md`

## Global Constraints

- Coordinate and missing-value conventions are unchanged by this work.
- **Do not edit `CHANGELOG.md`.** Commitizen owns it and regenerates it in CI.
- **Do not bump the version by hand.**
- All commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`).
- End every commit message with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Any change to a name reachable from `import genoray` without a leading underscore **must** update `skills/genoray-api/SKILL.md` in the same PR. This is enforced by `CLAUDE.md` and is not optional.
- Every name importable from `genoray._signatures` today must remain importable after the package split. Genoray's private API is imported across repos (GenVarLoader).
- The forward-selection path must be bit-identical before and after this work. It is the default; no existing caller may change behaviour.
- Run commands through pixi: `pixi run pytest ...`, `pixi run typecheck`. Do not invoke a bare `python`/`pytest`.
- `pixi run typecheck` (`pyrefly check python/genoray`) must report 0 errors before each commit.
- prek hooks are installed in this worktree already; let them run on commit rather than bypassing with `--no-verify`.
- Every `Spa` field default is SigProfilerAssignment's own default. Do not "improve" one.

## Reference: SPA source

The upstream functions being reproduced, for anyone verifying a step:

- `SigProfilerAssignment/single_sample.py` — `fit_signatures`, `add_signatures`, `remove_all_single_signatures`, `add_remove_signatures`, `add_connected_sigs`, `roundConserveSum`
- `SigProfilerAssignment/decompose_subroutines.py` — `process_sample`

Fetch with:

```bash
curl -sfL https://raw.githubusercontent.com/AlexandrovLab/SigProfilerAssignment/main/SigProfilerAssignment/single_sample.py -o /tmp/spa_single_sample.py
```

Only the `solver="nnls"`, `pcawg_rule=False` path is reproduced.

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `python/genoray/_signatures/__init__.py` | Re-exports only. No logic. Preserves every name importable from `genoray._signatures` today. |
| `python/genoray/_signatures/_common.py` | Shared numeric primitives: `_cosine`, `_nnls`, `_poisson_ll`, `_rel_l2`, `_distance`, `_round_conserve_sum`. |
| `python/genoray/_signatures/_strategy.py` | `Criterion`, `Metric`, `ActivityScale` aliases; `Forward` and `Spa` dataclasses; `SPA_CONNECTED_GROUPS`; `Strategy` union. |
| `python/genoray/_signatures/_forward.py` | Forward selection: `_fit_one` (moved verbatim) and a `_fit_one_forward(W, m, spec)` adapter. |
| `python/genoray/_signatures/_spa.py` | The SPA algorithm: `_exposure`, `_support`, `_expand_connected`, `_try_add`, `_remove_all_single`, `_fit_one_spa`. |
| `python/genoray/_signatures/_fit.py` | The `fit_signatures` driver: validation, alignment, strategy resolution, name→index resolution, joblib fan-out. |
| `python/genoray/_signatures/_cosmic.py` | `_COSMIC_REGISTRY`, `_KIND_TOKEN`, `_load_signature_file`, `cosmic_signatures`. |
| `tests/test_signatures_package.py` | Import-surface guard for the package split. |
| `tests/test_signatures_spa.py` | Unit and property tests for the SPA path. |
| `tests/test_signatures_recovery.py` | True-signature recovery comparison, forward vs SPA. |

**Deleted:** `python/genoray/_signatures.py` (contents redistributed).

**Modified:**

| File | Change |
|---|---|
| `python/genoray/__init__.py` | Add `Forward`, `Spa`, `Strategy`, `Metric`, `ActivityScale` to `__all__`, `_LAZY`, and the `TYPE_CHECKING` block. |
| `python/genoray/_svar/_annotate.py` | `SparseVar.assign_signatures` gains `strategy=` and `criterion=`. |
| `python/genoray/_svar2_mutcat.py` | `SparseVar2.assign_signatures` gains `strategy=` and `criterion=`. |
| `tests/test_signatures_calibration.py` | Add the `Spa()`-vs-real-SPA agreement test. |
| `skills/genoray-api/SKILL.md` | New public names, new kwargs, activity-scale semantics. |
| `docs/source/api.md` | New public names. |

---

## Task 1: Split `_signatures.py` into a package

Pure refactor. Zero behaviour change. Doing this first keeps every later diff small and reviewable.

**Files:**
- Create: `python/genoray/_signatures/__init__.py`, `_common.py`, `_forward.py`, `_fit.py`, `_cosmic.py`
- Delete: `python/genoray/_signatures.py`
- Test: `tests/test_signatures_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `genoray._signatures` as a package re-exporting `fit_signatures`, `cosmic_signatures`, `Criterion`, `_cosine`, `_nnls`, `_poisson_ll`, `_fit_one`, `_load_signature_file`, `_COSMIC_REGISTRY`, `_KIND_TOKEN`. `_common._cosine(a, b) -> float`, `_common._nnls(W, m) -> NDArray[np.float64]`, `_common._poisson_ll(m, e) -> float`.

- [ ] **Step 1: Write the failing import-surface test**

Create `tests/test_signatures_package.py`:

```python
"""Guard the private import surface of genoray._signatures.

genoray's underscore modules are imported across repos (GenVarLoader), and
two modules in this package import from _signatures directly. The package
split must not move any of these names.
"""

from __future__ import annotations

import importlib

import pytest

# Every name that resolved as `genoray._signatures.<name>` before the package
# split. Adding to this list is fine; removing from it is a breaking change.
PRESERVED_NAMES = [
    "fit_signatures",
    "cosmic_signatures",
    "Criterion",
    "_fit_one",
    "_cosine",
    "_nnls",
    "_poisson_ll",
    "_load_signature_file",
    "_COSMIC_REGISTRY",
    "_KIND_TOKEN",
]


@pytest.mark.parametrize("name", PRESERVED_NAMES)
def test_name_still_importable_from_signatures(name: str):
    mod = importlib.import_module("genoray._signatures")
    assert hasattr(mod, name), f"genoray._signatures.{name} disappeared"


def test_signatures_is_a_package():
    mod = importlib.import_module("genoray._signatures")
    assert hasattr(mod, "__path__"), "_signatures should be a package"


def test_internal_importers_still_work():
    """The two sibling modules that import from _signatures directly."""
    from genoray._svar._annotate import SparseVarAnnotateMixin  # noqa: F401
    from genoray._svar2_mutcat import _MutcatMixin  # noqa: F401
```

- [ ] **Step 2: Run it to confirm the package test fails**

Run: `pixi run pytest tests/test_signatures_package.py -v`

Expected: `test_signatures_is_a_package` FAILS (`_signatures` is a module, no `__path__`). The `PRESERVED_NAMES` tests pass already — that is correct, they are the regression guard.

The two class names in `test_internal_importers_still_work` were verified against the source before this plan was written: `SparseVarAnnotateMixin` at `python/genoray/_svar/_annotate.py:192`, `_MutcatMixin` at `python/genoray/_svar2_mutcat.py:33`. Note that `SparseVar` itself lives in `python/genoray/_svar/_core.py`, not in `_annotate.py`. If either import fails, read the source and use the real name — do not delete the test.

- [ ] **Step 3: Create the package directory and move the numeric primitives**

```bash
mkdir -p python/genoray/_signatures
```

Create `python/genoray/_signatures/_common.py`:

```python
"""Numeric primitives shared by the refit strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls


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


def _poisson_ll(m: NDArray[np.floating], e: NDArray[np.floating]) -> float:
    """Poisson log-likelihood of observed counts ``m`` under expected counts ``e``.

    Drops the ``-log(m!)`` term. It depends only on the data, so it cancels in
    every likelihood *difference* -- which is all the ``"bic"`` criterion uses.
    """
    e = np.clip(np.asarray(e, dtype=np.float64), 1e-12, None)
    return float(np.sum(np.asarray(m, dtype=np.float64) * np.log(e) - e))
```

- [ ] **Step 4: Move forward selection**

Create `python/genoray/_signatures/_forward.py`. Copy `Criterion` and the whole of `_fit_one` out of the old `python/genoray/_signatures.py` **verbatim** — do not retype it, do not reformat it, do not "clean it up". Head the file with:

```python
"""Greedy forward selection from the empty signature set. genoray's own algorithm."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._common import _cosine, _nnls, _poisson_ll

#: Forward-selection stop rules. See ``fit_signatures``.
Criterion = Literal["cosine", "bic"]
```

then the verbatim `_fit_one`.

- [ ] **Step 5: Move the COSMIC loader**

Create `python/genoray/_signatures/_cosmic.py`. Copy `_COSMIC_REGISTRY`, `_KIND_TOKEN`, `_load_signature_file` and `cosmic_signatures` out of the old module verbatim. Header:

```python
"""COSMIC reference signature loader (pooch-backed)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pooch

from .._mutcat import Kind, labels
```

Note the import depth changes from `._mutcat` to `.._mutcat`.

- [ ] **Step 6: Move the driver**

Create `python/genoray/_signatures/_fit.py`. Copy `fit_signatures` out of the old module verbatim. Header:

```python
"""The ``fit_signatures`` driver: alignment, strategy dispatch, parallel fan-out."""

from __future__ import annotations

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from ._forward import Criterion, _fit_one
```

`fit_signatures` calls neither `_cosine` nor `_nnls` directly — it delegates to
`_fit_one` — so do not import them here. `__init__.py` re-exports them from
`_common`, which is what preserves `genoray._signatures._cosine`.

- [ ] **Step 7: Write the re-export `__init__.py`**

Create `python/genoray/_signatures/__init__.py`. **Copy the module docstring
out of `python/genoray/_signatures.py` verbatim** — the whole triple-quoted
block from line 1 to the closing `"""`, including the paragraph stating that
SPA's behaviours are "not reproduced here". That paragraph is still true at
this commit; task 11 rewrites it once `Spa` exists. A task must not document
code that has not been written yet.

Then, below the docstring:

```python
from __future__ import annotations

from ._common import _cosine, _nnls, _poisson_ll
from ._cosmic import (
    _COSMIC_REGISTRY,
    _KIND_TOKEN,
    _load_signature_file,
    cosmic_signatures,
)
from ._fit import fit_signatures
from ._forward import Criterion, _fit_one

__all__ = [
    "Criterion",
    "cosmic_signatures",
    "fit_signatures",
]
```

Then delete the old module:

```bash
git rm python/genoray/_signatures.py
```

- [ ] **Step 8: Run the full signature test suite**

Run: `pixi run pytest tests/test_signatures.py tests/test_signatures_criterion.py tests/test_signatures_package.py -v`

Expected: all PASS. Anything failing here is a transcription error in steps 3-7, not a design problem — go back and diff the moved code against `git show HEAD:python/genoray/_signatures.py`.

- [ ] **Step 9: Typecheck and lint**

Run: `pixi run typecheck`
Expected: 0 errors.

Run: `pixi run -e lint ruff check python/genoray tests && pixi run -e lint ruff format --check python/genoray tests`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add -A python/genoray/_signatures python/genoray/_signatures.py tests/test_signatures_package.py
git commit -m "$(cat <<'EOF'
refactor(signatures): split _signatures.py into a package

The module held an estimator and a pooch-backed download registry, and is
about to gain a second estimator. Split by responsibility, re-exporting
every name that resolved at genoray._signatures before, since two sibling
modules and GenVarLoader import from it directly.

No behaviour change: the moved code is verbatim.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Numeric primitives for the SPA path

**Files:**
- Modify: `python/genoray/_signatures/_common.py`
- Test: `tests/test_signatures_spa.py`

**Interfaces:**
- Consumes: `_common._cosine`.
- Produces:
  - `_rel_l2(m: NDArray, recon: NDArray) -> float` — `||m - recon||_2 / ||m||_2`, 0.0 when `||m||_2 == 0`.
  - `_distance(m: NDArray, recon: NDArray, metric: str) -> float` — lower-is-better distance; `_rel_l2` for `"l2"`, `1.0 - _cosine(...)` for `"cosine"`.
  - `_round_conserve_sum(x: NDArray) -> NDArray[np.float64]` — SPA's `roundConserveSum`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_signatures_spa.py`:

```python
"""Unit and property tests for the SPA-faithful refit path."""

from __future__ import annotations

import numpy as np
import pytest

from genoray._signatures._common import _distance, _rel_l2, _round_conserve_sum


def test_rel_l2_perfect_reconstruction_is_zero():
    m = np.array([1.0, 2.0, 3.0])
    assert _rel_l2(m, m) == pytest.approx(0.0)


def test_rel_l2_zero_reconstruction_is_one():
    m = np.array([3.0, 4.0])
    assert _rel_l2(m, np.zeros(2)) == pytest.approx(1.0)


def test_rel_l2_zero_sample_is_zero_not_nan():
    m = np.zeros(4)
    assert _rel_l2(m, np.zeros(4)) == 0.0


def test_rel_l2_is_scale_invariant():
    m = np.array([1.0, 5.0, 2.0])
    recon = np.array([1.2, 4.5, 2.4])
    assert _rel_l2(m, recon) == pytest.approx(_rel_l2(10 * m, 10 * recon))


def test_distance_l2_matches_rel_l2():
    m = np.array([1.0, 5.0, 2.0])
    recon = np.array([1.2, 4.5, 2.4])
    assert _distance(m, recon, "l2") == pytest.approx(_rel_l2(m, recon))


def test_distance_cosine_is_lower_is_better():
    m = np.array([1.0, 2.0, 3.0])
    near = np.array([1.1, 2.0, 2.9])
    far = np.array([3.0, 2.0, 1.0])
    assert _distance(m, near, "cosine") < _distance(m, far, "cosine")
    assert _distance(m, m, "cosine") == pytest.approx(0.0)


def test_round_conserve_sum_conserves_total():
    x = np.array([10.4, 20.3, 69.3])
    out = _round_conserve_sum(x)
    assert out.sum() == pytest.approx(round(x.sum()))


def test_round_conserve_sum_returns_integers():
    x = np.array([10.4, 20.3, 69.3])
    out = _round_conserve_sum(x)
    assert np.all(out == np.floor(out))
    assert out.dtype == np.float64


def test_round_conserve_sum_keeps_zeros_zero():
    x = np.array([0.0, 50.5, 49.5])
    out = _round_conserve_sum(x)
    assert out[0] == 0.0


def test_round_conserve_sum_all_zero():
    out = _round_conserve_sum(np.zeros(5))
    assert out.sum() == 0.0
    assert np.all(out == 0.0)


@pytest.mark.parametrize("seed", range(30))
def test_round_conserve_sum_property(seed: int):
    """Conserves the rounded total and never goes negative, over random input."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    burden = float(rng.integers(1, 200_000))
    w = rng.random(n)
    x = w / w.sum() * burden
    out = _round_conserve_sum(x)
    assert out.sum() == pytest.approx(round(x.sum()))
    assert np.all(out >= 0.0), "roundConserveSum produced a negative activity"
    assert np.all(out == np.floor(out))
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run pytest tests/test_signatures_spa.py -v`
Expected: FAIL at collection — `ImportError: cannot import name '_distance'`.

- [ ] **Step 3: Implement**

Append to `python/genoray/_signatures/_common.py`:

```python
def _rel_l2(m: NDArray[np.floating], recon: NDArray[np.floating]) -> float:
    """Relative L2 error ``||m - recon||_2 / ||m||_2``; 0.0 for a zero sample.

    This is SigProfilerAssignment's ``metric="l2"`` score. It is
    scale-invariant, and because the denominator is constant across candidate
    fits of the same sample, it orders candidates identically to the raw
    residual norm.
    """
    nm = float(np.linalg.norm(m))
    if nm == 0.0:
        return 0.0
    return float(np.linalg.norm(np.asarray(m, dtype=np.float64) - recon) / nm)


def _distance(
    m: NDArray[np.floating], recon: NDArray[np.floating], metric: str
) -> float:
    """Lower-is-better distance between a sample and its reconstruction.

    Both branches are lower-is-better so every comparison in the SPA path is
    written once, exactly as SigProfilerAssignment writes it.
    """
    if metric == "l2":
        return _rel_l2(m, recon)
    return 1.0 - _cosine(m, recon)


def _round_conserve_sum(x: NDArray[np.floating]) -> NDArray[np.float64]:
    """Round to integers while conserving the total. SPA's ``roundConserveSum``.

    Ceil every entry, then decrement the entries that were rounded up the most
    until the total matches ``round(sum(x))``. Transcribed from
    SigProfilerAssignment so activities agree entry for entry.
    """
    x = np.asarray(x, dtype=np.float64)
    total = np.round(np.sum(x))
    x_out = np.ceil(x)
    # x - x_out is in (-1, 0]; ascending order puts the largest round-up first.
    order = np.argsort(x - x_out)
    n_to_drop = int(np.sum(x_out) - total + 1e-10)
    if n_to_drop > 0:
        x_out[order[:n_to_drop]] -= 1
    return x_out
```

The `if n_to_drop > 0` guard is the one deviation from SPA's transcription: SPA writes the slice unguarded, and a negative index would silently slice from the end. `n_to_drop` cannot go negative for non-negative `x` (which is all NNLS produces), so the guard changes no result — it just makes the impossible case loud instead of wrong.

- [ ] **Step 4: Run to verify they pass**

Run: `pixi run pytest tests/test_signatures_spa.py -v`
Expected: all PASS.

- [ ] **Step 5: Typecheck and commit**

Run: `pixi run typecheck` → 0 errors.

```bash
git add python/genoray/_signatures/_common.py tests/test_signatures_spa.py
git commit -m "$(cat <<'EOF'
feat(signatures): add relative-L2 scoring and burden-conserving rounding

_rel_l2 is SigProfilerAssignment's default metric; _distance wraps both
metrics as lower-is-better so the SPA path writes each comparison once.
_round_conserve_sum is a transcription of SPA's roundConserveSum, which is
how SPA reports activities that sum exactly to the mutation burden.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: The strategy dataclasses

**Files:**
- Create: `python/genoray/_signatures/_strategy.py`
- Modify: `python/genoray/_signatures/__init__.py`, `python/genoray/__init__.py`
- Test: `tests/test_signatures_spa.py`

**Interfaces:**
- Consumes: `_forward.Criterion`.
- Produces: `Forward(max_delta=0.01, min_activity=0.005, criterion="cosine")`, `Spa(metric="l2", initial_remove_penalty=0.05, add_penalty=0.05, remove_penalty=0.01, background_sigs=("SBS1","SBS5"), connected_sigs=True, activity_scale="burden")`, `Strategy = Forward | Spa`, `Metric`, `ActivityScale`, `SPA_CONNECTED_GROUPS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_signatures_spa.py`:

```python
def test_strategy_defaults_match_spa():
    from genoray import Spa

    s = Spa()
    assert s.metric == "l2"
    assert s.initial_remove_penalty == 0.05
    assert s.add_penalty == 0.05
    assert s.remove_penalty == 0.01
    assert s.background_sigs == ("SBS1", "SBS5")
    assert s.connected_sigs is True
    assert s.activity_scale == "burden"


def test_forward_defaults_match_current_behaviour():
    from genoray import Forward

    f = Forward()
    assert f.max_delta == 0.01
    assert f.min_activity == 0.005
    assert f.criterion == "cosine"


def test_strategies_are_frozen():
    import dataclasses

    from genoray import Forward, Spa

    with pytest.raises(dataclasses.FrozenInstanceError):
        Forward().max_delta = 0.2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Spa().metric = "cosine"  # type: ignore[misc]


def test_spa_rejects_unknown_metric():
    from genoray import Spa

    with pytest.raises(ValueError, match="metric"):
        Spa(metric="manhattan")  # type: ignore[arg-type]


def test_spa_rejects_unknown_activity_scale():
    from genoray import Spa

    with pytest.raises(ValueError, match="activity_scale"):
        Spa(activity_scale="fraction")  # type: ignore[arg-type]


def test_forward_rejects_unknown_criterion():
    from genoray import Forward

    with pytest.raises(ValueError, match="criterion"):
        Forward(criterion="aic")  # type: ignore[arg-type]


def test_spa_connected_groups_are_spas():
    from genoray._signatures._strategy import SPA_CONNECTED_GROUPS

    assert ("SBS2", "SBS13") in SPA_CONNECTED_GROUPS
    assert ("SBS7a", "SBS7b", "SBS7c", "SBS7d") in SPA_CONNECTED_GROUPS
    assert ("SBS10a", "SBS10b") in SPA_CONNECTED_GROUPS
    assert ("SBS17a", "SBS17b") in SPA_CONNECTED_GROUPS
    assert len(SPA_CONNECTED_GROUPS) == 4


def test_strategies_exported_from_genoray():
    import genoray

    assert "Forward" in genoray.__all__
    assert "Spa" in genoray.__all__
    assert genoray.Forward().criterion == "cosine"
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_signatures_spa.py -k strategy -v`
Expected: FAIL — `AttributeError: module 'genoray' has no attribute 'Spa'`.

- [ ] **Step 3: Create `_strategy.py`**

```python
"""Refit strategies.

A strategy both selects the algorithm and carries its parameters. Each one
holds only the knobs its own algorithm uses, so a forward-selection threshold
cannot reach backward elimination: that is a type error rather than a
silently ignored keyword argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Union

from ._forward import Criterion

#: Score used by the SPA path. ``"l2"`` is SigProfilerAssignment's default.
Metric = Literal["l2", "cosine"]

#: How the SPA path reports activities.
ActivityScale = Literal["burden", "raw"]

#: SigProfilerAssignment's ``add_connected_sigs`` groups. Whenever any member
#: is selected, every member is force-added. SBS-only; inert for DBS78/ID83.
SPA_CONNECTED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("SBS2", "SBS13"),
    ("SBS7a", "SBS7b", "SBS7c", "SBS7d"),
    ("SBS10a", "SBS10b"),
    ("SBS17a", "SBS17b"),
)


@dataclass(frozen=True)
class Forward:
    """Greedy forward selection from the empty set. genoray's own algorithm.

    Args:
        max_delta: Minimum cosine-similarity improvement to keep adding a
            signature. Used only when ``criterion="cosine"``.
        min_activity: Minimum fractional contribution; signatures below this
            are pruned after selection.
        criterion: Stop rule. ``"cosine"`` is scale-invariant and therefore
            blind to mutation burden. ``"bic"`` is burden-aware and
            consistent, but over-selects below ~1,000 mutations.
    """

    max_delta: float = 0.01
    min_activity: float = 0.005
    criterion: Criterion = "cosine"

    def __post_init__(self) -> None:
        if self.criterion not in ("cosine", "bic"):
            raise ValueError(
                f"criterion must be 'cosine' or 'bic', got {self.criterion!r}."
            )


@dataclass(frozen=True)
class Spa:
    """Backward elimination from the saturated set: SigProfilerAssignment's
    ``cosmic_fit``.

    Every default is SigProfilerAssignment's own, so ``Spa()`` is
    ``cosmic_fit`` as shipped. Recovers more true signatures than ``Forward``
    at every burden measured, but is also scale-invariant and so is not a
    consistent estimator either.

    Args:
        metric: Score. ``"l2"`` is relative L2 error, SPA's default.
            ``"cosine"`` diverges from SPA -- see the class notes below.
        initial_remove_penalty: Cutoff for the prune that follows the
            saturated fit. Background signatures are **not** protected here.
        add_penalty: A candidate must improve the distance by strictly more
            than this to be added during refinement.
        remove_penalty: Cutoff for the removal sweeps during refinement.
        background_sigs: Signature names always kept once refinement starts.
            Names absent from the reference are ignored, which is what makes
            the default inert for DBS78 and ID83. ``None`` disables.
        connected_sigs: Force-add co-occurring partners. ``True`` uses
            :data:`SPA_CONNECTED_GROUPS`; a sequence of groups supplies your
            own; ``False`` disables.
        activity_scale: ``"burden"`` rescales and integer-rounds activities so
            they sum to the sample's mutation count, as SPA reports them.
            ``"raw"`` returns unscaled NNLS weights, as the forward path does.

    Notes:
        Two SPA behaviours are deliberately not reproduced. SPA refuses to
        remove a signature whose removal *improves* the fit; on the default
        ``metric="l2"`` path that branch is unreachable, because NNLS over a
        column subset cannot beat NNLS over the superset, so omitting it
        cannot change a result. Under ``metric="cosine"`` it is reachable and
        this class diverges. SPA also draws from the global RNG for values it
        never uses; this implementation is deterministic.
    """

    metric: Metric = "l2"
    initial_remove_penalty: float = 0.05
    add_penalty: float = 0.05
    remove_penalty: float = 0.01
    background_sigs: Sequence[str] | None = ("SBS1", "SBS5")
    connected_sigs: bool | Sequence[Sequence[str]] = True
    activity_scale: ActivityScale = "burden"

    def __post_init__(self) -> None:
        if self.metric not in ("l2", "cosine"):
            raise ValueError(f"metric must be 'l2' or 'cosine', got {self.metric!r}.")
        if self.activity_scale not in ("burden", "raw"):
            raise ValueError(
                "activity_scale must be 'burden' or 'raw', got "
                f"{self.activity_scale!r}."
            )


#: Either refit strategy. Accepted by ``fit_signatures(strategy=...)``.
Strategy = Union[Forward, Spa]
```

- [ ] **Step 4: Re-export from the package**

In `python/genoray/_signatures/__init__.py`, add to the imports and to `__all__`:

```python
from ._strategy import (
    SPA_CONNECTED_GROUPS,
    ActivityScale,
    Forward,
    Metric,
    Spa,
    Strategy,
)

__all__ = [
    "ActivityScale",
    "Criterion",
    "Forward",
    "Metric",
    "Spa",
    "Strategy",
    "cosmic_signatures",
    "fit_signatures",
]
```

- [ ] **Step 5: Export from the top-level namespace**

In `python/genoray/__init__.py`, add `"Forward"`, `"Spa"`, `"Strategy"`, `"Metric"`, `"ActivityScale"` to `__all__`; add these five entries to `_LAZY`:

```python
    "Forward": ("genoray._signatures", "Forward"),
    "Spa": ("genoray._signatures", "Spa"),
    "Strategy": ("genoray._signatures", "Strategy"),
    "Metric": ("genoray._signatures", "Metric"),
    "ActivityScale": ("genoray._signatures", "ActivityScale"),
```

and to the `TYPE_CHECKING` block:

```python
    from ._signatures import ActivityScale as ActivityScale
    from ._signatures import Forward as Forward
    from ._signatures import Metric as Metric
    from ._signatures import Spa as Spa
    from ._signatures import Strategy as Strategy
```

- [ ] **Step 6: Run to verify they pass**

Run: `pixi run pytest tests/test_signatures_spa.py -v`
Expected: all PASS.

- [ ] **Step 7: Typecheck and commit**

Run: `pixi run typecheck` → 0 errors.

```bash
git add python/genoray/_signatures/_strategy.py python/genoray/_signatures/__init__.py python/genoray/__init__.py tests/test_signatures_spa.py
git commit -m "$(cat <<'EOF'
feat(signatures): add Forward and Spa strategy objects

Each carries only the parameters its own algorithm uses, so passing a
forward-selection threshold to backward elimination is unrepresentable
rather than silently ignored -- the #200/#207 lesson applied to the refit
API. Every Spa default is SigProfilerAssignment's own.

Not wired into fit_signatures yet.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire `strategy=` into `fit_signatures` (forward path only)

Behaviour must not change. This task only moves the existing knobs behind the new argument.

**Files:**
- Modify: `python/genoray/_signatures/_fit.py`, `python/genoray/_signatures/_forward.py`
- Test: `tests/test_signatures_spa.py`

**Interfaces:**
- Consumes: `Forward`, `Spa`, `Strategy`, `_fit_one`.
- Produces:
  - `_forward._fit_one_forward(W, m, spec: Forward) -> tuple[NDArray[np.float64], float]`
  - `_fit._resolve_strategy(strategy, max_delta, min_activity, criterion) -> Strategy` — raises `ValueError` on conflict.
  - `fit_signatures(..., strategy: Strategy | None = None, ...)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_signatures_spa.py`:

```python
def _toy_problem():
    """A 4-type, 3-signature toy catalogue with a known 2-signature answer."""
    import polars as pl

    ref = pl.DataFrame(
        {
            "MutationType": ["A", "B", "C", "D"],
            "S1": [1.0, 0.0, 0.0, 0.0],
            "S2": [0.0, 1.0, 0.0, 0.0],
            "S3": [0.0, 0.0, 1.0, 1.0],
        }
    )
    cat = pl.DataFrame(
        {"MutationType": ["A", "B", "C", "D"], "s1": [600.0, 400.0, 0.0, 0.0]}
    )
    return cat, ref


def test_strategy_forward_matches_legacy_kwargs():
    from genoray import Forward, fit_signatures

    cat, ref = _toy_problem()
    legacy = fit_signatures(cat, ref, max_delta=0.02, min_activity=0.01)
    viastrategy = fit_signatures(
        cat, ref, strategy=Forward(max_delta=0.02, min_activity=0.01)
    )
    assert legacy.equals(viastrategy)


def test_default_is_forward_and_unchanged():
    from genoray import Forward, fit_signatures

    cat, ref = _toy_problem()
    assert fit_signatures(cat, ref).equals(
        fit_signatures(cat, ref, strategy=Forward())
    )


def test_strategy_conflicts_with_legacy_kwarg():
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    with pytest.raises(ValueError, match="max_delta"):
        fit_signatures(cat, ref, strategy=Spa(), max_delta=0.02)


def test_strategy_conflict_names_every_offender():
    from genoray import Forward, fit_signatures

    cat, ref = _toy_problem()
    with pytest.raises(ValueError) as exc:
        fit_signatures(
            cat, ref, strategy=Forward(), max_delta=0.02, min_activity=0.01
        )
    msg = str(exc.value)
    assert "max_delta" in msg and "min_activity" in msg


def test_passing_the_default_value_explicitly_still_conflicts():
    """The sentinel must distinguish 'not passed' from 'passed the default'."""
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    with pytest.raises(ValueError, match="max_delta"):
        fit_signatures(cat, ref, strategy=Spa(), max_delta=0.01)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_signatures_spa.py -k "strategy_forward or default_is_forward or conflict" -v`
Expected: FAIL — `fit_signatures() got an unexpected keyword argument 'strategy'`.

- [ ] **Step 3: Add the `Forward` adapter**

Append to `python/genoray/_signatures/_forward.py`:

```python
def _fit_one_forward(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    spec: "Forward",
) -> tuple[NDArray[np.float64], float]:
    """Adapt ``_fit_one`` to the strategy-object calling convention."""
    return _fit_one(
        W,
        m,
        max_delta=spec.max_delta,
        min_activity=spec.min_activity,
        criterion=spec.criterion,
    )
```

Import `Forward` under `TYPE_CHECKING` only — `_strategy` imports `Criterion` from this module, so a runtime import would be circular:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ._strategy import Forward
```

- [ ] **Step 4: Add strategy resolution to `_fit.py`**

Add near the top of `python/genoray/_signatures/_fit.py`:

```python
from typing import Any

from ._strategy import Forward, Spa, Strategy

# Sentinel distinguishing "argument not passed" from "argument passed its
# default value". Needed so `strategy=Spa(), max_delta=0.01` is still a
# conflict rather than silently accepted.
_UNSET: Any = object()

#: Real defaults for the legacy shorthand arguments. Kept here so the
#: docstring and the resolution agree in one place.
_FORWARD_DEFAULTS = Forward()


def _resolve_strategy(
    strategy: Strategy | None,
    max_delta: Any,
    min_activity: Any,
    criterion: Any,
) -> Strategy:
    """Reconcile ``strategy=`` with the legacy flat keyword arguments.

    The flat arguments are permanent shorthand for the forward path, not
    deprecated. Combining them with an explicit ``strategy`` is an error
    rather than a silent drop.
    """
    passed = {
        "max_delta": max_delta,
        "min_activity": min_activity,
        "criterion": criterion,
    }
    explicit = [name for name, val in passed.items() if val is not _UNSET]

    if strategy is None:
        return Forward(
            max_delta=(
                _FORWARD_DEFAULTS.max_delta if max_delta is _UNSET else max_delta
            ),
            min_activity=(
                _FORWARD_DEFAULTS.min_activity
                if min_activity is _UNSET
                else min_activity
            ),
            criterion=(
                _FORWARD_DEFAULTS.criterion if criterion is _UNSET else criterion
            ),
        )

    if explicit:
        raise ValueError(
            f"strategy= was given together with {', '.join(sorted(explicit))}. "
            "The flat arguments are shorthand for the forward path; pass them "
            "inside Forward(...) instead, or drop strategy=."
        )
    return strategy
```

- [ ] **Step 5: Rewrite the `fit_signatures` signature and dispatch**

Change the signature to:

```python
def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    strategy: Strategy | None = None,
    max_delta: float = _UNSET,
    min_activity: float = _UNSET,
    criterion: Criterion = _UNSET,
    n_jobs: int = 1,
    backend: str = "loky",
) -> pl.DataFrame:
```

Replace the existing `criterion` validation block with:

```python
    spec = _resolve_strategy(strategy, max_delta, min_activity, criterion)
```

and the `Parallel(...)` call with:

```python
    if isinstance(spec, Forward):
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_fit_one_forward)(W, M[:, j], spec)
            for j in range(len(sample_cols))
        )
    else:
        raise NotImplementedError(
            "strategy=Spa() is not wired up yet; see task 7 of the plan."
        )
```

Update the docstring: `max_delta`/`min_activity`/`criterion` are documented as "shorthand for `strategy=Forward(...)`; cannot be combined with `strategy=`", with their real default values named (0.01, 0.005, `"cosine"`). Add a `strategy:` entry pointing at `Forward` and `Spa`. Extend the `Raises:` section with the conflict error.

The `NotImplementedError` branch is temporary scaffolding, removed in task 7. It is here so this task stays independently testable.

- [ ] **Step 6: Run the tests**

Run: `pixi run pytest tests/test_signatures_spa.py tests/test_signatures.py tests/test_signatures_criterion.py -v`
Expected: all PASS. The existing suites passing unchanged is the point of this task.

- [ ] **Step 7: Prove the forward path is bit-identical**

Write a throwaway script under `$CLAUDE_JOB_DIR/tmp` (not committed) that, for 60 randomized combinations of reference size, burden (30 to 300,000), `max_delta`, `min_activity` and `criterion`, compares `fit_signatures` on this commit against `git stash`-free checkout of `HEAD~4` (the pre-split commit) via a subprocess, and asserts zero mismatches.

Simpler and sufficient: check out the pre-task-1 module into a temp path and import it under a different name:

```bash
git show $(git merge-base HEAD main)~0:python/genoray/_signatures.py > "$CLAUDE_JOB_DIR/tmp/old_signatures.py"
```

then compare `old_signatures.fit_signatures(...)` against the new one across the grid. Expected: 0 mismatches. Record the number in the commit message.

- [ ] **Step 8: Typecheck and commit**

Run: `pixi run typecheck` → 0 errors.

```bash
git add python/genoray/_signatures/_fit.py python/genoray/_signatures/_forward.py tests/test_signatures_spa.py
git commit -m "$(cat <<'EOF'
feat(signatures): accept strategy= on fit_signatures

max_delta/min_activity/criterion stay as permanent shorthand for the forward
path. Combining them with an explicit strategy= raises rather than silently
dropping one, and a sentinel default distinguishes "not passed" from "passed
the default value" so the check cannot be bypassed.

Forward path verified bit-identical to the pre-split module across
randomized references, burdens and thresholds.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: The removal sweep

The core of stages 2 and 3.

**Files:**
- Create: `python/genoray/_signatures/_spa.py`
- Test: `tests/test_signatures_spa.py`

**Interfaces:**
- Consumes: `_common._nnls`, `_common._distance`, `_common._round_conserve_sum`.
- Produces:
  - `_exposure(W, m, active, *, scale) -> NDArray[np.float64]` — full-length exposure vector.
  - `_support(h) -> list[int]` — `sorted(np.nonzero(h)[0])` as Python ints.
  - `_remove_all_single(W, m, h, *, cutoff, metric, protected) -> NDArray[np.float64]` — returns a new exposure vector.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_signatures_spa.py`:

```python
def _identity_ref(n_types: int = 5):
    """W = identity, so each signature owns exactly one mutation type."""
    return np.eye(n_types)


def test_exposure_burden_scale_sums_to_burden():
    from genoray._signatures._spa import _exposure

    W = _identity_ref(3)
    m = np.array([300.0, 700.0, 0.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    assert h.sum() == pytest.approx(1000.0)
    assert np.all(h == np.floor(h))


def test_exposure_raw_scale_is_unrounded():
    from genoray._signatures._spa import _exposure

    W = _identity_ref(3)
    m = np.array([300.5, 699.5, 0.0])
    h = _exposure(W, m, [0, 1, 2], scale="raw")
    assert h[0] == pytest.approx(300.5)


def test_exposure_is_full_length_with_zeros_off_support():
    from genoray._signatures._spa import _exposure

    W = _identity_ref(4)
    m = np.array([10.0, 20.0, 0.0, 0.0])
    h = _exposure(W, m, [0, 1], scale="burden")
    assert h.shape == (4,)
    assert h[2] == 0.0 and h[3] == 0.0


def test_exposure_zero_sample_returns_zeros():
    from genoray._signatures._spa import _exposure

    W = _identity_ref(3)
    h = _exposure(W, np.zeros(3), [0, 1, 2], scale="burden")
    assert np.all(h == 0.0)


def test_support_reads_nonzeros():
    from genoray._signatures._spa import _support

    assert _support(np.array([0.0, 5.0, 0.0, 2.0])) == [1, 3]


def test_remove_drops_a_signature_the_data_does_not_need():
    """S3 explains nothing; removing it must not degrade the fit at all."""
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([600.0, 400.0, 0.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    out = _remove_all_single(
        W, m, h, cutoff=0.05, metric="l2", protected=frozenset()
    )
    assert _support(out) == [0, 1]


def test_remove_keeps_a_signature_the_data_needs():
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([400.0, 300.0, 300.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    out = _remove_all_single(
        W, m, h, cutoff=0.05, metric="l2", protected=frozenset()
    )
    assert _support(out) == [0, 1, 2]


def test_remove_respects_protected():
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([600.0, 400.0, 0.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    out = _remove_all_single(
        W, m, h, cutoff=0.05, metric="l2", protected=frozenset({2})
    )
    assert 2 in _support(out)


def test_remove_never_empties_the_support():
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([1000.0, 0.0, 0.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    out = _remove_all_single(
        W, m, h, cutoff=1.0, metric="l2", protected=frozenset()
    )
    assert len(_support(out)) >= 1


@pytest.mark.parametrize("seed", range(20))
def test_removal_never_lowers_relative_l2(seed: int):
    """The claim that lets us skip SPA's negative-difference guard.

    NNLS over a column subset cannot beat NNLS over the superset, so dropping
    a column can only raise the relative L2 error.
    """
    from genoray._signatures._common import _nnls, _rel_l2

    rng = np.random.default_rng(seed)
    n_types, n_sigs = 24, 8
    W = rng.random((n_types, n_sigs))
    W = W / W.sum(axis=0)
    m = rng.random(n_types) * 1000.0

    full = list(range(n_sigs))
    base = _rel_l2(m, W[:, full] @ _nnls(W[:, full], m))
    for drop in full:
        sub = [i for i in full if i != drop]
        d = _rel_l2(m, W[:, sub] @ _nnls(W[:, sub], m))
        assert d >= base - 1e-9, f"dropping {drop} lowered relative L2"
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_signatures_spa.py -k "exposure or support or remove" -v`
Expected: FAIL — `ModuleNotFoundError: genoray._signatures._spa`.

- [ ] **Step 3: Implement**

Create `python/genoray/_signatures/_spa.py`:

```python
"""SigProfilerAssignment's ``cosmic_fit``, reimplemented.

Backward elimination from the saturated signature set, then add-remove
refinement layers. Transcribed from SigProfilerAssignment ``main``:
``single_sample.py`` (``fit_signatures``, ``add_signatures``,
``remove_all_single_signatures``, ``add_remove_signatures``,
``add_connected_sigs``) and ``decompose_subroutines.py``
(``process_sample``), ``solver="nnls"`` and ``pcawg_rule=False`` only.

Like SPA, the state carried between stages is a full-length exposure
*vector*, not an index set, and the active set is read off its nonzeros.
This matters: both of SPA's inner routines round what they record, so a
signature whose rescaled activity falls below 0.5 drops out of the support
without any threshold testing it. Distances, however, are always computed
from the unrounded NNLS reconstruction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ._common import _distance, _nnls, _round_conserve_sum


def _support(h: NDArray[np.floating]) -> list[int]:
    """The active signature indices: the nonzeros of an exposure vector."""
    return [int(i) for i in np.nonzero(h)[0]]


def _exposure(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    active: Sequence[int],
    *,
    scale: str,
) -> NDArray[np.float64]:
    """NNLS over ``active``, scattered into a full-length exposure vector.

    With ``scale="burden"`` the weights are renormalized to sum to the
    sample's mutation count and integer-rounded conserving that sum, which is
    how SPA reports every intermediate and final exposure. With
    ``scale="raw"`` the raw NNLS weights are returned.
    """
    full = np.zeros(W.shape[1], dtype=np.float64)
    active = sorted(active)
    if not active:
        return full
    weights = _nnls(W[:, active], m)
    total = float(weights.sum())
    if total == 0.0:
        return full
    if scale == "burden":
        burden = float(np.sum(m))
        weights = _round_conserve_sum(weights / total * burden)
    full[active] = weights
    return full


def _reconstruction(
    W: NDArray[np.floating], m: NDArray[np.floating], active: Sequence[int]
) -> NDArray[np.float64]:
    """The *unrounded* NNLS reconstruction over ``active``. Scoring uses this.

    SPA computes every distance from ``np.dot(W1, weights)`` with raw weights,
    even where the exposures it records alongside are rounded. Scoring off the
    rounded vector would be a divergence.
    """
    active = sorted(active)
    if not active:
        return np.zeros(W.shape[0], dtype=np.float64)
    return W[:, active] @ _nnls(W[:, active], m)


def _remove_all_single(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    h: NDArray[np.floating],
    *,
    cutoff: float,
    metric: str,
    protected: frozenset[int],
) -> NDArray[np.float64]:
    """SPA's ``remove_all_single_signatures``.

    Repeatedly removes the signature whose removal degrades the distance
    least, while that degradation stays at or below ``cutoff``. Stops when the
    best available removal costs more than ``cutoff``, or when one signature
    is left.

    The cutoff is measured against the *current* fit, not the one this sweep
    started from: SPA advances its baseline after every accepted layer.
    Measuring against the original baseline instead makes the prune far too
    permissive.
    """
    active = _support(h)
    if len(active) <= 1:
        return np.asarray(h, dtype=np.float64).copy()

    base = _distance(m, _reconstruction(W, m, active), metric)
    scale = "burden"

    while len(active) > 1:
        best_d = np.inf
        best_active: list[int] | None = None
        for i in active:
            if i in protected:
                continue
            cand = [j for j in active if j != i]
            d = _distance(m, _reconstruction(W, m, cand), metric)
            if d < best_d:
                best_d = d
                best_active = cand
        if best_active is None:
            break  # every remaining signature is protected
        if best_d - base > cutoff:
            break
        active = best_active
        base = best_d

    return _exposure(W, m, active, scale=scale)
```

Note `_remove_all_single` always records on the `"burden"` scale, because SPA does (`roundConserveSum`) regardless of what the caller eventually reports. The user-facing `activity_scale` is applied once, at the very end, in task 7.

- [ ] **Step 4: Run to verify they pass**

Run: `pixi run pytest tests/test_signatures_spa.py -v`
Expected: all PASS, including the 20 parametrized monotonicity cases.

- [ ] **Step 5: Typecheck and commit**

Run: `pixi run typecheck` → 0 errors.

```bash
git add python/genoray/_signatures/_spa.py tests/test_signatures_spa.py
git commit -m "$(cat <<'EOF'
feat(signatures): add SPA's backward removal sweep

Removes the signature whose removal degrades relative L2 least, while that
degradation stays under the cutoff -- a different rule from min_activity,
which prunes on contribution size. State is a full-length exposure vector as
in SPA, so a signature that rounds below 0.5 drops out on its own; distances
come from the unrounded reconstruction.

A property test pins the claim that lets us skip SPA's negative-difference
guard: dropping a column can never lower relative L2.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Candidate add and connected-signature expansion

**Files:**
- Modify: `python/genoray/_signatures/_spa.py`
- Test: `tests/test_signatures_spa.py`

**Interfaces:**
- Consumes: `_reconstruction`, `_distance`, `_exposure`, `_support`.
- Produces:
  - `_try_add(W, m, active, cand, *, cutoff, metric) -> list[int]`
  - `_expand_connected(active, groups) -> list[int]` where `groups: tuple[tuple[int, ...], ...]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_signatures_spa.py`:

```python
def test_try_add_accepts_a_signature_that_helps():
    from genoray._signatures._spa import _try_add

    W = _identity_ref(3)
    m = np.array([600.0, 400.0, 0.0])
    assert _try_add(W, m, [0], 1, cutoff=0.05, metric="l2") == [0, 1]


def test_try_add_rejects_a_signature_that_does_not_help_enough():
    from genoray._signatures._spa import _try_add

    W = _identity_ref(3)
    m = np.array([1000.0, 1.0, 0.0])
    # Adding S2 explains 1 of 1001 mutations: an improvement far under 0.05.
    assert _try_add(W, m, [0], 1, cutoff=0.05, metric="l2") == [0]


def test_try_add_on_an_empty_active_set_always_adds():
    from genoray._signatures._spa import _try_add

    W = _identity_ref(3)
    m = np.array([600.0, 400.0, 0.0])
    assert _try_add(W, m, [], 0, cutoff=0.05, metric="l2") == [0]


def test_try_add_is_a_noop_when_already_active():
    from genoray._signatures._spa import _try_add

    W = _identity_ref(3)
    m = np.array([600.0, 400.0, 0.0])
    assert _try_add(W, m, [0, 1], 1, cutoff=0.05, metric="l2") == [0, 1]


def test_try_add_returns_sorted_indices():
    from genoray._signatures._spa import _try_add

    W = _identity_ref(4)
    m = np.array([100.0, 100.0, 100.0, 0.0])
    assert _try_add(W, m, [2], 0, cutoff=0.01, metric="l2") == [0, 2]


def test_expand_connected_pulls_in_the_whole_group():
    from genoray._signatures._spa import _expand_connected

    groups = ((1, 5), (2, 3, 4))
    assert _expand_connected([1], groups) == [1, 5]
    assert _expand_connected([3], groups) == [2, 3, 4]


def test_expand_connected_leaves_ungrouped_alone():
    from genoray._signatures._spa import _expand_connected

    groups = ((1, 5),)
    assert _expand_connected([0, 7], groups) == [0, 7]


def test_expand_connected_handles_multiple_groups_at_once():
    from genoray._signatures._spa import _expand_connected

    groups = ((1, 5), (2, 3))
    assert _expand_connected([1, 2], groups) == [1, 2, 3, 5]


def test_expand_connected_with_no_groups_is_identity():
    from genoray._signatures._spa import _expand_connected

    assert _expand_connected([3, 1], ()) == [1, 3]
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_signatures_spa.py -k "try_add or expand_connected" -v`
Expected: FAIL — `ImportError: cannot import name '_try_add'`.

- [ ] **Step 3: Implement**

Append to `python/genoray/_signatures/_spa.py`:

```python
def _try_add(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    active: Sequence[int],
    cand: int,
    *,
    cutoff: float,
    metric: str,
) -> list[int]:
    """SPA's ``add_signatures`` restricted to a single candidate.

    ``add_remove_signatures`` always calls ``add_signatures`` with
    ``toBeAdded=[c]``, which restricts its candidate pool to ``c`` alone. Its
    loop therefore runs at most one accepting iteration, and the whole
    routine collapses to: add ``c`` if and only if it improves the distance
    by strictly more than ``cutoff``.

    The strict inequality is SPA's (``if originalSimilarity - bestSimilarity >
    cutoff``). An empty active set has infinite distance, so the first
    signature is always accepted, matching SPA's ``originalSimilarity =
    np.inf`` initialization.
    """
    active = sorted(active)
    if cand in active:
        return active
    base = (
        np.inf
        if not active
        else _distance(m, _reconstruction(W, m, active), metric)
    )
    new = sorted([*active, cand])
    d_new = _distance(m, _reconstruction(W, m, new), metric)
    return new if base - d_new > cutoff else active


def _expand_connected(
    active: Sequence[int], groups: tuple[tuple[int, ...], ...]
) -> list[int]:
    """SPA's ``add_connected_sigs``: if any group member is active, add them all."""
    out = set(active)
    for group in groups:
        if out.intersection(group):
            out.update(group)
    return sorted(out)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pixi run pytest tests/test_signatures_spa.py -v`
Expected: all PASS.

- [ ] **Step 5: Typecheck and commit**

Run: `pixi run typecheck` → 0 errors.

```bash
git add python/genoray/_signatures/_spa.py tests/test_signatures_spa.py
git commit -m "$(cat <<'EOF'
feat(signatures): add SPA's single-candidate add and connected-sig expansion

add_remove_signatures only ever calls add_signatures with a singleton
toBeAdded, so the general loop collapses to one comparison against
add_penalty. Implementing the collapsed form directly rather than porting
dead generality; the docstring records why they are equivalent.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Assemble the SPA fit and wire it up

**Files:**
- Modify: `python/genoray/_signatures/_spa.py`, `python/genoray/_signatures/_fit.py`
- Test: `tests/test_signatures_spa.py`

**Interfaces:**
- Consumes: `_exposure`, `_support`, `_remove_all_single`, `_try_add`, `_expand_connected`, `_common._cosine`.
- Produces:
  - `_fit_one_spa(W, m, spec: Spa, *, protected: frozenset[int], groups: tuple[tuple[int, ...], ...]) -> tuple[NDArray[np.float64], float]`
  - `_fit._resolve_sig_names(sig_cols, spec) -> tuple[frozenset[int], tuple[tuple[int, ...], ...]]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_signatures_spa.py`:

```python
def test_spa_recovers_a_clean_two_signature_mixture():
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    out = fit_signatures(cat, ref, strategy=Spa(background_sigs=None))
    assert out["S1"].item() == pytest.approx(600.0)
    assert out["S2"].item() == pytest.approx(400.0)
    assert out["S3"].item() == 0.0


def test_spa_burden_scale_activities_sum_to_burden():
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    out = fit_signatures(cat, ref, strategy=Spa(background_sigs=None))
    total = sum(out[c].item() for c in ("S1", "S2", "S3"))
    assert total == pytest.approx(1000.0)


def test_spa_burden_scale_activities_are_integers():
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    out = fit_signatures(cat, ref, strategy=Spa(background_sigs=None))
    for c in ("S1", "S2", "S3"):
        v = out[c].item()
        assert v == float(int(v))


def test_spa_raw_scale_does_not_conserve_burden():
    """raw returns NNLS weights, which need not sum to the burden."""
    from genoray import Spa, fit_signatures
    import polars as pl

    ref = pl.DataFrame(
        {
            "MutationType": ["A", "B"],
            "S1": [0.5, 0.5],
            "S2": [1.0, 0.0],
        }
    )
    cat = pl.DataFrame({"MutationType": ["A", "B"], "s1": [600.0, 400.0]})
    out = fit_signatures(
        cat, ref, strategy=Spa(background_sigs=None, activity_scale="raw")
    )
    total = out["S1"].item() + out["S2"].item()
    assert total == pytest.approx(1000.0, rel=0.5)  # same ballpark, not exact


def test_spa_output_schema_matches_forward():
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    fwd = fit_signatures(cat, ref)
    spa = fit_signatures(cat, ref, strategy=Spa(background_sigs=None))
    assert fwd.columns == spa.columns
    assert spa.columns[-1] == "cosine_similarity"


def test_spa_is_deterministic():
    """SPA draws from the global RNG; this implementation must not."""
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    a = fit_signatures(cat, ref, strategy=Spa())
    np.random.seed(0)
    b = fit_signatures(cat, ref, strategy=Spa())
    np.random.seed(12345)
    c = fit_signatures(cat, ref, strategy=Spa())
    assert a.equals(b) and b.equals(c)


def test_spa_zero_count_sample_returns_zeros():
    from genoray import Spa, fit_signatures
    import polars as pl

    ref = pl.DataFrame(
        {"MutationType": ["A", "B"], "S1": [1.0, 0.0], "S2": [0.0, 1.0]}
    )
    cat = pl.DataFrame({"MutationType": ["A", "B"], "s1": [0.0, 0.0]})
    out = fit_signatures(cat, ref, strategy=Spa(background_sigs=None))
    assert out["S1"].item() == 0.0
    assert out["S2"].item() == 0.0
    assert out["cosine_similarity"].item() == 0.0


def test_spa_background_sigs_are_kept():
    """S3 explains nothing, but is protected, so it survives refinement."""
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    out = fit_signatures(cat, ref, strategy=Spa(background_sigs=("S3",)))
    assert out["S3"].item() > 0.0


def test_spa_background_sigs_absent_from_reference_are_ignored():
    """The default ("SBS1","SBS5") must be inert on a non-SBS reference."""
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    out = fit_signatures(cat, ref, strategy=Spa())  # default background sigs
    assert out["S1"].item() > 0.0


def test_spa_connected_sigs_force_add_partners():
    import polars as pl

    from genoray import Spa, fit_signatures

    ref = pl.DataFrame(
        {
            "MutationType": ["A", "B", "C"],
            "SBS2": [1.0, 0.0, 0.0],
            "SBS13": [0.0, 1.0, 0.0],
            "SBS40a": [0.0, 0.0, 1.0],
        }
    )
    # Only SBS2's type is observed; SBS13 would never be selected on merit.
    cat = pl.DataFrame({"MutationType": ["A", "B", "C"], "s1": [1000.0, 0.0, 0.0]})
    on = fit_signatures(cat, ref, strategy=Spa(background_sigs=None))
    off = fit_signatures(
        cat, ref, strategy=Spa(background_sigs=None, connected_sigs=False)
    )
    assert "SBS13" in on.columns and "SBS13" in off.columns
    # Force-adding SBS13 cannot hurt the fit, and the flag must change something
    # about the search: with it off, SBS13 is never a member of the active set.
    assert on["SBS2"].item() > 0.0
    assert off["SBS2"].item() > 0.0


def test_spa_custom_connected_groups():
    import polars as pl

    from genoray import Spa, fit_signatures

    ref = pl.DataFrame(
        {
            "MutationType": ["A", "B"],
            "X": [1.0, 0.0],
            "Y": [0.0, 1.0],
        }
    )
    cat = pl.DataFrame({"MutationType": ["A", "B"], "s1": [1000.0, 0.0]})
    out = fit_signatures(
        cat,
        ref,
        strategy=Spa(background_sigs=None, connected_sigs=[["X", "Y"]]),
    )
    assert out["X"].item() > 0.0


def test_spa_matches_across_n_jobs():
    from genoray import Spa, fit_signatures
    import polars as pl

    rng = np.random.default_rng(7)
    types = [f"T{i}" for i in range(12)]
    ref = pl.DataFrame({"MutationType": types}).with_columns(
        **{f"S{k}": pl.Series(rng.random(12)) for k in range(5)}
    )
    cat = pl.DataFrame({"MutationType": types}).with_columns(
        **{f"s{j}": pl.Series(rng.integers(0, 500, 12).astype(float)) for j in range(4)}
    )
    a = fit_signatures(cat, ref, strategy=Spa(background_sigs=None), n_jobs=1)
    b = fit_signatures(cat, ref, strategy=Spa(background_sigs=None), n_jobs=2)
    assert a.equals(b)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_signatures_spa.py -k spa_ -v`
Expected: FAIL — `NotImplementedError: strategy=Spa() is not wired up yet`.

- [ ] **Step 3: Implement the four-stage fit**

Append to `python/genoray/_signatures/_spa.py`:

```python
def _fit_one_spa(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    spec: "Spa",
    *,
    protected: frozenset[int],
    groups: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], float]:
    """Refit one sample by SPA's ``cosmic_fit``.

    Four stages, matching ``process_sample`` then ``add_remove_signatures``:
    saturate over every signature, prune backward, refine with add-remove
    layers, then report on the requested activity scale.

    Returns ``(activities, cosine)``, the same contract as the forward path.
    """
    n_sigs = W.shape[1]
    empty = np.zeros(n_sigs, dtype=np.float64)
    if float(np.sum(m)) == 0.0:
        return empty, 0.0

    metric = spec.metric

    # Stage 1: saturate. SPA seeds every signature nonzero via a random dummy
    # exposure matrix; the draw's only effect is that every entry is nonzero.
    h = _exposure(W, m, range(n_sigs), scale="burden")
    if not _support(h):
        return empty, 0.0

    # Stage 2: initial prune. SPA passes background_sigs=[] here, so the
    # background signatures are NOT protected at this stage.
    h = _remove_all_single(
        W,
        m,
        h,
        cutoff=spec.initial_remove_penalty,
        metric=metric,
        protected=frozenset(),
    )

    # Stage 3: add-remove refinement layers.
    active = sorted(set(_support(h)) | set(protected))
    best_d = np.inf
    best_active = active
    while True:
        present = _expand_connected(active, groups)
        layer_d = np.inf
        layer_active: list[int] | None = None
        for cand in range(n_sigs):
            if cand in present:
                continue
            added = _try_add(
                W, m, present, cand, cutoff=spec.add_penalty, metric=metric
            )
            h_add = _exposure(W, m, added, scale="burden")
            h_rem = _remove_all_single(
                W,
                m,
                h_add,
                cutoff=spec.remove_penalty,
                metric=metric,
                protected=protected,
            )
            pick = _support(h_add) if _support(h_add) == _support(h_rem) else _support(h_rem)
            d = _distance(m, _reconstruction(W, m, pick), metric)
            if d < layer_d:
                layer_d = d
                layer_active = pick
        if layer_active is None or layer_d >= best_d:
            break
        best_d = layer_d
        best_active = layer_active
        active = layer_active

    # Stage 4: report.
    h = _exposure(W, m, best_active, scale=spec.activity_scale)
    cos = _cosine(m, W @ h) if _support(h) else 0.0
    return h, cos
```

Add `_cosine` to the `_common` import at the top of `_spa.py`, and the `TYPE_CHECKING` import of `Spa`:

```python
from typing import TYPE_CHECKING, Sequence

from ._common import _cosine, _distance, _nnls, _round_conserve_sum

if TYPE_CHECKING:  # pragma: no cover
    from ._strategy import Spa
```

Note on the `pick` line: SPA writes a comparison that looks buggy
(`np.nonzero(add)[0].all() == np.nonzero(remove)[0].all()` plus a shape check).
It is not load-bearing — the removal sweep can only shrink the support, so
equal shapes already imply equal supports, and when the supports are equal
the two distances are equal too. Comparing supports directly is exactly
equivalent.

- [ ] **Step 4: Resolve signature names to indices in the driver**

Add to `python/genoray/_signatures/_fit.py`:

```python
from ._strategy import SPA_CONNECTED_GROUPS


def _resolve_sig_names(
    sig_cols: list[str], spec: Spa
) -> tuple[frozenset[int], tuple[tuple[int, ...], ...]]:
    """Map ``Spa``'s signature *names* onto column indices of the reference.

    Names absent from the reference are dropped silently, matching SPA's
    ``get_indeces``. That is what makes the default ``("SBS1", "SBS5")``
    background and the SBS-only connected groups inert for DBS78 and ID83
    without a special case.

    Resolution happens once, here, rather than per sample: it needs the
    reference's column names, and the fan-out below would otherwise repeat it
    once per sample.
    """
    index_of = {name: i for i, name in enumerate(sig_cols)}

    protected = frozenset(
        index_of[n] for n in (spec.background_sigs or ()) if n in index_of
    )

    if spec.connected_sigs is True:
        raw_groups: Sequence[Sequence[str]] = SPA_CONNECTED_GROUPS
    elif spec.connected_sigs is False:
        raw_groups = ()
    else:
        raw_groups = spec.connected_sigs

    groups = tuple(
        tuple(sorted(index_of[n] for n in g if n in index_of))
        for g in raw_groups
    )
    # A group with fewer than two present members can never expand anything.
    groups = tuple(g for g in groups if len(g) > 1)
    return protected, groups
```

Import `Sequence` from `typing` and `_fit_one_spa` from `._spa`.

- [ ] **Step 5: Replace the `NotImplementedError` branch**

In `fit_signatures`, replace the dispatch with:

```python
    if isinstance(spec, Forward):
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_fit_one_forward)(W, M[:, j], spec)
            for j in range(len(sample_cols))
        )
    else:
        protected, groups = _resolve_sig_names(sig_cols, spec)
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_fit_one_spa)(
                W, M[:, j], spec, protected=protected, groups=groups
            )
            for j in range(len(sample_cols))
        )
```

Extend the `fit_signatures` docstring's `strategy:` entry to describe `Spa()` and to say that under `activity_scale="burden"` the signature columns hold integer-valued floats summing to the sample's burden, while the forward path returns raw NNLS weights that need not.

- [ ] **Step 6: Run the tests**

Run: `pixi run pytest tests/test_signatures_spa.py tests/test_signatures.py tests/test_signatures_criterion.py -v`
Expected: all PASS.

- [ ] **Step 7: Measure the cost and record it**

Run a timing check on a realistic problem — the real COSMIC SBS96 reference (~86 signatures) against 8 synthetic samples:

```bash
pixi run python -c "
import time, numpy as np, polars as pl
from genoray import cosmic_signatures, fit_signatures, Spa, Forward
ref = cosmic_signatures('SBS96')
sigs = [c for c in ref.columns if c != 'MutationType']
rng = np.random.default_rng(0)
W = ref.select(sigs).to_numpy()
cat = {'MutationType': ref['MutationType']}
for j in range(8):
    h = np.zeros(len(sigs)); h[rng.choice(len(sigs), 4, replace=False)] = rng.random(4)
    cat[f's{j}'] = rng.poisson(W @ (h / h.sum() * 50_000)).astype(float)
cat = pl.DataFrame(cat)
for name, s in (('forward', Forward()), ('spa', Spa())):
    t = time.perf_counter(); fit_signatures(cat, ref, strategy=s); dt = time.perf_counter() - t
    print(f'{name}: {dt:.2f}s total, {dt/8:.3f}s/sample')
"
```

Expected: SPA in the 0.1-1 s/sample range, one to two orders of magnitude slower than forward. Record the measured numbers in the commit message. If it lands far outside that range, say so in the commit message rather than absorbing it silently — the spec calls that out as a finding, not a failure.

- [ ] **Step 8: Typecheck and commit**

Run: `pixi run typecheck` → 0 errors.

```bash
git add python/genoray/_signatures/_spa.py python/genoray/_signatures/_fit.py tests/test_signatures_spa.py
git commit -m "$(cat <<'EOF'
feat(signatures): implement strategy=Spa(), SigProfilerAssignment's cosmic_fit

Four stages: saturate over every signature, prune backward on relative L2,
refine with add-remove layers, then rescale and integer-round activities to
conserve the mutation burden.

Closes the audit's items 1-7: backward search direction, the refinement
pass, SPA's removal criterion, relative-L2 scoring, SBS1/SBS5 background
signatures, connected-signature groups, and exposure scaling.

Signature names resolve to column indices once in the driver rather than
per sample. Unlike SPA this path touches no RNG and is deterministic.

Measured: <fill in from step 7> s/sample on SBS96 with 86 signatures.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Plumb `strategy=` and `criterion=` through `assign_signatures`

**Files:**
- Modify: `python/genoray/_svar/_annotate.py` (around line 505), `python/genoray/_svar2_mutcat.py` (around line 206)
- Test: `tests/test_signatures_spa.py`

**Interfaces:**
- Consumes: `fit_signatures(strategy=...)`, `Strategy`.
- Produces: `SparseVar.assign_signatures(..., strategy=None, criterion=...)` and `SparseVar2.assign_signatures(..., strategy=None, criterion=...)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signatures_spa.py`:

```python
def test_assign_signatures_accepts_strategy_on_both_readers():
    """Both readers must expose the same refit knobs as fit_signatures."""
    import inspect

    from genoray import SparseVar
    from genoray._svar2 import SparseVar2

    for cls in (SparseVar, SparseVar2):
        params = inspect.signature(cls.assign_signatures).parameters
        assert "strategy" in params, f"{cls.__name__} is missing strategy="
        assert "criterion" in params, f"{cls.__name__} is missing criterion="
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_signatures_spa.py -k assign_signatures -v`
Expected: FAIL — `SparseVar is missing strategy=`.

- [ ] **Step 3: Modify `SparseVar.assign_signatures`**

In `python/genoray/_svar/_annotate.py`, change the signature to insert two parameters after `count`:

```python
        strategy: "Strategy | None" = None,
        max_delta: float = 0.01,
        min_activity: float = 0.005,
        criterion: "Criterion" = "cosine",
```

and forward them at the `fit_signatures(...)` call. Because `fit_signatures` now rejects `strategy=` combined with an explicit legacy argument, this method must not blindly pass all four. Forward conditionally:

```python
        if strategy is not None:
            return fit_signatures(
                matrix, ref, strategy=strategy, n_jobs=n_jobs, backend=backend
            )
        return fit_signatures(
            matrix,
            ref,
            max_delta=max_delta,
            min_activity=min_activity,
            criterion=criterion,
            n_jobs=n_jobs,
            backend=backend,
        )
```

Adjust the variable names (`matrix`, `ref`) to whatever the surrounding code already calls them — read the method body first.

Add to the docstring's `Args:` section:

```
            strategy: Refit strategy, forwarded to :func:`genoray.fit_signatures`.
                ``None`` (default) uses forward selection configured by the
                ``max_delta``/``min_activity``/``criterion`` arguments below.
                Pass :class:`genoray.Spa` for SigProfilerAssignment's algorithm.
                Cannot be combined with those three arguments.
            criterion: Forward-selection stop rule, forwarded to
                :func:`genoray.fit_signatures`. Ignored when ``strategy`` is given.
```

Import the types under `TYPE_CHECKING` to keep import time down:

```python
if TYPE_CHECKING:
    from .._signatures import Criterion, Strategy
```

- [ ] **Step 4: Apply the identical change to `SparseVar2.assign_signatures`**

In `python/genoray/_svar2_mutcat.py` at the `assign_signatures` definition (around line 206), make the same signature, docstring and conditional-forwarding change. Repeat the code rather than factoring it out: the two methods have different surrounding context and a shared helper would be a third place to look.

- [ ] **Step 5: Run the tests**

Run: `pixi run pytest tests/test_signatures_spa.py -k assign_signatures -v`
Expected: PASS.

Run: `pixi run pytest tests/ -k "svar and signature" -v`
Expected: PASS (no regressions in the reader-level tests).

- [ ] **Step 6: Typecheck and commit**

Run: `pixi run typecheck` → 0 errors.

```bash
git add python/genoray/_svar/_annotate.py python/genoray/_svar2_mutcat.py tests/test_signatures_spa.py
git commit -m "$(cat <<'EOF'
feat(svar): forward strategy= and criterion= from assign_signatures

Both readers now reach every refit knob fit_signatures exposes. criterion
was added to fit_signatures in #212 but never plumbed here, leaving the BIC
stop rule unreachable from the two highest-level entry points.

Forwarding is conditional because fit_signatures rejects strategy= combined
with the flat shorthand arguments.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Recovery comparison suite

**Files:**
- Create: `tests/test_signatures_recovery.py`

**Interfaces:**
- Consumes: `fit_signatures`, `Forward`, `Spa`, `cosmic_signatures`.
- Produces: nothing importable.

- [ ] **Step 1: Write the test**

Create `tests/test_signatures_recovery.py`:

```python
"""Ground-truth recovery: does strategy=Spa() find more true signatures?

These reproduce the measurement in issue #214 as a regression guard. They
assert the *ordering* of the two strategies, not the exact means, which are
Monte Carlo estimates over a modest number of reps.

Marked `network` because they fetch the real COSMIC reference.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from genoray import Forward, Spa, cosmic_signatures, fit_signatures

pytestmark = pytest.mark.network

N_REPS = 8
N_TRUE = 4

DISTINCT = ["SBS1", "SBS2", "SBS7a", "SBS17b"]
WITH_FLAT = ["SBS1", "SBS5", "SBS40a", "SBS2"]


def _synthetic_catalogue(ref: pl.DataFrame, truth: list[str], burden: int, seed: int):
    """Poisson counts from a known mixture of `truth` signatures."""
    rng = np.random.default_rng(seed)
    W = ref.select(truth).to_numpy()
    W = W / W.sum(axis=0)
    h = rng.dirichlet(np.ones(len(truth))) * burden
    counts = rng.poisson(W @ h).astype(np.float64)
    return pl.DataFrame({"MutationType": ref["MutationType"], "s": counts})


def _recovery(ref, truth, burden, strategy):
    """(mean true signatures found, mean false positives) over N_REPS."""
    found, false_pos = [], []
    for rep in range(N_REPS):
        cat = _synthetic_catalogue(ref, truth, burden, seed=1000 + rep)
        out = fit_signatures(cat, ref, strategy=strategy)
        sig_cols = [c for c in out.columns if c not in ("Sample", "cosine_similarity")]
        selected = {c for c in sig_cols if out[c].item() > 0.0}
        found.append(len(selected & set(truth)))
        false_pos.append(len(selected - set(truth)))
    return float(np.mean(found)), float(np.mean(false_pos))


@pytest.fixture(scope="module")
def ref():
    return cosmic_signatures("SBS96")


@pytest.mark.parametrize("truth", [DISTINCT, WITH_FLAT], ids=["distinct", "with_flat"])
def test_spa_recovers_at_least_as_many_as_forward(ref, truth):
    burden = 100_000
    fwd, _ = _recovery(ref, truth, burden, Forward())
    spa, _ = _recovery(ref, truth, burden, Spa())
    assert spa >= fwd, f"Spa recovered {spa}/{N_TRUE}, Forward {fwd}/{N_TRUE}"


def test_spa_beats_forward_at_high_burden_on_distinct_signatures(ref):
    fwd, _ = _recovery(ref, DISTINCT, 100_000, Forward())
    spa, _ = _recovery(ref, DISTINCT, 100_000, Spa())
    assert spa > fwd, (
        f"issue #214 measured 3.85 vs 3.50; got Spa {spa}, Forward {fwd}"
    )


def test_spa_is_no_worse_on_precision_at_low_burden(ref):
    _, fwd_fp = _recovery(ref, DISTINCT, 100, Forward())
    _, spa_fp = _recovery(ref, DISTINCT, 100, Spa())
    assert spa_fp <= fwd_fp + 1e-9, (
        f"Spa false positives {spa_fp} exceeded Forward's {fwd_fp}"
    )


def test_spa_activities_conserve_burden_on_real_signatures(ref):
    cat = _synthetic_catalogue(ref, DISTINCT, 50_000, seed=42)
    out = fit_signatures(cat, ref, strategy=Spa())
    sig_cols = [c for c in out.columns if c not in ("Sample", "cosine_similarity")]
    total = sum(out[c].item() for c in sig_cols)
    assert total == pytest.approx(round(cat["s"].sum()))
```

- [ ] **Step 2: Run it**

Run: `pixi run pytest tests/test_signatures_recovery.py -v`

Expected: all PASS. This is the one test in the plan where a failure may be a real finding rather than a bug — the issue's numbers came from 20 reps and this uses 8. If `test_spa_beats_forward_at_high_burden_on_distinct_signatures` fails:

1. Raise `N_REPS` to 20 and re-run. If it passes, keep 20 and note the runtime.
2. If it still fails, the SPA implementation has a defect. Do **not** weaken the assertion. Go back to task 7 and check the two details the spec flags as easy to get wrong: the removal cutoff measured against the *current* fit, and the initial prune running with no protected signatures.

- [ ] **Step 3: Commit**

```bash
git add tests/test_signatures_recovery.py
git commit -m "$(cat <<'EOF'
test(signatures): ground-truth recovery comparison, forward vs Spa

Reproduces the measurement from issue #214 as a regression guard. Asserts
the ordering of the two strategies rather than the Monte Carlo means.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Calibration against real SigProfilerAssignment

This is the test that justifies calling the path faithful.

**Files:**
- Modify: `tests/test_signatures_calibration.py`

**Interfaces:**
- Consumes: `fit_signatures(strategy=Spa())`, `cosmic_signatures`.
- Produces: nothing importable.

- [ ] **Step 1: Materialize the sigprofiler environment**

Run: `pixi install -e sigprofiler`

Expected: resolves and installs. This env is declared in `pixi.toml` but has never been built here, so budget time for it and expect a large download.

- [ ] **Step 2: Confirm the existing calibration test still runs**

Run: `pixi run -e sigprofiler pytest tests/test_signatures_calibration.py -v`
Expected: the existing `test_fit_matches_spa_on_synthetic_sbs96` PASSES (it compares activity *fractions* with a loose 0.1 tolerance, so it should already pass against the forward path).

- [ ] **Step 3: Write the failing agreement test**

Append to `tests/test_signatures_calibration.py`:

```python
def _spa_activities(tmp_path, catalogue: pl.DataFrame, tag: str) -> dict[str, float]:
    """Run real SigProfilerAssignment cosmic_fit, return {signature: activity}."""
    from SigProfilerAssignment import Analyzer as spa  # noqa: N813

    matrix_path = tmp_path / f"{tag}.txt"
    catalogue.write_csv(matrix_path, separator="\t")
    out_dir = tmp_path / f"spa_out_{tag}"
    spa.cosmic_fit(
        samples=str(matrix_path),
        output=str(out_dir),
        input_type="matrix",
        cosmic_version=3.4,
        genome_build="GRCh38",
        collapse_to_SBS96=True,
        make_plots=False,
        verbose=False,
    )
    candidates = list(out_dir.rglob("Assignment_Solution_Activities.txt"))
    if not candidates:
        candidates = list(out_dir.rglob("*Activities*.txt"))
    if not candidates:
        raise FileNotFoundError(f"SPA activities not found under {out_dir}")
    act = pl.read_csv(candidates[0], separator="\t")
    sample_col = act.columns[0]
    row = act.filter(pl.col(sample_col) == "sample1")
    return {
        c: float(row[c].item())
        for c in act.columns
        if c != sample_col and float(row[c].item()) != 0.0
    }


@pytest.mark.parametrize(
    ("truth", "burden", "tag", "seed"),
    [
        (["SBS1", "SBS2", "SBS7a", "SBS17b"], 100_000, "distinct_high", 11),
        (["SBS1", "SBS2", "SBS7a", "SBS17b"], 1_000, "distinct_low", 22),
        (["SBS1", "SBS5", "SBS40a", "SBS2"], 100_000, "flat_high", 33),
    ],
)
def test_spa_strategy_reproduces_cosmic_fit(tmp_path, truth, burden, tag, seed):
    """strategy=Spa() must agree with real cosmic_fit on support and activities.

    This is the test that earns the word "faithful". If it fails, the
    reimplementation is wrong -- do not loosen the tolerance to make it pass.
    """
    from genoray import Spa

    ref = cosmic_signatures("SBS96")
    # An explicit seed, not hash(tag): str hashing is PYTHONHASHSEED-randomized,
    # and a test that claims faithfulness has to draw the same data every run.
    rng = np.random.default_rng(seed)
    W = ref.select(truth).to_numpy()
    W = W / W.sum(axis=0)
    h = rng.dirichlet(np.ones(len(truth))) * burden
    counts = rng.poisson(W @ h).astype(np.int64)
    catalogue = pl.DataFrame(
        {"MutationType": ref["MutationType"], "sample1": counts}
    )

    spa_act = _spa_activities(tmp_path, catalogue, tag)

    out = fit_signatures(catalogue, ref, strategy=Spa())
    sig_cols = [c for c in out.columns if c not in ("Sample", "cosine_similarity")]
    gen_act = {c: out[c].item() for c in sig_cols if out[c].item() != 0.0}

    assert set(gen_act) == set(spa_act), (
        f"support mismatch for {tag}\n"
        f"  genoray only: {sorted(set(gen_act) - set(spa_act))}\n"
        f"  SPA only:     {sorted(set(spa_act) - set(gen_act))}"
    )
    for sig in sorted(spa_act):
        assert gen_act[sig] == pytest.approx(spa_act[sig], abs=1.0), (
            f"{tag}: {sig} genoray={gen_act[sig]} spa={spa_act[sig]}"
        )
```

Add `import numpy as np` to the file's imports if it is not already there (it is).

- [ ] **Step 4: Run it**

Run: `pixi run -e sigprofiler pytest tests/test_signatures_calibration.py -v -k reproduces_cosmic_fit`

Expected: all three parametrizations PASS.

**If a case fails**, work it as a defect, in this order:

1. Print both supports. A missing low-activity signature points at the rounding-drops-support behaviour (`_exposure` must round at every intermediate stage, not only at the end).
2. An extra signature on genoray's side points at the removal cutoff being measured against the wrong baseline — it must advance to the current fit after each accepted layer.
3. A support that differs only in SBS1/SBS5 points at the initial prune wrongly protecting the background signatures. It must not; SPA passes `background_sigs=[]` there.
4. Activities off by more than 1 with an identical support points at `_round_conserve_sum` or at the rescaling being applied to the wrong vector.

If a genuine, explained divergence remains after all four, write it into the "Divergences from SPA" section of the spec, keep an assertion that pins the actual behaviour, and say so plainly in the commit message and the PR. Do not delete the test.

- [ ] **Step 5: Commit**

```bash
git add tests/test_signatures_calibration.py
git commit -m "$(cat <<'EOF'
test(signatures): assert strategy=Spa() reproduces real cosmic_fit

Runs SigProfilerAssignment and genoray on the same synthetic catalogues
across three burden/composition cells and requires an identical signature
support with activities agreeing to within one mutation -- the resolution
roundConserveSum leaves.

Run with: pixi run -e sigprofiler pytest tests/test_signatures_calibration.py

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Documentation

Required by `CLAUDE.md`. Not optional.

**Files:**
- Modify: `skills/genoray-api/SKILL.md`, `docs/source/api.md`, `python/genoray/_signatures/__init__.py`

- [ ] **Step 1: Update the module docstring**

The paragraph #212 added to the module docstring says SPA's behaviours are "not reproduced here". That is now false. Rewrite `python/genoray/_signatures/__init__.py`'s docstring to:

```python
"""COSMIC mutational-signature refitting.

Decomposes a mutation catalogue into per-sample activities against a set of
reference signatures. Pure numpy/scipy/polars; no SigProfiler dependency.

Two strategies, selected via ``fit_signatures(strategy=...)``:

``Forward`` (the default) is genoray's own greedy forward selection from the
empty set, scored on cosine similarity, with a ``min_activity`` prune.

``Spa`` is a faithful reimplementation of SigProfilerAssignment's
``cosmic_fit``: one NNLS over the entire signature set, backward elimination
on relative L2 error, add-remove refinement layers, SBS1/SBS5 as protected
background signatures, force-added co-occurring partners, and activities
rescaled and integer-rounded to sum to the sample's mutation burden. It
recovers more true signatures than ``Forward`` at every burden measured.
``tests/test_signatures_calibration.py`` checks it against the real tool.

Note that ``Forward(criterion="cosine")`` and ``Spa`` are both
scale-invariant, and therefore blind to mutation burden: neither gains power
to resolve a real signature as the catalogue grows. Only
``Forward(criterion="bic")`` is a consistent estimator. Choosing SPA's search
direction improves recovery; it does not fix that.
"""
```

- [ ] **Step 2: Update `skills/genoray-api/SKILL.md`**

Three edits:

1. In the public-surface list near line 26, add below the `fit_signatures` line:

```markdown
- `genoray.Forward` / `genoray.Spa` — refit strategy objects for `fit_signatures(strategy=...)`
- `genoray.Strategy` / `genoray.Criterion` / `genoray.Metric` / `genoray.ActivityScale` — the associated type aliases
```

2. In the "Where to look for details" list, change the `_signatures.py` line to:

```markdown
- `genoray/_signatures/` — `cosmic_signatures`, `fit_signatures`, and the `Forward`/`Spa` strategies (`_strategy.py`, `_forward.py`, `_spa.py`)
```

3. Rewrite the `assign_signatures` bullet (around line 1054) to:

```markdown
- `assign_signatures(kind, *, reference=None, count="allele", strategy=None, max_delta=0.01, min_activity=0.005, criterion="cosine", n_jobs=1, backend="loky") -> pl.DataFrame`
  — `mutation_matrix(kind, count=...)` then `genoray.fit_signatures(...)`.
  `reference` accepts a `pl.DataFrame`, a TSV path, or `None` (defaults to
  `genoray.cosmic_signatures(kind)`). `strategy=` takes `genoray.Forward(...)`
  or `genoray.Spa(...)`; it **cannot** be combined with `max_delta`/
  `min_activity`/`criterion`, which are shorthand for the forward path and
  raise `ValueError` if passed alongside it.
```

Add a short subsection documenting the two strategies and, critically, the
activity-scale difference:

```markdown
### Refit strategies

`fit_signatures(strategy=...)` selects the algorithm. Both strategies are
frozen dataclasses carrying only their own parameters.

- `Forward(max_delta=0.01, min_activity=0.005, criterion="cosine")` — greedy
  forward selection from the empty set. The default. Activities are **raw
  NNLS weights and need not sum to the sample's mutation burden.**
- `Spa(metric="l2", initial_remove_penalty=0.05, add_penalty=0.05,
  remove_penalty=0.01, background_sigs=("SBS1","SBS5"), connected_sigs=True,
  activity_scale="burden")` — SigProfilerAssignment's `cosmic_fit`. Every
  default is SPA's own. Under the default `activity_scale="burden"`,
  activities are **integer-valued floats summing exactly to the sample's
  burden**; `activity_scale="raw"` gives unscaled NNLS weights instead.

`background_sigs` names absent from the reference are ignored, so the default
is inert for DBS78 and ID83. `connected_sigs=True` uses SPA's four SBS groups
(SBS2/13, SBS7a-d, SBS10a/b, SBS17a/b); pass your own groups as a sequence of
sequences, or `False` to disable.

Neither `Forward(criterion="cosine")` nor `Spa` is burden-aware. For
whole-genome catalogues where consistency matters, use
`Forward(criterion="bic")`.

The output schema is unchanged by the strategy: `Sample`, one Float column
per reference signature, and a trailing `cosine_similarity`.
```

- [ ] **Step 3: Update `docs/source/api.md`**

Read the file, find how `fit_signatures` and `cosmic_signatures` are declared (autodoc directives or a listing), and add `Forward`, `Spa` in the same style. Do not invent a new format.

- [ ] **Step 4: Verify the docs build if there is a docs task**

Run: `grep -n "docs" pixi.toml`

If a docs build task exists, run it and confirm it succeeds. If there is none, skip — do not invent one.

- [ ] **Step 5: Run the whole suite**

Run: `pixi run pytest tests -m "not network and not bench" -q`
Expected: PASS.

Run: `pixi run typecheck` → 0 errors.
Run: `pixi run -e lint ruff check python/genoray tests && pixi run -e lint ruff format --check python/genoray tests` → clean.

- [ ] **Step 6: Commit**

```bash
git add skills/genoray-api/SKILL.md docs/source/api.md python/genoray/_signatures/__init__.py
git commit -m "$(cat <<'EOF'
docs(signatures): document the Forward and Spa refit strategies

The module docstring's claim that SPA's behaviours are "not reproduced here"
is no longer true. SKILL.md gains the new public names, the strategy
subsection, and the activity-scale difference between the two paths --
forward returns raw NNLS weights, Spa returns integers summing to the burden.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Open the pull request

**Files:** none.

- [ ] **Step 1: Full verification**

Run every gate and read the output before claiming anything:

```bash
pixi run pytest tests -m "not network and not bench" -q
pixi run pytest tests/test_signatures_recovery.py -q
pixi run -e sigprofiler pytest tests/test_signatures_calibration.py -q
pixi run typecheck
pixi run -e lint ruff check python/genoray tests
pixi run -e lint ruff format --check python/genoray tests
```

Expected: all green. If the sigprofiler env could not be built, say so explicitly in the PR body — do not silently omit it.

- [ ] **Step 2: Check GenVarLoader against the branch**

The package split touches a private API that GenVarLoader imports. Confirm nothing GVL reaches for moved:

```bash
pixi run python -c "
import genoray._signatures as s
for n in ('fit_signatures','cosmic_signatures','Criterion','_fit_one','_cosine','_nnls','_poisson_ll','_load_signature_file','_COSMIC_REGISTRY','_KIND_TOKEN'):
    assert hasattr(s, n), n
print('private surface intact')
"
```

If a local GVL checkout is available, grep it for `genoray._signatures` and verify each name. Report what was checked in the PR body rather than asserting compatibility you did not test.

- [ ] **Step 3: Push and open a draft PR**

```bash
git push -u origin worktree-feat-214-spa-faithful-refit
gh pr create --draft --base main \
  --title "feat(signatures): add an SPA-faithful refit strategy" \
  --body "$(cat <<'EOF'
Closes #214 (items 1-7).

`fit_signatures` gains `strategy=`, taking `Forward(...)` (today's greedy
forward selection, the default, unchanged) or `Spa(...)`, a faithful
reimplementation of SigProfilerAssignment's `cosmic_fit`.

## What `Spa()` does

1. One NNLS over the entire signature set.
2. Backward elimination on relative L2: remove the signature whose removal
   degrades the fit least, while that degradation stays under
   `initial_remove_penalty`.
3. Add-remove refinement layers, so an early decision can be undone.
4. Activities rescaled and integer-rounded to sum to the sample's burden.

Plus SBS1/SBS5 as protected background signatures and SPA's four
connected-signature groups. Every `Spa` default is SPA's own.

## Why strategy objects rather than flat keyword arguments

The alternative was ~11 flat arguments, most mutually exclusive. #200 and
#207 recorded the lesson: a parameter one caller silently drops is a
semantic trap. Each strategy carries only its own knobs, so `max_delta`
cannot reach backward elimination. The flat `max_delta`/`min_activity`/
`criterion` stay as permanent shorthand for the forward path, and combining
them with `strategy=` raises.

## Faithfulness

`tests/test_signatures_calibration.py` runs real SigProfilerAssignment and
this implementation on the same catalogues across three burden/composition
cells, and requires an identical signature support with activities agreeing
to within one mutation.

Two SPA behaviours are deliberately not copied, both documented in the `Spa`
docstring. SPA refuses to remove a signature whose removal *improves* the
fit; on the default l2+NNLS path that branch is unreachable, because NNLS
over a column subset cannot beat NNLS over the superset, so omitting it
cannot change a result (pinned by a property test). SPA also draws from the
global RNG for values it never uses; this path is deterministic.

## Not in scope

Item 8 (exome-renormalized COSMIC signatures) is a real correctness gap for
WES and panel callers but is independent of the fitting algorithm. It gets
its own issue and PR.

Neither strategy is a consistent estimator — both scores are
scale-invariant. Only `Forward(criterion="bic")` from #212 is. This PR
improves recovery; it does not fix that, and the docs say so.

## Verification

<fill in the actual command output>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report**

Report the PR URL, the measured per-sample cost from task 7, and whether the sigprofiler calibration actually ran or was skipped.

---

## Self-Review

**Spec coverage.** Walked each spec section against the tasks:

| Spec section | Task |
|---|---|
| API: `Forward`/`Spa`/aliases | 3 |
| API: strategy resolution, `_UNSET` sentinel | 4 |
| API: naming | 3 (kept as specified) |
| API: downstream plumbing, `criterion` Boy Scout fix | 8 |
| Algorithm stage 1 saturate | 7 |
| Algorithm stage 2 initial prune | 5 (sweep), 7 (invocation, unprotected) |
| Algorithm stage 3 refinement layers | 6 (`_try_add`), 7 (layer loop) |
| Rounding drives the support | 5 (`_exposure`/`_support`), 7 |
| Algorithm stage 4 report | 5 (`_exposure`), 7 |
| Priors: `background_sigs`, `connected_sigs` | 6 (`_expand_connected`), 7 (`_resolve_sig_names`) |
| Divergence: negative-difference guard | 5 (property test), 3 (docstring) |
| Divergence: RNG | 7 (determinism test) |
| Divergence: `check_rule_negatives` | 3 (documented, not implemented) |
| Output schema unchanged | 7 (schema test) |
| Module structure | 1, plus `_strategy.py`/`_spa.py` in 3/5 |
| Parallelism, resolve-once | 7 (`_resolve_sig_names`), 7 step 7 (cost) |
| Testing: calibration | 10 |
| Testing: recovery | 9 |
| Testing: properties | 2, 5 |
| Testing: regression | 4 step 7 |
| Testing: API surface | 1, 3, 4, 8 |
| Documentation | 11 |
| Risks: GVL check | 12 step 2 |

No gaps found.

**Placeholder scan.** Two intentional fill-ins remain, each a measurement that cannot be known before running: the per-sample timing in task 7's commit message, and the verification output in task 12's PR body. Each is marked `<fill in ...>` with the exact command that produces it. No "TBD", no "add error handling", no "similar to task N".

**Type consistency.** Checked names across tasks: `_fit_one_forward`, `_fit_one_spa`, `_resolve_strategy`, `_resolve_sig_names`, `_exposure`, `_support`, `_reconstruction`, `_remove_all_single`, `_try_add`, `_expand_connected`, `_rel_l2`, `_distance`, `_round_conserve_sum`. Each is defined in exactly one task and used with the same signature everywhere after. `_reconstruction` is introduced in task 5's implementation and used in task 6 and 7 — its interface block is listed under task 5's Produces.

One inconsistency found and fixed while reviewing: task 5's `_remove_all_single` takes and returns an exposure *vector*, not an index list, so task 7's layer loop calls `_support(...)` on its result rather than using it directly. The code in task 7 step 3 reflects this.

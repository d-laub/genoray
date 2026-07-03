# Parallelize `fit_signatures` over samples

**Date:** 2026-06-26
**Status:** Design approved, pending spec review

## Problem

`fit_signatures` (`genoray/_signatures.py`) is the one un-parallelized hot path in
the mutation-catalogue subsystem. Classification (`_mutcat.py`) is already fully
vectorized/numba-parallel, but signature refitting loops over samples serially:

```python
for j in range(len(sample_cols)):
    h, cos = _fit_one(W, M[:, j], max_delta=max_delta, min_activity=min_activity)
    activities[j] = h
    cosines[j] = cos
```

`_fit_one` runs sparse forward selection: for each of up to N reference signatures
it calls `scipy.optimize.nnls`, re-running for every candidate at every step —
roughly **O(N²) `nnls` calls per sample** (~6400 for the ~80-signature COSMIC SBS96
set). For cohort-scale workloads (100s–1000s of samples) this serial loop dominates
total runtime.

`nnls` is scipy/Fortran, so numba cannot accelerate the inner work. But the samples
are embarrassingly parallel, and the per-sample function is already cleanly extracted
into `_fit_one(W, m, ...)` — a pure function of one sample's count vector and the
shared, read-only signature matrix `W`.

## Goal

Parallelize the per-sample loop with `joblib`, defaulting to all cores, with no
change to results or to the `_fit_one` algorithm.

## Decisions (from brainstorming)

- **Axis:** per-sample parallelism (not within-sample). Workload is many samples.
- **Backend:** processes (`loky`). The forward-selection orchestration is ~O(N²)
  Python loop iterations per sample holding the GIL; a thread pool would serialize
  on that contention. `W` is tiny (~96×80 float64 ≈ 60 KB) so process broadcast cost
  is negligible, and joblib auto-batches the short tasks. nnls problems are small
  (96×N) so inner-BLAS oversubscription inside workers is a non-issue.
- **Default:** parallel by default (`n_jobs=-1`). Accepted tradeoff: even small calls
  spawn a worker pool (first-call import cost in workers). Caller passes `n_jobs=1`
  to disable.
- **Surface:** expose **both** `n_jobs` and `backend` on `fit_signatures` *and*
  `assign_signatures` (symmetry; self-documenting process choice; immune to a future
  joblib default change).

## Changes

### 1. `genoray/_signatures.py::fit_signatures`

Add two keyword-only parameters:

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

Replace the serial loop with a joblib dispatch:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=n_jobs, backend=backend)(
    delayed(_fit_one)(W, M[:, j], max_delta=max_delta, min_activity=min_activity)
    for j in range(len(sample_cols))
)
for j, (h, cos) in enumerate(results):
    activities[j] = h
    cosines[j] = cos
```

`Parallel` returns results in **input order** regardless of completion order, so the
existing `activities[j]` / `cosines[j]` alignment and overall determinism are
preserved — no re-sorting required.

`_fit_one`, the input validation, the `W` normalization, and the `M` construction are
all unchanged.

### 2. `genoray/_svar.py::SparseVar.assign_signatures`

Thread matching parameters through to `fit_signatures`:

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
    ...
    return fit_signatures(
        catalogue, ref,
        max_delta=max_delta, min_activity=min_activity,
        n_jobs=n_jobs, backend=backend,
    )
```

## Error handling

No new failure modes. `_fit_one` is unchanged and self-contained; joblib propagates
any worker exception to the caller. Input validation (`MutationType` presence/alignment
checks) stays in the parent process, before dispatch.

## Testing

- **Equivalence:** assert `fit_signatures(..., n_jobs=-1)` is element-wise **exactly**
  equal to `n_jobs=1` on a fixture catalogue/reference. The fit is deterministic, so
  exact equality (not approximate) is the right assertion.
- **Serial path coverage:** keep an explicit `n_jobs=1` test so the loop body stays
  covered in CI without spawning processes.
- Existing `fit_signatures` correctness tests continue to pass unchanged (default is
  now parallel, results identical).

## Docs

`fit_signatures` and `assign_signatures` are public API. Per `CLAUDE.md`'s public-API
rule, update `skills/genoray-api/SKILL.md` to document the new `n_jobs` and `backend`
kwargs on both.

## Out of scope

- No algorithmic change to forward selection.
- No within-sample parallelism (single-sample / huge-reference case is not the target
  workload).
- No progress bar (`joblib-progress` is available but not requested here).

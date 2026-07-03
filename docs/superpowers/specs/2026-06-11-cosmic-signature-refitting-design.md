# COSMIC Signature Refitting for genoray

**Date:** 2026-06-11
**Status:** Approved (pending spec review)

## Goal

Add **mutational-signature refitting** on top of the existing mutation-catalogue
feature. Given a sample's SBS-96 / DBS-78 / ID-83 catalogue (as produced by
`SparseVar.mutation_matrix`), decompose it into per-sample **activities**
(exposures) against a set of COSMIC reference signatures.

This ports the core of
[SigProfilerAssignment](https://github.com/SigProfilerSuite/SigProfilerAssignment)
— the sparse forward-selection refit — into genoray as a lean numpy/scipy
implementation, rather than depending on the heavyweight SigProfiler package
tree. This mirrors how the mutation-catalogue feature ported the classification
core instead of depending on SigProfilerMatrixGenerator.

## Scope

**In scope (v1):**

- A pure refit function `fit_signatures(catalogue, reference, ...)` operating on
  any `mutation_matrix`-shaped catalogue DataFrame and a row-aligned reference
  signature DataFrame.
- A `pooch`-backed `cosmic_signatures(kind, ...)` loader that fetches and caches
  the official COSMIC reference signature matrices on demand.
- A thin `SparseVar.assign_signatures(kind, ...)` convenience method that wires
  `mutation_matrix` → `fit_signatures`.
- Faithful port of SigProfilerAssignment's forward-stepwise selection refit
  (NNLS + cosine-similarity-guided add + min-activity prune).
- Per-sample reconstruction quality (`cosine_similarity`) returned alongside
  activities.

**Out of scope (v1), documented as such:**

- De novo signature extraction (NMF over a cohort — that is SigProfilerExtractor,
  not Assignment).
- Genome/exome opportunity normalization of reference signatures.
- Bootstrapped confidence intervals / stability resampling.
- Plotting.

## Background / existing patterns

- `SparseVar.mutation_matrix(kind, *, count="allele"|"sample")`
  (`genoray/_svar.py:1550`) returns a Polars DataFrame with a `MutationType`
  string column followed by one `Int64` column per sample, rows in fixed COSMIC
  codebook order (96 / 78 / 83 rows). This is the input catalogue for refitting.
- The mutation-catalogue feature established the precedent of **porting the
  core** (numpy/numba classifiers in `genoray/_mutcat.py`) and vendoring only a
  thin `pysam`-backed `Reference` reader (`genoray/_reference.py`), rather than
  depending on SigProfiler packages.
- A SigProfiler **calibration env + cross-check test** already exists for the
  mutcat work (commit `a1dfa69`); the refit cross-check follows the same model.
- Public-API changes must update `skills/genoray-api/SKILL.md` (per CLAUDE.md).

## Architecture

### New module: `genoray/_signatures.py`

**`fit_signatures(catalogue, reference, *, max_delta=0.01, min_activity=0.005) -> pl.DataFrame`**

- `catalogue` — a `mutation_matrix`-shaped `pl.DataFrame`: a `MutationType`
  column + one numeric column per sample.
- `reference` — a `pl.DataFrame`: a `MutationType` column + one column per
  reference signature (e.g. `SBS1`, `SBS5`, …).
- Aligns rows by **joining on `MutationType`** (not positional), raising a clear
  error if the catalogue contains a type absent from the reference, or if row
  sets otherwise mismatch.
- Runs the per-sample sparse refit (see Algorithm) against the reference
  signature matrix `W` (shape: n_types × n_signatures).
- Returns an **activities** DataFrame: one row per sample, columns =
  `Sample` (string) + one column per reference signature (Float, counts scaled
  to the sample's total mutation burden) + `cosine_similarity` (Float,
  reconstruction quality of the final fit). Signatures not selected for a sample
  are `0.0`.
- `max_delta` — minimum cosine-similarity improvement required to keep adding a
  signature (forward-selection stop criterion).
- `min_activity` — minimum fractional contribution; signatures below this are
  pruned and the fit re-solved on survivors.

**`cosmic_signatures(kind, *, version="3.4", genome="GRCh38") -> pl.DataFrame`**

- `kind` — `"SBS96" | "DBS78" | "ID83"`.
- Fetches the official COSMIC reference signature TSV via `pooch` (registry with
  known hashes), caches to pooch's default cache dir, parses to a DataFrame in
  the same `MutationType`-rows orientation expected by `fit_signatures`.
- For ID83, `genome` is ignored (indel signatures are build-independent in the
  COSMIC release); for SBS96/DBS78 the `genome` selects the GRCh37/GRCh38
  variant. Validates that the returned `MutationType` rows match genoray's
  codebook order for that `kind`.

### Modified: `genoray/_svar.py`

**`SparseVar.assign_signatures(kind, *, reference=None, count="allele", **fit_kwargs) -> pl.DataFrame`**

- Calls `self.mutation_matrix(kind, count=count)` to get the catalogue.
- If `reference is None`, defaults to `cosmic_signatures(kind)`; otherwise
  accepts a `pl.DataFrame` (or a path to a signature TSV).
- Forwards `**fit_kwargs` (`max_delta`, `min_activity`) to `fit_signatures` and
  returns its result.

### Modified: `genoray/__init__.py`

- Export `fit_signatures` and `cosmic_signatures`.

### Modified: `pyproject.toml`

- Add `pooch` (fetch/cache) and `scipy` (NNLS) dependencies.

### Modified: `skills/genoray-api/SKILL.md`

- Document `fit_signatures`, `cosmic_signatures`, and
  `SparseVar.assign_signatures`, including the output orientation and threshold
  kwargs.

## Algorithm (ported from SigProfilerAssignment)

For each sample column `m` (observed catalogue count vector) against the
reference matrix `W` (types × signatures):

1. Start with an empty active signature set.
2. **Forward selection:** for each candidate signature not yet active, tentatively
   add it, solve NNLS (`scipy.optimize.nnls`) for the active set, and compute the
   cosine similarity between `m` and the reconstruction `W_active · h`. Add the
   candidate that yields the highest cosine similarity.
3. Repeat step 2 until the best achievable cosine-similarity improvement is below
   `max_delta`.
4. **Prune:** drop any active signature whose fractional contribution is below
   `min_activity`; re-solve NNLS on the survivors.
5. Report activities as counts (the NNLS solution `h` scaled so the
   reconstruction total matches the sample's observed total) and the final
   cosine similarity.

Defaults (`max_delta=0.01`, `min_activity=0.005`) follow SigProfilerAssignment's
behavior and are exposed as kwargs.

## Data flow

```
SparseVar (mutcat field)
   └─ mutation_matrix(kind)         → catalogue DataFrame (MutationType × samples)
                                          │
cosmic_signatures(kind) via pooch   → reference DataFrame (MutationType × signatures)
                                          │
   fit_signatures(catalogue, reference)   ▼
                                       activities DataFrame
                                       (Sample × signatures + cosine_similarity)

SparseVar.assign_signatures(kind)  = mutation_matrix(kind) ∘ fit_signatures(…, cosmic_signatures(kind))
```

## Error handling

- **Row mismatch:** `fit_signatures` raises `ValueError` if the catalogue and
  reference cannot be aligned on `MutationType` (missing/extra types).
- **Empty / all-zero sample:** a sample with zero mutations yields all-zero
  activities and `cosine_similarity = 0.0` (cosine is undefined for a zero
  observed vector; we define it as `0.0` rather than NaN), without raising.
- **Unknown `kind`:** `cosmic_signatures` / `assign_signatures` raise
  `ValueError` for kinds outside `{SBS96, DBS78, ID83}`.
- **pooch fetch failure / hash mismatch:** surfaced as pooch's own error; the
  network-dependent path is marked with the `network` pytest marker.

## Output orientation

Activities use **rows = sample** (`Sample` column + signature columns +
`cosine_similarity`), matching SigProfilerAssignment's `Activities.txt`. This is
the transpose of `mutation_matrix`'s rows=type orientation, chosen because
"per sample, how much of each signature" is the natural reading for exposures.

## Testing

- **Synthetic recovery:** construct `m = W · h_known` for a sparse known `h_known`;
  assert `fit_signatures` recovers the correct active signatures and a cosine
  similarity ≈ 1.0.
- **Sparsity behavior:** verify forward-selection does not return a dense
  solution (most signatures `0.0`) and that `min_activity` pruning works.
- **Row alignment:** mismatched / reordered reference rows are realigned by join;
  missing types raise `ValueError`.
- **SigProfilerAssignment cross-check:** in the existing SigProfiler calibration
  env, run SPA on a small fixture catalogue and assert genoray's activities match
  within tolerance (mirrors the mutcat calibration test).
- **`cosmic_signatures` loader:** validate parsed row order against the codebook;
  network fetch marked `network`, hash/registry verified.
- **`SparseVar.assign_signatures`:** end-to-end on an annotated `.svar` fixture.

## Public API summary (for SKILL.md)

```python
import genoray

ref = genoray.cosmic_signatures("SBS96")            # pooch-fetched, cached
cat = svar.mutation_matrix("SBS96")                 # existing
act = genoray.fit_signatures(cat, ref)              # activities + cosine_similarity

# convenience
act = svar.assign_signatures("SBS96")               # reference defaults to COSMIC
act = svar.assign_signatures("SBS96", reference=ref, min_activity=0.01)
```

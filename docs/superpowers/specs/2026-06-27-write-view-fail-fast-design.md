# `write_view` + `genoray view` fail-fast

**Date:** 2026-06-27
**Type:** performance / robustness patch

---

## Problem

`SparseVar.write_view` (and the `genoray view` CLI that wraps it) does some
cheap, certain-to-fail validation *after* expensive work, and performs the
destructive output-directory step *before* it is certain the write will
succeed. On large cohorts this means a trivial mistake — an already-existing
output, a typo'd reference path — only surfaces after region resolution and/or
a full genotype write, and in some cases after an existing output directory has
already been deleted.

Concretely, in the current `write_view` (`genoray/_svar.py:1888`+):

1. **Output existence / `overwrite`** is checked at `:1930`, *after* region
   resolution (`:1919`, which collects numeric index columns and runs
   `var_ranges`).
2. **`reference`** is only loaded/validated at the very end (`:2088`), *after*
   the entire output has been written. A bad FASTA path raises
   `FileNotFoundError` only once all the heavy numba passes and I/O are done.
3. **`rmtree` + `mkdir`** (`:1935`-`:1936`) runs before the MAC pre-pass. If
   every selected variant has MAC=0 in the chosen subset, the run raises
   `ValueError` (`:1964`) *after* an existing output directory was already
   deleted.
4. **`output == source`** is not guarded: `overwrite=True` with `output` equal
   to `self.path` would `rmtree` the source dataset and then fail to read it.

Missing samples already fails fast — `_normalize_samples` (`:184`) raises in
the cheap normalization band (`:1906`), before any heavy work — so that path
needs no change. It is the example that motivated the audit, not a current bug.

In the CLI (`genoray/_cli/__main__.py`, `view` at `:179`), `source`,
`--samples-file`, and `--regions-file` are not validated for existence up
front; a missing path fails mid-run with a raw Python error rather than a clean
parse-time message.

## Goal

Every cheap check, and every potentially-destructive decision, happens before
expensive work. A bad reference path, an existing output, a self-overwrite, or
an all-MAC=0 selection aborts **without** destroying an existing output
directory and **without** doing the full write. CLI path mistakes fail at
arg-parse time.

## Non-goals

- No change to output bytes, schema, dtypes, or coordinate/missing-value
  conventions.
- No validation that requested contigs in `regions` exist, or that the
  reference covers the subset's contigs (deferred; `annotate_mutations` already
  handles contig scope).
- CLI changes are scoped to the `view` command only; `index` / `write` are left
  alone.

---

## Architecture

### Part 1 — `write_view` validation reorder (`genoray/_svar.py`)

Reorganize the body of `write_view` into three bands. All raises happen before
any I/O; the destructive `rmtree` + `mkdir` moves down to just before the first
write pass.

**Band A — cheap raises (nothing touched):**

1. mutcat-without-reference → `ValueError` *(already present)*
2. **output exists && not `overwrite` → `FileExistsError`** — moved up from
   after region resolution. Only the *raise* moves; no `rmtree` here.
3. **`output` resolves to the same path as `self.path` → `ValueError`** — new
   guard preventing self-destruction under `overwrite=True`. Compare
   `Path(output).resolve() == self.path.resolve()`.
4. normalize regions / samples / fields; strip `mutcat` *(already present)*
5. empty-samples → `ValueError` *(already present)*
6. **if `reference` is given, construct the `Reference` up front** —
   `ref_obj = reference if isinstance(reference, Reference) else
   Reference.from_path(reference)`. This validates the FASTA exists and builds
   the `.fai` (the same work `annotate_mutations` does internally). The built
   instance is reused for the final `annotate_mutations(ref_obj, ...)` call, so
   the FASTA is opened/indexed once instead of twice.

**Band B — heavier work, still no output touched:**

7. resolve `kept_var_idxs`; "no variants" → `ValueError`
8. setup (`n_out`, `ploidy`, threads, `src_sample_idxs`) + MAC pre-pass;
   "all MAC=0" → `ValueError`. The MAC=0 partial-drop `warnings.warn` stays
   here (warning before dir creation is fine).

**Band C — commit:**

9. **`rmtree` existing + `mkdir`** (moved here, after band B) → write
   `offsets.npy` → `variant_idxs.npy` → fields → streaming index → metadata →
   `annotate_mutations(ref_obj, write_back=True)` using the band-A instance.

`Reference` is already imported in `_svar.py` (used by `annotate_mutations` at
`:1592`), so no new import is needed.

### Part 2 — CLI parse-time validation (`genoray/_cli/__main__.py`, `view`)

Add cyclopts path validation so bad paths fail at arg-parse, before the body
runs:

- `source` → must be an existing **directory** (`.svar` is a directory).
- `regions_file` → must be an existing **file** (when supplied).
- `samples_file` → must be an existing **file** (when supplied).

Implemented by stacking the existing `Parameter(name=[...])` with cyclopts'
path validation. Use `cyclopts.validators.Path(exists=True, dir_okay=..., 
file_okay=...)` (or the prebuilt `cyclopts.types.ExistingDirectory` /
`ExistingFile`); the exact symbol and constructor kwargs are confirmed against
the installed cyclopts during implementation. The `None` defaults remain valid
— the validator only runs when a value is supplied. The existing no-op and
mutex guards in the body are unchanged.

---

## Testing (TDD)

New tests in `tests/test_svar_write_view.py`:

- **bad reference fails before write**: a non-existent FASTA path with valid
  regions/samples → `FileNotFoundError`, and `not output.exists()`.
- **output-exists checked before region resolution**: monkeypatch
  `_resolve_kept_var_idxs` to raise; an existing output with `overwrite=False`
  → `FileExistsError` (proving resolution was never reached).
- **self-overwrite guard**: `output == source` with `overwrite=True` →
  `ValueError`, and the source dataset is intact afterward.
- **all-MAC=0 does not destroy existing output**: a pre-populated output dir
  with `overwrite=True`, selecting only variants that are MAC=0 in the chosen
  subset → `ValueError`, and the original output contents are intact.

CLI tests (alongside the existing CLI tests): `genoray view` with a
non-existent `source`, `--samples-file`, or `--regions-file` exits non-zero at
parse time.

## Docs / SKILL.md

These are timing and safety changes, not signature / shape / dtype / name
changes, so no `skills/genoray-api/SKILL.md` update is expected. The
implementation verifies this against that file and, if it enumerates
`write_view`'s error conditions, adds the new self-overwrite `ValueError`.
`write_view`'s docstring gains a note that `reference` is validated up front.

## Files

- **Modify** `genoray/_svar.py` — reorder `write_view` into the three bands;
  add the self-overwrite guard and the up-front `Reference` construction.
- **Modify** `genoray/_cli/__main__.py` — add cyclopts path validators to
  `view`'s `source`, `regions_file`, `samples_file`.
- **Modify** `tests/test_svar_write_view.py` — add the four fail-fast tests.
- **Modify** the CLI test module — add the three parse-time path tests.
- **Verify** `skills/genoray-api/SKILL.md` — likely no change.

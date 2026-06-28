# write_view + view fail-fast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `SparseVar.write_view` and the `genoray view` CLI do all cheap, certain-to-fail validation before any expensive work, and never delete an existing output directory on a run that is going to fail.

**Architecture:** Reorder the body of `write_view` into three bands — (A) cheap raises that touch nothing, (B) heavier checks (region resolution, MAC pre-pass) that still touch nothing, (C) the destructive `rmtree`+`mkdir` and the write passes. Add a self-overwrite guard and build the `Reference` up front (reusing it at the end). In the CLI, attach cyclopts path validators so bad paths fail at arg-parse time.

**Tech Stack:** Python, polars, numpy, numba, cyclopts, pytest, pixi.

**Spec:** `docs/superpowers/specs/2026-06-27-write-view-fail-fast-design.md`

## Global Constraints

- All commits follow Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`). (from `CLAUDE.md`)
- Run tests via pixi: `pixi run pytest <path>`. (from `CLAUDE.md`)
- No change to output bytes, schema, dtypes, or coordinate/missing-value conventions (0-based half-open `[start, end)`; missing genotypes `-1`, dosages `np.nan`).
- If any public name/shape/behavior reachable from `import genoray` (without underscores) changes, update `skills/genoray-api/SKILL.md` in the same PR. (from `CLAUDE.md`)
- Test data: `tests/data/biallelic.vcf.svar` (fixtures `svar`, `svar_wv` build it; `ddir = Path(__file__).parent / "data"`). `Reference` is already imported in `genoray/_svar.py` (`:36`). `SparseVar.path` is a `pathlib.Path` (`:583`).

---

## File Structure

- **Modify** `genoray/_svar.py` — reorder `write_view` (`:1888`-`:2090`) into the three bands; add the self-overwrite guard; build `Reference` up front and reuse it for the final `annotate_mutations`.
- **Modify** `genoray/_cli/__main__.py` — add cyclopts path validators to `view`'s `source`, `regions_file`, `samples_file` (`:179`-`:196`).
- **Modify** `tests/test_svar_write_view.py` — add fail-fast tests (reuses `svar`, `svar_wv`, `ddir`, `_index_raises` pattern).
- **Modify** `tests/cli/test_view_cli.py` — add parse-time path tests (reuses `_run`, `tiny_svar`).

---

## Task 1: Reorder `write_view` into fail-fast bands

Move the output-exists raise above region resolution, build `Reference` up front, and move the destructive `rmtree`+`mkdir` below the MAC pre-pass so a doomed run never deletes an existing output.

**Files:**
- Modify: `genoray/_svar.py` — `write_view` (`:1888`-`:2090`)
- Test: `tests/test_svar_write_view.py`

**Interfaces:**
- Consumes: `self._covers_all_variants`, `_resolve_kept_var_idxs` (module-level), `self.n_variants`, `self.genos`, `Reference.from_path`, `self.path`.
- Produces: `write_view` signature unchanged (`regions, samples, output, fields=None, reference=None, merge_overlapping=False, regions_overlap="pos", overwrite=False, threads=None) -> None`). New behavior: a missing `reference`, an existing `output` without `overwrite`, or an all-MAC=0 selection raises **before** any output directory is created or deleted.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar_write_view.py`:

```python
# ---------------------------------------------------------------------------
# Fail-fast: write_view band reorder
# ---------------------------------------------------------------------------


def _resolve_raises(*args, **kwargs):
    raise AssertionError("region resolution was reached before output check")


def test_write_view_bad_reference_fails_before_write(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    with pytest.raises(FileNotFoundError):
        svar.write_view(
            regions=(contig, 0, 1_000_000),
            samples=samples,
            output=out,
            reference=str(tmp_path / "does_not_exist.fa"),
        )
    # Reference is validated up front, so no output dir is created.
    assert not out.exists()


def test_write_view_output_exists_checked_before_resolution(
    monkeypatch, tmp_path: Path, svar: SparseVar
):
    out = tmp_path / "view.svar"
    out.mkdir()
    # If region resolution runs before the output-exists check, this raises
    # AssertionError instead of FileExistsError.
    monkeypatch.setattr("genoray._svar._resolve_kept_var_idxs", _resolve_raises)
    contig = svar.contigs[0]
    with pytest.raises(FileExistsError):
        svar.write_view(
            # narrow region so _covers_all_variants is False and the
            # _resolve_kept_var_idxs path would be taken
            regions=(contig, 0, 10_000),
            samples=svar.available_samples[:2],
            output=out,
            overwrite=False,
        )


def test_write_view_all_mac0_preserves_existing_output(
    tmp_path: Path, svar_wv: SparseVar
):
    out = tmp_path / "existing.svar"
    out.mkdir()
    marker = out / "DO_NOT_DELETE.txt"
    marker.write_text("keep me")
    samples_all = list(svar_wv.available_samples)
    # Same all-MAC=0 scenario as test_write_view_raises_when_all_variants_drop.
    with pytest.raises(ValueError, match="MAC=0"):
        svar_wv.write_view(
            regions=(svar_wv.contigs[0], 81264, 81265),
            samples=[samples_all[1]],
            output=out,
            overwrite=True,
        )
    # The doomed run must not have deleted the pre-existing output.
    assert marker.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_svar_write_view.py -k "bad_reference_fails_before_write or output_exists_checked_before_resolution or all_mac0_preserves_existing_output" -v`
Expected: all three FAIL — `bad_reference` leaves `out` created (reference validated last), `output_exists` raises `AssertionError` (resolution runs first), `all_mac0` finds `marker` deleted (rmtree runs before the MAC check).

- [ ] **Step 3: Reorder `write_view`**

In `genoray/_svar.py`, replace the body from the mutcat early-validation block down to the `mkdir` (`:1893`-`:1936`) so the order becomes the bands below. Keep every existing line's logic; only the order and the `Reference` construction change.

Replace this current sequence:

```python
        # --- Early validation: mutcat cannot be positionally copied ---
        if fields is not None and "mutcat" in fields:
            if reference is None:
                raise ValueError(
                    "'mutcat' cannot be copied through write_view because its codes "
                    "are dataset-specific (DBS adjacency is only valid for the full "
                    "variant set; subsetting may leave stale codes). "
                    "Pass reference= to recompute mutcat on the subset, or call "
                    "annotate_mutations() on the output view yourself."
                )

        # --- 1. Normalize inputs ---
        regions_df = _normalize_regions(regions, self._c_norm)
        caller_samples = _normalize_samples(samples, self.available_samples)
        fields_to_write = _validate_fields(fields, self.available_fields)
        # Always exclude the derived "mutcat" field from positional copy:
        # its codes encode cross-variant DBS adjacency that is only valid for
        # the full variant set.  Subsetting can drop a DBS 3' partner, leaving
        # an orphaned 5' code that mutation_matrix would miscount.
        # Use reference= to recompute mutcat on the output view instead.
        fields_to_write = [f for f in fields_to_write if f != "mutcat"]

        if not caller_samples:
            raise ValueError("write_view requires at least one sample")

        # --- 2. Resolve kept variant indices ---
        if self._covers_all_variants(regions_df, regions_overlap):
            # Fast path: every variant is selected; skip POS/ILEN materialization.
            kept_var_idxs = np.arange(self.n_variants, dtype=V_IDX_TYPE)
        else:
            kept_var_idxs = _resolve_kept_var_idxs(
                self, regions_df, regions_overlap, merge_overlapping
            )
        if len(kept_var_idxs) == 0:
            raise ValueError("no variants selected by `regions`")

        # --- 3. Output directory (after all validation, so no partial dir on error) ---
        if output.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output path {output} already exists. Use overwrite=True to overwrite."
                )
            shutil.rmtree(output)
        output.mkdir(parents=True)
```

with:

```python
        # --- Band A: cheap raises (fail fast; nothing on disk is touched) ---

        # mutcat cannot be positionally copied through a view.
        if fields is not None and "mutcat" in fields:
            if reference is None:
                raise ValueError(
                    "'mutcat' cannot be copied through write_view because its codes "
                    "are dataset-specific (DBS adjacency is only valid for the full "
                    "variant set; subsetting may leave stale codes). "
                    "Pass reference= to recompute mutcat on the subset, or call "
                    "annotate_mutations() on the output view yourself."
                )

        # Output existence: raise (but do NOT delete) before any heavy work.
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {output} already exists. Use overwrite=True to overwrite."
            )

        # Normalize inputs (cheap; missing samples/fields raise here).
        regions_df = _normalize_regions(regions, self._c_norm)
        caller_samples = _normalize_samples(samples, self.available_samples)
        fields_to_write = _validate_fields(fields, self.available_fields)
        # Always exclude the derived "mutcat" field from positional copy:
        # its codes encode cross-variant DBS adjacency that is only valid for
        # the full variant set.  Subsetting can drop a DBS 3' partner, leaving
        # an orphaned 5' code that mutation_matrix would miscount.
        # Use reference= to recompute mutcat on the output view instead.
        fields_to_write = [f for f in fields_to_write if f != "mutcat"]

        if not caller_samples:
            raise ValueError("write_view requires at least one sample")

        # Validate the reference up front (existence + .fai build) and reuse the
        # built instance for the final annotate_mutations, so a bad FASTA path
        # fails now instead of after the whole output is written.
        ref_obj: "Reference | None" = None
        if reference is not None:
            ref_obj = (
                reference
                if isinstance(reference, Reference)
                else Reference.from_path(reference)
            )

        # --- Band B: heavier checks (still nothing on disk is touched) ---

        # Resolve kept variant indices.
        if self._covers_all_variants(regions_df, regions_overlap):
            # Fast path: every variant is selected; skip POS/ILEN materialization.
            kept_var_idxs = np.arange(self.n_variants, dtype=V_IDX_TYPE)
        else:
            kept_var_idxs = _resolve_kept_var_idxs(
                self, regions_df, regions_overlap, merge_overlapping
            )
        if len(kept_var_idxs) == 0:
            raise ValueError("no variants selected by `regions`")
```

Next, move directory creation below the MAC pre-pass. Having removed the "--- 3. Output directory ---" block above, the remaining flow is "--- 4. Setup ---" then "--- 4.5. Pre-pass ... ---". After the MAC pre-pass block ending with the all-MAC=0 raise (`:1964`-`:1968`), insert the destructive directory step. Locate this block:

```python
        if len(kept_var_idxs) == 0:
            raise ValueError(
                "all variants in the selected regions have MAC=0 in the "
                "chosen sample subset; nothing to write"
            )
```

and immediately after it (before `# --- 5. Pass 1: count kept entries per output slot ---`) add:

```python
        # --- Band C: commit. All validation passed; now (re)create the output. ---
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
```

Finally, update the annotation step at the end (`:2087`-`:2090`) to reuse `ref_obj`:

```python
        # --- 11. Optionally recompute mutcat on the output view ---
        if ref_obj is not None:
            out_svar = SparseVar(output)
            out_svar.annotate_mutations(ref_obj, write_back=True)
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `pixi run pytest tests/test_svar_write_view.py -k "bad_reference_fails_before_write or output_exists_checked_before_resolution or all_mac0_preserves_existing_output" -v`
Expected: all three PASS.

- [ ] **Step 5: Run the full write_view + CLI suites for regressions**

Run: `pixi run pytest tests/test_svar_write_view.py tests/cli/test_view_cli.py -q`
Expected: all pass (no behavior regression; existing `test_write_view_raises_when_all_variants_drop`, overwrite, and roundtrip tests still green).

- [ ] **Step 6: Commit**

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "$(cat <<'EOF'
perf: fail fast in write_view; never delete output on a doomed run

Move output-exists check above region resolution, validate reference up
front (reusing the built instance), and move rmtree/mkdir below the MAC
pre-pass so a failing run never deletes an existing output directory.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Self-overwrite guard in `write_view`

Prevent `output` resolving to the source dataset path under `overwrite=True`, which today would `rmtree` the source and corrupt the in-progress read.

**Files:**
- Modify: `genoray/_svar.py` — `write_view` Band A (the cheap-raises block from Task 1)
- Test: `tests/test_svar_write_view.py`

**Interfaces:**
- Consumes: `self.path` (`pathlib.Path`), `output` (`pathlib.Path`).
- Produces: `write_view` raises `ValueError` when `Path(output).resolve() == self.path.resolve()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar_write_view.py`:

```python
def test_write_view_self_overwrite_guard(tmp_path: Path):
    import shutil

    # Copy the fixture so the destructive RED behavior can only touch the copy.
    src = tmp_path / "src.svar"
    shutil.copytree(ddir / "biallelic.vcf.svar", src)
    sv = SparseVar(src)
    with pytest.raises(ValueError, match="same path|output.*source|in place"):
        sv.write_view(
            regions=(sv.contigs[0], 0, 1_000_000),
            samples=sv.available_samples[:2],
            output=src,
            overwrite=True,
        )
    # Source dataset is intact.
    assert (src / "metadata.json").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/test_svar_write_view.py::test_write_view_self_overwrite_guard -v`
Expected: FAIL — without the guard, `write_view` deletes `src` and errors with something other than the expected `ValueError` message (and `metadata.json` is gone).

- [ ] **Step 3: Add the guard**

In `genoray/_svar.py`, in the Band A block added in Task 1, immediately after the output-exists `FileExistsError` check, add:

```python
        # Writing a view in place would rmtree the source under overwrite=True.
        if output.resolve() == self.path.resolve():
            raise ValueError(
                "output resolves to the same path as the source dataset; "
                "write_view cannot write a view in place"
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run pytest tests/test_svar_write_view.py::test_write_view_self_overwrite_guard -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "$(cat <<'EOF'
feat: guard write_view against writing a view in place

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CLI parse-time path validation for `genoray view`

Make `source`, `--samples-file`, and `--regions-file` fail at arg-parse time when the path does not exist.

**Files:**
- Modify: `genoray/_cli/__main__.py` — `view` (`:179`-`:196`)
- Test: `tests/cli/test_view_cli.py`

**Interfaces:**
- Consumes: `cyclopts.validators.Path` (verified present: `validators.Path(exists=True, file_okay=..., dir_okay=...)`).
- Produces: `genoray view` exits non-zero with a validation error before the function body runs when `source` is not an existing directory, or `--samples-file` / `--regions-file` is supplied but not an existing file. `None` defaults remain valid (validator only runs when a value is supplied).

- [ ] **Step 1: Write the failing tests**

Append to `tests/cli/test_view_cli.py`:

```python
def test_view_missing_source_errors(tmp_path: Path):
    out = tmp_path / "view.svar"
    r = _run(["view", str(tmp_path / "nope.svar"), str(out), "-s", "A"])
    assert r.returncode != 0
    assert "does not exist" in (r.stderr + r.stdout).lower()


def test_view_missing_samples_file_errors(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        ["view", str(tiny_svar), str(out), "-S", str(tmp_path / "nope.txt")]
    )
    assert r.returncode != 0
    assert "does not exist" in (r.stderr + r.stdout).lower()


def test_view_missing_regions_file_errors(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        ["view", str(tiny_svar), str(out), "-R", str(tmp_path / "nope.bed")]
    )
    assert r.returncode != 0
    assert "does not exist" in (r.stderr + r.stdout).lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/cli/test_view_cli.py -k "missing_source or missing_samples_file or missing_regions_file" -v`
Expected: FAIL — currently `source` errors deep in `SparseVar.__init__` (message may differ) and the file flags error mid-run with raw Python tracebacks rather than a clean "does not exist" validation message; at minimum the message assertion fails.

- [ ] **Step 3: Add validators to the `view` signature**

In `genoray/_cli/__main__.py`, ensure `validators` is imported:

```python
from cyclopts import App, Parameter, validators
```

Then update the `view` parameters (`:180`, `:184`-`:190`):

```python
    source: Annotated[
        Path, Parameter(validator=validators.Path(exists=True, dir_okay=True, file_okay=False))
    ],
    out: Path,
    *,
    regions: Annotated[str | None, Parameter(name=["--regions", "-r"])] = None,
    regions_file: Annotated[
        Path | None,
        Parameter(
            name=["--regions-file", "-R"],
            validator=validators.Path(exists=True, dir_okay=False, file_okay=True),
        ),
    ] = None,
    samples: Annotated[str | None, Parameter(name=["--samples", "-s"])] = None,
    samples_file: Annotated[
        Path | None,
        Parameter(
            name=["--samples-file", "-S"],
            validator=validators.Path(exists=True, dir_okay=False, file_okay=True),
        ),
    ] = None,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/cli/test_view_cli.py -k "missing_source or missing_samples_file or missing_regions_file" -v`
Expected: PASS.

- [ ] **Step 5: Run the full CLI suite for regressions**

Run: `pixi run pytest tests/cli/test_view_cli.py -q`
Expected: all pass (the existing happy-path and no-args tests still green).

- [ ] **Step 6: Commit**

```bash
git add genoray/_cli/__main__.py tests/cli/test_view_cli.py
git commit -m "$(cat <<'EOF'
feat: validate view source/regions-file/samples-file at parse time

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Docstring note + SKILL.md verification + full suite

Document the up-front reference validation and confirm no public-API doc drift.

**Files:**
- Modify: `genoray/_svar.py` — `write_view` docstring (`:1863`-`:1870`, the `reference` parameter)
- Verify: `skills/genoray-api/SKILL.md`

- [ ] **Step 1: Add a docstring note for `reference`**

In `write_view`'s docstring `reference` entry, append a sentence:

```
            annotation is performed and the output will not have a ``mutcat``
            field.  When provided, the FASTA is validated up front (before any
            output is written) so a bad path fails fast.
```

- [ ] **Step 2: Verify SKILL.md needs no change**

Run: `grep -n "write_view\|in place\|overwrite" skills/genoray-api/SKILL.md`
Inspect the matches. The changes are timing/safety only — the `write_view` signature, return type, output schema, and coordinate/missing-value conventions are unchanged. The only genuinely new error condition is the self-overwrite `ValueError`. If the SKILL.md `write_view` section enumerates error conditions or documents `overwrite`, add one line noting that `output` must not be the source directory; otherwise leave SKILL.md unchanged.

- [ ] **Step 3: Run the full test suite**

Run: `pixi run pytest -q`
Expected: all pass (or the same skip/xfail counts as before the branch).

- [ ] **Step 4: Lint**

Run: `pixi run ruff check genoray tests && pixi run ruff format --check genoray tests`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs: note up-front reference validation in write_view

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

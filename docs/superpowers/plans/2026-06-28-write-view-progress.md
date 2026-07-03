# write_view Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, phase-level `rich` progress bar to `SparseVar.write_view` and the `genoray view` CLI, defaulting off.

**Architecture:** A `progress: bool = False` keyword on `write_view` constructs a `rich.progress.Progress` bar that wraps only Band C (the write, after all fail-fast validation + the destructive `rmtree`/`mkdir`). Each major write step advances the bar one tick; the field loop advances once per field. When `progress=False`, no `Progress` object is built and the write path is byte-identical to today. The CLI passes a `--progress` flag straight through.

**Tech Stack:** Python, `rich.progress` (already a dependency, already imported in `_svar.py`), `cyclopts` (CLI), `numpy`/`numba`/`polars` (existing write internals), `pytest`.

## Global Constraints

- Stacked on PR #75. Branch `feat/write-view-progress` already exists off `feat/write-view-fail-fast` (current HEAD). PR base MUST be `feat/write-view-fail-fast`, not `main`.
- No change to output bytes, schema, dtypes, or coordinate/missing-value conventions. `progress=False` (default) must be byte-identical to current behavior with no `Progress` object constructed.
- Do NOT alter the fail-fast ordering from PR #75 (Bands A/B and the destructive `rmtree`/`mkdir` commit point). The bar begins only after that commit point.
- Conventional Commits for every commit (`feat:`, `test:`, `docs:`).
- Tests run via `pixi run pytest`. Lint via `ruff check genoray tests` + `ruff format genoray tests`.
- CLAUDE.md policy: `write_view`/`--progress` are public names, so `skills/genoray-api/SKILL.md` MUST be updated in this PR (Task 3).
- The `rich` symbols `Progress` and `MofNCompleteColumn` are already imported at `genoray/_svar.py:30` — reuse them; do not re-import.

---

### Task 1: `progress` keyword on `SparseVar.write_view`

**Files:**
- Modify: `genoray/_svar.py` — `write_view` signature (`:1828`-`:1839`), docstring (`:1880`-`:1889`), and Band C body (`:1990`-`:2115`).
- Test: `tests/test_svar_write_view.py` (append at end; `svar` fixture exists at `:420`, `ddir` at `:141`).

**Interfaces:**
- Consumes: existing Band C locals — `output` (`Path`), `fields_to_write` (`list[str]`), `ref_obj` (`Reference | None`), and all existing numba/memmap write code.
- Produces: `write_view(regions, samples, output, fields=None, reference=None, merge_overlapping=False, regions_overlap="pos", overwrite=False, threads=None, progress=False) -> None`. New trailing keyword `progress: bool = False`. When `True`, a phase-level bar is shown; when `False`, behavior and output bytes are unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar_write_view.py`:

```python
# ---------------------------------------------------------------------------
# Opt-in progress bar (phase-level)
# ---------------------------------------------------------------------------


def _dir_digest(root: Path) -> dict[str, bytes]:
    """Map every file under `root` to its raw bytes, for byte-identical compares."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_write_view_progress_defaults_false():
    import inspect

    sig = inspect.signature(SparseVar.write_view)
    assert sig.parameters["progress"].default is False


def test_write_view_progress_byte_identical(tmp_path: Path, svar: SparseVar):
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    a = tmp_path / "a.svar"
    b = tmp_path / "b.svar"
    svar.write_view(
        regions=(contig, 0, 1_000_000), samples=samples, output=a, progress=False
    )
    svar.write_view(
        regions=(contig, 0, 1_000_000), samples=samples, output=b, progress=True
    )
    # The bar must not perturb the written output in any way.
    assert _dir_digest(a) == _dir_digest(b)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_svar_write_view.py -k "progress" -v`
Expected: both FAIL — `test_write_view_progress_defaults_false` with `KeyError: 'progress'`, and `test_write_view_progress_byte_identical` with `TypeError: write_view() got an unexpected keyword argument 'progress'`.

- [ ] **Step 3: Add the `progress` parameter to the signature**

In `genoray/_svar.py`, change the `write_view` signature (currently ending at `:1838`-`:1839`):

```python
        overwrite: bool = False,
        threads: int | None = None,
    ) -> None:
```

to:

```python
        overwrite: bool = False,
        threads: int | None = None,
        progress: bool = False,
    ) -> None:
```

- [ ] **Step 4: Document the parameter in the docstring**

In `genoray/_svar.py`, in the `write_view` docstring, after the `threads` entry (`:1880`-`:1881`):

```python
        threads
            Number of Numba threads to use.  ``None`` uses all available CPUs.
```

add:

```python
        progress
            If ``True``, display a phase-level :mod:`rich` progress bar while the
            view is written (one tick per major step: counting, genotypes, each
            field, index build, and mutation annotation when *reference* is
            given).  Defaults to ``False`` (no bar, no overhead).
```

- [ ] **Step 5: Wrap Band C in the progress bar**

In `genoray/_svar.py`, add `nullcontext` to the local imports at the top of `write_view` (currently `:1890`):

```python
        from ._utils import _resolve_threads, numba_threads
```

becomes:

```python
        from contextlib import nullcontext

        from ._utils import _resolve_threads, numba_threads
```

Then replace the entire Band C block. The current block runs from the `# --- Band C: commit ...` comment (`:1990`) through the end of the method (`:2115`):

```python
        # --- Band C: commit. All validation passed; now (re)create the output. ---
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)

        # --- 5. Pass 1: count kept entries per output slot ---
        out_lengths = np.zeros(n_out * ploidy, dtype=np.int64)
        with numba_threads(threads_resolved):
            _nb_count_kept(
                self.genos.data,
                self.genos.offsets,
                src_sample_idxs,
                ploidy,
                kept_var_idxs,
                out_lengths,
            )

        new_offsets = lengths_to_offsets(out_lengths.reshape(n_out, ploidy))

        # --- 6. Write offsets.npy ---
        offsets_mm = np.memmap(
            output / "offsets.npy",
            dtype=np.int64,
            mode="w+",
            shape=new_offsets.shape,
        )
        offsets_mm[:] = new_offsets
        offsets_mm.flush()

        # Allocate output variant_idxs memmap
        n_entries = int(new_offsets[-1])
        out_var_idxs_mm = np.memmap(
            output / "variant_idxs.npy",
            dtype=V_IDX_TYPE,
            mode="w+",
            shape=(n_entries,),
        )

        # --- 7. Pass 2 (genos): write remapped variant indices ---
        with numba_threads(threads_resolved):
            _nb_write_var_idxs(
                self.genos.data,
                self.genos.offsets,
                src_sample_idxs,
                ploidy,
                kept_var_idxs,
                new_offsets.ravel(),
                out_var_idxs_mm,
            )
        out_var_idxs_mm.flush()

        # --- 8. Pass 2 (fields): write each field ---
        for name in fields_to_write:
            dtype = self.available_fields[name]
            src_field_rag = _open_fmt(
                name, dtype, self.path, (self.n_samples, ploidy, None), "r"
            )
            out_field_mm = np.memmap(
                output / f"{name}.npy",
                dtype=dtype,
                mode="w+",
                shape=(n_entries,),
            )
            with numba_threads(threads_resolved):
                _nb_write_field(
                    src_field_rag.data,
                    self.genos.data,
                    self.genos.offsets,
                    src_sample_idxs,
                    ploidy,
                    kept_var_idxs,
                    new_offsets.ravel(),
                    out_field_mm,
                )
            out_field_mm.flush()
            del src_field_rag

        # --- 9. Build new index (streaming: never materialize the full index) ---
        # Compute AFs over the written genos.
        n_alleles = n_out * ploidy
        afs = np.zeros(len(kept_var_idxs), dtype=np.float32)
        _nb_af_helper(afs, out_var_idxs_mm, new_offsets.ravel(), n_alleles)

        # Small, output-sized frame keyed by the kept physical row index.
        # The row-index column produced by scan_ipc is UInt32 (see _scan_index);
        # match that dtype so the join keys align.
        idx_dtype = self._index_lazy.collect_schema()["index"]
        af_frame = pl.DataFrame(
            {
                "index": pl.Series(kept_var_idxs).cast(idx_dtype),
                "AF": pl.Series(afs),
            }
        )

        base = self._index_lazy
        drop_existing_af = ["AF"] if "AF" in base.collect_schema().names() else []
        out_index = (
            base.drop(drop_existing_af)
            .join(
                af_frame.lazy(), on="index", how="inner"
            )  # filter to kept + attach AF
            .sort(
                "index"
            )  # row order must match the ascending kept_var_idxs / written genos
            .drop("index")  # physical row index is not part of the output schema
        )
        # sink_ipc forces the streaming engine, so the inner join filters the
        # scan down to output size before the sort — peak RAM scales with the
        # selected subset, not the full input index.
        out_index.sink_ipc(SparseVar._index_path(output))

        # --- 10. Write metadata.json ---
        with open(output / "metadata.json", "w") as f:
            json_str = SparseVarMetadata(
                version=CURRENT_VERSION,
                samples=caller_samples,
                ploidy=ploidy,
                contigs=self.contigs,
                fields={n: self.available_fields[n].name for n in fields_to_write},
            ).model_dump_json()
            f.write(json_str)

        # --- 11. Optionally recompute mutcat on the output view ---
        if ref_obj is not None:
            out_svar = SparseVar(output)
            out_svar.annotate_mutations(ref_obj, write_back=True)
```

Replace that entire block with the following. The numba/memmap/polars code is unchanged — only re-indented one level under the `with` and interleaved with `pbar` updates:

```python
        # --- Band C: commit. All validation passed; now (re)create the output. ---
        # Phase-level progress bar (opt-in): one tick per major write step,
        # plus one per field, plus the optional mutcat annotation. Built only
        # when progress=True so the default path constructs no Progress object.
        pbar = (
            Progress(*Progress.get_default_columns(), MofNCompleteColumn())
            if progress
            else None
        )
        n_steps = 3 + len(fields_to_write) + (1 if ref_obj is not None else 0)

        with pbar or nullcontext():
            task = (
                pbar.add_task("counting entries", total=n_steps)
                if pbar is not None
                else None
            )

            def _step(desc: str) -> None:
                """Mark the current phase complete and label the next one."""
                if pbar is not None:
                    pbar.advance(task)
                    pbar.update(task, description=desc)

            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True)

            # --- 5. Pass 1: count kept entries per output slot ---
            out_lengths = np.zeros(n_out * ploidy, dtype=np.int64)
            with numba_threads(threads_resolved):
                _nb_count_kept(
                    self.genos.data,
                    self.genos.offsets,
                    src_sample_idxs,
                    ploidy,
                    kept_var_idxs,
                    out_lengths,
                )

            new_offsets = lengths_to_offsets(out_lengths.reshape(n_out, ploidy))

            # --- 6. Write offsets.npy ---
            offsets_mm = np.memmap(
                output / "offsets.npy",
                dtype=np.int64,
                mode="w+",
                shape=new_offsets.shape,
            )
            offsets_mm[:] = new_offsets
            offsets_mm.flush()

            # Allocate output variant_idxs memmap
            n_entries = int(new_offsets[-1])
            out_var_idxs_mm = np.memmap(
                output / "variant_idxs.npy",
                dtype=V_IDX_TYPE,
                mode="w+",
                shape=(n_entries,),
            )
            _step("writing genotypes")

            # --- 7. Pass 2 (genos): write remapped variant indices ---
            with numba_threads(threads_resolved):
                _nb_write_var_idxs(
                    self.genos.data,
                    self.genos.offsets,
                    src_sample_idxs,
                    ploidy,
                    kept_var_idxs,
                    new_offsets.ravel(),
                    out_var_idxs_mm,
                )
            out_var_idxs_mm.flush()

            # --- 8. Pass 2 (fields): write each field ---
            for name in fields_to_write:
                _step(f"field: {name}")
                dtype = self.available_fields[name]
                src_field_rag = _open_fmt(
                    name, dtype, self.path, (self.n_samples, ploidy, None), "r"
                )
                out_field_mm = np.memmap(
                    output / f"{name}.npy",
                    dtype=dtype,
                    mode="w+",
                    shape=(n_entries,),
                )
                with numba_threads(threads_resolved):
                    _nb_write_field(
                        src_field_rag.data,
                        self.genos.data,
                        self.genos.offsets,
                        src_sample_idxs,
                        ploidy,
                        kept_var_idxs,
                        new_offsets.ravel(),
                        out_field_mm,
                    )
                out_field_mm.flush()
                del src_field_rag

            _step("building index")

            # --- 9. Build new index (streaming: never materialize the full index) ---
            # Compute AFs over the written genos.
            n_alleles = n_out * ploidy
            afs = np.zeros(len(kept_var_idxs), dtype=np.float32)
            _nb_af_helper(afs, out_var_idxs_mm, new_offsets.ravel(), n_alleles)

            # Small, output-sized frame keyed by the kept physical row index.
            # The row-index column produced by scan_ipc is UInt32 (see _scan_index);
            # match that dtype so the join keys align.
            idx_dtype = self._index_lazy.collect_schema()["index"]
            af_frame = pl.DataFrame(
                {
                    "index": pl.Series(kept_var_idxs).cast(idx_dtype),
                    "AF": pl.Series(afs),
                }
            )

            base = self._index_lazy
            drop_existing_af = ["AF"] if "AF" in base.collect_schema().names() else []
            out_index = (
                base.drop(drop_existing_af)
                .join(
                    af_frame.lazy(), on="index", how="inner"
                )  # filter to kept + attach AF
                .sort(
                    "index"
                )  # row order must match the ascending kept_var_idxs / written genos
                .drop("index")  # physical row index is not part of the output schema
            )
            # sink_ipc forces the streaming engine, so the inner join filters the
            # scan down to output size before the sort — peak RAM scales with the
            # selected subset, not the full input index.
            out_index.sink_ipc(SparseVar._index_path(output))

            # --- 10. Write metadata.json ---
            with open(output / "metadata.json", "w") as f:
                json_str = SparseVarMetadata(
                    version=CURRENT_VERSION,
                    samples=caller_samples,
                    ploidy=ploidy,
                    contigs=self.contigs,
                    fields={n: self.available_fields[n].name for n in fields_to_write},
                ).model_dump_json()
                f.write(json_str)

            # --- 11. Optionally recompute mutcat on the output view ---
            if ref_obj is not None:
                _step("annotating mutations")
                out_svar = SparseVar(output)
                out_svar.annotate_mutations(ref_obj, write_back=True)

            # Final advance so the bar reads N/N on completion.
            if pbar is not None:
                pbar.advance(task)
```

Note on tick accounting: the task starts at description `"counting entries"` with `0/n_steps`. Each `_step(...)` call advances by one and relabels to the phase about to run; the trailing `pbar.advance(task)` completes the count to `n_steps`. Number of advances = 1 (genotypes) + F (fields) + 1 (index) + (1 if ref) + 1 (final) = 3 + F + (0/1) = `n_steps`. ✓

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_svar_write_view.py -k "progress" -v`
Expected: both PASS.

- [ ] **Step 7: Run the full write_view suite for regressions**

Run: `pixi run pytest tests/test_svar_write_view.py -q`
Expected: all pass (previous count was 55 passed; now +2).

- [ ] **Step 8: Lint**

Run: `ruff check genoray tests && ruff format genoray tests`
Expected: clean (or only auto-formatting applied).

- [ ] **Step 9: Commit**

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat: opt-in phase-level progress bar in write_view"
```

---

### Task 2: `--progress` flag on the `genoray view` CLI

**Files:**
- Modify: `genoray/_cli/__main__.py` — `view` signature (`:179`-`:209`), docstring (`:243`-`:247`), and the `sv.write_view(...)` call (`:296`-`:305`).
- Test: `tests/cli/test_view_cli.py` (append; `_run` helper at `:11`, `tiny_svar` fixture used throughout, `SparseVar` imported at `:8`).

**Interfaces:**
- Consumes: `write_view(..., progress: bool = False)` from Task 1.
- Produces: `genoray view ... --progress` (default `False`) forwards `progress=` to `write_view`.

- [ ] **Step 1: Write the failing test**

Append to `tests/cli/test_view_cli.py`:

```python
def _dir_digest(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_view_progress_flag_byte_identical(tmp_path: Path, tiny_svar: Path):
    out_a = tmp_path / "a.svar"
    out_b = tmp_path / "b.svar"
    base = ["view", str(tiny_svar)]
    region = ["-r", "chr1:1-100", "-s", "A"]
    r1 = _run([*base, str(out_a), *region])
    r2 = _run([*base, str(out_b), *region, "--progress"])
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    # --progress must not change the written output.
    assert _dir_digest(out_a) == _dir_digest(out_b)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/cli/test_view_cli.py::test_view_progress_flag_byte_identical -v`
Expected: FAIL — `r2.returncode != 0` because cyclopts rejects the unknown `--progress` flag (error on stderr).

- [ ] **Step 3: Add the `progress` parameter to the CLI signature**

In `genoray/_cli/__main__.py`, change the end of the `view` signature (`:208`-`:209`):

```python
    threads: Annotated[int | None, Parameter(name=["--threads", "-@"])] = None,
) -> None:
```

to:

```python
    threads: Annotated[int | None, Parameter(name=["--threads", "-@"])] = None,
    progress: bool = False,
) -> None:
```

- [ ] **Step 4: Document it in the CLI docstring**

In `genoray/_cli/__main__.py`, after the `threads` docstring entry (`:245`-`:246`):

```python
    threads
        Number of threads. Defaults to all available CPUs.
```

add:

```python
    progress
        If set, show a phase-level progress bar while writing the view.
```

- [ ] **Step 5: Forward the flag to `write_view`**

In `genoray/_cli/__main__.py`, change the `sv.write_view(...)` call (`:296`-`:305`):

```python
    sv.write_view(
        regions=regions_arg,
        samples=samples_arg,
        output=out,
        fields=fields,
        merge_overlapping=merge_overlapping,
        regions_overlap=regions_overlap,
        overwrite=overwrite,
        threads=threads,
    )
```

to:

```python
    sv.write_view(
        regions=regions_arg,
        samples=samples_arg,
        output=out,
        fields=fields,
        merge_overlapping=merge_overlapping,
        regions_overlap=regions_overlap,
        overwrite=overwrite,
        threads=threads,
        progress=progress,
    )
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pixi run pytest tests/cli/test_view_cli.py::test_view_progress_flag_byte_identical -v`
Expected: PASS.

- [ ] **Step 7: Run the full CLI view suite for regressions**

Run: `pixi run pytest tests/cli/test_view_cli.py -q`
Expected: all pass.

- [ ] **Step 8: Lint**

Run: `ruff check genoray tests && ruff format genoray tests`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add genoray/_cli/__main__.py tests/cli/test_view_cli.py
git commit -m "feat: add --progress flag to genoray view CLI"
```

---

### Task 3: Document `progress` / `--progress` in SKILL.md

**Files:**
- Modify: `skills/genoray-api/SKILL.md`.

**Interfaces:**
- Consumes: the public `progress` kwarg (Task 1) and `--progress` flag (Task 2).
- Produces: SKILL.md documents both, satisfying the CLAUDE.md public-name policy.

There is no dedicated `write_view` signature block in SKILL.md today; `write_view` is referenced in the mutation-catalogue section and the Common Mistakes table. Add a short, self-contained subsection documenting the new option so an agentic reader can discover it.

- [ ] **Step 1: Locate the insertion point**

Run: `pixi run python -c "p=open('skills/genoray-api/SKILL.md').read().splitlines(); print([i+1 for i,l in enumerate(p) if l.startswith('### Overview')])"`
Expected: prints the line number of the `### Overview` heading (the mutation-catalogue overview, ~line 300). Insert the new subsection immediately *before* that heading so it sits at the end of the preceding section.

- [ ] **Step 2: Add the documentation subsection**

Open `skills/genoray-api/SKILL.md` and insert this block on the blank line immediately before the `### Overview` heading found in Step 1 (keep one blank line above and below):

```markdown
### `write_view` progress bar

`SparseVar.write_view(..., progress=False)` accepts an opt-in `progress` keyword.
When `True`, a phase-level `rich` progress bar is shown while the view is written
(one tick per major step: counting, genotypes, each carried field, the index
build, and mutation annotation when `reference=` is given). It defaults to
`False` — no bar and no overhead — so library and pipeline callers are
unaffected. The `genoray view` CLI exposes the same option as `--progress`
(also default off):

```bash
genoray view in.svar out.svar -r chr1:1-1000 -s A,B --progress
```

The bar is cosmetic: output bytes, schema, and dtypes are identical whether or
not it is enabled.
```

- [ ] **Step 3: Verify the doc renders and references are consistent**

Run: `pixi run python -c "s=open('skills/genoray-api/SKILL.md').read(); assert 'write_view' in s and '--progress' in s and 'progress=False' in s; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs: document write_view progress option in SKILL.md"
```

---

### Final verification

- [ ] **Step 1: Run the full test suite**

Run: `pixi run pytest -q`
Expected: all pass / skips & xfails unchanged from the PR #75 baseline (481 passed, 2 skipped, 16 xfailed), plus the 3 new tests → 484 passed.

- [ ] **Step 2: Lint the whole tree**

Run: `ruff check genoray tests && ruff format --check genoray tests`
Expected: clean.

- [ ] **Step 3: Push and open the stacked PR**

```bash
git push -u origin feat/write-view-progress
gh pr create --base feat/write-view-fail-fast \
  --title "feat: opt-in progress bar for write_view + view CLI" \
  --body "$(cat <<'EOF'
## Summary

Adds an opt-in, phase-level progress bar to `SparseVar.write_view` and the
`genoray view` CLI, defaulting off.

- `write_view(..., progress: bool = False)` — new trailing keyword. When `True`,
  a `rich` progress bar ticks once per major write step (count, genotypes, each
  field, index build, and mutation annotation when `reference=` is given). When
  `False` (default), no `Progress` object is constructed and output bytes are
  unchanged.
- `genoray view ... --progress` — same option on the CLI, default off.
- `skills/genoray-api/SKILL.md` documents both.

Stacked on #75; the bar begins only after that PR's fail-fast commit point.

## Testing

- New tests assert byte-identical output with `progress` on vs. off (API + CLI)
  and that the `progress` default is `False`.
- `pixi run pytest -q` green; `ruff check` + `ruff format --check` clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR opens with base `feat/write-view-fail-fast` (stacked on #75), NOT `main`.

# SP-1 — `_svar.py` → `_svar/` package (decompose along lifecycle seams)

**Date:** 2026-07-08
**Branch:** `sp1-svar-package`
**Roadmap:** [`docs/roadmap/clean-code-audit.md`](../../roadmap/clean-code-audit.md) — sub-project #1 (flagship)
**Findings:** [`docs/roadmap/audit-findings/01-svar.md`](../../roadmap/audit-findings/01-svar.md)
**Size:** L · **Risk:** low

## Goal

`python/genoray/_svar.py` is 3,283 lines holding ~10 unrelated responsibilities: a
region/sample normalization library, a dense↔sparse converter, two ~200-line parallel
writers, a pile of Numba kernels, a memmap I/O layer, a mutation-signature/GTF annotator,
and the ~1,700-line `SparseVar` class that ties them together. Nothing but history forces
them into one file, and the file is far past the point where a reader can hold it in their
head.

SP-1 splits it into a `genoray/_svar/` **package along lifecycle seams**, extracts the one
shared writer body that the two near-duplicate `from_vcf`/`from_pgen` methods currently
copy-paste, and converges the four hand-rolled sample-validation sites onto a single
helper.

**Core promise: pure code-motion + a re-export shim.** The public surface
(`genoray.SparseVar`, and the internal `genoray._svar.SparseVar` import path) and all
observable behavior stay identical, gated on the existing test suite (vcfixture oracle +
parity tests) staying green. There is **no** public-API change and therefore **no**
`skills/genoray-api/SKILL.md` update (confirmed at the end).

## Non-goals (deferred, to avoid scope creep)

These `01-svar.md` findings are **out of scope** for SP-1 and stay with their assigned
sub-projects:

- svar2 `dict[str, ndarray]` bundles → frozen dataclass — **SP-5**.
- svar2 milestone-jargon (M6a/b/c) docstrings; svar2 mixin `Any`/`type: ignore` → Protocol;
  overlapping `read_ranges`/`overlap_batch` reconciliation — **SP-6**.
- `dense2sparse` public-vs-private decision — **SP-6** (SP-1 keeps its current name and
  visibility; it only *moves* it into `_convert.py` and re-exports it unchanged from
  `_svar/__init__.py`).
- Vectorizing the Python row-loop in `_resolve_kept_rows` — its own perf item (not SP-1).
- Unifying the three progress-bar mechanisms (`joblib_progress`/`tqdm`/`rich`) — low
  priority, systemic, not this PR.
- Redundant runtime checks of `Literal`-typed args — policy decision, not code-motion.

**`python/genoray/_svar2.py`, `_svar2_batch.py`, `_svar2_decode.py` are not touched by
SP-1.** The one opportunistic exception (see Work item D) is the stale `self.var_table`
docstrings inside code we are already relocating.

## Target package layout

`genoray/_svar.py` (module) → `genoray/_svar/` (package). Mapping of current top-level
symbols to new files (all names stay private except the three re-exported from
`__init__.py`). Line numbers are current-tree references for orientation, not contracts —
re-locate symbols by name at implementation time.

| New file | Contents (current line refs) |
|---|---|
| `_svar/_regions.py` | `_coerce_bed_schema`, `_normalize_regions`, `_normalize_samples`, `_validate_fields`, `_resolve_kept_rows`, `_resolve_kept_var_idxs` (49–329) |
| `_svar/_convert.py` | `dense2sparse` (+ overloads), `_dense2sparse_with_length`, `_process_contig_vcf`, `_process_contig_pgen`, `_concat_data`, and the **new** `_write_from_reader` (333–460, 2445–2628, 2688–2807) |
| `_svar/_io.py` | index build/write helpers (`_write_filtered_index`, `_subset_var_idxs_and_recompute_af`, `_build_working_index`, `_write_index_from_working`) + memmap helpers (`_open_genos`, `_open_fmt`, `_write_genos`, `_write_dosages`) (2305–2443, 2629–2687) |
| `_svar/_kernels.py` | every `@nb.njit` function (`_nb_af_helper`, `_nb_count_kept`, `_nb_count_mac_per_kept`, `_nb_write_var_idxs`, `_nb_write_field`, `_copy_chunk_helper`, `_copy_chunk_dosages_helper`, `_find_starts_ends`, `_length_walk_n_keep`, `_dense2sparse_count`, `_dense2sparse_fill`, `_find_starts_ends_with_length`) — both blocks (2175–2303, 2808–3107) |
| `_svar/_annotate.py` | `SparseVarAnnotateMixin` (the four public annotate methods `annotate_with_gtf`, `annotate_mutations`, `mutation_matrix`, `assign_signatures`, plus `cache_afs`, `_load_all_attrs`, `_compute_afs`, `_write_afs`) + module helpers `_empty_annot`, `_get_strand_and_codon_pos`, `_load_gtf` (1457–1806, 3109–3283) |
| `_svar/_core.py` | `SparseVarMetadata` (pydantic `BaseModel`), `SparseVar` (open + query + `write_view`), inheriting `SparseVarAnnotateMixin` |
| `_svar/__init__.py` | `from ._core import SparseVar, SparseVarMetadata`; `from ._convert import dense2sparse`; matching `__all__` |

### Layout notes

- **Import cycle avoidance:** `_core` imports from `_regions`, `_convert`, `_io`,
  `_kernels`, `_annotate`; those leaf modules must **not** import `_core`. The
  `SparseVarAnnotateMixin` references host attributes (`self.index`, `self.genos`,
  `self.n_samples`, …) that exist only on the concrete `SparseVar` — this is resolved at
  runtime by the class inheriting the mixin, and typed loosely for now (a `SparseVar2Host`-
  style Protocol is an SP-6 concern; do not add one here). Keep the mixin's own imports
  limited to leaf modules and stdlib/third-party.
- **Numba cache:** moving `@nb.njit(cache=True)` kernels into `_svar/_kernels.py` changes
  their cache module path, so the **first** post-merge run recompiles them once. This is a
  one-time cache miss, not a behavior change.
- **`__init__.py` re-export is the compatibility contract.** `genoray/__init__.py` reaches
  `SparseVar` via the `_LAZY` map entry `("genoray._svar", "SparseVar")`; a package
  `_svar/__init__.py` that exports `SparseVar` keeps that path working with a zero-diff to
  `genoray/__init__.py`.

## Work items (commit-grouped)

### A. Package split — `refactor: split _svar.py into _svar/ package`

Pure code motion. Create `genoray/_svar/` with the files above, move each symbol group
verbatim (adjusting only imports), delete `genoray/_svar.py`, add the re-export
`__init__.py`. No logic edits in this commit. Gate: full suite green + `genoray._svar`
still importable.

### B. Writer DRY — `refactor(svar): extract _write_from_reader`

`from_vcf` and `from_pgen` share an identical spine, and their bodies carry the manual-DRY
comment *"mirrors from_vcf; keep in sync"*. The sequence both perform:

1. overwrite check
2. sample resolution
3. `_build_working_index`
4. `_resolve_kept_rows` + sort
5. per-contig keep-index bucketing
6. metadata write
7. up-front index write when **not** subsetting
8. `parse_memory` / job sizing
9. `TemporaryDirectory` + `joblib.Parallel`
10. `_concat_data`
11. the `subsetting_samples` MAC-drop finalize

The **only** reader-specific parts are: which per-contig task to build
(`_process_contig_vcf` vs `_process_contig_pgen`) and a couple of reader-derived values
(sample list, ploidy).

**Plan:** add a private `_write_from_reader(*, <common kwargs>, make_contig_task:
Callable)` in `_convert.py`. Both classmethods become thin wrappers passing their
reader-specific callback. ~400 lines → ~150; deletes the "keep in sync" hazard.

This is the one **med-risk** piece (shared control flow around parallel dispatch; must
preserve each writer's contig-block invariants and the not-subsetting-vs-subsetting index
paths). It lands as its **own commit** with the write/parity suite as the gate. Preserve
`from_vcf`/`from_pgen` signatures and return types **exactly** (including
`SparseVar.from_*` returning `None`; the `SparseVar2` int-return divergence is an SP-6
doc item, untouched here).

### C. Sample-validation convergence — `refactor(svar): converge sample validation`

Four sites hand-roll the same validation: `_find_starts_ends`,
`_find_starts_ends_with_length`, `read_ranges`, `read_ranges_with_length`. Each currently:

```python
samples = np.atleast_1d(np.array(samples))          # coerce
if missing := set(samples) - set(self.available_samples):
    raise ValueError(...)
s_idxs = cast(NDArray[np.int64], self._s2i[samples])  # names -> int indices
```

They do **not** dedup and they preserve caller order **and duplicates**.

**Decision (behavior-preserving; chosen over routing through `_normalize_samples`).**
`_normalize_samples` *dedups* and returns `list[str]`; routing the read paths through it
would silently collapse duplicate-sample input (`samples=["NA1","NA1"]` → 1 column instead
of 2), which contradicts SP-1's behavior-preserving promise. Instead, extract a dedicated
**read-path** helper in `_core` (or `_regions`):

```python
def _resolve_sample_idxs(samples, available, s2i) -> tuple[NDArray[str], NDArray[np.int64]]:
    """Validate `samples` against `available`, preserving order AND duplicates,
    returning the coerced name array and its integer sample indices."""
```

- Coerces to array **first** (so a bare `str` like `"NA00001"` is one sample, not
  iterated character-by-character — this is the SP-0 bug-#2 shape; SP-0's minimal hotfix in
  `read_ranges_with_length` is now subsumed here).
- Validates via set-difference, raising the existing `ValueError` message shape.
- Folds in the repeated `self._s2i[...]` lookup, removing more duplication than the
  audit's minimum.
- Preserves duplicates and order → strictly behavior-preserving.

All four sites call it and drop their inline blocks. `_normalize_samples` (dedup semantics)
is left as-is for the writer/subsetting paths that legitimately want dedup.

**The SP-0 regression test** for the bare-`str` case in `read_ranges_with_length` must stay
green. Add a small test asserting duplicate-sample input still yields duplicate columns
(guards the "don't accidentally dedup" invariant on the read path).

### D. Opportunistic doc fix (only within relocated code) — folded into commit A or C

While relocating `annotate_with_gtf`/`_get_strand_and_codon_pos`, fix the stale docstrings
that reference a nonexistent `self.var_table` (the attribute is `self.index`; the write-back
mutates `self.index`). Doc-only, no runtime effect. This is the sole in-scope touch of
otherwise-deferred findings, justified because we are already moving that exact code.

## Invariants & constraints

1. **Behavior-preserving.** No observable change to any public method's inputs, outputs,
   shapes, dtypes, or raised errors. The read-path sample helper (item C) is deliberately
   designed to preserve duplicates/order so this holds; the writer extraction (item B)
   preserves both writers' signatures and control flow.
2. **No public-API change → no `SKILL.md` change.** `genoray.__all__`/`_LAZY` and
   `genoray/__init__.py` get a **zero diff**. Confirm `genoray._svar.SparseVar`,
   `genoray.SparseVar`, `genoray._svar.SparseVarMetadata`, and `genoray._svar.dense2sparse`
   all still import. Verify at the end.
3. **Leaf modules do not import `_core`** (no import cycles).
4. **Small, reviewable commits** (A / B / C), Conventional Commits; no god-branch.
5. **Re-verify symbol locations by name** at implementation time — the line numbers above
   are orientation only and predate any local edits.

## Verification

All must pass before the PR is considered done:

- `pixi run pytest` — full Python suite green, including the SP-0 bare-`str` regression test
  and the new duplicate-sample read-path test.
- `cargo test --no-default-features --features conversion` — Rust suite green (the
  `--no-default-features` flag is required or the pyo3 test binary fails to link with
  `undefined symbol: _Py_Dealloc`; `--features conversion` is required or the integration
  tests fail to compile against `rust-htslib`).
- Pre-commit / pre-push hooks green: `ruff check`, `ruff format`, `pyrefly` type-check,
  `cargo fmt`, `cargo clippy`, `commitizen`.
- **Public-surface guard:** `git diff` on `genoray/__init__.py` is empty; a scratch
  `python -c "import genoray; from genoray._svar import SparseVar, SparseVarMetadata,
  dense2sparse"` succeeds.
- Line-count sanity: no single new `_svar/*.py` file approaches the old 3,283; the writer
  extraction reduces total LOC.

## Sequencing

SP-1 is on the Python track and is independent of the Rust track (SP-2/3/5). It should
precede SP-4 (VCF/PGEN consistency) and SP-6 (API reconciliation), which ratify
public-surface decisions this refactor surfaces. Each remaining sub-project keeps its own
brainstorm → spec → plan → PR cycle.

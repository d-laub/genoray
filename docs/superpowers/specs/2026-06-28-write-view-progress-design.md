# write_view progress bar (opt-in, phase-level)

**Date:** 2026-06-28
**Type:** feature (additive, opt-in)
**Stacked on:** PR #75 (`feat/write-view-fail-fast`); branch `feat/write-view-progress`, PR base `feat/write-view-fail-fast`.

---

## Problem

`SparseVar.write_view` (and the `genoray view` CLI that wraps it) can run for a
long time on large cohorts, but emits no progress feedback. A user has no signal
for how far along a write is, or whether it has stalled. We want an **opt-in**
progress bar, defaulting off so the silent behavior is unchanged for library
callers and pipelines.

## Goals

- Add a `progress: bool = False` keyword to `write_view`.
- Add a matching `--progress` flag (default `False`) to the `genoray view` CLI.
- When `progress=False`, behavior is byte-identical to today, with zero added
  overhead and no `Progress` object constructed.
- When `progress=True`, show a phase-level `rich` progress bar over the write.

## Non-goals

- **Per-variant granularity.** Rejected after analysis:
  - The numba kernels do not share a "variant" iteration unit. `_nb_count_kept`,
    `_nb_write_var_idxs`, and `_nb_write_field` `prange` over **samples**
    (`prange(n_out)`); only `_nb_count_mac_per_kept` iterates variants. A single
    "X/N variants" bar cannot honestly span them.
  - The genuinely slow steps on large cohorts are the memmap field I/O and the
    `sink_ipc` streaming **index build** (polars, not numba). A per-variant bar
    driven from the numba kernels would race to ~100% and then hang during
    `sink_ipc` — worse UX than phase-level.
  - It would require new kernel signatures + a shared per-thread counter +
    background-thread polling, forcing recompilation of all four `cache=True`
    kernels, for a misleading bar.
- Changing any output bytes, schema, dtypes, coordinate/missing-value
  conventions, or the fail-fast ordering from PR #75.

## Design

### Where the bar lives

The bar wraps **Band C only** — the actual write, which begins after all
fail-fast validation and the destructive `rmtree`/`mkdir` at
`genoray/_svar.py:1991`-`1993`. The pre-commit MAC pre-pass and all validation
stay *outside* the bar: nothing is on disk there, and showing a bar before the
commit point would be misleading.

### Ticks (per-field expanded)

Total ticks: `N = 3 + len(fields_to_write) + (1 if ref_obj is not None else 0)`
(count + genos + index, plus one per field, plus annotate if a reference is
given).

| Tick | Covers | Description shown |
|---|---|---|
| 1 | Pass 1 count (`_nb_count_kept`) + write `offsets.npy` | `counting entries` |
| 2 | Pass 2 genos (`_nb_write_var_idxs`) | `writing genotypes` |
| 3 … 2+F | one per field in the field loop | `field: <name>` |
| +1 | `sink_ipc` index build + `metadata.json` | `building index` |
| +1 *(only if `reference` given)* | `annotate_mutations` | `annotating mutations` |

Pattern per step: set the description, run the step, then `advance=1`.

### Implementation shape

`rich.progress.Progress` and `MofNCompleteColumn` are already imported at
`genoray/_svar.py:30` and used at `:2694`. Reuse exactly that style:

```python
from contextlib import nullcontext

pbar = (
    Progress(*Progress.get_default_columns(), MofNCompleteColumn())
    if progress
    else None
)
total = 3 + len(fields_to_write) + (1 if ref_obj is not None else 0)
task = pbar.add_task("Writing view", total=total) if pbar else None

def _step(desc: str) -> None:           # advance + relabel; no-op when disabled
    if pbar is not None:
        pbar.update(task, advance=1, description=desc)

with pbar or nullcontext():
    # ... existing Band C body, with _step(...) calls interleaved ...
```

When `progress=False`: `pbar is None`, `nullcontext()` is entered, `_step` is a
no-op, and the write path is byte-identical to today.

### API surface

- `write_view(..., threads=None, progress: bool = False) -> None` — new trailing
  keyword. Matches the existing `VCF(progress: bool = False)` house convention.
- CLI `view` gains `--progress` (`progress: bool = False` annotated parameter),
  passed straight through to `write_view`.

## Testing

- **API roundtrip:** parametrize the existing `write_view` roundtrip test over
  `progress=(False, True)`; assert output is **byte-identical** across both — the
  bar must not perturb output — and that `progress=True` completes cleanly.
- **CLI:** invoke `view ... --progress`; assert success and output identical to
  the no-flag run.
- **Default guard:** assert `write_view`'s `progress` parameter defaults to
  `False` via signature introspection.

## Docs (required by CLAUDE.md — public name added)

- `write_view` docstring: document the `progress` parameter.
- `skills/genoray-api/SKILL.md`: add `progress` to the `write_view`
  signature/kwargs and document the `--progress` CLI flag.

## Behavior guarantees

- No change to output bytes, schema, dtypes, or coordinate/missing-value
  conventions.
- `progress=False` (the default) is byte-identical to current behavior with no
  added object construction.
- Fail-fast ordering from PR #75 is untouched; the bar begins only after the
  Band C commit point.

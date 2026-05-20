# `genoray view` CLI for `SparseVar.write_view`

**Date:** 2026-05-20
**Branch:** `feat/svar-view-cli` (based on `main`; `feat/svar-write-subset` already merged)
**Companion repo:** [d-laub/genoray-cli](https://github.com/d-laub/genoray-cli)

## Goal

Expose `SparseVar.write_view(...)` (added on `feat/svar-write-subset`) as a CLI
subcommand `genoray view`, so users can subset an SVAR directory by region and
sample list without writing Python.

## Scope

In scope:

- Add `genoray-cli` as a git submodule of the `genoray` repo for coupled local
  development.
- Wire the pixi env to install the cli editably from the submodule path so
  changes are reflected without reinstalling.
- Add a `view` subcommand to `genoray_cli/__main__.py` that is a thin wrapper
  over `SparseVar.write_view`.
- Enforce the no-op guard at the CLI layer: at least one of regions or samples
  must be supplied.
- **`write_view` enhancement (this branch):** drop output variants whose
  minor allele count across the kept samples is zero. The sparse format has
  nothing to store for them; leaving them in the index is dead weight and a
  leaky abstraction.
- Add a CLI smoke test in `genoray-cli` that subprocess-invokes `genoray view`
  on a tiny SVAR fixture and verifies the output is a valid `SparseVar`.

Out of scope:

- Any other new CLI commands.
- Refactoring of the existing `index` / `write` cli subcommands.
- Enforcing the MAC>0 invariant **format-wide** in `dense2sparse`, `from_vcf`,
  `from_pgen`, or as a standalone validator. The aspiration is that every
  variant in an SVAR has MAC>0, but the wider enforcement is a separate PR
  train tracked under follow-up issue *TBD-link* (open after this branch).

## Architecture

### Submodule layout

```
genoray/                                  (this repo)
├── genoray/                              library source
├── genoray-cli/                          NEW: git submodule → d-laub/genoray-cli @ main
│   └── genoray_cli/
│       ├── __main__.py                   add `view` command here
│       └── ...
├── pixi.toml                             updated [pypi-dependencies]
└── pyproject.toml                        [cli] extra unchanged in spirit
```

### Dependency wiring

- `pyproject.toml`: keep `cli = ["genoray-cli>=0.2.0"]` (will be bumped to
  `>=0.3.0` after the cli release that includes `view`). This is what
  downstream pip users see.
- `pixi.toml` `[pypi-dependencies]`: replace any implicit cli sourcing with an
  editable path entry pointing at the submodule, e.g.
  ```toml
  genoray-cli = { path = "./genoray-cli", editable = true }
  ```
  This keeps the genoray editable install present, plus uses the local cli
  during dev.
- `.gitmodules` records the submodule pinned to a commit on `main`. The pinned
  commit is bumped manually after the cli `view` PR merges.

### `view` command surface (in genoray-cli)

Adopts **bcftools-style flag pairs** for region/sample selection so the UX is
familiar to the target audience, and so we cleanly separate inline values from
file paths without leaning on heuristic type sniffing.

```
genoray view SOURCE OUT
  ( -r/--regions      "chr:beg-end[,chr:beg-end ...]"
  | -R/--regions-file PATH                              )   [required, mutually exclusive]
  ( -s/--samples      "name[,name ...]"
  | -S/--samples-file PATH                              )   [required, mutually exclusive]
  [-f/--fields F ...]
  [--merge-overlapping]
  [--regions-overlap pos|record|variant]
  [--overwrite]
  [-@/--threads N]
```

Signature sketch in `genoray_cli/__main__.py`:

```python
@app.command
def view(
    source: Path,
    out: Path,
    *,
    regions: str | None = None,           # -r, comma-list, 1-based inclusive
    regions_file: Path | None = None,     # -R, BED (0-based half-open per _normalize_regions)
    samples: str | None = None,           # -s, comma-list
    samples_file: Path | None = None,     # -S, newline-delimited
    fields: list[str] | None = None,      # -f
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
    overwrite: bool = False,
    threads: int | None = None,           # -@
) -> None:
    ...
```

Mutual exclusion is enforced by a cyclopts validator that raises if both or
neither of each pair is set.

Translation to `write_view` arguments:

| CLI input                              | Value passed to `write_view`                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `-r "chr:1-100"` (single entry)        | the string itself — `_normalize_regions` parses 1-based inclusive → 0-based half-open                  |
| `-r "a:1-2,b:3-4"` (n>1)               | parse each entry with the same 1-based→0-based convention, build a `pl.DataFrame(chrom,start,end)`     |
| `-R path.bed`                          | `Path(path.bed)`                                                                                        |
| `-s "A,B"`                             | `["A", "B"]`                                                                                            |
| `-S path.txt`                          | `Path(path.txt)`                                                                                        |

The comma-list parser lives in `genoray_cli/__main__.py` (or a small helper
module if it grows). It reuses the regex from `_normalize_regions` if
exported, otherwise duplicates the simple `chrom:start-end` pattern — minor
duplication is fine; the canonical normalization still happens inside
`write_view` once we pass the DataFrame.

Help text adapts the docstring from `SparseVar.write_view`.

### No-op guard and "all" defaults

The `view` command requires the user to express *some* selection. Because an
SVAR has no human-readable representation, an unfiltered `view` is just a
slow copy — refuse it.

| User provides           | CLI behavior                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| regions only            | synthesize "all samples" = `sv.available_samples`, pass to `write_view`                                                   |
| samples only            | synthesize "all variants" = one row per contig from `sv.index` (`start=0`, `end=max(POS)+1`), build a `pl.DataFrame`      |
| neither                 | **error**: *"at least one of --regions/--regions-file or --samples/--samples-file is required"*                           |
| both                    | passthrough (already designed above)                                                                                      |
| `-f`/`--fields` alone   | still errors — fields-only subsetting is not what `view` is for                                                            |

This keeps `write_view`'s Python contract unchanged (regions and samples
remain required); the CLI is the only layer that synthesizes "all".

### `write_view` change: drop MAC=0 variants in output

After resolving `kept_var_idxs` and `caller_samples`, `write_view` will:

1. Run the existing pass-1 count kernel (`_nb_count_kept`).
2. Compute per-variant MAC across kept samples from that count (it's already
   counting non-ref entries per (sample, ploidy) slot — sum across the kept
   sample-axis dimension).
3. Drop any variants where MAC == 0 from `kept_var_idxs` *before* allocating
   output buffers or running pass 2.
4. Log/warn the number of dropped variants when > 0.
5. The output index reflects only retained variants.

Implementation notes:

- This is at most an extra reduction over an array already produced in pass
  1; no extra read pass needed.
- Edge case: if *all* kept variants are dropped, raise the same
  `"no variants selected by regions"`-style error rather than writing an
  empty SVAR. Reuse the existing error path.
- Tests: extend `tests/test_svar_write_view.py` with a case where a
  sample-subset zeros out the MAC of some variants and assert they don't
  appear in the output index.
- The `write_view` docstring updates to document the MAC>0 guarantee on
  outputs.

### Test plan

`genoray-cli/tests/test_view_cli.py`:

- Build a small SVAR via the existing `genoray write` (or a fixture from the
  genoray test corpus) in a `tmp_path`.
- Invoke `subprocess.run([sys.executable, "-m", "genoray_cli", "view", ...])`
  with a `chrom:start-end` regions arg and a single sample, writing to a
  second `tmp_path`.
- Assert the output dir exists, opens as a `SparseVar`, has the expected
  sample list and a non-empty variant count.
- One edge case: passing a BED file path for `-R` and a newline-file for
  `-S`, to confirm the path flags reach `_normalize_*`.
- No-op guard: invoking `genoray view src out` with no `-r/-R/-s/-S` exits
  non-zero with the expected error message.
- "All samples" synthesis: `-r chr1:1-100` alone produces an output whose
  sample list matches `sv.available_samples`.
- "All variants" synthesis: `-s name` alone produces an output covering
  every contig in the source.

Genoray-side tests (`tests/test_svar_write_view.py`):

- MAC>0 drop: construct a small SVAR where some variants are non-ref only in
  samples `X, Y`; call `write_view` keeping samples `Z, W`; assert the
  dropped variants do not appear in the output index and the kept ones do.
- "Everything drops": `write_view` against a sample-subset that zeros out
  every variant raises the same error as "no variants selected".

## Release sequencing

`feat/svar-write-subset` is already merged on `main`; the `write_view`
function ships in the next `genoray` release.

1. **genoray (this branch, feat/svar-view-cli):**
   - Add the `write_view` MAC>0 enhancement (genoray-side commits in
     `genoray/_svar.py` + tests).
   - Add `genoray-cli` as a submodule and wire `pixi.toml` to install it
     editably.
   - Merge → release `genoray` minor bump (the new `write_view` semantics
     are an output-shape change; bump minor, e.g. `2.5.0`).
2. **genoray-cli:** merge the `view` command branch in the cli repo →
   release `genoray-cli 0.3.0` with `dependencies = ["genoray>=2.5.0",
   "cyclopts"]`.
3. **genoray (follow-up commit):**
   - Bump `[cli]` extra floor to `genoray-cli>=0.3.0`.
   - Update the submodule pointer to the released cli commit/tag.

The submodule + pixi wiring on `feat/svar-view-cli` can be developed in
parallel with the cli `view` command (the submodule pointer can track the
cli's working branch during dev, then move to the released tag at step 3).

## Risks / open questions

- **Submodule + PyPI dep coexistence:** When end users `pip install
  genoray[cli]`, they get the PyPI cli, not the submodule. The submodule is
  purely for repo-local development. We must avoid drift between submodule
  HEAD and the published version — keep the pinned commit on `main` of the
  cli repo and bump the `>=` floor on every cli release.
- **CI:** GitHub Actions checkouts must use `submodules: true` (or
  `recursive`) in any workflow that needs the cli installed. To be confirmed
  during plan execution.
- **`feat/svar-write-subset` not yet on `main`:** Reviewers of this branch
  will see commits from that branch in the diff. Mitigation: open the PR
  against `feat/svar-write-subset` (stacked PR), then rebase onto `main`
  after the parent merges.

## Non-goals

- A general "CLI for every `SparseVar` method" — only `view` for now.
- Progress bars, structured logging, or fancy error formatting beyond what
  `cyclopts` provides by default.
- Backwards compatibility shims; this is additive.
- **bcftools `^` exclusion prefix** for `-s/-S` (e.g. `-s ^A,B` = "all
  samples except A and B"). `write_view` doesn't support exclusion; doing
  this at the CLI layer is a one-liner against `sv.available_samples`, but
  it deserves its own design pass (semantics of the exclusion + sample
  ordering rules), so it's deferred.

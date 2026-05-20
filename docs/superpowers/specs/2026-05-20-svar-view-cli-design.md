# `genoray view` CLI for `SparseVar.write_view`

**Date:** 2026-05-20
**Branch:** `feat/svar-view-cli` (cut from `feat/svar-write-subset`)
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
- Add a CLI smoke test in `genoray-cli` that subprocess-invokes `genoray view`
  on a tiny SVAR fixture and verifies the output is a valid `SparseVar`.

Out of scope:

- Changes to `SparseVar.write_view` itself (already covered by the prior spec).
- Any other new CLI commands.
- Refactoring of the existing `index` / `write` cli subcommands.

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

Signature in `genoray_cli/__main__.py`:

```python
@app.command
def view(
    source: Path,
    out: Path,
    regions: Path | str,
    samples: Path | str,
    fields: list[str] | None = None,
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
    overwrite: bool = False,
    threads: int | None = None,
) -> None:
    """Write a subset of an SVAR to a new SVAR directory."""
```

Behavior:

1. Open `SparseVar(source)`.
2. Forward all arguments verbatim to `sv.write_view(...)`.
3. No additional argument parsing — `write_view` already accepts strings, BED
   paths, and sample files via `_normalize_regions` / `_normalize_samples`.

`source` is the input SVAR directory (positional). `out` is the destination
SVAR directory (positional). All other arguments are options matching the
Python method's defaults.

Cyclopts will derive `--regions`, `--samples`, `--fields`, `--merge-overlapping`
(boolean flag), `--regions-overlap`, `--overwrite`, `--threads` from the
signature.

Help text adapts the docstring from `SparseVar.write_view`.

### Test plan

`genoray-cli/tests/test_view_cli.py`:

- Build a small SVAR via the existing `genoray write` (or a fixture from the
  genoray test corpus) in a `tmp_path`.
- Invoke `subprocess.run([sys.executable, "-m", "genoray_cli", "view", ...])`
  with a `chrom:start-end` regions arg and a single sample, writing to a
  second `tmp_path`.
- Assert the output dir exists, opens as a `SparseVar`, has the expected
  sample list and a non-empty variant count.
- One edge case: passing a BED file path for `--regions` and a newline-file
  for `--samples`, to confirm the CLI string-vs-path dispatch reaches
  `_normalize_*`.

No new genoray-side tests; `write_view` correctness is covered by
`tests/test_svar_write_view.py`.

## Release sequencing

1. **genoray:** merge `feat/svar-write-subset` (incl. `write_view`) → release
   `genoray 2.5.0` (next minor; exact version per cz bump).
2. **genoray-cli:** merge the `view` command branch → release
   `genoray-cli 0.3.0` with `dependencies = ["genoray>=2.5.0", "cyclopts"]`.
3. **genoray (this branch, feat/svar-view-cli):**
   - Bump `[cli]` extra floor to `genoray-cli>=0.3.0`.
   - Update the submodule pointer to the released cli commit/tag.
   - Merge.

Steps 1 and 2 must precede step 3's final merge, but the submodule + pixi
wiring on `feat/svar-view-cli` can be authored and reviewed in parallel —
they don't require an external release to land locally.

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

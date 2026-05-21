# Merge `genoray-cli` into `genoray` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Absorb the `genoray-cli` source into the main `genoray` package, eliminating the submodule + separate-PyPI-package split, while preserving sub-second `genoray --help` via a lazy `genoray/__init__.py`.

**Architecture:**

1. Make `genoray/__init__.py` lazy (PEP 562 `__getattr__`) so `import genoray` is cheap. Guard with a regression test.
2. Move cli source into `genoray/_cli/`, cli tests into `tests/cli/`. Wire up the `genoray` script entry point in `genoray/pyproject.toml`.
3. Tear down the git submodule + pixi/CI submodule wiring previously added to `feat/svar-view-cli`.
4. Close the in-flight cli PR and archive the cli repo.

**Tech Stack:** Python ≥ 3.10, cyclopts, polars (already deps), pytest, pixi, git submodules (for removal).

---

## File Structure

**Create:**

- `genoray/_cli/__init__.py` — empty marker.
- `genoray/_cli/__main__.py` — cyclopts `App` with `index`, `write`, `view` commands. Was `genoray-cli/genoray_cli/__main__.py`.
- `genoray/_cli/_view_helpers.py` — `parse_regions_arg`. Was `genoray-cli/genoray_cli/_view_helpers.py`.
- `tests/cli/__init__.py` — empty.
- `tests/cli/conftest.py` — was `genoray-cli/tests/conftest.py`.
- `tests/cli/test_view_cli.py` — was `genoray-cli/tests/test_view_cli.py`.
- `tests/cli/test_view_helpers.py` — was `genoray-cli/tests/test_view_helpers.py`.
- `tests/test_lazy_init.py` — regression test: `import genoray` does not pull in heavy deps.

**Modify:**

- `genoray/__init__.py` — eager imports → PEP 562 `__getattr__` lazy resolution.
- `genoray/_pgen.py` — move the `polars_bio` logger config (currently in `__init__.py`) here, near where the dep is first used.
- `pyproject.toml` — add `[project.scripts]` for `genoray`; remove `[project.optional-dependencies] cli` block.
- `pixi.toml` — remove `genoray-cli = { path = "./genoray-cli", editable = true }`; restore `extras = ["cli"]` removal (the cli is now shipped by `genoray` itself, so the genoray editable install is enough).
- `.github/workflows/test.yaml`, `.github/workflows/release.yaml`, `.github/workflows/prek.yaml` — remove `submodules: recursive` from checkout steps (no submodules anymore).

**Delete:**

- `.gitmodules` (file removed).
- `genoray-cli/` (the submodule directory + worktree).
- `.git/modules/genoray-cli/` (the submodule's separate gitdir).

**External (manual, end of plan):**

- Close PR https://github.com/d-laub/genoray-cli/pull/2 unmerged.
- Update `d-laub/genoray-cli` README to point at `pip install genoray`; archive that repo.

---

## Phase A — Lazy `genoray/__init__.py`

### Task 1: Regression test that `import genoray` doesn't load heavy deps

**Files:**

- Create: `tests/test_lazy_init.py`

- [ ] **Step 1: Write the failing test**

```python
"""Regression: importing genoray must not pull in heavy optional deps.

Heavy deps are loaded lazily on first attribute access (PEP 562 __getattr__ in
__init__.py). Verified in a subprocess so that test-collection imports don't
pollute the result.
"""
from __future__ import annotations

import subprocess
import sys


def test_import_genoray_does_not_load_heavy_deps():
    code = (
        "import sys\n"
        "import genoray\n"
        "heavy = {'cyvcf2', 'numba', 'polars_bio', 'pgenlib'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, f'genoray import pulled in heavy modules: {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_lazy_init.py -v`

Expected: FAIL — current `genoray/__init__.py` eagerly imports `_pgen`/`_svar`/`_vcf`/`exprs`, which pulls in cyvcf2, numba, polars_bio, pgenlib. The assertion lists the offending modules.

- [ ] **Step 3: Do NOT commit yet** — test is red. Task 2 turns it green.

---

### Task 2: Make `genoray/__init__.py` lazy

**Files:**

- Modify: `genoray/__init__.py` (replace contents)
- Modify: `genoray/_pgen.py` (relocate `polars_bio` logger config)

- [ ] **Step 1: Read the current `_pgen.py` top-of-file imports**

Run: `head -30 /Users/david/projects/genoray/genoray/_pgen.py`

Confirm whether `polars_bio` is already imported there or if it's only imported deeper. Note the location for the logger move.

- [ ] **Step 2: Move the polars_bio logger config out of `__init__.py`**

Add this block to `genoray/_pgen.py` at the top, immediately after the `polars_bio` import (or add the import + block if polars_bio isn't imported at top-level there). Use the canonical place where `polars_bio` is first imported in the codebase:

```python
import logging as _logging

_logging.getLogger("polars_bio").setLevel(_logging.ERROR)
```

(If `polars_bio` is imported inside functions only, add the same block at the top of `_pgen.py` outside any function — module load triggers it. As long as it runs before polars-bio emits its first log line, position doesn't matter.)

- [ ] **Step 3: Replace `genoray/__init__.py` with the lazy version**

```python
"""genoray package — lazy public API.

Heavy modules (`_pgen`, `_svar`, `_vcf`, `exprs`) are loaded on first access
via PEP 562 ``__getattr__``. This keeps ``import genoray`` (and therefore
``genoray --help``) sub-second.
"""
from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("genoray")

__all__ = ["PGEN", "VCF", "Reader", "SparseVar", "exprs"]

# Public name -> (module path, attribute name | None for the module itself).
_LAZY: dict[str, tuple[str, str | None]] = {
    "PGEN": ("genoray._pgen", "PGEN"),
    "VCF": ("genoray._vcf", "VCF"),
    "SparseVar": ("genoray._svar", "SparseVar"),
    "exprs": ("genoray.exprs", None),
}


def __getattr__(name: str):
    if name == "Reader":
        from ._pgen import PGEN
        from ._svar import SparseVar
        from ._vcf import VCF

        result = VCF | PGEN | SparseVar
        globals()[name] = result
        return result
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        result = mod if attr is None else getattr(mod, attr)
        globals()[name] = result
        return result
    raise AttributeError(f"module 'genoray' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from . import exprs as exprs
    from ._pgen import PGEN as PGEN
    from ._svar import SparseVar as SparseVar
    from ._vcf import VCF as VCF

    Reader = VCF | PGEN | SparseVar
```

- [ ] **Step 4: Run the lazy-init regression test**

Run: `pixi run pytest tests/test_lazy_init.py -v`

Expected: PASS — `import genoray` no longer eagerly pulls in cyvcf2/numba/polars_bio/pgenlib.

- [ ] **Step 5: Run the full existing test suite to confirm no regressions**

Run: `pixi run pytest -x`

Expected: all tests pass. If anything breaks because user code did `from genoray import <X>` and relied on `<X>` being a real attribute pre-set: PEP 562 `__getattr__` handles `from genoray import X` correctly, so this should not fail. But verify.

- [ ] **Step 6: Commit**

```bash
git add genoray/__init__.py genoray/_pgen.py tests/test_lazy_init.py
git commit -m "perf(init): make genoray.__init__ lazy via PEP 562 __getattr__"
```

---

## Phase B — Move the cli source

### Task 3: Copy cli source into `genoray/_cli/`

**Files:**

- Create: `genoray/_cli/__init__.py` (empty)
- Create: `genoray/_cli/__main__.py` (from submodule's `genoray-cli/genoray_cli/__main__.py`)
- Create: `genoray/_cli/_view_helpers.py` (from submodule's `genoray-cli/genoray_cli/_view_helpers.py`)

- [ ] **Step 1: Create the package directory**

```bash
cd /Users/david/projects/genoray
mkdir -p genoray/_cli
: > genoray/_cli/__init__.py
```

- [ ] **Step 2: Copy `__main__.py` from the submodule**

```bash
cp genoray-cli/genoray_cli/__main__.py genoray/_cli/__main__.py
```

- [ ] **Step 3: Copy `_view_helpers.py` from the submodule**

```bash
cp genoray-cli/genoray_cli/_view_helpers.py genoray/_cli/_view_helpers.py
```

- [ ] **Step 4: Fix imports in `__main__.py`**

Open `genoray/_cli/__main__.py`. The current cli imports look like:

```python
from importlib.metadata import version
# ...
from cyclopts import App, Parameter
# ...
from ._view_helpers import parse_regions_arg
```

The `from ._view_helpers import parse_regions_arg` line continues to work since `_view_helpers.py` is a sibling of `__main__.py`. Good.

The `version()` call currently reads `version('genoray-cli')`. Since the cli is now part of `genoray`, change the version display string:

Find:
```python
app = App(
    help_on_error=True,
    version=f"[magenta]genoray[/magenta] {version('genoray')}\n[cyan]genoray-cli[/cyan] {version('genoray-cli')}",
    version_format="rich",
    help="Tools for genoray, including SVAR files.",
)
```

Replace with:
```python
app = App(
    help_on_error=True,
    version=f"[magenta]genoray[/magenta] {version('genoray')}",
    version_format="rich",
    help="Tools for genoray, including SVAR files.",
)
```

- [ ] **Step 5: Sanity-check the import path resolves**

Run: `pixi run python -c "from genoray._cli.__main__ import app; print(app)"`

Expected: prints the cyclopts App object without error. (Note: this also triggers lazy genoray imports — that's fine for sanity.)

- [ ] **Step 6: Commit**

```bash
git add genoray/_cli/
git commit -m "feat(cli): move cli source into genoray._cli"
```

---

### Task 4: Move cli tests into `tests/cli/`

**Files:**

- Create: `tests/cli/__init__.py` (empty)
- Create: `tests/cli/conftest.py` (from `genoray-cli/tests/conftest.py`)
- Create: `tests/cli/test_view_cli.py` (from `genoray-cli/tests/test_view_cli.py`)
- Create: `tests/cli/test_view_helpers.py` (from `genoray-cli/tests/test_view_helpers.py`)

- [ ] **Step 1: Create the package directory + copy files**

```bash
cd /Users/david/projects/genoray
mkdir -p tests/cli
: > tests/cli/__init__.py
cp genoray-cli/tests/conftest.py        tests/cli/conftest.py
cp genoray-cli/tests/test_view_cli.py   tests/cli/test_view_cli.py
cp genoray-cli/tests/test_view_helpers.py tests/cli/test_view_helpers.py
```

- [ ] **Step 2: Fix imports in `tests/cli/test_view_helpers.py`**

The cli's helper module is now `genoray._cli._view_helpers`. Open `tests/cli/test_view_helpers.py` and change:

```python
from genoray_cli._view_helpers import parse_regions_arg
```

to:

```python
from genoray._cli._view_helpers import parse_regions_arg
```

- [ ] **Step 3: Fix subprocess entry in `tests/cli/test_view_cli.py`**

The CLI is invoked via `python -m <module>`. The old `genoray_cli` module becomes `genoray._cli`. Find:

```python
def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "genoray_cli", *argv],
```

Change to:

```python
def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "genoray._cli", *argv],
```

- [ ] **Step 4: Run the cli tests**

Run: `pixi run pytest tests/cli -v`

Expected: 9 passed (3 helper tests + 6 CLI tests). If `pixi run` fails to resolve because the cli's `genoray>=2.5.0` pin is still in effect, you may need to first proceed to Task 7 (which removes the submodule's editable install) — but at this point, the test files are in place and the genoray-side install is enough. Try the test run first; if it fails on env resolve, do a `pixi install --no-update-lockfile` first (or proceed to Task 7 and come back).

- [ ] **Step 5: Commit**

```bash
git add tests/cli/
git commit -m "test(cli): move cli tests into tests/cli"
```

---

### Task 5: Wire up `genoray` script entry point + drop `[cli]` extra

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1: Open `pyproject.toml` and locate the `[project.optional-dependencies]` block**

It currently contains:

```toml
[project.optional-dependencies]
cli = ["genoray-cli>=0.2.0"]
```

- [ ] **Step 2: Delete the `[project.optional-dependencies]` block**

Remove the section entirely. The cli now ships with `genoray`, so there's no separate extra. Also confirm that `cyclopts` is already in `[project] dependencies` — it is, from previous work.

- [ ] **Step 3: Add `[project.scripts]` block**

Add (anywhere in the `[project]` neighborhood, conventionally near `[project.urls]` if present, otherwise after `[project.optional-dependencies]`'s former position):

```toml
[project.scripts]
genoray = "genoray._cli.__main__:app"
```

- [ ] **Step 4: Verify the cli entry point installs and runs**

```bash
cd /Users/david/projects/genoray
pixi install
pixi run genoray --help
```

Expected: pixi resolves cleanly (no `>=2.5.0` failure because the submodule's pyproject is no longer in the resolver after Task 7 — but we haven't done Task 7 yet, so this may fail). If it fails on resolution, defer the `pixi run genoray --help` check to after Task 7 and continue with the commit:

If the install succeeds, `genoray --help` should print the top-level command list (`index`, `write`, `view`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat(pkg): expose genoray script; drop [cli] extra"
```

---

## Phase C — Tear down the submodule

### Task 6: Remove the editable submodule install from `pixi.toml`

**Files:**

- Modify: `pixi.toml`

- [ ] **Step 1: Open `pixi.toml` and locate `[pypi-dependencies]`**

Currently:

```toml
[pypi-dependencies]
genoray = { path = ".", editable = true }
genoray-cli = { path = "./genoray-cli", editable = true }
seqpro = { git = "https://github.com/ml4gland/seqpro.git", rev = "main" }
# dev deps
seaborn = "*"
pooch = "*"
```

- [ ] **Step 2: Delete the `genoray-cli` line**

After:

```toml
[pypi-dependencies]
genoray = { path = ".", editable = true }
seqpro = { git = "https://github.com/ml4gland/seqpro.git", rev = "main" }
# dev deps
seaborn = "*"
pooch = "*"
```

- [ ] **Step 3: Re-resolve and smoke test**

```bash
pixi install
pixi run genoray --help
pixi run pytest tests/cli -v
```

Expected: pixi resolves cleanly; cli `--help` works; 9 cli tests pass.

- [ ] **Step 4: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "chore: drop genoray-cli editable install (now part of genoray)"
```

---

### Task 7: Delete the submodule

**Files:**

- Delete: `genoray-cli/` directory.
- Delete: `.gitmodules` file.
- Delete: `.git/modules/genoray-cli/` directory.

- [ ] **Step 1: Run the standard 4-step submodule removal**

```bash
cd /Users/david/projects/genoray

# 1. Deinit (clean up .git/config entries)
git submodule deinit -f genoray-cli

# 2. Remove from git index and working tree
git rm -f genoray-cli

# 3. Remove the .git/modules directory
rm -rf .git/modules/genoray-cli

# 4. .gitmodules: `git rm` above edits it. If it ends up empty, delete it.
test -s .gitmodules && cat .gitmodules || { rm -f .gitmodules; git rm --cached .gitmodules 2>/dev/null || true; }
```

- [ ] **Step 2: Verify clean state**

Run: `git status`

Expected: `deleted: .gitmodules` (if it became empty), `deleted: genoray-cli`. No other unexpected changes.

Run: `ls -la genoray-cli` → should fail (directory gone).

- [ ] **Step 3: Verify the genoray test suite still runs end-to-end**

Run: `pixi run pytest -x`

Expected: all tests pass, including `tests/cli/` and `tests/test_lazy_init.py`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove genoray-cli submodule (merged into genoray)"
```

---

### Task 8: Undo `submodules: recursive` in CI workflows

**Files:**

- Modify: `.github/workflows/test.yaml`
- Modify: `.github/workflows/release.yaml`
- Modify: `.github/workflows/prek.yaml`

- [ ] **Step 1: Inventory the `submodules: recursive` occurrences**

Run: `grep -rn "submodules: recursive" .github/workflows/`

There should be 5 hits across the 3 files (matching commit `f65664d`).

- [ ] **Step 2: Edit each file to remove `submodules: recursive`**

For each `actions/checkout@vN` step that has:

```yaml
      - uses: actions/checkout@v4
        with:
          submodules: recursive
```

Change to:

```yaml
      - uses: actions/checkout@v4
```

If the `with:` block had only `submodules: recursive`, remove the entire `with:` block. If it had other entries (none currently, but check), leave them.

- [ ] **Step 3: Verify no remaining hits**

Run: `grep -rn "submodules" .github/workflows/`

Expected: zero hits.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/
git commit -m "ci: drop submodule checkout (no longer used)"
```

---

## Phase D — External actions (manual, no commits)

These are not git-tracked steps; complete them once the branch is merged or as part of the release process. List them as checkboxes for completion tracking but do not commit anything.

### Task 9: Close cli PR + archive cli repo

- [ ] **Step 1: Close PR #2 on `d-laub/genoray-cli`**

Run: `gh pr close 2 --repo d-laub/genoray-cli --comment "Superseded — cli source merged into genoray; see https://github.com/d-laub/genoray (branch feat/svar-view-cli)."`

- [ ] **Step 2: Update `d-laub/genoray-cli` README**

In the `d-laub/genoray-cli` repo (on `main`), replace `README.md` with a 3-line notice:

```markdown
# genoray-cli (archived)

The `genoray` command-line interface has moved into the main
[genoray](https://github.com/d-laub/genoray) package. Install with `pip install genoray`.
```

Commit + push.

- [ ] **Step 3: Archive the repo**

Either via the GitHub web UI (Settings → Archive this repository) or:

```bash
gh repo archive d-laub/genoray-cli --yes
```

---

## Self-Review

**1. Spec coverage:**

- Lazy `__init__.py` via PEP 562 ✓ (Task 2).
- Move cli source to `genoray/_cli/` ✓ (Task 3).
- Move cli tests to `tests/cli/` ✓ (Task 4).
- `[project.scripts]` entry ✓ (Task 5).
- Remove `[cli]` extra ✓ (Task 5).
- Regression test that `import genoray` is light ✓ (Task 1).
- Delete submodule + pixi wiring + CI checkout ✓ (Tasks 6, 7, 8).
- Close PR #2 ✓ (Task 9).
- Archive cli repo ✓ (Task 9).
- Move polars_bio logger config out of `__init__.py` ✓ (Task 2 Step 2).

**2. Placeholder scan:** No "TBD", "implement later", etc. Every code-change step has the actual code shown. The polars_bio logger relocation has a small "as long as it runs before polars-bio emits its first log line" caveat — that's a concrete reasoning aid, not a placeholder.

**3. Type consistency:** Module names used consistently — `genoray._cli`, `genoray._cli._view_helpers`, `genoray._cli.__main__:app`. Subprocess invocation `python -m genoray._cli` matches the entry point's module address. Test imports use `from genoray._cli._view_helpers import parse_regions_arg`.

**4. Order dependency:** Task 4's pytest run may need Task 7's submodule removal first (pixi resolution issue). Documented as an inline note in Task 4 Step 4 with explicit workaround. Tasks 5 → 6 → 7 → 8 are the strict completion path.

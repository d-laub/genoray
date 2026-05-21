# Merge `genoray-cli` into `genoray`

**Date:** 2026-05-20
**Branch:** `feat/svar-view-cli` (continued — drop the submodule commits, add this work)

## Goal

Eliminate the `genoray-cli` repository by absorbing its source into `genoray`. The split exists solely to keep `genoray <subcommand> --help` fast (sub-second). The same speed can be preserved without a separate package by making `genoray/__init__.py` lazy, which is the standard pattern.

## Motivation

The current split costs:

- Two repos to maintain (one as a git submodule of the other).
- Coupled version pinning (`genoray-cli` declares `genoray>=X`; bumping either requires coordinated work). We hit the resulting pixi resolution friction during the `feat/svar-view-cli` execution.
- Dual PR flow (`genoray-cli#2` separate from the `genoray` branch carrying everything else).
- The cli must use only the public `genoray` API. A subpackage can use private modules freely.

The benefit the split delivers — fast `--help` — is structural to *how* `genoray/__init__.py` is written, not to package boundaries.

## Scope

In scope:

- Convert `genoray/__init__.py` to lazy attribute resolution via PEP 562 `__getattr__`. `PGEN`, `VCF`, `SparseVar`, `exprs`, `Reader`, `__version__` continue to be reachable as `genoray.X` for users, but the heavy submodules import on first access, not at package import.
- Move the cli source from `genoray-cli/genoray_cli/` into `genoray/_cli/` (private subpackage of `genoray`).
- Move the cli tests into `tests/cli/` of the `genoray` repo.
- Add a `[project.scripts]` entry to `genoray/pyproject.toml` so `pip install genoray` exposes the `genoray` command.
- Remove the `[cli]` extra from `genoray/pyproject.toml`.
- Add a regression test that asserts `import genoray` does not import `cyvcf2`, `numba`, `polars_bio`, or any other slow dep — so a future eager import in `__init__` can't silently re-slow `--help`.
- Delete the git submodule + the pixi wiring + the CI submodule-checkout changes that were added on this branch.
- Close PR https://github.com/d-laub/genoray-cli/pull/2 unmerged.
- Stop publishing the `genoray-cli` PyPI package. `0.2.0` remains as the final release on the index; the README of the `d-laub/genoray-cli` repo notes that the cli was merged into `genoray` and points users at `pip install genoray`. No PyPI shim.

Out of scope:

- Behavior changes to the `view` / `write` / `index` commands themselves. The MAC>0 work from earlier on this branch stays.
- Renaming the entry point. It stays `genoray`.
- Removing `cyclopts` / `polars` from the genoray dependency list — both are already deps.

## Architecture

### Directory layout (after)

```
genoray/
├── genoray/
│   ├── __init__.py         (lazy: PEP 562 __getattr__)
│   ├── _cli/               (NEW — was genoray-cli/genoray_cli/)
│   │   ├── __init__.py
│   │   ├── __main__.py     (the cyclopts App)
│   │   └── _view_helpers.py
│   ├── _pgen.py
│   ├── _svar.py
│   ├── _vcf.py
│   ├── exprs.py
│   └── ...
├── tests/
│   ├── cli/                (NEW — was genoray-cli/tests/)
│   │   ├── conftest.py
│   │   ├── test_view_cli.py
│   │   └── test_view_helpers.py
│   ├── test_lazy_init.py   (NEW — regression guard)
│   └── ... (existing tests)
├── pyproject.toml          (scripts entry; drop [cli] extra)
└── pixi.toml               (no submodule wiring)
```

### Lazy `genoray/__init__.py`

Replace the current eager imports with PEP 562 `__getattr__`:

```python
from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("genoray")

__all__ = ["PGEN", "VCF", "Reader", "SparseVar", "exprs"]

# Mapping of public name → (module_path, attr_name) for lazy resolution.
_LAZY = {
    "PGEN": ("genoray._pgen", "PGEN"),
    "VCF": ("genoray._vcf", "VCF"),
    "SparseVar": ("genoray._svar", "SparseVar"),
    "exprs": ("genoray.exprs", None),  # whole module
}


def __getattr__(name: str):
    if name == "Reader":
        # Union type — resolve all three lazily.
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


if TYPE_CHECKING:
    from . import exprs as exprs
    from ._pgen import PGEN as PGEN
    from ._svar import SparseVar as SparseVar
    from ._vcf import VCF as VCF
    Reader = VCF | PGEN | SparseVar
```

The polars-bio logger config (`logger.setLevel(logging.ERROR)`) needs to move somewhere it doesn't force an import. Two options, picked at implementation: (a) move it into `_pgen.py` (or wherever polars-bio is first used), or (b) lazy-trigger it inside `__getattr__` only when one of the lazy attrs is resolved. Option (a) is cleaner.

### Cli entry point

In `genoray/pyproject.toml`:

```toml
[project.scripts]
genoray = "genoray._cli.__main__:app"
```

`genoray/_cli/__main__.py` contains the cyclopts `App` and the command definitions. The deferred imports inside each command (`from genoray import SparseVar` etc.) trigger the lazy `__getattr__` when a real command runs — `--help` only imports cyclopts.

### Regression test

`tests/test_lazy_init.py`:

```python
import subprocess
import sys


def test_import_genoray_does_not_load_heavy_deps():
    code = (
        "import sys; import genoray; "
        "heavy = {'cyvcf2', 'numba', 'polars_bio', 'pgenlib'}; "
        "loaded = heavy & set(sys.modules); "
        "assert not loaded, f'genoray import pulled in: {loaded}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

Subprocess because in-process imports persist across tests.

## Migration steps (summary, full sequencing in the plan)

1. Close `genoray-cli#2`.
2. On `feat/svar-view-cli`, drop the three commits that added the submodule and pixi/CI wiring (`git revert` or interactive rebase — to be picked in the plan).
3. Move cli code into `genoray/_cli/`. Drop `genoray_cli` namespace.
4. Move cli tests into `tests/cli/`. Adjust imports.
5. Lazy `__init__.py`. Add regression test.
6. Update `pyproject.toml` (scripts entry, drop `[cli]` extra).
7. Update `pixi.toml` (no submodule reference; ensure cli tests run in the default env).
8. Verify `genoray --help` is sub-second; full test suite passes.
9. On the `d-laub/genoray-cli` repo: update README to point at `pip install genoray`; archive the repo.

## Risks / open questions

- **Lazy `__init__` regression silently re-slowing `--help`** — mitigated by the regression test. Worth adding a `pytest.mark.benchmark`-style timing assertion too (e.g. `genoray --help` finishes in < 1s)? Probably overkill; the heavy-deps assertion is the proximal check.
- **`Reader = VCF | PGEN | SparseVar`** is used at type-check time and runtime. Resolving it lazily forces a full library load on first access. That's the same cost users currently pay on `import genoray`, so it doesn't regress anyone — it just means "users who use `Reader` once at startup" lose the speed benefit. The cli never references `Reader`, so `--help` stays fast.
- **`__version__` resolution** uses `importlib.metadata.version("genoray")` which is cheap — fine in eager init.
- **IDE / type-checker support** — the `TYPE_CHECKING` block at the bottom of `__init__.py` ensures static analyzers still see the public names. Confirm with `pyright` / `mypy` during implementation.
- **CI** — undo the `submodules: recursive` checkout changes added in T7. Use the original checkout step. (Or leave them; harmless once the submodule is gone, but cleaner to revert.)
- **The submodule directory** has to be removed cleanly. `git rm` + edit `.gitmodules` (delete entry) + edit `.git/config` (clean local submodule config) + delete `.git/modules/genoray-cli/`. Documented in the plan.

## Non-goals

- A PyPI shim package for `genoray-cli`. We deprecate the name outright.
- Changing the cli's command surface (`view`, `write`, `index` keep their flags).
- Migrating users — the entry point name is unchanged (`genoray`), and the only install command change is `pip install genoray[cli]` → `pip install genoray`. Documented in the cli repo's archived README.

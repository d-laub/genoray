# SVAR `view` CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `SparseVar.write_view` as a bcftools-style CLI `genoray view`, via the `genoray-cli` package added as a submodule; and enforce a MAC>0 drop invariant inside `write_view` so sample-subsetting never produces empty variants.

**Architecture:**

1. **genoray (this repo, branch `feat/svar-view-cli`)** — enhance `write_view` to drop output variants with MAC=0 after sample-subsetting; add `genoray-cli` as a submodule at `./genoray-cli/`; wire `pixi.toml` to install it editably.
2. **genoray-cli (submodule, own feature branch)** — add a `view` subcommand using cyclopts with bcftools-style `-r/-R/-s/-S` flag pairs, mutual-exclusion validators, and synthesized "all-samples"/"all-variants" when one side is omitted.

**Tech Stack:** Python ≥ 3.10, numba (kernels), polars (index/regions DataFrame), pytest (tests), cyclopts (CLI), pixi (env), git submodules.

---

## File Structure

**genoray (this repo):**

- Modify: `genoray/_svar.py` — add `_nb_count_mac_per_kept` kernel; insert MAC pre-pass into `write_view`; update its docstring.
- Modify: `tests/test_svar_write_view.py` — add MAC>0 drop tests + "all drop" raise test.
- Create: `.gitmodules` (auto) — submodule entry for `./genoray-cli`.
- Modify: `pixi.toml` — `[pypi-dependencies]` to install cli editably from submodule.
- Modify: `.github/workflows/*.yml` (if any do checkouts that need cli) — set `submodules: true`. Verified during T7.

**genoray-cli (submodule, separate repo `d-laub/genoray-cli`, new branch `feat/view`):**

- Create: `genoray_cli/_view_helpers.py` — `parse_regions_arg(s)`, mutual-exclusion validator.
- Modify: `genoray_cli/__main__.py` — add `@app.command def view(...)`.
- Create: `tests/__init__.py` (empty) and `tests/conftest.py` — build a tiny SVAR fixture.
- Create: `tests/test_view_cli.py` — subprocess-based CLI tests.
- Modify: `pyproject.toml` — bump `genoray-cli` to `0.3.0`, floor `genoray>=2.5.0`.
- Modify: `README.md` — document `view` subcommand.

---

## Phase A — `write_view` MAC>0 drop (genoray)

### Task 1: Failing test — sample-subset zeros some variants

**Files:**

- Modify: `tests/test_svar_write_view.py` — append a new test using the existing `biallelic.vcf.svar` fixture or build a synthetic SVAR via `dense2sparse`.

- [ ] **Step 1: Read the existing fixture options**

Run: `pixi run python -c "from genoray import SparseVar; sv = SparseVar('tests/data/biallelic.vcf.svar'); print(sv.available_samples, sv.n_variants)"`

Expected: prints the sample list (e.g. `['sample1', 'sample2', 'sample3']`) and variant count. Note the sample names for use in the test.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_svar_write_view.py`:

```python
def test_write_view_drops_mac_zero_variants(tmp_path: Path, svar_wv: SparseVar):
    """A variant non-ref only in samples we exclude must not appear in the output."""
    samples_all = list(svar_wv.available_samples)
    # Identify a sample s.t. some variants are non-ref *only* in that sample.
    # Use full-genome region and keep all samples *except* the first to provoke drops
    # if any variants were singletons in samples_all[0].
    keep = samples_all[1:]
    out = tmp_path / "subset.svar"
    svar_wv.write_view(
        regions=(svar_wv.contigs[0], 0, 10_000_000),
        samples=keep,
        output=out,
    )
    sub = SparseVar(out)
    # Every variant in the output must have AF > 0
    assert (sub.index["AF"] > 0).all(), \
        "Output index contains a variant with AF=0 (MAC=0)"
    # Sanity: total variants did not grow.
    assert sub.n_variants <= svar_wv.n_variants
```

- [ ] **Step 3: Run the test, confirm it fails**

Run: `pixi run pytest tests/test_svar_write_view.py::test_write_view_drops_mac_zero_variants -v`

Expected: FAIL — either an assertion failure (`AF=0 present`) **or** PASS if the fixture has no MAC-0-after-subset variants. If it passes spuriously, replace the body with a synthetic fixture: use the snippet below.

If the fixture is too sparse to provoke a MAC=0 drop, replace the test body with this synthetic-SVAR variant (also append at the same place):

```python
def test_write_view_drops_mac_zero_variants(tmp_path: Path):
    """Synthetic SVAR where a singleton in sample 0 must vanish when sample 0 is excluded."""
    from genoray._svar import dense2sparse, SparseVarMetadata, CURRENT_VERSION
    # 3 samples, ploidy 2, 3 variants. Variant 1 is a singleton in sample 0 only.
    genos = np.array(
        [
            [[0, 1, 0], [0, 0, 0]],  # sample 0: het at v0, het at v1 (singleton)
            [[0, 0, 0], [1, 0, 0]],  # sample 1: het at v0
            [[1, 0, 1], [0, 0, 0]],  # sample 2: het at v0, het at v2
        ],
        dtype=np.int8,
    )  # shape (n_samples=3, ploidy=2, n_variants=3)
    # Build source SVAR
    src = tmp_path / "src.svar"
    # (Use dense2sparse + a minimal helper to write metadata/index;
    #  if such a helper does not exist, build via existing test utilities.)
    # ... see tests/data/gen_svar.py for the pattern used to seed fixtures.
    raise NotImplementedError(
        "Replace with the canonical 'build small SVAR' helper used in this test suite"
    )
```

Note for the engineer: the suite already builds SVARs via `tests/data/gen_svar.py` and `gen_from_vcf.sh`. If the natural-fixture version above passes spuriously, copy the helper from `gen_svar.py` to construct a 3-sample fixture with a guaranteed singleton in sample 0 instead of using `raise NotImplementedError`.

- [ ] **Step 4: Do NOT commit yet**

Test is red. Move to Task 2.

---

### Task 2: Add `_nb_count_mac_per_kept` kernel and integrate

**Files:**

- Modify: `genoray/_svar.py` — add kernel + use it in `write_view`.

- [ ] **Step 1: Add the kernel below the existing `_nb_count_kept` (around line 1396)**

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_count_mac_per_kept(
    src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, mac_out
):
    """Count, per kept variant, the number of non-ref entries across (sample, ploidy)
    in the output. Outer prange is over kept variants so each writes its own slot —
    no atomics needed."""
    n_kept = kept_var_idxs.shape[0]
    n_samples = src_sample_idxs.shape[0]
    for k in nb.prange(n_kept):
        v = kept_var_idxs[k]
        count = 0
        for i in range(n_samples):
            s = src_sample_idxs[i]
            for p in range(ploidy):
                src_slot = s * ploidy + p
                lo = src_offsets[src_slot]
                hi = src_offsets[src_slot + 1]
                # binary search for v in src_data[lo:hi]
                idx = np.searchsorted(src_data[lo:hi], v)
                if idx < (hi - lo) and src_data[lo + idx] == v:
                    count += 1
        mac_out[k] = count
```

- [ ] **Step 2: Insert the pre-pass into `write_view`**

In `genoray/_svar.py`, locate the section labeled `# --- 5. Pass 1: count kept entries per output slot ---` (around line 1267). Immediately *before* it, insert:

```python
        # --- 4.5. Pre-pass: drop variants whose MAC across kept samples is 0 ---
        mac_per_kept = np.zeros(len(kept_var_idxs), dtype=np.int64)
        with numba_threads(threads_resolved):
            _nb_count_mac_per_kept(
                self.genos.data,
                self.genos.offsets,
                src_sample_idxs,
                ploidy,
                kept_var_idxs,
                mac_per_kept,
            )
        survive_mask = mac_per_kept > 0
        n_dropped = int((~survive_mask).sum())
        if n_dropped:
            logger.info(
                "write_view: dropping %d variant(s) with MAC=0 in the output sample set",
                n_dropped,
            )
            kept_var_idxs = kept_var_idxs[survive_mask]
        if len(kept_var_idxs) == 0:
            raise ValueError(
                "no variants selected by `regions` (all candidates have MAC=0 in "
                "the chosen sample subset)"
            )
```

Confirm `from loguru import logger` is already imported at the top of `_svar.py`; if not, add it.

- [ ] **Step 3: Run the test, confirm it passes**

Run: `pixi run pytest tests/test_svar_write_view.py::test_write_view_drops_mac_zero_variants -v`

Expected: PASS.

- [ ] **Step 4: Run the existing write_view test suite to confirm no regressions**

Run: `pixi run pytest tests/test_svar_write_view.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat(svar): drop MAC=0 variants from write_view output"
```

---

### Task 3: Failing test — all variants drop, raise error

**Files:**

- Modify: `tests/test_svar_write_view.py`.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_write_view_raises_when_all_variants_drop(tmp_path: Path, svar_wv: SparseVar):
    """If every candidate variant has MAC=0 in the kept sample set, raise."""
    samples_all = list(svar_wv.available_samples)
    # Pick a region where the kept sample has no non-ref calls — easiest way:
    # subset to a synthetic sample set known to be empty. We achieve this by
    # restricting samples to one and choosing a region we know is variant-free
    # for that sample. If the fixture doesn't admit this naturally, construct
    # a synthetic SVAR as in test_write_view_drops_mac_zero_variants.
    one = [samples_all[0]]
    out = tmp_path / "empty.svar"
    # Region with no variants for sample 0; the test author may need to swap
    # in coordinates appropriate for the fixture.
    with pytest.raises(ValueError, match="no variants selected"):
        svar_wv.write_view(
            regions=(svar_wv.contigs[0], 0, 1),  # tiny window
            samples=one,
            output=out,
        )
```

- [ ] **Step 2: Run and confirm it passes (already covered by step 2 from Task 2)**

Run: `pixi run pytest tests/test_svar_write_view.py::test_write_view_raises_when_all_variants_drop -v`

Expected: PASS — the new pre-pass raises `ValueError("no variants selected by `regions` ...")` whenever `kept_var_idxs` ends up empty. If `regions=(chr1, 0, 1)` happens to be non-empty for the fixture, narrow the window further until empty.

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar_write_view.py
git commit -m "test(svar): assert write_view raises when MAC>0 drops everything"
```

---

### Task 4: Update `write_view` docstring

**Files:**

- Modify: `genoray/_svar.py` — `write_view` docstring near line 1206.

- [ ] **Step 1: Edit the docstring**

Insert into the `Parameters`/`Returns`/`Notes` section of `write_view`, after the existing `threads` parameter doc:

```rst
        Notes
        -----
        Variants whose minor allele count is 0 in the chosen sample subset are
        dropped from the output. If every candidate variant drops, a
        :class:`ValueError` is raised — the same error path as "no variants
        selected by ``regions``".
```

- [ ] **Step 2: Commit**

```bash
git add genoray/_svar.py
git commit -m "docs(svar): note MAC>0 invariant on write_view outputs"
```

---

## Phase B — Submodule + pixi wiring (genoray)

### Task 5: Add `genoray-cli` as a submodule

**Files:**

- Create: `.gitmodules` (auto-managed by `git submodule add`).
- Create (as submodule): `./genoray-cli/` directory.

- [ ] **Step 1: Add the submodule**

Run:

```bash
git submodule add https://github.com/d-laub/genoray-cli.git genoray-cli
```

Expected: clones the repo into `./genoray-cli/`, creates `.gitmodules`, and stages both.

- [ ] **Step 2: Pin to current `main`**

Run:

```bash
git -C genoray-cli checkout main
git -C genoray-cli pull --ff-only
```

Expected: HEAD is at the latest `main` commit. We will move it to the cli's feature branch in Task 11; for now `main` is fine.

- [ ] **Step 3: Commit the submodule addition**

```bash
git add .gitmodules genoray-cli
git commit -m "chore: add genoray-cli as a submodule"
```

---

### Task 6: Wire pixi to install the cli editably from the submodule

**Files:**

- Modify: `pixi.toml` — `[pypi-dependencies]` section.

- [ ] **Step 1: Read the current pixi.toml [pypi-dependencies]**

The block currently reads:

```toml
[pypi-dependencies]
genoray = { path = ".", editable = true, extras = ["cli"] }
seqpro = { git = "https://github.com/ml4gland/seqpro.git", rev = "main" }
# dev deps
seaborn = "*"
pooch = "*"
```

The `extras = ["cli"]` resolves `genoray-cli` from PyPI. We want pixi to prefer the submodule.

- [ ] **Step 2: Modify the block**

Change `[pypi-dependencies]` to:

```toml
[pypi-dependencies]
genoray = { path = ".", editable = true }
genoray-cli = { path = "./genoray-cli", editable = true }
seqpro = { git = "https://github.com/ml4gland/seqpro.git", rev = "main" }
# dev deps
seaborn = "*"
pooch = "*"
```

We dropped `extras = ["cli"]` because we now explicitly install `genoray-cli` editably; the `[cli]` extra in `pyproject.toml` is left unchanged for downstream pip users.

- [ ] **Step 3: Re-resolve the pixi env**

Run: `pixi install`

Expected: resolves successfully; `pixi list | grep genoray-cli` shows it pointing at `./genoray-cli`.

- [ ] **Step 4: Smoke-test the CLI entry point still works**

Run: `pixi run genoray --version`

Expected: prints `genoray X.Y.Z` and `genoray-cli 0.2.0`.

- [ ] **Step 5: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "chore: install genoray-cli editably from submodule via pixi"
```

---

### Task 7: Ensure CI checks out submodules

**Files:**

- Modify: any `.github/workflows/*.yml` that runs `actions/checkout@v*`.

- [ ] **Step 1: Audit workflows**

Run: `grep -rn "actions/checkout" .github/workflows/`

For every match, inspect the corresponding step. If the workflow runs tests or builds the editable env (anything that calls `pixi`), the checkout step needs `submodules: true`.

- [ ] **Step 2: Add `submodules: true` to relevant workflows**

For each affected workflow, change:

```yaml
      - uses: actions/checkout@v4
```

to:

```yaml
      - uses: actions/checkout@v4
        with:
          submodules: recursive
```

Workflows that *only* run `cz bump` or publish (which don't depend on the cli) can be left alone — but defaulting all of them to `submodules: recursive` is safe and trivial.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/
git commit -m "ci: check out submodules recursively"
```

---

## Phase C — `view` command (genoray-cli submodule)

> All Phase C work happens **inside** `./genoray-cli/`. From the repo root:
> `cd genoray-cli && git checkout -b feat/view`.
>
> Commits in this phase land on `feat/view` in the `d-laub/genoray-cli` repo. The
> `genoray` repo's submodule pointer is bumped in Task 11.

### Task 8: Scaffold tests dir + minimal SVAR fixture

**Files (paths relative to `genoray-cli/`):**

- Create: `tests/__init__.py` (empty).
- Create: `tests/conftest.py`.

- [ ] **Step 1: Switch to a new branch in the submodule**

```bash
cd genoray-cli
git checkout -b feat/view
```

- [ ] **Step 2: Add an empty `tests/__init__.py`**

Create the file empty.

- [ ] **Step 3: Add `tests/conftest.py`**

```python
"""Shared fixtures for genoray-cli tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_vcf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny inline VCF with 3 samples × 4 variants on chr1."""
    p = tmp_path_factory.mktemp("vcf") / "tiny.vcf"
    p.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\n"
        "chr1\t10\t.\tA\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/0\n"   # singleton in A
        "chr1\t20\t.\tC\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\n"   # in B and C
        "chr1\t30\t.\tG\tA\t.\t.\t.\tGT\t0/0\t0/0\t1/0\n"   # singleton in C
        "chr1\t40\t.\tT\tC\t.\t.\t.\tGT\t1/0\t0/0\t0/0\n"   # singleton in A
    )
    return p


@pytest.fixture(scope="session")
def tiny_svar(tmp_path_factory: pytest.TempPathFactory, tiny_vcf: Path) -> Path:
    """A tiny SVAR built from `tiny_vcf` via `SparseVar.from_vcf`."""
    from genoray import VCF, SparseVar

    out = tmp_path_factory.mktemp("svar") / "tiny.svar"
    SparseVar.from_vcf(out, VCF(tiny_vcf), max_mem="64m", overwrite=True)
    return out
```

- [ ] **Step 4: Verify the fixture loads**

Run: `pytest tests/conftest.py --collect-only`

Expected: no collection errors. (Conftest itself is not a test; the goal is just to confirm no import errors.)

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: scaffold tests dir with tiny VCF/SVAR fixture"
```

---

### Task 9: Helper module — region parser + mutex validator

**Files:**

- Create: `genoray_cli/_view_helpers.py`.
- Create: `tests/test_view_helpers.py`.

- [ ] **Step 1: Write failing unit tests**

Create `tests/test_view_helpers.py`:

```python
from __future__ import annotations

import pytest
import polars as pl

from genoray_cli._view_helpers import parse_regions_arg, require_exactly_one


def test_parse_regions_arg_single():
    df = parse_regions_arg("chr1:10-20")
    assert df["chrom"].to_list() == ["chr1"]
    # 1-based inclusive -> 0-based half-open
    assert df["start"].to_list() == [9]
    assert df["end"].to_list() == [20]


def test_parse_regions_arg_comma_list():
    df = parse_regions_arg("chr1:10-20,chr2:30-40")
    assert df["chrom"].to_list() == ["chr1", "chr2"]
    assert df["start"].to_list() == [9, 29]
    assert df["end"].to_list() == [20, 40]


def test_parse_regions_arg_bad_format():
    with pytest.raises(ValueError, match="region"):
        parse_regions_arg("not_a_region")


def test_require_exactly_one_zero():
    with pytest.raises(ValueError, match="exactly one of"):
        require_exactly_one("regions", a=None, b=None)


def test_require_exactly_one_both():
    with pytest.raises(ValueError, match="exactly one of"):
        require_exactly_one("regions", a="x", b="y")


def test_require_exactly_one_ok():
    require_exactly_one("regions", a="x", b=None)  # no raise
    require_exactly_one("regions", a=None, b="y")  # no raise
```

- [ ] **Step 2: Run, confirm import-time failure**

Run: `pytest tests/test_view_helpers.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'genoray_cli._view_helpers'`.

- [ ] **Step 3: Implement the helpers**

Create `genoray_cli/_view_helpers.py`:

```python
"""Helpers for the `genoray view` subcommand."""
from __future__ import annotations

import re
from typing import Any

import polars as pl

_REGION_RE = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def parse_regions_arg(s: str) -> pl.DataFrame:
    """Parse a bcftools-style ``-r`` value into a 0-based half-open DataFrame.

    Accepts a single ``chrom:start-end`` (1-based inclusive) or a comma-
    separated list. Returns columns ``chrom`` (Utf8), ``start`` (Int32),
    ``end`` (Int32).
    """
    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for piece in (p.strip() for p in s.split(",") if p.strip()):
        m = _REGION_RE.match(piece)
        if m is None:
            raise ValueError(
                f"region {piece!r} does not match 'chrom:start-end' (1-based inclusive)"
            )
        chroms.append(m["chrom"])
        starts.append(int(m["start"]) - 1)  # 1-based -> 0-based
        ends.append(int(m["end"]))
    return pl.DataFrame(
        {"chrom": chroms, "start": starts, "end": ends},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )


def require_exactly_one(name: str, **kwargs: Any) -> None:
    """Raise ValueError unless exactly one of the keyword values is non-None.

    Used to enforce bcftools-style flag-pair mutual exclusion, e.g.
    ``require_exactly_one("regions", regions=..., regions_file=...)``.
    """
    set_keys = [k for k, v in kwargs.items() if v is not None]
    if len(set_keys) != 1:
        flags = " / ".join(f"--{k.replace('_', '-')}" for k in kwargs)
        raise ValueError(f"exactly one of {flags} is required for {name}")
```

- [ ] **Step 4: Run, confirm passing**

Run: `pytest tests/test_view_helpers.py -v`

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray_cli/_view_helpers.py tests/test_view_helpers.py
git commit -m "feat(view): add region parser + mutex validator helpers"
```

---

### Task 10: Implement the `view` subcommand

**Files:**

- Modify: `genoray_cli/__main__.py` — append a new `@app.command def view(...)`.

- [ ] **Step 1: Add the import block to `__main__.py`**

At the top of `genoray_cli/__main__.py`, add (or extend) the imports:

```python
from typing import Annotated, Literal

from cyclopts import Parameter
```

- [ ] **Step 2: Append the `view` subcommand**

Append to `genoray_cli/__main__.py` (before `if __name__ == "__main__":`):

```python
@app.command
def view(
    source: Path,
    out: Path,
    *,
    regions: Annotated[str | None, Parameter(name=["--regions", "-r"])] = None,
    regions_file: Annotated[
        Path | None, Parameter(name=["--regions-file", "-R"])
    ] = None,
    samples: Annotated[str | None, Parameter(name=["--samples", "-s"])] = None,
    samples_file: Annotated[
        Path | None, Parameter(name=["--samples-file", "-S"])
    ] = None,
    fields: Annotated[list[str] | None, Parameter(name=["--fields", "-f"])] = None,
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
    overwrite: bool = False,
    threads: Annotated[int | None, Parameter(name=["--threads", "-@"])] = None,
) -> None:
    """Write a subset of an SVAR to a new SVAR directory.

    At least one of --regions/--regions-file or --samples/--samples-file is
    required. The missing side, if any, is treated as "all".
    """
    from genoray import SparseVar
    import polars as pl

    from ._view_helpers import parse_regions_arg, require_exactly_one

    # No-op guard
    if regions is None and regions_file is None and samples is None and samples_file is None:
        raise ValueError(
            "at least one of --regions/--regions-file or --samples/--samples-file is required"
        )

    # Mutex (each side: at most one of the pair)
    if regions is not None and regions_file is not None:
        require_exactly_one("regions", regions=regions, regions_file=regions_file)
    if samples is not None and samples_file is not None:
        require_exactly_one("samples", samples=samples, samples_file=samples_file)

    sv = SparseVar(source)

    # Resolve regions arg
    if regions is not None:
        regions_arg: object = parse_regions_arg(regions)
    elif regions_file is not None:
        regions_arg = regions_file
    else:
        # "all variants" — one row per contig spanning [0, max_pos+1)
        bounds = (
            sv.index.group_by("CHROM", maintain_order=True)
            .agg(start=pl.lit(0), end=pl.col("POS").max() + 1)
            .rename({"CHROM": "chrom"})
            .with_columns(
                pl.col("start").cast(pl.Int32),
                pl.col("end").cast(pl.Int32),
            )
        )
        regions_arg = bounds.select("chrom", "start", "end")

    # Resolve samples arg
    if samples is not None:
        samples_arg: object = [s for s in samples.split(",") if s]
    elif samples_file is not None:
        samples_arg = samples_file
    else:
        samples_arg = list(sv.available_samples)

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

- [ ] **Step 3: Verify the command shows up in help**

Run (from `genoray-cli/`): `python -m genoray_cli view --help`

Expected: prints help text with `-r/-R/-s/-S/-f/-@` etc.

- [ ] **Step 4: Commit**

```bash
git add genoray_cli/__main__.py
git commit -m "feat: add `genoray view` subcommand"
```

---

### Task 11: CLI smoke test — single region + single sample

**Files:**

- Create: `tests/test_view_cli.py`.

- [ ] **Step 1: Write the test**

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from genoray import SparseVar


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "genoray_cli", *argv],
        check=False,
        capture_output=True,
        text=True,
    )


def test_view_single_region_single_sample(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run([
        "view", str(tiny_svar), str(out),
        "-r", "chr1:1-100",
        "-s", "A",
    ])
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    assert list(sub.available_samples) == ["A"]
    assert sub.n_variants >= 1  # A has at least one non-ref call in chr1:1-100
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_view_cli.py::test_view_single_region_single_sample -v`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_view_cli.py
git commit -m "test: CLI roundtrip for single region + single sample"
```

---

### Task 12: CLI test — BED file + sample file

**Files:**

- Modify: `tests/test_view_cli.py`.

- [ ] **Step 1: Append the test**

```python
def test_view_bed_and_sample_file(tmp_path: Path, tiny_svar: Path):
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t0\t100\n")
    samples_f = tmp_path / "s.txt"
    samples_f.write_text("A\nB\n")
    out = tmp_path / "view.svar"
    r = _run([
        "view", str(tiny_svar), str(out),
        "-R", str(bed),
        "-S", str(samples_f),
    ])
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    assert sorted(sub.available_samples) == ["A", "B"]
```

- [ ] **Step 2: Run + commit**

```bash
pytest tests/test_view_cli.py::test_view_bed_and_sample_file -v
git add tests/test_view_cli.py
git commit -m "test: CLI accepts -R BED and -S samples-file"
```

---

### Task 13: CLI test — comma-list regions

**Files:**

- Modify: `tests/test_view_cli.py`.

- [ ] **Step 1: Append**

```python
def test_view_regions_comma_list(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run([
        "view", str(tiny_svar), str(out),
        "-r", "chr1:1-15,chr1:25-35",
        "-s", "A,B,C",
    ])
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    # Variants at POS 10 (in 1-15) and POS 30 (in 25-35) should be selected;
    # variant at POS 20 should NOT be (outside both ranges).
    positions = sub.index["POS"].to_list()
    assert 10 in positions
    assert 30 in positions
    assert 20 not in positions
```

- [ ] **Step 2: Run + commit**

```bash
pytest tests/test_view_cli.py::test_view_regions_comma_list -v
git add tests/test_view_cli.py
git commit -m "test: CLI parses comma-list -r into a regions DataFrame"
```

---

### Task 14: CLI test — no-op guard + "all" synthesis

**Files:**

- Modify: `tests/test_view_cli.py`.

- [ ] **Step 1: Append**

```python
def test_view_no_args_errors(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(["view", str(tiny_svar), str(out)])
    assert r.returncode != 0
    assert "at least one of" in (r.stderr + r.stdout).lower()


def test_view_regions_only_uses_all_samples(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run([
        "view", str(tiny_svar), str(out),
        "-r", "chr1:1-100",
    ])
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    src = SparseVar(tiny_svar)
    assert sorted(sub.available_samples) == sorted(src.available_samples)


def test_view_samples_only_uses_all_variants(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run([
        "view", str(tiny_svar), str(out),
        "-s", "A,B,C",
    ])
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    src = SparseVar(tiny_svar)
    # All variants kept since all samples kept.
    assert sub.n_variants == src.n_variants
```

- [ ] **Step 2: Run + commit**

```bash
pytest tests/test_view_cli.py -v
git add tests/test_view_cli.py
git commit -m "test: CLI no-op guard and 'all' synthesis paths"
```

---

### Task 15: Bump genoray-cli version and pin genoray floor

**Files:**

- Modify: `pyproject.toml` (in genoray-cli submodule).
- Modify: `README.md` (optional but recommended).

- [ ] **Step 1: Edit pyproject.toml**

In `genoray-cli/pyproject.toml`, change the version and dependency floor:

```toml
[project]
name = "genoray-cli"
version = "0.3.0"
# ...
dependencies = ["genoray>=2.5.0", "cyclopts", "polars"]
```

Add `polars` if not already present (`_view_helpers.py` uses it).

- [ ] **Step 2: Add a `view` usage section to `README.md`**

Append a short example, e.g.:

```markdown
### `view` — subset an SVAR

genoray view SOURCE.svar OUT.svar -r chr1:100-200 -s A,B,C
genoray view SOURCE.svar OUT.svar -R regions.bed -S samples.txt
genoray view SOURCE.svar OUT.svar -r chr1:1-100   # all samples
genoray view SOURCE.svar OUT.svar -s A            # all variants

At least one of `--regions/--regions-file` or `--samples/--samples-file`
is required.
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml README.md
git commit -m "feat: bump to 0.3.0 with `view` subcommand"
```

- [ ] **Step 4: Push the cli branch**

```bash
git push -u origin feat/view
```

Open a PR against `d-laub/genoray-cli:main`. This is the cli-side PR; merge + release `0.3.0` once reviewed.

---

## Phase D — Bump submodule pointer in genoray (after cli release)

> Run after `genoray-cli 0.3.0` is tagged and the `feat/view` branch is merged
> to `genoray-cli:main`.

### Task 16: Move the submodule pointer to the released tag

**Files (in this `genoray` repo):**

- Modify: `genoray-cli` (submodule pointer).
- Modify: `pyproject.toml` — `[cli]` extra floor.

- [ ] **Step 1: Fast-forward the submodule**

```bash
cd genoray-cli
git fetch origin --tags
git checkout 0.3.0   # or the corresponding tag/ref
cd ..
```

- [ ] **Step 2: Bump the `[cli]` extra floor in `pyproject.toml`**

Change `cli = ["genoray-cli>=0.2.0"]` to `cli = ["genoray-cli>=0.3.0"]`.

- [ ] **Step 3: Re-resolve and commit**

```bash
pixi install
git add genoray-cli pyproject.toml pixi.lock
git commit -m "chore: pin genoray-cli to 0.3.0 (adds `view` subcommand)"
```

---

## Self-Review Pass

- **Spec coverage:** submodule add (T5) ✓, pixi wiring (T6) ✓, CI submodule checkout (T7) ✓, `view` subcommand (T9–T10) ✓, no-op guard + "all" synthesis (T10 + T14) ✓, comma-list regions (T9 + T13) ✓, MAC>0 drop in write_view (T1–T2) ✓, all-drop raise (T3) ✓, docstring update (T4) ✓, CLI tests for `-r/-R/-s/-S` (T11–T13) ✓, release sequencing (T15–T16) ✓.
- **Placeholder scan:** Task 1 step 3 contains a documented escape hatch ("if it passes spuriously, replace with synthetic"); this is intentional guidance, not a placeholder. The synthetic-fixture snippet still references `gen_svar.py` rather than embedding a duplicate `dense2sparse` recipe — acceptable because the test author should reuse the existing helper rather than diverge.
- **Type consistency:** `parse_regions_arg` and `require_exactly_one` are referenced consistently in T9 + T10. `_nb_count_mac_per_kept` signature matches its single call site. `regions_arg`/`samples_arg` are `object`-typed in T10 to allow the str | DataFrame | Path | list[str] union to pass through to `write_view` cleanly.
- **`logger` import:** verified to be re-checked at T2 Step 2 (add if missing).
- **`polars` dependency in cli:** added explicitly in T15 (helper imports `pl`).

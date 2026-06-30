# Atomic Sibling-Tmp Writes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every durable `genoray` output (`.svar` directories and `.gvi` index files) write to a sibling temp path and atomically rename into place, so a crash never leaves a partial/corrupt output.

**Architecture:** Two context-manager helpers in `genoray/_utils.py` — `atomic_write_path` (single files) and `atomic_write_dir` (directories, backup-then-swap). Every write site writes into the yielded sibling tmp and the helper does the atomic `os.replace` on clean exit, cleaning up on failure. The per-contig chunk `TemporaryDirectory` moves to the output's parent so concat stays on one filesystem.

**Tech Stack:** Python, `tempfile`, `os.replace`, `shutil`, polars (`write_ipc`/`sink_ipc`), pytest, pixi.

## Global Constraints

- Commit convention: Conventional Commits. **Use `fix:` for all code commits** (this is a patch release; do NOT use `feat:`, which would trigger a minor bump). `docs:`/`test:` allowed for doc/test-only commits.
- Output bytes, schema, dtypes, and coordinate/missing-value conventions must be **byte-identical** to before this change.
- **No new public kwargs** — atomic behavior is unconditional ("by default").
- Sibling tmp = same parent directory ⇒ same filesystem ⇒ `os.replace` is atomic. Must work on POSIX and Windows (never `os.replace` a dir onto an existing dir).
- Run tests with `pixi run pytest <path>`.
- Reference spec: `docs/superpowers/specs/2026-06-30-atomic-sibling-writes-design.md`.

---

### Task 1: Atomic-write helpers in `_utils.py`

**Files:**
- Modify: `genoray/_utils.py` (imports at top `:1-9`; append helpers at end of file)
- Test: `tests/test_atomic_write.py` (create)

**Interfaces:**
- Consumes: nothing (leaf utilities).
- Produces:
  - `atomic_write_path(dest: Path) -> ContextManager[Path]` — yields a sibling tmp **file** path; on clean exit `os.replace(tmp, dest)`; on exception unlinks tmp and re-raises.
  - `atomic_write_dir(dest: Path) -> ContextManager[Path]` — yields a sibling staging **directory** path; on clean exit swaps it into `dest` (backup-then-swap if `dest` exists, with rollback); always cleans up staging/backup; on exception removes staging and re-raises.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_atomic_write.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

import genoray._utils as U
from genoray._utils import atomic_write_dir, atomic_write_path


def test_atomic_write_path_replaces_and_is_sibling(tmp_path: Path):
    dest = tmp_path / "index.gvi"
    with atomic_write_path(dest) as tmp:
        assert tmp.parent == dest.parent  # sibling => same filesystem
        assert tmp != dest
        tmp.write_bytes(b"NEW")
    assert dest.read_bytes() == b"NEW"
    # no leftover temp siblings
    assert [p.name for p in tmp_path.iterdir()] == ["index.gvi"]


def test_atomic_write_path_failure_preserves_dest(tmp_path: Path):
    dest = tmp_path / "index.gvi"
    dest.write_bytes(b"OLD")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write_path(dest) as tmp:
            tmp.write_bytes(b"PARTIAL")
            raise RuntimeError("boom")
    assert dest.read_bytes() == b"OLD"  # untouched
    assert [p.name for p in tmp_path.iterdir()] == ["index.gvi"]  # tmp cleaned


def test_atomic_write_dir_swaps_new(tmp_path: Path):
    dest = tmp_path / "out.svar"
    with atomic_write_dir(dest) as staging:
        assert staging.parent == dest.parent
        assert staging.is_dir()
        (staging / "data.bin").write_bytes(b"X")
    assert (dest / "data.bin").read_bytes() == b"X"
    assert [p.name for p in tmp_path.iterdir()] == ["out.svar"]


def test_atomic_write_dir_overwrite_replaces(tmp_path: Path):
    dest = tmp_path / "out.svar"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"OLD")
    with atomic_write_dir(dest) as staging:
        (staging / "new.bin").write_bytes(b"NEW")
    assert (dest / "new.bin").read_bytes() == b"NEW"
    assert not (dest / "old.bin").exists()  # fully replaced, not merged
    assert [p.name for p in tmp_path.iterdir()] == ["out.svar"]


def test_atomic_write_dir_failure_preserves_existing(tmp_path: Path):
    dest = tmp_path / "out.svar"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"OLD")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write_dir(dest) as staging:
            (staging / "partial.bin").write_bytes(b"PARTIAL")
            raise RuntimeError("boom")
    assert (dest / "old.bin").read_bytes() == b"OLD"  # untouched
    assert not (dest / "partial.bin").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["out.svar"]  # staging cleaned


def test_atomic_write_dir_rollback_on_swap_failure(tmp_path: Path, monkeypatch):
    dest = tmp_path / "out.svar"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"OLD")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # 1=dest->backup, 2=staging->dest (fail), 3=rollback
            raise OSError("swap boom")
        return real_replace(src, dst)

    monkeypatch.setattr(U.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="swap boom"):
        with atomic_write_dir(dest) as staging:
            (staging / "new.bin").write_bytes(b"NEW")

    assert (dest / "old.bin").read_bytes() == b"OLD"  # rolled back
    assert not (dest / "new.bin").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["out.svar"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_atomic_write.py -v`
Expected: FAIL with `ImportError: cannot import name 'atomic_write_dir'` (helpers not defined yet).

- [ ] **Step 3: Add imports**

In `genoray/_utils.py`, update the top-of-file imports. Change line 6 and add two stdlib imports so the block reads:

```python
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeGuard, TypeVar, overload
```

(`Iterable`, `os`, `contextmanager`, `Path` already exist; you are adding `shutil`, `tempfile`, and `Iterator`.)

- [ ] **Step 4: Append the helpers**

Add to the **end** of `genoray/_utils.py`:

```python
def _unique_sibling(dest: Path, suffix: str) -> Path:
    """Return a not-yet-existing sibling path of *dest* with *suffix* injected.

    Used to move an existing output dir aside before an atomic swap. The name is
    hidden (leading dot) and keyed by PID; a counter disambiguates collisions.
    """
    base = dest.with_name(f".{dest.name}{suffix}.{os.getpid()}")
    candidate = base
    i = 0
    while candidate.exists():
        i += 1
        candidate = dest.with_name(f"{base.name}.{i}")
    return candidate


@contextmanager
def atomic_write_path(dest: Path) -> Iterator[Path]:
    """Write a single file atomically: yield a sibling temp path to write to, then
    ``os.replace`` it onto *dest* on clean exit. On any exception the temp file is
    removed and *dest* is left untouched.

    The temp file is created in ``dest.parent`` so the final rename stays on one
    filesystem (a true atomic rename, no cross-device copy).
    """
    dest = Path(dest)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    else:
        os.replace(tmp, dest)


@contextmanager
def atomic_write_dir(dest: Path) -> Iterator[Path]:
    """Write a directory atomically: yield a sibling staging dir to populate, then
    swap it into place on clean exit.

    Swap is backup-then-swap: if *dest* already exists it is first moved aside to a
    fresh sibling name (so the staging dir is renamed onto a *non-existent* path,
    which is portable on POSIX and Windows), then the moved-aside dir is removed.
    If the final rename fails, the moved-aside dir is rolled back. The staging dir
    is always cleaned up; on an exception in the body *dest* is left untouched.

    The staging dir is created in ``dest.parent`` (same filesystem) so the swap is
    a true atomic rename and intermediate writes never cross devices.
    """
    dest = Path(dest)
    staging = Path(
        tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    )
    backup: Path | None = None
    try:
        yield staging
        if dest.exists():
            backup = _unique_sibling(dest, ".old")
            os.replace(dest, backup)
        os.replace(staging, dest)
    except BaseException:
        # If we moved dest aside but failed before/at the swap, roll it back.
        if backup is not None and backup.exists() and not dest.exists():
            os.replace(backup, dest)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_atomic_write.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add genoray/_utils.py tests/test_atomic_write.py
git commit -m "fix: add atomic sibling-tmp write helpers"
```

---

### Task 2: Atomic `.gvi` index writes (VCF + PGEN)

**Files:**
- Modify: `genoray/_vcf.py:1215-1217` (the `write_ipc`); imports near top
- Modify: `genoray/_pgen.py:1253-1260` (`_write_index`); imports near top
- Test: `tests/test_atomic_write.py` (append)

**Interfaces:**
- Consumes: `atomic_write_path` from `genoray._utils` (Task 1).
- Produces: no signature changes — `VCF._write_gvi_index(...)` and PGEN module-level `_write_index(index_path)` keep their current call signatures; only their write becomes atomic.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_atomic_write.py`:

```python
import shutil

import polars as pl

from genoray import VCF

_DDIR = Path(__file__).parent / "data"


def _raise_boom(*args, **kwargs):
    raise RuntimeError("write boom")


def test_vcf_gvi_write_atomic_no_leftover(tmp_path: Path):
    for ext in (".vcf.gz", ".vcf.gz.csi"):
        shutil.copy(_DDIR / f"biallelic{ext}", tmp_path / f"biallelic{ext}")
    vcf = VCF(tmp_path / "biallelic.vcf.gz")
    vcf._write_gvi_index(overwrite=True)
    gvi = vcf._index_path()
    assert gvi.exists()
    pl.read_ipc(gvi)  # loads cleanly
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_vcf_gvi_write_preserves_on_failure(tmp_path: Path, monkeypatch):
    for ext in (".vcf.gz", ".vcf.gz.csi"):
        shutil.copy(_DDIR / f"biallelic{ext}", tmp_path / f"biallelic{ext}")
    vcf = VCF(tmp_path / "biallelic.vcf.gz")
    vcf._write_gvi_index(overwrite=True)
    gvi = vcf._index_path()
    good = gvi.read_bytes()
    monkeypatch.setattr(pl.DataFrame, "write_ipc", _raise_boom)
    with pytest.raises(RuntimeError, match="write boom"):
        vcf._write_gvi_index(overwrite=True)
    assert gvi.read_bytes() == good  # prior index intact
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_pgen_write_index_preserves_on_failure(tmp_path: Path, monkeypatch):
    from genoray._pgen import _write_index

    for ext in (".pgen", ".pvar", ".psam"):
        shutil.copy(_DDIR / f"biallelic{ext}", tmp_path / f"biallelic{ext}")
    idx = tmp_path / "biallelic.pvar.gvi"
    _write_index(idx)
    good = idx.read_bytes()
    monkeypatch.setattr(pl.LazyFrame, "sink_ipc", _raise_boom)
    with pytest.raises(RuntimeError, match="write boom"):
        _write_index(idx)
    assert idx.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_atomic_write.py -k "gvi or write_index" -v`
Expected: `test_*_preserves_on_failure` FAIL — without atomic writes the failed write truncates/leaves a partial `.gvi` (read_bytes != good) and/or leaves a `.tmp` sibling. (`test_vcf_gvi_write_atomic_no_leftover` may already pass; that is fine — the gating tests are the failure-preservation ones.)

- [ ] **Step 3: Make the VCF `.gvi` write atomic**

In `genoray/_vcf.py`, add `atomic_write_path` to the existing `from ._utils import ...` line (find it near the top of the file; it already imports from `._utils`). Then replace the write at `:1215-1217`:

```python
        index.with_columns(pl.col("ALT").list.join(",")).collect().write_ipc(
            self._index_path(), compression="zstd"
        )
```

with:

```python
        with atomic_write_path(self._index_path()) as _tmp:
            index.with_columns(pl.col("ALT").list.join(",")).collect().write_ipc(
                _tmp, compression="zstd"
            )
```

- [ ] **Step 4: Make the PGEN `.gvi` write atomic**

In `genoray/_pgen.py`, add `atomic_write_path` to the existing `from ._utils import ...` line. Then replace `_write_index` at `:1253-1260`:

```python
def _write_index(index_path: Path):
    """Write PVAR index."""

    (
        _scan_pvar(index_path.with_suffix(""))
        .rename({"#CHROM": "CHROM"})
        .sink_ipc(index_path)
    )
```

with:

```python
def _write_index(index_path: Path):
    """Write PVAR index."""

    with atomic_write_path(index_path) as _tmp:
        (
            _scan_pvar(index_path.with_suffix(""))
            .rename({"#CHROM": "CHROM"})
            .sink_ipc(_tmp)
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_atomic_write.py -v`
Expected: PASS (all, including Task 1's 6).

- [ ] **Step 6: Run the index-related suites for regressions**

Run: `pixi run pytest tests/test_vcf.py tests/test_pgen.py -q`
Expected: PASS (no regressions in index build/load).

- [ ] **Step 7: Commit**

```bash
git add genoray/_vcf.py genoray/_pgen.py tests/test_atomic_write.py
git commit -m "fix: write .gvi index files atomically via sibling tmp"
```

---

### Task 3: Atomic `.svar` write in `from_vcf`

**Files:**
- Modify: `genoray/_svar.py` — `from_vcf` body (`:1042`, `:1107-1178`); `from ._utils import` line (`:39`)
- Test: `tests/test_atomic_write.py` (append)

**Interfaces:**
- Consumes: `atomic_write_dir` from `genoray._utils` (Task 1); existing module-level `_concat_data`, `_subset_var_idxs_and_recompute_af`, `_write_index_from_working`, `cls._index_path`.
- Produces: `SparseVar.from_vcf(...)` unchanged signature; output is now built in a sibling staging dir and atomically swapped into `out`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_atomic_write.py`:

```python
from genoray import SparseVar


def _dir_digest(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_from_vcf_atomic_no_leftover(tmp_path: Path):
    out = tmp_path / "x.svar"
    SparseVar.from_vcf(out, VCF(_DDIR / "biallelic.vcf.gz"), max_mem="1g")
    assert (out / "metadata.json").exists()
    SparseVar(out)  # loads cleanly
    # only the final output remains in the parent dir
    assert [p.name for p in tmp_path.iterdir() if p.name != "x.svar"] == []


def test_from_vcf_atomic_preserves_existing_on_failure(tmp_path: Path, monkeypatch):
    import genoray._svar as S

    out = tmp_path / "x.svar"
    SparseVar.from_vcf(out, VCF(_DDIR / "biallelic.vcf.gz"), max_mem="1g")
    before = _dir_digest(out)

    monkeypatch.setattr(S, "_concat_data", _raise_boom)
    with pytest.raises(RuntimeError, match="write boom"):
        SparseVar.from_vcf(
            out, VCF(_DDIR / "biallelic.vcf.gz"), max_mem="1g", overwrite=True
        )

    assert _dir_digest(out) == before  # old output intact
    assert [p.name for p in tmp_path.iterdir() if p.name != "x.svar"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_atomic_write.py -k from_vcf -v`
Expected: `test_from_vcf_atomic_preserves_existing_on_failure` FAILs — current `from_vcf` writes in place, so the injected `_concat_data` failure leaves `out` mutated (digest != before) and/or a leftover chunk/temp dir.

- [ ] **Step 3: Update the `_utils` import**

In `genoray/_svar.py:39`, change:

```python
from ._utils import ContigNormalizer, format_memory, parse_memory
```

to:

```python
from ._utils import ContigNormalizer, atomic_write_dir, format_memory, parse_memory
```

- [ ] **Step 4: Replace `out.mkdir` with a parent mkdir**

In `from_vcf`, change `:1042`:

```python
        out.mkdir(parents=True, exist_ok=True)
```

to:

```python
        out.parent.mkdir(parents=True, exist_ok=True)
```

(The `FileExistsError` guard at `:1038` stays as-is — it runs before any disk write.)

- [ ] **Step 5: Wrap the write body in `atomic_write_dir` and redirect targets to `staging`**

The write body runs from the `metadata.json` write (`:1107`) to the end of the method (`:1178`). Wrap it in `with atomic_write_dir(out) as staging:` and replace every `out` write target with `staging`. After the edit the body reads:

```python
        with atomic_write_dir(out) as staging:
            with open(staging / "metadata.json", "w") as f:
                json_str = SparseVarMetadata(
                    version=CURRENT_VERSION,
                    contigs=contigs,
                    samples=caller_samples,
                    ploidy=out_ploidy,
                    fields={"dosages": np.dtype(DOSAGE_TYPE).name} if with_dosages else {},
                ).model_dump_json()
                f.write(json_str)

            subsetting_samples = samples is not None
            # When NOT subsetting samples, write the (region-restricted) index up front.
            if not subsetting_samples:
                _write_index_from_working(
                    working_df, kept_rows, cls._index_path(staging), alt_is_utf8, ilen_added
                )

            max_mem = parse_memory(max_mem)
            n_out = len(caller_samples)
            effective_n_jobs = joblib.cpu_count() if n_jobs == -1 else n_jobs
            effective_n_jobs = min(effective_n_jobs, len(contigs))
            job_mem = max_mem // effective_n_jobs

            with TemporaryDirectory(dir=out.parent) as chunk_dir:
                chunk_dir = Path(chunk_dir)

                shape = (n_out, out_ploidy)
                tasks = []
                for chunk_idx, c in enumerate(contigs):
                    task = joblib.delayed(_process_contig_vcf)(
                        vcf.path,
                        dosage_field=vcf.dosage_field if with_dosages else None,
                        max_mem=job_mem,
                        contig=c,
                        chunk_dir=chunk_dir,
                        chunk_idx=chunk_idx,
                        cyvcf2_filter=vcf._filter,
                        pl_filter=vcf._pl_filter,
                        caller_samples=None if samples is None else caller_samples,
                        keep_local=keep_local_by_contig.get(c),
                        haploid=haploid,
                    )
                    tasks.append(task)

                with (
                    joblib_progress(
                        description=f"Processing contigs using {effective_n_jobs} jobs",
                        total=len(tasks),
                    ),
                    joblib.Parallel(n_jobs=effective_n_jobs) as parallel,
                ):
                    results: list[tuple[int, int]] = list(parallel(tasks))  # type: ignore

                logger.info("Concatenating intermediate chunks")
                _concat_data(staging, chunk_dir, shape, results, with_dosages=with_dosages)

                if subsetting_samples:
                    survivors, af = _subset_var_idxs_and_recompute_af(
                        staging,
                        n_total=len(kept_rows),
                        n_out=n_out,
                        ploidy=out_ploidy,
                        with_dosages=with_dosages,
                    )
                    _write_index_from_working(
                        working_df,
                        kept_rows[survivors],
                        cls._index_path(staging),
                        alt_is_utf8,
                        ilen_added,
                        af=af,
                    )
```

Key changes vs. the original: the whole block is indented one level under `with atomic_write_dir(out) as staging:`; `out` → `staging` in the `metadata.json` open, both `cls._index_path(out)` calls, the `_concat_data(...)` call, and the `_subset_var_idxs_and_recompute_af(...)` call; and `TemporaryDirectory()` → `TemporaryDirectory(dir=out.parent)`. The block from `:1044` (`vcf._write_gvi_index`) through the contig/block computations stays **above** and outside the `with` (it touches the source VCF and in-memory state only, not `out`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run pytest tests/test_atomic_write.py -k from_vcf -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Run the from_vcf-touching suites for regressions**

Run: `pixi run pytest tests/test_svar.py tests/test_svar_haploid.py tests/test_symbolic_ilen.py -q`
Expected: PASS (no regressions; output still byte-identical/loadable).

- [ ] **Step 8: Commit**

```bash
git add genoray/_svar.py tests/test_atomic_write.py
git commit -m "fix: write .svar output atomically in from_vcf via sibling staging dir"
```

---

### Task 4: Atomic `.svar` write in `from_pgen`

**Files:**
- Modify: `genoray/_svar.py` — `from_pgen` body (`:1244`, `:1307-1394`)
- Test: `tests/test_atomic_write.py` (append)

**Interfaces:**
- Consumes: `atomic_write_dir` (already imported in Task 3); existing `_concat_data`, `_subset_var_idxs_and_recompute_af`, `_write_index_from_working`, `cls._index_path`.
- Produces: `SparseVar.from_pgen(...)` unchanged signature; output built in sibling staging dir and atomically swapped into `out`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_atomic_write.py`:

```python
from genoray import PGEN


def test_from_pgen_atomic_no_leftover(tmp_path: Path):
    out = tmp_path / "p.svar"
    SparseVar.from_pgen(out, PGEN(_DDIR / "biallelic.pgen"), max_mem="1g")
    assert (out / "metadata.json").exists()
    SparseVar(out)
    assert [p.name for p in tmp_path.iterdir() if p.name != "p.svar"] == []


def test_from_pgen_atomic_preserves_existing_on_failure(tmp_path: Path, monkeypatch):
    import genoray._svar as S

    out = tmp_path / "p.svar"
    SparseVar.from_pgen(out, PGEN(_DDIR / "biallelic.pgen"), max_mem="1g")
    before = _dir_digest(out)

    monkeypatch.setattr(S, "_concat_data", _raise_boom)
    with pytest.raises(RuntimeError, match="write boom"):
        SparseVar.from_pgen(
            out, PGEN(_DDIR / "biallelic.pgen"), max_mem="1g", overwrite=True
        )

    assert _dir_digest(out) == before
    assert [p.name for p in tmp_path.iterdir() if p.name != "p.svar"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_atomic_write.py -k from_pgen -v`
Expected: `test_from_pgen_atomic_preserves_existing_on_failure` FAILs (in-place write mutates `out` on the injected failure).

- [ ] **Step 3: Replace `out.mkdir` with a parent mkdir**

In `from_pgen`, change `:1244`:

```python
        out.mkdir(parents=True, exist_ok=True)
```

to:

```python
        out.parent.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Wrap the write body in `atomic_write_dir` and redirect targets to `staging`**

The write body runs from the `metadata.json` write (`:1307`) to the end of the method (`:1394`). Wrap it in `with atomic_write_dir(out) as staging:` and replace every `out` write target with `staging`. After the edit the body reads:

```python
        with atomic_write_dir(out) as staging:
            with open(staging / "metadata.json", "w") as f:
                json_str = SparseVarMetadata(
                    version=CURRENT_VERSION,
                    contigs=contigs,
                    samples=caller_samples,
                    ploidy=out_ploidy,
                    fields={"dosages": np.dtype(DOSAGE_TYPE).name} if with_dosages else {},
                ).model_dump_json()
                f.write(json_str)

            if with_dosages and pgen._sei is None:
                raise ValueError("PGEN must be bi-allelic with filters applied")

            subsetting_samples = samples is not None
            # (mirrors from_vcf; keep in sync) metadata written + no-subset index path
            if not subsetting_samples:
                _write_index_from_working(
                    working_df, kept_rows, cls._index_path(staging), alt_is_utf8, ilen_added
                )

            max_mem = parse_memory(max_mem)
            effective_n_jobs = joblib.cpu_count() if n_jobs == -1 else n_jobs
            effective_n_jobs = min(effective_n_jobs, len(contigs))
            job_mem = max_mem // effective_n_jobs
            mem_per_var = pgen._mem_per_variant(
                pgen.GenosDosages if with_dosages else pgen.Genos  # type: ignore
            )

            shape = (n_out, out_ploidy)
            with TemporaryDirectory(dir=out.parent) as contig_dir:
                contig_dir = Path(contig_dir)

                tasks: list[Any] = []
                for c in contigs:
                    keep_idxs = keep_by_contig.get(c)
                    if keep_idxs is None or len(keep_idxs) == 0:
                        continue

                    task = joblib.delayed(_process_contig_pgen)(
                        geno_path=pgen.geno_path,
                        dosage_path=pgen.dosage_path if with_dosages else None,
                        max_mem=job_mem,
                        keep_idxs=keep_idxs,
                        mem_per_var=mem_per_var,
                        n_samples=pgen.n_samples,
                        ploidy=pgen.ploidy,
                        chunk_dir=contig_dir,
                        chunk_idx=len(tasks),
                        sample_subset=sample_subset,
                        haploid=haploid,
                    )
                    tasks.append(task)

                pgen._free_index()
                # PgenReaders can be multi-GB allocations, close them to free memory
                pgen._geno_pgen.close()
                if pgen.dosage_path is not None:
                    pgen._dose_pgen.close()

                with (
                    joblib_progress(
                        description=f"Processing contigs using {effective_n_jobs} jobs",
                        total=len(tasks),
                    ),
                    joblib.Parallel(n_jobs=effective_n_jobs) as parallel,
                ):
                    results: list[tuple[int, int]] = list(parallel(tasks))  # type: ignore

                logger.info("Concatenating intermediate chunks")
                _concat_data(staging, contig_dir, shape, results, with_dosages=with_dosages)

                # (mirrors from_vcf; keep in sync) MAC-drop + subset-sample index finalize
                if subsetting_samples:
                    survivors, af = _subset_var_idxs_and_recompute_af(
                        staging,
                        n_total=len(kept_rows),
                        n_out=n_out,
                        ploidy=out_ploidy,
                        with_dosages=with_dosages,
                    )
                    _write_index_from_working(
                        working_df,
                        kept_rows[survivors],
                        cls._index_path(staging),
                        alt_is_utf8,
                        ilen_added,
                        af=af,
                    )
```

Key changes vs. the original: the block is indented one level under `with atomic_write_dir(out) as staging:`; `out` → `staging` in the `metadata.json` open, both `cls._index_path(out)` calls, `_concat_data(...)`, and `_subset_var_idxs_and_recompute_af(...)`; `TemporaryDirectory()` → `TemporaryDirectory(dir=out.parent)`. Everything above `:1307` (region/contig resolution, the POS-alignment asserts) stays outside the `with`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_atomic_write.py -k from_pgen -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the from_pgen-touching suites for regressions**

Run: `pixi run pytest tests/test_pgen.py tests/test_svar.py tests/test_svar_haploid.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add genoray/_svar.py tests/test_atomic_write.py
git commit -m "fix: write .svar output atomically in from_pgen via sibling staging dir"
```

---

### Task 5: Atomic `.svar` write in `write_view`

**Files:**
- Modify: `genoray/_svar.py` — `write_view` Band C (`:2009-2157`)
- Test: `tests/test_atomic_write.py` (append)

**Interfaces:**
- Consumes: `atomic_write_dir` (already imported); existing Band-C locals (`output`, `fields_to_write`, `ref_obj`, `pbar`, etc.).
- Produces: `SparseVar.write_view(...)` unchanged signature; the destructive `rmtree`+`mkdir` is replaced by staging + atomic swap.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_atomic_write.py`:

```python
def test_write_view_atomic_no_leftover(tmp_path: Path):
    src = SparseVar(_DDIR / "biallelic.vcf.svar")
    out = tmp_path / "v.svar"
    region = (src.contigs[0], 0, 1_000_000)
    src.write_view(region, src.available_samples, out)
    SparseVar(out)
    assert [p.name for p in tmp_path.iterdir() if p.name != "v.svar"] == []


def test_write_view_atomic_preserves_existing_on_failure(tmp_path: Path, monkeypatch):
    import genoray._svar as S

    src = SparseVar(_DDIR / "biallelic.vcf.svar")
    out = tmp_path / "v.svar"
    region = (src.contigs[0], 0, 1_000_000)
    src.write_view(region, src.available_samples, out)
    before = _dir_digest(out)

    # lengths_to_offsets is called inside Band C, after staging is created but
    # before the output is finalized — a clean seam to simulate a mid-write crash.
    monkeypatch.setattr(S, "lengths_to_offsets", _raise_boom)
    with pytest.raises(RuntimeError, match="write boom"):
        src.write_view(region, src.available_samples, out, overwrite=True)

    assert _dir_digest(out) == before
    assert [p.name for p in tmp_path.iterdir() if p.name != "v.svar"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_atomic_write.py -k write_view -v`
Expected: `test_write_view_atomic_preserves_existing_on_failure` FAILs — current Band C `rmtree`s `output` before writing, so the injected failure destroys the existing output (digest mismatch / `out` missing).

- [ ] **Step 3: Replace `rmtree`+`mkdir` with `atomic_write_dir` staging**

In `write_view` Band C, the current structure (`:2009`+) is:

```python
        with pbar or nullcontext():
            task = (
                pbar.add_task("counting entries", total=n_steps)
                if pbar is not None
                else None
            )

            def _step(desc: str) -> None:
                ...

            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True)

            # --- 5. Pass 1: count kept entries per output slot ---
            ...
```

Change the `with` line to also enter `atomic_write_dir`, delete the `if output.exists(): rmtree / mkdir` block, and replace every `output` write target inside the block with `staging`. The header becomes:

```python
        with atomic_write_dir(output) as staging, (pbar or nullcontext()):
            task = (
                pbar.add_task("counting entries", total=n_steps)
                if pbar is not None
                else None
            )

            def _step(desc: str) -> None:
                """Mark the current phase complete and label the next one."""
                if pbar is not None:
                    assert task is not None
                    pbar.advance(task)
                    pbar.update(task, description=desc)

            # (no rmtree/mkdir: atomic_write_dir already created `staging`)

            # --- 5. Pass 1: count kept entries per output slot ---
```

Then, in the rest of Band C (steps 6–11, `:2042-2152`), replace each `output` with `staging`. The exact occurrences to change:

- `np.memmap(output / "offsets.npy", ...)` → `np.memmap(staging / "offsets.npy", ...)`
- `np.memmap(output / "variant_idxs.npy", ...)` → `staging / "variant_idxs.npy"`
- `np.memmap(output / f"{name}.npy", ...)` → `staging / f"{name}.npy"`
- `out_index.sink_ipc(SparseVar._index_path(output))` → `...(SparseVar._index_path(staging))`
- `with open(output / "metadata.json", "w") as f:` → `staging / "metadata.json"`
- `out_svar = SparseVar(output)` → `out_svar = SparseVar(staging)`

`output.parent` is guaranteed to exist by the caller (the CLI/user passes a path whose parent exists); `atomic_write_dir` uses `tempfile.mkdtemp(dir=output.parent, ...)`. No extra `mkdir` is needed because Band A already validated and the staging dir creation handles directory creation. (The fail-fast `output.exists() && !overwrite` and `output == source` guards in Band A are unchanged and still run before Band C.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_atomic_write.py -k write_view -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full write_view suite for regressions**

Run: `pixi run pytest tests/test_svar_write_view.py tests/test_svar_from_subset.py -q`
Expected: PASS (byte-identical output, including the existing `test_write_view_progress_byte_identical` and overwrite tests).

- [ ] **Step 6: Commit**

```bash
git add genoray/_svar.py tests/test_atomic_write.py
git commit -m "fix: write .svar output atomically in write_view via sibling staging dir"
```

---

### Task 6: Document atomic writes in SKILL.md + full-suite gate

**Files:**
- Modify: `skills/genoray-api/SKILL.md` (add a short note)
- Test: full suite

**Interfaces:**
- Consumes: behavior from Tasks 1–5.
- Produces: doc note; no code.

- [ ] **Step 1: Add the atomic-write note to SKILL.md**

Open `skills/genoray-api/SKILL.md`, find the section covering `write`/`from_vcf`/`from_pgen`/`write_view` (the `.svar` writing docs near the `write_view` progress note at `:300`). Add a short paragraph (place it near the `write_view`/writing docs):

```markdown
Writes are crash-safe and atomic. `from_vcf`, `from_pgen`, and `write_view` build
the `.svar` directory in a hidden sibling staging directory (`.<name>.tmp…` next
to the output) and atomically rename it into place only after the write fully
succeeds; `.gvi` index files are written the same way. A crash mid-write never
leaves a partial or corrupt output, and overwriting an existing output preserves
it until the replacement is complete. Output bytes are unchanged — this is a
durability guarantee only.
```

- [ ] **Step 2: Verify no placeholder/stale references**

Run: `grep -n "sibling staging" skills/genoray-api/SKILL.md`
Expected: the new note is present.

- [ ] **Step 3: Run the full test suite**

Run: `pixi run test`
Expected: PASS (all tests, no regressions; this also regenerates test data via `gen_from_vcf.sh`).

- [ ] **Step 4: Lint/format**

Run: `ruff check genoray tests && ruff format --check genoray tests`
Expected: clean (or run `ruff format genoray tests` to fix formatting, then re-stage).

- [ ] **Step 5: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs: note atomic crash-safe .svar/.gvi writes in SKILL.md"
```

---

## Self-Review

**Spec coverage:**
- Sibling-tmp + atomic rename for `.svar` dirs → Tasks 3 (from_vcf), 4 (from_pgen), 5 (write_view). ✓
- Sibling-tmp + atomic rename for `.gvi` files → Task 2 (VCF `_write_gvi_index`, PGEN `_write_index`). ✓
- Helpers `atomic_write_path` / `atomic_write_dir` with backup-then-swap + rollback → Task 1. ✓
- Sibling chunk dir (`TemporaryDirectory(dir=out.parent)`) → Tasks 3 & 4 (Step 5/Step 4). ✓
- Byte-identical output → regression suites in Tasks 3/4/5 + full suite in Task 6; existing `test_write_view_progress_byte_identical` covers write_view. ✓
- No new public kwargs; `fix:` commits (patch) → Global Constraints + commit messages. ✓
- `.svar`-internal `index.arrow` writes inherit dir-level atomicity (no change) → covered by staging redirect; not a separate task (correct per spec non-goals). ✓
- SKILL.md note → Task 6. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**Type/name consistency:** `atomic_write_path(dest) -> Iterator[Path]` and `atomic_write_dir(dest) -> Iterator[Path]` defined in Task 1 and used by the same names in Tasks 2–5; `_raise_boom` / `_dir_digest` test helpers defined once in `tests/test_atomic_write.py` (Tasks 2 & 3) and reused by later tasks in the same file. `_unique_sibling(dest, suffix)` defined and used only within `atomic_write_dir`. ✓

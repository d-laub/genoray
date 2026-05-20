# SparseVar.write_view Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SparseVar.write_view(regions, samples, output, ...)` that writes a region- and sample-subset of an SVAR directory to disk, with numba-parallel two-pass copy and no per-thread scratch.

**Architecture:** A single new method on `SparseVar` orchestrates: (1) normalize regions/samples/fields, (2) resolve a sorted, deduped `kept_var_idxs` honoring `regions_overlap` mode, (3) numba kernel pass 1 computes new offsets, (4) allocate output memmaps, (5) numba kernel pass 2 streams variant indices and field values into the output (membership + remap inlined per element — zero per-thread scratch), (6) compute AFs into the new index, (7) write `metadata.json` + `index.arrow`.

**Tech Stack:** Python, numpy, polars, numba, seqpro, pyranges (via `sp.bed`), hirola, pgenlib-stack already in use by genoray.

**Spec:** `docs/superpowers/specs/2026-05-19-svar-write-view-design.md`

---

## File Structure

- **Modify** `genoray/_utils.py` — add `_resolve_threads`, `numba_threads` context manager.
- **Modify** `genoray/_svar.py` — add public `SparseVar.write_view`, private helpers (`_normalize_regions`, `_normalize_samples`, `_validate_fields`, `_resolve_kept_var_idxs`), and three numba kernels (`_nb_count_kept`, `_nb_write_var_idxs`, `_nb_write_field`).
- **Modify** `tests/test_utils.py` — tests for `_resolve_threads` and `numba_threads`.
- **Create** `tests/test_svar_write_view.py` — end-to-end tests of `write_view`.

No new modules.

---

## Task 1: Threading helpers in `_utils.py`

**Files:**
- Modify: `genoray/_utils.py`
- Test: `tests/test_utils.py`

### - [ ] Step 1.1: Write failing test for `_resolve_threads`

Add to `tests/test_utils.py`:

```python
from unittest.mock import patch

from genoray._utils import _resolve_threads


def test_resolve_threads_explicit():
    assert _resolve_threads(4) == 4
    assert _resolve_threads(1) == 1


def test_resolve_threads_default_uses_affinity():
    with patch("os.sched_getaffinity", return_value={0, 1, 2}, create=True):
        assert _resolve_threads(None) == 3


def test_resolve_threads_default_falls_back_to_cpu_count():
    with (
        patch("os.sched_getaffinity", side_effect=AttributeError, create=True),
        patch("os.cpu_count", return_value=8),
    ):
        assert _resolve_threads(None) == 8


def test_resolve_threads_default_falls_back_to_one():
    with (
        patch("os.sched_getaffinity", side_effect=AttributeError, create=True),
        patch("os.cpu_count", return_value=None),
    ):
        assert _resolve_threads(None) == 1
```

### - [ ] Step 1.2: Run tests to verify they fail

Run: `pixi run pytest tests/test_utils.py::test_resolve_threads_explicit -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_threads'`.

### - [ ] Step 1.3: Implement `_resolve_threads` and `numba_threads`

Append to `genoray/_utils.py`:

```python
import os
from contextlib import contextmanager


def _resolve_threads(threads: int | None) -> int:
    """Resolve the effective number of threads.

    - If `threads` is given, return it as-is.
    - Else prefer `os.sched_getaffinity(0)` (Linux), else `os.cpu_count()`, else 1.
    """
    if threads is not None:
        return threads
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1


@contextmanager
def numba_threads(n: int):
    """Temporarily set the numba thread count, restoring the previous value on exit."""
    import numba

    prev = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        yield
    finally:
        numba.set_num_threads(prev)
```

### - [ ] Step 1.4: Run `_resolve_threads` tests

Run: `pixi run pytest tests/test_utils.py -k resolve_threads -v`
Expected: 4 passed.

### - [ ] Step 1.5: Write failing test for `numba_threads`

Add to `tests/test_utils.py`:

```python
def test_numba_threads_sets_and_restores():
    import numba

    from genoray._utils import numba_threads

    original = numba.get_num_threads()
    target = max(1, original - 1) if original > 1 else 1

    with numba_threads(target):
        assert numba.get_num_threads() == target
    assert numba.get_num_threads() == original


def test_numba_threads_restores_on_exception():
    import numba

    from genoray._utils import numba_threads

    original = numba.get_num_threads()
    target = max(1, original - 1) if original > 1 else 1

    with pytest.raises(RuntimeError):
        with numba_threads(target):
            assert numba.get_num_threads() == target
            raise RuntimeError("boom")
    assert numba.get_num_threads() == original
```

If `pytest` isn't already imported in this file, add `import pytest` at the top.

### - [ ] Step 1.6: Run numba_threads tests

Run: `pixi run pytest tests/test_utils.py -k numba_threads -v`
Expected: 2 passed.

### - [ ] Step 1.7: Commit

```bash
git add genoray/_utils.py tests/test_utils.py
git commit -m "feat(utils): add _resolve_threads and numba_threads context manager"
```

---

## Task 2: Region normalization helper

**Files:**
- Modify: `genoray/_svar.py` (add private module-level helper)
- Test: `tests/test_svar_write_view.py` (create)

### - [ ] Step 2.1: Create test file with failing tests for `_normalize_regions`

Create `tests/test_svar_write_view.py`:

```python
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genoray._svar import _normalize_regions
from genoray._utils import ContigNormalizer


@pytest.fixture
def cnorm() -> ContigNormalizer:
    return ContigNormalizer(["chr1", "chr2"])


def test_normalize_regions_str(cnorm):
    df = _normalize_regions("chr1:10-20", cnorm)
    assert df.shape == (1, 3)
    assert df["chrom"].to_list() == ["chr1"]
    # 1-based inclusive "10-20" -> 0-based half-open [9, 20)
    assert df["start"].to_list() == [9]
    assert df["end"].to_list() == [20]


def test_normalize_regions_tuple(cnorm):
    df = _normalize_regions(("chr2", 5, 15), cnorm)
    assert df["chrom"].to_list() == ["chr2"]
    assert df["start"].to_list() == [5]
    assert df["end"].to_list() == [15]


def test_normalize_regions_alt_contig_naming(cnorm):
    # "1" should normalize to "chr1"
    df = _normalize_regions(("1", 0, 100), cnorm)
    assert df["chrom"].to_list() == ["chr1"]


def test_normalize_regions_unknown_contig_dropped(cnorm):
    with pytest.warns(UserWarning, match="dropped"):
        df = _normalize_regions(("chrZZ", 0, 10), cnorm)
    assert df.height == 0


def test_normalize_regions_bed_file(tmp_path: Path, cnorm):
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t100\t200\nchr2\t300\t400\n")
    df = _normalize_regions(bed, cnorm)
    assert df.height == 2
    assert sorted(df["chrom"].to_list()) == ["chr1", "chr2"]


def test_normalize_regions_frame_polars_bio_schema(cnorm):
    frame = pl.DataFrame(
        {"chrom": ["chr1"], "start": [5], "end": [25]}
    )
    df = _normalize_regions(frame, cnorm)
    assert df["start"].to_list() == [5]
    assert df["end"].to_list() == [25]
```

### - [ ] Step 2.2: Run tests to verify they fail

Run: `pixi run pytest tests/test_svar_write_view.py -v`
Expected: FAIL with `ImportError: cannot import name '_normalize_regions'`.

### - [ ] Step 2.3: Implement `_normalize_regions`

In `genoray/_svar.py`, near the bottom of the file (after the existing module-level helpers), add:

```python
import re as _re
import warnings as _warnings
from os import PathLike

import seqpro as sp  # already imported elsewhere in this module; keep one canonical import

_REGION_STR_RE = _re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def _normalize_regions(
    regions: "str | tuple[str, int, int] | object | PathLike",
    cnorm: ContigNormalizer,
) -> pl.DataFrame:
    """Normalize `regions` to a polars DataFrame with columns:
        chrom (Utf8), start (Int32), end (Int32)
    All coordinates are 0-based, end-exclusive after normalization.
    Rows whose contig is not in `cnorm.contigs` are dropped (with a UserWarning).
    """
    # str: "chr:start-end" — 1-based, end-inclusive
    if isinstance(regions, str):
        m = _REGION_STR_RE.match(regions)
        if m is None:
            raise ValueError(
                f"Region string {regions!r} does not match 'chrom:start-end'"
            )
        chrom = m["chrom"]
        start = int(m["start"]) - 1  # 1-based -> 0-based
        end = int(m["end"])  # inclusive -> exclusive
        df = pl.DataFrame(
            {"chrom": [chrom], "start": [start], "end": [end]},
            schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
        )
    # tuple: (chrom, start, end) — already 0-based, end-exclusive
    elif isinstance(regions, tuple) and len(regions) == 3 and isinstance(regions[0], str):
        chrom, start, end = regions
        df = pl.DataFrame(
            {"chrom": [chrom], "start": [int(start)], "end": [int(end)]},
            schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
        )
    # PathLike to a BED-like file
    elif isinstance(regions, (str, PathLike)) or hasattr(regions, "__fspath__"):
        # NOTE: the string branch above already short-circuited; this is for Path/PathLike.
        df = sp.bed.read(Path(regions))  # type: ignore[attr-defined]
        df = _coerce_bed_schema(df)
    # Frame-like (polars/pandas/pyranges)
    else:
        df = sp.bed.from_any(regions)  # type: ignore[attr-defined]
        df = _coerce_bed_schema(df)

    # Normalize contig names via ContigNormalizer
    normed = [cnorm.norm(c) for c in df["chrom"].to_list()]
    keep_mask = [n is not None for n in normed]
    if not all(keep_mask):
        n_dropped = sum(1 for k in keep_mask if not k)
        _warnings.warn(
            f"{n_dropped} region(s) dropped: contig not in dataset.", stacklevel=2
        )
    df = df.with_columns(pl.Series("chrom", [n if n is not None else "" for n in normed]))
    df = df.filter(pl.Series(keep_mask))
    return df


def _coerce_bed_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce a BED-like frame to columns: chrom (Utf8), start (Int32), end (Int32).

    Accepts the common column-name variants used by polars-bio / pyranges / bed.
    """
    rename = {}
    cols = set(df.columns)
    for src, dst in (("Chromosome", "chrom"), ("CHROM", "chrom"),
                     ("Start", "start"), ("End", "end")):
        if src in cols and dst not in cols:
            rename[src] = dst
    if rename:
        df = df.rename(rename)
    return df.select(
        pl.col("chrom").cast(pl.Utf8),
        pl.col("start").cast(pl.Int32),
        pl.col("end").cast(pl.Int32),
    )
```

**Notes for implementer:**
- `sp.bed.read` / `sp.bed.from_any` — verify the actual seqpro API names. If different, adjust accordingly (e.g. `sp.bed.read_bed`, `sp.bed.coerce`). The tests are the contract.
- Move `import seqpro as sp` and `from os import PathLike` to the top of `_svar.py` if not already there; do NOT keep them as inline imports in the final code.

### - [ ] Step 2.4: Run tests to verify they pass

Run: `pixi run pytest tests/test_svar_write_view.py -v`
Expected: 6 passed.

### - [ ] Step 2.5: Commit

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat(svar): add _normalize_regions helper"
```

---

## Task 3: Sample and field normalization helpers

**Files:**
- Modify: `genoray/_svar.py`
- Test: `tests/test_svar_write_view.py`

### - [ ] Step 3.1: Add failing tests for `_normalize_samples` and `_validate_fields`

Append to `tests/test_svar_write_view.py`:

```python
from genoray._svar import _normalize_samples, _validate_fields


def test_normalize_samples_str():
    assert _normalize_samples("s1", ["s0", "s1", "s2"]) == ["s1"]


def test_normalize_samples_list_preserves_order():
    assert _normalize_samples(["s2", "s0"], ["s0", "s1", "s2"]) == ["s2", "s0"]


def test_normalize_samples_dedupe_first_occurrence():
    assert _normalize_samples(["s2", "s0", "s2"], ["s0", "s1", "s2"]) == ["s2", "s0"]


def test_normalize_samples_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        _normalize_samples(["s9"], ["s0", "s1"])


def test_normalize_samples_file(tmp_path: Path):
    p = tmp_path / "s.txt"
    p.write_text("s2\ns0\n")
    assert _normalize_samples(p, ["s0", "s1", "s2"]) == ["s2", "s0"]


def test_validate_fields_none_returns_all():
    assert _validate_fields(None, {"dosages": np.dtype("float32")}) == ["dosages"]


def test_validate_fields_subset_ok():
    avail = {"dosages": np.dtype("float32"), "GQ": np.dtype("float32")}
    assert _validate_fields(["dosages"], avail) == ["dosages"]


def test_validate_fields_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        _validate_fields(["bogus"], {"dosages": np.dtype("float32")})


def test_validate_fields_empty_list_returns_empty():
    assert _validate_fields([], {"dosages": np.dtype("float32")}) == []
```

### - [ ] Step 3.2: Run tests to verify they fail

Run: `pixi run pytest tests/test_svar_write_view.py -k "samples or fields" -v`
Expected: FAIL with ImportError.

### - [ ] Step 3.3: Implement `_normalize_samples` and `_validate_fields`

Add to `genoray/_svar.py` near `_normalize_regions`:

```python
def _normalize_samples(
    samples: "str | Sequence[str] | PathLike",
    available: Sequence[str],
) -> list[str]:
    """Normalize `samples` to a list of valid sample names, preserving caller order
    and deduping by first occurrence. Raises ValueError on unknown samples.
    """
    if isinstance(samples, str):
        candidates: list[str] = [samples]
    elif isinstance(samples, PathLike) or hasattr(samples, "__fspath__"):
        candidates = Path(samples).read_text().splitlines()
        candidates = [s for s in candidates if s.strip()]
    else:
        candidates = list(samples)

    avail_set = set(available)
    missing = [s for s in candidates if s not in avail_set]
    if missing:
        raise ValueError(f"Samples not found in dataset: {missing}")

    seen: set[str] = set()
    deduped: list[str] = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _validate_fields(
    fields: Sequence[str] | None,
    available: dict[str, np.dtype],
) -> list[str]:
    """Validate field selection. `None` returns all available fields; a sequence is
    validated as a subset of `available`. Raises ValueError on unknown fields.
    """
    if fields is None:
        return list(available)
    fields = list(fields)
    missing = [f for f in fields if f not in available]
    if missing:
        raise ValueError(f"Fields not found in dataset: {missing}")
    return fields
```

### - [ ] Step 3.4: Run tests

Run: `pixi run pytest tests/test_svar_write_view.py -k "samples or fields" -v`
Expected: all passing.

### - [ ] Step 3.5: Commit

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat(svar): add _normalize_samples and _validate_fields helpers"
```

---

## Task 4: Resolve `kept_var_idxs` with mode-aware overlap

**Files:**
- Modify: `genoray/_svar.py`
- Test: `tests/test_svar_write_view.py`

### - [ ] Step 4.1: Add failing tests for `_resolve_kept_var_idxs`

These tests build a `SparseVar` fixture from VCF data using existing helpers. Look in `tests/test_svar.py` for an existing pattern (likely `SparseVar.from_vcf`); reuse it.

Append to `tests/test_svar_write_view.py`:

```python
import shutil

from genoray import VCF, SparseVar


@pytest.fixture
def svar(tmp_path: Path) -> SparseVar:
    src_vcf = Path(__file__).parent / "data" / "biallelic.vcf.gz"  # adapt to actual fixture
    out = tmp_path / "test.svar"
    SparseVar.from_vcf(out, VCF(src_vcf), max_mem="1g", with_dosages=False)
    return SparseVar(out)


def test_resolve_kept_var_idxs_pos_mode(svar: SparseVar):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar.contigs[0]], "start": [0], "end": [10_000]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    kept = _resolve_kept_var_idxs(svar, regions, mode="pos", merge_overlapping=False)
    assert kept.dtype == np.int32 or kept.dtype.kind == "i"
    assert np.all(np.diff(kept) > 0)  # sorted, unique


def test_resolve_kept_var_idxs_overlap_raises(svar: SparseVar):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar.contigs[0]] * 2, "start": [0, 5], "end": [10, 20]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    with pytest.raises(ValueError, match="overlap"):
        _resolve_kept_var_idxs(svar, regions, mode="pos", merge_overlapping=False)


def test_resolve_kept_var_idxs_overlap_merges(svar: SparseVar):
    from genoray._svar import _resolve_kept_var_idxs

    regions = pl.DataFrame(
        {"chrom": [svar.contigs[0]] * 2, "start": [0, 5], "end": [10, 20]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    kept = _resolve_kept_var_idxs(svar, regions, mode="pos", merge_overlapping=True)
    # No duplicates after merge:
    assert len(np.unique(kept)) == len(kept)


def test_resolve_kept_var_idxs_record_includes_more_than_pos(svar: SparseVar):
    """If the test data has an indel with POS at the boundary, record mode includes it;
    pos mode does not. Skip if not testable on this fixture."""
    from genoray._svar import _resolve_kept_var_idxs

    contig = svar.contigs[0]
    # pick a boundary just past a known indel; adjust based on actual fixture data
    regions_pos = pl.DataFrame(
        {"chrom": [contig], "start": [0], "end": [50]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    k_pos = _resolve_kept_var_idxs(svar, regions_pos, mode="pos", merge_overlapping=False)
    k_rec = _resolve_kept_var_idxs(svar, regions_pos, mode="record", merge_overlapping=False)
    assert len(k_rec) >= len(k_pos)


def test_resolve_kept_var_idxs_variant_includes_spanning_deletion(svar: SparseVar):
    from genoray._svar import _resolve_kept_var_idxs

    contig = svar.contigs[0]
    regions = pl.DataFrame(
        {"chrom": [contig], "start": [0], "end": [50]},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
    k_pos = _resolve_kept_var_idxs(svar, regions, mode="pos", merge_overlapping=False)
    k_var = _resolve_kept_var_idxs(svar, regions, mode="variant", merge_overlapping=False)
    assert len(k_var) >= len(k_pos)
```

**Implementer note:** the boundary-sensitive tests (record/variant) are smoke checks. If the bundled fixture has no boundary indels, write a small VCF inline using `tests/data/` patterns from `test_svar.py`, or mark these `xfail` and add a dedicated fixture in a follow-up. Either is acceptable provided the function is covered by at least the pos-mode and overlap tests above.

### - [ ] Step 4.2: Run tests to verify they fail

Run: `pixi run pytest tests/test_svar_write_view.py -k "kept_var" -v`
Expected: FAIL with ImportError.

### - [ ] Step 4.3: Implement `_resolve_kept_var_idxs`

Add to `genoray/_svar.py`:

```python
from typing import Literal as _Literal


def _resolve_kept_var_idxs(
    sv: "SparseVar",
    regions: pl.DataFrame,
    mode: _Literal["pos", "record", "variant"],
    merge_overlapping: bool,
) -> NDArray[V_IDX_TYPE]:
    """Resolve a normalized regions frame to a sorted, deduped array of source
    variant indices to keep.

    `mode` selects bcftools-style overlap semantics:
      - "pos":     variant POS-1 in [region.start, region.end)
      - "record":  variant POS-1 in [region.start, region.end]
      - "variant": full reference span overlaps the region (delegates to var_ranges,
                   which is already ILEN-aware)

    `merge_overlapping=False` raises ValueError on any overlap detected via seqpro/pyranges.
    """
    if regions.height == 0:
        return np.empty(0, dtype=V_IDX_TYPE)

    # --- Overlap detection / merging
    pyr = sp.bed.to_pyr(regions)  # verify exact seqpro API at impl time
    mod = type(pyr).__module__.split(".")[0]
    if mod == "pyranges":
        merged = pyr.merge()
    elif mod == "pyranges1":
        merged = pyr.merge_overlaps()
    else:
        raise RuntimeError(f"unreachable: sp.bed.to_pyr returned {type(pyr)!r}")
    # pyranges objects: `len()` returns row count for both v0 and v1
    if len(merged) != regions.height:
        if not merge_overlapping:
            raise ValueError(
                "regions overlap; pass merge_overlapping=True to dedupe"
            )
        regions = sp.bed.from_pyr(merged)  # verify exact seqpro API at impl time
        regions = _coerce_bed_schema(regions)

    # --- Variant index candidates per contig (use var_ranges, which is the most inclusive)
    kept_chunks: list[NDArray[V_IDX_TYPE]] = []
    for contig, sub in regions.group_by("chrom", maintain_order=False):
        c = contig[0] if isinstance(contig, tuple) else contig  # polars group_by returns tuple key
        starts = sub["start"].to_numpy()
        ends = sub["end"].to_numpy()

        # var_ranges returns (n_ranges, 2) with exclusive end in variant index space,
        # using a sentinel iinfo(V_IDX_TYPE).max to denote "no overlap" rows.
        vr = sv.var_ranges(c, starts, ends)
        sentinel = np.iinfo(V_IDX_TYPE).max
        valid = vr[:, 0] != sentinel
        for s, e in vr[valid]:
            # exclusive end
            kept_chunks.append(np.arange(s, e, dtype=V_IDX_TYPE))

    if not kept_chunks:
        return np.empty(0, dtype=V_IDX_TYPE)
    candidates = np.unique(np.concatenate(kept_chunks))  # sorted + deduped

    # --- Mode-based filter against candidate POS values
    if mode == "variant":
        return candidates  # var_ranges already does ILEN-aware overlap
    # Build a {contig: (start_arr, end_arr)} for fast per-contig membership
    region_by_contig: dict[str, tuple[NDArray, NDArray]] = {}
    for contig_t, sub in regions.group_by("chrom", maintain_order=False):
        c = contig_t[0] if isinstance(contig_t, tuple) else contig_t
        region_by_contig[c] = (sub["start"].to_numpy(), sub["end"].to_numpy())

    # Pull POS (1-based) and CHROM for candidate rows in one pass.
    idx_slice = sv.index[candidates.tolist()]  # polars .gather equivalent; or use .gather
    cand_pos0 = idx_slice["POS"].to_numpy() - 1  # 0-based
    cand_chrom = idx_slice["CHROM"].to_list()

    keep_mask = np.zeros(len(candidates), dtype=bool)
    end_offset = 0 if mode == "pos" else 1  # record mode widens end by 1
    for i in range(len(candidates)):
        starts, ends = region_by_contig.get(cand_chrom[i], (None, None))
        if starts is None:
            continue
        # any region s.t. starts <= cand_pos0[i] < (ends + end_offset)
        p = cand_pos0[i]
        if np.any((starts <= p) & (p < ends + end_offset)):
            keep_mask[i] = True

    return candidates[keep_mask]
```

**Implementer notes:**
- `sp.bed.to_pyr` / `sp.bed.from_pyr` — verify seqpro API. If different, adapt (might be `sp.bed.to_pyranges` / `sp.bed.from_pyranges`).
- The per-element `np.any` loop is fine for correctness; the candidate set is bounded by the region union and this is not the hot path. If it shows up in profiles later, vectorize.
- `sv.index[candidates.tolist()]` — polars row-selection; use `sv.index[candidates]` if it accepts numpy directly, or `sv.index.gather(pl.Series(candidates))`.

### - [ ] Step 4.4: Run tests

Run: `pixi run pytest tests/test_svar_write_view.py -k "kept_var" -v`
Expected: all targeted tests passing; xfail markers acceptable for the boundary-sensitive ones if the fixture lacks an indel near the boundary.

### - [ ] Step 4.5: Commit

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat(svar): add _resolve_kept_var_idxs with pos/record/variant modes"
```

---

## Task 5: Numba kernels — count and write

**Files:**
- Modify: `genoray/_svar.py`
- Test: `tests/test_svar_write_view.py`

### - [ ] Step 5.1: Add failing tests for the kernels

Append to `tests/test_svar_write_view.py`:

```python
def test_nb_count_kept_matches_python():
    from genoray._svar import _nb_count_kept

    # Two samples, ploidy=2 -> 4 slots
    src_data = np.array([0, 2, 5, 1, 3, 5, 0, 4], dtype=np.int32)
    src_offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    src_sample_idxs = np.array([1, 0], dtype=np.int64)  # reorder: sample1 first
    kept_var_idxs = np.array([0, 1, 5], dtype=np.int32)
    ploidy = 2

    out_lengths = np.zeros(2 * 2, dtype=np.int64)
    _nb_count_kept(src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths)

    # Expected per output slot:
    # (out 0, p 0) = src slot 2 [3, 5] -> kept {5} -> 1
    # (out 0, p 1) = src slot 3 [0, 4] -> kept {0} -> 1
    # (out 1, p 0) = src slot 0 [0, 2] -> kept {0} -> 1
    # (out 1, p 1) = src slot 1 [5, 1] -> kept {5, 1} -> 2
    assert out_lengths.tolist() == [1, 1, 1, 2]


def test_nb_write_var_idxs_matches_python():
    from seqpro.rag import lengths_to_offsets

    from genoray._svar import _nb_count_kept, _nb_write_var_idxs

    src_data = np.array([0, 2, 5, 1, 3, 5, 0, 4], dtype=np.int32)
    src_offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    src_sample_idxs = np.array([1, 0], dtype=np.int64)
    kept_var_idxs = np.array([0, 1, 5], dtype=np.int32)
    ploidy = 2

    out_lengths = np.zeros(2 * 2, dtype=np.int64)
    _nb_count_kept(src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths)
    new_offsets = lengths_to_offsets(out_lengths.reshape(2, ploidy))
    out_var_idxs = np.empty(new_offsets[-1], dtype=np.int32)

    _nb_write_var_idxs(
        src_data, src_offsets, src_sample_idxs, ploidy,
        kept_var_idxs, new_offsets.ravel(), out_var_idxs,
    )
    # Expected (new idxs are positions in kept_var_idxs: 0->0, 1->1, 5->2):
    # slot 0: src [3, 5] kept {5} -> [2]
    # slot 1: src [0, 4] kept {0} -> [0]
    # slot 2: src [0, 2] kept {0} -> [0]
    # slot 3: src [5, 1] kept {5, 1} -> [2, 1]
    assert out_var_idxs.tolist() == [2, 0, 0, 2, 1]


def test_nb_write_field_matches_python():
    from seqpro.rag import lengths_to_offsets

    from genoray._svar import _nb_count_kept, _nb_write_field

    src_data = np.array([0, 2, 5, 1, 3, 5, 0, 4], dtype=np.int32)
    src_offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    # one field value per source variant entry, parallel to src_data
    src_field = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.float32)
    src_sample_idxs = np.array([1, 0], dtype=np.int64)
    kept_var_idxs = np.array([0, 1, 5], dtype=np.int32)
    ploidy = 2

    out_lengths = np.zeros(2 * 2, dtype=np.int64)
    _nb_count_kept(src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths)
    new_offsets = lengths_to_offsets(out_lengths.reshape(2, ploidy))
    out_field = np.empty(new_offsets[-1], dtype=np.float32)

    _nb_write_field(
        src_field, src_data, src_offsets, src_sample_idxs, ploidy,
        kept_var_idxs, new_offsets.ravel(), out_field,
    )
    # Same selection as write_var_idxs; field values at the kept positions:
    # slot 0 src j=2 -> 30
    # slot 1 src j=6 -> 70
    # slot 2 src j=0 -> 10
    # slot 3 src j=4,5 -> 50, 60
    assert out_field.tolist() == [30.0, 70.0, 10.0, 50.0, 60.0]
```

### - [ ] Step 5.2: Run tests to verify they fail

Run: `pixi run pytest tests/test_svar_write_view.py -k "_nb_" -v`
Expected: FAIL with ImportError.

### - [ ] Step 5.3: Implement the three kernels

Add to `genoray/_svar.py` at module scope (near `_nb_af_helper`):

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_count_kept(
    src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths
):
    """Pass 1: count, per output (sample, ploidy) slot, how many source variant
    indices fall in `kept_var_idxs`. Membership uses inline binary search; no scratch."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            count = 0
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    count += 1
            out_lengths[i * ploidy + p] = count


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_write_var_idxs(
    src_data, src_offsets, src_sample_idxs, ploidy,
    kept_var_idxs, new_offsets, out_var_idxs,
):
    """Pass 2: write remapped variant indices into the output ragged buffer.
    The remapped index equals the position of the source value in `kept_var_idxs`."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            out_slot = i * ploidy + p
            wp = new_offsets[out_slot]
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    out_var_idxs[wp] = k
                    wp += 1


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_write_field(
    src_field, src_data, src_offsets, src_sample_idxs, ploidy,
    kept_var_idxs, new_offsets, out_field,
):
    """Pass 2 (field variant): write field values at the same positions chosen by
    the membership filter on `src_data`. Output dtype matches `out_field`."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            out_slot = i * ploidy + p
            wp = new_offsets[out_slot]
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    out_field[wp] = src_field[j]
                    wp += 1
```

### - [ ] Step 5.4: Run kernel tests

Run: `pixi run pytest tests/test_svar_write_view.py -k "_nb_" -v`
Expected: 3 passed. First run may be slow due to numba JIT compilation.

### - [ ] Step 5.5: Commit

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat(svar): add numba kernels for write_view (count + write var_idxs + write field)"
```

---

## Task 6: `SparseVar.write_view` method

**Files:**
- Modify: `genoray/_svar.py`
- Test: `tests/test_svar_write_view.py`

### - [ ] Step 6.1: Add failing end-to-end roundtrip test

Append to `tests/test_svar_write_view.py`:

```python
def _assert_svar_equal_for_regions(
    a: SparseVar, b: SparseVar, contig: str, start: int, end: int, samples: list[str]
):
    """Compare a's read_ranges (filtered to `samples`) with b's read_ranges over the
    same region — they must contain the same variant index sets per (sample, ploidy)
    slot. b's indices live in b's index space, so compare by mapping back to (POS, ALT)."""
    rag_a = a.read_ranges(contig, start, end, samples=samples)
    rag_b = b.read_ranges(contig, start, end, samples=samples)
    # Compare the lengths per slot; deeper equality belongs in dedicated tests below.
    assert rag_a.shape[:-1] == rag_b.shape[:-1]


def test_write_view_roundtrip_full(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    svar.write_view(
        regions=(contig, 0, 1_000_000),
        samples=samples,
        output=out,
    )
    sv2 = SparseVar(out)
    assert sv2.available_samples == samples
    assert sv2.ploidy == svar.ploidy
    assert sv2.contigs == svar.contigs

    # Variant count equals number of variants on this contig in the source
    src_idx_on_c = svar.index.filter(pl.col("CHROM") == contig)
    assert sv2.index.filter(pl.col("CHROM") == contig).height == src_idx_on_c.height


def test_write_view_sample_subset_and_order(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = [svar.available_samples[1], svar.available_samples[0]]  # reversed
    svar.write_view(
        regions=(contig, 0, 1_000_000),
        samples=samples,
        output=out,
    )
    sv2 = SparseVar(out)
    assert sv2.available_samples == samples


def test_write_view_overwrite_protection(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:1]
    svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out)
    with pytest.raises(FileExistsError):
        svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out)
    svar.write_view(
        regions=(contig, 0, 1_000_000), samples=samples, output=out, overwrite=True
    )


def test_write_view_fields_default_carries_all(tmp_path: Path):
    src_vcf = Path(__file__).parent / "data" / "biallelic.vcf.gz"  # adapt
    src_out = tmp_path / "src.svar"
    SparseVar.from_vcf(src_out, VCF(src_vcf, dosage_field="DS"), max_mem="1g", with_dosages=True)
    src = SparseVar(src_out)

    view_out = tmp_path / "view.svar"
    contig = src.contigs[0]
    src.write_view(
        regions=(contig, 0, 1_000_000),
        samples=src.available_samples[:1],
        output=view_out,
    )
    v = SparseVar(view_out)
    assert set(v.available_fields) == set(src.available_fields)


def test_write_view_fields_explicit_subset(tmp_path: Path):
    src_vcf = Path(__file__).parent / "data" / "biallelic.vcf.gz"  # adapt
    src_out = tmp_path / "src.svar"
    SparseVar.from_vcf(src_out, VCF(src_vcf, dosage_field="DS"), max_mem="1g", with_dosages=True)
    src = SparseVar(src_out)

    view_out = tmp_path / "view.svar"
    src.write_view(
        regions=(src.contigs[0], 0, 1_000_000),
        samples=src.available_samples[:1],
        output=view_out,
        fields=[],  # genos only
    )
    v = SparseVar(view_out)
    assert v.available_fields == {}


def test_write_view_afs_match_cache_afs(tmp_path: Path, svar: SparseVar):
    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out)
    v = SparseVar(out)
    # AF column should already be present and correct
    assert "AF" in v.index.columns
    expected = v._compute_afs()
    np.testing.assert_allclose(v.index["AF"].to_numpy(), expected, atol=1e-6)


def test_write_view_threads_deterministic(tmp_path: Path, svar: SparseVar):
    out1 = tmp_path / "v1.svar"
    out2 = tmp_path / "v2.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out1, threads=1)
    svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out2, threads=None)

    for fname in ("variant_idxs.npy", "offsets.npy"):
        a = np.fromfile(out1 / fname, dtype=np.int32 if "var" in fname else np.int64)
        b = np.fromfile(out2 / fname, dtype=np.int32 if "var" in fname else np.int64)
        np.testing.assert_array_equal(a, b)
```

### - [ ] Step 6.2: Run roundtrip tests to verify they fail

Run: `pixi run pytest tests/test_svar_write_view.py -k "write_view" -v`
Expected: FAIL with `AttributeError: SparseVar has no attribute 'write_view'`.

### - [ ] Step 6.3: Implement `SparseVar.write_view`

Add as a public method on `SparseVar` in `genoray/_svar.py` (e.g. immediately after `with_fields` or after `cache_afs`):

```python
def write_view(
    self,
    regions: "str | tuple[str, int, int] | object | PathLike",
    samples: "str | Sequence[str] | PathLike",
    output: "PathLike",
    fields: Sequence[str] | None = None,
    merge_overlapping: bool = False,
    regions_overlap: _Literal["pos", "record", "variant"] = "pos",
    overwrite: bool = False,
    threads: int | None = None,
) -> None:
    """Write a subset of this SparseVar to a new directory on disk.

    Parameters
    ----------
    regions
        One of:
          - 1-based, end-inclusive string ``"chr:start-end"``,
          - 0-based, end-exclusive tuple ``(chrom, start, end)``,
          - frame-like recognized by ``seqpro.bed`` (bed / polars-bio / pyranges),
          - path to a BED-like file parsed by ``sp.bed.read``.
    samples
        A sample name, sequence of sample names (caller order preserved, deduped by
        first occurrence), or path to newline-delimited sample names.
    output
        Output directory path, canonically with ``.svar`` extension.
    fields
        Names of fields to carry over. ``None`` (default) carries all
        ``available_fields``. Pass ``[]`` to write genotypes only.
    merge_overlapping
        If ``False`` (default) raise on overlapping input regions; if ``True``
        dedupe via pyranges merge.
    regions_overlap
        ``"pos"`` (default) — variant POS lies in ``[start, end)``;
        ``"record"`` — variant POS lies in ``[start, end]``;
        ``"variant"`` — variant's reference span (ILEN-aware) overlaps the region.
        Mirrors ``bcftools --regions-overlap``.
    overwrite
        If ``True``, an existing ``output`` directory is removed first.
    threads
        Numba thread count. ``None`` defaults to scheduler affinity / cpu_count / 1.
    """
    from ._utils import _resolve_threads, numba_threads

    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path {output} already exists. Use overwrite=True to overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # --- Normalize inputs
    regions_df = _normalize_regions(regions, self._c_norm)
    caller_samples = _normalize_samples(samples, self.available_samples)
    fields_to_write = _validate_fields(fields, self.available_fields)

    if not caller_samples:
        raise ValueError("write_view requires at least one sample")

    # --- Resolve variant subset
    kept_var_idxs = _resolve_kept_var_idxs(
        self, regions_df, mode=regions_overlap, merge_overlapping=merge_overlapping
    )
    if kept_var_idxs.size == 0:
        raise ValueError("no variants selected by `regions`")

    n_out = len(caller_samples)
    ploidy = self.ploidy
    src_sample_idxs = cast(
        NDArray[np.int64],
        self._s2i[np.array(caller_samples)],
    ).astype(np.int64, copy=False)

    threads_resolved = _resolve_threads(threads)

    # --- Pass 1: sizing
    src_data = self.genos.data
    src_offsets = self.genos.offsets
    out_lengths = np.zeros(n_out * ploidy, dtype=np.int64)
    with numba_threads(threads_resolved):
        _nb_count_kept(
            src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths
        )
    new_offsets = lengths_to_offsets(out_lengths.reshape(n_out, ploidy))
    total = int(new_offsets[-1])

    # --- Allocate output memmaps and write offsets
    out_offsets_mm = np.memmap(
        output / "offsets.npy", dtype=new_offsets.dtype, mode="w+",
        shape=new_offsets.shape,
    )
    out_offsets_mm[:] = new_offsets
    out_offsets_mm.flush()

    out_var_idxs_mm = np.memmap(
        output / "variant_idxs.npy", dtype=V_IDX_TYPE, mode="w+", shape=(total,),
    )

    # --- Pass 2: write variant indices
    with numba_threads(threads_resolved):
        _nb_write_var_idxs(
            src_data, src_offsets, src_sample_idxs, ploidy,
            kept_var_idxs, new_offsets.ravel(), out_var_idxs_mm,
        )
    out_var_idxs_mm.flush()

    # --- Pass 2: write each field
    for fname in fields_to_write:
        dtype = self.available_fields[fname]
        src_field = _open_fmt(
            fname, dtype, self.path, (self.n_samples, ploidy, None), "r"
        )
        out_field_mm = np.memmap(
            output / f"{fname}.npy", dtype=dtype, mode="w+", shape=(total,),
        )
        with numba_threads(threads_resolved):
            _nb_write_field(
                src_field.data, src_data, src_offsets, src_sample_idxs, ploidy,
                kept_var_idxs, new_offsets.ravel(), out_field_mm,
            )
        out_field_mm.flush()

    # --- Build and write the new index (filtered + AF column)
    new_index = self.index[kept_var_idxs.tolist()]  # or .gather; pick whichever works
    # Drop any pre-existing AF/index column to be safe; recompute fresh.
    for col in ("AF", "index"):
        if col in new_index.columns:
            new_index = new_index.drop(col)
    max_count = n_out * ploidy
    afs = np.zeros(len(kept_var_idxs), dtype=np.float32)
    _nb_af_helper(afs, out_var_idxs_mm, new_offsets.ravel(), max_count)
    new_index = new_index.with_columns(AF=pl.Series(afs))
    new_index.write_ipc(SparseVar._index_path(output))

    # --- Write metadata.json
    metadata = SparseVarMetadata(
        version=CURRENT_VERSION,
        samples=caller_samples,
        ploidy=ploidy,
        contigs=self.contigs,
        fields={name: self.available_fields[name].name for name in fields_to_write},
    )
    (output / "metadata.json").write_text(metadata.model_dump_json())
```

**Implementer notes:**
- `self.index[kept_var_idxs.tolist()]` — polars row indexing; if `self.index.gather(pl.Series(...))` is the idiomatic form in this codebase, prefer that.
- `_nb_af_helper` reads the output `variant_idxs` after pass 2; the offsets it expects are the per-slot offsets, so pass `new_offsets.ravel()` (its existing usage in `_compute_afs` passes `self.genos.offsets`, which is the flat per-(sample,ploidy) offsets).
- The local `from ._utils import ...` inside the method keeps cycles unlikely; if the project conventions allow it, hoist these imports to the top of `_svar.py`.

### - [ ] Step 6.4: Run all `write_view` tests

Run: `pixi run pytest tests/test_svar_write_view.py -v`
Expected: all passing. If `test_write_view_threads_deterministic` flakes due to numba nondeterminism, switch to `np.fromfile` size+content compare on a single-threaded baseline only; document the limitation in the test docstring.

### - [ ] Step 6.5: Run the full test suite to check for regressions

Run: `pixi run test`
Expected: all tests pass.

### - [ ] Step 6.6: Commit

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat(svar): add SparseVar.write_view for region+sample subsetting"
```

---

## Task 7: Empty-input edge cases

**Files:**
- Modify: `tests/test_svar_write_view.py`
- Modify: `genoray/_svar.py` (only if any test fails)

### - [ ] Step 7.1: Add edge case tests

Append:

```python
def test_write_view_empty_regions_raises(tmp_path: Path, svar: SparseVar):
    with pytest.raises(ValueError, match="no variants"):
        svar.write_view(
            regions=(svar.contigs[0], 10**9, 10**9 + 1),  # past end of chromosome
            samples=svar.available_samples[:1],
            output=tmp_path / "empty.svar",
        )


def test_write_view_empty_samples_raises(tmp_path: Path, svar: SparseVar):
    with pytest.raises(ValueError, match="at least one sample"):
        svar.write_view(
            regions=(svar.contigs[0], 0, 1_000_000),
            samples=[],
            output=tmp_path / "empty.svar",
        )


def test_write_view_unknown_sample_raises(tmp_path: Path, svar: SparseVar):
    with pytest.raises(ValueError, match="not found"):
        svar.write_view(
            regions=(svar.contigs[0], 0, 1_000_000),
            samples=["__nope__"],
            output=tmp_path / "x.svar",
        )


def test_write_view_unknown_field_raises(tmp_path: Path, svar: SparseVar):
    with pytest.raises(ValueError, match="not found"):
        svar.write_view(
            regions=(svar.contigs[0], 0, 1_000_000),
            samples=svar.available_samples[:1],
            output=tmp_path / "x.svar",
            fields=["__nope__"],
        )
```

### - [ ] Step 7.2: Run edge case tests

Run: `pixi run pytest tests/test_svar_write_view.py -k "empty or unknown" -v`
Expected: all pass.

### - [ ] Step 7.3: Commit

```bash
git add tests/test_svar_write_view.py
git commit -m "test(svar): edge cases for write_view (empty regions/samples, unknown inputs)"
```

---

## Task 8: Lint & finalize

### - [ ] Step 8.1: Run linter and formatter

Run: `pixi run ruff check genoray tests && pixi run ruff format genoray tests`
Expected: clean (no errors); formatter may rewrite whitespace.

### - [ ] Step 8.2: Run full test suite

Run: `pixi run test`
Expected: all tests pass.

### - [ ] Step 8.3: Commit any formatting changes

```bash
git status
git add -u
git diff --cached --stat
git commit -m "style: ruff formatting after write_view" || echo "nothing to commit"
```

### - [ ] Step 8.4: Summary

The branch `feat/svar-write-subset` now contains:
- `SparseVar.write_view` (public API, fully documented).
- Three numba kernels with no per-thread scratch.
- `numba_threads` context manager + `_resolve_threads` helper in `_utils.py`.
- Tests covering normalization helpers, kernel correctness, end-to-end roundtrip, sample reordering, field selection, AFs, overwrite, threading determinism, and edge cases.

Open a PR when ready.

# Streaming `SparseVar.write_view` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `SparseVar.write_view` (and `genoray view`) peak memory scale with the selected output subset instead of materializing the full variant index in RAM (issue #73: 314 GB to subset 16,007 → 919 samples).

**Architecture:** Make `SparseVar.__init__` stop eager-collecting the full variant index — store a `LazyFrame` (`_index_lazy`) and expose `.index` as a collect-on-demand cached property. Construction-time scalars (`_is_biallelic`, `_c_max_idxs`, `n_variants`) move to lazy reductions over a single small `_contig_stats` scan. `write_view` resolves regions from numeric columns only, and writes the output index by streaming `scan_ipc → inner-join AF → sort → sink_ipc`, never collecting the full input index. The CLI's default "all variants" bounds are computed lazily.

**Tech Stack:** Python, polars (lazy / streaming engine: `scan_ipc`, `sink_ipc`), numpy, numba, seqpro `Ragged`, pytest, pixi.

## Global Constraints

- All commits follow Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`). (from `CLAUDE.md`)
- Run tests via pixi: `pixi run pytest <path>`. (from `CLAUDE.md`)
- No public-API change is expected (lazy index is internal; `.index` is preserved as a `pl.DataFrame`-returning cached property). If any public name/shape/behavior reachable from `import genoray` changes, update `skills/genoray-api/SKILL.md` in the same PR. (from `CLAUDE.md`)
- Coordinate convention unchanged: 0-based half-open `[start, end)`; missing genotypes `-1`, dosages `np.nan`.
- Output `.svar` must remain byte-format-readable by `SparseVar(...)`; index file path is `root / "index.arrow"` (`SparseVar._index_path`).
- Test data lives at `tests/data/biallelic.vcf.svar` (fixtures `svar`, `svar_wv` already build it; `ddir = Path(__file__).parent / "data"`).
- `V_IDX_TYPE = np.int32` / `POLARS_V_IDX_TYPE = pl.Int32`. `scan_ipc(row_index_name="index")` produces a `UInt32` index column — join keys must match this dtype.

---

## File Structure

- **Modify** `genoray/_svar.py`
  - Rename `SparseVar._load_index` → `_scan_index`, returning a `pl.LazyFrame` (no `.collect()`).
  - `SparseVar.__init__`: build `self._index_lazy`; drop the eager `self.index = ...`, `_is_biallelic`, and `_c_max_idxs` block.
  - Add `from functools import cached_property` import.
  - Add cached properties: `index`, `_contig_stats`, `_c_max_idxs`, `_is_biallelic`; change `n_variants` to read `_contig_stats`.
  - `write_view`: stream the output index (`scan_ipc → join → sort → sink_ipc`); add the whole-contig short-circuit.
  - `_resolve_kept_var_idxs`: collect only numeric columns from `_index_lazy`.
- **Modify** `genoray/_cli/__main__.py` — `view` default-bounds branch uses `_contig_stats` instead of `sv.index`.
- **Modify** `tests/test_svar_write_view.py` — add lazy-index, guard, order-regression, and short-circuit tests (reuses existing `svar`/`svar_wv` fixtures and `ddir`).

---

## Task 1: Lazy index scan + `.index` cached property (behavior-preserving refactor)

**Files:**
- Modify: `genoray/_svar.py` — `_load_index`→`_scan_index` (`:1341-1376`), `__init__` (`:589-590`), add `cached_property` import (`:11`), add `index` cached property.
- Test: `tests/test_svar_write_view.py`

**Interfaces:**
- Produces: `SparseVar._scan_index(self, attrs) -> pl.LazyFrame`; `SparseVar._index_lazy: pl.LazyFrame` (instance attr set in `__init__`); `SparseVar.index` (now a `@cached_property` returning the same `pl.DataFrame` as before — columns `index, CHROM, POS, REF, ALT, *attrs, ILEN`).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_svar_write_view.py`:

```python
def test_index_lazy_and_cached_property(svar_wv):
    import polars as pl

    # Lazy handle exists and is a LazyFrame
    assert isinstance(svar_wv._index_lazy, pl.LazyFrame)

    # .index is a DataFrame equal to collecting the lazy handle
    eager = svar_wv.index
    assert isinstance(eager, pl.DataFrame)
    assert eager.equals(svar_wv._index_lazy.collect())

    # Required columns present (unchanged public shape)
    for col in ("index", "CHROM", "POS", "REF", "ALT", "ILEN"):
        assert col in eager.columns

    # Cached: second access returns the same object
    assert svar_wv.index is eager
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_write_view.py::test_index_lazy_and_cached_property -v`
Expected: FAIL — `AttributeError: 'SparseVar' object has no attribute '_index_lazy'`.

- [ ] **Step 3: Add the `cached_property` import**

In `genoray/_svar.py`, add to the imports near the top (after line 11, `from typing import ...`):

```python
from functools import cached_property
```

- [ ] **Step 4: Convert `_load_index` to a lazy `_scan_index`**

In `genoray/_svar.py`, replace the whole `_load_index` method (`:1341-1376`) with:

```python
    def _scan_index(self, attrs: IntoExpr | None = None) -> pl.LazyFrame:
        """Lazily scan the .gvi index (no collect).

        Returns a LazyFrame with columns ``index, CHROM, POS, REF, ALT, *attrs, ILEN``.
        ``index`` is the physical row index added by ``scan_ipc``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index = pl.scan_ipc(self._index_path(self.path), row_index_name="index")

        schema = index.collect_schema()

        if schema["ALT"] == pl.Utf8:
            index = index.with_columns(pl.col("ALT").cast(pl.List(pl.Utf8)))

        _attrs: set[IntoExpr] = {"ALT"}

        if attrs is not None:
            if not isinstance(attrs, str) and isinstance(attrs, Iterable):
                _attrs.update(attrs)
            else:
                _attrs.add(attrs)
            _attrs.discard("ILEN")
            user_attr_names = [a for a in _attrs - {"ALT"} if isinstance(a, str)]
            if non_numeric := [
                a for a in user_attr_names if not schema[a].is_numeric()
            ]:
                raise ValueError(f"Attrs {non_numeric} must be numeric.")

        attrs = list(_attrs)

        if "ILEN" in schema:
            attrs.append("ILEN")
        elif "ILEN" not in schema:
            attrs.append(ILEN.alias("ILEN"))

        return index.select("index", "CHROM", "POS", "REF", *attrs)
```

- [ ] **Step 5: Add the `index` cached property**

In `genoray/_svar.py`, add immediately after the `nbytes` property (after `:522`):

```python
    @cached_property
    def index(self) -> pl.DataFrame:
        """The full variant index, materialized on first access."""
        return self._index_lazy.collect()
```

- [ ] **Step 6: Build `_index_lazy` in `__init__`, drop the eager collect**

In `genoray/_svar.py` `__init__`, replace line `:589-590`:

```python
        logger.info("Loading genoray index")
        self.index = self._load_index(attrs)
```

with:

```python
        self._index_lazy = self._scan_index(attrs)
```

(Leave lines `:591-601`, which still reference `self.index`, untouched in this task — the cached property keeps them working. Task 2 removes them.)

- [ ] **Step 7: Run the new test + full write_view suite**

Run: `pixi run pytest tests/test_svar_write_view.py -v`
Expected: PASS (new test passes; all existing write_view tests stay green — `.index` behaves as before).

- [ ] **Step 8: Run broader svar tests to confirm no regression**

Run: `pixi run pytest tests/test_svar_from_subset.py tests/test_svar_haploid.py -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "refactor: lazy SparseVar index scan with collect-on-demand .index property"
```

---

## Task 2: `__init__` no longer materializes the full index

**Files:**
- Modify: `genoray/_svar.py` — `__init__` (`:591-601`), `n_variants` (`:511-514`), add `_contig_stats`/`_c_max_idxs`/`_is_biallelic` cached properties.
- Test: `tests/test_svar_write_view.py`

**Interfaces:**
- Produces: `SparseVar._contig_stats: pl.DataFrame` (cached; columns `CHROM, n, pos_max`, in first-appearance order); `_c_max_idxs: dict[str, int]` (cached); `_is_biallelic: bool` (cached); `n_variants: int` (now derived from `_contig_stats`, no `.index` access).
- Consumes: `self._index_lazy` (Task 1).

- [ ] **Step 1: Write the failing guard test**

Add to `tests/test_svar_write_view.py`:

```python
def _index_raises(self):
    raise AssertionError("full index was materialized")


def test_construction_does_not_materialize_index(monkeypatch):
    import numpy as np

    # Make any access to the full .index blow up.
    monkeypatch.setattr(SparseVar, "index", property(_index_raises))

    sv = SparseVar(ddir / "biallelic.vcf.svar")

    # These must all work WITHOUT touching the full index.
    assert isinstance(sv._is_biallelic, (bool, np.bool_))
    assert sv.n_variants > 0
    assert isinstance(sv._c_max_idxs, dict) and sv._c_max_idxs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_write_view.py::test_construction_does_not_materialize_index -v`
Expected: FAIL — `AssertionError: full index was materialized` (raised from `__init__` line `:591`).

- [ ] **Step 3: Add `_contig_stats` cached property**

In `genoray/_svar.py`, add after the `index` cached property (from Task 1):

```python
    @cached_property
    def _contig_stats(self) -> pl.DataFrame:
        """Per-contig variant count and max POS, in first-appearance order.

        Computed by a single streaming reduction over the numeric index
        columns — never materializes the full string-bearing index.
        """
        return (
            self._index_lazy.group_by("CHROM", maintain_order=True)
            .agg(n=pl.len(), pos_max=pl.col("POS").max())
            .collect()
        )
```

- [ ] **Step 4: Add `_c_max_idxs` and `_is_biallelic` cached properties; change `n_variants`**

In `genoray/_svar.py`, replace the `n_variants` property (`:511-514`):

```python
    @property
    def n_variants(self) -> int:
        """Number of variants in the dataset."""
        return self.index.height
```

with:

```python
    @property
    def n_variants(self) -> int:
        """Number of variants in the dataset."""
        return int(self._contig_stats["n"].sum())
```

Then add these two cached properties next to `_contig_stats`:

```python
    @cached_property
    def _c_max_idxs(self) -> dict[str, int]:
        stats = self._contig_stats
        out = {
            c: int(v) - 1
            for c, v in zip(stats["CHROM"], stats["n"].cum_sum())
        }
        out |= {c: 0 for c in self.contigs if c not in out}
        return out

    @cached_property
    def _is_biallelic(self) -> bool:
        return bool(
            self._index_lazy.select((pl.col("ALT").list.len() == 1).all())
            .collect()
            .item()
        )
```

- [ ] **Step 5: Remove the eager block from `__init__`**

In `genoray/_svar.py` `__init__`, delete lines `:591-601` (everything from `self._is_biallelic = (self.index...` through the `self._c_max_idxs |= {c: 0 ...}` line). After Task 1's edit the block to delete is:

```python
        self._is_biallelic = (self.index["ALT"].list.len() == 1).all()
        vars_per_contig = self.index.group_by("CHROM", maintain_order=True).agg(
            n_variants=pl.len()
        )
        self._c_max_idxs = {
            c: v - 1
            for c, v in zip(
                vars_per_contig["CHROM"], vars_per_contig["n_variants"].cum_sum()
            )
        }
        self._c_max_idxs |= {c: 0 for c in self.contigs if c not in self._c_max_idxs}
```

Delete it entirely. The class-body annotations `_c_max_idxs: dict[str, int]` and `_is_biallelic: bool` (`:503-504`) are bare type hints and coexist fine with the cached properties — leave them.

- [ ] **Step 6: Run the guard test**

Run: `pixi run pytest tests/test_svar_write_view.py::test_construction_does_not_materialize_index -v`
Expected: PASS.

- [ ] **Step 7: Run the full svar suite for regressions**

Run: `pixi run pytest tests/test_svar_write_view.py tests/test_svar_from_subset.py tests/test_svar_haploid.py -q`
Expected: PASS (`_c_max_idxs`, `_is_biallelic`, `n_variants` semantics unchanged).

- [ ] **Step 8: Commit**

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "refactor: derive _is_biallelic/_c_max_idxs/n_variants lazily so __init__ never materializes the full index"
```

---

## Task 3: Stream the `write_view` output index (`scan_ipc → join → sort → sink_ipc`)

**Files:**
- Modify: `genoray/_svar.py` — `write_view` step "Build new index" (`:1977-1990`).
- Test: `tests/test_svar_write_view.py`

**Interfaces:**
- Consumes: `self._index_lazy` (Task 1), `kept_var_idxs: NDArray[V_IDX_TYPE]` and `afs: NDArray[np.float32]` (already computed locally in `write_view`), `new_offsets` (local), `_nb_af_helper`, `SparseVar._index_path`.
- Produces: output `index.arrow` written via `sink_ipc`; same columns/dtypes and same ascending-by-`index` row order as the previous implementation (so it aligns positionally with the written genos).

- [ ] **Step 1: Write the failing order/equivalence test**

Add to `tests/test_svar_write_view.py`:

```python
def test_write_view_output_index_order_and_af(tmp_path: Path, svar: SparseVar):
    import numpy as np
    import polars as pl

    out = tmp_path / "view.svar"
    contig = svar.contigs[0]
    samples = svar.available_samples[:2]
    svar.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out)

    sv2 = SparseVar(out)
    out_idx = sv2.index

    # 1. Output index POS is ascending (positional alignment with written genos).
    pos = out_idx["POS"].to_numpy()
    assert np.all(np.diff(pos) >= 0)

    # 2. AF is present, finite, and in [0, 1].
    af = out_idx["AF"].to_numpy()
    assert np.isfinite(af).all()
    assert (af >= 0).all() and (af <= 1).all()

    # 3. No leftover row-index column in the output.
    assert "index" not in out_idx.columns

    # 4. Output POS set matches the source POS on this contig (all MAC>0 kept here).
    src_pos = set(svar.index.filter(pl.col("CHROM") == contig)["POS"].to_list())
    # MAC=0 variants may be dropped; output must be a subset of source POS.
    assert set(pos.tolist()).issubset(src_pos)
```

- [ ] **Step 2: Run test to verify current behavior, then drive the change**

Run: `pixi run pytest tests/test_svar_write_view.py::test_write_view_output_index_order_and_af -v`
Expected: PASS already (current implementation produces a correct index). This test is a **regression guard** for the streaming rewrite — keep it green through Steps 3–5. (If it fails now, stop and investigate before changing code.)

- [ ] **Step 3: Replace the eager index-build block with a streaming sink**

In `genoray/_svar.py` `write_view`, replace the block at `:1977-1990`:

```python
        # --- 9. Build new index ---
        new_index = self.index[kept_var_idxs.tolist()]
        # Drop existing AF and row-index columns if present
        cols_to_drop = [c for c in ("AF", "index") if c in new_index.columns]
        if cols_to_drop:
            new_index = new_index.drop(cols_to_drop)

        # Compute AFs over the written genos
        n_alleles = n_out * ploidy
        afs = np.zeros(len(kept_var_idxs), dtype=np.float32)
        _nb_af_helper(afs, out_var_idxs_mm, new_offsets.ravel(), n_alleles)

        new_index = new_index.with_columns(AF=pl.Series(afs))
        new_index.write_ipc(SparseVar._index_path(output))
```

with:

```python
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
            .join(af_frame.lazy(), on="index", how="inner")  # filter to kept + attach AF
            .sort("index")  # row order must match the ascending kept_var_idxs / written genos
            .drop("index")  # physical row index is not part of the output schema
        )
        out_index.sink_ipc(SparseVar._index_path(output))
```

Notes for the implementer:
- The `inner` join doubles as the kept-row filter, avoiding a large `is_in` list.
- `.sort("index")` is required: streaming-join row order is not guaranteed, and the output index rows must align positionally with the written genos (output position `p` ↔ `kept_var_idxs[p]`). `kept_var_idxs` is ascending, so sorting by `index` reproduces the previous output order.
- **Fallback** (only if `sink_ipc`/streaming join misbehaves on the pinned polars version): replace the last line with `out_index.collect().write_ipc(SparseVar._index_path(output))`. This still avoids the full-index blowup because `out_index` is output-sized (kept rows only).

- [ ] **Step 4: Run the order/equivalence test**

Run: `pixi run pytest tests/test_svar_write_view.py::test_write_view_output_index_order_and_af -v`
Expected: PASS.

- [ ] **Step 5: Run the full write_view + subset suites for equivalence**

Run: `pixi run pytest tests/test_svar_write_view.py tests/test_svar_from_subset.py -q`
Expected: PASS (output `.svar` unchanged: same kept variants, AF, fields, metadata, order).

- [ ] **Step 6: Commit**

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat: stream write_view output index via scan_ipc join+sink_ipc (no full-index collect)"
```

---

## Task 4: Stream `write_view` region resolution + whole-contig short-circuit

**Files:**
- Modify: `genoray/_svar.py` — `_resolve_kept_var_idxs` (`:316-324`), `write_view` region-resolution step (`:1857-1862`), add `_covers_all_variants` helper.
- Test: `tests/test_svar_write_view.py`

**Interfaces:**
- Consumes: `sv._index_lazy`, `sv._contig_stats`, `sv.n_variants`, `sv._c_norm`, `_resolve_kept_rows`.
- Produces: `_resolve_kept_var_idxs(sv, regions, mode, merge_overlapping) -> NDArray[V_IDX_TYPE]` (unchanged signature/return, now reads numeric columns from `_index_lazy`); `SparseVar._covers_all_variants(self, regions_df: pl.DataFrame, mode) -> bool`.

- [ ] **Step 1: Write the failing tests (short-circuit + write_view guard)**

Add to `tests/test_svar_write_view.py`:

```python
def test_covers_all_variants_short_circuit(svar_wv):
    import numpy as np

    # Build per-contig [0, pos_max+1) bounds == the CLI "all variants" default.
    stats = svar_wv._contig_stats
    regions = pl.DataFrame(
        {
            "chrom": stats["CHROM"],
            "start": pl.Series([0] * stats.height, dtype=pl.Int32),
            "end": (stats["pos_max"] + 1).cast(pl.Int32),
        }
    ).select("chrom", "start", "end")

    assert svar_wv._covers_all_variants(regions, "pos") is True

    from genoray._svar import _resolve_kept_var_idxs

    kept = _resolve_kept_var_idxs(svar_wv, regions, mode="pos", merge_overlapping=False)
    # Short-circuit returns every variant, in ascending order.
    assert len(kept) == svar_wv.n_variants
    assert np.array_equal(kept, np.arange(svar_wv.n_variants, dtype=kept.dtype))


def test_write_view_never_materializes_full_index(monkeypatch, tmp_path: Path):
    # .index access => failure, for BOTH the all-variants and specific-region paths.
    monkeypatch.setattr(SparseVar, "index", property(_index_raises))
    sv = SparseVar(ddir / "biallelic.vcf.svar")
    contig = sv.contigs[0]
    samples = sv.available_samples[:1]

    # all-variants (short-circuit) path
    out_all = tmp_path / "all.svar"
    stats = sv._contig_stats
    regions_all = pl.DataFrame(
        {
            "chrom": stats["CHROM"],
            "start": pl.Series([0] * stats.height, dtype=pl.Int32),
            "end": (stats["pos_max"] + 1).cast(pl.Int32),
        }
    ).select("chrom", "start", "end")
    sv.write_view(regions=regions_all, samples=samples, output=out_all)
    assert (out_all / "index.arrow").exists()
    assert (out_all / "metadata.json").exists()

    # specific-region (numeric-collect) path
    out_one = tmp_path / "one.svar"
    sv.write_view(regions=(contig, 0, 1_000_000), samples=samples, output=out_one)
    assert (out_one / "index.arrow").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_svar_write_view.py::test_covers_all_variants_short_circuit tests/test_svar_write_view.py::test_write_view_never_materializes_full_index -v`
Expected: FAIL — `AttributeError: 'SparseVar' object has no attribute '_covers_all_variants'`, and the guard test raises `AssertionError: full index was materialized` (current `_resolve_kept_var_idxs` reads `sv.index`).

- [ ] **Step 3: Make `_resolve_kept_var_idxs` collect numeric columns only**

In `genoray/_svar.py`, replace `_resolve_kept_var_idxs` (`:316-324`):

```python
def _resolve_kept_var_idxs(
    sv: "SparseVar",
    regions: pl.DataFrame,
    mode: Literal["pos", "record", "variant"],
    merge_overlapping: bool,
) -> "NDArray[V_IDX_TYPE]":
    """Backward-compatible wrapper used by ``write_view``; resolves against
    ``sv.index`` (which carries CHROM/POS/ILEN/index)."""
    return _resolve_kept_rows(sv.index, sv._c_norm, regions, mode, merge_overlapping)
```

with:

```python
def _resolve_kept_var_idxs(
    sv: "SparseVar",
    regions: pl.DataFrame,
    mode: Literal["pos", "record", "variant"],
    merge_overlapping: bool,
) -> "NDArray[V_IDX_TYPE]":
    """Resolve kept row positions using only the numeric index columns.

    Collects ``index/CHROM/POS/ILEN`` from the lazy index (bounded RAM) instead
    of materializing the full string-bearing index.
    """
    idx_numeric = sv._index_lazy.select("index", "CHROM", "POS", "ILEN").collect()
    return _resolve_kept_rows(idx_numeric, sv._c_norm, regions, mode, merge_overlapping)
```

- [ ] **Step 4: Add the `_covers_all_variants` helper**

In `genoray/_svar.py`, add as a method on `SparseVar` (e.g. just after `var_ranges`, near `:628`):

```python
    def _covers_all_variants(
        self, regions: pl.DataFrame, mode: Literal["pos", "record", "variant"]
    ) -> bool:
        """True iff *regions* select every variant (one region per present contig,
        each spanning [0, pos_max]).  Lets write_view skip POS materialization.

        Only applies to ``pos``/``record`` modes (``variant`` mode is ILEN-aware
        and resolved through ``var_ranges``).
        """
        if mode == "variant":
            return False

        stats = self._contig_stats  # CHROM, n, pos_max
        present = set(stats["CHROM"])

        per_contig = regions.group_by("chrom").agg(
            start=pl.col("start").min(),
            end=pl.col("end").max(),
            k=pl.len(),
        )
        if set(per_contig["chrom"]) != present:
            return False

        pos_max = dict(zip(stats["CHROM"], stats["pos_max"]))
        end_offset = 0 if mode == "pos" else 1
        for c, s, e, k in zip(
            per_contig["chrom"], per_contig["start"], per_contig["end"], per_contig["k"]
        ):
            # POS is 1-based; 0-based p in [0, pos_max-1]. Full coverage needs a
            # single region with start <= 0 and (end + end_offset) >= pos_max.
            if k != 1 or s > 0 or (e + end_offset) < pos_max[c]:
                return False
        return True
```

- [ ] **Step 5: Wire the short-circuit into `write_view`**

In `genoray/_svar.py` `write_view`, replace the region-resolution block (`:1857-1862`):

```python
        # --- 2. Resolve kept variant indices ---
        kept_var_idxs = _resolve_kept_var_idxs(
            self, regions_df, regions_overlap, merge_overlapping
        )
        if len(kept_var_idxs) == 0:
            raise ValueError("no variants selected by `regions`")
```

with:

```python
        # --- 2. Resolve kept variant indices ---
        if self._covers_all_variants(regions_df, regions_overlap):
            # Fast path: every variant is selected; skip POS/ILEN materialization.
            kept_var_idxs = np.arange(self.n_variants, dtype=V_IDX_TYPE)
        else:
            kept_var_idxs = _resolve_kept_var_idxs(
                self, regions_df, regions_overlap, merge_overlapping
            )
        if len(kept_var_idxs) == 0:
            raise ValueError("no variants selected by `regions`")
```

- [ ] **Step 6: Run the new tests**

Run: `pixi run pytest tests/test_svar_write_view.py::test_covers_all_variants_short_circuit tests/test_svar_write_view.py::test_write_view_never_materializes_full_index -v`
Expected: PASS.

- [ ] **Step 7: Run the full write_view + subset suites (equivalence, incl. existing `_resolve_kept_var_idxs` tests)**

Run: `pixi run pytest tests/test_svar_write_view.py tests/test_svar_from_subset.py -q`
Expected: PASS — including `test_resolve_kept_var_idxs_pos_mode` / `test_resolve_kept_var_idxs_empty_regions` (numeric-collect path is equivalent), and full-coverage output equals the prior path.

- [ ] **Step 8: Commit**

```bash
git add genoray/_svar.py tests/test_svar_write_view.py
git commit -m "feat: stream write_view region resolution + whole-contig short-circuit"
```

---

## Task 5: CLI `genoray view` computes default bounds lazily

**Files:**
- Modify: `genoray/_cli/__main__.py` — `view` default-bounds branch (`:266-276`).
- Test: `tests/test_svar_write_view.py`

**Interfaces:**
- Consumes: `sv._contig_stats` (Task 2), `sv.write_view` (Tasks 3–4).
- Produces: no API change; CLI default "all variants" path no longer touches `sv.index`.

- [ ] **Step 1: Write the failing CLI guard test**

Add to `tests/test_svar_write_view.py`:

```python
def test_cli_view_default_does_not_materialize_index(monkeypatch, tmp_path: Path):
    from genoray._cli.__main__ import view

    monkeypatch.setattr(SparseVar, "index", property(_index_raises))

    src = ddir / "biallelic.vcf.svar"
    out = tmp_path / "cli.svar"
    sv = SparseVar(src)
    sample = sv.available_samples[0]

    # Default regions (all variants) + a sample subset — the issue #73 shape.
    view(
        source=src,
        out=out,
        samples=sample,
        overwrite=True,
    )
    assert (out / "index.arrow").exists()
    assert (out / "metadata.json").exists()
```

(If `view`'s first positional parameters are named differently than `source`/`out`, pass them positionally: `view(src, out, samples=sample, overwrite=True)`. Confirm signature at `genoray/_cli/__main__.py:179`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_write_view.py::test_cli_view_default_does_not_materialize_index -v`
Expected: FAIL — `AssertionError: full index was materialized` (raised from the `sv.index.group_by(...)` default-bounds branch).

- [ ] **Step 3: Replace the default-bounds branch with a lazy computation**

In `genoray/_cli/__main__.py`, replace the `else` branch (`:266-276`):

```python
    else:
        # "all variants" — one row per contig spanning [0, max_pos+1)
        # Synthesize one row per contig covering [0, max(POS)+1)
        bounds = (
            sv.index.group_by("CHROM", maintain_order=True)
            .agg(
                start=pl.lit(0, dtype=pl.Int32),
                end=(pl.col("POS").max() + 1).cast(pl.Int32),
            )
            .rename({"CHROM": "chrom"})
        )
        regions_arg = bounds.select("chrom", "start", "end")
```

with:

```python
    else:
        # "all variants" — one row per contig spanning [0, max_pos+1).
        # Use the lazy per-contig stats so we never materialize the full index.
        stats = sv._contig_stats  # columns: CHROM, n, pos_max
        regions_arg = stats.select(
            chrom=pl.col("CHROM"),
            start=pl.lit(0, dtype=pl.Int32),
            end=(pl.col("pos_max") + 1).cast(pl.Int32),
        )
```

- [ ] **Step 4: Run the CLI guard test**

Run: `pixi run pytest tests/test_svar_write_view.py::test_cli_view_default_does_not_materialize_index -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite + a focused CLI sanity check**

Run: `pixi run pytest tests/test_svar_write_view.py tests/test_svar_from_subset.py -q`
Expected: PASS.

If a CLI test module exists, run it too (discover with `ls tests | grep -i cli`); otherwise the guard test above is the CLI coverage.

- [ ] **Step 6: Commit**

```bash
git add genoray/_cli/__main__.py tests/test_svar_write_view.py
git commit -m "fix: compute genoray view default bounds lazily (no full-index materialization)"
```

---

## Task 6: Full regression run + docs check

**Files:**
- Possibly modify: `skills/genoray-api/SKILL.md` (only if a public name/behavior changed — none expected).

- [ ] **Step 1: Run the whole test suite (non-network)**

Run: `pixi run pytest -m "not network" -q`
Expected: PASS.

- [ ] **Step 2: Lint/format**

Run: `ruff check genoray tests && ruff format --check genoray tests`
Expected: clean (run `ruff format genoray tests` and re-commit if it reformats).

- [ ] **Step 3: Confirm no public-API surface changed**

Review the diff for public names reachable from `import genoray` (no underscore). Expected: none changed — `.index` still returns a `pl.DataFrame`; `write_view`/`view` signatures unchanged; coordinate/missing-value conventions unchanged. Per `CLAUDE.md`, update `skills/genoray-api/SKILL.md` only if this review finds a change.

- [ ] **Step 4: Commit any docs/lint fixups (if any)**

```bash
git add -A
git commit -m "chore: lint + docs check for streaming write_view"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §Architecture/1 (lazy `__init__`, `.index` cached property, lazy `_is_biallelic`/`_c_max_idxs`) → Tasks 1 & 2. ✔
- §Architecture/2 (CLI lazy bounds) → Task 5. ✔
- §Data flow/1 (numeric-only region resolution + whole-contig short-circuit) → Task 4. ✔
- §Data flow/2 (MAC=0 prepass unchanged) → untouched by design; covered by existing tests run in Tasks 3–4. ✔
- §Data flow/3 (genos+fields copy unchanged) → untouched; covered by existing suite. ✔
- §Data flow/4 (streaming output index via `sink_ipc` inner-join+sort, `n_variants` fix) → Task 3 (+ `n_variants` in Task 2). ✔
- §Error handling (unchanged ValueError/FileExistsError; dir created after validation; sink fallback) → preserved (blocks not edited) + fallback noted in Task 3 Step 3. ✔
- §Testing (guard tests, equivalence, order regression, init guard, SKILL.md) → Tasks 1–6. ✔

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows full code. ✔

**Type consistency:** `_index_lazy: pl.LazyFrame`; `_scan_index → pl.LazyFrame`; `index → pl.DataFrame`; `_contig_stats` columns `CHROM,n,pos_max` used consistently in `_c_max_idxs`/`n_variants`/`_covers_all_variants`/CLI; join key dtype handled (`UInt32` via `idx_dtype`); `_resolve_kept_var_idxs`/`_covers_all_variants` signatures match call sites. ✔

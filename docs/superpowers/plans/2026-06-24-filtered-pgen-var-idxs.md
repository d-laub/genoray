# Filtered-PGEN `var_idxs()` Coordinate-Space Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PGEN.var_idxs()` return indices that are consistent with the reader's own filtered `_index`, fixing the out-of-bounds / mis-alignment bug in issue #69.

**Architecture:** The single shared `_var_ranges.var_indices()` helper currently returns the `"index"` column, which holds *physical* (pre-filter, file-global) PVAR row ids. After a filter compacts `_index`, those ids no longer index `_index`. We change `var_indices()` to return *positional* row numbers over the filtered table. PGEN's internal read paths still need physical ids for pgenlib random access, so they go through a new private `_var_idxs_phys()` helper that maps positional → physical via the (still physical) `_index["index"]` column. The public `var_idxs()` returns positional. VCF needs no code change (its `_var_idxs` is private and never gathers `_index`); it gets consistency for free and is covered by a regression test.

**Tech Stack:** Python, polars, numpy, pgenlib, cyvcf2, pytest / pytest-cases, vcfixture (test oracle), plink2 (test-data generation).

## Global Constraints

- Coordinate convention: ranges are 0-based, half-open `[start, end)`. Missing genotypes are `-1`. (Unchanged by this work.)
- `var_idxs()` is a **public** name (reachable via `import genoray` without underscore); its docstring MUST be updated when its contract changes (CLAUDE.md rule). `skills/genoray-api/SKILL.md` must be updated *if* it documents `var_idxs` or its coordinate space.
- Commits follow Conventional Commits (`fix:`, `test:`, `docs:`).
- With **no** filter, positional == physical — existing unfiltered behavior and the existing `tests/test_pgen.py::test_var_idxs` MUST remain green.
- `_index["index"]` stays = physical (do NOT rename it): `_svar.py` (`from_pgen`) and `_c_max_idxs` depend on it.

---

### Task 1: Core fix — `var_indices()` positional + `_var_idxs_phys()` remap + PGEN regression test

**Files:**
- Modify: `genoray/_var_ranges.py:123-148` (`var_indices`)
- Modify: `genoray/_pgen.py` — add `_var_idxs_phys` helper (after `var_idxs`, ~line 443); switch 5 internal call sites at `:479`, `:546`, `:633`, `:718`, `:853`; update `var_idxs` docstring (`:433-437`)
- Test: `tests/test_pgen.py` (add `test_filtered_var_idxs_consistent_with_index`)
- Verify (likely no-op): `skills/genoray-api/SKILL.md`

**Interfaces:**
- Consumes: `genoray.exprs.is_snp` (filter expr); `tests._oracle`; `tests.data.fixtures.FIXTURES["biallelic"]`; existing committed fixtures `tests/data/biallelic.pgen` (+ `.pvar`).
- Produces:
  - `var_indices(idx_dtype, c_norm, var_table, contig, starts, ends) -> (NDArray positional, NDArray offsets)` — return values are now positional row indices into `var_table`.
  - `PGEN.var_idxs(contig, starts, ends) -> (NDArray positional, NDArray offsets)` — positional into `self._index`.
  - `PGEN._var_idxs_phys(contig, starts, ends) -> (NDArray physical, NDArray offsets)` — physical PVAR row ids for pgenlib.

**Background facts (the `biallelic` fixture):**
Records in physical/file order — chr1: (0) `81262 GAT>A`, (1) `81262 G>A`, (2) `81265 T>C`; chr2: (3) `81262 GAT>A`, (4) `81262 G>A`, (5) `81265 T>C`. `exprs.is_snp` (ILEN==0) drops the two `GAT>A` indels (physical 0 and 3). Filtered `_index`: physical `[1,2,4,5]` ↔ positional `[0,1,2,3]`, height 4. A `chr2` query selects positional `[2,3]` (physical `[4,5]`). Pre-fix, `var_idxs("chr2", …)` returns `[4,5]` whose max (5) ≥ height (4) → `_index[[4,5]]` is out of bounds.

- [ ] **Step 1: Write the failing PGEN regression test**

Add to `tests/test_pgen.py` (imports `from genoray import exprs` at top with the other imports if not present; `PGEN`, `Genos`, `POS_MAX` are already imported; `_BIALLELIC`, `ddir` already defined):

```python
def test_filtered_var_idxs_consistent_with_index():
    """Regression for #69: filtered PGEN var_idxs must index its own _index.

    `is_snp` drops the interior GAT>A indels (physical rows 0 and 3), so for a
    filtered reader physical != positional. var_idxs() must return positional
    indices into the (filtered) _index, and reads must still return the correct
    (physical) genotypes.
    """
    from genoray import exprs

    g_filt = PGEN(ddir / "biallelic.pgen", filter=exprs.is_snp)
    g_full = PGEN(ddir / "biallelic.pgen")

    # Filter drops physical rows 0 and 3 -> 4 rows remain.
    assert g_filt._index.height == 4

    # chr2 query: positional [2, 3] (NOT physical [4, 5]).
    vi, offsets = g_filt.var_idxs("chr2", 0, POS_MAX)
    assert int(vi.max()) < g_filt._index.height          # #69: in-bounds
    assert np.array_equal(vi, np.array([2, 3], dtype=vi.dtype))
    assert np.array_equal(offsets, np.array([0, 2], dtype=offsets.dtype))

    # _index[var_idxs] selects the kept chr2 SNPs (POS 81262, 81265).
    assert g_filt._index[vi]["POS"].to_list() == [81262, 81265]

    # Read correctness: a filtered chr1 read returns exactly the same genotypes
    # as the corresponding (physical) variants from an unfiltered read. chr1's
    # SNPs are physical rows [1, 2] -> within the full chr1 read those are the
    # 2nd and 3rd variants on the variant axis.
    filt = g_filt.read("chr1", 0, POS_MAX, mode=Genos)        # (s, p, 2)
    full = g_full.read("chr1", 0, POS_MAX, mode=Genos)        # (s, p, 3)
    assert filt.shape[-1] == 2
    assert np.array_equal(filt, full[..., [1, 2]])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/test_pgen.py::test_filtered_var_idxs_consistent_with_index -v`
Expected: FAIL — `var_idxs("chr2", …)` returns physical `[4, 5]`, so `int(vi.max()) < 4` is False (and/or `g_filt._index[vi]` raises out-of-bounds).

- [ ] **Step 3: Change `var_indices()` to return positional indices**

In `genoray/_var_ranges.py`, replace the `var_table = (...)` block (currently `:123-133`) with a version that assigns a positional row index over the full filtered table *before* the `CHROM` filter and selects it as `"index"`:

```python
    var_table = (
        var_table.lazy()
        .with_row_index("__pos")
        .filter(pl.col("CHROM") == c)
        .select(
            pl.col("__pos").cast(np_to_pl_dtype(idx_dtype)).alias("index"),
            chrom=pl.col("CHROM").cast(pl.Utf8),
            start=pl.col("POS") - 1,
            end=pl.col("POS")
            - pl.col("ILEN").list.first().clip(upper_bound=0).fill_null(0),
        )
    )
```

Rationale: `with_row_index("__pos")` numbers rows `0..N-1` over the entire (already user-filtered) `var_table`, which is exactly the positional space of `_index`. The downstream join/rename/`join["index"]` code is unchanged because we alias the selected column back to `"index"`. The original physical `"index"` column is simply not selected.

- [ ] **Step 4: Add the `_var_idxs_phys` helper to `PGEN`**

In `genoray/_pgen.py`, immediately after the `var_idxs` method (after `:442`), add:

```python
    def _var_idxs_phys(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
    ) -> tuple[NDArray[V_IDX_TYPE], NDArray[OFFSET_TYPE]]:
        """Internal: like :meth:`var_idxs`, but maps the positional indices to
        physical (file-global) PVAR row ids for pgenlib random access. The
        ``_index["index"]`` column holds physical ids; ``var_idxs`` returns
        positions into ``_index`` (issue #69), so reads remap here.
        """
        pos, offsets = self.var_idxs(contig, starts, ends)
        assert self._index is not None
        phys = self._index["index"].to_numpy()[pos].astype(V_IDX_TYPE)
        return phys, offsets
```

(`OFFSET_TYPE`, `ArrayLike`, `NDArray`, `V_IDX_TYPE`, `POS_MAX` are already imported in this module — `var_idxs` uses all of them.)

- [ ] **Step 5: Point the 5 internal read sites at `_var_idxs_phys`**

In `genoray/_pgen.py`, change `self.var_idxs(` to `self._var_idxs_phys(` at exactly these call sites (the public `var_idxs` definition at `:416` and its docstring are NOT touched here):

- `:479` `var_idxs, _ = self.var_idxs(c, start, end)` (in `read`)
- `:546` `var_idxs, _ = self.var_idxs(c, start, end)` (in `chunk`)
- `:633` `var_idxs, offsets = self.var_idxs(c, starts, ends)` (in `read_ranges`)
- `:718` `var_idxs, offsets = self.var_idxs(c, starts, ends)` (in `chunk_ranges`)
- `:853` `var_idxs, offsets = self.var_idxs(c, starts, ends)` (in `_chunk_ranges_with_length`)

Each becomes, e.g.: `var_idxs, offsets = self._var_idxs_phys(c, starts, ends)`. Leave the surrounding code unchanged — every downstream use (`_read_*`, `_sei`-indexing in `_gen_with_length`, `_c_max_idxs`, returned chunk idxs) keeps operating in physical space exactly as before.

- [ ] **Step 6: Update the public `var_idxs` docstring**

In `genoray/_pgen.py`, in the `var_idxs` Returns section (`:433-437`), replace the first return-value description so it states the coordinate space:

```python
        Returns
        -------
            Shape: (tot_variants). Variant indices for the given ranges, as
            0-based **positions into this reader's (filtered) ``_index``** — i.e.
            ``reader._index[var_idxs]`` is always valid. With no filter these
            equal the physical PVAR row order.

            Shape: (ranges+1). Offsets to get variant indices for each range.
```

- [ ] **Step 7: Run the new test to verify it passes**

Run: `pixi run pytest tests/test_pgen.py::test_filtered_var_idxs_consistent_with_index -v`
Expected: PASS.

- [ ] **Step 8: Run the existing PGEN suite to confirm no regressions**

Run: `pixi run pytest tests/test_pgen.py tests/test_parity.py -q`
Expected: PASS (including the unchanged `test_var_idxs` — with no filter, positional == physical).

- [ ] **Step 9: Run the svar suite (shares `_index["index"]` semantics)**

Run: `pixi run pytest tests/test_svar.py tests/test_svar_filtering.py tests/test_svar_from_subset.py -q`
Expected: PASS — `_svar.py` reads `_index["index"]` (still physical) and never calls `var_idxs()`, so it is unaffected.

- [ ] **Step 10: Verify `skills/genoray-api/SKILL.md`**

Run: `grep -n "var_idxs" skills/genoray-api/SKILL.md`
Expected: no matches. `var_idxs` is not documented in the skill, so no SKILL.md edit is required (the docstring in Step 6 is the public-facing contract). If the grep *does* return matches, update those lines to state that `var_idxs` returns positional indices into the reader's filtered `_index`.

- [ ] **Step 11: Commit**

```bash
git add genoray/_var_ranges.py genoray/_pgen.py tests/test_pgen.py
git commit -m "fix: filtered PGEN var_idxs returns positional indices into _index (#69)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: VCF regression guard

**Files:**
- Test: `tests/test_vcf.py` (add `test_filtered_var_idxs_consistent_with_index`)

**Interfaces:**
- Consumes: `VCF`; `genoray.exprs.is_snp`; committed fixture `tests/data/biallelic.vcf.gz`; `POS_MAX` (import from `genoray._vcf`).

**Why no VCF code change:** `VCF._var_idxs` is private and never used to gather `_index`; reads stream by genomic position via cyvcf2 and apply the per-record `filter` in `_fill_genos`, counting via `var_counts`. The shared `var_indices()` change (Task 1) makes `_var_idxs` positional too — this test guards that consistency and proves filtered reads are correct. The `_var_idxs` in-bounds assertion below would have FAILED before Task 1 (it returned physical `[4,5]` into a height-4 index); it passes after.

- [ ] **Step 1: Write the VCF regression test**

Add to `tests/test_vcf.py` (`VCF` and `_BIALLELIC`/`ddir` are already imported/defined; add `POS_MAX` to the `from genoray._vcf import ...` line):

```python
def test_filtered_var_idxs_consistent_with_index():
    """Regression guard for #69 on the VCF backend.

    VCF._var_idxs is private and never gathers _index, but it shares
    var_indices() with PGEN, so it must also return positional indices. Filtered
    reads must return the same genotypes as the matching unfiltered variants.
    """
    from genoray import exprs

    # is_snp as a (cyvcf2 record callable, pl.Expr) pair (VCF requires both).
    record_is_snp = lambda rec: len(rec.REF) == 1 and all(len(a) == 1 for a in rec.ALT)

    v_filt = VCF(
        ddir / "biallelic.vcf.gz",
        filter=record_is_snp,
        pl_filter=exprs.is_snp,
    )
    v_filt._write_gvi_index()
    v_filt._load_index()
    v_full = VCF(ddir / "biallelic.vcf.gz")

    assert v_filt._index.height == 4

    # _var_idxs is positional and in-bounds (pre-#69 it returned physical [4,5]).
    vi, _ = v_filt._var_idxs("chr2", 0, POS_MAX)
    assert int(vi.max()) < v_filt._index.height
    assert np.array_equal(vi, np.array([2, 3], dtype=vi.dtype))
    assert v_filt._index[vi]["POS"].to_list() == [81262, 81265]

    # Filtered chr1 read == the SNP variants (physical [1, 2]) of the full read.
    filt = v_filt.read("chr1", 0, POS_MAX, mode=VCF.Genos16)
    full = v_full.read("chr1", 0, POS_MAX, mode=VCF.Genos16)
    assert filt.shape[-1] == 2
    assert np.array_equal(filt, full[..., [1, 2]])
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pixi run pytest tests/test_vcf.py::test_filtered_var_idxs_consistent_with_index -v`
Expected: PASS (Task 1 already made `_var_idxs` positional; VCF read was already filter-correct).

- [ ] **Step 3: Run the full VCF suite**

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_vcf.py
git commit -m "test: guard filtered VCF var_idxs consistency (#69)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Full suite + issue cross-reference

**Files:** none (verification + bookkeeping).

- [ ] **Step 1: Run the entire test suite**

Run: `pixi run test`
Expected: PASS (this also regenerates test data via `gen_from_vcf.sh`; the committed fixtures are unchanged by this work).

- [ ] **Step 2: If `pixi run test` is too slow locally, run the targeted set**

Run: `pixi run pytest -m "not network" -q`
Expected: PASS.

- [ ] **Step 3 (optional, if pushing a PR): note the fix on the issue**

When the branch is pushed and a PR opened, reference `Fixes #69` in the PR body so the issue closes on merge. (Do not push or open the PR unless the user asks.)

---

## Self-Review

**1. Spec coverage:**
- Spec §"Chosen approach" / §1 (`var_indices` positional) → Task 1 Step 3. ✔
- Spec §2 (PGEN positional→physical remap at read boundary) → Task 1 Steps 4-5 (`_var_idxs_phys` + 5 sites; spec said "four" entry points — corrected to the 5 actual internal call sites, including `_chunk_ranges_with_length`, which also feeds physical-indexed `_sei`/`_c_max_idxs`). ✔
- Spec §3 (VCF verify-only) → Task 2. ✔
- Spec §4 (docstring mandatory; SKILL.md only if referenced) → Task 1 Steps 6 & 10. ✔
- Spec §Testing (in-bounds, index-alignment, read-correctness; VCF read + in-bounds) → Task 1 Step 1 / Task 2 Step 1. Read-correctness uses filtered-vs-unfiltered-slice comparison (stronger and not sensitive to half-call/phase oracle decoding) plus POS alignment via the index. ✔
- Spec §"Unchanged" (unfiltered `test_var_idxs` stays green) → Task 1 Step 8. ✔
- Spec §Out of scope (gvl guard; no new fixtures) → respected; reuses committed `biallelic` fixtures. ✔

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows full code. ✔

**3. Type consistency:** `var_indices` returns `(NDArray, NDArray offsets)`; `var_idxs` returns positional `(NDArray[V_IDX_TYPE], NDArray[OFFSET_TYPE])`; `_var_idxs_phys` returns physical with the same signature; all 5 call sites unpack `(idxs, offsets)`. `_index["index"]` referenced consistently as physical throughout. ✔

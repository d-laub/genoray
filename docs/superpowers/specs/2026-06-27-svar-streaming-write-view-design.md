# Streaming `SparseVar.write_view` — Design

**Issue:** [#73](https://github.com/d-laub/genoray/issues/73) — `genoray view` sample-subsetting peak RAM scales with the full input dataset, not the selected subset (314 GB to subset 16,007 → 919 samples; output `.svar` is only ~404 MB).

**Date:** 2026-06-27

---

## Problem

`write_view`'s **data** path is already streaming: input (`variant_idxs.npy`, `offsets.npy`, each `{field}.npy`) is memmapped (`_open_genos`/`_open_fmt`), the numba kernels (`_nb_count_mac_per_kept`, `_nb_count_kept`, `_nb_write_var_idxs`, `_nb_write_field`) read element-by-element, and output is written through `np.memmap`. Peak RAM for the variant payload is bounded by the output.

The **variant index** is not. Two things stack up:

1. **`SparseVar.__init__` (`_svar.py:590`) eagerly runs `_load_index(...).collect()`** — materializing every variant row, including the heavy string columns `REF` and `ALT` (as `List[Utf8]`) and `CHROM`, into RAM. For a 338M-variant cohort, decompressing the ~24 GB zstd Arrow index into polars' in-memory representation is the bulk of the 314 GB. The process dies during *"Loading genoray index"*, before `write_view` runs.
2. Inside `write_view`, `self.index[kept_var_idxs.tolist()]` (`_svar.py:1978`) makes a **second full-width copy** of the kept slice; `_is_biallelic` (`:591`) and region resolution each scan the full frame. These transient copies push peak above the base materialization.

The late MAC=0 filter warning is a symptom, not the root cause. The root cause is **eager full-index materialization with string columns**, when the operation only needs:

- numeric/small columns (`index`, `CHROM`, `POS`, `ILEN`) to map regions → variant indices, and
- the heavy `REF`/`ALT` strings **only for the final kept set** (post-region, post-MAC), which for a sample subset is a tiny fraction of the cohort.

## Goal

Peak memory for `write_view` and `genoray view` scales with the **selected output**, not the full input. No change to output bytes/semantics. No public-API change (`.index` preserved).

## Non-goals

- Making `self.index` lazy across *all* of `SparseVar` (`read_ranges`, `chunk`, `mutation_matrix`, …). Those keep today's behavior (collect-on-demand). Only the view path is made streaming.
- Changing the on-disk `.svar` layout or the index Arrow format.

---

## Architecture

Two independent changes, **both required** — the CLI re-touches the full index, so fixing only `write_view` would not fix #73.

### 1. `SparseVar.__init__` stops eager-collecting the full index

- Store `self._index_lazy = pl.scan_ipc(self._index_path(self.path), row_index_name="index")`, carrying the existing `ALT`→`List[Utf8]` cast and `ILEN` derivation **lazily** (the current `_load_index` logic, minus the final `.collect()`). `row_index_name="index"` adds the physical row index at scan time, matching today.
- `self.index` becomes a `@cached_property` that collects the full frame — preserving today's behavior and public surface for every existing consumer (`read_ranges`, `chunk`, etc.). On the huge cohort this property is simply never accessed by the view path.
- `_is_biallelic` and `_c_max_idxs` move off the eager frame onto lazy reductions / cached properties computed by streaming scalars/tiny frames out of the engine — never materializing 338M string rows at construction:
  - `_is_biallelic`: `self._index_lazy.select((pl.col("ALT").list.len() == 1).all()).collect().item()`
  - `_c_max_idxs`: `self._index_lazy.group_by("CHROM").agg(pl.len(), pl.col("POS").max()).collect()` → reduce to the same `{contig: max_idx}` mapping as today (preserving contig order via the existing `cum_sum` logic).
  - Both are computed lazily (cached properties) so construction itself triggers no full materialization.

### 2. CLI `genoray view` (`_cli/__main__.py`)

The default "all variants" branch currently does
`sv.index.group_by("CHROM", maintain_order=True).agg(start=0, end=POS.max()+1)`
which forces a full collect. Replace it with the lazy equivalent off `_index_lazy` (or reuse `_c_max_idxs` + a per-contig POS-max from the same lazy reduction), so the CLI never triggers a full collect.

---

## Data flow in `write_view` (peak ∝ output)

1. **Region resolution.** Collect only `[index, CHROM, POS, ILEN]` from `_index_lazy` for region → index mapping (bounded; ~5–10 GB worst case even for the whole genome). `_resolve_kept_rows` is otherwise unchanged.
   - **Whole-contig short-circuit:** when `regions` cover entire contigs (the CLI sample-subset default — `[0, maxPOS+1)` per contig), skip `var_ranges` and take `kept = arange` per contig from `_c_max_idxs`. The common sample-subset case then materializes **zero** numeric columns for region resolution.
2. **MAC=0 prepass.** Unchanged — runs on memmaps via `_nb_count_mac_per_kept`; narrows `kept_var_idxs`. Same warning and the same all-MAC-0 `ValueError`.
3. **Genos + fields copy.** Unchanged — already streaming via memmaps (`_nb_count_kept`, `_nb_write_var_idxs`, `_nb_write_field`) into memmapped output.
4. **Output index — fully streaming, no full collect, no output-sized collect.** Recompute `AF` over the written genos (`_nb_af_helper` over `out_var_idxs_mm`) into a small, output-sized frame keyed by the kept indices:
   ```python
   af_frame = pl.DataFrame({"index": kept_var_idxs, "AF": afs})
   (
       pl.scan_ipc(index_path, row_index_name="index")
         .join(af_frame.lazy(), on="index", how="inner")  # inner join = filter to kept + attach AF
         .drop(cols_to_drop)                               # drop pre-existing AF (and any stale "index" col), as today
         .sort("index")                                    # row order must == written-genos order
         .sink_ipc(out_index_path)
   )
   ```
   - The **inner join** doubles as the row filter (only kept rows survive) and the AF attach, in one streaming pass — avoiding a large `is_in(kept)` list.
   - The **`.sort("index")` is required**: streaming-join row order is not contractual ([[polars]] join-order caveat), and the output index rows must align *positionally* with the written genos (output variant position `p` ↔ `kept_var_idxs[p]`). `kept_var_idxs` is ascending (sorted by `_resolve_kept_rows`, order preserved by the MAC mask), so sorting the joined frame by `index` yields exactly the written order. Polars' streaming sort spills to disk, keeping RAM bounded.
   - The output index must carry the same columns/dtypes as today's `write_view` output (drop the old `AF`/`index` columns, write the new `AF`). Confirm the `ALT`/`ILEN` lazy schema matches the eager schema written previously.

---

## Error handling

Semantics unchanged:

- `ValueError` when `regions` select no variants, and when all candidates have MAC=0 in the subset.
- `FileExistsError` when `output` exists without `overwrite=True`.
- `mutcat`-without-`reference` `ValueError` unchanged.
- Output directory is still created only **after** all validation, so no partial directory on error.

**Fallback:** if `sink_ipc` + streaming inner-join is problematic on the pinned polars version, collect **only the kept (output-sized) subset** and write with `write_ipc` — still cheap, since the kept subset is output-sized (~hundreds of MB, not the full cohort). The disaster being eliminated is collecting the *full* input index, which this design removes regardless of the sink-vs-collect choice for the kept rows.

---

## Testing

Per the chosen verification strategy (guard test that the streaming path never materializes the full index):

- **Guard — `write_view`:** monkeypatch the `SparseVar.index` cached property to raise, then run `write_view` end-to-end on a fixture and assert it completes — proving the view path never collects the full index.
- **Guard — CLI:** same monkeypatch, drive the `view` command (default all-variants + sample subset) and assert success.
- **Guard — construction:** constructing `SparseVar` does not access the `.index` cached property (assert via the same raising monkeypatch around `SparseVar(path)`).
- **Equivalence:** existing `tests/test_svar_write_view.py` and `tests/test_svar_from_subset.py` stay green — output `.svar` identical to today (same kept variants, same `AF`, same row order, same fields/metadata).
- **Order regression:** explicitly assert the output index row order matches the written-genos order (e.g. output `read_ranges` variant set/positions align per slot), guarding the `.sort("index")` requirement.

## Docs / skill impact

No expected public-API change: lazy index is internal, `.index` is preserved as a collect-on-demand cached property. Per `CLAUDE.md`, re-check during implementation; update `skills/genoray-api/SKILL.md` only if a public name, return shape, or documented behavior changes.

## File touchpoints (anticipated)

- `genoray/_svar.py` — `SparseVar.__init__`, `_load_index` (split into lazy builder), `_is_biallelic`/`_c_max_idxs` → cached properties, `write_view` step 1 (region resolution + short-circuit) and step 9 (streaming output index), add `index` cached property.
- `genoray/_cli/__main__.py` — `view` default-bounds branch → lazy.
- `tests/test_svar_write_view.py`, `tests/test_svar_from_subset.py` — guard + order tests.

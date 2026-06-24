# Fix #69 — Consistent coordinate space for filtered PGEN `var_idxs()`

**Date:** 2026-06-23
**Issue:** [#69](https://github.com/d-laub/genoray/issues/69)
**Status:** Approved design, pending implementation

## Problem

For a **filtered** `PGEN`, `var_idxs()` returns variant indices in the
**unfiltered / physical** (file-global) coordinate space, while `_index` is the
**filtered / compacted** table. The two no longer share an index space, so a
consumer that does `_index[var_idx]` (or indexes any array aligned to `_index`
with values from `var_idxs()`) goes out of bounds or reads the wrong row.

This silently breaks downstream consumers. GenVarLoader's `gvl.write()` hit it as
an `IndexError` when reading a filtered PGEN.

### Root cause

In `_var_ranges.var_indices()` the returned indices come from the `"index"`
column of the variant table. That column is assigned by
`pl.scan_ipc(..., row_index_name="index")` **before** the user filter is applied
(`_pgen._load_index`, `_vcf._load_index`). After `index.filter(filter)` the table
is compacted to N rows, but the `"index"` column still holds the original
physical/global row ids (max can exceed N-1). So `var_idxs()` returns physical
ids while `_index` has N positional rows.

### Why the physical ids are load-bearing (constraint on the fix)

The physical ids are **not** simply wrong to compute — they are required
internally:

- `PGEN._read_genos` / `_read_dosages` / `_read_genos_phasing` / etc. pass
  `var_idxs` straight to pgenlib's `read_alleles_list` / `read_dosages_list`,
  which address the underlying `.pgen` by **physical file row id**
  (`genoray/_pgen.py:945`).
- `_svar.py` `from_pgen` reads `pgen._index["index"]` as the physical id array
  (`phys`) for pgenlib random access (`genoray/_svar.py:1192`, comment at
  1219-1221). It does **not** call `var_idxs()`.

So the fix must keep a physical mapping available internally while making the
**public** `var_idxs()` contract consistent with `_index`.

## Chosen approach (issue Option 1)

`var_idxs()` returns **positional** indices into the reader's own (filtered)
`_index` — the least-surprising contract: "indices from a reader index that
reader's own `_index`."

Crucially, because polars `_index[idx_array]` gathers **by row position**, the
`"index"` column does **not** need to be renamed or recomputed. We keep
`"index"` = physical (so `_svar.py` and `_c_max_idxs` are untouched) and change
only what `var_indices()` returns.

### Rejected alternatives

- **Option 2** (keep `_index` as the full/unfiltered table, expose a mask):
  large blast radius — much internal code assumes `_index` is already filtered
  (sample reads, contig offsets, chunking).
- **Option 3** (keep `var_idxs()` physical, add an explicit physical→filtered
  mapping helper): least invasive but leaves the footgun — `_index[var_idxs]`
  stays wrong unless the caller knows to map first.

## Changes

### 1. `genoray/_var_ranges.py` — `var_indices()` returns positional indices

Assign a fresh row index over the **full** (already-filtered) `var_table`
*before* the `CHROM == c` filter, and return that positional column instead of
`pl.col("index")`. The positional index is over the entire filtered `_index`, so
it correctly gathers `_index` rows across all contigs.

Order guarantee: `_index` is built by `scan_ipc` in file order then filtered
(stable), so positional `k` ↔ the k-th kept variant in file order, and
`_index["index"][k]` is its physical id.

This function is shared by `PGEN.var_idxs` and `VCF._var_idxs`; both become
positional from this one change (see §3).

### 2. `genoray/_pgen.py` — map positional → physical at the read boundary

In each of the four read entry points — `read`, `chunk`, `read_ranges`,
`chunk_ranges` — after `var_idxs, ... = self.var_idxs(...)`, compute:

```python
phys = self._index["index"].to_numpy()[var_idxs]
```

and pass `phys` into the `_read_*` helpers. The `_read_*` helper signatures keep
their existing "physical indices" meaning (no change there). Returned `offsets`
are unchanged. For `chunk` / `chunk_ranges`, map before chunk-splitting
(positional→physical is elementwise, so splitting either array is equivalent).

**Invariant:** with no filter, positional == physical, so unfiltered reads and
the existing `test_var_idxs` are unchanged.

### 3. `genoray/_vcf.py` — no code change (verify only)

VCF is structurally immune to the original bug: `_var_idxs` is **private** and
unused in the read path; reads stream by position via cyvcf2, apply the
per-record `_filter` in `_fill_genos`, and count via `var_counts`. The shared
`var_indices()` change makes `_var_idxs` positional too — a free consistency
win, asserted by a test.

### 4. Public API docs

- Update the `PGEN.var_idxs` docstring: indices are 0-based **positions into the
  reader's filtered `_index`**, not file-global ids.
- `skills/genoray-api/SKILL.md` does not currently document `var_idxs` or its
  coordinate space; update it only if the implementation decides to surface this
  name. The docstring change is mandatory regardless (per CLAUDE.md public-name
  rule).

## Testing (TDD — tests written first, must fail before the fix)

Reuse the existing `biallelic` fixture + `exprs.is_snp`, which drops the interior
`GAT→A` indels at **physical rows 0 and 3** (one per contig), making physical ≠
positional. The vcfixture `GroundTruth` oracle (`tests/_oracle.py`) supplies
expected values.

### `tests/test_pgen.py` — filtered PGEN

1. **In-bounds (direct #69 assertion):** for a chr2 query,
   `var_idxs().max() < g._index.height` (pre-fix returns physical `[4, 5]` into a
   4-row `_index`).
2. **Index alignment:** `g._index[var_idxs]["POS"]` matches the oracle's kept
   variants.
3. **Read correctness:** `read()` / `read_ranges()` genotypes equal
   `_oracle.genos(truth, physical_idx)` for the kept variants — proves the
   positional→physical remap is correct, not merely in-bounds.

### `tests/test_vcf.py` — filtered VCF

- Filtered `read()` returns oracle-correct genotypes.
- `_var_idxs` values are in-bounds positional (`< _index.height`).

### Unchanged

- Existing unfiltered `test_var_idxs` stays green (positional == physical).

## Out of scope

- gvl-side guard (separate repo; cross-ref gvl PR #241). This spec fixes the root
  cause in genoray only.
- Adding new fixture data or a mixed SNP/SV fixture (the existing `biallelic`
  fixture + `is_snp` is sufficient to reproduce physical ≠ positional).

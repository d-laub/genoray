# Fix `annotate_mutations` 1-based POS off-by-one (issue #59)

**Date:** 2026-06-12
**Issue:** [#59](https://github.com/d-laub/genoray/issues/59)
**Status:** Approved design

## Problem

`SparseVar.annotate_mutations` passes the variant index to `classify_variants`
without converting `POS` from 1-based to 0-based. `SparseVar.index.POS` is stored
**1-based** (preserved from the source VCF, and documented as such in
`skills/genoray-api/SKILL.md:188`), but `classify_variants` uses `POS` directly as
a 0-based reference coordinate. Every variant's reference context is therefore
fetched one base to the right.

Two consequences:

1. **Silent corruption:** SNV trinucleotide context and indel repeat context are
   computed at the wrong position, so `mutation_matrix("SBS96")` / `assign_signatures`
   return a systematically shifted spectrum rather than erroring.
2. **Crash:** for a deletion whose deleted unit is not tandem-repeated immediately
   downstream, the off-by-one makes the repeat scan return `n_rep == 0`, so
   `classify_id83` computes `_repeat_bucket(n_rep - 1) = _repeat_bucket(-1) = -1` and
   raises `KeyError: '<size>:Del:R:-1'` (valid buckets are `R:0`..`R:5`).

### Root cause (confirmed)

`classify_variants` (`genoray/_mutcat.py:468`) is the only place `POS` is used as an
absolute coordinate — the SNV trinucleotide fetch (`:491-492`) and the ID-83
downstream repeat scan in `classify_id83` (`scan_start = pos + 1`, `:254`). Its two
callers disagree on convention:

- `annotate_mutations` passes the 1-based `self.index` (`genoray/_svar.py:1491`).
- `tests/test_mutcat_calibration.py:294` pre-subtracts 1 before calling.

The adjacency kernel `_entry_codes_kernel` (`genoray/_mutcat.py:321`) uses only
*relative* POS deltas (`var_pos[w] - var_pos[v] == 1`), so DBS detection is
unaffected by the convention.

## Design decisions

- **Single source of truth:** genoray's index `POS` is 1-based everywhere.
  `classify_variants` is the boundary to 0-based reference coordinates and performs
  the conversion. (Confirmed with user.)
- **Guard behavior:** a deletion with `n_rep == 0` indicates `REF` disagrees with the
  reference genome (a serious "wrong reference" condition). Return `UNCLASSIFIED` and
  emit a single aggregated warning rather than fabricating a plausible-but-wrong
  classification. (Confirmed with user.)
- **No public API change.** `classify_variants` is private (`_mutcat`). The fix makes
  behavior match the already-documented `SKILL.md` convention, so no SKILL.md update
  is required.

## Changes

### Change 1 — convert POS in `classify_variants` (the single source of truth)

In `classify_variants` (`genoray/_mutcat.py`, around `:488`):

```python
p = int(pos[i]) - 1   # index POS is 1-based (VCF convention); reference.fetch is 0-based
```

- Update the docstring (`:471`) from *"POS (0-based int)"* to
  *"POS (1-based, VCF convention)"*.
- Remove the manual `- 1` conversion in `tests/test_mutcat_calibration.py:294`
  (it would otherwise double-subtract).

This single change corrects both the SNV trinucleotide context and the indel repeat
scan, because both derive their reference coordinates from `p`.

### Change 2 — defensive guard + aggregated wrong-reference warning

After Change 1, a correctly-anchored deletion always has `n_rep >= 1`: the deleted
unit `REF[1:]` is by definition present in the reference at `scan_start = pos + 1`
when `REF` matches the reference. Therefore `n_rep == 0` for a deletion is equivalent
to `REF` disagreeing with the reference genome.

- In `classify_id83` (`genoray/_mutcat.py:234`): when `is_del and n_rep == 0`, return a
  module-private sentinel `_REF_MISMATCH` instead of evaluating `_repeat_bucket(n_rep - 1)`.
  This covers both the 1 bp deletion branch (`:273`) and the ≥2 bp deletion branch
  (`:283`). `_REF_MISMATCH` is **not** added to the public `SENTINELS` dict; it is an
  internal boundary signal whose value must not collide with any valid code or existing
  sentinel (e.g. `-99`).
- In `classify_variants` (`genoray/_mutcat.py:468`): translate `_REF_MISMATCH` to
  `SENTINELS["UNCLASSIFIED"]` in the output array and increment a counter. After the
  loop, if the counter is > 0, emit a single `loguru` warning summarizing the count and
  a few example `CHROM:POS` locations, e.g.:

  > `{n}/{total} deletions have REF disagreeing with the reference genome at their position (e.g. chr1:100, chr2:55) — wrong reference build?`

  Aggregation (one warning, not one-per-variant) is required to avoid log-flooding on
  genome-wide data (the issue was found on a ≈348M-variant `.svar`).

This requires adding `from loguru import logger` to `genoray/_mutcat.py`.

**Known limitation (intentionally out of scope):** a wrong reference still silently
corrupts SNV trinucleotide context (consequence #1). The deletion mismatch is the only
signal available for free from the existing repeat scan; a general per-variant
`REF[0]` vs `reference[p]` check is deferred (YAGNI).

### Change 3 — invalidate stale persisted catalogues

Existing `.svar` files carry a `mutcat.npy` computed under the buggy code; those values
are wrong. Bump `MUTCAT_VERSION` from `1` to `2` (`genoray/_mutcat.py:145`) so a stored
`mutcat_version` no longer matches the current version.

Inspect the `mutcat` load path in `SparseVar.__init__` and add a staleness warning when
a loaded file's `mutcat_version` is present and less than `MUTCAT_VERSION`:

> `mutcat field was computed with an older version (v{old} < v{current}); recompute via annotate_mutations()`

If no load-time version comparison currently exists, add the comparison at the point
where the `mutcat` field is registered/opened.

## Tests

Add to `tests/test_svar_mutations.py` and `tests/test_mutcat.py` as appropriate:

1. **Regression (off-by-one crash):** the issue's minimal reproducer — a 5 bp deletion
   at 1-based POS=20 in a non-repetitive sequence — runs `annotate_mutations(ref,
   write_back=False)` without raising `KeyError` and produces a valid ID-83 code
   (`5:Del:R:0`).
2. **SNV context correctness:** a SNV whose true trinucleotide context (at `POS-1`)
   differs from the +1-shifted context; assert the resulting SBS-96 code matches the
   `POS-1` context, not the shifted one.
3. **Guard + warning:** drive `classify_id83` (and/or `classify_variants`) with a `REF`
   whose deleted unit is absent at `scan_start` (a deliberate REF/reference mismatch);
   assert the result is `UNCLASSIFIED` and that the aggregated warning is emitted
   (capture via a loguru sink / `caplog`).
4. **Calibration test update:** remove the `- 1` at `tests/test_mutcat_calibration.py:294`
   so it relies on the new in-function conversion.

## Out of scope

- General per-variant `REF` vs reference verification for SNVs/insertions.
- Any change to the public `genoray` API surface or `SKILL.md` (the fix aligns code
  with existing documentation).

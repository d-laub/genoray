# Contig-scoped `annotate_mutations`: select contigs + a distinct NOT_ANNOTATED sentinel

**Date:** 2026-06-13
**Issue:** [#62](https://github.com/d-laub/genoray/issues/62)
**Status:** Approved design (pending spec review)
**Branches off:** `vectorize-mutation-matrices` (integrates with its per-contig
`classify_variants` loop). Complements [#61](https://github.com/d-laub/genoray/issues/61).

## Problem

`SparseVar.annotate_mutations` classifies **every** variant in the index,
all-or-nothing, and aborts the whole pass if any contig is absent from the
reference. Real analyses are usually scoped to a contig subset — e.g. the 24
nuclear contigs (`chr1..chr22, chrX, chrY`) — and want to deliberately leave
mitochondrial, pseudoautosomal (`PAR1`/`PAR2`), and decoy/alt contigs out,
without crashing and without conflating "not annotated" with "couldn't be
classified."

Concretely: a genome-wide SBS96 run on a 16,007-tumor `.svar` dies on
`ValueError: Contig 'MT' not found in reference GRCh38.fa.` The `.svar` also
carries `PAR1` (300,353 rows) and `PAR2` (39,747 rows), which no standard FASTA
has a sequence for — so [#61](https://github.com/d-laub/genoray/issues/61)'s
`MT`↔`chrM` naming fix alone cannot unblock them. An allowlist scoping the run
to the nuclear contigs is the only clean fix.

Two asks:

1. **Let callers select which contigs to annotate.** Only listed contigs are
   classified; entries on excluded contigs are marked with a dedicated sentinel
   rather than triggering a reference fetch.
2. **A distinct "not annotated" category + write-back metadata.** `UNCLASSIFIED`
   means "we tried and it isn't a classifiable substitution" — semantically
   different from "we chose not to annotate this contig." Use a separate
   sentinel, and persist the annotation scope so a later open knows the `mutcat`
   field is contig-scoped.

## Design decisions

### 1. New sentinel `NOT_ANNOTATED = -4`

Add to `SENTINELS` in `genoray/_mutcat.py`:

```python
SENTINELS: dict[str, int] = {
    "DBS_PARTNER": -1,    # 3' half of an adjacency doublet; never counted
    "UNCLASSIFIED": -2,   # symbolic/complex/MNV>2bp/non-ACGT/ref-mismatch
    "MISSING": -3,
    "NOT_ANNOTATED": -4,  # entry on a contig excluded from the annotation scope
}
```

The issue suggested `-3`, but `-3` is already `MISSING`; `-4` is the next free
slot. Semantics: "entry on a contig deliberately left out of scope," distinct
from `UNCLASSIFIED` ("tried, not a classifiable substitution"). Because it is
negative, `NOT_ANNOTATED` is already excluded from every `mutation_matrix`
count by the existing `code < 0` guard in `_count_kernel`
(`genoray/_mutcat.py`) — no counting change is needed.

### 2. `classify_variants(index, reference, contigs=None)`

New keyword-or-positional parameter `contigs: list[str] | None = None`
(`None` = all contigs, current behavior).

In the vectorize branch, `classify_variants` already loops per contig:

```python
for contig in contigs present in the index:
    seq  = reference.contig_array(contig)
    rows = where(contig_code == contig)
    ...
```

Contig scoping integrates at this seam:

- Normalize the allowlist through the index's `ContigNormalizer` (so `chr1`/`1`
  both match the index's naming scheme), producing the set of in-scope contig
  ids and a per-row in-scope mask.
- **Initialize `out`** to `NOT_ANNOTATED` for out-of-scope rows and
  `UNCLASSIFIED` for in-scope rows. (In-scope but unclassifiable variants thus
  stay `UNCLASSIFIED`; out-of-scope rows that are never visited stay
  `NOT_ANNOTATED`.)
- The per-contig loop iterates **only contigs in the allowlist**, so excluded
  contigs are never fetched.
- When `contigs is None`, behavior is identical to today: `out` initializes to
  `UNCLASSIFIED` everywhere and all contigs are visited.

**Bad allowlist entry** — a listed contig that, after normalization, matches no
contig in the index (e.g. a typo `chr23`): emit **one aggregated loguru
warning** naming the unmatched entries, then proceed annotating the contigs that
did match. (Chosen over fail-fast: surfaces typos without aborting a long run.)

**In-scope contig absent from the reference FASTA** — a listed contig present in
the index but with no reference sequence (e.g. explicitly listing `MT` against a
reference that lacks it): `reference.contig_array()` **raises** as today. You
explicitly asked to annotate it, so a missing reference sequence is a real
error. The documented way to avoid the crash is to not list the contig.

### 3. `annotate_mutations(reference, *, contigs=None, write_back=True)`

- Threads `contigs` through to `classify_variants`.
- Computes the per-variant in-scope mask (same `ContigNormalizer`-based logic)
  and applies `is_snv &= in_scope` before calling `build_entry_codes`. This is
  required for correctness: the DBS adjacency override in `_entry_codes_kernel`
  (`genoray/_mutcat.py:351`) fires only when both the variant and its neighbor
  are `var_is_snv`. Without the mask, two adjacent **out-of-scope** SNVs would be
  collapsed into a DBS code (and a `DBS_PARTNER`), overwriting their
  `NOT_ANNOTATED` codes. Masking `is_snv` to in-scope leaves out-of-scope
  variants as plain non-SNV broadcasts, so their `NOT_ANNOTATED` `var_code`
  propagates unchanged to every entry.
- On `write_back=True`, records the **normalized** annotated contig names in
  metadata (see below).

### 4. Metadata + version bump

`SparseVarMetadata` (`genoray/_svar.py:461`) gains:

```python
mutcat_contigs: list[str] | None = None  # normalized contigs annotated; None = all
```

Backward-compatible (new field has a default; every existing construction site
keeps working). When `annotate_mutations` writes back, it sets
`mutcat_contigs` to the normalized list of contigs actually annotated, or `None`
when `contigs is None`. A later `SparseVar(...)` open can then tell which
contigs are `NOT_ANNOTATED` versus genuinely processed.

Bump `MUTCAT_VERSION` from `2` to `3` (`genoray/_mutcat.py:151`): the on-disk
`mutcat` semantics changed (new `NOT_ANNOTATED` sentinel + contig scoping).
Reopening a `mutcat_version=2` file with `fields=["mutcat"]` triggers the
existing staleness warning in `SparseVar.__init__`.

### 5. Relationship to #61

This spec resolves the underlying need behind #61's optional "skip/mark
contigs absent from the reference" nicety — but via the **explicit allowlist**,
not a separate auto-skip mode:

- The allowlist is the sanctioned, intentional way to avoid the abort: scope to
  the nuclear contigs and `MT`/`PAR1`/`PAR2` become `NOT_ANNOTATED`, never
  fetched, no crash.
- `contigs=None` stays **raise-on-absent**, consistent with the in-scope rule in
  §2 ("if you asked for it and it's missing, that's a real error"). Adding an
  auto-skip mode would contradict that principle and add a second behavior.
- #61 proper remains narrowly the `chrM`↔`MT` **naming** fix (so a contig whose
  sequence exists under another name resolves), which is a genuinely different
  bug.

### 6. Public API docs (required)

Per `CLAUDE.md`, this PR updates `skills/genoray-api/SKILL.md`:

- `annotate_mutations` signature gains `contigs=None`; document the allowlist
  semantics, `ContigNormalizer` matching, the bad-entry warning, and the
  in-scope-absent-from-reference raise.
- Add the `NOT_ANNOTATED` sentinel to the sentinel/category documentation and
  note that out-of-scope entries are excluded from `mutation_matrix` counts.
- Document `mutcat_contigs` in the persisted-metadata description.

## Architecture and data flow

```
annotate_mutations(reference, contigs=CONTIGS):
    normalize CONTIGS via index ContigNormalizer -> in_scope contig set
    warn about CONTIGS entries that match no index contig
    var_code = classify_variants(index, reference, contigs=CONTIGS)
        # out-of-scope rows -> NOT_ANNOTATED
        # in-scope rows     -> SBS/DBS/ID code or UNCLASSIFIED
        # raises if an in-scope contig is absent from the reference
    is_snv = per-variant SNV mask & in_scope        # gate DBS adjacency
    entry_codes = build_entry_codes(..., var_code, ..., is_snv, ...)
    register in-memory field
    if write_back:
        memmap mutcat.npy <- entry_codes
        metadata.fields["mutcat"] = "int16"
        metadata.mutcat_version   = MUTCAT_VERSION   # 3
        metadata.mutcat_contigs   = normalized list (or None)
```

`build_entry_codes`, `_entry_codes_kernel`, `count_matrix`, and `_count_kernel`
are **unchanged** — out-of-scope handling lives entirely in `classify_variants`
(code selection) and the `is_snv &= in_scope` gate (adjacency suppression).

## Testing

Extend `tests/test_svar_mutations.py` (and `tests/test_mutcat.py` for the
`classify_variants` level where convenient):

1. **Subset annotation** — `contigs=` listing a subset: variants on listed
   contigs get real codes; entries on excluded contigs are `NOT_ANNOTATED`
   (value `-4`), **not** `UNCLASSIFIED`; `mutation_matrix` counts are identical
   to a run on a `.svar` that physically lacked the excluded contigs.
2. **Normalization** — an allowlist given as `chr1` matches an index stored as
   `1` (and vice versa).
3. **Bad allowlist entry** — listing a contig absent from the index logs a
   warning and the run still annotates the matching contigs.
4. **In-scope contig absent from reference** — listing a contig that exists in
   the index but not the reference raises (regression guard for the documented
   behavior).
5. **Adjacency suppression** — two adjacent out-of-scope SNVs are **not**
   DBS-collapsed; both entries remain `NOT_ANNOTATED`.
6. **Metadata round-trip** — `mutcat_contigs` persists and reloads;
   `contigs=None` stores `None`.
7. **Staleness** — a `mutcat_version=2` file reopened with `fields=["mutcat"]`
   warns (existing mechanism, new version number).

No reference-genome-free codebook changes; the existing oracle/differential
tests in the vectorize branch are unaffected because `contigs=None` preserves
element-for-element behavior.

## Scope

**In scope:** `contigs=` parameter on `annotate_mutations` and
`classify_variants`; `NOT_ANNOTATED` sentinel; `mutcat_contigs` metadata;
`MUTCAT_VERSION` bump to 3; `is_snv` adjacency gate; `SKILL.md` update; tests.

**Out of scope:** auto-skip mode for `contigs=None` (see §5); the `chrM`↔`MT`
naming fix (#61); `write_view` propagation of a scoped `mutcat`
(`write_view` already never copies `mutcat` positionally); any change to the
int16 code space, `build_entry_codes`, or the counting kernels.

## Files

- **Modify** `genoray/_mutcat.py` — add `NOT_ANNOTATED` to `SENTINELS`; bump
  `MUTCAT_VERSION` to 3; add `contigs` param to `classify_variants` with
  allowlist normalization, scoped per-contig loop, out-of-scope `NOT_ANNOTATED`
  initialization, and the bad-entry warning.
- **Modify** `genoray/_svar.py` — add `mutcat_contigs` to `SparseVarMetadata`;
  add `contigs` param to `annotate_mutations`, compute the in-scope mask, gate
  `is_snv`, thread `contigs` to `classify_variants`, and persist
  `mutcat_contigs` on write-back.
- **Modify** `skills/genoray-api/SKILL.md` — document `contigs=`,
  `NOT_ANNOTATED`, and `mutcat_contigs`.
- **Modify** `tests/test_svar_mutations.py` (and `tests/test_mutcat.py` as
  needed) — the tests above.

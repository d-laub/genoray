# Design: SBS-96 / DBS-78 / ID-83 mutation catalogues on `SparseVar`

**Date:** 2026-06-11
**Status:** Approved (pending spec review)

## Goal

Port the core of [SigProfilerMatrixGenerator](https://github.com/SigProfilerSuite/SigProfilerMatrixGenerator)
into `genoray` so that mutational-signature catalogues can be computed directly on a
`SparseVar`, with an ideally faster (vectorized / numba) implementation. Two outputs share
one classification core:

1. **Per-entry category annotation** — a `mutcat` field (int16, enum-encoded) aligned to the
   sparse genotypes and written to the `.svar` directory like any FORMAT field. The category
   is stored per sparse genotype entry (per sample × haplotype × variant occurrence), which is
   what makes sample-specific DBS detection representable.
2. **Count matrices** — `mutation_matrix(kind=...)` returns an in-memory Polars DataFrame
   (rows = mutation types, columns = samples), one per `{SBS96, DBS78, ID83}`.

## Scope

**In scope (v1):**

- SBS-96 (single-base substitutions in trinucleotide context).
- DBS-78 (doublet-base substitutions).
- ID-83 (PCAWG/SigProfiler indel classification).
- Per-entry `mutcat` annotation field, persisted to the `.svar` dir (with a `write_back` flag).
- Per-sample count matrices as Polars DataFrames, with a configurable counting unit
  (per-allele vs per-sample presence).
- A genoray-vendored `Reference` reader backed by `pysam`.

**Out of scope (v1), documented as such:**

- Strand-bias contexts (SBS-192/384, DBS-186), other SBS/DBS/ID context sizes.
- Runs of ≥3 adjacent SNVs forming higher-order events — left as individual SBS in v1.
- MNV records longer than 2 bp — classified as `UNCLASSIFIED` in v1.

## Background / existing patterns

- `SparseVar.annotate_with_gtf` (`genoray/_svar.py:1344`) is the precedent: it computes a
  per-variant annotation, joins it onto the in-memory `index` table, and (when
  `write_back=True`) rewrites `index.arrow`. The new mutation annotation follows the same
  write-back ergonomics, but stores a **per-entry field** rather than an index column.
- FORMAT-like fields are stored as `{name}.npy` mmap arrays that share the genotype
  `offsets.npy`, and are registered in `metadata.json`'s `fields` map (name → numpy dtype).
  See `_open_fmt` / `_write_genos` (`genoray/_svar.py:2146`, `:2161`) and
  `SparseVarMetadata.fields` (`genoray/_svar.py:463`). The `mutcat` field drops into exactly
  this slot.
- `genos` has shape `(n_samples, ploidy, None)`: for each `(sample, haplotype)` it holds the
  variable-length list of variant indices that are non-ref on that haplotype. A field aligned
  to `genos` therefore has one value per `(sample, haplotype, variant)` occurrence.
- `genoray._utils.ContigNormalizer` already handles `chr`-prefixed vs unprefixed contigs and
  is reused by the vendored `Reference`.

## New modules

### `genoray/_reference.py` — vendored `Reference` (public as `genoray.Reference`)

A genoray-local reader modeled on `gvl.Reference`'s interface, but backed by
`pysam.FastaFile` for on-demand reads rather than the `.gvlfa` whole-genome cache.

- `Reference.from_path(fasta: str | Path, contigs: list[str] | None = None) -> Reference`
- `Reference.fetch(contig, starts, ends) -> Ragged[np.bytes_]` (or a single byte array for a
  scalar query) — the same call shape as `gvl.Reference.fetch`.
- Implementation: load **one contig into a numpy `uint8` array at a time** and slice from it;
  cache only the current contig and drop it when a different contig is requested. This keeps
  per-variant context lookups vectorized and fast (variants are sorted within a contig in
  `SparseVar`) without holding the whole genome in RAM or vendoring the 338-line cache builder.
- Reuses `genoray._utils.ContigNormalizer` so queries work regardless of `chr` prefix.
- Adds **`pysam`** to `pyproject.toml` dependencies.

### `genoray/_mutcat.py` — classification core + matrix builder

- COSMIC codebooks: the canonical SBS-96, DBS-78, and ID-83 label lists as module constants,
  in a fixed order that defines the integer codes.
- numba kernels that assign integer codes given `REF`/`ALT`/`POS` arrays plus fetched reference
  context. Pure / array-based for isolated testing and speed.
- The per-sample count-matrix accumulator kernel.

## Unified codebook (the enum)

A single `int16` code space shared by the `mutcat` field:

- `0..95`   → SBS-96
- `96..173` → DBS-78
- `174..256` → ID-83
- Sentinels (negative or top-of-range, fixed constants):
  - `DBS_PARTNER` — the 3′ half of an adjacency-detected doublet; excluded from every matrix
    so the doublet is counted exactly once.
  - `UNCLASSIFIED` — symbolic/complex variants, MNV records > 2 bp, anything not in a category.
  - `MISSING` — reserved for missing/sentinel entries.

The code↔label mapping and a `mutcat_version` integer are written into `metadata.json` so the
stored field is self-describing and forward-compatible. `SparseVarMetadata` gains optional
`mutcat_version` and (if needed) a stored label list; `available_fields`/`fields` register
`mutcat` like any other field.

### Storage tradeoff (decided)

A per-entry code duplicates the intrinsic SBS/ID code across every sample carrying that
variant. The alternative hybrid representation (per-*variant* SBS/ID code in `index` + a
per-*entry* DBS-membership flag) is ~half the storage but splits counting into two lookups.
**Decision: unified per-entry field for v1** — simpler, matches the "treat it like a FORMAT
field" intent, and easier to verify. Storage cost is ~½ of `genos` (int16 vs int32, same
offsets). The hybrid remains a possible future optimization.

## Classification algorithms

### SBS-96

For variants where REF and ALT are both single ACGT bases:

- Trinucleotide context `ref[p-1] · REF · ref[p+1]` from the reference.
- Folded to the pyrimidine convention: if REF ∈ {A, G}, reverse-complement both the context
  and the substitution.
- 6 substitution classes × 4 (5′ base) × 4 (3′ base) = 96; label form `A[C>A]A`.

### DBS-78

Two routes, deduplicated:

1. **Native MNV records** — REF and ALT both exactly 2 bp: classified directly as a single
   entry.
2. **Adjacency** — within each `(sample, haplotype)` track, exactly two consecutive SNVs at
   positions `p` and `p+1` combine into a doublet. The 5′ entry receives the DBS code; the 3′
   entry receives `DBS_PARTNER`. Same-track adjacency = same-haplotype (exact when phased;
   documented assumption when unphased).

Because atomized inputs split MNVs into adjacent SNVs (route 2) and non-atomized inputs may
carry native 2 bp MNV records (route 1), the two routes are mutually exclusive per locus and
do not double-count. Runs of ≥3 consecutive SNVs are left as individual SBS in v1.

### ID-83

PCAWG/SigProfiler indel channels (83 total): insertion vs deletion × size {1, 2, 3, 4, 5+} ×
repeat-unit count {0..5+}, plus the microhomology channels for deletions. A bounded reference
window around each indel is fetched to count repeat units and microhomology length. Assumes
left-aligned, atomized indels (the form in which `SparseVar` is fed into GenVarLoader).

## `SparseVar` API

```python
svar.annotate_mutations(reference, *, write_back=True) -> None
svar.mutation_matrix(kind, *, count="allele" | "sample") -> pl.DataFrame
```

- `annotate_mutations`
  - `reference`: a `genoray.Reference` or a FASTA path (auto-wrapped into a `Reference`).
  - Computes the per-entry `mutcat` codes (SBS/ID intrinsic per-variant; DBS via the two routes
    above) and either persists them (`write_back=True`, default) — writing `mutcat.npy` and
    updating `metadata.json`, mirroring `annotate_with_gtf` — or holds them in memory
    (`write_back=False`).
- `mutation_matrix`
  - `kind`: one of `"SBS96"`, `"DBS78"`, `"ID83"`.
  - `count`: `"allele"` (het = 1, hom = 2) or `"sample"` (presence; het = hom = 1).
  - Returns a Polars DataFrame with a `MutationType` column (the category labels, in codebook
    order) plus one column per sample — SigProfiler's row-types-×-sample-columns layout.
  - Raises a clear error if `mutcat` has not been computed or loaded.

## Counting

A numba kernel walks the genotype offsets and accumulates `counts[sample, code]`. For a given
`kind`, the matching code range is sliced, the label list attached as `MutationType`, and the
Polars DataFrame emitted. `DBS_PARTNER`, `UNCLASSIFIED`, and `MISSING` are never counted.
`count="allele"` accumulates one per occurrence; `count="sample"` collapses to presence per
`(sample, code)`.

## Testing

- Unit tests per classifier against hand-computed small cases: every SBS pyrimidine fold;
  native + adjacency DBS (including the `DBS_PARTNER` exclusion); representative ID-83 channels
  including microhomology and repeat-count boundaries.
- Round-trip test: `annotate_mutations(write_back=True)` → re-open `SparseVar` → `mutation_matrix`.
- Counting-unit test: per-allele vs per-sample on het/hom fixtures.
- Where feasible, a small fixture cross-checked against SigProfilerMatrixGenerator output;
  `vcfixture` generates the inputs.
- Network-free; runs under `pixi run pytest`.

## Docs

Per `CLAUDE.md`, `skills/genoray-api/SKILL.md` is updated in the same PR for the new public
surface: `genoray.Reference`, `SparseVar.annotate_mutations`, `SparseVar.mutation_matrix`, the
`mutcat` field and its codebook, the `kind`/`count` parameters, and the documented v1 scope
limits.

## Open risks / notes

- **DBS adjacency under no phasing**: per-track adjacency is a proxy for same-haplotype;
  accurate when phased, documented otherwise.
- **`pysam` dependency**: new to genoray; justified by the on-demand FASTA reads. Confirm it
  resolves cleanly under Pixi/conda-forge.
- **ID-83 fidelity**: the indel classifier is the most intricate piece; correctness is pinned
  by the hand-computed channel tests and the SigProfiler cross-check fixture.

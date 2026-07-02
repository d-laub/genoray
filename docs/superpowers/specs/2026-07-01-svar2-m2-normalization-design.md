# SVAR 2.0 — M2: Variant normalization during conversion (split + atomize)

> Spec for roadmap milestone **M2** (`docs/roadmap/svar-2.md`). Branch:
> `svar-2-m2-normalize` (worktree under `.claude/worktrees`). Companion docs:
> [`data-model.md#variant-normalization`](../../roadmap/data-model.md) and
> [`architecture.md#conversion-pipeline`](../../roadmap/architecture.md).

## Goal

Remove the reader's "input must already be normalized" preconditions so conversion
accepts un-normalized VCFs directly. Today `vcf_reader.rs` asserts biallelic
(`alleles.len() <= 2`) and non-complex (`!(alt_len>1 && alt_len<ref_len)`) input and
panics otherwise. After M2, multi-allelic, MNP, and complex records flow through an
inline normalization step that emits atomized biallelic primitives.

## Scope

In scope:

- **Biallelic split** — a site with ALTs `A₁,…,Aₙ` becomes `n` independent biallelic
  variants, one per ALT.
- **Atomization** — MNPs decompose into per-position SNPs; complex REF/ALT pairs
  decompose into SNP atoms plus one anchored indel, mirroring bcftools'
  `_atomize_allele` (`abuf.c`).

Explicitly **out of scope** (deferred):

- **Left-alignment → M2b.** It is the *only* part of roadmap M2 that requires a
  reference genome (FASTA/faidx) dependency and a new required conversion argument, and
  it complicates the position-reorder bound (leftward shifts). Deferring it keeps this
  PR self-contained and reference-free. The roadmap's M2 entry will be updated to note
  the split.

Non-goals this PR:

- **Symbolic / breakend ALTs** (`<DEL>`, `<INS>`, `N[chr:pos[`, …) — long-read/SV
  territory, out of scope for the short-read format. The reader returns a
  `ConversionError` rather than panicking.
- **`*` spanning-deletion ALTs** — skipped (no atom emitted). Presence for a haplotype
  covered by an upstream deletion is carried by that deletion's own record.

No change to the Python API signature (`run_conversion_pipeline`) or the on-disk
layout — this PR purely widens input acceptance.

## Background: how bcftools atomizes

`_atomize_allele` (bcftools `abuf.c`, referenced from `vcfnorm.c#L2629`):

1. Trim the shared **suffix**: `while rlen>1 && alen>1 && ref[rlen-1]==alt[alen-1] { rlen--; alen-- }`
   (keeps ≥1 base per side).
2. Walk positions left→right over the overlap. Emit a **SNV atom** at each position
   where `ref[i] != alt[i]`.
3. At the length-difference boundary (`i+1>=rlen || i+1>=alen`), emit a **left-anchored
   indel** atom: a single anchor base plus the inserted/deleted tail.

The three resulting atom shapes — SNV, anchored insertion, anchored deletion — map
exactly onto the shapes the existing encode seam already accepts:

| Atom | ref_len | alt_len | `ilen` | Encoder path |
| --- | --- | --- | --- | --- |
| SNP | 1 | 1 | 0 | `encode_snp_2bit` (2-bit SNP stream) |
| Insertion (anchored) | 1 | ≥2 | >0 | `pack_variant` INS (`alt` = anchor+inserted) |
| Deletion (anchored) | ≥2 | 1 | <0 | `pack_variant` DEL (`alt` = anchor base) |

Because atoms are exactly these shapes, `rvk.rs` / `streams.rs` / the executor / the
merge are **untouched** — the encoding seam stays sealed.

## Design

### Component 1 — `normalize.rs` (pure, isolated)

A single pure function is the algorithmic core, with no I/O and no genotype knowledge,
so it is trivially unit- and property-testable in isolation:

```rust
pub struct Atom {
    pub pos: u32,               // 0-based atom start
    pub ilen: i32,              // len(alt) - len(ref): 0=SNP, >0=INS, <0=DEL
    pub alt: SmallVec<[u8; 4]>, // SNP: 1 base; INS: anchor+inserted; DEL: 1 anchor base
    pub source_alt_index: u16,  // 1-based index into the record's original ALTs (GT remap)
}

/// Decompose one VCF record (already split across its ALTs internally) into atomic
/// biallelic primitives, appended to `out`. Skips `*` ALTs. Returns an error for
/// symbolic/breakend/invalid alleles.
pub fn atomize_record(
    pos: u32,
    ref_allele: &[u8],
    alts: &[&[u8]],
    out: &mut Vec<Atom>,
) -> Result<(), NormalizeError>;
```

Guarantees (enforced by construction and asserted in tests):

- Every emitted `Atom` is one of the three encoder-valid shapes above.
- `atom.pos ∈ [pos, pos + ref_len)` (no left-alignment ⇒ atoms never precede the
  record start).
- `source_alt_index` identifies which original ALT the atom came from, for genotype
  remapping upstream.

This is the **only** module that knows atomization rules.

### Component 2 — reader restructure + reorder buffer (`vcf_reader.rs`)

Two consequences of normalization the current reader cannot handle:

**(a) Variable atom count per record.** One record now yields 0..n atoms. The reader can
no longer pre-size `BitGrid3` to `chunk_size` rows up front.

**(b) Position reordering.** Atomization spreads atom positions rightward: an MNP at
pos 100 emits atoms at 100, 101, 102, … which can leapfrog a following record's start
(records may overlap). The Phase-2 merge is a pure **interleaving** merge — it assumes
each per-`(sample, ploid)` stream is already position-sorted. Atomization can violate
that, so the reader must restore global position order.

Changes:

- **Genotype remapping.** Read the *integer* allele index per `(sample, ploid)` via
  `GenotypeAllele::index()` (not the current `== 1` match). An atom's presence bit for a
  haplotype is `allele_index == atom.source_alt_index`. MNP- and complex-derived atoms
  inherit their parent ALT's presence, so heterozygous `1/2` calls are handled naturally
  by the per-`(sample, ploid)` grid. Missing/`.` alleles (`index() == None`) and
  reference (`0`) set no presence bit.

- **Reorder buffer.** A min-heap of pending atoms keyed by `pos`, held as reader state
  across `read_next_chunk` calls. With no left-alignment, every atom's position is
  ≥ its source record's start ≥ the current read frontier. So the exact (non-heuristic)
  flush rule is:

  > pop and emit all buffered atoms with `pos < current_record_start`; at EOF, flush
  > everything.

  This yields globally position-sorted emission across chunk boundaries with no
  arbitrary lookahead window. Buffered atoms reference their source record's decoded
  genotypes (shared, not copied per atom), so buffer memory stays ~one genotype vector
  per still-open overlapping record.

- **Chunk building.** Accumulate flushed atoms until `chunk_size`, then build a
  `DenseChunk` (`BitGrid3` sized to the actual atom count, plus `pos` / `ilens` / `alt` /
  `alt_offsets`). `DenseChunk`, the executor, the encode seam, and the merge are
  unchanged. `chunk_size` now counts **output atoms** rather than input records.

### Component 3 — error handling & API

- Remove the two `assert!`s (biallelic, non-complex) in `vcf_reader.rs`.
- Add `ConversionError` variants (or a `NormalizeError` mapped into it) for symbolic /
  breakend / invalid alleles, surfaced as a Python `RuntimeError` (matching the existing
  error path), not a panic.
- `run_conversion_pipeline` signature, on-disk layout, and `meta.json` are unchanged.

## Testing (TDD)

`normalize.rs` unit tests:

- SNP passthrough (`A>C`).
- Biallelic split (`A>C,G` → two atoms, `source_alt_index` 1 and 2).
- MNP → per-position SNPs (`AC>GT` → `A>G`@pos, `C>T`@pos+1; equal bases skipped).
- Anchored insertion (`A>AGG`) and deletion (`ATG>A`).
- Complex `GCG>GTGA` → SNV(s) + one anchored indel at the expected positions.
- Shared prefix/suffix trimming.
- `*` ALT skipped; symbolic/BND ALT → error.

`normalize.rs` property tests:

- Every emitted atom is encoder-valid (one of the three ref/alt shapes) and
  `pos ∈ [record_pos, record_pos + ref_len)`.

Reader-level tests:

- Overlapping and MNP-spanning records emit **globally position-sorted** atoms across
  chunk boundaries.
- Genotype remapping: multiallelic `1/2` produces correct per-atom presence.

End-to-end:

- Convert an un-normalized VCF and cross-check positions / ALTs / genotypes against
  `bcftools norm -m -any --atomize` as an oracle when `bcftools` is on `PATH`; otherwise
  a hand-built fixture with expected atoms.

## Roadmap bookkeeping

Per the SVAR 2.0 working agreement, in the implementing PR:

- Update `svar-2.md` M2 status and note left-alignment split to a new **M2b** entry.
- Reconcile `data-model.md#variant-normalization` (it currently lists all three
  transforms as one milestone) to reflect that left-alignment ships separately.

## Open questions

- **Complex-variant decomposition parity.** Match bcftools `_atomize_allele` exactly for
  the anchored-indel boundary, or accept a simpler documented rule where they diverge on
  rare pathological inputs? Default: mirror bcftools; pin exact behavior in the plan
  against the `abuf.c` source.
- **`chunk_size` semantics.** Confirm counting output atoms (not input records) is
  acceptable given downstream memory assumptions; document the change where `chunk_size`
  is exposed.

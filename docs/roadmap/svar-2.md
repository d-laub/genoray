# SVAR 2.0 Roadmap

> **Status:** active epic · **Branch:** `svar-2` · **Home:** `genoray` (new `SparseVar2` class)
>
> This is the entry point for the SVAR 2.0 effort. It is a living document. See
> the supplements for detail:
> - [`data-model.md`](data-model.md) — encodings, bit layouts, LUT, dense matrix, the dense/sparse cost model, on-disk layout, and format constraints (with rationale).
> - [`architecture.md`](architecture.md) — conversion pipeline, the encoding-agnostic abstraction seam, query algorithm, and the Python decode path.
>
> Everything here is our **current best approximation**. It is version-controlled
> and meant to be corrected as we learn. When something turns out to be wrong, fix
> the doc in the same PR that proves it wrong.

## Vision

SVAR 2.0 extends the existing SparseVar format (SVAR 1.0, `python/genoray/_svar.py`)
into a **hybrid variant store tuned for the empirical distribution of variants from
short-read / NGS data**. SVAR 1.0 stores genotypes as sparse `u32` pointers into a
single variant table. SVAR 2.0 adds two new ideas:

1. **Inline, dense variant encoding** — most short-read variants are SNPs or small
   indels whose ALT allele fits in a couple of bytes. Instead of paying for a pointer
   plus a table row, we encode the variant *inline* using a compact
   [VariantKey](https://www.biorxiv.org/content/10.1101/473744v3)-style key (see
   [`data-model.md`](data-model.md)).
2. **Per-variant choice of representation** — for each variant we deterministically
   pick the cheapest representation given its allele frequency, trading inline sparse
   storage against a dense 1-bit genotype matrix.

We optimize for **short-read NGS**. Long-read data (with its long, abundant alleles)
is explicitly **out of scope** — its variant distribution breaks the assumptions that
make the inline encoding a win.

## The hybrid at a glance

Within each contig, every variant is assigned to exactly one of these representations
by the cost model (see [`data-model.md`](data-model.md#dense-vs-sparse-cost-model)):

| Representation | Code name | What it stores | Best when |
| --- | --- | --- | --- |
| **Inline variant key** | `var_key` | Per-call inline key in a 2-bit SNP stream or a 32-bit indel stream; long indel alleles spill to a LUT (SNPs never do) | Low allele frequency (most variants) |
| **Pointer** | `pointer` | Sparse `u32`/`u64` pointers into a shared variant table, split into 2-bit SNP and 32-bit indel tables (this is SVAR 1.0) | Variant info is large (many INFO/FORMAT fields) |
| **Dense** | `dense` | 1-bit `(sample, ploid, variant)` matrix + variant table, each split into 2-bit SNP and 32-bit indel sub-streams | High allele frequency |

There are two axes here, which is the source of the "4-way" / "3-way" naming:

- **Variant-info encodings — the "4-way hybrid" (MVP):** inline SNP (2-bit), inline
  indel (32-bit), LUT pointer (long-allele spill), and dense. All four exist in the MVP
  design.
- **Genotype-storage representations:** `var_key` and `dense` in the MVP; adding the
  SVAR-1.0-style `pointer` representation later makes this axis a **"3-way hybrid"**
  (inline + pointer + dense), milestone M11.

## Milestones

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

### MVP — VCF → SVAR2

- [~] **M1. VCF → SVAR2 conversion.** Streaming, chunked, multi-threaded
  conversion producing the `var_key` representation. Builds on the existing
  conversion pipeline on this branch. See [`architecture.md`](architecture.md#conversion-pipeline).
  *Implemented and tested for the `var_key` happy path:* a per-contig pipeline
  (reader → encode → writers → Phase-2 merge) fanned out across contigs by rayon
  with bounded-channel backpressure; a PEXT (BMI2) / portable-SWAR inline encoder
  proven byte-identical by proptest; a bit-packed dense read buffer (`BitGrid3`); a
  streaming long-allele LUT; and a memory-bounded parallel tile merge. Covered by 30
  in-source unit/proptests + 5 e2e tests. An optional per-contig monitoring sampler
  (`GENORAY_SAMPLE_INTERVAL`) reports channel fill and per-thread CPU%. *Remaining:*
  variant normalization (M2, currently a precondition — see below) and the dense
  routing of M4; the on-disk filenames are still provisional (see M3). The current
  encoder writes a single **uniform 32-bit** `var_key` stream; the data model now
  specifies splitting it into 2-bit `snp/` and 32-bit `indel/` sub-streams (see
  [`data-model.md`](data-model.md#on-disk-layout)), which remains to be implemented.
- [ ] **M2. Variant normalization during conversion.** Left-alignment, atomization,
  and biallelic splitting (split multi-allelic sites) applied inline as variants stream
  through. See [`data-model.md`](data-model.md#variant-normalization).
  *Not started — currently an input precondition.* The reader asserts normalized input
  and panics on multi-allelic or complex records; the inline encoder bakes in the
  atomized invariant `ref_len = 1` (it stores `ILEN = alt_len − 1` and decodes
  `alt_len = ILEN + 1`). M2 is what will let conversion accept un-normalized VCFs
  directly.
- [~] **M3. Per-contig split + sidecar positions.** Partition the SVAR2 directory by
  contig; keep positions as sidecar arrays. See
  [`architecture.md`](architecture.md#on-disk-layout).
  *Core done:* output is partitioned per contig (`{out}/{contig}/var_key/`) and the
  merge emits sorted position + offset sidecars. *Provisional / remaining:* the current
  scratch filenames are `final_positions.bin` / `final_keys.bin` (raw little-endian via
  bytemuck) and `final_offsets.npy`, which differ from the `.npy` names in the layout
  spec; `meta.json` and the per-contig `max_del.npy` are not yet written.
- [ ] **M4. Dense representation + cost-model routing.** Implement the 1-bit dense
  genotype matrix and the deterministic per-variant dense/sparse decision. See
  [`data-model.md`](data-model.md#dense-vs-sparse-cost-model).
- [ ] **M5. `(range, sample)` queries.** Fast overlap queries via binary search,
  starting from the [left-tree static search tree](https://curiouscoding.nl/posts/static-search-tree/#left-tree).
  Must handle deletions spanning the query start (see
  [`data-model.md`](data-model.md#overlap-queries-and-deletions)).
- [ ] **M6. Python decode.** Decode query results into user-facing structs/classes
  (to be spec'd). Requires fast sorted **unions** to merge data from multiple
  position-sorted sources. See [`architecture.md`](architecture.md#python-decode-path).

### Beyond MVP

- [ ] **M7. PGEN → SVAR2 conversion.** Likely requires FFI to C++ / `pgenlib`.
- [ ] **M8. Merge / split by contig.** Cheap because contigs are independent on disk.
  See [Format constraints](data-model.md#format-constraints-and-non-goals).
- [ ] **M9. Subset by region.** Non-MVP. Doesn't affect cost-model calculations (no
  new variants; only shrinks variant tables), so it should be cheap.

### Longer term

- [ ] **M10. Checkpointing / resume during conversion.**
- [ ] **M11. 3-way hybrid.** Add the SVAR-1.0-style `pointer` representation as a
  third co-resident option, selected by the cost model when variant info is large.
- [ ] **M12. Bulk merge of multiple SVAR2 files.** A general N-way merge (more
  involved than the by-contig merge of M8 because it can change the cost model and
  must rebuild LUTs / variant tables).

## Working agreement

SVAR 2.0 spans many PRs. To keep the design and the code from drifting apart:

**Any PR that touches the SVAR 2.0 effort MUST read and update this roadmap and the
supplements in the same PR.** Concretely, before opening the PR:

- [ ] Re-read `svar-2.md`, `data-model.md`, and `architecture.md`.
- [ ] Update milestone checkboxes and statuses to match reality.
- [ ] Reconcile any data-model or architecture changes with the supplements — if the
      code now disagrees with a doc, the doc is wrong; fix it here.
- [ ] If you discovered a new open question, add it to the relevant "Open questions"
      section rather than leaving it in your head.

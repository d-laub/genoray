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

- [x] **M1. VCF → SVAR2 conversion.** Streaming, chunked, multi-threaded
  conversion producing the `var_key` representation. Builds on the existing
  conversion pipeline on this branch. See [`architecture.md`](architecture.md#conversion-pipeline).
  *Implemented and tested for the `var_key` happy path:* a per-contig pipeline
  (reader → encode → writers → Phase-2 merge) fanned out across contigs by rayon
  with bounded-channel backpressure; a PEXT (BMI2) / portable-SWAR inline encoder
  proven byte-identical by proptest; a bit-packed dense read buffer (`BitGrid3`); a
  streaming long-allele LUT; and a memory-bounded parallel tile merge. Covered by 30
  in-source unit/proptests + 5 e2e tests. An optional per-contig monitoring sampler
  (`GENORAY_SAMPLE_INTERVAL`) reports channel fill and per-thread CPU%. Exposed to
  Python as `run_conversion_pipeline` (PyO3). *Done — its former preconditions all
  landed:* variant normalization (M2) and left-alignment (M2b) are integrated into
  the reader, and the on-disk filenames were finalized in M3. The
  `var_key` stream is split into 2-bit `snp/` and 32-bit `indel/`
  sub-streams per [`data-model.md`](data-model.md#on-disk-layout); the SNP stream
  is 2-bit-packed post-merge and carries no LUT.
- [x] **M2. Variant normalization during conversion.** Atomization and biallelic
  splitting (split multi-allelic sites) applied inline as variants stream through. See
  [`data-model.md`](data-model.md#variant-normalization). *Shipped:* the reader accepts
  un-normalized VCFs — a pure `normalize.rs` decomposes each record into atomic
  biallelic primitives (SNP / anchored INS / anchored DEL) mirroring bcftools
  `_atomize_allele`, and a position-keyed reorder buffer preserves the merge's
  sorted-position invariant across chunk boundaries. Genotypes are remapped by comparing
  each haplotype's integer allele index to the atom's source ALT index. The former
  "input must be normalized" asserts are gone; symbolic/breakend ALTs are rejected and
  `*`/`.` alleles are skipped.
- [x] **M2b. Left-alignment during conversion.** Shift indels to their leftmost
  equivalent position. Deferred from M2 because it is the only normalization step that
  needs a reference genome (FASTA/faidx) and a new required conversion argument, and it
  widens the reorder-buffer bound (leftward shifts). See
  [`data-model.md`](data-model.md#variant-normalization).
  *Done:* `normalize::left_align` rolls anchored indels leftward (repeat-slide, capped at
  `L_MAX`), validated against `bcftools norm -a -m- -f`; `validate_ref` fails fast on
  REF/FASTA disagreement; the reference FASTA threads through `process_chromosome` and the
  PyO3 entry point as a required argument; and `vcf_reader::next_atom`'s emit bound is
  widened by `L_MAX` to keep emission position-sorted.
- [x] **M3. Per-contig split + sidecar positions + format finalization.** Partition the
  SVAR2 directory by contig; keep positions as sidecar arrays; finalize the on-disk
  format. See [`architecture.md`](architecture.md#on-disk-layout).
  *Done:* output is partitioned per contig
  (`{out}/{contig}/{var_key,dense}/{snp,indel}/`) and the merge emits sorted position +
  offset sidecars. The final per-stream files use their spec base names —
  `positions.bin` / `alleles.bin` / `genotypes.bin` (raw little-endian, mmap-friendly)
  and `offsets.npy` (one-shot) — plus the shared `long_alleles.bin` +
  `long_allele_offsets.npy` under `indel/`. A top-level `meta.json`
  (`format_version` / `samples` / `contigs` / `ploidy`) is written once after all
  contigs succeed. Array dtypes are a `format_version` convention documented in
  [`data-model.md`](data-model.md#on-disk-layout). *(The per-contig `max_del.npy` is
  produced and consumed by the overlap search, so it lands with M5, not here.)*
- [x] **M4. Dense representation + cost-model routing.** Implement the 1-bit dense
  genotype matrix and the deterministic per-variant dense/sparse decision. See
  [`data-model.md`](data-model.md#dense-vs-sparse-cost-model).
  *Implemented and tested:* an exact-integer-bit cost model (`cost_model.rs`) picks
  `Dense` vs. `VarKey` per variant, strictly cheaper wins and ties break to
  `VarKey` (see [`data-model.md`](data-model.md#dense-vs-sparse-cost-model)); routing
  is wired directly into the dense→sparse transpose (`dense2sparse_vk` in `rvk.rs`),
  decided per variant, locally within its chunk, from the variant's plane popcount
  (`BitGrid3::popcount_plane`). Dense variants are written per class (2-bit SNP /
  32-bit indel) as a per-chunk positions/keys/genotype-bits payload (`dense.rs`,
  `writer.rs`) and consolidated by a rectangular dense merge (`dense_merge.rs`) that
  bit-transposes the hap-major matrix across chunks and 2-bit-packs the SNP keys.
  Indel long alleles — whether routed to `var_key` or `dense` — spill to the single
  shared per-contig LUT (see [`data-model.md`](data-model.md#long-allele-lookup-table-lut)).
  Covered by unit/proptests in `cost_model.rs`, `dense.rs`, `dense_merge.rs`, `bits.rs`,
  and `rvk.rs`, plus an e2e dense round-trip test. *Out of scope (deferred to later
  milestones):* `max_del.npy` / overlap queries (M5) and the `pointer` representation
  (M11) are not implemented here. `meta.json` and the unification of the dense final
  filenames with `var_key`'s spec base names (`positions.bin` / `alleles.bin` /
  `genotypes.bin`; see the dense-representation section of
  [`data-model.md`](data-model.md#dense-representation-dense)) landed in M3.
- [x] **M5. `(range, sample)` queries.** Fast overlap queries via binary search,
  starting from the [left-tree static search tree](https://curiouscoding.nl/posts/static-search-tree/#left-tree).
  Must handle deletions spanning the query start (see
  [`data-model.md`](data-model.md#overlap-queries-and-deletions)).
  *Shipped — search core (PR #83):* a self-contained `src/search.rs` with a static
  `(B+1)`-ary B-tree (`B=16`, left-tree layout) over sorted `u32` starts
  (`lower_bound`/`upper_bound`), plus `overlap_range`, a format-independent resolver
  returning the `[s_idx, e_idx)` variant range overlapping `[q_start, q_end)`.
  Deletions spanning the query start are handled by shifting the query with a
  saturating subtraction against a `max_region_length` bound — collapsing the two
  `searchsorted` calls onto one tree instead of building a second. Gated by proptests
  vs `slice::partition_point` (bounds) and a brute-force O(n) oracle (overlap). Depends
  only on in-memory slices — no on-disk types.
  *Shipped — `max_del` producer post-pass (M5 part 2a):* a standalone `src/max_del.rs`
  post-pass over a finished contig's indel key streams emits `max_del.npy`
  (per-`(sample, ploid)`, `(n_samples, ploidy)` `u32`) and `dense/max_del.npy` (`(1,)`
  `u32`), decoding each pure-DEL key's reference-base deletion length via
  `rvk::deletion_len` (the single decode site for the 32-bit key layout) — no LUT,
  reference genome, or merge-path coupling. All-zero artifacts are always written for a
  no-deletion contig so the consumer never special-cases a missing file. Wired into the
  orchestrator after each contig's merge; gated by unit tests, a brute-force per-column
  oracle proptest, and an e2e round-trip that feeds the produced `max_del` into
  `overlap_range`.
  *Shipped — disk-integrated `(range, sample)` query (PR #86):* `src/query.rs` wires the
  `search.rs` core to the on-disk sidecars. `ContigReader::open` mmaps a finished contig's
  `var_key/{snp,indel}` and `dense/{snp,indel}` sidecars, CSR offsets, and the shared LUT
  (tolerating missing sub-streams as empty); `overlap_sample` resolves per-`(sample, ploid)`
  index ranges via `overlap_range`, consumes `max_del.npy` (per-column for `var_key/indel`,
  scalar `dense/max_del.npy` for `dense/indel`) for the leftward bound, genotype-filters the
  two dense classes, decodes each hit through `rvk` (LUT-resolved for long alleles), and
  k-way-merges the four sub-streams into one position-sorted `QueryResult` per haplotype.
  Gated by in-source unit tests plus a disk-integration e2e (`tests/test_query.rs`, 6 tests
  incl. proptests) that builds finished contigs through the real conversion pipeline. The
  batched multi-region/multi-sample **consumer interface**, the `svar2-codec` extraction,
  and uniform-key re-expansion are M6, not M5 — `overlap_sample` is a single-sample Rust
  core, not yet exposed to Python.
- [ ] **M6. Query decode core.** The shared spine both consumers build on. M5's
  `(range, sample)` query landed (`overlap_sample` in `src/query.rs`); M6 generalizes it
  into the batched multi-region × multi-sample consumer spine, re-expands SNPs to the
  uniform 32-bit key at query time, and implements the fast sorted **union** across the
  `{var_key, dense} × {snp, indel}` sub-streams.
  - [x] **M6.0 `svar2-codec` seam crate (done).** The key ↔ `(ILEN, ALT)` encode/decode
    seam is extracted out of `rvk.rs` into the dependency-light, crates.io publish-ready
    **`svar2-codec`** workspace crate (std-only, zero runtime deps), so genoray and gvl
    share one decoder. Verified: 16 codec tests green (PEXT↔SWAR parity, all key-lane
    round-trips), `cargo publish --dry-run` clean. Publishing to crates.io is a maintainer
    action, still pending (it release-gates M6b's gvl side alongside the `svar-2` PyPI
    release). See
    [`../superpowers/plans/2026-07-02-svar2-codec-crate.md`](../superpowers/plans/2026-07-02-svar2-codec-crate.md).
  - [x] **M6.1 consumer spine (done).** `src/spine.rs` (`KeyRef` / `gather_keys` / `merge_keys`)
    + `overlap_batch` / `BatchResult` / `decode_hap` in `src/query.rs` deliver the batched
    two-channel spine; SNPs re-expand via `svar2_codec::snp_code_to_key`; `overlap_sample`
    re-expressed on the spine (M5 tests are the regression oracle); cross-checked by
    `tests/test_batch.rs`. See [`architecture.md`](architecture.md#python-decode-path) and
    the design spec [`../superpowers/specs/2026-07-02-svar2-m6-consumer-interfaces-design.md`](../superpowers/specs/2026-07-02-svar2-m6-consumer-interfaces-design.md).
    Remaining for M6: the shared PyO3 seam is **M6a**; raw `BatchResult` exposure and
    dense-window subsetting are **M6b**; `seqpro.rag.Ragged` materialization is **M6c**.
  - [x] **M6a. PyO3 query foundation (land first).** *(shipped: `numpy` dep + `src/py_convert.rs`
    Rust→numpy helpers, `PyContigReader` pyclass over `ContigReader::open` registered in `_core`,
    and a Python `SparseVar2` skeleton reading `meta.json`; the `BatchResult` → numpy contract was
    frozen in the M6a carve-out.)* The shared seam both consumers stand
    on. Adds the `numpy` crate, a `PyContigReader`
    pyclass over `ContigReader::open` + `_core` registration, shared array-conversion
    helpers, a Python `SparseVar2` skeleton (reads `meta.json`), and **freezes the
    `BatchResult` → numpy contract** so M6b and M6c code against a fixed interface.
    **Merges before the M6b/M6c fan-out.** See the design spec's
    [M6a section](../superpowers/specs/2026-07-02-svar2-m6-consumer-interfaces-design.md).
- [ ] **M6b. gvl Rust variant interface.** *(built first among M6 consumers; on top of
  M6a, in parallel with M6c)* The primary consumer. genoray returns a **two-channel** query result — `var_key` gathered per-hap
  inline (no dedup, no barrier) + a shared decode-once `dense` table with per-hap presence
  bitmasks — and gvl's Rust core consumes it via a **two-source splice** (`var_key ⋈
  dense` in position order), calling `svar2-codec` inline to decode keys straight into the
  reconstructed haplotype / re-aligned track with no intermediate allele buffer. Track
  re-align needs only `ilen`/`deletion_len`, no alleles. **Cross-repo (genoray +
  GenVarLoader) and release-gated:** the gvl PR is only mergeable after `svar2-codec` is
  on crates.io and `svar-2` is on PyPI; sequence managed manually. genoray-side work is
  independently mergeable.
- [ ] **M6c. Python decode → `seqpro.rag.Ragged` + region variant counts.** *(on top of
  M6a, in parallel with M6b)* The analysis
  consumer. Materialize decoded query results into a `seqpro` `Ragged` record
  (`from_fields`: `pos` / `ilen` numeric + `allele` opaque-string, shared variant-axis
  offsets, shape `(ranges, samples, ploidy, None)` — byte-identical to gvl's
  `RaggedVariants`), and provide a **decode-free** variants-per-`(sample, ploid)` count
  (offset diffs + dense-mask popcount) as the simplified replacement for SVAR 1.0's
  `SparseVar.var_ranges` (variant indices no longer fit the data model — there is no
  unified variant table).

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

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
  `*`/`.` alleles are skipped. *(Making that hard rejection an opt-in skip instead of a
  panic is [M13](#beyond-mvp).)*
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
- [~] **M6. Query decode core.** The shared spine both consumers build on. M5's
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
- [~] **M6b. gvl Rust variant interface.** *(built first among M6 consumers; on top of
  M6a, in parallel with M6c)* The primary consumer. genoray returns a **two-channel** query result — `var_key` gathered per-hap
  inline (no dedup, no barrier) + a shared decode-once `dense` table with per-hap presence
  bitmasks — and gvl's Rust core consumes it via a **two-source splice** (`var_key ⋈
  dense` in position order), calling `svar2-codec` inline to decode keys straight into the
  reconstructed haplotype / re-aligned track with no intermediate allele buffer. Track
  re-align needs only `ilen`/`deletion_len`, no alleles. **Cross-repo (genoray +
  GenVarLoader) and release-gated:** the gvl PR is only mergeable after `svar2-codec` is
  on crates.io and `svar-2` is on PyPI; sequence managed manually. genoray-side work is
  independently mergeable.
  *Built and benchmarked, blocked only on the release gate:* the genoray-side two-channel
  result + query-only `genoray_core` (M6a/M6d/M6e) are shipped on `svar-2`. The gvl side is
  the complete, reviewed **[mcvickerlab/GenVarLoader#266](https://github.com/mcvickerlab/GenVarLoader/pull/266)**
  (draft, branch `svar2-m6b-kernel`, 69 commits ahead of gvl `main`): `.svar2` is a supported
  `gvl.write` source (write-time ranges cache) and a live `Dataset` read backend reconstructing
  all four output modes via one read-bound all-Rust FFI per read (zero `SearchTree`/dense-union
  rebuild, byte-identical to the `.svar`/union oracle). It also adds the **`variant-windows`**
  sequence mode and **`unphased_union`** ploidy folding, and carries the read-bound `getitem`
  perf optimizations (B1–B5) — extensively benchmarked, profiled, and optimized (see the PR and
  its `tmp/svar2_mvp/` harness); storage win is real (1.46–5.67×). It dev-wires genoray via
  local path-deps + a local wheel and **will not build upstream until genoray `svar-2` is
  published**; merge only after (1) genoray `svar-2` released, (2) path-deps → published
  versions (`svar2-codec` on crates.io, `genoray_core` pinned by git tag, `genoray` wheel from
  PyPI), (3) pyo3/numpy pins reconciled. See the ship plan
  [`../superpowers/specs/2026-07-07-svar2-mvp-ship-plan-design.md`](../superpowers/specs/2026-07-07-svar2-mvp-ship-plan-design.md).
- [x] **M6c. Python decode → `seqpro.rag.Ragged` + region variant counts.** *(on top of
  M6a, in parallel with M6b)* The analysis
  consumer. Materialize decoded query results into a `seqpro` `Ragged` record
  (`from_fields`: `pos` / `ilen` numeric + `allele` opaque-string, shared variant-axis
  offsets, shape `(ranges, samples, ploidy, None)` — byte-identical to gvl's
  `RaggedVariants`), and provide a **decode-free** variants-per-`(sample, ploid)` count
  (offset diffs + dense-mask popcount) as the simplified replacement for SVAR 1.0's
  `SparseVar.var_ranges` (variant indices no longer fit the data model — there is no
  unified variant table).
  *Done:* `python/genoray/_svar2_decode.py` (`_DecodeMixin`) adds `SparseVar2.decode(contig,
  regions) -> seqpro.rag.Ragged` (record with `pos`/`ilen`/`allele` fields, shape `(R, S, P,
  None)`, empty ALT for pure deletions) and the decode-free `region_counts(contig, regions) ->
  (R, S, P)` offset-diff + dense-popcount count. Wired into `SparseVar2` via `_DecodeMixin`;
  covered by `tests/test_svar2_decode.py` (the SVAR2 pytest suite is green).
- [x] **M6d. Search/gather split for write-time overlap caching (shipped, `svar-2`).**
  Splits the fused `query::overlap_batch` into a **search-only** `find_ranges` (runs every
  `SearchTree::new` and returns a compact `RangesBundle` of index ranges), a **tree-free**
  `gather_ranges` (pure slicing + `q_start < v_end` left-overlap re-check + k-way merge, no
  trees — a thread-local `SearchTree` build counter backs an integration test proving it
  builds zero trees), and a fused `read_ranges = gather_ranges(find_ranges(...))` wrapper.
  Lets a downstream cache (gvl `write`) run the interval search **once** at write time and
  replay it at read time with no `SearchTree::build`. Exposed on `PyContigReader` and the
  Python `SparseVar2` (`find_ranges`/`gather_ranges`/`read_ranges`, each with `samples=`
  subsetting matching every other `SparseVar` range method; `out=` streaming on `find_ranges`
  writes the bundle into caller-preallocated arrays so `gvl.write` can stream straight to a
  memmap). **Additive:** `overlap_batch` stays byte-unchanged; the byte-identical parity
  contract `overlap_batch ≡ read_ranges ≡ gather_ranges(find_ranges(...))` is pinned
  field-for-field by cargo + pytest tests, and `overlap_batch` remains the `decode`-validated
  oracle (`decode = overlap_batch → decode_hap`). See the design plan
  [`../superpowers/plans/2026-07-03-svar2-genoray-search-gather-split.md`](../superpowers/plans/2026-07-03-svar2-genoray-search-gather-split.md).
  *Done:* `RangesBundle` + `find_ranges`/`gather_ranges`/`read_ranges` in `src/query.rs`;
  `PyContigReader` bindings in `src/py_query_ranges.rs`; `SparseVar2` methods in
  `python/genoray/_svar2_batch.py`; tests `tests/test_ranges_split.rs` + `tests/test_svar2_ranges.py`.
  *Open question (follow-up):* whether to convert the read path to **fully read-bound** by also
  caching the region-independent dense union (so the read replay touches no per-query merge at
  all) — **resolved in M6e**, which eliminates the contig-wide dense union entirely via a
  per-class gather.

- [x] **M6e. Read-bound per-class gather + query-only build (shipped, `svar-2`).** Answers
  M6d's open question and gives gvl an htslib-free Rust path-dep on genoray. Two additive
  changes on top of M6d:
  - A **`conversion` cargo feature (default-on)** gates every htslib-touching module, so
    `cargo build --no-default-features` compiles the read/query core alone (no `rust-htslib`).
    That is exactly what gvl links as `genoray_core = { path = …, default-features = false }`.
    The default (wheel) build is behavior-unchanged. (The repo's test suites run under
    `--no-default-features --features conversion`: default `extension-module` breaks
    test-binary linking, and the e2e tests still need htslib.)
  - `find_ranges` additionally emits per-class `dense_snp_range` / `dense_indel_range`
    (each per-region, computed by a per-class `SearchTree` at search time), and a new
    `gather_ranges_readbound` slices each on-disk dense **class** window directly into a
    split-dense `BatchResultSplit` (var_key merged per hap as before; dense split per class),
    **never calling `dense_union()`** and building **zero** `SearchTree`s — eliminating M6d's
    O(N_contig) per-read dense residual. A flat-per-query `gather_haps_readbound` is the
    primitive gvl links for its arbitrary-`(region, sample)` reads. The Python `find_ranges`
    dict exposes the two new `(R, 2)` **i32** range keys (matching the sibling `dense_range`;
    streamable via the existing generic `out=` path) for the gvl write cache.
    Perf note: because the two per-class ranges are computed unconditionally, `find_ranges`
    (and therefore `read_ranges`) now builds **2 extra per-region `SearchTree`s**
    (dense_snp/dense_indel overlap) for *every* caller, including union-only readers that
    never consume `dense_snp_range`/`dense_indel_range`. This is a small region-level cost —
    comparable to the existing `dense_union` region overlap `SearchTree` — not the dominant
    per-hap tree builds M6d already removed.
  **Additive:** M6d's `find_ranges`/`gather_ranges`/`read_ranges`/`overlap_batch`/`BatchResult`
  stay byte-unchanged as the parity oracle. Parity is pinned per hap against both the union
  path and the `decode_hap` oracle, with a zero-`SearchTree` control test and both dense
  classes (SNP + indel) exercised. See the design plan
  [`../superpowers/plans/2026-07-04-svar2-genoray-readbound-gather.md`](../superpowers/plans/2026-07-04-svar2-genoray-readbound-gather.md).
  *Done:* `conversion` feature (`Cargo.toml` + `src/lib.rs` cfg-gates); per-class
  `dense_snp_overlap`/`dense_indel_overlap` + `RangesBundle` fields + `BatchResultSplit` +
  `gather_ranges_readbound`/`gather_haps_readbound` in `src/query.rs`; the two new dict keys in
  `src/py_query_ranges.rs`; tests `tests/test_query_only_build.rs` + `tests/test_readbound_gather.rs`
  + `tests/test_py_ranges_readbound.py`.

### Beyond MVP

- [ ] **M7. PGEN → SVAR2 conversion.** Likely requires FFI to C++ / `pgenlib`.
- [ ] **M8. Merge / split by contig.** Cheap because contigs are independent on disk.
  See [Format constraints](data-model.md#format-constraints-and-non-goals).
- [ ] **M9. Subset by region.** Non-MVP. Doesn't affect cost-model calculations (no
  new variants; only shrinks variant tables), so it should be cheap.
- [x] **M13. Opt-in skip for out-of-scope alleles during conversion.** Today the reader
  treats symbolic (`<DEL>`, `<INS:ME:*>`, `<DUP>`, `<INV>`, …) and breakend ALTs as a
  hard error: `normalize::atomize_record` returns `SymbolicAllele` and
  `vcf_reader::decompose_current_record` `.expect()`s on it, so a single out-of-scope
  record aborts the entire conversion. Add an opt-in **skip-out-of-scope** mode to the
  conversion entry point (`run_conversion_pipeline` / the `SparseVar2` writer) that drops
  symbolic/breakend records — exactly as `*`/`.` alleles are already skipped in
  `atomize_record` — and reports a count of what was dropped, rather than panicking.
  Real short-read VCFs routinely carry a small tail of SV symbolic records that are
  legitimately out of SVAR2's scope (e.g. 1000G chr21: 1368 symbolic ALTs out of ~1.0M,
  surfaced during the gvl SVAR2 MVP validation), and requiring an external
  `bcftools view -V other,bnd` pre-filter is avoidable friction. The strict default
  (error) stays, so any silent variant drop is explicitly opt-in.
  *Done:* `atomize_record` takes an opt-in `skip_out_of_scope` flag and returns the
  dropped-allele count instead of erroring (`normalize.rs`); the count threads through
  `VcfChunkReader` → `process_chromosome` → `run_conversion_pipeline`, which also gained an
  optional `reference_path` (`Option<String>`) — omitting it skips `validate_ref`/`left_align`
  for pre-normalized input — and now returns the total dropped count; a new `index_vcf` PyO3
  helper builds a `.csi` for un-indexed input; the Python `SparseVar2.from_vcf(out, source,
  reference=None, *, no_reference=False, skip_out_of_scope=False, ...)` classmethod wraps all
  of the above (exactly one of `reference`/`no_reference` required, auto-indexes, VCF/BCF-only,
  returns the dropped count as `int`); and the `genoray write` CLI now defaults to SVAR2
  (`--reference` XOR `--no-reference`, `--no-symbolic`/`--no-breakend` coupled → skip and print
  the dropped count) with `genoray write svar1` for the previous SVAR 1.0 behavior. See the
  design spec [`../superpowers/specs/2026-07-03-svar2-cli-write-and-m13-skip-design.md`](../superpowers/specs/2026-07-03-svar2-cli-write-and-m13-skip-design.md).

- [x] **M14. Parallel presence-packing in the reader (single-/few-contig throughput).**
  A single-contig VCF→SVAR2 run leaves ~24 cores idle: the per-contig pipeline
  (M1) fans across contigs via rayon, so one contig uses only its 4 pipeline
  threads + htslib decode threads. This milestone puts the idle cores to work on
  the reader's genotype-presence bit-packing, keeping output **byte-identical**.
  See the design plan
  [`../superpowers/plans/2026-07-07-svar2-parallel-reader.md`](../superpowers/plans/2026-07-07-svar2-parallel-reader.md).
  - **Thread budget (`src/budget.rs`).** `MAX_HTSLIB_THREADS` raised 4→8 (Lever 0);
    `ThreadPlan` gains `processing_threads` = idle cores after the pipeline + htslib
    threads across all concurrent contigs (`usable − concurrent·(pipeline+htslib)`,
    floored at 1). `plan_thread_budget` computes it in both branches.
  - **Processing pool + word-aligned parallel packing (`src/orchestrator.rs`,
    `src/vcf_reader.rs`).** `process_chromosome` builds a dedicated rayon pool sized
    to `processing_threads` and hands it to the reader thread. The reader stays
    sequential for all *ordering* work (htslib iteration, atomize, left-align, the
    reorder heap, chunk cutting — all **frozen**); only presence packing fans out.
    Variant row `vi` occupies bits `[vi·columns, (vi+1)·columns)`, so a variant
    block of `g = 64/gcd(columns,64)` rows spans exactly `words_per_block =
    columns/gcd(columns,64)` whole `u64` words — `par_chunks_mut(words_per_block)`
    hands each rayon task a **word-disjoint** slice, so there are no shared boundary
    words and no atomics. `PendingAtom.gt` moved `Rc`→`Arc` for cross-thread
    sharing; packing self-gates to sequential below `PARALLEL_MIN_VARIANTS = 512`
    or without a ≥2-thread pool. A 300-case proptest pins `pack_presence_par` ==
    `pack_presence_seq` bit-for-bit across shapes crossing block boundaries,
    missing (`-1`), and out-of-range alleles.
  - **Measured (32 cores, single contig chr21; oracle-verified byte-identical on
    both germline `chr21.filt.bcf` and gdc `gdc.chr21.filt.bcf`).** gdc:
    **1051 s → 981 s (−6.7 %)** from the parallel packing. germline unchanged
    (36.3 s — already fast, not packing-bound). The htslib-cap bump (Lever 0) was
    **inert on its own** (1051 s → 1046 s, within noise).
  - **Why it stops there — the reader is htslib-input-bound, not decode-bound.** A
    frame-pointer `perf` profile of this build on gdc puts **78 %** of samples in the
    reader thread, whose identifiable self-time is dominated by htslib **input**
    work: `inflate` (BGZF decompress) ≈ 32 %, `vcf_parse` ≈ 9 %, name tokenization
    ≈ 14 %. The genoray-side GT-decode and packing do **not** surface as a
    meaningful fraction (nothing at ≥0.05 %). This is why raising the htslib decode
    cap didn't help — the reader inflates/parses on its own serial path — and why
    the plan's optional **Task 6 (fuse raw-GT decode into the parallel pass) was
    declined**: its premise (a meaningful sequential GT-decode cost) is falsified by
    the profile, so it would add sample-subset/ploidy correctness risk for ~0 gain.
    Further single-contig speedup requires attacking htslib BGZF inflate / VCF
    parsing (e.g. more effective threaded block decode, or a leaner parse path), not
    the genoray packing.
  *Done:* `src/budget.rs` (cap + `processing_threads`), `src/vcf_reader.rs`
  (`Rc`→`Arc`, `pack_row`/`pack_presence_seq`/`pack_presence_par`/`gcd`, gated
  dispatch, proptest), `src/orchestrator.rs` + `src/lib.rs` (pool built and threaded
  through). No public API change. Rust suite + 525 pytest green; both oracle hashes
  byte-identical.

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

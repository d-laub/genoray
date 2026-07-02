# SVAR 2.0 — M5 (part 2b): `(range, sample)` query — disk integration (design)

> Partial spec for roadmap milestone **M5** in
> [`docs/roadmap/svar-2.md`](../../roadmap/svar-2.md).
> Supplements: [`architecture.md`](../../roadmap/architecture.md#query-path),
> [`data-model.md`](../../roadmap/data-model.md#overlap-queries-and-deletions).
> Branch off `svar-2`, own worktree. **Parallel-safe** with the `max_del` post-pass spec
> (shares only the frozen `max_del.npy` contract) and the M2b spec (disjoint files).

## Context

M5 part 1 shipped the format-independent core (`src/search.rs`: `SearchTree`,
`overlap_range`), depending only on in-memory `u32` slices. This spec wires that core to a
**finished SVAR2 contig on disk** to answer a `(range, sample)` query: given a contig, a
`[q_start, q_end)` region, and a sample, return that sample's variants (both haplotypes)
overlapping the region.

A single contig spreads a sample's variants across up to four sub-streams today
(`var_key/{snp,indel}` and `dense/{snp,indel}`; `pointer/*` arrives with M11). Each is
independently position-sorted. Answering the query means: search each relevant sub-stream,
combine the hits in position order, and gather the sample's genotype/allele for each.

## Scope

**In:**

- **Sidecar loading** for a contig: mmap `positions.bin` (raw LE `u32` → build
  `SearchTree`), `alleles.bin` (keys), `offsets.npy` (per-column bounds), `genotypes.bin`
  (dense), and `max_del.npy` / `dense/max_del.npy` (from the post-pass spec's contract).
- **Per-sub-stream overlap** using `overlap_range`:
  - `var_key/*` — per-column `(sample, ploid)`: slice the column via `offsets.npy`, build a
    `SearchTree` over that column's positions, run `overlap_range` with the column's
    `max_del[s,p]` (indel) or `0` (snp). Reconstruct `v_ends` from keys via the `rvk`
    decoder (deletion length), respecting the encoding seam.
  - `dense/*` — one shared position table per contig: run `overlap_range` once with the
    per-contig `dense/max_del` (indel) / `0` (snp), then select the sample's carried
    variants from the 1-bit `(sample, ploid, variant)` matrix (`genotypes.bin`, read via
    `BitGrid3`/`bits.rs`) within the returned `[s_idx, e_idx)`.
- **Sorted union** across the sub-streams: each `overlap_range` yields an index range into
  one sorted stream; merge the referenced `(position, key/genotype)` records into one
  **position-sorted** result. A k-way merge over ≤4 already-sorted runs
  (see [`architecture.md`](../../roadmap/architecture.md#python-decode-path) — the
  union-shaped dual of sorted intersection).
- **Genotype gather**: assemble the per-`(sample, ploid)` overlapping calls into an
  **intermediate Rust-side result** — positions, decoded `(ILEN, ALT)` (via the `rvk`
  decoder + LUT for long alleles), and which haplotype(s) carry each. Shape TBD with M6
  but must be a plain owned struct, not a Python type.
- Single-query API first; a thin batched wrapper (many ranges) is a follow-up — SVAR 1.0
  batches, and the core is single-query.

**Out (deferred):**

- **Producing** `max_del.npy` — the post-pass spec. This spec **consumes** it, developing
  against a hand-written fixture until that lands.
- **Python-facing structs/classes** and the `SparseVar2` reader class — **M6**. This spec
  stops at the intermediate Rust result + (optionally) a minimal PyO3 accessor returning
  arrays, only if needed to test end-to-end.
- `pointer` representation (M11): design the union so adding a 5th/6th sub-stream is a list
  entry, not a rewrite, but do not implement it.
- Left-alignment (M2b) — orthogonal; this spec reads whatever positions are on disk.

## Design

- New module `src/query.rs` (or extend `search.rs` with a disk-facing submodule; keep the
  pure core in `search.rs` untouched so its proptests stay valid).
- **`ContigReader`** — opens a contig dir, mmaps the sidecars, exposes typed views:
  per-column `var_key` slices, the shared `dense` table + matrix, the LUT, and the loaded
  `max_del` arrays. Owns lifetimes so the mmaps outlive the query.
- **`overlap_sample(reader, sample, q_start, q_end) -> QueryResult`**:
  1. For `p in 0..ploidy`: run `var_key/snp` + `var_key/indel` overlaps for column
     `(sample, p)`; run `dense/snp` + `dense/indel` overlaps (shared) and filter to
     variants where the sample/ploid bit is set.
  2. Decode each hit's `(ILEN, ALT)` and true `v_end` via `rvk` (+ LUT).
  3. k-way merge the ≤4 runs per haplotype into one position-sorted list.
  4. Return `QueryResult { per_hap: [Vec<Call>; ploidy] }` (owned; `Call` = position,
     ILEN, ALT bytes or LUT handle, allele/dosage).
- **`v_ends` reconstruction** must go through a single `rvk` decoder entry point
  (`deletion_len` / `decode_key`), not inline bit math — same seam discipline as the core.
- **Empty/degenerate:** no overlap ⇒ `s_idx == e_idx` ⇒ empty run; a pure-SNP contig ⇒
  all `max_region_length == 0`; a contig with no dense dir ⇒ skip dense runs.

## Files

- **New:** `src/query.rs` (`ContigReader`, `overlap_sample`, `QueryResult`, `Call`, the
  k-way sorted union).
- **Reuse:** `src/search.rs` (`SearchTree`, `overlap_range` — unchanged), `src/rvk.rs`
  (key decode + LUT), `src/bits.rs`/`BitGrid3` (dense matrix reads), `src/layout.rs`
  (paths — add sidecar path helpers if missing), `memmap2`.
- **Fixture:** a small committed SVAR2 dir (or a builder that converts a tiny VCF) plus a
  hand-written `max_del.npy` for pre-integration testing.

## Testing

- **Unit:** `ContigReader` loads sidecars with correct shapes/dtypes; `v_end`
  reconstruction from keys matches `rvk` decode.
- **Oracle proptest:** against a brute-force reference that holds the whole contig in
  memory and linearly finds every overlapping variant a sample carries — assert
  `overlap_sample` returns exactly those, in position order, across random contigs
  (mixed SNP/indel, var_key/dense routing, random deletions, empty and no-overlap
  queries, deletions spanning `q_start`).
- **Cross-check the dense/var_key split:** a variant routed to dense vs. var_key must yield
  identical query results for the carrying sample (routing is an internal storage choice,
  invisible to the query).
- **e2e:** convert a small VCF → query a known region for a known sample → assert calls
  match the VCF. Once the post-pass spec lands, drop the fixture `max_del.npy` and use the
  produced one; results must be identical.

## Open questions

- **Result shape** — finalize with M6 so the Rust `QueryResult` maps cleanly to the Python
  struct without a second reshuffle. Keep it array-of-structs vs. struct-of-arrays open
  until M6's decode needs are known; lean SoA (columnar) since M6 wants a fast sorted union
  and numpy-friendly output.
- **Union placement** — do the sorted union in Rust (here) vs. Python (M6)?
  Recommendation: Rust, since `search.rs` already returns index ranges and the union is
  cheap over ≤4 runs; M6 then only decodes/boxes. Revisit if M6 needs cross-contig unions.
- **Batched queries** — single-query now; confirm the batch wrapper shape when M6's access
  pattern (one region vs. many) is fixed.

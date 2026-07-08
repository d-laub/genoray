# SVAR 2.0 M6 — Consumer Interfaces (gvl Rust + Python decode)

> **Status:** design approved, pre-implementation · **Epic:** SVAR 2.0 (`svar-2` branch) ·
> **Supersedes:** the single-line M6 ("Python decode") in [`../../roadmap/svar-2.md`](../../roadmap/svar-2.md).
>
> This spec designs how SVAR2 query results reach their two consumers and splits the
> old M6 into three milestones. It reconciles with
> [`../../roadmap/data-model.md`](../../roadmap/data-model.md) and
> [`../../roadmap/architecture.md`](../../roadmap/architecture.md); where it changes the
> data/architecture model, those docs are updated in the same PR that implements the change.

## Context

M1–M5 built the SVAR2 store and the format-independent overlap search. The remaining
MVP question is: **how do query results become usable?** SVAR 1.0 answered this by
returning a `seqpro` `Ragged[u32]` of *variant indices* into a single unified variant
table, which consumers then joined against for `POS`/`ILEN`/`ALT`.

**SVAR2 has no unified variant table.** Variants are inline-encoded across
`var_key`/`dense` representations, each split into `snp/` (2-bit) and `indel/` (32-bit)
sub-streams. A carrier's variant lives inline in its `(sample, ploid)` stream (`var_key`)
or as a bit in a shared per-variant matrix (`dense`). There is no global variant id to
return. So M6 must **decode query results into actual variant records** (position + ILEN
+ ALT bytes) rather than indices.

### Consumers (in priority order)

1. **GenVarLoader (gvl)** — the primary consumer, now with a Rust core
   (`pyo3 0.28` + `numpy 0.28` + `ndarray`). Its kernels reconstruct haplotypes and
   re-align tracks across indels. Both consume a **per-query variant table** (`v_starts:
   i32`, `ilens: i32`, `alt_alleles: u8` + `alt_offsets: i64`) plus **sparse
   per-`(region, sample, ploid)` genotype references**, and gather alleles on the fly.
   gvl's public output type `RaggedVariants` subclasses `seqpro.rag.Ragged`.
   - `reconstruct_haplotypes_from_sparse` (`src/reconstruct/mod.rs`) needs, per variant,
     genomic `start` + `ilen` + `alt` bytes, and per `(query, hap)` a sparse variant list.
   - `shift_and_realign_tracks_sparse` (`src/tracks/mod.rs`) needs **only** `start` +
     `ilen` per variant (no allele bytes) plus the same sparse references.
   - gvl's `ilen` convention is anchored/atomized: its `v_ref_end = v_pos − min(0, ilen)
     + 1` is consistent with genoray's `ILEN = len(ALT) − len(REF)` (SNP `ilen=0` →
     1 ref base; anchored 3-base DEL `ilen=−3` → 4 ref bases; anchored INS `ilen=+3` →
     1 ref base). The `+1` in gvl's kernel comment describes the `ref_end` computation,
     **not** the `ilen` definition; the definitions align. Locked by a round-trip test
     (see [Verification](#verification)).

2. **Python decode into `seqpro.rag.Ragged`** — general analysis: counting variants per
   `(sample, ploid)` in a set of regions (a simplified replacement for SVAR 1.0's
   `SparseVar.var_ranges`, which no longer maps onto the data model since there are no
   variant indices), and materializing variant records for downstream work.
   `seqpro`'s `Ragged.from_fields` builds a record whose fields share one variant-axis
   offsets object; the canonical layout is `(n_ranges, n_samples, ploidy, None)` — the
   same object gvl's `RaggedVariants` is.

## Design goals

- **Performance over internal ergonomics.** None of this is user-facing; optimize the
  machine, not the API aesthetics.
- **No duplication of hot (dense / high-AF) alleles** across carriers.
- **No dedup barrier** on the dominant (low-AF `var_key`) path; scale linearly with
  `rayon` over `(region, sample, ploid)`.
- **Decode-free counting.** "Variants per `(sample, ploid)`" must never touch allele
  bytes.
- **Preserve the encoding seam.** Key-layout knowledge stays in exactly one place.

## Architecture

Both consumers share one spine and differ only at the terminal materialization:

```
overlap search (finish M5)  →  per-(region, sample, ploid) index ranges
        │
        │  sorted union across {var_key, dense} × {snp, indel} sub-streams
        │  decode seam (key + LUT → ILEN, ALT), via svar2-codec
        ▼
   ┌────┴───────────────────────────────┐
   ▼                                     ▼
M6b: gvl Rust path                  M6c: Python analysis path
two-channel sparse, fused decode    materialized seqpro Ragged record
(no intermediate allele buffer)     + decode-free region variant counts
```

### Shared spine (M6)

1. **Finish M5 disk integration** *(M5).* Turn a `(contig, ranges, samples)` query into index
   ranges per `(sample, ploid)` per sub-stream: consume `max_del.npy` for the `indel/`
   leftward bound, read the on-disk sidecars (`positions.bin`, `offsets.npy`, dense
   `genotypes.bin`), and resolve `[s_idx, e_idx)` via the existing `overlap_range`
   (`src/search.rs`). SNP sub-streams use a plain `[LB, UB)` (their `max_del ≡ 0`).

2. **`svar2-codec` crate.** Extract the key ↔ `(ILEN, ALT)` seam out of `src/rvk.rs`
   into a new, dependency-light **workspace crate published to crates.io** (no `pyo3`, no
   `htslib`). It owns the SNP 2-bit and indel 32-bit layouts and exposes:
   - `ilen(key: u32) -> i32`
   - `deletion_len(key: u32) -> u32`
   - `decode_alt_into(key: u32, lut: LutView, out: &mut Vec<u8>)` (and/or a borrowing
     `decode_alt<'a>(key, lut) -> Cow<'a, [u8]>`)
   - the format-version constant it implements.

   genoray's `pyo3` crate depends on `svar2-codec` (single source of truth — the seam is
   no longer duplicated). gvl depends on the **published crates.io version**, so gvl
   decodes raw keys inline inside its reconstruct/realign loops with no intermediate
   allele buffer crossing FFI. Track re-align calls only `ilen`/`deletion_len` — zero
   allele materialization.

3. **Uniform in-memory key** *(M6.1).* On disk stays split (`snp/` 2-bit-packed, `indel/` 32-bit)
   for space. A query result **re-expands SNPs into the uniform 32-bit key** (`ILEN=0`,
   ALT base inline — the pre-M1-split encoding) via `svar2_codec::snp_code_to_key`.
   Consumers then have a single `decode(u32, lut)` path, and gvl's per-hap merge drops from
   3-way (snp/indel/dense) to 2-way (var_key/dense). Cost: one shift per SNP in genoray
   (the 2-bit code is already unpacked at read time).

4. **Sorted union** *(M6.1).* Branch-light merge of position-sorted sub-streams (the query hot
   loop flagged in [`architecture.md`](../../roadmap/architecture.md#python-decode-path)).
   `overlap_batch` + `merge_keys` in `src/spine.rs` pre-union `snp` + `indel` *within*
   `var_key` (into one per-hap stream) and *within* `dense` (into one shared table).
   The final `var_key ⋈ dense` merge is done by the consumer, interleaved with its own work
   (splice / materialize).

### M6a — PyO3 query foundation (land first)

M6.0 + M6.1 landed the whole spine in Rust (`overlap_batch` → `BatchResult`,
`decode_hap`), but **nothing query-related crosses into Python yet** — `src/lib.rs`
exposes only `run_conversion_pipeline`, there is no `numpy` dependency, and there is no
way to open a finished contig from Python. Both M6b-genoray and M6c stand on that missing
seam, and both would otherwise build the same numpy plumbing. So M6a carves it out and
**lands + merges before the M6b/M6c fan-out**:

1. **`numpy` dependency.** Add the `numpy` (rust-numpy) crate to `Cargo.toml` at the
   release that matches the in-tree `pyo3 0.29` (rust-numpy tracks pyo3's version — pick the
   matching one from crates.io at implementation time). The FFI *contract* is numpy arrays
   and is version-independent across the boundary; gvl's own crate may pin a different
   `pyo3`/`numpy` pair on its side.

2. **`PyContigReader` pyclass** (`src/py_query.rs`) wrapping
   `ContigReader::open(base_out_dir, chrom, n_samples, ploidy)` — the signature already
   matches. `meta.json` stays **Python-read** (its established convention): the Python
   `SparseVar2` class loads `samples`/`ploidy` and passes them in. Register the class in
   `_core`.

3. **Shared numpy conversion helpers** (`Vec<u32|i32|usize> → PyArray1`, offset arrays as
   `i64`) that both consumers reuse, so the array plumbing exists once.

4. **Python `SparseVar2` skeleton** in `python/genoray/`: read `meta.json`, construct a
   `PyContigReader` per contig. The query methods themselves land in M6b/M6c.

**Frozen `BatchResult` → numpy contract.** M6b and M6c code against this fixed dtype/shape
table so Phase-1 streams don't wait on each other. `H = n_regions · n_samples · ploidy`,
hap-slices in region-major order `h = (r·n_samples + s)·ploidy + p`:

| Array | dtype | shape | meaning |
| --- | --- | --- | --- |
| `vk_pos` | `i32` | `[·]` | flat var_key positions, ragged by hap |
| `vk_key` | `u32`→exposed as `i32` bit-pattern | `[·]` | flat uniform 32-bit keys, aligned with `vk_pos` |
| `vk_off` | `i64` | `[H+1]` | CSR offsets slicing `vk_*` per hap |
| `dense_pos` | `i32` | `[D]` | shared dense-union positions, sorted |
| `dense_key` | `u32`→`i32` bits | `[D]` | shared dense-union uniform keys |
| `dense_range` | `i32` | `[R, 2]` | `[s, e)` into the dense arrays per region |
| `dense_present` | `u8` | `[·]` | per-hap LSB-first presence bitmask, concatenated |
| `dense_present_off` | `i64` | `[H+1]` | **bit** offsets into `dense_present` per hap |
| `lut_bytes` | `u8` | `[·]` | shared long-allele LUT bytes |
| `lut_off` | `i64` | `[·]` | LUT row offsets |

`u32` keys cross as their `i32` bit-pattern (numpy has no unsigned-friendly zero-copy path
gvl uses; both sides reinterpret). M6c's decoded product (`pos`/`ilen` `i32`, `allele`
`u8`+`str_offsets`) is a separate contract, already pinned in [M6c](#m6c--python-decode--seqproragragged--region-variant-counts).

Phase-1 coordination is then limited to `_core` registration and adding methods to the
same `PyContigReader` — separate `#[pymethods]` blocks (pyo3 0.29 allows multiple), one
per stream, so the two land without conflict.

### M6b — gvl Rust variant interface (two-channel, built first)

Per query (R regions × S samples × P ploidy), per contig, genoray returns:

**Shared, read-only (decode-once, no duplication):**
- `dense_pos: i32[D]`, `dense_key: u32[D]` — the dense variant table over the union of
  the query windows, position-sorted, SNPs re-expanded to uniform 32-bit keys.
- `lut_bytes: u8[·]`, `lut_off: i64[·]` — the shared long-allele LUT, referenced by any
  key with LSB set (both `var_key` and `dense`).

**`var_key` channel (sparse, per-hap, inline):**
- `vk_pos: i32[·]`, `vk_key: u32[·]` — flat, ragged by `(region, sample, ploid)` via
  `vk_off: i64` (an `(H+1,)` or `(2, H)` offsets object, `H = R·S·P`). Each hap's slice
  is already `snp`+`indel` position-unioned. Rare variants stay inline: no dedup, no
  barrier, `rayon` over haps.

**`dense` channel (per-hap presence):**
- `dense_range: i32[R, 2]` — each region's `[s_idx, e_idx)` into the dense table.
- The dense genotype bit-matrix **window slice** addressable per `(region, sample,
  ploid)` → a `(e_idx − s_idx)`-bit mask. A bitmask beats a carrier-index list here
  because dense = high-AF (≈16× smaller at AF ≈ 0.5). Exact multi-region packing of the
  hap-major on-disk matrix into the returned slice is the least-pinned detail — spec'd as
  a per-region column-range gather of `dense/{snp,indel}/genotypes.bin`
  (see [Open questions](#open-questions)).

**gvl-side change:** a two-source splice — per hap, walk the `vk` stream ⋈ its dense
set-bits in position order, calling `svar2_codec::decode` (or just `ilen` /
`deletion_len` for track re-align) directly into the output. Nothing is materialized
between genoray and gvl. gvl replaces its `_write_from_svar` densification + memmap-store
round-trip with a direct feed from this interface.

**FFI shape.** Consistent with gvl's existing boundary: `i32` for positions/keys/ranges,
`i64` for offsets, `u8` for LUT/masks — numpy arrays via `pyo3`/`numpy`, zero-copy on
inputs, GIL released for the heavy loops.

### M6c — Python decode → `seqpro.rag.Ragged` + region variant counts

Same search + union, two products:

1. **Materialized variant record.** For each `(region, sample, ploid)`, decode the merged
   `var_key` + `dense` variant list into flat buffers and build:
   ```python
   Ragged.from_fields({
       "pos":    Ragged.from_offsets(pos_i32,    (R, S, P, None), off),
       "ilen":   Ragged.from_offsets(ilen_i32,   (R, S, P, None), off),
       "allele": Ragged.from_offsets(alt_S1,     (R, S, P, None), off, str_offsets=str_off),
   })
   ```
   shape `(R, S, P, None)`, one shared `int64` offsets object, opaque-string alleles via
   `str_offsets`. This is byte-identical to gvl's `RaggedVariants`. Dense alleles are
   duplicated across carriers here — acceptable: this is the cold analysis path, and
   `seqpro`'s `Ragged` wants contiguous flat buffers regardless.

2. **Decode-free region variant counts** (the simplified `var_ranges` replacement).
   Variants-per-`(sample, ploid)` in a region = `vk_off` diffs + dense-mask `popcount`.
   Returns counts/offsets shaped `(R, S, P)` — no unified table, no variant indices, no
   allele decode.

## Cross-repo release sequencing

M6b spans two repos, both owned by us. The dependency is release-gated; **the user manages
the sequence.** The mergeability order is:

1. genoray (`svar-2`): implement M6 + M6b's genoray side, including the `svar2-codec`
   crate. Commit in genoray's `.claude/worktrees/`.
2. Release: publish `svar2-codec` (and the SVAR2 Python interface) — `svar2-codec` to
   **crates.io**, genoray/`svar-2` to **PyPI**.
3. gvl: implement the two-source kernel + direct-feed against the **published**
   `svar2-codec` crates.io version and the released genoray. Commit in
   GenVarLoader's `.claude/worktrees/`. **This PR is only mergeable after step 2.**

The genoray-side work (M6, M6b-genoray, M6c) is independently mergeable and does not wait
on gvl.

## Roadmap changes

Replace the single M6 in [`svar-2.md`](../../roadmap/svar-2.md) with:

- **M6 — Query decode core.** Finish M5 disk wiring (`max_del` consumer, sidecar reads,
  per-`(contig, sample, ploid)` index ranges); extract `svar2-codec`; uniform-key
  re-expansion; sorted union. *(M6.0 codec + M6.1 spine done.)*
- **M6a — PyO3 query foundation.** `numpy` dependency, `PyContigReader` pyclass +
  `_core` registration, shared array-conversion helpers, Python `SparseVar2` skeleton,
  and the frozen `BatchResult` → numpy contract. **Lands and merges before the M6b/M6c
  fan-out** — both consumers stand on it.
- **M6b — gvl Rust variant interface.** Two-channel sparse result + `svar2-codec`
  consumed by gvl's reconstruct/realign (two-source kernel). Cross-repo, release-gated.
  *Built first among the M6 consumers, on top of M6a.*
- **M6c — Python decode → `seqpro.rag.Ragged` + region variant counts.** Materialized
  record + decode-free count/ploidy shortcut.

M7–M12 keep their numbers.

## Verification

- **`ilen` convention round-trip.** A test converting known SNP/INS/DEL atoms → SVAR2 →
  decode, asserting genoray `ilen` and gvl's `ref_end` math agree for each class.
- **Codec parity.** `svar2-codec` decode is byte-identical to the pre-extraction
  `rvk.rs` decode (proptest over random keys + LUT spills).
- **Sorted-union oracle.** Union output equals a brute-force merged-and-sorted reference
  over all sub-streams for random queries.
- **Two-channel ≡ materialized.** For a random query, gvl's two-source splice over the
  two-channel form yields the same per-hap variant sequence as the M6c materialized
  `Ragged` (cross-check the two consumers against each other).
- **Count shortcut.** Decode-free counts equal `len()` of the materialized `Ragged`
  per `(region, sample, ploid)`.

## Open questions

- **Dense channel multi-region packing.** How to return the hap-major
  `dense/{snp,indel}/genotypes.bin` window slices across R regions with the least copy —
  a single concatenated bit-buffer + per-region/per-hap offsets, vs. per-region arrays.
  Resolve during M6b implementation against a benchmark.
- **`svar2-codec` LUT view type.** Whether the crate borrows the LUT as `(&[u8], &[i64])`
  slices or wraps a small `LutView` struct; affects how gvl passes its mmap'd LUT in.
- **Two-source vs. measured fallback.** The two-source kernel is the target. If its
  complexity outweighs the win for the initial gvl integration, fall back to merging the
  two channels into one gathered per-hap stream feeding gvl's existing single-table
  kernel (bounded dense duplication) — decide by measurement, not up front.

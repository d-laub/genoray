# SVAR1 → SVAR2 conversion (`SparseVar2.from_svar1`)

Status: **PARTIAL DRAFT — blocked.** The architecture below (native Rust
pipeline, entry seam, normalization contract, scope guards) is settled. The
**field carry-through** (dosages and any other INFO/FORMAT fields) is
intentionally left open because it depends on two specs that must land first:

1. **Write INFO/FORMAT fields** — the generalized SVAR2 sidecar-field write
   mechanism (dosage is the first FORMAT field; mutcat is the existing
   precedent).
2. **Read/output INFO/FORMAT fields** — wiring those sidecars into the SVAR2
   query/decode/output path.

Finish brainstorming this spec **after #1 and #2 land**. Do not start an
implementation plan for it before then.

Date: 2026-07-11

## Problem / motivation

Users with existing SVAR1 (`SparseVar`) stores want to migrate to SVAR2
(`SparseVar2`) without re-reading the original VCF/BCF — which may be gone, and
which is the expensive part: VCF→SVAR2 conversion is htslib-reader-bound
(≈78% of wall time; see
[`2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md`](2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md)).
SVAR2 is 1.46–5.67× smaller on disk and read-optimized, so migration is
worthwhile on its own.

A **native** SVAR1→SVAR2 path skips htslib entirely and exploits the fact that
SVAR1 is already stored in the layout the current pipeline works to *produce*.

## Key facts that shape the design

### On-disk layouts

**SVAR1** (`python/genoray/_svar/`), one directory:

- `metadata.json` — `SparseVarMetadata` (`version`, `samples`, `ploidy`,
  `contigs`, `fields: {name -> numpy dtype}`, mutcat stamps).
- `index.arrow` — Arrow IPC/Feather (written by polars `write_ipc`,
  zstd-compressed). Columns `CHROM`, `POS`, `REF`, `ALT` (comma-`Utf8` on disk),
  `ILEN`, plus any attrs. **Variant-major, contig-contiguous** (each contig is a
  single contiguous run of rows; variants sorted within a contig).
- `variant_idxs.npy` — `int32` (`V_IDX_TYPE`), **sample-major sparse**: for each
  `(sample, ploidy)` haplotype, the sorted **global** variant-row ids that are
  non-ref. Presence model is `geno == 1` → **biallelic** (one ALT per variant).
- `offsets.npy` — `int64`, Ragged offsets over `(n_samples, ploidy)`.
- Optional per-entry field arrays parallel to `variant_idxs.npy`, e.g.
  `dosages.npy` (`float32`), `mutcat.npy` (`int16`).

**SVAR2** (`python/genoray/_svar2.py` + `src/`), per-contig native store:

- Four sub-streams per contig — `var_key/{snp,indel}`, `dense/{snp,indel}`
  (`src/layout.rs`). var_key stores records **per call**; dense stores records
  **per distinct variant** plus a hap-major 1-bit genotype matrix. The
  **cost model** (`choose_representation`, `src/cost_model.rs`) routes each
  variant by carrier count.
- **Reference-relative diff format: REF is never stored.** SNP records carry a
  2-bit ALT code; indel ALTs live in the long-allele bank; indels are
  left-aligned/validated against the FASTA so REF ≡ reference substring at POS.
  Decode returns ALT only (empty for pure DELs); the consumer injects
  `ref[pos]`. This is why the left-alignment invariant is load-bearing.
- `meta.json` — `{format_version, samples, contigs, ploidy}`.
- Sidecars (mutcat today; INFO/FORMAT fields via specs #1/#2) live in per-contig
  dirs positionally aligned to each sub-stream's records
  (`2026-07-10-svar2-mutational-signatures-design.md` §2).

### The pipeline seam (the architectural insight)

Current conversion is built for **variant-major dense** sources (VCF/BCF, and
the intended PGEN/BED generalization):

```
Reader → DenseChunk (V,S,P bit grid + pos/ilen/alt)      [src/vcf_reader.rs]
       → dense2sparse_vk: transpose + cost-route          [src/rvk.rs]
       → SparseChunk { streams: var_key…, dense: DenseSubChunk… }  [src/types.rs]
       → merge_mini_sc                                     [src/merge.rs]
       → writer                                            [src/writer.rs]
```

**SVAR1 is already in the `SparseChunk` shape** — sample-major sparse. So the
native SVAR1 path **enters at the `SparseChunk` boundary**, skipping the dense
`(V,S,P)` grid and the `dense2sparse_vk` transpose entirely, and reuses the cost
model, `svar2-codec` key packing, `merge_mini_sc`, and the writer **unchanged**.

## Design (settled)

### Public API — thin Python shim, no SVAR1 parsing in Python

```python
@classmethod
def from_svar1(
    cls,
    out: str | Path,
    svar1: str | Path,
    reference: str | Path | None = None,
    *,
    no_reference: bool = False,
    skip_out_of_scope: bool = False,
    threads: int | None = None,
    overwrite: bool = False,
    long_allele_capacity: int = 8 * 1024 * 1024,
) -> int: ...
```

Mirrors `from_vcf` (return value = count of out-of-scope ALTs dropped). Python
reads only SVAR1's `metadata.json` (samples/ploidy/contigs) to marshal args and
call the Rust pyfunction; **all SVAR1 data parsing happens in Rust**.

### Normalization contract — mirror `from_vcf`

Exactly one of `reference` (FASTA) or `no_reference=True`:

- `reference` — validate + left-align indels against the FASTA. Because SVAR2 is
  reference-relative (REF recovered from FASTA), this is the path that yields a
  canonical store byte-comparable to `from_vcf(reference)` on the same variants.
- `no_reference=True` — trust SVAR1's existing normalization; pack ALTs as-is,
  skip left-alignment. Same trust model and caveat as `from_vcf(no_reference)`.
  (SVAR1 is not guaranteed left-aligned; the user owns that guarantee.)

### New Rust pieces

1. **`arrow` crate dependency** — stream `index.arrow` per contig, pulling only
   `POS/REF/ALT` in record batches (never full-materialize the index).
   *Open:* validate that `arrow-rs`'s IPC reader decompresses the polars-written
   zstd IPC — settle in the plan.
2. **`svar1_reader.rs`** — mmap `variant_idxs.npy`/`offsets.npy` via `memmap2`
   (existing dep). Each hap's per-contig slice is a contiguous binary-search
   range on its sorted global ids (contigs are contiguous in global id space, so
   `local_id = global_id − contig_start`). mmap optional field arrays
   (`dosages.npy`, …) — **carry-through deferred to §Blocked**.
3. **`process_chromosome_svar1`** (parallel to `process_chromosome`):
   - Stream index → per-contig variant records; compute `ilen`, pack keys
     (reuse `svar2-codec`); left-align/validate against `reference` when given.
   - **Carrier-count prepass**: bincount over the contig's calls to get per-
     variant carrier counts (AF is *not* reliably present in the SVAR1 index —
     only written on the sample-subset path), then `choose_representation` per
     variant.
   - Route each sample-major call → var_key sub-stream (direct; SVAR1 is already
     sample-major per-call) or scatter into the dense hap-major block. Assemble
     `SparseChunk`(s) → existing `merge_mini_sc` → existing writer.
   - Emit `meta.json`, `max_del`, positions, and the search tree via the
     existing writer machinery — output is the same on-disk SVAR2 layout.
4. **`run_svar1_conversion_pipeline`** pyfunction bridging `from_svar1` → Rust.

*Open (plan-level):* whether SVAR1's already-fully-sample-major ordering lets us
bypass the chunk-merge and stream straight into the writer, or whether we chunk
by variant range and reuse `merge_mini_sc` verbatim. Decide after reading
`merge.rs`/`writer.rs` in the plan; default to reusing `merge_mini_sc`.

### Scope guards (v1)

- **Biallelic SVAR1 only** — matches SVAR1's `geno==1` model and SVAR2's
  biallelic invariant. Reject multiallelic input with a clear error (SVAR1
  exposes `_is_biallelic`).
- `skip_out_of_scope` handled as in `from_vcf` (symbolic/breakend ALTs).
- No sample/region subsetting — subset at the SVAR1 stage. Matches SVAR2
  `from_vcf`'s minimal surface.

## BLOCKED — field carry-through (depends on specs #1 and #2)

The whole point of a *native* migration is to be lossless, so SVAR1's stored
fields (starting with `dosages.npy`, plus any INFO/FORMAT fields SVAR1 gains)
must flow into SVAR2. That requires a general SVAR2 sidecar-field mechanism that
does not exist yet:

- **Spec #1 (write INFO/FORMAT fields)** defines: the sidecar directory/SoA
  layout for arbitrary fields, per-field granularity (per-call vs per-variant —
  dosage is inherently **per-call** because it varies per sample, unlike
  mutcat's per-variant dense code), dtype handling, and the `meta.json` field
  schema.
- **Spec #2 (read/output)** defines how those sidecars surface on
  `SparseVar2` read/decode/output.

Once #1/#2 land, this spec's remaining work is a **mapping step**: read SVAR1's
`metadata.json.fields`, mmap each field array, and hand it to the sidecar-field
writer from #1 at the granularity #1 prescribes. Fill in here at that point:

- [ ] Exact mapping of SVAR1 `fields` dict → SVAR2 sidecar-field writer.
- [ ] Per-call dosage alignment in both var_key and dense sub-streams
      (one value per call / per carried bit, in sub-stream record order).
- [ ] `meta.json` field-schema propagation.
- [ ] Whether unsupported/lossy field dtypes are dropped-with-warning or error.

## Testing strategy (sketch — expand after #1/#2)

- **Round-trip parity**: on shared fixtures, `from_svar1(reference)` produces a
  store equal to `from_vcf(reference)` on the same variants (positions, keys,
  genotype membership, max_del, search tree). Byte-identical is the goal under
  matching normalization.
- **Genotype membership**: decode SVAR2 and assert per-hap carried variants
  equal SVAR1's `genos` (the fundamental correctness check).
- **Cost-model routing**: variants land in the same var_key/dense split as
  `from_vcf` for identical carrier counts.
- **no_reference vs reference**: exercise both; confirm left-alignment only when
  a reference is supplied.
- **Scope guards**: multiallelic input rejected; `skip_out_of_scope` count
  matches `from_vcf`.
- **Field parity**: deferred to post-#1/#2 (round-trip dosages through
  SVAR1→SVAR2→read).

## Documentation / housekeeping

- `skills/genoray-api/SKILL.md` — new public `SparseVar2.from_svar1` classmethod
  (mandatory per the repo public-API rule in `CLAUDE.md`).
- CHANGELOG entry (Conventional Commits `feat:`).
- Reconcile the SVAR2 roadmap docs.

## Open questions (resolve when unblocked)

- Field marshalling details (see BLOCKED section).
- Byte-for-byte equivalence of SVAR1-derived left-alignment vs `from_vcf`'s —
  confirm the same codec/normalization code is reused so results cannot drift.
- Chunk-merge bypass vs reuse (see §New Rust pieces).
- `arrow-rs` zstd-IPC read compatibility with polars-written `index.arrow`.

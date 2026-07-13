# SVAR1 → SVAR2 conversion (`SparseVar2.from_svar1`)

Status: **READY.** Both blocking specs have landed on `main`
(PR #100 write INFO/FORMAT fields, PR #101 read INFO/FORMAT fields), and PR #102
(`SparseVar2.from_pgen`) introduced the `RecordSource` extension seam this design
builds on. Supersedes
[`2026-07-11-svar1-to-svar2-conversion-design.md`](2026-07-11-svar1-to-svar2-conversion-design.md),
which was drafted before `from_pgen` and assumed a now-obsolete
"enter at the `SparseChunk` seam" path plus a new `arrow-rs` Rust dependency.

Date: 2026-07-13

## Problem / motivation

Users with existing SVAR1 (`SparseVar`) stores want to migrate to SVAR2
(`SparseVar2`) without re-reading the original VCF/BCF — which may be gone, and
which is the expensive part: VCF→SVAR2 conversion is htslib-reader-bound (≈78% of
wall time; see
[`2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md`](2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md)).
SVAR2 is 1.46–5.67× smaller on disk and read-optimized, so migration is
worthwhile on its own. A native SVAR1→SVAR2 path skips htslib entirely.

## Key facts that shape the design

### The `RecordSource` seam (the architectural insight)

`from_pgen` (PR #102) generalized the conversion pipeline around a backend-
agnostic seam. `process_chromosome` (`src/orchestrator.rs`) now dispatches on a
`SourceSpec` enum (`Vcf | Pgen`) and pulls records through a minimal trait:

```rust
// src/record_source.rs
pub struct RawRecord {
    pub pos: u32,                             // 0-based
    pub reference: Vec<u8>,
    pub alts: Vec<Vec<u8>>,                   // ALT1 = alts[0]
    pub gt: Vec<i32>,                         // len num_samples*ploidy, sample-major
                                              // ploidy-minor; 0=REF, k=ALTk, -1=missing
    pub info_raw: Vec<Option<Vec<f64>>>,      // per requested INFO FieldSpec
    pub format_raw: Vec<Option<Vec<Vec<f64>>>>, // [FORMAT field][selected sample]
}
pub trait RecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError>;
}
```

A `ChunkAssembler` turns any `RecordSource` into `DenseChunk`s, handling
left-alignment against the FASTA, `skip_out_of_scope`, and INFO/FORMAT field
staging **uniformly for every backend**. Everything downstream of the assembler
(`dense2sparse_vk` cost-routing, `svar2-codec` key packing, `merge_mini_sc`, the
writer, and `field_finalize`) is identical for VCF and PGEN today.

**Therefore `from_svar1` = add a third `SourceSpec::Svar1` variant + one
`Svar1RecordSource`, and reuse the entire downstream pipeline unchanged.** This
is exactly the shape PR #102 used to add PGEN.

This supersedes the old doc's three settled-but-now-moot pieces:
- **No `arrow-rs` crate** — the index is read in Python via polars (below).
- **No hand-assembled `SparseChunk`** — we produce `RawRecord`s and let the
  shared assembler build the dense grid, exactly like VCF/PGEN.
- **No carrier-count prepass** — `dense2sparse_vk` already derives per-variant
  carrier counts from the dense grid for cost-model routing.

### On-disk layouts

**SVAR1** (`python/genoray/_svar/`), one directory:

- `metadata.json` — `SparseVarMetadata`: `version`, `samples: list[str]`,
  `ploidy: int`, `contigs: list[str]`, `fields: dict[str, str]` (name → numpy
  dtype string, e.g. `"float32"`), `mutcat_version`, `mutcat_contigs`.
- `index.arrow` — polars-written Arrow IPC (zstd). Columns `CHROM, POS, REF,
  ALT` (comma-`Utf8`), `ILEN`, + optional attrs. Variant-major,
  contig-contiguous (each contig is one contiguous run of rows).
- `variant_idxs.npy` — `int32`, sample-major sparse: for each `(sample, ploidy)`
  haplotype, the sorted **global** non-ref variant-row ids. Presence model:
  `geno == 1` ⇒ biallelic (one ALT per variant).
- `offsets.npy` — `int64` CSR offsets over `(n_samples, ploidy)`, shared by the
  genotypes and every field array.
- Optional per-entry field arrays aligned 1:1 with `variant_idxs.npy` and
  sharing `offsets.npy`: `dosages.npy` (`float32`), `mutcat.npy` (`int16`).
- `_is_biallelic` is a `cached_property` on `SparseVar`.

**SVAR2** (`python/genoray/_svar2.py` + `src/`), per-contig native store —
reference-relative diff format (REF never stored; recovered from FASTA), four
sub-streams per contig (`var_key/{snp,indel}`, `dense/{snp,indel}`), a top-level
`meta.json`, and per-field sidecars at
`{out}/{contig}/fields/{category}/{name}/{sub_label}/values.bin` with
`sub_label ∈ {var_key_snp, var_key_indel, dense_snp, dense_indel}`. The `fields`
manifest in `meta.json` records `{name, category, dtype, default}` per field.

## Design

### Public API — thin Python shim (mirrors `from_pgen`)

```python
@classmethod
def from_svar1(
    cls,
    out: str | Path,
    source: str | Path,                 # a SVAR1 store directory
    reference: str | Path | None = None,
    *,
    no_reference: bool = False,
    skip_out_of_scope: bool = False,
    chunk_size: int | None = None,      # memory-derived default, as in from_pgen
    threads: int | None = None,
    overwrite: bool = False,
    long_allele_capacity: int = 8 * 1024 * 1024,
    signatures: bool = False,
) -> int: ...
```

- **No `ploidy` parameter** — read from SVAR1 `metadata.json`.
- **No `info_fields`/`format_fields` kwargs** — unlike `from_vcf`, there is no
  VCF to extract from; fields are whatever SVAR1 already stored (see below).
- Normalization contract is identical to `from_vcf`/`from_pgen`: exactly one of
  `reference` (validate + left-align indels against the FASTA — the canonical,
  byte-comparable path) or `no_reference=True` (trust SVAR1's normalization,
  pack ALTs as-is). `signatures=True` requires a reference.
- Returns the count of out-of-scope (symbolic/breakend) ALTs dropped (parity
  with `from_vcf`/`from_pgen`; typically 0 since SVAR1 was built from an already
  filtered VCF).

### Python responsibilities (mirrors `from_pgen`'s metadata handling)

1. Read `metadata.json` → `samples`, `ploidy`, `contigs`, `fields`.
2. Read `index.arrow` with **polars** (existing dependency — this is why no
   `arrow-rs` crate is needed and the old doc's "does arrow-rs decompress
   polars zstd-IPC" open question disappears). Derive, per contig, the
   `POS/REF/ALT` columns and each contig's contiguous global variant-id range.
3. Build the `FieldSpec` list from `metadata.fields` (see field policy).
4. Reject multiallelic input up front via `SparseVar._is_biallelic`.
5. Marshal per-contig `POS/REF/ALT` + the `.npy` file paths + `samples`,
   `ploidy`, `FieldSpec`s to a new `run_svar1_conversion_pipeline` pyfunction,
   which fans contigs out over `process_chromosome` exactly as the VCF/PGEN
   pipelines do.

   REF/ALT are marshalled as packed bytes + `i64` offset arrays (numpy), not a
   `list[str]`, so biobank-scale variant counts do not pay per-string FFI
   overhead. (Exact encoding is a plan-level detail.)

### `Svar1RecordSource` (the only substantial new Rust — `src/svar1_reader.rs`)

Implements `RecordSource`. SVAR1 is sample-major sparse but `RawRecord.gt` is
variant-major, so the source:

1. mmaps `variant_idxs.npy` (i32) + `offsets.npy` (i64) via `memmap2` (existing
   dep), plus each carried field array (`dosages.npy`, …).
2. Per contig, performs a one-pass **transpose** from the sample-major CSR into a
   variant-major CSR of carrier haplotype columns (a counting-sort scatter over
   the contig's carrier entries). Cheap and one-time, bounded by the contig's
   carrier-entry count. `local_id = global_id − contig_start` (contigs are
   contiguous in global id space).
3. Serves records in variant order: for variant `v`, `gt[h] = 1` for each
   carrier haplotype column `h` and `0` elsewhere (biallelic ⇒ single ALT,
   allele code 1; SVAR1 has no missing genotypes). `pos/reference/alts` come from
   the marshalled index columns.
4. For each carried FORMAT field, `format_raw[field][sample]` = the stored value
   at a carrier sample, else the field's default/missing sentinel (SVAR1 stores
   values only at carriers).

`SourceSpec::Svar1 { variant_idxs_path, offsets_path, field_paths, var_start,
var_end, pos, ref_bytes, ref_offsets, alt_bytes, alt_offsets, ... }` is added to
the enum in `src/orchestrator.rs`, and the reader-thread `match` gains a
`SourceSpec::Svar1 => Box::new(Svar1RecordSource::new(...))` arm. Nothing else in
`process_chromosome` changes.

### Field carry-through

SVAR1 stores all custom fields as **FORMAT Number=G, per-carrier-entry**
(`dosages.npy` is the one used in practice). Policy:

- **Auto-carry every field** in `metadata.fields` as a SVAR2 **FORMAT** field.
  `htype` is inferred from the numpy dtype string (float → `Float`, int →
  `Integer`); storage dtype defaults to `Auto` (lossless narrowing, matching
  `from_vcf`'s default). No field-selection kwarg — lossless for what SVAR1
  stores.
- **Drop `mutcat`.** It is signature machinery (a per-variant `int16` code
  stamped by `mutcat_version`/`mutcat_contigs`), not a genuine FORMAT field, and
  carrying it as one would be incorrect. Users who want signatures on the SVAR2
  store pass `signatures=True`, which **recomputes** them from the reference via
  the existing pipeline path — strictly better than carrying stale codes.
- **Known field caveat.** SVAR1 already discarded non-carrier FORMAT values, so
  for a **dense-routed** variant the non-carrier cells are filled with the
  field's default/missing sentinel. Field output is therefore byte-identical to
  `from_vcf` only for **var_key** (carrier-only) routing; for dense routing it is
  faithful to SVAR1 (carrier values preserved) but not to the original VCF. This
  is documented in the docstring and CHANGELOG. Genotype streams are unaffected
  and remain byte-identical under matching normalization.

### Scope guards (v1)

- **Biallelic SVAR1 only** — matches SVAR1's `geno==1` model and SVAR2's
  biallelic invariant. Reject multiallelic input with a clear error via
  `SparseVar._is_biallelic`.
- `skip_out_of_scope` handled by the shared `ChunkAssembler`, as in `from_vcf`.
- No sample/region subsetting — subset at the SVAR1 stage. Matches
  `from_vcf`/`from_pgen`.

## Testing strategy

Round-trip verification is now fully available because the field **read** API
(PR #101) landed — `SparseVar2.with_fields([...])` + field-aware `decode`.

- **Round-trip parity vs `from_vcf`**: build a SVAR1 store from a VCF fixture
  (`SparseVar.from_vcf`), then compare `from_svar1(reference)` against
  `from_vcf(reference)` on the same variants — positions, keys, genotype
  membership, `max_del`, and search tree byte-identical under matching
  normalization.
- **Genotype membership (fundamental check)**: decode the SVAR2 store and assert
  each haplotype's carried variants equal SVAR1's `genos`.
- **Field round-trip**: `from_svar1` a store with `dosages.npy`, open with
  `with_fields(["dosages"])`, decode, and assert carrier dosages equal SVAR1's;
  assert var_key-routed values match `from_vcf` byte-for-byte and document the
  dense non-carrier caveat with an explicit test.
- **Cost-model routing**: variants land in the same var_key/dense split as
  `from_vcf` for identical carrier counts (guaranteed by construction — same
  dense grid feeds `dense2sparse_vk` — but asserted).
- **`no_reference` vs `reference`**: exercise both; confirm left-alignment only
  when a reference is supplied.
- **Scope guards**: multiallelic input rejected with a clear error;
  `skip_out_of_scope` drop count matches `from_vcf`.

Fixtures: reuse the VCF fixtures the existing conversion tests build from (the
`vcfixture` oracle is available), plus a SVAR1 store derived from them.

## Documentation / housekeeping

- `skills/genoray-api/SKILL.md` — document the new public
  `SparseVar2.from_svar1` classmethod (mandatory per the repo public-API rule in
  `CLAUDE.md`). Note the field caveat and the "no `ploidy`/`info_fields`" surface.
- `CHANGELOG.md` — `## Unreleased` entry (Conventional Commits `feat:`), stating
  the biallelic scope guard, the auto-carry-fields / drop-mutcat policy, and the
  dense-routed non-carrier field caveat.
- Add the top-of-file "superseded by" note to
  `2026-07-11-svar1-to-svar2-conversion-design.md`.

## Resolved / non-questions (vs the old doc)

- **arrow-rs zstd-IPC compatibility** — moot; the index is read in Python.
- **SparseChunk-entry vs chunk-merge reuse** — moot; we enter at the
  `RecordSource` seam and reuse the dense→sparse path verbatim.
- **carrier-count prepass** — moot; `dense2sparse_vk` counts carriers itself.

## Open questions (plan-level)

- Exact numpy encoding for marshalling REF/ALT across the FFI (packed bytes +
  offsets vs a small Arrow C-data handoff). Settle in the plan; both avoid a
  Rust arrow dependency.
- Whether the per-contig transpose is materialized fully or streamed in
  chunk_size windows. Default to full-per-contig (bounded by carrier entries);
  revisit only if a biobank-scale store proves it a memory problem.

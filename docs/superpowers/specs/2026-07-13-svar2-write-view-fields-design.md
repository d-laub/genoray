> **SUPERSEDED** by `docs/superpowers/specs/2026-07-13-svar2-view-unify-routing-design.md`.
> This plan carried fields through the `reroute=True` **re-conversion** path, which no
> longer exists — `reroute=True` is now a routing policy inside the slicer, which
> already carried fields. Never implemented. Kept for history.

# SVAR2 `write_view` INFO/FORMAT field carry-through + signature recompute

**Status:** design (brainstorm output)
**Follows:** `docs/superpowers/specs/2026-07-12-svar2-concat-split-write-view-design.md`
(which shipped `write_view` genotypes-only, `fields`→`ValueError`, `reference`
accepted-but-unwired).
**Resolves deferred call-outs from PR #105:** INFO/FORMAT field carry-through;
`mutcat`/signatures recompute via `reference=`.

## Goal

`SparseVar2.write_view(regions, samples, output, fields=[...], reference=...)`
must carry the store's INFO/FORMAT fields through the region/sample subset
re-conversion, and recompute the mutational-signature (`mutcat`) sidecar on the
subset when a `reference=` FASTA is given. Today the Rust `run_view_pipeline`
hard-errors on any non-empty `fields`, `Svar2Source` emits empty
`info_raw`/`format_raw`, and `reference` is discarded.

## Key facts established by investigation (all verified against the code)

1. **The conversion pipeline is source-agnostic for fields.** `ChunkAssembler`
   (`src/chunk_assembler.rs`), `dense2sparse_vk` (`src/rvk.rs`), and
   `finalize_fields` (`src/field_finalize.rs`) consume only
   `RawRecord.info_raw`/`format_raw` and never inspect the source kind.
   **`svar1_reader.rs` is the living precedent** — SVAR1 `write_view` already
   carries FORMAT through a *store-backed* reader
   (`Svar1RecordSource::next_record`, `src/svar1_reader.rs:201-222`) producing
   exactly the `format_raw: Vec<Option<Vec<Vec<f64>>>>` shape (per-sample vec,
   sentinel for non-carriers). So no pipeline-core change is needed.

2. **Per-call provenance exists via `RecordSrc`.** `RecordSrc { is_dense,
   is_indel, idx }` (`src/query/gather.rs:837-842`) is the `(FieldSub, index)`
   pair needed to index a `FieldView` (`src/query/field.rs`): `idx` is the
   var_key absolute call index (`!is_dense`) or the dense variant row
   (`is_dense`). `BatchResult::decode_hap_src` (`gather.rs:850-936`) yields
   `(HapCalls, Vec<RecordSrc>)`; `gather_batch_fields`
   (`src/py_query_decode.rs:156-203`) is the exact mapping template
   (sub-stream from `(is_dense,is_indel)`; INFO `value_at(idx)`; dense FORMAT
   `idx*n_cohort + orig_sample`).

3. **Provenance is only populated on the provenance-carrying gather** (`SrcKeyRef`
   element type; `overlap_batch_src` / the readbound `_src` variants). The plain
   paths leave `vk_src`/`dense_src` empty and pay nothing.

4. **Signatures are store-based, not VCF-entangled.** `annotate_contig`
   (`src/mutcat/annotate.rs:13-54`) reads a *finished* store's positions/keys via
   `ContigReader` and needs only the contig reference sequence for tri-nucleotide
   / indel context. `process_chromosome` runs it as a write-time post-pass gated
   on `signatures && fasta_path.is_some()` (`src/orchestrator.rs:567-583`); the
   pure post-hoc entry is `SparseVar2.annotate_mutations`
   (`python/genoray/_svar2_mutcat.py`).

5. **`fasta_path` is overloaded.** In `process_chromosome` the same `fasta_path`
   drives both `validate_ref`/`left_align` (`chunk_assembler.rs:319-339`) and the
   signature post-pass. The view path MUST keep normalization's `fasta_path=None`
   (its `Svar2Source` REF bytes are synthetic, `src/svar2_source.rs:211-223`), so
   the two FASTA uses must be decoupled.

## Architecture — three change sites

### A. `Svar2Source` populates fields via a subset-aware provenance decode

**Decision (chosen):** keep decoding only the **requested sample subset** (no
whole-cohort decode). The current read-bound subset path
(`read_ranges` → `find_ranges`/`gather_ranges`) discards provenance, so this
work adds a provenance-carrying decode for that path:

- Use the `_src` read-bound gather (`gather_haps_readbound_src`,
  `src/query/gather.rs:773-775`) so the subset batch carries `vk_src`/`dense_src`.
- Add a **`decode_hap_src` equivalent for the split/read-bound result**
  (`BatchResultSplit`) — mirroring `BatchResult::decode_hap_src` — returning
  `(HapCalls, Vec<RecordSrc>)` for the subset. (Task 1 of the plan; if a ready
  method already covers this, use it.)
- Open the field sidecars once per contig with an `OpenField`-style helper
  (four `FieldView` sub-streams per `(category, name, dtype)`;
  `py_query_decode.rs:72-122`), using the **source `meta.json` concrete dtype**.
- At the existing keep-point (`svar2_source.rs:138-151`), for each kept call also
  read its field values from its `RecordSrc`:
  - **INFO** (per variant): one `value_at(idx)` per group (all sources of a
    `(pos,ilen,alt)` group are the same store variant, so the value is
    consistent). `RawRecord.info_raw[spec] = Some(vec![v_f64])`, or `None` when
    the store's element is the field's missing sentinel.
  - **FORMAT** (per output sample): `format_raw[spec][s_out] = vec![v_f64]` for a
    carrier (dense: `format_at(row, orig_sample)`; var_key: `value_at(call_idx)`),
    and the source's missing sentinel for non-carrier output samples — exactly
    `Svar1RecordSource`'s shape.
- **Unchanged:** the variant-major `BTreeMap<(pos,ilen,alt), …>` ordering, the
  `Pos`/`Record`/`Variant` keep rules, the `Record` `+1` query widening, MAC=0
  drop, and multi-region carrier OR. Values are widened to `f64` (exact for
  i32/f32-range storage).

Values flow through the untouched `f64` → narrow finalize. **No pipeline-core
change.**

### B. `run_view_pipeline` threads real field specs + a signature post-pass

- Remove the `fields.is_empty()` guard (`src/lib.rs:477-482`).
- Build `Vec<FieldSpec>` from the source store manifest instead of `Vec::new()`
  (`lib.rs:574`). Each `FieldSpec` uses the field's **already-resolved concrete
  `StorageDtype`** from `meta.json` (via `StorageDtype::from_meta_str`), NOT
  `Auto` — so `finalize_fields` takes the explicit-validate branch and cannot
  re-narrow the subset to a different on-disk dtype than the source
  (`field_finalize.rs:321-392`). `htype` is reconstructed from the dtype
  (float dtypes → `Float`; int/bool → `Int`) — it only feeds the missing
  sentinel, which the concrete-dtype pass-through renders moot.
- Pass these specs to `process_chromosome` and `finalize_fields`.
- **Signature recompute:** keep `fasta_path=None` in `process_chromosome`. When
  `reference` is provided, run `annotate_contig` as a **post-pass over the
  finished output store** for each output contig (load the contig reference array,
  open the output `ContigReader`, call `annotate_contig`) — the same store-based
  path `annotate_mutations` uses. Do not route `reference` into normalization.

### C. Python `write_view`

- Resolve requested `fields` against `available_fields` and pass
  `(name, category, dtype, default)` tuples (not bare strings) to
  `run_view_pipeline`. `fields=None` still means genotypes-only; `fields=[...]`
  selects a subset of the manifest (minus `mutcat`). Keep the existing fail-fast
  band ordering.
- When `reference` is given, drive the signature post-pass (and the existing
  `mutcat`-without-`reference` → `ValueError` guard stays).
- Update the docstring (fields now carried; `reference` now recomputes
  signatures).

## Semantics

- **INFO**: one value per variant, carried verbatim (source sentinel → `None`).
- **FORMAT**: per output sample. var_key-routed source variants stored carriers
  only → non-carriers get the missing sentinel (matches re-conversion, which
  re-emits only carrier calls). Dense-routed → all samples have a value.
- **dtype preservation**: the output field's on-disk dtype equals the source's
  (concrete dtype passed through; no per-subset re-narrowing).
- **mutcat**: never copied positionally (DBS adjacency is only valid for the full
  variant set); recomputed from `reference` on the subset, or absent.
- **Fast path** (unchanged from the parent spec): a full-region + all-sample view
  degenerating to a Component-A copy is out of scope here (still re-converts).

## Testing strategy

- **Byte-parity oracle (strong):** a full-region / all-sample `write_view` with
  `fields=<all>` reproduces the source store's field `values.bin` sidecars
  byte-for-byte (extends the existing genotype byte-parity test).
- **Subset correctness:** decode the view and the source over the same
  region+samples with `with_fields`; field values match per (variant, sample).
- **dtype preservation:** a store with a narrow field (e.g. `u16`) whose subset
  only exercises small values still writes `u16`, not `u8`.
- **var_key FORMAT carrier-only:** non-carrier output samples read back the
  missing sentinel.
- **Signature recompute parity:** `write_view(reference=…)` over a full
  region/all samples reproduces the source's `mutcat` sidecar; a subset that drops
  DBS-adjacent variants recomputes correctly (no positional copy).
- **No-reference:** `write_view(fields=[…])` without `reference` carries INFO/
  FORMAT but writes no `mutcat`; requesting `mutcat` in `fields` without
  `reference` still raises.
- Rust unit tests for the new `decode_hap_src` (split path) provenance and the
  `FieldView` reads, following `tests/test_field_provenance.rs`.

## Public API / docs

`fields=` and `reference=` on `write_view` change behavior (were errors/no-ops) →
`skills/genoray-api/SKILL.md` and `CHANGELOG.md` (`## Unreleased`) MUST be updated
in the same PR. CLI `genoray view` `--fields`/`--reference` gain real behavior.

## Non-goals / deferred

- Streaming `Svar2Source` (the eager `Vec<RawRecord>` materialization; see
  `docs/superpowers/notes/2026-07-13-svar2-eager-materialization-benchmark.md`).
  The subset-aware provenance decode added here is streaming-compatible but the
  full streaming rewrite is separate.
- `reroute=False` stays `NotImplementedError` (permanent; see the reroute
  measurement note).
- Non-scalar / String fields (SVAR2 fields are scalar-numeric + INFO Flag, as in
  `from_vcf`).

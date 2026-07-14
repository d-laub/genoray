# SVAR2 `write_view(reroute=False)` — representation-preserving array-slicer

**Status:** design (brainstorm output)
**Follows:** `2026-07-12-svar2-concat-split-write-view-design.md` (shipped
`reroute=True` re-conversion; `reroute=False`→`NotImplementedError`) and the
reroute measurement note `docs/superpowers/notes/2026-07-12-svar2-reroute-measurement.md`
(verdict: build `reroute=False`).
**Priority over** `2026-07-13-svar2-write-view-fields-design.md` (that plan — fields
via the `reroute=True` re-conversion — becomes a follow-up).

## Goal

Implement `SparseVar2.write_view(regions, samples, output, fields=…, reference=…,
reroute=False)` as a **representation-preserving array-slicer**: produce a
region/sample subset of a finished SVAR2 store by slicing its on-disk sidecars
directly — no cost model, no conversion pipeline, no per-hap `RawRecord`
materialization. This is the **low-memory** path (footprint ~O(sliced output))
and is ~optimal for somatic / all-rare stores (reroute spike: ≤0.01 % of variants
would flip). It carries INFO/FORMAT fields natively (by slicing their sidecars)
and recomputes `mutcat` from `reference` (positional copy is invalid).

`reroute=True` (re-router) stays the size-optimal **default**; `"auto"`→`True`.
`reroute=False` is the opt-in for somatic/rare data, memory-constrained runs, or
layout stability.

## Why an array-slicer (not the pipeline)

Investigation (verified against the code) established:

- **No high-level writer seam** produces finished sidecars from in-memory arrays —
  the finished layout is emitted by the *merge* stage from per-chunk temp files.
  So the slicer writes the four var_key/dense sidecars directly.
- **But every non-trivial encoding is a reusable standalone helper**, so no byte
  format is re-implemented by hand:
  - 2-bit SNP keys: `svar2_codec::pack_snp_keys` / `unpack_snp_key_at`.
  - dense hap-major bitmask: `bits::set_bit` / `copy_bits` (`hap*n_dense+col`).
  - `max_del.npy` + `dense/max_del.npy`: `max_del::write_max_del(contig_dir,
    n_samples, ploidy)` — recomputes from the *finished sliced* indel sidecars
    (pure scan; no pipeline).
  - `mutcat`: `mutcat::annotate::annotate_contig(&reader, &paths, &ref_seq)` —
    standalone, no pipeline dependency.
  - LUT: **verbatim file copy** (`indel/long_alleles.bin` +
    `long_allele_offsets.npy`) — indel keys are absolute row indices
    (`decode_key → Lookup{row}`), so a verbatim copy keeps them valid; no
    renumbering.
  - `positions.bin` (raw u32 LE `bytemuck::cast_slice`) and `offsets.npy`
    (`ndarray_npy::write_npy` of a `u64` CSR) are one-liners used throughout.
- Every slice is a **pure array gather** (details below).

## Architecture — new slicer + thin wiring

### A. Rust slicer `src/svar2_slice.rs`

Per contig, given the source store, the region set, the subset sample columns
(original indices, in output order), ploidy, overlap mode, the resolved field
manifest, and an optional reference:

1. **Select kept records** matching `Svar2Source`'s semantics exactly (so the
   `reroute=False` and `reroute=True` variant sets are identical):
   - Use the reader's extent overlap (`query::read_ranges`/`find_ranges`) as the
     candidate filter, then the per-mode POS predicate:
     `Pos`: `q_start ≤ pos < q_end`; `Record`: `q_start ≤ pos ≤ q_end` **and** the
     reader query window widened by `+1` (`qe.saturating_add(1)`); `Variant`: keep
     every extent-overlapping call (no POS filter). (`src/svar2_source.rs:46-60,
     110-144`.)
2. **var_key/{snp,indel}:** for each kept output column `c_out = s_out*ploidy+p`,
   gather its region-overlapping calls; emit new `positions` (u32), new
   `offsets.npy` (u64 CSR, len `n_subset*ploidy+1`, prefix-sum of kept-per-column
   counts), and `alleles.bin` — indel keys copied verbatim (u32), SNP codes
   unpacked → gathered → re-packed via `pack_snp_keys`. A var_key variant with 0
   subset carriers yields no calls → auto-dropped (var_key has no separate variant
   record).
3. **dense/{snp,indel}:** keep region-overlapping variant rows with ≥1 subset
   carrier (drop MAC=0 by omission); copy `positions`/`keys` (SNP re-packed);
   rebuild `genotypes.bin` at `hap_out*n_dense_kept + row_out` via
   `bits::set_bit`/`copy_bits`.
4. **fields** (`fields/{cat}/{name}/{sub}/values.bin`, element width from the
   source `meta.json` dtype): gather parallel to the kept calls/rows —
   var_key: one element per kept call (same order as positions/keys); dense INFO:
   one per kept row; dense FORMAT: `kept_rows × subset_samples`, **re-strided** to
   `n_subset` (`row*n_samples_orig + orig_sample` → new layout).
5. **LUT:** `fs::copy` the two `indel/long_alleles.*` files verbatim (MVP; keys
   stay valid). *(Compaction is a documented optional follow-up.)*
6. **Post-passes on the finished sliced dir:** `write_max_del(contig_dir,
   n_subset, ploidy)`; then, if `reference` given, open a fresh
   `ContigReader::open(out, chrom, n_subset, ploidy)` and `annotate_contig` with
   the reference contig sequence (`vcf_reader::load_contig_seq`).

### B. `run_slice_view` pyfunction (`src/lib.rs`)

`#[cfg(feature="conversion")]` (needs `load_contig_seq` / htslib for the reference;
gate consistently with the other view pyfuncs). Signature mirrors
`run_view_pipeline` but drives the slicer:
`(store_path, out_dir, contigs, samples, regions, regions_overlap,
merge_overlapping, fields:[(name,category,dtype,default)], reference=None,
overwrite=false)`. Loops contigs, calls the slicer, then writes `meta.json`.

### C. Python `write_view` (`python/genoray/_svar2.py`)

- `reroute=False` → call `_core.run_slice_view` (was `NotImplementedError`).
- `reroute="auto"`/`True` → unchanged (`run_view_pipeline`).
- Resolve `fields` against `available_fields` (exclude `mutcat`) → tuples, same as
  the `reroute=True` field plan.
- `mutcat` requested without `reference` still raises; `reference` given → drive
  the slicer's `annotate_contig` post-pass.
- meta.json for the subset is written the same way Component A does
  (`_svar2_ops`): subset `samples`, kept `contigs`, unchanged
  `ploidy`/`format_version`/`fields`. (Rust slicer writes sidecars; meta written
  Rust-side by `run_slice_view` via `meta::write_meta`, OR Python-side via
  `_svar2_ops` — pick one; spec picks **Rust `write_meta`** so the store is
  complete when the pyfunction returns, matching `run_view_pipeline`.)
- CLI `genoray view --no-reroute` now works (was the same `NotImplementedError`).

## Semantics

- **Identical variant set to `reroute=True`** for the same region/samples/overlap
  mode (shared keep predicate). Genotypes identical. Only the on-disk
  *representation* differs (preserved vs re-routed).
- **Fields**: carried verbatim (exact source dtype/bytes; no re-derivation, no
  re-narrowing).
- **mutcat**: recomputed from `reference` on the subset, or omitted.
- **MAC=0** variants dropped (both var_key auto-drop and explicit dense-row drop).
- **Fail-fast band ordering** (as `reroute=True`): all raises before any output
  dir is created.

## Testing strategy

- **Byte-parity oracle (full region + all samples):** `write_view(reroute=False)`
  over the whole store reproduces the source store's sidecars **byte-for-byte**
  (positions/alleles/offsets/genotypes/max_del/fields), except the LUT which is a
  verbatim copy (also byte-equal). This is the strongest correctness test — a
  full-coverage identity slice.
- **Equivalence to `reroute=True` genotypes:** for a real subset, decode both a
  `reroute=False` and a `reroute=True` view over the same region/samples; decoded
  genotypes + field values must match per (variant, sample) (representations
  differ, decoded content does not).
- **Field carry-through:** subset view decodes the same INFO/FORMAT values as
  `source.with_fields(...).decode(...)` over the region/samples; dtypes preserved.
- **var_key/dense representation preserved:** a variant dense in the source stays
  dense in the `reroute=False` output (assert via `svar2_variant_stats`
  `src_dense`), unlike `reroute=True` which may re-route it.
- **Signature recompute:** `reroute=False` + `reference=` reproduces a
  from-scratch `mutcat` on the subset; a subset dropping DBS-adjacent variants
  recomputes correctly.
- **Overlap-mode parity:** `pos`/`record`/`variant` select the same variants as
  `reroute=True` (shared predicate) — parametrized test.
- **LUT validity:** a store with long-allele indels, sliced, still decodes those
  indels' ALTs correctly (verbatim LUT + unchanged keys).
- Rust unit tests for the slicer per sub-stream + the field gather; Python
  integration tests in `tests/test_svar2_write_view.py`.

## Public API / docs

`reroute=False` becomes functional (was `NotImplementedError`); `write_view`
`fields=`/`reference=` gain behavior on this path. Update
`skills/genoray-api/SKILL.md`, `CHANGELOG.md` (`## Unreleased`),
`docs/roadmap/data-model.md` (M9), and the `write_view` docstring + `genoray view
--no-reroute` CLI help. Document the usage guidance from the reroute note
(recommended for somatic/rare + memory-constrained; `reroute=True` default).

## Non-goals / deferred

- LUT compaction (verbatim copy for MVP).
- `reroute=True` field carry-through (separate committed plan
  `2026-07-13-svar2-write-view-fields.md` — follow-up).
- Streaming `reroute=True` `Svar2Source` (the eager-materialization follow-up).
- Non-scalar/String fields (scalar-numeric + INFO Flag only, as `from_vcf`).

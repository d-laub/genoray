## Unreleased

### Added

- Added `SparseVar2.from_vcf_list` to build **one** SVAR2 store from many
  **single-sample** VCFs/BCFs via a native k-way merge (no `bcftools merge`,
  no intermediate multi-sample VCF). `sources` accepts an explicit
  `Sequence` of paths, a directory (all `*.vcf.gz` then all `*.bcf`), or a
  manifest file. A site present in some inputs but absent from another fills
  **hom-ref (`0`)** for the samples that lack it. `reference`/
  `no_reference` are supported with the same semantics as `from_vcf`
  (skipping left-alignment under `no_reference` means a site shared across
  files only joins into one output row if every file already represents it
  identically — same caller, or all already `bcftools norm`'d against the
  same reference). `info_fields=`/`format_fields=` are also supported:
  **INFO** merges first-carrier-wins (the earliest-in-`sources` file
  carrying a shared atom supplies the value), while **FORMAT** stays
  per-sample (each carrier keeps its own file's value; a non-carrier gets
  the field's default), matching `from_vcf`. Every input file must already
  be position-sorted per contig and use a consistent contig naming scheme
  across the cohort (both are validated up front, raising `ValueError`
  rather than silently corrupting the merge); opening very large cohorts
  (roughly N > 500) may require raising the process's open-file limit,
  which `from_vcf_list` detects and reports with the `ulimit -n` remedy.
- `SparseVar2.from_pgen` converts a PLINK2 PGEN (`.pgen`/`.pvar`/`.psam`) to an
  SVAR2 store through the same normalization, atom, and merge spine as
  `from_vcf`. Diploid-only (no `ploidy=` kwarg); dosages, INFO/FORMAT fields,
  and sample subsetting are out of scope for this entry point. Verified
  byte-for-byte equivalent to `from_vcf` on a 3202-sample/1,001,385-variant
  germline cohort (chr21, symbolics filtered so both backends see the same
  variant set); on that cohort `from_pgen` converts in **152.7s** vs
  `from_vcf`'s **547.0s** (**3.6× faster**, confirming the reader-not-the-
  bottleneck hypothesis), at the cost of **~2.1× peak RSS** (1065 MiB vs
  497 MiB) — see the benchmark notes in
  `docs/superpowers/specs/2026-07-12-pgen-to-svar2-design.md` for the full
  methodology, the `.pvar`-skip and OpenMP-oversubscription findings, and
  follow-ups.
- `SparseVar2.annotate_mutations`, `mutation_matrix`, and `assign_signatures`
  for COSMIC mutational-signature workflows (SBS96/ID83/DBS78), implemented in
  Rust with a per-record sidecar and streaming count matrix. `from_vcf` gains a
  `signatures=` flag to classify during the write (factored into the cost model).
- `SparseVar2.from_vcf` can extract scalar-numeric INFO/FORMAT fields into the
  store via new `info_fields=`/`format_fields=` kwargs and `InfoField`/
  `FormatField` config types (`genoray.InfoField`, `genoray.FormatField`).
  Scoped to `Integer`/`Float`/`Flag` header types with `Number ∈ {1, A}` (plus
  `Flag`'s `Number=0`); integer widths are losslessly auto-narrowed to the
  observed range by default, `f16` is opt-in and lossy. FORMAT storage is
  genotype-aligned — values at non-carrier genotypes are dropped, not stored
  independently. A read/decode API for these stored fields is added below
  (`SparseVar2.available_fields`/`with_fields`/`fields=`).
- **SVAR2: read INFO/FORMAT fields.** `SparseVar2(path, fields=…)` /
  `SparseVar2.with_fields(…)` / `SparseVar2.available_fields` opt a reader
  into decoding fields written by `from_vcf(info_fields=, format_fields=)`;
  fields are **not** decoded by default (each one costs extra I/O).
  `SparseVar2.decode()` now attaches one `Ragged` per selected field to its
  result, sharing the same variant-axis offsets object as `pos`/`ilen`/
  `allele` (access via `rag["KEY"]`). Values come back in the dtype they are
  stored as (no widening — an auto-narrowed field may be `int8`); missing
  entries keep the store's `default` or its reserved sentinel
  (`NaN`/`iinfo.min`/`iinfo.max`), returned as-is. Field keys are the bare
  name when unique across INFO/FORMAT, else bcftools-style `INFO/DP` /
  `FORMAT/DP` when a name is used by both categories. New `StoredField`
  dataclass (`genoray._svar2_fields`) is the manifest entry type
  `available_fields` returns.
- **Public Rust read API** for consumers that do their own channel merge
  (e.g. GenVarLoader): `query::FieldView` (mmap reader over a field's
  `values.bin`), `vk_src` provenance on `BatchResult`/`BatchResultSplit` via
  `query::gather_haps_readbound_src` / `query::overlap_batch_src`, plus
  `query::dense_abs_row`.
- SVAR2 field write internals: the finalize pass streams staged values
  through `chunks_exact` (no intermediate `Vec<f64>` materialization) and
  runs in parallel across fields/files; the var_key and pos/key merges now
  share a common offset-derivation + tile-gather implementation. Field output
  is byte-identical to before. Internal-only (no public API change) —
  finalize's own instruction count drops ~13% (callgrind Ir, small workload)
  and peak RSS on a 200-sample/50k-record conversion drops ~7% (~487 MiB →
  ~452 MiB); end-to-end wall time is unchanged since conversion remains
  reader/htslib-bound, not finalize-bound.

### BREAKING CHANGES

- `Phantom` mode `empty()` classmethods now take a uniform
  `empty(n_samples, ploidy, n_variants)` signature on both VCF and PGEN
  backends. VCF's former 4th `phasing` argument is removed; pass the effective
  ploidy (`ploidy + phasing`) instead.
- VCF chunk-size memory estimates now double when a sample subset/reorder is
  active, matching PGEN. This only affects internal chunk sizing (more
  conservative memory use), never returned data.
- VCF filtering now uses a single `genoray.Filter(record=, expr=)` value object.
  The `pl_filter=` constructor kwarg and the `(filter, pl_filter)` tuple
  getter/setter are removed. Migrate `VCF(p, filter=fn, pl_filter=expr)` to
  `VCF(p, filter=Filter(record=fn, expr=expr))`.
- `VCF._chunk_ranges_with_length` now yields `(data, end, chunk_idxs)` (a
  `uint32` variant-index array) as its third tuple element instead of an
  `n_extension_vars` count, matching `PGEN._chunk_ranges_with_length`.

### Perf

- Conversion reader staging memory no longer scales with `chunk_size`:
  presence bits are packed in word-aligned windows and each atom's per-column
  genotype vector is dropped as soon as its bits are set, bounding reader
  memory at `window * n_samples * ploidy * 4` bytes instead of
  `chunk_size * n_samples * ploidy * 4`. Output is bit-identical. Benefits
  both `from_vcf` and `from_pgen`.

### Fix

- **vcf**: apply configured `filter` on `VCF.read(..., mode=Genos*Dosages)` when no
  `.gvi` index is loaded, matching the genotype-only and dosage-only modes
  (previously the filter was silently ignored on this path).
- **svar2**: `SparseVar2.from_pgen` no longer silently corrupts every variant
  after the first monomorphic site (`.pvar` ALT `.`, which plink2 routinely
  emits for real cohorts). A null ALT was propagating as NaN into a `uintp`
  cumulative-sum array, corrupting `allele_idx_offsets` file-wide; the Rust
  `.pvar` reader also no longer treats a bare `.` ALT as a literal one-character
  allele.

## 2.15.0 (2026-06-30)

### Feat

- add --progress flag to genoray view CLI
- opt-in phase-level progress bar in write_view
- validate view source/regions-file/samples-file at parse time
- guard write_view against writing a view in place

### Fix

- **deps**: require seqpro>=0.21.1 for empty-row to_numpy; drop sentinel-based empty checks
- **svar**: emit in-bounds empty range for no-variant queries

### Perf

- fail fast in write_view; never delete output on a doomed run

## 2.14.0 (2026-06-27)

### Feat

- stream write_view region resolution + whole-contig short-circuit
- stream write_view output index via scan_ipc join+sink_ipc (no full-index collect)
- forward n_jobs/backend through SparseVar.assign_signatures
- parallelize fit_signatures over samples with joblib

### Fix

- compute genoray view default bounds lazily (no full-index materialization)

### Refactor

- derive _is_biallelic/_c_max_idxs/n_variants lazily so __init__ never materializes the full index
- lazy SparseVar index scan with collect-on-demand .index property
- default fit_signatures/assign_signatures to serial (n_jobs=1); relax parallel-equivalence tests to tolerance

## 2.13.0 (2026-06-25)

### Feat

- add --haploid flag to genoray write CLI
- add haploid OR-collapse option to SparseVar.from_pgen
- add haploid OR-collapse option to SparseVar.from_vcf

### Fix

- filtered PGEN var_idxs returns positional indices into _index (#69)

## 2.12.3 (2026-06-23)

### Fix

- adapt genoray to rust-backed seqpro Ragged
- **svar**: preserve index order and fix varID column in annotate_with_gtf

## 2.12.2 (2026-06-14)

### Fix

- remove seqpro cap

## 2.12.1 (2026-06-14)

### Fix

- bump seqpro cap

## 2.12.0 (2026-06-13)

### Feat

- **svar**: contig scope + NOT_ANNOTATED + mutcat_contigs in annotate_mutations (#62)
- **svar**: add mutcat_contigs to SparseVarMetadata
- **mutcat**: contig allowlist in classify_variants -> NOT_ANNOTATED
- **mutcat**: add NOT_ANNOTATED sentinel and bump MUTCAT_VERSION to 3
- **mutcat**: parallel numba ID-83 indel kernel + derived LUTs
- **mutcat**: vectorized DBS-78 batch classifier
- **mutcat**: vectorized SBS-96 batch classifier
- **reference**: expose cached contig_array for vectorized lookups

### Fix

- **mutcat**: report deletion REF-mismatch count against indel total, not all rows
- **utils**: treat mitochondrial aliases M/MT/chrM/chrMT as equivalent (#61)

### Refactor

- **mutcat**: tidy SBS-96 test imports and comment

### Perf

- **mutcat**: parallelize count kernel over samples
- **mutcat**: parallelize entry-code kernel over tracks
- **mutcat**: vectorize classify_variants (SNV/DBS numpy, indel kernel)

## 2.11.1 (2026-06-13)

### Fix

- **svar**: convert 1-based POS to 0-based in classify_variants (#59)
- **mutcat**: guard deletion repeat bucket against REF/reference mismatch
- **svar**: bump MUTCAT_VERSION and warn on stale persisted mutcat

## 2.11.0 (2026-06-12)

### Feat

- **signatures**: export fit_signatures and cosmic_signatures
- **svar**: add SparseVar.assign_signatures convenience
- **signatures**: add pooch-backed cosmic_signatures loader
- **signatures**: add fit_signatures DataFrame orchestration + row alignment
- **signatures**: add single-sample forward-selection refit
- **signatures**: add cosine + NNLS primitives
- **svar**: recompute mutcat on write_view via opt-in reference; never copy stale codes
- **svar**: export Reference and document mutation catalogues in SKILL.md
- **svar**: add mutation_matrix with per-allele/per-sample counting
- **svar**: add annotate_mutations writing per-entry mutcat field
- **mutcat**: add per-entry codes with DBS adjacency override
- **mutcat**: add per-variant classification dispatcher
- **mutcat**: add ID-83 indel classifier
- **mutcat**: add DBS-78 doublet classifier
- **mutcat**: add SBS-96 single-variant classifier
- **mutcat**: add SBS-96/DBS-78/ID-83 codebooks and code space
- **reference**: vendor pysam-backed Reference reader

### Fix

- **signatures**: maintain row order in cosmic_signatures join; add DBS78/ID83 loader tests
- **signatures**: add maintain_order="left" to fit_signatures join
- **signatures**: guard cosmic_signatures against codebook null rows; drop dead constant
- **svar**: exclude derived mutcat field from default write_view carry-over

## 2.10.0 (2026-06-11)

### Feat

- **svar**: from_pgen region + sample subsetting during conversion
- **svar**: from_vcf sample subsetting with MAC=0 drop during conversion
- **svar**: MAC=0 drop + AF recompute finalize for conversion subsetting
- **svar**: from_vcf region subsetting during conversion
- **svar**: add working-index build/write helpers for conversion subsetting

### Fix

- **svar**: make from_vcf contig-contiguity assert actually detect interleaving

### Refactor

- **svar**: align MAC-drop messaging with write_view, dedupe + minimal diff
- **svar**: simplify from_vcf keep_local + document contig-block invariant
- **svar**: extract _resolve_kept_rows from _resolve_kept_var_idxs

## 2.9.2 (2026-06-08)

### Fix

- **deps**: allow seqpro 0.15 (relax upper bound to <0.16)
- **types**: add pyrefly hook and resolve strict type errors

## 2.9.1 (2026-06-07)

### Fix

- **vcf**: return (filter, pl_filter) tuple from VCF.filter getter

## 2.9.0 (2026-06-05)

### Feat

- **vcf**: enforce filter/pl_filter pair invariant in VCF.filter setter
- **cli**: replace --skip-symbolic-alts with --no-symbolic and --no-breakend
- **exprs**: add record-level _record_is_symbolic/_record_is_breakend predicates
- **exprs**: add is_breakend; treat breakend ALTs as un-sizable (null ILEN)

## 2.8.0 (2026-06-05)

### Feat

- **pgen**: size symbolic SVs from PVAR INFO (SVLEN/END)
- **vcf**: size symbolic SVs via SVLEN/END; persist corrected ILEN
- **exprs**: symbolic_ilen helper + is_imprecise expression
- **cli**: --skip-symbolic-alts builds source filter for VCF and PGEN
- **vcf,svar**: add skip_symbolic_alts option to filter <DEL>/<INS>/<DUP>/...

### Fix

- **exprs**: is_snp/is_indel exclude null-ILEN symbolic SVs; test END fallback
- **svar**: coerce null ILEN to 0 in with-length read + overlap; test lazy/svar paths
- **ilen**: coerce null ILEN to 0 at numpy materialization boundaries
- **vcf**: restrict INFO header detection to INFO; strengthen ILEN alignment guard
- **vcf**: don't leak SV INFO placeholder cols; guard ILEN concat alignment
- **svar**: use compacted chunk index in from_pgen dispatch; add multi-contig + dosage filter tests
- **tests**: update fixtures.py to vcfixture 0.6.0 version= API
- **svar**: from_pgen inherits and applies the source PGEN filter
- **vcf**: apply cyvcf2 filter in chunk() so from_vcf genotypes match the filtered index
- **svar**: from_vcf inherits and applies the source VCF filter
- **vcf**: split ALT to list before applying pl_filter in _load_index

### Refactor

- **svar**: drop redundant int8 cast in _process_contig_pgen; harden pgen alignment test
- **svar**: rename _process_contig_vcf filter param to avoid shadowing builtin
- **vcf**: remove skip_symbolic_alts flag

## 2.7.3 (2026-06-01)

### Fix

- bump seqpro

## 2.7.2 (2026-06-01)

### Fix

- bump seqpro

## 2.7.1 (2026-05-31)

### Fix

- bump seqpro

## 2.7.0 (2026-05-30)

### Feat

- add private _dense2sparse_with_length bridge

### Fix

- VCF with_length hap_lens shape under phasing with indel in extension

### Refactor

- simplify _length_walk_n_keep to early-returns with type hints
- extract shared _length_walk_n_keep helper for with_length

### Perf

- numba-accelerate _dense2sparse_with_length

## 2.6.0 (2026-05-21)

### Feat

- **pkg**: expose genoray script; drop [cli] extra
- **cli**: move cli source into genoray._cli
- **svar**: drop MAC=0 variants from write_view output

### Perf

- **init**: make genoray.__init__ lazy via PEP 562 __getattr__

## 2.5.0 (2026-05-20)

### Feat

- **svar**: add SparseVar.write_view for region+sample subsetting
- **svar**: add numba kernels for write_view (count + write var_idxs + write field)
- **svar**: add _resolve_kept_var_idxs with pos/record/variant modes
- **svar**: add _normalize_samples and _validate_fields helpers
- **svar**: add _normalize_regions helper
- **utils**: add _resolve_threads and numba_threads context manager

### Fix

- **svar**: defer write_view output mkdir until after validation; verify genotype round-trip
- **svar**: _resolve_kept_var_idxs use exclusive end from var_ranges
- **svar**: support pandas/pyranges in _normalize_regions; tighten tests and types

## 2.4.0 (2026-05-20)

### Feat

- **svar**: add nbytes property covering resident index only
- **pgen**: add nbytes property summing index + StartsEndsIlens
- **vcf**: add nbytes property for resident memory size

## 2.3.3 (2026-05-13)

### Fix

- **vcf**: open subset VCF with only requested samples; fix _s_sorter for repeated set_samples calls

## 2.3.2 (2026-05-12)

### Fix

- **svar**: correct inverse permutation and length-budget early-exit in _find_starts_ends*

## 2.3.1 (2026-05-11)

### Fix

- VCF set_samples now returns genotypes in requested sample order

## 2.3.0 (2026-05-08)

### Feat

- bump seqpro to 0.11, add available_fields attr, simplify fields/attrs API
- add arbitrary field loading to SparseVar

## 2.2.3 (2026-04-22)

### Fix

- var_counts scatter non-zero counts by query index

## 2.2.2 (2026-03-31)

### Fix

- sample reordering and selection for PGEN

## 2.2.1 (2026-03-09)

### Fix

- queries with no variants

## 2.2.0 (2026-03-09)

### Feat

- **private**: unify interface to _find_starts_and_ends methods
- **perf**: index-free VCF->SVAR to reduce memory usage

## 2.1.3 (2026-02-10)

### Fix

- no variants with no VCF gvi index

## 2.1.2 (2026-02-06)

### Fix

- too many open memmaps during X->SVAR

## 2.1.1 (2026-02-05)

### Fix

- overwriting chunks when chunks smaller than full contig

## 2.1.0 (2026-02-04)

### Feat

- specify number of jobs for X->SVAR conversion

## 2.0.1 (2026-02-04)

### Fix

- **perf**: reduce memory usage for PGEN->SVAR
- VCF.chunk no longer requires an index
- **perf**: write SVAR with parallelization over contigs

## 2.0.0 (2026-01-28)

### Feat

- support python 3.13

### Fix

- use polars_config_meta to set coordinate system for polars_bio==0.20.1
- update to polars-bio 0.20.1, deprecates Python 3.9
- remove broken `out` argument from `PGEN.read`
- remove broken `out` argument from `PGEN.read`

## 1.0.1 (2025-12-20)

### Fix

- indices from var_ranges need to be converted from relative to absolute

## 1.0.0 (2025-12-20)

### Feat

- no more SparseVar.granges, switched to polars bio. faster and lower mem alg for var_ranges.

### Fix

- don't include index in svar._to_df()
- negative start queries for PGEN, SVAR (#16)

### Perf

- enable projection pushdown

## 0.17.0 (2025-12-03)

### Feat

- add CDS-based GTF annotation workflow and codon logic

### Fix

- check contig has variants for concatenate
- pyranges natsorts chroms, changing order wrt index

## 0.16.1 (2025-11-19)

### Fix

- **cli**: bump cli version
- type hints

## 0.16.0 (2025-10-20)

### Feat

- caching AFs as variant attributes.

### Fix

- bump seqpro

## 0.15.0 (2025-08-22)

### Feat

- option to write dosages in svartools
- use awkward subclass version of Ragged arrays

### Fix

- **vcf**: clip starts to >= 0 for any pyrange ops. bump seqpro dep
- pyranges can return wrong result with regions containing negative coordinates
- variant file types with preceding dots. perf: SVAR queries
- svar read_ranges_with_length will no longer include an extra variant on the right when less than the distance between that variant and the penultimate variant is sufficient to span the requested range

## 0.14.6 (2025-07-19)

### Fix

- handle contigs/chunks with no variants

## 0.14.5 (2025-07-19)

### Fix

- svar no_var offsets
- min_attrs for VCF._write_gvi_index

## 0.14.4 (2025-07-09)

### Fix

- **perf**: move svartools to standalone script

## 0.14.3 (2025-07-09)

### Fix

- move svartools to be inside package dir
- use pl.Series instead of pl.lit to define VCF filter column

## 0.14.2 (2025-06-26)

### Fix

- use last_idx in pgen chunk_with_length loop

## 0.14.1 (2025-06-11)

### Fix

- contig normalizer

## 0.14.0 (2025-06-11)

### Feat

- contig normalizer can map unnormalized contigs to indices.

## 0.13.1 (2025-06-11)

### Fix

- raise error in VCF.get_record_info if contig is unspecified but start or end is.

## 0.13.0 (2025-06-10)

### Fix

- adjust for breaking changes in seqpro Ragged API

## 0.12.2 (2025-06-05)

### Fix

- compatibility with zstd compressed PVAR
- compatibility with zstd compressed PVAR

## 0.12.1 (2025-06-04)

### Fix

- extend index suffix, not replace it
- recognize .pvar.zst files automatically

## 0.12.0 (2025-05-27)

### Feat

- change all methods to never return None, and instead return arrays with 0 variants

## 0.11.3 (2025-05-19)

### Fix

- treat missing fields as null in get_record_info

## 0.11.2 (2025-05-17)

### Fix

- PGEN variants are only guaranteed to be sorted within contigs. perf: cache SVAR bi-allelic status

## 0.11.1 (2025-05-17)

### Fix

- pgen chunking with length
- ILEN filter needs to be before column selection

## 0.11.0 (2025-05-17)

### Feat

- add exprs submodule for more convenient filtering

### Fix

- set ILEN to 0 for vars that are filtered out so they don't affect length calc
- mem per variant should be doubled when needing to sort by sample
- check samples for PGEN.set_samples
- parse PVAR "." as null values
- more logging

## 0.10.8 (2025-05-15)

### Fix

- relax pgenlib required version

## 0.10.7 (2025-05-13)

### Fix

- contig max indices for SVAR

## 0.10.6 (2025-05-13)

### Fix

- contig max indices for SVAR

### Perf

- lazily process vcf index and sink to disk

## 0.10.5 (2025-05-13)

### Fix

- keep iterating if any region has no variants

## 0.10.4 (2025-05-13)

### Fix

- yield None for each range if no variants from PGEN

## 0.10.3 (2025-05-10)

### Fix

- pgen chunk with length var_idxs for full chunk, not just last extension

## 0.10.2 (2025-05-07)

### Fix

- wrong var ranges for queries with no overlapping variants

## 0.10.1 (2025-05-06)

### Fix

- incrementing start coordinates twice for VCFs, consistent encoding of missing contig return value for SVAR
- correct number of ranges returned by chunk methods when n_variants == 0 or contig not found. raise warning for missing contigs

## 0.10.0 (2025-05-04)

### Feat

- change SVAR CCFs to dosages, do not infer germline CCFs automatically

## 0.9.0 (2025-05-01)

### Feat

- SparseCCFs and support in SVAR files

## 0.8.0 (2025-04-29)

### Feat

- index filtering for SVAR. feat: svartools CLI for writing SVAR files

### Fix

- bugfixes in handling start > contig end
- handle starts > contig ends for SVAR
- constrain python for cyvcf2 builds

## 0.7.1 (2025-04-25)

### Fix

- bump seqpro version
- only filter the pyranges, not record info

## 0.7.0 (2025-04-21)

### Fix

- rename SparseVar.samples to SparseVar.available_samples for consistency

## 0.6.0 (2025-04-21)

### Feat

- **wip**: SVAR file format prototyped and is very fast!
- **wip**: sparse variant file format

### Fix

- pass all tests

## 0.5.1 (2025-04-19)

### Fix

- handle >1d arrays for lengths_to_offsets
- str memory parsing

## 0.5.0 (2025-04-17)

### Feat

- convenience methods for automatically writing a gvl-compat index
- make with_length methods private/experimental

### Fix

- correct output index when vcf filter is applied
- type error in pgen.n_vars
- bug in computing var_idx offsets for ranges with no variants

### Perf

- faster reads by avoiding re-opening the VCF for each query

## 0.4.4 (2025-04-16)

### Fix

- with_length methods need to return where end was extended to

## 0.4.3 (2025-04-16)

### Fix

- relax set_samples type to be array-like

## 0.4.2 (2025-04-16)

### Fix

- set and test minimum dependencies

## 0.4.1 (2025-04-16)

### Fix

- relax typing-extensions version

## 0.4.0 (2025-04-15)

### Feat

- chunk_ranges_with_length and everything passes all tests

## 0.3.0 (2025-04-14)

### Feat

- multi-allelics, PGEN dosages, more precise typing and API
- improve pbar injection via context manager
- prototype for PGEN dosages
- prototype for PGEN dosages
- prototype for injecting a progress bar

### Fix

- make pbar context behavior match docstring

### Refactor

- clarify default for end/ends to be max value of np.int32

## 0.2.0 (2025-04-12)

### Feat

- change read_ranges to return offsets which are more immediately useful

## 0.1.0 (2025-04-12)

### Feat

- sketching out support for PGEN dosages
- refactor readers to be type safe. pass all tests.
- **wip**: reasonable output from PGEN in notebook
- initial PGEN support
- rename package to genoray
- rename package to genoray
- **wip**: initial prototype of VCF reader
- **wip**: VCF support

### Fix

- use future annotations for union types

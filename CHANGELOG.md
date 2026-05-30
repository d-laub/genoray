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

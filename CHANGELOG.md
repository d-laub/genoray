## 3.3.0 (2026-07-22)

### Feat

- **pgen**: tag record-stream survivors with true .pvar global row index
- **chunk**: carry per-variant global_idx through DenseChunk assembly
- **cli**: --progress/--log-level on write subcommands
- **svar2**: functional progress/log_level on write_view
- **svar2**: functional progress/log_level on from_pgen/from_vcf_list/from_svar1
- **svar2**: functional progress/log_level on from_vcf
- **logging**: drain-thread write_reporting context manager
- **logging**: rich ProgressRenderer with heartbeat fallback
- **logging**: report contig names resolved via normalization
- **logging**: per-event debug + summary info for excluded/normalized variants
- **logging**: emit contig/progress events and route milestones through tracing
- **logging**: PyEventReceiver binding with GIL-releasing recv_timeout
- **logging**: tracing ChannelLayer, scoped subscriber, GENORAY_LOG fmt fallback
- **logging**: add Event enum and buffered EventSink
- **svar2**: GENORAY_TRACE heartbeats + shard-worker CPU sampling for #135 diagnosis
- **svar2**: synthetic multi-contig VAF cohort generator for the #135 repro
- **cli**: split write into per-source subcommands; rename legacy writer to write-svar1
- **svar2**: read PGEN dosages as FORMAT fields in from_pgen
- **svar2**: add field selection to from_svar1 (default carries all)
- **svar2**: thread dosage field specs and reader pools into PGEN pipeline
- **svar2**: add DosageField spec and CLI field-string parser
- **svar2**: normalize contigs= in annotate_mutations; raise on all-miss
- **svar2**: normalize contig names in subset_contigs
- **svar2**: normalize contig names in reader query methods
- **vcf**: VcfRecordSource::with_sample_indices + resolve_sample_indices for precomputed sample lookup
- **bench**: record per-contig RSS high-water + arena-heap count
- support Python 3.14 (raise ceiling to <3.15)

### Fix

- **logging**: forward the GENORAY_LOG-resolved level to the channel gate
- **logging**: install a process-global tracing subscriber so pipeline-thread events reach the channel
- **logging**: key progress buffer per-chrom to fix concurrent-contig misattribution
- **svar2**: split livelock regression cohorts + fix unset-env skip guard
- **svar2**: resolve contig for decode_batch_fields path arg
- **svar2**: normalize contig names against reference FASTA in conversion
- **bench**: import generate_cohort by name so parallel default survives forkserver
- **bench**: register dynamically-loaded generate_cohort in sys.modules so the parallel default is picklable
- **svar2**: gate from_vcf_list malloc_trim on glibc (target_env=gnu) for musl source builds

### Refactor

- **logging**: remove dead PyEventReceiver::tx_clone and fix stale drain comment
- **logging**: drop dead level_from_str superseded by level_rank
- **logging**: route remaining pipeline milestone prints through tracing
- **logging**: demote pipeline sampler to trace target
- **logging**: route cohort-complete milestone through tracing
- **logging**: route remaining lib.rs milestone prints through tracing
- **svar2**: hoist separate dosage-source validation out of from_pgen contig loop

### Perf

- **svar2**: trim glibc arenas between contigs to stop the from_vcf_list RSS ratchet (#120)
- **bench**: parallelize from_vcf_list cohort generation across cores

## 3.2.1 (2026-07-18)

### Fix

- **svar2**: thread per-file contig membership into from_vcf_list merge

## 3.2.0 (2026-07-17)

### Feat

- **svar1**: PySvar1Reader bindings (ungated, numpy-dict contract)
- **svar1**: ungated cartesian find_ranges (Stage B) via partition_point
- **svar1**: ungated Svar1Reader (mmap variant_idxs, resident offsets)
- **svar1**: ungated var_ranges (Stage A) over search::overlap_range
- **svar2**: make chunk_size field-aware and add a max_mem ceiling
- **svar2**: add Carriers/Calls, a carrier-sparse genotype representation
- **bench**: from_vcf_list driver (time/memray) + profiler README
- **bench**: native run_vcf_list bench binary with dhat-heap feature
- **bench**: synthetic single-sample cohort generator for from_vcf_list

### Fix

- **svar1**: fail-fast on corrupt offsets + guard find_ranges range order
- **svar2**: move FORMAT resolution to un-gated record_source for query-core build
- **svar2**: correct rvk/chunk_assembler comments and close a Sparse+word_base gap
- **svar2**: collapse duplicate cols in from_vcf_list carrier merge
- **bench**: repair Task 1 review findings for from_vcf_list harness
- **bench**: position-deterministic REF/ALT so large cohorts avoid cross-file REF disagreement
- **bench**: fail loud on /usr/bin/time parse miss, clean up temp runner in run_bench

### Refactor

- **svar2**: carry format_by_carrier on DenseChunk (unused)
- **svar2**: extract the duplicated dense2sparse_vk routing pre-pass
- **svar2**: carry RawRecord genotypes as Calls instead of Vec<i32>

### Perf

- **svar2**: skip dense FORMAT staging for carrier chunks — kills the O(N^2)
- **svar2**: resolve FORMAT from format_by_carrier in rvk
- **svar2**: stop spawning one htslib thread per file per contig
- **svar2**: reserve dense field staging on the dense count, not the chunk
- **svar2**: route before densifying so sparse variants skip the grid scan
- **svar2**: carry FORMAT values per carrier instead of F x N per atom
- **svar2**: collapse duplicate carrier cols without a per-record BTreeMap
- **svar2**: build carriers from the frontier group instead of widening to N
- **bench**: capture CPU time in from_vcf_list bench for load-robust scaling fits
- **svar2**: frontier min-heap makes from_vcf_list merge selection O(log N)
- **svar2**: budget-derive from_vcf_list chunk_size so dense RAM is flat in N

## 3.1.0 (2026-07-16)

### Feat

- **cli**: route regions/samples to pgen and svar1 writes
- **svar2**: add regions and samples to from_svar1
- **svar2**: add regions and samples to from_pgen
- **svar2**: add regions to from_vcf_list
- **svar2**: shared per-record overlap filter in VCF reader
- **svar2**: add region and sample VCF conversion
- --check-ref flag on genoray write
- check_ref option for from_vcf_list (k-way merge)
- check_ref option for from_pgen and from_svar1
- check_ref option for from_vcf (single-source engine)
- add CheckRef policy and apply_check_ref helper
- **svar2**: shard PGEN conversion by variant-index range
- **svar2**: PGEN variant-index shard planner with boundary padding
- **bench**: matched PGEN generation helper (plink2)
- **bench**: scaling metrics + byte-identical oracle gate in region bench
- **svar2**: add region and sample VCF conversion

### Fix

- **svar2**: reconcile Rust test call sites after check_ref rebase
- **svar2**: reject regions_overlap='variant' with multiple regions per contig
- **svar2**: aggregate check_ref exclusions across VCF shards
- **svar2**: eliminate GIL convoy in concurrent PGEN shard reads
- **svar2**: fetch shard-boundary variant so PGEN sharding stays byte-identical
- **bench**: restore crash-durable jsonl + guard pgen filters, test oracle_hash
- **svar2**: include VCF shard context in worker errors
- **svar2**: avoid double-reserving VCF shard workers
- make long-allele positioned reads cross-platform (Windows)
- ungate field_finalize so genoray builds without conversion

### Refactor

- **svar2**: backend-agnostic region front-end; from_vcf uses OverlapMode
- gate PyValueError import behind conversion feature
- **svar2**: extract backend-agnostic shard planner

### Perf

- **svar2**: disable PGEN sub-contig sharding pending pgenlib GIL fix
- **svar2**: over-decompose VCF shards for work-stealing load balance
- **svar2**: work-stealing shard collector with reorder buffer (VCF)
- **svar2**: add region conversion benchmark harness
- **svar2**: shard VCF conversion within contigs
- **svar2**: parallelize VCF normalization batches

## 3.0.0 (2026-07-14)

### BREAKING CHANGE

- genoray.exprs.symbolic_ilen and genoray.exprs.IndexSchema
are renamed to _symbolic_ilen and _IndexSchema. They were internal
index-build helpers that were accidentally un-underscored; the public
exprs surface is now exactly the documented predicates (is_snp, is_indel,
is_biallelic, is_symbolic, is_breakend, is_imprecise) plus ILEN.
- the third tuple element is now a uint32 variant-index array
(matching PGEN), not an n_extension_vars count.
- VCF filtering uses genoray.Filter(record=, expr=). The
pl_filter= kwarg and the (filter, pl_filter) tuple API are removed.
- VCF now doubles the per-variant memory estimate when a sample
sorter is active, matching PGEN. Chunk-sizing only; no output change.
- Phantom mode empty() drops the VCF-only phasing argument;
pass effective ploidy (ploidy + phasing) instead. Uniform across VCF/PGEN.
- gather_haps_readbound(reader, region_starts, orig_samples,
vk_snp_range, vk_indel_range, dense_snp_range, dense_indel_range, ploidy) is now
gather_haps_readbound(reader, &HapRanges::new(...)). Downstream gvl updated in
lockstep (svar2-m6b-kernel).

### Feat

- **svar2**: annotate_mutations(gtf=) + SBS192/384 matrix + assign guard
- **mutcat**: GTF -> strand-class interval builder
- **mutcat**: accept optional strand intervals in annotate_mutations binding
- **mutcat**: emit SBS384 codes in streaming count
- **mutcat**: strand-aware contig annotation via interval sweep
- **mutcat**: add optional 2-bit strand.bin sidecar stream
- **mutcat**: add SNV transcriptional-strand classifier
- **mutcat**: mirror SBS384 code space + strand constants in rust
- **mutcat**: add SBS384/SBS192 code space to python codebook
- **svar2**: write_view routes both backends through the slicer; auto prefers fidelity when FORMAT is carried
- **svar2**: run_slice_view takes a routing policy and slices contigs in parallel
- **svar2**: compact the long-allele LUT when slicing a view
- **svar2**: carry INFO/FORMAT fields across representation flips
- **svar2**: Routing::Recompute -- re-run the cost model on the subset
- **svar2**: write_view(reroute=False) representation-preserving view
- **svar2**: run_slice_view pyfunction (meta + mutcat recompute)
- **svar2**: reroute=False slices INFO/FORMAT field sidecars
- **svar2**: reroute=False genotype array-slicer (var_key/dense/LUT/max_del)
- **svar2**: SparseVar2.from_svar1 Python shim + validation
- **svar2**: run_svar1_conversion_pipeline pyfunction
- **svar2**: Svar1RecordSource + SourceSpec::Svar1 wiring
- **svar2**: SVAR1 sample-major->variant-major transpose
- **svar2**: support no_reference in from_vcf_list
- **svar2**: INFO(first-carrier)/FORMAT(per-sample) fields in from_vcf_list
- **svar2**: SparseVar2.from_vcf_list public API + input resolution
- **svar2**: wire VcfList source into orchestrator + _core pyfunction
- **svar2**: VcfListRecordSource k-way merge of single-sample VCFs (genotypes)
- **svar2**: add atomize_to_vcf_biallelic reconstruction helper
- **cli**: genoray view targets SVAR2; legacy SVAR1 under view svar1
- **svar2**: SparseVar2.write_view (region/sample subset via re-conversion)
- **svar2**: run_view_pipeline pyfunction + SourceSpec::Svar2
- **svar2**: Svar2Source RecordSource re-emitting a finished contig
- **cli**: genoray concat + split for SVAR2 stores
- **svar2**: SparseVar2.concat (disjoint-contig merge)
- **svar2**: SparseVar2.split_by_contig
- **svar2**: SparseVar2.subset_contigs (metadata contig subset)
- **svar2**: SparseVar2.decode(fields=...) returns fields on shared offsets
- **svar2**: decode_batch_fields — field bytes per decoded record
- **svar2**: parse the meta.json field manifest for reads
- **svar2**: export FieldView + vk_src provenance as public Rust API
- **svar2**: thread vk_src provenance through the var_key gathers
- **svar2**: vk_src provenance packing + generic merge_by_position
- **svar2**: FieldView mmap reader for INFO/FORMAT sidecars
- **svar2**: public StorageDtype::from_meta_str
- **svar2**: field values.bin paths in layout
- **cli**: genoray write accepts .pgen; document from_pgen
- **svar2**: SparseVar2.from_pgen
- **svar2**: SourceSpec + run_pgen_conversion_pipeline
- **svar2**: PgenRecordSource backed by pgenlib read_alleles_range
- **svar2**: streaming .pvar/.pvar.zst variant-metadata reader
- **svar2**: from_vcf info_fields/format_fields kwargs
- **svar2**: record field manifest in meta.json
- **svar2**: global field finalize + lossless int auto-narrow
- **svar2**: merge dense field values into per-contig values.bin
- **svar2**: merge var_key field values into per-contig values.bin
- **svar2**: write per-chunk field files
- **svar2**: factor INFO/FORMAT field bits into cost model
- **svar2**: route INFO/FORMAT field values in dense2sparse_vk
- **svar2**: extract INFO/FORMAT fields into DenseChunk
- **svar2**: thread field manifest through FFI + process_chromosome
- **svar2**: field-spec typing core (Rust)
- **svar2**: InfoField/FormatField config + header validation
- **mutcat**: SparseVar2 annotate_mutations/mutation_matrix/assign_signatures
- **mutcat**: wire signatures=True into the SVAR2 conversion write path
- **mutcat**: PyContigReader.annotate_mutations + count_matrix
- **mutcat**: streaming per-contig count matrix with DBS pairing
- **mutcat**: SNV->SBS/DBS emission core with v1 isolation rule
- **mutcat**: cost model accounts for sidecar bits when signatures on
- **mutcat**: classify finalized contig records into the sidecar
- **mutcat**: sidecar reader/writer (u8 code + 2-bit ref)
- **mutcat**: mutcat/ sidecar path helpers
- **mutcat**: DBS78 doublet classifier (literal + revcomp fold)
- **mutcat**: ID83 indel classifier (port of _id83_kernel)
- **mutcat**: SBS96 classifier (pyrimidine-folded trinucleotide)
- **mutcat**: scaffold mutcat module + code-space constants
- **cli**: collapse svar2 write flags to --skip-symbolics-and-breakends
- **api**: drop the Reader type alias (no shared protocol)
- **svar2**: privatize raw-dict FFI methods (overlap_batch/find_ranges/gather_ranges)
- **svar2**: rename SparseVar2.samples to available_samples
- **vcf**: replace paired (filter, pl_filter) with a Filter value object
- **modes**: add phantom-mode factories for array and tuple modes
- **error**: add Input/MissingFile categories and PyErr mapping
- **build**: add opt-in abi3 cargo feature for release wheels
- **budget**: plan a processing-thread count from idle cores
- **py**: find_ranges dict exposes dense_snp_range/dense_indel_range
- **query**: gather_ranges_readbound + gather_haps_readbound + BatchResultSplit
- **query**: find_ranges emits per-class dense_snp_range/dense_indel_range
- **query**: conversion feature gates htslib; query core builds --no-default-features
- **svar2**: SparseVar2 find/gather/read_ranges with samples= and out=
- **svar2**: PyContigReader find/gather/read_ranges numpy bindings
- **svar2**: add find_ranges/gather_ranges/read_ranges query core
- **cli**: default 'write' to SVAR2, add 'write svar1' for SVAR 1.0
- **svar2**: SparseVar2.from_vcf wrapper (optional reference, auto-index, skip count)
- **svar2**: add index_vcf PyO3 helper to build a .csi index
- **svar2**: optional reference + skip/count plumbing through conversion pipeline
- **svar2**: opt-in skip + count for out-of-scope ALTs in atomize_record
- **svar2**: SparseVar2.decode Ragged record + region_counts (M6c)
- **svar2**: PyContigReader.region_counts decode-free count (M6c)
- **svar2**: PyContigReader.decode_batch flat variant materialization (M6c)
- **svar2**: SparseVar2.overlap_batch raw two-channel query (M6b)
- **svar2**: PyContigReader.overlap_batch raw two-channel numpy method
- **svar2**: raw LUT accessors for M6b two-channel exposure
- **svar2**: mix _BatchQueryMixin/_DecodeMixin into SparseVar2 for M6 fan-out
- **svar2**: stub owned py_query_batch/decode modules for M6 fan-out
- **svar2**: add SparseVar2 Python skeleton reading meta.json (M6a)
- **svar2**: expose PyContigReader over ContigReader::open (M6a)
- **svar2**: add rust-numpy dep and Rust->numpy conversion helpers (M6a)
- **query**: add batched two-channel overlap_batch spine
- **spine**: add reader-free KeyRef gather + merge algorithms
- **svar2-codec**: add snp_code_to_key uniform-key re-expansion
- **svar-2**: wire max_del post-pass into orchestrator + e2e round-trip
- **svar-2**: max_del post-pass producer (var_key per-column + dense scalar)
- **svar-2**: add max_del post-pass path helpers to layout
- **svar-2**: expose rvk::deletion_len decoder for max_del post-pass
- **svar-2**: validate REF + left-align atoms in the reader (M2b)
- **svar-2**: thread reference FASTA through reader/orchestrator/pyo3; widen reorder bound (M2b)
- **svar-2**: left_align repeat roll with equivalence proptest (M2b)
- **svar-2**: add L_MAX, ref-mismatch errors, validate_ref (M2b)
- overlap_sample unions sub-streams into per-hap QueryResult
- ContigReader opens and mmaps SVAR2 contig sidecars
- gather_run and kway_merge query primitives
- query module scaffold with mmap sidecar loaders
- layout path helpers for max_del.npy and dense/max_del.npy
- rvk query-side decode seam (decode_key, deletion_len, snp helpers)
- **svar-2**: overlap_range resolver with edge-case tests
- **svar-2**: SearchTree upper_bound
- **svar-2**: SearchTree construction and lower_bound
- **svar-2**: search.rs scaffold with block-scan primitives
- **svar-2**: write top-level meta.json after all contigs convert
- **svar-2**: add write_meta helper + FORMAT_VERSION (meta.json writer)
- **svar-2**: rename final_* sidecars to spec base names (positions/alleles/offsets/genotypes)
- **svar-2**: orchestrator drives dense merge; dense e2e round-trip
- **svar-2**: route variants dense vs var_key by cost model
- **svar-2**: rectangular dense merge (concat + bit-transpose + snp pack)
- **svar-2**: executor returns Phase1Output with dense ledgers
- **svar-2**: writer emits dense per-chunk pos/key/geno files
- **svar-2**: dense on-disk path helpers (dirs, geno, final)
- **svar-2**: dense class registry + empty dense chunk payload
- **svar-2**: LSB-first bit-slice helpers for dense matrix
- **svar-2**: BitGrid3::popcount_plane for allele-count routing
- **svar-2**: cost model for per-variant dense/sparse routing
- **svar-2**: atomize + reorder in the reader; accept un-normalized VCFs
- **svar-2**: normalize.rs — biallelic split + atomization core
- **svar-2**: bounded Result-based error path at pipeline boundaries
- **svar-2**: add StreamTag registry (streams.rs) for tag-routed sub-streams
- **svar-2**: split var_key conversion into 2-bit SNP + 32-bit indel streams
- **svar-2**: add SNP/indel variant classifier (VarKey)
- **svar-2**: add 2-bit SNP codec helpers (encode/pack/unpack)
- **svar-2**: main-thread orchestration, parallel merge, PEXT encoder, monitoring

### Fix

- **mutcat**: order-independent strand lookup for CSR var_key/snp positions
- **svar2**: validate reference up-front + harden reroute dispatch in write_view(reroute=False)
- **svar2**: apply per-element indel left-extent re-check in array-slicer
- **svar**: split comma-joined ALT in _scan_index instead of casting
- **svar2**: let from_svar1 guard tolerate variant-less declared contigs
- **svar2**: guard from_svar1 against non-contiguous/misordered SVAR1 index
- **svar2**: guard from_vcf_list against contig-naming and FD-limit pitfalls
- **svar2**: harden from_vcf_list's native merge against silent failure modes
- **svar2**: guard SparseVar2.concat against writing over a source store
- **svar2**: write_view default carries no fields (genotypes only), not all fields
- **svar2**: correct OverlapMode::Record semantics; batch the Svar2Source gather
- **svar2**: decode each hap once in decode_batch_fields + copy field bytes verbatim
- **svar2**: amortize dense provenance + pin field-byte gather (Task 8 review)
- **svar2**: harden dense_abs_row guard + cover vk_src provenance types in reachability test
- **svar2**: harden FieldView per review findings
- **svar2**: stop monomorphic .pvar rows corrupting from_pgen
- **svar2**: namespace field values by category to avoid same-name collision
- **svar2**: category-aware finalize error mapping + signed-narrow test
- **svar2**: index Number=A field buffers 0-based by source_alt_index
- **mutcat**: uppercase reference in annotate_contig so post-hoc matches write-time on soft-masked references
- **mutcat**: count an isolated SNV doublet once, not twice (v1 DBS parity)
- **error**: align exception-type mapping to the documented contract
- **ffi**: make bundle_from_dict fallible (KeyError/TypeError not panic)
- **query**: thread io::Result through sidecar loaders and LUT open
- **merge**: return ConversionError from merge/dense-merge/pack paths
- **writer**: return ConversionError::Io from writer threads
- **vcf**: surface reader errors as typed ValueError instead of WorkerPanicked
- correct Genos16 TypeGuard, PGEN log backend name, orphaned docstring
- **nrvk**: correct 31-bit capacity message; drop redundant mask
- **pgen**: guard __del__ against double-closing shared reader
- **svar**: accept bare-str sample in read_ranges_with_length
- **vcf**: apply filter on no-index Genos*Dosages read path
- **py**: declare samples on _BatchQueryMixin so the mixin type-checks
- **svar2**: guard find_ranges(out=) against dtype mismatch too
- **types**: repair the pyrefly type-check gate (path + error triage)
- **cli**: update direct write() callers to write_svar1 after write became an App
- write .svar output atomically in write_view via sibling staging dir
- write .svar output atomically in from_pgen via sibling staging dir
- write .svar output atomically in from_vcf via sibling staging dir
- write .gvi index files atomically via sibling tmp
- add atomic sibling-tmp write helpers
- **query**: make var_key and dense channels exact on spanning deletions
- **svar-2**: debug_assert half-open query precondition in overlap_range
- **svar-2**: assert dense_ledger cardinality matches num_chunks in merge
- **svar-2**: keep DecodedKey/decode_key local to test_e2e per task scope
- **svar-2**: join all pipeline threads on error path to prevent sampler/executor leak
- **rust**: make conversion crate compile and build as genoray._core
- **rvk**: remove stray closing brace in encode_alt_inline

### Refactor

- **svar2**: delete the pipeline-backed SVAR2 view path
- **svar2**: split the slicer into gather -> route -> emit
- **svar2**: extract the shared view selection predicate into svar2_view
- **svar2**: validate reference contigs up-front without holding all sequences
- **svar2**: expose resolve_scalar as pub(crate) for merge reader reuse
- **svar2**: split VcfChunkReader into RecordSource + ChunkAssembler
- **svar2**: share offset derivation + tile gather across merges
- **svar2**: type dense field merge by FieldCategory, not str prefix
- **exprs**: privatize symbolic_ilen and IndexSchema
- **utils**: table-driven np_to_pl_dtype, typed variant_file_type
- **var-ranges**: extract shared _var_end_expr, drop name shadowing
- **utils**: split _utils.py into _io/_contigs/_genos by domain
- **mutcat**: split _mutcat.py into codebook/classify/count package
- **mutcat**: relocate scalar classifier oracles to tests
- **svar2**: TypedDict the batch-query dict contracts
- **utils**: import DTYPE TypeVar from _types instead of redefining
- **mutcat**: own the Kind Literal in the codebook module
- **mutcat**: SENTINELS dict -> Sentinel IntEnum
- **svar2-codec**: name the payload bit-shift layout constants
- **rust**: add StreamTag::class() bridge to cost_model::Class
- **rust**: unify DenseMap/StreamMap behind EnumKey/EnumMap
- **query**: finish (usize,usize) -> Range<usize> in gather/oracle
- **vcf**: yield chunk_idxs from _chunk_ranges_with_length
- **vcf**: move _oxbow_reader out of the get_record_info overloads
- **vcf**: merge _ext_genos/_ext_genos_dosages into _ext_with_length
- **query**: compute _mem_per_variant via nbytes_per_variant
- **query**: share contig-normalize and empty-result helpers
- **query**: replace mode issubclass ladders with dispatch dicts
- **vcf**: extract _extract_dosage helper for the 5 dosage sites
- **vcf**: define modes via factory and unify empty() to 3-arg
- **pgen**: define phantom modes via the shared factory
- **query**: factor PresenceBitWriter::reserve_row; drop redundant path
- **query**: gather_haps_readbound takes a HapRanges params struct
- **query**: group oracle-only entry points under query::oracle
- **query**: use Range<usize> for internal half-open ranges
- **query**: DenseUnion src uses DenseClass + dense_view accessor
- **query**: extract gather_vk for the var_key snp+indel merge
- **query**: extract PresenceBitWriter for CSR presence bitmasks
- **query**: split query.rs into query/ module (pure code motion)
- **svar**: converge sample validation onto _resolve_sample_idxs
- **svar**: extract _write_from_reader from from_vcf/from_pgen
- split _svar.py into _svar/ package
- drop bare annotations shadowing SparseVar cached_property
- drop obsolete allow(dead_code) on PyContigReader::inner
- remove unused is_dtype and POLARS_V_IDX_TYPE
- remove dead src/utils.rs (never compiled, unused macros)
- **vcf_reader**: extract pack helpers, thread processing pool (still sequential)
- **vcf_reader**: Rc→Arc on PendingAtom.gt for cross-thread sharing
- **query**: re-express overlap_sample on the uniform-key spine
- **svar2-codec**: move indel decode + DEL/lookup encode lanes into the crate
- **svar2-codec**: move inline indel encoder (PEXT/SWAR) into the crate
- **svar2-codec**: move SNP 2-bit code + pack/unpack into the crate
- **svar2-codec**: move key-layout constants + BASES into the crate
- **svar-2**: clarify reorder memory-bound comment; name missing contig in FASTA error (M2b)
- **svar-2**: route pack_snp_key_file through layout::alleles; propagate meta serialize error
- **svar-2**: relocate long-allele LUT to shared per-contig indel dir
- **svar-2**: debug_assert non-empty REF/ALT; document reorder-buffer memory bound
- **svar-2**: extract BCF test harness into tests/common/mod.rs and drop obsolete panic tests
- **svar-2**: LongAlleleReader::get_allele uses pread + &self
- **svar-2**: move process_chromosome into orchestrator.rs; slim lib.rs
- **svar-2**: route sub-streams by StreamTag with byte-erased keys
- **svar-2**: de-generify merge over runtime key_bytes
- **svar-2**: centralize on-disk layout in layout.rs
- **svar-2**: extract /proc monitoring into monitor.rs
- **svar-2**: extract testable plan_thread_budget into budget.rs
- **svar-2**: delete dead commented code, stray checkpoints, and write-only num_variants
- **svar-2**: generalize tile merge over key element width

### Perf

- **svar2**: link vendored htslib against libdeflate
- **svar2**: fuse from_vcf_list merge scans, drop per-atom key clone
- **svar2**: benchmark PGEN vs VCF conversion, record results
- **svar2**: pack presence bits in windows, drop gt eagerly
- **svar2**: stream + parallelize finalize, drop f64 materialization
- **mutcat**: ~1.9x fewer instrs in SVAR2 count matrix
- **mutcat**: parallelize count matrix over sample-columns
- **contigs**: precompute name_to_index, drop O(n^2) remapper build
- **vcf_reader**: parallel presence packing over word-aligned variant blocks
- **budget**: raise htslib decode-thread cap 4→8 for idle-core workloads
- **convert**: reader-bound VCF→SVAR2 — raw GT decode + per-word bit pack
- **query**: block-copy dense-SNP presence via per-query threshold
- **bits**: byte-batch copy_bits body, was per-bit
- **query**: gather_haps_readbound asm fix — kill skip/take waste, elide bounds checks, inline the per-hap merge (byte-identical)

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

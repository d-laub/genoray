# Audit: Rust core — remaining modules + PyO3 boundary

## Summary
Overall health is good: the conversion pipeline is well-decomposed into single-purpose leaf modules (`bits`, `cost_model`, `spine`, `svar2-codec`), each with strong proptest coverage and clear docstrings. The biggest *structural* issue is duplication of the fixed-enum-map + registry machinery: `dense.rs` and `streams.rs` are near-identical (two array-backed maps `DenseMap`/`StreamMap`, two enums with hand-rolled `COUNT`/`ALL`/`index()`, two `*_REGISTRY` tables), and the domain's fundamental SNP-vs-indel axis is re-encoded across *three* separate enums (`cost_model::Class`, `dense::DenseClass`, `streams::StreamTag`). The specifically-flagged `merge.rs`↔`dense_merge.rs` pair is genuinely different (ragged parallel tile merge vs. rectangular sequential concat) and does *not* warrant unification beyond a shared temp-file-cleanup/read helper. The `budget.rs`/`cost_model.rs`/`monitor.rs` trio was checked for the suspected responsibility overlap and is in fact cleanly separated — thread planning vs. per-variant routing vs. runtime CPU sampling share nothing and should stay split. The biggest *idiom* issue is pervasive `panic!`/`.expect()`/`.unwrap()` on both I/O and user-input conditions (missing contig, missing sample, REF mismatch) in worker paths, which the orchestrator can only surface as a context-free `WorkerPanicked`. Also notable: `src/utils.rs` is fully dead (not a declared module; its two macros are never used), and the PyO3 surface has one verbatim ~40-line dict-assembly duplication plus an unwrap-heavy Python-dict parser.

## Findings

### [structure] `DenseMap`/`StreamMap` + enum/registry boilerplate duplicated wholesale
- **Location:** src/dense.rs:60-87 and src/streams.rs:51-75 (maps); src/dense.rs:7-34 + 44-57 and src/streams.rs:11-48 (enums + registries)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** `DenseMap<T>` and `StreamMap<T>` are the same generic fixed-size array-backed map (`from_fn`, `get`, `get_mut`, `iter`, `into_iter_tagged`) differing only in the key enum. `DenseClass` and `StreamTag` each hand-roll identical `const COUNT`, `const ALL`, and `index()`, and each has a parallel `*_REGISTRY: [Spec; COUNT]` with a `subdir`/`key_bytes`/post-hook shape. This is ~150 lines of copy-paste that must be kept in lockstep by hand.
- **Recommendation:** Introduce one `trait EnumKey: Copy { const COUNT: usize; const ALL: [Self; COUNT]; fn index(self)->usize; }` and a single generic `EnumMap<K: EnumKey, T>`. Keep the two enums and two registry constants, but delete the duplicated map impls and the duplicated `COUNT`/`ALL`/`index` bodies (a derive macro or a small declarative macro can generate them). Behavior-preserving.

### [structure] `src/utils.rs` is entirely dead code
- **Location:** src/utils.rs:1-19 (whole file); not declared in src/lib.rs
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** `utils.rs` is never declared with `mod utils` anywhere, so it does not compile into the crate at all, and its `ravel!`/`unravel!` macros have zero call sites (`grep` across `src/` finds only the definitions). Worse, it encodes a `(v,s,p)` flat-index convention that silently duplicates `BitGrid3`'s internal math (types.rs) — a trap for anyone who finds it and assumes it's live.
- **Recommendation:** Delete the file. If a 3D ravel helper is ever wanted, it belongs as a method on `BitGrid3`.

### [structure] `process_chromosome` is an oversized multi-responsibility function with an 11-arg signature
- **Location:** src/orchestrator.rs:43-295
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** One ~250-line function builds stream dirs, dense dirs, and the shared indel dir; creates three channels; spawns and names five OS threads; joins them in a leak-safe order; then runs two separate phase-2 merge loops (var_key + dense), writes the long-allele offsets npy, and kicks off the `max_del` post-pass. It also carries `#[allow(clippy::too_many_arguments)]` for 11 largely-primitive parameters (`chunk_size`, `ploidy`, `htslib_threads`, `long_allele_capacity`, `skip_out_of_scope`, `processing_threads`, …).
- **Recommendation:** Extract `setup_output_dirs(...)`, `spawn_pipeline(...) -> Handles`, `join_pipeline(handles) -> Result<Phase1Output>`, and `run_phase2_merges(...)`. Group the tuning primitives into a `ConversionParams` struct (also removes the `too_many_arguments` allow and de-duplicates the identical `ConversionError::Io { context: format!("create_dir_all …") }` blocks at 71-96).

### [structure] `layout.rs` "single source of truth" is bypassed and internally split
- **Location:** src/orchestrator.rs:63-68, 84-89, 271-274 (inline path building); src/layout.rs:20-96 (`ContigPaths` methods vs. free `var_key_indel_dir`/`dense_indel_dir`)
- **Severity:** med
- **Effort:** S
- **Risk:** low
- **Problem:** layout.rs's docstring claims every read/written path is constructed there, but the orchestrator hand-builds `Path::new(base_out_dir).join(chrom).join(spec.subdir)` in three places instead of routing through layout. Separately, layout itself exposes the same directory two ways — e.g. `ContigPaths::var_key_indel_dir(&self)` and the free `layout::var_key_indel_dir(contig_dir)` — because the post-pass takes a `{out}/{contig}` dir while `ContigPaths` takes `base + chrom`. The two conventions invite drift.
- **Recommendation:** Add a `ContigPaths::stream_subdir(&self, subdir: &str)` (or a `subdir_for(tag)`) and have the orchestrator call it; unify the free functions by giving `ContigPaths` a `from_contig_dir` constructor so there is one path-building type.

### [structure] Duplicated alignment-agnostic `u32`-LE file reader
- **Location:** src/max_del.rs:110-122 (`read_keys`, production); src/merge.rs:248-258 (`read_u32_bin`, test); src/dense_merge.rs:117-122 (`read_u32`, test); plus the temp-file cleanup loops at merge.rs:221-224 and dense_merge.rs:79-83
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** The `chunks_exact(4).map(u32::from_le_bytes)` decode (needed because `std::fs::read` may return an unaligned buffer that defeats `bytemuck::cast_slice`) is re-implemented three times, and the "remove per-chunk temp files" loop is duplicated between the two merge modules.
- **Recommendation:** Hoist a `layout::read_u32_le(path)` (or a small `io_util` fn) and a `layout::cleanup_chunk_files(dir, num_chunks)` and call both from production and tests. Minor, but removes the last real overlap between merge.rs and dense_merge.rs.

### [consistency] Worker paths panic on user-input and I/O instead of returning typed errors
- **Location:** src/vcf_reader.rs:162-214, 266-269, 288-301, 353; src/nrvk.rs:84, 110-113, 129; src/merge.rs:55-88, 162-212; src/writer.rs:56-75; src/dense_merge.rs:41-89
- **Severity:** high
- **Effort:** L
- **Risk:** low
- **Problem:** `error.rs` already defines a good `ConversionError`, and the orchestrator threads/joins are set up to return it — but the workers `.expect()`/`panic!` on conditions that include *user-recoverable input errors*: "Chromosome not found in VCF header", "Sample {} not found in VCF", "REF disagrees with reference FASTA", missing `.tbi/.csi`. These surface to Python only as `WorkerPanicked { thread }`, discarding the real message. I/O panics (`Failed to pwrite`, `create {}`) are similar. This is the single biggest divergence from the maintainer's "Result/`?`/typed errors over panic" principle.
- **Recommendation:** Make `read_next_chunk`/`decompose_current_record`/`merge_mini_sc`/`merge_dense_class`/writers return `Result<_, ConversionError>`, propagate with `?`, and have the reader thread return `Result` so the join surfaces the message. At minimum, promote the user-input validations (contig, sample, REF, index-missing) to a distinct `ConversionError::Input`/`InvalidData` variant now — they are not "hot loop" panics and the error.rs follow-up note does not cover them.

### [consistency] The SNP-vs-indel axis is encoded in three parallel enums
- **Location:** src/cost_model.rs:6-10 (`Class`), src/dense.rs:7-11 (`DenseClass`), src/streams.rs:11-15 (`StreamTag {VarKeySnp, VarKeyIndel}`)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** `Class::{Snp,Indel}`, `DenseClass::{Snp,Indel}`, and `StreamTag::{VarKeySnp,VarKeyIndel}` all model the same two-valued domain distinction. `DenseClass::cost_class()` bridges to `Class`, but `StreamTag` has no such bridge, so the mapping lives partly in code and partly in convention. Adding a third variant class (the roadmap's `pointer`/M11) means touching three enums plus two registries with no compiler-enforced link.
- **Recommendation:** Treat `cost_model::Class` as the canonical variant class and have `DenseClass`/`StreamTag` carry a `fn class(self) -> Class` (StreamTag is missing it). Longer term, consider a single `VariantClass` with per-representation registry rows keyed off it, so "SNP vs indel" is defined once.

### [consistency] On-disk key-layout shift constants are re-derived as bare literals across the codec
- **Location:** svar2-codec/src/lib.rs:45-47, 121, 129, 141-160, 179-205, 209-220
- **Severity:** med
- **Effort:** S
- **Risk:** low
- **Problem:** The crate is billed as the "single source of truth" for the bit layout, yet the payload offsets appear as raw magic numbers in ~6 functions: `<< 25` (snp_code_to_key, encode_snp, swar shift init), `<< 27` (ilen field), `<< 11` (`25-14`), `>> 5` (`30-25`), `25 - (i*2)` (decode). A layout change requires finding every literal by hand; the comments carry the derivation but the code does not.
- **Recommendation:** Introduce named consts — e.g. `const ALT_BASE0_SHIFT: u32 = 25; const ILEN_SHIFT: u32 = 27; const LOOKUP_FLAG_BIT: u32 = 0; const DEL_FLAG_BIT: u32 = 31;` — and express the encode/decode shifts in terms of them. Keeps the "one place" promise literal.

### [consistency] `&str` output-dir params force `Path→str→Path` round-trips
- **Location:** src/merge.rs:29 (`output_dir: &str`), src/dense_merge.rs:22 (`output_dir: &str`); call sites src/orchestrator.rs:261 and 281 (`dir.to_str().unwrap()`)
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** `merge_mini_sc`/`merge_dense_class` take `&str` and immediately `Path::new(output_dir)`, while every other path-taking fn in the codebase uses `&Path`. The orchestrator holds `PathBuf`s and must call `.to_str().unwrap()` to pass them, adding a non-UTF-8 panic path for no reason.
- **Recommendation:** Change both signatures to `output_dir: &Path`. Behavior-preserving, deletes two `.unwrap()`s.

### [consistency] Misleading capacity assert in `push_long_allele` + redundant mask
- **Location:** src/nrvk.rs:40-43, 62
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** The assert checks `self.row_index <= 0x7FFFFFFF` (2³¹−1 = 2,147,483,647) but its message reads "Exceeded 31-bit (4,294,967,295) index capacity" — 4,294,967,295 is 2³²−1 (u32::MAX), the wrong bound. The subsequent `current_index & 0x7FFFFFFF` ("masking … just in case") is dead: the assert already guarantees the high bit is clear.
- **Recommendation:** Fix the number in the message (2,147,483,647) and drop the redundant mask, returning `current_index`. Consider a `LongAlleleRow(u32)` newtype so the "31-bit row index" invariant is in the type, not a comment.

### [api-hygiene] `overlap_batch` re-implements `batch_result_to_dict` verbatim
- **Location:** src/py_query_batch.rs:23-67 vs. src/py_query_ranges.rs:28-70
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** `PyContigReader::overlap_batch` inlines the exact `BatchResult → PyDict` assembly (all 13 keys, the `dense_range` `[R,2]` reshape, the `lut_off` i64 cast) that `py_query_ranges.rs::batch_result_to_dict` already factors out — that helper's own docstring says "Identical to `py_query_batch.rs::overlap_batch`'s dict assembly". The split originates from separate M6b/M6c worktrees, but both now live in the crate, so it is pure duplication that will drift on any contract change.
- **Recommendation:** Make `batch_result_to_dict` (or a shared `pub(crate)` helper) the single assembler and have `overlap_batch` call it: `batch_result_to_dict(py, self.inner.lut_arrays(), &overlap_batch(&self.inner, &regions))`.

### [api-hygiene] `bundle_from_dict` panics on any malformed Python dict
- **Location:** src/py_query_ranges.rs:137-201
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** Every field access is `d.get_item(k).unwrap().unwrap().cast::<…>().unwrap().readonly().as_slice().unwrap()`. A Python caller passing a bundle that is missing a key, or has the wrong dtype/shape, triggers a Rust `unwrap` panic (process-level `PyErr`/abort) rather than a clean `KeyError`/`TypeError`. This is the FFI boundary — exactly where untrusted input arrives.
- **Recommendation:** Return `PyResult<RangesBundle>` and convert each failed lookup/cast into a `PyValueError`/`PyKeyError` with the offending key name. At minimum wrap the closures to map `None`/cast failure to a descriptive `PyErr`.

### [api-hygiene] `PyContigReader` query surface has overlapping, hard-to-disambiguate entry points
- **Location:** src/py_query_batch.rs:23 (`overlap_batch`), src/py_query_decode.rs:20 (`decode_batch`), :64 (`region_counts`), src/py_query_ranges.rs:208 (`read_ranges`), :222 (`find_ranges`), :235 (`gather_ranges`)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** Six public methods spread over three files, several producing the *same* contract: `read_ranges` and `overlap_batch` return the identical 13-key dict, and `read_ranges` is documented as just `find_ranges` + `gather_ranges` fused. A Python user cannot tell from the surface which to call. All are additionally marked `pub` "so the integration-test crate can call them as plain Rust" — mixing the FFI contract with a test-access hack.
- **Recommendation:** Document the intended call graph in one place (the `#[pymethods]` on the primary file) and consider making the redundant path (`overlap_batch` vs. `read_ranges`) explicitly delegate or be marked as the low-level primitive. For the test-access concern, expose a thin `pub(crate)` Rust API the tests call, keeping the `#[pymethods]` free to be scoped as the FFI contract demands.

### [api-hygiene] Obsolete `#[allow(dead_code)]` on `PyContigReader::inner`
- **Location:** src/py_query.rs:13-17
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** The comment says `inner` is "Not yet read outside this module — M6b/M6c query methods … will consume `inner`." Those methods now exist (py_query_batch/decode/ranges all read `self.inner`), so the field is live and the `#[allow(dead_code)]` is stale and misleading.
- **Recommendation:** Remove the `#[allow(dead_code)]` and the outdated comment.

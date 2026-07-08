# Audit: Rust core — query.rs, rvk.rs, search.rs, normalize.rs

## Summary
`search.rs` and `normalize.rs` are in good shape: focused, idiomatic, densely
proptested, with typed errors where they matter (`NormalizeError`). The health
problem is concentrated in **`query.rs` (1,402 lines)**, which conflates five
distinct responsibilities (sidecar mmap I/O, the `ContigReader`, the dense union,
decode, and six parallel batch-gather entry points) in one file, and carries
heavy copy-paste: the "var_key snp/indel gather" and "dense presence-bit" inner
loops are re-implemented nearly verbatim in four functions (`overlap_batch`,
`gather_ranges`, `gather_ranges_readbound`, `gather_haps_readbound`). The natural
decomposition is a `query/` module split (sidecar / reader / union / gather /
result). Secondary themes: `.expect()`-on-I/O panics inside a library (`rvk.rs`
`pack_snp_key_file`, `query.rs` npy loaders), primitive-obsession (`(bool, usize)`
dense-source pairs where a `DenseClass` enum already exists; bare `(usize,usize)`
ranges everywhere), un-idiomatic `_pub`-suffixed public oracle helpers, and a
public FFI surface with too many near-identical query variants (one of which,
`gather_ranges_readbound`, is now only a test oracle). The `unsafe get_unchecked`
cluster in `rvk.rs::dense2sparse_vk` lacks safety comments.

## Findings

### [structure] query.rs decomposition — split 1,402 lines into a `query/` module
- **Location:** src/query.rs:1-1402
- **Severity:** high
- **Effort:** M
- **Risk:** low
- **Problem:** One file owns five unrelated concerns, forcing a reader to hold the
  whole thing in their head. There is no logical grouping; helper I/O, the reader
  struct, the dense-union algorithm, the decode path, six public gather functions,
  and three result structs interleave.
- **Recommendation:** Convert to a `query/` directory with `mod.rs` re-exporting
  the public surface, split along the existing seams:
  - `query/sidecar.rs` — the mmap/npy loading layer: `mmap_file`, `as_u32`,
    `as_bytes`, `load_offsets`, `load_max_del`, `load_dense_max_del`, `open_dense`,
    and the `SubStreamView` / `DenseView` structs + their accessors (lines 24-211).
    This is pure disk-format plumbing with no query logic.
  - `query/reader.rs` — `ContigReader` (fields, `open`, `lut_arrays`) and its
    per-class overlap methods `vk_snp_overlap` / `vk_indel_overlap` /
    `dense_snp_overlap` / `dense_indel_overlap` / `vk_slice` / `dense_carried`
    (lines 213-459, 669-750).
  - `query/union.rs` — `DenseUnion` and `ContigReader::dense_union` (lines 296-421).
  - `query/decode.rs` — `Call`, `decode_keyref`, `decode_keyref_pub`,
    `decode_keyref_alt_pub`, `HapCalls`, `QueryResult` (lines 62-126, 461-475).
  - `query/gather.rs` — the batch entry points and result structs `BatchResult`,
    `BatchResultSplit`, `RangesBundle`, `overlap_sample`, `overlap_batch`,
    `find_ranges`, `gather_ranges`, `gather_ranges_readbound`,
    `gather_haps_readbound`, `read_ranges`, `decode_hap` (lines 477-1345).
  This is behavior-preserving (pure code motion + `pub(crate)`/`pub use` wiring)
  and is the prerequisite that makes the DRY finding below tractable.

### [structure] Four copy-pasted gather bodies — extract the shared inner loops
- **Location:** src/query.rs:582-623, 839-908, 986-1060, 1156-1284
- **Severity:** high
- **Effort:** M
- **Risk:** med
- **Problem:** The "dense presence bitmask" inner loop (`nbits`/`bit_base`/
  `need_bytes`/`div_ceil(8)`/`resize`/per-column `carried && qs<v_end`/`set_bit`/
  `push(bit_base+nbits)`) is duplicated in `overlap_batch` (594-620), `gather_ranges`
  (879-905), and twice more (split-dense form) in `gather_ranges_readbound` and
  `gather_haps_readbound`. The "var_key snp+indel gather then merge" block is
  likewise duplicated 3-4×. Each copy re-derives the same CSR-bit bookkeeping, so
  a bug fix or format change must be made in four places. The verbose
  "provably byte-identical" comments (1207-1229) are a direct symptom of hand-
  maintained parallel implementations.
- **Recommendation:** Extract helpers: a `PresenceBitWriter` (or free fn) owning
  the `(dense_present, dense_present_off)` pair with a `push_hap(carried_iter)`
  method, and a `gather_vk(reader, snp_range, indel_range, qs) -> Vec<KeyRef>` for
  the var_key channel. `overlap_batch`/`gather_ranges` collapse to loops over those
  helpers. Do this AFTER the module split; keep the specialized 2-way merge in
  `gather_haps_readbound` only if a benchmark still justifies it (otherwise route it
  through the same helper and drop the `merge_keys`-equivalence comment block).

### [structure] `gather_haps_readbound` is a ~215-line function doing four passes
- **Location:** src/query.rs:1086-1302
- **Severity:** med
- **Effort:** M
- **Risk:** med
- **Problem:** Single function: validates 5 length invariants, builds per-query
  dense-snp windows, dense-indel windows, then a per-hap loop that itself does
  var_key snp gather, indel gather, a hand-inlined merge, snp presence (block-copy
  path), and indel presence. Too much to hold at once; the `#[allow(clippy::
  needless_range_loop, clippy::too_many_arguments)]` at 1085 flags it.
- **Recommendation:** Once the shared helpers above exist, this reduces to
  window-build + a per-hap loop calling them. Also fold its 8 positional args into
  a borrowed params struct (see api-hygiene finding on the arg list).

### [consistency] `.expect()`/`panic!` on I/O in library code (npy + pack)
- **Location:** src/query.rs:170, 183, 190 (npy loaders), 264-268; src/rvk.rs:23-56 (`pack_snp_key_file`)
- **Severity:** high
- **Effort:** M
- **Risk:** low
- **Problem:** `ContigReader::open` returns `io::Result`, but its helpers
  `load_offsets`/`load_max_del`/`load_dense_max_del` call
  `ndarray_npy::read_npy(...).expect(...)` — a corrupt/truncated sidecar aborts the
  process instead of surfacing an error to Python. `rvk::pack_snp_key_file` returns
  `()` and `expect()`s on every `File::open`/`read`/`write_all`/`rename`. Per the
  maintainer's "Result/`?` over panics in library code" principle, disk-format
  faults are recoverable inputs, not invariants.
- **Recommendation:** Thread `io::Result` (or a typed `QueryError`) through the npy
  loaders and `open`; make `pack_snp_key_file -> io::Result<()>` and `?` its I/O.
  Reserve `expect` for true invariants (e.g. the LUT-present decode path).

### [consistency] `unsafe get_unchecked` cluster without SAFETY comments
- **Location:** src/rvk.rs:133-136, 150, 187, 191, 193, 207
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** `dense2sparse_vk` uses ~8 `unsafe { *chunk.*.get_unchecked(..) }` /
  `words.get_unchecked(flat_idx >> 6)` with no `// SAFETY:` justifying the bound
  (unlike `mmap_file`'s exemplary SAFETY note at query.rs:33). The indices derive
  from `v < v_variants` and `alt_offsets`, which are plausibly in-bounds, but that
  contract is undocumented and unchecked.
- **Recommendation:** Either add one `// SAFETY:` block per elision citing the loop
  bound / offset invariant, or (preferred for a hot-but-not-critical pre-pass)
  drop to safe indexing and let the maintainer measure whether the bounds checks
  matter — "measure, don't guess." At minimum document `flat_idx >> 6` fits `words`.

### [consistency] Primitive-obsession: `(bool, usize)` dense source instead of `DenseClass`
- **Location:** src/query.rs:301-307 (`src: Vec<(bool, usize)>`), 441-452, 602-615, 887-900
- **Severity:** med
- **Effort:** S
- **Risk:** low
- **Problem:** `DenseUnion.src` encodes "is this an indel?" as a bare `bool`, then
  every consumer branches `if is_indel { dense_indel } else { dense_snp }`. A
  `DenseClass { Snp, Indel }` enum already exists in `dense.rs` (used by rvk.rs).
  The `bool` is stringly-typed-adjacent and the `if is_indel` dispatch is repeated
  in three places.
- **Recommendation:** Store `Vec<(DenseClass, usize)>` and add
  `ContigReader::dense_view(class) -> Option<&DenseView>` so the three dispatch
  sites become one `reader.dense_view(class).expect(..).carried(hap, col)`. Makes
  the invalid "is_indel with no indel table" state unrepresentable at the call site.

### [consistency] Bare `(usize, usize)` half-open ranges everywhere
- **Location:** src/query.rs — `column` (141), all `*_overlap` returns (673-749), `dense_range`/`vk_snp_range`/`vk_indel_range` fields (653-666), `RangesBundle`
- **Severity:** low
- **Effort:** M
- **Risk:** low
- **Problem:** `(usize, usize)` is used pervasively for `[start, end)` slices, with
  the half-open convention living only in comments. Nothing prevents swapping the
  pair or passing a `(len, offset)` by mistake, and `.0`/`.1` accessors read poorly.
- **Recommendation:** Use `std::ops::Range<usize>` (enables `&buf[r.clone()]`,
  `r.len()`, iteration) or a small `CallRange` newtype. Lower priority than the
  structural work; fold in during the module split.

### [consistency] Repeated `let hap = col; // == flat column` aliasing
- **Location:** src/query.rs:494, 587, 846, 994, 1171 (and the `// sample-major hap index == flat column` comment ×4)
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** Every gather loop rebinds `let hap = col;` with an identical comment,
  a no-op alias that adds noise and invites the reader to wonder when they might
  differ (they never do).
- **Recommendation:** Use `col` directly (it already means the sample-major hap
  index), or introduce a `HapCol` newtype once and drop the per-loop comment.

### [consistency] `*dense_present_off.last().unwrap()` idiom repeated across gathers
- **Location:** src/query.rs:596, 881, 1026, 1043, 1234, 1261
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** Reading "current bit base" as `*off.last().unwrap()` and then
  `off.push(base + nbits)` is the CSR-append pattern hand-rolled six times.
- **Recommendation:** Subsumed by the `PresenceBitWriter` extraction (structure
  finding) — the writer owns `last`/`push` internally.

### [consistency] Test-only `TREE_BUILDS` counter compiled into production
- **Location:** src/search.rs:47-57, 76
- **Severity:** low
- **Effort:** S
- **Risk:** low
- **Problem:** Every `SearchTree::new` bumps a `thread_local` `Cell` whose only
  consumer is `search_tree_build_count()`, a `#[doc(hidden)]` observability hook for
  the search/gather-split tests. It ships (non-gated) in release builds.
- **Recommendation:** Acceptable as-is (a `Cell` increment is ~free), but consider
  `#[cfg(any(test, feature = "instrument"))]`-gating both the counter and the
  increment so the hot path and public surface stay clean in release.

### [api-hygiene] Too many near-identical public query entry points; one is dead-in-prod
- **Location:** src/query.rs:481 (`overlap_sample`), 565 (`overlap_batch`), 756 (`find_ranges`), 821 (`gather_ranges`), 929 (`gather_ranges_readbound`), 1086 (`gather_haps_readbound`), 1307 (`read_ranges`)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** Seven `pub` query functions with heavily overlapping semantics.
  Cross-referencing usage: the Python FFI (`py_query_*.rs`) uses `overlap_batch`,
  `find_ranges`, `gather_ranges`, `read_ranges`; `gather_haps_readbound` is the live
  read-bound path; but **`gather_ranges_readbound` is referenced only by
  `tests/test_readbound_gather.rs`** — it is now purely a parity oracle, yet ships
  as top-level public API indistinguishable from production entry points.
  `overlap_sample` is likewise test/oracle-only in-tree. Nothing in the names or
  docs marks oracle vs production.
- **Recommendation:** Demote the oracle-only functions (`gather_ranges_readbound`,
  and `overlap_sample` if gvl doesn't call it) to `#[doc(hidden)]` or a clearly
  named `oracle`/`testing` submodule, and document the intended production entry
  points in `query/mod.rs`. Confirm downstream (gvl) call sites before narrowing
  visibility.

### [api-hygiene] `_pub`-suffixed oracle helpers are un-idiomatic public names
- **Location:** src/query.rs:112 (`decode_keyref_pub`), 123 (`decode_keyref_alt_pub`); re-exported via lib.rs alongside `bits_get_bit`
- **Severity:** med
- **Effort:** S
- **Risk:** med
- **Problem:** `_pub` is not a Rust naming convention (`pub` already conveys
  visibility); the suffix exists only to distinguish these gvl-side parity-oracle
  wrappers from the private `decode_keyref`. They pollute the public surface with
  test-scaffolding names. `bits_get_bit` (a thin `bits::get_bit` re-export) is the
  same smell.
- **Recommendation:** Group them under a `pub mod oracle` (or `testing`) so they
  read as `query::oracle::decode_keyref` / `::decode_keyref_alt` without the
  suffix. Risk is med because these are named in gvl test oracles — coordinate the
  rename with the downstream (a re-export shim can bridge one release).

### [api-hygiene] `gather_haps_readbound` takes 8 positional args (4 parallel range slices)
- **Location:** src/query.rs:1085-1095
- **Severity:** med
- **Effort:** S
- **Risk:** low
- **Problem:** `#[allow(clippy::too_many_arguments)]` over a signature taking
  `region_starts`, `orig_samples`, and four separate `&[(usize,usize)]` range
  arrays plus `ploidy`. The five `assert_eq!` length checks at 1097-1101 exist
  precisely because the parallel-slice contract is easy to violate at the call site.
- **Recommendation:** Introduce a borrowed `HapRanges<'_>` params struct bundling
  the six slices + ploidy (mirroring `RangesBundle` but per-query), so the invariant
  is expressed once in the type and the asserts become construction-time. Removes
  the clippy allow.

### [api-hygiene] `BatchResult` vs `BatchResultSplit` — overlapping result types
- **Location:** src/query.rs:516-560
- **Severity:** low
- **Effort:** M
- **Risk:** med
- **Problem:** Two public result structs share the `n_regions/n_samples/ploidy/vk/
  vk_off` prefix and diverge only in unified-dense (`dense`, `dense_range`,
  `dense_present*`) vs split-dense (`dense_snp*`, `dense_indel*`) tails. Consumers
  must know which producer yields which. The duplication of the common header is a
  mild DRY/YAGNI concern.
- **Recommendation:** Low priority (both feed distinct FFI dict contracts). If
  touched, factor the common header into a shared struct or generic dense payload;
  otherwise leave and just co-document the pairing in `query/mod.rs`.

### [api-hygiene] `KeyRef` re-export comment signals a fragile public coupling
- **Location:** src/query.rs:19-22
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** `pub use crate::spine::KeyRef` is re-exported (not just imported) so
  downstream can name `query::KeyRef` to match `BatchResultSplit.vk: Vec<KeyRef>`.
  The public API leaks a `spine` type through `query`, and the coupling is
  documented in a comment rather than enforced.
- **Recommendation:** Fine to keep, but during the module split decide the canonical
  home for `KeyRef` (likely a shared `types`/`spine` public module) and re-export
  consistently so it isn't surfaced from two paths.

# SP-2 — `query.rs` → `query/` module + gather DRY

**Date:** 2026-07-08
**Branch:** `sp2-query-module`
**Roadmap:** [`docs/roadmap/clean-code-audit.md`](../../roadmap/clean-code-audit.md) — sub-project #2
**Findings:** [`docs/roadmap/audit-findings/05-rust-query.md`](../../roadmap/audit-findings/05-rust-query.md)
**Size:** M–L · **Risk:** med

## Goal

`src/query.rs` is 1,402 lines conflating five unrelated responsibilities — sidecar mmap/npy
I/O, the `ContigReader`, the dense union, decode, and six parallel batch-gather entry points
— in one file, and carries heavy copy-paste: the "dense presence-bit" inner loop and the
"var_key snp+indel gather then merge" block are each re-implemented nearly verbatim in four
gather functions. A reader must hold the whole file in their head; a format change or bug fix
must be made in four places.

SP-2 splits it into a `src/query/` module **along the existing seams** (`mod` /
`sidecar` / `reader` / `union` / `decode` / `gather` / `oracle`), extracts the duplicated
inner loops behind a `PresenceBitWriter` + `gather_vk` helper, applies the two
type-safety cleanups (`DenseClass` dense source, `Range<usize>` half-open ranges), and
demotes the oracle-only entry points into a `query::oracle` submodule.

**Core promise: behavior-preserving.** Every observable result is identical, gated on the
Rust integration suite (oracle-parity + byte-identical asserts) and the Python FFI suite
staying green. There is exactly **one** deliberate public-surface change — folding
`gather_haps_readbound`'s 8 positional args into a `HapRanges<'_>` params struct — and it
is landed **in lockstep** with a matching update to the downstream GenVarLoader
`svar2-m6b-kernel` branch (see [Downstream coordination](#downstream-coordination)).

**No `skills/genoray-api/SKILL.md` change.** Every symbol touched is a `genoray_core` Rust
name, not reachable via `import genoray`. The Python public API is untouched.

## Downstream coordination

The `genoray_core` crate has one downstream consumer beyond this repo's own tests: the
GenVarLoader **`svar2-m6b-kernel`** branch (unmerged). Its imports were traced directly
from that branch's tree; the full coupling is:

| `genoray_core` symbol | Downstream site | Kind |
|---|---|---|
| `query::ContigReader` (+ `::open`) | `src/svar2/store.rs` | **production** |
| `query::gather_haps_readbound` | `src/ffi/mod.rs` (×4 call sites, 8 positional args) | **production** |
| `query::BatchResultSplit` (+ fields `vk_pos`/`vk_key`/`vk_off`/`dense_snp*`/`dense_indel*`/`vk`/`dense_snp`/`dense_indel`) | `src/svar2/mod.rs::split_to_flat` | **production** |
| `bits_get_bit` (crate root) | `src/svar2/mod.rs::split_to_flat` | **production** |
| `query::KeyRef` | `src/svar2/mod.rs` (`#[cfg(test)]` mod) | test-only, but a field type of the production `BatchResultSplit` |

This **corrects two assumptions in `05-rust-query.md`**: `bits_get_bit` and
`BatchResultSplit`/`KeyRef` are *production* downstream, not test scaffolding — so they are
**not** demotable to an oracle submodule. Their paths (`genoray_core::bits_get_bit`,
`genoray_core::query::BatchResultSplit`, `genoray_core::query::KeyRef`,
`genoray_core::query::ContigReader`, `genoray_core::query::gather_haps_readbound`) are
**stability guarantees** for SP-2.

Conversely, the four truly oracle-only symbols — `overlap_sample`, `gather_ranges_readbound`,
`decode_keyref_pub`, `decode_keyref_alt_pub` — are referenced **only** by this repo's own
`tests/*.rs`. They can be relocated into `query::oracle` and de-`_pub`-suffixed with no
cross-repo impact.

The single coordinated change (`gather_haps_readbound` → `HapRanges<'_>`) lands as a
**paired PR in the GenVarLoader repo** against `svar2-m6b-kernel`, updating its 4
`ffi/mod.rs` call sites (and any `(usize,usize)`→`Range` marshalling) to match. That branch
is rebuilt and its Rust + Python tests run green before SP-2 is considered done.

## Non-goals (deferred, to avoid scope creep)

These `05-rust-query.md` findings are **out of scope** and stay with their assigned
sub-projects or are explicitly left alone:

- `.expect()`/`panic!`-on-I/O in the npy loaders (`load_offsets`/`load_max_del`/
  `load_dense_max_del`, `ContigReader::open`) and `rvk.rs::pack_snp_key_file` →
  **SP-3** (Rust panics → typed errors). SP-2 relocates these functions unchanged; it does
  **not** re-thread their error handling.
- `BatchResult` vs `BatchResultSplit` header dedup — **left as-is** (they feed two distinct
  FFI dict contracts); SP-2 only co-documents the pairing in `query/mod.rs`.
- `KeyRef` canonical-home decision — **left as-is**; SP-2 keeps `pub use crate::spine::KeyRef`
  re-exported from `query` (the downstream import path) and does not move `spine::KeyRef`.
- `TREE_BUILDS` counter `cfg`-gating in `search.rs` — the finding rates it "acceptable
  as-is"; not touched.

## Target module layout

`src/query.rs` (module) → `src/query/` (package). Mapping of current top-level symbols to
new files (all visibility preserved except the oracle relocation and the `HapRanges`
addition). Line numbers are current-tree references for orientation, **not** contracts —
relocate symbols by name at implementation time.

| New file | Contents (current line refs) |
|---|---|
| `query/mod.rs` | `pub use` surface + `pub mod` wiring; a doc-comment table marking **production** vs **oracle** entry points and co-documenting the `BatchResult`/`BatchResultSplit` pairing; keeps `pub use crate::spine::KeyRef` |
| `query/sidecar.rs` | `mmap_file`, `as_u32`, `as_bytes`, `load_offsets`, `load_max_del`, `load_dense_max_del`, `open_dense`, `SubStreamView` (+ `positions`/`column`), `DenseView` (+ `positions`/`carried`) (24–211) |
| `query/reader.rs` | `ContigReader` (fields, `open`, `lut_arrays`) + `vk_slice`, `dense_carried`, `vk_snp_overlap`, `vk_indel_overlap`, `dense_snp_overlap`, `dense_indel_overlap`, and the new `dense_view(class)` accessor (213–459, 669–750) |
| `query/union.rs` | `DenseUnion` (+ `overlap`) + `ContigReader::dense_union` (296–421) |
| `query/decode.rs` | `Call`, `decode_keyref`, `HapCalls`, `QueryResult` (62–126, 461–475) |
| `query/gather.rs` | `BatchResult`, `BatchResultSplit`, `RangesBundle`, `HapRanges` (new), `overlap_batch`, `find_ranges`, `gather_ranges`, `gather_haps_readbound`, `read_ranges`, `BatchResult::decode_hap`, and the new shared helpers `PresenceBitWriter` + `gather_vk` (477–1345) |
| `query/oracle.rs` | `pub mod oracle`: `overlap_sample`, `gather_ranges_readbound`, `decode_keyref` (was `decode_keyref_pub`), `decode_keyref_alt` (was `decode_keyref_alt_pub`) |

Shared helpers (`PresenceBitWriter`, `gather_vk`) are `pub(crate)` in `query/gather.rs` so
both the production gathers and `oracle::gather_ranges_readbound` route through them.

## Work items

### A. Module split (pure code motion + re-export)

Convert `query.rs` → `query/` per the table above. `mod.rs` re-exports the public surface so
external paths (`genoray_core::query::*`) are unchanged for the stability-guaranteed symbols.
Wire cross-file visibility with `pub(crate)` where a symbol is used across the new files but
not part of the public surface. This item is **zero logic change** and should compile with
the entire existing test suite green before any later item begins.

### B. DRY extraction (after A)

- **`PresenceBitWriter`** — owns the `(present: Vec<u8>, present_off: Vec<u64>)` CSR pair;
  a `push_hap(carried_iter, nbits)` method absorbs the `div_ceil(8)`/`resize`/`set_bit`/
  `*off.last().unwrap()`+`push(base+nbits)` bookkeeping duplicated 6× (query.rs:596, 881,
  1026, 1043, 1234, 1261).
- **`gather_vk(reader, snp_range, indel_range, qs) -> Vec<KeyRef>`** — the var_key snp+indel
  gather-then-merge block duplicated 3–4× (582–623, 839–908, 986–1060, 1156–1284).
- `overlap_batch` / `gather_ranges` / `oracle::gather_ranges_readbound` collapse to loops
  over these two helpers.
- **`gather_haps_readbound`'s hand-inlined 2-way merge** (the "provably byte-identical"
  comment block, 1207–1229): `PresenceBitWriter` is extracted unconditionally (pure
  bookkeeping, the bulk of the duplication). Routing the *merge* through `gather_vk` is
  gated on a micro-benchmark — **measure, don't guess**: if unifying shows no regression on
  a representative read-bound gather, unify and delete the equivalence-comment prose;
  otherwise keep the tuned merge and drop only the now-redundant prose. Either way the
  `test_gather_haps_readbound_byte_identical` test is the correctness net.

### C. Type-safety cleanups

- `DenseUnion.src: Vec<(bool, usize)>` → `Vec<(DenseClass, usize)>` (enum already in
  `dense.rs`); add `ContigReader::dense_view(class: DenseClass) -> Option<&DenseView>` and
  collapse the three `if is_indel { dense_indel } else { dense_snp }` dispatch sites
  (441–452, 602–615, 887–900) to one call. Makes "is_indel with no indel table"
  unrepresentable at the call site.
- Bare `(usize, usize)` half-open ranges → `std::ops::Range<usize>` for the internal range
  types: `SubStreamView::column`, the four `*_overlap` returns, `DenseUnion::overlap`, and
  the `RangesBundle`/`HapRanges` range fields. Enables `&buf[r.clone()]`, `r.len()`,
  iteration and removes the swap-the-pair footgun. (The FFI marshalling in
  `py_query_ranges.rs` and the coordinated gvl `ffi/mod.rs` map their numpy `usize` pairs
  into `Range` at construction.)
- Delete the `let hap = col;` alias and its repeated comment (494, 587, 846, 994, 1171);
  use `col` directly.

### D. API-hygiene

- `query::oracle` submodule for `overlap_sample`, `gather_ranges_readbound`,
  `decode_keyref` (de-`_pub`), `decode_keyref_alt` (de-`_pub`). Update this repo's
  `tests/*.rs` imports (`test_query.rs`, `test_batch.rs`, `test_decode_mat.rs`,
  `test_readbound_gather.rs`) to the new `query::oracle::*` paths. No gvl impact.
- **`gather_haps_readbound` → borrowed `HapRanges<'_>` params struct** bundling
  `region_starts`, `orig_samples`, `vk_snp_range`, `vk_indel_range`, `dense_snp_range`,
  `dense_indel_range`, and `ploidy`. The five `assert_eq!` length checks (1097–1101) become
  construction-time validation in `HapRanges::new(...) -> HapRanges` (fail-fast on
  mismatched slice lengths). Removes the `#[allow(clippy::too_many_arguments)]`. Landed with
  the paired gvl PR.
- **Optional (fold in unless you'd rather defer):** add `// SAFETY:` comments to the
  `rvk.rs::dense2sparse_vk` `get_unchecked` cluster (133–136, 150, 187, 191, 193, 207),
  each citing the loop bound / `alt_offsets` invariant (mirroring `mmap_file`'s exemplary
  note). S effort, zero risk; same findings file.

## Verification

- **Rust:** `pixi run bash -lc 'cargo test --no-default-features …'` (the `--no-default-features`
  flag is required — otherwise the pyo3 test binary fails to link with
  `undefined symbol: _Py_Dealloc`). The integration suite is the safety net and exercises
  every gather path with oracle-parity / byte-identical asserts: `test_query.rs`,
  `test_batch.rs`, `test_decode_mat.rs`, `test_readbound_gather.rs` (incl.
  `test_gather_haps_readbound_byte_identical`), `test_ranges_split.rs`, `test_batch_raw.rs`,
  `test_e2e.rs`, `test_query_only_build.rs`.
- **Python FFI:** `pixi run test`.
- **Downstream:** rebuild GenVarLoader `svar2-m6b-kernel` against the SP-2 branch after the
  coordinated `HapRanges` + range-marshal update; run its Rust + Python tests green.
- **Micro-bench (item B gate):** a representative read-bound gather timed before/after any
  merge unification; regression → keep the tuned merge.

## PR structure

One genoray branch `sp2-query-module`, commits ordered to keep review tractable:

1. **Module split** (item A) — pure code motion + re-export shim; suite green.
2. **DRY extraction** (item B) — `PresenceBitWriter` + `gather_vk`; guarded by byte-identical
   tests + the micro-bench.
3. **Type cleanups** (item C) — `DenseClass`, `Range<usize>`, alias removal.
4. **API demotion + `HapRanges`** (item D).

Ships as a single genoray PR if the diff stays reviewable; split at the A↔B seam (mechanical
vs semantic) only if it grows unwieldy. The gvl call-site update is necessarily a **separate
PR in the GenVarLoader repo** against `svar2-m6b-kernel`, referencing this one, merged in
lockstep.

## Invariants (from the roadmap)

1. **Behavior-preserving**, gated on the existing test suites staying green — except the one
   scoped `gather_haps_readbound` signature change, which is source-compatible-by-coordination
   (all known call sites updated in lockstep).
2. **No `import genoray` public-API change** → no `skills/genoray-api/SKILL.md` update.
3. **Small, reviewable PRs** — no god-branch; commit sequence above.

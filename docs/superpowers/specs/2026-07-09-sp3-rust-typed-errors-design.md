# SP-3 — Rust panics → typed errors

**Date:** 2026-07-09
**Roadmap item:** SP-3 (`docs/roadmap/clean-code-audit.md`), size L, risk low (behavior improves)
**Findings source:** `docs/roadmap/audit-findings/05-rust-query.md`,
`docs/roadmap/audit-findings/06-rust-rest.md`
**Depends on:** SP-2 (merged — `query.rs` is now the `query/` package; sidecar loaders
live in `src/query/sidecar.rs`).

## Goal

Replace `panic!` / `.expect()` / `.unwrap()` on **I/O and user-input** conditions in the
Rust worker and FFI paths with propagated typed errors, so a Python caller sees the real
message and a meaningful exception type instead of a context-free `WorkerPanicked` or a
process-level abort.

**Guiding principle:** panics are retained *only* for genuine invariants (proptested
internal contracts). After SP-3, a Rust panic reaching Python means "genoray bug," never
"your VCF/args were wrong."

## Background — current state

- `ConversionError` (`src/error.rs`) has `Io`, `WorkerPanicked`, `Npy`, `ReadNpy`. Every
  conversion error surfaces to Python as a flat `PyRuntimeError::new_err(e.to_string())`
  at `src/lib.rs:184` — no category distinction.
- `src/normalize.rs` **already** defines a typed `NormalizeError`
  (`SymbolicAllele`, `RefMismatch`, `RefOutOfContig`), but `src/vcf_reader.rs:289,301`
  discards it with `.expect()`.
- The reader-thread closure panics internally; `orchestrator.rs` `join()` catches the
  panic and flattens it to `WorkerPanicked { thread }`, discarding the message.
- **Query path:** the sidecar npy loaders and the `nrvk::LongAlleleReader` random-access
  path `.expect()` on I/O. `ContigReader::open` already returns `io::Result` and is mapped
  to `PyOSError` in `src/py_query.rs:23` — so threading a `Result` through the loaders
  simply lights up that existing surface.
- `rvk::pack_snp_key_file` is a conversion-time `post_merge` hook (`fn(&Path)` in the
  `streams.rs` registry), not a direct Python call; it `.expect()`s on every I/O op.
- **FFI parser:** `bundle_from_dict` (`src/py_query_ranges.rs:139`) panics per field; it is
  called by `read_ranges` / `gather_ranges`, which already return `PyResult`.

## Design

### 1. Rust error taxonomy (two enums, one Python mapping)

The conversion pipeline and the query path are separate lifecycles with separate pyo3
entry points, so they keep separate error types, but both map to Python builtins through
one convention. **No new public Python names** (decision: builtins-only surface).

**`ConversionError`** — extend `src/error.rs` with two category variants:

- `Input(String)` — user-recoverable *content* errors: missing contig in VCF header,
  missing sample, symbolic/breakend ALT out of scope, REF/FASTA mismatch, contig absent
  from FASTA.
- `MissingFile { path }` — missing `.tbi` / `.csi` / `.fai`.
- Keep `Io`, `Npy`, `ReadNpy` (failed/corrupt disk I/O on genoray's own files) and
  `WorkerPanicked` (now reserved for *genuine* panics only).
- Add `impl From<NormalizeError> for ConversionError` → `Input`, reusing the already-typed
  `RefMismatch` / `SymbolicAllele` / `RefOutOfContig` that `vcf_reader.rs` currently
  `.expect()`s away.

**Query path** — no new enum. The sidecar loaders live inside `ContigReader::open ->
io::Result`. Thread `io::Result` through `load_offsets` / `load_max_del` /
`load_dense_max_del` and the `nrvk::LongAlleleReader` npy/pread calls, wrapping
`ndarray_npy::ReadNpyError` via `std::io::Error::other(..)`. Rides the existing
`open` → `PyOSError` mapping (`py_query.rs:23`).

**Python mapping** — replace the flat `PyRuntimeError` at `src/lib.rs:184` with
`impl From<ConversionError> for PyErr`:

| `ConversionError` variant | Python exception       |
| ------------------------- | ---------------------- |
| `Input`                   | `PyValueError`         |
| `MissingFile`             | `PyFileNotFoundError`  |
| `Io` / `Npy` / `ReadNpy`  | `PyOSError`            |
| `WorkerPanicked`          | `PyRuntimeError`       |

### 2. Threading Results through the conversion workers

- `VcfChunkReader::new` (`vcf_reader.rs:153`) → `Result<Self, ConversionError>` (index
  open, header contig lookup, sample lookup, FASTA open/contig lookup).
- `read_next_chunk` (`vcf_reader.rs:363`) → `Result<Option<DenseChunk>, ConversionError>`;
  the reader-thread loop becomes `while let Some(c) = reader.read_next_chunk(...)? { ... }`.
- The reader-thread closure returns `Result<u64, ConversionError>`. The **orchestrator
  join** (`orchestrator.rs:208–232`) gains a three-way match per thread:
  - `Ok(Ok(v))` → value,
  - `Ok(Err(e))` → **surface `e`** (the fix),
  - `Err(_)` → `WorkerPanicked` (genuine panic only).

  Applied to the reader, executor (nrvk/rvk), and writer threads.
- **Phase-2 merges** (`merge_mini_sc`, `merge_dense_class`) run on the orchestrator thread,
  not spawned — so `merge.rs` / `dense_merge.rs` worker fns return `Result` and propagate
  with plain `?` at `orchestrator.rs:256/276`. No join plumbing.
- `pack_snp_key_file` (`rvk.rs:23`) → `Result<(), ConversionError>`; the `post_merge` hook
  type in `streams.rs:31` becomes `Option<fn(&Path) -> Result<(), ConversionError>>`,
  propagated where `merge_mini_sc` invokes it.

Convert only **I/O and user-input** panics. Genuine internal invariants stay panics — in
particular the `push_long_allele` 31-bit capacity `assert!` (`nrvk.rs:40`) is a bug-guard,
not user input, and is left as-is (its message-typo fix is an SP-0 item, out of scope
here).

### 3. FFI dict parser

`bundle_from_dict` (`py_query_ranges.rs:139`) → `PyResult<RangesBundle>`. The per-field
closures map: missing key → `PyKeyError(key)`, wrong dtype/cast → `PyTypeError`, bad shape
→ `PyValueError`. Callers `read_ranges` / `gather_ranges` already return `PyResult`, so
they add `?`.

### 4. Out of scope (deferred to other sub-projects)

Everything else in findings 05/06: DRY gather-loop extraction, `DenseMap`/`StreamMap`
unification, `(bool,usize)` → `DenseClass`, the three parallel SNP/indel enums, `unsafe`
SAFETY comments (SP-2 follow-ups / SP-5). The `push_long_allele` assert-message typo is
SP-0. This spec touches error handling only.

## Commit / PR structure

One branch, three ordered commits, each independently gated green (keeps the roadmap's
"small, reviewable PRs — no god-branch" invariant):

1. **Taxonomy foundation** — extend `ConversionError` (`Input`, `MissingFile`),
   `From<NormalizeError>`, `From<ConversionError> for PyErr`, update the `lib.rs` mapping.
   Mapping change only; the happy path is unchanged.
2. **Conversion workers → Result** — `vcf_reader`, executor/nrvk-writer, `merge` /
   `dense_merge`, `pack_snp_key_file`, and the orchestrator join surfacing.
3. **Query sidecar + FFI** — sidecar / `LongAlleleReader` `io::Result`,
   `bundle_from_dict → PyResult`.

## Testing

New regression tests asserting the exception **type and message** Python sees (not merely
"it raises"):

- bad contig name → `ValueError`
- missing sample → `ValueError`
- REF/FASTA mismatch → `ValueError`
- symbolic/breakend ALT → `ValueError`
- missing `.tbi`/`.csi` → `FileNotFoundError`
- truncated / corrupt sidecar npy → `OSError`
- malformed `bundle` dict (missing key) → `KeyError`

The full existing suite (`pixi run test`) and `cargo test --no-default-features` (per the
pyo3-link requirement) stay green — behavior-preserving on the happy path.

## Public-surface / SKILL update

Builtins-only adds **no new public names**, but "bad sample now raises `ValueError`, not
`RuntimeError`" is an observable contract change. Add a short **Errors** subsection to
`skills/genoray-api/SKILL.md` documenting the exception-type contract (per the CLAUDE.md
public-behavior rule). No `docs/source/api.md` autodoc change is required.

## Invariants (per roadmap)

1. Behavior-preserving on the happy path; the only intended behavior change is
   panics/aborts → typed exceptions with real messages.
2. No new public Python names; the exception-contract change is reflected in `SKILL.md`.
3. Small, reviewable commits (three, above).

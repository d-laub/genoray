# PR #79 Architectural Fixup — Design

> Final architectural review and fixup of PR #79
> (`svar-2-orchestration`, base `svar-2`, draft) before marking it ready for
> review and merging. Goal: remove spaghetti and code smells, decouple
> orchestration from stream layout, DRY up on-disk-path handling, and improve
> overall software design — **without changing on-disk output or behavior**.

## Context

The core algorithms are sound and well-tested (41 passing: PEXT/SWAR encoder,
`BitGrid3`, tile merge, SNP/indel split). The problems are in orchestration,
module boundaries, and hygiene:

- **The encoding-agnostic seam leaks into orchestration.** `architecture.md`
  makes this the load-bearing constraint ("if you find orchestration code
  branching on SNP-vs-indel bit positions, that's a leak"). `process_chromosome`
  hand-unrolls the two streams — `merge_mini_sc::<u8>` + `pack_snp_key_file` for
  SNP, `merge_mini_sc::<u32>` for indel, plus an indel-specific offsets write.
  When `pointer`/`dense` representations arrive (M4/M11, each also split
  snp/indel), this copy-paste multiplies.
- **`lib.rs` is a dumping ground** (~484 lines): pyo3 glue + a ~150-line
  `process_chromosome` god function + ~130 lines of `/proc` monitoring + ~70
  lines of inline thread-budget arithmetic.
- **On-disk layout is stringly-duplicated** across `lib.rs`, `writer.rs`,
  `merge.rs`, `nrvk.rs`. `LongAlleleReader::new` hand-re-derives the indel path.
  No single source of truth — and filenames are still provisional.
- **~450 lines of dead commented-out code** across `rvk.rs`, `nrvk.rs`,
  `vcf_reader.rs`, `types.rs`; a 140-line `tests/test_engine.rs` with 0 active
  tests; stray `src/.ipynb_checkpoints/*.rs`; a write-only `DenseChunk.num_variants`.

Scope decisions (confirmed with maintainer): **Deep** fixup, **full
generalization** of the stream routing keyed by tag, **delete** `test_engine.rs`.

## Non-goals

- No change to on-disk output bytes, directory layout, or filenames (those
  renames are deferred to M6 decode per the PR).
- No new `pointer/` or `dense/` producers — only the machinery that lets them
  plug in later.
- No full Result-ification of worker-thread hot loops (bounded error pass only).
- No cost-model routing (M4).

## Target module map

`lib.rs` stops being a dumping ground; each concern gets a home.

| Module | Responsibility | Change |
|---|---|---|
| `lib.rs` | pyo3 `_core` module + `run_conversion_pipeline` glue only | slimmed to ~40 lines |
| `orchestrator.rs` | `process_chromosome` — thread wiring + join choreography only | **new** (moved from `lib.rs`) |
| `monitor.rs` | sampler, `/proc` parsing, `CLK_TCK` | **new** (extracted, ~130 lines out of `lib.rs`) |
| `budget.rs` | `plan_thread_budget(cores, n_chroms) -> ThreadPlan`, pure + unit-tested | **new** (extracted from the pyfunction) |
| `layout.rs` | `ContigPaths` + all path/filename construction | **new** — single source of truth |
| `streams.rs` | `StreamTag`, `StreamSpec`, `REGISTRY`, `StreamMap` | **new** — the seam registry |
| `types.rs` | `SparseSubStream` (byte-erased), `SparseChunk { streams }` | refactor; drop `num_variants` |
| `merge.rs` | `merge_mini_sc(key_bytes, …)` byte-based, no `<K>` | de-generify |
| `executor.rs` | `run_compute_engine` builds ledgers per stream generically | refactor |
| `writer.rs` | `run_io_writer` iterates streams; `run_long_allele_writer` moved in from `lib.rs` | refactor |
| `rvk.rs` | encoder/decoder + `classify_variant`; dead code deleted | refactor |
| `nrvk.rs` | `LongAlleleTableWriter` + `LongAlleleReader` (`&self` + `pread`); dead code deleted | refactor |
| `vcf_reader.rs` | reader; dead code deleted | refactor |

## The seam generalization (load-bearing)

Route by **tag**, never by width. A registry describes each on-disk sub-stream:

```rust
// streams.rs
/// On-disk sub-stream identity. Today only the var_key representation has a
/// producer; pointer/dense (M4/M11) extend this enum + REGISTRY with no changes
/// to orchestrator, writer, executor, or merge.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StreamTag { VarKeySnp, VarKeyIndel }

pub struct StreamSpec {
    pub tag: StreamTag,
    pub subdir: &'static str,          // "var_key/snp"
    pub key_bytes: usize,              // staging width: 1 (SNP u8), 4 (indel u32)
    pub post_merge: Option<fn(&Path)>, // pack_snp_key_file for SNP, None for indel
}

pub const REGISTRY: &[StreamSpec] = &[
    StreamSpec { tag: StreamTag::VarKeySnp,   subdir: "var_key/snp",   key_bytes: 1, post_merge: Some(pack_snp_key_file) },
    StreamSpec { tag: StreamTag::VarKeyIndel, subdir: "var_key/indel", key_bytes: 4, post_merge: None },
];
```

- `SparseChunk` holds a `StreamMap<SparseSubStream>` — an array indexed by
  `tag as usize` (O(1), no hashing, iterable). `SparseSubStream` stores keys as
  `Vec<u8>` plus `key_bytes`. **Type erasure is what lets an arbitrary set of
  heterogeneous-width streams live in one collection** — the crux of "keyed by
  tag."

  ```rust
  pub struct SparseSubStream {
      pub call_positions: Vec<u32>,
      pub call_keys: Vec<u8>,       // key_bytes per call, little-endian
      pub sample_lengths: Vec<u32>, // len == samples * ploidy
      pub key_bytes: usize,
  }
  ```

- `classify_variant` still returns the typed `VarKey` (`Snp(u8)`/`Indel(u32)`) —
  the seam boundary stays typed. `dense2sparse_vk` matches it once and pushes
  `key.to_le_bytes()[..key_bytes]` into `streams[tag]`. For SNP that's the 1-byte
  2-bit code (packed to 2 bits only in the post-merge pass, unchanged); for indel
  the 4-byte key.

- `merge_mini_sc` takes `key_bytes: usize` at runtime (positions stay `u32`;
  buffers become `Vec<u8>`). This **deletes the `::<u8>`/`::<u32>` + pack branch**
  from orchestration. The merge loop becomes:

  ```rust
  for spec in streams::REGISTRY {
      let dir = paths.stream_dir(spec);
      merge::merge_mini_sc(spec.key_bytes, num_chunks, n_samples, ploidy, &dir, ledgers[spec.tag]);
      if let Some(hook) = spec.post_merge { hook(&dir); }
  }
  ```

  Executor ledgers (`StreamMap<Vec<Vec<u32>>>`) and the writer iterate the
  registry the same way. Adding pointer/dense = extend the enum + `REGISTRY` +
  teach `classify_variant` to route — **zero changes** to
  orchestrator/writer/executor/merge.

- The long-allele LUT stays an indel-stream concern (only indels spill — no
  premature LUT generalization), but its path comes from `layout`/`StreamSpec`,
  not a hardcoded string. The indel offsets write is keyed off the indel
  `StreamSpec`, not special-cased inline.

### Behavior-preservation invariant

The generalized path must produce byte-identical on-disk output to the current
code for the existing `var_key/{snp,indel}` streams. The e2e tests
(`tests/test_e2e.rs`) assert both sub-streams' offsets, positions, unpacked SNP
codes, and decoded indel keys — they are the regression gate and must stay green
unmodified (except for any that reference removed helpers).

## `layout.rs` — single source of truth

A `ContigPaths` value built from `(base_out_dir, chrom)` owns every path and
filename currently `format!`'d inline:

```rust
pub struct ContigPaths { /* base_out_dir, chrom */ }
impl ContigPaths {
    pub fn new(base_out_dir: &str, chrom: &str) -> Self;
    pub fn stream_dir(&self, spec: &StreamSpec) -> PathBuf;       // {out}/{chrom}/var_key/snp
    pub fn chunk_pos(&self, dir: &Path, chunk_id: usize) -> PathBuf; // chunk_{id}_pos.bin
    pub fn chunk_key(&self, dir: &Path, chunk_id: usize) -> PathBuf;
    pub fn final_positions(&self, dir: &Path) -> PathBuf;         // final_positions.bin
    pub fn final_keys(&self, dir: &Path) -> PathBuf;
    pub fn final_offsets(&self, dir: &Path) -> PathBuf;           // final_offsets.npy
    pub fn long_alleles_bin(&self) -> PathBuf;                    // indel/long_alleles.bin
    pub fn long_allele_offsets(&self) -> PathBuf;                 // indel/long_allele_offsets.npy
}
```

`merge.rs`, `writer.rs`, `nrvk.rs` (`LongAlleleReader::new`), and the
orchestrator all derive paths through this — so the eventual provisional-name
rename (pre-M6) is a one-file change and `LongAlleleReader` can no longer drift
from the writer.

## `budget.rs` — extracted, testable

The ~70-line core-budgeting arithmetic in `run_conversion_pipeline` becomes a
pure function with a struct return, decoupled from `println!` side effects:

```rust
pub struct ThreadPlan { pub concurrent_chroms: usize, pub htslib_threads: usize }
pub fn plan_thread_budget(available_cores: usize, n_chroms: usize) -> ThreadPlan;
```

Unit-tested across the low-end (below `MIN_THREADS_PER_CHROM`), high-end
(fan-out + redistribute), and clamp branches. The pyfunction keeps the logging.

## `monitor.rs` — extracted observability

`CLK_TCK_HZ`, `find_thread_tid_by_name`, `read_thread_cpu_ticks`,
`sample_interval_secs`, `spawn_sampler` move verbatim into `monitor.rs`. This is
Linux-`/proc`-specific observability; isolating it documents that and shrinks
`lib.rs`. Public surface: `spawn_sampler(...)` unchanged.

**Platform note:** the per-thread CPU sampling reads `/proc/self/task/<tid>/stat`
and hardcodes `CLK_TCK = 100` — both Linux-only. On macOS there is no `/proc`, so
`find_thread_tid_by_name` returns `None` and the sampler prints channel fill
levels with `0%` CPU (it never panics — the degrade path already exists). Make
this explicit in `monitor.rs`: a module-level doc comment stating the CPU
figures are Linux-only, and the CPU columns emit `n/a` instead of a misleading
`0%` when no TID resolves (cheap honesty; still no `cfg` gate needed since the
code compiles and runs on both).

## Platform support

Linux and macOS are both supported; Windows is explicitly out of scope.

- **`pread`/`pwrite`** (`merge.rs`, and the new `LongAlleleReader` path) use
  `std::os::unix::fs::FileExt` (`read_exact_at` / `write_all_at`), which is
  implemented for all Unix targets including macOS — no `cfg` needed. `merge.rs`
  already depends on this today.
- **`/proc` monitoring** is the only Linux-specific surface; it degrades to
  no-CPU-numbers on macOS as described above rather than failing.
- No other module uses platform-specific APIs. The default (`extension-module`)
  and `--no-default-features` test builds should be exercised on at least one
  Unix target in CI; macOS-specific CI is not required by this PR but nothing
  here blocks it.

## Bounded error-handling pass (Tier 3, cuttable)

Introduce `ConversionError` (via `thiserror`):

- `process_chromosome` and `run_conversion_pipeline` return `Result<(), ConversionError>`.
- I/O `.expect()` in **non-threaded** code (dir creation, offset writes, file
  opens outside worker loops) become `?`.
- Thread-join panics convert to `ConversionError::WorkerPanicked { thread, .. }`.
- `run_conversion_pipeline` maps `ConversionError -> PyErr` at the pyo3 boundary.
- Hot-loop `unwrap`s inside worker threads stay (converting them is follow-up,
  flagged in the roadmap notes).

If this destabilizes the branch, it is the first thing to trim — the rest of the
fixup stands on its own.

## Hygiene

- Delete dead commented-out code: `rvk.rs` (`encode_variant_key`, old
  `dense2sparse_vk`), `nrvk.rs` (`NonReversibleLongAllele`), `vcf_reader.rs`
  (three dead fns), `types.rs` (`main()` sketch). Git history preserves them.
- Delete `tests/test_engine.rs` (0 active tests) and its
  `tests/test_engine.proptest-regressions` sidecar.
- Delete stray `src/.ipynb_checkpoints/*.rs`; add `.ipynb_checkpoints/` to
  `.gitignore`.
- Remove write-only `DenseChunk.num_variants` (redundant with `genos.shape.0`).
- `LongAlleleReader::get_allele`: `&mut self` + `seek`/`read` → `&self` +
  `read_exact_at` (pread), matching `merge.rs`'s stateless pattern.

## Sequenced workstream (tests green at every step)

1. **Hygiene** — delete dead code, `test_engine.rs` + sidecar, `.ipynb_checkpoints`
   (+ gitignore), drop `num_variants`.
2. **Pure extractions** — `monitor.rs`, `budget.rs` (+ unit tests).
3. **`layout.rs`** — replace every inline path string; fix `LongAlleleReader`'s
   hand-derived path.
4. **`streams.rs` + `types.rs`** — `StreamTag`/`StreamSpec`/`REGISTRY`/`StreamMap`,
   byte-erased `SparseSubStream`, `SparseChunk { streams }`.
5. **Generalize** `merge` (`key_bytes` runtime), `executor` (stream-keyed
   ledgers), `writer` (iterate streams + long-allele writer moved in), `rvk`
   classify routing.
6. **`orchestrator.rs`** — slim `process_chromosome`, registry-driven merge loop;
   `lib.rs` reduced to pyo3 surface.
7. `LongAlleleReader` → `&self` + `pread`.
8. Bounded error-handling pass.
9. Final gate — `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`,
   `cargo test` (default env + `--no-default-features`) all green; reconcile
   `docs/roadmap/` references to module names.

## Verification

- Full suite green: `pixi run cargo test` (currently 41 passing → expect
  budget/layout unit tests added, `test_engine.rs`'s 0 removed; e2e count
  unchanged). The e2e tests are the byte-identical-output regression gate.
- `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` clean.
- Spot-check: a full conversion of the e2e fixture produces the same
  `final_positions.bin` / `final_keys.bin` / `final_offsets.npy` /
  `long_alleles.bin` bytes before and after.

## Implementation approach

TDD + subagent-driven development on **Sonnet** (per maintainer's global prefs;
Opus only for second-pass fixes if an implementer critically fails). Work in a
git worktree under `.claude/worktrees`. Ensure prek hooks are installed before
committing (`.pre-commit-config.yaml` present).

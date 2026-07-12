# SVAR2 INFO/FORMAT fields — PR#100 follow-ups, cleanup & field-path optimization

**Date:** 2026-07-12
**Branch:** `worktree-svar2-field-specs` (open PR #100)
**Status:** Design approved; ready for implementation plan.

## Context

PR #100 added the **write path** for scalar-numeric INFO (per-variant) and FORMAT
(per-call) fields to `SparseVar2.from_vcf` (spec #1 of the 3-part SVAR2 fields
effort). It is functionally complete and green, but its PR body lists three
deferred follow-ups, and a code survey surfaced a few concrete smells plus one
clear field-specific performance target.

This spec covers a single round of polish on that branch: close the follow-ups,
DRY/clean the new code, and run a **measurement-driven** optimization pass scoped
to the field-write code this PR newly adds. All work lands as new commits on the
existing PR #100 branch.

### Explicitly out of scope

- **The reader/htslib path.** Prior profiling established VCF→SVAR2 conversion is
  reader-bound (htslib inflate+parse ~78%), and that path already received a
  GT-decode + per-word-pack + parallel-pack optimization round. We do not
  re-open it here.
- The read/query path for fields (that is spec #2 of the effort).
- Any public API change. No new/renamed/removed names reachable from
  `import genoray`; `skills/genoray-api/SKILL.md` is expected to need no change
  (to be confirmed at close-out).

## Sequencing

Work is ordered so the risky stages (refactor, optimize) run under a test safety
net, and so the optimization is bracketed by before/after measurements:

1. **Lock behavior** — add the missing tests (follow-ups #2, #3).
2. **DRY & cleanup** — follow-up #1 + surveyed smells, refactor kept green.
3. **Baseline profile** — capture perf + callgrind on a fields-enabled run.
4. **Optimize** — measurement-driven, output byte-identical, kept green.
5. **Re-profile & document** — report before/after, update CHANGELOG.

Each stage ends with both suites green:
`cargo test --no-default-features --features conversion` and `pixi run test`.

## Stage 1 — Tests first (follow-ups #2, #3)

Add coverage the PR body flagged, *before* refactoring so it guards the changes:

- **Combined `signatures=True` + fields e2e.** A `from_vcf` round-trip with
  `signatures=True` and both `info_fields` and `format_fields` set. Cost terms are
  additive/orthogonal; this confirms they compose end-to-end.
- **`VarKeyIndel` `field_calls` path.** The var_key-indel sub-label is currently
  untested; add a case that routes an indel through it and asserts stored values.
- **Multi-field ordering.** ≥2 fields of each category; assert each field's
  per-`values.bin` content is correct and fields do not cross-contaminate.
- **`f32→f32` no-op rewrite.** Finalize on an already-concrete `f32` field
  (the rewrite-to-same-dtype path).

These are Rust unit/integration tests and/or Python e2e tests, matching where the
existing field tests live (`field_finalize.rs` tests, `tests/`, and the Python
`from_vcf` round-trip test).

## Stage 2 — DRY & cleanup

### Follow-up #1: DRY the two merges

`merge_mini_sc` (`src/merge.rs:25`) and `merge_var_key_field_values`
(`src/merge.rs:294`) share:

- **Phase A** — global per-column offset derivation + per-chunk local offsets from
  the RAM ledger (identical arithmetic; the code comment at merge.rs:307 flags the
  deliberate duplication).
- **Phase B** — the adaptive-tile, parallel pread→interleave→pwrite gather loop.

Extract a shared helper parameterized by item width and the per-item transfer so
both call sites use it. `merge_mini_sc` moves two payloads per item (pos u32 + key
`key_bytes`), `merge_var_key_field_values` moves one (`item_width`); the helper
must express that difference cleanly (e.g. a slice of `(src_file, item_width,
dest_file)` payloads sharing one offset/tile schedule) rather than by widening to
`Any`-style dynamic dispatch. Target: eliminate the ~110 duplicated lines while
keeping the working pos/key merge behavior identical.

### Stringly-typed prefix → enum

`merge_dense_field_values` (`src/dense_merge.rs:120`) takes `prefix: &str` and
matches `"finfo"`/`"fformat"` at runtime, returning a `ConversionError::Input` on
anything else. Replace with the existing `FieldCategory` enum (`src/field.rs`),
selecting the chunk-path builder by variant. Makes the invalid state
unrepresentable and removes the runtime error branch.

### Minor

While in these files, audit remaining `&str` sub-labels / field naming and tidy
only where it is a clear win. No speculative refactors, no gold-plating.

## Stages 3–5 — Field-path optimization (measured)

### Baseline (Stage 3)

Build with the existing profiling profile:

```
RUSTFLAGS="-C force-frame-pointers=yes" maturin develop --profile profiling
```

Run `SparseVar2.from_vcf` with `info_fields`/`format_fields` enabled on the
`gen_from_vcf.sh` test data, via a throwaway driver in `$CLAUDE_JOB_DIR/tmp`
(no committed benchmark). Capture with `perf` (sampling / flamegraph) and
`callgrind` (instruction-level). Record the field-code cost share as the baseline.

### Prime target: `src/field_finalize.rs`

The global finalize pass is the newest field-specific cost and is independent of
the reader-bound story. Three concrete issues:

1. **Double disk read.** `finalize_one` calls `scan(&files, …)` then, per file,
   `rewrite_file(…)`, and both call `read_staged`, so every staged `values.bin` is
   read from disk twice. Fuse to a single read (or `mmap` the file and make two
   passes over one in-memory buffer).
2. **Full `Vec<f64>` materialization.** `read_staged` collects the whole file into
   a `Vec<f64>` (2× on-disk size) — for FORMAT fields that is O(total carrier
   calls). Iterate over the byte buffer instead of collecting.
3. **Serial.** `finalize_fields` maps over fields serially and each field's
   file loop is serial, while the rest of the pipeline is rayon-parallel.
   Parallelize across fields and/or files.

Constraint: on-disk output must be **byte-for-byte identical** to the current
finalize. The new multi-field/no-op tests plus the existing round-trip assertions
verify this.

### Secondary target: `merge_dense_field_values`

Currently reads each chunk file fully into one growing `Vec<u8>` and writes the
concatenation. Stream chunk→dest (buffered copy) to bound peak memory. Apply only
if the baseline profile shows it is worth it.

### `cargo asm`

After the changes, inspect generated code for the hot inner loops — `encode`
(`field_finalize.rs:399`) and the field-routing loop in `dense2sparse_vk` — to
confirm codegen (no surprise bounds checks / spills) on the per-element path.

### Close-out (Stage 5)

- Re-profile the same fields-enabled run; report before/after: callgrind
  instructions retired, wall-time, and peak RSS for the finalize stage.
- Add a human-readable entry under `## Unreleased` in `CHANGELOG.md`.
- Confirm no `skills/genoray-api/SKILL.md` change is required (no public surface
  touched).

## Guardrails & success criteria

- Both test suites green at the end of every stage.
- Optimization produces byte-identical SVAR2 field output vs. pre-optimization.
- Measured, reported reduction in the field-code cost share of a fields-enabled
  conversion (finalize instructions/RSS the headline metric); no regression in the
  reader path.
- No public API surface change.

## Notes

Two untracked doc files sit in the worktree from the PR #100 session
(`docs/superpowers/specs/2026-07-11-svar1-to-svar2-conversion-design.md` and
`docs/superpowers/plans/2026-07-11-svar2-info-format-fields-write.md`). Left
untouched unless the user decides to commit them.

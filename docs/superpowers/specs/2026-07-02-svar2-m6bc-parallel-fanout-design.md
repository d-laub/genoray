# SVAR 2.0 M6b + M6c — Parallel Worktree Fan-out

> **Status:** design approved, pre-implementation · **Epic:** SVAR 2.0 · **Builds on:** M6a (shipped, `svar-2` @ `d836a37`).
>
> This is an **orchestration** spec: how the two remaining M6 consumers (M6b, M6c) are
> carved into conflict-free parallel worktrees and driven via
> `superpowers:subagent-driven-development`. The *what* of each milestone is already
> designed in
> [`2026-07-02-svar2-m6-consumer-interfaces-design.md`](2026-07-02-svar2-m6-consumer-interfaces-design.md)
> and the [roadmap](../../roadmap/svar-2.md); this spec pins the *how of running them in
> parallel* — topology, the conflict boundary, cross-repo dev wiring, and execution.

## Goal

Complete **M6b in full** (genoray-side raw interface **and** the GenVarLoader Rust
consumer) and **M6c** (Python decode) as three parallel workstreams, each in its own
worktree, mergeable independently, coordinating only through the **frozen
`BatchResult` → numpy contract** locked in M6a.

## What M6a already gives us (no re-derivation)

- `BatchResult` (`src/query.rs`) already holds the whole two-channel payload: `vk`/`vk_off`,
  shared `dense`, `dense_range`, `dense_present`/`dense_present_off`, and `decode_hap()`.
  Neither milestone needs to touch `query.rs` or `lib.rs`.
- `PyContigReader` (`src/py_query.rs`) is registered in `_core`; `py_convert.rs` has the
  `u32→i32` / `i64` / `u8` numpy helpers; `SparseVar2` (`python/genoray/_svar2.py`) reads
  `meta.json` and holds one reader per contig.
- The frozen `BatchResult → numpy` dtype/shape table (M6 spec, "Frozen contract") is the
  fixed interface all three worktrees code against.

## Topology

Three worktrees, all branched from current `svar-2` (genoray) / `main` (gvl):

| # | Repo | Path | Branch | Scope |
| --- | --- | --- | --- | --- |
| 1 | genoray | `.claude/worktrees/svar-2-m6b` | `svar-2-m6b` | M6b genoray side: raw two-channel numpy exposure + dense-window subsetting; hosts `svar2-codec` |
| 2 | genoray | `.claude/worktrees/svar-2-m6c` | `svar-2-m6c` | M6c: decode → `seqpro.rag.Ragged` + decode-free region counts |
| 3 | gvl | `~/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel` | `svar2-m6b-kernel` | M6b gvl side: two-source splice kernel + direct-feed |

All worktree dirs live under each repo's `.claude/worktrees/` per the workspace convention.

## Step 0 — shared prep commit on `svar-2` (before branching 1 & 2)

Done once on `svar-2` so both genoray worktrees inherit it with **zero shared-line edits
afterward**:

1. **Enable `multiple-pymethods`** on the `pyo3` dependency in `Cargo.toml`. Without it,
   two `#[pymethods] impl PyContigReader` blocks in two files do not compile, and one shared
   block guarantees a merge conflict. (The M6 spec assumed multiple blocks "just work"; they
   require this feature.)
2. **Stub the owned Rust modules:** `src/py_query_batch.rs` (M6b) and `src/py_query_decode.rs`
   (M6c), each an empty `#[pymethods] impl PyContigReader {}`, declared `mod` in `lib.rs`.
3. **Stub the owned Python mixins:** `_svar2_batch.py` (`class _BatchQueryMixin`) and
   `_svar2_decode.py` (`class _DecodeMixin`), and change the class to
   `class SparseVar2(_BatchQueryMixin, _DecodeMixin)`.

After Step 0, each genoray worktree edits **only files it exclusively owns** (its
`py_query_*.rs`, its `_svar2_*.py` mixin, its own test file). The branches merge back to
`svar-2` in any order with no conflict.

## Conflict boundary — file ownership

**Worktree 1 (M6b genoray):** `src/py_query_batch.rs`, `python/genoray/_svar2_batch.py`,
`tests/test_svar2_batch.py`. Converts `BatchResult` fields → the frozen numpy contract; adds
dense-window subsetting. Exposes a `SparseVar2` method returning the raw arrays.

**Worktree 2 (M6c):** `src/py_query_decode.rs`, `python/genoray/_svar2_decode.py`,
`tests/test_svar2_decode.py`. Decodes in **Rust** (reuse `BatchResult::decode_hap`, return
flat `pos`/`ilen` `i32` + `allele` `u8` + `str_offsets` + shared variant-axis offsets as
numpy); Python wraps in `Ragged.from_fields` shape `(R, S, P, None)` and computes the
decode-free `(R, S, P)` counts (`vk_off` diffs + `dense_present` popcount).

Neither touches `query.rs`, `lib.rs`, `py_convert.rs`, or `_svar2.py` after Step 0 — the
data and helpers they need already exist.

## Cross-repo dev wiring (worktree 3 → worktree 1)

gvl develops against the **local** genoray M6b worktree, not the unpublished release:

- **Rust:** gvl's core only ever links `svar2-codec` (std-only, zero runtime deps — no `pyo3`).
  Add it as a **Cargo path-dep** to worktree 1's crate:
  `svar2-codec = { path = "/carter/users/dlaub/projects/genoray/.claude/worktrees/svar-2-m6b/svar2-codec" }`.
  No `pyo3`/`numpy` version conflict with gvl's `0.28` vs genoray's `0.29`, because the FFI
  boundary is numpy arrays (version-independent) and `svar2-codec` links no pyo3.
- **Python:** override gvl's `genoray==2.12.3` pin with an **editable path-install** of
  worktree 1 in a dedicated pixi dev feature (maturin-built). The editable install shadows the
  PyPI pin so gvl imports the svar-2 `SparseVar2`.

### Release gate (unchanged, manual)

The dev wiring is for iteration only. **Worktree 3's PR is mergeable only after** `svar2-codec`
is on crates.io and `svar-2` genoray is on PyPI. The final commit on worktree 3 flips both
deps from local paths back to the published versions; that flip and the publish are the
maintainer's manual call. Worktrees 1 and 2 (genoray-side) are independently mergeable and do
**not** wait on the release.

## Dependency ordering

- Worktrees 1 & 2 are fully independent (frozen contract + file ownership) — run in parallel.
- Worktree 3 codes against the **frozen numpy contract**, so it starts in parallel too. Its
  *runtime* integration needs worktree 1's methods to actually exist, so worktree 1 lands the
  contract-satisfying `SparseVar2` batch method **early** (its first task), unblocking gvl's
  end-to-end runs.

## Execution via subagent-driven-development

- Each worktree gets its own implementation plan (`superpowers:writing-plans`), then is driven
  by `superpowers:subagent-driven-development`. Implementers are **Sonnet** (per standing
  preference); Opus only for second-pass fixes where an implementer critically failed.
- **Cross-repo cwd hazard:** subagents default to the main repo, not the worktree
  ([[subagent-cwd-in-worktrees]]). Every gvl-worktree task must `cd` into worktree 3 and guard
  with `git rev-parse --show-toplevel` before any git/build action — doubly important here
  because worktree 3 is in a *different repo* (`GenVarLoader`, not `genoray`).
- Pre-commit / prek hooks installed in each worktree before committing.

## Verification (per the M6 spec's table)

- **Codec parity** (worktree 1 boundary): `svar2-codec` decode byte-identical to `rvk.rs`
  (already green).
- **Two-channel ≡ materialized** (worktrees 1+2 cross-check): gvl's two-source splice over the
  raw arrays yields the same per-hap variant sequence as M6c's materialized `Ragged`. The
  in-genoray oracle is `BatchResult::decode_hap`; the cross-repo oracle is worktree 3's kernel
  vs. worktree 2's `Ragged`.
- **Count shortcut** (worktree 2): decode-free `(R,S,P)` counts equal `len()` of the
  materialized `Ragged` per `(region, sample, ploid)`.
- **`ilen` round-trip** (worktree 3): known SNP/INS/DEL atoms → SVAR2 → gvl decode, asserting
  genoray `ILEN` and gvl's `ref_end` math agree per class.

## Open questions (deferred to implementation, not blockers)

- **Dense-window multi-region packing** (worktree 1): concatenated bit-buffer + per-region/hap
  offsets vs. per-region arrays — resolve by benchmark during M6b, per the M6 spec.
- **Two-source vs. measured fallback** (worktree 3): if the two-source kernel's complexity
  outweighs the win for the first integration, fall back to one gathered per-hap stream feeding
  gvl's existing single-table kernel (bounded dense duplication). Decide by measurement.

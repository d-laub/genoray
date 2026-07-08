# SVAR2 MVP Ship Plan

> **Status:** design · **Date:** 2026-07-07 · **Branch:** `svar-2` (genoray) + `svar2-m6b-kernel` / [mcvickerlab/GenVarLoader#266](https://github.com/mcvickerlab/GenVarLoader/pull/266) (gvl)
>
> Companion to [`../../roadmap/svar-2.md`](../../roadmap/svar-2.md). This is the ordered
> release checklist that takes the SVAR2 MVP from "code-complete on branches" to
> "published on PyPI/crates.io with gvl support merged."

## Verdict

The SVAR2 MVP is **code-complete**. Shipping it "alongside GVL support" is gated by the
**release chain**, not by missing functionality.

- **genoray (`svar-2`)** — every MVP milestone is implemented and tested: conversion
  (M1–M2b), per-contig layout (M3), dense + cost model (M4), overlap queries (M5), the
  query decode core and its consumer seams (M6.0/M6.1/M6a), the search/gather split and
  read-bound per-class gather (M6d/M6e), the Python decode → `Ragged` + region counts
  (M6c), and the beyond-MVP conversion hardening (M13 skip-out-of-scope, M14 parallel
  reader). The branch is 213 commits ahead of `main`.
- **gvl (GenVarLoader#266)** — the complete, reviewed M6b implementation: `.svar2` as a
  `gvl.write` source and a live `Dataset` read backend across all four output modes via a
  read-bound all-Rust FFI, plus `variant-windows` and `unphased_union`, plus the read-bound
  `getitem` perf optimizations (B1–B5). **Extensively benchmarked, profiled, and
  optimized** (PR #266 + its `tmp/svar2_mvp/` harness); storage win is real (1.46–5.67×).
  The PR is a **draft blocked on the genoray release gate** — it dev-wires genoray via local
  path-deps + a local wheel and cannot build upstream until genoray `svar-2` is published.

No genoray-side functionality blocks ship. The remaining work is the publish/merge
sequence below plus pre-release doc finalization.

## Decisions (resolved)

- **genoray version bump: `3.0.0` (major).** The `genoray write` CLI now **defaults to
  SVAR2** (M13), a breaking change for existing SVAR1 CLI users (`genoray write svar1`
  preserves the old behavior). `SparseVar2` itself is additive, but the default flip is
  semver-breaking, so major is the honest bump.
- **gvl resolves `genoray_core` by git tag.** gvl pins the query-only `genoray_core` crate
  to a git tag/rev of the genoray release commit (`default-features = false`, htslib-free).
  This avoids publishing `genoray_core` to crates.io and keeps that Rust core private.
  Only **`svar2-codec`** goes to crates.io.
- **MVP scope includes `variant-windows` + `unphased_union`.** They are done, benchmarked,
  and on the branch; ship them as part of the MVP rather than deferring.
- **No post-ship latency gate.** Latency/throughput profiling and optimization are already
  done in PR #266 (B1–B5, `tmp/svar2_mvp/` harness). Nothing latency-related is deferred.

## Release checklist (ordered)

### 1. Pre-flight — genoray (`svar-2`)
- [ ] Finalize `skills/genoray-api/SKILL.md`: drop the "read/query API still evolving"
      hedge on `SparseVar2`; document the now-shipped read/query surface — `decode`,
      `region_counts`, `find_ranges` / `gather_ranges` / `read_ranges` (with `samples=` /
      `out=`), and `from_vcf` (`reference` XOR `no_reference`, `skip_out_of_scope`, returns
      dropped count). This is required by the repo's public-API-sync rule in `CLAUDE.md`.
- [ ] Reconcile `docs/roadmap/data-model.md` and `docs/roadmap/architecture.md` with the
      shipped code (per the roadmap working agreement) — any remaining doc/code drift is
      fixed doc-side in the release PR.
- [ ] CHANGELOG / release notes: highlight the breaking `genoray write` default and the new
      `SparseVar2` public surface.
- [ ] Full suite green under the repo's test config (`--no-default-features --features
      conversion` for cargo; the SVAR2 pytest suite).
- [ ] `cz bump` → **3.0.0**.

### 2. Publish `svar2-codec` → crates.io
- [ ] Maintainer action; `cargo publish --dry-run` is already clean (M6.0). Publish
      `svar2-codec 0.1.0`.

### 3. Release genoray → PyPI + tag
- [ ] Merge `svar-2` → `main`.
- [ ] Build + publish `genoray 3.0.0` to PyPI.
- [ ] **Tag the release commit** (e.g. `v3.0.0`) — this tag is gvl's `genoray_core` pin.

### 4. gvl M6b close-out (GenVarLoader#266)
- [ ] Repoint deps off local paths: `svar2-codec` → crates.io `0.1`; `genoray_core` → git
      tag/rev of the genoray release commit (`default-features = false`); `genoray` Python
      dep in `pixi.toml` → PyPI `3.0.0` (currently a local `2.15.0` wheel).
- [ ] Reconcile the pyo3 / numpy pins with the published genoray build.
- [ ] Confirm upstream CI builds; merge `svar2-m6b-kernel` → gvl `main` (includes
      `variant-windows` + `unphased_union`).
- [ ] Update gvl `api.md` for the new `.svar2` source + read backend (gvl's docs-audit gate).

### 5. Announce
- [ ] Note SVAR2 availability + the `genoray write` default change in both repos' release
      notes.

## Open questions

None blocking. The two structural decisions (version level, `genoray_core` resolution) are
resolved above. If the pyo3/numpy pin reconciliation in step 4 surfaces an ABI conflict,
that is handled inside the gvl PR, not here.

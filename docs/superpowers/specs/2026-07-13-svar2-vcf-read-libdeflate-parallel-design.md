# VCF→SVAR2 conversion read-path speedup — libdeflate + parallel BGZF

**Date:** 2026-07-13 · **Home:** `genoray` · **Status:** design, not yet started

> Follow-on to the two 2026-07-07 profiling designs
> ([`svar1-vs-svar2-timing-and-profiling`](2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md),
> [`svar2-parallel-reader`](2026-07-07-svar2-parallel-reader-design.md)). Read those
> first — the perf findings, the byte-identical discipline, and the harness they
> describe are the foundation here. SVAR2 is unreleased, so we are free to overhaul
> the read path as long as the store stays byte-identical.

## Problem

After the landed reader micro-opts (raw-GT decode, per-word packing, parallel
packing), single-contig VCF→SVAR2 conversion is **htslib input-bound**. The gdc
(16007-sample, 4.5M-variant, somatic chr21) reader profile is dominated by:

- **`inflate_fast` ~31% + `crc32_z` ~14% ≈ 45%** — BGZF block decompression.
- `bcf_get_format_values` ~9% — htslib copying the raw GT FORMAT array out.
- packing ~16% (already parallelized), executor transpose ~17% (single-thread,
  out of scope — hits bank determinism).

Raising `MAX_HTSLIB_THREADS` (the htslib `bgzf_mt` decode-thread cap) was previously
found **inert**. ~24 of 32 cores sit idle on a single-contig file.

## Root cause found during design: htslib is linked against zlib, not libdeflate

`inflate_fast` and `crc32_z` are **zlib** symbols. `rust-htslib` is declared in
`Cargo.toml` as `default-features = false` with **no `libdeflate` feature**, so
hts-sys vendors and compiles htslib against zlib's inflate + slice-by-8 CRC32.

Two facts make this the headline:

1. `rust-htslib` (v1.0) exposes `libdeflate = ["hts-sys/libdeflate"]`, and
   `hts-sys` (v2.2.0) exposes `libdeflate = ["libdeflate-sys"]`. The feature is
   present and simply switched off.
2. `libdeflate.so` already ships in the pixi env (`.pixi/envs/*/lib/libdeflate.so`).

libdeflate — the same decompressor PLINK2 uses — is typically ~1.5–2.5× faster at
DEFLATE decompression than zlib and uses a SIMD (PCLMULQDQ) CRC32. Turning it on is
a one-line change that attacks the single largest cost directly, and it is orthogonal
to thread count (each block's inflate gets cheaper on 1 or 8 threads), which is also
the most likely reason the thread-cap bump was inert: more threads on slow zlib
blocks, gated by CRC + serial parse.

## Goal & stopping rule

Speed up single-contig VCF/BCF→SVAR2 conversion, gated at every step by the harness.
**Stopping rule: diminishing returns** — push each stage, stop when the next stage's
expected win is not worth its risk/effort; report before/after per stage. No fixed
wall-clock target. **Non-negotiable invariant: byte-identical store output** on both
germline and gdc after every change.

## Approach: staged, measurement-gated

Each stage is gated on the harness. We escalate to the expensive overhaul (Stage 2)
**only if** the cheap wins (Stage 0/1) plateau short of what the profile says is
achievable. This honors the project's "measure, don't guess" principle and the
user's explicit "overhaul *if need be*" framing.

### Stage 0 — Enable libdeflate (near-free, do first)

- Change `rust-htslib = { version = "1.0", default-features = false, optional = true }`
  → add `features = ["libdeflate"]`. Update the `Cargo.toml` comment (which currently
  documents only the dropped bzip2/lzma/curl) to record that libdeflate is on and why.
- Confirm the vendored htslib actually links libdeflate: rebuild, then verify the
  profile no longer shows `inflate_fast`/`crc32_z` (should show `libdeflate_*`
  symbols instead), or check the linked `libhts` build config.
- **Wheel/CI implication:** the release wheels build from `ci/wheel/pixi.toml` with
  auditwheel/delocate repair. libdeflate becomes a build-and-runtime dependency of
  the vendored htslib — confirm it is available on all four wheel platforms in the CI
  manifest and gets vendored/repaired into the wheel (this is the one non-trivial
  part of an otherwise one-line change). If bundling is a problem, fall back to the
  hts-sys static libdeflate.
- **Expected:** the ~45% BGZF cost roughly halves. Rough projection (measure, don't
  trust): gdc ~18min → ~13–14min, germline 36.5s → ~28s. Byte-identical.

### Stage 1 — Re-sweep htslib decode threads with the fast decompressor

- With per-block inflate now cheap, htslib's `bgzf_mt` may finally scale where it was
  inert against zlib. Re-sweep `htslib_threads` (8/12/16) on the idle cores; keep the
  win behind `plan_thread_budget` (only widen when spare cores exist), mirroring the
  existing budget tests.
- If htslib's own multithreading now scales acceptably, **stop here** — Stage 2's
  custom decompressor is unnecessary.

### Stage 2 — Parallel BGZF frontend (the overhaul; only if Stage 1 plateaus)

If htslib's `bgzf_mt` still fails to use the idle cores after libdeflate, build a
custom multithreaded BGZF decompressor that feeds htslib's **trusted BCF parser**:

- **Why this shape:** BGZF blocks (≤64KB compressed) are independent — inflate is
  embarrassingly parallel. A custom layer reads compressed blocks serially (cheap
  I/O), fans `libdeflate_deflate_decompress` across idle cores (via the `libdeflater`
  crate), and reassembles the decompressed BCF byte stream in order.
- **Feed htslib, don't replace it:** hand the decompressed bytes to htslib's BCF
  record parser (through a pipe / in-memory `hFILE` / uncompressed-BCF stream — exact
  mechanism to prototype). htslib does **zero inflate**, only parse. This keeps the
  exact atomize / left-align / raw-GT-decode semantics that the byte-identical gate
  depends on — we are moving *where inflate happens*, not *what gets parsed*.
- **Inspiration, not code:** PLINK2's `plink2_bgzf.cc` is the reference algorithm for
  a parallel libdeflate BGZF reader, but plink-ng is **GPLv3 — no code may be copied**
  into genoray. Study it for the block-queue/tuning approach and reimplement clean.
  genoio (Rust, less battle-tested) is a secondary reference. Note the existing
  precedent: the `zstd` dep comment already documents "no plink-ng code."
- Sequential full-contig read needs no CSI random-access, so ordered streaming (no
  virtual-offset seeks) is sufficient.
- **Fallback of last resort (not the plan):** a full pure-Rust reader/parser via
  noodles (MIT — dependency-safe) or hand-rolled. Highest semantic risk — it must
  reproduce htslib's exact record decoding — so only if the pipe-to-htslib frontend
  proves unworkable.

## Profiling harness

Reuse and extend `/carter/users/dlaub/svar_bench/`, which already has: both filtered
chr21 BCFs (`chr21.filt.bcf` germline 3202 samples, `gdc.chr21.filt.bcf` somatic
16007 samples), CSI indexes, `.gvi`, oracle store hashes (`oracle.chr21.hash`,
`oracle.gdc.hash`), `run_svar2.py`, `storehash.sh`, and perf capture scripts.

Add for the fast dev-loop:

- **A small slice of each dataset** — e.g. the first ~5–10 Mb of chr21 via
  `bcftools view <file> chr21:1-10000000 -Ob` — giving ~10–30s germline / ~1–2min
  gdc runs so profile-optimize-remeasure iterates in minutes, not the ~18-min full
  gdc run. Generate slice-specific oracle hashes too.
- Keep **full chr21** as the byte-identical validation gate (the small slice is for
  iteration speed only; correctness is always confirmed on the full file).

Profiling protocol (unchanged from the prior pass):

- Build the profiling wheel:
  `RUSTFLAGS="-C force-frame-pointers=yes" maturin develop --profile profiling`;
  restore `maturin develop --release` after.
- Run on a **dedicated `carter-compute` node** (`sbatch -p carter-compute -c 32
  --mem=128G` holder + `srun --jobid=<id> --overlap`) — the login node is unusable
  under load.
- `perf record -g --call-graph fp` → `perf report` + flamegraph; separate the
  per-thread `read-*` / `exec-*` split so inflate vs parse vs pack is attributable.
- Reference FASTA: `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`.

## Correctness & verification (non-negotiable)

After **every** stage, on **both** full datasets:

- **Byte-identical store hash** vs the oracle (`storehash.sh` sha256 of every store
  file == `oracle.<dataset>.hash`). Any diff is a regression, full stop.
- `cargo test --no-default-features --features conversion` (the link-flag gotcha —
  default `extension-module` breaks the test binary; see the memory note on
  `--no-default-features`).
- `pytest tests -m "not network"`.
- Stage 2 only: a targeted proptest that the parallel-decompressed BCF byte stream is
  identical to htslib's own decompression across block boundaries and short final
  blocks (missing/vector-end GT sentinels included).

## Skill / docs impact

This is a pure performance change with **no public-API surface change** — return
shapes, dtypes, mode constants, and coordinate/missing conventions are unchanged. No
`skills/genoray-api/SKILL.md` update expected. Add a human-readable `CHANGELOG.md`
entry under `## Unreleased` (do not bump the version by hand). Update the roadmap
`svar-2.md` if the pipeline/thread-model description changes.

## Risks / open questions

- **libdeflate in the release wheels** — the main non-trivial part of Stage 0. Must
  be vendored/repaired into all four platform wheels; verify against `ci/wheel/
  pixi.toml`. Toolchain pins are duplicated between the two manifests and must stay
  in sync.
- **libdeflate actually gets linked** — the hts-sys `libdeflate` feature must cause
  the vendored htslib to build `--with-libdeflate`; confirm by symbol inspection
  post-build, don't assume the feature flag alone did it.
- **Stage 1 may already suffice** — if the thread re-sweep scales after libdeflate,
  Stage 2 is dead code; do not build it speculatively.
- **Stage 2 pipe/hFILE mechanism** — feeding decompressed bytes into htslib's parser
  without its own BGZF layer needs a prototype spike; the exact plumbing
  (uncompressed-BCF stream vs custom `hFILE` plugin vs pipe) is unresolved and is the
  first thing to de-risk if we reach Stage 2.
- **GPL boundary** — no plink-ng code copied; algorithm inspiration only. noodles
  (MIT) is the only read-path code we may depend on directly.
- **Executor transpose (~17%) is the next ceiling** and is explicitly out of scope —
  parallelizing it hits long-allele-bank determinism (byte-identical hazard).

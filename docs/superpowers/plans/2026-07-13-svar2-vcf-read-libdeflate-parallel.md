# VCF→SVAR2 Read-Path Speedup (libdeflate + parallel BGZF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up single-contig VCF/BCF→SVAR2 conversion, whose reader is bottlenecked on zlib BGZF decompression, starting with a near-free switch to libdeflate and escalating to parallel decompression only if measurement justifies it.

**Architecture:** Staged and measurement-gated. Stage 0 turns on the `libdeflate` cargo feature so the vendored htslib decompresses with libdeflate instead of zlib. Stage 1 re-sweeps htslib decode threads now that per-block inflate is cheap. Stage 2 (conditional, gated on Stage 1's result) is a custom parallel BGZF frontend that feeds htslib's trusted BCF parser. Correctness is defended by a byte-identical store-hash oracle after every change.

**Tech Stack:** Rust (`rust-htslib`/`hts-sys` vendored htslib, `rayon`, `libdeflate`/`libdeflater`), PyO3, maturin, pixi, `perf`, SLURM `carter-compute`.

## Global Constraints

- **Byte-identical output is non-negotiable.** After every change, the SVAR2 store content-hash MUST equal the oracle on **both** datasets: `oracle.chr21.hash = f43712ff68ca438f853ca9883533a4c4e9db6b4b6e9ed752efdbaf1fa6d71a8b`, `oracle.gdc.hash = dc33c477bd117e59d9280178792580bcafc3c1a9c312380c0dc3a4755e1f5ae0`. Any diff is a regression — stop.
- **Rust test command:** `cargo test --no-default-features --features conversion` (default `extension-module` breaks the test binary: `undefined symbol: _Py_Dealloc`). Always `export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$` first — cargo on the NFS `target/` bus-errors.
- **Python test command:** `pixi run pytest tests -m "not network"`.
- **No plink-ng code copied** — plink-ng is GPLv3; algorithm inspiration only. noodles (MIT) is the only read-path code we may depend on directly.
- **Do not bump the version or edit the changelog version headers by hand** — accumulate human-readable entries under `## Unreleased` in `CHANGELOG.md`; CI cuts the versioned section.
- **No public-API change expected** — if any public name/shape/dtype changes, `skills/genoray-api/SKILL.md` must be updated in the same change. None is anticipated here.
- **All timing/perf runs on a dedicated `carter-compute` node** (`sbatch -p carter-compute -c 32 --mem=128G` holder + `srun --jobid=<id> --overlap`), never the login node.
- **Build from the worktree** so code changes take effect: run `maturin develop` with `--manifest-path <worktree>/pyproject.toml` (or `cd <worktree>` first), not the main checkout.
- **Harness:** `/carter/users/dlaub/svar_bench/` (`BENCH`). Repo reference FASTA `REF=/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`. Driver `run_svar2.py <bcf> <out_store> <ref> <threads>`; hasher `storehash.sh <store_dir>`.

---

## Task 1: Build the fast-iteration harness slice

**Files:**
- Create: `/carter/users/dlaub/svar_bench/chr21.slice.bcf` (+ `.csi`)
- Create: `/carter/users/dlaub/svar_bench/gdc.chr21.slice.bcf` (+ `.csi`)
- Create: `/carter/users/dlaub/svar_bench/oracle.chr21.slice.hash`
- Create: `/carter/users/dlaub/svar_bench/oracle.gdc.slice.hash`
- Create: `/carter/users/dlaub/svar_bench/make_slices.sh`

**Interfaces:**
- Produces: two small BCFs (~first 10 Mb of chr21) and their byte-identical store-hash oracles, used by every later task for a ~10–30s germline / ~1–2min gdc measurement loop. Full-chr21 files and their oracles remain the correctness gate.

- [ ] **Step 1: Write the slice-builder script**

```bash
cat > /carter/users/dlaub/svar_bench/make_slices.sh <<'EOF'
#!/bin/bash
# Carve a ~10 Mb chr21 slice from each filtered BCF for the fast dev-profile loop.
set -euo pipefail
BENCH=/carter/users/dlaub/svar_bench
REGION=chr21:1-10000000
cd "$BENCH"
for base in chr21 gdc.chr21; do
  bcftools view "$base.filt.bcf" "$REGION" -Ob -o "$base.slice.bcf"
  bcftools index --csi "$base.slice.bcf"
  echo "$base.slice.bcf: $(bcftools index -n "$base.slice.bcf") variants"
done
EOF
chmod +x /carter/users/dlaub/svar_bench/make_slices.sh
```

- [ ] **Step 2: Build the slices (on a compute node)**

Run: `bash /carter/users/dlaub/svar_bench/make_slices.sh`
Expected: two `*.slice.bcf` files created, each printing a non-zero variant count (germline ~a few ×10k, gdc larger). If the region label must be unprefixed (`21:1-10000000`), adjust `REGION` — confirm with `bcftools index -s chr21.filt.bcf`.

- [ ] **Step 3: Generate slice oracle hashes with the CURRENT (pre-change) build**

Ensure the current released build is installed (`pixi run maturin develop --release` from the worktree), then:

Run:
```bash
cd /carter/users/dlaub/svar_bench
pixi run --manifest-path <worktree>/pyproject.toml python run_svar2.py chr21.slice.bcf stores/chr21.slice.base $REF 32
bash storehash.sh stores/chr21.slice.base | tee oracle.chr21.slice.hash
pixi run --manifest-path <worktree>/pyproject.toml python run_svar2.py gdc.chr21.slice.bcf stores/gdc.slice.base $REF 32
bash storehash.sh stores/gdc.slice.base | tee oracle.gdc.slice.hash
```
Expected: two 64-hex-char hashes written. These are the slice correctness oracles for later tasks.

- [ ] **Step 4: Commit the script (data files stay out of the repo)**

```bash
cd <worktree>
git add -N docs  # no repo files change here; slices live under svar_bench, not the repo
echo "Slices + oracles live in /carter/users/dlaub/svar_bench (not version-controlled)."
```
Expected: nothing to commit in-repo; harness artifacts recorded under `svar_bench`. (Task 1 is infrastructure — its deliverable is the slice files + oracle hashes on disk, verified by Step 3 printing hashes.)

---

## Task 2: Enable libdeflate (Stage 0)

**Files:**
- Modify: `Cargo.toml:26` (the `rust-htslib` dependency line) and the comment block at `Cargo.toml:23-25`
- Modify: `CHANGELOG.md` (add an entry under `## Unreleased`)

**Interfaces:**
- Consumes: slice oracles from Task 1; full-chr21 oracles (Global Constraints).
- Produces: a build whose vendored htslib decompresses with libdeflate (verified by symbol inspection), byte-identical store output, and a recorded before/after timing.

- [ ] **Step 1: Capture the pre-change baseline timing (slice, fast)**

Run (compute node, current build):
```bash
cd /carter/users/dlaub/svar_bench
pixi run --manifest-path <worktree>/pyproject.toml python run_svar2.py gdc.chr21.slice.bcf stores/gdc.slice.pre $REF 32 | grep from_vcf
```
Expected: a `SVAR2 from_vcf: <t>s` line. Record `t` as the Stage-0 baseline.

- [ ] **Step 2: Turn on the libdeflate feature**

Edit `Cargo.toml`. Change:
```toml
rust-htslib = { version = "1.0", default-features = false, optional = true }
```
to:
```toml
# libdeflate: hts-sys builds vendored htslib against libdeflate (SIMD CRC + ~2x
# faster inflate) instead of zlib — the BGZF decompression bottleneck for
# VCF->SVAR2 conversion. See docs/superpowers/specs/2026-07-13-svar2-vcf-read-*.
rust-htslib = { version = "1.0", default-features = false, features = ["libdeflate"], optional = true }
```
Keep the existing `default-features = false` comment above it intact.

- [ ] **Step 3: Rebuild and verify libdeflate is actually linked**

Run:
```bash
cd <worktree>
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run maturin develop --release 2>&1 | tail -3
# Inspect the installed extension module for libdeflate symbols:
SO=$(python -c "import genoray, glob, os; print(glob.glob(os.path.join(os.path.dirname(genoray.__file__), '*.so'))[0])")
nm -D --defined-only "$SO" 2>/dev/null | grep -i libdeflate | head || nm "$SO" 2>/dev/null | grep -i libdeflate | head
```
Expected: at least one `libdeflate_*` symbol present (e.g. `libdeflate_deflate_decompress`, `libdeflate_crc32`). If empty, libdeflate did NOT link — investigate hts-sys build.rs / the `libdeflate-sys` build before proceeding (do not continue on an unverified link).

- [ ] **Step 4: Correctness gate — Rust + Python suites**

Run:
```bash
cd <worktree>
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
cargo test --no-default-features --features conversion 2>&1 | tail -5
pixi run pytest tests -m "not network" 2>&1 | tail -5
```
Expected: all Rust tests pass (~185), all pytest pass (~525). Zero failures.

- [ ] **Step 5: Correctness gate — byte-identical store hash (slice, then FULL)**

Run:
```bash
cd /carter/users/dlaub/svar_bench
# slice first (fast)
pixi run --manifest-path <worktree>/pyproject.toml python run_svar2.py gdc.chr21.slice.bcf stores/gdc.slice.ld $REF 32 | grep from_vcf
[ "$(bash storehash.sh stores/gdc.slice.ld)" = "$(cat oracle.gdc.slice.hash)" ] && echo "SLICE OK" || echo "SLICE MISMATCH"
# full germline + gdc (the real gate)
pixi run --manifest-path <worktree>/pyproject.toml python run_svar2.py chr21.filt.bcf stores/chr21.ld $REF 32 | grep from_vcf
[ "$(bash storehash.sh stores/chr21.ld)" = "$(cat oracle.chr21.hash)" ] && echo "GERMLINE OK" || echo "GERMLINE MISMATCH"
pixi run --manifest-path <worktree>/pyproject.toml python run_svar2.py gdc.chr21.filt.bcf stores/gdc.ld $REF 32 | grep from_vcf
[ "$(bash storehash.sh stores/gdc.ld)" = "$(cat oracle.gdc.hash)" ] && echo "GDC OK" || echo "GDC MISMATCH"
```
Expected: `SLICE OK`, `GERMLINE OK`, `GDC OK`. The `from_vcf` timings on the full files are the Stage-0 after-numbers — compare against the prior pass (germline 36.5s, gdc 1076s). Any `MISMATCH` is a hard stop.

- [ ] **Step 6: (Optional but recommended) confirm the symbol shift in a profile**

Run a `perf record`/`report` on the gdc slice (per the profiling protocol) and confirm `inflate_fast`/`crc32_z` are gone, replaced by `libdeflate_*`. This is the mechanistic proof the change did what the design claims.

- [ ] **Step 7: CHANGELOG + commit**

Add under `## Unreleased` in `CHANGELOG.md`:
```markdown
- perf(svar2): build vendored htslib against libdeflate (faster BGZF decompression) for VCF→SVAR2 conversion
```
Then:
```bash
cd <worktree>
git add Cargo.toml Cargo.lock CHANGELOG.md
git commit -m "perf(svar2): link vendored htslib against libdeflate

Switches the BGZF decompressor from zlib to libdeflate for the VCF->SVAR2
read path. Byte-identical store output on germline + gdc chr21.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Expected: commit created; `Cargo.lock` may gain `libdeflate-sys` — include it.

---

## Task 3: Ensure libdeflate ships in the release wheels

**Files:**
- Modify: `ci/wheel/pixi.toml` (add `libdeflate` if the vendored build needs a system copy on any of the four wheel platforms)
- Reference: `.github/workflows/release.yaml` (auditwheel/delocate repair step)

**Interfaces:**
- Consumes: the libdeflate-enabled build from Task 2.
- Produces: a release-wheel build path that vendors/repairs libdeflate into the wheel on linux-64, osx-arm64, and the two additional wheel platforms declared in `ci/wheel/pixi.toml`.

- [ ] **Step 1: Determine how libdeflate reaches the wheel**

Inspect whether `hts-sys`'s `libdeflate` feature pulls a **statically-linked** `libdeflate-sys` (self-contained, no wheel repair needed) or expects a system `libdeflate.so` (must be added to `ci/wheel/pixi.toml` and repaired in).

Run:
```bash
cd <worktree>
find ~/.cargo -path '*libdeflate-sys*/Cargo.toml' | head -1 | xargs grep -n -i "static\|build\|links" | head
grep -n -i "deflate\|auditwheel\|delocate\|repair" ci/wheel/pixi.toml .github/workflows/release.yaml
```
Expected: a clear read on static-vs-dynamic. If `libdeflate-sys` builds/statics its own copy, no manifest change is needed — record that and skip to Step 3.

- [ ] **Step 2: If dynamic, add libdeflate to the wheel manifest**

If Step 1 shows a system dependency, add `libdeflate` to the `[dependencies]` (or the platform tables) of `ci/wheel/pixi.toml` so all four platforms provide it at build time, and confirm the repair step (auditwheel `--exclude`/bundle, delocate) captures it. Keep the toolchain pins (`rust`, `clangdev=18`) in sync between `ci/wheel/pixi.toml` and the main `pixi.toml` per the existing CI note.

- [ ] **Step 3: Validate a wheel build locally (linux-64)**

Run:
```bash
cd <worktree>
pixi run --manifest-path ci/wheel/pixi.toml maturin build --release --features abi3 2>&1 | tail -5
# repaired wheel should import and convert:
python -m pip install --force-reinstall dist/*.whl 2>&1 | tail -2 || true
```
Expected: wheel builds; a smoke import + a tiny `from_vcf` on the slice succeeds against the installed wheel. (Full four-platform validation happens in release CI; local checks linux-64.)

- [ ] **Step 4: Commit**

```bash
cd <worktree>
git add ci/wheel/pixi.toml
git commit -m "build(ci): ensure libdeflate is vendored into release wheels

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Expected: commit created (or, if Step 1 found static linking, a no-op documented in the task notes with no commit).

---

## Task 4: Re-sweep htslib decode threads (Stage 1) + Stage-2 decision gate

**Files:**
- Modify: `src/budget.rs:13` (`MAX_HTSLIB_THREADS`) — experiment only; keep the winning value behind `plan_thread_budget`.
- Reference: `src/budget.rs` budget tests, `src/lib.rs:152-188`, `src/vcf_reader.rs:139-161`.

**Interfaces:**
- Consumes: the libdeflate build (Task 2).
- Produces: a recorded thread-scaling curve for gdc after libdeflate, an updated `MAX_HTSLIB_THREADS` if it scales, and an explicit **go/no-go decision for Task 5 (Stage 2)**.

- [ ] **Step 1: Sweep htslib_threads on the gdc slice**

Temporarily raise `MAX_HTSLIB_THREADS` in `src/budget.rs` and rebuild, then measure the gdc slice at a few effective thread counts (the sweep is driven by the `threads` arg to `run_svar2.py` feeding `plan_thread_budget`). Record wall-time at 4 / 8 / 12 / 16.

Run (per trial, after each `MAX_HTSLIB_THREADS` edit + `maturin develop --release`):
```bash
cd /carter/users/dlaub/svar_bench
pixi run --manifest-path <worktree>/pyproject.toml python run_svar2.py gdc.chr21.slice.bcf stores/gdc.slice.t $REF 32 | grep from_vcf
```
Expected: a monotonic (or plateauing) time-vs-threads table. libdeflate makes per-block inflate cheap, so this may now scale where it was previously inert.

- [ ] **Step 2: Confirm the best full-gdc number and byte-identical output**

With the best `MAX_HTSLIB_THREADS` from Step 1, run the FULL gdc + germline and re-verify the oracle hashes (as in Task 2 Step 5). Record the full-file timing.
Expected: `GERMLINE OK`, `GDC OK`; a full-gdc wall-time to compare against Task 2's Stage-0 number.

- [ ] **Step 3: Encode the winning cap in the budget logic + tests**

If a higher cap wins, set `MAX_HTSLIB_THREADS` to it and update/extend the `budget.rs` tests (mirror the existing low-end/high-end/clamp cases) so the multi-contig path is unchanged and only spare cores feed extra decode threads. Run:
```bash
cd <worktree>; export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
cargo test --no-default-features --features conversion budget 2>&1 | tail -5
```
Expected: budget tests pass. If no cap change wins, revert `budget.rs` and note it.

- [ ] **Step 4: Commit (if changed) + record the Stage-2 decision**

If the cap changed:
```bash
cd <worktree>
git add src/budget.rs && git commit -m "perf(svar2): raise htslib decode-thread cap now that inflate is cheap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Then write the **decision** into the plan's task notes / commit message:
- **NO-GO (stop here):** if libdeflate + thread re-sweep uses the idle cores and the next win (Stage 2) is small relative to its risk/effort → per the diminishing-returns stopping rule, the effort is complete. Task 5 is not executed.
- **GO:** if htslib `bgzf_mt` still fails to scale and the profile still shows decompression dominating with idle cores → proceed to Task 5, which warrants its own detailed plan (see Task 5).

---

## Task 5: Parallel BGZF frontend (Stage 2 — CONDITIONAL, gated on Task 4 GO)

> **Do not start unless Task 4 recorded a GO.** This task is a de-risking spike, not a specified implementation — the spec deliberately leaves the htslib-feed mechanism unresolved. Its deliverable is a measured go/no-go on the plumbing, after which it MUST spawn its own detailed plan (re-invoke superpowers:writing-plans) before any production implementation.

**Files:**
- Prototype (throwaway, outside the crate or behind a `#[cfg(feature = "bgzf_spike")]` gate): a standalone Rust binary/example that reads a BGZF BCF, decompresses blocks in parallel with `libdeflater`, and reassembles the ordered byte stream.
- Reference: `src/vcf_reader.rs` (`VcfChunkReader`, htslib handle ownership), PLINK2 `plink2_bgzf.cc` (GPLv3 — read for algorithm, copy nothing), noodles `bgzf::MultithreadedReader` (MIT — may depend on).

**Interfaces:**
- Consumes: the GO decision + profile from Task 4.
- Produces: a measured answer to "can we feed htslib's BCF parser decompressed bytes while decompressing in parallel, byte-identically, faster than Stage 1?" — and, if yes, a follow-on plan.

- [ ] **Step 1: Spike the decompression correctness**

Build a throwaway that parallel-decompresses `gdc.chr21.slice.bcf`'s BGZF blocks via `libdeflater` and concatenates them in order; assert the result is byte-identical to htslib's own decompression (`bgzip -d` / a single-thread reference). Include the short final block and the BGZF EOF marker.
Expected: byte-identical decompressed stream. If not, the parser-feed is unsafe — record NO-GO.

- [ ] **Step 2: Spike the htslib feed mechanism**

Prototype ONE feed path and measure it end-to-end vs Stage 1: (a) pipe the decompressed uncompressed-BCF stream into htslib via an `hFILE`/fd, or (b) a custom `hFILE` plugin. Verify htslib parses records identically (record count + a GT spot-check) and that inflate no longer appears in the profile.
Expected: a wall-time and a byte-identical record stream, or a documented reason the mechanism is unworkable.

- [ ] **Step 3: Decide and hand off**

Write the spike result into `docs/superpowers/specs/` (a short addendum) and:
- **If the spike wins and is byte-identical:** re-invoke superpowers:writing-plans to author the production implementation plan (thread-budget integration, `VcfChunkReader` restructure, proptest for cross-block decompression, wheel implications of `libdeflater`).
- **If it does not win or cannot be made byte-identical:** stop — Stage 0/1 stand as the delivered speedup, per diminishing returns.

- [ ] **Step 4: Commit the spike findings (not the throwaway prototype)**

```bash
cd <worktree>
git add docs/superpowers/specs/
git commit -m "docs(svar2): record parallel-BGZF frontend spike result

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Root cause (zlib→libdeflate) → Task 2. ✓
- Stage 0 (libdeflate) → Task 2; wheel/CI caveat → Task 3. ✓
- Stage 1 (thread re-sweep) → Task 4. ✓
- Stage 2 (parallel BGZF frontend, gated) → Task 5, explicitly conditional on Task 4 GO. ✓
- Harness slice + full-file validation → Task 1 + reused throughout. ✓
- Byte-identical gate, cargo/pytest gates → Global Constraints + every task's gate steps. ✓
- Diminishing-returns stopping rule → encoded as the Task 4 decision gate and Task 5 hand-off. ✓
- GPL boundary, CHANGELOG, no-API-change → Global Constraints. ✓

**Placeholder scan:** Stage 2 is intentionally a gated spike (the spec leaves its mechanism open); it is scoped as "measure then spawn its own plan," not fabricated code — this is honest gating, not a placeholder. All executable steps carry exact commands. `<worktree>` is a substitution the executor fills with the current worktree path.

**Type consistency:** No new cross-task types/signatures introduced (config + build + experiment tasks). `MAX_HTSLIB_THREADS`, `plan_thread_budget`, `VcfChunkReader` names match the current source.

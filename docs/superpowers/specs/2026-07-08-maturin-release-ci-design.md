# maturin-based release CI for the genoray Rust extension

**Date:** 2026-07-08
**Status:** Approved design

## Problem & goal

`genoray` now ships a compiled Rust extension module (`genoray._core`) built by
maturin. The extension vendors htslib (via `hts-sys`, gated behind the default
`conversion` feature) and requires a version-pinned build toolchain:
`cxx-compiler`, `zlib`, and `libclang 18` (Cargo.toml notes that libclang 22
breaks `rust-htslib 1.0`'s bindgen). That toolchain is currently pinned only in
the project's `pixi.toml`.

The existing `release.yaml` `publish` job is a leftover from the pure-Python era:
it runs `uv build` on a single `ubuntu-latest` runner. For a Rust extension this
produces at best a non-portable, Linux-only wheel. We need release CI that builds
portable binary wheels across the target platforms plus an sdist, and publishes
them via the existing OIDC trusted-publishing flow.

**Wheel targets:** `linux-64`, `linux-aarch64`, `osx-arm64`, `osx-64`.

## Decisions (settled)

1. **Build inside pixi**, not in manylinux/cibuildwheel containers — reuse the
   proven, version-pinned toolchain instead of re-solving libclang 18 / htslib
   inside a container.
2. **Adopt pyo3 abi3 (`abi3-py310`)** — one wheel per platform covers Python
   3.10–3.13, collapsing the matrix from 16 (4 platforms × 4 versions) to 4.
3. **Cover all four platforms**, including `linux-aarch64` and `osx-64` (Intel
   mac), beyond the `linux-64` / `osx-arm64` that the main workspace declares.
4. **Isolated build manifest** at `ci/wheel/pixi.toml` — do NOT add the two new
   platforms to the main `pixi.toml`.

## Build model — isolated pixi build manifest

Adding `linux-aarch64` and `osx-64` to the main `pixi.toml` `platforms` would
force the entire default/test workspace (`oxbow`, `polars-bio`, `plink2`, and the
rest) to re-solve on those platforms, likely making the lock unsolvable. Instead,
create a dedicated minimal manifest:

**`ci/wheel/pixi.toml`** — declares all four platforms and only the build
toolchain. maturin builds the repo-root crate; this environment supplies tools
only, so it is decoupled from the main workspace and its lock file.

Contents (minimal build environment):

- `platforms = ["linux-64", "linux-aarch64", "osx-arm64", "osx-64"]`
- `rust` (pinned to match the main workspace: `>=1.93.1,<1.94`)
- `maturin >=1.12,<2`
- `cxx-compiler`
- `clangdev = "18.*"`  (with `LIBCLANG_PATH = "$CONDA_PREFIX/lib"` in
  `[activation.env]`)
- `zlib`
- `python` (>= 3.10, for the abi3 build interpreter)
- Repair tooling, platform-gated: `auditwheel` + `patchelf` on Linux,
  `delocate` on macOS.

The ~6 duplicated build-toolchain pins are a deliberate trade to keep the dev
workspace at two platforms and its lock solvable. If a pin ever drifts, both
manifests must be updated together.

## abi3 feature — opt-in only at wheel-build time

Two existing cargo entry points must keep working untouched:

- The lint hooks run `cargo check/clippy --all-targets` with **default** features
  (which include `extension-module`).
- `test-rust` runs `cargo test --no-default-features --features conversion`
  (drops `extension-module` so the libpython-linking test binary can link the
  `auto-initialize` dev-dependency).

To disturb neither, define an `abi3` feature that is **not** in `default`, and
enable it only when building the shipped wheel via `maturin build --features
abi3`:

```toml
[features]
abi3 = ["pyo3/abi3-py310"]
```

- `maturin build --features abi3` → keeps default features (`conversion` +
  `extension-module`) and adds abi3 → single `cp310-abi3-<platform>` wheel
  covering 3.10–3.13. maturin auto-tags the wheel from the enabled abi3 feature.
- Lint hooks and `test-rust` → unchanged (no abi3).
- Local `maturin develop` for dev → unchanged (full, version-specific API).

This also gives a cheap local compatibility gate: `cargo check --features abi3`
compiles the whole tree (including the `numpy` crate) under the limited API, so it
fails fast if rust-numpy 0.29 or any pyo3 usage is abi3-incompatible.

No public Python API name changes, so `skills/genoray-api/SKILL.md` does not need
an update for this change.

## Wheel repair — portable tags

`maturin build --release` in the conda toolchain links conda's `libgcc_s`,
`libz`, etc. Repair makes each wheel self-contained:

- **Linux:** `auditwheel repair` bundles the conda `.so`s, rewrites RPATH, and
  auto-assigns the minimal `manylinux_*` tag from the conda sysroot's glibc floor.
- **macOS:** `delocate-wheel` bundles dylibs.

Statically vendored htslib lives inside the `.so` and is untouched by repair.

## Release workflow restructure

`release.yaml` jobs:

1. **`bump`** *(unchanged)* — commitizen version bump, changelog, GitHub release,
   tag; outputs `version`.
2. **`build-wheels`** *(new)* — matrix over four runners, `fail-fast: false`:
   - `linux-64` → `ubuntu-latest`
   - `linux-aarch64` → `ubuntu-24.04-arm` (native ARM runner)
   - `osx-arm64` → `macos-14`
   - `osx-64` → `macos-13`

   Each job: checkout the tag (`ref: ${{ needs.bump.outputs.version }}`) →
   `setup-pixi` against `ci/wheel` → `maturin build --release --features abi3` →
   repair (auditwheel/delocate) → upload the repaired wheel as an artifact.
3. **`sdist`** *(new)* — `maturin sdist` once on Linux → upload artifact. This is
   the fallback for uncovered platforms and source installs.
4. **`publish`** *(reworked)* — `needs: [build-wheels, sdist]`. Keeps the `pypi`
   environment and `id-token: write` OIDC trusted publishing. Downloads all
   artifacts into `dist/`, then `uv publish dist/*`.
5. **`merge`** *(unchanged)* — main → stable.

## Validation gates (the real risks)

These are checked before committing to the design's happy path:

1. **abi3 + rust-numpy 0.29 + `multiple-pymethods` compiles and imports.**
   Primary risk. First implementation step is a local smoke build:
   `pixi run maturin build` with the abi3 feature, then `python -c "import
   genoray; import genoray._core"`. **If it fails, fall back** to a per-version
   matrix (cp310–cp313); the rest of the design (isolated manifest, repair,
   workflow shape) is unchanged, only the matrix widens.
2. **`clangdev=18` resolves on `linux-aarch64` and `osx-64`** in the `ci/wheel`
   environment (conda-forge ships these; validated when the lock is generated).
3. **auditwheel accepts the conda-linked wheel** and emits a sane `manylinux`
   tag rather than rejecting an out-of-policy library.

## Out of scope

- musllinux wheels
- Windows wheels
- Publishing debug/profiling builds
- Changes to `test.yaml` (the existing Rust + pytest CI already works)

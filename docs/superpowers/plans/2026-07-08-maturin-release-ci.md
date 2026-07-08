# maturin Release CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pure-Python `uv build` publish step with maturin-built, repaired, portable binary wheels for the Rust extension across linux-64/linux-aarch64/osx-arm64/osx-64, plus an sdist, published via the existing OIDC trusted-publishing flow.

**Architecture:** Wheels build *inside pixi* to reuse the version-pinned htslib/libclang-18/zlib toolchain, but from an **isolated** `ci/wheel/pixi.toml` manifest declaring the four wheel platforms — so the main workspace's lock stays at two platforms and solvable. An `abi3` cargo feature (opt-in only at wheel-build time) collapses the per-Python-version matrix to one wheel per platform. `auditwheel` (Linux) / `delocate` (macOS) repair the conda-linked wheels into portable, correctly-tagged distributions.

**Tech Stack:** maturin, pyo3 abi3, pixi, GitHub Actions, auditwheel/patchelf, delocate, uv (publish).

## Global Constraints

- Rust toolchain pin (must match main workspace): `rust = ">=1.93.1,<1.94"`.
- maturin: `>=1.12,<2` (installed maturin is 1.14).
- libclang **must** be `clangdev = "18.*"` — libclang 22 breaks `rust-htslib 1.0` bindgen. Set `LIBCLANG_PATH = "$CONDA_PREFIX/lib"`.
- htslib is vendored by `hts-sys` and needs `cxx-compiler` + `zlib` at build time.
- `abi3` must **not** be in cargo `default` features. It is enabled only via `maturin build --features abi3`. Rationale: lint hooks run `cargo check/clippy --all-targets` with default features; `test-rust` runs `cargo test --no-default-features --features conversion`. Both must stay unchanged.
- abi3 floor is Python 3.10 (`pyo3/abi3-py310`); pyproject `requires-python = ">=3.10,<3.14"` already matches.
- Wheel platforms: `linux-64`, `linux-aarch64`, `osx-arm64`, `osx-64`. Runners: `ubuntu-latest`, `ubuntu-24.04-arm`, `macos-14`, `macos-13`. (`ubuntu-24.04-arm` is free on public repos; genoray is public.)
- Reuse the existing pinned action SHAs where a job is unchanged. `setup-pixi` uses `@v0.9.5` to match `test.yaml`.
- Commit messages follow Conventional Commits (`ci:`, `chore:` etc.).
- No public Python API names change → no `skills/genoray-api/SKILL.md` update required.

---

### Task 1: Add the `abi3` cargo feature and prove abi3/rust-numpy compatibility

This is the **primary risk gate**. If Step 3 or Step 4 fails, abi3 is not viable with rust-numpy 0.29 — STOP and see "Fallback" at the end of this task.

**Files:**
- Modify: `Cargo.toml` (the `[features]` table)

**Interfaces:**
- Produces: cargo feature `abi3 = ["pyo3/abi3-py310"]`, enabled downstream via `maturin build --features abi3`.

- [ ] **Step 1: Baseline sanity — current cargo lint passes**

Run: `pixi run -e lint cargo check --all-targets`
Expected: finishes with `Finished` (no errors). Confirms the toolchain works before we touch features.

- [ ] **Step 2: Add the `abi3` feature to `Cargo.toml`**

In the `[features]` table (currently):

```toml
[features]
default = ["conversion", "extension-module"]
conversion = ["dep:rust-htslib"]
extension-module = ["pyo3/extension-module"]
```

Add one line so it becomes:

```toml
[features]
default = ["conversion", "extension-module"]
conversion = ["dep:rust-htslib"]
extension-module = ["pyo3/extension-module"]
# abi3 (stable ABI) is opt-in ONLY at wheel-build time (`maturin build --features
# abi3`) so lint hooks (default features) and `cargo test --no-default-features`
# stay unaffected. abi3-py310 => one cpXY-abi3 wheel covers Python 3.10-3.13.
abi3 = ["pyo3/abi3-py310"]
```

- [ ] **Step 3: Compile-compat gate — cargo check under abi3**

Run: `pixi run -e lint cargo check --all-targets --features abi3`
Expected: `Finished` with no errors. This compiles the whole tree — including the `numpy` (rust-numpy 0.29) crate and all pyo3 usage — under the limited API. A failure here means abi3 is incompatible (see Fallback).

- [ ] **Step 4: Link+import gate — build and import the abi3 extension**

Run:
```bash
pixi run maturin develop --release --features abi3
pixi run python -c "import genoray; import genoray._core; print('abi3 import ok')"
```
Expected: prints `abi3 import ok`. Proves the abi3 extension links and imports in a real environment.

- [ ] **Step 5: Confirm the Rust test path is untouched**

Run: `pixi run -e lint test-rust`
Expected: the Rust unit tests pass (this path uses `--no-default-features`, so abi3 is absent — it must behave exactly as before).

- [ ] **Step 6: Restore the normal (non-abi3) editable dev build**

Run: `pixi run maturin develop`
Expected: rebuilds the editable extension without abi3 for normal local development. (Purely hygiene; the shipped wheel is built fresh in CI.)

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml
git commit -m "feat(build): add opt-in abi3 cargo feature for release wheels"
```

**Fallback (only if Step 3 or Step 4 fails):** abi3 is not viable. Remove the `abi3` feature line. The rest of the plan still holds, but the build matrix must build per Python version instead of one abi3 wheel: in `ci/wheel/pixi.toml` drop the `--features abi3` from the `build` task, add a `[feature]`-based set of Python envs (`py310`..`py313`), and in `release.yaml` add a nested `python: [3.10, 3.11, 3.12, 3.13]` matrix dimension that selects the interpreter via `maturin build -i pythonX.Y`. Pause and flag this to the user before proceeding, since it quadruples build jobs.

---

### Task 2: Create the isolated `ci/wheel/pixi.toml` build manifest and validate a repaired linux wheel

**Files:**
- Create: `ci/wheel/pixi.toml`
- (Generated) `ci/wheel/pixi.lock`

**Interfaces:**
- Consumes: the `abi3` cargo feature from Task 1; the repo-root `Cargo.toml` (referenced as `../../Cargo.toml`).
- Produces: pixi tasks `build`, `sdist`, and per-platform `repair`; output wheels in repo-root `wheelhouse/` (raw) and `dist/` (repaired).

- [ ] **Step 1: Create `ci/wheel/pixi.toml`**

```toml
[workspace]
name = "genoray-wheel-build"
channels = ["conda-forge"]
platforms = ["linux-64", "linux-aarch64", "osx-arm64", "osx-64"]

[dependencies]
# Rust toolchain pinned to match ../../pixi.toml.
rust = ">=1.93.1,<1.94"
maturin = ">=1.12,<2"
# htslib (vendored by hts-sys) build deps: C/C++ compiler builds vendored htslib,
# clangdev provides libclang for rust-htslib's mandatory bindgen (18 is required;
# 22 breaks rust-htslib 1.0), zlib provides zlib.h.
cxx-compiler = "*"
clangdev = "18.*"
zlib = "*"
# abi3 build interpreter: any >=3.10 satisfies abi3-py310; the wheel is tagged cp310.
python = ">=3.10"

[activation.env]
# rust-htslib's bindgen step needs to locate libclang (provided by clangdev).
LIBCLANG_PATH = "$CONDA_PREFIX/lib"

[tasks]
# Tasks run with cwd = this manifest's directory, so ../../ is the repo root.
build = "maturin build --release --features abi3 -m ../../Cargo.toml -o ../../wheelhouse"
sdist = "maturin sdist -m ../../Cargo.toml -o ../../dist"

# Wheel repair is platform-specific: auditwheel (Linux) bundles conda .so deps and
# assigns a manylinux tag; delocate does the equivalent on macOS.
[target.linux-64.dependencies]
auditwheel = "*"
patchelf = "*"
[target.linux-aarch64.dependencies]
auditwheel = "*"
patchelf = "*"
[target.osx-arm64.dependencies]
delocate = "*"
[target.osx-64.dependencies]
delocate = "*"

[target.linux-64.tasks]
repair = "auditwheel repair ../../wheelhouse/*.whl -w ../../dist"
[target.linux-aarch64.tasks]
repair = "auditwheel repair ../../wheelhouse/*.whl -w ../../dist"
[target.osx-arm64.tasks]
repair = "delocate-wheel -w ../../dist -v ../../wheelhouse/*.whl"
[target.osx-64.tasks]
repair = "delocate-wheel -w ../../dist -v ../../wheelhouse/*.whl"
```

- [ ] **Step 2: Generate the lock and confirm all four platforms resolve**

Run: `pixi lock --manifest-path ci/wheel/pixi.toml`
Then: `pixi run --manifest-path ci/wheel/pixi.toml pixi info` is not needed; instead confirm the four platforms are present in the lock:
Run: `grep -c -E "linux-64|linux-aarch64|osx-arm64|osx-64" ci/wheel/pixi.lock`
Expected: `pixi lock` completes without a solve error (this proves `clangdev=18` and `rust` resolve on `linux-aarch64` and `osx-64`), and the grep returns a non-zero count. A solve failure here means a build dep is unavailable on one of the new platforms — report which package/platform.

- [ ] **Step 3: Build a real wheel on this (linux-64) runner**

Run: `pixi run --manifest-path ci/wheel/pixi.toml build`
Expected: `wheelhouse/genoray-<version>-cp310-abi3-linux_x86_64.whl` is produced (raw, un-repaired `linux_x86_64` tag at this stage).

- [ ] **Step 4: Confirm the abi3 extension is inside the wheel**

Run: `unzip -l wheelhouse/*.whl | grep _core`
Expected: a line containing `genoray/_core.abi3.so` (the `.abi3.so` suffix confirms the stable-ABI build).

- [ ] **Step 5: Repair the wheel and confirm a manylinux tag**

Run:
```bash
pixi run --manifest-path ci/wheel/pixi.toml repair
ls dist/
```
Expected: `dist/` contains `genoray-<version>-cp310-abi3-manylinux_*_x86_64.whl` (auditwheel rewrote the tag from `linux_x86_64` to `manylinux_*`). If auditwheel *errors* that a library is outside policy rather than bundling it, capture its message — that is validation gate #3 from the spec failing.

- [ ] **Step 6: Ignore build output dirs in git**

Add to `.gitignore` (if not already covered):
```
/wheelhouse/
```
`/dist/` is already tracked with its own `.gitignore` (keep as-is); verify `wheelhouse/` is ignored:
Run: `git check-ignore wheelhouse/ && echo ignored`
Expected: prints `ignored`.

- [ ] **Step 7: Commit**

```bash
git add ci/wheel/pixi.toml ci/wheel/pixi.lock .gitignore
git commit -m "ci: add isolated pixi manifest for building release wheels"
```

---

### Task 3: Rewrite `release.yaml` to build/repair wheels + sdist and publish them

**Files:**
- Modify (full rewrite): `.github/workflows/release.yaml`

**Interfaces:**
- Consumes: the `build`, `sdist`, and `repair` pixi tasks from Task 2; the `bump` job's `version` output.
- Produces: artifacts `wheel-<platform>` and `sdist`, merged into `dist/` and published with `uv publish`.

- [ ] **Step 1: Replace `.github/workflows/release.yaml` with the full workflow below**

```yaml
name: Release

on:
  workflow_dispatch:

jobs:
  bump:
    name: Bump version and create release
    runs-on: ubuntu-latest
    permissions:
      contents: write
    outputs:
      version: ${{ steps.cz.outputs.version }}
    steps:
      - name: Check out
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          fetch-depth: 0
          token: ${{ secrets.GH_ACTIONS }}
      - name: Create bump and changelog
        id: cz
        uses: commitizen-tools/commitizen-action@338bbd841b75aaee6bf5340e1fa12f6ab58ff9ff # 0.27.1
        with:
          github_token: ${{ secrets.GH_ACTIONS }}
          branch: main
          changelog_increment_filename: body.md
      - name: Release
        uses: softprops/action-gh-release@b4309332981a82ec1c5618f44dd2e27cc8bfbfda # v3.0.0
        with:
          body_path: body.md
          tag_name: ${{ steps.cz.outputs.version }}
          token: ${{ secrets.GH_ACTIONS }}

  build-wheels:
    name: Build wheel (${{ matrix.platform }})
    needs: bump
    runs-on: ${{ matrix.runner }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - runner: ubuntu-latest
            platform: linux-64
          - runner: ubuntu-24.04-arm
            platform: linux-aarch64
          - runner: macos-14
            platform: osx-arm64
          - runner: macos-13
            platform: osx-64
    steps:
      - name: Check out release tag
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          ref: ${{ needs.bump.outputs.version }}
      - name: Setup pixi (wheel build env)
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.70.2
          manifest-path: ci/wheel/pixi.toml
      - name: Build wheel
        run: pixi run --manifest-path ci/wheel/pixi.toml build
      - name: Repair wheel
        run: pixi run --manifest-path ci/wheel/pixi.toml repair
      - name: Upload wheel artifact
        uses: actions/upload-artifact@v4
        with:
          name: wheel-${{ matrix.platform }}
          path: dist/*.whl
          if-no-files-found: error

  sdist:
    name: Build sdist
    needs: bump
    runs-on: ubuntu-latest
    steps:
      - name: Check out release tag
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          ref: ${{ needs.bump.outputs.version }}
      - name: Setup pixi (wheel build env)
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.70.2
          manifest-path: ci/wheel/pixi.toml
      - name: Build sdist
        run: pixi run --manifest-path ci/wheel/pixi.toml sdist
      - name: Upload sdist artifact
        uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*.tar.gz
          if-no-files-found: error

  publish:
    name: Publish to PyPI
    needs: [bump, build-wheels, sdist]
    runs-on: ubuntu-latest
    environment:
      name: pypi
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: dist
          merge-multiple: true
      - name: Install uv
        uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0
      - name: Publish
        run: uv publish

  merge:
    name: Merge main -> stable
    needs: publish
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Check out
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          ref: stable
          fetch-depth: 0
          token: ${{ secrets.GH_ACTIONS }}
      - name: Config git
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
      - name: Merge main -> stable
        run: |
          git rebase origin/main
          git push origin stable
```

- [ ] **Step 2: Validate the workflow YAML parses**

Run: `pixi run python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yaml')); print('yaml ok')"`
Expected: prints `yaml ok`.

- [ ] **Step 3: Lint the workflow if actionlint is available (optional but preferred)**

Run: `command -v actionlint >/dev/null && actionlint .github/workflows/release.yaml || echo "actionlint not installed - skipping"`
Expected: either `actionlint` reports no errors, or the skip message. (Do not add actionlint as a project dep just for this.)

- [ ] **Step 4: Cross-check the workflow against the locally-proven commands**

Confirm by reading the diff that every `run:` command in `build-wheels`/`sdist` is exactly one of the pixi tasks proven in Task 2 (`build`, `repair`, `sdist`) invoked with `--manifest-path ci/wheel/pixi.toml`, and that `publish` downloads into `dist/` then runs `uv publish` (which defaults to publishing `dist/*`). No new untested commands.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/release.yaml
git commit -m "ci: build and publish maturin wheels + sdist across platforms"
```

---

### Task 4: Documentation — record the release/build workflow

**Files:**
- Modify: `CLAUDE.md` (Environment section) — add a short note on the isolated wheel-build manifest.

**Interfaces:**
- Consumes: nothing. Produces: nothing runtime.

- [ ] **Step 1: Add a note to `CLAUDE.md`**

Under the `## Environment` section, after the existing pixi commands block, add:

```markdown
### Release wheels

Release CI (`.github/workflows/release.yaml`) builds the Rust extension as
portable wheels from an **isolated** pixi manifest at `ci/wheel/pixi.toml`
(declares all four wheel platforms; the main `pixi.toml` stays at linux-64 +
osx-arm64). Wheels use pyo3 abi3 (`maturin build --features abi3`) → one
`cpXY-abi3` wheel per platform covers Python 3.10–3.13, then `auditwheel`
(Linux) / `delocate` (macOS) repair them. The `abi3` cargo feature is opt-in
only at wheel-build time so lint hooks and the Rust test suite are unaffected.
Toolchain pins (`rust`, `clangdev=18`) are duplicated between the two manifests
and must be kept in sync.
```

- [ ] **Step 2: Verify the note renders and paths are correct**

Run: `grep -n "ci/wheel/pixi.toml" CLAUDE.md`
Expected: at least one match in the new section.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document isolated maturin wheel-build workflow"
```

---

## Self-Review

**Spec coverage:**
- Isolated `ci/wheel/pixi.toml` build manifest → Task 2. ✓
- abi3 opt-in feature gating → Task 1. ✓
- Four platforms + runner mapping → Task 3 matrix. ✓
- auditwheel/delocate repair → Task 2 (local proof) + Task 3 (CI). ✓
- Release workflow restructure (bump/build-wheels/sdist/publish/merge) → Task 3. ✓
- OIDC trusted publishing retained (`environment: pypi`, `id-token: write`, `uv publish`) → Task 3. ✓
- Validation gates: abi3/rust-numpy (Task 1 Steps 3–4 + Fallback), clangdev-18 on new platforms (Task 2 Step 2), auditwheel tag (Task 2 Step 5). ✓
- Out of scope (musllinux, Windows, test.yaml) → not touched. ✓
- No public Python API change → no SKILL update (stated in Global Constraints). ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code/config step shows full content; the Fallback is a concrete, bounded procedure. ✓

**Type/name consistency:** pixi task names (`build`, `sdist`, `repair`) and artifact names (`wheel-<platform>`, `sdist`) are used identically in Tasks 2 and 3; `--features abi3` string is consistent across Cargo.toml, the `build` task, and the spec. ✓

# SVAR 2.0 — M3 Format Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finalize the SVAR2 on-disk format — rename the provisional `final_*` scratch files to their spec base names, settle the wire-format question (`.bin` vs `.npy`), emit a top-level `meta.json`, and reconcile the roadmap docs.

**Architecture:** A mechanical, write-side change. The tile merge already writes correct bytes; this only changes the *names* it writes them under (via the single-source-of-truth `layout.rs`), adds one new one-shot `meta.json` writer wired into the top-level pipeline, and updates the three roadmap docs to match. No read/query path, no pwrite refactor.

**Tech Stack:** Rust (pyo3 extension, tested via `cargo test`), `serde_json` (new dep, one-shot JSON writer only), pixi for the toolchain, prek for git hooks.

## Global Constraints

- **Worktree:** `.claude/worktrees/svar-2-m3-format-finalize`, branch `svar-2-m3-format-finalize` off `svar-2`, PR into `svar-2`. (Already created; prek hooks already installed.)
- **Rust test command (canonical):** `pixi run -e lint cargo test --no-default-features` — bare `cargo` fails (`stddef.h` not found) because it misses the pixi conda toolchain; the `lint` env provides clangdev/cxx-compiler and `--no-default-features` drops `extension-module` so libpython links for the test binary.
- **fmt/clippy gate (prek runs these on commit):** `pixi run -e lint cargo fmt --all` and `pixi run -e lint cargo clippy --all-targets -- -D warnings`.
- **New dependency:** `serde_json` only. No `serde` derive. Rust never reads `meta.json` back — only Python does.
- **`format_version` is the integer `1`.**
- **`meta.json` schema (exact keys):** `{ "format_version": 1, "samples": [...], "contigs": [...], "ploidy": 2 }`.
- **Renames apply identically** under `var_key/{snp,indel}/` and `dense/{snp,indel}/`.
- **Out of scope (do NOT touch):** `max_del.npy` (deferred to M5), `{field}.npy` INFO/FORMAT columns, the `pointer` representation (M11), the M5 search tree / query path. In `data-model.md`'s layout tree, the `pointer/` sub-blocks keep their `.npy` names.
- **Wire-format decision (the resolution of the open question):** bulk parallel-`pwrite`n data arrays are raw `.bin` (mmap-friendly, no npy-header friction); small one-shot index/metadata sidecars stay self-describing `.npy`. Concretely: `positions.bin` (u32 LE), `alleles.bin` (u8 packed SNP / u32 indel), `genotypes.bin` (u8 packed 1-bit matrix) are `.bin`; `offsets.npy` (u64), `long_allele_offsets.npy` (u64) stay `.npy`; `long_alleles.bin` (u8) stays `.bin`.

---

## File Structure

Files created or modified, by responsibility:

- `src/layout.rs` (modify) — single source of truth for on-disk paths. The four `final_*` free functions get renamed to their spec base names; this is where the filename strings live.
- `src/merge.rs` (modify) — ragged `var_key` tile merge; calls the renamed layout fns; has string-literal filename assertions in its `#[cfg(test)]` module. **Note:** also has an *in-memory local variable* named `final_offsets` (the offsets vec) that is NOT a filename and must be left alone.
- `src/dense_merge.rs` (modify) — rectangular dense merge; calls the renamed layout fns; its tests call the layout fns directly (they follow the rename automatically).
- `src/rvk.rs` (modify) — `pack_snp_key_file` streams the `var_key` SNP keys and refers to the merged key file by string literal (`final_keys.bin`) plus a `.packed.tmp` temp; must track the `alleles.bin` rename.
- `tests/test_e2e.rs` (modify) — end-to-end tests that assert on the on-disk filenames via string literals.
- `src/meta.rs` (**create**) — new module: `FORMAT_VERSION` const + `write_meta` one-shot JSON writer + its unit test.
- `src/lib.rs` (modify) — register `pub mod meta;` and call `write_meta` once in `run_conversion_pipeline` after all contigs succeed.
- `Cargo.toml` (modify) — add `serde_json = "1"`.
- `docs/roadmap/data-model.md` (modify) — flip the three filenames to `.bin` in the layout tree (var_key + dense only), resolve the "var_key sidecar wire format" open question, add the dtype-vs-`format_version` convention, drop the "provisional filenames" notes.
- `docs/roadmap/svar-2.md` (modify) — M3 `[~]` → `[x]` + rewrite; remove `max_del.npy` from M3's list; update M4's "unify dense filenames as part of M3" note to done.
- `docs/roadmap/architecture.md` (verify only) — confirm the `meta.json` / on-disk-layout bullets still read true; edit only if they don't.

---

## Task 1: Rename `final_*` on-disk files to spec base names

Rename the four provisional filenames everywhere they are produced or asserted, atomically (a fn rename breaks all callers' compilation, and a filename change breaks all on-disk assertions — so these land together in one commit).

| now | → | dtype |
| --- | --- | --- |
| `final_positions.bin` | `positions.bin` | u32 LE |
| `final_keys.bin` | `alleles.bin` | u8 (packed SNP) / u32 (indel) |
| `final_offsets.npy` | `offsets.npy` | u64 |
| `final_genotypes.bin` | `genotypes.bin` | u8 (packed 1-bit matrix) |

**Files:**
- Modify: `src/layout.rs`, `src/merge.rs`, `src/dense_merge.rs`, `src/rvk.rs`
- Test: `tests/test_e2e.rs`, plus the in-source `#[cfg(test)]` modules of `layout.rs`, `merge.rs`, `dense_merge.rs`

**Interfaces:**
- Produces (for later tasks / the read path): the four free functions in `layout.rs` are now
  - `pub fn positions(dir: &Path) -> PathBuf` → `dir/"positions.bin"`
  - `pub fn alleles(dir: &Path) -> PathBuf` → `dir/"alleles.bin"`
  - `pub fn offsets(dir: &Path) -> PathBuf` → `dir/"offsets.npy"`
  - `pub fn genotypes(dir: &Path) -> PathBuf` → `dir/"genotypes.bin"`
- Consumes: nothing from other tasks.

- [ ] **Step 1: Pre-edit the string-literal filename assertions to the NEW names (the "red" step)**

These assertions read files by string literal (not via the layout fns), so they can be flipped *before* the production rename to watch them fail. In **`tests/test_e2e.rs`**, replace every occurrence (use replace_all):
- `final_positions` → `positions`
- `final_keys` → `alleles`
- `final_offsets` → `offsets`
- `final_genotypes` → `genotypes`

(This also updates the two explanatory comments at the top of the file, e.g. `// → final_offsets = [0, 2, 4, 5, 6]`.)

In **`src/merge.rs`**, inside the `#[cfg(test)]` module only, flip the quoted filename literals — replace_all:
- `"final_positions.bin"` → `"positions.bin"`
- `"final_keys.bin"` → `"alleles.bin"`
- `"final_offsets.npy"` → `"offsets.npy"`

**Do NOT** rename the bare in-memory local variable `final_offsets` (the offsets `Vec<u64>` at `merge.rs:39` and its uses) — it is not a filename. Restricting to the quoted forms above leaves it untouched.

- [ ] **Step 2: Run the suite to confirm the assertions now FAIL**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: FAIL — the renamed reads (`.join("positions.bin")`, etc.) open files that production still writes as `final_positions.bin`, so `File::open(...).unwrap()` / the `read_*` helpers panic. (Production code was not touched yet, so this is a genuine red.)

- [ ] **Step 3: Rename the four functions and their output strings in `src/layout.rs`**

Apply replace_all in `src/layout.rs`:
- `final_positions` → `positions`
- `final_keys` → `alleles`
- `final_offsets` → `offsets`
- `final_genotypes` → `genotypes`

This renames the four `pub fn` definitions, their output string literals (`"final_positions.bin"` → `"positions.bin"`, `"final_keys.bin"` → `"alleles.bin"`, `"final_offsets.npy"` → `"offsets.npy"`, `"final_genotypes.bin"` → `"genotypes.bin"`), and the two in-source unit tests (`test_dense_chunk_and_final_names`, `test_chunk_and_final_names`) that call the fns and assert the path strings — all in one file.

Then update the now-stale module doc comment at the top of `src/layout.rs`. Replace:

```rust
//! Single source of truth for the SVAR2 on-disk directory + file layout. Every
//! path the pipeline reads or writes is constructed here so the (still
//! provisional) filenames can be changed in exactly one place before M6 decode.
```

with:

```rust
//! Single source of truth for the SVAR2 on-disk directory + file layout. Every
//! path the pipeline reads or writes is constructed here, so the finalized
//! on-disk file names are defined in exactly one place.
```

- [ ] **Step 4: Update the callers**

**`src/dense_merge.rs`** — apply replace_all (safe here; the local `positions`/`keys`/`final_key_bytes` vars do not contain these exact tokens):
- `final_positions` → `positions`
- `final_keys` → `alleles`
- `final_genotypes` → `genotypes`

(This updates the three `layout::final_*(dir)` calls at lines ~44/50/76 and the test assertions that call the layout fns.)

**`src/rvk.rs`** — apply replace_all:
- `final_keys` → `alleles`

(This updates the `pack_snp_key_file` string literals `format!("{}/final_keys.bin", dir)` → `.../alleles.bin` and `final_keys.packed.tmp` → `alleles.packed.tmp`, the `expect(...)` messages, and the doc comment.)

**`src/merge.rs`** — apply these replace_all edits (targeted, to protect the `final_offsets` local var):
- `final_positions` → `positions`   *(covers `layout::final_positions`, the `"...final_positions.bin"` `expect` messages, and the header doc comment — `final_positions` never names a local here)*
- `final_keys` → `alleles`   *(covers `layout::final_keys`, the `expect` messages, and the doc comment)*
- `layout::final_offsets` → `layout::offsets`   *(the fn call at line ~55 only)*
- `final_offsets.npy` → `offsets.npy`   *(covers the header doc comment `writes \`final_offsets.npy\``; the bare local `final_offsets` has no `.npy` suffix so it is untouched)*

- [ ] **Step 5: Run the full suite to confirm GREEN**

Run: `pixi run -e lint cargo test --no-default-features`
Expected: PASS — all ~94 tests (85 lib + e2e). Behavior is unchanged; only names moved.

- [ ] **Step 6: Verify the rename is complete and the only residual is the intentional local var**

Run: `grep -rn "final_positions\|final_keys\|final_offsets\|final_genotypes" src/ tests/`
Expected: matches ONLY the in-memory `final_offsets` local variable in `src/merge.rs` (declaration + arithmetic uses, e.g. `let mut final_offsets = vec![0u64; ...]`, `final_offsets[col + 1] = ...`, `final_offsets_ref`, `final_offsets[total_columns]`). No filename strings, no `layout::` calls, nothing in the other four files.

- [ ] **Step 7: Format, lint, commit**

Run:
```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
```
Expected: fmt makes no/whitespace-only changes; clippy clean.

Then commit:
```bash
git add src/layout.rs src/merge.rs src/dense_merge.rs src/rvk.rs tests/test_e2e.rs
git commit -m "feat(svar-2): rename final_* sidecars to spec base names (positions/alleles/offsets/genotypes)"
```

---

## Task 2: `write_meta` helper + `FORMAT_VERSION` in a new `meta.rs`

A one-shot writer for the top-level `meta.json`, extracted into its own module so it is unit-testable without going through the `#[pyfunction]`.

**Files:**
- Create: `src/meta.rs`
- Modify: `Cargo.toml` (add `serde_json = "1"`)
- Test: in-source `#[cfg(test)]` module in `src/meta.rs`

**Interfaces:**
- Produces (consumed by Task 3):
  - `pub const FORMAT_VERSION: u32 = 1;`
  - `pub fn write_meta(output_dir: &std::path::Path, format_version: u32, samples: &[String], contigs: &[String], ploidy: usize) -> std::io::Result<()>` — writes `output_dir/meta.json`.
- Consumes: nothing.

- [ ] **Step 1: Add the `serde_json` dependency**

In `Cargo.toml`, under `[dependencies]`, add the line (alphabetical order is fine, e.g. after `rust-htslib`):

```toml
serde_json = "1"
```

- [ ] **Step 2: Write the failing unit test in a new `src/meta.rs`**

Create `src/meta.rs` with only the test (the module items don't exist yet, so it won't compile — that is the red):

```rust
//! Top-level `meta.json` writer. Written once by the conversion pipeline after
//! every contig succeeds — the only scope that knows the full samples / contigs
//! / ploidy. Rust never reads this back; Python does (via `json.load`).

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tempfile::tempdir;

    #[test]
    fn test_write_meta_round_trip() {
        let tmp = tempdir().unwrap();
        let samples = vec!["s1".to_string(), "s2".to_string()];
        let contigs = vec!["chr1".to_string(), "chr2".to_string()];

        write_meta(tmp.path(), FORMAT_VERSION, &samples, &contigs, 2).unwrap();

        let text = std::fs::read_to_string(tmp.path().join("meta.json")).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["format_version"], 1);
        assert_eq!(v["samples"], serde_json::json!(["s1", "s2"]));
        assert_eq!(v["contigs"], serde_json::json!(["chr1", "chr2"]));
        assert_eq!(v["ploidy"], 2);
    }
}
```

Register the module so it compiles: in `src/lib.rs`, add `pub mod meta;` to the `pub mod ...;` block (e.g. right after `pub mod merge;`).

- [ ] **Step 3: Run the test to verify it fails**

Run: `pixi run -e lint cargo test --no-default-features meta::`
Expected: FAIL to compile — `cannot find function \`write_meta\`` / `cannot find value \`FORMAT_VERSION\``.

- [ ] **Step 4: Implement `FORMAT_VERSION` and `write_meta`**

Prepend the implementation to `src/meta.rs` (above the `#[cfg(test)] mod tests`):

```rust
use serde_json::json;
use std::path::Path;

/// Integer schema version for the on-disk SVAR2 layout, used by `SparseVar2` to
/// negotiate. Bump on any breaking layout/dtype change. Array dtypes are keyed
/// to this version (see `docs/roadmap/data-model.md`), not duplicated in the JSON.
pub const FORMAT_VERSION: u32 = 1;

/// Write the top-level `{output_dir}/meta.json`. Called once, after all contigs
/// convert successfully.
pub fn write_meta(
    output_dir: &Path,
    format_version: u32,
    samples: &[String],
    contigs: &[String],
    ploidy: usize,
) -> std::io::Result<()> {
    let meta = json!({
        "format_version": format_version,
        "samples": samples,
        "contigs": contigs,
        "ploidy": ploidy,
    });
    // Serializing a serde_json::Value cannot fail (no custom Serialize), so the
    // only real error is the filesystem write.
    let bytes = serde_json::to_vec_pretty(&meta).expect("serialize meta.json value");
    std::fs::write(output_dir.join("meta.json"), bytes)
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e lint cargo test --no-default-features meta::`
Expected: PASS (`test_write_meta_round_trip ... ok`).

- [ ] **Step 6: Format, lint, commit**

Run:
```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
```
Then commit:
```bash
git add Cargo.toml Cargo.lock src/meta.rs src/lib.rs
git commit -m "feat(svar-2): add write_meta helper + FORMAT_VERSION (meta.json writer)"
```

---

## Task 3: Wire `write_meta` into `run_conversion_pipeline`

Emit `meta.json` once, after every contig succeeds, from the only scope that owns the full `chroms` / `samples` / `ploidy`.

**Files:**
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes (from Task 2): `crate::meta::write_meta`, `crate::meta::FORMAT_VERSION`.
- Produces: nothing further (top of the write path).

**Testing note:** `run_conversion_pipeline` is a `#[pyfunction]` with no Rust callable harness (the e2e tests drive `process_chromosome` directly, and there is no Python binding for the pipeline yet). Per the spec, `write_meta` is verified by its own unit test (Task 2); this task's 3-line wiring is verified by a clean `cargo check --all-targets`, clippy, and the full suite staying green.

- [ ] **Step 1: Add the `write_meta` call after the contig-result loop**

In `src/lib.rs`, the tail of `run_conversion_pipeline` currently reads:

```rust
    for r in results {
        r.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    Ok(())
}
```

Replace it with:

```rust
    for r in results {
        r.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    // All contigs converted — write the top-level meta.json describing the cohort.
    crate::meta::write_meta(
        std::path::Path::new(&output_dir),
        crate::meta::FORMAT_VERSION,
        &samples,
        &chroms,
        ploidy,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to write meta.json: {e}"))
    })?;

    Ok(())
}
```

(`samples`, `chroms`, `output_dir`, and `ploidy` are `run_conversion_pipeline` params; the `py.detach` closure above borrows them, so they are still in scope here.)

- [ ] **Step 2: Verify it compiles and the suite stays green**

Run:
```bash
pixi run -e lint cargo check --all-targets
pixi run -e lint cargo test --no-default-features
```
Expected: `cargo check` clean; all tests PASS.

- [ ] **Step 3: Format, lint, commit**

Run:
```bash
pixi run -e lint cargo fmt --all
pixi run -e lint cargo clippy --all-targets -- -D warnings
```
Then commit:
```bash
git add src/lib.rs
git commit -m "feat(svar-2): write top-level meta.json after all contigs convert"
```

---

## Task 4: Reconcile `data-model.md`

Flip the three filenames to `.bin` in the layout tree (var_key + dense only), resolve the wire-format open question, document the dtype-vs-`format_version` convention, and drop the "provisional filenames" note.

**Files:**
- Modify: `docs/roadmap/data-model.md`

- [ ] **Step 1: Replace the dense-representation "Provisional filenames" bullet**

Find this bullet (in the `## Dense representation (dense)` section):

```markdown
- **Provisional filenames.** Like `var_key` (see [Open questions](#open-questions)),
  the current scratch implementation writes raw `.bin` final files —
  `final_positions.bin` / `final_keys.bin` / `final_genotypes.bin` under
  `{contig}/dense/{snp,indel}/` — mirroring the `var_key` naming rather than the
  `.npy` names in the layout tree below. These will be unified with the `var_key`
  wire format before the decode path (M6), as part of M3.
```

Replace it with:

```markdown
- **On-disk filenames.** The dense final files share `var_key`'s wire-format
  convention (see [Open questions](#open-questions) → resolved): `positions.bin` /
  `alleles.bin` / `genotypes.bin` under `{contig}/dense/{snp,indel}/`, all raw
  little-endian `.bin`. The dense representation has no ragged `offsets` sidecar —
  every hap contributes the same per-variant count, so the matrix shape is derived
  from `len(positions)` and `(n_samples, ploidy)` from `meta.json`.
```

- [ ] **Step 2: Flip the filenames in the on-disk layout tree (dense + var_key blocks only)**

In the ``` code block under `## On-disk layout`, replace the **dense/snp** block:

```
    │   ├── snp/
    │   │   ├── positions.npy       # sidecar SNP-variant positions (sorted)
    │   │   ├── alleles.npy         # 2-bit packed keys, one per SNP variant (no LUT)
    │   │   ├── {field}.npy         # per-variant INFO/FORMAT fields
    │   │   └── genotypes.npy       # 1-bit (sample, ploid, snp_variant) matrix, C-order
```

with:

```
    │   ├── snp/
    │   │   ├── positions.bin       # sidecar SNP-variant positions (sorted), u32 LE
    │   │   ├── alleles.bin         # 2-bit packed keys, one per SNP variant (no LUT), u8
    │   │   ├── {field}.npy         # per-variant INFO/FORMAT fields
    │   │   └── genotypes.bin       # 1-bit (sample, ploid, snp_variant) matrix, C-order, u8
```

Replace the **dense/indel** block:

```
    │   └── indel/
    │       ├── positions.npy
    │       ├── alleles.npy         # 32-bit keys, one per indel variant (points into shared LUT)
    │       ├── {field}.npy
    │       └── genotypes.npy       # 1-bit (sample, ploid, indel_variant) matrix, C-order
```

with:

```
    │   └── indel/
    │       ├── positions.bin       # u32 LE
    │       ├── alleles.bin         # 32-bit keys, one per indel variant (points into shared LUT), u32 LE
    │       ├── {field}.npy
    │       └── genotypes.bin       # 1-bit (sample, ploid, indel_variant) matrix, C-order, u8
```

Replace the **var_key/snp** block:

```
        ├── snp/
        │   ├── positions.npy       # per-call SNP positions (sorted within each hap)
        │   ├── alleles.npy         # 2-bit packed ALT, 4 calls/byte (uint8), no LUT
        │   ├── offsets.npy         # per (sample, ploid) ragged offsets into snp calls
        │   └── {field}.npy         # per-call INFO/FORMAT for SNP calls
```

with:

```
        ├── snp/
        │   ├── positions.bin       # per-call SNP positions (sorted within each hap), u32 LE
        │   ├── alleles.bin         # 2-bit packed ALT, 4 calls/byte (uint8), no LUT
        │   ├── offsets.npy         # per (sample, ploid) ragged offsets into snp calls, u64
        │   └── {field}.npy         # per-call INFO/FORMAT for SNP calls
```

Replace the **var_key/indel** block:

```
        └── indel/
            ├── positions.npy       # per-call indel positions (sorted within each hap)
            ├── alleles.npy         # 32-bit keys, one per call (ragged, points into shared LUT)
            ├── offsets.npy         # per (sample, ploid) ragged offsets into alleles
            └── {field}.npy
```

with:

```
        └── indel/
            ├── positions.bin       # per-call indel positions (sorted within each hap), u32 LE
            ├── alleles.bin         # 32-bit keys, one per call (ragged, points into shared LUT), u32 LE
            ├── offsets.npy         # per (sample, ploid) ragged offsets into alleles, u64
            └── {field}.npy
```

**Leave the `pointer/` sub-blocks (M11) untouched — they keep `.npy`.** Leave `long_alleles.bin`, `long_allele_offsets.npy`, and `max_del.npy` untouched.

- [ ] **Step 3: Add the dtype-vs-`format_version` convention paragraph**

Immediately after the closing ``` of the layout tree and its following `meta.json` paragraph (the paragraph that begins "`meta.json` carries the format version..."), insert a new paragraph:

```markdown
**Array dtypes are a `format_version` convention, not duplicated in `meta.json`.**
For `format_version = 1`: `positions.bin` is `u32` little-endian; `alleles.bin` is
`u8` (2-bit-packed SNP codes, 4/byte) in `snp/` and `u32` little-endian (inline value
or shared-LUT pointer) in `indel/`; `genotypes.bin` is `u8` (raw 1-bit hap-major
matrix, LSB-first); `offsets.npy` and `long_allele_offsets.npy` are `u64`;
`long_alleles.bin` is `u8`. The bulk parallel-`pwrite`n arrays (`positions` /
`alleles` / `genotypes`) are raw `.bin` — mmap-friendly, no npy-header offset to align
every `pwrite` past — and Python reads them with `np.memmap(path, dtype=…, mode='r')`,
deriving shape from `len(positions)` and `meta.json`. The small one-shot index/metadata
sidecars (`offsets`, `long_allele_offsets`) stay self-describing `.npy`.
```

- [ ] **Step 4: Resolve the "var_key sidecar wire format" open question**

In the `## Open questions` section, find:

```markdown
- **`var_key` sidecar wire format.** The merge currently writes positions and keys as
  raw little-endian `.bin` (`final_positions.bin` / `final_keys.bin`, via `bytemuck`)
  and offsets as `final_offsets.npy`, whereas the layout spec above names them `.npy`.
  Settle on one (raw `.bin` is mmap-friendly and avoids the npy header; `.npy` is
  self-describing) and align the names before the decode path (M6) is built.
```

Replace it with:

```markdown
- **`var_key` sidecar wire format.** *Resolved (M3): raw `.bin` for pwritten data
  arrays, `.npy` for one-shot sidecars.* The bulk arrays the tile merge writes via
  concurrent positional `pwrite` (`positions.bin`, `alleles.bin`, and dense
  `genotypes.bin`) are raw little-endian `.bin`: they carry no logical shape in the
  file (2-bit-packed SNP / 1-bit-packed genotype bytes), and wrapping them in `.npy`
  would force a hand-rolled 64-byte-aligned header that every `pwrite` must offset past
  — real fragility for no gain. The small one-shot index/metadata sidecars
  (`offsets.npy`, `long_allele_offsets.npy`) stay self-describing `.npy`. Array dtypes
  are keyed to `format_version` (see [On-disk layout](#on-disk-layout)).
```

- [ ] **Step 5: Verify the doc changes**

Run:
```bash
grep -n "final_positions\|final_keys\|final_offsets\|final_genotypes\|Provisional filenames" docs/roadmap/data-model.md
grep -n "positions.bin\|alleles.bin\|genotypes.bin" docs/roadmap/data-model.md
```
Expected: the first grep returns NOTHING (no provisional names, no "Provisional filenames" heading left). The second shows the flipped names under the dense + var_key blocks and in the new dtype paragraph. Confirm by eye that the `pointer/` block still says `positions.npy` / `alleles.npy`.

- [ ] **Step 6: Commit**

```bash
git add docs/roadmap/data-model.md
git commit -m "docs(svar-2): finalize data-model wire format (.bin sidecars, dtype convention)"
```

---

## Task 5: Reconcile `svar-2.md` and verify `architecture.md`

Mark M3 done, rewrite its remaining text, remove `max_del.npy` from M3's list (leaving it under M5), update M4's "unify dense filenames as part of M3" note, and confirm `architecture.md` still reads true.

**Files:**
- Modify: `docs/roadmap/svar-2.md`
- Verify (edit only if inaccurate): `docs/roadmap/architecture.md`

- [ ] **Step 1: Mark M3 done and rewrite its body**

In `docs/roadmap/svar-2.md`, find the M3 bullet:

```markdown
- [~] **M3. Per-contig split + sidecar positions.** Partition the SVAR2 directory by
  contig; keep positions as sidecar arrays. See
  [`architecture.md`](architecture.md#on-disk-layout).
  *Core done:* output is partitioned per contig (`{out}/{contig}/var_key/`) and the
  merge emits sorted position + offset sidecars. *Provisional / remaining:* the current
  scratch filenames live under `{out}/{contig}/var_key/{snp,indel}/` as
  `final_positions.bin` / `final_keys.bin` (raw little-endian via bytemuck) and
  `final_offsets.npy`, plus `long_alleles.bin` + `long_allele_offsets.npy` under
  `indel/`, which differ from the `.npy` names in the layout spec; `meta.json` and
  the per-contig `max_del.npy` are not yet written.
```

Replace it with:

```markdown
- [x] **M3. Per-contig split + sidecar positions + format finalization.** Partition the
  SVAR2 directory by contig; keep positions as sidecar arrays; finalize the on-disk
  format. See [`architecture.md`](architecture.md#on-disk-layout).
  *Done:* output is partitioned per contig
  (`{out}/{contig}/{var_key,dense}/{snp,indel}/`) and the merge emits sorted position +
  offset sidecars. The final per-stream files use their spec base names —
  `positions.bin` / `alleles.bin` / `genotypes.bin` (raw little-endian, mmap-friendly)
  and `offsets.npy` (one-shot) — plus the shared `long_alleles.bin` +
  `long_allele_offsets.npy` under `indel/`. A top-level `meta.json`
  (`format_version` / `samples` / `contigs` / `ploidy`) is written once after all
  contigs succeed. Array dtypes are a `format_version` convention documented in
  [`data-model.md`](data-model.md#on-disk-layout). *(The per-contig `max_del.npy` is
  produced and consumed by the overlap search, so it lands with M5, not here.)*
```

(`max_del.npy` is no longer claimed by M3; M4's note below already attributes it to M5.)

- [ ] **Step 2: Update M4's "provisional dense filenames / meta.json" note**

In `docs/roadmap/svar-2.md`, find the tail of the M4 bullet:

```markdown
  milestones):* `max_del.npy` / overlap queries (M5), `meta.json` (M3), and the
  `pointer` representation (M11) are not implemented here; the dense final filenames
  are provisional and mirror `var_key`'s (see the dense-representation section of
  [`data-model.md`](data-model.md#dense-representation-dense)), to be unified as part
  of M3.
```

Replace it with:

```markdown
  milestones):* `max_del.npy` / overlap queries (M5) and the `pointer` representation
  (M11) are not implemented here. `meta.json` and the unification of the dense final
  filenames with `var_key`'s spec base names (`positions.bin` / `alleles.bin` /
  `genotypes.bin`; see the dense-representation section of
  [`data-model.md`](data-model.md#dense-representation-dense)) landed in M3.
```

- [ ] **Step 3: Verify `architecture.md` reads true**

Run: `grep -n "meta.json\|max_del" docs/roadmap/architecture.md`
Read the `meta.json` bullet (around lines 91–94) and the on-disk-layout key points (around lines 85–90). Confirm the `meta.json` bullet lists exactly "version (for `SparseVar2` negotiation), samples, contigs, and ploidy" and that `max_del.npy` is described as a separate per-contig array (not part of `meta.json`).
Expected: both already read true — **no edit required.** Only if a statement is now false (e.g. it named a provisional filename), fix that one line.

- [ ] **Step 4: Verify no stale `final_*` names remain in the roadmap docs**

Run: `grep -rn "final_positions\|final_keys\|final_offsets\|final_genotypes" docs/roadmap/`
Expected: NOTHING.

- [ ] **Step 5: Commit**

```bash
git add docs/roadmap/svar-2.md
# add architecture.md too only if Step 3 required an edit:
# git add docs/roadmap/architecture.md
git commit -m "docs(svar-2): mark M3 done; move max_del.npy under M5; reconcile M4 note"
```

---

## Final verification (before opening the PR)

- [ ] **Full Rust suite green:** `pixi run -e lint cargo test --no-default-features` → all ~95 tests pass (94 pre-existing + `test_write_meta_round_trip`).
- [ ] **fmt + clippy clean:** `pixi run -e lint cargo fmt --all -- --check` and `pixi run -e lint cargo clippy --all-targets -- -D warnings`.
- [ ] **No provisional names anywhere but the intentional local var:** `grep -rn "final_positions\|final_keys\|final_offsets\|final_genotypes" src/ tests/ docs/` returns only the `final_offsets` in-memory local in `src/merge.rs`.
- [ ] Open the PR into `svar-2`.

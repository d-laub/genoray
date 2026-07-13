# SVAR2 `write_view` INFO/FORMAT fields + signature recompute — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry the store's INFO/FORMAT fields through `SparseVar2.write_view`'s region/sample subset re-conversion, and recompute the `mutcat` signature sidecar on the subset when `reference=` is given.

**Architecture:** Three change sites, no pipeline-core changes (the conversion pipeline is source-agnostic for fields — `svar1_reader.rs` already carries FORMAT from a store). (A) `Svar2Source` reads field values per variant via a **subset-aware provenance decode** and populates `RawRecord.info_raw`/`format_raw`, mirroring `Svar1RecordSource`. (B) `run_view_pipeline` rebuilds `FieldSpec`s from the source `meta.json` (concrete dtypes, not `Auto`) and runs a signature post-pass over the finished output store when `reference=` is given. (C) the Python `write_view` shim passes resolved field specs and drives the recompute.

**Tech Stack:** Rust (pyo3, memmap2, ndarray), Python (polars, numpy, cyclopts), pixi, pytest, cargo.

**Spec:** `docs/superpowers/specs/2026-07-13-svar2-write-view-fields-design.md`

## Global Constraints

- **Rust test invocation:** `pixi run -e lint test-rust` (= `cargo test --no-default-features --features conversion`). New pyfunctions/methods behind `#[cfg(feature="conversion")]` as needed. Never bare `cargo test` (pyo3 link error `undefined symbol: _Py_Dealloc`).
- **Cargo-on-NFS bus error:** before any commit that triggers cargo lint hooks, `export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$` in the same shell. Prefer a warm, stable target dir (e.g. `/tmp/genoray-cargo-commit`) so hook rebuilds are incremental and don't exceed the 2-min foreground limit — otherwise run the commit in the background.
- **Rebuild after Rust edits:** `pixi run -e py310 maturin develop --release` (foreground; do not background). Required before any Python test that exercises new Rust.
- **Python tests:** `pixi run -e py310 pytest <path>`.
- **Prek hooks** must be installed; commits run cargo fmt/check/clippy (`-D warnings`) + pyrefly + commitizen. Fix `needless_range_loop` (use `.iter().enumerate()`) and `type_complexity` (`#[allow(clippy::type_complexity)]`, as elsewhere in `lib.rs`) preemptively.
- **Conventional Commits** for every commit.
- **Public-API rule:** `write_view(fields=, reference=)` behavior changes → update `skills/genoray-api/SKILL.md` and `CHANGELOG.md` (`## Unreleased`) in this work (Task 6).
- **Coordinate/missing conventions unchanged:** 0-based half-open internally; SVAR2 biallelic post-atomization. Fields are scalar-numeric (+ INFO Flag), matching `from_vcf`.
- **No pipeline-core edits:** do NOT modify `chunk_assembler.rs`, `rvk.rs`, `field_finalize.rs`, `mutcat/annotate.rs` — they are already source-agnostic. If a task seems to need one, stop and reconsider.

---

## File Structure

- **Modify** `src/query/gather.rs` — add subset-aware provenance to the decode surface `Svar2Source` uses (Task 1).
- **Modify** `src/svar2_source.rs` — open source `FieldView`s; populate `info_raw`/`format_raw` per variant group (Task 2).
- **Modify** `src/lib.rs` (`run_view_pipeline`) — accept resolved field specs; drop the empty-fields guard; build `FieldSpec`s with concrete dtypes; signature post-pass when `reference=` given (Tasks 3, 4).
- **Modify** `python/genoray/_svar2.py` (`write_view`) — pass resolved specs; drive recompute; docstring (Task 5).
- **Test** `tests/test_svar2_source.rs`, `tests/test_view_pipeline.rs` (Rust); `tests/test_svar2_write_view.py` (Python).
- **Modify** `skills/genoray-api/SKILL.md`, `CHANGELOG.md`, `docs/roadmap/data-model.md` (Task 6).

---

## Task 1: Subset-aware provenance decode primitive (Rust)

**Problem:** `Svar2Source::new` gets its calls from `read_ranges(reader, regions, Some(sample_orig_idx))` → `BatchResult`, then `decode_hap` (`src/query/gather.rs:808`). `BatchResult::decode_hap_src` (`gather.rs:850-936`) already returns `(HapCalls, Vec<RecordSrc>)` — **but it asserts `vk_src`/`dense_src` are populated**, which only `overlap_batch_src` does (whole cohort, no sample subset). The subset `find_ranges`/`gather_ranges` path leaves them empty. We must make the **subset** decode provenance-carrying without decoding the whole cohort.

**Chosen approach:** make the subset unified path (`find_ranges` → `gather_ranges`) populate `vk_src`/`dense_src` on its `BatchResult`, so the existing `BatchResult::decode_hap_src` works unchanged. Mirror how `overlap_batch`→`overlap_batch_src` gained provenance via the `SrcKeyRef` element type and the `dense_src` capture (`gather.rs:191-294`), applying the same generic (`T: VkElem`, `T::CARRIES_SRC`) to `gather_ranges`. Add a public `read_ranges_src(reader, regions, samples) -> BatchResult` wrapper (analogous to `read_ranges`, `gather.rs:794-800`) that goes through the provenance-carrying gather.

*(Rationale for not using `BatchResultSplit`: the read-bound split result (`gather_haps_readbound_src`) carries `vk_src` but has NO decode method and a different split-dense layout; adding a decode there is more surface than parameterizing the unified subset gather that already has `decode_hap_src`.)*

**Files:**
- Modify: `src/query/gather.rs` (`find_ranges`/`gather_ranges` src population; add `read_ranges_src`)
- Modify: `src/query/mod.rs` (export `read_ranges_src`, `RecordSrc` if not already)
- Test: `tests/test_svar2_source.rs` (Rust)

**Interfaces:**
- Produces: `pub fn read_ranges_src(reader: &ContigReader, regions: &[(u32,u32)], samples: Option<&[usize]>) -> BatchResult` — a `BatchResult` with `vk_src.len()==vk.len()` and `dense_src.len()==dense.len()`, so `BatchResult::decode_hap_src(reader, r, s, p) -> (HapCalls, Vec<RecordSrc>)` is callable on a **sample subset**. `RecordSrc { is_dense, is_indel, idx }` (`gather.rs:837`): `idx` = var_key absolute call index (`!is_dense`) or dense variant row (`is_dense`).

- [ ] **Step 1: Read the existing provenance machinery.** Study `overlap_batch_src` and the generic `overlap_batch_impl::<T>` (`gather.rs:191-294`), `SrcKeyRef`/`KeyRef`/`VkElem`/`CARRIES_SRC` (`src/spine.rs:33-57`), `DenseSrcElem` and the `dense_src` capture, and `find_ranges`/`gather_ranges` (search `fn find_ranges`, `fn gather_ranges`). Identify where `gather_ranges` builds `vk`/`dense` and where the `SrcKeyRef` packing + `dense_src` capture must hook in. Confirm whether `find_ranges` already carries enough to pack `vk_src` (the CSR provenance) for a subset, or whether the subset gather needs the same `T::split`/`dense_src` capture as `overlap_batch_impl`.

- [ ] **Step 2: Write the failing Rust test.** In `tests/test_svar2_source.rs`, build (or reuse the existing test's) small 2-sample svar2 store with a mix of var_key and dense variants, open a `ContigReader`, and assert `read_ranges_src` + `decode_hap_src` returns, for a chosen `(r,s,p)`, the same `HapCalls` as `read_ranges`+`decode_hap` AND a `RecordSrc` per record whose `(is_dense,is_indel,idx)` indexes back to the correct on-disk position (cross-check `idx` against `reader.vk_snp_positions()` / dense positions).

```rust
#[test]
fn read_ranges_src_carries_provenance_for_subset() {
    // build/open a tiny 2-sample store with >=1 var_key snp, >=1 dense variant
    let reader = /* ContigReader::open(...) */;
    let regions = [(0u32, 1_000u32)];
    let subset = [0usize]; // subset to sample 0 only
    let br = read_ranges_src(&reader, &regions, Some(&subset));
    let (hc, srcs) = br.decode_hap_src(&reader, 0, 0, 0);
    assert_eq!(hc.positions.len(), srcs.len());
    for (pos, src) in hc.positions.iter().zip(&srcs) {
        // provenance must point at a real record with this position
        if src.is_dense {
            // dense row -> dense_{snp,indel} positions[idx] == *pos
        } else {
            // var_key call idx -> vk_{snp,indel} positions[idx] == *pos
        }
    }
    // and genotypes identical to the non-provenance subset decode:
    let br0 = read_ranges(&reader, &regions, Some(&subset));
    assert_eq!(hc.positions, br0.decode_hap(&reader, 0,0,0).positions);
}
```

- [ ] **Step 3: Run the test — verify it fails** (`read_ranges_src` undefined / `decode_hap_src` panics on empty `vk_src`).

Run: `export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$; pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion --test test_svar2_source read_ranges_src'`
Expected: FAIL.

- [ ] **Step 4: Implement.** Parameterize `gather_ranges` (and, if needed per Step 1, `find_ranges`) over the element type so `SrcKeyRef` packs `vk_src` and the `dense_src` capture runs — mirroring `overlap_batch_impl::<T>` (`gather.rs:191-294`). Add:
```rust
pub fn read_ranges_src(reader: &ContigReader, regions: &[(u32, u32)], samples: Option<&[usize]>) -> BatchResult {
    // same as read_ranges but through the provenance-carrying gather:
    gather_ranges_src(reader, &find_ranges(reader, regions, samples))
}
```
Keep `read_ranges` (no-src) byte-identical and zero-cost (the `KeyRef` path must not pay for provenance). Export `read_ranges_src` from `src/query/mod.rs`.

- [ ] **Step 5: Run the test — verify it passes.** Same command as Step 3. Expected: PASS. Also run the full file to catch regressions: `... --test test_svar2_source`.

- [ ] **Step 6: Commit.**
```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add src/query/gather.rs src/query/mod.rs tests/test_svar2_source.rs
git commit -m "feat(svar2): subset-aware provenance decode (read_ranges_src)"
```

---

## Task 2: `Svar2Source` populates `info_raw`/`format_raw` (Rust)

**Files:**
- Modify: `src/svar2_source.rs`
- Test: `tests/test_svar2_source.rs`

**Interfaces:**
- Consumes: `read_ranges_src` + `BatchResult::decode_hap_src` (Task 1); `FieldView` (`src/query/field.rs`), `FieldValue`; `RawRecord.info_raw: Vec<Option<Vec<f64>>>` / `format_raw: Vec<Option<Vec<Vec<f64>>>>` (`src/record_source.rs:22-31`).
- Produces: `Svar2Source::new(store_path, chrom, sample_orig_idx, ploidy, regions, overlap_mode, fields: &[FieldSpec])` — new `fields` param; when non-empty, emitted `RawRecord`s carry field values in spec order.

**Template to mirror:** `gather_batch_fields` (`src/py_query_decode.rs:156-203`) for the `RecordSrc`→`FieldView` mapping; `Svar1RecordSource::next_record` (`src/svar1_reader.rs:201-222`) for the exact `format_raw` shape (per-sample vec; source missing-sentinel for non-carriers).

- [ ] **Step 1: Write the failing Rust test.** Build a tiny store with a known INFO field (e.g. `AF` float, dense) and a FORMAT field (e.g. `DP` int, var_key + dense variants), via `SparseVar2.from_vcf(..., info_fields=["AF"], format_fields=["DP"])` in a fixture (or a Rust harness that shells to Python once). Assert `Svar2Source::new(..., fields=&[AF_spec, DP_spec])` yields, for a specific variant, `info_raw[0] == Some(vec![af_value])` and `format_raw[0][s_out] == Some(vec![dp_value])` for carriers, sentinel for non-carriers. (Values cross-checked against a direct `FieldView` read.)

- [ ] **Step 2: Run — verify it fails** (`new` has no `fields` param / fields empty).

Run: `... --test test_svar2_source svar2_source_carries_fields`
Expected: FAIL.

- [ ] **Step 3: Implement.**
  - Change `Svar2Source::new` signature to take `fields: &[crate::field::FieldSpec]`.
  - Switch the gather to `read_ranges_src` and the decode to `decode_hap_src` (Task 1), so each kept call yields a `RecordSrc`.
  - Open the field sidecars once: for each `FieldSpec`, an `OpenField`-style set of four `FieldView`s keyed by `FieldSub::all()` (see `py_query_decode.rs:72-122`), using the spec's concrete `StorageDtype`.
  - Extend the group value (currently `Vec<bool>` carriers) to also accumulate, per group: INFO scalar (read once from the first kept call's `RecordSrc`) and per-output-sample FORMAT scalar. Store `FieldValue`→`f64` (widen). Use the sub-stream picked from `(src.is_dense, src.is_indel)` and `src.idx`; for dense FORMAT stride `src.idx` is the dense row and the cohort sample is `sample_orig_idx[s_out]`.
  - In `to_raw_record`, emit `info_raw`/`format_raw` in spec order: `None` when the source element equals the field's missing sentinel; per-sample sentinel for FORMAT non-carriers (mirror `svar1_reader.rs:201-213`).
  - Update the two existing callers of `Svar2Source::new` (`src/orchestrator.rs` `SourceSpec::Svar2` arm, and any test) to pass `fields` (empty slice preserves today's genotypes-only behavior).

- [ ] **Step 4: Run — verify it passes.** `... --test test_svar2_source`. Expected: PASS (new + existing genotype tests).

- [ ] **Step 5: Rebuild + commit.**
```bash
pixi run -e py310 maturin develop --release
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add src/svar2_source.rs src/orchestrator.rs tests/test_svar2_source.rs
git commit -m "feat(svar2): Svar2Source carries INFO/FORMAT field values"
```

---

## Task 3: `run_view_pipeline` threads field specs (Rust)

**Files:**
- Modify: `src/lib.rs` (`run_view_pipeline`, ~`457-642`)
- Test: `tests/test_view_pipeline.rs`

**Interfaces:**
- Produces (NEW pyo3 signature — Task 5 consumes it): replace `fields: Vec<String>` with `fields: Vec<(String, String, String, Option<f64>)>` = `(name, category, dtype_str, default)`, the resolved concrete-dtype manifest from the source store. `regions_overlap`, `merge_overlapping`, `reference`, etc. unchanged.

- [ ] **Step 1: Write the failing Rust test.** In `tests/test_view_pipeline.rs`, build a source store with `info_fields`/`format_fields`, run the view pipeline over a **full region + all samples** with the resolved field specs, then assert the output contig's `field/*/*/values.bin` sidecars are **byte-identical** to the source's (the strong oracle; extend the existing genotype byte-parity test).

- [ ] **Step 2: Run — verify it fails** (guard rejects non-empty fields).

Run: `... --test test_view_pipeline view_carries_fields_byte_parity`
Expected: FAIL (`field carry-through is not yet implemented`).

- [ ] **Step 3: Implement.**
  - Change the pyo3 signature per Interfaces; update the `#[pyo3(signature=...)]`.
  - Remove the `if !fields.is_empty() { return Err(...) }` guard (`lib.rs:477-482`).
  - Build `Vec<FieldSpec>` from the tuples: `StorageDtype::from_meta_str(dtype_str)` (`src/field.rs:65`); reconstruct `HtslibType` from the dtype (float dtypes → `Float`; `bool`/int → `Int`); `FieldCategory` from the category string; `default` passed through. **Do not** use `Auto`.
  - Pass this `fields_spec` (not `Vec::new()`) into `process_chromosome` (`lib.rs:617`) and `finalize_fields` (`lib.rs:628`).
  - Pass `fields_spec` into `Svar2Source::new` via the `SourceSpec::Svar2` construction (Task 2's new param) — thread through `orchestrator.rs` `SourceSpec::Svar2 { ..., fields }` if that is how the source is built.
  - Ensure `write_meta` records these fields in the output `meta.json`.
  - `fasta_path` stays `None` (normalization untouched).

- [ ] **Step 4: Run — verify it passes.** `... --test test_view_pipeline`. Expected: PASS (byte-parity).

- [ ] **Step 5: Rebuild + commit.**
```bash
pixi run -e py310 maturin develop --release
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add src/lib.rs src/orchestrator.rs tests/test_view_pipeline.rs
git commit -m "feat(svar2): run_view_pipeline carries INFO/FORMAT fields"
```

---

## Task 4: Signature recompute post-pass (Rust)

**Files:**
- Modify: `src/lib.rs` (`run_view_pipeline`)
- Test: `tests/test_view_pipeline.rs`

**Interfaces:**
- Consumes: `reference: Option<String>` (already in the signature); `crate::mutcat::annotate::annotate_contig` (`src/mutcat/annotate.rs:13`); `load_contig_seq` (used at `orchestrator.rs:570`); `ContigReader::open`.
- Produces: when `reference` is `Some`, the output store gains a `mutcat` sidecar per contig equivalent to `from_vcf(signatures=True)` on the same subset.

- [ ] **Step 1: Write the failing Rust test.** Build a source store, run the view over full region/all samples **with `reference=<fasta>`**, and assert (a) the output `mutcat` sidecar exists and (b) it is byte-identical to a `from_vcf(signatures=True)`-built store's `mutcat` on the same data (parity). Add a second assertion: a subset that drops a DBS-adjacent variant still produces a valid recomputed `mutcat` (no positional copy).

- [ ] **Step 2: Run — verify it fails** (no mutcat written; `reference` ignored — `lib.rs:475`).

Run: `... --test test_view_pipeline view_recomputes_signatures`
Expected: FAIL.

- [ ] **Step 3: Implement.** After all output contigs are written (and `finalize_fields`/`write_meta` done), if `reference.is_some()`: for each output contig, `let ref_seq = load_contig_seq(fasta, chrom)?; let reader = ContigReader::open(out_dir, chrom, n_samples_out, ploidy)?; annotate::annotate_contig(&reader, &paths, &ref_seq)?;` — the same store-based path `process_chromosome` uses at `orchestrator.rs:567-583` and `SparseVar2.annotate_mutations`. Do NOT route `reference` into `process_chromosome`'s `fasta_path` (keep normalization at `None`). Ensure `meta.json` records that signatures are present (match what `from_vcf(signatures=True)` writes).

- [ ] **Step 4: Run — verify it passes.** `... --test test_view_pipeline`. Expected: PASS.

- [ ] **Step 5: Rebuild + commit.**
```bash
pixi run -e py310 maturin develop --release
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add src/lib.rs tests/test_view_pipeline.rs
git commit -m "feat(svar2): write_view recomputes mutcat signatures from reference"
```

---

## Task 5: Python `write_view` plumbing (Python)

**Files:**
- Modify: `python/genoray/_svar2.py` (`write_view`, ~`326-440`)
- Test: `tests/test_svar2_write_view.py`

**Interfaces:**
- Consumes: `_core.run_view_pipeline` with the new `fields: list[tuple[str,str,str,float|None]]` signature (Task 3); `self.available_fields` (`StoredField`, `python/genoray/_svar2_fields.py`).

- [ ] **Step 1: Write the failing Python test.** In `tests/test_svar2_write_view.py`:
```python
def test_write_view_carries_fields(tmp_path, svar2_store_with_fields):
    src = svar2_store_with_fields  # from_vcf(..., info_fields=["AF"], format_fields=["DP"])
    out = tmp_path / "view.svar2"
    src.write_view((CONTIG, 0, END), src.available_samples, out, fields=["AF", "DP"], overwrite=True)
    view = SparseVar2(out, fields=["AF", "DP"])
    # decode both over the same region/samples; AF/DP values match per (variant, sample)
    r_src = src.with_fields(["AF","DP"]).decode(CONTIG, [(0, END)])
    r_view = view.decode(CONTIG, [(0, END)])
    assert_ragged_field_equal(r_src, r_view, "info/AF")
    assert_ragged_field_equal(r_src, r_view, "format/DP")

def test_write_view_dtype_preserved(tmp_path, svar2_store_u16_field):
    # subset exercising only small values still writes u16, not u8
    ...
    assert SparseVar2(out).available_fields["INFO_U16"].dtype == np.uint16

def test_write_view_recomputes_signatures(tmp_path, svar2_store, ref_fasta):
    svar2_store.write_view((CONTIG,0,END), svar2_store.available_samples, out,
                           reference=ref_fasta, overwrite=True)
    assert "mutcat" in SparseVar2(out).available_fields  # or the mutcat presence check

def test_write_view_fields_without_reference_no_mutcat(...):
    # fields=["AF"] but no reference -> AF carried, no mutcat
    ...

def test_write_view_mutcat_field_without_reference_raises(...):
    with pytest.raises(ValueError, match="mutcat"):
        store.write_view(..., fields=["mutcat"])  # no reference
```

- [ ] **Step 2: Run — verify they fail.**

Run: `pixi run -e py310 pytest tests/test_svar2_write_view.py -k "carries_fields or dtype_preserved or recomputes or without_reference or mutcat" -x`
Expected: FAIL.

- [ ] **Step 3: Implement.** In `write_view`: resolve requested `fields` against `self.available_fields`, excluding `mutcat`, into `(name, category, dtype_str, default)` tuples (dtype from `StoredField.dtype`, category from `StoredField.category`); pass to `_core.run_view_pipeline` (new arg). Keep `fields=None` → genotypes-only (`[]`). Keep the existing `mutcat`-without-`reference` → `ValueError` guard and fail-fast band ordering. Update the docstring (fields carried; `reference` recomputes signatures). Update `genoray view` CLI `--fields`/`--reference` help if wording claims "not implemented".

- [ ] **Step 4: Run — verify they pass.**

Run: `pixi run -e py310 pytest tests/test_svar2_write_view.py -x -q`
Expected: PASS (new + existing genotype-only tests).

- [ ] **Step 5: Commit.**
```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add python/genoray/_svar2.py tests/test_svar2_write_view.py
git commit -m "feat(svar2): write_view fields + reference signature recompute (Python)"
```

---

## Task 6: Docs (SKILL.md + CHANGELOG + roadmap)

**Files:**
- Modify: `skills/genoray-api/SKILL.md`, `CHANGELOG.md`, `docs/roadmap/data-model.md`

- [ ] **Step 1: Update `SKILL.md`.** Document that `write_view(fields=[...])` now carries INFO/FORMAT fields (was "not yet implemented"), and `reference=` recomputes the `mutcat` sidecar on the subset. Note `fields=None` = genotypes only; requesting `mutcat` without `reference` raises; output field dtypes preserve the source's.

- [ ] **Step 2: Update `CHANGELOG.md`** under `## Unreleased` (human-readable entry; do NOT bump version).

- [ ] **Step 3: Update `docs/roadmap/data-model.md`** M9 section: field carry-through + signature recompute now implemented; only the streaming `Svar2Source` and `reroute=False` (permanent `NotImplementedError`) remain deferred.

- [ ] **Step 4: Commit.**
```bash
git add skills/genoray-api/SKILL.md CHANGELOG.md docs/roadmap/data-model.md
git commit -m "docs(svar2): document write_view field carry-through + signature recompute"
```

---

## Self-review notes (plan author)

- **Spec coverage:** Site A → Tasks 1+2; Site B (fields) → Task 3; Site B (signatures) → Task 4; Site C → Task 5; API/docs → Task 6. Byte-parity/subset/dtype/signature/no-reference tests all placed. ✓
- **Type consistency:** `read_ranges_src` (T1) → `decode_hap_src`/`RecordSrc` (existing) → `Svar2Source::new(fields)` (T2) → `run_view_pipeline` tuple `fields` (T3) → Python tuple pass (T5). Signature-recompute uses `annotate_contig` (T4). Consistent across tasks. ✓
- **Known open item (Task 1 Step 1 resolves):** exact hook point for provenance in `find_ranges`/`gather_ranges` — the implementer must confirm whether `find_ranges` alone can pack `vk_src` for a subset or the `dense_src` capture must move into `gather_ranges` (mirroring `overlap_batch_impl::<T>`). This is genuine investigation, not a placeholder — the interface (`read_ranges_src` → provenance-carrying `BatchResult`) is fixed.

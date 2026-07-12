# SVAR2 fields — PR#100 follow-ups, cleanup & field-path optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close PR#100's three deferred follow-ups, DRY/clean the new field-write code, and cut the finalize pass's cost via a measurement-driven, byte-identical optimization.

**Architecture:** Work lands as new commits on the open PR#100 branch `worktree-svar2-field-specs`. Tests come first (lock behavior), then the refactors run green, then a bracketed profile→optimize→re-profile pass scoped to the field-write code. The reader/htslib path is explicitly off-limits.

**Tech Stack:** Rust (rust-htslib, rayon, bytemuck, ndarray-npy), PyO3, Python (pytest, vcfixture), `perf` + `valgrind --tool=callgrind` + `cargo asm` (cargo-show-asm) for profiling.

## Global Constraints

- **Commits:** Conventional Commits (`feat:`/`fix:`/`refactor:`/`test:`/`perf:`/`docs:`/`chore:`). End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Rust tests:** always `pixi run bash -lc 'cargo test --no-default-features --features conversion ...'` — without `--no-default-features` the pyo3 test binary fails to link (`undefined symbol: _Py_Dealloc`).
- **Python tests:** `pixi run pytest tests/<file>.py -v` for a single file; `pixi run test` for the full suite (it regenerates fixtures via `gen_from_vcf.sh`).
- **prek hooks** are already installed in this repo's git-common-dir; commits/pushes run them.
- **Byte-identical output:** the optimization must produce on-disk field `values.bin` bytes identical to pre-optimization. This is the acceptance bar for Tasks 7–8.
- **No public API change:** nothing reachable from `import genoray` changes name/semantics; `skills/genoray-api/SKILL.md` needs no edit (confirmed at close-out).
- **Off-limits:** the reader/htslib conversion path (known reader-bound; prior optimization round already done). Do not modify `vcf_reader.rs` for perf.
- **Versioning:** do NOT bump the version or edit versioned `CHANGELOG.md` sections; accumulate entries under `## Unreleased` only.
- Profiling artifacts (drivers, synthetic VCFs, callgrind out files) live under `$CLAUDE_JOB_DIR/tmp` and are NOT committed.

---

## Stage 1 — Tests first (follow-ups #2, #3)

### Task 1: Rust unit test — `f32→f32` no-op finalize rewrite

Covers the finalize path where a Float field already has an explicit concrete `f32` dtype, so `resolve_dtype` returns `F32` and `rewrite_file` re-encodes f32→f32 (no width change). Currently untested.

**Files:**
- Test: `src/field_finalize.rs` (append to the existing `#[cfg(test)] mod tests`, after the last test near line 690)

**Interfaces:**
- Consumes (existing test helpers in this module): `float_field(name: &str, dtype: StorageDtype, default: Option<f64>) -> FieldSpec` (field_finalize.rs:507), `write_f32_field(root: &Path, contig: &str, name: &str, sub: &str, values: &[f32])` (field_finalize.rs:481), `read_values_bin(path: &Path) -> Vec<u8>` (field_finalize.rs:493), `finalize_fields(output_dir: &Path, contigs: &[String], fields: &[FieldSpec]) -> Result<Vec<ResolvedField>, ConversionError>` (field_finalize.rs:84).
- Produces: nothing consumed downstream (test only).

- [ ] **Step 1: Write the failing test**

Append inside `mod tests` in `src/field_finalize.rs`:

```rust
    #[test]
    fn explicit_f32_rewrites_in_place_byte_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let contig = "chr1";
        let vals: [f32; 4] = [0.5, -1.25, 0.0, 3.5];
        // stage under one var_key sub label
        write_f32_field(root, contig, "DS", "var_key_snp", &vals);
        let path = root
            .join(contig)
            .join("fields")
            .join("format")
            .join("DS")
            .join("var_key_snp")
            .join("values.bin");
        let before = read_values_bin(&path);

        let field = float_field("DS", StorageDtype::F32, None);
        let resolved = finalize_fields(root, &[contig.to_string()], &[field]).unwrap();

        assert_eq!(resolved[0].dtype, StorageDtype::F32);
        let after = read_values_bin(&path);
        // f32 staged -> f32 resolved: rewrite must reproduce the same 16 bytes.
        assert_eq!(before, after, "f32->f32 finalize must be byte-identical");
    }
```

Note: verify `float_field` builds a **format**-category field (the path uses `fields/format/DS`). If `float_field` defaults to INFO, either add a category arg inline or construct the `FieldSpec` literal in-test with `category: FieldCategory::Format`. Read field_finalize.rs:507 first and match the actual helper.

- [ ] **Step 2: Run test to verify it fails (or reveals the helper's category)**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --lib field_finalize::tests::explicit_f32_rewrites_in_place_byte_identical -- --nocapture'`
Expected: initially FAIL if the path/category is wrong (fix the fixture to match the helper), then compile+run.

- [ ] **Step 3: Adjust fixture to the real helper signature**

If `float_field` is INFO-only, replace its use with an explicit spec so the staged path matches `finalize`'s enumeration:

```rust
        let field = FieldSpec {
            name: "DS".into(),
            category: FieldCategory::Format,
            htype: HtslibType::Float,
            dtype: StorageDtype::F32,
            default: None,
        };
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --lib field_finalize::tests::explicit_f32_rewrites_in_place_byte_identical'`
Expected: PASS (`test result: ok. 1 passed`).

- [ ] **Step 5: Commit**

```bash
git add src/field_finalize.rs
git commit -m "test(svar2): cover f32->f32 no-op finalize rewrite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Python e2e — VarKeyIndel FORMAT field_calls + multi-field ordering

Two behaviors the PR body flagged as untested, both exercised end-to-end through routing → merge → finalize:
1. A rare indel (few carriers, small cohort) routes to the **var_key_indel** stream, and its FORMAT field's `var_key_indel/values.bin` is written non-empty.
2. Multiple INFO and multiple FORMAT fields resolve to distinct, correct per-field files with no cross-contamination.

Routing rationale (do not guess): `choose_representation` (src/cost_model.rs:60) makes an indel route to VarKey when `x*(64+info+format) <= 32+32+np + info + format*n_samples`. With 2 samples (np=4) and a single-carrier indel, `var_key ≈ 64 < dense ≈ 68`, so a rare indel is var_key deterministically.

**Files:**
- Test: `tests/test_svar2_fields.py` (append new tests + a fixture builder)

**Interfaces:**
- Consumes: `vcfixture` `VcfBuilder`, `Number`, `Seq`, `Type` (already imported at tests/test_svar2_fields.py:19); `genoray.SparseVar2.from_vcf(out, source, *, no_reference=True, info_fields=..., format_fields=...)`; `genoray._svar2_fields.FormatField` (imported at line 21).
- Produces: nothing downstream (tests only).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar2_fields.py`:

```python
def _rare_indel_vcf(tmp_path: Path) -> Path:
    """2 samples; a biallelic indel carried by exactly ONE call, plus a FORMAT
    float DS. With np=4 and a single carrier call, choose_representation routes
    the indel to var_key_indel (var_key ~64 bits < dense ~68), so DS must land
    in fields/format/DS/var_key_indel/values.bin."""
    doc = (
        VcfBuilder(samples=["s0", "s1"], contigs=[("chr1", 100_000)])
        .fmt("GT")
        .fmt("DS", Number.ONE, Type.FLOAT)
        .record(
            "chr1",
            100,
            ref="AT",
            alt=[Seq("A")],  # 1bp deletion (indel)
            gt=["0|1", "0|0"],  # one carrier call total
            DS=[[0.5], [0.0]],
        )
    )
    return doc.write(tmp_path / "rare_indel.vcf.gz", bgzip=True, index=True)


def test_from_vcf_varkey_indel_format_field_written(tmp_path: Path):
    from genoray import SparseVar2

    src = _rare_indel_vcf(tmp_path)
    out = tmp_path / "store_vk_indel.svar2"
    SparseVar2.from_vcf(
        out, str(src), no_reference=True, format_fields=[FormatField("DS", default=0.0)]
    )
    import json

    meta = json.loads((out / "meta.json").read_text())
    contig = meta["contigs"][0]
    vk_indel = out / contig / "fields" / "format" / "DS" / "var_key_indel" / "values.bin"
    assert vk_indel.is_file(), "var_key_indel DS values.bin missing"
    assert vk_indel.stat().st_size > 0, "var_key_indel DS field_calls not written"


def _multifield_vcf(tmp_path: Path) -> Path:
    """2 INFO + 2 FORMAT scalar fields with distinct values, to prove per-field
    files don't cross-contaminate through routing/merge/finalize."""
    doc = (
        VcfBuilder(samples=["s0", "s1"], contigs=[("chr1", 100_000)])
        .info("AC", Number.A, Type.INTEGER)
        .info("AN", Number.ONE, Type.INTEGER)
        .fmt("GT")
        .fmt("DP", Number.ONE, Type.INTEGER)
        .fmt("DS", Number.ONE, Type.FLOAT)
        .record(
            "chr1",
            100,
            ref="A",
            alt=[Seq("C")],
            gt=["0|1", "1|1"],
            info={"AC": 3, "AN": 4},
            DP=[[10], [20]],
            DS=[[0.25], [0.75]],
        )
    )
    return doc.write(tmp_path / "multifield.vcf.gz", bgzip=True, index=True)


def test_from_vcf_multi_field_no_cross_contamination(tmp_path: Path):
    import json

    from genoray import SparseVar2

    src = _multifield_vcf(tmp_path)
    out = tmp_path / "store_multi.svar2"
    SparseVar2.from_vcf(
        out,
        str(src),
        no_reference=True,
        info_fields=["AC", "AN"],
        format_fields=["DP", "DS"],
    )
    meta = json.loads((out / "meta.json").read_text())
    by_name = {f["name"]: f for f in meta["fields"]}
    # Each field is recorded exactly once under its own category.
    assert by_name["AC"]["category"] == "info"
    assert by_name["AN"]["category"] == "info"
    assert by_name["DP"]["category"] == "format"
    assert by_name["DS"]["category"] == "format"
    # DS is the only Float field -> f32; the ints auto-narrow to a sub-4b width.
    assert by_name["DS"]["dtype"] == "f32"
    assert by_name["DP"]["dtype"] in {"i8", "u8", "i16", "u16"}
    contig = meta["contigs"][0]
    # Every declared field has at least one non-empty values.bin, and int fields
    # stay a whole number of their own resolved width (proves disjoint files).
    width = {"bool": 1, "i8": 1, "u8": 1, "i16": 2, "u16": 2, "f16": 2, "f32": 4, "i32": 4, "u32": 4}
    for name, spec in by_name.items():
        cat = spec["category"]
        files = list((out / contig / "fields" / cat / name).glob("*/values.bin"))
        assert files, f"{cat}/{name} has no values.bin"
        assert sum(p.stat().st_size for p in files) > 0, f"{cat}/{name} all empty"
        w = width[spec["dtype"]]
        assert all(p.stat().st_size % w == 0 for p in files), f"{cat}/{name} bad width"
```

- [ ] **Step 2: Run tests to verify they pass (behavior already implemented)**

Run: `pixi run pytest tests/test_svar2_fields.py::test_from_vcf_varkey_indel_format_field_written tests/test_svar2_fields.py::test_from_vcf_multi_field_no_cross_contamination -v`
Expected: both PASS. These lock existing behavior. If `var_key_indel` is empty, the indel routed dense — reduce carriers or raise sample count until routing flips (per the cost formula above); do not weaken the assertion to `>= 0`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar2_fields.py
git commit -m "test(svar2): cover var_key_indel FORMAT calls + multi-field isolation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Python e2e — combined `signatures=True` + fields

Confirms the mutcat write-time annotation and the INFO/FORMAT field pipeline compose (their cost terms are additive; this is the missing integration point). `signatures=True` requires a reference.

**Files:**
- Test: `tests/test_svar2_fields.py` (append)

**Interfaces:**
- Consumes: `SparseVar2.from_vcf(out, source, reference, *, signatures=True, info_fields=..., format_fields=...)`; a reference FASTA whose bases match the VCF REF alleles.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar2_fields.py`:

```python
def _ref_and_fields_vcf(tmp_path: Path) -> tuple[Path, Path]:
    """A reference FASTA + a bgzipped/indexed VCF whose REF bases match it,
    carrying one INFO int (AC) and one FORMAT float (DS). Reuses the 40bp
    reference convention from tests/test_svar2_from_vcf.py (_REF)."""
    import subprocess

    ref_seq = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"  # POS3='A', POS7='C'
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{ref_seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    doc = (
        VcfBuilder(samples=["s0", "s1"], contigs=[("chr1", 40)])
        .info("AC", Number.A, Type.INTEGER)
        .fmt("GT")
        .fmt("DS", Number.ONE, Type.FLOAT)
        .record("chr1", 3, ref="A", alt=[Seq("G")], gt=["1|0", "0|0"], info={"AC": 1}, DS=[[0.5], [0.0]])
        .record("chr1", 7, ref="C", alt=[Seq("CAT")], gt=["0|1", "1|1"], info={"AC": 3}, DS=[[0.9], [0.2]])
    )
    vcf = doc.write(tmp_path / "ref_fields.vcf.gz", bgzip=True, index=True)
    return vcf, ref


def test_from_vcf_signatures_and_fields_compose(tmp_path: Path):
    import json

    from genoray import SparseVar2

    vcf, ref = _ref_and_fields_vcf(tmp_path)
    out = tmp_path / "store_sig_fields.svar2"
    SparseVar2.from_vcf(
        out,
        str(vcf),
        str(ref),
        signatures=True,
        info_fields=["AC"],
        format_fields=[FormatField("DS", default=0.0)],
        threads=1,
    )
    meta = json.loads((out / "meta.json").read_text())
    names = {f["name"] for f in meta["fields"]}
    assert {"AC", "DS"} <= names, "field manifest missing AC/DS under signatures=True"
    contig = meta["contigs"][0]
    # fields written
    assert list((out / contig / "fields" / "info" / "AC").glob("*/values.bin"))
    assert list((out / contig / "fields" / "format" / "DS").glob("*/values.bin"))
    # mutcat sidecar written alongside (signatures path ran)
    sidecar = list((out / contig).rglob("*sig*")) + list((out / contig).rglob("*mutcat*"))
    assert sidecar, "signatures=True produced no mutcat sidecar next to fields"
```

- [ ] **Step 2: Run and adjust the sidecar assertion to the real artifact name**

Run: `pixi run pytest tests/test_svar2_fields.py::test_from_vcf_signatures_and_fields_compose -v`
If it fails only on the `sidecar` glob, inspect the store to find the real mutcat artifact name and tighten the assertion:

Run: `pixi run pytest tests/test_svar2_fields.py::test_from_vcf_signatures_and_fields_compose -v -s` then, from a scratch run, list the contig dir. Cross-check the sidecar path against `src/mutcat/sidecar.rs` / `src/mutcat/mod.rs` and replace the `rglob` with the exact filename.
Expected after fix: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar2_fields.py
git commit -m "test(svar2): cover signatures=True composed with INFO/FORMAT fields

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Stage 2 — DRY & cleanup

### Task 4: Replace stringly-typed `prefix: &str` with `FieldCategory` in dense field merge

`merge_dense_field_values` (src/dense_merge.rs:120) takes `prefix: &str` and matches `"finfo"`/`"fformat"` at runtime with an error arm. Replace with the existing `FieldCategory` enum; select the chunk-path builder by variant. Removes the runtime error branch and its test.

**Files:**
- Modify: `src/dense_merge.rs:120` (signature + body + the `"bogus"` error test)
- Modify: `src/orchestrator.rs:337-370` (call site — pass `field.category` instead of the `"finfo"`/`"fformat"` string; drop the `prefix` local)

**Interfaces:**
- Consumes: `crate::field::FieldCategory` (Info/Format), `layout::chunk_field_info` (layout.rs:121), `layout::chunk_field_format` (layout.rs:124).
- Produces: `pub fn merge_dense_field_values(output_dir: &str, num_chunks: usize, dense_ledger: &[u32], category: FieldCategory, field_ix: usize, dest_values_bin: &Path) -> Result<(), ConversionError>`.

- [ ] **Step 1: Change the signature and body**

In `src/dense_merge.rs`, replace `prefix: &str` with `category: crate::field::FieldCategory` and the `match prefix { "finfo" => ..., "fformat" => ..., other => Err(...) }` block with:

```rust
        let path = match category {
            crate::field::FieldCategory::Info => layout::chunk_field_info(dir, c, field_ix),
            crate::field::FieldCategory::Format => layout::chunk_field_format(dir, c, field_ix),
        };
```

Delete the now-impossible `other =>` error arm entirely.

- [ ] **Step 2: Update the call site**

In `src/orchestrator.rs`, in the dense merge loop (around 337-370), drop the `let (prefix, field_ix) = match field.category { ... "finfo" ... "fformat" ... }` string binding; keep the `info_ix`/`format_ix` counters for `field_ix`, and pass `field.category` directly:

```rust
            let field_ix = match field.category {
                crate::field::FieldCategory::Info => {
                    let ix = info_ix;
                    info_ix += 1;
                    ix
                }
                crate::field::FieldCategory::Format => {
                    let ix = format_ix;
                    format_ix += 1;
                    ix
                }
            };
            // ... dest_dir unchanged ...
            crate::dense_merge::merge_dense_field_values(
                dir.to_str().unwrap(),
                num_chunks,
                &ledger,
                field.category,
                field_ix,
                &dest_values_bin,
            )?;
```

- [ ] **Step 3: Fix the unit tests in dense_merge.rs**

Update the two passing tests (dense_merge.rs:325, :358) to pass `FieldCategory::Info` / `FieldCategory::Format` instead of `"finfo"`/`"fformat"`. **Delete** the `"bogus"` error test (dense_merge.rs:378 area) — the invalid state is now unrepresentable, so the test no longer compiles. Add `use crate::field::FieldCategory;` to the test module if needed.

- [ ] **Step 4: Build + run affected tests**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --lib dense_merge'`
Expected: PASS, no `unknown prefix` references remain.

- [ ] **Step 5: Full Rust + Python guard**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion'` then `pixi run pytest tests/test_svar2_fields.py -q`
Expected: all PASS (the same-name INFO/FORMAT test still green — category routing unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/dense_merge.rs src/orchestrator.rs
git commit -m "refactor(svar2): type dense field merge by FieldCategory, not str prefix

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Extract shared offset + tile-gather helper (follow-up #1)

`merge_mini_sc` (src/merge.rs:25) and `merge_var_key_field_values` (src/merge.rs:294) share (a) Phase-A per-column/per-chunk offset derivation and (b) the adaptive-tile, parallel pread→interleave→pwrite gather. The only real difference: `merge_mini_sc` moves two payload streams per item (pos u32 = 4 bytes, key = `key_bytes`), the field merge moves one (`item_width`). Model both as a **list of byte-payload streams** sharing one offset/tile schedule.

**Files:**
- Modify: `src/merge.rs` (add a private helper; rewrite both fns to call it)

**Interfaces:**
- Produces (private to `merge.rs`):
  - `struct Payload<'a> { item_width: usize, chunk_files: &'a [File], dest: &'a File }` (one per byte-stream).
  - `fn derive_offsets(num_chunks: usize, total_columns: usize, ram_ledger: &[Vec<u32>]) -> (Vec<u64>, Vec<Vec<u32>>)` returning `(final_offsets, chunk_offsets)`.
  - `fn gather_columns(total_columns: usize, num_chunks: usize, ram_ledger: &[Vec<u32>], final_offsets: &[u64], chunk_offsets: &[Vec<u32>], payloads: &[Payload]) -> Result<(), ConversionError>` — the tile loop, byte-generic: for each tile allocate one `Vec<u8>` per payload sized `tile_items * item_width`, pread each chunk's slice, interleave column-major via per-column write heads (in items × item_width bytes), pwrite each payload's tile to its byte range.
- Consumes: `TILE_RAM_BUDGET_BYTES`, `layout::*`, `read_exact_at`/`write_all_at` (already used).

Note on positions: `merge_mini_sc` currently casts pos bytes to `&[u32]` for the copy, but the copy is a straight `copy_from_slice`; treating positions as raw 4-byte items in `gather_columns` is byte-identical. The `offsets.npy` write and the Phase-C chunk-file cleanup stay in `merge_mini_sc` (they are pos/key-specific and not shared).

- [ ] **Step 1: Add `derive_offsets` and unit-test it against the inline arithmetic**

Add `derive_offsets` to `src/merge.rs`, then add a test asserting it reproduces the existing inline result for a small ledger:

```rust
    #[test]
    fn derive_offsets_matches_inline() {
        // 2 chunks, 3 columns
        let ledger = vec![vec![2u32, 0, 1], vec![1u32, 3, 0]];
        let (final_offsets, chunk_offsets) = derive_offsets(2, 3, &ledger);
        assert_eq!(final_offsets, vec![0, 3, 6, 7]); // col totals 3,3,1
        assert_eq!(chunk_offsets[0], vec![0, 2, 2, 3]);
        assert_eq!(chunk_offsets[1], vec![0, 1, 4, 4]);
    }
```

- [ ] **Step 2: Run the offset test**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --lib merge::tests::derive_offsets_matches_inline'`
Expected: PASS.

- [ ] **Step 3: Add `gather_columns` and route `merge_var_key_field_values` through it**

Rewrite `merge_var_key_field_values` to: `derive_offsets(...)`, create+`set_len` the dest file, open chunk field files, then call `gather_columns` with a single `Payload { item_width, chunk_files: &field_files, dest: &dest_file }`. Keep its existing behavior (no offsets.npy, no chunk cleanup — those belong to the pos/key merge).

- [ ] **Step 4: Verify var_key field merge tests still pass**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --lib merge'`
Expected: PASS — including the existing `merge_var_key_field_values` test (merge.rs:659) and all `merge_mini_sc` tests (still on the old code path at this point).

- [ ] **Step 5: Route `merge_mini_sc` through the shared helpers**

Rewrite `merge_mini_sc`'s Phase A to `derive_offsets(...)`, keep the `offsets.npy` write, then replace its inline tile loop with `gather_columns` given two payloads: `Payload { item_width: 4, chunk_files: &pos_files, dest: &final_pos_file }` and `Payload { item_width: key_bytes, chunk_files: &key_files, dest: &final_key_file }`. Keep Phase-C cleanup (`remove_file` chunk pos/key) after the gather. This requires opening pos and key chunk files as two separate `Vec<File>` (they are currently a `Vec<(File, File)>` — split into two vecs so each becomes a payload's `chunk_files`).

- [ ] **Step 6: Full merge + orchestrator round-trip**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion'`
Expected: ALL PASS — the `merge_mini_sc` tests (merge.rs:504-735) verify the pos/key output byte-for-byte, so this proves the refactor preserved behavior.

- [ ] **Step 7: Python e2e guard**

Run: `pixi run pytest tests/test_svar2_fields.py tests/test_svar2_from_vcf.py -q`
Expected: PASS — real conversions produce identical stores.

- [ ] **Step 8: Commit**

```bash
git add src/merge.rs
git commit -m "refactor(svar2): share offset derivation + tile gather across merges

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Stage 3 — Baseline profile

### Task 6: Capture a fields-enabled baseline (perf + callgrind)

Measurement only — **no code change, no commit**. Establishes the pre-optimization cost of the finalize pass on a workload big enough for it to register.

**Files:**
- Create (throwaway, under `$CLAUDE_JOB_DIR/tmp`, not committed): `gen_big_vcf.py`, `drive_convert.py`

**Interfaces:** none (scripts).

- [ ] **Step 1: Synthesize a moderately large fields-carrying VCF**

Write `$CLAUDE_JOB_DIR/tmp/gen_big_vcf.py` that emits a bgzipped, indexed VCF: 1 contig, ~200 samples, ~50,000 biallelic records (mix ~10% indels), each with `FORMAT DS` (Float, Number=1), `FORMAT DP` (Integer, Number=1), and `INFO AC` (Integer, Number=A). Use `vcfixture.VcfBuilder` if it scales, else write the text VCF directly and `bgzip`/`bcftools index`. This makes finalize handle millions of staged FORMAT calls.

Run: `pixi run python $CLAUDE_JOB_DIR/tmp/gen_big_vcf.py $CLAUDE_JOB_DIR/tmp/big.vcf.gz`
Expected: `big.vcf.gz` + index produced.

- [ ] **Step 2: Build the profiling extension**

Run: `pixi run bash -lc 'RUSTFLAGS="-C force-frame-pointers=yes" maturin develop --profile profiling'`
Expected: builds `genoray_core` with release opt + debuginfo.

- [ ] **Step 3: Write the driver**

Write `$CLAUDE_JOB_DIR/tmp/drive_convert.py`:

```python
import sys, shutil
from pathlib import Path
from genoray import SparseVar2
from genoray._svar2_fields import FormatField

src, out = sys.argv[1], Path(sys.argv[2])
if out.exists():
    shutil.rmtree(out)
SparseVar2.from_vcf(
    out, src, no_reference=True, threads=1,
    info_fields=["AC"], format_fields=[FormatField("DS", default=0.0), "DP"],
)
```

(`threads=1` keeps callgrind's serial model honest; the finalize parallelism win is measured separately via wall-time in Task 7.)

- [ ] **Step 4: perf record + report**

Run:
```bash
pixi run bash -lc 'cd $CLAUDE_JOB_DIR/tmp && perf record -g --call-graph fp -o perf.data -- python drive_convert.py big.vcf.gz store.svar2 && perf report -i perf.data --stdio | head -60'
```
Expected: a symbolized profile. Record the % of samples inside `finalize`, `read_staged`, `scan`, `rewrite_file`, `encode`, and `merge_dense_field_values`.

- [ ] **Step 5: callgrind for instruction counts**

Run:
```bash
pixi run bash -lc 'cd $CLAUDE_JOB_DIR/tmp && valgrind --tool=callgrind --callgrind-out-file=cg.baseline.out python drive_convert.py big.vcf.gz store_cg.svar2 && callgrind_annotate cg.baseline.out | head -60'
```
Expected: instruction-retired breakdown. Record `Ir` totals for the finalize functions above. (Callgrind is slow — a smaller VCF, e.g. 50 samples × 10k records, is acceptable for the instruction ratio; note the size used.)

- [ ] **Step 6: Record baseline wall time + peak RSS**

Run:
```bash
pixi run bash -lc 'cd $CLAUDE_JOB_DIR/tmp && /usr/bin/time -v python drive_convert.py big.vcf.gz store_t.svar2 2>&1 | grep -E "Elapsed|Maximum resident"'
```
Expected: baseline wall-clock + peak RSS. **Write all baseline numbers into `$CLAUDE_JOB_DIR/tmp/BASELINE.md`** for Task 9's before/after table. No commit.

---

## Stage 4 — Optimize (byte-identical, guarded)

### Task 7: Streaming + parallel finalize

Cut the finalize pass's cost per Task 6's profile. Three changes, all byte-identical:
1. **No `Vec<f64>` materialization** — scan and encode iterate over the raw byte buffer via `chunks_exact(4)`, never collecting an intermediate `Vec<f64>`.
2. **Parallelize** — scan files with rayon and reduce `ScanStats`; rewrite files with rayon; run fields concurrently.
3. **(Profile-gated) single read** — if Task 6 shows disk-read (not decode/encode) dominates, retain each file's raw `Vec<u8>` from the scan pass and reuse it in rewrite to avoid the second `std::fs::read`. If page cache makes reads free (decode/encode dominate), skip this and keep two reads to bound memory to one file per thread.

**Files:**
- Modify: `src/field_finalize.rs` — `finalize_fields` (84), `finalize_one` (95), `scan` (176), `read_staged` (139), `rewrite_file` (359)

**Interfaces:**
- `finalize_fields`/`finalize_one`/`resolve_dtype`/`ResolvedField` signatures unchanged (public shape preserved).
- Internal: replace `read_staged(...) -> Vec<f64>` usage with a byte-iterating helper, e.g. `fn decode_elem(chunk: [u8;4], is_float: bool) -> f64` applied inside `scan`/`rewrite` loops over `bytes.chunks_exact(4)`.

- [ ] **Step 1: Lock byte-identity — add a pre-refactor golden test**

Add a finalize test that stages a mixed field (values forcing an auto-narrow, plus a staged-missing sentinel), captures the finalized bytes, and asserts them against a hard-coded expected byte vector — so any behavior drift in the refactor fails loudly:

```rust
    #[test]
    fn finalize_auto_narrow_byte_golden() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // i32 staged values incl. one MISSING sentinel; auto -> narrow.
        write_i32_field(root, "chr1", "AC", "var_key_snp", &[0, 5, 127, i32::MIN, 3]);
        let field = int_field("AC", StorageDtype::Auto, None);
        let resolved = finalize_fields(root, &[String::from("chr1")], &[field]).unwrap();
        let path = root.join("chr1").join("fields").join("info").join("AC")
            .join("var_key_snp").join("values.bin");
        let got = read_values_bin(&path);
        // Record the ACTUAL bytes+dtype from a run BEFORE editing scan/rewrite,
        // then hard-code them here so the optimization must reproduce them.
        // (Fill `expected` and `resolved[0].dtype` from the pre-refactor output.)
        // assert_eq!(resolved[0].dtype, StorageDtype::__);
        // assert_eq!(got, vec![__]);
        let _ = (resolved, got);
    }
```

Run it once on current code (`... --lib field_finalize::tests::finalize_auto_narrow_byte_golden -- --nocapture`), read the produced dtype+bytes, fill in the `expected`/dtype asserts, and re-run to confirm it passes on the un-optimized code. Commit this test in this step so it guards the next steps.

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --lib field_finalize::tests::finalize_auto_narrow_byte_golden'`
Expected: PASS on current code.

```bash
git add src/field_finalize.rs
git commit -m "test(svar2): golden byte-level guard for finalize auto-narrow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 2: Remove the `Vec<f64>` materialization**

Rewrite `scan` to iterate `bytes.chunks_exact(4)` decoding each element inline (via a small `decode_elem` helper) and folding into `ScanStats`, without building a `Vec<f64>`. Rewrite `rewrite_file` to read the bytes, then iterate `chunks_exact(4)` decoding+`encode`-ing directly into `out`. Keep `read_staged` only if still used; otherwise delete it and its length-check into the new helper (preserve the "not a multiple of 4 bytes" error).

- [ ] **Step 3: Verify byte-identity + all finalize tests**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --lib field_finalize'`
Expected: PASS — golden test + f32-noop + all existing narrow/overflow tests.

- [ ] **Step 4: Parallelize scan/rewrite/fields with rayon**

`finalize_fields`: `fields.par_iter().map(...).collect()`. `scan`: `files.par_iter()` producing per-file `ScanStats`, reduced with a combine (min/max/has_missing/observe merge — add `ScanStats::merge`). `finalize_one` rewrite loop: `files.par_iter().try_for_each(rewrite_file)`. Add `use rayon::prelude::*;`.

- [ ] **Step 5: (Profile-gated) retain bytes to drop the second read**

Only if Task 6 flagged read I/O as dominant: have the scan pass return `(ScanStats, Vec<(PathBuf, Vec<u8>)>)` and feed the retained bytes into rewrite. Otherwise skip — note the decision in the commit body.

- [ ] **Step 6: Full guard (Rust + Python)**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion'` then `pixi run pytest tests/test_svar2_fields.py tests/test_svar2_from_vcf.py -q`
Expected: ALL PASS. Byte-identity holds (golden + round-trip asserts).

- [ ] **Step 7: Commit**

```bash
git add src/field_finalize.rs
git commit -m "perf(svar2): stream + parallelize finalize, drop f64 materialization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: (Conditional) Stream `merge_dense_field_values`

Only if Task 6's profile shows `merge_dense_field_values` (dense_merge.rs:120) is a meaningful share. It currently reads every chunk file fully into one growing `Vec<u8>` before writing. Replace with a buffered chunk→dest copy to bound peak memory.

**Files:**
- Modify: `src/dense_merge.rs:120`

- [ ] **Step 1: Decide from the profile**

If `merge_dense_field_values` is < a few % of instructions/RSS in Task 6, **skip this task** and note "dense field merge not a hotspot — skipped" in Task 9's writeup. Otherwise continue.

- [ ] **Step 2: Stream the concatenation**

Open `dest` once; for each non-empty chunk (per `dense_ledger`), open the chunk file and `std::io::copy` it into a `BufWriter<File>` over dest, then remove the chunk file (preserve the existing skip-empty + cleanup semantics). Do not hold all bytes in one `Vec`.

- [ ] **Step 3: Verify**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --lib dense_merge'` then `pixi run pytest tests/test_svar2_fields.py -q`
Expected: PASS (dense field bytes identical).

- [ ] **Step 4: Commit**

```bash
git add src/dense_merge.rs
git commit -m "perf(svar2): stream dense field merge to bound peak memory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Stage 5 — cargo asm, re-profile & document

### Task 9: Inspect codegen, re-profile, and write it up

**Files:**
- Modify: `CHANGELOG.md` (append under `## Unreleased`)
- Verify (no edit expected): `skills/genoray-api/SKILL.md`

- [ ] **Step 1: `cargo asm` on the hot inner loops**

Inspect generated assembly for the per-element encode and the field-routing loop to confirm no surprise bounds checks / spills after the refactor:

```bash
pixi run bash -lc 'cargo asm --no-default-features --features conversion --rust genoray_core::field_finalize::encode 2>&1 | head -80'
pixi run bash -lc 'cargo asm --no-default-features --features conversion --rust genoray_core::rvk::dense2sparse_vk 2>&1 | head -120'
```
Expected: readable asm. If `cargo asm` can't resolve the path, list candidates with `cargo asm --no-default-features --features conversion 2>&1 | grep -iE "encode|dense2sparse"` and use the exact printed symbol. Note any bounds-check in the innermost encode loop; if present and cheap to remove (e.g. via `chunks_exact`/slice pre-slicing), do so and re-run Task 7 Step 6's guard.

- [ ] **Step 2: Re-profile with the same Task 6 workload**

Rebuild (`RUSTFLAGS="-C force-frame-pointers=yes" maturin develop --profile profiling`) and re-run the perf, callgrind, and `/usr/bin/time -v` commands from Task 6 against the same `big.vcf.gz`. Capture finalize `Ir`, wall time, and peak RSS.

- [ ] **Step 3: Build the before/after table**

Compare against `$CLAUDE_JOB_DIR/tmp/BASELINE.md`. Compute deltas for: finalize callgrind `Ir`, total wall time, peak RSS. Confirm no regression in the reader path's share.

- [ ] **Step 4: CHANGELOG entry**

Append a human-readable line under `## Unreleased` in `CHANGELOG.md`, e.g.:

```markdown
- SVAR2 field write: finalize pass streams staged values (no intermediate f64
  materialization) and runs in parallel across fields/files; DRY'd the var_key
  and pos/key merges onto a shared tile-gather. Field output is byte-identical.
  (finalize: <X>% fewer instructions, peak RSS <Y>→<Z>.)
```

Fill `<X>/<Y>/<Z>` from Step 3.

- [ ] **Step 5: Confirm no public API drift**

Verify no `import genoray` name changed (all edits were private modules `merge`/`dense_merge`/`field_finalize`, internal-only). Confirm `skills/genoray-api/SKILL.md` needs no change:

Run: `git diff --name-only main...HEAD -- python/ | grep -v '_' || echo 'no public python surface touched'`
Expected: no public (non-underscore) module changes → SKILL.md untouched. State this explicitly in the commit body.

- [ ] **Step 6: Commit + push + update PR**

```bash
git add CHANGELOG.md
git commit -m "docs(svar2): changelog for field finalize/merge cleanup + perf

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

Then post the before/after table as a PR#100 comment (`gh pr comment 100 --body ...`). Do not merge.

---

## Notes

Two untracked doc files from the PR#100 session sit in the worktree
(`docs/superpowers/specs/2026-07-11-svar1-to-svar2-conversion-design.md`,
`docs/superpowers/plans/2026-07-11-svar2-info-format-fields-write.md`). Left
untouched unless the user asks to commit them.

# SVAR2 `write_view(reroute=False)` array-slicer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `SparseVar2.write_view(..., reroute=False)` as a representation-preserving array-slicer that produces a region/sample subset by slicing a finished store's sidecars directly — low memory (O(output)), fields carried natively, `mutcat` recomputed from `reference`.

**Architecture:** A new per-contig Rust slicer (`src/svar2_slice.rs`) reads the source sidecars (`ContigReader` views), selects kept var_key calls / dense rows with the SAME overlap predicate as `Svar2Source`, and writes new sidecars via reusable low-level helpers (`pack_snp_keys`, `bits::*`, `write_npy`/`cast_slice`), then reuses the standalone `max_del::write_max_del` and `mutcat::annotate::annotate_contig` post-passes and a verbatim LUT copy. A `run_slice_view` pyfunction and a Python `write_view(reroute=False)` branch drive it.

**Tech Stack:** Rust (pyo3, memmap2, ndarray, ndarray-npy, bytemuck), Python (numpy, cyclopts), pixi, pytest, cargo.

**Spec:** `docs/superpowers/specs/2026-07-13-svar2-reroute-false-slicer-design.md`

## Global Constraints

- **Rust tests:** `pixi run -e lint test-rust` (= `cargo test --no-default-features --features conversion`). New pyfunction gated `#[cfg(feature="conversion")]` (needs `load_contig_seq`). Never bare `cargo test`.
- **Cargo-on-NFS:** `export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit` (warm, stable) before any commit triggering cargo hooks; run the commit in the background if the incremental hook build might exceed the 2-min foreground limit.
- **Rebuild after Rust edits:** `pixi run -e py310 maturin develop --release` (foreground) before Python tests exercising new Rust.
- **Python tests:** `pixi run -e py310 pytest <path>`.
- **Clippy `-D warnings`:** preempt `needless_range_loop` (`.iter().enumerate()`) and `type_complexity` (`#[allow(clippy::type_complexity)]`).
- **Conventional Commits.**
- **Public-API rule:** `reroute=False` becomes functional → update `skills/genoray-api/SKILL.md` + `CHANGELOG.md` (Task 5).
- **Match `Svar2Source` overlap semantics exactly** (`src/svar2_source.rs:46-60,110-144`): `Pos` `q_start≤pos<q_end`; `Record` `q_start≤pos≤q_end` + reader window `+1`; `Variant` extent-overlap no POS filter. The `reroute=False` and `reroute=True` variant sets MUST be identical.
- **No pipeline / no cost model / no `RawRecord`** in this path — it is a direct sidecar slicer. Memory must stay O(output).

---

## File Structure

- **Create** `src/svar2_slice.rs` — the per-contig slicer (Tasks 1, 2). Its own module keeps `svar2_source.rs` (the `reroute=True` source) untouched.
- **Modify** `src/lib.rs` — `run_slice_view` pyfunction + registration + meta write + mutcat post-pass (Task 3).
- **Modify** `src/svar2_source.rs` OR a shared module — factor the `OverlapMode` keep predicate so the slicer reuses it verbatim (Task 1, Step 1).
- **Modify** `python/genoray/_svar2.py` — `write_view(reroute=False)` branch (Task 4).
- **Test** `tests/test_svar2_slice.rs` (Rust), `tests/test_svar2_write_view.py` (Python).
- **Modify** `skills/genoray-api/SKILL.md`, `CHANGELOG.md`, `docs/roadmap/data-model.md` (Task 5).

---

## Task 1: Genotype slicer — var_key + dense + LUT + max_del (Rust)

**Files:**
- Create: `src/svar2_slice.rs`
- Modify: `src/svar2_source.rs` (make `OverlapMode` + keep predicate reusable) and `src/lib.rs`/`mod` wiring so `svar2_slice` compiles.
- Test: `tests/test_svar2_slice.rs`

**Interfaces:**
- Consumes: `query::ContigReader` (`open`, `vk_snp`/`vk_indel`/`dense_snp`/`dense_indel` views), `svar2_codec::{pack_snp_keys, unpack_snp_key_at, encode_snp_2bit}`, `bits::{set_bit, get_bit, copy_bits, for_each_set_bit}`, `max_del::write_max_del(contig_dir:&Path, n_samples, ploidy)`, `layout::ContigPaths`, `svar2_source::OverlapMode`.
- Produces: `pub fn slice_contig_genos(src_store:&str, out_store:&str, chrom:&str, sample_orig_idx:&[usize], ploidy:usize, regions:&[(u32,u32)], overlap:OverlapMode) -> Result<usize, ConversionError>` — writes `{out_store}/{chrom}/var_key/*`, `dense/*`, copies the LUT, runs `write_max_del`; returns the number of variants written. (Fields + mutcat + meta come in later tasks.)

- [ ] **Step 1: Factor the overlap keep predicate.** Extract from `Svar2Source::new` (`src/svar2_source.rs:110-144`) a reusable helper, e.g. `pub fn query_window(regions:&[(u32,u32)], m:OverlapMode) -> Vec<(u32,u32)>` (the `Record` `+1` widening) and `pub fn keeps(m:OverlapMode, q_start:u32, q_end:u32, pos:u32) -> bool`. Make `OverlapMode` `pub`. Replace the inline logic in `Svar2Source::new` with these (no behavior change — run the existing `test_svar2_source` suite to confirm).

- [ ] **Step 2: Write the failing byte-parity test.** In `tests/test_svar2_slice.rs`: build a small store with var_key + dense, SNP + indel variants (via a Python fixture builder, or reuse the `test_svar2_source` store builder); slice it over the FULL region + ALL samples; assert every `var_key/*` and `dense/*` and `max_del.npy` file is byte-identical to the source (a full-coverage identity slice), and the LUT files are byte-identical.

```rust
#[test]
fn slice_full_coverage_is_byte_identical_genos() {
    // src = build_store(...); out = tmp;
    let n = slice_contig_genos(src, out, "chr1", &(0..n_samples).collect::<Vec<_>>(),
                               2, &[(0, u32::MAX)], OverlapMode::Variant).unwrap();
    for rel in ["var_key/snp/positions.bin","var_key/snp/alleles.bin","var_key/snp/offsets.npy",
                "var_key/indel/positions.bin","var_key/indel/alleles.bin","var_key/indel/offsets.npy",
                "dense/snp/positions.bin","dense/snp/alleles.bin","dense/snp/genotypes.bin",
                "dense/indel/positions.bin","dense/indel/alleles.bin","dense/indel/genotypes.bin",
                "max_del.npy","dense/max_del.npy",
                "indel/long_alleles.bin","indel/long_allele_offsets.npy"] {
        assert_eq!(read_if_exists(src, rel), read_if_exists(out, rel), "{rel}");
    }
}
```

- [ ] **Step 3: Run — verify it fails** (`slice_contig_genos` undefined).

Run: `export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$; pixi run -e lint bash -lc 'cargo test --no-default-features --features conversion --test test_svar2_slice slice_full_coverage'`
Expected: FAIL.

- [ ] **Step 4: Implement `slice_contig_genos`.** Per the spec §A(1-3,5-6):
  - Open source `ContigReader::open(src_store, chrom, n_samples_orig, ploidy)`.
  - **var_key:** for each output column `c_out = s_out*ploidy+p` (source column `sample_orig_idx[s_out]*ploidy+p`), take its call range from the source CSR (`SubStreamView::column`), keep calls passing `keeps(...)` on `positions[i]`; append to new `positions` and new `alleles` (indel: copy u32 key verbatim; snp: `unpack_snp_key_at` source code, push to a code Vec); record per-column count → new `offsets` prefix-sum (len `n_subset*ploidy+1`). After gathering all snp codes, `pack_snp_keys(&codes)` → `alleles.bin`. Write `positions.bin` (`bytemuck::cast_slice`), `offsets.npy` (`write_npy(Array1::from_vec(offsets_u64))`), `alleles.bin`, into `layout` paths.
  - **dense:** for each dense variant row, keep if region-overlapping (`keeps`) AND ≥1 subset hap carries it (`DenseView::for_each_carried`/`get_bit` over subset haps); copy `positions[row]`/`keys[row]` (snp re-packed), and set new genotype bits `hap_out*n_dense_kept + row_out` via `bits::set_bit`. Write the three dense files.
  - **LUT:** `fs::copy` `indel/long_alleles.bin` + `indel/long_allele_offsets.npy` if present.
  - **max_del:** `max_del::write_max_del(out_contig_dir, n_subset, ploidy)`.
  - Create output dirs via `ContigPaths`.

- [ ] **Step 5: Run — verify it passes.** Same command. Expected: PASS. Then a **subset** test: slice a 1-sample subset, open the output `ContigReader`, decode, and assert genotypes match the source decoded over that sample+region (equivalence).

- [ ] **Step 6: Commit.**
```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add src/svar2_slice.rs src/svar2_source.rs src/lib.rs tests/test_svar2_slice.rs
git commit -m "feat(svar2): reroute=False genotype array-slicer (var_key/dense/LUT/max_del)"
```

---

## Task 2: Field slicing (Rust)

**Files:**
- Modify: `src/svar2_slice.rs`
- Test: `tests/test_svar2_slice.rs`

**Interfaces:**
- Consumes: `query::field::FieldView` (or read `fields/*/values.bin` directly with the dtype width from the passed manifest); `field::StorageDtype::width_bytes`.
- Produces: `slice_contig(src, out, chrom, sample_orig_idx, ploidy, regions, overlap, fields:&[FieldSpec]) -> Result<usize>` — extends `slice_contig_genos` to also slice each field's `values.bin`. (Rename Task 1's fn to `_genos` internal, or add `fields` param.)

- [ ] **Step 1: Write the failing test.** Build a store with an INFO field (`AF`, float, dense) and a FORMAT field (`DP`, int, var_key+dense). Full-coverage slice with those fields; assert each `fields/{cat}/{name}/{sub}/values.bin` is byte-identical to source. Then a subset test: field values decode-equal to the source subset.

- [ ] **Step 2: Run — verify it fails** (`slice_contig` ignores fields).

Run: `... --test test_svar2_slice slice_fields`
Expected: FAIL.

- [ ] **Step 3: Implement.** During the var_key/dense gather (Task 1), also gather field bytes at the field's element width (`StorageDtype::width_bytes`):
  - var_key `values.bin`: one element per kept call, gathered in the SAME new call order (parallel index map to positions/keys).
  - dense INFO `values.bin`: one element per kept row.
  - dense FORMAT `values.bin`: for each kept row, for each subset `orig_sample`, copy source element `row*n_samples_orig + orig_sample` → new layout `row_out*n_subset + s_out`.
  Write each into its `layout::field_values(cat,name,sub)` path. Skip a field/sub whose source `values.bin` is missing/empty (legal empty sub-stream).

- [ ] **Step 4: Run — verify it passes.** `... --test test_svar2_slice`. Expected: PASS.

- [ ] **Step 5: Rebuild + commit.**
```bash
pixi run -e py310 maturin develop --release
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add src/svar2_slice.rs tests/test_svar2_slice.rs
git commit -m "feat(svar2): reroute=False slices INFO/FORMAT field sidecars"
```

---

## Task 3: `run_slice_view` pyfunction + meta + mutcat post-pass (Rust)

**Files:**
- Modify: `src/lib.rs`
- Test: `tests/test_svar2_slice.rs` (or a new `tests/test_slice_view.rs`)

**Interfaces:**
- Produces (pyo3): `run_slice_view(store_path, out_dir, contigs:Vec<String>, samples:Vec<String>, regions:Vec<(String,u32,u32)>, regions_overlap:String, merge_overlapping:bool, fields:Vec<(String,String,String,Option<f64>)>, reference:Option<String>, overwrite:bool) -> PyResult<()>`. Registered in the `_core` pymodule.

- [ ] **Step 1: Write the failing test.** Build a source store (with fields + mutcat via `from_vcf(signatures=True, info_fields=…, format_fields=…)`); call `run_slice_view` over full region/all samples with `reference=<fasta>`; assert (a) output `meta.json` has subset samples + same fields; (b) all sidecars byte-parity; (c) `mutcat/*/code.bin` present and byte-identical to source (full-coverage identity).

- [ ] **Step 2: Run — verify it fails.**

Run: `... --test test_svar2_slice run_slice_view`
Expected: FAIL.

- [ ] **Step 3: Implement.** Add the pyfunction (`#[cfg(feature="conversion")]`, `#[allow(clippy::type_complexity)]`):
  - Resolve `samples` → `sample_orig_idx` via source `meta.json`; resolve `regions` per contig (normalize + `merge_overlapping`); map `regions_overlap` → `OverlapMode` (reuse `run_view_pipeline`'s match).
  - Build `Vec<FieldSpec>` from the tuples (concrete dtype via `StorageDtype::from_meta_str`; htype from dtype).
  - Create `out_dir` (respect `overwrite`); for each contig call `svar2_slice::slice_contig(...)`.
  - If `reference.is_some()`: for each output contig, `load_contig_seq(fasta, chrom)`, `ContigReader::open(out_dir, chrom, n_subset, ploidy)`, `annotate::annotate_contig(&reader, &paths, &ref_seq)`; stamp the mutcat presence keys into `meta.json` to match `from_vcf(signatures=True)`.
  - Write `meta.json` via `meta::write_meta` with subset samples, kept contigs, unchanged ploidy/fields.
  - Register `run_slice_view` in the `_core` pymodule.

- [ ] **Step 4: Run — verify it passes.** `... --test test_svar2_slice`. Expected: PASS.

- [ ] **Step 5: Rebuild + commit.**
```bash
pixi run -e py310 maturin develop --release
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add src/lib.rs tests/test_svar2_slice.rs
git commit -m "feat(svar2): run_slice_view pyfunction (meta + mutcat recompute)"
```

---

## Task 4: Python `write_view(reroute=False)` (Python)

**Files:**
- Modify: `python/genoray/_svar2.py` (`write_view`, `~326-440`)
- Test: `tests/test_svar2_write_view.py`

**Interfaces:**
- Consumes: `_core.run_slice_view` (Task 3); `self.available_fields`, `self.available_samples`, `self.contigs`, `self.ploidy`.

- [ ] **Step 1: Write the failing tests.**
```python
def test_reroute_false_equivalent_to_true(tmp_path, svar2_store_with_fields):
    src = svar2_store_with_fields
    a = tmp_path/"t.svar2"; b = tmp_path/"f.svar2"
    samples = src.available_samples[:2]
    src.write_view((CONTIG,0,END), samples, a, fields=["AF","DP"], reroute=True, overwrite=True)
    src.write_view((CONTIG,0,END), samples, b, fields=["AF","DP"], reroute=False, overwrite=True)
    # decoded genotypes + field values identical; representation may differ
    assert_decoded_equal(SparseVar2(a,fields=["AF","DP"]), SparseVar2(b,fields=["AF","DP"]), CONTIG, (0,END))

def test_reroute_false_preserves_representation(tmp_path, svar2_store):
    out = tmp_path/"f.svar2"
    svar2_store.write_view((CONTIG,0,END), svar2_store.available_samples, out, reroute=False, overwrite=True)
    # a source-dense variant stays dense (unlike reroute=True which may re-route)
    import genoray._core as c
    _ii, sd_src, *_ = c.svar2_variant_stats(str(svar2_store.path), CONTIG, list(range(svar2_store.n_samples)))
    _ii2, sd_out, *_ = c.svar2_variant_stats(str(out), CONTIG, list(range(svar2_store.n_samples)))
    assert int(sd_src.sum()) == int(sd_out.sum())  # same dense count = representation preserved

def test_reroute_false_recomputes_signatures(tmp_path, svar2_store, ref_fasta): ...
def test_reroute_false_dtype_preserved(tmp_path, svar2_store_u16_field): ...
def test_reroute_false_lut_indels_decode(tmp_path, svar2_store_long_indels): ...
```

- [ ] **Step 2: Run — verify they fail** (`reroute=False` raises `NotImplementedError`).

Run: `pixi run -e py310 pytest tests/test_svar2_write_view.py -k reroute_false -x`
Expected: FAIL.

- [ ] **Step 3: Implement.** In `write_view`: when `reroute is False` (after the `"auto"`→`True` resolution), call `_core.run_slice_view(...)` with the resolved field tuples and `reference`, instead of raising. Keep `reroute=True`/`"auto"` on `run_view_pipeline`. Reuse the field-resolution + `mutcat`-without-`reference` guard from the shared code path. Update the docstring (`reroute=False` now implemented; usage guidance: somatic/rare + low-memory). Update `genoray view --no-reroute` CLI help.

- [ ] **Step 4: Run — verify they pass.**

Run: `pixi run -e py310 pytest tests/test_svar2_write_view.py -x -q`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit.**
```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-commit
git add python/genoray/_svar2.py tests/test_svar2_write_view.py
git commit -m "feat(svar2): write_view(reroute=False) representation-preserving view"
```

---

## Task 5: Docs

**Files:** `skills/genoray-api/SKILL.md`, `CHANGELOG.md`, `docs/roadmap/data-model.md`

- [ ] **Step 1: `SKILL.md`** — `write_view(reroute=False)` now implemented: representation-preserving, low-memory, carries fields, recomputes `mutcat` from `reference`. Guidance: recommended for somatic/all-rare + memory-constrained; `reroute=True` (default, `"auto"`) is size-optimal; `reroute=False` output can be up to ~6.6% larger for aggressive germline sample-subsets.
- [ ] **Step 2: `CHANGELOG.md`** `## Unreleased` entry (no version bump).
- [ ] **Step 3: `docs/roadmap/data-model.md`** M9 — `reroute=False` implemented; note remaining deferred (LUT compaction; `reroute=True` field carry-through; streaming source).
- [ ] **Step 4: Commit.**
```bash
git add skills/genoray-api/SKILL.md CHANGELOG.md docs/roadmap/data-model.md
git commit -m "docs(svar2): document write_view(reroute=False)"
```

---

## Self-review notes (plan author)

- **Spec coverage:** §A(1-3,5-6) genotypes+LUT+max_del → Task 1; §A(4) fields → Task 2; §B pyfunction + §A(6) mutcat + meta → Task 3; §C Python → Task 4; API/docs → Task 5. Byte-parity/equivalence/field/representation/signature/overlap/LUT tests all placed. ✓
- **Type consistency:** `slice_contig_genos` (T1) → `slice_contig(..., fields)` (T2) → `run_slice_view` tuple `fields` (T3) → Python tuple pass (T4). `OverlapMode`/`keeps`/`query_window` shared by slicer + `Svar2Source` (T1 S1). `write_max_del`/`annotate_contig`/`pack_snp_keys`/`bits::*` are existing signatures. ✓
- **Memory invariant:** every task gathers per-contig/column into `Vec`s sized to the OUTPUT, never the source cohort × variants — O(output), the whole point of `reroute=False`. ✓

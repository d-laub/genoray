# FORMAT route-before-densify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the residual O(N²) in `from_vcf_list` by carrying FORMAT values carrier-sparse into `rvk` (mirroring how PR #121 carried genotype `Carriers`), instead of densifying them to an `F × chunk_size × N` grid that is then read back only at carrier positions.

**Architecture:** Add `DenseChunk.format_by_carrier: Option<Vec<Arc<FormatVals>>>`, a sibling to the existing `carriers`. When a chunk is carrier-bearing (the k-way merge over single-sample VCFs), `chunk_assembler` moves each variant's already-sparse `Arc<FormatVals>` into that field and **skips building `format_staged`**; `rvk`'s two FORMAT consumers (`emit_call`'s `VarKey` arm and `route_variants`' dense second-pass fill) read from it via the existing `resolve_format`. Natively-dense sources (multi-sample VCF, PGEN) are unchanged: `format_by_carrier = None`, `format_staged` as today.

**Tech Stack:** Rust (rust-htslib, rayon, bytemuck), PyO3, pixi, pytest, proptest.

## Global Constraints

- **Byte-identical output.** Every change must keep the store bit-for-bit identical for the same input. The Rust differential test (`carrier_driven_emission_matches_the_grid_scan_exactly`) and the Python cross-path oracle (`test_from_vcf_list_no_reference_matches_bcftools_merge_oracle`) are the gates; both must stay green through every commit.
- **Conventional Commits** (`feat:`, `fix:`, `perf:`, `test:`, `refactor:`, `docs:`) — CI's commitizen owns `CHANGELOG.md`; never edit it.
- **Rust tests run with** `pixi run bash -lc 'cargo test --no-default-features --features conversion ...'` — the default-feature pyo3 test binary fails to link (`undefined symbol: _Py_Dealloc`).
- **On NFS**, set `export CARGO_TARGET_DIR=/tmp/genoray_<uniq>` before any `cargo` invocation (NFS `target/` bus-errors on mmap). `/tmp` is reaped — park bench artifacts in `$CLAUDE_JOB_DIR/tmp`.
- **`test-rust <arg>` filters by test NAME, not file** — a nonmatching arg vacuously passes 0 tests. Use `--test <file>` or a known test-name substring, and confirm the run count is non-zero.
- **No public API change** in this plan (`format_by_carrier` is internal; `max_mem`/`chunk_size` already shipped). If any task finds itself touching a name reachable from `import genoray` without an underscore, STOP and update `skills/genoray-api/SKILL.md` in the same commit (repo rule).
- **prek hooks** must be installed (`pixi run prek-install`) before committing.

---

## File Structure

- `src/types.rs` — `DenseChunk` struct: add `format_by_carrier` field + `Arc`/`FormatVals` imports.
- `src/chunk_assembler.rs` — `read_next_chunk`: build `format_by_carrier` and make `format_staged` conditional on carrier-bearing; fix the now-stale O(N²) comments.
- `src/rvk.rs` — `emit_call` (gains a `format_specs` param + FORMAT source switch); `route_variants` dense-fill switch; test helpers (`private_chunk`) + the differential test; a new read-switch unit test.
- `src/executor.rs` — one test `DenseChunk` literal gains the new field.
- `tests/test_svar2_from_vcf_list.py` — extend the bcftools-merge oracle with FORMAT fields + a second contig.

Tasks are largely sequential on one seam (types → read-switch → conditional staging → tests). Task 1 is pure scaffolding folded into the first behavioral change's prerequisites; Tasks 4 and 5 (Rust differential strengthening, Python oracle) are independent of each other once Task 3 lands and could be done in parallel.

---

### Task 1: Add `format_by_carrier` to `DenseChunk` and populate it (no behavior change)

Adds the field, defaults it `None` at every construction site, and has `chunk_assembler` fill it for carrier-bearing chunks — while STILL building `format_staged`. Nothing reads the new field yet, so output is unchanged. This isolates the struct/wiring churn from the behavioral switch in Task 2.

**Files:**
- Modify: `src/types.rs:1-2` (imports), `src/types.rs:136-167` (`DenseChunk`)
- Modify: `src/chunk_assembler.rs:718-783` (`read_next_chunk` metadata pass + `DenseChunk` construction)
- Modify: `src/executor.rs:76-88` (test literal), `src/rvk.rs:732-743` (test literal)

**Interfaces:**
- Produces: `DenseChunk.format_by_carrier: Option<Vec<Arc<FormatVals>>>` — `Some(v)` with `v.len() == v_variants` iff `carriers.is_some()`; `None` otherwise. Entry `v` is the `Arc<FormatVals>` for variant `v` (chunk order), always the `ByCarrier` variant when `Some`.

- [ ] **Step 1: Add imports to `src/types.rs`**

At the top of `src/types.rs`, extend the `record_source` import and add `Arc`:

```rust
use crate::record_source::{Carriers, FormatVals};
use std::sync::Arc;
```

(The file currently has `use crate::record_source::Carriers;` at line 1 — replace it with the two lines above.)

- [ ] **Step 2: Add the field to `DenseChunk`**

In `src/types.rs`, immediately after the `carriers` field (currently ends at line 166), add:

```rust
    /// Per-variant FORMAT values, carrier-sparse, one entry per variant in
    /// chunk order. `Some` iff `carriers.is_some()` (the k-way merge over
    /// single-sample VCFs); `None` for natively dense sources (multi-sample
    /// VCF, PGEN), which keep `format_staged`. `carriers` and this are two
    /// carrier-sparse encodings of the same chunk, keyed differently:
    /// `carriers` by haplotype column (genotype presence), this by sample
    /// (FORMAT is per-sample, not per-haplotype). When `Some`, `format_staged`
    /// is left empty (see chunk_assembler.rs) and `rvk` resolves FORMAT from
    /// here instead.
    pub format_by_carrier: Option<Vec<Arc<FormatVals>>>,
```

- [ ] **Step 3: Populate it in `chunk_assembler.rs` (still building `format_staged`)**

In `src/chunk_assembler.rs`, the metadata pass loop (currently lines 719-749) takes each atom's carriers via `carrier_opts.push(a.carriers.take())`. Alongside it, collect the FORMAT Arcs. After the `let mut carrier_opts: Vec<Option<Carriers>> = Vec::with_capacity(metas.len());` line (721), add:

```rust
        let mut format_arcs: Vec<Arc<FormatVals>> = Vec::with_capacity(metas.len());
```

Inside the `for a in metas.iter_mut()` loop, immediately before `carrier_opts.push(a.carriers.take());` (line 748), add:

```rust
            format_arcs.push(Arc::clone(&a.format_vals));
```

Then, where `carriers` is derived from `carrier_opts` (the `all_some`/`all_none` block, lines 756-771), add a sibling right after that block:

```rust
        // `format_by_carrier` is Some/None in lockstep with `carriers`: both
        // come from a carrier-bearing source or neither does (the `all_some`/
        // `all_none` uniformity asserted above).
        let format_by_carrier = if all_some { Some(format_arcs) } else { None };
```

Finally, add the field to the `DenseChunk { .. }` literal (currently lines 773-783), after `carriers,`:

```rust
            carriers,
            format_by_carrier,
```

- [ ] **Step 4: Add the field to the two test literals**

`src/executor.rs` — in `one_snp_chunk` (the literal ending at line 88), after `carriers: None,` add:

```rust
            carriers: None,
            format_by_carrier: None,
```

`src/rvk.rs` — in `build_test_chunk` (literal ending at line 742), after `carriers: None,` add:

```rust
            carriers: None,
            format_by_carrier: None,
```

- [ ] **Step 5: Build and run the full Rust suite to verify no behavior change**

Run:
```bash
export CARGO_TARGET_DIR=/tmp/genoray_$USER_$$
pixi run bash -lc "cargo test --no-default-features --features conversion --lib"
```
Expected: PASS, non-zero test count. The new field compiles everywhere; nothing reads it, so all existing tests (including `carrier_driven_emission_matches_the_grid_scan_exactly`) stay green.

- [ ] **Step 6: Commit**

```bash
git add src/types.rs src/chunk_assembler.rs src/executor.rs src/rvk.rs
git commit -m "refactor(svar2): carry format_by_carrier on DenseChunk (unused)"
```

---

### Task 2: Read FORMAT from `format_by_carrier` in `rvk` (the read switch)

Make both FORMAT consumers prefer `format_by_carrier` when present. TDD: a unit test builds a carrier-bearing chunk with a **deliberately wrong** `format_staged` and a **correct** `format_by_carrier`, and asserts the emitted FORMAT matches `format_by_carrier` — proving the read no longer comes from the grid. This test stays as the permanent guard.

**Files:**
- Modify: `src/rvk.rs:168-213` (`emit_call` signature + `VarKey` FORMAT branch)
- Modify: `src/rvk.rs:399-427` (`route_variants` dense second-pass fill)
- Modify: `src/rvk.rs:516-527`, `src/rvk.rs:653-664` (the two `emit_call` call sites — pass `format_specs`)
- Test: `src/rvk.rs` tests module (new `carrier_format_read_prefers_format_by_carrier`)

**Interfaces:**
- Consumes: `DenseChunk.format_by_carrier` (Task 1); `resolve_format(&FormatVals, &FieldSpec, u16, usize, usize) -> f64` and `record_source::CarrierFormat` (existing).
- Produces: `emit_call(route, chunk, v, s, hap, num_samples, per_cat, format_specs, streams, dense, counts)` — one new `format_specs: &[&FieldSpec]` parameter inserted after `per_cat`.

- [ ] **Step 1: Write the failing unit test**

Add to the `tests` module in `src/rvk.rs` (near `carrier_driven_emission_matches_the_grid_scan_exactly`). It reuses `private_chunk`/`two_format_fields`, but overwrites `format_by_carrier` with known-correct values and `format_staged` with garbage:

```rust
    // The carrier path must resolve FORMAT from `format_by_carrier`, NOT the
    // `format_staged` grid. Build a carrier-bearing chunk whose grid is
    // deliberately wrong and whose carrier FORMAT is correct; the emitted
    // field values must match the carrier FORMAT.
    #[test]
    fn carrier_format_read_prefers_format_by_carrier() {
        use crate::record_source::{CarrierFormat, FormatVals};
        use std::sync::Arc;

        let v_variants = 8usize;
        let n_samples = 4usize;
        let mut chunk = private_chunk(v_variants, n_samples); // carriers: Some(..)

        // Correct carrier FORMAT: variant i carried by sample (i % n_samples);
        // DP = 100 + i, GQ = 200 + i for that one carrier sample.
        let mut fbc: Vec<Arc<FormatVals>> = Vec::with_capacity(v_variants);
        for i in 0..v_variants {
            let mut cf = CarrierFormat::new(2);
            cf.push_sample((i % n_samples) as u32, &[(100 + i) as f64, (200 + i) as f64]);
            fbc.push(Arc::new(FormatVals::ByCarrier(cf)));
        }
        chunk.format_by_carrier = Some(fbc);

        // Poison the grid so a stray read from it is caught.
        chunk.format_staged = vec![
            StagedColumn::Int(vec![-1; v_variants * n_samples]),
            StagedColumn::Int(vec![-1; v_variants * n_samples]),
        ];

        let mut bank = make_bank();
        let out = dense2sparse_vk(&chunk, &mut bank, false, &two_format_fields());

        // All variants are single-carrier SNPs => VarKey stream. Collect its
        // two field columns and assert they equal the carrier FORMAT, never -1.
        let snp = out.streams.get(crate::streams::StreamTag::VarKeySnp);
        assert_eq!(snp.field_calls.len(), 2);
        let dp: Vec<i32> = match &snp.field_calls[0] {
            StagedColumn::Int(v) => v.clone(),
            _ => panic!("DP staged as non-int"),
        };
        assert!(dp.iter().all(|&x| x != -1), "read leaked from poisoned grid: {dp:?}");
        assert_eq!(dp.len(), v_variants);
        // Emission is column-major; every value is 100 + (its variant index).
        let mut sorted = dp.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..v_variants).map(|i| (100 + i) as i32).collect::<Vec<_>>());
    }
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
```bash
pixi run bash -lc "cargo test --no-default-features --features conversion --lib carrier_format_read_prefers_format_by_carrier"
```
Expected: FAIL — the emitted DP contains `-1` (still reading the poisoned `format_staged`).

- [ ] **Step 3: Add the `format_specs` parameter to `emit_call` and switch its FORMAT read**

In `src/rvk.rs`, change `emit_call`'s signature (currently lines 169-180) to insert `format_specs` after `per_cat`:

```rust
fn emit_call(
    route: &Route,
    chunk: &DenseChunk,
    v: usize,
    s: usize,
    hap: usize,
    num_samples: usize,
    per_cat: &[(bool, usize)],
    format_specs: &[&FieldSpec],
    streams: &mut crate::streams::StreamMap<SparseSubStream>,
    dense: &mut DenseMap<DenseSubChunk>,
    counts: &mut crate::streams::StreamMap<u32>,
) {
```

Replace the `VarKey` FORMAT read (currently lines 197-204) with the source switch:

```rust
            for (i, &(is_format, idx)) in per_cat.iter().enumerate() {
                let val = if is_format {
                    match chunk.format_by_carrier.as_ref() {
                        // Carrier-sparse: resolve for this calling sample. The
                        // ByCarrier arm ignores source_alt_index (values were
                        // resolved against each file's own ALT at merge time),
                        // so 0 is inert. `idx` is the per-category FORMAT index,
                        // which is also the field index into FormatVals.
                        Some(fbc) => crate::chunk_assembler::resolve_format(
                            &fbc[v], format_specs[idx], 0, s, idx,
                        ),
                        None => staged_f64(&chunk.format_staged[idx], v * num_samples + s),
                    }
                } else {
                    staged_f64(&chunk.info_staged[idx], v)
                };
                st.field_calls[i].push_f64(val);
            }
```

Note: `resolve_format` is currently a private `fn` in `chunk_assembler.rs`. Make it `pub(crate)`:
in `src/chunk_assembler.rs:243`, change `fn resolve_format(` to `pub(crate) fn resolve_format(`.

- [ ] **Step 4: Switch the dense second-pass fill in `route_variants`**

In `src/rvk.rs`, the dense-fill FORMAT branch (currently lines 412-421) reads `staged_f64(&chunk.format_staged[idx], ..)` for carrier samples. Replace that inner block with the same switch (note `format_specs` is already in scope here, built at lines 269-272):

```rust
            if is_format {
                let spec = format_specs[idx];
                for s in 0..num_samples {
                    let val = match chunk.format_by_carrier.as_ref() {
                        // Carrier source: value() returns None for a non-carrier,
                        // which resolve_format maps to the field default -- exactly
                        // the `is_carrier ? staged : default` the grid path did.
                        Some(fbc) => crate::chunk_assembler::resolve_format(
                            &fbc[v], spec, 0, s, idx,
                        ),
                        None => {
                            if is_carrier(v, s) {
                                staged_f64(&chunk.format_staged[idx], v * num_samples + s)
                            } else {
                                spec.missing_sentinel()
                            }
                        }
                    };
                    sub.field_format[idx].push_f64(val);
                }
            } else {
                let val = staged_f64(&chunk.info_staged[idx], v);
                sub.field_info[idx].push_f64(val);
            }
```

(This replaces the whole `if is_format { .. } else { .. }` at lines 411-425. The `default_val` binding it had, `format_specs[idx].missing_sentinel()`, is now inlined as `spec.missing_sentinel()`.)

- [ ] **Step 5: Pass `format_specs` at both `emit_call` call sites**

`format_specs` is built inside `route_variants` but the two `emit_call` callers are in `dense2sparse_vk_by_scan` and `dense2sparse_vk`. Compute it locally in each. In `dense2sparse_vk_by_scan`, after `route_variants(..)` returns (currently ~line 474), add:

```rust
    let format_specs: Vec<&FieldSpec> = fields
        .iter()
        .filter(|f| f.category == FieldCategory::Format)
        .collect();
```

and pass `&format_specs` in the `emit_call(route, chunk, v, s, hap, num_samples, &per_cat, &format_specs, &mut streams, &mut dense, &mut counts)` call (currently lines 516-527).

Do the same in `dense2sparse_vk` (add the `format_specs` binding after `route_variants(..)` ~line 602, and pass `&format_specs` in the `emit_call` at lines 653-664).

- [ ] **Step 6: Run the new test and the full suite**

Run:
```bash
pixi run bash -lc "cargo test --no-default-features --features conversion --lib"
```
Expected: PASS — `carrier_format_read_prefers_format_by_carrier` now green, and every existing test (differential, oracle-adjacent, dense-fill) stays green. If `carrier_driven_emission_matches_the_grid_scan_exactly` fails here, STOP: it means `private_chunk`'s `format_by_carrier` (still `None` from Task 1's helper) diverges from its `format_staged` — Task 4 fixes that helper, but at THIS task the carrier path reads `format_by_carrier` which `private_chunk` leaves `None`, so it falls through to `format_staged` and stays identical. Confirm that reasoning holds; if not, it is a real bug.

- [ ] **Step 7: Commit**

```bash
git add src/rvk.rs src/chunk_assembler.rs
git commit -m "perf(svar2): resolve FORMAT from format_by_carrier in rvk"
```

---

### Task 3: Stop building `format_staged` for carrier-bearing chunks (the O(N²) removal)

Now that no consumer reads `format_staged` for a carrier chunk, stop building it. This deletes the `O(V × F × N)` staging loop — the actual asymptote fix and the ~10 GB/chunk peak-RAM drop.

**Files:**
- Modify: `src/chunk_assembler.rs:712-747` (conditional `format_staged` staging)
- Modify: `src/chunk_assembler.rs:731-736`, `src/chunk_assembler.rs:261-267` (stale O(N²) comments)

**Interfaces:**
- Consumes: `format_by_carrier` populated in Task 1; the read switch from Task 2.
- Produces: for carrier-bearing chunks, `format_staged` is an empty `Vec` (one empty `StagedColumn` per FORMAT field is NOT allocated); `info_staged` unchanged.

- [ ] **Step 1: Make the `format_staged` build conditional**

In `src/chunk_assembler.rs`, the metadata pass currently unconditionally allocates `format_staged` (lines 712-716) and fills it in the inner `for (j, col) in format_staged.iter_mut()` loop (lines 737-747). We only know `all_some` after the loop today, so hoist the carrier-bearing determination.

Replace the `format_staged` allocation (lines 712-716) with a deferred allocation, and guard the fill. Concretely:

1. Determine carrier-bearing-ness up front. The metadata pass drains `a.carriers` per atom; instead, peek before the loop:

```rust
        // A chunk is carrier-bearing iff its first atom carries (uniformity is
        // asserted below). Decide once, up front, so FORMAT staging can be
        // skipped entirely for carrier-bearing chunks.
        let carrier_bearing = metas.first().is_some_and(|a| a.carriers.is_some());
```

2. Allocate `format_staged` only for the dense path:

```rust
        let mut format_staged: Vec<StagedColumn> = if carrier_bearing {
            Vec::new()
        } else {
            self.format_fields
                .iter()
                .map(|spec| StagedColumn::with_capacity(spec.stage_is_float(), v * num_samples))
                .collect()
        };
```

3. Guard the inner fill loop (lines 737-747) so it runs only for the dense path:

```rust
            if !carrier_bearing {
                for (j, col) in format_staged.iter_mut().enumerate() {
                    for s in 0..num_samples {
                        col.push_f64(resolve_format(
                            &a.format_vals,
                            &self.format_fields[j],
                            a.source_alt_index,
                            s,
                            j,
                        ));
                    }
                }
            }
```

The existing `all_some`/`all_none` debug-assert (lines 756-761) still runs and now also validates `carrier_bearing` was the right call — if a later atom disagrees with the first, that assert fires.

- [ ] **Step 2: Fix the stale O(N²) comments**

Replace the comment above the (now-guarded) fill loop (currently lines 731-736, "Stage every sample's value for every atom here (F x N per atom, unconditional...)") with:

```rust
            // Dense-source chunks stage every sample's value per atom (F x N):
            // for these, `genos`/`format_staged` IS the representation and the
            // per-sample column is the real work. Carrier-bearing chunks skip
            // this entirely -- their FORMAT rides `format_by_carrier` and `rvk`
            // resolves it per carrier (route-before-densify), so the old
            // unconditional F x N staging (the from_vcf_list O(N^2)) is gone.
```

Update the `AtomMeta` doc (lines 261-267) that says FORMAT resolution is lazy "the metadata pass in `read_next_chunk` needs it to resolve `format_vals`'s `Dense` arm" — it is still accurate for the dense path; append one sentence:

```rust
// resolve `format_vals`'s `Dense` arm ... . Carrier-bearing chunks never run
// that resolution here: their `format_vals` is moved wholesale into
// `DenseChunk::format_by_carrier` and resolved in `rvk` per carrier instead.
```

- [ ] **Step 3: Run the full suite**

Run:
```bash
pixi run bash -lc "cargo test --no-default-features --features conversion --lib"
```
Expected: PASS. `carrier_format_read_prefers_format_by_carrier` (its synthetic chunk sets `format_staged` directly, unaffected) and all existing tests stay green.

- [ ] **Step 4: Add a debug-assert that carrier chunks never index `format_staged`**

Defense in depth: in `src/rvk.rs`, at the top of `dense2sparse_vk` right after the `carriers` guard (after line 568), add:

```rust
    debug_assert!(
        chunk.format_by_carrier.is_some() || fields.iter().all(|f| f.category != FieldCategory::Format),
        "carrier-bearing chunk must carry format_by_carrier whenever FORMAT fields are requested"
    );
    debug_assert!(
        chunk.format_staged.is_empty() || chunk.format_by_carrier.is_none(),
        "carrier-bearing chunk must not also stage a dense FORMAT grid"
    );
```

Re-run the suite (same command) — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chunk_assembler.rs src/rvk.rs
git commit -m "perf(svar2): skip dense FORMAT staging for carrier chunks — kills the O(N^2)"
```

---

### Task 4: Strengthen the differential test to cover both FORMAT encodings

Today `private_chunk` populates only `format_staged`, and the differential test's scan clone shares the same grid. After Tasks 2-3 the carrier path reads `format_by_carrier`. Make `private_chunk` populate BOTH encodings consistently, and have the scan clone clear `format_by_carrier` so `via_scan` genuinely reads the grid while `via_carriers` reads the carrier source — turning the existing test into a real cross-encoding FORMAT parity.

**Files:**
- Modify: `src/rvk.rs:752-777` (`private_chunk`)
- Modify: `src/rvk.rs:870-890` (`carrier_driven_emission_matches_the_grid_scan_exactly`)

**Interfaces:**
- Consumes: `CarrierFormat`, `FormatVals::ByCarrier`, `DenseChunk.format_by_carrier`.
- Produces: `private_chunk` returns a chunk whose `format_staged` and `format_by_carrier` encode identical per-carrier FORMAT values.

- [ ] **Step 1: Populate `format_by_carrier` in `private_chunk`, consistent with the grid**

In `src/rvk.rs`, `private_chunk` currently sets `format_staged` to two integer ramps and leaves `format_by_carrier = None`. The grid values it reads back at the carrier cell for variant `i`, sample `s = i % n_samples` are `field0 = i*n_samples + s` and `field1 = (i*n_samples + s) * 2`. Build a `ByCarrier` that returns exactly those at the carrier sample. After the `chunk.format_staged = vec![..]` block (ends line 775), add:

```rust
        // Same values the grid holds at each variant's single carrier cell, so
        // the carrier path and the grid path are two encodings of ONE dataset
        // (mirrors how `genos` and `carriers` already are). The differential
        // test clears one encoding per side to prove they agree.
        use crate::record_source::{CarrierFormat, FormatVals};
        use std::sync::Arc;
        let mut fbc: Vec<Arc<FormatVals>> = Vec::with_capacity(v_variants);
        for i in 0..v_variants {
            let s = i % n_samples;
            let grid0 = (i * n_samples + s) as f64;
            let mut cf = CarrierFormat::new(2);
            cf.push_sample(s as u32, &[grid0, grid0 * 2.0]);
            fbc.push(Arc::new(FormatVals::ByCarrier(cf)));
        }
        chunk.format_by_carrier = Some(fbc);
        chunk
```

(Replace the trailing bare `chunk` at line 776 — do not leave two.)

- [ ] **Step 2: In the differential test, clear `format_by_carrier` on the scan clone**

In `carrier_driven_emission_matches_the_grid_scan_exactly`, the scan clone currently only clears `carriers` (lines 880-881). Add the FORMAT clear so `via_scan` reads the grid:

```rust
            // Same chunk with the carrier list AND carrier-FORMAT withheld =>
            // forced down the grid-scan path for BOTH genotypes and FORMAT,
            // so this asserts the two FORMAT encodings agree, not just genotypes.
            let mut scanned = chunk.clone();
            scanned.carriers = None;
            scanned.format_by_carrier = None;
```

- [ ] **Step 3: Run the differential test and the full suite**

Run:
```bash
pixi run bash -lc "cargo test --no-default-features --features conversion --lib carrier_driven_emission_matches_the_grid_scan_exactly"
pixi run bash -lc "cargo test --no-default-features --features conversion --lib"
```
Expected: PASS for both. `via_carriers` (carrier FORMAT) now equals `via_scan` (grid FORMAT), proving the encodings agree.

- [ ] **Step 4: Commit**

```bash
git add src/rvk.rs
git commit -m "test(svar2): differential test now covers both FORMAT encodings"
```

---

### Task 5: Python cross-path FORMAT + multi-contig oracle parity

Extend the existing bcftools-merge oracle so `from_vcf` (dense grid path) and `from_vcf_list` (carrier path) are compared **with FORMAT fields requested and across two contigs** — the byte-identical gate the spec §3 requires. This is the end-to-end proof that carrier-FORMAT resolution equals dense-FORMAT resolution on real files.

**Files:**
- Modify: `tests/test_svar2_from_vcf_list.py:223-...` (`test_from_vcf_list_no_reference_matches_bcftools_merge_oracle`) — or add a sibling `..._with_fields`
- Reference: `tests/test_svar2_from_vcf_list.py:1-75` (`_ss` helper, `_REF`), `python/genoray/_svar2.py:997-1090` (`from_vcf_list` signature — `format_fields` kwarg), `skills/genoray-api/SKILL.md` (decode field-output convention)

**Interfaces:**
- Consumes: `SparseVar2.from_vcf`/`from_vcf_list(..., format_fields=[...])`, `SparseVar2.decode(contig, ranges)` returning FORMAT field arrays.

- [ ] **Step 1: Read the existing oracle test and the `_ss` helper**

Run:
```bash
sed -n '1,75p;223,300p' tests/test_svar2_from_vcf_list.py
```
Confirm the `_ss(dir, name, sample, rows)` VCF-writer signature, the `_REF` constant, and how the current test calls `decode`. Confirm from the `genoray-api` skill how decoded FORMAT values are keyed in the returned object (e.g. `ro["DP"]` or an annotated field accessor).

- [ ] **Step 2: Add a fields+multi-contig oracle test**

Add to `tests/test_svar2_from_vcf_list.py`. Note the FORMAT header lines and per-record `GT:DP:VAF` columns; use SNPs only (the `no_reference` oracle is SNP-restricted — see the existing test's docstring) and put variants on `chr1` and `chr2`:

```python
def test_from_vcf_list_fields_multicontig_matches_dense_oracle(tmp_path: Path):
    """Carrier-path FORMAT (from_vcf_list) must byte-match dense-path FORMAT
    (from_vcf over a bcftools merge), across >1 contig, for requested FORMAT
    fields. This is the end-to-end gate for route-before-densify: it fails if
    carrier-sparse resolution diverges from the grid on any (sample, field)."""
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        "##contig=<ID=chr2,length=1000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Alt frac">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{s}\n"
    )

    def ss(name: str, sample: str, rows: str) -> Path:
        plain = tmp_path / f"{name}.vcf"
        plain.write_text(header.format(s=sample) + rows)
        gz = tmp_path / f"{name}.vcf.gz"
        with open(gz, "wb") as fh:
            subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
        subprocess.run(["bcftools", "index", str(gz)], check=True)
        return gz

    a = ss(
        "a", "SA",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DP:VAF\t1|0:30:0.5\n"   # shared chr1 SNP
        "chr1\t12\t.\tG\tC\t.\t.\t.\tGT:DP:VAF\t0|1:22:0.9\n"  # private chr1
        "chr2\t5\t.\tT\tA\t.\t.\t.\tGT:DP:VAF\t1|0:11:0.3\n",  # private chr2
    )
    b = ss(
        "b", "SB",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DP:VAF\t0|1:18:0.4\n"   # shared chr1 SNP
        "chr2\t5\t.\tT\tA\t.\t.\t.\tGT:DP:VAF\t0|1:27:0.7\n",  # shared chr2 SNP
    )
    paths = [a, b]

    merge = subprocess.run(
        ["bcftools", "merge", "-0", *map(str, paths)],
        check=True, stdout=subprocess.PIPE,
    )
    merged = tmp_path / "merged.vcf.gz"
    with open(merged, "wb") as fh:
        subprocess.run(["bgzip", "-c"], input=merge.stdout, check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(merged)], check=True)

    fields = dict(format_fields=["DP", "VAF"])
    dense_out, list_out = tmp_path / "dense", tmp_path / "list"
    SparseVar2.from_vcf(dense_out, merged, no_reference=True, threads=1, **fields)
    dropped = SparseVar2.from_vcf_list(list_out, paths, no_reference=True, threads=1, **fields)
    assert dropped == 0

    dense, native = SparseVar2(dense_out), SparseVar2(list_out)
    assert dense.available_samples == native.available_samples == ["SA", "SB"]

    for contig, length in [("chr1", 1000), ("chr2", 1000)]:
        region = [(0, length)]
        d = dense.decode(contig, region)
        n = native.decode(contig, region)
        # Genotype-level parity (positions/keys) AND FORMAT parity. Compare every
        # array the decode returns; the FORMAT field arrays (DP, VAF) are the
        # ones this change touches.
        for key in ("pos", "DP", "VAF"):
            np.testing.assert_array_equal(
                np.asarray(d[key].data), np.asarray(n[key].data),
                err_msg=f"{contig}:{key} diverged between dense and carrier paths",
            )
```

If the decode field-access convention differs from `d["DP"].data` (confirmed in Step 1), adapt the accessor — the assertion intent (dense FORMAT == carrier FORMAT, per contig) is what matters.

- [ ] **Step 3: Run the new test**

Run:
```bash
pixi run pytest tests/test_svar2_from_vcf_list.py::test_from_vcf_list_fields_multicontig_matches_dense_oracle -v
```
Expected: PASS. If it fails on the FORMAT arrays, that is a real carrier-vs-grid resolution bug — do NOT weaken the assertion; debug the resolution (systematic-debugging).

- [ ] **Step 4: Run the full from_vcf_list + parity suites**

Run:
```bash
pixi run pytest tests/test_svar2_from_vcf_list.py tests/test_svar2_from_vcf_list_parity.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_svar2_from_vcf_list.py
git commit -m "test(svar2): cross-path FORMAT + multi-contig parity for from_vcf_list"
```

---

### Task 6: Verify the asymptote, peak RAM, and churn (measurement gate)

The behavioral change is done and green; this task proves it actually moved the numbers the spec targets. These are heavy, environment-sensitive runs — follow the env caveats exactly. No code change unless a measurement reveals a bug.

**Files:**
- Reference: `src/bin/bench_from_vcf_list.rs` (already accepts a FORMAT field list; no change), the spec `docs/superpowers/specs/2026-07-17-svar2-format-route-before-densify-design.md` §3, §6

- [ ] **Step 1: dhat churn regression on the fixed bench**

Build and run the dhat bench (3 contigs, F=7), parking output off NFS/`/tmp`:

```bash
export CARGO_TARGET_DIR=/tmp/genoray_dhat_$$
pixi run bash -lc "cargo build --release --no-default-features --features conversion,dhat-heap --bin bench_from_vcf_list"
OUT=$CLAUDE_JOB_DIR/tmp/dhat_after
./target/release/bench_from_vcf_list <manifest> $OUT "1,2,3" <ref.fa> \
    "VAF:float,DP:int,PURPLE_AF:float,PURPLE_CN:float,PURPLE_VCN:float,PURPLE_MACN:float,SUBCL:float"
```
Expected vs the 158.8M-block / 140.9 GB-churn baseline: **total blocks fall ≥10×**, and neither `vcf_list_reader.rs:485` nor `chunk_assembler.rs` staging appears in the top-5 churn sites. **Confirm `run_exit == 0` and real output before trusting any dhat number** (a reaped manifest prints a tiny byte count that looks like a huge win — it's a panic).

- [ ] **Step 2: N-sweep for the slope change**

Run the CPU-time sweep (single contig, F=7) at N ∈ {250, 500, 1000, 2000, 4000} via the harness in the memory design (`run_arm.py`), capturing `cpu_s`. Fit the exponent.
Expected: the exponent drops from ~N^1.75 toward ~1.0 (linear). Record the table. If it stays ~1.75, the O(N²) is NOT gone — reopen (the most likely miss is a `format_staged` read path still live for carrier chunks; the Task 3 Step 4 debug-asserts should have caught it, so re-run with a debug build).

- [ ] **Step 3: Peak-RAM / ratchet trace**

Run the 3-contig N=1000 RSS trace (`run_trace.py`). Expected: the ~10 GB/chunk `format_staged` is gone from peak; peak RSS drops materially and the per-contig ratchet flattens versus the memory-design baseline (9.20 GB, +2.3 GB/contig).

- [ ] **Step 4: perf profile — `dense2sparse_vk` leaves the top**

Profile a single-contig F=7 run. Expected: `dense2sparse_vk` is no longer a top self-time symbol.

- [ ] **Step 5: Record the results and update the PR baseline doc**

Append an "after FORMAT route-before-densify" section to `docs/superpowers/plans/2026-07-16-svar2-from-vcf-list-memory-baseline.md` (or a new `-format-fix-baseline.md`) with the dhat, N-sweep, RAM, and perf numbers. Commit:

```bash
git add docs/superpowers/plans/*baseline*.md
git commit -m "docs(svar2): record the FORMAT route-before-densify measurements"
```

- [ ] **Step 6: Re-run the real 7089-file cohort (if the cluster slot is available)**

Per spec §3 "the real thing": re-run the Hartwig cohort (24 contigs, F=7) against the 132 GB baseline. This is the only test at full N × genome. Record MaxRSS/VmHWM and wall-clock. (If no slot is available this session, note it as the remaining acceptance step, not a blocker for the PR.)

---

## Self-Review

**Spec coverage** (against `2026-07-17-svar2-format-route-before-densify-design.md`):
- §2.1 representation → Task 1. §2.2 conditional staging → Task 3. §2.3 rvk read switch (both consumers) → Task 2. §2.4 cost / §2.5 peak RAM → verified in Task 6. §2.6 unchanged budgeting → no task needed (explicitly no change). §3 verification: byte-identical (Task 5 cross-path + Task 4 differential), unit test (Task 2), dhat gate (Task 6.1), asymptote (Task 6.2), peak RAM (Task 6.3), the real thing (Task 6.6). §4 Design B → assessment lives in the spec, nothing to build. §5 risks: differential-test blind spot addressed by Task 5's fixture; germline routing-threshold — see note below; Arc lifetime → the `Arc` frees per chunk (chunk dropped after `dense2sparse_vk`); public API → none (Global Constraints).
- **Gap found & noted:** §5 "confirm behavior on a germline cohort." Tasks 4-5 use somatic/private fixtures. Added coverage: `chunk_with_dense_variants` (existing rvk helper, all-columns-set → routes Dense) already exercises the dense-fill FORMAT path under the switch via `carrier_format_read_prefers_format_by_carrier`'s sibling paths — but to be explicit, Task 5's oracle includes a shared site (chr1:3, both samples) and the dense oracle routes it through the grid, so dense-FORMAT-fill parity IS covered. No separate task required.

**Placeholder scan:** No TBD/TODO. Bench/harness invocations in Task 6 use `<manifest>`/`<ref.fa>` angle-brackets — these are genuine per-environment inputs (the cohort path), not placeholders for code; every code step shows complete code.

**Type consistency:** `format_by_carrier: Option<Vec<Arc<FormatVals>>>` used identically in Tasks 1-4. `emit_call`'s new `format_specs: &[&FieldSpec]` param inserted after `per_cat` consistently at definition (Task 2 Step 3) and both call sites (Task 2 Step 5). `resolve_format` made `pub(crate)` (Task 2 Step 3) before rvk calls it. `CarrierFormat::new(n)`/`push_sample(u32, &[f64])`/`value(usize,usize)->Option<f64>` match `record_source.rs:169-198`.

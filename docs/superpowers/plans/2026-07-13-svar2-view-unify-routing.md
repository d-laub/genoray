# SVAR2 `write_view` Backend Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement `write_view(reroute=True)` as a *routing policy* inside the existing array-slicer, delete the pipeline-backed view path, and parallelize the slicer across contigs — resolving all four of PR #105's deferred call-outs.

**Architecture:** `src/svar2_slice.rs` is refactored from a one-phase "gather-and-write" into **gather → route → emit**. A `Routing` enum picks each variant's output stream: `Preserve` (its source stream — today's `reroute=False`) or `Recompute` (`cost_model::choose_representation` against the *subset's* carrier count — the new `reroute=True`). Variants that flip stream are re-encoded in both directions, fields included. `Svar2Source` / `SourceSpec::Svar2` / `run_view_pipeline`'s SVAR2 branch are deleted once a differential test proves the new `reroute=True` matches the old one.

**Tech Stack:** Rust (pyo3, rayon, bytemuck, ndarray-npy, memmap2), Python 3.10+, pixi, pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-svar2-view-unify-routing-design.md`

## Global Constraints

- **Lands in PR #105, before merge** (branch `worktree-svar2-merge-split-view`). Do not open a new PR.
- **`export CARGO_TARGET_DIR=/tmp/genoray-target-$$` before any cargo/prek command.** The repo lives on NFS; prek's cargo hooks bus-error against the NFS `target/`.
- **Rust tests must run `--no-default-features`**: `pixi run bash -lc 'cargo test --no-default-features --test <file>'`. Otherwise the pyo3 test binary fails to link (`undefined symbol: _Py_Dealloc`).
- **`pixi run test-rust <arg>` filters by TEST NAME, not file.** A non-matching arg vacuously passes 0 tests. Always use `--test <file>`.
- **Conventional Commits** (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`) — enforced by a commitizen hook.
- **Public API changes MUST update `skills/genoray-api/SKILL.md` in the same PR** (project CLAUDE.md; non-optional).
- **Do not hand-bump the version or `CHANGELOG.md`'s versioned sections.** Accumulate entries under `## Unreleased` only.
- **Never run long cargo/maturin builds in the background** — run them in the foreground and wait.
- Python builds against the Rust extension via `pixi run develop` (maturin). After any Rust change, rebuild before running Python tests.

## Key invariant the whole plan rests on

**A var_key call's field value is the *sample's* value, duplicated across that sample's carrier haps.** `from_vcf` builds `RawRecord.format_raw` per *sample* (`Vec<Option<Vec<f64>>>`), and `dense2sparse_vk` writes one element per *call*. So two carrier calls of the same sample at the same variant hold identical bytes. This is what makes a var_key→dense flip well-defined (pick any carrier call of that sample) and a dense→var_key flip lossless *for carriers*. Non-carrier FORMAT values have no var_key slot and are dropped — the `"auto"` rule (Task 7) exists to steer the default around exactly this.

---

### Task 1: Extract the shared selection predicate into `src/svar2_view.rs`

Pure move, no behavior change. `OverlapMode` / `query_window` / `keeps` / `read_n_samples` are the guarantee that both routings select an identical variant set; they must outlive `Svar2Source` (deleted in Task 9).

**Files:**
- Create: `src/svar2_view.rs`
- Modify: `src/svar2_source.rs` (remove the moved items; re-import them)
- Modify: `src/svar2_slice.rs:49`, `src/lib.rs:487-489,691-693`, `src/orchestrator.rs:71`, `src/main.rs`-style module list in `src/lib.rs`

**Interfaces:**
- Produces: `crate::svar2_view::{OverlapMode, query_window, keeps, read_n_samples}` — same signatures as today's `crate::svar2_source::*`.

- [ ] **Step 1: Create `src/svar2_view.rs` by moving the four items verbatim**

Cut `OverlapMode` (the enum), `query_window`, `keeps`, and `read_n_samples` out of `src/svar2_source.rs` and paste them into a new `src/svar2_view.rs`, unchanged. Header:

```rust
//! Region/sample **selection** for SVAR2 views — the predicate shared by BOTH
//! routings of `svar2_slice`.
//!
//! `query_window` widens a region per overlap mode before the tree search;
//! `keeps` applies the final POS-precision filter. Both routings call these, so
//! `reroute=True` and `reroute=False` provably select the IDENTICAL variant set.
//! Do not re-derive this logic anywhere else.
```

- [ ] **Step 2: Re-point every import**

In `src/lib.rs`, add `mod svar2_view;` next to the other `mod` declarations. Then replace `crate::svar2_source::OverlapMode` → `crate::svar2_view::OverlapMode` at `src/lib.rs:487-489` and `src/lib.rs:691-693`, and `crate::svar2_source::OverlapMode` → `crate::svar2_view::OverlapMode` at `src/orchestrator.rs:71`. In `src/svar2_slice.rs:49`, change:

```rust
use crate::svar2_view::{OverlapMode, keeps, query_window, read_n_samples};
```

In `src/svar2_source.rs`, add `use crate::svar2_view::{OverlapMode, keeps, query_window, read_n_samples};` so the (still-alive) `Svar2Source` keeps compiling.

- [ ] **Step 3: Verify nothing changed**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo build --no-default-features && cargo clippy --no-default-features --all-targets -- -D warnings'`
Expected: builds clean, zero warnings.

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice --test test_svar2_source'`
Expected: all PASS (this is a pure move; any failure means something was altered).

- [ ] **Step 4: Commit**

```bash
git add src/svar2_view.rs src/svar2_source.rs src/svar2_slice.rs src/lib.rs src/orchestrator.rs
git commit -m "refactor(svar2): extract the shared view selection predicate into svar2_view"
```

---

### Task 2: Two-phase slicer — gather into provenance, then emit

The enabling refactor. Today `slice_var_key_snp` / `slice_dense` gather *and* produce final byte buffers in one shot, so no variant can change stream. Split them: gather produces per-call / per-row records with source provenance; a separate emit stage writes the sidecars. Introduce `Routing`, but implement **only `Routing::Preserve`** here — output must stay byte-identical, which the existing byte-parity tests prove.

**Files:**
- Modify: `src/svar2_slice.rs` (the gather fns + `slice_genos_inner`)
- Test: `tests/test_svar2_slice.rs` (existing tests must pass unchanged)

**Interfaces:**
- Produces:
```rust
/// Which on-disk stream each variant lands in.
pub enum Routing {
    /// `reroute=False`: a variant's output stream is its source stream.
    Preserve,
    /// `reroute=True`: re-run the cost model against the SUBSET's carrier count.
    Recompute,
}

/// Where one OUTPUT var_key call's genotype + field bytes come from.
#[derive(Clone, Copy)]
enum CallSrc {
    /// Unflipped: a source var_key call index.
    VarKey { call: usize },
    /// Flipped dense -> var_key: source dense row + ORIGINAL sample column.
    Dense { row: usize, s_orig: usize },
}

/// Where one OUTPUT dense row's genotype + field bytes come from.
enum RowSrc {
    /// Unflipped: a source dense row index.
    Dense { row: usize },
    /// Flipped var_key -> dense. `per_sample_call[s_out]` is a representative
    /// source var_key call for that output sample (any carrier call — see the
    /// key invariant), or `None` for a non-carrier (=> field sentinel).
    VarKey {
        per_sample_call: Vec<Option<usize>>,
        /// Any source call of this variant — INFO is per-variant, so any will do.
        info_call: usize,
    },
}

/// One gathered var_key call, before routing.
struct GatheredCall {
    src: usize,      // source call index
    col_out: usize,  // OUTPUT hap column (s_out*ploidy + p)
    pos: u32,
    key: u32,        // SNP: snp_code_to_key(code); indel: the raw key
}

/// One gathered dense row, before routing.
struct GatheredRow {
    src: usize,               // source row index
    pos: u32,
    key: u32,
    carriers_out: Vec<usize>, // OUTPUT hap columns carrying it
}
```

- [ ] **Step 1: Write the failing test — Routing::Preserve is byte-identical**

Add to `tests/test_svar2_slice.rs`. It calls the new two-phase entry point and compares against the *existing* identity-slice expectation, so it fails to compile until the refactor lands.

```rust
#[test]
fn preserve_identity_slice_is_byte_parity() {
    let src = fixture_store_with_fields();       // existing helper in this file
    let out = tempfile::tempdir().unwrap();
    let all: Vec<usize> = (0..src.n_samples).collect();
    let regions = vec![(0u32, u32::MAX)];

    svar2_slice::slice_contig(
        &src.path, out.path().to_str().unwrap(), "chr1",
        &all, 2, &regions, OverlapMode::Variant,
        &src.field_specs,
        svar2_slice::Routing::Preserve,   // <-- new trailing arg
    ).unwrap();

    assert_sidecars_byte_equal(&src.path, out.path().to_str().unwrap(), "chr1");
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice'`
Expected: FAIL — compile error, `Routing` not found / `slice_contig` takes 8 args, not 9.

- [ ] **Step 3: Add `Routing` + the provenance types**

Add the enums/structs from **Interfaces** above to `src/svar2_slice.rs`. Add `use svar2_codec::snp_code_to_key;` so SNP codes and indel keys share one `u32` key space for grouping (a SNP's `key` is only ever compared against other SNPs' — the two classes never mix streams).

- [ ] **Step 4: Convert the gather fns to return `GatheredCall` / `GatheredRow`**

Rewrite `slice_var_key_snp` / `slice_var_key_indel` into one shape that stops building byte buffers. The region-hit logic (`region_hits`, the `v_end_of` closures, the `col_src` / `s_orig` / `p` arguments to `vk_indel_overlap`) is **unchanged** — only the accumulation changes:

```rust
/// Gather the kept var_key calls of one class. `key_of(i)` reads the source
/// call's key; `v_end_of(i)` its right extent. Column order is the OUTPUT
/// order (`sample_orig_idx` x ploidy) so the CSR emit stays a simple scan.
fn gather_var_key(
    positions: &[u32],
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
    overlap_range: impl Fn(usize, usize, usize, u32, u32) -> Range<usize>, // (col_src, s_orig, p, qsw, qew)
    key_of: impl Fn(usize) -> u32,
    v_end_of: impl Fn(usize) -> u32,
) -> Vec<GatheredCall> {
    let mut out = Vec::new();
    for (s_out, &s_orig) in sample_orig_idx.iter().enumerate() {
        for p in 0..ploidy {
            let col_src = s_orig * ploidy + p;
            let col_out = s_out * ploidy + p;
            let hits = region_hits(
                positions, regions, query_regions, overlap,
                |qsw, qew| overlap_range(col_src, s_orig, p, qsw, qew),
                &v_end_of,
            );
            for i in hits {
                out.push(GatheredCall { src: i, col_out, pos: positions[i], key: key_of(i) });
            }
        }
    }
    out
}
```

Call it with `key_of = |i| snp_code_to_key(unpack_snp_key_at(keys, i))` for SNP and `key_of = |i| keys[i]` for indel; the `overlap_range` closures are lifted verbatim from today's `reader.vk_snp_overlap(col_src, ..)` / `reader.vk_indel_overlap(col_src, s_orig, p, ..)`.

Rewrite `slice_dense` the same way — keep the `carried_by_subset` prepass and the `region_hits` call **exactly** as they are, but instead of writing bits, record each kept row's output carriers:

```rust
fn gather_dense(
    dense: Option<&DenseView>,
    is_snp: bool,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
    overlap_range: impl Fn(u32, u32) -> Range<usize>,
) -> Vec<GatheredRow> {
    // ... identical hits + carried_by_subset logic as today's slice_dense ...
    let mut rows: Vec<GatheredRow> = Vec::new();
    let mut row_of_src: Vec<i64> = vec![-1; n_dense];
    for &row in &hits {
        if !carried_by_subset[row] { continue; }
        row_of_src[row] = rows.len() as i64;
        let key = if is_snp {
            snp_code_to_key(unpack_snp_key_at(keys_bytes, row))
        } else {
            keys_u32[row]
        };
        rows.push(GatheredRow { src: row, pos: positions[row], key, carriers_out: Vec::new() });
    }
    for (s_out, &s_orig) in sample_orig_idx.iter().enumerate() {
        for p in 0..ploidy {
            let hap_src = s_orig * ploidy + p;
            let hap_out = s_out * ploidy + p;
            d.for_each_carried(hap_src, |col| {
                let r = row_of_src[col];
                if r >= 0 { rows[r as usize].carriers_out.push(hap_out); }
            });
        }
    }
    rows
}
```

- [ ] **Step 5: Add the emit stage**

Two writers, consuming routed decisions. Under `Routing::Preserve` they receive exactly the unflipped inputs, reproducing today's bytes.

```rust
/// Emit one var_key class. `calls` must already be sorted by (col_out, pos, key).
/// Returns the per-output-call provenance, parallel to the written sidecars.
fn emit_var_key(
    dir: &Path,
    calls: &[(CallSrc, usize /*col_out*/, u32 /*pos*/, u32 /*key*/)],
    is_snp: bool,
    n_cols_out: usize,
) -> Result<Vec<CallSrc>, ConversionError> {
    let mut positions = Vec::with_capacity(calls.len());
    let mut codes: Vec<u8> = Vec::new();   // snp
    let mut keys: Vec<u32> = Vec::new();   // indel
    let mut offsets: Vec<u64> = Vec::with_capacity(n_cols_out + 1);
    let mut prov = Vec::with_capacity(calls.len());

    offsets.push(0);
    let mut c = 0usize;
    for col in 0..n_cols_out {
        while c < calls.len() && calls[c].1 == col {
            let (src, _, pos, key) = calls[c];
            positions.push(pos);
            if is_snp { codes.push(decode_snp_2bit_code(key)); } else { keys.push(key); }
            prov.push(src);
            c += 1;
        }
        offsets.push(positions.len() as u64);
    }

    create_dir(dir)?;
    write_bytes(&layout::positions(dir), bytemuck::cast_slice(&positions))?;
    if is_snp {
        write_bytes(&layout::alleles(dir), &pack_snp_keys(&codes))?;
    } else {
        write_bytes(&layout::alleles(dir), bytemuck::cast_slice(&keys))?;
    }
    write_offsets(&layout::offsets(dir), &offsets)?;
    Ok(prov)
}

/// Emit one dense class. `rows` must already be sorted by (pos, key).
fn emit_dense(
    dir: &Path,
    rows: Vec<(RowSrc, u32 /*pos*/, u32 /*key*/, Vec<usize> /*carriers_out*/)>,
    is_snp: bool,
    n_cols_out: usize,
) -> Result<Vec<RowSrc>, ConversionError> {
    let n_kept = rows.len();
    let mut positions = Vec::with_capacity(n_kept);
    let mut key_bytes: Vec<u8> = Vec::new();
    let mut bits = vec![0u8; (n_cols_out * n_kept).div_ceil(8)];
    let mut prov = Vec::with_capacity(n_kept);

    for (row_out, (src, pos, key, carriers)) in rows.into_iter().enumerate() {
        positions.push(pos);
        if is_snp {
            key_bytes.push(decode_snp_2bit_code(key));
        } else {
            key_bytes.extend_from_slice(&key.to_le_bytes());
        }
        for hap_out in carriers {
            set_bit(&mut bits, hap_out * n_kept + row_out);   // hap-major, as today
        }
        prov.push(src);
    }

    create_dir(dir)?;
    write_bytes(&layout::positions(dir), bytemuck::cast_slice(&positions))?;
    let alleles = if is_snp { pack_snp_keys(&key_bytes) } else { key_bytes };
    write_bytes(&layout::alleles(dir), &alleles)?;
    write_bytes(&layout::genotypes(dir), &bits)?;
    Ok(prov)
}
```

`decode_snp_2bit_code(key: u32) -> u8` is a tiny local inverse of `snp_code_to_key` (mask the 2-bit payload); add it next to the types and unit-test the round-trip `snp_code_to_key(c) -> decode_snp_2bit_code -> c` for `c in 0..4`.

- [ ] **Step 6: Rewrite `slice_genos_inner` as gather → route → emit**

For this task the route stage is the identity:

```rust
let vk_snp_g   = gather_var_key(/* snp */);
let vk_indel_g = gather_var_key(/* indel */);
let d_snp_g    = gather_dense(/* snp */);
let d_indel_g  = gather_dense(/* indel */);

let plan = route(
    routing, vk_snp_g, vk_indel_g, d_snp_g, d_indel_g,
    sample_orig_idx.len(), ploidy, sidecar_bits_enabled, info_bits, format_bits,
);   // Task 3 gives `route` its Recompute arm; here it only handles Preserve

let vk_snp_prov   = emit_var_key(&out_paths.var_key_snp_dir(),   &plan.vk_snp,   true,  n_cols_out)?;
let vk_indel_prov = emit_var_key(&out_paths.var_key_indel_dir(), &plan.vk_indel, false, n_cols_out)?;
let d_snp_prov    = emit_dense(&out_paths.dense_snp_dir(),       plan.d_snp,     true,  n_cols_out)?;
let d_indel_prov  = emit_dense(&out_paths.dense_indel_dir(),     plan.d_indel,   false, n_cols_out)?;
```

For `Routing::Preserve`, `route` maps each `GatheredCall` → `(CallSrc::VarKey { call: g.src }, g.col_out, g.pos, g.key)` (already in `(col_out, pos)` order — the gather loop is column-major and `region_hits` sorts within a column) and each `GatheredRow` → `(RowSrc::Dense { row: g.src }, g.pos, g.key, g.carriers_out)` (already position-ascending).

Keep the LUT verbatim copy, `write_max_del`, and the distinct-variant count **exactly as they are**; `GenoProvenance` now holds `Vec<CallSrc>` / `Vec<RowSrc>` instead of `Vec<usize>`.

- [ ] **Step 7: Adapt the field pass to the new provenance (mechanical for now)**

`slice_field_var_key` / `slice_field_dense` take `&[CallSrc]` / `&[RowSrc]`. In this task every variant is unflipped, so match only the unflipped arms and `unreachable!()` on the flipped ones (Task 4 fills them in):

```rust
for src in src_calls {
    match src {
        CallSrc::VarKey { call } => out.extend_from_slice(view.bytes_at(*call)),
        CallSrc::Dense { .. } => unreachable!("flipped calls arrive in Task 4"),
    }
}
```

- [ ] **Step 8: Add `Routing::Preserve` as the trailing arg on both entry points**

`slice_contig(..., fields: &[FieldSpec], routing: Routing)` and `slice_contig_genos(..., routing: Routing)`. Update `src/lib.rs`'s `run_slice_view` to pass `Routing::Preserve` (still the only caller).

- [ ] **Step 9: Run the full slicer suite**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice'`
Expected: PASS — **including the pre-existing byte-parity identity test.** That test is the whole point of this task: it proves the refactor changed no bytes.

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo clippy --no-default-features --all-targets -- -D warnings'`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/svar2_slice.rs src/lib.rs tests/test_svar2_slice.rs
git commit -m "refactor(svar2): split the slicer into gather -> route -> emit"
```

---

### Task 3: `Routing::Recompute` — the cost model and genotype flips

**Files:**
- Modify: `src/svar2_slice.rs` (the `route` fn)
- Test: `tests/test_svar2_slice.rs`

**Interfaces:**
- Consumes: `Routing`, `GatheredCall`, `GatheredRow`, `CallSrc`, `RowSrc`, `emit_var_key`, `emit_dense` (Task 2).
- Produces: `route(...) -> RoutePlan` with the `Recompute` arm live. `slice_contig` gains `sidecar_bits_enabled: bool`, `info_bits: u64`, `format_bits: u64` — computed by the caller (Task 6) as: `sidecar_bits_enabled = reference.is_some()`; `info_bits`/`format_bits` = summed `spec.dtype.width_bytes().unwrap() * 8` over the INFO / FORMAT specs in `fields`, mirroring `src/rvk.rs:230-238`.

- [ ] **Step 1: Write the failing tests — one flip in each direction**

Add to `tests/test_svar2_slice.rs`. Build a fixture whose routing provably flips: with `n_samples`, `ploidy=2`, no fields/signatures, the SNP crossover is `dense iff 32+2+np < x*(32+2)`, i.e. `x > (34 + 2*n)/34`.

```rust
/// A SNP carried by MANY haps of a large cohort is dense at full size; keep a
/// 2-sample subset in which it has 2 carriers and it must flip dense -> var_key.
#[test]
fn recompute_flips_dense_to_var_key() {
    let src = fixture_dense_snp_store(/*n_samples*/ 200, /*carrier_haps*/ 180);
    let out = tempfile::tempdir().unwrap();
    let outp = out.path().to_str().unwrap();

    // subset = 2 samples, both carriers -> x_sub = 2, n_sub = 2:
    // dense = 32+2+4 = 38 bits; var_key = 2*34 = 68 -> stays DENSE. Use 1 sample:
    // dense = 32+2+2 = 36; var_key = 1*34 = 34 -> var_key is cheaper. FLIP.
    svar2_slice::slice_contig_genos(
        &src.path, outp, "chr1", &[0], 2, &[(0, u32::MAX)],
        OverlapMode::Variant, svar2_slice::Routing::Recompute,
    ).unwrap();

    // The variant must now live in var_key/snp, and dense/snp must be empty.
    assert!(var_key_snp_positions(outp, "chr1").contains(&SNP_POS));
    assert!(dense_snp_positions(outp, "chr1").is_empty());
    // ...and it must still decode to the same genotypes as the source subset.
    assert_decode_matches_source(&src.path, outp, "chr1", &[0]);
}

/// A var_key SNP with many carriers in a SMALL subset must flip var_key -> dense.
#[test]
fn recompute_flips_var_key_to_dense() {
    // 100 samples; the SNP is carried by 40 haps -> var_key at n=100
    // (dense = 34+200 = 234 vs var_key = 40*34 = 1360 -> actually dense; so make
    // it carried by 3 haps: var_key = 102 < 234 -> VAR_KEY at full size).
    let src = fixture_var_key_snp_store(/*n_samples*/ 100, /*carrier_samples*/ &[0, 1]);
    let out = tempfile::tempdir().unwrap();
    let outp = out.path().to_str().unwrap();

    // subset = the 2 carrier samples: x_sub = 3, n_sub = 2 ->
    // dense = 32+2+4 = 38; var_key = 3*34 = 102 -> DENSE is cheaper. FLIP.
    svar2_slice::slice_contig_genos(
        &src.path, outp, "chr1", &[0, 1], 2, &[(0, u32::MAX)],
        OverlapMode::Variant, svar2_slice::Routing::Recompute,
    ).unwrap();

    assert!(dense_snp_positions(outp, "chr1").contains(&SNP_POS));
    assert!(var_key_snp_positions(outp, "chr1").is_empty());
    assert_decode_matches_source(&src.path, outp, "chr1", &[0, 1]);
}

/// Full coverage => x_sub == x_full => zero flips => Recompute == Preserve,
/// which is byte-parity with the source. The strongest test in the suite.
#[test]
fn recompute_full_coverage_is_byte_parity() {
    let src = fixture_store_with_fields();
    let out = tempfile::tempdir().unwrap();
    let all: Vec<usize> = (0..src.n_samples).collect();
    svar2_slice::slice_contig(
        &src.path, out.path().to_str().unwrap(), "chr1", &all, 2,
        &[(0, u32::MAX)], OverlapMode::Variant, &src.field_specs,
        svar2_slice::Routing::Recompute,
    ).unwrap();
    assert_sidecars_byte_equal(&src.path, out.path().to_str().unwrap(), "chr1");
}
```

Compute each fixture's expected routing by hand from `cost_model::choose_representation` before writing the assertion — do **not** assert whatever the code happens to produce.

- [ ] **Step 2: Run to confirm they fail**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice recompute'`
Expected: FAIL — `Routing::Recompute` hits the Task-2 `todo!()`/identity arm, so the flips don't happen.

- [ ] **Step 3: Implement the `Recompute` arm of `route`**

```rust
fn route(
    routing: Routing,
    vk_snp: Vec<GatheredCall>, vk_indel: Vec<GatheredCall>,
    d_snp: Vec<GatheredRow>,   d_indel: Vec<GatheredRow>,
    n_subset: usize, ploidy: usize,
    sidecar_bits_enabled: bool, info_bits: u64, format_bits: u64,
) -> RoutePlan {
    match routing {
        Routing::Preserve => /* identity mapping from Task 2 */,
        Routing::Recompute => {
            let mut plan = RoutePlan::default();
            route_class(&mut plan, Class::Snp,   vk_snp,   d_snp,   /* .. */);
            route_class(&mut plan, Class::Indel, vk_indel, d_indel, /* .. */);
            plan.sort();   // see Step 4
            plan
        }
    }
}
```

`route_class` handles one class (SNP or indel — the two never exchange variants):

```rust
fn route_class(
    plan: &mut RoutePlan, class: Class,
    calls: Vec<GatheredCall>, rows: Vec<GatheredRow>,
    n_subset: usize, ploidy: usize,
    sidecar_bits: u64, info_bits: u64, format_bits: u64,
) {
    // 1. x_sub per var_key variant: one gathered call == one carrier hap.
    let mut by_variant: HashMap<(u32, u32), Vec<GatheredCall>> = HashMap::new();
    for c in calls { by_variant.entry((c.pos, c.key)).or_default().push(c); }

    // 2. var_key variants: stay, or flip to dense.
    for ((pos, key), group) in by_variant {
        let x_sub = group.len();
        match choose_representation(class, n_subset, ploidy, x_sub,
                                    sidecar_bits, info_bits, format_bits) {
            Representation::VarKey => {
                for c in group {
                    plan.push_call(class, CallSrc::VarKey { call: c.src }, c.col_out, pos, key);
                }
            }
            Representation::Dense => {
                // Carriers by OUTPUT hap column; a representative source call per
                // OUTPUT SAMPLE for the field pass (all carrier calls of a sample
                // hold identical field bytes -- see "Key invariant").
                let mut carriers_out: Vec<usize> = group.iter().map(|c| c.col_out).collect();
                carriers_out.sort_unstable();
                let mut per_sample_call = vec![None; n_subset];
                for c in &group {
                    per_sample_call[c.col_out / ploidy].get_or_insert(c.src);
                }
                let info_call = group[0].src;
                plan.push_row(class, RowSrc::VarKey { per_sample_call, info_call },
                              pos, key, carriers_out);
            }
        }
    }

    // 3. dense rows: stay, or flip to var_key.
    for r in rows {
        let x_sub = r.carriers_out.len();   // subset popcount of the row
        match choose_representation(class, n_subset, ploidy, x_sub,
                                    sidecar_bits, info_bits, format_bits) {
            Representation::Dense => {
                plan.push_row(class, RowSrc::Dense { row: r.src }, r.pos, r.key, r.carriers_out);
            }
            Representation::VarKey => {
                for &col_out in &r.carriers_out {
                    let s_orig = /* the ORIGINAL sample col of this out hap */;
                    plan.push_call(class, CallSrc::Dense { row: r.src, s_orig },
                                   col_out, r.pos, r.key);
                }
            }
        }
    }
}
```

`s_orig` comes from `sample_orig_idx[col_out / ploidy]` — pass `sample_orig_idx` into `route_class`.

**`x_sub` must never be 0.** A var_key variant with no subset carriers has no gathered calls, so it never enters `by_variant`; a dense row with no subset carriers was already dropped by `carried_by_subset`. That preserves the existing MAC=0 drop under both routings. Add a `debug_assert!(x_sub > 0)`.

- [ ] **Step 4: Sort each output stream deterministically**

`emit_var_key` requires `(col_out, pos, key)` order; `emit_dense` requires `(pos, key)` order. Flipped variants arrive out of order, so `RoutePlan::sort()` does:

```rust
fn sort(&mut self) {
    for s in [&mut self.vk_snp, &mut self.vk_indel] {
        s.sort_by_key(|&(_, col, pos, key)| (col, pos, key));
    }
    for s in [&mut self.d_snp, &mut self.d_indel] {
        s.sort_by_key(|(_, pos, key, _)| (*pos, *key));
    }
}
```

`sort_by_key` is stable, so same-`(pos, key)` entries keep their insertion order. Same-position **ties across different keys** are ordered by `key` — the one place the byte layout may differ from a fresh `from_vcf`. That is why `reroute=True` is verified by decode-equivalence + routing-equality + size (Tasks 3, 8), not by byte-parity against `from_vcf`. Note the *shipped* `reroute=True` also re-orders (its `Svar2Source` emits from a `BTreeMap<(pos, ilen, alt), _>`), so this is not new in kind. Put that reasoning in a comment above `sort`.

- [ ] **Step 5: Run the tests**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice'`
Expected: PASS — both flip tests, the `Recompute` full-coverage byte-parity test, and every pre-existing `Preserve` test.

- [ ] **Step 6: Commit**

```bash
git add src/svar2_slice.rs tests/test_svar2_slice.rs
git commit -m "feat(svar2): Routing::Recompute -- re-run the cost model on the subset"
```

---

### Task 4: Fields across flips

**Files:**
- Modify: `src/svar2_slice.rs` (`slice_field_var_key`, `slice_field_dense`)
- Test: `tests/test_svar2_slice.rs`

**Interfaces:**
- Consumes: `CallSrc`, `RowSrc` (Task 2), `FieldSpec::missing_sentinel()` (`src/field.rs:121`), `FieldView::bytes_at` / `open_source_field`.

- [ ] **Step 1: Write the failing tests**

```rust
/// var_key -> dense: non-carrier output samples must read back the field's
/// missing sentinel, exactly as rvk.rs's dense push fills them.
#[test]
fn flip_var_key_to_dense_fills_non_carrier_format_with_sentinel() {
    // 100-sample store, FORMAT field "DP" (u16), SNP carried by samples 0,1 only.
    let src = fixture_var_key_snp_with_format(100, &[0, 1]);
    let out = tempfile::tempdir().unwrap();
    let outp = out.path().to_str().unwrap();

    // subset {0, 1, 2}: x_sub = 3 (samples 0,1 carriers), n_sub = 3 ->
    // dense = 32+2+6+16*3 = 88 bits; var_key = 3*(34+16) = 150 -> DENSE. FLIP.
    svar2_slice::slice_contig(
        &src.path, outp, "chr1", &[0, 1, 2], 2, &[(0, u32::MAX)],
        OverlapMode::Variant, &src.field_specs, svar2_slice::Routing::Recompute,
    ).unwrap();

    let dp = read_dense_format_values(outp, "chr1", "DP");   // row-major, n_sub wide
    assert_eq!(dp[0], src.dp_of_sample(0));
    assert_eq!(dp[1], src.dp_of_sample(1));
    assert_eq!(dp[2], sentinel_u16());   // sample 2 is a NON-carrier
}

/// dense -> var_key: carrier calls keep the sample's value; non-carrier values
/// are DROPPED (var_key has no slot). This asserts the documented loss.
#[test]
fn flip_dense_to_var_key_keeps_carrier_format_and_drops_the_rest() {
    let src = fixture_dense_snp_with_format(200, /*carrier_haps*/ 180);
    let out = tempfile::tempdir().unwrap();
    let outp = out.path().to_str().unwrap();

    svar2_slice::slice_contig(
        &src.path, outp, "chr1", &[0], 2, &[(0, u32::MAX)],
        OverlapMode::Variant, &src.field_specs, svar2_slice::Routing::Recompute,
    ).unwrap();

    // sample 0 carries on hap 0 only -> exactly one var_key call, holding
    // sample 0's DP; nothing else is stored for this variant.
    let dp = read_var_key_format_values(outp, "chr1", "DP");
    assert_eq!(dp, vec![src.dp_of_sample(0)]);
}

/// INFO is per-variant: a flip must not change its value in either direction.
#[test]
fn flip_preserves_info_value_both_directions() { /* both fixtures, assert INFO unchanged */ }
```

- [ ] **Step 2: Run to confirm they fail**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice flip_'`
Expected: FAIL — panics at the Task-2 `unreachable!("flipped calls arrive in Task 4")`.

- [ ] **Step 3: Handle `CallSrc::Dense` in the var_key field gather**

A flipped call's value comes from the source *dense* sub-stream of the same class, so the gather must be able to open **both** the source's var_key sub AND its dense sub for one output sub-stream. Pass both views:

```rust
fn slice_field_var_key(
    src_paths: &ContigPaths, out_paths: &ContigPaths,
    spec: &FieldSpec, width: usize, sub: FieldSub,
    src_calls: &[CallSrc], n_samples_orig: usize,
) -> Result<(), ConversionError> {
    let vk_view = open_source_field(src_paths, spec, sub, n_samples_orig)?;
    // A flipped call reads from the DENSE sub of the same class.
    let dense_sub = match sub {
        FieldSub::VkSnp => FieldSub::DenseSnp,
        FieldSub::VkIndel => FieldSub::DenseIndel,
        _ => unreachable!("slice_field_var_key takes only var_key subs"),
    };
    let d_view = open_source_field(src_paths, spec, dense_sub, n_samples_orig)?;
    if vk_view.is_none() && d_view.is_none() { return Ok(()); }

    let mut out = Vec::with_capacity(src_calls.len() * width);
    for src in src_calls {
        match *src {
            CallSrc::VarKey { call } => {
                let v = vk_view.as_ref().expect("unflipped call needs the source var_key sub");
                out.extend_from_slice(v.bytes_at(call));
            }
            CallSrc::Dense { row, s_orig } => {
                let v = d_view.as_ref().expect("flipped call needs the source dense sub");
                match spec.category {
                    // INFO is one element per dense ROW.
                    FieldCategory::Info => out.extend_from_slice(v.bytes_at(row)),
                    // FORMAT is (row, sample) -> the sample's own value.
                    FieldCategory::Format => {
                        out.extend_from_slice(v.bytes_at(row * n_samples_orig + s_orig))
                    }
                }
            }
        }
    }
    write_field_values(out_paths, spec, sub, &out)
}
```

- [ ] **Step 4: Handle `RowSrc::VarKey` in the dense field gather**

```rust
fn slice_field_dense(
    src_paths: &ContigPaths, out_paths: &ContigPaths,
    spec: &FieldSpec, width: usize, sub: FieldSub,
    src_rows: &[RowSrc], sample_orig_idx: &[usize], n_samples_orig: usize,
) -> Result<(), ConversionError> {
    let d_view = open_source_field(src_paths, spec, sub, n_samples_orig)?;
    let vk_sub = match sub {
        FieldSub::DenseSnp => FieldSub::VkSnp,
        FieldSub::DenseIndel => FieldSub::VkIndel,
        _ => unreachable!("slice_field_dense takes only dense subs"),
    };
    let vk_view = open_source_field(src_paths, spec, vk_sub, n_samples_orig)?;
    if d_view.is_none() && vk_view.is_none() { return Ok(()); }

    // The field's missing value, encoded at the field's on-disk dtype. Same
    // sentinel rvk.rs writes for a dense non-carrier (src/rvk.rs, dense push).
    let sentinel = spec.encode_scalar(spec.missing_sentinel());   // -> Vec<u8>, width bytes

    let mut out: Vec<u8> = Vec::new();
    for src in src_rows {
        match src {
            RowSrc::Dense { row } => {
                let v = d_view.as_ref().expect("unflipped row needs the source dense sub");
                match spec.category {
                    FieldCategory::Info => out.extend_from_slice(v.bytes_at(*row)),
                    FieldCategory::Format => {
                        for &orig in sample_orig_idx {
                            out.extend_from_slice(v.bytes_at(row * n_samples_orig + orig));
                        }
                    }
                }
            }
            RowSrc::VarKey { per_sample_call, info_call } => {
                let v = vk_view.as_ref().expect("flipped row needs the source var_key sub");
                match spec.category {
                    // INFO: one element per row; any call of the variant carries it.
                    FieldCategory::Info => out.extend_from_slice(v.bytes_at(*info_call)),
                    // FORMAT: a full n_subset-wide column; NON-CARRIERS get the sentinel.
                    FieldCategory::Format => {
                        for call in per_sample_call {
                            match call {
                                Some(c) => out.extend_from_slice(v.bytes_at(*c)),
                                None => out.extend_from_slice(&sentinel),
                            }
                        }
                    }
                }
            }
        }
    }
    write_field_values(out_paths, spec, sub, &out)
}
```

If `FieldSpec` has no `encode_scalar`, add one next to `missing_sentinel` in `src/field.rs`: match on `self.dtype` and write `missing_sentinel()` as that dtype's LE bytes. Reuse whatever `finalize_fields` already uses to narrow `f64` → the on-disk dtype rather than writing a second encoder — grep `src/field_finalize.rs` for the narrowing helper and call it.

- [ ] **Step 5: Run the tests**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice'`
Expected: PASS — all three new field-flip tests plus every earlier test (the `Preserve` byte-parity test still guards the unflipped path).

- [ ] **Step 6: Commit**

```bash
git add src/svar2_slice.rs src/field.rs tests/test_svar2_slice.rs
git commit -m "feat(svar2): carry INFO/FORMAT fields across representation flips"
```

---

### Task 5: LUT compaction

The deleted pipeline rebuilt the long-allele LUT from scratch (compacting it). Without compaction a re-routed view can carry dead LUT rows and come out **larger** than the path it replaces — regressing the one axis `reroute=True` exists for.

**Files:**
- Modify: `src/svar2_slice.rs` (replace the verbatim LUT copy at `src/svar2_slice.rs:248-259`)
- Test: `tests/test_svar2_slice.rs`

**Interfaces:**
- Consumes: `svar2_codec::{decode_key, encode_lookup, DecodedKey}`; the output indel key streams from `emit_var_key` / `emit_dense`.
- Produces: `compact_lut(out_paths, src_paths, vk_indel_keys: &mut [u32], dense_indel_keys: &mut [u32]) -> Result<(), ConversionError>` — call it **before** the indel `alleles.bin` files are written, since it rewrites keys.

- [ ] **Step 1: Write the failing tests**

```rust
/// A subset that references only some long alleles must drop the unreferenced
/// LUT rows and renumber the surviving Lookup keys.
#[test]
fn lut_is_compacted_and_keys_renumbered() {
    // Store with 3 long-allele indels, one per sample.
    let src = fixture_long_allele_store(/*n_samples*/ 3);
    let out = tempfile::tempdir().unwrap();
    let outp = out.path().to_str().unwrap();

    // Keep sample 1 only -> exactly 1 of the 3 LUT rows is still referenced.
    svar2_slice::slice_contig_genos(
        &src.path, outp, "chr1", &[1], 2, &[(0, u32::MAX)],
        OverlapMode::Variant, svar2_slice::Routing::Preserve,
    ).unwrap();

    assert_eq!(lut_row_count(outp, "chr1"), 1);
    // ...and the ALT still decodes correctly through the renumbered key.
    assert_eq!(decode_alt_of_only_indel(outp, "chr1"), src.alt_of_sample(1));
}

/// Full coverage references every LUT row, so compaction is the IDENTITY --
/// this is what keeps the byte-parity identity test valid.
#[test]
fn lut_compaction_is_identity_at_full_coverage() {
    let src = fixture_long_allele_store(3);
    let out = tempfile::tempdir().unwrap();
    let outp = out.path().to_str().unwrap();
    svar2_slice::slice_contig_genos(
        &src.path, outp, "chr1", &[0, 1, 2], 2, &[(0, u32::MAX)],
        OverlapMode::Variant, svar2_slice::Routing::Preserve,
    ).unwrap();
    assert_lut_files_byte_equal(&src.path, outp, "chr1");
}
```

- [ ] **Step 2: Run to confirm they fail**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice lut_'`
Expected: `lut_is_compacted_and_keys_renumbered` FAILS (`lut_row_count` is 3 — the verbatim copy); `lut_compaction_is_identity_at_full_coverage` PASSES already (a verbatim copy is trivially identity).

- [ ] **Step 3: Implement `compact_lut`**

```rust
/// Rebuild the shared indel LUT with ONLY the rows the sliced output still
/// references, renumbering the surviving `Lookup` keys in place.
///
/// Correct because an indel key is either self-contained (`Inline` / `PureDel`)
/// or a `Lookup { row }` into this contig's LUT -- so remapping every Lookup row
/// and rewriting the table together is closed over the output's key set.
fn compact_lut(
    src_paths: &ContigPaths,
    out_paths: &ContigPaths,
    vk_indel_keys: &mut [u32],
    dense_indel_keys: &mut [u32],
) -> Result<(), ConversionError> {
    create_dir(&out_paths.shared_indel_dir())?;
    if !src_paths.long_alleles_bin().exists() {
        return Ok(());
    }

    // 1. Which source rows survive?
    let mut referenced: Vec<u32> = Vec::new();
    for k in vk_indel_keys.iter().chain(dense_indel_keys.iter()) {
        if let DecodedKey::Lookup { row } = decode_key(*k) {
            referenced.push(row);
        }
    }
    referenced.sort_unstable();
    referenced.dedup();

    // 2. Read the source LUT (CSR: offsets.npy of u64 + a flat bytes blob).
    let src_bytes = fs::read(src_paths.long_alleles_bin()) /* map_err -> ConversionError::Io */;
    let src_offsets: Array1<u64> = ndarray_npy::read_npy(src_paths.long_allele_offsets()) /* map_err */;

    // 3. Rebuild, preserving ascending source-row order (so full coverage is a
    //    byte-identical no-op).
    let mut new_bytes: Vec<u8> = Vec::new();
    let mut new_offsets: Vec<u64> = vec![0];
    let mut remap: HashMap<u32, u32> = HashMap::with_capacity(referenced.len());
    for (new_row, &old_row) in referenced.iter().enumerate() {
        let s = src_offsets[old_row as usize] as usize;
        let e = src_offsets[old_row as usize + 1] as usize;
        new_bytes.extend_from_slice(&src_bytes[s..e]);
        new_offsets.push(new_bytes.len() as u64);
        remap.insert(old_row, new_row as u32);
    }

    // 4. Renumber every Lookup key in the OUTPUT streams.
    for k in vk_indel_keys.iter_mut().chain(dense_indel_keys.iter_mut()) {
        if let DecodedKey::Lookup { row } = decode_key(*k) {
            *k = encode_lookup(remap[&row]);
        }
    }

    write_bytes(&out_paths.long_alleles_bin(), &new_bytes)?;
    write_offsets(&out_paths.long_allele_offsets(), &new_offsets)
}
```

- [ ] **Step 4: Wire it into `slice_genos_inner` BEFORE the indel emits**

The indel key vectors must be compacted before `emit_var_key` / `emit_dense` serialize them. Restructure so `route` hands back the plan, then:

```rust
// Collect the output indel keys, compact the LUT + renumber, then emit.
let mut vk_indel_keys: Vec<u32> = plan.vk_indel.iter().map(|&(_, _, _, k)| k).collect();
let mut d_indel_keys:  Vec<u32> = plan.d_indel.iter().map(|(_, _, k, _)| *k).collect();
compact_lut(&src_paths, &out_paths, &mut vk_indel_keys, &mut d_indel_keys)?;
for (e, k) in plan.vk_indel.iter_mut().zip(&vk_indel_keys) { e.3 = *k; }
for (e, k) in plan.d_indel.iter_mut().zip(&d_indel_keys)  { e.2 = *k; }
```

Then delete the verbatim `copy_file` LUT block (`src/svar2_slice.rs:248-259`).

**Order matters:** `write_max_del` reads the *output* indel key streams and must run **after** the emits, as it does today. Renumbering does not change `deletion_len` (a `Lookup` key's deletion length is 0), so `max_del` is unaffected — but keep the ordering anyway.

- [ ] **Step 5: Run the tests**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice'`
Expected: PASS — both LUT tests, and **the byte-parity identity test still passes** (full coverage ⇒ every row referenced ⇒ compaction is the identity).

- [ ] **Step 6: Commit**

```bash
git add src/svar2_slice.rs tests/test_svar2_slice.rs
git commit -m "feat(svar2): compact the long-allele LUT when slicing a view"
```

---

### Task 6: `run_slice_view` — routing arg, cost-model inputs, rayon across contigs

**Files:**
- Modify: `src/lib.rs` (`run_slice_view`, ~`src/lib.rs:670-870`)
- Test: `tests/test_svar2_slice.rs` (a multi-contig determinism test)

**Interfaces:**
- Produces: `run_slice_view(store_path, out_dir, contigs, samples, regions, regions_overlap, merge_overlapping, fields, reference=None, reroute=false, max_threads=None, overwrite=false)`.
- Consumes: `slice_contig(..., routing)` (Tasks 2–5).

- [ ] **Step 1: Write the failing test — `threads` must not change the output**

```rust
/// Contigs are independent: threading changes wall time, never bytes.
#[test]
fn multi_contig_slice_is_thread_invariant() {
    let src = fixture_multi_contig_store(&["chr1", "chr2", "chr3"]);
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    slice_all_contigs(&src, a.path(), /*threads*/ Some(1));
    slice_all_contigs(&src, b.path(), /*threads*/ Some(4));
    for chrom in ["chr1", "chr2", "chr3"] {
        assert_sidecars_byte_equal(a.path().to_str().unwrap(),
                                   b.path().to_str().unwrap(), chrom);
    }
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice multi_contig'`
Expected: FAIL — no `threads` parameter exists yet.

- [ ] **Step 3: Add `reroute` + `max_threads` to the pyfunction signature**

```rust
#[pyo3(signature = (store_path, out_dir, contigs, samples, regions, regions_overlap,
                    merge_overlapping, fields, reference=None, reroute=false,
                    max_threads=None, overwrite=false))]
```

Map `reroute` → `Routing`:

```rust
let routing = if reroute { Routing::Recompute } else { Routing::Preserve };
```

- [ ] **Step 4: Compute the cost-model inputs once, outside the contig loop**

Mirror `src/rvk.rs:230-238` exactly:

```rust
// Signatures ride along ONLY when a reference was given (mutcat is recomputed
// as a post-pass below), so the sidecar term is live iff `reference.is_some()`.
let sidecar_bits_enabled = reference.is_some();
let info_bits: u64 = field_specs.iter()
    .filter(|f| f.category == FieldCategory::Info)
    .map(|f| f.dtype.width_bytes().unwrap_or(4) as u64 * 8)
    .sum();
let format_bits: u64 = field_specs.iter()
    .filter(|f| f.category == FieldCategory::Format)
    .map(|f| f.dtype.width_bytes().unwrap_or(4) as u64 * 8)
    .sum();
```

Pass all three into `slice_contig`. (`unwrap_or(4)` is unreachable here — a finished store's specs always have a concrete dtype — but keep it identical to `rvk.rs` so the two cannot silently diverge.)

- [ ] **Step 5: Parallelize across contigs with rayon**

Follow `run_conversion` (`src/lib.rs:141-190`) for thread discovery, but **do NOT use `budget::plan_thread_budget`** — it splits a budget into htslib + pipeline threads per chrom, and the slicer has neither.

```rust
let results: Vec<Result<usize, ConversionError>> = py.detach(|| {
    let available_cores = match max_threads {
        Some(t) if t > 0 => t,
        _ => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
    };
    // One rayon task per contig; the slicer spawns no threads of its own.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(available_cores.min(contigs.len()).max(1))
        .thread_name(|i| format!("slice-{}", i))
        .build()
        .unwrap();
    pool.install(|| {
        contigs.par_iter().map(|chrom| {
            let n = slice_contig(&store_path, &out_dir, chrom, &sample_idx, ploidy,
                                 &regions_of[chrom], overlap, &field_specs, routing,
                                 sidecar_bits_enabled, info_bits, format_bits)?;
            if let Some(fasta) = reference.as_deref() {
                annotate_output_contig(&out_dir, chrom, fasta, n_subset, ploidy)?;  // existing post-pass
            }
            Ok(n)
        }).collect()
    })
});
```

Keep the existing **fail-fast band** (reroute validation, unknown contig/sample, `validate_contigs_in_fasta`) ahead of the pool, and keep `write_meta` after it. Collect per-contig errors exactly as `run_conversion` does.

**Memory note for the docstring (Task 7):** peak is now O(output per contig) × concurrent contigs, and with `reference=` each in-flight contig also holds that contig's reference sequence (~250 MB for chr1).

- [ ] **Step 6: Run the tests**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features --test test_svar2_slice && cargo clippy --no-default-features --all-targets -- -D warnings'`
Expected: PASS + clean.

- [ ] **Step 7: Commit**

```bash
git add src/lib.rs tests/test_svar2_slice.rs
git commit -m "feat(svar2): run_slice_view takes a routing policy and slices contigs in parallel"
```

---

### Task 7: Python + CLI — dispatch both routings to the slicer, the `"auto"` rule, real `threads`

**Files:**
- Modify: `python/genoray/_svar2.py:332-470` (`write_view`)
- Modify: `python/genoray/_cli/__main__.py:302` (`reroute` default) and its docstring
- Test: `tests/test_svar2_write_view.py`, `tests/cli/test_view_svar2_cli.py`

**Interfaces:**
- Consumes: `_core.run_slice_view(..., reroute: bool, max_threads: int | None, ...)` (Task 6).

- [ ] **Step 1: Write the failing tests**

```python
def test_auto_resolves_to_preserve_when_format_carried(svar2_with_format, tmp_path):
    """A dense->var_key flip has no slot for non-carrier FORMAT, so "auto"
    must NOT re-route when a FORMAT field is carried."""
    out = tmp_path / "view.svar2"
    svar2_with_format.write_view(
        regions=None, samples=["S0"], output=out,
        fields=["DP"],            # a FORMAT field
        reroute="auto",
    )
    # The source-dense variant must still be dense (representation preserved).
    assert variant_is_dense(out, "chr1", DENSE_POS)


def test_auto_reroutes_when_only_info_carried(svar2_with_info, tmp_path):
    out = tmp_path / "view.svar2"
    svar2_with_info.write_view(
        regions=None, samples=["S0"], output=out,
        fields=["AF"],            # INFO only -> no fidelity risk -> re-route
        reroute="auto",
    )
    assert not variant_is_dense(out, "chr1", DENSE_POS)   # flipped to var_key


def test_reroute_true_now_carries_fields(svar2_with_format, tmp_path):
    """Was: ValueError('field carry-through is not yet implemented')."""
    out = tmp_path / "view.svar2"
    svar2_with_format.write_view(
        regions=None, samples=["S0", "S1"], output=out,
        fields=["DP"], reroute=True,
    )
    view = genoray.SparseVar2(out, fields=["DP"])
    assert view.available_fields["DP"] is not None


def test_reroute_true_recomputes_mutcat(svar2_store, reference_fasta, tmp_path):
    """Was: `reference` silently discarded on reroute=True."""
    out = tmp_path / "view.svar2"
    svar2_store.write_view(regions=None, samples=["S0"], output=out,
                           reference=reference_fasta, reroute=True)
    assert (out / "chr1" / "mutcat" / "var_key_snp" / "code.bin").exists()


def test_threads_is_honored_and_output_is_invariant(multi_contig_svar2, tmp_path):
    a, b = tmp_path / "a.svar2", tmp_path / "b.svar2"
    multi_contig_svar2.write_view(regions=None, samples=None, output=a, threads=1)
    multi_contig_svar2.write_view(regions=None, samples=None, output=b, threads=4)
    assert_stores_byte_equal(a, b)
```

Plus a CLI test asserting `genoray view` defaults to `"auto"` (not `True`), so CLI users get the same FORMAT protection:

```python
def test_cli_view_defaults_to_auto(svar2_with_format, tmp_path):
    run_cli(["view", str(svar2_with_format.path), str(tmp_path / "v.svar2"),
             "-s", "S0", "-f", "DP"])
    assert variant_is_dense(tmp_path / "v.svar2", "chr1", DENSE_POS)
```

- [ ] **Step 2: Run to confirm they fail**

Run: `pixi run develop && pixi run pytest tests/test_svar2_write_view.py -k "auto or reroute_true or threads" -v`
Expected: FAIL — `reroute=True` still raises `ValueError` on fields; `"auto"` resolves unconditionally to `True`.

- [ ] **Step 3: Implement the `"auto"` rule and unify the dispatch**

Replace the `reroute` resolution and the two-way dispatch in `python/genoray/_svar2.py` (currently `_svar2.py:396-470`):

```python
if reroute not in (True, False, "auto"):
    raise ValueError(f"reroute must be 'auto', True, or False; got {reroute!r}")

# ... resolve `fields_to_write` (unchanged) ...

if reroute == "auto":
    # A dense->var_key flip stores one value per CARRIER CALL and has no slot
    # for a non-carrier sample's FORMAT value, so re-routing a source-dense
    # variant would silently drop it. Prefer fidelity when FORMAT is in play;
    # take the size-optimal re-route otherwise (that is also where the win is:
    # genotype-only / INFO-only views).
    carries_format = any(
        self.available_fields[key].category == "FORMAT" for key in fields_to_write
    )
    reroute = not carries_format

_core.run_slice_view(
    str(self.path), str(output), contigs, sample_idx, regions_arr,
    regions_overlap, merge_overlapping, field_tuples, reference_str,
    reroute,            # <- routing policy; both paths now slice
    threads,            # <- max_threads: real, not ignored
    overwrite,
)
```

Delete the `run_view_pipeline` branch entirely.

- [ ] **Step 4: Rewrite the `write_view` docstring**

It must state, in the `reroute` / `fields` / `reference` / `threads` sections:
- `reroute=True` re-runs the cost model on the subset (size-optimal); `reroute=False` preserves each variant's source representation; **both** carry `fields` and recompute `mutcat` from `reference`.
- **`"auto"` resolves to `False` when any FORMAT field is carried, `True` otherwise** — and *why*: a dense→var_key flip drops non-carrier FORMAT values.
- `threads` caps concurrent contigs (autodetect when `None`), same as `from_vcf`. **Delete** the "accepted on both paths for interface parity [but ignored]" caveat at `_svar2.py:379-382`.
- Peak memory is O(output **per contig**) × `threads`; with `reference=`, each in-flight contig also holds that contig's reference sequence.

- [ ] **Step 5: Change the CLI default to `"auto"`**

`python/genoray/_cli/__main__.py:302`:

```python
reroute: bool | Literal["auto"] = "auto",
```

and update its docstring block (`_cli/__main__.py:335-352`) to state the `"auto"` rule and the non-carrier-FORMAT loss. **Delete** the stale text at `_cli/__main__.py:335-338` claiming `reroute` "does not carry fields through … raises `ValueError`" — that is no longer true.

- [ ] **Step 6: Run the tests**

Run: `pixi run develop && pixi run pytest tests/test_svar2_write_view.py tests/cli/test_view_svar2_cli.py -v`
Expected: PASS — the new tests plus all 21 pre-existing `write_view` tests.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_svar2.py python/genoray/_cli/__main__.py tests/test_svar2_write_view.py tests/cli/test_view_svar2_cli.py
git commit -m "feat(svar2)!: write_view routes both backends through the slicer; auto prefers fidelity when FORMAT is carried"
```

---

### Task 8: Differential test — new `reroute=True` vs. the OLD pipeline

**This runs BEFORE the deletion in Task 9, and it is the only evidence that we have not regressed the mode we are reimplementing.** It is a *temporary* test: it calls both `_core` entry points directly, and Task 9 deletes it along with the old path. That is intentional — say so in the commit message.

**Files:**
- Create: `tests/test_svar2_view_differential.py` (deleted again in Task 9)

- [ ] **Step 1: Write the differential test**

```python
"""TEMPORARY. Proves the slicer-backed reroute=True matches the pipeline-backed
one before the pipeline is deleted (see the unify-routing plan, Task 8/9).
Deleted together with run_view_pipeline in Task 9."""

import pytest
from genoray import _core


@pytest.mark.parametrize("samples", [["S0"], ["S0", "S3"], None])
@pytest.mark.parametrize("overlap", ["pos", "record", "variant"])
def test_new_reroute_matches_old_pipeline(svar2_store, samples, overlap, tmp_path):
    old, new = tmp_path / "old.svar2", tmp_path / "new.svar2"
    idx = sample_indices(svar2_store, samples)

    _core.run_view_pipeline(          # the path being deleted
        str(svar2_store.path), str(old), CONTIGS, idx, REGIONS, overlap,
        False, [], None, None, False,
    )
    _core.run_slice_view(             # the replacement, reroute=True
        str(svar2_store.path), str(new), CONTIGS, idx, REGIONS, overlap,
        False, [], None, True, None, False,
    )

    # 1. Same variant set + same decoded genotypes.
    assert_decoded_genotypes_equal(old, new, samples)
    # 2. Same per-variant routing -- the whole reason reroute=True exists.
    assert routing_of(new) == routing_of(old)     # _core.svar2_variant_stats -> src_dense
    # 3. No size regression (LUT compaction may make the new one SMALLER).
    assert store_size(new) <= store_size(old)
```

`routing_of` wraps the existing `_core.svar2_variant_stats(store, chrom, subset)` and returns `{(pos, key): src_dense}`.

- [ ] **Step 2: Run it**

Run: `pixi run develop && pixi run pytest tests/test_svar2_view_differential.py -v`
Expected: **PASS for every parametrization.** If routing differs on any variant, the `Recompute` cost-model inputs are wrong — fix Task 3/6 (usual suspects: `n_samples` passed as the source cohort instead of the subset; `sidecar_bits` not gated on `reference.is_some()`; `info_bits`/`format_bits` omitted). Do not proceed to Task 9 until this is green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar2_view_differential.py
git commit -m "test(svar2): differential -- slicer reroute=True matches the pipeline it replaces"
```

---

### Task 9: Delete the pipeline-backed view path

**Files:**
- Delete: `src/svar2_source.rs`, `tests/test_svar2_source.rs`, `tests/test_view_pipeline.rs`, `tests/test_svar2_view_differential.py`
- Modify: `src/orchestrator.rs:44-60,271-283` (drop `SourceSpec::Svar2`), `src/lib.rs` (drop `run_view_pipeline` + its `mod svar2_source;`)

- [ ] **Step 1: Delete `Svar2Source` and its `SourceSpec` variant**

Remove `src/svar2_source.rs`, the `mod svar2_source;` declaration, the `SourceSpec::Svar2 { .. }` variant and its `Box::new(crate::svar2_source::Svar2Source::new(..))` arm in `src/orchestrator.rs`, and the whole `run_view_pipeline` pyfunction + its `#[pymodule]` registration in `src/lib.rs`.

**Leave `SourceSpec::Svar1` and `src/svar1_reader.rs` completely alone** — the SVAR1 view path still uses the pipeline.

- [ ] **Step 2: Delete the now-dead tests**

`tests/test_svar2_source.rs` and `tests/test_view_pipeline.rs` test only the deleted path. `tests/test_svar2_view_differential.py` cannot run without it. Delete all three.

Any test in them that covers the *shared selection predicate* (`OverlapMode` / `query_window` / `keeps`) rather than the pipeline **must be moved to `tests/test_svar2_slice.rs`, not deleted** — that logic survives in `src/svar2_view.rs`. Grep both files for `OverlapMode` before deleting and port what applies.

- [ ] **Step 3: Verify the whole suite**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$$ && pixi run bash -lc 'cargo test --no-default-features && cargo clippy --no-default-features --all-targets -- -D warnings'`
Expected: PASS, clean, and **no dead-code warnings** (a leftover warning means something still references the deleted path).

Run: `pixi run develop && pixi run test`
Expected: the full Python suite passes. (Note: `tests/test_vcf.py` has *pre-existing* failures on this branch when the `gen_from_vcf.sh` fixtures are missing — `pixi run test` regenerates them. Failures there are unrelated to this work; confirm they are the same ones present before Task 1.)

- [ ] **Step 4: Commit**

```bash
git rm src/svar2_source.rs tests/test_svar2_source.rs tests/test_view_pipeline.rs tests/test_svar2_view_differential.py
git add -A src/ python/ tests/
git commit -m "refactor(svar2)!: delete the pipeline-backed SVAR2 view path

Both routings now slice. Removes Svar2Source, SourceSpec::Svar2 and
run_view_pipeline, along with the temporary differential test that proved
the replacement matches it. The eager Vec<RawRecord> materialization (31 GB
peak for a 0.2 GB output) goes with them. SVAR1's view path is untouched."
```

---

### Task 10: Documentation

**Files:**
- Modify: `skills/genoray-api/SKILL.md`, `CHANGELOG.md`, `docs/roadmap/data-model.md`
- Modify: `docs/superpowers/specs/2026-07-13-svar2-write-view-fields-design.md`, `docs/superpowers/plans/2026-07-13-svar2-write-view-fields.md` (mark superseded)

- [ ] **Step 1: `skills/genoray-api/SKILL.md`** — mandatory (project CLAUDE.md). Update `write_view`: `fields` and `reference` now work on **both** routings; `threads` is real; and the **`"auto"` rule with its rationale** (a dense→var_key flip drops non-carrier FORMAT values). Update the backend-choice table — the old "INFO/FORMAT fields: not yet carried (raises)" and "Memory: eager materialization" rows for `reroute=True` are both obsolete.

- [ ] **Step 2: `CHANGELOG.md`** under `## Unreleased` (do not touch versioned sections):

```markdown
- `SparseVar2.write_view` now routes **both** `reroute=True` and `reroute=False`
  through the array-slicer. `reroute=True` gains INFO/FORMAT carry-through and
  `mutcat` recompute (both previously raised / were ignored) and drops from
  O(variants x haps) peak memory (~31 GB for a whole-chr21 germline view) to
  O(output per contig).
- `reroute="auto"` now resolves to `False` when any FORMAT field is carried and
  `True` otherwise. Re-routing a source-dense variant to var_key has no slot for
  non-carrier FORMAT values, so the default prefers fidelity when FORMAT is in
  play and size-optimality otherwise.
- `write_view`'s `threads` is honored (it was accepted and ignored on the slicer
  path); contigs are now sliced in parallel.
- The long-allele LUT is compacted when slicing, instead of copied verbatim.
```

- [ ] **Step 3: `docs/roadmap/data-model.md`** — update M9 to describe one slicer with two routing policies, not two backends.

- [ ] **Step 4: Mark the superseded docs.** Add to the top of both `2026-07-13-svar2-write-view-fields-design.md` and `docs/superpowers/plans/2026-07-13-svar2-write-view-fields.md`:

```markdown
> **SUPERSEDED** by `docs/superpowers/specs/2026-07-13-svar2-view-unify-routing-design.md`.
> This plan carried fields through the `reroute=True` **re-conversion** path, which no
> longer exists — `reroute=True` is now a routing policy inside the slicer, which
> already carried fields. Never implemented. Kept for history.
```

- [ ] **Step 5: Commit**

```bash
git add skills/genoray-api/SKILL.md CHANGELOG.md docs/
git commit -m "docs(svar2): document the unified view backend and the auto routing rule"
```

---

### Task 11: Benchmark gate — retire the eager-materialization warning

**Files:**
- Modify: `docs/superpowers/notes/2026-07-13-svar2-eager-materialization-benchmark.md`
- Modify: `scripts/svar2_eager_bench.py` (point it at the new `reroute=True`)

- [ ] **Step 1: Re-run the benchmark on the new path**

The store is `data/chr21.germline.svar2` (whole chr21, 3202 samples, 1,001,385 variants). Run on **node-local disk** — the merge stage mmaps output and SIGBUSes on NFS:

```bash
sbatch -p carter-compute -c 16 --mem=64G --wrap "cd $(pwd) && export CARGO_TARGET_DIR=/tmp/genoray-target-\$\$ && pixi run develop && pixi run -e py310 python scripts/svar2_eager_bench.py --store data/chr21.germline.svar2 --out-dir /scratch/\$USER/bench --ks 100 500 1000 3202 --threads 16"
```

Expected: peak RSS collapses from the recorded **0.84 / 2.99 / 6.89 / 30.96 GB** to roughly the output size (0.005 / 0.025 / 0.052 / 0.203 GB) plus a working set — i.e. **no longer linear in `k` at ~9.7 MB/sample**. Wall time should also drop (a gather beats a full re-conversion).

- [ ] **Step 2: If wall time regressed, investigate before proceeding**

The old path had `max_threads` pipeline parallelism *within* a contig; the new one parallelizes *across* contigs, and this benchmark is single-contig — so it deliberately isolates the slicer. A regression here is real and must be understood, not averaged away by adding contigs.

- [ ] **Step 3: Rewrite the note**

Replace the TL;DR and the Verdict. The **"Do NOT advertise cohort-scale whole-store copies"** gate is *lifted* — that is what this work existed to do. Keep the old table for historical contrast, clearly labelled as the deleted pipeline path, and add the new measurements beside it.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/notes/2026-07-13-svar2-eager-materialization-benchmark.md scripts/svar2_eager_bench.py
git commit -m "docs(svar2): re-benchmark write_view on the slicer; lift the cohort-scale gate"
```

---

### Task 12: Update the PR

- [ ] **Step 1: Rewrite PR #105's "Deliberately deferred" section.** All four call-outs (reroute=True fields, reroute=True `mutcat`, eager materialization, LUT compaction) are **resolved**. Replace the section with a short "Resolved in this PR" note and update the backend-choice table (the `reroute=True` "fields: not yet carried" and "memory: eager materialization" rows are gone; add the `"auto"` rule).

- [ ] **Step 2: Push.**

```bash
git push origin worktree-svar2-merge-split-view
gh pr view 105 --json url
```

---

## Self-Review

**Spec coverage:** §A `Routing` → Task 2/3. §B cost-model inputs → Task 3 + Task 6 Step 4. §C flips (genotypes) → Task 3; (fields) → Task 4; emit order → Task 3 Step 4. §D LUT compaction → Task 5. §D2 rayon + `threads` → Task 6, Task 7. §E deletions + `svar2_view.rs` → Task 1, Task 9. §F Python → Task 7. `"auto"` rule → Task 7 (+ CLI default). Testing §1 identity → Tasks 2, 3; §2 differential → Task 8; §3 flips → Tasks 3, 4; §4 fields change routing → covered by Task 3's cost-model inputs and asserted in Task 4's dense-flip fixture (whose flip only occurs *because* `format_bits` is nonzero); §5 reference changes routing → Task 7's `mutcat` test; §6 `"auto"` → Task 7; §7 overlap-mode parity → Task 8's parametrization; §8 threads → Tasks 6, 7; §9 benchmark → Task 11. Docs → Task 10.

**Placeholders:** none. Every code step carries real code; every test step names the command and the expected result. The `route`/`RoutePlan` bodies elide only the `Preserve` arm already written in Task 2.

**Type consistency:** `Routing`, `CallSrc`, `RowSrc`, `GatheredCall`, `GatheredRow` are declared once (Task 2 Interfaces) and used with those exact field names in Tasks 3–6. `slice_contig`'s final signature — `(src_store, out_store, chrom, sample_orig_idx, ploidy, regions, overlap, fields, routing, sidecar_bits_enabled, info_bits, format_bits)` — is reached incrementally (Task 2 adds `routing`; Task 3 adds the three cost-model args) and matches the `run_slice_view` call in Task 6 Step 5.

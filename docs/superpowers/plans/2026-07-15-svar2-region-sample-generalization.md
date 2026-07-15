# SVAR2 Region/Sample Subsetting Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `SparseVar2.from_vcf_list`, `from_pgen`, and `from_svar1` the same `regions`/`samples`/`merge_overlapping`/`regions_overlap` subsetting `from_vcf` gained in PR #114, via one shared front-end and one shared per-record overlap filter.

**Architecture:** A backend-agnostic Python front-end coalesces requested regions into per-contig 0-based half-open genomic intervals. Each Rust record source (`VcfRecordSource`, `PgenRecordSource`, `Svar1RecordSource`) receives those intervals plus an `OverlapMode` and applies the *same* per-record `keeps` + extent predicate already used by the finished-store view path (`svar2_view::{OverlapMode, query_window, keeps}`), so conversion-time filtering is byte-identical to the store's query semantics. Sample subsetting reuses v1 `_normalize_samples`; VCF filters by name in-reader (done), PGEN uses pgenlib `change_sample_subset` + a caller-order permutation, svar1 remaps its CSR.

**Tech Stack:** Python 3.10+, Rust, PyO3, rust-htslib `IndexedReader`, `pgenlib`, cyvcf2, Polars, Cyclopts, pytest, cargo (`--no-default-features --features conversion`).

## Global Constraints

- **Stacks on PR #114.** This plan assumes #114's `from_vcf(regions=, samples=)` + CLI flags are present. Before Task 1, rebase this branch onto #114 (merge #114 to `main` and rebase, or branch from `refs/pull/114/head`). Verify with `grep -c region_ranges python/genoray/_svar2.py` returning ≥1.
- `regions=None, samples=None` MUST be byte-compatible with the current full-file conversion path for every method.
- Region strings `"chrom:start-end"` are 1-based inclusive → 0-based half-open; tuple/BED/frame inputs are already 0-based half-open. (v1 `_normalize_regions` convention.)
- `regions_overlap ∈ {"pos","record","variant"}` matches bcftools `--regions-overlap`. `variant` is **per-record** (Option A): a multiallelic record is kept whole if ANY allele's anchor-trimmed span overlaps; per-allele dropping is not performed. This MUST appear in every docstring that accepts `regions_overlap`.
- `merge_overlapping=False` raises on overlapping requested regions; `True` coalesces first.
- Samples are selected and reordered by name, **caller order preserved**, deduped by first occurrence (v1 `_normalize_samples`).
- `from_vcf_list` does NOT accept `samples` (single-sample inputs; cohort = the file set).
- `threads` stays a total process budget — do not multiply it across readers.
- Never edit `CHANGELOG.md`; write Conventional Commit messages. All commits end with the repo's `Co-Authored-By` trailer.
- CI gate per task: the focused pytest + cargo commands listed in that task; `pixi run pytest tests -m "not network"` and `pixi run prek run --all-files` before PR readiness. Rust conversion tests run as `pixi run bash -lc 'cargo test --no-default-features --features conversion <filter>'`. Because readers now reference `svar2_view`, also run `pixi run bash -lc 'cargo check --no-default-features'` (query-only core) at least once (Task 1).
- Rust cargo hooks bus-error on the NFS `target/`: `export CARGO_TARGET_DIR=/tmp/genoray-target` before any cargo/maturin/commit in this worktree.
- Any change to a public name (`from_*` signatures, `regions_overlap` semantics) MUST update `skills/genoray-api/SKILL.md` (Task 7).

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `src/svar2_view.rs` | `OverlapMode`/`query_window`/`keeps` predicate (query-core) | Add a shared `extent_overlaps` helper + a `parse_overlap_mode(&str)`; ensure `pub` reachable from conversion readers. |
| `src/vcf_reader.rs` | Indexed VCF/BCF record source | Replace POS-only skip with `keeps`+extent using `OverlapMode`; add `overlap` arg; fetch over `query_window`. |
| `src/vcf_list_reader.rs` | k-way single-sample merge | Thread `regions` + `overlap` into per-file `VcfRecordSource::new`. |
| `src/pgen_reader.rs` | PGEN record source | Add per-record `keeps`+extent filter over genomic regions; add `sample_perm` column remap. |
| `src/svar1_reader.rs` | SVAR1 record source | Add per-record `keeps`+extent filter in cursor loop; add sample subset/remap of the CSR. |
| `src/orchestrator.rs` | `SourceSpec` + dispatch | Add `regions`/`overlap` (and svar1 `sample_idx`) fields to the enum variants; thread into each reader ctor. |
| `src/lib.rs` | pyo3 pipeline entrypoints | Add `regions`/`regions_overlap` (+ pgen/svar1 sample args) to `run_pgen_/_svar1_/_vcf_list_conversion_pipeline`. |
| `python/genoray/_svar2.py` | Public API | Shared `_normalize_svar2_regions`; wire regions/samples into `from_vcf_list`/`from_pgen`/`from_svar1`; docstrings. |
| `python/genoray/_cli/__main__.py` | `genoray write` | Route `--regions*/--samples*` to pgen/svar1; reject `--samples*` for vcf-list form. |
| `docs/source/svar.md` | User docs | All four methods, three overlap modes, per-record `variant` note. |
| `skills/genoray-api/SKILL.md` | Agent API skill | Updated signatures + `regions_overlap` semantics. |
| `tests/test_svar2_from_vcf.py`, `tests/test_svar2_from_pgen.py`, `tests/test_svar2_from_svar1.py`, `tests/cli/test_write_cli.py` | Tests | New region/sample/mode coverage. |
| `tests/test_*_e2e.rs` | Rust tests | Reader-level mode filter + sample remap. |

**Dependency graph:** Task 1 → Task 2 (foundation, sequential). Then Tasks 3 (vcf_list), 4 (pgen), 5 (svar1) are **parallelizable** (distinct readers, distinct pipeline fns, distinct `_svar2.py` methods; only low-conflict adjacency in `lib.rs`/`orchestrator.rs`). Task 6 (CLI) depends on 4+5. Task 7 (docs/skill) depends on all. Dispatch 3/4/5 with `superpowers:dispatching-parallel-agents` + `superpowers:subagent-driven-development` (Sonnet implementers); when two touch `lib.rs`/`orchestrator.rs`, land them sequentially or reconcile the adjacent hunks.

---

### Task 1: Shared Rust overlap predicate + VCF reader mode filter

Make the VCF reader apply the full pos/record/variant overlap semantics per record, reusing the store-side `OverlapMode`. Backward compatible: default mode `"pos"` reproduces #114 behavior, so `from_vcf` is unaffected until Task 2 rewires it.

**Files:**
- Modify: `src/svar2_view.rs`
- Modify: `src/vcf_reader.rs`
- Modify: `src/orchestrator.rs` (`SourceSpec::Vcf`)
- Modify: `src/lib.rs` (`run_conversion_pipeline`)
- Test: `tests/test_e2e.rs` (or a new `tests/test_regions_e2e.rs`)

**Interfaces:**
- Consumes: `svar2_view::{OverlapMode, query_window, keeps}` (existing).
- Produces:
  - `svar2_view::parse_overlap_mode(s: &str) -> Result<OverlapMode, ConversionError>` (accepts `"pos"|"record"|"variant"`).
  - `svar2_view::extent_overlaps(pos: u32, ref_len: u32, alts: &[&[u8]], ref_allele: &[u8], q_start: u32, q_end: u32) -> bool` — true iff any allele's anchor-trimmed span overlaps `[q_start,q_end)`.
  - `VcfRecordSource::new(..., regions: Vec<(u32,u32)>, overlap: OverlapMode)`.
  - `run_conversion_pipeline(..., region_ranges: Vec<(String,u32,u32)>, regions_overlap: String="pos")`.

- [ ] **Step 1: Write the failing Rust test**

Add to `tests/test_e2e.rs` (reuse the crate's existing fixture helpers in `tests/common/mod.rs`; mirror an existing conversion test's setup). This asserts the three modes on a deletion whose POS sits one base before a region.

```rust
#[test]
fn overlap_modes_delimit_by_pos_record_variant() {
    // Fixture: chr1 with a SNP at POS=10 (0-based 9) and a 3bp deletion at
    // POS=5 (0-based 4, REF="ACGT", ALT="A") whose extent [4,8) reaches into a
    // region starting at 6. Region requested: [6, 12) 0-based half-open.
    use genoray_core::svar2_view::{extent_overlaps, keeps, OverlapMode};

    // pos mode: deletion POS=4 is outside [6,12) -> dropped; SNP POS=9 kept.
    assert!(!keeps(OverlapMode::Pos, 6, 12, 4));
    assert!(keeps(OverlapMode::Pos, 6, 12, 9));
    // record mode: still POS-based, deletion POS=4 outside -> dropped.
    assert!(!keeps(OverlapMode::Record, 6, 12, 4));
    // variant mode: deletion extent [4,8) overlaps [6,12) -> kept.
    let ref_allele = b"ACGT";
    let alts: [&[u8]; 1] = [b"A"];
    assert!(extent_overlaps(4, ref_allele.len() as u32, &alts, ref_allele, 6, 12));
    // A SNP just past the region end is out under variant too.
    let snp_ref = b"C";
    let snp_alts: [&[u8]; 1] = [b"T"];
    assert!(!extent_overlaps(20, 1, &snp_alts, snp_ref, 6, 12));
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target; pixi run bash -lc 'cargo test --no-default-features --features conversion overlap_modes_delimit'`
Expected: FAIL — `parse_overlap_mode`/`extent_overlaps` not found (and `keeps` may not be `pub` to tests).

- [ ] **Step 3: Add the shared helpers to `src/svar2_view.rs`**

After `keeps` (line ~70) add:

```rust
/// Parse the public `regions_overlap` string into an [`OverlapMode`].
pub fn parse_overlap_mode(s: &str) -> Result<OverlapMode, ConversionError> {
    match s {
        "pos" => Ok(OverlapMode::Pos),
        "record" => Ok(OverlapMode::Record),
        "variant" => Ok(OverlapMode::Variant),
        other => Err(ConversionError::Input(format!(
            "regions_overlap must be 'pos', 'record', or 'variant'; got {other:?}"
        ))),
    }
}

/// True iff ANY allele's anchor-trimmed genomic span overlaps `[q_start, q_end)`.
/// Anchor-trimming removes the shared prefix/suffix of REF vs each ALT so a
/// deletion `ACGT>A` is judged on the deleted `CGT` (offset span `[pos+1,pos+4)`),
/// not the full record span — matching bcftools `--regions-overlap variant`.
/// An allele that fully trims away (a pure insertion) is a zero-width point at
/// the insertion offset and overlaps iff `q_start <= point < q_end`.
pub fn extent_overlaps(
    pos: u32,
    _ref_len: u32,
    alts: &[&[u8]],
    ref_allele: &[u8],
    q_start: u32,
    q_end: u32,
) -> bool {
    for alt in alts {
        // shared prefix
        let mut p = 0usize;
        let max_p = ref_allele.len().min(alt.len());
        while p < max_p && ref_allele[p] == alt[p] {
            p += 1;
        }
        // shared suffix (not crossing the prefix already consumed)
        let mut s = 0usize;
        while s < (ref_allele.len() - p).min(alt.len() - p)
            && ref_allele[ref_allele.len() - 1 - s] == alt[alt.len() - 1 - s]
        {
            s += 1;
        }
        let v_start = pos + p as u32;
        let ref_consumed = ref_allele.len().saturating_sub(p + s) as u32;
        let v_end = v_start + ref_consumed; // may equal v_start for insertions
        let overlaps = if v_end == v_start {
            q_start <= v_start && v_start < q_end // zero-width insertion point
        } else {
            v_start < q_end && q_start < v_end
        };
        if overlaps {
            return true;
        }
    }
    false
}
```

Ensure `keeps`, `query_window`, `OverlapMode`, and the two new fns are `pub` (they already are for the first three; confirm the module is reachable — `svar2_view` is in the query-core, always built).

- [ ] **Step 4: Run the helper test to confirm it passes**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target; pixi run bash -lc 'cargo test --no-default-features --features conversion overlap_modes_delimit'`
Expected: PASS.

- [ ] **Step 5: Add `overlap` to `SourceSpec::Vcf` and the pipeline**

In `src/orchestrator.rs`, extend the `Vcf` variant (currently `{ vcf_path, htslib_threads, regions }`):

```rust
Vcf {
    vcf_path: String,
    htslib_threads: usize,
    regions: Vec<(u32, u32)>,
    overlap: crate::svar2_view::OverlapMode,
},
```

In the dispatch `match` (around `orchestrator.rs:236`), pass `overlap` into `VcfRecordSource::new(..., regions, overlap)`.

In `src/lib.rs` `run_conversion_pipeline`, add `regions_overlap: String` to the `#[pyo3(signature = ...)]` (default `"pos".to_string()` — use `regions_overlap="pos"`) and the fn signature; parse once via `svar2_view::parse_overlap_mode(&regions_overlap)?`; set `overlap` on every `SourceSpec::Vcf` it builds (the `ranges_by_chrom` block).

- [ ] **Step 6: Implement the mode-aware per-record filter in `src/vcf_reader.rs`**

Add `overlap: OverlapMode` to the struct and `new` (last param, after `regions`). Store `query_window(&regions, overlap)` as the fetch intervals (so `Record` widens by 1), but keep the *original* `regions` for the `keeps` test. In the read loop where it currently does `if pos < start || pos >= end { continue }` (around `vcf_reader.rs:344`), bind alleles first and replace with:

```rust
let alleles = self.record.alleles();
let ref_allele = alleles[0];
if let Some((q_start, q_end)) = self.active_region() {
    let kept = match self.overlap {
        OverlapMode::Variant => svar2_view::extent_overlaps(
            pos, ref_allele.len() as u32, &alleles[1..], ref_allele, q_start, q_end,
        ),
        m => svar2_view::keeps(m, q_start, q_end, pos),
    };
    if !kept {
        continue;
    }
}
```

Here `active_region()` must return the *original* (unwidened) `[q_start, q_end)` for that record's fetch interval. Since fetch intervals are `query_window`-widened but the reader stores originals in parallel, index the original by `current_region`. Keep `advance_region` fetching the widened interval but testing against the original.

- [ ] **Step 7: Add the end-to-end Python-visible Rust test**

Extend `tests/test_e2e.rs` with a conversion test that writes a store via the pipeline with `regions_overlap="variant"` and asserts the spanning deletion is present, then `"pos"` and asserts it is absent. (Use the existing e2e harness pattern that calls the conversion entrypoint and reads back the store.)

- [ ] **Step 8: Run the full conversion + query-core checks**

Run:
```bash
export CARGO_TARGET_DIR=/tmp/genoray-target
pixi run bash -lc 'cargo test --no-default-features --features conversion'
pixi run bash -lc 'cargo check --no-default-features'   # query-only core must still build
```
Expected: all pass; query-core builds (readers referencing `svar2_view` don't break the no-conversion build).

- [ ] **Step 9: Commit**

```bash
git add src/svar2_view.rs src/vcf_reader.rs src/orchestrator.rs src/lib.rs tests/test_e2e.rs
git commit -m "feat(svar2): shared per-record overlap filter in VCF reader"
```

---

### Task 2: Shared Python region front-end + `from_vcf` rewire

Replace #114's VCF-specific `_normalize_svar2_vcf_regions` with a backend-agnostic helper that returns coalesced genomic intervals (mode-independent) and pass `regions_overlap` through to Rust as a string. Drop the `record`-mode end-widening hack (now handled by `query_window` in Rust) and the redundant Python coalesce.

**Files:**
- Modify: `python/genoray/_svar2.py` (`_normalize_svar2_vcf_regions` → `_normalize_svar2_regions`, `from_vcf`)
- Test: `tests/test_svar2_from_vcf.py`

**Interfaces:**
- Consumes: Task 1 `run_conversion_pipeline(..., region_ranges, regions_overlap)`.
- Produces: `_normalize_svar2_regions(regions, available_contigs, *, merge_overlapping) -> list[tuple[str,int,int]]` — coalesced per-contig 0-based half-open genomic intervals, mode-independent. Used by Tasks 3–5.

- [ ] **Step 1: Write the failing Python tests**

Add to `tests/test_svar2_from_vcf.py` (reuse the module's `_write_ref`/`_write_vcf` helpers). Include a fixture variant that is an indel whose POS is one base past a region end.

```python
def test_from_vcf_regions_overlap_variant_keeps_spanning_deletion(tmp_path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)  # ensure it has the deletion
    out = tmp_path / "variant_mode"
    # Region chosen so a deletion's POS is before the region but its extent reaches in.
    SparseVar2.from_vcf(out, vcf, ref, regions="chr1:7-12",
                        regions_overlap="variant", threads=1)
    sv = SparseVar2(out)
    counts = sv.var_counts("chr1", [0], [40])
    assert int(counts.sum()) >= 1  # spanning deletion present

def test_from_vcf_regions_overlap_pos_excludes_spanning_deletion(tmp_path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out_v = tmp_path / "v"; out_p = tmp_path / "p"
    SparseVar2.from_vcf(out_v, vcf, ref, regions="chr1:7-12",
                        regions_overlap="variant", threads=1, overwrite=True)
    SparseVar2.from_vcf(out_p, vcf, ref, regions="chr1:7-12",
                        regions_overlap="pos", threads=1, overwrite=True)
    assert int(SparseVar2(out_v).var_counts("chr1",[0],[40]).sum()) > \
           int(SparseVar2(out_p).var_counts("chr1",[0],[40]).sum())
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/test_svar2_from_vcf.py -k 'overlap_variant or overlap_pos' -q`
Expected: FAIL — variant/pos currently give identical results (or `variant` raised before Task 1/2).

- [ ] **Step 3: Replace the helper**

In `python/genoray/_svar2.py`, rename `_normalize_svar2_vcf_regions` to `_normalize_svar2_regions`, drop the `regions_overlap` parameter and the `end_offset` widening, and drop the `regions_overlap=="variant"` raise. Keep the parse (`_normalize_regions` per input + frame concat), the `merge_overlapping` overlap detection/raise, and the coalescing into `list[tuple[str,int,int]]`. Signature:

```python
def _normalize_svar2_regions(
    regions: "str | tuple[str, int, int] | PathLike | object | None",
    available_contigs: "Sequence[str]",
    *,
    merge_overlapping: bool,
) -> list[tuple[str, int, int]]:
```

- [ ] **Step 4: Rewire `from_vcf`**

In `from_vcf`, call `_normalize_svar2_regions(regions, all_contigs, merge_overlapping=merge_overlapping)` (no mode), keep the existing per-contig `has_variant` probe / contig ordering, and pass `regions_overlap` (the raw string) as the new trailing arg to `_core.run_conversion_pipeline(...)`. Keep the `regions_overlap` validation at the top of `from_vcf` (reject values outside the three) — or delegate to Rust's `parse_overlap_mode`; do the Python-side check so the error is raised before opening files.

- [ ] **Step 5: Update the `from_vcf` docstring**

Replace the "`regions_overlap="variant"` is reserved…" paragraph with the three-mode description and the per-record `variant` note:

```
`regions_overlap` controls which variants a region keeps, matching bcftools
--regions-overlap: "pos" (POS inside [start,end)), "record" (POS in
[start,end+1), so an indel at the region's last base is kept), or "variant"
(the anchor-trimmed variant extent overlaps the region). In "variant" mode a
multiallelic record is kept whole if ANY of its alleles truly overlaps the
region; individual non-overlapping alleles are not dropped.
```

- [ ] **Step 6: Run tests**

Run:
```bash
pixi run maturin develop
pixi run pytest tests/test_svar2_from_vcf.py -q
```
Expected: all pass, including the two new mode tests and the existing #114 region/sample tests (regression).

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_svar2.py tests/test_svar2_from_vcf.py
git commit -m "refactor(svar2): backend-agnostic region front-end; from_vcf uses OverlapMode"
```

---

### Task 3: `from_vcf_list` regions

Add `regions`/`merge_overlapping`/`regions_overlap` to `from_vcf_list` (no `samples`). Thread the coalesced genomic intervals + mode through the k-way merge into each per-file `VcfRecordSource`.

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_vcf_list`)
- Modify: `src/lib.rs` (`run_vcf_list_conversion_pipeline`)
- Modify: `src/orchestrator.rs` (`SourceSpec::VcfList`)
- Modify: `src/vcf_list_reader.rs` (`VcfListRecordSource::new`)
- Test: `tests/test_svar2_from_vcf.py` (or new `tests/test_svar2_from_vcf_list.py`), `tests/test_e2e.rs`

**Interfaces:**
- Consumes: Task 1 `VcfRecordSource::new(..., regions, overlap)`, Task 2 `_normalize_svar2_regions`.
- Produces: `from_vcf_list(..., regions=None, merge_overlapping=False, regions_overlap="pos")` (NO `samples`).

- [ ] **Step 1: Write the failing Python test**

```python
def test_from_vcf_list_regions_restricts(tmp_path):
    ref = _write_ref(tmp_path)
    a = _write_single_sample_vcf(tmp_path, "A", indexed=True)  # helper per module conventions
    b = _write_single_sample_vcf(tmp_path, "B", indexed=True)
    out = tmp_path / "vl_regions"
    SparseVar2.from_vcf_list(out, [a, b], ref, regions="chr1:1-4", threads=1)
    sv = SparseVar2(out)
    assert sv.available_samples == ["A", "B"]
    assert int(sv.var_counts("chr1", [0], [40]).sum()) >= 1
    # A variant outside chr1:1-4 must be absent vs the full conversion.
    full = tmp_path / "vl_full"
    SparseVar2.from_vcf_list(full, [a, b], ref, threads=1)
    assert int(sv.var_counts("chr1",[0],[40]).sum()) < \
           int(SparseVar2(full).var_counts("chr1",[0],[40]).sum())
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/test_svar2_from_vcf.py -k from_vcf_list_regions -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'regions'`.

- [ ] **Step 3: Thread regions through Rust**

`src/vcf_list_reader.rs` `VcfListRecordSource::new`: add params `regions: Vec<(u32,u32)>, overlap: OverlapMode`; replace the hardcoded `Vec::new()` at the per-file `VcfRecordSource::new(...)` call (line ~266-273) with `regions.clone(), overlap`.

`src/orchestrator.rs` `SourceSpec::VcfList`: add `regions: Vec<(u32,u32)>, overlap: OverlapMode`; pass into `VcfListRecordSource::new` in the dispatch.

`src/lib.rs` `run_vcf_list_conversion_pipeline`: add `region_ranges: Vec<(String,u32,u32)>` and `regions_overlap: String="pos"` to the pyo3 signature/args; build the same `ranges_by_chrom` map as `run_conversion_pipeline`; parse the mode once; set `regions`/`overlap` on each `SourceSpec::VcfList`.

- [ ] **Step 4: Wire `from_vcf_list` in Python**

Add keyword-only params after `reference`:

```python
regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
merge_overlapping: bool = False,
regions_overlap: "Literal['pos', 'record', 'variant']" = "pos",
```

Validate `regions_overlap`. After computing `contigs` (natsorted union), call `region_ranges = _normalize_svar2_regions(regions, sorted(contig_set), merge_overlapping=merge_overlapping)`; if `regions` is given, restrict `contigs` to those appearing in `region_ranges` (preserve natsorted order) and error if none remain. Pass `region_ranges` (contig-filtered, ordered) + `regions_overlap` as the two new trailing args to `_core.run_vcf_list_conversion_pipeline(...)`.

- [ ] **Step 5: Docstring**

Add a `regions`/`regions_overlap` paragraph mirroring `from_vcf`, plus: "`from_vcf_list` has no `samples` parameter — each input is single-sample and the cohort is defined by `sources`." Include the per-record `variant` note.

- [ ] **Step 6: Add a Rust e2e test**

In `tests/test_e2e.rs`, extend the existing vcf-list conversion test to pass a region range + `"pos"` and assert only in-region variants are written.

- [ ] **Step 7: Run tests**

Run:
```bash
export CARGO_TARGET_DIR=/tmp/genoray-target
pixi run bash -lc 'cargo test --no-default-features --features conversion vcf_list'
pixi run maturin develop
pixi run pytest tests/test_svar2_from_vcf.py -k from_vcf_list -q
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add python/genoray/_svar2.py src/lib.rs src/orchestrator.rs src/vcf_list_reader.rs tests/test_svar2_from_vcf.py tests/test_e2e.rs
git commit -m "feat(svar2): add regions to from_vcf_list"
```

---

### Task 4: `from_pgen` regions + samples

Add `regions`/`samples`/`merge_overlapping`/`regions_overlap` to `from_pgen`. Regions: covering-range narrow in Python + per-record `keeps`/extent filter in `PgenRecordSource`. Samples: `change_sample_subset` (sorted) + a caller-order `sample_perm` remap in the reader.

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_pgen`, `_pvar_contig_ranges` or a new POS helper)
- Modify: `src/lib.rs` (`run_pgen_conversion_pipeline`)
- Modify: `src/orchestrator.rs` (`SourceSpec::Pgen`)
- Modify: `src/pgen_reader.rs` (`PgenRecordSource`)
- Test: new `tests/test_svar2_from_pgen.py`, `tests/test_pgen_e2e.rs` (if present; else `tests/test_e2e.rs`)

**Interfaces:**
- Consumes: Task 2 `_normalize_svar2_regions`; v1 `_normalize_samples`; `svar2_view::{keeps, extent_overlaps, OverlapMode}`.
- Produces: `from_pgen(..., regions=None, samples=None, merge_overlapping=False, regions_overlap="pos")`.
  - `run_pgen_conversion_pipeline(..., region_ranges, regions_overlap, sample_perm)`.
  - `PgenRecordSource::new(reader, pvar_path, var_start, var_end, num_samples, chunk_size, regions: Vec<(u32,u32)>, overlap: OverlapMode, sample_perm: Vec<usize>)`.

- [ ] **Step 1: Write the failing Python tests**

Create `tests/test_svar2_from_pgen.py` (guard with the existing plink2/pgen availability marker used by other pgen tests; reuse fixtures from `tests/test_svar2_from_vcf.py` or `gen_from_vcf.sh`-produced pgen).

```python
def test_from_pgen_regions_restrict(tmp_path, tiny_pgen):  # tiny_pgen: fixture -> Path
    out = tmp_path / "pg_regions"
    SparseVar2.from_pgen(out, tiny_pgen, no_reference=True, regions="chr1:1-4")
    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]
    assert int(sv.var_counts("chr1", [0], [40]).sum()) >= 1

def test_from_pgen_samples_preserve_caller_order(tmp_path, tiny_pgen):
    out = tmp_path / "pg_samples"
    SparseVar2.from_pgen(out, tiny_pgen, no_reference=True, samples=["S1", "S0"])
    assert SparseVar2(out).available_samples == ["S1", "S0"]

def test_from_pgen_unknown_sample_raises(tmp_path, tiny_pgen):
    with pytest.raises(ValueError, match="not found"):
        SparseVar2.from_pgen(tmp_path/"x", tiny_pgen, no_reference=True, samples=["NOPE"])
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/test_svar2_from_pgen.py -q`
Expected: FAIL — unexpected keyword `regions`/`samples`.

- [ ] **Step 3: Add `sample_perm` + region filter to `PgenRecordSource`**

`src/pgen_reader.rs`: add fields `regions: Vec<(u32,u32)>`, `overlap: OverlapMode`, `cur_region: usize`, `sample_perm: Vec<usize>` (length = output `num_samples`; `sample_perm[out] = position within the sorted pgenlib subset`). In `refill`, the pgenlib buffer width is `2 * subset_len`; add a `subset_len` field. In `next_record`, filter by the pvar `meta.pos` against `active region` using the shared predicate (advance region when POS passes the current interval end, mirroring VCF). When copying genotype columns, gather via `sample_perm`:

```rust
for (out_s, &src_s) in self.sample_perm.iter().enumerate() {
    for p in 0..2usize {
        gt[out_s * 2 + p] = {
            let code = flat[base + src_s * 2 + p];
            if code < 0 { -1 } else { code }
        };
    }
}
```

For `variant` mode, the pvar meta gives REF/ALT bytes — call `extent_overlaps(pos, ref_len, &alt_refs, ref_bytes, q_start, q_end)`; for `pos`/`record`, call `keeps`. If the pvar cursor runs past all regions on the contig, return `Ok(None)`.

- [ ] **Step 4: Thread through orchestrator + pipeline**

`src/orchestrator.rs` `SourceSpec::Pgen`: add `regions: Vec<(u32,u32)>`, `overlap: OverlapMode`, `sample_perm: Vec<usize>`; pass into `PgenRecordSource::new`.

`src/lib.rs` `run_pgen_conversion_pipeline`: add `region_ranges: Vec<(String,u32,u32)>`, `regions_overlap: String`, `sample_perm: Vec<usize>` to signature; build `ranges_by_chrom`; parse mode; set on each `SourceSpec::Pgen` (the same `sample_perm` applies to every contig reader).

- [ ] **Step 5: Wire `from_pgen` in Python**

Add keyword-only `regions`, `samples`, `merge_overlapping`, `regions_overlap`. Steps:
1. Read psam samples (existing). If `samples` given: `selected = _normalize_samples(samples, all_psam_samples)`; compute `sel_idx = [psam_index[s] for s in selected]`; `sorted_idx = sorted(set(sel_idx))`; `sample_perm = [sorted_idx.index(i) for i in sel_idx]`; call `reader.change_sample_subset(np.asarray(sorted_idx, dtype=np.uint32))` on each `PgenReader` after construction; the `samples` list written = `selected` (caller order); `num_samples` passed = `len(selected)`. If `samples is None`: `sample_perm = list(range(n_samples))`, no subset call.
2. If `regions` given: `region_ranges = _normalize_svar2_regions(regions, contigs, merge_overlapping=merge_overlapping)`; narrow each contig's `(lo,hi)` from `_pvar_contig_ranges` to the covering index range of that contig's regions using the pvar POS column (add a helper `_pvar_covering_ranges(pvar, contigs, region_ranges) -> list[(str,(int,int))]` that searchsorts POS per contig); restrict `contigs`/`ranges` to contigs with ≥1 region.
3. Pass `region_ranges`, `regions_overlap`, `sample_perm` as new trailing args to `_core.run_pgen_conversion_pipeline(...)`.

Update the docstring: remove the "Sample subsetting: all samples converted" non-support note; add the `regions`/`samples`/`regions_overlap` paragraphs incl. the per-record `variant` note and caller-order guarantee.

- [ ] **Step 6: Add a Rust e2e test**

Assert `sample_perm` reorders columns (subset `[1,0]` → store column 0 is sample index 1's genotypes) and that a region restricts variants.

- [ ] **Step 7: Run tests**

Run:
```bash
export CARGO_TARGET_DIR=/tmp/genoray-target
pixi run bash -lc 'cargo test --no-default-features --features conversion pgen'
pixi run maturin develop
pixi run pytest tests/test_svar2_from_pgen.py -q
```
Expected: all pass (skips if plink2/pgen fixture unavailable — match the repo's existing pgen test guard).

- [ ] **Step 8: Commit**

```bash
git add python/genoray/_svar2.py src/lib.rs src/orchestrator.rs src/pgen_reader.rs tests/test_svar2_from_pgen.py tests/test_e2e.rs
git commit -m "feat(svar2): add regions and samples to from_pgen"
```

---

### Task 5: `from_svar1` regions + samples

Add `regions`/`samples`/`merge_overlapping`/`regions_overlap` to `from_svar1`. Regions: per-record `keeps`/extent filter in the cursor loop. Samples: subset + caller-order remap of the CSR (the deepest reader change).

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_svar1`)
- Modify: `src/lib.rs` (`run_svar1_conversion_pipeline`)
- Modify: `src/orchestrator.rs` (`SourceSpec::Svar1`)
- Modify: `src/svar1_reader.rs` (`Svar1RecordSource`, `build_variant_major`)
- Test: new `tests/test_svar2_from_svar1.py`, `tests/test_e2e.rs`

**Interfaces:**
- Consumes: Task 2 `_normalize_svar2_regions`; v1 `_normalize_samples`; `svar2_view::{keeps, extent_overlaps, OverlapMode}`.
- Produces: `from_svar1(..., regions=None, samples=None, merge_overlapping=False, regions_overlap="pos")`.
  - `run_svar1_conversion_pipeline(..., region_ranges, regions_overlap, sample_idx)`.
  - `Svar1RecordSource::new(..., regions: Vec<(u32,u32)>, overlap: OverlapMode, sample_idx: Vec<usize>)`.

- [ ] **Step 1: Write the failing Python tests**

Create `tests/test_svar2_from_svar1.py`. Build a SVAR1 store first (via `SparseVar.from_vcf`/`from_pgen` per the repo's svar1 fixture helper), then convert.

```python
def test_from_svar1_regions_restrict(tmp_path, tiny_svar1):
    out = tmp_path / "s1_regions"
    SparseVar2.from_svar1(out, tiny_svar1, no_reference=True, regions="chr1:1-4")
    assert int(SparseVar2(out).var_counts("chr1",[0],[40]).sum()) >= 1

def test_from_svar1_samples_reorder(tmp_path, tiny_svar1):
    out = tmp_path / "s1_samples"
    SparseVar2.from_svar1(out, tiny_svar1, no_reference=True, samples=["S1", "S0"])
    sv = SparseVar2(out)
    assert sv.available_samples == ["S1", "S0"]
    # genotype alignment: S1's calls appear under column 0
    # (compare against the full-cohort store's S1 column)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/test_svar2_from_svar1.py -q`
Expected: FAIL — unexpected keyword `regions`/`samples`.

- [ ] **Step 3: Region filter in the svar1 cursor loop**

`src/svar1_reader.rs`: add fields `regions: Vec<(u32,u32)>`, `overlap: OverlapMode`, `cur_region: usize`. In `next_record` (`:184`) cursor loop, after picking `v = self.cursor`, test `self.pos[v]` (and REF/ALT via `ref_offsets`/`alt_offsets` slices for `variant`) against the active region with `keeps`/`extent_overlaps`; if not kept, `self.cursor += 1; continue` (advance region when `pos[v]` passes the current interval). REF/ALT slices:

```rust
let r0 = self.ref_offsets[v] as usize;
let r1 = self.ref_offsets[v + 1] as usize;
let ref_allele = &self.ref_bytes[r0..r1];
let a0 = self.alt_offsets[v] as usize;
let a1 = self.alt_offsets[v + 1] as usize;
let alt_allele = &self.alt_bytes[a0..a1];  // biallelic: single ALT
```

- [ ] **Step 4: Sample subset/remap in `build_variant_major`**

`src/svar1_reader.rs` `build_variant_major` (`:17`) and `Svar1RecordSource::new`: accept `sample_idx: &[usize]` (original sample indices in caller order). Restrict the transpose to selected haps and remap to output columns: for each selected output sample `out_s` with original `orig_s = sample_idx[out_s]`, its haps are `orig_s*ploidy .. orig_s*ploidy+ploidy` mapping to output haps `out_s*ploidy..`. Set `num_samples = sample_idx.len()`, `num_haps = num_samples * ploidy`. GT fill (`:196-199`) and FORMAT fill (`:201-213`) then run over the remapped `buckets`. Ensure `s_refs.len()` passed by the orchestrator equals the subset length (see Step 5).

- [ ] **Step 5: Thread through orchestrator + pipeline**

`src/orchestrator.rs` `SourceSpec::Svar1`: add `regions`, `overlap`, `sample_idx: Vec<usize>`; pass into `Svar1RecordSource::new`. Note `s_refs` (the selected sample names) must already be the subset — the pipeline builds readers from the `samples` arg, so passing subset names makes `s_refs.len()` == subset length automatically; `sample_idx` tells the reader which *original* CSR columns those names map to.

`src/lib.rs` `run_svar1_conversion_pipeline`: add `region_ranges`, `regions_overlap`, `sample_idx: Vec<usize>`; build `ranges_by_chrom`; parse mode; set on each `SourceSpec::Svar1`.

- [ ] **Step 6: Wire `from_svar1` in Python**

Add keyword-only `regions`, `samples`, `merge_overlapping`, `regions_overlap`. After reading svar1 metadata samples:
- If `samples`: `selected = _normalize_samples(samples, meta_samples)`; `sample_idx = [meta_samples.index(s) for s in selected]`; pass `selected` as the `samples` pipeline arg and `sample_idx` as the new arg. Else `selected = meta_samples`, `sample_idx = list(range(len(meta_samples)))`.
- If `regions`: `region_ranges = _normalize_svar2_regions(regions, contigs, merge_overlapping=merge_overlapping)`; restrict `contigs` (and the parallel `starts`/`lens` / per-contig index arrays) to contigs with ≥1 region.
- Pass `region_ranges`, `regions_overlap`, `sample_idx` as new trailing args.

Update the docstring with the `regions`/`samples`/`regions_overlap` paragraphs + per-record `variant` note + caller-order guarantee.

- [ ] **Step 7: Add a Rust e2e test**

Assert region restriction and sample reorder/subset alignment at the reader level.

- [ ] **Step 8: Run tests**

Run:
```bash
export CARGO_TARGET_DIR=/tmp/genoray-target
pixi run bash -lc 'cargo test --no-default-features --features conversion svar1'
pixi run maturin develop
pixi run pytest tests/test_svar2_from_svar1.py -q
```
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add python/genoray/_svar2.py src/lib.rs src/orchestrator.rs src/svar1_reader.rs tests/test_svar2_from_svar1.py tests/test_e2e.rs
git commit -m "feat(svar2): add regions and samples to from_svar1"
```

---

### Task 6: CLI parity for pgen/svar1

`genoray write` grew `--regions/-r`, `--regions-file/-R`, `--samples/-s`, `--samples-file/-S` in #114 for the VCF path. Route them to `from_pgen`/`from_svar1`, and reject `--samples*` when the input resolves to the vcf-list form.

**Files:**
- Modify: `python/genoray/_cli/__main__.py`
- Test: `tests/cli/test_write_cli.py`

**Interfaces:**
- Consumes: Tasks 4/5 `from_pgen`/`from_svar1` kwargs.

- [ ] **Step 1: Write the failing CLI tests**

```python
def test_write_cli_pgen_regions_samples(tmp_path, tiny_pgen):
    out = tmp_path / "cli_pg"
    run_cli(["write", str(tiny_pgen), str(out), "--no-reference",
             "--regions", "chr1:1-4", "--samples", "S1", "--threads", "1"])
    assert SparseVar2(out).available_samples == ["S1"]

def test_write_cli_vcf_list_rejects_samples(tmp_path, vcf_dir):
    with pytest.raises(SystemExit):  # or the CLI's error type
        run_cli(["write", str(vcf_dir), str(tmp_path/"o"), "--no-reference",
                 "--samples", "S0"])
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/cli/test_write_cli.py -k 'pgen_regions or rejects_samples' -q`
Expected: FAIL (regions/samples ignored for pgen; no rejection for vcf-list).

- [ ] **Step 3: Implement dispatch**

In the `write` command, after resolving the source kind, pass the already-parsed `regions`/`samples` (via #114's `parse_regions_arg` and comma/file parsing) into the `from_pgen`/`from_svar1` calls. For the vcf-list input form, raise a clear error if `--samples`/`--samples-file` was given: `"--samples is not supported for multi-file (vcf-list) input; each input file contributes its own sample."`

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/cli/test_write_cli.py -q`
Expected: pass (pgen test skips if fixture unavailable, matching existing guards).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_cli/__main__.py tests/cli/test_write_cli.py
git commit -m "feat(cli): route regions/samples to pgen and svar1 writes"
```

---

### Task 7: Docs + SKILL.md

**Files:**
- Modify: `docs/source/svar.md`
- Modify: `skills/genoray-api/SKILL.md`

- [ ] **Step 1: Update `docs/source/svar.md`**

Generalize the "Region/sample-restricted SVAR2 conversion" section to all four methods. Remove the "`regions_overlap=variant` … reserved for the follow-up sub-contig sharding" caveat (now supported). Document the three overlap modes with the per-record `variant` note, and that `from_vcf_list` takes no `samples`.

- [ ] **Step 2: Update `skills/genoray-api/SKILL.md`**

Update the signatures of `from_vcf_list`, `from_pgen`, `from_svar1` to include `regions`/`merge_overlapping`/`regions_overlap` (and `samples` for pgen/svar1), and note the `regions_overlap` semantics + per-record `variant` behavior + caller-order sample guarantee. Confirm `from_vcf` already reflects #114.

- [ ] **Step 3: Verify + commit**

Run:
```bash
pixi run pytest tests -m "not network" -q
export CARGO_TARGET_DIR=/tmp/genoray-target
pixi run bash -lc 'cargo test --no-default-features --features conversion'
pixi run bash -lc 'cargo check --no-default-features'
pixi run prek run --all-files
```
Expected: all green.

```bash
git add docs/source/svar.md skills/genoray-api/SKILL.md
git commit -m "docs: generalize SVAR2 region/sample conversion docs and skill"
```

---

## Self-Review Notes

- **Spec coverage:** front-end (T2), VCF mode/variant (T1+T2), vcf_list regions (T3), pgen regions+samples (T4), svar1 regions+samples (T5), CLI (T6), docs+skill (T7), byte-compat via default `"pos"`/`None` (constraint + each task's regression run). No spec requirement left untasked.
- **Type consistency:** `OverlapMode` (Rust) / `regions_overlap: str` (Python boundary) parsed once per pipeline via `parse_overlap_mode`; `_normalize_svar2_regions` returns `list[tuple[str,int,int]]` consumed identically by T2–T5; `sample_perm`/`sample_idx` are `Vec<usize>` in caller order everywhere.
- **Prereq:** all tasks assume the #114 rebase (Global Constraints).

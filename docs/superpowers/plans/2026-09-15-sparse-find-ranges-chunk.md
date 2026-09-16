# Sparse `find_ranges_chunk` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make genoray's chunked SVAR2 range search emit only the non-empty
`(region, sample, ploid)` windows, in region-major CSR form, so the 99.55%-empty
dense `(n_haps, R, 2)` intermediate never exists.

**Architecture:** A second kernel beside `find_ranges_haps` that pushes a
`SparseCell` per non-empty window into per-hap `Vec`s, concatenates them in hap
order, and stable-counting-sorts them by region. A new PyO3 binding returns
seven numpy arrays; the Python layer wraps them in a frozen, slotted
`SparseRangesChunk`. The existing dense path is untouched and serves as the
parity oracle.

**Tech Stack:** Rust (rayon, PyO3 0.29, rust-numpy, proptest), Python 3.10+
(numpy, dataclasses), pixi tasks, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-sparse-find-ranges-chunk-design.md`

## Global Constraints

- Coordinates are 0-based, half-open `[start, end)`. Unchanged by this work.
- `cell_id` is `selected_sample * ploidy + ploid`, **absolute within the
  selection** — not relative to the chunk. A chunk starting at `sample_start`
  emits `cell_id >= sample_start * ploidy`.
- Output order is region-major, and `cell_id` strictly ascends inside each
  region block. Both are part of the contract, not incidental.
- A cell non-empty in only one channel still carries the other channel's **raw
  start** with length `0`. Do not zero it.
- `gather_haps_readbound_impl` (`src/query/gather.rs:772`) must not be touched.
  The whole design rests on it never reading an empty range's `start`.
- `MAX_END_SHIFT = 21`; max-end keys are `(pos << MAX_END_SHIFT) | ext` with
  `ext = 1 + deletion_len`, reduced with an integer max **before** unpacking.
- Python floor is 3.10 (`pyproject.toml:8`), so use `typing.TypeVar` /
  `typing.Generic`, never PEP 695 `class Foo[T]` syntax.
- Dense path (`find_ranges_haps`, `find_ranges_chunk`, `_find_ranges_chunked`)
  stays behaviourally identical. Any diff to it that is not `slots=True` is a
  bug.
- Commits follow Conventional Commits. Do not edit `CHANGELOG.md` or bump the
  version.
- No public API changes: `skills/genoray-api/SKILL.md` must NOT be modified.
- Build/test commands: `pixi run test-rust` (Rust), `pixi run test` (Python; it
  regenerates test data first), `pixi run pytest tests/test_svar2_ranges.py` for
  a single file. `prek` hooks run `cargo fmt`, `cargo clippy`, `ruff` and
  `pyrefly` on commit — let them.

---

### Task 1: The region reorder

The only piece with a non-obvious invariant, and the only one testable as a pure
function. Build it first, alone.

**Files:**
- Modify: `src/query/gather.rs` (add `SparseCell` and `sort_cells_by_region`
  after `MAX_END_SHIFT`, around `:291`; add tests to the existing
  `mod tests` at `:1083`)
- Modify: `src/query/mod.rs:36-41` (re-export)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub struct SparseCell { region: u32, cell_id: u32, snp_start: i64,
  indel_start: i64, snp_len: i32, indel_len: i32 }` (`Clone, Copy, Debug,
  Default, PartialEq, Eq`) and
  `pub fn sort_cells_by_region(cells: &[SparseCell], r: usize) -> (Vec<i64>, Vec<SparseCell>)`
  returning `(region_ptr of length r + 1, reordered cells)`.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `src/query/gather.rs`. The module already has
`use super::*;`, so `SparseCell` and `sort_cells_by_region` resolve once
Step 3 defines them. Add `use proptest::prelude::*;` at the top of the module.

```rust
    fn cell(region: u32, cell_id: u32) -> SparseCell {
        SparseCell {
            region,
            cell_id,
            snp_start: cell_id as i64 * 10,
            indel_start: region as i64 * 100,
            snp_len: 1,
            indel_len: 0,
        }
    }

    #[test]
    fn test_sort_cells_by_region_groups_and_keeps_hap_order() {
        // Hap-ascending input: hap 0 hits regions 0 and 2, hap 1 hits region 0,
        // hap 2 hits regions 1 and 2. Region-major output must interleave them.
        let input = vec![
            cell(0, 0),
            cell(2, 0),
            cell(0, 1),
            cell(1, 2),
            cell(2, 2),
        ];
        let (ptr, out) = sort_cells_by_region(&input, 3);
        assert_eq!(ptr, vec![0, 2, 3, 5]);
        assert_eq!(
            out,
            vec![cell(0, 0), cell(0, 1), cell(1, 2), cell(2, 0), cell(2, 2)]
        );
    }

    #[test]
    fn test_sort_cells_by_region_empty_input() {
        let (ptr, out) = sort_cells_by_region(&[], 3);
        assert_eq!(ptr, vec![0, 0, 0, 0]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_sort_cells_by_region_zero_regions() {
        let (ptr, out) = sort_cells_by_region(&[], 0);
        assert_eq!(ptr, vec![0]);
        assert!(out.is_empty());
    }

    proptest! {
        #[test]
        fn prop_sort_cells_is_region_major_and_stable(
            r in 1usize..8,
            raw in prop::collection::vec((0u32..8, 0u32..16), 0..64),
        ) {
            // Shape the input the way the kernel emits it: hap-ascending
            // outer, region-ascending inner, no duplicate (hap, region).
            let mut pairs: Vec<(u32, u32)> = raw
                .into_iter()
                .map(|(region, cell_id)| (cell_id, region % r as u32))
                .collect();
            pairs.sort_unstable();
            pairs.dedup();
            let input: Vec<SparseCell> =
                pairs.iter().map(|&(c, reg)| cell(reg, c)).collect();

            let (ptr, out) = sort_cells_by_region(&input, r);

            prop_assert_eq!(ptr.len(), r + 1);
            prop_assert_eq!(ptr[0], 0);
            prop_assert_eq!(*ptr.last().unwrap(), input.len() as i64);
            prop_assert!(ptr.windows(2).all(|w| w[0] <= w[1]));

            for reg in 0..r {
                let s = ptr[reg] as usize;
                let e = ptr[reg + 1] as usize;
                for c in &out[s..e] {
                    prop_assert_eq!(c.region as usize, reg);
                }
                // Stability: hap-ascending input => cell_id ascends per region.
                prop_assert!(out[s..e].windows(2).all(|w| w[0].cell_id < w[1].cell_id));
            }

            // Output is a permutation of the input.
            let mut a = input.clone();
            let mut b = out.clone();
            a.sort_by_key(|c| (c.region, c.cell_id));
            b.sort_by_key(|c| (c.region, c.cell_id));
            prop_assert_eq!(a, b);
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run test-rust 2>&1 | tail -30`
Expected: compile error — `cannot find function 'sort_cells_by_region'` and
`cannot find struct 'SparseCell'`.

- [ ] **Step 3: Write the implementation**

Insert into `src/query/gather.rs` immediately after the `MAX_END_SHIFT`
constant (before the `find_ranges_haps` doc comment):

```rust
/// One non-empty `(region, hap)` var_key window.
///
/// Emitted only when at least one channel overlaps. A cell empty in one channel
/// still carries that channel's raw `start` with length `0`:
/// `gather_haps_readbound_impl` derives `j = vs + k` inside the var-key loop and
/// so never reads an empty range's start, and an unconditionally in-bounds start
/// is strictly safer than a synthesized one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseCell {
    /// Index into the caller's `regions`.
    pub region: u32,
    /// `selected_sample * ploidy + ploid`, absolute within the selection.
    pub cell_id: u32,
    /// Absolute start into `vk_snp`'s packed positions/keys.
    pub snp_start: i64,
    /// Absolute start into `vk_indel`'s packed positions/keys.
    pub indel_start: i64,
    /// Overlap width in the SNP channel; `0` when empty.
    pub snp_len: i32,
    /// Overlap width in the indel channel; `0` when empty.
    pub indel_len: i32,
}

/// Stable counting sort of `cells` into region-major order, with CSR offsets.
///
/// Returns `(region_ptr, reordered)` where `region_ptr` has length `r + 1`,
/// starts at `0` and ends at `cells.len()`.
///
/// Stability is a correctness requirement, not an optimization. The caller
/// concatenates per-hap blocks in ascending hap order, so preserving input order
/// within a bucket is exactly what makes `cell_id` ascend inside each region —
/// the ordering the consumer's CSR lookup binary-searches on.
///
/// `O(n + r)`, against `O(n log n)` for a comparison sort: at cohort scale `n`
/// is ~18e6 entries per contig and this runs once per chunk.
pub fn sort_cells_by_region(cells: &[SparseCell], r: usize) -> (Vec<i64>, Vec<SparseCell>) {
    let mut ptr = vec![0i64; r + 1];
    for c in cells {
        ptr[c.region as usize + 1] += 1;
    }
    for i in 0..r {
        ptr[i + 1] += ptr[i];
    }
    let mut cursor: Vec<i64> = ptr[..r].to_vec();
    let mut out = vec![SparseCell::default(); cells.len()];
    for c in cells {
        let slot = &mut cursor[c.region as usize];
        out[*slot as usize] = *c;
        *slot += 1;
    }
    (ptr, out)
}
```

- [ ] **Step 4: Re-export from the query module**

In `src/query/mod.rs`, add `SparseCell` and `sort_cells_by_region` to the
existing `pub use gather::{...}` list, keeping it alphabetical within its
case group:

```rust
pub use gather::{
    BatchResult, BatchResultSplit, HapRanges, MAX_END_SHIFT, PAR_COLUMN_THRESHOLD, RangesBundle,
    SparseCell, dense_abs_row, find_ranges, find_ranges_haps, gather_haps_readbound,
    gather_haps_readbound_src, gather_ranges, overlap_batch, read_ranges, sort_cells_by_region,
};
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run test-rust 2>&1 | tail -30`
Expected: PASS, including `prop_sort_cells_is_region_major_and_stable`.

- [ ] **Step 6: Commit**

```bash
git add src/query/gather.rs src/query/mod.rs
git commit -m "feat(svar2): add SparseCell and a stable region counting sort"
```

---

### Task 2: The sparse kernel and its binding

Ends with the sparse emitter reachable from Python and proved byte-equal to the
dense one. The test fixture lands here because this is the first task that can
observe cell ordering.

**Files:**
- Modify: `src/query/gather.rs` (add `find_ranges_haps_sparse` after
  `find_ranges_haps`, which ends at `:402`)
- Modify: `src/query/mod.rs` (re-export)
- Modify: `src/py_query_ranges.rs` (add `find_ranges_chunk_sparse` after
  `find_ranges_chunk`, which ends at `:470`)
- Modify: `tests/conftest.py` (add `build_svar2_singleton_store` after
  `build_two_contig_svar2`, which starts at `:97`)
- Test: `tests/test_svar2_ranges.py`

**Interfaces:**
- Consumes: `SparseCell`, `sort_cells_by_region` from Task 1.
- Produces:
  - `pub fn find_ranges_haps_sparse(reader: &ContigReader, regions: &[(u32, u32)],
    sample_cols: &[usize], hap_lo: usize, hap_hi: usize) -> (Vec<i64>, Vec<SparseCell>, Vec<u64>)`
  - `PyContigReader.find_ranges_chunk_sparse(regions, samples, hap_lo, hap_hi)`
    returning the 7-tuple
    `(region_ptr i64[R+1], cell_id i32[N], snp_start i64[N], snp_len i32[N],
    indel_start i64[N], indel_len i32[N], max_end_keys i64[R])`
  - `build_svar2_singleton_store(tmp_path, n_samples=12) -> Path` in
    `tests/conftest.py`
  - `_dense_to_sparse(snp, indel, sample_start, ploidy)` test helper in
    `tests/test_svar2_ranges.py`

- [ ] **Step 1: Add the var_key-routed fixture**

The session `svar2_store` fixture has 2 samples, and `choose_representation`
(`src/cost_model.rs:60`) routes a variant dense when
`dense_bits < var_key_bits`. Its INS@6 and DEL@11 each have 3 carrier calls, so
both go dense, leaving exactly one non-empty var_key cell in 18 — too few to
observe ordering. Singletons (`x_calls == 1`) always route var_key.

Append to `tests/conftest.py`:

```python
def build_svar2_singleton_store(tmp_path, n_samples: int = 12) -> Path:
    """An svar2 store whose variants all route to the var_key channel.

    One singleton SNP per sample, all inside ``[0, 20)``, so a single region
    query returns one non-empty var_key cell per sample with strictly ascending
    ``cell_id``. That is what makes the sparse emitter's ordering observable:
    the session ``svar2_store`` fixture has one non-empty var_key cell in 18,
    because at 2 samples the cost model routes its INS and DEL dense.

    Sample ``i`` carries SNP ``i`` on hap ``i % 2``, so cell ids are
    ``2*i + (i % 2)`` -- distinct, ascending in ``i``, and not simply ``0..H``.
    """
    import subprocess
    from pathlib import Path

    d = Path(tmp_path)
    assert n_samples <= 20, "keep every singleton inside [0, 20)"

    ref = d / "ref.fa"
    ref.write_text(">chr1\n" + _REF + "\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    samples = [f"S{i}" for i in range(n_samples)]
    rows = []
    for i in range(n_samples):
        pos = i + 1  # 1-based VCF POS, so REF is _REF[i]
        ref_base = _REF[i]
        alt = "A" if ref_base != "A" else "C"
        gt = ["0|0"] * n_samples
        gt[i] = "1|0" if i % 2 == 0 else "0|1"
        rows.append(
            f"chr1\t{pos}\t.\t{ref_base}\t{alt}\t.\t.\t.\tGT\t" + "\t".join(gt)
        )
    vcf = d / "singletons.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples)
        + "\n"
        + "\n".join(rows)
        + "\n"
    )
    bcf = d / "singletons.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store"
    _core.run_conversion_pipeline(
        vcf_path=str(bcf),
        reference_path=str(ref),
        output_dir=str(out),
        regions=RegionSpec(chroms=["chr1"], samples=samples),
        fields=FieldSpec(),
        plan=PlanSettings(
            chunk_size=25_000,
            max_threads=1,
            long_allele_capacity=8 * 1024 * 1024,
        ),
        ploidy=2,
    )
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


@pytest.fixture(scope="session")
def svar2_singleton_store(tmp_path_factory) -> Path:
    """Session-scoped :func:`build_svar2_singleton_store`."""
    return build_svar2_singleton_store(tmp_path_factory.mktemp("svar2-singletons"))
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_svar2_ranges.py`:

```python
def _dense_to_sparse(
    snp: np.ndarray, indel: np.ndarray, sample_start: int, ploidy: int
):
    """Reference sparsifier: GVL's ``nonempty_entries``, as region-major CSR.

    ``snp``/``indel`` are hap-major ``(n_samples, ploidy, R, 2)``. Transposing to
    ``(R, n_samples, ploidy, 2)`` and running ``np.nonzero`` walks the LOGICAL
    shape in C order, so the result is region-major with ``cell_id`` ascending
    inside each region -- exactly the contract the Rust kernel must reproduce.
    """
    R = snp.shape[2]
    s = snp.transpose(2, 0, 1, 3)
    i = indel.transpose(2, 0, 1, 3)
    ne = (s[..., 1] > s[..., 0]) | (i[..., 1] > i[..., 0])
    ri, sj, pj = np.nonzero(ne)
    ptr = np.concatenate(
        [[0], np.cumsum(np.bincount(ri, minlength=R))]
    ).astype(np.int64)
    cell = ((sample_start + sj) * ploidy + pj).astype(np.int32)
    return (
        ptr,
        cell,
        s[ri, sj, pj, 0].astype(np.int64),
        (s[ri, sj, pj, 1] - s[ri, sj, pj, 0]).astype(np.int32),
        i[ri, sj, pj, 0].astype(np.int64),
        (i[ri, sj, pj, 1] - i[ri, sj, pj, 0]).astype(np.int32),
    )


def _assert_sparse_matches_dense(got, snp, indel, sample_start, ploidy):
    want = _dense_to_sparse(snp, indel, sample_start, ploidy)
    names = ["region_ptr", "cell_id", "snp_start", "snp_len", "indel_start", "indel_len"]
    for name, g, w in zip(names, got, want):
        np.testing.assert_array_equal(np.asarray(g), w, err_msg=name)
        assert np.asarray(g).dtype == w.dtype, name


def test_sparse_chunk_matches_dense_chunk(svar2_singleton_store: Path):
    """Binding-level parity, per hap slice, against the np.nonzero reference."""
    sv = SparseVar2(svar2_singleton_store)
    starts, ends = [0, 0], [20, 5]
    reg = list(zip(starts, ends))
    reader = sv._reader("chr1")
    P, S = sv.ploidy, sv.n_samples

    for hap_lo, hap_hi in [(0, S * P), (0, P), (P, 3 * P), (S * P, S * P)]:
        d = reader.find_ranges_chunk(reg, None, hap_lo, hap_hi)
        shape = ((hap_hi - hap_lo) // P, P, len(reg), 2)
        snp = np.asarray(d["vk_snp_range"]).reshape(shape)
        indel = np.asarray(d["vk_indel_range"]).reshape(shape)

        got = reader.find_ranges_chunk_sparse(reg, None, hap_lo, hap_hi)
        _assert_sparse_matches_dense(got[:6], snp, indel, hap_lo // P, P)
        np.testing.assert_array_equal(
            np.asarray(got[6], np.int64), np.asarray(d["max_end_keys"], np.int64)
        )


def test_sparse_chunk_cell_ids_ascend_within_each_region(
    svar2_singleton_store: Path,
):
    """The counting sort's stability, observed end to end.

    Every sample carries one singleton, so region [0, 20) has one non-empty cell
    per sample and region [0, 5) has a strict subset of them. A non-stable sort
    would scramble cell_id inside a region block; a hap-major emitter would put
    region 1's entries before region 0's.
    """
    sv = SparseVar2(svar2_singleton_store)
    reg = [(0, 20), (0, 5)]
    reader = sv._reader("chr1")
    S, P = sv.n_samples, sv.ploidy
    ptr, cell, *_ = reader.find_ranges_chunk_sparse(reg, None, 0, S * P)
    ptr = np.asarray(ptr)
    cell = np.asarray(cell)

    assert ptr.shape == (len(reg) + 1,)
    assert ptr[0] == 0 and ptr[-1] == len(cell)
    assert np.all(np.diff(ptr) >= 0)
    # Region 0 spans every singleton; region 1 only the first few.
    assert ptr[1] - ptr[0] == S
    assert 0 < ptr[2] - ptr[1] < S
    for r in range(len(reg)):
        block = cell[ptr[r] : ptr[r + 1]]
        assert np.all(np.diff(block) > 0), f"region {r}: {block}"
    # Sample i carries SNP i on hap i % 2.
    np.testing.assert_array_equal(
        cell[ptr[0] : ptr[1]],
        np.array([2 * i + (i % 2) for i in range(S)], np.int32),
    )


def test_sparse_chunk_empty_region_yields_no_entries(svar2_singleton_store: Path):
    """A region with no variants contributes an empty CSR block, not a row."""
    sv = SparseVar2(svar2_singleton_store)
    reader = sv._reader("chr1")
    S, P = sv.n_samples, sv.ploidy
    ptr, cell, snp_start, snp_len, indel_start, indel_len, keys = (
        reader.find_ranges_chunk_sparse([(30, 40)], None, 0, S * P)
    )
    np.testing.assert_array_equal(np.asarray(ptr), np.zeros(2, np.int64))
    for arr in (cell, snp_start, snp_len, indel_start, indel_len):
        assert len(np.asarray(arr)) == 0
    np.testing.assert_array_equal(np.asarray(keys), np.zeros(1, np.int64))


def test_sparse_chunk_sample_subset(svar2_singleton_store: Path):
    """cell_id indexes the SELECTION, not the store's sample axis."""
    sv = SparseVar2(svar2_singleton_store)
    sub = [sv.available_samples[3], sv.available_samples[1]]
    reader = sv._reader("chr1")
    idxs = sv._sample_idxs(sub)
    P = sv.ploidy
    reg = [(0, 20)]
    d = reader.find_ranges_chunk(reg, idxs, 0, len(sub) * P)
    snp = np.asarray(d["vk_snp_range"]).reshape(len(sub), P, 1, 2)
    indel = np.asarray(d["vk_indel_range"]).reshape(len(sub), P, 1, 2)
    got = reader.find_ranges_chunk_sparse(reg, idxs, 0, len(sub) * P)
    _assert_sparse_matches_dense(got[:6], snp, indel, 0, P)
    assert np.asarray(got[1]).max() < len(sub) * P


def test_sparse_chunk_rejects_out_of_bounds_hap_slice(svar2_singleton_store: Path):
    sv = SparseVar2(svar2_singleton_store)
    reader = sv._reader("chr1")
    H = sv.n_samples * sv.ploidy
    with pytest.raises(ValueError, match="out of bounds"):
        reader.find_ranges_chunk_sparse([(0, 20)], None, 0, H + 1)
    with pytest.raises(ValueError, match="out of bounds"):
        reader.find_ranges_chunk_sparse([(0, 20)], None, 2, 1)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_svar2_ranges.py -k sparse -x 2>&1 | tail -20`
Expected: FAIL with `AttributeError: 'PyContigReader' object has no attribute
'find_ranges_chunk_sparse'`.

- [ ] **Step 4: Write the Rust kernel**

Insert into `src/query/gather.rs` immediately after `find_ranges_haps` ends
(`:402`). Add `use std::sync::atomic::{AtomicU64, Ordering};` to the file's
imports.

```rust
/// Sparse twin of `find_ranges_haps`: the same column-outer sweep, emitting only
/// the `(region, hap)` windows where at least one channel overlaps.
///
/// Returns `(region_ptr, cells, max_end_keys)` with `cells` region-major and
/// `cell_id` ascending inside each region — `region_ptr` has length
/// `regions.len() + 1`.
///
/// Peak memory is ~2x the emitted payload (the per-hap blocks, then the
/// concatenation plus its reorder), against the dense path's
/// `n_haps * R * 32` bytes. At the All of Us chr22 grid that is ~1 GB per
/// contig against ~128 GB, because 0.45% of cells hold a variant.
///
/// The max-end accumulator is a `Vec<AtomicU64>` of length `R` rather than a
/// per-hap `Vec<u64>` reduced at the end: rayon's ordered `collect` is what
/// makes the concatenation hap-ascending (and hence the counting sort stable),
/// and a `fold`/`reduce` that carried the accumulator would give that up.
/// `R` atomics cost 8 bytes each; per-hap accumulators would cost `R * n_haps`.
pub fn find_ranges_haps_sparse(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    sample_cols: &[usize],
    hap_lo: usize,
    hap_hi: usize,
) -> (Vec<i64>, Vec<SparseCell>, Vec<u64>) {
    let ploidy = reader.ploidy;
    let r = regions.len();
    let n_haps = hap_hi - hap_lo;
    if n_haps == 0 || r == 0 {
        return (vec![0i64; r + 1], Vec::new(), vec![0u64; r]);
    }

    let acc: Vec<AtomicU64> = (0..r).map(|_| AtomicU64::new(0)).collect();

    let fill = |h_off: usize, out: &mut Vec<SparseCell>| {
        let h = hap_lo + h_off;
        let s = sample_cols[h / ploidy];
        let p = h % ploidy;
        let snp_ix = reader.vk_snp_index(s * ploidy + p);
        let indel_ix = reader.vk_indel_index(s, p);
        let snp_pos = reader.vk_snp.positions();
        let indel_pos = reader.vk_indel.positions();
        let indel_keys = as_u32(&reader.vk_indel.keys);
        for (ri, &(qs, qe)) in regions.iter().enumerate() {
            let a = snp_ix.overlap(qs, qe);
            let b = indel_ix.overlap(qs, qe);
            if a.end == a.start && b.end == b.start {
                continue;
            }

            // Same packing as `find_ranges_haps`: positions are sorted within a
            // column and the overlap range is contiguous, so the last element is
            // the highest-position overlapping variant.
            let mut k = 0u64;
            if a.end > a.start {
                let pos = snp_pos[a.end - 1] as u64;
                k = k.max((pos << MAX_END_SHIFT) | 1); // SNP/INS: ext = 1
            }
            if b.end > b.start {
                let i = b.end - 1;
                let pos = indel_pos[i] as u64;
                let ext = 1 + rvk::deletion_len(indel_keys[i]) as u64;
                k = k.max((pos << MAX_END_SHIFT) | ext);
            }
            acc[ri].fetch_max(k, Ordering::Relaxed);

            out.push(SparseCell {
                region: ri as u32,
                cell_id: h as u32,
                snp_start: a.start as i64,
                indel_start: b.start as i64,
                snp_len: (a.end - a.start) as i32,
                indel_len: (b.end - b.start) as i32,
            });
        }
    };

    let one_hap = |h_off: usize| -> Vec<SparseCell> {
        let mut v = Vec::new();
        fill(h_off, &mut v);
        v
    };

    // Same threshold as `find_ranges_haps`, for the same reason: below it the
    // serial path keeps `search::search_tree_build_count` (a thread-local)
    // observable in tests.
    let blocks: Vec<Vec<SparseCell>> = if n_haps < PAR_COLUMN_THRESHOLD {
        (0..n_haps).map(one_hap).collect()
    } else {
        (0..n_haps).into_par_iter().map(one_hap).collect()
    };

    let total: usize = blocks.iter().map(Vec::len).sum();
    let mut flat = Vec::with_capacity(total);
    // Move each block in and drop it as we go, so peak is 2x the payload, not 3x.
    for b in blocks {
        flat.extend(b);
    }
    let (ptr, cells) = sort_cells_by_region(&flat, r);
    drop(flat);
    let max_keys: Vec<u64> = acc.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    (ptr, cells, max_keys)
}
```

Add `find_ranges_haps_sparse` to the `pub use gather::{...}` list in
`src/query/mod.rs`, after `find_ranges_haps`.

- [ ] **Step 5: Write the binding**

In `src/py_query_ranges.rs`, add `find_ranges_haps_sparse` to the existing
`use crate::query::{...}` import (not `SparseCell` — the binding only reads its
fields and never names the type, and an unused import trips clippy), and add
the type alias just below the imports:

```rust
/// `find_ranges_chunk_sparse`'s return: `(region_ptr, cell_id, snp_start,
/// snp_len, indel_start, indel_len, max_end_keys)`. A tuple, not a dict: this is
/// a single-consumer payload with a fixed field set, and the Python layer
/// unpacks it into a frozen, slotted `SparseRangesChunk` at one call site.
type SparseChunkArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
);
```

Then add this method to the same `#[pymethods]` block, after
`find_ranges_chunk`:

```rust
    /// Sparse twin of `find_ranges_chunk`: only the non-empty
    /// `(region, sample, ploid)` windows of the hap slice `[hap_lo, hap_hi)`,
    /// region-major, as CSR.
    ///
    /// `cell_id` is `selected_sample * ploidy + ploid` and is absolute within
    /// the SELECTION, so a chunk starting at sample `s0` emits
    /// `cell_id >= s0 * ploidy`. It ascends strictly inside each region block.
    ///
    /// A cell non-empty in only one channel still carries the other channel's
    /// raw start with length `0` — see `SparseCell`.
    pub fn find_ranges_chunk_sparse<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
        hap_lo: usize,
        hap_hi: usize,
    ) -> PyResult<SparseChunkArrays<'py>> {
        let sample_cols: Vec<usize> = match &samples {
            Some(s) => s.clone(),
            None => (0..self.inner.n_samples).collect(),
        };
        let h_total = sample_cols.len() * self.inner.ploidy;
        if hap_lo > hap_hi || hap_hi > h_total {
            return Err(PyValueError::new_err(format!(
                "hap slice [{hap_lo}, {hap_hi}) out of bounds for {h_total} haps"
            )));
        }

        let (ptr, cells, max_keys) = py.detach(|| {
            find_ranges_haps_sparse(&self.inner, &regions, &sample_cols, hap_lo, hap_hi)
        });

        let n = cells.len();
        let cell_id = PyArray1::<i32>::zeros(py, [n], false);
        let snp_start = PyArray1::<i64>::zeros(py, [n], false);
        let snp_len = PyArray1::<i32>::zeros(py, [n], false);
        let indel_start = PyArray1::<i64>::zeros(py, [n], false);
        let indel_len = PyArray1::<i32>::zeros(py, [n], false);
        {
            // Fill in place rather than building five Vecs and copying: the
            // payload then exists once, matching `find_ranges_chunk`'s reason
            // for writing into freshly allocated arrays.
            let mut w_cell = cell_id.readwrite();
            let mut w_ss = snp_start.readwrite();
            let mut w_sl = snp_len.readwrite();
            let mut w_is = indel_start.readwrite();
            let mut w_il = indel_len.readwrite();
            let c_s = w_cell.as_slice_mut()?;
            let ss_s = w_ss.as_slice_mut()?;
            let sl_s = w_sl.as_slice_mut()?;
            let is_s = w_is.as_slice_mut()?;
            let il_s = w_il.as_slice_mut()?;
            for (i, c) in cells.iter().enumerate() {
                c_s[i] = c.cell_id as i32;
                ss_s[i] = c.snp_start;
                sl_s[i] = c.snp_len;
                is_s[i] = c.indel_start;
                il_s[i] = c.indel_len;
            }
        }
        let max_keys_i64: Vec<i64> = max_keys.iter().map(|&x| x as i64).collect();
        Ok((
            PyArray1::from_slice(py, &ptr),
            cell_id,
            snp_start,
            snp_len,
            indel_start,
            indel_len,
            PyArray1::from_slice(py, &max_keys_i64),
        ))
    }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pixi run test-rust 2>&1 | tail -15`
Expected: PASS.

Run: `pixi run pytest tests/test_svar2_ranges.py -x 2>&1 | tail -20`
Expected: PASS — the five new `sparse` tests plus every pre-existing test in the
file.

If `test_sparse_chunk_cell_ids_ascend_within_each_region` fails on
`ptr[1] - ptr[0] == S`, some singleton routed dense after all: print
`reader.find_ranges_chunk(reg, None, 0, S * P)` and check whether the var_key
ranges are empty. The fix is in the fixture (fewer carriers per variant), not in
the kernel.

- [ ] **Step 7: Commit**

```bash
git add src/query/gather.rs src/query/mod.rs src/py_query_ranges.rs tests/conftest.py tests/test_svar2_ranges.py
git commit -m "feat(svar2): emit sparse region-major ranges from the chunked search"
```

---

### Task 3: The Python stream

Ends with `_find_ranges_chunked_sparse` usable as a drop-in for
`_find_ranges_chunked`, over any chunking.

**Files:**
- Modify: `python/genoray/_svar2_batch.py` (`RangesChunk` at `:23`,
  `RangesStream` at `:46`, `_find_ranges_chunked` at `:243`)
- Test: `tests/test_svar2_ranges.py`

**Interfaces:**
- Consumes: `PyContigReader.find_ranges_chunk_sparse` from Task 2.
- Produces:
  - `SparseRangesChunk` — frozen, slotted, fields `sample_start: int`,
    `n_samples: int`, `region_ptr`, `cell_id`, `snp_start`, `snp_len`,
    `indel_start`, `indel_len`, `max_end_keys` (all `np.ndarray`).
  - `RangesStream` becomes `Generic[C]` with `chunks: Iterator[C]`.
  - `SparseVar2._find_ranges_chunked_sparse(contig, starts, ends, samples=None,
    *, max_mem=None) -> RangesStream[SparseRangesChunk]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar2_ranges.py`:

```python
def _reassemble_sparse(stream):
    """Concatenate a sparse stream into one region-major CSR table.

    Chunks partition the SAMPLE axis, so each chunk holds a full CSR over all
    regions; merging means interleaving their region blocks, which is exactly
    what the consumer's per-contig merge does.
    """
    R = stream.n_regions
    keys = stream.dense_max_end_keys.copy()
    blocks = [[] for _ in range(R)]
    for ch in stream.chunks:
        ptr = np.asarray(ch.region_ptr)
        for r in range(R):
            s, e = ptr[r], ptr[r + 1]
            blocks[r].append(
                (
                    np.asarray(ch.cell_id)[s:e],
                    np.asarray(ch.snp_start)[s:e],
                    np.asarray(ch.snp_len)[s:e],
                    np.asarray(ch.indel_start)[s:e],
                    np.asarray(ch.indel_len)[s:e],
                )
            )
        np.maximum(keys, np.asarray(ch.max_end_keys), out=keys)
    cols = [np.concatenate([b[i] for r in blocks for b in r]) for i in range(5)]
    counts = [sum(len(b[0]) for b in r) for r in blocks]
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return (ptr, *cols), keys


@pytest.mark.parametrize("max_mem", [None, 1 << 30, 1 << 8])
def test_sparse_stream_matches_dense_stream(svar2_singleton_store: Path, max_mem):
    """Every chunking, down to one sample per chunk, reassembles identically.

    At two regions and ploidy 2, ``bytes_per_sample`` is 128, so ``1 << 8``
    sizes chunks at exactly one sample -- the most adversarial split, and the
    smallest value that does not raise.
    """
    sv = SparseVar2(svar2_singleton_store)
    starts, ends = [0, 0], [20, 5]
    dense = sv._find_ranges_chunked("chr1", starts, ends, max_mem=max_mem)
    snp, indel, dense_keys = _reassemble(dense)

    sparse = sv._find_ranges_chunked_sparse("chr1", starts, ends, max_mem=max_mem)
    got, sparse_keys = _reassemble_sparse(sparse)
    _assert_sparse_matches_dense(got, snp, indel, 0, sv.ploidy)
    np.testing.assert_array_equal(sparse_keys, dense_keys)


def test_sparse_stream_cell_ids_are_absolute_across_chunks(
    svar2_singleton_store: Path,
):
    """A chunk's cell_id indexes the whole selection, not the chunk."""
    sv = SparseVar2(svar2_singleton_store)
    stream = sv._find_ranges_chunked_sparse("chr1", [0], [20], max_mem=1 << 8)
    P = stream.ploidy
    seen = []
    for ch in stream.chunks:
        cid = np.asarray(ch.cell_id)
        assert np.all(cid >= ch.sample_start * P)
        assert np.all(cid < (ch.sample_start + ch.n_samples) * P)
        seen.append(cid)
    assert stream.samples_per_chunk < stream.n_samples, "expected several chunks"
    np.testing.assert_array_equal(
        np.concatenate(seen),
        np.array([2 * i + (i % 2) for i in range(stream.n_samples)], np.int32),
    )


def test_sparse_stream_rejects_unusable_max_mem(svar2_singleton_store: Path):
    sv = SparseVar2(svar2_singleton_store)
    with pytest.raises(ValueError, match="max_mem"):
        sv._find_ranges_chunked_sparse("chr1", [0], [20], max_mem=1)


def test_sparse_stream_sample_subset(svar2_singleton_store: Path):
    sv = SparseVar2(svar2_singleton_store)
    sub = [sv.available_samples[2], sv.available_samples[5]]
    dense = sv._find_ranges_chunked("chr1", [0], [20], samples=sub)
    snp, indel, _ = _reassemble(dense)
    sparse = sv._find_ranges_chunked_sparse("chr1", [0], [20], samples=sub)
    assert sparse.n_samples == 2
    got, _ = _reassemble_sparse(sparse)
    _assert_sparse_matches_dense(got, snp, indel, 0, sv.ploidy)


def test_ranges_dataclasses_are_slotted():
    from dataclasses import fields

    from genoray._svar2_batch import RangesChunk, RangesStream, SparseRangesChunk

    for cls in (RangesChunk, RangesStream, SparseRangesChunk):
        # `cls.__dict__`, not `hasattr`: an inherited `__slots__` would pass
        # hasattr while the class itself still carried a per-instance dict.
        slots = cls.__dict__.get("__slots__")
        assert slots is not None, cls.__name__
        assert set(slots) == {f.name for f in fields(cls)}, cls.__name__
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_svar2_ranges.py -k "sparse_stream or slotted" -x 2>&1 | tail -20`
Expected: FAIL with `AttributeError: 'SparseVar2' object has no attribute
'_find_ranges_chunked_sparse'`.

- [ ] **Step 3: Add the dataclasses**

In `python/genoray/_svar2_batch.py`, extend the `typing` import to
`from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar`, add
`slots=True` to `RangesChunk`'s decorator, and insert after it:

```python
@dataclass(frozen=True, slots=True)
class SparseRangesChunk:
    """One hap slice of a chunked ``_find_ranges``, sparse and region-major.

    Only the ``(region, sample, ploid)`` windows where at least one channel
    overlaps. At cohort scale that is well under 1% of the grid, so this is the
    form to build a region-CSR cache from -- the dense :class:`RangesChunk` is
    ~128 GB per All of Us chr22 contig, over 99% of it the pair ``(0, 0)``.

    Attributes:
        sample_start: Offset of this chunk on the SELECTED sample axis.
        n_samples: Number of selected samples in this chunk.
        region_ptr: Shape ``(n_regions + 1,)`` int64. CSR offsets;
            ``region_ptr[0] == 0`` and ``region_ptr[-1] == N``.
        cell_id: Shape ``(N,)`` int32. ``selected_sample * ploidy + ploid``,
            absolute within the selection -- NOT relative to this chunk, so
            every value is in ``[sample_start * ploidy, (sample_start +
            n_samples) * ploidy)``. Strictly ascending inside each region block.
        snp_start: Shape ``(N,)`` int64. Absolute start into the SNP channel.
        snp_len: Shape ``(N,)`` int32. Overlap width; ``0`` when that channel is
            empty for this cell, in which case ``snp_start`` is still the raw
            insertion point rather than a synthesized ``0``.
        indel_start: Shape ``(N,)`` int64. As ``snp_start``, indel channel.
        indel_len: Shape ``(N,)`` int32. As ``snp_len``, indel channel.
        max_end_keys: Shape ``(n_regions,)``. Identical to
            :attr:`RangesChunk.max_end_keys` -- reduce across chunks with an
            elementwise maximum BEFORE unpacking.
    """

    sample_start: int
    n_samples: int
    region_ptr: "np.ndarray"
    cell_id: "np.ndarray"
    snp_start: "np.ndarray"
    snp_len: "np.ndarray"
    indel_start: "np.ndarray"
    indel_len: "np.ndarray"
    max_end_keys: "np.ndarray"


#: Chunk type of a :class:`RangesStream` -- :class:`RangesChunk` (dense) or
#: :class:`SparseRangesChunk`.
C = TypeVar("C")
```

Then make `RangesStream` generic and slotted — replace its decorator and class
line with:

```python
@dataclass(frozen=True, slots=True)
class RangesStream(Generic[C]):
```

and its `chunks` field with:

```python
    chunks: "Iterator[C]"
```

- [ ] **Step 4: Extract the shared chunk plan**

The `max_mem` arithmetic must not be duplicated. In the same module, add above
the `SparseVar2` methods (next to the other module-level dataclasses):

```python
@dataclass(frozen=True, slots=True)
class _ChunkPlan:
    """Everything both chunked range streams need before they diverge."""

    reader: Any
    reg: "list[tuple[int, int]]"
    sample_idxs: "list[int] | None"
    header: "Mapping[str, Any]"
    per: int
    n_regions: int
    n_samples: int
    ploidy: int
```

In the `SparseVar2` class, add:

```python
    def _ranges_chunk_plan(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None",
        max_mem: int | None,
    ) -> _ChunkPlan:
        """Header plus sample-chunk sizing, shared by the dense and sparse streams.

        The budget is computed from the DENSE per-sample payload in both cases.
        The sparse stream's realized allocation is proportional to the non-empty
        cells instead -- far smaller -- but sizing it from a measured fill would
        trade a hard worst-case bound for a heuristic, so the bound stays.
        """
        reg = self._regions(starts, ends)
        sample_idxs = self._sample_idxs(samples)
        reader = self._reader(contig)
        header = reader.find_ranges_header(reg, sample_idxs)

        n_regions = int(header["n_regions"])
        n_samples = int(header["n_samples"])
        ploidy = int(header["ploidy"])

        # `bytes_per_sample` is the real per-sample payload: both channels
        # (snp + indel) x 2 endpoints x int64 x ploidy x regions. The extra
        # `2 *` in the division below -- not this factor -- is the safety
        # margin, covering the transient the binding holds while handing the
        # freshly filled arrays back.
        bytes_per_sample = n_regions * ploidy * 2 * 8 * 2
        if max_mem is None:
            per = max(n_samples, 1)
        else:
            per = (
                int(max_mem) // (2 * bytes_per_sample)
                if bytes_per_sample
                else n_samples
            )
            if per < 1:
                raise ValueError(
                    f"max_mem ({int(max_mem)} bytes) is too small for even one "
                    f"sample of {n_regions} regions at ploidy {ploidy}: needs at "
                    f"least {2 * bytes_per_sample} bytes."
                )
            per = min(per, max(n_samples, 1))

        return _ChunkPlan(
            reader=reader,
            reg=reg,
            sample_idxs=sample_idxs,
            header=header,
            per=per,
            n_regions=n_regions,
            n_samples=n_samples,
            ploidy=ploidy,
        )

    def _ranges_stream(self, plan: _ChunkPlan, chunks) -> "RangesStream":
        """Wrap a chunk generator in the header both streams share."""
        return RangesStream(
            n_regions=plan.n_regions,
            n_samples=plan.n_samples,
            ploidy=plan.ploidy,
            samples_per_chunk=plan.per,
            region_starts=np.asarray(plan.header["region_starts"]),
            dense_range=np.asarray(plan.header["dense_range"]),
            dense_snp_range=np.asarray(plan.header["dense_snp_range"]),
            dense_indel_range=np.asarray(plan.header["dense_indel_range"]),
            sample_cols=np.asarray(plan.header["sample_cols"]),
            dense_max_end_keys=np.asarray(plan.header["dense_max_end_keys"], np.int64),
            chunks=chunks,
        )
```

Now rewrite `_find_ranges_chunked`'s body below its docstring to use them,
deleting the inlined header/sizing block and the inlined `RangesStream(...)`
construction. The docstring, signature and behaviour are unchanged; only the
return annotation gains its parameter:

```python
    ) -> "RangesStream[RangesChunk]":
        ...  # docstring unchanged
        plan = self._ranges_chunk_plan(contig, starts, ends, samples, max_mem)

        def _gen() -> "Iterator[RangesChunk]":
            for s0 in range(0, plan.n_samples, plan.per):
                s1 = min(s0 + plan.per, plan.n_samples)
                d = plan.reader.find_ranges_chunk(
                    plan.reg, plan.sample_idxs, s0 * plan.ploidy, s1 * plan.ploidy
                )
                cs = s1 - s0
                shape = (cs, plan.ploidy, plan.n_regions, 2)
                yield RangesChunk(
                    sample_start=s0,
                    n_samples=cs,
                    vk_snp_range=np.asarray(d["vk_snp_range"]).reshape(shape),
                    vk_indel_range=np.asarray(d["vk_indel_range"]).reshape(shape),
                    max_end_keys=np.asarray(d["max_end_keys"], np.int64),
                )

        return self._ranges_stream(plan, _gen())
```

- [ ] **Step 5: Add the sparse stream**

Append to `SparseVar2`, immediately after `_find_ranges_chunked`:

```python
    def _find_ranges_chunked_sparse(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None" = None,
        *,
        max_mem: int | None = None,
    ) -> "RangesStream[SparseRangesChunk]":
        """Sparse, chunked, memory-bounded ``_find_ranges``.

        Identical to :meth:`_find_ranges_chunked` in header, chunking and
        arguments; each chunk carries only the non-empty
        ``(region, sample, ploid)`` windows, region-major, as CSR. Build a
        region-CSR cache from this rather than from the dense stream: at cohort
        scale under 1% of the grid is non-empty, so the dense intermediate is
        ~128 GB per All of Us chr22 contig and a consumer's first act is to
        throw ~99.55% of it away.

        Chunks partition the SAMPLE axis, so each one is a complete CSR over
        every region for its samples; merging chunks means interleaving their
        per-region blocks.

        Args:
            contig: Contig name.
            starts: 0-based start positions of the query regions.
            ends: 0-based, exclusive end positions of the query regions.
            samples: Sample names selecting (and reordering) a subset.
            max_mem: Approximate byte budget for one chunk, computed from the
                DENSE payload. ``None`` yields a single chunk.

        Returns:
            A :class:`RangesStream` whose ``chunks`` generator yields
            :class:`SparseRangesChunk` in ascending ``sample_start`` order.

        Raises:
            ValueError: If ``max_mem`` cannot fit a single sample's payload, or
                if the contig's largest deletion overflows the max-end key
                packing width.
        """
        plan = self._ranges_chunk_plan(contig, starts, ends, samples, max_mem)

        def _gen() -> "Iterator[SparseRangesChunk]":
            for s0 in range(0, plan.n_samples, plan.per):
                s1 = min(s0 + plan.per, plan.n_samples)
                (
                    region_ptr,
                    cell_id,
                    snp_start,
                    snp_len,
                    indel_start,
                    indel_len,
                    max_end_keys,
                ) = plan.reader.find_ranges_chunk_sparse(
                    plan.reg, plan.sample_idxs, s0 * plan.ploidy, s1 * plan.ploidy
                )
                yield SparseRangesChunk(
                    sample_start=s0,
                    n_samples=s1 - s0,
                    region_ptr=np.asarray(region_ptr),
                    cell_id=np.asarray(cell_id),
                    snp_start=np.asarray(snp_start),
                    snp_len=np.asarray(snp_len),
                    indel_start=np.asarray(indel_start),
                    indel_len=np.asarray(indel_len),
                    max_end_keys=np.asarray(max_end_keys, np.int64),
                )

        return self._ranges_stream(plan, _gen())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_svar2_ranges.py 2>&1 | tail -20`
Expected: PASS, all tests in the file — the dense tests are the regression gate
on the Step 4 refactor.

- [ ] **Step 7: Run the full suite**

Run: `pixi run test 2>&1 | tail -20`
Expected: PASS. `_find_ranges_chunked` is used elsewhere in the suite; the Step 4
refactor must not have moved it.

- [ ] **Step 8: Commit**

```bash
git add python/genoray/_svar2_batch.py tests/test_svar2_ranges.py
git commit -m "feat(svar2): add _find_ranges_chunked_sparse on a generic RangesStream"
```

---

### Task 4: Measurement and handoff

The change claims a materialization win. Measure it, then leave the follow-ups
the spec's rollout section names.

**Files:**
- Create: `scripts/bench_sparse_ranges.py`
- Modify: none

**Interfaces:**
- Consumes: `SparseVar2._find_ranges_chunked` and
  `._find_ranges_chunked_sparse` from Task 3, `build_svar2_singleton_store`
  from Task 2.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the benchmark script**

A script, not a gated test: the spec asks for dense-vs-sparse wall time per
chunk at a realistic fill, and there is no CI machine where a cohort-scale store
is cheap. Create `scripts/bench_sparse_ranges.py`:

```python
"""Dense vs sparse chunk emission on a synthetic cohort store.

Not a CI gate -- a cohort-scale store is too expensive to build per run. Run it
by hand when changing either kernel:

    pixi run python scripts/bench_sparse_ranges.py --samples 200 --regions 400

Reports per-chunk wall time and the realized fill, which is the number that
decides whether the sparse path is worth anything: at the All of Us chr22 grid
it is 0.45%.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from conftest import build_svar2_singleton_store  # noqa: E402

from genoray import SparseVar2  # noqa: E402


def _time(fn, reps: int) -> float:
    """Minimum over `reps` runs: scheduling noise only ever adds time."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    # The fixture puts one singleton per sample inside a 40 bp contig, so 20 is
    # its ceiling. Raising it means a bigger reference, not a bigger flag.
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--regions", type=int, default=400)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    # A fresh directory per run: the conversion pipeline writes a new store and
    # will not overwrite one left behind by the last invocation.
    tmp = Path(tempfile.mkdtemp(prefix="genoray-bench-sparse-"))
    store = build_svar2_singleton_store(tmp, n_samples=args.samples)
    sv = SparseVar2(store)

    rng = np.random.default_rng(0)
    starts = rng.integers(0, 20, size=args.regions).astype(np.int64)
    ends = np.minimum(starts + 8, 40)

    dense = _time(
        lambda: [c for c in sv._find_ranges_chunked("chr1", starts, ends).chunks],
        args.reps,
    )
    sparse = _time(
        lambda: [
            c for c in sv._find_ranges_chunked_sparse("chr1", starts, ends).chunks
        ],
        args.reps,
    )

    chunks = list(sv._find_ranges_chunked_sparse("chr1", starts, ends).chunks)
    n = sum(len(c.cell_id) for c in chunks)
    cells = args.regions * sv.n_samples * sv.ploidy
    print(f"regions={args.regions} samples={sv.n_samples} ploidy={sv.ploidy}")
    print(f"fill    {n}/{cells} = {100 * n / cells:.2f}%")
    print(f"dense   {1000 * dense:8.2f} ms")
    print(f"sparse  {1000 * sparse:8.2f} ms  ({dense / sparse:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `pixi run python scripts/bench_sparse_ranges.py --samples 20 --regions 400`
Expected: it prints a fill percentage and two timings without raising. The
absolute numbers are not a gate — the fixture is 20 samples on a 40 bp contig,
so it reports shape, not cohort-scale truth. Record what it printed in the
commit message.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_sparse_ranges.py
git commit -m "test(svar2): bench dense vs sparse chunk emission"
```

- [ ] **Step 4: File the follow-ups**

Two issues the spec's rollout section defers. Run both:

```bash
gh issue create --repo d-laub/genoray \
  --title "remove the dense find_ranges_chunk path once GVL consumes the sparse one" \
  --body "\`find_ranges_chunk\` / \`_find_ranges_chunked\` are kept only as the parity oracle for the sparse emitter and because GVL's open PR #407 consumes the dense format. Once GVL's writer moves to \`_find_ranges_chunked_sparse\` and its genoray pin is bumped, delete the dense chunk kernel, binding and stream, and re-point the parity tests at \`find_ranges\` (non-chunked, still dense).

Blocked on: mcvickerlab/genvarloader#405 landing downstream.
Design: \`docs/superpowers/specs/2026-09-15-sparse-find-ranges-chunk-design.md\`"

gh issue create --repo d-laub/genoray \
  --title "fill-adaptive sample chunking for the sparse range stream" \
  --body "\`_ranges_chunk_plan\` sizes sample chunks from the DENSE per-sample payload, so the sparse stream gets the same chunk count as the dense one while allocating ~200x less. Sizing from a fill rate measured on the first chunk would cut the chunk count, at the cost of replacing a hard worst-case bound with a heuristic -- deliberately out of scope in the sparse-emission design.

Worth doing only if chunk count turns out to cost measurable wall time at cohort scale; measure before building.
Design: \`docs/superpowers/specs/2026-09-15-sparse-find-ranges-chunk-design.md\`"
```

- [ ] **Step 5: Report the downstream handoff**

This plan ends at genoray's edge. Tell the user, in the completion report, that
the remaining rollout steps are downstream and not covered here:

1. `mcvickerlab/genvarloader#407` lands as written, against the dense format.
2. genoray cuts a release (the binding is Python-facing).
3. A GVL follow-up flips its writer to `_find_ranges_chunked_sparse`, collapses
   `nonempty_entries` to a passthrough, and bumps both the genoray Python
   requirement and the `genoray_core` rev in `Cargo.toml` (`d66ec0e` today).

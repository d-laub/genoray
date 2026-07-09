# SP-2 — `query.rs` → `query/` module + gather DRY — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `src/query.rs` (1,402 L) into a `src/query/` module, eliminate the 4× copy-pasted gather inner loops, apply the `DenseClass`/`Range` type cleanups, and demote oracle-only entry points — behavior-preserving except one coordinated signature change.

**Architecture:** Split by responsibility along the audit's seams (`sidecar`/`reader`/`union`/`decode`/`gather`/`oracle`), re-exporting the public surface from `query/mod.rs` so all downstream-consumed paths stay stable. Then extract two shared helpers (`PresenceBitWriter`, `gather_vk`) the four gather functions currently duplicate. The existing Rust integration suite (oracle-parity + byte-identical asserts) is the safety net — this is a refactor, so the test cycle per task is "the existing suite stays green," with new focused tests only for genuinely new code (`HapRanges::new` validation).

**Tech Stack:** Rust, pyo3 (query-core, `--no-default-features --features conversion`), `ndarray`, `memmap2`. Downstream consumer: GenVarLoader `svar2-m6b-kernel` branch.

## Global Constraints

- **Behavior-preserving** except Task 7's `gather_haps_readbound` → `HapRanges<'_>` signature change, landed in lockstep with the gvl update (Task 8).
- **Rust verification command:** `pixi run test-rust` (= `cargo test --no-default-features --features conversion`). The `--no-default-features` is mandatory — with default features the pyo3 test binary fails to link (`undefined symbol: _Py_Dealloc`).
- **Python verification command:** `pixi run test`.
- **Stability guarantees (paths that MUST NOT change)** — downstream gvl `svar2-m6b-kernel` production imports: `genoray_core::query::ContigReader` (+ `::open`, `::lut_arrays`), `genoray_core::query::gather_haps_readbound` (signature changes only in Task 7+8), `genoray_core::query::BatchResultSplit` (+ all fields), `genoray_core::query::KeyRef`, `genoray_core::bits_get_bit`.
- **Safe to relocate** (this repo's tests only): `overlap_sample`, `gather_ranges_readbound`, `decode_keyref_pub`, `decode_keyref_alt_pub`.
- **Commit convention:** Conventional Commits. Pre-commit hooks run `cargo fmt`/`cargo check`/`cargo clippy`; must pass. End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **No `skills/genoray-api/SKILL.md` change** — all symbols are `genoray_core` Rust, not reachable via `import genoray`.
- Line numbers below are current-tree references for orientation, **not** contracts — relocate symbols by name.

---

### Task 1: Module split — pure code motion + re-export

Convert `src/query.rs` into a `src/query/` package. Zero logic change; the whole suite must be green before Task 2.

**Files:**
- Delete: `src/query.rs`
- Create: `src/query/mod.rs`, `src/query/sidecar.rs`, `src/query/reader.rs`, `src/query/union.rs`, `src/query/decode.rs`, `src/query/gather.rs`, `src/query/oracle.rs`
- Modify: none outside `src/query/` (re-exports keep `crate::query::*` paths intact)

**Interfaces:**
- Produces: the module tree. `query/mod.rs` re-exports every currently-public symbol at its original path (`pub use gather::{overlap_batch, find_ranges, gather_ranges, gather_haps_readbound, read_ranges, BatchResult, BatchResultSplit, RangesBundle}; pub use reader::ContigReader; pub use decode::{HapCalls, QueryResult}; pub use crate::spine::KeyRef;`). Cross-file-but-private symbols become `pub(crate)`.
- In this task ONLY, `overlap_sample`/`gather_ranges_readbound`/`decode_keyref_pub`/`decode_keyref_alt_pub` are re-exported at their **original** `query::` paths too (they move into `oracle.rs` physically but stay re-exported from `mod.rs` — the path rename happens in Task 6, keeping this task a pure move).

- [ ] **Step 1: Create the directory and move symbols by the spec's file→symbol table**

Symbol placement (from the spec's "Target module layout"):
- `sidecar.rs`: `mmap_file`, `as_u32`, `as_bytes`, `load_offsets`, `load_max_del`, `load_dense_max_del`, `open_dense`, `SubStreamView` (+ `positions`/`column`), `DenseView` (+ `positions`/`carried`).
- `reader.rs`: `ContigReader` (fields, `open`, `lut_arrays`), `vk_slice`, `dense_carried`, `vk_snp_overlap`, `vk_indel_overlap`, `dense_snp_overlap`, `dense_indel_overlap`.
- `union.rs`: `DenseUnion` (+ `overlap`), `ContigReader::dense_union`.
- `decode.rs`: `Call`, `decode_keyref`, `HapCalls`, `QueryResult`.
- `gather.rs`: `BatchResult`, `BatchResultSplit`, `RangesBundle`, `overlap_batch`, `find_ranges`, `gather_ranges`, `gather_haps_readbound`, `read_ranges`, `BatchResult::decode_hap`. Move the in-file `#[cfg(test)] mod tests` here (its 3 tests exercise `BatchResult`/mmap helpers).
- `oracle.rs`: `overlap_sample`, `gather_ranges_readbound`, `decode_keyref_pub`, `decode_keyref_alt_pub` (names unchanged this task).

Add `use` imports per file (`crate::spine::KeyRef`, `crate::bits`, `crate::search::*`, `crate::dense::*`, etc.); mark cross-file internals `pub(crate)` (e.g. `ContigReader` fields used by `gather.rs`, `sidecar` accessors, `decode_keyref`, `DenseUnion`).

- [ ] **Step 2: Wire `query/mod.rs`**

```rust
pub mod decode;
pub mod gather;
pub mod oracle;
pub mod reader;
pub mod sidecar;
pub mod union;

pub use crate::spine::KeyRef;
pub use decode::{HapCalls, QueryResult};
pub use gather::{
    BatchResult, BatchResultSplit, RangesBundle, find_ranges, gather_haps_readbound,
    gather_ranges, overlap_batch, read_ranges,
};
pub use reader::ContigReader;

// Task 6 will move these to `query::oracle::*`; kept re-exported here this task
// so the split is a pure move.
pub use oracle::{decode_keyref_alt_pub, decode_keyref_pub, gather_ranges_readbound, overlap_sample};
```

`src/lib.rs` already has `pub mod query;` (line 50) — no change needed.

- [ ] **Step 3: Build**

Run: `pixi run bash -lc 'cargo check --no-default-features --features conversion'`
Expected: compiles clean (resolve any missed `pub(crate)`/`use` until it does).

- [ ] **Step 4: Run the full Rust suite**

Run: `pixi run test-rust`
Expected: PASS — all of `test_query`, `test_batch`, `test_decode_mat`, `test_readbound_gather`, `test_ranges_split`, `test_batch_raw`, `test_e2e`, `test_query_only_build` green (they import `genoray_core::query::*` at unchanged paths).

- [ ] **Step 5: Run the Python suite**

Run: `pixi run test`
Expected: PASS (FFI path unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/query src/query.rs
git commit -m "refactor(query): split query.rs into query/ module (pure code motion)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Extract `PresenceBitWriter`

Replace the 6× hand-rolled dense presence-bit CSR bookkeeping (query.rs:596, 881, 1026, 1043, 1234, 1261) with one helper in `gather.rs`.

**Files:**
- Modify: `src/query/gather.rs` (add helper; rewrite the presence-bit loops in `overlap_batch`, `gather_ranges`)
- Modify: `src/query/oracle.rs` (rewrite the presence-bit loops in `gather_ranges_readbound`)

**Interfaces:**
- Produces: `pub(crate)` helper consumed by Task 3 and Tasks 6/7 gathers.

```rust
/// CSR presence-bitmask accumulator: owns the `(bits, offsets)` pair, one row of
/// `nbits` bits appended per hap. `offsets` starts `[0]`; after each `push_hap`,
/// `offsets.last()` is the total bit count.
pub(crate) struct PresenceBitWriter {
    bits: Vec<u8>,
    offsets: Vec<usize>,
}

impl PresenceBitWriter {
    pub(crate) fn new() -> Self {
        Self { bits: Vec::new(), offsets: vec![0] }
    }

    /// Append one hap row of `nbits` bits; `set` is called with each in-row
    /// index `k in 0..nbits` and must return whether bit `k` is present.
    pub(crate) fn push_hap(&mut self, nbits: usize, mut set: impl FnMut(usize) -> bool) {
        let base = *self.offsets.last().unwrap();
        let need_bytes = (base + nbits).div_ceil(8);
        if self.bits.len() < need_bytes {
            self.bits.resize(need_bytes, 0);
        }
        for k in 0..nbits {
            if set(k) {
                crate::bits::set_bit(&mut self.bits, base + k);
            }
        }
        self.offsets.push(base + nbits);
    }

    /// Consume into the `(present, present_off)` fields the result structs expect.
    pub(crate) fn into_parts(self) -> (Vec<u8>, Vec<usize>) {
        (self.bits, self.offsets)
    }
}
```

- [ ] **Step 1: Add the helper to `gather.rs`**

Paste the struct above.

- [ ] **Step 2: Rewrite `overlap_batch`'s presence loop**

Replace the `dense_present`/`dense_present_off` locals + inner loop (current 579-620) with:

```rust
let mut presence = PresenceBitWriter::new();
// ... inside the per-hap loop, after the vk_slice push:
let (ds, de) = ranges[r];
let nbits = de - ds;
presence.push_hap(nbits, |k| {
    let j = ds + k;
    let (is_indel, dcol) = dense.src[j];
    let carried = if is_indel {
        reader.dense_indel.as_ref().expect("indel src implies table").carried(hap, dcol)
    } else {
        reader.dense_snp.as_ref().expect("snp src implies table").carried(hap, dcol)
    };
    carried && dense.v_ends[j] > qs
});
// ... at the end:
let (dense_present, dense_present_off) = presence.into_parts();
```

- [ ] **Step 3: Rewrite `gather_ranges` and `oracle::gather_ranges_readbound` presence loops the same way**

Apply the identical `PresenceBitWriter` pattern to each remaining presence-bit site (`gather_ranges` ~879-905; `gather_ranges_readbound`'s split snp/indel presence — two writers, one per class; `gather_haps_readbound`'s two split writers). Preserve the split-dense form: `gather_ranges_readbound`/`gather_haps_readbound` use **two** `PresenceBitWriter`s (`snp_presence`, `indel_presence`) feeding `dense_snp_present*`/`dense_indel_present*`.

- [ ] **Step 4: Run the Rust suite (byte-identical is the gate)**

Run: `pixi run test-rust`
Expected: PASS — critically `test_gather_haps_readbound_byte_identical`, `test_readbound_gather.rs`, and the `overlap_batch`/`overlap_sample` parity tests confirm the bit layout is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/query/gather.rs src/query/oracle.rs
git commit -m "refactor(query): extract PresenceBitWriter for CSR presence bitmasks

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Extract `gather_vk` (var_key snp+indel gather+merge)

Collapse the 3–4× duplicated "var_key snp gather + indel gather + merge" block (query.rs:582-623, 839-908, 986-1060, 1156-1284). `gather_haps_readbound`'s hand-inlined merge is unified **only if** a micro-bench shows no regression.

**Files:**
- Modify: `src/query/gather.rs` (add `gather_vk`; route `gather_ranges`, `gather_haps_readbound`, `overlap_batch` through it)
- Modify: `src/query/oracle.rs` (`gather_ranges_readbound`)

**Interfaces:**
- Consumes: `ContigReader::vk_slice` (already the single-slice primitive), `crate::spine::merge_keys`.
- Produces:

```rust
/// The var_key channel for one flat hap-column over one region window: SNP and
/// indel packed slices decoded to uniform `KeyRef`s and merged position-sorted.
/// This is the shared body the batch/ranges/read-bound gathers previously
/// hand-inlined. `vk_slice` already performs the union+merge for a column; this
/// wrapper is the seam the read-bound path replays from precomputed ranges.
pub(crate) fn gather_vk(
    reader: &ContigReader,
    vk_snp_range: (usize, usize),
    vk_indel_range: (usize, usize),
    q_start: u32,
) -> Vec<KeyRef>;
```

(Note: `vk_slice` is the region→slice path; `gather_vk` is the range→slice replay path used by the read-bound gathers, which already hold precomputed `[start,end)` windows. Match the existing decode/merge exactly — extract, do not redesign. The `(usize, usize)` params migrate to `Range<usize>` in Task 5 alongside the rest of the internal ranges — do NOT introduce `Range` here, since the range types are still tuples until then.)

- [ ] **Step 1: Micro-bench baseline (measure, don't guess)**

Run a representative read-bound gather before any merge change and record wall time:

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion --release readbound -- --nocapture'`
Record the timing of `test_gather_haps_readbound_byte_identical` / the readbound e2e as the baseline reference. (If no timing prints, add a `std::time::Instant` around the `gather_haps_readbound` call in a scratch `#[test]` — remove before commit.)

- [ ] **Step 2: Add `gather_vk` and route `gather_ranges` + `oracle::gather_ranges_readbound` through it**

Extract the snp-gather + indel-gather + `merge_keys` sequence into `gather_vk`, replacing the duplicated blocks in `gather_ranges` and `gather_ranges_readbound`.

- [ ] **Step 3: Decide `gather_haps_readbound`'s merge via the bench**

Re-run Step 1's bench after routing `gather_haps_readbound` through `gather_vk`.
- **No regression** → keep the unified `gather_vk` path; delete the "provably byte-identical" comment prose (query.rs:1207-1229).
- **Regression** → revert `gather_haps_readbound` to its tuned hand-inlined merge; delete only the now-redundant equivalence prose, keeping a one-line note that the tuned merge is retained per bench. Record the numbers in the commit message.

- [ ] **Step 4: Run the Rust suite**

Run: `pixi run test-rust`
Expected: PASS (byte-identical tests confirm merged output unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/query/gather.rs src/query/oracle.rs
git commit -m "refactor(query): extract gather_vk for the var_key snp+indel merge

<one line: bench result + unify-or-retain decision for gather_haps_readbound>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `DenseClass` dense source + `dense_view` accessor

Replace `DenseUnion.src: Vec<(bool, usize)>` with `Vec<(DenseClass, usize)>` and collapse the three `if is_indel { dense_indel } else { dense_snp }` dispatch sites behind one accessor.

**Files:**
- Modify: `src/query/union.rs` (`DenseUnion.src` field type; `dense_union` construction)
- Modify: `src/query/reader.rs` (add `dense_view`)
- Modify: `src/query/gather.rs` (the `PresenceBitWriter` `set` closures from Task 2)

**Interfaces:**
- Consumes: `crate::dense::DenseClass` (`enum { Snp = 0, Indel = 1 }`, already exists at `src/dense.rs:8`).
- Produces:

```rust
impl ContigReader {
    /// The dense view backing `class`, or `None` if this contig has no table of
    /// that class. Replaces the `if is_indel { &self.dense_indel } else { ... }`
    /// dispatch that a `(bool, usize)` src forced at every carriage test.
    pub(crate) fn dense_view(&self, class: DenseClass) -> Option<&sidecar::DenseView> {
        match class {
            DenseClass::Snp => self.dense_snp.as_ref(),
            DenseClass::Indel => self.dense_indel.as_ref(),
        }
    }
}
```

- [ ] **Step 1: Change `DenseUnion.src` to `Vec<(DenseClass, usize)>`**

In `union.rs`, update the field type and the `dense_union` builder to push `(DenseClass::Snp, col)` / `(DenseClass::Indel, col)` instead of `(false, col)` / `(true, col)`. Add `use crate::dense::DenseClass;`.

- [ ] **Step 2: Add `dense_view` to `reader.rs`**

Paste the accessor above (adjust the `DenseView` path to wherever Task 1 placed it — `sidecar::DenseView`).

- [ ] **Step 3: Rewrite the carriage closures**

In `gather.rs`, each `PresenceBitWriter::push_hap` `set` closure becomes:

```rust
presence.push_hap(nbits, |k| {
    let j = ds + k;
    let (class, dcol) = dense.src[j];
    let carried = reader
        .dense_view(class)
        .expect("dense src implies table")
        .carried(hap, dcol);
    carried && dense.v_ends[j] > qs
});
```

- [ ] **Step 4: Run the Rust suite**

Run: `pixi run test-rust`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/query
git commit -m "refactor(query): DenseUnion src uses DenseClass + dense_view accessor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `Range<usize>` internal ranges + drop `let hap = col` alias

Migrate the clearly-internal half-open range types to `std::ops::Range<usize>` and delete the no-op `hap` alias. **`HapRanges` stays on `(usize,usize)` (Task 7)** to keep the gvl coordination mechanical; do NOT touch `gather_haps_readbound`'s current arg types here.

**Files:**
- Modify: `src/query/reader.rs` (`vk_snp_overlap`/`vk_indel_overlap`/`dense_snp_overlap`/`dense_indel_overlap` returns; `SubStreamView::column`), `src/query/sidecar.rs` (`column`), `src/query/union.rs` (`DenseUnion::overlap` return), `src/query/gather.rs` (`RangesBundle` range fields; `gather_vk` params from Task 3; alias removal), `src/py_query_ranges.rs` (RangesBundle construction from numpy)

**Interfaces:**
- Produces: `RangesBundle` fields `dense_range`, `vk_snp_range`, `vk_indel_range`, `dense_snp_range`, `dense_indel_range` become `Vec<std::ops::Range<usize>>`; the four `*_overlap` methods and `column`/`DenseUnion::overlap` return `Range<usize>`.

- [ ] **Step 1: Change `column`, `*_overlap`, `DenseUnion::overlap`, and `gather_vk` range types to `Range<usize>`**

Replace `(usize, usize)` returns/params with `Range<usize>` (`a..b` instead of `(a, b)`), including `gather_vk`'s two range params added in Task 3. Update call sites to use `r.start`/`r.end` or `&buf[r.clone()]` / `r.len()` instead of `.0`/`.1`.

- [ ] **Step 2: Change `RangesBundle` range fields to `Vec<Range<usize>>`**

Update `find_ranges` (builder) and `gather_ranges` (consumer) accordingly.

- [ ] **Step 3: Update `py_query_ranges.rs`**

Where it constructs `RangesBundle` from numpy pair arrays, map each `(s, e)` to `s..e`; where it reads back ranges for the return dicts, map `r` to `(r.start, r.end)` (the numpy on-wire format is unchanged).

- [ ] **Step 4: Delete the `let hap = col;` alias**

Remove `let hap = col; // sample-major hap index == flat column` at each gather loop (587, 846, 994, 1171 and the `overlap_batch` copy) and use `col` directly in the carriage closures.

- [ ] **Step 5: Build + run both suites**

Run: `pixi run bash -lc 'cargo check --no-default-features --features conversion'` then `pixi run test-rust` then `pixi run test`
Expected: all PASS (the `py_query_ranges` numpy contract is byte-for-byte unchanged; `test_ranges_split.rs` + Python FFI tests confirm).

- [ ] **Step 6: Commit**

```bash
git add src/query src/py_query_ranges.rs
git commit -m "refactor(query): use Range<usize> for internal half-open ranges

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `query::oracle` submodule — de-`_pub` the oracle names

Rename the four oracle-only symbols out of the top-level `query::` namespace into `query::oracle::`, dropping the un-idiomatic `_pub` suffix, and update this repo's test imports. No gvl impact (gvl imports none of these).

**Files:**
- Modify: `src/query/oracle.rs` (rename fns), `src/query/mod.rs` (remove the top-level re-exports added in Task 1 Step 2)
- Modify: `tests/test_query.rs`, `tests/test_batch.rs`, `tests/test_decode_mat.rs`, `tests/test_readbound_gather.rs` (import paths)

**Interfaces:**
- Produces (new canonical paths): `query::oracle::overlap_sample`, `query::oracle::gather_ranges_readbound`, `query::oracle::decode_keyref`, `query::oracle::decode_keyref_alt`.

- [ ] **Step 1: Rename in `oracle.rs`**

`decode_keyref_pub` → `decode_keyref`, `decode_keyref_alt_pub` → `decode_keyref_alt`. Keep `overlap_sample`/`gather_ranges_readbound` names. Update their doc-comments (drop the "`_pub`/`bits_get_bit`" naming references; the module rustdoc now says "oracle/testing surface").

- [ ] **Step 2: Drop the top-level re-exports from `mod.rs`**

Remove the `pub use oracle::{...}` line added in Task 1 Step 2. Keep `pub mod oracle;`. Add a `mod.rs` rustdoc table documenting production entry points (`overlap_batch`/`find_ranges`/`gather_ranges`/`gather_haps_readbound`/`read_ranges`) vs `oracle::*`, and co-document the `BatchResult` (unified-dense) vs `BatchResultSplit` (split-dense) pairing.

- [ ] **Step 3: Update this repo's test imports**

- `tests/test_query.rs:13`: `use genoray_core::query::oracle::overlap_sample;`
- `tests/test_batch.rs:10`: `use genoray_core::query::{ContigReader, overlap_batch}; use genoray_core::query::oracle::overlap_sample;`
- `tests/test_decode_mat.rs:8`: `use genoray_core::query::{ContigReader}; use genoray_core::query::oracle::overlap_sample;`
- `tests/test_readbound_gather.rs:9-10`: move `gather_ranges_readbound` to `query::oracle::gather_ranges_readbound`; `decode_keyref_alt_pub` → `query::oracle::decode_keyref_alt`. Keep `bits_get_bit`, `BatchResultSplit`, `ContigReader`, `HapCalls`, `KeyRef`, `find_ranges`, `gather_haps_readbound`, `overlap_batch` at their unchanged paths.

- [ ] **Step 4: Run the Rust suite**

Run: `pixi run test-rust`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/query tests
git commit -m "refactor(query): group oracle-only entry points under query::oracle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `HapRanges<'_>` params struct for `gather_haps_readbound`

Fold the 8 positional args into a borrowed params struct with construction-time length validation. **This is the one public-surface change** — gvl's call sites are updated in Task 8, so the tree is expected to build but gvl won't compile against it until then.

**Files:**
- Modify: `src/query/gather.rs` (add `HapRanges`; change `gather_haps_readbound` signature + body)
- Modify: `tests/test_readbound_gather.rs` (the two `gather_haps_readbound` call sites, ~260, 332, 490)

**Interfaces:**
- Produces:

```rust
/// Borrowed per-query range slices for `gather_haps_readbound`, bundling the six
/// parallel slices + ploidy so the parallel-length contract is validated once at
/// construction instead of via five `assert_eq!`s at every call. Row layout is
/// unchanged: `dense_*_range` are per-query (len `n_q`); `vk_*_range` are
/// per-(query,ploid) (len `n_q * ploidy`, row `q*ploidy + p`).
pub struct HapRanges<'a> {
    pub region_starts: &'a [u32],
    pub orig_samples: &'a [usize],
    pub vk_snp_range: &'a [(usize, usize)],
    pub vk_indel_range: &'a [(usize, usize)],
    pub dense_snp_range: &'a [(usize, usize)],
    pub dense_indel_range: &'a [(usize, usize)],
    pub ploidy: usize,
}

impl<'a> HapRanges<'a> {
    /// Validate the parallel-slice length contract (panics on mismatch, matching
    /// the invariants `gather_haps_readbound` previously asserted inline).
    pub fn new(
        region_starts: &'a [u32],
        orig_samples: &'a [usize],
        vk_snp_range: &'a [(usize, usize)],
        vk_indel_range: &'a [(usize, usize)],
        dense_snp_range: &'a [(usize, usize)],
        dense_indel_range: &'a [(usize, usize)],
        ploidy: usize,
    ) -> Self {
        let n_q = region_starts.len();
        assert_eq!(orig_samples.len(), n_q, "orig_samples len must equal n_q");
        assert_eq!(dense_snp_range.len(), n_q, "dense_snp_range len must equal n_q");
        assert_eq!(dense_indel_range.len(), n_q, "dense_indel_range len must equal n_q");
        assert_eq!(vk_snp_range.len(), n_q * ploidy, "vk_snp_range len must equal n_q*ploidy");
        assert_eq!(vk_indel_range.len(), n_q * ploidy, "vk_indel_range len must equal n_q*ploidy");
        Self { region_starts, orig_samples, vk_snp_range, vk_indel_range, dense_snp_range, dense_indel_range, ploidy }
    }
}

pub fn gather_haps_readbound(reader: &ContigReader, rb: &HapRanges<'_>) -> BatchResultSplit;
```

- [ ] **Step 1: Write the failing test for construction validation**

Add to `tests/test_readbound_gather.rs`:

```rust
#[test]
#[should_panic(expected = "vk_snp_range len must equal n_q*ploidy")]
fn test_hapranges_new_rejects_mismatched_vk_len() {
    use genoray_core::query::HapRanges;
    let _ = HapRanges::new(&[0u32, 100], &[0, 1], &[(0, 1)], &[(0, 1)], &[(0, 0), (0, 0)], &[(0, 0), (0, 0)], 2);
}
```

- [ ] **Step 2: Run it — expect a compile error (HapRanges undefined)**

Run: `pixi run bash -lc 'cargo test --no-default-features --features conversion test_hapranges_new_rejects_mismatched_vk_len 2>&1 | tail -20'`
Expected: FAIL — `cannot find ... HapRanges`.

- [ ] **Step 3: Add `HapRanges` and change the signature**

Paste the struct + impl into `gather.rs`; re-export `HapRanges` from `mod.rs` (`pub use gather::HapRanges;`). Change `gather_haps_readbound` to `(reader, rb: &HapRanges<'_>)`, replace the five inline `assert_eq!`s with reads off `rb.*`, delete `#[allow(clippy::too_many_arguments)]` (keep `needless_range_loop` if still needed).

- [ ] **Step 4: Update the in-repo call sites**

In `tests/test_readbound_gather.rs`, wrap each existing 8-arg call in `HapRanges::new(...)`:

```rust
let rb = HapRanges::new(&region_starts, &orig_samples, &vk_snp_range, &vk_indel_range, &dense_snp_range, &dense_indel_range, ploidy);
let flat = gather_haps_readbound(&reader, &rb);
```

- [ ] **Step 5: Run the test + suite**

Run: `pixi run test-rust`
Expected: PASS — including the new `#[should_panic]` test and the byte-identical/readbound tests.

- [ ] **Step 6: Commit**

```bash
git add src/query tests/test_readbound_gather.rs
git commit -m "refactor(query)!: gather_haps_readbound takes a HapRanges params struct

BREAKING CHANGE: gather_haps_readbound(reader, region_starts, orig_samples,
vk_snp_range, vk_indel_range, dense_snp_range, dense_indel_range, ploidy) is now
gather_haps_readbound(reader, &HapRanges::new(...)). Downstream gvl updated in
lockstep (svar2-m6b-kernel).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Coordinated GenVarLoader update (paired PR, separate repo)

Update the gvl `svar2-m6b-kernel` branch's 4 `gather_haps_readbound` call sites to the `HapRanges` API and rebuild against the SP-2 branch.

**Files (in `/carter/users/dlaub/projects/GenVarLoader`, branch `svar2-m6b-kernel`):**
- Modify: `src/ffi/mod.rs` (4 call sites: ~963, 1116, 1227, and the 4th)
- Modify: `Cargo.toml` if it path-pins genoray_core to a rev/branch (point at the SP-2 branch for CI; the maintainer decides the final pin)

**Interfaces:**
- Consumes: `genoray_core::query::{gather_haps_readbound, HapRanges}` from Task 7.

- [ ] **Step 1: Rewrite each call site**

Each site currently reads:

```rust
let br = genoray_core::query::gather_haps_readbound(
    reader, &region_starts_v, &orig_samples_v, &vk_snp_range_v, &vk_indel_range_v,
    &dense_snp_range_v, &dense_indel_range_v, ploidy,
);
```

becomes:

```rust
let rb = genoray_core::query::HapRanges::new(
    &region_starts_v, &orig_samples_v, &vk_snp_range_v, &vk_indel_range_v,
    &dense_snp_range_v, &dense_indel_range_v, ploidy,
);
let br = genoray_core::query::gather_haps_readbound(reader, &rb);
```

The existing `to_pairs(...)` helpers already produce `Vec<(usize,usize)>`, which `HapRanges` borrows as-is — no marshalling change.

- [ ] **Step 2: Point gvl's genoray_core dep at the SP-2 branch and build**

Run (in the gvl repo): `cargo build` (or gvl's maturin build task).
Expected: compiles clean against the SP-2 branch.

- [ ] **Step 3: Run gvl's Rust + Python read-bound tests**

Run gvl's Rust suite and `tests/dataset/test_svar2_readbound_haps.py`.
Expected: PASS — confirms the coordinated signature change is behavior-identical downstream.

- [ ] **Step 4: Commit on the gvl branch (paired PR)**

```bash
git add src/ffi/mod.rs Cargo.toml
git commit -m "refactor(svar2): adopt genoray HapRanges for gather_haps_readbound

Pairs with genoray SP-2 (query.rs module split). See genoray#<PR>.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9 (optional): `rvk.rs::dense2sparse_vk` SAFETY comments

Document the `get_unchecked` bound invariants (audit item; S effort, zero risk). Cut this task if SP-2 should stay strictly `query/`-scoped.

**Files:**
- Modify: `src/rvk.rs:133-136, 150, 187, 191, 193, 207`

- [ ] **Step 1: Add a `// SAFETY:` comment above each `get_unchecked` cluster**

For each elision, cite the in-bounds invariant, e.g.:

```rust
// SAFETY: `v < v_variants` bounds `flat_idx = v * ... + col`; `flat_idx >> 6`
// therefore indexes within `words` (len = ceil(n_bits/64)). `alt_offsets[v]`
// and `alt_offsets[v+1]` bound the chunk slice by construction of the CSR.
unsafe { *words.get_unchecked(flat_idx >> 6) }
```

Match each comment to the actual index/invariant at that line (read the surrounding loop to state the real bound — do not paste the example verbatim).

- [ ] **Step 2: Build with clippy**

Run: `pixi run bash -lc 'cargo clippy --no-default-features --features conversion -- -D warnings'`
Expected: clean.

- [ ] **Step 3: Run the Rust suite**

Run: `pixi run test-rust`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/rvk.rs
git commit -m "docs(rvk): add SAFETY comments to dense2sparse_vk get_unchecked cluster

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification (before opening the PR)

- [ ] `pixi run test-rust` — full Rust suite green.
- [ ] `pixi run test` — full Python suite green.
- [ ] `git grep -n "let hap = col"` — returns nothing (alias fully removed).
- [ ] `git grep -n "decode_keyref_pub\|decode_keyref_alt_pub\|_pub"` in `src/query/` — returns nothing.
- [ ] `git grep -n "(bool, usize)" src/query/` — returns nothing (DenseClass migration complete).
- [ ] gvl `svar2-m6b-kernel` builds + tests green against the SP-2 branch (Task 8).
- [ ] Confirm no `import genoray` public name changed → no `skills/genoray-api/SKILL.md` edit (recorded in PR description).

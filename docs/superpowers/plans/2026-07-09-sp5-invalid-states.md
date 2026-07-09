# SP-5 "Invalid states unrepresentable" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stringly-typed / primitive-obsessed representations of small closed sets with enums, shared `Literal`s, `TypedDict`s, `Range`, and named constants — a behavior-preserving type-hygiene pass.

**Architecture:** Two independent, behavior-preserving PRs. PR-A (Python) tightens `_mutcat`, `_signatures`, `_svar2_batch`, `_types`, `_utils`. PR-B (Rust) tightens `dense.rs`, `streams.rs`, `src/query/`, `svar2-codec`. No runtime values, on-disk formats, or query results change; correctness is gated by the *existing* test suites plus the type checker.

**Tech Stack:** Python 3.10+ (numpy, polars, `IntEnum`, `TypedDict`, `pyrefly`), Rust (pyo3/abi3, rayon), `pixi` for envs, `ruff` + `prek` hooks.

## Global Constraints

- **Behavior-preserving.** No new runtime values, on-disk formats, or query results. The regression gate is the *existing* suite staying green, not new golden values.
- **Two separate PRs.** PR-A (Tasks A1–A4) on the current `sp5-invalid-states` branch (it carries the spec + this plan). PR-B (Tasks B1–B4) on a fresh `sp5-rust` branch cut from `main`. The two touch disjoint files and may land in either order.
- **Python test command:** `pixi run pytest <paths>`; full gate `pixi run test`. Type-check gate: `pixi run pyrefly check` (or the repo's configured `pyrefly` invocation).
- **Rust test command:** `pixi run bash -lc 'cargo test --no-default-features'` — the `--no-default-features` flag is REQUIRED or the pyo3 test binary fails to link with `undefined symbol: _Py_Dealloc`. Lint gate: `cargo clippy --no-default-features` + `cargo fmt`.
- **Commit style:** Conventional Commits (`feat:`/`fix:`/`refactor:`/`docs:`). End every commit body with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **prek hooks** must be installed (`pixi run prek-install`) before committing; hooks run `cargo fmt/check/clippy`, `pyrefly`, and `commitizen`.
- **Public-surface rule:** only Task A4 touches public surface (annotations on already-public `SparseVar2` batch methods); it updates `skills/genoray-api/SKILL.md` in the same commit.
- **SP-7 coexistence:** Task A1 must update *every* current `SENTINELS[...]` site in `_mutcat.py`; SP-7's later oracle relocation carries the new name along. No ordering dependency.

---

## PR-A — Python type hygiene (branch `sp5-invalid-states`)

### Task A1: `SENTINELS` dict → `IntEnum`

**Files:**
- Modify: `python/genoray/_mutcat.py` (definition at :136-141; 23 `SENTINELS[...]` call sites: 192, 195, 197, 219, 221, 228, 251, 253, 258, 318, 368, 510, 520, 533, 695, 726, 781, 786, 845 — plus the three docstring mentions 186, 189, 213)
- Test: `tests/test_mutcat_sentinel.py` (new)

**Interfaces:**
- Produces: `class Sentinel(IntEnum)` with members `DBS_PARTNER = -1`, `UNCLASSIFIED = -2`, `MISSING = -3`, `NOT_ANNOTATED = -4`, exported from `_mutcat`. Module-level `_REF_MISMATCH = -99` stays a bare constant (not an enum member).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mutcat_sentinel.py`:
```python
import numpy as np

from genoray._mutcat import Sentinel


def test_sentinel_values_unchanged():
    # The int values are a public wire contract (they appear in returned arrays).
    assert int(Sentinel.DBS_PARTNER) == -1
    assert int(Sentinel.UNCLASSIFIED) == -2
    assert int(Sentinel.MISSING) == -3
    assert int(Sentinel.NOT_ANNOTATED) == -4


def test_sentinel_is_int_compatible():
    # IntEnum members must behave as ints for numpy fills / comparisons.
    arr = np.full(3, Sentinel.UNCLASSIFIED, dtype=np.int16)
    assert arr.dtype == np.int16
    assert bool((arr == Sentinel.UNCLASSIFIED).all())
    assert np.int16(Sentinel.UNCLASSIFIED) == np.int16(-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_mutcat_sentinel.py -v`
Expected: FAIL with `ImportError: cannot import name 'Sentinel'`.

- [ ] **Step 3: Add the `IntEnum`**

In `python/genoray/_mutcat.py`, ensure `from enum import IntEnum` is imported at the top, then replace the `SENTINELS` dict (lines 136-141) with:
```python
class Sentinel(IntEnum):
    """Negative sentinel codes in the int16 mutation-code space (closed set)."""

    DBS_PARTNER = -1  # 3' half of an adjacency doublet; never counted
    UNCLASSIFIED = -2  # symbolic/complex/MNV>2bp/non-ACGT
    MISSING = -3
    NOT_ANNOTATED = -4
```
Leave `_REF_MISMATCH = -99` and its comment (lines 143-146) exactly as-is.

- [ ] **Step 4: Rewrite every call site**

Replace across `_mutcat.py` (use a global search for `SENTINELS[`):
- `SENTINELS["UNCLASSIFIED"]` → `Sentinel.UNCLASSIFIED`
- `SENTINELS["DBS_PARTNER"]` → `Sentinel.DBS_PARTNER`
- `SENTINELS["NOT_ANNOTATED"]` → `Sentinel.NOT_ANNOTATED`
- `SENTINELS["MISSING"]` → `Sentinel.MISSING`

Update the docstring mentions at :186, :189, :213 from `SENTINELS['UNCLASSIFIED']` to `Sentinel.UNCLASSIFIED`. Confirm zero remaining matches for `SENTINELS` (grep must return nothing).

- [ ] **Step 5: Run new + existing mutcat tests**

Run: `pixi run pytest tests/test_mutcat_sentinel.py tests/test_mutcat.py tests/test_svar_mutations.py -v`
Expected: PASS (the oracle tests confirm identical code values).

- [ ] **Step 6: Type-check**

Run: `pixi run pyrefly check python/genoray/_mutcat.py`
Expected: no new errors.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_mutcat.py tests/test_mutcat_sentinel.py
git commit -m "refactor(mutcat): SENTINELS dict -> Sentinel IntEnum

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task A2: Own the `Kind` `Literal` in `_mutcat`

**Files:**
- Modify: `python/genoray/_mutcat.py` (add `Kind`; annotate `labels` :167, `code_ranges` :158, `count_matrix` :660)
- Modify: `python/genoray/_signatures.py` (:22 remove local `Kind`, import it instead)

**Interfaces:**
- Produces: `Kind = Literal["SBS96", "DBS78", "ID83"]` defined in `_mutcat`. `_signatures.Kind` remains a resolvable name (re-imported).

- [ ] **Step 1: Define `Kind` in `_mutcat.py`**

Ensure `from typing import Literal` is present in `_mutcat.py`. Add near the top of the codebook section (just below `MUTCAT_VERSION` / above `code_ranges`):
```python
Kind = Literal["SBS96", "DBS78", "ID83"]
```

- [ ] **Step 2: Annotate the three functions**

In `_mutcat.py`:
- `def code_ranges() -> dict[str, tuple[int, int]]:` → `-> dict[Kind, tuple[int, int]]:`
- `def labels(kind: str) -> list[str]:` → `def labels(kind: Kind) -> list[str]:` (keep the runtime `ValueError` guard at :169-173 as defense-in-depth)
- `def count_matrix(..., kind: Literal["SBS96", "DBS78", "ID83"], ...)` → `kind: Kind`

- [ ] **Step 3: Re-point `_signatures.py`**

In `python/genoray/_signatures.py`:
- Change the existing import `from ._mutcat import labels` → `from ._mutcat import Kind, labels`
- Delete the local redefinition `Kind = Literal["SBS96", "DBS78", "ID83"]` at :22
- Leave `from typing import Literal` in place only if still used elsewhere in the file; otherwise remove it.

- [ ] **Step 4: Type-check + run signature/mutcat tests**

Run: `pixi run pyrefly check python/genoray/_mutcat.py python/genoray/_signatures.py`
Expected: no new errors.
Run: `pixi run pytest tests/test_mutcat.py tests/test_svar_mutations.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_mutcat.py python/genoray/_signatures.py
git commit -m "refactor(mutcat): own the Kind Literal in the codebook module

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task A3: Dedup the `DTYPE` TypeVar

**Files:**
- Modify: `python/genoray/_utils.py` (:18 remove local `DTYPE`, import from `_types`)

**Interfaces:**
- Consumes: `DTYPE = TypeVar("DTYPE", bound=np.generic)` from `_types.py:9` (no cycle: `_types.py` imports only `typing` + `numpy`).

- [ ] **Step 1: Re-point `_utils.py`**

In `python/genoray/_utils.py`:
- Delete the line `DTYPE = TypeVar("DTYPE", bound=np.generic)` (:18).
- Add `from ._types import DTYPE` to the internal-imports block.
- If `TypeVar` is no longer referenced in `_utils.py` after removal, drop it from the `from typing import ...` line (keep `overload` etc. if still used).

- [ ] **Step 2: Type-check + import smoke test**

Run: `pixi run pyrefly check python/genoray/_utils.py python/genoray/_types.py`
Expected: no new errors (in particular, no import-cycle error).
Run: `pixi run python -c "import genoray; from genoray import _utils"`
Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add python/genoray/_utils.py
git commit -m "refactor(utils): import DTYPE TypeVar from _types instead of redefining

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task A4: `TypedDict`s for the svar2 bundle shapes + SKILL

**Files:**
- Modify: `python/genoray/_svar2_batch.py` (add TypedDicts; annotate `overlap_batch` :22, `read_ranges` :50, `find_ranges` :71, `gather_ranges` :108)
- Modify: `skills/genoray-api/SKILL.md` (document the field-name/dtype contract near the batch-method section, ~:275-283)

**Interfaces:**
- Produces: `BatchResult` and `RangesBundle` TypedDicts. `overlap_batch`/`read_ranges`/`gather_ranges` → `BatchResult`; `find_ranges` → `RangesBundle` with `out: Mapping[str, np.ndarray] | None`.
- Note: these are static-only types; at runtime the values are the plain `dict`s Rust returns. No FFI, `out=` reuse, or genvarloader behavior changes.

- [ ] **Step 1: Add the TypedDicts**

At the top of `python/genoray/_svar2_batch.py` (after imports; add `from collections.abc import Mapping` and `from typing import TypedDict` to the existing import block):
```python
class BatchResult(TypedDict):
    """Two-channel batch-query result contract (see py_query_batch.rs)."""

    vk_pos: np.ndarray
    vk_key: np.ndarray
    vk_off: np.ndarray
    dense_pos: np.ndarray
    dense_key: np.ndarray
    dense_range: np.ndarray
    dense_present: np.ndarray
    lut_bytes: np.ndarray
    lut_off: np.ndarray
    n_regions: int
    n_samples: int
    ploidy: int


class RangesBundle(TypedDict):
    """Compact search-only bundle replayed by ``gather_ranges`` (see py_query_ranges.rs)."""

    dense_range: np.ndarray
    region_starts: np.ndarray
    sample_cols: np.ndarray
    vk_snp_range: np.ndarray
    vk_indel_range: np.ndarray
    dense_snp_range: np.ndarray
    dense_indel_range: np.ndarray
    n_regions: int
    n_samples: int
    ploidy: int
```

- [ ] **Step 2: Annotate the four methods**

In `_svar2_batch.py`, change the return annotations (bodies unchanged):
- `overlap_batch(...) -> dict[str, "np.ndarray"]:` → `-> BatchResult:`
- `read_ranges(...) -> dict[str, "np.ndarray"]:` → `-> BatchResult:`
- `gather_ranges(...) -> dict[str, "np.ndarray"]:` → `-> BatchResult:`
- `find_ranges(...)`: change `out: dict[str, "np.ndarray"] | None = None` → `out: Mapping[str, "np.ndarray"] | None = None`, and the return `-> dict[str, Any]:` → `-> RangesBundle:`

Leave `_regions`/`_sample_idxs` and all method bodies exactly as-is. If `Any` is now unused, remove it from the `typing` import.

- [ ] **Step 3: Type-check + run svar2 batch tests**

Run: `pixi run pyrefly check python/genoray/_svar2_batch.py`
Expected: no new errors.
Run: `pixi run pytest tests/ -k "svar2 or batch or ranges" -v`
Expected: PASS (identifies and runs the svar2 batch-query tests; if the `-k` filter matches nothing, fall back to `pixi run pytest tests/test_svar2.py -v` or the file that exercises `overlap_batch`).

- [ ] **Step 4: Update the SKILL**

In `skills/genoray-api/SKILL.md`, near the batch-method docs (~:275-283), add a short line stating the returned-dict field contract, e.g.:
```markdown
  These methods return a plain dict with a fixed field set (`BatchResult`):
  `vk_pos`/`vk_key`/`vk_off`, `dense_pos`/`dense_key`/`dense_range`/`dense_present`,
  `lut_bytes`/`lut_off`, and scalars `n_regions`/`n_samples`/`ploidy`. `find_ranges`
  returns a `RangesBundle` (`dense_range`/`region_starts`/`sample_cols`/
  `vk_snp_range`/`vk_indel_range`/`dense_snp_range`/`dense_indel_range` + the three
  scalars). The dict contract is unchanged; only static type annotations were added.
```

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_svar2_batch.py skills/genoray-api/SKILL.md
git commit -m "refactor(svar2): TypedDict the batch-query dict contracts

Adds BatchResult/RangesBundle TypedDicts for the SparseVar2 batch methods.
Static-only: the returned dict contract is unchanged. SKILL updated.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Full Python gate before opening PR-A**

Run: `pixi run test`
Expected: PASS. Open PR-A from `sp5-invalid-states` → `main`.

---

## PR-B — Rust type hygiene (branch `sp5-rust`, cut from `main`)

### Task B0: Branch setup

- [ ] **Step 1: Cut the Rust branch from main**

```bash
git checkout main
git checkout -b sp5-rust
```

---

### Task B1: Finish `(usize, usize)` → `Range<usize>`

**Files:**
- Modify: `src/query/gather.rs` (tuple sites: 86, 109, 116, 175, 398-401, 479, 481)
- Modify: `src/query/oracle.rs` (95, 97)

**Interfaces:**
- `Range<usize>` is already imported at `gather.rs:7`. `gather.rs` already uses it at 128-129, 243-256, 279-285 — this task removes the remaining `(usize, usize)` representations so the module holds one range type.

- [ ] **Step 1: Verify the FFI boundary first**

Run: `grep -n "dense_range\|snp_range\|indel_range" src/py_query_ranges.rs src/py_query_batch.rs`
Confirm whether these fields cross to Python as tuples or arrays. `dense_range` is emitted as a prebuilt pyarray, so the *in-Rust* type may change freely as long as the pyarray builder still produces the same output. If any field is `.set_item`'d directly as a tuple list, keep the tuple → `Range` conversion local to the builder. Note the findings here before editing.

- [ ] **Step 2: Convert the `RangesBundle` fields and locals**

In `src/query/gather.rs`, change the struct field types (and only these; do not rename fields):
- `pub dense_range: Vec<(usize, usize)>,` → `Vec<Range<usize>>` (:86)
- `pub dense_snp_range: Vec<(usize, usize)>,` → `Vec<Range<usize>>` (:109)
- `pub dense_indel_range: Vec<(usize, usize)>,` → `Vec<Range<usize>>` (:116)
- the local at :175 (`let ranges: Vec<(usize, usize)>`) → `Vec<Range<usize>>`
- the borrowed params-struct slices at :398-401 (`&'a [(usize, usize)]`) → `&'a [Range<usize>]`
- the `_out` locals at :479, :481 → `Vec<Range<usize>>`

At each construction site replace `(a, b)` with `a..b`; at each read site replace `.0`/`.1` with `.start`/`.end` (or use `&buf[r.clone()]`, `r.len()`, direct iteration where it simplifies). Apply the identical change to `src/query/oracle.rs:95,97`.

- [ ] **Step 3: Build, test, clippy**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: PASS (the query/oracle parity tests are the gate that ranges are identical).
Run: `pixi run bash -lc 'cargo clippy --no-default-features'`
Expected: no new warnings; `cargo fmt`.

- [ ] **Step 4: Python FFI parity**

Run: `pixi run pytest tests/ -k "svar2 or batch or ranges" -v`
Expected: PASS (confirms the pyarray builders still emit byte-identical dicts).

- [ ] **Step 5: Commit**

```bash
git add src/query/gather.rs src/query/oracle.rs
git commit -m "refactor(query): finish (usize,usize) -> Range<usize> in gather/oracle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task B2: `EnumKey` trait + generic `EnumMap<K, T, N>`

**Files:**
- Modify: `src/dense.rs` (`DenseClass` COUNT/ALL/index, `DenseMap<T>`)
- Modify: `src/streams.rs` (`StreamTag` COUNT/ALL/index, `StreamMap<T>`)
- Create: `src/enum_map.rs` (the shared trait + generic map)
- Modify: `src/lib.rs` (add `mod enum_map;`)

**Interfaces:**
- Produces: `pub trait EnumKey: Copy { const COUNT: usize; const ALL: &'static [Self]; fn index(self) -> usize; }` and `pub struct EnumMap<K: EnumKey, T, const N: usize>`. `pub type DenseMap<T> = EnumMap<DenseClass, T, { DenseClass::COUNT }>;` and `pub type StreamMap<T> = EnumMap<StreamTag, T, { StreamTag::COUNT }>;` keep every existing call site and unit test source-compatible.

- [ ] **Step 1: Write `src/enum_map.rs`**

```rust
//! One array-backed fixed-size map keyed by a small enum. Shared by `dense.rs`
//! (DenseClass) and `streams.rs` (StreamTag) — see EnumKey impls there.

use std::marker::PhantomData;

/// A small closed enum usable as a fixed-size map key. `ALL` lists every
/// variant in `index()` order; `COUNT == ALL.len()`.
pub trait EnumKey: Copy {
    const COUNT: usize;
    const ALL: &'static [Self];
    fn index(self) -> usize;
}

/// Fixed-size map keyed by `K`, backed by an array (O(1), no hashing).
/// `N` is pinned to `K::COUNT` by the concrete type aliases.
pub struct EnumMap<K: EnumKey, T, const N: usize> {
    slots: [T; N],
    _k: PhantomData<K>,
}

impl<K: EnumKey, T, const N: usize> EnumMap<K, T, N> {
    pub fn from_fn(mut f: impl FnMut(K) -> T) -> Self {
        Self {
            slots: std::array::from_fn(|i| f(K::ALL[i])),
            _k: PhantomData,
        }
    }
    #[inline]
    pub fn get(&self, k: K) -> &T {
        &self.slots[k.index()]
    }
    #[inline]
    pub fn get_mut(&mut self, k: K) -> &mut T {
        &mut self.slots[k.index()]
    }
    pub fn iter(&self) -> impl Iterator<Item = (K, &T)> {
        K::ALL.iter().copied().zip(self.slots.iter())
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut T)> {
        K::ALL.iter().copied().zip(self.slots.iter_mut())
    }
    pub fn into_iter_tagged(self) -> impl Iterator<Item = (K, T)> {
        K::ALL.iter().copied().zip(self.slots)
    }
}
```
Add `mod enum_map;` to `src/lib.rs` (near the other `mod` declarations).

- [ ] **Step 2: Compile-check the generic map in isolation FIRST**

Run: `pixi run bash -lc 'cargo check --no-default-features'`
Expected: `enum_map.rs` compiles. **If `std::array::from_fn` or the const-generic `N` fails on the pinned toolchain, STOP and apply the fallback** (see Step 6) before touching `dense.rs`/`streams.rs`.

- [ ] **Step 3: Port `dense.rs`**

In `src/dense.rs`:
- Replace the inherent `impl DenseClass { const COUNT; const ALL; fn index }` block's `COUNT`/`ALL`/`index` by implementing the trait instead. Keep `key_bytes()` and `cost_class()` as inherent methods:
```rust
use crate::enum_map::{EnumKey, EnumMap};

impl EnumKey for DenseClass {
    const COUNT: usize = 2;
    const ALL: &'static [DenseClass] = &[DenseClass::Snp, DenseClass::Indel];
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}
```
- Delete the whole `pub struct DenseMap<T> { ... }` + its `impl` block (lines 59-87) and replace with:
```rust
pub type DenseMap<T> = EnumMap<DenseClass, T, { DenseClass::COUNT }>;
```
- `DENSE_REGISTRY` and its `[DenseSpec; DenseClass::COUNT]` type stay valid (`COUNT` is still an associated const on `DenseClass`).

- [ ] **Step 4: Port `streams.rs`**

In `src/streams.rs`, mirror the change:
```rust
use crate::enum_map::{EnumKey, EnumMap};

impl EnumKey for StreamTag {
    const COUNT: usize = 2;
    const ALL: &'static [StreamTag] = &[StreamTag::VarKeySnp, StreamTag::VarKeyIndel];
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}
```
Delete `pub struct StreamMap<T> { ... }` + its `impl` (lines 53-78) and add:
```rust
pub type StreamMap<T> = EnumMap<StreamTag, T, { StreamTag::COUNT }>;
```
Keep `REGISTRY`, `StreamSpec`, `PostMergeHook` untouched.

- [ ] **Step 5: Build + run the module unit tests (unchanged)**

Run: `pixi run bash -lc 'cargo test --no-default-features dense streams enum_map'`
Expected: PASS — the existing `test_densemap_get_set` / `test_streammap_get_set` / registry tests run against the aliases unchanged.
Run: `pixi run bash -lc 'cargo clippy --no-default-features'`; `cargo fmt`.

- [ ] **Step 6: FALLBACK (only if Step 2 failed)**

If const-generic `N` via `{ DenseClass::COUNT }` is rejected on the pinned toolchain: keep `EnumKey` as above but do **not** introduce `EnumMap`. Instead, keep the two `DenseMap`/`StreamMap` structs and factor only their shared method bodies through a `macro_rules! impl_enum_map { ($Map:ident, $Key:ty) => { ... } }` in `enum_map.rs`, invoked once per map. This still deletes the hand-rolled `COUNT`/`ALL`/`index` (now from the trait) and de-duplicates the map bodies, satisfying the finding without const-generics. Re-run Steps 3-5 adapted to the macro.

- [ ] **Step 7: Commit**

```bash
git add src/enum_map.rs src/lib.rs src/dense.rs src/streams.rs
git commit -m "refactor(rust): unify DenseMap/StreamMap behind EnumKey/EnumMap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task B3: `StreamTag::class() -> Class`

**Files:**
- Modify: `src/streams.rs` (add inherent `class()` on `StreamTag`)

**Interfaces:**
- Consumes: `crate::cost_model::Class` (`{Snp, Indel}`). `DenseClass::cost_class()` is the existing sibling; this adds the symmetric bridge on `StreamTag`.

- [ ] **Step 1: Write a failing test**

Add to the `#[cfg(test)] mod tests` in `src/streams.rs`:
```rust
#[test]
fn test_streamtag_class_bridge() {
    use crate::cost_model::Class;
    assert_eq!(StreamTag::VarKeySnp.class(), Class::Snp);
    assert_eq!(StreamTag::VarKeyIndel.class(), Class::Indel);
}
```

- [ ] **Step 2: Run it, expect a compile failure**

Run: `pixi run bash -lc 'cargo test --no-default-features streams'`
Expected: FAIL — `no method named 'class'`.

- [ ] **Step 3: Add the bridge**

In `src/streams.rs`, add to `impl StreamTag`:
```rust
    /// Bridge to the canonical variant-class axis (mirrors `DenseClass::cost_class`).
    #[inline]
    pub fn class(self) -> crate::cost_model::Class {
        match self {
            StreamTag::VarKeySnp => crate::cost_model::Class::Snp,
            StreamTag::VarKeyIndel => crate::cost_model::Class::Indel,
        }
    }
```
(If `Class` derives `PartialEq`/`Debug`, the test compiles as written; if not, add `#[derive(Debug, PartialEq)]` to `Class` in `cost_model.rs` — it is a trivial two-variant enum.)

- [ ] **Step 4: Run test**

Run: `pixi run bash -lc 'cargo test --no-default-features streams'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/streams.rs src/cost_model.rs
git commit -m "refactor(rust): add StreamTag::class() bridge to cost_model::Class

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task B4: Name the svar2-codec bit-shift magic numbers

**Files:**
- Modify: `svar2-codec/src/lib.rs` (add consts; use in `snp_code_to_key` :46, `encode_snp` :181, `swar_reduce_portable` :142, `encode_alt_inline` :204, `decode_alt_inline` :215, `pext_reduce` :121/:129)
- Modify: `src/rvk.rs` and `src/query/gather.rs:719` (import + use `PAYLOAD_TOP_SHIFT` for the `1 << 25` sites)

**Interfaces:**
- Produces: `pub const PAYLOAD_TOP_SHIFT: u32 = 25;` and `pub const ILEN_SHIFT: u32 = 27;` in `svar2-codec`.

- [ ] **Step 1: Add the consts**

In `svar2-codec/src/lib.rs`, near the existing `MIN_I31`/`MAX_INLINE_ALT_LEN` consts:
```rust
/// Bit position of base[0]'s 2-bit code in the inline payload (base[0] at [26:25]).
/// The single source of truth for the payload's low half; PEXT/SWAR shifts derive
/// from it.
pub const PAYLOAD_TOP_SHIFT: u32 = 25;

/// Bit position of the 5-bit `ilen` field (occupies [31:27]).
pub const ILEN_SHIFT: u32 = 27;
```

- [ ] **Step 2: Substitute the literals (values must not change)**

- `snp_code_to_key` (:46): `((code & 3) as u32) << 25` → `<< PAYLOAD_TOP_SHIFT`
- `encode_snp` (:181): `... << 25` → `<< PAYLOAD_TOP_SHIFT`
- `swar_reduce_portable` (:142): `let mut shift: i32 = 25;` → `= PAYLOAD_TOP_SHIFT as i32;`
- `encode_alt_inline` (:204): `payload | (ilen << 27)` → `<< ILEN_SHIFT`
- `decode_alt_inline` (:211): `(payload >> 27)` → `>> ILEN_SHIFT`; (:215) `let shift = 25 - (i * 2);` → `PAYLOAD_TOP_SHIFT as usize - (i * 2)` (keep the surrounding `(payload >> shift)` intact)
- `pext_reduce` (:121): `<< 11` → `<< (PAYLOAD_TOP_SHIFT - 14)` (retain the `// 25 - 14` comment); (:129) `>> 5` → `>> (30 - PAYLOAD_TOP_SHIFT)` (retain `// 30 - 25`). The `14`/`30` are BMI2 extraction positions, not payload layout — leave as literals with the existing comments.

- [ ] **Step 3: Update the two external users**

- `src/query/gather.rs:719` (`key: 1 << 25,`) → import `use svar2_codec::PAYLOAD_TOP_SHIFT;` (match the crate's existing import path/name in that file) and write `key: 1 << PAYLOAD_TOP_SHIFT,`.
- In `src/rvk.rs`, if any non-comment `<< 25` / `>> 27` layout literal exists (the earlier grep showed only a comment at :383 — verify with `grep -n "<< 25\|>> 27\|<< 27" src/rvk.rs`), replace it likewise; if only comments, no change.

- [ ] **Step 4: Run the codec proptests (the exactness gate)**

Run: `pixi run bash -lc 'cargo test --no-default-features -p svar2-codec'`
Expected: PASS — the encode/decode round-trip proptests prove the const substitution is bit-exact.
Run: `pixi run bash -lc 'cargo test --no-default-features'` (whole workspace, covers rvk/gather).
Expected: PASS. Then `cargo clippy --no-default-features` + `cargo fmt`.

- [ ] **Step 5: Commit**

```bash
git add svar2-codec/src/lib.rs src/query/gather.rs src/rvk.rs
git commit -m "refactor(svar2-codec): name the payload bit-shift layout constants

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Full Rust + parity gate before opening PR-B**

Run: `pixi run test`
Expected: PASS. Open PR-B from `sp5-rust` → `main`.

---

## Notes for the implementer

- This is a **behavior-preserving refactor**. There are almost no new golden values to assert; the regression gate is the *existing* oracle/parity suites plus the type checker. The few new tests (A1 sentinel values, B3 bridge) exist because they assert a genuinely new, small invariant.
- Do not rename any public field, method, enum variant, or dict key. Only *types* and *internal representations* change.
- If any "Expected: PASS" step fails, treat it as a real regression (a value or contract changed) — do not paper over it; the point of SP-5 is that nothing observable changes.

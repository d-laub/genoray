# SP-5 — "Invalid states unrepresentable" type pass

**Date:** 2026-07-09
**Roadmap item:** [SP-5](../../roadmap/clean-code-audit.md#sp-5--invalid-states-unrepresentable-type-pass) · size M · risk low
**Findings:** [`03-mutcat-utils.md`](../../roadmap/audit-findings/03-mutcat-utils.md),
[`01-svar.md`](../../roadmap/audit-findings/01-svar.md),
[`05-rust-query.md`](../../roadmap/audit-findings/05-rust-query.md),
[`06-rust-rest.md`](../../roadmap/audit-findings/06-rust-rest.md)

## Goal

Close the small closed sets in the codebase that are currently modelled with
open, stringly-typed, or primitive-obsessed representations, so that invalid
states become unrepresentable and typos/layout-drift are caught statically
rather than at runtime. This is a **type-hygiene pass, not a behavior change**:
SP-5 changes no runtime values, no on-disk formats, and no query results.

## Scope

Two independently-shippable, behavior-preserving PRs from this one spec,
matching the roadmap's "splittable into Python and Rust PRs":

- **PR-A (Python):** `_mutcat.py`, `_signatures.py`, `_svar2_batch.py`,
  `_types.py`, `_utils.py`
- **PR-B (Rust):** `src/dense.rs`, `src/streams.rs`, `src/query/`,
  `svar2-codec/src/lib.rs`

The two PRs touch disjoint files and can land in either order.

### Rescoped / dropped since the audit

The code moved between the 2026-07-08 audit and this spec (SP-2 landed
`query.rs` → `query/`; SP-4 landed the `Filter` value object):

- **DROPPED — `(bool,usize)` dense source → `DenseClass`.** Already done by SP-2:
  `src/query/union.rs:22` is `Vec<(DenseClass, usize)>`. No work remains.
- **RESCOPED — `(usize,usize)` ranges → `Range`.** SP-2 already converted most of
  `gather.rs` to `Range<usize>`; the item becomes **finish** the stragglers, not
  a from-scratch conversion. See B1.

## Invariants

1. **Behavior-preserving.** Gated on the existing test suite (vcfixture oracle +
   parity tests + Rust proptests) staying green. No new runtime values.
2. **Public surface.** The only public-surface touch is A4 (type annotations on
   already-public `SparseVar2` batch methods; the dict *contract* is unchanged).
   That PR updates `skills/genoray-api/SKILL.md` in the same PR per `CLAUDE.md`.
3. **Small, reviewable PRs** — two PRs, no god-branch.

## Cross-project note

The svar2 batch-method dicts are consumed inline by the downstream
**genvarloader** M6b decode path. A4 keeps them structurally plain `dict`s
(TypedDict is a static-only annotation), so genvarloader is unaffected. The
heavier frozen-dataclass wrapping + privatization is **deferred to SP-6**, where
the `._raw` grouping / privatization decision for these methods already lives.

---

## PR-A — Python type hygiene

### A1 · `SENTINELS` dict → `IntEnum`

**Where:** `_mutcat.py:136-141` and every `SENTINELS["..."]` lookup
(192, 219, 318, 368, 510, 695, 726, 781-787, and any others found by grep).

**Now:**
```python
SENTINELS: dict[str, int] = {
    "DBS_PARTNER": -1,
    "UNCLASSIFIED": -2,
    "MISSING": -3,
    "NOT_ANNOTATED": -4,
}
_REF_MISMATCH = -99
```
A typo like `SENTINELS["UNCLASSFIED"]` is a runtime `KeyError`; values are
re-wrapped `np.int16(SENTINELS[...])` at many sites.

**After:**
```python
class Sentinel(IntEnum):
    DBS_PARTNER = -1     # 3' half of an adjacency doublet; never counted
    UNCLASSIFIED = -2    # symbolic/complex/MNV>2bp/non-ACGT
    MISSING = -3
    NOT_ANNOTATED = -4

_REF_MISMATCH = -99      # internal boundary signal, NOT a public sentinel
```
Rewrite `SENTINELS["UNCLASSIFIED"]` → `Sentinel.UNCLASSIFIED`. Because
`IntEnum` members *are* `int`, `np.int16(Sentinel.UNCLASSIFIED)`, array
comparisons (`code == Sentinel.UNCLASSIFIED`), and use as numpy fill values all
keep working unchanged. A misspelled member is now an `AttributeError` caught at
lint/type-check time.

**Keep `_REF_MISMATCH` a module-level constant, not an enum member** — the audit
and the existing comment are explicit that it is an internal boundary signal,
not a public sentinel; folding it into the public `Sentinel` enum would leak it.

**Interaction with SP-7:** `SENTINELS` is referenced by both the shipped
vectorized path (e.g. `_UNCL = np.int16(...)` at :318) and the scalar oracle
cluster (`classify_sbs96`/`classify_id83`/…) that SP-7 will relocate to the test
side. A1 must update **all** current call sites in `_mutcat.py`; SP-7's later
relocation carries `Sentinel` along with the oracle. No ordering dependency —
both spellings resolve to the same int.

### A2 · Own the `Kind` `Literal` in `_mutcat`

**Where:** `_mutcat.py:167` (`labels(kind: str)`), `:158` (`code_ranges` str
keys), `:662` (`count_matrix(..., kind: Literal[...])`); `_signatures.py:22`
(`Kind = Literal["SBS96","DBS78","ID83"]`).

The same three-way kind is expressed three ways: a bare `str` with a runtime
`ValueError` (`labels`), an inline `Literal` (`count_matrix`), and a named `Kind`
alias defined **downstream** in `_signatures.py` — which imports `labels` from
`_mutcat` yet redefines the type, so the canonical name lives below the module
that owns the codebook.

**After:**
- Define `Kind = Literal["SBS96", "DBS78", "ID83"]` **once in `_mutcat.py`** (the
  codebook owner).
- `_signatures.py` imports it (`from ._mutcat import Kind`) instead of
  redefining; `_signatures.Kind` still resolves for any existing reference.
- Annotate `labels(kind: Kind)`, `code_ranges() -> dict[Kind, tuple[int, int]]`
  (or annotate its keys as `Kind`), and `count_matrix(..., kind: Kind)`.
- The runtime `ValueError` in `labels` (169-173) stays as defense-in-depth, no
  longer the primary contract.

No public value changes; `Kind` is not in `__init__.py`'s `__all__`.

### A3 · Dedup the `DTYPE` TypeVar

**Where:** `_types.py:9` and `_utils.py:18` — both define
`DTYPE = TypeVar("DTYPE", bound=np.generic)` verbatim; `_utils.py` does not
import from `_types.py`.

**After:** keep the `_types.py` definition, delete the `_utils.py` one, and add
`from ._types import DTYPE` to `_utils.py`. Confirm no import cycle (`_types.py`
must not import `_utils.py`; it currently doesn't).

### A4 · `TypedDict` for the svar2 bundle shapes

**Where:** `_svar2_batch.py` — `overlap_batch`, `read_ranges`, `find_ranges`
(incl. its `out=` param), `gather_ranges`. These return/accept
`dict[str, np.ndarray]` / `dict[str, Any]` today.

**Decision (confirmed with maintainer):** TypedDict now; frozen dataclass +
privatization deferred to SP-6.

**After:** introduce two `TypedDict`s in `_svar2_batch.py`:

- **`BatchResult`** — the `overlap_batch`/`read_ranges`/`gather_ranges` output
  contract: `vk_pos`, `vk_key`, `vk_off`, the `dense_*` fields, the `lut_*`
  fields, and the scalars `n_regions` / `n_samples` / `ploidy`.
- **`RangesBundle`** — the `find_ranges` output / `gather_ranges` input:
  `dense_range`, `region_starts`, `sample_cols`, `vk_snp_range`,
  `vk_indel_range`, and any siblings.

**Source of truth for exact keys/dtypes:** enumerate them from the Rust FFI
dict-assembly code (`src/py_query_*.rs`) that constructs these dicts, not from
the docstrings alone — the docstrings are illustrative. If a scalar field
(`n_regions`/`n_samples`/`ploidy`) is a Python `int` rather than an array, type
it as `int`; use `NotRequired[...]` for any field a method may omit.

Annotate the four methods' returns/params with these types. Because a
`TypedDict` is a plain `dict` at runtime, **nothing about the FFI round-trip,
the `out=` buffer reuse, or genvarloader's consumption changes** — this is a
static-checking-only improvement. `out=` stays `RangesBundle | None`; the
in-place fill logic is untouched.

**SKILL update:** these four methods are documented in
`skills/genoray-api/SKILL.md`. Add the field-name/dtype contract there
(annotations only; dict contract unchanged).

---

## PR-B — Rust type hygiene

### B1 · Finish `(usize, usize)` → `Range<usize>`

**Where (remaining tuple sites):** `src/query/gather.rs` lines 86, 109, 116
(`RangesBundle` `dense_range` / `dense_snp_range` / `dense_indel_range`), 175
(local), 398-401 (borrowed `*_range` slice fields in the per-query params
struct), 479 & 481 (`dense_*_range_out` locals); `src/query/oracle.rs` 95, 97.
`gather.rs` already uses `Range<usize>` at 128-129, 243-256, 279-285, etc., so
today the module holds **both** representations — the inconsistency this item
removes.

**After:** convert the listed `(usize, usize)` fields/locals to `Range<usize>`
(already imported at `gather.rs:7`). This enables `&buf[r.clone()]`, `r.len()`,
and iteration directly. Update construction sites from `(a, b)` to `a..b` and
field reads from `.0`/`.1` to `.start`/`.end`. Keep `oracle.rs` (test/oracle
path) consistent with the main path.

**Constraint:** any of these fields that is serialized or crosses the PyO3 FFI
boundary as a tuple must keep its wire representation — convert only the
in-Rust-memory type, mapping to/from tuples at the FFI seam if one exists.
Verify against `py_query_*.rs` before changing a field that appears in a
returned dict.

### B2 · `EnumKey` trait + generic `EnumMap<K, T>`

**Where:** `src/dense.rs` (`DenseClass`, `DenseMap<T>`, `DENSE_REGISTRY`) and
`src/streams.rs` (`StreamTag`, `StreamMap<T>`, `REGISTRY`).

`DenseMap<T>` and `StreamMap<T>` are the same array-backed fixed-size map
(`from_fn`, `get`, `get_mut`, `iter`, `iter_mut`, `into_iter_tagged`) differing
only in the key enum. `DenseClass` and `StreamTag` each hand-roll identical
`const COUNT`, `const ALL`, and `fn index()`. ~150 lines kept in lockstep by hand.

**After:**
```rust
pub trait EnumKey: Copy {
    const COUNT: usize;
    const ALL: [Self; Self::COUNT];
    fn index(self) -> usize;
}

pub struct EnumMap<K: EnumKey, T> { slots: [T; K::COUNT] }
// from_fn / get / get_mut / iter / iter_mut / into_iter_tagged, once.
```
- Implement `EnumKey` for `DenseClass` and `StreamTag`. A small declarative
  macro (or hand-written impls — only two of them) generates
  `COUNT`/`ALL`/`index`; prefer the macro only if it reads more clearly than two
  short impls. **Do not** reach for a proc-macro/derive crate for two enums.
- Delete the two duplicated map `impl` blocks.
- Provide `pub type DenseMap<T> = EnumMap<DenseClass, T>;` and
  `pub type StreamMap<T> = EnumMap<StreamTag, T>;` so **all existing call sites
  and the module's unit tests are untouched**.
- `EnumMap` needs `slots: [T; K::COUNT]` — const-generic-via-assoc-const arrays
  require the array length to be nameable. If the compiler rejects `[T; K::COUNT]`
  on the stable toolchain pinned by this repo, fall back to keeping `COUNT` as a
  free const generic param (`EnumMap<K, T, const N: usize>`) hidden behind the
  two type aliases, or keep the two thin map structs but factor the shared body
  through a macro. **Pick whichever compiles on the repo's pinned Rust with no
  feature flags; verify before finalizing.**
- Keep both enums and both registry constants (`DENSE_REGISTRY`, `REGISTRY`) as-is.

Behavior-preserving; the array-backed O(1) no-hash access is unchanged.

### B3 · `StreamTag::class() -> Class`

**Where:** `src/streams.rs` (`StreamTag`), bridging to `src/cost_model.rs`
(`Class`). `DenseClass` already has `cost_class() -> Class` (`dense.rs:28-33`);
`StreamTag` has no such bridge, so its SNP/indel mapping lives in convention.

**After:** add
```rust
impl StreamTag {
    pub fn class(self) -> Class {
        match self {
            StreamTag::VarKeySnp => Class::Snp,
            StreamTag::VarKeyIndel => Class::Indel,
        }
    }
}
```
Name it `class()` per the audit (`DenseClass`'s is historically `cost_class()`;
leave that name alone to avoid churn). This makes `cost_model::Class` the single
canonical variant axis that both representation enums bridge to — setting up, but
not forcing, an eventual unified `VariantClass` (out of scope here).

### B4 · Name the svar2-codec bit-shift magic numbers

**Where:** `svar2-codec/src/lib.rs` — the payload-layout literals appear raw in
~6 functions: `<< 25` (`snp_code_to_key:46`, `encode_snp:181`,
`swar_reduce_portable:142` shift init), `<< 27` (`encode_alt_inline:204`
ilen field), `<< 11` (`pext_reduce:121`, = `25 - 14`), `>> 5`
(`pext_reduce:129`, = `30 - 25`), `25 - (i*2)` (`decode_alt_inline:215`).
External users: `src/rvk.rs` and `src/query/gather.rs:719` (`1 << 25`).

The crate is billed as the "single source of truth" for the layout, yet the
offsets are literals; the derivations live only in comments.

**After:** promote the layout offsets to named `pub const`s in the crate, and
derive the PEXT/SWAR shifts from them so the layout is stated once:
```rust
/// Bit position of base[0]'s 2-bit code in the payload ([26:25]).
pub const PAYLOAD_TOP_SHIFT: u32 = 25;
/// Bit position of the 5-bit ilen field ([31:27]).
pub const ILEN_SHIFT: u32 = 27;
```
- `snp_code_to_key`, `encode_snp`, `swar_reduce_portable`'s initial `shift`, and
  `gather.rs:719` use `PAYLOAD_TOP_SHIFT`.
- `encode_alt_inline` / `decode_alt_inline` ilen field uses `ILEN_SHIFT`.
- `decode_alt_inline`'s `25 - (i*2)` becomes `PAYLOAD_TOP_SHIFT - (i as u32 * 2)`.
- The PEXT constants `<< 11` and `>> 5` are the extraction-field offsets
  `PAYLOAD_TOP_SHIFT - 14` and `30 - PAYLOAD_TOP_SHIFT` — express them that way
  (with the existing `// 25 - 14` / `// 30 - 25` comments retained) so a layout
  change flows from the two named consts. The PEXT-internal `14`/`30` are the
  BMI2 extraction positions, not payload layout; leave a comment but they need
  not be consts.
- Export the consts (`pub`) and have `rvk.rs`/`gather.rs` import them rather than
  re-derive.

**Verification:** the crate's existing proptests (encode/decode round-trip)
must stay green — they are the guard that the const substitution is exact.

---

## Testing

No new behavior ⇒ no new golden values. Both PRs are gated on the **existing**
suites:

- **PR-A:** `pixi run pytest tests/test_mutcat.py tests/test_svar_mutations.py`
  (exercise the sentinel/`Kind` paths and the mutation oracles), plus the svar2
  batch-query tests that touch `overlap_batch`/`read_ranges`/`find_ranges`/
  `gather_ranges`. `pyrefly` (or the repo's configured type checker) is the gate
  that A1/A2/A4's static improvements actually type-check. Full `pixi run test`
  before merge.
- **PR-B:** `pixi run bash -lc 'cargo test --no-default-features'` (the
  `--no-default-features` is required or the pyo3 test binary fails to link with
  `undefined symbol: _Py_Dealloc`), covering the `dense.rs`/`streams.rs` unit
  tests (untouched via the type aliases) and the svar2-codec round-trip
  proptests. Then the Python parity suite (`pixi run test`) to confirm the FFI
  results are byte-identical.

## Out of scope (explicitly)

- Frozen dataclasses / privatization / `._raw` grouping for the svar2 batch
  methods → **SP-6**.
- Unifying `Class`/`DenseClass`/`StreamTag` into one `VariantClass` → deliberately
  left; B3 only adds the missing bridge.
- Splitting `_mutcat.py` into `codebook`/`classify`/`count` and relocating the
  scalar oracles → **SP-7**.
- Any change to on-disk formats, query semantics, or public runtime values.

# SVAR2 — reading INFO/FORMAT fields (spec #2 of 3)

Status: **design, ready for spec review.** This is spec **#2** of a three-part
effort:

1. **Write INFO/FORMAT fields** into SVAR2 at conversion time — **landed**
   (PR #100; [`2026-07-11-svar2-info-format-fields-write-design.md`](2026-07-11-svar2-info-format-fields-write-design.md),
   plus [`2026-07-12-svar2-fields-followup-cleanup-optimization-design.md`](2026-07-12-svar2-fields-followup-cleanup-optimization-design.md)).
2. **(this doc) Read/output** those fields on the SVAR2 query/decode path.
3. **SVAR1 → SVAR2 conversion** —
   ([`2026-07-11-svar1-to-svar2-conversion-design.md`](2026-07-11-svar1-to-svar2-conversion-design.md),
   partial; explicitly blocked on #1 **and** #2). This spec unblocks it.

Date: 2026-07-12

## Problem / motivation

Spec #1 gave SVAR2 the ability to *store* scalar-numeric INFO and FORMAT fields
(`{contig}/fields/{cat}/{name}/{sub}/values.bin`), but nothing reads them back.
There is currently **no field-read path at all** in `src/query/` or
`src/py_query*.rs`.

The driving consumer is GenVarLoader. Its SVAR2 path today has **no field
plumbing whatsoever** — `_svar2_haps.py:174` hardcodes
`available_var_fields = ["alt","ilen","start"]` and `:256` passes
`dosages=None`. Meanwhile its SVAR1 path supplies dosages (and arbitrary
`Number=G` custom fields) as a `Ragged` memmapped **on the genotype offsets**.
Until SVAR2 can serve the same data, the SVAR1→SVAR2 migration (spec #3) would
be a **data-loss migration** — which is why spec #3 is blocked on this one.

## Scope

**In:**
- A public, mmap-backed **`FieldView`** read primitive over the on-disk
  `values.bin` sidecars, mirroring the proven `mutcat` sidecar reader.
- The **one missing piece of provenance** (`vk_src`) that lets a consumer map a
  merged var_key record back to its absolute call index.
- A Python **`decode(fields=…)`** surface returning a `Ragged` whose field
  arrays share the genotype offsets (SVAR1 parity).

**Out (deferred):**
- Any dtype conversion or widening (see §4 — explicitly rejected).
- Fields on the **merged Python `BatchResult`** dict contract
  (`overlap_batch`/`read_ranges`/`gather_ranges`). No consumer needs them; gvl
  uses the readbound/split path and its own merge, and Python users get
  `decode()`. Revisit only on demand.
- gvl-side consumer work (its own PR; see §7).

## Background — the facts that shape this design

### On-disk layout (from spec #1, confirmed in code)

```
{contig}/fields/{info|format}/{name}/{var_key_snp|var_key_indel|dense_snp|dense_indel}/values.bin
```

| sub-stream | INFO | FORMAT |
| --- | --- | --- |
| `var_key_snp`, `var_key_indel` | 1 value / **call** (duplicated per carrier) | 1 value / **call** (sample implicit in the CSR column) |
| `dense_snp`, `dense_indel` | 1 value / **dense variant** | `n_dense_variants × n_samples`, variant-major: `idx = dense_row * n_samples + sample` |

Two gotchas the reader **must** honour:

- **Element width is not in the file.** Values are staged as 4-byte `i32`/`f32`
  and rewritten in place to the resolved dtype by `field_finalize::rewrite_file`
  (`src/field_finalize.rs:394-430`). Width comes **only** from `meta.json`'s
  `fields[].dtype` (`src/meta.rs:21-39`).
- **Dense FORMAT is indexed by the *original cohort* sample index**, not the
  selected slot. A sample-subset query must use `HapRanges.orig_samples` /
  `RangesBundle.sample_cols`.

Missing values are `i*::MIN` / `u*::MAX` / `NaN` (`field_finalize.rs:437-495`),
and a sentinel exists **only** when the field's `default` is `None`.

### Precedent: the mutcat sidecar reader

`MutcatView` (`src/mutcat/sidecar.rs:38-72`) is exactly the shape we need: an
`Option<Mmap>` per sub-stream with `code_at(i)`, opened by `open_sidecar`.
`mutcat::count::count_column_into` (`src/mutcat/count.rs:137-198`) already
indexes it by the same two kinds of index this spec needs — the absolute var_key
call index (`abs_i = o0 + i`) and the per-class dense variant row (`dcol`).
`FieldView` mirrors it.

### Where provenance survives — and where it doesn't

| channel | provenance after gather | why |
| --- | --- | --- |
| `dense_snp` / `dense_indel` | **free** | Each per-region window is a *contiguous copy* of the on-disk slice, and `dense_*_range[q]` is retained. Absolute dense row = pure arithmetic. |
| `vk` | **lost** | Per hap, var_key snp+indel are **merged by position** into one flat `vk` (`gather.rs:162`, and the hand-inlined twin at `gather.rs:586-597`). `KeyRef` is only `(position: u32, key: u32)` — 8 bytes, no record index. |

Worse, the var_key runs are built by **filtering** (`if qs < v_end`), so the
absolute index cannot be recovered from a record's position within the run
afterwards — it must be captured **as records are kept**.

This holds for **both** `BatchResult` (merged) and `BatchResultSplit`
(readbound) — `BatchResultSplit.vk` is also a merged snp+indel channel
(`gather.rs:103-105`). The dense side of the split path is the only free part.

### What gvl actually needs

gvl consumes genoray as a **Rust Cargo path-dep** and never asks for a
materialized decode. It takes `BatchResultSplit` and runs its own streaming
3-way merge over `(vk, dense_snp, dense_indel)`
(`GenVarLoader/src/svar2/mod.rs:330-371`), decoding each key inline. **At the
moment it emits a record it already holds the source channel and index**
(`i_vk`, `i_sn`, `i_in`).

That is the load-bearing observation: gvl does **not** want per-decoded-record
field arrays. It wants to look a value up by the channel index it already has.
So genoray does not need to gather, materialize, or even *know about* any
particular field.

## Design

### 1. `FieldView` — the read primitive (public)

New `src/query/field.rs`, mirroring `mutcat/sidecar.rs`:

```rust
pub struct FieldView {
    values: Option<Mmap>,   // {contig}/fields/{cat}/{name}/{sub}/values.bin
    dtype: StorageDtype,    // from meta.json — NOT from the file
    n_samples: usize,       // cohort width, for dense FORMAT striding
}

impl FieldView {
    /// var_key call `i`, or dense INFO variant row `i`.
    pub fn value_at(&self, i: usize) -> FieldValue;
    /// dense FORMAT: `dense_row * n_samples + orig_sample`.
    /// `orig_sample` is the ORIGINAL cohort index, never the selected slot.
    pub fn format_at(&self, dense_row: usize, orig_sample: usize) -> FieldValue;
    /// Zero-copy typed slice for bulk consumers (bytemuck), dtype-checked.
    pub fn as_slice<T: Pod>(&self) -> Option<&[T]>;
    pub fn dtype(&self) -> StorageDtype;
}
```

Opened on demand from a `ContigReader` (`open_field(cat, name, sub)`), so
genoray never carries a field selection. `half = "2"` is already a dependency,
so `f16` needs no new deps.

**This type is public and crosses the genoray↔gvl seam deliberately.** Both
packages are first-party; a lazy `FieldView` lookup beats materializing
`n_selected_samples × dense_len` values per query, and it makes the cost of an
additional field **zero** on the genoray side.

### 2. `vk_src` — the single new piece of state

`BatchResultSplit` (and `BatchResult`) gain one array parallel to `vk`:

```rust
/// Absolute var_key call index per `vk` record.
/// bit 31 = sub-stream tag (0 = snp, 1 = indel); bits 0..=30 = call index.
pub vk_src: Vec<u32>,
```

Captured with `.enumerate()` at the filtered push sites (`gather.rs:139-160`,
`:546-561`, `reader.rs:143`) and merged in lockstep with the keys.

**Packing + the assert.** Bit 31 caps var_key calls per contig at 2³¹ ≈ 2.1e9.
Real stores are far below this, but a silent overflow would produce *silently
wrong field values*, which is the worst possible failure mode. So the packing
helper carries a hard assert:

```rust
#[inline]
fn pack_vk_src(is_indel: bool, call_idx: usize) -> u32 {
    assert!(call_idx < (1 << 31), "var_key call index {call_idx} exceeds the 2^31 vk_src ceiling");
    (call_idx as u32) | ((is_indel as u32) << 31)
}
```

The assert also fires at write/merge time (where the sub-stream call count is
known) so an over-large store fails at conversion, not at query.

### 3. One merge, not two

`spine::merge_keys(runs: Vec<Vec<KeyRef>>)` (`spine.rs:67`) generalizes to a
single implementation:

```rust
pub trait Positioned: Copy { fn position(&self) -> u32; }
impl Positioned for KeyRef {}
impl Positioned for (KeyRef, u32) {}   // key + vk_src

pub fn merge_by_position<T: Positioned>(runs: Vec<Vec<T>>) -> Vec<T>;
pub fn merge_keys(runs: Vec<Vec<KeyRef>>) -> Vec<KeyRef> { merge_by_position(runs) }
```

Monomorphization means the **no-fields path stays byte-identical and
zero-cost** — load-bearing, because the hand-inlined twin at `gather.rs:586-597`
carries a long comment justifying its tuning against the shared helper. Both
merge sites must preserve the same stable tie-break (snp before indel on equal
position); the existing "these two must stay identical" comments
(`spine.rs:64-66`, `gather.rs:567-585`) get updated to cover the payload.

This **shrinks** the divergence hazard (one generic merge) rather than doubling
it.

### 4. Dtype: preserved, never converted

The stored dtype (`bool`, `i8`…`u32`, `f16`, `f32`) is carried through to the
consumer unchanged. **No widening, no `as_f32`, no sentinel translation.**

An earlier draft proposed widening to `f32` at the gvl seam to feed gvl's
`dosage` slot. That was wrong: gvl's containers are **already generic over
arbitrary named fields of arbitrary numpy scalar dtype**:

- `RaggedVariants.__init__(alt, start, ref, ilen, dosage, **fields)`
  (`_rag_variants.py:210-232`) — `dosage` is a dedicated *kwarg* but lands in the
  same dict as `**fields`; the annotation is plain `Ragged`, with **no dtype
  constraint**. `_share_offsets` branches only on string-vs-numeric and never
  inspects `.data.dtype`.
- `_FlatVariants.fields: dict[str, Any]` (`_flat_variants.py:341`) is generic.
- SVAR1's `Number=G` custom fields already flow end-to-end **at their own on-disk
  dtype** (`_haps.py:444-452`), never coerced.

Widening would therefore be a lossy solution to a non-problem (and would lose
precision for `i32` above 2²⁴). Missing values stay exactly as written —
`i*::MIN` / `u*::MAX` / `NaN`, present only when `default` is `None` — and are
documented, not translated.

### 5. Python surface — `decode(fields=…)`

SVAR1 parity (`SparseVar`'s `fields=` / `with_fields` / `available_fields`):

```python
sv = SparseVar2(path, fields=["DS"])       # or sv.with_fields(["DS"])
sv.available_fields                        # {"DS": ("format", dtype("float32")), ...}
rag = sv.decode("chr1", regions)
# rag: pos, ilen, allele, DS — all sharing ONE offsets object, shape (R, S, P, None)
```

- **INFO is duplicated per carrier entry** — that is already how it is stored in
  var_key, and it keeps the decoded `Ragged` uniform.
- **Non-carrier FORMAT loss is invisible here.** `decode` only ever emits carrier
  records, so spec #1's Option-A limitation cannot be observed on this surface.
- **Name collisions**: INFO `DP` and FORMAT `DP` are already namespaced on disk.
  The canonical key is the **bare name when unique across categories**, and
  bcftools-style **`INFO/DP` / `FORMAT/DP`** when not. `available_fields` always
  reports canonical keys.
- Arrays come back in the **stored dtype**.

### 6. What does *not* change

- No new field arrays on `BatchResult`/`BatchResultSplit` (only `vk_src`).
- No field selection state on `ContigReader`.
- The `overlap_batch`/`read_ranges`/`gather_ranges` Python dict contract gains
  `vk_src` and nothing else.
- Adding a 2nd or 10th field costs genoray **nothing** at query time.

### 7. Dependent (gvl-side, not this spec)

Tracked here so spec #3 can sequence, but implemented in GenVarLoader's own PR:

- `_svar2_haps.py:174` / `:256` — wire `available_var_fields` / `var_field_data`
  for SVAR2, and push a value per decoded record inside
  `decode_variants_from_split` (`GenVarLoader/src/svar2/mod.rs:282`, at the
  `pos`/`ilen` push site `:367-368`).
- **Perf note:** gvl's gather/compact kernels export only `i32`/`f32`
  monomorphizations of already-generic Rust cores, with a **dtype-preserving
  numpy-loop fallback**. SVAR2 auto-narrows integers to the smallest lossless
  width, so `i8`/`i16` fields would land on the slow fallback path. Correct, but
  quietly slow at exactly the widths SVAR2 prefers — worth exporting the extra
  monomorphizations.
- **Pre-existing bug (SVAR1-side):** `_impl.py:1443-1445` looks a non-builtin
  field up in `variants.info`, but custom FORMAT fields are deliberately excluded
  from `info` (`_haps.py:445`) → `KeyError` in memory estimation. Needs a
  `var_field_data` branch.

## Testing strategy

- **Rust unit:** `FieldView` element access across every dtype (incl. `f16`,
  `bool`) and both index kinds; dense FORMAT striding against a **subset**
  query, asserting it uses the *original* cohort sample index (this is the
  easiest thing in the design to get wrong).
- **Rust unit:** `pack_vk_src` round-trip; the 2³¹ assert fires.
- **Rust unit:** `merge_by_position` preserves the stable snp-before-indel
  tie-break for both `KeyRef` and `(KeyRef, u32)`; `vk_src` stays aligned to `vk`
  through a merge with **filtered** runs (the case that motivates `.enumerate()`).
- **Golden/oracle:** existing readbound-vs-union oracle
  (`oracle::gather_ranges_readbound`, `tests/test_readbound_gather.rs`) extends
  to assert `vk_src` agreement between the two paths.
- **e2e (Python):** convert a fixture with `from_vcf(info_fields=…,
  format_fields=…)`, then `decode(fields=…)` and check values against the source
  VCF at every carrier; missing → `default`/sentinel; INFO duplicated per
  carrier; stored dtype preserved; `INFO/DP` vs `FORMAT/DP` disambiguation.
- **Regression:** a no-fields `decode`/gather must be byte-identical to today's
  output (guards the zero-cost claim).

## Documentation / housekeeping

- `skills/genoray-api/SKILL.md` — **mandatory** per `CLAUDE.md`'s public-API
  rule: `SparseVar2(fields=…)`, `with_fields`, `available_fields`,
  `decode(fields=…)`, the canonical field-key convention, dtype/missing
  semantics.
- `CHANGELOG.md` — entry under `## Unreleased` (`feat:`).
- Reconcile the SVAR2 roadmap/data-model docs with the read path.
- Unblock and finish [`2026-07-11-svar1-to-svar2-conversion-design.md`](2026-07-11-svar1-to-svar2-conversion-design.md).

## Milestones

1. **`FieldView` (public) + `vk_src` + `merge_by_position`.** Unblocks gvl to
   read any field lazily, and unblocks spec #3.
2. **Python `decode(fields=…)`** — `Ragged` on shared offsets, stored dtype,
   canonical field keys.

## Rejected alternatives

- **Materialize per-channel field arrays on `BatchResultSplit`.** Costs
  `n_selected_samples × dense_len` per query for dense FORMAT, scales with field
  count, and buys nothing: gvl already holds the channel index at emit time.
  Rejected in favour of lazy `FieldView` lookup.
- **Widen everything to `f32` at the seam.** Rejected — see §4; gvl's containers
  are already dtype-generic, and widening is lossy above 2²⁴.
- **Grow `KeyRef` to carry provenance.** Would grow it 8→12 bytes in the hottest
  structure in the query path and churn the frozen `vk_pos`/`vk_key` Python
  contract. A parallel `vk_src` keeps `KeyRef` 8 bytes and `Copy`.

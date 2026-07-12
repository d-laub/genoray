# SVAR2 — writing INFO/FORMAT fields (spec #1 of 3)

Status: **design, ready for spec review.** This is spec **#1** of a three-part
effort:

1. **(this doc) Write INFO/FORMAT fields** into SVAR2 at conversion time.
2. **Read/output** those fields on `SparseVar2` (query/decode path) — separate spec.
3. **SVAR1 → SVAR2** conversion — separate spec
   ([`2026-07-11-svar1-to-svar2-conversion-design.md`](2026-07-11-svar1-to-svar2-conversion-design.md),
   partial; unblocks once #1/#2 land).

Date: 2026-07-11

## Problem / motivation

SVAR2 stores genotype **presence** only. It has no way to carry per-variant
(INFO) or per-sample (FORMAT) numeric annotations — dosages, VAF, CCF, DP, GQ,
etc. SVAR1 supported a narrow slice of this (dosages + `Number=G` FORMAT
fields); SVAR2 has nothing. This spec adds a **general, source-agnostic
write mechanism** for scalar-numeric INFO and FORMAT fields, modeled on the
existing mutcat sidecar. Dosage is the first concrete consumer (needed by spec
#3's SVAR1→SVAR2 migration), but the mechanism is not dosage-specific.

## Scope

**In:**
- Scalar-numeric **INFO** (per-variant) and **FORMAT** (per-call) fields.
- `Integer` (logically `i32`), `Float` (`f32`), and `Flag` (`bool`, INFO-only)
  header types; `Number=1` and biallelic-split `Number=A` arity.
- Per-field storage-dtype selection with lossless integer auto-narrowing and an
  opt-in lossy `f16`, both range-/type-validated at conversion time.
- Extraction in `from_vcf` (Rust/htslib) and the storage/plumbing/merge layer.
  The layer is source-agnostic so spec #3 can feed it from SVAR1 arrays.

**Out (deferred):**
- The **read/output** path (spec #2).
- Multi-valued arities (`Number=R`, `Number=G`, `Number=.`) and
  String/Character fields (need ragged offsets / a byte-bank — future).
- The **independent/lossless FORMAT stream** (see §3, Option B) — deferred to
  avoid downstream ripple; v1 is genotype-aligned only.

## Background

- **Sidecar precedent (mutcat)**: `src/mutcat/sidecar.rs` writes per-contig,
  per-sub-stream SoA arrays positionally aligned to each sub-stream's records;
  `src/layout.rs` owns all paths. Fields follow the same shape.
- **Pipeline stages** (`src/orchestrator.rs`, `src/types.rs`): reader →
  `DenseChunk (V,S,P)` → `dense2sparse_vk` (cost-routes each variant to var_key
  or dense) → `SparseChunk { streams: var_key…, dense: DenseSubChunk… }` →
  `merge_mini_sc` → writer. **Unlike mutcat, fields cannot be a post-pass** —
  their values exist only in the source and must ride this pipeline to stay
  aligned to the finalized records.
- **var_key vs dense** (`src/cost_model.rs`): var_key stores per **call**
  (sample-major: `SparseSubStream` with `call_positions` + per-hap
  `sample_lengths`); dense stores a per-**variant** table + a hap-major 1-bit
  genotype matrix. The cost model routes by carrier count `x`.

## Design

### 1. Public API

`from_vcf` gains two parameters; a bare `str` uses inferred defaults, a config
object overrides them:

```python
@dataclass(frozen=True)
class InfoField:
    name: str
    dtype: FieldDtype | None = None   # None → auto (see §2)
    default: float | int | None = None  # missing-fill value

@dataclass(frozen=True)
class FormatField:
    name: str
    dtype: FieldDtype | None = None
    default: float | int | None = None  # missing-fill; genotype-aligned (§3)

FieldDtype = Literal["bool","i8","u8","i16","u16","i32","u32","f16","f32"]

@classmethod
def from_vcf(cls, ...,
    info_fields: Sequence[str | InfoField] | None = None,
    format_fields: Sequence[str | FormatField] | None = None,
) -> int: ...
```

Separate INFO/FORMAT lists (a name may exist in both). `default`, when set, is
the value written for VCF-missing entries. (For FORMAT under the
genotype-aligned model in §3, `default` is purely the missing-fill — it does
**not** trigger omission in v1, since Option B is deferred.)

### 2. Field typing, dtype selection, validation

Storage dtype is resolved per field against the VCF header **and** the observed
data:

- **`dtype=None` (default):**
  - Integer/Flag → **losslessly auto-narrowed**: chunks carry `i32`; a
    **global finalize pass** (after all per-contig merges) observes the
    field's global `[min,max]` across all contigs (and whether any missing
    occurred) and writes the smallest width that fits the range plus a
    reserved missing sentinel; per-contig storage is uniform because dtype is
    global in `meta.json`. Integer narrowing is lossless, so it is safe to
    automate.
  - Float → **`f32`** (the source precision; never auto-downcast to `f16`).
- **Explicit `dtype`:** validated on two axes at conversion time —
  - *Header-type compat:* Integer → any int/`bool` width; Float → `f16`/`f32`;
    Flag → `bool`/`u8`. Reject Float→int, int→Float, etc.
  - *Observed-range compat:* every value is already scanned during extraction,
    so overflow of the requested width (or a nonzero fraction stored to an int)
    is an error (loud, not silent).
- **`f16`** is the only lossy option and must be requested explicitly; overflow
  of the `f16` range (~65504) errors.

**Why not read BCF's adaptive int width?** BCF's int8/16/32 descriptor is
**per record**, not per field, and rust-htslib's safe API decodes into a
requested width rather than exposing the descriptor. It cannot yield a single
columnar width. Observing the global range during our existing extraction scan
gives the same "smallest int" result, authoritatively, at merge time.

**Missing values:** map to the field's `default` if set, else to the width's
reserved sentinel — `NaN` for `f16`/`f32`, `INT*_MIN` (per chosen width) for
ints. Flag never missing (absent ⟺ `false`).

### 3. Storage model — genotype-aligned (Option A)

**Chosen: no independent field positions in either representation** — the field
reuses whatever structure the genotype rep already has. This is what
"genotype-aligned" means here, and it differs by representation:

- **var_key:** FORMAT values ride the genotype *calls* — one value per call,
  the per-sample value broadcast across that sample's carrier haplotypes (exactly
  SVAR1's `broadcast(dosages)[genos==1]`). Array is parallel to `call_positions`,
  no new positions.
- **dense:** FORMAT is stored as a **full per-sample column** (`n_samples`
  values per dense variant), O(1)-indexable by sample with no popcount/offsets,
  matching how the dense genotype already stores a full `np`-bit "everyone"
  mask. Non-carrier slots hold `default`, keeping read semantics uniform with
  var_key. No positions needed — the column is dense.

Both preserve the strictly sample-major layout and neither introduces a separate
field position stream.

Granularity, per representation:

| | var_key (per call, sample-major) | dense |
| --- | --- | --- |
| **INFO** (per-variant) | value duplicated per call → `width·x` | 1 value / variant → `width` |
| **FORMAT** (per-call) | 1 value / call → `width·n_calls` (`= width·x`) | full per-sample column → `width·n_samples` |

**Deferred — Option B (independent/lossless FORMAT stream):** storing a FORMAT
field as its *own* sample-major sparse sub-stream (per-hap positions + values,
keyed on the field's non-`default` set rather than the carrier set) would be
lossless for decoupled fields (e.g. imputed dosage nonzero at ref-genotype
samples). It ripples through routing, merge, decode, and query, so it is
**out of scope for v1**. Consequence documented for users: FORMAT values at
non-carrier samples are dropped.

### 4. Cost model

Generalize the existing scalar `sidecar_bits` in `choose_representation`
(`src/cost_model.rs`) to per-representation field sums:

```
var_key_bits = x·(POS+key) + Σ_info width_i·x + Σ_fmt width_f·x
dense_bits   = POS+key+np  + Σ_info width_i   + Σ_fmt width_f·n_samples
```

- **INFO shifts the crossover toward dense** (`width_i·x` in var_key vs
  `width_i` once in dense — real amortization).
- **FORMAT also shifts toward dense for high-carrier variants** — var_key pays
  `width_f·x` (per carrier call) while dense pays `width_f·n_samples` (full
  column). It does **not** cancel: dense is chosen when `x` is large (up to
  `np ≥ n_samples`), so FORMAT reinforces the genotype routing rather than being
  neutral to it.

So both an `info_bits` (per-variant amortizable) and a `format_bits`
(`width·x` var_key / `width·n_samples` dense) term feed `choose_representation`,
threaded through the pipeline as the existing sidecar-bits knob generalizes.

### 5. On-disk layout + `meta.json`

Under each contig, mirroring mutcat (`src/layout.rs` owns the paths):

```
{contig}/fields/{name}/var_key_snp/values.bin     # 1 / call   (INFO dup'd; FORMAT per call)
{contig}/fields/{name}/var_key_indel/values.bin
{contig}/fields/{name}/dense_snp/values.bin        # INFO: 1 / variant; FORMAT: full n_samples column / variant
{contig}/fields/{name}/dense_indel/values.bin
```

`values.bin` is a raw little-endian array in the resolved storage dtype, its
length and alignment determined by the sub-stream and category: **var_key** →
one value per call (INFO duplicated; FORMAT per carrier call); **dense INFO** →
one value per dense variant; **dense FORMAT** → an `n_samples`-long per-sample
column per dense variant (`n_samples · n_dense_variants` total, sample-indexed,
non-carrier slots = `default`). No offsets are needed for any scalar field —
each is a fixed stride from `n_samples`/`n_dense_variants`/call counts already
in the layout.

`meta.json` gains a `fields` array (one entry per stored field), consumed by
spec #2's reader:

```json
"fields": [
  {"name": "DS", "category": "format", "dtype": "f32", "default": 0.0},
  {"name": "AF", "category": "info",   "dtype": "f16", "default": null}
]
```

### 6. Reader extraction (`from_vcf`, Rust/htslib)

The VCF reader (`src/vcf_reader.rs`) reads the declared fields per record:
- **INFO:** one scalar per variant. For biallelic-split `Number=A`, take the
  split allele's element; `Number=1` is the scalar; Flag → bool presence.
- **FORMAT:** the per-sample vector; `Number=1` → one scalar per sample, later
  broadcast to that sample's carrier haps at routing. Missing → `default`/sentinel.
- The header supplies `Type`/`Number` for validation (§2); a field absent from
  the header, or non-scalar arity, errors before conversion starts.

### 7. Pipeline plumbing (the bulk of the work)

- **`DenseChunk`** (`src/types.rs`) gains typed field columns: INFO per-variant
  (`len V`) and FORMAT per-sample-per-variant (`len V·n_samples`, source is
  per-sample) for each requested field. Type-erased byte columns tagged by dtype.
- **`dense2sparse_vk`** routes field values alongside genotypes in the existing
  passes: on a var_key call push the call's field value(s); on a dense variant
  push INFO once (per variant) and FORMAT as the full `n_samples`-long per-sample
  column (carrier samples' values, `default` elsewhere). Arrays stay aligned by
  construction.
- **`SparseSubStream` / `DenseSubChunk`** gain parallel field arrays; the
  **chunked writer** writes per-chunk field files; **`merge_mini_sc`**
  generalizes to move field columns with their records (and performs the
  integer auto-narrowing in §2, since it is the stage that sees the whole
  contig).

### 8. Memory

htslib hands FORMAT out densely per record, so a per-chunk `V·n_samples·width`
block per FORMAT field is materialized transiently (≈100 MB for
`chunk_size=25k`, `n_samples=1000`, `f32`). Bounded but real: document it and
auto-shrink `chunk_size` when `format_fields` is non-empty (or expose a knob).
INFO adds only `V·width` per field — negligible.

## Testing strategy

- **Rust unit tests:** dtype resolution + validation (header-compat matrix,
  range overflow errors, `f16` overflow, auto-narrow width selection incl.
  sentinel reservation); routing unchanged by FORMAT bits, shifted by INFO bits
  (`choose_representation` tests); field-array alignment through
  `dense2sparse_vk` for mixed var_key/dense routing.
- **e2e (`tests/test_e2e.rs` + a Python test):** `from_vcf(info_fields=…,
  format_fields=…)` on a fixture writes `fields/…/values.bin` with the expected
  lengths, dtypes, and values; missing → `default`/sentinel; FORMAT values match
  the source at carriers and are absent at non-carriers (documented Option-A loss).
- **Merge/auto-narrow:** integer field with values spanning multiple chunks
  narrows to the correct global width; a wider-than-`u8` value forces `i16`; etc.
- Read-side round-trip is deferred to spec #2's tests.

## Documentation / housekeeping

- `skills/genoray-api/SKILL.md` — new `InfoField`/`FormatField` public types and
  the `from_vcf(info_fields=, format_fields=)` kwargs (mandatory per the
  public-API rule in `CLAUDE.md`).
- CHANGELOG entry under `## Unreleased` (Conventional Commits `feat:`).
- Reconcile the SVAR2 roadmap/data-model docs with the new `fields/` layout and
  `meta.json` schema.

## Open questions (settle in the plan)

- Exact `values.bin` file naming + whether `fields/` sits beside or under
  `mutcat/` conventions (defer to `src/layout.rs`).
- Type-erasure representation for field columns on `DenseChunk`/`SparseChunk`
  (tagged enum vs trait object) — pick the one that keeps the hot routing loop
  monomorphized.
- Whether the missing-sentinel reservation shrinks the auto-narrow range only
  when missing values are actually observed (measure) or always (simpler).
- Interaction with `signatures=True` (both add per-record sidecars; confirm the
  cost-model knobs compose).

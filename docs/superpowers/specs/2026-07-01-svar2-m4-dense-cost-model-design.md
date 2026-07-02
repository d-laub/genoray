# SVAR 2.0 — M4: Dense Representation + Cost-Model Routing — Design

> Roadmap milestone **M4** (see [`svar-2.md`](../../roadmap/svar-2.md)). Adds the
> 1-bit dense genotype matrix and the deterministic per-variant dense/sparse
> routing decision on top of the M1 `var_key` pipeline (PR #79, now merged).
>
> Per the SVAR 2.0 working agreement, the PR that implements this MUST reconcile
> [`data-model.md`](../../roadmap/data-model.md) and
> [`architecture.md`](../../roadmap/architecture.md) with the decisions here — the
> concrete doc edits are enumerated in [§9](#9-documentation-reconciliation).

## Context

After M1, every set genotype bit is stored inline per-call in one of two
`var_key` sub-streams (`snp` 2-bit, `indel` 32-bit), consolidated by the tile
merge. This is cheapest for **low-frequency** variants but wasteful for common
ones: a variant carried by half the cohort pays a `u32` position **plus** a key
for *every* carrier.

M4 adds the **dense** representation — a 1-bit `(sample, ploid, variant)` matrix
plus a per-variant table — and a **cost model** that routes each variant to
whichever representation is cheaper on disk. This turns SVAR2 into the hybrid
store the vision describes.

Two facts (verified against the current code) make M4 tractable:

- **A variant's full allele count is local to one chunk.** `VcfChunkReader`
  reads `chunk_size` *complete* variant rows, so the popcount of a variant's
  `(S,P)` plane — a contiguous bit range in `BitGrid3` at
  `[v·S·P, (v+1)·S·P)` — is exactly its total call count `x`. Routing is a
  per-variant local decision; no cross-chunk aggregation.
- **The key encoding is representation-independent.** The encoder in `rvk.rs`
  produces the same key bits regardless of where they are stored. Dense reuses
  `classify_variant`/`pack_variant` unchanged; only *storage* forks.

## Scope

**In scope (M4):**

1. A pure, deterministic cost model choosing `VarKey` vs `Dense` per variant.
2. Representation-aware routing in the executor transpose.
3. Dense per-chunk output, writer, and a rectangular dense merge producing
   `dense/{snp,indel}/{positions, alleles, genotypes}`.
4. Promoting the long-allele LUT to a **single shared per-contig indel table**
   referenced by both representations.
5. Doc reconciliation ([§9](#9-documentation-reconciliation)).

**Out of scope (owned by other milestones):**

- `max_del.npy` and deletion-overlap handling — **M5** (queries). Dense indel
  deletions will need the same treatment; noted, not built here.
- `meta.json` — **M3**. M4 avoids depending on it: every dense array's shape is
  derivable from `positions` length + global `(n, ploidy)`.
- `.bin`↔`.npy` final-sidecar wire-format cleanup — **M3**.
- The `pointer` representation and its cost-model term — **M11**. The cost model
  is written so adding a third branch is a local change.

## 1. The two-axis model

A variant is placed by two orthogonal axes:

- **class** ∈ {`Snp`, `Indel`} — from `ilen` (unchanged from M1). Fixes key
  width (2-bit vs 32-bit) and LUT applicability.
- **representation** ∈ {`VarKey`, `Dense`} — from the cost model (new).

Cross product = four buckets: `var_key/{snp,indel}`, `dense/{snp,indel}`. Each
variant lives in exactly one.

M1's `StreamTag` enum fuses class into the *per-call* stream machinery
(`SparseSubStream` + tile merge). Dense is a **different storage shape**
(per-variant table + fixed bit matrix), so it does **not** extend
`StreamTag`/`REGISTRY` — that registry stays the "per-call sub-stream" registry
(`var_key` now, `pointer` later). Dense gets a small parallel structure keyed by
class. This keeps each machinery's types honest rather than forcing a matrix into
a `SparseSubStream`.

## 2. Cost model (`cost_model.rs`, new module)

A pure function, no I/O, fully unit-testable:

```rust
pub enum Representation { VarKey, Dense }

pub fn choose_representation(
    class: Class,        // Snp | Indel
    n_samples: usize,
    ploidy: usize,
    x_calls: usize,      // popcount of the variant's plane
) -> Representation
```

Cost = **actual on-disk bytes for one variant** (decision: positions count —
see below). With `np = n_samples · ploidy`:

| representation | byte cost for one variant |
| --- | --- |
| `VarKey` | `x · (POS_BYTES + key_bytes(class))` |
| `Dense`  | `POS_BYTES + key_bytes(class) + ⌈np/8⌉` |

Route to `Dense` iff its cost is **strictly** less than `VarKey`'s; ties → `VarKey`
(deterministic, documented). Constants live in one place and are tunable:

- `POS_BYTES = 4` (u32 position; per-call in var_key, once in dense).
- `key_bytes(Snp) = 0.25` (2 bits), `key_bytes(Indel) = 4`. Represented as an
  exact rational (e.g. bits, integer arithmetic) to avoid float rounding at the
  crossover.

**Decision — positions count toward cost.** The roadmap's provisional model
(`s·x` vs `s + ⌈np/8⌉`) left open whether the per-call `u32` position is part of
`s`. It is: var_key stores a position for *every* call (`SparseSubStream.
call_positions`), and for SNPs that 4 bytes dwarfs the 2-bit key. Excluding it
would misprice the crossover badly (dense would almost never win for SNPs).
Costing actual bytes is the principled ("measure, don't guess") choice.

Worked SNP example (`n=1000`, diploid → `np=2000`, `⌈np/8⌉=250`): dense wins when
`250 + 4.25 < 4.25·x`, i.e. `x ≳ 60` calls (allele frequency ≳ 3%). Sanity-checks
as a reasonable crossover.

**Tests:** monotonicity (once `Dense` wins at `x`, it wins for all `x' > x`);
crossover index matches the closed-form solution of the inequality; class- and
`np`-boundary cases; `x=0` and `x=np` extremes.

## 3. Routing in the executor (`rvk.rs`)

`dense2sparse_vk` already runs a per-variant pre-pass to encode keys once. Extend
it to also compute `x` (plane popcount) and call the cost model, yielding a
per-variant routing record:

```rust
struct Routed { rep: Representation, key: VarKey }  // key carries class (Snp/Indel)
let routed: Vec<Routed> = /* one per variant in the chunk */;
```

Popcount is cheap: the plane is a contiguous bit range, so it is a
word-wise `count_ones` over `BitGrid3.words` (add `BitGrid3::popcount_plane(v)`).

The sample-major transpose loop then routes each **set** bit by `routed[v].rep`:

- `VarKey` → `push_call` into the matching `SparseSubStream` (**exactly as M1**).
- `Dense` → append the bit to this column's dense row for the matching class
  (see §4). var_key streams simply never see dense variants' calls.

Because the loop is already `for s { for p { for v { … } } }`, each hap's dense
bits are produced **in variant order** — i.e. directly in the target
`(sample, ploid, variant)` C-order, one row at a time. No extra transpose.

**Conservation invariant (tested):** for every chunk, `popcount(genos)` ==
(total var_key calls) + (total dense set bits). Every set bit is stored exactly
once, in exactly one representation.

## 4. Dense per-chunk output + writer

Add dense payloads to the chunk the executor emits (extend `SparseChunk`, or a
sibling `DenseChunk` field — naming settled in the plan). Per dense **class**:

- **Genotype block:** a hap-major bit block sized `(S, P, V_dense_chunk)` —
  built directly by the transpose loop above. `V_dense_chunk` is the number of
  dense variants of that class in the chunk (**identical for every hap**).
- **Variant metadata (once per variant, not per hap):** `positions: Vec<u32>`
  and `keys` (2-bit codes for snp, `u32` for indel) in variant order.

**Dense ledger is trivial.** Unlike var_key's ragged per-column counts, every hap
contributes the same `V_dense_chunk` bits, so the merge needs only
`dense_variants_per_chunk[class]` — a scalar per chunk, not a per-column matrix.

The dense writer streams each chunk's genotype block and metadata to per-chunk
temp files under `dense/{snp,indel}/`, mirroring the var_key writer's
`chunk_{id}_*` convention.

## 5. Dense merge (rectangular)

Mirrors the existing tile/`pwrite` merge but is **rectangular** (uniform
per-chunk counts) rather than ragged:

- **Genotypes:** for each hap (parallelizable across haps, the natural tile
  unit), **bit-concatenate** its per-chunk rows in chunk order into that hap's
  slice of `genotypes.bin`. Chunk boundaries are bit-aligned, not byte-aligned
  (`V_dense_chunk` need not be a multiple of 8), so concatenation is bitwise —
  a small bit-append routine (or a shared `BitWriter`). Output is raw
  bit-packed `⌈np · V_dense/8⌉` bytes; shape `(n, p, V_dense)` is **derivable**
  from `len(positions)` + global `(n, ploidy)`, so no shape sidecar is needed.
- **Variant table:** concatenate `positions` and `keys` across chunks in chunk
  order → `positions` + `alleles`. SNP `alleles` are 2-bit-packed post-merge via
  the existing `pack_snp_keys` (reuse the `post_merge` hook shape).

`genotypes.bin` is a raw bit-packed matrix (**not** an `.npy` bool array, which
would be 8× larger and violate the "1-bit" requirement).

## 6. Shared long-allele LUT

**Decision — one shared per-contig indel LUT.** Both `var_key/indel` and
`dense/indel` spilled alleles reference a single table. This keeps the executor's
**single `LongAlleleTableWriter`** unchanged (it is already
representation-agnostic — it only hands out monotonic 31-bit row indices) and
makes a spilled key's meaning representation-portable, consistent with the
encoding seam.

Move the LUT from `var_key/indel/` to a contig-level home shared by both indel
sub-streams, e.g. `{contig}/indel/long_alleles.{bin,offsets}` (exact path fixed
in `layout.rs`; the plan settles it). Update `ContigPaths` and
`LongAlleleReader::new` to the shared path. SNP streams never spill, so no SNP
LUT exists in either representation.

## 7. On-disk layout additions (`layout.rs`)

`layout.rs` remains the single source of truth. Add dense paths:

```
{contig}/
├── indel/long_alleles.{bin,offsets}     # shared LUT (moved; §6)
├── var_key/{snp,indel}/…                # unchanged (M1)
└── dense/
    ├── snp/   { positions, alleles (2-bit packed), genotypes.bin }
    └── indel/ { positions, alleles (u32),          genotypes.bin }
```

`genotypes.bin` is raw bit-packed. Final sidecar file *extensions* (`.bin` vs
`.npy`) inherit whatever M3 settles; M4 introduces no new naming policy, only new
files.

## 8. Component boundaries

| Unit | Responsibility | Depends on |
| --- | --- | --- |
| `cost_model.rs` | pure `choose_representation` + constants | nothing (leaf) |
| `rvk.rs` (extended) | popcount, route, build var_key + dense chunk payloads | `cost_model`, `BitGrid3`, encoder |
| dense writer | stream per-chunk dense temp files | chunk payload types |
| dense merge | rectangular bit-concat + table concat + SNP pack | `layout`, dense ledger |
| `layout.rs` (extended) | dense + shared-LUT paths | — |
| `nrvk.rs` (touched) | shared-LUT path in `ContigPaths`/reader | `layout` |

Each is understandable and testable in isolation; the cost model in particular
is a pure leaf with exhaustive proptests.

## 9. Documentation reconciliation

The implementing PR updates, in the same PR:

- **`data-model.md` — cost model:** state that `s` **includes** the per-call
  position (`POS_BYTES`), give the concrete per-representation byte costs from
  §2, and mark the "does packed-position count?" open question resolved.
- **`data-model.md` — LUT / on-disk layout:** the indel LUT is a **single shared
  per-contig table**, not per-representation; move it out of
  `var_key/indel/`/`dense/indel/` in the layout tree.
- **`data-model.md` — dense representation:** `genotypes` is a raw bit-packed
  matrix whose shape is derived from `positions` length + `(n, ploidy)`.
- **`architecture.md` — open question "routing granularity":** resolved —
  strictly per variant, decided locally within a chunk from the plane popcount.
- **`svar-2.md`:** flip **M4** to `[~]`/`[x]` as appropriate and update the M4
  status line.

## 10. Testing strategy

- **Cost model:** unit + proptests (monotonicity, crossover, boundaries) — §2.
- **Routing/conservation:** proptest — `popcount(genos)` == var_key calls +
  dense set bits, over random chunks with mixed classes and frequencies.
- **Dense merge:** rectangular analog of the existing merge proptests — the
  reassembled `(S,P,V)` matrix equals the per-hap chunk-order bit-concatenation;
  metadata concatenates in variant order; SNP alleles round-trip through pack.
- **End-to-end:** a synthetic VCF with both rare and common variants routes as
  the cost model predicts; decoding var_key ∪ dense reproduces the input
  genotype matrix bit-for-bit. (Extends the existing e2e suite.)

## 11. Open questions

- **Dense merge tiling unit.** Per-hap is the obvious parallel unit; whether to
  further tile very wide cohorts (millions of haps) to bound RAM like the
  var_key merge's `TILE_RAM_BUDGET_BYTES` is a tuning question deferred until we
  measure.
- **Cost-model constants.** `POS_BYTES` and `key_bytes` are principled but still
  want validation against real conversions; the module centralizes them for easy
  revision (shared with the roadmap's existing cost-model-constants question).
- **`SparseChunk` vs sibling struct.** Whether dense payloads extend
  `SparseChunk` or ride a parallel channel is an implementation-shape call for
  the plan, not a data-model decision.
</content>
</invoke>

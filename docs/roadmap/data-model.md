# SVAR 2.0 Data Model & Rationale

> Supplement to [`svar-2.md`](svar-2.md). Describes how variants and genotypes are
> encoded on disk and why. **Current best approximation — correctable.** The exact
> bit layouts in particular are expected to change during the experimentation phase;
> the architecture is built so they can (see
> [`architecture.md`](architecture.md#the-encoding-agnostic-seam)).

## VariantKey lineage

We use *a form of* the [VariantKey](https://www.biorxiv.org/content/10.1101/473744v3)
encoding, adapted for our access pattern. The original VariantKey packs
`CHROM | POS | REF:ALT` into a single 64-bit key. We diverge in two ways:

- **Position is a sidecar, not part of the key.** Variants are already partitioned by
  contig (one directory per contig) and sorted by position, and positions are stored
  in a parallel `positions` array. So the key does **not** carry CHROM or POS — only
  the information needed to reconstruct the ALT allele relative to the reference.
- **The key encodes `ILEN` + `ALT`.** Given the position and the reference genome, a
  SNP is fully described by its 2-bit ALT base; an indel by its length change (`ILEN =
  len(ALT) − len(REF)`) plus the inserted/changed bases.

This keeps keys small enough to inline next to each call, which is the whole point of
the `var_key` representation.

## Inline variant encoding (`var_key`)

The inline encoding branches into two flavors, and each flavor is stored in its **own
per-call stream** — a 2-bit SNP stream and a 32-bit indel stream (see
[on-disk layout](#on-disk-layout)). A variant is a SNP iff `ref_len == 1 && alt_len ==
1` (equivalently `ILEN == 0` after atomization); everything else is an indel. The
orchestration code treats each stream's key as opaque fixed-width bits (see
[`architecture.md`](architecture.md#the-encoding-agnostic-seam)); only the
encode/decode layer knows the layouts below.

### SNP flavor — 2 bits

A SNP changes one base and `ILEN = 0`. With position and reference known, only the ALT
base is needed, and it always fits in 2 bits — so the SNP stream needs **no LUT** and
its keys pack **4 per byte** (call index `i` → byte `i >> 2`, bit-pair `(i & 3)`,
low-pair-first):

```
2-bit ALT:  A=00  C=01  T=10  G=11
```

This mapping is `(base_ascii >> 1) & 0b11`, a branchless ASCII→2-bit trick (no
lookup, no match) that the encoder relies on; the decoder's inverse table is
`[A, C, T, G]`. Note `T` and `G` are swapped relative to the obvious alphabetical
assignment — the bit values are an implementation detail of the encode/decode layer
and carry no meaning outside it.

### Indel flavor — 32 bits

Indels use a 32-bit key. The **least significant bit (LSB) is a flag**: `0` = inline,
`1` = LUT pointer. Because SNPs live in their own stream, `ILEN = 0` never occurs here;
the inline positive lane is insertion-only (`alt_len ∈ 2..=13`).

```
ILEN > 0 (insertion / length-increasing), inline (LSB = 0):

  bit:  31              27 26                                   1   0
        ┌─────────────────┬──────────────────────────────────────┬───┐
        │  ILEN (5b, ≥0)  │  ALT (26b = 13 × 2-bit bases)         │ 0 │
        └─────────────────┴──────────────────────────────────────┴───┘
        max inline ALT length = 13 nucleotides

ILEN < 0 (deletion), inline (LSB = 0):

  bit:  31                                                       1   0
        ┌──────────────────────────────────────────────────────────┬───┐
        │  ILEN (31b, signed)                                        │ 0 │
        └──────────────────────────────────────────────────────────┴───┘
        (a pure deletion needs no ALT bases beyond the anchor)

Either sign, LUT pointer (LSB = 1):

  bit:  31                                                       1   0
        ┌──────────────────────────────────────────────────────────┬───┐
        │  LUT row index (31b)                                       │ 1 │
        └──────────────────────────────────────────────────────────┴───┘
```

An ALT allele spills to the LUT when it cannot be represented inline — i.e. an
insertion whose ALT exceeds 13 nt, or a deletion whose `ILEN` falls outside the signed
31-bit range. Empirically this is **extremely rare** for short-read NGS, so the LUT
stays tiny.

> **Note on widths.** The SNP flavor is 2 bits and the indel flavor is 32 bits. They do
> **not** share a stream: each is written to a separate fixed-width stream (see
> [on-disk layout](#on-disk-layout)), so no per-call tag bit is needed and the SNP
> stream stays a dense 2-bit bitstream. A query reads both streams and merges them (the
> [sorted union](architecture.md#python-decode-path) the decode path already performs).
> The cost model stays written in terms of *bits per variant info* (`s = POS_BITS +
> key_bits`), which is **class-dependent**: `key_bits = 2` for a SNP, `32` for an
> indel (see [cost model](#dense-vs-sparse-cost-model)).

## Long-allele lookup table (LUT)

The LUT belongs to the **indel stream only** — SNPs always fit in 2 bits and never
spill. It is a struct-of-arrays holding the (indel) alleles that don't fit inline:

- `ILEN` — `i32` per row.
- `ALT` — 2-bit packed DNA, concatenated.
- `offsets` — `u64` array indexing into the packed ALT (row `i` spans
  `offsets[i] .. offsets[i+1]`). `u64` matches the byte-offset type used by the
  reader's `seek`, avoiding a cast on the read path.

The LUT is a **single table shared per contig**, at `{contig}/indel/long_alleles.bin`
(+ `{contig}/indel/long_allele_offsets.npy`) — it lives *outside* the
`var_key`/`dense`/`pointer` representation subdirs and is referenced by **both**
`var_key/indel` and `dense/indel` streams, since a spilled long allele is
representation-portable (the same allele can be looked up regardless of which
representation the carrying variant routed to). A 31-bit key references a row by
index. Because long alleles are rare in short-read data, we do **not** do a full pass
to pre-size the LUT — we stream and append, accepting that the LUT is small.

## Dense representation (`dense`)

When a variant's allele frequency is high enough (per the
[cost model](#dense-vs-sparse-cost-model)), storing per-call data is wasteful and we
switch to a **dense 1-bit genotype matrix**, chosen **strictly per variant** (see
[routing granularity](architecture.md#open-questions)):

- `genotypes` is a **raw bit-packed matrix**, LSB-first, **hap-major**
  `(sample, ploid, variant)` with variant as the fastest-varying axis — i.e. flat bit
  index `h * V_dense + v` where `h` ranges over `sample * ploidy` haplotypes and `v`
  over that class's dense variants. One bit per `(sample, ploid, variant)`: present /
  absent.
- The matrix carries **no shape sidecar** — its shape is derived from `len(positions)`
  (the per-class dense variant count, `V_dense`) together with `(n_samples, ploidy)`
  from `meta.json`.
- The variant info is stored **once per variant** in a variant table alongside the
  matrix — not inline per call. That table is **split by class the same way the
  `var_key` streams are**: a 2-bit SNP variant table (no LUT) and a 32-bit indel
  variant table (spills to the [shared per-contig LUT](#long-allele-lookup-table-lut)),
  each with its own matrix over its own variants. Splitting the matrix's variant axis
  by class is free (same total bits) and shrinks the SNP variant table's `alleles`
  column 4 B → 0.25 B per variant — a large win whenever annotations are sparse and
  `alleles` dominates the row.
- **On-disk filenames.** The dense final files share `var_key`'s wire-format
  convention (see [Open questions](#open-questions) → resolved): `positions.bin` /
  `alleles.bin` / `genotypes.bin` under `{contig}/dense/{snp,indel}/`, all raw
  little-endian `.bin`. The dense representation has no ragged `offsets` sidecar —
  every hap contributes the same per-variant count, so the matrix shape is derived
  from `len(positions)` and `(n_samples, ploidy)` from `meta.json`.

## Dense vs. sparse cost model

For each variant we pick the representation with the smallest on-disk **bit** cost
given its observed number of carrier calls. The model is implemented in
[`cost_model.rs`](../../src/cost_model.rs) as exact integer bit arithmetic — no
floats, no fractional bytes — so the crossover is deterministic and reproducible.
Definitions:

$$
\begin{aligned}
n &\coloneqq \text{number of samples}, & n &\in \mathbb{Z}_{>0} \\
p &\coloneqq \text{ploidy}, & p &\in \mathbb{Z}_{>0} \\
np &\coloneqq \text{number of haplotypes (columns)} \\
x &\coloneqq \text{number of carrier calls}, & x &\in [0,\, np] \\
s &\coloneqq \text{POS\_BITS} + \text{key\_bits(class)} \;\; \text{(bits per variant, class-dependent)}
\end{aligned}
$$

`s` **includes** the per-call `u32` position (`POS_BITS = 32`) — packed-position
bytes count toward the cost, both inline per call (`var_key`) and once per variant
(`dense`). `key_bits(class)` is `2` for a SNP (2-bit ALT code) and `32` for an indel
(inline value or LUT pointer). Concrete per-representation bit costs for one variant:

$$
\underbrace{x \cdot s}_{\text{var\_key}}
\quad\lessgtr\quad
\underbrace{s + np}_{\text{dense}}
$$

i.e. `var_key = x·(32 + key_bits)` (position + key inlined per call) and
`dense = 32 + key_bits + np` (one position + key per variant, plus a 1-bit-per-hap
mask). Route to `Dense` **iff strictly cheaper** (`dense_bits < var_key_bits`); an
exact tie breaks to `VarKey`.

> **Pointer (`PT`) representation not yet modeled here.** The `pointer`
> representation (M11) is not part of the MVP cost-model routing above; when it's
> added, its cost (`s` bytes table row + a pointer per call) needs to be folded into
> the same comparison. The LUT size is deliberately **ignored** — pricing it would
> require a full pass over all variants, and it is empirically negligible for
> short-read data.

## On-disk layout

SVAR2 is a directory. It is split by contig; each contig directory holds up to three
representation subdirectories, populated according to the cost model, plus a single
**shared `indel/` LUT directory** at the contig level. Variants in a contig are
partitioned across `var_key` / `pointer` / `dense` — each variant lives in exactly
one.

Every representation splits its variant/allele storage into a **`snp/` sub-stream**
(2-bit keys, no LUT) and an **`indel/` sub-stream** (32-bit keys, spills to the
shared LUT). The long-allele LUT itself is **not** duplicated per representation —
there is exactly one `{contig}/indel/long_alleles.bin` (+
`long_allele_offsets.npy`), referenced by both `var_key/indel` and `dense/indel`
(and, eventually, `pointer/indel` — M11):

```
svar2/
├── meta.json                       # version, samples, contigs, ploidy, ...
└── {contig}/
    ├── max_del.npy                 # max deletion length per (sample, ploid); bounds overlap search (indel sub-streams only)
    ├── indel/
    │   ├── long_alleles.bin        # shared LUT for indel alleles that don't fit inline
    │   └── long_allele_offsets.npy # row offsets into the packed ALT bytes above
    ├── dense/
    │   ├── snp/
    │   │   ├── positions.bin       # sidecar SNP-variant positions (sorted), u32 LE
    │   │   ├── alleles.bin         # 2-bit packed keys, one per SNP variant (no LUT), u8
    │   │   ├── {field}.npy         # per-variant INFO/FORMAT fields
    │   │   └── genotypes.bin       # 1-bit (sample, ploid, snp_variant) matrix, C-order, u8
    │   └── indel/
    │       ├── positions.bin       # u32 LE
    │       ├── alleles.bin         # 32-bit keys, one per indel variant (points into shared LUT), u32 LE
    │       ├── {field}.npy
    │       └── genotypes.bin       # 1-bit (sample, ploid, indel_variant) matrix, C-order, u8
    ├── pointer/                    # = SVAR 1.0 representation (longer-term, M11)
    │   ├── snp/
    │   │   ├── positions.npy
    │   │   ├── alleles.npy         # 2-bit packed variant table (no LUT)
    │   │   ├── {field}.npy
    │   │   ├── pointers.npy        # u32/u64 pointers into the SNP variant table
    │   │   └── offsets.npy         # per (sample, ploid) ragged offsets into pointers
    │   └── indel/
    │       ├── positions.npy
    │       ├── alleles.npy         # 32-bit variant table (points into shared LUT)
    │       ├── {field}.npy
    │       ├── pointers.npy
    │       └── offsets.npy
    └── var_key/
        ├── snp/
        │   ├── positions.bin       # per-call SNP positions (sorted within each hap), u32 LE
        │   ├── alleles.bin         # 2-bit packed ALT, 4 calls/byte (uint8), no LUT
        │   ├── offsets.npy         # per (sample, ploid) ragged offsets into snp calls, u64
        │   └── {field}.npy         # per-call INFO/FORMAT for SNP calls
        └── indel/
            ├── positions.bin       # per-call indel positions (sorted within each hap), u32 LE
            ├── alleles.bin         # 32-bit keys, one per call (ragged, points into shared LUT), u32 LE
            ├── offsets.npy         # per (sample, ploid) ragged offsets into alleles, u64
            └── {field}.npy
```

`meta.json` carries the format version (so `SparseVar2` can negotiate), sample list,
contig list, and ploidy. The per-stream maximum deletion length needed for overlap
queries lives in a separate `max_del.npy` per contig (see below) rather than in
`meta.json`, because it is a structured, potentially large array (e.g. 1M diploid
samples × 20 contigs).

**Array dtypes are a `format_version` convention, not duplicated in `meta.json`.**
For `format_version = 1`: `positions.bin` is `u32` little-endian; `alleles.bin` is
`u8` (2-bit-packed SNP codes, 4/byte) in `snp/` and `u32` little-endian (inline value
or shared-LUT pointer) in `indel/`; `genotypes.bin` is `u8` (raw 1-bit hap-major
matrix, LSB-first); `offsets.npy` and `long_allele_offsets.npy` are `u64`;
`long_alleles.bin` is `u8`. The bulk parallel-`pwrite`n arrays (`positions` /
`alleles` / `genotypes`) are raw `.bin` — mmap-friendly, no npy-header offset to align
every `pwrite` past — and Python reads them with `np.memmap(path, dtype=…, mode='r')`,
deriving shape from `len(positions)` and `meta.json`. The small one-shot index/metadata
sidecars (`offsets`, `long_allele_offsets`) stay self-describing `.npy`.

A contig query reads both sub-streams of each representation it touches and merges the
results in position order. `max_del.npy` describes the **indel sub-streams only**; SNP
sub-streams span exactly one base, so they need no leftward overlap extension.

## Variant normalization

Conversion normalizes variants inline as they stream through, so the on-disk model is
always normalized:

- **Atomization (M2)** — break complex/MNV records into atomic primitives (SNP /
  anchored INS / anchored DEL), mirroring bcftools `_atomize_allele`.
- **Biallelic split (M2)** — split multi-allelic sites into separate biallelic records,
  remapping genotypes by original ALT index.
- **Left-alignment (M2b, deferred)** — shift indels to their leftmost equivalent
  position. Deferred because it is the only step requiring a reference genome.

This keeps `ILEN`/ALT semantics simple and makes the inline encoding well-defined.
Because atomization spreads atom positions rightward (and, once M2b lands,
left-alignment shifts them leftward), the reader emits atoms through a position-keyed
reorder buffer so each per-`(sample, ploid)` stream stays position-sorted for the
interleaving merge.

## Overlap queries and deletions

A deletion spans more than one reference base, so a variant that *starts* before a
query range can still *overlap* it. Range queries must therefore behave like region
overlap, not point lookup.

Two regions A and B overlap iff (from the VariantKey paper, §8):

```
(A_CHROM = B_CHROM) and (A_STARTPOS < B_ENDPOS) and (A_ENDPOS > B_STARTPOS)
```

Given a list `L` of sorted keys and the maximum region (deletion) length
`L_MAX_REGION_LENGTH` over `L`, find entries overlapping a query region `R`:

1. Binary search on CHROM + STARTPOS only:
   - **Upper bound (UB):** maximal entry with `L_CHROM = R_CHROM` and `L_STARTPOS < R_ENDPOS`.
   - **Lower bound (LB):** minimal entry `< UB` with `L_CHROM = R_CHROM` and
     `L_STARTPOS > (R_STARTPOS − L_MAX_REGION_LENGTH)`.
2. Linear scan between LB and UB for `L_ENDPOS > R_STARTPOS`.

Consequently we **track the maximum deletion length per (contig, sample, ploid)**
stream and store it in a per-contig `max_del.npy` (shape `(sample, ploid)`) — not in
`meta.json`, since it is structured and potentially large (e.g. 1M diploid samples ×
20 contigs). That bounds how far left of the query start a spanning deletion can begin,
so the binary search stays tight. This applies to **indel sub-streams only**: a SNP
spans exactly one base (`ENDPOS = STARTPOS + 1`), so a SNP sub-stream's `max_del` is
identically 0 and its overlap query reduces to a plain half-open range `[LB, UB)` with
no leftward extension. The SVAR 1.0 reader
already does an analogous length-aware scan
(`_find_starts_ends_with_length` in `python/genoray/_svar.py`); SVAR 2.0 generalizes it
across the three representations.

### Implementation status (M5, part 1 — search core)

The **format-independent search core** for this algorithm ships in `src/search.rs`; the
disk-integrated `(range, sample)` query is still pending. What is implemented:

- `SearchTree` — the left-tree static B-tree above, over a sorted-ascending `u32`
  position array (`B = 16` keys/node; `u32::MAX` is the padding sentinel, so stored
  positions must be `< u32::MAX`). Built once, queried many times; exposes
  `lower_bound`/`upper_bound`.
- `overlap_range(tree, v_ends, max_region_length, q_start, q_end) -> (s_idx, e_idx)` —
  the resolver, over 0-based half-open ends (`v_end = v_start + 1` for a SNP,
  `v_start + 1 + d` for a `d`-base deletion). It realizes step 1 as **one tree, not
  two**: `UB = lower_bound(q_end)`, and `LB = lower_bound(q_start.saturating_sub(
  max_region_length))` — the max-deletion bound is applied by shifting the *query* down
  with a saturating subtraction rather than building a second tree over
  `v_starts + max_region_length`. The concrete lower bound is therefore
  `v_start >= q_start − max_region_length` (saturating at 0), which is conservative
  (never misses a spanning deletion); step 2's forward/backward scan over `v_ends` then
  tightens it to the exact first/last truly-overlapping index. An empty result is
  returned as `s_idx == e_idx` — correctly empty on a no-overlap query, unlike SVAR
  1.0's `var_ranges` end-sentinel.

The core depends only on in-memory slices — no on-disk types. **Remaining for full
M5:** producing/consuming `max_del.npy`, the sorted union across the `snp/`+`indel/`
sub-streams (and representations), and the genotype gather that turns an index range
into user-facing calls.

## Format constraints and non-goals

SVAR2 is a **compute-oriented, derived format — not an archival format.**

- **No sample appends.** You cannot add samples to an existing SVAR2 file. Adding
  samples changes per-variant allele frequencies, which changes cost-model decisions
  and can invalidate every LUT and variant table. Re-convert instead.
- **Cheap merge/split by contig (M8).** Contigs are fully independent on disk, so
  merging or splitting SVAR2 files along contig boundaries is a near-trivial file
  operation.
- **Cheap region subsetting (M9, non-MVP).** Subsetting by region introduces no new
  variants and only shrinks variant tables, so it doesn't perturb the cost model.
- **Bulk N-way merge is harder (M12).** A general merge of multiple SVAR2 files can
  change allele frequencies and must rebuild LUTs and variant tables — deferred.

## Open questions

- **SNP/indel width coexistence.** *Resolved: separate streams.* The 2-bit SNP flavor
  and 32-bit indel flavor do **not** share a stream — every representation splits into a
  `snp/` sub-stream (2-bit, no LUT) and an `indel/` sub-stream (32-bit, LUT). SNPs are
  >90% of variants after atomization (no MNPs or compound SNP-indels survive), so a
  dedicated 2-bit stream shrinks the dominant case ~16× on its key/allele bytes at the
  cost of one extra sorted-union at query time (which the decode path already performs
  across representations). *Prior M1 implementation:* a **uniform 32-bit** stream with
  SNPs encoded as `ILEN = 0` and the ALT base in bits `[26:25]`; the split supersedes
  it. The encoder's SNP fast path (`encode_snp`) still applies — it now emits the bare
  2-bit code into the SNP stream instead of shifting it to bits `[26:25]`.
- **`var_key` sidecar wire format.** *Resolved (M3): raw `.bin` for pwritten data
  arrays, `.npy` for one-shot sidecars.* The bulk arrays the tile merge writes via
  concurrent positional `pwrite` (`positions.bin`, `alleles.bin`, and dense
  `genotypes.bin`) are raw little-endian `.bin`: they carry no logical shape in the
  file (2-bit-packed SNP / 1-bit-packed genotype bytes), and wrapping them in `.npy`
  would force a hand-rolled 64-byte-aligned header that every `pwrite` must offset past
  — real fragility for no gain. The small one-shot index/metadata sidecars
  (`offsets.npy`, `long_allele_offsets.npy`) stay self-describing `.npy`. Array dtypes
  are keyed to `format_version` (see [On-disk layout](#on-disk-layout)).
- **Cost-model constants (pointer representation only).** `var_key` vs. `dense` are
  implemented and measured in exact integer bits (see
  [cost model](#dense-vs-sparse-cost-model)); the remaining open constant is the
  pointer width (32 vs. 64 bit) for the not-yet-implemented `pointer` representation
  (M11), which still needs to be folded into the same comparison.
- **`s` for `var_key` / whether packed-position bytes count.** *Resolved:* yes —
  `s = POS_BITS + key_bits(class)` includes the per-call `u32` position, so
  positions count toward the cost on both sides of the comparison. `s` is
  class-dependent (`key_bits = 2` for SNP, `32` for indel), so the routing threshold
  differs by class (SNPs stay in `var_key` up to a higher allele frequency before
  dense wins). See [`cost_model.rs`](../../src/cost_model.rs).

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

The inline encoding branches into two flavors. The orchestration code treats the key
as opaque fixed-width bits (see [`architecture.md`](architecture.md#the-encoding-agnostic-seam));
only the encode/decode layer knows the layout below.

### SNP flavor — 2 bits

A SNP changes one base and `ILEN = 0`. With position and reference known, only the ALT
base is needed:

```
2-bit ALT:  A=00  C=01  G=10  T=11
```

### Indel flavor — 32 bits

Indels (and SNPs, when stored in a 32-bit stream) use a 32-bit key. The **least
significant bit (LSB) is a flag**: `0` = inline, `1` = LUT pointer.

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

> **Note on widths.** The SNP flavor is 2 bits and the indel flavor is 32 bits. How
> these coexist within a single per-call stream (fixed-width vs. tagged) is an
> [open question](#open-questions) — the cost model and the agnostic seam are written
> in terms of *bytes per variant info* (`s`), not a single fixed width.

## Long-allele lookup table (LUT)

The LUT is a struct-of-arrays holding the alleles that don't fit inline:

- `ILEN` — `i32` per row.
- `ALT` — 2-bit packed DNA, concatenated.
- `offsets` — `u32` array indexing into the packed ALT (row `i` spans
  `offsets[i] .. offsets[i+1]`).

On disk this is `long_alleles.bin` (+ its offsets). A 31-bit key references a row by
index. Because long alleles are rare in short-read data, we do **not** do a full pass
to pre-size the LUT — we stream and append, accepting that the LUT is small.

## Dense representation (`dense`)

When a variant's allele frequency is high enough (per the
[cost model](#dense-vs-sparse-cost-model)), storing per-call data is wasteful and we
switch to a **dense 1-bit genotype matrix**:

- Shape `(sample, ploid, variant)`, **C-order** (variant is the fastest-varying axis).
- One bit per `(sample, ploid, variant)`: present / absent.
- The variant info itself still uses the two inline flavors (SNP 2-bit / indel 32-bit),
  stored once per variant in a variant table alongside the matrix — not inline per call.

## Dense vs. sparse cost model

For each variant we pick the representation with the smallest on-disk byte cost given
its observed number of calls. Definitions:

$$
\begin{aligned}
n &\coloneqq \text{number of samples}, & n &\in \mathbb{Z}_{>0} \\
p &\coloneqq \text{ploidy}, & p &\in \mathbb{Z}_{>0} \\
x &\coloneqq \text{number of calls}, & x &\in [0,\, np] \\
a &\coloneqq \tfrac{x}{np} \;\; \text{(allele frequency)} \\
s &\coloneqq \text{bytes per variant info}, & s &\in \mathbb{Z}_{\ge 2}
\end{aligned}
$$

Byte cost of each representation for one variant:

$$
\underbrace{s\,x}_{\text{VK (var\_key)}}
\quad\lessgtr\quad
\underbrace{s + 8x}_{\text{PT (pointer)}}
\quad\lessgtr\quad
\underbrace{s + \lceil np/8 \rceil}_{\text{DN (dense)}}
$$

Interpretation by regime, as allele frequency `a` (hence `x`) grows:

- **VK** costs `s` bytes per call (variant info inlined `x` times) — cheapest at low `a`.
- **PT** costs one `s`-byte table row plus a pointer per call — wins in the middle,
  *and* when `s` is large (many INFO/FORMAT fields) so paying for the table row once
  beats inlining it.
- **DN** costs one `s`-byte table row plus a fixed `⌈np/8⌉`-byte bitmask independent of
  `x` — cheapest at high `a`.

> **This math is provisional and likely out of date.** For example, the `8x` term for
> PT assumes a 64-bit pointer; it can be `4x` (32-bit) when the variant count is small
> enough to index with `u32`. The LUT size is deliberately **ignored** here — pricing
> it would require a full pass over all variants, and it is empirically negligible for
> short-read data. Revisit the constants when we have measurements.

## On-disk layout

SVAR2 is a directory. It is split by contig; each contig directory holds up to three
representation subdirectories, populated according to the cost model. Variants in a
contig are partitioned across `var_key` / `pointer` / `dense` — each variant lives in
exactly one.

```
svar2/
├── meta.json                       # version, samples, contigs, ploidy, ...
└── {contig}/
    ├── max_del.npy                 # max deletion length per (sample, ploid) for this contig; bounds overlap search
    ├── dense/
    │   ├── long_alleles.bin        # LUT for alleles that don't fit inline
    │   ├── positions.npy           # sidecar variant positions (sorted)
    │   ├── alleles.npy             # inline variant keys (variant table)
    │   ├── {field}.npy             # per-variant INFO/FORMAT fields
    │   └── genotypes.npy           # 1-bit (sample, ploid, variant) matrix, C-order
    ├── pointer/                    # = SVAR 1.0 representation (longer-term, M11)
    │   ├── long_alleles.bin
    │   ├── positions.npy
    │   ├── alleles.npy
    │   ├── {field}.npy
    │   ├── pointers.npy            # u32/u64 pointers into the variant table
    │   └── offsets.npy             # per (sample, ploid) ragged offsets into pointers
    └── var_key/
        ├── long_alleles.bin
        ├── positions.npy
        ├── alleles.npy             # inline keys, one per call (ragged)
        ├── offsets.npy             # per (sample, ploid) ragged offsets into alleles
        └── {field}.npy
```

`meta.json` carries the format version (so `SparseVar2` can negotiate), sample list,
contig list, and ploidy. The per-stream maximum deletion length needed for overlap
queries lives in a separate `max_del.npy` per contig (see below) rather than in
`meta.json`, because it is a structured, potentially large array (e.g. 1M diploid
samples × 20 contigs).

## Variant normalization

Conversion normalizes variants inline as they stream through, so the on-disk model is
always normalized:

- **Left-alignment** — shift indels to their leftmost equivalent position.
- **Atomization** — break complex/MNV records into atomic primitives.
- **Biallelic split** — split multi-allelic sites into separate biallelic records.

This keeps `ILEN`/ALT semantics simple and makes the inline encoding well-defined.

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
so the binary search stays tight. The SVAR 1.0 reader
already does an analogous length-aware scan
(`_find_starts_ends_with_length` in `python/genoray/_svar.py`); SVAR 2.0 generalizes it
across the three representations.

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

- **SNP/indel width coexistence.** How do the 2-bit SNP flavor and 32-bit indel flavor
  share a single per-call stream? Options: a uniform 32-bit width (simplest, treats
  every call as the indel flavor), a tagged/variable width, or separate streams. The
  cost model is written in terms of bytes-per-variant-info `s` to stay agnostic, but
  the wire format must pick one.
- **Cost-model constants.** Pointer width (32 vs 64 bit) and the exact `s` per
  representation need measurement; the current inequalities are placeholders.
- **`s` for `var_key`.** When variant info is inlined per call, what exactly counts
  toward `s` (key only, or key + decoded ALT bytes)? Pin this down once the wire format
  is fixed.

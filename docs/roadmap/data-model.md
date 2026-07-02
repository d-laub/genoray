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
> The cost model stays written in terms of *bytes per variant info* (`s`), which is now
> **class-dependent** (`s ≈ 2 bits` for a SNP, `4 B` for an inline indel).

## Long-allele lookup table (LUT)

The LUT belongs to the **indel stream only** — SNPs always fit in 2 bits and never
spill. It is a struct-of-arrays holding the (indel) alleles that don't fit inline:

- `ILEN` — `i32` per row.
- `ALT` — 2-bit packed DNA, concatenated.
- `offsets` — `u64` array indexing into the packed ALT (row `i` spans
  `offsets[i] .. offsets[i+1]`). `u64` matches the byte-offset type used by the
  reader's `seek`, avoiding a cast on the read path.

On disk this is `long_alleles.bin` (+ its offsets). A 31-bit key references a row by
index. Because long alleles are rare in short-read data, we do **not** do a full pass
to pre-size the LUT — we stream and append, accepting that the LUT is small.

## Dense representation (`dense`)

When a variant's allele frequency is high enough (per the
[cost model](#dense-vs-sparse-cost-model)), storing per-call data is wasteful and we
switch to a **dense 1-bit genotype matrix**:

- Shape `(sample, ploid, variant)`, **C-order** (variant is the fastest-varying axis).
- One bit per `(sample, ploid, variant)`: present / absent.
- The variant info is stored **once per variant** in a variant table alongside the
  matrix — not inline per call. That table is **split by class the same way the
  `var_key` streams are**: a 2-bit SNP variant table (no LUT) and a 32-bit indel
  variant table (with LUT), each with its own matrix over its own variants. Splitting
  the matrix's variant axis by class is free (same total bits) and shrinks the SNP
  variant table's `alleles` column 4 B → 0.25 B per variant — a large win whenever
  annotations are sparse and `alleles` dominates the row.

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

Every representation splits its variant/allele storage into a **`snp/` sub-stream**
(2-bit keys, no LUT) and an **`indel/` sub-stream** (32-bit keys, LUT):

```
svar2/
├── meta.json                       # version, samples, contigs, ploidy, ...
└── {contig}/
    ├── max_del.npy                 # max deletion length per (sample, ploid); bounds overlap search (indel sub-streams only)
    ├── dense/
    │   ├── snp/
    │   │   ├── positions.npy       # sidecar SNP-variant positions (sorted)
    │   │   ├── alleles.npy         # 2-bit packed keys, one per SNP variant (no LUT)
    │   │   ├── {field}.npy         # per-variant INFO/FORMAT fields
    │   │   └── genotypes.npy       # 1-bit (sample, ploid, snp_variant) matrix, C-order
    │   └── indel/
    │       ├── long_alleles.bin    # LUT for indel alleles that don't fit inline
    │       ├── positions.npy
    │       ├── alleles.npy         # 32-bit keys, one per indel variant
    │       ├── {field}.npy
    │       └── genotypes.npy       # 1-bit (sample, ploid, indel_variant) matrix, C-order
    ├── pointer/                    # = SVAR 1.0 representation (longer-term, M11)
    │   ├── snp/
    │   │   ├── positions.npy
    │   │   ├── alleles.npy         # 2-bit packed variant table (no LUT)
    │   │   ├── {field}.npy
    │   │   ├── pointers.npy        # u32/u64 pointers into the SNP variant table
    │   │   └── offsets.npy         # per (sample, ploid) ragged offsets into pointers
    │   └── indel/
    │       ├── long_alleles.bin
    │       ├── positions.npy
    │       ├── alleles.npy         # 32-bit variant table
    │       ├── {field}.npy
    │       ├── pointers.npy
    │       └── offsets.npy
    └── var_key/
        ├── snp/
        │   ├── positions.npy       # per-call SNP positions (sorted within each hap)
        │   ├── alleles.npy         # 2-bit packed ALT, 4 calls/byte (uint8), no LUT
        │   ├── offsets.npy         # per (sample, ploid) ragged offsets into snp calls
        │   └── {field}.npy         # per-call INFO/FORMAT for SNP calls
        └── indel/
            ├── long_alleles.bin
            ├── positions.npy       # per-call indel positions (sorted within each hap)
            ├── alleles.npy         # 32-bit keys, one per call (ragged)
            ├── offsets.npy         # per (sample, ploid) ragged offsets into alleles
            └── {field}.npy
```

`meta.json` carries the format version (so `SparseVar2` can negotiate), sample list,
contig list, and ploidy. The per-stream maximum deletion length needed for overlap
queries lives in a separate `max_del.npy` per contig (see below) rather than in
`meta.json`, because it is a structured, potentially large array (e.g. 1M diploid
samples × 20 contigs).

A contig query reads both sub-streams of each representation it touches and merges the
results in position order. `max_del.npy` describes the **indel sub-streams only**; SNP
sub-streams span exactly one base, so they need no leftward overlap extension.

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
so the binary search stays tight. This applies to **indel sub-streams only**: a SNP
spans exactly one base (`ENDPOS = STARTPOS + 1`), so a SNP sub-stream's `max_del` is
identically 0 and its overlap query reduces to a plain half-open range `[LB, UB)` with
no leftward extension. The SVAR 1.0 reader
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
- **`var_key` sidecar wire format.** The merge currently writes positions and keys as
  raw little-endian `.bin` (`final_positions.bin` / `final_keys.bin`, via `bytemuck`)
  and offsets as `final_offsets.npy`, whereas the layout spec above names them `.npy`.
  Settle on one (raw `.bin` is mmap-friendly and avoids the npy header; `.npy` is
  self-describing) and align the names before the decode path (M6) is built.
- **Cost-model constants.** Pointer width (32 vs 64 bit) and the exact `s` per
  representation need measurement; the current inequalities are placeholders.
- **`s` for `var_key`.** When variant info is inlined per call, what exactly counts
  toward `s` (key only, or key + decoded ALT bytes)? With the SNP/indel split, `s` is
  now **class-dependent** — `≈2 bits` for a SNP call, `4 B` for an inline indel — so the
  cost-model routing threshold differs by class (SNPs stay in `var_key` up to a higher
  allele frequency before dense wins). Pin down the exact per-class `s` (including
  whether packed-position bytes count) alongside the cost-model constants.

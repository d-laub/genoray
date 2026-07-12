# SVAR2 mutational signatures (Rust)

Status: approved design, ready for implementation planning
Date: 2026-07-10

## Problem

genoray v1 (`SparseVar`) supports COSMIC mutational-signature workflows via
`_svar/_annotate.py` + `_mutcat/` (numba) + `_signatures.py` (numpy/scipy). SVAR2
(`SparseVar2`, Rust-backed) has none of this. The v1 implementation is
**prohibitively memory-hungry**: it stores a `mutcat` field as one `int16` per
*genotype entry* (parallel to `genos.data`) and `annotate_mutations` materializes
the entire cohort's sparse genotypes in memory to build it.

Port the functionality to SVAR2 with **feature parity** and make it **as fast or
faster while using far less memory**. The signature *refit* (`fit_signatures`,
`cosmic_signatures` — pure numpy/scipy, off the hot path) is reused unchanged.
The Rust work is **classification** and **catalogue counting**.

## Background: what v1 does

1. `classify_variants(index, reference)` → one intrinsic `int16` code per
   *variant* (SBS96 `0–95` / native-DBS78 / ID83), reference-dependent.
2. `build_entry_codes(...)` → broadcasts to one code per *genotype entry* and
   pairs adjacent SNVs carried on the same haplotype into DBS-78 (the 5′ entry
   gets the doublet code, the 3′ entry a `DBS_PARTNER` sentinel).
3. Writes `mutcat.npy` — `int16` per genotype call — and stamps `metadata.json`.
4. `mutation_matrix(kind, count)` counts codes → `(samples × channels)`;
   `assign_signatures` NNLS-refits vs COSMIC.

The memory blow-up is step 2+3: the per-*entry* expansion, materialized for the
whole cohort at once.

## Key facts about SVAR2 that shape the design

- **MNVs are atomized into SNVs** (`src/rvk.rs:124`, invariant `ilen == 0 ⟺ SNP
  with `alt_len == 1`). Native 2bp doublet records therefore *do not exist* in
  SVAR2 — every DBS-78 must come from adjacent-SNV pairing. Full parity (below)
  is the only option that yields any DBS at all.
- **Storage is split into four per-contig sub-streams**: `var_key/snp`,
  `var_key/indel`, `dense/snp`, `dense/indel` (`src/layout.rs`). var_key stores
  records **per call** (`positions.bin` is one `u32` per call); dense stores
  records **per distinct variant** (`positions.bin` is one `u32` per variant,
  plus a hap-major 1-bit genotype matrix).
- **The cost model routes per variant** (`src/rvk.rs:177`,
  `choose_representation`): var_key when carriers are few, dense when many. This
  per-call vs per-variant asymmetry is what makes "factor signatures into the
  cost model" meaningful.
- **REF is always recoverable from the FASTA**: SNP records store only a 2-bit
  ALT code (no REF); indels are left-aligned/validated against the reference, so
  REF equals the reference substring at POS. Classification needs the FASTA;
  counting must not.

## The asymmetry that drives everything

SBS96 and ID83 are pure **per-variant** functions of `(REF, ALT, reference
context)`. Only DBS-78's adjacent-SNV pairing is genotype/haplotype-dependent —
it is the one thing that cannot be reduced to a per-variant code, and it is
exactly what forced v1 into per-entry storage. The design stores a small
**per-record** code (mirroring each sub-stream's native granularity) and computes
DBS pairing on the fly during a single **streaming** count.

## Scope

In scope:
- SBS96, ID83, DBS78 (via adjacency), with `count="allele"|"sample"` — full v1
  parity.
- Classification during `from_vcf(..., signatures=True)` **and** post-hoc via
  `annotate_mutations`.
- Cost-model integration for the write-time path.

Non-goals:
- De-novo signature *extraction* (only refit, unchanged).
- Changing the refit math.
- Retroactive re-routing on post-hoc annotate (layout is locked).

## Design

### 1. Public API (Python, on `SparseVar2`)

Mirror v1 names so downstream code and the skill stay consistent:

- `sv2.annotate_mutations(reference, *, contigs=None)` — post-hoc. Classify the
  finalized records against the FASTA, write the sidecar, stamp `meta.json`
  (`mutcat_version`, `mutcat_contigs`). Layout is **locked**; no re-routing.
- `sv2.mutation_matrix(kind, *, count="allele"|"sample")` → `pl.DataFrame` with a
  `MutationType` column plus one column per sample, rows in fixed codebook order.
- `sv2.assign_signatures(kind, *, reference=None, max_delta=..., min_activity=...,
  n_jobs=..., backend=...)` → per-sample activities; delegates to the existing
  `fit_signatures`.
- `SparseVar2.from_vcf(..., signatures: bool = False)` — when true, classify
  during the write and factor the sidecar into the cost model (§4).

`fit_signatures` and `cosmic_signatures` stay module-level and unchanged. The
`_mutcat` **codebook** (labels, code ranges, `N_CODES`, `MUTCAT_VERSION`,
sentinels) is the single source of truth and is reused by both v1 and SVAR2;
the Rust classifier must produce identical code indices.

### 2. Sidecar data model

A new per-contig `mutcat/` directory whose arrays are **positionally aligned to
each existing sub-stream's records** — read in lockstep with the sub-stream's
`positions.bin`, no lookups, no global distinct-variant table:

| sub-stream    | `mutcat_code`       | `mutcat_ref`             |
| ------------- | ------------------- | ------------------------ |
| `dense/snp`   | u8 × n_dense_snp    | 2-bit × n_dense_snp      |
| `var_key/snp` | u8 × n_snp_calls    | 2-bit × n_snp_calls      |
| `dense/indel` | u8 × n_dense_indel  | —                        |
| `var_key/indel`| u8 × n_indel_calls | —                        |

Dense stores one code per distinct variant (amortized across all carriers — the
main win over v1); var_key stores one code per call (the rare-variant minority,
so per-call here is bounded). Exact on-disk file names are decided in the plan
but follow `src/layout.rs` conventions (e.g. `mutcat/{sub}/code.bin`,
`mutcat/{sub}/ref.bin`).

**Encoding — `mutcat_code`: `u8`, one per record, byte-aligned (not bit-packed).**
- SNP sub-streams: the SBS96 index, `0–95`.
- Indel sub-streams: the ID83 index, `0–82`.
- Two reserved high values for sentinels: `254 = UNCLASSIFIED`,
  `255 = NOT_ANNOTATED`. v1's negative `int16` sentinels do not survive into
  unsigned; the counter skips any `code ≥ N_CODES`.
- DBS78 is **never stored** — it is a count-time transform of a SNP pair, so the
  stored code is always the variant's own SBS96/ID83.
- All values fit in `< 128`; 7 bits would suffice, but `u8` is kept because the
  code is read on *every* count increment and byte-aligned lockstep indexing is
  simpler and more cache-friendly than a 7-bit stream straddling byte
  boundaries. The 1 wasted bit/record is not worth the non-aligned reader.

**Encoding — `mutcat_ref`: 2-bit packed, snp sub-streams only.**
- The one extra datum DBS pairing needs is the *actual* (unfolded) reference base
  at each SNV — 2 bits (`A/C/G/T`). ALT is already in the existing key stream;
  the SBS96 code alone cannot recover the strand-correct ref base (pyrimidine
  folding is lossy), and we do not want to require the FASTA at count time.
- Store it with the **existing `pack_snp_keys` / `unpack_snp_key_at` 2-bit
  codec** (`svar2-codec`) — the ref-base stream is structurally identical to the
  alt-key stream (one 2-bit code per snp record), so it is direct reuse with no
  new packing code. Read only when reconstructing a doublet (the rare
  adjacent-pair case), so its packing cost never touches the hot path.
- Indel sub-streams have no ref stream.

Why not v1's uniform `int16`? That is 16 bits/snp-call vs `8 + 2 = 10` here, and
the var_key/snp per-call stream dominates the new cost — `u8 code + 2-bit ref` is
the memory-conscious choice while staying byte-aligned on the hot field.

### 3. Classification pass (one code path, two entry points)

A single Rust routine walks the finalized per-contig records of each sub-stream,
reconstructs `(REF, ALT)` (ALT from the 2-bit key / indel key / long-allele bank;
REF from the FASTA), computes the SBS96 / ID83 index and the SNP reference base
via a Rust port of the v1 LUTs and `_id83_kernel`, and writes the sidecar. It is
called:
- automatically at the end of `from_vcf(signatures=True)` (the FASTA is already
  open), and
- by `annotate_mutations` post-hoc (re-opens the FASTA + streams the records).

Classification is per-variant and needs no genotypes — cheap, parallelizable
over contigs. `contigs=` scoping and the REF-mismatch warning behavior match v1
(`classify_variants`). Because it reads the *finalized* records, the sidecar is
guaranteed aligned to exactly what decode/count later read.

### 4. Cost-model integration (the write-time "factor it in")

When `signatures=True` **at write time**, extend `choose_representation` with the
sidecar's marginal bits (using actual storage widths):

- SNP record sidecar = `8 (code) + 2 (ref) = 10` bits.
- Indel record sidecar = `8 (code)` bits.

So per variant, with `x` = carrier count:
- var_key cost `+= sidecar_bits · x` (per call),
- dense cost `+= sidecar_bits` (per variant).

This slightly shifts the SNP/indel crossover toward dense when signatures are on
(dense amortizes the code across carriers) — a real, principled effect. The knob
is threaded through the conversion pipeline as a boolean (or bit-width constants)
so `choose_representation` stays a pure integer function.

**Post-hoc annotation cannot do this** — the layout is already locked, so the
sidecar is written at whatever granularity the existing var_key/dense split
dictates and `choose_representation` is not re-run. This is exactly why the cost
model is ignored there.

### 5. Count matrix — streaming pass with DBS pairing

`mutation_matrix` runs a single pass, parallelizable over sample-columns. For
each haplotype column:

- **SNVs**: merge-walk the column's SNVs from **both** sources in position order —
  `var_key/snp` calls (per-call code + ref) ⋈ `dense/snp` carried variants
  (per-variant code + ref, via the hap-major carried bit). This is the existing
  two-source decode pattern. For each **isolated adjacent pair** (`Δpos == 1`, no
  flanking SNV on either side — the exact v1 `_entry_codes_kernel` isolation
  rule), reconstruct the doublet from `(ref0, alt0, ref1, alt1)` and look up its
  DBS78 code via a `(4,4,4,4)` table (port of `_build_dbs_table`); both members
  of the pair contribute to the DBS channel. Every other SNV emits its stored
  SBS96 code.
- **Indels**: walk `var_key/indel` ⋈ `dense/indel`; each emits its stored ID83
  code.

Accumulate into an `(n_samples, N_CODES)` `int64` matrix: `count="allele"` sums
every carried copy; `count="sample"` marks presence at most once per sample.
Slice the requested `kind`'s code range and return the codebook-ordered
DataFrame (matching v1's `count_matrix`).

Peak memory ≈ the accumulator + one column's merge working set. No per-entry code
array ever exists — this is the memory win.

### 6. Refit (reused, unchanged)

`assign_signatures` builds the `kind` catalogue via `mutation_matrix` and calls
the existing `fit_signatures` against `cosmic_signatures(kind)` (or a supplied
reference). No changes to the refit code.

## Testing strategy

- **Rust unit tests**: the classifier port (SBS96 folding, ID83 repeat/microhomology
  logic) against the v1 LUT expectations; the streaming counter for DBS edge
  cases — runs of ≥3 adjacent SNVs stay SBS, doublets split across the
  var_key/dense boundary, `allele` vs `sample` counting.
- **Python parity test**: on shared fixtures, assert SVAR2's `mutation_matrix`
  (all three kinds, both count modes) equals v1's `SparseVar.mutation_matrix`,
  and `assign_signatures` matches within refit tolerance.
- **Cost-model tests**: crossover-shift unit tests for `choose_representation`
  with signature bits on/off (SNP and indel).
- **Round-trip**: `from_vcf(signatures=True)` and post-hoc `annotate_mutations`
  on the same input produce identical sidecars and matrices.

## Documentation / housekeeping

- Update `skills/genoray-api/SKILL.md` — this adds public `SparseVar2` methods and
  a `from_vcf` kwarg (mandatory per the repo's public-API rule in `CLAUDE.md`).
- CHANGELOG entry (Conventional Commits `feat:`).
- Bump `MUTCAT_VERSION` if the codebook or code semantics change (they should
  not — codes are reused verbatim).

## Open questions

- Exact on-disk file names / directory shape under `mutcat/` — settled in the
  plan against `src/layout.rs`.
- Whether the count pass parallelizes over columns with rayon or reuses an
  existing SVAR2 query executor — settled in the plan after reading
  `src/executor.rs` / `src/query/`.

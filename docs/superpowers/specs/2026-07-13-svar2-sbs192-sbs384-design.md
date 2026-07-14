# SBS192 / SBS384 transcriptional-strand-bias catalogs for SparseVar2

**Date:** 2026-07-13
**Status:** Approved design
**Scope:** `SparseVar2` (Rust-backed svar2 signature path) only. v1 `SparseVar` is untouched.

## Motivation

svar2 currently classifies SNVs into the strand-agnostic SBS96 catalog (plus
DBS78 and ID83). Transcriptional strand bias — the asymmetry between mutations
on the transcribed (template) vs. untranscribed (coding) strand of genes — is a
key readout for distinguishing mutational processes (e.g. transcription-coupled
repair signatures). Capturing it requires splitting each SBS96 channel by the
strand class of the variant's position, which in turn requires a gene model
(GTF). This adds the two standard SigProfiler strand-resolved catalogs:

- **SBS384** = 96 trinucleotide channels × 4 strand categories, in SigProfiler
  order `[T, U, N, B]`:
  - **T** — Transcribed (pyrimidine on the gene's template strand)
  - **U** — Untranscribed (pyrimidine on the gene's coding strand)
  - **N** — Nontranscribed (position in no gene footprint / intergenic)
  - **B** — Bidirectional (position covered by genes on both strands)
- **SBS192** = SBS384 restricted to `{T, U}`, i.e. exactly `SBS384[:192]` given
  the `[T, U, N, B]` block ordering. Implementing the 4-way strand split yields
  both catalogs; SBS192 needs no separate storage.

### Refit scope (decided)

COSMIC publishes reference signatures in **96-context only**. There are no
official SBS192/SBS384 reference sets — these catalogs exist for strand-bias
*analysis*, not refitting. Therefore:

- `mutation_matrix("SBS192" | "SBS384")` **is** the deliverable (the count
  matrix people feed to strand-bias tests).
- `assign_signatures("SBS192" | "SBS384")` raises `NotImplementedError` with a
  message pointing to `mutation_matrix`. SBS96/DBS78/ID83 refit is unchanged.

## The strand rule

SBS channels are pyrimidine-folded (reference base normalized to C or T). For a
genic SNV, define:

- `pyr_on_plus = refbase ∈ {C, T}` — is the pyrimidine of the ref/alt pair on
  the `+` (reference) strand?
- `gene_on_plus = gene strand is +`

Then (matching SigProfiler):

- position covered by genes on **both** strands → **B**
- position in **no** gene → **N**
- otherwise a single gene strand:
  - **Untranscribed (U)** iff `pyr_on_plus == gene_on_plus`
  - **Transcribed (T)** otherwise

Worked checks (gene on `+` strand):
- `A>T` folds to `T>A`; ref base `A` is a purine so `pyr_on_plus = false`;
  `false != true` → **T** (transcribed). ✓
- `C>G`; ref base `C` is a pyrimidine so `pyr_on_plus = true`;
  `true == true` → **U** (untranscribed). ✓

Strand applies to **SNVs only**. DBS78 and ID83 are unaffected.

## Public API

On `SparseVar2` (`python/genoray/_svar2_mutcat.py`, `_MutcatMixin`):

- `annotate_mutations(reference, *, gtf=None, contigs=None) -> None`
  - New optional `gtf=` (path-like). When supplied, a per-record strand class is
    computed and stored alongside the SBS96 codes, enabling SBS192/384.
  - `meta.json` gains `mutcat_strand: bool` and the GTF path for provenance.
  - Without `gtf=`, behavior is exactly as today (SBS96/DBS78/ID83).
- `mutation_matrix(kind, *, count=...) -> pl.DataFrame`
  - `kind` Literal gains `"SBS192"`, `"SBS384"`.
  - If the sidecar was not strand-annotated (`mutcat_strand` false / absent),
    raises a clear error instructing the user to re-annotate with `gtf=`. It does
    **not** silently return zeros.
- `assign_signatures(kind, ...)`
  - `"SBS192"`/`"SBS384"` → `NotImplementedError` (see refit scope above).

Write-time path (unchanged):

- `SparseVar2.from_vcf(..., signatures=True)` / `from_pgen(...)` continue to
  write **strand-free** sidecars (SBS96/DBS78/ID83) exactly as today. There is
  no `gtf=` kwarg on the conversion path. To obtain SBS192/384, call
  `annotate_mutations(reference, gtf=...)` once after conversion — it overwrites
  the sidecars, adding `strand.bin`. This keeps GTF parsing entirely in Python
  (via `seqpro`) and leaves the deep Rust conversion pipeline untouched.

## GTF → strand intervals (Python side)

Reuse the existing `seqpro.gtf.scan` machinery already used by v1
`_svar/_annotate.py::_load_gtf`. Per contig, flatten `feature == "gene"`
footprints (full gene body, introns included — matches SigProfiler's "gene
footprint") into a **sorted, disjoint interval partition** and pass it across the
pyo3 boundary as a struct-of-arrays:

```
strand_starts:  int32[]   # 0-based, half-open [start, stop)
strand_stops:   int32[]
strand_values:  uint8[]   # 1 = +only, 2 = −only, 3 = both (B); gaps ⇒ N (0)
```

- Only genic intervals are materialized; N is the implicit gap. Size is
  ~gene-count (tens of thousands genome-wide), not contig-length. No per-base
  array.
- Construction: for each strand collect gene spans, mark `+1`/`-1` at
  `start`/`stop`, cumsum to a coverage indicator, combine the two strands'
  coverage into the 4-way class, and emit maximal runs as `(start, stop, value)`.
- Default feature filter is `gene`. Biotype selection is the user's
  responsibility (pre-filter the GTF); documented as such.
- Contig-name normalization goes through the existing `ContigNormalizer` so
  `chr`-prefixed vs. unprefixed GTF/reference/variant names interoperate.

### Design rationale (interval SoA vs. per-base array)

An earlier draft passed a per-position `int8` array parallel to `ref_seq`
(~250 MB for chr1). The interval SoA is the chosen approach: payload is
proportional to gene count, GTF parsing stays in tested Python, and Rust
classifies with a two-pointer sweep (below) — no per-base array, no binary
search.

## Sidecar storage (Rust)

When a GTF is supplied, write a new **2-bit-packed `strand.bin`** stream for the
SNP sub-streams (`var_key_snp`, `dense_snp`), parallel to the existing 2-bit
`ref.bin` — one 2-bit value per SNP record encoding `{0:T, 1:U, 2:N, 3:B}`.

- Written **only** when strand-annotated; absent otherwise (backward compatible
  — older sidecars simply lack the stream and report `mutcat_strand: false`).
- Indel sub-streams get no strand stream (strand is SNV-only).
- `MutcatSub`/`has_ref` in `src/layout.rs` gain a parallel `has_strand()` (true
  for the SNP subs) and a `mutcat_strand` path builder mirroring `mutcat_ref`.
- `MutcatView` (`src/mutcat/sidecar.rs`) gains a `strand_at(i)` accessor,
  present only when the stream exists.
- Cost model (`src/cost_model.rs`): `SIDECAR_BITS_SNP` accounts for +2 bits per
  SNP when strand annotation is enabled.

## Classification pass (Rust)

`annotate_contig` (`src/mutcat/annotate.rs`) gains the interval SoA
(`strand_starts`, `strand_stops`, `strand_values`) alongside `ref_seq`, and
threads it into the SNP path. Because SVAR2 records are position-sorted per
contig and the intervals are sorted, classification is a **two-pointer sweep**:

- advance the interval cursor while `stop <= pos`
- the current interval value (or 0/N if `pos` falls in a gap) is the position's
  strand class
- combine with the pyrimidine rule (above) using `refb = ref_seq[p]` to produce
  `{T, U, N, B}`

Complexity O(n_variants + n_intervals), no per-base allocation. The stored SBS96
code is unchanged; the strand byte is written to `strand.bin`. A SNP that is
`UNCLASSIFIED`/`NOT_ANNOTATED` for SBS96 gets an arbitrary strand value (it is
never counted).

`snp_record` returns the strand class in addition to `(code, ref2bit)`; the
write path packs it into `strand.bin`.

## Codebook + count

Codebook (`python/genoray/_mutcat/codebook.py` and `src/mutcat/mod.rs`, kept in
lockstep, enforced by `code_space_matches_codebook`):

- Add one **SBS384 block** to the unified code space:
  `SBS384 = [257, 641)` (i.e. `SBS384_OFFSET = N_CODES_old = 257`,
  `N_CODES = 641`).
- Labels in `[T, U, N, B] × SBS96` order, formatted
  `{strand}:{5'}[{ref}>{alt}]{3'}` (e.g. `T:A[C>A]A`). The exact separator/format
  is pinned against SigProfiler output during implementation.
- `SBS192` is a Python-level view = the SBS384 block sliced `[:192]` (the T and U
  channels). No separate stored range, no separate Rust offset.
- `Kind` Literals/enums gain `"SBS384"` (and `"SBS192"` at the Python API layer,
  resolving to the `SBS384[:192]` slice). Bump `MUTCAT_VERSION`.

Count (`src/mutcat/count.rs`, `src/py_mutcat.rs`):

- The accumulator widens to `(n_samples, N_CODES=641)` `i64`.
- For each valid SNV, in addition to its SBS96 bin, increment
  `257 + strand_idx*96 + sbs96_code` **when the sidecar has a strand stream**.
  `strand_idx` uses `[T,U,N,B] = [0,1,2,3]`.
- When there is no strand stream, no SBS384 emission occurs.
- `mutation_matrix("SBS384")` slices `[257, 641)`; `"SBS192"` slices the first
  192 of that block. Both raise (via the API guard on `mutcat_strand`) rather
  than returning zeros when strand was not annotated.
- DBS78 pairing and ID83 counting are unchanged.

## Backward compatibility (no change to existing layout or cost)

Adding this feature does **not** change how SBS96/DBS78/ID83 are stored or
counted:

- `code.bin` stores **within-kind** codes as today: `sbs96_code` (0–95) for SNPs,
  `id83_code` (0–82) for indels, sentinels 254/255. All fit `u8`. The SBS384 code
  is never stored as a single value — SBS96 (`code.bin`, `u8`) and strand
  (`strand.bin`, 2-bit) are stored separately. SBS96 does **not** need >8-bit
  encoding.
- The existing offsets (`SBS96=0, DBS78=96, ID83=174`) are unchanged; the SBS384
  block is appended at 257. Offsets are applied only at count time when indexing
  the `i64` accumulator, so widening `N_CODES` 257→641 affects only the transient
  in-memory count matrix — never on-disk bytes, never a `u8`.
- A sidecar annotated **without** a GTF is byte-for-byte identical to today.
  `strand.bin` is purely additive and written only when `gtf=` is supplied.
- Old sidecars remain valid under the new codebook (stored within-kind codes map
  to identical unified indices). The `MUTCAT_VERSION` bump is a
  **backward-compatible read**: SBS384 availability is gated on `strand.bin`
  presence (`mutcat_strand`), not on version equality — no forced re-annotation.
- Implementation note: the hard-coded `N_CODES == 257` asserts
  (`src/mutcat/mod.rs` `code_space_matches_codebook`, `src/mutcat/count.rs`
  `emit_snv_codes`) update to 641.

## Testing

Rust:
- Strand rule unit tests against hand-worked SigProfiler cases: gene-on-`+`
  `A>T` → T (transcribed); gene-on-`+` `C>G` → U (untranscribed); gene-on-`−`
  mirror cases; overlapping opposite-strand genes → B; intergenic → N.
- Two-pointer sweep tests: variants before / after / between / inside intervals,
  half-open boundary handling (`pos == start`, `pos == stop`), multi-interval
  contigs, empty interval set (all N).
- Sidecar round-trip: `strand.bin` write → `strand_at` read for both SNP subs.

Python:
- Parity: a small synthetic VCF + FASTA + GTF where the SBS384 matrix collapsed
  over the 4 strand blocks equals the SBS96 matrix, and `SBS192 == SBS384[:192]`.
- Contig-name normalization: `chr`-prefixed GTF against unprefixed variants (and
  vice versa) via `ContigNormalizer`.
- `mutation_matrix("SBS384")` on a sidecar annotated without `gtf=` raises a
  clear error.
- `assign_signatures("SBS384")` raises `NotImplementedError`.
- Prefer `vcfixture` for the synthetic VCF + ground-truth oracle where it fits.

## Docs

- Update `skills/genoray-api/SKILL.md` — the svar2 signatures section: add
  SBS192/384, the `annotate_mutations(gtf=...)` kwarg, the strand rule and
  category definitions, the note that write-time `signatures=True` stays
  strand-free (use post-hoc `annotate_mutations(gtf=...)`), and the
  refit-not-supported note. Clarify that the existing "no SBS-192" limitation
  note applies to **v1 `SparseVar`** only.
- Add a `CHANGELOG.md` Unreleased → Added entry.

## Out of scope

- v1 `SparseVar` strand-resolved catalogs (unchanged; keeps its documented
  no-strand limitation).
- Write-time strand annotation via `from_vcf`/`from_pgen`/`from_vcf_list`
  (`signatures=True` stays strand-free; SBS192/384 comes from a post-hoc
  `annotate_mutations(reference, gtf=...)`). Deferred to avoid threading a GTF
  through the Rust conversion pipeline for a one-FASTA-reload saving.
- SBS288 (`96 × {T,U,N}`), SBS1536/SBS6144, replication strand bias.
- A dedicated strand-asymmetry statistical test helper (users compute it from
  the returned matrix). Possible future work.
- Refitting SBS192/384 against any reference (no COSMIC set exists).

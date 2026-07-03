# SparseVar haploid (OR-collapse) write option — design

**Date:** 2026-06-24
**Status:** Approved (pre-implementation)

## Problem

Somatic variant data is typically unphased. Storing it in the diploid
`(samples, ploidy=2, variants)` SparseVar layout is redundant and can be
misleading: the two haplotype slots carry no real phase information, yet they
double the stored entries for homozygous calls and invite consumers to read
significance into a per-haplotype split that does not exist.

We want a write-time option that collapses the ploidy axis into a single
haploid representation by OR-ing presence across haplotypes (a variant present
on *any* haplotype becomes one haploid call), and records `ploidy=1` in the
SparseVar metadata so the haploid nature is explicit to every downstream
reader.

## Key finding: infrastructure is already ploidy-generic

The read, `write_view`, and mutation-catalogue paths already thread `ploidy`
from `metadata.json` through every kernel:

- `SparseVar.__init__` reads `self.ploidy = metadata.ploidy` and builds all
  shapes as `(n_samples, self.ploidy, None)`.
- `read_ranges`, `read_ranges_with_length`, `_find_starts_ends*` use
  `self.ploidy`.
- `write_view` and its Numba kernels (`_nb_count_mac_per_kept`,
  `_nb_count_kept`, `_nb_write_var_idxs`, `_nb_write_field`, `_nb_af_helper`)
  take `ploidy` as a parameter.
- `_mutcat.py` DBS-adjacency kernels take `ploidy` as a parameter
  (`_mutcat.py:632`).
- `_subset_var_idxs_and_recompute_af` computes `af = mac / (n_out * ploidy)`.

The **only** place ploidy is pinned to 2 is the write entry points
(`from_vcf` / `from_pgen`), which copy `vcf.ploidy` / `pgen.ploidy` and never
collapse. So this feature is primarily a collapse step at write time plus an
audit confirming nothing silently assumes diploid — not a rewrite.

## Approach

Add `haploid: bool = False` to `SparseVar.from_vcf` and `SparseVar.from_pgen`,
surfaced as `--haploid` on the CLI `write` command. When set:

1. The dense per-contig workers OR-collapse the ploidy axis before
   `dense2sparse`.
2. The output metadata records `ploidy=1`.
3. All downstream read/view/annotate paths work unchanged because they already
   honor `metadata.ploidy`.

### Scope decision

The collapse is **whole-file and unconditional**. A single flag collapses the
ploidy axis to 1 for all samples and variants. Phase is not gated on — phase
was never stored in SparseVar, so there is nothing to check, and the somatic
use case is uniform across the cohort.

### Where the collapse happens

In the dense per-contig workers, immediately after the dense
`(samples, ploidy, variants)` array is assembled and **before**
`dense2sparse`:

```python
if haploid:
    genos = (genos == 1).any(axis=1, keepdims=True).astype(np.int8)  # (s, 1, v)
```

- `_process_contig_vcf` — collapse after the chunk's `genos` is sliced to the
  kept variants (after the `keep_sorted` selection).
- `_process_contig_pgen` — pgenlib is still read at native diploid; collapse
  after the `(v, s*p) -> (s, p, v)` reshape and `-9 -> -1` fixups, before
  `dense2sparse`.

`dense2sparse` already keys off `keep = genos == 1` and derives lengths/offsets
from `genos.shape`, so a `(s, 1, v)` input yields correct haploid offsets with
no change to that function.

### OR semantics

OR is taken over `genos == 1`:

| Haplotype A | Haplotype B | Haploid result |
|-------------|-------------|----------------|
| ALT (1)     | ALT (1)     | present (1 entry) |
| ALT (1)     | ref (0)     | present |
| ALT (1)     | missing (-1)| present |
| ref (0)     | ref (0)     | absent |
| ref (0)     | missing (-1)| absent |
| missing (-1)| missing (-1)| absent (no entry stored — unchanged from today) |

### Dosages

Dosages have shape `(samples, variants)` with no ploidy axis. In
`dense2sparse` they are broadcast to the genotype shape and indexed by `keep`.
With a collapsed `(s, 1, v)` genotype shape, each present haploid call keeps its
single dosage value — no reduction and no special-casing required. A
homozygous-ALT variant that stored its dosage twice under diploid now stores it
once.

## Wiring in the write entry points

In both `from_vcf` and `from_pgen`:

- `out_ploidy = 1 if haploid else source.ploidy`
- write `metadata.json` with `ploidy=out_ploidy`
- `shape = (n_out, out_ploidy)` (drives `_concat_data` offsets and the per-chunk
  ragged offsets)
- pass `haploid` into the `_process_contig_*` joblib task
- pass `ploidy=out_ploidy` to `_subset_var_idxs_and_recompute_af` (yields the
  correct haploid AF, denominator `n_out`)

## CLI

Add `--haploid` to the `write` command, styled like the existing
`--no-symbolic` / `--no-breakend` flags
(`Annotated[bool, Parameter(name="--haploid", negative="")] = False`), threaded
into both the `from_vcf` and `from_pgen` calls. The docstring documents the
intended unphased-somatic use case and that it sets `ploidy=1` in metadata.

## Audit: "support arbitrary ploidy"

This is a verification pass, not a rewrite. Confirm no method silently assumes
`ploidy == 2`, fixing any spot found to read `self.ploidy` / the passed
`ploidy`:

- `read_ranges`, `read_ranges_with_length`, `_find_starts_ends*` — use
  `self.ploidy` (confirmed).
- `write_view` and its Numba kernels — `ploidy` is a parameter (confirmed).
- `_mutcat.py` DBS adjacency — `ploidy` is a parameter (confirmed).
- `_open_genos` / `_open_fmt` shape construction — confirm shapes come from
  metadata ploidy, not a literal (confirmed in `__init__` / `with_fields`).
- Phantom `empty()` classmethods and any other shape literals — confirm none
  hardcode 2.

## Testing

- Round-trip `from_vcf` with `haploid=True`: assert `metadata.ploidy == 1`,
  `read_ranges` returns shape `(ranges, samples, 1, ~variants)`, and the
  haploid call set equals the OR of the two diploid haplotypes for the same
  fixture.
- Round-trip `from_pgen` with `haploid=True`: same assertions.
- Dosages + `haploid=True`: present calls carry the expected single dosage;
  a hom-ALT variant stores one entry, not two.
- Sample-subset + `haploid=True`: MAC=0 drop still fires; AF denominator is
  `n_out` (not `2 * n_out`).
- `write_view` and `annotate_mutations` on a `ploidy=1` SparseVar work
  end-to-end.
- CLI `write --haploid` smoke test for both VCF and PGEN sources.
- Use `vcfixture` for a ground-truth oracle where practical (derive the
  expected haploid OR from the decoded diploid genotypes).

## Docs

Per the project `CLAUDE.md`, the same PR updates the public-API skill and docs:

- `skills/genoray-api/SKILL.md` — change the "Ploidy is always 2" convention
  line to note that ploidy is 2 unless written with `haploid=True`, in which
  case it is 1; document the `haploid` kwarg on `from_vcf` / `from_pgen`.
- `docs/source/svar.md` — document the `haploid` write option and the
  `--haploid` CLI flag.

## Out of scope

- No per-sample or per-region collapse — the flag is whole-file.
- No phasing detection or rejection of phased sources.
- No collapse to ploidy values other than 1 (the mechanism generalizes, but no
  use case is requested).
- No read-time / lazy collapse — the collapse is physical at write time.

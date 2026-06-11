# VCF/PGEN → SVAR conversion with on-the-fly region & sample subsetting

**Status:** approved for plan-writing
**Branch:** `feat/svar-convert-subset`
**Date:** 2026-06-10

## Goal

Let `SparseVar.from_vcf` and `SparseVar.from_pgen` subset the source by **region**
and **sample** *during* conversion, so a region/sample slice of a VCF/PGEN can be
written to an SVAR without first materializing the full SVAR and then calling
`write_view`. Semantics mirror `write_view` (`merge_overlapping`,
`regions_overlap`, MAC=0 dropping on sample subsets).

## Public API

Four optional, keyword-only parameters added to **both** `from_vcf` and
`from_pgen`:

```python
@classmethod
def from_vcf(
    cls, out, vcf, max_mem,
    overwrite=False, with_dosages=False, n_jobs=-1,
    *,
    regions: str | tuple[str, int, int] | IntoFrameT | PathLike | None = None,
    samples: str | Sequence[str] | PathLike | None = None,
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
): ...

@classmethod
def from_pgen(
    cls, out, pgen, max_mem,
    overwrite=False, with_dosages=False, n_jobs=-1,
    *,
    regions: ... = None,
    samples: ... = None,
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
): ...
```

- `regions=None` → all variants (current behavior). Non-`None` accepts the same
  input types as `write_view`: `"chr:start-end"` (1-based inclusive),
  `(chrom, start, end)` (0-based half-open), frame-like
  (polars/pandas/pyranges), or a BED file path.
- `samples=None` → all samples (current behavior). Non-`None`: a name, sequence
  of names, or path to a newline-delimited file. Caller order is preserved,
  deduped by first occurrence.
- `merge_overlapping` / `regions_overlap` → identical semantics to `write_view`;
  consulted only when `regions` is given.
- The new params are **keyword-only** so existing positional callers of
  `overwrite, with_dosages, n_jobs` are unaffected.
- Subsets compose with the source's existing `_pl_filter` by **intersection**: a
  variant is kept iff it passes the source filter *and* falls in a region.
- When `regions is None and samples is None`, the code path is byte-identical to
  today (regression-guarded by test).

This is a public-API change → `skills/genoray-api/SKILL.md` must be updated in the
same PR (per CLAUDE.md).

## Architecture

The existing conversion pipeline writes the index **up front**
(`_write_filtered_index`), then runs the genotype scan independently; the two
agree only because both apply the same `_pl_filter`. Regions can ride that shared
filter, but **MAC-drop on a sample subset cannot** — MAC is only known after the
scan. So on the subsetting path the index is finalized *after* the scan.

### 1. Region & sample resolution (up front)

Runs in `from_vcf` / `from_pgen` before dispatching workers.

**Samples.** `_normalize_samples(samples, source.available_samples)` (reused from
`write_view`) → ordered, deduped `caller_samples`; derive `src_sample_idxs`
(positions into the source sample axis) for the workers. Metadata `samples`
becomes `caller_samples` (was `available_samples`). Empty `samples` (`[]`) →
`ValueError` (mirrors `write_view`'s "at least one sample" guard).

**Regions → kept variant rows.** The filtered index (post `_pl_filter`) is the
variant-numbering source of truth, so region resolution runs against *that* frame:

1. Materialize the filtered index frame once — the same lazy filter
   `_write_filtered_index` builds, collected to a DataFrame with `CHROM, POS,
   ILEN`. Variant *i* in the SVAR == row *i* of this frame.
2. `_normalize_regions(regions, source._c_norm)` → 0-based half-open BED frame
   (reused verbatim).
3. Resolve region membership over the filtered frame, honoring
   `merge_overlapping` + `regions_overlap`, via the module-level
   `_var_ranges.var_ranges(...)` for `"variant"` mode and POS-membership for
   `"pos"`/`"record"`.

**Shared resolver.** Extract the body of `_resolve_kept_var_idxs` into a
frame-level helper `_resolve_kept_rows(index_df, c_norm, regions, mode,
merge_overlapping) -> NDArray` operating purely on an index frame. `write_view`'s
existing `_resolve_kept_var_idxs` becomes a thin wrapper (`sv.index`,
`sv._c_norm`); conversion calls the same helper. One implementation, two callers.

The result is a sorted array of kept **row positions** in the filtered index. For
PGEN these map directly to the per-contig physical `keep_idxs` already used by the
dispatch. For VCF they become a per-contig keep-set the worker filters against.

### 2. Scan changes

**VCF worker (`_process_contig_vcf`)** gains `caller_samples` and the per-contig
sorted keep-set (row positions in the filtered stream for this contig):

- `vcf.set_samples(caller_samples)` so genotypes are read for the subset only.
- Read via `chunk(contig, start=min_kept_start, end=max_kept_end, ...)` — a cheap
  single-range seek that trims the contig head/tail outside any region (full
  `chunk` when `regions is None`).
- As chunks stream in (filtered order), maintain a running filtered-position
  counter and emit only rows in the keep-set. Both the on-disk index and this
  scan apply the same `_pl_filter` *and* the same row selection, so **scan order
  == index order by construction** — no coordinate-matching fragility. (Per-region
  seeking is a possible future I/O optimization, not needed for correctness.)

**PGEN worker (`_process_contig_pgen`)** gains `sample_subset` (positions into the
source samples):

- The `keep_idxs` it already receives are the region∩filter-resolved physical
  indices — region restriction is *free* here.
- `geno_reader.change_sample_subset(sample_idxs)` (and the dosage reader, if any)
  before `read_alleles_list`; read into a `(v, n_kept_samples*ploidy)` buffer.

In both workers `caller_samples` defaults to all samples → unchanged behavior when
`samples is None`.

### 3. MAC-drop & index finalization

Runs only when `samples is not None` (a sample subset can zero a variant; keeping
all samples cannot, so the region-only path skips this and stays simple).

**Count MAC without an extra pass.** The scan emits sparse data: per
`(sample, ploidy)` slot, the non-ref variant indices. A variant's MAC across the
kept samples is exactly how many times its index appears in the concatenated
sparse data: `np.bincount(all_var_idxs, minlength=n_kept)` over the written
genotype indices — no genotype re-read.

**Prune + remap.**

1. `survivors = where(mac > 0)` → compacted numbering `old_idx → new_idx`.
2. Remap the sparse genotype var-index data to the compacted range, dropping
   entries for MAC=0 variants and recomputing offsets — the same remap shape
   `write_view`'s pass-2 performs. **Reuse `write_view`'s existing numba remap
   kernels** rather than writing new ones. If `with_dosages`, the parallel dosage
   arrays are remapped alongside.
3. Filter the index frame to `survivors` before writing `index.arrow`; recompute
   AFs over the kept samples.
4. If *every* candidate drops → raise the same `"no variants selected"`-style
   `ValueError` `write_view` uses.
5. `warnings.warn` the dropped count when > 0 (matches `write_view`).

**Flow placement.** Today: write index → scan → `_concat_data`. Subsetting flow
when `samples` is set: scan → `_concat_data` → compute MAC → prune/remap data +
finalize index. The index write moves *after* the scan on this path only; the
up-front `_write_filtered_index` path is untouched when `samples is None`.

## Edge cases & errors

- `regions` selects no variants → `ValueError("no variants selected by regions")`.
- Overlapping regions with `merge_overlapping=False` → `ValueError` (shared
  resolver).
- Unknown sample names → `ValueError` (`_normalize_samples`).
- Region contigs absent from the source → dropped with `UserWarning`
  (`_normalize_regions`).
- `regions=None, samples=None` → byte-identical to current output.
- `samples=[]` → `ValueError`.
- `with_dosages=True` composes with both subsets; dosages remapped with
  genotypes.

## Testing

New `tests/test_svar_from_subset.py` (or extend existing `from_*` tests), for
both VCF and PGEN backends:

- regions-only: output variant set == source variants in those regions (per
  `regions_overlap` mode), all samples retained.
- samples-only: correct samples/order in metadata; MAC=0 variants dropped; AF
  recomputed.
- combined regions + samples.
- `merge_overlapping` true/false; overlap with `False` raises.
- **equivalence oracle:** `from_vcf(regions=R, samples=S)` produces an SVAR equal
  to `from_vcf()` followed by `write_view(R, S)` — same variants, genotypes,
  dosages. Validates the "convert-time subset == post-hoc view" promise directly.
- no-arg regression: subset params unset == current output.
- PGEN tests guarded on `plink2` availability (existing convention).

## Docs

- Update `skills/genoray-api/SKILL.md` (new kwargs on `from_vcf`/`from_pgen`).
- `CHANGELOG.md` `feat` entry.
- Docstrings on both methods.

## Files (anticipated)

- Modify `genoray/_svar.py` — add subset params + resolution/finalize logic to
  `from_vcf`/`from_pgen`; extract `_resolve_kept_rows`; rework
  `_process_contig_vcf` (samples + keep-set) and `_process_contig_pgen`
  (sample_subset); MAC-drop finalize reusing `write_view`'s remap kernels.
- Modify `skills/genoray-api/SKILL.md`, `CHANGELOG.md`.
- Create `tests/test_svar_from_subset.py`.

## Out of scope

- Per-region seeking I/O optimization for VCF (whole-contig-span read is the
  initial approach).
- Enforcing the MAC>0 invariant when `samples is None` (kept samples can't zero a
  variant; relies on the source already satisfying it).
- Any CLI surface for these flags (separate `genoray-cli` change if desired).

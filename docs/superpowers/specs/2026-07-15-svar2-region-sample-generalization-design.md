# Generalize Region And Sample Subsetting Across SVAR2 Conversion Methods

**Status:** Design approved, pending spec review.

**Stacks on:** PR #114 (`feat(svar2): add region and sample VCF conversion`),
which adds `regions=`/`samples=` to `SparseVar2.from_vcf` only. This work lands
as a follow-up PR that merges #114 as the VCF foundation, then extends the same
capability to the remaining conversion methods and refactors the shared
front-end.

## Goal

Give every `SparseVar2` conversion classmethod a uniform region- and
sample-subsetting API:

| Method | `regions` | `samples` |
| --- | --- | --- |
| `from_vcf` | done in #114 | done in #114 |
| `from_vcf_list` | **add** | **N/A** (each input is single-sample; the cohort *is* the file set) |
| `from_pgen` | **add** | **add** |
| `from_svar1` | **add** | **add** |

The public parameters mirror #114 exactly, so the four methods present the same
surface:

```python
regions: str | tuple[str, int, int] | PathLike | object | None = None,
samples: str | Sequence[str] | PathLike | None = None,   # omitted on from_vcf_list
merge_overlapping: bool = False,
regions_overlap: Literal["pos", "record", "variant"] = "pos",
```

`regions=None, samples=None` MUST stay byte-compatible with the current
full-file conversion path for every method.

## Coordinate & mode conventions (unchanged from #114 / v1)

- Region strings `"chrom:start-end"` are 1-based inclusive → converted to
  0-based half-open. Tuple/BED/frame inputs are already 0-based half-open.
- `merge_overlapping=False` raises on overlapping requested regions;
  `merge_overlapping=True` coalesces them first.
- `regions_overlap` matches bcftools `--regions-overlap`:
  - `pos` — keep a variant only if its POS falls inside `[start, end)`.
  - `record` — also keep records overlapping by their VCF REF span
    (`[POS, POS+len(REF))`), i.e. indels whose POS sits just past the region end.
  - `variant` — keep only true overlapping sequence variation: trim the shared
    anchor base(s) from REF/ALT and test the trimmed span.

### Correction to the #114 framing: `variant` is NOT deferred

PR #114 reserved `regions_overlap="variant"` for a future "sub-contig sharding"
PR. That deferral conflated two independent concerns:

1. **Shard ownership** — deduplicating a left-aligned indel that crosses a
   parallel-worker boundary. This is a *parallelism* problem and genuinely
   belongs to the sub-contig follow-up.
2. **Region overlap filtering** (pos/record/variant) — a **per-record**
   decision that htslib's interval fetch plus an anchor-trim test resolves in
   the serial reader today, exactly as bcftools does. No left-alignment and no
   sharding are required.

This design implements all three overlap modes now, for all backends. The
sub-contig follow-up remains a separate, orthogonal performance effort.

### `variant`-mode granularity (approved: per-record)

For a **multiallelic** VCF record in `variant` mode, some ALT alleles may
overlap the region while others do not. We keep the whole record if **any**
allele's anchor-trimmed span overlaps the region (bcftools' granularity), rather
than dropping individual non-overlapping alleles after atomization. For
biallelic variants — the overwhelming majority — per-record and per-atom are
identical.

**This behavior MUST be called out in the docstring of every method that
accepts `regions_overlap`** (`from_vcf`, `from_vcf_list`, `from_pgen`,
`from_svar1`): in `variant` mode a multiallelic record is kept whole if any of
its alleles truly overlaps the region; per-allele dropping is not performed.
PGEN and svar1 are inherently per-atom (their index rows are already atoms) — a
minor documented cross-backend nuance.

## Architecture

### Shared region front-end (Python)

Replace the #114 VCF-specific `_normalize_svar2_vcf_regions` with a
backend-agnostic front-end in `python/genoray/_svar2.py`:

```
_normalize_svar2_regions(regions, available_contigs, *, merge_overlapping)
    -> pl.DataFrame           # normalized regions: chrom (Utf8), start, end (0-based half-open)
```

Responsibilities:
- Parse `str` / `tuple` / `PathLike` (BED) / frame via the existing v1
  `genoray._svar._regions._normalize_regions` and `ContigNormalizer`.
- Detect overlaps and coalesce iff `merge_overlapping=True`, else raise.
- Return a normalized `regions_df`. It does **not** bake `regions_overlap` into
  coordinates (this removes #114's `record`-mode "+1 bp fetch" hack); the mode
  is carried separately to the backend filter.

Two translators consume `regions_df`:

- **Genomic-interval translator** (VCF, vcf_list): `regions_df` → coalesced
  per-contig `(chrom, start, end)` fetch intervals. The overlap mode is applied
  in the Rust reader, per record.
- **Index-interval translator** (PGEN, svar1): reuse v1's
  `_resolve_kept_rows(index_df, cnorm, regions_df, mode, merge_overlapping)`,
  which already implements pos/record/variant on a POS+ILEN index, then coalesce
  kept indices into contiguous `[lo, hi)` variant-index runs per contig.

### Sample front-end (Python)

All sample-accepting methods reuse v1 `genoray._svar._regions._normalize_samples`
(preserves caller order, dedupes first occurrence, raises on unknown names).
The uniform contract is: **samples are selected and reordered by name, caller
order preserved.**

## Per-backend implementation

### `from_vcf` (already merged in #114 — refactor only)

Repoint it at the shared `_normalize_svar2_regions`, pass the un-widened
intervals plus the `regions_overlap` mode to Rust, and drop the local
end-widening / double-coalesce. Extend its docstring with the `variant`-mode
granularity note.

### `from_vcf_list`

- Regions: thread the coalesced genomic intervals + `regions_overlap` mode
  through `VcfListRecordSource::new` (`src/vcf_list_reader.rs`), which currently
  hardcodes `Vec::new()` when constructing each per-file `VcfRecordSource::new`.
  Every per-file reader fetches the same intervals and applies the same
  per-record overlap filter.
- Samples: **not accepted** — no `samples` parameter is added. Each input is
  single-sample and the cohort is defined by `sources`.
- All three overlap modes come for free from the shared VCF reader.

### `from_vcf` / `from_vcf_list` Rust reader (`src/vcf_reader.rs`)

Replace the pos-only skip predicate with a mode-aware, per-record overlap test
against the active region:

- `pos` — `start <= POS < end`.
- `record` — `[POS, POS + len(REF))` overlaps `[start, end)`.
- `variant` — anchor-trimmed span overlaps `[start, end)` (trim common
  prefix/suffix of REF/ALT; for a multiallelic record, keep if any allele's
  trimmed span overlaps).

htslib's `IndexedReader.fetch` indexes by `pos + rlen`, so the interval fetch
already returns the overlapping superset (including indels starting before
`start`); no fetch-window change is required. `VcfRecordSource::new` gains a
`regions_overlap` argument alongside the existing `regions: Vec<(u32, u32)>`.

### `from_pgen`

- Regions: extend the pvar scan (`_pvar_contig_ranges` / `_scan_pvar`) to
  produce an `index_df` (CHROM/POS/ILEN/index), run `_resolve_kept_rows`, and
  coalesce kept indices into contiguous `[lo, hi)` variant-index runs per contig.
  Extend `PgenRecordSource` (`src/pgen_reader.rs`) to iterate **multiple** index
  intervals per contig (mirrors the VCF reader's `advance_region` /
  `current_region` state).
- Samples: `_normalize_samples` → psam sample indices → `change_sample_subset`
  per reader (already used in `_pgen.py:283`). pgenlib requires *sorted*
  indices, so add a `sample_perm: Vec<usize>` remap in `PgenRecordSource` (the
  PGEN analogue of the VCF reader's `sample_indices`) to restore caller order in
  the emitted columns. The `samples` list written to the store reflects caller
  order.

### `from_svar1` (heaviest)

`from_svar1` reads sparse genotype runs by variant index through
`run_svar1_conversion_pipeline`, so both subsettings push into that Rust reader:

- Regions: `_resolve_kept_var_idxs(sv1, regions_df, mode, merge_overlapping)`
  (v1 already exposes this) → kept-variant index set passed into the pipeline;
  the svar1 reader gains a kept-variant filter (skip variant indices not in the
  set) and narrows the per-contig index arrays accordingly.
- Samples: `_normalize_samples` → sample-index array passed into the pipeline;
  the svar1 reader filters and remaps the sparse per-variant sample entries to
  the selected, caller-ordered subset. This is the deepest reader change of the
  four backends.

## CLI (`python/genoray/_cli/__main__.py`)

`genoray write` already grew `--regions/-r`, `--regions-file/-R`,
`--samples/-s`, `--samples-file/-S` in #114 for the VCF path. Extend dispatch so
the same flags reach `from_pgen` and `from_svar1`. Reject `--samples`/
`--samples-file` for the vcf-list input form with a clear error. Reject
combining inline and file variants (as #114 does).

## Testing

Per backend (`tests/test_svar2_from_vcf.py`, plus new
`tests/test_svar2_from_pgen.py`, `tests/test_svar2_from_svar1.py`,
`tests/test_svar2_from_vcf_list.py` as needed, and `tests/cli/test_write_cli.py`):

- Region restriction produces exactly the expected variants.
- All three `regions_overlap` modes, including an indel whose POS sits just past
  a region end (`pos` excludes, `record`/`variant` include per bcftools) and a
  multiallelic record where only some alleles overlap in `variant` mode
  (per-record: kept whole).
- Sample subset + reordering preserves caller order; unknown sample raises
  `ValueError`; unknown contig drops with `UserWarning` (v1 behavior).
- `merge_overlapping=False` raises on overlap; `True` coalesces.
- `regions=None, samples=None` matches the full-file conversion byte-for-byte.
- Rust: `cargo test --no-default-features --features conversion` for the reader
  changes (VCF multi-interval + mode filter, PGEN multi-range + sample_perm,
  svar1 kept-variant + sample remap).

Gate: focused pytest per task, `pixi run pytest tests -m "not network"` before
PR readiness, `pixi run prek run --all-files` before push.

## Docs & skill (mandatory)

- Update `docs/source/svar.md` region/sample section to cover all four methods,
  the three overlap modes (no longer "variant reserved"), and the per-record
  `variant` granularity note.
- Update `skills/genoray-api/SKILL.md` — this changes public signatures on three
  methods (`from_vcf_list`, `from_pgen`, `from_svar1`) and the semantics of
  `regions_overlap` on `from_vcf`. Required per project CLAUDE.md.

## Non-goals

- Sub-contig parallel normalization / shard ownership (the separate follow-up).
- Per-atom `variant`-mode allele dropping for VCF (per-record chosen).
- `bcftools merge`-style batched/hierarchical merge for very large vcf-list
  cohorts (unchanged; raise the open-file limit).

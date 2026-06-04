# Design: SparseVar filter inheritance + symbolic-allele filtering (PR #51 rework)

**Date:** 2026-06-04
**Branch:** `feat/skip-symbolic-alts`
**Supersedes:** the `skip_symbolic_alts` flag approach in PR #51.

## Problem

VCFs (and PGENs derived from them via plink2) can carry VCF 4.x symbolic ALT
alleles (`<DEL>`, `<INS>`, …). genoray stores those literal `<DEL>` bytes in its
index and sparse data; downstream haplotype tooling (genvarloader) then writes
them into DNA buffers, producing non-canonical bytes / NUL / X residues in
translated proteins. See the seqlab investigation linked in PR #51.

PR #51 solved this with an opt-in `skip_symbolic_alts: bool` flag on
`VCF.__init__`, `SparseVar.from_vcf`, and the CLI. That flag is bloat: genoray
already has a filter API (`filter` + `pl_filter` on `VCF`, `filter` on `PGEN`,
plus the `genoray.exprs` expressions). The right fix is to make that existing
API actually work end-to-end and express "drop symbolic alts" as a normal
filter.

### Two latent gaps this rework fixes

1. **`SparseVar.from_vcf` / `from_pgen` silently ignore the source's filter.**
   Both `shutil.copy` the *unfiltered* on-disk `.gvi` into the SVAR, and their
   per-contig workers re-open the source with **no** filter. A filter passed to
   `VCF`/`PGEN` is accepted but has zero effect on the written SVAR.

2. **`VCF._load_index` applies `pl_filter` before normalizing `ALT`.** The
   on-disk VCF `.gvi` stores `ALT` as comma-joined `Utf8` (deliberately, for
   compression — see `_write_gvi_index`, `pl.col("ALT").list.join(",")`).
   `_load_index` filters first, then splits to `list[str]`. So list-typed
   expressions (`is_symbolic`, `is_biallelic`) — documented in `genoray.exprs`
   to operate on the in-memory `ALT: list[str]` schema — would error on the VCF
   path.

## Goals

- Delete `skip_symbolic_alts` everywhere (constructor kwarg, attribute, all
  branching). One and only one way to filter: the `filter` / `pl_filter` API.
- Make `SparseVar.from_vcf` and `from_pgen` inherit and apply the source's
  filter — the single canonical way to filter an SVAR's source.
- Keep `genoray.exprs.is_symbolic` as a convenience expression that encourages
  use of the filter API.
- Give the CLI a thin `--skip-symbolic-alts` flag (no expression API on the CLI
  yet) that constructs the appropriate filter on a VCF *or* PGEN source.

## Non-goals

- Symbolic-allele *expansion* (`<DEL>` → precise `REF→first_ref_base` via
  `INFO/END` / `SVLEN`). Tracked as future work.
- A general expression DSL for the CLI. A single flag is sufficient for now.
- Changing the default filtering behavior of `VCF` / `PGEN` (still permissive).

## Empirical findings (grounding)

- **plink2 carries symbolic alleles verbatim.** `plink2 --vcf … --make-pgen`
  on a VCF with `<DEL>` / `<INS>` writes those literal strings into the `.pvar`
  ALT column (verified locally, plink2 v2.0.0-a.7.1). So the PGEN→SVAR path is
  *not* immune; PGEN symbolic filtering is needed and `~is_symbolic` works
  natively because PGEN's index `ALT` is already `list[str]`.
- **PgenReader exposes `read_alleles_list` / `read_dosages_list`**, which read
  an arbitrary (sorted) list of physical variant indices — used to read only
  the kept variants on the filtered PGEN scan.
- **PGEN `_load_index` assigns the `index` (physical) column before filtering**,
  so a filtered `pgen._index` retains original physical indices (with gaps) —
  exactly the kept-index list the scan needs.

## Design

### 1. `genoray/_vcf.py`

- Remove the `skip_symbolic_alts` kwarg, `_skip_symbolic_alts` attribute, the
  filter-composition block (`_not_symbolic_cyvcf2`, `_combined_filter`,
  `_symbolic_only_filter`), and the symbolic-only auto-load special case.
  Restore the original auto-load condition (`self._filter is None`).
- **`_load_index`:** always normalize `ALT` (comma-`Utf8` → `list[str]`) *before*
  applying `_pl_filter`, so the in-memory schema documented in `genoray.exprs`
  holds at filter time and every expression works on the VCF path. Remove the
  symbolic-specific early-split + logging block added by PR #51.

Result: dropping symbolic alts from a VCF is just
`VCF(path, filter=lambda v: not any(a.startswith("<") for a in v.ALT), pl_filter=~exprs.is_symbolic)`
— identical in shape to any other filter.

### 2. `genoray/_svar.py` — `from_vcf`

- Delete the `skip_symbolic_alts` param and all its branching (sibling-VCF
  rebuild, conflict `ValueError`, in-place index re-materialization).
- **Index write:** replace `shutil.copy` with a lazy filter → sink that
  preserves the existing on-disk index byte-format:

  ```
  pl.scan_ipc(vcf._index_path())
    → split ALT comma-Utf8 → list[str]   (only if ALT is Utf8)
    → .filter(vcf._pl_filter)             (skip when None)
    → re-join ALT → comma-Utf8
    → .sink_ipc(cls._index_path(out), compression="zstd")
  ```

  When `pl_filter` is None this is byte-equivalent to the old `shutil.copy`.
- **Genotype scan:** pass `vcf._filter` and `vcf._pl_filter` to
  `_process_contig_vcf`, which rebuilds its worker `VCF(path, filter=…,
  pl_filter=…, with_gvi_index=False)`. `VCF.chunk` already applies the cyvcf2
  `filter` inline, so the scan and the written index agree. Filters reach joblib
  workers via cloudpickle (loky default backend).

### 3. `genoray/_svar.py` — `from_pgen`

- **Index write:** same lazy filter → sink pattern, using `pgen._filter`.
- **Genotype scan:** the worker reads only the kept physical variant indices via
  `read_alleles_list` / `read_dosages_list` (sorted `index` column from the
  filtered `pgen._index`), instead of `read_alleles_range` over a contiguous
  physical span. Output var-count then matches the filtered index.
- **Offset correctness:** physical contig boundaries must be derived from the
  **unfiltered** index (so a dropped variant at a contig edge does not skew the
  next contig's start). The filter only produces the per-contig keep-index list.
- Memory/chunking: chunk the kept-index list to honor `max_mem` the same way the
  range scan chunks today.

### 4. `genoray/_cli/__main__.py`

`write --skip-symbolic-alts` stays a thin flag that constructs the filter on the
source, then lets inheritance carry it into the SVAR:

- VCF source: `VCF(source, dosage_field=…, filter=<not-symbolic lambda>,
  pl_filter=~exprs.is_symbolic)`.
- PGEN source: `PGEN(source, filter=~exprs.is_symbolic)`.

The flag is the CLI's stand-in for the (currently absent) expression API; both
backends are supported.

### 5. `genoray/exprs.py`

Keep `is_symbolic`. Rewrite its docstring to drop the deleted
`skip_symbolic_alts` reference and show `filter` / `pl_filter` usage on both VCF
and PGEN.

### 6. `skills/genoray-api/SKILL.md`

- Remove the `skip_symbolic_alts` kwarg from the VCF / SparseVar quick-reference
  and the `from_vcf` snippet.
- Keep `is_symbolic` in the listed `genoray.exprs` (5 entries).
- Document that `SparseVar.from_vcf` / `from_pgen` inherit and apply the
  source's filter (the canonical way to filter an SVAR's source).

## Behavior change

`SparseVar.from_vcf` / `from_pgen` now honor the source filter; previously they
silently ignored it. Treated as a **bugfix** (filters were accepted but had no
effect). Documented in the commit / CHANGELOG; no deprecation cycle.

## Testing

New/updated tests (vcfixture 0.6.0 provides the symbolic-allele builder API:
`Sym`, `Seq`, `Star`, `Bnd`):

1. **VCF→SVAR inherits a general filter** — with `is_snp` and with a custom
   `filter`/`pl_filter` pair: assert the SVAR index rows and the genotype
   var-count match the filtered set, and genotypes match a vcfixture oracle.
2. **VCF→SVAR drops symbolic alts** via `pl_filter=~is_symbolic` + paired cyvcf2
   lambda; surviving genotypes intact.
3. **PGEN→SVAR inherits `~is_symbolic`** — build a PGEN with plink2 from a
   vcfixture VCF mixing SNV/indel + `<DEL>`/`<INS>`; assert symbolic dropped and
   survivors' genotypes intact.
4. **`VCF._load_index` regression** — `is_symbolic` and `is_biallelic`
   (list-typed) evaluate on the VCF path without error (would have failed under
   the old filter-then-split order).
5. **Back-compat** — no filter ⇒ all records kept, both backends (regression
   guard for the lazy-sink path being byte-equivalent to `shutil.copy`).
6. **`is_symbolic` unit test** against the `.gvi` schema.
7. **CLI** — `genoray write --skip-symbolic-alts` drops symbolic records for a
   VCF source and for a PGEN source.

Tests avoid network access; PGEN tests are skipped/guarded where plink2 is
unavailable (consistent with existing `test_svar.py` PGEN handling).

## Files touched

- `genoray/_vcf.py` — remove flag/composition; fix `_load_index` ALT order.
- `genoray/_svar.py` — `from_vcf` + `from_pgen` filter inheritance; lazy sink;
  `_process_contig_vcf` / `_process_contig_pgen` signatures.
- `genoray/_cli/__main__.py` — `--skip-symbolic-alts` constructs source filter.
- `genoray/exprs.py` — `is_symbolic` docstring.
- `skills/genoray-api/SKILL.md` — remove kwarg, document inheritance.
- `tests/test_skip_symbolic_alts.py` — rework to the filter-inheritance tests
  above (rename to `test_svar_filtering.py` if it better reflects the broadened
  scope).
- `pixi.toml` / `pixi.lock` — vcfixture 0.6.0 (already committed).

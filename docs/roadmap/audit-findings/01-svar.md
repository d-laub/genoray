# Audit: _svar.py + svar2 modules

## Summary
`_svar.py` (3,285 lines) is the health problem: a single module that is simultaneously a
region/sample normalization library, a dense↔sparse converter, two ~200-line parallel
writers, a pile of Numba kernels, a memmap I/O layer, a mutation-signature/GTF annotator,
and the 1,700-line `SparseVar` class that ties them together. The single biggest issue is
that `SparseVar` mixes three unrelated lifecycles — read/query, bulk write, and
annotation — and that `from_vcf`/`from_pgen` are near-duplicate 200-line methods. The
natural split is by lifecycle: a `_svar/` package with `_regions`, `_convert` (writers),
`_io`, `_kernels`, `_mutcat_ops` (annotation mixin), and a lean `_core` holding the class.
The svar2 trio is far healthier but leaks internal milestone jargon (M6a/b/c) into public
docstrings and passes untyped `dict[str, ndarray]` "bundles" across its public surface
where a typed result object belongs. Consistency is the weakest axis: sample-validation
boilerplate is hand-rolled 4+ times (with one latent scalar-string bug) instead of reusing
the existing `_normalize_samples`.

## Findings

### [structure] `_svar.py` should be split into a `_svar/` package along lifecycle seams
- **Location:** python/genoray/_svar.py:1-3285 (whole file)
- **Severity:** high
- **Effort:** L
- **Risk:** low (pure code motion; imports re-exported from `_svar/__init__.py`)
- **Problem:** One module holds ~10 distinct responsibilities (enumerated below). Nothing
  forces them together except history; the file is far past the point where a reader can
  hold it in their head, and the `SparseVar` class alone spans lines 479-2174.
- **Recommendation:** Create a `genoray/_svar/` package and move cohesive groups out,
  re-exporting the public `SparseVar`/`SparseVarMetadata`/`dense2sparse` from
  `_svar/__init__.py` so `genoray._svar.SparseVar` keeps working:
  - `_svar/_regions.py` — `_coerce_bed_schema`, `_normalize_regions`, `_normalize_samples`,
    `_validate_fields`, `_resolve_kept_rows`, `_resolve_kept_var_idxs` (lines 49-329).
  - `_svar/_convert.py` — `dense2sparse`, `_dense2sparse_with_length`,
    `_process_contig_vcf`, `_process_contig_pgen`, `_concat_data`, and the shared writer
    body extracted from `from_vcf`/`from_pgen` (lines 332-460, 2446-2806).
  - `_svar/_io.py` — index build/write helpers (`_write_filtered_index`,
    `_build_working_index`, `_write_index_from_working`,
    `_subset_var_idxs_and_recompute_af`) and memmap helpers (`_open_genos`, `_open_fmt`,
    `_write_genos`, `_write_dosages`) (lines 2306-2443, 2630-2687).
  - `_svar/_kernels.py` — every `@nb.njit` function (`_nb_*`, `_copy_chunk_*`,
    `_find_starts_ends*`, `_length_walk_n_keep`, `_dense2sparse_count/_fill`)
    (lines 2176-2303, 2809-3107).
  - `_svar/_annotate.py` — GTF/mutation methods as a `SparseVarAnnotateMixin`
    (`annotate_with_gtf`, `annotate_mutations`, `mutation_matrix`, `assign_signatures`)
    plus `_empty_annot`, `_get_strand_and_codon_pos`, `_load_gtf` (lines 1457-1806,
    3110-3284).
  - `_svar/_core.py` — `SparseVarMetadata`, `SparseVar` (read/query + `write_view`), which
    then inherits the annotate mixin.

### [structure] `from_vcf` and `from_pgen` are near-duplicate 200-line writers
- **Location:** python/genoray/_svar.py:978-1412
- **Severity:** high
- **Effort:** M
- **Risk:** med (shared control flow around parallel dispatch; must preserve the
  contig-block invariants each currently asserts)
- **Problem:** Both methods perform the identical sequence: overwrite check, sample
  resolution via `_normalize_samples`, `_build_working_index`, `_resolve_kept_rows` +
  sort, per-contig keep-index bucketing, metadata write, up-front index write when not
  subsetting, `parse_memory`/job sizing, `TemporaryDirectory` + `joblib.Parallel`,
  `_concat_data`, and the `subsetting_samples` MAC-drop finalize. The bodies are commented
  "(mirrors from_vcf; keep in sync)" (lines 1333, 1396) — a manual DRY liability.
- **Recommendation:** Extract a private `_write_from_reader(...)` that takes a
  per-contig task-builder callback (the only genuinely reader-specific part) plus the
  common kwargs, and have both classmethods delegate to it. Collapses ~400 lines to ~150
  and deletes the "keep in sync" hazard.

### [structure] `SparseVar` class conflates query, write, and annotation responsibilities
- **Location:** python/genoray/_svar.py:479-2174
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** A single class exposes read paths (`read_ranges`, `var_ranges`,
  `_find_starts_ends*`), a bulk writer (`write_view`, `from_vcf`, `from_pgen`), and a whole
  annotation subsystem (`annotate_with_gtf`, `annotate_mutations`, `mutation_matrix`,
  `assign_signatures`, `cache_afs`). These have disjoint state and reviewers must scroll
  past all of them to reason about any one.
- **Recommendation:** Move the annotation cluster to a `SparseVarAnnotateMixin` (see the
  decomposition finding). Keep the reader core focused on open + query + `write_view`.

### [consistency] Sample validation hand-rolled 4+ times; one copy has a scalar-string bug
- **Location:** python/genoray/_svar.py:718-723, 789-794, 856-861, 911-916
- **Severity:** high
- **Effort:** S
- **Risk:** low (the fix changes behavior only for currently-broken scalar input)
- **Problem:** The `set(samples) - set(self.available_samples)` validation is copy-pasted
  in `_find_starts_ends`, `_find_starts_ends_with_length`, `read_ranges`, and
  `read_ranges_with_length`, instead of reusing the existing `_normalize_samples` helper
  (lines 167-192). Worse, in `read_ranges_with_length` (911-916) the `set(samples)` check
  runs on the *raw* argument **before** `np.atleast_1d(np.array(...))`, whereas the other
  three convert first. A single sample passed as a bare string (e.g. `"NA001"`) is iterated
  character-by-character there, raising a spurious "Samples {...} not found". `read_ranges`
  also validates and then re-validates inside `_find_starts_ends` (double work).
- **Recommendation:** Route all four through `_normalize_samples(samples,
  self.available_samples)` (which already dedupes/preserves order and accepts str) and
  drop the inline set-difference checks.

### [consistency] `dict[str, ndarray]` "bundle" contract across the svar2 public surface
- **Location:** python/genoray/_svar2_batch.py:22-132; _svar2_decode.py:22-52
- **Severity:** med
- **Effort:** M
- **Risk:** med (return-type change is user-visible)
- **Problem:** `overlap_batch`/`read_ranges`/`find_ranges`/`gather_ranges` all return and
  consume untyped `dict[str, np.ndarray]` keyed by stringly-typed field names
  (`"vk_pos"`, `"dense_range"`, `"sample_cols"`, …). `find_ranges` even re-validates the
  `out=` dict's shapes/dtypes at runtime (batch.py:92-105) and `gather_ranges` re-parses
  `ranges["sample_cols"]` — classic "invalid states representable." The contract is only
  described in prose docstrings.
- **Recommendation:** Introduce a frozen dataclass (or `TypedDict` at minimum) for the
  batch result and the ranges bundle so field names/dtypes are checked at construction, not
  by hand at each seam. Keep a `.to_dict()` shim if the raw dict is part of the FFI contract.

### [consistency] Three different progress-bar mechanisms
- **Location:** python/genoray/_svar.py:1159 (`joblib_progress`), 1507 (`tqdm`), 2020-2039 & 2752 (`rich.progress`)
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** `from_vcf`/`from_pgen` use `joblib_progress`, `annotate_with_gtf` uses
  `tqdm`, and `write_view`/`_concat_data` use `rich.progress.Progress`. Three deps, three
  UX styles, for the same concept.
- **Recommendation:** Standardize on one (rich is already the richest). Low priority but
  systemic.

### [consistency] Python row-loop in `_resolve_kept_rows` where NumPy/Polars would vectorize
- **Location:** python/genoray/_svar.py:304-314
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** After `var_ranges` produces candidates, pos/record filtering loops in Python
  over every candidate row (`for i in range(len(cand_ids))`) doing an inner `np.any` over
  region arrays — O(candidates × regions_per_contig). This is exactly the "Python loops
  over arrays are a code smell" case the coding principles call out.
- **Recommendation:** Vectorize per contig: build region start/end arrays and use a
  broadcasted comparison or an interval-overlap join (the module already depends on
  `polars_bio`/pyranges) to compute `keep_mask` without the Python loop.

### [consistency] Stale docstrings referencing a nonexistent `self.var_table`
- **Location:** python/genoray/_svar.py:1478, 1484-1487
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** `annotate_with_gtf` says "update self.var_table in-place" and describes a
  `varID` return column, but the attribute is `self.index` (the write-back at 1554 mutates
  `self.index`). The parameter name `var_table` in `_get_strand_and_codon_pos` (3124)
  perpetuates the dead vocabulary.
- **Recommendation:** Rename doc references to `self.index`; consider renaming the helper
  param to `index` for one consistent noun.

### [consistency] Redundant runtime checks of `Literal`-typed args
- **Location:** python/genoray/_svar.py:1724-1729 (`mutation_matrix`), 1496-1503 (`annotate_with_gtf` isinstance guards)
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** `kind`/`count` are already `Literal[...]`-typed yet re-checked with
  `if kind not in (...)`, and `annotate_with_gtf` does `isinstance(level_filter, int)` /
  `isinstance(strand_encoding, dict)` guards that duplicate the type annotations. Mild
  tension with "fail fast: compile-time > runtime." These are defensible at a public
  boundary but are applied inconsistently (most other methods trust the annotation).
- **Recommendation:** Pick one policy. If runtime validation of public enums is the house
  style, apply it uniformly; otherwise drop the redundant guards and lean on the type
  checker.

### [api-hygiene] `dense2sparse` is public-named but undocumented/unexported and its sibling is private
- **Location:** python/genoray/_svar.py:332-383 vs 386-460
- **Severity:** med
- **Effort:** S
- **Risk:** low
- **Problem:** `dense2sparse` has no leading underscore (reads as intended-public) but is
  absent from `genoray.__all__` and the lazy `_LAZY` map, while the immediately following
  `_dense2sparse_with_length` *is* underscore-private. Either it is a supported utility
  (then export + document in the SKILL) or it is internal (then rename `_dense2sparse`).
  The inconsistency is confusing to any maintainer.
- **Recommendation:** Decide public vs private and make the name/exports agree with the
  decision.

### [api-hygiene] svar2 public docstrings leak internal milestone jargon and are stale
- **Location:** python/genoray/_svar2.py:32-38; _svar2_batch.py:1,28; _svar2_decode.py:1
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** The public `SparseVar2` class docstring says "M6a skeleton … Query methods
  land in M6b … and M6c" — those methods now exist, so the docstring is both stale and
  written in internal project vocabulary. Module docstrings ("M6b: raw two-channel…",
  "M6c: decoded…") and inline references ("see the M6b plan") expose sprint codes to end
  users reading `help(SparseVar2)`.
- **Recommendation:** Rewrite public docstrings in user terms (what the method returns and
  when to use it); drop milestone identifiers.

### [api-hygiene] Redundant class-level attribute annotations shadow their `cached_property`
- **Location:** python/genoray/_svar.py:503-509 vs 529-560
- **Severity:** low
- **Effort:** S
- **Risk:** low
- **Problem:** `index`, `_c_max_idxs`, and `_is_biallelic` are declared both as bare class
  annotations (503-509) and as `@cached_property` (529, 547, 554). The plain annotations
  are dead/misleading — a reader can't tell whether `index` is an eagerly-set attribute or
  lazily computed.
- **Recommendation:** Delete the duplicate bare annotations for the three names backed by
  `cached_property`.

### [api-hygiene] svar2 has overlapping/aliased public query methods
- **Location:** python/genoray/_svar2_batch.py:22-132; _svar2_decode.py:44-52
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** `read_ranges(samples=None)` is documented as "byte-identical to
  `overlap_batch`" (batch.py:62-64), so two public entry points do the same thing;
  `find_ranges`/`gather_ranges` are advanced buffer-reuse split-outs of `read_ranges` that
  most users won't need but sit at the same visibility level. `region_counts`'s docstring
  bills itself as "the simplified `SparseVar.var_ranges` replacement" yet returns *counts*,
  not ranges — vocabulary drift between the two reader classes.
- **Recommendation:** Consider demoting `find_ranges`/`gather_ranges` to underscore-private
  (or documenting them as an explicit "advanced/zero-copy" tier), fold `overlap_batch` into
  `read_ranges`, and reconcile method vocabulary across `SparseVar`/`SparseVar2`.

### [api-hygiene] svar2 mixins model host state with `Any` + `type: ignore` instead of a Protocol
- **Location:** python/genoray/_svar2_batch.py:17-19; _svar2_decode.py:20, 52
- **Severity:** low
- **Effort:** M
- **Risk:** low
- **Problem:** `_BatchQueryMixin`/`_DecodeMixin` redeclare host attributes as
  `_readers: dict[str, Any]` / `samples: list[str]` and reach `self.n_samples` behind
  `# type: ignore[missing-attribute]` (decode.py:52). The coupling between mixin and host
  is real but expressed by suppression, not by type — the "make invalid states
  unrepresentable / generics when coupled" principle points at a shared base.
- **Recommendation:** Define a `SparseVar2Host` Protocol (typed `_readers`, `samples`,
  `n_samples`, `ploidy`) and bound the mixins to it, removing the ignores.

### [consistency] `SparseVar2.from_vcf` return-value and API shape diverge from `SparseVar`
- **Location:** python/genoray/_svar2.py:58-119 vs _svar.py:978-1412
- **Severity:** low
- **Effort:** M
- **Risk:** med (aligning would be a public signature change)
- **Problem:** `SparseVar2.from_vcf` returns an `int` (dropped-ALT count) while
  `SparseVar.from_vcf`/`from_pgen` return `None`; the two reader families also differ in
  query shape (per-contig `regions` iterables vs `contig, starts, ends`). For one library's
  public surface this is a discoverability tax.
- **Recommendation:** Document the intended relationship (is `SparseVar2` a successor?) and,
  if so, converge naming/return conventions; at minimum note the divergence in the SKILL.

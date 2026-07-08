# Audit: Public API + SKILL.md/README hygiene

## Summary
The `import genoray` surface is broadly coherent and SKILL.md is impressively current for the newest subsystems (SVAR2, mutation catalogues, signatures) — it clearly gets the most maintenance attention. The real rot is in the older, human-facing docs: **README.md and `docs/source/index.md` still describe the PGEN filter API with a schema that no longer exists** (`Chromosome`/`Start`/`End`/`ilen`/`kind`, `pl.col("kind")…`), so the canonical "narrative tour" SKILL points users to contains a filter example that raises. Secondary problems are inventory gaps (public names absent from SKILL: `symbolic_ilen`, `IndexSchema`, `VCF.get_record_info`, `SparseVar.annotate_with_gtf`/`read_ranges_with_length`/`cache_afs`) and a genuine consistency wart: three different names for "the samples in this file" (`current_samples` / `available_samples` / `samples`) plus two different sample-subset mechanisms across the four reader classes. `docs/source/api.md` also autodocs only 3 of the ~8 public classes/functions. Nothing here is a correctness bug in the code; it is documentation drift plus naming inconsistency.

## Public surface inventory
- **Classes:** `VCF`, `PGEN`, `SparseVar`, `SparseVar2`, `Reference`; `Reader` = type alias `VCF | PGEN | SparseVar` (note: excludes `SparseVar2`).
- **Functions:** `cosmic_signatures`, `fit_signatures`.
- **Module:** `genoray.exprs`.
- **VCF public methods/props:** `read`, `chunk`, `set_samples`, `current_samples`, `n_samples`, `nbytes`, `n_vars_in_ranges`, `get_record_info`, `filter` (getter/setter); mode constants `Genos8`, `Genos16`, `Dosages`, `Genos8Dosages`, `Genos16Dosages`.
- **PGEN public methods/props:** `read`, `chunk`, `read_ranges`, `chunk_ranges`, `set_samples`, `current_samples`, `n_samples`, `nbytes`, `n_vars_in_ranges`, `var_idxs`, `geno_path`, `dosage_path`, `filter`; mode constants `Genos`, `Dosages`, `GenosPhasing`, `GenosDosages`, `GenosPhasingDosages`.
- **SparseVar public methods/props:** `from_vcf`, `from_pgen`, `read_ranges`, `read_ranges_with_length`, `with_fields`, `var_ranges`, `write_view`, `annotate_mutations`, `annotate_with_gtf`, `mutation_matrix`, `assign_signatures`, `cache_afs`, `available_samples`, `n_samples`, `n_variants`, `nbytes`, `index`.
- **SparseVar2 public methods/props:** `from_vcf`, `decode`, `region_counts`, `overlap_batch`, `read_ranges`, `find_ranges`, `gather_ranges`, `n_samples`, `samples`, `contigs`, `ploidy`, `format_version`.
- **Reference:** `from_path`, `fetch`, `contig_array`.
- **exprs public names:** `is_snp`, `is_indel`, `is_biallelic`, `is_symbolic`, `is_breakend`, `is_imprecise`, `ILEN`, **`symbolic_ilen`** (function), **`IndexSchema`** (constant).
- **CLI:** `genoray index`, `genoray write` (SVAR2 default) / `genoray write svar1`, `genoray view`.

## Findings

### [drift] README + docs PGEN-filter schema is entirely stale (broken example)
- **Location:** README.md:135-148; docs/source/index.md:144-157
- **Severity:** high
- **Effort:** S
- **Risk:** none
- **Problem:** Both docs say a PGEN filter expression operates on columns `Chromosome`, `Start`, `End`, `ALT`, `ilen` (indel length), `kind` ("SNP"/"INDEL"/"MNP"/"OTHER"), and give the example `PGEN("file.pgen", filter=pl.col("kind").list.eval(pl.element() == "SNP").list.all())`. The actual `.gvi` index schema (exprs.py:29-35 `IndexSchema`, and every `exprs` expression) is `CHROM` (Enum), `POS` (Int64), `REF`, `ALT` (List[Utf8]), `ILEN` (List[Int32]). There is no `Chromosome`/`Start`/`End`/`ilen`/`kind` column anymore, and `is_snp`/`is_indel` are derived from `ILEN`, not a `kind` column. The documented example raises at query time. SKILL.md:32 explicitly directs users to `docs/source/index.md` as the "narrative tour with full examples," so the drift is actively surfaced.
- **Recommendation:** Rewrite the README/index.md filter sections to the current schema and steer users to `genoray.exprs` (`is_snp`, `is_biallelic`, etc.) as the primary path, with `pl.col("CHROM"/"POS"/"REF"/"ALT"/"ILEN")` for custom predicates — matching SKILL.md's Filtering section.

### [drift] SKILL claims `exprs` "complete set (currently 7)" but two public names are missing
- **Location:** skills/genoray-api/SKILL.md:42, 328-336; exprs.py:144 (`symbolic_ilen`), exprs.py:29 (`IndexSchema`)
- **Severity:** med
- **Effort:** S
- **Risk:** low
- **Problem:** SKILL states the exprs surface is "the *complete* set … (currently 7)". But `genoray.exprs.symbolic_ilen` (a public builder function) and `genoray.exprs.IndexSchema` (a public dict constant) are both reachable without underscores and are rendered publicly by `.. automodule:: genoray.exprs` in docs/source/api.md:16. So the "complete" claim is inaccurate, and per the CLAUDE.md hard rule these public names must be either documented or made private.
- **Recommendation:** Either (a) underscore-prefix `symbolic_ilen`/`IndexSchema` if they are meant to be internal (they read as index-build internals), or (b) list them in SKILL and update the "7" count. Given `IndexSchema` is already `:exclude-members:`'d in api.md:17, leaning private is the honest signal.

### [drift] `VCF.get_record_info` is public but absent from SKILL
- **Location:** _vcf.py:1014 (overloads at 975/985/996); skills/genoray-api/SKILL.md:35
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** `VCF.get_record_info(contig, start, end, fields, info, lazy)` is a public method returning a polars DataFrame/LazyFrame of record-level annotations. SKILL's VCF pointer (line 35) lists only "constructor, `read`, `chunk`, mode constants" and never mentions it. A downstream user reaching it via `import genoray` has no skill coverage.
- **Recommendation:** Add `get_record_info` to the SKILL VCF "where to look" line (and note its return type), or confirm intent and privatize if it is not meant to be user-facing.

### [drift] SKILL SparseVar method inventory omits several public methods
- **Location:** _svar.py:885 (`read_ranges_with_length`), :1457 (`annotate_with_gtf`), :1808 (`cache_afs`); skills/genoray-api/SKILL.md:37
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** SKILL.md:37 enumerates SparseVar methods but omits `read_ranges_with_length` (length-guaranteed range read — semantically distinct from `read_ranges`), `annotate_with_gtf` (a substantial public annotation entry point with `level_filter`/`write_back` kwargs), and `cache_afs`. `annotate_with_gtf` in particular is a whole feature with zero SKILL coverage. These are all public per the CLAUDE.md rule.
- **Recommendation:** Add these to the SKILL "where to look" SparseVar list (one line each), or privatize any that are genuinely internal. At minimum `annotate_with_gtf` and `read_ranges_with_length` deserve a mention.

### [drift] docs/source/api.md autodocs only 3 of ~8 public classes/functions
- **Location:** docs/source/api.md:1-18
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** api.md renders `VCF`, `PGEN`, `SparseVar`, and `genoray.exprs`. It has no autodoc for the public `SparseVar2`, `Reference`, `cosmic_signatures`, or `fit_signatures`. The rendered API reference silently misses half the public surface, and SKILL.md:41/38 points users at these classes.
- **Recommendation:** Add `.. autoclass:: SparseVar2`, `.. autoclass:: Reference`, and `.. autofunction::` entries for the two signature functions to api.md.

### [drift] README "two classes / five methods" framing understates the surface
- **Location:** README.md:5-16
- **Severity:** med
- **Effort:** M
- **Risk:** none
- **Problem:** The README Summary says the API "boils down to just two classes and up to five methods" (VCF/PGEN + `read`/`chunk`/`read_ranges`/`chunk_ranges`/`set_samples`). The public surface now also includes `SparseVar`, `SparseVar2`, `Reference`, `exprs`, `cosmic_signatures`/`fit_signatures`, and a three-command CLI. A new reader gets a materially incomplete mental model.
- **Recommendation:** Add a short "Also included" bullet list (SparseVar/SparseVar2 sparse stores, Reference, mutation catalogues + signatures, CLI) pointing at the skill/docs, without necessarily expanding the whole README.

### [consistency] Three different names for the file's sample list across reader classes
- **Location:** _vcf.py:382 / _pgen.py:270 (`current_samples`), _svar.py:496 (`available_samples`), _svar2.py:44 (`samples`)
- **Severity:** med
- **Effort:** M
- **Risk:** med (rename is a compat break)
- **Problem:** "The samples present in this file" is spelled three ways: `VCF`/`PGEN` expose a `current_samples` **property**; `SparseVar` exposes an `available_samples` plain **attribute**; `SparseVar2` exposes a `samples` plain attribute. This forces users (and gvl) to special-case each class, and violates the "one obvious way" principle. It also drives finding below (SKILL documents only `.samples` for SVAR2).
- **Recommendation:** Converge on one name (`samples` reads cleanest and matches SVAR2/pgenlib/cyvcf2 conventions). If a compat break is unacceptable, add `samples` as an alias property on all four and document the canonical one in SKILL.

### [consistency] Two different sample-subset mechanisms (stateful `set_samples` vs per-call `samples=`)
- **Location:** _vcf.py:391 / _pgen.py:325 (`set_samples(...) -> Self`), _svar.py:827 / _svar2_batch.py:50 (`samples=` kwarg on `read_ranges`)
- **Severity:** med
- **Effort:** L
- **Risk:** med
- **Problem:** VCF/PGEN subset+reorder samples via a stateful `set_samples()` that returns `Self`; SparseVar/SparseVar2 instead take a per-call `samples=` argument on the read methods. Same user intent, two incompatible idioms. SKILL never documents `set_samples` at all (only the SVAR `samples=` kwarg), so this asymmetry is invisible to skill users.
- **Recommendation:** Pick one model. Least-disruptive: keep both but document both explicitly in SKILL and note the divergence. Ideal: support a `samples=` kwarg everywhere (or `set_samples` everywhere) so the pattern is uniform.

### [api-hygiene] `Reader` alias excludes the public `SparseVar2`
- **Location:** __init__.py:42-47, 69; skills/genoray-api/SKILL.md:18
- **Severity:** low
- **Effort:** S
- **Risk:** low
- **Problem:** `genoray.Reader` is advertised (SKILL.md:18) as the reader union but is hardcoded to `VCF | PGEN | SparseVar`, silently omitting the newest public reader `SparseVar2`. As a type alias for "any genoray reader" it is now misleading. (It may be intentional because SVAR2's query API diverges — no `read`, dict-returning `read_ranges` — in which case the alias name over-promises.)
- **Recommendation:** Either add `SparseVar2` to the union, or rename/clarify the alias (and SKILL) to state it covers only the `read`/`read_ranges`-array readers, explicitly excluding SVAR2.

### [api-hygiene] Low-level SVAR2 FFI methods are public without an underscore
- **Location:** _svar2_batch.py:22 (`overlap_batch`), :71 (`find_ranges`), :108 (`gather_ranges`); skills/genoray-api/SKILL.md:267-280
- **Severity:** low
- **Effort:** M
- **Risk:** med
- **Problem:** `overlap_batch`, `find_ranges`, and `gather_ranges` return a "raw two-channel `BatchResult` → numpy dict (`vk_pos`/`vk_key`/`vk_off`, `dense_*`, `lut_*` …)" that SKILL itself describes as "what gvl's Rust core consumes." These are FFI-seam internals exposed on the public class; they lock a low-level wire contract into the public API (per CLAUDE.md, any change now requires a SKILL update). `decode`/`region_counts`/`read_ranges` are the intended user-facing methods.
- **Recommendation:** Consider underscore-prefixing the raw-dict methods (or grouping them behind a `._raw`/`._batch` accessor) so only `decode`/`region_counts`/`read_ranges` are public, freeing the FFI dict shape to evolve without a public-API/skill churn.

### [api-hygiene] Public exprs docstring references a private method (`VCF._load_index`)
- **Location:** exprs.py:10 (module docstring), rendered by docs/source/api.md:16 `automodule`
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** The `genoray.exprs` module docstring — surfaced in the public API docs via `automodule` — says expressions are "Applicable for PGEN files and the experimental `VCF._load_index` method," naming an underscore-private method to public readers. Leaks an internal name into public documentation and dates the module (the private method may not even still exist under that name).
- **Recommendation:** Reword to describe the capability ("PGEN indexes, and VCF indexes when built") without naming the private method.

### [consistency] `--no-symbolic`/`--no-breakend` mean different things on `write` vs `write svar1`
- **Location:** _cli/__main__.py:66-67,107 (SVAR2: coupled → `skip_out_of_scope`) vs :132-133,180-185 (svar1: independent filters); skills/genoray-api/SKILL.md:298-306
- **Severity:** low
- **Effort:** M
- **Risk:** low
- **Problem:** The same two flag names are **independent** on `genoray write svar1` (each adds its own polars/record predicate) but **coupled** on the default `genoray write` (either one sets `skip_out_of_scope=True`, dropping both classes, because the SVAR2 core cannot distinguish them). Identical flag names with divergent semantics between sibling subcommands is a footgun. SKILL documents the divergence (good), but the surface itself is inconsistent.
- **Recommendation:** On SVAR2 `write`, collapse to a single honest flag (e.g. `--no-out-of-scope` / `--skip-out-of-scope`) and drop the two coupled aliases, so the flag name matches the actual granularity. Keep the independent pair only where it is truly independent (svar1).

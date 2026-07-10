# SP-6 + SP-7 — API reconciliation & small-module cleanup (genoray 3.0.0)

**Date:** 2026-07-10
**Roadmap:** [`../../roadmap/clean-code-audit.md`](../../roadmap/clean-code-audit.md) §SP-6, §SP-7
**Findings:** [`../../roadmap/audit-findings/04-api-skill.md`](../../roadmap/audit-findings/04-api-skill.md),
[`../../roadmap/audit-findings/03-mutcat-utils.md`](../../roadmap/audit-findings/03-mutcat-utils.md)

## Summary

This is **one work unit** combining the genoray public-API reconciliation (SP-6), the
small-module cleanup (SP-7), **and the coordinated `genvarloader` PR**. All of it ships
together as genoray **3.0.0** alongside the gvl SVAR2 MVP + a matching gvl bump.

The audit's SP-6 was scoped as a set of "compat decisions required." Those decisions are
now made (see below), so this spec is prescriptive rather than exploratory. SP-7 is folded
in because it is small, internal, and low-risk; keeping it in the same 3.0.0 cycle avoids a
second churn pass over the same modules.

## Decisions (settled during brainstorming)

1. **Clean breaks, no deprecation aliases.** genoray is already breaking for the SVAR2 MVP
   and releasing 3.0.0; SP-6 renames land as hard breaks in that major bump.
2. **`available_samples` is the canonical name** for "all samples in the file." Only
   `SparseVar2` diverges today (`samples`); it is renamed.
3. **The `set_samples()` (stateful) vs `samples=` (per-call) split is deliberate,
   performance-motivated design — not an inconsistency to unify.** Subsetting samples is a
   costly action for VCF/PGEN (reader re-initialization) but ~free for SparseVar/SparseVar2.
   SP-6 *documents this rationale*; it does **not** unify the idioms.
4. **Privatize the SVAR2 raw-dict FFI methods** (`overlap_batch`/`find_ranges`/`gather_ranges`).
   All three are privatized (not deleted): `gather_ranges` is exercised by genoray's own
   test suite as the reference replay half of the search/gather split.
5. **Drop the `Reader` type alias entirely.** It is neither an ABC nor a Protocol, so it
   carries no enforceable contract — dead weight that over-promises a common interface.
6. **CLI flag name: `--skip-symbolics-and-breakends`** (not `--skip-out-of-scope`;
   "out-of-scope" is opaque to naive users).
7. **The `genvarloader` PR is in scope** for this work unit and co-released.

## Current state (verified)

- Package lives under `python/genoray/`. Version is `2.15.0` (→ `3.0.0`).
- Sample accessors: `available_samples` (all samples) is used by `VCF`, `PGEN`, and
  `SparseVar` (`_vcf.py:158`, `_pgen.py:95`, `_svar/_core.py:90`); only `SparseVar2` uses
  `samples` (`_svar2.py:44`). `current_samples` (selected subset, a property) + stateful
  `set_samples()` exist **only** on VCF/PGEN (`_vcf.py:270,279`; `_pgen.py:197,252`).
  SparseVar/SparseVar2 subset via a per-call `samples=` kwarg.
- SVAR2 raw-dict methods are defined in `_svar2_batch.py`: `overlap_batch` (:55),
  `find_ranges` (:104), `gather_ranges` (:141). `read_ranges` (:83) stays public.
- `symbolic_ilen` (`exprs.py:148`) is imported internally only by `_vcf.py:35` and
  `_pgen.py:31`. `IndexSchema` (`exprs.py:29`) has no internal callers and is already
  `:exclude-members:`'d in `docs/source/api.md:17`. `dense2sparse` is already private
  (`genoray._svar`), imported by gvl via that underscore path.
- SP-7 targets are untouched: `_mutcat.py` (862 L), `_utils.py` (303 L),
  `_var_ranges.py` (250 L). The scalar classifiers
  (`classify_sbs96`/`classify_dbs78`/`classify_id83`/`_microhomology_len`/
  `_classify_variants_scalar`, `_mutcat.py:191–689`) are private to `_mutcat` (not
  re-exported in `__init__.py`).
- **gvl usage** (worktree `~/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel`):
  - `SparseVar2.find_ranges` is a **production** dependency (`_dataset/_svar2_store_py.py`
    ×3, `_dataset/_write.py:1198`).
  - `SparseVar2.overlap_batch` is used in the retained oracle adapter
    (`_dataset/_svar2_source.py:41`).
  - `SparseVar2.gather_ranges` is **not** called anywhere in gvl.
  - `genoray.Reader` is imported at `_dataset/_write.py:21` and used in the
    `write(variants: str | Path | Reader | None = ...)` signature (`_write.py:107`).
  - `dense2sparse` is imported from `genoray._svar` (private path) in gvl tests.

## Part A — genoray public API (SP-6)

**A1. SVAR2 sample accessor.** Rename `SparseVar2.samples` → `available_samples`
(`_svar2.py:44`). Update internal reads: `_svar2.py` (`n_samples`, `PyContigReader`
construction) and `_svar2_batch.py::_sample_idxs` (`:73–80`). `n_samples` keeps its name.

**A2. Privatize SVAR2 raw-dict FFI methods** (`_svar2_batch.py`):
`overlap_batch`→`_overlap_batch`, `find_ranges`→`_find_ranges`,
`gather_ranges`→`_gather_ranges`. The user-facing SVAR2 query surface is `decode`,
`region_counts`, `read_ranges`. Update genoray's own test call-sites
(`tests/test_svar2_ranges.py`, `tests/test_svar2_errors.py`,
`tests/test_py_ranges_readbound.py`, and any Rust-parity tests referencing the Python
wrappers) to the underscored names. The Rust query layer is untouched (that was SP-2).

**A3. Drop `Reader`.** Remove it from `__init__.py`: the `__all__` entry (:21), the
`__getattr__` special-case (:44–49), and the `TYPE_CHECKING` alias (:72).

**A4. Privatize exprs internals.**
- `symbolic_ilen` → `_symbolic_ilen` (`exprs.py:148`); update the two internal imports
  (`_vcf.py:35`, `_pgen.py:31`) and the `:func:`symbolic_ilen`` cross-ref (`exprs.py:112`).
- `IndexSchema` → `_IndexSchema` (`exprs.py:29`); remove the now-moot
  `:exclude-members: IndexSchema` line from `docs/source/api.md:17`.
- `dense2sparse` stays as-is (already private).

**A5. CLI flag collapse.** In `python/genoray/_cli/__main__.py`, on the svar2 `write`
command (:66–67, dispatched at :107), replace the coupled `--no-symbolic`/`--no-breakend`
flags with a single `--skip-symbolics-and-breakends` mapping to the existing
`skip_out_of_scope` kwarg. Keep the independent `--no-symbolic`/`--no-breakend` pair on
`write svar1` (:132–133), where each drives a genuinely independent filter.

## Part B — Documentation (SP-6, no API change)

**B1.** Rewrite the stale PGEN-filter example in `README.md:135–148` and
`docs/source/index.md:144–157`. The example currently references a nonexistent
`Chromosome/Start/End/ilen/kind` schema and **raises at query time**. Rewrite to the real
`.gvi` schema (`CHROM`/`POS`/`REF`/`ALT`/`ILEN`) and steer users to `genoray.exprs`
(`is_snp`, `is_biallelic`, …) as the primary path, with `pl.col("CHROM"/"POS"/…)` for
custom predicates — matching the SKILL Filtering section.

**B2.** Broaden the README Summary (`README.md:5–16`) beyond "two classes and up to five
methods" with a short "Also included" list: SparseVar/SparseVar2 (sparse stores),
Reference, mutation catalogues + signatures, and the three-command CLI. Point at the
skill/docs; do not expand the whole README.

**B3.** `docs/source/api.md`: add autodoc entries for `SparseVar2`, `Reference`, and
`autofunction` for `cosmic_signatures` and `fit_signatures` (currently only `VCF`, `PGEN`,
`SparseVar`, and `genoray.exprs` render).

**B4.** Reword the `genoray.exprs` module docstring (`exprs.py:10`, rendered publicly by
`automodule`) to describe the capability ("applicable to PGEN indexes, and VCF indexes when
built") without naming the private `VCF._load_index`.

## Part C — SKILL.md (SP-6, mandated in the same PR by CLAUDE.md)

Update `skills/genoray-api/SKILL.md`:

- Document `available_samples` as canonical across all four readers; note that VCF/PGEN
  additionally expose `current_samples` + `set_samples()`.
- **Document both subset idioms with the performance rationale**: stateful `set_samples()`
  on VCF/PGEN (subsetting is costly — reader re-init), per-call `samples=` on
  SparseVar/SparseVar2 (subsetting is ~free). This converts the apparent inconsistency into
  documented, principled design.
- Add the missing inventory: `VCF.get_record_info`, `SparseVar.annotate_with_gtf`,
  `SparseVar.read_ranges_with_length`, `SparseVar.cache_afs`.
- Fix the "exprs complete set (currently 7)" claim; the now-private
  `_symbolic_ilen`/`_IndexSchema` are removed from the public exprs surface.
- SVAR2 section: public query surface is `decode`/`region_counts`/`read_ranges`; remove the
  raw-dict `overlap_batch`/`find_ranges`/`gather_ranges` from public docs. State that the
  underscored FFI methods are an internal, gvl-only wire contract not covered by semver.
- Remove all `Reader` references. Update the CLI section to `--skip-symbolics-and-breakends`.

## Part D — Small-module cleanup (SP-7, internal / behavior-preserving)

All of Part D is internal code motion + micro-refactors over private modules. No public
name changes; the existing test suite is the gate.

**D1. `_mutcat.py` (862 L) → `_mutcat/` package** split along `codebook` (SBS96/ID83/DBS78
codebook construction), `classify` (classification logic), and `count`
(counting/matrix assembly). Re-export the public surface from `_mutcat/__init__.py` so
importers are unaffected.

**D2. Relocate the scalar mutation-catalogue oracles** out of shipped code into a test-side
oracle module (e.g. `tests/_mutcat_oracle.py`): `_classify_variants_scalar`,
`classify_sbs96`, `classify_id83`, `_microhomology_len`. These are test-only (verified: the
sole non-test caller of each is `_classify_variants_scalar` itself, which only tests call).
**`classify_dbs78` stays shipped** — it is production-live via `_build_dbs_table`
(`_mutcat.py:534`). The relocated oracle imports `classify_dbs78` back from the shipped
module. Repoint the test imports (`tests/test_mutcat.py`, `tests/test_svar_mutations.py`).

**D3. `_utils.py` (303 L) → `_io` + `_contigs`.** Split IO helpers (`variant_file_type`,
`np_to_pl_dtype`, memory parsing) from `ContigNormalizer`. Move `hap_ilens` (`_utils.py:117`)
next to the genotype/haplotype-indel-length code it belongs with.

**D4. Extract `_var_end_expr()`** shared by the three `_var_ranges.py` functions
(`var_ranges`/`var_counts`/`var_indices`).

**D5. Micro-refactors:** convert the `np_to_pl_dtype` (`_utils.py:157`) if/elif ladder to a
lookup table; annotate `variant_file_type` (`_utils.py:145`) with
`-> Literal[...] | None`; precompute `ContigNormalizer.name_to_index` in `__init__`.

## Part E — genvarloader PR (SP-6, co-released)

Landed against the resulting genoray 3.0.0 wheel; merges together with the genoray PR.

- Bump gvl's genoray dependency to 3.0.0.
- Underscore the privatized SVAR2 method calls: `sv.find_ranges` → `sv._find_ranges`,
  `sv.overlap_batch` → `sv._overlap_batch` (`_dataset/_svar2_store_py.py`,
  `_dataset/_write.py`, `_dataset/_svar2_source.py`, and any tests). `gather_ranges` is
  unused in gvl — no change needed there.
- `sv.samples` → `sv.available_samples` wherever gvl uses it (test/generate scripts).
- Drop the `Reader` import (`_dataset/_write.py:21`) and change the `write` signature
  (`_write.py:107`) `variants: str | Path | Reader | None` → the explicit union
  `str | Path | VCF | PGEN | SparseVar | SparseVar2 | None` (gvl already imports all four).
- Test genoray-side Python edits against gvl via a `PYTHONPATH=<genoray>/python` shadow
  (gvl pins a frozen genoray wheel for Python, so an editable path-dep is not in play).

## Part F — Testing, sequencing, non-goals

**Testing.** Behavior-preserving except the renames and the CLI flag change. Gates:
- genoray suite (`pixi run test`; vcfixture oracle + parity) stays green.
- gvl suite stays green against the 3.0.0 wheel.
- Update genoray tests referencing renamed names (`sv.samples`, `sv.gather_ranges`,
  `sv.find_ranges`, `sv.overlap_batch`, `symbolic_ilen`, `IndexSchema`).
- Add no new oracle behavior.

**Sequencing.**
1. **Part D** first — internal, low-risk, no cross-repo coordination.
2. **Parts A–C** — the genoray public break + docs/SKILL, all in the 3.0.0 branch.
3. **Part E** — gvl, built against the resulting wheel.
4. genoray 3.0.0 and the gvl bump merge together.

**Commit separability.** Even as one tracked work unit, keep commits separable
(SP-7 cleanup; SP-6 API; SP-6 docs/SKILL; gvl) to honor "small, reviewable PRs" as far as
the co-release allows.

**Non-goals.**
- No subset-idiom unification (the divergence is deliberate, performance-motivated).
- No Rust query-API changes (that was SP-2/SP-3).
- No new oracle behavior or correctness work beyond what the renames force.

## Public-surface change summary (for the 3.0.0 changelog)

| Before | After | Kind |
|---|---|---|
| `SparseVar2.samples` | `SparseVar2.available_samples` | rename |
| `SparseVar2.overlap_batch/find_ranges/gather_ranges` | `_overlap_batch/_find_ranges/_gather_ranges` | privatize |
| `genoray.Reader` | *(removed)* | remove |
| `genoray.exprs.symbolic_ilen` | `_symbolic_ilen` | privatize |
| `genoray.exprs.IndexSchema` | `_IndexSchema` | privatize |
| `genoray write --no-symbolic/--no-breakend` | `--skip-symbolics-and-breakends` | CLI rename |

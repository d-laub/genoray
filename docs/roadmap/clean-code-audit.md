# Clean-Code Audit & Refactor Roadmap

**Date:** 2026-07-08
**Branch:** `clean-code-audit`
**Scope:** Full-repository audit for maintainability, consistency/idiom, and public-API
hygiene — *not* primarily a correctness or test-coverage push (though latent bugs found
along the way are tracked below).

This is the high-level roadmap. Detailed, line-cited findings live in
[`audit-findings/`](./audit-findings/) (six files, one per module group).

## Method

Six read-only auditor passes over ~16,600 LOC of first-party source (8,718 Python +
7,891 Rust), each applying the maintainer's clean-code principles (make invalid states
unrepresentable; permissive inputs / specific outputs; `Result`/typed errors over
panics; DRY & YAGNI; vectorized over Python loops; "measure, don't guess"). **89 findings**
total, categorized `structure` / `consistency` / `api-hygiene` and rated by severity,
effort, and behavior-change risk.

## Overall assessment

The codebase is **fundamentally healthy**: hot paths are properly vectorized (numpy LUTs +
numba kernels, Rust with proptests), and the newest subsystems (SVAR2, mutation
signatures) are well-factored and well-tested. The problems are concentrated and
predictable:

| Theme | Where it bites | Primary driver |
|---|---|---|
| **God-file decomposition** | `_svar.py` (3,284 L), `query.rs` (1,402 L), `_mutcat.py` (856 L), `_utils.py` (321 L) | maintainability |
| **Systematic copy-paste** | `from_vcf`/`from_pgen` writers (~200 L ×2), 4 gather inner-loops, phantom-mode blocks (×5/file), mode-dispatch ladders (×5), `DenseMap`/`StreamMap`, `batch_result_to_dict` | maintainability |
| **Panic-in-library** | Rust worker paths, npy loaders, `pack_snp_key_file`, `bundle_from_dict` FFI — surface to Python as context-free `WorkerPanicked` | consistency/idiom |
| **Invalid states representable** | `SENTINELS` stringly-typed, `(bool,usize)` vs `DenseClass`, svar2 `dict[str,ndarray]` bundles, VCF paired-filter, three parallel SNP/indel enums | consistency/idiom |
| **Dead code** | `src/utils.rs` (whole file, never compiled), scalar classifier (~120 L), `gather_ranges_readbound` (test-only but public), `is_dtype`, `POLARS_V_IDX_TYPE` | maintainability |
| **Public-surface + doc drift** | 3 names for "the file's samples", README/docs PGEN-filter example that **raises**, SKILL inventory gaps, raw FFI dicts public | api-hygiene |

## Latent bugs surfaced (not the driver, but tracked)

The audit was not a correctness pass, but four real bugs turned up. They are folded into
the sub-projects noted; #0 fixes the standalone ones.

1. **VCF filter silently ignored** — `_fill_genos_and_dosages` (`_vcf.py:1416`) omits
   `self._filter` on the no-index path, unlike its genos-only/dosages-only siblings. A
   filtered `Genos*Dosages` read with no `.gvi` returns unfiltered data. *(→ SP-0 / SP-4)*
2. **Scalar-string sample crash** — `read_ranges_with_length` (`_svar.py:911`) runs
   `set(samples)` on the raw argument before array coercion, so a single sample passed as
   a bare `str` iterates characters and raises spurious "not found". *(→ SP-1, via
   `_normalize_samples` routing; SP-0 may hotfix)*
3. **PGEN `__del__` double-close** — when `dosage_path is None`, `_dose_pgen is
   _geno_pgen`; `__del__` closes the same handle twice. *(→ SP-0)*
4. **nrvk wrong assert bound** — `push_long_allele` message says `4,294,967,295` (2³²−1)
   but the check is `2³¹−1`; plus a dead defensive mask. *(→ SP-0)*

## Sub-project roadmap

Each sub-project is an independently-shippable spec → plan → PR, **behavior-preserving
unless explicitly noted**. Any change to a public name reachable via `import genoray`
without underscores carries a mandatory `skills/genoray-api/SKILL.md` update in the same PR
(per `CLAUDE.md`). Ordered by recommended sequence.

### SP-0 — Quick wins  ·  size S  ·  risk low  ·  **first**  ·  *spec written*
De-risk and shrink the surface before the big decompositions.
See [`../superpowers/specs/2026-07-08-sp0-quick-wins-design.md`](../superpowers/specs/2026-07-08-sp0-quick-wins-design.md).
- Delete dead code: `src/utils.rs` (whole file), `is_dtype`, `POLARS_V_IDX_TYPE`, stale
  `#[allow(dead_code)]` on `PyContigReader::inner`, dead bare annotations shadowing
  `cached_property`.
  - **Correction (found during SP-0 design):** the scalar classifier cluster
    (`_classify_variants_scalar` + `classify_sbs96`/`classify_id83`/`_microhomology_len`) is
    **not dead** — it is a live test oracle (`tests/test_mutcat.py`,
    `tests/test_svar_mutations.py`). Reassigned to **SP-7** as a *relocate-to-tests* task,
    not a deletion.
- Fix the 4 latent bugs above (with regression tests).
- Fix broken docs: README.md + `docs/source/index.md` PGEN-filter example that references
  the nonexistent `Chromosome/Start/End/ilen/kind` schema and raises at query time.
- Trivial hygiene: `TypeGuard[...Genos8...]` → `Genos16` typo (`_vcf.py:140`),
  "not found in VCF file" wrong-backend log strings in PGEN, orphaned floating docstring
  (`_vcf.py:36`), nrvk assert message.

### SP-1 — `_svar.py` → `_svar/` package  ·  size L  ·  risk low  ·  **flagship**
Split the 3,284-line module along lifecycle seams (`_regions`, `_convert`, `_io`,
`_kernels`, `_annotate` mixin, lean `_core`), re-exporting the public surface from
`_svar/__init__.py`. Extract the shared `_write_from_reader` collapsing the two ~200-line
`from_vcf`/`from_pgen` writers. Route all four hand-rolled sample-validation sites through
the existing `_normalize_samples` (fixes bug #2). Pure code-motion + re-export shim.

### SP-2 — `query.rs` → `query/` module + gather DRY  ·  size M–L  ·  risk med
Split into `query/{sidecar,reader,union,decode,gather}.rs`. Extract the 4× copy-pasted
gather inner loops behind a `PresenceBitWriter` + `gather_vk` helper. Fold the 8-arg
`gather_haps_readbound` into a params struct. Demote oracle-only entry points
(`gather_ranges_readbound`, `overlap_sample`, `decode_keyref_pub`) to a `testing`/`oracle`
submodule. **Coordinate the public-name/oracle renames with the `genvarloader` downstream**
(it names some of these in test oracles).

### SP-3 — Rust panics → typed errors  ·  size L  ·  risk low (behavior improves)
Thread `Result<_, ConversionError>` / a new `QueryError` through the worker paths
(`vcf_reader`, `nrvk`, `merge`, `writer`, `dense_merge`), the npy loaders and
`pack_snp_key_file`, and the `bundle_from_dict` FFI parser. Promote user-recoverable
validations (missing contig/sample, REF mismatch, missing index) to a distinct
`Input`/`InvalidData` variant so Python sees the real message instead of `WorkerPanicked`.
Needs an error-taxonomy design step.

### SP-4 — VCF/PGEN consistency + boilerplate  ·  size M–L  ·  risk med
Extract a shared phantom-mode factory (`make_array_mode`), a mode→method dispatch `dict`,
`_extract_dosage`, `_norm_or_warn`/`_empty` helpers. Unify the `empty(...)` signature to the
documented `(n_samples, ploidy, n_variants)` contract. Introduce a `Filter` value object so
the "both-or-neither" paired filter is unrepresentable-when-invalid. **Decide** the intended
common public surface for the range API (`read_ranges`/`chunk_ranges`/`var_idxs` are
PGEN-only today). Absorbs bug #1. Touches public surface → SKILL update.

### SP-5 — "Invalid states unrepresentable" type pass  ·  size M  ·  risk low
Python: `SENTINELS` dict → `IntEnum`; one shared `Kind` `Literal` owned by `_mutcat`;
svar2 `dict[str,ndarray]` bundles → frozen dataclass; dedup the `DTYPE` TypeVar. Rust:
`(bool,usize)` dense source → `DenseClass`; `(usize,usize)` ranges → `Range`/newtype;
unify `DenseMap`/`StreamMap` behind one `EnumKey`/`EnumMap`; give `StreamTag` a `class()`
bridge; name the svar2-codec bit-shift magic numbers. Splittable into Python and Rust PRs.

### SP-6 — Public API + SKILL/README/docs reconciliation  ·  size M  ·  risk med
Converge the three sample-accessor names (`current_samples`/`available_samples`/`samples`)
and the two subset idioms (`set_samples()` vs per-call `samples=`) — compat decisions
required. Fix the `Reader` alias (excludes `SparseVar2`). Privatize the raw-dict SVAR2 FFI
methods (`overlap_batch`/`find_ranges`/`gather_ranges`) or group behind `._raw`. Collapse
the coupled `--no-symbolic`/`--no-breakend` SVAR2 CLI flags to one honest flag. Close SKILL
inventory gaps (`get_record_info`, `annotate_with_gtf`, `read_ranges_with_length`,
`cache_afs`) and the `api.md` autodoc gaps (`SparseVar2`, `Reference`, signature fns).
Decide public-vs-private for `symbolic_ilen`/`IndexSchema`/`dense2sparse`.

### SP-7 — Small-module cleanup  ·  size M  ·  risk low
Split `_mutcat.py` → `codebook`/`classify`/`count`; split `_utils.py` → `_io`/`_contigs`
(+ move `hap_ilens` next to genotype code); extract the shared `_var_end_expr()` from the
three `_var_ranges` functions; convert `np_to_pl_dtype` if/elif ladder to a lookup table;
annotate `variant_file_type -> Literal[...] | None`; precompute `ContigNormalizer`'s
`name_to_index`. **Relocate the scalar mutation-catalogue oracles**
(`_classify_variants_scalar`, `classify_sbs96`/`classify_id83`/`classify_dbs78`/
`_microhomology_len`) out of the shipped `_mutcat.py` into a test-side oracle module —
they are used only by `tests/` to validate the vectorized path (reassigned here from SP-0).

## Sequencing & dependencies

```
SP-0 (quick wins) ──► SP-1 (_svar.py)      ──┐
                 └──► SP-7 (small modules) ──┤  Python track
                      SP-4 (vcf/pgen) ───────┤
                                             ├─► SP-6 (API/SKILL reconciliation)
SP-2 (query.rs) ─────► SP-3 (rust errors) ──┤  Rust track
                       SP-5 (types pass) ────┘
```

- **SP-0 first** — cheapest, highest signal, removes dead code and doc traps that would
  otherwise confuse later work; fixes standalone bugs.
- **SP-6 last (of the majors)** — it ratifies public-surface decisions that SP-1/SP-4 will
  have surfaced, so it benefits from going after them.
- Python and Rust tracks are largely independent and can proceed in parallel.
- Each sub-project gets its own brainstorm → spec (`docs/superpowers/specs/`) → plan →
  implementation cycle.

## Invariants for every sub-project

1. **Behavior-preserving**, gated on the existing test suite (vcfixture oracle + parity
   tests) staying green — except where a finding explicitly scopes a behavior change (the
   4 bug fixes; SP-3's panic→error; any SP-6 compat break).
2. **No public-API change** except where an `api-hygiene`/SP-6 item scopes one — and any
   such change updates `skills/genoray-api/SKILL.md` in the same PR.
3. **Small, reviewable PRs** — no god-branch.

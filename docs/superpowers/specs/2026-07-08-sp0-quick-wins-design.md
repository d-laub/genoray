# SP-0 — Quick Wins: dead-code removal, latent-bug fixes, doc hygiene

**Date:** 2026-07-08
**Branch:** `clean-code-audit`
**Roadmap:** [`docs/roadmap/clean-code-audit.md`](../../roadmap/clean-code-audit.md) — sub-project #0
**Size:** S · **Risk:** low

## Goal

The first, de-risking pass of the clean-code refactor. Delete verified-dead code, fix the
four latent bugs the audit surfaced (with regression tests), and repair documentation that
is actively wrong — before the larger structural sub-projects begin. This shrinks the
surface area and removes traps (dead files, a doc example that raises) that would otherwise
mislead later work.

Everything here is either a **pure deletion of verified-dead code**, a **narrowly-scoped
bug fix with a regression test**, or a **documentation/string correction with no runtime
effect**. No structural decomposition, no public-surface redesign (those are SP-1, SP-4,
SP-6).

## Non-goals

- Splitting or restructuring any module (SP-1, SP-2, SP-7).
- Relocating the mutation-catalogue test oracles (see "Scope correction" — reassigned to
  SP-7).
- Any public-API rename or privatization (SP-6). The `TypeGuard` and log-string fixes here
  do not change any public name.

## Scope correction discovered during design

The audit's finding "dead scalar classification path in `_mutcat.py`" (finding 03-#1) was
based on a grep of `python/` only. Re-grepping `tests/` shows `_classify_variants_scalar`,
`classify_sbs96`, `classify_id83`, and `classify_dbs78` are **live test oracles**:
`tests/test_mutcat.py:499,520` do `exp = _classify_variants_scalar(index, ref)` to validate
the shipped vectorized `classify_variants`, and `tests/test_svar_mutations.py:473-493` /
many `test_mutcat.py` sites use the per-variant classifiers as ground truth. Deleting them
would delete real test coverage. This is test-scaffolding shipping inside a public module —
a *relocate-the-oracle* task (parallel to the Rust `_pub`/oracle-fn demotion in SP-2), **not
a deletion**. It is therefore removed from SP-0 and reassigned to **SP-7**. The roadmap is
updated accordingly.

This is exactly why the invariant "re-grep `python/` **and** `tests/` for callers before any
deletion" (below) is mandatory.

## Work items

Grouped by the conventional-commit they belong to. Order: deletions → bug fixes → docs.
Each deletion item names the verification that proves it is dead.

### A. Pure deletions — `refactor: remove dead code`

| Item | Location | Proof-of-death (re-verify before deleting) |
|---|---|---|
| A1. Delete `src/utils.rs` entirely | `src/utils.rs` (whole file) | `grep -rn "mod utils" src/` → no hit (never compiled); `grep -rn "ravel!\|unravel!" src/` → only the `macro_rules!` defs |
| A2. Delete `is_dtype` | `python/genoray/_utils.py:80-95` | `grep -rn "is_dtype" python/ tests/` → only the `def`; not in any `__all__` / re-export |
| A3. Delete `POLARS_V_IDX_TYPE` | `python/genoray/_types.py:9` | `grep -rn "POLARS_V_IDX_TYPE" python/ tests/` → only the def |
| A4. Remove stale `#[allow(dead_code)]` + outdated comment on `inner` | `src/py_query.rs:13-17` | `inner` is read by `py_query_batch/decode/ranges` (`grep -rn "self.inner" src/`) — field is live, the allow is obsolete |
| A5. Delete the three bare annotations shadowing `cached_property` | `python/genoray/_svar.py` — the bare `index`, `_c_max_idxs`, `_is_biallelic` annotations (~503-509) that duplicate the `@cached_property` defs (~529/547/554) | Each name has a `@cached_property` descriptor; the bare annotation is a redundant type-only entry. **Preserve** the richer docstring currently on the bare `index` annotation by moving it onto the `index` property. Do NOT touch the other bare annotations (`genos`, `available_fields`, `fields`, `_c_norm`, `_s2i`) — those are real `__init__`-set attributes. |

A1–A4 are behavior-preserving by construction (deleting never-referenced code). A5 removes
type-only annotations that create no attribute; behavior-preserving.

### B. Bug fixes — `fix: ...` (each: failing regression test first, then fix)

**B1. VCF filter ignored on the no-index `Genos*Dosages` path**
- **Location:** `python/genoray/_vcf.py` — `_fill_genos_and_dosages` (~1416). Compare
  `_fill_genos` (filters at 1286-1287, before the `out is None`/`out is not None` split) and
  `_fill_dosages` (filters at ~1352-1353). `_fill_genos_and_dosages` applies the filter only
  inside the `out is not None` branch.
- **Test first:** with a `VCF` that has a `filter` set and **no `.gvi` index loaded**
  (so `n_variants is None` → `out is None`), assert that `read(..., mode=Genos8Dosages)`
  returns the *filtered* variant set, matching `mode=Genos8` and `mode=Dosages` on the same
  query. Test must fail on current code.
- **Fix:** hoist `if self._filter is not None: vcf = filter(self._filter, vcf)` to the top
  of `_fill_genos_and_dosages`, before the `out is None` branch, mirroring the siblings.
- **Behavior change:** yes, and intended — the mode now honors the filter it silently
  dropped. Note in CHANGELOG.

**B2. Scalar-string sample crash in `read_ranges_with_length`**
- **Location:** `python/genoray/_svar.py:911-916` — `set(samples)` runs on the raw argument
  *before* `np.atleast_1d(np.array(...))`, so a single sample passed as a bare `str`
  (`"NA001"`) is iterated character-by-character and raises a spurious
  "Samples {...} not found". The sibling methods (`_find_starts_ends`,
  `_find_starts_ends_with_length`, `read_ranges`) coerce to an array first.
- **Test first:** call `read_ranges_with_length(contig, start, end, samples="NA00001")`
  (a real single sample name as a bare `str`) and assert it succeeds and returns that one
  sample. Fails on current code.
- **Fix (minimal hotfix):** coerce `samples` to the array form before the membership check
  in this one method, matching the other three sites. **Do not** refactor to
  `_normalize_samples` here — that convergence is SP-1's job; keep this a one-site fix so
  the two sub-projects don't collide. The regression test is the durable artifact and
  survives SP-1.

**B3. PGEN `__del__` double-closes the reader**
- **Location:** `python/genoray/_pgen.py` — `_dose_pgen = self._geno_pgen` when
  `dosage_path is None` (lines 242, 386); `__del__` (389-393) calls `.close()` on
  `_geno_pgen` then `_dose_pgen` — the same object.
- **Test first:** construct a `PGEN` with no separate `dosage_path`, trigger `__del__`
  (`del` + `gc.collect()`), and assert no error / `close` called once on the shared handle
  (spy/mock on the pgenlib reader's `close`). Fails (double close) on current code.
- **Fix:** guard the second close: `if self._dose_pgen is not self._geno_pgen:
  self._dose_pgen.close()`.

**B4. nrvk wrong assert bound + dead mask**
- **Location:** `src/nrvk.rs:41-42` (assert `row_index <= 0x7FFFFFFF`, message says the wrong
  number `4,294,967,295` = 2³²−1) and `:62` (`current_index & 0x7FFFFFFF`, dead — the assert
  already guarantees the high bit is clear).
- **Test first:** none needed for the message text; add/confirm a Rust unit test (run with
  `cargo test --no-default-features`) that `push_long_allele` returns the raw
  `current_index` unchanged for a representative value, guarding against the mask removal
  regressing.
- **Fix:** correct the message to `2,147,483,647` (2³¹−1); return `current_index` directly,
  dropping the `& 0x7FFFFFFF`.

### C. Documentation & string hygiene — `docs: ...` / `fix: ...`, no runtime effect

**C1. Fix the broken PGEN-filter example in README and docs**
- **Location:** `README.md:135-148` and `docs/source/index.md:144-157`. Both document a
  filter over columns `Chromosome/Start/End/ALT/ilen/kind` and the example
  `PGEN("file.pgen", filter=pl.col("kind")...)`. That schema does not exist; the real `.gvi`
  schema (`exprs.py` `IndexSchema`) is `CHROM` (Enum), `POS` (Int64), `REF`, `ALT`
  (List[Utf8]), `ILEN` (List[Int32]). The example raises at query time.
- **Fix:** rewrite both sections to the current schema, presenting `genoray.exprs`
  (`is_snp`, `is_indel`, `is_biallelic`, …) as the primary path and
  `pl.col("CHROM"/"POS"/"REF"/"ALT"/"ILEN")` for custom predicates — matching SKILL.md's
  Filtering section. Keep the examples runnable (verify against the real schema).

**C2. `TypeGuard` copy-paste typo**
- **Location:** `python/genoray/_vcf.py:140` — `_is_genos16_dosages` is annotated
  `TypeGuard[tuple[Genos8, Dosages]]`; should be `Genos16`.
- **Fix:** change the annotation to `Genos16`. This is a static-typing annotation only —
  confirm the predicate *body* already checks the correct dtype (no runtime change). Not a
  public name → no SKILL update.

**C3. Wrong-backend log messages in PGEN**
- **Location:** `python/genoray/_pgen.py:566,651,734,861` — all log
  `"...not found in VCF file..."` inside the PGEN backend.
- **Fix:** say "PGEN file" (or make backend-agnostic). Log-string only.

**C4. Orphaned floating docstring**
- **Location:** `python/genoray/_vcf.py:36-37` — a bare triple-quoted string describing
  "int64 ... CSI indexes" that documents nothing and contradicts the adjacent
  `V_IDX_TYPE = np.uint32`.
- **Fix:** remove it (or, if it was meant as `V_IDX_TYPE`'s doc, correct and attach it —
  but the content is stale, so removal is preferred).

## Invariants & constraints

1. **Re-verify every deletion is dead** with a fresh `grep` across **both** `python/` and
   `tests/` (and `src/` for Rust) at implementation time — do not trust the audit's grep.
   This spec was itself corrected once by this rule (scalar classifier).
2. **TDD for all four bug fixes**: the failing regression test is written and observed to
   fail *before* the fix.
3. **No public-API surface change.** Nothing here renames, removes, or alters a public name
   reachable via `import genoray` without underscores, so **no `SKILL.md` change is
   required**. (`is_dtype`/`POLARS_V_IDX_TYPE` are unprefixed but unexported and unused; the
   `TypeGuard`/log fixes are internal.) Confirm this holds at the end.
4. **Behavior-preserving except B1, B2, B4-mask** — all intended and covered by tests;
   record B1 (filter now applied) in `CHANGELOG.md`.
5. **Small, reviewable commits** following Conventional Commits, grouped A/B/C.

## Verification

All must pass before the PR is considered done:

- `pixi run pytest` — full Python suite green (includes the new B1/B2/B3 regression tests).
- `cargo test --no-default-features` — Rust suite green (the `--no-default-features` flag is
  required or the pyo3 test binary fails to link with `undefined symbol: _Py_Dealloc`).
- Pre-commit/pre-push hooks green: `ruff check`, `ruff format`, `cargo fmt`, `cargo check`,
  `cargo clippy`, `pyrefly` type-check, `commitizen`.
- Docs build (or at least the edited examples are executed once against a real PGEN/`.gvi`
  to confirm they no longer raise).
- Final `grep` confirming the deleted names have zero remaining references.

## Out of scope / follow-ups created

- **SP-7** gains: relocate the mutation-catalogue scalar oracles
  (`_classify_variants_scalar`, `classify_sbs96`, `classify_id83`, `_microhomology_len`,
  and the still-live `classify_dbs78`) out of the shipped `_mutcat.py` into a test-side
  oracle module, so they stop shipping on the public surface while preserving the coverage.

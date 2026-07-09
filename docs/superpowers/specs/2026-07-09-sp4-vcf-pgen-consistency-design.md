# SP-4 — VCF/PGEN consistency + boilerplate

**Date:** 2026-07-09
**Branch:** `sp4-vcf-pgen-consistency` (off `main`)
**Roadmap:** [`docs/roadmap/clean-code-audit.md`](../../roadmap/clean-code-audit.md) — sub-project #4
**Findings:** [`docs/roadmap/audit-findings/02-vcf-pgen.md`](../../roadmap/audit-findings/02-vcf-pgen.md)
**Size:** M–L · **Risk:** med · **Release:** part of the **genoray 3.0.0** breaking-change wave

## Goal

Collapse the parallel-but-divergent `_vcf.py` / `_pgen.py` backends onto shared helpers,
and make the invalid states the audit surfaced unrepresentable. The two files share a real
conceptual core (phantom modes, mode dispatch, dosage extraction, contig-normalize + empty
results, per-variant memory math, the extend-to-haplotype-length algorithm) but implement
it twice with copy-paste and even divergent public surfaces.

This lands as **breaking changes in the 3.0.0 wave** (SP-4 / SP-5 / SP-6). Because the
release is explicitly breaking, we do **not** carry compat shims for the API changes here —
we make the clean break and update `SKILL.md` + `CHANGELOG.md` in the same PR.

### Scope correction vs. the roadmap

Two roadmap assumptions changed after inspecting the current tree:

1. **Bug #1 (filter ignored on the no-index `Genos*Dosages` path) is already fixed** — it
   landed in SP-0 (`3c6188f fix(vcf): apply filter on no-index Genos*Dosages read path`).
   SP-4 does **not** re-do it; it only inherits the now-consistent filter behavior.
2. **The range-API asymmetry is already a decided, measured call**, not accidental
   divergence: `SKILL.md` documents "VCF intentionally has **no `read_ranges`** —
   benchmarking showed no throughput benefit." SP-4 therefore does **not** add
   `read_ranges`/`chunk_ranges`/`var_idxs` to VCF; it keeps VCF streaming-only and keeps the
   doc note (see §C).

## Non-goals

- Adding multi-range APIs to VCF (`read_ranges`/`chunk_ranges`/`var_idxs`) — measured
  PGEN/SVAR-only decision stands (§C).
- Sample-accessor name convergence, `Reader` alias, raw-dict FFI privatization — those are
  **SP-6**.
- Any Rust change — SP-4 is Python + the coordinated gvl edit.
- Hand-bumping the version. Breaking commits are marked with `!` / `BREAKING CHANGE:` so
  `cz bump` computes 3.0.0 once the wave completes (see §Process).

## Part A — Internal dedup (behavior-preserving, no SKILL change)

### A1. Phantom-mode factory — new `genoray/_modes.py`

Every phantom mode today repeats a three-part pattern (an `_is_*` `TypeGuard`, the
`NDArray[...], Phantom` subclass, an `empty` classmethod) — ~120 L in `_vcf.py`
(`Genos8`/`Genos16`/`Dosages`/`Genos8Dosages`/`Genos16Dosages`) and ~110 L in `_pgen.py`
(`Genos`/`Dosages`/`Phasing`/`GenosPhasing`/`GenosDosages`/`GenosPhasingDosages`), almost
all mechanical; two docstrings are copy-pasted verbatim.

Provide two factories in `genoray/_modes.py`:

- `make_array_mode(name, dtype, ndim, shape_pred=None) -> type[Phantom]` — generates the
  predicate (`isinstance(np.ndarray)` + dtype + ndim + optional `shape_pred`, e.g.
  `shape[1] in (2, 3)` for genotypes), the `NDArray[dtype], Phantom` subclass carrying the
  dtype attributes the callers rely on (`_gdtype` for VCF genos, `_dtype`/`_dtypes` for
  PGEN — unify the attribute name; see A5), a uniform
  `empty(n_samples, ploidy, n_variants)` (§B1), and a `nbytes_per_variant(n_samples, ploidy)`
  (§A5).
- `make_tuple_mode(name, *component_modes) -> type[Phantom]` — the `(genos, dosages)` /
  `(genos, phasing, dosages)` tuple modes: generates the per-element isinstance predicate
  and an `empty` that delegates to the component modes.

`_vcf.py` and `_pgen.py` then declare their modes via these factories. The public names
(`Genos8`, `Dosages`, `PGEN.Genos`, …) are unchanged — only the definition mechanism moves.

### A2. Mode → method dispatch dict

The `if issubclass(mode, Genos): … elif Dosages … elif GenosPhasing …` ladder appears 5×
in `_pgen.py` (`read`, `chunk`, `read_ranges`, `chunk_ranges`, `_chunk_ranges_with_length`)
and twice in `_vcf.py` (`read`, duplicated across the `out is None` / `out is not None`
branches). Build one `dict[type[Phantom], Callable]` per backend (mapping mode →
`self._read_*`), look up `reader = table[mode]`. The with-length variant (which omits
`Dosages`) filters the table. In VCF, collapse the two `read` branches so the dispatch is
written once and `out` is threaded through.

### A3. VCF `_extract_dosage` helper

`d = v.format(field); if d is None: raise DosageFieldError; if d.shape[1] > 1: raise
MultiallelicDosageError; … d.squeeze(1)` is copy-pasted 5× (`chunk`, `_fill_dosages` ×2,
`_fill_genos_and_dosages` ×2, `_ext_genos_dosages_with_length`) with identical messages.
Extract `_extract_dosage(self, v) -> NDArray[np.float32]` performing fetch + both checks +
squeeze; all sites call it.

### A4. Contig-normalize + empty-result helpers

Each range/chunk method opens with `c = norm(contig); if c is None: warn; yield/return
empty`, and most repeat a `n_variants == 0` empty block with the identical
`(mode.empty(...) for _ in range(1))` generator. Add small private helpers —
`_norm_or_warn(contig) -> str | None`, `_empty(mode, …)`, `_empty_gen(mode, …)` — and route
the ~dozen near-identical blocks per file through them. (The wrong-backend "not found in VCF
file" log strings inside PGEN were already corrected in SP-0.)

### A5. `_mem_per_variant` unification

Both backends sum per-array `n_samples * axis * itemsize`, but read the dtype from
different attributes (`mode._gdtype` vs `mode._dtype`/`mode._dtypes`) and **only PGEN**
doubles the estimate when a sample sorter is active (`_s_unsorter` is an ndarray). Have each
mode expose `nbytes_per_variant(n_samples, ploidy)` (natural once the factory exists) so
`_mem_per_variant` becomes a one-liner. **Deliberate decision:** unify the sample-copy
doubling to apply to **both** backends when a sorter is active — it is a chunk-sizing-only
estimate (more conservative memory bound, no correctness/output effect) and the divergence
was accidental. Unify the mode dtype attribute name while here (A1).

### A6. Merge the two `_ext_*_with_length` methods

`_ext_genos_with_length` and `_ext_genos_dosages_with_length` share the entire control
structure (contig coord build, warnings filter, per-variant loop, `v.start < ext_start`
skip, indel `hap_lens` update, `_CHECK_LEN_EVERY_N` break, trailing `last_end` fixup),
differing only in whether dosages are collected; `_CHECK_LEN_EVERY_N = 20` is redefined in
each. Merge into one method parameterized by `want_dosages: bool`, reusing `_extract_dosage`,
with a single module-level `_CHECK_LEN_EVERY_N`.

### A7. Relocate `_oxbow_reader`

`_oxbow_reader` sits interleaved between the `get_record_info` `@overload` stubs and the
concrete implementation. Move it out of the overload block (below `get_record_info`) so the
stubs sit directly above their implementation. Pure code motion.

## Part B — Breaking changes (SKILL + CHANGELOG)

### B1. `empty()` signature unified to `(n_samples, ploidy, n_variants)`

CLAUDE.md documents "All `Phantom` subclasses have `empty(n_samples, ploidy, n_variants)`,"
but VCF's variants take a 4th `phasing` arg PGEN's do not, and `Dosages.empty` accepts+
ignores `ploidy`/`phasing`. Standardize on the documented 3-arg signature: the VCF phasing
axis (`ploidy + phasing`) is computed by the **caller** and passed as effective ploidy. The
factory (A1) generates this uniform signature. `empty()` is not in `SKILL.md`, but note the
signature change in `CHANGELOG.md` (breaking) and confirm the CLAUDE.md contract now holds
verbatim.

### B2. `Filter` value object — clean Filter-only break (public)

The paired `(filter, pl_filter)` is really one concept (a cyvcf2 record predicate with its
matching `.gvi` polars expression) whose both-or-neither invariant is enforced at runtime by
`_check_filter_pair` at three sites plus a tuple-shape check in the setter.

Introduce a frozen `Filter(record: Callable[[cyvcf2.Variant], bool], expr: pl.Expr)`
(dataclass, `frozen=True`, exported from `genoray`). **Clean break, no tuple compat shim**
(confirmed with maintainer):

- Constructor: `VCF(path, filter: Filter | None = None, …)` — the separate `filter=` /
  `pl_filter=` kwargs are **removed**.
- Property: `vcf.filter` getter/setter is `Filter | None`.
- `_check_filter_pair` is **deleted** — the half-None state is now unrepresentable by
  construction. `self._filter: Filter | None` replaces the `_filter` / `_pl_filter` pair;
  internal reads use `self._filter.record` / `self._filter.expr`.

Rewrite the `SKILL.md` Filtering section (currently documents the `(filter, pl_filter)`
tuple getter/setter at lines ~323–334 and constructor kwargs) to the `Filter` API. Record
the break in `CHANGELOG.md`. Migration note: `VCF(p, filter=fn, pl_filter=expr)` →
`VCF(p, filter=Filter(record=fn, expr=expr))`.

### B3. `_chunk_ranges_with_length` tuple reconciled (coordinated with gvl)

Two sibling methods with the same name return structurally different third tuple elements:
VCF yields `n_extension_vars: int`, PGEN yields `chunk_idxs: NDArray[V_IDX_TYPE]`. A single
downstream consumer — gvl `_dataset/_write.py` — calls both (`vcf._chunk_ranges_with_length`
at :721, `pgen._chunk_ranges_with_length` at :837) and must branch per backend.

Reconcile to **one contract**: `(data, end, chunk_idxs: NDArray[V_IDX_TYPE])` on both
backends (`n_extension_vars` is derivable from `chunk_idxs.shape[0]` if still needed).

**Implementation-risk spike (resolve before committing the exact semantics):** VCF's
streaming with-length path currently produces a *count*, not a global variant-index array.
Emitting `chunk_idxs` may require the `.gvi`-index-backed path, or synthesizing indices as
variants stream. Before locking the array semantics, confirm what gvl's write path actually
consumes from that third element — if gvl only needs a count / a contiguous index range, the
"richer" contract may be a contiguous `chunk_idxs` the VCF path can produce cheaply. The
plan opens with this spike; the reconciled shape is whatever satisfies both (a) PGEN's
existing `chunk_idxs` semantics and (b) gvl's actual use, chosen to be cheaply producible on
the VCF path.

**gvl coordination (same effort, must land together):**
- Update `python/genvarloader/_dataset/_write.py` (:721, :837) to consume the unified
  `(data, end, chunk_idxs)` from both backends; collapse the per-backend branching.
- Bump the genoray pin `genoray>=2.12.3,<3` → `genoray>=3,<4` in gvl `pyproject.toml`.
- Commit on the existing branch **`svar2-m6b-kernel`** and push to **draft PR #266**
  ("SVAR2 read-bound dataset wiring", currently *blocked on genoray svar-2 release*).
- gvl's test suite on that branch must pass against a local genoray build of this branch.

## Part C — Deliberate "leave asymmetric, document" calls

- **C1. Range API stays PGEN/SVAR-only.** `read_ranges`/`chunk_ranges`/`var_idxs` are a
  measured decision (no VCF throughput benefit, per SKILL). VCF keeps `_var_idxs` private.
  No implementation; keep/verify the SKILL doc note.
- **C2. `read(out=...)` stays VCF-only.** PGEN random-access allocates fresh; adding `out=`
  is unmotivated (YAGNI). Document the asymmetry as intentional (a one-line SKILL note),
  mirroring the range-API rationale.

## Process, PRs, versioning

- **Commits:** small and reviewable, grouped A → B, Conventional Commits. Behavior-
  preserving parts (A) are plain `refactor:`; API breaks (B1/B2/B3) use `!` / a
  `BREAKING CHANGE:` footer so `cz bump` resolves the wave to **3.0.0**.
- **Do NOT hand-bump the version.** 3.0.0 is cut once the SP-4/5/6 wave lands; SP-4 only
  contributes the breaking-commit markers.
- **Branches:** genoray work on `sp4-vcf-pgen-consistency`; the coordinated gvl edit on
  `svar2-m6b-kernel` (→ PR #266). Ensure prek hooks are installed before committing/pushing
  in both repos.
- **SKILL.md:** rewrite the Filtering section (B2); add the `read(out=)` VCF-only note (C2);
  confirm the range-API PGEN-only note (C1). `empty()` is not in SKILL — CHANGELOG only.

## Testing & verification

- **Part A (behavior-preserving):** the existing suite stays green — `pixi run pytest`
  (vcfixture oracle + VCF/PGEN parity). No output/coordinate/dtype changes.
- **Part B (breaking):** update tests to the new `Filter` API, the 3-arg `empty()`, and the
  unified `_chunk_ranges_with_length` contract. Add a **cross-backend property test**:
  for the same query, VCF and PGEN with-length paths both yield haplotypes ≥ the query
  length (guards the extend algorithm the audit flagged as the deepest divergence, which we
  intentionally keep implemented twice per its recommendation).
- **gvl:** `svar2-m6b-kernel` test suite green against a local build of
  `sp4-vcf-pgen-consistency`.
- **Hooks/typing:** `ruff check`, `ruff format`, `pyrefly` green (both repos). Rust is
  untouched, but if any hook runs `cargo`, use `--no-default-features` for tests.

## Invariants

1. **Part A is behavior-preserving**, gated on the existing test suite staying green.
2. **Every public break (B1/B2/B3) updates `SKILL.md`/`CHANGELOG.md` in the same PR** and
   carries a `BREAKING CHANGE:` marker; migration notes included.
3. **The gvl tuple-contract change lands with the genoray change** — the two are one logical
   edit split across repos (genoray branch + gvl `svar2-m6b-kernel` / PR #266). Neither is
   "done" without the other.
4. **Small, reviewable commits** — no god-branch.
5. **B3 opens with the VCF-`chunk_idxs` feasibility spike** before the reconciled semantics
   are locked.

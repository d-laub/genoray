# SVAR write CLI: symbolic + breakend filter flags

**Date:** 2026-06-05
**Branch:** `feat/breakend-alts` (current)

## Problem

The `genoray write` CLI (VCF/PGEN → SVAR) exposes one filtering flag,
`--skip-symbolic-alts`, which drops records carrying VCF 4.x symbolic ALT
alleles (`<DEL>`, `<INS>`, …). Breakend (BND) ALTs — a *distinct* ALT class
(`G[chr2:321[`, `]chr2:321]G`, `.TGCA`, `TGCA.`) that `is_symbolic` does **not**
flag — are equally un-injectable into DNA buffers by downstream haplotype
consumers (e.g. genvarloader), but the CLI offers no way to drop them.

Users who want a gvl-ready SVAR currently cannot express "drop symbolic **and**
breakend" through the CLI.

## Goals

- Drop the symbolic ALT records via a clearer flag name, `--no-symbolic`
  (rename of `--skip-symbolic-alts`; treated as non-breaking — the flag is new
  enough to rename freely).
- Add `--no-breakend` to drop breakend ALT records.
- Make the two flags independent and composable (`AND` when both are passed).
- Preserve current behavior when neither flag is set (no filtering).

## Non-goals

- **No `--compat-gvl` convenience flag.** Bundling these into a single
  "gvl-compatible" switch couples genoray to gvl's *versioned* definition of
  compatibility (which may change). The gvl-compat recipe — "pass
  `--no-symbolic --no-breakend`" — belongs in gvl's own docs, not in a genoray
  flag.
- **No `--no-imprecise` flag.** For haplotype consumers, `~is_symbolic &
  ~is_breakend` already drops everything un-injectable, including precise
  `<DEL>`/`<INS>`/`<DUP>` (gvl cannot expand those either). `is_imprecise`
  serves a *different* goal (keep precise SVs, drop only un-sizable ones); users
  who want that can use the library `filter`/`pl_filter` API with
  `genoray.exprs.is_imprecise`. The CLI does not expose it.
- Version bump, CHANGELOG, gvl-side documentation.

## Background: how the CLI builds filters

`genoray/_cli/__main__.py::write` constructs the source reader, then hands it to
`SparseVar.from_vcf` / `from_pgen`, which inherit the source's filter.

- **VCF** requires *both* a `cyvcf2` record-level callable `filter` **and** a
  matching polars `pl_filter` — `VCF.__init__` raises if only one is supplied.
  Both must express the same predicate so the genotype scan and the index stay
  aligned (a divergence shifts variant indices and corrupts the sparse data).
- **PGEN** takes a single polars `filter` expression.

`genoray.exprs` already ships the polars predicates `is_symbolic` and
`is_breakend` (the latter built on the private regex `_BND_PATTERN`).

## Design

### 1. CLI surface (`genoray/_cli/__main__.py`, `write`)

Replace `skip_symbolic_alts: bool = False` with two independent flags:

```python
no_symbolic: Annotated[bool, Parameter(name="--no-symbolic", negative="")] = False,
no_breakend: Annotated[bool, Parameter(name="--no-breakend", negative="")] = False,
```

- `negative=""` suppresses cyclopts' auto-generated inverse
  (`--no-no-symbolic`), so the only spellings are `--no-symbolic` /
  `--no-breakend`. Flag present ⇒ drop those records.
- Both default `False` ⇒ no filtering, preserving prior behavior.
- Exact cyclopts negation mechanics to be confirmed against the `cyclopts`
  skill during implementation; `negative=""` is the documented way to remove the
  auto-inverse.

Update the `write` docstring: remove the `skip_symbolic_alts` entry, document
`no_symbolic` and `no_breakend`.

### 2. Filter composition

Build both filter representations from whichever flags are set:

- **polars side** (VCF & PGEN): collect `~exprs.is_symbolic` and/or
  `~exprs.is_breakend` into a list; combine with `reduce(operator.and_, ...)`;
  empty ⇒ `None`.
- **cyvcf2 callable side** (VCF only): collect the negated record-level
  predicates; the composed callable returns
  `all(pred(rec.ALT) for pred in preds)`; empty ⇒ `None`.
- **PGEN**: `filter=` the composed polars expr (or `None`).

VCF is always constructed with both `filter` and `pl_filter` set (or both
`None`), satisfying `VCF.__init__`'s paired-filter requirement.

### 3. Parity helpers (`genoray/exprs.py`)

The central correctness risk is the cyvcf2 record callable drifting from the
polars expr it must mirror. Anchor parity by co-locating two **private**
record-level predicates with the exprs they mirror:

```python
import re  # new top-level import

def _record_is_symbolic(alts: Iterable[str]) -> bool:   # mirrors is_symbolic
    return any(a.startswith("<") for a in alts)

def _record_is_breakend(alts: Iterable[str]) -> bool:   # mirrors is_breakend
    return any(re.search(_BND_PATTERN, a) is not None for a in alts)
```

The CLI imports these and negates them to form the keep-callable.

*Rejected alternative:* inline both lambdas in the CLI (as the current symbolic
lambda is). Rejected because `_BND_PATTERN` lives in `exprs.py`; inlining the
breakend regex invites drift between the callable and the expr.

### 4. Tests (`tests/test_svar_filtering.py`)

- Update `test_cli_write_skip_symbolic_vcf` / `_pgen` to pass `no_symbolic=True`.
- Add a `_breakend_vcf` fixture (a BND record via `vcfixture.Bnd` plus a plain
  SNV) and a `_breakend_pgen` derived from it via plink2.
- New tests: `--no-breakend` drops the BND on both the VCF and PGEN paths.
- One combined test (`no_symbolic=True, no_breakend=True`) asserting both ALT
  classes are dropped together.

### 5. Docs / SKILL.md

- Update the `write` docstring (see §1).
- **No `skills/genoray-api/SKILL.md` change required.** SKILL.md documents the
  Python import API, not CLI flags; the new helpers are underscore-private; and
  `is_symbolic` / `is_breakend` semantics are unchanged.

## Files

| File | Change |
|------|--------|
| `genoray/_cli/__main__.py` | Replace `skip_symbolic_alts` with `no_symbolic` + `no_breakend`; compose filters | Modify |
| `genoray/exprs.py` | Add `_record_is_symbolic`, `_record_is_breakend`, `import re` | Modify |
| `tests/test_svar_filtering.py` | Update renamed-flag tests; add breakend + combined tests | Modify |

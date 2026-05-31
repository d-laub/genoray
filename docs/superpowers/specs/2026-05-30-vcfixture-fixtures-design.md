# Replace tracked VCFs with vcfixture-generated fixtures + GroundTruth oracle

**Date:** 2026-05-30
**Status:** Approved (design)

## Goal

Eliminate the hand-written `.vcf` files tracked in `tests/data/` by generating
them programmatically with [`vcfixture`](https://github.com/d-laub/vcfixture),
and adopt vcfixture's `GroundTruth` oracle to replace hand-coded expected arrays
in genoray's tests wherever it mechanically fits.

Two wins: the `VcfBuilder` definitions become the single source of truth (no
opaque tracked test data), and tests assert against a decoded oracle instead of
literals that can silently drift from the file contents.

## Context (current state)

- `tests/data/` tracks 4 source VCFs: `biallelic.vcf`, `indels.vcf`,
  `multiallelic.vcf`, `three_samples_unsorted.vcf`.
- `tests/data/.gitignore` only tracks `*.vcf`, `*.sh`, `*.py`, `README.md`. The
  `.vcf.gz`, `.pgen`, and `.svar` derivatives are **generated** by
  `gen_from_vcf.sh` (bgzip/index → plink2 → `gen_svar.py`) at `pixi run test`
  time — they are not tracked.
- So "replace tracked VCFs" means: replace the 4 static `.vcf` with a vcfixture
  generation step feeding the existing downstream pipeline.
- vcfixture v0.2.1 already expresses everything these fixtures need: phased/
  unphased GT, half-calls (`./1`), `Number=A` DS, missing components (`.,1.0`),
  multi-allelic ALTs, contigs with/without length. No upstream feature gap is
  currently known; if one surfaces we implement it in a vcfixture worktree.
- genoray genotype reads return `(samples, ploidy[+phasing], variants)`;
  `GroundTruth.genotypes` is `(records, samples, ploidy)`. An adapter bridges
  the two. Dosages are `(samples, variants)`.

## Key constraint: oracle supplies values, not range selection

`GroundTruth` is a faithful decode of **file contents**. genoray range queries
also involve genoray's **own** selection logic (which variants fall in
`[start, end)`, spanning-deletion handling). Therefore the oracle perfectly
supplies the **values** of returned variants, while **which** variants a range
returns sometimes still needs to be stated explicitly in the test. The adapter
never re-implements genoray's range logic.

## Design

### 1. Worktrees & isolation

- **genoray worktree** at `.claude/worktrees/test-vcfixture` on branch
  `test/vcfixture`, created via the using-git-worktrees skill. All genoray
  changes happen here.
- **vcfixture worktree** on its own branch off vcfixture `main`, so we don't
  collide with the parallel genvarloader session using vcfixture `main`. Any
  vcfixture feature gap we hit is implemented here, committed, and (later)
  released; genoray points at this worktree during dev.

### 2. Dependency wiring

- vcfixture becomes a genoray **test/dev dependency**, added as an **editable
  path** pointing at the vcfixture worktree (pixi pypi-dependencies
  `{ path = "...", editable = true }`). genoray picks up upstream edits
  instantly.
- **Pre-merge:** replace the path dep with a published vcfixture version spec.
  (Checklist item; must not merge an editable local-path dep.)
- No change to genoray's runtime dependencies — vcfixture is test-only.

### 3. Fixture definitions module (single source of truth)

New tracked file `tests/data/fixtures.py`. Each fixture is a function returning
a `vcfixture.VcfBuilder`, reproducing the **exact** content of today's `.vcf`:

- `biallelic` — chr1/2/3 (no length), duplicate-POS spanning-del records,
  `Number=A` DS, half-calls (`./1`), missing dosages.
- `multiallelic` — chr1, `G→A,C`, `0/2:.,1.0`.
- `indels` — the contrived with_length layout at POS 1000/2000/3000/5000,
  positions preserved verbatim (see its `##COMMENT`); drives with_length edge
  cases.
- `three_samples_unsorted` — chr1 length=200, sample order C/A/B, GT-only.

A `FIXTURES` registry maps name → builder, shared by both the generator and the
tests.

Generated VCFs need not be byte-identical to today's files (fileformat version
and header ordering will differ); they must be **semantically equivalent** so
downstream plink2/svar derivatives and all assertions still hold.

### 4. Generation step & untracking

- New `tests/data/gen_vcfs.py`: iterates `FIXTURES`, writes each `<name>.vcf`
  into `tests/data/`.
- `gen_from_vcf.sh` gains a first step `python "$ddir"/gen_vcfs.py` (before
  bgzip). The rest of the script is unchanged.
- `tests/data/.gitignore`: drop `!*.vcf` so the 4 `.vcf` become ignored
  generated artifacts. Tracked in `tests/data/` after this: `fixtures.py`,
  `gen_vcfs.py`, `gen_from_vcf.sh`, `gen_svar.py`, `README.md`, `.gitignore`.
- `README.md` updated: the `.py` builders are now the source of truth; edits go
  there, then run `gen_from_vcf.sh`.

### 5. Oracle adapter (`tests/_oracle.py`)

The one genuinely new bridging component. Converts `GroundTruth` → genoray
conventions:

- `genos(truth, idx) -> (samples, ploidy, variants)` — `genotypes[idx]`
  transposed `(1,2,0)`, cast to int8/int16.
- `phasing(truth, idx) -> (samples, variants)` — `phasing[idx]` transposed
  `(1,0)`.
- `dosages(truth, idx, field="DS") -> (samples, variants)` — pull
  `format[rec][sample][field]`, map missing/`.`/NaN → `np.nan`, following
  genoray's `dosage_field` convention (1:1 for biallelic; for `Number=A`
  multi-alt it mirrors genoray's choice).
- `indices_in_range(truth, contig, start, end)` — convenience selector by POS
  (0-based half-open) for the common case. **Documented caveat:** spanning
  deletions and other genoray-specific selection are not modeled here; tests
  pass explicit indices for those and the oracle supplies only the values.
- Helper splitting genoray's phased read output `(s, ploidy+1, v)` into
  `(genos, phasing)` — mirrors the existing `np.array_split` idiom.

### 6. conftest wiring

New `tests/conftest.py` exposing session fixtures: the loaded `GroundTruth` per
dataset (built from `FIXTURES`) plus reader objects, so tests request `truth`
and assert genoray output equals `oracle.genos(truth, idx)` etc.

### 7. Migration (phased — "everywhere it mechanically fits")

1. **Infra** (§1–§4): full suite green with generated VCFs, no assertion changes
   yet.
2. **Adapter + conftest** (§5–§6), with unit tests for the adapter itself.
3. **Dense VCF** — migrate `test_vcf.py` value assertions to the oracle.
4. **Dense PGEN** — `test_pgen.py`.
5. **Remaining where it fits** — `test_svar.py`, `test_vcf_set_samples.py`,
   `test_issue36.py`, and svar/parity value-checks that map cleanly. Where
   `GroundTruth` is not a clean oracle (with_length parsimony in
   `test_dense2sparse_with_length.py`, sparse internals in `test_parity.py`),
   leave existing assertions and note why.

### 8. Verification

- After each phase: `pixi run test` (regenerates data + runs suite) green.
- Once, before untracking the originals: decode generated vs. original `.vcf`
  with cyvcf2 and compare records to prove semantic equivalence.

## Non-goals / notes

- **No public API change** → no `skills/genoray-api/SKILL.md` update needed
  (test infra only). Called out so the CLAUDE.md SKILL rule isn't tripped.
- plink2/bgzip orchestration stays in the shell script (not moved into Python).
- GTF/CLI test data (`test_gtf_annotation.py`, `tests/cli/`) adopt the oracle
  only if they already use the 4 VCF fixtures; no new fixture types invented.
- Commits follow Conventional Commits (`test:`, `build:`, `chore:` as fits).

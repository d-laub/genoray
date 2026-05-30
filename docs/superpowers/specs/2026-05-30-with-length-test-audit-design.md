# `*_with_length` test audit & improvement — design

**Date:** 2026-05-30
**Status:** Approved, ready for implementation plan

## Background

The `*_with_length` family extends a query region variant-by-variant, tracking
realized haplotype length, until each haplotype carries enough length to cover
the original query span. Deletions (negative ILEN) shrink haplotype length and
therefore *force* extension; insertions/SNPs do not.

Three parallel implementations exist:

| Backend | Core length logic | Existing tests |
|---|---|---|
| VCF | `_ext_genos_with_length`, `_ext_genos_dosages_with_length`, `_chunk_with_length_helper`, `_chunk_ranges_with_length` | `tests/test_vcf.py::test_chunk_with_length` (3 cases) + `tests/test_issue36.py` |
| PGEN | `_gen_with_length` (doubling-extension loop), `_chunk_ranges_with_length` | `tests/test_pgen.py::test_chunk_with_length` (3 cases) |
| SparseVar | `_find_starts_ends_with_length` (method + numba fn), `read_ranges_with_length` | `tests/test_svar.py::test_read_ranges_with_length` (2 cases) + `tests/test_svar_internals.py` (3 unit tests) |

### Two distinct semantics (important)

- **VCF & PGEN** are variant-major and return *dense* genotypes. They extend to
  the single variant boundary at which the **worst-case** haplotype across all
  samples reaches target length. All samples/haplotypes therefore share one
  (over-extended) variant window. This is intrinsic to dense formats.
- **SparseVar** is sample-major and intrinsically sparse. It extends each
  `(sample, haplotype)` **independently** to its own minimal length, so per
  haplotype it generally includes *fewer* variants than the shared dense window.

## Problem: what the current tests miss

The entire feature is exercised by `tests/data/biallelic.vcf`, which has exactly
**3 variants per contig**: `81262 GAT>A` (the only indel — a `-2` deletion),
`81262 G>A` (SNP), `81265 T>C` (SNP).

Identified gaps:

1. **One tiny deletion drives everything.** Only an ILEN `-2` deletion triggering
   a single variant of extension. No large, compounding, or multi-variant-span
   deletions.
2. **PGEN's doubling-extension loop is dead in tests.** `_gen_with_length` doubles
   `_idx_extension` and loops `while (hap_lens < length).any()` across multiple
   read rounds. With 3 variants this multi-round path never runs.
3. **Per-haplotype divergence untested.** A deletion on one haplotype but not the
   other should extend the two haplotypes differently. No case isolates this.
4. **Contig-end / clamp boundary untested.** All three clamp extension at
   `contig_max_idx` (or cyvcf2 iterator exhaustion). The "deletion near contig
   end, cannot reach target length, must stop short" path is never hit.
5. **SVAR public method has no empty/missing-contig case** (`length_none`), unlike
   VCF and PGEN.
6. **No cross-backend parity.** The same data exists as VCF, PGEN, and SVAR; each
   is checked against separate hand-written expectations. Nothing asserts they
   agree.
7. **The promised invariant is never asserted directly.** Tests check exact
   offsets/variant-sets but never the actual guarantee: realized haplotype length
   ≥ query length.

The bright spot is `tests/test_svar_internals.py`: focused unit regression tests
with documented bugs. VCF and PGEN length logic have no equivalent unit-level
tests.

## Scope

This is primarily a **test improvement**, plus **one new private production
function** (`_dense2sparse_with_length`) with a small shared-helper refactor that
makes exact cross-backend parity possible.

Out of scope: making any `with_length` symbol public, changing public API,
touching `skills/genoray-api/SKILL.md` (nothing public changes).

## Design

### 1. New fixture — `tests/data/indels.vcf`

A biallelic-only VCF (SVAR `with_length` rejects multiallelic), 2 samples,
phased, carrying a `DS` field so it converts to PGEN and SVAR through the
existing pipeline. Single contig, laid out as four spatially-isolated regions —
each targeting one untested path:

| Region | Construction | Exercises |
|---|---|---|
| **A — big deletion** | one `-10`bp deletion, then several SNPs | extension across multiple variants (not just one) |
| **B — per-haplotype divergence** | a deletion present on hap A (`1\|0`) but not hap B (`0\|1`) | the two haplotypes extend by different amounts |
| **C — SNP-dense after big deletion** | a `-50`bp deletion, then ~30 SNPs packed within <40bp | PGEN's doubling loop (`_idx_extension *= 2`, multi-round `while`): the first 20-variant grab spans <50bp, forcing a 2nd round |
| **D — deletion near contig end** | a deletion close to the last variant on the contig | `contig_max_idx` clamp / iterator exhaustion: extension stops short, realized length legitimately < query length |

Region C is deliberately contrived (a deletion wider than the span of 20
downstream variants is the only way to force PGEN's multi-round loop). It must
carry a comment in the VCF / test code stating this so it is not mistaken for a
realistic genome.

**Pipeline wiring:** add an `indels` entry to `tests/data/gen_from_vcf.sh`
(bgzip + index + `plink2 --make-pgen ... dosage=DS`) and to
`tests/data/gen_svar.py` (`SparseVar.from_vcf` + `SparseVar.from_pgen` +
`cache_afs`), mirroring the existing `biallelic` handling. This produces
`indels.vcf.gz`, `indels.pgen`/`.psam`/`.pvar`, `indels.vcf.svar`,
`indels.pgen.svar`. Existing `biallelic`-based tests are left untouched.

### 2. New production code — `_dense2sparse_with_length` (private)

Lives in `genoray/_svar.py` alongside `dense2sparse`. Private (leading
underscore), experimental — consistent with the rest of the `with_length`
family and with `dense2sparse` itself (also internal). Not exported; genvarloader
imports the private name, as it already does for other `with_length` methods.

**Purpose:** convert dense VCF/PGEN `with_length` output (the shared,
over-extended variant window) into per-haplotype sparse output, re-trimming each
`(sample, haplotype)` to its own minimal length — producing output **identical
to** `SparseVar.read_ranges_with_length` for the same query. That identity is
what makes it a valid parity bridge.

**Well-defined because:** the dense window extends until the worst-case
haplotype (most deletions) reaches Q; every other haplotype needs ≤ that many
variants, so each haplotype's independent minimal set is always a subset of the
dense window. The contig-end clamp degrades gracefully (each haplotype takes
what is available).

**Signature** (mirrors `dense2sparse` plus the inputs `_gen_with_length`
already threads):

```python
def dense2sparse_with_length(   # exposed as the private name `_dense2sparse_with_length`
    genos,         # (s p v) dense — the with_length window
    var_idxs,      # (v,) global variant indices of the window
    q_start, q_end,  # query span → target length
    v_starts,      # (v,) 0-based positions of window variants
    ilens,         # (v,) ILEN of window variants
    dosages=None,  # (s v) optional
) -> Ragged[V_IDX_TYPE] | tuple[Ragged[V_IDX_TYPE], Ragged[DOSAGE_TYPE]]:
    ...
```

(Final public-facing name is `_dense2sparse_with_length`; the body above uses a
placeholder for readability.)

**Shared-helper refactor:** the per-haplotype length-accounting walk (currently
inlined in the numba `_find_starts_ends_with_length` at `genoray/_svar.py:1930`)
is factored into a single shared helper that both `_find_starts_ends_with_length`
and `_dense2sparse_with_length` call. This guarantees the dense→sparse path and
the native sparse path cannot drift. The refactor must be behavior-preserving for
`_find_starts_ends_with_length` (verified by the existing
`test_svar_internals.py` regression tests still passing).

### 3. Invariant helper — `tests/_length_helpers.py`

A shared module with assertion helpers expressing the feature's actual
guarantee, computed via the existing `hap_ilens` util:

- **Dense (VCF/PGEN):** `max` over `(sample, haplotype)` realized length ≥ Q, and
  the window is minimal for that worst-case haplotype (dropping the last variant
  would push the worst case below Q) — unless extension was clamped at contig end.
- **Sparse (SVAR):** *every* `(sample, haplotype)` realized length ≥ Q and
  individually minimal — unless clamped.

These replace brittle exact-offset assertions with the guarantee the feature
exists to make, and are reused by the parity and case tests.

### 4. Cross-backend parity test (new fixture)

A single test running the same `with_length` query against all three backends:

- **PGEN ≡ VCF — exact.** Both dense and variant-major, both extend to the same
  worst-case boundary. Assert equality of genotypes, dosages, and final `end`.
- **SVAR ≡ `_dense2sparse_with_length(<VCF/PGEN dense output>)` — exact.** The
  bridge re-trims the dense window per haplotype; the result must equal
  `SparseVar.read_ranges_with_length` (same `Ragged` data + offsets).

Run across all four fixture regions (A–D), including the clamp region D.

### 5. Unit tests (test_svar_internals.py style)

- **PGEN `_gen_with_length`** — genuinely unit-testable: it takes a `read`
  callable plus synthetic `v_starts`/`v_ends`/`ilens`. Drive it with a fake
  `read` and synthetic arrays to assert the multi-round extension (doubling
  loop runs ≥ 2 rounds) and the contig-end clamp directly, no file needed.
- **SVAR `_find_starts_ends_with_length`** — extend `test_svar_internals.py`
  with large-deletion, per-haplotype-divergence, and contig-end cases.
- **`_dense2sparse_with_length`** — direct unit tests on synthetic dense windows,
  including dosages and the contig-end (truncated window) case.
- **VCF `_ext_genos_with_length`** — bound to `self._vcf(coord)`, so not cleanly
  unit-testable; covered via the fixture (integration). Note this asymmetry in
  the test module rather than forcing an awkward mock.

### 6. SVAR public-method gap

Add a `length_none` case (missing/empty contig) to
`tests/test_svar.py::test_read_ranges_with_length`, matching the `length_none`
cases that VCF and PGEN already have.

## Files touched

**Production**
- `genoray/_svar.py` — add `_dense2sparse_with_length`; factor shared
  length-accounting helper out of `_find_starts_ends_with_length`.

**Test data**
- `tests/data/indels.vcf` (new)
- `tests/data/gen_from_vcf.sh` (add `indels` entry)
- `tests/data/gen_svar.py` (add `indels` conversion)
- generated artifacts: `indels.vcf.gz`(+`.csi`), `indels.pgen`/`.psam`/`.pvar`,
  `indels.vcf.svar`, `indels.pgen.svar`

**Tests**
- `tests/_length_helpers.py` (new — invariant assertions)
- `tests/test_parity.py` (new — or add to an existing module — cross-backend parity)
- `tests/test_svar_internals.py` (new cases: large del, per-hap, contig-end;
  `_dense2sparse_with_length` unit tests)
- `tests/test_pgen.py` (new `_gen_with_length` unit test; richer fixture cases)
- `tests/test_vcf.py` (richer fixture integration cases)
- `tests/test_svar.py` (add `length_none` case)

## Acceptance criteria

1. `pixi run test` passes (data regeneration via `gen_from_vcf.sh` included).
2. New fixture exercises: multi-variant extension, per-haplotype divergence,
   PGEN multi-round doubling loop (≥ 2 rounds), and contig-end clamp.
3. `_dense2sparse_with_length` output exactly equals
   `SparseVar.read_ranges_with_length` across all fixture regions.
4. PGEN `with_length` output exactly equals VCF `with_length` output across all
   fixture regions.
5. Invariant helper assertions hold for every backend on every region (with the
   clamp exemption for region D).
6. Existing `test_svar_internals.py` regression tests still pass after the
   shared-helper refactor (behavior preserved).
7. No public API change; `skills/genoray-api/SKILL.md` untouched.

## Open considerations (non-blocking)

- Whether the parity test lives in a new `tests/test_parity.py` or is appended to
  an existing module is left to the implementation plan.
- Exact coordinates/genotypes for the four fixture regions are to be chosen
  during implementation so each region demonstrably triggers its target path
  (verify region C actually forces a 2nd PGEN extension round; verify region D
  actually clamps).

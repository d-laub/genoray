# Symbolic-allele-aware ILEN

**Status:** design approved, ready for implementation plan
**Date:** 2026-06-04 (revised after PR #51 merge)
**Scope:** VCF/BCF **and** PGEN paths (`VCF`, `PGEN`, `SparseVar.from_vcf` / `from_pgen`)
**Builds on:** PR #51 (`genoray.exprs.is_symbolic`, source-filter inheritance in
`SparseVar`, ALT-normalize-before-filter in `_load_index`). PR #51 named this exact
work as its future-work item: *"symbolic-allele expansion (`<DEL>` → precise via
INFO/END / SVLEN)."*

## 1. Problem

`genoray.exprs.ILEN` is defined as the byte-length difference between ALT and REF:

```python
ILEN = pl.col("ALT").list.eval(pl.element().str.len_bytes().cast(pl.Int32)) \
       - pl.col("REF").str.len_bytes().cast(pl.Int32)
```

For symbolic ALTs this is meaningless. `<DEL>` against a single padding REF base
yields `len("<DEL>") - len("G") = 5 - 1 = +4` — a *positive* ILEN for a deletion.
The wrong value then propagates, silently and untested, into:

- variant span computation — `_var_ranges.py:65`, `end = POS - ILEN.clip(upper_bound=0)`
- `"variant"` overlap mode — `_svar.py:273`
- `var_counts` / range queries
- haplotype-length deltas — `_utils.py` `hap_ilens`, and downstream GVL `get_diffs_sparse`

PR #51 added `genoray.exprs.is_symbolic` so users *can* filter symbolic alleles out,
but a symbolic allele that is left in still carries a garbage ILEN. There is no
correct-ILEN computation and zero test coverage for it.

## 2. Why this matters (and what it does *not* fix)

genoray's ILEN feeds two kinds of consumer:

1. **Range / span / overlap / count** — `var_ranges`, `"variant"` mode, `var_counts`.
   A correct symbolic ILEN makes precise SVs queryable and correctly-spanned.
2. **Haplotype reconstruction** — GenVarLoader (the primary consumer) sums ILEN
   for length deltas *and* copies the literal ALT bytes into the haplotype buffer
   (`GenVarLoader/python/genvarloader/_dataset/_genotypes.py:311,380`).

For a symbolic allele there is **no literal ALT sequence**, so haplotype
reconstruction cannot emit it regardless of how correct ILEN is. GVL already drops
symbolic alleles upstream (`bcftools norm`) and has no concept of `IMPRECISE` /
`CIPOS` / `CIEND`. Therefore:

- This work makes precise symbolic SVs **queryable and correctly-spanned**.
- It does **not** enable haplotype reconstruction of symbolic alleles — that is
  inherent. A haplotype consumer such as GVL filters `~genoray.exprs.is_symbolic`
  to drop *all* symbolic alleles (precise or not), since none have literal bytes.

## 3. Correct ILEN for precise SVs

For a **precise** symbolic SV, ILEN is the change in haplotype length, derived from
`SVLEN` (or `END - POS` for `<DEL>`/`<DUP>`), with the sign applied by SV type. This
normalizes the VCF 4.3/4.4 `SVLEN` sign-convention flip to a magnitude plus a
type-determined sign.

| ALT     | ILEN        | Reference span (`end`)                  |
|---------|-------------|------------------------------------------|
| `<DEL>` | `-|SVLEN|`  | `POS + |SVLEN|` (extends, like a literal deletion) |
| `<INS>` | `+|SVLEN|`  | `POS` (point footprint, like a literal insertion)  |
| `<DUP>` | `+|SVLEN|`  | `POS` (net insertion — matches genoray's insertion model) |

A precise symbolic SV then behaves identically to the equivalent literal indel for
range/span/overlap, which is the only genoray surface that can use it (§2).

`SVLEN` magnitude is taken as `|SVLEN|` when present; otherwise `|END - POS|` for
`<DEL>`/`<DUP>`. `<INS>` has no meaningful END, so it requires `SVLEN`.

## 4. Policy for un-sizable symbolic variants — NO new kwarg

A variant is **un-sizable** when it is symbolic and we cannot derive a precise
length: the `IMPRECISE` flag is set, `SVLEN`/`END` are missing or unparseable, or the
type is unsupported (`<BND>`, `<CNV>`, `<INV>`, `<*>` / `<NON_REF>`).

**genoray stays permissive — it does not drop or error, and adds no mode flag.** This
honors "one and only one way to do things" and PR #51's explicit non-goal of changing
default filtering. Concretely:

- Every variant is kept in the index (as today).
- Precise `<DEL>`/`<INS>`/`<DUP>` get the correct ILEN from §3.
- Un-sizable symbolic variants get **`ILEN = null`** in the polars index (per ALT
  element). `null` is honest: it is neither a literal length nor a fake `0` that
  would collide with SNPs.
- A new **derived** expression `genoray.exprs.is_imprecise` identifies them:

  ```python
  is_imprecise = pl.col("ILEN").list.eval(pl.element().is_null()).list.any()
  ```

  No persisted column, no constructor kwarg. Filtering is the user's job via the
  existing `filter` / `pl_filter` API, exactly as PR #51 established:
  `pgen = PGEN("f.pgen", filter=~genoray.exprs.is_imprecise)` keeps precise SVs and
  drops only the un-sizable ones; `~is_symbolic` drops all symbolic.

### null → 0 only at the numpy boundary

A `null` ILEN cannot flow into the int32 numba kernels (`var_ranges` end
computation, `SparseVar` start/end/ilen arrays, GVL). Every site that materializes
ILEN to numpy must coerce `null → 0` with `.fill_null(0)` *at that boundary* (a
null/un-sizable variant becomes a zero-length point variant — safe: no false
deletion-span extension, no length delta). Known sites to audit and fix:

- `genoray/_var_ranges.py:65,126,180` — `pl.col("ILEN").list.first().clip(upper_bound=0)`
- `genoray/_svar.py` — the `ILEN.list.first()` → numpy materializations
  (e.g. `:720`, `:2245-2249`, and the SEI cache in `_pgen.py`)

These are the "0 otherwise" fallbacks; the polars index keeps `null`.

## 5. Architecture

### 5.1 VCF / BCF path (`genoray/_vcf.py`)

- **Compute-at-construction, persist corrected ILEN.** `_write_gvi_index`
  (lines ~1037–1077) currently persists only `{CHROM, POS, REF, ALT}` and ILEN is
  computed lazily at load. Change it to compute the corrected ILEN at write time and
  persist it, so `_load_index` (the `if "ILEN" not in schema` block, ~line 1108)
  finds it on disk and does not recompute.
- **Header-gated SVLEN/END extraction.** To size symbolic SVs we need `SVLEN`/`END`,
  which live in INFO. Requesting an INFO field that the VCF header does not declare
  can error in oxbow, and most plain VCFs declare neither. So: introspect the cyvcf2
  header for which of `SVLEN`/`END` are declared (helper `_declared_info_fields`),
  request only those via `get_record_info(info=[...])`, compute ILEN, then **do not
  persist SVLEN/END** unless the user explicitly asked for them via `info=`. A
  symbolic variant in a VCF that declares neither field is therefore un-sizable →
  `null` ILEN → `is_imprecise` (correct).
- INFO fields come back uppercased; `Number=A` fields like `SVLEN` are `list`-typed
  (`.list.first()` to read).

### 5.2 PGEN path (`genoray/_pgen.py`)

- plink2 writes symbolic ALTs verbatim into the PVAR (PR #51 empirical finding), and
  the PGEN `.gvi` already persists the raw `INFO` string column (`_scan_pvar`,
  ~lines 1212–1256; sunk by `_write_index`). So at PGEN `_load_index`
  (the ILEN block ~line 1161), extract `SVLEN`/`END` from the persisted INFO string
  with polars regex (`str.extract(r"(?:^|;)SVLEN=(-?\d+)")`, etc.), detect the
  `IMPRECISE` flag, and compute the corrected ILEN with the same shared helper as the
  VCF path. No header dependency — a missing `SVLEN=` simply yields null.

### 5.3 Shared ILEN logic (DRY)

A single private helper produces the corrected per-ALT ILEN `List[Int32]` expression
(with `null` for un-sizable), given `ALT`, `REF`, and the `SVLEN`/`END`/`IMPRECISE`
values already extracted into columns. Both paths extract those columns their own way
(oxbow INFO columns for VCF; regex over the INFO string for PGEN) and then call it.
Likely home: `genoray/exprs.py` (private) or `genoray/_utils.py`. The public
`genoray.exprs.ILEN` expression stays **literal** (`len(ALT) - len(REF)`) — GVL and
others import and apply it to already-normalized literal-sequence indexes
(`GenVarLoader/.../_haps.py:26`); requiring an SVLEN column there would break them.

### 5.4 SparseVar (`genoray/_svar.py`)

After PR #51, `_write_filtered_index` passes a persisted-on-disk ILEN straight
through (`ilen_added = False`), so once VCF/PGEN persist the corrected ILEN it flows
into the SVAR unchanged. Only the numpy-boundary `fill_null(0)` fixes (§4) are needed
on the SVAR query/scan side.

## 6. Public API surface & SKILL.md

Public names touched, so `skills/genoray-api/SKILL.md` **must** be updated in the same
PR (per CLAUDE.md):

- corrected `ILEN` semantics for precise symbolic SVs, and `null` ILEN for un-sizable
  ones (behavior change)
- new derived expression `genoray.exprs.is_imprecise`
- guidance: filter `~is_symbolic` (haplotype consumers, drops all SVs) vs
  `~is_imprecise` (range consumers, keeps precise SVs)

No new constructor kwarg is added.

## 7. Testing (the "harden the tests" goal)

Using vcfixture 0.6.0's symbolic support (`Sym.deletion/insertion/duplication`,
`Bnd`, `SVLEN`/`SVCLAIM`/`END`/`IMPRECISE` in `info=`):

- **Named fixtures** (`tests/data/fixtures.py`): a `symbolic` builder with precise
  `<DEL>` / `<INS>` / `<DUP>` (with `SVLEN`, plus `SVCLAIM` where the version
  requires it), plus un-sizable cases — an `IMPRECISE` `<DEL>`, a `<BND>`, and a
  `<CNV>`. Add a symbolic VCF→PGEN step to `gen_from_vcf.sh` so the PGEN path is
  covered (plink2 carries the symbolic ALTs through).
- **Oracle helper** (`tests/_oracle.py`): `expected_ilen(truth, idx)` deriving the
  expected per-record ILEN from `GroundTruth.alts_truth[rec][alt]`
  (`.sv_type` → DEL `-`, INS/DUP `+`; `.svlen`/`.sv_end`), and `None` for un-sizable.
- **VCF + PGEN ILEN tests**: persisted ILEN equals the oracle expectation for precise
  SVs; `is_imprecise` is `True` exactly for the un-sizable rows; the un-sizable rows'
  ILEN is `null`.
- **Span / overlap**: a precise `<DEL>` of length N is returned by a query overlapping
  `[POS, POS+N)` and excluded just outside it — locks the `_var_ranges` fix and the
  `fill_null(0)` boundary.
- **Filter parity**: `~is_symbolic` drops all symbolic; `~is_imprecise` keeps precise
  SVs and drops un-sizable; both work on VCF (paired cyvcf2 + pl_filter) and PGEN, and
  inherit into `SparseVar`.
- **Property test**: `@given(vs.symbolic_documents())` round-tripped through `VCF`,
  asserting ILEN matches the oracle for the sizable subset and the policy
  (`null`/`is_imprecise`) holds for the rest.

## 8. Out of scope

- Haplotype reconstruction of symbolic alleles (inherent — no literal sequence).
- Breakend mate / confidence-interval (`CIPOS`/`CIEND`) semantics.
- Multiallelic records mixing precise and symbolic ALTs are handled best-effort
  (per-ALT `null` for the un-sizable element); fixtures focus on biallelic symbolic.
- Changing the public literal `genoray.exprs.ILEN` expression.

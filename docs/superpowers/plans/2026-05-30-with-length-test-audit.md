# `*_with_length` Test Audit & Improvement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the coverage gaps in the `*_with_length` family (VCF/PGEN/SparseVar) with a richer indel fixture, a direct invariant-assertion helper, exact cross-backend parity (via a new private `_dense2sparse_with_length` bridge), and unit tests for the PGEN/SVAR length logic.

**Architecture:** A new biallelic `indels.vcf` fixture with four isolated regions (big deletion, per-haplotype divergence, SNP-dense-after-big-deletion, deletion-near-contig-end) drives integration tests across all three backends. The per-haplotype length-accounting walk inside the SVAR numba helper is extracted into one shared njit function reused by both `_find_starts_ends_with_length` and the new `_dense2sparse_with_length`, so the dense and sparse code paths cannot drift. Parity is asserted exactly: PGEN≡VCF on dense output, and SVAR≡`_dense2sparse_with_length(dense output)`.

**Tech Stack:** Python, NumPy, numba, polars, seqpro `Ragged`, cyvcf2, pgenlib, pytest, pytest-cases, Pixi. Spec: `docs/superpowers/specs/2026-05-30-with-length-test-audit-design.md`.

---

## Reference facts (verified against the codebase)

- Run tests inside Pixi: `pixi run pytest <path>` (or `pixi run test` to also regenerate data).
- Phantom types (`genoray/_pgen.py:40-130`): `Genos` is `(s p v)` int32; build via `Genos.parse(arr)`. `GenosPhasingDosages` is a 3-tuple. `Genos.empty(n_samples, ploidy, n_variants)`.
- `hap_ilens(genotypes (s p v), ilens (v,)) -> (s p)` at `genoray/_utils.py:127` — sums ilens where `genos == 1`.
- `dense2sparse(genos, var_idxs, dosages=None)` at `genoray/_svar.py:313` — `keep = genos == 1`, builds `Ragged[V_IDX_TYPE].from_offsets(data, (*lengths.shape, None), lengths_to_offsets(lengths))`.
- Imports already in `genoray/_svar.py`: `from seqpro.rag import OFFSET_TYPE, Ragged, lengths_to_offsets` (line 30); `import numba as nb` (line 15); `from genoray._types import POS_TYPE, V_IDX_TYPE, DOSAGE_TYPE` (verify exact names in `_types.py`).
- `_find_starts_ends_with_length` numba fn at `genoray/_svar.py:1929`, decorated `@nb.njit(parallel=False, nogil=True, cache=True)`. The inner per-haplotype walk is lines ~2001-2051.
- SVAR method `_find_starts_ends_with_length` (`genoray/_svar.py:580`) passes `v_starts = (self.index["POS"] - 1).to_numpy()`, `ilens = self.index["ILEN"].list.first().to_numpy()`, `contig_max_idx = self._c_max_idxs[c]`.
- `SparseVar.read_ranges_with_length` (`genoray/_svar.py:709`) returns `Ragged[V_IDX_TYPE]` with shape `(n_ranges, n_samples, ploidy, None)` and `.data` = global variant indices, `.offsets`.
- PGEN `_chunk_ranges_with_length` (`genoray/_pgen.py:759`) yields, per range, a generator of `(data, end, var_idxs)`. `var_idxs` aligns with the variant axis of `data`.
- PGEN `_gen_with_length` (`genoray/_pgen.py:986`) signature: `(v_chunks, q_start, q_end, read, v_starts, v_ends, ilens, contig_max_idx)`; `read: Callable[[NDArray[V_IDX_TYPE]], L]` where `L` is e.g. `Genos`; uses `isinstance(out, Genos)` to branch; doubling loop `_idx_extension *= 2` with `while (hap_lens < length).any()`.
- VCF `_chunk_ranges_with_length` (`genoray/_vcf.py:749`) yields per range a generator of `(data, end, n_extension)`; genos dense `(s p v)` int8.
- Coordinates: all queries 0-based half-open `[start, end)`. VCF POS is 1-based → 0-based start is `POS-1`.

---

## File Structure

**Production**
- `genoray/_svar.py` — add njit `_length_walk_n_keep` helper; refactor `_find_starts_ends_with_length` to call it; add `_dense2sparse_with_length`.

**Test data**
- `tests/data/indels.vcf` (new)
- `tests/data/gen_from_vcf.sh` (add `indels` block)
- `tests/data/gen_svar.py` (add `indels` conversions)
- generated: `indels.vcf.gz`(+`.csi`), `indels.pgen`/`.psam`/`.pvar`(+`.gvi`), `indels.vcf.svar`, `indels.pgen.svar`

**Tests**
- `tests/_length_helpers.py` (new — invariant assertions, importable helper module; leading underscore so pytest does not collect it as a test file)
- `tests/test_dense2sparse_with_length.py` (new — unit tests for the bridge)
- `tests/test_parity.py` (new — cross-backend parity)
- `tests/test_svar_internals.py` (extend — new SVAR cases)
- `tests/test_pgen.py` (extend — `_gen_with_length` unit test + indels integration)
- `tests/test_vcf.py` (extend — indels integration)
- `tests/test_svar.py` (extend — `length_none` case)

---

## Task 1: Create the `indels.vcf` fixture and wire the generation pipeline

**Files:**
- Create: `tests/data/indels.vcf`
- Modify: `tests/data/gen_from_vcf.sh`
- Modify: `tests/data/gen_svar.py`

- [ ] **Step 1: Write `tests/data/indels.vcf`**

Four isolated regions on `chr1`, 2 phased samples, `GT:DS`. DS = count of ALT alleles. Region C's SNPs sit *after* the deletion's reference span (no spanning overlap) and are dense enough to force PGEN's multi-round extension.

```
##fileformat=VCFv4.2
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1	sample2
chr1	1000	.	GAAAAAAAAAA	G	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	1011	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	1013	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	1015	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	1017	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	1019	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	1021	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	2000	.	GAAAA	G	.	.	.	GT:DS	1|0:1.0	0|0:0.0
chr1	2002	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	2004	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	2006	.	T	C	.	.	.	GT:DS	1|1:2.0	1|1:2.0
chr1	3000	.	GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA	G	.	.	.	GT:DS	1|1:2.0	1|1:2.0
```

Then append **40 SNPs** at POS `3031, 3032, …, 3070` (1bp apart), each `T	C` with `1|1:2.0	1|1:2.0`. (Generate these 40 lines programmatically when authoring the file; they are mechanical.) Region C is deliberately contrived — add a VCF comment line above POS 3000:
`##COMMENT=Region C: -30 deletion + 40 dense SNPs, contrived to force PGEN's multi-round extension loop. Not a realistic genome.`

Finally region D (deletion as the LAST variant on chr1):
```
chr1	5000	.	GAAAAAAAAAA	G	.	.	.	GT:DS	1|1:2.0	1|1:2.0
```

Region intents (queries are 0-based; documented here for later tasks):
- **A** big deletion: query `("chr1", 999, 1010)` (len 11). `-10` del carried both haps → extends across SNPs.
- **B** per-hap: query `("chr1", 1999, 2006)` (len 7). `-4` del on sample1 hapA only → hapA extends more than hapB; sample2 not at all.
- **C** SNP-dense: query `("chr1", 2999, 3030)` (len 31). `-30` del carried both haps → forces ≥2 PGEN extension rounds.
- **D** clamp: query `("chr1", 4999, 5040)` (len 41). `-10` del is last variant → extension clamps, realized length < 41.

- [ ] **Step 2: Add the `indels` block to `tests/data/gen_from_vcf.sh`**

After the existing `unsorted` handling and before the SVAR conversion call, mirror the `biallelic` block (it uses `dosage=DS` and `--vcf-half-call r`):

```bash
indels=$ddir/indels.vcf

bgzip -c "$indels" >| "$indels".gz
bcftools index "$indels".gz
rm -f "$indels".gz.gvi

prefix="${indels%.vcf}"
plink2 --make-pgen --vcf "$indels".gz 'dosage=DS' --out "$prefix" --vcf-half-call r
rm -f "$prefix".log
rm -f "$prefix".pvar.gvi
```

- [ ] **Step 3: Add `indels` conversions to `tests/data/gen_svar.py`**

In `main()`, after the existing PGEN→SVAR block, add (mirroring the `biallelic` VCF and PGEN blocks):

```python
    # indels fixture (with_length edge cases)
    ivcf = VCF(ddir / "indels.vcf.gz", dosage_field="DS")
    ivcf._write_gvi_index()
    ivcf_path = ddir / "indels.vcf.svar"
    if ivcf_path.exists():
        shutil.rmtree(ivcf_path)
    max_mem = ivcf._mem_per_variant(ivcf.Genos8Dosages) * min(
        len(ivcf.contigs), joblib.cpu_count()
    )
    SparseVar.from_vcf(ivcf_path, ivcf, max_mem, overwrite=True, with_dosages=True)
    SparseVar(ivcf_path).cache_afs()

    ipgen = PGEN(ddir / "indels.pgen", dosage_path=ddir / "indels.pgen")
    ipgen_path = ddir / "indels.pgen.svar"
    if ipgen_path.exists():
        shutil.rmtree(ipgen_path)
    assert ipgen.contigs is not None
    max_mem = ipgen._mem_per_variant(ipgen.GenosDosages) * min(
        len(ipgen.contigs), joblib.cpu_count()
    )
    SparseVar.from_pgen(ipgen_path, ipgen, max_mem, overwrite=True, with_dosages=True)
    SparseVar(ipgen_path).cache_afs()
```

- [ ] **Step 4: Regenerate test data**

Run: `cd tests/data && pixi run bash gen_from_vcf.sh`
Expected: creates `indels.vcf.gz`, `indels.pgen`/`.psam`/`.pvar`, `indels.vcf.svar/`, `indels.pgen.svar/` with no errors.

- [ ] **Step 5: Sanity-check the fixture loads and queries resolve**

Run:
```bash
pixi run python -c "
from pathlib import Path
from genoray import VCF
d = Path('tests/data')
vcf = VCF(d/'indels.vcf.gz', dosage_field='DS')
print('contigs', vcf.contigs)
print('n_vars chr1', vcf.n_vars_in_ranges('chr1', 0, 10_000))
"
```
Expected: `contigs` includes `chr1`; variant count is 53 (1+6 + 1+3 + 1+40 + 1).

- [ ] **Step 6: Commit**

```bash
git add tests/data/indels.vcf tests/data/gen_from_vcf.sh tests/data/gen_svar.py \
        tests/data/indels.vcf.gz tests/data/indels.vcf.gz.csi \
        tests/data/indels.pgen tests/data/indels.psam tests/data/indels.pvar \
        tests/data/indels.vcf.svar tests/data/indels.pgen.svar
git commit -m "test: add indels fixture for with_length edge cases"
```

---

## Task 2: Extract shared length-walk helper and refactor `_find_starts_ends_with_length`

This must be **behavior-preserving** — the existing `tests/test_svar_internals.py` regression tests guard it.

**Files:**
- Modify: `genoray/_svar.py` (add helper before line 1929; edit the numba walk ~2001-2051)
- Test: `tests/test_svar_internals.py` (already exists — used as the guard)

- [ ] **Step 1: Run the existing SVAR internals tests to confirm green baseline**

Run: `pixi run pytest tests/test_svar_internals.py -v`
Expected: PASS (3 tests).

- [ ] **Step 2: Add the shared njit helper**

Insert immediately above `@nb.njit(parallel=False, nogil=True, cache=True)` / `def _find_starts_ends_with_length` (`genoray/_svar.py:1929`):

```python
@nb.njit(nogil=True, cache=True)
def _length_walk_n_keep(sp_genos, v_starts, ilens, start_idx, max_idx, q_start, q_end):
    """Number of leading carried variants (from start_idx) to keep so one
    haplotype reaches q_end - q_start in length, extending past q_end only as
    needed. Variants strictly inside [q_start, q_end) are always kept; the
    length budget only gates extension past q_end. Returns a count in
    [0, max_idx - start_idx]."""
    q_len = q_end - q_start
    last_v_end = q_start
    written_len = 0
    i = -1
    for j in range(start_idx, max_idx):
        i = j - start_idx
        v_idx = sp_genos[j]
        v_start = v_starts[v_idx]
        ilen = ilens[v_idx]

        maybe_add_one = POS_TYPE(v_start >= q_start)
        past_query = v_start >= q_end

        if v_start >= q_start:
            written_len += v_start - last_v_end
            if past_query and written_len >= q_len:
                i -= 1
                break
            written_len += max(0, ilen) + maybe_add_one
            if past_query and written_len >= q_len:
                break

        v_end = v_start - min(0, ilen) + maybe_add_one
        last_v_end = max(last_v_end, v_end)

    return i + 1
```

- [ ] **Step 3: Replace the inner walk in `_find_starts_ends_with_length`**

Replace the block from `q_start: POS_TYPE = q_starts[r]` through `out[1, r, s, p] = geno_idx + o_s + 1` (the per-`r` walk, ~lines 2001-2051) with a call to the helper:

```python
                n_keep = _length_walk_n_keep(
                    sp_genos,
                    v_starts,
                    ilens,
                    start_idx,
                    max_idx,
                    q_starts[r],
                    q_ends[r],
                )
                out[1, r, s, p] = start_idx + o_s + n_keep
```

Keep everything above it (the `var_ranges[r,0]==var_ranges[r,1]` guard, `out[0,...] = start_idx + o_s`, and the `start_idx == max_idx` early-continue) unchanged. Equivalence check: original `end = geno_idx + o_s + 1` with `geno_idx = start_idx + i`; helper returns `n_keep = i + 1`, so `start_idx + o_s + n_keep = start_idx + i + 1 + o_s = geno_idx + o_s + 1`. The immediate-saturation `geno_idx -= 1` case maps to `i -= 1 → return 0`, giving `end = start_idx + o_s` (matches).

- [ ] **Step 4: Run the regression tests to confirm behavior preserved**

Run: `pixi run pytest tests/test_svar_internals.py tests/test_svar.py -v`
Expected: PASS (all existing cases, including `test_find_starts_ends_with_length_includes_variant_at_query_edge`).

- [ ] **Step 5: Commit**

```bash
git add genoray/_svar.py
git commit -m "refactor: extract shared _length_walk_n_keep helper for with_length"
```

---

## Task 3: Implement `_dense2sparse_with_length` with unit tests (TDD)

**Files:**
- Create: `tests/test_dense2sparse_with_length.py`
- Modify: `genoray/_svar.py` (add `_dense2sparse_with_length` after `dense2sparse`, ~line 353)

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_dense2sparse_with_length.py`:

```python
"""Unit tests for _dense2sparse_with_length (the dense->sparse parity bridge)."""

from __future__ import annotations

import numpy as np
from seqpro.rag import OFFSET_TYPE

from genoray._svar import _dense2sparse_with_length
from genoray._types import V_IDX_TYPE


def test_no_indels_keeps_all_carriers_within_query():
    # 1 sample, ploidy 1, window of 3 SNPs all inside the query, all carried.
    # genos (s p v)
    genos = np.array([[[1, 1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1, 2], dtype=V_IDX_TYPE)
    v_starts = np.array([10, 11, 12], dtype=np.int32)
    ilens = np.array([0, 0, 0], dtype=np.int32)
    rag = _dense2sparse_with_length(genos, var_idxs, 10, 13, v_starts, ilens)
    # query len 3, three carried SNPs inside -> all kept
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3], dtype=OFFSET_TYPE))
    np.testing.assert_array_equal(rag.data, np.array([0, 1, 2], dtype=V_IDX_TYPE))


def test_deletion_forces_extension_past_query():
    # 1 sample, ploidy 1. A -2 deletion at the query start consumes the
    # length budget, so a variant past q_end must be kept to reach length.
    genos = np.array([[[1, 1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1, 2], dtype=V_IDX_TYPE)
    v_starts = np.array([10, 11, 20], dtype=np.int32)  # last is past q_end
    ilens = np.array([-2, 0, 0], dtype=np.int32)        # deletion first
    # query [10, 13): len 3. del removes 2 -> must extend past q_end.
    rag = _dense2sparse_with_length(genos, var_idxs, 10, 13, v_starts, ilens)
    # all three carriers kept (the post-query variant is needed)
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3], dtype=OFFSET_TYPE))
    np.testing.assert_array_equal(rag.data, np.array([0, 1, 2], dtype=V_IDX_TYPE))


def test_per_haplotype_independent_trim():
    # 1 sample, ploidy 2. hapA carries a deletion (needs more), hapB does not.
    # window vars: idx0 del, idx1 snp(in query), idx2 snp(past query)
    genosA = [1, 1, 1]  # carries deletion + both snps
    genosB = [0, 1, 0]  # only the in-query snp
    genos = np.array([[genosA, genosB]], dtype=np.int8)
    var_idxs = np.array([0, 1, 2], dtype=V_IDX_TYPE)
    v_starts = np.array([10, 11, 20], dtype=np.int32)
    ilens = np.array([-2, 0, 0], dtype=np.int32)
    rag = _dense2sparse_with_length(genos, var_idxs, 10, 13, v_starts, ilens)
    # hapA keeps all 3 carriers (0,1,2); hapB keeps its single in-query carrier (1)
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3, 4], dtype=OFFSET_TYPE))
    np.testing.assert_array_equal(rag.data, np.array([0, 1, 2, 1], dtype=V_IDX_TYPE))


def test_dosages_follow_genotypes():
    genos = np.array([[[1, 1]]], dtype=np.int8)
    var_idxs = np.array([5, 6], dtype=V_IDX_TYPE)
    v_starts = np.array([10, 11], dtype=np.int32)
    ilens = np.array([0, 0], dtype=np.int32)
    dosages = np.array([[0.5, 0.9]], dtype=np.float32)  # (s v)
    rag, drag = _dense2sparse_with_length(
        genos, var_idxs, 10, 12, v_starts, ilens, dosages
    )
    np.testing.assert_array_equal(rag.data, np.array([5, 6], dtype=V_IDX_TYPE))
    np.testing.assert_allclose(drag.data, np.array([0.5, 0.9], dtype=np.float32))


def test_clamp_keeps_what_is_available():
    # Window cannot reach target length (no variants past the deletion).
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=V_IDX_TYPE)
    v_starts = np.array([10], dtype=np.int32)
    ilens = np.array([-5], dtype=np.int32)
    rag = _dense2sparse_with_length(genos, var_idxs, 10, 30, v_starts, ilens)
    # only the one carried variant is available -> kept; no error
    np.testing.assert_array_equal(rag.offsets, np.array([0, 1], dtype=OFFSET_TYPE))
    np.testing.assert_array_equal(rag.data, np.array([0], dtype=V_IDX_TYPE))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_dense2sparse_with_length.py -v`
Expected: FAIL with `ImportError: cannot import name '_dense2sparse_with_length'`.

- [ ] **Step 3: Implement `_dense2sparse_with_length`**

Insert in `genoray/_svar.py` after the `dense2sparse` definition (after ~line 352). It walks each `(sample, haplotype)`'s carried variants with the shared `_length_walk_n_keep` helper, trims, and builds a `Ragged`:

```python
def _dense2sparse_with_length(
    genos: NDArray[np.integer],
    var_idxs: NDArray[V_IDX_TYPE],
    q_start: int,
    q_end: int,
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    dosages: NDArray[DOSAGE_TYPE] | None = None,
) -> Ragged[V_IDX_TYPE] | tuple[Ragged[V_IDX_TYPE], Ragged[DOSAGE_TYPE]]:
    """Convert a dense ``with_length`` window (shared, over-extended across all
    samples/haplotypes) into per-haplotype-minimal sparse output, identical to
    ``SparseVar.read_ranges_with_length`` for the same query.

    Parameters
    ----------
    genos
        Dense genotypes for the window. Shape: (samples, ploidy, variants).
    var_idxs
        Global variant indices of the window. Shape: (variants,).
    q_start, q_end
        0-based, half-open original query span (before extension).
    v_starts
        0-based start positions of the window's variants (i.e. POS - 1).
        Shape: (variants,).
    ilens
        ILEN of the window's variants (ALT - REF length). Shape: (variants,).
    dosages
        Optional dense dosages. Shape: (samples, variants).

    Returns
    -------
        ``Ragged[V_IDX_TYPE]`` of shape (samples, ploidy, ~variants), or a tuple
        with a matching ``Ragged[DOSAGE_TYPE]`` when ``dosages`` is given.
    """
    if genos.ndim != 3:
        raise ValueError(
            "Dense genotypes must have shape (samples, ploidy, variants)."
        )
    n_samples, ploidy, _ = genos.shape

    lengths = np.empty((n_samples, ploidy), dtype=np.int64)
    data_parts: list[NDArray[V_IDX_TYPE]] = []
    dose_parts: list[NDArray[DOSAGE_TYPE]] = []

    for s in range(n_samples):
        for p in range(ploidy):
            # local window indices of variants this haplotype carries (ALT)
            carried_local = np.flatnonzero(genos[s, p] == 1).astype(V_IDX_TYPE)
            carried_global = var_idxs[carried_local]
            n_keep = _length_walk_n_keep(
                carried_global,
                v_starts,
                ilens,
                0,
                len(carried_global),
                POS_TYPE(q_start),
                POS_TYPE(q_end),
            )
            kept_local = carried_local[:n_keep]
            lengths[s, p] = n_keep
            data_parts.append(var_idxs[kept_local])
            if dosages is not None:
                dose_parts.append(dosages[s, kept_local])

    data = (
        np.concatenate(data_parts)
        if data_parts
        else np.empty(0, dtype=V_IDX_TYPE)
    )
    offsets = lengths_to_offsets(lengths)
    shape = (n_samples, ploidy, None)
    rag = Ragged[V_IDX_TYPE].from_offsets(data, shape, offsets)

    if dosages is not None:
        dose_data = (
            np.concatenate(dose_parts)
            if dose_parts
            else np.empty(0, dtype=DOSAGE_TYPE)
        )
        drag = Ragged[DOSAGE_TYPE].from_offsets(dose_data, shape, offsets)
        return rag, drag
    return rag
```

If `DOSAGE_TYPE` is not already imported in `_svar.py`, add it to the `from genoray._types import ...` line (verify the symbol name in `genoray/_types.py`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_dense2sparse_with_length.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add genoray/_svar.py tests/test_dense2sparse_with_length.py
git commit -m "feat: add private _dense2sparse_with_length bridge"
```

---

## Task 4: Invariant-assertion helper module

**Files:**
- Create: `tests/_length_helpers.py`
- Test: covered indirectly; add a tiny self-test in `tests/test_dense2sparse_with_length.py` is unnecessary — exercise via Task 5/8.

- [ ] **Step 1: Write the helper module**

Create `tests/_length_helpers.py`:

```python
"""Shared assertions for *_with_length tests.

The feature's guarantee: each haplotype carries enough length to cover the
original query span. Dense backends (VCF/PGEN) extend to one shared boundary
(the worst-case haplotype reaches Q); sparse (SparseVar) extends each
haplotype independently. ``clamped=True`` exempts cases where extension hit the
contig end and legitimately could not reach Q.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from genoray._utils import hap_ilens


def realized_hap_lengths(
    genos: NDArray[np.integer], ilens: NDArray[np.int32], q_len: int
) -> NDArray[np.int32]:
    """Realized haplotype lengths for a dense window. genos: (s p v)."""
    # base query length + net indel contribution carried by each haplotype
    return q_len + hap_ilens(genos, ilens)


def assert_dense_reaches_length(
    genos: NDArray[np.integer],
    ilens: NDArray[np.int32],
    q_len: int,
    *,
    clamped: bool = False,
) -> None:
    """Dense (VCF/PGEN): the worst-case haplotype must reach q_len unless clamped."""
    hap_lens = realized_hap_lengths(genos, ilens, q_len)
    if clamped:
        return
    assert hap_lens.max() >= q_len, (
        f"no haplotype reached query length {q_len}; max realized {hap_lens.max()}"
    )


def assert_sparse_reaches_length(
    hap_lens: NDArray[np.int32], q_len: int, *, clamped: bool = False
) -> None:
    """Sparse (SparseVar): every haplotype must independently reach q_len."""
    if clamped:
        return
    assert (hap_lens >= q_len).all(), (
        f"some haplotype did not reach query length {q_len}: {hap_lens}"
    )
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `pixi run python -c "import tests._length_helpers as h; print(h.realized_hap_lengths)"`
Expected: prints the function object, no error.

- [ ] **Step 3: Commit**

```bash
git add tests/_length_helpers.py
git commit -m "test: add with_length invariant assertion helpers"
```

---

## Task 5: Cross-backend parity test

Asserts PGEN≡VCF (dense, exact) and SVAR≡`_dense2sparse_with_length(dense)` (exact), across all four fixture regions.

**Files:**
- Create: `tests/test_parity.py`

- [ ] **Step 1: Write the parity test**

Create `tests/test_parity.py`:

```python
"""Cross-backend parity for *_with_length on the indels fixture.

PGEN and VCF are dense/variant-major: they extend to one shared boundary, so
their dense outputs must be identical. SparseVar extends each haplotype
independently; its output must equal _dense2sparse_with_length applied to the
dense window (the parity bridge).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from genoray import PGEN, VCF, SparseVar
from genoray._svar import _dense2sparse_with_length

ddir = Path(__file__).parent / "data"

# (label, contig, start, end, clamped)
REGIONS = [
    ("A_big_deletion", "chr1", 999, 1010, False),
    ("B_per_haplotype", "chr1", 1999, 2006, False),
    ("C_snp_dense", "chr1", 2999, 3030, False),
    ("D_contig_end_clamp", "chr1", 4999, 5040, True),
]


def _collect_pgen(pgen, contig, start, end):
    """Concatenate a single range's chunks into one dense window + var_idxs."""
    mode = PGEN.GenosPhasingDosages
    # large max_mem -> few chunks; still concatenate to be safe
    gen = pgen._chunk_ranges_with_length(contig, start, end, "1g", mode)
    genos_parts, dose_parts, idx_parts = [], [], []
    end_pos = None
    for range_ in gen:
        for chunk, e, v_idxs in range_:
            g, p, d = chunk
            genos_parts.append(np.asarray(g))
            dose_parts.append(np.asarray(d))
            idx_parts.append(np.asarray(v_idxs))
            end_pos = e
        break  # single range queried
    genos = np.concatenate(genos_parts, axis=-1)
    dosages = np.concatenate(dose_parts, axis=-1)
    var_idxs = np.concatenate(idx_parts)
    return genos, dosages, var_idxs, end_pos


def _collect_vcf(vcf, contig, start, end):
    mode = VCF.Genos16Dosages
    vcf.phasing = True
    max_mem = "1g"
    gen = vcf._chunk_ranges_with_length(contig, start, end, max_mem, mode)
    genos_parts, dose_parts = [], []
    end_pos = None
    for range_ in gen:
        for chunk, e, _n_ext in range_:
            gp, d = chunk
            g, _p = np.array_split(gp, 2, 1)
            genos_parts.append(np.asarray(g))
            dose_parts.append(np.asarray(d))
            end_pos = e
        break
    genos = np.concatenate(genos_parts, axis=-1)
    dosages = np.concatenate(dose_parts, axis=-1)
    return genos, dosages, end_pos


@pytest.mark.parametrize("label,contig,start,end,clamped", REGIONS)
def test_pgen_equals_vcf_dense(label, contig, start, end, clamped):
    pgen = PGEN(ddir / "indels.pgen", dosage_path=ddir / "indels.pgen")
    vcf = VCF(ddir / "indels.vcf.gz", dosage_field="DS")

    pg, pd, _vidx, _pe = _collect_pgen(pgen, contig, start, end)
    vg, vd, _ve = _collect_vcf(vcf, contig, start, end)

    # same variant window for all haplotypes -> identical genos/dosages
    np.testing.assert_array_equal(pg.astype(np.int16), vg.astype(np.int16))
    np.testing.assert_allclose(pd, vd, rtol=1e-4, equal_nan=True)


@pytest.mark.parametrize("label,contig,start,end,clamped", REGIONS)
def test_svar_equals_bridge(label, contig, start, end, clamped):
    pgen = PGEN(ddir / "indels.pgen", dosage_path=ddir / "indels.pgen")
    svar = SparseVar(ddir / "indels.pgen.svar")

    genos, dosages, var_idxs, _end = _collect_pgen(pgen, contig, start, end)
    # global per-variant attributes for the window
    v_starts = (svar.index["POS"] - 1).to_numpy()
    ilens = svar.index["ILEN"].list.first().to_numpy()

    bridged = _dense2sparse_with_length(
        genos.astype(np.int8), var_idxs, start, end, v_starts, ilens, dosages
    )
    brag, bdrag = bridged

    actual = svar.read_ranges_with_length(contig, start, end)
    # actual shape (1 range, s, p, ~v); bridge shape (s, p, ~v)
    np.testing.assert_array_equal(actual.data, brag.data)
    np.testing.assert_array_equal(
        np.asarray(actual.offsets).ravel(), np.asarray(brag.offsets).ravel()
    )
```

Note on the SVAR offsets comparison: `read_ranges_with_length` returns shape `(1, s, p, None)` while the bridge returns `(s, p, None)`; both flatten to the same per-haplotype offsets for a single range. If the raveled offsets differ by the leading singleton range dimension, adjust the comparison to slice `actual` to its single range first (`actual[0]`).

- [ ] **Step 2: Run the parity test**

Run: `pixi run pytest tests/test_parity.py -v`
Expected: PASS (8 parametrized cases). If `test_svar_equals_bridge` fails on offsets shape, apply the `actual[0]` adjustment noted above and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_parity.py
git commit -m "test: cross-backend parity for with_length on indels fixture"
```

---

## Task 6: PGEN `_gen_with_length` unit test (fake `read`)

Drives the doubling-extension loop and the contig-end clamp directly, no PGEN file.

**Files:**
- Modify: `tests/test_pgen.py` (append unit tests)

- [ ] **Step 1: Write the failing/asserting unit tests**

Append to `tests/test_pgen.py` (imports at top of file: ensure `from genoray._pgen import _gen_with_length, Genos` and `from genoray._types import V_IDX_TYPE, POS_TYPE`, `import numpy as np`):

```python
def _fake_read_factory(dense_genos):
    """dense_genos: (s p V) global. Returns a read(var_idx)->Genos closure."""
    def read(var_idx):
        return Genos.parse(dense_genos[:, :, var_idx].astype(np.int32))
    return read


def test_gen_with_length_multi_round_extension():
    # 1 sample, ploidy 1. Variant 0 is a -30 deletion (carried). Variants 1..40
    # are SNPs 1bp apart starting just past the deletion's reference end. A
    # single 20-variant extension cannot recover 30 bp, forcing a 2nd round.
    n = 41
    v_starts = np.empty(n, dtype=POS_TYPE)
    v_starts[0] = 2999            # deletion at 0-based 2999
    v_starts[1:] = np.arange(3030, 3030 + (n - 1), dtype=POS_TYPE)
    ilens = np.zeros(n, dtype=np.int32)
    ilens[0] = -30
    v_ends = v_starts + 1
    v_ends[0] = 2999 + 31         # REF length 31

    dense = np.ones((1, 1, n), dtype=np.int32)  # all carried
    read = _fake_read_factory(dense)

    q_start, q_end = 2999, 3030   # len 31
    # first chunk = just the deletion (the in-query variant)
    v_chunks = [np.array([0], dtype=V_IDX_TYPE)]

    out = list(
        _gen_with_length(
            v_chunks=v_chunks,
            q_start=q_start,
            q_end=q_end,
            read=read,
            v_starts=v_starts,
            v_ends=v_ends,
            ilens=ilens,
            contig_max_idx=n - 1,
        )
    )
    # extension must have pulled in well more than the first 20-variant batch
    final_genos, _end, final_idx = out[-1]
    assert final_idx[-1] > 20, (
        f"multi-round extension not triggered; reached idx {final_idx[-1]}"
    )


def test_gen_with_length_clamps_at_contig_end():
    # Deletion is the last variant -> extension cannot proceed, must clamp.
    v_starts = np.array([4999], dtype=POS_TYPE)
    v_ends = np.array([4999 + 11], dtype=POS_TYPE)
    ilens = np.array([-10], dtype=np.int32)
    dense = np.ones((1, 1, 1), dtype=np.int32)
    read = _fake_read_factory(dense)

    out = list(
        _gen_with_length(
            v_chunks=[np.array([0], dtype=V_IDX_TYPE)],
            q_start=4999,
            q_end=5040,
            read=read,
            v_starts=v_starts,
            v_ends=v_ends,
            ilens=ilens,
            contig_max_idx=0,  # this IS the last variant
        )
    )
    final_genos, _end, final_idx = out[-1]
    # nothing to extend into: only the deletion variant returned
    np.testing.assert_array_equal(final_idx, np.array([0], dtype=V_IDX_TYPE))
```

- [ ] **Step 2: Run the unit tests**

Run: `pixi run pytest tests/test_pgen.py -k "gen_with_length" -v`
Expected: PASS (2 tests). If `test_gen_with_length_multi_round_extension` does not exceed idx 20, the helper math differs from assumed — print `final_idx[-1]` and confirm the deletion/SNP geometry forces a 2nd round (increase deletion magnitude or SNP spacing), then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pgen.py
git commit -m "test: unit-test PGEN _gen_with_length multi-round + clamp"
```

---

## Task 7: New SVAR internals cases (large deletion, per-haplotype, contig-end)

**Files:**
- Modify: `tests/test_svar_internals.py` (append cases)

- [ ] **Step 1: Write the new cases**

Append to `tests/test_svar_internals.py` (reuse existing imports `_find_starts_ends_with_length`, `np`, `OFFSET_TYPE`, `POS_TYPE`, `V_IDX_TYPE`):

```python
def test_with_length_large_deletion_extends_multiple_variants():
    # 1 sample, ploidy 1. A -10 deletion at the query start forces extension
    # across several downstream SNPs to recover length.
    v_starts = np.array([999, 1010, 1012, 1014, 1016], dtype=np.int32)
    ilens = np.array([-10, 0, 0, 0, 0], dtype=np.int32)
    n_vars = len(v_starts)
    genos = np.arange(n_vars, dtype=V_IDX_TYPE)          # all carried by the hap
    geno_offsets = np.array([0, n_vars], dtype=OFFSET_TYPE)
    sample_idxs = np.array([0], dtype=np.int64)

    q_starts = np.array([999], dtype=POS_TYPE)
    q_ends = np.array([1010], dtype=POS_TYPE)             # len 11
    var_ranges = np.array([[0, 1]], dtype=V_IDX_TYPE)     # query covers only the deletion

    out = _find_starts_ends_with_length(
        genos, geno_offsets, q_starts, q_ends, var_ranges,
        v_starts, ilens, sample_idxs, 1, int(v_starts[-1]) + 1,
    )
    start = int(out[0, 0, 0, 0])
    end = int(out[1, 0, 0, 0])
    assert start == 0
    assert end > 2, f"large deletion did not extend across multiple variants: [{start},{end})"


def test_with_length_per_haplotype_divergence():
    # 1 sample, ploidy 2. hapA carries a deletion (needs extension); hapB does not.
    v_starts = np.array([1999, 2002, 2004, 2006], dtype=np.int32)
    ilens = np.array([-4, 0, 0, 0], dtype=np.int32)
    # sparse storage is per (sample, ploidy): hapA carries all 4, hapB carries only the 3 SNPs
    genos = np.array([0, 1, 2, 3,     1, 2, 3], dtype=V_IDX_TYPE)
    geno_offsets = np.array([0, 4, 7], dtype=OFFSET_TYPE)  # hapA: [0,4), hapB: [4,7)
    sample_idxs = np.array([0], dtype=np.int64)

    q_starts = np.array([1999], dtype=POS_TYPE)
    q_ends = np.array([2006], dtype=POS_TYPE)               # len 7
    var_ranges = np.array([[0, 1]], dtype=V_IDX_TYPE)

    out = _find_starts_ends_with_length(
        genos, geno_offsets, q_starts, q_ends, var_ranges,
        v_starts, ilens, sample_idxs, 2, int(v_starts[-1]) + 1,
    )
    len_hapA = int(out[1, 0, 0, 0]) - int(out[0, 0, 0, 0])
    len_hapB = int(out[1, 0, 0, 1]) - int(out[0, 0, 0, 1])
    assert len_hapA > len_hapB, (
        f"haplotype carrying the deletion should extend further: A={len_hapA} B={len_hapB}"
    )


def test_with_length_clamps_at_contig_end():
    # Single carried deletion is the last variant on the contig -> cannot extend.
    v_starts = np.array([4999], dtype=np.int32)
    ilens = np.array([-10], dtype=np.int32)
    genos = np.array([0], dtype=V_IDX_TYPE)
    geno_offsets = np.array([0, 1], dtype=OFFSET_TYPE)
    sample_idxs = np.array([0], dtype=np.int64)

    q_starts = np.array([4999], dtype=POS_TYPE)
    q_ends = np.array([5040], dtype=POS_TYPE)               # len 41, unreachable
    var_ranges = np.array([[0, 1]], dtype=V_IDX_TYPE)

    out = _find_starts_ends_with_length(
        genos, geno_offsets, q_starts, q_ends, var_ranges,
        v_starts, ilens, sample_idxs, 1, int(v_starts[-1]) + 1,
    )
    start = int(out[0, 0, 0, 0])
    end = int(out[1, 0, 0, 0])
    assert (start, end) == (0, 1), f"clamp should keep exactly the one variant: [{start},{end})"
```

- [ ] **Step 2: Run the SVAR internals tests**

Run: `pixi run pytest tests/test_svar_internals.py -v`
Expected: PASS (3 existing + 3 new). If the per-haplotype assertion's offset arithmetic surprises you, print `out[:, 0, 0, :]` to inspect both haplotypes' `[start, end)` and adjust expectations to the documented semantics (deletion-carrying hap keeps more variants).

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar_internals.py
git commit -m "test: add large-deletion, per-haplotype, and clamp SVAR cases"
```

---

## Task 8: Richer VCF/PGEN integration cases on the indels fixture (with invariant helper)

**Files:**
- Modify: `tests/test_vcf.py` (add an indels-driven test using the invariant helper)
- Modify: `tests/test_pgen.py` (same)

- [ ] **Step 1: Add the VCF integration test**

Append to `tests/test_vcf.py` (ensure `from tests._length_helpers import assert_dense_reaches_length` and `from pathlib import Path`, `ddir` already defined in the module):

```python
import pytest as _pytest_for_indels  # if pytest not already imported, use existing import


@_pytest_for_indels.mark.parametrize(
    "label,start,end,clamped",
    [
        ("A_big_deletion", 999, 1010, False),
        ("B_per_haplotype", 1999, 2006, False),
        ("C_snp_dense", 2999, 3030, False),
        ("D_contig_end_clamp", 4999, 5040, True),
    ],
)
def test_indels_with_length_reaches_query_length(label, start, end, clamped):
    vcf = VCF(ddir / "indels.vcf.gz", dosage_field="DS")
    vcf.phasing = True
    mode = VCF.Genos16Dosages
    gen = vcf._chunk_ranges_with_length("chr1", start, end, "1g", mode)

    # collect this range's dense window
    g_parts = []
    for range_ in gen:
        for chunk, _e, _n in range_:
            gp, _d = chunk
            g, _p = np.array_split(gp, 2, 1)
            g_parts.append(np.asarray(g))
        break
    genos = np.concatenate(g_parts, axis=-1)

    # window ILENs from the gvi index, sliced to the returned variant window
    ilens_all = vcf._index["ILEN"].to_numpy() if vcf._index is not None else None
    # fall back: derive ILEN from the SVAR view to avoid index-internal coupling
    from genoray import SparseVar
    svar = SparseVar(ddir / "indels.vcf.svar")
    # global ilens; the dense window starts at the first variant in [start,end)
    # and runs contiguously, so map by genomic order via var positions.
    # Simplest robust check: the worst-case realized length must reach q_len.
    q_len = end - start
    # Use only the carried-indel contribution we can see in this window:
    # reconstruct window ilens by matching count.
    ilens = svar.index["ILEN"].list.first().to_numpy()
    n = genos.shape[-1]
    # the window is the first n variants at/after the query start in index order
    # (var_ranges are contiguous); find the start index via positions.
    pos = (svar.index["POS"] - 1).to_numpy()
    w_start = int(np.searchsorted(pos, start))
    window_ilens = ilens[w_start : w_start + n].astype(np.int32)

    assert_dense_reaches_length(genos, window_ilens, q_len, clamped=clamped)
```

Note: this couples the window-ILEN reconstruction to contiguous variant ordering. If `vcf._index` already exposes ILEN cleanly, prefer slicing it directly; the SVAR fallback above exists only to avoid depending on `_index` internals. Keep whichever is simpler in this codebase after inspecting `vcf._index` columns.

- [ ] **Step 2: Add the PGEN integration test**

Append to `tests/test_pgen.py` an analogous test. PGEN exposes `var_idxs` per chunk, so window ILENs are exact (no reconstruction):

```python
def test_indels_with_length_reaches_query_length_pgen():
    from tests._length_helpers import assert_dense_reaches_length
    from genoray import SparseVar
    pgen = PGEN(ddir / "indels.pgen", dosage_path=ddir / "indels.pgen")
    svar = SparseVar(ddir / "indels.pgen.svar")
    ilens_global = svar.index["ILEN"].list.first().to_numpy().astype(np.int32)

    cases = [
        (999, 1010, False),
        (1999, 2006, False),
        (2999, 3030, False),
        (4999, 5040, True),
    ]
    mode = PGEN.GenosPhasingDosages
    for start, end, clamped in cases:
        gen = pgen._chunk_ranges_with_length("chr1", start, end, "1g", mode)
        g_parts, idx_parts = [], []
        for range_ in gen:
            for chunk, _e, v_idxs in range_:
                g, _p, _d = chunk
                g_parts.append(np.asarray(g))
                idx_parts.append(np.asarray(v_idxs))
            break
        genos = np.concatenate(g_parts, axis=-1)
        var_idxs = np.concatenate(idx_parts)
        window_ilens = ilens_global[var_idxs]
        assert_dense_reaches_length(genos, window_ilens, end - start, clamped=clamped)
```

- [ ] **Step 3: Run both integration tests**

Run: `pixi run pytest tests/test_vcf.py -k indels tests/test_pgen.py -k "indels" -v`
Expected: PASS. If region C's worst-case does not reach length, the fixture geometry needs tuning (Task 1) — increase SNP count after POS 3070.

- [ ] **Step 4: Commit**

```bash
git add tests/test_vcf.py tests/test_pgen.py
git commit -m "test: indels with_length integration with invariant assertions"
```

---

## Task 9: SparseVar `length_none` case

**Files:**
- Modify: `tests/test_svar.py` (add a `length_none` case to the `length_` case group)

- [ ] **Step 1: Add the case**

In `tests/test_svar.py`, alongside `length_no_ext` and `length_ext` (~line 180-196), add:

```python
def length_none():
    # Missing contig -> empty result for every haplotype.
    cse = "chr3", 0, 1
    shape = (1, N_SAMPLES, PLOIDY, None)
    offsets = np.zeros((N_SAMPLES, PLOIDY + 1), OFFSET_TYPE)
    desired = Ragged[V_IDX_TYPE].from_offsets(DATA, shape, offsets)
    return cse, desired
```

Verify against the method: `_find_starts_ends_with_length` returns `np.full((n_ranges, len(samples), self.ploidy, 2), -1, ...)` when the contig is missing (`c is None`, `genoray/_svar.py:626-627`). Confirm how `read_ranges_with_length` reshapes that into a `Ragged` (it calls `.reshape(2, -1)`); if a missing contig yields all-equal start/end offsets (empty per haplotype), the `offsets` above (all zeros → zero-length ragged rows) is correct. If the missing-contig path produces a different offset layout, set `desired` to match the actual empty-per-haplotype `Ragged` (data length 0, monotonic offsets). Inspect by running the method once:

```bash
pixi run python -c "
from pathlib import Path
from genoray import SparseVar
s = SparseVar(Path('tests/data/biallelic.vcf.svar'))
r = s.read_ranges_with_length('chr3', 0, 1)
print(r.shape); print(r.offsets); print(r.data)
"
```
Set the expected `offsets`/`data` in `length_none` to whatever an empty result actually is, so the test asserts the real contract.

- [ ] **Step 2: Run the SVAR read_ranges_with_length cases**

Run: `pixi run pytest tests/test_svar.py -k read_ranges_with_length -v`
Expected: PASS (`length_no_ext`, `length_ext`, `length_none`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar.py
git commit -m "test: add missing-contig (length_none) case for SVAR with_length"
```

---

## Task 10: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite (regenerates data)**

Run: `pixi run test`
Expected: all tests pass, including network-independent ones. If a network test fails for connectivity reasons, re-run with `pixi run pytest -m "not network"`.

- [ ] **Step 2: Lint/format**

Run: `pixi run ruff check genoray tests && pixi run ruff format --check genoray tests`
Expected: clean (or run `ruff format genoray tests` and re-commit if formatting changed).

- [ ] **Step 3: Confirm no public API changed**

Verify `genoray/__init__.py` is unchanged and `_dense2sparse_with_length` is private (leading underscore, not exported). `skills/genoray-api/SKILL.md` requires no edit. Confirm:

```bash
git diff --stat main -- genoray/__init__.py skills/genoray-api/SKILL.md
```
Expected: no output (both unchanged).

- [ ] **Step 4: Final commit (if formatting or stray changes)**

```bash
git add -A
git commit -m "chore: lint/format with_length test additions"
```

---

## Self-Review

**Spec coverage:**
- §1 fixture → Task 1 (all four regions A–D wired through gen scripts). ✓
- §2 `_dense2sparse_with_length` + shared-helper refactor → Tasks 2 & 3. ✓
- §3 invariant helper (dense + sparse flavors) → Task 4. ✓
- §4 parity (PGEN≡VCF exact; SVAR≡bridge) → Task 5. ✓
- §5 unit tests: PGEN `_gen_with_length` → Task 6; SVAR internals new cases → Task 7; `_dense2sparse_with_length` units → Task 3; VCF via fixture integration → Task 8. ✓
- §6 SVAR `length_none` → Task 9. ✓
- Acceptance criteria 1–7 → covered by Tasks 10 (full run, lint, no-public-API) and the per-task assertions. ✓

**Placeholder scan:** Fixture coordinates and Region C geometry are concrete but flagged as empirically tunable with explicit verification steps (Tasks 1/6/8) — this is honest, not a placeholder, because each carries a measured expected outcome and a tuning instruction. No "TBD"/"handle edge cases"/"write tests for the above" left.

**Type/name consistency:** `_length_walk_n_keep` (Task 2) is reused with the same signature by `_dense2sparse_with_length` (Task 3) and indirectly via `_find_starts_ends_with_length`. `_dense2sparse_with_length` signature `(genos, var_idxs, q_start, q_end, v_starts, ilens, dosages=None)` is identical in Task 3 (def), Task 5 (call), and matches the spec. Invariant helpers `assert_dense_reaches_length` / `assert_sparse_reaches_length` named identically in Tasks 4 and 8. Phantom `Genos.parse` usage (Task 6) matches `genoray/_pgen.py:40-45`.

**Known soft spots (verify during execution, not blockers):**
- Region C must force ≥2 PGEN extension rounds — Task 6 unit test asserts this on synthetic data (deterministic), and Task 8 asserts the fixture reaches length; if the *fixture* doesn't trigger multi-round in the integration path, the unit test (Task 6) still covers the loop. Acceptable.
- SVAR `length_none` exact offsets must be read off the real method (Task 9 includes the inspection command before finalizing expected values).
- The Task 8 VCF window-ILEN reconstruction couples to contiguous variant ordering; prefer `vcf._index["ILEN"]` slicing if cleaner after inspection.

# Vectorize / numba-ify SBS-96 / DBS-78 / ID-83 classification on SparseVar

**Date:** 2026-06-13
**Status:** Approved (pending spec review)

## Goal

Make `SparseVar.annotate_mutations` and `SparseVar.mutation_matrix` as fast as
possible on huge files by removing the per-variant Python loop in
`classify_variants` and parallelizing the existing entry-code and counting
kernels. The mutation-classification *results* must match the current
implementation on all common SBS/DBS/ID cases; rare/degenerate inputs
(non-ACGT, MNV > 2 bp, symbolic) continue to map to `UNCLASSIFIED` as today.

This is a performance refactor. The public surface — `annotate_mutations`,
`mutation_matrix`, the int16 code space, the sentinels, and `classify_variants`'
signature and return type — is unchanged, so `skills/genoray-api/SKILL.md`
needs no API update.

## Current state and bottleneck

The pipeline behind `mutation_matrix` has three stages (`genoray/_mutcat.py`):

1. `classify_variants(index, reference)` (`:478`) — a **pure-Python `for` loop
   over every variant row**. Per variant it encodes REF/ALT strings, makes 1–2
   `reference.fetch()` calls (SNV) or a window fetch (indel), and classifies via
   string-formatting + dict lookups (`classify_sbs96` / `classify_dbs78` /
   `classify_id83`). Scales with the number of variants.
2. `build_entry_codes(...)` (`:392`) — broadcasts per-variant codes to per-entry
   codes and applies the DBS adjacency override. Already `nb.njit`, single-thread.
3. `count_matrix(...)` (`:451`) — per-sample accumulation into a
   `(n_samples, N_CODES)` matrix. Already `nb.njit`, single-thread.

Stage 1 is the dominant per-variant cost. Stages 2–3 scale with total sparse
*entries* (samples × variants-per-sample), which dominate on many-sample files;
they are already numba but single-threaded.

All three stages are in scope.

## Design decisions

- **Approach: hybrid.** Vectorize the dominant context-free / fancy-index cases
  (SNV → SBS-96, native 2 bp doublet → DBS-78) in plain numpy; isolate the one
  genuinely sequential case (indel repeat/microhomology scan → ID-83) in a single
  parallel numba kernel. Add `prange` to stages 2 and 3.
- **Multi-threaded.** Numba kernels use `parallel=True` + `prange`. (No
  thread-count knob in v1; callers control concurrency via `NUMBA_NUM_THREADS`.)
- **Correctness contract:** element-for-element equal to the current
  `classify_variants` on ACGT inputs across SNV/DBS/indel. The existing scalar
  `classify_sbs96` / `classify_dbs78` / `classify_id83` are **retained as the
  test oracle**, not deleted.
- **Single-sourced codebooks:** all label→code mappings (SBS-96 arithmetic,
  ID-83 lookup arrays) are derived once from the existing `SBS96` /
  `ID83_INDEX` / `_DBS_TABLE` definitions and asserted equal in tests, so there
  is no second hand-maintained copy of the codebook order.

## Architecture and data flow

`classify_variants(index, reference)` keeps its signature and return type
(`int16[n_variants]`) and is reimplemented as:

```
classify_variants(index, reference):
    pos        = index["POS"].to_numpy().astype(int32)   # 1-based (VCF)
    ref_buf, ref_off = flat uint8 buffer + int32 offsets for REF
    alt_buf, alt_off = flat uint8 buffer + int32 offsets for ALT.list.first()
    contig_code = per-row contig id (equality semantics only)

    rlen = ref_off[1:] - ref_off[:-1]
    alen = alt_off[1:] - alt_off[:-1]
    snv_mask   = (rlen == 1) & (alen == 1)
    dbs_mask   = (rlen == 2) & (alen == 2)
    indel_mask = (rlen != alen)          # anchored check inside the kernel

    out = full(n_variants, UNCLASSIFIED, int16)
    for contig in contigs present in the index:
        seq = reference.contig_array(contig)             # uint8, loaded once
        rows = where(contig_code == contig)
        out[snv  rows] = _sbs96_vec(seq, pos, ref_buf/off, alt_buf/off)
        out[dbs  rows] = _dbs78_vec(ref_buf/off, alt_buf/off)
        out[indel rows] = _id83_kernel(seq, pos, ref_buf/off, alt_buf/off)
    # REF-mismatch handling (deletions with n_rep == 0):
    #   kernel returns _REF_MISMATCH sentinel; collect those rows, emit the
    #   same aggregated loguru warning with chrom:pos examples, set UNCLASSIFIED
    return out
```

REF/ALT are extracted as one flat `uint8` buffer + `int32` offsets via a
vectorized Polars/numpy path (no per-row Python string list). The three masks
are mutually exclusive; unmatched rows stay `UNCLASSIFIED` (MNV > 2 bp,
symbolic, non-ACGT — same as today).

`build_entry_codes` and `count_matrix` keep their interfaces; only their kernels
change (stages 2–3 below).

### Stage 1a — SNV → SBS-96 (vectorized numpy)

For the SNV-masked rows on a contig, in one vectorized pass:

```
p     = pos[snv] - 1                       # 1-based → 0-based, int32
ref_b = REF first byte ; alt_b = ALT first byte
five  = seq[clip(p-1)] ; three = seq[clip(p+1)]   # gather; OOB → invalid
r,a,f,t = BASE2IDX[ref_b], BASE2IDX[alt_b], BASE2IDX[five], BASE2IDX[three]
valid = (r>=0)&(a>=0)&(f>=0)&(t>=0)&(r!=a) & in-bounds(p-1) & in-bounds(p+1)
purine = (r in {A,G}) → r,a = comp(r),comp(a); f,t = comp(t),comp(f)   # vectorized
sub_idx = SUB_LUT[r,a]                      # 4x4 → 0..5
code    = sub_idx*16 + f*4 + t              # COSMIC order: sub outer, 5' next, 3' inner
out[snv] = where(valid, code, UNCLASSIFIED)
```

- `BASE2IDX`: 256-entry LUT, `A/C/G/T → 0/1/2/3`, else `-1`.
- `comp(x) = 3 - x` (A↔T, C↔G under the 0–3 encoding = exact complement).
- `SUB_LUT`, the complement map, and the base LUT are module-level constants.
- Contig-boundary variants (first/last base) → OOB flank → `valid=False` →
  `UNCLASSIFIED`, matching today's `N`-padded fetch (`N ∉ _COMP`).
- A test asserts `sub_idx*16 + f*4 + t == SBS96_INDEX[label]` for all 96 labels.

### Stage 1b — DBS-78 native doublet (vectorized table lookup)

Reuses the existing `_DBS_TABLE[r0,r1,a0,a1]` (built at import from the scalar
`classify_dbs78`, inheriting fold/revcomp correctness). For `dbs_mask` rows:

```
r0,r1 = BASE2IDX[ref_byte0], BASE2IDX[ref_byte1]
a0,a1 = BASE2IDX[alt_byte0], BASE2IDX[alt_byte1]
valid = all >= 0
out[dbs] = where(valid, _DBS_TABLE[r0,r1,a0,a1], UNCLASSIFIED)
```

Context-free; no sequence needed. Non-catalogue doublets already return
`UNCLASSIFIED` from the table; `where` only guards LUT bounds for non-ACGT bytes.

### Stage 1c — ID-83 indel kernel (numba, parallel)

`@nb.njit(parallel=True, nogil=True, cache=True)`, `prange` over the indel subset
on one contig. Inputs: the contig `seq` (uint8), `pos0` (0-based anchor), REF/ALT
flat `uint8` buffers + `int32` offsets for the indel rows, and three precomputed
code-lookup arrays derived once from `ID83_INDEX`:

- `id1_code[kind, base, rep]` — 1 bp channel (kind∈{Del=0,Ins=1}, base∈{C=0,T=1}, rep 0–5)
- `idR_code[kind, size_bucket, rep]` — ≥2 bp repeat channel (size bucket 2,3,4,5+)
- `idM_code[size_bucket, mh]` — microhomology deletions

Per indel the kernel reproduces `classify_id83` exactly:

- Re-check anchor `ref[0]==alt[0]`; if not → `UNCLASSIFIED`.
- `is_del = rlen > alen`; indel unit = `ref[1:]` (del) or `alt[1:]` (ins); validate
  ACGT (else `UNCLASSIFIED`).
- Count tandem repeats of the unit by scanning `seq` downstream from `p+1`;
  reads past contig end simply stop matching (equivalent to today's `N`-padding).
- For deletions, compute microhomology length (partial prefix match, capped per
  size bucket).
- 1 bp: purine→pyrimidine base fold; `is_del and n_rep==0` → `_REF_MISMATCH`;
  else `rep = repeat_bucket(n_rep-1 if del else n_rep)`; `code = id1_code[...]`.
- ≥2 bp: if `is_del` and `mh>0` and `n_rep<=1` → `idM_code[...]`; else
  `is_del and n_rep==0` → `_REF_MISMATCH`; else `idR_code[...]`.

After the kernel, Python collects rows equal to `_REF_MISMATCH`, emits the
**same aggregated warning** (count + up to 5 `chrom:pos` examples) and sets them
to `UNCLASSIFIED`. UX is unchanged from the current implementation.

### Stage 2 — `build_entry_codes` parallelism

Add `parallel=True` and `prange` over tracks (the `slot` loop). Each track writes
only its own `[o_s, o_e)` range of `out`, so there are no write races. The DBS
adjacency logic within a track is unchanged.

### Stage 3 — `count_matrix` parallelism

Parallelize over **samples** (not slots): each thread owns disjoint rows of the
`(n_samples, N_CODES)` accumulator, so no atomics and no false sharing on the
sample axis. A sample's slots are the contiguous `ploidy` slots
`[sample*ploidy, (sample+1)*ploidy)`. Both `allele` and `sample` (presence) modes
stay in the one kernel.

## Reference access change

Add `Reference.contig_array(contig) -> NDArray[uint8]` that returns the cached
full-contig array (it already builds `_cur_seq` in `_load_contig`; this exposes
it, preserving the one-contig-in-memory cache and `ContigNormalizer` behavior).
The vectorized SNV gather and the ID-83 kernel read this array directly instead
of per-variant `fetch()` calls. `fetch()` is retained for other callers and as
the scalar oracle's backend.

## Testing

Oracle-based, exploiting the "common cases must match" contract:

1. **Codebook arithmetic** — assert the SBS-96 arithmetic code equals
   `SBS96_INDEX[label]` for all 96 labels, and the ID-83 lookup arrays equal
   `ID83_INDEX` for every channel. No reference genome needed.
2. **Differential / property tests** — synthetic reference + randomized
   SNV/DBS/indel variants; assert the new `classify_variants` output equals the
   old scalar path element-for-element on ACGT inputs. Extend
   `tests/test_mutcat.py` and `tests/test_mutcat_calibration.py`; the scalar
   `classify_sbs96/dbs78/id83` remain as the oracle.
3. **Regression** — existing `tests/test_svar_mutations.py` (incl. the #59 POS
   off-by-one and the deletion REF-mismatch warning) must pass unchanged; add a
   contig-boundary variant case (first/last base of a contig).
4. **Parallelism determinism** — results identical with `NUMBA_NUM_THREADS=1`
   vs many, so threading cannot reorder or corrupt counts.

No formal benchmark gate (consistent with repo convention — no existing perf
harness). The plan includes an optional ad-hoc timing script on a large `.svar`
to confirm the speedup.

## Scope

**In scope:** vectorize SNV/DBS classification, numba indel kernel, parallelize
entry-code and count kernels, `Reference.contig_array`, oracle-based tests.

**Out of scope:** changing the public API or code space; new context sizes
(SBS-192/384, DBS-186); runs of ≥3 adjacent SNVs (still individual SBS, handled
by the unchanged `build_entry_codes` adjacency logic); a thread-count API knob;
a committed benchmark harness.

## Files

- **Modify** `genoray/_mutcat.py` — reimplement `classify_variants`; add
  `_sbs96_vec`, `_dbs78_vec`, the ID-83 numba kernel and its derived lookup
  arrays; add `parallel=True`/`prange` to `_entry_codes_kernel` and
  `_count_kernel`. Keep scalar `classify_*` as oracle.
- **Modify** `genoray/_reference.py` — add `contig_array`.
- **Modify** `tests/test_mutcat.py`, `tests/test_mutcat_calibration.py`,
  `tests/test_svar_mutations.py` — codebook, differential, regression,
  determinism tests.

# Vectorize SBS/DBS/INDEL Matrix Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-variant Python loop in `classify_variants` with vectorized numpy (SNV/DBS) plus one parallel numba kernel (indels), and parallelize the entry-code and count kernels, with results identical to the current implementation on all common ACGT SBS/DBS/ID cases.

**Architecture:** Hybrid. SNV→SBS-96 and native-doublet→DBS-78 become pure-numpy fancy-index passes; the indel→ID-83 repeat/microhomology scan becomes a single `nb.njit(parallel=True)` kernel reading a per-contig reference array. The existing `_entry_codes_kernel` and `_count_kernel` gain `prange`. The current scalar classifiers are retained as the test oracle.

**Tech Stack:** Python, NumPy, numba (`prange`, `parallel=True`), Polars + PyArrow (zero-copy REF/ALT byte buffers), pysam (`Reference`), pytest. All commands run under Pixi: prefix with `pixi run`.

**Spec:** `docs/superpowers/specs/2026-06-13-vectorize-mutation-matrices-design.md`

---

## Key facts (read before starting)

- `genoray/_mutcat.py` holds the codebooks, scalar classifiers, kernels, and `classify_variants`. `genoray/_reference.py` holds `Reference`.
- `classify_variants(index, reference)` (`_mutcat.py:478`) takes a `pl.DataFrame` with columns `CHROM` (str), `POS` (1-based int), `REF` (str), `ALT` (List[str]; first used) and returns `int16[index.height]`. **POS is 1-based; the function converts to 0-based internally (`p = POS - 1`).**
- Base encoding used throughout: `A=0, C=1, G=2, T=3`. Complement under this encoding is `3 - x` (A↔T, C↔G). Purine refs are `A` and `G` (encoded `0` and `2`).
- `SBS96 = [f"{five}[{sub}]{three}" for sub in _SBS_SUBS for five in _BASES for three in _BASES]` with `_SBS_SUBS = ["C>A","C>G","C>T","T>A","T>C","T>G"]` and `_BASES = ["A","C","G","T"]`. So the code for a folded SNV is `sub_idx*16 + five_idx*4 + three_idx`.
- `_DBS_TABLE` (`_mutcat.py:326`) is a `(4,4,4,4)` int16 array `[r0,r1,a0,a1] -> code | UNCLASSIFIED`, already built from the scalar `classify_dbs78`.
- `_BASE2IDX` (`_mutcat.py:307`) is a 256-entry int64 LUT: `A/C/G/T -> 0/1/2/3`, else `-1`.
- Sentinels: `SENTINELS["UNCLASSIFIED"] = -2`, `SENTINELS["DBS_PARTNER"] = -1`; `_REF_MISMATCH = -99` (internal).
- Polars `Series.to_arrow()` yields a `pyarrow.LargeStringArray` (int64 offsets). `arr.buffers()` is `[validity, offsets(int64), data(uint8)]`. PyArrow ≥21 is a hard dependency.
- numba thread count is controlled at runtime by `numba.set_num_threads(n)` and `numba.get_num_threads()`. Use these in determinism tests.
- Tests live in `tests/`, import private helpers directly, and run via `pixi run pytest tests/<file> -v`.
- Commit convention: Conventional Commits (`feat:`, `test:`, `perf:`, `refactor:`, `docs:`).

---

## File structure

- **Modify** `genoray/_reference.py` — add `Reference.contig_array`.
- **Modify** `genoray/_mutcat.py` — add `_SUB_LUT`; `_sbs96_codes`; `_dbs78_codes`; `_build_id83_luts` + module-level `_ID1_CODE/_IDR_CODE/_IDM_CODE`; `_id83_kernel`; `_utf8_flat`; rename the current `classify_variants` body to `_classify_variants_scalar` (oracle) and write a new vectorized `classify_variants`; add `parallel=True`/`prange` to `_entry_codes_kernel` and `_count_kernel`.
- **Modify** `tests/test_mutcat.py` — codebook-arithmetic, per-classifier differential, kernel-vs-scalar, full differential, boundary, determinism tests.
- **Modify** `tests/test_svar_mutations.py` — parallel-determinism regression for `mutation_matrix`.

---

## Task 1: Add `Reference.contig_array`

Expose the already-cached full-contig uint8 array so the vectorized paths can gather flanks and the indel kernel can scan downstream, without per-variant `fetch()` calls.

**Files:**
- Modify: `genoray/_reference.py`
- Test: `tests/test_reference.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_reference.py`:

```python
import numpy as np
import pysam

from genoray._reference import Reference


def _write_fasta(path, seq, contig="chr1"):
    path.write_text(f">{contig}\n{seq}\n")
    pysam.faidx(str(path))


def test_contig_array_matches_fetch(tmp_path):
    fa = tmp_path / "ref.fa"
    _write_fasta(fa, "ACGTACGTAC")
    ref = Reference.from_path(fa)
    arr = ref.contig_array("chr1")
    assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
    assert bytes(arr) == b"ACGTACGTAC"
    # equals fetch over the full span
    assert bytes(ref.fetch("chr1", 0, 10)) == bytes(arr)
    # chr-prefix normalization still works
    assert bytes(ref.contig_array("1")) == b"ACGTACGTAC"
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_reference.py::test_contig_array_matches_fetch -v`
Expected: FAIL — `AttributeError: 'Reference' object has no attribute 'contig_array'`.

- [ ] **Step 3: Implement `contig_array`**

In `genoray/_reference.py`, add this method to `Reference` (after `_load_contig`, before `fetch`):

```python
    def contig_array(self, contig: str) -> NDArray[np.uint8]:
        """Return the full contig sequence as a cached uint8 array.

        Accepts ``chr``-prefixed or unprefixed names. One contig is held in
        memory at a time (shared with :meth:`fetch`).
        """
        return self._load_contig(contig)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run pytest tests/test_reference.py::test_contig_array_matches_fetch -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_reference.py tests/test_reference.py
git commit -m "feat(reference): expose cached contig_array for vectorized lookups"
```

---

## Task 2: Vectorized SBS-96 classifier `_sbs96_codes`

Compute SBS-96 codes for a batch of SNVs on one contig with pure numpy, and prove the arithmetic matches the codebook order.

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mutcat.py`:

```python
from genoray._mutcat import _sbs96_codes, classify_sbs96  # noqa: E402


def test_sbs96_arithmetic_matches_codebook():
    # The vectorized substitution LUT + arithmetic must reproduce SBS96_INDEX
    # for every one of the 96 labels (pyrimidine-folded form, no boundary issues).
    from genoray._mutcat import _BASE2IDX, _SUB_LUT, _BASES, _SBS_SUBS

    for sub_idx, sub in enumerate(_SBS_SUBS):
        r_char, a_char = sub[0], sub[2]
        r, a = _BASE2IDX[ord(r_char)], _BASE2IDX[ord(a_char)]
        assert _SUB_LUT[r, a] == sub_idx
        for five_idx, five in enumerate(_BASES):
            for three_idx, three in enumerate(_BASES):
                label = f"{five}[{sub}]{three}"
                code = sub_idx * 16 + five_idx * 4 + three_idx
                assert code == SBS96_INDEX[label]


def test_sbs96_codes_match_scalar_on_random_snvs():
    rng = np.random.default_rng(0)
    bases = np.frombuffer(b"ACGT", np.uint8)
    n = 500
    seq = np.frombuffer(bytes(bases[rng.integers(0, 4, 200)]), np.uint8)
    # interior positions only so flanks exist
    p0 = rng.integers(1, len(seq) - 1, n).astype(np.int64)
    ref_b = seq[p0].copy()  # ref must equal reference base is NOT required by classify,
    alt_b = bases[rng.integers(0, 4, n)].copy()
    got = _sbs96_codes(seq, p0, ref_b, alt_b)
    for i in range(n):
        five = bytes(seq[p0[i] - 1 : p0[i]])
        three = bytes(seq[p0[i] + 1 : p0[i] + 2])
        exp = classify_sbs96(five, bytes(ref_b[i : i + 1]), bytes(alt_b[i : i + 1]), three)
        assert got[i] == exp, (i, five, ref_b[i], alt_b[i], three)
```

- [ ] **Step 2: Run them to verify failure**

Run: `pixi run pytest tests/test_mutcat.py -k sbs96_ -v`
Expected: FAIL — `ImportError: cannot import name '_sbs96_codes'` (and `_SUB_LUT`).

- [ ] **Step 3: Implement `_SUB_LUT` and `_sbs96_codes`**

In `genoray/_mutcat.py`, after the `_BASE2IDX` definition (`:307-308`), add:

```python
# (ref_idx, alt_idx) -> SBS substitution index 0..5, for pyrimidine-folded refs.
# _SBS_SUBS order: C>A, C>G, C>T, T>A, T>C, T>G  (encoding A=0,C=1,G=2,T=3)
_SUB_LUT = np.full((4, 4), -1, dtype=np.int64)
for _si, _sub in enumerate(_SBS_SUBS):
    _SUB_LUT[_BASE2IDX[ord(_sub[0])], _BASE2IDX[ord(_sub[2])]] = _si

_UNCL = np.int16(SENTINELS["UNCLASSIFIED"])


def _sbs96_codes(
    seq: NDArray[np.uint8],
    p0: NDArray[np.int64],
    ref_b: NDArray[np.uint8],
    alt_b: NDArray[np.uint8],
) -> NDArray[np.int16]:
    """SBS-96 codes for SNVs on one contig. ``p0`` is the 0-based REF position."""
    n = len(seq)
    r = _BASE2IDX[ref_b]
    a = _BASE2IDX[alt_b]
    fpos = p0 - 1
    tpos = p0 + 1
    in_f = (fpos >= 0) & (fpos < n)
    in_t = (tpos >= 0) & (tpos < n)
    f = _BASE2IDX[seq[np.clip(fpos, 0, n - 1)]]
    t = _BASE2IDX[seq[np.clip(tpos, 0, n - 1)]]
    valid = (r >= 0) & (a >= 0) & (f >= 0) & (t >= 0) & (r != a) & in_f & in_t
    purine = (r == 0) | (r == 2)  # A or G
    rr = np.where(purine, 3 - r, r)
    aa = np.where(purine, 3 - a, a)
    # flanks swap and complement when folding: new 5' = comp(old 3'), new 3' = comp(old 5')
    ff = np.where(purine, 3 - t, f)
    tt = np.where(purine, 3 - f, t)
    sub = _SUB_LUT[np.clip(rr, 0, 3), np.clip(aa, 0, 3)]
    code = (sub * 16 + ff * 4 + tt).astype(np.int16)
    return np.where(valid, code, _UNCL)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_mutcat.py -k sbs96_ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): vectorized SBS-96 batch classifier"
```

---

## Task 3: Vectorized DBS-78 classifier `_dbs78_codes`

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mutcat.py`:

```python
from genoray._mutcat import _dbs78_codes  # noqa: E402


def test_dbs78_codes_match_scalar():
    bases = b"ACGT"
    ref0, ref1, alt0, alt1, exp = [], [], [], [], []
    for r0 in bases:
        for r1 in bases:
            for a0 in bases:
                for a1 in bases:
                    ref0.append(r0); ref1.append(r1)
                    alt0.append(a0); alt1.append(a1)
                    exp.append(
                        classify_dbs78(bytes([r0, r1]), bytes([a0, a1]))
                    )
    got = _dbs78_codes(
        np.array(ref0, np.uint8), np.array(ref1, np.uint8),
        np.array(alt0, np.uint8), np.array(alt1, np.uint8),
    )
    assert list(got) == exp
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_mutcat.py::test_dbs78_codes_match_scalar -v`
Expected: FAIL — `ImportError: cannot import name '_dbs78_codes'`.

- [ ] **Step 3: Implement `_dbs78_codes`**

In `genoray/_mutcat.py`, after `_sbs96_codes`, add:

```python
def _dbs78_codes(
    ref_b0: NDArray[np.uint8],
    ref_b1: NDArray[np.uint8],
    alt_b0: NDArray[np.uint8],
    alt_b1: NDArray[np.uint8],
) -> NDArray[np.int16]:
    """DBS-78 codes for native 2bp doublets (context-free table lookup)."""
    r0 = _BASE2IDX[ref_b0]
    r1 = _BASE2IDX[ref_b1]
    a0 = _BASE2IDX[alt_b0]
    a1 = _BASE2IDX[alt_b1]
    valid = (r0 >= 0) & (r1 >= 0) & (a0 >= 0) & (a1 >= 0)
    code = _DBS_TABLE[
        np.clip(r0, 0, 3), np.clip(r1, 0, 3), np.clip(a0, 0, 3), np.clip(a1, 0, 3)
    ]
    return np.where(valid, code, _UNCL)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run pytest tests/test_mutcat.py::test_dbs78_codes_match_scalar -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): vectorized DBS-78 batch classifier"
```

---

## Task 4: ID-83 lookup arrays + parallel numba kernel

Replace `classify_id83`'s Python window scan with a `prange` kernel that reads the contig array directly. Code labels are mapped to int16 codes via arrays derived once from `ID83_INDEX`.

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mutcat.py`:

```python
from genoray._mutcat import (  # noqa: E402
    _ID1_CODE,
    _IDM_CODE,
    _IDR_CODE,
    _id83_codes_for_contig,
)


def test_id83_luts_match_index():
    for ki, k in enumerate(("Del", "Ins")):
        for bi, b in enumerate(("C", "T")):
            for r in range(6):
                assert _ID1_CODE[ki, bi, r] == ID83_INDEX[f"1:{k}:{b}:{r}"]
        for si, s in enumerate(("2", "3", "4", "5")):
            for r in range(6):
                assert _IDR_CODE[ki, si, r] == ID83_INDEX[f"{s}:{k}:R:{r}"]
    for si, s in enumerate(("2", "3", "4", "5")):
        cap = {"2": 1, "3": 2, "4": 3, "5": 5}[s]
        for m in range(1, cap + 1):
            assert _IDM_CODE[si, m] == ID83_INDEX[f"{s}:Del:M:{m}"]


def test_id83_kernel_matches_scalar_random():
    # Build a random contig and a set of indels; compare to classify_id83.
    rng = np.random.default_rng(7)
    bases = b"ACGT"
    seq = np.frombuffer(
        bytes(np.frombuffer(bases, np.uint8)[rng.integers(0, 4, 400)]), np.uint8
    )

    def fetch(s, e, _seq=seq):
        out = np.full(e - s, ord("N"), np.uint8)
        a, b = max(s, 0), min(e, len(_seq))
        if b > a:
            out[a - s : b - s] = _seq[a:b]
        return bytes(out)

    refs, alts, p0s = [], [], []
    for _ in range(300):
        p = int(rng.integers(2, len(seq) - 10))
        anchor = bytes(seq[p : p + 1])
        size = int(rng.integers(1, 5))
        unit = bytes(np.frombuffer(bases, np.uint8)[rng.integers(0, 4, size)])
        if rng.random() < 0.5:  # deletion
            refs.append(anchor + unit); alts.append(anchor)
        else:  # insertion
            refs.append(anchor); alts.append(anchor + unit)
        p0s.append(p)

    # flat buffers for the kernel
    ref_data = np.frombuffer(b"".join(refs), np.uint8)
    alt_data = np.frombuffer(b"".join(alts), np.uint8)
    ref_off = np.concatenate(([0], np.cumsum([len(r) for r in refs]))).astype(np.int64)
    alt_off = np.concatenate(([0], np.cumsum([len(a) for a in alts]))).astype(np.int64)
    p0 = np.array(p0s, np.int64)

    got = _id83_codes_for_contig(
        seq, p0, ref_data, ref_off[:-1], (ref_off[1:] - ref_off[:-1]),
        alt_data, alt_off[:-1], (alt_off[1:] - alt_off[:-1]),
    )
    for i in range(len(refs)):
        exp = classify_id83(int(p0[i]), refs[i], alts[i], fetch)
        assert got[i] == exp, (i, refs[i], alts[i], int(p0[i]), got[i], exp)
```

- [ ] **Step 2: Run them to verify failure**

Run: `pixi run pytest tests/test_mutcat.py -k id83 -v`
Expected: FAIL — `ImportError: cannot import name '_ID1_CODE'`.

- [ ] **Step 3: Implement the LUTs, kernel, and Python wrapper**

In `genoray/_mutcat.py`, after `_dbs78_codes`, add:

```python
def _build_id83_luts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = SENTINELS["UNCLASSIFIED"]
    id1 = np.full((2, 2, 6), u, dtype=np.int16)  # [kind(Del,Ins), base(C,T), rep]
    idr = np.full((2, 4, 6), u, dtype=np.int16)  # [kind, size_bucket(2..5), rep]
    idm = np.full((4, 6), u, dtype=np.int16)  # [size_bucket(2..5), mh]
    for ki, k in enumerate(("Del", "Ins")):
        for bi, b in enumerate(("C", "T")):
            for r in range(6):
                id1[ki, bi, r] = ID83_INDEX[f"1:{k}:{b}:{r}"]
        for si, s in enumerate(("2", "3", "4", "5")):
            for r in range(6):
                idr[ki, si, r] = ID83_INDEX[f"{s}:{k}:R:{r}"]
    for si, (s, cap) in enumerate((("2", 1), ("3", 2), ("4", 3), ("5", 5))):
        for m in range(1, cap + 1):
            idm[si, m] = ID83_INDEX[f"{s}:Del:M:{m}"]
    return id1, idr, idm


_ID1_CODE, _IDR_CODE, _IDM_CODE = _build_id83_luts()
_MH_CAP = np.array([1, 2, 3, 5], dtype=np.int64)  # by size bucket 2,3,4,5


@nb.njit(parallel=True, nogil=True, cache=True)
def _id83_kernel(
    seq: NDArray[np.uint8],
    p0: NDArray[np.int64],
    ref_data: NDArray[np.uint8],
    ref_s: NDArray[np.int64],
    ref_len: NDArray[np.int64],
    alt_data: NDArray[np.uint8],
    alt_s: NDArray[np.int64],
    alt_len: NDArray[np.int64],
    base2idx: NDArray[np.int64],
    id1: NDArray[np.int16],
    idr: NDArray[np.int16],
    idm: NDArray[np.int16],
    mh_cap: NDArray[np.int64],
    uncl: np.int16,
    ref_mismatch: np.int16,
    out: NDArray[np.int16],
) -> None:
    n = len(seq)
    for k in nb.prange(len(p0)):
        rs, asz = ref_s[k], alt_s[k]
        rl, al = ref_len[k], alt_len[k]
        if rl == 0 or al == 0 or ref_data[rs] != alt_data[asz]:
            out[k] = uncl
            continue
        is_del = rl > al
        if is_del:
            buf, us, ilen = ref_data, rs + 1, rl - 1
        else:
            buf, us, ilen = alt_data, asz + 1, al - 1
        ok = True
        for i in range(ilen):
            if base2idx[buf[us + i]] < 0:
                ok = False
                break
        if not ok:
            out[k] = uncl
            continue
        # count tandem repeats of the unit downstream from p0+1
        scan = p0[k] + 1
        n_rep = 0
        i = 0
        while scan + i + ilen <= n:
            match = True
            for j in range(ilen):
                if seq[scan + i + j] != buf[us + j]:
                    match = False
                    break
            if not match:
                break
            n_rep += 1
            i += ilen
        if ilen == 1:
            bi = base2idx[buf[us]]
            if bi == 0 or bi == 2:  # A or G -> fold to pyrimidine partner
                bi = 3 - bi
            base_idx = 0 if bi == 1 else 1  # C->0, T->1
            if is_del and n_rep == 0:
                out[k] = ref_mismatch
                continue
            rep = n_rep - 1 if is_del else n_rep
            if rep > 5:
                rep = 5
            out[k] = id1[0 if is_del else 1, base_idx, rep]
            continue
        sb = ilen if ilen < 5 else 5
        si = sb - 2  # 0..3
        if is_del:
            mh = 0
            for kk in range(1, ilen):
                eq = True
                for j in range(kk):
                    if scan + j >= n or seq[scan + j] != buf[us + j]:
                        eq = False
                        break
                if eq:
                    mh = kk
            if mh > 0 and n_rep <= 1:
                cap = mh_cap[si]
                m = mh if mh < cap else cap
                out[k] = idm[si, m]
                continue
            if n_rep == 0:
                out[k] = ref_mismatch
                continue
            rep = n_rep - 1
        else:
            rep = n_rep
        if rep > 5:
            rep = 5
        out[k] = idr[0 if is_del else 1, si, rep]


def _id83_codes_for_contig(
    seq: NDArray[np.uint8],
    p0: NDArray[np.int64],
    ref_data: NDArray[np.uint8],
    ref_s: NDArray[np.int64],
    ref_len: NDArray[np.int64],
    alt_data: NDArray[np.uint8],
    alt_s: NDArray[np.int64],
    alt_len: NDArray[np.int64],
) -> NDArray[np.int16]:
    """ID-83 codes for indels on one contig. May return ``_REF_MISMATCH``
    entries, which the caller maps to UNCLASSIFIED with a warning."""
    out = np.empty(len(p0), dtype=np.int16)
    _id83_kernel(
        np.ascontiguousarray(seq),
        np.ascontiguousarray(p0),
        np.ascontiguousarray(ref_data),
        np.ascontiguousarray(ref_s),
        np.ascontiguousarray(ref_len),
        np.ascontiguousarray(alt_data),
        np.ascontiguousarray(alt_s),
        np.ascontiguousarray(alt_len),
        _BASE2IDX,
        _ID1_CODE,
        _IDR_CODE,
        _IDM_CODE,
        _MH_CAP,
        np.int16(SENTINELS["UNCLASSIFIED"]),
        np.int16(_REF_MISMATCH),
        out,
    )
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_mutcat.py -k id83 -v`
Expected: PASS (the kernel compiles on first call; allow a few seconds).

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): parallel numba ID-83 indel kernel + derived LUTs"
```

---

## Task 5: Rewrite `classify_variants` (vectorized), keep scalar oracle

Rename the current implementation to `_classify_variants_scalar` (the test oracle), then write the new vectorized `classify_variants` that dispatches by variant type and groups by contig.

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Rename the current implementation to the oracle**

In `genoray/_mutcat.py`, rename the existing `def classify_variants(` (`:478`) to `def _classify_variants_scalar(` — keep its body byte-for-byte identical. This preserves the exact current behavior as a reference for differential tests.

- [ ] **Step 2: Write the failing differential + boundary tests**

Append to `tests/test_mutcat.py`:

```python
from genoray._mutcat import _classify_variants_scalar  # noqa: E402


def _random_index_and_ref(tmp_path, rng):
    import pysam

    seq = "".join(rng.choice(list("ACGT"), size=300))
    fa = tmp_path / "ref.fa"
    fa.write_text(f">chr1\n{seq}\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)

    chrom, pos, refs, alts = [], [], [], []
    for _ in range(400):
        p = int(rng.integers(2, len(seq) - 8))  # 0-based interior
        kind = rng.integers(0, 4)
        anchor = seq[p]
        if kind == 0:  # SNV
            alt = rng.choice([b for b in "ACGT" if b != anchor])
            refs.append(anchor); alts.append(alt)
        elif kind == 1:  # native doublet
            refs.append(seq[p : p + 2])
            alts.append("".join(rng.choice(list("ACGT"), 2)))
        elif kind == 2:  # deletion
            size = int(rng.integers(1, 4))
            refs.append(seq[p : p + 1 + size]); alts.append(anchor)
        else:  # insertion
            size = int(rng.integers(1, 4))
            refs.append(anchor)
            alts.append(anchor + "".join(rng.choice(list("ACGT"), size)))
        chrom.append("chr1"); pos.append(p + 1)  # store 1-based

    index = pl.DataFrame(
        {"CHROM": chrom, "POS": pos, "REF": refs, "ALT": [[a] for a in alts]}
    )
    return index, ref


def test_classify_variants_matches_scalar(tmp_path):
    rng = np.random.default_rng(123)
    index, ref = _random_index_and_ref(tmp_path, rng)
    got = classify_variants(index, ref)
    exp = _classify_variants_scalar(index, ref)
    assert list(got) == list(exp)


def test_classify_variants_contig_boundary(tmp_path):
    import pysam

    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGT\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)
    # SNV at first and last base: no 5'/3' flank -> UNCLASSIFIED, matching scalar
    index = pl.DataFrame(
        {"CHROM": ["chr1", "chr1"], "POS": [1, 4], "REF": ["A", "T"],
         "ALT": [["C"], ["G"]]}
    )
    got = classify_variants(index, ref)
    exp = _classify_variants_scalar(index, ref)
    assert list(got) == list(exp) == [SENTINELS["UNCLASSIFIED"]] * 2
```

- [ ] **Step 3: Run them to verify failure**

Run: `pixi run pytest tests/test_mutcat.py -k classify_variants -v`
Expected: FAIL — `classify_variants` still points at the old (now renamed) name, so import of `classify_variants` fails (`ImportError`), or the differential test errors. Confirm it fails before implementing.

- [ ] **Step 4: Implement `_utf8_flat` and the vectorized `classify_variants`**

In `genoray/_mutcat.py`, add `import pyarrow as pa` near the top imports, then add (placing `classify_variants` where the old one was):

```python
def _utf8_flat(s: "pl.Series") -> tuple[NDArray[np.uint8], NDArray[np.int64], NDArray[np.bool_]]:
    """Zero-copy flat byte buffer + int64 offsets + not-null mask for a Utf8 series."""
    arr = s.rechunk().to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    assert arr.offset == 0
    bufs = arr.buffers()
    offsets = np.frombuffer(bufs[1], dtype=np.int64)[: len(s) + 1]
    data = (
        np.frombuffer(bufs[2], dtype=np.uint8)
        if bufs[2] is not None
        else np.empty(0, dtype=np.uint8)
    )
    not_null = s.is_not_null().to_numpy()
    return data, offsets, not_null


def classify_variants(index: "pl.DataFrame", reference: Reference) -> np.ndarray:
    """Return an int16 array of intrinsic mutation codes, one per row of ``index``.

    Vectorized: SNV->SBS-96 and native 2bp doublet->DBS-78 via numpy; indels->ID-83
    via a parallel numba kernel. ``index`` must have columns CHROM, POS (1-based),
    REF (str), ALT (List[str]; first used). POS is converted to 0-based internally.
    """
    n = index.height
    out = np.full(n, SENTINELS["UNCLASSIFIED"], dtype=np.int16)
    if n == 0:
        return out

    chrom = index["CHROM"].to_numpy()
    pos0 = index["POS"].to_numpy().astype(np.int64) - 1  # 1-based -> 0-based
    ref_data, ref_off, ref_nn = _utf8_flat(index["REF"])
    alt_data, alt_off, alt_nn = _utf8_flat(index["ALT"].list.first())

    rlen = (ref_off[1:] - ref_off[:-1]).astype(np.int64)
    alen = (alt_off[1:] - alt_off[:-1]).astype(np.int64)
    valid_row = ref_nn & alt_nn
    snv_mask = valid_row & (rlen == 1) & (alen == 1)
    dbs_mask = valid_row & (rlen == 2) & (alen == 2)
    indel_mask = valid_row & (rlen != alen)

    n_mismatch = 0
    mismatch_examples: list[str] = []

    uniq, inv = np.unique(chrom, return_inverse=True)
    for gi, c in enumerate(uniq):
        rows = np.nonzero(inv == gi)[0]
        seq = reference.contig_array(str(c))

        s_rows = rows[snv_mask[rows]]
        if len(s_rows):
            out[s_rows] = _sbs96_codes(
                seq, pos0[s_rows], ref_data[ref_off[s_rows]], alt_data[alt_off[s_rows]]
            )

        d_rows = rows[dbs_mask[rows]]
        if len(d_rows):
            sr, sa = ref_off[d_rows], alt_off[d_rows]
            out[d_rows] = _dbs78_codes(
                ref_data[sr], ref_data[sr + 1], alt_data[sa], alt_data[sa + 1]
            )

        i_rows = rows[indel_mask[rows]]
        if len(i_rows):
            codes = _id83_codes_for_contig(
                seq, pos0[i_rows],
                ref_data, ref_off[i_rows], rlen[i_rows],
                alt_data, alt_off[i_rows], alen[i_rows],
            )
            mm = codes == _REF_MISMATCH
            if mm.any():
                mm_rows = i_rows[mm]
                n_mismatch += int(mm.sum())
                for ri in mm_rows[:5]:
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append(f"{chrom[ri]}:{int(pos0[ri]) + 1}")
                codes = codes.copy()
                codes[mm] = SENTINELS["UNCLASSIFIED"]
            out[i_rows] = codes

    if n_mismatch:
        examples = ", ".join(mismatch_examples)
        logger.warning(
            f"{n_mismatch}/{n} deletions have REF disagreeing with the "
            f"reference genome at their position (e.g. {examples}) — wrong reference "
            "build? These were marked UNCLASSIFIED."
        )

    return out
```

- [ ] **Step 5: Run the differential and boundary tests**

Run: `pixi run pytest tests/test_mutcat.py -k classify_variants -v`
Expected: PASS.

- [ ] **Step 6: Run the full mutcat + calibration suite (no regressions)**

Run: `pixi run pytest tests/test_mutcat.py tests/test_mutcat_calibration.py -v`
Expected: PASS (existing calibration test still matches SigProfiler).

- [ ] **Step 7: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "perf(mutcat): vectorize classify_variants (SNV/DBS numpy, indel kernel)"
```

---

## Task 6: Parallelize `_entry_codes_kernel`

Each track writes only its own `[o_s, o_e)` slice, so the slot loop is race-free under `prange`.

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing determinism test**

Append to `tests/test_mutcat.py`:

```python
import numba as nb  # noqa: E402


def test_entry_codes_thread_invariant():
    rng = np.random.default_rng(5)
    n_var = 200
    var_code = rng.integers(0, 96, n_var).astype(np.int16)
    var_pos = np.sort(rng.integers(0, 10_000, n_var)).astype(np.int64)
    var_contig = np.zeros(n_var, np.int32)
    var_is_snv = np.ones(n_var, np.bool_)
    var_ref_b = np.frombuffer(b"ACGT", np.uint8)[rng.integers(0, 4, n_var)].copy()
    var_alt_b = np.frombuffer(b"ACGT", np.uint8)[rng.integers(0, 4, n_var)].copy()
    # 8 tracks over the variant indices
    data = np.tile(np.arange(n_var, dtype=np.int32), 8)
    offsets = (np.arange(9) * n_var).astype(np.int64)

    args = (data, offsets, var_code, var_pos, var_contig, var_is_snv,
            var_ref_b, var_alt_b)
    prev = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        a = build_entry_codes(*args)
        nb.set_num_threads(max(2, prev))
        b = build_entry_codes(*args)
    finally:
        nb.set_num_threads(prev)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run it to verify it passes single-threaded but with current serial kernel**

Run: `pixi run pytest tests/test_mutcat.py::test_entry_codes_thread_invariant -v`
Expected: PASS even now (kernel is serial). This test is a guard for Step 3; keep it.

- [ ] **Step 3: Add `prange` parallelism to `_entry_codes_kernel`**

In `genoray/_mutcat.py`, change the decorator on `_entry_codes_kernel` (`:330`) from
`@nb.njit(nogil=True, cache=True)` to `@nb.njit(parallel=True, nogil=True, cache=True)`,
and change the outer loop
`for slot in range(len(offsets) - 1):` to
`for slot in nb.prange(len(offsets) - 1):`.
Leave the inner `while j < o_e:` body unchanged.

- [ ] **Step 4: Run the determinism test (now parallel)**

Run: `pixi run pytest tests/test_mutcat.py::test_entry_codes_thread_invariant -v`
Expected: PASS — output identical across thread counts.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "perf(mutcat): parallelize entry-code kernel over tracks"
```

---

## Task 7: Parallelize `_count_kernel` over samples

`prange` over slots would race when `ploidy > 1` (two slots share one sample row). Parallelize over **samples** so each thread owns a disjoint accumulator row.

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing determinism test**

Append to `tests/test_mutcat.py`:

```python
def test_count_matrix_thread_invariant():
    rng = np.random.default_rng(9)
    n_samples, ploidy = 6, 2
    per_track = 50
    n_slots = n_samples * ploidy
    entry_codes = rng.integers(0, 96, n_slots * per_track).astype(np.int16)
    offsets = (np.arange(n_slots + 1) * per_track).astype(np.int64)
    names = [f"s{i}" for i in range(n_samples)]

    from genoray._mutcat import count_matrix

    prev = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        a = count_matrix(entry_codes, offsets, ploidy, n_samples, names, "SBS96", False)
        nb.set_num_threads(max(2, prev))
        b = count_matrix(entry_codes, offsets, ploidy, n_samples, names, "SBS96", False)
    finally:
        nb.set_num_threads(prev)
    assert a.equals(b)
```

- [ ] **Step 2: Run it to verify it passes with the current serial kernel**

Run: `pixi run pytest tests/test_mutcat.py::test_count_matrix_thread_invariant -v`
Expected: PASS now (serial). Guard for Step 3.

- [ ] **Step 3: Rewrite `_count_kernel` to parallelize over samples**

In `genoray/_mutcat.py`, replace the `_count_kernel` body (`:422-448`) with:

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def _count_kernel(
    data_codes: NDArray[np.int16],
    offsets: NDArray[np.int64],
    ploidy: np.int64,
    n_samples: np.int64,
    n_codes: np.int64,
    per_sample: np.bool_,
    out: NDArray[np.int64],
) -> None:
    """out[sample, code] accumulator over genotype entries.

    Parallelized over samples: each thread owns a disjoint row of ``out``.
    When ``per_sample`` is True, a code is counted at most once per sample.
    """
    for sample in nb.prange(n_samples):
        for slot in range(sample * ploidy, (sample + 1) * ploidy):
            o_s, o_e = offsets[slot], offsets[slot + 1]
            for j in range(o_s, o_e):
                code = data_codes[j]
                if code < 0 or code >= n_codes:
                    continue
                if per_sample:
                    out[sample, code] = 1
                else:
                    out[sample, code] += 1
```

- [ ] **Step 4: Run the determinism test (now parallel)**

Run: `pixi run pytest tests/test_mutcat.py::test_count_matrix_thread_invariant -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "perf(mutcat): parallelize count kernel over samples"
```

---

## Task 8: End-to-end determinism regression + full suite

Confirm the whole `annotate_mutations` → `mutation_matrix` path is thread-count invariant and that nothing regressed.

**Files:**
- Test: `tests/test_svar_mutations.py`

- [ ] **Step 1: Write the end-to-end determinism test**

Add to `tests/test_svar_mutations.py` (reuse the file's existing fixtures for building an annotated `.svar`; mirror the construction used by the other tests there). The assertion:

```python
def test_mutation_matrix_thread_invariant(tmp_path):
    import numba as nb

    svar, ref = _build_annotated_svar(tmp_path)  # use this file's existing helper/fixture
    prev = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        svar.annotate_mutations(ref)
        a = svar.mutation_matrix("SBS96")
        nb.set_num_threads(max(2, prev))
        svar.annotate_mutations(ref)
        b = svar.mutation_matrix("SBS96")
    finally:
        nb.set_num_threads(prev)
    assert a.equals(b)
```

If no shared helper exists in `tests/test_svar_mutations.py`, inline the same `.svar`-building steps already used by the nearest existing test in that file (do not invent a new fixture API).

- [ ] **Step 2: Run it**

Run: `pixi run pytest tests/test_svar_mutations.py::test_mutation_matrix_thread_invariant -v`
Expected: PASS.

- [ ] **Step 3: Run the full affected suite**

Run: `pixi run pytest tests/test_mutcat.py tests/test_mutcat_calibration.py tests/test_svar_mutations.py tests/test_reference.py -v`
Expected: PASS.

- [ ] **Step 4: Lint/format**

Run: `pixi run ruff check genoray tests && pixi run ruff format genoray tests`
Expected: clean (format may rewrite; re-stage if so).

- [ ] **Step 5: Commit**

```bash
git add tests/test_svar_mutations.py
git commit -m "test(mutcat): end-to-end thread-invariance for mutation_matrix"
```

- [ ] **Step 6 (optional): Ad-hoc timing sanity check**

On a large real `.svar`, time `annotate_mutations` before/after (compare against the `git` tag prior to Task 5). No committed benchmark; this is a manual confidence check only. Record the speedup in the PR description.

---

## Self-review notes

- **Spec coverage:** SNV (Task 2), DBS (Task 3), ID-83 kernel (Task 4), per-contig vectorized dispatch + REF-mismatch warning + boundary handling (Task 5), `Reference.contig_array` (Task 1), entry-code parallelism (Task 6), count parallelism over samples (Task 7), oracle-based differential + codebook-arithmetic + determinism tests (Tasks 2–8). No public API or code-space change → no SKILL.md edit (matches spec).
- **Oracle:** the current `classify_variants` is preserved verbatim as `_classify_variants_scalar`; the vectorized version is asserted equal to it on random ACGT mixes and at contig boundaries.
- **Type consistency:** the kernel wrapper `_id83_codes_for_contig` takes `(seq, p0, ref_data, ref_s, ref_len, alt_data, alt_s, alt_len)`; both the Task-4 test and the Task-5 caller pass that exact tuple. `_sbs96_codes(seq, p0, ref_b, alt_b)` and `_dbs78_codes(ref_b0, ref_b1, alt_b0, alt_b1)` signatures match their call sites. LUT names `_ID1_CODE/_IDR_CODE/_IDM_CODE` and `_SUB_LUT` are consistent across definition, tests, and callers.

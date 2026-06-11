# SBS-96 / DBS-78 / ID-83 Mutation Catalogues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SigProfiler-style SBS-96 / DBS-78 / ID-83 mutation classification to `SparseVar`, storing a per-entry category field on disk and producing per-sample count matrices as Polars DataFrames.

**Architecture:** A vendored `pysam`-backed `Reference` reader supplies flanking bases. A standalone `_mutcat` module holds the COSMIC codebooks, per-variant classifiers (numpy), and numba kernels for per-entry DBS adjacency + per-sample counting. `SparseVar` gains `annotate_mutations` (writes a `mutcat` int16 field, mirroring `annotate_with_gtf`'s write-back) and `mutation_matrix` (builds the DataFrames).

**Tech Stack:** Python, numpy, numba, polars, pysam, seqpro `Ragged`, pydantic (metadata), pytest + pytest-cases. All dev commands run under Pixi (`pixi run pytest ...`).

**Spec:** `docs/superpowers/specs/2026-06-11-svar-mutation-catalogues-design.md`

---

## Key codebase facts (read before starting)

- `genoray/_svar.py` holds `SparseVar`. Fields are stored as `{name}.npy` mmap arrays sharing the genotype `offsets.npy`, registered in `metadata.json` via `SparseVarMetadata.fields` (`name -> numpy dtype string`). See `_open_fmt` (`_svar.py:2146`), `_write_genos` (`:2161`), `SparseVarMetadata` (`:458`).
- `genos` is a `seqpro.rag.Ragged[V_IDX_TYPE]` with shape `(n_samples, ploidy, None)`. Its `offsets` is a flat `int64` array of length `n_samples*ploidy + 1`; slot `i` covers entries for `sample = i // ploidy`, haplotype `i % ploidy`. `genos.data` is the flat `int32` array of variant indices. A field aligned to `genos` has one value per `genos.data` element (same offsets).
- The in-memory `index` is a `pl.DataFrame` with columns `index, CHROM, POS, REF, ALT (List[Utf8]), ILEN`. `POS` is 0-based `int32`. `SparseVar` is biallelic-per-row in practice (`_is_biallelic`); v1 classifies `ALT.list.first()`.
- `ContigNormalizer` (`genoray/_utils.py`) maps `chr`-prefixed ↔ unprefixed contigs.
- `_index_path(root)` → `root / "index.arrow"`. Metadata written via `SparseVarMetadata(...).model_dump_json()`.
- Existing numba kernels (`_nb_af_helper` at `_svar.py:1699`) iterate `for i in range(len(offsets)-1): o_s, o_e = offsets[i], offsets[i+1]` — follow this slot pattern.
- Tests live in `tests/`, use `pytest`/`pytest_cases`, and import private helpers directly (see `tests/test_gtf_annotation.py`). Run with `pixi run pytest tests/<file> -v`.
- Commit convention: Conventional Commits (`feat:`, `test:`, `docs:`, `chore:`).

---

## File structure

- **Create** `genoray/_reference.py` — `Reference` class (pysam-backed, per-contig cache) + `from_path`/`fetch`.
- **Create** `genoray/_mutcat.py` — codebooks, sentinels, per-variant classifiers, per-entry numba kernels, count-matrix builder.
- **Modify** `genoray/_svar.py` — add `SparseVar.annotate_mutations` and `SparseVar.mutation_matrix`; extend `SparseVarMetadata` with `mutcat_version`.
- **Modify** `genoray/__init__.py` — export `Reference`.
- **Modify** `pyproject.toml` — add `pysam` dependency.
- **Modify** `skills/genoray-api/SKILL.md` — document new public surface.
- **Create** `tests/test_reference.py`, `tests/test_mutcat.py`, `tests/test_svar_mutations.py`.

---

## Task 1: Add pysam dependency and vendored `Reference`

**Files:**
- Modify: `pyproject.toml:9-31` (dependencies list)
- Create: `genoray/_reference.py`
- Test: `tests/test_reference.py`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, inside the `dependencies = [ ... ]` array (after line 15's `"cyvcf2>=0.31.1",`), add:

```toml
    "pysam>=0.22",
```

- [ ] **Step 2: Install it**

Run: `pixi add pysam`
Expected: `pixi.toml`/`pixi.lock` updated, pysam available. (If `pixi add` chooses a different version pin, that's fine — keep the `pyproject.toml` line too.)

- [ ] **Step 3: Write the failing test**

Create `tests/test_reference.py`:

```python
from __future__ import annotations

import numpy as np
import pysam
import pytest

from genoray._reference import Reference


@pytest.fixture
def tiny_fasta(tmp_path):
    # contig "chr1": positions 0..9 = ACGTACGTAC
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n>chr2\nTTTTGGGG\n")
    pysam.faidx(str(fa))  # writes ref.fa.fai
    return fa


def test_fetch_single_window(tiny_fasta):
    ref = Reference.from_path(tiny_fasta)
    # 0-based, half-open [2, 5) -> "GTA"
    seq = ref.fetch("chr1", 2, 5)
    assert bytes(seq) == b"GTA"


def test_fetch_handles_chr_prefix_mismatch(tiny_fasta):
    ref = Reference.from_path(tiny_fasta)
    # query without "chr" prefix must still resolve via ContigNormalizer
    seq = ref.fetch("1", 0, 4)
    assert bytes(seq) == b"ACGT"


def test_fetch_out_of_bounds_pads_with_N(tiny_fasta):
    ref = Reference.from_path(tiny_fasta)
    # left of contig start -> padded with N
    seq = ref.fetch("chr1", -2, 3)
    assert bytes(seq) == b"NNACG"
```

- [ ] **Step 4: Run it to verify failure**

Run: `pixi run pytest tests/test_reference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'genoray._reference'`.

- [ ] **Step 5: Implement `Reference`**

Create `genoray/_reference.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pysam
from numpy.typing import NDArray

from ._utils import ContigNormalizer

_PAD = ord("N")


class Reference:
    """A reference genome backed by an indexed FASTA, read on demand via pysam.

    One contig is held in memory at a time and sliced for flanking-base lookups.
    Queries accept ``chr``-prefixed or unprefixed contig names interchangeably.

    Do not instantiate directly; use :meth:`Reference.from_path`.
    """

    def __init__(self, path: Path, contigs: list[str]):
        self.path = path
        self._fasta = pysam.FastaFile(str(path))
        # pysam reports the FASTA's own contig names; build a normalizer from them.
        self._c_norm = ContigNormalizer(list(self._fasta.references))
        self.contigs = contigs
        self._cur_contig: str | None = None
        self._cur_seq: NDArray[np.uint8] | None = None

    @classmethod
    def from_path(
        cls, fasta: str | Path, contigs: list[str] | None = None
    ) -> "Reference":
        path = Path(fasta)
        if not path.exists():
            raise FileNotFoundError(f"FASTA {path} does not exist.")
        fai = path.with_suffix(path.suffix + ".fai")
        if not fai.exists():
            pysam.faidx(str(path))
        f = pysam.FastaFile(str(path))
        all_contigs = list(f.references)
        f.close()
        return cls(path, contigs if contigs is not None else all_contigs)

    def _load_contig(self, contig: str) -> NDArray[np.uint8]:
        norm = self._c_norm.norm(contig)
        if norm is None:
            raise ValueError(f"Contig {contig!r} not found in reference {self.path}.")
        if norm != self._cur_contig:
            seq = self._fasta.fetch(norm)  # whole contig as str
            self._cur_seq = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            self._cur_contig = norm
        assert self._cur_seq is not None
        return self._cur_seq

    def fetch(self, contig: str, start: int, end: int) -> NDArray[np.uint8]:
        """Return reference bytes for 0-based half-open ``[start, end)``.

        Positions outside the contig are padded with ``N``. Returns a uint8
        array; ``bytes(...)`` gives the ASCII sequence.
        """
        seq = self._load_contig(contig)
        n = len(seq)
        out = np.full(end - start, _PAD, dtype=np.uint8)
        src_s = max(start, 0)
        src_e = min(end, n)
        if src_e > src_s:
            out[src_s - start : src_e - start] = seq[src_s:src_e]
        return out
```

Verify `ContigNormalizer.norm` accepts a single string and returns the matching FASTA contig name or `None`; if its API differs (e.g. expects a list), adapt the call (`self._c_norm.norm([contig])[0]`).

- [ ] **Step 6: Run tests to verify pass**

Run: `pixi run pytest tests/test_reference.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml pixi.toml pixi.lock genoray/_reference.py tests/test_reference.py
git commit -m "feat(reference): vendor pysam-backed Reference reader"
```

---

## Task 2: Mutation-category codebooks and sentinels

**Files:**
- Create: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mutcat.py`:

```python
from __future__ import annotations

from genoray._mutcat import (
    DBS78,
    ID83,
    SBS96,
    SENTINELS,
    code_ranges,
)


def test_codebook_sizes():
    assert len(SBS96) == 96
    assert len(DBS78) == 78
    assert len(ID83) == 83


def test_sbs96_labels_well_formed():
    assert SBS96[0] == "A[C>A]A"
    assert SBS96[-1] == "T[T>G]T"
    assert all(lbl[1:2] == "[" or "[" in lbl for lbl in SBS96)


def test_dbs78_known_members():
    assert "AC>CA" in DBS78
    assert "TT>GG" in DBS78


def test_id83_known_members():
    assert ID83[0] == "1:Del:C:0"
    assert "5:Del:M:5" in ID83
    assert ID83[-1] == "5:Del:M:5"


def test_code_ranges_are_contiguous_and_disjoint():
    r = code_ranges()
    assert r["SBS96"] == (0, 96)
    assert r["DBS78"] == (96, 174)
    assert r["ID83"] == (174, 257)


def test_sentinels_outside_category_ranges():
    # sentinels must not collide with 0..256
    assert all(v < 0 for v in SENTINELS.values())
    assert SENTINELS["DBS_PARTNER"] != SENTINELS["UNCLASSIFIED"]
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_mutcat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'genoray._mutcat'`.

- [ ] **Step 3: Implement codebooks**

Create `genoray/_mutcat.py`:

```python
from __future__ import annotations

import numpy as np

# ---- SBS-96 (COSMIC order: substitution outer, 5' base, 3' base inner) ----
_SBS_SUBS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
_BASES = ["A", "C", "G", "T"]
SBS96: list[str] = [
    f"{five}[{sub}]{three}"
    for sub in _SBS_SUBS
    for five in _BASES
    for three in _BASES
]

# ---- DBS-78 (canonical COSMIC list) ----
DBS78: list[str] = [
    "AC>CA", "AC>CG", "AC>CT", "AC>GA", "AC>GG", "AC>GT", "AC>TA", "AC>TG", "AC>TT",
    "AT>CA", "AT>CC", "AT>CG", "AT>GA", "AT>GC", "AT>TA",
    "CC>AA", "CC>AG", "CC>AT", "CC>GA", "CC>GG", "CC>GT", "CC>TA", "CC>TG", "CC>TT",
    "CG>AT", "CG>GC", "CG>GT", "CG>TA", "CG>TC", "CG>TT",
    "CT>AA", "CT>AC", "CT>AG", "CT>GA", "CT>GC", "CT>GG", "CT>TA", "CT>TC", "CT>TG",
    "GC>AA", "GC>AG", "GC>AT", "GC>CA", "GC>CG", "GC>TA",
    "TA>AT", "TA>CG", "TA>CT", "TA>GC", "TA>GG", "TA>GT",
    "TC>AA", "TC>AG", "TC>AT", "TC>CA", "TC>CG", "TC>CT", "TC>GA", "TC>GG", "TC>GT",
    "TG>AA", "TG>AC", "TG>AT", "TG>CA", "TG>CC", "TG>CT", "TG>GA", "TG>GC", "TG>GT",
    "TT>AA", "TT>AC", "TT>AG", "TT>CA", "TT>CC", "TT>CG", "TT>GA", "TT>GC", "TT>GG",
]


def _build_id83() -> list[str]:
    out: list[str] = []
    # 1bp del/ins, by base C/T, repeat count 0..5(+)
    for kind in ("Del", "Ins"):
        for base in ("C", "T"):
            for r in range(6):
                out.append(f"1:{kind}:{base}:{r}")
    # >1bp del/ins at repeats, size 2..5(+), repeat 0..5(+)
    for kind in ("Del", "Ins"):
        for size in ("2", "3", "4", "5"):
            for r in range(6):
                out.append(f"{size}:{kind}:R:{r}")
    # microhomology deletions
    for size, mh_max in (("2", 1), ("3", 2), ("4", 3), ("5", 5)):
        for m in range(1, mh_max + 1):
            out.append(f"{size}:Del:M:{m}")
    return out


ID83: list[str] = _build_id83()

assert len(SBS96) == 96 and len(DBS78) == 78 and len(ID83) == 83

# ---- unified int16 code space ----
SBS96_OFFSET = 0
DBS78_OFFSET = SBS96_OFFSET + len(SBS96)   # 96
ID83_OFFSET = DBS78_OFFSET + len(DBS78)    # 174
N_CODES = ID83_OFFSET + len(ID83)          # 257

SENTINELS: dict[str, int] = {
    "DBS_PARTNER": -1,   # 3' half of an adjacency doublet; never counted
    "UNCLASSIFIED": -2,  # symbolic/complex/MNV>2bp/non-ACGT
    "MISSING": -3,
}

# index -> label, for building DataFrames
SBS96_INDEX = {lbl: SBS96_OFFSET + i for i, lbl in enumerate(SBS96)}
DBS78_INDEX = {lbl: DBS78_OFFSET + i for i, lbl in enumerate(DBS78)}
ID83_INDEX = {lbl: ID83_OFFSET + i for i, lbl in enumerate(ID83)}

MUTCAT_VERSION = 1


def code_ranges() -> dict[str, tuple[int, int]]:
    """Half-open ``[start, end)`` code range per matrix kind."""
    return {
        "SBS96": (SBS96_OFFSET, DBS78_OFFSET),
        "DBS78": (DBS78_OFFSET, ID83_OFFSET),
        "ID83": (ID83_OFFSET, N_CODES),
    }


def labels(kind: str) -> list[str]:
    return {"SBS96": SBS96, "DBS78": DBS78, "ID83": ID83}[kind]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run pytest tests/test_mutcat.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): add SBS-96/DBS-78/ID-83 codebooks and code space"
```

---

## Task 3: SBS-96 per-variant classifier

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mutcat.py`:

```python
import numpy as np
from genoray._mutcat import SBS96_INDEX, classify_sbs96


def test_sbs96_pyrimidine_direct():
    # context A[C>A]G : ref=C (pyrimidine), keep as-is
    code = classify_sbs96(five=b"A", ref=b"C", alt=b"A", three=b"G")
    assert code == SBS96_INDEX["A[C>A]G"]


def test_sbs96_purine_folds_to_revcomp():
    # ref=G>T in context T_A : revcomp -> A[C>A]A (G>T == C>A on opposite strand,
    # flanks swap+complement: 5'=T,3'=A -> 5'=T(comp of A on 3') ... compute:
    # original T[G>T]A ; revcomp substitution G>T -> C>A ; flanks: comp(A)=T (new 5'),
    # comp(T)=A (new 3') -> T[C>A]A
    code = classify_sbs96(five=b"T", ref=b"G", alt=b"T", three=b"A")
    assert code == SBS96_INDEX["T[C>A]A"]
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_mutcat.py::test_sbs96_pyrimidine_direct -v`
Expected: FAIL — `ImportError: cannot import name 'classify_sbs96'`.

- [ ] **Step 3: Implement `classify_sbs96`**

Append to `genoray/_mutcat.py`:

```python
_COMP = {ord("A"): ord("T"), ord("T"): ord("A"), ord("C"): ord("G"), ord("G"): ord("C")}
_PYR = {ord("C"), ord("T")}


def _comp(b: int) -> int:
    return _COMP.get(b, b)


def classify_sbs96(five: bytes, ref: bytes, alt: bytes, three: bytes) -> int:
    """Return the SBS-96 code for one SNV, or SENTINELS['UNCLASSIFIED']."""
    f, r, a, t = five[0], ref[0], alt[0], three[0]
    if r not in _COMP or a not in _COMP or r == a:
        return SENTINELS["UNCLASSIFIED"]
    if f not in _COMP or t not in _COMP:
        return SENTINELS["UNCLASSIFIED"]
    if r not in _PYR:  # purine ref -> fold to reverse complement
        r, a = _comp(r), _comp(a)
        f, t = _comp(t), _comp(f)  # flanks swap and complement
    label = f"{chr(f)}[{chr(r)}>{chr(a)}]{chr(t)}"
    return SBS96_INDEX[label]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run pytest tests/test_mutcat.py -k sbs96 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): add SBS-96 single-variant classifier"
```

---

## Task 4: DBS-78 doublet classifier

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mutcat.py`:

```python
from genoray._mutcat import DBS78_INDEX, classify_dbs78


def test_dbs78_direct_member():
    # AC>CA is directly in the catalogue
    assert classify_dbs78(b"AC", b"CA") == DBS78_INDEX["AC>CA"]


def test_dbs78_folds_to_revcomp_member():
    # GT>TG : revcomp of ref GT is AC, revcomp of alt TG is CA -> AC>CA
    assert classify_dbs78(b"GT", b"TG") == DBS78_INDEX["AC>CA"]


def test_dbs78_non_doublet_unclassified():
    from genoray._mutcat import SENTINELS
    assert classify_dbs78(b"AC", b"AC") == SENTINELS["UNCLASSIFIED"]  # no change
    assert classify_dbs78(b"ACG", b"TTT") == SENTINELS["UNCLASSIFIED"]  # >2bp
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_mutcat.py -k dbs78 -v`
Expected: FAIL — `ImportError: cannot import name 'classify_dbs78'`.

- [ ] **Step 3: Implement `classify_dbs78`**

Append to `genoray/_mutcat.py`:

```python
def _revcomp(seq: bytes) -> bytes:
    return bytes(_comp(b) for b in reversed(seq))


def classify_dbs78(ref: bytes, alt: bytes) -> int:
    """Return the DBS-78 code for a 2bp doublet, or SENTINELS['UNCLASSIFIED'].

    Tries the literal ``REF>ALT`` first, then its reverse-complement, since
    DBS-78 collapses strand-equivalent doublets.
    """
    if len(ref) != 2 or len(alt) != 2 or ref == alt:
        return SENTINELS["UNCLASSIFIED"]
    if any(b not in _COMP for b in ref) or any(b not in _COMP for b in alt):
        return SENTINELS["UNCLASSIFIED"]
    key = f"{ref.decode()}>{alt.decode()}"
    if key in DBS78_INDEX:
        return DBS78_INDEX[key]
    rc_key = f"{_revcomp(ref).decode()}>{_revcomp(alt).decode()}"
    if rc_key in DBS78_INDEX:
        return DBS78_INDEX[rc_key]
    return SENTINELS["UNCLASSIFIED"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run pytest tests/test_mutcat.py -k dbs78 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): add DBS-78 doublet classifier"
```

---

## Task 5: ID-83 indel classifier

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

ID-83 needs the reference sequence immediately 3′ of the indel. We follow the PCAWG/SigProfiler rules. Convention: `SparseVar` indels are left-aligned and biallelic with the shared anchor base in `REF[0]`/`ALT[0]` (standard VCF). The inserted/deleted sequence is the part after the anchor. `pos` is the 0-based POS of the anchor base.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mutcat.py`:

```python
from genoray._mutcat import ID83_INDEX, classify_id83


def _ref_fn(seq: bytes):
    """Build a reference-fetch callable over a single contig given as bytes."""
    arr = np.frombuffer(seq, dtype=np.uint8)

    def fetch(start: int, end: int) -> bytes:
        out = np.full(end - start, ord("N"), dtype=np.uint8)
        s, e = max(start, 0), min(end, len(arr))
        if e > s:
            out[s - start : e - start] = arr[s:e]
        return out.tobytes()

    return fetch


def test_id83_1bp_deletion_in_homopolymer():
    # ref: ...A CCCCC G...  delete one C from a run of 5 C's
    # contig: index: 0=A,1..5=CCCCC,6=G ; anchor at pos 0 (A), REF="AC", ALT="A"
    # deleted base C, the run downstream of the deletion has 4 remaining C's -> repeat class
    fetch = _ref_fn(b"ACCCCCG")
    code = classify_id83(pos=0, ref=b"AC", alt=b"A", fetch=fetch)
    # 1bp Del of C with >=5 surrounding repeats caps at index 5
    assert ID83_INDEX["1:Del:C:5"] == code or ID83_INDEX["1:Del:C:4"] == code


def test_id83_1bp_insertion_T():
    # insert a T with no downstream T repeat
    fetch = _ref_fn(b"AGGGG")
    code = classify_id83(pos=0, ref=b"A", alt=b"AT", fetch=fetch)
    assert code == ID83_INDEX["1:Ins:T:0"]


def test_id83_non_indel_unclassified():
    from genoray._mutcat import SENTINELS
    fetch = _ref_fn(b"ACGT")
    assert classify_id83(pos=0, ref=b"A", alt=b"C", fetch=fetch) == SENTINELS["UNCLASSIFIED"]
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_mutcat.py -k id83 -v`
Expected: FAIL — `ImportError: cannot import name 'classify_id83'`.

- [ ] **Step 3: Implement `classify_id83`**

Append to `genoray/_mutcat.py`. `fetch(start, end) -> bytes` returns reference bytes for 0-based half-open coordinates (N-padded out of bounds).

```python
from collections.abc import Callable


def _size_bucket(n: int) -> str:
    return "5" if n >= 5 else str(n)


def _repeat_bucket(n: int) -> int:
    return 5 if n >= 5 else n


def classify_id83(
    pos: int, ref: bytes, alt: bytes, fetch: Callable[[int, int], bytes]
) -> int:
    """Classify a single indel into one of the 83 ID channels.

    ``pos`` is the 0-based anchor (REF[0]) position. REF/ALT must share the
    anchor base (standard left-aligned VCF representation).
    """
    if len(ref) == len(alt):  # SNV or MNV, not an indel
        return SENTINELS["UNCLASSIFIED"]
    if len(ref) >= 1 and len(alt) >= 1 and ref[0] != alt[0]:
        return SENTINELS["UNCLASSIFIED"]  # not anchored; complex
    is_del = len(ref) > len(alt)
    indel = ref[1:] if is_del else alt[1:]
    ilen = len(indel)
    if any(b not in _COMP for b in indel):
        return SENTINELS["UNCLASSIFIED"]

    # downstream sequence begins just after the anchor (deletions) / after pos (ins)
    # The first changed base sits at pos+1.
    scan_start = pos + 1
    window = fetch(scan_start, scan_start + ilen * 6 + ilen)  # enough for >=5 repeats

    # count tandem repeats of `indel` immediately downstream
    n_rep = 0
    i = 0
    if is_del:
        # for deletions the deleted unit itself is the first copy in the reference
        # count copies including the deleted one
        while window[i : i + ilen] == indel and i + ilen <= len(window):
            n_rep += 1
            i += ilen
    else:
        while window[i : i + ilen] == indel and i + ilen <= len(window):
            n_rep += 1
            i += ilen

    kind = "Del" if is_del else "Ins"

    if ilen == 1:
        base = indel.decode()
        # fold purine to pyrimidine for the 1bp channel
        if base in ("A", "G"):
            base = chr(_comp(ord(base)))
        rep = _repeat_bucket(n_rep - 1 if is_del else n_rep)
        return ID83_INDEX[f"1:{kind}:{base}:{rep}"]

    # >=2bp: repeat channel unless a microhomology deletion applies
    size = _size_bucket(ilen)
    if is_del:
        mh = _microhomology_len(indel, window, ilen)
        if mh > 0 and n_rep <= 1:
            mh_cap = {2: 1, 3: 2, 4: 3}.get(ilen, 5)
            return ID83_INDEX[f"{size}:Del:M:{min(mh, mh_cap)}"]
    rep = _repeat_bucket(n_rep - 1 if is_del else n_rep)
    return ID83_INDEX[f"{size}:{kind}:R:{rep}"]


def _microhomology_len(indel: bytes, downstream: bytes, ilen: int) -> int:
    """Length of partial-match microhomology between the deleted unit and the
    sequence flanking the deletion (downstream side), capped at ilen-1."""
    mh = 0
    for k in range(1, ilen):
        if downstream[:k] == indel[:k]:
            mh = max(mh, k)
    return mh
```

> NOTE FOR IMPLEMENTER: ID-83 boundary semantics (whether the deleted copy counts toward `n_rep`, microhomology on both flanks) are subtle. The tests above pin the common cases; if cross-checking against SigProfilerMatrixGenerator (Task 10 fixture) reveals off-by-one repeat-bucket differences, adjust `n_rep` accounting and `_microhomology_len` to match SigProfiler's published ID-83 rules. Do not add new channels — only correct the bucketing.

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run pytest tests/test_mutcat.py -k id83 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): add ID-83 indel classifier"
```

---

## Task 6: Per-variant classification dispatcher

Builds the intrinsic `var_code` array (one int16 per variant): SNV→SBS96, native 2bp MNV→DBS78, indel→ID83, else UNCLASSIFIED. Pulls reference context via a `Reference`.

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mutcat.py`:

```python
import polars as pl
from genoray._mutcat import classify_variants
from genoray._reference import Reference


def test_classify_variants_mixed(tmp_path):
    import pysam
    fa = tmp_path / "ref.fa"
    # chr1: A C G T A C G T  (0..7)
    fa.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)

    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1"],
            "POS": [1, 2],  # 0-based
            "REF": ["C", "G"],
            "ALT": [["A"], ["GT"]],  # SNV ; insertion
        }
    )
    codes = classify_variants(index, ref)
    assert codes.dtype == np.int16
    assert len(codes) == 2
    # first is an SNV -> within SBS96 range
    assert 0 <= codes[0] < 96
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_mutcat.py::test_classify_variants_mixed -v`
Expected: FAIL — `ImportError: cannot import name 'classify_variants'`.

- [ ] **Step 3: Implement `classify_variants`**

Append to `genoray/_mutcat.py`:

```python
import polars as pl

from ._reference import Reference


def classify_variants(index: pl.DataFrame, reference: Reference) -> np.ndarray:
    """Return an int16 array of intrinsic mutation codes, one per row of ``index``.

    ``index`` must have columns CHROM, POS (0-based int), REF (str), ALT
    (List[str]; first ALT used). Reference context is fetched per contig.
    """
    chrom = index["CHROM"].to_numpy()
    pos = index["POS"].to_numpy().astype(np.int64)
    ref = index["REF"].to_list()
    alt0 = index["ALT"].list.first().to_list()

    out = np.full(index.height, SENTINELS["UNCLASSIFIED"], dtype=np.int16)

    for i in range(index.height):
        r = ref[i]
        a = alt0[i]
        if a is None or r is None:
            continue
        rb, ab = r.encode(), a.encode()
        c = chrom[i]
        p = int(pos[i])

        if len(rb) == 1 and len(ab) == 1:  # SNV
            five = reference.fetch(c, p - 1, p).tobytes()
            three = reference.fetch(c, p + 1, p + 2).tobytes()
            out[i] = classify_sbs96(five, rb, ab, three)
        elif len(rb) == 2 and len(ab) == 2:  # native MNV doublet
            out[i] = classify_dbs78(rb, ab)
        elif len(rb) != len(ab):  # indel
            def fetch(s: int, e: int, _c=c) -> bytes:
                return reference.fetch(_c, s, e).tobytes()

            out[i] = classify_id83(p, rb, ab, fetch)
        # else: MNV>2bp or symbolic -> stays UNCLASSIFIED
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run pytest tests/test_mutcat.py::test_classify_variants_mixed -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): add per-variant classification dispatcher"
```

---

## Task 7: Per-entry broadcast + DBS adjacency override (numba)

Broadcast `var_code` to per-genotype-entry codes, then within each `(sample, haplotype)` track convert isolated adjacent-SNV pairs into a DBS (5′ entry → DBS code, 3′ entry → DBS_PARTNER).

**Files:**
- Modify: `genoray/_mutcat.py`
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mutcat.py`:

```python
from genoray._mutcat import (
    DBS78_OFFSET,
    SENTINELS,
    build_entry_codes,
)


def test_build_entry_codes_marks_adjacent_dbs():
    # 3 variants. var 0 and 1 are SNVs at adjacent positions p, p+1.
    # var 2 is an isolated SNV.
    var_code = np.array([10, 11, 12], dtype=np.int16)  # arbitrary SBS codes
    var_pos = np.array([100, 101, 200], dtype=np.int64)
    var_contig = np.array([0, 0, 0], dtype=np.int32)
    var_is_snv = np.array([True, True, True])
    var_ref_b = np.frombuffer(b"ACG", np.uint8).copy()   # ref base per variant
    var_alt_b = np.frombuffer(b"GTA", np.uint8).copy()   # alt base per variant

    # one sample, ploidy 1, track has all three variant indices
    data = np.array([0, 1, 2], dtype=np.int32)
    offsets = np.array([0, 3], dtype=np.int64)

    codes = build_entry_codes(
        data, offsets, var_code, var_pos, var_contig,
        var_is_snv, var_ref_b, var_alt_b,
    )
    # entry for var0 -> a DBS code (>=DBS78_OFFSET), var1 -> DBS_PARTNER, var2 unchanged
    assert codes[0] >= DBS78_OFFSET and codes[0] < DBS78_OFFSET + 78
    assert codes[1] == SENTINELS["DBS_PARTNER"]
    assert codes[2] == 12


def test_build_entry_codes_no_false_pair_when_not_adjacent():
    var_code = np.array([10, 12], dtype=np.int16)
    var_pos = np.array([100, 105], dtype=np.int64)
    var_contig = np.array([0, 0], dtype=np.int32)
    var_is_snv = np.array([True, True])
    var_ref_b = np.frombuffer(b"AC", np.uint8).copy()
    var_alt_b = np.frombuffer(b"GT", np.uint8).copy()
    data = np.array([0, 1], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int64)
    codes = build_entry_codes(
        data, offsets, var_code, var_pos, var_contig,
        var_is_snv, var_ref_b, var_alt_b,
    )
    assert codes.tolist() == [10, 12]
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_mutcat.py -k build_entry_codes -v`
Expected: FAIL — `ImportError: cannot import name 'build_entry_codes'`.

- [ ] **Step 3: Implement the doublet code table + kernel**

Append to `genoray/_mutcat.py`. We need a numba-callable lookup mapping a 4-base doublet to its DBS code. Precompute a dense table indexed by encoded bases (A=0,C=1,G=2,T=3).

```python
import numba as nb

_BASE2IDX = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate("ACGT"):
    _BASE2IDX[ord(_b)] = _i


def _build_dbs_table() -> np.ndarray:
    """tbl[r0, r1, a0, a1] -> DBS-78 code or DBS_PARTNER-sentinel(-1) sentinel.

    Uses UNCLASSIFIED for doublets not in the (folded) catalogue.
    """
    tbl = np.full((4, 4, 4, 4), SENTINELS["UNCLASSIFIED"], dtype=np.int16)
    bases = b"ACGT"
    for r0 in range(4):
        for r1 in range(4):
            for a0 in range(4):
                for a1 in range(4):
                    ref = bytes([bases[r0], bases[r1]])
                    alt = bytes([bases[a0], bases[a1]])
                    tbl[r0, r1, a0, a1] = classify_dbs78(ref, alt)
    return tbl


_DBS_TABLE = _build_dbs_table()
_DBS_PARTNER = SENTINELS["DBS_PARTNER"]
_BASE2IDX_CONST = _BASE2IDX


@nb.njit(cache=True)
def _entry_codes_kernel(
    data, offsets, var_code, var_pos, var_contig, var_is_snv,
    ref_idx, alt_idx, dbs_table, base_missing, out, dbs_partner,
):
    for slot in range(len(offsets) - 1):
        o_s, o_e = offsets[slot], offsets[slot + 1]
        j = o_s
        while j < o_e:
            v = data[j]
            out[j] = var_code[v]
            # try to pair with the next entry in this track
            if (
                j + 1 < o_e
                and var_is_snv[v]
            ):
                w = data[j + 1]
                if (
                    var_is_snv[w]
                    and var_contig[v] == var_contig[w]
                    and var_pos[w] - var_pos[v] == 1
                ):
                    # isolated pair only: the entry after w must not also be at w+1
                    isolated = True
                    if j + 2 < o_e:
                        x = data[j + 2]
                        if (
                            var_is_snv[x]
                            and var_contig[w] == var_contig[x]
                            and var_pos[x] - var_pos[w] == 1
                        ):
                            isolated = False
                    if isolated:
                        ri0 = ref_idx[v]
                        ri1 = ref_idx[w]
                        ai0 = alt_idx[v]
                        ai1 = alt_idx[w]
                        if (
                            ri0 >= 0 and ri1 >= 0 and ai0 >= 0 and ai1 >= 0
                        ):
                            code = dbs_table[ri0, ri1, ai0, ai1]
                            out[j] = code
                            out[j + 1] = dbs_partner
                            j += 2
                            continue
            j += 1


def build_entry_codes(
    data, offsets, var_code, var_pos, var_contig, var_is_snv, var_ref_b, var_alt_b,
):
    """Return int16 per-entry codes aligned to ``data`` (genos.data)."""
    ref_idx = _BASE2IDX_CONST[var_ref_b.astype(np.int64)].astype(np.int64)
    alt_idx = _BASE2IDX_CONST[var_alt_b.astype(np.int64)].astype(np.int64)
    out = np.empty(len(data), dtype=np.int16)
    _entry_codes_kernel(
        data.astype(np.int32), offsets.astype(np.int64), var_code,
        var_pos.astype(np.int64), var_contig.astype(np.int32),
        var_is_snv.astype(np.bool_), ref_idx, alt_idx,
        _DBS_TABLE, np.int64(-1), out, np.int16(_DBS_PARTNER),
    )
    return out
```

> NOTE: runs of ≥3 adjacent SNVs are intentionally left as individual SBS (the `isolated` guard). This matches the documented v1 scope.

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run pytest tests/test_mutcat.py -k build_entry_codes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): add per-entry codes with DBS adjacency override"
```

---

## Task 8: `SparseVar.annotate_mutations`

Wires Tasks 6+7 into `SparseVar`, persisting `mutcat.npy` and metadata.

**Files:**
- Modify: `genoray/_svar.py` (add method; extend `SparseVarMetadata`; import helpers)
- Test: `tests/test_svar_mutations.py`

- [ ] **Step 1: Extend metadata**

In `genoray/_svar.py`, modify `SparseVarMetadata` (`:458`) to add an optional version field:

```python
class SparseVarMetadata(BaseModel):
    version: int | None = None
    samples: list[str]
    ploidy: int
    contigs: list[str]
    fields: dict[str, str] = {}  # field_name -> numpy dtype name (e.g. "float32")
    mutcat_version: int | None = None  # set when annotate_mutations has run
```

Every existing `SparseVarMetadata(...)` construction site (`:1024`, `:1215`, `:1688`) keeps working because the new field has a default. When writing metadata after annotation, include `mutcat_version`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_svar_mutations.py`:

```python
from __future__ import annotations

import numpy as np
import polars as pl
import pysam
import pytest

import genoray
from genoray import SparseVar
from genoray._reference import Reference


@pytest.fixture
def annotated_svar(tmp_path):
    """Build a tiny SVAR by hand + a matching reference, then annotate it."""
    # Reference chr1: A C G T A C G T A C  (0..9)
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n")
    pysam.faidx(str(fa))
    svar_dir = tmp_path / "tiny.svar"
    _build_tiny_svar(svar_dir)
    svar = SparseVar(svar_dir)
    svar.annotate_mutations(Reference.from_path(fa), write_back=True)
    return svar_dir


def _build_tiny_svar(path):
    """Write a minimal valid SVAR directory with 2 samples, ploidy 1, 3 SNVs."""
    from genoray._svar import SparseVarMetadata, _write_genos
    from seqpro.rag import Ragged

    path.mkdir(parents=True)
    # 3 variants on chr1 at POS 1,2,8 (0-based); all SNVs
    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1"],
            "POS": np.array([1, 2, 8], dtype=np.int32),
            "REF": ["C", "G", "A"],
            "ALT": [["A"], ["T"], ["C"]],
            "ILEN": np.array([0, 0, 0], dtype=np.int32),
        }
    )
    index.write_ipc(path / "index.arrow")

    # sample 0 carries variants 0 and 1 (adjacent -> DBS); sample 1 carries variant 2
    data = np.array([0, 1, 2], dtype=np.int32)
    offsets = np.array([0, 2, 3], dtype=np.int64)  # (n_samples*ploidy + 1) = 3
    genos = Ragged.from_offsets(data, (2, 1, None), offsets)
    _write_genos(path, genos)

    with open(path / "metadata.json", "w") as f:
        f.write(
            SparseVarMetadata(
                version=1, samples=["s0", "s1"], ploidy=1, contigs=["chr1"]
            ).model_dump_json()
        )


def test_annotate_writes_mutcat_field(annotated_svar):
    assert (annotated_svar / "mutcat.npy").exists()
    # re-open and confirm metadata records it
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    assert "mutcat" in svar.available_fields
    assert svar.available_fields["mutcat"] == np.dtype("int16")


def test_annotate_dbs_partner_present(annotated_svar):
    from genoray._mutcat import SENTINELS

    mut = np.memmap(annotated_svar / "mutcat.npy", dtype=np.int16, mode="r")
    # sample 0's two adjacent SNVs -> [DBS code, DBS_PARTNER]
    assert mut[1] == SENTINELS["DBS_PARTNER"]
```

- [ ] **Step 3: Run it to verify failure**

Run: `pixi run pytest tests/test_svar_mutations.py -v`
Expected: FAIL — `AttributeError: 'SparseVar' object has no attribute 'annotate_mutations'`.

- [ ] **Step 4: Implement `annotate_mutations`**

In `genoray/_svar.py`, add imports near the top (after line 38):

```python
from ._mutcat import (
    MUTCAT_VERSION,
    build_entry_codes,
    classify_variants,
)
from ._reference import Reference
```

Add this method to `SparseVar` (place it next to `annotate_with_gtf`, around `:1454`):

```python
    def annotate_mutations(
        self,
        reference: "Reference | str | Path",
        *,
        write_back: bool = True,
    ) -> None:
        """Classify every variant into SBS-96 / DBS-78 / ID-83 channels and store
        a per-genotype-entry ``mutcat`` field (int16, enum-encoded).

        Adjacent SNVs carried on the same haplotype are combined into DBS; the
        5' entry receives the DBS code and the 3' entry a ``DBS_PARTNER`` sentinel.

        Parameters
        ----------
        reference
            A :class:`genoray.Reference` or a path to an indexed FASTA used to
            fetch flanking bases.
        write_back
            If True (default), write ``mutcat.npy`` and update ``metadata.json``.
            If False, keep the field in memory only (``self.fields['mutcat']``).
        """
        if not isinstance(reference, Reference):
            reference = Reference.from_path(reference)

        # 1. intrinsic per-variant codes
        var_code = classify_variants(self.index, reference)

        # 2. per-variant arrays needed by the adjacency kernel
        pos = self.index["POS"].to_numpy().astype(np.int64)
        ref0 = self.index["REF"].to_list()
        alt0 = self.index["ALT"].list.first().to_list()
        is_snv = np.array(
            [
                r is not None and a is not None and len(r) == 1 and len(a) == 1
                for r, a in zip(ref0, alt0)
            ]
        )
        # contig id per variant (variants are contig-contiguous in the index)
        contig_codes = (
            self.index["CHROM"].cast(pl.Categorical).to_physical().to_numpy()
        ).astype(np.int32)
        ref_b = np.array(
            [ord(r[0]) if r else 0 for r in ref0], dtype=np.uint8
        )
        alt_b = np.array(
            [ord(a[0]) if a else 0 for a in alt0], dtype=np.uint8
        )

        # 3. broadcast to entries + DBS adjacency override
        entry_codes = build_entry_codes(
            self.genos.data,
            self.genos.offsets,
            var_code,
            pos,
            contig_codes,
            is_snv,
            ref_b,
            alt_b,
        )

        # 4. register + optionally persist
        from seqpro.rag import Ragged

        self.available_fields["mutcat"] = np.dtype("int16")
        self.fields["mutcat"] = Ragged.from_offsets(
            entry_codes, self.genos.shape, self.genos.offsets
        )

        if write_back:
            mm = np.memmap(
                self.path / "mutcat.npy",
                dtype=np.int16,
                mode="w+",
                shape=entry_codes.shape,
            )
            mm[:] = entry_codes
            mm.flush()

            with open(self.path / "metadata.json", "rb") as f:
                meta = SparseVarMetadata.model_validate_json(f.read())
            meta.fields["mutcat"] = "int16"
            meta.mutcat_version = MUTCAT_VERSION
            with open(self.path / "metadata.json", "w") as f:
                f.write(meta.model_dump_json())
```

If `CHROM.cast(pl.Categorical).to_physical()` does not preserve first-appearance contig order across the whole frame, replace it with a manual map built from `self.contigs` (`pl.col("CHROM").replace_strict({c: i for i, c in enumerate(self.contigs)})`).

- [ ] **Step 5: Run tests to verify pass**

Run: `pixi run pytest tests/test_svar_mutations.py -v`
Expected: PASS (the two tests defined so far).

- [ ] **Step 6: Commit**

```bash
git add genoray/_svar.py tests/test_svar_mutations.py
git commit -m "feat(svar): add annotate_mutations writing per-entry mutcat field"
```

---

## Task 9: `SparseVar.mutation_matrix` + counting kernel

**Files:**
- Modify: `genoray/_mutcat.py` (counting kernel + DataFrame builder)
- Modify: `genoray/_svar.py` (`mutation_matrix` method)
- Test: `tests/test_svar_mutations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar_mutations.py`:

```python
def test_mutation_matrix_shapes_and_samples(annotated_svar):
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    sbs = svar.mutation_matrix("SBS96", count="allele")
    assert sbs.columns[0] == "MutationType"
    assert sbs.height == 96
    assert set(svar.available_samples).issubset(set(sbs.columns))

    dbs = svar.mutation_matrix("DBS78")
    assert dbs.height == 78
    # sample s0 has exactly one DBS event
    assert dbs["s0"].sum() == 1


def test_mutation_matrix_requires_annotation(tmp_path):
    # build an un-annotated svar
    from tests.test_svar_mutations import _build_tiny_svar  # reuse builder
    d = tmp_path / "x.svar"
    _build_tiny_svar(d)
    svar = SparseVar(d)
    with pytest.raises(ValueError, match="mutcat"):
        svar.mutation_matrix("SBS96")


def test_count_unit_allele_vs_sample(annotated_svar):
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    a = svar.mutation_matrix("SBS96", count="allele")
    s = svar.mutation_matrix("SBS96", count="sample")
    # totals: allele >= sample (hom collapses to 1 under "sample")
    assert a.select(svar.available_samples).sum().sum_horizontal().item() >= \
           s.select(svar.available_samples).sum().sum_horizontal().item()
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run pytest tests/test_svar_mutations.py -k mutation_matrix -v`
Expected: FAIL — `AttributeError: ... 'mutation_matrix'`.

- [ ] **Step 3: Implement counting kernel + builder in `_mutcat.py`**

Append to `genoray/_mutcat.py`:

```python
@nb.njit(cache=True)
def _count_kernel(data_codes, offsets, ploidy, n_samples, n_codes, per_sample, out):
    """out[sample, code] accumulator over genotype entries.

    ``data_codes`` is the per-entry int16 code array (aligned to genos.data).
    When ``per_sample`` is True, a code is counted at most once per sample.
    """
    for slot in range(len(offsets) - 1):
        sample = slot // ploidy
        o_s, o_e = offsets[slot], offsets[slot + 1]
        for j in range(o_s, o_e):
            code = data_codes[j]
            if code < 0 or code >= n_codes:
                continue
            if per_sample:
                if out[sample, code] == 0:
                    out[sample, code] = 1
            else:
                out[sample, code] += 1


def count_matrix(
    entry_codes: np.ndarray,
    offsets: np.ndarray,
    ploidy: int,
    n_samples: int,
    sample_names: list[str],
    kind: str,
    per_sample: bool,
) -> "pl.DataFrame":
    counts = np.zeros((n_samples, N_CODES), dtype=np.int64)
    _count_kernel(
        entry_codes.astype(np.int16),
        offsets.astype(np.int64),
        np.int64(ploidy),
        np.int64(n_samples),
        np.int64(N_CODES),
        per_sample,
        counts,
    )
    lo, hi = code_ranges()[kind]
    block = counts[:, lo:hi]  # (n_samples, n_categories)
    out = {"MutationType": labels(kind)}
    for s_i, name in enumerate(sample_names):
        out[name] = block[s_i]
    return pl.DataFrame(out)
```

- [ ] **Step 4: Implement `SparseVar.mutation_matrix`**

In `genoray/_svar.py`, add import:

```python
from ._mutcat import count_matrix
```

Add the method to `SparseVar` (next to `annotate_mutations`):

```python
    def mutation_matrix(
        self,
        kind: Literal["SBS96", "DBS78", "ID83"],
        *,
        count: Literal["allele", "sample"] = "allele",
    ) -> pl.DataFrame:
        """Build a per-sample mutation count matrix.

        Requires :meth:`annotate_mutations` to have been run (or the ``mutcat``
        field to be loaded). Returns a DataFrame with a ``MutationType`` column
        plus one column per sample (rows in fixed codebook order).

        Parameters
        ----------
        kind
            One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
        count
            ``"allele"`` counts every non-ref allele copy; ``"sample"`` counts
            each category at most once per sample (presence).
        """
        if kind not in ("SBS96", "DBS78", "ID83"):
            raise ValueError(f"Unknown matrix kind {kind!r}.")
        mut = self.fields.get("mutcat")
        if mut is None:
            if "mutcat" in self.available_fields:
                mut = _open_fmt(
                    "mutcat", self.available_fields["mutcat"], self.path,
                    (self.n_samples, self.ploidy, None), "r",
                )
            else:
                raise ValueError(
                    "No 'mutcat' field found. Run annotate_mutations() first "
                    "(or open with fields=['mutcat'])."
                )
        return count_matrix(
            np.asarray(mut.data),
            np.asarray(self.genos.offsets),
            self.ploidy,
            self.n_samples,
            self.available_samples,
            kind,
            per_sample=(count == "sample"),
        )
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pixi run pytest tests/test_svar_mutations.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add genoray/_mutcat.py genoray/_svar.py tests/test_svar_mutations.py
git commit -m "feat(svar): add mutation_matrix with per-allele/per-sample counting"
```

---

## Task 10: Public exports, SKILL.md, and SigProfiler cross-check

**Files:**
- Modify: `genoray/__init__.py`
- Modify: `skills/genoray-api/SKILL.md`
- Test: `tests/test_svar_mutations.py` (optional cross-check, network-marked)

- [ ] **Step 1: Export `Reference`**

In `genoray/__init__.py`:
- add `"Reference"` to `__all__` (line 16).
- add to `_LAZY`: `"Reference": ("genoray._reference", "Reference"),`
- add to the `TYPE_CHECKING` block: `from ._reference import Reference as Reference`.

- [ ] **Step 2: Verify the import works**

Run: `pixi run python -c "import genoray; print(genoray.Reference)"`
Expected: prints `<class 'genoray._reference.Reference'>`.

- [ ] **Step 3: Write the failing test for the public surface**

Append to `tests/test_svar_mutations.py`:

```python
def test_public_reference_export():
    assert hasattr(genoray, "Reference")
    from genoray import Reference as R
    assert R is genoray.Reference
```

Run: `pixi run pytest tests/test_svar_mutations.py::test_public_reference_export -v`
Expected: PASS.

- [ ] **Step 4: Update SKILL.md**

In `skills/genoray-api/SKILL.md`, add documentation (matching the file's existing style) for:
- `genoray.Reference` — `from_path(fasta, contigs=None)`, `fetch(contig, start, end)`; pysam-backed; `chr`-prefix agnostic.
- `SparseVar.annotate_mutations(reference, *, write_back=True)` — classifies variants into SBS-96/DBS-78/ID-83, stores the int16 `mutcat` field per genotype entry; adjacent same-haplotype SNVs become DBS (5' = DBS code, 3' = `DBS_PARTNER`).
- `SparseVar.mutation_matrix(kind, *, count="allele"|"sample")` — returns a Polars DataFrame (`MutationType` column + one column per sample); `kind ∈ {SBS96, DBS78, ID83}`.
- The `mutcat` field (int16, enum-encoded) and the documented v1 scope limits (no strand bias; runs of ≥3 adjacent SNVs stay SBS; MNV > 2bp → UNCLASSIFIED).

- [ ] **Step 5: Optional SigProfiler cross-check (network-marked)**

If `SigProfilerMatrixGenerator` is available in the env, add a test marked `@pytest.mark.network` that runs it on a small fixture VCF and asserts the genoray SBS-96 / ID-83 column totals match SigProfiler's matrices. Use this to calibrate the ID-83 `n_rep`/microhomology accounting flagged in Task 5. If the package is unavailable, skip (document in the test docstring that calibration was done against published ID-83 rules).

- [ ] **Step 6: Run the full suite**

Run: `pixi run pytest tests/test_reference.py tests/test_mutcat.py tests/test_svar_mutations.py -v`
Expected: PASS. Also run `ruff check genoray tests && ruff format --check genoray tests`.

- [ ] **Step 7: Commit**

```bash
git add genoray/__init__.py skills/genoray-api/SKILL.md tests/test_svar_mutations.py
git commit -m "feat(svar): export Reference and document mutation catalogues in SKILL.md"
```

---

## Self-review notes

- **Spec coverage:** §1 deliverables → Tasks 8 (annotation) + 9 (matrices). §2 Reference → Task 1. §2 `_mutcat` → Tasks 2-7. §3 codebook/sentinels → Task 2. §4 SBS/DBS/ID algorithms → Tasks 3/4/5; adjacency → Task 7. §5 API → Tasks 8/9. §6 counting → Task 9. §7 testing → tests in every task + Task 10 cross-check. §8 docs → Task 10.
- **Known calibration risk:** ID-83 repeat/microhomology bucketing (Task 5 note) is the one area most likely to need adjustment against SigProfiler; isolated in one function, pinned by tests.
- **Type consistency:** `mutcat` is `int16` everywhere; codes use the unified space from `code_ranges()`; `build_entry_codes`/`count_matrix`/`classify_variants` signatures are referenced consistently across Tasks 6-9.

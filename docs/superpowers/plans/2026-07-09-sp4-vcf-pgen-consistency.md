# SP-4 VCF/PGEN Consistency + Boilerplate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the divergent `_vcf.py` / `_pgen.py` backends onto shared helpers and make the audit's invalid states unrepresentable, landing the public breaks in the genoray 3.0.0 wave.

**Architecture:** A new `genoray/_modes.py` provides phantom-mode factories both backends call, unifying the `empty()` signature. VCF gets a `Filter` value object (clean break), an `_extract_dosage` helper, and merged extend-length methods. Both backends get a mode→method dispatch dict and shared contig/empty helpers. The `_chunk_ranges_with_length` third tuple element is reconciled to `chunk_idxs` on both backends, with the matching change landed in GenVarLoader (branch `svar2-m6b-kernel`, PR #266).

**Tech Stack:** Python 3.10+, numpy, polars, phantom-types, cyvcf2, pgenlib; pixi env; pytest; ruff + pyrefly hooks; commitizen (Conventional Commits).

## Global Constraints

- **Release:** part of the **genoray 3.0.0** breaking-change wave. Mark every API-breaking commit with `!` and a `BREAKING CHANGE:` footer so `cz bump` resolves to 3.0.0. **Do NOT hand-edit the version** in `pyproject.toml`.
- **`empty()` contract:** the uniform signature is exactly `empty(n_samples, ploidy, n_variants)` across both backends (matches CLAUDE.md). VCF folds phasing into effective ploidy (`ploidy + phasing`) at the call site.
- **`_chunk_ranges_with_length` contract:** both backends yield `(data, end, chunk_idxs: NDArray[V_IDX_TYPE])`. `V_IDX_TYPE = np.uint32`.
- **Public-name changes require a `skills/genoray-api/SKILL.md` update in the same PR** (per `CLAUDE.md`). Public names touched here: `Filter` (new), the `VCF.filter` getter/setter and `VCF(...)` constructor kwargs.
- **Coordinated repos:** genoray on branch `sp4-vcf-pgen-consistency`; GenVarLoader on branch `svar2-m6b-kernel` (draft PR #266). The `_chunk_ranges_with_length` change is one logical edit split across both — neither is done without the other.
- **Behavior-preserving tasks** (2, 4, 5, 6, 8, 9) are gated on the existing suite staying green; use `refactor:`/`chore:` commits. **Breaking tasks** (3, 7, 10, 11, 12) get new/updated tests and `!` markers.
- **Verify commands:** `pixi run pytest` (Python), `pixi run pytest tests/test_vcf.py -q` / `tests/test_pgen.py` for focused runs. Rust is untouched; if a hook runs cargo tests use `--no-default-features`. Ensure prek hooks are installed (`pixi run prek-install`) before committing.

---

### Task 1: `_modes.py` phantom-mode factories

**Files:**
- Create: `python/genoray/_modes.py`
- Test: `tests/test_modes.py`

**Interfaces:**
- Produces:
  - `make_array_mode(name: str, dtype: type[np.generic], ndim: int, *, genos: bool = False) -> type` — a `Phantom` subclass of `NDArray[dtype]` with attributes `_dtype`, `_gdtype` (both `= dtype`), a `classmethod empty(n_samples, ploidy, n_variants)` (allocates `(n_samples, ploidy, n_variants)` when `ndim == 3`, else `(n_samples, n_variants)`), and a `classmethod nbytes_per_variant(n_samples, ploidy) -> int`. When `genos=True` the predicate additionally requires `shape[1] in (2, 3)`.
  - `make_tuple_mode(name: str, components: tuple[type, ...], *, genos_dtype: type[np.generic]) -> type` — a `Phantom` subclass of `tuple[components]` with `_dtype`/`_gdtype = genos_dtype`, `empty` delegating to each component, and `nbytes_per_variant` summing components.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_modes.py
import numpy as np
import pytest
from genoray._modes import make_array_mode, make_tuple_mode

Geno = make_array_mode("Geno", np.int8, 3, genos=True)
Dose = make_array_mode("Dose", np.float32, 2)
GenoDose = make_tuple_mode("GenoDose", (Geno, Dose), genos_dtype=np.int8)


def test_array_mode_empty_shapes():
    g = Geno.empty(3, 2, 5)
    assert g.shape == (3, 2, 5) and g.dtype == np.int8
    d = Dose.empty(3, 2, 5)  # ploidy ignored for 2D modes
    assert d.shape == (3, 5) and d.dtype == np.float32


def test_array_mode_predicate():
    assert isinstance(np.empty((3, 2, 5), np.int8), Geno)
    assert not isinstance(np.empty((3, 4, 5), np.int8), Geno)  # bad ploidy axis
    assert not isinstance(np.empty((3, 2, 5), np.int16), Geno)  # bad dtype


def test_nbytes_per_variant():
    # genos: n_samples * ploidy * itemsize; dosages: n_samples * 1 * itemsize
    assert Geno.nbytes_per_variant(3, 2) == 3 * 2 * 1
    assert Dose.nbytes_per_variant(3, 2) == 3 * 1 * 4
    assert GenoDose.nbytes_per_variant(3, 2) == 3 * 2 * 1 + 3 * 1 * 4


def test_tuple_mode_empty():
    g, d = GenoDose.empty(3, 2, 5)
    assert g.shape == (3, 2, 5) and d.shape == (3, 5)
    assert isinstance((g, d), GenoDose)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_modes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'genoray._modes'`.

- [ ] **Step 3: Implement `_modes.py`**

```python
# python/genoray/_modes.py
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from phantom import Phantom


def make_array_mode(
    name: str,
    dtype: type[np.generic],
    ndim: int,
    *,
    genos: bool = False,
) -> type:
    """Build a phantom ``NDArray`` mode class.

    Parameters
    ----------
    name
        The generated class ``__name__``.
    dtype
        NumPy scalar type the array must have (e.g. ``np.int8``).
    ndim
        Required number of dimensions (3 for genotype arrays, 2 for
        dosage/phasing-style arrays).
    genos
        If True, the ploidy axis (``shape[1]``) must be in ``(2, 3)`` and
        ``empty`` allocates a 3D ``(n_samples, ploidy, n_variants)`` array.
    """

    def predicate(obj: Any) -> bool:
        if not (
            isinstance(obj, np.ndarray)
            and obj.dtype.type == dtype
            and obj.ndim == ndim
        ):
            return False
        if genos:
            return obj.shape[1] in (2, 3)
        return True

    def empty(cls, n_samples: int, ploidy: int, n_variants: int):
        shape = (
            (n_samples, ploidy, n_variants)
            if ndim == 3
            else (n_samples, n_variants)
        )
        return cls.parse(np.empty(shape, dtype=dtype))

    def nbytes_per_variant(cls, n_samples: int, ploidy: int) -> int:
        axis = ploidy if ndim == 3 else 1
        return n_samples * axis * np.dtype(dtype).itemsize

    namespace = {
        "_dtype": dtype,
        "_gdtype": dtype,
        "empty": classmethod(empty),
        "nbytes_per_variant": classmethod(nbytes_per_variant),
    }
    return type(name, (NDArray[dtype], Phantom), namespace, predicate=predicate)


def make_tuple_mode(
    name: str,
    components: tuple[type, ...],
    *,
    genos_dtype: type[np.generic],
) -> type:
    """Build a phantom tuple-of-modes class (e.g. ``(Genos, Dosages)``)."""

    def predicate(obj: Any) -> bool:
        return (
            isinstance(obj, tuple)
            and len(obj) == len(components)
            and all(isinstance(o, c) for o, c in zip(obj, components))
        )

    def empty(cls, n_samples: int, ploidy: int, n_variants: int):
        return cls.parse(
            tuple(c.empty(n_samples, ploidy, n_variants) for c in components)
        )

    def nbytes_per_variant(cls, n_samples: int, ploidy: int) -> int:
        return sum(c.nbytes_per_variant(n_samples, ploidy) for c in components)

    namespace = {
        "_dtype": genos_dtype,
        "_gdtype": genos_dtype,
        "empty": classmethod(empty),
        "nbytes_per_variant": classmethod(nbytes_per_variant),
    }
    return type(name, (tuple[components], Phantom), namespace, predicate=predicate)
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_modes.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_modes.py tests/test_modes.py
git commit -m "feat(modes): add phantom-mode factories for array and tuple modes"
```

---

### Task 2: Migrate PGEN modes to the factory (behavior-preserving)

**Files:**
- Modify: `python/genoray/_pgen.py:37-147` (the six mode blocks)

**Interfaces:**
- Consumes: `make_array_mode`, `make_tuple_mode` from Task 1.
- Produces: unchanged public names `Genos`, `Dosages`, `Phasing`, `GenosPhasing`, `GenosDosages`, `GenosPhasingDosages`; `empty()` already 3-arg here, so this is pure code-motion.

- [ ] **Step 1: Baseline the suite is green**

Run: `pixi run pytest tests/test_pgen.py -q`
Expected: PASS (record the count).

- [ ] **Step 2: Replace the six mode blocks**

In `python/genoray/_pgen.py`, add near the top imports:

```python
from ._modes import make_array_mode, make_tuple_mode
```

Replace lines 37-147 (the `_is_genos`/`Genos` … `GenosPhasingDosages` definitions) with:

```python
Genos = make_array_mode("Genos", np.int32, 3, genos=True)
Dosages = make_array_mode("Dosages", np.float32, 2)
Phasing = make_array_mode("Phasing", np.bool_, 2)
GenosPhasing = make_tuple_mode("GenosPhasing", (Genos, Phasing), genos_dtype=np.int32)
GenosDosages = make_tuple_mode("GenosDosages", (Genos, Dosages), genos_dtype=np.int32)
GenosPhasingDosages = make_tuple_mode(
    "GenosPhasingDosages", (Genos, Phasing, Dosages), genos_dtype=np.int32
)
```

Note: PGEN's `Genos` predicate previously required `shape[1] == 2`; `genos=True` relaxes to `shape[1] in (2, 3)`. PGEN is always diploid (ploidy 2, no phasing axis), so the array is always `shape[1] == 2` in practice — the wider predicate never changes behavior. Keep the `T`/`L` `TypeVar`s at lines 150-151 unchanged (they reference the new names).

- [ ] **Step 3: Run PGEN + mode suites**

Run: `pixi run pytest tests/test_pgen.py tests/test_modes.py -q`
Expected: PASS, same count as Step 1.

- [ ] **Step 4: Commit**

```bash
git add python/genoray/_pgen.py
git commit -m "refactor(pgen): define phantom modes via the shared factory"
```

---

### Task 3: Migrate VCF modes to the factory + unify `empty()` to 3-arg (BREAKING)

**Files:**
- Modify: `python/genoray/_vcf.py:46-176` (mode blocks + TypeVars), and every `mode.empty(...)`/`Mode.empty(...)` call site in the file
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `make_array_mode`, `make_tuple_mode`.
- Produces: unchanged public names `Genos8`, `Genos16`, `Dosages`, `Genos8Dosages`, `Genos16Dosages`; **`empty()` signature changes** from `(n_samples, ploidy, n_variants, phasing)` to `(n_samples, ploidy, n_variants)`. Callers pass effective ploidy `self.ploidy + self.phasing`.

- [ ] **Step 1: Baseline the VCF suite is green**

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS (record the count).

- [ ] **Step 2: Replace the VCF mode blocks**

Add near the top imports of `python/genoray/_vcf.py`:

```python
from ._modes import make_array_mode, make_tuple_mode
```

Replace lines 46-170 (from `GDTYPE = TypeVar(...)` through `Genos16Dosages`) with:

```python
GDTYPE = TypeVar("GDTYPE", np.int8, np.int16)

Genos8 = make_array_mode("Genos8", np.int8, 3, genos=True)
Genos16 = make_array_mode("Genos16", np.int16, 3, genos=True)
Dosages = make_array_mode("Dosages", np.float32, 2)
Genos8Dosages = make_tuple_mode(
    "Genos8Dosages", (Genos8, Dosages), genos_dtype=np.int8
)
Genos16Dosages = make_tuple_mode(
    "Genos16Dosages", (Genos16, Dosages), genos_dtype=np.int16
)
```

Keep the `T`/`L`/`G`/`GD` `TypeVar`s at lines 173-176 unchanged.

- [ ] **Step 3: Update every VCF `empty()` call site to the 3-arg form**

The VCF genos axis length is `ploidy + phasing`. Change each call of the form
`mode.empty(self.n_samples, self.ploidy, N, self.phasing)` →
`mode.empty(self.n_samples, self.ploidy + self.phasing, N)`. Known sites (line numbers pre-edit; re-grep to be exact):

- `read`: 627-629, 636-638, 645-647
- `_chunk_ranges_with_length` empty yields: 848, 858, 877
- `_chunk_with_length_helper`: 920-921, 926
- `_fill_genos` empty (search `Genos8.empty`/`mode.empty` around 1290-1310)
- `_fill_dosages` empty: 1371
- `_fill_genos_and_dosages` empty: 1454
- `chunk` empty (search in the `chunk` method body)

After editing, verify no 4-arg empties remain:

Run: `grep -n "\.empty(.*self\.phasing" python/genoray/_vcf.py`
Expected: no output.

Run: `grep -nE "\.empty\([^)]*,[^)]*,[^)]*,[^)]*\)" python/genoray/_vcf.py`
Expected: no output (no 4-argument `.empty(` calls).

- [ ] **Step 4: Add the CHANGELOG entry**

In `CHANGELOG.md`, under an `## Unreleased` (or the 3.0.0 heading if present), add:

```markdown
### BREAKING CHANGES

- `Phantom` mode `empty()` classmethods now take a uniform
  `empty(n_samples, ploidy, n_variants)` signature on both VCF and PGEN
  backends. VCF's former 4th `phasing` argument is removed; pass the effective
  ploidy (`ploidy + phasing`) instead.
```

- [ ] **Step 5: Run the full suite**

Run: `pixi run pytest -q`
Expected: PASS, same VCF count as Step 1 (behavior unchanged; only the internal `empty()` arity changed).

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_vcf.py CHANGELOG.md
git commit -m "refactor(vcf)!: define modes via factory and unify empty() to 3-arg

BREAKING CHANGE: Phantom mode empty() drops the VCF-only phasing argument;
pass effective ploidy (ploidy + phasing) instead. Uniform across VCF/PGEN."
```

---

### Task 4: Extract `_extract_dosage` helper (VCF, behavior-preserving)

**Files:**
- Modify: `python/genoray/_vcf.py` — add `_extract_dosage`; replace the 5 copy-paste blocks

**Interfaces:**
- Produces: `VCF._extract_dosage(self, v: cyvcf2.Variant, dosage_field: str) -> NDArray[np.float32]` — returns the 1-D per-sample dosage (`d.squeeze(1)`), raising `DosageFieldError` if the field is absent and `MultiallelicDosageError` if `d.shape[1] > 1`.

- [ ] **Step 1: Baseline green**

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS.

- [ ] **Step 2: Add the helper**

Add to the `VCF` class (near `_fill_dosages`):

```python
def _extract_dosage(
    self, v: cyvcf2.Variant, dosage_field: str
) -> NDArray[np.float32]:
    """Fetch, validate, and squeeze the per-sample dosage for one record."""
    d = v.format(dosage_field)
    if d is None:
        raise DosageFieldError(
            f"Dosage field '{dosage_field}' not found for record {v!r}"
        )
    if d.shape[1] > 1:
        raise MultiallelicDosageError(
            f"Multiallelic dosages are not supported, encountered in VCF record {v!r}"
        )
    return d.squeeze(1)
```

- [ ] **Step 3: Replace the 5 blocks**

At each site, replace the `d = v.format(...); if d is None: raise ...; if d.shape[1] > 1: raise ...; ... squeeze(1)` block with a call:
- `_fill_dosages` (out-is-None ~1355-1365): `out_ls.append(self._extract_dosage(v, dosage_field))`
- `_fill_dosages` (out-not-None ~1391-1400): `out[..., i] = self._extract_dosage(v, dosage_field)[self._s_sorter]`
- `_fill_genos_and_dosages` (out-is-None ~1438-1448): `dosage_ls.append(self._extract_dosage(v, dosage_field))`
- `_fill_genos_and_dosages` (out-not-None branch): mirror with `[self._s_sorter]` indexing exactly as the original did
- `_ext_genos_dosages_with_length` (~1618-1624): `dosages = self._extract_dosage(v, dosage_field)[self._s_sorter, None]`
- `chunk` (~757-767): mirror the original indexing

Preserve each site's existing post-squeeze indexing (`[self._s_sorter]`, `[self._s_sorter, None]`) exactly — only the fetch+validate+squeeze moves into the helper.

- [ ] **Step 4: Run the suite**

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS (same count; the `DosageFieldError`/`MultiallelicDosageError` tests still pass through the helper).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_vcf.py
git commit -m "refactor(vcf): extract _extract_dosage helper for the 5 dosage sites"
```

---

### Task 5: Mode→method dispatch dicts (PGEN + VCF, behavior-preserving)

**Files:**
- Modify: `python/genoray/_pgen.py` — the 5 `issubclass` ladders (`read`, `chunk`, `read_ranges`, `chunk_ranges`, `_chunk_ranges_with_length` at ~905-914)
- Modify: `python/genoray/_vcf.py:623-665` — collapse the duplicated `read` dispatch

**Interfaces:**
- Produces (PGEN): `PGEN._reader_for(mode) -> Callable` returning the bound `_read_*` method for a mode via a dict lookup.

- [ ] **Step 1: Baseline green**

Run: `pixi run pytest tests/test_pgen.py tests/test_vcf.py -q`
Expected: PASS.

- [ ] **Step 2: Add the PGEN dispatch helper**

Add to `PGEN`:

```python
def _reader_for(self, mode: type) -> Callable:
    """Map a mode class to its bound ``_read_*`` method."""
    table = {
        Genos: self._read_genos,
        GenosPhasing: self._read_genos_phasing,
        GenosDosages: self._read_genos_dosages,
        GenosPhasingDosages: self._read_genos_phasing_dosages,
    }
    # Dosages-only maps to the genos reader's dosage path where applicable;
    # preserve each call site's existing mapping — see Step 3.
    try:
        return table[mode]
    except KeyError:
        assert_never(mode)  # type: ignore[bad-argument-type]
```

Important: before writing the table, read each of the 5 ladders and copy its exact mode→method mapping (they are not all identical — e.g. `read`/`chunk` include `Dosages`, while `_chunk_ranges_with_length` at 905-914 omits it). If the mappings differ across sites, give `_reader_for` a `include_dosages: bool = True` parameter (or build a second table) so each site keeps its exact current mapping. Do **not** unify a mode into a method it did not previously dispatch to.

- [ ] **Step 3: Replace each PGEN ladder**

At each of the 5 sites, replace the `if issubclass(mode, Genos): read = self._read_genos; elif ...` block with `read = self._reader_for(mode)` (passing `include_dosages=False` at the `_chunk_ranges_with_length` site if that flavor omitted `Dosages`). Keep the surrounding `read = cast(...)` lines.

- [ ] **Step 4: Collapse the VCF `read` dispatch**

In `python/genoray/_vcf.py` `read` (623-665), the `out is None` and `out is not None` branches duplicate the same mode dispatch. Rewrite so the dispatch is written once: compute `data` (allocating via `mode.empty(self.n_samples, self.ploidy + self.phasing, n_variants)` when `n_variants is not None`, else `None`) then dispatch once through the fill methods, threading `out` when provided. Preserve the exact `_fill_genos`/`_fill_dosages`/`_fill_genos_and_dosages` calls and the `assert self.dosage_field is not None` guards.

- [ ] **Step 5: Run the suite**

Run: `pixi run pytest tests/test_pgen.py tests/test_vcf.py -q`
Expected: PASS (same counts).

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_pgen.py python/genoray/_vcf.py
git commit -m "refactor(query): replace mode issubclass ladders with dispatch dicts"
```

---

### Task 6: Shared `_norm_or_warn` / `_empty` / `_empty_gen` helpers (behavior-preserving)

**Files:**
- Modify: `python/genoray/_pgen.py` (range/chunk method preambles) and `python/genoray/_vcf.py` (range/chunk method preambles)

**Interfaces:**
- Produces on each backend:
  - `_norm_or_warn(self, contig: str) -> str | None` — normalize the contig; on `None`, emit the existing `logger.warning(...)` (backend-correct message) and return `None`.
  - `_empty(self, mode, n_variants: int = 0)` — return `mode.empty(self.n_samples, <effective ploidy>, n_variants)` (VCF passes `self.ploidy + self.phasing`; PGEN passes `self.ploidy`).
  - `_empty_gen(self, mode, end)` — return the single-element generator `((self._empty(mode), end, <empty idx>) for _ in range(1))` used by the range methods (third element: `np.empty(0, dtype=V_IDX_TYPE)` after Task 11; a plain empty for pre-Task-11 shape — implement the current shape now, Task 11 updates it).

- [ ] **Step 1: Baseline green**

Run: `pixi run pytest tests/test_pgen.py tests/test_vcf.py -q`
Expected: PASS.

- [ ] **Step 2: Add the helpers to each backend**

PGEN:

```python
def _norm_or_warn(self, contig: str) -> str | None:
    c = self._c_norm.norm(contig) if self._c_norm is not None else None
    if c is None:
        logger.warning(
            f"Query contig {contig} not found in PGEN file, even after "
            "normalizing for UCSC/Ensembl nomenclature."
        )
    return c

def _empty(self, mode, n_variants: int = 0):
    return mode.empty(self.n_samples, self.ploidy, n_variants)
```

VCF (mirror, `PGEN`→`VCF` in the message, effective ploidy):

```python
def _norm_or_warn(self, contig: str) -> str | None:
    c = self._c_norm.norm(contig)
    if c is None:
        logger.warning(
            f"Query contig {contig} not found in VCF file, even after "
            "normalizing for UCSC/Ensembl nomenclature."
        )
    return c

def _empty(self, mode, n_variants: int = 0):
    return mode.empty(self.n_samples, self.ploidy + self.phasing, n_variants)
```

- [ ] **Step 3: Route the preambles through the helpers**

Replace each `c = self._c_norm.norm(contig); if c is None: logger.warning(...); <yield/return empty>` block with `c = self._norm_or_warn(contig); if c is None: <yield/return empty via self._empty/_empty_gen>`. Replace inline `mode.empty(self.n_samples, ...)` empty-result allocations with `self._empty(mode, n)`. Sites: PGEN 563-575, 649-663, 732-749, 859-891; VCF 608-613, 713-719, 844-861, and the empty yields inside `_chunk_ranges_with_length` (846-861, 874-879).

- [ ] **Step 4: Run the suite**

Run: `pixi run pytest tests/test_pgen.py tests/test_vcf.py -q`
Expected: PASS (same counts; log messages already backend-correct from SP-0).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_pgen.py python/genoray/_vcf.py
git commit -m "refactor(query): share contig-normalize and empty-result helpers"
```

---

### Task 7: `_mem_per_variant` via `nbytes_per_variant` + unify copy-doubling (BREAKING: chunk sizing)

**Files:**
- Modify: `python/genoray/_vcf.py:1510-1539` and `python/genoray/_pgen.py:948-968`
- Modify: `CHANGELOG.md`
- Test: `tests/test_vcf.py`

**Interfaces:**
- Consumes: `mode.nbytes_per_variant(n_samples, ploidy)` from Task 1.
- Produces: `_mem_per_variant(self, mode) -> int` on both backends, both applying the sample-copy doubling when a sorter array is active.

- [ ] **Step 1: Write the test for the VCF doubling change**

Add to `tests/test_vcf.py` (this asserts the *new* unified behavior — VCF now doubles when a sample subset/sorter is active):

```python
def test_vcf_mem_per_variant_doubles_when_sorted(vcf_path):
    vcf = VCF(vcf_path)
    base = vcf._mem_per_variant(VCF.Genos16)
    vcf.set_samples([vcf.available_samples[0]])  # activates the sample sorter
    assert vcf._mem_per_variant(VCF.Genos16) == 2 * vcf._mem_per_variant_unsorted(VCF.Genos16) \
        if hasattr(vcf, "_mem_per_variant_unsorted") else True
    # Simpler invariant: with a sorter active the estimate is doubled vs. the
    # per-mode nbytes for the same effective ploidy.
    ploidy = vcf.ploidy + vcf.phasing
    assert vcf._mem_per_variant(VCF.Genos16) == 2 * VCF.Genos16.nbytes_per_variant(
        vcf.n_samples, ploidy
    )
```

Use whatever VCF fixture the existing `tests/test_vcf.py` uses (e.g. the vcfixture path fixture); match its name.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_vcf.py::test_vcf_mem_per_variant_doubles_when_sorted -q`
Expected: FAIL — current VCF `_mem_per_variant` never doubles.

- [ ] **Step 3: Rewrite both `_mem_per_variant` bodies**

VCF:

```python
def _mem_per_variant(self, mode: type[T]) -> int:
    mem = mode.nbytes_per_variant(self.n_samples, self.ploidy + self.phasing)
    if isinstance(self._s_sorter, np.ndarray):
        mem *= 2  # a copy is made to reorder by samples
    return mem
```

PGEN:

```python
def _mem_per_variant(self, mode: type[T]) -> int:
    mem = mode.nbytes_per_variant(self.n_samples, self.ploidy)
    if isinstance(self._s_unsorter, np.ndarray):
        mem *= 2  # a copy is made to reorder by samples
    return mem
```

- [ ] **Step 4: Run the suite**

Run: `pixi run pytest tests/test_vcf.py tests/test_pgen.py -q`
Expected: PASS including the new test. Chunk-count-sensitive tests (if any) still pass — doubling only shrinks chunk sizes, never changes output values.

- [ ] **Step 5: CHANGELOG + commit**

Add under BREAKING CHANGES in `CHANGELOG.md`:

```markdown
- VCF chunk-size memory estimates now double when a sample subset/reorder is
  active, matching PGEN. This only affects internal chunk sizing (more
  conservative memory use), never returned data.
```

```bash
git add python/genoray/_vcf.py python/genoray/_pgen.py tests/test_vcf.py CHANGELOG.md
git commit -m "refactor(query)!: compute _mem_per_variant via nbytes_per_variant

BREAKING CHANGE: VCF now doubles the per-variant memory estimate when a sample
sorter is active, matching PGEN. Chunk-sizing only; no output change."
```

---

### Task 8: Merge the two `_ext_*_with_length` methods (VCF, behavior-preserving)

**Files:**
- Modify: `python/genoray/_vcf.py:1541-1643` (replace both methods with one) and their two call sites in `_chunk_with_length_helper` (939-952)

**Interfaces:**
- Produces: `VCF._ext_with_length(self, contig, start, end, hap_lens, mode, last_end, *, dosage_field: str | None = None) -> tuple[list, int]` — one method covering both genos-only (`dosage_field is None`) and genos+dosages (`dosage_field` set) extension, reusing `_extract_dosage`.

- [ ] **Step 1: Baseline green**

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS.

- [ ] **Step 2: Add the merged method**

Add a single module-level constant and one method (replacing lines 1541-1643):

```python
_CHECK_LEN_EVERY_N = 20


def _ext_with_length(
    self,
    contig: str,
    start: int | np.integer,
    end: int | np.integer,
    hap_lens: NDArray[np.int32],
    mode: type,
    last_end: int,
    *,
    dosage_field: str | None = None,
) -> tuple[list, int]:
    ploidy = self.ploidy + self.phasing
    length = end - start
    ext_start = end
    coord = f"{contig}:{ext_start + 1}"

    out_ls: list = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="no intervals found for", category=UserWarning
        )
        for i, v in enumerate(self._vcf(coord)):
            if v.start < ext_start or (
                self._filter is not None and not self._filter(v)
            ):
                continue

            genos = v.genotype.array()[:, :ploidy, None].astype(mode._gdtype)
            if dosage_field is None:
                out_ls.append(genos)
            else:
                dosages = self._extract_dosage(v, dosage_field)[self._s_sorter, None]
                out_ls.append((genos, dosages))

            if v.is_indel:
                ilen = len(v.ALT[0]) - len(v.REF)
                dist = v.start - last_end
                hap_lens += dist + np.where(
                    genos[:, : self.ploidy] == 1, ilen, 0
                ).squeeze(-1)
                last_end = cast(int, v.end)

            if i % _CHECK_LEN_EVERY_N == 0 and (hap_lens >= length).all():
                break

    if len(out_ls) > 0:
        last_end = cast(int, v.end)  # type: ignore | bound by len(out_ls) > 0

    return out_ls, last_end
```

Note: the genos-only path originally applied no `[self._s_sorter]` to `genos` (it used `v.genotype.array()` directly); preserve that — only the dosages path indexes by `_s_sorter`, exactly as in the original two methods.

- [ ] **Step 3: Update the two call sites**

In `_chunk_with_length_helper` (939-952) replace:
- the genos branch call `self._ext_genos_with_length(contig, start, end, hap_lens, mode, last_end)` → `self._ext_with_length(contig, start, end, hap_lens, mode, last_end)`
- the genos+dosages branch call → `self._ext_with_length(contig, start, end, hap_lens, mode, last_end, dosage_field=self.dosage_field)`

Remove the now-unused per-method `_CHECK_LEN_EVERY_N` locals.

- [ ] **Step 4: Run the suite**

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS (same count).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_vcf.py
git commit -m "refactor(vcf): merge _ext_genos/_ext_genos_dosages into _ext_with_length"
```

---

### Task 9: Relocate `_oxbow_reader` out of the overload block (VCF, trivial)

**Files:**
- Modify: `python/genoray/_vcf.py:1005-1012` (move below `get_record_info`)

- [ ] **Step 1: Move the method**

Cut the `_oxbow_reader` method body (currently interleaved between the last `@overload` stub ~995-1004 and the concrete `get_record_info` implementation ~1014) and paste it immediately **after** the concrete `get_record_info` implementation ends. No body change.

- [ ] **Step 2: Run the suite**

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add python/genoray/_vcf.py
git commit -m "refactor(vcf): move _oxbow_reader out of the get_record_info overloads"
```

---

### Task 10: `Filter` value object — clean break (BREAKING, public)

**Files:**
- Modify: `python/genoray/_vcf.py` — add `Filter`; replace `_filter`/`_pl_filter` with `_filter: Filter | None`; constructor + property; delete `_check_filter_pair`
- Modify: `python/genoray/__init__.py` — export `Filter`
- Modify: `skills/genoray-api/SKILL.md` — rewrite the Filtering section
- Modify: `CHANGELOG.md`
- Test: `tests/test_vcf.py`

**Interfaces:**
- Produces: `genoray.Filter` — `@dataclass(frozen=True) class Filter: record: Callable[[cyvcf2.Variant], bool]; expr: pl.Expr`. `VCF(path, filter: Filter | None = None, ...)`; `VCF.filter` getter/setter are `Filter | None`. Internal reads use `self._filter.record` / `self._filter.expr`.

- [ ] **Step 1: Write the new-API tests**

```python
# tests/test_vcf.py
import polars as pl
import pytest
from genoray import VCF, Filter


def test_filter_object_roundtrip(vcf_path):
    f = Filter(record=lambda v: True, expr=pl.col("POS") > 0)
    vcf = VCF(vcf_path, filter=f)
    assert vcf.filter is f
    vcf.filter = None
    assert vcf.filter is None


def test_filter_object_is_frozen():
    f = Filter(record=lambda v: True, expr=pl.lit(True))
    with pytest.raises(Exception):
        f.record = lambda v: False  # frozen dataclass


def test_old_pl_filter_kwarg_removed(vcf_path):
    with pytest.raises(TypeError):
        VCF(vcf_path, pl_filter=pl.lit(True))  # kwarg no longer exists
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_vcf.py -k filter_object -q`
Expected: FAIL — `ImportError: cannot import name 'Filter'`.

- [ ] **Step 3: Implement `Filter` and rewire VCF**

Add near the top of `python/genoray/_vcf.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Filter:
    """A cyvcf2 record predicate paired with its matching ``.gvi`` polars expression.

    Both travel together so the record scan (``record``) and the index-level
    filter (``expr``) can never diverge. ``record`` should return True for
    variants to keep; ``expr`` must be an equivalent predicate over the index
    columns (``CHROM``, ``POS``, ``REF``, ``ALT``, ``ILEN``).
    """

    record: Callable[[cyvcf2.Variant], bool]
    expr: pl.Expr
```

In `VCF`:
- Replace the `_filter` / `_pl_filter` attribute annotations (239-242) with a single `_filter: Filter | None`.
- Constructor (268-285): drop the `pl_filter` parameter; change `filter` to `filter: Filter | None = None`. Delete the `self._check_filter_pair(...)` call. Set `self._filter = filter`. The index-load guard `... and self._filter is None` (302) is unchanged.
- Delete `_check_filter_pair` (309-320).
- Rewrite the `filter` property/setter (322-367): getter returns `self._filter` (`Filter | None`); setter accepts `Filter | None`, invalidates the index (`self._index = None`), assigns `self._filter`.
- Update every internal read of the old pair:
  - record-scan sites (`if self._filter is not None: vcf = filter(self._filter, vcf)`) → `if self._filter is not None: vcf = filter(self._filter.record, vcf)`
  - the extend-length predicate `self._filter is not None and not self._filter(v)` → `... and not self._filter.record(v)`
  - the `_pl_filter` sites (index filtering at 1072-1073, 1256-1257, and the save/restore at 1158-1163) → use `self._filter.expr` guarded by `self._filter is not None`. For the temporary-clear block at 1158-1163, save/restore `self._filter` (the whole object) instead of `_pl_filter`.
  - `n_vars` no-index path (499-502) `self._filter(v)` → `self._filter.record(v)`.

Run to confirm no stale references remain:

Run: `grep -n "_pl_filter\|_check_filter_pair" python/genoray/_vcf.py`
Expected: no output.

- [ ] **Step 4: Export `Filter`**

In `python/genoray/__init__.py`: add `"Filter"` to `__all__`, add `"Filter": ("genoray._vcf", "Filter")` to the lazy-import map, and add `from ._vcf import Filter as Filter` in the `TYPE_CHECKING` block.

- [ ] **Step 5: Rewrite the SKILL Filtering section**

In `skills/genoray-api/SKILL.md`, replace the `(filter, pl_filter)` tuple documentation (constructor kwargs, the getter/setter tuple description ~323-334, and the Filtering-guidance block ~362-375) with the `Filter` object:

```python
from genoray import VCF, Filter
import genoray.exprs as gx

vcf = VCF("file.vcf", filter=Filter(
    record=lambda v: not v.INFO.get("SVTYPE"),   # cyvcf2 record predicate
    expr=~gx.is_symbolic,                          # matching .gvi index predicate
))
vcf.filter = None                                  # clear
f = vcf.filter                                     # -> Filter | None
```

State that VCF requires **both** halves and they are bundled in one `Filter`; the old `pl_filter=` kwarg and `(filter, pl_filter)` tuple are removed in 3.0.0.

- [ ] **Step 6: CHANGELOG + run + commit**

Add under BREAKING CHANGES:

```markdown
- VCF filtering now uses a single `genoray.Filter(record=, expr=)` value object.
  The `pl_filter=` constructor kwarg and the `(filter, pl_filter)` tuple
  getter/setter are removed. Migrate `VCF(p, filter=fn, pl_filter=expr)` to
  `VCF(p, filter=Filter(record=fn, expr=expr))`.
```

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS including the new filter tests. Update any existing tests that used `pl_filter=`/tuple filters to the `Filter` object.

```bash
git add python/genoray/_vcf.py python/genoray/__init__.py skills/genoray-api/SKILL.md CHANGELOG.md tests/test_vcf.py
git commit -m "feat(vcf)!: replace paired (filter, pl_filter) with a Filter value object

BREAKING CHANGE: VCF filtering uses genoray.Filter(record=, expr=). The
pl_filter= kwarg and the (filter, pl_filter) tuple API are removed."
```

---

### Task 11: Reconcile `_chunk_ranges_with_length` to `chunk_idxs` on VCF (BREAKING, genoray side)

**Files:**
- Modify: `python/genoray/_vcf.py` — `_chunk_ranges_with_length` (790-882) and `_chunk_with_length_helper` (884-969)
- Modify: `CHANGELOG.md`
- Test: `tests/test_vcf.py`

**Interfaces:**
- Consumes: `VCF._var_idxs(contig, starts, ends) -> (NDArray[integer], NDArray[OFFSET_TYPE])` (line 537, requires the index).
- Produces: VCF `_chunk_with_length_helper` and `_chunk_ranges_with_length` yield inner tuples `(data, end, chunk_idxs: NDArray[V_IDX_TYPE])` where `chunk_idxs` are the 0-based variant indices in the chunk, including extension variants (which are contiguous immediately after the last in-range index). Empty yields use `np.empty(0, dtype=V_IDX_TYPE)`.

- [ ] **Step 1: Write the characterization test (new contract == gvl's old reconstruction)**

```python
# tests/test_vcf.py
import numpy as np
from genoray import VCF


def test_chunk_ranges_with_length_yields_chunk_idxs(indexed_vcf_path):
    """The 3rd tuple element is now the variant-index array, and the extension
    indices are contiguous after the last in-range index (what gvl assumed)."""
    vcf = VCF(indexed_vcf_path)  # fixture with a .gvi index loaded
    contig = vcf.contigs[0]
    starts = np.array([0], dtype=np.int32)
    ends = np.array([10_000], dtype=np.int32)

    v_idx, offsets = vcf._var_idxs(contig, starts, ends)
    reg_unext = v_idx[offsets[0] : offsets[1]].astype(np.uint32)

    all_idxs = []
    for range_ in vcf._chunk_ranges_with_length(contig, starts, ends, "4g", VCF.Genos8):
        for genos, _end, chunk_idxs in range_:
            assert chunk_idxs.dtype == np.uint32
            assert chunk_idxs.shape[0] == genos.shape[-1]
            all_idxs.append(chunk_idxs)
    full = np.concatenate(all_idxs) if all_idxs else np.empty(0, np.uint32)

    # in-range prefix matches _var_idxs; any extension is contiguous after it
    assert np.array_equal(full[: reg_unext.size], reg_unext)
    if full.size > reg_unext.size:
        ext = full[reg_unext.size :]
        assert np.array_equal(ext, np.arange(reg_unext[-1] + 1, reg_unext[-1] + 1 + ext.size))
```

Use the existing indexed-VCF fixture name from `tests/test_vcf.py` (the one that constructs a `VCF` with a `.gvi` index). If none exists, build the index in the test via the same path the suite uses to force `_load_index()`.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_vcf.py -k chunk_idxs -q`
Expected: FAIL — the third element is currently an `int` (`n_extension_vars`), so `chunk_idxs.dtype` / `.shape` access raises.

- [ ] **Step 3: Thread in-range indices into the helper and emit `chunk_idxs`**

In `_chunk_ranges_with_length` (790), after normalizing `c` and computing per-range counts, compute the in-range indices once:

```python
v_idx, v_offsets = self._var_idxs(c, starts, ends)  # starts here are 0-based (pre +1)
```

Do this **before** the `starts = starts + 1` cyvcf2 adjustment (use the original 0-based `starts`). For each range `ri`, slice `range_idxs = v_idx[v_offsets[ri]:v_offsets[ri+1]].astype(V_IDX_TYPE)` and pass it to `_chunk_with_length_helper`.

Change the helper signature/body:

```python
def _chunk_with_length_helper(
    self,
    n: int,
    vars_per_chunk: int,
    contig: str,
    start: POS_TYPE,
    end: POS_TYPE,
    mode: type[L],
    range_idxs: NDArray[V_IDX_TYPE],
) -> Generator[tuple[L, int, NDArray[V_IDX_TYPE]]]:
    ...
    consumed = 0  # how many in-range indices emitted so far
    for _, is_last, chunk_size in mark_ends(chunk_sizes):
        ...  # existing read of `out`, `last_end`
        base_idxs = range_idxs[consumed : consumed + chunk_size]
        consumed += chunk_size
        if not is_last:
            yield cast(L, out), last_end, base_idxs
            continue

        # last chunk: extend and append contiguous extension indices
        ls_ext, last_end = self._ext_with_length(
            contig, start, end, hap_lens, mode, last_end,
            dosage_field=self.dosage_field if issubclass(
                mode, (Genos8Dosages, Genos16Dosages)
            ) else None,
        )
        if len(ls_ext) > 0:
            # concatenate genos/tuple exactly as before
            ...
            last_in_range = int(range_idxs[-1])
            ext_idxs = np.arange(
                last_in_range + 1, last_in_range + 1 + len(ls_ext), dtype=V_IDX_TYPE
            )
            chunk_idxs = np.concatenate([base_idxs, ext_idxs])
        else:
            chunk_idxs = base_idxs
        yield cast(L, out), last_end, chunk_idxs
```

Update the method return-type annotations (790-801, 884-892) from `tuple[L, int, int]` to `tuple[L, POS_TYPE, NDArray[V_IDX_TYPE]]`, matching PGEN. Update the empty-yield third elements (846-861, 874-879) to `np.empty(0, dtype=V_IDX_TYPE)` (fold through `_empty_gen` from Task 6). Simplify `_ext_with_length` dispatch by using the merged method from Task 8 (shown above).

Note: `range_idxs` is guaranteed non-empty whenever the helper runs (`n > 0` gate at the caller), so `range_idxs[-1]` is safe.

- [ ] **Step 4: Run the suite**

Run: `pixi run pytest tests/test_vcf.py -q`
Expected: PASS including the new characterization test.

- [ ] **Step 5: CHANGELOG + commit**

Add under BREAKING CHANGES:

```markdown
- `VCF._chunk_ranges_with_length` now yields `(data, end, chunk_idxs)` (a
  `uint32` variant-index array) as its third tuple element instead of an
  `n_extension_vars` count, matching `PGEN._chunk_ranges_with_length`.
```

```bash
git add python/genoray/_vcf.py CHANGELOG.md tests/test_vcf.py
git commit -m "refactor(vcf)!: yield chunk_idxs from _chunk_ranges_with_length

BREAKING CHANGE: the third tuple element is now a uint32 variant-index array
(matching PGEN), not an n_extension_vars count."
```

---

### Task 12: GenVarLoader coordination (branch `svar2-m6b-kernel`, PR #266)

**Files (in `/carter/users/dlaub/projects/GenVarLoader`):**
- Modify: `python/genvarloader/_dataset/_write.py:731-789` (VCF extend branch)
- Modify: `pyproject.toml` (genoray pin)

**Interfaces:**
- Consumes: the reconciled `(data, end, chunk_idxs)` VCF contract from Task 11.

- [ ] **Step 1: Switch to the gvl branch and install the local genoray**

```bash
cd /carter/users/dlaub/projects/GenVarLoader
git rev-parse --show-toplevel   # confirm you are in GenVarLoader, not genoray
git checkout svar2-m6b-kernel
git status                       # confirm clean before editing
```

Point gvl's env at the local genoray `sp4-vcf-pgen-consistency` build (editable/path install into the gvl pixi env) so tests exercise the new contract. Confirm: `pixi run python -c "import genoray, inspect; print(genoray.__file__)"` resolves to the local checkout.

- [ ] **Step 2: Rewrite the VCF extend branch to consume `chunk_idxs`**

In `python/genvarloader/_dataset/_write.py`, the VCF extend branch (738-768) currently reconstructs `var_idxs` from `n_ext_total`. Replace it to mirror the PGEN branch (852-869) — concatenate `chunk_idxs` directly:

```python
if extend_to_length:
    genos_list: list[NDArray] = []
    idx_list: list[NDArray] = []
    for chunk_genos, _chunk_end, chunk_idxs in range_:
        genos_list.append(chunk_genos)
        idx_list.append(chunk_idxs.astype(V_IDX_TYPE))
    genos = np.concatenate(genos_list, axis=-1)
    var_idxs = (
        np.concatenate(idx_list) if idx_list else np.empty(0, dtype=V_IDX_TYPE)
    )

    if var_idxs.size == 0:
        yield [dense2sparse(genos, var_idxs)], q_end, desc
        continue

    v_starts = (pos[var_idxs] - 1).astype(np.int32)
    ilens = ilen_all[var_idxs].astype(np.int32)
    rag = _window_to_sparse(genos, var_idxs, q_start, q_end, v_starts, ilens, True)
    region_end = _region_end(rag, v_ends, q_end)
    yield [rag], region_end, desc
```

Remove the now-dead `reg_unext`/`unextended_idxs`/`vcf._var_idxs` computation from the **extend** path (714-716) — but keep it for the non-extend path (774-782), which still needs `reg_unext`. Guard the `_var_idxs` call so it only runs when `not extend_to_length`.

- [ ] **Step 3: Bump the genoray pin**

In `GenVarLoader/pyproject.toml`, change `"genoray>=2.12.3,<3"` → `"genoray>=3,<4"`.

- [ ] **Step 4: Run the gvl write/dataset tests**

Run (in the gvl env): `pixi run pytest -k "write or svar2 or dataset" -q`
Expected: PASS. If a test hard-pins genoray `<3`, update it alongside the pin.

- [ ] **Step 5: Commit and push to PR #266**

```bash
cd /carter/users/dlaub/projects/GenVarLoader
git add python/genvarloader/_dataset/_write.py pyproject.toml
git commit -m "feat!: consume reconciled genoray chunk_idxs contract (genoray 3.0)

BREAKING CHANGE: requires genoray>=3; VCF/PGEN _chunk_ranges_with_length now
both yield (data, end, chunk_idxs)."
git push origin svar2-m6b-kernel
```

Confirm the push updates draft PR #266 (`gh pr view 266`).

- [ ] **Step 6: Return to genoray**

```bash
cd /carter/users/dlaub/projects/genoray
git rev-parse --abbrev-ref HEAD   # confirm sp4-vcf-pgen-consistency
```

---

### Task 13: SKILL `read(out=)` note + final reconciliation & verification

**Files:**
- Modify: `skills/genoray-api/SKILL.md`
- Modify: `CHANGELOG.md` (tidy the 3.0.0 section)

- [ ] **Step 1: Document the intentional asymmetries**

In `skills/genoray-api/SKILL.md`, add one line each near the VCF/PGEN method docs:
- `read(out=...)` is VCF-only (PGEN random-access allocates fresh; no `out=` buffer).
- Confirm the existing "VCF intentionally has no `read_ranges`" note is still present and correct (it is — do not remove).

- [ ] **Step 2: Verify the SKILL matches the code**

Run: `grep -n "pl_filter\|(filter, pl_filter)" skills/genoray-api/SKILL.md`
Expected: no output (all replaced by the `Filter` object in Task 10).

- [ ] **Step 3: Full verification**

Run: `pixi run pytest -q`
Expected: entire suite PASS.

Run: `ruff check python/genoray tests && ruff format --check python/genoray tests`
Expected: clean.

Run: `pixi run pyrefly check` (or the project's pyrefly task)
Expected: clean.

Run: `grep -rn "_check_filter_pair\|_ext_genos_with_length\|_ext_genos_dosages_with_length\|_pl_filter" python/genoray`
Expected: no output (all removed/renamed).

- [ ] **Step 4: Commit**

```bash
git add skills/genoray-api/SKILL.md CHANGELOG.md
git commit -m "docs(skill): document VCF-only read(out=) and finalize 3.0.0 notes"
```

- [ ] **Step 5: Open the genoray PR**

```bash
git push -u origin sp4-vcf-pgen-consistency
gh pr create --fill --title "SP-4: VCF/PGEN consistency + boilerplate (3.0.0)" \
  --body "Implements docs/superpowers/specs/2026-07-09-sp4-vcf-pgen-consistency-design.md. Coordinated with GenVarLoader PR #266. Part of the genoray 3.0.0 breaking-change wave."
```

---

## Self-Review

**Spec coverage:**
- §A1 phantom factory → Tasks 1–3. §A2 dispatch dict → Task 5. §A3 `_extract_dosage` → Task 4. §A4 norm/empty helpers → Task 6. §A5 mem unification → Task 7. §A6 `_ext_*` merge → Task 8. §A7 `_oxbow_reader` → Task 9.
- §B1 `empty()` unification → Task 3. §B2 `Filter` → Task 10. §B3 tuple reconcile → Tasks 11 (genoray) + 12 (gvl).
- §C1 range-API PGEN-only → Task 13 (verify note stays; no code). §C2 `read(out=)` → Task 13.
- Process (no hand-bump, `!` markers, SKILL/CHANGELOG) → Global Constraints + per-task commits.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". The one deliberately-open item from the spec (VCF `chunk_idxs` feasibility) was resolved during planning: VCF produces `chunk_idxs` via `_var_idxs` + contiguous extension indices (Task 11, Step 3), with a characterization test (Step 1).

**Type consistency:** `empty(n_samples, ploidy, n_variants)` uniform (Tasks 1–3). `nbytes_per_variant(n_samples, ploidy)` used in Tasks 1 & 7. `_chunk_ranges_with_length` third element `NDArray[V_IDX_TYPE]` (uint32) in Tasks 11 & 12. `Filter(record=, expr=)` consistent across Task 10 and the SKILL. `_ext_with_length(..., *, dosage_field=None)` consistent across Tasks 8 & 11.

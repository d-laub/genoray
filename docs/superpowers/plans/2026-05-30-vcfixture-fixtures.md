# vcfixture-generated Test Fixtures + GroundTruth Oracle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 4 hand-written `.vcf` files tracked in `tests/data/` with `vcfixture.VcfBuilder` definitions, and assert genoray reads against vcfixture's `GroundTruth` oracle wherever it mechanically fits.

**Architecture:** `VcfBuilder` definitions in `tests/data/fixtures.py` become the single source of truth. A new front-step in `gen_from_vcf.sh` writes the `.vcf` files; the existing bgzip→plink2→`gen_svar.py` pipeline is unchanged. A new `tests/_oracle.py` adapter reshapes `GroundTruth` `(records, samples, ploidy)` into genoray's `(samples, ploidy[+phasing], variants)` convention. Tests import the same fixtures to obtain truth.

**Tech Stack:** Python 3.10+, pixi, pytest + pytest-cases, vcfixture (editable path dep during dev), cyvcf2, numpy.

**Spec:** `docs/superpowers/specs/2026-05-30-vcfixture-fixtures-design.md`

**Key constraint:** `GroundTruth` decodes *file contents*. genoray range queries also use genoray's *own* selection logic (spanning deletions). The oracle supplies variant **values**; the test supplies **which** variant indices a range returns.

---

## Phase 0 — Worktrees & dependency wiring

### Task 0.1: Create the genoray worktree

**Files:** none (workspace setup)

- [ ] **Step 1: Create the worktree via the using-git-worktrees skill**

Invoke the `superpowers:using-git-worktrees` skill to create a worktree at `.claude/worktrees/test-vcfixture` on branch `test/vcfixture` off `main`. All remaining tasks run inside that worktree.

- [ ] **Step 2: Verify**

Run: `git -C .claude/worktrees/test-vcfixture rev-parse --abbrev-ref HEAD`
Expected: `test/vcfixture`

### Task 0.2: Create the vcfixture upstream worktree

**Files:** none (workspace setup, in the separate `/Users/david/projects/vcfixture` repo)

- [ ] **Step 1: Create a branch + worktree off vcfixture main**

The parallel genvarloader session uses vcfixture `main`, so do NOT touch `main`. Create an isolated worktree:

```bash
git -C /Users/david/projects/vcfixture worktree add \
  /Users/david/projects/vcfixture/.worktrees/genoray-fixtures -b feat/genoray-fixtures
```

- [ ] **Step 2: Verify**

Run: `git -C /Users/david/projects/vcfixture/.worktrees/genoray-fixtures rev-parse --abbrev-ref HEAD`
Expected: `feat/genoray-fixtures`

This worktree is where any vcfixture feature gap gets implemented. If none is needed, it stays empty and is removed at the end.

### Task 0.3: Add vcfixture as an editable path dev-dependency

**Files:**
- Modify: `pixi.toml` (the `[pypi-dependencies]` table, dev-deps area)

- [ ] **Step 1: Add the editable path dep**

In `pixi.toml`, under `[pypi-dependencies]`, in the `# dev deps` block (next to `seaborn`/`pooch`), add:

```toml
vcfixture = { path = "/Users/david/projects/vcfixture/.worktrees/genoray-fixtures", editable = true }
```

- [ ] **Step 2: Install and verify import**

Run: `pixi run -e py310 python -c "import vcfixture; print(vcfixture.__version__)"`
Expected: a version string prints (no ImportError). pysam + hypothesis resolve transitively.

- [ ] **Step 3: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "build: add vcfixture as editable-path dev dependency"
```

> **Pre-merge checklist (do NOT do now):** before merging `test/vcfixture`, replace this path dep with a published `vcfixture` version spec and re-lock. Merging an editable local-path dep is forbidden.

---

## Phase 1 — Fixture definitions, generation step, untracking

### Task 1.1: Write the fixture definitions module

**Files:**
- Create: `tests/data/fixtures.py`
- Test: `tests/test_fixtures_render.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixtures_render.py
from __future__ import annotations

import cyvcf2
import pytest

from tests.data.fixtures import FIXTURES


@pytest.mark.parametrize("name", ["biallelic", "multiallelic", "indels", "three_samples_unsorted"])
def test_fixture_renders_and_parses(tmp_path, name):
    builder = FIXTURES[name]()
    path = builder.write(tmp_path / f"{name}.vcf")
    vcf = cyvcf2.VCF(str(path))  # parses without error
    records = list(vcf)
    assert len(records) > 0


def test_biallelic_record_count():
    # 6 records: 3 on chr1, 3 on chr2
    builder = FIXTURES["biallelic"]()
    assert len(builder.build().records) == 6


def test_indels_positions():
    builder = FIXTURES["indels"]()
    pos = [r.pos for r in builder.build().records]
    assert pos[0] == 1000 and 5000 in pos
    # Region A: del at 1000 + 6 SNPs (1011..1021 step 2)
    assert 1011 in pos and 1021 in pos
    # Region C: del at 3000 + 40 SNPs (3031..3070)
    assert 3031 in pos and 3070 in pos
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e py310 pytest tests/test_fixtures_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.data.fixtures'`

- [ ] **Step 3: Write `tests/data/fixtures.py`**

Reproduces the exact content of today's `.vcf` files. DS is `Number.A` for biallelic/multiallelic, `Number(1)` for indels (matching the originals).

```python
# tests/data/fixtures.py
"""Single source of truth for genoray test VCFs.

Each function returns a vcfixture.VcfBuilder reproducing the exact contig/pos/
ref/alt/GT/DS content of the formerly-tracked .vcf files. `gen_vcfs.py` renders
these to tests/data/<name>.vcf; tests import the same builders to obtain the
decoded GroundTruth oracle.
"""

from __future__ import annotations

import numpy as np

from vcfixture import Number, Type, VcfBuilder

NAN = float("nan")


def biallelic() -> VcfBuilder:
    b = (
        VcfBuilder(
            samples=["sample1", "sample2"],
            contigs=[("chr1", None), ("chr2", None), ("chr3", None)],
            fileformat="VCFv4.1",
        )
        .fmt("GT")
        .fmt("DS", Number.A, Type.FLOAT)
    )
    # chr1
    b.record("chr1", 81262, ref="GAT", alt=["A"], gt=["0|1", "1|1"], DS=[[1.0], [2.0]])
    b.record("chr1", 81262, ref="G", alt=["A"], gt=["./.", "0/1"], DS=[NAN, [1.0]])
    b.record("chr1", 81265, ref="T", alt=["C"], gt=["1|0", "./."], DS=[[0.9], NAN])
    # chr2
    b.record("chr2", 81262, ref="GAT", alt=["A"], gt=["0|0", "1|1"], DS=[[0.0], [2.0]])
    b.record("chr2", 81262, ref="G", alt=["A"], gt=["./1", "0/1"], DS=[[1.0], [1.0]])
    b.record("chr2", 81265, ref="T", alt=["C"], gt=["1|0", "./."], DS=[[0.9], NAN])
    return b


def multiallelic() -> VcfBuilder:
    b = (
        VcfBuilder(
            samples=["sample1", "sample2"],
            contigs=[("chr1", None)],
            fileformat="VCFv4.1",
        )
        .fmt("GT")
        .fmt("DS", Number.A, Type.FLOAT)
    )
    b.record("chr1", 81262, ref="GAT", alt=["A"], gt=["0|1", "1|1"], DS=[[1.0], [2.0]])
    b.record(
        "chr1", 81262, ref="G", alt=["A", "C"],
        gt=["./.", "0/2"], DS=[NAN, [NAN, 1.0]],
    )
    return b


def three_samples_unsorted() -> VcfBuilder:
    b = (
        VcfBuilder(
            samples=["sample_C", "sample_A", "sample_B"],
            contigs=[("chr1", 200)],
            fileformat="VCFv4.2",
        )
        .fmt("GT")
    )
    b.record("chr1", 100, ref="T", alt=["A"], gt=["0|1", "1|1", "0|0"])
    return b


def indels() -> VcfBuilder:
    """with_length edge-case fixture. POS preserved verbatim.

    Region A (1000): -10 deletion + 6 SNPs. Region B (2000): -4 deletion het on
    sample1 hapA only + 3 SNPs. Region C (3000): -30 deletion + 40 dense SNPs.
    Region D (5000): lone -10 deletion as last variant. Not a realistic genome.
    """
    b = (
        VcfBuilder(
            samples=["sample1", "sample2"],
            contigs=[("chr1", None)],
            fileformat="VCFv4.2",
        )
        .fmt("GT")
        .fmt("DS", Number(1), Type.FLOAT)
    )

    def snp(pos: int, gt: list[str], ds: list[float]) -> None:
        b.record("chr1", pos, ref="T", alt=["C"], gt=gt, DS=ds)

    # Region A
    b.record("chr1", 1000, ref="G" + "A" * 10, alt=["G"], gt=["1|1", "1|1"], DS=[2.0, 2.0])
    for p in range(1011, 1022, 2):  # 1011,1013,1015,1017,1019,1021
        snp(p, ["1|1", "1|1"], [2.0, 2.0])

    # Region B
    b.record("chr1", 2000, ref="G" + "A" * 4, alt=["G"], gt=["1|0", "0|0"], DS=[1.0, 0.0])
    for p in (2002, 2004, 2006):
        snp(p, ["1|1", "1|1"], [2.0, 2.0])

    # Region C
    b.record("chr1", 3000, ref="G" + "A" * 30, alt=["G"], gt=["1|1", "1|1"], DS=[2.0, 2.0])
    for p in range(3031, 3071):  # 3031..3070 inclusive (40 SNPs)
        snp(p, ["1|1", "1|1"], [2.0, 2.0])

    # Region D
    b.record("chr1", 5000, ref="G" + "A" * 10, alt=["G"], gt=["1|1", "1|1"], DS=[2.0, 2.0])
    return b


FIXTURES = {
    "biallelic": biallelic,
    "multiallelic": multiallelic,
    "three_samples_unsorted": three_samples_unsorted,
    "indels": indels,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e py310 pytest tests/test_fixtures_render.py -v`
Expected: PASS (all cases)

- [ ] **Step 5: Commit**

```bash
git add tests/data/fixtures.py tests/test_fixtures_render.py
git commit -m "test: add vcfixture fixture definitions for test VCFs"
```

### Task 1.2: Write the generation script

**Files:**
- Create: `tests/data/gen_vcfs.py`

- [ ] **Step 1: Write `tests/data/gen_vcfs.py`**

```python
# tests/data/gen_vcfs.py
"""Render every fixture in fixtures.FIXTURES to tests/data/<name>.vcf."""

from __future__ import annotations

from pathlib import Path

from fixtures import FIXTURES  # run from tests/data/ via gen_from_vcf.sh


def main() -> None:
    ddir = Path(__file__).parent
    for name, build in FIXTURES.items():
        out = ddir / f"{name}.vcf"
        build().write(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it standalone to verify the 4 files appear**

Run: `cd tests/data && pixi run -e py310 python gen_vcfs.py && cd ../..`
Expected: prints 4 `wrote .../<name>.vcf` lines; `git status` shows the `.vcf` modified (still tracked at this point).

- [ ] **Step 3: Commit**

```bash
git add tests/data/gen_vcfs.py
git commit -m "test: add gen_vcfs.py to render fixtures to .vcf"
```

### Task 1.3: Wire generation into gen_from_vcf.sh and verify the full pipeline

**Files:**
- Modify: `tests/data/gen_from_vcf.sh` (add a first step)

- [ ] **Step 1: Add the generation front-step**

In `tests/data/gen_from_vcf.sh`, immediately after the `indels=...` variable
assignments and before the `echo "Bgzipping..."` line, insert:

```bash
echo "Generating VCFs from vcfixture builders..."
python "$ddir"/gen_vcfs.py
```

- [ ] **Step 2: Run the full data-gen pipeline**

Run: `pixi run gen`
Expected: VCF generation runs first, then bgzip/index, plink2, and `gen_svar.py` all succeed (exit 0). `.vcf.gz`, `.pgen`, `.svar` regenerated.

- [ ] **Step 3: Run the full test suite against generated data (no assertion changes yet)**

Run: `pixi run -e py310 pytest tests -q`
Expected: PASS — identical to pre-change behavior. This proves the generated VCFs are semantically equivalent to the originals.

- [ ] **Step 4: Commit**

```bash
git add tests/data/gen_from_vcf.sh
git commit -m "test: generate test VCFs via vcfixture in gen_from_vcf.sh"
```

### Task 1.4: Prove semantic equivalence, then untrack the .vcf files

**Files:**
- Modify: `tests/data/.gitignore`
- Modify: `tests/data/README.md`
- Remove from tracking: `tests/data/{biallelic,multiallelic,indels,three_samples_unsorted}.vcf`

- [ ] **Step 1: Decode-compare generated vs. original VCFs (one-time equivalence check)**

Restore the originals from git into a temp dir and compare records with cyvcf2:

```bash
mkdir -p /tmp/orig_vcf
for n in biallelic multiallelic indels three_samples_unsorted; do
  git show HEAD:tests/data/$n.vcf > /tmp/orig_vcf/$n.vcf
done
pixi run -e py310 python - <<'PY'
import cyvcf2
for n in ["biallelic","multiallelic","indels","three_samples_unsorted"]:
    a = list(cyvcf2.VCF(f"/tmp/orig_vcf/{n}.vcf"))
    b = list(cyvcf2.VCF(f"tests/data/{n}.vcf"))
    assert len(a) == len(b), (n, len(a), len(b))
    for x, y in zip(a, b):
        assert (x.CHROM, x.POS, x.REF, tuple(x.ALT)) == (y.CHROM, y.POS, y.REF, tuple(y.ALT)), (n, str(x), str(y))
        assert x.genotypes == y.genotypes, (n, str(x))
    print(n, "OK", len(a), "records")
PY
```

Expected: each fixture prints `OK` with the same record count. Note: today's originals are the source git copies; this check runs *before* untracking.

- [ ] **Step 2: Untrack the .vcf and ignore them going forward**

In `tests/data/.gitignore`, remove the `!*.vcf` line so generated `.vcf` are ignored. Then stop tracking the 4 files:

```bash
git rm --cached tests/data/biallelic.vcf tests/data/multiallelic.vcf \
  tests/data/indels.vcf tests/data/three_samples_unsorted.vcf
```

- [ ] **Step 3: Update README**

Replace the first sentence of `tests/data/README.md` (currently "All test data are sourced from the .vcf files...") with:

```
Test VCFs are generated from the vcfixture builders in `fixtures.py` — edit
those, not any `.vcf` file. After changes, run `gen_from_vcf.sh` (or
`pixi run gen`) to regenerate the .vcf, .vcf.gz, .pgen, and .svar derivatives.
Note that PLINK 2 does not support multi-allelics and dosages at the same time!
Thus, there is a bi-allelic VCF with dosages and a multi-allelic VCF without.
```

- [ ] **Step 4: Verify the .vcf are ignored and the suite still passes**

Run: `git status --porcelain tests/data/*.vcf` → expect no output (ignored).
Run: `pixi run -e py310 pytest tests -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/data/.gitignore tests/data/README.md
git commit -m "test: untrack static VCFs; fixtures.py is now source of truth"
```

---

## Phase 2 — Oracle adapter + conftest

### Task 2.1: Write the oracle adapter

**Files:**
- Create: `tests/_oracle.py`
- Test: `tests/test_oracle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle.py
from __future__ import annotations

import numpy as np

from tests import _oracle
from tests.data.fixtures import FIXTURES


def _truth(name):
    return FIXTURES[name]().truth()


def test_genos_shape_and_values():
    truth = _truth("biallelic")
    # all 6 records, both samples, ploidy 2 -> (samples=2, ploidy=2, variants=6)
    g = _oracle.genos(truth, slice(None))
    assert g.shape == (2, 2, 6)
    # record 0 (chr1:81262 GAT>A): sample1 0|1, sample2 1|1
    np.testing.assert_array_equal(g[:, :, 0], np.array([[0, 1], [1, 1]]))
    # record 1: sample1 ./. -> [-1,-1]
    np.testing.assert_array_equal(g[0, :, 1], np.array([-1, -1]))


def test_phasing_shape_and_values():
    truth = _truth("biallelic")
    p = _oracle.phasing(truth, slice(None))
    assert p.shape == (2, 6)
    # record 0: sample1 0|1 phased, sample2 1|1 phased
    assert p[0, 0] and p[1, 0]
    # record 1: sample2 0/1 unphased
    assert not p[1, 1]


def test_dosages_missing_is_nan():
    truth = _truth("biallelic")
    d = _oracle.dosages(truth, slice(None))
    assert d.shape == (2, 6)
    np.testing.assert_array_equal(d[:, 0], np.array([1.0, 2.0], np.float32))
    assert np.isnan(d[0, 1])  # sample1 record1 DS "."


def test_index_subset():
    truth = _truth("biallelic")
    idx = [0, 2]
    g = _oracle.genos(truth, idx)
    assert g.shape == (2, 2, 2)


def test_split_phased_read_roundtrip():
    truth = _truth("biallelic")
    g = _oracle.genos(truth, slice(None)).astype(np.int8)
    p = _oracle.phasing(truth, slice(None))
    # emulate genoray phased output (s, ploidy+1, v): stack phasing as last row
    stacked = np.concatenate([g, p[:, None, :].astype(np.int8)], axis=1)
    g2, p2 = _oracle.split_phased(stacked)
    np.testing.assert_array_equal(g2, g)
    np.testing.assert_array_equal(p2, p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e py310 pytest tests/test_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests._oracle'`

- [ ] **Step 3: Write `tests/_oracle.py`**

```python
# tests/_oracle.py
"""Adapt vcfixture GroundTruth to genoray's array conventions.

GroundTruth.genotypes is (records, samples, ploidy); genoray reads are
(samples, ploidy[+phasing], variants). These helpers transpose/slice truth so
tests assert genoray output against a decoded oracle instead of literals.

The oracle supplies variant VALUES. It does NOT model genoray's range-selection
logic (spanning deletions): callers pass the variant indices a query returns.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from vcfixture import GroundTruth

Index = slice | Sequence[int] | NDArray[np.intp]


def genos(truth: GroundTruth, idx: Index) -> NDArray[np.int32]:
    """(samples, ploidy, variants), -1 = missing."""
    sub = truth.genotypes[idx]  # (v, s, p)
    return np.ascontiguousarray(sub.transpose(1, 2, 0))


def phasing(truth: GroundTruth, idx: Index) -> NDArray[np.bool_]:
    """(samples, variants) — True where fully phased."""
    sub = truth.phasing[idx]  # (v, s)
    return np.ascontiguousarray(sub.transpose(1, 0))


def dosages(truth: GroundTruth, idx: Index, field: str = "DS") -> NDArray[np.float32]:
    """(samples, variants); missing/"."/NaN -> np.nan.

    Mirrors genoray's dosage_field convention: one scalar per (sample, variant).
    For Number=A fields the first ALT's dosage is used (the biallelic fixture is
    1:1; the multiallelic fixture has no PGEN dosage path).
    """
    fmt = truth.format  # list[record] -> list[sample] -> dict
    n_rec = len(fmt)
    rec_ids = list(range(n_rec))[idx] if isinstance(idx, slice) else list(idx)
    n_smp = len(truth.samples)
    out = np.full((n_smp, len(rec_ids)), np.nan, np.float32)
    for vi, ri in enumerate(rec_ids):
        per_sample = fmt[ri]
        for si in range(n_smp):
            val = per_sample[si].get(field)
            if isinstance(val, (list, tuple)):
                val = val[0] if len(val) else None
            if val is None:
                continue
            fval = float(val)
            out[si, vi] = np.nan if np.isnan(fval) else fval
    return out


def split_phased(gp: NDArray) -> tuple[NDArray, NDArray[np.bool_]]:
    """Split genoray phased output (s, ploidy+1, v) into (genos, phasing)."""
    g, p = np.array_split(gp, 2, axis=1)
    return g, p.squeeze(1).astype(bool)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e py310 pytest tests/test_oracle.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/_oracle.py tests/test_oracle.py
git commit -m "test: add GroundTruth->genoray oracle adapter"
```

### Task 2.2: Add a top-level conftest exposing truth fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
# tests/conftest.py
from __future__ import annotations

import pytest

from tests.data.fixtures import FIXTURES


@pytest.fixture(scope="session")
def truths():
    """name -> vcfixture GroundTruth for every fixture."""
    return {name: build().truth() for name, build in FIXTURES.items()}


@pytest.fixture(scope="session")
def biallelic_truth(truths):
    return truths["biallelic"]


@pytest.fixture(scope="session")
def multiallelic_truth(truths):
    return truths["multiallelic"]


@pytest.fixture(scope="session")
def indels_truth(truths):
    return truths["indels"]


@pytest.fixture(scope="session")
def three_samples_truth(truths):
    return truths["three_samples_unsorted"]
```

- [ ] **Step 2: Verify fixtures are discoverable**

Run: `pixi run -e py310 python -c "from tests.data.fixtures import FIXTURES; print({k: len(v().build().records) for k,v in FIXTURES.items()})"`
Expected: prints record counts, e.g. `{'biallelic': 6, 'multiallelic': 2, 'three_samples_unsorted': 1, 'indels': 53}` (indels: 4 deletions + 6 + 3 + 40 SNPs).

Run: `pixi run -e py310 pytest tests -q` → still PASS (conftest adds fixtures, changes nothing yet).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: expose GroundTruth session fixtures via conftest"
```

---

## Phase 3 — Migrate dense read-path tests to the oracle

### Task 3.1: Migrate `test_vcf.py` value assertions (worked exemplar)

**Files:**
- Modify: `tests/test_vcf.py` (the `read_*` case functions, lines ~30-65)

The `read_*` case functions return hand-coded `genos`/`phasing`/`dosages` for a
range query `cse`. Replace the literals with oracle calls keyed to the variant
indices that genoray returns for that range. The `cse`/index mapping is genoray's
selection logic, stated explicitly per case.

- [ ] **Step 1: Add oracle import and a session-truth handle**

At the top of `tests/test_vcf.py`, add:

```python
from tests import _oracle
from tests.data.fixtures import FIXTURES

_BIALLELIC = FIXTURES["biallelic"]().truth()
```

- [ ] **Step 2: Rewrite `read_all` and `read_spanning_del` to derive from the oracle**

`read_all` queries `chr1:81261-81263` and genoray returns the first 2 chr1
records (indices 0,1). `read_spanning_del` queries `chr1:81262-81263` and returns
the spanning-deletion records (indices 0,1 on chr1) collapsed to ploidy-1 reads.
Keep the explicit index lists; replace the value literals:

```python
def read_all():
    cse = "chr1", 81261, 81263
    idx = [0, 1]  # genoray returns chr1 records 0 and 1 for this range
    genos = _oracle.genos(_BIALLELIC, idx).astype(np.int8)
    phasing = _oracle.phasing(_BIALLELIC, idx)
    dosages = _oracle.dosages(_BIALLELIC, idx)
    return cse, genos, phasing, dosages
```

> NOTE: only migrate cases where the oracle reproduces the existing literal
> exactly. `read_spanning_del` returns a ploidy-collapsed `(s, 1, v)` view that
> reflects genoray's spanning-deletion read, NOT a plain transpose of truth —
> leave its hand-coded arrays in place and add a one-line comment: `# spanning-del
> read shape is genoray-specific; not a clean oracle case`. Same for the empty
> `read_missing_contig`/`read_none` cases (already use `.empty(...)`).

- [ ] **Step 3: Run the migrated tests**

Run: `pixi run -e py310 pytest tests/test_vcf.py -q`
Expected: PASS (identical outcomes; values now oracle-derived where migrated).

- [ ] **Step 4: Sanity-check the oracle actually matches (guard against trivially-equal bugs)**

Temporarily break a fixture value (e.g. change `read_all` idx to `[0, 2]`), run
`pytest tests/test_vcf.py::test_read -q`, confirm it FAILS, then revert. This
proves the assertions are live.

- [ ] **Step 5: Commit**

```bash
git add tests/test_vcf.py
git commit -m "test: assert VCF dense reads against GroundTruth oracle"
```

### Task 3.2: Migrate `test_pgen.py` value assertions

**Files:**
- Modify: `tests/test_pgen.py`

PGEN reads the same `biallelic`/`indels` data. PGEN genotypes are int32 with the
same `(samples, ploidy, variants)` layout; PGEN has no phasing-row concept in the
same way — assert genos/dosages via the oracle, leave phasing handling as-is.

- [ ] **Step 1: Add imports + truth handle**

```python
from tests import _oracle
from tests.data.fixtures import FIXTURES

_BIALLELIC = FIXTURES["biallelic"]().truth()
```

- [ ] **Step 2: Replace hand-coded expected genotype/dosage arrays**

For each test that asserts a literal genotype/dosage array for a known range,
replace the literal with `_oracle.genos(_BIALLELIC, idx)` / `_oracle.dosages(...)`
using the same index list the test already implies. Where PGEN normalizes
half-calls differently from cyvcf2 (e.g. `vcf-half-call r` reference-fills), keep
the existing literal and comment `# PGEN half-call normalization differs from
GroundTruth; literal retained`.

- [ ] **Step 3: Run**

Run: `pixi run -e py310 pytest tests/test_pgen.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pgen.py
git commit -m "test: assert PGEN dense reads against GroundTruth oracle"
```

---

## Phase 4 — Migrate remaining tests where the oracle mechanically fits

For each file below, the recipe is identical: import `_oracle` + the relevant
truth, replace literal expected arrays with oracle calls keyed to the variant
indices the test uses, run the file's tests, confirm PASS, commit. Keep any
literal the oracle cannot reproduce exactly (sparse internals, with_length
parsimony, PGEN half-call normalization) with a one-line comment explaining why.

### Task 4.1: Migrate `test_svar.py`

**Files:** Modify `tests/test_svar.py`

- [ ] **Step 1:** Add `from tests import _oracle` + `_BIALLELIC = FIXTURES["biallelic"]().truth()`.
- [ ] **Step 2:** Replace literal genotype/dosage expectations for `biallelic.vcf.svar` reads with `_oracle.genos`/`_oracle.dosages` at the matching indices. Leave AF/sparse-offset internals (`test_svar_internals.py` territory) untouched.
- [ ] **Step 3:** Run `pixi run -e py310 pytest tests/test_svar.py -q` → PASS.
- [ ] **Step 4:** Commit `git commit -am "test: assert SVAR reads against GroundTruth oracle"`.

### Task 4.2: Migrate `test_vcf_set_samples.py`

**Files:** Modify `tests/test_vcf_set_samples.py`

Uses `three_samples_unsorted` (sample order C/A/B). The oracle's `samples` tuple
preserves that order, so sample-reindexing assertions can derive expected
genotypes from `_oracle.genos(three_samples_truth, idx)` then reorder columns to
match the requested sample subset.

- [ ] **Step 1:** Import `_oracle`; use the `three_samples_truth` conftest fixture.
- [ ] **Step 2:** Replace literal genotype expectations with oracle-derived arrays, reordering the sample axis to match each test's `set_samples(...)` argument.
- [ ] **Step 3:** Run `pixi run -e py310 pytest tests/test_vcf_set_samples.py -q` → PASS.
- [ ] **Step 4:** Commit `git commit -am "test: assert set_samples reads against GroundTruth oracle"`.

### Task 4.3: Migrate `test_issue36.py`

**Files:** Modify `tests/test_issue36.py`

- [ ] **Step 1:** Import `_oracle` + `_BIALLELIC`.
- [ ] **Step 2:** Replace the chr1:81262/81265 literal expectations with oracle-derived values at the matching indices.
- [ ] **Step 3:** Run `pixi run -e py310 pytest tests/test_issue36.py -q` → PASS.
- [ ] **Step 4:** Commit `git commit -am "test: assert issue36 regression against GroundTruth oracle"`.

### Task 4.4: Audit the remaining test files (no forced migration)

**Files:** review only — `tests/test_parity.py`, `tests/test_dense2sparse_with_length.py`, `tests/test_svar_internals.py`, `tests/test_svar_write_view.py`, `tests/test_gtf_annotation.py`, `tests/cli/*`, `tests/test_utils.py`, `tests/test_lazy_init.py`.

- [ ] **Step 1:** For each, decide if any literal genotype/dosage array maps cleanly to the oracle. If yes, migrate using the Phase-4 recipe and commit per file. If no (sparse offsets, with_length parsimony, GTF/CLI/util logic, no VCF-fixture genotypes), leave as-is.
- [ ] **Step 2:** Record the decision for each file in a short comment block at the top of this plan's PR description (which files migrated, which deliberately did not and why).

---

## Phase 5 — Full verification & cleanup

### Task 5.1: Full-suite green across the default environment

- [ ] **Step 1:** Run the canonical command: `pixi run test` (regenerates data via `gen`, then runs `pytest tests`).
Expected: PASS, exit 0.
- [ ] **Step 2:** Confirm no `.vcf` are tracked: `git ls-files tests/data/*.vcf` → empty output.
- [ ] **Step 3:** Confirm tracked sources of truth present: `git ls-files tests/data` → includes `fixtures.py`, `gen_vcfs.py`, `gen_from_vcf.sh`, `gen_svar.py`, `README.md`, `.gitignore`.

### Task 5.2: vcfixture upstream worktree disposition

- [ ] **Step 1:** If no vcfixture feature gap was hit, remove the empty worktree:
`git -C /Users/david/projects/vcfixture worktree remove .worktrees/genoray-fixtures`.
- [ ] **Step 2:** If a feature WAS added in vcfixture, commit it there, push `feat/genoray-fixtures`, and cut/publish a vcfixture release that includes it.

### Task 5.3: Pre-merge dependency pin (gate before merging `test/vcfixture`)

- [ ] **Step 1:** In `pixi.toml`, replace the editable path `vcfixture = { path = ..., editable = true }` with a published version spec, e.g. `vcfixture = ">=0.2.1,<0.3"` (bump if an upstream feature was needed). Re-lock: `pixi install`.
- [ ] **Step 2:** Run `pixi run test` → PASS with the released vcfixture.
- [ ] **Step 3:** Commit `git commit -am "build: pin vcfixture to a published release"`.
- [ ] **Step 4:** Use the `superpowers:finishing-a-development-branch` skill to merge/PR.

---

## Self-Review notes

- **Spec coverage:** §1 worktrees → Tasks 0.1–0.2; §2 dependency → 0.3 + 5.3; §3 fixtures → 1.1; §4 generation/untracking → 1.2–1.4; §5 oracle adapter → 2.1; §6 conftest → 2.2; §7 migration phases → 3.1–3.2 (dense) + 4.1–4.4 (everywhere it fits); §8 verification → 1.3/1.4 step 1 + 5.1; §9 non-goals (no SKILL.md change, plink2 stays in shell, no new fixtures) honored throughout.
- **No public API change** → `skills/genoray-api/SKILL.md` intentionally untouched.
- **Type consistency:** oracle functions `genos`/`phasing`/`dosages`/`split_phased` are defined once in Task 2.1 and used by those exact names in Tasks 2.1–4.x. `FIXTURES` dict and per-fixture builder names match between 1.1 and all consumers.

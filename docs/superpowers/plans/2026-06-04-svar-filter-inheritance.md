# SVAR Filter Inheritance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `skip_symbolic_alts` flag and make `SparseVar.from_vcf` / `from_pgen` inherit and apply the source's filter, so symbolic-allele filtering is expressed as a normal filter (`pl_filter=~exprs.is_symbolic`).

**Architecture:** Fix the latent `VCF._load_index` ALT-ordering bug so list-typed expressions work on the VCF path; then make both SparseVar builders write a *filtered* index (lazy `scan_ipc → filter → sink_ipc`) and scan only kept variants (VCF: workers re-apply the cyvcf2 filter via `chunk`; PGEN: workers read kept physical indices via `read_alleles_list`). The CLI keeps a thin `--skip-symbolic-alts` flag that constructs the filter on the source.

**Tech Stack:** Python, polars (lazy IPC), cyvcf2, pgenlib, joblib (loky/cloudpickle), vcfixture 0.6.0 (test fixtures), plink2 (PGEN test fixtures), pytest.

**Environment:** All commands run under pixi. Work in the existing worktree at `.claude/worktrees/pr-51` (branch `feat/skip-symbolic-alts`). Prefix every command with `pixi run`.

---

## Spec reference

`docs/superpowers/specs/2026-06-04-svar-filter-inheritance-design.md`

## File structure

- `genoray/_vcf.py` — remove flag + filter-composition; fix `_load_index` ALT order; restore auto-load condition.
- `genoray/_svar.py` — add `_write_filtered_index` helper; rework `from_vcf`, `from_pgen`, `_process_contig_vcf`, `_process_contig_pgen`.
- `genoray/_cli/__main__.py` — `--skip-symbolic-alts` constructs source filter for VCF and PGEN.
- `genoray/exprs.py` — fix `is_symbolic` docstring.
- `skills/genoray-api/SKILL.md` — remove kwarg, document filter inheritance.
- `tests/test_svar_filtering.py` — new test file (replaces `tests/test_skip_symbolic_alts.py`).

## Shared test fixture

Defined once at the top of `tests/test_svar_filtering.py` in Task 1 and reused by all later tasks. The fixture VCF has 4 records on `chr1`: SNV `A>T`@100, `<DEL>`@200, `<INS>`@300, insertion `G>GAT`@400. The two symbolic records (POS 200, 300) are the ones a symbolic filter drops, leaving POS [100, 400].

---

### Task 1: Fix `VCF._load_index` ALT ordering

**Why:** On-disk VCF `.gvi` stores `ALT` as comma-joined `Utf8`. `_load_index` currently applies `pl_filter` *before* splitting `ALT` to `list[str]`, so list-typed expressions (`is_symbolic`, `is_biallelic`) raise `InvalidOperationError: list.eval operation not supported for dtype str`. Split before filtering.

**Files:**
- Create: `tests/test_svar_filtering.py`
- Modify: `genoray/_vcf.py` (`_load_index`, lines ~1156-1210)

- [ ] **Step 1: Write the failing test (and shared fixture)**

Create `tests/test_svar_filtering.py`:

```python
"""Filter-inheritance tests for VCF/PGEN -> SVAR.

Replaces the old skip_symbolic_alts flag tests: symbolic filtering is now just
`pl_filter=~exprs.is_symbolic` (+ paired cyvcf2 `filter`), and SparseVar inherits
the source's filter.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from vcfixture import Seq, Sym, VcfBuilder, VcfVersion

from genoray import PGEN, VCF, SparseVar
from genoray import exprs as gexprs


def _not_symbolic(rec) -> bool:
    return not any(a.startswith("<") for a in rec.ALT)


def _mixed_vcf(tmp_path: Path) -> Path:
    """chr1: SNV A>T@100, <DEL>@200, <INS>@300, ins G>GAT@400."""
    b = (
        VcfBuilder(
            samples=["s1", "s2"],
            contigs=[("chr1", 1_000_000)],
            version=VcfVersion.V4_4,
        )
        .info("SVLEN")
        .info("SVCLAIM")
        .info("END")
        .fmt("GT")
    )
    b.record("chr1", 100, ref="A", alt=[Seq("T")], gt=["0|1", "1|1"])
    b.record(
        "chr1", 200, ref="A", alt=[Sym.deletion()], gt=["0|1", "0|0"],
        info={"SVLEN": [50], "SVCLAIM": ["D"], "END": [250]},
    )
    b.record(
        "chr1", 300, ref="C", alt=[Sym.insertion()], gt=["0|0", "0|1"],
        info={"SVLEN": [60]},
    )
    b.record("chr1", 400, ref="G", alt=[Seq("GAT")], gt=["1|1", "0|1"])
    return b.write(tmp_path / "mixed.vcf.gz", bgzip=True, index=True)


def test_vcf_load_index_list_typed_filter(tmp_path):
    """A list-typed pl_filter (is_symbolic) must evaluate on the VCF path."""
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(vcf_path, filter=_not_symbolic, pl_filter=~gexprs.is_symbolic)
    v._write_gvi_index()
    v._load_index()
    assert v._index is not None
    assert v._index.height == 2
    assert v._index["POS"].to_list() == [100, 400]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_filtering.py::test_vcf_load_index_list_typed_filter -v`
Expected: FAIL with `InvalidOperationError: list.eval operation not supported for dtype str`.

- [ ] **Step 3: Rewrite the `_load_index` body**

In `genoray/_vcf.py`, replace everything from the `logger.info("Loading genoray index.")` line through `self._index = index.collect()` (the block that currently does the conditional skip-split, filter, then a second schema check) with:

```python
        logger.info("Loading genoray index.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index = pl.scan_ipc(
                self._index_path(), row_index_name="index"
            ).with_columns(pl.col("CHROM").cast(pl.Enum(self.contigs)))

        # Normalize ALT (on-disk comma-Utf8) to list[str] BEFORE applying the
        # filter so the in-memory schema documented in genoray.exprs holds and
        # list-typed expressions (is_symbolic, is_biallelic) work on this path.
        schema = index.collect_schema()
        if schema["ALT"] == pl.Utf8:
            index = index.with_columns(pl.col("ALT").str.split(","))

        if self._pl_filter is not None:
            index = index.filter(self._pl_filter)

        if "ILEN" not in schema:
            index = index.with_columns(ILEN=ILEN)

        self._index = index.collect()

        return self
```

(`ILEN` is imported at module top via `from .exprs import ILEN` — confirm it is already imported; it is used elsewhere in this file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_svar_filtering.py::test_vcf_load_index_list_typed_filter -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add genoray/_vcf.py tests/test_svar_filtering.py
git commit -m "fix(vcf): split ALT to list before applying pl_filter in _load_index

List-typed expressions (is_symbolic, is_biallelic) previously errored on the
VCF index path because pl_filter was applied while ALT was still comma-Utf8."
```

---

### Task 2: Remove `skip_symbolic_alts` from `VCF`

**Why:** The flag is bloat — symbolic filtering is now expressible via `filter`/`pl_filter`. Remove the kwarg, attribute, filter-composition block, and the symbolic-only auto-load special case.

**Files:**
- Modify: `genoray/_vcf.py` (docstring ~223-243, attribute ~258-261, `__init__` ~298-365)
- Delete: `tests/test_skip_symbolic_alts.py`

- [ ] **Step 1: Delete the obsolete flag test file**

```bash
git rm tests/test_skip_symbolic_alts.py
```

- [ ] **Step 2: Remove the `skip_symbolic_alts` docstring block**

In the `VCF` class docstring, delete the entire `skip_symbolic_alts` parameter description (the paragraph beginning `skip_symbolic_alts` and its `.. note::` about expansion).

- [ ] **Step 3: Remove the attribute declaration**

Delete these two lines from the class body:

```python
    _skip_symbolic_alts: bool
    """Whether records with any symbolic ALT allele (e.g. ``<DEL>``) are skipped. See constructor."""
```

- [ ] **Step 4: Remove the kwarg and composition block in `__init__`**

Delete the `skip_symbolic_alts: bool = False,` parameter. Delete the entire block from `_symbolic_only_filter = False` through the end of the `if skip_symbolic_alts:` branch (the `_not_symbolic_cyvcf2` / `_combined_filter` definitions). Delete the line `self._skip_symbolic_alts = skip_symbolic_alts`.

- [ ] **Step 5: Restore the original auto-load condition**

Replace the `_safe_to_autoload` block:

```python
        # Auto-load the index when:
        # - no user filter is set ...
        _safe_to_autoload = self._filter is None or _symbolic_only_filter
        if with_gvi_index and self._valid_index() and _safe_to_autoload:
            self._load_index()
```

with:

```python
        if with_gvi_index and self._valid_index() and self._filter is None:
            self._load_index()
```

- [ ] **Step 6: Verify the flag is gone from the library core**

Run: `pixi run python -c "import genoray; help(genoray.VCF.__init__)" 2>/dev/null | grep -c skip_symbolic_alts`
Expected: `0`

Run: `grep -rn "skip_symbolic_alts" genoray/_vcf.py genoray/_svar.py`
Expected: only `genoray/_svar.py` matches remain (removed in Task 3); `_vcf.py` has none.

- [ ] **Step 7: Run the existing VCF + new filtering tests**

Run: `pixi run pytest tests/test_vcf.py tests/test_svar_filtering.py -q`
Expected: PASS (no collection error from the deleted test file).

- [ ] **Step 8: Commit**

```bash
git add genoray/_vcf.py tests/test_skip_symbolic_alts.py
git commit -m "refactor(vcf): remove skip_symbolic_alts flag

Symbolic filtering is now expressed via the existing filter/pl_filter API
(pl_filter=~exprs.is_symbolic). One way to do things."
```

---

### Task 3: `from_vcf` inherits the source filter

**Why:** `from_vcf` `shutil.copy`s the unfiltered index and the workers re-open the VCF with no filter, so any filter is silently ignored. Write a filtered index and pass the filter to workers.

**Files:**
- Modify: `genoray/_svar.py` (`from_vcf` ~879-1003 and its joblib dispatch; `_process_contig_vcf` ~1678; add module-level helper)
- Modify: `tests/test_svar_filtering.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar_filtering.py`:

```python
def test_from_vcf_inherits_symbolic_filter(tmp_path):
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(vcf_path, filter=_not_symbolic, pl_filter=~gexprs.is_symbolic)
    out = tmp_path / "out.svar"
    SparseVar.from_vcf(out, v, max_mem="1g", overwrite=True)

    sv = SparseVar(out)
    assert sv.n_variants == 2
    assert sv._index["POS"].to_list() == [100, 400]
    # Genotype scan and index agree: reading the contig yields 2 variants.
    genos = sv.read_ranges("chr1", 0, 1_000_000)
    assert genos.shape[-1] == 2


def test_from_vcf_inherits_general_filter(tmp_path):
    """A non-symbolic filter (is_snp) is also honored, proving the path is general."""
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(
        vcf_path,
        filter=lambda rec: len(rec.REF) == 1 and all(len(a) == 1 for a in rec.ALT),
        pl_filter=gexprs.is_snp,
    )
    out = tmp_path / "snp.svar"
    SparseVar.from_vcf(out, v, max_mem="1g", overwrite=True)
    sv = SparseVar(out)
    # Only the SNV A>T@100 is a pure SNP (200/300 symbolic, 400 indel).
    assert sv.n_variants == 1
    assert sv._index["POS"].to_list() == [100]


def test_from_vcf_no_filter_keeps_all(tmp_path):
    """Back-compat: no filter -> all records written."""
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(vcf_path)
    out = tmp_path / "all.svar"
    SparseVar.from_vcf(out, v, max_mem="1g", overwrite=True)
    sv = SparseVar(out)
    assert sv.n_variants == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_svar_filtering.py -k from_vcf -v`
Expected: `test_from_vcf_inherits_symbolic_filter` and `_general_filter` FAIL (assert 4 != 2 / != 1); `_no_filter_keeps_all` PASSES.

- [ ] **Step 3: Add the `_write_filtered_index` helper**

Add at module level in `genoray/_svar.py` (near the other module-level helpers, e.g. above `_process_contig_vcf`):

```python
def _write_filtered_index(src: Path, dst: Path, pl_filter: pl.Expr | None) -> None:
    """Stream a (possibly filtered) genoray index from ``src`` to ``dst``.

    When ``pl_filter`` is None this is byte-equivalent to copying. Otherwise the
    filter is applied lazily; ALT is normalized to list[str] for the filter and
    re-joined to the on-disk comma-Utf8 form so the SVAR index format is
    unchanged.
    """
    if pl_filter is None:
        shutil.copy(src, dst)
        return
    lf = pl.scan_ipc(src)
    alt_is_utf8 = lf.collect_schema()["ALT"] == pl.Utf8
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").str.split(","))
    lf = lf.filter(pl_filter)
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").list.join(","))
    lf.sink_ipc(dst, compression="zstd")
```

- [ ] **Step 4: Rewrite the `from_vcf` index write + remove the flag**

Remove the `skip_symbolic_alts: bool | None = None,` parameter and its docstring block. Replace the whole block from `# Resolve the effective skip flag:` through the `else: shutil.copy(vcf._index_path(), cls._index_path(out))` (the effective_skip / rebuild-sibling / re-materialize logic) with:

```python
        if not vcf._index_path().exists():
            logger.info("Genoray VCF index not found, creating index.")
            vcf._write_gvi_index()
        _write_filtered_index(vcf._index_path(), cls._index_path(out), vcf._pl_filter)
```

- [ ] **Step 5: Pass the filter to the VCF workers**

In the `from_vcf` joblib dispatch loop, change the `joblib.delayed(_process_contig_vcf)(...)` call to add the filter args and drop `skip_symbolic_alts=effective_skip`:

```python
                task = joblib.delayed(_process_contig_vcf)(
                    vcf.path,
                    dosage_field=vcf.dosage_field if with_dosages else None,
                    max_mem=job_mem,
                    contig=c,
                    chunk_dir=chunk_dir,
                    chunk_idx=chunk_idx,
                    filter=vcf._filter,
                    pl_filter=vcf._pl_filter,
                )
```

- [ ] **Step 6: Update `_process_contig_vcf` to apply the filter**

Change its signature and the `VCF(...)` construction:

```python
def _process_contig_vcf(
    path: str | Path,
    dosage_field: str | None,
    max_mem: int | str,
    contig: str,
    chunk_dir: Path,
    chunk_idx: int,
    filter=None,
    pl_filter=None,
) -> tuple[int, int]:
    vcf = VCF(
        path,
        filter=filter,
        pl_filter=pl_filter,
        dosage_field=dosage_field,
        with_gvi_index=False,
    )
```

(The source VCF guarantees `filter`/`pl_filter` are paired, satisfying `VCF.__init__`. `chunk()` applies the cyvcf2 `filter` inline, so the scan matches the filtered index. Filters reach workers via cloudpickle.)

- [ ] **Step 7: Run the from_vcf tests**

Run: `pixi run pytest tests/test_svar_filtering.py -k from_vcf -v`
Expected: all three PASS.

- [ ] **Step 8: Run the broader SVAR suite for regressions**

Run: `pixi run pytest tests/test_svar.py tests/test_svar_write_view.py tests/test_svar_internals.py -q -k "not pgen"`
Expected: PASS (VCF-backed SVAR paths unaffected; `pgen` cases may error if plink2 absent — excluded here).

- [ ] **Step 9: Commit**

```bash
git add genoray/_svar.py tests/test_svar_filtering.py
git commit -m "fix(svar): from_vcf inherits and applies the source VCF filter

Writes a filtered index (lazy scan->filter->sink) and re-applies the cyvcf2
filter in the per-contig workers. Previously from_vcf silently ignored the
VCF's filter. Adds _write_filtered_index helper."
```

---

### Task 4: `from_pgen` inherits the source filter

**Why:** Same gap on the PGEN path. plink2 carries `<DEL>` into the `.pvar` verbatim, so PGEN-backed SVARs need symbolic filtering too. Write a filtered index and read only kept physical variant indices via `read_alleles_list`.

**Files:**
- Modify: `genoray/_svar.py` (`from_pgen` ~1050-1161; `_process_contig_pgen` ~1727-1809)
- Modify: `tests/test_svar_filtering.py`

- [ ] **Step 1: Write the failing tests (plink2-guarded)**

Append to `tests/test_svar_filtering.py`:

```python
def _mixed_pgen(tmp_path: Path) -> Path:
    """Convert the mixed VCF to PGEN via plink2 (symbolic alleles carried verbatim)."""
    if shutil.which("plink2") is None:
        pytest.skip("plink2 not available")
    vcf_path = _mixed_vcf(tmp_path)
    prefix = tmp_path / "mixed"
    subprocess.run(
        ["plink2", "--vcf", str(vcf_path), "--make-pgen",
         "--out", str(prefix), "--allow-extra-chr"],
        check=True, capture_output=True,
    )
    return prefix.with_suffix(".pgen")


def test_from_pgen_inherits_symbolic_filter(tmp_path):
    pgen_path = _mixed_pgen(tmp_path)
    pgen = PGEN(pgen_path, filter=~gexprs.is_symbolic)
    out = tmp_path / "pg.svar"
    SparseVar.from_pgen(out, pgen, max_mem="1g", overwrite=True)

    sv = SparseVar(out)
    assert sv.n_variants == 2
    # symbolic POS 200/300 dropped; precise 100/400 kept
    assert set(sv._index["POS"].to_list()) == {100, 400}


def test_from_pgen_no_filter_keeps_all(tmp_path):
    pgen_path = _mixed_pgen(tmp_path)
    pgen = PGEN(pgen_path)
    out = tmp_path / "pg_all.svar"
    SparseVar.from_pgen(out, pgen, max_mem="1g", overwrite=True)
    sv = SparseVar(out)
    assert sv.n_variants == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_svar_filtering.py -k from_pgen -v`
Expected (plink2 present): `test_from_pgen_inherits_symbolic_filter` FAILS (assert 4 != 2); `_no_filter_keeps_all` PASSES. (If plink2 absent: both SKIP — note this and proceed; CI/Linux has plink2.)

- [ ] **Step 3: Rewrite the `from_pgen` index write + per-contig kept indices**

In `from_pgen`, replace `shutil.copy(pgen._index_path(), cls._index_path(out))` with:

```python
        _write_filtered_index(pgen._index_path(), cls._index_path(out), pgen._filter)
```

Then delete the `offsets = np.array([0] + [pgen._c_max_idxs[c] + 1 ...])` line. Before the `pgen._free_index()` call, build the per-contig kept physical indices from the (already filter-applied) `pgen._index`:

```python
        keep_by_contig = {
            chrom: np.asarray(idxs, dtype=np.uint32)
            for chrom, idxs in (
                pgen._index.group_by("CHROM", maintain_order=True)
                .agg(pl.col("index"))
                .iter_rows()
            )
        }
```

- [ ] **Step 4: Rewrite the `from_pgen` joblib dispatch loop**

Replace the `for chunk_idx, (start, end) in enumerate(sliding_window_view(offsets, 2)):` loop with:

```python
            tasks = []
            for chunk_idx, c in enumerate(contigs):
                keep_idxs = keep_by_contig.get(c)
                if keep_idxs is None or len(keep_idxs) == 0:
                    continue
                task = joblib.delayed(_process_contig_pgen)(
                    geno_path=pgen.geno_path,
                    dosage_path=pgen.dosage_path if with_dosages else None,
                    max_mem=job_mem,
                    keep_idxs=keep_idxs,
                    mem_per_var=mem_per_var,
                    n_samples=pgen.n_samples,
                    ploidy=pgen.ploidy,
                    chunk_dir=contig_dir,
                    chunk_idx=chunk_idx,
                )
                tasks.append(task)
```

(`keep_by_contig` is built before `pgen._free_index()`, which still runs after the loop.)

- [ ] **Step 5: Rewrite `_process_contig_pgen` to read kept indices**

Replace the function with:

```python
def _process_contig_pgen(
    geno_path: str | Path,
    dosage_path: str | Path | None,
    max_mem: int,
    keep_idxs: np.ndarray,
    mem_per_var: int,
    n_samples: int,
    ploidy: int,
    chunk_dir: Path,
    chunk_idx: int,
) -> tuple[int, int]:
    geno_reader = PgenReader(bytes(Path(geno_path)), n_samples)
    dose_reader = (
        PgenReader(bytes(Path(dosage_path))) if dosage_path is not None else None
    )

    keep_idxs = np.ascontiguousarray(keep_idxs, dtype=np.uint32)
    n_total = int(len(keep_idxs))
    vars_per_chunk = min(max_mem // mem_per_var, n_total) if n_total else 0
    if n_total and vars_per_chunk == 0:
        raise ValueError(
            f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
            + f" Memory per variant: {format_memory(mem_per_var)}."
        )

    contig_dir = chunk_dir / f"c{chunk_idx}"
    contig_dir.mkdir(parents=True, exist_ok=True)

    total_vars = 0
    n_chunks = 0
    for i, c0 in enumerate(range(0, n_total, vars_per_chunk) if n_total else []):
        idxs = keep_idxs[c0 : c0 + vars_per_chunk]
        n_vars = int(len(idxs))
        if n_vars == 0:
            continue
        n_chunks += 1

        out_path = contig_dir / str(i)
        out_path.mkdir(parents=True, exist_ok=True)

        # Read genotypes for exactly the kept variant indices.
        # (v, s*p)
        genos = np.empty((n_vars, n_samples * ploidy), dtype=np.int32)
        geno_reader.read_alleles_list(idxs, genos)
        genos = genos.astype(np.int8)
        # (v, s, p) -> (s, p, v)
        genos = genos.reshape(n_vars, n_samples, ploidy).transpose(1, 2, 0)
        genos[genos == -9] = -1

        dosages = None
        if dose_reader is not None:
            dosages = np.empty((n_vars, n_samples), dtype=np.float32)
            dose_reader.read_dosages_list(idxs, dosages)
            dosages = dosages.transpose(1, 0)
            dosages[dosages == -9] = np.nan

        var_idxs = np.arange(total_vars, total_vars + n_vars, dtype=np.int32)
        if dosages is not None:
            sp_genos, sp_dosages = dense2sparse(genos.astype(np.int8), var_idxs, dosages)
            _write_genos(out_path, sp_genos)
            _write_dosages(out_path, sp_dosages.data)
        else:
            sp_genos = dense2sparse(genos.astype(np.int8), var_idxs)
            _write_genos(out_path, sp_genos)

        total_vars += n_vars
    return total_vars, n_chunks
```

- [ ] **Step 6: Run the from_pgen tests**

Run: `pixi run pytest tests/test_svar_filtering.py -k from_pgen -v`
Expected (plink2 present): both PASS. (If plink2 absent: SKIP — verify on a Linux box / CI.)

- [ ] **Step 7: Run the existing PGEN SVAR test**

Run: `pixi run pytest tests/test_svar.py -k pgen -q`
Expected: PASS where plink2 is available (no-filter PGEN path unchanged in behavior).

- [ ] **Step 8: Commit**

```bash
git add genoray/_svar.py tests/test_svar_filtering.py
git commit -m "fix(svar): from_pgen inherits and applies the source PGEN filter

Writes a filtered index and reads only kept physical variant indices via
read_alleles_list / read_dosages_list. Previously from_pgen ignored the
PGEN's filter."
```

---

### Task 5: CLI `--skip-symbolic-alts` constructs the source filter

**Why:** The CLI has no expression API; the flag is its stand-in. It now constructs the filter on a VCF *or* PGEN source and lets inheritance carry it into the SVAR.

**Files:**
- Modify: `genoray/_cli/__main__.py` (`write`, ~45-113)
- Modify: `tests/test_svar_filtering.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar_filtering.py`:

```python
def test_cli_write_skip_symbolic_vcf(tmp_path):
    from genoray._cli.__main__ import write as cli_write

    vcf_path = _mixed_vcf(tmp_path)
    out = tmp_path / "cli.svar"
    cli_write(vcf_path, out, max_mem="1g", overwrite=True, skip_symbolic_alts=True)
    sv = SparseVar(out)
    assert sv.n_variants == 2
    assert sv._index["POS"].to_list() == [100, 400]


def test_cli_write_skip_symbolic_pgen(tmp_path):
    from genoray._cli.__main__ import write as cli_write

    pgen_path = _mixed_pgen(tmp_path)
    out = tmp_path / "cli_pg.svar"
    cli_write(pgen_path, out, max_mem="1g", overwrite=True, skip_symbolic_alts=True)
    sv = SparseVar(out)
    assert sv.n_variants == 2
    assert set(sv._index["POS"].to_list()) == {100, 400}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_svar_filtering.py -k cli -v`
Expected: FAIL — the current CLI passes `skip_symbolic_alts` to `VCF(...)`, which no longer accepts it (`TypeError: unexpected keyword argument`), and the PGEN branch ignores the flag.

- [ ] **Step 3: Update the `write` command**

Replace the VCF and PGEN branches in `write` with:

```python
    from genoray import exprs

    if file_type == "vcf":
        if dosages is not None and Path(dosages).exists():
            raise ValueError(
                "The `dosages` argument appears to be a path to an existing file, but VCF requires a FORMAT field name."
            )

        if skip_symbolic_alts:
            vcf = VCF(
                source,
                dosage_field=dosages,
                filter=lambda rec: not any(a.startswith("<") for a in rec.ALT),
                pl_filter=~exprs.is_symbolic,
            )
        else:
            vcf = VCF(source, dosage_field=dosages)
        SparseVar.from_vcf(
            out, vcf, max_mem, overwrite, with_dosages=with_dosages, n_jobs=threads
        )
    elif file_type == "pgen":
        pgen = PGEN(
            source,
            dosage_path=dosages,
            filter=(~exprs.is_symbolic) if skip_symbolic_alts else None,
        )
        SparseVar.from_pgen(
            out, pgen, max_mem, overwrite, with_dosages=with_dosages, n_jobs=threads
        )
```

Update the `skip_symbolic_alts` docstring: remove "VCF only." and note it applies to VCF and PGEN sources.

- [ ] **Step 4: Run the CLI tests**

Run: `pixi run pytest tests/test_svar_filtering.py -k cli -v`
Expected: PASS (PGEN test SKIPs if plink2 absent).

- [ ] **Step 5: Verify `--help` still renders sub-second and shows the flag**

Run: `pixi run genoray write --help 2>&1 | grep -i symbolic`
Expected: the `--skip-symbolic-alts` flag appears.

- [ ] **Step 6: Commit**

```bash
git add genoray/_cli/__main__.py tests/test_svar_filtering.py
git commit -m "feat(cli): --skip-symbolic-alts builds source filter for VCF and PGEN

The flag now constructs filter/pl_filter on the source and relies on SparseVar
filter inheritance; supports both VCF and PGEN sources."
```

---

### Task 6: Docs — `is_symbolic` docstring + SKILL.md

**Why:** CLAUDE.md requires `skills/genoray-api/SKILL.md` to track public-API changes. Remove the deleted kwarg, keep `is_symbolic`, and document filter inheritance.

**Files:**
- Modify: `genoray/exprs.py` (`is_symbolic` docstring ~44-61)
- Modify: `skills/genoray-api/SKILL.md`

- [ ] **Step 1: Rewrite the `is_symbolic` docstring**

Replace the `is_symbolic` docstring body (the part referencing `skip_symbolic_alts`) with:

```python
is_symbolic = pl.col("ALT").list.eval(pl.element().str.starts_with("<")).list.any()
"""True if any ALT allele is a symbolic allele (e.g. :code:`<DEL>`, :code:`<INS>`,
:code:`<DUP>`, :code:`<INV>`, :code:`<CNV>`, :code:`<BND>` — anything matching ``<…>``
per the VCF 4.x spec).

Symbolic ALTs are placeholders for structural variants whose exact replacement
nucleotides are unknown. Downstream haplotype injection (e.g. via
``genvarloader``) cannot expand them — the literal ``<DEL>`` ASCII bytes end up
in personalized DNA buffers and become non-canonical bytes for translators.

To drop symbolic records, pass this as a filter. For PGEN, the single ``filter``
expression suffices::

    pgen = genoray.PGEN("file.pgen", filter=~genoray.exprs.is_symbolic)

For VCF, pair it with the equivalent cyvcf2 ``filter`` (both are required)::

    vcf = genoray.VCF(
        "file.vcf.gz",
        filter=lambda rec: not any(a.startswith("<") for a in rec.ALT),
        pl_filter=~genoray.exprs.is_symbolic,
    )

``SparseVar.from_vcf`` / ``from_pgen`` inherit the source's filter, so the SVAR
is filtered to match.
"""
```

- [ ] **Step 2: Update SKILL.md**

In `skills/genoray-api/SKILL.md`:
- In the VCF constructor snippet, remove the `skip_symbolic_alts=True,` line and add a comment line showing symbolic filtering via the filter pair, e.g.:
  `    pl_filter=~genoray.exprs.is_symbolic,  # drop <DEL>/<INS>/... ; pair with matching `filter``
- In the `SparseVar.from_vcf` build snippet, remove `skip_symbolic_alts=True)  # opt-in...` and the trailing comment; restore the prior call signature.
- Add a sentence near the SparseVar section: `SparseVar.from_vcf / from_pgen inherit and apply the source's filter — filter the VCF/PGEN to filter the SVAR.`
- Keep the `is_symbolic` bullet in the `genoray.exprs` list (5 entries). Leave the "currently 5" count as-is.

- [ ] **Step 3: Verify no stale `skip_symbolic_alts` references remain (except the CLI flag)**

Run: `grep -rn "skip_symbolic_alts" genoray/ skills/`
Expected: matches only in `genoray/_cli/__main__.py` (the CLI flag name + docstring).

- [ ] **Step 4: Commit**

```bash
git add genoray/exprs.py skills/genoray-api/SKILL.md
git commit -m "docs: update is_symbolic docstring and SKILL.md for filter inheritance

Removes references to the deleted skip_symbolic_alts flag; documents that
SparseVar inherits the source filter."
```

---

### Task 7: Full suite + final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full non-network suite**

Run: `pixi run pytest tests -q -m "not network"`
Expected: PASS. PGEN-dependent tests SKIP only if plink2 is unavailable locally (run on Linux/CI to exercise them). Report any failures with output — do not claim success without seeing the summary line.

- [ ] **Step 2: Lint/format**

Run: `pixi run ruff check genoray tests && pixi run ruff format --check genoray tests`
Expected: clean (run `pixi run ruff format genoray tests` and amend if needed).

- [ ] **Step 3: Confirm the design's behavior-change note holds**

Manually confirm `test_from_vcf_inherits_general_filter` and the symbolic tests demonstrate the bugfix (filters now honored). No deprecation shim was added (treated as bugfix per spec).

- [ ] **Step 4: Update the PR description**

Rewrite the PR #51 body to reflect the new approach (delete `skip_symbolic_alts`; filter inheritance; CLI flag for both backends; `_load_index` fix). Do this only when the user asks to push/update the PR.

---

## Self-review

**Spec coverage:**
- Delete `skip_symbolic_alts` everywhere → Tasks 2 (VCF), 3 (from_vcf), 5 (CLI keeps only the flag name). ✓
- `_load_index` ALT-order fix → Task 1. ✓
- `from_vcf` inherit filter (lazy sink + worker filter) → Task 3. ✓
- `from_pgen` inherit filter (lazy sink + `read_alleles_list`) → Task 4. ✓
- CLI flag for VCF + PGEN → Task 5. ✓
- `is_symbolic` docstring + SKILL.md → Task 6. ✓
- Tests: load-fix regression (T1), VCF general+symbolic+back-compat (T3), PGEN symbolic+back-compat (T4), CLI VCF+PGEN (T5), full suite (T7). ✓
- Behavior-change-as-bugfix (no deprecation) → commit messages + T7 Step 3. ✓
- `is_symbolic` unit test: covered implicitly by `test_vcf_load_index_list_typed_filter` (T1) exercising the expression on the index; an isolated expr unit test is redundant given that coverage. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type/name consistency:** `_write_filtered_index(src, dst, pl_filter)` defined in T3, reused in T4. `_process_contig_pgen(..., keep_idxs, ...)` signature in T4 matches its dispatch call. `_process_contig_vcf(..., filter, pl_filter)` signature in T3 matches its dispatch call. `read_alleles_list(idxs, out)` / `read_dosages_list(idxs, out)` match verified pgenlib signatures. ✓

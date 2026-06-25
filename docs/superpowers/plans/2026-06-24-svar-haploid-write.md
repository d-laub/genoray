# SparseVar haploid (OR-collapse) write option — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a write-time `haploid` option to `SparseVar.from_vcf`/`from_pgen` (and a `--haploid` CLI flag) that OR-collapses the ploidy axis into a single haploid call per sample and records `ploidy=1` in the SparseVar metadata.

**Architecture:** The dense per-contig writer workers OR-collapse `(samples, ploidy, variants)` to `(samples, 1, variants)` via `(genos == 1).any(axis=1, keepdims=True)` before `dense2sparse`. The write entry points set the output ploidy to 1 and thread it into metadata, the concat shape, and the AF recompute. All read/view/annotate paths already honor `metadata.ploidy`, so they are unchanged — only verified.

**Tech Stack:** Python, NumPy, polars, joblib, Numba, `seqpro.rag.Ragged`, cyclopts (CLI), pytest / pytest-cases. Environment via Pixi (`pixi run pytest ...`).

## Global Constraints

- Conventional Commits for every commit (`feat:`, `test:`, `docs:`, etc.). Verbatim from project `CLAUDE.md`.
- Any change to a public name (anything reachable from `import genoray` without an underscore) MUST update `skills/genoray-api/SKILL.md` in the same PR. `from_vcf`/`from_pgen`/the CLI are public — Task 5 covers this.
- Coordinate convention: ranges 0-based half-open `[start, end)`; missing genotype `-1`; `svar.index.POS` is 1-based.
- OR semantics: collapse is over `genos == 1` (exactly the predicate `dense2sparse` already uses), so the haploid call set equals the union of the per-haplotype call sets by construction. A slot ALT on one haplotype and missing (`-1`) on the other → present; ref+missing → absent; missing+missing → absent (no entry).
- The collapse is whole-file and unconditional when `haploid=True`. No per-sample/per-region collapse, no phasing detection.
- All test commands run inside Pixi: prefix with `pixi run`.

---

### Task 1: `from_vcf` haploid collapse + wiring

**Files:**
- Modify: `genoray/_svar.py` — `_process_contig_vcf` (around `genoray/_svar.py:2250-2324`) and `from_vcf` (around `genoray/_svar.py:918-1109`)
- Test: `tests/test_svar_haploid.py` (create)

**Interfaces:**
- Consumes: existing `dense2sparse(genos, var_idxs[, dosages])` keyed on `keep = genos == 1`; `_subset_var_idxs_and_recompute_af(out, n_total, n_out, ploidy, with_dosages)` which returns `(survivor_rows, af)` with `af = mac / (n_out * ploidy)`.
- Produces: `SparseVar.from_vcf(out, vcf, max_mem, overwrite=False, with_dosages=False, n_jobs=-1, *, regions=None, samples=None, merge_overlapping=False, regions_overlap="pos", haploid=False)`. When `haploid=True`, output `metadata.ploidy == 1` and stored genos are the per-sample OR across haplotypes. `_process_contig_vcf(..., haploid: bool = False)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_svar_haploid.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from genoray import VCF, SparseVar

ddir = Path(__file__).parent / "data"
VCF_PATH = ddir / "biallelic.vcf.gz"  # 2 samples (sample1, sample2), contigs chr1/chr2/chr3


def _haploid_call_sets(sv: SparseVar) -> dict[tuple[str, int], set[int]]:
    """Map (contig, sample_idx) -> set of global variant indices present (ploidy=1)."""
    out: dict[tuple[str, int], set[int]] = {}
    for c in sv.contigs:
        rag = sv.read_ranges(c)  # (1, n_samples, 1, ~v)
        for i in range(sv.n_samples):
            out[(c, i)] = set(rag[0, i, 0].to_numpy().tolist())
    return out


def _diploid_union_call_sets(sv: SparseVar) -> dict[tuple[str, int], set[int]]:
    """Map (contig, sample_idx) -> union of both haplotypes' variant indices (ploidy=2)."""
    out: dict[tuple[str, int], set[int]] = {}
    for c in sv.contigs:
        rag = sv.read_ranges(c)  # (1, n_samples, 2, ~v)
        for i in range(sv.n_samples):
            hap0 = set(rag[0, i, 0].to_numpy().tolist())
            hap1 = set(rag[0, i, 1].to_numpy().tolist())
            out[(c, i)] = hap0 | hap1
    return out


def test_from_vcf_haploid_metadata_and_or(tmp_path: Path):
    dip = tmp_path / "dip.svar"
    hap = tmp_path / "hap.svar"
    SparseVar.from_vcf(dip, VCF(VCF_PATH), max_mem="1g", overwrite=True)
    SparseVar.from_vcf(hap, VCF(VCF_PATH), max_mem="1g", overwrite=True, haploid=True)

    sv_dip = SparseVar(dip)
    sv_hap = SparseVar(hap)

    # metadata + shape
    assert sv_hap.ploidy == 1
    assert sv_hap.genos.shape[1] == 1
    # same variant set as the diploid build (no variants gained/lost by collapse)
    assert sv_hap.n_variants == sv_dip.n_variants

    # OR invariant: haploid call set == union of the two diploid haplotype sets
    assert _haploid_call_sets(sv_hap) == _diploid_union_call_sets(sv_dip)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_haploid.py::test_from_vcf_haploid_metadata_and_or -v`
Expected: FAIL — `from_vcf()` got an unexpected keyword argument `haploid`.

- [ ] **Step 3: Add the `haploid` param and collapse to `_process_contig_vcf`**

In `genoray/_svar.py`, change the `_process_contig_vcf` signature to add `haploid: bool = False` (place it after `keep_local`):

```python
def _process_contig_vcf(
    path: str | Path,
    dosage_field: str | None,
    max_mem: int | str,
    contig: str,
    chunk_dir: Path,
    chunk_idx: int,
    cyvcf2_filter: Callable[..., bool] | None = None,
    pl_filter: pl.Expr | None = None,
    caller_samples: list[str] | None = None,
    keep_local: np.ndarray | None = None,
    haploid: bool = False,
) -> tuple[int, int]:
```

Inside the chunk loop, immediately after the `keep_sorted` slice block and before `n_vars = genos.shape[-1]`, insert the collapse (replace the existing `n_vars = genos.shape[-1]` line context):

```python
        else:
            contig_local_pos += n_in

        if haploid:
            # OR across haplotypes -> single haploid slot. Uses `== 1`, the same
            # predicate dense2sparse keys on, so the haploid call set equals the
            # union of the per-haplotype call sets. Dosages keep their (s, v)
            # shape and broadcast against the collapsed genos in dense2sparse.
            genos = (genos == 1).any(axis=-2, keepdims=True).astype(np.int8)

        n_vars = genos.shape[-1]
```

- [ ] **Step 4: Add the `haploid` param and wiring to `from_vcf`**

Change the `from_vcf` signature to add `haploid: bool = False` in the keyword-only block (after `regions_overlap`):

```python
        regions_overlap: Literal["pos", "record", "variant"] = "pos",
        haploid: bool = False,
    ):
```

Add to the docstring's Parameters section:

```
        haploid
            If ``True``, OR-collapse the ploidy axis into a single haploid call
            per sample (a variant present on any haplotype becomes one call) and
            record ``ploidy=1`` in the output metadata. Intended for unphased
            somatic data. Default ``False``.
```

In the metadata write (around `genoray/_svar.py:1039-1047`), compute the output ploidy once just before it and use it:

```python
        out_ploidy = 1 if haploid else vcf.ploidy

        with open(out / "metadata.json", "w") as f:
            json_str = SparseVarMetadata(
                version=CURRENT_VERSION,
                contigs=contigs,
                samples=caller_samples,
                ploidy=out_ploidy,
                fields={"dosages": np.dtype(DOSAGE_TYPE).name} if with_dosages else {},
            ).model_dump_json()
            f.write(json_str)
```

Change the concat shape (was `shape = (n_out, vcf.ploidy)` around `genoray/_svar.py:1065`):

```python
            shape = (n_out, out_ploidy)
```

Add `haploid=haploid` to the `_process_contig_vcf` delayed call (in the task loop around `genoray/_svar.py:1068-1079`), after `keep_local=...`:

```python
                    keep_local=keep_local_by_contig.get(c),
                    haploid=haploid,
                )
```

Change the `_subset_var_idxs_and_recompute_af` call (around `genoray/_svar.py:1095-1101`) to pass `out_ploidy`:

```python
                survivors, af = _subset_var_idxs_and_recompute_af(
                    out,
                    n_total=len(kept_rows),
                    n_out=n_out,
                    ploidy=out_ploidy,
                    with_dosages=with_dosages,
                )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pixi run pytest tests/test_svar_haploid.py::test_from_vcf_haploid_metadata_and_or -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add genoray/_svar.py tests/test_svar_haploid.py
git commit -m "feat: add haploid OR-collapse option to SparseVar.from_vcf"
```

---

### Task 2: `from_vcf` haploid — dosages and sample-subset AF

**Files:**
- Test: `tests/test_svar_haploid.py` (extend)

**Interfaces:**
- Consumes: `SparseVar.from_vcf(..., with_dosages=True, samples=..., haploid=True)` from Task 1; `biallelic.vcf.gz` has a `DS` FORMAT field (Number=A) and samples `sample1`, `sample2`.
- Produces: no new code — this task verifies Task 1's wiring for dosages and AF. If a test fails, the fix lives in Task 1's code paths.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar_haploid.py`:

```python
def _dose_pairs(sv: SparseVar) -> dict[int, set[tuple[int, float]]]:
    """Map sample_idx -> set of (variant_idx, rounded dosage) over all contigs."""
    out: dict[int, set[tuple[int, float]]] = {i: set() for i in range(sv.n_samples)}
    for c in sv.contigs:
        rag = sv.read_ranges(c)  # record array: .genos, .dosages
        for i in range(sv.n_samples):
            for p in range(sv.ploidy):
                vi = rag.genos[0, i, p].to_numpy()
                ds = rag.dosages[0, i, p].to_numpy()
                for v, d in zip(vi.tolist(), ds.tolist()):
                    out[i].add((int(v), round(float(d), 4)))
    return out


def test_from_vcf_haploid_with_dosages(tmp_path: Path):
    dip = tmp_path / "dip.svar"
    hap = tmp_path / "hap.svar"
    SparseVar.from_vcf(dip, VCF(VCF_PATH, dosage_field="DS"), max_mem="1g",
                       overwrite=True, with_dosages=True)
    SparseVar.from_vcf(hap, VCF(VCF_PATH, dosage_field="DS"), max_mem="1g",
                       overwrite=True, with_dosages=True, haploid=True)

    sv_dip = SparseVar(dip, fields=["dosages"])
    sv_hap = SparseVar(hap, fields=["dosages"])

    # genos and dosages share offsets, so equal entry counts per build
    rag = sv_hap.read_ranges(sv_hap.contigs[0])
    assert rag.genos.data.shape == rag.dosages.data.shape
    assert rag.dosages.data.dtype == np.float32

    # every (variant, dosage) pair present in the haploid build also exists in the
    # diploid build for that sample (dosage is per-(sample,variant), so collapse
    # preserves the value; a hom-ALT call simply stops being double-stored).
    dip_pairs = _dose_pairs(sv_dip)
    hap_pairs = _dose_pairs(sv_hap)
    for i in range(sv_hap.n_samples):
        assert hap_pairs[i].issubset(dip_pairs[i])


def test_from_vcf_haploid_sample_subset_af(tmp_path: Path):
    hap = tmp_path / "hap_sub.svar"
    SparseVar.from_vcf(hap, VCF(VCF_PATH), max_mem="1g", overwrite=True,
                       haploid=True, samples=["sample1"])
    sv = SparseVar(hap, attrs="AF")
    assert sv.ploidy == 1
    assert list(sv.available_samples) == ["sample1"]
    # haploid AF denominator is n_out * 1 = 1 survivor sample, so every surviving
    # variant (MAC>0) has AF in (0, 1]; with one haploid sample AF is exactly 1.0.
    afs = sv.index["AF"].to_numpy()
    assert afs.size == sv.n_variants
    assert np.all(afs > 0.0)
    assert np.all(afs <= 1.0)
```

- [ ] **Step 2: Run tests**

Run: `pixi run pytest tests/test_svar_haploid.py -k "dosages or sample_subset_af" -v`
Expected: PASS (Task 1 already wired `out_ploidy` into the AF recompute and dosages need no special handling). If FAIL, fix the corresponding wiring in `from_vcf` / `_process_contig_vcf` from Task 1, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar_haploid.py
git commit -m "test: cover haploid from_vcf dosages and sample-subset AF"
```

---

### Task 3: `from_pgen` haploid collapse + wiring

**Files:**
- Modify: `genoray/_svar.py` — `_process_contig_pgen` (around `genoray/_svar.py:2327-2416`) and `from_pgen` (around `genoray/_svar.py:1111-1316`)
- Test: `tests/test_svar_haploid.py` (extend)

**Interfaces:**
- Consumes: `_process_contig_pgen(..., ploidy, ..., sample_subset=None)` where `ploidy` is the NATIVE read ploidy used to reshape pgenlib output — it must stay at the source ploidy for reading; the new `haploid` flag controls only the post-read collapse.
- Produces: `SparseVar.from_pgen(out, pgen, max_mem, overwrite=False, with_dosages=False, n_jobs=-1, *, regions=None, samples=None, merge_overlapping=False, regions_overlap="pos", haploid=False)`. `_process_contig_pgen(..., haploid: bool = False)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar_haploid.py`:

```python
import shutil
import subprocess

import pytest

from genoray import PGEN

VCF_FOR_PGEN = str(ddir / "biallelic.vcf")


def _vcf_to_pgen(tmp_path: Path) -> Path:
    if shutil.which("plink2") is None:
        pytest.skip("plink2 not available")
    prefix = tmp_path / "conv"
    subprocess.run(
        [
            "plink2", "--vcf", VCF_FOR_PGEN, "--make-pgen", "--out", str(prefix),
            "--allow-extra-chr", "--vcf-half-call", "haploid",
        ],
        check=True, capture_output=True,
    )
    return prefix.with_suffix(".pgen")


def test_from_pgen_haploid_metadata_and_or(tmp_path: Path):
    pgen_path = _vcf_to_pgen(tmp_path)
    dip = tmp_path / "dip.svar"
    hap = tmp_path / "hap.svar"
    SparseVar.from_pgen(dip, PGEN(pgen_path), max_mem="1g", overwrite=True)
    SparseVar.from_pgen(hap, PGEN(pgen_path), max_mem="1g", overwrite=True, haploid=True)

    sv_dip = SparseVar(dip)
    sv_hap = SparseVar(hap)

    assert sv_hap.ploidy == 1
    assert sv_hap.genos.shape[1] == 1
    assert sv_hap.n_variants == sv_dip.n_variants
    assert _haploid_call_sets(sv_hap) == _diploid_union_call_sets(sv_dip)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_haploid.py::test_from_pgen_haploid_metadata_and_or -v`
Expected: FAIL — `from_pgen()` got an unexpected keyword argument `haploid` (or SKIP if plink2 absent — if skipped, install plink2 or note coverage gap and proceed).

- [ ] **Step 3: Add the `haploid` param and collapse to `_process_contig_pgen`**

Change the `_process_contig_pgen` signature to add `haploid: bool = False` (after `sample_subset`):

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
    sample_subset: np.ndarray | None = None,
    haploid: bool = False,
) -> tuple[int, int]:
```

After the genos are reshaped, `-9`-fixed, and unsorter-reordered, and before the dosage read / `dense2sparse` (i.e. immediately after the `if unsorter is not None: genos = genos[unsorter]` block, around `genoray/_svar.py:2392-2393`), insert:

```python
        if haploid:
            # OR across haplotypes -> single haploid slot. `ploidy` above is the
            # native read ploidy (needed to reshape pgenlib output); the collapse
            # changes only the STORED ploidy. Dosages keep (s, v) and broadcast.
            genos = (genos == 1).any(axis=-2, keepdims=True).astype(np.int8)
```

- [ ] **Step 4: Add the `haploid` param and wiring to `from_pgen`**

Change the `from_pgen` signature to add `haploid: bool = False` in the keyword-only block (after `regions_overlap`):

```python
        regions_overlap: Literal["pos", "record", "variant"] = "pos",
        haploid: bool = False,
    ):
```

Add the same docstring Parameters entry as Task 1:

```
        haploid
            If ``True``, OR-collapse the ploidy axis into a single haploid call
            per sample (a variant present on any haplotype becomes one call) and
            record ``ploidy=1`` in the output metadata. Intended for unphased
            somatic data. Default ``False``.
```

Compute `out_ploidy` just before the metadata write (around `genoray/_svar.py:1230-1238`) and use it for metadata:

```python
        out_ploidy = 1 if haploid else pgen.ploidy

        with open(out / "metadata.json", "w") as f:
            json_str = SparseVarMetadata(
                version=CURRENT_VERSION,
                contigs=contigs,
                samples=caller_samples,
                ploidy=out_ploidy,
                fields={"dosages": np.dtype(DOSAGE_TYPE).name} if with_dosages else {},
            ).model_dump_json()
            f.write(json_str)
```

Change the concat shape (was `shape = (n_out, pgen.ploidy)` around `genoray/_svar.py:1258`):

```python
        shape = (n_out, out_ploidy)
```

Add `haploid=haploid` to the `_process_contig_pgen` delayed call (in the task loop around `genoray/_svar.py:1268-1279`), after `sample_subset=sample_subset`. Keep the existing `ploidy=pgen.ploidy` argument unchanged (native read ploidy):

```python
                    sample_subset=sample_subset,
                    haploid=haploid,
                )
```

Change the `_subset_var_idxs_and_recompute_af` call (around `genoray/_svar.py:1302-1308`) to pass `out_ploidy`:

```python
                survivors, af = _subset_var_idxs_and_recompute_af(
                    out,
                    n_total=len(kept_rows),
                    n_out=n_out,
                    ploidy=out_ploidy,
                    with_dosages=with_dosages,
                )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pixi run pytest tests/test_svar_haploid.py::test_from_pgen_haploid_metadata_and_or -v`
Expected: PASS (or SKIP if plink2 unavailable)

- [ ] **Step 6: Commit**

```bash
git add genoray/_svar.py tests/test_svar_haploid.py
git commit -m "feat: add haploid OR-collapse option to SparseVar.from_pgen"
```

---

### Task 4: CLI `--haploid` flag

**Files:**
- Modify: `genoray/_cli/__main__.py` — `write` command (`genoray/_cli/__main__.py:46-158`)
- Test: `tests/test_svar_haploid.py` (extend)

**Interfaces:**
- Consumes: `SparseVar.from_vcf(..., haploid=...)` and `SparseVar.from_pgen(..., haploid=...)` from Tasks 1 and 3.
- Produces: CLI `write` accepts `--haploid`; callable as `genoray._cli.__main__.write(source, out, ..., haploid=True)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar_haploid.py`:

```python
def test_cli_write_haploid_vcf(tmp_path: Path):
    from genoray._cli.__main__ import write as cli_write

    out = tmp_path / "cli.svar"
    cli_write(VCF_PATH, out, max_mem="1g", overwrite=True, haploid=True)
    sv = SparseVar(out)
    assert sv.ploidy == 1
    assert sv.genos.shape[1] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_haploid.py::test_cli_write_haploid_vcf -v`
Expected: FAIL — `write()` got an unexpected keyword argument `haploid`.

- [ ] **Step 3: Add the flag to the `write` command**

In `genoray/_cli/__main__.py`, add a `haploid` parameter to the `write` signature, after `no_breakend` (mirroring the existing flag style — a plain bool defaulting to False is sufficient; cyclopts exposes it as `--haploid`):

```python
    no_breakend: Annotated[bool, Parameter(name="--no-breakend", negative="")] = False,
    haploid: Annotated[bool, Parameter(name="--haploid", negative="")] = False,
) -> None:
```

Add to the docstring Parameters section:

```
    haploid
        If set, OR-collapse the ploidy axis into a single haploid call per
        sample (a variant present on any haplotype becomes one call) and record
        ``ploidy=1`` in the output metadata. Intended for unphased somatic data.
```

Thread `haploid=haploid` into both conversion calls (around `genoray/_cli/__main__.py:145-156`):

```python
        SparseVar.from_vcf(
            out, vcf, max_mem, overwrite, with_dosages=with_dosages, n_jobs=threads,
            haploid=haploid,
        )
    elif file_type == "pgen":
        pgen = PGEN(
            source,
            dosage_path=dosages,
            filter=pl_filter,
        )
        SparseVar.from_pgen(
            out, pgen, max_mem, overwrite, with_dosages=with_dosages, n_jobs=threads,
            haploid=haploid,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_svar_haploid.py::test_cli_write_haploid_vcf -v`
Expected: PASS

- [ ] **Step 5: Add a subprocess smoke test for the flag**

Append to `tests/test_svar_haploid.py`:

```python
import sys


def test_cli_write_haploid_subprocess(tmp_path: Path):
    out = tmp_path / "cli_sub.svar"
    r = subprocess.run(
        [sys.executable, "-m", "genoray._cli", "write", str(VCF_PATH), str(out),
         "--max-mem", "1g", "--overwrite", "--haploid"],
        check=False, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert SparseVar(out).ploidy == 1
```

- [ ] **Step 6: Run the smoke test**

Run: `pixi run pytest tests/test_svar_haploid.py::test_cli_write_haploid_subprocess -v`
Expected: PASS. (If the CLI uses a different option name for `max_mem`, run `pixi run python -m genoray._cli write --help` and use the printed flag name.)

- [ ] **Step 7: Commit**

```bash
git add genoray/_cli/__main__.py tests/test_svar_haploid.py
git commit -m "feat: add --haploid flag to genoray write CLI"
```

---

### Task 5: Ploidy=1 end-to-end audit (write_view + annotate_mutations) and docs

**Files:**
- Test: `tests/test_svar_haploid.py` (extend)
- Modify: `skills/genoray-api/SKILL.md`
- Modify: `docs/source/svar.md`

**Interfaces:**
- Consumes: a `ploidy=1` SparseVar produced by Task 1; `SparseVar.write_view(...)` and `SparseVar.annotate_mutations(reference, ...)` which already take `ploidy` from metadata.
- Produces: documentation reflecting the new public `haploid` option; a regression test proving the ploidy-generic read/view/annotate paths work at `ploidy=1`.

- [ ] **Step 1: Write the failing/guarding end-to-end test**

Append to `tests/test_svar_haploid.py`:

```python
def test_haploid_write_view_roundtrip(tmp_path: Path):
    hap = tmp_path / "hap.svar"
    SparseVar.from_vcf(hap, VCF(VCF_PATH), max_mem="1g", overwrite=True, haploid=True)
    sv = SparseVar(hap)

    out = tmp_path / "view.svar"
    # all variants (one row per contig), all samples
    import polars as pl

    bounds = (
        sv.index.group_by("CHROM", maintain_order=True)
        .agg(start=pl.lit(0, dtype=pl.Int32),
             end=(pl.col("POS").max() + 1).cast(pl.Int32))
        .rename({"CHROM": "chrom"})
        .select("chrom", "start", "end")
    )
    sv.write_view(regions=bounds, samples=list(sv.available_samples),
                  output=out, overwrite=True)

    view = SparseVar(out)
    assert view.ploidy == 1
    assert view.genos.shape[1] == 1
    # read paths return a ploidy-1 axis end to end
    rag = view.read_ranges(view.contigs[0])
    assert rag.shape[2] == 1
```

- [ ] **Step 2: Run the test**

Run: `pixi run pytest tests/test_svar_haploid.py::test_haploid_write_view_roundtrip -v`
Expected: PASS — the view/read paths are already ploidy-generic. If it FAILS, locate the spot that assumes `ploidy == 2` (search `genoray/_svar.py` for literal `2` in shape construction or AF denominators) and fix it to read `self.ploidy` / the passed `ploidy`, then re-run.

- [ ] **Step 3: Run the full new test file and the existing SparseVar suite (no regressions)**

Run: `pixi run pytest tests/test_svar_haploid.py tests/test_svar.py tests/test_svar_write_view.py -v`
Expected: all PASS (plink2-guarded tests may SKIP).

- [ ] **Step 4: Update `skills/genoray-api/SKILL.md`**

Find the cross-cutting conventions line `- Ploidy is always 2.` and replace it with:

```
- Ploidy is 2 by default; `SparseVar.from_vcf`/`from_pgen` (and `genoray write`) accept `haploid=True` / `--haploid`, which OR-collapses haplotypes into a single haploid call per sample and records `ploidy=1` in metadata (intended for unphased somatic data).
```

In the "SparseVar (`.svar`) — quick reference" Build block, add after the existing `from_pgen` example:

```python
# Unphased somatic data: collapse to a single haploid call per sample (ploidy=1)
genoray.SparseVar.from_vcf("out.svar", vcf, max_mem="4g", haploid=True)
```

- [ ] **Step 5: Update `docs/source/svar.md`**

Add a short subsection documenting the `haploid` write option and the `--haploid` CLI flag, mirroring the SKILL.md wording (what it does: OR-collapse across haplotypes; result: `ploidy=1` in metadata; intended use: unphased somatic cohorts). Verify the surrounding doc style first:

Run: `pixi run python -c "print(open('docs/source/svar.md').read()[:2000])"`

Then add the subsection in the same heading style as the existing write/CLI sections.

- [ ] **Step 6: Commit**

```bash
git add tests/test_svar_haploid.py skills/genoray-api/SKILL.md docs/source/svar.md
git commit -m "docs: document SparseVar haploid write option; add ploidy=1 e2e test"
```

---

## Self-Review

**1. Spec coverage:**
- Add `haploid` to `from_vcf` → Task 1. ✓
- Add `haploid` to `from_pgen` → Task 3. ✓
- CLI `--haploid` → Task 4. ✓
- OR-collapse in dense workers before `dense2sparse` → Tasks 1, 3. ✓
- Metadata `ploidy=1`, concat shape, AF denominator → Tasks 1, 3. ✓
- Dosages need no special handling → Task 2 verifies. ✓
- Sample-subset MAC drop + haploid AF → Task 2 verifies. ✓
- "Support arbitrary ploidy" audit (read/view/annotate at ploidy=1) → Task 5. ✓
- Docs: SKILL.md + svar.md → Task 5. ✓
- Out-of-scope items (per-sample/region collapse, phasing detection, lazy collapse) → not implemented. ✓

**2. Placeholder scan:** No TBD/TODO. Every code step shows complete code; the only deliberately-prose step is Task 5 Step 5 (svar.md prose), which gives exact content requirements and a style-check command. ✓

**3. Type consistency:**
- `haploid: bool = False` used identically across `from_vcf`, `from_pgen`, `_process_contig_vcf`, `_process_contig_pgen`, and the CLI `write`. ✓
- `out_ploidy = 1 if haploid else <source>.ploidy` used consistently for metadata, concat `shape`, and `_subset_var_idxs_and_recompute_af(ploidy=out_ploidy)`. ✓
- Collapse expression `(genos == 1).any(axis=-2, keepdims=True).astype(np.int8)` identical in both workers; `axis=-2` is the ploidy axis for both the VCF `(s, p, v)` and the reshaped PGEN `(s, p, v)` arrays. ✓
- Test helpers `_haploid_call_sets` / `_diploid_union_call_sets` reused across Tasks 1 and 3. ✓

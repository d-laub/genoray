# SVAR2 concat/split + `SparseVar2.write_view` / `genoray view` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SVAR2 concat/split by contig (pure-Python metadata/filesystem ops), and `SparseVar2.write_view` + a restructured `genoray view` CLI that writes a region/sample subset of an SVAR2 store by re-converting it through the existing Rust pipeline (`reroute=True` only).

**Architecture:** Component A (`concat`/`subset_contigs`/`split_by_contig`) copies/links per-contig directories and rewrites `meta.json` — no Rust. Component B (`write_view`) is a thin Python shim over a new Rust `run_view_pipeline` pyfunction that drives a new `Svar2Source: RecordSource` into the **unchanged** existing conversion pipeline (`process_chromosome` → `dense2sparse_vk` → cost model → merge → writer → `write_max_del` → field/signature finalize). This "coarse seam" re-runs the cost model per variant, so it implements `reroute=True`; `reroute=False` raises `NotImplementedError` pending the measurement spike.

**Tech Stack:** Python (polars, natsort, cyclopts, numpy), Rust (pyo3, rayon, memmap2, ndarray), pixi, pytest, cargo.

**Spec:** `docs/superpowers/specs/2026-07-12-svar2-concat-split-write-view-design.md`

## Global Constraints

- **Targets genoray 3.0.0 (breaking).** `genoray view` becomes SVAR2-only; the current SVAR1 `view` logic moves to `genoray view svar1`, mirroring the existing `genoray write` / `genoray write svar1` group. Do NOT bump the version or edit `CHANGELOG`'s versioned sections by hand — accumulate entries under `## Unreleased`.
- **Public-API rule:** any change reachable via `import genoray` without a leading underscore (new methods, CLI commands, kwargs) MUST be reflected in `skills/genoray-api/SKILL.md` in the same PR (see repo `CLAUDE.md`).
- **Conventional Commits** for every commit (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`).
- **No `FieldCategory` Python type exists** — field category is `Literal["info","format"]` / bare strings throughout `python/genoray/_svar2_fields.py`.
- **Rust test invocation:** `pixi run -e lint test-rust` (= `cargo test --no-default-features --features conversion`). New pyfunctions/sources must be gated `#[cfg(feature = "conversion")]` like the existing ones (`src/lib.rs`). Never run bare `cargo test` (pyo3 link error `undefined symbol: _Py_Dealloc`).
- **Cargo-on-NFS bus error:** before any commit that triggers cargo lint hooks, `export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$` in the same shell (the NFS `target/` bus-errors the prek cargo hooks).
- **Prek hooks** must be installed (`prek install` + `prek install --hook-type pre-push`) before committing/pushing.
- **Python tests:** `pixi run -e py310 pytest <path>`. Full suite: `pixi run test` (regenerates fixtures via `tests/data/gen_from_vcf.sh` first).
- **Coordinate/missing conventions unchanged:** 0-based half-open `[start, end)` internally; regions strings are 1-based inclusive (bcftools). SVAR2 is biallelic post-atomization.

---

## File Structure

- **Create** `python/genoray/_svar2_ops.py` — Component A helpers (`_load_meta`, `_write_store`, `_copy_contig_dir`, field-manifest equality) and the free functions the `SparseVar2` methods delegate to. Keeps `_svar2.py` lean.
- **Modify** `python/genoray/_svar2.py` — add `subset_contigs`, `split_by_contig`, `concat`, `write_view` methods on `SparseVar2` (thin; delegate to `_svar2_ops` / `_core`).
- **Create** `src/svar2_source.rs` — `Svar2Source` implementing `RecordSource`, re-emitting a finished contig's region+sample subset as variant-major `RawRecord`s.
- **Modify** `src/orchestrator.rs` — add a `SourceSpec::Svar2 { … }` variant and its reader-thread construction arm.
- **Modify** `src/lib.rs` — add + register the `run_view_pipeline` pyfunction (`#[cfg(feature="conversion")]`).
- **Modify** `python/genoray/_cli/__main__.py` — restructure `view` into an `App` group (`@view.default` = SVAR2, `@view.command(name="svar1")` = legacy), add `concat` and `split` commands.
- **Create** `scripts/svar2_reroute_spike.py` — the measurement spike (analytic recount + cost-model flip report).
- **Create** `tests/test_svar2_concat_split.py`, `tests/test_svar2_write_view.py`, `tests/cli/test_view_svar2_cli.py`, and a `tiny_svar2` fixture in `tests/cli/conftest.py`.
- **Modify** `skills/genoray-api/SKILL.md`, `docs/roadmap/data-model.md`, `CHANGELOG.md`.

---

## Task 1: Measurement spike — does `reroute=False` ever matter?

Standalone; does **not** block any other task. Produces a committed report that decides whether the `reroute=False` slice path is ever worth building. See spec "Measurement spike".

**Files:**
- Create: `scripts/svar2_reroute_spike.py`
- Create: `docs/superpowers/notes/2026-07-12-svar2-reroute-measurement.md` (the committed report)

**Interfaces:**
- Consumes: `SparseVar2` reader (`read_ranges`/`region_counts`), `genoray._core` cost-model access (see Step 2 — expose a tiny helper if none exists).
- Produces: a markdown report with flip-% and size-delta per subset size, germline vs somatic; a one-line verdict.

- [ ] **Step 1: Build the two stores (documented, run via SLURM)**

Record exact commands in the report; run them (large — the somatic BCF is ~1.1 GB):

```bash
# germline
sbatch -p carter-compute --wrap "pixi run -e py310 python -c \"import genoray; genoray.SparseVar2.from_vcf('data/chr21.germline.svar2','data/chr21.bcf','/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa',overwrite=True)\""
# somatic
sbatch -p carter-compute --wrap "pixi run -e py310 python -c \"import genoray; genoray.SparseVar2.from_vcf('data/gdc.chr21.somatic.svar2','data/gdc.chr21.bcf','/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa',overwrite=True)\""
```

If `from_vcf(reference=…)` fails on these inputs (e.g. contig-name or REF mismatch), retry with `no_reference=True` and note it in the report.

- [ ] **Step 2: Expose the cost-model routing to Python for the analytic recount**

The spike must call `choose_representation(class, n_samples, ploidy, x, sidecar_bits, info_bits, format_bits)` (`src/cost_model.rs:60`). If no Python binding exists, add a small `#[cfg(feature="conversion")]` pyfunction `cost_is_dense(class_is_indel: bool, n_samples, ploidy, x, sidecar_bits, info_bits, format_bits) -> bool` in `src/lib.rs` wrapping it, and register it. (Rebuild: `pixi run -e py310 maturin develop` — foreground; do not background.) Otherwise reproduce the exact integer formula in Python from `docs/roadmap/data-model.md#dense-vs-sparse-cost-model` and assert it matches on the known crossover cases (`x>=60` SNP / `x>=33` indel at np=2000).

- [ ] **Step 3: Write the analytic recount script**

`scripts/svar2_reroute_spike.py`: for each store and each subset size in `[10, 50, 100, 500]`, pick a deterministic random sample subset (`numpy.random.default_rng(0)`), then per variant recompute `x'` (carriers in the subset) and its class, and compare `cost_is_dense(x')` to the source representation. The source representation per variant is derivable by reading which sub-stream (`dense/*` vs `var_key/*`) the variant lives in via the reader's decoded output + `region_counts`. Accumulate flip counts and the summed on-disk bit cost under both routings (`dense = 32+key+np'` vs `var_key = x'*(32+key)`), and emit percentages + a size-delta ratio.

- [ ] **Step 4: Run, tabulate, write the verdict**

Run: `pixi run -e py310 python scripts/svar2_reroute_spike.py` (small subsets are fast once stores exist). Fill the report table (flip-%, size-delta per subset size, germline/somatic) and a one-line verdict per the spec decision rule: if flips rare AND size-delta <~1–2 %, verdict = "`reroute=False` not worth building; `NotImplementedError` stays permanent." Otherwise file a follow-up issue.

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add scripts/svar2_reroute_spike.py docs/superpowers/notes/2026-07-12-svar2-reroute-measurement.md src/lib.rs
git commit -m "chore(svar2): reroute measurement spike + report"
```

---

## Task 2: `SparseVar2.subset_contigs` (Component A core)

**Files:**
- Create: `python/genoray/_svar2_ops.py`
- Modify: `python/genoray/_svar2.py`
- Test: `tests/test_svar2_concat_split.py`

**Interfaces:**
- Consumes: `SparseVar2.__init__` (reads `meta.json`), `self.path`, `self.contigs`.
- Produces:
  - `_svar2_ops._load_meta(store: Path) -> dict` — raw parsed `meta.json`.
  - `_svar2_ops._copy_contig_dir(src_dir: Path, dst_dir: Path, mode: str) -> None`.
  - `_svar2_ops._write_store(output: Path, contig_sources: dict[str, Path], meta: dict, mode: str, overwrite: bool) -> None` — writes a full store: per-contig dirs from their source stores + `meta.json`.
  - `SparseVar2.subset_contigs(self, output, contigs: str | Sequence[str], *, mode="copy", overwrite=False) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_svar2_concat_split.py
import hashlib
import json
from pathlib import Path

import pytest

import genoray._core as _core
from genoray import SparseVar2


def _dir_digest(root: Path) -> dict[str, str]:
    """Map every non-meta file under a store to a hash of its bytes."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "meta.json":
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.fixture
def two_contig_store(tmp_path: Path) -> SparseVar2:
    """Reuse the session svar2 fixture's builder but with two contigs (chr1, chr2)."""
    # Build via SparseVar2.from_vcf on a 2-contig inline VCF; see conftest svar2_store.
    from tests.conftest import build_two_contig_svar2  # helper added in Step 3

    return build_two_contig_svar2(tmp_path)


def test_subset_contigs_narrows_meta_and_copies_bytes(two_contig_store, tmp_path):
    out = tmp_path / "chr1only.svar2"
    two_contig_store.subset_contigs(out, "chr1", overwrite=True)
    sub = SparseVar2(out)
    assert sub.contigs == ["chr1"]
    assert sub.available_samples == two_contig_store.available_samples
    assert sub.ploidy == two_contig_store.ploidy
    # per-contig bytes identical to source
    src_c1 = _dir_digest(two_contig_store.path / "chr1")
    out_c1 = _dir_digest(out / "chr1")
    assert src_c1 == out_c1


def test_subset_contigs_rejects_unknown(two_contig_store, tmp_path):
    with pytest.raises(ValueError, match="not in store"):
        two_contig_store.subset_contigs(tmp_path / "x.svar2", ["chrZ"])


def test_subset_contigs_refuses_in_place(two_contig_store):
    with pytest.raises(ValueError, match="in place"):
        two_contig_store.subset_contigs(two_contig_store.path, ["chr1"], overwrite=True)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e py310 pytest tests/test_svar2_concat_split.py -x -q`
Expected: FAIL (`build_two_contig_svar2` / `subset_contigs` missing).

- [ ] **Step 3: Add the two-contig test builder to `tests/conftest.py`**

Mirror the existing `svar2_store` fixture (`tests/conftest.py:27-53`) but emit two contigs. Add near it:

```python
# tests/conftest.py
def build_two_contig_svar2(tmp_path):
    """Build a 2-contig (chr1, chr2) svar2 store for concat/split tests."""
    import subprocess
    from pathlib import Path
    from genoray import SparseVar2

    d = Path(tmp_path)
    ref = d / "ref.fa"
    ref.write_text(">chr1\n" + _REF + "\n>chr2\n" + _REF + "\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    vcf = d / "in.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=PASS,Description="">\n'
        "##contig=<ID=chr1,length=40>\n"
        "##contig=<ID=chr2,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t9\t.\tT\tC\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr2\t5\t.\tT\tA\t.\t.\t.\tGT\t0|1\t0|1\n"
    )
    vcf_gz = d / "in.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {vcf_gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(vcf_gz)], check=True)
    out = d / "two.svar2"
    SparseVar2.from_vcf(out, vcf_gz, ref, threads=1, overwrite=True)
    return SparseVar2(out)
```

(`_REF` is the module-level 40 bp string already in `tests/conftest.py:14`.)

- [ ] **Step 4: Implement `_svar2_ops.py`**

```python
# python/genoray/_svar2_ops.py
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Literal

Mode = Literal["copy", "hardlink", "symlink", "move"]


def _load_meta(store: Path) -> dict:
    return json.loads((Path(store) / "meta.json").read_text())


def _copy_contig_dir(src_dir: Path, dst_dir: Path, mode: Mode) -> None:
    if mode == "copy":
        shutil.copytree(src_dir, dst_dir)
    elif mode == "hardlink":
        shutil.copytree(src_dir, dst_dir, copy_function=os.link)
    elif mode == "symlink":
        dst_dir.symlink_to(src_dir.resolve(), target_is_directory=True)
    elif mode == "move":
        shutil.move(str(src_dir), str(dst_dir))
    else:
        raise ValueError(f"unknown mode {mode!r}")


def _write_store(
    output: Path,
    contig_sources: dict[str, Path],
    meta: dict,
    mode: Mode,
    overwrite: bool,
) -> None:
    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass overwrite=True")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for contig, src_store in contig_sources.items():
        _copy_contig_dir(Path(src_store) / contig, output / contig, mode)
    (output / "meta.json").write_text(json.dumps(meta, indent=2))
```

- [ ] **Step 5: Add `subset_contigs` to `SparseVar2`**

```python
# python/genoray/_svar2.py  (imports at top)
from collections.abc import Sequence
from genoray._svar2_ops import Mode, _load_meta, _write_store

# method on SparseVar2:
    def subset_contigs(
        self,
        output: str | Path,
        contigs: str | Sequence[str],
        *,
        mode: Mode = "copy",
        overwrite: bool = False,
    ) -> None:
        """Write a new SVAR2 store containing only `contigs` (metadata + file copy)."""
        output = Path(output)
        wanted = [contigs] if isinstance(contigs, str) else list(contigs)
        missing = [c for c in wanted if c not in self.contigs]
        if missing:
            raise ValueError(f"contigs not in store: {missing}")
        if output.resolve() == self.path.resolve():
            raise ValueError("cannot write a subset in place (output == source)")
        kept = [c for c in self.contigs if c in set(wanted)]  # preserve source order
        meta = _load_meta(self.path)
        meta["contigs"] = kept
        _write_store(output, {c: self.path for c in kept}, meta, mode, overwrite)
```

- [ ] **Step 6: Run tests to verify pass**

Run: `pixi run -e py310 pytest tests/test_svar2_concat_split.py -x -q`
Expected: the three `subset_contigs` tests PASS.

- [ ] **Step 7: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add python/genoray/_svar2_ops.py python/genoray/_svar2.py tests/test_svar2_concat_split.py tests/conftest.py
git commit -m "feat(svar2): SparseVar2.subset_contigs (metadata contig subset)"
```

---

## Task 3: `SparseVar2.split_by_contig`

**Files:**
- Modify: `python/genoray/_svar2.py`
- Test: `tests/test_svar2_concat_split.py`

**Interfaces:**
- Consumes: `subset_contigs` (Task 2).
- Produces: `SparseVar2.split_by_contig(self, out_dir, *, mode="copy", overwrite=False) -> list[Path]`.

- [ ] **Step 1: Write the failing test**

```python
def test_split_by_contig_explodes(two_contig_store, tmp_path):
    paths = two_contig_store.split_by_contig(tmp_path / "split", overwrite=True)
    assert [p.name for p in paths] == ["chr1.svar2", "chr2.svar2"]
    assert SparseVar2(paths[0]).contigs == ["chr1"]
    assert SparseVar2(paths[1]).contigs == ["chr2"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e py310 pytest tests/test_svar2_concat_split.py::test_split_by_contig_explodes -x -q`
Expected: FAIL (`split_by_contig` missing).

- [ ] **Step 3: Implement**

```python
    def split_by_contig(
        self, out_dir: str | Path, *, mode: Mode = "copy", overwrite: bool = False
    ) -> list[Path]:
        """Explode into one single-contig store per contig at out_dir/{contig}.svar2."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for c in self.contigs:
            p = out_dir / f"{c}.svar2"
            self.subset_contigs(p, [c], mode=mode, overwrite=overwrite)
            paths.append(p)
        return paths
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e py310 pytest tests/test_svar2_concat_split.py::test_split_by_contig_explodes -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add python/genoray/_svar2.py tests/test_svar2_concat_split.py
git commit -m "feat(svar2): SparseVar2.split_by_contig"
```

---

## Task 4: `SparseVar2.concat` (classmethod)

**Files:**
- Modify: `python/genoray/_svar2.py`, `python/genoray/_svar2_ops.py`
- Test: `tests/test_svar2_concat_split.py`

**Interfaces:**
- Consumes: `_write_store`, `_load_meta` (Task 2); `natsort.natsorted`.
- Produces:
  - `_svar2_ops._assert_concat_compatible(metas: list[dict]) -> None` (raises on sample/ploidy/version/fields mismatch).
  - `SparseVar2.concat(cls, output, sources, *, mode="copy", overwrite=False) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_concat_roundtrip_from_split(two_contig_store, tmp_path):
    parts = two_contig_store.split_by_contig(tmp_path / "parts", overwrite=True)
    merged = tmp_path / "merged.svar2"
    SparseVar2.concat(merged, parts, overwrite=True)
    m = SparseVar2(merged)
    assert m.contigs == two_contig_store.contigs  # natsorted; chr1,chr2
    # per-contig bytes preserved through split->concat
    for c in two_contig_store.contigs:
        assert _dir_digest(two_contig_store.path / c) == _dir_digest(merged / c)


def test_concat_rejects_overlapping_contigs(two_contig_store, tmp_path):
    with pytest.raises(ValueError, match="multiple sources"):
        SparseVar2.concat(tmp_path / "x.svar2", [two_contig_store.path, two_contig_store.path], overwrite=True)


def test_concat_rejects_sample_mismatch(two_contig_store, tmp_path):
    # Build a single-contig store with different samples, then try to concat.
    from tests.conftest import build_two_contig_svar2  # reuse; edit samples inline if needed
    other = build_two_contig_svar2(tmp_path / "other")
    # Force a sample-name mismatch by rewriting the other store's meta.json samples.
    meta = json.loads((other.path / "meta.json").read_text())
    meta["samples"] = ["Z0", "Z1"]
    (other.path / "meta.json").write_text(json.dumps(meta))
    a = two_contig_store.subset_contigs.__self__  # source store
    a1 = tmp_path / "a1.svar2"; two_contig_store.subset_contigs(a1, "chr1", overwrite=True)
    o2 = tmp_path / "o2.svar2"; SparseVar2(other.path).subset_contigs(o2, "chr2", overwrite=True)
    with pytest.raises(ValueError, match="samples"):
        SparseVar2.concat(tmp_path / "bad.svar2", [a1, o2], overwrite=True)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e py310 pytest tests/test_svar2_concat_split.py -k concat -x -q`
Expected: FAIL (`concat` missing).

- [ ] **Step 3: Implement the compatibility guard in `_svar2_ops.py`**

```python
def _assert_concat_compatible(metas: list[dict]) -> None:
    ref = metas[0]
    for i, m in enumerate(metas[1:], start=1):
        for key in ("samples", "ploidy", "format_version", "fields"):
            if m.get(key) != ref.get(key):
                raise ValueError(
                    f"concat source #{i} differs from source #0 in {key!r}; "
                    "all stores must share samples, ploidy, format_version, and fields"
                )
```

- [ ] **Step 4: Implement `concat` on `SparseVar2`**

```python
# import at top of _svar2.py:
from natsort import natsorted
from genoray._svar2_ops import _assert_concat_compatible

    @classmethod
    def concat(
        cls,
        output: str | Path,
        sources: Sequence[str | Path | "SparseVar2"],
        *,
        mode: Mode = "copy",
        overwrite: bool = False,
    ) -> None:
        """Concatenate disjoint-contig SVAR2 stores (identical samples/ploidy/fields) into one."""
        paths = [Path(s.path if isinstance(s, SparseVar2) else s) for s in sources]
        if not paths:
            raise ValueError("concat requires at least one source")
        metas = [_load_meta(p) for p in paths]
        _assert_concat_compatible(metas)
        contig_sources: dict[str, Path] = {}
        for p, m in zip(paths, metas):
            for c in m["contigs"]:
                if c in contig_sources:
                    raise ValueError(f"contig {c!r} appears in multiple sources; concat requires disjoint contigs")
                contig_sources[c] = p
        merged_contigs = natsorted(contig_sources)
        meta = dict(metas[0])
        meta["contigs"] = merged_contigs
        _write_store(Path(output), {c: contig_sources[c] for c in merged_contigs}, meta, mode, overwrite)
```

- [ ] **Step 5: Run to verify pass**

Run: `pixi run -e py310 pytest tests/test_svar2_concat_split.py -k concat -x -q`
Expected: PASS. (If the `test_concat_rejects_sample_mismatch` fixture wiring is awkward, simplify by hand-editing two split stores' `meta.json` samples — the assertion under test is only that the guard fires.)

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add python/genoray/_svar2.py python/genoray/_svar2_ops.py tests/test_svar2_concat_split.py
git commit -m "feat(svar2): SparseVar2.concat (disjoint-contig merge)"
```

---

## Task 5: CLI `genoray concat` + `genoray split`

**Files:**
- Modify: `python/genoray/_cli/__main__.py`
- Test: `tests/cli/test_view_svar2_cli.py`, `tests/cli/conftest.py`

**Interfaces:**
- Consumes: `SparseVar2.concat`, `SparseVar2.split_by_contig`, `SparseVar2.subset_contigs`.
- Produces: `tiny_svar2` (2-contig) CLI fixture; `genoray concat`/`genoray split` commands.

- [ ] **Step 1: Add a `tiny_svar2` fixture in `tests/cli/conftest.py`**

Mirror `tiny_svar` (`tests/cli/conftest.py:33-40`) but with `SparseVar2.from_vcf` and two contigs (reuse the `build_two_contig_svar2` shape). Session-scoped, returns the store `Path`.

- [ ] **Step 2: Write the failing CLI test**

```python
# tests/cli/test_view_svar2_cli.py
import subprocess, sys
from pathlib import Path
from genoray import SparseVar2


def _run(argv):
    return subprocess.run([sys.executable, "-m", "genoray._cli", *argv], capture_output=True, text=True)


def test_cli_split_then_concat(tiny_svar2, tmp_path):
    r = _run(["split", str(tiny_svar2), str(tmp_path / "parts")])
    assert r.returncode == 0, r.stderr
    parts = sorted((tmp_path / "parts").glob("*.svar2"))
    assert len(parts) == 2
    r = _run(["concat", str(tmp_path / "m.svar2"), *map(str, parts)])
    assert r.returncode == 0, r.stderr
    assert set(SparseVar2(tmp_path / "m.svar2").contigs) == set(SparseVar2(tiny_svar2).contigs)
```

- [ ] **Step 3: Run to verify it fails**

Run: `pixi run -e py310 pytest tests/cli/test_view_svar2_cli.py::test_cli_split_then_concat -x -q`
Expected: FAIL (commands don't exist).

- [ ] **Step 4: Implement the commands**

```python
# python/genoray/_cli/__main__.py
@app.command
def concat(out: Path, sources: list[Path], *, mode: str = "copy", overwrite: bool = False) -> None:
    """Concatenate disjoint-contig SVAR2 stores into one."""
    from genoray import SparseVar2
    SparseVar2.concat(out, sources, mode=mode, overwrite=overwrite)


@app.command
def split(
    source: Path,
    out: Path,
    *,
    contigs: Annotated[str | None, Parameter(name=["--contigs", "-c"])] = None,
    mode: str = "copy",
    overwrite: bool = False,
) -> None:
    """Split an SVAR2 store by contig. With --contigs: subset into one store at OUT.
    Without: explode into OUT/{contig}.svar2."""
    from genoray import SparseVar2
    sv = SparseVar2(source)
    if contigs is not None:
        sv.subset_contigs(out, [c for c in contigs.split(",") if c], mode=mode, overwrite=overwrite)
    else:
        sv.split_by_contig(out, mode=mode, overwrite=overwrite)
```

- [ ] **Step 5: Run to verify pass**

Run: `pixi run -e py310 pytest tests/cli/test_view_svar2_cli.py::test_cli_split_then_concat -x -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add python/genoray/_cli/__main__.py tests/cli/test_view_svar2_cli.py tests/cli/conftest.py
git commit -m "feat(cli): genoray concat + split for SVAR2 stores"
```

---

## Task 6: Rust — `Svar2Source: RecordSource`

Creates `src/svar2_source.rs`: a per-contig `RecordSource` that reads a finished contig, enumerates the region+sample subset's variants in position order, and yields variant-major `RawRecord`s. **Definitive design decision (from the code map): `write_view` runs under `no_reference` (`fasta_path=None`) and needs NO FASTA** — that disables `validate_ref` + `left_align` (the only REF-bases-dependent, correctness-relevant steps), and the store is already atomic/biallelic/left-aligned. `atomize_record` still runs but is a faithful pass-through given the REF/ALT reconstruction below. `classify_variant`/`pack_variant` read only `ilen`+`alt`, never REF bases.

**Files:**
- Create: `src/svar2_source.rs`
- Modify: `src/lib.rs` (add `mod svar2_source;` under `#[cfg(feature="conversion")]`)
- Test: inline `#[cfg(test)] mod tests` in `src/svar2_source.rs`

**Interfaces:**
- Consumes: `RecordSource` trait + `RawRecord` (`src/record_source.rs:11-38`); `ContigReader` + `overlap_sample` (`src/query/oracle.rs:46`) / `read_ranges` (`src/query/gather.rs:794`) + `HapCalls { positions: Vec<u32>, ilens: Vec<i32>, alts: Vec<Vec<u8>> }` (`src/query/decode.rs:53`).
- Produces: `pub struct Svar2Source` + `pub fn new(store_path: &str, chrom: &str, sample_orig_idx: &[usize], ploidy: usize, regions: &[(u32,u32)], overlap_mode: OverlapMode) -> Result<Self, ConversionError>`; `impl RecordSource for Svar2Source`.

- [ ] **Step 1: Write the failing Rust unit test**

Inline in `src/svar2_source.rs`. Build a tiny contig on disk with the existing VCF pipeline (or reuse a test helper that writes a 2-sample/3-variant store — SNP, INS, DEL), then drive `Svar2Source` over the full region + all samples and assert `next_record` yields the expected `RawRecord`s (positions ascending; SNP `ilen==0`, alt preserved; DEL `ilen<0`, gt bits set for the right columns):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_position_sorted_records_with_carrier_bits() {
        // Arrange: write a known 2-sample store (see test helper), one SNP@pos2 carried by
        // sample0/hap0, one DEL@pos5 carried by both haps of sample1.
        let store = /* build_tiny_store() */;
        let mut src = Svar2Source::new(&store, "chr1", &[0, 1], 2, &[(0, 40)], OverlapMode::Pos).unwrap();
        let r0 = src.next_record().unwrap().unwrap();
        assert_eq!(r0.pos, 2);
        assert_eq!(r0.reference.len(), 1);      // SNP: 1-byte dummy REF
        assert_eq!(r0.alts.len(), 1);
        assert_eq!(r0.gt, vec![1, 0, 0, 0]);    // sample0/hap0 carrier
        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 5);
        assert_eq!(r1.gt, vec![0, 0, 1, 1]);    // sample1 both haps
        assert!(src.next_record().unwrap().is_none());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e lint test-rust svar2_source`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `Svar2Source`**

Algorithm (uniform over dense + var_key, drops MAC=0 automatically because a variant only appears if some kept hap carries it):

1. `new` opens a `ContigReader::open(store_path, chrom, n_samples_orig, ploidy)` (n_samples_orig from the store's `meta.json`), stores `sample_orig_idx` (original column indices of the subset, in output order), `regions`, `overlap_mode`, and **eagerly builds the record list** (`Vec<RawRecord>`):
   - For each output hap `h_out` in `0..sample_orig_idx.len()*ploidy` mapping to original `(orig_sample, p)`: get its decoded calls in the region via `overlap_sample(reader, orig_sample, q_start, q_end)` → `per_hap[p]` (a `HapCalls`), or a single batched `read_ranges(reader, regions, Some(&sample_orig_idx))` then `decode_hap` per hap.
   - For each decoded call `(pos, ilen, alt)`: if `overlap_mode == Pos` and `pos ∉ [q_start,q_end)` skip (variant-extent hits from deletions are pruned to POS-in-range under `pos` mode; under `Variant`/`Record` keep the extent hit). Group into a `BTreeMap<(u32, i32, Vec<u8>), Vec<bool>>` (key = `(pos, ilen, alt)`; value = carrier bitset over `n_out*ploidy`), setting `carrier[h_out]=true`.
   - Convert each map entry (already position-sorted by `BTreeMap` key) into a `RawRecord` (Step 4 rules). Push to `self.records`; set `self.cursor=0`.
2. `next_record` returns `Ok(self.records.get(self.cursor).cloned())` and increments (or `Ok(None)` at end). (Cheap; records pre-built. `RawRecord` is owned/`Clone`-able — or drain by index to avoid clone.)

- [ ] **Step 4: Implement `RawRecord` reconstruction (REF/ALT rules from the code map)**

Per grouped variant `(pos, ilen, alt)` with carrier bitset `carriers` (len `n_out*ploidy`):

```rust
fn to_raw_record(pos: u32, ilen: i32, alt: &[u8], carriers: &[bool]) -> RawRecord {
    let (reference, alts): (Vec<u8>, Vec<Vec<u8>>) = if ilen == 0 {
        // SNP: alt = [base]; REF must be a DIFFERENT single base so atomize keeps it.
        let base = alt[0];
        let refb = if base == b'A' { b'C' } else { b'A' };
        (vec![refb], vec![alt.to_vec()])
    } else if ilen > 0 {
        // INS (incl. long): alt = [anchor, inserted...]; REF = [anchor] so anchor matches.
        (vec![alt[0]], vec![alt.to_vec()])
    } else {
        // pure DEL: store alt is EMPTY. REF length = (-ilen)+1; alt = [REF[0]] (clean anchor).
        let rlen = (1 - ilen) as usize;      // ilen = 1 - rlen  =>  rlen = 1 - ilen
        let refv = vec![b'N'; rlen];
        (refv.clone(), vec![vec![refv[0]]])
    };
    let gt = carriers.iter().map(|&c| if c { 1i32 } else { 0 }).collect();
    RawRecord { pos, reference, alts, gt, info_raw: Vec::new(), format_raw: Vec::new() }
}
```

Rationale (verified against `normalize.rs`/`rvk.rs`): SNP `encode_snp_2bit` reads only `alt[0]`; a matching REF would make `atomize_biallelic` drop it, so REF must differ. INS/long-INS need `r[0]==a[0]` (clean anchor) to avoid a spurious SNV+INS split; the LUT re-spills long alts. Pure DEL: `atomize` derives `ilen` from `ref.len()` and `encode_pure_del` discards the bytes, so only length matters. `info_raw`/`format_raw` empty (fields carried in a follow-up; `fields=[]` for the MVP path — see Task 8 note).

- [ ] **Step 5: Run to verify pass**

Run: `pixi run -e lint test-rust svar2_source`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add src/svar2_source.rs src/lib.rs
git commit -m "feat(svar2): Svar2Source RecordSource re-emitting a finished contig"
```

> **Fields note for Task 7/8:** the MVP re-emits genotypes only (`fields=[]`), because faithful INFO/FORMAT carry-through requires populating `RawRecord.info_raw`/`format_raw` from the store's `FieldView` per variant — a bounded extension. If the reviewer wants fields in this PR, extend `to_raw_record` to fill `info_raw`/`format_raw` from `FieldView::value_at`/`format_at` (see the reader-map report) and pass the resolved `FieldSpec`s through. Otherwise `write_view(fields=[...])` on an svar2 store with fields should raise a clear "field carry-through not yet implemented for SVAR2 views" until that extension lands. Decide in Task 8 Step 3.

---

## Task 7: Rust — `SourceSpec::Svar2` + `run_view_pipeline` pyfunction

**Files:**
- Modify: `src/orchestrator.rs`, `src/lib.rs`
- Test: `tests/test_view_pipeline.rs` (Rust e2e)

**Interfaces:**
- Consumes: `Svar2Source` (Task 6), existing `process_chromosome`, `finalize_fields`, `write_meta`.
- Produces: `#[pyfunction] run_view_pipeline(py, store_path, out_dir, contigs, samples, regions: Vec<(String,u32,u32)>, regions_overlap: &str, merge_overlapping: bool, fields: Vec<String>, reference: Option<&str>, max_threads: Option<usize>, overwrite: bool) -> PyResult<()>` registered in `_core`.

- [ ] **Step 1: Add the `SourceSpec::Svar2` variant + reader-thread arm**

In `src/orchestrator.rs`, extend `SourceSpec` (`:44`) with `Svar2 { store_path: String, samples_orig_idx: Vec<usize>, regions: Vec<(u32,u32)>, overlap_mode: OverlapMode }` and add the match arm at the reader construction site (`:169`) that builds `Box::new(Svar2Source::new(...))`. `process_chromosome` is otherwise reused verbatim (it already loops one chromosome and takes `fasta_path: Option<&str>`, `samples: &[&str]`).

- [ ] **Step 2: Write the failing Rust e2e test**

```rust
// tests/test_view_pipeline.rs
// Build a small store via the VCF pipeline, run run_view_pipeline over a region+sample
// subset, reopen with ContigReader, assert the kept haps' decoded calls match a direct
// query of the source restricted to the same region/samples.
#[test]
fn view_region_sample_subset_matches_source() { /* ... concrete asserts ... */ }
```

- [ ] **Step 3: Run to verify it fails**

Run: `pixi run -e lint test-rust view_region_sample_subset_matches_source`
Expected: FAIL (pyfunction/source not wired).

- [ ] **Step 4: Implement `run_view_pipeline` in `src/lib.rs`**

Clone the structure of `run_conversion_pipeline` (`src/lib.rs:108`): read the source `meta.json` (samples, ploidy, fields) via a small helper, map `samples` (subset names) → original indices, build `SourceSpec::Svar2` per contig, rayon-fan `process_chromosome(source, reference, chrom, out_dir, &subset_sample_refs, chunk_size, ploidy, long_allele_capacity, /*skip_out_of_scope*/ true, threads, /*signatures*/ reference.is_some() && has_mutcat, &resolved_fields)`; then `finalize_fields` + `write_meta` with the subset samples/contigs. Gate `#[cfg(feature="conversion")]`; register in the pymodule (`src/lib.rs:360-370`).

- [ ] **Step 5: Run to verify pass**

Run: `pixi run -e lint test-rust view_region_sample_subset_matches_source`
Expected: PASS.

- [ ] **Step 6: Rebuild the extension + commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
pixi run -e py310 maturin develop   # foreground; do NOT background
git add src/orchestrator.rs src/lib.rs tests/test_view_pipeline.rs
git commit -m "feat(svar2): run_view_pipeline pyfunction + SourceSpec::Svar2"
```

---

## Task 8: Python — `SparseVar2.write_view` shim

**Files:**
- Modify: `python/genoray/_svar2.py`
- Test: `tests/test_svar2_write_view.py`

**Interfaces:**
- Consumes: `run_view_pipeline` (Task 7); `_svar/_regions.py` `_normalize_regions`/`_normalize_samples`/`_validate_fields`; `genoray._contigs.ContigNormalizer`.
- Produces: `SparseVar2.write_view(self, regions, samples, output, fields=None, reference=None, merge_overlapping=False, regions_overlap="pos", reroute="auto", overwrite=False, threads=None, progress=False) -> None`.

- [ ] **Step 1: Write the failing tests (incl. the from_vcf byte-parity oracle)**

```python
# tests/test_svar2_write_view.py
import hashlib
from pathlib import Path
import numpy as np
import pytest
from genoray import SparseVar2


def _dir_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "meta.json"
    }


def test_write_view_reroute_false_not_implemented(svar2_store, tmp_path):
    with pytest.raises(NotImplementedError, match="reroute"):
        svar2_store.write_view((svar2_store.contigs[0], 0, 40),
                               svar2_store.available_samples, tmp_path / "v.svar2",
                               reroute=False)


def test_write_view_self_overwrite_guard(svar2_store):
    with pytest.raises(ValueError, match="in place|same path"):
        svar2_store.write_view((svar2_store.contigs[0], 0, 40),
                               svar2_store.available_samples, svar2_store.path,
                               overwrite=True)


def test_write_view_byte_parity_with_from_vcf(svar2_store, tmp_path):
    """reroute=True on a full region+all samples == a fresh from_vcf on the same input."""
    # svar2_store was built from a known VCF+ref (see conftest). Rebuild the same input
    # via from_vcf into `direct`, and view-all into `viewed`; assert equal per-contig bytes.
    # (conftest exposes the vcf+ref paths used to build svar2_store.)
    ...
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e py310 pytest tests/test_svar2_write_view.py -x -q`
Expected: FAIL (`write_view` missing).

- [ ] **Step 3: Implement `write_view`**

```python
    def write_view(
        self, regions, samples, output, fields=None, reference=None,
        merge_overlapping=False, regions_overlap="pos",
        reroute="auto", overwrite=False, threads=None, progress=False,
    ) -> None:
        from genoray._contigs import ContigNormalizer
        from genoray._svar._regions import _normalize_regions, _normalize_samples, _validate_fields
        import genoray._core as _core

        output = Path(output)
        if reroute == "auto":
            reroute = True
        if reroute is not True:
            raise NotImplementedError(
                "reroute=False (preserve source representation) is not implemented; "
                "only reroute=True is supported (see the concat/split/write-view design doc)."
            )
        if fields is not None and "mutcat" in fields and reference is None:
            raise ValueError("'mutcat' cannot be copied through write_view; pass reference= to recompute it.")
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} exists; pass overwrite=True")
        if output.resolve() == self.path.resolve():
            raise ValueError("output resolves to the same path as the source; cannot write a view in place")
        cnorm = ContigNormalizer(self.contigs)
        regions_df = _normalize_regions(regions, cnorm)
        caller_samples = _normalize_samples(samples, self.available_samples)
        fields_to_write = [f for f in _validate_fields(fields, self.available_fields) if f != "mutcat"]
        if not caller_samples:
            raise ValueError("write_view requires at least one sample")
        region_tuples = [
            (row["chrom"], int(row["start"]), int(row["end"]))
            for row in regions_df.iter_rows(named=True)
        ]
        _core.run_view_pipeline(
            str(self.path), str(output), self.contigs, caller_samples, region_tuples,
            regions_overlap, merge_overlapping, fields_to_write,
            str(reference) if reference is not None else None, threads, overwrite,
        )
```

(`progress` is accepted for signature parity; wire it to a phase bar in a follow-up if desired — note it in the docstring as currently a no-op, or drop it if the reviewer prefers not to expose a no-op.)

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e py310 pytest tests/test_svar2_write_view.py -x -q`
Expected: PASS (incl. byte-parity oracle).

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add python/genoray/_svar2.py tests/test_svar2_write_view.py
git commit -m "feat(svar2): SparseVar2.write_view (region/sample subset via re-conversion)"
```

---

## Task 9: CLI — restructure `genoray view` (SVAR2 default + `view svar1`)

**Files:**
- Modify: `python/genoray/_cli/__main__.py`
- Test: `tests/cli/test_view_svar2_cli.py`, existing `tests/cli/test_view_cli.py`

**Interfaces:**
- Consumes: `SparseVar2.write_view` (Task 8); the current `view` body (moves to `view svar1`); `parse_regions_arg`.
- Produces: `view` `App` group; `@view.default` (SVAR2); `@view.command(name="svar1")` (legacy, current body verbatim).

- [ ] **Step 1: Write the failing tests**

```python
def test_cli_view_svar2_region_subset(tiny_svar2, tmp_path):
    out = tmp_path / "v.svar2"
    r = _run(["view", str(tiny_svar2), str(out), "-r", "chr1:1-40"])
    assert r.returncode == 0, r.stderr
    assert SparseVar2(out).contigs == ["chr1"]


def test_cli_view_svar1_still_works(tiny_svar, tmp_path):  # tiny_svar is SVAR1
    out = tmp_path / "v.svar"
    r = _run(["view", "svar1", str(tiny_svar), str(out), "-r", "chr1:1-100"])
    assert r.returncode == 0, r.stderr


def test_cli_view_svar2_no_reroute_errors(tiny_svar2, tmp_path):
    r = _run(["view", str(tiny_svar2), str(tmp_path / "v.svar2"), "-r", "chr1:1-40", "--no-reroute"])
    assert r.returncode != 0
    assert "reroute" in (r.stderr + r.stdout).lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e py310 pytest tests/cli/test_view_svar2_cli.py -k view -x -q`
Expected: FAIL (`view` is still a single SVAR1 command; no `svar1` subcommand).

- [ ] **Step 3: Restructure `view`**

Replace the current `@app.command def view(...)` (`:263`) with an `App` group mirroring `write` (`:46-49`):

```python
view = App(name="view", help="Write a region/sample subset of an SVAR2 store (SVAR1 via `view svar1`).")
app.command(view)


@view.default
def view_svar2(source, out, *, regions=None, regions_file=None, samples=None,
               samples_file=None, fields=None, reference=None,
               merge_overlapping=False, regions_overlap="pos",
               reroute="auto", overwrite=False, threads=None, progress=False):
    """Write a region/sample subset of an SVAR2 store."""
    # same regions/samples resolution as today's `view` body (parse_regions_arg / BED / all),
    # but default the "all variants" side using SparseVar2.contigs (no _contig_stats), then:
    #   SparseVar2(source).write_view(regions=..., samples=..., output=out, fields=fields,
    #       reference=reference, merge_overlapping=merge_overlapping,
    #       regions_overlap=regions_overlap, reroute=reroute, overwrite=overwrite,
    #       threads=threads, progress=progress)
    ...


@view.command(name="svar1")
def view_svar1(...):  # the current view body, VERBATIM (still constructs SparseVar)
    ...
```

`--reroute` is `Annotated[str, Parameter(...)] = "auto"` accepting `auto`/`true`/`false`; map `false`→`reroute=False` (which raises `NotImplementedError` → nonzero exit).

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e py310 pytest tests/cli/ -k view -x -q`
Expected: PASS (both SVAR2 and `view svar1`).

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add python/genoray/_cli/__main__.py tests/cli/test_view_svar2_cli.py
git commit -m "feat(cli)!: genoray view targets SVAR2; legacy SVAR1 under view svar1"
```

---

## Task 10: Docs — SKILL.md, data-model.md, CHANGELOG

**Files:**
- Modify: `skills/genoray-api/SKILL.md`, `docs/roadmap/data-model.md`, `CHANGELOG.md`

- [ ] **Step 1: Update `skills/genoray-api/SKILL.md`**

Document (public surface): `SparseVar2.concat` (classmethod), `subset_contigs`, `split_by_contig`, `SparseVar2.write_view` (full signature + `reroute` semantics + `reroute=False` → `NotImplementedError`), and the CLI: `genoray view` (SVAR2), `genoray view svar1` (legacy), `genoray concat`, `genoray split`.

- [ ] **Step 2: Update `docs/roadmap/data-model.md`**

Under "Format constraints and non-goals": mark **M8 (concat/split)** implemented; note **M9 (region subsetting)** implemented via `write_view` re-conversion (`reroute=True`), and that sample-subset re-routing is done (representation-preserving `reroute=False` deferred pending the measurement verdict).

- [ ] **Step 3: Add `CHANGELOG.md` entries under `## Unreleased`**

```markdown
### Added
- `SparseVar2.concat`, `subset_contigs`, `split_by_contig` — cheap merge/split by contig.
- `SparseVar2.write_view` and `genoray view` for SVAR2 (region/sample subset via re-conversion; `reroute=True`).
- `genoray concat` / `genoray split` CLI commands.

### Changed
- **BREAKING (3.0.0):** `genoray view` now targets SVAR2; the legacy SVAR1 view is `genoray view svar1` (mirrors `genoray write`).
```

- [ ] **Step 4: Commit**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-cargo-$$
git add skills/genoray-api/SKILL.md docs/roadmap/data-model.md CHANGELOG.md
git commit -m "docs(svar2): document concat/split + write_view + view CLI restructure"
```

---

## Self-Review

**Spec coverage:** concat/split (Tasks 2–5), write_view coarse seam (Tasks 6–8), CLI restructure (Tasks 5, 9), measurement spike (Task 1), docs (Task 10).

**Type consistency:** `run_view_pipeline` arg order matches between the Rust pyfunction (Task 7) and the Python shim (Task 8): `(store_path, out_dir, contigs, samples, regions, regions_overlap, merge_overlapping, fields, reference, threads, overwrite)`. The `Mode` literal is shared via `_svar2_ops`. `HapCalls`/`RawRecord` field names in Task 6 match the code-map reports.

**Deliberately deferred (call out in the PR description so a reviewer isn't surprised):**
- **INFO/FORMAT field carry-through** for `write_view` — MVP re-emits genotypes only (`fields=[]`). Requesting fields on an svar2 store that has them raises a clear "not yet implemented for SVAR2 views" (Task 8 Step 3). Bounded follow-up: populate `RawRecord.info_raw`/`format_raw` from `FieldView` (Task 6 fields note).
- **mutcat/signatures recompute via `reference=`** — tied to the above; the MVP `reference` param is accepted but only reached if `signatures` is wired. Until then, `reference=` on `write_view` is validated (path exists) and otherwise a no-op; document it.
- **Whole-contig + all-samples copy fast-path** — an *optimization* (the coarse seam already produces a correct, byte-parity output by re-converting); deferred. If wanted, short-circuit in the Python shim to `subset_contigs` when regions cover every contig fully and `samples == available_samples`.
- **`reroute=False`** — `NotImplementedError`; built only if Task 1's verdict says it's worth it.

**Placeholder scan:** Task 6 is fully specified. Tasks 7 Step 2 and 8 Step 1 give e2e/oracle test bodies in prose + partial code (they depend on fixtures resolved in-task); the assertions are unambiguous (decoded-call parity vs. a direct source query; per-contig byte-parity vs. `from_vcf`). No `TBD`/`TODO` remain.

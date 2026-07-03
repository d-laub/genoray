# VCF/PGEN → SVAR Conversion Subsetting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `regions` / `samples` / `merge_overlapping` / `regions_overlap` keyword-only params to `SparseVar.from_vcf` and `SparseVar.from_pgen`, so a region- and/or sample-subset of a VCF/PGEN is written to an SVAR during conversion (no full intermediate SVAR), matching `write_view` semantics including MAC=0 dropping on sample subsets.

**Architecture:** Resolve kept variant rows (region ∩ source filter) against the filtered source index up front; thread the sample subset into the per-contig workers (`set_samples` for VCF, `change_sample_subset` for PGEN); for region-only/no-subset write the output index up front (current behavior), and for any sample subset defer index writing until after the scan, then drop MAC=0 variants (pure-numpy `bincount` over `variant_idxs.npy` + a vectorized index remap), recompute AF, and write the index.

**Tech Stack:** Python, numpy, polars, numba (existing kernels only), seqpro (`sp.bed`), pyranges (via `sp.bed`), cyvcf2/oxbow (VCF), pgenlib (PGEN). Test env via `pixi`.

**Spec:** `docs/superpowers/specs/2026-06-10-svar-convert-subset-design.md`

---

## File Structure

- **Modify** `genoray/_svar.py`
  - Extract `_resolve_kept_rows(index_df, c_norm, regions_df, mode, merge_overlapping)` from the body of `_resolve_kept_var_idxs`; rewrite `_resolve_kept_var_idxs` as a thin wrapper.
  - Add `_build_working_index(src_index_path, pl_filter)` — filtered index frame with ALT-as-list, ILEN, and an `index` row-position column (the SVAR variant id), plus format flags.
  - Add `_write_index_from_working(working_df, rows, dst, alt_is_utf8, ilen_added, af)` — write an index frame to disk in the canonical SVAR on-disk format.
  - Add `_subset_var_idxs_and_recompute_af(out_path, n_total, n_out, ploidy, with_dosages)` — bincount MAC over `variant_idxs.npy`, remap surviving ids in place, return `(survivor_rows, af)`.
  - Rework `from_vcf` and `from_pgen`: new params, resolution, per-contig keep dispatch, deferred index + MAC-drop finalize.
  - Rework `_process_contig_vcf` (add `caller_samples`, per-contig `keep_local`) and `_process_contig_pgen` (add `sample_subset`).
- **Modify** `skills/genoray-api/SKILL.md`, `CHANGELOG.md` — public-API + changelog.
- **Create** `tests/test_svar_from_subset.py` — end-to-end + equivalence-oracle tests.

No new modules.

---

## Conventions used by every task

Run tests inside pixi:

```bash
pixi run pytest tests/test_svar_from_subset.py -v
```

PGEN tests are guarded with `pytest.importorskip`-style `shutil.which("plink2")` skips (existing convention in `tests/test_svar_filtering.py`).

---

## Task 1: Extract `_resolve_kept_rows` (pure refactor)

**Files:**
- Modify: `genoray/_svar.py` (`_resolve_kept_var_idxs` at ~206-298)
- Test: `tests/test_svar_from_subset.py` (new)

### - [ ] Step 1.1: Write a failing unit test for the extracted helper

Create `tests/test_svar_from_subset.py`:

```python
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genoray import VCF, PGEN, SparseVar
from genoray._svar import _resolve_kept_rows
from genoray._utils import ContigNormalizer


def _index_df(rows):
    """Build a minimal working-index frame: CHROM, POS, ILEN(list[int]), index(row id)."""
    df = pl.DataFrame(
        {
            "CHROM": [r[0] for r in rows],
            "POS": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "ILEN": [[0] for _ in rows],
        }
    )
    return df.with_row_index("index")


def test_resolve_kept_rows_pos_mode():
    # variants at 1-based POS 10, 20, 30 on chr1
    df = _index_df([("chr1", 10), ("chr1", 20), ("chr1", 30)])
    cnorm = ContigNormalizer(["chr1"])
    # region 0-based [9, 21) covers POS 10 (0-based 9) and POS 20 (0-based 19)
    regions = pl.DataFrame(
        {"chrom": ["chr1"], "start": pl.Series([9], pl.Int32), "end": pl.Series([21], pl.Int32)}
    )
    kept = _resolve_kept_rows(df, cnorm, regions, "pos", merge_overlapping=False)
    assert kept.tolist() == [0, 1]
```

### - [ ] Step 1.2: Run it, confirm it fails

Run: `pixi run pytest tests/test_svar_from_subset.py::test_resolve_kept_rows_pos_mode -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_kept_rows'`.

### - [ ] Step 1.3: Extract the helper

In `genoray/_svar.py`, replace the existing `_resolve_kept_var_idxs` (lines ~206-298) with the helper plus a wrapper. The helper is the existing body with `sv.var_ranges(c, ...)` replaced by the module-level `var_ranges(c_norm, index_df, c, ...)` and `sv.index` replaced by `index_df`:

```python
def _resolve_kept_rows(
    index_df: pl.DataFrame,
    c_norm: ContigNormalizer,
    regions: pl.DataFrame,
    mode: Literal["pos", "record", "variant"],
    merge_overlapping: bool,
) -> "NDArray[V_IDX_TYPE]":
    """Return a sorted, deduplicated array of kept row positions (values from the
    ``index`` column) in *index_df*.

    *index_df* must have columns ``CHROM`` (Utf8), ``POS`` (Int32), ``ILEN``
    (list[int]) and ``index`` (the id returned for kept rows). Coordinates in
    *regions* are 0-based, half-open (chrom/start/end).
    """
    if regions.height == 0:
        return np.empty(0, dtype=V_IDX_TYPE)

    # --- overlap detection / optional merge ---
    pyr_input = regions.rename({"start": "chromStart", "end": "chromEnd"})
    pyr = sp.bed.to_pyr(pyr_input)
    mod = type(pyr).__module__.split(".")[0]
    if mod == "pyranges":
        merged = pyr.merge()
    elif mod == "pyranges1":
        merged = pyr.merge_overlaps()
    else:
        raise RuntimeError(f"Unexpected PyRanges module: {type(pyr)!r}")

    if len(merged) != regions.height:
        if not merge_overlapping:
            raise ValueError("regions overlap; pass merge_overlapping=True to dedupe")
        regions = _coerce_bed_schema(sp.bed.from_pyr(merged))

    # --- collect candidate ids via var_ranges over the index frame ---
    kept_chunks: list[NDArray[V_IDX_TYPE]] = []
    sentinel = np.iinfo(V_IDX_TYPE).max
    for contig_key, sub in regions.group_by("chrom", maintain_order=False):
        c = contig_key[0] if isinstance(contig_key, tuple) else contig_key
        starts = sub["start"].to_numpy()
        ends = sub["end"].to_numpy()
        vr = var_ranges(c_norm, index_df, c, starts, ends)  # (n_ranges, 2)
        valid = vr[:, 0] != sentinel
        for s, e in vr[valid]:
            kept_chunks.append(np.arange(s, e, dtype=V_IDX_TYPE))

    if not kept_chunks:
        return np.empty(0, dtype=V_IDX_TYPE)
    candidates = np.unique(np.concatenate(kept_chunks))

    if mode == "variant":
        return candidates

    # --- pos / record mode: filter by POS membership ---
    region_by_contig: dict[str, tuple[NDArray, NDArray]] = {}
    for contig_key, sub in regions.group_by("chrom", maintain_order=False):
        c = contig_key[0] if isinstance(contig_key, tuple) else contig_key
        region_by_contig[c] = (sub["start"].to_numpy(), sub["end"].to_numpy())

    # candidates are values from the "index" column; map back to rows to read POS/CHROM
    by_id = index_df.filter(pl.col("index").is_in(candidates.tolist())).sort("index")
    cand_pos0 = by_id["POS"].to_numpy() - 1
    cand_chrom = by_id["CHROM"].to_list()
    cand_ids = by_id["index"].to_numpy()

    end_offset = 0 if mode == "pos" else 1
    keep_mask = np.zeros(len(cand_ids), dtype=bool)
    for i in range(len(cand_ids)):
        pair = region_by_contig.get(cand_chrom[i])
        if pair is None:
            continue
        r_starts, r_ends = pair
        p = cand_pos0[i]
        if np.any((r_starts <= p) & (p < r_ends + end_offset)):
            keep_mask[i] = True

    return cand_ids[keep_mask].astype(V_IDX_TYPE)


def _resolve_kept_var_idxs(
    sv: "SparseVar",
    regions: pl.DataFrame,
    mode: Literal["pos", "record", "variant"],
    merge_overlapping: bool,
) -> "NDArray[V_IDX_TYPE]":
    """Backward-compatible wrapper used by ``write_view``; resolves against
    ``sv.index`` (which carries CHROM/POS/ILEN/index)."""
    return _resolve_kept_rows(sv.index, sv._c_norm, regions, mode, merge_overlapping)
```

Ensure `var_ranges` is imported at the top of `_svar.py` (it is used by `SparseVar.var_ranges`; confirm the module-level name is in scope — add `from ._var_ranges import var_ranges` if not already imported).

### - [ ] Step 1.4: Run the new test + the full write_view suite

Run:
```bash
pixi run pytest tests/test_svar_from_subset.py::test_resolve_kept_rows_pos_mode tests/test_svar_write_view.py -v
```
Expected: all PASS (write_view behavior unchanged; new unit test passes).

### - [ ] Step 1.5: Commit

```bash
git add genoray/_svar.py tests/test_svar_from_subset.py
git commit -m "refactor(svar): extract _resolve_kept_rows from _resolve_kept_var_idxs"
```

---

## Task 2: Index helpers (`_build_working_index`, `_write_index_from_working`)

**Files:**
- Modify: `genoray/_svar.py`
- Test: `tests/test_svar_from_subset.py`

### - [ ] Step 2.1: Write failing test for the working-index builder

Append to `tests/test_svar_from_subset.py`:

```python
from genoray._svar import _build_working_index, SparseVar


def _make_svar_from_vcf(tmp_path, vcf_path) -> Path:
    out = tmp_path / "full.svar"
    SparseVar.from_vcf(out, VCF(vcf_path), max_mem="1g", overwrite=True)
    return out


def test_build_working_index_has_required_columns(tmp_path):
    sv_path = _make_svar_from_vcf(tmp_path, "tests/data/biallelic.vcf")
    df, alt_is_utf8, ilen_added = _build_working_index(
        SparseVar._index_path(sv_path), None
    )
    assert {"CHROM", "POS", "ILEN", "index"} <= set(df.columns)
    assert df["index"].to_list() == list(range(df.height))
    # ALT present as list[str] for filtering
    assert df.schema["ALT"] == pl.List(pl.Utf8)
```

> If `tests/data/biallelic.vcf` does not exist, generate fixtures first with
> `pixi run pytest tests/test_svar.py -q` (the test data is regenerated via
> `gen_from_vcf.sh`), then inspect `tests/data/` and substitute an existing VCF
> path. Do not invent a path that isn't present.

### - [ ] Step 2.2: Run it, confirm it fails

Run: `pixi run pytest tests/test_svar_from_subset.py::test_build_working_index_has_required_columns -v`
Expected: FAIL with `ImportError: cannot import name '_build_working_index'`.

### - [ ] Step 2.3: Implement both helpers

Add to `genoray/_svar.py` near `_write_filtered_index`:

```python
def _build_working_index(
    src_index_path: Path, pl_filter: "pl.Expr | None"
) -> "tuple[pl.DataFrame, bool, bool]":
    """Load the source index, apply ``pl_filter`` (if any), and return a working
    frame with ALT as list[str], an ILEN list column, and an ``index`` column
    holding each row's position (the SVAR variant id). Also returns
    ``(alt_is_utf8, ilen_added)`` so the on-disk format can be reconstructed.
    """
    lf = pl.scan_ipc(src_index_path)
    schema = lf.collect_schema()
    alt_is_utf8 = schema["ALT"] == pl.Utf8
    ilen_added = "ILEN" not in schema
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").str.split(","))
    if ilen_added:
        lf = lf.with_columns(ILEN=ILEN)
    if pl_filter is not None:
        lf = lf.filter(pl_filter)
    df = lf.collect().with_row_index("index")
    return df, alt_is_utf8, ilen_added


def _write_index_from_working(
    working_df: "pl.DataFrame",
    rows: "NDArray[V_IDX_TYPE]",
    dst: Path,
    alt_is_utf8: bool,
    ilen_added: bool,
    af: "NDArray[np.float32] | None" = None,
) -> None:
    """Write the rows of *working_df* selected by *rows* (in the given order) to
    *dst* in the canonical SVAR on-disk index format: ALT re-joined to comma-Utf8
    if it was originally Utf8, ILEN dropped if we added it, and the helper
    ``index`` column dropped. If *af* is given, (re)sets an ``AF`` column."""
    frame = working_df[rows.tolist()]
    if af is not None:
        if "AF" in frame.columns:
            frame = frame.drop("AF")
        frame = frame.with_columns(AF=pl.Series(af))
    if ilen_added and "ILEN" in frame.columns:
        frame = frame.drop("ILEN")
    if alt_is_utf8:
        frame = frame.with_columns(pl.col("ALT").list.join(","))
    frame = frame.drop("index")
    frame.write_ipc(dst, compression="zstd")
```

### - [ ] Step 2.4: Run the test

Run: `pixi run pytest tests/test_svar_from_subset.py::test_build_working_index_has_required_columns -v`
Expected: PASS.

### - [ ] Step 2.5: Commit

```bash
git add genoray/_svar.py tests/test_svar_from_subset.py
git commit -m "feat(svar): add working-index build/write helpers for conversion subsetting"
```

---

## Task 3: VCF region-only subsetting (`samples=None`)

**Files:**
- Modify: `genoray/_svar.py` (`from_vcf` ~880-966, `_process_contig_vcf` ~1642-1694)
- Test: `tests/test_svar_from_subset.py`

### - [ ] Step 3.1: Write failing end-to-end test (regions only, equivalence vs write_view)

Append to `tests/test_svar_from_subset.py`:

```python
def _read_all(sv: SparseVar):
    """Return per-(sample,ploidy) sets of (CHROM, POS) for deep comparison."""
    idx = sv.index
    chrom = idx["CHROM"].to_list()
    pos = idx["POS"].to_list()
    return sv, [(c, p) for c, p in zip(chrom, pos)]


def test_from_vcf_regions_only_matches_write_view(tmp_path):
    vcf_path = "tests/data/biallelic.vcf"
    full = _make_svar_from_vcf(tmp_path, vcf_path)
    sv_full = SparseVar(full)
    contig = sv_full.contigs[0]
    region = (contig, 0, 10_000_000)

    # Convert-time subset
    direct = tmp_path / "direct.svar"
    SparseVar.from_vcf(direct, VCF(vcf_path), max_mem="1g", overwrite=True, regions=region)
    sv_direct = SparseVar(direct)

    # Post-hoc view over the same region, all samples
    view = tmp_path / "view.svar"
    sv_full.write_view(regions=region, samples=list(sv_full.available_samples), output=view)
    sv_view = SparseVar(view)

    assert sv_direct.index["POS"].to_list() == sv_view.index["POS"].to_list()
    assert sv_direct.available_samples == list(sv_full.available_samples)
```

### - [ ] Step 3.2: Run it, confirm it fails

Run: `pixi run pytest tests/test_svar_from_subset.py::test_from_vcf_regions_only_matches_write_view -v`
Expected: FAIL with `TypeError: from_vcf() got an unexpected keyword argument 'regions'`.

### - [ ] Step 3.3: Add params + resolution + per-contig keep dispatch to `from_vcf`

Replace the `from_vcf` signature and body. New signature (keyword-only subset params):

```python
    @classmethod
    def from_vcf(
        cls,
        out: str | Path,
        vcf: VCF,
        max_mem: int | str,
        overwrite: bool = False,
        with_dosages: bool = False,
        n_jobs: int = -1,
        *,
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        samples: "str | Sequence[str] | PathLike | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: Literal["pos", "record", "variant"] = "pos",
    ):
```

Inside `from_vcf`, after the existing `out`, dosage-field, and `overwrite` checks and `out.mkdir(...)`, and after ensuring the gvi index exists (`vcf._write_gvi_index()` block), replace the `_write_filtered_index(...)` line and the metadata/dispatch with the following. The key change: build a working index, resolve kept rows, compute per-contig keep, and decide samples.

```python
        # --- resolve sample subset (None => all) ---
        if samples is None:
            caller_samples = list(vcf.available_samples)
            src_sample_idxs = None
        else:
            caller_samples = _normalize_samples(samples, vcf.available_samples)
            if not caller_samples:
                raise ValueError("from_vcf: `samples` selected no samples")
            src_sample_idxs = vcf._s2i[np.array(caller_samples)].astype(np.int64)

        # --- build working index (filtered) and resolve kept rows ---
        working_df, alt_is_utf8, ilen_added = _build_working_index(
            vcf._index_path(), vcf._pl_filter
        )
        if regions is None:
            kept_rows = working_df["index"].to_numpy().astype(V_IDX_TYPE)
        else:
            regions_df = _normalize_regions(regions, vcf._c_norm)
            kept_rows = _resolve_kept_rows(
                working_df, vcf._c_norm, regions_df, regions_overlap, merge_overlapping
            )
            if len(kept_rows) == 0:
                raise ValueError("no variants selected by `regions`")
            kept_rows = np.sort(kept_rows)

        # rows kept on each contig, as positions LOCAL to that contig's filtered
        # block (workers number variants per contig starting at 0).
        contigs = vcf.contigs
        kept_chrom = working_df["CHROM"].to_numpy()
        # contig block boundaries in the filtered frame (frame is in contig order)
        block_start = {}
        running = 0
        # working_df rows are grouped by contig in file order:
        counts = (
            working_df.group_by("CHROM", maintain_order=True)
            .agg(pl.len().alias("n"))
        )
        for c, n in zip(counts["CHROM"].to_list(), counts["n"].to_list()):
            block_start[c] = running
            running += n
        keep_local_by_contig: dict[str, np.ndarray] = {}
        kept_set = kept_rows
        for c in contigs:
            if c not in block_start:
                continue
            start = block_start[c]
            n = int(counts.filter(pl.col("CHROM") == c)["n"][0])
            in_block = kept_set[(kept_set >= start) & (kept_set < start + n)]
            keep_local_by_contig[c] = (in_block - start).astype(np.int64)
```

Then write metadata with `caller_samples` and `vcf.ploidy`:

```python
        with open(out / "metadata.json", "w") as f:
            json = SparseVarMetadata(
                version=CURRENT_VERSION,
                contigs=contigs,
                samples=caller_samples,
                ploidy=vcf.ploidy,
                fields={"dosages": np.dtype(DOSAGE_TYPE).name} if with_dosages else {},
            ).model_dump_json()
            f.write(json)

        subsetting_samples = samples is not None
        # When NOT subsetting samples, write the (region-restricted) index up front,
        # exactly as before for the no-region case.
        if not subsetting_samples:
            _write_index_from_working(
                working_df, kept_rows, cls._index_path(out), alt_is_utf8, ilen_added
            )
```

In the dispatch loop, pass `caller_samples` and the per-contig `keep_local`:

```python
            n_out = len(caller_samples)
            shape = (n_out, vcf.ploidy)
            tasks = []
            for chunk_idx, c in enumerate(contigs):
                task = joblib.delayed(_process_contig_vcf)(
                    vcf.path,
                    dosage_field=vcf.dosage_field if with_dosages else None,
                    max_mem=job_mem,
                    contig=c,
                    chunk_dir=chunk_dir,
                    chunk_idx=chunk_idx,
                    cyvcf2_filter=vcf._filter,
                    pl_filter=vcf._pl_filter,
                    caller_samples=None if samples is None else caller_samples,
                    keep_local=keep_local_by_contig.get(c),
                )
                tasks.append(task)
```

After `_concat_data(...)`, add the finalize for the sample-subset path (implemented in Task 5; for now, when `subsetting_samples` is False this branch is skipped, so Task 3 only exercises the up-front index path):

```python
            if subsetting_samples:
                survivors, af = _subset_var_idxs_and_recompute_af(
                    out, n_total=len(kept_rows), n_out=n_out,
                    ploidy=vcf.ploidy, with_dosages=with_dosages,
                )
                _write_index_from_working(
                    working_df, kept_rows[survivors], cls._index_path(out),
                    alt_is_utf8, ilen_added, af=af,
                )
```

> Note: `_concat_data` uses `shape = (n_out, vcf.ploidy)`; pass that `shape` (built above) into `_concat_data` instead of the old `(vcf.n_samples, vcf.ploidy)`.

### - [ ] Step 3.4: Update `_process_contig_vcf` to subset samples + keep rows

Replace `_process_contig_vcf` (lines ~1642-1694). Add the two params and the per-chunk column selection:

```python
def _process_contig_vcf(
    path: str | Path,
    dosage_field: str | None,
    max_mem: int | str,
    contig: str,
    chunk_dir: Path,
    chunk_idx: int,
    cyvcf2_filter: "Callable[..., bool] | None" = None,
    pl_filter: "pl.Expr | None" = None,
    caller_samples: "list[str] | None" = None,
    keep_local: "np.ndarray | None" = None,
) -> tuple[int, int]:
    vcf = VCF(
        path,
        filter=cyvcf2_filter,
        pl_filter=pl_filter,
        dosage_field=dosage_field,
        with_gvi_index=False,
    )
    if caller_samples is not None:
        vcf.set_samples(caller_samples)

    if dosage_field is not None:
        chunker = vcf.chunk(contig, max_mem=max_mem, mode=VCF.Genos8Dosages)
    else:
        chunker = vcf.chunk(contig, max_mem=max_mem, mode=VCF.Genos8)

    keep_sorted = None if keep_local is None else np.asarray(keep_local, dtype=np.int64)

    total_vars = 0
    n_chunks = 0
    contig_local_pos = 0  # running filtered-record position within this contig

    contig_dir = chunk_dir / f"c{chunk_idx}"
    contig_dir.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(chunker):
        if isinstance(data, tuple):
            genos, dosages = data
        else:
            genos = data
            dosages = None

        n_in = genos.shape[-1]
        if keep_sorted is not None:
            # positions in [contig_local_pos, contig_local_pos + n_in) that are kept
            lo = np.searchsorted(keep_sorted, contig_local_pos)
            hi = np.searchsorted(keep_sorted, contig_local_pos + n_in)
            sel = keep_sorted[lo:hi] - contig_local_pos
            contig_local_pos += n_in
            genos = genos[..., sel]
            if dosages is not None:
                dosages = dosages[..., sel]
        else:
            contig_local_pos += n_in

        n_vars = genos.shape[-1]
        if n_vars == 0:
            continue

        out_path = contig_dir / str(n_chunks)
        out_path.mkdir(parents=True, exist_ok=True)
        n_chunks += 1

        var_idxs = np.arange(total_vars, total_vars + n_vars, dtype=np.int32)
        if dosages is not None:
            sp_genos, sp_dosages = dense2sparse(genos, var_idxs, dosages)
            _write_genos(out_path, sp_genos)
            _write_dosages(out_path, sp_dosages.data)
        else:
            sp_genos = dense2sparse(genos, var_idxs)
            _write_genos(out_path, sp_genos)
        total_vars += n_vars
    return total_vars, n_chunks
```

> Two correctness points baked in here: (1) chunk output indices use `n_chunks`
> (a counter incremented only for non-empty emitted chunks) so `_concat_data`'s
> `range(n_chunks)` finds every directory even when some chunks select nothing;
> (2) `contig_local_pos` advances by the *unfiltered-by-region* chunk width so it
> stays aligned with the filtered-frame contig block.

### - [ ] Step 3.5: Run the regions-only test

Run: `pixi run pytest tests/test_svar_from_subset.py::test_from_vcf_regions_only_matches_write_view -v`
Expected: PASS.

### - [ ] Step 3.6: Run the no-arg regression (byte-compatible output)

Append and run this test:

```python
def test_from_vcf_no_subset_unchanged(tmp_path):
    vcf_path = "tests/data/biallelic.vcf"
    a = tmp_path / "a.svar"
    b = tmp_path / "b.svar"
    SparseVar.from_vcf(a, VCF(vcf_path), max_mem="1g", overwrite=True)
    SparseVar.from_vcf(b, VCF(vcf_path), max_mem="1g", overwrite=True, regions=None, samples=None)
    sa, sb = SparseVar(a), SparseVar(b)
    assert sa.index["POS"].to_list() == sb.index["POS"].to_list()
    assert sa.available_samples == sb.available_samples
    assert sa.n_variants == sb.n_variants
```

Run: `pixi run pytest tests/test_svar_from_subset.py::test_from_vcf_no_subset_unchanged -v`
Expected: PASS.

### - [ ] Step 3.7: Run the full existing svar suite (regression guard)

Run: `pixi run pytest tests/test_svar.py tests/test_svar_write_view.py tests/test_svar_filtering.py -q`
Expected: all PASS.

### - [ ] Step 3.8: Commit

```bash
git add genoray/_svar.py tests/test_svar_from_subset.py
git commit -m "feat(svar): from_vcf region subsetting during conversion"
```

---

## Task 4: MAC-drop finalize helper

**Files:**
- Modify: `genoray/_svar.py`
- Test: `tests/test_svar_from_subset.py`

### - [ ] Step 4.1: Write failing unit test for the finalize helper

Append to `tests/test_svar_from_subset.py`:

```python
from genoray._svar import _subset_var_idxs_and_recompute_af, V_IDX_TYPE
from seqpro.rag import lengths_to_offsets


def test_subset_var_idxs_drops_mac_zero(tmp_path):
    # 1 sample, ploidy 2, 3 candidate variants (ids 0,1,2). Variant 1 has MAC 0.
    out = tmp_path / "v.svar"
    out.mkdir()
    # slot 0 has variant 0; slot 1 has variant 2; variant 1 never appears.
    data = np.array([0, 2], dtype=V_IDX_TYPE)
    lengths = np.array([[1, 1]], dtype=np.int64)  # (n_samples=1, ploidy=2)
    offsets = lengths_to_offsets(lengths)
    np.memmap(out / "variant_idxs.npy", dtype=V_IDX_TYPE, mode="w+", shape=data.shape)[:] = data
    np.memmap(out / "offsets.npy", dtype=offsets.dtype, mode="w+", shape=offsets.shape)[:] = offsets

    survivors, af = _subset_var_idxs_and_recompute_af(
        out, n_total=3, n_out=1, ploidy=2, with_dosages=False
    )
    assert survivors.tolist() == [0, 2]               # variant 1 dropped
    # remapped ids: 0 -> 0, 2 -> 1
    vi = np.memmap(out / "variant_idxs.npy", dtype=V_IDX_TYPE, mode="r")
    assert sorted(vi.tolist()) == [0, 1]
    # AF over n_out*ploidy = 2 alleles: each surviving variant present once
    assert np.allclose(af, [0.5, 0.5])
```

### - [ ] Step 4.2: Run it, confirm it fails

Run: `pixi run pytest tests/test_svar_from_subset.py::test_subset_var_idxs_drops_mac_zero -v`
Expected: FAIL with `ImportError: cannot import name '_subset_var_idxs_and_recompute_af'`.

### - [ ] Step 4.3: Implement the finalize helper

Add to `genoray/_svar.py`:

```python
def _subset_var_idxs_and_recompute_af(
    out_path: Path,
    n_total: int,
    n_out: int,
    ploidy: int,
    with_dosages: bool,
) -> "tuple[NDArray[V_IDX_TYPE], NDArray[np.float32]]":
    """After concat, drop variants whose MAC across the (already sample-subset)
    output is 0 and remap surviving variant ids to a compacted range.

    A MAC=0 variant contributes no entries to ``variant_idxs.npy`` (it is never
    non-ref in any kept sample/ploidy slot), so dropping it requires only a
    remap of the stored ids — no entries are removed and offsets are unchanged.

    Returns ``(survivor_rows, af)`` where ``survivor_rows`` indexes into the
    ``n_total`` candidate rows (use it to subset the index frame) and ``af`` is
    the recomputed allele frequency for each survivor.
    """
    var_idxs = np.memmap(out_path / "variant_idxs.npy", dtype=V_IDX_TYPE, mode="r+")
    mac = np.bincount(np.asarray(var_idxs, dtype=np.int64), minlength=n_total)
    survivor_mask = mac > 0
    n_surv = int(survivor_mask.sum())
    if n_surv == 0:
        raise ValueError(
            "all selected variants have MAC=0 in the chosen sample subset; "
            "nothing to write"
        )
    n_dropped = n_total - n_surv
    if n_dropped:
        warnings.warn(
            f"from_*: dropping {n_dropped} variant(s) with MAC=0 in the output "
            "sample set",
            stacklevel=2,
        )
    remap = np.empty(n_total, dtype=V_IDX_TYPE)
    remap[survivor_mask] = np.arange(n_surv, dtype=V_IDX_TYPE)
    # every referenced id survives by construction, so this never hits a gap
    var_idxs[:] = remap[np.asarray(var_idxs, dtype=np.int64)]
    var_idxs.flush()
    del var_idxs

    survivor_rows = np.flatnonzero(survivor_mask).astype(V_IDX_TYPE)
    af = (mac[survivor_mask] / (n_out * ploidy)).astype(np.float32)
    return survivor_rows, af
```

> `with_dosages` is accepted for signature symmetry and future use; dosages are
> stored in the same per-entry order as `variant_idxs.npy` and are *not* reordered
> by a remap of ids (no entries move), so no dosage rewrite is needed here.

### - [ ] Step 4.4: Run the test

Run: `pixi run pytest tests/test_svar_from_subset.py::test_subset_var_idxs_drops_mac_zero -v`
Expected: PASS.

### - [ ] Step 4.5: Commit

```bash
git add genoray/_svar.py tests/test_svar_from_subset.py
git commit -m "feat(svar): MAC=0 drop + AF recompute finalize for conversion subsetting"
```

---

## Task 5: VCF sample subsetting end-to-end (wires Task 4 into `from_vcf`)

**Files:**
- Modify: `genoray/_svar.py` (the `if subsetting_samples:` branch added in Task 3 already calls the Task 4 helper)
- Test: `tests/test_svar_from_subset.py`

### - [ ] Step 5.1: Write failing test (samples subset + equivalence vs write_view)

Append to `tests/test_svar_from_subset.py`:

```python
def test_from_vcf_samples_subset_matches_write_view(tmp_path):
    vcf_path = "tests/data/biallelic.vcf"
    full = _make_svar_from_vcf(tmp_path, vcf_path)
    sv_full = SparseVar(full)
    keep_samples = list(sv_full.available_samples)[:1]

    direct = tmp_path / "direct.svar"
    SparseVar.from_vcf(direct, VCF(vcf_path), max_mem="1g", overwrite=True, samples=keep_samples)
    sv_direct = SparseVar(direct)

    view = tmp_path / "view.svar"
    sv_full.write_view(
        regions=(sv_full.contigs[0], 0, 1_000_000_000),
        samples=keep_samples, output=view,
    )
    sv_view = SparseVar(view)

    assert sv_direct.available_samples == keep_samples
    assert sv_direct.index["POS"].to_list() == sv_view.index["POS"].to_list()
    # MAC>0 invariant on output
    assert (sv_direct.index["AF"] > 0).all()
```

### - [ ] Step 5.2: Run it, confirm it fails

Run: `pixi run pytest tests/test_svar_from_subset.py::test_from_vcf_samples_subset_matches_write_view -v`
Expected: FAIL (samples not yet propagated to workers / finalize mismatch — e.g. wrong sample count in output, or `KeyError`/`AF` missing).

### - [ ] Step 5.3: Verify the `from_vcf` sample path

Confirm the Task 3 edits already: (a) build `src_sample_idxs`/`caller_samples`, (b) pass `caller_samples` to `_process_contig_vcf`, (c) use `shape=(n_out, vcf.ploidy)` in dispatch and `_concat_data`, (d) run the `if subsetting_samples:` finalize calling `_subset_var_idxs_and_recompute_af` and `_write_index_from_working(..., af=af)`. Fix any gaps.

The `write_view` equivalence holds because both produce variants in filtered+region order restricted to MAC>0; the VCF region for the view call spans the whole genome so the only subsetting is by sample.

### - [ ] Step 5.4: Run the test + combined regions+samples test

Append:

```python
def test_from_vcf_regions_and_samples(tmp_path):
    vcf_path = "tests/data/biallelic.vcf"
    full = _make_svar_from_vcf(tmp_path, vcf_path)
    sv_full = SparseVar(full)
    contig = sv_full.contigs[0]
    keep_samples = list(sv_full.available_samples)[:1]
    region = (contig, 0, 10_000_000)

    direct = tmp_path / "d.svar"
    SparseVar.from_vcf(
        direct, VCF(vcf_path), max_mem="1g", overwrite=True,
        regions=region, samples=keep_samples,
    )
    view = tmp_path / "v.svar"
    sv_full.write_view(regions=region, samples=keep_samples, output=view)

    assert SparseVar(direct).index["POS"].to_list() == SparseVar(view).index["POS"].to_list()
```

Run: `pixi run pytest tests/test_svar_from_subset.py -k "samples_subset or regions_and_samples" -v`
Expected: PASS.

### - [ ] Step 5.5: Commit

```bash
git add genoray/_svar.py tests/test_svar_from_subset.py
git commit -m "feat(svar): from_vcf sample subsetting with MAC=0 drop during conversion"
```

---

## Task 6: PGEN region + sample subsetting

**Files:**
- Modify: `genoray/_svar.py` (`from_pgen` ~969-1083, `_process_contig_pgen` ~1697-1765)
- Test: `tests/test_svar_from_subset.py`

### - [ ] Step 6.1: Write failing PGEN test (guarded on plink2)

Append to `tests/test_svar_from_subset.py`:

```python
def _vcf_to_pgen(tmp_path, vcf_path) -> Path:
    if shutil.which("plink2") is None:
        pytest.skip("plink2 not available")
    prefix = tmp_path / "conv"
    subprocess.run(
        ["plink2", "--vcf", str(vcf_path), "--make-pgen", "--out", str(prefix),
         "--allow-extra-chr"],
        check=True, capture_output=True,
    )
    return prefix.with_suffix(".pgen")


def test_from_pgen_regions_and_samples(tmp_path):
    vcf_path = "tests/data/biallelic.vcf"
    pgen_path = _vcf_to_pgen(tmp_path, vcf_path)

    full = tmp_path / "full.svar"
    SparseVar.from_pgen(full, PGEN(pgen_path), max_mem="1g", overwrite=True)
    sv_full = SparseVar(full)
    contig = sv_full.contigs[0]
    keep_samples = list(sv_full.available_samples)[:1]
    region = (contig, 0, 10_000_000)

    direct = tmp_path / "d.svar"
    SparseVar.from_pgen(
        direct, PGEN(pgen_path), max_mem="1g", overwrite=True,
        regions=region, samples=keep_samples,
    )
    view = tmp_path / "v.svar"
    sv_full.write_view(regions=region, samples=keep_samples, output=view)

    sv_d = SparseVar(direct)
    assert sv_d.available_samples == keep_samples
    assert sv_d.index["POS"].to_list() == SparseVar(view).index["POS"].to_list()
    assert (sv_d.index["AF"] > 0).all()
```

### - [ ] Step 6.2: Run it, confirm it fails

Run: `pixi run pytest tests/test_svar_from_subset.py::test_from_pgen_regions_and_samples -v`
Expected: FAIL with `TypeError: from_pgen() got an unexpected keyword argument 'regions'` (or skip if no plink2 — in that case install/skip is acceptable, but the signature change must still land).

### - [ ] Step 6.3: Add params + resolution + sample subset to `from_pgen`

Replace the `from_pgen` signature with the same keyword-only params as `from_vcf`:

```python
    @classmethod
    def from_pgen(
        cls,
        out: str | Path,
        pgen: PGEN,
        max_mem: int | str,
        overwrite: bool = False,
        with_dosages: bool = False,
        n_jobs: int = -1,
        *,
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        samples: "str | Sequence[str] | PathLike | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: Literal["pos", "record", "variant"] = "pos",
    ):
```

After `pgen._init_index()` and the existing asserts, and before writing metadata, build the working index against the *same* filtered frame the dispatch uses. Because PGEN needs physical variant indices for `read_alleles_list`, carry them alongside the filtered-frame row id.

```python
        # --- resolve sample subset ---
        if samples is None:
            caller_samples = list(pgen.available_samples)
            sample_subset = None
        else:
            caller_samples = _normalize_samples(samples, pgen.available_samples)
            if not caller_samples:
                raise ValueError("from_pgen: `samples` selected no samples")
            sample_subset = pgen._s2i[np.array(caller_samples)].astype(np.uint32)
        n_out = len(caller_samples)

        # --- working index (filtered) for region resolution + output index ---
        working_df, alt_is_utf8, ilen_added = _build_working_index(
            pgen._index_path(), pgen._filter
        )
        # physical pgen variant index aligned row-for-row with working_df
        # (pgen._index is the same filtered set in the same order, with an
        # "index" column holding the physical id).
        assert pgen._index is not None
        phys = pgen._index["index"].to_numpy().astype(np.uint32)
        assert len(phys) == working_df.height, "filtered index / pgen._index misaligned"

        if regions is None:
            kept_rows = working_df["index"].to_numpy().astype(V_IDX_TYPE)
        else:
            regions_df = _normalize_regions(regions, pgen._c_norm)
            kept_rows = _resolve_kept_rows(
                working_df, pgen._c_norm, regions_df, regions_overlap, merge_overlapping
            )
            if len(kept_rows) == 0:
                raise ValueError("no variants selected by `regions`")
            kept_rows = np.sort(kept_rows)

        # physical keep ids per contig, in kept-row (=output var id) order
        kept_chrom = working_df["CHROM"].to_numpy()[kept_rows]
        kept_phys = phys[kept_rows]
        keep_by_contig = {}
        for c in pgen.contigs:
            m = kept_chrom == c
            if m.any():
                keep_by_contig[c] = np.ascontiguousarray(kept_phys[m], dtype=np.uint32)
```

Write metadata with `caller_samples`, `n_out`-based `shape`:

```python
        contigs = pgen.contigs
        with open(out / "metadata.json", "w") as f:
            json = SparseVarMetadata(
                version=CURRENT_VERSION,
                contigs=contigs,
                samples=caller_samples,
                ploidy=pgen.ploidy,
                fields={"dosages": np.dtype(DOSAGE_TYPE).name} if with_dosages else {},
            ).model_dump_json()
            f.write(json)

        subsetting_samples = samples is not None
        if not subsetting_samples:
            _write_index_from_working(
                working_df, kept_rows, cls._index_path(out), alt_is_utf8, ilen_added
            )
```

In the dispatch loop, replace the `keep_by_contig` construction (the old `pgen._index.group_by(...)` block) with the `keep_by_contig` built above, and pass `sample_subset` + `n_out` to the worker:

```python
        shape = (n_out, pgen.ploidy)
        ...
            tasks: list[Any] = []
            for c in contigs:
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
                    chunk_idx=len(tasks),
                    sample_subset=sample_subset,
                )
                tasks.append(task)
```

> `mem_per_var` is computed from `pgen._mem_per_variant(...)` over the full sample
> count; that is a safe (conservative) over-estimate when a sample subset is read.
> Leave it unchanged.

After `_concat_data(...)`, add the finalize (same as `from_vcf`):

```python
            if subsetting_samples:
                survivors, af = _subset_var_idxs_and_recompute_af(
                    out, n_total=len(kept_rows), n_out=n_out,
                    ploidy=pgen.ploidy, with_dosages=with_dosages,
                )
                _write_index_from_working(
                    working_df, kept_rows[survivors], cls._index_path(out),
                    alt_is_utf8, ilen_added, af=af,
                )
```

### - [ ] Step 6.4: Update `_process_contig_pgen` to subset samples

Replace `_process_contig_pgen` (lines ~1697-1765). Add `sample_subset` and use it:

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
    sample_subset: "np.ndarray | None" = None,
) -> tuple[int, int]:
    geno_reader = PgenReader(bytes(Path(geno_path)), n_samples)
    dose_reader = (
        PgenReader(bytes(Path(dosage_path))) if dosage_path is not None else None
    )
    if sample_subset is not None:
        ss = np.ascontiguousarray(sample_subset, dtype=np.uint32)
        geno_reader.change_sample_subset(ss)
        if dose_reader is not None:
            dose_reader.change_sample_subset(ss)
        n_out = int(len(ss))
    else:
        n_out = n_samples

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

        genos = np.empty((n_vars, n_out * ploidy), dtype=np.int32)
        geno_reader.read_alleles_list(idxs, genos)
        genos = genos.astype(np.int8)
        genos = genos.reshape(n_vars, n_out, ploidy).transpose(1, 2, 0)
        genos[genos == -9] = -1

        dosages = None
        if dose_reader is not None:
            dosages = np.empty((n_vars, n_out), dtype=np.float32)
            dose_reader.read_dosages_list(idxs, dosages)
            dosages = dosages.transpose(1, 0)
            dosages[dosages == -9] = np.nan

        var_idxs = np.arange(total_vars, total_vars + n_vars, dtype=np.int32)
        if dosages is not None:
            sp_genos, sp_dosages = dense2sparse(genos, var_idxs, dosages)
            _write_genos(out_path, sp_genos)
            _write_dosages(out_path, sp_dosages.data)
        else:
            sp_genos = dense2sparse(genos, var_idxs)
            _write_genos(out_path, sp_genos)
        total_vars += n_vars
    return total_vars, n_chunks
```

> The PGEN `keep_idxs` are already restricted to region∩filter and sorted in
> output-var-id order, so the per-contig scan and the index agree by construction;
> no per-chunk selection (as in the VCF worker) is needed.

### - [ ] Step 6.5: Run PGEN tests

Run: `pixi run pytest tests/test_svar_from_subset.py -k pgen -v`
Expected: PASS (or SKIP if plink2 is unavailable).

### - [ ] Step 6.6: Run the full svar + pgen suites (regression)

Run: `pixi run pytest tests/test_svar.py tests/test_svar_write_view.py tests/test_svar_filtering.py tests/test_pgen.py -q`
Expected: all PASS / SKIP.

### - [ ] Step 6.7: Commit

```bash
git add genoray/_svar.py tests/test_svar_from_subset.py
git commit -m "feat(svar): from_pgen region + sample subsetting during conversion"
```

---

## Task 7: Error cases + dosages coverage

**Files:**
- Test: `tests/test_svar_from_subset.py`

### - [ ] Step 7.1: Write error-path tests

Append:

```python
def test_from_vcf_regions_no_match_raises(tmp_path):
    with pytest.raises(ValueError, match="no variants selected by `regions`"):
        SparseVar.from_vcf(
            tmp_path / "x.svar", VCF("tests/data/biallelic.vcf"),
            max_mem="1g", overwrite=True,
            regions=("chr1", 999_000_000, 999_000_100),
        )


def test_from_vcf_overlapping_regions_raise(tmp_path):
    regions = pl.DataFrame(
        {"chrom": ["chr1", "chr1"],
         "start": pl.Series([0, 50], pl.Int32),
         "end": pl.Series([100, 200], pl.Int32)}
    )
    with pytest.raises(ValueError, match="regions overlap"):
        SparseVar.from_vcf(
            tmp_path / "x.svar", VCF("tests/data/biallelic.vcf"),
            max_mem="1g", overwrite=True, regions=regions, merge_overlapping=False,
        )


def test_from_vcf_unknown_sample_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        SparseVar.from_vcf(
            tmp_path / "x.svar", VCF("tests/data/biallelic.vcf"),
            max_mem="1g", overwrite=True, samples=["NOT_A_SAMPLE"],
        )


def test_from_vcf_empty_samples_raises(tmp_path):
    with pytest.raises(ValueError, match="selected no samples"):
        SparseVar.from_vcf(
            tmp_path / "x.svar", VCF("tests/data/biallelic.vcf"),
            max_mem="1g", overwrite=True, samples=[],
        )
```

> Adjust the contig name (`chr1`) and overlapping coordinates to match the actual
> fixture if it uses a different contig — inspect with
> `SparseVar(full).contigs` from a prior test run.

### - [ ] Step 7.2: Write a with_dosages subset test

Append (uses a dosage-bearing fixture; if none exists, build one — check `tests/data/fixtures.py` / `gen_svar.py` for the dosage VCF/PGEN used by existing dosage tests and reuse that path):

```python
def test_from_vcf_subset_with_dosages(tmp_path):
    # Reuse whichever dosage-bearing VCF the existing dosage tests use.
    vcf_path = "tests/data/biallelic.vcf"  # replace with the dosage fixture if needed
    v = VCF(vcf_path, dosage_field="DS")
    full = tmp_path / "full.svar"
    SparseVar.from_vcf(full, VCF(vcf_path, dosage_field="DS"), max_mem="1g",
                       overwrite=True, with_dosages=True)
    sv_full = SparseVar(full)
    keep_samples = list(sv_full.available_samples)[:1]
    contig = sv_full.contigs[0]

    direct = tmp_path / "d.svar"
    SparseVar.from_vcf(direct, VCF(vcf_path, dosage_field="DS"), max_mem="1g",
                       overwrite=True, with_dosages=True,
                       regions=(contig, 0, 10_000_000), samples=keep_samples)
    view = tmp_path / "v.svar"
    sv_full.write_view(regions=(contig, 0, 10_000_000), samples=keep_samples, output=view)

    assert SparseVar(direct).index["POS"].to_list() == SparseVar(view).index["POS"].to_list()
```

> If no VCF in `tests/data` declares a `DS`/dosage FORMAT field, mark this test
> with `pytest.skip` referencing the missing fixture rather than fabricating one.

### - [ ] Step 7.3: Run all subset tests

Run: `pixi run pytest tests/test_svar_from_subset.py -v`
Expected: PASS / SKIP (no failures).

### - [ ] Step 7.4: Commit

```bash
git add tests/test_svar_from_subset.py
git commit -m "test(svar): error paths + dosage coverage for conversion subsetting"
```

---

## Task 8: Docstrings, SKILL.md, CHANGELOG

**Files:**
- Modify: `genoray/_svar.py` (`from_vcf`, `from_pgen` docstrings)
- Modify: `skills/genoray-api/SKILL.md`
- Modify: `CHANGELOG.md`

### - [ ] Step 8.1: Add docstring sections for the new params

In both `from_vcf` and `from_pgen`, extend the `Parameters` section. Add verbatim (indent to match the existing numpydoc block):

```rst
        regions
            Restrict the output to variants overlapping these region(s). Accepts
            a ``"chrom:start-end"`` string (1-based inclusive), a
            ``(chrom, start, end)`` tuple (0-based half-open), a
            polars/pandas/pyranges frame, or a BED file path. ``None`` (default)
            keeps all variants.
        samples
            Restrict the output to these samples (name, sequence of names, or
            path to a newline-delimited file). Caller order is preserved, deduped
            by first occurrence. ``None`` (default) keeps all samples. Variants
            whose minor allele count is 0 across the chosen samples are dropped
            from the output; if every variant drops, a ``ValueError`` is raised.
        merge_overlapping
            If ``regions`` contains overlapping intervals: ``False`` (default)
            raises ``ValueError``; ``True`` merges them.
        regions_overlap
            How variants are matched to regions — ``"pos"`` (default),
            ``"record"``, or ``"variant"`` — mirroring ``write_view``.
```

### - [ ] Step 8.2: Update SKILL.md

In `skills/genoray-api/SKILL.md`, find the SparseVar quick-reference block (around the `SparseVar.from_pgen("out.svar", ...)` example near line 142 and the filter note near line 145) and add a line documenting the new kwargs. Add after the existing `from_vcf`/`from_pgen` filter note:

```markdown
- `SparseVar.from_vcf` / `from_pgen` accept `regions=`, `samples=`,
  `merge_overlapping=`, `regions_overlap=` to subset by region and/or sample
  during conversion (same semantics as `SparseVar.write_view`); a sample subset
  drops MAC=0 variants from the output.
```

### - [ ] Step 8.3: Update CHANGELOG.md

Add under the unreleased/next `### Feat` section (match the existing CHANGELOG style):

```markdown
  - **svar**: from_vcf/from_pgen accept regions/samples/merge_overlapping/regions_overlap to subset during conversion (MAC=0 drop on sample subsets), mirroring write_view
```

### - [ ] Step 8.4: Verify docs reference real names

Run: `pixi run python -c "from genoray import SparseVar; import inspect; print(inspect.signature(SparseVar.from_vcf)); print(inspect.signature(SparseVar.from_pgen))"`
Expected: both signatures show `regions`, `samples`, `merge_overlapping`, `regions_overlap`.

### - [ ] Step 8.5: Commit

```bash
git add genoray/_svar.py skills/genoray-api/SKILL.md CHANGELOG.md
git commit -m "docs(svar): document conversion subsetting on from_vcf/from_pgen"
```

---

## Task 9: Full suite + lint

**Files:** none (verification only)

### - [ ] Step 9.1: Run the full test suite

Run: `pixi run test`
Expected: all PASS / SKIP (no failures, no regressions).

### - [ ] Step 9.2: Lint & format

Run:
```bash
ruff check genoray tests
ruff format --check genoray tests
```
Expected: clean. Run `ruff format genoray tests` and re-commit if formatting changes are needed.

### - [ ] Step 9.3: Final commit (if lint changed anything)

```bash
git add -A
git commit -m "style: ruff format for conversion subsetting"
```

---

## Self-Review notes (addressed in this plan)

- **Spec coverage:** API params (Task 3/6), intersection with `_pl_filter` (working index built with `pl_filter` — Task 2/3/6), regions+samples (Task 5/6), MAC=0 drop + AF recompute (Task 4/5/6), no-subset byte-compat (Task 3.6), equivalence-vs-`write_view` oracle (Task 3/5/6), errors (Task 7), docs+SKILL.md+CHANGELOG (Task 8). The "variant"-mode region match is exercised by the shared `_resolve_kept_rows` path already covered by `write_view` tests plus the new unit test (Task 1).
- **Deviation from spec (intentional, simpler & equivalent):** the spec said "reuse write_view's numba remap kernels"; because the workers subset samples *before* writing sparse data, a MAC=0 variant produces no entries, so the finalize is a pure-numpy `bincount` + id remap (Task 4) rather than the `_nb_count_kept`/`_nb_write_var_idxs` passes. Same output, less code.
- **Type/name consistency:** `_resolve_kept_rows`, `_build_working_index`, `_write_index_from_working`, `_subset_var_idxs_and_recompute_af`, worker params `caller_samples`/`keep_local` (VCF) and `sample_subset` (PGEN), and `shape=(n_out, ploidy)` are used consistently across tasks.
- **Open verification for the implementer:** confirm fixture paths/contig names in `tests/data` (regenerate via the existing test data step if absent) and the dosage FORMAT field name before running the dosage test; the plan flags each such spot inline.

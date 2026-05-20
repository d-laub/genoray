# SparseVar.write_view design

**Status:** approved for plan-writing
**Branch:** `feat/svar-write-subset`
**Date:** 2026-05-19

## Goal

Add `SparseVar.write_view` — produce a new on-disk SVAR directory that is a region- and sample-subset of an existing SVAR, with optional field selection.

## Public API

```python
def write_view(
    self,
    regions: str | tuple[str, int, int] | IntoFrameT | PathLike,
    samples: str | Sequence[str] | PathLike,
    output: PathLike,
    fields: Sequence[str] | None = None,
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
    overwrite: bool = False,
    threads: int | None = None,
) -> None
```

- `regions`: one of
  - `str` `"chr:start-end"` (1-based, end-inclusive),
  - `tuple[str, int, int]` (0-based, end-exclusive),
  - Frame-like recognized by `seqpro.bed` (bed / polars-bio / pyranges),
  - path to a BED-like file parsed by `sp.bed.read`.
- `samples`: sample name, sequence of sample names, or path to newline-delimited sample names. Caller's order is preserved (deduped, first occurrence wins).
- `output`: target directory, canonically `.svar`.
- `fields`: `None` → carry all `available_fields`; else a subset of `available_fields`. Genotypes are always written.
- `merge_overlapping`: `False` (default) → raise on overlapping regions; `True` → dedupe variants.
- `regions_overlap`: mirrors `bcftools --regions-overlap` (string-only; integer aliases not accepted).
- `overwrite`: matches `from_vcf` / `from_pgen` convention.
- `threads`: `None` → `len(os.sched_getaffinity(0))` if available, else `os.cpu_count() or 1`.

## On-disk layout (unchanged from existing SVAR)

```
<output>/
  metadata.json
  index.arrow
  variant_idxs.npy   # ragged data, dtype V_IDX_TYPE
  offsets.npy        # length N_out * ploidy + 1
  <field>.npy        # one per field written
                     # NOTE: AFs are stored as the "AF" column in index.arrow
                     # (matching cache_afs/_write_afs), not as a separate file.
```

## Algorithm

### 1. Input normalization (private helpers in `_svar.py`)

- `_normalize_regions(regions) -> pl.DataFrame` with columns `chrom: str, start: i32, end: i32` (0-based, half-open):
  - `str` → parse `chr:start-end` (1-based inclusive → 0-based half-open).
  - `tuple[str, int, int]` → one-row frame.
  - `PathLike` to a file → `sp.bed.read(path)`.
  - Frame-like → seqpro's bed-normalizer.
  - Apply `ContigNormalizer`; drop rows whose contig isn't in `self.contigs` with a warning.
- `_normalize_samples(samples) -> list[str]`:
  - `str` → singleton list; `Sequence[str]` → list; `PathLike` → newline-split.
  - Validate against `self.available_samples`; raise on unknowns. Dedupe preserving first occurrence.
- `_validate_fields(fields) -> list[str]`: `None` → `list(self.available_fields)`; else validate subset.

### 2. Variant selection

1. Sort regions by `(contig_order, start)`.
2. Overlap handling via seqpro/pyranges:
   ```python
   pyr = sp.bed.to_pyr(regions)
   mod = type(pyr).__module__.split(".")[0]
   if mod == "pyranges":
       merged = pyr.merge()
   elif mod == "pyranges1":
       merged = pyr.merge_overlaps()
   else:
       raise RuntimeError("unreachable: sp.bed.to_pyr returned unexpected type")
   merged_height = len(merged)  # pyranges len is row count; verify accessor at impl time
   if regions.height != merged_height:
       if not merge_overlapping:
           raise ValueError("regions overlap; pass merge_overlapping=True to dedupe")
       regions = _pyr_to_frame(merged)  # round-trip via seqpro; pin exact API at impl time
   ```
3. Resolve to variant indices according to `regions_overlap`:
   - `"pos"`: use existing `self.var_ranges(contig, starts, ends)` (POS in `[start, end)`).
   - `"record"`: widen each region's `end` by 1 before `var_ranges` (catches indels whose POS sits one past the right boundary).
   - `"variant"`: ILEN-aware variant-range resolution (true sequence-level overlap accounting for indel span). The existing `_find_starts_ends_with_length` does this at the offset level; for `write_view` we need the analogous variant-index range. Implementation will either (a) extract the ILEN-aware widening from `_find_starts_ends_with_length` into a `var_ranges_with_length` helper in `_var_ranges.py`, or (b) derive variant indices from the offsets returned by the existing helper. Choose at plan time.
4. Per `(start, end)` inclusive variant-range pair, build `np.arange(start, end + 1)`, concatenate per-contig.
5. `kept_var_idxs = np.unique(concat)` — globally sorted, deduped.
6. Remap is implicit: new index of source variant `v` is `np.searchsorted(kept_var_idxs, v)`.

### 3. Sample resolution

- `caller_samples` = normalized list (caller order, deduped).
- `src_sample_idxs = self._s2i[np.array(caller_samples)]` (`int64[N_out]`).
- Output ragged shape: `(N_out, ploidy, ~variants)`. Outer slot `(i, p)` ← source slot `(src_sample_idxs[i], p)`.

### 4. Two-pass write (numba-accelerated, no per-thread scratch)

Both kernels are `@njit(parallel=True, nogil=True, cache=True)` with `prange` over the outer sample axis. Membership and remap are inlined per element — no per-slot `pos`/`keep` arrays.

**Pass 1 — sizing** (touches source `offsets.npy`, `variant_idxs.npy`, and `kept_var_idxs`):

```python
@njit(parallel=True, nogil=True, cache=True)
def _count_kept(src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, out_lengths):
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in prange(n_out):
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            count = 0
            for j in range(src_offsets[src_slot], src_offsets[src_slot + 1]):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    count += 1
            out_lengths[i * ploidy + p] = count
```

Then `new_offsets = lengths_to_offsets(out_lengths.reshape(N_out, ploidy))`. Write `offsets.npy`. Allocate `variant_idxs.npy` of size `new_offsets[-1]` and each field's data array similarly.

**Pass 2 — write** (one kernel call for genos; once more per field):

```python
@njit(parallel=True, nogil=True, cache=True)
def _write_var_idxs(src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, new_offsets, out_var_idxs):
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in prange(n_out):
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            out_slot = i * ploidy + p
            wp = new_offsets[out_slot]
            for j in range(src_offsets[src_slot], src_offsets[src_slot + 1]):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    out_var_idxs[wp] = k   # k is already the new idx
                    wp += 1

@njit(parallel=True, nogil=True, cache=True)
def _write_field(src_field, src_data, src_offsets, src_sample_idxs, ploidy, kept_var_idxs, new_offsets, out_field):
    # same structure; writes src_field[j] instead of k
    ...
```

Both passes run inside `with numba_threads(threads_resolved):`.

**Memory footprint:** O(`kept_var_idxs.nbytes`) + O(offsets) + scratch loop scalars only. Independent of slot size and thread count. Source and destination arrays are memmapped; OS handles paging.

### 5. Threading context manager (new, in `_utils.py`)

```python
from contextlib import contextmanager

@contextmanager
def numba_threads(n: int):
    import numba
    prev = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        yield
    finally:
        numba.set_num_threads(prev)
```

And resolution helper:

```python
def _resolve_threads(threads: int | None) -> int:
    if threads is not None:
        return threads
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1
```

### 6. Index, AFs, metadata

- **Index**: `new_index = self.index.gather(kept_var_idxs)` (or equivalent polars idiom). Write via existing `SparseVar._index_path` helper.
- **AFs**: recompute over freshly written genos by reusing the existing `_nb_af_helper`:
  ```python
  afs = np.zeros(len(kept_var_idxs), dtype=np.float32)
  _nb_af_helper(afs, out_var_idxs_mm, new_offsets.ravel(), N_out * ploidy)
  new_index = new_index.with_columns(AF=pl.Series(afs))
  ```
  Persisted as the `AF` column of `index.arrow` (matching `cache_afs`/`_write_afs`); no separate file.
- **metadata.json**:
  ```python
  SparseVarMetadata(
      version=CURRENT_VERSION,
      samples=caller_samples,
      ploidy=self.ploidy,
      contigs=self.contigs,       # unchanged; zero-variant contigs allowed
      fields={name: self.available_fields[name].name for name in fields_to_write},
  )
  ```

### 7. Output directory lifecycle

- `output = Path(output)`.
- If `output.exists()` and not `overwrite` → `FileExistsError`.
- If `overwrite=True` and exists → `shutil.rmtree(output)`.
- `output.mkdir(parents=True)`.
- For atomicity if a write fails mid-way: not required for v1 (callers can re-run); keep behavior simple. Document this.

## Errors

- Unknown samples → `ValueError`.
- Unknown fields → `ValueError`.
- Overlapping regions with `merge_overlapping=False` → `ValueError` naming the offending neighbors.
- Empty regions / no variants selected → `ValueError("no variants selected")`.
- `regions_overlap="variant"` on a multiallelic dataset → `ValueError` (reuses the constraint in `_find_starts_ends_with_length`).

## Tests (`tests/test_svar.py`)

- Roundtrip: write_view → reopen → `read_ranges` over kept regions equals source's `read_ranges` for the same regions and sample subset.
- Sample reordering: output `available_samples` matches caller order.
- `merge_overlapping`: True succeeds with overlapping regions; False raises.
- All three `regions_overlap` modes against a fixture with a deletion spanning a region boundary.
- Field selection: `fields=None` (all), explicit subset, empty list (genos only).
- `afs.npy` matches `cache_afs()` recomputed on the new file.
- Overwrite semantics: raise without flag, succeed with flag.
- Empty selection raises.
- Threads: `threads=1` and `threads=None` both produce byte-identical output.

## Open follow-ups (out of scope for v1)

- Atomic writes via tempdir + rename.
- CLI exposure via `cyclopts`.
- Streaming subset to a different format (e.g., back to PGEN).

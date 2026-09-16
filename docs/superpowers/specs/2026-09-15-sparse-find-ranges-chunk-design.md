# Sparse emission from `find_ranges_chunk`

Status: approved design, not yet implemented.
Tracking: [mcvickerlab/genvarloader#405]. Consumer: [mcvickerlab/genvarloader#407].

## Problem

`find_ranges_chunk` materializes a dense `(n_haps, R, 2)` int64 block per channel
and its only production consumer throws away more than 99% of it.

The chunked search has three layers:

| layer | at | shape it produces |
|---|---|---|
| `find_ranges_haps` | `src/query/gather.rs:307` | fills two `(n_haps, R, 2)` int64 slices |
| `find_ranges_chunk` | `src/py_query_ranges.rs:343` | allocates those two as numpy, returns a dict |
| `_find_ranges_chunked` | `python/genoray/_svar2_batch.py:243` | reshapes to `(S, P, R, 2)`, yields a `RangesChunk` |

At the All of Us chr22 grid — `R` = 3,734 regions, `S` = 535,662 samples,
`P` = 2 — that is 4.0e9 `(region, sample, ploid)` cells at 32 bytes each across
both channels: **~128 GB per contig**. Only **0.45%** of those cells hold a
variant, so ~18e6 entries carry the payload and the other 99.55% are the pair
`(0, 0)`.

GVL recovers the useful cells with `np.nonzero` over a transposed view. That scan
is **71% of GVL's per-chunk write kernel** — more than the four gathers that
follow it combined — and it exists only because the dense block exists.

The dense block is also why `gvl.write` has to stream in sample chunks at all:
`max_mem` divides `S` until one chunk's dense payload fits, so chunk count is set
by an intermediate that is almost entirely zeros.

## What this changes

genoray gains a second emission format for the chunked search: only the
non-empty cells, region-major, in CSR form. The search itself is untouched.

### 1. `find_ranges_haps_sparse` (Rust)

Same `overlap()` loop as `find_ranges_haps`, same rayon split at
`PAR_COLUMN_THRESHOLD`, same `max_end_keys` accumulation. The difference is what
happens per cell: instead of writing `(start, end)` into a preallocated row, a
hap row pushes a record into a thread-local `Vec` only when
`snp.end > snp.start || indel.end > indel.start`.

Output ordering is **region-major, `cell_id`-ascending within a region** — the
order GVL's `nonempty_entries` produces today, so that function collapses to a
passthrough. Two facts make it cheap to reach:

- rayon's `map(...).collect::<Vec<_>>()` preserves index order, so concatenating
  the per-hap `Vec`s yields hap-ascending blocks, each internally
  region-ascending.
- A **stable counting sort** keyed on region over that concatenation is therefore
  sufficient: `R` buckets, one counting pass and one scatter pass over `N`
  entries. Stability is what makes `cell_id` ascend within each region; it is a
  correctness requirement, not an optimization.

The sort is O(N) over ~18e6 entries per chr22 contig, not over 4.0e9 cells.

### 2. `find_ranges_chunk_sparse` (binding)

Returns a **frozen, slotted dataclass**, not a dict. The existing dict contract
on `find_ranges`/`gather_ranges`/`overlap_batch` is a wire format shared by four
call sites; this is a single-consumer payload with a fixed field set, so a
stringly-keyed dict buys nothing and loses the type checker.

PyO3 cannot construct a Python dataclass directly, so the split is:

- the binding returns a 7-tuple of numpy arrays, unpacked at exactly one call
  site;
- `_find_ranges_chunked_sparse` builds `SparseRangesChunk` from it.

Fields, with `N` = non-empty cells in this chunk:

| field | dtype | shape | meaning |
|---|---|---|---|
| `sample_start` | `int` | — | offset on the selected-sample axis |
| `n_samples` | `int` | — | selected samples in this chunk |
| `region_ptr` | int64 | `(R+1,)` | CSR offsets; `region_ptr[-1] == N` |
| `cell_id` | int32 | `(N,)` | `sel_sample * ploidy + ploid` |
| `snp_start` | int64 | `(N,)` | var-key start, SNP channel |
| `snp_len` | int32 | `(N,)` | var-key length, SNP channel |
| `indel_start` | int64 | `(N,)` | var-key start, indel channel |
| `indel_len` | int32 | `(N,)` | var-key length, indel channel |
| `max_end_keys` | int64 | `(R,)` | unchanged from the dense chunk |

Four decisions inside that table:

**CSR, not a region column.** The output is already region-major, so `region_ptr`
is `R+1` int64 against `N` int32 for an explicit column. GVL's `append_contig`
counts regions today; with `region_ptr` it does not have to. A consumer that
wants the column gets it from `np.repeat(np.arange(R), np.diff(region_ptr))`.

**Lengths, not ends.** Matches GVL's `ENTRY_DTYPE`
(`snp_start <i8`, `indel_start <i8`, `snp_len <i4`, `indel_len <i4`), so no
subtraction on the consumer side.

**Columnar, not a packed record.** Five parallel `N`-length columns rather than
one structured dtype: genoray does not need to know GVL's on-disk record, and
packing `N`
entries into it is negligible next to what the scan cost.

**`cell_id` is absolute within the selection**, not chunk-relative. A chunk
covering `[sample_start, sample_start + n_samples)` emits
`cell_id = sel_sample * ploidy + ploid` with `sel_sample` indexed against the
whole selection. The consumer adds its own slot offset, or nothing at all when
selection order is slot order.

**A cell that is non-empty in only one channel still carries the other channel's
raw start** with length 0. genoray does not zero it: `gather_haps_readbound_impl`
derives `j = vs + k` inside the var-key loop and so never reads an empty range's
start (`src/query/gather.rs:772`), which is the invariant GVL's whole sparse
design rests on. Nothing in this change touches that function.

### 3. `_find_ranges_chunked_sparse` (Python)

Reuses `find_ranges_header` unchanged — everything O(regions) is already
computed there and is identical for both formats. `RangesStream` becomes generic
in its chunk type so the two streams share one header:

```python
RangesStream[RangesChunk]        # _find_ranges_chunked, dense
RangesStream[SparseRangesChunk]  # _find_ranges_chunked_sparse
```

A `sparse: bool` flag on one method would make the return type depend on an
argument value, which the type checker cannot follow. Two methods, one generic
container.

`RangesChunk` and `RangesStream` gain `slots=True` alongside the new dataclass —
they are already `frozen=True`, and the three should not disagree.

### 4. The dense path stays

`find_ranges_chunk` and `_find_ranges_chunked` are unchanged. Two reasons, both
temporary:

- it is the parity oracle for the sparse tests (below);
- GVL #407 is open and consumes the dense format; genoray must not break it
  before the pin flips.

Removing it once GVL's pin moves to the sparse source is filed as a follow-up,
not done here.

## Memory

`max_mem` sizing is **unchanged**: `bytes_per_sample` still assumes every cell is
non-empty, so chunk count is identical to today's. What changes is the realized
peak.

| | today | after |
|---|---|---|
| per-chunk allocation | dense, `n_haps * R * 32 B` | `~2 x 32 B x N` (thread-locals plus the final arrays) |
| chr22 contig total | ~128 GB streamed in chunks | ~1 GB at 0.45% fill |

Fill-adaptive chunk sizing — using a measured fill rate from the first chunk to
grow later ones — is deliberately **out of scope**. It trades a hard worst-case
bound for a heuristic, and the win here does not depend on it.

This also does not shrink GVL's per-contig entry accumulator (~60 bytes per
entry, ~1.1 GB at chr22, ~3.4 GB at chr19). That accumulator is O(total entries)
because a region-CSR table cannot be finalized until every sample chunk has
contributed; sparsifying the source does not change the entry count. Bounding it
needs region-axis chunking, which is a different change.

## Testing

Property testing lives in Rust, parity testing in Python. genoray has no
`hypothesis` dependency, and every Python-side store costs a `bcftools` call
plus a full conversion, so generating stores per example is not viable.
`proptest` is already a Rust dev-dependency, and the reorder — the only part of
this change with a non-obvious invariant — is a pure function over a `Vec`.

**Rust properties** over `sort_cells_by_region`, in `gather.rs`'s existing
`mod tests`: for random hap-ascending input, `region_ptr` is non-decreasing with
`region_ptr[0] == 0` and `region_ptr[-1] == N`; every cell in
`[region_ptr[r], region_ptr[r+1])` has `region == r`; `cell_id` strictly ascends
inside each of those slices (the stability requirement); and the output is a
permutation of the input. Plus units for `N == 0` and `R == 0`.

**Python parity** against the dense path, which is what the dense path is being
kept for. The reference is GVL's `nonempty_entries` reimplemented in the test as
a transpose plus `np.nonzero`; the sparse chunk must match it in `N`, in the
`(region, cell_id)` sequence *in order*, and in all four payload columns. Run at
the binding level per hap slice, and at the stream level over several `max_mem`
values including one sample per chunk. Also: an all-empty query region yields
`N == 0` with `region_ptr` all zeros, and `max_end_keys` matches the dense
kernel's exactly — it is accumulated separately in the sparse path, so a
divergence would otherwise be silent.

**Fixture.** The session `svar2_store` cannot exercise ordering: at 2 samples
the cost model (`choose_representation`, `src/cost_model.rs:60`) routes its INS
and DEL dense — both have 3 carrier calls — leaving one non-empty var_key cell
in 18. The same gap weakened three tests in GVL #407 (filed there as #406). This
change adds a `build_svar2_singleton_store` helper: one singleton SNP per
sample, so `x_calls == 1` routes every variant var_key, and a single region
returns one non-empty cell per sample with strictly ascending `cell_id`.

**Benchmark.** Dense-fill versus sparse-fill wall time on a synthetic store at a
realistic fill, reported per chunk. This measures what the change actually
claims: materialization cost, not search cost.

## What this does not do

The `overlap()` search cost is unchanged — this removes materialization, not
search. The 71% figure is GVL-side kernel time; genoray's own share of the write
path is the allocation and the fill. If the write path proves search-bound
afterwards, that is a new measurement and a new issue.

## Rollout

1. GVL #407 lands as written, against the dense format.
2. This change lands in genoray; a release is cut, since the binding is
   Python-facing.
3. A GVL follow-up flips its writer to `_find_ranges_chunked_sparse`, collapses
   `nonempty_entries` to a passthrough, and bumps both the genoray Python
   requirement and the `genoray_core` rev in `Cargo.toml` (pinned at `d66ec0e`
   today).
4. genoray follow-up: remove the dense chunk path.

No public genoray name changes — `find_ranges_chunk` is a `PyContigReader`
method and `_find_ranges_chunked` is underscore-private, so
`skills/genoray-api/SKILL.md` needs no update.

[mcvickerlab/genvarloader#405]: https://github.com/mcvickerlab/genvarloader/issues/405
[mcvickerlab/genvarloader#407]: https://github.com/mcvickerlab/genvarloader/pull/407

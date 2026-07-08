# SVAR 2.0 — M5 (part 2a): `max_del` standalone post-pass (design)

> Partial spec for roadmap milestone **M5** in
> [`docs/roadmap/svar-2.md`](../../roadmap/svar-2.md).
> Supplements: [`data-model.md`](../../roadmap/data-model.md#overlap-queries-and-deletions),
> [`architecture.md`](../../roadmap/architecture.md#query-path).
> Branch off `svar-2`, own worktree. **Parallel-safe** with the `(range, sample)` query
> spec (shares only the frozen `max_del.npy` contract below) and with the M2b
> left-alignment spec (disjoint files).

## Context

`overlap_range` (M5 part 1, `src/search.rs`) already takes a `max_region_length: u32`
bound as a **parameter**. Full M5 needs that bound to come from disk. Per
[`data-model.md`](../../roadmap/data-model.md#overlap-queries-and-deletions) the bound is
the **maximum deletion length per `(contig, sample, ploid)`**, stored in a per-contig
`max_del.npy`. This spec covers **producing** that file.

The key enabling fact: a deletion's length is fully recoverable from the finished
on-disk `var_key/indel` and `dense/indel` streams — a pure DEL encodes its `ilen` as a
signed-i31 **inside the inline key** (`src/rvk.rs`, `pack_variant`: `bit 31 = 1 → pure
DEL, bits[31:1] = signed ilen`), so it never spills to the LUT. Computing `max_del` is
therefore a **pure scan of the already-written `alleles.bin` keys** — no LUT reads, no
reference genome, and crucially **no coupling to the conversion pipeline**. It runs as a
standalone pass over a finished SVAR2 directory.

## Scope

**In:**

- A standalone function/binary that, given a finished SVAR2 contig directory, reads the
  indel key streams and emits the `max_del` artifacts (contract below).
- Decoding DEL `ilen` from a 32-bit indel key: reuse/expose the existing decode in
  `src/rvk.rs` rather than re-deriving the bit layout (respect the encoding seam — see
  [`architecture.md`](../../roadmap/architecture.md#the-encoding-agnostic-seam)). Deletion
  length `d = -ilen` for `ilen < 0`; `0` for SNP/INS keys.
- **`var_key/indel`** producer: for each column `c` (flattened `(sample, ploid)`), scan
  that column's slice of `alleles.bin` (bounded by `offsets.npy[c]..offsets.npy[c+1]`),
  take `max(d)`; write the `(sample, ploid)` array.
- **`dense/indel`** producer: the dense indel variant table is **shared across samples**
  (one position/key list per contig, genotypes selected by the 1-bit matrix), so its
  overlap search runs once over shared positions → a **single per-contig scalar**
  `max(d)` over the dense indel keys.
- Runs after a contig's merge completes; callable per-contig so it composes with the
  per-contig conversion fan-out (an orchestrator can invoke it as each contig finalizes,
  or as a batch sweep over an existing directory).

**Out (deferred / other specs):**

- Consuming `max_del.npy` in the query — the `(range, sample)` spec.
- Any change to the conversion/merge write path. (This is explicitly a *post*-pass; we do
  not piggyback on the merge, keeping it decoupled from M2b left-alignment which edits the
  read/normalize side.)
- `pointer` representation (M11) — no `pointer/indel` stream exists yet; when it lands,
  its indel keys join the same per-`(sample, ploid)` max.

## The frozen contract (shared with the `(range, sample)` query spec)

Per contig directory `{out}/{contig}/`:

- **`max_del.npy`** — dtype `u32`, shape `(n_samples, ploidy)`. `max_del[s, p]` = the
  maximum deletion length (in reference bases) over the `var_key/indel` stream for column
  `(s, p)`, or `0` if that column carries no deletions. The `(s, p)` → flat-column mapping
  **must match `src/merge.rs`** (`total_columns = num_samples * ploidy`, column index as
  used to slice `offsets.npy`); the producer derives it from the same convention, and the
  consumer reshapes identically. A pure-SNP contig still emits an all-zero array (so the
  consumer never special-cases a missing file).
- **`dense/max_del.npy`** — dtype `u32`, shape `()` (0-d scalar) or `(1,)`: the single
  max deletion length over the shared `dense/indel` key table for this contig, `0` if
  none / no dense indel stream.
- SNP sub-streams (`var_key/snp`, `dense/snp`) get **no** file — their overlap is a plain
  half-open range (`max_region_length == 0`), as the query spec relies on.

`overlap_range`'s existing proptest (`overlap_max_region_length_overshoot_is_safe`) proves
an **over**-estimate is always correct, only looser — so the producer may be conservative,
but the per-column tight bound keeps the linear sub-scan short.

## Design

- New module, e.g. `src/max_del.rs`. Public entry: `write_max_del(contig_dir: &Path,
  n_samples: usize, ploidy: usize) -> Result<(), Error>`.
- Read `var_key/indel/{offsets.npy, alleles.bin}` via mmap (`memmap2`, matching the
  reader access model). `alleles.bin` is a raw little-endian `u32` array; `offsets.npy` is
  the `total_columns + 1` prefix-sum from `merge.rs`.
- Per column, iterate keys in `[offsets[c], offsets[c+1])`, decode `ilen`, accumulate the
  max deletion length. This is `O(total indel calls)` and embarrassingly parallel across
  columns (rayon) if needed — but a single pass is likely I/O-bound and fine serial;
  **measure before parallelizing.**
- Read `dense/indel/alleles.bin` (if present), single max over all keys → scalar.
- Write both artifacts with `ndarray-npy::write_npy` (same dependency the merge already
  uses).
- Absent `var_key/indel` or `dense/indel` dir ⇒ treat as empty ⇒ zero output (still write
  `max_del.npy` so the consumer's contract holds; skip `dense/max_del.npy` only if the
  consumer is specified to default-absent-to-0 — **decide with the query spec**, keep the
  two symmetric).

## Files

- **New:** `src/max_del.rs` (+ `mod max_del` wiring).
- **Touch (minimal):** `src/rvk.rs` — expose a `pub fn deletion_len(key: u32) -> u32` (or
  reuse an existing decoder) so the bit layout stays in one place. `src/layout.rs` — add
  path helpers `max_del(contig_dir)` and `dense_max_del(contig_dir)`.
- **Optional:** `src/orchestrator.rs` — invoke `write_max_del` per contig after merge, so
  a normal conversion produces `max_del.npy` end-to-end. (Can be a follow-up; the pass is
  usable standalone against any finished directory.)

## Testing

- **Unit:** `deletion_len` decode vs. hand-constructed keys (SNP→0, INS→0, small DEL,
  large DEL near `MIN_I31`), mirroring the existing `rvk.rs` key-layout tests.
- **Proptest:** build a small synthetic contig (random atomized variants with known
  deletion lengths across random columns), run the pass, assert `max_del[s,p]` equals the
  brute-force per-column max — the same oracle style as `search.rs`.
- **e2e round-trip:** extend an existing conversion e2e test to assert `max_del.npy`
  exists, has shape `(n_samples, ploidy)`, and its values match a direct scan of the input
  VCF's deletions per sample. Feed the produced `max_del` into `overlap_range` on a
  deletion that spans a query start and assert the deletion is returned (closes the
  producer↔consumer loop once the query spec lands).

## Open questions

- **`dense/max_del` shape** — 0-d scalar vs `(1,)` vs folding into a length-1 axis of the
  main file. Pick whatever the query spec finds cleanest to load; keep both specs in sync.
- **Where the pass is triggered** — per-contig inside `orchestrator.rs` vs. a separate
  sweep entrypoint. Standalone-first; wire into the orchestrator once green.

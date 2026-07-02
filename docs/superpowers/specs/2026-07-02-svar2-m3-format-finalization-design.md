# SVAR 2.0 — M3: per-contig format finalization (design)

> Spec for roadmap milestone **M3** in
> [`docs/roadmap/svar-2.md`](../../roadmap/svar-2.md).
> Supplements: [`architecture.md`](../../roadmap/architecture.md#on-disk-layout),
> [`data-model.md`](../../roadmap/data-model.md#on-disk-layout).
> Branch off `svar-2`, own worktree.

## Context

M3's core is already done: output is partitioned per contig (`{out}/{contig}/…`) and the
merge emits sorted position + offset sidecars. What remains is **format finalization** — the
on-disk names are still provisional scratch names (`final_*`) that differ from the layout
spec, no `meta.json` is written, and the wire-format open question is unsettled. Until this
lands, the M5/M6 read path would be built against churny names.

This is a small, write-side, mostly-mechanical PR. It does **not** add the query path.

## Scope

**In:**

1. Rename the provisional `final_*` files to their spec base names and **settle the
   wire-format open question** (`.bin` vs `.npy`).
2. Emit a top-level `meta.json`.
3. Reconcile the roadmap + supplements (required by the working agreement), including moving
   `max_del.npy` cleanly under M5 to remove the current M3/M5 contradiction.

**Out:**

- `max_del.npy` — deferred to M5 (produced at conversion time but consumed only by the
  overlap algorithm; co-located with its consumer to prevent deletion-span drift).
- `{field}.npy` INFO/FORMAT per-variant columns (not yet extracted in the MVP).
- The `pointer` representation (M11) and the M5 search tree / query path.

## Design

### 1. Wire format — raw `.bin` for pwritten arrays, `.npy` for one-shot metadata

**Decision.** The tile merge writes `positions` / `alleles` / `genotypes` via concurrent
positional `pwrite` into a pre-sized file ([`merge.rs`](../../../src/merge.rs),
[`dense_merge.rs`](../../../src/dense_merge.rs)). Wrapping those in `.npy` would require
hand-rolling a fixed, 64-byte-aligned npy header and offsetting every `pwrite` past it — real
fragility for marginal gain, since these streams carry **no logical shape in the file** anyway
(2-bit-packed SNP / 1-bit-packed genotype bytes; shape is derived from `len(positions)` +
`meta.json`). So:

| Stream | Format | dtype | Written by |
| --- | --- | --- | --- |
| `positions.bin` | raw LE | `u32` | concurrent `pwrite` |
| `alleles.bin` | raw LE | `u8` (packed SNP) / `u32` (indel) | concurrent `pwrite` |
| `genotypes.bin` | raw LE | `u8` (packed 1-bit matrix) | concurrent `pwrite` |
| `offsets.npy` | npy | `u64` | one-shot `write_npy` |
| `long_alleles.bin` | raw LE | `u8` | streaming append (unchanged) |
| `long_allele_offsets.npy` | npy | `u64` | one-shot `write_npy` (unchanged) |

Principle: **bulk parallel-pwritten data arrays are raw `.bin` (mmap-friendly, no header
friction); small one-shot index/metadata sidecars stay self-describing `.npy`.** Array dtypes
are fixed by `format_version` and documented in `data-model.md`; Python reads the `.bin`
arrays via `np.memmap(path, dtype=…, mode='r')`.

### Concrete renames (`layout.rs` + doc only — no pwrite refactor)

Applies identically under `var_key/{snp,indel}/` and `dense/{snp,indel}/`:

| now | → |
| --- | --- |
| `final_positions.bin` | `positions.bin` |
| `final_keys.bin` | `alleles.bin` |
| `final_offsets.npy` | `offsets.npy` |
| `final_genotypes.bin` | `genotypes.bin` |

`final_keys` → `alleles` unifies naming with the layout spec (which already calls the packed
keys `alleles`). The doc tree currently names positions/alleles/genotypes `.npy`; flip those
three to `.bin` and record the resolution in the "var_key sidecar wire format" open question.

### 2. `meta.json`

Top-level `{output_dir}/meta.json`, written **once** in
[`run_conversion_pipeline`](../../../src/lib.rs) after all contigs succeed — the only scope
that knows the full `chroms` / `samples` / `ploidy`. Per-contig `process_chromosome` cannot
own it. Minimal, spec-matching schema:

```json
{ "format_version": 1, "samples": ["…"], "contigs": ["…"], "ploidy": 2 }
```

- `format_version` is an integer schema version for `SparseVar2` negotiation; bump on any
  breaking layout/dtype change.
- Array dtypes are **not** duplicated here — they are a `format_version` convention documented
  in `data-model.md`.
- New dep: `serde_json` only (build a `json!` value and `to_writer`; no `derive`, and Rust
  never reads it back — Python does via `json.load`).
- Extract a `write_meta(output_dir, format_version, samples, contigs, ploidy)` helper so it is
  unit-testable without going through the pyfunction.

### 3. Doc + test reconciliation

Required by the roadmap working agreement — same PR:

- **`data-model.md`**: flip the three filenames to `.bin` in the on-disk-layout tree; resolve
  the "var_key sidecar wire format" open question (record the `.bin`-for-pwritten /
  `.npy`-for-one-shot rationale); drop the "provisional filenames" notes in the dense +
  var_key sections; document the array dtype convention keyed to `format_version`.
- **`svar-2.md`**: M3 `[~]` → `[x]`; rewrite the remaining-text; **remove `max_del.npy` from
  M3's remaining list**, leaving it under M5 (fixes the contradiction); update M4's "unify
  dense filenames as part of M3" note to done.
- **`architecture.md`**: verify the `meta.json` / on-disk-layout bullets still read true.
- **Tests**: update name assertions in [`merge.rs`](../../../src/merge.rs),
  [`dense_merge.rs`](../../../src/dense_merge.rs), and [`layout.rs`](../../../src/layout.rs);
  add a `write_meta` unit test (round-trip the JSON and assert fields); existing e2e tests
  (which call `process_chromosome` per contig) get the renamed-file assertions.

## Worktree & dependencies

- Worktree: `.claude/worktrees/svar-2-m3-format-finalize`, branch off `svar-2`, PR into
  `svar-2`. Install prek hooks before committing.
- Blocks the **disk-integration** half of M5/M6 (they read this format), but not M5's
  format-independent search core — see
  [`2026-07-02-svar2-m5-partial-search-tree-design.md`](2026-07-02-svar2-m5-partial-search-tree-design.md),
  which can proceed fully in parallel.
</content>

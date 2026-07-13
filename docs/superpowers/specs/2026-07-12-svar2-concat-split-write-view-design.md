# SVAR2 concat/split by contig + `SparseVar2.write_view` / `genoray view`

**Date:** 2026-07-12
**Type:** feature (two components) + one breaking CLI change (targets 3.0.0)

---

## Problem / motivation

SVAR1 has `SparseVar.write_view(regions, samples, output, ...)` and a
`genoray view` CLI that writes a region/sample subset of a store. SVAR2 has no
equivalent, and no way to cheaply merge or split stores along contig
boundaries. This spec adds both to SVAR2:

- **A. concat / split by contig** — near-trivial metadata + filesystem ops.
  Contigs are fully independent on disk (`docs/roadmap/data-model.md`
  "Cheap merge/split by contig (M8)"), so only `meta.json.contigs` changes.
- **B. `SparseVar2.write_view` + `genoray view`** — region + sample subsetting,
  mirroring the SVAR1 surface, implemented as a re-conversion through the
  existing `SparseChunk` pipeline seam.

genoray is going to **3.0.0** with breaking changes. `genoray write` already
made SVAR2 the default (`@write.default`) with legacy SVAR1 behind
`genoray write svar1`. **`genoray view` mirrors this**: `genoray view` becomes
SVAR2-only and the current SVAR1 logic moves to `genoray view svar1`.

## Key facts that shape the design

### On-disk layout (why A is trivial and B is not)

SVAR2 is a directory: a top-level `meta.json`
(`{format_version, samples, contigs, ploidy, fields}`) plus one fully
self-contained subdirectory per contig (`{out}/{contig}/...`; see
`src/layout.rs`, `docs/roadmap/data-model.md#on-disk-layout`). Rust never reads
`meta.json` back — each `PyContigReader` is built from
`(path, contig, n_samples, ploidy)` (`python/genoray/_svar2.py:55`). So:

- **concat/split** only move/link per-contig directories and rewrite the
  `contigs` list in `meta.json`. No per-contig bytes are touched.
- **write_view** must physically rewrite per-contig arrays (bit-packed
  hap-major dense matrices, per-hap var_key call streams, `max_del`, the search
  tree, field sidecars). This is Rust work.

### The cost model and why sample-subsetting is the hard axis

Each variant is routed to `dense` or `var_key` by `choose_representation`
(`src/cost_model.rs`), a pure function of `(class, n_samples, ploidy, x_calls,
sidecar_bits, info_bits, format_bits)` where `x_calls` is the carrier count.
Route to `Dense` iff strictly cheaper; ties → `VarKey`.

- **Region subsetting** removes whole variants; every surviving variant keeps
  all its carriers, so `x_calls` is unchanged and routing is **unperturbed**.
- **Sample subsetting** changes every variant's `x_calls` (carriers among kept
  samples), which can (a) flip a variant's cheapest representation and (b) drop
  variants to carrier count 0 (MAC=0). This is why it's the expensive axis.

The `svar2-codec` keys (ILEN + ALT) are **reference-relative and portable
across representations**, so moving a variant between `dense` and `var_key`
requires **no reference** and no re-validation/left-alignment. A reference is
needed **only** to recompute the signatures/mutcat sidecar on a subset.

### The `SparseChunk` pipeline seam (implementation insight)

Conversion flows `Reader → DenseChunk → dense2sparse_vk → SparseChunk →
merge_mini_sc → writer` (`docs/superpowers/specs/2026-07-11-svar1-to-svar2-conversion-design.md`).
`SparseChunk` is the **sample-major sparse** shape. SVAR2's own `var_key`
streams are already exactly this shape, and dense variants expand into it
trivially. So `write_view` **reads the source store back into `SparseChunk`,
applies region+sample filters, and re-runs the existing `cost_model` →
`merge_mini_sc` → writer** — reusing all tested finalize/writer code and
re-deriving `max_del`/search tree/offsets/fields for free. This mirrors the
(not-yet-implemented) `from_svar1` `process_chromosome_svar1` design; `from_svar1`
is explicitly **out of scope** here, though this builds shared groundwork.

## Design (settled)

### Component A — concat / split by contig (pure Python)

New methods on `SparseVar2` (`python/genoray/_svar2.py`):

```python
# instance: keep only a subset of contigs (covers "split off one" and "extract several")
def subset_contigs(self, output, contigs, *, mode="copy", overwrite=False) -> None

# instance convenience: explode into one single-contig store per contig
def split_by_contig(self, out_dir, *, mode="copy", overwrite=False) -> list[Path]

# classmethod: concatenate disjoint-contig stores into one
@classmethod
def concat(cls, output, sources, *, mode="copy", overwrite=False) -> None
```

- `mode: Literal["copy", "hardlink", "symlink", "move"] = "copy"`. `copy` is the
  safe/portable default; `hardlink` is the zero-data-copy fast path on a single
  filesystem; `symlink` links back to the source; `move` is destructive and
  opt-in.
- `contigs` for `subset_contigs` is a single contig name or a sequence; each
  must exist in the source (else `ValueError`). Output `meta.json` = source with
  `contigs` narrowed (order preserved as given, then it is irrelevant — readers
  key by name).
- `split_by_contig` writes `out_dir/{contig}.svar2` for every contig and returns
  the list of paths.
- **`concat` guards** — every source must have identical `samples` (order +
  identity), `ploidy`, `format_version`, and `fields` manifest, and mutually
  **disjoint** contigs; otherwise `ValueError` naming the mismatch. Output
  `meta.json.contigs` = union of all sources' contigs, re-sorted with `natsort`
  (matches `from_vcf`'s contig ordering). `sources` is a sequence of paths or
  `SparseVar2` instances.
- **`overwrite`** guards the output existing, mirroring `from_vcf`.
- No source directory is consumed unless `mode="move"`.

### Component B — `SparseVar2.write_view` (Rust pipeline + Python shim)

```python
def write_view(
    self,
    regions: str | tuple[str, int, int] | Path | pl.DataFrame,
    samples: str | Sequence[str] | Path,
    output: str | Path,
    fields: Sequence[str] | None = None,
    reference: "str | Path | None" = None,
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
    reroute: bool | Literal["auto"] = "auto",
    overwrite: bool = False,
    threads: int | None = None,
    progress: bool = False,
) -> None
```

Signature mirrors `SparseVar.write_view`; the only addition is **`reroute`**.

**Implementation strategy.** A thin Python shim (mirroring `from_vcf`) that
normalizes/validates inputs and calls a new Rust pyfunction (working name
`run_view_pipeline` / `process_chromosome_svar2`). Per contig, the Rust side:

1. Resolves kept variant indices from `regions` (`regions_overlap` semantics
   identical to SVAR1: `pos` / `record` / `variant`).
2. Reads the source's four sub-streams back into `SparseChunk`: `var_key` calls
   filtered to kept haplotypes ∩ region (already sample-major per-call); dense
   variants in-region expanded to per-hap calls for kept haplotypes.
3. Routes each variant to an output representation (see `reroute` below) and
   hands `SparseChunk`(s) to the **existing** `merge_mini_sc` + writer, which
   re-derive `max_del`, the search tree, offsets, and field sidecars.
4. Writes `meta.json` with the subset `samples`/`contigs`.

**`reroute` semantics** (only meaningful when subsetting samples — region-only
leaves `x_calls` unchanged, so routing is stable and output is byte-comparable
to the source on those contigs):

- `True` — recompute each variant's subset carrier count `x'` and call
  `choose_representation(x')`. Size-optimal; byte-comparable to a fresh
  `from_vcf` on the same region+sample subset.
- `False` — keep each variant's **source** representation; still drop MAC=0
  variants and rebuild streams. Correct, avoids dense↔var_key data movement,
  possibly size-suboptimal.
- `"auto"` — resolves to the default chosen by the measurement spike below.

**Semantics mirrored from SVAR1 `write_view`:**

- MAC=0 variants (0 carriers in the kept subset) are dropped. If every candidate
  variant drops (or `regions` selects none), raise `ValueError`.
- **Fail-fast band ordering** (as in
  `docs/superpowers/specs/2026-06-27-write-view-fail-fast-design.md`): all raises
  (`output` exists & not `overwrite`; `output` resolves to `self.path`; empty
  samples; bad region/sample/field names; bad `reference` path) happen **before**
  any output directory is created.
- **Signatures/mutcat are never copied positionally** — their codes (esp. DBS
  adjacency) are only valid for the full variant set. Pass `reference=` to
  recompute the signatures sidecar on the subset during the rewrite; otherwise
  the output has no mutcat. Explicitly requesting the mutcat field without
  `reference=` raises.
- **`reference` is otherwise not required** (keys are portable; no re-validation
  or left-alignment on a view).
- **Fields**: `None` = carry all `fields` from the source manifest except
  mutcat; `[]` = none; the sidecar values are sliced/re-emitted parallel to the
  new sub-stream record order.
- **Fast path**: a region that covers all variants on a contig **and** keeps all
  samples degenerates to a Component-A directory copy of that contig (no
  rewrite) — the SVAR2 analogue of SVAR1's `_covers_all_variants`.

### CLI (`genoray view` / `concat` / `split`) — breaking for 3.0.0

Mirror the `genoray write` group structure (`python/genoray/_cli/__main__.py:46`):

```python
view = App(name="view", help="Write a subset (region/sample) of an SVAR2 store (SVAR1 via `view svar1`).")
app.command(view)

@view.default
def view_svar2(source, out, *, regions=None, regions_file=None, samples=None,
               samples_file=None, fields=None, reference=None,
               merge_overlapping=False, regions_overlap="pos",
               reroute="auto", overwrite=False, threads=None, progress=False): ...

@view.command(name="svar1")
def view_svar1(...):  # the current `@app.command def view` body, verbatim
    ...
```

- `genoray view` → SVAR2 (`SparseVar2.write_view`). Adds `--reroute/--no-reroute`
  (tri-state incl. `auto`; default `auto`) and `--reference`.
- `genoray view svar1` → the existing SVAR1 logic, moved unchanged (same flags:
  `-r/-R/-s/-S/-f`, `--merge-overlapping`, `--regions-overlap`, `--overwrite`,
  `-@`, `--progress`). No `--reroute`.
- `--regions/--regions-file` and `--samples/--samples-file` stay mutually
  exclusive; the omitted side defaults to "all" (as today).

Two new top-level commands wrapping Component A:

```
genoray concat OUT SRC1 SRC2 ...  [--mode copy|hardlink|symlink|move] [--overwrite]
genoray split  SRC OUT            [--contigs chr1,chr2] [--mode ...] [--overwrite]
```

`genoray split` with `--contigs` → `subset_contigs` into one store at `OUT`;
without `--contigs` → `split_by_contig` exploding into `OUT/{contig}.svar2`.

## Measurement spike — sets the `"auto"` default

Runs **before finalizing the default**, on the real chr21 data (see
`data/README.md` for input + reference paths).

1. Build SVAR2 stores from `data/chr21.bcf` (germline, ref
   `/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa`)
   and `data/gdc.chr21.bcf` (somatic, ref
   `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`). Large (esp. the ~1.1 GB
   somatic) → `sbatch -p carter-compute`.
2. **Analytic recount (no full write needed).** For random sample subsets
   (keep 10 / 50 / 100 / 500 of the >1k samples), recount each variant's carrier
   count `x'` in the subset and compare `choose_representation(x')` to its source
   representation. Report **% of variants that flip** and the **total on-disk
   size delta** (reroute vs no-reroute), per subset size, germline vs somatic.
3. **Decision rule.** A carrier recount over dense variants is needed regardless
   (to drop MAC=0). If flips are rare **and** the size delta is small
   (≈ <1–2 %) across subset sizes, `"auto"` → `reroute=False` (avoids
   dense↔var_key movement for near-zero size cost). If flips or size delta are
   material, `"auto"` → `reroute=True`. Both modes ship regardless; only the
   default is chosen here.

## Testing strategy

**Component A:**

- `split_by_contig` → `concat` round-trip reconstructs a store whose per-contig
  reads (`read_ranges`/`decode`) match the original.
- `concat` raises on mismatched `samples`/`ploidy`/`format_version`/`fields` and
  on overlapping contigs; output `contigs` is `natsort`ed.
- `subset_contigs` narrows `contigs` correctly and rejects unknown contigs.
- Each `mode` yields byte-identical per-contig files (except `symlink`/`move`
  semantics).

**Component B:**

- Region-only view = source restricted to the region (per-hap decode parity).
- Sample subset drops the correct variants and all MAC=0 variants.
- **`reroute=True` produces a store byte-identical to `from_vcf` on the same
  region+sample subset** — the strong parity oracle.
- `reroute=False` decodes identically to `reroute=True` (same genotypes; possibly
  different representation/size).
- Fail-fast: every listed raise fires before the output directory is created;
  `output == source` is rejected.
- Whole-contig + all-samples region hits the copy fast path.
- CLI: `genoray view` targets SVAR2; `genoray view svar1` reproduces the current
  SVAR1 behavior; `--reroute` rejected on `view svar1`.

## Documentation / housekeeping

- `skills/genoray-api/SKILL.md` (**mandatory** — all public surface): `concat`,
  `subset_contigs`, `split_by_contig`, `SparseVar2.write_view`, the `reroute`
  parameter, and the new/restructured CLI (`view` = SVAR2, `view svar1` = legacy,
  `concat`, `split`).
- `docs/roadmap/data-model.md` — update M8 (concat/split) and M9 (region
  subsetting) implementation status.
- `CHANGELOG.md` — entries under `## Unreleased`, flagging the breaking
  `genoray view` restructure for 3.0.0. (Do **not** bump the version by hand.)
- `data/README.md` — records the input + reference FASTA paths (added with this
  spec).

## Scope guards

- **No `from_svar1`** — this builds shared `SparseChunk`-seam groundwork but does
  not implement the SVAR1→SVAR2 native migration.
- **No cross-sample-set concat** — `concat` requires identical sample sets; a
  general N-way merge that re-derives allele frequencies (M12) is out of scope.
- **Biallelic invariant** unchanged (SVAR2 is already biallelic post-atomization).

## Open questions (resolve in the plan)

- Whether the germline `chr21.bcf` needs `--reference` (indel left-alignment) or
  can use `--no-reference` for the spike — confirm against the input's
  normalization state.
- Whether `write_view` chunks by variant range and reuses `merge_mini_sc`
  verbatim, or streams straight into the writer (SVAR2 source is already
  sample-major) — decide after reading `merge.rs`/`writer.rs`, defaulting to
  reusing `merge_mini_sc`.
- Exact `SparseChunk` reconstruction API from a `PyContigReader` (new Rust read
  entrypoint vs. reusing `gather_ranges`).

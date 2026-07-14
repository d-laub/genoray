# SVAR2 `write_view` — unify both backends onto the slicer with a routing policy

**Status:** design (brainstorm output)
**Resolves the `reroute=True` deferred call-outs from PR #105:** INFO/FORMAT field
carry-through; `mutcat` recompute; eager `Vec<RawRecord>` materialization. Also
resolves the `reroute=False` LUT-compaction call-out (it becomes load-bearing here).
**Supersedes:** `2026-07-13-svar2-write-view-fields-design.md` and its plan
`docs/superpowers/plans/2026-07-13-svar2-write-view-fields.md` (fields via the
`reroute=True` re-conversion — never implemented; the re-conversion path is deleted
by this design).
**Builds on:** `2026-07-13-svar2-reroute-false-slicer-design.md` (shipped the slicer).
**Lands in:** PR #105, before merge — the design deletes code that #105 adds, and
shipping a default path documented as OOM-prone only to delete it next PR is churn.

## Goal

Collapse `write_view`'s **two** backends into **one**. The array-slicer
(`src/svar2_slice.rs`, shipped as `reroute=False`) gains a **routing policy**;
`reroute=True` is reimplemented as that policy instead of as a re-run of the
conversion pipeline. The pipeline-backed view path (`Svar2Source`,
`SourceSpec::Svar2`, `run_view_pipeline`'s SVAR2 branch) is deleted.

The public API is **unchanged**: `reroute` still takes `True` / `False` / `"auto"`
with the same meanings (size-optimal re-router vs. representation-preserving
slicer). Only the implementation of `reroute=True` changes — and it gains
INFO/FORMAT carry-through, `mutcat` recompute, and O(output) memory.

## Why unify (the case against fixing `reroute=True` in place)

Current state, verified against the code:

| | `reroute=True` (pipeline) | `reroute=False` (slicer) |
|---|---|---|
| INFO/FORMAT fields | **raises** (`src/lib.rs:479`) | carried at source dtype |
| `mutcat` from `reference` | **discarded** (`let _ = reference;`, `lib.rs:477`) | recomputed (`annotate_contig` post-pass) |
| peak memory | **O(n_variants × n_haps)** — 31 GB measured for a 0.2 GB output | O(output) |
| output size | size-optimal | ≤ +6.6 % (aggressive germline sample-subsets); ~0 % otherwise |

(Memory figure: `docs/superpowers/notes/2026-07-13-svar2-eager-materialization-benchmark.md`
— whole chr21, 3202 samples. Size figures:
`docs/superpowers/notes/2026-07-12-svar2-reroute-measurement.md`.)

So the **default** backend is the worse one on every axis except output size. And
its size-optimality comes *entirely* from re-running `choose_representation` per
variant, which needs nothing from the conversion pipeline except a subset carrier
count — a number the slicer already has in hand from the calls and rows it gathers.
Fixing `reroute=True` in place would mean implementing fields, `mutcat`, and a
streaming rewrite *a second time*, then keeping two variant-selection semantics in
sync forever. Reimplementing it as a routing policy resolves all four deferred
items at once and **removes** a subsystem instead of growing one.

## Architecture

### A. `Routing` policy in `src/svar2_slice.rs`

```rust
enum Routing {
    /// `reroute=False`: a variant's output stream is its source stream.
    Preserve,
    /// `reroute=True`: re-run the cost model against the subset.
    Recompute,
}
```

The slicer's shape becomes **gather → route → emit → post-passes**. Gather
(`slice_var_key_snp` / `slice_var_key_indel` / `slice_dense`) and the post-passes
(LUT, `max_del`, `mutcat`) are unchanged. `Routing::Preserve` makes the route stage
the identity, so today's `reroute=False` behavior is bit-for-bit preserved.

### B. The route stage (`Routing::Recompute`)

Per distinct variant, call
`cost_model::choose_representation(class, n_subset, ploidy, x_sub, sidecar_bits,
info_bits, format_bits)` with **exactly** the inputs `rvk.rs:262-278` feeds during
`from_vcf`, so a view's routing equals a fresh conversion of the same subset:

- **`x_sub`** — subset carrier haps. var_key: the count of kept calls, grouped by
  `(pos, key)` across the kept hap-columns. dense: a popcount of the subset's bits
  in that row. (This is the same recount the reroute spike validated analytically —
  `src_mismatch = 0` on every row of its control.)
- **`n_samples`** — the **subset** size, not the source cohort. This is the whole
  driver: the dense cost carries an `n_samples·ploidy` term, so shrinking the cohort
  moves the crossover.
- **`sidecar_bits`** — `SIDECAR_BITS_SNP` / `SIDECAR_BITS_INDEL` iff `mutcat` will
  exist in the output, i.e. iff `reference` was given; else 0. The old pipeline path
  passed 0 unconditionally because it *discarded* `reference`, so this is strictly
  more correct than what it replaces.
- **`info_bits` / `format_bits`** — summed `StorageDtype::width_bytes() * 8`
  (`src/field.rs:38`) over the carried INFO / FORMAT fields, mirroring
  `rvk.rs:230-238`. The slicer's `FieldSpec`s already carry concrete (never `Auto`)
  dtypes, so the `unwrap_or(4)` staging fallback in `rvk.rs` is unreachable here.

**Consequence to document:** routing depends on *which fields are carried* and on
*whether a reference was given*. Two views over the same region/samples that carry
different fields may route the same variant differently. This is exactly `from_vcf`'s
behavior, not a new quirk.

### C. Flip handling — the only genuinely new code

- **var_key → dense.** Build a hap-major bit row from the variant's carrier columns
  (`bits::set_bit`, `hap*n_dense + col`). INFO: the variant's single value. FORMAT:
  a full `n_subset`-wide column — carriers get their per-call value, **non-carriers
  get `FieldSpec::missing_sentinel()`** (`src/field.rs:121`). This is the identical
  fill `rvk.rs` performs on its dense push.
- **dense → var_key.** Scan the row's subset bits into per-column calls. INFO:
  the row's value, repeated per call. FORMAT: `format_at(row, orig_sample)` per
  carrier call. **Non-carrier FORMAT values are dropped** — see *Semantics* below.

**Emit order.** A flipped variant merges into the opposite stream at its position in
ascending `(pos, key)` order. Deterministic. Note the *shipped* `reroute=True` also
re-orders same-position variants (its `Svar2Source` emits from a
`BTreeMap<(pos, ilen, alt), …>`, i.e. sorted, not source-VCF order), so a
`(pos, key)` merge is not a new-in-kind property. Same-position ties are the only
case where the byte layout may differ from a fresh `from_vcf`; `reroute=True` is
therefore verified by decode-equivalence + routing-equality + size, **not** by
byte-parity against `from_vcf`. (The full-coverage identity slice *is* byte-parity —
see *Testing*.)

**Memory.** Every route-stage structure is sized to the output: the gathered calls
(O(kept calls)), the kept dense bit rows (O(kept rows × n_subset haps) = the dense
output itself), and a `(pos, key) → x_sub` map (O(distinct kept variants)). A
var_key→dense flip produces one bit row; a dense→var_key flip produces ≤ `x_sub`
calls. So `reroute=True` becomes **O(output)**, like `reroute=False`.

### D. LUT compaction (now load-bearing, not optional)

The deleted pipeline rebuilt the long-allele LUT from scratch, so its output was
compacted. The slicer copies `indel/long_alleles.*` **verbatim** (valid — indel keys
are absolute row indices, and a subset can only leave rows unreferenced). If
`reroute=True` inherits the verbatim copy, a heavily-subset view can carry dead LUT
rows and come out **larger** than the path it replaces — regressing the one axis
`reroute=True` exists for. So the slicer gains a compaction pass: scan the written
indel key streams → collect referenced LUT rows → rebuild `long_alleles.bin` +
`long_allele_offsets.npy` → remap the `Lookup{row}` keys.

Applied under **both** routings (strictly smaller, decode-identical). It does not
break the identity-slice byte-parity test: at full coverage every LUT row is
referenced, so compaction is the identity.

**Cut line.** If scope bloats, this is the first thing to drop — fall back to the
verbatim copy and document that a re-routed view may be slightly larger than a fresh
`from_vcf`. Everything else in this design stands without it.

### E. Deletions and the surviving shared predicate

Deleted: `Svar2Source` (the `RecordSource` impl, `src/svar2_source.rs`),
`SourceSpec::Svar2` (`src/orchestrator.rs:271-283`), and `run_view_pipeline`'s SVAR2
branch (`src/lib.rs`). The **SVAR1** view path (`SourceSpec::Svar1`,
`src/svar1_reader.rs`) is untouched.

Surviving: the shared selection predicate — `OverlapMode`, `query_window`, `keeps`,
`read_n_samples`. This is what guarantees both routings select an **identical variant
set** (including the per-element `q_start < v_end` indel extent re-check that #105's
review caught), so it must not be duplicated. It moves to **`src/svar2_view.rs`**,
since "source" stops meaning anything once the `RecordSource` is gone. Update the
`lib.rs` / `orchestrator.rs` / `svar2_slice.rs` imports accordingly.

### F. Python `write_view` (`python/genoray/_svar2.py`)

- `reroute=False` and `reroute=True` **both** dispatch to `run_slice_view`, passing
  the routing policy. `run_view_pipeline` is no longer called for SVAR2.
- `run_slice_view` gains a `reroute: bool` parameter (Rust-side `Routing`).
- `fields=` / `reference=` resolution is unchanged (it already works for the slicer),
  and now applies to `reroute=True` as well — the `fields`→`ValueError` guard on the
  re-conversion path disappears with the path.
- **`max_threads`**: the pipeline's thread pool goes away with it. Keep the kwarg
  only if the slicer parallelizes across contigs (see *Testing*, benchmark gate);
  otherwise remove it from `run_slice_view`'s surface rather than accept-and-ignore
  it. Decide from the benchmark, not up front.

## Semantics

### The `"auto"` rule (the one user-visible behavior change)

A **dense → var_key flip loses non-carrier FORMAT values**: the var_key
representation stores one value per *carrier call* and has no slot for a non-carrier
sample. This is not a bug — `from_vcf` has the identical property for any var_key
variant — but *subsetting can newly trigger it* on a variant that was dense (and so
had a full per-sample FORMAT column) in the source. `reroute=True` is the default, so
the default could silently drop data.

Therefore:

> **`reroute="auto"` resolves to `False` when any FORMAT field is carried, and to
> `True` otherwise.**

Genotype-only and INFO-only views — which is where the ≤6.6 % size win actually lives
— stay size-optimal. Views carrying FORMAT default to lossless. An explicit
`reroute=True` still honors the request and performs the flip.

**This rule is only acceptable if it is documented.** It MUST appear, in the same PR,
in: the `write_view` docstring (the `reroute` parameter), `skills/genoray-api/SKILL.md`,
`CHANGELOG.md` (`## Unreleased`), and the `genoray view` CLI help for `--reroute` /
`--no-reroute`. Each mention must state both halves — the resolution rule *and* the
non-carrier-FORMAT loss it exists to avoid.

### Other semantics (unchanged from the shipped slicer)

- **Identical variant set** across both routings for the same region / samples /
  overlap mode (shared predicate). Only the on-disk *representation* differs.
- **INFO**: carried verbatim at the source dtype under both routings (a flip never
  changes an INFO value — it is per-variant in both representations).
- **`mutcat`**: recomputed from `reference` on the subset under both routings, or
  omitted. Never copied positionally (DBS adjacency is only valid for the full variant
  set). Presence stays on-disk-only; nothing is stamped into `meta.json`.
- **MAC=0** variants dropped under both routings.
- **Fail-fast band ordering**: every raise (reroute validation, `mutcat`-without-
  `reference`, unknown contig/sample, reference faidx validation) happens before any
  output byte is written. Unchanged.

## Testing strategy

1. **Identity slice, both routings (the strongest test).** A full-region /
   all-sample view reproduces the source store's sidecars **byte-for-byte**
   (positions / alleles / offsets / genotypes / `max_del` / fields / LUT). This now
   covers `reroute=True`: at full coverage `x_sub == x_full`, so zero variants flip
   and `Recompute` degenerates to `Preserve`. It is also what makes LUT compaction
   safe (all rows referenced ⇒ compaction is the identity).
2. **Differential against the old pipeline — run BEFORE deleting it.** This is the
   assurance that we do not regress the mode we are reimplementing. For a real subset,
   new `reroute=True` must match old `reroute=True` on: per-variant routing (via the
   existing `_core.svar2_variant_stats` `src_dense` flags), decoded genotypes, and
   output size (new ≤ old, modulo the LUT). Only then delete `Svar2Source`. Sequence
   this as its own plan task.
3. **Flip-specific, both directions.** A fixture with a variant that provably flips
   each way. Assert the output representation (`svar2_variant_stats`), decode
   equivalence, and field values — including the non-carrier `missing_sentinel()` fill
   on var_key→dense and the **documented** non-carrier drop on dense→var_key.
4. **Routing depends on carried fields.** The same subset, carried with vs. without a
   wide FORMAT field, routes at least one variant differently.
5. **Routing depends on `reference`.** `reference=` given ⇒ `sidecar_bits` nonzero ⇒
   routing can differ from the no-reference view. Assert.
6. **`"auto"` resolution.** `reroute="auto"` + a FORMAT field ⇒ representation
   preserved (slicer, `Preserve`); `"auto"` + genotypes-only or INFO-only ⇒ re-routed.
7. **Overlap-mode parity.** `pos` / `record` / `variant` select the same variant set
   under both routings (shared predicate) — parametrized, as today.
8. **Benchmark gate.** Rerun `scripts/svar2_eager_bench.py` against the new
   `reroute=True` on `data/chr21.germline.svar2`. Expect peak RSS to collapse from
   ~31 GB to O(output). Wall time should also drop (a gather beats a full
   re-conversion), but the old path had `max_threads` pipeline parallelism and the
   slicer is serial — **if wall time regresses, parallelize the slicer across contigs
   with rayon** and only then decide whether `max_threads` stays on the API (§F).
   Update the benchmark note with the new numbers and retire its "do not advertise
   cohort-scale whole-store copies" gate.

Rust unit tests for the route stage and the flip re-shaping; Python integration tests
extend `tests/test_svar2_write_view.py`.

## Public API / docs

The `write_view` / `genoray view` **signature** is unchanged. Behavior changes:
`reroute=True` now carries `fields=` and honors `reference=` (both were errors /
no-ops), and `"auto"` resolves per the rule above. Update in the same PR:

- `python/genoray/_svar2.py` — `write_view` docstring (`reroute`, `fields`, `reference`).
- `skills/genoray-api/SKILL.md` — mandatory per CLAUDE.md (public-name semantics change).
- `CHANGELOG.md` — `## Unreleased`.
- `docs/roadmap/data-model.md` — M9.
- `genoray view` CLI help (`--reroute` / `--no-reroute`).
- PR #105's "Deliberately deferred" section — all four call-outs are resolved.
- Mark `2026-07-13-svar2-write-view-fields-design.md` and its plan **superseded**.

## Non-goals / deferred

- **Streaming anything.** There is nothing left to stream: the pipeline-backed view
  path is gone and the slicer is already O(output).
- **Non-scalar / String fields** — SVAR2 fields stay scalar-numeric + INFO Flag, as in
  `from_vcf`.
- **A full-coverage fast path** — a whole-store, all-sample view still slices rather
  than degenerating to a Component-A file copy. Unchanged from the parent specs.
- **Rescuing non-carrier FORMAT across a dense→var_key flip.** Not possible in the
  var_key representation; the `"auto"` rule steers around it instead.

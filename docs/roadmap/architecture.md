# SVAR 2.0 Software Architecture

> Supplement to [`svar-2.md`](svar-2.md). Describes the conversion pipeline, the
> abstraction seam that keeps orchestration independent of the inline encoding, the
> query algorithm, and the Python decode path. Described in terms of **contracts**, not
> the current scratch code — the implementation on this branch is a starting point, not
> a spec.

## Goals

- **Streaming and bounded memory.** Conversion must handle larger-than-RAM inputs by
  processing variants in chunks with backpressure between stages.
- **Encoding-agnostic orchestration.** The pipeline must not bake in a specific inline
  bit layout, so we can change the encoding during experimentation without touching
  orchestration. See [the seam](#the-encoding-agnostic-seam).
- **Per-contig independence.** Contigs convert and store independently, enabling
  parallelism and cheap merge/split (roadmap M8).

## Conversion pipeline

VCF → SVAR2 is a multi-threaded, channel-connected dataflow. Each stage is a thread
(or thread pool) communicating over bounded channels so a slow stage applies
backpressure upstream.

```
                 bounded channels (backpressure)
 ┌──────────┐   dense   ┌───────────┐  sparse  ┌──────────────┐
 │  Reader  ├──────────►│ Compute / ├─────────►│   Writer(s)  │
 │ (htslib) │   chunks  │  Encode   │ payloads │  pos / keys  │
 └──────────┘           └─────┬─────┘          └──────────────┘
                              │ long alleles
                              ▼
                        ┌───────────┐
                        │ LUT writer│
                        └───────────┘

 ── then ──►  Phase 2: K-way merge into per-(sample, ploid) sorted streams
```

1. **Reader** — pulls variant records for one contig from the VCF/BCF (via htslib),
   builds dense genotype chunks (variant-major), and emits them.
2. **Compute / Encode** — normalizes variants ([`data-model.md`](data-model.md#variant-normalization)),
   transposes dense → sparse (sample-major), and encodes each call's variant info via
   the [encoding seam](#the-encoding-agnostic-seam). Alleles that don't fit inline are
   pushed to the LUT.
3. **Writers** — stream the sparse payloads (positions, keys) and the LUT buffers to
   disk sequentially.
4. **Phase 2 merge** — a K-way merge consolidates per-chunk outputs into the final
   per-`(sample, ploid)` position-sorted streams and the offsets sidecar.

Per-variant **routing to `var_key` / `dense` / (`pointer`)** is driven by the
[cost model](data-model.md#dense-vs-sparse-cost-model). The MVP routes between
`var_key` and `dense`.

## The encoding-agnostic seam

This is the load-bearing design constraint. Orchestration (chunking, transpose,
routing, writing, merging) must treat the inline variant encoding as **opaque
fixed-width bits**. Only a thin encode/decode layer knows the bit layout in
[`data-model.md`](data-model.md#inline-variant-encoding-var_key).

Contract for the encoder:

- **Input:** `(ILEN, ALT bytes)` plus a handle to the LUT for spill.
- **Output:** a fixed-width key (e.g. `u32`) — inline value or LUT pointer, with the
  flag bit set per the layout.
- **Constraint:** as long as the key width is unchanged, the encoding internals can be
  swapped freely; orchestration code does not branch on the layout.

A matching decoder reverses it: `key (+ LUT) → (ILEN, ALT)`. Keep these two functions
(and their width constant) as the *only* place that knows the layout. The cost model
and on-disk layout reference the encoding solely through `s` (bytes per variant info),
not through any specific field positions.

> If you find orchestration code branching on SNP-vs-indel bit positions, that's a
> leak of the seam — push it back into the encode/decode layer.

## On-disk layout

See [`data-model.md`](data-model.md#on-disk-layout) for the full tree. Key points for
the reader/writer:

- One directory per contig; positions are sidecar arrays sorted within a contig.
- Each contig has up to three representation subdirs (`var_key`, `pointer`, `dense`);
  a variant lives in exactly one.
- `meta.json` holds version (for `SparseVar2` negotiation), samples, contigs, and
  ploidy. The per-`(contig, sample, ploid)` maximum deletion length for overlap queries
  is stored separately in a per-contig `max_del.npy` (it is structured and potentially
  large), not in `meta.json`.

`SparseVar2` (in `genoray`, alongside the existing `SparseVar`) memory-maps these
sidecars so genotype data stays on disk until accessed, matching the SVAR 1.0 access
model.

## Query path

`(range, sample)` queries (roadmap M5) resolve overlaps, not point hits, because
deletions span reference bases.

- **Index structure:** binary search over the sorted position sidecar, starting from
  the [left-tree static search tree](https://curiouscoding.nl/posts/static-search-tree/#left-tree)
  for cache-friendly lookups.
- **Overlap handling:** use the max-deletion-length bound to set the lower search
  bound, then linear-scan to the upper bound, per the algorithm in
  [`data-model.md`](data-model.md#overlap-queries-and-deletions).
- **Across representations:** a query may touch variants in `var_key`, `pointer`, and
  `dense` subdirs for the same contig; results from each must be combined in
  position-sorted order (next section).

## Python decode path

Query results (roadmap M6) decode into user-facing structs/classes (exact API TBD).
Because a single contig's variants are spread across up to three representations — each
an independent, position-sorted source — assembling a result requires a **fast sorted
union / merge** of multiple sorted streams.

- This is the union-shaped dual of fast sorted-array **intersection**; the techniques
  in [Doug Turnbull's "Faster intersect"](https://softwaredoug.com/blog/2024/05/05/faster-intersect)
  (branch-light, SIMD-friendly merging of sorted integer arrays) are good inspiration
  for the union we need.
- The decoder reverses the [encoding seam](#the-encoding-agnostic-seam):
  `key (+ LUT) → (ILEN, ALT)`, reconstructing the ALT against the reference and
  position.

## Open questions

- **Routing granularity.** Is representation chosen strictly per variant, or can a
  contig be forced into a single representation for simplicity in early milestones?
- **Merge stage memory.** Bounds and chunking strategy for the Phase-2 K-way merge at
  scale (many samples × many chunks).
- **PGEN path (M7).** Whether the htslib reader stage is cleanly swappable for a
  `pgenlib` FFI source behind the same chunk contract.

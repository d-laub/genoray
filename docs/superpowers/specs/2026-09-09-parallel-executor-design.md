# A deterministic parallel executor for cohort-width conversion

Status: approved design, revised after adversarial code review; not yet implemented.
Issue: [#176](https://github.com/d-laub/genoray/issues/176).
Depends on: the `Tuning` API (`2026-09-08-explicit-tuning-api-design.md`), unmerged.

## Problem

At All of Us width (S = 535,662 diploid samples, 1,071,324 haplotype columns)
`SparseVar2.from_vcf` is bound by the single per-contig executor thread. From two
production conversions on genoray 4.0.1 (`reader_workers=20`, `overshard=40`,
`chunk_size=5000`, n2-highmem-64):

| conversion | elapsed | `exec-<chrom>` CPU | exec busy | each of 20 readers |
|---|---|---|---|---|
| chr12 | 58,486 s | 56,915 s | **97 %** | ~21,100 s (~36 %) |
| chr11 | 6,591 s | 6,120 s | **93 %** | ~1,400 s (~21 %) |

The monitor line confirms the shape: `dense=6 dense_cap=6` (executor's input queue
full), `pending=100` units ≈ 54 GB queued, `cpu_exec=100%`, `cpu_shard=1983%`,
`cpu_cw=0%`. Twenty readers are producing work at ~20 cores that one core cannot
consume.

This is the exact opposite of the measurement behind `754e255` ("drop the
parallel-executor prototype"), which found exec ~7 % busy and readers at 80 % of
all CPU — but at S = 1,000 and S = 16,000. Both measurements are correct; the
balance point moves with cohort width.

Executor cost per variant is also not stable across runs on the same host:
~1.6 ms/variant on an idle box (chr19) versus ~7.7 ms/variant with a co-tenant
(chr12).

### Which function is actually hot

`SparseVar2.from_vcf` (`python/genoray/_svar2.py:645` → `_core.run_conversion_pipeline`)
reads **one multi-sample VCF**. That source produces `Calls::Dense`
(`vcf_reader.rs:699`) → `RecordCalls::Masks` (`chunk_assembler.rs:582-608`) →
`AtomCalls::Masks` → `carriers = None` (`chunk_assembler.rs:472`; the dispatch is
spelled out in `DenseChunk::carriers`' own doc comment, `types.rs:174-181`). So
`dense2sparse_vk` falls straight through to **`dense2sparse_vk_by_scan`**
(`rvk.rs:589-596`). The carrier-driven arm of `dense2sparse_vk` runs only for
`from_vcf_list` (the k-way merge over single-sample VCFs).

`by_scan`'s emission loop is variant-inner over the full grid
(`rvk.rs:519-540`):

```rust
for s in 0..num_samples { for p in 0..ploidy { for v in 0..v_variants {
    let flat_idx = (v * columns) + s * ploidy + p;   // stride = columns BITS
    let word = unsafe { *words.get_unchecked(flat_idx >> 6) };
    if (word >> (flat_idx & 63)) & 1 == 0 { continue; }
    ...
} } }
```

That is `columns × v_variants` = 1,071,324 × 5,000 = **5.36 × 10⁹ single-bit
tests per chunk**, one strided load each. At ~1.5 ns/iteration that is 8.0 s per
chunk = **1.6 ms/variant** — exactly the issue's idle-box chr19 figure; at ~7 ns
under cache pressure it is 7.7 ms/variant — exactly the co-tenant chr12 figure.
Nothing on the carrier-driven path has that shape. The issue's "memory-bandwidth
and cache bound" description is a description of *this* loop, and it is what any
change here has to move.

`754e255` removed the prototype for two stated reasons. Reason (a), "the executor
is not the bottleneck", is false at cohort width. Reason (b), "output was
non-deterministic at every `GENORAY_EXEC_WORKERS > 1`", was a property of *that*
design — it parallelised across whole chunks, so several executors pushed into
one long-allele bank concurrently and `merge_bank_offsets` had to reconcile them.
The design below never splits the bank, so (b) cannot recur: `emit_call`
(`rvk.rs:169`) takes no `bank` parameter at all, and every
`LongAlleleTableWriter` write happens in serial `route_variants` (`rvk.rs:249`,
via `classify_variant`/`pack_variant`).

## Approach

Keep **one** `exec-{chrom}` OS thread per contig. Split the work inside both
emission loops along the axis that grows with cohort width — the haplotype
column — and leave every per-variant decision serial.

Per chunk, the executor:

1. Runs `route_variants` **serially** (`rvk.rs:249`). Per-variant work: allele
   classification, `choose_representation`, dense table sizing, and the only
   writes to `LongAlleleTableWriter`. See "The serial remainder" below for the
   one case where this is *not* O(V).
2. Fans the per-column emission across a rayon pool of `exec_workers` threads
   over a **fixed** partition of `0..columns` (see "Slice partition").
3. Concatenates each slice's output in slice order.

The bank is never shared. There is no `merge_bank_offsets`. There is one dense
input queue, one `SparseChunk` per chunk, and one writer — the pipeline's shape
is unchanged outside `dense2sparse_vk*`.

### Why not one executor per sample slice

Issue #176 proposes N executor threads, one per fixed sample slice, each
consuming whole `DenseChunk`s. That would give every executor its own bank (the
`754e255` failure mode, or an offset-merge to avoid it), duplicate `route_variants`
N times per chunk, and multiply the dense-queue fan-out. Slicing inside one
executor gets the same near-linear scaling on the per-sample work with none of
that.

### Do the serial win first

Before any fan-out, restructure `by_scan`'s loop to be **64-column-block outer,
variant inner**, testing one `u64` word per (variant, block) and skipping it
whole when zero, then emitting per column within the block.
`bits::for_each_set_bit` (`bits.rs:79`) already implements exactly that
scan-and-skip idiom over a contiguous window.

Work per chunk drops from `V × columns` bit tests (5.36 × 10⁹) to
`V × columns/64` word loads (83.7 × 10⁶) plus one extraction per set bit. The
number of *nonzero* words per variant is ≈ its allele count for any variant with
`x ≪ columns/64` (16,739 at AoU width) — i.e. for the overwhelming majority of a
biobank cohort's variants. Expected serial win: order 60×, which is larger than
any plausible `exec_workers` and composes with it. Common variants gain nothing
(their words are mostly nonzero) but they route Dense and cost O(x) either way.

**This is a prerequisite measurement, not an optional extra.** The bench in
"Measurement" runs the blocked-serial arm first; if it lands near 60×, the
fan-out's remaining value is much smaller than this document assumes and the
`e`-derivation below should be re-argued before it ships.

## Determinism

Output is byte-identical to the serial path **by construction**, not by test.

| Output | Why it cannot vary |
|---|---|
| Long-allele bank (`nrvk`) | Written only by serial `route_variants`; `emit_call` has no `bank` parameter (`rvk.rs:169`), so workers cannot reach it. |
| Slice boundaries | A pure function of `columns` and `exec_workers` (see "Slice partition"). No work stealing decides them; rayon only decides *when* a slice runs. |
| `call_positions`, `call_keys`, `field_calls` | Each worker fills its own `SparseSubStream`; the executor concatenates in slice order. The concatenation is the serial column loop's byte sequence. |
| `sample_lengths` | Same, **provided** every worker pushes exactly `hi - lo` entries for **every** `StreamTag`, including streams and slices with zero calls. A short push does not corrupt one value, it shifts the whole ledger (`merge`'s per-column arithmetic reads `sample_lengths` positionally). Asserted per slice, not just on the concatenation. |
| Dense `geno_bits` | Slice boundaries are multiples of 8 columns, so each slice's byte range is disjoint. Workers write into `split_at_mut` views of the single preallocated buffer — nothing to merge. |
| `Phase1Output` ledgers | Built from the concatenated streams after the fan-out joins (`executor.rs:80-86`). |
| FORMAT values | `resolve_format`/`CarrierFormat::value` are pure `&self` reads over an `Arc<FormatVals>`. No interior mutability, nothing to order. |

### The column-alignment argument

`emit_call`'s dense arm writes bit `hap * n_dense_variants + col`
(`rvk.rs:224`), where `hap` **is** the haplotype column: `by_scan` passes
`hap = s * ploidy + p` (`rvk.rs:523`) and the carrier arm passes `column`
(`rvk.rs:686-694`). A column slice `[lo, hi)` therefore owns the contiguous bit
range `[lo · n_dense_variants, hi · n_dense_variants)`.

`geno_bits` is a `Vec<u8>` (`types.rs:287`) and `bits::set_bit` is byte-addressed
(`bits.rs:7`), so a `split_at_mut` boundary needs
`lo · n_dense_variants ≡ 0 (mod 8)`. `lo ≡ 0 (mod 8)` gives that for **any**
`n_dense_variants` — so the alignment holds for both dense classes
simultaneously, and for every chunk, without knowing either class's dense count
in advance. The final slice's end is `columns`, which need not be aligned; the
buffer is already `div_ceil(8)`-sized for that tail (`rvk.rs:396`).

Three consequences, stated explicitly because an earlier draft of this spec got
all three wrong:

- **8, not 64.** 64 is correct but 8× stricter than needed, and the only thing
  it costs is the `exec_workers` cap — which is what makes the existing test
  fixtures (8–1024 columns) silently clamp `exec_workers` to 1 and pass
  vacuously. Cap is `columns / 8`.
- **8 does not buy false-sharing freedom, and neither does 64.** Line-disjoint
  boundaries would need `lo ≡ 0 (mod 512)`. We accept the sharing: it is at most
  `2(e − 1)` boundary cache lines per dense class per chunk, each touched by two
  workers — order 10⁴ line transfers against 10⁸–10⁹ loop iterations.
- **`set_bit` takes an index into the slice it is handed.** A worker's bit index
  must be rebased to `hap * n_dense_variants + col − lo * n_dense_variants`.

### `emit_call`'s cross-worker state

The bank is *not* the only hazard. `emit_call` mutably borrows the whole
`DenseMap<DenseSubChunk>` and does `dense.get_mut(*class)` (`rvk.rs:222-225`),
which both **reads** `sub.n_dense_variants` and **writes** `sub.geno_bits`. Its
signature must change so a worker receives, per dense class, a `&mut [u8]`
window plus the `n_dense_variants` scalar and the rebasing offset — not the map.

`emit_call` is shared verbatim between the two emission loops on purpose, so this
signature change lands on both. Full inventory:

| State `emit_call` touches | Disposition |
|---|---|
| `streams: &mut StreamMap<SparseSubStream>` | per-slice instance |
| `counts: &mut StreamMap<u32>` | per-column local (`rvk.rs:521`, `rvk.rs:684`), already per-slice |
| `dense: &mut DenseMap<DenseSubChunk>` | **replaced** by per-class `(&mut [u8], n_dense_variants, bit_offset)` |
| `chunk: &DenseChunk` (incl. `format_by_carrier`) | shared immutable read |
| `per_cat`, `format_specs` | shared immutable read |
| bank | not reachable |

## Slice partition

`columns` splits into `n_slices` half-open ranges. Boundary rule, stated once so
Test group 5 has something to assert:

```
width  = max(8, ((columns / n_slices) / 8) * 8)     // rounded DOWN to a multiple of 8
lo_i   = min(i * width, columns)
hi_i   = if i == n_slices - 1 { columns } else { min((i+1) * width, columns) }
```

`n_slices` is capped at `columns / 8` so no slice is empty by construction, and
the last slice absorbs the remainder (at most `n_slices + 7` extra columns —
bounded, unlike a round-up rule's fat tail).

**Over-decompose:** `n_slices = 4 · exec_workers`, not `exec_workers`.
Determinism needs the *slices* fixed and concatenated in index order; it does not
need one slice per thread. Four slices per worker lets rayon steal, which matters
on the carrier-driven path where carrier density is not uniform across samples
(equal-width slices are perfectly balanced for `by_scan`, whose per-column cost is
`V` regardless, but not for `from_vcf_list`). Cost of over-decomposition: `4e`
`SparseSubStream` sets instead of `e` — see "Per-slice reservations".

## Per-slice reservations

`SparseSubStream::with_capacity(key_bytes, nnz, columns)` (`types.rs:259-266`) is
called with `estimated_nnz = v_variants * columns / 20` (`rvk.rs:504`,
`rvk.rs:648`). At V = 5,000 and columns = 1,071,324 that is 267,831,000 —
`call_positions` reserves **1.07 GB** and `call_keys` 0.27 GB (SNP) / 1.07 GB
(indel), *per stream, per chunk, today*. Naively reusing those arguments per
slice multiplies that by `n_slices`.

Per-slice construction therefore **must** be sized on the slice:

```
nnz_i         = v_variants * (hi_i - lo_i) / 20
sample_lens_i = hi_i - lo_i
```

which keeps the chunk total where it is today. `route_variants`' own comment
(`rvk.rs:388-396`) records that a large per-chunk address-space reservation was a
real bug worth fixing; do not reintroduce it `n_slices`-fold.

(The `/20` heuristic is itself wrong by orders of magnitude at cohort width. Out
of scope here; spun out as a follow-up.)

## Concatenation cost

Quantified rather than waved. Per chunk at S = 535,662, `chunk_size = 5000`:

| buffer | bytes |
|---|---|
| `sample_lengths`, 2 streams × `columns` × 4 B | 8.57 MB |
| `call_positions`, 4 B/call | 4 B × calls |
| `call_keys`, 1 B (SNP) / 4 B (indel) per call | ≤ 4 B × calls |
| `field_calls`, 4 B × F per call | 4F B × calls |

At ~5 × 10⁵ calls/chunk with no FORMAT fields that is ~11 MB, ≈ 1.1 ms at
10 GB/s, against a chunk that costs seconds — **under 0.1 %**. Concatenation is
cheap; the transient double-buffering is ~11 MB. No segmentation needed.

Segmentation *would* be byte-identical if it were ever wanted: `write_bin` does
one `File::create` plus one `write_all` (`writer.rs:102-110`), and N ordered
`write_all`s produce the same file. It is not worth it: `sample_lengths` never
reaches the writer — the executor clones it into `var_key_ledgers`
(`executor.rs:84-86`) — so segmenting would have to change the ledger shape as
well, for a sub-0.1 % saving.

## Per-slice counting sort (carrier-driven path only)

Applies to `dense2sparse_vk`'s carrier arm — i.e. `from_vcf_list`, **not** the
`from_vcf` workload in the Problem section. Kept in scope because the fan-out
lands on both loops and this is the carrier arm's shared-state equivalent.

The current sort is shared and cohort-width-hostile (`rvk.rs:663-679`):

```rust
let mut counts_per_col = vec![0u32; columns + 1];   // 4.29 MB at S = 535k
...
let mut cursor = counts_per_col.clone();            // another 4.29 MB
let mut by_col: Vec<u32> = vec![0u32; total];       // scattered writes
```

Both passes scatter across a 4.29 MB array — larger than L2, and thrashing L3
against 20 concurrently decompressing readers.

Each worker can instead find its own window of every variant's carrier list with
`partition_point` and run a counting sort sized to *its* columns only:

- per worker: O(V · log C) to locate windows, plus O(carriers/e + columns/e)
- no shared arrays, no cross-worker synchronisation
- working set per worker divided by `e`

Two preconditions the current code does **not** give us for free:

1. **`Carriers::cols` is private** (`record_source.rs:53`); `iter()`
   (`record_source.rs:85`) is the only accessor. Needs a
   `pub fn cols(&self) -> &[u32]`.
2. **`partition_point` makes a debug-only invariant load-bearing in release.**
   The counting sort above is order-*independent*: it buckets by column, so an
   unsorted or duplicated carrier list still yields correct column-ordered
   output, and an out-of-range column *panics* on `counts_per_col[col + 1]`
   (`rvk.rs:666`). `partition_point` changes both — an unsorted list silently
   drops calls, and an out-of-range column silently falls into no slice. The
   ascending invariant is enforced only by a `debug_assert` in `push`
   (`record_source.rs:66-74`), compiled out in release, and it has already been
   violated by real input once: `62c78f6` ("collapse duplicate cols in
   from_vcf_list carrier merge"), whose message notes release builds "would read
   back an unspecified genotype". Duplicates are in fact harmless for
   `partition_point` (equal keys stay adjacent); **ordering and range are not.**

   Required: a release-mode guard. Cheapest sufficient form is a single
   `is_sorted()` + `last() < columns` check per variant carrier list, done once
   in the serial pre-pass (O(carriers), already paid by `carrier_count`), failing
   loudly rather than silently dropping calls.

The cache effect is expected to be a material part of the win on this path,
separate from the core count. The bench in "Measurement" separates the two.

## The serial remainder

`route_variants` is O(V) **only when no FORMAT fields are requested.** Its second
pass (`rvk.rs:426-440`) walks `routes` and, for each `Route::Dense` variant and
each FORMAT field, fills a full per-sample column:

```rust
for &(is_format, idx) in &per_cat {
    if is_format { for s in 0..num_samples { ... } }   // O(V_dense × S × F)
```

With no FORMAT fields, `per_cat` has no `is_format` entry and the `s` loop never
runs — the pass degenerates to O(V) plus one INFO push per dense variant. Not a
literal no-op, but not cohort-scaling.

With FORMAT fields it is O(V_dense × S × F) and stays serial: at S = 535,662,
V_dense = 100/chunk, F = 2 that is 1.07 × 10⁸ `resolve_format`/`is_carrier` calls
per chunk (~0.5–2 s), and it becomes the Amdahl term that caps this whole change.

**Precondition, stated up front:** the speedup this design claims holds for
conversions that request **no FORMAT fields**. Confirm against the target run's
`format_fields` before promising a number. Slicing that pass is a different
change (variant-major output layout); spun out as a follow-up.

Other per-chunk work that scales with `columns` rather than `V`, checked and
found benign: `sub.geno_bits = vec![0u8; bits.div_ceil(8)]` (`rvk.rs:396`) is
`alloc_zeroed`, i.e. lazily-faulted address space, and the pages it *does* fault
are faulted inside the slice that writes them, so that cost parallelises.

## Scope boundaries

**In scope**

- `dense2sparse_vk_by_scan` (`rvk.rs:474`) — the multi-sample-VCF and PGEN path,
  and the one issue #176 measured. Gets the blocked-serial rewrite and the column
  fan-out. It needs **no** counting sort: its columns are already the outer loop.
- `dense2sparse_vk`'s carrier arm (`rvk.rs:589+`) — `from_vcf_list`. Gets the
  same fan-out plus the per-slice counting sort.
- `emit_call`'s dense-arm signature (shared by both).
- `plan_sharded` core budgeting, and the `Tuning` field that overrides it.
- The monitor's `cpu_exec` attribution, which reads a `TidRegistry`
  (`monitor.rs:43`, `monitor.rs:303`); worker TIDs must register or `cpu_exec`
  will under-report the stage by a factor of `exec_workers`.

**Out of scope, deliberately**

- The dense-FORMAT second pass in `route_variants` — see "The serial remainder".
- `SparseSubStream`'s `estimated_nnz = V × columns / 20` heuristic.
- Reader/executor back-pressure. #174 already landed it upstream; it is unreleased
  (tag 4.0.2 predates the merge). Cutting a release is the fix for issue #176's
  item 1 and needs no code.

**The oracle problem this creates.** Today's differential tests
(`rvk.rs:937`, `rvk.rs:989`) drive the same data through the carrier arm and
through `by_scan` and assert byte-identical `SparseChunk`s. Once *both* are
parallelised, that test no longer has an independent serial oracle — it would
compare two parallel implementations. Replace the oracle with a `#[cfg(test)]`-only,
deliberately naive serial reference emitter (the current `by_scan` body, frozen),
and keep the cross-arm test as a second, weaker check. Without this, a shared bug
in the fan-out passes every existing assertion.

## Thread budget

`plan_sharded` (`budget.rs:548`) currently hardcodes per-contig CPU demand as
`1 + reader_workers` (`budget.rs:559`, `budget.rs:572-577`), with a doc comment
asserting the executor is "a serial recv loop, pegged at ~100 % of one core".
That comment becomes false. The change:

```
core_bound = pool / (exec_workers + reader_workers)        // pool per budget.rs:106
processing_threads_for(usable, cc, e, w) = usable - cc * (e + w), floored at 1
```

`exec_workers` defaults to a **fitted** function of cohort width, not a guessed
constant. Define `r(S)` = (single-core executor cost for one chunk) / (single-core
reader cost for one chunk) at cohort width S. `r` is measured — the executor half
by the microbench below (**against the blocked-serial `by_scan`, not today's
loop**), the reader half by the existing conversion bench.

### `e` is a fixed point on the derive path, not a substitution

An earlier draft claimed "this is not circular: `w` is already an input to
`plan_sharded`". That is **false** for the default path. `inp.reader_workers` is
`Option<usize>` (`budget.rs:458`) and `lib.rs:279` passes
`bench_env_reader_workers().or(reader_workers)` — `None` unless the operator sets
it. On that branch `w` is an *output*: `w_max = (pool / cc) - 1` (`budget.rs:577`)
then a downward memory scan picks `w` (`budget.rs:581-590`). Substituting
`e = ceil(r · w)` into `w = pool/cc − e` is a fixed point, and
`depth_cap = pool / (1 + W_TARGET)` (`budget.rs:572`) needs `e` before `cc` even
exists.

Resolve it in closed form, in this order:

| step | rule | why it is well-defined |
|---|---|---|
| 1 | `e_target = max(1, ceil(r(S) · W_TARGET))` | depends only on `S` and a constant |
| 2 | `depth_cap = (pool / (e_target + W_TARGET)).max(1)`; `cc = min(n_contigs, depth_cap)` | `e_target` known |
| 3 | derive path: `w = max(1, (pool / cc) / (1 + r(S)))`, then `e = max(1, min(ceil(r(S) · w), pool/cc − w))` | one division, no iteration |
| 4 | explicit-`w` path: `e = max(1, ceil(r(S) · w))`; `cc_cap = (pool / (e + w)).max(1)` | `w` genuinely is an input here (`budget.rs:559`) |
| 5 | memory scan gives back readers before contigs, unchanged | `e` is a function of the `w` under test |

Two constraints on the fit:

1. **`e = 1` at S ≤ 16k.** That regime is what `754e255` measured, and today's
   plan is correct there. A fit that changes the plan at S = 1,000 is wrong.
2. **Never extrapolate past the measured range.** The fit is clamped at its
   largest measured S, per the RamLaw lesson (`fit-bound-laws-as-lp-envelopes`).

### `W_TARGET`'s rationale is invalidated, not inherited

`W_TARGET = 8` (`budget.rs:51`) is justified as "each extra concurrent contig
costs a dedicated executor core and a full `per_contig_mb` of RAM while buying no
extra reader cores" — literally the `1` this change replaces with `e`. Depth now
costs `e` cores per contig, which strengthens the depth-over-breadth argument but
changes its magnitude. `W_TARGET` is already marked a starting value to be set
from the sweep; re-derive it here rather than carrying it over silently.

### The merge tail must not be squeezed to its floor

`processing_threads_for` floors at 1 (`budget.rs:660`), and 1 gives back commit
`c49d1d7`'s 2.77–2.82× merge-tiling win. On the derive path
`cc · (e + w) ≤ pool = usable − ceil(usable/MERGE_RESERVE_DIV)` (`budget.rs:106`,
`budget.rs:38`) holds by construction, so `processing_threads ≥ usable/4`. On the
**explicit**-`reader_workers` path it does not: `cc_cap` floors at 1
(`budget.rs:559`), so a large explicit `w` plus a derived `e` can exceed `usable`.
Add an assertion/test that `processing_threads ≥ usable / MERGE_RESERVE_DIV` on
the derive path, and that the explicit path's overshoot is reported rather than
silent.

### Memory

`memory_fits` (`budget.rs:629`) prices a contig as
`per_contig_mb + kappa · w · chunk_MB + in_flight_MB(w)`. Nothing in `RamLaw` has
an executor-width term, and `RamLaw::VCF`'s coefficients were fitted with **one**
executor thread. Under this change a contig additionally holds `n_slices`
per-slice `SparseSubStream` sets plus one transient concatenation copy.

With "Per-slice reservations" applied, the *live* bytes are the same partition of
the same calls, and the transient is ~11 MB/chunk (see "Concatenation cost") —
i.e. ~1 chunk-MB, well inside the existing `in_flight` term. Ship with that
argument **and a measured `rss_mark` A/B at e = 1 vs e = 8** on the microbench
before trusting it; if peak moves, add an explicit `e · concat_MB` term rather
than widening `kappa` (which already absorbs constants it should not —
`ram-law-kappa-absorbs-constant-terms`).

### The `Tuning` contract

`Tuning.exec_workers` overrides the derived value and is honoured, not silently
shrunk — matching the tuning spec's stated contract. That contract and a clamp to
`[1, columns/8]` are in conflict; resolve it as: **a request above `columns/8` is
refused with an error naming both numbers**, never silently reduced. The only
other refusal path stays the memory budget.

The `pipeline config` banner must print the resolved `exec_workers`. The
downstream cost of *not* doing this is on record: on #174 a user lost a day to a
banner that printed a `reader_workers` the run was not using.

### Pool ownership

The executor gets its **own** `rayon::ThreadPool`, built explicitly **once per
contig** (not per chunk) inside `run_compute_engine`, and entered with
`pool.install`.

The reason is *not* the one an earlier draft gave. The executor is a plain
`std::thread` — `thread::Builder::new().name(format!("exec-{}", chrom))`
(`orchestrator.rs:984-986`) — spawned *from* a rayon worker of `lib.rs`'s
`concurrent_chroms`-sized dispatch pool (`lib.rs:353-354`). A freshly spawned OS
thread is **not** a rayon worker: `rayon::current_thread_index()` is `None` there,
so a bare `par_iter()` from the executor runs on rayon's **global** registry —
sized to every core by default. The failure mode is therefore *oversubscription*
(`concurrent_chroms × available_parallelism` threads fighting the readers), not
silent serialisation. Silent serialisation is the trap for code running *on* a
dispatch-pool worker, which is what `dense_merge.rs:31-38` documents — a different
thread, a different failure.

An explicit pool also lets the workers be named (`xw-{chrom}-{i}`) so `top -H` can
find them, and gives a `start_handler` hook to register TIDs (below).

### Feature gating

`rvk` is **ungated** (`lib.rs:88`) — it compiles into the query-only core
GenVarLoader links with `default-features = false` — while `monitor`
(`lib.rs:51-52`) and `executor` (`lib.rs:31-32`) are `#[cfg(feature = "conversion")]`.
So `dense2sparse_vk*` **cannot name `crate::monitor::TidRegistry`**. Registration
belongs in the pool's `start_handler`, built in `executor.rs` (gated), or is
passed into `rvk` as an ungated `&(dyn Fn() + Sync)`. `rayon` itself is an
unconditional dependency (`Cargo.toml:36`), so the pool type is fine.

`cargo check --no-default-features` is a required gate on this change
(`genoray-query-core-ci-gap`). If the bench binary lands in `src/bin/`, it is
auto-discovered by every `cargo build`, including the no-features one — declare
it in `Cargo.toml` with whatever `required-features` it actually needs, as
`bench_from_vcf_list` does (`Cargo.toml:89-91`).

### Monitor attribution

`TidRegistry` is `Arc<Mutex<Vec<i32>>>` (`monitor.rs:43`) and `pool_cpu_pct`
(`monitor.rs:116`) clones it under the lock each tick and *replaces* its `prev`
map, so concurrent pushes and a registry that grows mid-run are both safe. Its
doc comment already anticipates an executor **pool** (`monitor.rs:47`) — leftover
from the removed prototype.

Two consequences to handle rather than discover:

- Register every worker TID **before the first chunk** (pool `start_handler`), or
  a TID first seen on tick N contributes its whole lifetime's CPU as that tick's
  delta and prints a spurious spike.
- `cpu_exec` becomes a pool aggregate and will exceed 100 %. Parsing is fine —
  `probe.py`'s `_field` (`probe.py:46`) plus `float(v.rstrip("%"))`
  (`probe.py:81-84`) has no upper bound, and `cpu_shard=1983%` already exceeds
  100. **The knee model is not fine:** `model.py::knee_from_probe`
  (`model.py:433-442`) computes `ceil(c_read / c_exec)` from these two series and
  the module docstring pins them as "CPU UTILIZATION percentages bounded in
  (0, 100]" (`model.py:670`). With `e > 1` the ratio shifts by a factor of `e`
  and every predicted knee moves. Either divide `cpu_exec` by the resolved `e` in
  `probe.py`, or record `e` per row and correct in `model.py` — and say which,
  because a silently rescaled knee is exactly the class of bug
  `resumable-sweeps-serve-stale-rows` cost a review cycle.

## Measurement

No 535k-sample corpus is needed, and none should be built. A standalone bench
(`src/bin/bench_dense2sparse.rs`) constructs a synthetic `DenseChunk` directly —
`chunk_size` variants, S samples, ploidy 2, carriers drawn at a given alt-call
frequency — and calls the emission path under test. This isolates the executor
exactly, runs in seconds, and is the same seam the production path uses.

Arms, in this order (the first two decide whether the third is worth shipping):

1. **`by_scan`, today's loop, e = 1.** Establishes the baseline and checks the
   1.6 ms/variant arithmetic in the Problem section against a real number.
2. **`by_scan`, blocked-serial rewrite, e = 1.** The ~60× claim.
3. **`by_scan`, blocked-serial, e ∈ {1, 2, 4, 8, 16}.** The fan-out.
4. **Carrier arm, shared vs per-slice counting sort, e ∈ {1, 2, 4, 8}.** Separates
   the cache effect from the core count.

`carriers=None` and `carriers=Some(..)` must both be constructed, since they
select different functions (`rvk.rs:594-596`) — a bench that only builds one arm
measures only one.

It reports, per (S, alt-freq, `exec_workers`, arm):

1. wall time and CPU time for the stage,
2. the serial/parallel split (`route_variants` vs. fan-out),
3. peak `rss_bytes()` (`monitor.rs:64`) for the memory question above,
4. a store digest, asserted equal across every `exec_workers` at fixed input.

Requirements on how it is run, from prior burns in this repo:

- Pin `--nodelist` for any cross-configuration comparison; the same job has run
  151.9 s on cn-03 and 73.2 s on cn-04.
- Run it under `sbatch`, not on the login node.
- Confirm a fresh `.so` (`maturin develop --release`) before believing any
  Python-level number; `pixi run test` does not rebuild the extension.
- Export `CARGO_TARGET_DIR` off NFS.
- Demand "Compiling … (scratch path)" plus a changed `.so` sha before believing
  any A/B (`shared-cargo-target-dir-fakes-ab-builds`).

## Testing

1. **Equivalence, unit — against a frozen serial oracle.** A `#[cfg(test)]`
   verbatim copy of today's `by_scan` body is the reference. The parallel
   `by_scan` and the parallel carrier arm are each asserted byte-identical to it
   at `exec_workers ∈ {1, 2, 3, 7}`. Odd and prime worker counts are deliberate:
   they force unaligned tail slices.
   **Fixture widths are load-bearing.** Today's differential fixtures are built by
   `private_chunk` (`rvk.rs:804`) at 16, 1024 and 128 columns (`rvk.rs:937`), and
   the dense-routing fixture is 8 columns (`rvk.rs:989`). With an
   `n_slices ≤ columns/8` cap, `exec_workers = 7` clamps to 1 or 2 on three of
   the four, and the test passes vacuously. Add fixtures with
   `columns ≥ 8 · max(exec_workers)` and assert the *realised* slice count, not
   the requested one.
2. **Equivalence, property.** A proptest over (V, S, ploidy, carrier pattern,
   dense/sparse routing mix) asserting the `SparseChunk` is equal at
   `exec_workers = 1` and `exec_workers > 1`. Must include ploidy ≠ 2 (slice
   boundaries then split a sample's haplotypes across workers; the carrier arm
   recovers the sample as `s = column / ploidy`, `rvk.rs:686`).
3. **Empty slice.** A fixture with a contiguous run of ≥ one slice-width of
   columns carrying nothing, asserting `sample_lengths.len() == columns` for
   every stream and that the zeros land in the right positions. Nothing in
   groups 1–2 forces a wholly-empty slice to exist, and a worker that
   early-returns on an empty window shifts the entire ledger.
4. **Equivalence, end-to-end.** Store digest of a converted fixture, asserted
   identical across `exec_workers` and against the pre-change build, on **both**
   `from_vcf` (dense/`by_scan`) and `from_vcf_list` (carrier) inputs. This is the
   claim `754e255` could not make.
5. **Alignment invariant.** A test asserting every slice boundary is a multiple
   of 8 and that slices tile `0..columns` exactly, for a spread of
   (columns, n_slices) including `columns` not divisible by 8 and
   `n_slices > columns / 8`.
6. **Carrier precondition, release semantics.** A test (not `debug_assert`-gated)
   that an unsorted or out-of-range `Carriers::cols` is rejected loudly rather
   than silently dropping calls — the `62c78f6` failure mode, which today's
   counting sort tolerates and `partition_point` will not.
7. **Planner.** `plan_sharded` tests for the `e + w` demand, for `e = 1` at
   S ≤ 16k, for the clamp at the largest measured S, for the fixed-point
   resolution order above, for the refusal (not shrink) of an over-large
   `Tuning.exec_workers`, and for `processing_threads ≥ usable / MERGE_RESERVE_DIV`
   on the derive path.
8. **Build gates.** `cargo check --no-default-features` (query-core, what gvl
   links) and `cargo test --no-default-features --features conversion`
   (`genoray-cargo-test-no-default-features`).

### What still gets through

Groups 1–8 all compare within one build on synthetic fixtures. The residual
exposure is the FORMAT-heavy configuration: with FORMAT fields requested, the
serial second pass dominates (see "The serial remainder") and every equivalence
test still passes while the change delivers no speedup at all. That is a
*performance* regression in expectation, not correctness, and only the
`format_fields` precondition check catches it. Confirm it against the target run
before promising a number.

## Follow-up issues to open

- **Ledger memory at cohort width.** The executor clones each stream's
  `sample_lengths` into `var_key_ledgers` per chunk (`executor.rs:84-86`), and
  holds them for the whole contig. `sample_lengths.len() == columns` for every
  stream (`types.rs:248`), so at S = 535k that is 2 × 1,071,324 × 4 B ≈ 8.57 MB
  per chunk; a chr12-sized contig at `chunk_size=5000` is ~2,200 chunks
  ≈ 18.9 GB retained. The reported run showed `rss_mb=79983`. Worth its own
  investigation; not this change.
- **Dense-FORMAT second pass** — O(V_dense × S × F), serial, variant-major
  (`rvk.rs:426-440`).
- **`estimated_nnz = V × columns / 20`** (`rvk.rs:504`, `rvk.rs:648`) reserves
  1.07 GB per stream per chunk at AoU width on a density assumption that is wrong
  by orders of magnitude.
- **`orchestrator.rs:447`'s "SparseChunks are tiny (~hundreds of KB)"** is stale:
  `sample_lengths` alone is 8.57 MB/chunk at cohort width, so `tx_sparse`
  capacity 8 is ~69 MB, not ~1 MB.
- **`W_TARGET` re-derivation** once per-contig executor demand is `e`, not 1.
- **`model.py` knee rescaling** for a pool-aggregate `cpu_exec`.
- **Release cut.** #174's back-pressure is merged and unreleased; issue #176's
  item 1 resolves on a release, not a code change.

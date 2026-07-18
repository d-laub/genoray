# `from_vcf_list`: carry FORMAT by carrier — break the residual O(N²)

Design for the deferred FORMAT half of the route-before-densify work (issue #120,
PR #121). Supersedes the "Next step" paragraph of
`2026-07-16-svar2-from-vcf-list-memory-design.md` §3.1: the genotype half of that
design shipped; this document specifies the FORMAT half that closes the O(N²).

## 1. Where we are

PR #121 landed Design A. The genotype round-trip is gone: `DenseChunk.carriers`
carries per-variant carrier lists from the k-way merge into `rvk`, which routes
before densifying and emits sparse-routed genotypes straight from the carrier list
(`dense2sparse_vk`'s carrier-driven path). The scaling exponent did **not** move —
N^1.756 → N^1.747 — because the **FORMAT** half of the same round-trip is still in
place.

**The residual is one loop, `chunk_assembler.rs:737-747`:**

```rust
for (j, col) in format_staged.iter_mut().enumerate() {
    for s in 0..num_samples {                       // F × N per atom, unconditional
        col.push_f64(resolve_format(&a.format_vals, &self.format_fields[j],
                                    a.source_alt_index, s, j));
    }
}
```

For a somatic cohort the union variant count V grows ∝ N, so this is
**O(V × F × N) = O(N²)** — ~22.8 billion `f64` pushes at N=2000, F=7. It flattens
the already-sparse `FormatVals::ByCarrier` values into a dense
`format_staged` grid of shape `v × num_samples`, laid out variant-major
(`types.rs:154`).

**Both downstream consumers read that grid back only at carrier positions:**

- `emit_call`'s `VarKey` arm reads `format_staged[idx][v * num_samples + s]` for the
  single calling sample `s` (`rvk.rs:199`).
- `route_variants`' dense second-pass fill reads it for `is_carrier(v, s)` samples
  and defaults the rest (`rvk.rs:416`).

Neither consumer ever needs a non-carrier's value from the grid. The grid is a lossy
round-trip — sparse `ByCarrier` → dense `v × N` → read back only the carrier
positions — identical to the genotype round-trip Task 8 removed, left in place for
FORMAT.

### What already exists (do not rebuild)

Landed since the memory note and directly reused here:

- `FormatVals::ByCarrier(CarrierFormat)` (`record_source.rs:161-209`) — carrier-sparse
  FORMAT with `value(sample, field) -> Option<f64>` (binary search over ascending
  `samples`, `None` for a non-carrier).
- `resolve_format` (`chunk_assembler.rs:243`) — the lazy resolver whose `ByCarrier`
  arm returns `cf.value(s, j)` and, for a non-carrier, resolves to the spec default
  via `resolve_scalar(None, 0, spec)`. Its `ByCarrier` arm deliberately **ignores**
  `source_alt_index` (values were already resolved against each file's own ALT index
  at merge time; re-applying would double-resolve — see the arm's comment).
- `AtomMeta.format_vals: Arc<FormatVals>` (`chunk_assembler.rs:274`) — the per-atom
  `Arc` this design moves into the chunk instead of dropping after staging.
- `DenseChunk.carriers: Option<Vec<Carriers>>` (`types.rs:166`) — the genotype
  precedent this design mirrors for FORMAT.

## 2. Design — carry FORMAT by carrier

One change, mirroring the genotype fix exactly: carry the carrier-sparse FORMAT into
`rvk` and read it there, instead of densifying to a grid and reading the grid back.

### 2.1 Representation

Add one field to `DenseChunk` (`types.rs`), sibling to `carriers`:

```rust
/// Per-variant FORMAT values, carrier-sparse, one entry per variant in chunk
/// order. `Some` iff `carriers.is_some()` (the k-way merge over single-sample
/// VCFs); `None` for natively dense sources (multi-sample VCF, PGEN), which
/// keep `format_staged`. `carriers` and this are two carrier-sparse encodings
/// of the same chunk, keyed differently: `carriers` by haplotype column
/// (genotype presence), this by sample (FORMAT is per-sample, not per-hap).
pub format_by_carrier: Option<Vec<Arc<FormatVals>>>,
```

`carriers` and `format_by_carrier` are `Some`/`None` together — both come from a
carrier-bearing source or neither does. `source_alt_index` is **not** carried: the
`ByCarrier` arm ignores it (§1).

### 2.2 `chunk_assembler.rs`

The chunk already asserts carrier-bearing xor dense uniformity (`all_some`/`all_none`,
`chunk_assembler.rs:756`). Branch the FORMAT staging on it:

- **Carrier-bearing** (`all_some`): **do not build `format_staged`** — leave the
  columns empty. Move each variant's `Arc<FormatVals>` (already `ByCarrier`) into a
  `Vec<Arc<FormatVals>>`, in the same variant order as `pos`/`carriers`, and set
  `format_by_carrier = Some(..)`. This deletes the O(V × F × N) loop entirely.
- **Dense source** (`all_none`): unchanged — build `format_staged` as today, set
  `format_by_carrier = None`.

INFO staging is untouched: it is O(V) per field (one value per variant), not O(V×N),
so it stays in `info_staged` for both source kinds.

### 2.3 `rvk.rs`

Both FORMAT consumers gain a source switch on `chunk.format_by_carrier`:

- **`emit_call` `VarKey` arm** (`rvk.rs:197-204`): when `format_by_carrier.is_some()`,
  resolve FORMAT via `resolve_format(&fbc[v], spec, _, s, j)` for the calling sample
  `s`; otherwise index `format_staged` as today. The INFO branch is unchanged
  (`info_staged` always populated).
- **`route_variants` dense second-pass fill** (`rvk.rs:411-421`): same switch. For a
  dense-routed variant in a carrier-bearing chunk, call `resolve_format` across all N
  samples — `ByCarrier::value` returns `None`→default for non-carriers, exactly
  reproducing today's `is_carrier(v, s) ? staged : default`. This is genuine
  O(V_dense × N) dense work and is the real cost of a variant that routes dense.

`emit_call` and `route_variants` are shared by both `dense2sparse_vk` (carrier-driven)
and `dense2sparse_vk_by_scan` (grid-scan). The scan path only ever runs on a natively
dense chunk (`format_by_carrier == None`), so it always takes the `format_staged`
branch — no behavior change there.

### 2.4 Cost

Total FORMAT cost becomes **O(total_carriers + V_dense × N)**:

- Somatic (`V_dense ≈ 0`) ⇒ **linear in cohort size** — the slope change this closes.
- Germline ⇒ unchanged, because those variants genuinely route dense and want the
  per-sample column.

The routing threshold, not the cohort, decides — as intended.

### 2.5 Peak RAM (a second win, not just time)

`format_staged` for a carrier-bearing chunk is `F × chunk_size × num_samples × 8`
bytes ≈ **~9.9 GB per chunk** at N=7089, F=7, chunk_size=25000. Removing it drops peak
RAM by that much per chunk; `format_by_carrier` replaces it with
`Σ carriers × F × 8` ≈ negligible for somatic. The memory design's "peak RAM still
grows N^0.73" was measured with the FORMAT half still in place — this is the change
that moves it.

### 2.6 `chunk_size` / `max_mem` budgeting — unchanged

`_auto_chunk_size` stays **worst-case (all-dense)** per the memory design §3.4: the
dense fraction is a per-chunk, data-dependent routing outcome not knowable when the
chunk size is chosen. So `max_mem` remains a genuine ceiling — somatic cohorts come in
under it once §2.2 lands, but under-budgeting on a predicted dense fraction would
reintroduce the OOM. No change to the budgeting formula.

## 3. Verification

- **Byte-identical output (the gate that must never go red).** The PR #121 parity
  fixture — extended to a **fields + multi-contig** cohort — must produce a
  byte-identical store through every commit. Carrier-sparse FORMAT resolution must
  equal today's grid-densified staging.
- **Differential `dense2sparse_vk` vs `_by_scan`.** Still runs a carrier-bearing chunk
  through the scan path (carriers withheld) to prove the two iteration orders emit
  identically. It exercises the `format_staged` branch on one side and the
  `format_by_carrier` branch on the other, so it now also guards the source switch.
- **New unit test.** Carrier-sparse FORMAT resolution: a non-carrier column resolves
  to the field default; a dense-routed variant in a carrier-bearing chunk fills all N
  samples correctly (carriers → value, non-carriers → default); a multiallelic record;
  a record missing a requested field.
- **dhat regression gate.** On the fixed bench (N ≥ 250, 3 contigs, F=7): assert total
  blocks fall ≥10× versus the 158.8M baseline, and that the two named churn sites
  (`vcf_list_reader.rs:485`, `chunk_assembler.rs`) leave the top-5.
- **Asymptote (the headline).** N-sweep {250, 500, 1000, 2000, 4000}, single contig,
  F=7. `cpu_s` scaling must go from ~N^1.75 to ~linear. `dense2sparse_vk` must leave
  the top of `perf` (it is a top self-time symbol today).
- **Peak RAM.** 3-contig N=1000 trace: the ~10 GB/chunk `format_staged` disappears;
  confirm peak drops and the per-contig ratchet flattens.
- **The real thing.** Re-run the 7089-file Hartwig cohort (24 contigs, F=7) against the
  132 GB baseline. Only this exercises N=7089 × full genome.

## 4. Design B — assessed, deferred, measurement-gated

Design B (two-pass, sample-parallel; memory design §3.1a) is **not** built here. This
section records why, and the gate that decides when it is.

**Part 1 (this design) narrows B's remaining justification.** B was pitched to fix
four things; §2 of this design takes two of them:

| problem | after Part 1 |
|---|---|
| O(N²) scan | **fixed** (§2.4) |
| peak RAM / allocation churn | **largely fixed** (§2.5) — the ~10 GB/chunk grid is gone |
| serial read phase (1 of 8 cores) | **unchanged** — the k-way merge is inherently serial |
| N files open at once / `ulimit -n` (#122) | **unchanged** |

So B's live levers are now only **throughput** (parallelize the idle ~7 cores — B's
pass 2 is embarrassingly parallel per sample) and the **fd / open-files ceiling** (B
bounds pass 2 to ~cores). Peak RAM and the asymptote are no longer B's to fix.

**Adopt Design B if — and only if — one of §3.1a's triggers fires after Part 1 lands
and is re-measured** (decide on measurements, not taste):

1. **Part 1 is linear but still slow.** If the N-sweep confirms linear scaling yet
   wall-clock at N=7089 is still hours, the remaining lever is the idle cores — that
   is B.
2. **`read≈99% exec=0%` persists.** Part 1 does not touch the serial structure; if the
   read phase is still one-core-bound, B is the only design here that changes it.
3. **The fd / open-files ceiling bites again.** `_check_fd_budget` and #122's
   contig-index-seek failure both trace to holding N files open; B bounds pass 2 to
   ~cores.

**Part 1 is the incremental half of B, not throwaway.** `Carriers`, the
route-before-densify inversion, `format_by_carrier`, and the field plumbing are all
reused by B's pass 2 (which is itself per-sample carrier emission). If B follows, it
builds on this seam rather than replacing it.

## 5. Risks

- **The differential test does not cover `route_variants` itself.** Routing is shared
  code, so a routing bug reproduces identically on both iteration orders and the
  differential test still passes (see `dense2sparse_vk_by_scan`'s comment,
  `rvk.rs:436-450`). The byte-identical **fixture** parity (§3) is the backstop that
  actually pins routing correctness — keep it, and extend it to fields + multi-contig.
- **Routing threshold now has teeth on FORMAT too.** With sparse and dense FORMAT
  paths diverging in cost, a cohort that misroutes many variants to `Dense` pays the
  old O(V_dense × N). Confirm behavior on a **germline** cohort, not just somatic —
  there the dense fill is expected and correct, but the parity must still hold.
- **`Arc<FormatVals>` lifetime.** Moving the `Arc` into `DenseChunk` extends its live
  range from staging to `rvk` consumption. This is the intended trade (a small
  per-variant `Arc` replacing a `F × N` grid) but confirm the chunk is dropped
  promptly after `dense2sparse_vk` so the `Arc`s free per chunk, not per contig.
- **Public API.** No public surface changes (`format_by_carrier` is internal;
  `max_mem`/`chunk_size` already shipped in PR #121). No `skills/genoray-api/SKILL.md`
  update is required by this design — confirm before merge that nothing public moved.

## 6. Reproducing

```bash
# dhat (the measurement that settles live-vs-fragmentation)
export CARGO_TARGET_DIR=/tmp/genoray_dhat     # NFS target/ bus-errors on mmap
cargo build --release --no-default-features --features conversion,dhat-heap \
    --bin bench_from_vcf_list
./target/release/bench_from_vcf_list <manifest> <out> "1,2,3" <ref.fa> \
    "VAF:float,DP:int,PURPLE_AF:float,PURPLE_CN:float,PURPLE_VCN:float,PURPLE_MACN:float,SUBCL:float"

# N-sweep + RSS trace harnesses (from the memory design)
python run_arm.py   <manifest> <out>.svar2 on|off <label>   # RSS + arena-heap split
python run_trace.py <manifest> <out>.svar2 <label>          # RSS time series (ratchet)
```

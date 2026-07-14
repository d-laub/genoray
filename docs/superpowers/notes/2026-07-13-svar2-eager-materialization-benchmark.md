# SVAR2 `write_view` eager-materialization benchmark

**Deferred call-out (PR #105):** *"Eager `Vec<RawRecord>` materialization in
`Svar2Source` (plan-mandated) — builds the whole contig view in memory; worth a
benchmark gate before advertising cohort-scale whole-store copies."*

> **RESOLVED (unify-routing work, PR #105).** The eager path this note gated has
> been **deleted**. `write_view` no longer has a pipeline/`Svar2Source` backend at
> all: both `reroute=True` and `reroute=False` now route through the array-slicer
> (`run_slice_view`), whose peak is **O(output per contig)**, not
> O(n_variants × n_haps). The "do NOT advertise cohort-scale whole-store copies"
> gate below is therefore **lifted** — see [Verdict](#verdict--gate). The original
> TL;DR / method / results are kept verbatim under
> [Historical](#historical-the-deleted-eager-pipeline-path) for contrast.

## TL;DR (current)

The eager `Svar2Source` materialization is gone (deleted in the unify-routing
work). `SparseVar2.write_view` is now a single array-slicer with two routing
policies; it gathers each variant's provenance and emits the output stream
per contig, so peak memory is bounded by the **output size of one in-flight
contig** (× concurrent contigs via `threads`), plus — when `reference=` is
given — that contig's reference sequence (~250 MB for chr1). It is no longer
linear in `n_variants × n_haps`, so the cohort-scale OOM this note warned about
does not occur on the current code path.

**Fresh cohort-scale measurements are pending** a re-run of
`scripts/svar2_eager_bench.py` on `data/chr21.germline.svar2` (whole chr21, 3202
samples). That store is **not present in this worktree**, so the numbers below in
[Results](#results-historical-deleted-eager-path) are the *old* eager-path
measurements, retained only for historical contrast. When the store is available,
run (on node-local disk — the merge stage mmaps output and SIGBUSes on NFS):

```bash
sbatch -p carter-compute -c 16 --mem=64G --wrap "cd $(pwd) && \
  export CARGO_TARGET_DIR=/tmp/genoray-target-\$\$ && pixi reinstall genoray && \
  pixi run -e py310 python scripts/svar2_eager_bench.py \
    --store data/chr21.germline.svar2 --out-dir /scratch/\$USER/bench \
    --ks 100 500 1000 3202 --threads 16"
```

Expected: peak RSS collapses from the historical **0.84 / 2.99 / 6.89 / 30.96 GB**
to roughly the output size (**0.005 / 0.025 / 0.052 / 0.203 GB**) plus a working
set — i.e. **no longer linear in `k`** at ~9.7 MB/sample. Wall time should also
drop (a gather beats a full re-conversion). Paste the new table beside the
historical one when measured.

## Verdict / gate

- **Gate LIFTED.** The eager `Svar2Source` (`Vec<RawRecord>` + per-variant carrier
  `BTreeMap`, ≈ `n_variants · n_haps · 5 B` resident) was deleted; there is no
  longer a code path that materializes the whole contig view before draining it.
  The slicer's peak is O(output per contig), so cohort-scale whole-store copies
  through `write_view` are no longer memory-gated by construction. Advertising
  them is safe on the current code path. (The one remaining per-contig cost is the
  reference sequence when `reference=` is passed.)
- **Confirmation pending measurement** on the real chr21 store (absent here) — the
  claim above follows from the deletion, but the fresh RSS numbers should be
  recorded to close the call-out empirically. This does not block merge: the
  benchmark existed to decide whether the *eager path* was safe to ship, and that
  path no longer exists.
- `concat` / `split` (Component A, pure file ops) were always unaffected — this
  only ever concerned `write_view`.

---

## Historical: the deleted eager-pipeline path

*Everything below describes `Svar2Source`, the pipeline-backed `reroute=True`
backend that the unify-routing work deleted. It is retained for historical
contrast only; it does not describe the current code.*

### TL;DR (historical)

`SparseVar2.write_view` peak memory was **O(n_variants × n_haps)** and tracked the
analytic lower bound almost exactly (~**31 GB** for a whole-chr21 view of the full
3202-sample germline cohort, against a **0.2 GB** output — a ~150× blowup). It was
fine for modest cohorts/regions (≤~1 GB up to a few hundred samples whole-chr21)
but grew linearly with `samples × variants` and would OOM at biobank cohort scale
or on large chromosomes × thousands of samples. **Recommendation (historical): do
not advertise cohort-scale whole-store copies through `write_view` until
`Svar2Source` is made streaming.** — *superseded: `Svar2Source` was deleted, not
made streaming; the slicer replaced it.*

### Why (historical)

`Svar2Source::new` (`src/svar2_source.rs`, now deleted) decoded the entire contig
subset up front:

1. a `BTreeMap<(pos,ilen,alt), Vec<bool>>` — one `n_haps`-long carrier bitset per
   variant (~`n_variants · n_haps` bytes), then
2. a `Vec<RawRecord>` — each record's `gt` a `Vec<i32>` of length `n_haps`
   (~`n_variants · n_haps · 4` bytes),

both fully resident before `next_record` drained the first record into the
pipeline. So the source alone cost ≈ `n_variants · n_haps · 5` bytes, on top of
the conversion pipeline's own working set.

### Method (historical)

`scripts/svar2_eager_bench.py` ran `write_view` over the whole contig for a
deterministic first-`k`-sample subset, one k per subprocess (so `ru_maxrss` is
that run's peak), on node-local disk. Store: `data/chr21.germline.svar2` (whole
chr21, 3202 samples, 1,001,385 variants). `eager lower-bound` = `n_variants ·
k·ploidy · 5 B`.

### Results (historical, deleted eager path)

| samples k | haps | wall (s) | peak RSS (GB) | out size (GB) | eager lower-bound (GB) |
|---|---|---|---|---|---|
| 100 | 200 | 15.2 | 0.84 | 0.005 | 1.00 |
| 500 | 1000 | 34.5 | 2.99 | 0.025 | 5.01 |
| 1000 | 2000 | 65.4 | 6.89 | 0.052 | 10.01 |
| 3202 | 6404 | 256.3 | 30.96 | 0.203 | 32.06 |

Peak RSS was linear in `k` (≈ 9.7 MB per sample) and sat at ~0.97× the analytic
lower bound — i.e. the eager source structures dominated the footprint, exactly as
predicted. Output was ~150× smaller than peak RSS. The `eager lower-bound` column
no longer has meaning on the current path (there is no eager materialization);
the new path's peak is the output size plus a per-contig working set.

### Extrapolation (historical — why it gated cohort-scale)

Peak ≈ `n_variants · k · ploidy · 5 B`. Holding the whole-chr21 germline variant
count:

| scenario | approx peak RSS (eager path, deleted) |
|---|---|
| chr21, 3202 samples | ~31 GB (measured) |
| chr21, 100k samples | ~1 TB |
| chr1 (~4–5× chr21 variants), 3202 samples | ~140 GB |
| chr1, 100k samples | multi-TB |

Any of these would have OOM'd a normal node on the eager path. The slicer that
replaced it does not have this scaling — its peak is per-contig output size, not
`n_variants · n_haps`.

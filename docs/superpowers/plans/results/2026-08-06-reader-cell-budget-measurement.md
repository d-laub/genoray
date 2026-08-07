# Reader cell budget measurement and `PARALLEL_MIN_CELLS` (Task 7)

Measurement-only deliverable for the presence-mask / byte-budget migration
(Tasks 1-4). Measures the reader RSS win from retaining `PresenceMasks`
instead of `Calls::Dense`, and sweeps `PARALLEL_MIN_CELLS` for wall time.

## Run record

- **Node:** `carter-cn-03` (pinned via `--nodelist`; node speed on this
  cluster varies up to 2.08x, so cross-point comparisons are only valid
  pinned to one node).
- **Commit measured:** `25e3d4ac1d69945d823f847b46b1802a8da34747` (HEAD at
  the time of measurement; echoed by both sbatch scripts).
- **RSS ladder job:** 13351676, `--cpus-per-task=48 --mem=64G`,
  `sacct`: COMPLETED, exit `0:0`.
- **`PARALLEL_MIN_CELLS` sweep job:** 13351677, same allocation,
  `sacct`: COMPLETED, exit `0:0`, elapsed 00:09:43.
- **Corpora:** all four (`s2000`, `s8000`, `s32000`, `s128000`) at
  `/local/dlaub/pgen-mem` were **reused as-is** — none were regenerated.
  They are node-local to `carter-cn-03`, 22 autosomes x 1,000 variants/contig
  each, which is why both jobs pin that node.

## Step 1 — RSS ladder

**Methodology.** Single contig (`chr1`, `regions="chr1:1-1000000"`), 1,000
variants, `chunk_size=4096`, `GENORAY_CONCURRENT_CHROMS=1`, one
`log_level="debug"` run per S, parsing the `RSSMARK` trace
(`sed -r 's/\x1b\[[0-9;]*[mK]//g'` first — Rich injects ANSI escapes inside
the numbers).

**"Reader delta" definition.** `rss_mb(reader_drained) - rss_mb(reader_ready)`
for the `chr1` marks. This is the **same** definition the pre-migration
baseline used: the `a93d1fc` commit message that introduced `rss_mark`
reports, verbatim, `reader_drained 1574 MB <- +1094 MB, all of it here` against
`reader_ready 480 MB` at S=128,000 (1574-480=1094), and issue.md's "Evidence"
table is the same subtraction. Both marks bracket exactly
`ChunkAssembler::read_next_chunk`'s loop (`orchestrator.rs:907-914`), so
nothing upstream of `reader_ready` (reader construction, pool construction) is
included by either definition. But the marks bracket wall-clock time, not a
call stack, and RSS is process-wide: the executor thread
(`orchestrator.rs:930-952`), the chunk writer (`:960-963`), and the
long-allele writer (`:966-973`) are all spawned before the reader loop
finishes and run
**concurrently** with it, consuming the `DenseChunk`s the reader is sending
over `tx_dense`. Their allocations inside the `reader_ready`-`reader_drained`
window are therefore counted in the delta too — it is not reader-only. This
does not affect the before/after comparison above (the baseline used the same
window, so the same concurrent contribution is present on both sides), but it
matters for the residual accounting below: the executor's concurrent working
set, not just the reader-side rayon pool, is a candidate for the unexplained
residual. I did not have to guess or reconstruct the marks' placement — the
commit message that shipped the instrumentation states the arithmetic
directly.

For completeness (the reader also confirmed this doesn't change the
conclusion), here is `reader_drained - contig_enter` alongside it. The two
differ by only 3-14 MB (the cost of `ChunkAssembler::new` + building the
reader-side rayon pool, both of which run between the two marks and grow
mildly with S) and tell the identical story:

| S | reader_ready MB | reader_drained MB | **drained − ready** | drained − contig_enter |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 909 | 958 | **49** | 56 |
| 8,000 | 919 | 1,053 | **134** | 137 |
| 32,000 | 938 | 1,083 | **145** | 152 |
| 128,000 | 959 | 1,149 | **190** | 204 |

**Comparison to baseline.**

| S | Before (a93d1fc) MB | Before KB/sample | After MB | After KB/sample |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 13 | 6.50 | 49 | 24.50 |
| 8,000 | 42 | 5.25 | 134 | 16.75 |
| 32,000 | 256 | 8.00 | 145 | 4.53 |
| 128,000 | 1,020 | 7.97 | 190 | 1.48 |

(Overall slope, `Δbytes/ΔS` across the ladder: before ≈ 7.99 KB/sample,
matching the brief's figure; after ≈ 1.12 KB/sample.)

**The shape changed, not just the magnitude.** Before the migration, growth
is close to linear in S (13→1,020 MB is a 78.5x increase against a 64x
increase in S — that's what an `O(S)` retained buffer looks like). After the
migration, growth is **sub-linear and plateauing**: S=8,000→128,000 is a 16x
increase in S but only a 1.42x increase in the delta (134→190 MB). That
plateau is the two byte budgets (`RAW_STAGE_BYTES`, `MASK_STAGE_BYTES`, both
64 MiB) doing their job — once `columns` is large enough that the budget
binds instead of the `MAX_BATCH_RECORDS`/`MAX_PACK_WINDOW` caps, the two
buffers stop growing with S at all.

**A real, stated trade at the narrow end.** At S=2,000 the new number (49 MB)
is *higher* than the 13 MB baseline. This is not noise — it is an intended
regression at narrow widths, the direct consequence of replacing a per-sample-scaled cap
with a fixed byte budget: at S=2,000, `batch_records`/`pack_window` both still
clamp to their `MAX_*` caps (1,024), so the reader stages/retains close to the
same absolute bytes it would at any width up to where the budget starts
binding (~S=8,192 for the batch, ~S=262,144 for the window) — it no longer
scales down just because the cohort is narrow. Narrow cohorts now pay more in
absolute terms than they used to, in exchange for wide cohorts being bounded
at all. This is worth stating plainly because it is a genuine behavior change,
not just a magnitude improvement.

### Residual accounting — measured is above the Task-4 arithmetic-only estimate

Task 4's arithmetic bounds only the two reader-side staging/retention buffers
(`batch_records(columns) * columns * 4` and
`pack_window(columns) * columns.div_ceil(64) * 8`). Evaluating those functions
exactly at each S in the ladder:

| S | columns | batch_records() | batch bytes (MB) | pack_window() | window bytes (MB) | **arithmetic total (MB)** | measured (MB) | **gap (MB)** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 4,000 | 1,024 (capped) | 16.38 | 1,024 (capped) | 0.52 | 16.90 | 49 | 32.10 |
| 8,000 | 16,000 | 1,024 (capped) | 65.54 | 1,024 (capped) | 2.05 | 67.58 | 134 | 66.42 |
| 32,000 | 64,000 | 262 (budget-bound) | 67.07 | 1,024 (capped) | 8.19 | 75.26 | 145 | 69.74 |
| 128,000 | 256,000 | 65 (budget-bound) | 66.56 | 1,024 (capped) | 32.77 | 99.33 | 190 | 90.67 |

The gap is materially above what Task 4's two budgets alone predict (roughly
matching the plan's own back-of-envelope ~102 MB at S=128,000), at every S,
not just S=128,000. I diagnosed this rather than shipping it unexamined:

**The out-of-scope suspect (eagerly-constructed `pgenlib.PgenReader`s) does
NOT explain this gap, for two independent, structural reasons — not just a
measurement that happened to come out small:**

1. **Only one reader is built for this measurement, not 22.**
   `SparseVar2.from_pgen` (`python/genoray/_svar2.py:1108-1125`) filters
   `contigs` down to whatever `region_ranges` covers *before* the reader-pool
   loop at line 1169 (`readers = [[pgenlib.PgenReader(...) for _ in
   range(P)] for _ in contigs]`, `P=1`). Since every ladder run passes
   `regions="chr1:1-1000000"`, `contigs == ["chr1"]` by the time that loop
   runs — one `PgenReader`, not one per each of the file's 22 contigs.
2. **Even if it built 22, it would be invisible to this delta anyway.** The
   reader-construction loop runs entirely in Python, and its output
   (`readers`) is passed into `_core.run_pgen_conversion_pipeline(...)`
   (line ~1273) — the Rust entry point where `rss_mark(chrom, "contig_enter")`
   fires for the *first* time (`orchestrator.rs:369`). Construction is
   therefore complete before `contig_enter`, let alone `reader_ready`, is ever
   measured; it cannot appear in either `drained-ready` or `drained-enter`.

I also ran `$CLAUDE_JOB_DIR/tmp/reader_rss.py` (22 synthetic readers, its own
tiny corpora at S=1,000/4,000/16,000/32,000) as a sanity check on the
mechanism's cost in the abstract: construction was 0.0-0.5 MB total across 22
readers at every S tested, ≤0.52 KB/sample — consistent with the existing
memory note that this candidate was "ruled out... pgenlib readers (0.02
KB/sample)". So even setting aside the structural argument, the absolute cost
of this mechanism is far too small to produce a 32-91 MB gap.

**What I could attribute, partially.** `read_next_chunk` builds a packed
`DenseChunk` (`genos: BitGrid3`, one bit per `(variant, sample, ploidy)`,
`src/types.rs:163`) and sends it to the executor over a *bounded* channel
(`VCF_LIST_DENSE_CHANNEL_CAP = 6`, `orchestrator.rs:347`, "plus one in-flight
on each side"). This buffer is not part of either Task-4 budget — its size is
`min(V, chunk_size) * columns / 8` bytes, and at `chunk_size=4096 >= V=1,000`
exactly one such chunk exists per contig. At S=128,000 that's `1,000 x
256,000 / 8` = 32 MB; at S=2,000 it's only 0.5 MB. Adding this term narrows
the S=128,000 gap from 90.67 MB to about 58.7 MB, but does almost nothing at
S=2,000 (32.1 MB gap, essentially unchanged) — so it is a real, legitimate,
*bounded* (capped at `dense_cap+2` chunks in flight) contributor at wide
cohorts, but it cannot be the whole story, and isn't a story at all at the
narrow end. This `min(V, chunk_size) * columns / 8` credit assumes the
`BitGrid3::zeros(chunk_size, ...)` allocation (`chunk_assembler.rs:855`,
4,096 rows regardless of S — ~131 MB of address space at S=128,000) is
resident only for the `V=1,000` rows actually written by this corpus, not for
its full `chunk_size` extent; that is consistent with the
`bitgrid-zeros-calloc-not-resident` finding (calloc pages cost address space,
not RSS, until touched) and with `a93d1fc`'s own accounting, but it is
load-bearing here — without this credit the residual gap runs 32.1 / 66.4 /
69.7 / 90.7 MB and *grows* 1.37x from S=8,000 to S=128,000 instead of staying
flat. The flatness claim below is conditional on this assumption.

**What remains unexplained.** After both the two Task-4 budgets and the
dense-chunk-channel term, the residual is:

| S | residual after budgets + dense-chunk term (MB) |
| ---: | ---: |
| 2,000 | 31.6 |
| 8,000 | 64.4 |
| 32,000 | 61.7 |
| 128,000 | 58.7 |

This residual is **not proportional to S** — it jumps from S=2,000 to
S=8,000 and then stays flat (58-65 MB) through S=128,000, which is the
signature of a roughly per-run/per-thread fixed cost rather than a per-sample
one. One candidate is the reader-side rayon pool (`orchestrator.rs:890-894`,
sized 5-31 threads across the configurations measured here — confirmed via
`GENORAY_LOG=genoray=info`, which bypasses the message-only Python log bridge
and prints the `processing_threads` field directly) doing its first real
allocating work inside the measured window (thread stacks, work-stealing
deques, and/or glibc per-thread malloc arenas touched for the first time when
several threads pack concurrently). But as noted above, the `reader_ready`/
`reader_drained` window is process-wide and the executor thread
(`orchestrator.rs:930-952`) — plus the chunk and long-allele writer threads
(`:960-963`, `:966-973`) — run concurrently with the reader inside that same
window, consuming the `DenseChunk`s this section already credits. The
executor's own working set (its compute-engine state, sink buffers, and
whatever it allocates decoding/re-encoding each `DenseChunk`) is at least as
plausible a candidate as the reader-side pool, and is not something I ruled
out. I did not instrument either candidate far enough to attribute the residual
to a specific byte. I am reporting it as unexplained rather than asserting a
mechanism I have not verified.
**This is not the eagerly-constructed-`PgenReader` candidate** (ruled out
above on structural grounds, independent of its measured cost), and it does
not grow with S beyond S≈8,000, so it does not reopen the biobank-scale
danger the migration fixes — but it is real, and it means the honest
post-migration number at S=128,000 is **190 MB, not ~102 MB**.

**Bottom line.** The migration is a genuine, large win — 7.99 → ~1.12-1.48
KB/sample (5.4x at the S=128,000 endpoint), and critically the growth curve
changed from linear-in-S (unbounded, unsafe at biobank scale) to
plateauing (bounded past S≈8,000-262,000 depending on which of the two
budgets is binding). It is not the full ~102 MB the two-buffer arithmetic
alone predicts; roughly a third of the S=128,000 gap is attributable to a
real, bounded, already-existing buffer (the dense-chunk output channel), and
the remaining ~59-65 MB per run is unexplained but flat in S, not the
previously-flagged (and here, structurally ruled out) eager-reader
candidate.

## Step 2 — `PARALLEL_MIN_CELLS` wall-time sweep

**Methodology.** Same node, same four corpora, full (all 22 contig) default
`from_pgen` conversions (auto `chunk_size`, default concurrency planner,
`max_mem="900GiB"` so the planner isn't budget-refused on this box —
concurrency itself is otherwise whatever the planner picks, not forced).
Swept `PARALLEL_MIN_CELLS ∈ {0 (always parallel), 512*1_024 (seeded),
8*512*1_024, usize::MAX (never parallel)}`; each value required its own
source edit + `pixi run maturin develop --release` (confirmed via a
`Compiling genoray v0.1.0 (...)` line in the build log and a changed `.so`
sha256 before every measurement — recorded in `sweep_pmc_13351677.log`). 3
repeats per (value, S) cell, 48 conversions total plus 4 rebuilds (one per
value) and a final restoring rebuild, all inside one sbatch job
(00:09:43 elapsed).

**Wall time (mean of 3 reps, seconds):**

| S | always (0) | seeded (512×1,024) | 8x (4,194,304) | never (`usize::MAX`) |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 4.19 | 4.12 | 4.09 | 4.09 |
| 8,000 | 4.83 | 4.88 | 4.79 | 4.89 |
| 32,000 | 6.70 | 6.70 | 6.84 | 6.69 |
| 128,000 | 16.21 | 16.65 | 16.14 | 16.36 |

**Raw per-rep values** (seconds), for the noise estimate below:

| S | always (0) | seeded | 8x | never |
| ---: | --- | --- | --- | --- |
| 2,000 | 4.36, 4.09, 4.14 | 4.15, 4.10, 4.11 | 4.13, 4.10, 4.04 | 4.14, 4.03, 4.10 |
| 8,000 | 4.82, 4.88, 4.79 | 4.88, 4.92, 4.83 | 4.81, 4.75, 4.81 | 4.89, 4.87, 4.92 |
| 32,000 | 6.80, 6.60, 6.68 | 6.66, 6.71, 6.74 | 6.70, 6.87, 6.93 | 6.80, 6.63, 6.65 |
| 128,000 | 16.12, 16.74, 15.76 | 16.08, 16.38, 17.48 | 15.86, 16.06, 16.49 | 16.63, 16.70, 15.73 |

**Noise estimate.** Within a single (value, S) cell, the 3-rep range spans
1.0-6.4% of the mean at S≤32,000 (min 0.0497s, `never`@S=8,000; max 0.2686s,
`always`@S=2,000) and up to 8.4% at S=128,000 (seeded: 16.08-17.48s). This is
driven by `carter-cn-03` being a shared node, not by anything in the code
under test. The spread *across* the four values (max mean − min mean) is
0.11s at S=2,000, 0.10s at S=8,000, 0.14s at S=32,000, and 0.51s at
S=128,000 — 2.1-3.2% of the mean at every S. That is **comparable to**, not
smaller than, the within-cell run-to-run spread: at S=8,000 the across-value
spread (0.1018s) actually *exceeds* the largest within-cell range at that S
(0.0924s, `always`). A Welch t-test across all 24 pairwise (value, value')
comparisons, one per S, finds exactly one nominally significant separation
(p=0.0241, unequal-variance): `eight_x` vs `never` at S=8,000 (4.791s vs
4.893s, t=-3.86, a 2.1% effect); `seeded` vs `eight_x` at that same S is
close but short of significance (t=2.54). One nominal hit at p=0.0241 is not
remarkable on its own once multiplicity is accounted for: across 24
independent-ish comparisons, the family-wise probability of at least one hit
this extreme by chance is ≈0.44 (1−(1−0.0241)^24) — coin-flip odds, not
evidence of a real difference. (For scale, this design's per-S minimum
detectable effect — n=3 reps/arm, pooled sd, α=0.05 two-sided, power 0.8 —
is roughly 6.0% / 2.5% / 4.2% / 10.1% at S=2,000/8,000/32,000/128,000; the
2.1% hit sits right at the S=8,000 floor, consistent with a noise floor that
occasionally pokes through by chance rather than a real effect too small to
usually detect.) One nominal hit in 24 comparisons, at a multiplicity-corrected
p≈0.44 and with no consistent direction across the other three widths, is
what noise looks like, not signal.

A further limitation on the noise estimate itself: all three reps of a given
value ran consecutively (`always`'s three reps, then a rebuild, then
`seeded`'s three reps, and so on) in a fixed value order, on a node shared
with other users' jobs. Reps were not interleaved or randomized across
values, and no independent load record was captured. So *value* is
confounded with *time-order* (and whatever load drift happened over the
sweep's ~10-minute span) — this design cannot separate a genuine
per-value effect from a drift effect, which is one more reason to read the
one nominal hit as noise rather than a real difference.

**What this design can discriminate — and what it cannot.** All four
corpora are 22 contigs x 1,000 variants, and `pack_window() >= 1,024` at
every S tested, so each contig packs inside exactly one `flush_window` call
of ~1,000 atoms. Cells per window are `1,000 * columns`: 4M / 16M / 64M /
256M at S=2,000/8,000/32,000/128,000. Against the swept candidates
`{0, 512*1,024 (=524,288), 8*512*1,024 (=4,194,304)}`, every candidate
selects the **same branch** (parallel vs. sequential) in 11 of the 12
non-`never` (value, S) cells: `always` (threshold 0) and `seeded` (threshold
524,288) select parallel at all four S, and so does `eight_x` (threshold
4,194,304) at S=8,000/32,000/128,000 — the sole divergence is `eight_x` at
S=2,000, where 4,000,000 < 4,194,304 flips it to sequential while `always`
and `seeded` stay parallel. So `always` and `seeded` are, across this entire
sweep, a **null replicate of the same branch**, not a comparison of two
threshold placements — and `eight_x` differs from them in only one of four
cells. What this sweep actually measured is parallel-vs-sequential packing
at these four widths, not where within the parallel range the gate should
sit.

Given that, the honest finding is narrower than "the constant doesn't
matter": **on this corpus, the gate's placement is not wall-time-visible,
and parallel packing itself does not measurably beat sequential packing at
these widths.** (The one nominal hit above sits precisely on this contrast —
`eight_x`, parallel at S=8,000, beating `never`, sequential at every S. That
doesn't overturn the call: it's one hit at a multiplicity-corrected p≈0.44,
it's confounded with time-order (above), and the direction isn't even
consistent across widths — at S=2,000, where `eight_x` is itself in the
sequential branch alongside `never`, that sequential pair (means 4.09-4.09s)
is nominally *faster* than the parallel pair `always`/`seeded` (means
4.12-4.19s), the opposite direction from S=8,000.) Demonstrating that
placement matters (or doesn't) would
need corpora whose per-window cell count straddles the candidate thresholds
more finely — e.g. varying `chunk_size`/`pack_window` independently of S,
not just S itself — which is out of scope for this task.

**Chosen constant: `PARALLEL_MIN_CELLS = 512 * 1_024` (unchanged from the
seeded value).** Per the plan: when the sweep shows no measurable difference,
the correct action is to keep the seeded value and record that finding, not
to pick a value out of noise. That is what happened here — recorded as the
finding, not skipped. The caveats above (comparable-not-smaller noise, one
nominal hit, the design's inability to discriminate placement, and the
time-order confound) narrow what can be *claimed* from this sweep, but they
do not change the decision: it remains plan-compliant to keep the seeded
value, and nothing above is a retraction of that call.

## Verification (commit `25e3d4a` + this task's changes)

- `CARGO_TARGET_DIR=/local/dlaub/cargo-target-diag pixi run -e lint cargo test --no-default-features --features conversion`:
  **472 passed, 0 failed, 1 ignored** — matches the stated baseline exactly.
- `CARGO_TARGET_DIR=/local/dlaub/cargo-target-diag pixi run -e lint cargo check --no-default-features`: clean, no warnings.
- `pixi run test`: **1006 passed, 7 skipped, 16 xfailed** — matches the stated
  baseline exactly. No `batch_records_*`/`pack_window_*`/parallel-gate test
  needed updating: `PARALLEL_MIN_CELLS`'s final value is unchanged from HEAD,
  and no Rust test asserts a literal threshold against it (only the two
  budget *functions* are asserted, and those are untouched by this task).

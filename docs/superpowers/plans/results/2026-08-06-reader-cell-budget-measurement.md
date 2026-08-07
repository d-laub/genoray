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
`ChunkAssembler::read_next_chunk`'s loop (`orchestrator.rs:907-914`); nothing
upstream of `reader_ready` (reader construction, pool construction) or
downstream of `reader_drained` (executor, writers, merge) is included by either
definition. I did not have to guess or reconstruct this — the commit message
that shipped the instrumentation states the arithmetic directly.

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

(Repeated directly on this node outside the sbatch job, S=128,000 only, as a
determinism check: 673-480 = 193 MB — within 2% of the sbatch job's 190 MB.)

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
is *higher* than the 13 MB baseline. This is not noise and not a regression —
it is the direct, intended consequence of replacing a per-sample-scaled cap
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
narrow end.

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
one. The most plausible remaining candidate is the reader-side rayon pool
(`orchestrator.rs:890-894`, sized 5-31 threads across the configurations
measured here — confirmed via `GENORAY_LOG=genoray=info`, which bypasses the
message-only Python log bridge and prints the `processing_threads` field
directly) doing its first real allocating work inside the measured window
(thread stacks, work-stealing deques, and/or glibc per-thread malloc arenas
touched for the first time when several threads pack concurrently) — but I
did not instrument far enough to attribute it to the byte. I am reporting it
as unexplained rather than asserting a mechanism I have not verified.
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
| 2,000 | 4.20 | 4.12 | 4.09 | 4.09 |
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

**Noise estimate.** Within a single (value, S) cell, the 3-rep range is
0.06-0.27s at S≤32,000 (2-4% of the mean) and up to 1.40s at S=128,000
(seeded: 16.08-17.48, ~8.4% of the mean) — driven by `carter-cn-03` being a
shared node, not by anything in the code under test. At every S, the spread
*across* the four values (max mean − min mean: 0.11s at S=2,000, 0.10s at
S=8,000, 0.15s at S=32,000, 0.51s at S=128,000) is smaller than the
within-cell rep-to-rep spread at that same S. No value is distinguishably
faster or slower than any other at any measured width — the differences are
noise, not signal.

**Chosen constant: `PARALLEL_MIN_CELLS = 512 * 1_024` (unchanged from the
seeded value).** Per the plan: when the sweep shows no measurable difference,
the correct action is to keep the seeded value and record that finding, not
to pick a value out of noise. That is what happened here — recorded as the
finding, not skipped.

## Verification (commit `25e3d4a` + this task's changes)

- `CARGO_TARGET_DIR=/local/dlaub/cargo-target-diag pixi run -e lint cargo test --no-default-features --features conversion`:
  **472 passed, 0 failed, 1 ignored** — matches the stated baseline exactly.
- `CARGO_TARGET_DIR=/local/dlaub/cargo-target-diag pixi run -e lint cargo check --no-default-features`: clean, no warnings.
- `pixi run test`: **1006 passed, 7 skipped, 16 xfailed** — matches the stated
  baseline exactly. No `batch_records_*`/`pack_window_*`/parallel-gate test
  needed updating: `PARALLEL_MIN_CELLS`'s final value is unchanged from HEAD,
  and no Rust test asserts a literal threshold against it (only the two
  budget *functions* are asserted, and those are untouched by this task).

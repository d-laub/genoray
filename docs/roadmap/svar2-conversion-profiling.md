# SVAR2 Conversion Profiling Recipes

Committed, reproducible profiling recipes for the VCF/PGEN → SVAR2 conversion
pipeline. Each script under `scripts/profile/` wraps the same benchmark entry
point (`run_svar2.py`, or `from_pgen` for PGEN inputs) so every profiler
observes the identical workload — differences in the numbers come from the
profiler, not from a changed input or code path.

## Baseline result these recipes exist to track

The current known baseline (see memory `svar2-conversion-reader-bound`): the
VCF → SVAR2 single-contig conversion is **reader-bound** — the reader stage is
~78% of wall time at 1 thread. After prior GT-decode, per-word-pack, and
parallel-pack optimizations, the reader is now **htslib-input-bound**
(zlib inflate + record parsing), not GT-decode. The executor stage,
`dense2sparse_vk`, is the other side of the pipeline; its wall-time share is
the number the later decision gate (Task 11) watches as shard count rises —
if the executor's share climbs enough as sharding increases reader
throughput, it becomes the next bottleneck to attack. These scripts are how
that baseline was established and how it should be re-checked whenever the
pipeline changes.

## Prerequisites

- `perf` (Linux `perf_events`) for `perf_stat.sh`, `perf_record.sh`,
  `perf_sched.sh`.
- `valgrind` (`callgrind`, `callgrind_annotate`) for `callgrind.sh`.
- [`cargo-show-asm`](https://github.com/pacak/cargo-show-asm) (`cargo asm`
  subcommand) for `cargo_asm.sh`.
- All scripts invoke `pixi run python`, so run them from a checkout with the
  pixi environment installed (`pixi s` / `pixi install`).

## `perf_stat.sh`

**Question:** Are all N threads actually busy, or is the pipeline
serializing/blocking despite requesting N threads?

**Invocation:**

```bash
scripts/profile/perf_stat.sh <src> <out-dir> <ref> <threads>
```

Runs `perf stat` around the conversion, collecting `task-clock`,
`context-switches`, `cpu-migrations`, `cache-misses`, `instructions`, and
`cycles`.

**How to read it:**

- Compare `task-clock` (summed CPU-seconds across threads) to wall-clock
  time. `task-clock / wall ≈ threads` means the cores are genuinely busy for
  the whole run. `task-clock / wall << threads` means the run is spending
  significant wall time with most threads idle — i.e. serialization
  somewhere in the pipeline (a stage waiting on a channel, a lock, or I/O).
- High `context-switches` / `cpu-migrations` relative to run length points at
  contention or a scheduler bouncing threads between cores — worth
  cross-checking with `perf_sched.sh`.
- `cache-misses` and `instructions`/`cycles` (IPC) are a coarse signal for
  memory-bound vs. compute-bound work; a low IPC alongside high cache-misses
  suggests the hot loop is stalling on memory rather than executing
  instructions inefficiently.

## `perf_record.sh`

**Question:** Which functions/call stacks actually consume wall time? Used to
confirm inflate/parse dominance in the reader and to track the executor's
share of total time.

**Invocation:**

```bash
scripts/profile/perf_record.sh <src> <out-dir> <ref> <threads>
```

Runs `perf record -g --call-graph dwarf` around the conversion, writes
`perf.data`, then prints the top of `perf report --stdio`.

**How to read it:**

- Look at the top self/children percentages: on the reader-bound baseline,
  zlib inflate and htslib record-parsing frames should dominate — that is
  the expected, already-diagnosed bottleneck, not a bug to chase.
  See [Baseline result](#baseline-result-these-recipes-exist-to-track).
- Watch `dense2sparse_vk` (the executor stage) frames as a group: track their
  combined percentage of total samples. As shard count / thread count rises
  and the reader gets faster, expect this share to climb — that is the
  signal Task 11's decision gate uses to decide whether the executor becomes
  the next thing to optimize.
- `perf report` output is inherently a snapshot of one run; re-run and eyeball
  stability of the top few frames before drawing conclusions from small
  percentage shifts.

## `perf_sched.sh`

**Question:** Where does time go *off*-CPU? Exposes scheduling stalls that
`perf record`'s on-CPU sampling can't see — thread wakeup latency, channel
backpressure, and any accidental serialization between pipeline stages
(e.g. a collector thread waiting on the reader instead of overlapping with
it).

**Invocation:**

```bash
scripts/profile/perf_sched.sh <src> <out-dir> <ref> <threads>
```

Runs `perf sched record`, writes `perf.sched.data`, then prints the top of
`perf sched latency`.

**How to read it:**

- Large "average delay" or "maximum delay" entries for a pipeline-stage
  thread mean that thread is frequently ready-to-run but waiting on the
  scheduler or a blocking channel recv/send — this is the concrete evidence
  for "serialization" hinted at by a low `task-clock/wall` ratio in
  `perf_stat.sh`.
- Cross-reference thread names/TIDs against the pipeline stages (reader,
  collector/merge, `dense2sparse_vk` executor, writer) to attribute stalls to
  a specific stage rather than "the process" in aggregate.
- Use this when `perf_stat.sh` shows `task-clock/wall << threads` and you
  need to know *which* stage is idle and *why* (channel full/empty vs. lock
  contention vs. genuine I/O wait).

## `callgrind.sh`

**Question:** What is the exact instruction/cache cost of one unit of work,
independent of scheduling noise or thread-count effects? Used for
deterministic A/B comparisons of a code change (e.g. "did this change reduce
instructions in the GT-decode loop?").

**Invocation:**

```bash
scripts/profile/callgrind.sh <src> <out-dir> <ref>
```

Note: no `<threads>` argument — the script always runs with `1` thread. Runs
`valgrind --tool=callgrind`, writes `callgrind.out`, then prints the top of
`callgrind_annotate`.

**How to read it:**

- **Callgrind is single-threaded** (valgrind serializes execution under
  instrumentation). Use it strictly for **per-work-item instruction cost**
  comparisons on a small, fixed input — **never** for scaling or
  parallelism conclusions. Any thread-count/scaling question belongs to
  `perf_stat.sh` / `perf_record.sh` / `perf_sched.sh` on an uninstrumented
  binary instead.
- Instruction counts under callgrind are deterministic (unlike wall-clock
  timing), which makes it the right tool for verifying a micro-optimization
  actually reduced work rather than just reducing noise.
- Use a small input on purpose — callgrind's instrumentation overhead is
  large (often 10-50x), so this is not meant to reflect realistic wall-clock
  behavior, only relative instruction/cache cost between two versions of the
  same code run against the same small input.

## `cargo_asm.sh`

**Question:** What machine code did the compiler actually generate for a hot
inner loop — did it inline and vectorize the way we expect?

**Invocation:**

```bash
scripts/profile/cargo_asm.sh <function-path>
```

Where `<function-path>` is a `cargo asm`-style path to the function (e.g.
`genoray::dense2sparse_vk::pack_word`). Requires the
[`cargo-show-asm`](https://github.com/pacak/cargo-show-asm) `cargo asm`
subcommand. Builds with `--no-default-features --features conversion` and a
scratch `CARGO_TARGET_DIR` (`/tmp/genoray-target-$$`) to avoid colliding with
other builds and to sidestep NFS-linker issues on shared filesystems.

**How to read it:**

- Confirm the hot loop is actually inlined into its caller (no unexpected
  `call` to the function under test) and that SIMD instructions appear where
  vectorization is expected (e.g. `pmovmskb`/`vpshufb`-style instructions for
  packed bit manipulation).
- Use this only after `perf_record.sh`/`callgrind.sh` have already
  identified a specific function as hot — it answers "why is this function
  costly" at the codegen level, not "which function is costly."

## Suggested workflow

1. `perf_stat.sh` first, to see whether the run is compute-bound (cores busy)
   or serialized (idle cores).
2. If serialized, `perf_sched.sh` to find which stage/thread is stalling and
   why.
3. If compute-bound, `perf_record.sh` to find which functions dominate wall
   time, and to track the reader-vs-`dense2sparse_vk` split described in
   [Baseline result](#baseline-result-these-recipes-exist-to-track).
4. For a specific hot function found in step 3, `callgrind.sh` for a
   deterministic before/after instruction-count comparison across a code
   change, and `cargo_asm.sh` to inspect the generated codegen directly.

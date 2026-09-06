# SVAR2 Reader Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `reader_workers` a real public knob, narrow the sharded-VCF reorder frontier so the executor is fed continuously, and bound the reorder backlog so peak RSS is predictable at biobank cohort width.

**Architecture:** Three independent Rust changes — a head-exempt byte
backpressure gate in `shard_exec`, a record-sized (rather than base-pair-sized)
work-unit planner in `shard`, and an ordered `(concurrent_chroms,
reader_workers)` planner in `budget` — then one wiring task in
`lib.rs`/`orchestrator.rs`, then the Python/CLI/docs surface, then the
byte-identity gate and the performance repro.

**Tech Stack:** Rust (pyo3, crossbeam-channel, rayon), Python 3.10+, pixi,
pytest, prek/ruff.

**Spec:** `docs/superpowers/specs/2026-09-05-svar2-reader-frontier-design.md`

## Global Constraints

- **Byte-identical output is the hard gate.** Every change here is a
  scheduling/ordering change. The emitted store must be unchanged at every
  `(concurrent_chroms, reader_workers, chunk_size)`. `tests/test_svar2_schedule_invariance.py`
  is the existing gate; Task 7 extends it.
- **Rust tests must be run as `cargo test --no-default-features --features conversion`.**
  Dropping `extension-module` is required or the pyo3 test binary will not
  link; keeping `conversion` is required or you silently skip the entire
  conversion path (341 tests vs 189).
- **`export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub` before any cargo
  invocation.** The repo lives on NFS and cargo bus-errors trying to mmap
  object files from an NFS `target/`. This applies to `git commit` too — the
  prek hooks run `cargo check`/`cargo clippy`.
- **Python tests must be run as `pixi run test`, never bare `pixi run pytest`.**
  `pixi run test` regenerates the VCF fixtures first; without it you get ~277
  bogus `FileNotFoundError`s that look exactly like a broken branch.
- **`pixi run test` does NOT rebuild the Rust extension.** After any Rust
  change, run `maturin develop --release` before any Python-level test that
  must observe it.
- **Do not run long cargo/maturin builds in the background and return early.**
  Run them in the foreground and wait.
- **Commit convention:** Conventional Commits (`feat:`, `fix:`, `refactor:`,
  `test:`, `docs:`, `perf:`). Never edit `CHANGELOG.md` — commitizen owns it.
- **Public API rule:** any change to a name reachable from `import genoray`
  without an underscore MUST update `skills/genoray-api/SKILL.md` in the same
  change. Task 6 covers this.
- **New tuning constants must carry their provenance.** `W_TARGET`,
  `MERGE_RESERVE_DIV`, `UNITS_TARGET_CHUNKS`, `PENDING_BUDGET_CHUNKS` are
  starting values. Each doc comment must say so and name the spec section (D)
  that will set it.

## Parallelization

Tasks 1, 2 and 3 touch three disjoint Rust modules and have no dependency on
each other — **dispatch them in parallel** using
`superpowers:dispatching-parallel-agents` with
`superpowers:subagent-driven-development`. Use Sonnet or weaker for
implementers; reserve Opus for review and for a second pass where an
implementer critically failed.

```
Wave A (parallel):  Task 1 (shard_exec)   Task 2 (shard)   Task 3 (budget)
Wave B (serial):    Task 4 (lib.rs + orchestrator.rs wiring)
Wave C (serial):    Task 5 (Python API + CLI)
Wave D (parallel):  Task 6 (SKILL.md)     Task 7 (byte-identity tests)
Wave E (serial):    Task 8 (performance repro)
```

Subagents default their cwd to the main repo, not this worktree. Every
dispatched agent must be told to `cd` to the worktree root and verify with
`git rev-parse --show-toplevel` before touching anything.

---

### Task 1: Head-exempt byte backpressure in the shard collector

**Files:**
- Modify: `src/shard_exec.rs` (add `Frontier`, a `ReorderBuffer::head()`
  accessor, a `pending_budget_bytes` parameter on `run`, the producer-side
  `admit` call, and the collector-side `publish` call)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `pub fn run(chrom: &str, units: Vec<WorkUnit>, workers: usize, make_assembler: F, err_context: G, chunk_size: usize, tx_dense: &Sender<DenseChunk>, worker_tids: &Mutex<Vec<i32>>, pending_gauge: &PendingGauge, pending_budget_bytes: u64) -> Result<ShardTotals, ConversionError>`
    — one new trailing parameter. `u64::MAX` means unbounded.
  - `impl ReorderBuffer { pub fn head(&self) -> usize }`

**Background:** `ReorderBuffer` hands out global chunk ids in strict
`(ordinal, local)` order. Only the unit sitting on the head streams; every
other in-flight unit's chunks pile into `PendingBacklog`, which today has an
observing gauge but no ceiling. At 5,000 variants × 535k samples one pending
chunk is hundreds of MB.

The fix must never block the unit that owns the head, or the pipeline
deadlocks. It is safe because the work queue is a FIFO MPMC channel seeded in
ordinal order, so units are dequeued in ascending ordinal order: if unit `h`
is still queued, no worker can be holding any `j > h`, so the head is always
either already done or in flight at an exempt worker.

- [ ] **Step 1: Write the failing tests**

Add to the `mod tests` block at the bottom of `src/shard_exec.rs`:

```rust
    use super::Frontier;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    /// The unit that owns the head is NEVER parked, even with the backlog
    /// over budget. This is the whole deadlock-freedom argument: if the head
    /// could park, nothing would ever advance it.
    #[test]
    fn admit_never_parks_the_head_unit() {
        let f = Frontier::new(0);
        f.publish(u64::MAX, 7);
        let cancel = AtomicBool::new(false);
        // Would hang forever if the head were subject to the budget.
        f.admit(7, &cancel);
    }

    /// A non-head producer parks while the backlog is over budget, and wakes
    /// once the head advances past it -- not only when bytes drop.
    #[test]
    fn admit_parks_a_non_head_unit_until_the_head_advances() {
        let f = Frontier::new(0);
        f.publish(100, 0);
        let cancel = AtomicBool::new(false);
        std::thread::scope(|s| {
            let waiter = s.spawn(|| {
                f.admit(3, &cancel);
            });
            // Give the waiter a chance to actually park before we release it.
            std::thread::sleep(Duration::from_millis(50));
            assert!(!waiter.is_finished(), "unit 3 must park behind head 0");
            f.publish(100, 3);
            waiter.join().unwrap();
        });
    }

    /// A parked producer must observe `cancel` so the error path can tear the
    /// pool down instead of hanging in `join()`.
    #[test]
    fn admit_releases_a_parked_unit_on_cancel() {
        let f = Frontier::new(0);
        f.publish(100, 0);
        let cancel = AtomicBool::new(false);
        std::thread::scope(|s| {
            let waiter = s.spawn(|| {
                f.admit(3, &cancel);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(!waiter.is_finished());
            cancel.store(true, Ordering::Relaxed);
            f.wake_all();
            waiter.join().unwrap();
        });
    }

    /// A budget of `u64::MAX` disables the gate entirely -- the PGEN path and
    /// the existing tests rely on this.
    #[test]
    fn an_unbounded_budget_never_parks_anything() {
        let f = Frontier::new(u64::MAX);
        f.publish(u64::MAX, 0);
        let cancel = AtomicBool::new(false);
        f.admit(99, &cancel);
    }

    /// The reorder head must be readable by the collector so it can publish
    /// it; without this the gate has nothing to compare against.
    #[test]
    fn reorder_buffer_exposes_its_head() {
        let mut rb = ReorderBuffer::new(2);
        assert_eq!(rb.head(), 0);
        rb.push(0, 0, false, &mut |_gid, _tag| {});
        rb.push(0, 0, true, &mut |_gid, _tag| {});
        assert_eq!(rb.head(), 1, "head advances past a completed ordinal");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
cargo test --no-default-features --features conversion --lib shard_exec
```

Expected: FAIL — `cannot find type Frontier in this scope`, and `no method
named head found for struct ReorderBuffer`.

- [ ] **Step 3: Add the `head()` accessor**

In `src/shard_exec.rs`, inside `impl ReorderBuffer`, above `push`:

```rust
    /// The ordinal currently being emitted/awaited. The collector publishes
    /// this to [`Frontier`] so non-head producers know whether they are
    /// subject to the backlog budget.
    pub fn head(&self) -> usize {
        self.head
    }
```

- [ ] **Step 4: Add the `Frontier` type**

In `src/shard_exec.rs`, after the `PendingBacklog` impl block, add:

```rust
/// Producer-side admission gate over the collector's reorder backlog.
///
/// `PendingBacklog` is unbounded by construction: every unit ahead of
/// `ReorderBuffer::head` buffers everything it produces. At biobank cohort
/// width one buffered chunk is hundreds of MB, so the backlog -- not
/// `max_mem`, and not `workers * chunk_bytes` -- becomes the dominant peak-RSS
/// term, and it is invisible to the caller. This gate parks a producer whose
/// unit is ahead of the head once the backlog exceeds `budget_bytes`.
///
/// # Why this cannot deadlock
///
/// The unit that owns the current head is NEVER parked. That exemption is
/// sufficient because the work queue is a FIFO MPMC channel seeded in ordinal
/// order, so units are dequeued in ascending ordinal order. Let `h` be the
/// head. If unit `h` were still queued, no worker could be holding any
/// `j > h` (it would have had to be dequeued before `h`), so every worker
/// would hold an ordinal `< h` -- but those are all complete by definition of
/// the head. Contradiction. So `h` is always either already done, or in
/// flight at a worker that is exempt. That worker runs to its `Done`, the
/// collector advances the head, and the condvar wakes the next holder.
///
/// The other two blocking edges are ordinary backpressure, not deadlock: a
/// worker blocked on the bounded `tx_res` is drained by the collector, and a
/// collector blocked on the bounded `tx_dense` is drained by the executor.
pub(crate) struct Frontier {
    state: Mutex<FrontierState>,
    cv: std::sync::Condvar,
    /// `u64::MAX` disables the gate (the PGEN path, which runs a single unit,
    /// and the unit tests).
    budget_bytes: u64,
}

/// Head and backlog bytes live under ONE mutex so a parked producer's
/// predicate reads a consistent pair. Split atomics would let a waiter see a
/// stale head against fresh bytes and park after the head had already passed
/// it -- a lost wakeup with no one left to issue another.
struct FrontierState {
    head: usize,
    pending_bytes: u64,
}

impl Frontier {
    pub(crate) fn new(budget_bytes: u64) -> Self {
        Self {
            state: Mutex::new(FrontierState {
                head: 0,
                pending_bytes: 0,
            }),
            cv: std::sync::Condvar::new(),
            budget_bytes,
        }
    }

    /// Producer side: block until this unit may send another chunk.
    /// Returns immediately for the unit that owns the head, and on cancel.
    pub(crate) fn admit(&self, ordinal: usize, cancel: &AtomicBool) {
        let mut st = self.state.lock().unwrap();
        while !cancel.load(Ordering::Relaxed)
            && ordinal > st.head
            && st.pending_bytes > self.budget_bytes
        {
            st = self.cv.wait(st).unwrap();
        }
    }

    /// Collector side: publish the whole backlog state after handling one
    /// message. Publishing the totals (rather than deltas) keeps this
    /// impossible to get wrong from the collector's two call sites, and the
    /// cost is one uncontended lock per message.
    pub(crate) fn publish(&self, pending_bytes: u64, head: usize) {
        {
            let mut st = self.state.lock().unwrap();
            st.pending_bytes = pending_bytes;
            st.head = head;
        }
        self.cv.notify_all();
    }

    /// Wake every parked producer without changing the state -- the teardown
    /// path, after `cancel` is set.
    ///
    /// The lock is acquired and immediately dropped ON PURPOSE; do not
    /// "simplify" it away. `admit`'s predicate reads `cancel`, a plain
    /// `AtomicBool` this mutex does not protect, so a bare `notify_all` races:
    /// a worker can evaluate `!cancel` as true while holding the lock, an
    /// erroring worker can then `cancel.store(true)` and `notify_all` before
    /// that worker registers on the condvar, and the worker parks forever. It
    /// never returns, so its `tx_res` clone never drops, so the collector's
    /// `recv` loop never ends, so `join()` hangs and the conversion stalls
    /// with no error surfaced. Taking the lock orders store-then-notify
    /// against check-then-park: either we block until `wait` releases it (the
    /// worker is parked and will see the notify), or the worker has not yet
    /// loaded `cancel` and will see `true`.
    pub(crate) fn wake_all(&self) {
        drop(self.state.lock().unwrap());
        self.cv.notify_all();
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
cargo test --no-default-features --features conversion --lib shard_exec
```

Expected: PASS, all five new tests plus the three pre-existing
`ReorderBuffer`/`PendingBacklog` tests.

- [ ] **Step 6: Commit the gate**

```bash
git add src/shard_exec.rs
git commit -m "feat(shard_exec): add a head-exempt backlog admission gate"
```

- [ ] **Step 7: Wire the gate into `run`**

Three edits in `src/shard_exec.rs`:

**(a)** Add the parameter. Change the `pub fn run` signature's last parameter
list to end with:

```rust
    pending_gauge: &crate::monitor::PendingGauge,
    /// Byte ceiling on the collector's reorder backlog. `u64::MAX` disables
    /// the gate. See [`Frontier`] for why the head unit is exempt.
    pending_budget_bytes: u64,
) -> Result<ShardTotals, ConversionError>
```

**(b)** Construct the frontier next to `cancel`, just after
`let cancel = Arc::new(AtomicBool::new(false));`:

```rust
    let frontier = Arc::new(Frontier::new(pending_budget_bytes));
```

and clone it into each worker alongside `cancel` (inside the
`for i in 0..workers` loop, next to `let cancel = Arc::clone(&cancel);`):

```rust
            let frontier = Arc::clone(&frontier);
```

**(c)** In the worker's inner `loop`, park before sending a chunk. Replace:

```rust
                                Ok(Some(chunk)) => {
                                    trace_ll!(
```

with:

```rust
                                Ok(Some(chunk)) => {
                                    // Park if the backlog is over budget and
                                    // this unit is not the head. See
                                    // `Frontier` for the deadlock argument.
                                    frontier.admit(unit.ordinal, &cancel);
                                    if cancel.load(Ordering::Relaxed) {
                                        return;
                                    }
                                    trace_ll!(
```

and in the worker's two error arms, after each
`cancel.store(true, Ordering::Relaxed);`, add:

```rust
                                frontier.wake_all();
```

(there are two such sites: the `make_assembler` failure and the
`read_next_chunk` failure).

**(d)** In the collector loop, publish after handling each message. At the end
of the `Msg::Chunk` arm, after the `rb.push(...)` call, add:

```rust
                    frontier.publish(pending.bytes, rb.head());
```

At the end of the `Msg::Done` arm, immediately before the
`if done_count == n_units {` check, add:

```rust
                    frontier.publish(pending.bytes, rb.head());
```

In the `Msg::Err` arm, after `cancel.store(true, Ordering::Relaxed);`, add:

```rust
                    frontier.wake_all();
```

And after the `while let Ok(msg) = rx_res.recv()` loop closes, before the
`for (name, handle) in handles` join loop, add:

```rust
        // Defensive: no producer should still be parked once the collector
        // has seen every unit's `Done`, but a `break` out of the recv loop on
        // the error path can leave one, and `join()` below would hang.
        frontier.wake_all();
```

`pending.bytes` is a private field of `PendingBacklog` in the same module, so
it is directly readable from the collector.

- [ ] **Step 8: Fix the two call sites so it compiles**

`src/orchestrator.rs` calls `shard_exec::run` twice (the VCF branch and the
PGEN branch). Pass `u64::MAX` at both for now — Task 4 threads the real
budget through the VCF one. Add the argument after `&pending_gauge`:

```rust
                                u64::MAX,
```

- [ ] **Step 9: Verify the whole Rust suite still passes**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
cargo test --no-default-features --features conversion
```

Expected: PASS. Confirm the run reports ~341 tests, not ~189 — a lower count
means the `conversion` feature was dropped and the conversion path was not
compiled at all.

- [ ] **Step 10: Commit**

```bash
git add src/shard_exec.rs src/orchestrator.rs
git commit -m "feat(shard_exec): bound the reorder backlog with a byte budget"
```

---

### Task 2: Size work units by records, not base pairs

**Files:**
- Modify: `src/shard.rs` (add three constants and `plan_unit_count`, plus
  tests)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `pub const UNITS_TARGET_CHUNKS: usize = 4;`
  - `pub const MAX_UNITS_PER_CONTIG: usize = 4096;`
  - `pub fn plan_unit_count(contig_records: Option<u64>, workers: usize, chunk_size: usize, overshard_factor: usize) -> usize`

**Background:** `orchestrator.rs` currently asks for
`reader_workers * OVERSHARD_FACTOR` units per contig, with
`OVERSHARD_FACTOR = 4`. At 20 workers that is 80 units over a 10.1 M-record
contig — **four waves**. Within a wave, only the head unit streams, so the
executor's input rate is capped at one reader's rate no matter how many
readers run; when the head clears, the wave releases in a burst. Sizing units
by record count instead makes the head advance at the aggregate reader rate.

`contig_records` must be `Some` **only** when the caller holds exact per-contig
record counts. `ContigCosts` has a second tier whose values are base-pair
contig lengths — a different unit entirely — flagged by `exact_counts`. Task 4
enforces that; this task only has to honour `None`.

- [ ] **Step 1: Write the failing tests**

Add to the `mod tests` block at the bottom of `src/shard.rs`:

```rust
    #[test]
    fn plan_unit_count_falls_back_to_overshard_without_exact_counts() {
        // No exact record count -> the pre-#169 shape, so a header-length
        // fallback tier's base-pair values can never be mistaken for records.
        assert_eq!(plan_unit_count(None, 20, 5_000, 4), 80);
        assert_eq!(plan_unit_count(None, 1, 5_000, 4), 4);
    }

    #[test]
    fn plan_unit_count_targets_four_chunks_of_records_per_unit() {
        // chr21 of the #169 report: 10.1M records at chunk_size 5,000.
        // 4 chunks/unit -> 20,000 records/unit -> 505 units, versus the 80
        // the old `workers * OVERSHARD_FACTOR` shape produced at w=20.
        assert_eq!(plan_unit_count(Some(10_100_000), 20, 5_000, 4), 505);
        // Independent of the worker count once the record floor dominates.
        assert_eq!(plan_unit_count(Some(10_100_000), 3, 5_000, 4), 505);
    }

    #[test]
    fn plan_unit_count_floors_at_the_worker_count() {
        // A tiny contig must still give every worker something to steal.
        assert_eq!(plan_unit_count(Some(100), 20, 5_000, 4), 20);
    }

    #[test]
    fn plan_unit_count_caps_at_max_units_per_contig() {
        // Each unit is an independent indexed fetch, so the unit count is
        // also a seek count -- the cap wins over both the record target and
        // the worker floor.
        assert_eq!(
            plan_unit_count(Some(u64::MAX), 20, 1, 4),
            MAX_UNITS_PER_CONTIG
        );
        assert_eq!(
            plan_unit_count(Some(100), 100_000, 5_000, 4),
            MAX_UNITS_PER_CONTIG
        );
    }

    #[test]
    fn plan_unit_count_is_never_zero() {
        assert_eq!(plan_unit_count(Some(0), 1, 5_000, 4), 1);
        assert_eq!(plan_unit_count(None, 0, 0, 0), 1);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
cargo test --no-default-features --features conversion --lib shard
```

Expected: FAIL — `cannot find function plan_unit_count in this scope`.

- [ ] **Step 3: Implement the constants and the function**

Add to `src/shard.rs`, after the `WorkUnit` struct and before `plan_ranges`:

```rust
/// How many `chunk_size`-record chunks one work unit should cover.
///
/// This is the knob that trades indexed-fetch padding against reorder-frontier
/// width. Every unit re-decodes `normalize::L_MAX` (1000) bp of padding on
/// each side, so smaller units cost redundant decode; larger units make the
/// reorder head advance in coarser jumps, which is what starved the executor
/// in issue #169. At 4 chunks of 5,000 records on chr21 density (~220
/// records/kbp) a unit spans ~91 kbp, so padding is ~2% of decoded records.
///
/// STARTING VALUE, not a fitted constant -- set it from the sweep in
/// `docs/superpowers/specs/2026-09-05-svar2-reader-frontier-design.md`
/// section D and record the measurement here when you do.
pub const UNITS_TARGET_CHUNKS: usize = 4;

/// Hard cap on work units per contig. Each unit is an independent indexed
/// fetch, so this is also a per-contig seek cap -- the guard against a
/// pathological `chunk_size`/record-count combination asking for millions of
/// seeks. The cap wins over the worker floor.
pub const MAX_UNITS_PER_CONTIG: usize = 4096;

/// How many work units to split one contig into.
///
/// `contig_records` is `Some` ONLY when the caller holds EXACT per-contig
/// record counts (`contig_cost::ContigCosts::exact_counts`). The
/// header-length fallback tier's values are base pairs, a different unit
/// entirely, and feeding them here would mis-size the frontier by orders of
/// magnitude -- pass `None` and take the pre-#169 `workers * overshard_factor`
/// shape instead.
pub fn plan_unit_count(
    contig_records: Option<u64>,
    workers: usize,
    chunk_size: usize,
    overshard_factor: usize,
) -> usize {
    let workers = workers.max(1);
    match contig_records {
        None => workers
            .saturating_mul(overshard_factor.max(1))
            .min(MAX_UNITS_PER_CONTIG)
            .max(1),
        Some(records) => {
            let per_unit = (UNITS_TARGET_CHUNKS as u64).saturating_mul(chunk_size.max(1) as u64);
            let target = records.div_ceil(per_unit.max(1));
            // Order matters: the worker floor keeps every worker fed on a
            // small contig, then the seek cap overrides it -- a worker count
            // above the cap gets the cap, not a runaway unit count.
            usize::try_from(target)
                .unwrap_or(MAX_UNITS_PER_CONTIG)
                .max(workers)
                .min(MAX_UNITS_PER_CONTIG)
                .max(1)
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
cargo test --no-default-features --features conversion --lib shard
```

Expected: PASS, all five new tests plus the four pre-existing `plan_ranges`
tests.

- [ ] **Step 5: Commit**

```bash
git add src/shard.rs
git commit -m "feat(shard): size work units by record count instead of bp span"
```

---

### Task 3: Ordered `(concurrent_chroms, reader_workers)` planning

**Files:**
- Modify: `src/budget.rs` (add three constants and two helpers, rewrite
  `plan_sharded`, change `PlanInputs.reader_workers` to `Option<usize>`,
  delete the now-dead `ThreadPlan.reader_workers` field and `fn reader_workers`)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `pub const MERGE_RESERVE_DIV: usize = 4;`
  - `pub const W_TARGET: usize = 8;`
  - `pub const PENDING_BUDGET_CHUNKS: u64 = 8;`
  - `pub fn pending_budget_bytes(chunk_bytes: u64) -> u64` — the ceiling handed
    to `shard_exec::run`; the collector's backlog map ONLY
  - `pub fn in_flight_budget_bytes(chunk_bytes: u64, workers: usize) -> u64` —
    backlog ceiling + the `workers * 2` bounded-channel capacity; the number
    the memory law must use. Task 4 passes `pending_budget_bytes` to
    `shard_exec::run` and never passes this one anywhere.
  - `pub fn reader_pool_cores(usable_cores: usize) -> usize`
  - `PlanInputs { usable_cores, n_contigs, n_samples, chunk_bytes, max_mem_bytes, reader_workers: Option<usize>, ram }`
    — **`reader_workers` changes type from `usize` to `Option<usize>`.**
    `None` means "derive it"; `Some(w)` is an explicit caller request that is
    honoured or refused, never silently shrunk.
  - `pub fn plan_sharded(inp: PlanInputs) -> Result<ShardedPlan, PlanError>`
    — unchanged signature, new semantics: it now chooses `reader_workers` too.
  - `ThreadPlan` loses its `reader_workers` field.

**Background:** `DEFAULT_READER_WORKERS = 3` in `src/lib.rs:160` is the only
source of the per-contig reader count on the VCF path, and `threads=` does not
influence it. Meanwhile `budget::reader_workers(usable_cores, concurrent)`
already derives a value from cores, is stored in `ThreadPlan.reader_workers`,
and is **never read** — confirm with
`rg 'plan\.reader_workers' src/`, which returns nothing.

Once `w` derives from cores-per-contig, `cc` and `w` are mutually dependent.
Break it by planning in a fixed order: reserve the merge tail, choose `cc`
preferring depth, fill the depth, then re-check memory.

Two things that are easy to get wrong and are why the algorithm below is
written out in full:

1. **Reserve the merge tail *before* sizing readers.** `processing_threads_for`
   sizes the var_key gather pool and `dense_merge`'s bit-transpose from
   whatever cores the readers leave over. Spending every leftover core on
   readers floors that pool at 1 and gives back the 2.77–2.82× merge-tiling
   win from commit `c49d1d7`. The reserve constrains the *derive* path only:
   an explicit `reader_workers` larger than `pool` is still honoured, because
   this planner refuses only on the memory budget, never on cores. Say so in
   the doc rather than claiming the reserve holds unconditionally.
2. **When memory is tight, shrink `w` first, not `cc`.** Decrementing `cc`
   *raises* `w` (`w = pool / cc − 1`), which raises per-contig memory — a loop
   that decrements `cc` first does not converge.

- [ ] **Step 1: Write the failing tests**

Add to the `mod tests` block in `src/budget.rs`:

```rust
    #[test]
    fn reader_pool_reserves_a_quarter_of_cores_for_the_merge_tail() {
        // Spending every leftover core on readers floors `processing_threads_for`
        // at 1 and gives back the 2.77-2.82x merge-tiling win (c49d1d7).
        assert_eq!(reader_pool_cores(31), 23); // 31 - ceil(31/4)=8
        assert_eq!(reader_pool_cores(16), 12); // 16 - 4
        assert_eq!(reader_pool_cores(1), 1); // never zero
        assert_eq!(reader_pool_cores(0), 1);
    }

    #[test]
    fn pending_budget_is_a_fixed_multiple_of_the_chunk() {
        // Deliberately independent of max_mem: plan_sharded consumes the
        // budget to pick cc, and max_mem is what bounds cc, so deriving one
        // from the other is circular.
        assert_eq!(pending_budget_bytes(10_000_000), 80_000_000);
        assert_eq!(pending_budget_bytes(0), 0);
    }

    #[test]
    fn in_flight_budget_adds_the_bounded_result_channel() {
        // `Frontier` bounds only the collector's backlog map. `shard_exec`'s
        // `tx_res` holds up to `workers * 2` more assembled chunks, so
        // planning against the backlog alone under-counts real peak by 2*w
        // chunks per contig. The readers' own `w` working chunks are NOT
        // added here -- they live in the planner's `kappa * w` term.
        //   w=3:  8*10MB + 2*3*10MB  =  80 +  60 = 140 MB
        //   w=16: 8*10MB + 2*16*10MB =  80 + 320 = 400 MB
        assert_eq!(in_flight_budget_bytes(10_000_000, 3), 140_000_000);
        assert_eq!(in_flight_budget_bytes(10_000_000, 16), 400_000_000);
        // Never cheaper than the backlog ceiling alone, even at w=0.
        assert_eq!(
            in_flight_budget_bytes(10_000_000, 0),
            100_000_000,
            "workers floors at 1"
        );
    }

    #[test]
    fn plan_prefers_depth_over_breadth_on_the_reported_machine() {
        // The #169 machine: 32 vCPU -> 31 usable, 22 contigs, no budget.
        //   pool      = 31 - 8 = 23
        //   depth_cap = 23 / (1 + W_TARGET=8) = 2
        //   cc        = min(22, 2) = 2
        //   w         = 23 / 2 - 1 = 10
        // Today the same machine yields cc=7, w=3.
        let plan = plan_sharded(PlanInputs {
            usable_cores: 31,
            n_contigs: 22,
            n_samples: 535_662,
            chunk_bytes: 10_000_000,
            max_mem_bytes: None,
            reader_workers: None,
            ram: RamLaw::VCF,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 2,
                reader_workers: 10,
            }
        );
    }

    #[test]
    fn plan_never_exceeds_the_contig_count() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 96,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 1_000_000,
            max_mem_bytes: None,
            reader_workers: None,
            ram: RamLaw::VCF,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 1);
        // pool = 96 - 24 = 72; w = 72/1 - 1 = 71.
        assert_eq!(plan.reader_workers, 71);
    }

    #[test]
    fn a_tight_budget_shrinks_workers_before_contigs() {
        // Shrinking cc first would RAISE w (w = pool/cc - 1) and so raise
        // per-contig memory -- the loop would not converge. The plan must
        // give back readers first.
        let roomy = plan_sharded(PlanInputs {
            usable_cores: 31,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: None,
            reader_workers: None,
            ram: RamLaw::VCF,
        })
        .unwrap();
        let tight = plan_sharded(PlanInputs {
            usable_cores: 31,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_200_000_000),
            reader_workers: None,
            ram: RamLaw::VCF,
        })
        .unwrap();
        assert!(
            tight.reader_workers < roomy.reader_workers,
            "a tight budget must cost readers: {tight:?} vs {roomy:?}"
        );
        assert!(tight.reader_workers >= 1);
    }

    #[test]
    fn an_explicit_reader_workers_is_honoured_or_refused_never_shrunk() {
        let inp = PlanInputs {
            usable_cores: 31,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: None,
            reader_workers: Some(24),
            ram: RamLaw::VCF,
        };
        assert_eq!(plan_sharded(inp).unwrap().reader_workers, 24);

        // Same request against a budget that cannot hold it: refuse rather
        // than quietly hand back a slower plan the caller did not ask for.
        assert!(matches!(
            plan_sharded(PlanInputs {
                max_mem_bytes: Some(600_000_000),
                ..inp
            }),
            Err(PlanError::InsufficientMemory { .. })
        ));
    }

    #[test]
    fn a_budget_below_the_cohort_baseline_still_reports_the_baseline() {
        // The `budget_mb < baseline_mb` branch of PlanError's Display is the
        // one that tells a caller chunk_size cannot help them; keep it
        // reachable.
        let err = plan_sharded(PlanInputs {
            usable_cores: 31,
            n_contigs: 1,
            n_samples: 10_000_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_000_000),
            reader_workers: None,
            ram: RamLaw::VCF,
        })
        .unwrap_err();
        let PlanError::InsufficientMemory {
            budget_mb,
            baseline_mb,
            ..
        } = err;
        assert!(budget_mb < baseline_mb);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
cargo test --no-default-features --features conversion --lib budget
```

Expected: FAIL — `cannot find function reader_pool_cores`, and
`expected usize, found Option<usize>` at the `reader_workers: None` sites.

- [ ] **Step 3: Add the constants and helpers**

In `src/budget.rs`, next to the other `pub const`s near the top:

```rust
/// Denominator of the usable-core fraction reserved for the per-contig merge
/// tail: `ceil(usable / MERGE_RESERVE_DIV)` cores never go to readers.
///
/// `processing_threads_for` sizes `merge.rs`'s var_key gather pool and
/// `dense_merge`'s bit-transpose from whatever the readers leave over.
/// Spending every leftover core on readers floors that pool at 1 and gives
/// back the 2.77-2.82x merge-tiling win from commit c49d1d7.
///
/// STARTING VALUE -- set it from the sweep in
/// `docs/superpowers/specs/2026-09-05-svar2-reader-frontier-design.md`
/// section D and record the measurement here when you do.
pub const MERGE_RESERVE_DIV: usize = 4;

/// Target readers per concurrent contig when choosing `concurrent_chroms`.
///
/// Depth is preferred over breadth: each extra concurrent contig costs a
/// dedicated executor core and a full `per_contig_mb` of RAM while buying no
/// extra reader cores, because the readers are CPU-saturated and total read
/// throughput tracks total reader cores however they are partitioned. The old
/// argument for shallow depth ("surplus readers steal cores from other
/// contigs' executors") was really an argument about the reorder frontier,
/// which `shard::plan_unit_count` now fixes.
///
/// STARTING VALUE -- see the same spec, section D.
pub const W_TARGET: usize = 8;

/// Chunks of reorder backlog `shard_exec`'s collector may hold before non-head
/// readers park. Floored at 2 in [`pending_budget_bytes`] -- a one-chunk
/// budget serializes the frontier back to the head, which is the pathology
/// this whole change exists to remove.
///
/// STARTING VALUE -- see the same spec, section D.
pub const PENDING_BUDGET_CHUNKS: u64 = 8;

/// Byte ceiling for `shard_exec`'s reorder backlog -- the collector's
/// `PendingBacklog` map ALONE. This is the number handed to
/// `shard_exec::run`; it is NOT the pipeline's total in-flight bytes. Use
/// [`in_flight_budget_bytes`] for memory planning.
///
/// Deliberately independent of `max_mem`: `plan_sharded` consumes this budget
/// to choose `concurrent_chroms`, and `max_mem` is what bounds
/// `concurrent_chroms`, so deriving one from the other is circular. Instead
/// the budget is a fixed multiple of the chunk, and `max_mem` constrains
/// concurrency through the law in `plan_sharded`.
///
/// `chunk_bytes` must be the `resident_chunk_size`-narrowed value (see
/// `lib.rs`), not `chunk_size * per_variant_bytes`: `BitGrid3::zeros` is a
/// calloc, so nominal chunk bytes are address space, not RSS.
pub fn pending_budget_bytes(chunk_bytes: u64) -> u64 {
    chunk_bytes.saturating_mul(PENDING_BUDGET_CHUNKS.max(2))
}

/// Chunk-shaped bytes one contig can hold in flight, for the memory law.
///
/// `Frontier` bounds only the collector's `PendingBacklog`. Two other places
/// hold assembled chunks at the same time, and pricing only the backlog
/// under-counts real peak RSS:
///
/// - the collector's `PendingBacklog` map: [`pending_budget_bytes`], enforced;
/// - `shard_exec`'s bounded result channel `tx_res`, capacity `workers * 2`,
///   enforced by the channel itself;
/// - each reader's own working chunk -- including the assembled chunk a parked
///   producer is holding, since `admit` is called BEFORE `tx_res.send`. That
///   term is `w` chunks and is already carried by `ram.kappa * w * chunk_MB`
///   in `plan_sharded`, so it is deliberately NOT repeated here.
///
/// So this returns the backlog ceiling plus the channel capacity. Adding the
/// channel term makes the planner strictly more conservative; it is not a
/// refit of `RamLaw`'s fitted coefficients.
pub fn in_flight_budget_bytes(chunk_bytes: u64, workers: usize) -> u64 {
    pending_budget_bytes(chunk_bytes)
        .saturating_add(chunk_bytes.saturating_mul(2u64.saturating_mul(workers.max(1) as u64)))
}

/// Cores available to executors and readers after the merge-tail reserve.
/// Floored at 1 so a single-core host still plans.
pub fn reader_pool_cores(usable_cores: usize) -> usize {
    let usable = usable_cores.max(1);
    usable
        .saturating_sub(usable.div_ceil(MERGE_RESERVE_DIV))
        .max(1)
}
```

- [ ] **Step 4: Change `PlanInputs.reader_workers` to `Option<usize>`**

In `src/budget.rs`, in `pub struct PlanInputs`, replace:

```rust
    pub reader_workers: usize,
```

with:

```rust
    /// `None` asks the planner to derive the reader count from the core
    /// budget. `Some(w)` is an explicit caller request, honoured or refused
    /// with `InsufficientMemory` -- never silently shrunk, because a caller
    /// who asked for 24 readers and got 3 has no way to find out.
    pub reader_workers: Option<usize>,
```

- [ ] **Step 5: Rewrite `plan_sharded`**

Replace the whole body of `pub fn plan_sharded` (keep the signature) and add
the `memory_fits` helper below it:

```rust
pub fn plan_sharded(inp: PlanInputs) -> Result<ShardedPlan, PlanError> {
    let n_contigs = inp.n_contigs.max(1);
    // Step 1: reserve the merge tail before anything else claims cores.
    let pool = reader_pool_cores(inp.usable_cores);

    // An explicit request is honoured or refused -- never silently shrunk.
    // Note that `w` here is the CALLER's, so concurrency is sized from it and
    // NOT from `W_TARGET`: the caller has already stated its per-contig reader
    // demand, and reserving depth for a hypothetical `W_TARGET` readers would
    // strand cores. PGEN pins `w = 1`, so seeding from `depth_cap` would cost
    // it 2.7-4x its contig concurrency (48-core host: 8 -> 3). Concurrency may
    // still come down to make it fit -- that is the planner's own knob, not
    // the caller's.
    if let Some(req) = inp.reader_workers {
        let w = req.max(1);
        let mut cc = std::cmp::min(n_contigs, (pool / (1 + w)).max(1));
        while cc > 1 && memory_fits(&inp, cc, w).is_err() {
            cc -= 1;
        }
        memory_fits(&inp, cc, w)?;
        return Ok(ShardedPlan {
            concurrent_chroms: cc,
            reader_workers: w,
        });
    }

    // Step 2: choose the contig concurrency, preferring depth. `W_TARGET` is
    // the DERIVE path's standing guess at per-contig reader demand; it has no
    // business on the explicit path above, which knows the real number.
    let depth_cap = (pool / (1 + W_TARGET)).max(1);
    let mut cc = std::cmp::min(n_contigs, depth_cap);

    loop {
        // Step 3: fill the depth -- one core for this contig's executor, the
        // rest for its readers.
        let w_max = (pool / cc).saturating_sub(1).max(1);
        // Step 4: memory re-check. Give back READERS before contigs:
        // decrementing `cc` raises `w`, which raises per-contig memory, so a
        // contig-first loop does not converge.
        if let Some(w) = (1..=w_max)
            .rev()
            .find(|&w| memory_fits(&inp, cc, w).is_ok())
        {
            return Ok(ShardedPlan {
                concurrent_chroms: cc,
                reader_workers: w,
            });
        }
        if cc == 1 {
            // The scan above covered w=1 at cc=1, so this is a genuine error.
            return Err(memory_fits(&inp, 1, 1)
                .expect_err("cc=1, w=1 just failed the scan above"));
        }
        cc -= 1;
    }
}

/// Does one `(cc, w)` shape fit the caller's byte budget?
///
/// ```text
///   baseline   = base_mb + per_sample_mb * samples
///   per_contig = per_contig_mb + kappa * w * chunk_MB + in_flight_MB(w)
///   fits       <=> budget - baseline >= cc * per_contig
/// ```
///
/// The backlog term was `kappa * (w + (w-1)) * chunk_MB` before issue #169 --
/// quadratic-ish in `w`, which is what made a large reader count unaffordable.
/// `shard_exec::Frontier` now ENFORCES a fixed backlog ceiling, so it becomes
/// an explicit additive budget. This is not a refit of `RamLaw::VCF`: the
/// fitted coefficients are untouched. It is NOT uniformly more conservative,
/// though. Per unit of `w` the old term charged `2 * kappa` (12.21 chunk-MB)
/// against this one's `kappa + 2` (8.11), so the two cross at
/// `w = 14.1058 / 4.1058 ~= 3.44`: at `w <= 3` this law charges MORE, at
/// `w >= 4` it charges LESS -- 21% less per contig at `w = 10` and a 10 MB
/// chunk, which is what the derive path picks on the machine in issue #169.
/// That percentage is chunk-dependent, since `per_contig_mb` is not: on the
/// chunk-scaled part alone the reduction is 23.2%, which is what it approaches
/// at a production-sized chunk (3125 MB at 500,000 samples). That is defensible
/// only because `Frontier` now ENFORCES the backlog ceiling the `(w-1)` term
/// merely fitted; if that enforcement is ever removed or bypassed, this law
/// under-predicts peak RSS at exactly the reader counts #169 exists to reach.
///
/// `in_flight_MB` is [`in_flight_budget_bytes`], NOT `pending_budget_bytes`:
/// the enforced backlog ceiling plus the `workers * 2` capacity of
/// `shard_exec`'s bounded result channel. Pricing only the backlog
/// under-counts real peak by up to `2 * w` chunks per contig. The readers'
/// own `w` working chunks stay in the `kappa` term and are not double-counted
/// here.
fn memory_fits(inp: &PlanInputs, cc: usize, w: usize) -> Result<(), PlanError> {
    let Some(budget) = inp.max_mem_bytes else {
        return Ok(());
    };
    let budget_mb = budget as f64 / 1e6;
    let baseline_mb = inp.ram.base_mb + inp.ram.per_sample_mb * inp.n_samples as f64;
    let per_contig_mb = inp.ram.per_contig_mb
        + inp.ram.kappa * w as f64 * (inp.chunk_bytes as f64 / 1e6)
        + in_flight_budget_bytes(inp.chunk_bytes, w) as f64 / 1e6;
    let needed_mb = baseline_mb + per_contig_mb * cc as f64;
    if budget_mb < needed_mb {
        return Err(PlanError::InsufficientMemory {
            needed_mb,
            budget_mb,
            baseline_mb,
        });
    }
    Ok(())
}
```

- [ ] **Step 6: Delete the dead `reader_workers` derivation**

Three deletions in `src/budget.rs` (`rg 'plan\.reader_workers' src/` returns
nothing, so nothing outside this file reads them):

1. The `pub reader_workers: usize,` field and its doc comment from
   `pub struct ThreadPlan`.
2. The two `reader_workers: reader_workers(usable_cores, ...),` initializers
   in `plan_thread_budget`.
3. The whole `fn reader_workers(usable_cores: usize, concurrent: usize) -> usize`
   function and its doc comment.

Then delete the two tests that only asserted that dead field:
`test_sharded_vcf_reclaims_unused_htslib_budget_for_reader_workers` and
`test_sharded_vcf_reader_workers_are_bounded_across_concurrent_contigs`.
Their intent — readers reclaim the unused HTSlib budget, and readers stay
bounded across concurrent contigs — is now covered by
`plan_prefers_depth_over_breadth_on_the_reported_machine` and
`plan_never_exceeds_the_contig_count`.

- [ ] **Step 7: Update the remaining `PlanInputs` construction sites in tests**

Every `PlanInputs { .. }` literal in `src/budget.rs`'s test module now needs
`reader_workers: Some(n)` or `reader_workers: None`. Rules:

- A test that was pinning a specific `w` to exercise the memory law keeps its
  value as `Some(n)`.
- Rewrite `a_high_worker_count_can_exceed_a_budget_the_default_fits` — its
  comment block derives the old `kappa*(w + (w-1))` arithmetic, which no
  longer holds. Replace the whole test with this one, which asserts the same
  property against the new law:

```rust
    // Per-contig memory is now
    //   per_contig_mb + kappa*w*chunk_MB + in_flight_MB(w)
    // where in_flight_MB = pending ceiling (8 chunks, enforced by
    // `Frontier`) + tx_res channel capacity (2*w chunks, enforced by the
    // bounded channel). The kappa term is linear in `w` (it was
    // kappa*(w + (w-1)) before #169) and carries the readers' own `w` working
    // chunks, so those are not double-counted below.
    // n_samples=1_000, chunk_bytes=10_000_000 (10 MB), against the
    // 2026-08-11 RamLaw::VCF envelope:
    //   baseline   = 457.259 + 0.011017*1_000  =  468.276 MB
    //   in_flight  = 8*10 + 2*w*10             =   80 + 20w MB
    //   per-contig = 111.426 + 61.05786*w + 80 + 20w = 191.426 + 81.05786*w
    //     w=3  ->  191.426 +  243.174 =  434.600 MB -> needs  902.876 MB
    //     w=16 ->  191.426 + 1296.926 = 1488.352 MB -> needs 1956.628 MB
    //   budget = 1_200 MB at cc=1: fits w=3, rejects w=16.
    #[test]
    fn a_high_worker_count_can_exceed_a_budget_a_lower_one_fits() {
        let inp = PlanInputs {
            usable_cores: 47,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_200_000_000),
            reader_workers: Some(16),
            ram: RamLaw::VCF,
        };
        assert!(matches!(
            plan_sharded(inp),
            Err(PlanError::InsufficientMemory { .. })
        ));
        assert_eq!(
            plan_sharded(PlanInputs {
                reader_workers: Some(3),
                ..inp
            })
            .unwrap(),
            ShardedPlan {
                concurrent_chroms: 1,
                reader_workers: 3,
            }
        );
    }
```

For every other test you touch, recompute the expected value from the
formulas above and write the arithmetic into the test's comment the way the
existing tests do. Do not adjust an expected number until you can show the
derivation.

- [ ] **Step 8: Run the tests to verify they pass**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
cargo test --no-default-features --features conversion --lib budget
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/budget.rs
git commit -m "feat(budget): plan concurrent_chroms and reader_workers together"
```

---

### Task 4: Wire the planner, unit sizing and backlog budget through Rust

**Files:**
- Modify: `src/lib.rs:147-160` (delete `DEFAULT_READER_WORKERS` and its doc
  comment), `src/lib.rs:167` (pyo3 signature), `src/lib.rs:284-329` (planning
  + `pipeline config` log), the `SourceSpec::Vcf` literal in the contig
  dispatch closure below it, and `src/lib.rs:573` (`reader_workers: 1` in the
  PGEN `plan_sharded` call)
- Modify: `src/orchestrator.rs:88-96` (`SourceSpec::Vcf` fields),
  `src/orchestrator.rs:518-531` (delete the env reads, use the passed values),
  `src/orchestrator.rs:574` (pass the backlog budget)

**Interfaces:**
- Consumes: `shard_exec::run(.., pending_budget_bytes: u64)` (Task 1);
  `shard::plan_unit_count(contig_records, workers, chunk_size, overshard_factor)`
  and `shard::{UNITS_TARGET_CHUNKS, MAX_UNITS_PER_CONTIG}` (Task 2);
  `budget::{plan_sharded, PlanInputs, pending_budget_bytes, reader_pool_cores, W_TARGET, MERGE_RESERVE_DIV}`
  (Task 3).
- Produces:
  - `run_conversion_pipeline(..., max_mem_bytes=None, reader_workers=None, log_level="info", receiver=None)`
    — one new optional pyo3 parameter, `reader_workers: Option<usize>`, placed
    **after** `max_mem_bytes` so every existing positional call still binds
    correctly.
  - `SourceSpec::Vcf` gains `overshard: usize`, `contig_records: Option<u64>`,
    and `pending_budget_bytes: u64`.

- [ ] **Step 1: Extend `SourceSpec::Vcf`**

In `src/orchestrator.rs`, in the `SourceSpec::Vcf` variant, after
`reader_workers: usize,` add:

```rust
        /// Fallback over-decomposition factor, used ONLY when
        /// `contig_records` is `None`. Resolved in `lib.rs` (including the
        /// `GENORAY_OVERSHARD` bench override) so the `pipeline config` log
        /// prints the effective value.
        overshard: usize,
        /// EXACT record count for this contig, or `None` when only the
        /// header-length fallback tier was available. See
        /// `shard::plan_unit_count` -- the fallback tier's values are base
        /// pairs, not records, and must never be passed as records.
        contig_records: Option<u64>,
        /// Byte ceiling on `shard_exec`'s reorder backlog for this contig.
        pending_budget_bytes: u64,
```

- [ ] **Step 2: Use the passed values instead of reading the environment**

In `src/orchestrator.rs`, in the `SourceSpec::Vcf` destructuring arm, add the
three new names to the pattern:

```rust
                    SourceSpec::Vcf {
                        vcf_path,
                        htslib_threads,
                        reader_workers,
                        overshard,
                        contig_records,
                        pending_budget_bytes,
                        regions,
                        overlap,
                    } => {
```

Then delete these three statements (they moved to `lib.rs`):

```rust
                        let reader_workers = bench_env("GENORAY_READER_WORKERS")
                            .unwrap_or(reader_workers)
                            .max(1);
                        let shard_htslib = bench_env("GENORAY_SHARD_HTSLIB")
                            .unwrap_or(crate::budget::SHARDED_VCF_HTSLIB_THREADS_PER_READER);
                        let overshard = bench_env("GENORAY_OVERSHARD")
                            .unwrap_or(OVERSHARD_FACTOR)
                            .max(1);
```

replacing them with:

```rust
                        // `reader_workers` and `overshard` arrive already
                        // resolved (planner value, then any GENORAY_* bench
                        // override) so the `pipeline config` line in lib.rs
                        // prints what actually ran -- issue #169 proposal 4.
                        let shard_htslib =
                            crate::budget::SHARDED_VCF_HTSLIB_THREADS_PER_READER;
```

and replace the `plan_vcf_shards` call's `max_shards` argument:

```rust
                                reader_workers.saturating_mul(overshard),
```

with:

```rust
                                crate::shard::plan_unit_count(
                                    contig_records,
                                    reader_workers,
                                    chunk_size,
                                    overshard,
                                ),
```

- [ ] **Step 3: Log the effective per-contig shard plan and pass the budget**

Still in the `SourceSpec::Vcf` arm, replace the existing `trace_ll!` plan line:

```rust
                            trace_ll!(
                                "[plan {chr}] workers={} shards={}",
                                reader_workers,
                                units.len()
                            );
```

with:

```rust
                            trace_ll!(
                                "[plan {chr}] workers={} shards={}",
                                reader_workers,
                                units.len()
                            );
                            // The frontier width and the backlog ceiling are
                            // the two terms that decided both wall time and
                            // peak RSS in issue #169; neither was observable.
                            tracing::debug!(
                                chrom = %chr,
                                reader_workers,
                                units = units.len(),
                                pending_budget_mb =
                                    pending_budget_bytes as f64 / 1e6,
                                "shard plan"
                            );
```

and add the budget as the new trailing argument of the VCF branch's
`shard_exec::run` call, replacing the `u64::MAX` that Task 1 left there:

```rust
                                pending_budget_bytes,
```

Leave the PGEN branch's `shard_exec::run` at `u64::MAX` — PGEN sub-contig
sharding is pinned to one unit at the Python layer, so its head is always the
only unit and the gate would be inert anyway.

- [ ] **Step 4: Delete `DEFAULT_READER_WORKERS` and re-plan in `lib.rs`**

Delete the whole `DEFAULT_READER_WORKERS` constant and its doc comment at
`src/lib.rs:147-160`. Its comment justifies the value 3 with "the executor is
the bottleneck, not the reader" — true only in the pre-parallel-executor
regime; the measured knee has since moved to w ≈ 16–32. Do not preserve the
claim anywhere.

Add `reader_workers` to the pyo3 signature at `src/lib.rs:167`, immediately
after `max_mem_bytes=None`:

```rust
#[pyo3(signature = (vcf_path, reference_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608, skip_out_of_scope=false, signatures=false, info_fields=Vec::new(), format_fields=Vec::new(), check_ref="e".to_string(), region_ranges=Vec::new(), regions_overlap="pos".to_string(), max_mem_bytes=None, reader_workers=None, log_level = "info".to_string(), receiver = None))]
```

and the matching parameter in the `fn run_conversion_pipeline` argument list,
after `max_mem_bytes: Option<u64>,`:

```rust
    reader_workers: Option<usize>,
```

- [ ] **Step 5: Replace the planning block**

In `src/lib.rs`, replace the `plan_sharded` call and everything down to the
`"pipeline config"` log with:

```rust
            // BENCH-ONLY overrides are resolved HERE, not in
            // `process_chromosome`, so the `pipeline config` line below prints
            // what actually ran. Before #169 the override was applied
            // per-contig and the log printed the planner's value, making a
            // 20-worker run indistinguishable from a 3-worker one in the logs.
            let requested_workers =
                orchestrator::bench_env_reader_workers().or(reader_workers);

            let sharded = crate::budget::plan_sharded(crate::budget::PlanInputs {
                usable_cores: available_cores.saturating_sub(1).max(1),
                n_contigs: chroms.len(),
                n_samples: samples.len(),
                chunk_bytes,
                max_mem_bytes,
                reader_workers: requested_workers,
                ram: crate::budget::RamLaw::VCF,
            });
            let sharded = match sharded {
                Ok(p) => p,
                Err(e) => return vec![Err(crate::error::ConversionError::from(e))],
            };

            let plan = crate::budget::plan_thread_budget(available_cores, chroms.len());
            let concurrent_chroms =
                orchestrator::bench_concurrent_chroms(sharded.concurrent_chroms);
            let htslib_threads = plan.htslib_threads; // monolithic path only
            let reader_workers = sharded.reader_workers;
            let overshard = orchestrator::bench_overshard();
            let pending_budget_bytes = crate::budget::pending_budget_bytes(chunk_bytes);
            let processing_threads = crate::budget::processing_threads_for(
                available_cores.saturating_sub(1).max(1),
                concurrent_chroms,
                reader_workers,
            );

            let monolithic_reader_active =
                concurrent_chroms * (crate::budget::PIPELINE_THREADS_PER_CHROM + htslib_threads);
            let sharded_vcf_active = concurrent_chroms
                * (crate::budget::PIPELINE_THREADS_PER_CHROM
                    + reader_workers * (1 + crate::budget::SHARDED_VCF_HTSLIB_THREADS_PER_READER));
            tracing::info!(cores = available_cores, "using cores");
            tracing::info!(
                concurrent_chroms,
                htslib_threads,
                monolithic_reader_active,
                reader_workers,
                overshard,
                pending_budget_mb = pending_budget_bytes as f64 / 1e6,
                sharded_vcf_active,
                processing_threads,
                "pipeline config"
            );
```

Note `chunk_bytes` is the existing `resident_chunk_size`-narrowed value
computed just above — do not substitute the nominal
`chunk_size * per_variant_bytes`.

- [ ] **Step 6: Pass the new fields at dispatch**

In the `par_iter().map(|chrom| ...)` closure in `src/lib.rs`, extend the
`SourceSpec::Vcf` literal:

```rust
                            orchestrator::SourceSpec::Vcf {
                                vcf_path: vcf_path.clone(),
                                htslib_threads,
                                reader_workers,
                                overshard,
                                // `costs.values` is a record count ONLY on the
                                // exact tier; the fallback tier holds base-pair
                                // contig lengths, which would mis-size the
                                // frontier by orders of magnitude.
                                contig_records: if costs.exact_counts {
                                    costs.values.get(chrom.as_str()).copied()
                                } else {
                                    None
                                },
                                pending_budget_bytes,
                                regions: ranges_by_chrom.get(chrom).cloned().unwrap_or_default(),
                                overlap: overlap_mode,
                            },
```

- [ ] **Step 7: Add the two bench-override accessors**

In `src/orchestrator.rs`, next to `bench_concurrent_chroms`, add:

```rust
/// BENCH-ONLY: `GENORAY_READER_WORKERS`. Resolved by `lib.rs` alongside the
/// public `reader_workers=` argument (the env var wins) so the effective
/// value reaches both the planner and the `pipeline config` log. Before #169
/// this was read per-contig inside `process_chromosome`, where it could not
/// reach the log line at all.
pub(crate) fn bench_env_reader_workers() -> Option<usize> {
    bench_env("GENORAY_READER_WORKERS").map(|w| w.max(1))
}

/// BENCH-ONLY: `GENORAY_OVERSHARD`, falling back to [`OVERSHARD_FACTOR`].
/// Only consulted when a contig has no exact record count -- see
/// `shard::plan_unit_count`.
pub(crate) fn bench_overshard() -> usize {
    bench_env("GENORAY_OVERSHARD")
        .unwrap_or(OVERSHARD_FACTOR)
        .max(1)
}
```

- [ ] **Step 8: Fix the PGEN `plan_sharded` call site**

At `src/lib.rs:573`, `reader_workers: 1,` becomes:

```rust
                reader_workers: Some(1),
```

- [ ] **Step 9: Build and run the whole Rust suite**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
cargo check --no-default-features
cargo test --no-default-features --features conversion
```

The bare `cargo check --no-default-features` is not redundant: rust CI only
ever built with `conversion` ON, and the no-conversion query-core build that
GenVarLoader links against has broken before on exactly this kind of module
gating.

Expected: both PASS, ~341 tests.

- [ ] **Step 10: Commit**

```bash
git add src/lib.rs src/orchestrator.rs
git commit -m "feat(convert): make reader_workers a planned, logged, overridable knob"
```

---

### Task 5: Python API and CLI

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_vcf` signature ~line 645, docstring,
  and the `_core.run_conversion_pipeline` call ~line 884)
- Modify: `python/genoray/_cli/__main__.py` (`write_vcf` signature ~line 86,
  the multi-file guard ~line 205, the `from_vcf` call ~line 233)
- Test: `tests/test_svar2_from_vcf.py`

**Interfaces:**
- Consumes: `_core.run_conversion_pipeline(..., reader_workers=...)` (Task 4).
- Produces:
  - `SparseVar2.from_vcf(..., threads=None, reader_workers=None, ...) -> int`
    — new keyword-only argument, placed immediately after `threads`. `None`
    selects the planner's derive path; an `int` selects honour-or-refuse; a
    value below 1 raises `ValueError` rather than being clamped to 1 by the
    Rust side, where the clamp would silently deliver the explicit path to a
    caller who meant the derive path.
  - `genoray write vcf --reader-workers N`, with the same rejection, and a
    multi-file guard tested the way `tests/cli/test_write_cli.py` already tests
    the analogous `--samples` guard.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar2_from_vcf.py`:

Both tests must be able to FAIL if `reader_workers` stops reaching the planner.
That rules out the two obvious shapes, and the reasons are worth stating because
both were written into an earlier draft of this plan and both shipped vacuous:

- **A byte-identity assertion cannot work here.** Identical output across reader
  counts must hold whether or not the knob is wired at all — that is the whole
  point of Tasks 1-4, and `tests/test_svar2_schedule_invariance.py` already owns
  it. Such a test passes with the argument deleted.
- **A `max_mem` boundary cannot work here either.** On this fixture `chunk_MB` is
  about 0.000128, so `w` barely moves the memory law; the only term separating the
  paths is `cc`, which depends on the host's core count. A budget tuned on a
  48-core machine misfires on a 4-core CI runner. It is also easy to pick a budget
  so small it cannot fit the fixed 457 MB baseline, in which case the call raises
  for a reason having nothing to do with `reader_workers`.

Assert on the effective-config log instead — Task 4 added it precisely so the
planner's actual choice is observable — and make the assertion RELATIVE, so it
carries no assumption about the host:

```python
def test_from_vcf_reader_workers_reaches_the_planner(tmp_path: Path, capfd):
    """An explicit `reader_workers` must arrive at `plan_sharded` unchanged.

    Two different requests are compared against each other rather than against a
    fixed number: if the argument stops being threaded through, both runs fall
    back to the same planner-derived value and the inequality fails on every
    machine. Anything asserting a constant would encode this host's core count.
    """
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    seen = []
    for w in (1, 5):
        SparseVar2.from_vcf(
            tmp_path / f"w{w}", vcf, no_reference=True,
            reader_workers=w, log_level="info",
        )
        # `capfd`, NOT `capsys`: this crosses the Rust/Python boundary at the
        # OS fd level. Read BOTH streams: with no `GENORAY_LOG` set the line
        # arrives only through the Rust-tracing -> Python event channel ->
        # `rich.console.Console()` path, whose default target is stdout; the
        # direct-stderr `tracing_subscriber` fmt layer activates only when
        # `GENORAY_LOG` is set. Combining the two makes the test independent
        # of whichever the environment routes it to.
        captured = capfd.readouterr()
        line = next(
            ln for ln in (captured.out + captured.err).splitlines()
            if "pipeline config" in ln
        )
        seen.append(int(re.search(r"reader_workers=(\d+)", line).group(1)))
    assert seen == [1, 5]


def test_from_vcf_rejects_a_reader_workers_below_one(tmp_path: Path):
    """`0` must not silently become 1.

    `plan_sharded` clamps with `req.max(1)`, so a caller passing 0 to mean "let
    the planner choose" would instead get the EXPLICIT path with one reader and
    `cc` sized off `w=1` -- the opposite of the intent, with no warning. `None`
    is the way to ask for the derive path.
    """
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    with pytest.raises(ValueError, match="reader_workers"):
        SparseVar2.from_vcf(tmp_path / "zero", vcf, no_reference=True, reader_workers=0)
```

Parse the `reader_workers` field out of the line rather than matching the whole
line, so adding or reordering log fields does not break the test.

Also RENAME the stale `test_from_vcf_reader_workers_that_cannot_fit_max_mem_raises`
this plan previously specified. It is vacuous for the same reason the bullets
above give: `max_mem="1M"` is below the fixed 457 MB cohort baseline, so the call
raises with no `reader_workers` argument at all. What it does genuinely cover is
that a `PlanError` crosses the pyo3 boundary as a Python exception whose message
names `max_mem`, which is worth keeping. Rename it to
`test_from_vcf_planner_refusal_surfaces_to_python`, drop the `reader_workers=64`
argument (it plays no part in the refusal), and say in the docstring that the
honour-or-refuse semantics themselves are owned by
`an_explicit_reader_workers_is_honoured_or_refused_never_shrunk` in
`src/budget.rs`, which tests them at a realistic 10 MB chunk where `w` actually
moves the memory law.

`_write_vcf(d, *, symbolic, indexed)` is this module's existing helper — it
writes a two-record bgzipped, bcftools-indexed `chr1` VCF and returns the
`.vcf.gz` path. Do NOT edit its body: the `symbolic=False` bytes are pinned by
`tests/test_svar2_reader_identity.py`'s expected digests. `_oracle.store_digest`
is the same digest helper `tests/test_svar2_schedule_invariance.py` uses; add
the `from tests import _oracle` import at the top of the module if it is not
already there.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run test tests/test_svar2_from_vcf.py -k reader_workers
```

Expected: FAIL with `TypeError: from_vcf() got an unexpected keyword argument
'reader_workers'`.

- [ ] **Step 3: Add the argument to `from_vcf`**

In `python/genoray/_svar2.py`, in the `from_vcf` signature, immediately after
`threads: int | None = None,`:

```python
        reader_workers: int | None = None,
```

and in the `_core.run_conversion_pipeline(...)` call, after the
`max_mem_bytes,` positional argument, add the keyword:

```python
                reader_workers=reader_workers,
```

- [ ] **Step 4: Document it in the docstring**

In `from_vcf`'s docstring, immediately after the paragraph that documents
`samples`, add:

```
        `reader_workers` sets how many independent indexed shard readers run
        per concurrent contig. `None` (the default) derives it from the core
        budget: the planner reserves a quarter of usable cores for the merge
        tail, picks contig concurrency preferring depth, then spends the rest
        on readers. Pass an explicit value to override; a value that cannot
        fit `max_mem` raises rather than being silently reduced. This is a
        scheduling knob only — output is byte-identical at every value.
```

- [ ] **Step 5: Add the CLI flag**

In `python/genoray/_cli/__main__.py`, in `write_vcf`'s signature, immediately
after the `threads:` parameter:

```python
    reader_workers: Annotated[
        int | None, Parameter(name="--reader-workers")
    ] = None,
```

In the docstring's argument list, after the `threads:` line:

```
        reader_workers: Independent shard readers per concurrent contig. Defaults to a core-derived value. Single-file input only.
```

In the `if not is_single_vcf:` branch, alongside the existing `--samples`
guard, add:

```python
        if reader_workers is not None:
            raise ValueError(
                "--reader-workers is not supported for multi-file (vcf-list) "
                "input; that path uses one reader per contig and does not "
                "shard within a contig."
            )
```

and in the `else:` branch's `SparseVar2.from_vcf(...)` call, after
`threads=threads,`:

```python
            reader_workers=reader_workers,
```

- [ ] **Step 6: Rebuild the extension and run the tests**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
maturin develop --release
pixi run test tests/test_svar2_from_vcf.py -k reader_workers
```

`pixi run test` does not rebuild the `.so`, so the `maturin develop` line is
required or the new pyo3 argument will not exist at runtime. Run it in the
foreground; it takes several minutes.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_svar2.py python/genoray/_cli/__main__.py tests/test_svar2_from_vcf.py
git commit -m "feat(svar2): expose reader_workers on from_vcf and the write CLI"
```

---

### Task 6: Update the installable API skill

**Files:**
- Modify: `skills/genoray-api/SKILL.md` (line 256 signature, the `threads`
  bullet at 305-312, the shared-CLI-options paragraph at ~1045)

**Interfaces:**
- Consumes: the `from_vcf` signature and CLI flag from Task 5. Nothing
  produces from this task.

The repo rule is explicit: any change to a name reachable from
`import genoray` without an underscore must update `SKILL.md` in the same
change. `reader_workers` is such a name.

- [ ] **Step 1: Update the `from_vcf` signature line**

At `skills/genoray-api/SKILL.md:256`, insert `reader_workers=None,`
immediately after `threads=None,` in the signature string.

- [ ] **Step 2: Correct the `threads` bullet**

The bullet at lines 305-312 currently claims "Sub-contig sharding is driven
entirely by this existing `threads` value — no separate knob". That is now
wrong twice over: it was never true (the count came from a hard-coded
constant, not from `threads`), and there is now a separate knob. Replace the
bullet with:

```markdown
- `threads=None` — total thread budget (autodetected if `None`). Drives contig
  concurrency and, through the planner, the per-contig reader count.
- `reader_workers=None` — independent indexed shard readers per concurrent
  contig, the knob that sets sub-contig read parallelism. `None` derives it
  from the core budget: a quarter of usable cores is reserved for the merge
  tail, contig concurrency is chosen preferring depth (~8 readers per contig),
  and the rest goes to readers. An explicit value is honoured or refused —
  one that cannot fit `max_mem` raises `InsufficientMemory` rather than being
  silently reduced. Output is byte-identical at every value. (`from_vcf_list`,
  the N-single-sample-VCF merge path, does not shard within a contig and does
  not accept this argument.) See "Parallel conversion" in
  `docs/source/svar.md` for scaling numbers.
```

- [ ] **Step 3: Document the CLI flag**

In the shared-`write`-options paragraph at ~line 1045, after the
`--threads`/`-@` entry, add a sentence noting the flag is **not** shared:

```markdown
`write vcf` additionally accepts `--reader-workers N` (single-file input only;
passing it with a directory/manifest raises).
```

- [ ] **Step 4: Correct the stale `max_mem` floor passage**

`SKILL.md` around lines 380-405 publishes concrete `max_mem` floor figures
qualified as "at `from_vcf`'s own defaults (`chunk_size=25_000`,
`reader_workers=3`)". Both halves of that qualifier are now wrong. `from_vcf`
no longer pins `w=3` -- it defaults to `reader_workers=None`, and the derive
path in `plan_sharded` scans `w` DOWNWARD from `w_max` to 1 before it gives up
a contig, so a tight budget now yields a smaller `w` instead of a refusal.
Task 3 also changed the per-`w` charge from `kappa * (2w - 1)` to
`w * (kappa + 2) + 8` per chunk-MB.

The published numbers were computed under the old law at a fixed `w=3`. Replace
them with the floors that the shipped planner actually enforces at the new
default, i.e. at `cc=1, w=1`. Use these values verbatim -- they are computed
from the law's own coefficients and reproduce the three currently-published
figures to within 0.29%, 0.01% and 0.00% when evaluated under the OLD law,
which is what validates the model:

| cohort | published (old law, `w=3`) | actual floor now (`w=1`) |
|---|---|---|
| S=4,000 | 1,380 MB | **1,015 MB** |
| S=128,000 | 26,400 MB | **14,864 MB** |
| S=500,000 | 101,480 MB | **56,408 MB** |

Two consequences stated in that passage must be corrected too:

- The 64 GB host claim SURVIVES: `max_mem` defaults to 80% of detected RAM
  (52,429 MB), still below the 56,408 MB floor at S=500,000, so it does still
  raise `PlanError::InsufficientMemory`. Keep the claim; the margin is now
  narrow (52.4 vs 56.4 GB) rather than enormous.
- The 128 GB host claim is now WRONG. It no longer "clears by under 1% of
  headroom": at 104,858 MB it plans `cc=1, w=2` with roughly 48 GB to spare.
  Say that instead.

Rewrite the passage so it does NOT quote a fixed default `w`. State that the
floor is the `cc=1, w=1` point, that an explicit `reader_workers` raises the
floor because it is honoured or refused rather than degraded, and give the
`w=3` column only as the illustrative comparison it now is.

- [ ] **Step 5: Verify nothing else in SKILL.md still claims there is no knob**

```bash
rg -n "no separate knob|reader_workers|--reader-workers" skills/genoray-api/SKILL.md
```

Expected: no `no separate knob` hit remains. Do NOT expect exactly three
`reader_workers` hits -- `SKILL.md` already mentioned `reader_workers` four
times before this task (in the `max_mem` floor passage that Step 4 rewrites),
so the count after your edits will be higher. Check that every remaining
mention is TRUE, not that they number three.

- [ ] **Step 6: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(skill): document reader_workers on from_vcf and write vcf"
```

---

### Task 7: Byte-identity and frontier regression tests

**Files:**
- Modify: `tests/test_svar2_schedule_invariance.py`

**Interfaces:**
- Consumes: `SparseVar2.from_vcf(reader_workers=...)` (Task 5);
  `GENORAY_OVERSHARD` (Task 4).

**Background:** This file is the existing byte-identity gate — it converts a
purpose-built eight-contig VCF under several `(concurrent_chroms,
reader_workers)` schedules and asserts one digest. Its
`multi_contig_vcf` fixture already plants long-ALT records at contig
midpoints specifically so long-allele bank offsets have a chance to interleave
differently. This task widens the schedule matrix to cover the new frontier
and backlog code paths.

- [ ] **Step 1: Widen the schedule matrix**

Replace the `SCHEDULES` list and its comment near the top of
`tests/test_svar2_schedule_invariance.py`:

```python
# (concurrent_chroms, reader_workers) -- spans the corners the planner can
# reach: one contig at a time with many readers, and many contigs with few.
# The wide-w rows matter more since #169: `reader_workers` is now derived from
# cores rather than pinned at 3, so a schedule the planner never used to
# produce is now the default on a large host.
SCHEDULES = [(1, 1), (1, 12), (1, 32), (4, 3), (8, 2)]
```

- [ ] **Step 2: Add a frontier-granularity invariance test**

Append to the same file:

```python
def test_digest_is_invariant_across_frontier_granularities(
    multi_contig_vcf, tmp_path, monkeypatch
):
    """Unit granularity must not move a single output byte.

    Since #169 the work-unit count comes from the contig's RECORD count
    (`shard::plan_unit_count`), not from `workers * OVERSHARD_FACTOR`, and the
    reorder backlog is bounded so non-head readers park mid-stream. Both
    change WHEN a chunk reaches the collector; neither may change what is
    written. The `GENORAY_OVERSHARD` values below drive the no-exact-counts
    fallback path, which this small fixture takes.
    """
    digests = {}
    for overshard in (1, 4, 40):
        monkeypatch.setenv("GENORAY_OVERSHARD", str(overshard))
        out = tmp_path / f"ov{overshard}.svar"
        digests[overshard] = _convert(multi_contig_vcf, out, 1, 8, monkeypatch)
    assert len(set(digests.values())) == 1, (
        f"unit granularity changed output: {digests}"
    )


def test_explicit_reader_workers_matches_the_derived_default(
    multi_contig_vcf, tmp_path, monkeypatch
):
    """The public knob and the planner's own choice must agree byte-for-byte.

    `_convert` sets GENORAY_READER_WORKERS; this asserts the public
    `reader_workers=` argument lands on the same code path and produces the
    same store.
    """
    from tests import _oracle

    env_out = tmp_path / "via_env.svar"
    env_digest = _convert(multi_contig_vcf, env_out, 1, 6, monkeypatch)

    monkeypatch.delenv("GENORAY_READER_WORKERS", raising=False)
    monkeypatch.delenv("GENORAY_CONCURRENT_CHROMS", raising=False)
    arg_out = tmp_path / "via_arg.svar"
    SparseVar2.from_vcf(
        arg_out,
        multi_contig_vcf,
        no_reference=True,
        chunk_size=CHUNK_SIZE,
        reader_workers=6,
    )
    assert _oracle.store_digest(arg_out) == env_digest
```

- [ ] **Step 3: Pin the documented `max_mem` floors against the planner**

Task 6 rewrites `skills/genoray-api/SKILL.md`'s published `max_mem` floor
figures. Those numbers came from a hand evaluation of the RAM law and nothing
stops them drifting silently the next time a coefficient or the per-`w` charge
moves -- which is exactly how they became wrong in the first place. Add a test
to `src/budget.rs`'s `mod tests` that pins them to the planner itself:

```rust
#[test]
fn the_documented_max_mem_floors_match_the_planner() {
    // These three figures are published in skills/genoray-api/SKILL.md. If
    // this test fails, the law changed and that document is now lying to
    // users about how much memory they need -- update BOTH.
    //
    // The floor is the cc=1, w=1 point, because `plan_sharded`'s derive path
    // scans `w` down to 1 before giving up a contig. `chunk_bytes` is
    // `0.25 * n_samples * chunk_size`: two haplotypes at one bit each.
    for (n_samples, floor_mb) in [(4_000u64, 1_015u64), (128_000, 14_864), (500_000, 56_408)] {
        let chunk_bytes = n_samples * 25_000 / 4;
        let inp = |budget_mb: u64| PlanInputs {
            usable_cores: 31,
            n_contigs: 22,
            n_samples: n_samples as usize,
            chunk_bytes,
            max_mem_bytes: Some(budget_mb * 1_000_000),
            reader_workers: None,
            ram: RamLaw::VCF,
        };
        // One MB under the published floor must refuse...
        assert!(
            plan_sharded(inp(floor_mb - 1)).is_err(),
            "S={n_samples}: planner accepted a budget below the documented floor"
        );
        // ...and the floor itself must plan, at exactly one reader.
        let plan = plan_sharded(inp(floor_mb)).expect("documented floor must plan");
        assert_eq!(plan.reader_workers, 1, "S={n_samples}");
        assert_eq!(plan.concurrent_chroms, 1, "S={n_samples}");
    }
}
```

If a value here disagrees with the planner, the PLANNER is right: fix the
number in both this test and `SKILL.md`, and say so in the commit message.
Do not adjust the tolerance to make it pass.

- [ ] **Step 4: Cover `shard_exec::run` with a finite backlog budget**

**This is the most important test in the task.** `shard_exec::run` currently has
**zero test callers** -- `rg -n 'shard_exec::run' src/` finds only the two
production sites in `orchestrator.rs`. `Frontier` itself is well unit-tested
(`admit_never_parks_the_head_unit`,
`admit_parks_a_non_head_unit_until_the_head_advances`,
`admit_releases_a_parked_unit_on_cancel`,
`an_unbounded_budget_never_parks_anything`), but those drive `Frontier` directly
through `publish`/`admit`. Nothing exercises the INTEGRATION: real scoped worker
threads, the FIFO work queue, the reorder buffer and the admission gate running
together. When Task 4 instrumented `Frontier::admit` during an end-to-end
conversion it measured **zero park events** -- so the head-exemption, the
property the entire deadlock-freedom argument rests on, has never actually run.

No verbatim code is given for this step, deliberately. Constructing `RawRecord`
(`Calls`, `FormatVals`, `info_raw`) and `ChunkAssembler::new`'s eight arguments
correctly cannot be done from the summaries in this plan, and four defects on
this branch already came from plan-supplied code that did not survive contact
with the source. Write it TDD-style against the real signatures and report what
you land. The contract below is binding; the shape is yours.

The seam that makes this cheap: `RecordSource` (`src/record_source.rs:40`) is a
**one-method trait** --
`fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError>` --
and `ChunkAssembler::new` takes `Box<dyn RecordSource + Send>`. A stub source is
therefore a `Vec<RawRecord>` plus a cursor. Build records with no reference
(`fasta_path: None`) and no INFO/FORMAT fields so nothing needs a FASTA or a
real header.

Two tests, both in `src/shard_exec.rs`'s `mod tests`:

1. `run_emits_every_unit_in_order_under_a_bounded_backlog`
   - Seed enough units (ordinals `0..=7`) that a small `pending_budget_bytes` --
     about two chunks' worth -- is genuinely exceeded.
   - **Force the backlog to build deterministically.** Left alone, units may
     complete in order and never park, which is precisely how the end-to-end
     measurement missed this path. Make the stub source for **ordinal 0** sleep
     briefly per record (~5 ms) so later units finish first and pile up behind
     the head.
   - **Assert the gate was actually exercised:** the run's `pending_gauge`
     high-water mark must be **> 0**. A test that never observes a non-zero
     backlog has not tested admission control and must fail loudly rather than
     pass vacuously. This assertion is the point of the test.
   - Assert chunks arrive on `tx_dense` in ascending ordinal order and that the
     returned `ShardTotals` accounts for every record.

2. `bounded_and_unbounded_backlogs_produce_identical_output`
   - Run the same units twice: once at `pending_budget_bytes = u64::MAX` (gate
     disabled -- the PGEN path's setting) and once at test 1's tiny budget.
     Assert the two runs emit identical `DenseChunk` sequences and equal
     `ShardTotals`. This is the invariant the feature claims: admission control
     changes WHEN a chunk is produced, never WHAT.

**Deadlock guard, required on both tests.** A bug in the head-exemption
deadlocks rather than failing, and a deadlocked `cargo test` hangs until the
harness kills it -- which reads as infrastructure trouble, not a test failure.
Run each `run(...)` call on a spawned thread and join with a timeout (10 s is
generous for microsecond workloads plus the sleeps); on timeout, fail with a
message naming the deadlock. Do not use a bare `handle.join()`.

- [ ] **Step 5: Run the gate**

```bash
export CARGO_TARGET_DIR=/tmp/genoray-target-dlaub
maturin develop --release
pixi run test tests/test_svar2_schedule_invariance.py -v
```

Expected: PASS, every test. A failure here means the reorder or bank ordering
broke and nothing else in this plan matters.

- [ ] **Step 6: Run the full suite**

```bash
pixi run test
```

Expected: PASS. Compare the pass count against `git stash`-free `main` if
anything looks off; do not accept a lower collected count as "fine".

- [ ] **Step 7: Commit**

```bash
git add tests/test_svar2_schedule_invariance.py
git commit -m "test(svar2): gate byte-identity across frontier width and backlog budget"
```

---

### Task 8: Performance repro on a wide cohort

**Files:**
- Create: `scripts/bench_svar2/frontier_points.py`
- Create: `scripts/bench_svar2/frontier.sbatch`
- Create: `docs/superpowers/plans/results/2026-09-05-reader-frontier.md`

**Interfaces:**
- Consumes: everything above. `scripts/bench_svar2/probe.py` already sets
  `GENORAY_READER_WORKERS`, `GENORAY_CONCURRENT_CHROMS`, `GENORAY_OVERSHARD`
  and `GENORAY_SHARD_HTSLIB` from a `SweepPoint`, and already parses
  `pending_highwater` out of the `genoray::monitor` trace stream;
  `scripts/bench_svar2/sweep.py` (`--plan/--results/--outdir`) owns manifest
  loading, `code_id` stamping and resume. Reuse both; do not write a second
  harness and do not call `probe.run_point` directly.

**Background and traps.** Read these before running anything:

- `SweepPoint.point_id` hashes the *configuration* only, not the binary. The
  A/B arms here differ only in the built `.so`, so resume-by-`point_id` alone
  would serve `main`'s rows back as the branch's own — the failure that got a
  RamLaw refit shipped before review caught it (#159). `sweep.py` already
  guards this: `pending_points` keys on `(point_id, code_id)`, where
  `build_code_id()` is a sha256 of the loaded `_core` extension. **That is why
  this task drives `sweep.py` rather than calling `probe.run_point` directly.**
  Still confirm each arm's rows carry a different `code_id`; if they match,
  `maturin develop --release` did not rebuild and both arms measured one
  binary.
- The login node runs at 30–75 loadavg; the same three points measured
  68/54/119 s there versus ~5 s on a dedicated allocation. Everything below
  goes through `sbatch`.
- Inside an `sbatch` script, `unset CLAUDE_JOB_DIR` on the first line and
  write scratch to `/local/$USER`. `$CLAUDE_JOB_DIR/tmp` is a symlink to
  node-local scratch on the *submitting* node and dangles anywhere else;
  `sbatch` exports the environment by default, so a helper that reads the
  variable will silently override a correct path you chose. This killed a 7 h
  sweep once.
- Node speed varies 2.08× across the cluster. Pin `--nodelist` and stamp the
  node into the results file. Cross-scale timing claims from unpinned runs are
  not admissible.

- [ ] **Step 1: Build the wide corpus (one sbatch job)**

Write `scripts/bench_svar2/frontier.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=genoray-frontier
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# MUST be first: $CLAUDE_JOB_DIR points at node-local scratch on the
# SUBMITTING node and dangles here, and helpers read the variable themselves.
unset CLAUDE_JOB_DIR

SCRATCH=/local/$USER/genoray-frontier
mkdir -p "$SCRATCH"
# Per-ARM target dir. A SHARED CARGO_TARGET_DIR can hand a second source tree
# the first tree's artifact byte-for-byte, which silently turns an A/B into
# an A/A.
export CARGO_TARGET_DIR=/local/$USER/genoray-target-${ARM:?set ARM=main or ARM=branch}
echo "node=$(hostname) arm=$ARM scratch=$SCRATCH"

CORPUS="$SCRATCH/wide.vcf.gz"
if [ ! -f "$CORPUS" ]; then
  pixi run python -m scripts.bench_svar2.scale_corpus \
    --out "$CORPUS" \
    --samples 500000 \
    --variants 200000 \
    --contigs chr21 \
    --procs 32 \
    --bgzip-threads 8
fi

pixi run python -m scripts.bench_svar2.frontier_points \
  --manifest "$SCRATCH/wide.manifest.json" \
  --out "$SCRATCH/plan.json"

# ${ARM:?} forces the caller to name the arm: sweep.py resumes on
# (point_id, code_id), so two arms sharing a results file is CORRECT and
# informative, but two arms sharing an --outdir would race on bench.svar.
pixi run python -m scripts.bench_svar2.sweep \
  --plan "$SCRATCH/plan.json" \
  --results "$SCRATCH/frontier.ndjson" \
  --outdir "$SCRATCH/out-${ARM:?set ARM=main or ARM=branch}"
```

Note `set -u` and the `${ARM:?...}` guard: a bash `local A=$1 B="${A}x"` style
expansion under `set -u` aborts with an unbound-variable error whose message
gets eaten by the output buffer, showing only a 0-second FAILED job. Export
`ARM` explicitly on the `sbatch` line.

Submit with an explicit node:

```bash
sbatch --nodelist=<node> --export=ALL,ARM=main scripts/bench_svar2/frontier.sbatch
```

Pick `<node>` from `sinfo` and record it. The corpus is ~500k samples ×
200k variants — roughly 1/50 of the reported chr21, which reproduces the
pathology's shape because per-chunk bytes scale with samples, not variants.

- [ ] **Step 2: Write the points module**

Create `scripts/bench_svar2/frontier_points.py`:

```python
"""Reader-frontier plan for issue #169.

Emits a `sweep.py --plan` JSON file. `sweep.py` owns corpus-manifest loading,
`code_id` stamping and resume; `probe.py` owns the instrumented child run and
already parses `pending_highwater` out of the `genoray::monitor` trace stream.
Neither is reimplemented here -- this module is only the point list.

Run as a module so a ProcessPool worker can re-import it by NAME. Python
3.14's Linux default start method is forkserver, and a worker whose function
lives in a `spec_from_file_location`-loaded module dies with
ModuleNotFoundError -> BrokenProcessPool.

    python -m scripts.bench_svar2.frontier_points \\
        --manifest wide.manifest.json --out plan.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from scripts.bench_svar2.records import SweepPoint

# One contig, so contig concurrency is pinned at 1 and `reader_workers` is the
# only thing moving. The three worker counts bracket the old default (3), the
# new depth target (W_TARGET=8), and the count the reporter actually used (20).
READER_WORKERS = (3, 8, 20)
CHUNK_SIZE = 5_000


def points(manifest: str) -> list[SweepPoint]:
    """`SweepPoint.corpus` is the path to a corpus MANIFEST json (written by
    `scale_corpus.py` next to the .vcf.gz), not the .vcf.gz itself --
    `sweep.py` reads shape from the manifest so a corpus can be relocated."""
    return [
        SweepPoint(
            corpus=manifest,
            reader_workers=w,
            concurrent_chroms=1,
            shard_htslib=0,
            overshard=4,
            chunk_size=CHUNK_SIZE,
            threads=32,
            reps=1,
        )
        for w in READER_WORKERS
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(
        json.dumps([dataclasses.asdict(pt) for pt in points(a.manifest)], indent=2)
    )


if __name__ == "__main__":
    main()
```

`scale_corpus.py` writes the manifest to the corpus path with its extensions
replaced by `.manifest.json` (`out.with_suffix("").with_suffix(".manifest.json")`),
so `wide.vcf.gz` yields `wide.manifest.json`.

- [ ] **Step 3: Run both arms**

Run the sbatch job twice against the **same pinned node**, once from a
worktree at `main` (`ARM=main`) and once from this branch (`ARM=branch`).
`maturin develop --release` must run in each arm's own environment before its
sweep, or both arms load one `.so` and `code_id` will be identical.

On `main`, `reader_workers` is not a `from_vcf` argument, but `probe.py`
drives it through `GENORAY_READER_WORKERS`, which exists on both — so the same
plan file works unchanged for the baseline.

A shared `CARGO_TARGET_DIR` across the two trees can hand the second tree the
first tree's artifact byte-for-byte. Demand a `Compiling genoray` line naming
the arm's own path, and confirm the two arms' `code_id` values differ, before
believing any delta.

- [ ] **Step 4: Record the results**

Write `docs/superpowers/plans/results/2026-09-05-reader-frontier.md` with, for
each of the six runs (2 arms × 3 worker counts):

- node name, `git rev-parse HEAD`, and the row's own `code_id` (the `.so`
  sha256 `sweep.py` stamped) — the two arms' values MUST differ
- wall seconds, peak RSS MB, `pending_highwater`
- the per-contig span from the `done: (Xs)` log line, which includes the merge
- the store digest, which must be identical across all six

State plainly which of `W_TARGET`, `MERGE_RESERVE_DIV`, `UNITS_TARGET_CHUNKS`
and `PENDING_BUDGET_CHUNKS` the data actually constrains, and which remain at
their starting values with no measurement behind them. Do not update a
constant's doc comment to claim a measurement the run did not make.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_svar2/frontier_points.py scripts/bench_svar2/frontier.sbatch docs/superpowers/plans/results/2026-09-05-reader-frontier.md
git commit -m "perf(bench): reader-frontier arms for the wide-cohort repro"
```

- [ ] **Step 6: Confirm nothing is still running**

```bash
squeue -u $USER
```

An sbatch job left running past the session is fine (it has its own cgroup),
but you must report it as still running rather than reporting the task
complete.

---

## Self-Review

**Spec coverage.**

| Spec section | Task |
|---|---|
| A — public `reader_workers`, derived default, ordered `(cc, w)` plan | 3 (planner), 4 (Rust wiring), 5 (Python/CLI) |
| A — effective-config logging (issue proposal 4) | 4 steps 2, 5, 7 |
| B — record-sized units, `exact_counts` fallback | 2 (planner), 4 steps 2, 6 |
| C — head-exempt backpressure, deadlock argument, budget | 1 |
| C — planner term change `kappa*(w+(w-1))` → `kappa*w + pending` | 3 step 5 |
| D — byte-identity gate | 7 |
| D — unit tests in `shard_exec`, `budget`, `shard` | 1 step 1, 2 step 1, 3 step 1 |
| D — wide-cohort performance repro | 8 |
| Public API impact — `SKILL.md` | 6 |
| Deferred — executor parallelism | filed as #170, no task |

**Type consistency.** `plan_unit_count(Option<u64>, usize, usize, usize) ->
usize` is defined in Task 2 and called in Task 4 step 2 with
`(contig_records, reader_workers, chunk_size, overshard)`. `Frontier::{new,
admit, publish, wake_all}` and `ReorderBuffer::head` are defined in Task 1
steps 3-4 and used in step 7. `pending_budget_bytes(u64) -> u64` is defined in
Task 3 and called in Task 4 step 5. `PlanInputs.reader_workers` becomes
`Option<usize>` in Task 3 step 4, and both construction sites are fixed in
Task 4 (steps 5 and 8). `shard_exec::run`'s new trailing `u64` is added in
Task 1 step 7, stubbed at both call sites in step 8, and the VCF one is given
the real budget in Task 4 step 3.

**Ordering note for the reviewer.** Task 1 step 8 and Task 4 step 2 both edit
`src/orchestrator.rs`. If Tasks 1-3 run in parallel, Task 1 owns
`orchestrator.rs` and Tasks 2 and 3 must not touch it.

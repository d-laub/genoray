//! Backend-agnostic work-stealing shard worker pool + reorder buffer.
//!
//! Replaces the one-thread-per-shard + strict-ordinal-drain design (formerly
//! `orchestrator::read_vcf_shards_to_dense`) with a FIXED pool of `workers`
//! threads pulling `WorkUnit`s from a shared MPMC queue -- a worker that
//! finishes its unit early just requests the next one (work-stealing by
//! construction), instead of every shard owning a dedicated thread that sits
//! idle once its region is exhausted.
//!
//! The catch: `DenseChunk`s must still reach `tx_dense` in the SAME global
//! `(ordinal, local)` order the old serial-ordinal drain produced, because
//! everything downstream (chunk_id-indexed ledgers, the Phase-3 merge) keys
//! off a monotonic `chunk_id` that encodes that order. Shards may now
//! COMPLETE out of order -- [`ReorderBuffer`] is the pure ordering oracle
//! that buffers early arrivals and releases them (assigning the next global
//! id) only once every shard before them has fully drained.

use crossbeam_channel::{Sender, bounded, unbounded};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::chunk_assembler::ChunkAssembler;
use crate::error::ConversionError;
use crate::shard::WorkUnit;
use crate::trace::trace_ll;
use crate::types::DenseChunk;

#[cfg(target_os = "linux")]
use crate::monitor::current_tid;

/// Pure ordering oracle: decides WHEN a `(ordinal, local)` tag may be handed
/// the next global id, given shards can finish (or even stream their own
/// chunks) out of ordinal order. Carries no payload -- the caller looks up
/// the actual `DenseChunk` by tag when `emit` fires.
///
/// Invariant reproduced: global ids 0, 1, 2, ... are assigned in strict
/// `(ordinal, local)` order, identical to draining shard 0's channel to
/// exhaustion, then shard 1's, etc. -- regardless of arrival order.
pub struct ReorderBuffer {
    /// Ordinal currently being emitted/awaited.
    head: usize,
    /// Next global id to assign.
    next_gid: usize,
    /// ordinal -> locals that arrived while `ordinal > head`, in arrival order.
    buffered: HashMap<usize, Vec<usize>>,
    /// Ordinals whose shard has sent its `Done` signal.
    done: HashSet<usize>,
}

impl ReorderBuffer {
    pub fn new(n: usize) -> Self {
        Self {
            head: 0,
            next_gid: 0,
            buffered: HashMap::with_capacity(n),
            done: HashSet::with_capacity(n),
        }
    }

    /// The ordinal currently being emitted/awaited. The collector publishes
    /// this to [`Frontier`] so non-head producers know whether they are
    /// subject to the backlog budget.
    pub fn head(&self) -> usize {
        self.head
    }

    /// Record one arrival: either a chunk (`done = false`, tagged
    /// `(ordinal, local)`) or a shard-completion signal (`done = true`;
    /// `local` is ignored). Calls `emit(global_id, (ordinal, local))` for
    /// every tag that becomes releasable, in the order they must be written.
    pub fn push(
        &mut self,
        ordinal: usize,
        local: usize,
        done: bool,
        emit: &mut impl FnMut(usize, (usize, usize)),
    ) {
        if done {
            self.done.insert(ordinal);
        } else if ordinal == self.head {
            // The head shard streams its own chunks immediately -- no need to
            // wait for its `Done`.
            emit(self.next_gid, (ordinal, local));
            self.next_gid += 1;
            return;
        } else {
            self.buffered.entry(ordinal).or_default().push(local);
            return;
        }
        // A `done` arrived: the head shard may now be fully complete (all its
        // chunks already emitted via the `ordinal == head` branch above).
        // Advance past it and flush every already-buffered, now-current
        // ordinal in arrival order, repeating for chains of already-done
        // successors.
        while self.done.contains(&self.head) {
            self.head += 1;
            if let Some(locals) = self.buffered.remove(&self.head) {
                for l in locals {
                    emit(self.next_gid, (self.head, l));
                    self.next_gid += 1;
                }
            }
        }
    }
}

/// The collector's reorder backlog: chunks that have arrived but are not yet
/// releasable (their ordinal is ahead of `ReorderBuffer::head`), plus a
/// running byte total.
///
/// The ONLY insert site ([`PendingBacklog::insert_observing`]) observes the
/// gauge BEFORE adding the arriving chunk, so the high-water reflects the
/// backlog that was already waiting, excluding the chunk currently arriving.
/// A chunk that lands on `ReorderBuffer::head` and is released immediately
/// (an in-order stream never buffers anything) therefore contributes 0, not a
/// permanent `+1` -- see [`crate::monitor::PendingGauge`]'s doc comment.
///
/// Remove sites deliberately do NOT observe: `PendingGauge::observe` is a
/// `fetch_max`, and a remove can only lower `len()`/`bytes`, so a post-remove
/// observe would be a provable no-op.
struct PendingBacklog {
    map: HashMap<(usize, usize), DenseChunk>,
    bytes: u64,
}

impl PendingBacklog {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            bytes: 0,
        }
    }

    /// Record the backlog already waiting (excluding `chunk`), then insert
    /// `chunk` under `tag`.
    fn insert_observing(
        &mut self,
        tag: (usize, usize),
        chunk: DenseChunk,
        gauge: &crate::monitor::PendingGauge,
    ) {
        gauge.observe(self.map.len(), self.bytes);
        self.bytes += chunk.approx_bytes();
        self.map.insert(tag, chunk);
    }

    /// Remove a released chunk, if present (a `Done` tag has none to remove).
    fn remove(&mut self, tag: &(usize, usize)) -> Option<DenseChunk> {
        let chunk = self.map.remove(tag)?;
        self.bytes = self.bytes.saturating_sub(chunk.approx_bytes());
        Some(chunk)
    }
}

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
    /// Diagnostic counter (`PendingGauge::parks`) incremented each time a
    /// non-head worker actually blocks in [`Frontier::admit`]. A nonzero
    /// backlog high-water mark proves records piled up in the reorder
    /// buffer, but not that admission control did anything about it -- this
    /// is the only signal that the parking branch itself ran.
    ///
    /// Owned (`Arc`, not a borrow): `admit`'s deadlock-guard tests spawn a
    /// detached, non-scoped `thread::spawn` that may outlive the test
    /// function on a regression (that's the whole point -- see
    /// `admit_completes_within`'s doc comment), which requires `Frontier`
    /// itself to be `'static`.
    gauge: Arc<crate::monitor::PendingGauge>,
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
    pub(crate) fn new(budget_bytes: u64, gauge: Arc<crate::monitor::PendingGauge>) -> Self {
        Self {
            state: Mutex::new(FrontierState {
                head: 0,
                pending_bytes: 0,
            }),
            cv: std::sync::Condvar::new(),
            budget_bytes,
            gauge,
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
            // This IS the head-exemption path: a non-head worker is about to
            // block. Record it before waiting so a spurious wakeup that
            // re-parks counts again -- each `wait` call is a real park.
            self.gauge.record_park();
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
    /// Takes and immediately drops the state lock before notifying -- do not
    /// "simplify" this away. `admit`'s predicate reads `cancel`, a plain
    /// `AtomicBool` the state lock does not otherwise guard, so without this
    /// a caller could set `cancel` and call `notify_all` in the window
    /// between a waiter re-checking the predicate (false) and actually
    /// registering on the condvar via `wait` -- a lost wakeup that parks the
    /// waiter forever, since nothing else is guaranteed to notify it again.
    /// Acquiring the lock here forces a happens-before edge: either the
    /// waiter already holds the lock (this call blocks until its `wait`
    /// atomically releases it and parks, so the notify is guaranteed to
    /// reach it), or it has not yet re-locked to re-check the predicate (so
    /// it observes the now-`true` `cancel` on its next check).
    pub(crate) fn wake_all(&self) {
        drop(self.state.lock().unwrap());
        self.cv.notify_all();
    }
}

/// RAII guard held for the whole lifetime of one worker's closure, so its
/// `Drop` runs on every exit path -- a normal return, an early `return` on an
/// ordinary error, AND an unwinding panic. The panic case is the one nothing
/// else in this file reaches: every explicit `cancel.store` / `wake_all` pair
/// above sits on a path this worker chooses to take, but a panic unwinds
/// straight past all of them. Without this guard, a worker that panics while
/// it owns the reorder head leaves every other worker parked in
/// `Frontier::admit` forever -- the head-exemption argument on `Frontier`
/// says the head is the ONLY thing that can ever unpark them, and a dead
/// thread can't advance it. The result channel then never closes (the
/// parked workers still hold their `tx_res` clones), so the collector's
/// `rx_res.recv()` loop -- and the `wake_all()` after it at the bottom of
/// `run` -- never runs, and `join()` (which is what would turn this panic
/// into `ConversionError::WorkerPanicked`) is never reached either.
///
/// Order matters: `cancel` is set BEFORE waking, mirroring the same hazard
/// `Frontier::wake_all`'s doc comment describes -- a woken waiter that
/// re-checks its predicate before `cancel` is visible just re-parks, and
/// nothing else is guaranteed to wake it again.
struct PanicGuard {
    cancel: Arc<AtomicBool>,
    frontier: Arc<Frontier>,
}

impl Drop for PanicGuard {
    fn drop(&mut self) {
        if thread::panicking() {
            self.cancel.store(true, Ordering::Relaxed);
        }
        // Runs on the non-panic paths too: a spuriously woken parked worker
        // just re-evaluates its `while` condition in `Frontier::admit` and
        // re-parks, which `record_park` already counts as a real park by
        // design -- so this is a harmless no-op there and load-bearing only
        // on the panic path above.
        self.frontier.wake_all();
    }
}

/// One worker's report to the collector.
// `DenseChunk` legitimately outgrew clippy's large-enum-variant threshold when
// `global_idx: Vec<i32>` was added alongside `pos`/`ilens`/`alt_offsets`; every
// worker already streams one owned `DenseChunk` per message on this channel,
// so boxing it here would just move the allocation rather than avoid one.
#[allow(clippy::large_enum_variant)]
enum Msg {
    Chunk {
        unit_ordinal: usize,
        local: usize,
        chunk: DenseChunk,
    },
    Done {
        unit_ordinal: usize,
        dropped: u64,
        ref_excluded: u64,
        normalized: u64,
    },
    Err(ConversionError),
}

/// Per-contig counters returned by [`run`], summed across every shard. Both are
/// diagnostic totals the caller reports after the shards drain (the sharded
/// output itself is byte-identical regardless of how the units were split).
pub struct ShardTotals {
    /// Out-of-scope (symbolic/breakend) ALTs dropped across all shards.
    pub dropped_out_of_scope: u64,
    /// Records excluded by `CheckRef::Exclude` across all shards (each shard
    /// tallies only the records it owns, so a padded boundary record is counted
    /// once even when it appears in two shards' fetch windows).
    pub ref_excluded: u64,
    /// Atoms whose position moved during left-alignment across all shards.
    pub normalized_total: u64,
}

/// Distribute `units` across a fixed pool of `workers` threads pulling from a
/// shared work queue. Each worker builds a fresh `ChunkAssembler` via
/// `make_assembler` for its unit (VCF: a fresh indexed fetch over
/// `unit.fetch_start..unit.fetch_end`; PGEN: a variant-index range) and
/// streams its local `DenseChunk`s to a bounded results channel. This thread
/// (the collector) feeds every arrival through a [`ReorderBuffer`], which
/// reassigns `chunk.chunk_id` to a global monotonic counter in
/// `(ordinal, local)` order -- the same ordering the old serial-ordinal drain
/// produced -- before forwarding to `tx_dense`.
///
/// `err_context` decorates a worker's `ConversionError` with the failing
/// unit's shard-region context (see `orchestrator::with_vcf_shard_context`);
/// this module stays backend-agnostic about how a `WorkUnit` is described.
///
/// `chrom` labels the `GENORAY_TRACE` heartbeats emitted at the reader
/// assembly and `tx_dense`-forward seams (see `trace_ll!` call sites below);
/// it is not otherwise used when tracing is off.
///
/// `worker_tids` is a per-chrom registry each spawned `shard-worker-*`
/// thread pushes its own OS TID into on startup, for `monitor.rs` to sample.
/// A shared registry -- rather than resolving TIDs by matching the
/// `shard-worker-{i}` thread `comm` name -- is required because worker
/// names are NOT namespaced by chrom (just the pool-local index `i`): when
/// multiple chromosomes run concurrently (`concurrent_chroms > 1`, the
/// livelock repro's regime), each chromosome's pool spawns its own
/// `shard-worker-0`, `shard-worker-1`, ... independently, so e.g. two
/// concurrent single-worker pools BOTH name their one thread
/// `shard-worker-0` -- a `comm`-name lookup would resolve to whichever one
/// `/proc/self/task` iteration happens to find first, misattributing CPU
/// across chromosomes. The caller creates a fresh registry per
/// `process_chromosome` call and hands the same `Arc` to `monitor::spawn_sampler`.
///
/// `pending_gauge` is the same per-chrom high-water gauge the sampler reads:
/// the collector's reorder backlog (`pending`) is unbounded and otherwise
/// unobservable, yet it is a real peak-RSS term. Observed on every insert via
/// [`PendingBacklog::insert_observing`], which reads the backlog BEFORE
/// adding the arriving chunk -- so `pending_hw` measures chunks already
/// waiting, not the arriving one, and an in-order stream that never buffers
/// anything records 0.
///
/// `pending_budget_bytes` bounds that same backlog: once its bytes exceed the
/// budget, [`Frontier::admit`] parks every worker except the one whose unit
/// owns the reorder buffer's head, so the backlog can shrink instead of
/// growing without limit. Pass `u64::MAX` to disable the gate.
///
/// Returns the [`ShardTotals`] (summed `dropped_out_of_scope`, `ref_excluded`,
/// and `normalized_total`) across every unit, or the first error encountered
/// (context-decorated).
#[allow(clippy::too_many_arguments)]
pub fn run<F, G>(
    chrom: &str,
    units: Vec<WorkUnit>,
    workers: usize,
    make_assembler: F,
    err_context: G,
    chunk_size: usize,
    tx_dense: &Sender<DenseChunk>,
    worker_tids: &Mutex<Vec<i32>>,
    pending_gauge: &Arc<crate::monitor::PendingGauge>,
    pending_budget_bytes: u64,
) -> Result<ShardTotals, ConversionError>
where
    F: Fn(&WorkUnit) -> Result<ChunkAssembler, ConversionError> + Sync,
    G: Fn(ConversionError, &WorkUnit) -> ConversionError + Sync,
{
    let n_units = units.len();
    if n_units == 0 {
        return Ok(ShardTotals {
            dropped_out_of_scope: 0,
            ref_excluded: 0,
            normalized_total: 0,
        });
    }
    let workers = workers.max(1);
    let cancel = Arc::new(AtomicBool::new(false));
    let frontier = Arc::new(Frontier::new(
        pending_budget_bytes,
        Arc::clone(pending_gauge),
    ));

    // Seed every unit up front on an unbounded queue, then drop the sending
    // half: a worker's `rx_work.recv()` returns `Err` once the queue is
    // drained and no senders remain -- its natural "no more work" signal.
    // (Unbounded is safe here: `units.len()` is small -- one entry per shard,
    // not per chunk/record.)
    let (tx_work, rx_work) = unbounded::<WorkUnit>();
    for u in &units {
        tx_work.send(*u).expect("seed shard work queue");
    }
    drop(tx_work);

    let (tx_res, rx_res) = bounded::<Msg>(workers * 2);

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let rx_work = rx_work.clone();
            let tx_res = tx_res.clone();
            let cancel = Arc::clone(&cancel);
            let frontier = Arc::clone(&frontier);
            // `F`/`G` are `Sync`, so `&F`/`&G` are `Send` -- borrowing them
            // (rather than requiring `Clone`) into every scoped worker is
            // sound and avoids cloning the closures' captured state.
            let make_assembler = &make_assembler;
            let err_context = &err_context;
            let name = format!("shard-worker-{i}");
            let handle = thread::Builder::new()
                .name(name.clone())
                .spawn_scoped(scope, move || {
                    // Register this OS thread's TID once at startup so
                    // `monitor.rs` can sample its CPU under the OWNING
                    // chrom (see `worker_tids`'s doc comment on `run` for
                    // why comm-name matching alone can't do this). Linux
                    // only (no `/proc`, and no portable `gettid`, elsewhere
                    // -- `monitor.rs`'s CPU columns already print `n/a` on
                    // those platforms). The `let _` keeps the capture used
                    // on every platform so non-Linux builds don't warn on
                    // an otherwise-unused closure capture.
                    let _ = &worker_tids;
                    #[cfg(target_os = "linux")]
                    worker_tids.lock().unwrap().push(current_tid());
                    // See `PanicGuard`'s doc comment: this must live for the
                    // whole closure so a panic anywhere below -- not just an
                    // explicit `Err` return -- still cancels the pool and
                    // wakes every parked worker.
                    let _panic_guard = PanicGuard {
                        cancel: Arc::clone(&cancel),
                        frontier: Arc::clone(&frontier),
                    };
                    while !cancel.load(Ordering::Relaxed) {
                        let unit = match rx_work.recv() {
                            Ok(u) => u,
                            Err(_) => break, // queue drained, no more work
                        };
                        let unit_start = std::time::Instant::now();
                        // Reader reuse note: a fresh `ChunkAssembler` (and
                        // therefore a fresh indexed-fetch `RecordSource`) is
                        // built per unit rather than re-fetching an existing
                        // reader's region -- acceptable since fetch/seek cost
                        // is dwarfed by the per-shard decode work it does
                        // once positioned, and it keeps this pool
                        // backend-agnostic (PGEN readers seek by variant
                        // index, not by an HTSlib-style `fetch`).
                        let mut asm = match make_assembler(&unit) {
                            Ok(a) => a,
                            Err(e) => {
                                let _ = tx_res.send(Msg::Err(err_context(e, &unit)));
                                cancel.store(true, Ordering::Relaxed);
                                frontier.wake_all();
                                return;
                            }
                        };
                        let mut local = 0usize;
                        loop {
                            if cancel.load(Ordering::Relaxed) {
                                return;
                            }
                            match asm.read_next_chunk(chunk_size, local, None) {
                                Ok(Some(chunk)) => {
                                    // Park if the backlog is over budget and this unit is not the
                                    // head. See `Frontier` for the deadlock argument.
                                    frontier.admit(unit.ordinal, &cancel);
                                    if cancel.load(Ordering::Relaxed) {
                                        return;
                                    }
                                    trace_ll!(
                                        "[trace {chrom}] reader: shard {i} assembled chunk \
                                         (unit ordinal {}) local={local} rows={}",
                                        unit.ordinal,
                                        chunk.pos.len()
                                    );
                                    if tx_res
                                        .send(Msg::Chunk {
                                            unit_ordinal: unit.ordinal,
                                            local,
                                            chunk,
                                        })
                                        .is_err()
                                    {
                                        // Collector gone -- nothing left to do.
                                        return;
                                    }
                                    local += 1;
                                }
                                Ok(None) => break,
                                Err(e) => {
                                    let _ = tx_res.send(Msg::Err(err_context(e, &unit)));
                                    cancel.store(true, Ordering::Relaxed);
                                    frontier.wake_all();
                                    return;
                                }
                            }
                        }
                        // Per-unit wall time. Shard skew is what distinguishes
                        // "too few readers" from "readers unevenly loaded", and
                        // it cannot be inferred from aggregate CPU.
                        tracing::trace!(
                            target: "genoray::monitor",
                            unit_ordinal = unit.ordinal,
                            unit_secs = unit_start.elapsed().as_secs_f64(),
                            "shard unit done"
                        );
                        if tx_res
                            .send(Msg::Done {
                                unit_ordinal: unit.ordinal,
                                dropped: asm.dropped_out_of_scope(),
                                ref_excluded: asm.ref_excluded(),
                                normalized: asm.normalized_total(),
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                })
                .expect("spawn shard worker");
            handles.push((name, handle));
        }
        // Drop the collector's own clones so the results channel closes once
        // every worker thread has exited (they hold the only remaining
        // senders) -- the recv loop below relies on that as its
        // deadlock-free terminal condition on the error path, where
        // `done_count` may never reach `n_units` (a cancelled worker can
        // abandon a unit it never started).
        drop(rx_work);
        drop(tx_res);

        let mut pending = PendingBacklog::new();
        let mut rb = ReorderBuffer::new(n_units);
        let mut total_dropped = 0u64;
        let mut total_ref_excluded = 0u64;
        let mut total_normalized = 0u64;
        let mut first_err: Option<ConversionError> = None;
        let mut done_count = 0usize;

        while let Ok(msg) = rx_res.recv() {
            match msg {
                Msg::Chunk {
                    unit_ordinal,
                    local,
                    chunk,
                } => {
                    // Every emitted tag corresponds to a chunk inserted here
                    // first (either released immediately by the
                    // `ordinal == head` fast path, or buffered and later
                    // flushed on a `Done`) -- a `Done` never emits a chunk
                    // tag, so `pending.remove` below always finds its entry.
                    // `insert_observing` reads the gauge BEFORE adding this
                    // chunk, so the head-fast-path case (never actually
                    // waits) contributes 0 to `pending_hw`.
                    pending.insert_observing((unit_ordinal, local), chunk, pending_gauge);
                    if first_err.is_none() {
                        rb.push(unit_ordinal, local, false, &mut |gid, tag| {
                            if let Some(mut c) = pending.remove(&tag) {
                                c.chunk_id = gid;
                                trace_ll!(
                                    "[trace {chrom}] reader: forwarded ordinal {gid} to tx_dense"
                                );
                                tx_dense.send(c).ok();
                            }
                        });
                    }
                    frontier.publish(pending.bytes, rb.head());
                }
                Msg::Done {
                    unit_ordinal,
                    dropped,
                    ref_excluded,
                    normalized,
                } => {
                    done_count += 1;
                    if first_err.is_none() {
                        total_dropped += dropped;
                        total_ref_excluded += ref_excluded;
                        total_normalized += normalized;
                        rb.push(unit_ordinal, 0, true, &mut |gid, tag| {
                            if let Some(mut c) = pending.remove(&tag) {
                                c.chunk_id = gid;
                                trace_ll!(
                                    "[trace {chrom}] reader: forwarded ordinal {gid} to tx_dense"
                                );
                                tx_dense.send(c).ok();
                            }
                        });
                    }
                    frontier.publish(pending.bytes, rb.head());
                    if done_count == n_units {
                        // Every unit accounted for -- no further messages
                        // are possible. Don't wait for the channel to
                        // physically close (workers may still be spinning
                        // on an empty `rx_work`).
                        break;
                    }
                }
                Msg::Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                    cancel.store(true, Ordering::Relaxed);
                    frontier.wake_all();
                    // Keep draining (don't `break`): a worker may be
                    // blocked on `tx_res.send` for a message already
                    // in flight, and only stops once it observes `cancel`
                    // at its next check -- breaking early here could
                    // deadlock that worker's `join()` below. The loop's
                    // natural exit is the channel closing once every
                    // worker has returned.
                }
            }
        }

        // Defensive: any producer still parked on `admit` after the recv
        // loop exits (e.g. the collector broke out via `done_count ==
        // n_units` while a non-head unit was mid-wait) must be woken so its
        // worker thread can observe `cancel` (if set) or simply return --
        // otherwise the `join()` below would hang on it.
        frontier.wake_all();

        for (name, handle) in handles {
            if handle.join().is_err() && first_err.is_none() {
                first_err = Some(ConversionError::WorkerPanicked { thread: name });
            }
        }

        first_err.map(Err).unwrap_or(Ok(ShardTotals {
            dropped_out_of_scope: total_dropped,
            ref_excluded: total_ref_excluded,
            normalized_total: total_normalized,
        }))
    })
}

#[cfg(test)]
mod tests {
    use super::{Frontier, PendingBacklog, ReorderBuffer, ShardTotals, run};
    use crate::chunk_assembler::ChunkAssembler;
    use crate::error::ConversionError;
    use crate::monitor::PendingGauge;
    use crate::normalize::CheckRef;
    use crate::record_source::{Calls, FormatVals, RawRecord, RecordSource};
    use crate::shard::WorkUnit;
    use crate::types::{BitGrid3, DenseChunk};
    use crossbeam_channel::unbounded;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// A trivially small, distinctly-non-empty chunk -- just enough that
    /// `approx_bytes()` is nonzero so the byte high-water is exercisable too.
    fn tiny_chunk() -> DenseChunk {
        let mut genos = BitGrid3::zeros(1, 1, 2);
        genos.or_bit(0, true);
        genos.or_bit(1, true);
        DenseChunk {
            chunk_id: 0,
            pos: vec![100],
            global_idx: vec![-1],
            ilens: vec![0],
            alt: b"C".to_vec(),
            alt_offsets: vec![0, 1],
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
            carriers: None,
            format_by_carrier: None,
        }
    }

    #[test]
    fn pending_hw_zero_for_in_order_stream() {
        // Two shards, both streamed strictly in (ordinal, local) order --
        // exactly the "perfectly ordered" case the bug made impossible to
        // observe as zero. Every chunk lands on `ReorderBuffer::head` and is
        // released via the `ordinal == head` fast path, so it should never
        // sit in `pending` at all.
        let gauge = PendingGauge::default();
        let mut pending = PendingBacklog::new();
        let mut rb = ReorderBuffer::new(2);

        for ordinal in [0usize, 1] {
            for local in [0usize, 1] {
                pending.insert_observing((ordinal, local), tiny_chunk(), &gauge);
                rb.push(ordinal, local, false, &mut |_gid, tag| {
                    pending.remove(&tag);
                });
            }
            pending_done(&mut rb, &mut pending, ordinal);
        }

        assert_eq!(
            gauge.len_highwater.load(Ordering::Relaxed),
            0,
            "an in-order stream never buffers anything -- pending_hw must be 0, not floored at 1"
        );
        assert_eq!(gauge.bytes_highwater.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn pending_hw_positive_for_out_of_order_stream() {
        // Shard 1 races ahead of shard 0 (same arrival pattern as
        // `emits_in_ordinal_order_despite_out_of_order_arrival` below): its
        // chunks must sit in `pending` until shard 0 finishes, so the
        // high-water must reflect that real backlog.
        let gauge = PendingGauge::default();
        let mut pending = PendingBacklog::new();
        let mut rb = ReorderBuffer::new(2);

        pending.insert_observing((1, 0), tiny_chunk(), &gauge);
        rb.push(1, 0, false, &mut |_gid, tag| {
            pending.remove(&tag);
        });
        pending.insert_observing((1, 1), tiny_chunk(), &gauge);
        rb.push(1, 1, false, &mut |_gid, tag| {
            pending.remove(&tag);
        });
        pending_done(&mut rb, &mut pending, 1);

        assert!(
            gauge.len_highwater.load(Ordering::Relaxed) >= 1,
            "shard 1's chunks genuinely wait on shard 0 -- pending_hw must be > 0"
        );
        assert!(gauge.bytes_highwater.load(Ordering::Relaxed) > 0);

        pending_done(&mut rb, &mut pending, 0);
    }

    /// Drive a shard's `Done` signal through the same reorder-then-release
    /// dance the collector loop uses for `Msg::Done`.
    fn pending_done(rb: &mut ReorderBuffer, pending: &mut PendingBacklog, ordinal: usize) {
        rb.push(ordinal, 0, true, &mut |_gid, tag| {
            pending.remove(&tag);
        });
    }

    #[test]
    fn emits_in_ordinal_order_despite_out_of_order_arrival() {
        // Two shards; shard 1 finishes its chunks before shard 0.
        let mut rb = ReorderBuffer::new(2);
        let mut emitted = Vec::new();
        // (ordinal, local, done)
        rb.push(1, 0, false, &mut |gid, tag| emitted.push((gid, tag)));
        rb.push(1, 1, false, &mut |gid, tag| emitted.push((gid, tag)));
        rb.push(1, 0, true, &mut |gid, tag| emitted.push((gid, tag))); // shard 1 done
        assert!(emitted.is_empty(), "must wait for shard 0");
        rb.push(0, 0, false, &mut |gid, tag| emitted.push((gid, tag)));
        rb.push(0, 0, true, &mut |gid, tag| emitted.push((gid, tag))); // shard 0 done -> flush 0 then 1
        assert_eq!(
            emitted,
            vec![(0, (0, 0)), (1, (1, 0)), (2, (1, 1))],
            "global ids 0,1,2 assigned in (ordinal, local) order"
        );
    }

    /// Runs `f.admit(ordinal, &cancel)` on its own thread (never `thread::scope`,
    /// whose implicit join at scope-exit would itself hang if `admit` never
    /// returns) and reports whether it completed within `timeout`. A
    /// still-parked thread is simply abandoned -- process teardown reclaims
    /// it, and letting it leak is what lets a regression here fail the
    /// assertion below instead of hanging the whole test binary (fatal on
    /// this project's Slurm/NFS cluster, where a hung process can't always be
    /// killed and can drain a compute node).
    fn admit_completes_within(f: Arc<Frontier>, ordinal: usize, timeout: Duration) -> bool {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            f.admit(ordinal, &cancel);
            let _ = tx.send(());
        });
        rx.recv_timeout(timeout).is_ok()
    }

    /// The unit that owns the head is NEVER parked, even with the backlog
    /// over budget. This is the whole deadlock-freedom argument: if the head
    /// could park, nothing would ever advance it.
    #[test]
    fn admit_never_parks_the_head_unit() {
        let gauge = Arc::new(PendingGauge::default());
        let f = Arc::new(Frontier::new(0, Arc::clone(&gauge)));
        f.publish(u64::MAX, 7);
        assert!(
            admit_completes_within(f, 7, Duration::from_secs(2)),
            "the head unit must never park, even over budget"
        );
        assert_eq!(
            gauge.parks.load(Ordering::Relaxed),
            0,
            "the head-exemption branch must never record a park"
        );
    }

    /// A non-head producer parks while the backlog is over budget, and wakes
    /// once the head advances past it -- not only when bytes drop.
    #[test]
    fn admit_parks_a_non_head_unit_until_the_head_advances() {
        let gauge = Arc::new(PendingGauge::default());
        let f = Arc::new(Frontier::new(0, Arc::clone(&gauge)));
        f.publish(100, 0);
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        {
            let f = Arc::clone(&f);
            let cancel = Arc::clone(&cancel);
            thread::spawn(move || {
                f.admit(3, &cancel);
                let _ = tx.send(());
            });
        }
        assert_eq!(
            rx.recv_timeout(Duration::from_millis(200)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "unit 3 must park behind head 0"
        );
        f.publish(100, 3);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)),
            Ok(()),
            "unit 3 must wake once the head advances past it"
        );
        assert!(
            gauge.parks.load(Ordering::Relaxed) > 0,
            "unit 3's block above must have gone through the parking branch"
        );
    }

    /// A parked producer must observe `cancel` so the error path can tear the
    /// pool down instead of hanging in `join()`.
    #[test]
    fn admit_releases_a_parked_unit_on_cancel() {
        let gauge = Arc::new(PendingGauge::default());
        let f = Arc::new(Frontier::new(0, Arc::clone(&gauge)));
        f.publish(100, 0);
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        {
            let f = Arc::clone(&f);
            let cancel = Arc::clone(&cancel);
            thread::spawn(move || {
                f.admit(3, &cancel);
                let _ = tx.send(());
            });
        }
        assert_eq!(
            rx.recv_timeout(Duration::from_millis(200)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "unit 3 must park behind head 0"
        );
        cancel.store(true, Ordering::Relaxed);
        f.wake_all();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)),
            Ok(()),
            "a parked unit must observe cancel and return"
        );
    }

    /// A budget of `u64::MAX` disables the gate entirely -- the PGEN path and
    /// the existing tests rely on this.
    #[test]
    fn an_unbounded_budget_never_parks_anything() {
        let gauge = Arc::new(PendingGauge::default());
        let f = Arc::new(Frontier::new(u64::MAX, Arc::clone(&gauge)));
        f.publish(u64::MAX, 0);
        assert!(
            admit_completes_within(f, 99, Duration::from_secs(2)),
            "u64::MAX must disable the gate entirely"
        );
        assert_eq!(
            gauge.parks.load(Ordering::Relaxed),
            0,
            "an unbounded budget must never invoke the parking branch"
        );
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

    // ---- `run()` integration coverage ----
    //
    // Everything above drives `Frontier` directly through `publish`/`admit`.
    // Nothing exercises the INTEGRATION: real scoped worker threads, the FIFO
    // work queue, the reorder buffer, and the admission gate running
    // together through `run()` itself. `RecordSource` is a one-method trait
    // behind a `Box<dyn>` (see `record_source.rs`), so a stub source is a
    // `VecDeque<RawRecord>` plus an optional per-record sleep.

    const N_SAMPLES: usize = 1;
    const PLOIDY: usize = 2;
    /// Records per unit. Both fit in a single `read_next_chunk` call (well
    /// under the `chunk_size` used below), so each unit produces exactly one
    /// `DenseChunk`.
    const RECORDS_PER_UNIT: u32 = 2;
    /// Enough units that a park is FORCED by channel capacity rather than won
    /// by a scheduling race.
    ///
    /// `run` calls `Frontier::admit` once per assembled chunk, immediately
    /// before pushing it onto `tx_res` -- and `tx_res` is `bounded(workers * 2)`,
    /// so with `workers = 4` the first 8 messages (each unit sends a `Chunk`
    /// plus a `Done`, so the first four non-head units) are buffered without
    /// the collector ever having to run. Until the collector runs, nothing is
    /// added to the backlog, `pending_bytes` stays 0, and every `admit` in that
    /// window passes.
    ///
    /// At the old value of 8 there were only 7 non-head units, so that free
    /// window covered most of the run and whether ANY `admit` ever saw a
    /// non-empty backlog came down to how promptly the collector thread got
    /// scheduled. On a 48-core dev node it always did; on a 2-core CI runner it
    /// did not, and `parks > 0` failed against a perfectly healthy gate.
    ///
    /// With 32 units the buffer saturates, `tx_res.send` blocks, and the
    /// collector MUST drain to let the producers continue. Two drained chunks
    /// (92 bytes) already exceed the 64-byte budget the tests use, and the head
    /// is held open meanwhile, so the backlog only grows -- leaving ~27 further
    /// admits that must park. That is a capacity argument, not a timing one.
    const N_UNITS: usize = 32;

    /// How long ordinal 0 waits for the first park before giving up and letting
    /// its test's `parks > 0` assertion report the real failure. Comfortably
    /// inside `run_with_deadlock_guard`'s 10s `recv_timeout`, so a gate that
    /// never parks still fails with the assertion's message rather than the
    /// guard's misleading "deadlocked" one.
    const PARK_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(3);

    /// Blocks until `gauge` records its first park, or `PARK_HANDSHAKE_TIMEOUT`
    /// elapses.
    ///
    /// This handshake is what makes ordinal 0 finish LAST *deterministically*.
    /// It replaces a 5ms-per-record `thread::sleep`, which only held the head
    /// open for as long as the other workers needed if they happened to get
    /// enough CPU inside those 5ms. Waiting on the park counter itself takes
    /// the machine out of that question: ordinal 0 proceeds exactly when a
    /// non-head worker has actually parked behind it, on any number of cores,
    /// and holds the head open for as long as that takes.
    ///
    /// This is one half of the fix for a CI failure ("zero parks recorded") that
    /// reproduced only on a 2-core runner; `N_UNITS` carries the other half and
    /// the fuller explanation. Holding the head open is not on its own enough to
    /// force a park -- there also has to be more work than `tx_res` can buffer.
    fn wait_for_first_park(gauge: &PendingGauge) {
        let deadline = Instant::now() + PARK_HANDSHAKE_TIMEOUT;
        while gauge.parks.load(Ordering::Relaxed) == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// A `RecordSource` over an in-memory queue that, for ordinal 0, blocks
    /// until some other worker has parked. Used to make ordinal 0 finish LAST
    /// despite being dequeued first, forcing every other unit's chunk to buffer
    /// behind the still-open reorder head instead of streaming through by
    /// accident -- which is exactly how the end-to-end conversion measurement
    /// (Task 4) missed this path (it observed zero park events).
    ///
    /// `panic_after_wait`, when set, panics (after the wait, if any) instead of
    /// ever returning a record -- the injection point for the `PanicGuard`
    /// regression test: a worker that panics mid-unit instead of returning
    /// `Ok`/`Err` normally. Pairing it with `wait_for_park` is what guarantees
    /// the panic lands with workers already parked behind the head, which is
    /// the only arrangement that reaches the deadlock `PanicGuard` prevents.
    struct VecSource {
        records: VecDeque<RawRecord>,
        wait_for_park: Option<Arc<PendingGauge>>,
        panic_after_wait: bool,
    }

    impl RecordSource for VecSource {
        fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
            if let Some(gauge) = &self.wait_for_park {
                wait_for_first_park(gauge);
            }
            if self.panic_after_wait {
                panic!("VecSource: injected panic for PanicGuard regression test");
            }
            Ok(self.records.pop_front())
        }
    }

    /// One biallelic SNV atom: dense calls sized for `N_SAMPLES * PLOIDY`
    /// columns, no INFO/FORMAT fields (the `ChunkAssembler` below is built
    /// with `fields: &[]`), no reference (`fasta_path: None`).
    fn record(pos: u32) -> RawRecord {
        RawRecord {
            pos,
            reference: b"A".to_vec(),
            alts: vec![b"C".to_vec()],
            calls: Calls::Dense(vec![1, 0]),
            info_raw: Vec::new(),
            format_vals: FormatVals::Dense(Vec::new()),
            global_idx: pos as i32,
        }
    }

    /// `N_UNITS` work units, ordinals `0..N_UNITS`, each fed `RECORDS_PER_UNIT`
    /// records at globally-distinct positions. Ordinal 0's records carry the
    /// artificial per-record delay described on `VecSource`.
    fn seed_units() -> (Vec<WorkUnit>, Arc<HashMap<usize, Vec<RawRecord>>>) {
        let mut units = Vec::with_capacity(N_UNITS);
        let mut records_by_ordinal = HashMap::with_capacity(N_UNITS);
        for ordinal in 0..N_UNITS {
            let base = (ordinal as u32) * 100;
            units.push(WorkUnit {
                own_start: base,
                own_end: base + RECORDS_PER_UNIT,
                fetch_start: base,
                fetch_end: base + RECORDS_PER_UNIT,
                ordinal,
            });
            let records: Vec<RawRecord> = (0..RECORDS_PER_UNIT).map(|i| record(base + i)).collect();
            records_by_ordinal.insert(ordinal, records);
        }
        (units, Arc::new(records_by_ordinal))
    }

    /// Ordinal deliberately given more than `RECORDS_PER_UNIT` records by
    /// [`seed_units_with_a_multi_chunk_unit`], so it spans several chunks
    /// under a small `chunk_size` instead of the usual one-chunk-per-unit
    /// shape every other test here relies on.
    const MULTI_CHUNK_ORDINAL: usize = 3;
    const MULTI_CHUNK_RECORDS: u32 = 5;

    /// Like [`seed_units`], except ordinal `MULTI_CHUNK_ORDINAL` gets
    /// `MULTI_CHUNK_RECORDS` records instead of `RECORDS_PER_UNIT`. At the
    /// `chunk_size=2` used by its caller this spans 3 chunks
    /// (`ceil(5 / 2)`), exercising the reorder buffer's multi-`local` path
    /// -- a single shard streaming more than one chunk -- which no other
    /// test in this module reaches.
    fn seed_units_with_a_multi_chunk_unit() -> (Vec<WorkUnit>, Arc<HashMap<usize, Vec<RawRecord>>>)
    {
        let mut units = Vec::with_capacity(N_UNITS);
        let mut records_by_ordinal = HashMap::with_capacity(N_UNITS);
        for ordinal in 0..N_UNITS {
            let base = (ordinal as u32) * 100;
            let count = if ordinal == MULTI_CHUNK_ORDINAL {
                MULTI_CHUNK_RECORDS
            } else {
                RECORDS_PER_UNIT
            };
            units.push(WorkUnit {
                own_start: base,
                own_end: base + count,
                fetch_start: base,
                fetch_end: base + count,
                ordinal,
            });
            let records: Vec<RawRecord> = (0..count).map(|i| record(base + i)).collect();
            records_by_ordinal.insert(ordinal, records);
        }
        (units, Arc::new(records_by_ordinal))
    }

    /// Builds the `make_assembler` closure `run` requires: a fresh
    /// `ChunkAssembler` per unit wrapping a fresh `VecSource` cloned out of
    /// the shared record table. Only ordinal 0 gets the park handshake, which
    /// is what holds the reorder head open until the backlog behind it is real.
    /// `panic_ordinal`, when `Some`, makes that one ordinal's `VecSource`
    /// panic instead of ever returning a record (see `VecSource::panic_after_wait`)
    /// -- the `PanicGuard` regression test's injection point.
    fn make_assembler_fn(
        records_by_ordinal: Arc<HashMap<usize, Vec<RawRecord>>>,
        panic_ordinal: Option<usize>,
        gauge: Arc<PendingGauge>,
    ) -> impl Fn(&WorkUnit) -> Result<ChunkAssembler, ConversionError> + Sync + 'static {
        move |unit: &WorkUnit| {
            let records = records_by_ordinal
                .get(&unit.ordinal)
                .cloned()
                .unwrap_or_default();
            let wait_for_park = (unit.ordinal == 0).then(|| Arc::clone(&gauge));
            let panic_after_wait = panic_ordinal == Some(unit.ordinal);
            let source: Box<dyn RecordSource + Send> = Box::new(VecSource {
                records: records.into(),
                wait_for_park,
                panic_after_wait,
            });
            ChunkAssembler::new(
                source,
                N_SAMPLES,
                PLOIDY,
                None, // fasta_path: no reference needed
                "chrTest",
                false, // skip_out_of_scope
                CheckRef::Error,
                &[], // fields: no INFO/FORMAT requested
            )
        }
    }

    /// Runs `run(...)` on its own thread and joins with a timeout. Required
    /// per the task brief: a broken head-exemption HANGS rather than fails,
    /// and a hung `cargo test` reads as infrastructure trouble, not a red
    /// test, so a bare `handle.join()` is not an option here.
    fn run_with_deadlock_guard(
        units: Vec<WorkUnit>,
        records_by_ordinal: Arc<HashMap<usize, Vec<RawRecord>>>,
        workers: usize,
        chunk_size: usize,
        pending_budget_bytes: u64,
        panic_ordinal: Option<usize>,
    ) -> (
        Result<ShardTotals, ConversionError>,
        Vec<DenseChunk>,
        Arc<PendingGauge>,
    ) {
        let (tx_dense, rx_dense) = unbounded::<DenseChunk>();
        let gauge = Arc::new(PendingGauge::default());
        let (result_tx, result_rx) = mpsc::channel();
        {
            let gauge = Arc::clone(&gauge);
            let make_assembler =
                make_assembler_fn(records_by_ordinal, panic_ordinal, Arc::clone(&gauge));
            thread::spawn(move || {
                let worker_tids: Mutex<Vec<i32>> = Mutex::new(Vec::new());
                let res = run(
                    "chrTest",
                    units,
                    workers,
                    make_assembler,
                    |e, _u| e,
                    chunk_size,
                    &tx_dense,
                    &worker_tids,
                    &gauge,
                    pending_budget_bytes,
                );
                let _ = result_tx.send(res);
            });
        }
        let result = match result_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(res) => res,
            Err(_) => panic!(
                "shard_exec::run deadlocked -- the head-exemption did not release a \
                 parked producer within 10s (pending_budget_bytes={pending_budget_bytes})"
            ),
        };
        let mut chunks = Vec::new();
        while let Ok(c) = rx_dense.try_recv() {
            chunks.push(c);
        }
        (result, chunks, gauge)
    }

    #[test]
    fn run_emits_every_unit_in_order_under_a_bounded_backlog() {
        let (units, records_by_ordinal) = seed_units();
        // ~1.4 chunks' worth: one `DenseChunk` of RECORDS_PER_UNIT=2 atoms at
        // N_SAMPLES=1/PLOIDY=2 (no staged fields) costs, per
        // `DenseChunk::approx_bytes`: meta (pos+global_idx+ilens 4B/atom,
        // alt 1B/atom, alt_offsets (v+1)*4B) + genos (1 word) =
        // (8+8+8+2+12) + 8 = 46 bytes. A budget of 64 lets exactly one
        // shard's chunk through before a second non-head arrival (92 bytes
        // total) genuinely exceeds it.
        let pending_budget_bytes = 64u64;
        let (result, chunks, gauge) = run_with_deadlock_guard(
            units,
            records_by_ordinal,
            4,
            100,
            pending_budget_bytes,
            None,
        );
        let totals = result.expect("run must succeed");

        assert!(
            gauge.len_highwater.load(Ordering::Relaxed) > 0,
            "ordinal 0 was made to lag on purpose -- a real backlog must have \
             been observed, or admission control was never exercised"
        );
        // `len_highwater > 0` only proves a backlog FORMED, not that
        // `Frontier::admit`'s parking branch ran -- a no-op gate would let
        // the same backlog accumulate. `parks` is the only signal that a
        // non-head worker actually blocked; this is the instrumentation
        // Task 4's end-to-end measurement found zero of, end to end.
        assert!(
            gauge.parks.load(Ordering::Relaxed) > 0,
            "zero parks recorded -- the head-exemption path in Frontier::admit \
             never ran, so this test exercised no admission control at all"
        );

        assert_eq!(chunks.len(), N_UNITS, "one chunk per unit expected");
        for (expect_id, c) in chunks.iter().enumerate() {
            assert_eq!(
                c.chunk_id, expect_id,
                "chunks must arrive on tx_dense in ascending ordinal order"
            );
            // `chunk_id` is assigned by the collector monotonically AT EMIT
            // TIME regardless of which unit actually produced the chunk, so
            // the check above alone is vacuous -- it would pass even if
            // chunk contents were shuffled across units. Tie the payload
            // itself to the unit that must have produced it: `seed_units`
            // gives ordinal `o` records starting at `o * 100`
            // (`record(base + i)` in `seed_units`), and this unit produces
            // exactly one chunk, so its first record's `pos` pins it to
            // ordinal `expect_id`.
            assert_eq!(
                c.pos[0],
                (expect_id as u32) * 100,
                "chunk {expect_id} carries the wrong unit's records"
            );
        }
        let total_records: usize = chunks.iter().map(|c| c.pos.len()).sum();
        assert_eq!(
            total_records,
            N_UNITS * RECORDS_PER_UNIT as usize,
            "every record fed to a unit must reach tx_dense inside some chunk"
        );
        assert_eq!(totals.dropped_out_of_scope, 0);
        assert_eq!(totals.ref_excluded, 0);
        assert_eq!(totals.normalized_total, 0);
    }

    #[test]
    fn bounded_and_unbounded_backlogs_produce_identical_output() {
        #[allow(clippy::type_complexity)]
        fn chunk_signature(
            c: &DenseChunk,
        ) -> (
            usize,
            Vec<u32>,
            Vec<i32>,
            Vec<i32>,
            Vec<u8>,
            Vec<u32>,
            Vec<u64>,
            (usize, usize, usize),
        ) {
            (
                c.chunk_id,
                c.pos.clone(),
                c.global_idx.clone(),
                c.ilens.clone(),
                c.alt.clone(),
                c.alt_offsets.clone(),
                c.genos.words.clone(),
                c.genos.shape,
            )
        }

        // `seed_units` gives every unit exactly `RECORDS_PER_UNIT` records,
        // which fits in a single `read_next_chunk` call -- so every unit
        // ever produces exactly one chunk, and the reorder buffer's
        // multi-`local` path (a single shard streaming more than one chunk)
        // is never exercised. `seed_units_with_a_multi_chunk_unit` gives one
        // ordinal enough records to span several chunks at the
        // `chunk_size=2` used below, while every other unit still fits in
        // one -- both runs use the same shape, so the comparison remains
        // apples-to-apples.
        let (units_a, records_a) = seed_units_with_a_multi_chunk_unit();
        let (result_a, chunks_a, _) =
            run_with_deadlock_guard(units_a, records_a, 4, 2, u64::MAX, None);

        let (units_b, records_b) = seed_units_with_a_multi_chunk_unit();
        let (result_b, chunks_b, _) = run_with_deadlock_guard(units_b, records_b, 4, 2, 64, None);

        let totals_a = result_a.expect("unbounded run must succeed");
        let totals_b = result_b.expect("bounded run must succeed");
        assert_eq!(totals_a.dropped_out_of_scope, totals_b.dropped_out_of_scope);
        assert_eq!(totals_a.ref_excluded, totals_b.ref_excluded);
        assert_eq!(totals_a.normalized_total, totals_b.normalized_total);

        // Two empty vectors would trivially compare equal below -- pin both
        // non-emptiness and the exact expected record/chunk counts (the
        // oversized unit alone spans multiple chunks) before trusting the
        // signature comparison that follows.
        assert!(
            !chunks_a.is_empty(),
            "run produced no chunks -- an empty comparison proves nothing"
        );
        let expected_total_records =
            (N_UNITS - 1) * RECORDS_PER_UNIT as usize + MULTI_CHUNK_RECORDS as usize;
        for (label, chunks) in [("bounded", &chunks_a), ("unbounded", &chunks_b)] {
            let total_records: usize = chunks.iter().map(|c| c.pos.len()).sum();
            assert_eq!(
                total_records, expected_total_records,
                "{label} run: every record fed to a unit (including the oversized \
                 one) must reach tx_dense"
            );
            assert!(
                chunks.len() > N_UNITS,
                "{label} run: the oversized unit must have spanned more than one \
                 chunk, so total chunks must exceed the unit count"
            );
        }

        let sig_a: Vec<_> = chunks_a.iter().map(chunk_signature).collect();
        let sig_b: Vec<_> = chunks_b.iter().map(chunk_signature).collect();
        assert_eq!(
            sig_a, sig_b,
            "admission control must change WHEN a chunk is produced, never WHAT"
        );
    }

    /// `PanicGuard` regression test: ordinal 0 -- the reorder head -- panics
    /// instead of ever completing, and its park handshake (see `VecSource`'s
    /// doc comment) holds that panic back until the other units have filled the
    /// backlog past `pending_budget_bytes` and genuinely parked in
    /// `Frontier::admit` behind the still-open head. That is exactly the
    /// finding's scenario: a dead head with parked workers behind it and no
    /// one left who can ever advance it. Without `PanicGuard`, this hangs
    /// instead of returning -- caught only because it is routed through
    /// `run_with_deadlock_guard`, whose 10s `recv_timeout` turns that hang
    /// into a failing assertion instead of a wedged `cargo test` process.
    #[test]
    fn panicking_head_worker_returns_worker_panicked_instead_of_hanging() {
        let (units, records_by_ordinal) = seed_units();
        // Same arithmetic as `run_emits_every_unit_in_order_under_a_bounded_backlog`:
        // one chunk costs 46 bytes, so a budget of 64 admits exactly one
        // non-head chunk before the next one (92 bytes total) genuinely
        // exceeds it and must park.
        let pending_budget_bytes = 64u64;
        let (result, _chunks, gauge) = run_with_deadlock_guard(
            units,
            records_by_ordinal,
            4,
            100,
            pending_budget_bytes,
            Some(0),
        );

        assert!(
            gauge.parks.load(Ordering::Relaxed) > 0,
            "a non-head worker must have actually parked behind ordinal 0 \
             before it panicked, or this test never reaches the deadlock \
             PanicGuard is meant to prevent"
        );
        match result {
            Err(ConversionError::WorkerPanicked { .. }) => {}
            Ok(_) => panic!("expected Err(ConversionError::WorkerPanicked {{ .. }}), got Ok(_)"),
            Err(other) => {
                panic!("expected Err(ConversionError::WorkerPanicked {{ .. }}), got Err({other})")
            }
        }
    }
}

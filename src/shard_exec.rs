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

/// Current OS thread id (Linux `gettid`, distinct from the process pid and
/// from Rust's internal `std::thread::ThreadId`). Used ONLY to populate
/// `worker_tids` for `monitor.rs`'s per-chrom CPU sampling -- see that
/// param's doc comment on [`run`] for why a shared TID registry is needed
/// instead of matching threads by `comm` name.
#[cfg(target_os = "linux")]
fn current_tid() -> i32 {
    // SAFETY: SYS_gettid takes no arguments and cannot fail.
    unsafe { libc::syscall(libc::SYS_gettid) as i32 }
}

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
                    while !cancel.load(Ordering::Relaxed) {
                        let unit = match rx_work.recv() {
                            Ok(u) => u,
                            Err(_) => break, // queue drained, no more work
                        };
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
                                    return;
                                }
                            }
                        }
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

        let mut pending: HashMap<(usize, usize), DenseChunk> = HashMap::new();
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
                    pending.insert((unit_ordinal, local), chunk);
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
    use super::ReorderBuffer;

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
}

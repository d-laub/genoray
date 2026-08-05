// src/monitor.rs
//! Optional per-contig pipeline monitoring. Channel fill levels work on any
//! platform; per-thread CPU% is read from Linux `/proc/self/task/<tid>/stat` and
//! is unavailable on macOS (no `/proc`) — there the CPU columns print `n/a`.
//
// ─────────────────────────────────────────────────────────────────────────────
// Periodic monitoring sampler
//
// Runs as its own OS thread per chrom. Every 2 seconds, prints to stderr:
//   - bounded channel fill levels (tx_dense / tx_sparse / tx_long)
//   - per-thread CPU% for the four pipeline threads (read / exec / cw / lw)
//
// CPU% is derived from /proc/self/task/<TID>/stat (utime+stime ticks). TIDs are
// resolved by walking /proc/self/task/* and matching each thread's `comm` file
// against the names we set with thread::Builder::name() — which is why thread
// naming is a hard prerequisite for this sampler.
//
// Linux clock ticks/sec (CLK_TCK) is hardcoded to 100. That's CONFIG_HZ_100,
// the kernel default for x86_64 servers in most modern distros (Ubuntu, Debian,
// stock kernels). Other configs (250, 300, 1000) make the printed % off by a
// constant factor; relative comparisons across stages remain valid.
// ─────────────────────────────────────────────────────────────────────────────
use crossbeam_channel::Sender;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::types::{DenseChunk, SparseChunk};

const CLK_TCK_HZ: f64 = 100.0;

/// A per-chrom registry of OS thread ids for one worker POOL, populated by the
/// pool's threads at startup and read by the sampler.
///
/// Needed wherever a pool's thread names are not unique per chrom (shard
/// workers: `shard-worker-0` in every chrom's pool) or not unique at all
/// (executor workers: the outer `exec-{chrom}` thread and worker 0 share a
/// name, so a `comm` lookup resolves whichever `/proc/self/task` iteration
/// finds first -- and the outer thread is blocked in `thread::scope`, so
/// resolving to IT reports 0% while the pool is pegged).
pub type TidRegistry = Arc<Mutex<Vec<i32>>>;

/// Current OS thread id (Linux `gettid`, distinct from the process pid and
/// from Rust's internal `std::thread::ThreadId`). Used ONLY to populate a
/// [`TidRegistry`] for per-chrom CPU sampling.
#[cfg(target_os = "linux")]
pub fn current_tid() -> i32 {
    // SAFETY: SYS_gettid takes no arguments and cannot fail.
    unsafe { libc::syscall(libc::SYS_gettid) as i32 }
}

/// Aggregate CPU% across a whole pool since the previous tick.
///
/// `elapsed` MUST be the measured wall time since the previous tick, not the
/// nominal sample interval: the sampler's `sleep` overshoots under load, and
/// dividing a longer interval's tick delta by the shorter nominal one reports
/// CPU% above what the threads could possibly have used.
///
/// `prev` is this pool's own per-TID tick history and is REPLACED, not merged:
/// a TID that vanished between ticks must not keep contributing its last
/// delta. `None` when the registry is empty (pool never ran, or a
/// non-registering fallback path) so the caller can print `n/a` rather than a
/// 0% that reads as "ran, but idle".
fn pool_cpu_pct(
    tids: &TidRegistry,
    prev: &mut HashMap<i32, u64>,
    elapsed: Duration,
) -> Option<f64> {
    let tids: Vec<i32> = tids.lock().unwrap().clone();
    if tids.is_empty() {
        return None;
    }
    let mut dt_ticks = 0f64;
    let mut next_prev = HashMap::with_capacity(tids.len());
    for tid in &tids {
        let cur = read_thread_cpu_ticks(*tid);
        dt_ticks += cur.saturating_sub(prev.get(tid).copied().unwrap_or(0)) as f64;
        next_prev.insert(*tid, cur);
    }
    *prev = next_prev;
    Some(100.0 * dt_ticks / CLK_TCK_HZ / elapsed.as_secs_f64())
}

fn find_thread_tid_by_name(name: &str) -> Option<i32> {
    let entries = std::fs::read_dir("/proc/self/task").ok()?;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        // Skip entries that aren't valid numeric TIDs — `continue`, don't abort.
        let Some(tid_str) = file_name.to_str() else {
            continue;
        };
        let Ok(tid) = tid_str.parse::<i32>() else {
            continue;
        };
        if let Ok(comm) = std::fs::read_to_string(entry.path().join("comm"))
            && comm.trim() == name
        {
            return Some(tid);
        }
    }
    None
}

fn read_thread_cpu_ticks(tid: i32) -> u64 {
    // Per `man 5 proc`: the comm field is parenthesized and may contain spaces.
    // Split on the LAST `)` to skip past it, then index into space-separated fields.
    // After (comm), fields map to cols[0..]:
    //   col[0]=state, col[1]=ppid, col[2]=pgrp, col[3]=session, col[4]=tty_nr,
    //   col[5]=tpgid, col[6]=flags, col[7..10]=minflt/cminflt/majflt/cmajflt,
    //   col[11]=utime, col[12]=stime
    let s = match std::fs::read_to_string(format!("/proc/self/task/{}/stat", tid)) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let close = match s.rfind(')') {
        Some(i) => i,
        None => return 0,
    };
    let cols: Vec<&str> = s[close + 1..].split_whitespace().collect();
    let utime: u64 = cols.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stime: u64 = cols.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
    utime + stime
}

// Sample cadence in seconds. Read once at sampler-spawn time from
// `GENORAY_SAMPLE_INTERVAL` (default 5). Set to "0" to disable monitoring entirely
// for production runs where stderr volume matters.
fn sample_interval_secs() -> u64 {
    std::env::var("GENORAY_SAMPLE_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

/// Granularity at which a sleeping sampler notices it has been asked to stop.
///
/// Small enough to be invisible against a contig, large enough that the wakeup
/// is free: 100 ticks over a default 5s interval, each doing one atomic load.
const STOP_POLL: Duration = Duration::from_millis(50);

/// Wait one sample interval, but wake early if `stop` is set. Returns whether
/// the wait was cut short (i.e. the caller should exit rather than sample).
///
/// A plain `sleep(interval)` is what made the sampler cost real wall time.
/// `process_chromosome` shuts a contig down by setting `stop` and then joining
/// this thread, and the loop only tested `stop` AFTER its sleep -- so every
/// contig's shutdown waited out the remainder of the current tick. Measured on
/// a 22-contig human-skewed corpus, per-contig time was pinned at 5.3-5.5s
/// (the 5s default interval plus the 300ms settle) no matter the cohort width
/// or the contig's record count, against 1.0-1.8s of actual work: 1.8-3.2x of
/// wall time, on both this branch and main, spent joining a sleeping thread.
///
/// Sampling CADENCE is unchanged -- this still waits the full interval when
/// nothing is shutting down, so every emitted `pipeline sampler` line covers
/// the same span it did before and the CPU% columns stay comparable.
fn sleep_until_tick_or_stop(interval: Duration, stop: &AtomicBool) -> bool {
    let deadline = Instant::now() + interval;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return false;
        }
        if stop.load(Ordering::Relaxed) {
            return true;
        }
        std::thread::sleep(std::cmp::min(STOP_POLL, remaining));
    }
}

/// High-water gauge for the shard collector's reorder backlog.
///
/// `shard_exec`'s `pending` map is unbounded by construction: a fast shard's
/// chunks accumulate while the reorder head waits on a slow one. That backlog
/// is a second peak-RSS term alongside the in-flight `workers * chunk_bytes`,
/// and it is otherwise invisible -- nothing else observes the map.
///
/// High-water rather than instantaneous: the sampler ticks on a multi-second
/// interval and would routinely miss the transient peak that actually sets
/// peak RSS.
///
/// # Counting convention
///
/// `observe` is called from `shard_exec`'s single insert site BEFORE the
/// arriving chunk is added, so a high-water of `n` means `n` chunks were
/// ALREADY waiting when the next one arrived. An in-order stream -- every
/// chunk landing on the reorder head and releasing immediately -- therefore
/// reports 0, not a permanent 1.
///
/// This is load-bearing, not a detail. Downstream analysis
/// (`scripts/bench_svar2`) reads 0 as "no backlog exists"; a `+1` floor here
/// reads instead as a real second peak-RSS term that is present in every run,
/// which is enough on its own to decide the reader-budget question the
/// benchmark exists to answer.
///
/// Remove sites deliberately do NOT observe: `observe` is a `fetch_max` and a
/// remove can only lower both values, so observing after one is a provable
/// no-op.
#[derive(Debug, Default)]
pub struct PendingGauge {
    pub len_highwater: AtomicUsize,
    pub bytes_highwater: AtomicU64,
}

impl PendingGauge {
    /// Record the collector's current backlog. Monotonic in both dimensions;
    /// `Relaxed` is sufficient because these are diagnostics with no
    /// happens-before obligation to any other state.
    pub fn observe(&self, len: usize, bytes: u64) {
        self.len_highwater.fetch_max(len, Ordering::Relaxed);
        self.bytes_highwater.fetch_max(bytes, Ordering::Relaxed);
    }
}

/// The per-chrom gauges the sampler reads but does not own.
///
/// Grouped rather than passed positionally: each is a separate pool/backlog
/// that only the sampler correlates, and passing them individually put
/// `spawn_sampler` over clippy's argument limit.
pub struct PipelineProbes {
    /// Registry of `shard-worker-*` OS TIDs, populated by `shard_exec::run`
    /// (only the sharded VCF/PGEN branches use it -- stays empty, and the
    /// printed `shard` column stays `n/a`, for the single-reader fallback
    /// path). NOT resolved by matching the `shard-worker-{i}` thread `comm`
    /// name: worker names are pool-local (`shard-worker-0`, `-1`, ...), not
    /// chrom-qualified, so under `concurrent_chroms > 1` (the #135 livelock
    /// repro's regime) two chromosomes' pools both name a thread
    /// `shard-worker-0` -- a comm lookup would resolve to whichever one
    /// `/proc/self/task` iteration finds first, misattributing CPU across
    /// chromosomes. See `shard_exec::run`'s `worker_tids` doc comment.
    pub shard_worker_tids: TidRegistry,
    /// Registry of executor-worker OS TIDs, populated by
    /// `executor::run_compute_engine_multi`. Same motivation as
    /// `shard_worker_tids` -- see [`TidRegistry`] for why the executor pool
    /// cannot be sampled by `comm` name either.
    pub exec_worker_tids: TidRegistry,
    /// Reorder-backlog high-water for THIS chrom, updated by the shard
    /// collector. Stays zero on the single-reader fallback path.
    pub pending_gauge: Arc<PendingGauge>,
}

pub fn spawn_sampler(
    chrom: String,
    tx_dense: Sender<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    tx_long: Sender<Vec<u8>>,
    stop: Arc<AtomicBool>,
    probes: PipelineProbes,
) -> thread::JoinHandle<()> {
    let PipelineProbes {
        shard_worker_tids,
        exec_worker_tids,
        pending_gauge,
    } = probes;
    thread::Builder::new()
        .name(format!("samp-{}", chrom))
        .spawn(move || {
            let interval_secs = sample_interval_secs();
            // Disabled — drop Sender clones and exit immediately.
            if interval_secs == 0 {
                return;
            }
            let interval = Duration::from_secs(interval_secs);
            let start = Instant::now();
            // Match the names assigned to the singleton pipeline threads.
            // `exec` is deliberately absent: it is a POOL, sampled through
            // `exec_worker_tids` below (see [`TidRegistry`]).
            let names: Vec<String> = ["read", "cw", "lw"]
                .iter()
                .map(|p| format!("{}-{}", p, chrom))
                .collect();

            // Brief settle so the four pipeline threads register their /proc/.../comm
            // entries before the first lookup. Missing TIDs are re-resolved each tick.
            // Interruptible for the same reason the tick is: a contig whose whole
            // pipeline finishes inside the settle would otherwise still pay it in full.
            if sleep_until_tick_or_stop(Duration::from_millis(300), &stop) {
                return;
            }
            let mut tids: Vec<Option<i32>> =
                names.iter().map(|n| find_thread_tid_by_name(n)).collect();
            let mut prev_ticks: Vec<u64> = vec![0; names.len()];
            // Per-TID previous tick counts for the two POOL aggregates below
            // (the singleton pipeline threads use the parallel `prev_ticks`
            // Vec instead, since their TIDs are looked up by name once).
            let mut prev_shard_ticks: HashMap<i32, u64> = HashMap::new();
            let mut prev_exec_ticks: HashMap<i32, u64> = HashMap::new();

            // Channel capacities (bounded() guarantees Some(cap)).
            let dense_cap = tx_dense.capacity().unwrap_or(0);
            let sparse_cap = tx_sparse.capacity().unwrap_or(0);
            let long_cap = tx_long.capacity().unwrap_or(0);

            // Wall clock of the previous tick. Every CPU% below divides by the
            // MEASURED gap rather than `interval`, because `sleep` overshoots
            // once the machine is busy -- and an overshoot charged against the
            // nominal interval inflates every column, worst exactly when the
            // pipeline is most loaded and the reading matters most.
            let mut last_tick = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                if sleep_until_tick_or_stop(interval, &stop) {
                    break;
                }
                let now = Instant::now();
                let gap = now.duration_since(last_tick);
                last_tick = now;

                // Re-resolve any not-yet-found TIDs (handles slow startup).
                for (i, t) in tids.iter_mut().enumerate() {
                    if t.is_none() {
                        *t = find_thread_tid_by_name(&names[i]);
                    }
                }

                let cur: Vec<u64> = tids
                    .iter()
                    .map(|t| t.map(read_thread_cpu_ticks).unwrap_or(0))
                    .collect();
                let cpu_pcts: Vec<Option<f64>> = tids
                    .iter()
                    .zip(prev_ticks.iter())
                    .zip(cur.iter())
                    .map(|((t, p), c)| {
                        t.map(|_| {
                            let dt_ticks = c.saturating_sub(*p) as f64;
                            100.0 * dt_ticks / CLK_TCK_HZ / gap.as_secs_f64()
                        })
                    })
                    .collect();
                prev_ticks = cur;

                // `cpu_read` alone would under-report the sharded path: the
                // decode work lives in the `shard-worker-*` pool, and
                // `read-{chrom}` retains only the collector loop. Aggregate
                // the pool into its own `cpu_shard` column -- `n/a` when the
                // registry is empty (single-reader fallback path, where
                // `cpu_read` IS the whole reader and nothing else to sample).
                let shard_pct = pool_cpu_pct(&shard_worker_tids, &mut prev_shard_ticks, gap);
                let exec_pct = pool_cpu_pct(&exec_worker_tids, &mut prev_exec_ticks, gap);

                let fmt =
                    |o: Option<f64>| o.map_or_else(|| "n/a".to_string(), |v| format!("{:.0}%", v));
                let elapsed = start.elapsed().as_secs();
                tracing::trace!(
                    target: "genoray::monitor",
                    chrom = %chrom,
                    elapsed_s = elapsed,
                    // Measured gap this tick's CPU% are computed over. Well
                    // above the nominal interval means the sampler itself was
                    // starved, which is worth seeing next to the numbers.
                    gap_ms = gap.as_millis(),
                    dense = tx_dense.len(), dense_cap = dense_cap,
                    sparse = tx_sparse.len(), sparse_cap = sparse_cap,
                    long = tx_long.len(), long_cap = long_cap,
                    pending = pending_gauge.len_highwater.load(Ordering::Relaxed),
                    pending_bytes = pending_gauge.bytes_highwater.load(Ordering::Relaxed),
                    // In the sharded path `cpu_read` is the COLLECTOR (reorder
                    // + forward to tx_dense), not the decode work -- that is
                    // `cpu_shard`, aggregated across this chrom's shard-worker
                    // pool. `cpu_exec` is likewise a pool aggregate.
                    cpu_read = %fmt(cpu_pcts[0]),
                    cpu_shard = %fmt(shard_pct),
                    cpu_exec = %fmt(exec_pct),
                    cpu_cw = %fmt(cpu_pcts[1]),
                    cpu_lw = %fmt(cpu_pcts[2]),
                    "pipeline sampler"
                );
            }
            // tx_dense, tx_sparse, tx_long Sender clones drop here as the closure ends —
            // letting the executor / writer rx ends close once the original Senders also drop.
        })
        .expect("spawn sampler")
}

#[cfg(test)]
mod tests {
    use super::{PendingGauge, STOP_POLL, sleep_until_tick_or_stop};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    // The bug this guards: `process_chromosome` stops the sampler and then
    // joins it, so a wait that ignores `stop` until it has elapsed puts a
    // whole sample interval under every contig's shutdown. Measured at
    // 5.3-5.5s per contig against 1.0-1.8s of real work before the fix.
    #[test]
    fn an_already_stopped_wait_returns_without_burning_the_interval() {
        let stop = AtomicBool::new(true);
        let t0 = Instant::now();
        assert!(sleep_until_tick_or_stop(Duration::from_secs(30), &stop));
        assert!(
            t0.elapsed() < Duration::from_secs(1),
            "waited {:?} on an already-stopped sampler",
            t0.elapsed()
        );
    }

    #[test]
    fn a_wait_stopped_midway_wakes_within_one_poll_slice() {
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let setter = std::sync::Arc::clone(&stop);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            setter.store(true, Ordering::Relaxed);
        });
        let t0 = Instant::now();
        assert!(sleep_until_tick_or_stop(Duration::from_secs(30), &stop));
        // 100ms to the store, plus at most one poll slice to notice it, plus
        // slack for a loaded CI box. The point is "not 30 seconds".
        assert!(
            t0.elapsed() < Duration::from_millis(100) + STOP_POLL * 20,
            "took {:?} to notice the stop flag",
            t0.elapsed()
        );
    }

    // Cadence must not change: the emitted CPU% columns divide by the measured
    // gap, and a wait that returned early when nothing was stopping would
    // silently shorten every sampled span.
    #[test]
    fn an_uninterrupted_wait_still_lasts_the_full_interval() {
        let stop = AtomicBool::new(false);
        let interval = STOP_POLL * 4;
        let t0 = Instant::now();
        assert!(!sleep_until_tick_or_stop(interval, &stop));
        assert!(
            t0.elapsed() >= interval,
            "returned after {:?}, short of the {:?} interval",
            t0.elapsed(),
            interval
        );
    }

    #[test]
    fn gauge_records_len_highwater_not_current() {
        let g = PendingGauge::default();
        g.observe(3, 300);
        g.observe(7, 700);
        g.observe(1, 100);
        assert_eq!(g.len_highwater.load(Ordering::Relaxed), 7);
        assert_eq!(g.bytes_highwater.load(Ordering::Relaxed), 700);
    }

    #[test]
    fn gauge_starts_at_zero() {
        let g = PendingGauge::default();
        assert_eq!(g.len_highwater.load(Ordering::Relaxed), 0);
        assert_eq!(g.bytes_highwater.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn gauge_tracks_bytes_independently_of_len() {
        // A single very large chunk must raise the byte high-water even though
        // the length high-water is already higher from an earlier tick.
        let g = PendingGauge::default();
        g.observe(9, 90);
        g.observe(1, 5_000);
        assert_eq!(g.len_highwater.load(Ordering::Relaxed), 9);
        assert_eq!(g.bytes_highwater.load(Ordering::Relaxed), 5_000);
    }
}

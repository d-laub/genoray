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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::types::{DenseChunk, SparseChunk};

const CLK_TCK_HZ: f64 = 100.0;

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

pub fn spawn_sampler(
    chrom: String,
    tx_dense: Sender<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    tx_long: Sender<Vec<u8>>,
    stop: Arc<AtomicBool>,
    // Per-chrom registry of `shard-worker-*` OS TIDs, populated by
    // `shard_exec::run` (only the sharded VCF/PGEN branches use it -- stays
    // empty, and the printed `shard` column stays `n/a`, for the single-reader
    // fallback path). NOT resolved by matching the `shard-worker-{i}` thread
    // `comm` name: worker names are pool-local (`shard-worker-0`, `-1`, ...),
    // not chrom-qualified, so under `concurrent_chroms > 1` (the #135 livelock
    // repro's regime) two chromosomes' pools both name a thread
    // `shard-worker-0` -- a comm lookup would resolve to whichever one
    // `/proc/self/task` iteration finds first, misattributing CPU across
    // chromosomes. See `shard_exec::run`'s `worker_tids` doc comment.
    shard_worker_tids: Arc<Mutex<Vec<i32>>>,
) -> thread::JoinHandle<()> {
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
            // Match the names assigned to the four pipeline threads below.
            let names: Vec<String> = ["read", "exec", "cw", "lw"]
                .iter()
                .map(|p| format!("{}-{}", p, chrom))
                .collect();

            // Brief settle so the four pipeline threads register their /proc/.../comm
            // entries before the first lookup. Missing TIDs are re-resolved each tick.
            std::thread::sleep(Duration::from_millis(300));
            let mut tids: Vec<Option<i32>> =
                names.iter().map(|n| find_thread_tid_by_name(n)).collect();
            let mut prev_ticks: Vec<u64> = vec![0; names.len()];
            // Per-TID previous tick count for the shard-worker aggregate below
            // (the fixed four pipeline threads use the parallel `prev_ticks`
            // Vec instead, since their TIDs are looked up by name once).
            let mut prev_shard_ticks: HashMap<i32, u64> = HashMap::new();

            // Channel capacities (bounded() guarantees Some(cap)).
            let dense_cap = tx_dense.capacity().unwrap_or(0);
            let sparse_cap = tx_sparse.capacity().unwrap_or(0);
            let long_cap = tx_long.capacity().unwrap_or(0);

            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(interval);

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
                            100.0 * dt_ticks / CLK_TCK_HZ / interval.as_secs_f64()
                        })
                    })
                    .collect();
                prev_ticks = cur;

                // De-lie `read=`: in the sharded VCF/PGEN path, `read-{chrom}`
                // (sampled above via `cpu_pcts[0]`) just blocks in
                // `thread::scope` waiting for the shard-worker pool, so it
                // reads 0% even while the pool is pegged. Aggregate CPU
                // across every registered `shard-worker-*` TID for THIS
                // chrom instead (see `shard_worker_tids`'s doc comment above)
                // and print it as its own `shard=` column -- `n/a` when the
                // registry is empty (single-reader fallback path, nothing to
                // sample).
                let shard_tids: Vec<i32> = shard_worker_tids.lock().unwrap().clone();
                let shard_pct = if shard_tids.is_empty() {
                    None
                } else {
                    let mut dt_ticks = 0f64;
                    let mut next_prev = HashMap::with_capacity(shard_tids.len());
                    for tid in &shard_tids {
                        let cur_ticks = read_thread_cpu_ticks(*tid);
                        let prev_ticks = prev_shard_ticks.get(tid).copied().unwrap_or(0);
                        dt_ticks += cur_ticks.saturating_sub(prev_ticks) as f64;
                        next_prev.insert(*tid, cur_ticks);
                    }
                    prev_shard_ticks = next_prev;
                    Some(100.0 * dt_ticks / CLK_TCK_HZ / interval.as_secs_f64())
                };

                let fmt =
                    |o: Option<f64>| o.map_or_else(|| "n/a".to_string(), |v| format!("{:.0}%", v));
                let elapsed = start.elapsed().as_secs();
                tracing::trace!(
                    target: "genoray::monitor",
                    chrom = %chrom,
                    elapsed_s = elapsed,
                    dense = tx_dense.len(), dense_cap = dense_cap,
                    sparse = tx_sparse.len(), sparse_cap = sparse_cap,
                    long = tx_long.len(), long_cap = long_cap,
                    // `cpu_read` reads 0% in the sharded path (the reader thread
                    // just blocks on the shard-worker pool); `cpu_shard` is the
                    // de-lied aggregate CPU across this chrom's shard workers.
                    cpu_read = %fmt(cpu_pcts[0]),
                    cpu_shard = %fmt(shard_pct),
                    cpu_exec = %fmt(cpu_pcts[1]),
                    cpu_cw = %fmt(cpu_pcts[2]),
                    cpu_lw = %fmt(cpu_pcts[3]),
                    "pipeline sampler"
                );
            }
            // tx_dense, tx_sparse, tx_long Sender clones drop here as the closure ends —
            // letting the executor / writer rx ends close once the original Senders also drop.
        })
        .expect("spawn sampler")
}

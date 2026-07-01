// src/lib.rs
use crossbeam_channel::{Sender, bounded};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub mod executor;
pub mod merge;
pub mod nrvk;
pub mod rvk;
pub mod types;
pub mod vcf_reader;
pub mod writer;

use nrvk::LongAlleleTableWriter;
use vcf_reader::VcfChunkReader;

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

fn spawn_sampler(
    chrom: String,
    tx_dense: Sender<crate::types::DenseChunk>,
    tx_sparse: Sender<crate::types::SparseChunk>,
    tx_long: Sender<Vec<u8>>,
    stop: Arc<AtomicBool>,
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
                let cpu_pcts: Vec<f64> = cur
                    .iter()
                    .zip(prev_ticks.iter())
                    .map(|(c, p)| {
                        let dt_ticks = c.saturating_sub(*p) as f64;
                        100.0 * dt_ticks / CLK_TCK_HZ / interval.as_secs_f64()
                    })
                    .collect();
                prev_ticks = cur;

                let elapsed = start.elapsed().as_secs();
                eprintln!(
                    "[{} t={}s] tx_dense={}/{} tx_sparse={}/{} tx_long={}/{} | \
                     cpu read={:.0}% exec={:.0}% cw={:.0}% lw={:.0}%",
                    chrom,
                    elapsed,
                    tx_dense.len(),
                    dense_cap,
                    tx_sparse.len(),
                    sparse_cap,
                    tx_long.len(),
                    long_cap,
                    cpu_pcts[0],
                    cpu_pcts[1],
                    cpu_pcts[2],
                    cpu_pcts[3],
                );
            }
            // tx_dense, tx_sparse, tx_long Sender clones drop here as the closure ends —
            // letting the executor / writer rx ends close once the original Senders also drop.
        })
        .expect("spawn sampler")
}

/*
ARCHITECTURE & TENSOR LAYOUT LIFECYCLE

This pipeline converts sequential VCF rows into Sample-Major Sparse Tensors via a 3-Stage
parallel memory architecture.

1. Stage 1: Reader -> DenseChunk
    - Layout: (V, S, P) -> (Variants, Samples, Ploidy)
    - The VCF is read horizontally row-by-row. Data is packed into a dense 3D boolean grid.

2. Stage 2: RVK/Executor -> SparseChunk
    - Layout: (S, P, ~V) -> (Samples, Ploidy, sparse Variants)
    - The dense grid is transposed and compressed. For each sample/ploidy, we only store
     the variants (~V) that are actually mutated.
    * NOTE: Rearranging (v, s, p) -> (s, p, ~v) in these tiny temporary chunks does not
     offer an obvious IO benefit during the final merge. However, we do it here because
     it provides a massive CPU cache-locality advantage during the SIMD encoding phase,
     and it allows the Phase 3 Tile-Merger to confidently copy continuous blocks of memory
     using `copy_from_slice` rather than picking single scattered elements.

3. Stage 3: Merge Phase -> Final Tensors
    - Layout: (S, P, ~V) -> Monolithic 1D Array
    - N-many temporary SparseChunks are interleaved. All Chunk 0 -> Chunk N data for
     Sample 0 is stitched together in RAM, then Sample 1, etc., achieving the final
     read-optimized layout for the PyTorch dataloader.

*/

//The rust pipeline (Per chromosome conversion from Dense to Sparse)
#[allow(clippy::too_many_arguments)]
pub fn process_chromosome(
    vcf_path: &str,
    chrom: &str,
    base_out_dir: &str,
    samples: &[&str],
    chunk_size: usize,
    ploidy: usize,
    htslib_threads: usize,
    long_allele_capacity: usize,
) {
    // Directory Formatting: svar2/{contig}/var_key/{snp,indel}
    let var_key_dir = format!("{}/{}/var_key", base_out_dir, chrom);
    let snp_dir = format!("{}/snp", var_key_dir);
    let indel_dir = format!("{}/indel", var_key_dir);
    fs::create_dir_all(&snp_dir).expect("Failed to create snp output directory");
    fs::create_dir_all(&indel_dir).expect("Failed to create indel output directory");

    // Channel capacities tuned for cohort-scale workloads.
    // - tx_dense=6: smooths HTSlib BGZF block-boundary jitter so the executor
    //   never starves on `rx_dense.recv()`. Each DenseChunk is ~chunk_size × S × P / 8 bytes.
    // - tx_sparse=8: SparseChunks are tiny (~hundreds of KB); deeper queue is free.
    // - tx_long=2: each buffer is up to long_allele_capacity bytes — keep small.
    let (tx_dense, rx_dense) = bounded::<crate::types::DenseChunk>(6);
    let (tx_sparse, rx_sparse) = bounded::<crate::types::SparseChunk>(8);
    let (tx_long, rx_long) = bounded::<Vec<u8>>(2);

    // Periodic monitoring sampler. Owns Sender clones for read-only len()/capacity()
    // introspection. The clones drop when the sampler joins, allowing the executor's
    // rx_dense.recv() to see channel-close once the reader's Sender also drops.
    let stop_sampler = Arc::new(AtomicBool::new(false));
    let sampler_thread = spawn_sampler(
        chrom.to_string(),
        tx_dense.clone(),
        tx_sparse.clone(),
        tx_long.clone(),
        stop_sampler.clone(),
    );

    // Step 1 -> The Producer
    let reader_thread = thread::Builder::new()
        .name(format!("read-{}", chrom))
        .spawn({
            let vcf = vcf_path.to_string();
            let chr = chrom.to_string();
            // Convert references into owned Strings that can safely live forever in the thread
            let s_owned: Vec<String> = samples.iter().map(|&s| s.to_string()).collect();

            move || {
                // passing the thread budget down to HTSLib
                let s_refs: Vec<&str> = s_owned.iter().map(|s| s.as_str()).collect();
                let mut reader = VcfChunkReader::new(&vcf, &chr, &s_refs, htslib_threads, ploidy);
                let mut chunk_id = 0;
                while let Some(dense_chunk) = reader.read_next_chunk(chunk_size, chunk_id) {
                    tx_dense.send(dense_chunk).unwrap();
                    chunk_id += 1;
                }
            }
        })
        .expect("spawn reader");

    // Step 2 -> The Executor
    let executor_thread = thread::Builder::new()
        .name(format!("exec-{}", chrom))
        .spawn({
            move || {
                let bank = LongAlleleTableWriter::new(tx_long, long_allele_capacity);
                executor::run_compute_engine(rx_dense, tx_sparse, bank)
            }
        })
        .expect("spawn executor");

    // Step 3a -> The chunk writer
    let chunk_writer_thread = thread::Builder::new()
        .name(format!("cw-{}", chrom))
        .spawn({
            let snp = snp_dir.clone();
            let indel = indel_dir.clone();
            move || writer::run_io_writer(rx_sparse, &snp, &indel)
        })
        .expect("spawn chunk writer");

    // Step 3b -> The long allele chunk writer
    let long_allele_writer_thread = thread::Builder::new()
        .name(format!("lw-{}", chrom))
        .spawn({
            let out_dir = indel_dir.clone();
            let chrom_label = chrom.to_string();

            move || {
                let file_path = Path::new(&out_dir).join("long_alleles.bin");
                let file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(file_path)
                    .unwrap();

                let mut disk_writer = BufWriter::with_capacity(1024 * 1024, file);

                // sleeps until the Executor throws a buffer over the channel
                while let Ok(buffer) = rx_long.recv() {
                    disk_writer
                        .write_all(&buffer)
                        .expect("Failed to write long alleles");
                }
                disk_writer.flush().unwrap();
                println!(
                    "[{}] Long Allele Writer: All buffer data safely committed.",
                    chrom_label
                );
            }
        })
        .expect("spawn long allele writer");

    // Wait for the reader to finish (drops its tx_dense Sender).
    reader_thread.join().unwrap();

    // Stop the sampler so its tx_* clones drop. Once these AND the reader's tx_dense
    // are gone, the executor's rx_dense.recv() returns Err and the executor unwinds.
    stop_sampler.store(true, Ordering::Relaxed);
    sampler_thread.join().unwrap();

    let (snp_ledger, indel_ledger, long_allele_offsets) = executor_thread.join().unwrap();
    chunk_writer_thread.join().unwrap();
    long_allele_writer_thread.join().unwrap();

    println!(
        "[{}] Phase 1 Complete. Triggering Phase 2 In-Memory Merge...",
        chrom
    );

    // Long-allele offsets belong to the indel stream.
    let offsets_array = ndarray::Array1::from_vec(long_allele_offsets);
    ndarray_npy::write_npy(
        format!("{}/long_allele_offsets.npy", indel_dir),
        &offsets_array,
    )
    .expect("Failed to write long allele offsets");

    let num_chunks = snp_ledger.len(); // == indel_ledger.len() (one row per chunk)

    // SNP stream: merge u8 keys, then bit-pack the merged stream to 2 bits/call.
    merge::merge_mini_sc::<u8>(num_chunks, samples.len(), ploidy, &snp_dir, snp_ledger);
    rvk::pack_snp_key_file(&snp_dir);

    // Indel stream: merge 32-bit keys.
    merge::merge_mini_sc::<u32>(num_chunks, samples.len(), ploidy, &indel_dir, indel_ledger);

    println!("[{}] Pipeline Execution Finished Successfully.", chrom);
}

//The Python Wrapper and resource allocator
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (vcf_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608))]
fn run_conversion_pipeline(
    py: Python,
    vcf_path: String,
    chroms: Vec<String>, // now taking a vector
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize, // default 25K variants/chunk — halves per-chunk plumbing overhead vs the old 10K
    ploidy: usize,
    max_threads: Option<usize>,  // accepts an optional integer from Python
    long_allele_capacity: usize, // default 8MB — old 100MB rarely flushed mid-run, blocking executor at finalize
) -> PyResult<()> {
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();

    py.detach(|| {
        // Step 1 -> HW discovery/override and budgeting
        let available_cores = match max_threads {
            Some(t) if t > 0 => {
                println!("Notice: Using user-provided thread limit: {}", t);
                t
            }
            _ => {
                let detected = std::thread::available_parallelism().unwrap().get();
                println!(
                    "Notice: No thread limit provided. Hardware Detected: {} cores.",
                    detected
                );
                detected
            }
        };

        // 4 fixed OS threads per chrom: reader + executor + chunk_writer + long_allele_writer.
        // Mostly I/O-bound (blocked on channels) but they still count against the scheduler.
        const PIPELINE_THREADS_PER_CHROM: usize = 4;
        // Floor for HTSlib decode threads — below this the executor channel starves
        // (decompression is the measured bottleneck, not the encode/transpose).
        const MIN_HTSLIB_THREADS: usize = 2;
        // Ceiling for HTSlib decode threads — diminishing returns past 4 due to BGZF
        // block-level parallelism limits. More threads = contention, not throughput.
        const MAX_HTSLIB_THREADS: usize = 4;
        // Min viable allocation for one chrom end-to-end. Below this we accept thrashing.
        const MIN_THREADS_PER_CHROM: usize = PIPELINE_THREADS_PER_CHROM + MIN_HTSLIB_THREADS;

        // Reserve 1 core for the OS + Python main thread so the box doesn't freeze.
        let usable_cores = std::cmp::max(1, available_cores.saturating_sub(1));
        let n_chroms = std::cmp::max(1, chroms.len());

        let (concurrent_chroms, htslib_threads) = if usable_cores < MIN_THREADS_PER_CHROM {
            // Low-end machine: can't satisfy the minimum. Run one chrom and pour every
            // remaining core into HTSlib decompression. Pipeline threads will thrash with
            // decode threads — fine, they spend most of their time blocked on channel recv().
            let htslib = std::cmp::max(1, usable_cores.saturating_sub(PIPELINE_THREADS_PER_CHROM));
            let htslib = std::cmp::min(htslib, MAX_HTSLIB_THREADS);
            (1usize, htslib)
        } else {
            // High-end path: decide concurrency FIRST, capped by chromosome count,
            // then redistribute every remaining core into HTSlib up to MAX_HTSLIB_THREADS.

            // Pass 1: how many chroms can we fan out to in principle?
            let max_concurrent_by_cores = usable_cores / MIN_THREADS_PER_CHROM;
            // Cap by actual chromosome count — no point reserving budget for chroms that don't exist.
            let concurrent = std::cmp::max(1, std::cmp::min(max_concurrent_by_cores, n_chroms));

            // Pass 2: divvy up the actual core budget across the actual concurrent count.
            let cores_per_chrom = usable_cores / concurrent;
            let htslib_unclamped = cores_per_chrom.saturating_sub(PIPELINE_THREADS_PER_CHROM);
            // Clamp into [MIN_HTSLIB_THREADS, MAX_HTSLIB_THREADS]. Surplus cores past
            // MAX_HTSLIB_THREADS are intentionally left idle — adding them hurts due to
            // BGZF contention. Better to leave OS scheduler some slack than oversubscribe.
            let htslib = htslib_unclamped.clamp(MIN_HTSLIB_THREADS, MAX_HTSLIB_THREADS);

            (concurrent, htslib)
        };

        let total_active = concurrent_chroms * (PIPELINE_THREADS_PER_CHROM + htslib_threads);
        println!("Using: {} cores.", available_cores);
        println!(
            "Pipeline Config: {} concurrent chromosomes | {} HTSlib decompression threads each \
             ({} total active, {} reserved for OS/idle).",
            concurrent_chroms,
            htslib_threads,
            total_active,
            available_cores.saturating_sub(total_active),
        );

        // Step 2 -> Rayon Pool
        // Rayon hosts one task per concurrent chrom; each task spawns its own pipeline
        // OS threads (reader, executor, writers) plus htslib_threads HTSlib decode threads.
        // Naming the rayon workers makes them grep-able in `top -H` / pidstat alongside
        // the per-chrom threads (read-chr1, exec-chr1, etc.).
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrent_chroms)
            .thread_name(|i| format!("rayon-{}", i))
            .build()
            .unwrap();

        // Step 3 -> Dispatch
        pool.install(|| {
            chroms.par_iter().for_each(|chrom| {
                println!("==> Processing {}", chrom);
                process_chromosome(
                    &vcf_path,
                    chrom,
                    &output_dir,
                    &sample_refs,
                    chunk_size,
                    ploidy,
                    htslib_threads,
                    long_allele_capacity,
                );
            });
        });

        println!("Cohort Processing Complete.");
    });

    Ok(())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_conversion_pipeline, m)?)?;
    Ok(())
}

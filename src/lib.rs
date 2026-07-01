// src/lib.rs
use crossbeam_channel::bounded;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

pub mod budget;
pub mod executor;
pub mod layout;
pub mod merge;
pub mod monitor;
pub mod nrvk;
pub mod rvk;
pub mod types;
pub mod vcf_reader;
pub mod writer;

use nrvk::LongAlleleTableWriter;
use vcf_reader::VcfChunkReader;

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
    let paths = crate::layout::ContigPaths::new(base_out_dir, chrom);
    let snp_dir = paths.var_key_snp_dir().to_str().unwrap().to_string();
    let indel_dir = paths.var_key_indel_dir().to_str().unwrap().to_string();
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
    let sampler_thread = monitor::spawn_sampler(
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
            let file_path = paths.long_alleles_bin();
            let chrom_label = chrom.to_string();

            move || {
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
    ndarray_npy::write_npy(paths.long_allele_offsets(), &offsets_array)
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

        let plan = crate::budget::plan_thread_budget(available_cores, chroms.len());
        let concurrent_chroms = plan.concurrent_chroms;
        let htslib_threads = plan.htslib_threads;

        let total_active =
            concurrent_chroms * (crate::budget::PIPELINE_THREADS_PER_CHROM + htslib_threads);
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

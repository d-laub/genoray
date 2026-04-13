// src/lib.rs
use pyo3::prelude::*;
use rayon::prelude::*;
use crossbeam_channel::bounded;
use std::thread;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

mod types;
mod vcf_reader;
mod rvk;
mod nrvk;
mod executor;
mod writer;
mod merge;

use vcf_reader::VcfChunkReader;
use nrvk::LongAlleleTableWriter;

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
pub fn process_chromosome(
    vcf_path: &str,
    chrom: &str,
    base_out_dir: &str,
    samples: &[&str],
    chunk_size: usize,
    ploidy: usize,
    htslib_threads: usize,
    long_allele_capacity: usize, 
)   {
        // Directory Formatting: svar2/{contig}/var_key
        let chrom_out_dir = format!("{}/{}/var_key", base_out_dir, chrom);
        fs::create_dir_all(&chrom_out_dir).expect("Failed to create chromosome output directory");

        // the buffer channels
        let (tx_dense, rx_dense) = bounded::<crate::types::DenseChunk>(3);  // Raw VCF chunks
        let (tx_sparse, rx_sparse) = bounded::<crate::types::SparseChunk>(3); // Transposed Payload chunks
        let (tx_long, rx_long) = bounded::<Vec<u8>>(2);     // 128MB Long Allele Buffers

        // Step 1 -> The Producer
        let reader_thread = thread::spawn({
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
        });

        // Step 2 -> The Executor
        let executor_thread = thread::spawn({
            let num_samples = samples.len();
            
            move || {
                let bank = LongAlleleTableWriter::new(tx_long, long_allele_capacity);
                executor::run_compute_engine(rx_dense, tx_sparse, bank)
            }
        });

        // Step 3a -> The chunk writer
        let chunk_writer_thread = thread::spawn({
            let out_dir = chrom_out_dir.clone();
            move || writer::run_io_writer(rx_sparse, &out_dir)
        });

        // Step 3b -> The long allele chunk writer
        let long_allele_writer_thread = thread::spawn({
            let out_dir = chrom_out_dir.clone();
            
            move || {
                let file_path = Path::new(&out_dir).join("long_alleles.bin"); //file name hardcoded for now
                let file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(file_path)
                    .unwrap();

                let mut disk_writer = BufWriter::with_capacity(1024 * 1024, file);

                // sleeps until the Executor throws a buffer over the channel
                while let Ok(buffer) = rx_long.recv() {
                    disk_writer.write_all(&buffer).expect("Failed to write long alleles");
                }
                disk_writer.flush().unwrap();
                    // 'buffer' drops out of scope freeing buffer mem
                println!("Long Allele Writer: All buffer data safely committed.");
            }
        });

        // wait for all stages to spin down
        reader_thread.join().unwrap();
        let (ram_ledger, long_allele_offsets) = executor_thread.join().unwrap();
        
        // The Executor dropping its senders will cause these threads to break their while loops
        chunk_writer_thread.join().unwrap();
        long_allele_writer_thread.join().unwrap();

        println!("Phase 1 Complete. Triggering Phase 2 In-Memory Merge...");

        // Save the Long Allele offsets to disk
        let offsets_array = ndarray::Array1::from_vec(long_allele_offsets);
        ndarray_npy::write_npy(
            format!("{}/long_allele_offsets.npy", base_out_dir), 
            &offsets_array
        ).expect("Failed to write long allele offsets");

        // trigger the K-Way Tile Merge
        let num_chunks = ram_ledger.len();
        merge::merge_mini_sc(num_chunks, samples.len(), ploidy, &base_out_dir, ram_ledger);

        println!("Pipeline Execution Finished Successfully.");
    }

//The Python Wrapper and resource allocator
#[pyfunction]
#[pyo3(signature = (vcf_path, chroms, output_dir, samples, chunk_size=10_000, ploidy=2, max_threads=None, long_allele_capacity=104_857_600))]
fn run_conversion_pipeline(
    py: Python,
    vcf_path: String,
    chroms: Vec<String>, // now taking a vector
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize,
    ploidy: usize,
    max_threads: Option<usize>, // accepts an optional integer from Python
    long_allele_capacity: usize, // default set as 100MB -> pass as bytes
) -> PyResult<()> {

    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();

    py.allow_threads(|| {
        // Step 1 -> HW discovery/override and budgeting
        let available_cores = match max_threads {
            Some(t) if t > 0 => {
                println!("Notice: Using user-provided thread limit: {}", t);
                t
            },
            _ => {
                let detected = std::thread::available_parallelism().unwrap().get();
                println!("Notice: No thread limit provided. Hardware Detected: {} cores.", detected);
                detected
            }
        };

        // leaving 1 core for the OS and the Python main thread to prevent freezing
        let usable_cores = std::cmp::max(1, available_cores.saturating_sub(1));

        let concurrent_chroms: usize;
        let htslib_threads: usize;

        if usable_cores <= 3 {
            // Low-end machine: 1 chrom at a time. Put all remaining cores into HTSlib decompression.
            concurrent_chroms = 1;
            htslib_threads = std::cmp::max(1, usable_cores - 1); 
        } else {
            // High-end machine: 4-core pipeline per chromosome (3 HTSlib + 1 Compute)
            concurrent_chroms = std::cmp::max(1, usable_cores / 4);
            htslib_threads = 3;
        }

        println!("Hardware Detected: {} cores.", available_cores);
        println!("Pipeline Config: {} concurrent chromosomes | {} HTSlib decompression threads each.", concurrent_chroms, htslib_threads);

        // Step 2 -> Rayon Pool 
        // limit Rayon to "concurrent_chroms" threads to prevent OS scheduling thrash.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrent_chroms)
            .build()
            .unwrap();

        // Step 3 -> Dispatch
        pool.install(|| {
            chroms.par_iter().for_each(|chrom| {
                println!("==> Processing {}", chrom);
                process_chromosome(&vcf_path, chrom, &output_dir, &sample_refs, chunk_size, ploidy, htslib_threads, long_allele_capacity);
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
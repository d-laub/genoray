// src/lib.rs
use pyo3::prelude::*;
use crossbeam_channel::bounded;
use std::thread;
use std::fs::OpenOptions;
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
use nrvk::LongAlleleTable;

#[pyfunction]
#[pyo3(signature = (vcf_path, chrom, output_dir, samples, chunk_size=10_000, ploidy=2))]
fn run_conversion_pipeline(
    py: Python,
    vcf_path: String,
    chrom: String,
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize,
    ploidy: usize,
) -> PyResult<()> {

    py.allow_threads(|| {
        println!("Starting the conversion Pipeline...");

        // the buffer channels
        let (tx_dense, rx_dense) = bounded(3);  // Raw VCF chunks
        let (tx_sparse, rx_sparse) = bounded(3); // Transposed Payload chunks
        let (tx_long, rx_long) = bounded(2);     // 128MB Long Allele Buffers

        let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();

        // Step 1 -> The Producer
        let reader_thread = thread::spawn({
            let vcf_path = vcf_path.clone();
            let chrom = chrom.clone();
            let sample_refs = sample_refs.clone();
            
            move || {
                let mut reader = VcfChunkReader::new(&vcf_path, &chrom, &sample_refs);
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
                // We pass the long allele sender directly into the bank
                let bank = LongAlleleTable::new(tx_long);
                executor::run_compute_engine(rx_dense, tx_sparse, bank, num_samples, ploidy)
            }
        });

        // Step 3a -> The chunk writer
        let chunk_writer_thread = thread::spawn({
            let out_dir = output_dir.clone();
            move || {
                writer::run_io_writer(rx_sparse, &out_dir);
            }
        });

        // Step 3b -> The long allele chunk writer
        let long_allele_writer_thread = thread::spawn({
            let out_dir = output_dir.clone();
            
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
                    disk_writer.flush().unwrap();
                    // 'buffer' drops out of scope freeing buffer mem
                }
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
            format!("{}/long_allele_offsets.npy", output_dir), 
            &offsets_array
        ).expect("Failed to write long allele offsets");

        // trigger the K-Way Tile Merge
        let num_chunks = ram_ledger.len();
        merge::merge_mini_sc(num_chunks, samples.len(), ploidy, &output_dir, ram_ledger);

        println!("Pipeline Execution Finished Successfully.");
    });

    Ok(())
}

#[pymodule]
fn genvarformer_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_conversion_pipeline, m)?)?;
    Ok(())
}
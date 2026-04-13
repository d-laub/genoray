use crossbeam_channel::{Receiver, Sender};
use crate::types::{DenseChunk, SparseChunk};
use crate::nrvk::LongAlleleTableWriter;
use crate::rvk::dense2sparse_vk;


// pulls raw chunks, does the math, manages the Bank, 
// and streams the payloads to the Writer.
pub fn run_compute_engine(
    rx_dense: Receiver<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    mut bank: LongAlleleTableWriter,
) -> (Vec<Vec<u32>>, Vec<u64>) {
    
    // The ram ledge -> Stores the lengths array for every single chunk.
    let mut ram_ledger: Vec<Vec<u32>> = Vec::with_capacity(10_000); 

    // executor blocks here until a Producer thread pushes a chunk.
    // when the Producers finish the VCF and drop their senders, this loop safely breaks.
    while let Ok(chunk) = rx_dense.recv() {
        
        // hand the chunk and our exclusive mutable reference of the bank to the math engine.
        // Zero Mutex locking occurs here.
        let sparse_chunk = dense2sparse_vk(&chunk, &mut bank);

        // copy just the tiny array of u32 counts into our Phase 2 tracking ledger.
        ram_ledger.push(sparse_chunk.sample_lengths.clone());

        // push the heavy memory vectors (pos and keys) to the Writer thread.
        // If the Writer is currently blasting data to the NVMe drive, this channel 
        // will fill up, automatically pausing the Executor and applying backpressure.
        tx_sparse.send(sparse_chunk).expect("Failed to send SparseChunk to Writer");
    }

    // reach this point when the entire chromosome is finished.
    println!("Executor: VCF fully processed. Flushing remaining long alleles...");
    
    // Force the bank to write any remaining bytes in its 128MB buffer to disk,
    // and extract the offsets tensor.
    let long_allele_offsets: Vec<u64> = bank.finalize();

    // Return the Phase 2 metadata to the orchestrator
    (ram_ledger, long_allele_offsets)
}
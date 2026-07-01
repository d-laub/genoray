use crate::nrvk::LongAlleleTableWriter;
use crate::rvk::dense2sparse_vk;
use crate::types::{DenseChunk, SparseChunk};
use crossbeam_channel::{Receiver, Sender};

// Pulls raw chunks, encodes/splits, manages the bank, streams to the writer.
// Returns (snp_ledger, indel_ledger, long_allele_offsets) — one ledger per
// sub-stream, each row a chunk's per-column call counts.
pub fn run_compute_engine(
    rx_dense: Receiver<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    mut bank: LongAlleleTableWriter,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<u64>) {
    let mut snp_ledger: Vec<Vec<u32>> = Vec::with_capacity(10_000);
    let mut indel_ledger: Vec<Vec<u32>> = Vec::with_capacity(10_000);

    while let Ok(chunk) = rx_dense.recv() {
        let sparse_chunk = dense2sparse_vk(&chunk, &mut bank);

        snp_ledger.push(sparse_chunk.snp.sample_lengths.clone());
        indel_ledger.push(sparse_chunk.indel.sample_lengths.clone());

        tx_sparse
            .send(sparse_chunk)
            .expect("Failed to send SparseChunk to Writer");
    }

    println!("Executor: VCF fully processed. Flushing remaining long alleles...");
    let long_allele_offsets: Vec<u64> = bank.finalize();

    (snp_ledger, indel_ledger, long_allele_offsets)
}

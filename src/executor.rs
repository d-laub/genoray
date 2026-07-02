use crate::nrvk::LongAlleleTableWriter;
use crate::rvk::dense2sparse_vk;
use crate::streams::StreamMap;
use crate::types::{DenseChunk, SparseChunk};
use crossbeam_channel::{Receiver, Sender};

// Pulls raw chunks, encodes/splits, manages the bank, streams to the writer.
// Returns (ledgers, long_allele_offsets) — one ledger per active stream tag,
// each row a chunk's per-column call counts.
pub fn run_compute_engine(
    rx_dense: Receiver<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    mut bank: LongAlleleTableWriter,
) -> (StreamMap<Vec<Vec<u32>>>, Vec<u64>) {
    let mut ledgers: StreamMap<Vec<Vec<u32>>> = StreamMap::from_fn(|_| Vec::with_capacity(10_000));

    while let Ok(chunk) = rx_dense.recv() {
        let sparse_chunk = dense2sparse_vk(&chunk, &mut bank);

        for (tag, sub) in sparse_chunk.streams.iter() {
            ledgers.get_mut(tag).push(sub.sample_lengths.clone());
        }

        tx_sparse
            .send(sparse_chunk)
            .expect("Failed to send SparseChunk to Writer");
    }

    println!("Executor: VCF fully processed. Flushing remaining long alleles...");
    let long_allele_offsets: Vec<u64> = bank.finalize();

    (ledgers, long_allele_offsets)
}

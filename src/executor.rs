use crate::dense::DenseMap;
use crate::nrvk::LongAlleleTableWriter;
use crate::rvk::dense2sparse_vk;
use crate::streams::StreamMap;
use crate::trace::trace_ll;
use crate::types::{DenseChunk, SparseChunk};
use crossbeam_channel::{Receiver, Sender};

/// Phase-1 outputs consumed by the merge stage.
pub struct Phase1Output {
    /// One row per chunk of per-column call counts, per var_key stream.
    pub var_key_ledgers: StreamMap<Vec<Vec<u32>>>,
    /// One scalar per chunk (n_dense_variants), per dense class. Rectangular:
    /// every hap contributes the same count, so no per-column matrix.
    pub dense_ledgers: DenseMap<Vec<u32>>,
    pub long_allele_offsets: Vec<u64>,
}

// Pulls raw chunks, encodes/splits, manages the bank, streams to the writer.
// Returns Phase1Output — a ledger per active stream tag (each row a chunk's
// per-column call counts), a scalar ledger per dense class, and the
// long-allele bank offsets.
pub fn run_compute_engine(
    chrom: &str,
    rx_dense: Receiver<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    mut bank: LongAlleleTableWriter,
    sidecar_bits_enabled: bool,
    fields: &[crate::field::FieldSpec],
) -> Phase1Output {
    let mut var_key_ledgers: StreamMap<Vec<Vec<u32>>> =
        StreamMap::from_fn(|_| Vec::with_capacity(10_000));
    let mut dense_ledgers: DenseMap<Vec<u32>> = DenseMap::from_fn(|_| Vec::with_capacity(10_000));

    while let Ok(chunk) = rx_dense.recv() {
        let chunk_id = chunk.chunk_id;
        trace_ll!("[trace {chrom}] exec: dense2sparse enter chunk {chunk_id}");
        let sparse_chunk = dense2sparse_vk(&chunk, &mut bank, sidecar_bits_enabled, fields);
        trace_ll!("[trace {chrom}] exec: dense2sparse exit chunk {chunk_id}");

        for (tag, sub) in sparse_chunk.streams.iter() {
            var_key_ledgers
                .get_mut(tag)
                .push(sub.sample_lengths.clone());
        }
        for (class, sub) in sparse_chunk.dense.iter() {
            dense_ledgers
                .get_mut(class)
                .push(sub.n_dense_variants as u32);
        }

        tx_sparse
            .send(sparse_chunk)
            .expect("Failed to send SparseChunk to Writer");
        trace_ll!("[trace {chrom}] exec: sent SparseChunk {chunk_id}");
    }

    println!("Executor: VCF fully processed. Flushing remaining long alleles...");
    let long_allele_offsets: Vec<u64> = bank.finalize();

    Phase1Output {
        var_key_ledgers,
        dense_ledgers,
        long_allele_offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DenseClass;
    use crate::streams::StreamTag;
    use crate::types::BitGrid3;
    use crate::types::DenseChunk;
    use crossbeam_channel::bounded;

    fn one_snp_chunk() -> DenseChunk {
        // 1 variant, 1 sample, 2 ploidy, both haps carry it (SNP A→C).
        let mut genos = BitGrid3::zeros(1, 1, 2);
        genos.or_bit(0, true);
        genos.or_bit(1, true);
        DenseChunk {
            chunk_id: 0,
            pos: vec![100],
            ilens: vec![0],
            alt: b"C".to_vec(),
            alt_offsets: vec![0, 1],
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
            carriers: None,
            format_by_carrier: None,
        }
    }

    #[test]
    fn test_phase1_output_shapes() {
        let (tx_d, rx_d) = bounded(4);
        let (tx_s, rx_s) = bounded(4);
        let (tx_l, _rx_l) = bounded(4);
        tx_d.send(one_snp_chunk()).unwrap();
        drop(tx_d);

        let bank = crate::nrvk::LongAlleleTableWriter::new(tx_l, 1 << 16);
        let out = run_compute_engine("chrTest", rx_d, tx_s, bank, false, &[]);

        // one chunk processed → one ledger row per stream and per dense class
        assert_eq!(out.var_key_ledgers.get(StreamTag::VarKeySnp).len(), 1);
        assert_eq!(out.dense_ledgers.get(DenseClass::Snp).len(), 1);
        assert_eq!(out.dense_ledgers.get(DenseClass::Indel).len(), 1);
        // drain sparse so the channel doesn't leak
        while rx_s.recv().is_ok() {}
    }
}

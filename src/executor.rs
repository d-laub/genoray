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
    /// Total kept (emitted) variants across every `DenseChunk` this stage
    /// consumed -- `chunk.pos.len()` summed. This is the single choke point
    /// EVERY `DenseChunk` passes through regardless of source (single-reader,
    /// sharded VCF/PGEN, VcfList, Svar1), so it's the simplest place to
    /// accumulate a cohort-wide "kept" count for `EventSink::contig_done`.
    pub kept_total: u64,
}

// Pulls raw chunks, encodes/splits, manages the bank, streams to the writer.
// Returns Phase1Output — a ledger per active stream tag (each row a chunk's
// per-column call counts), a scalar ledger per dense class, and the
// long-allele bank offsets.
pub fn run_compute_engine(
    rx_dense: Receiver<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    mut bank: LongAlleleTableWriter,
    sidecar_bits_enabled: bool,
    fields: &[crate::field::FieldSpec],
    chrom: &str,
    sink: &crate::logging::EventSink,
) -> Phase1Output {
    let mut var_key_ledgers: StreamMap<Vec<Vec<u32>>> =
        StreamMap::from_fn(|_| Vec::with_capacity(10_000));
    let mut dense_ledgers: DenseMap<Vec<u32>> = DenseMap::from_fn(|_| Vec::with_capacity(10_000));
    let mut kept_total: u64 = 0;

    while let Ok(chunk) = rx_dense.recv() {
        let chunk_id = chunk.chunk_id;
        let n = chunk.pos.len() as u64;
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

        sink.tick(chrom, n);
        kept_total += n;
    }

    tracing::debug!(
        chrom = %chrom,
        "Executor: VCF fully processed. Flushing remaining long alleles..."
    );
    let long_allele_offsets: Vec<u64> = bank.finalize();

    Phase1Output {
        var_key_ledgers,
        dense_ledgers,
        long_allele_offsets,
        kept_total,
    }
}

/// Merge the per-worker `LongAlleleTableWriter` offset tables into the single
/// table `long_allele_offsets.npy` expects.
///
/// `LongAlleleTableWriter::new` seeds `alt_offsets` with a sentinel `0` ("the
/// first allele starts at byte 0"), so a bank's table is a PREFIX array and is
/// **never empty**: N workers that saw no long alleles at all still return N
/// copies of `[0]`. Concatenating them is therefore wrong even on a corpus with
/// zero long alleles -- measured as the ONLY difference between a serial and a
/// 4-worker store (136 B -> 160 B, 1 differing file out of 17, everything else
/// byte-identical).
///
/// Rebasing genuine long-allele offsets additionally needs the order in which
/// the workers' buffers landed in the shared `long_alleles.bin`, which the
/// streaming `flush_buffer` does not preserve. That is the real design work if
/// this graduates; until then this handles the cases it can prove correct and
/// refuses loudly rather than writing a plausible-looking but wrong table.
fn merge_bank_offsets(mut per_worker: Vec<Vec<u64>>) -> Vec<u64> {
    // Single worker: the bank's own table is already correct, long alleles or
    // not. Keeps `GENORAY_EXEC_WORKERS=1` byte-identical to the serial loop.
    if per_worker.len() == 1 {
        return per_worker.pop().expect("length checked");
    }
    // No worker saw a long allele (every table is just its sentinel), so the
    // correct merged table is a single sentinel.
    if per_worker.iter().all(|table| table.len() <= 1) {
        return vec![0];
    }
    panic!(
        "PROTOTYPE LIMITATION: long alleles present with {} executor workers. \
         Merging per-bank offset tables requires rebasing against the order the \
         workers' byte buffers landed in long_alleles.bin, which is not \
         implemented. Re-run with GENORAY_EXEC_WORKERS=1. \
         (per-worker table lengths: {:?})",
        per_worker.len(),
        per_worker.iter().map(Vec::len).collect::<Vec<_>>(),
    );
}

/// PROTOTYPE ONLY -- number of executor threads, from `GENORAY_EXEC_WORKERS`
/// (default 1, i.e. today's behaviour exactly).
///
/// This exists to measure whether parallelising the executor moves the
/// per-contig ceiling before committing to the real design. It is NOT a
/// shippable knob: see [`run_compute_engine_multi`]'s long-allele caveat.
pub fn exec_workers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("GENORAY_EXEC_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1)
    })
}

/// Everything [`run_compute_engine_multi`] needs beyond its three channels.
///
/// Grouped rather than passed positionally because the serial
/// [`run_compute_engine`] was already at 7 parameters and the parallel version
/// adds the bank-construction pair (`tx_long` is a channel, but the capacity
/// and worker count are not), which puts it over clippy's limit.
pub struct ExecutorParams<'a> {
    /// Number of executor threads; `0` is treated as `1`.
    pub workers: usize,
    /// Staging-buffer capacity for each worker's `LongAlleleTableWriter`.
    pub long_allele_capacity: usize,
    pub sidecar_bits_enabled: bool,
    pub fields: &'a [crate::field::FieldSpec],
    pub chrom: &'a str,
    pub sink: &'a crate::logging::EventSink,
}

/// PROTOTYPE: `workers` executor threads sharing one `rx_dense` (crossbeam
/// `Receiver` is MPMC, so this is work-stealing by construction).
///
/// Measured motivation: with `w >= 5` reader shards the executor pegs at
/// 98-99% CPU with the dense channel 64-78% full, capping per-contig speedup
/// at ~2.5-2.7x. `perf` puts `dense2sparse_vk` + `emit_call` at ~15-22% of all
/// self time against ~58-64% for the reader, so the reader parallelises and
/// the executor does not.
///
/// Three pieces of shared state make the serial loop order-dependent; this
/// handles two of them and deliberately does NOT handle the third:
///
/// 1. **Ledgers** -- `merge.rs` indexes `ram_ledger[chunk_id][col]`, so rows
///    must be in `chunk_id` order. Workers tag each row with its `chunk_id`
///    and the rows are sorted after the join, so ordering is restored exactly.
/// 2. **The writer** -- needs nothing: `writer::run_io_writer` keys its output
///    files by `chunk.chunk_id`, so it is already order-free.
/// 3. **The long-allele bank** -- handled only for the no-long-allele case;
///    see [`merge_bank_offsets`], which panics rather than emit a wrong table.
pub fn run_compute_engine_multi(
    rx_dense: Receiver<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    tx_long: Sender<Vec<u8>>,
    params: ExecutorParams<'_>,
) -> Phase1Output {
    let ExecutorParams {
        workers,
        long_allele_capacity,
        sidecar_bits_enabled,
        fields,
        chrom,
        sink,
    } = params;
    let workers = workers.max(1);
    type Row = (usize, StreamMap<Vec<u32>>, DenseMap<u32>);

    let mut all_rows: Vec<Row> = Vec::with_capacity(10_000);
    let mut per_worker_offsets: Vec<Vec<u64>> = Vec::with_capacity(workers);
    let mut kept_total: u64 = 0;

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let rx = rx_dense.clone();
            let tx_s = tx_sparse.clone();
            let tx_l = tx_long.clone();
            // Worker 0 keeps the bare `exec-{chrom}` name so `monitor.rs`'s
            // `cpu_exec` column still resolves a TID and the sampler stays
            // usable while measuring (it matches by thread name).
            let name = if i == 0 {
                format!("exec-{chrom}")
            } else {
                format!("exec-{chrom}-{i}")
            };
            let handle = std::thread::Builder::new()
                .name(name)
                .spawn_scoped(scope, move || {
                    let mut bank = LongAlleleTableWriter::new(tx_l, long_allele_capacity);
                    let mut rows: Vec<Row> = Vec::with_capacity(4096);
                    let mut kept: u64 = 0;
                    while let Ok(chunk) = rx.recv() {
                        let chunk_id = chunk.chunk_id;
                        let n = chunk.pos.len() as u64;
                        let sparse_chunk =
                            dense2sparse_vk(&chunk, &mut bank, sidecar_bits_enabled, fields);
                        // Snapshot the ledger row BEFORE handing the chunk to
                        // the writer -- `send` moves it.
                        let srow = StreamMap::from_fn(|tag| {
                            sparse_chunk.streams.get(tag).sample_lengths.clone()
                        });
                        let drow = DenseMap::from_fn(|class| {
                            sparse_chunk.dense.get(class).n_dense_variants as u32
                        });
                        rows.push((chunk_id, srow, drow));
                        tx_s.send(sparse_chunk)
                            .expect("Failed to send SparseChunk to Writer");
                        sink.tick(chrom, n);
                        kept += n;
                    }
                    (rows, bank.finalize(), kept)
                })
                .expect("spawn executor worker");
            handles.push(handle);
        }
        // Joined in spawn order, so `per_worker_offsets` is indexed by worker
        // and the merge below is deterministic.
        for handle in handles {
            let (rows, offsets, kept) = handle.join().expect("executor worker panicked");
            all_rows.extend(rows);
            per_worker_offsets.push(offsets);
            kept_total += kept;
        }
    });

    let long_allele_offsets = merge_bank_offsets(per_worker_offsets);

    // Restore strict chunk order: `merge.rs` indexes ledger rows by chunk_id.
    all_rows.sort_unstable_by_key(|(chunk_id, _, _)| *chunk_id);

    let n = all_rows.len();
    let mut var_key_ledgers: StreamMap<Vec<Vec<u32>>> =
        StreamMap::from_fn(|_| Vec::with_capacity(n));
    let mut dense_ledgers: DenseMap<Vec<u32>> = DenseMap::from_fn(|_| Vec::with_capacity(n));
    for (_, mut srow, drow) in all_rows {
        // `take` rather than `clone`: `sample_lengths` is one u32 per column,
        // so a second copy per chunk would be megabytes at large cohort sizes
        // and would show up in the very measurement this exists to make.
        for (tag, v) in srow.iter_mut() {
            var_key_ledgers.get_mut(tag).push(std::mem::take(v));
        }
        for (class, v) in drow.iter() {
            dense_ledgers.get_mut(class).push(*v);
        }
    }

    Phase1Output {
        var_key_ledgers,
        dense_ledgers,
        long_allele_offsets,
        kept_total,
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
            global_idx: vec![-1],
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

    // A single bank's table is already correct and must pass through verbatim,
    // long alleles included -- this is what keeps GENORAY_EXEC_WORKERS=1
    // byte-identical to the serial loop.
    #[test]
    fn merge_bank_offsets_single_worker_is_verbatim() {
        assert_eq!(merge_bank_offsets(vec![vec![0, 12, 30]]), vec![0, 12, 30]);
        assert_eq!(merge_bank_offsets(vec![vec![0]]), vec![0]);
    }

    // The bug this whole function exists for: `alt_offsets` is seeded with a
    // sentinel 0, so N workers that saw NO long alleles still return N copies
    // of `[0]`. Concatenating gave `[0,0,0,0]` and moved the store digest.
    #[test]
    fn merge_bank_offsets_collapses_sentinels() {
        assert_eq!(
            merge_bank_offsets(vec![vec![0], vec![0], vec![0], vec![0]]),
            vec![0]
        );
    }

    #[test]
    #[should_panic(expected = "PROTOTYPE LIMITATION")]
    fn merge_bank_offsets_refuses_real_long_alleles() {
        // Worker 1 banked an allele; its offsets are bank-relative and cannot
        // be rebased without knowing buffer arrival order, so this must refuse
        // rather than emit a plausible-looking wrong table.
        merge_bank_offsets(vec![vec![0], vec![0, 17]]);
    }

    #[test]
    fn test_phase1_output_shapes() {
        let (tx_d, rx_d) = bounded(4);
        let (tx_s, rx_s) = bounded(4);
        let (tx_l, _rx_l) = bounded(4);
        tx_d.send(one_snp_chunk()).unwrap();
        drop(tx_d);

        let bank = crate::nrvk::LongAlleleTableWriter::new(tx_l, 1 << 16);
        let sink = crate::logging::EventSink::disabled();
        let out = run_compute_engine(rx_d, tx_s, bank, false, &[], "chrTest", &sink);

        // one chunk processed → one ledger row per stream and per dense class
        assert_eq!(out.var_key_ledgers.get(StreamTag::VarKeySnp).len(), 1);
        assert_eq!(out.dense_ledgers.get(DenseClass::Snp).len(), 1);
        assert_eq!(out.dense_ledgers.get(DenseClass::Indel).len(), 1);
        assert_eq!(out.kept_total, 1);
        // drain sparse so the channel doesn't leak
        while rx_s.recv().is_ok() {}
    }
}

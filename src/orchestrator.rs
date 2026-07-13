// src/orchestrator.rs
use crossbeam_channel::bounded;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use crate::enum_map::EnumKey;
use crate::error::ConversionError;
use crate::nrvk::LongAlleleTableWriter;
use crate::streams::{REGISTRY, StreamMap, StreamTag};
use crate::{executor, merge, monitor, writer};

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

/// Which backend a contig's records come from. Everything downstream of
/// `ChunkAssembler` is identical for both.
pub enum SourceSpec {
    Vcf {
        vcf_path: String,
        htslib_threads: usize,
    },
    Pgen {
        pgen_path: String,
        pvar_path: String,
        var_start: usize,
        var_end: usize,
        /// A `pgenlib.PgenReader` for THIS contig. Readers seek independently, so
        /// each concurrent contig needs its own -- never share one.
        reader: pyo3::Py<pyo3::PyAny>,
    },
    Svar1 {
        svar1_dir: String,
        contig_start: usize,
        n_local: usize,
        pos: Vec<u32>,
        ref_bytes: Vec<u8>,
        ref_offsets: Vec<i64>,
        alt_bytes: Vec<u8>,
        alt_offsets: Vec<i64>,
        format_fields: Vec<crate::field::FieldSpec>,
        format_src_dtypes: Vec<String>,
    },
}

//The rust pipeline (Per chromosome conversion from Dense to Sparse)
#[allow(clippy::too_many_arguments)]
pub fn process_chromosome(
    source: SourceSpec,
    fasta_path: Option<&str>,
    chrom: &str,
    base_out_dir: &str,
    samples: &[&str],
    chunk_size: usize,
    ploidy: usize,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    processing_threads: usize,
    signatures: bool,
    fields: &[crate::field::FieldSpec],
) -> Result<u64, ConversionError> {
    // Directory Formatting: svar2/{contig}/var_key/{snp,indel}
    let paths = crate::layout::ContigPaths::new(base_out_dir, chrom);

    // Stream dirs keyed by tag, built up front (no side effects — StreamMap::from_fn
    // can't propagate a Result out of its closure). Adding a new stream means
    // extending `streams::REGISTRY` only — nothing here needs to change.
    let stream_dirs: StreamMap<std::path::PathBuf> = StreamMap::from_fn(|tag| {
        let spec = &REGISTRY[tag.index()];
        std::path::Path::new(base_out_dir)
            .join(chrom)
            .join(spec.subdir)
    });
    // Actually create the directories in a separate loop, where `?` is available.
    for (_, dir) in stream_dirs.iter() {
        fs::create_dir_all(dir).map_err(|e| ConversionError::Io {
            context: format!("create_dir_all {:?}", dir),
            source: e,
        })?;
    }
    // Shared per-contig indel LUT dir (long alleles for var_key + dense indels).
    fs::create_dir_all(paths.shared_indel_dir()).map_err(|e| ConversionError::Io {
        context: format!("create_dir_all {:?}", paths.shared_indel_dir()),
        source: e,
    })?;

    // Dense per-class dirs, built up front the same way as `stream_dirs`.
    let dense_dirs: crate::dense::DenseMap<std::path::PathBuf> =
        crate::dense::DenseMap::from_fn(|c| {
            let spec = &crate::dense::DENSE_REGISTRY[c.index()];
            std::path::Path::new(base_out_dir)
                .join(chrom)
                .join(spec.subdir)
        });
    for spec in &crate::dense::DENSE_REGISTRY {
        let dir = dense_dirs.get(spec.class);
        fs::create_dir_all(dir).map_err(|e| ConversionError::Io {
            context: format!("create_dir_all {:?}", dir),
            source: e,
        })?;
    }

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

    // Dedicated rayon pool for the reader's intra-chunk presence packing. Sized to
    // the idle cores (budget::plan_thread_budget). Built even at size 1 so the
    // reader always has a handle; parallel packing self-gates off below 2 threads.
    // NOTE (multi-contig): process_chromosome runs once per concurrent contig, and
    // each builds a pool of `processing_threads` — the *global* idle-core count — so
    // N concurrent contigs allocate N pools and can oversubscribe. This is deliberate
    // and harmless: the target is the single-contig case (concurrent == 1, exact fit),
    // and profiling (roadmap M14) shows packing is <0.05% of reader time, so the extra
    // threads sit idle. Divide by concurrent_chroms here if packing ever grows costly.
    let processing_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(processing_threads.max(1))
            .thread_name(|i| format!("pack-{}", i))
            .build()
            .expect("build processing pool"),
    );

    // Step 1 -> The Producer
    let reader_thread = thread::Builder::new()
        .name(format!("read-{}", chrom))
        .spawn({
            let fasta = fasta_path.map(|s| s.to_string());
            let chr = chrom.to_string();
            // Convert references into owned Strings that can safely live forever in the thread
            let s_owned: Vec<String> = samples.iter().map(|&s| s.to_string()).collect();
            let pool = Arc::clone(&processing_pool);
            let fields_owned: Vec<crate::field::FieldSpec> = fields.to_vec();

            move || -> Result<u64, ConversionError> {
                // passing the thread budget down to HTSLib
                let s_refs: Vec<&str> = s_owned.iter().map(|s| s.as_str()).collect();
                let src: Box<dyn crate::record_source::RecordSource + Send> = match source {
                    SourceSpec::Vcf {
                        vcf_path,
                        htslib_threads,
                    } => Box::new(crate::vcf_reader::VcfRecordSource::new(
                        &vcf_path,
                        &chr,
                        &s_refs,
                        htslib_threads,
                        ploidy,
                        &fields_owned,
                    )?),
                    SourceSpec::Pgen {
                        pgen_path: _,
                        pvar_path,
                        var_start,
                        var_end,
                        reader,
                    } => Box::new(crate::pgen_reader::PgenRecordSource::new(
                        reader,
                        &pvar_path,
                        var_start,
                        var_end,
                        s_refs.len(),
                        chunk_size,
                    )?),
                    SourceSpec::Svar1 {
                        svar1_dir,
                        contig_start,
                        n_local,
                        pos,
                        ref_bytes,
                        ref_offsets,
                        alt_bytes,
                        alt_offsets,
                        format_fields,
                        format_src_dtypes,
                    } => Box::new(crate::svar1_reader::Svar1RecordSource::new(
                        &svar1_dir,
                        contig_start,
                        n_local,
                        s_refs.len(),
                        ploidy,
                        pos,
                        ref_bytes,
                        ref_offsets,
                        alt_bytes,
                        alt_offsets,
                        &format_fields,
                        &format_src_dtypes,
                    )?),
                };
                let mut reader = crate::chunk_assembler::ChunkAssembler::new(
                    src,
                    s_refs.len(),
                    ploidy,
                    fasta.as_deref(),
                    &chr,
                    skip_out_of_scope,
                    &fields_owned,
                )?;
                let mut chunk_id = 0;
                while let Some(dense_chunk) =
                    reader.read_next_chunk(chunk_size, chunk_id, Some(&pool))?
                {
                    tx_dense.send(dense_chunk).unwrap();
                    chunk_id += 1;
                }
                Ok(reader.dropped_out_of_scope())
            }
        })
        .expect("spawn reader");

    // Step 2 -> The Executor
    let fields_exec: Vec<crate::field::FieldSpec> = fields.to_vec();
    let executor_thread = thread::Builder::new()
        .name(format!("exec-{}", chrom))
        .spawn({
            move || {
                let bank = LongAlleleTableWriter::new(tx_long, long_allele_capacity);
                executor::run_compute_engine(rx_dense, tx_sparse, bank, signatures, &fields_exec)
            }
        })
        .expect("spawn executor");

    // Step 3a -> The chunk writer
    // StreamMap isn't Clone, so build a separate owned copy (PathBuf IS Clone)
    // for the writer thread to move into its closure; `stream_dirs` itself is
    // kept for the post-Phase-1 merge loop below.
    let dirs_for_writer = StreamMap::from_fn(|tag| stream_dirs.get(tag).clone());
    let dense_dirs_for_writer = crate::dense::DenseMap::from_fn(|c| dense_dirs.get(c).clone());
    let chunk_writer_thread = thread::Builder::new()
        .name(format!("cw-{}", chrom))
        .spawn(move || writer::run_io_writer(rx_sparse, dirs_for_writer, dense_dirs_for_writer))
        .expect("spawn chunk writer");

    // Step 3b -> The long allele chunk writer
    let long_allele_writer_thread = thread::Builder::new()
        .name(format!("lw-{}", chrom))
        .spawn({
            let out_path = paths.long_alleles_bin();
            let chrom_label = chrom.to_string();
            move || crate::writer::run_long_allele_writer(rx_long, &out_path, &chrom_label)
        })
        .expect("spawn long allele writer");

    // Shutdown must be leak-free even on the error path: a detached sampler keeps
    // its tx_* channel clones alive, which would block the executor/writers on
    // recv() forever. So finish the reader, tell the sampler to stop (dropping its
    // clones so the executor can see channel-close and drain), join EVERY thread,
    // and only then surface the first panic as a WorkerPanicked error.
    let reader_res = reader_thread.join();
    stop_sampler.store(true, Ordering::Relaxed);
    let sampler_res = sampler_thread.join();
    let executor_res = executor_thread.join();
    let chunk_writer_res = chunk_writer_thread.join();
    let long_allele_writer_res = long_allele_writer_thread.join();

    let dropped = match reader_res {
        Ok(r) => r?, // ConversionError propagates with its real message
        Err(_) => {
            return Err(ConversionError::WorkerPanicked {
                thread: format!("read-{}", chrom),
            });
        }
    };
    sampler_res.map_err(|_| ConversionError::WorkerPanicked {
        thread: format!("samp-{}", chrom),
    })?;
    let phase1 = executor_res.map_err(|_| ConversionError::WorkerPanicked {
        thread: format!("exec-{}", chrom),
    })?;
    let crate::executor::Phase1Output {
        var_key_ledgers: ledgers,
        dense_ledgers,
        long_allele_offsets,
    } = phase1;
    match chunk_writer_res {
        Ok(r) => r?,
        Err(_) => {
            return Err(ConversionError::WorkerPanicked {
                thread: format!("cw-{}", chrom),
            });
        }
    }
    match long_allele_writer_res {
        Ok(r) => r?,
        Err(_) => {
            return Err(ConversionError::WorkerPanicked {
                thread: format!("lw-{}", chrom),
            });
        }
    }

    println!(
        "[{}] Phase 1 Complete. Triggering Phase 2 In-Memory Merge...",
        chrom
    );

    // Long-allele offsets belong to the indel stream.
    let offsets_array = ndarray::Array1::from_vec(long_allele_offsets);
    ndarray_npy::write_npy(paths.long_allele_offsets(), &offsets_array).map_err(|source| {
        ConversionError::Npy {
            path: paths.long_allele_offsets().to_string_lossy().into_owned(),
            source,
        }
    })?;

    // num_chunks is identical across streams — one ledger row per chunk.
    let num_chunks = ledgers.get(StreamTag::VarKeyIndel).len();
    let mut ledgers = ledgers; // make mutable to move rows out
    for spec in &REGISTRY {
        let dir = stream_dirs.get(spec.tag).clone();

        // Var_key field values are staged 1:1 with calls, so they share the
        // ledger's column-major reordering with the pos/key streams merged
        // below. Merge every field FIRST (borrowing the ledger) — merge_mini_sc
        // moves the ledger out right after, and the two merges touch disjoint
        // per-chunk files (chunk_{c}_field*.bin vs chunk_{c}_pos/key.bin), so
        // ordering between them doesn't matter for correctness.
        let sub_label = spec.subdir.replace('/', "_");
        for (field_ix, field) in fields.iter().enumerate() {
            let dest_dir = std::path::Path::new(base_out_dir)
                .join(chrom)
                .join("fields")
                .join(field.category.as_str())
                .join(&field.name)
                .join(&sub_label);
            fs::create_dir_all(&dest_dir).map_err(|e| ConversionError::Io {
                context: format!("create_dir_all {:?}", dest_dir),
                source: e,
            })?;
            let dest_values_bin = dest_dir.join("values.bin");
            merge::merge_var_key_field_values(
                dir.to_str().unwrap(),
                num_chunks,
                samples.len(),
                ploidy,
                ledgers.get(spec.tag),
                field_ix,
                4, // staged width (i32/f32); narrowed to final dtype at finalize (Task 9)
                &dest_values_bin,
            )?;
        }

        let ledger = std::mem::take(ledgers.get_mut(spec.tag));
        merge::merge_mini_sc(
            spec.key_bytes,
            num_chunks,
            samples.len(),
            ploidy,
            dir.to_str().unwrap(),
            ledger,
        )?;
        if let Some(hook) = spec.post_merge {
            hook(&dir)?;
        }
    }

    // Dense merge: one rectangular merge per dense class (no-op-safe when empty).
    let mut dense_ledgers = dense_ledgers; // make mutable to move rows out
    for spec in &crate::dense::DENSE_REGISTRY {
        let dir = std::path::Path::new(base_out_dir)
            .join(chrom)
            .join(spec.subdir);
        let ledger = std::mem::take(dense_ledgers.get_mut(spec.class));

        // Dense field values are staged 1:1 with dense variants under this
        // class, so they need only chunk-order concatenation (no ledger-driven
        // reordering). Merge every field FIRST (borrowing the ledger) — the
        // per-chunk field files (chunk_{c}_finfo{i}.bin / chunk_{c}_fformat{j}.bin)
        // are disjoint from the pos/key/geno files merge_dense_class consumes
        // right after, so ordering between the two doesn't matter for correctness.
        let sub_label = spec.subdir.replace('/', "_");
        let mut info_ix = 0usize;
        let mut format_ix = 0usize;
        for field in fields.iter() {
            let field_ix = match field.category {
                crate::field::FieldCategory::Info => {
                    let ix = info_ix;
                    info_ix += 1;
                    ix
                }
                crate::field::FieldCategory::Format => {
                    let ix = format_ix;
                    format_ix += 1;
                    ix
                }
            };
            let dest_dir = std::path::Path::new(base_out_dir)
                .join(chrom)
                .join("fields")
                .join(field.category.as_str())
                .join(&field.name)
                .join(&sub_label);
            fs::create_dir_all(&dest_dir).map_err(|e| ConversionError::Io {
                context: format!("create_dir_all {:?}", dest_dir),
                source: e,
            })?;
            let dest_values_bin = dest_dir.join("values.bin");
            crate::dense_merge::merge_dense_field_values(
                dir.to_str().unwrap(),
                num_chunks,
                &ledger,
                field.category,
                field_ix,
                &dest_values_bin,
            )?;
        }

        crate::dense_merge::merge_dense_class(
            num_chunks,
            samples.len(),
            ploidy,
            spec.key_bytes,
            spec.pack_snp,
            dir.to_str().unwrap(),
            ledger,
        )?;
    }

    // M5 post-pass: emit max-deletion-length artifacts for the overlap query.
    // A pure scan of the finished indel key streams — decoupled from the merge.
    let contig_dir = std::path::Path::new(base_out_dir).join(chrom);
    crate::max_del::write_max_del(&contig_dir, samples.len(), ploidy)?;

    // Optional M-signatures write-time annotation: classify SBS96/ID83 codes
    // and store the mutcat sidecar now, while we're already in the
    // conversion-gated write path. Requires a reference (checked in Python).
    if signatures && let Some(fasta) = fasta_path {
        let ref_seq = crate::vcf_reader::load_contig_seq(fasta, chrom)?;
        let reader = crate::query::ContigReader::open(base_out_dir, chrom, samples.len(), ploidy)
            .map_err(|e| ConversionError::Io {
            context: format!("open ContigReader for mutcat annotate {chrom}"),
            source: e,
        })?;
        crate::mutcat::annotate::annotate_contig(&reader, &paths, &ref_seq).map_err(|e| {
            ConversionError::Io {
                context: format!("annotate mutcat {chrom}"),
                source: e,
            }
        })?;
    }

    println!("[{}] Pipeline Execution Finished Successfully.", chrom);

    Ok(dropped)
}

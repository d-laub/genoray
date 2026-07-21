// src/orchestrator.rs
use crossbeam_channel::bounded;
use std::fs;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use crate::enum_map::EnumKey;
use crate::error::ConversionError;
use crate::nrvk::LongAlleleTableWriter;
use crate::streams::{REGISTRY, StreamMap, StreamTag};
use crate::trace::trace_ll;
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

/// Over-decompose VCF work units beyond the worker count so the work-stealing
/// collector (`shard_exec::run`) can absorb density skew across the contig --
/// more shards than workers means an idle worker always has something to
/// steal. 4 is a starting value to be tuned from benchmark imbalance data.
///
/// VCF-only: PGEN is intentionally NOT over-decomposed here. `readers_pool`
/// provisions exactly one `pgenlib.PgenReader` per potential shard
/// (`max_shards` caps `readers.len()`), and pgenlib holds the GIL through its
/// decode, so concurrent PGEN shard reads are GIL-serialized -- sharding adds
/// pure overhead there rather than parallelism (see the convoy fix fa47530;
/// reproducible chr21c benchmark: 44.9s sharded vs 32.6s serial -- PGEN
/// conversion is already fast and executor/IO-bound, not decode-bound, so PGEN
/// sub-contig sharding is disabled by default at the Python layer). See the
/// `SourceSpec::Pgen` branch below for the unchanged `max_shards` calc.
pub(crate) const OVERSHARD_FACTOR: usize = 4;

/// Which backend a contig's records come from. Everything downstream of
/// `ChunkAssembler` is identical for both.
pub enum SourceSpec {
    Vcf {
        vcf_path: String,
        htslib_threads: usize,
        regions: Vec<(u32, u32)>,
        overlap: crate::svar2_view::OverlapMode,
    },
    Pgen {
        pgen_path: String,
        pvar_path: String,
        var_start: usize,
        var_end: usize,
        /// A pool of `pgenlib.PgenReader`s for THIS contig, one per potential
        /// shard. Readers seek independently, so each concurrent shard needs
        /// its own -- never share one. `readers.len()` upper-bounds how many
        /// shards this contig can be split into. (Sub-contig PGEN sharding is
        /// disabled at the Python layer -- `from_pgen` pins P=1 -- so at runtime
        /// this pool holds exactly one reader and the sharded branch below is
        /// dead; kept as intact infrastructure pending a pgenlib GIL fix.)
        readers: Vec<pyo3::Py<pyo3::PyAny>>,
        /// One `pgenlib.PgenReader` per dosage `FieldSpec`, in field order --
        /// empty when no dosage fields were requested. Unlike `readers`, this
        /// is NOT a per-shard pool: PGEN sub-contig sharding is dead at
        /// runtime (see `readers`'s doc comment above), so the same flat
        /// per-field set is reused at both `PgenRecordSource::new` call sites
        /// in `process_chromosome` below.
        dosage_readers: Vec<pyo3::Py<pyo3::PyAny>>,
        regions: Vec<(u32, u32)>,
        overlap: crate::svar2_view::OverlapMode,
        /// Same permutation for every contig -- one cohort-wide sample
        /// selection, not per-contig. See `PgenRecordSource::sample_perm`.
        sample_perm: Vec<usize>,
    },
    /// `SparseVar2.from_vcf_list`: N single-sample VCFs with possibly disjoint
    /// site lists, k-way merged into ONE record stream (`VcfListRecordSource`).
    /// `vcf_paths[i]` is parallel to `process_chromosome`'s `samples` slice --
    /// column `i`'s sample is `samples[i]`, read from `vcf_paths[i]`.
    VcfList {
        vcf_paths: Vec<String>,
        /// Small per-file HTSlib decompression thread count (e.g. `1`) -- N
        /// files are open concurrently per contig, unlike the single-file
        /// `Vcf` variant.
        htslib_threads: usize,
        regions: Vec<(u32, u32)>,
        overlap: crate::svar2_view::OverlapMode,
        /// `members[i]` is `false` when `vcf_paths[i]` has no records on THIS
        /// contig (Python's per-file cyvcf2 probe). Non-members are never opened
        /// -- their column is hom-ref filled. This is per-contig, so a fresh
        /// `Vec` is built for each `process_chromosome` call (see
        /// `run_vcf_list`). See `VcfListRecordSource::new`.
        members: Vec<bool>,
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
        regions: Vec<(u32, u32)>,
        overlap: crate::svar2_view::OverlapMode,
        /// Original SVAR1 sample indices, in OUTPUT/caller order -- same
        /// permutation for every contig, one cohort-wide sample selection, not
        /// per-contig. See `Svar1RecordSource::new`'s bucket remap.
        sample_idx: Vec<usize>,
    },
}

/// Wraps `VcfListRecordSource`, mirroring its running `dropped_out_of_scope()`
/// count into a shared cell on every call. Needed because ownership of the
/// concrete source moves into `Box<dyn RecordSource + Send>` for
/// `ChunkAssembler`, so it's otherwise unreachable after construction --
/// unlike `ChunkAssembler` itself (which re-atomizes the already-atomic merged
/// stream and reports 0 additional drops), the REAL out-of-scope drops for
/// this source happen inside the per-file atomization stage during the merge,
/// so that count must be surfaced too.
struct VcfListDroppedProxy {
    inner: crate::vcf_list_reader::VcfListRecordSource,
    dropped_out: Arc<AtomicU64>,
    ref_excluded_out: Arc<AtomicU64>,
}

impl crate::record_source::RecordSource for VcfListDroppedProxy {
    fn next_record(&mut self) -> Result<Option<crate::record_source::RawRecord>, ConversionError> {
        let rec = self.inner.next_record()?;
        // `dropped_out_of_scope()` sums over every `FileCursor` (O(N) in the
        // file count) -- reading it on EVERY merged record made this an
        // O(N * records) scan on the pipeline's hot path (e.g. N=1000 files
        // and 5M merged records ⇒ 5e9 strided loads). The count only needs
        // to be correct by the time the caller (`process_chromosome`'s
        // reader thread) reads `vcf_list_dropped` after its `while let
        // Some(dense_chunk) = reader.read_next_chunk(...)` loop exits --
        // which happens only once the underlying source (and therefore this
        // proxy) has returned `None` at least once. So it's enough to
        // refresh the stored total on that one EOF transition instead of
        // every record.
        if rec.is_none() {
            self.dropped_out
                .store(self.inner.dropped_out_of_scope(), Ordering::Relaxed);
            self.ref_excluded_out
                .store(self.inner.ref_excluded(), Ordering::Relaxed);
        }
        Ok(rec)
    }
}

/// Emit the per-contig `check_ref=x` exclusion summary (nothing when none were
/// excluded). Shared by the single-reader and sub-contig-sharded paths so both
/// report the same line; the sharded path sums the count across shards first.
fn report_ref_excluded(chrom: &str, ref_excluded: u64) {
    if ref_excluded > 0 {
        tracing::info!(
            chrom = %chrom,
            excluded = ref_excluded,
            "check_ref=x: excluded records whose REF disagreed with the reference FASTA"
        );
    }
}

/// Emit the per-contig left-alignment summary (nothing when no atoms moved).
/// Shared by the single-reader and sub-contig-sharded paths, mirroring
/// `report_ref_excluded` above.
fn report_normalized(chrom: &str, normalized_total: u64) {
    if normalized_total > 0 {
        tracing::info!(chrom = %chrom, normalized = normalized_total, "left-aligned indels");
    }
}

fn with_vcf_shard_context(
    err: ConversionError,
    chrom: &str,
    shard: crate::vcf_reader::VcfShard,
) -> ConversionError {
    let label = format!(
        "VCF shard {chrom} ordinal {} ownership [{}, {}) fetch [{}, {})",
        shard.ordinal, shard.own_start, shard.own_end, shard.fetch_start, shard.fetch_end
    );
    match err {
        ConversionError::Input(msg) => ConversionError::Input(format!("{label} failed: {msg}")),
        ConversionError::ContigNotInHeader { chrom: missing } => ConversionError::Input(format!(
            "{label} failed: Chromosome '{missing}' not found in VCF header"
        )),
        ConversionError::Io { context, source } => ConversionError::Io {
            context: format!("{label} failed while {context}"),
            source,
        },
        ConversionError::MissingFile { path } => ConversionError::MissingFile {
            path: format!("{path} ({label})"),
        },
        ConversionError::WorkerPanicked { thread } => ConversionError::WorkerPanicked {
            thread: format!("{thread} ({label})"),
        },
        ConversionError::Npy { path, source } => ConversionError::Npy {
            path: format!("{path} ({label})"),
            source,
        },
        ConversionError::ReadNpy { path, source } => ConversionError::ReadNpy {
            path: format!("{path} ({label})"),
            source,
        },
    }
}

/// Generic PGEN-shard error decorator, mirroring `with_vcf_shard_context`'s
/// per-arm structure. PGEN has no VCF-style header/error strings to special-
/// case, so every arm gets the same shard-region label prefix. `unit.own_*`
/// are 0-based reference *positions* (ownership is position-based -- see
/// `chunk_assembler::ChunkAssembler::with_reference`); `unit.fetch_*` are
/// absolute `.pvar`/`.pgen` variant indices.
fn with_pgen_shard_context(
    err: ConversionError,
    chrom: &str,
    unit: &crate::shard::WorkUnit,
) -> ConversionError {
    let label = format!(
        "PGEN shard {chrom} ordinal {} owned pos [{}, {}) fetch var [{}, {})",
        unit.ordinal, unit.own_start, unit.own_end, unit.fetch_start, unit.fetch_end
    );
    match err {
        ConversionError::Input(msg) => ConversionError::Input(format!("{label} failed: {msg}")),
        ConversionError::ContigNotInHeader { chrom: missing } => {
            ConversionError::Input(format!("{label} failed: Chromosome '{missing}' not found"))
        }
        ConversionError::Io { context, source } => ConversionError::Io {
            context: format!("{label} failed while {context}"),
            source,
        },
        ConversionError::MissingFile { path } => ConversionError::MissingFile {
            path: format!("{path} ({label})"),
        },
        ConversionError::WorkerPanicked { thread } => ConversionError::WorkerPanicked {
            thread: format!("{thread} ({label})"),
        },
        ConversionError::Npy { path, source } => ConversionError::Npy {
            path: format!("{path} ({label})"),
            source,
        },
        ConversionError::ReadNpy { path, source } => ConversionError::ReadNpy {
            path: format!("{path} ({label})"),
            source,
        },
    }
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
    check_ref: crate::normalize::CheckRef,
    processing_threads: usize,
    signatures: bool,
    fields: &[crate::field::FieldSpec],
    sink: &crate::logging::EventSink,
) -> Result<u64, ConversionError> {
    let contig_started = std::time::Instant::now();
    sink.contig_start(chrom, None); // streaming: total unknown

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

    // Registry of `shard-worker-*` OS TIDs for THIS contig, populated by
    // `shard_exec::run` (sharded VCF/PGEN branches only) and read by the
    // monitor sampler below to de-lie its `read=0%` column -- see
    // `shard_exec::run`'s `worker_tids` doc comment for why a shared registry
    // is required instead of matching threads by `comm` name.
    let shard_worker_tids: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));

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
        Arc::clone(&shard_worker_tids),
    );

    // Step 1 -> The Producer
    let reader_thread = thread::Builder::new()
        .name(format!("read-{}", chrom))
        .spawn({
            let fasta = fasta_path.map(|s| s.to_string());
            let chr = chrom.to_string();
            // Convert references into owned Strings that can safely live forever in the thread
            let s_owned: Vec<String> = samples.iter().map(|&s| s.to_string()).collect();
            let fields_owned: Vec<crate::field::FieldSpec> = fields.to_vec();
            let shard_worker_tids = Arc::clone(&shard_worker_tids);

            // Returns `(dropped_out_of_scope, ref_excluded, normalized_total)` so
            // the caller can feed `EventSink::contig_done`'s `excluded` arg (and
            // the left-alignment summary log) without a second cross-thread
            // channel.
            move || -> Result<(u64, u64, u64), ConversionError> {
                // passing the thread budget down to HTSLib
                let s_refs: Vec<&str> = s_owned.iter().map(|s| s.as_str()).collect();
                // Only populated (and only meaningful) for `SourceSpec::VcfList`;
                // stays 0 for the other variants.
                let vcf_list_dropped = Arc::new(AtomicU64::new(0));
                let vcf_list_ref_excluded = Arc::new(AtomicU64::new(0));
                let src: Box<dyn crate::record_source::RecordSource + Send> = match source {
                    SourceSpec::Vcf {
                        vcf_path,
                        htslib_threads,
                        regions,
                        overlap,
                    } => {
                        // Sub-contig sharding tiles the coalesced regions into
                        // POS-ownership ranges and dedups every record by POS
                        // (`ChunkAssembler.owned_range`). That composes with the
                        // region filter ONLY when the overlap semantics ARE that
                        // POS partition -- i.e. `Pos` mode, which the whole-contig
                        // default also uses. Under `Record` (keeps `POS == q_end`,
                        // one base past every shard's half-open own-range) or
                        // `Variant` (keeps a deletion whose POS precedes the region
                        // but whose extent spans in -- a POS owned by no shard),
                        // POS-ownership would silently drop kept records, breaking
                        // byte-identity. Those two modes are only ever set with an
                        // explicit user region, so they take the single reader
                        // below, where `VcfRecordSource` applies the exact
                        // per-record predicate. For `Pos`, passing each shard's
                        // padded fetch window as the reader's `regions` with `Pos`
                        // reproduces the whole-contig split byte-for-byte
                        // (`query_window(Pos)` is identity; `keeps(Pos, ..)` is the
                        // half-open POS test the pre-region sharded reader used).
                        let shards = if overlap == crate::svar2_view::OverlapMode::Pos {
                            crate::vcf_reader::plan_vcf_shards(
                                &regions,
                                &chr,
                                processing_threads.saturating_mul(OVERSHARD_FACTOR),
                                chunk_size as u32,
                            )?
                        } else {
                            Vec::new()
                        };
                        if shards.len() > 1 {
                            let (ref_seq, has_reference) = match fasta.as_deref() {
                                Some(path) => (
                                    Arc::new(crate::vcf_reader::load_contig_seq(path, &chr)?),
                                    true,
                                ),
                                None => (Arc::new(Vec::new()), false),
                            };
                            let units: Vec<crate::shard::WorkUnit> = shards
                                .iter()
                                .map(|s| crate::shard::WorkUnit {
                                    own_start: s.own_start,
                                    own_end: s.own_end,
                                    fetch_start: s.fetch_start,
                                    fetch_end: s.fetch_end,
                                    ordinal: s.ordinal,
                                })
                                .collect();
                            trace_ll!(
                                "[plan {chr}] workers={} shards={}",
                                processing_threads,
                                units.len()
                            );
                            let totals = crate::shard_exec::run(
                                &chr,
                                units,
                                processing_threads,
                                |unit| {
                                    let source = crate::vcf_reader::VcfRecordSource::new(
                                        &vcf_path,
                                        &chr,
                                        &s_refs,
                                        1, // htslib_threads: many concurrent shard readers, keep each small
                                        ploidy,
                                        &fields_owned,
                                        // The shard's padded fetch window IS the
                                        // reader's region; `Pos` makes the fetch
                                        // unwidened and the per-record filter the
                                        // plain half-open POS test. `owned_range`
                                        // (below) does the cross-shard POS dedup.
                                        vec![(unit.fetch_start, unit.fetch_end)],
                                        crate::svar2_view::OverlapMode::Pos,
                                    )?;
                                    Ok(crate::chunk_assembler::ChunkAssembler::with_reference(
                                        Box::new(source),
                                        s_refs.len(),
                                        ploidy,
                                        &chr,
                                        Arc::clone(&ref_seq),
                                        has_reference,
                                        skip_out_of_scope,
                                        check_ref,
                                        &fields_owned,
                                        Some((unit.own_start, unit.own_end)),
                                    ))
                                },
                                |e, unit| {
                                    with_vcf_shard_context(
                                        e,
                                        &chr,
                                        crate::vcf_reader::VcfShard::from(*unit),
                                    )
                                },
                                chunk_size,
                                &tx_dense,
                                &shard_worker_tids,
                            )?;
                            report_ref_excluded(&chr, totals.ref_excluded);
                            report_normalized(&chr, totals.normalized_total);
                            return Ok((
                                totals.dropped_out_of_scope,
                                totals.ref_excluded,
                                totals.normalized_total,
                            ));
                        }
                        Box::new(crate::vcf_reader::VcfRecordSource::new(
                            &vcf_path,
                            &chr,
                            &s_refs,
                            htslib_threads,
                            ploidy,
                            &fields_owned,
                            regions,
                            overlap,
                        )?)
                    }
                    SourceSpec::Pgen {
                        pgen_path: _,
                        pvar_path,
                        var_start,
                        var_end,
                        readers,
                        dosage_readers,
                        regions,
                        overlap,
                        sample_perm,
                    } => {
                        // Ownership is by POSITION (see ChunkAssembler::with_reference),
                        // but the split is by INDEX (plan_pgen_units) -- so read this
                        // contig's .pvar positions up front to translate index-range
                        // shards into contiguous, gap-free position ranges. Every
                        // variant (including co-located multiallelic groups sharing a
                        // position) is then owned by exactly one shard.
                        let mut positions: Vec<u32> = Vec::with_capacity(var_end - var_start);
                        {
                            let mut pvar = crate::pvar::PvarReader::open(&pvar_path, var_start)?;
                            for _ in var_start..var_end {
                                match pvar.next_variant()? {
                                    Some(rec) => positions.push(rec.pos),
                                    None => break,
                                }
                            }
                        }
                        let n = positions.len();
                        // Deliberately NOT over-decomposed with `OVERSHARD_FACTOR`
                        // (VCF-only, see its doc comment): pgenlib holds the GIL
                        // through its decode, so concurrent PGEN shard reads are
                        // GIL-serialized -- extra shards here would add pure
                        // overhead to an already GIL-bound path rather than
                        // improving stealing (fa47530 fixed a convoy from this;
                        // full chr21c: 340s sharded vs 273s serial). This also
                        // stays capped at `readers.len()` because each concurrent
                        // shard needs its own dedicated `pgenlib.PgenReader` --
                        // `readers_pool` provisions exactly one per potential
                        // shard, indexed by ordinal.
                        //
                        // Sharding is additionally restricted to `Pos` overlap:
                        // like the VCF path, POS-ownership dedup only reproduces
                        // `Record`/`Variant` semantics for `Pos`, so those modes
                        // (only set with an explicit user region) take the
                        // single-reader fallback where `PgenRecordSource` applies
                        // the exact per-record predicate. This is belt-and-braces
                        // -- `from_pgen` pins P=1, so `readers.len() == 1` already
                        // forces the fallback regardless of overlap mode.
                        let max_shards = if overlap == crate::svar2_view::OverlapMode::Pos {
                            processing_threads.min(readers.len()).max(1)
                        } else {
                            1
                        };
                        let punits = crate::pgen_shard::plan_pgen_units(
                            &positions,
                            max_shards,
                            crate::normalize::L_MAX,
                        );
                        if punits.len() <= 1 {
                            // Single-reader fallback: identical to today's behavior.
                            let reader = readers
                                .into_iter()
                                .next()
                                .expect("caller provisions >= 1 PGEN reader per contig");
                            Box::new(crate::pgen_reader::PgenRecordSource::new(
                                reader,
                                &pvar_path,
                                var_start,
                                var_end,
                                s_refs.len(),
                                chunk_size,
                                regions,
                                overlap,
                                sample_perm,
                                dosage_readers,
                            )?)
                        } else {
                            let (ref_seq, has_reference) = match fasta.as_deref() {
                                Some(path) => (
                                    Arc::new(crate::vcf_reader::load_contig_seq(path, &chr)?),
                                    true,
                                ),
                                None => (Arc::new(Vec::new()), false),
                            };
                            let units: Vec<crate::shard::WorkUnit> = punits
                                .iter()
                                .map(|u| {
                                    let own_start = if u.own_lo == 0 {
                                        0
                                    } else {
                                        positions[u.own_lo]
                                    };
                                    let own_end = if u.own_hi == n {
                                        u32::MAX
                                    } else {
                                        positions[u.own_hi]
                                    };
                                    crate::shard::WorkUnit {
                                        own_start,
                                        own_end,
                                        fetch_start: (var_start + u.fetch_lo) as u32,
                                        fetch_end: (var_start + u.fetch_hi) as u32,
                                        ordinal: u.ordinal,
                                    }
                                })
                                .collect();
                            // Each shard needs a UNIQUE reader (readers seek
                            // independently), but `make_assembler` below is `Fn + Sync`
                            // and shared across worker threads -- a `Mutex<Option<_>>`
                            // per ordinal lets each worker take its one reader exactly
                            // once. `unit.ordinal < readers.len()` always holds because
                            // `max_shards` (and therefore `punits.len()`) is capped at
                            // `readers.len()` above.
                            let readers_pool: Vec<Mutex<Option<pyo3::Py<pyo3::PyAny>>>> =
                                readers.into_iter().map(|r| Mutex::new(Some(r))).collect();
                            trace_ll!(
                                "[plan {chr}] workers={} shards={}",
                                processing_threads,
                                units.len()
                            );
                            let totals = crate::shard_exec::run(
                                &chr,
                                units,
                                processing_threads,
                                |unit| {
                                    let reader = readers_pool[unit.ordinal]
                                        .lock()
                                        .unwrap()
                                        .take()
                                        .expect("pgen reader taken twice");
                                    let source = crate::pgen_reader::PgenRecordSource::new(
                                        reader,
                                        &pvar_path,
                                        unit.fetch_start as usize,
                                        unit.fetch_end as usize,
                                        s_refs.len(),
                                        chunk_size,
                                        // Sharding is `Pos`-only (see `max_shards`),
                                        // so each shard applies the plain half-open
                                        // POS filter over its own region slice while
                                        // `owned_range` dedups; cloned because the
                                        // closure runs once per shard.
                                        regions.clone(),
                                        overlap,
                                        sample_perm.clone(),
                                        // Not a per-shard pool (see `SourceSpec::Pgen::dosage_readers`
                                        // doc comment) -- this dead branch reuses the
                                        // same flat per-field readers for every shard.
                                        // `Py<PyAny>` needs the GIL to bump its refcount.
                                        pyo3::Python::attach(|py| {
                                            dosage_readers.iter().map(|r| r.clone_ref(py)).collect()
                                        }),
                                    )?;
                                    Ok(crate::chunk_assembler::ChunkAssembler::with_reference(
                                        Box::new(source),
                                        s_refs.len(),
                                        ploidy,
                                        &chr,
                                        Arc::clone(&ref_seq),
                                        has_reference,
                                        skip_out_of_scope,
                                        check_ref,
                                        &fields_owned,
                                        Some((unit.own_start, unit.own_end)),
                                    ))
                                },
                                |e, unit| with_pgen_shard_context(e, &chr, unit),
                                chunk_size,
                                &tx_dense,
                                &shard_worker_tids,
                            )?;
                            report_ref_excluded(&chr, totals.ref_excluded);
                            report_normalized(&chr, totals.normalized_total);
                            return Ok((
                                totals.dropped_out_of_scope,
                                totals.ref_excluded,
                                totals.normalized_total,
                            ));
                        }
                    }
                    SourceSpec::VcfList {
                        vcf_paths,
                        htslib_threads,
                        regions,
                        overlap,
                        members,
                    } => {
                        let ref_seq_opt: Option<Vec<u8>> = match fasta.as_deref() {
                            Some(f) => Some(crate::vcf_reader::load_contig_seq(f, &chr)?),
                            None => None,
                        };
                        let vcf_list = crate::vcf_list_reader::VcfListRecordSource::new(
                            &vcf_paths,
                            &s_refs,
                            &chr,
                            ref_seq_opt.as_deref(),
                            ploidy,
                            htslib_threads,
                            skip_out_of_scope,
                            check_ref,
                            &fields_owned,
                            regions,
                            overlap,
                            &members,
                        )?;
                        Box::new(VcfListDroppedProxy {
                            inner: vcf_list,
                            dropped_out: Arc::clone(&vcf_list_dropped),
                            ref_excluded_out: Arc::clone(&vcf_list_ref_excluded),
                        })
                    }
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
                        regions,
                        overlap,
                        sample_idx,
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
                        regions,
                        overlap,
                        sample_idx,
                    )?),
                };

                // Dedicated rayon pool for reader-side CPU work: bounded per-record
                // normalization batches plus intra-chunk presence packing. The
                // sharded VCF branch above returns before this point because its
                // independent indexed readers consume the same `processing_threads`
                // budget directly; building both would double-reserve cores.
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(processing_threads.max(1))
                    .thread_name(|i| format!("pack-{}", i))
                    .build()
                    .expect("build processing pool");

                let mut reader = crate::chunk_assembler::ChunkAssembler::new(
                    src,
                    s_refs.len(),
                    ploidy,
                    fasta.as_deref(),
                    &chr,
                    skip_out_of_scope,
                    check_ref,
                    &fields_owned,
                )?;
                let mut chunk_id = 0;
                while let Some(dense_chunk) =
                    reader.read_next_chunk(chunk_size, chunk_id, Some(&pool))?
                {
                    tx_dense.send(dense_chunk).unwrap();
                    chunk_id += 1;
                }
                let ref_excluded_total =
                    reader.ref_excluded() + vcf_list_ref_excluded.load(Ordering::Relaxed);
                report_ref_excluded(&chr, ref_excluded_total);
                report_normalized(&chr, reader.normalized_total());
                Ok((
                    reader.dropped_out_of_scope() + vcf_list_dropped.load(Ordering::Relaxed),
                    ref_excluded_total,
                    reader.normalized_total(),
                ))
            }
        })
        .expect("spawn reader");

    // Step 2 -> The Executor
    let fields_exec: Vec<crate::field::FieldSpec> = fields.to_vec();
    let executor_thread = thread::Builder::new()
        .name(format!("exec-{}", chrom))
        .spawn({
            let exec_sink = sink.clone();
            let exec_chrom = chrom.to_string();
            move || {
                let bank = LongAlleleTableWriter::new(tx_long, long_allele_capacity);
                executor::run_compute_engine(
                    rx_dense,
                    tx_sparse,
                    bank,
                    signatures,
                    &fields_exec,
                    &exec_chrom,
                    &exec_sink,
                )
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

    // `_normalized_total` (left-aligned atom count) is already reported via
    // `report_normalized` inside the reader thread closure above -- kept here
    // only so the tuple shape stays self-documenting at the call site.
    let (dropped, ref_excluded, _normalized_total) = match reader_res {
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
        kept_total,
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

    tracing::debug!(chrom = %chrom, "Phase 1 complete; triggering in-memory merge");

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
        crate::mutcat::annotate::annotate_contig(&reader, &paths, &ref_seq, None).map_err(|e| {
            ConversionError::Io {
                context: format!("annotate mutcat {chrom}"),
                source: e,
            }
        })?;
    }

    tracing::info!(chrom = %chrom, "pipeline execution finished successfully");

    sink.flush(chrom);
    sink.contig_done(
        chrom,
        kept_total,
        ref_excluded + dropped,
        contig_started.elapsed().as_millis() as u64,
    );

    Ok(dropped)
}

/// `SparseVar2.from_vcf_list`: build ONE SVAR2 store from N single-sample VCFs
/// with possibly disjoint site lists. `vcf_paths[i]`'s sample is `samples[i]`.
///
/// MVP concurrency: contigs are processed SEQUENTIALLY (a plain loop, no rayon
/// pool) -- this bounds open file descriptors to roughly N (one per input
/// file) rather than N * concurrent_chroms. Cross-contig parallelism is
/// explicit future work. Mirrors `run_conversion_pipeline`
/// (`src/lib.rs`)'s hardware-budget derivation, `parse_manifest`,
/// `finalize_fields`, and `write_meta` tail, minus the rayon dispatch.
///
/// Returns the total number of out-of-scope (symbolic/breakend) ALTs dropped
/// across every input file and contig.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_vcf_list(
    vcf_paths: &[String],
    reference_path: Option<&str>,
    chroms: &[String],
    output_dir: &str,
    samples: &[String],
    chunk_size: usize,
    ploidy: usize,
    max_threads: Option<usize>,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    check_ref: crate::normalize::CheckRef,
    signatures: bool,
    info_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
    format_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
    region_ranges: Vec<(String, u32, u32)>,
    overlap: crate::svar2_view::OverlapMode,
    // `contig_membership[c][i]` is `true` when `vcf_paths[i]` has records on
    // `chroms[c]` (Python's per-file cyvcf2 probe). Outer parallel to `chroms`,
    // inner parallel to `vcf_paths`. Threaded into each contig's
    // `SourceSpec::VcfList` so non-member files are never opened for that contig
    // (issue #122).
    contig_membership: Vec<Vec<bool>>,
    sink: &crate::logging::EventSink,
) -> Result<u64, ConversionError> {
    if contig_membership.len() != chroms.len() {
        return Err(ConversionError::Input(format!(
            "contig_membership must be parallel to chroms (one row per contig): \
             got {} rows and {} contigs",
            contig_membership.len(),
            chroms.len()
        )));
    }
    // `from_vcf_list` holds every input open at once, per contig, so a per-file decode
    // thread is N threads per contig -- 7,089 on the #120 cohort, recreated 24 times.
    // Each wave takes fresh glibc arenas whose 64 MB heaps are never unmapped. They buy
    // nothing: the read phase is single-core-bound (`read~99% exec=0%`). 0 => inline.
    const VCF_LIST_HTSLIB_THREADS: usize = 0;

    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();

    let mut raw = info_fields;
    raw.extend(format_fields);
    let fields = crate::field::parse_manifest(raw)?;

    let mut ranges_by_chrom: std::collections::HashMap<String, Vec<(u32, u32)>> =
        std::collections::HashMap::new();
    for (chrom, start, end) in region_ranges {
        ranges_by_chrom.entry(chrom).or_default().push((start, end));
    }

    let available_cores = match max_threads {
        Some(t) if t > 0 => t,
        _ => std::thread::available_parallelism().unwrap().get(),
    };
    // concurrent_chroms is forced to 1 (sequential loop below) regardless of
    // what the plan suggests -- only `processing_threads` is consumed here.
    let plan = crate::budget::plan_thread_budget(available_cores, 1);
    let processing_threads = plan.processing_threads;
    tracing::info!(threads = processing_threads, "pipeline configured");

    let fasta_ref = reference_path;
    let mut total_dropped: u64 = 0;
    for (chrom, members) in chroms.iter().zip(contig_membership) {
        tracing::info!(chrom = %chrom, "processing contig");
        let dropped = process_chromosome(
            SourceSpec::VcfList {
                vcf_paths: vcf_paths.to_vec(),
                htslib_threads: VCF_LIST_HTSLIB_THREADS,
                regions: ranges_by_chrom.get(chrom).cloned().unwrap_or_default(),
                overlap,
                members,
            },
            fasta_ref,
            chrom,
            output_dir,
            &sample_refs,
            chunk_size,
            ploidy,
            long_allele_capacity,
            skip_out_of_scope,
            check_ref,
            processing_threads,
            signatures,
            &fields,
            sink,
        )?;
        total_dropped += dropped;

        // glibc keeps freed per-contig arena heaps mapped, so RSS ratchets across the
        // 24-contig cohort (issue #120). With htslib threads at 0 and one processing
        // thread, malloc_trim is cheap here (no arena-lock contention) and returns the
        // freed heaps to the OS between contigs. `malloc_trim` is a glibc extension, so
        // gate on `target_env = "gnu"` -- it is absent under musl (Alpine source builds)
        // and on non-Linux; a no-op / compiled-out everywhere but glibc-Linux.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        // SAFETY: malloc_trim takes no ownership and only releases free top-of-heap memory.
        unsafe {
            libc::malloc_trim(0);
        }
    }
    tracing::info!("cohort processing complete");

    // All contigs staged — resolve each field's global on-disk dtype and
    // rewrite its staged values.bin files to that width.
    let resolved_fields =
        crate::field_finalize::finalize_fields(std::path::Path::new(output_dir), chroms, &fields)?;

    // All contigs converted — write the top-level meta.json describing the cohort.
    crate::meta::write_meta(
        std::path::Path::new(output_dir),
        crate::meta::FORMAT_VERSION,
        samples,
        chroms,
        ploidy,
        &resolved_fields,
    )
    .map_err(|e| ConversionError::Io {
        context: format!("write meta.json under {output_dir}"),
        source: e,
    })?;

    Ok(total_dropped)
}

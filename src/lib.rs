// src/lib.rs
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "conversion")]
use rayon::prelude::*;

pub mod bits;

/// Test-facing re-export of `bits::get_bit` for downstream (gvl-side) test
/// oracles that reconstruct per-hap presence bitmasks from `BatchResultSplit`.
///
/// Intentionally public (not `cfg(test)`-gated): ships in `genoray_core`'s
/// public API as part of the gvl-side read-bound parity-oracle surface,
/// alongside `query::oracle::decode_keyref`/`query::oracle::decode_keyref_alt`.
pub fn bits_get_bit(bytes: &[u8], i: usize) -> bool {
    bits::get_bit(bytes, i)
}
#[cfg(feature = "conversion")]
pub mod budget;
#[cfg(feature = "conversion")]
pub mod chunk_assembler;
pub mod cost_model;
pub mod dense;
#[cfg(feature = "conversion")]
pub mod dense_merge;
mod enum_map;
pub mod error;
#[cfg(feature = "conversion")]
pub mod executor;
pub mod field;
#[cfg(feature = "conversion")]
pub mod field_finalize;
pub mod layout;
#[cfg(feature = "conversion")]
pub mod max_del;
#[cfg(feature = "conversion")]
pub mod merge;
#[cfg(feature = "conversion")]
pub mod meta;
#[cfg(feature = "conversion")]
pub mod monitor;
pub mod mutcat;
#[cfg(feature = "conversion")]
pub mod normalize;
pub mod nrvk;
#[cfg(feature = "conversion")]
pub mod orchestrator;
#[cfg(feature = "conversion")]
pub mod pgen_reader;
#[cfg(feature = "conversion")]
pub mod pvar;
#[cfg(feature = "conversion")]
pub mod svar1_reader;
// NOTE: `py_convert` is *not* conversion-only despite being in the original gate
// list: query-core `py_query_batch.rs`/`py_query_decode.rs`/`py_query_ranges.rs`
// import its numpy-array conversion helpers unconditionally, and py_convert.rs
// itself has zero htslib dependency (pure numpy/pyo3 glue). Stays ungated as
// shared infra, same reasoning as `streams` above.
pub mod py_convert;
pub mod py_mutcat;
pub mod py_query;
pub mod py_query_batch;
pub mod py_query_decode;
pub mod py_query_ranges;
pub mod query;
#[cfg(feature = "conversion")]
pub mod record_source;
pub mod rvk;
pub mod search;
pub mod spine;
// NOTE: `streams` (StreamTag/StreamMap/REGISTRY) is *not* conversion-only despite
// being in the original gate list: query-core `rvk.rs` and `types.rs` depend on it
// unconditionally (StreamMap is a real struct field / decode-path type, not just an
// errant import), and streams.rs itself has zero htslib dependency. Gating it broke
// the query-core build; it stays ungated as shared infra.
pub mod streams;
pub mod types;
#[cfg(feature = "conversion")]
pub mod vcf_list_reader;
#[cfg(feature = "conversion")]
pub mod vcf_reader;
#[cfg(feature = "conversion")]
pub mod writer;

#[cfg(feature = "conversion")]
pub use orchestrator::process_chromosome;

/// Build a `.csi` index next to a bgzipped-VCF / BCF at `path`. CSI (min_shift 14)
/// is valid for both, so one path covers `.vcf.gz` and `.bcf`.
#[cfg(feature = "conversion")]
pub fn index_bcf_csi(path: &str) -> Result<(), String> {
    let idx = format!("{path}.csi");
    rust_htslib::bcf::index::build(
        path,
        Some(idx.as_str()),
        1,
        rust_htslib::bcf::index::Type::Csi(14),
    )
    .map_err(|e| format!("failed to build .csi index for {path}: {e:?}"))
}

#[cfg(feature = "conversion")]
#[pyfunction]
fn index_vcf(path: String) -> PyResult<()> {
    index_bcf_csi(&path).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

//The Python Wrapper and resource allocator
#[cfg(feature = "conversion")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (vcf_path, reference_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608, skip_out_of_scope=false, signatures=false, info_fields=Vec::new(), format_fields=Vec::new()))]
fn run_conversion_pipeline(
    py: Python,
    vcf_path: String,
    reference_path: Option<String>,
    chroms: Vec<String>, // now taking a vector
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize, // default 25K variants/chunk — halves per-chunk plumbing overhead vs the old 10K
    ploidy: usize,
    max_threads: Option<usize>,  // accepts an optional integer from Python
    long_allele_capacity: usize, // default 8MB — old 100MB rarely flushed mid-run, blocking executor at finalize
    skip_out_of_scope: bool,
    signatures: bool,
    info_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
    format_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
) -> PyResult<usize> {
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();

    let mut raw = info_fields;
    raw.extend(format_fields);
    let fields =
        crate::field::parse_manifest(raw).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let results: Vec<Result<u64, crate::error::ConversionError>> = py.detach(|| {
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
        let processing_threads = plan.processing_threads;

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
        let fasta_ref: Option<&str> = reference_path.as_deref();
        let results = pool.install(|| {
            chroms
                .par_iter()
                .map(|chrom| {
                    println!("==> Processing {}", chrom);
                    orchestrator::process_chromosome(
                        orchestrator::SourceSpec::Vcf {
                            vcf_path: vcf_path.clone(),
                            htslib_threads,
                        },
                        fasta_ref,
                        chrom,
                        &output_dir,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        long_allele_capacity,
                        skip_out_of_scope,
                        processing_threads,
                        signatures,
                        &fields,
                    )
                })
                .collect::<Vec<_>>()
        });

        println!("Cohort Processing Complete.");
        results
    });

    let mut total_dropped: u64 = 0;
    for r in results {
        total_dropped += r?; // ConversionError -> PyErr via From (category-aware)
    }

    // All contigs staged — resolve each field's global on-disk dtype and
    // rewrite its staged values.bin files to that width.
    let resolved_fields = crate::field_finalize::finalize_fields(
        std::path::Path::new(&output_dir),
        &chroms,
        &fields,
    )?;

    // All contigs converted — write the top-level meta.json describing the cohort.
    crate::meta::write_meta(
        std::path::Path::new(&output_dir),
        crate::meta::FORMAT_VERSION,
        &samples,
        &chroms,
        ploidy,
        &resolved_fields,
    )
    .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("failed to write meta.json: {e}")))?;

    Ok(total_dropped as usize)
}

/// Convert a PLINK2 PGEN to an SVAR2 store.
///
/// `contig_ranges[i]` is the half-open `[var_start, var_end)` variant index range
/// of `chroms[i]` within the `.pvar`. `pgen_readers[i]` is a distinct
/// `pgenlib.PgenReader` for `chroms[i]` -- readers seek independently, so contigs
/// must not share one.
#[cfg(feature = "conversion")]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (pgen_path, pvar_path, reference_path, chroms, contig_ranges, output_dir, samples, chunk_size, max_threads, long_allele_capacity, skip_out_of_scope, signatures, pgen_readers))]
fn run_pgen_conversion_pipeline(
    py: Python,
    pgen_path: String,
    pvar_path: String,
    reference_path: Option<String>,
    chroms: Vec<String>,
    contig_ranges: Vec<(usize, usize)>,
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize,
    max_threads: Option<usize>,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    signatures: bool,
    pgen_readers: Vec<Py<PyAny>>,
) -> PyResult<usize> {
    if chroms.len() != contig_ranges.len() || chroms.len() != pgen_readers.len() {
        return Err(PyValueError::new_err(
            "chroms, contig_ranges, and pgen_readers must be the same length",
        ));
    }
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
    // PGEN is diploid-only.
    let ploidy = 2usize;
    // PGEN carries no FORMAT, and .pvar INFO extraction is out of scope.
    let fields: Vec<crate::field::FieldSpec> = Vec::new();

    // Pair each contig with its own reader BEFORE detaching, so the Py handles move
    // into the worker threads (Py<PyAny> is Send; PyAny is not).
    let jobs: Vec<(String, (usize, usize), Py<PyAny>)> = chroms
        .iter()
        .cloned()
        .zip(contig_ranges.iter().copied())
        .zip(pgen_readers)
        .map(|((c, r), rd)| (c, r, rd))
        .collect();

    let results: Vec<Result<u64, crate::error::ConversionError>> = py.detach(|| {
        let available_cores = match max_threads {
            Some(t) if t > 0 => t,
            _ => std::thread::available_parallelism().unwrap().get(),
        };
        let plan = crate::budget::plan_thread_budget(available_cores, jobs.len());
        let concurrent_chroms = plan.concurrent_chroms;
        let processing_threads = plan.processing_threads;
        println!(
            "Pipeline Config (PGEN): {} concurrent chromosomes | {} processing threads each.",
            concurrent_chroms, processing_threads
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrent_chroms)
            .thread_name(|i| format!("chrom-{}", i))
            .build()
            .expect("build chrom pool");

        pool.install(|| {
            jobs.into_par_iter()
                .map(|(chrom, (lo, hi), reader)| {
                    orchestrator::process_chromosome(
                        orchestrator::SourceSpec::Pgen {
                            pgen_path: pgen_path.clone(),
                            pvar_path: pvar_path.clone(),
                            var_start: lo,
                            var_end: hi,
                            reader,
                        },
                        reference_path.as_deref(),
                        &chrom,
                        &output_dir,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        long_allele_capacity,
                        skip_out_of_scope,
                        processing_threads,
                        signatures,
                        &fields,
                    )
                })
                .collect()
        })
    });

    let mut dropped = 0u64;
    for r in results {
        dropped += r?;
    }

    // All contigs staged — resolve each field's global on-disk dtype (no-op:
    // PGEN carries no fields) and rewrite its staged values.bin files to that
    // width, then write the top-level meta.json describing the cohort. Mirrors
    // the tail of `run_conversion_pipeline` -- without this, `SparseVar2(out)`
    // has no `meta.json` to load.
    let resolved_fields = crate::field_finalize::finalize_fields(
        std::path::Path::new(&output_dir),
        &chroms,
        &fields,
    )?;

    crate::meta::write_meta(
        std::path::Path::new(&output_dir),
        crate::meta::FORMAT_VERSION,
        &samples,
        &chroms,
        ploidy,
        &resolved_fields,
    )
    .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("failed to write meta.json: {e}")))?;

    Ok(dropped as usize)
}

/// Convert N single-sample VCFs (with possibly disjoint site lists) into ONE
/// SVAR2 store (`SparseVar2.from_vcf_list`). `vcf_paths[i]`'s sample is
/// `samples[i]` -- the two lists are parallel, one file per sample. Contigs
/// run sequentially (see `orchestrator::run_vcf_list`'s docs); `signatures`
/// requires a reference (validated Python-side).
#[cfg(feature = "conversion")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (vcf_paths, reference_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608, skip_out_of_scope=false, signatures=false, info_fields=Vec::new(), format_fields=Vec::new()))]
fn run_vcf_list_conversion_pipeline(
    py: Python,
    vcf_paths: Vec<String>,
    reference_path: Option<String>,
    chroms: Vec<String>,
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize,
    ploidy: usize,
    max_threads: Option<usize>,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    signatures: bool,
    info_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
    format_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
) -> PyResult<usize> {
    let dropped: u64 = py.detach(|| {
        orchestrator::run_vcf_list(
            &vcf_paths,
            reference_path.as_deref(),
            &chroms,
            &output_dir,
            &samples,
            chunk_size,
            ploidy,
            max_threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            info_fields,
            format_fields,
        )
    })?;
    Ok(dropped as usize)
}

/// Convert a SVAR1 (`SparseVar`) store to an SVAR2 store natively (no htslib).
///
/// Per-contig `POS`/`REF`/`ALT` come from Python (it reads `index.arrow` via
/// polars); the big sample-major sparse arrays and per-entry field arrays are
/// mmap'd in Rust from `svar1_dir`. `contig_starts[i]`/`contig_lens[i]` give
/// contig `i`'s global variant-id start and length. `pos_per_contig[i]` is 0-based
/// (`POS-1`). `format_fields` is the SVAR1 field manifest (all FORMAT);
/// `format_src_dtypes[j]` is the numpy dtype of `format_fields[j]`'s on-disk array.
#[cfg(feature = "conversion")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (svar1_dir, reference_path, chroms, contig_starts, contig_lens, output_dir, samples, ploidy, chunk_size, max_threads, long_allele_capacity, skip_out_of_scope, signatures, pos_per_contig, ref_bytes_per_contig, ref_offsets_per_contig, alt_bytes_per_contig, alt_offsets_per_contig, format_fields, format_src_dtypes))]
fn run_svar1_conversion_pipeline(
    py: Python,
    svar1_dir: String,
    reference_path: Option<String>,
    chroms: Vec<String>,
    contig_starts: Vec<usize>,
    contig_lens: Vec<usize>,
    output_dir: String,
    samples: Vec<String>,
    ploidy: usize,
    chunk_size: usize,
    max_threads: Option<usize>,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    signatures: bool,
    pos_per_contig: Vec<Vec<u32>>,
    ref_bytes_per_contig: Vec<Vec<u8>>,
    ref_offsets_per_contig: Vec<Vec<i64>>,
    alt_bytes_per_contig: Vec<Vec<u8>>,
    alt_offsets_per_contig: Vec<Vec<i64>>,
    format_fields: Vec<(String, String, String, Option<String>, Option<f64>)>,
    format_src_dtypes: Vec<String>,
) -> PyResult<usize> {
    let n = chroms.len();
    if [
        contig_starts.len(),
        contig_lens.len(),
        pos_per_contig.len(),
        ref_bytes_per_contig.len(),
        ref_offsets_per_contig.len(),
        alt_bytes_per_contig.len(),
        alt_offsets_per_contig.len(),
    ]
    .iter()
    .any(|&l| l != n)
    {
        return Err(PyValueError::new_err(
            "all per-contig inputs must have the same length as `chroms`",
        ));
    }
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
    let fields = crate::field::parse_manifest(format_fields)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Move per-contig owned data into jobs before detaching.
    let mut jobs: Vec<_> = Vec::with_capacity(n);
    for i in 0..n {
        jobs.push((
            chroms[i].clone(),
            contig_starts[i],
            contig_lens[i],
            pos_per_contig[i].clone(),
            ref_bytes_per_contig[i].clone(),
            ref_offsets_per_contig[i].clone(),
            alt_bytes_per_contig[i].clone(),
            alt_offsets_per_contig[i].clone(),
        ));
    }

    let results: Vec<Result<u64, crate::error::ConversionError>> = py.detach(|| {
        let available_cores = match max_threads {
            Some(t) if t > 0 => t,
            _ => std::thread::available_parallelism().unwrap().get(),
        };
        let plan = crate::budget::plan_thread_budget(available_cores, jobs.len());
        let concurrent_chroms = plan.concurrent_chroms;
        let processing_threads = plan.processing_threads;
        println!(
            "Pipeline Config (SVAR1): {} concurrent chromosomes | {} processing threads each.",
            concurrent_chroms, processing_threads
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrent_chroms)
            .thread_name(|i| format!("chrom-{}", i))
            .build()
            .expect("build chrom pool");

        pool.install(|| {
            jobs.into_par_iter()
                .map(|(chrom, start, len, pos, rb, ro, ab, ao)| {
                    orchestrator::process_chromosome(
                        orchestrator::SourceSpec::Svar1 {
                            svar1_dir: svar1_dir.clone(),
                            contig_start: start,
                            n_local: len,
                            pos,
                            ref_bytes: rb,
                            ref_offsets: ro,
                            alt_bytes: ab,
                            alt_offsets: ao,
                            format_fields: fields.clone(),
                            format_src_dtypes: format_src_dtypes.clone(),
                        },
                        reference_path.as_deref(),
                        &chrom,
                        &output_dir,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        long_allele_capacity,
                        skip_out_of_scope,
                        processing_threads,
                        signatures,
                        &fields,
                    )
                })
                .collect()
        })
    });

    let mut dropped = 0u64;
    for r in results {
        dropped += r?;
    }

    let resolved_fields = crate::field_finalize::finalize_fields(
        std::path::Path::new(&output_dir),
        &chroms,
        &fields,
    )?;
    crate::meta::write_meta(
        std::path::Path::new(&output_dir),
        crate::meta::FORMAT_VERSION,
        &samples,
        &chroms,
        ploidy,
        &resolved_fields,
    )
    .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("failed to write meta.json: {e}")))?;

    Ok(dropped as usize)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_pgen_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_vcf_list_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_svar1_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(index_vcf, m)?)?;
    m.add_class::<crate::py_query::PyContigReader>()?;
    Ok(())
}

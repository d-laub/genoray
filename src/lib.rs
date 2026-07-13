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
#[cfg(feature = "conversion")]
pub mod svar2_source;
pub mod types;
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

/// Read `{store_path}/meta.json` and return `(samples, ploidy)`. `run_view_pipeline`
/// needs both: `samples` to map the caller's subset sample NAMES to original
/// column indices (the per-contig sidecars are indexed by original column, not
/// subset position), `ploidy` because `Svar2Source`/`ContigReader::open` must be
/// opened with the store's actual ploidy.
#[cfg(feature = "conversion")]
fn read_store_meta(
    store_path: &str,
) -> Result<(Vec<String>, usize), crate::error::ConversionError> {
    use crate::error::ConversionError;
    let path = std::path::Path::new(store_path).join("meta.json");
    if !path.exists() {
        return Err(ConversionError::MissingFile {
            path: path.display().to_string(),
        });
    }
    let text = std::fs::read_to_string(&path).map_err(|e| ConversionError::Io {
        context: path.display().to_string(),
        source: e,
    })?;
    let meta: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ConversionError::Input(format!("malformed {}: {e}", path.display())))?;
    let samples: Vec<String> = meta["samples"]
        .as_array()
        .ok_or_else(|| {
            ConversionError::Input(format!("{} has no `samples` array", path.display()))
        })?
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();
    let ploidy = meta["ploidy"]
        .as_u64()
        .ok_or_else(|| ConversionError::Input(format!("{} has no `ploidy`", path.display())))?
        as usize;
    Ok((samples, ploidy))
}

/// Merge overlapping/adjacent `[start, end)` regions on one contig.
///
/// NOT a correctness requirement: `Svar2Source` already ORs carrier bits across
/// overlapping regions via its internal `BTreeMap` (see `carrier_bits_or_across_
/// overlapping_regions` in `tests/test_svar2_source.rs`), so duplicated/adjacent
/// spans decode to the same result either way. This is purely an optimization
/// to avoid redundant `read_ranges` work when `merge_overlapping = true`.
#[cfg(feature = "conversion")]
fn merge_regions(mut regions: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    regions.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(regions.len());
    for (start, end) in regions {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
            continue;
        }
        merged.push((start, end));
    }
    merged
}

/// Write a region/sample subset of a finished SVAR2 store by re-running the
/// ordinary conversion pipeline (`process_chromosome`) over a synthetic
/// [`svar2_source::Svar2Source`] (`SourceSpec::Svar2`) instead of a VCF/PGEN
/// reader. Backs `SparseVar2.write_view`'s "coarse seam" (see
/// `docs/superpowers/specs/2026-07-12-svar2-concat-split-write-view-design.md`).
///
/// # `fasta_path` is HARDCODED to `None` below — do not change this
///
/// `Svar2Source::to_raw_record` fabricates REF bases (a deliberately mismatched
/// base for SNPs, `b'N'` filler bytes for deletions) because a finished store
/// keeps no REF bytes at all. That is safe **only** because `fasta_path = None`
/// disables the two REF-bases-dependent normalization steps, `validate_ref` and
/// `left_align`, in `ChunkAssembler::decompose_record` (both gated on
/// `self.has_reference`) — a finished store is already atomic, biallelic, and
/// left-aligned, and everything downstream (`classify_variant`/`pack_variant`)
/// reads only `ilen`+`alt`, never REF bytes. Forwarding a real FASTA here would
/// compare the synthetic REF against the real genome and fail or corrupt the
/// run. `reference` is accepted only for **signature parity** with the Python
/// shim (whose `reference=` recomputes the mutcat sidecar — a feature this MVP
/// defers, hence `signatures = false` below too) and MUST NOT reach
/// `process_chromosome`'s `fasta_path`.
///
/// # Fields
///
/// The MVP is genotypes-only: `Svar2Source` emits empty `info_raw`/`format_raw`,
/// so INFO/FORMAT carry-through is impossible on this path. `fields` must
/// therefore be empty, or this fails fast with `ValueError` before touching the
/// filesystem, rather than silently dropping the requested fields.
#[cfg(feature = "conversion")]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (store_path, out_dir, contigs, samples, regions, regions_overlap, merge_overlapping, fields, reference=None, max_threads=None, overwrite=false))]
pub fn run_view_pipeline(
    py: Python,
    store_path: String,
    out_dir: String,
    contigs: Vec<String>,
    samples: Vec<String>,
    regions: Vec<(String, u32, u32)>,
    regions_overlap: String,
    merge_overlapping: bool,
    fields: Vec<String>,
    reference: Option<String>,
    max_threads: Option<usize>,
    overwrite: bool,
) -> PyResult<()> {
    // `reference` is accepted only for signature parity with the Python shim;
    // see the doc comment above for why it must never reach `process_chromosome`'s
    // `fasta_path`.
    let _ = reference;

    if !fields.is_empty() {
        return Err(crate::error::ConversionError::Input(
            "field carry-through is not yet implemented for SVAR2 views".to_string(),
        )
        .into());
    }

    let overlap_mode = match regions_overlap.as_str() {
        "pos" => crate::svar2_source::OverlapMode::Pos,
        "record" => crate::svar2_source::OverlapMode::Record,
        "variant" => crate::svar2_source::OverlapMode::Variant,
        other => {
            return Err(crate::error::ConversionError::Input(format!(
                "regions_overlap must be one of 'pos', 'record', 'variant'; got {other:?}"
            ))
            .into());
        }
    };

    // Fail fast on a region naming a contig outside `contigs` before touching
    // the filesystem.
    let known: std::collections::HashSet<&str> = contigs.iter().map(|s| s.as_str()).collect();
    for (chrom, _, _) in &regions {
        if !known.contains(chrom.as_str()) {
            return Err(crate::error::ConversionError::Input(format!(
                "region references unknown contig {chrom:?} (not in {contigs:?})"
            ))
            .into());
        }
    }

    // Group regions per contig, preserving `contigs`' order. Only contigs with
    // >=1 region are actually processed / written to the output.
    let mut by_contig: std::collections::HashMap<&str, Vec<(u32, u32)>> =
        std::collections::HashMap::new();
    for (chrom, start, end) in &regions {
        by_contig
            .entry(chrom.as_str())
            .or_default()
            .push((*start, *end));
    }
    let out_contigs: Vec<String> = contigs
        .iter()
        .filter(|c| by_contig.contains_key(c.as_str()))
        .cloned()
        .collect();
    if out_contigs.is_empty() {
        return Err(crate::error::ConversionError::Input(
            "no regions selected any contig".to_string(),
        )
        .into());
    }

    // Read the SOURCE store's meta.json to map subset sample NAMES -> original
    // column indices, and to inherit ploidy (the per-contig sidecars are
    // indexed by original column, not by subset position -- `Svar2Source`
    // needs the original indices to resolve them).
    let (src_samples, ploidy) = read_store_meta(&store_path)?;
    let mut samples_orig_idx = Vec::with_capacity(samples.len());
    for name in &samples {
        let idx = src_samples.iter().position(|s| s == name).ok_or_else(|| {
            crate::error::ConversionError::Input(format!(
                "sample {name:?} not found in {store_path}"
            ))
        })?;
        samples_orig_idx.push(idx);
    }
    if samples_orig_idx.is_empty() {
        return Err(crate::error::ConversionError::Input(
            "write_view requires at least one sample".to_string(),
        )
        .into());
    }

    // Output dir: fail fast unless `overwrite`, then start from a clean slate --
    // a view re-derives every byte, so leftover contigs from a previous run at
    // the same path must not survive (mirrors `_write_store`'s
    // check-then-`rmtree` convention used by `subset_contigs`/`concat`/`split`).
    let out_path = std::path::Path::new(&out_dir);
    if out_path.exists() {
        if !overwrite {
            return Err(crate::error::ConversionError::Input(format!(
                "{out_dir} exists; pass overwrite=True"
            ))
            .into());
        }
        std::fs::remove_dir_all(out_path).map_err(|e| crate::error::ConversionError::Io {
            context: format!("remove_dir_all {out_dir}"),
            source: e,
        })?;
    }
    std::fs::create_dir_all(out_path).map_err(|e| crate::error::ConversionError::Io {
        context: format!("create_dir_all {out_dir}"),
        source: e,
    })?;

    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
    let fields_spec: Vec<crate::field::FieldSpec> = Vec::new();

    let results: Vec<Result<u64, crate::error::ConversionError>> = py.detach(|| {
        let available_cores = match max_threads {
            Some(t) if t > 0 => t,
            _ => std::thread::available_parallelism().unwrap().get(),
        };
        let plan = crate::budget::plan_thread_budget(available_cores, out_contigs.len());
        let concurrent_chroms = plan.concurrent_chroms;
        let processing_threads = plan.processing_threads;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrent_chroms)
            .thread_name(|i| format!("view-{}", i))
            .build()
            .expect("build view pool");

        pool.install(|| {
            out_contigs
                .par_iter()
                .map(|chrom| {
                    let mut chrom_regions =
                        by_contig.get(chrom.as_str()).cloned().unwrap_or_default();
                    if merge_overlapping {
                        chrom_regions = merge_regions(chrom_regions);
                    }
                    orchestrator::process_chromosome(
                        orchestrator::SourceSpec::Svar2 {
                            store_path: store_path.clone(),
                            samples_orig_idx: samples_orig_idx.clone(),
                            regions: chrom_regions,
                            overlap_mode,
                        },
                        None, // fasta_path -- SEE DOC COMMENT ABOVE. MUST STAY None.
                        chrom,
                        &out_dir,
                        &sample_refs,
                        25_000, // chunk_size -- same default as run_conversion_pipeline
                        ploidy,
                        8_388_608, // long_allele_capacity -- same default as run_conversion_pipeline
                        true,      // skip_out_of_scope
                        processing_threads,
                        false, // signatures -- deferred; mutcat recompute is out of MVP scope
                        &fields_spec,
                    )
                })
                .collect::<Vec<_>>()
        })
    });

    for r in results {
        r?;
    }

    let resolved_fields =
        crate::field_finalize::finalize_fields(out_path, &out_contigs, &fields_spec)?;

    crate::meta::write_meta(
        out_path,
        crate::meta::FORMAT_VERSION,
        &samples,
        &out_contigs,
        ploidy,
        &resolved_fields,
    )
    .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("failed to write meta.json: {e}")))?;

    Ok(())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_pgen_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_view_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(index_vcf, m)?)?;
    m.add_class::<crate::py_query::PyContigReader>()?;
    Ok(())
}

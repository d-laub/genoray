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
// NOTE: `field_finalize` is *not* conversion-only despite housing the write-path
// `finalize_fields` scan/rewrite code: query-core `field.rs::encode_scalar` (the
// SVAR2 decode non-carrier fill, used by gvl with `default-features = false`) calls
// its `encode`/`is_staged_missing` to stay byte-identical to the finalize path, and
// the module has zero htslib dependency (only `error`/`field`/rayon/std). Gating it
// broke the query-core build; it stays ungated as shared infra, same reasoning as
// `py_convert` and `streams` below.
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
#[cfg(feature = "conversion")]
pub mod svar2_slice;
#[cfg(feature = "conversion")]
pub mod svar2_view;
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
#[pyo3(signature = (vcf_path, reference_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608, skip_out_of_scope=false, signatures=false, info_fields=Vec::new(), format_fields=Vec::new(), check_ref="e".to_string()))]
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
    check_ref: String,
) -> PyResult<usize> {
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();

    let mut raw = info_fields;
    raw.extend(format_fields);
    let fields =
        crate::field::parse_manifest(raw).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let check_ref: crate::normalize::CheckRef = check_ref.parse().map_err(PyValueError::new_err)?;

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
                        check_ref,
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
#[pyo3(signature = (pgen_path, pvar_path, reference_path, chroms, contig_ranges, output_dir, samples, chunk_size, max_threads, long_allele_capacity, skip_out_of_scope, signatures, pgen_readers, check_ref))]
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
    check_ref: String,
) -> PyResult<usize> {
    if chroms.len() != contig_ranges.len() || chroms.len() != pgen_readers.len() {
        return Err(PyValueError::new_err(
            "chroms, contig_ranges, and pgen_readers must be the same length",
        ));
    }
    let check_ref: crate::normalize::CheckRef = check_ref.parse().map_err(PyValueError::new_err)?;
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
                        check_ref,
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

/// Read `{store_path}/meta.json` and return `(samples, ploidy)`. `run_slice_view`
/// needs both: `samples` to map the caller's subset sample NAMES to original
/// column indices (the per-contig sidecars are indexed by original column, not
/// subset position), `ploidy` because `ContigReader::open` must be opened with
/// the store's actual ploidy.
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
/// NOT a correctness requirement: `svar2_slice::region_hits` already sorts and
/// dedups the kept call/row indices across all of a variant's matching regions,
/// so duplicated/adjacent spans decode to the same result either way. This is
/// purely an optimization to avoid redundant search-tree work when
/// `merge_overlapping = true`.
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

/// Write a region/sample subset of a finished SVAR2 store by DIRECTLY slicing
/// its finished sidecars (`svar2_slice::slice_contig`). O(output) memory, no
/// cost model, no pipeline re-run. `reroute` selects a routing policy within
/// this same slicer rather than dispatching to a separate code path — see
/// `svar2_slice::Routing` below.
///
/// This carries INFO/FORMAT fields (the slicer byte-copies each field's
/// `values.bin` at the source dtype) and recomputes the mutcat sidecar from
/// `reference` when given, so the output is a fully self-describing store.
/// Fields are the store's finalized specs as `(name, category, dtype,
/// default)`; `dtype` must be a concrete finalized dtype (never `"auto"`).
///
/// # Why `reference` is used here
///
/// The slicer copies genotype/field bytes verbatim, so no FASTA is needed for
/// them. `reference` is used ONLY to recompute the mutcat (mutational-signature)
/// sidecar over the sliced output, matching `from_vcf(signatures=True)`: mutcat
/// presence is detected purely from the on-disk `mutcat/*/code.bin` sidecar, so
/// this stamps nothing into `meta.json`. If `reference` is `None`, mutcat is
/// skipped entirely.
///
/// `reroute` selects the slicer's own routing policy (`Routing::Recompute`
/// re-runs the cost model on the subset and may flip a variant's stream;
/// `Routing::Preserve`, the default, keeps each variant's source stream) — see
/// `svar2_slice::Routing`. Contigs are sliced concurrently across a rayon pool
/// sized by `max_threads` (`None` autodetects); slicing is independent per
/// contig, so this changes wall time only, never a single output byte.
#[cfg(feature = "conversion")]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (store_path, out_dir, contigs, samples, regions, regions_overlap, merge_overlapping, fields, reference=None, reroute=false, max_threads=None, overwrite=false))]
pub fn run_slice_view(
    py: Python,
    store_path: String,
    out_dir: String,
    contigs: Vec<String>,
    samples: Vec<String>,
    regions: Vec<(String, u32, u32)>,
    regions_overlap: String,
    merge_overlapping: bool,
    fields: Vec<(String, String, String, Option<f64>)>,
    reference: Option<String>,
    reroute: bool,
    max_threads: Option<usize>,
    overwrite: bool,
) -> PyResult<()> {
    use crate::error::ConversionError;
    use crate::field::{FieldCategory, FieldSpec, HtslibType, StorageDtype};
    use crate::field_finalize::ResolvedField;
    use crate::svar2_slice::Routing;

    // --- fail-fast band: every raise here happens BEFORE the output dir is
    // created, so a rejected request leaves no bytes. ---
    let overlap_mode = match regions_overlap.as_str() {
        "pos" => crate::svar2_view::OverlapMode::Pos,
        "record" => crate::svar2_view::OverlapMode::Record,
        "variant" => crate::svar2_view::OverlapMode::Variant,
        other => {
            return Err(ConversionError::Input(format!(
                "regions_overlap must be one of 'pos', 'record', 'variant'; got {other:?}"
            ))
            .into());
        }
    };

    // Reject a region naming a contig outside `contigs`.
    let known: std::collections::HashSet<&str> = contigs.iter().map(|s| s.as_str()).collect();
    for (chrom, _, _) in &regions {
        if !known.contains(chrom.as_str()) {
            return Err(ConversionError::Input(format!(
                "region references unknown contig {chrom:?} (not in {contigs:?})"
            ))
            .into());
        }
    }

    // Group regions per contig, preserving `contigs`' order; only contigs with
    // >=1 region are written.
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
        return Err(ConversionError::Input("no regions selected any contig".to_string()).into());
    }

    // Source meta -> subset sample column indices + inherited ploidy.
    let (src_samples, ploidy) = read_store_meta(&store_path)?;
    let mut samples_orig_idx = Vec::with_capacity(samples.len());
    for name in &samples {
        let idx = src_samples.iter().position(|s| s == name).ok_or_else(|| {
            ConversionError::Input(format!("sample {name:?} not found in {store_path}"))
        })?;
        samples_orig_idx.push(idx);
    }
    if samples_orig_idx.is_empty() {
        return Err(
            ConversionError::Input("write_view requires at least one sample".to_string()).into(),
        );
    }

    // Build the field specs (for the slicer) and resolved fields (for meta.json)
    // from the SAME tuples. The slicer byte-copies at the source dtype, so meta
    // records exactly the requested subset at that dtype -- do NOT re-finalize
    // (finalize could re-narrow the dtype and corrupt the copied provenance).
    let mut fields_spec: Vec<FieldSpec> = Vec::with_capacity(fields.len());
    let mut resolved_fields: Vec<ResolvedField> = Vec::with_capacity(fields.len());
    for (name, category, dtype, default) in &fields {
        let category = match category.as_str() {
            "info" => FieldCategory::Info,
            "format" => FieldCategory::Format,
            other => {
                return Err(ConversionError::Input(format!("bad field category {other:?}")).into());
            }
        };
        let dtype = StorageDtype::from_meta_str(dtype).ok_or_else(|| {
            ConversionError::Input(format!(
                "field {name:?} has non-finalized/invalid dtype {dtype:?}; \
                 a finalized store never stores \"auto\""
            ))
        })?;
        // htype is only needed to satisfy `FieldSpec`; the slicer copies raw
        // bytes and never reads it. Float dtypes -> Float, all else -> Int.
        let htype = match dtype {
            StorageDtype::F16 | StorageDtype::F32 => HtslibType::Float,
            _ => HtslibType::Int,
        };
        fields_spec.push(FieldSpec {
            name: name.clone(),
            category,
            htype,
            dtype,
            default: *default,
        });
        resolved_fields.push(ResolvedField {
            name: name.clone(),
            category,
            dtype,
            default: *default,
        });
    }

    // `reroute` selects the routing policy: `Recompute` re-runs the cost model
    // on the subset (may flip a variant's stream), `Preserve` (default) keeps
    // each variant's source stream.
    let routing = if reroute {
        Routing::Recompute
    } else {
        Routing::Preserve
    };

    // Cost-model terms for `Routing::Recompute` (`reroute=True`), computed
    // exactly as the production converter does (`src/rvk.rs:230-238`): the
    // mutational-signature sidecar is on iff a reference was given, and
    // `info_bits`/`format_bits` are the summed per-record field widths in bits.
    // These are IGNORED under `Routing::Preserve`, but computed unconditionally
    // so the two routes cannot silently diverge.
    let sidecar_bits_enabled = reference.is_some();
    let info_bits: u64 = fields_spec
        .iter()
        .filter(|f| f.category == FieldCategory::Info)
        .map(|f| f.dtype.width_bytes().unwrap_or(4) as u64 * 8)
        .sum();
    let format_bits: u64 = fields_spec
        .iter()
        .filter(|f| f.category == FieldCategory::Format)
        .map(|f| f.dtype.width_bytes().unwrap_or(4) as u64 * 8)
        .sum();

    // Reference FASTA (if any): validate the path is openable and contains
    // every out-contig UP FRONT, before any output byte is written. Without
    // this, a missing/unreadable `reference` would only surface inside the
    // per-contig loop below (mutcat recompute step), by which point earlier
    // contigs' sidecars are already on disk -- a partially-written, unreadable
    // store (sidecars present, meta.json absent) rather than a clean fail-fast
    // raise. This validation fetches NO sequence bytes (only faidx open +
    // per-contig length lookups); the loop still loads each contig's sequence
    // LAZILY, keeping peak memory at O(1 contig) rather than O(genome) -- the
    // property `reroute=False` promises for whole-genome reference views.
    if let Some(fasta) = reference.as_deref() {
        crate::vcf_reader::validate_contigs_in_fasta(fasta, &out_contigs)?;
    }

    // Output dir: fail fast unless `overwrite`, then start from a clean slate.
    let out_path = std::path::Path::new(&out_dir);
    if out_path.exists() {
        if !overwrite {
            return Err(
                ConversionError::Input(format!("{out_dir} exists; pass overwrite=True")).into(),
            );
        }
        std::fs::remove_dir_all(out_path).map_err(|e| ConversionError::Io {
            context: format!("remove_dir_all {out_dir}"),
            source: e,
        })?;
    }
    std::fs::create_dir_all(out_path).map_err(|e| ConversionError::Io {
        context: format!("create_dir_all {out_dir}"),
        source: e,
    })?;

    // --- per-contig slice (+ optional mutcat recompute), sliced concurrently
    // across contigs with rayon (Task 6) --- contigs are independent, so
    // threading changes wall time only, never a single output byte (see
    // `multi_contig_slice_is_thread_invariant`). The slicer spawns no threads
    // of its own -- unlike `run_conversion`, there's no htslib-decode budget to
    // split per contig, so this is a plain thread-count pool, NOT
    // `budget::plan_thread_budget` (which splits htslib + pipeline threads the
    // slicer doesn't have). Release the GIL for the pure Rust/IO work.
    //
    // Peak memory: O(output per contig) x concurrent contigs; with
    // `reference=` each in-flight contig also holds that contig's reference
    // sequence resident (~250 MB for chr1) for its mutcat recompute.
    let n_subset = samples_orig_idx.len();
    let results: Vec<Result<usize, ConversionError>> = py.detach(|| {
        let available_cores = match max_threads {
            Some(t) if t > 0 => t,
            _ => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(available_cores.min(out_contigs.len()).max(1))
            .thread_name(|i| format!("slice-{i}"))
            .build()
            .unwrap();
        pool.install(|| {
            out_contigs
                .par_iter()
                .map(|chrom| {
                    let mut chrom_regions =
                        by_contig.get(chrom.as_str()).cloned().unwrap_or_default();
                    if merge_overlapping {
                        chrom_regions = merge_regions(chrom_regions);
                    }
                    let n = crate::svar2_slice::slice_contig(
                        &store_path,
                        &out_dir,
                        chrom,
                        &samples_orig_idx,
                        ploidy,
                        &chrom_regions,
                        overlap_mode,
                        &fields_spec,
                        routing,
                        sidecar_bits_enabled,
                        info_bits,
                        format_bits,
                    )?;

                    // Mutcat is detected purely from `mutcat/*/code.bin` on disk (see
                    // Python's `_is_annotated`), so recompute it over the sliced output
                    // when a reference is given -- nothing is stamped into meta.json.
                    // The reference was validated up front (see the fail-fast band), so
                    // this loads lazily: peak O(1 contig) resident PER IN-FLIGHT TASK,
                    // dropped once that task's (single) contig is annotated.
                    if let Some(fasta) = reference.as_deref() {
                        let ref_seq = crate::vcf_reader::load_contig_seq(fasta, chrom)?;
                        let reader =
                            crate::query::ContigReader::open(&out_dir, chrom, n_subset, ploidy)
                                .map_err(|e| ConversionError::Io {
                                    context: format!("{out_dir}/{chrom}"),
                                    source: e,
                                })?;
                        let paths = crate::layout::ContigPaths::new(&out_dir, chrom);
                        // Views are strand-free (write_view takes no `gtf=`,
                        // mirroring write-time `signatures=True`): no
                        // StrandIntervals, so SBS192/384 are unavailable on a
                        // sliced view, same as a `from_vcf(signatures=True)` store.
                        crate::mutcat::annotate::annotate_contig(&reader, &paths, &ref_seq, None)
                            .map_err(|e| ConversionError::Io {
                            context: format!("annotate mutcat {out_dir}/{chrom}"),
                            source: e,
                        })?;
                    }
                    Ok(n)
                })
                .collect()
        })
    });
    for r in results {
        r?;
    }

    // meta.json: subset samples, kept contigs, inherited ploidy, and the fields
    // exactly as requested (source dtype -- see the no-finalize note above).
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

/// Per-variant routing/carrier stats for one finished SVAR2 contig, for the
/// `reroute` measurement spike (`scripts/svar2_reroute_spike.py`). Returns four
/// parallel 1-D arrays, one entry per variant: `is_indel` (u8 0/1), `src_dense`
/// (u8 0/1 — 1 iff stored dense), `x_full` (u32 whole-cohort carrier haps), and
/// `x_sub` (u32 carrier haps among `subset`). `subset` holds *original* sample
/// column indices. No gather is performed — see `ContigReader::variant_stats`.
#[cfg(feature = "conversion")]
#[pyfunction]
#[allow(clippy::type_complexity)]
fn svar2_variant_stats<'py>(
    py: Python<'py>,
    store_path: String,
    chrom: String,
    subset: Vec<usize>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<u8>>,
    Bound<'py, numpy::PyArray1<u8>>,
    Bound<'py, numpy::PyArray1<u32>>,
    Bound<'py, numpy::PyArray1<u32>>,
)> {
    let (samples, ploidy) =
        read_store_meta(&store_path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let reader = crate::query::ContigReader::open(&store_path, &chrom, samples.len(), ploidy)
        .map_err(|e| PyValueError::new_err(format!("open {store_path}/{chrom}: {e}")))?;
    let stats = reader.variant_stats(&subset);
    let is_indel: Vec<u8> = stats.is_indel.iter().map(|&b| b as u8).collect();
    let src_dense: Vec<u8> = stats.src_dense.iter().map(|&b| b as u8).collect();
    Ok((
        numpy::PyArray1::from_slice(py, &is_indel),
        numpy::PyArray1::from_slice(py, &src_dense),
        numpy::PyArray1::from_slice(py, &stats.x_full),
        numpy::PyArray1::from_slice(py, &stats.x_sub),
    ))
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
#[pyo3(signature = (vcf_paths, reference_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608, skip_out_of_scope=false, signatures=false, info_fields=Vec::new(), format_fields=Vec::new(), check_ref="e".to_string()))]
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
    check_ref: String,
) -> PyResult<usize> {
    let check_ref: crate::normalize::CheckRef = check_ref.parse().map_err(PyValueError::new_err)?;
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
            check_ref,
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
#[pyo3(signature = (svar1_dir, reference_path, chroms, contig_starts, contig_lens, output_dir, samples, ploidy, chunk_size, max_threads, long_allele_capacity, skip_out_of_scope, signatures, pos_per_contig, ref_bytes_per_contig, ref_offsets_per_contig, alt_bytes_per_contig, alt_offsets_per_contig, format_fields, format_src_dtypes, check_ref))]
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
    check_ref: String,
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
    let check_ref: crate::normalize::CheckRef = check_ref.parse().map_err(PyValueError::new_err)?;

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
                        check_ref,
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
    m.add_function(wrap_pyfunction!(run_slice_view, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(svar2_variant_stats, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_vcf_list_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_svar1_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(index_vcf, m)?)?;
    m.add_class::<crate::py_query::PyContigReader>()?;
    Ok(())
}

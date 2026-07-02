// src/lib.rs
use pyo3::prelude::*;
use rayon::prelude::*;

pub mod bits;
pub mod budget;
pub mod cost_model;
pub mod dense;
pub mod dense_merge;
pub mod error;
pub mod executor;
pub mod layout;
pub mod max_del;
pub mod merge;
pub mod meta;
pub mod monitor;
pub mod normalize;
pub mod nrvk;
pub mod orchestrator;
pub mod query;
pub mod rvk;
pub mod search;
pub mod streams;
pub mod types;
pub mod vcf_reader;
pub mod writer;

pub use orchestrator::process_chromosome;

//The Python Wrapper and resource allocator
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (vcf_path, reference_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608))]
fn run_conversion_pipeline(
    py: Python,
    vcf_path: String,
    reference_path: String,
    chroms: Vec<String>, // now taking a vector
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize, // default 25K variants/chunk — halves per-chunk plumbing overhead vs the old 10K
    ploidy: usize,
    max_threads: Option<usize>,  // accepts an optional integer from Python
    long_allele_capacity: usize, // default 8MB — old 100MB rarely flushed mid-run, blocking executor at finalize
) -> PyResult<()> {
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();

    let results: Vec<Result<(), crate::error::ConversionError>> = py.detach(|| {
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
        let results = pool.install(|| {
            chroms
                .par_iter()
                .map(|chrom| {
                    println!("==> Processing {}", chrom);
                    orchestrator::process_chromosome(
                        &vcf_path,
                        &reference_path,
                        chrom,
                        &output_dir,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        htslib_threads,
                        long_allele_capacity,
                    )
                })
                .collect::<Vec<_>>()
        });

        println!("Cohort Processing Complete.");
        results
    });

    for r in results {
        r.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    // All contigs converted — write the top-level meta.json describing the cohort.
    crate::meta::write_meta(
        std::path::Path::new(&output_dir),
        crate::meta::FORMAT_VERSION,
        &samples,
        &chroms,
        ploidy,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to write meta.json: {e}"))
    })?;

    Ok(())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_conversion_pipeline, m)?)?;
    Ok(())
}

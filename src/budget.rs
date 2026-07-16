// Thread-budget planning for the cohort orchestrator. Pure arithmetic, split out
// of the pyo3 entry point so the low-end / high-end / clamp branches are testable
// without side effects.

// 4 fixed OS threads per chrom: reader + executor + chunk_writer + long_allele_writer.
pub const PIPELINE_THREADS_PER_CHROM: usize = 4;
// Floor for HTSlib decode threads — below this the executor channel starves.
const MIN_HTSLIB_THREADS: usize = 2;
// Ceiling for HTSlib decode threads. Bumped 4→8 for single-/few-contig
// workloads with many idle cores: gdc's 16007-sample records mean very large
// BGZF blocks where extra decode threads still pay. Multi-contig runs clamp
// well below this via cores_per_chrom, so the bump only bites when cores are idle.
const MAX_HTSLIB_THREADS: usize = 8;
// Min viable allocation for one chrom end-to-end.
const MIN_THREADS_PER_CHROM: usize = PIPELINE_THREADS_PER_CHROM + MIN_HTSLIB_THREADS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPlan {
    pub concurrent_chroms: usize,
    pub htslib_threads: usize,
    // Cores left idle after the pipeline + htslib threads across all concurrent
    // chroms. For splittable VCF contigs this caps concurrent shard readers;
    // otherwise it sizes the reader-side processing pool used for bounded
    // normalization batches plus intra-chunk presence packing.
    pub processing_threads: usize,
}

/// Decide how many chromosomes to convert concurrently and how many HTSlib decode
/// threads each gets, given the detected/overridden core count and chromosome count.
/// Reserves 1 core for the OS + Python main thread.
pub fn plan_thread_budget(available_cores: usize, n_chroms: usize) -> ThreadPlan {
    let usable_cores = std::cmp::max(1, available_cores.saturating_sub(1));
    let n_chroms = std::cmp::max(1, n_chroms);

    if usable_cores < MIN_THREADS_PER_CHROM {
        // Low-end: run one chrom, pour remaining cores into HTSlib decode.
        let htslib = std::cmp::max(1, usable_cores.saturating_sub(PIPELINE_THREADS_PER_CHROM));
        let htslib = std::cmp::min(htslib, MAX_HTSLIB_THREADS);
        let processing = processing_threads(usable_cores, 1, htslib);
        ThreadPlan {
            concurrent_chroms: 1,
            htslib_threads: htslib,
            processing_threads: processing,
        }
    } else {
        // High-end: pick concurrency first (capped by chrom count), then redistribute.
        let max_concurrent_by_cores = usable_cores / MIN_THREADS_PER_CHROM;
        let concurrent = std::cmp::max(1, std::cmp::min(max_concurrent_by_cores, n_chroms));
        let cores_per_chrom = usable_cores / concurrent;
        let htslib_unclamped = cores_per_chrom.saturating_sub(PIPELINE_THREADS_PER_CHROM);
        let htslib = htslib_unclamped.clamp(MIN_HTSLIB_THREADS, MAX_HTSLIB_THREADS);
        let processing = processing_threads(usable_cores, concurrent, htslib);
        ThreadPlan {
            concurrent_chroms: concurrent,
            htslib_threads: htslib,
            processing_threads: processing,
        }
    }
}

/// Cores left idle after `concurrent` chroms each claim the pipeline threads plus
/// `htslib` decode threads. Floored at 1 so the processing pool always builds.
fn processing_threads(usable_cores: usize, concurrent: usize, htslib: usize) -> usize {
    let active = concurrent * (PIPELINE_THREADS_PER_CHROM + htslib);
    usable_cores.saturating_sub(active).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_end_one_chrom_min_htslib() {
        assert_eq!(
            plan_thread_budget(4, 8),
            ThreadPlan {
                concurrent_chroms: 1,
                htslib_threads: 1,
                processing_threads: 1,
            }
        );
    }

    #[test]
    fn test_single_core_machine() {
        assert_eq!(
            plan_thread_budget(1, 22),
            ThreadPlan {
                concurrent_chroms: 1,
                htslib_threads: 1,
                processing_threads: 1,
            }
        );
    }

    #[test]
    fn test_high_end_fans_out_and_clamps_htslib() {
        assert_eq!(
            plan_thread_budget(65, 22),
            ThreadPlan {
                concurrent_chroms: 10,
                htslib_threads: 2,
                processing_threads: 4,
            }
        );
    }

    #[test]
    fn test_concurrency_capped_by_chrom_count() {
        // Many cores but only 2 chroms → at most 2 concurrent.
        let plan = plan_thread_budget(64, 2);
        assert_eq!(plan.concurrent_chroms, 2);
        assert!(
            plan.htslib_threads >= MIN_HTSLIB_THREADS && plan.htslib_threads <= MAX_HTSLIB_THREADS
        );
    }

    #[test]
    fn test_htslib_never_exceeds_max() {
        // Huge core count, 1 chrom → htslib clamped at MAX_HTSLIB_THREADS.
        assert_eq!(
            plan_thread_budget(256, 1).htslib_threads,
            MAX_HTSLIB_THREADS
        );
    }

    #[test]
    fn test_high_end_single_chrom_uses_raised_htslib_cap() {
        // 33 cores → usable 32; 1 chrom → concurrent 1; cores_per_chrom 32;
        // htslib_unclamped = 32 - 4 = 28, clamped to [2, MAX_HTSLIB_THREADS=8] → 8.
        let plan = plan_thread_budget(33, 1);
        assert_eq!(plan.concurrent_chroms, 1);
        assert_eq!(plan.htslib_threads, 8);
    }

    #[test]
    fn test_processing_threads_absorb_idle_cores() {
        // 33 cores → usable 32; 1 chrom → concurrent 1; htslib 8 (Task 1 cap).
        // active = 1 * (PIPELINE_THREADS_PER_CHROM(4) + 8) = 12.
        // processing = max(1, 32 - 12) = 20.
        let plan = plan_thread_budget(33, 1);
        assert_eq!(plan.processing_threads, 20);
    }

    #[test]
    fn test_processing_threads_floored_at_one_when_saturated() {
        // 65 cores → usable 64; 22 chroms → concurrent 10; htslib 2.
        // active = 10 * (4 + 2) = 60. processing = max(1, 64 - 60) = 4.
        assert_eq!(plan_thread_budget(65, 22).processing_threads, 4);
        // Fully saturated: 7 cores → usable 6 == MIN_THREADS_PER_CHROM → high-end branch
        // (boundary: 6 < 6 is false), 1 chrom, htslib = clamp(6-4, 2, 8) = 2.
        // active = 1*(4+2)=6. processing = max(1, 6-6) = 1 (floored).
        assert_eq!(plan_thread_budget(7, 1).processing_threads, 1);
    }
}

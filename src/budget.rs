// Thread-budget planning for the cohort orchestrator. Pure arithmetic, split out
// of the pyo3 entry point so the low-end / high-end / clamp branches are testable
// without side effects.

// 4 fixed OS threads per chrom: reader + executor + chunk_writer + long_allele_writer.
pub const PIPELINE_THREADS_PER_CHROM: usize = 4;
// Floor for HTSlib decode threads — below this the executor channel starves.
const MIN_HTSLIB_THREADS: usize = 2;
// Ceiling for HTSlib decode threads — diminishing returns past 4 (BGZF block limits).
const MAX_HTSLIB_THREADS: usize = 4;
// Min viable allocation for one chrom end-to-end.
const MIN_THREADS_PER_CHROM: usize = PIPELINE_THREADS_PER_CHROM + MIN_HTSLIB_THREADS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPlan {
    pub concurrent_chroms: usize,
    pub htslib_threads: usize,
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
        ThreadPlan {
            concurrent_chroms: 1,
            htslib_threads: htslib,
        }
    } else {
        // High-end: pick concurrency first (capped by chrom count), then redistribute.
        let max_concurrent_by_cores = usable_cores / MIN_THREADS_PER_CHROM;
        let concurrent = std::cmp::max(1, std::cmp::min(max_concurrent_by_cores, n_chroms));
        let cores_per_chrom = usable_cores / concurrent;
        let htslib_unclamped = cores_per_chrom.saturating_sub(PIPELINE_THREADS_PER_CHROM);
        let htslib = htslib_unclamped.clamp(MIN_HTSLIB_THREADS, MAX_HTSLIB_THREADS);
        ThreadPlan {
            concurrent_chroms: concurrent,
            htslib_threads: htslib,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_end_one_chrom_min_htslib() {
        // 4 cores → usable 3 < 6 → 1 chrom, htslib = max(1, 3-4)=1 (clamped ≤4).
        assert_eq!(
            plan_thread_budget(4, 8),
            ThreadPlan {
                concurrent_chroms: 1,
                htslib_threads: 1
            }
        );
    }

    #[test]
    fn test_single_core_machine() {
        // 1 core → usable 1 → low-end → 1 chrom, htslib 1.
        assert_eq!(
            plan_thread_budget(1, 22),
            ThreadPlan {
                concurrent_chroms: 1,
                htslib_threads: 1
            }
        );
    }

    #[test]
    fn test_high_end_fans_out_and_clamps_htslib() {
        // 65 cores → usable 64; 64/6 = 10 concurrent (capped by n_chroms=22 → 10);
        // cores_per_chrom 64/10=6; htslib 6-4=2 clamped to [2,4] → 2.
        assert_eq!(
            plan_thread_budget(65, 22),
            ThreadPlan {
                concurrent_chroms: 10,
                htslib_threads: 2
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
}

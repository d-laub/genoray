// Thread-budget planning for the cohort orchestrator. Pure arithmetic, split out
// of the pyo3 entry point so the low-end / high-end / clamp branches are testable
// without side effects.

// 4 fixed OS threads per chrom: reader + executor + chunk_writer + long_allele_writer.
// Governs the MONOLITHIC reader path only (plan_thread_budget); the sharded
// path's per-contig demand is `1 + reader_workers` (see plan_sharded).
pub const PIPELINE_THREADS_PER_CHROM: usize = 4;
// Independent indexed VCF shard readers decompress in their worker thread.
// Giving each one an HTSlib background pool would multiply the process-wide
// thread budget by the shard count.
pub const SHARDED_VCF_HTSLIB_THREADS_PER_READER: usize = 0;
// Floor for HTSlib decode threads — below this the executor channel starves.
// Governs the MONOLITHIC reader path only (plan_thread_budget); the sharded
// path never allocates this pool (shard readers decompress inline).
const MIN_HTSLIB_THREADS: usize = 2;
// Ceiling for HTSlib decode threads. Bumped 4→8 for single-/few-contig
// workloads with many idle cores: gdc's 16007-sample records mean very large
// BGZF blocks where extra decode threads still pay. Multi-contig runs clamp
// well below this via cores_per_chrom, so the bump only bites when cores are idle.
// Governs the MONOLITHIC reader path only (plan_thread_budget); the sharded
// path never allocates this pool (shard readers decompress inline).
const MAX_HTSLIB_THREADS: usize = 8;
// Min viable allocation for one chrom end-to-end.
const MIN_THREADS_PER_CHROM: usize = PIPELINE_THREADS_PER_CHROM + MIN_HTSLIB_THREADS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPlan {
    pub concurrent_chroms: usize,
    pub htslib_threads: usize,
    // Indexed VCF shard readers per concurrent contig. Unlike
    // `processing_threads`, this budget reclaims the monolithic reader's
    // HTSlib pool because each shard decompresses inline.
    pub reader_workers: usize,
    // Cores left idle after the pipeline + htslib threads across all concurrent
    // chroms. This sizes the non-sharded reader-side processing pool used for
    // bounded normalization batches plus intra-chunk presence packing.
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
            reader_workers: reader_workers(usable_cores, 1),
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
            reader_workers: reader_workers(usable_cores, concurrent),
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

/// Per-contig worker count for the indexed/sharded VCF backend.
///
/// Shard readers replace the monolithic reader's HTSlib pool rather than
/// running alongside it, so split the usable process budget evenly across
/// active contigs and spend the remainder after their fixed pipeline threads.
fn reader_workers(usable_cores: usize, concurrent: usize) -> usize {
    let cores_per_chrom = usable_cores / concurrent.max(1);
    let worker_cost = 1 + SHARDED_VCF_HTSLIB_THREADS_PER_READER;
    cores_per_chrom
        .saturating_sub(PIPELINE_THREADS_PER_CHROM)
        .checked_div(worker_cost)
        .unwrap_or(0)
        .max(1)
}

// Peak-RSS coefficients from the scale-bench RAM law, fitted 2026-08-03:
//   peak_rss_mb ~ 932 + 0.01115*samples + 1.371*(w+pending)*chunk_bytes
//   R^2 = 0.9040, n = 44
// See docs/superpowers/specs/2026-08-03-svar2-tuned-load-balancing-design.md.
// These are load-bearing in production, not just in the bench: a bad refit
// becomes an OOM. Change them only alongside a refit that says so.
pub const RAM_BASE_MB: f64 = 932.0;
pub const RAM_PER_SAMPLE_MB: f64 = 0.01115;
pub const RAM_KAPPA: f64 = 1.371;

/// Inputs to the sharded-VCF concurrency plan. Every field is data the caller
/// already has before opening a single record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanInputs {
    pub usable_cores: usize,
    pub n_contigs: usize,
    pub n_samples: usize,
    /// Bytes of one FULL dense chunk:
    /// `chunk_size * (n_samples*ploidy/8 + n_format_fields*n_samples*4)`.
    pub chunk_bytes: u64,
    /// `None` means the caller declined a budget; only the core bound applies.
    pub max_mem_bytes: Option<u64>,
    pub reader_workers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardedPlan {
    pub concurrent_chroms: usize,
    pub reader_workers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanError {
    /// The budget cannot fit the cohort baseline plus one contig's chunks.
    InsufficientMemory { needed_mb: f64, budget_mb: f64 },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::InsufficientMemory {
                needed_mb,
                budget_mb,
            } => write!(
                f,
                "max_mem is {budget_mb:.0} MB but converting this cohort needs \
                 at least {needed_mb:.0} MB for one concurrent contig; raise \
                 max_mem or lower chunk_size"
            ),
        }
    }
}

/// Plan contig concurrency for the sharded VCF reader.
///
/// Per-contig CPU demand is `1 + reader_workers`: one executor
/// (`run_compute_engine`, a serial recv loop, pegged at ~100% of one core) plus
/// the shard readers. It is NOT `PIPELINE_THREADS_PER_CHROM + htslib_threads`
/// -- the dispatcher and both writers are nearly always blocked (a measured
/// 22-contig run put 16 threads on 2.02 cores), and the HTSlib decode pool is
/// `SHARDED_VCF_HTSLIB_THREADS_PER_READER` = 0 on this path because shard
/// readers decompress inline.
///
/// Memory bounds concurrency independently: each concurrent contig holds
/// `reader_workers + pending` chunks in flight, where `pending` is the reorder
/// buffer's structural floor `reader_workers - 1` (the units ahead of the head
/// keep everything they produce buffered even with perfectly balanced readers).
pub fn plan_sharded(inp: PlanInputs) -> Result<ShardedPlan, PlanError> {
    let w = inp.reader_workers.max(1);
    let n_contigs = inp.n_contigs.max(1);
    let usable = inp.usable_cores.max(1);

    let core_bound = (usable / (1 + w)).max(1);

    let cc = match inp.max_mem_bytes {
        None => std::cmp::min(core_bound, n_contigs),
        Some(budget) => {
            let budget_mb = budget as f64 / 1e6;
            let baseline_mb = RAM_BASE_MB + RAM_PER_SAMPLE_MB * inp.n_samples as f64;
            let pending = w.saturating_sub(1);
            let per_contig_mb = RAM_KAPPA * (w + pending) as f64 * (inp.chunk_bytes as f64 / 1e6);
            let headroom_mb = budget_mb - baseline_mb;
            if headroom_mb < per_contig_mb {
                return Err(PlanError::InsufficientMemory {
                    needed_mb: baseline_mb + per_contig_mb,
                    budget_mb,
                });
            }
            let mem_bound = (headroom_mb / per_contig_mb).floor() as usize;
            std::cmp::min(std::cmp::min(core_bound, n_contigs), mem_bound.max(1))
        }
    };

    Ok(ShardedPlan {
        concurrent_chroms: cc,
        reader_workers: w,
    })
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
                reader_workers: 1,
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
                reader_workers: 1,
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
                reader_workers: 2,
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
    fn test_sharded_vcf_reclaims_unused_htslib_budget_for_reader_workers() {
        // Sharded VCF readers each use one inline HTSlib thread, so the separate
        // 8-thread HTSlib decode pool is not active on this backend. A 16-core,
        // one-contig run therefore has 15 usable cores: 4 fixed pipeline threads
        // plus 11 independent shard readers.
        let plan = plan_thread_budget(16, 1);
        assert_eq!(plan.reader_workers, 11);
        assert_eq!(
            plan.concurrent_chroms
                * (PIPELINE_THREADS_PER_CHROM
                    + plan.reader_workers * (1 + SHARDED_VCF_HTSLIB_THREADS_PER_READER)),
            15
        );
    }

    #[test]
    fn test_sharded_vcf_reader_workers_are_bounded_across_concurrent_contigs() {
        // 65 cores → 64 usable; 10 concurrent contigs × (4 fixed + 2 readers)
        // = 60 active sharded-path threads, leaving four cores of headroom.
        let plan = plan_thread_budget(65, 22);
        let active = plan.concurrent_chroms
            * (PIPELINE_THREADS_PER_CHROM
                + plan.reader_workers * (1 + SHARDED_VCF_HTSLIB_THREADS_PER_READER));
        assert_eq!(plan.reader_workers, 2);
        assert!(active <= 64);
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

    // 48 cores -> 47 usable. w=2 -> demand 3/contig -> 15 concurrent, under
    // the 22 available. The OLD planner returned 7 here, because it charged
    // 6 cores per contig for 4 mostly-blocked pipeline threads plus an
    // HTSlib pool the sharded path never allocates.
    #[test]
    fn core_bound_concurrency() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 4_000,
            chunk_bytes: 10_937_000,
            max_mem_bytes: None,
            reader_workers: 2,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 15,
                reader_workers: 2
            }
        );
    }

    // Fewer contigs than cores allow: never spawn a pipeline with no contig.
    #[test]
    fn contig_count_bounds_concurrency() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 4,
            n_samples: 4_000,
            chunk_bytes: 10_937_000,
            max_mem_bytes: None,
            reader_workers: 2,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 4,
                reader_workers: 2
            }
        );
    }

    // The memory constraint must actually bind, or it is decoration.
    // S=500,000, ploidy 2, no FORMAT fields, chunk_size 25,000:
    //   chunk_bytes = 25_000 * (500_000*2/8) = 3.125e9 B = 3125 MB
    //   base        = 932 + 0.01115*500_000 = 6507 MB
    //   per-contig  = 1.371 * (2 + 1) * 3125 = 12853.125 MB
    //   budget      = 52428 MB  ->  (52428 - 6507)/12853.125 = 3.57 -> 3
    // The core bound alone would have allowed 15.
    #[test]
    fn memory_bound_beats_core_bound_at_biobank_scale() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 500_000,
            chunk_bytes: 3_125_000_000,
            max_mem_bytes: Some(52_428 * 1_000_000),
            reader_workers: 2,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 3,
                reader_workers: 2
            }
        );
    }

    // A budget below the cohort baseline cannot fit even one contig. Failing
    // loudly beats planning cc=0 (which dispatches nothing and "succeeds"
    // with an empty store) or cc=1 (which OOMs).
    #[test]
    fn budget_below_baseline_is_an_error() {
        let err = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 500_000,
            chunk_bytes: 3_125_000_000,
            max_mem_bytes: Some(5_000 * 1_000_000),
            reader_workers: 2,
        })
        .unwrap_err();
        match err {
            PlanError::InsufficientMemory {
                needed_mb,
                budget_mb,
            } => {
                assert!(needed_mb > budget_mb);
                assert!((budget_mb - 5_000.0).abs() < 1.0);
            }
        }
    }

    // `from_vcf` and `from_vcf_list` now share ONE `max_mem` meaning (a
    // whole-process budget), so there is no more mix-up for the message to
    // flag -- it just needs to name the two actionable remedies.
    #[test]
    fn insufficient_memory_message_names_remedies() {
        let err = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 1_000,
            max_mem_bytes: Some(1_000_000), // 1 MB -- far below the cohort baseline
            reader_workers: 2,
        })
        .unwrap_err();
        let msg = err.to_string();
        // "max_mem" alone would pass regardless (it's also in the message's
        // opening clause) -- pin the actual remedy phrase so this asserts
        // something load-bearing.
        assert!(
            msg.contains("raise max_mem or lower chunk_size"),
            "message = {msg:?}"
        );
    }

    // Degenerate hardware must still produce a runnable plan.
    #[test]
    fn single_core_single_contig_still_runs() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 1,
            n_contigs: 1,
            n_samples: 250,
            chunk_bytes: 64_000,
            max_mem_bytes: None,
            reader_workers: 4,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 1,
                reader_workers: 4
            }
        );
    }

    // Zero contigs is a caller bug, not a plan: clamp rather than divide by it.
    #[test]
    fn zero_contigs_clamps_to_one() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 0,
            n_samples: 250,
            chunk_bytes: 64_000,
            max_mem_bytes: None,
            reader_workers: 2,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 1,
                reader_workers: 2
            }
        );
    }

    // Per-contig memory is `kappa*(w+(w-1))*chunk_bytes`, so it grows roughly
    // linearly in `w`: raising the reader count from the fitted default of 3
    // to 16 costs 6.2x (kappa*(w+w-1) is 31 vs 5) and can turn a plan that
    // fits into `InsufficientMemory`. This is why `DEFAULT_READER_WORKERS` is
    // a fitted constant rather than something a caller or a runtime probe
    // dials up freely -- the memory law, not just throughput, bounds it.
    //
    // n_samples=1_000, chunk_bytes=10_000_000 (10 MB):
    //   baseline    = 932 + 0.01115*1_000        = 943.15 MB
    //   per-contig  = 1.371 * (w + (w-1)) * 10 MB
    //     w=3  -> 1.371*5*10  =  68.55 MB -> needs  1011.70 MB
    //     w=16 -> 1.371*31*10 = 425.01 MB -> needs  1368.16 MB
    //   budget = 1_200 MB: fits w=3, rejects w=16.
    #[test]
    fn a_high_worker_count_can_exceed_a_budget_the_default_fits() {
        let inp = PlanInputs {
            usable_cores: 47,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_200_000_000),
            reader_workers: 16,
        };
        assert!(matches!(
            plan_sharded(inp),
            Err(PlanError::InsufficientMemory { .. })
        ));
        assert_eq!(
            plan_sharded(PlanInputs {
                reader_workers: 3,
                ..inp
            })
            .unwrap(),
            ShardedPlan {
                concurrent_chroms: 1,
                reader_workers: 3,
            }
        );
    }
}

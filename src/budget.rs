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

/// Fitted peak-RSS coefficients for one conversion backend:
///   peak_rss_mb ~ base_mb + per_sample_mb*samples + kappa*(w+pending)*chunk_bytes
///
/// These are load-bearing in production, not just in the bench: a bad refit
/// becomes an OOM. Change a law only alongside a refit that says so, and
/// record that refit's R^2 and n in the constant's doc comment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamLaw {
    pub base_mb: f64,
    pub per_sample_mb: f64,
    pub kappa: f64,
}

impl RamLaw {
    /// Sharded VCF path. Fitted 2026-08-03, R^2 = 0.9040, n = 44.
    /// See docs/superpowers/specs/2026-08-03-svar2-tuned-load-balancing-design.md.
    pub const VCF: RamLaw = RamLaw {
        base_mb: 932.0,
        per_sample_mb: 0.01115,
        kappa: 1.371,
    };

    /// PGEN path, re-fit on the bounded reader (issue #155, PR #154).
    /// Fitted 2026-08-07, n = 18.
    /// See docs/superpowers/plans/results/2026-08-07-pgen-ram-law-refit.md.
    ///
    /// NOT comparable coefficient-by-coefficient with `RamLaw::VCF`: the two
    /// corpora come from different generators (vcfixture bulk vs
    /// scale_corpus.py), so each law is valid only for its own backend.
    ///
    /// Fitted on a corpus with multiallelic_rate 0.0, so this law is not
    /// claimed to cover multiallelic-heavy cohorts.
    ///
    /// `fit_ram_law`'s unconstrained 3-regressor fit on this data gives
    /// R^2 = 0.5353 -- lower than the precedent's 0.8872, but because of the
    /// sweep's DESIGN, not the code under test: 6 of the 18 rows are a
    /// `concurrent_chroms` ladder (S=4,000, chunk fixed), and `_ram_rows` /
    /// `fit_ram_law` never carry `concurrent_chroms` in the regressor, so
    /// those 6 rows share one identical design-matrix row while their
    /// measured RSS spans 1860.2 -> 4416.1 MB (2.37x). That replicate-X
    /// spread caps R^2 by construction; it is not evidence of worse
    /// behaviour than the precedent law.
    ///
    /// `per_sample_mb` is the 95% UPPER CONFIDENCE BOUND, not the point
    /// estimate -- and is NOT slack to be tuned away, for the same reason
    /// `kappa` below is a bound rather than a rate. The unconstrained point
    /// estimate is -0.011099 (95% CI [-0.03540914, +0.01321036]:
    /// indistinguishable from zero), which is exactly the outcome Tasks 1-4
    /// were built to cause -- the per-sample reader term is gone. Shipping
    /// the negative point estimate is impossible: at S=500,000 it would put
    /// the baseline at -2,335 MB. Taking the upper bound (+0.013210) is the
    /// conservative choice consistent with that non-result.
    ///
    /// `kappa` = 11.987, SE 4.780035, 95% CI [1.853822, 22.120264] under a
    /// residualised fit that partials out the cc-collinearity below -- this
    /// CI EXCLUDES zero, unlike the precedent's [-9.99, +23.68].
    ///
    /// The margin's real provenance has to be stated plainly, because it
    /// cannot be reasoned about from the coefficients alone: `fit_ram_law`
    /// fits `kappa` cc-blind (its chunk regressor is
    /// `(reader_workers + pending) * chunk_bytes`, never multiplied by
    /// `concurrent_chroms`, and every row here has reader_workers=1,
    /// pending=0), so `kappa` absorbs this sweep's dominant cc=8 into
    /// itself; `plan_sharded` then multiplies by `cc` AGAIN when it charges
    /// `per_contig_mb`. The resulting over-charge is what makes this law
    /// over-predict -- it is an artifact of the fit's specification, not a
    /// deliberately chosen safety factor. Tracked as issue #158.
    ///
    /// It ships anyway because the law OVER-predicts memory at all 18
    /// measured points the way `plan_sharded` evaluates them (worst-case
    /// margin +938.44 MB / 1.359x at S=4,000, chunk_bytes=3.125 MB, cc=8;
    /// the previously shipped law's worst case was +621 MB / 1.159x). The
    /// failure mode this margin trades away is under-utilization and a
    /// spurious `PlanError::InsufficientMemory`, never an OOM.
    ///
    /// A measured per-contig constant this 3-coefficient form cannot
    /// represent: regressing the cc-ladder rows alone (n=6, chunk fixed)
    /// gives 107.05 MB/contig (SE 28.23, 95% CI [28.67, 185.44],
    /// R^2 = 0.782), and the known `RAW_STAGE_BYTES + MASK_STAGE_BYTES` =
    /// 128 MiB = 134.2177 MB/contig falls inside that CI -- so the
    /// per-assembler constant this form is missing is now directly
    /// measured, not just inferred. A single multiplicative `kappa` cannot
    /// price it across the swept chunk range: covering it at
    /// chunk_bytes=3.125 MB needs kappa ~= 42.95, which then overshoots
    /// 16.2x at chunk_bytes=249.98 MB, cc=8. Tracked as issue #158.
    ///
    /// Validity domain:
    /// - `cc <= 8` is ENFORCED IN CODE, not just documented: `lib.rs:589`
    ///   clamps the plan to `PGEN_MAX_CONCURRENT` (`= 8`, below); `cc > 8`
    ///   is reachable only through the `GENORAY_CONCURRENT_CHROMS` bench
    ///   override, which is how this sweep's ladder rows reached
    ///   11/16/22. That clamp is what keeps the under-priced
    ///   small-chunk/high-cc regime unreachable by users.
    /// - S in {4,000, 32,000, 128,000}; chunk_bytes 3.125-250 MB;
    ///   `reader_workers == 1` and `pending == 0` in every row; 22
    ///   contigs; one node (carter-cn-04); one profile; multiallelic_rate
    ///   0.0.
    /// - No FORMAT/dosage fields were requested in this sweep, so the law
    ///   is scoped to the no-FORMAT path (issue #156).
    /// - `per_sample_mb` is extrapolated ~3.9x beyond the largest measured
    ///   cohort (128,000 -> 500,000) to reach biobank scale -- an
    ///   improvement on the precedent's 15.6x, but still an extrapolation.
    pub const PGEN: RamLaw = RamLaw {
        base_mb: 3197.585732182869,
        per_sample_mb: 0.013210358143200408,
        kappa: 11.987043102903698,
    };
}

/// Measured ceiling on useful PGEN contig concurrency: `pgenlib` holds the
/// GIL through decode, so past this point extra concurrent contigs buy no
/// wall time while still costing memory. Measured 2026-08-05 on carter-cn-04
/// (48 CPUs / 64 GB) at
/// one corpus shape (S=4,000, V=1,000,000, 22 contigs): wall time fell
/// 31.20 -> 12.81 -> 10.18 s at cc = 1, 4, 8, then stayed within +/-2.4%
/// through cc = 22 (cc=16->22 was actually +1.8%, slightly worse) while RSS
/// trends upward (+12.7% cc=8->22, non-monotonically -- it actually falls
/// 3917->3586 MB from cc=8->11 before rising again to 4416 MB by cc=22) for
/// no further wall-time benefit. See
/// docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit.md. NOT a
/// guess; if a future pgenlib release drops the GIL through decode,
/// re-measure before raising it.
///
/// CAVEAT: the sweep that produced this value (commit 80b5fd8) ran BEFORE
/// `processing_threads_for` was wired onto the PGEN path (commit a39ebcb),
/// so all 12 fitted/measured rows ran under `plan_thread_budget`'s
/// `processing_threads = 5`, not the shipped `47 - 2*cc` (= 31 at cc=8) merge
/// tail. Memory is unaffected either way -- both merge-tail consumers
/// (`merge.rs`'s `TILE_RAM_BUDGET_BYTES`, a whole-stage budget divided by
/// thread count, and `dense_merge`'s single output buffer split across
/// threads) are thread-count-flat -- but the wall-time knee above was not
/// measured under the thread configuration this constant now gates in
/// production. Re-measure with the shipped tail-pool sizing before trusting
/// the wall-time numbers precisely, not just the memory ones.
pub const PGEN_MAX_CONCURRENT: usize = 8;

/// Inputs to the sharded-VCF concurrency plan. Every field is data the caller
/// already has before opening a single record.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Which backend's fitted peak-RSS law to plan against.
    pub ram: RamLaw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardedPlan {
    pub concurrent_chroms: usize,
    pub reader_workers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanError {
    /// The budget cannot fit the cohort baseline plus one contig's chunks.
    ///
    /// `baseline_mb` is carried separately from `needed_mb` (rather than
    /// leaving the caller to re-derive it) because the two failure shapes
    /// need different advice: when `budget_mb` doesn't even cover
    /// `baseline_mb`, `chunk_size` is powerless -- the cohort-baseline term
    /// alone (fixed cost + per-sample cost, independent of `chunk_size`)
    /// already exceeds the budget, so only a larger `max_mem` or a smaller
    /// cohort can help.
    InsufficientMemory {
        needed_mb: f64,
        budget_mb: f64,
        baseline_mb: f64,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::InsufficientMemory {
                needed_mb,
                budget_mb,
                baseline_mb,
            } => {
                if budget_mb < baseline_mb {
                    write!(
                        f,
                        "max_mem is {budget_mb:.0} MB but this cohort's baseline \
                         memory alone is {baseline_mb:.0} MB, before any \
                         concurrent contig's chunk buffers -- only a larger \
                         max_mem or a smaller cohort can help"
                    )
                } else {
                    write!(
                        f,
                        "max_mem is {budget_mb:.0} MB but converting this cohort needs \
                         at least {needed_mb:.0} MB for one concurrent contig; raise \
                         max_mem or lower chunk_size"
                    )
                }
            }
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
            let baseline_mb = inp.ram.base_mb + inp.ram.per_sample_mb * inp.n_samples as f64;
            let pending = w.saturating_sub(1);
            let per_contig_mb =
                inp.ram.kappa * (w + pending) as f64 * (inp.chunk_bytes as f64 / 1e6);
            let headroom_mb = budget_mb - baseline_mb;
            if headroom_mb < per_contig_mb {
                return Err(PlanError::InsufficientMemory {
                    needed_mb: baseline_mb + per_contig_mb,
                    budget_mb,
                    baseline_mb,
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

/// Cores left after the planned concurrency's executors and readers.
///
/// Sizes the merge tail — `merge.rs`'s var_key gather pool and
/// `dense_merge`'s bit-transpose — which runs per contig once its pipeline
/// drains. Both backends use this so the tail is sized against the
/// concurrency actually dispatched, not against a different planner's
/// hypothetical one.
///
/// Floors at 1: `rayon::ThreadPoolBuilder::num_threads(0)` means "use the
/// global default", not "no threads", so returning 0 here would silently
/// oversubscribe rather than serialize.
pub fn processing_threads_for(usable_cores: usize, cc: usize, w: usize) -> usize {
    usable_cores.saturating_sub(cc * (1 + w)).max(1)
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
            ram: RamLaw::VCF,
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
            ram: RamLaw::VCF,
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
            ram: RamLaw::VCF,
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
            ram: RamLaw::VCF,
        })
        .unwrap_err();
        match err {
            PlanError::InsufficientMemory {
                needed_mb,
                budget_mb,
                baseline_mb,
            } => {
                assert!(needed_mb > budget_mb);
                assert!((budget_mb - 5_000.0).abs() < 1.0);
                assert!(budget_mb < baseline_mb);
            }
        }
    }

    // This budget (1 MB) is far below the cohort baseline (~943 MB), so
    // `chunk_size` is powerless here -- only `max_mem` or a smaller cohort
    // can help. The message must say so, and must NOT claim `chunk_size`
    // would help (that would be actionable-sounding but false in this
    // regime).
    #[test]
    fn insufficient_memory_message_names_remedies() {
        let err = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 1_000,
            max_mem_bytes: Some(1_000_000), // 1 MB -- far below the cohort baseline
            reader_workers: 2,
            ram: RamLaw::VCF,
        })
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("max_mem"), "message = {msg:?}");
        assert!(
            !msg.contains("chunk_size"),
            "chunk_size cannot help when the budget is below baseline; \
             message = {msg:?}"
        );
    }

    // The budget-above-baseline-but-below-needed case is the one where
    // `chunk_size` genuinely IS an actionable remedy alongside `max_mem`, so
    // the message must still offer both there.
    #[test]
    fn insufficient_memory_message_names_both_remedies_when_baseline_fits() {
        let err = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_200_000_000), // covers baseline (~943 MB), not per-contig
            reader_workers: 16,
            ram: RamLaw::VCF,
        })
        .unwrap_err();
        let msg = err.to_string();
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
            ram: RamLaw::VCF,
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
            ram: RamLaw::VCF,
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
            ram: RamLaw::VCF,
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

    #[test]
    fn ram_law_vcf_reproduces_the_fitted_coefficients() {
        assert_eq!(RamLaw::VCF.base_mb, 932.0);
        assert_eq!(RamLaw::VCF.per_sample_mb, 0.01115);
        assert_eq!(RamLaw::VCF.kappa, 1.371);
    }

    #[test]
    fn processing_threads_for_returns_the_cores_left_after_executors_and_readers() {
        // 47 usable, 11 contigs at (1 executor + 3 readers) = 44 spent, 3 left.
        assert_eq!(processing_threads_for(47, 11, 3), 3);
        // PGEN shape: w = 1, so each contig costs 2.
        assert_eq!(processing_threads_for(47, 22, 1), 3);
    }

    #[test]
    fn processing_threads_for_floors_at_one_when_oversubscribed() {
        // Never 0: the merge tail must always get a usable thread count, and a
        // rayon pool of 0 threads panics at build time.
        assert_eq!(processing_threads_for(4, 8, 3), 1);
        assert_eq!(processing_threads_for(1, 1, 1), 1);
    }

    #[test]
    // The whole point is asserting on RamLaw::PGEN's const fields, so clippy
    // sees a compile-time-constant condition; that's the guard, not a bug.
    #[allow(clippy::assertions_on_constants)]
    fn ram_law_pgen_is_a_usable_law() {
        // Guards against a placeholder shipping: a zero kappa would make the
        // memory bound vacuous and silently restore the unbounded planning
        // this whole change exists to remove.
        assert!(RamLaw::PGEN.kappa > 0.0, "kappa must be positive");
        assert!(RamLaw::PGEN.base_mb > 0.0, "baseline must be positive");
        assert!(RamLaw::PGEN.per_sample_mb >= 0.0);
    }

    #[test]
    fn pgen_memory_bound_actually_binds() {
        // A budget that fits the baseline plus two contigs (with headroom to
        // 2.5, so floor(2.5) = 2 regardless of float representation of the
        // fitted coefficients) must plan 2, not the core bound. Uses
        // RamLaw::PGEN's real coefficients, so it fails if a future refit
        // makes the law nonsensical.
        let chunk_bytes = 100_000_000u64;
        let baseline_mb = RamLaw::PGEN.base_mb + RamLaw::PGEN.per_sample_mb * 1000.0;
        let per_contig_mb = RamLaw::PGEN.kappa * 1.0 * (chunk_bytes as f64 / 1e6);
        let budget = ((baseline_mb + 2.5 * per_contig_mb) * 1e6) as u64;

        let plan = plan_sharded(PlanInputs {
            usable_cores: 64,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes,
            max_mem_bytes: Some(budget),
            reader_workers: 1,
            ram: RamLaw::PGEN,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 2);
    }

    #[test]
    fn pgen_budget_too_small_for_one_contig_is_an_error_not_a_silent_cc_of_one() {
        // Below the baseline + one contig, planning must FAIL. Clamping to
        // cc=1 and proceeding would OOM at the exact scale the budget exists
        // to protect, and would do it after writing a partial store.
        //
        // This budget (1 MB) is far below even the cohort baseline
        // (~16,408 MB at S=1,000,000 under RamLaw::PGEN), so this exercises
        // the baseline-dominated branch: `chunk_size` cannot help here, only
        // `max_mem` (or a smaller cohort) can.
        let err = plan_sharded(PlanInputs {
            usable_cores: 64,
            n_contigs: 22,
            n_samples: 1_000_000,
            chunk_bytes: 10_000_000_000,
            max_mem_bytes: Some(1_000_000),
            reader_workers: 1,
            ram: RamLaw::PGEN,
        })
        .unwrap_err();
        match err {
            PlanError::InsufficientMemory {
                needed_mb,
                budget_mb,
                baseline_mb,
            } => {
                assert!(
                    needed_mb > budget_mb,
                    "needed {needed_mb} must exceed budget {budget_mb}"
                );
                assert!(budget_mb < baseline_mb, "budget must be baseline-dominated");
            }
        }
        // The message must name the one knob that can actually help here,
        // and must NOT dangle `chunk_size` as a false remedy.
        let msg = err.to_string();
        assert!(msg.contains("max_mem"), "{msg}");
        assert!(!msg.contains("chunk_size"), "{msg}");
    }

    #[test]
    fn plan_sharded_uses_the_supplied_ram_law_not_a_global() {
        // Two identical inputs differing ONLY in the law: a law with twice the
        // kappa must halve the memory-bound concurrency. If plan_sharded still
        // read module constants, both would return the same cc. Budget is
        // sized (3 GB) so the memory bound -- not the core bound of 32 -- is
        // the binding constraint in both arms.
        let base = PlanInputs {
            usable_cores: 64,
            n_contigs: 32,
            n_samples: 1_000,
            chunk_bytes: 100_000_000,
            max_mem_bytes: Some(3_000_000_000),
            reader_workers: 1,
            ram: RamLaw {
                base_mb: 1000.0,
                per_sample_mb: 0.0,
                kappa: 1.0,
            },
        };
        let doubled = PlanInputs {
            ram: RamLaw {
                kappa: 2.0,
                ..base.ram
            },
            ..base
        };
        let a = plan_sharded(base).unwrap().concurrent_chroms;
        let b = plan_sharded(doubled).unwrap().concurrent_chroms;
        assert_eq!(a, 2 * b, "cc must scale inversely with kappa: {a} vs {b}");
    }
}

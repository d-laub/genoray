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

/// Denominator of the usable-core fraction reserved for the per-contig merge
/// tail: `ceil(usable / MERGE_RESERVE_DIV)` cores never go to readers.
///
/// `processing_threads_for` sizes `merge.rs`'s var_key gather pool and
/// `dense_merge`'s bit-transpose from whatever the readers leave over.
/// Spending every leftover core on readers floors that pool at 1 and gives
/// back the 2.77-2.82x merge-tiling win from commit c49d1d7.
///
/// STARTING VALUE -- set it from the sweep in
/// `docs/superpowers/specs/2026-09-05-svar2-reader-frontier-design.md`
/// section D and record the measurement here when you do.
pub const MERGE_RESERVE_DIV: usize = 4;

/// Target readers per concurrent contig when choosing `concurrent_chroms`.
///
/// Depth is preferred over breadth: each extra concurrent contig costs a
/// dedicated executor core and a full `per_contig_mb` of RAM while buying no
/// extra reader cores, because the readers are CPU-saturated and total read
/// throughput tracks total reader cores however they are partitioned. The old
/// argument for shallow depth ("surplus readers steal cores from other
/// contigs' executors") was really an argument about the reorder frontier,
/// which `shard::plan_unit_count` now fixes.
///
/// STARTING VALUE -- see the same spec, section D.
pub const W_TARGET: usize = 8;

/// Chunks of reorder backlog `shard_exec`'s collector may hold before non-head
/// readers park. Floored at 2 in [`pending_budget_bytes`] -- a one-chunk
/// budget serializes the frontier back to the head, which is the pathology
/// this whole change exists to remove.
///
/// STARTING VALUE -- see the same spec, section D.
pub const PENDING_BUDGET_CHUNKS: u64 = 8;

/// Byte ceiling for `shard_exec`'s reorder backlog -- the collector's
/// `PendingBacklog` map ALONE. This is the number handed to
/// `shard_exec::run`; it is NOT the pipeline's total in-flight bytes. Use
/// [`in_flight_budget_bytes`] for memory planning.
///
/// Deliberately independent of `max_mem`: `plan_sharded` consumes this budget
/// to choose `concurrent_chroms`, and `max_mem` is what bounds
/// `concurrent_chroms`, so deriving one from the other is circular. Instead
/// the budget is a fixed multiple of the chunk, and `max_mem` constrains
/// concurrency through the law in `plan_sharded`.
///
/// `chunk_bytes` must be the `resident_chunk_size`-narrowed value (see
/// `lib.rs`), not `chunk_size * per_variant_bytes`: `BitGrid3::zeros` is a
/// calloc, so nominal chunk bytes are address space, not RSS.
pub fn pending_budget_bytes(chunk_bytes: u64) -> u64 {
    chunk_bytes.saturating_mul(PENDING_BUDGET_CHUNKS.max(2))
}

/// Whether the path being planned actually runs `shard_exec`'s reorder
/// frontier -- i.e. whether [`pending_budget_bytes`] is a ceiling that exists.
///
/// An explicit discriminator rather than something derived from
/// [`PlanInputs::ram`]: `RamLaw` is a struct of fitted `f64` coefficients, and
/// deciding a control-flow question by comparing floats for identity would
/// break silently the first time a law is refitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacklogGate {
    /// `shard_exec::run` is handed [`pending_budget_bytes`], and `Frontier`
    /// parks producers at that ceiling -- so the ceiling is real memory the
    /// plan must afford. The sharded VCF path (`orchestrator.rs`).
    Enforced,
    /// `shard_exec::run` is handed `u64::MAX`: no backlog ceiling is enforced,
    /// so no memory is reserved for one. The PGEN path (`orchestrator.rs`),
    /// which also pins `reader_workers = 1` -- one producer per contig means
    /// chunks arrive in order and the collector's `PendingBacklog` never fills.
    ///
    /// Both halves of that matter. Charging nothing here is sound BECAUSE the
    /// path is single-producer; re-enabling PGEN sub-contig sharding while
    /// still passing `u64::MAX` would make this term under-count real peak
    /// RSS, which is the one direction that risks an OOM. `debug_assert`ed in
    /// [`in_flight_budget_bytes`].
    Disabled,
}

/// Chunk-shaped bytes one contig can hold in flight, for the memory law.
///
/// `Frontier` bounds only the collector's `PendingBacklog`. Two other places
/// hold assembled chunks at the same time, and pricing only the backlog
/// under-counts real peak RSS:
///
/// - the collector's `PendingBacklog` map: [`pending_budget_bytes`], enforced
///   only when `gate` is [`BacklogGate::Enforced`] (see that type);
/// - `shard_exec`'s bounded result channel `tx_res`, capacity `workers * 2`,
///   enforced by the channel itself on BOTH backends;
/// - each reader's own working chunk -- including the assembled chunk a parked
///   producer is holding, since `admit` is called BEFORE `tx_res.send`. That
///   term is `w` chunks and is already carried by `ram.kappa * w * chunk_MB`
///   in `plan_sharded`, so it is deliberately NOT repeated here.
///
/// So this returns the channel capacity, plus the backlog ceiling when one is
/// enforced. Adding the channel term makes the planner strictly more
/// conservative; it is not a refit of `RamLaw`'s fitted coefficients, and
/// neither is gating the backlog term -- the `in_flight` term was added on top
/// of the fitted laws as deliberate extra conservatism.
pub fn in_flight_budget_bytes(chunk_bytes: u64, workers: usize, gate: BacklogGate) -> u64 {
    debug_assert!(
        gate == BacklogGate::Enforced || workers <= 1,
        "BacklogGate::Disabled is only sound single-producer; got workers={workers}"
    );
    let backlog = match gate {
        BacklogGate::Enforced => pending_budget_bytes(chunk_bytes),
        BacklogGate::Disabled => 0,
    };
    backlog.saturating_add(chunk_bytes.saturating_mul(2u64.saturating_mul(workers.max(1) as u64)))
}

/// Cores available to executors and readers after the merge-tail reserve.
/// Floored at 1 so a single-core host still plans.
pub fn reader_pool_cores(usable_cores: usize) -> usize {
    let usable = usable_cores.max(1);
    usable
        .saturating_sub(usable.div_ceil(MERGE_RESERVE_DIV))
        .max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPlan {
    pub concurrent_chroms: usize,
    pub htslib_threads: usize,
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

/// HTSlib decode threads for the MONOLITHIC reader when the SHARDED planner
/// chose the concurrency.
///
/// `process_chromosome` falls back to the single monolithic `VcfRecordSource`
/// whenever a contig's shard plan yields at most one shard, and takes it for
/// EVERY contig under `Record`/`Variant` overlap (POS-ownership would drop kept
/// records). The concurrency those contigs run at came from `plan_sharded`,
/// which bills a contig `1 + reader_workers` cores -- an executor plus its
/// readers -- while [`plan_thread_budget`] bills the monolithic shape
/// `PIPELINE_THREADS_PER_CHROM + htslib_threads` at a `concurrent_chroms` of
/// its own. Taking the decode count from that other planner is what
/// oversubscribes the allocation: at 48 cores and 22 contigs the sharded
/// planner picks `cc = 11`, and the monolithic figure for `cc = 7` is then
/// spent 11 times over (#152).
///
/// So spend the contig's OWN billed budget instead: of the `1 + w` cores
/// `plan_sharded` reserved for it, one is the executor and one is the single
/// reader thread it actually runs, leaving `w - 1` for htslib decode. The
/// fallback then costs exactly what was planned for it, whatever `w` is.
///
/// Zero is a legitimate result (`w = 1`) and means "no decode pool" --
/// `vcf_reader::open_vcf` skips `set_threads` entirely at 0, which is the
/// htslib default rather than an error.
pub fn fallback_htslib_threads(reader_workers: usize) -> usize {
    reader_workers.saturating_sub(1)
}

/// Cores left idle after `concurrent` chroms each claim the pipeline threads plus
/// `htslib` decode threads. Floored at 1 so the processing pool always builds.
///
/// `pub(crate)` so a caller that overrides the planner's `concurrent_chroms`
/// can re-size the merge tail against the concurrency it actually dispatches,
/// using this exact formula rather than a second copy of it. Passing the
/// planner's own `concurrent_chroms` reproduces `ThreadPlan::processing_threads`
/// exactly, so recomputing unconditionally is safe.
///
/// This is the MONOLITHIC-reader shape (`PIPELINE_THREADS_PER_CHROM` + htslib
/// decode threads per contig). The sharded VCF path bills readers instead and
/// uses [`processing_threads_for`]; the two are not interchangeable.
pub(crate) fn processing_threads(usable_cores: usize, concurrent: usize, htslib: usize) -> usize {
    let active = concurrent * (PIPELINE_THREADS_PER_CHROM + htslib);
    usable_cores.saturating_sub(active).max(1)
}

/// Fitted peak-RSS coefficients for one conversion backend:
///
/// ```text
///   peak_rss_mb ~ base_mb
///               + per_sample_mb * samples
///               + cc * (per_contig_mb + kappa * w * chunk_MB + in_flight_MB(w))
/// ```
///
/// where `cc` is the concurrently-processed contig count. Everything inside
/// the bracket is owned by ONE live contig pipeline; everything outside it is
/// process-wide.
///
/// A law is an UPPER BOUND, not a prediction. `plan_sharded` divides available
/// headroom by the bracket, so a coefficient that is too small becomes an OOM
/// while one that is too large only costs concurrency. Fit these as envelopes
/// -- the tightest coefficients that over-predict every measured point, plus a
/// stated safety margin -- not by least squares. Fitting the mean and then
/// padding to CI upper bounds optimises the wrong objective and leaves slack
/// that is an accident of the residual spread; on the PGEN data that produced
/// a 10.1x worst-case over-allocation where 2.4x was achievable from the same
/// functional form.
///
/// These are load-bearing in production, not just in the bench: a bad refit
/// becomes an OOM. Change a law only alongside a refit that says so, and
/// record that refit's gate result and n in the constant's doc comment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamLaw {
    pub base_mb: f64,
    pub per_sample_mb: f64,
    /// RAM held by ONE live contig pipeline that does not scale with
    /// `chunk_bytes` -- staging buffers and per-contig bookkeeping. Charged
    /// per concurrent contig alongside the `kappa` term, so the per-contig
    /// bracket `plan_sharded` divides headroom by is
    /// `per_contig_mb + kappa * w * chunk_MB + in_flight_MB(w)`, where
    /// `in_flight_MB` is [`in_flight_budget_bytes`] -- an additive term
    /// outside `kappa`, priced separately (see `memory_fits`'s doc comment).
    ///
    /// `0.0` means "not fitted for this backend", which leaves the law
    /// exactly as it behaved before this field existed.
    pub per_contig_mb: f64,
    pub kappa: f64,
}

impl RamLaw {
    /// Sharded VCF path, refitted 2026-08-11 against the crossed sweep (Slurm
    /// job 13355527, `carter-cn-03`, `n = 48/48` points measured on one
    /// node). See
    /// docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed.md.
    ///
    /// **Fitted as an ENVELOPE (LP), not by least squares.** `plan_sharded`
    /// uses this law as an upper bound, and the shipping gate is
    /// over-prediction at every measured point, not squared error. `r2` is
    /// printed by the fitter for description only and is **not** the
    /// shipping criterion -- see `RamLaw`'s top-level doc comment and the
    /// `RamLaw::PGEN` comment below for why an OLS-plus-CI-padding fit is the
    /// wrong objective for a bound.
    ///
    /// ```text
    ///   minimise  t
    ///   s.t.      1.25 * y_i  <=  X_i . b  <=  t * y_i    for every point
    ///             b >= 0
    /// ```
    ///
    /// Gate: over-predicts all 48 measured points as `plan_sharded` evaluates
    /// them, worst **3.7267x**, mean 2.4417x, min exactly 1.2500x (the
    /// margin, honoured as an equality at the binding point).
    ///
    /// **The 1.25 margin is a CHOSEN safety factor, stated here, not
    /// inherited.** An envelope fit touches the data only at its binding
    /// points and therefore carries no margin of its own, and the law is
    /// applied ~3.9x beyond the largest measured cohort (S=128,000 ->
    /// 500,000). Sensitivity, same 48 rows:
    ///
    /// | margin | worst   | mean    |
    /// |-------:|--------:|--------:|
    /// | 1.00   | 2.9814x | 1.9534x |
    /// | 1.25   | 3.7267x | 2.4417x |
    /// | 1.50   | 4.4721x | 2.9301x |
    /// | 2.00   | 5.9628x | 3.9068x |
    ///
    /// `per_contig_mb` is now MEASURED, not zero. This sweep is the first on
    /// this backend to vary `concurrent_chroms` at fixed `(S, chunk_size)`,
    /// and the fitted value prices the ~128 MiB per-live-contig
    /// `ChunkAssembler` staging allocation that the old cc-blind OLS fit left
    /// unpriced -- an UNDER-prediction, the OOM direction, unlike that fit's
    /// double-counted `kappa` (issue #158).
    ///
    /// An interaction term (`per_contig_per_sample_mb`, entering as
    /// `cc * samples`) was fitted and gated against the committed 2026-08-08
    /// PGEN crossed data before this sweep ran, under a decision rule
    /// pre-registered in issue #158: adopt only on a >=20% worst-case
    /// improvement. It reached only 1.76% (2.4189x -> 2.3763x). NO-GO.
    /// `RamLaw::VCF` therefore ships the same four-coefficient FORM as
    /// before (`base_mb`, `per_sample_mb`, `per_contig_mb`, `kappa`) -- not
    /// the same VALUES: all four moved in this refit, and `per_contig_mb`
    /// came off its old `0.0`. `per_contig_per_sample_mb` stays the
    /// tested-but-dormant `0.0` default.
    ///
    /// Validity domain: S in {4,000, 32,000, 128,000}, chunk_MB roughly
    /// 4-100 MB -- but chunk_MB above 16 exists ONLY at S=32,000 (2 chunk
    /// sizes, 25 and 100 MB, each measured at cc=1 and cc=8); the largest
    /// chunk actually measured at S=128,000 is 15.9 MB. `cc` in {1,4,8,16},
    /// `reader_workers == 1` and `pending == 0` in every row, 22 contigs, one
    /// node (`carter-cn-03`), `multiallelic_rate` 0.0, no FORMAT/dosage
    /// fields (gt-only payload, matching issue #156).
    ///
    /// Three extrapolations carry this law from the measured domain to
    /// `from_vcf`'s production defaults (`chunk_size: int = 25_000`,
    /// `DEFAULT_READER_WORKERS = 3`), and `per_sample_mb`'s is the smallest
    /// of the three: ~3.9x beyond the largest measured cohort (128,000) to
    /// reach S=500,000. `from_vcf` does not size chunks via
    /// `_auto_chunk_size` -- that helper is called only from `from_pgen`,
    /// `from_vcf_list`, and `from_svar1` -- it takes a fixed
    /// `chunk_size: int = 25_000`, which is 800 MB at S=128,000 and 3,125 MB
    /// at S=500,000: **8x and 31x** the largest chunk this sweep measured
    /// (`VCF_BIGCHUNK`, 100 MB). And every one of the 48 measured rows pinned
    /// `reader_workers = 1`, but `plan_sharded`'s VCF call site passes
    /// `Some(DEFAULT_READER_WORKERS)` = `Some(3)` today (`src/lib.rs`), so
    /// `kappa * w * chunk_MB` is applied at ~3x the measured `w` -- ~93x
    /// combined with the 31x chunk extrapolation at S=500,000. Once a later
    /// task switches this call site to the derive path, `w` reaches ~10 on
    /// the machine issue #169 was measured on: ~10x the measured `w`, ~312x
    /// combined. `in_flight_budget_bytes`'s contribution is additive and
    /// sits OUTSIDE `kappa`, so it carries no `kappa` extrapolation of its
    /// own -- only the `kappa * w * chunk_MB` term above does. `RamLaw::PGEN`
    /// carries no equivalent `w` extrapolation: `from_pgen` pins
    /// `reader_workers = Some(1)` in production (`src/lib.rs`), matching its
    /// own measured domain exactly, so this asymmetry is VCF-specific.
    ///
    /// `cc = 16` sits OUTSIDE the production domain. Unlike the PGEN path,
    /// nothing in `lib.rs` clamps VCF `concurrent_chroms`, so `cc=16` was
    /// reachable only through a since-removed bench-only sweep override,
    /// applied here purely to give the per-contig term a lever arm.
    ///
    /// Residual sigma (an unconstrained OLS fit of the same four-term form,
    /// computed for description only, never shipped) is ~336 MB, ~5.3x the
    /// 63 MB reproducibility floor measured on the PGEN cc ladder -- the law
    /// still does not describe the mechanism; the per-contig bracket grows
    /// with cohort width (~48 -> ~90 -> ~175 MB at the smallest measured
    /// chunk size across S=4,000/32,000/128,000), the same non-additive
    /// pattern the PGEN sweep showed. The envelope is safe regardless --
    /// that is the point of fitting a bound rather than a mean -- but the
    /// per-contig term's true functional form is a separate open research
    /// question, tracked as its own follow-up rather than under #158 or
    /// #160. This refit closes both: #158's umbrella ask -- a cc-aware,
    /// per-contig-priced envelope for every backend -- is now met by both
    /// `RamLaw::PGEN` and `RamLaw::VCF`, and #160, which #158 split out
    /// specifically for VCF's still-cc-blind OLS fit after c100d51 fixed
    /// only PGEN, is fixed by this same envelope LP.
    pub const VCF: RamLaw = RamLaw {
        base_mb: 457.25887672659735,
        per_sample_mb: 0.011017408281198566,
        per_contig_mb: 111.42612019477279,
        kappa: 6.105786273211022,
    };

    /// PGEN path, re-fitted 2026-08-08 against the crossed sweep (job
    /// 13351716, carter-cn-04, 58/58 points measured on one node). See
    /// docs/superpowers/plans/results/2026-08-08-pgen-ram-law-crossed.md.
    ///
    /// **Fitted as an ENVELOPE, not by least squares.** This law is not a
    /// prediction of RSS -- `plan_sharded` uses it as an upper bound and the
    /// shipping gate is over-prediction at every measured point. Fitting by
    /// OLS and then raising slopes to their 95% CI upper bounds (the
    /// 2026-08-07 construction) optimises squared error and *then* pads, so
    /// the resulting slack is an accident of the residual spread: it left the
    /// old law over-predicting by up to 10.111x while no OLS form passed the
    /// gate at all. The coefficients below solve, over the 46 rows where
    /// `concurrent_chroms` was pinned and therefore observed:
    ///
    /// ```text
    ///   minimise  t
    ///   s.t.      1.25 * y_i  <=  X_i . b  <=  t * y_i    for every point
    ///             b >= 0
    /// ```
    ///
    /// so `t` is the worst-case over-allocation and is OPTIMAL for this
    /// functional form -- no other coefficients do better. Gate: over-predicts
    /// all 46 points, worst **2.4189x**, mean 1.8816x, min exactly 1.2500x.
    /// The 2026-08-07 law on this same data is 10.1113x / 3.0497x, so this is
    /// **4.18x less over-allocation at the worst case** and 1.62x on average.
    ///
    /// **The 1.25 margin is a CHOSEN safety factor, stated here, not an
    /// artifact.** The previous law's margin came from `fit_ram_law` fitting
    /// `kappa` cc-blind while `plan_sharded` multiplies by `cc` a second time
    /// -- a double count (issue #158). An envelope fit removes that accident
    /// but would also remove *the* margin, since it touches the data at its
    /// binding points while the law is applied ~3.9x beyond the largest
    /// measured cohort. Raising or lowering it is a deliberate trade of
    /// under-utilisation and spurious `PlanError::InsufficientMemory` against
    /// an OOM; margin 1.00 gives 1.935x worst case, 1.50 gives 2.903x.
    ///
    /// `per_contig_mb` is new and measured: at S=4,000 the per-contig slope is
    /// 83.7-88.8 MB across three chunk sizes, independently reproducing the
    /// 89.67 MB the 2026-08-07 concurrency ladder gave. It is NOT the analytic
    /// `RAW_STAGE_BYTES + MASK_STAGE_BYTES` = 128 MiB = 134.2177 MB: a large
    /// `calloc` costs address space, not resident pages. The shipped value is
    /// above the measured slope because the envelope must also cover S=32,000
    /// and S=128,000, where the slope rises (263 MB, 301 MB) but the CIs are
    /// too wide to identify it -- so the term is a BOUND over the measured
    /// domain, not a per-contig rate.
    ///
    /// **A per-chunk (`n_chunks`) term was measured and deliberately NOT
    /// shipped.** With `chunk_size` pinned and V varied -- holding
    /// `chunk_bytes` exactly constant -- RSS is linear in chunk count at R^2
    /// up to 1.0000, so the term is real, and including it is tightest
    /// in-sample (1.774x). But it would be applied at 40,000 chunks for
    /// V=1e9, ~300x beyond the 32-128 measured, where it dominates and takes
    /// the S=500,000 projection to 160.7 GiB against 65.3 GiB without it. The
    /// ratchet must also saturate physically, since `libc::malloc_trim(0)`
    /// runs at every contig boundary. A well-measured local coefficient is not
    /// licensed for a 300x extrapolation.
    ///
    /// Residual sigma remains ~9x the 63 MB reproducibility floor (measured
    /// from six launches differing only in `cc`, R^2 0.9903): both new
    /// coefficients vary with `S` and `cc`, so the law still does not describe
    /// the mechanism. The envelope is safe regardless -- that is the point of
    /// fitting a bound rather than a mean. #158's own substance (a measured,
    /// non-zero `per_contig_mb` fitted as an envelope) is now closed -- by
    /// this PGEN refit for this backend, and by the 2026-08-11 `RamLaw::VCF`
    /// refit below for the other -- but the per-contig term's true
    /// functional form remains a separate, still-open research question
    /// (see `RamLaw::VCF`'s doc comment).
    ///
    /// Validity domain: S in {4,000, 32,000, 128,000}, chunk_bytes 3.125-250
    /// MB, `cc` in {1,4,8,16}, `reader_workers == 1` and `pending == 0` in
    /// every row, 22 contigs, one node (carter-cn-04), `multiallelic_rate`
    /// 0.0, no FORMAT/dosage fields (see issue #156). `per_sample_mb` is
    /// extrapolated ~3.9x beyond the largest measured cohort to reach
    /// S=500,000.
    ///
    /// `cc <= 8` is enforced in code, not just documented: `src/lib.rs` clamps
    /// every planned `concurrent_chroms` to `PGEN_MAX_CONCURRENT` below, so
    /// `cc > 8` was reachable only via a since-removed bench-only sweep
    /// override. The `cc=16` rows exist to give the per-contig slope a lever
    /// arm and sit OUTSIDE the production domain.
    ///
    /// Still NOT comparable coefficient-by-coefficient with `RamLaw::VCF`,
    /// even though both now come from `vcfixture bulk` against the same
    /// fitted `germline-1kgp-varskew` profile: this corpus predates
    /// vcfixture-rs v0.5.0 (its manifests carry an un-migrated
    /// `profile_hash` that v0.5.x cannot even load), while the 2026-08-11 VCF
    /// sweep used v0.5.0+. v0.5.0 gave variant positions their own PRNG
    /// stream, so a given seed realizes different variants under the two CLI
    /// versions -- the fitted distributions (SFS, class mix, gaps) are
    /// identical, but the realized draws are not, so the two corpora are not
    /// byte-comparable and neither are the two laws' coefficients.
    pub const PGEN: RamLaw = RamLaw {
        base_mb: 2696.785976670047,
        per_sample_mb: 0.01575147162905773,
        per_contig_mb: 209.8696589690541,
        kappa: 2.3847735782388906,
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
    /// `chunk_size * (n_samples*ploidy/8 + n_format_fields*n_samples*12)`.
    ///
    /// The per-field 12 is `crate::types::FORMAT_BYTES_PER_SAMPLE_PER_VARIANT`:
    /// 8 B of raw `DenseField` retained by `AtomMeta` for the chunk's lifetime
    /// plus the 4 B staged column. It charged the staged 4 alone until #156,
    /// which is the cost of the column and not of the buffer behind it.
    pub chunk_bytes: u64,
    /// `None` means the caller declined a budget; only the core bound applies.
    pub max_mem_bytes: Option<u64>,
    /// `None` asks the planner to choose the contig concurrency. `Some(cc)`
    /// is an explicit caller request, honoured or refused with
    /// `InsufficientMemory` -- the same contract as `reader_workers`, and for
    /// the same reason. It is planned here rather than applied to the returned
    /// plan because overriding `concurrent_chroms` afterwards would leave
    /// `reader_workers` sized for a concurrency that is not running, and would
    /// skip the memory check entirely.
    pub concurrent_chroms: Option<usize>,
    /// `None` asks the planner to derive the reader count from the core
    /// budget. `Some(w)` is an explicit caller request, honoured or refused
    /// with `InsufficientMemory` -- never silently shrunk, because a caller
    /// who asked for 24 readers and got 3 has no way to find out.
    pub reader_workers: Option<usize>,
    /// Which backend's fitted peak-RSS law to plan against.
    pub ram: RamLaw,
    /// Whether this path enforces `shard_exec`'s reorder-backlog ceiling, and
    /// so has to afford it. Separate from `ram` on purpose -- see
    /// [`BacklogGate`].
    pub backlog: BacklogGate,
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

/// Plan contig concurrency AND per-contig reader count together for the
/// sharded VCF reader.
///
/// `cc` and `w` are mutually dependent once `w` derives from cores-per-contig:
/// more concurrent contigs means fewer cores left for each one's readers, and
/// vice versa. This breaks the cycle by planning in a fixed order:
///
/// 1. Reserve the merge tail ([`reader_pool_cores`]) BEFORE sizing readers.
///    `processing_threads_for` sizes `merge.rs`'s var_key gather pool and
///    `dense_merge`'s bit-transpose from whatever the readers leave over;
///    spending every leftover core on readers floors that pool at 1 and gives
///    back the 2.77-2.82x merge-tiling win from commit c49d1d7. This holds
///    only up to the reader pool: an explicit `reader_workers` larger than
///    the pool is still honoured (never silently shrunk) and can starve the
///    merge pool to 1, because refusing a request is reserved for the memory
///    budget, not the core count.
/// 2. Choose `cc`. On the derive path (`inp.reader_workers == None`), prefer
///    depth ([`W_TARGET`] readers per contig) over breadth. On the explicit
///    path, size concurrency from the CALLER's own `w` instead -- a caller
///    who asked for 1 reader per contig has already told us its per-contig
///    demand, and sizing its `cc` off `W_TARGET` (a derive-path default it
///    never asked for) needlessly starves concurrency.
/// 3. Fill the depth: one core for the contig's executor, the rest for its
///    readers.
/// 4. Re-check memory, giving back READERS before contigs when it's tight.
///    Decrementing `cc` first would *raise* `w` (`w = pool / cc - 1`), which
///    raises per-contig memory -- a contig-first loop would not converge.
///
/// An explicit `inp.reader_workers` is honoured or refused, never silently
/// shrunk: a caller who asked for 24 readers and got 3 back has no way to
/// find out. Concurrency may still come down to fit the budget -- that's the
/// planner's own knob, not the caller's.
pub fn plan_sharded(inp: PlanInputs) -> Result<ShardedPlan, PlanError> {
    let n_contigs = inp.n_contigs.max(1);
    // Step 1: reserve the merge tail before anything else claims cores.
    let pool = reader_pool_cores(inp.usable_cores);

    // An explicit contig concurrency is the caller's to set, so it is honoured
    // or refused -- never silently shrunk, and never applied after the fact.
    // Deriving `w` AT that `cc` is the point of doing this here: a caller's
    // `cc` stamped onto a plan whose `w` was sized for a different
    // concurrency is a shape the planner never validated and never would.
    if let Some(req_cc) = inp.concurrent_chroms {
        let cc = req_cc.max(1);
        // Same precedent as the PGEN concurrency knee (`lib.rs`): an explicit
        // request past the modelled domain is honoured, not clamped -- but a
        // `cc` past the contig count is strictly more obviously wrong than
        // past a measured constant, so it gets the same warning. The excess
        // contig-pool threads will never receive work.
        if cc > n_contigs {
            tracing::warn!(
                concurrent_chroms = cc,
                n_contigs,
                "explicit concurrent_chroms exceeds the number of contigs; \
                 honouring it, but the excess contig threads will be idle"
            );
        }
        if let Some(req_w) = inp.reader_workers {
            let w = req_w.max(1);
            memory_fits(&inp, cc, w)?;
            return Ok(ShardedPlan {
                concurrent_chroms: cc,
                reader_workers: w,
            });
        }
        // Same give-back order as the planner's own loop: readers before
        // contigs, because `cc` is fixed here and only `w` can yield.
        let w_max = (pool / cc).saturating_sub(1).max(1);
        if let Some(w) = (1..=w_max)
            .rev()
            .find(|&w| memory_fits(&inp, cc, w).is_ok())
        {
            return Ok(ShardedPlan {
                concurrent_chroms: cc,
                reader_workers: w,
            });
        }
        // w=1 is the cheapest shape at this `cc`; if it does not fit, nothing
        // does, and the caller gets the budget error rather than a quiet
        // downgrade.
        return Err(memory_fits(&inp, cc, 1).expect_err("w=1 just failed the scan above"));
    }

    // An explicit request is honoured or refused. Concurrency may still come
    // down to make it fit -- that is the planner's own knob, not the
    // caller's. Size `cc` from the caller's own `w`, not from `W_TARGET`:
    // the caller has already told us its per-contig demand.
    if let Some(req) = inp.reader_workers {
        let w = req.max(1);
        let cc_cap = (pool / (1 + w)).max(1);
        let mut cc = std::cmp::min(n_contigs, cc_cap);
        while cc > 1 && memory_fits(&inp, cc, w).is_err() {
            cc -= 1;
        }
        memory_fits(&inp, cc, w)?;
        return Ok(ShardedPlan {
            concurrent_chroms: cc,
            reader_workers: w,
        });
    }

    // Step 2: choose the contig concurrency, preferring depth.
    let depth_cap = (pool / (1 + W_TARGET)).max(1);
    let mut cc = std::cmp::min(n_contigs, depth_cap);
    loop {
        // Step 3: fill the depth -- one core for this contig's executor, the
        // rest for its readers.
        let w_max = (pool / cc).saturating_sub(1).max(1);
        // Step 4: memory re-check. Give back READERS before contigs:
        // decrementing `cc` raises `w`, which raises per-contig memory, so a
        // contig-first loop does not converge.
        if let Some(w) = (1..=w_max)
            .rev()
            .find(|&w| memory_fits(&inp, cc, w).is_ok())
        {
            return Ok(ShardedPlan {
                concurrent_chroms: cc,
                reader_workers: w,
            });
        }
        if cc == 1 {
            // The scan above covered w=1 at cc=1, so this is a genuine error.
            return Err(memory_fits(&inp, 1, 1).expect_err("cc=1, w=1 just failed the scan above"));
        }
        cc -= 1;
    }
}

/// Does one `(cc, w)` shape fit the caller's byte budget?
///
/// ```text
///   baseline   = base_mb + per_sample_mb * samples
///   per_contig = per_contig_mb + kappa * w * chunk_MB + in_flight_MB(w)
///   fits       <=> budget - baseline >= cc * per_contig
/// ```
///
/// The backlog term was `kappa * (w + (w-1)) * chunk_MB` before issue #169 --
/// quadratic-ish in `w`, which is what made a large reader count unaffordable.
/// `shard_exec::Frontier` now ENFORCES a fixed backlog ceiling, so it becomes
/// an explicit additive budget. This is not a refit of `RamLaw::VCF`: the
/// fitted coefficients are untouched. It is NOT uniformly more conservative,
/// though. Per unit of `w` the old term charged `2 * kappa` (12.21 chunk-MB)
/// against this one's `kappa + 2` (8.11), so the two cross at
/// `w = 14.1058 / 4.1058 ~= 3.44`: at `w <= 3` this law charges MORE, at
/// `w >= 4` it charges LESS -- 21% less per contig at `w = 10` and a 10 MB
/// chunk, which is what the derive path picks on the machine in issue #169.
/// That percentage is chunk-dependent, since `per_contig_mb` is not: on the
/// chunk-scaled part alone the reduction is 23.2%, which is what it approaches
/// at a production-sized chunk (3125 MB at 500,000 samples). That is defensible
/// only because `Frontier` now ENFORCES the backlog ceiling the `(w-1)` term
/// merely fitted; if that enforcement is ever removed or bypassed, this law
/// under-predicts peak RSS at exactly the reader counts #169 exists to reach.
///
/// `in_flight_MB` is [`in_flight_budget_bytes`], NOT `pending_budget_bytes`:
/// the `workers * 2` capacity of `shard_exec`'s bounded result channel, plus
/// the backlog ceiling on the paths that enforce one (`inp.backlog`; see
/// [`BacklogGate`]). Pricing only the backlog under-counts real peak by up to
/// `2 * w` chunks per contig, and charging a ceiling the path never enforces
/// over-counts it by `PENDING_BUDGET_CHUNKS` chunks -- which refuses PGEN
/// plans that would in fact fit (#173). The readers' own `w` working chunks
/// stay in the `kappa` term and are not double-counted here.
fn memory_fits(inp: &PlanInputs, cc: usize, w: usize) -> Result<(), PlanError> {
    let Some(budget) = inp.max_mem_bytes else {
        return Ok(());
    };
    let budget_mb = budget as f64 / 1e6;
    let baseline_mb = inp.ram.base_mb + inp.ram.per_sample_mb * inp.n_samples as f64;
    let per_contig_mb = inp.ram.per_contig_mb
        + inp.ram.kappa * w as f64 * (inp.chunk_bytes as f64 / 1e6)
        + in_flight_budget_bytes(inp.chunk_bytes, w, inp.backlog) as f64 / 1e6;
    let needed_mb = baseline_mb + per_contig_mb * cc as f64;
    if budget_mb < needed_mb {
        return Err(PlanError::InsufficientMemory {
            needed_mb,
            budget_mb,
            baseline_mb,
        });
    }
    Ok(())
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

    // 48 cores -> 47 usable. pool = 47 - ceil(47/4) = 47 - 12 = 35.
    // reader_workers is explicit (Some(2)), so cc seeds from the caller's
    // own w, not W_TARGET (Finding 2, issue #169 review): cc_cap = 35 /
    // (1+2) = 11. n_contigs=22 doesn't bind (11 < 22), so cc = 11.
    // max_mem_bytes is None, so memory_fits always succeeds and w=2 is
    // honoured as-is. The OLD planner (before this task) returned 15 here
    // (core-bound via `usable/(1+w)`, no merge-tail reserve).
    #[test]
    fn core_bound_concurrency() {
        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 4_000,
            chunk_bytes: 10_937_000,
            max_mem_bytes: None,
            reader_workers: Some(2),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 11,
                reader_workers: 2
            }
        );
    }

    // Fewer contigs than the concurrency cap allows: never spawn a pipeline
    // with no contig. pool = 47 - ceil(47/4) = 35; reader_workers is
    // explicit (Some(2)), so cc_cap = 35/(1+2) = 11 > n_contigs=2, so
    // n_contigs binds instead. Unconstrained by memory (max_mem_bytes:
    // None).
    #[test]
    fn contig_count_bounds_concurrency() {
        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 2,
            n_samples: 4_000,
            chunk_bytes: 10_937_000,
            max_mem_bytes: None,
            reader_workers: Some(2),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 2,
                reader_workers: 2
            }
        );
    }

    // The memory constraint must actually bind, or it is decoration.
    // S=500,000, ploidy 2, no FORMAT fields, chunk_size 25,000, reader_workers
    // explicit (Some(2)), re-derived for the memory law now including
    // `in_flight_budget_bytes`:
    //   chunk_bytes = 25_000 * (500_000*2/8) = 3.125e9 B = 3125 MB
    //   baseline    = 457.259 + 0.011017*500_000            =  5965.963 MB
    //   kappa*w*chunk_MB (w=2) = 6.105786*2*3125            = 38161.164 MB
    //   in_flight(w=2) = 8*3125 + 2*2*3125 = 25000 + 12500  = 37500     MB
    //   per-contig  = 111.426 + 38161.164 + 37500           = 75772.590 MB
    //   cc=3: needed = 5965.963 + 3*75772.590 = 233283.733 MB > 200,000 -> fails
    //   cc=2: needed = 5965.963 + 2*75772.590 = 157511.143 MB <= 200,000 -> fits
    // reader_workers is explicit (Some(2)): cc_cap = pool/(1+w) = 35/3 = 11,
    // so the scan starts at cc=min(22,11)=11 and the shrink loop gives back
    // contigs until it fits at cc=2 -- the huge in-flight term at this chunk
    // size dominates regardless of where the scan starts, since per-contig
    // cost here does not depend on cc.
    #[test]
    fn memory_bound_beats_core_bound_at_biobank_scale() {
        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 500_000,
            chunk_bytes: 3_125_000_000,
            max_mem_bytes: Some(200_000 * 1_000_000),
            reader_workers: Some(2),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 2,
                reader_workers: 2
            }
        );
    }

    // An explicit `concurrent_chroms` is the caller's, so a roomy budget must
    // return it verbatim rather than the planner's own preference. Asking for
    // fewer contigs than the planner would choose is the interesting
    // direction: a plan that quietly raised it would oversubscribe a caller
    // who lowered it on purpose.
    #[test]
    fn explicit_concurrent_chroms_is_returned_verbatim() {
        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: Some(2),
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes: 1_000_000,
            max_mem_bytes: None,
            reader_workers: None,
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 2);
        assert!(plan.reader_workers >= 1);
    }

    // `cc` and `w` together, both explicit: neither may be adjusted to
    // accommodate the other.
    #[test]
    fn explicit_concurrent_chroms_and_reader_workers_are_both_honoured() {
        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: Some(3),
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes: 1_000_000,
            max_mem_bytes: None,
            reader_workers: Some(5),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 3);
        assert_eq!(plan.reader_workers, 5);
    }

    // The contract this whole field exists for: an explicit `cc` that does not
    // fit `max_mem` is REFUSED, not silently shrunk. Before it was planned
    // here it was stamped onto the plan afterwards in `lib.rs`, which skipped
    // the memory check entirely -- an explicit `cc` could blow the budget with
    // no error and no warning.
    #[test]
    fn explicit_concurrent_chroms_that_busts_the_budget_is_refused() {
        let err = plan_sharded(PlanInputs {
            concurrent_chroms: Some(16),
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 500_000,
            chunk_bytes: 3_125_000_000,
            max_mem_bytes: Some(64_000 * 1_000_000),
            reader_workers: None,
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap_err();
        match err {
            PlanError::InsufficientMemory {
                needed_mb,
                budget_mb,
                ..
            } => assert!(needed_mb > budget_mb),
        }
    }

    // The same budget with the planner choosing must SUCCEED -- otherwise the
    // test above would prove only that the budget is impossible, not that the
    // explicit request is what broke it.
    #[test]
    fn the_same_budget_is_plannable_without_the_explicit_request() {
        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 500_000,
            chunk_bytes: 3_125_000_000,
            max_mem_bytes: Some(64_000 * 1_000_000),
            reader_workers: None,
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert!(plan.concurrent_chroms < 16);
    }

    // A budget below the cohort baseline cannot fit even one contig. Failing
    // loudly beats planning cc=0 (which dispatches nothing and "succeeds"
    // with an empty store) or cc=1 (which OOMs). Assertions are formula-
    // agnostic (they just check the baseline-dominated shape of the error),
    // so they hold unchanged under the new law.
    #[test]
    fn budget_below_baseline_is_an_error() {
        let err = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 500_000,
            chunk_bytes: 3_125_000_000,
            max_mem_bytes: Some(5_000 * 1_000_000),
            reader_workers: Some(2),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
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

    // This budget (1 MB) is far below the cohort baseline (468.28 MB), so
    // `chunk_size` is powerless here -- only `max_mem` or a smaller cohort
    // can help. The message must say so, and must NOT claim `chunk_size`
    // would help (that would be actionable-sounding but false in this
    // regime).
    #[test]
    fn insufficient_memory_message_names_remedies() {
        let err = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 1_000,
            max_mem_bytes: Some(1_000_000), // 1 MB -- far below the cohort baseline
            reader_workers: Some(2),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
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
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_200_000_000), // covers baseline (468.28 MB), not per-contig
            reader_workers: Some(16),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
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
            concurrent_chroms: None,
            usable_cores: 1,
            n_contigs: 1,
            n_samples: 250,
            chunk_bytes: 64_000,
            max_mem_bytes: None,
            reader_workers: Some(4),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
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
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 0,
            n_samples: 250,
            chunk_bytes: 64_000,
            max_mem_bytes: None,
            reader_workers: Some(2),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
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

    // Per-contig memory is now
    //   per_contig_mb + kappa*w*chunk_MB + in_flight_MB(w)
    // where in_flight_MB = pending ceiling (8 chunks, enforced by
    // `Frontier`) + tx_res channel capacity (2*w chunks, enforced by the
    // bounded channel). The kappa term is linear in `w` (it was
    // kappa*(w + (w-1)) before #169) and carries the readers' own `w` working
    // chunks, so those are not double-counted below.
    // n_samples=1_000, chunk_bytes=10_000_000 (10 MB), against the
    // 2026-08-11 RamLaw::VCF envelope:
    //   baseline   = 457.259 + 0.011017*1_000  =  468.276 MB
    //   in_flight  = 8*10 + 2*w*10             =   80 + 20w MB
    //   per-contig = 111.426 + 61.05786*w + 80 + 20w = 191.426 + 81.05786*w
    //     w=3  ->  191.426 +  243.174 =  434.600 MB -> needs  902.876 MB
    //     w=16 ->  191.426 + 1296.926 = 1488.352 MB -> needs 1956.628 MB
    //   budget = 1_200 MB at cc=1: fits w=3, rejects w=16.
    #[test]
    fn a_high_worker_count_can_exceed_a_budget_a_lower_one_fits() {
        let inp = PlanInputs {
            concurrent_chroms: None,
            usable_cores: 47,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_200_000_000),
            reader_workers: Some(16),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        };
        assert!(matches!(
            plan_sharded(inp),
            Err(PlanError::InsufficientMemory { .. })
        ));
        assert_eq!(
            plan_sharded(PlanInputs {
                concurrent_chroms: None,
                reader_workers: Some(3),
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
        // Re-derived for the 2026-08-11 envelope refit (issue #158). See
        // docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed.md --
        // reproduce with (paths matter: `BACKEND_SWEEPS["vcf"]` pools all
        // seven VCF sweep families, so pointing this at a directory holding
        // OTHER sweeps too would silently fit a different law):
        //
        //   D=docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed-data
        //   pixi run python -m scripts.bench_svar2.fit_ram \
        //     --results $D --plans $D --manifests $D/manifests \
        //     --backend vcf --margin 1.25
        assert_eq!(RamLaw::VCF.base_mb, 457.25887672659735);
        assert_eq!(RamLaw::VCF.per_sample_mb, 0.011017408281198566);
        assert_eq!(RamLaw::VCF.per_contig_mb, 111.42612019477279);
        assert_eq!(RamLaw::VCF.kappa, 6.105786273211022);
    }

    #[test]
    // The whole point is asserting on RamLaw::VCF's const fields, so clippy
    // sees a compile-time-constant condition; that's the guard, not a bug.
    #[allow(clippy::assertions_on_constants)]
    fn ram_law_vcf_is_a_usable_law() {
        // Guards against a placeholder shipping: a zero kappa would make the
        // memory bound vacuous and silently restore unbounded planning.
        assert!(RamLaw::VCF.kappa > 0.0, "kappa must be positive");
        assert!(RamLaw::VCF.base_mb > 0.0, "baseline must be positive");
        assert!(RamLaw::VCF.per_sample_mb >= 0.0);
        // Measured by the 2026-08-11 crossed sweep, which varied
        // concurrent_chroms at fixed (S, chunk_size) for the first time on
        // this backend. A refit that drops it back to zero has silently
        // discarded the per-contig staging cost and restored the
        // under-prediction #158 was opened about.
        assert!(
            RamLaw::VCF.per_contig_mb > 0.0,
            "VCF's per-contig term is measured, not optional"
        );
    }

    #[test]
    fn the_monolithic_fallback_never_costs_more_than_the_sharded_plan_billed() {
        // `plan_sharded` bills a contig `1 + w` cores (executor + readers) and
        // sizes `concurrent_chroms` against that. A contig that falls back to
        // the monolithic reader runs an executor, ONE reader thread, and
        // `fallback_htslib_threads(w)` decode threads -- which must fit inside
        // the same bill, or the fallback oversubscribes a concurrency that was
        // never planned for it (#152).
        for w in 1..=64 {
            let billed = 1 + w;
            let spent = 1 + 1 + fallback_htslib_threads(w);
            assert!(
                spent <= billed,
                "w={w}: monolithic fallback spends {spent} of {billed} billed cores"
            );
        }
        // And it spends the whole bill rather than leaving decode threads on
        // the table: the fallback is not meant to be slower than it was paid
        // for.
        assert_eq!(fallback_htslib_threads(8), 7);
        // `w = 1` leaves nothing for a decode pool. 0 is "htslib default", not
        // an error -- see `vcf_reader::zero_htslib_threads_opens_without_a_decode_pool`.
        assert_eq!(fallback_htslib_threads(1), 0);
        assert_eq!(fallback_htslib_threads(0), 0);
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
        // The PGEN law FITTED this term (0.0 would mean "not fitted for
        // this backend" -- true of neither law since the 2026-08-11 VCF
        // refit, see `ram_law_vcf_is_a_usable_law` below). A refit that
        // drops it back to zero here has silently discarded the per-contig
        // staging cost the 2026-08-08 crossed sweep measured.
        assert!(
            RamLaw::PGEN.per_contig_mb > 0.0,
            "PGEN's per-contig term is measured, not optional"
        );
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
        // Must mirror `plan_sharded`'s bracket EXACTLY, including the
        // chunk-independent `per_contig_mb` AND `in_flight_budget_bytes`
        // (backlog ceiling + bounded result channel). Computing only the
        // kappa term here would under-size the budget and silently turn this
        // into a "cc=1" test that still looks like it is asserting 2.
        let per_contig_mb = RamLaw::PGEN.per_contig_mb
            + RamLaw::PGEN.kappa * 1.0 * (chunk_bytes as f64 / 1e6)
            + in_flight_budget_bytes(chunk_bytes, 1, BacklogGate::Disabled) as f64 / 1e6;
        let budget = ((baseline_mb + 2.5 * per_contig_mb) * 1e6) as u64;

        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 64,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes,
            max_mem_bytes: Some(budget),
            reader_workers: Some(1),
            ram: RamLaw::PGEN,
            backlog: BacklogGate::Disabled,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 2);
    }

    #[test]
    fn a_pgen_plan_is_not_charged_for_a_backlog_ceiling_it_never_enforces() {
        // The PGEN path hands `shard_exec::run` `u64::MAX`, so `Frontier`
        // enforces no backlog ceiling and none needs to be afforded. Charging
        // it anyway reserved `PENDING_BUDGET_CHUNKS` (8) chunks per concurrent
        // contig against a legitimate 2 -- 4x the real result-channel reserve,
        // which refuses PGEN plans that would in fact run (#173).
        //
        // The budget below is exactly baseline + one contig under the
        // corrected bracket. Planning the SAME inputs with the gate marked
        // enforced must refuse: that is the over-charge, isolated.
        let chunk_bytes = 100_000_000u64;
        let n_samples = 1_000usize;
        let baseline_mb = RamLaw::PGEN.base_mb + RamLaw::PGEN.per_sample_mb * n_samples as f64;
        let per_contig_mb = RamLaw::PGEN.per_contig_mb
            + RamLaw::PGEN.kappa * (chunk_bytes as f64 / 1e6)
            + in_flight_budget_bytes(chunk_bytes, 1, BacklogGate::Disabled) as f64 / 1e6;
        let budget = ((baseline_mb + per_contig_mb) * 1e6).ceil() as u64;
        let inp = |backlog| PlanInputs {
            concurrent_chroms: None,
            usable_cores: 64,
            n_contigs: 22,
            n_samples,
            chunk_bytes,
            max_mem_bytes: Some(budget),
            reader_workers: Some(1),
            ram: RamLaw::PGEN,
            backlog,
        };
        assert_eq!(
            plan_sharded(inp(BacklogGate::Disabled))
                .expect("a budget sized to the real bracket must plan")
                .concurrent_chroms,
            1
        );
        assert!(
            plan_sharded(inp(BacklogGate::Enforced)).is_err(),
            "the same budget must be short by the backlog ceiling when it IS \
             enforced -- otherwise this test is not measuring the over-charge"
        );
    }

    #[test]
    fn pgen_budget_too_small_for_one_contig_is_an_error_not_a_silent_cc_of_one() {
        // Below the baseline + one contig, planning must FAIL. Clamping to
        // cc=1 and proceeding would OOM at the exact scale the budget exists
        // to protect, and would do it after writing a partial store.
        //
        // This budget (1 MB) is far below even the cohort baseline
        // (base_mb + per_sample_mb*1_000_000 ~= 18,448 MB under RamLaw::PGEN),
        // so this exercises the baseline-dominated branch: `chunk_size`
        // cannot help here, only `max_mem` (or a smaller cohort) can.
        let err = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 64,
            n_contigs: 22,
            n_samples: 1_000_000,
            chunk_bytes: 10_000_000_000,
            max_mem_bytes: Some(1_000_000),
            reader_workers: Some(1),
            ram: RamLaw::PGEN,
            backlog: BacklogGate::Disabled,
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
        // per_contig_mb must halve the memory-bound concurrency. If
        // plan_sharded still read module constants, both would return the
        // same cc.
        //
        // chunk_bytes: 0 zeroes both `kappa*w*chunk_MB` and
        // `in_flight_budget_bytes` (`pending_budget_bytes(0) == 0`, and the
        // channel term is chunk_bytes-scaled too), isolating `per_contig_mb`
        // as the only per-contig cost. With chunk_bytes > 0 the fixed
        // backlog/channel budget is the same in both arms, so doubling ONE
        // coefficient (kappa, as the pre-#169 version of this test did) does
        // NOT exactly double the per-contig bracket, and an exact-2x
        // assertion would be unfounded arithmetic; doubling `per_contig_mb`
        // with chunk_bytes=0 keeps the bracket purely proportional.
        //
        // baseline = 1000 + 0*1000 = 1000 MB; headroom = 3000 - 1000 = 2000 MB.
        // pool = reader_pool_cores(64) = 64 - ceil(64/4) = 48. reader_workers
        // is `Some(1)` (explicit), so cc seeds from the caller's own w, not
        // W_TARGET: cc_cap = (48 / (1+1)).max(1) = 24, cc = min(n_contigs=32,
        // 24) = 24 for both arms. The while loop then shrinks cc by 1 until
        // memory_fits passes.
        //   per_contig_mb=500:  needed(cc) = 1000 + cc*500; fits once cc <= 4.
        //     Loop shrinks 24 -> 4: cc=4 -> needed=1000+4*500=3000 <= 3000 -> a=4.
        //   per_contig_mb=1000: needed(cc) = 1000 + cc*1000; fits once cc <= 2.
        //     Loop shrinks 24 -> 2: cc=2 -> needed=1000+2*1000=3000 <= 3000 -> b=2.
        let base = PlanInputs {
            concurrent_chroms: None,
            usable_cores: 64,
            n_contigs: 32,
            n_samples: 1_000,
            chunk_bytes: 0,
            max_mem_bytes: Some(3_000_000_000),
            reader_workers: Some(1),
            ram: RamLaw {
                base_mb: 1000.0,
                per_sample_mb: 0.0,
                per_contig_mb: 500.0,
                kappa: 0.0,
            },
            backlog: BacklogGate::Enforced,
        };
        let doubled = PlanInputs {
            concurrent_chroms: None,
            ram: RamLaw {
                per_contig_mb: 1000.0,
                ..base.ram
            },
            ..base
        };
        let a = plan_sharded(base).unwrap().concurrent_chroms;
        let b = plan_sharded(doubled).unwrap().concurrent_chroms;
        assert_eq!(
            a,
            2 * b,
            "cc must scale inversely with per_contig_mb: {a} vs {b}"
        );
    }

    #[test]
    fn reader_pool_reserves_a_quarter_of_cores_for_the_merge_tail() {
        // Spending every leftover core on readers floors `processing_threads_for`
        // at 1 and gives back the 2.77-2.82x merge-tiling win (c49d1d7).
        assert_eq!(reader_pool_cores(31), 23); // 31 - ceil(31/4)=8
        assert_eq!(reader_pool_cores(16), 12); // 16 - 4
        assert_eq!(reader_pool_cores(1), 1); // never zero
        assert_eq!(reader_pool_cores(0), 1);
    }

    #[test]
    fn pending_budget_is_a_fixed_multiple_of_the_chunk() {
        // Deliberately independent of max_mem: plan_sharded consumes the
        // budget to pick cc, and max_mem is what bounds cc, so deriving one
        // from the other is circular.
        assert_eq!(pending_budget_bytes(10_000_000), 80_000_000);
        assert_eq!(pending_budget_bytes(0), 0);
    }

    #[test]
    fn in_flight_budget_adds_the_bounded_result_channel() {
        // `Frontier` bounds only the collector's backlog map. `shard_exec`'s
        // `tx_res` holds up to `workers * 2` more assembled chunks, so
        // planning against the backlog alone under-counts real peak by 2*w
        // chunks per contig. The readers' own `w` working chunks are NOT
        // added here -- they live in the planner's `kappa * w` term.
        //   w=3:  8*10MB + 2*3*10MB  =  80 +  60 = 140 MB
        //   w=16: 8*10MB + 2*16*10MB =  80 + 320 = 400 MB
        assert_eq!(
            in_flight_budget_bytes(10_000_000, 3, BacklogGate::Enforced),
            140_000_000
        );
        assert_eq!(
            in_flight_budget_bytes(10_000_000, 16, BacklogGate::Enforced),
            400_000_000
        );
        // Never cheaper than the backlog ceiling alone, even at w=0.
        assert_eq!(
            in_flight_budget_bytes(10_000_000, 0, BacklogGate::Enforced),
            100_000_000,
            "workers floors at 1"
        );
    }

    #[test]
    fn a_disabled_backlog_gate_charges_only_the_result_channel() {
        // The PGEN path passes `u64::MAX` as `pending_budget_bytes`
        // (`orchestrator.rs`), so no backlog ceiling exists to pay for -- but
        // `tx_res` is bounded by the channel itself on every path, so that
        // half stays (#173).
        //   enforced, w=1: 8*10MB + 2*1*10MB = 80 + 20 = 100 MB
        //   disabled, w=1:          2*1*10MB =       20 =  20 MB
        assert_eq!(
            in_flight_budget_bytes(10_000_000, 1, BacklogGate::Disabled),
            20_000_000
        );
        assert_eq!(
            in_flight_budget_bytes(10_000_000, 1, BacklogGate::Enforced),
            100_000_000
        );
        // Still floors `workers` at 1: a zero must not zero the channel term.
        assert_eq!(
            in_flight_budget_bytes(10_000_000, 0, BacklogGate::Disabled),
            20_000_000
        );
    }

    #[test]
    fn plan_prefers_depth_over_breadth_on_the_reported_machine() {
        // The #169 machine: 32 vCPU -> 31 usable, 22 contigs, no budget.
        //   pool      = 31 - 8 = 23
        //   depth_cap = 23 / (1 + W_TARGET=8) = 2
        //   cc        = min(22, 2) = 2
        //   w         = 23 / 2 - 1 = 10
        // Today the same machine yields cc=7, w=3.
        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 31,
            n_contigs: 22,
            n_samples: 535_662,
            chunk_bytes: 10_000_000,
            max_mem_bytes: None,
            reader_workers: None,
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 2,
                reader_workers: 10,
            }
        );
    }

    #[test]
    fn plan_never_exceeds_the_contig_count() {
        let plan = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 96,
            n_contigs: 1,
            n_samples: 1_000,
            chunk_bytes: 1_000_000,
            max_mem_bytes: None,
            reader_workers: None,
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 1);
        // pool = 96 - 24 = 72; w = 72/1 - 1 = 71.
        assert_eq!(plan.reader_workers, 71);
    }

    #[test]
    fn a_tight_budget_shrinks_workers_before_contigs() {
        // Shrinking cc first would RAISE w (w = pool/cc - 1) and so raise
        // per-contig memory -- the loop would not converge. The plan must
        // give back readers first.
        let roomy = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 31,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: None,
            reader_workers: None,
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        let tight = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 31,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_200_000_000),
            reader_workers: None,
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap();
        assert!(
            tight.reader_workers < roomy.reader_workers,
            "a tight budget must cost readers: {tight:?} vs {roomy:?}"
        );
        assert!(tight.reader_workers >= 1);
    }

    #[test]
    fn an_explicit_reader_workers_is_honoured_or_refused_never_shrunk() {
        let inp = PlanInputs {
            concurrent_chroms: None,
            usable_cores: 31,
            n_contigs: 22,
            n_samples: 1_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: None,
            reader_workers: Some(24),
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        };
        assert_eq!(plan_sharded(inp).unwrap().reader_workers, 24);

        // Same request against a budget that cannot hold it: refuse rather
        // than quietly hand back a slower plan the caller did not ask for.
        assert!(matches!(
            plan_sharded(PlanInputs {
                concurrent_chroms: None,
                max_mem_bytes: Some(600_000_000),
                ..inp
            }),
            Err(PlanError::InsufficientMemory { .. })
        ));
    }

    #[test]
    fn a_budget_below_the_cohort_baseline_still_reports_the_baseline() {
        // The `budget_mb < baseline_mb` branch of PlanError's Display is the
        // one that tells a caller chunk_size cannot help them; keep it
        // reachable.
        let err = plan_sharded(PlanInputs {
            concurrent_chroms: None,
            usable_cores: 31,
            n_contigs: 1,
            n_samples: 10_000_000,
            chunk_bytes: 10_000_000,
            max_mem_bytes: Some(1_000_000),
            reader_workers: None,
            ram: RamLaw::VCF,
            backlog: BacklogGate::Enforced,
        })
        .unwrap_err();
        let PlanError::InsufficientMemory {
            budget_mb,
            baseline_mb,
            ..
        } = err;
        assert!(budget_mb < baseline_mb);
    }

    #[test]
    fn the_documented_max_mem_floors_match_the_planner() {
        // These three figures are published in skills/genoray-api/SKILL.md. If
        // this test fails, the law changed and that document is now lying to
        // users about how much memory they need -- update BOTH.
        //
        // The floor is the cc=1, w=1 point, because `plan_sharded`'s derive path
        // scans `w` down to 1 before giving up a contig. `chunk_bytes` is
        // `0.25 * n_samples * chunk_size`: two haplotypes at one bit each.
        for (n_samples, floor_mb) in [(4_000u64, 1_016u64), (128_000, 14_864), (500_000, 56_408)] {
            let chunk_bytes = n_samples * 25_000 / 4;
            let inp = |budget_mb: u64| PlanInputs {
                concurrent_chroms: None,
                usable_cores: 31,
                n_contigs: 22,
                n_samples: n_samples as usize,
                chunk_bytes,
                max_mem_bytes: Some(budget_mb * 1_000_000),
                reader_workers: None,
                ram: RamLaw::VCF,
                backlog: BacklogGate::Enforced,
            };
            // One MB under the published floor must refuse...
            assert!(
                plan_sharded(inp(floor_mb - 1)).is_err(),
                "S={n_samples}: planner accepted a budget below the documented floor"
            );
            // ...and the floor itself must plan, at exactly one reader.
            let plan = plan_sharded(inp(floor_mb)).expect("documented floor must plan");
            assert_eq!(plan.reader_workers, 1, "S={n_samples}");
            assert_eq!(plan.concurrent_chroms, 1, "S={n_samples}");
        }
    }
}

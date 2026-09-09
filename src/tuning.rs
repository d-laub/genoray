//! Explicit scheduling knobs, replacing the `GENORAY_*` environment channel.
//!
//! Two types, with a deliberate split:
//!
//! - [`TuningIn`] is what Python sends: every field optional, `None` meaning
//!   "planner's choice". It is the caller's REQUEST.
//! - [`ResolvedTuning`] is what the pipeline runs on: every field concrete. It
//!   keeps the original request so the `pipeline config` log line can say, per
//!   field, whether the value came from the caller or the planner -- the
//!   distinction that a `GENORAY_*` override could never report.

use pyo3::FromPyObject;

/// Monitor sampling cadence when the caller does not choose one. Matches the
/// default of the environment-variable sampler interval this API replaces.
pub const DEFAULT_SAMPLE_INTERVAL_SECS: usize = 5;

/// A caller's tuning request, extracted from the Python `Tuning` dataclass by
/// attribute name. Field names MUST stay in lockstep with
/// `python/genoray/_tuning.py`.
#[derive(FromPyObject, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TuningIn {
    pub concurrent_chroms: Option<usize>,
    pub reader_workers: Option<usize>,
    pub overshard: Option<usize>,
    pub dense_cap: Option<usize>,
    pub merge_threads: Option<usize>,
    pub sample_interval: Option<usize>,
}

/// A resolution in progress: everything except `merge_threads`, which cannot be
/// defaulted yet because its planner default is `processing_threads`, computed
/// from `concurrent_chroms` further downstream.
///
/// This exists so that "resolved but missing `merge_threads`" is not a state
/// [`ResolvedTuning`] can be in. The only way out is
/// [`Self::with_merge_threads`], so a caller cannot forget the second phase and
/// silently run with a zero merge budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartialTuning {
    concurrent_chroms: usize,
    reader_workers: usize,
    overshard: usize,
    dense_cap: usize,
    sample_interval: usize,
    requested: TuningIn,
}

/// The concrete values this run uses, plus the request they came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedTuning {
    pub concurrent_chroms: usize,
    pub reader_workers: usize,
    pub overshard: usize,
    pub dense_cap: usize,
    pub merge_threads: usize,
    pub sample_interval: usize,
    /// The caller's original request, kept so `*_src` can report provenance
    /// without a parallel bool per field.
    pub requested: TuningIn,
}

fn tag(explicit: bool) -> &'static str {
    if explicit { "explicit" } else { "planner" }
}

impl TuningIn {
    /// Apply the planner's `(cc, w)` decision to this request.
    ///
    /// `concurrent_chroms` and `reader_workers` are passed in rather than read
    /// from `self` because the planner has already honoured-or-refused them
    /// against `max_mem`: the value that actually runs is the planner's output,
    /// and the request only supplies the source tag.
    ///
    /// A request that is `Some` is honoured as-is. Values are NOT clamped here:
    /// the API contract is honour-or-refuse, and Python's `Tuning.__post_init__`
    /// has already rejected anything below each field's minimum, so a clamp
    /// could only silently rewrite a pure-Rust caller's explicit choice.
    pub fn resolve(self, concurrent_chroms: usize, reader_workers: usize) -> PartialTuning {
        PartialTuning {
            concurrent_chroms,
            reader_workers,
            overshard: self
                .overshard
                .unwrap_or(crate::orchestrator::OVERSHARD_FACTOR),
            dense_cap: self
                .dense_cap
                .unwrap_or(crate::orchestrator::VCF_LIST_DENSE_CHANNEL_CAP),
            sample_interval: self.sample_interval.unwrap_or(DEFAULT_SAMPLE_INTERVAL_SECS),
            requested: self,
        }
    }
}

impl PartialTuning {
    /// The resolved shard-oversubscription factor.
    ///
    /// Exposed before `with_merge_threads` because the VCF path needs it to
    /// size `plan_unit_count` and `SourceSpec::Vcf`, and that sizing feeds
    /// `processing_threads` -- which `with_merge_threads` in turn requires.
    /// Reading it here is what stops the caller re-deriving `resolve`'s own
    /// formula and letting the two drift, which would make the `pipeline
    /// config` banner report a shard count the pipeline never used.
    ///
    /// Only `overshard` is exposed: it is the sole knob resolved in phase one
    /// that a caller must act on before phase two.
    pub fn overshard(&self) -> usize {
        self.overshard
    }

    /// Supply the planner's `processing_threads` and finish the resolution.
    ///
    /// An explicit request wins; otherwise `planner_default` is used, floored at
    /// 1 because a computed 0 would mean a merge with no threads at all. The
    /// floor deliberately applies only to the planner's own value, never to a
    /// caller's request.
    pub fn with_merge_threads(self, planner_default: usize) -> ResolvedTuning {
        let merge_threads = self
            .requested
            .merge_threads
            .unwrap_or(planner_default.max(1));
        ResolvedTuning {
            concurrent_chroms: self.concurrent_chroms,
            reader_workers: self.reader_workers,
            overshard: self.overshard,
            dense_cap: self.dense_cap,
            merge_threads,
            sample_interval: self.sample_interval,
            requested: self.requested,
        }
    }
}

impl ResolvedTuning {
    pub fn concurrent_chroms_src(&self) -> &'static str {
        tag(self.requested.concurrent_chroms.is_some())
    }
    pub fn reader_workers_src(&self) -> &'static str {
        tag(self.requested.reader_workers.is_some())
    }
    pub fn overshard_src(&self) -> &'static str {
        tag(self.requested.overshard.is_some())
    }
    pub fn dense_cap_src(&self) -> &'static str {
        tag(self.requested.dense_cap.is_some())
    }
    pub fn merge_threads_src(&self) -> &'static str {
        tag(self.requested.merge_threads.is_some())
    }
    pub fn sample_interval_src(&self) -> &'static str {
        tag(self.requested.sample_interval.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_fields_take_planner_defaults() {
        let r = TuningIn::default().resolve(4, 6).with_merge_threads(12);
        assert_eq!(r.concurrent_chroms, 4);
        assert_eq!(r.reader_workers, 6);
        assert_eq!(r.overshard, crate::orchestrator::OVERSHARD_FACTOR);
        assert_eq!(r.dense_cap, crate::orchestrator::VCF_LIST_DENSE_CHANNEL_CAP);
        assert_eq!(r.merge_threads, 12);
        assert_eq!(r.sample_interval, DEFAULT_SAMPLE_INTERVAL_SECS);
    }

    #[test]
    fn unset_fields_report_the_planner_as_their_source() {
        let r = TuningIn::default().resolve(4, 6).with_merge_threads(12);
        assert_eq!(r.concurrent_chroms_src(), "planner");
        assert_eq!(r.reader_workers_src(), "planner");
        assert_eq!(r.overshard_src(), "planner");
        assert_eq!(r.dense_cap_src(), "planner");
        assert_eq!(r.merge_threads_src(), "planner");
        assert_eq!(r.sample_interval_src(), "planner");
    }

    #[test]
    fn explicit_fields_win_and_report_themselves() {
        let requested = TuningIn {
            overshard: Some(40),
            dense_cap: Some(24),
            merge_threads: Some(2),
            sample_interval: Some(0),
            ..TuningIn::default()
        };
        let r = requested.resolve(4, 6).with_merge_threads(12);
        assert_eq!(r.overshard, 40);
        assert_eq!(r.dense_cap, 24);
        assert_eq!(r.merge_threads, 2);
        assert_eq!(r.sample_interval, 0);
        assert_eq!(r.overshard_src(), "explicit");
        assert_eq!(r.dense_cap_src(), "explicit");
        assert_eq!(r.merge_threads_src(), "explicit");
        // 0 is a real, meaningful request here -- it disables the sampler --
        // so it must NOT be mistaken for "unset".
        assert_eq!(r.sample_interval_src(), "explicit");
    }

    #[test]
    fn concurrency_and_workers_come_from_the_planner_but_report_the_request() {
        // The planner has already honoured-or-refused these two by the time
        // `resolve` runs, so the resolved value is the planner's either way;
        // only the SOURCE tag distinguishes them.
        let requested = TuningIn {
            concurrent_chroms: Some(2),
            reader_workers: Some(20),
            ..TuningIn::default()
        };
        let r = requested.resolve(2, 20).with_merge_threads(12);
        assert_eq!(r.concurrent_chroms, 2);
        assert_eq!(r.reader_workers, 20);
        assert_eq!(r.concurrent_chroms_src(), "explicit");
        assert_eq!(r.reader_workers_src(), "explicit");
    }

    #[test]
    fn with_merge_threads_does_not_override_an_explicit_request() {
        let requested = TuningIn {
            merge_threads: Some(2),
            ..TuningIn::default()
        };
        let r = requested.resolve(4, 6).with_merge_threads(99);
        assert_eq!(r.merge_threads, 2);
    }

    #[test]
    fn a_planner_default_of_zero_is_floored_but_a_request_is_not() {
        // The floor exists for a planner that computed 0 threads, not to
        // second-guess a caller: honour-or-refuse means an explicit value is
        // never silently rewritten.
        let r = TuningIn::default().resolve(4, 6).with_merge_threads(0);
        assert_eq!(r.merge_threads, 1);
        assert_eq!(r.merge_threads_src(), "planner");
    }

    #[test]
    fn an_explicit_request_is_honoured_verbatim_without_clamping() {
        // Python rejects these before they get here, so a value this low can
        // only come from a pure-Rust caller -- who gets what they asked for.
        let requested = TuningIn {
            overshard: Some(0),
            dense_cap: Some(0),
            ..TuningIn::default()
        };
        let r = requested.resolve(4, 6).with_merge_threads(12);
        assert_eq!(r.overshard, 0);
        assert_eq!(r.dense_cap, 0);
        assert_eq!(r.overshard_src(), "explicit");
    }
}

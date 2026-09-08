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
/// old `GENORAY_SAMPLE_INTERVAL` default.
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

impl ResolvedTuning {
    /// Build from the planner's `(cc, w)` decision plus the caller's request.
    ///
    /// `concurrent_chroms` and `reader_workers` are passed in rather than read
    /// from `requested` because the planner has already honoured-or-refused
    /// them against `max_mem`: the value that actually runs is the planner's
    /// output, and `requested` only supplies the source tag.
    ///
    /// `merge_threads` cannot be defaulted here -- its planner default is
    /// `processing_threads`, which is computed from `concurrent_chroms`
    /// downstream -- so it is left at 0 until [`Self::with_merge_threads`].
    pub fn resolve(requested: TuningIn, concurrent_chroms: usize, reader_workers: usize) -> Self {
        Self {
            concurrent_chroms,
            reader_workers,
            overshard: requested
                .overshard
                .unwrap_or(crate::orchestrator::OVERSHARD_FACTOR)
                .max(1),
            dense_cap: requested
                .dense_cap
                .unwrap_or(crate::orchestrator::VCF_LIST_DENSE_CHANNEL_CAP)
                .max(1),
            merge_threads: 0,
            sample_interval: requested
                .sample_interval
                .unwrap_or(DEFAULT_SAMPLE_INTERVAL_SECS),
            requested,
        }
    }

    /// Fill `merge_threads` from the planner's `processing_threads`, unless the
    /// caller asked for a specific value.
    pub fn with_merge_threads(mut self, planner_default: usize) -> Self {
        self.merge_threads = self
            .requested
            .merge_threads
            .unwrap_or(planner_default)
            .max(1);
        self
    }

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
        let r = ResolvedTuning::resolve(TuningIn::default(), 4, 6).with_merge_threads(12);
        assert_eq!(r.concurrent_chroms, 4);
        assert_eq!(r.reader_workers, 6);
        assert_eq!(r.overshard, crate::orchestrator::OVERSHARD_FACTOR);
        assert_eq!(r.dense_cap, crate::orchestrator::VCF_LIST_DENSE_CHANNEL_CAP);
        assert_eq!(r.merge_threads, 12);
        assert_eq!(r.sample_interval, DEFAULT_SAMPLE_INTERVAL_SECS);
    }

    #[test]
    fn unset_fields_report_the_planner_as_their_source() {
        let r = ResolvedTuning::resolve(TuningIn::default(), 4, 6).with_merge_threads(12);
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
        let r = ResolvedTuning::resolve(requested, 4, 6).with_merge_threads(12);
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
        let r = ResolvedTuning::resolve(requested, 2, 20).with_merge_threads(12);
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
        let r = ResolvedTuning::resolve(requested, 4, 6).with_merge_threads(99);
        assert_eq!(r.merge_threads, 2);
    }
}

//! SigProfilerClusters' per-mutation subclassification, ported.
//!
//! Faithful port of `SigProfilerClusters.classifyFunctions.
//! findClustersOfClusters` with `correction=False` and of
//! `findClustersOfClusters_noVAF`. The decision tree, the sentinels, and the
//! deliberate deviations are documented in
//! `docs/superpowers/specs/2026-09-19-svar2-cluster-labels-design.md`.

pub mod classify;
pub mod write;

/// Storage schema version stamped into `meta.json` as `cluster_version`.
pub const CLUSTER_VERSION: u32 = 1;

/// Wire codebook for the `cluster_class` FORMAT field (pinned again by tests
/// in Rust and Python).
pub use classify::{DOUBLET, KATAEGIS, MBS, NONCLUSTERED, NOT_ANNOTATED, OMIKLI, OTHER};

//! Disk-facing `(range, sample)` query for a finished SVAR2 contig (M5 part 2b).
//! Wires the pure `search.rs` overlap core to the on-disk sidecars: for a contig,
//! region `[q_start, q_end)`, and sample, return that sample's overlapping
//! variants per haplotype. `search.rs` is untouched.
//!
//! Split (Task 1 of the SP-2 module-split plan) into per-concern files:
//! `sidecar` (mmap'd file I/O), `reader` (`ContigReader` + var_key/dense search
//! helpers), `union` (`DenseUnion`), `decode` (`Call`/`decode_keyref` +
//! `HapCalls`/`QueryResult`), `gather` (the batched `BatchResult{,Split}` query
//! paths), and `oracle` (test/gvl-facing reference wrappers).
//!
//! ## Production entry points vs. the `oracle` reference/testing surface
//!
//! | Production (`query::*`)          | Oracle (`query::oracle::*`)               | Notes                                       |
//! |-----------------------------------|--------------------------------------------|----------------------------------------------|
//! | [`overlap_batch`]                 | [`oracle::overlap_sample`]                 | per-sample reference impl                    |
//! | [`find_ranges`] + [`gather_ranges`] | —                                         | tree-driven batch query -> [`BatchResult`]   |
//! | [`gather_haps_readbound`]         | [`oracle::gather_ranges_readbound`]        | split-dense read-bound gather                |
//! | [`read_ranges`]                   | —                                          | —                                             |
//! | [`decode::decode_keyref`]         | [`oracle::decode_keyref`] / [`oracle::decode_keyref_alt`] | oracle wrappers over the same decode logic |
//!
//! [`BatchResult`] is the unified-dense result produced by [`overlap_batch`] /
//! [`gather_ranges`]. [`BatchResultSplit`] is the split-dense counterpart
//! produced by [`gather_haps_readbound`] and its oracle twin
//! [`oracle::gather_ranges_readbound`].

pub mod decode;
pub mod gather;
pub mod oracle;
pub mod reader;
pub mod sidecar;
pub mod union;

pub use crate::spine::KeyRef;
pub use decode::{HapCalls, QueryResult};
pub use gather::{
    BatchResult, BatchResultSplit, RangesBundle, find_ranges, gather_haps_readbound, gather_ranges,
    overlap_batch, read_ranges,
};
pub use reader::ContigReader;

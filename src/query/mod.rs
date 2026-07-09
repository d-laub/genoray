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

// Task 6 will move these to `query::oracle::*`; kept re-exported here this task
// so the split is a pure move.
pub use oracle::{
    decode_keyref_alt_pub, decode_keyref_pub, gather_ranges_readbound, overlap_sample,
};

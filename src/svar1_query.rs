//! Ungated SVAR1 range-query core: the query counterpart to the conversion-gated
//! `svar1_reader::Svar1RecordSource`.
//!
//! Two independent stages, mirroring `python/genoray/_var_ranges.py` +
//! `python/genoray/_svar/_kernels.py::_find_starts_ends`:
//!
//! * [`var_ranges`] — POS ranges -> global variant-id ranges. Pure; a thin wrapper
//!   over `search::overlap_range`, which already ports the Python algorithm.
//! * [`find_ranges`] — variant-id ranges -> absolute CSR index pairs into the
//!   `variant_idxs` mmap, via two `partition_point`s per haplotype.
//!
//! There is deliberately **no `gather_ranges`**: SVAR2 needs one because it merges
//! two channels and decodes keys, but SVAR1's on-disk layout is already the target
//! representation, so consumers build a zero-copy view straight from the index pairs
//! (cf. `SparseVar.read_ranges` -> `Ragged.from_offsets`).

use std::fs::File;
use std::ops::Range;
use std::path::Path;

use memmap2::Mmap;

use crate::search::{SearchTree, overlap_range};

/// POS ranges -> **global** half-open variant-id ranges, one per region, in
/// `regions` order.
///
/// * `v_starts` / `v_ends` — this contig's LOCAL 0-based variant starts (ascending)
///   and exclusive ends (`v_end = POS - min(ILEN, 0)`; a SNP at `s` has `v_end == s+1`).
/// * `max_v_len` — `max(v_ends - v_starts)` over the contig, i.e. **Python's
///   `var_ranges` convention** (`_var_ranges.py:78`). `overlap_range` only requires a
///   `>=` bound on the deletion span, so this over-estimates by exactly 1 and is
///   provably overshoot-safe (it merely widens the candidate window). Do NOT subtract
///   1 to "tighten" it — under-estimating IS a correctness bug.
/// * `contig_start` — this contig's first variant's GLOBAL id. Contigs are contiguous
///   in global-id space.
///
/// Nothing overlapping yields an **in-bounds zero-length** range (`start == end`),
/// never a sentinel: an out-of-range offset is poison for downstream byte math
/// (seqpro `Ragged.to_packed` overflows int64 even for an empty row). This
/// deliberately differs from Python `var_ranges`, which returns `INT32_MAX`.
///
/// Only the endpoints are guaranteed to overlap — an interior id can be a
/// deletion-spanned non-overlap. Same contract as `search::overlap_range` and SVAR 1.0
/// `var_ranges`.
pub fn var_ranges(
    v_starts: &[u32],
    v_ends: &[u32],
    max_v_len: u32,
    contig_start: u32,
    regions: &[(u32, u32)],
) -> Vec<Range<u32>> {
    debug_assert_eq!(v_starts.len(), v_ends.len());
    // An empty contig has no tree to build and no ends to scan.
    if v_starts.is_empty() {
        return regions.iter().map(|_| contig_start..contig_start).collect();
    }
    // One tree for the whole batch: `overlap_range` is called per region but the
    // tree build is hoisted, mirroring the SVAR2 search/gather split's intent.
    let tree = SearchTree::new(v_starts);
    regions
        .iter()
        .map(|&(q_start, q_end)| {
            let (s, e) = overlap_range(&tree, v_ends, max_v_len, q_start, q_end);
            (contig_start + s as u32)..(contig_start + e as u32)
        })
        .collect()
}

/// mmap a file read-only, returning `None` for a zero-length file (memmap2 rejects
/// empty maps; an SVAR1 store where no hap carries a call has an empty
/// `variant_idxs`). Local rather than reusing `query::sidecar::mmap_file`, which is
/// `pub(crate)` inside a private module — and keeping this module's dependencies
/// minimal is what keeps it ungated.
fn mmap_ro(path: &Path) -> std::io::Result<Option<Mmap>> {
    let f = File::open(path)?;
    if f.metadata()?.len() == 0 {
        return Ok(None);
    }
    // SAFETY: a finished, read-only store artifact; we never mutate the file while
    // it is mapped. Same contract as `query::sidecar::mmap_file`.
    Ok(Some(unsafe { Mmap::map(&f)? }))
}

/// An SVAR1 store opened for range queries. Holds `variant_idxs` mmap'd (it is one
/// entry per non-ref call — never materialize it) and the small CSR `offsets`
/// resident (`num_haps + 1`), mirroring the SVAR2 `SubStreamView` split.
///
/// Unlike SVAR2's `ContigReader`, this takes **no `chrom`**: an SVAR1 store is one
/// flat directory and contigs are contiguous in global-id space. Per-contig scoping
/// is the caller's job, via `var_ranges`'s `contig_start`.
pub struct Svar1Reader {
    n_samples: usize,
    ploidy: usize,
    variant_idxs: Option<Mmap>,
    offsets: Vec<i64>,
}

impl Svar1Reader {
    /// Open the SVAR1 store rooted at `svar1_dir` for a cohort of `n_samples` at
    /// `ploidy`.
    ///
    /// NOTE: `variant_idxs.npy` and `offsets.npy` are **headerless raw buffers**
    /// despite the extension (Python np.memmaps them). Do not reach for
    /// `ndarray_npy::read_npy` — that is only correct for SVAR2's real `.npy`
    /// sidecars.
    pub fn open(svar1_dir: &str, n_samples: usize, ploidy: usize) -> std::io::Result<Self> {
        let dir = Path::new(svar1_dir);
        let variant_idxs = mmap_ro(&dir.join("variant_idxs.npy"))?;

        // `fs::read` gives an unaligned Vec<u8>, so `cast_slice` would panic;
        // `pod_collect_to_vec` copies element-wise and is alignment-safe. `offsets`
        // is tiny (num_haps + 1), so the copy is free.
        let offsets_bytes = std::fs::read(dir.join("offsets.npy"))?;
        let offsets: Vec<i64> = bytemuck::pod_collect_to_vec(&offsets_bytes);

        let want = n_samples * ploidy + 1;
        if offsets.len() != want {
            return Err(std::io::Error::other(format!(
                "{}/offsets.npy has {} entries; expected n_samples*ploidy+1 = {} \
                 (n_samples={n_samples}, ploidy={ploidy})",
                svar1_dir,
                offsets.len(),
                want,
            )));
        }

        Ok(Self {
            n_samples,
            ploidy,
            variant_idxs,
            offsets,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn ploidy(&self) -> usize {
        self.ploidy
    }

    /// The flat `variant_idxs` buffer: each hap's `offsets[h]..offsets[h+1]` slice
    /// holds its sorted global non-ref variant ids. Exposed so consumers can hand it
    /// straight to a kernel as a zero-copy sparse-index input.
    ///
    /// mmap pages are page-aligned, so `bytemuck`'s alignment check always passes;
    /// a missing/empty map yields an empty slice.
    pub fn variant_idxs(&self) -> &[i32] {
        match &self.variant_idxs {
            Some(m) => bytemuck::cast_slice(&m[..]),
            None => &[],
        }
    }

    /// CSR offsets over haplotypes; `len() == n_samples * ploidy + 1`. Hap column is
    /// `sample * ploidy + p`.
    pub fn offsets(&self) -> &[i64] {
        &self.offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Three variants on a contig whose global ids start at 100.
    // local 0: SNP  at 10 -> v_end 11
    // local 1: DEL  at 20, ILEN -3 -> v_end 23
    // local 2: SNP  at 30 -> v_end 31
    // max_v_len (Python convention) = max(v_ends - v_starts) = max(1, 3, 1) = 3
    fn fixture() -> (Vec<u32>, Vec<u32>, u32) {
        (vec![10, 20, 30], vec![11, 23, 31], 3)
    }

    #[test]
    fn var_ranges_maps_local_overlap_to_global_ids() {
        let (vs, ve, mvl) = fixture();
        // [10, 21) overlaps local 0 (SNP@10) and local 1 (DEL@20) -> global 100..102
        let got = var_ranges(&vs, &ve, mvl, 100, &[(10, 21)]);
        assert_eq!(got, vec![100..102]);
    }

    #[test]
    fn var_ranges_deletion_spanning_query_start_is_included() {
        // The whole point of the sub-scan: a DEL starting BEFORE the query still
        // deletes bases inside it. Query [21, 22) starts after the DEL's POS (20)
        // but before its end (23), so local 1 must be included.
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 100, &[(21, 22)]);
        assert_eq!(got, vec![101..102]);
    }

    #[test]
    fn var_ranges_no_overlap_is_zero_length_not_sentinel() {
        // A zero-length in-bounds range -- NEVER a sentinel like u32::MAX. An
        // out-of-range offset overflows int64 in seqpro's Ragged.to_packed.
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 100, &[(50, 60)]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].start, got[0].end, "no-overlap must be zero-length");
    }

    #[test]
    fn var_ranges_empty_contig_yields_zero_length_ranges() {
        // n_local == 0: must not panic (a .max() over an empty slice would).
        let got = var_ranges(&[], &[], 0, 42, &[(0, 100), (5, 6)]);
        assert_eq!(got, vec![42..42, 42..42]);
    }

    #[test]
    fn var_ranges_batches_regions_in_order() {
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 0, &[(30, 31), (10, 11)]);
        assert_eq!(got, vec![2..3, 0..1], "output must be in `regions` order");
    }

    use std::io::Write;

    /// Write a HEADERLESS raw buffer. SVAR1's `*.npy` files have no npy header
    /// despite the extension -- Python np.memmaps them, Rust bytemucks them.
    /// Mirrors `svar1_reader.rs`'s test helper of the same name.
    fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(bytemuck::cast_slice(data)).unwrap();
    }

    /// 2 samples x ploidy 2 = 4 haps. Per-hap sorted global ids:
    ///   hap0: [0, 2, 4]   hap1: [3]   hap2: [2]   hap3: []
    fn write_store(dir: &std::path::Path) {
        write_raw::<i32>(dir, "variant_idxs.npy", &[0, 2, 4, 3, 2]);
        write_raw::<i64>(dir, "offsets.npy", &[0, 3, 4, 5, 5]);
    }

    #[test]
    fn reader_opens_and_exposes_raw_buffers() {
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        assert_eq!(r.n_samples(), 2);
        assert_eq!(r.ploidy(), 2);
        assert_eq!(r.variant_idxs(), &[0, 2, 4, 3, 2]);
        assert_eq!(r.offsets(), &[0, 3, 4, 5, 5]);
    }

    #[test]
    fn reader_missing_dir_is_err() {
        assert!(Svar1Reader::open("/no/such/svar1/store", 2, 2).is_err());
    }

    #[test]
    fn reader_rejects_offsets_of_wrong_length() {
        // offsets MUST be num_haps + 1. A mismatch means the caller's
        // n_samples/ploidy disagree with the store -- fail loudly rather than
        // index out of bounds later inside find_ranges.
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 2]); // len 3 => 2 haps
        let err = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2); // wants 5
        assert!(err.is_err(), "offsets length mismatch must be an error");
    }

    #[test]
    fn reader_empty_variant_idxs_is_ok() {
        // A store where no hap carries any non-ref call: variant_idxs is
        // zero-length. memmap2 rejects empty maps, so this must not blow up.
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[] as &[i32]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 0, 0, 0, 0]);
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        assert_eq!(r.variant_idxs(), &[] as &[i32]);
    }
}

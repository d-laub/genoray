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

        // The CSR total: `offsets`'s last entry must equal `variant_idxs`'s entry
        // count, or a truncated/mismatched `variant_idxs.npy` would otherwise pass
        // `open()` and panic deep inside `find_ranges` at `vi[o_s..o_e]`. Also reject
        // a negative last offset (a corrupt store) up front, since `find_ranges`
        // casts offsets to `usize` assuming non-negativity.
        let last = offsets[want - 1];
        let n_entries = variant_idxs
            .as_ref()
            .map_or(0, |m| m.len() / std::mem::size_of::<i32>());
        if last < 0 || last as usize != n_entries {
            return Err(std::io::Error::other(format!(
                "{svar1_dir}/offsets.npy's last entry is {last}; expected it to equal \
                 variant_idxs.npy's entry count = {n_entries}",
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

/// Absolute CSR index pairs for a cartesian `(range, sample, ploid)` query.
///
/// `starts`/`stops` are each `n_ranges * n_samples * ploidy` long in C-order
/// `(range, sample, ploid)` — i.e. exactly Python `_find_starts_ends`'s `(2, r, s, p)`
/// output with the leading axis split into two vectors. Indices are absolute into
/// [`Svar1Reader::variant_idxs`].
pub struct Svar1RangesBundle {
    pub n_ranges: usize,
    pub n_samples: usize,
    pub ploidy: usize,
    /// Original sample indices, in output order (identity when `samples` was `None`).
    pub sample_cols: Vec<usize>,
    pub starts: Vec<i64>,
    pub stops: Vec<i64>,
}

/// Variant-id ranges -> absolute CSR index pairs. The Rust port of
/// `_find_starts_ends` (`python/genoray/_svar/_kernels.py`).
///
/// Each hap's CSR run holds **sorted** global variant ids, so a `[v_lo, v_hi)` id
/// range maps to a sub-slice by two `partition_point`s. `samples`, if given, selects
/// (and reorders) a sample subset by original index; `None` means all samples in
/// store order.
///
/// Empty results are **in-bounds zero-length** (`start == stop`), never a sentinel —
/// see [`var_ranges`].
///
/// Single-threaded by design, matching the SVAR2 query core (`query/gather.rs`); the
/// consumer owns parallelism.
pub fn find_ranges(
    reader: &Svar1Reader,
    ranges: &[Range<u32>],
    samples: Option<&[usize]>,
) -> Svar1RangesBundle {
    let ploidy = reader.ploidy();
    let sample_cols: Vec<usize> = match samples {
        Some(s) => s.to_vec(),
        None => (0..reader.n_samples()).collect(),
    };

    let vi = reader.variant_idxs();
    let offs = reader.offsets();
    let n_ranges = ranges.len();
    let n_samples = sample_cols.len();
    let n = n_ranges * n_samples * ploidy;

    let mut starts = Vec::with_capacity(n);
    let mut stops = Vec::with_capacity(n);

    for r in ranges {
        debug_assert!(
            r.start <= r.end,
            "find_ranges: inverted range {}..{} (start > end)",
            r.start,
            r.end
        );
        // u32 -> i32: safe because global variant ids are stored as `i32` (SVAR1's
        // on-disk dtype), so a valid id is always < 2^31 and cannot lose data here.
        let (lo, hi) = (r.start as i32, r.end as i32);
        for &s in &sample_cols {
            for p in 0..ploidy {
                let h = s * ploidy + p; // sample-major, ploidy-minor
                let o_s = offs[h];
                let o_e = offs[h + 1];
                // i64 -> usize: safe because `Svar1Reader::open` rejects a negative
                // last offset (and CSR offsets are monotone non-decreasing), so every
                // `offs[h]` here is non-negative.
                debug_assert!(o_s >= 0 && o_e >= o_s, "corrupt CSR offsets: {o_s}..{o_e}");
                let hap = &vi[o_s as usize..o_e as usize];
                // + o_s makes the index absolute into the flat buffer (Python does
                // the same: `np.searchsorted(sp_genos, var_ranges).T + o_s`).
                starts.push(hap.partition_point(|&g| g < lo) as i64 + o_s);
                stops.push(hap.partition_point(|&g| g < hi) as i64 + o_s);
            }
        }
    }

    Svar1RangesBundle {
        n_ranges,
        n_samples,
        ploidy,
        sample_cols,
        starts,
        stops,
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
    fn reader_rejects_offsets_last_not_matching_variant_idxs() {
        // offsets.len() is correct for the cohort (2 samples x ploidy 2 -> 5), but
        // its last entry (9) doesn't match variant_idxs's actual entry count (2) --
        // a truncated/corrupt variant_idxs.npy. Must be rejected in `open()`, not
        // panic later in `find_ranges`.
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 2, 3, 9]);
        let err = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2);
        assert!(
            err.is_err(),
            "offsets last-entry / variant_idxs length mismatch must be an error"
        );
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

    #[test]
    // A one-element `&[2..5]` is intentional: it's a single-range query, not a
    // "did you mean the range itself" typo.
    #[allow(clippy::single_range_in_vec_init)]
    fn find_ranges_binary_searches_each_hap_csr() {
        // hap0: [0, 2, 4] @ entries 0..3   hap1: [3] @ entry 3
        // hap2: [2] @ entry 4              hap3: []  @ entry 5..5
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();

        // id range [2, 5): hap0 -> entries 1..3 ([2,4]); hap1 -> 3..4 ([3]);
        //                  hap2 -> 4..5 ([2]);          hap3 -> 5..5 (empty)
        let b = find_ranges(&r, &[2..5], None);
        assert_eq!(b.n_ranges, 1);
        assert_eq!(b.n_samples, 2);
        assert_eq!(b.ploidy, 2);
        assert_eq!(b.sample_cols, vec![0, 1]);
        // C-order (range, sample, ploid) -> hap0, hap1, hap2, hap3
        assert_eq!(b.starts, vec![1, 3, 4, 5]);
        assert_eq!(b.stops, vec![3, 4, 5, 5]);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn find_ranges_empty_id_range_is_in_bounds_zero_length() {
        // A zero-length input range must produce start == stop, in bounds --
        // never a sentinel (poison for seqpro Ragged.to_packed's int64 math).
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        let b = find_ranges(&r, &[7..7], None);
        for (s, e) in b.starts.iter().zip(&b.stops) {
            assert_eq!(s, e, "empty range must be zero-length");
            assert!(
                *s >= 0 && *s <= 5,
                "offset {s} must be in bounds of variant_idxs"
            );
        }
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn find_ranges_sample_subset_selects_and_reorders() {
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        // sample 1 only -> haps 2, 3
        let b = find_ranges(&r, &[2..5], Some(&[1]));
        assert_eq!(b.n_samples, 1);
        assert_eq!(b.sample_cols, vec![1]);
        assert_eq!(b.starts, vec![4, 5]);
        assert_eq!(b.stops, vec![5, 5]);

        // reordered subset -> hap order follows sample_cols, not store order
        let b = find_ranges(&r, &[2..5], Some(&[1, 0]));
        assert_eq!(b.starts, vec![4, 5, 1, 3]);
    }

    #[test]
    fn find_ranges_multiple_ranges_are_c_order() {
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        let b = find_ranges(&r, &[0..1, 2..5], None);
        assert_eq!(b.n_ranges, 2);
        // range 0 ([0,1)): hap0 -> 0..1, others empty
        // range 1 ([2,5)): as above
        assert_eq!(b.starts, vec![0, 3, 4, 5, /* range 1 */ 1, 3, 4, 5]);
        assert_eq!(b.stops, vec![1, 3, 4, 5, /* range 1 */ 3, 4, 5, 5]);
    }
}

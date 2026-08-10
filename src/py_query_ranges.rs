//! SVAR2 search/gather split: numpy-dict bindings on `PyContigReader` for the
//! `query::find_ranges`/`gather_ranges`/`read_ranges` core (Tasks 1-3). Separate
//! `#[pymethods]` block (multiple-pymethods) so the M6b `overlap_batch` binding
//! in `py_query_batch.rs` is untouched.
//!
//! `read_ranges`/`gather_ranges` produce the exact same dict contract as
//! `overlap_batch` (see `py_query_batch.rs`): `vk_pos, vk_key, vk_off, dense_pos,
//! dense_key, dense_range, dense_present, dense_present_off, lut_bytes, lut_off,
//! n_regions, n_samples, ploidy`. `find_ranges` returns the compact `RangesBundle`
//! dict instead; `gather_ranges` consumes one to replay a `BatchResult`.
//!
//! `out=` streaming (writing directly into a caller-provided memmap) is
//! deferred to the Python layer (Task 5) — `find_ranges` here always returns
//! freshly-allocated numpy arrays.
//!
//! The read-bound (split-dense) half of the same split is exposed by
//! `find_dense_class_ranges` (per-class dense windows, built WITHOUT
//! `dense_union()`) + `gather_haps_readbound`, which replays a FLAT
//! per-(region, sample) `HapRanges` into the `BatchResultSplit` dict contract.
//! That contract's `lut_bytes`/`lut_off` are OPT-IN (`with_lut=True`), unlike
//! `read_ranges`/`gather_ranges` where they are always present — the LUT is
//! whole-contig, so on a path whose whole point is cell-proportional cost it is
//! the caller's choice to pay for it.
//! Its var_key search half is already exposed by `find_ranges_chunk` (a thin
//! binding over `query::find_ranges_haps`), so there is exactly one way to run
//! each of the two searches.

use std::ops::Range;

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::py_convert::{u8_to_pyarray, u32_to_i32_pyarray, usize_to_i64_pyarray};
use crate::py_query::PyContigReader;
use crate::query::{
    BatchResult, BatchResultSplit, HapRanges, MAX_END_SHIFT, RangesBundle, dense_max_end_keys,
    find_ranges, find_ranges_haps, gather_haps_readbound, gather_ranges, read_ranges,
};

/// Identical to `py_query_batch.rs::overlap_batch`'s dict assembly — the whole
/// point of the search/gather split is that `read_ranges`/`gather_ranges`
/// produce the same numpy contract as `overlap_batch`.
fn batch_result_to_dict<'py>(
    py: Python<'py>,
    reader_lut: (Vec<u8>, Vec<u64>),
    br: &BatchResult,
) -> PyResult<Bound<'py, PyDict>> {
    let vk_pos: Vec<u32> = br.vk.iter().map(|k| k.position).collect();
    let vk_key: Vec<u32> = br.vk.iter().map(|k| k.key).collect();
    let dense_pos: Vec<u32> = br.dense.iter().map(|k| k.position).collect();
    let dense_key: Vec<u32> = br.dense.iter().map(|k| k.key).collect();

    // dense_range as [R, 2] i32.
    let r = br.dense_range.len();
    let mut dr: Vec<i32> = Vec::with_capacity(r * 2);
    for range in &br.dense_range {
        dr.push(range.start as i32);
        dr.push(range.end as i32);
    }
    let dense_range = Array2::from_shape_vec((r, 2), dr)
        .expect("dense_range shape")
        .to_pyarray(py);

    let (lut_bytes, lut_off_u64) = reader_lut;
    let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();

    let d = PyDict::new(py);
    d.set_item("vk_pos", u32_to_i32_pyarray(py, &vk_pos))?;
    d.set_item("vk_key", u32_to_i32_pyarray(py, &vk_key))?;
    d.set_item("vk_off", usize_to_i64_pyarray(py, &br.vk_off))?;
    d.set_item("dense_pos", u32_to_i32_pyarray(py, &dense_pos))?;
    d.set_item("dense_key", u32_to_i32_pyarray(py, &dense_key))?;
    d.set_item("dense_range", dense_range)?;
    d.set_item("dense_present", u8_to_pyarray(py, &br.dense_present))?;
    d.set_item(
        "dense_present_off",
        usize_to_i64_pyarray(py, &br.dense_present_off),
    )?;
    d.set_item("lut_bytes", u8_to_pyarray(py, &lut_bytes))?;
    d.set_item("lut_off", PyArray1::from_slice(py, &lut_off))?;
    d.set_item("n_regions", br.n_regions)?;
    d.set_item("n_samples", br.n_samples)?;
    d.set_item("ploidy", br.ploidy)?;
    Ok(d)
}

/// Compact `RangesBundle` -> numpy dict: `dense_range (R,2) i32`, `region_starts
/// (R) i32`, `sample_cols (n_samples) i64`, `vk_snp_range (R*H,2) i64`,
/// `vk_indel_range (R*H,2) i64`, `dense_snp_range (R,2) i32`, `dense_indel_range
/// (R,2) i32`, plus `n_regions`/`n_samples`/`ploidy` scalars (H =
/// n_samples*ploidy).
fn bundle_to_dict<'py>(py: Python<'py>, rb: &RangesBundle) -> PyResult<Bound<'py, PyDict>> {
    let pairs_to_i32_flat = |v: &[Range<usize>]| -> Vec<i32> {
        let mut o = Vec::with_capacity(v.len() * 2);
        for r in v {
            o.push(r.start as i32);
            o.push(r.end as i32);
        }
        o
    };
    let pairs_to_i64_flat = |v: &[Range<usize>]| -> Vec<i64> {
        let mut o = Vec::with_capacity(v.len() * 2);
        for r in v {
            o.push(r.start as i64);
            o.push(r.end as i64);
        }
        o
    };

    let dr = pairs_to_i32_flat(&rb.dense_range);
    let dense_range = Array2::from_shape_vec((rb.n_regions, 2), dr)
        .expect("dense_range shape")
        .to_pyarray(py);

    let dsr = pairs_to_i32_flat(&rb.dense_snp_range);
    let dense_snp_range = Array2::from_shape_vec((rb.n_regions, 2), dsr)
        .expect("dense_snp_range shape")
        .to_pyarray(py);
    let dir_ = pairs_to_i32_flat(&rb.dense_indel_range);
    let dense_indel_range = Array2::from_shape_vec((rb.n_regions, 2), dir_)
        .expect("dense_indel_range shape")
        .to_pyarray(py);

    let h = rb.n_samples * rb.ploidy;
    let vk_snp = pairs_to_i64_flat(&rb.vk_snp_range);
    let vk_snp_range = Array2::from_shape_vec((rb.n_regions * h, 2), vk_snp)
        .expect("vk_snp_range shape")
        .to_pyarray(py);
    let vk_indel = pairs_to_i64_flat(&rb.vk_indel_range);
    let vk_indel_range = Array2::from_shape_vec((rb.n_regions * h, 2), vk_indel)
        .expect("vk_indel_range shape")
        .to_pyarray(py);

    let sample_cols: Vec<i64> = rb.sample_cols.iter().map(|&x| x as i64).collect();

    let d = PyDict::new(py);
    d.set_item("dense_range", dense_range)?;
    d.set_item("region_starts", u32_to_i32_pyarray(py, &rb.region_starts))?;
    d.set_item("sample_cols", PyArray1::from_slice(py, &sample_cols))?;
    d.set_item("vk_snp_range", vk_snp_range)?;
    d.set_item("vk_indel_range", vk_indel_range)?;
    d.set_item("dense_snp_range", dense_snp_range)?;
    d.set_item("dense_indel_range", dense_indel_range)?;
    d.set_item("n_regions", rb.n_regions)?;
    d.set_item("n_samples", rb.n_samples)?;
    d.set_item("ploidy", rb.ploidy)?;
    Ok(d)
}

/// Strict numpy-dict readers shared by `bundle_from_dict` (the `RangesBundle`
/// contract) and `hap_ranges_from_dict` (the `HapRanges` contract). Casts are
/// exact (`PyArray1<i32>` / `PyArray1<i64>` / `PyArray2<i32>` / `PyArray2<i64>`),
/// so a dtype slip fails loudly with a TypeError instead of being silently
/// reinterpreted.
mod dict_get {
    use super::*;

    pub(super) fn require<'py>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Bound<'py, PyAny>> {
        d.get_item(k)?
            .ok_or_else(|| PyKeyError::new_err(format!("bundle missing key '{k}'")))
    }

    pub(super) fn i32s(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<i32>> {
        let obj = require(d, k)?;
        let arr = obj.cast::<PyArray1<i32>>().map_err(|_| {
            PyTypeError::new_err(format!("bundle key '{k}' must be an int32 1D array"))
        })?;
        Ok(arr.readonly().as_slice()?.to_vec())
    }

    pub(super) fn i64s(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<i64>> {
        let obj = require(d, k)?;
        let arr = obj.cast::<PyArray1<i64>>().map_err(|_| {
            PyTypeError::new_err(format!("bundle key '{k}' must be an int64 1D array"))
        })?;
        Ok(arr.readonly().as_slice()?.to_vec())
    }

    pub(super) fn i32_pairs(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<Range<usize>>> {
        let obj = require(d, k)?;
        let arr = obj
            .cast::<PyArray2<i32>>()
            .map_err(|_| {
                PyTypeError::new_err(format!("bundle key '{k}' must be an int32 (N,2) array"))
            })?
            .readonly();
        Ok(arr
            .as_array()
            .rows()
            .into_iter()
            .map(|row| (row[0] as usize)..(row[1] as usize))
            .collect())
    }

    pub(super) fn i64_pairs(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<Range<usize>>> {
        let obj = require(d, k)?;
        let arr = obj
            .cast::<PyArray2<i64>>()
            .map_err(|_| {
                PyTypeError::new_err(format!("bundle key '{k}' must be an int64 (N,2) array"))
            })?
            .readonly();
        Ok(arr
            .as_array()
            .rows()
            .into_iter()
            .map(|row| (row[0] as usize)..(row[1] as usize))
            .collect())
    }

    pub(super) fn usize_(d: &Bound<'_, PyDict>, k: &str) -> PyResult<usize> {
        require(d, k)?.extract()
    }
}

/// Inverse of `bundle_to_dict`: read a `find_ranges` dict back into a
/// `RangesBundle` for `gather_ranges`. Fallible: a missing key or wrong
/// dtype/shape becomes a Python KeyError/TypeError rather than a Rust panic.
fn bundle_from_dict(d: &Bound<'_, PyDict>) -> PyResult<RangesBundle> {
    let get_i32 = |k: &str| dict_get::i32s(d, k);
    let get_i64 = |k: &str| dict_get::i64s(d, k);
    let get_i32_pairs = |k: &str| dict_get::i32_pairs(d, k);
    let get_i64_pairs = |k: &str| dict_get::i64_pairs(d, k);
    let get_usize = |k: &str| dict_get::usize_(d, k);

    Ok(RangesBundle {
        n_regions: get_usize("n_regions")?,
        n_samples: get_usize("n_samples")?,
        ploidy: get_usize("ploidy")?,
        region_starts: get_i32("region_starts")?
            .into_iter()
            .map(|x| x as u32)
            .collect(),
        dense_range: get_i32_pairs("dense_range")?,
        sample_cols: get_i64("sample_cols")?
            .into_iter()
            .map(|x| x as usize)
            .collect(),
        vk_snp_range: get_i64_pairs("vk_snp_range")?,
        vk_indel_range: get_i64_pairs("vk_indel_range")?,
        dense_snp_range: get_i32_pairs("dense_snp_range")?,
        dense_indel_range: get_i32_pairs("dense_indel_range")?,
    })
}

/// Owned backing store for a `HapRanges`, which borrows all six of its slices.
/// The dict -> Rust conversion has to materialize the `Vec<Range<usize>>`s
/// somewhere; keeping them in one struct with a `view()` makes the borrow
/// obviously outlive the `gather_haps_readbound` call.
struct OwnedHapRanges {
    region_starts: Vec<u32>,
    orig_samples: Vec<usize>,
    vk_snp_range: Vec<Range<usize>>,
    vk_indel_range: Vec<Range<usize>>,
    dense_snp_range: Vec<Range<usize>>,
    dense_indel_range: Vec<Range<usize>>,
    ploidy: usize,
}

impl OwnedHapRanges {
    fn view(&self) -> HapRanges<'_> {
        HapRanges::new(
            &self.region_starts,
            &self.orig_samples,
            &self.vk_snp_range,
            &self.vk_indel_range,
            &self.dense_snp_range,
            &self.dense_indel_range,
            self.ploidy,
        )
    }
}

/// Read a FLAT per-(region, sample) `HapRanges` dict. Dtypes deliberately match
/// `bundle_to_dict`'s (`region_starts` int32, sample indices int64, `vk_*_range`
/// int64 (N,2), `dense_*_range` int32 (N,2)) so a caller can slice a
/// `find_ranges`/`find_ranges_chunk` result straight in with no casting.
///
/// Every length/bound the Rust core would otherwise `assert!` or index-panic on
/// is checked here first, so a malformed dict is a Python ValueError.
fn hap_ranges_from_dict(
    d: &Bound<'_, PyDict>,
    n_samples_total: usize,
    reader_ploidy: usize,
) -> PyResult<OwnedHapRanges> {
    let ploidy = dict_get::usize_(d, "ploidy")?;
    if ploidy != reader_ploidy {
        return Err(PyValueError::new_err(format!(
            "hap_ranges ploidy {ploidy} != this contig's ploidy {reader_ploidy}"
        )));
    }
    let region_starts: Vec<u32> = dict_get::i32s(d, "region_starts")?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let n_q = region_starts.len();

    let orig_samples: Vec<usize> = dict_get::i64s(d, "orig_samples")?
        .into_iter()
        .map(|x| x as usize)
        .collect();
    if let Some(&bad) = orig_samples.iter().find(|&&s| s >= n_samples_total) {
        return Err(PyValueError::new_err(format!(
            "orig_samples contains {bad}, out of bounds for {n_samples_total} samples"
        )));
    }

    let vk_snp_range = dict_get::i64_pairs(d, "vk_snp_range")?;
    let vk_indel_range = dict_get::i64_pairs(d, "vk_indel_range")?;
    let dense_snp_range = dict_get::i32_pairs(d, "dense_snp_range")?;
    let dense_indel_range = dict_get::i32_pairs(d, "dense_indel_range")?;

    let check = |name: &str, got: usize, want: usize| -> PyResult<()> {
        if got == want {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "hap_ranges key '{name}' has {got} rows, expected {want}"
            )))
        }
    };
    check("orig_samples", orig_samples.len(), n_q)?;
    check("dense_snp_range", dense_snp_range.len(), n_q)?;
    check("dense_indel_range", dense_indel_range.len(), n_q)?;
    check("vk_snp_range", vk_snp_range.len(), n_q * ploidy)?;
    check("vk_indel_range", vk_indel_range.len(), n_q * ploidy)?;

    Ok(OwnedHapRanges {
        region_starts,
        orig_samples,
        vk_snp_range,
        vk_indel_range,
        dense_snp_range,
        dense_indel_range,
        ploidy,
    })
}

/// `BatchResultSplit` -> numpy dict. The split-dense analog of
/// `batch_result_to_dict`: same `vk_*` keys, but the single `dense_*` channel is
/// replaced by the per-class `dense_snp_*` / `dense_indel_*` pair, and the dense
/// payload covers ONLY the queried windows (no whole-contig `DenseUnion` is ever
/// built).
///
/// `reader_lut` is an `Option` because the LUT is the one remaining O(contig)
/// term on this path: `None` omits `lut_bytes` / `lut_off` entirely (absent, not
/// empty — see `gather_haps_readbound`). `batch_result_to_dict` keeps taking its
/// LUT unconditionally, so no existing caller changes.
fn batch_result_split_to_dict<'py>(
    py: Python<'py>,
    reader_lut: Option<(Vec<u8>, Vec<u64>)>,
    br: &BatchResultSplit,
) -> PyResult<Bound<'py, PyDict>> {
    let pairs_to_2d = |v: &[Range<usize>]| {
        let mut o = Vec::with_capacity(v.len() * 2);
        for r in v {
            o.push(r.start as i32);
            o.push(r.end as i32);
        }
        Array2::from_shape_vec((v.len(), 2), o)
            .expect("dense range shape")
            .to_pyarray(py)
    };

    let vk_pos: Vec<u32> = br.vk.iter().map(|k| k.position).collect();
    let vk_key: Vec<u32> = br.vk.iter().map(|k| k.key).collect();
    let snp_pos: Vec<u32> = br.dense_snp.iter().map(|k| k.position).collect();
    let snp_key: Vec<u32> = br.dense_snp.iter().map(|k| k.key).collect();
    let indel_pos: Vec<u32> = br.dense_indel.iter().map(|k| k.position).collect();
    let indel_key: Vec<u32> = br.dense_indel.iter().map(|k| k.key).collect();

    let d = PyDict::new(py);
    d.set_item("vk_pos", u32_to_i32_pyarray(py, &vk_pos))?;
    d.set_item("vk_key", u32_to_i32_pyarray(py, &vk_key))?;
    d.set_item("vk_off", usize_to_i64_pyarray(py, &br.vk_off))?;
    d.set_item("dense_snp_pos", u32_to_i32_pyarray(py, &snp_pos))?;
    d.set_item("dense_snp_key", u32_to_i32_pyarray(py, &snp_key))?;
    d.set_item("dense_snp_range", pairs_to_2d(&br.dense_snp_range))?;
    d.set_item(
        "dense_snp_present",
        u8_to_pyarray(py, &br.dense_snp_present),
    )?;
    d.set_item(
        "dense_snp_present_off",
        usize_to_i64_pyarray(py, &br.dense_snp_present_off),
    )?;
    d.set_item("dense_indel_pos", u32_to_i32_pyarray(py, &indel_pos))?;
    d.set_item("dense_indel_key", u32_to_i32_pyarray(py, &indel_key))?;
    d.set_item("dense_indel_range", pairs_to_2d(&br.dense_indel_range))?;
    d.set_item(
        "dense_indel_present",
        u8_to_pyarray(py, &br.dense_indel_present),
    )?;
    d.set_item(
        "dense_indel_present_off",
        usize_to_i64_pyarray(py, &br.dense_indel_present_off),
    )?;
    if let Some((lut_bytes, lut_off_u64)) = reader_lut {
        let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();
        d.set_item("lut_bytes", u8_to_pyarray(py, &lut_bytes))?;
        d.set_item("lut_off", PyArray1::from_slice(py, &lut_off))?;
    }
    d.set_item("n_regions", br.n_regions)?;
    d.set_item("n_samples", br.n_samples)?;
    d.set_item("ploidy", br.ploidy)?;
    Ok(d)
}

#[pymethods]
impl PyContigReader {
    /// Search + gather in one call, returning the same dict contract as
    /// `overlap_batch` (see `py_query_batch.rs`). `samples`, if given, selects
    /// (and reorders) a sample subset by original index.
    pub fn read_ranges<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let br = read_ranges(&self.inner, &regions, samples.as_deref());
        batch_result_to_dict(py, self.inner.lut_arrays(), &br)
    }

    /// Search-only: returns a compact `RangesBundle` dict (index ranges into the
    /// dense union and var_key sub-streams) with no per-element gather. No `out=`
    /// param — streaming into a caller buffer is a Python-layer (Task 5) concern;
    /// this always returns fresh arrays.
    pub fn find_ranges<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rb = find_ranges(&self.inner, &regions, samples.as_deref());
        bundle_to_dict(py, &rb)
    }

    /// Tree-free gather: replay a `find_ranges` bundle dict into the same dict
    /// contract as `overlap_batch`/`read_ranges`. No `samples` param — the bundle
    /// already fixes the sample subset via `find_ranges`.
    pub fn gather_ranges<'py>(
        &self,
        py: Python<'py>,
        bundle: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rb = bundle_from_dict(&bundle)?;
        let br = gather_ranges(&self.inner, &rb);
        batch_result_to_dict(py, self.inner.lut_arrays(), &br)
    }

    /// Per-class dense windows for the read-bound path: `[s, e)` into
    /// `dense/snp` and `dense/indel` for each region. The read-bound counterpart
    /// of the `dense_range` half of `find_ranges` — dense is cohort-shared, so
    /// these are per-region, not per-(region, sample).
    ///
    /// Unlike `find_ranges`/`find_ranges_header` this never calls
    /// `dense_union()`, which is the whole point: the union merge is O(contig)
    /// per call and is the fixed floor a read-bound gather must not pay.
    pub fn find_dense_class_ranges<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let snp_ix = self.inner.dense_snp_index();
        let indel_ix = self.inner.dense_indel_index();
        let pairs_to_2d = |v: Vec<Range<usize>>| {
            let mut o = Vec::with_capacity(v.len() * 2);
            for r in &v {
                o.push(r.start as i32);
                o.push(r.end as i32);
            }
            Array2::from_shape_vec((v.len(), 2), o)
                .expect("dense range shape")
                .to_pyarray(py)
        };
        let snp: Vec<Range<usize>> = regions.iter().map(|&(s, e)| snp_ix.overlap(s, e)).collect();
        let indel: Vec<Range<usize>> = regions
            .iter()
            .map(|&(s, e)| indel_ix.overlap(s, e))
            .collect();

        let d = PyDict::new(py);
        d.set_item("dense_snp_range", pairs_to_2d(snp))?;
        d.set_item("dense_indel_range", pairs_to_2d(indel))?;
        d.set_item("n_regions", regions.len())?;
        Ok(d)
    }

    /// Tree-free read-bound gather over a FLAT list of `(region, sample)`
    /// queries — the exact-cell counterpart of `gather_ranges`, whose bundle is
    /// inherently a region x sample rectangle.
    ///
    /// `hap_ranges` is the dict contract read by `hap_ranges_from_dict`:
    /// `region_starts (n_q) int32`, `orig_samples (n_q) int64`, `vk_snp_range` /
    /// `vk_indel_range (n_q*ploidy, 2) int64` (row `q*ploidy + p`),
    /// `dense_snp_range` / `dense_indel_range (n_q, 2) int32`, plus a `ploidy`
    /// scalar. Returns the `BatchResultSplit` dict (per-class dense channels);
    /// `n_regions = n_q`, `n_samples = 1`, hap index `q*ploidy + p`.
    ///
    /// Builds zero SearchTrees and never calls `dense_union()`, so the GATHER
    /// itself scales with the cells requested rather than with the contig.
    ///
    /// `with_lut` is off by default because it does not: the LUT (long-INS
    /// allele bytes) is a whole-contig structure, and copying it into the
    /// result costs O(contig) per call — measured at ~0.3 ms/MB on 1kGP, which
    /// is most of this call's fixed cost for a small query. Pass
    /// `with_lut=True` to get `lut_bytes` / `lut_off`, which a consumer needs to
    /// decode long insertions; when it is off those two keys are ABSENT rather
    /// than empty, so a decoder that forgot to ask fails with a KeyError
    /// instead of silently resolving long-INS alleles against an empty table.
    #[pyo3(signature = (hap_ranges, with_lut=false))]
    pub fn gather_haps_readbound<'py>(
        &self,
        py: Python<'py>,
        hap_ranges: Bound<'py, PyDict>,
        with_lut: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let owned = hap_ranges_from_dict(&hap_ranges, self.inner.n_samples, self.inner.ploidy)?;
        let br = gather_haps_readbound(&self.inner, &owned.view());
        // `lut_arrays()` is the O(contig) copy — only call it when asked.
        let lut = if with_lut {
            Some(self.inner.lut_arrays())
        } else {
            None
        };
        batch_result_split_to_dict(py, lut, &br)
    }

    /// Region-level half of a chunked `find_ranges`: everything whose size is
    /// O(regions) rather than O(regions * samples * ploidy), plus the dense
    /// channel's max-end contribution. Cheap enough to compute eagerly.
    pub fn find_ranges_header<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Fail fast rather than silently corrupting a packed key. `ext` is
        // 1 + deletion_len and must fit below the position field.
        let max_del = self.inner.max_deletion_len();
        if (1u64 + max_del as u64) >= (1u64 << MAX_END_SHIFT) {
            return Err(PyValueError::new_err(
                "variant footprint exceeds tie-break packing width",
            ));
        }

        let all_samples = samples.is_none();
        let sample_cols: Vec<usize> = match &samples {
            Some(s) => s.clone(),
            None => (0..self.inner.n_samples).collect(),
        };

        let dense = self.inner.dense_union();
        let dense_ix = dense.index();
        let dense_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| dense_ix.overlap(qs, qe))
            .collect();
        let dense_snp_ix = self.inner.dense_snp_index();
        let dense_snp_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| dense_snp_ix.overlap(qs, qe))
            .collect();
        let dense_indel_ix = self.inner.dense_indel_index();
        let dense_indel_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| dense_indel_ix.overlap(qs, qe))
            .collect();
        let region_starts: Vec<u32> = regions.iter().map(|&(qs, _)| qs).collect();
        let dmax = dense_max_end_keys(
            &self.inner,
            &regions,
            &dense_range,
            &sample_cols,
            all_samples,
        );

        let pairs_i32 = |v: &[Range<usize>]| -> Vec<i32> {
            let mut o = Vec::with_capacity(v.len() * 2);
            for r in v {
                o.push(r.start as i32);
                o.push(r.end as i32);
            }
            o
        };
        let to2d = |v: Vec<i32>| {
            Array2::from_shape_vec((regions.len(), 2), v)
                .expect("region pair shape")
                .to_pyarray(py)
        };

        let d = PyDict::new(py);
        d.set_item("dense_range", to2d(pairs_i32(&dense_range)))?;
        d.set_item("dense_snp_range", to2d(pairs_i32(&dense_snp_range)))?;
        d.set_item("dense_indel_range", to2d(pairs_i32(&dense_indel_range)))?;
        d.set_item("region_starts", u32_to_i32_pyarray(py, &region_starts))?;
        let cols: Vec<i64> = sample_cols.iter().map(|&x| x as i64).collect();
        d.set_item("sample_cols", PyArray1::from_slice(py, &cols))?;
        let dmax_i64: Vec<i64> = dmax.iter().map(|&x| x as i64).collect();
        d.set_item("dense_max_end_keys", PyArray1::from_slice(py, &dmax_i64))?;
        d.set_item("n_regions", regions.len())?;
        d.set_item("n_samples", sample_cols.len())?;
        d.set_item("ploidy", self.inner.ploidy)?;
        Ok(d)
    }

    /// One hap slice `[hap_lo, hap_hi)` of a chunked `find_ranges`. Fills freshly
    /// allocated numpy arrays IN PLACE, so the payload exists exactly once —
    /// unlike `find_ranges`, whose `Vec<Range<usize>>` -> `Vec<i64>` ->
    /// `ToPyArray` chain holds three copies at peak. Releases the GIL for the
    /// search so rayon and the caller's progress bar can both run.
    ///
    /// `vk_snp_range` / `vk_indel_range` come back hap-major, shape
    /// `(n_haps * R, 2)`; reshape to `(n_haps_samples, ploidy, R, 2)` in Python.
    pub fn find_ranges_chunk<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
        hap_lo: usize,
        hap_hi: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sample_cols: Vec<usize> = match &samples {
            Some(s) => s.clone(),
            None => (0..self.inner.n_samples).collect(),
        };
        let h_total = sample_cols.len() * self.inner.ploidy;
        if hap_lo > hap_hi || hap_hi > h_total {
            return Err(PyValueError::new_err(format!(
                "hap slice [{hap_lo}, {hap_hi}) out of bounds for {h_total} haps"
            )));
        }
        let n_haps = hap_hi - hap_lo;
        let r = regions.len();

        let snp = PyArray2::<i64>::zeros(py, [n_haps * r, 2], false);
        let indel = PyArray2::<i64>::zeros(py, [n_haps * r, 2], false);
        let max_keys = {
            let mut snp_rw = snp.readwrite();
            let mut indel_rw = indel.readwrite();
            let snp_s = snp_rw.as_slice_mut()?;
            let indel_s = indel_rw.as_slice_mut()?;
            py.detach(|| {
                find_ranges_haps(
                    &self.inner,
                    &regions,
                    &sample_cols,
                    hap_lo,
                    hap_hi,
                    snp_s,
                    indel_s,
                )
            })
        };

        let keys_i64: Vec<i64> = max_keys.iter().map(|&x| x as i64).collect();
        let d = PyDict::new(py);
        d.set_item("vk_snp_range", snp)?;
        d.set_item("vk_indel_range", indel)?;
        d.set_item("max_end_keys", PyArray1::from_slice(py, &keys_i64))?;
        d.set_item("hap_lo", hap_lo)?;
        d.set_item("hap_hi", hap_hi)?;
        Ok(d)
    }
}

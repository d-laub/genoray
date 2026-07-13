//! M6c: decoded `seqpro.rag.Ragged` materialization + decode-free counts on
//! `PyContigReader`. Owned by the `svar-2-m6c` worktree; separate `#[pymethods]`
//! block (multiple-pymethods) so M6b and M6c never touch the same file.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::bits;
use crate::field::StorageDtype;
use crate::layout::{ContigPaths, FieldSub};
use crate::py_convert::{i32_to_pyarray, u8_to_pyarray};
use crate::py_query::PyContigReader;
use crate::query::gather::overlap_batch_src;
use crate::query::{BatchResult, ContigReader, FieldValue, FieldView, overlap_batch};

/// The two legal field categories `decode_batch_fields` accepts. Parsing the
/// caller-supplied `category: &str` into this enum up front means a typo (or
/// any other garbage string) surfaces as a Python `ValueError` from
/// `OpenField::open`, rather than silently opening four empty sub-streams
/// (`FieldView::open`'s "missing file = legal empty sub-stream" contract) and
/// panicking later in `bytes_at`/`value_at` — which would cross FFI as an
/// opaque `PanicException`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldCategoryArg {
    Info,
    Format,
}

impl FieldCategoryArg {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "info" => Some(Self::Info),
            "format" => Some(Self::Format),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Format => "format",
        }
    }
}

/// Why `OpenField::open` failed, kept distinct from a `PyErr` so the helper
/// stays plain Rust (testable with no `Python<'py>`/GIL) and the pymethod
/// wrapper decides the exact exception type.
#[derive(Debug)]
pub enum OpenFieldError {
    /// Bad `category` string or unresolved/unknown `dtype` string — a
    /// `ValueError` in Python: the caller passed a bad manifest entry.
    BadInput(String),
    /// The on-disk sidecar could not be mmap'd — an `OSError` in Python.
    Io(std::io::Error),
}

impl OpenFieldError {
    fn into_pyerr(self) -> PyErr {
        match self {
            Self::BadInput(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            Self::Io(e) => pyo3::exceptions::PyOSError::new_err(e.to_string()),
        }
    }
}

/// One requested field's four open sub-stream views (`FieldSub::all()` order:
/// `VkSnp, VkIndel, DenseSnp, DenseIndel`), plus the metadata
/// `gather_field_bytes` needs to select the right sub-stream and stride a
/// dense-FORMAT column correctly.
pub struct OpenField {
    key: String,
    is_format: bool,
    width: usize,
    views: [FieldView; 4],
}

impl OpenField {
    /// Open all four sub-streams for one `(category, name)` field at
    /// `base_dir`/`contig`. `dtype_str` is the finalized `meta.json` dtype
    /// string. Fails on a bad category, an unresolved/unknown dtype, or an
    /// I/O error opening a sidecar — never panics.
    pub fn open(
        paths: &ContigPaths,
        category: &str,
        name: &str,
        dtype_str: &str,
        n_samples_cohort: usize,
    ) -> Result<Self, OpenFieldError> {
        let cat = FieldCategoryArg::parse(category)
            .ok_or_else(|| OpenFieldError::BadInput(format!("bad field category {category:?}")))?;
        let dtype = StorageDtype::from_meta_str(dtype_str).ok_or_else(|| {
            OpenFieldError::BadInput(format!(
                "field {name:?} has unresolved/unknown dtype {dtype_str:?}"
            ))
        })?;
        let width = dtype
            .width_bytes()
            .expect("from_meta_str rejects unresolved dtypes");
        let mut views = Vec::with_capacity(4);
        for sub in FieldSub::all() {
            views.push(
                FieldView::open(paths, cat.as_str(), name, sub, dtype, n_samples_cohort)
                    .map_err(OpenFieldError::Io)?,
            );
        }
        Ok(Self {
            key: format!("{}/{name}", cat.as_str()),
            is_format: cat == FieldCategoryArg::Format,
            width,
            views: views
                .try_into()
                .unwrap_or_else(|_| unreachable!("FieldSub::all() always yields exactly 4")),
        })
    }
}

/// Reduce a `FieldView` element to its raw little-endian bytes, left-aligned
/// in a fixed 4-byte buffer (callers slice to `width`). Round-trips bit-exact
/// for every `StorageDtype`: every variant was itself decoded via
/// `from_le_bytes` in `FieldView::value_at`, so re-encoding via `to_le_bytes`
/// reverses that exactly — no arithmetic, so NaN/sentinel bit patterns survive
/// untouched.
fn field_value_le_bytes(v: FieldValue) -> [u8; 4] {
    match v {
        FieldValue::Bool(b) => [b as u8, 0, 0, 0],
        FieldValue::I8(x) => [x as u8, 0, 0, 0],
        FieldValue::U8(x) => [x, 0, 0, 0],
        FieldValue::I16(x) => {
            let b = x.to_le_bytes();
            [b[0], b[1], 0, 0]
        }
        FieldValue::U16(x) => {
            let b = x.to_le_bytes();
            [b[0], b[1], 0, 0]
        }
        FieldValue::I32(x) => x.to_le_bytes(),
        FieldValue::U32(x) => x.to_le_bytes(),
        FieldValue::F16(x) => {
            let b = x.to_le_bytes();
            [b[0], b[1], 0, 0]
        }
        FieldValue::F32(x) => x.to_le_bytes(),
    }
}

/// Gather one raw-bytes buffer per requested field, aligned 1:1 with `br`'s
/// decoded records in exactly `decode_hap_src`'s record order (region-major
/// `h = (r*n_samples + s)*ploidy + p`, then per-hap merge order). Pure Rust —
/// no `Python<'py>` in the signature — so this is directly unit-testable with
/// no GIL. `br` MUST come from `overlap_batch_src` (asserted transitively by
/// `decode_hap_src`).
///
/// For a dense FORMAT field, the element is read via `FieldView::format_at`
/// (dense row, ORIGINAL cohort sample index) rather than hand-deriving the
/// `row * n_samples + sample` stride here — that formula has exactly one
/// copy, in `FieldView::format_at` itself. Every other case (var_key, dense
/// INFO) reads via `FieldView::value_at`, which `format_at` itself calls.
pub fn gather_field_bytes(
    reader: &ContigReader,
    br: &BatchResult,
    fields: &[OpenField],
) -> Vec<Vec<u8>> {
    let mut fbytes: Vec<Vec<u8>> = vec![Vec::new(); fields.len()];
    for r in 0..br.n_regions {
        for s in 0..br.n_samples {
            for p in 0..br.ploidy {
                let (_, srcs) = br.decode_hap_src(reader, r, s, p);
                for src in srcs {
                    // FieldSub::all() order: VkSnp, VkIndel, DenseSnp, DenseIndel
                    let sub_ix = match (src.is_dense, src.is_indel) {
                        (false, false) => 0,
                        (false, true) => 1,
                        (true, false) => 2,
                        (true, true) => 3,
                    };
                    for (fi, f) in fields.iter().enumerate() {
                        let view = &f.views[sub_ix];
                        // Dense FORMAT strides by the ORIGINAL cohort sample
                        // index. `s` here is already the cohort index because
                        // `overlap_batch_src` runs over the whole cohort.
                        let value = if src.is_dense && f.is_format {
                            view.format_at(src.idx, s)
                        } else {
                            view.value_at(src.idx)
                        };
                        let bytes = field_value_le_bytes(value);
                        fbytes[fi].extend_from_slice(&bytes[..f.width]);
                    }
                }
            }
        }
    }
    fbytes
}

#[pymethods]
impl PyContigReader {
    /// Decode every hap's overlapping variants across `regions` into flat, shared
    /// variant-axis buffers (the M6c materialization primitive). Hap order is
    /// region-major `h = (r*n_samples + s)*ploidy + p`.
    // pub: called directly by the external integration-test crate (see PyContigReader::new)
    pub fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let br = overlap_batch(&self.inner, &regions);
        let mut pos: Vec<i32> = Vec::new();
        let mut ilen: Vec<i32> = Vec::new();
        let mut allele: Vec<u8> = Vec::new();
        let mut str_off: Vec<i64> = vec![0];
        let mut off: Vec<i64> = vec![0];

        for r in 0..br.n_regions {
            for s in 0..br.n_samples {
                for p in 0..br.ploidy {
                    let hc = br.decode_hap(&self.inner, r, s, p);
                    for i in 0..hc.positions.len() {
                        pos.push(hc.positions[i] as i32);
                        ilen.push(hc.ilens[i]);
                        allele.extend_from_slice(&hc.alts[i]);
                        str_off.push(allele.len() as i64);
                    }
                    off.push(pos.len() as i64);
                }
            }
        }

        let d = PyDict::new(py);
        d.set_item("pos", i32_to_pyarray(py, &pos))?;
        d.set_item("ilen", i32_to_pyarray(py, &ilen))?;
        d.set_item("allele", u8_to_pyarray(py, &allele))?;
        d.set_item("str_off", PyArray1::from_slice(py, &str_off))?;
        d.set_item("off", PyArray1::from_slice(py, &off))?;
        d.set_item("n_regions", br.n_regions)?;
        d.set_item("n_samples", br.n_samples)?;
        d.set_item("ploidy", br.ploidy)?;
        Ok(d)
    }

    /// As `decode_batch`, plus one raw-bytes buffer per requested field, aligned
    /// 1:1 with the decoded records. `fields` is `(category, name, dtype_str)`.
    /// Values are returned as little-endian bytes + an itemsize; Python applies
    /// the numpy dtype (same trick as `allele`), so Rust does no dtype dispatch.
    ///
    /// Thin marshalling wrapper: `OpenField::open` + `gather_field_bytes` do the
    /// actual work and are plain Rust (no `Python<'py>`), so they are covered by
    /// a Rust-only test in `tests/test_field_provenance.rs` with no GIL needed.
    #[pyo3(signature = (regions, fields, base_dir, contig))]
    pub fn decode_batch_fields<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        fields: Vec<(String, String, String)>,
        base_dir: &str,
        contig: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let br = overlap_batch_src(&self.inner, &regions);
        let paths = ContigPaths::new(base_dir, contig);
        let n_samples_cohort = self.inner.n_samples();

        let mut open: Vec<OpenField> = Vec::with_capacity(fields.len());
        for (category, name, dtype_str) in &fields {
            open.push(
                OpenField::open(&paths, category, name, dtype_str, n_samples_cohort)
                    .map_err(OpenFieldError::into_pyerr)?,
            );
        }

        let mut pos: Vec<i32> = Vec::new();
        let mut ilen: Vec<i32> = Vec::new();
        let mut allele: Vec<u8> = Vec::new();
        let mut str_off: Vec<i64> = vec![0];
        let mut off: Vec<i64> = vec![0];

        for r in 0..br.n_regions {
            for s in 0..br.n_samples {
                for p in 0..br.ploidy {
                    let hc = br.decode_hap(&self.inner, r, s, p);
                    for i in 0..hc.positions.len() {
                        pos.push(hc.positions[i] as i32);
                        ilen.push(hc.ilens[i]);
                        allele.extend_from_slice(&hc.alts[i]);
                        str_off.push(allele.len() as i64);
                    }
                    off.push(pos.len() as i64);
                }
            }
        }

        let fbytes = gather_field_bytes(&self.inner, &br, &open);

        let d = PyDict::new(py);
        d.set_item("pos", i32_to_pyarray(py, &pos))?;
        d.set_item("ilen", i32_to_pyarray(py, &ilen))?;
        d.set_item("allele", u8_to_pyarray(py, &allele))?;
        d.set_item("str_off", PyArray1::from_slice(py, &str_off))?;
        d.set_item("off", PyArray1::from_slice(py, &off))?;
        d.set_item("n_regions", br.n_regions)?;
        d.set_item("n_samples", br.n_samples)?;
        d.set_item("ploidy", br.ploidy)?;
        for (fi, f) in open.iter().enumerate() {
            d.set_item(format!("field_{}", f.key), u8_to_pyarray(py, &fbytes[fi]))?;
            d.set_item(format!("field_itemsize_{}", f.key), f.width)?;
        }
        Ok(d)
    }

    /// Per-hap variant count over `regions` WITHOUT decoding: var_key slice length
    /// (`vk_off` diff) + dense-present popcount. Flat length `H`, region-major hap
    /// order; the caller reshapes to `(n_regions, n_samples, ploidy)`. The simplified
    /// `SparseVar.var_ranges` replacement.
    // pub: called directly by the external integration-test crate (see PyContigReader::new)
    pub fn region_counts<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let br = overlap_batch(&self.inner, &regions);
        let h = br.n_regions * br.n_samples * br.ploidy;
        let mut counts: Vec<i64> = Vec::with_capacity(h);
        for i in 0..h {
            let vk_n = br.vk_off[i + 1] - br.vk_off[i];
            let (b0, b1) = (br.dense_present_off[i], br.dense_present_off[i + 1]);
            let dn = (b0..b1)
                .filter(|&b| bits::get_bit(&br.dense_present, b))
                .count();
            counts.push((vk_n + dn) as i64);
        }
        Ok(PyArray1::from_slice(py, &counts))
    }
}

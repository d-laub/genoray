//! Rust→numpy array-conversion helpers shared by the M6 consumers (M6b raw
//! two-channel exposure, M6c decoded materialization). One place for the
//! frozen-contract dtype conversions so the plumbing is not duplicated.

use numpy::PyArray1;
use pyo3::prelude::*;

/// Bit-preserving `u32` → `i32` numpy array. Positions and 32-bit keys both cross
/// the FFI as their `i32` bit-pattern; `x as i32` in Rust reinterprets the bits
/// (no value clamping), so the numpy side recovers the original `u32` via
/// `.view(np.uint32)`.
pub fn u32_to_i32_pyarray<'py>(py: Python<'py>, xs: &[u32]) -> Bound<'py, PyArray1<i32>> {
    let v: Vec<i32> = xs.iter().map(|&x| x as i32).collect();
    PyArray1::from_slice(py, &v)
}

/// `i32` slice → numpy array (ilens, flattened index ranges).
pub fn i32_to_pyarray<'py>(py: Python<'py>, xs: &[i32]) -> Bound<'py, PyArray1<i32>> {
    PyArray1::from_slice(py, xs)
}

/// `usize` slice → `i64` numpy array (CSR offsets, bitmask bit-offsets).
pub fn usize_to_i64_pyarray<'py>(py: Python<'py>, xs: &[usize]) -> Bound<'py, PyArray1<i64>> {
    let v: Vec<i64> = xs.iter().map(|&x| x as i64).collect();
    PyArray1::from_slice(py, &v)
}

/// `u8` slice → numpy array (presence bitmasks, LUT bytes, packed alleles).
pub fn u8_to_pyarray<'py>(py: Python<'py>, xs: &[u8]) -> Bound<'py, PyArray1<u8>> {
    PyArray1::from_slice(py, xs)
}

#[cfg(test)]
mod tests {
    use numpy::PyArrayMethods;

    use super::*;

    #[test]
    fn test_u32_to_i32_preserves_bit_pattern() {
        Python::attach(|py| {
            let xs: Vec<u32> = vec![0, 100, 0x8000_0001, u32::MAX];
            let arr = u32_to_i32_pyarray(py, &xs);
            let ro = arr.readonly();
            let back = ro.as_slice().unwrap();
            let expect: Vec<i32> = xs.iter().map(|&x| x as i32).collect();
            assert_eq!(back, &expect[..]);
        });
    }

    #[test]
    fn test_i32_roundtrip() {
        Python::attach(|py| {
            let xs: Vec<i32> = vec![-3, 0, 1, 42];
            let arr = i32_to_pyarray(py, &xs);
            let ro = arr.readonly();
            assert_eq!(ro.as_slice().unwrap(), &xs[..]);
        });
    }

    #[test]
    fn test_usize_to_i64() {
        Python::attach(|py| {
            let xs: Vec<usize> = vec![0, 1, 1, 5];
            let arr = usize_to_i64_pyarray(py, &xs);
            let ro = arr.readonly();
            assert_eq!(ro.as_slice().unwrap(), &[0i64, 1, 1, 5]);
        });
    }

    #[test]
    fn test_u8_roundtrip() {
        Python::attach(|py| {
            let xs: Vec<u8> = vec![0b1010_0101, 0x00, 0xFF];
            let arr = u8_to_pyarray(py, &xs);
            let ro = arr.readonly();
            assert_eq!(ro.as_slice().unwrap(), &xs[..]);
        });
    }
}

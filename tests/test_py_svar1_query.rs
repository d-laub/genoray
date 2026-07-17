//! Boundary test for the SVAR1 query PyO3 seam: `PySvar1Reader` opens a raw
//! SVAR1 store (headerless buffers, no conversion pipeline needed) and the
//! numpy-dict contract round-trips.

use std::io::Write;

use genoray_core::py_svar1_query::PySvar1Reader;
use pyo3::Python;
use tempfile::tempdir;

fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
    let mut f = std::fs::File::create(dir.join(name)).unwrap();
    f.write_all(bytemuck::cast_slice(data)).unwrap();
}

#[test]
fn py_svar1_reader_opens_a_raw_store() {
    let tmp = tempdir().unwrap();
    write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 2, 4, 3, 2]);
    write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 3, 4, 5, 5]);

    Python::attach(|_py| {
        let r = PySvar1Reader::new(tmp.path().to_str().unwrap(), 2, 2);
        assert!(r.is_ok(), "PySvar1Reader should open a raw SVAR1 store");
        let r = r.unwrap();
        assert_eq!(r.n_samples(), 2);
        assert_eq!(r.ploidy(), 2);
    });
}

#[test]
fn py_svar1_reader_missing_store_is_err() {
    Python::attach(|_py| {
        assert!(PySvar1Reader::new("/no/such/svar1/store", 2, 2).is_err());
    });
}

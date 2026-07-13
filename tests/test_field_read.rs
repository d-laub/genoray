//! Proves the field-read API is reachable from OUTSIDE the crate — the same way
//! GenVarLoader consumes it (Cargo path-dep, `default-features = false`). This
//! is the Milestone-1 acceptance test: everything gvl needs (`FieldView`,
//! `FieldValue`, `vk_src` pack/unpack, `gather_haps_readbound_src`, and
//! `dense_abs_row`) must be reachable via a short, stable `genoray_core::...`
//! path without reaching into private modules.

use genoray_core::field::StorageDtype;
use genoray_core::layout::{ContigPaths, FieldSub};
use genoray_core::query::gather::dense_abs_row;
use genoray_core::query::{FieldValue, FieldView, pack_vk_src, unpack_vk_src};

#[test]
fn field_view_is_publicly_constructible() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
    let p = paths.field_values("format", "DS", FieldSub::VkSnp);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    let vals: [f32; 3] = [0.0, 1.5, 2.0];
    std::fs::write(&p, bytemuck::cast_slice(&vals)).unwrap();

    let v = FieldView::open(
        &paths,
        "format",
        "DS",
        FieldSub::VkSnp,
        StorageDtype::F32,
        1,
    )
    .unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(v.value_at(1), FieldValue::F32(1.5));
    assert_eq!(v.as_slice::<f32>().unwrap(), &vals[..]);
}

#[test]
fn vk_src_helpers_are_public() {
    let s = pack_vk_src(true, 42);
    assert_eq!(unpack_vk_src(s), (true, 42));
}

#[test]
fn dense_abs_row_is_public_and_correct() {
    // on-disk window [10, 16), output window [0, 6) — pure translation.
    let on_disk = 10usize..16usize;
    let out = 0usize..6usize;
    assert_eq!(dense_abs_row(&on_disk, &out, 0), 10);
    assert_eq!(dense_abs_row(&on_disk, &out, 5), 15);

    // Out window need not start at 0 (e.g. a second query's slice appended
    // after a first): the offset within `out` is what matters.
    let out2 = 6usize..12usize;
    assert_eq!(dense_abs_row(&on_disk, &out2, 6), 10);
    assert_eq!(dense_abs_row(&on_disk, &out2, 11), 15);
}

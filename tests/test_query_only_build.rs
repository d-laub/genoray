//! Compile-guard: the query core must build & link without the `conversion`
//! feature (no rust-htslib). If this file compiles under
//! `--no-default-features`, the gate is correct.
#[test]
fn query_core_symbols_are_reachable_without_conversion() {
    // Referencing these paths forces the query core to be part of the
    // no-default-features build graph.
    use genoray_core::query::{ContigReader, find_ranges, gather_ranges};
    let _ = ContigReader::open;
    let _ = find_ranges;
    let _ = gather_ranges;
}

/// The SVAR1 query core must ALSO build without `conversion`. This is the whole
/// point of `svar1_query` existing separately from the conversion-gated
/// `svar1_reader`: gvl links `genoray_core` with `default-features = false` and
/// must be able to query SVAR1 with no htslib.
///
/// NOTE: `test-rust` always runs with `conversion` ON, so this test passing under
/// `cargo test` proves nothing by itself — the gate is enforced by
/// `pixi run -e lint check-core` (`cargo check --no-default-features`), which is
/// what actually compiles this file without the feature.
#[test]
fn svar1_query_symbols_are_reachable_without_conversion() {
    use genoray_core::svar1_query::{Svar1Reader, find_ranges, var_ranges};
    let _ = Svar1Reader::open;
    let _ = var_ranges;
    let _ = find_ranges;
}

/// The PyO3 seam for SVAR1 queries must be ungated too (`py_query` already is).
#[test]
fn py_svar1_query_symbols_are_reachable_without_conversion() {
    use genoray_core::py_svar1_query::PySvar1Reader;
    let _ = PySvar1Reader::new;
}

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

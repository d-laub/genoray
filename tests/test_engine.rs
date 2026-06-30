// use proptest::prelude::*;
// use crossbeam_channel::bounded;
// use ndarray::Array3;
// use genoray_core::{types::DenseChunk, executor, nrvk::LongAlleleTableWriter};

// // Property-Based Testing for the core math engine
// proptest! {
// #![proptest_config(ProptestConfig::with_cases(5))]

// #[test]
// fn test_dense_to_sparse_mutation_conservation(
//     num_variants in 1..50usize,
//     num_samples in 1..10usize,
//     // Generate a random boolean grid representing alt-allele presence
//     raw_grid in prop::collection::vec(any::<bool>(), 0..1000)
// ) {
//     // Constrain the grid size to perfectly match the 3D dimensions (V, S, P)
//     let ploidy = 2;
//     let total_elements = num_variants * num_samples * ploidy;

//     let mut safe_grid = raw_grid;
//     safe_grid.resize(total_elements, false); // Pad or truncate to fit

//     // Count the exact number of `true` values (mutations) in the raw data
//     let expected_total_mutations: usize = safe_grid.iter().filter(|&&b| b).count();

//     // 1. Build a structurally perfect ALT string and offsets array
//     let mut mock_alt_bytes = Vec::new();
//     let mut mock_alt_offsets = Vec::new(); // Start empty
//     let mut mock_ilens = Vec::new();

//     // The first offset is ALWAYS 0
//     let mut current_offset = 0u32;
//     mock_alt_offsets.push(current_offset);

//     for _ in 0..num_variants {
//         // A perfect 1-base SNP (e.g., A -> G)
//         let variant_alt = b"G";
//         mock_alt_bytes.extend_from_slice(variant_alt);

//         // Advance the offset tracker by the length of the string (1 byte)
//         current_offset += variant_alt.len() as u32;
//         mock_alt_offsets.push(current_offset);

//         // For a 1-base substitution (A -> G), ilen (alt.len - ref.len) is 0
//         mock_ilens.push(0);
//     }

//     let chunk = DenseChunk {
//         chunk_id: 0,
//         pos: vec![1000; num_variants],
//         ilens: mock_ilens,
//         alt: mock_alt_bytes,
//         alt_offsets: mock_alt_offsets,
//         num_variants,
//         genos: Array3::from_shape_vec((num_variants, num_samples, ploidy), safe_grid).unwrap(),
//     };

//     // 2. Run Engine
//     let (tx_dense, rx_dense) = bounded(1);
//     let (tx_sparse, rx_sparse) = bounded(1);
//     let (tx_long, _rx_long) = bounded(1);

//     tx_dense.send(chunk).unwrap();
//     drop(tx_dense);

//     let bank = LongAlleleTableWriter::new(tx_long, 1024 * 1024);

//     let (ram_ledger, _long_allele_offsets) = executor::run_compute_engine(rx_dense, tx_sparse, bank);

//     let actual_total_mutations: usize = ram_ledger[0].iter().map(|&x| x as usize).sum();

//     prop_assert_eq!(
//         expected_total_mutations,
//         actual_total_mutations,
//         "CRITICAL: The sparse transposer lost or hallucinated mutations!"
//     );
// }
// }

// // // In tests/test_engine.rs
// // use proptest::prelude::*;
// // use ndarray::Array3;

// // proptest! {
// //     #[test]
// //     fn test_dense_to_sparse_invariants(
// //         num_variants in 1..1000usize,
// //         num_samples in 1..100usize,
// //         // Generate a flat vector of booleans to reshape into a 3D grid
// //         grid_data in prop::collection::vec(any::<bool>(), 1..=200000)
// //     ) {
// //         // ... (Use grid_data to populate a mock DenseChunk)
// //         // ... (Run your compute engine)

// //         // Assert the Core Invariant:
// //         // The sum of all `true` values in the DenseGrid MUST exactly equal
// //         // the total length of the resulting Sparse arrays. No mutations lost!
// //     }
// // }

// // use crossbeam_channel::bounded;
// // use core::{types::DenseChunk, executor, nrvk::LongAlleleWriter};
// // use ndarray::Array3;

// // #[test]
// // fn test_dense_to_sparse_logic() {
// //     let (tx_dense, rx_dense) = bounded(1);
// //     let (tx_sparse, rx_sparse) = bounded(1);
// //     let (tx_long, _rx_long) = bounded(1);

// //     // 1. Mock a 2-Variant, 2-Sample, Diploid Dense Grid
// //     let mock_chunk = DenseChunk {
// //         chunk_id: 0,
// //         pos: vec![100, 200],
// //         ilens: vec![0, 0],
// //         alt: b"AG".to_vec(),
// //         alt_offsets: vec![0, 1, 2],
// //         num_variants: 2,
// //         genos: Array3::from_shape_vec((2, 2, 2), vec![
// //             true, false,  // Var 0, Sample 0: (1|0)
// //             false, false, // Var 0, Sample 1: (0|0) -> SHOULD BE DROPPED
// //             true, true,   // Var 1, Sample 0: (1|1)
// //             false, true,  // Var 1, Sample 1: (0|1)
// //         ]).unwrap(),
// //     };

// //     tx_dense.send(mock_chunk).unwrap();
// //     drop(tx_dense); // Close channel to let executor finish

// //     let bank = LongAlleleWriter::new(tx_long, 1024);

// //     // 2. Run the math engine
// //     let (ram_ledger, _) = executor::run_compute_engine(rx_dense, tx_sparse, bank);

// //     // 3. Assert the Math (The RAM Ledger drives the whole disk merger)
// //     // Sample 0 has 3 mutations total. Sample 1 has 1 mutation total.
// //     assert_eq!(ram_ledger[0][0], 3, "Sample 0 mutation count wrong");
// //     assert_eq!(ram_ledger[0][1], 1, "Sample 1 mutation count wrong");
// // }

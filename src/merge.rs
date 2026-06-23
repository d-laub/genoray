use ndarray::Array1;
use ndarray_npy::write_npy;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

// performs the K-Way Sequential Merge to build the final Sample-Major tensors.
// sparce column merge -> will change as causing read thrashing now
//
// `chunk_lengths` is the in-memory ledger produced by the executor: for each chunk,
// the number of calls per (sample, ploid) column.
pub fn merge_mini_sc(
    num_chunks: usize,
    num_samples: usize,
    ploidy: usize,
    output_dir: &str,
    chunk_lengths: Vec<Vec<u32>>,
) {
    let total_columns = num_samples * ploidy;

    // open all Chunk Files for sequential reading
    let mut pos_readers: Vec<BufReader<File>> = (0..num_chunks)
        .map(|c| {
            let f = File::open(format!("{}/chunk_{}_pos.bin", output_dir, c)).unwrap();
            BufReader::with_capacity(1024 * 1024, f) // 1MB read buffer
        })
        .collect();

    let mut key_readers: Vec<BufReader<File>> = (0..num_chunks)
        .map(|c| {
            let f = File::open(format!("{}/chunk_{}_key.bin", output_dir, c)).unwrap();
            BufReader::with_capacity(1024 * 1024, f) // 1MB read buffer
        })
        .collect();

    // open the single fat output files for sequential writing
    let mut final_pos_file =
        BufWriter::new(File::create(format!("{}/positions.bin", output_dir)).unwrap());
    let mut final_key_file =
        BufWriter::new(File::create(format!("{}/alleles.bin", output_dir)).unwrap());

    // global Offsets array (size N + 1)
    let mut final_offsets = Vec::with_capacity(total_columns + 1);
    let mut current_global_offset = 0u64;
    final_offsets.push(current_global_offset);

    // reusable byte buffers to avoid heap allocations in the hot loop
    let mut pos_buffer = Vec::new();
    let mut key_buffer = Vec::new();

    println!(
        "Executing K-Way Block Merge across {} chunks...",
        num_chunks
    );

    // merging. col_idx indexes a different per-chunk vector each inner iteration,
    // so a range loop is the clearest expression here.
    #[allow(clippy::needless_range_loop)]
    for col_idx in 0..total_columns {
        for chunk_id in 0..num_chunks {
            let num_calls = chunk_lengths[chunk_id][col_idx] as usize;
            if num_calls == 0 {
                continue;
            }

            // both Pos and Key are u32, so they are exactly 4 bytes per call
            let bytes_to_read = num_calls * 4;

            // resize buffers exactly (without losing capacity)
            pos_buffer.resize(bytes_to_read, 0);
            key_buffer.resize(bytes_to_read, 0);

            // read exact bytes from the chunks
            pos_readers[chunk_id].read_exact(&mut pos_buffer).unwrap();
            key_readers[chunk_id].read_exact(&mut key_buffer).unwrap();

            // append directly to the fat files
            final_pos_file.write_all(&pos_buffer).unwrap();
            final_key_file.write_all(&key_buffer).unwrap();

            current_global_offset += num_calls as u64;
        }

        // lock in the global offset boundary for this genome copy
        final_offsets.push(current_global_offset);
    }

    // flush writers to physical disk
    final_pos_file.flush().unwrap();
    final_key_file.flush().unwrap();

    // save the final offsets array
    println!("Saving global offsets tensor...");
    let offsets_array = Array1::from_vec(final_offsets);
    write_npy(format!("{}/offsets.npy", output_dir), &offsets_array)
        .expect("Failed to write final offsets");

    // cleanup Temporary Chunks
    for c in 0..num_chunks {
        let _ = std::fs::remove_file(format!("{}/chunk_{}_pos.bin", output_dir, c));
        let _ = std::fs::remove_file(format!("{}/chunk_{}_key.bin", output_dir, c));
    }

    println!("Consolidation Complete. Monolithic Sample-Major sparse tensors are ready.");
}

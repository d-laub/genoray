use std::fs::File;
use std::io::{BufWriter, Read, Write, Seek, SeekFrom};
use ndarray_npy::write_npy;
use ndarray::Array1;
use bytemuck;

/// Performs the Tile-Based Interleaving Merge.
pub fn merge_mini_sc(
    num_chunks: usize,
    num_samples: usize,
    ploidy: usize,
    output_dir: &str,
    ram_ledger: Vec<Vec<u32>>, 
) {
    let total_columns = num_samples * ploidy;
    
    // tile size -> # of columns to merge together in memory at once.
    let columns_per_tile = 10_000;  // rought estimate to take 400MB

    println!("Phase A -> Executing In-Memory Metadata Pass");

    // pre-compute global offsets and local chunk offsets using the RAM Ledger
    let mut final_offsets = vec![0u64; total_columns + 1];
    let mut chunk_offsets = vec![vec![0u32; total_columns + 1]; num_chunks];

    for col in 0..total_columns {
        let mut col_total = 0u64;
        
        for chunk_id in 0..num_chunks {
            let calls = ram_ledger[chunk_id][col];
            chunk_offsets[chunk_id][col + 1] = chunk_offsets[chunk_id][col] + calls;
            col_total += calls as u64;
        }
        final_offsets[col + 1] = final_offsets[col] + col_total;
    }

    // save the global offsets array immediately
    let offsets_array = Array1::from_vec(final_offsets.clone());
    write_npy(format!("{}/final_offsets.npy", output_dir), &offsets_array)
        .expect("Failed to write final offsets");

    println!("Phase B -> Executing Tile-Based Interleaving Gather");

    // opening the single monolithic output files for strict sequential writing
    let mut final_pos_file = BufWriter::new(
        File::create(format!("{}/final_positions.bin", output_dir)).unwrap()
    );
    let mut final_key_file = BufWriter::new(
        File::create(format!("{}/final_keys.bin", output_dir)).unwrap()
    );

    // the tile loop
    for tile_start_col in (0..total_columns).step_by(columns_per_tile) {
        let tile_end_col = std::cmp::min(tile_start_col + columns_per_tile, total_columns);
        
        let tile_start_item = final_offsets[tile_start_col] as usize;
        let tile_end_item = final_offsets[tile_end_col] as usize;
        let tile_total_items = tile_end_item - tile_start_item;

        if tile_total_items == 0 { continue; }

        let mut tile_pos_buffer = vec![0u32; tile_total_items];
        let mut tile_key_buffer = vec![0u32; tile_total_items];

        let mut tile_write_heads = vec![0usize; tile_end_col - tile_start_col];
        for i in 0..(tile_end_col - tile_start_col) {
            let col = tile_start_col + i;
            tile_write_heads[i] = (final_offsets[col] as usize) - tile_start_item;
        }

        // gather from chunks 
        for chunk_id in 0..num_chunks {
            let chunk_start_item = chunk_offsets[chunk_id][tile_start_col] as usize;
            let chunk_end_item = chunk_offsets[chunk_id][tile_end_col] as usize;
            let chunk_items_to_read = chunk_end_item - chunk_start_item;

            if chunk_items_to_read == 0 { continue; }

            // Open, seek, read ONE massive block, then automatically close.
            let mut pos_file = File::open(format!("{}/chunk_{}_pos.bin", output_dir, chunk_id)).unwrap();
            let mut key_file = File::open(format!("{}/chunk_{}_key.bin", output_dir, chunk_id)).unwrap();

            pos_file.seek(SeekFrom::Start((chunk_start_item * 4) as u64)).unwrap();
            key_file.seek(SeekFrom::Start((chunk_start_item * 4) as u64)).unwrap();

            let mut chunk_pos_bytes = vec![0u8; chunk_items_to_read * 4];
            let mut chunk_key_bytes = vec![0u8; chunk_items_to_read * 4];

            pos_file.read_exact(&mut chunk_pos_bytes).unwrap();
            key_file.read_exact(&mut chunk_key_bytes).unwrap();

            // zero-copy cast back to u32 slices
            let chunk_pos_u32: &[u32] = bytemuck::cast_slice(&chunk_pos_bytes);
            let chunk_key_u32: &[u32] = bytemuck::cast_slice(&chunk_key_bytes);

            // stitch this chunk's block into the main Tile buffer
            let mut local_chunk_cursor = 0;
            
            for i in 0..(tile_end_col - tile_start_col) {
                let col = tile_start_col + i;
                let calls = ram_ledger[chunk_id][col] as usize;
                
                if calls == 0 { continue; }

                let dest_start = tile_write_heads[i];
                let dest_end = dest_start + calls;

                tile_pos_buffer[dest_start..dest_end]
                    .copy_from_slice(&chunk_pos_u32[local_chunk_cursor..local_chunk_cursor + calls]);
                
                tile_key_buffer[dest_start..dest_end]
                    .copy_from_slice(&chunk_key_u32[local_chunk_cursor..local_chunk_cursor + calls]);

                tile_write_heads[i] += calls; 
                local_chunk_cursor += calls;
            }
        }

        // sequential drive copy
        let tile_pos_bytes: &[u8] = bytemuck::cast_slice(&tile_pos_buffer);
        let tile_key_bytes: &[u8] = bytemuck::cast_slice(&tile_key_buffer);

        final_pos_file.write_all(tile_pos_bytes).unwrap();
        final_key_file.write_all(tile_key_bytes).unwrap();
    }

    final_pos_file.flush().unwrap();
    final_key_file.flush().unwrap();

    println!("Phase C -> Cleaning up temporary chunk files");
    for c in 0..num_chunks {
        let _ = std::fs::remove_file(format!("{}/chunk_{}_pos.bin", output_dir, c));
        let _ = std::fs::remove_file(format!("{}/chunk_{}_key.bin", output_dir, c));
    }

    println!("Merge Complete.");
}
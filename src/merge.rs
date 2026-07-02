use crate::layout;
use bytemuck;
use ndarray::Array1;
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

// Target peak RAM per tile (per worker). Two u32 buffers per tile, so a 256 MB
// budget caps a tile at ~32M items. Workers run in parallel via rayon — peak
// process RAM = TILE_RAM_BUDGET × rayon_threads.
const TILE_RAM_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Performs the Tile-Based Interleaving Merge.
///
/// Phase A: in-memory metadata pass — derives global per-column offsets and per-chunk
///          local offsets from the RAM Ledger, writes `final_offsets.npy`.
/// Phase B: parallel tile gather — each rayon worker owns one tile, reads the slice
///          of every chunk via positional reads, scatters into per-column slots,
///          then `pwrite`s the assembled tile to its pre-computed byte range in
///          `final_positions.bin` / `final_keys.bin`.
/// Phase C: cleanup of per-chunk temp files.
pub fn merge_mini_sc(
    key_bytes: usize,
    num_chunks: usize,
    num_samples: usize,
    ploidy: usize,
    output_dir: &str,
    ram_ledger: Vec<Vec<u32>>,
) {
    let output_dir_path = Path::new(output_dir);
    let total_columns = num_samples * ploidy;
    let pos_size = std::mem::size_of::<u32>(); // positions are always u32

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
    write_npy(layout::final_offsets(output_dir_path), &offsets_array)
        .expect("Failed to write final offsets");

    let total_items: u64 = final_offsets[total_columns];
    let pos_total_bytes: u64 = total_items * pos_size as u64;
    let key_total_bytes: u64 = total_items * key_bytes as u64;

    println!("Phase B -> Executing Parallel Tile-Based Interleaving Gather");

    // Pre-create the monolithic outputs at full size so worker pwrites land in
    // disjoint byte ranges. set_len doesn't allocate disk space (sparse file)
    // until each tile actually writes.
    let final_pos_file = File::create(layout::final_positions(output_dir_path))
        .expect("Failed to create final_positions.bin");
    final_pos_file
        .set_len(pos_total_bytes)
        .expect("Failed to size final_positions.bin");
    let final_key_file =
        File::create(layout::final_keys(output_dir_path)).expect("Failed to create final_keys.bin");
    final_key_file
        .set_len(key_total_bytes)
        .expect("Failed to size final_keys.bin");

    // Open every chunk's pos/key file exactly once; pread() is stateless and
    // safe to call concurrently from multiple rayon workers.
    let chunk_files: Vec<(File, File)> = (0..num_chunks)
        .map(|c| {
            let pf = File::open(layout::chunk_pos(output_dir_path, c))
                .unwrap_or_else(|e| panic!("Failed to open chunk_{}_pos.bin: {}", c, e));
            let kf = File::open(layout::chunk_key(output_dir_path, c))
                .unwrap_or_else(|e| panic!("Failed to open chunk_{}_key.bin: {}", c, e));
            (pf, kf)
        })
        .collect();

    // Adaptive tile size: target TILE_RAM_BUDGET per tile.
    // pos buffer (u32) + key buffer (key_bytes-wide) → (pos_size + key_bytes) bytes per item.
    let bytes_per_item = (pos_size + key_bytes) as u64;
    let avg_calls_per_col =
        std::cmp::max(1u64, total_items / std::cmp::max(1, total_columns) as u64);
    let columns_per_tile = std::cmp::max(
        1usize,
        std::cmp::min(
            total_columns.max(1),
            (TILE_RAM_BUDGET_BYTES / (avg_calls_per_col * bytes_per_item)) as usize,
        ),
    );
    println!(
        "Tile size: {} columns ({} items, ~{} MB per tile)",
        columns_per_tile,
        columns_per_tile as u64 * avg_calls_per_col,
        (columns_per_tile as u64 * avg_calls_per_col * bytes_per_item) / (1024 * 1024),
    );

    // Tile start columns — independent work units, parallelized across rayon.
    let tile_starts: Vec<usize> = (0..total_columns).step_by(columns_per_tile).collect();

    // All shared state below is read-only or via &File pwrite, which is Sync.
    let final_offsets_ref = &final_offsets;
    let chunk_offsets_ref = &chunk_offsets;
    let ram_ledger_ref = &ram_ledger;
    let chunk_files_ref = &chunk_files;
    let final_pos_ref = &final_pos_file;
    let final_key_ref = &final_key_file;

    tile_starts.par_iter().for_each(|&tile_start_col| {
        let tile_end_col = std::cmp::min(tile_start_col + columns_per_tile, total_columns);
        let tile_n_cols = tile_end_col - tile_start_col;
        let tile_start_item = final_offsets_ref[tile_start_col] as usize;
        let tile_end_item = final_offsets_ref[tile_end_col] as usize;
        let tile_total_items = tile_end_item - tile_start_item;

        if tile_total_items == 0 {
            return;
        }

        let mut tile_pos_buffer = vec![0u32; tile_total_items];
        let mut tile_key_buffer = vec![0u8; tile_total_items * key_bytes];

        // per-column write head (offset within this tile buffer)
        let mut tile_write_heads = vec![0usize; tile_n_cols];
        // index loop: `i` indexes tile_write_heads while `col` offsets into the global ledger.
        #[allow(clippy::needless_range_loop)]
        for i in 0..tile_n_cols {
            let col = tile_start_col + i;
            tile_write_heads[i] = (final_offsets_ref[col] as usize) - tile_start_item;
        }

        // gather from chunks
        for chunk_id in 0..num_chunks {
            let chunk_start_item = chunk_offsets_ref[chunk_id][tile_start_col] as usize;
            let chunk_end_item = chunk_offsets_ref[chunk_id][tile_end_col] as usize;
            let chunk_items_to_read = chunk_end_item - chunk_start_item;

            if chunk_items_to_read == 0 {
                continue;
            }

            // Stateless positional reads — multiple workers can read the same File
            // concurrently without locking or seek-state contention.
            let mut chunk_pos_bytes = vec![0u8; chunk_items_to_read * pos_size];
            let mut chunk_key_bytes = vec![0u8; chunk_items_to_read * key_bytes];
            let pos_byte_offset = (chunk_start_item * pos_size) as u64;
            let key_byte_offset = (chunk_start_item * key_bytes) as u64;
            chunk_files_ref[chunk_id]
                .0
                .read_exact_at(&mut chunk_pos_bytes, pos_byte_offset)
                .expect("Failed to pread chunk pos");
            chunk_files_ref[chunk_id]
                .1
                .read_exact_at(&mut chunk_key_bytes, key_byte_offset)
                .expect("Failed to pread chunk key");

            // zero-copy cast back to typed slice (positions only — keys stay raw bytes)
            let chunk_pos_u32: &[u32] = bytemuck::cast_slice(&chunk_pos_bytes);

            // stitch this chunk's block into the main Tile buffer
            let mut local_chunk_cursor = 0usize;
            // index loop: `i` indexes tile-local write heads while `col` offsets the ledger.
            #[allow(clippy::needless_range_loop)]
            for i in 0..tile_n_cols {
                let col = tile_start_col + i;
                let calls = ram_ledger_ref[chunk_id][col] as usize;
                if calls == 0 {
                    continue;
                }

                let dest_start = tile_write_heads[i];
                let dest_end = dest_start + calls;

                tile_pos_buffer[dest_start..dest_end].copy_from_slice(
                    &chunk_pos_u32[local_chunk_cursor..local_chunk_cursor + calls],
                );

                let key_dest_start = dest_start * key_bytes;
                let key_src_start = local_chunk_cursor * key_bytes;
                tile_key_buffer[key_dest_start..key_dest_start + calls * key_bytes]
                    .copy_from_slice(
                        &chunk_key_bytes[key_src_start..key_src_start + calls * key_bytes],
                    );

                tile_write_heads[i] += calls;
                local_chunk_cursor += calls;
            }
        }

        // pwrite the assembled tile to its known byte range in the final files.
        // Tiles are disjoint by construction (final_offsets is monotonically increasing),
        // so concurrent write_all_at calls touch non-overlapping regions.
        let tile_pos_byte_offset = (tile_start_item * pos_size) as u64;
        let tile_key_byte_offset = (tile_start_item * key_bytes) as u64;
        let tile_pos_bytes: &[u8] = bytemuck::cast_slice(&tile_pos_buffer);
        final_pos_ref
            .write_all_at(tile_pos_bytes, tile_pos_byte_offset)
            .expect("Failed to pwrite tile to final_positions.bin");
        final_key_ref
            .write_all_at(&tile_key_buffer, tile_key_byte_offset)
            .expect("Failed to pwrite tile to final_keys.bin");
    });

    // Drop file handles to flush metadata before cleanup
    drop(chunk_files);
    drop(final_pos_file);
    drop(final_key_file);

    println!("Phase C -> Cleaning up temporary chunk files");
    for c in 0..num_chunks {
        let _ = std::fs::remove_file(layout::chunk_pos(output_dir_path, c));
        let _ = std::fs::remove_file(layout::chunk_key(output_dir_path, c));
    }

    println!("Merge Complete.");
}

#[cfg(test)]
mod tests {
    // Test loops mirror the (column, chunk) ledger structure with explicit indices,
    // which reads more clearly than iterator adapters for the index-math assertions.
    #![allow(clippy::needless_range_loop)]
    use super::*;
    use proptest::prelude::*;
    use std::io::Write;
    use std::path::Path;
    use tempfile::tempdir;

    // Helper: stage one chunk's pos and key arrays to disk in the layout merge expects.
    fn write_chunk_files(dir: &Path, chunk_id: usize, pos: &[u32], key: &[u32]) {
        let mut pf = File::create(dir.join(format!("chunk_{}_pos.bin", chunk_id))).unwrap();
        pf.write_all(bytemuck::cast_slice(pos)).unwrap();
        let mut kf = File::create(dir.join(format!("chunk_{}_key.bin", chunk_id))).unwrap();
        kf.write_all(bytemuck::cast_slice(key)).unwrap();
    }

    fn read_u32_bin(path: &Path) -> Vec<u32> {
        // std::fs::read returns a Vec<u8> with u8 alignment; bytemuck::cast_slice
        // would fail TargetAlignmentGreater when the buffer happens to be unaligned
        // (notably for empty files where Vec::new uses NonNull::dangling()). Use
        // chunks_exact + from_le_bytes — alignment-agnostic.
        let bytes = std::fs::read(path).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_offsets_npy(path: &Path) -> Vec<u64> {
        let arr: ndarray::Array1<u64> = ndarray_npy::read_npy(path).unwrap();
        arr.to_vec()
    }

    // Single chunk passthrough: with one chunk the final files should byte-equal
    // the input chunk (no interleaving across chunks).
    #[test]
    fn test_merge_single_chunk_passthrough() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // 2 samples × 2 ploidy = 4 columns. ram_ledger says [2, 1, 0, 3] calls.
        let ram_ledger = vec![vec![2u32, 1, 0, 3]];
        let pos: Vec<u32> = vec![100, 200, 300, 400, 500, 600]; // 6 total calls
        let key: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
        write_chunk_files(dir, 0, &pos, &key);

        merge_mini_sc(4, 1, 2, 2, dir.to_str().unwrap(), ram_ledger);

        let final_pos = read_u32_bin(&dir.join("final_positions.bin"));
        let final_key = read_u32_bin(&dir.join("final_keys.bin"));
        let final_off = read_offsets_npy(&dir.join("final_offsets.npy"));

        assert_eq!(final_pos, pos);
        assert_eq!(final_key, key);
        assert_eq!(final_off, vec![0u64, 2, 3, 3, 6]);
    }

    // Multi-chunk interleaving: per-sample slices must concatenate chunk-by-chunk
    // in chunk_id order, samples in column order.
    #[test]
    fn test_merge_multi_chunk_interleave() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // 2 samples × 1 ploidy = 2 columns. 2 chunks.
        // Chunk 0: col0=2 calls (a, b), col1=1 call (c)              → [a, b, c]
        // Chunk 1: col0=1 call (d),     col1=2 calls (e, f)          → [d, e, f]
        // Expected final order:
        //   col0: chunk0 (a, b) + chunk1 (d)        → a, b, d
        //   col1: chunk0 (c)    + chunk1 (e, f)     → c, e, f
        // → [a, b, d, c, e, f]
        let ram_ledger = vec![vec![2u32, 1], vec![1u32, 2]];
        write_chunk_files(dir, 0, &[100, 200, 300], &[1, 2, 3]);
        write_chunk_files(dir, 1, &[400, 500, 600], &[4, 5, 6]);

        merge_mini_sc(4, 2, 2, 1, dir.to_str().unwrap(), ram_ledger);

        let final_pos = read_u32_bin(&dir.join("final_positions.bin"));
        let final_key = read_u32_bin(&dir.join("final_keys.bin"));
        let final_off = read_offsets_npy(&dir.join("final_offsets.npy"));

        assert_eq!(final_pos, vec![100, 200, 400, 300, 500, 600]);
        assert_eq!(final_key, vec![1, 2, 4, 3, 5, 6]);
        assert_eq!(final_off, vec![0u64, 3, 6]);
    }

    // Edge: every column has zero calls → final files exist, are empty, offsets are all zero.
    #[test]
    fn test_merge_all_empty() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        let ram_ledger = vec![vec![0u32; 4]];
        write_chunk_files(dir, 0, &[], &[]);

        merge_mini_sc(4, 1, 2, 2, dir.to_str().unwrap(), ram_ledger);

        let final_pos = read_u32_bin(&dir.join("final_positions.bin"));
        let final_off = read_offsets_npy(&dir.join("final_offsets.npy"));

        assert_eq!(final_pos.len(), 0);
        assert_eq!(final_off, vec![0u64; 5]);
    }

    // Edge: chunk_0 contributes nothing, chunk_1 carries all calls. Validates the
    // tile gather correctly skips zero-call chunks for a column.
    #[test]
    fn test_merge_skips_empty_chunks() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        let ram_ledger = vec![vec![0u32, 0], vec![3u32, 2]];
        write_chunk_files(dir, 0, &[], &[]);
        write_chunk_files(dir, 1, &[10, 20, 30, 40, 50], &[1, 2, 3, 4, 5]);

        merge_mini_sc(4, 2, 2, 1, dir.to_str().unwrap(), ram_ledger);

        let final_pos = read_u32_bin(&dir.join("final_positions.bin"));
        let final_off = read_offsets_npy(&dir.join("final_offsets.npy"));

        assert_eq!(final_pos, vec![10, 20, 30, 40, 50]);
        assert_eq!(final_off, vec![0u64, 3, 5]);
    }

    // Helper: read a u8 key file (one byte per call).
    fn read_u8_bin(path: &Path) -> Vec<u8> {
        std::fs::read(path).unwrap()
    }

    // The merge must work with u8 keys (the SNP stream), interleaving exactly like
    // the u32 case. Reuses the multi-chunk interleave scenario with 1-byte keys.
    #[test]
    fn test_merge_u8_keys_interleave() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // 2 samples × 1 ploidy = 2 columns, 2 chunks (mirrors test_merge_multi_chunk_interleave).
        let ram_ledger = vec![vec![2u32, 1], vec![1u32, 2]];
        // pos still u32; keys are u8.
        {
            let mut pf = File::create(dir.join("chunk_0_pos.bin")).unwrap();
            pf.write_all(bytemuck::cast_slice(&[100u32, 200, 300]))
                .unwrap();
            let mut kf = File::create(dir.join("chunk_0_key.bin")).unwrap();
            kf.write_all(&[1u8, 2, 3]).unwrap();
            let mut pf = File::create(dir.join("chunk_1_pos.bin")).unwrap();
            pf.write_all(bytemuck::cast_slice(&[400u32, 500, 600]))
                .unwrap();
            let mut kf = File::create(dir.join("chunk_1_key.bin")).unwrap();
            kf.write_all(&[4u8, 5, 6]).unwrap();
        }

        merge_mini_sc(1, 2, 2, 1, dir.to_str().unwrap(), ram_ledger);

        let final_pos = read_u32_bin(&dir.join("final_positions.bin"));
        let final_key = read_u8_bin(&dir.join("final_keys.bin"));
        let final_off = read_offsets_npy(&dir.join("final_offsets.npy"));

        assert_eq!(final_pos, vec![100, 200, 400, 300, 500, 600]);
        assert_eq!(final_key, vec![1, 2, 4, 3, 5, 6]);
        assert_eq!(final_off, vec![0u64, 3, 6]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(150))]

        // Property: for any ram_ledger + chunk data, the final stream is the
        // sample-major concatenation of per-chunk slices in chunk_id order.
        // Catches off-by-one in tile boundaries, cursor mismanagement, and
        // pwrite offset bugs.
        #[test]
        fn test_merge_interleave_property(
            num_chunks in 1usize..5,
            num_samples in 1usize..5,
            ploidy in 1usize..3,
            // Calls per (chunk, column) — bounded so the test is fast
            seed in any::<u64>(),
        ) {
            let total_columns = num_samples * ploidy;

            // Generate ram_ledger and chunk data deterministically from seed
            let mut state = seed | 1;
            let mut next = || {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                state
            };

            let mut ram_ledger: Vec<Vec<u32>> = Vec::with_capacity(num_chunks);
            for _ in 0..num_chunks {
                let row: Vec<u32> = (0..total_columns).map(|_| (next() % 7) as u32).collect();
                ram_ledger.push(row);
            }

            // Stage chunk files. For each chunk, calls are concatenated column-by-column.
            let tmp = tempdir().unwrap();
            let dir = tmp.path();

            // Track expected per-(chunk, column) data so we can reconstruct ground truth.
            // chunk_data[chunk_id][col] = Vec<(pos, key)> for that column in that chunk
            let mut chunk_data: Vec<Vec<Vec<(u32, u32)>>> =
                vec![vec![vec![]; total_columns]; num_chunks];

            for chunk_id in 0..num_chunks {
                let mut pos_buf: Vec<u32> = Vec::new();
                let mut key_buf: Vec<u32> = Vec::new();
                for col in 0..total_columns {
                    let n = ram_ledger[chunk_id][col] as usize;
                    for _ in 0..n {
                        let p = next() as u32;
                        let k = next() as u32;
                        pos_buf.push(p);
                        key_buf.push(k);
                        chunk_data[chunk_id][col].push((p, k));
                    }
                }
                write_chunk_files(dir, chunk_id, &pos_buf, &key_buf);
            }

            merge_mini_sc(4, num_chunks, num_samples, ploidy, dir.to_str().unwrap(), ram_ledger.clone());

            let final_pos = read_u32_bin(&dir.join("final_positions.bin"));
            let final_key = read_u32_bin(&dir.join("final_keys.bin"));
            let final_off = read_offsets_npy(&dir.join("final_positions.bin").with_file_name("final_offsets.npy"));

            // Build expected: walk columns, then chunks within each column.
            let mut expected_pos: Vec<u32> = Vec::new();
            let mut expected_key: Vec<u32> = Vec::new();
            let mut expected_off: Vec<u64> = vec![0];
            for col in 0..total_columns {
                let mut col_total = 0u64;
                for chunk_id in 0..num_chunks {
                    for &(p, k) in &chunk_data[chunk_id][col] {
                        expected_pos.push(p);
                        expected_key.push(k);
                        col_total += 1;
                    }
                }
                expected_off.push(*expected_off.last().unwrap() + col_total);
            }

            prop_assert_eq!(final_pos, expected_pos);
            prop_assert_eq!(final_key, expected_key);
            prop_assert_eq!(final_off, expected_off);
        }
    }
}

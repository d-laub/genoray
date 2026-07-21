use crate::error::ConversionError;
use crate::layout;
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

/// Phase A (shared): derive global per-column offsets and per-chunk local
/// offsets from the RAM Ledger. Both `merge_mini_sc` and
/// `merge_var_key_field_values` need the identical column-major schedule —
/// field values are staged 1:1 with calls, so the same reordering applies
/// regardless of per-item byte width.
///
/// Returns `(final_offsets, chunk_offsets)`:
/// - `final_offsets[col]` is the global item index where column `col` starts
///   (length `total_columns + 1`, monotonically increasing).
/// - `chunk_offsets[chunk_id][col]` is the local item index within
///   `chunk_id`'s own stream where column `col` starts.
fn derive_offsets(
    num_chunks: usize,
    total_columns: usize,
    ram_ledger: &[Vec<u32>],
) -> (Vec<u64>, Vec<Vec<u32>>) {
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

    (final_offsets, chunk_offsets)
}

/// One byte-payload stream sharing the offset/tile schedule computed by
/// `derive_offsets`: per-chunk source files (opened once, read via stateless
/// `pread`) and the pre-sized destination file (written via `pwrite`).
struct Payload<'a> {
    /// Byte width of one item in this stream (e.g. 4 for positions/staged
    /// i32 field values, `key_bytes` for the allele-key stream).
    item_width: usize,
    chunk_files: &'a [File],
    dest: &'a File,
}

/// Phase B (shared): adaptive-tile, parallel pread→interleave→pwrite gather.
///
/// Each rayon worker owns one tile (a contiguous run of columns). For every
/// payload it allocates one `Vec<u8>` sized `tile_items * item_width`, reads
/// each chunk's contributing slice via positional reads, scatters bytes
/// column-major into the tile buffer via per-column write heads (tracked in
/// items, applied in bytes), then `pwrite`s the assembled tile to its
/// pre-computed byte range in the payload's destination file.
///
/// Per-column write heads depend only on `final_offsets`/`ram_ledger` (not on
/// any payload's data), so they are identical across payloads for the same
/// tile — computed once per tile and reused (cloned) for each payload.
fn gather_columns(
    total_columns: usize,
    num_chunks: usize,
    ram_ledger: &[Vec<u32>],
    final_offsets: &[u64],
    chunk_offsets: &[Vec<u32>],
    payloads: &[Payload],
) -> Result<(), ConversionError> {
    let total_items: u64 = final_offsets[total_columns];

    // Adaptive tile size: target TILE_RAM_BUDGET per tile. Payloads are
    // gathered sequentially below (one tile_buffer live at a time), but we
    // size against the sum of all payload item widths (e.g. pos + key for
    // merge_mini_sc) as a conservative bound that keeps peak tile RAM under
    // budget even if the gather were made concurrent.
    let bytes_per_item: u64 = payloads.iter().map(|p| p.item_width as u64).sum();
    let avg_calls_per_col =
        std::cmp::max(1u64, total_items / std::cmp::max(1, total_columns) as u64);
    let columns_per_tile = std::cmp::max(
        1usize,
        std::cmp::min(
            total_columns.max(1),
            (TILE_RAM_BUDGET_BYTES / (avg_calls_per_col * bytes_per_item)) as usize,
        ),
    );

    // Tile start columns — independent work units, parallelized across rayon.
    let tile_starts: Vec<usize> = (0..total_columns).step_by(columns_per_tile).collect();

    tile_starts
        .par_iter()
        .try_for_each(|&tile_start_col| -> Result<(), ConversionError> {
            let tile_end_col = std::cmp::min(tile_start_col + columns_per_tile, total_columns);
            let tile_n_cols = tile_end_col - tile_start_col;
            let tile_start_item = final_offsets[tile_start_col] as usize;
            let tile_end_item = final_offsets[tile_end_col] as usize;
            let tile_total_items = tile_end_item - tile_start_item;

            if tile_total_items == 0 {
                return Ok(());
            }

            // per-column write head (offset within this tile buffer, in items).
            // Identical for every payload — computed once, cloned per payload below.
            let mut tile_write_heads_base = vec![0usize; tile_n_cols];
            #[allow(clippy::needless_range_loop)]
            for i in 0..tile_n_cols {
                let col = tile_start_col + i;
                tile_write_heads_base[i] = (final_offsets[col] as usize) - tile_start_item;
            }

            for payload in payloads {
                let item_width = payload.item_width;
                let mut tile_buffer = vec![0u8; tile_total_items * item_width];
                let mut tile_write_heads = tile_write_heads_base.clone();

                // gather from chunks
                for chunk_id in 0..num_chunks {
                    let chunk_start_item = chunk_offsets[chunk_id][tile_start_col] as usize;
                    let chunk_end_item = chunk_offsets[chunk_id][tile_end_col] as usize;
                    let chunk_items_to_read = chunk_end_item - chunk_start_item;

                    if chunk_items_to_read == 0 {
                        continue;
                    }

                    // Stateless positional read — multiple workers can read the
                    // same File concurrently without locking or seek contention.
                    let mut chunk_bytes = vec![0u8; chunk_items_to_read * item_width];
                    let byte_offset = (chunk_start_item * item_width) as u64;
                    payload.chunk_files[chunk_id]
                        .read_exact_at(&mut chunk_bytes, byte_offset)
                        .map_err(|e| ConversionError::Io {
                            context: "pread chunk payload".into(),
                            source: e,
                        })?;

                    // stitch this chunk's block into the main Tile buffer
                    let mut local_chunk_cursor = 0usize;
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..tile_n_cols {
                        let col = tile_start_col + i;
                        let calls = ram_ledger[chunk_id][col] as usize;
                        if calls == 0 {
                            continue;
                        }

                        let dest_start = tile_write_heads[i] * item_width;
                        let src_start = local_chunk_cursor * item_width;
                        tile_buffer[dest_start..dest_start + calls * item_width].copy_from_slice(
                            &chunk_bytes[src_start..src_start + calls * item_width],
                        );

                        tile_write_heads[i] += calls;
                        local_chunk_cursor += calls;
                    }
                }

                // pwrite the assembled tile to its known byte range in the
                // destination file. Tiles are disjoint by construction
                // (final_offsets is monotonically increasing), so concurrent
                // write_all_at calls touch non-overlapping regions.
                let tile_byte_offset = (tile_start_item * item_width) as u64;
                payload
                    .dest
                    .write_all_at(&tile_buffer, tile_byte_offset)
                    .map_err(|e| ConversionError::Io {
                        context: "pwrite payload".into(),
                        source: e,
                    })?;
            }
            Ok(())
        })
}

/// Performs the Tile-Based Interleaving Merge.
///
/// Phase A: in-memory metadata pass — derives global per-column offsets and per-chunk
///          local offsets from the RAM Ledger, writes `offsets.npy`.
/// Phase B: parallel tile gather — each rayon worker owns one tile, reads the slice
///          of every chunk via positional reads, scatters into per-column slots,
///          then `pwrite`s the assembled tile to its pre-computed byte range in
///          `positions.bin` / `alleles.bin`.
/// Phase C: cleanup of per-chunk temp files.
pub fn merge_mini_sc(
    key_bytes: usize,
    num_chunks: usize,
    num_samples: usize,
    ploidy: usize,
    output_dir: &str,
    ram_ledger: Vec<Vec<u32>>,
) -> Result<(), ConversionError> {
    let output_dir_path = Path::new(output_dir);
    let total_columns = num_samples * ploidy;
    let pos_size = std::mem::size_of::<u32>(); // positions are always u32

    tracing::debug!("Phase A -> Executing In-Memory Metadata Pass");

    // pre-compute global offsets and local chunk offsets using the RAM Ledger
    let (final_offsets, chunk_offsets) = derive_offsets(num_chunks, total_columns, &ram_ledger);

    // save the global offsets array immediately
    let offsets_array = Array1::from_vec(final_offsets.clone());
    write_npy(layout::offsets(output_dir_path), &offsets_array).map_err(|source| {
        ConversionError::Npy {
            path: layout::offsets(output_dir_path)
                .to_string_lossy()
                .into_owned(),
            source,
        }
    })?;

    let total_items: u64 = final_offsets[total_columns];
    let pos_total_bytes: u64 = total_items * pos_size as u64;
    let key_total_bytes: u64 = total_items * key_bytes as u64;

    tracing::debug!("Phase B -> Executing Parallel Tile-Based Interleaving Gather");

    // Pre-create the monolithic outputs at full size so worker pwrites land in
    // disjoint byte ranges. set_len doesn't allocate disk space (sparse file)
    // until each tile actually writes.
    let final_pos_file =
        File::create(layout::positions(output_dir_path)).map_err(|e| ConversionError::Io {
            context: "creating positions.bin".to_string(),
            source: e,
        })?;
    final_pos_file
        .set_len(pos_total_bytes)
        .map_err(|e| ConversionError::Io {
            context: "sizing positions.bin".to_string(),
            source: e,
        })?;
    let final_key_file =
        File::create(layout::alleles(output_dir_path)).map_err(|e| ConversionError::Io {
            context: "creating alleles.bin".to_string(),
            source: e,
        })?;
    final_key_file
        .set_len(key_total_bytes)
        .map_err(|e| ConversionError::Io {
            context: "sizing alleles.bin".to_string(),
            source: e,
        })?;

    // Open every chunk's pos/key file exactly once; pread() is stateless and
    // safe to call concurrently from multiple rayon workers. Split into two
    // parallel Vec<File> so each stream becomes its own gather_columns Payload.
    let pos_chunk_files: Vec<File> = (0..num_chunks)
        .map(|c| -> Result<File, ConversionError> {
            File::open(layout::chunk_pos(output_dir_path, c)).map_err(|e| ConversionError::Io {
                context: format!("opening chunk_{c}_pos.bin"),
                source: e,
            })
        })
        .collect::<Result<_, _>>()?;
    let key_chunk_files: Vec<File> = (0..num_chunks)
        .map(|c| -> Result<File, ConversionError> {
            File::open(layout::chunk_key(output_dir_path, c)).map_err(|e| ConversionError::Io {
                context: format!("opening chunk_{c}_key.bin"),
                source: e,
            })
        })
        .collect::<Result<_, _>>()?;

    tracing::debug!(
        tile_mb = TILE_RAM_BUDGET_BYTES / (1024 * 1024),
        "Tile size target (adaptive; computed inside gather_columns)"
    );

    gather_columns(
        total_columns,
        num_chunks,
        &ram_ledger,
        &final_offsets,
        &chunk_offsets,
        &[
            Payload {
                item_width: pos_size,
                chunk_files: &pos_chunk_files,
                dest: &final_pos_file,
            },
            Payload {
                item_width: key_bytes,
                chunk_files: &key_chunk_files,
                dest: &final_key_file,
            },
        ],
    )?;

    // Drop file handles to flush metadata before cleanup
    drop(pos_chunk_files);
    drop(key_chunk_files);
    drop(final_pos_file);
    drop(final_key_file);

    tracing::debug!("Phase C -> Cleaning up temporary chunk files");
    for c in 0..num_chunks {
        let _ = std::fs::remove_file(layout::chunk_pos(output_dir_path, c));
        let _ = std::fs::remove_file(layout::chunk_key(output_dir_path, c));
    }

    tracing::debug!("Merge Complete.");
    Ok(())
}

/// Merge one var_key field's per-chunk `chunk_{c}_field{field_ix}.bin` files into
/// `dest_values_bin`, in the same column-major order as `alleles.bin`/`positions.bin`.
///
/// `item_width` is the staged per-value byte width (4 for the i32/f32 staged
/// representation Task 7 writes — narrowing to a final storage dtype happens later,
/// at finalize time, not here). `ram_ledger` is the SAME calls-per-(chunk, column)
/// ledger `merge_mini_sc` uses for the pos/key streams — field values are staged
/// 1:1 with calls, so the identical column-major reordering applies; only the
/// per-item width differs.
///
/// This calls the same `derive_offsets` (Phase A) + `gather_columns` (Phase B)
/// helpers `merge_mini_sc` uses, with a single `Payload` for the flat byte
/// buffer (no separate pos/key arrays, and no `offsets.npy` — the offsets
/// already written by `merge_mini_sc` for this stream apply unchanged to every
/// field, since fields share the same per-call ordering).
///
/// The caller is responsible for creating `dest_values_bin`'s parent directory
/// before calling this function (this function does not call `create_dir_all`).
///
/// On success, the per-chunk `chunk_{c}_field{field_ix}.bin` source files are
/// removed (Phase C cleanup), mirroring `merge_mini_sc`'s pos/key cleanup.
#[allow(clippy::too_many_arguments)]
pub fn merge_var_key_field_values(
    output_dir: &str,
    num_chunks: usize,
    num_samples: usize,
    ploidy: usize,
    ram_ledger: &[Vec<u32>],
    field_ix: usize,
    item_width: usize,
    dest_values_bin: &Path,
) -> Result<(), ConversionError> {
    let output_dir_path = Path::new(output_dir);
    let total_columns = num_samples * ploidy;

    // Phase A (shared): derive global per-column offsets + per-chunk local
    // offsets from the RAM Ledger. Identical schedule merge_mini_sc uses for
    // the pos/key streams — field values are staged 1:1 with calls, so the
    // same column-major reordering applies.
    let (final_offsets, chunk_offsets) = derive_offsets(num_chunks, total_columns, ram_ledger);

    let total_items: u64 = final_offsets[total_columns];
    let total_bytes: u64 = total_items * item_width as u64;

    // Pre-create the monolithic output at full size so worker pwrites land in
    // disjoint byte ranges (sparse file — no disk space consumed until written).
    let dest_file = File::create(dest_values_bin).map_err(|e| ConversionError::Io {
        context: format!("creating {:?}", dest_values_bin),
        source: e,
    })?;
    dest_file
        .set_len(total_bytes)
        .map_err(|e| ConversionError::Io {
            context: format!("sizing {:?}", dest_values_bin),
            source: e,
        })?;

    // Open every chunk's field file exactly once; pread() is stateless and safe
    // to call concurrently from multiple rayon workers.
    let chunk_files: Vec<File> = (0..num_chunks)
        .map(|c| -> Result<File, ConversionError> {
            File::open(layout::chunk_field(output_dir_path, c, field_ix)).map_err(|e| {
                ConversionError::Io {
                    context: format!("opening chunk_{c}_field{field_ix}.bin"),
                    source: e,
                }
            })
        })
        .collect::<Result<_, _>>()?;

    // Phase B (shared): parallel tile gather — a single byte-payload stream.
    gather_columns(
        total_columns,
        num_chunks,
        ram_ledger,
        &final_offsets,
        &chunk_offsets,
        &[Payload {
            item_width,
            chunk_files: &chunk_files,
            dest: &dest_file,
        }],
    )?;

    // Drop file handles to flush metadata before cleanup
    drop(chunk_files);
    drop(dest_file);

    for c in 0..num_chunks {
        let _ = std::fs::remove_file(layout::chunk_field(output_dir_path, c, field_ix));
    }

    Ok(())
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

    #[test]
    fn derive_offsets_matches_inline() {
        // 2 chunks, 3 columns
        let ledger = vec![vec![2u32, 0, 1], vec![1u32, 3, 0]];
        let (final_offsets, chunk_offsets) = derive_offsets(2, 3, &ledger);
        assert_eq!(final_offsets, vec![0, 3, 6, 7]); // col totals 3,3,1
        assert_eq!(chunk_offsets[0], vec![0, 2, 2, 3]);
        assert_eq!(chunk_offsets[1], vec![0, 1, 4, 4]);
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

        merge_mini_sc(4, 1, 2, 2, dir.to_str().unwrap(), ram_ledger).unwrap();

        let final_pos = read_u32_bin(&dir.join("positions.bin"));
        let final_key = read_u32_bin(&dir.join("alleles.bin"));
        let final_off = read_offsets_npy(&dir.join("offsets.npy"));

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

        merge_mini_sc(4, 2, 2, 1, dir.to_str().unwrap(), ram_ledger).unwrap();

        let final_pos = read_u32_bin(&dir.join("positions.bin"));
        let final_key = read_u32_bin(&dir.join("alleles.bin"));
        let final_off = read_offsets_npy(&dir.join("offsets.npy"));

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

        merge_mini_sc(4, 1, 2, 2, dir.to_str().unwrap(), ram_ledger).unwrap();

        let final_pos = read_u32_bin(&dir.join("positions.bin"));
        let final_off = read_offsets_npy(&dir.join("offsets.npy"));

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

        merge_mini_sc(4, 2, 2, 1, dir.to_str().unwrap(), ram_ledger).unwrap();

        let final_pos = read_u32_bin(&dir.join("positions.bin"));
        let final_off = read_offsets_npy(&dir.join("offsets.npy"));

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

        merge_mini_sc(1, 2, 2, 1, dir.to_str().unwrap(), ram_ledger).unwrap();

        let final_pos = read_u32_bin(&dir.join("positions.bin"));
        let final_key = read_u8_bin(&dir.join("alleles.bin"));
        let final_off = read_offsets_npy(&dir.join("offsets.npy"));

        assert_eq!(final_pos, vec![100, 200, 400, 300, 500, 600]);
        assert_eq!(final_key, vec![1, 2, 4, 3, 5, 6]);
        assert_eq!(final_off, vec![0u64, 3, 6]);
    }

    // Helper: stage one chunk's field values (raw i32 bytes) to disk in the
    // layout merge_var_key_field_values expects.
    fn write_chunk_field_file(dir: &Path, chunk_id: usize, field_ix: usize, values: &[i32]) {
        let mut f = File::create(layout::chunk_field(dir, chunk_id, field_ix)).unwrap();
        f.write_all(bytemuck::cast_slice(values)).unwrap();
    }

    fn read_i32_bin(path: &Path) -> Vec<i32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // Field values must interleave IDENTICALLY to the keys given the same
    // ledger — mirrors test_merge_multi_chunk_interleave's scenario exactly,
    // but for a single field's per-chunk staged i32 values instead of pos/key.
    #[test]
    fn test_merge_var_key_field_values_multi_chunk_interleave() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();

        // 2 samples x 1 ploidy = 2 columns. 2 chunks.
        // Chunk 0: col0=2 calls (10, 20), col1=1 call (30)  -> [10, 20, 30]
        // Chunk 1: col0=1 call (40),      col1=2 calls (50, 60) -> [40, 50, 60]
        // Expected final column-major order:
        //   col0: chunk0 (10, 20) + chunk1 (40)      -> 10, 20, 40
        //   col1: chunk0 (30)     + chunk1 (50, 60)  -> 30, 50, 60
        // -> [10, 20, 40, 30, 50, 60]
        let ram_ledger = vec![vec![2u32, 1], vec![1u32, 2]];
        write_chunk_field_file(dir, 0, 0, &[10, 20, 30]);
        write_chunk_field_file(dir, 1, 0, &[40, 50, 60]);

        let dest = dir.join("fields").join("DP").join("var_key_snp");
        std::fs::create_dir_all(&dest).unwrap();
        let dest_values_bin = dest.join("values.bin");

        merge_var_key_field_values(
            dir.to_str().unwrap(),
            2,
            2,
            1,
            &ram_ledger,
            0,
            4,
            &dest_values_bin,
        )
        .unwrap();

        let final_values = read_i32_bin(&dest_values_bin);
        assert_eq!(final_values, vec![10, 20, 40, 30, 50, 60]);
        assert_eq!(final_values.len() * 4, 6 * 4); // total_calls(6) * item_width(4)

        // Phase C cleanup: per-chunk field files must be gone.
        assert!(!layout::chunk_field(dir, 0, 0).exists());
        assert!(!layout::chunk_field(dir, 1, 0).exists());
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

            merge_mini_sc(4, num_chunks, num_samples, ploidy, dir.to_str().unwrap(), ram_ledger.clone()).unwrap();

            let final_pos = read_u32_bin(&dir.join("positions.bin"));
            let final_key = read_u32_bin(&dir.join("alleles.bin"));
            let final_off = read_offsets_npy(&dir.join("positions.bin").with_file_name("offsets.npy"));

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

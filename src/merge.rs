use crate::error::ConversionError;
use crate::layout;
use ndarray::Array1;
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Target peak RAM for one `gather_columns` call AS A WHOLE — not per worker.
/// Each worker holds one tile buffer at a time, so the per-worker share is this
/// divided by the thread budget, and peak stays flat as threads grow.
///
/// It was previously documented as a per-worker budget ("peak = TILE_RAM_BUDGET
/// × rayon_threads"), which only stayed survivable because the `par_iter` below
/// was silently running on a one-thread pool. Actually parallelising it at that
/// definition would have made peak scale with the thread budget — ~8.75 GB at
/// 35 threads.
const TILE_RAM_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Floor on a single worker's share of the budget. Dividing the whole-stage
/// budget by a wide thread count would otherwise shrink tiles until each
/// `pread` is too small to amortise, trading RAM for I/O efficiency. Worst-case
/// peak is therefore `MIN_TILE_RAM_BYTES * threads` (~280 MB at 35 threads)
/// rather than unbounded.
const MIN_TILE_RAM_BYTES: u64 = 8 * 1024 * 1024;

/// Width, in `u32`s, of one spilled-ledger row. See [`crate::layout::ledger`]
/// for the file's layout.
#[inline]
fn ledger_row_len(total_columns: usize) -> usize {
    total_columns + 1
}

/// Read chunk `chunk_id`'s ledger prefix offsets for columns
/// `[start_col, start_col + out.len() - 1]` into `out`.
///
/// One contiguous `pread`, which is the whole reason the ledger stores prefix
/// sums rather than raw counts: a tile needs each chunk's *local start offset*
/// for its first column, and a count-only row could only produce that by
/// summing from column zero.
fn read_ledger_slice(
    ledger: &File,
    total_columns: usize,
    chunk_id: usize,
    start_col: usize,
    out: &mut [u32],
) -> Result<(), ConversionError> {
    let row_len = ledger_row_len(total_columns);
    let byte_offset = ((chunk_id * row_len + start_col) * std::mem::size_of::<u32>()) as u64;
    ledger
        .read_exact_at(bytemuck::cast_slice_mut(out), byte_offset)
        .map_err(|e| ConversionError::Io {
            context: format!("pread ledger row {chunk_id} at column {start_col}"),
            source: e,
        })
}

/// Phase A (shared): derive global per-column offsets by streaming the spilled
/// ledger once. Both `merge_mini_sc` and `merge_var_key_field_values` need the
/// identical column-major schedule — field values are staged 1:1 with calls, so
/// the same reordering applies regardless of per-item byte width.
///
/// `final_offsets[col]` is the global item index where column `col` starts
/// (length `total_columns + 1`, monotonically increasing).
///
/// The per-chunk local offsets this used to return alongside it are *not*
/// materialized: they were a second `num_chunks x total_columns` array — ~19 GB
/// at 535,662 diploid samples over a chr12-sized contig, on top of the identical
/// amount the executor was already retaining (#183). Each tile now preads its
/// own slice of them from the ledger instead, so RAM here is `O(total_columns)`
/// regardless of chunk count.
fn derive_final_offsets(
    ledger: &File,
    num_chunks: usize,
    total_columns: usize,
) -> Result<Vec<u64>, ConversionError> {
    // Accumulate per-column totals in place, then prefix-sum them, so this is
    // the single `total_columns`-sized allocation of the whole pass.
    let mut final_offsets = vec![0u64; total_columns + 1];
    let mut row = vec![0u32; ledger_row_len(total_columns)];

    for chunk_id in 0..num_chunks {
        read_ledger_slice(ledger, total_columns, chunk_id, 0, &mut row)?;
        for col in 0..total_columns {
            final_offsets[col + 1] += (row[col + 1] - row[col]) as u64;
        }
    }
    for col in 0..total_columns {
        final_offsets[col + 1] += final_offsets[col];
    }

    Ok(final_offsets)
}

/// A stream's spilled ledger, opened once and scanned once.
///
/// Both merge entry points need the same two things from it -- the open file to
/// pread tile slices from, and the global per-column offsets -- and a stream is
/// merged once for its pos/key payloads plus once per field. Deriving the
/// offsets inside each call would re-read the whole ledger `n_fields + 1` times
/// (the previous in-RAM version rebuilt the equivalent array just as often, but
/// from memory, where a redundant pass is far cheaper).
///
/// Ownership of the file's lifetime sits with the caller: [`Self::remove`]
/// deletes it, and must be called only after every merge for the stream.
pub struct LedgerView {
    file: File,
    path: PathBuf,
    /// `final_offsets[col]` is the global item index where column `col` starts
    /// (length `total_columns + 1`, monotonically increasing).
    final_offsets: Vec<u64>,
}

impl LedgerView {
    /// Open `stream_dir`'s ledger and derive its global per-column offsets.
    pub fn open(
        stream_dir: &Path,
        num_chunks: usize,
        total_columns: usize,
    ) -> Result<Self, ConversionError> {
        let path = layout::ledger(stream_dir);
        let file = File::open(&path).map_err(|e| ConversionError::Io {
            context: format!("opening {}", path.display()),
            source: e,
        })?;
        let final_offsets = derive_final_offsets(&file, num_chunks, total_columns)?;
        Ok(Self {
            file,
            path,
            final_offsets,
        })
    }

    /// Total items across every column — the length of the merged stream.
    pub fn total_items(&self) -> u64 {
        self.final_offsets[self.final_offsets.len() - 1]
    }

    /// Delete the backing file. Call once the stream's every merge has run.
    pub fn remove(self) {
        drop(self.file);
        let _ = std::fs::remove_file(&self.path);
    }
}

/// How many columns one tile spans, given a thread budget.
///
/// `threads` workers each hold one tile buffer at a time, so a tile gets
/// `TILE_RAM_BUDGET_BYTES / threads` and peak stays ~flat as threads grow --
/// this is the whole point of taking `threads` here rather than sizing per
/// worker. `MIN_TILE_RAM_BYTES` floors the share so a wide budget cannot shrink
/// tiles into `pread`s too small to amortise; past that floor peak grows again,
/// but only at ~8 MB per thread.
///
/// Pure and separately tested: the RAM coupling this encodes is the part that
/// silently held only because the gather used to run single-threaded.
fn tile_columns(
    total_items: u64,
    total_columns: usize,
    bytes_per_item: u64,
    threads: usize,
) -> usize {
    let per_worker_budget = (TILE_RAM_BUDGET_BYTES / threads.max(1) as u64).max(MIN_TILE_RAM_BYTES);
    let avg_calls_per_col =
        std::cmp::max(1u64, total_items / std::cmp::max(1, total_columns) as u64);
    std::cmp::max(
        1usize,
        std::cmp::min(
            total_columns.max(1),
            (per_worker_budget / (avg_calls_per_col * bytes_per_item.max(1))) as usize,
        ),
    )
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
/// Each rayon worker owns one tile (a contiguous run of columns) and one
/// `Vec<u8>` per payload, sized `tile_items * item_width`. It walks the chunks
/// in order; for each it preads that chunk's slice of the ledger (the tile's
/// prefix offsets), then each payload's contributing bytes, and scatters them
/// column-major into that payload's tile buffer via per-column write heads
/// (tracked in items, applied in bytes). Assembled tiles are `pwrite`n to their
/// pre-computed byte ranges at the end.
///
/// Chunks are the outer loop and payloads the inner one so that each chunk's
/// ledger slice is read **once** and shared across payloads; reading it per
/// payload instead would multiply this stage's pread count by the payload
/// count. The cost is that every payload's tile buffer is live at once, which
/// is exactly what `bytes_per_item` below already budgets for.
///
/// Per-column write heads depend only on `final_offsets` and the ledger (not on
/// any payload's data), so they start identical across payloads for the same
/// tile — computed once per tile and cloned per payload.
fn gather_columns(
    total_columns: usize,
    num_chunks: usize,
    ledger: &File,
    final_offsets: &[u64],
    payloads: &[Payload],
    threads: usize,
) -> Result<(), ConversionError> {
    let total_items: u64 = final_offsets[total_columns];
    let threads = threads.max(1);

    // Every payload's tile buffer is live at once (see the note above on loop
    // order), so the budget must be against the SUM of the payload item widths
    // (e.g. pos + key for merge_mini_sc) rather than the widest single one.
    // This was already the sizing rule when payloads were gathered one at a
    // time -- it was conservative then and is exact now.
    let bytes_per_item: u64 = payloads.iter().map(|p| p.item_width as u64).sum();
    let columns_per_tile = tile_columns(total_items, total_columns, bytes_per_item, threads);

    // Tile start columns — independent work units, parallelized across rayon.
    let tile_starts: Vec<usize> = (0..total_columns).step_by(columns_per_tile).collect();

    // Tile count is the ceiling on this gather's parallelism, and it is set by
    // the DATA (total sparse bytes / per-worker budget), not by `threads`. A
    // stream small enough to fit one tile cannot go faster no matter how wide
    // the budget — worth logging, because otherwise "no speedup" is
    // indistinguishable from "the pool never engaged".
    tracing::debug!(
        target: "genoray::monitor",
        tiles = tile_starts.len(),
        columns_per_tile,
        threads,
        payload_mb = (total_items * bytes_per_item) / (1024 * 1024),
        "gather tiling"
    );

    let gather_tile = |tile_start_col: usize| -> Result<(), ConversionError> {
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

        let mut tile_buffers: Vec<Vec<u8>> = payloads
            .iter()
            .map(|p| vec![0u8; tile_total_items * p.item_width])
            .collect();
        let mut tile_write_heads: Vec<Vec<usize>> = payloads
            .iter()
            .map(|_| tile_write_heads_base.clone())
            .collect();

        // This tile's slice of one chunk's ledger row: the prefix offsets for
        // columns [tile_start_col, tile_end_col]. `tile_n_cols + 1` u32s, so a
        // few KB even at cohort width, and the only per-chunk metadata a worker
        // ever holds.
        let mut ledger_slice = vec![0u32; tile_n_cols + 1];

        for chunk_id in 0..num_chunks {
            read_ledger_slice(
                ledger,
                total_columns,
                chunk_id,
                tile_start_col,
                &mut ledger_slice,
            )?;

            let chunk_start_item = ledger_slice[0] as usize;
            let chunk_items_to_read = ledger_slice[tile_n_cols] as usize - chunk_start_item;
            if chunk_items_to_read == 0 {
                continue;
            }

            for (p_ix, payload) in payloads.iter().enumerate() {
                let item_width = payload.item_width;

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
                let tile_buffer = &mut tile_buffers[p_ix];
                let heads = &mut tile_write_heads[p_ix];
                let mut local_chunk_cursor = 0usize;
                #[allow(clippy::needless_range_loop)]
                for i in 0..tile_n_cols {
                    let calls = (ledger_slice[i + 1] - ledger_slice[i]) as usize;
                    if calls == 0 {
                        continue;
                    }

                    let dest_start = heads[i] * item_width;
                    let src_start = local_chunk_cursor * item_width;
                    tile_buffer[dest_start..dest_start + calls * item_width]
                        .copy_from_slice(&chunk_bytes[src_start..src_start + calls * item_width]);

                    heads[i] += calls;
                    local_chunk_cursor += calls;
                }
            }
        }

        // pwrite each assembled tile to its known byte range in the destination
        // file. Tiles are disjoint by construction (final_offsets is
        // monotonically increasing), so concurrent write_all_at calls touch
        // non-overlapping regions.
        for (p_ix, payload) in payloads.iter().enumerate() {
            let tile_byte_offset = (tile_start_item * payload.item_width) as u64;
            payload
                .dest
                .write_all_at(&tile_buffers[p_ix], tile_byte_offset)
                .map_err(|e| ConversionError::Io {
                    context: "pwrite payload".into(),
                    source: e,
                })?;
        }
        Ok(())
    };

    // Skip the pool when it cannot pay for itself. Building one costs `threads`
    // OS thread spawns, and the gather is called several times per contig
    // (once per var_key stream, plus once per field), so on cohorts whose
    // variants are mostly DENSE — where the sparse streams hold few calls and
    // fit in a single tile — that spawn cost is pure loss. Measured at ~4-5 ms
    // per call against a 36 ms stage, i.e. a real regression if taken
    // unconditionally.
    if threads == 1 || tile_starts.len() <= 1 {
        return tile_starts.iter().try_for_each(|&c| gather_tile(c));
    }

    // Otherwise run on an OWN pool rather than the ambient one.
    // `process_chromosome` runs inside `lib.rs`'s dispatch pool, which is sized
    // to `concurrent_chroms` (1 by default), so a bare `par_iter()` reached from
    // here sees a ONE-thread pool and silently executes serially — the same trap
    // documented on `DenseMergeParams::threads`. Tiles differ in item count
    // (columns are not uniformly dense), so rayon's work-stealing is worth
    // keeping here over the static `std::thread::scope` split the dense
    // transpose uses.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.min(tile_starts.len()))
        .thread_name(|i| format!("merge-{i}"))
        .build()
        .expect("build merge gather pool");

    pool.install(|| tile_starts.par_iter().try_for_each(|&c| gather_tile(c)))
}

/// Performs the Tile-Based Interleaving Merge.
///
/// Phase A: metadata pass — streams the stream's spilled ledger
///          ([`crate::layout::ledger`]) to derive global per-column offsets,
///          writes `offsets.npy`.
/// Phase B: parallel tile gather — each rayon worker owns one tile, preads its
///          slice of every chunk's ledger row and payload, scatters into
///          per-column slots, then `pwrite`s the assembled tile to its
///          pre-computed byte range in `positions.bin` / `alleles.bin`.
/// Phase C: cleanup of per-chunk temp files.
///
/// The ledger arrives on disk rather than as an argument: held in RAM it was
/// `num_chunks x total_columns x 4 B` per stream, ~19 GB by the end of a
/// chr12-sized contig at 535,662 diploid samples, and Phase A doubled it (#183).
/// This does NOT delete it -- the caller owns that, via
/// [`LedgerView::remove`], because the stream's field merges read the same
/// ledger.
///
/// `threads` is the budget for the Phase B gather. Like
/// [`crate::dense_merge::DenseMergeParams::threads`] it is passed in rather than
/// taken from the ambient rayon pool: `process_chromosome` runs inside `lib.rs`'s
/// dispatch pool, sized to `concurrent_chroms`, so an ambient `par_iter()` here
/// would silently run on one thread.
#[allow(clippy::too_many_arguments)]
pub fn merge_mini_sc(
    key_bytes: usize,
    num_chunks: usize,
    num_samples: usize,
    ploidy: usize,
    output_dir: &str,
    threads: usize,
    ledger: &LedgerView,
) -> Result<(), ConversionError> {
    let output_dir_path = Path::new(output_dir);
    let total_columns = num_samples * ploidy;
    let pos_size = std::mem::size_of::<u32>(); // positions are always u32

    let final_offsets = &ledger.final_offsets;

    // save the global offsets array immediately
    let offsets_array = Array1::from_vec(final_offsets.to_vec());
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
        stage_budget_mb = TILE_RAM_BUDGET_BYTES / (1024 * 1024),
        threads,
        "Tile size target (adaptive; whole-stage budget split across workers)"
    );

    gather_columns(
        total_columns,
        num_chunks,
        &ledger.file,
        final_offsets,
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
        threads,
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
/// at finalize time, not here). This reads the SAME spilled ledger
/// ([`crate::layout::ledger`]) `merge_mini_sc` uses for the pos/key streams —
/// field values are staged 1:1 with calls, so the identical column-major
/// reordering applies; only the per-item width differs.
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
    ledger: &LedgerView,
    field_ix: usize,
    item_width: usize,
    dest_values_bin: &Path,
    threads: usize,
) -> Result<(), ConversionError> {
    let output_dir_path = Path::new(output_dir);
    let total_columns = num_samples * ploidy;

    // Phase A is already done: the caller derived this stream's global
    // per-column offsets once. Field values are staged 1:1 with calls, so the
    // schedule merge_mini_sc uses for the pos/key streams applies unchanged.
    let total_bytes: u64 = ledger.total_items() * item_width as u64;

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
        &ledger.file,
        &ledger.final_offsets,
        &[Payload {
            item_width,
            chunk_files: &chunk_files,
            dest: &dest_file,
        }],
        threads,
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

    /// Deliberately > 1 so every merge test drives the pooled gather path
    /// rather than the single-thread one that masked the ambient-pool bug.
    const TEST_THREADS: usize = 4;

    // ---- tile sizing / RAM coupling ----
    //
    // These fixtures are far too small to span more than one tile, so they
    // cannot exercise the RAM math end-to-end. `tile_columns` is pure, so the
    // coupling is asserted directly instead -- this is the invariant that used
    // to hold only by accident, because the gather never actually ran on more
    // than one thread.

    #[test]
    fn tile_budget_is_whole_stage_not_per_worker() {
        // 64M items x 4 bytes = 256 MB of payload across 1024 columns.
        let (items, cols, width) = (64u64 << 20, 1024usize, 4u64);

        // Peak = threads concurrently-live tiles. Under the OLD per-worker
        // reading this product grew linearly with `threads`; it must not.
        let peak = |threads: usize| -> u64 {
            let per_tile_cols = tile_columns(items, cols, width, threads) as u64;
            let items_per_tile = per_tile_cols * (items / cols as u64);
            items_per_tile * width * threads as u64
        };

        // Within the floor's reach, widening the pool must not raise peak RAM.
        assert!(peak(8) <= TILE_RAM_BUDGET_BYTES + (1 << 20));
        assert!(peak(16) <= TILE_RAM_BUDGET_BYTES + (1 << 20));
        // 32 threads x 8 MB floor = 256 MB, i.e. still at budget, not 32x it.
        assert!(
            peak(32) <= 8 * TILE_RAM_BUDGET_BYTES,
            "peak {} blew past the floor-bounded ceiling",
            peak(32)
        );
    }

    #[test]
    fn tile_columns_shrinks_with_threads_then_hits_the_floor() {
        let (items, cols, width) = (64u64 << 20, 1024usize, 4u64);
        let t1 = tile_columns(items, cols, width, 1);
        let t8 = tile_columns(items, cols, width, 8);
        assert!(
            t8 < t1,
            "more threads must mean smaller tiles ({t1} -> {t8})"
        );

        // Past the floor, the share stops shrinking -- otherwise a wide budget
        // would grind the gather down into tiny preads.
        let huge = tile_columns(items, cols, width, 4096);
        let huger = tile_columns(items, cols, width, 65536);
        assert_eq!(
            huge, huger,
            "tile size must bottom out at MIN_TILE_RAM_BYTES"
        );
        assert!(huge >= 1);
    }

    #[test]
    fn tile_columns_is_always_a_usable_span() {
        // Degenerate shapes must still yield a legal, non-zero tile width:
        // `step_by(0)` panics and a tile wider than the matrix is meaningless.
        for &(items, cols, width, threads) in &[
            (0u64, 0usize, 0u64, 0usize),
            (0, 4, 4, 1),
            (1, 1, 1, 1),
            (u32::MAX as u64, 3, 8, 64),
        ] {
            let t = tile_columns(items, cols, width, threads);
            assert!(
                t >= 1,
                "tile_columns returned 0 for {items}/{cols}/{width}/{threads}"
            );
            assert!(t <= cols.max(1));
        }
    }

    // Helper: stage one chunk's pos and key arrays to disk in the layout merge expects.
    fn write_chunk_files(dir: &Path, chunk_id: usize, pos: &[u32], key: &[u32]) {
        let mut pf = File::create(dir.join(format!("chunk_{}_pos.bin", chunk_id))).unwrap();
        pf.write_all(bytemuck::cast_slice(pos)).unwrap();
        let mut kf = File::create(dir.join(format!("chunk_{}_key.bin", chunk_id))).unwrap();
        kf.write_all(bytemuck::cast_slice(key)).unwrap();
    }

    /// Helper: spill a per-(chunk, column) call-count ledger to the file the
    /// merge reads, in the prefix-sum form the writer produces. Tests keep
    /// stating raw counts because that is what `sample_lengths` holds and what
    /// the expected interleavings are easiest to reason about.
    fn spill_ledger(dir: &Path, ram_ledger: &[Vec<u32>]) {
        let mut f = File::create(layout::ledger(dir)).unwrap();
        for row in ram_ledger {
            let mut prefixed = Vec::with_capacity(row.len() + 1);
            let mut acc = 0u32;
            prefixed.push(acc);
            for &calls in row {
                acc += calls;
                prefixed.push(acc);
            }
            f.write_all(bytemuck::cast_slice(&prefixed)).unwrap();
        }
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
    fn derive_final_offsets_matches_inline() {
        // 2 chunks, 3 columns
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        spill_ledger(dir, &[vec![2u32, 0, 1], vec![1u32, 3, 0]]);
        let ledger = File::open(layout::ledger(dir)).unwrap();

        let final_offsets = derive_final_offsets(&ledger, 2, 3).unwrap();
        assert_eq!(final_offsets, vec![0, 3, 6, 7]); // col totals 3,3,1
    }

    // The per-chunk local offsets `derive_offsets` used to materialize are now
    // read a tile at a time from the ledger. Same numbers, same meaning: pin
    // both the full rows and a mid-row tile slice, since the tile read is where
    // an offset error would actually bite.
    #[test]
    fn ledger_slices_carry_the_per_chunk_local_offsets() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        spill_ledger(dir, &[vec![2u32, 0, 1], vec![1u32, 3, 0]]);
        let ledger = File::open(layout::ledger(dir)).unwrap();

        let mut row = vec![0u32; ledger_row_len(3)];
        read_ledger_slice(&ledger, 3, 0, 0, &mut row).unwrap();
        assert_eq!(row, vec![0, 2, 2, 3]);
        read_ledger_slice(&ledger, 3, 1, 0, &mut row).unwrap();
        assert_eq!(row, vec![0, 1, 4, 4]);

        // A tile covering columns [1, 3) reads 3 prefix offsets starting at
        // column 1 -- chunk 1's calls there are 3 and 0.
        let mut tile = vec![0u32; 3];
        read_ledger_slice(&ledger, 3, 1, 1, &mut tile).unwrap();
        assert_eq!(tile, vec![1, 4, 4]);
    }

    // The ledger outlives every merge for its stream and the caller deletes it.
    // Pin that: merging must NOT remove it, and `remove` must.
    #[test]
    fn ledger_view_owns_the_files_lifetime() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        spill_ledger(dir, &[vec![1u32, 1]]);
        write_chunk_files(dir, 0, &[100, 200], &[10, 20]);

        let ledger = LedgerView::open(dir, 1, 2).unwrap();
        assert_eq!(ledger.total_items(), 2);
        merge_mini_sc(4, 1, 2, 1, dir.to_str().unwrap(), TEST_THREADS, &ledger).unwrap();
        assert!(layout::ledger(dir).exists());

        ledger.remove();
        assert!(!layout::ledger(dir).exists());
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

        spill_ledger(dir, &ram_ledger);
        let ledger = LedgerView::open(dir, 1, 2 * 2).unwrap();
        merge_mini_sc(4, 1, 2, 2, dir.to_str().unwrap(), TEST_THREADS, &ledger).unwrap();

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

        spill_ledger(dir, &ram_ledger);
        let ledger = LedgerView::open(dir, 2, 2).unwrap();
        merge_mini_sc(4, 2, 2, 1, dir.to_str().unwrap(), TEST_THREADS, &ledger).unwrap();

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

        spill_ledger(dir, &ram_ledger);
        let ledger = LedgerView::open(dir, 1, 2 * 2).unwrap();
        merge_mini_sc(4, 1, 2, 2, dir.to_str().unwrap(), TEST_THREADS, &ledger).unwrap();

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

        spill_ledger(dir, &ram_ledger);
        let ledger = LedgerView::open(dir, 2, 2).unwrap();
        merge_mini_sc(4, 2, 2, 1, dir.to_str().unwrap(), TEST_THREADS, &ledger).unwrap();

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

        spill_ledger(dir, &ram_ledger);
        let ledger = LedgerView::open(dir, 2, 2).unwrap();
        merge_mini_sc(1, 2, 2, 1, dir.to_str().unwrap(), TEST_THREADS, &ledger).unwrap();

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

        spill_ledger(dir, &ram_ledger);
        let ledger = LedgerView::open(dir, 2, 2).unwrap();
        merge_var_key_field_values(
            dir.to_str().unwrap(),
            2,
            2,
            1,
            &ledger,
            0,
            4,
            &dest_values_bin,
            TEST_THREADS,
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
        //
        // `threads` is swept so the property covers the serial path AND the
        // pooled one against the SAME independently-computed ground truth --
        // the gather must be thread-count invariant. Before `gather_columns`
        // took an explicit budget it always ran on one thread regardless, so
        // this dimension could not have failed.
        #[test]
        fn test_merge_interleave_property(
            num_chunks in 1usize..5,
            num_samples in 1usize..5,
            ploidy in 1usize..3,
            threads in 1usize..9,
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

            spill_ledger(dir, &ram_ledger);
            let ledger = LedgerView::open(dir, num_chunks, num_samples * ploidy).unwrap();
            merge_mini_sc(4, num_chunks, num_samples, ploidy, dir.to_str().unwrap(), threads, &ledger)
                .unwrap();

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

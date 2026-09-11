use crate::dense::{DENSE_REGISTRY, DenseMap};
use crate::error::ConversionError;
use crate::layout;
use crate::streams::StreamMap;
use crate::types::SparseChunk;
use crossbeam_channel::Receiver;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

// I/O Writer Thread. Writes each chunk's active sub-streams into their
// per-tag directory: per-chunk positions (u32), byte-erased keys, and the
// chunk's row of the stream's RAM Ledger. Also writes each chunk's dense
// per-class payload (pos/key/geno) for classes with dense variants in this
// chunk.
pub fn run_io_writer(
    rx_sparse: Receiver<SparseChunk>,
    dirs: StreamMap<PathBuf>,
    dense_dirs: DenseMap<PathBuf>,
) -> Result<(), ConversionError> {
    // One ledger file per stream, created up front so that a stream which never
    // sees a call still leaves a full-length file of zeros for the merge. Rows
    // are written by `pwrite` at `chunk_id * row_bytes`, so the writer does not
    // depend on chunks arriving in order.
    let mut ledger_files: StreamMap<Option<File>> = StreamMap::from_fn(|_| None);
    for (tag, slot) in ledger_files.iter_mut() {
        let path = layout::ledger(dirs.get(tag));
        *slot = Some(File::create(&path).map_err(|e| ConversionError::Io {
            context: format!("creating {}", path.display()),
            source: e,
        })?);
    }

    while let Ok(chunk) = rx_sparse.recv() {
        let id = chunk.chunk_id;

        // var_key per-call streams (unchanged)
        for (tag, sub) in chunk.streams.iter() {
            let dir = dirs.get(tag);
            write_bin(
                &layout::chunk_pos(dir, id),
                bytemuck::cast_slice(&sub.call_positions),
            )?;
            write_bin(&layout::chunk_key(dir, id), &sub.call_keys)?; // already bytes
            for (i, col) in sub.field_calls.iter().enumerate() {
                write_bin(&layout::chunk_field(dir, id, i), staged_bytes(col))?;
            }
            // The executor used to keep this row in RAM for the whole contig --
            // `2 streams x columns x 4 B` per chunk, which is ~19 GB by the end
            // of a chr12-sized contig at 535,662 diploid samples (#183). It is
            // written once here and read once by the merge, so disk is where it
            // belongs.
            write_ledger_row(
                ledger_files.get(tag).as_ref().expect("ledger file opened"),
                id,
                &sub.sample_lengths,
            )?;
        }

        // dense per-class matrix + table (only classes with dense variants)
        for spec in &DENSE_REGISTRY {
            let sub = chunk.dense.get(spec.class);
            if sub.n_dense_variants == 0 {
                continue;
            }
            let dir = dense_dirs.get(spec.class);
            write_bin(
                &layout::chunk_pos(dir, id),
                bytemuck::cast_slice(&sub.positions),
            )?;
            write_bin(&layout::chunk_key(dir, id), &sub.keys)?;
            write_bin(&layout::chunk_geno(dir, id), &sub.geno_bits)?;
            for (i, col) in sub.field_info.iter().enumerate() {
                write_bin(&layout::chunk_field_info(dir, id, i), staged_bytes(col))?;
            }
            for (i, col) in sub.field_format.iter().enumerate() {
                write_bin(&layout::chunk_field_format(dir, id, i), staged_bytes(col))?;
            }
        }
    }

    tracing::debug!("writer thread: all chunks committed");
    Ok(())
}

pub fn run_long_allele_writer(
    rx_long: Receiver<Vec<u8>>,
    out_path: &Path,
    chrom_label: &str,
) -> Result<(), ConversionError> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(out_path)
        .map_err(|e| ConversionError::Io {
            context: format!("creating {}", out_path.display()),
            source: e,
        })?;
    let mut disk_writer = BufWriter::with_capacity(1024 * 1024, file);
    while let Ok(buffer) = rx_long.recv() {
        disk_writer
            .write_all(&buffer)
            .map_err(|e| ConversionError::Io {
                context: format!("writing long alleles to {}", out_path.display()),
                source: e,
            })?;
    }
    disk_writer.flush().map_err(|e| ConversionError::Io {
        context: format!("flushing {}", out_path.display()),
        source: e,
    })?;
    tracing::debug!(chrom = %chrom_label, "long-allele writer: buffers committed");
    Ok(())
}

/// Write chunk `chunk_id`'s row of a stream's RAM Ledger.
///
/// The row is the running prefix sum of `sample_lengths`, length
/// `sample_lengths.len() + 1`: `row[0] == 0` and `row[col + 1] - row[col]` is
/// the chunk's call count for column `col`. Prefix sums rather than raw counts
/// because the merge reads column *tiles* -- a tile needs each chunk's local
/// start offset, which a count-only row could only give by summing from column
/// zero.
///
/// `u32` matches the `chunk_offsets` width this replaces: a chunk's call total
/// has always had to fit `u32`, since it also indexes that chunk's own
/// `chunk_{id}_pos.bin`.
///
/// Rows are fixed-width, so the row for chunk `id` lives at `id * row_bytes`
/// and this is a positional write -- chunk arrival order does not matter.
fn write_ledger_row(
    file: &File,
    chunk_id: usize,
    sample_lengths: &[u32],
) -> Result<(), ConversionError> {
    let mut row = Vec::with_capacity(sample_lengths.len() + 1);
    let mut acc = 0u32;
    row.push(acc);
    for &calls in sample_lengths {
        acc += calls;
        row.push(acc);
    }
    let offset = (chunk_id * row.len() * std::mem::size_of::<u32>()) as u64;
    file.write_all_at(bytemuck::cast_slice(&row), offset)
        .map_err(|e| ConversionError::Io {
            context: format!("writing ledger row for chunk {chunk_id}"),
            source: e,
        })
}

/// View a staged field column as raw bytes for a flat write, matching the
/// element type's native representation (`i32`/`f32`).
fn staged_bytes(col: &crate::types::StagedColumn) -> &[u8] {
    match col {
        crate::types::StagedColumn::Int(v) => bytemuck::cast_slice(v),
        crate::types::StagedColumn::Float(v) => bytemuck::cast_slice(v),
    }
}

fn write_bin(path: &Path, bytes: &[u8]) -> Result<(), ConversionError> {
    let f = File::create(path).map_err(|e| ConversionError::Io {
        context: format!("creating {}", path.display()),
        source: e,
    })?;
    let mut f = BufWriter::new(f);
    f.write_all(bytes).map_err(|e| ConversionError::Io {
        context: format!("writing {}", path.display()),
        source: e,
    })?;
    f.flush().map_err(|e| ConversionError::Io {
        context: format!("flushing {}", path.display()),
        source: e,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::{DenseClass, DenseMap};
    use crate::enum_map::EnumKey;
    use crate::streams::{StreamMap, StreamTag};
    use crate::types::{DenseSubChunk, SparseChunk, SparseSubStream};
    use crossbeam_channel::bounded;
    use tempfile::tempdir;

    #[test]
    fn write_bin_returns_io_error_on_unwritable_path() {
        // A path whose parent dir does not exist cannot be created.
        let bad = std::path::Path::new("/nonexistent-sp3-dir/child/out.bin");
        let err = write_bin(bad, &[1u8, 2, 3]).unwrap_err();
        match err {
            crate::error::ConversionError::Io { context, .. } => {
                assert!(context.contains("out.bin"));
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn test_writer_persists_dense_chunk_files() {
        let tmp = tempdir().unwrap();
        // dense/snp dir with 2 dense variants, np=2 → geno = ceil(2*2/8)=1 byte.
        let snp_dir = tmp.path().join("dense/snp");
        let indel_dir = tmp.path().join("dense/indel");
        std::fs::create_dir_all(&snp_dir).unwrap();
        std::fs::create_dir_all(&indel_dir).unwrap();

        // var_key dirs (empty streams, still iterated by the writer)
        let vk_snp = tmp.path().join("var_key/snp");
        let vk_indel = tmp.path().join("var_key/indel");
        std::fs::create_dir_all(&vk_snp).unwrap();
        std::fs::create_dir_all(&vk_indel).unwrap();

        let mut dense = DenseMap::from_fn(|c| DenseSubChunk::empty(c.key_bytes()));
        let snp = dense.get_mut(DenseClass::Snp);
        snp.n_dense_variants = 2;
        snp.positions = vec![100, 200];
        snp.keys = vec![1u8, 2u8]; // 2 raw codes
        snp.geno_bits = vec![0b0000_1011u8]; // arbitrary

        let streams = StreamMap::from_fn(|tag| {
            let kb = crate::streams::REGISTRY[tag.index()].key_bytes;
            SparseSubStream::with_capacity(kb, 0, 0)
        });
        let _ = StreamTag::VarKeySnp; // keep import used

        let chunk = SparseChunk {
            chunk_id: 0,
            streams,
            dense,
        };

        let (tx, rx) = bounded(1);
        tx.send(chunk).unwrap();
        drop(tx);

        let dirs = StreamMap::from_fn(|tag| match tag {
            StreamTag::VarKeySnp => vk_snp.clone(),
            StreamTag::VarKeyIndel => vk_indel.clone(),
        });
        let dense_dirs = DenseMap::from_fn(|c| match c {
            DenseClass::Snp => snp_dir.clone(),
            DenseClass::Indel => indel_dir.clone(),
        });

        run_io_writer(rx, dirs, dense_dirs).unwrap();

        // dense snp chunk files exist with the right bytes
        let pos = std::fs::read(snp_dir.join("chunk_0_pos.bin")).unwrap();
        assert_eq!(pos.len(), 2 * 4); // 2 u32 positions
        let geno = std::fs::read(snp_dir.join("chunk_0_geno.bin")).unwrap();
        assert_eq!(geno, vec![0b0000_1011u8]);
        // indel had 0 dense variants → no files written
        assert!(!indel_dir.join("chunk_0_geno.bin").exists());
    }

    // The ledger the merge reads is written here, one fixed-width row per chunk
    // at `chunk_id * row_bytes` (#183). Two properties matter and neither is
    // exercised by the merge's own tests, which spill their ledgers by hand:
    // rows are prefix sums of `sample_lengths`, and a row lands at its chunk's
    // slot regardless of arrival order.
    #[test]
    fn test_writer_spills_ledger_rows_by_chunk_id() {
        let tmp = tempdir().unwrap();
        let vk_snp = tmp.path().join("var_key/snp");
        let vk_indel = tmp.path().join("var_key/indel");
        let snp_dir = tmp.path().join("dense/snp");
        let indel_dir = tmp.path().join("dense/indel");
        for d in [&vk_snp, &vk_indel, &snp_dir, &indel_dir] {
            std::fs::create_dir_all(d).unwrap();
        }

        // 2 columns. Chunk 0 has calls [2, 1] on the snp stream, chunk 1 [0, 3].
        let chunk_of = |id: usize, snp_lengths: Vec<u32>| {
            let mut streams = StreamMap::from_fn(|tag| {
                let kb = crate::streams::REGISTRY[tag.index()].key_bytes;
                SparseSubStream::with_capacity(kb, 0, 0)
            });
            let snp = streams.get_mut(StreamTag::VarKeySnp);
            let total: u32 = snp_lengths.iter().sum();
            snp.call_positions = (0..total).collect();
            snp.call_keys = vec![0u8; total as usize];
            snp.sample_lengths = snp_lengths;
            // The indel stream still contributes a row of zeros -- every stream
            // must have `num_chunks` rows or the merge's row offsets shift.
            streams.get_mut(StreamTag::VarKeyIndel).sample_lengths = vec![0, 0];
            SparseChunk {
                chunk_id: id,
                streams,
                dense: DenseMap::from_fn(|c| DenseSubChunk::empty(c.key_bytes())),
            }
        };

        // Send chunk 1 FIRST: the writer must place it by id, not by arrival.
        let (tx, rx) = bounded(2);
        tx.send(chunk_of(1, vec![0, 3])).unwrap();
        tx.send(chunk_of(0, vec![2, 1])).unwrap();
        drop(tx);

        let dirs = StreamMap::from_fn(|tag| match tag {
            StreamTag::VarKeySnp => vk_snp.clone(),
            StreamTag::VarKeyIndel => vk_indel.clone(),
        });
        let dense_dirs = DenseMap::from_fn(|c| match c {
            DenseClass::Snp => snp_dir.clone(),
            DenseClass::Indel => indel_dir.clone(),
        });
        run_io_writer(rx, dirs, dense_dirs).unwrap();

        let read_u32 = |p: &std::path::Path| -> Vec<u32> {
            std::fs::read(p)
                .unwrap()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        // 2 chunks x (2 columns + 1) prefix offsets, chunk 0's row first.
        let snp_rows = read_u32(&layout::ledger(&vk_snp));
        assert_eq!(snp_rows, vec![0, 2, 3, /* chunk 1: */ 0, 0, 3]);

        // A stream with no calls anywhere still gets a full-length file of
        // zeros, so the merge can address row `c` without a special case.
        let indel_rows = read_u32(&layout::ledger(&vk_indel));
        assert_eq!(indel_rows, vec![0u32; 6]);
    }

    #[test]
    fn test_writer_persists_field_chunk_files() {
        let tmp = tempdir().unwrap();

        let vk_snp = tmp.path().join("var_key/snp");
        let vk_indel = tmp.path().join("var_key/indel");
        std::fs::create_dir_all(&vk_snp).unwrap();
        std::fs::create_dir_all(&vk_indel).unwrap();

        let snp_dir = tmp.path().join("dense/snp");
        let indel_dir = tmp.path().join("dense/indel");
        std::fs::create_dir_all(&snp_dir).unwrap();
        std::fs::create_dir_all(&indel_dir).unwrap();

        // var_key/snp stream: 3 calls, one staged Float field.
        let mut streams = StreamMap::from_fn(|tag| {
            let kb = crate::streams::REGISTRY[tag.index()].key_bytes;
            SparseSubStream::with_capacity(kb, 0, 0)
        });
        let snp_stream = streams.get_mut(StreamTag::VarKeySnp);
        snp_stream.call_positions = vec![10, 20, 30];
        snp_stream.call_keys = vec![1u8, 2u8, 3u8];
        snp_stream.field_calls = vec![crate::types::StagedColumn::Float(vec![0.5, 1.5, 2.5])];

        // dense/snp class: 2 dense variants, one INFO field + one FORMAT field.
        let mut dense = DenseMap::from_fn(|c| DenseSubChunk::empty(c.key_bytes()));
        let snp_dense = dense.get_mut(DenseClass::Snp);
        snp_dense.n_dense_variants = 2;
        snp_dense.positions = vec![100, 200];
        snp_dense.keys = vec![1u8, 2u8];
        snp_dense.geno_bits = vec![0u8];
        snp_dense.field_info = vec![crate::types::StagedColumn::Int(vec![7, 8])];
        snp_dense.field_format = vec![crate::types::StagedColumn::Float(vec![1.0, 2.0, 3.0, 4.0])];

        let chunk = SparseChunk {
            chunk_id: 0,
            streams,
            dense,
        };

        let (tx, rx) = bounded(1);
        tx.send(chunk).unwrap();
        drop(tx);

        let dirs = StreamMap::from_fn(|tag| match tag {
            StreamTag::VarKeySnp => vk_snp.clone(),
            StreamTag::VarKeyIndel => vk_indel.clone(),
        });
        let dense_dirs = DenseMap::from_fn(|c| match c {
            DenseClass::Snp => snp_dir.clone(),
            DenseClass::Indel => indel_dir.clone(),
        });

        run_io_writer(rx, dirs, dense_dirs).unwrap();

        // var_key/snp field0: 3 f32 = 12 bytes.
        let field0 = std::fs::read(vk_snp.join("chunk_0_field0.bin")).unwrap();
        assert_eq!(field0.len(), 12);

        // dense/snp finfo0: 2 i32 = 8 bytes.
        let finfo0 = std::fs::read(snp_dir.join("chunk_0_finfo0.bin")).unwrap();
        assert_eq!(finfo0.len(), 8);

        // dense/snp fformat0: 4 f32 = 16 bytes.
        let fformat0 = std::fs::read(snp_dir.join("chunk_0_fformat0.bin")).unwrap();
        assert_eq!(fformat0.len(), 16);
    }
}

use crate::dense::{DENSE_REGISTRY, DenseMap};
use crate::error::ConversionError;
use crate::layout;
use crate::streams::StreamMap;
use crate::types::SparseChunk;
use crossbeam_channel::Receiver;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

// I/O Writer Thread. Writes each chunk's active sub-streams into their
// per-tag directory: per-chunk positions (u32) and byte-erased keys. Also
// writes each chunk's dense per-class payload (pos/key/geno) for classes
// with dense variants in this chunk.
pub fn run_io_writer(
    rx_sparse: Receiver<SparseChunk>,
    dirs: StreamMap<PathBuf>,
    dense_dirs: DenseMap<PathBuf>,
) -> Result<(), ConversionError> {
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

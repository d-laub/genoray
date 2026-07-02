use crate::dense::{DENSE_REGISTRY, DenseMap};
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
) {
    while let Ok(chunk) = rx_sparse.recv() {
        let id = chunk.chunk_id;

        // var_key per-call streams (unchanged)
        for (tag, sub) in chunk.streams.iter() {
            let dir = dirs.get(tag);
            write_bin(
                &layout::chunk_pos(dir, id),
                bytemuck::cast_slice(&sub.call_positions),
            );
            write_bin(&layout::chunk_key(dir, id), &sub.call_keys); // already bytes
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
            );
            write_bin(&layout::chunk_key(dir, id), &sub.keys);
            write_bin(&layout::chunk_geno(dir, id), &sub.geno_bits);
        }
    }

    println!("Writer Thread: Channel closed. All chunks safely committed to SSD.");
}

pub fn run_long_allele_writer(rx_long: Receiver<Vec<u8>>, out_path: &Path, chrom_label: &str) {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(out_path)
        .unwrap();
    let mut disk_writer = BufWriter::with_capacity(1024 * 1024, file);
    while let Ok(buffer) = rx_long.recv() {
        disk_writer
            .write_all(&buffer)
            .expect("Failed to write long alleles");
    }
    disk_writer.flush().unwrap();
    println!(
        "[{}] Long Allele Writer: All buffer data safely committed.",
        chrom_label
    );
}

fn write_bin(path: &Path, bytes: &[u8]) {
    let mut f = BufWriter::new(
        File::create(path).unwrap_or_else(|e| panic!("create {}: {}", path.display(), e)),
    );
    f.write_all(bytes).expect("write chunk bytes");
    f.flush().expect("flush chunk bytes");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::{DenseClass, DenseMap};
    use crate::streams::{StreamMap, StreamTag};
    use crate::types::{DenseSubChunk, SparseChunk, SparseSubStream};
    use crossbeam_channel::bounded;
    use tempfile::tempdir;

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

        run_io_writer(rx, dirs, dense_dirs);

        // dense snp chunk files exist with the right bytes
        let pos = std::fs::read(snp_dir.join("chunk_0_pos.bin")).unwrap();
        assert_eq!(pos.len(), 2 * 4); // 2 u32 positions
        let geno = std::fs::read(snp_dir.join("chunk_0_geno.bin")).unwrap();
        assert_eq!(geno, vec![0b0000_1011u8]);
        // indel had 0 dense variants → no files written
        assert!(!indel_dir.join("chunk_0_geno.bin").exists());
    }
}

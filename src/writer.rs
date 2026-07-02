use crate::layout;
use crate::streams::StreamMap;
use crate::types::SparseChunk;
use crossbeam_channel::Receiver;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

// I/O Writer Thread. Writes each chunk's active sub-streams into their
// per-tag directory: per-chunk positions (u32) and byte-erased keys.
pub fn run_io_writer(rx_sparse: Receiver<SparseChunk>, dirs: StreamMap<PathBuf>) {
    while let Ok(chunk) = rx_sparse.recv() {
        let id = chunk.chunk_id;

        for (tag, sub) in chunk.streams.iter() {
            let dir = dirs.get(tag);
            write_bin(
                &layout::chunk_pos(dir, id),
                bytemuck::cast_slice(&sub.call_positions),
            );
            write_bin(&layout::chunk_key(dir, id), &sub.call_keys); // already bytes
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

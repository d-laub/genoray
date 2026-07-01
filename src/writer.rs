use crate::types::SparseChunk;
use crossbeam_channel::Receiver;
use std::fs::File;
use std::io::{BufWriter, Write};

// I/O Writer Thread. Writes each chunk's two sub-streams into `snp_dir` and
// `indel_dir`: per-chunk positions (u32) and keys (u8 for SNP, u32 for indel).
pub fn run_io_writer(rx_sparse: Receiver<SparseChunk>, snp_dir: &str, indel_dir: &str) {
    while let Ok(chunk) = rx_sparse.recv() {
        let id = chunk.chunk_id;

        write_bin(
            &format!("{}/chunk_{}_pos.bin", snp_dir, id),
            bytemuck::cast_slice(&chunk.snp.call_positions),
        );
        write_bin(
            &format!("{}/chunk_{}_key.bin", snp_dir, id),
            bytemuck::cast_slice(&chunk.snp.call_keys), // u8 → identity cast
        );
        write_bin(
            &format!("{}/chunk_{}_pos.bin", indel_dir, id),
            bytemuck::cast_slice(&chunk.indel.call_positions),
        );
        write_bin(
            &format!("{}/chunk_{}_key.bin", indel_dir, id),
            bytemuck::cast_slice(&chunk.indel.call_keys),
        );
    }

    println!("Writer Thread: Channel closed. All chunks safely committed to SSD.");
}

fn write_bin(path: &str, bytes: &[u8]) {
    let mut f =
        BufWriter::new(File::create(path).unwrap_or_else(|e| panic!("create {}: {}", path, e)));
    f.write_all(bytes).expect("write chunk bytes");
    f.flush().expect("flush chunk bytes");
}

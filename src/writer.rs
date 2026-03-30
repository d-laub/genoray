use std::fs::File;
use std::io::{BufWriter, Write};
use bytemuck;
use crossbeam_channel::Receiver;
use crate::types::SparseChunk;

// I/O Writer Thread.
pub fn run_io_writer(rx_sparse: Receiver<SparseChunk>, output_dir: &str) {
    
    // thread goes to sleep here, waking up instantly 
    // whenever the Executor pushes a chunk into the channel.
    while let Ok(chunk) = rx_sparse.recv() {
        let chunk_id = chunk.chunk_id;

        // paths for the binary payloads
        let pos_path = format!("{}/chunk_{}_pos.bin", output_dir, chunk_id);
        let key_path = format!("{}/chunk_{}_key.bin", output_dir, chunk_id);

        // buffered writers
        let mut pos_file = BufWriter::new(File::create(&pos_path).expect("Failed to create pos.bin"));
        let mut key_file = BufWriter::new(File::create(&key_path).expect("Failed to create key.bin"));

        // zero-copy cast using bytemuck
        let pos_bytes: &[u8] = bytemuck::cast_slice(&chunk.call_positions);
        let key_bytes: &[u8] = bytemuck::cast_slice(&chunk.call_keys);

        // fast sequential OS writes
        pos_file.write_all(pos_bytes).expect("Failed to write chunk positions");
        key_file.write_all(key_bytes).expect("Failed to write chunk keys");

        // flush physical SSD buffers
        pos_file.flush().unwrap();
        key_file.flush().unwrap();
        
        // chunk is automatically dropped here, freeing the RAM block 
    }

    println!("Writer Thread: Channel closed. All chunks safely committed to SSD.");
}
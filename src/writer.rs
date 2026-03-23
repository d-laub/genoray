use std::fs::File;
use std::io::{BufWriter, Write};
use ndarray::Array1;
use ndarray_npy::write_npy;
use bytemuck;
use crate::types::SparseChunk;


pub fn write_sparse_chunk(chunk: SparseChunk, output_dir: &str) {
    let chunk_id = chunk.chunk_id;

    // file paths for this specific chunk
    let pos_path = format!("{}/chunk_{}_pos.bin", output_dir, chunk_id);
    let key_path = format!("{}/chunk_{}_key.bin", output_dir, chunk_id);
    let len_path = format!("{}/chunk_{}_lengths.npy", output_dir, chunk_id);

    // open buffered writers
    // BufWriter prevents the OS from making a system call for every single byte
    let mut pos_file = BufWriter::new(File::create(&pos_path).expect("Failed to create pos.bin"));
    let mut key_file = BufWriter::new(File::create(&key_path).expect("Failed to create key.bin"));

    // // cast (Vec<u32> -> &[u8])
    // let pos_bytes = unsafe {
    //     std::slice::from_raw_parts(
    //         chunk.call_positions.as_ptr() as *const u8,
    //         chunk.call_positions.len() * 4, // 4 bytes per u32
    //     )
    // };

    // let key_bytes = unsafe {
    //     std::slice::from_raw_parts(
    //         chunk.call_keys.as_ptr() as *const u8,
    //         chunk.call_keys.len() * 4, // 4 bytes per u32
    //     )
    // };

    // safe cast using bytemuck
    // this safely views the Vec<u32> as a raw &[u8] byte slice without copying.
    let pos_bytes: &[u8] = bytemuck::cast_slice(&chunk.call_positions);
    let key_bytes: &[u8] = bytemuck::cast_slice(&chunk.call_keys);

    // write the raw bytes directly to disk
    pos_file.write_all(pos_bytes).expect("Failed to write chunk positions");
    key_file.write_all(key_bytes).expect("Failed to write chunk keys");

    // flush the writers to ensure the OS moves the data from RAM to physical storage
    pos_file.flush().unwrap();
    key_file.flush().unwrap();

    // saving the lengths array as a NumPy .npy file
    // this allows the K-Way merge to load it instantly later
    let lengths_array = Array1::from_vec(chunk.sample_lengths);
    write_npy(&len_path, &lengths_array).expect("Failed to write chunk lengths.npy");
}
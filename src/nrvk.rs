use crossbeam_channel::Sender;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// strictly bounded memory bank for long alleles (non-blocking, double-buffered memory)
// Public API
pub struct LongAlleleTableWriter {
    tx_long: Sender<Vec<u8>>, //channel to the writer thread
    alt_offsets: Vec<u64>, // using 64 bits as same is used in seek - no extra casting
    buffer: Vec<u8>, // staging area
    global_offset: usize,     // The exact byte position
    row_index: u32, // Tracks the number of long alleles and 31-bit integer returned to the variant key
}

impl LongAlleleTableWriter {
    // because we need to map the file, it must be pre-allocated to the exact 
    // byte size calculated during your 1st Pass.
    pub fn new(tx_long: Sender<Vec<u8>>, buffer_capacity: usize) -> Self {
        // 128 MB RAM Limit to prevent OOM crashes -> can be changed or made run time
        // let buffer_capacity = 128 * 1024 * 1024; -> now parameter

        let mut initial_offsets = Vec::with_capacity(1_000_000); //hardcoded capacity for now, can optimize
        initial_offsets.push(0u64); // The first allele starts at byte 0

        Self {
            buffer: Vec::with_capacity(buffer_capacity),
            tx_long,
            global_offset: 0,
            row_index: 0,
            alt_offsets: initial_offsets,
        }
    }

    // Pushes a long allele into the buffer and returns the 31-bit pointer (offset)
    // to be stored in Variant Key array.
    #[inline(always)]
    pub fn push_long_allele(&mut self, alt_bytes: &[u8]) -> u32 {

        // enforcing the 31-bit capacity constraint
        assert!(
            self.row_index <= 0x7FFFFFFF,
            "Exceeded 31-bit (4,294,967,295) index capacity! Cannot proceed with this many long alleles!"
        );

        // checking Capacity Bounds
        // If this string pushes us over buffer cap, trigger the synchronous hardware flush
        if self.buffer.len() + alt_bytes.len() > self.buffer.capacity() {
            self.flush_buffer();
        }

        // store the current index to return later
        let current_index = self.row_index;
        self.buffer.extend_from_slice(alt_bytes); //pushing to ram

        // calculate where this string logically lives
        let next_byte_offset = (self.global_offset + self.buffer.len()) as u64;
        self.alt_offsets.push(next_byte_offset);

        self.row_index += 1;

        // return strictly the 31 LSBs (masking out the 32nd bit just in case)
        current_index & 0x7FFFFFFF
    }

    // Double-Buffer Swap 
    #[cold]
    fn flush_buffer(&mut self) {
        if self.buffer.is_empty() { return; }

        // empty vector
        let fresh_buffer = Vec::with_capacity(self.buffer.capacity());
        
        // swap full buffer with new
        let full_buffer = std::mem::replace(&mut self.buffer, fresh_buffer);

        // update the global byte tracker so offsets remain accurate.
        self.global_offset += full_buffer.len();

        // send the full buffer to the background writer thread.
        self.tx_long.send(full_buffer).expect("Long Allele Writer Thread panicked");
    }

    // called by the Executor at the very end to flush any remaining bytes
    pub fn finalize(mut self) -> Vec<u64> {
        if !self.buffer.is_empty() {
            // send the final partial buffer
            let final_buffer = std::mem::take(&mut self.buffer);
            self.tx_long.send(final_buffer).unwrap();
        }
        self.alt_offsets
    }
}

// Immutable state - Exists after conversion (phase 1)
pub struct LongAlleleReader {
    file: File,
    offsets: Vec<u64>,
}

impl LongAlleleReader {
    // TODO: Decide which will call this (or create this instance)
    pub fn new(output_dir: &str, chrom: &str) -> Self {
        let chrom_dir = format!("{}/{}", output_dir, chrom);
        
        let file_path = Path::new(&chrom_dir).join("long_alleles.bin");
        let file = File::open(file_path).expect("Failed to open long_alleles.bin");

        let offsets_path = Path::new(&chrom_dir).join("long_allele_offsets.npy");
        let offsets_array: ndarray::Array1<u64> = ndarray_npy::read_npy(offsets_path)
            .expect("Failed to load offsets npy");

        Self {
            file,
            offsets: offsets_array.into_raw_vec_and_offset().0,
        }
    }

    // Fetches the exact DNA string from the disk using the 31-bit row index.
    pub fn get_allele(&mut self, row_index: u32) -> Vec<u8> {
        let idx = row_index as usize;
        let start_byte = self.offsets[idx];
        let end_byte = self.offsets[idx + 1];
        let len = (end_byte - start_byte) as usize;

        let mut string_buffer = vec![0u8; len];
        self.file.seek(SeekFrom::Start(start_byte)).unwrap();
        self.file.read_exact(&mut string_buffer).unwrap();

        string_buffer
    }
}






// // Contains the NonReversibleLongAllele struct, MmapMut initialization, and long allele read/write methods.
// pub struct NonReversibleLongAllele {
//     pub ilens: Vec<i32>,
//     pub alt_offsets: Vec<u32>,
//     pub alt_allele: MmapMut, //data of all alts -> mem mapped disk file
//     pub curr_allele_idx: usize, // idx of where we are writing in the mem mapped file
// }
// //2 pass -> then allocate first (on disk -> mem maps) -> memmap

// impl NonReversibleLongAllele {
//     // Initialize the SoA with pre-allocated memory to prevent slow reallocations
//     pub fn new(capacity: usize) -> Self {
//         let mut offsets = Vec::with_capacity(capacity + 1);
//         offsets.push(0); // first offset is always 0

//         Self {
//             ilens: Vec::with_capacity(capacity),
//             alt_offsets: offsets,
//             alt_allele: Vec::with_capacity(capacity * 4),
//         }
//     }

//     /// Performs the 2-way fill -> inserts the data and returns the 32-bit packed pointer (31 bit index + 1 bit flag for non reversible key)
//     // pub fn push_variant(&mut self, ilen: i32, alt_allele: &[u8]) -> u32 {
//     //     // get the current row index
//     //     let row_index = self.ilens.len() as u32; 
//     //     // 0 <-> chr1:100 ( alt allele > 13 )
//     //     // 1 <-> chr22:5 ( alt allele > 13 )
//     //     assert!(
//     //         row_index <= 0x7FFFFFFF,
//     //         "Exceeded 31-bit index capacity! Cannot proceed with these many long alleles!"
//     //     );

//     //     // push alt to the alt allele vec
//     //     self.alt_allele.extend_from_slice(alt_allele);

//     //     // push the new end boundary to the offsets array
//     //     self.alt_offsets.push(self.alt_allele.len() as u32);

//     //     // add the ilen as well
//     //     self.ilens.push(ilen);

//     //     // packing the 31-bit index with the 1-bit lookup flag (non reversible flag)
//     //     (row_index << 1) | 1 // coupled here -> eg: unaware of vk
//     // }

//     // extracts the data using a 31-bit index pointer
//     pub fn get_variant(&self, row_index: usize) -> (i32, &[u8]) {
//         let start = self.alt_offsets[row_index] as usize;
//         let end = self.alt_offsets[row_index + 1] as usize;

//         (self.ilens[row_index], &self.alt_allele[start..end])
//     }

//     /// Performs the 2-way fill -> inserts the data and returns the 32-bit packed pointer (31 bit index + 1 bit flag for non reversible key)
//     pub fn push_variant(&mut self, ilen: i32, alt_allele: &[u8]) -> u32 {
//         // get the current row index
//         let row_index = self.ilens.len() as u32; 
//         // 0 <-> chr1:100 ( alt allele > 13 )
//         // 1 <-> chr22:5 ( alt allele > 13 )
//         assert!(
//             row_index <= 0x7FFFFFFF,
//             "Exceeded 31-bit index capacity! Cannot proceed with these many long alleles!"
//         );

//         // push alt to the alt allele vec
//         self.alt_allele.extend_from_slice(alt_allele);

//         // push the new end boundary to the offsets array
//         self.alt_offsets.push(self.alt_allele.len() as u32);

//         // add the ilen as well
//         self.ilens.push(ilen);

//         // packing the 31-bit index with the 1-bit lookup flag (non reversible flag)
//         (row_index << 1) | 1 // coupled here -> eg: unaware of vk
//     }

// }

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::bounded;

    #[test]
    fn test_global_u64_offsets() {
        let (tx, _rx) = bounded(10);
        let mut bank = LongAlleleTableWriter::new(tx, 1024);
        
        // Push 10 bytes, then 5 bytes
        bank.push_long_allele(b"0123456789"); 
        bank.push_long_allele(b"ABCDE");
        
        let offsets = bank.finalize();
        
        // First allele starts at 0, second at 10, EOF at 15.
        // MUST perfectly match OS SeekFrom requirements.
        assert_eq!(offsets, vec![0u64, 10u64, 15u64]); 
    }
}
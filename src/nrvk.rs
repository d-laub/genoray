use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Mutex;
use memmap2::MmapMut;
use ndarray::Array1;
use ndarray_npy::write_npy;

// A thread-safe, buffered Memory-Mapped struct for storing Long Alleles.
// Public API
pub struct LongAlleleTable {
    inner: Mutex<MmapWriteContext>,
    output_dir: String, // Need to store this to save the long allele table .npy file later
}

// Contains the mmap struct, MmapMut initialization, and long allele read/write methods.
// isolated inside mutex
struct MmapWriteContext {
    mmap: MmapMut, // virtual mem map -> change to vec for in mem
    alt_offsets: Vec<u32>, 
    buffer: Vec<u8>, // staging area
    global_offset: usize,     // The exact byte position in the memmap
    row_index: u32, // Tracks the number of long alleles
}

impl LongAlleleTable {
    // because we need to map the file, it must be pre-allocated to the exact 
    // byte size calculated during your 1st Pass.
    pub fn new(output_dir: &str, filename: &str, exact_byte_size: usize) -> Self {
        let file_path = Path::new(output_dir).join(filename);

        // open and stretch the file to the exact required size
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true) //wipes it if it already exists
            .open(&file_path)
            .expect("Failed to open long allele file");

        file.set_len(exact_byte_size as u64)
            .expect("Failed to stretch long allele file");

        // request the OS to map the file into virtual memory
        let mmap = unsafe {
            MmapMut::map_mut(&file).expect("Failed to map long allele arena into memory")
        };

        let mut initial_offsets = Vec::with_capacity(1_000_000); //hardcoded capacity for now, can optimize
        initial_offsets.push(0); // The first allele starts at byte 0

        Self {
            output_dir: output_dir.to_string(),
            inner: Mutex::new(MmapWriteContext {
                mmap,
                buffer: Vec::with_capacity(1024 * 1024), // 1MB buffer -> can be user defined
                global_offset: 0,
                row_index: 0,
                alt_offsets: initial_offsets,
            }),
        }
    }

    // Pushes a long allele into the mmap and returns the 32-bit pointer (offset)
    // to be stored in Variant Key array.
    pub fn push_long_allele(&self, alt_bytes: &[u8]) -> usize {
        // acquire the lock, and because this is per-chromosome, there is almost zero contention.
        let mut state = self.inner.lock().unwrap();

        // enforcing the 31-bit capacity constraint
        assert!(
            state.row_index <= 0x7FFFFFFF,
            "Exceeded 31-bit index capacity! Cannot proceed with this many long alleles!"
        );

        // Check if this new sequence pushes us over the buffer limit
        if state.buffer.len() + alt_bytes.len() > state.buffer.capacity() {
            
            // bulk-copy the buffer directly into the memory-mapped file
            let start = state.global_offset;
            let end = start + state.buffer.len();
            state.mmap[start..end].copy_from_slice(&state.buffer);
            
            // update the global disk pointer and clear the buffer
            state.global_offset = end;
            state.buffer.clear(); 
        }

        let current_index = state.row_index;
        
        //pPush the new bytes into the buffer
        state.buffer.extend_from_slice(alt_bytes);

        let next_byte_offset = state.global_offset + state.buffer.len();
        state.alt_offsets.push(next_byte_offset);

        // increment the tracker for the next allele
        state.row_index += 1;

        // the lock is automatically released here
        // return strictly the 31 LSBs (masking out the 32nd bit just in case)
        current_index & 0x7FFFFFFF
    }

    // called by the Main thread when the VCF is fully parsed to ensure 
    // any remaining bytes in the buffer are safely written to the memmap.
    pub fn final_flush(&self) {
        let mut state = self.inner.lock().unwrap();
        
        if !state.buffer.is_empty() {
            let start = state.global_offset;
            let end = start + state.buffer.len();
            state.mmap[start..end].copy_from_slice(&state.buffer);
            
            state.global_offset = end;
            state.buffer.clear();
        }
        
        // MmapMut safely flushes to physical disk automatically when it is dropped,
        // but calling state.mmap.flush() here to immediately flush it
        state.mmap.flush().expect("Failed to flush mmap to physical disk");
        let offsets_array = Array1::from_vec(state.alt_offsets.clone());
        let npy_path = format!("{}/long_allele_table_offsets.npy", self.output_dir);
        write_npy(&npy_path, &offsets_array).expect("Failed to write long allele offsets");
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

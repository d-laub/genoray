use crossbeam_channel::Sender;

// strictly bounded memory bank for long alleles (non-blocking, double-buffered memory)
// Public API
pub struct LongAlleleTable {
    tx_long: Sender<Vec<u8>>, //channel to the writer thread
    alt_offsets: Vec<u32>, // TODO - should this be u32 or u64
    buffer: Vec<u8>, // staging area
    global_offset: usize,     // The exact byte position
    row_index: u32, // Tracks the number of long alleles and 31-bit integer returned to the variant key
}

impl LongAlleleTable {
    // because we need to map the file, it must be pre-allocated to the exact 
    // byte size calculated during your 1st Pass.
    pub fn new(tx_long: Sender<Vec<u8>>) -> Self {
        // 128 MB RAM Limit to prevent OOM crashes -> can be changed or made run time
        let buffer_capacity = 128 * 1024 * 1024;

        let mut initial_offsets = Vec::with_capacity(1_000_000); //hardcoded capacity for now, can optimize
        initial_offsets.push(0); // The first allele starts at byte 0

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
        // checking Capacity Bounds
        // If this string pushes us over 128MB, trigger the synchronous hardware flush
        if self.buffer.len() + alt_bytes.len() > self.buffer.capacity() {
            self.flush_buffer();
        }

        // enforcing the 31-bit capacity constraint
        assert!(
            self.row_index <= 0x7FFFFFFF,
            "Exceeded 31-bit index capacity! Cannot proceed with this many long alleles!"
        );

        // store the current index to return later
        let current_index = self.row_index;

        self.buffer.extend_from_slice(alt_bytes); //pushing to ram

        // calculate where this string logically lives
        let next_byte_offset = (self.global_offset + self.buffer.len()) as u32;
        self.alt_offsets.push(next_byte_offset);

        self.row_index += 1;

        // return strictly the 31 LSBs (masking out the 32nd bit just in case)
        current_index & 0x7FFFFFFF
    }

    // Double-Buffer Swap 
    #[cold]
    fn flush_buffer(&mut self) {
        if self.buffer.is_empty() { return; }

        // empty 128MB vector.
        let fresh_buffer = Vec::with_capacity(128 * 1024 * 1024);
        
        // swap full buffer with new
        let full_buffer = std::mem::replace(&mut self.buffer, fresh_buffer);

        // send the full buffer to the background writer thread.
        self.tx_long.send(full_buffer).expect("Long Allele Writer Thread panicked");
        
        // update the global byte tracker so offsets remain accurate.
        self.global_offset += 128 * 1024 * 1024;
    }

    // called by the Executor at the very end to flush any remaining bytes
    pub fn finalize(mut self) -> Vec<u32> {
        if !self.buffer.is_empty() {
            // send the final partial buffer
            let final_buffer = std::mem::take(&mut self.buffer);
            self.tx_long.send(final_buffer).unwrap();
        }
        self.alt_offsets
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

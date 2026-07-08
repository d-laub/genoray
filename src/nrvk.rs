use crate::layout::ContigPaths;
use crossbeam_channel::Sender;
use std::fs::File;
use std::os::unix::fs::FileExt;

// strictly bounded memory bank for long alleles (non-blocking, double-buffered memory)
// Public API
pub struct LongAlleleTableWriter {
    tx_long: Sender<Vec<u8>>, //channel to the writer thread
    alt_offsets: Vec<u64>,    // using 64 bits as same is used in seek - no extra casting
    buffer: Vec<u8>,          // staging area
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
        if self.buffer.is_empty() {
            return;
        }

        // empty vector
        let fresh_buffer = Vec::with_capacity(self.buffer.capacity());

        // swap full buffer with new
        let full_buffer = std::mem::replace(&mut self.buffer, fresh_buffer);

        // update the global byte tracker so offsets remain accurate.
        self.global_offset += full_buffer.len();

        // send the full buffer to the background writer thread.
        self.tx_long
            .send(full_buffer)
            .expect("Long Allele Writer Thread panicked");
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
        // Layout matches the writer: {output_dir}/{chrom}/indel/{long_alleles.bin, long_allele_offsets.npy}
        let paths = ContigPaths::new(output_dir, chrom);

        let file = File::open(paths.long_alleles_bin()).expect("Failed to open long_alleles.bin");

        let offsets_array: ndarray::Array1<u64> =
            ndarray_npy::read_npy(paths.long_allele_offsets()).expect("Failed to load offsets npy");

        Self {
            file,
            offsets: offsets_array.into_raw_vec_and_offset().0,
        }
    }

    // Fetches the exact DNA string from the disk using the 31-bit row index.
    pub fn get_allele(&self, row_index: u32) -> Vec<u8> {
        let idx = row_index as usize;
        let start_byte = self.offsets[idx];
        let end_byte = self.offsets[idx + 1];
        let len = (end_byte - start_byte) as usize;
        let mut buf = vec![0u8; len];
        self.file
            .read_exact_at(&mut buf, start_byte)
            .expect("pread long allele");
        buf
    }

    /// CSR row offsets (`len == n_long_alleles + 1`); byte range of row `i` is
    /// `offsets[i]..offsets[i+1]`.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// The entire long-allele byte bank (M6b raw-LUT exposure). Reads the whole
    /// file once; the LUT holds only long-INS spills (SNPs never spill, most
    /// indels are inline), so it is typically small.
    pub fn all_bytes(&self) -> Vec<u8> {
        let total = *self.offsets.last().unwrap_or(&0) as usize;
        let mut buf = vec![0u8; total];
        if total > 0 {
            self.file
                .read_exact_at(&mut buf, 0)
                .expect("pread long_alleles.bin");
        }
        buf
    }
}

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

    #[test]
    fn test_reader_get_allele_shared_borrow() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("chr1").join("indel");
        std::fs::create_dir_all(&dir).unwrap();
        // bytes: "AAAA" then "CC" → offsets [0, 4, 6]
        let mut f = std::fs::File::create(dir.join("long_alleles.bin")).unwrap();
        f.write_all(b"AAAACC").unwrap();
        let offsets = ndarray::Array1::from_vec(vec![0u64, 4, 6]);
        ndarray_npy::write_npy(dir.join("long_allele_offsets.npy"), &offsets).unwrap();

        let reader = LongAlleleReader::new(tmp.path().to_str().unwrap(), "chr1");
        // &self: two immutable calls, no &mut needed
        assert_eq!(reader.get_allele(0), b"AAAA".to_vec());
        assert_eq!(reader.get_allele(1), b"CC".to_vec());
    }

    #[test]
    fn test_reader_all_bytes_and_offsets() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("chr1").join("indel");
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::File::create(dir.join("long_alleles.bin")).unwrap();
        f.write_all(b"AAAACC").unwrap();
        let offsets = ndarray::Array1::from_vec(vec![0u64, 4, 6]);
        ndarray_npy::write_npy(dir.join("long_allele_offsets.npy"), &offsets).unwrap();

        let reader = LongAlleleReader::new(tmp.path().to_str().unwrap(), "chr1");
        assert_eq!(reader.offsets(), &[0u64, 4, 6]);
        assert_eq!(reader.all_bytes(), b"AAAACC".to_vec());
    }
}

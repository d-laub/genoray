use crate::types::DenseChunk;
use ndarray::Array3;
use rust_htslib::bcf::{IndexedReader, Read, record::GenotypeAllele};

pub struct VcfChunkReader {
    inner_reader: IndexedReader,
    /// Column indices (into the file's full sample list) of the requested samples,
    /// in request order. rust-htslib's IndexedReader doesn't expose htslib-level
    /// sample subsetting, so we decode all samples per record and extract these.
    // TODO: push subsetting down to htslib if rust-htslib exposes it, to avoid
    // decoding genotypes for unwanted samples.
    keep_idx: Vec<usize>,
    num_samples: usize,
    ploidy: usize,
}

impl VcfChunkReader {
    // opens the file, resolves the requested samples to column indices, and jumps to the chromosome.
    pub fn new(vcf_path: &str, chrom: &str, samples: &[&str]) -> Self {
        let mut reader = IndexedReader::from_path(vcf_path)
            .expect("Failed to open VCF/BCF index. Is there a .tbi or .csi file?");

        // Resolve requested sample names to their column indices in the file header.
        // Scoped so the immutable header borrow ends before the mutable fetch below.
        let keep_idx: Vec<usize> = {
            let all_samples = reader.header().samples();
            samples
                .iter()
                .map(|s| {
                    all_samples
                        .iter()
                        .position(|fs| *fs == s.as_bytes())
                        .unwrap_or_else(|| panic!("Sample {s} not found in VCF header"))
                })
                .collect()
        };

        // fetch takes a numeric contig id; resolve it from the header, then fetch the
        // whole contig (start 0, no end).
        let rid = reader
            .header()
            .name2rid(chrom.as_bytes())
            .expect("Failed to find chromosome in VCF header");
        reader
            .fetch(rid, 0, None)
            .expect("Failed to fetch chromosome");

        Self {
            inner_reader: reader,
            num_samples: keep_idx.len(),
            keep_idx,
            ploidy: 2, // hardcoded to diploid for this package -> can be changed later (can cause issues)
        }
    }

    // pulls the next `chunk_size` variants from the disk and builds the DenseChunk
    // returns Option so that the thread knows exactly when EOF is reached
    pub fn read_next_chunk(
        &mut self,
        chunk_size: usize,
        chunk_id: usize,
    ) -> Option<DenseChunk<u32>> {
        // pre-allocate the arrays
        let mut pos = Vec::with_capacity(chunk_size);
        let mut ilens = Vec::with_capacity(chunk_size);

        let mut alt = Vec::with_capacity(chunk_size * 5); // flat array of all ALT characters
        let mut alt_offsets = Vec::with_capacity(chunk_size + 1);
        alt_offsets.push(0u32);

        let mut genos_flat = vec![false; chunk_size * self.num_samples * self.ploidy];

        let mut current_v_idx = 0;
        let mut current_alt_offset = 0u32;

        let mut record = self.inner_reader.empty_record();

        while current_v_idx < chunk_size {
            // read the next row. None => EOF for this chromosome.
            match self.inner_reader.read(&mut record) {
                None => break,
                Some(Ok(())) => {}
                Some(Err(e)) => panic!("Failed to read VCF record: {e}"),
            }

            // position
            pos.push(record.pos() as u32);

            // alleles ALT and ILEN
            let alleles = record.alleles();
            let ref_len = alleles[0].len() as i32;
            let mut alt_len = 0i32;

            // handling ALT -> Taking the first ALT if multi-allelic
            if alleles.len() > 1 {
                let alt_allele = alleles[1];
                alt.extend_from_slice(alt_allele);
                alt_len = alt_allele.len() as i32;
                current_alt_offset += alt_len as u32;
            }
            // if there is no ALT allele, the offset doesn't change
            alt_offsets.push(current_alt_offset);
            ilens.push(alt_len - ref_len); // storing ilen directly

            // genotypes
            let genotypes = record.genotypes().expect("Failed to read genotypes");

            for (j, &file_idx) in self.keep_idx.iter().enumerate() {
                let sample_gt = genotypes.get(file_idx);

                let allele_1 = matches!(
                    sample_gt[0],
                    GenotypeAllele::Unphased(1) | GenotypeAllele::Phased(1)
                );

                let allele_2 = if sample_gt.len() > 1 {
                    matches!(
                        sample_gt[1],
                        GenotypeAllele::Unphased(1) | GenotypeAllele::Phased(1)
                    )
                } else {
                    false
                };

                // 1D to 3D Memory Mapping
                let base_idx = (current_v_idx * self.num_samples * self.ploidy) + (j * self.ploidy);
                genos_flat[base_idx] = allele_1;
                genos_flat[base_idx + 1] = allele_2;

                //this is for the ravel operation - need to check on this
                // // The shape of our tensor block
                // let shape = [chunk_size, self.num_samples, self.ploidy];

                // // Raveling the 3D coordinates into a 1D memory pointer
                // let base_idx = ravel!(shape, [current_v_idx, s_idx, 0]);

                // genos_flat[base_idx] = allele_1;
                // genos_flat[base_idx + 1] = allele_2;
            }

            current_v_idx += 1;
        }

        // if no read, return None to signal EOF
        if current_v_idx == 0 {
            return None;
        }

        // truncate the flat genos array if we hit EOF and didn't fill the whole chunk size
        genos_flat.truncate(current_v_idx * self.num_samples * self.ploidy);

        // reshape
        let genos_3d =
            Array3::from_shape_vec((current_v_idx, self.num_samples, self.ploidy), genos_flat)
                .expect("Failed to reshape genos array");

        Some(DenseChunk {
            chunk_id,
            pos,
            ilens,
            alt,
            alt_offsets,
            genos: genos_3d,
        })
    }
}

// No more 1st pass
// // executes a 1st Pass over the VCF to calculate the exact
// // byte size required for the Long Allele Memory-Mapped Table.
// // It skips genotype parsing to maximize disk read speed.
// pub fn long_allele_table_byte_size(vcf_path: &str, chrom: &str) -> usize {
//     let mut reader = IndexedReader::from_path(vcf_path)
//         .expect("Failed to open VCF/BCF index.");

//     reader.fetch(chrom).expect("Failed to fetch chromosome");

//     let mut total_mmap_bytes = 0usize;
//     let mut record = reader.empty_record();

//     while reader.read(&mut record).unwrap_or(false) {
//         let alleles = record.alleles();

//         // Check if there is an ALT allele
//         if alleles.len() > 1 {
//             let alt_allele = alleles[1];

//             // If it's larger than your vk packing limit -> 13bp
//             if alt_allele.len() > 13 {
//                 total_mmap_bytes += alt_allele.len();
//             }
//         }
//     }
//     total_mmap_bytes
// }

// /*
// The Reader Thread runs a loop, calling read_vcf_chunk and pushing the resulting
// DenseChunk into a crossbeam channel.
// */
// pub fn read_vcf_chunk<I: PrimInt>(
//     file_path: &str,
//     chrom: &str,
//     start_row: usize,
//     chunk_size: usize,
//     samples: &[&str], // &[&str] is the zero-cost Rust equivalent of Vec<str>
// ) -> DenseChunk<I> {

//     // 1. Initialize empty Vecs for pos, ref, ref_offsets, alt, alt_offsets
//     // 2. Initialize the flat genos vector
//     // 3. Loop through the htslib VCF records for 'chunk_size' iterations
//     // 4. Populate the Vecs and reshape genos into Array3
//     // 5. Return the DenseChunk struct

//     unimplemented!()
// }

// pub fn read_vcf_genotypes(
//     file_path: &str,
//     chrom: &str,  // this must match the vcf file format chrom -> have to make it more robust
//     start_idx: usize, // this is relative to the start of chrom - eg chr1 starts at 20 start_idx  = 20  so idx  = 40
//     end_idx: usize,
//     samples: &[&str],
// ) -> Array<bool, Ix3> {

//     let num_variants = end_idx - start_idx;
//     let num_samples = samples.len();
//     let ploidy = 2; // diploid

//     // flat memory allocation
//     let mut flat_genos = vec![false; num_variants * num_samples * ploidy];

//     // indexed reader for fast seeks to that chrom
//     let mut reader = IndexedReader::from_path(file_path)
//         .expect("Failed to open index. Make sure a .tbi or .csi file exists!");

//     // converting strgin slices to byte slices for htslib
//     let sample_bytes: Vec<&[u8]> = samples.iter().map(|s| s.as_bytes()).collect();

//     //only decode these specific samples
//     reader.set_samples(&sample_bytes).expect("Failed to set sample subset");

//     reader.fetch(chrom).expect("Failed to fetch region. Is the chromosome name correct?");

//     let mut current_variant_idx = 0;

//     // 3. Iterate through the records
//     for record_result in reader.records() {
//         let mut record = record_result.unwrap();

//         if current_variant_idx < start_idx {
//             current_variant_idx += 1;
//             continue;
//         }
//         if current_variant_idx >= end_idx {
//             break;
//         }

//         let local_v_idx = current_variant_idx - start_idx;

//         // Extract the genotype data for all samples in this variant row
//         let genotypes = record.genotypes().expect("Failed to read genotypes");

//         for (s_idx, sample_gt) in genotypes.into_iter().enumerate() {

//             // using htslib to parse the alleles (handling phasing and missing data) - assuming diploid
//             let allele_1 = match sample_gt[0] {
//                 GenotypeAllele::Unphased(1) | GenotypeAllele::Phased(1) => true,
//                 _ => false, // 0, missing, or other
//             };

//             let allele_2 = if sample_gt.len() > 1 {
//                 match sample_gt[1] {
//                     GenotypeAllele::Unphased(1) | GenotypeAllele::Phased(1) => true,
//                     _ => false,
//                 }
//             } else {
//                 false // haploid fallback - for Y chrom or X chrom or some mito DNA
//             };

//             let base_idx = (local_v_idx * num_samples * ploidy) + (s_idx * ploidy);
//             flat_genos[base_idx] = allele_1;
//             flat_genos[base_idx + 1] = allele_2;
//         }

//         current_variant_idx += 1;
//     }

//     // reshape to req array
//     Array::from_shape_vec((num_variants, num_samples, ploidy), flat_genos)
//         .expect("Failed to reshape flat vector into 3D array")
// }

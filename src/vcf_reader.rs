use ndarray::{Array, Array3, Ix3};
use rust_htslib::bcf::{self, Read, IndexedReader, record::GenotypeAllele};
use crate::types::DenseChunk;
use crate::{ravel, unravel};

pub struct VcfChunkReader {
    inner_reader: IndexedReader,
    num_samples: usize,
    ploidy: usize,
}

impl VcfChunkReader {
    // opens the file, applies the sample filter at the C-level, and jumps to the chromosome.
    pub fn new(
        vcf_path: &str, 
        chrom: &str, 
        samples: &[&str]) 
    -> Self {
        let mut reader = IndexedReader::from_path(vcf_path)
            .expect("Failed to open VCF/BCF index. Is there a .tbi or .csi file?");

        // converting string slices to byte slices for htslib
        let sample_bytes: Vec<&[u8]> = samples.iter().map(|s| s.as_bytes()).collect();
        
        reader.set_samples(&sample_bytes).expect("Failed to set sample subset");
        reader.fetch(chrom).expect("Failed to fetch chromosome");

        Self {
            inner_reader: reader,
            num_samples: samples.len(),
            ploidy: 2, // hardcoded to diploid for this package -> can be changed later (can cause issues)
        }
    }

    // pulls the next `chunk_size` variants from the disk and builds the DenseChunk
    pub fn read_next_chunk(&mut self, chunk_size: usize, chunk_id: usize) -> DenseChunk<u32> {
        // pre-allocate the arrays
        let mut pos = Vec::with_capacity(chunk_size);
        
        let mut refe = Vec::new(); // flat array of all REF characters
        let mut ref_offsets = Vec::with_capacity(chunk_size + 1);
        ref_offsets.push(0u32);
        
        let mut alt = Vec::new(); // flat array of all ALT characters
        let mut alt_offsets = Vec::with_capacity(chunk_size + 1);
        alt_offsets.push(0u32); 
        
        let mut genos_flat = vec![false; chunk_size * self.num_samples * self.ploidy];
        
        let mut current_v_idx = 0;
        let mut current_ref_offset = 0u32;
        let mut current_alt_offset = 0u32;

        let mut record = self.inner_reader.empty_record();

        while current_v_idx < chunk_size {
            // read the next row. If it fails, we've hit EOF for this chromosome.
            if !self.inner_reader.read(&mut record).unwrap_or(false) {
                break;
            }

            // position
            pos.push(record.pos() as u32);

            // alleles (REF and ALT)
            let alleles = record.alleles();
            
            // handling REF
            let ref_allele = alleles[0];
            refe.extend_from_slice(ref_allele);
            current_ref_offset += ref_allele.len() as u32;
            ref_offsets.push(current_ref_offset);

            // handling ALT -> Taking the first ALT if multi-allelic
            if alleles.len() > 1 {
                let alt_allele = alleles[1];
                alt.extend_from_slice(alt_allele);
                current_alt_offset += alt_allele.len() as u32;
            } else {
                // if there is no ALT allele, the offset doesn't change
            }
            alt_offsets.push(current_alt_offset);

            // genotypes
            let genotypes = record.genotypes().expect("Failed to read genotypes");
            
            for (s_idx, sample_gt) in genotypes.into_iter().enumerate() {
                if s_idx >= self.num_samples { break; }

                let allele_1 = match sample_gt[0] {
                    GenotypeAllele::Unphased(1) | GenotypeAllele::Phased(1) => true,
                    _ => false, 
                };

                // safe parsing for allele 2 - Haploid fallback
                let allele_2 = if sample_gt.len() > 1 {
                    match sample_gt[1] {
                        GenotypeAllele::Unphased(1) | GenotypeAllele::Phased(1) => true,
                        _ => false,
                    }
                } else {
                    false 
                };

                // 1D to 3D Memory Mapping
                // The shape of our tensor block
                let shape = [chunk_size, self.num_samples, self.ploidy];
                
                // Raveling the 3D coordinates into a 1D memory pointer
                let base_idx = ravel!(shape, [current_v_idx, s_idx, 0]);

                genos_flat[base_idx] = allele_1;
                genos_flat[base_idx + 1] = allele_2;
            }

            current_v_idx += 1;
        }

        // if no read, return an empty chunk to signal EOF
        if current_v_idx == 0 {
            return DenseChunk {
                chunk_id, pos: vec![], refe: vec![], ref_offsets: vec![],
                alt: vec![], alt_offsets: vec![],
                genos: Array3::from_shape_vec((0, 0, 0), vec![]).unwrap(),
            };
        }

        // truncate the flat genos array if we hit EOF and didn't fill the whole chunk size
        genos_flat.truncate(current_v_idx * self.num_samples * self.ploidy);

        // reshape
        let genos_3d = Array3::from_shape_vec(
            (current_v_idx, self.num_samples, self.ploidy), 
            genos_flat
        ).expect("Failed to reshape genos array");

        DenseChunk {
            chunk_id,
            pos,
            refe,
            ref_offsets,
            alt,
            alt_offsets,
            genos: genos_3d,
        }
    }
}

// executes a 1st Pass over the VCF to calculate the exact 
// byte size required for the Long Allele Memory-Mapped Table.
// It skips genotype parsing to maximize disk read speed.
pub fn long_allele_table_byte_size(vcf_path: &str, chrom: &str) -> usize {
    let mut reader = IndexedReader::from_path(vcf_path)
        .expect("Failed to open VCF/BCF index.");

    reader.fetch(chrom).expect("Failed to fetch chromosome");

    let mut total_mmap_bytes = 0usize;
    let mut record = reader.empty_record();

    while reader.read(&mut record).unwrap_or(false) {
        let alleles = record.alleles();
        
        // Check if there is an ALT allele
        if alleles.len() > 1 {
            let alt_allele = alleles[1];
            
            // If it's larger than your vk packing limit -> 13bp
            if alt_allele.len() > 13 {
                total_mmap_bytes += alt_allele.len();
            }
        }
    }
    total_mmap_bytes
}


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
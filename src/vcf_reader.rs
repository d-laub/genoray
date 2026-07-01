use crate::types::{BitGrid3, DenseChunk};
use rust_htslib::bcf::{IndexedReader, Read, record::GenotypeAllele};

pub struct VcfChunkReader {
    inner_reader: IndexedReader,
    num_samples: usize,
    ploidy: usize,
    sample_indices: Vec<usize>,
}

impl VcfChunkReader {
    // opens the file, applies the sample filter at the C-level, and jumps to the chromosome.
    pub fn new(
        vcf_path: &str,
        chrom: &str,
        samples: &[&str],
        htslib_threads: usize,
        ploidy: usize,
    ) -> Self {
        let mut reader = IndexedReader::from_path(vcf_path)
            .expect("Failed to open VCF/BCF index. Is there a .tbi or .csi file?");

        // Tells C library to spawn exactly N background decompression threads.
        reader
            .set_threads(htslib_threads)
            .expect("Failed to allocate HTSlib background threads");

        // Clone the header to extract Region IDs and Sample IDs
        let header = reader.header().clone();

        // Resolve the string to a numeric Region ID (rid)
        let rid = header
            .name2rid(chrom.as_bytes())
            .expect("Chromosome not found in VCF header");

        // Fetch the entire chromosome (start: 0, end: None)
        reader
            .fetch(rid, 0, None)
            .expect("Failed to fetch chromosome region");

        // Map requested sample strings to their integer indices in the VCF
        let sample_indices: Vec<usize> = samples
            .iter()
            .map(|name| {
                header
                    .sample_id(name.as_bytes())
                    .unwrap_or_else(|| panic!("Sample {} not found in VCF", name))
            })
            .collect();

        Self {
            inner_reader: reader,
            num_samples: samples.len(),
            ploidy, //: 2, // hardcoded to diploid for this package -> can be changed later (can cause issues)
            sample_indices,
        }
    }

    // pulls the next `chunk_size` variants from the disk and builds the DenseChunk
    // returns Option so that the thread knows exactly when EOF is reached
    pub fn read_next_chunk(&mut self, chunk_size: usize, chunk_id: usize) -> Option<DenseChunk> {
        // pre-allocate the arrays
        let mut pos = Vec::with_capacity(chunk_size);
        let mut ilens = Vec::with_capacity(chunk_size);

        let mut alt = Vec::with_capacity(chunk_size * 5); // flat array of all ALT characters
        let mut alt_offsets = Vec::with_capacity(chunk_size + 1);
        alt_offsets.push(0u32);

        // Bit-packed dense grid: 1 bit per (variant, sample, ploidy) entry.
        // 8x smaller than Vec<bool> for the same logical layout.
        let mut genos = BitGrid3::zeros(chunk_size, self.num_samples, self.ploidy);

        let mut current_v_idx = 0;
        let mut current_alt_offset = 0u32;

        let mut record = self.inner_reader.empty_record();

        while current_v_idx < chunk_size {
            // read the next row. If it fails, we've hit EOF for this chromosome.
            match self.inner_reader.read(&mut record) {
                Some(Ok(_)) => {}
                Some(Err(e)) => panic!("VCF Read Error: {}", e), // fail on corruption
                None => break, // End of file (or end of chromosome region)
            }

            // position
            pos.push(record.pos() as u32);

            // alleles ALT and ILEN
            let alleles = record.alleles();

            // Bi-allelic invariant: input must be normalized with `bcftools norm -m -any`.
            // Multi-allelic records would silently lose ALT[2..], so we panic loudly.
            assert!(
                alleles.len() <= 2,
                "VCF must be normalized with `bcftools norm -m -any`. \
                 Multi-allelic record at pos {} has {} alleles.",
                record.pos(),
                alleles.len(),
            );

            let ref_len = alleles[0].len() as i32;
            let mut alt_len = 0i32;

            // handling ALT -> exactly one ALT after the bi-allelic check above
            if alleles.len() > 1 {
                let alt_allele = alleles[1];
                alt_len = alt_allele.len() as i32;

                // Atomized invariant: complex variants (alt_len > 1 AND alt_len < ref_len)
                // can't be encoded inline by the 1-bit-tag scheme. `bcftools norm --atomize`
                // splits these into SNPs/INS/pure-DEL primitives. Panic if we see one.
                assert!(
                    !(alt_len > 1 && alt_len < ref_len),
                    "VCF must be atomized with `bcftools norm --atomize`. \
                     Complex record at pos {} has REF_LEN={} ALT_LEN={} (cannot be encoded inline).",
                    record.pos(),
                    ref_len,
                    alt_len,
                );

                alt.extend_from_slice(alt_allele);
                current_alt_offset += alt_len as u32;
            }
            // if there is no ALT allele, the offset doesn't change
            alt_offsets.push(current_alt_offset);
            ilens.push(alt_len - ref_len); // storing ilen directly

            // genotypes
            let genotypes = record.genotypes().expect("Failed to read genotypes");

            // Loop through your pre-mapped sample indices instead!
            for (s_idx, &real_vcf_idx) in self.sample_indices.iter().enumerate() {
                if s_idx >= self.num_samples {
                    break;
                }

                // Fetch the genotype for this specific sample
                let sample_gt = genotypes.get(real_vcf_idx);

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

                // 1D to 3D Memory Mapping (matches BitGrid3 row-major C-order)
                let base_idx =
                    (current_v_idx * self.num_samples * self.ploidy) + (s_idx * self.ploidy);
                genos.or_bit(base_idx, allele_1);
                genos.or_bit(base_idx + 1, allele_2);

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

        // shrink V dim if we hit EOF before filling the whole chunk size
        genos.truncate_v(current_v_idx);

        Some(DenseChunk {
            chunk_id,
            pos,
            ilens,
            alt,
            alt_offsets,
            genos,
        })
    }
}

use ndarray::Array3;

// The exact boundary limits for 31-bit integer space
pub const MIN_I31: i32 = -(1 << 30);
pub const MAX_INLINE_LEN: i32 = 13;

// Pre-compute the total size of the valid range at compile time
pub const VALID_RANGE_SPAN: u32 = (MAX_INLINE_LEN - MIN_I31) as u32;

// Defines DenseChunk and SparseChunk structs. All other files import from here.

// The struct produced by the VCF Reader and consumed by the Compute Thread (variant key)
pub struct DenseChunk {
    pub chunk_id: usize,

    // Variant Metadata
    pub pos: Vec<u32>,
    // pub refe: Vec<u8>,
    // pub ref_offsets: Vec<I>,
    pub ilens: Vec<i32>,       // Pre-calculated (ALT len - REF len)
    pub alt: Vec<u8>,
    pub alt_offsets: Vec<u32>, // Taking u32 as chunk should not exceed this range

    // Dense Genotype Tensor - Shape (Variants, Samples, Ploidy)
    pub genos: Array3<bool>, // (V, S, P)
    pub num_variants: usize,
}

// The transposed, sparse packet produced by the Compute Thread and consumed by the Writer Thread
pub struct SparseChunk {
    pub chunk_id: usize,

    // The packed data ordered by Sample
    pub call_positions: Vec<u32>,
    pub call_keys: Vec<u32>, // alt packed keys - variant keys

    // for the K-Way Merge
    // Tracks exactly how many mutations each sample had in this specific chunk.
    // Length must be exactly (Samples * Ploidy).
    pub sample_lengths: Vec<u32>, // Replaced offsets with lengths
}

// mod nrvk;
// mod rvk;

// use nrvk::NonReversibleLongAllele;
// use rvk::dense2sparse_vk;

// fn main() {
//     /* things main has to do:
//         `Stream everything` -> 1st pass for long alleles -> allocation on disk
//         then streaming the conversion and encode (FS with directories in 1st pass) + file lock (create dir + lock it)
//         read chunk of genotype -> put inside the DS
//         || capability -> across contig
//         1. create the genos array based on reading files (vcf, plink)
//         2. additional data - metadata and all
//         3. create contig based long allele memory
//         4. writing them on files (for each contig a different directory)
//     */
// }

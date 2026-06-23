use crate::nrvk::LongAlleleTable;
use crate::types::{DenseChunk, MIN_I31, SparseChunk, VALID_RANGE_SPAN};
use num_traits::PrimInt;

//map (s,p) -> flattened index
#[inline]
#[allow(dead_code)]
fn sp_index(s: usize, p: usize, ploidy: usize) -> usize {
    s * ploidy + p
}

/// SIMD Within A Register (SWAR) string encoder.
/// Packs up to 13 DNA bases into a single u32.
#[inline(always)]
fn encode_alt_inline(alt_allele: &[u8], len: u32) -> u32 {
    debug_assert!(len <= 13);

    // fixed 16-byte memory block initialized to zero
    let mut padded = [0u8; 16];

    padded[..len as usize].copy_from_slice(alt_allele);

    // read the DNA at once into two 64-bit CPU registers
    // using little-endian to ensure predictable bit layouts.
    let block1 = u64::from_le_bytes(padded[0..8].try_into().unwrap());
    let block2 = u64::from_le_bytes(padded[8..16].try_into().unwrap());

    // ASCII-to-2-bit math on 8 characters at once.
    // the mask 0x0303030303030303 applies the "& 3" logic to all 8 bytes at once.
    let bits1 = (block1 >> 1) & 0x0303030303030303;
    let bits2 = (block2 >> 1) & 0x0303030303030303;

    // compressing the bits down into 32-bit space.
    // The shifts align the target 2 bits from each byte into a contiguous block.
    let mut payload = 0u32;

    // left-aligned packing starting exactly at bit 26
    payload |= ((bits1 & 3) as u32) << 25;
    payload |= (((bits1 >> 8) & 3) as u32) << 23;
    payload |= (((bits1 >> 16) & 3) as u32) << 21;
    payload |= (((bits1 >> 24) & 3) as u32) << 19;
    payload |= (((bits1 >> 32) & 3) as u32) << 17;
    payload |= (((bits1 >> 40) & 3) as u32) << 15;
    payload |= (((bits1 >> 48) & 3) as u32) << 13;
    payload |= (((bits1 >> 56) & 3) as u32) << 11;

    // block 2
    payload |= ((bits2 & 3) as u32) << 9;
    payload |= (((bits2 >> 8) & 3) as u32) << 7;
    payload |= (((bits2 >> 16) & 3) as u32) << 5;
    payload |= (((bits2 >> 24) & 3) as u32) << 3;
    payload |= (((bits2 >> 32) & 3) as u32) << 1;

    // tag the length into the top 5 bits and return
    // (this also keeps the LSB 0, demarking "no lookup")
    payload | (len << 27)
}

// Packs (ilen, alt) into a single 32-bit variant key, spilling to the long-allele
// table when the allele cannot be represented inline.
#[inline(always)]
pub fn pack_variant(ilen: i32, alt_allele: &[u8], bank: &mut LongAlleleTable) -> u32 {
    // this evaluates (MIN_I31 <= ilen <= 13)
    if (ilen.wrapping_sub(MIN_I31) as u32) <= VALID_RANGE_SPAN {
        // Reversible packing
        if ilen >= 0 {
            // Positive length (0 to 13) -> inline encoding
            encode_alt_inline(alt_allele, ilen as u32)
        } else {
            // Valid negative length -> direct integer packing.
            // Shift left by 1 -> LSB = 0 and the upper 31 bits are the signed i31 ilen.
            (ilen as u32) << 1
        }
    } else {
        // Long allele table: either an insertion (> 13) or a deletion (< MIN_I31).
        let row_index = bank.push_long_allele(alt_allele);
        // Return the pointer with the LSB (lookup bit) flagged to 1.
        (row_index << 1) | 1
    }
}

// The core dense-to-sparse matrix transposer.
// Flips row-major VCF data into column-major (sample-major) sparse tensors.
pub fn dense2sparse_vk<I: PrimInt>(
    chunk: &DenseChunk<I>,
    bank: &mut LongAlleleTable,
) -> SparseChunk {
    let shape = chunk.genos.shape();
    let v_variants = shape[0];
    let num_samples = shape[1];
    let ploidy = shape[2];
    let columns = num_samples * ploidy;

    // pre-allocate assuming roughly 5% sparsity to prevent heap reallocations
    let estimated_nnz = (v_variants * columns) / 20;
    let mut call_positions = Vec::with_capacity(estimated_nnz);
    let mut call_keys = Vec::with_capacity(estimated_nnz);
    let mut sample_lengths = Vec::with_capacity(columns);

    // Get a flat slice to completely bypass ndarray bounds-checking overhead
    let genos_slice = chunk
        .genos
        .as_slice()
        .expect("Genos array must be contiguous");

    // Outer loops are sample/ploidy, inner loop is variants.
    for s in 0..num_samples {
        for p in 0..ploidy {
            let mut current_sample_calls = 0u32;

            // Fixed stride for CPU cache prefetching
            let base_idx = (s * ploidy) + p;
            let stride = columns;

            for v in 0..v_variants {
                let flat_idx = (v * stride) + base_idx;

                if unsafe { *genos_slice.get_unchecked(flat_idx) } {
                    let pos = chunk.pos[v];
                    let ilen = chunk.ilens[v];

                    let start_idx = chunk.alt_offsets[v].to_usize().unwrap();
                    let end_idx = chunk.alt_offsets[v + 1].to_usize().unwrap();

                    let alt_allele = unsafe { chunk.alt.get_unchecked(start_idx..end_idx) };

                    let packed_key = pack_variant(ilen, alt_allele, bank);

                    call_positions.push(pos);
                    call_keys.push(packed_key);

                    current_sample_calls += 1;
                }
            }
            sample_lengths.push(current_sample_calls);
        }
    }

    SparseChunk {
        chunk_id: chunk.chunk_id,
        call_positions, // Vec<u32>
        call_keys,      // Vec<u32>
        sample_lengths, // Vec<u32>
    }
}

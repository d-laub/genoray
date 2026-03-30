use crate::nrvk::LongAlleleTable;
use ndarray::{Array, Array1, Array3, ArrayView3, Ix1, Ix3, s};
use num_traits::PrimInt;
use std::sync::Arc;
use crate::types::{DenseChunk, SparseChunk};
use packed_seq::{PackedSeqVec, Seq};

// fn hasher()

//map (s,p) -> flattened index
#[inline]
fn sp_index(s: usize, p: usize, ploidy: usize) -> usize {
    s * ploidy + p
}

// inline bit packing helper
#[inline(always)]
fn encode_base(base: u8) -> u32 {
    match base {
        b'A' | b'a' => 0b00,
        b'C' | b'c' => 0b01,
        b'G' | b'g' => 0b10,
        b'T' | b't' => 0b11,
        _ => 0,
    }
}

// bit-trick to convert ASCII DNA to 2-bit values without branching
// faster as no branching so highly unrollable and max ILP
// 'A' (65) 0100 0001 -> shift right -> 0010 0000 and 3 (011) -> 00
// 'C' (67) -> 0100 0011 -> 01
// 'T' (84) -> 0101 0100 -> shift right -> 0010 1010 and 3 (011) -> 10
// 'G' (71) -> 0100 0111 -> shift right -> 0010 0011 and 3 (011) -> 11
#[inline(always)]
fn ascii_to_2bit(base: u8) -> u32 {
    ((base >> 1) & 3) as u32
}
// instead of doing this do by taking a slice of alt

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
    
    // packing Block 1 (Bases 0 to 7) -> Occupies bits 0 to 15
    payload |= ((bits1 >> 0)  & 0x00000003) as u32;
    payload |= ((bits1 >> 6)  & 0x0000000C) as u32;
    payload |= ((bits1 >> 12) & 0x00000030) as u32;
    payload |= ((bits1 >> 18) & 0x000000C0) as u32;
    payload |= ((bits1 >> 24) & 0x00000300) as u32;
    payload |= ((bits1 >> 30) & 0x00000C00) as u32;
    payload |= ((bits1 >> 36) & 0x00003000) as u32;
    payload |= ((bits1 >> 42) & 0x0000C000) as u32;

    // packing Block 2 (Bases 8 to 12) -> Occupies bits 16 to 25
    let mut payload2 = 0u32;
    payload2 |= ((bits2 >> 0)  & 0x00000003) as u32;
    payload2 |= ((bits2 >> 6)  & 0x0000000C) as u32;
    payload2 |= ((bits2 >> 12) & 0x00000030) as u32;
    payload2 |= ((bits2 >> 18) & 0x000000C0) as u32;
    payload2 |= ((bits2 >> 24) & 0x00000300) as u32;


    // TODO: have to check for correctness beyond this point

    // combine them (Block 2 is shifted up by 16 bits to sit right above Block 1)
    payload |= payload2 << 16;

    // masking any extra bits that came from the padded zeros
    // If len = 3, we want to keep 6 bits. (1 << 6) - 1 = 00111111 binary.
    let valid_bits_mask = (1 << (len * 2)) - 1;
    payload &= valid_bits_mask;
    
    // adding the length tag to bits 27-30 and return. 
    // Bit 31 remains 0, confirming this is an inline payload.
    payload | (len << 27)
}

// SIMD version of generate variant key for better parallel encoding
#[inline(always)]
pub fn pack_variant(
    ref_len: usize,
    alt_len: usize,
    alt_allele: &[u8],
    long_allele_table: &LongAlleleTable,
) -> u32 {
    let ilen: i32 = (alt_len as i32) - (ref_len as i32);

    // this evaluates (MIN_I31 <= ilen <= 13)
    if (ilen.wrapping_sub(MIN_I31) as u32) <= VALID_RANGE_SPAN {
        // Reversible Packing
        if ilen >= 0 {
            // Positive length (0 to 13) -> Inline Encoding
            return encode_alt_inline(alt_allele, ilen as u32);
        } else {
            // Valid negative length -> Direct Integer Packing
            // Mask out the lsb to ensure it doesn't trigger the lookup flag
            // return (ilen as u32) & 0xFFFFFFFE; -> can also use this
            // shift left by 1 -> LSB = 0 and MSB 31 bits are signed ilen i31
            return (ilen as u32) << 1;
        }
        
    } else {
        // Long Allele Table
        // either an insertion (> 13) or a deletion (< MIN_I31)
        let row_index = long_allele_table.push_allele(alt_allele);
        return (row_index << 1) | 1; // Return the pointer with the lsb (lookup bit) flagged to 1
    }
}



// encoding the ilen, and alt allele (with rk or nrk) into a single key
#[inline(always)]
fn encode_variant_key(
    ref_len: usize,
    alt_len: usize,
    alt_allele: &[u8],
    long_allele_bank: &mut NonReversibleLongAllele,
) -> u32 {
    // will have to check based on the size so that cn decide if to use the ptr or not
    // lookup flag ->
    // 		true -> when ilen is > 13 or if ilen if < minimum value of i31
    //		false -> o.w. it is an actual vk not a pointer
    /* if lookup flag {
            1. yes -> 31 bits for an unsigned address
            2. no ->
                a. if ilen >= 0 then ilen = i5 and alt = 26 bits of 2 bit each nucleotide
                b. if ilen < 0 then ilen = i31
        }
    */
    // AC 
    // 0010 00000000000 >> 5 | ilen | flag


    let ilen: i32 = (alt_len as i32) - (ref_len as i32);
    let min_i31: i32 = -(1 << 30); //constant
    let mut seq_bits: u32 = 0; //vk

    if (ilen >= 0 && ilen <= 13) || (ilen < 0 && ilen >= min_i31) {
        // Path A: Reversible Packing (when +ve ilen but <= 13 alt len, or when -ve but in range of i31)
        if ilen >= 0 {
            //then it can fit in the 31 bits with last bit being 0
            let key = ((ilen as u32) & 0x1F) << 27; //write decimal instead of hex
            for (i, &b) in alt_allele.iter().enumerate() {
                // prevent shifting out of bounds if a rogue string gets through
                if i >= 13 {
                    break; //panic 
                }
                // u8s to bits -> vectorized 
                // calculate exactly how far left to push this specific base -> left aligned
                // Base 0 shifts by 25. Base 1 shifts by 23. Base 2 shifts by 21...
                let shift_amount = 25 - (i * 2);

                // encoding the base and dropping it to the assigned slot
                // eg: Alt: ACGT ilen = 4 -> seq_bits = 00100 00011011 000000000000000000 0
                seq_bits |= encode_base(b) << shift_amount;
            }
            seq_bits |= key;
        } else {
            // shift left by 1 -> LSB = 0 and MSB 31 bits are signed ilen i31
            seq_bits = (ilen as u32) << 1;
        }
    } else {
        // Path B: Ptr Fallback for large variants (> 13 alt len or more than i31 deletions)
        // 31 bits for an usigned add + 1 bit for flag True
        // modifies the data of long allele bank and returns the pointer to that with lsb 1 as lookup flag
        row_index = long_allele_bank.push_variant(ilen, alt_allele); // should not couple in table 
        // tagging the lsb to 1
        return (row_index << 1) | 1;
    }
    seq_bits
}

pub fn dense2sparse_vk<I: PrimInt>(
    chunk: &DenseChunk<I>,
    long_allele_bank: &Arc<SharedArena>, 
) -> SparseChunk {
    
    // TODO

    // 1. Iterate Sample-Major (for s in 0..samples { for p in 0..ploidy { for v in 0..variants } })
    // 2. Check if chunk.genos[[v, s, p]] is true
    // 3. If true, extract pos, ref, alt using the v index
    // 4. Do SIMD encoding or push to long_allele_bank depending upon the len
    // 5. Push to call_positions and call_keys
    // 6. Track the number of calls for this specific sample and push to sample_lengths
    
    // Return formatted SparseChunk ready for the Writer Thread
    SparseChunk {
        chunk_id: chunk.chunk_id,
        call_positions, // Vec<u32>
        call_keys,      // Vec<u32>
        sample_lengths, // Vec<u32> -> sum of all variants across samples
    }
}

// pub fn dense2sparse_vk<I: PrimInt, H: Fn(u64) -> u64>(
//     pos: &[u32],
//     refe: &[u8],
//     ref_offsets: &[I],
//     alt: &[u8],
//     alt_offsets: &[I],
//     start: usize,
//     end: usize,
//     genos: &Array<bool, Ix3>, // Varients, Samples, ploidy (V, S, P)
//     long_allele_bank: &mut NonReversibleLongAllele,
// ) -> (Vec<u32>, Vec<u32>, Vec<u64>) {
//     /*
//     Convert dense genotypes to sparse variant-key genotypes

//     Args:
//         pos: (total_variants)
//         ref: (total_ref_length)
//         ref_offsets: (total_variants + 1)
//         alt: (total_alt_length)
//         alt_offsets: (total_variants + 1)
//         start_idx
//         end_idx: end_idx - start_idx == n_variants
//         genos: (variants, samples, ploidy)
//         long_allele_bank: ref to SoA for the nr alleles

//     Returns:
//         positions: (n_calls)
//         ilen_alt: (n_calls)
//         offsets: (samples * ploidy + 1)
//     */

//     let genos = genos.slice(s![start..end, .., ..]);

//     let shape = genos.shape();
//     let samples = shape[1];
//     let ploidy = shape[2];
//     let n_hap = samples * ploidy;

//     //step 1 - counting mutations per hap
//     let mut counts = vec![0u64; n_hap];

//     for ((_v, s, p), &has_mut) in genos.indexed_iter() {
//         if has_mut {
//             counts[s * ploidy + p] += 1;
//         }
//     }

//     //step 2 - prefix sum for offsets
//     let mut offsets = vec![0u64; n_hap + 1];
//     for i in 0..n_hap {
//         offsets[i + 1] = offsets[i] + counts[i];
//     }

//     let n_calls = offsets[n_hap] as usize;

//     //step 3
//     let mut out_pos = vec![0u32; n_calls];
//     let mut ilen_alt = vec![0u32; n_calls];

//     let mut write_heads = offsets.clone();

//     for ((v, s, p), &has_mutation) in genos.indexed_iter() {
//         if has_mutation {
//             let global_idx = start + v;
//             let hap_idx = s * ploidy + p;
//             let write_idx = write_heads[hap_idx] as usize;

//             // get start end end indices from the offsets arrays

//             let ref_start = ref_offsets[global_idx] as usize;
//             let ref_end = ref_offsets[global_idx + 1] as usize;

//             let alt_start = alt_offsets[global_idx] as usize;
//             let alt_end = alt_offsets[global_idx + 1] as usize;

//             // don't require ref slice for now -> can be added later
//             let alt_slice = &alt[alt_start..alt_end];

//             let ref_len = ref_end - ref_start;
//             let alt_len = alt_slice.len();

//             // generate the lower 32 bits -> ilen (5 bits signed) + alt encoded (26 bits) + flag for ptr lookup (1 bit)

//             let ilen_alt_flag = encode_variant_key(ref_len, alt_len, alt_slice, long_allele_bank);

//             // scatter the data to correct positions
//             out_pos[write_idx] = pos[global_idx];
//             ilen_alt[write_idx] = ilen_alt_flag; // ilen (5 bits signed) + alt encoded (26 bits) + flag for ptr lookup (1 bit)

//             // moving the write head forward for next mut for this haplotype to be right next
//             write_heads[hap_idx] += 1;
//         }
//     }
//     (out_pos, ilen_alt, offsets)
// }

use ndarray::{Array, Array1, ArrayView3, Ix1, Ix3};
use num_traits::PrimInt;

// fn hasher()

//map (s,p) -> flattened index
#[inline]
fn sp_index(s: usize, p: usize, ploidy: usize) -> usize {
    s * ploidy + p
}

// inline bit packing helper
#[inline(always)]
fn encode_base(base: u8) -> u64 {
    match base {
        b'A' | b'a' => 0b00,
        b'C' | b'c' => 0b01,
        b'G' | b'g' => 0b10,
        b'T' | b't' => 0b11,
        _ => 0,
    }
}

// encoding the ilen, and alt allele (with rk or nrk) into a single key
#[inline(always)]
fn encode_variant_key(ref_len: u64, alt_len: u64, alt_allele: &[u8]) -> u64 {
    // will have to check based on the size so that cn decide if to use the ptr or not
    // lookup flag ->
    // 		true -> when ilen if +ve is > 13 or if ilen if -ve is less than i31
    //		false -> o.w.
    /* if lookup flag {
            1. yes -> 31 bits for an unsigned address
            2. no ->
                a. if ilen >= 0 then ilen = i5 and alt = 26 bits of 2 bit each nucleotide
                b. if ilen < 0 then ilen = i31
        }
    */

    let ilen: i32 = (alt_len as i32) - (ref_len as i32);
    let min_i31: i32 = -(1 << 30);
    let mut seq_bits: u32 = 0;

    if (ilen >= 0 && ilen <= 13) || (ilen < 0 && ilen >= min_i31) {
        // Path A: Reversible Packing (when +ve ilen but <= 13 alt len, or when -ve but in range of i31)
        if ilen >= 0 {
            //then it can fit in the 31 bits with last bit being 0
            let key = ((ilen as u32) & 0x1F) << 27;
            for (i, &b) in alt_allele.iter().enumerate() {
                // prevent shifting out of bounds if a rogue string gets through
                if i >= 13 {
                    break;
                }

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
            seq_bits = ilen << 1;
        }
    } else {
        // Path B: Ptr Fallback for large variants (> 13 alt len or more than i31 deletions)
        // 31 bits for an usigned add + 1 bit for flag True
        let hash_value: u32 = 123456789;
        seq_bits = hash_value | 0b1;
    }
    seq_bits
}

pub fn dense2sparse_vk<I: PrimInt, H: Fn(u64) -> u64>(
    pos: &[u32],
    refe: &[u8],
    ref_offsets: &[I],
    alt: &[u8],
    alt_offsets: &[I],
    start: usize,
    end: usize,
    genos: &Array<bool, { I * 3 }>, // Varients, Samples, ploidy (V, S, P)
    hasher: H,
) -> (Vec<u32>, Vec<u32>, Vec<u64>) {
    /*
    Convert dense genotypes to sparse variant-key genotypes

    Args:
        pos: (total_variants)
        ref: (total_ref_length)
        ref_offsets: (total_variants + 1)
        alt: (total_alt_length)
        alt_offsets: (total_variants + 1)
        start_idx
        end_idx: end_idx - start_idx == n_variants
        genos: (variants, samples, ploidy)
        hasher

    Returns:
        positions: (n_calls)
        ilen_alt: (n_calls)
        offsets: (samples * ploidy + 1)
    */

    let genos = genos.slice(s![start_idx..end_idx, .., ..]);

    let shape = genos.shape();
    let samples = shape[1];
    let ploidy = shape[2];
    let n_hap = samples * ploidy;

    //step 1 - counting mutations per hap
    let mut counts = vec![0u64; n_hap];

    for ((_v, s, p), &has_mut) in genos.indexed_iter() {
        if has_mut {
            counts[s * ploidy + p] += 1;
        }
    }

    //step 2 - prefix sum for offsets
    let mut offsets = vec![0u64; n_hap + 1];
    for i in 0..n_hap {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    let n_calls = offsets[n_hap] as usize;

    //step 3
    let mut out_pos = vec![0u32; n_calls];
    let mut ilen_alt = vec![0u32; n_calls];

    let mut write_heads = offsets.clone();

    for ((v, s, p), &has_mutation) in genos.indexed_iter() {
        if has_mutation {
            let global_idx = start_idx + v;
            let hap_idx = s * ploidy + p;
            let wirte_idx = write_heads[hap_idx] as usize;

            // get start end end indices from the offsets arrays

            let ref_start = ref_offsets[global_idx] as usize;
            let ref_end = ref_offsets[global_idx + 1] as usize;

            let alt_start = alt_offsets[global_idx] as usize;
            let alt_end = alt_offsets[global_idx + 1] as usize;

            // don't require ref slice for now -> can be added later
            let alt_slice = &alt[alt_start..alt_end];

            let ref_len = ref_end - ref_start;
            let alt_len = alt_slice.len();

            // generate the lower 32 bits -> ilen (5 bits signed) + alt encoded (26 bits) + flag for ptr lookup (1 bit)

            let ilen_alt_flag = encode_variant_key(ref_len, alt_len, alt_slice);

            // scatter the data to correct positions
            out_pos[write_idx] = pos[global_idx];
            ilen_alt[write_idx] = ilen_alt_flag; // ilen (5 bits signed) + alt encoded (26 bits) + flag for ptr lookup (1 bit)

            // moving the write head forward for next mut for this haplotype to be right next
            write_heads[hap_idx] += 1;
        }
    }
    (out_pos, ilen_alt, offsets)
}

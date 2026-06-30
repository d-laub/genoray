use crate::nrvk::LongAlleleTableWriter;
use crate::types::{DenseChunk, MAX_INLINE_ALT_LEN, MIN_I31, SparseChunk};

// Reversed byte loading
//
// Places base[0] in the HIGH byte of block1, base[7] in the LOW byte.
// After SWAR or PEXT, base[0]'s 2-bit code naturally lands at bits [15:14]
// of the extracted u32 — MSB-first, no reverse_bits needed.
//
// Block1: up to 8 bases, base[i] → padded[7 - i]
// Block2: up to 5 more bases (13 - 8), base[8 + i] → padded[15 - i]
#[inline(always)]
fn load_padded_reversed(alt_allele: &[u8], n: usize) -> (u64, u64) {
    debug_assert!(n <= 13);
    let mut padded = [0u8; 16];

    let n1 = n.min(8);
    for i in 0..n1 {
        padded[7 - i] = alt_allele[i]; // base[0] → slot 7 (high byte of block1)
    }

    let n2 = n.saturating_sub(8);
    for i in 0..n2 {
        padded[15 - i] = alt_allele[8 + i]; // base[8] → slot 15 (high byte of block2)
    }

    let block1 = u64::from_le_bytes(padded[0..8].try_into().unwrap());
    let block2 = u64::from_le_bytes(padded[8..16].try_into().unwrap());
    (block1, block2)
}

// PEXT reduce (BMI2). Replaces the 13-step OR+SHIFT loop with 2 PEXT ops.
//
// After (block >> 1) & 0x0303..03, each byte holds the 2-bit DNA code in its
// low 2 bits. PEXT collapses those 8 codes into a contiguous 16-bit field;
// because of the reversed load, base[0]'s code occupies bits [15:14].
//
// Layout target: top_shift = 25 (base[0] at payload bits [26:25]).
//   part1 = extracted1 << (25 - 14)   → base[0] at [26:25]
//   part2 = extracted2 >> (30 - 25)   → base[8] at [10:9]
// LSB is left at 0 here — the tag bit is OR'd in by the caller.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
unsafe fn pext_reduce(block1: u64, block2: u64, n_bases: usize) -> u32 {
    use std::arch::x86_64::_pext_u64;
    const SWAR_MASK: u64 = 0x0303030303030303;

    let bits1 = (block1 >> 1) & SWAR_MASK;
    let extracted1 = _pext_u64(bits1, SWAR_MASK) as u32;
    let part1 = extracted1 << 11; // 25 - 14

    if n_bases <= 8 {
        return part1;
    }

    let bits2 = (block2 >> 1) & SWAR_MASK;
    let extracted2 = _pext_u64(bits2, SWAR_MASK) as u32;
    let part2 = extracted2 >> 5; // 30 - 25
    part1 | part2
}

// Portable SWAR (SIMD within a register) fallback (no BMI2). Produces byte-identical output to PEXT.
// Walks the bytes in MSB-first order so base[0] lands at the top_shift slot.
#[inline(always)]
fn swar_reduce_portable(block1: u64, block2: u64, n_bases: usize) -> u32 {
    const SWAR_MASK: u64 = 0x0303030303030303;
    let bits1 = (block1 >> 1) & SWAR_MASK;
    let bits2 = (block2 >> 1) & SWAR_MASK;

    let mut payload: u32 = 0;
    let mut shift: i32 = 25;

    let n1 = n_bases.min(8);
    for i in 0..n1 {
        let byte_shift = (7 - i) * 8; // high byte first → reversed-load order
        payload |= (((bits1 >> byte_shift) & 3) as u32) << shift;
        shift -= 2;
    }

    if n_bases > 8 {
        let n2 = n_bases - 8;
        for i in 0..n2 {
            let byte_shift = (7 - i) * 8;
            payload |= (((bits2 >> byte_shift) & 3) as u32) << shift;
            shift -= 2;
        }
    }

    payload
}

// Dispatch: PEXT if BMI2 is available at runtime, otherwise portable SWAR.
#[inline(always)]
fn encode_bases(alt_allele: &[u8], n_bases: usize) -> u32 {
    let (block1, block2) = load_padded_reversed(alt_allele, n_bases);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            return unsafe { pext_reduce(block1, block2, n_bases) };
        }
    }
    swar_reduce_portable(block1, block2, n_bases)
}

// SNP fast path (~95% of variants). Skip the 16-byte block load entirely:
// 2 ops to extract the 2-bit code, 1 shift. Top 5 bits stay 0 (ilen=0 for SNP).
#[inline(always)]
fn encode_snp(alt_base: u8) -> u32 {
    // (byte >> 1) & 3 → A=00, C=01, T=10, G=11.
    ((alt_base as u32 >> 1) & 3) << 25
}

// Packs up to 13 DNA bases into a single u32 with the inline-encoding layout:
// [ ilen:5 | base[0..alt_len] × 2bit | _ | flag=0 ].
//
// - `alt_allele` carries the bases to encode (length used directly as `alt_len`).
// - `ilen` goes into the top 5 bits. For atomized INS/SNP, ilen = alt_len - 1.
// - Decode reconstructs alt_len as ilen + 1.
#[inline(always)]
fn encode_alt_inline(alt_allele: &[u8], ilen: u32) -> u32 {
    let alt_len = alt_allele.len();
    if alt_len > 13 {
        panic!("Inline ALT must be 13 bases or fewer");
    }
    debug_assert!(ilen <= 12, "inline ilen must be ≤ 12 (alt_len ≤ 13)");

    // SNP fast path — dominates real-world VCF distributions. ilen=0 implicit.
    if alt_len == 1 {
        return encode_snp(alt_allele[0]);
    }

    let payload = encode_bases(alt_allele, alt_len);
    payload | (ilen << 27) // LSB stays 0 → no lookup flag
}

#[inline(always)]
pub fn decode_alt_inline(payload: u32) -> Vec<u8> {
    // Top 5 bits hold ilen; alt_len = ilen + 1 (atomized invariant).
    let ilen = (payload >> 27) as usize;
    let alt_len = ilen + 1;
    let mut decoded = Vec::with_capacity(alt_len);

    const BASES: [u8; 4] = [b'A', b'C', b'T', b'G'];

    for i in 0..alt_len {
        let shift = 25 - (i * 2);
        let bit_val = ((payload >> shift) & 3) as usize;
        decoded.push(BASES[bit_val]);
    }

    decoded
}

// Pack a single variant into its 32-bit key.
//
// Layout (LSB → MSB):
//   bit 0       lookup flag (1 = bank row index, 0 = inline)
//   bits 1..31  payload, interpreted by the discriminator below
//
// Inline discriminator:
//   bit 31 = 0 → INS/SNP. Top 5 bits = ilen, base[i] at bits[26-2i:25-2i],
//                alt_len = ilen + 1 ≤ 13.
//   bit 31 = 1 → pure DEL. Bits[31:1] = signed ilen (i31). alt_len = 1 implicit.
//
// Spill: only long INS (alt_len > 13) goes to the bank. Atomized DELs always
// fit in i31 (chromosomes ≤ 250 Mbp), so they never spill.
#[inline(always)]
pub fn pack_variant(ilen: i32, alt_allele: &[u8], bank: &mut LongAlleleTableWriter) -> u32 {
    if ilen >= 0 {
        // INS or SNP. Atomized inputs satisfy alt_len = ilen + 1.
        if alt_allele.len() <= MAX_INLINE_ALT_LEN {
            return encode_alt_inline(alt_allele, ilen as u32);
        }
        // Long INS overflow: spill alt bytes; ilen is recoverable as
        // alt_bytes.len() - 1 since atomized inputs always have ref_len = 1.
        let row_index = bank.push_long_allele(alt_allele);
        return (row_index << 1) | 1;
    }

    // Pure DEL. ilen is signed-i31; top bit is set → distinguishes from inline INS/SNP.
    debug_assert!(
        ilen >= MIN_I31,
        "pure DEL ilen below MIN_I31 would alias the inline-positive lane",
    );
    (ilen as u32) << 1
}

// // encoding the ilen, and alt allele (with rk or nrk) into a single key
// #[inline(always)]
// fn encode_variant_key(
//     ref_len: usize,
//     alt_len: usize,
//     alt_allele: &[u8],
//     long_allele_bank: &mut NonReversibleLongAllele,
// ) -> u32 {
//     // will have to check based on the size so that cn decide if to use the ptr or not
//     // lookup flag ->
//     // 		true -> when ilen is > 13 or if ilen if < minimum value of i31
//     //		false -> o.w. it is an actual vk not a pointer
//     /* if lookup flag {
//             1. yes -> 31 bits for an unsigned address
//             2. no ->
//                 a. if ilen >= 0 then ilen = i5 and alt = 26 bits of 2 bit each nucleotide
//                 b. if ilen < 0 then ilen = i31
//         }
//     */
//     // AC
//     // 0010 00000000000 >> 5 | ilen | flag

//     let ilen: i32 = (alt_len as i32) - (ref_len as i32);
//     let min_i31: i32 = -(1 << 30); //constant
//     let mut seq_bits: u32 = 0; //vk

//     if (ilen >= 0 && ilen <= 13) || (ilen < 0 && ilen >= min_i31) {
//         // Path A: Reversible Packing (when +ve ilen but <= 13 alt len, or when -ve but in range of i31)
//         if ilen >= 0 {
//             //then it can fit in the 31 bits with last bit being 0
//             let key = ((ilen as u32) & 0x1F) << 27; //write decimal instead of hex
//             for (i, &b) in alt_allele.iter().enumerate() {
//                 // prevent shifting out of bounds if a rogue string gets through
//                 if i >= 13 {
//                     break; //panic
//                 }
//                 // u8s to bits -> vectorized
//                 // calculate exactly how far left to push this specific base -> left aligned
//                 // Base 0 shifts by 25. Base 1 shifts by 23. Base 2 shifts by 21...
//                 let shift_amount = 25 - (i * 2);

//                 // encoding the base and dropping it to the assigned slot
//                 // eg: Alt: ACGT ilen = 4 -> seq_bits = 00100 00011011 000000000000000000 0
//                 seq_bits |= encode_base(b) << shift_amount;
//             }
//             seq_bits |= key;
//         } else {
//             // shift left by 1 -> LSB = 0 and MSB 31 bits are signed ilen i31
//             seq_bits = (ilen as u32) << 1;
//         }
//     } else {
//         // Path B: Ptr Fallback for large variants (> 13 alt len or more than i31 deletions)
//         // 31 bits for an usigned add + 1 bit for flag True
//         // modifies the data of long allele bank and returns the pointer to that with lsb 1 as lookup flag
//         row_index = long_allele_bank.push_variant(ilen, alt_allele); // should not couple in table
//         // tagging the lsb to 1
//         return (row_index << 1) | 1;
//     }
//     seq_bits
// }

// The core Dense-to-Sparse Matrix Transposer.
// Flips Row-Major VCF data into Column-Major (Sample-Major) Sparse Tensors.
pub fn dense2sparse_vk(chunk: &DenseChunk, bank: &mut LongAlleleTableWriter) -> SparseChunk {
    // TODO

    // 1. Iterate Sample-Major (for s in 0..samples { for p in 0..ploidy { for v in 0..variants } })
    // 2. Check if chunk.genos[[v, s, p]] is true
    // 3. If true, extract pos, ref, alt using the v index
    // 4. Do SIMD encoding or push to long_allele_bank depending upon the len
    // 5. Push to call_positions and call_keys
    // 6. Track the number of calls for this specific sample and push to sample_lengths

    // Return formatted SparseChunk ready for the Writer Thread

    let (v_variants, num_samples, ploidy) = chunk.genos.shape;
    let columns = num_samples * ploidy;

    // pre-allocate assuming roughly 5% sparsity to prevent heap allocations
    let estimated_nnz = (v_variants * columns) / 20;
    let mut call_positions = Vec::with_capacity(estimated_nnz);
    let mut call_keys = Vec::with_capacity(estimated_nnz);
    let mut sample_lengths = Vec::with_capacity(columns);

    // Pre-pack each variant's (ilen, alt) into its 32-bit key exactly once.
    // Without this, a common variant carried by N haplotypes pays the SWAR/PEXT
    // encode N times, and — critically — pushes its long allele into the bank
    // N times as well, bloating long_alleles.bin with duplicate rows.
    let mut packed_keys: Vec<u32> = Vec::with_capacity(v_variants);
    for v in 0..v_variants {
        let ilen = unsafe { *chunk.ilens.get_unchecked(v) };
        let start_idx = unsafe { *chunk.alt_offsets.get_unchecked(v) } as usize;
        let end_idx = unsafe { *chunk.alt_offsets.get_unchecked(v + 1) } as usize;
        let alt_allele = unsafe { chunk.alt.get_unchecked(start_idx..end_idx) };
        packed_keys.push(pack_variant(ilen, alt_allele, bank));
    }

    // Raw u64 word slice for the bit-packed dense grid. Same flat-index math
    // as the previous Vec<bool> layout — we just shift to find the word/bit.
    let words: &[u64] = &chunk.genos.words;

    // Outer loops are Sample/Ploidy, Inner loop is Variants.
    for s in 0..num_samples {
        for p in 0..ploidy {
            let mut current_sample_calls = 0u32;

            // Fixed stride for perfect CPU cache prefetching
            let base_idx = (s * ploidy) + p;
            let stride = columns;

            for v in 0..v_variants {
                let flat_idx = (v * stride) + base_idx;
                let word = unsafe { *words.get_unchecked(flat_idx >> 6) };

                if (word >> (flat_idx & 63)) & 1 != 0 {
                    let pos = unsafe { *chunk.pos.get_unchecked(v) };
                    let packed_key = unsafe { *packed_keys.get_unchecked(v) };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BitGrid3;
    use crossbeam_channel::bounded;
    use proptest::prelude::*;

    // Big-capacity bank so push_long_allele never flushes during tests.
    fn make_bank() -> LongAlleleTableWriter {
        let (tx, _rx) = bounded(1024);
        LongAlleleTableWriter::new(tx, 1 << 20)
    }

    // Build a synthetic DenseChunk for engine tests. `bit_pattern` is row-major
    // (V, S, P) — exactly what BitGrid3 stores. `alts` and `refs` parallel pos.
    fn build_test_chunk(
        n_variants: usize,
        n_samples: usize,
        ploidy: usize,
        refs: &[&[u8]],
        alts: &[&[u8]],
        bit_pattern: &[bool],
    ) -> DenseChunk {
        assert_eq!(refs.len(), n_variants);
        assert_eq!(alts.len(), n_variants);
        assert_eq!(bit_pattern.len(), n_variants * n_samples * ploidy);

        let mut alt_buf = Vec::new();
        let mut alt_offsets = vec![0u32];
        let mut ilens = Vec::new();
        for v in 0..n_variants {
            alt_buf.extend_from_slice(alts[v]);
            alt_offsets.push(alt_buf.len() as u32);
            ilens.push(alts[v].len() as i32 - refs[v].len() as i32);
        }

        let mut genos = BitGrid3::zeros(n_variants, n_samples, ploidy);
        for (i, &b) in bit_pattern.iter().enumerate() {
            genos.or_bit(i, b);
        }

        let pos: Vec<u32> = (0..n_variants as u32).map(|i| 100 + i * 10).collect();

        DenseChunk {
            chunk_id: 0,
            pos,
            ilens,
            alt: alt_buf,
            alt_offsets,
            genos,
            num_variants: n_variants,
        }
    }

    // Deterministic xorshift bit pattern keyed off `seed`.
    fn random_bits(n: usize, seed: u64) -> Vec<bool> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state & 1 == 1
            })
            .collect()
    }

    #[test]
    fn test_encoding_roundtrip() {
        let original = b"GATTACA";
        // Atomized convention: ilen = alt_len - 1
        let payload = encode_alt_inline(original, (original.len() - 1) as u32);
        let decoded = decode_alt_inline(payload);
        assert_eq!(original, &decoded[..], "ALT decoder corrupted the sequence");
    }

    #[test]
    fn test_alt_exact_binary_layout() {
        // Layout: top 5 bits = ilen (= alt_len - 1 for atomized variants),
        // base[i] at bits[26-2i:25-2i], LSB = 0 (no lookup flag).

        // "ACGT" → alt_len=4, ilen=3 → top 5 = 3 << 27 = 0x18000000
        // 'A' (00) << 25 = 0
        // 'C' (01) << 23 = 0x00800000
        // 'G' (11) << 21 = 0x00600000
        // 'T' (10) << 19 = 0x00100000  → bases sum = 0x00F00000
        // total = 0x18F00000
        let seq = b"ACGT";
        let payload = encode_alt_inline(seq, 3);
        assert_eq!(
            payload, 0x18F00000,
            "ALT binary layout changed! This will break compatibility."
        );

        // "ACGTTGCAGCATT" → alt_len=13, ilen=12 → top 5 = 12 << 27 = 0x60000000
        // bases (verified) = 0x00F5A694
        let seq = b"ACGTTGCAGCATT";
        let payload = encode_alt_inline(seq, 12);
        assert_eq!(
            payload, 0x60F5A694,
            "ALT binary layout changed! This will break compatibility."
        );

        // "ACGTTGCAGC" → alt_len=10, ilen=9 → top 5 = 9 << 27 = 0x48000000
        // bases (verified) = 0x00F5A680
        let seq = b"ACGTTGCAGC";
        let payload = encode_alt_inline(seq, 9);
        assert_eq!(
            payload, 0x48F5A680,
            "ALT binary layout changed! This will break compatibility."
        );
    }

    #[test]
    #[should_panic(expected = "Inline ALT must be 13 bases or fewer")]
    fn test_alt_overflow_protection() {
        // alt_allele.len() = 16 > 13 → panics regardless of `ilen` argument.
        encode_alt_inline(b"ACGTACGTACGTACGT", 0);
    }

    // SNP fast path must produce the same output as the general SWAR/PEXT encode.
    // Both yield the 2-bit base shifted to bits[26:25] with top 5 (ilen) = 0.
    #[test]
    fn test_snp_fast_path_matches_general_path() {
        for &b in b"ACGT" {
            let buf = [b];
            let fast = encode_snp(b);
            // General path bypassing the alt_len==1 short-circuit
            let (b1, b2) = load_padded_reversed(&buf, 1);
            let general = swar_reduce_portable(b1, b2, 1); // ilen=0 → no top-5 OR
            assert_eq!(
                fast, general,
                "SNP fast path diverged from general path for base {}",
                b as char
            );
        }
    }

    // PEXT path (when BMI2 present) must produce byte-identical output to SWAR.
    #[cfg(target_arch = "x86_64")]
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(5000))]

        #[test]
        fn test_pext_swar_equivalence(dna in "[ACGT]{1,13}") {
            if !is_x86_feature_detected!("bmi2") { return Ok(()); }
            let bytes = dna.as_bytes();
            let n = bytes.len();
            let (b1, b2) = load_padded_reversed(bytes, n);
            let pext_out = unsafe { pext_reduce(b1, b2, n) };
            let swar_out = swar_reduce_portable(b1, b2, n);
            prop_assert_eq!(pext_out, swar_out);
        }

        // Full encode→decode roundtrip across all valid alt lengths (1..=13).
        // Atomized convention: ilen = alt_len - 1.
        #[test]
        fn test_encode_decode_roundtrip(dna in "[ACGT]{1,13}") {
            let bytes = dna.as_bytes();
            let ilen = (bytes.len() - 1) as u32;
            let payload = encode_alt_inline(bytes, ilen);
            let decoded = decode_alt_inline(payload);
            prop_assert_eq!(decoded.as_slice(), bytes);
        }
    }

    // pack_variant lane-dispatch tests — exercise SNP/INS, pure DEL, lookup
    #[test]
    fn test_pack_variant_snp() {
        // SNP: ref=A alt=C, ilen=0. Inline lane, LSB=0, top 5 bits = 0 (ilen=0).
        // 'C' code = 01 → bits[26:25] = 01 → 1 << 25 = 0x02000000.
        let mut bank = make_bank();
        let key = pack_variant(0, b"C", &mut bank);
        assert_eq!(key & 1, 0, "SNP must not set lookup flag");
        assert_eq!(key, 0x02000000);
        assert_eq!(decode_alt_inline(key), b"C".to_vec());
    }

    #[test]
    fn test_pack_variant_insertion() {
        // INS: ref=A alt=ACG, ilen=2. Inline lane.
        let mut bank = make_bank();
        let key = pack_variant(2, b"ACG", &mut bank);
        assert_eq!(key & 1, 0);
        assert_eq!(decode_alt_inline(key), b"ACG".to_vec());
    }

    #[test]
    fn test_pack_variant_pure_del() {
        // DEL: ref=ACGT alt=A, ilen=-3. Direct integer pack: (ilen << 1), LSB=0.
        let mut bank = make_bank();
        let key = pack_variant(-3, b"A", &mut bank);
        assert_eq!(key & 1, 0);
        assert_eq!(((key as i32) >> 1), -3);
        // (-3i32 << 1) = -6 → as u32 = 0xFFFFFFFA
        assert_eq!(key, 0xFFFFFFFAu32);
    }

    #[test]
    fn test_pack_variant_lookup_long_insertion() {
        // alt > 13bp → spills to long-allele bank, returns (row << 1) | 1
        let mut bank = make_bank();
        let alt = b"ACGTACGTACGTACGT"; // 16 bp
        let key = pack_variant(15, alt, &mut bank);
        assert_eq!(key & 1, 1, "Long allele must set lookup flag");
        assert_eq!(key >> 1, 0, "First long allele → row index 0");
    }

    // Note: huge DEL (ilen < MIN_I31) is unreachable in atomized real-world data
    // (chromosomes ≤ 250 Mbp << 2^30), so pure DEL has no bank fallback path.
    // pack_variant carries a debug_assert that fires in debug builds if violated.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        // Inline lane: any ALT 1..=13bp must roundtrip via pack→decode_alt_inline.
        #[test]
        fn test_pack_variant_inline_roundtrip(dna in "[ACGT]{1,13}") {
            let alt = dna.as_bytes();
            let ilen = alt.len() as i32 - 1; // single-base ref
            let mut bank = make_bank();
            let key = pack_variant(ilen, alt, &mut bank);
            prop_assert_eq!(key & 1, 0, "inline lane must clear lookup flag");
            let temp_alt = decode_alt_inline(key);
            prop_assert_eq!(temp_alt.as_slice(), alt);
        }

        // Pure DEL lane: ilen ∈ [MIN_I31, 0) packs as (ilen << 1) and recovers
        // exactly via arithmetic right shift by 1.
        #[test]
        fn test_pack_variant_pure_del_roundtrip(ilen in MIN_I31..0i32) {
            let mut bank = make_bank();
            let key = pack_variant(ilen, b"A", &mut bank);
            prop_assert_eq!(key & 1, 0);
            prop_assert_eq!((key as i32) >> 1, ilen);
        }

        // Lookup lane: oversized ALTs land in the bank and the returned key has
        // LSB=1 with row index strictly increasing per push.
        #[test]
        fn test_pack_variant_lookup_indices_monotonic(
            dnas in proptest::collection::vec("[ACGT]{14,40}", 1..6),
        ) {
            let mut bank = make_bank();
            let mut prev_row: Option<u32> = None;
            for dna in &dnas {
                let alt = dna.as_bytes();
                let ilen = alt.len() as i32 - 1;
                let key = pack_variant(ilen, alt, &mut bank);
                prop_assert_eq!(key & 1, 1);
                let row = key >> 1;
                if let Some(prev) = prev_row {
                    prop_assert!(row > prev, "row index must strictly increase");
                }
                prev_row = Some(row);
            }
        }
    }

    // dense2sparse_vk engine tests — mutation conservation and per-sample order

    #[test]
    fn test_dense2sparse_empty_chunk() {
        // Zero variants, just shape edge case.
        let chunk = build_test_chunk(0, 2, 2, &[], &[], &[]);
        let mut bank = make_bank();
        let sparse = dense2sparse_vk(&chunk, &mut bank);
        assert_eq!(sparse.call_positions.len(), 0);
        assert_eq!(sparse.call_keys.len(), 0);
        assert_eq!(sparse.sample_lengths, vec![0u32; 4]);
    }

    #[test]
    fn test_dense2sparse_all_true_sample_major_layout() {
        // 3 variants x 2 samples x 2 ploidy, every bit set.
        // Each (s,p) hap should see all 3 variant positions in v-order.
        let refs: Vec<&[u8]> = vec![b"A", b"A", b"A"];
        let alts: Vec<&[u8]> = vec![b"C", b"G", b"T"];
        let n_variants = 3;
        let n_samples = 2;
        let ploidy = 2;
        let bit_pattern = vec![true; n_variants * n_samples * ploidy];

        let chunk = build_test_chunk(n_variants, n_samples, ploidy, &refs, &alts, &bit_pattern);
        let mut bank = make_bank();
        let sparse = dense2sparse_vk(&chunk, &mut bank);

        assert_eq!(sparse.call_positions.len(), 12);
        assert_eq!(sparse.sample_lengths, vec![3, 3, 3, 3]);
        // First hap's slice is positions [100, 110, 120]
        assert_eq!(&sparse.call_positions[0..3], &[100, 110, 120]);
        assert_eq!(&sparse.call_positions[3..6], &[100, 110, 120]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1500))]

        // Mutation conservation: every true bit in the dense grid produces exactly
        // one sparse call. Total calls (sum of sample_lengths) == popcount(genos).
        // Catches lost mutations and double-counting.
        #[test]
        fn test_dense2sparse_mutation_conservation(
            n_variants in 1usize..30,
            n_samples in 1usize..8,
            ploidy in 1usize..3,
            seed in any::<u64>(),
        ) {
            let total_bits = n_variants * n_samples * ploidy;
            let bit_pattern = random_bits(total_bits, seed);
            let true_count = bit_pattern.iter().filter(|&&b| b).count();

            // SNPs only — keeps everything in the inline lane.
            let refs: Vec<&[u8]> = vec![&b"A"[..]; n_variants];
            let alts: Vec<&[u8]> = vec![&b"C"[..]; n_variants];
            let chunk = build_test_chunk(n_variants, n_samples, ploidy, &refs, &alts, &bit_pattern);

            let mut bank = make_bank();
            let sparse = dense2sparse_vk(&chunk, &mut bank);

            let total_calls: u32 = sparse.sample_lengths.iter().sum();
            prop_assert_eq!(total_calls as usize, true_count, "lost or doubled mutations");
            prop_assert_eq!(sparse.call_positions.len(), true_count);
            prop_assert_eq!(sparse.call_keys.len(), true_count);
            prop_assert_eq!(sparse.sample_lengths.len(), n_samples * ploidy);
        }

        // Per-sample correctness: for each (s,p) hap, the slice of call_positions
        // identified by sample_lengths must exactly match the variants where that
        // hap's bit is set, in v-order.
        #[test]
        fn test_dense2sparse_per_sample_calls(
            n_variants in 1usize..15,
            n_samples in 1usize..6,
            ploidy in 1usize..3,
            seed in any::<u64>(),
        ) {
            let total_bits = n_variants * n_samples * ploidy;
            let bit_pattern = random_bits(total_bits, seed);

            let refs: Vec<&[u8]> = vec![&b"A"[..]; n_variants];
            let alts: Vec<&[u8]> = vec![&b"C"[..]; n_variants];
            let chunk = build_test_chunk(n_variants, n_samples, ploidy, &refs, &alts, &bit_pattern);

            let mut bank = make_bank();
            let sparse = dense2sparse_vk(&chunk, &mut bank);

            let mut cursor = 0usize;
            for s in 0..n_samples {
                for p in 0..ploidy {
                    let hap_idx = s * ploidy + p;
                    let calls = sparse.sample_lengths[hap_idx] as usize;

                    // Compute expected positions from the bit pattern + chunk.pos
                    let expected: Vec<u32> = (0..n_variants).filter_map(|v| {
                        let flat_idx = v * n_samples * ploidy + s * ploidy + p;
                        if bit_pattern[flat_idx] { Some(chunk.pos[v]) } else { None }
                    }).collect();

                    let actual: Vec<u32> = sparse.call_positions[cursor..cursor + calls].to_vec();
                    prop_assert_eq!(&actual, &expected, "hap {} positions mismatch", hap_idx);
                    cursor += calls;
                }
            }
            prop_assert_eq!(cursor, sparse.call_positions.len());
        }
    }
}

//! LSB-first bit-slice helpers for the dense genotype matrix. Bit `i` lives at
//! byte `i >> 3`, position `i & 7`. `copy_bits` is the single hot spot for the
//! dense merge's per-hap bit-concatenation (see dense_merge.rs); it is kept
//! isolated so it can be optimized in place later (see M4 open questions).

#[inline(always)]
pub fn set_bit(buf: &mut [u8], idx: usize) {
    buf[idx >> 3] |= 1u8 << (idx & 7);
}

#[inline(always)]
pub fn get_bit(buf: &[u8], idx: usize) -> bool {
    (buf[idx >> 3] >> (idx & 7)) & 1 != 0
}

/// Copy `n` bits from `src` (starting at bit `src_bit`) into `dst` (starting at
/// bit `dst_bit`). Bits already set in the untouched region of `dst` are
/// preserved; the target window is OR-written from a zeroed assumption
/// (callers pass a zeroed destination window).
pub fn copy_bits(dst: &mut [u8], dst_bit: usize, src: &[u8], src_bit: usize, n: usize) {
    for i in 0..n {
        if get_bit(src, src_bit + i) {
            set_bit(dst, dst_bit + i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_set_get_roundtrip_across_byte_boundary() {
        let mut b = vec![0u8; 2];
        set_bit(&mut b, 0);
        set_bit(&mut b, 7);
        set_bit(&mut b, 8);
        set_bit(&mut b, 15);
        assert!(get_bit(&b, 0) && get_bit(&b, 7) && get_bit(&b, 8) && get_bit(&b, 15));
        assert!(!get_bit(&b, 1));
        assert_eq!(b, vec![0b1000_0001u8, 0b1000_0001u8]);
    }

    #[test]
    fn test_copy_bits_unaligned() {
        // src bits [3..3+5) = pattern; copy to dst starting at bit 6.
        let mut src = vec![0u8; 2];
        // set src bits 3,4,6,7 (within the 5-bit window [3,8))
        for &i in &[3usize, 4, 6, 7] {
            set_bit(&mut src, i);
        }
        let mut dst = vec![0u8; 2];
        copy_bits(&mut dst, 6, &src, 3, 5); // copies src[3..8] → dst[6..11]
        // src window bits (offset j in 0..5): j=0(src3)=1, j=1(src4)=1, j=2(src5)=0,
        // j=3(src6)=1, j=4(src7)=1 → dst bits 6,7,9,10 set; dst 8 clear.
        for &(i, want) in &[(6usize, true), (7, true), (8, false), (9, true), (10, true)] {
            assert_eq!(get_bit(&dst, i), want, "dst bit {}", i);
        }
    }

    proptest! {
        // copy_bits matches a naive reference for arbitrary offsets/lengths.
        #[test]
        fn test_copy_bits_matches_reference(
            src_bytes in proptest::collection::vec(any::<u8>(), 1..8),
            dst_bit in 0usize..40,
            src_bit in 0usize..40,
            n in 0usize..40,
        ) {
            // ensure buffers are big enough
            let need_src = (src_bit + n).div_ceil(8).max(src_bytes.len());
            let mut src = src_bytes.clone();
            src.resize(need_src, 0);
            let dst_len = (dst_bit + n).div_ceil(8).max(1);
            let mut dst = vec![0u8; dst_len];
            let mut reference = vec![0u8; dst_len];

            copy_bits(&mut dst, dst_bit, &src, src_bit, n);
            for i in 0..n {
                if get_bit(&src, src_bit + i) {
                    set_bit(&mut reference, dst_bit + i);
                }
            }
            prop_assert_eq!(dst, reference);
        }
    }
}

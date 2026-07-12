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

/// Read 8 LSB-first bits from `src` starting at absolute bit `bit`, returned as
/// a byte whose bit `j` is `src` bit `bit + j`. `bit .. bit + 8` must be in range
/// except that the final partial byte is padded with zeros (`get`/`unwrap_or`).
#[inline(always)]
fn read_byte(src: &[u8], bit: usize) -> u8 {
    let byte_idx = bit >> 3;
    let off = bit & 7;
    if off == 0 {
        src[byte_idx]
    } else {
        let lo = src[byte_idx] >> off;
        let hi = (*src.get(byte_idx + 1).unwrap_or(&0)) << (8 - off);
        lo | hi
    }
}

/// Copy `n` bits from `src` (starting at bit `src_bit`) into `dst` (starting at
/// bit `dst_bit`). Bits already set in the untouched region of `dst` are
/// preserved; the target window is OR-written from a zeroed assumption
/// (callers pass a zeroed destination window).
///
/// The middle of the window is copied a byte (8 bits) at a time once `dst` is
/// byte-aligned — this removes the per-bit branch that made the bit-by-bit
/// version the dense-merge hot spot. The unaligned head/tail stay bit-by-bit.
pub fn copy_bits(dst: &mut [u8], dst_bit: usize, src: &[u8], src_bit: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut i = 0usize;
    // Head: advance bit-by-bit until the destination cursor is byte-aligned.
    while i < n && (dst_bit + i) & 7 != 0 {
        if get_bit(src, src_bit + i) {
            set_bit(dst, dst_bit + i);
        }
        i += 1;
    }
    // Body: destination is byte-aligned — OR whole bytes at a time.
    while i + 8 <= n {
        let byte = read_byte(src, src_bit + i);
        if byte != 0 {
            dst[(dst_bit + i) >> 3] |= byte;
        }
        i += 8;
    }
    // Tail: remaining < 8 bits, unaligned again.
    while i < n {
        if get_bit(src, src_bit + i) {
            set_bit(dst, dst_bit + i);
        }
        i += 1;
    }
}

/// Call `f(k)` for every set bit `k` in `0..n` of the contiguous bit-window
/// `[start_bit, start_bit + n)` of `buf`, in ascending `k`. `k` is the offset
/// *within the window*, so a hap-major row `[hap * n_dense, +n_dense)` yields
/// exactly the carried dense-variant columns.
///
/// Whole-byte body scan (skipping zero bytes, then iterating each nonzero
/// byte's set bits via `trailing_zeros`) replaces the per-column `get_bit`
/// call — one branch + one shift per *carried* variant instead of per variant.
/// Head/tail handle the unaligned ends bit-by-bit, mirroring `copy_bits`.
#[inline]
pub fn for_each_set_bit(buf: &[u8], start_bit: usize, n: usize, mut f: impl FnMut(usize)) {
    if n == 0 {
        return;
    }
    let mut i = 0usize;
    // Head: bit-by-bit until the source cursor is byte-aligned.
    while i < n && (start_bit + i) & 7 != 0 {
        if get_bit(buf, start_bit + i) {
            f(i);
        }
        i += 1;
    }
    // Body: whole bytes; skip zero bytes, expand set bits of nonzero ones.
    while i + 8 <= n {
        let byte = buf[(start_bit + i) >> 3];
        if byte != 0 {
            let mut m = byte;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                f(i + b);
                m &= m - 1; // clear lowest set bit
            }
        }
        i += 8;
    }
    // Tail: remaining < 8 bits, unaligned again.
    while i < n {
        if get_bit(buf, start_bit + i) {
            f(i);
        }
        i += 1;
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
    fn for_each_set_bit_matches_get_bit_over_windows() {
        // A pseudo-random byte buffer; check the set-bit scan against a
        // per-bit get_bit oracle for several unaligned window offsets/lengths.
        let mut buf = vec![0u8; 40];
        let mut x = 0x1234_5678u32;
        for b in buf.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 16) as u8;
        }
        let total = buf.len() * 8;
        for &(start, n) in &[
            (0usize, 0usize),
            (0, 1),
            (3, 5),
            (7, 20),
            (1, 250),
            (5, total - 5),
        ] {
            let mut got = Vec::new();
            for_each_set_bit(&buf, start, n, |k| got.push(k));
            let want: Vec<usize> = (0..n).filter(|&k| get_bit(&buf, start + k)).collect();
            assert_eq!(got, want, "start={start} n={n}");
        }
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

    /// Test-only helper: strictly independent of `copy_bits`/`get_bit`/`set_bit`.
    /// Decodes bytes into a `Vec<bool>` using inline `>>`/`&` arithmetic.
    fn decode_bits_ref(bytes: &[u8]) -> Vec<bool> {
        let mut bits = Vec::with_capacity(bytes.len() * 8);
        for &b in bytes {
            for j in 0..8u8 {
                bits.push((b >> j) & 1 != 0);
            }
        }
        bits
    }

    /// Test-only helper: strictly independent of `copy_bits`/`get_bit`/`set_bit`.
    /// Re-encodes a `Vec<bool>` into bytes using inline `<<`/`|` arithmetic.
    fn encode_bits_ref(bits: &[bool]) -> Vec<u8> {
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (idx, &bit) in bits.iter().enumerate() {
            if bit {
                bytes[idx / 8] |= 1u8 << (idx % 8);
            }
        }
        bytes
    }

    proptest! {
        // copy_bits matches an INDEPENDENTLY computed reference for arbitrary offsets/lengths.
        // The reference decodes dst/src into bool vectors via inline bit arithmetic (never
        // calling copy_bits/get_bit/set_bit), ORs the copy window, and re-encodes to bytes —
        // so a bug shared between copy_bits and the reference's indexing math cannot hide.
        #[test]
        fn test_copy_bits_matches_reference(
            src_bytes in proptest::collection::vec(any::<u8>(), 1..8),
            dst_bytes_seed in proptest::collection::vec(any::<u8>(), 1..8),
            dst_bit in 0usize..40,
            src_bit in 0usize..40,
            n in 0usize..40,
        ) {
            // ensure buffers are big enough
            let need_src = (src_bit + n).div_ceil(8).max(src_bytes.len());
            let mut src = src_bytes.clone();
            src.resize(need_src, 0);

            // Seed dst with random pre-existing (non-zero) bytes so OR-vs-overwrite is exercised.
            let dst_len = (dst_bit + n).div_ceil(8).max(dst_bytes_seed.len()).max(1);
            let mut dst = dst_bytes_seed.clone();
            dst.resize(dst_len, 0);
            let dst_before = dst.clone();

            copy_bits(&mut dst, dst_bit, &src, src_bit, n);

            // Independent reference: decode -> OR window -> re-encode.
            let mut expected_bits = decode_bits_ref(&dst_before);
            let src_bits = decode_bits_ref(&src);
            for i in 0..n {
                expected_bits[dst_bit + i] |= src_bits[src_bit + i];
            }
            let expected = encode_bits_ref(&expected_bits);

            prop_assert_eq!(dst, expected);
        }
    }

    #[test]
    fn test_copy_bits_preserves_and_ors_dst() {
        // 3-byte dst (24 bits); copy window is dst bits [8, 20).
        let mut dst = vec![0u8; 3];
        // Pre-set bits OUTSIDE the window.
        for &i in &[0usize, 2, 23] {
            set_bit(&mut dst, i);
        }
        // Pre-set bits INSIDE the window:
        //  - bit 9  (window offset 1): src offset 1 will also be 1 -> stays 1 either way.
        //  - bit 12 (window offset 4): src offset 4 will be 0 -> must STAY 1 under OR
        //    semantics (overwrite would incorrectly clear it to 0).
        for &i in &[9usize, 12] {
            set_bit(&mut dst, i);
        }
        let dst_before = dst.clone();

        // src pattern: bits 0, 1, 9, 11 set (bit 4 deliberately left 0).
        let mut src = vec![0u8; 2];
        for &i in &[0usize, 1, 9, 11] {
            set_bit(&mut src, i);
        }
        assert!(!get_bit(&src, 4), "test setup: src offset 4 must be 0");

        copy_bits(&mut dst, 8, &src, 0, 12);

        // (a) every bit outside the window is unchanged.
        for i in (0..8).chain(20..24) {
            assert_eq!(
                get_bit(&dst, i),
                get_bit(&dst_before, i),
                "outside-window bit {} changed",
                i
            );
        }

        // (b) inside the window, result == dst_before OR src (OR, not overwrite).
        for j in 0..12usize {
            let want = get_bit(&dst_before, 8 + j) || get_bit(&src, j);
            assert_eq!(get_bit(&dst, 8 + j), want, "window offset {}", j);
        }

        // Explicit check of the dst=1,src=0 case: proves OR semantics rather than overwrite.
        assert!(
            get_bit(&dst, 12),
            "dst bit 12 was pre-set with src=0 at that offset; OR must keep it 1"
        );
    }
}

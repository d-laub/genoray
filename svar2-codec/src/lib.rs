//! Encode/decode for the SVAR 2.0 variant-key bit layout.
//!
//! This crate is the **single source of truth** for the on-disk key layouts:
//! the 2-bit SNP code stream and the 32-bit indel key (inline INS/SNP, pure DEL,
//! and long-allele-bank lookup lanes). Both the pure encode primitives and the
//! decode primitives live here, so the two halves of the layout can never drift.
//!
//! Pure and std-only: no I/O, no pyo3, no long-allele bank. Callers that need
//! those (file packing, bank spill) live in the `genoray` crate and call in here.

/// Minimum signed `ilen` representable inline as a pure DEL (i31 two's complement).
/// Real data won't approach this — atomized DELs span at most chromosome length
/// (~250 Mbp).
pub const MIN_I31: i32 = -(1 << 30);

/// Maximum ALT byte length that fits the inline encoding (26 bits ÷ 2 bits/base =
/// 13). Beyond this, a pure-INS variant spills to the long-allele bank.
pub const MAX_INLINE_ALT_LEN: usize = 13;

/// 2-bit code → ALT base. `A=00 C=01 T=10 G=11`. `T`/`G` are swapped vs. the
/// obvious alphabetical order — the values are an implementation detail of this
/// crate and carry no meaning outside it.
pub const BASES: [u8; 4] = [b'A', b'C', b'T', b'G'];

/// Bare 2-bit ALT code for the SNP stream: `A=00 C=01 T=10 G=11`. Branchless
/// `(base >> 1) & 3` — no lookup, no match.
#[inline(always)]
pub fn encode_snp_2bit(base: u8) -> u8 {
    (base >> 1) & 3
}

/// Recover the ALT base for a 2-bit SNP code. Inverse of [`encode_snp_2bit`].
#[inline]
pub fn decode_snp_2bit(code: u8) -> u8 {
    BASES[(code & 3) as usize]
}

/// Pack 2-bit SNP codes 4-per-byte, little-pair-first: code `i` occupies bits
/// `[(i&3)*2 + 1 : (i&3)*2]` of byte `i >> 2`. The final byte is zero-padded when
/// `codes.len()` is not a multiple of 4. Offsets index CALLS, not bytes.
pub fn pack_snp_keys(codes: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; codes.len().div_ceil(4)];
    for (i, &c) in codes.iter().enumerate() {
        out[i >> 2] |= (c & 3) << ((i & 3) * 2);
    }
    out
}

/// Inverse of [`pack_snp_keys`]. Returns the first `n` codes.
pub fn unpack_snp_keys(packed: &[u8], n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (packed[i >> 2] >> ((i & 3) * 2)) & 3)
        .collect()
}

/// Read the 2-bit SNP code at call index `i` from a 2-bit-packed buffer
/// (4 codes/byte; see [`pack_snp_keys`]) without materializing the whole array.
#[inline]
pub fn unpack_snp_key_at(packed: &[u8], i: usize) -> u8 {
    (packed[i >> 2] >> ((i & 3) * 2)) & 3
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn snp_code_round_trip_all_bases() {
        for &b in &[b'A', b'C', b'G', b'T'] {
            let code = encode_snp_2bit(b);
            assert_eq!(decode_snp_2bit(code), b, "base {} round-trips", b as char);
        }
    }

    proptest! {
        #[test]
        fn snp_pack_unpack_round_trips(codes in proptest::collection::vec(0u8..4, 0..64)) {
            let packed = pack_snp_keys(&codes);
            prop_assert_eq!(unpack_snp_keys(&packed, codes.len()), codes.clone());
            for (i, &c) in codes.iter().enumerate() {
                prop_assert_eq!(unpack_snp_key_at(&packed, i), c);
            }
        }
    }

    #[test]
    fn test_encode_snp_2bit_mapping() {
        // A=00 C=01 T=10 G=11 via (base >> 1) & 3
        assert_eq!(encode_snp_2bit(b'A'), 0);
        assert_eq!(encode_snp_2bit(b'C'), 1);
        assert_eq!(encode_snp_2bit(b'T'), 2);
        assert_eq!(encode_snp_2bit(b'G'), 3);
    }

    #[test]
    fn test_pack_snp_keys_exact_layout() {
        // codes [1, 2, 3, 0, 1] → byte0 = 1 | (2<<2) | (3<<4) | (0<<6) = 0b00_11_10_01 = 0x39
        //                        → byte1 = 1 (only low pair) = 0x01
        let packed = pack_snp_keys(&[1, 2, 3, 0, 1]);
        assert_eq!(packed, vec![0x39, 0x01]);
    }

    #[test]
    fn test_decode_snp_2bit_inverts_encode() {
        for &b in b"ACTG" {
            assert_eq!(decode_snp_2bit(encode_snp_2bit(b)), b);
        }
    }

    #[test]
    fn test_unpack_snp_key_at_matches_unpack_all() {
        let codes = [1u8, 2, 3, 0, 2, 1, 3];
        let packed = pack_snp_keys(&codes);
        let all = unpack_snp_keys(&packed, codes.len());
        for i in 0..codes.len() {
            assert_eq!(unpack_snp_key_at(&packed, i), all[i], "code {}", i);
            assert_eq!(unpack_snp_key_at(&packed, i), codes[i]);
        }
    }
}

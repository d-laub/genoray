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
pub fn encode_alt_inline(alt_allele: &[u8], ilen: u32) -> u32 {
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

/// Decode the inline INS/SNP lane's ALT bases. Top 5 bits hold `ilen`;
/// `alt_len = ilen + 1` (atomized invariant). Bases read MSB-first from `[26:25]`.
#[inline(always)]
pub fn decode_alt_inline(payload: u32) -> Vec<u8> {
    let ilen = (payload >> 27) as usize;
    let alt_len = ilen + 1;
    let mut decoded = Vec::with_capacity(alt_len);
    for i in 0..alt_len {
        let shift = 25 - (i * 2);
        let bit_val = ((payload >> shift) & 3) as usize;
        decoded.push(BASES[bit_val]);
    }
    decoded
}

/// Discriminated form of a 32-bit indel key. Mirrors the encode lanes:
/// bit 0 = lookup flag, bit 31 (of a non-lookup key) = pure-DEL flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedKey {
    /// Inline INS/SNP; `alt` is the decoded ALT bases (`alt.len() == ilen + 1`).
    Inline { alt: Vec<u8> },
    /// Pure deletion of `-ilen` reference bases (`ilen < 0`). The anchor base is
    /// recovered from the reference downstream, not stored in the key.
    PureDel { ilen: i32 },
    /// Long insertion spilled to the long-allele bank at row `row`.
    Lookup { row: u32 },
}

/// Decode a packed 32-bit indel key into its discriminated form. Single decode
/// entry point — no caller re-derives the bit layout.
pub fn decode_key(key: u32) -> DecodedKey {
    if key & 1 == 1 {
        DecodedKey::Lookup { row: key >> 1 }
    } else if (key as i32) < 0 {
        DecodedKey::PureDel {
            ilen: (key as i32) >> 1,
        }
    } else {
        DecodedKey::Inline {
            alt: decode_alt_inline(key),
        }
    }
}

/// Reference-base deletion length encoded in a 32-bit indel key. Inverse of the
/// DEL lane of [`encode_pure_del`]: a pure DEL clears the lookup flag (bit 0) and
/// sets bit 31, storing signed `ilen` in `[31:1]`; the length is `-ilen`. Inline
/// INS/SNP keys (bit 31 clear) and lookup keys (bit 0 set) both yield `0`.
#[inline]
pub fn deletion_len(key: u32) -> u32 {
    if key & 1 == 1 {
        return 0;
    }
    if key & (1 << 31) == 0 {
        return 0;
    }
    let ilen = (key as i32) >> 1;
    debug_assert!(
        ilen < 0,
        "top-bit-set inline key must be a negative-ilen DEL"
    );
    ilen.unsigned_abs()
}

/// Encode a pure deletion of `-ilen` reference bases into the DEL lane
/// (`bit 0 = 0`, `bit 31 = 1`, signed `ilen` in `[31:1]`). `ilen` must be `< 0`
/// and `>= MIN_I31`.
#[inline]
pub fn encode_pure_del(ilen: i32) -> u32 {
    debug_assert!(ilen < 0, "encode_pure_del expects a negative ilen");
    debug_assert!(
        ilen >= MIN_I31,
        "pure DEL ilen below MIN_I31 aliases the inline lane"
    );
    (ilen as u32) << 1
}

/// Encode a long-allele-bank row index into the lookup lane (`(row << 1) | 1`).
/// `row` must fit in 31 bits.
#[inline]
pub fn encode_lookup(row: u32) -> u32 {
    debug_assert!(row < (1 << 31), "bank row index must fit in 31 bits");
    (row << 1) | 1
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
    }

    #[test]
    fn deletion_len_snp_ins_lookup_are_zero() {
        // inline SNP (ilen 0), inline INS, and a lookup key all decode to 0.
        assert_eq!(deletion_len(encode_alt_inline(b"A", 0)), 0);
        assert_eq!(deletion_len(encode_alt_inline(b"ACG", 2)), 0);
        assert_eq!(deletion_len(encode_lookup(5)), 0);
    }

    #[test]
    fn pure_del_round_trips() {
        for d in [1i32, 3, 100, 1 << 20] {
            let key = encode_pure_del(-d);
            assert_eq!(deletion_len(key), d as u32);
            assert_eq!(decode_key(key), DecodedKey::PureDel { ilen: -d });
        }
    }

    #[test]
    fn lookup_round_trips() {
        assert_eq!(
            decode_key(encode_lookup(42)),
            DecodedKey::Lookup { row: 42 }
        );
    }

    // A long-INS lookup key sets the LSB. Even with the top bit set (a large
    // row index), the LSB check must win → not a deletion. Moved from
    // `genoray::rvk` — pure-decode edge case, codec-native now.
    #[test]
    fn test_deletion_len_lookup_key_is_zero() {
        assert_eq!(deletion_len(0x0000_0003), 0); // LSB set
        assert_eq!(deletion_len(0xFFFF_FFFF), 0); // LSB set AND top bit set
    }

    proptest! {
        #[test]
        fn inline_ins_snp_round_trips(
            bases in proptest::collection::vec(
                prop::sample::select(vec![b'A', b'C', b'G', b'T']), 1..=13usize)) {
            let ilen = (bases.len() - 1) as u32;
            let key = encode_alt_inline(&bases, ilen);
            prop_assert_eq!(decode_key(key), DecodedKey::Inline { alt: bases.clone() });
        }
    }
}

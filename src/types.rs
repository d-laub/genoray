use crate::streams::StreamMap;

// Minimum signed ilen representable inline as a pure DEL (i31 signed two's complement).
// Real data won't approach this — atomized DELs span at most chromosome length (~250 Mbp).
pub const MIN_I31: i32 = -(1 << 30);

// Maximum ALT byte length that fits in the inline encoding (26 bits ÷ 2 bits/base = 13).
// Beyond this, a pure-INS variant spills to the long-allele bank.
pub const MAX_INLINE_ALT_LEN: usize = 13;

// 3-D bit grid laid out row-major (V, S, P), packed 64 bits per u64 word.
// Equivalent to an Array3<bool> with default C-order layout: bit (v, s, p)
// lives at flat index `v * S * P + s * P + p`, in word `idx >> 6`, bit
// position `idx & 63`. 8x smaller than `Vec<bool>` (1 byte/entry → 1 bit/entry),
// keeping the inner transpose loop's stride-by-(S*P) pattern intact at 1/8
// the cache footprint.
pub struct BitGrid3 {
    pub words: Vec<u64>,
    pub shape: (usize, usize, usize), // (V, S, P)
}

impl BitGrid3 {
    pub fn zeros(v: usize, s: usize, p: usize) -> Self {
        let total_bits = v
            .checked_mul(s)
            .and_then(|x| x.checked_mul(p))
            .expect("BitGrid3 shape overflow");
        let n_words = total_bits.div_ceil(64);
        Self {
            words: vec![0u64; n_words],
            shape: (v, s, p),
        }
    }

    /// OR-set the bit at `flat_idx`. Branchless: writes `(value as u64) << bit`,
    /// which is a no-op when `value == false`. Bits start at zero and are never
    /// cleared by callers, so this is the only mutator the writer needs.
    #[inline(always)]
    pub fn or_bit(&mut self, flat_idx: usize, value: bool) {
        let w = flat_idx >> 6;
        let b = flat_idx & 63;
        self.words[w] |= (value as u64) << b;
    }

    /// Read the bit at `flat_idx`. `#[inline(always)]` so it folds into the
    /// hot transpose loop in `dense2sparse_vk`.
    #[inline(always)]
    pub fn get_bit(&self, flat_idx: usize) -> bool {
        let w = flat_idx >> 6;
        let b = flat_idx & 63;
        (self.words[w] >> b) & 1 != 0
    }

    /// Shrink the V axis after early EOF. Bits beyond the new range (including
    /// any partially-used trailing word) stayed zero because callers only ever
    /// OR-set, so no masking is required.
    pub fn truncate_v(&mut self, new_v: usize) {
        debug_assert!(new_v <= self.shape.0);
        let (_, ns, np) = self.shape;
        self.shape = (new_v, ns, np);
        let n_words = (new_v * ns * np).div_ceil(64);
        self.words.truncate(n_words);
    }

    /// Number of set bits in variant `v`'s (S, P) plane — its allele count `x`.
    /// The plane is the contiguous bit range `[v*S*P, (v+1)*S*P)`; counted
    /// word-wise (masked head/tail) so it is O(plane_bits / 64), not per-bit.
    pub fn popcount_plane(&self, v: usize) -> usize {
        let (_, s, p) = self.shape;
        let plane = s * p;
        let start = v * plane;
        let end = start + plane;
        debug_assert!(end <= self.shape.0 * plane);

        let mut count = 0u32;
        let mut bit = start;
        while bit < end {
            let word_idx = bit >> 6;
            let bit_in_word = bit & 63;
            // bits available in this word from `bit_in_word` to end of word (64)
            let word_end_bit = (word_idx + 1) << 6;
            let span_end = end.min(word_end_bit);
            let take = span_end - bit; // 1..=64 bits from this word
            let word = self.words[word_idx];
            // mask the [bit_in_word, bit_in_word+take) window
            let mask = if take == 64 {
                u64::MAX
            } else {
                ((1u64 << take) - 1) << bit_in_word
            };
            count += (word & mask).count_ones();
            bit = span_end;
        }
        count as usize
    }
}

// Defines DenseChunk and SparseChunk structs. All other files import from here.

// The struct produced by the VCF Reader and consumed by the Compute Thread (variant key)
pub struct DenseChunk {
    pub chunk_id: usize,

    // Variant Metadata
    pub pos: Vec<u32>,
    // pub refe: Vec<u8>,
    // pub ref_offsets: Vec<I>,
    pub ilens: Vec<i32>, // Pre-calculated (ALT len - REF len)
    pub alt: Vec<u8>,
    pub alt_offsets: Vec<u32>, // Taking u32 as chunk should not exceed this range

    // Dense Genotype Tensor - Shape (Variants, Samples, Ploidy), bit-packed
    pub genos: BitGrid3, // (V, S, P)
}

// One position-sorted sub-stream of calls with byte-erased keys (`key_bytes`
// wide, little-endian). Type erasure lets streams of differing widths live in
// a single `StreamMap`.
pub struct SparseSubStream {
    // Per-call data, sample-major (sample, ploid, ~variant).
    pub call_positions: Vec<u32>,
    pub call_keys: Vec<u8>, // key_bytes per call
    // Calls per (sample, ploid) in THIS sub-stream. Length == samples * ploidy.
    pub sample_lengths: Vec<u32>,
    pub key_bytes: usize,
}

impl SparseSubStream {
    pub fn with_capacity(key_bytes: usize, nnz: usize, columns: usize) -> Self {
        Self {
            call_positions: Vec::with_capacity(nnz),
            call_keys: Vec::with_capacity(nnz * key_bytes),
            sample_lengths: Vec::with_capacity(columns),
            key_bytes,
        }
    }
    #[inline(always)]
    pub fn push_call(&mut self, pos: u32, key_le: &[u8]) {
        debug_assert_eq!(key_le.len(), self.key_bytes);
        self.call_positions.push(pos);
        self.call_keys.extend_from_slice(key_le);
    }
}

// The transposed, sparse packet produced by the Compute Thread and consumed by
// the Writer Thread — one `SparseSubStream` per active `StreamTag` (see
// data-model.md#on-disk-layout).
pub struct SparseChunk {
    pub chunk_id: usize,
    pub streams: StreamMap<SparseSubStream>,
}

#[cfg(test)]
mod tests {
    // Flat-index loops below deliberately index by `idx` to validate BitGrid3's
    // internal (v, s, p) → flat math against a parallel Vec<bool>.
    #![allow(clippy::needless_range_loop)]
    use super::*;
    use proptest::prelude::*;

    // Manual flat-index helper to validate against BitGrid3's internal math.
    fn flat(v: usize, s: usize, p: usize, ns: usize, np: usize) -> usize {
        v * ns * np + s * np + p
    }

    #[test]
    fn test_zeros_all_unset() {
        let g = BitGrid3::zeros(4, 3, 2);
        assert_eq!(g.shape, (4, 3, 2));
        for v in 0..4 {
            for s in 0..3 {
                for p in 0..2 {
                    assert!(!g.get_bit(flat(v, s, p, 3, 2)));
                }
            }
        }
    }

    #[test]
    fn test_or_bit_no_op_for_false() {
        let mut g = BitGrid3::zeros(2, 2, 2);
        g.or_bit(0, false);
        g.or_bit(3, false);
        assert!(g.words.iter().all(|&w| w == 0));
    }

    #[test]
    fn test_or_bit_idempotent() {
        // OR-set the same bit twice — must remain set (no toggle).
        let mut g = BitGrid3::zeros(1, 1, 64);
        g.or_bit(13, true);
        g.or_bit(13, true);
        assert!(g.get_bit(13));
        // exactly one bit set
        let popcount: u32 = g.words.iter().map(|w| w.count_ones()).sum();
        assert_eq!(popcount, 1);
    }

    #[test]
    fn test_truncate_v_drops_trailing_words() {
        // shape (10, 4, 2) = 80 bits → 2 u64 words
        let mut g = BitGrid3::zeros(10, 4, 2);
        // set a bit in the to-be-truncated tail
        g.or_bit(70, true);
        assert!(g.get_bit(70));
        g.truncate_v(5); // 5 * 4 * 2 = 40 bits → 1 word
        assert_eq!(g.shape, (5, 4, 2));
        assert_eq!(g.words.len(), 1);
        // the bit at flat=70 was in the dropped word — it's gone
        // (within the new shape the indexable range is 0..40)
    }

    #[test]
    #[should_panic(expected = "BitGrid3 shape overflow")]
    fn test_zeros_overflow_panics() {
        // V * S * P overflows usize on a 64-bit machine
        let _ = BitGrid3::zeros(usize::MAX, usize::MAX, 2);
    }

    #[test]
    fn test_popcount_plane_counts_set_bits_in_variant() {
        // shape (3, 2, 2): plane size = S*P = 4 bits per variant.
        let mut g = BitGrid3::zeros(3, 2, 2);
        // variant 0: set 2 bits (flat 0..4)
        g.or_bit(0, true);
        g.or_bit(3, true);
        // variant 1: set 4 bits (flat 4..8)
        for i in 4..8 {
            g.or_bit(i, true);
        }
        // variant 2: set 0 bits (flat 8..12)
        assert_eq!(g.popcount_plane(0), 2);
        assert_eq!(g.popcount_plane(1), 4);
        assert_eq!(g.popcount_plane(2), 0);
    }

    #[test]
    fn test_popcount_plane_word_aligned_boundaries() {
        // Exercise popcount_plane's `take == 64` branch (the dominant
        // real-world case: plane = S*P is usually >= 64 bits, and this
        // hand-written special case exists solely to avoid `1u64 << 64` UB)
        // together with the boundary shapes around it: sub-word (63),
        // exactly one word (64), one word + 1 bit (65, straddles a word
        // boundary), and exactly two words (128, two consecutive take==64
        // iterations).
        for &plane in &[63usize, 64, 65, 128] {
            let variants = 4;
            // shape (variants, plane, 1) so S*P == plane exactly, one plane
            // per variant.
            let mut g = BitGrid3::zeros(variants, plane, 1);

            // Deterministic, non-trivial pattern: bit `i` of variant `vi` is
            // set iff `(i * (vi + 1)) % 5 < 2`. Not all-zero/all-one, and it
            // hits bit 0, the last bit of the plane, and (for plane >= 64)
            // bits on both sides of every word boundary.
            for vi in 0..variants {
                for i in 0..plane {
                    let set = (i * (vi + 1)) % 5 < 2;
                    g.or_bit(vi * plane + i, set);
                }
            }

            for vi in 0..variants {
                // Ground truth: an independent naive per-bit count via
                // get_bit, not the masked-word logic under test.
                let expected = (0..plane).filter(|&i| g.get_bit(vi * plane + i)).count();
                assert_eq!(
                    g.popcount_plane(vi),
                    expected,
                    "plane={} bits, variant={}",
                    plane,
                    vi
                );
            }
        }
    }

    proptest! {
        // popcount_plane equals a naive per-bit count for arbitrary shapes/patterns.
        #[test]
        fn test_popcount_plane_matches_naive(
            v in 1usize..12,
            s in 1usize..8,
            p in 1usize..3,
            seed in any::<u64>(),
        ) {
            let mut bg = BitGrid3::zeros(v, s, p);
            let plane = s * p;
            let mut expected = vec![0usize; v];
            let mut state = seed | 1;
            for idx in 0..(v * plane) {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                let val = state & 1 != 0;
                bg.or_bit(idx, val);
                if val { expected[idx / plane] += 1; }
            }
            for vi in 0..v {
                prop_assert_eq!(bg.popcount_plane(vi), expected[vi], "variant {}", vi);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        // BitGrid3 stores bits losslessly: every set→get round-trip returns the
        // exact bool that was OR-set, regardless of layout, for arbitrary shapes.
        #[test]
        fn test_set_get_roundtrip(
            v in 1usize..16,
            s in 1usize..16,
            p in 1usize..4,
        ) {
            let mut bg = BitGrid3::zeros(v, s, p);
            // Use a deterministic checker pattern keyed off the flat index.
            let mut expected = vec![false; v * s * p];
            for vi in 0..v {
                for si in 0..s {
                    for pi in 0..p {
                        let idx = flat(vi, si, pi, s, p);
                        // arbitrary bit pattern
                        let val = (idx * 2654435761) & 1 != 0;
                        bg.or_bit(idx, val);
                        expected[idx] = val;
                    }
                }
            }
            for idx in 0..(v * s * p) {
                prop_assert_eq!(bg.get_bit(idx), expected[idx], "mismatch at flat idx {}", idx);
            }
        }

        // BitGrid3 must reproduce any access pattern that `Vec<bool>` would
        // produce for the same writes — this is the contract dense2sparse_vk relies on.
        #[test]
        fn test_matches_vec_bool(
            v in 1usize..12,
            s in 1usize..8,
            p in 1usize..3,
            seed in any::<u64>(),
        ) {
            let mut bg = BitGrid3::zeros(v, s, p);
            let mut vb = vec![false; v * s * p];

            // xorshift64 — deterministic per seed
            let mut state = seed | 1;
            for idx in 0..(v * s * p) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let val = state & 1 != 0;
                bg.or_bit(idx, val);
                vb[idx] = val;
            }
            for idx in 0..(v * s * p) {
                prop_assert_eq!(bg.get_bit(idx), vb[idx]);
            }
        }
    }
}

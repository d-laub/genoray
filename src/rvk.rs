use crate::cost_model::{Class, Representation, choose_representation};
use crate::dense::{DenseClass, DenseMap};
use crate::layout;
use crate::nrvk::LongAlleleTableWriter;
use crate::streams::StreamTag;
use crate::types::{DenseChunk, DenseSubChunk, MAX_INLINE_ALT_LEN, SparseChunk, SparseSubStream};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// SNP 2-bit key primitives and the indel key codec (encode/decode) now live in
// `svar2-codec`. Re-exported so existing `crate::rvk::<name>` call sites
// (query.rs, max_del.rs, tests) resolve.
pub use svar2_codec::{
    DecodedKey, decode_alt_inline, decode_key, decode_snp_2bit, deletion_len, encode_snp_2bit,
    pack_snp_keys, unpack_snp_key_at, unpack_snp_keys,
};

// Post-merge pass for the SNP stream: read the merged `alleles.bin` (one
// u8 code per call) and rewrite it 2-bit packed (4 calls/byte). Streams in
// 4-MiB blocks (a multiple of 4, so no pack straddles a block boundary except
// the genuine EOF tail). Offsets are call-indexed and need no change.
pub fn pack_snp_key_file(dir: &Path) {
    let src = layout::alleles(dir);
    let tmp = dir.join("alleles.packed.tmp");

    let mut reader = BufReader::new(File::open(&src).expect("open snp alleles.bin"));
    let mut writer = BufWriter::new(File::create(&tmp).expect("create packed tmp"));

    const BLOCK: usize = 4 * 1024 * 1024; // multiple of 4
    let mut buf = vec![0u8; BLOCK];
    loop {
        // Fill `buf` fully unless we hit EOF (so intermediate blocks stay a
        // multiple of 4 and pack cleanly byte-aligned).
        let mut filled = 0usize;
        while filled < BLOCK {
            match reader.read(&mut buf[filled..]).expect("read snp keys") {
                0 => break,
                n => filled += n,
            }
        }
        if filled == 0 {
            break;
        }
        let packed = pack_snp_keys(&buf[..filled]);
        writer.write_all(&packed).expect("write packed snp keys");
        if filled < BLOCK {
            break; // EOF tail handled
        }
    }
    writer.flush().expect("flush packed snp keys");
    drop(writer);
    drop(reader);

    std::fs::rename(&tmp, &src).expect("replace snp alleles.bin with packed");
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
        if alt_allele.len() <= MAX_INLINE_ALT_LEN {
            return svar2_codec::encode_alt_inline(alt_allele, ilen as u32);
        }
        let row_index = bank.push_long_allele(alt_allele);
        return svar2_codec::encode_lookup(row_index);
    }
    svar2_codec::encode_pure_del(ilen)
}

// The two inline flavors, tagged for stream routing (the encoding-seam output;
// see architecture.md#the-encoding-agnostic-seam).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKey {
    /// SNP: a bare 2-bit ALT code (0..=3) for the SNP stream.
    Snp(u8),
    /// Indel: the 32-bit key (inline value or LUT pointer) for the indel stream.
    Indel(u32),
}

// Classify one atomized variant into its stream + key.
//
// Under the atomization precondition, `ilen == 0` ⟺ SNP (ref_len == 1 &&
// alt_len == 1); `ilen != 0` ⟺ indel (INS ilen > 0, pure DEL ilen < 0). SNPs
// never spill; indels spill long insertions to the bank via `pack_variant`.
#[inline(always)]
pub fn classify_variant(ilen: i32, alt_allele: &[u8], bank: &mut LongAlleleTableWriter) -> VarKey {
    if ilen == 0 {
        debug_assert_eq!(
            alt_allele.len(),
            1,
            "ilen == 0 must be an atomized SNP (alt_len == 1)"
        );
        VarKey::Snp(encode_snp_2bit(alt_allele[0]))
    } else {
        VarKey::Indel(pack_variant(ilen, alt_allele, bank))
    }
}

// The core Dense-to-Sparse Matrix Transposer.
// Flips Row-Major VCF data into Column-Major (Sample-Major) Sparse Tensors.
pub fn dense2sparse_vk(chunk: &DenseChunk, bank: &mut LongAlleleTableWriter) -> SparseChunk {
    let (v_variants, num_samples, ploidy) = chunk.genos.shape;
    let columns = num_samples * ploidy;

    // Per-variant routing record built in one pre-pass.
    enum Route {
        VarKey(VarKey),
        // dense variant: which class + its column index within that class's table
        Dense { class: DenseClass, col: u32 },
    }

    // Dense per-class tables accumulate positions + keys as we discover dense
    // variants; geno_bits is sized after the pre-pass (n_dense known). This
    // also ensures a long insertion is pushed to the bank a single time, not
    // once per carrying haplotype.
    let mut dense = DenseMap::from_fn(|c| DenseSubChunk::empty(c.key_bytes()));
    let mut routes: Vec<Route> = Vec::with_capacity(v_variants);

    for v in 0..v_variants {
        let ilen = unsafe { *chunk.ilens.get_unchecked(v) };
        let start_idx = unsafe { *chunk.alt_offsets.get_unchecked(v) } as usize;
        let end_idx = unsafe { *chunk.alt_offsets.get_unchecked(v + 1) } as usize;
        let alt_allele = unsafe { chunk.alt.get_unchecked(start_idx..end_idx) };

        let vk = classify_variant(ilen, alt_allele, bank);
        let (class, dclass) = match vk {
            VarKey::Snp(_) => (Class::Snp, DenseClass::Snp),
            VarKey::Indel(_) => (Class::Indel, DenseClass::Indel),
        };
        let x = chunk.genos.popcount_plane(v);

        match choose_representation(class, num_samples, ploidy, x) {
            Representation::VarKey => routes.push(Route::VarKey(vk)),
            Representation::Dense => {
                let sub = dense.get_mut(dclass);
                let col = sub.n_dense_variants as u32;
                let pos = unsafe { *chunk.pos.get_unchecked(v) };
                sub.positions.push(pos);
                match vk {
                    VarKey::Snp(code) => sub.keys.push(code),
                    VarKey::Indel(key) => sub.keys.extend_from_slice(&key.to_le_bytes()),
                }
                sub.n_dense_variants += 1;
                routes.push(Route::Dense { class: dclass, col });
            }
        }
    }

    // Size each dense class's hap-major geno block now that n_dense is known.
    for (_c, sub) in dense.iter_mut() {
        let bits = columns * sub.n_dense_variants;
        sub.geno_bits = vec![0u8; bits.div_ceil(8)];
    }

    let estimated_nnz = (v_variants * columns) / 20;
    let mut streams = crate::streams::StreamMap::from_fn(|tag| {
        let spec = &crate::streams::REGISTRY[tag.index()];
        SparseSubStream::with_capacity(spec.key_bytes, estimated_nnz, columns)
    });

    let words: &[u64] = &chunk.genos.words;

    // Sample-major transpose; route each set call to its stream or dense bit.
    for s in 0..num_samples {
        for p in 0..ploidy {
            // per-tag running counts for this column
            let mut counts = crate::streams::StreamMap::from_fn(|_| 0u32);
            let base_idx = (s * ploidy) + p;
            let hap = base_idx; // hap index in sample-major order
            let stride = columns;

            for v in 0..v_variants {
                let flat_idx = (v * stride) + base_idx;
                let word = unsafe { *words.get_unchecked(flat_idx >> 6) };
                if (word >> (flat_idx & 63)) & 1 == 0 {
                    continue;
                }
                match unsafe { routes.get_unchecked(v) } {
                    Route::VarKey(vk) => {
                        let pos = unsafe { *chunk.pos.get_unchecked(v) };
                        let (tag, key_le): (StreamTag, [u8; 4]) = match vk {
                            VarKey::Snp(code) => (StreamTag::VarKeySnp, [*code, 0, 0, 0]),
                            VarKey::Indel(key) => (StreamTag::VarKeyIndel, key.to_le_bytes()),
                        };
                        let spec = &crate::streams::REGISTRY[tag.index()];
                        streams
                            .get_mut(tag)
                            .push_call(pos, &key_le[..spec.key_bytes]);
                        *counts.get_mut(tag) += 1;
                    }
                    Route::Dense { class, col } => {
                        let sub = dense.get_mut(*class);
                        let bit = hap * sub.n_dense_variants + (*col as usize);
                        crate::bits::set_bit(&mut sub.geno_bits, bit);
                    }
                }
            }
            for (tag, c) in counts.into_iter_tagged() {
                streams.get_mut(tag).sample_lengths.push(c);
            }
        }
    }

    SparseChunk {
        chunk_id: chunk.chunk_id,
        streams,
        dense,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BitGrid3, MIN_I31};
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
    fn test_decode_key_roundtrips_pack_variant() {
        let mut bank = make_bank();
        // SNP A->C: ilen 0, alt "C" inline.
        let snp = pack_variant(0, b"C", &mut bank);
        assert_eq!(decode_key(snp), DecodedKey::Inline { alt: b"C".to_vec() });
        assert_eq!(deletion_len(snp), 0);
        // INS A->AT: ilen 1, alt "AT" inline.
        let ins = pack_variant(1, b"AT", &mut bank);
        assert_eq!(
            decode_key(ins),
            DecodedKey::Inline {
                alt: b"AT".to_vec()
            }
        );
        assert_eq!(deletion_len(ins), 0);
        // Pure DEL of 2 bases: ilen -2, alt "A".
        let del = pack_variant(-2, b"A", &mut bank);
        assert_eq!(decode_key(del), DecodedKey::PureDel { ilen: -2 });
        assert_eq!(deletion_len(del), 2);
        // Long INS (> 13 bases) spills to the bank -> Lookup, deletion_len 0.
        let long = pack_variant(19, b"ACGTACGTACGTACGTACGT", &mut bank);
        assert!(matches!(decode_key(long), DecodedKey::Lookup { .. }));
        assert_eq!(deletion_len(long), 0);
    }

    #[test]
    fn test_deletion_len_near_min_i31() {
        let mut bank = make_bank();
        let big = pack_variant(MIN_I31, b"A", &mut bank);
        assert_eq!(deletion_len(big), (-(MIN_I31 as i64)) as u32);
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
    fn test_deletion_len_snp_and_ins_are_zero() {
        let mut bank = make_bank();
        // SNP (ilen 0) and INS (ilen > 0) carry no deletion.
        let snp = pack_variant(0, b"C", &mut bank);
        let ins = pack_variant(2, b"ACG", &mut bank);
        assert_eq!(deletion_len(snp), 0);
        assert_eq!(deletion_len(ins), 0);
    }

    #[test]
    fn test_deletion_len_small_and_large_del() {
        let mut bank = make_bank();
        // ref=ACGT alt=A → ilen=-3, deletion of 3 reference bases.
        let small = pack_variant(-3, b"A", &mut bank);
        assert_eq!(deletion_len(small), 3);
        // Largest atomizable deletion: ilen == MIN_I31 → d == 1 << 30.
        let large = pack_variant(crate::types::MIN_I31, b"A", &mut bank);
        assert_eq!(deletion_len(large), (1u32 << 30));
    }

    // Note: test_deletion_len_lookup_key_is_zero moved to svar2-codec — it is a
    // pure-decode edge case (raw keys, no bank), now codec-native.

    #[test]
    fn test_classify_variant_routes_by_ilen() {
        let mut bank = make_bank();

        // SNP: ilen == 0 → Snp(2-bit code). ref=A alt=C → code 1.
        match classify_variant(0, b"C", &mut bank) {
            VarKey::Snp(code) => assert_eq!(code, 1),
            other => panic!("expected Snp, got {:?}", other),
        }

        // INS: ilen > 0 → Indel(inline key), decodes back to the ALT.
        match classify_variant(2, b"ACG", &mut bank) {
            VarKey::Indel(key) => {
                assert_eq!(key & 1, 0, "inline INS clears lookup flag");
                assert_eq!(decode_alt_inline(key), b"ACG".to_vec());
            }
            other => panic!("expected Indel, got {:?}", other),
        }

        // Pure DEL: ilen < 0 → Indel(signed key).
        match classify_variant(-3, b"A", &mut bank) {
            VarKey::Indel(key) => assert_eq!((key as i32) >> 1, -3),
            other => panic!("expected Indel, got {:?}", other),
        }
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
        let snp = sparse.streams.get(StreamTag::VarKeySnp);
        let indel = sparse.streams.get(StreamTag::VarKeyIndel);
        assert_eq!(snp.call_positions.len(), 0);
        assert_eq!(snp.call_keys.len(), 0);
        assert_eq!(snp.sample_lengths, vec![0u32; 4]);
        assert_eq!(indel.call_positions.len(), 0);
        assert_eq!(indel.sample_lengths, vec![0u32; 4]);
    }

    #[test]
    fn test_dense2sparse_all_true_sample_major_layout() {
        // 6 variants x 2 samples x 2 ploidy (np=4). Each variant has exactly
        // ONE carrier (x=1), well under the dense crossover (34*1=34 <=
        // 34+np=38), so every variant stays var_key. hap0 carries the
        // even-indexed variants, hap1 the odd-indexed ones — exercising
        // multiple var_key calls per hap, across multiple haps, in v-order.
        let refs: Vec<&[u8]> = vec![b"A"; 6];
        let alts: Vec<&[u8]> = vec![b"C", b"G", b"T", b"C", b"G", b"T"];
        let n_variants = 6;
        let n_samples = 2;
        let ploidy = 2;
        let columns = n_samples * ploidy;
        let mut bit_pattern = vec![false; n_variants * columns];
        for v in 0..n_variants {
            let hap = if v % 2 == 0 { 0 } else { 1 };
            bit_pattern[v * columns + hap] = true;
        }

        let chunk = build_test_chunk(n_variants, n_samples, ploidy, &refs, &alts, &bit_pattern);
        let mut bank = make_bank();
        let sparse = dense2sparse_vk(&chunk, &mut bank);
        let snp = sparse.streams.get(StreamTag::VarKeySnp);
        let indel = sparse.streams.get(StreamTag::VarKeyIndel);

        assert_eq!(snp.call_positions.len(), 6);
        assert_eq!(snp.sample_lengths, vec![3, 3, 0, 0]);
        // hap0's slice is v-ordered [v0, v2, v4] = [100, 120, 140]
        assert_eq!(&snp.call_positions[0..3], &[100, 120, 140]);
        // hap1's slice is v-ordered [v1, v3, v5] = [110, 130, 150]
        assert_eq!(&snp.call_positions[3..6], &[110, 130, 150]);
        assert_eq!(indel.call_positions.len(), 0);
        // No variant meets the dense crossover with a single carrier.
        assert_eq!(sparse.dense.get(DenseClass::Snp).n_dense_variants, 0);
    }

    // A mixed chunk: variant 0 is a SNP (A→C), variant 1 is an INS (A→AT),
    // variant 2 is a pure DEL (AT→A). Each variant has a SINGLE carrier
    // (x=1), which always stays var_key regardless of cohort size
    // (var_key_bits = key_bits+32 <= dense_bits = key_bits+32+np for np>=0).
    // hap0 carries the SNP + INS, hap1 carries the DEL, so a hap sees both
    // streams and calls stay v-ordered. The SNP must land in `snp`, the
    // INS/DEL in `indel`.
    #[test]
    fn test_dense2sparse_splits_snp_and_indel() {
        let refs: Vec<&[u8]> = vec![b"A", b"A", b"AT"];
        let alts: Vec<&[u8]> = vec![b"C", b"AT", b"A"];
        // (variant, hap): v0(SNP)->hap0, v1(INS)->hap0, v2(DEL)->hap1
        #[rustfmt::skip]
        let bit_pattern = vec![
            true, false, // v0: hap0 carries
            true, false, // v1: hap0 carries
            false, true, // v2: hap1 carries
        ];
        let chunk = build_test_chunk(3, 1, 2, &refs, &alts, &bit_pattern);

        let mut bank = make_bank();
        let sparse = dense2sparse_vk(&chunk, &mut bank);
        let snp = sparse.streams.get(StreamTag::VarKeySnp);
        let indel = sparse.streams.get(StreamTag::VarKeyIndel);

        // hap0: 1 SNP call + 1 indel call (INS); hap1: 1 indel call (DEL).
        assert_eq!(snp.sample_lengths, vec![1, 0]);
        assert_eq!(indel.sample_lengths, vec![1, 1]);

        // SNP stream: position 100 (variant 0), code for 'C' == 1.
        assert_eq!(snp.call_positions, vec![100]);
        assert_eq!(snp.call_keys, vec![1u8]);

        // Indel stream: hap0's INS@110 then hap1's DEL@120, keys decode back.
        assert_eq!(indel.call_positions, vec![110, 120]);
        let decode_key =
            |i: usize| u32::from_le_bytes(indel.call_keys[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(decode_alt_inline(decode_key(0)), b"AT".to_vec());
        assert_eq!((decode_key(1) as i32) >> 1, -1); // DEL ilen = -1

        // Every variant has a single carrier → all stayed var_key.
        assert_eq!(sparse.dense.get(DenseClass::Snp).n_dense_variants, 0);
        assert_eq!(sparse.dense.get(DenseClass::Indel).n_dense_variants, 0);
    }

    // A SNP carried by every hap in a small cohort routes to DENSE, leaving the
    // var_key snp stream empty; its bit lands in the dense geno block.
    #[test]
    fn test_common_snp_routes_to_dense() {
        // n=2 diploid, np=4. One SNP A→C, all 4 haps carry it.
        // dense = 32+2+4 = 38 bits; var_key = 4*(32+2)=136 → dense wins.
        let refs: Vec<&[u8]> = vec![b"A"];
        let alts: Vec<&[u8]> = vec![b"C"];
        let bits = vec![true; 2 * 2]; // n_variants=1 * n_samples=2 * ploidy=2
        let chunk = build_test_chunk(1, 2, 2, &refs, &alts, &bits);

        let mut bank = make_bank();
        let sparse = dense2sparse_vk(&chunk, &mut bank);

        // var_key snp stream is empty (variant went dense)
        let snp = sparse.streams.get(StreamTag::VarKeySnp);
        assert_eq!(snp.call_positions.len(), 0);

        // dense snp table has 1 variant, all 4 haps' bits set
        let d = sparse.dense.get(DenseClass::Snp);
        assert_eq!(d.n_dense_variants, 1);
        assert_eq!(d.positions, vec![100]);
        assert_eq!(d.keys, vec![1u8]); // 'C' code == 1
        // np=4 haps, v_dense=1 → 4 bits, one per hap at flat idx h*1+0
        for h in 0..4 {
            assert!(crate::bits::get_bit(&d.geno_bits, h), "hap {} bit", h);
        }
    }

    // A rare SNP (single carrier) stays var_key; dense table empty.
    #[test]
    fn test_rare_snp_stays_var_key() {
        // n=100 diploid np=200. One SNP, single carrier → var_key wins
        // (34 < dense 32+2+200=234).
        let n = 100;
        let refs: Vec<&[u8]> = vec![b"A"];
        let alts: Vec<&[u8]> = vec![b"C"];
        let mut bits = vec![false; n * 2]; // n_variants=1 * n_samples=n * ploidy=2
        bits[0] = true; // one hap carries it
        let chunk = build_test_chunk(1, n, 2, &refs, &alts, &bits);

        let mut bank = make_bank();
        let sparse = dense2sparse_vk(&chunk, &mut bank);
        assert_eq!(
            sparse
                .streams
                .get(StreamTag::VarKeySnp)
                .call_positions
                .len(),
            1
        );
        assert_eq!(sparse.dense.get(DenseClass::Snp).n_dense_variants, 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1500))]

        // Mutation conservation: every true bit in the dense grid produces exactly
        // one sparse call, either var_key or dense. Total calls == popcount(genos).
        // Catches lost mutations and double-counting, across ALL buckets.
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

            // SNPs only — keeps everything in the inline lane. Some may still
            // route to dense under the cost model, so conservation must be
            // checked across both var_key and dense buckets.
            let refs: Vec<&[u8]> = vec![&b"A"[..]; n_variants];
            let alts: Vec<&[u8]> = vec![&b"C"[..]; n_variants];
            let chunk = build_test_chunk(n_variants, n_samples, ploidy, &refs, &alts, &bit_pattern);

            let mut bank = make_bank();
            let sparse = dense2sparse_vk(&chunk, &mut bank);
            let snp = sparse.streams.get(StreamTag::VarKeySnp);
            let indel = sparse.streams.get(StreamTag::VarKeyIndel);

            let vk_calls: usize = snp.sample_lengths.iter().map(|&c| c as usize).sum::<usize>()
                + indel.sample_lengths.iter().map(|&c| c as usize).sum::<usize>();
            let dense_calls: usize = sparse
                .dense
                .iter()
                .map(|(_c, sub)| sub.geno_bits.iter().map(|b| b.count_ones() as usize).sum::<usize>())
                .sum();

            prop_assert_eq!(vk_calls + dense_calls, true_count, "lost or double-counted a call");
            prop_assert_eq!(snp.sample_lengths.len(), n_samples * ploidy);
            prop_assert_eq!(indel.sample_lengths.len(), n_samples * ploidy);
        }

        // Per-sample correctness: for each (s,p) hap, the slice of call_positions
        // identified by sample_lengths must exactly match the variants where
        // that hap's bit is set AND the variant stayed var_key, in v-order.
        // (Variants routed to dense are excluded from the var_key stream —
        // this replicates the routing decision from the bit pattern itself.)
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
            let snp = sparse.streams.get(StreamTag::VarKeySnp);

            let np = n_samples * ploidy;
            // Replicate the routing decision per variant: SNP stays var_key
            // iff 34*x <= 34+np (see cost_model::choose_representation).
            let stays_var_key: Vec<bool> = (0..n_variants)
                .map(|v| {
                    let x: usize = (0..np).filter(|&c| bit_pattern[v * np + c]).count();
                    34 * x <= 34 + np
                })
                .collect();

            let mut cursor = 0usize;
            for s in 0..n_samples {
                for p in 0..ploidy {
                    let hap_idx = s * ploidy + p;
                    let calls = snp.sample_lengths[hap_idx] as usize;

                    // Compute expected positions from the bit pattern + chunk.pos,
                    // filtered to variants that stayed var_key.
                    let expected: Vec<u32> = (0..n_variants).filter_map(|v| {
                        let flat_idx = v * n_samples * ploidy + s * ploidy + p;
                        if bit_pattern[flat_idx] && stays_var_key[v] { Some(chunk.pos[v]) } else { None }
                    }).collect();

                    let actual: Vec<u32> = snp.call_positions[cursor..cursor + calls].to_vec();
                    prop_assert_eq!(&actual, &expected, "hap {} positions mismatch", hap_idx);
                    cursor += calls;
                }
            }
            prop_assert_eq!(cursor, snp.call_positions.len());
        }

        // Conservation across routing: every set genotype bit is stored exactly
        // once — either as a var_key call or a dense set bit.
        #[test]
        fn test_routing_conserves_all_calls(
            n_variants in 1usize..20,
            n_samples in 1usize..6,
            ploidy in 1usize..3,
            seed in any::<u64>(),
        ) {
            let bits = random_bits(n_variants * n_samples * ploidy, seed);
            let true_count = bits.iter().filter(|&&b| b).count();
            let refs: Vec<&[u8]> = vec![&b"A"[..]; n_variants];
            let alts: Vec<&[u8]> = vec![&b"C"[..]; n_variants]; // all SNPs
            let chunk = build_test_chunk(n_variants, n_samples, ploidy, &refs, &alts, &bits);

            let mut bank = make_bank();
            let sparse = dense2sparse_vk(&chunk, &mut bank);

            let vk_calls: usize = sparse.streams.get(StreamTag::VarKeySnp)
                .sample_lengths.iter().map(|&c| c as usize).sum::<usize>()
                + sparse.streams.get(StreamTag::VarKeyIndel)
                .sample_lengths.iter().map(|&c| c as usize).sum::<usize>();

            let mut dense_calls = 0usize;
            for (_c, sub) in sparse.dense.iter() {
                dense_calls += sub.geno_bits.iter().map(|b| b.count_ones() as usize).sum::<usize>();
            }
            prop_assert_eq!(vk_calls + dense_calls, true_count, "lost or double-counted a call");
        }
    }
}

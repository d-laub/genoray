//! Rectangular dense merge: concatenate per-chunk dense variant tables and
//! bit-transpose per-chunk hap-major geno blocks into one (S, P, V_dense)
//! matrix. Unlike the ragged var_key tile merge (merge.rs), every hap
//! contributes the SAME per-chunk count, so offsets are uniform — the only
//! non-trivial step is the per-hap bit concatenation across chunks.

use crate::bits::copy_bits;
use crate::layout;
use crate::rvk::pack_snp_keys;
use std::fs;
use std::io::Write;
use std::path::Path;

pub fn merge_dense_class(
    num_chunks: usize,
    num_samples: usize,
    ploidy: usize,
    // Indel key width is intrinsic to raw (unpacked) key bytes on disk today;
    // reserved for a future variable-width indel key encoding.
    _key_bytes: usize,
    pack_snp: bool,
    output_dir: &str,
    dense_ledger: Vec<u32>,
) {
    debug_assert_eq!(
        dense_ledger.len(),
        num_chunks,
        "dense_ledger must have exactly one row per chunk"
    );
    let dir = Path::new(output_dir);
    let np = num_samples * ploidy;
    let v_total: usize = dense_ledger.iter().map(|&c| c as usize).sum();

    // ---- positions + keys: sequential concat in chunk order ----
    let mut positions: Vec<u8> = Vec::new();
    let mut keys: Vec<u8> = Vec::new();
    for (c, &count) in dense_ledger.iter().enumerate().take(num_chunks) {
        if count == 0 {
            continue;
        }
        positions.extend_from_slice(&fs::read(layout::chunk_pos(dir, c)).expect("read dense pos"));
        keys.extend_from_slice(&fs::read(layout::chunk_key(dir, c)).expect("read dense key"));
    }
    write_all(&layout::positions(dir), &positions);
    let final_key_bytes = if pack_snp {
        pack_snp_keys(&keys) // keys are one raw 2-bit code per variant
    } else {
        keys
    };
    write_all(&layout::alleles(dir), &final_key_bytes);

    // ---- genotypes: per-hap bit concatenation across chunks ----
    // output bit (hap h, global col g) at flat index h * v_total + g.
    let out_bits_len = (np * v_total).div_ceil(8);
    let mut out = vec![0u8; out_bits_len];

    // prefix sum of dense variants per chunk = global column offset per chunk
    let mut col_prefix = vec![0usize; num_chunks + 1];
    for c in 0..num_chunks {
        col_prefix[c + 1] = col_prefix[c] + dense_ledger[c] as usize;
    }

    for c in 0..num_chunks {
        let v_c = dense_ledger[c] as usize;
        if v_c == 0 {
            continue;
        }
        let block = fs::read(layout::chunk_geno(dir, c)).expect("read dense geno");
        // block bit (hap h, local col d) at h*v_c + d.
        for h in 0..np {
            let src_bit = h * v_c;
            let dst_bit = h * v_total + col_prefix[c];
            copy_bits(&mut out, dst_bit, &block, src_bit, v_c);
        }
    }
    write_all(&layout::genotypes(dir), &out);

    // ---- cleanup per-chunk temp files ----
    for c in 0..num_chunks {
        let _ = fs::remove_file(layout::chunk_pos(dir, c));
        let _ = fs::remove_file(layout::chunk_key(dir, c));
        let _ = fs::remove_file(layout::chunk_geno(dir, c));
    }
}

fn write_all(path: &Path, bytes: &[u8]) {
    let mut f =
        fs::File::create(path).unwrap_or_else(|e| panic!("create {}: {}", path.display(), e));
    f.write_all(bytes).expect("write dense final");
    f.flush().expect("flush dense final");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{get_bit, set_bit};
    use tempfile::tempdir;

    // Build a hap-major block (np rows × v_c cols) from a bool matrix
    // indexed [hap][col], and stage it as chunk `c`'s geno + pos + key files.
    fn stage_chunk(dir: &Path, c: usize, positions: &[u32], keys: &[u8], mat: &[Vec<bool>]) {
        let np = mat.len();
        let v_c = if np > 0 { mat[0].len() } else { 0 };
        let mut block = vec![0u8; (np * v_c).div_ceil(8)];
        for (h, row) in mat.iter().enumerate().take(np) {
            for (d, &bit) in row.iter().enumerate().take(v_c) {
                if bit {
                    set_bit(&mut block, h * v_c + d);
                }
            }
        }
        write_all(&layout::chunk_pos(dir, c), bytemuck::cast_slice(positions));
        write_all(&layout::chunk_key(dir, c), keys);
        write_all(&layout::chunk_geno(dir, c), &block);
    }

    fn read_u32(path: &Path) -> Vec<u32> {
        let b = fs::read(path).unwrap();
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn test_merge_dense_two_chunks_transpose() {
        // np=2 haps. chunk0: 2 variants, chunk1: 1 variant → v_total=3.
        // hap0: [1,0 | 1] ; hap1: [0,1 | 0]  (col order = chunk0 cols then chunk1)
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        stage_chunk(
            dir,
            0,
            &[100, 200],
            &[1u8, 2u8],
            &[vec![true, false], vec![false, true]],
        );
        stage_chunk(dir, 1, &[300], &[3u8], &[vec![true], vec![false]]);

        merge_dense_class(
            2,
            1,
            2,
            1,
            /*pack_snp=*/ false,
            dir.to_str().unwrap(),
            vec![2, 1],
        );

        // positions concat in chunk order
        assert_eq!(read_u32(&layout::positions(dir)), vec![100, 200, 300]);
        // keys concat (pack_snp=false → raw)
        assert_eq!(fs::read(layout::alleles(dir)).unwrap(), vec![1u8, 2, 3]);

        // genotypes: hap0 row = [1,0,1], hap1 row = [0,1,0], flat h*v_total+g.
        let geno = fs::read(layout::genotypes(dir)).unwrap();
        let expect_bits = [
            (0usize, true),
            (1, false),
            (2, true), // hap0 cols 0,1,2
            (3, false),
            (4, true),
            (5, false), // hap1 cols 0,1,2
        ];
        for (idx, want) in expect_bits {
            assert_eq!(get_bit(&geno, idx), want, "geno bit {}", idx);
        }
        // temp files removed
        assert!(!layout::chunk_geno(dir, 0).exists());
    }

    #[test]
    fn test_merge_dense_empty() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        merge_dense_class(1, 2, 2, 1, true, dir.to_str().unwrap(), vec![0]);
        assert_eq!(fs::read(layout::positions(dir)).unwrap().len(), 0);
        assert_eq!(fs::read(layout::genotypes(dir)).unwrap().len(), 0);
    }

    #[test]
    fn test_merge_dense_snp_packs_keys() {
        // pack_snp=true: 5 raw codes → packed into ceil(5/4)=2 bytes.
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        // single chunk, np=1, 5 dense variants, one hap all-set.
        stage_chunk(
            dir,
            0,
            &[1, 2, 3, 4, 5],
            &[1u8, 2, 3, 0, 1],
            &[vec![true, true, true, true, true]],
        );
        merge_dense_class(1, 1, 1, 1, true, dir.to_str().unwrap(), vec![5]);
        // pack_snp_keys([1,2,3,0,1]) == [0x39, 0x01] (see rvk.rs test)
        assert_eq!(fs::read(layout::alleles(dir)).unwrap(), vec![0x39u8, 0x01]);
    }
}

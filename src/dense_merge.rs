//! Rectangular dense merge: concatenate per-chunk dense variant tables and
//! bit-transpose per-chunk hap-major geno blocks into one (S, P, V_dense)
//! matrix. Unlike the ragged var_key tile merge (merge.rs), every hap
//! contributes the SAME per-chunk count, so offsets are uniform — the only
//! non-trivial step is the per-hap bit concatenation across chunks.

use crate::bits::copy_bits;
use crate::error::ConversionError;
use crate::layout;
use crate::rvk::pack_snp_keys;
use memmap2::Mmap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Greatest common divisor, for the hap-group alignment in
/// [`merge_dense_class`].
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Everything [`merge_dense_class`] needs besides the ledger itself.
pub struct DenseMergeParams<'a> {
    pub num_chunks: usize,
    pub num_samples: usize,
    pub ploidy: usize,
    /// Indel key width is intrinsic to raw (unpacked) key bytes on disk today;
    /// reserved for a future variable-width indel key encoding.
    pub key_bytes: usize,
    pub pack_snp: bool,
    pub output_dir: &'a str,
    /// Thread budget for the genotype transpose. Passed in rather than taken
    /// from the ambient rayon pool ON PURPOSE: `process_chromosome` runs inside
    /// `lib.rs`'s dispatch pool, which is sized to `concurrent_chroms` -- 1
    /// today. Any `rayon::current_num_threads()` or `par_iter()` reached from
    /// here therefore sees a ONE-thread pool and silently runs serially, which
    /// is exactly the trap this field exists to avoid.
    pub threads: usize,
}

pub fn merge_dense_class(
    params: DenseMergeParams<'_>,
    dense_ledger: Vec<u32>,
) -> Result<(), ConversionError> {
    let DenseMergeParams {
        num_chunks,
        num_samples,
        ploidy,
        key_bytes: _,
        pack_snp,
        output_dir,
        threads,
    } = params;
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
        let pos_path = layout::chunk_pos(dir, c);
        positions.extend_from_slice(&fs::read(&pos_path).map_err(|e| ConversionError::Io {
            context: format!("reading {}", pos_path.display()),
            source: e,
        })?);
        let key_path = layout::chunk_key(dir, c);
        keys.extend_from_slice(&fs::read(&key_path).map_err(|e| ConversionError::Io {
            context: format!("reading {}", key_path.display()),
            source: e,
        })?);
    }
    write_all(&layout::positions(dir), &positions)?;
    let final_key_bytes = if pack_snp {
        pack_snp_keys(&keys) // keys are one raw 2-bit code per variant
    } else {
        keys
    };
    write_all(&layout::alleles(dir), &final_key_bytes)?;

    // ---- genotypes: per-hap bit concatenation across chunks ----
    // output bit (hap h, global col g) at flat index h * v_total + g.
    let out_bits_len = (np * v_total).div_ceil(8);
    let mut out = vec![0u8; out_bits_len];

    // prefix sum of dense variants per chunk = global column offset per chunk
    let mut col_prefix = vec![0usize; num_chunks + 1];
    for c in 0..num_chunks {
        col_prefix[c + 1] = col_prefix[c] + dense_ledger[c] as usize;
    }

    if v_total > 0 && np > 0 {
        // Map every non-empty chunk block up front so the transpose can run
        // HAP-major. Chunk-major (the shape this loop used to have) makes each
        // chunk sweep the entire output: consecutive haps land `v_total` bits
        // apart, so a single chunk touches every page of a matrix that is
        // hundreds of MB at cohort scale, and does it once per chunk. Hap-major
        // keeps each worker inside one contiguous output region instead.
        //
        // `Mmap` rather than `fs::read` so this costs page cache -- clean,
        // file-backed, evictable under pressure -- instead of adding a second
        // heap copy of the whole matrix alongside `out`. These files were
        // written moments ago by this same process, so they are already cached.
        let blocks: Vec<Option<Mmap>> = (0..num_chunks)
            .map(|c| {
                if dense_ledger[c] == 0 {
                    return Ok(None);
                }
                let geno_path = layout::chunk_geno(dir, c);
                let f = fs::File::open(&geno_path).map_err(|e| ConversionError::Io {
                    context: format!("opening {}", geno_path.display()),
                    source: e,
                })?;
                // SAFETY: these per-chunk files are private to this conversion
                // and are not modified (or removed) until the cleanup below,
                // which runs after every map is dropped.
                let m = unsafe { Mmap::map(&f) }.map_err(|e| ConversionError::Io {
                    context: format!("mmapping {}", geno_path.display()),
                    source: e,
                })?;
                Ok(Some(m))
            })
            .collect::<Result<Vec<_>, ConversionError>>()?;

        // A worker can only be handed WHOLE bytes of `out`, so hap-group
        // boundaries have to be byte boundaries. Hap `h` starts at bit
        // `h * v_total`, which is byte-aligned exactly when `h` is a multiple
        // of `align_haps` -- at most 8, so this never meaningfully constrains
        // group sizing. Aligning this way is what lets `chunks_mut` hand out
        // provably disjoint slices: no unsafe aliasing, and no second pass to
        // OR together overlapping boundary bytes.
        let align_haps = 8 / gcd(v_total, 8);
        let group_haps = np
            .div_ceil(threads.max(1))
            .next_multiple_of(align_haps)
            .max(align_haps);
        // Exact: `group_haps` is a multiple of `align_haps = 8/gcd(v_total,8)`,
        // so `group_haps * v_total` is a multiple of 8.
        let bytes_per_group = group_haps * v_total / 8;

        let ledger = &dense_ledger;
        let prefix = &col_prefix;
        let maps = &blocks;
        std::thread::scope(|scope| {
            for (g, slab) in out.chunks_mut(bytes_per_group).enumerate() {
                scope.spawn(move || {
                    let h0 = g * group_haps;
                    let h1 = (h0 + group_haps).min(np);
                    for h in h0..h1 {
                        let dst_base = (h - h0) * v_total;
                        for (c, block) in maps.iter().enumerate() {
                            let Some(block) = block else { continue };
                            let v_c = ledger[c] as usize;
                            // block bit (hap h, local col d) at h*v_c + d.
                            copy_bits(slab, dst_base + prefix[c], block, h * v_c, v_c);
                        }
                    }
                });
            }
        });

        // Unmap before the cleanup below unlinks the files these map.
        drop(blocks);
    }
    write_all(&layout::genotypes(dir), &out)?;

    // ---- cleanup per-chunk temp files ----
    for c in 0..num_chunks {
        let _ = fs::remove_file(layout::chunk_pos(dir, c));
        let _ = fs::remove_file(layout::chunk_key(dir, c));
        let _ = fs::remove_file(layout::chunk_geno(dir, c));
    }
    Ok(())
}

/// Concatenate one dense field's per-chunk `chunk_{c}_{finfo|fformat}{field_ix}.bin`
/// files, in chunk order, into `dest_values_bin`.
///
/// Dense field values are staged 1:1 with dense variants (no ragged ledger, no
/// transpose — an INFO value is one value per dense variant; a FORMAT value is
/// `n_dense_variants[c] * num_samples` values, variant-major), so this mirrors
/// `merge_dense_class`'s positions/keys concat exactly: a pure chunk-order byte
/// concatenation reproduces the final layout. Chunks with `dense_ledger[c] == 0`
/// wrote no per-chunk field file (Task 7) and are skipped here too.
///
/// `category` selects which per-chunk file family to read: `FieldCategory::Info`
/// for INFO fields (`layout::chunk_field_info`) or `FieldCategory::Format` for
/// FORMAT fields (`layout::chunk_field_format`). `field_ix` is the per-category
/// (INFO-only or FORMAT-only) field index Task 7 staged under.
///
/// The caller is responsible for creating `dest_values_bin`'s parent directory
/// before calling this function (this function does not call `create_dir_all`),
/// mirroring `merge::merge_var_key_field_values`'s contract.
///
/// On success, the consumed per-chunk field files are removed.
pub fn merge_dense_field_values(
    output_dir: &str,
    num_chunks: usize,
    dense_ledger: &[u32],
    category: crate::field::FieldCategory,
    field_ix: usize,
    dest_values_bin: &Path,
) -> Result<(), ConversionError> {
    debug_assert_eq!(
        dense_ledger.len(),
        num_chunks,
        "dense_ledger must have exactly one row per chunk"
    );
    let dir = Path::new(output_dir);
    let mut values: Vec<u8> = Vec::new();
    let mut consumed: Vec<PathBuf> = Vec::new();
    for (c, &count) in dense_ledger.iter().enumerate().take(num_chunks) {
        if count == 0 {
            continue;
        }
        let path = match category {
            crate::field::FieldCategory::Info => layout::chunk_field_info(dir, c, field_ix),
            crate::field::FieldCategory::Format => layout::chunk_field_format(dir, c, field_ix),
        };
        values.extend_from_slice(&fs::read(&path).map_err(|e| ConversionError::Io {
            context: format!("reading {}", path.display()),
            source: e,
        })?);
        consumed.push(path);
    }
    write_all(dest_values_bin, &values)?;
    for path in consumed {
        let _ = fs::remove_file(path);
    }
    Ok(())
}

fn write_all(path: &Path, bytes: &[u8]) -> Result<(), ConversionError> {
    let mut f = fs::File::create(path).map_err(|e| ConversionError::Io {
        context: format!("creating {}", path.display()),
        source: e,
    })?;
    f.write_all(bytes).map_err(|e| ConversionError::Io {
        context: format!("writing {}", path.display()),
        source: e,
    })?;
    f.flush().map_err(|e| ConversionError::Io {
        context: format!("flushing {}", path.display()),
        source: e,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{get_bit, set_bit};
    use crate::field::FieldCategory;
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
        write_all(&layout::chunk_pos(dir, c), bytemuck::cast_slice(positions)).unwrap();
        write_all(&layout::chunk_key(dir, c), keys).unwrap();
        write_all(&layout::chunk_geno(dir, c), &block).unwrap();
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
            DenseMergeParams {
                num_chunks: 2,
                num_samples: 1,
                ploidy: 2,
                key_bytes: 1,
                pack_snp: false,
                output_dir: dir.to_str().unwrap(),
                threads: 1,
            },
            vec![2, 1],
        )
        .unwrap();

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

    /// The transpose hands each rayon worker a disjoint byte range of the
    /// output, which is only sound because hap-group boundaries are chosen to
    /// land on byte boundaries. A `v_total` coprime to 8 is the case that
    /// stresses it: every hap row then starts at a different bit offset within
    /// its byte, so an off-by-one in the alignment math corrupts bits at the
    /// seam between two workers -- invisible for `v_total % 8 == 0`, and
    /// invisible whenever the whole matrix fits in one group.
    #[test]
    fn test_merge_dense_transpose_across_worker_boundaries() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        // v_total = 5+7+3 = 15, coprime to 8. np = 96 haps forces several
        // groups on any plausible rayon pool size.
        let widths = [5usize, 7, 3];
        let np = 96usize;
        let v_total: usize = widths.iter().sum();

        // Deterministic pseudo-random fill; `want[h][g]` is the expected
        // output bit for hap `h`, global column `g`.
        let bit_of = |h: usize, g: usize| (h * 31 + g * 17 + h * g).is_multiple_of(3);

        let mut base = 0usize;
        for (c, &v_c) in widths.iter().enumerate() {
            let mat: Vec<Vec<bool>> = (0..np)
                .map(|h| (0..v_c).map(|d| bit_of(h, base + d)).collect())
                .collect();
            let positions: Vec<u32> = (0..v_c as u32).map(|d| base as u32 + d).collect();
            let keys = vec![0u8; v_c];
            stage_chunk(dir, c, &positions, &keys, &mat);
            base += v_c;
        }

        let ledger: Vec<u32> = widths.iter().map(|&v| v as u32).collect();
        merge_dense_class(
            DenseMergeParams {
                num_chunks: widths.len(),
                num_samples: np / 2,
                ploidy: 2, // -> np haps
                key_bytes: 1,
                pack_snp: false,
                output_dir: dir.to_str().unwrap(),
                // Several worker groups, so the byte-aligned seams are exercised.
                threads: 8,
            },
            ledger,
        )
        .unwrap();

        let geno = fs::read(layout::genotypes(dir)).unwrap();
        assert_eq!(geno.len(), (np * v_total).div_ceil(8));
        for h in 0..np {
            for g in 0..v_total {
                assert_eq!(
                    get_bit(&geno, h * v_total + g),
                    bit_of(h, g),
                    "hap {h} col {g}"
                );
            }
        }
    }

    #[test]
    fn test_merge_dense_empty() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        merge_dense_class(
            DenseMergeParams {
                num_chunks: 1,
                num_samples: 2,
                ploidy: 2,
                key_bytes: 1,
                pack_snp: true,
                output_dir: dir.to_str().unwrap(),
                threads: 4,
            },
            vec![0],
        )
        .unwrap();
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
        merge_dense_class(
            DenseMergeParams {
                num_chunks: 1,
                num_samples: 1,
                ploidy: 1,
                key_bytes: 1,
                pack_snp: true,
                output_dir: dir.to_str().unwrap(),
                threads: 4,
            },
            vec![5],
        )
        .unwrap();
        // pack_snp_keys([1,2,3,0,1]) == [0x39, 0x01] (see rvk.rs test)
        assert_eq!(fs::read(layout::alleles(dir)).unwrap(), vec![0x39u8, 0x01]);
    }

    fn read_i32(path: &Path) -> Vec<i32> {
        let bytes = fs::read(path).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = fs::read(path).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn test_merge_dense_field_values_finfo_skips_empty_chunk() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        // chunk0: 2 dense variants -> finfo0 = [10, 20] (i32)
        write_all(
            &layout::chunk_field_info(dir, 0, 0),
            bytemuck::cast_slice(&[10i32, 20]),
        )
        .unwrap();
        // chunk1: 1 dense variant -> finfo0 = [30] (i32)
        write_all(
            &layout::chunk_field_info(dir, 1, 0),
            bytemuck::cast_slice(&[30i32]),
        )
        .unwrap();
        // chunk2: 0 dense variants -> Task 7 wrote NO finfo file for it.
        let dense_ledger = vec![2u32, 1, 0];
        let dest = dir
            .join("fields")
            .join("DP")
            .join("dense_snp")
            .join("values.bin");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();

        merge_dense_field_values(
            dir.to_str().unwrap(),
            3,
            &dense_ledger,
            FieldCategory::Info,
            0,
            &dest,
        )
        .unwrap();

        assert_eq!(read_i32(&dest), vec![10, 20, 30]);
        // Consumed per-chunk field files are removed.
        assert!(!layout::chunk_field_info(dir, 0, 0).exists());
        assert!(!layout::chunk_field_info(dir, 1, 0).exists());
    }

    #[test]
    fn test_merge_dense_field_values_fformat_concat() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path();
        // num_samples=2, variant-major. chunk0: 1 dense variant -> 2 values.
        write_all(
            &layout::chunk_field_format(dir, 0, 1),
            bytemuck::cast_slice(&[1.0f32, 2.0]),
        )
        .unwrap();
        // chunk1: 2 dense variants -> 4 values.
        write_all(
            &layout::chunk_field_format(dir, 1, 1),
            bytemuck::cast_slice(&[3.0f32, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        let dense_ledger = vec![1u32, 2];
        let dest = dir
            .join("fields")
            .join("DS")
            .join("dense_indel")
            .join("values.bin");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();

        merge_dense_field_values(
            dir.to_str().unwrap(),
            2,
            &dense_ledger,
            FieldCategory::Format,
            1,
            &dest,
        )
        .unwrap();

        assert_eq!(read_f32(&dest), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(!layout::chunk_field_format(dir, 0, 1).exists());
        assert!(!layout::chunk_field_format(dir, 1, 1).exists());
    }
}

//! Standalone `max_del` post-pass (SVAR 2.0, M5 part 2a).
//!
//! Scans a finished contig's indel key streams and emits the max-deletion-length
//! artifacts consumed by the `(range, sample)` overlap query. A deletion's length
//! is recoverable from the inline pure-DEL key alone (see [`crate::rvk::deletion_len`]),
//! so this is a pure scan of `alleles.bin` — no LUT reads, no reference genome, no
//! coupling to the conversion/merge write path. Runs after a contig's merge
//! completes; callable per-contig or as a batch sweep over an existing directory.

use crate::error::ConversionError;
use crate::layout;
use crate::rvk::deletion_len;
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use std::path::Path;

/// Emit `{contig_dir}/max_del.npy` (shape `(n_samples, ploidy)`, `u32`) and
/// `{contig_dir}/dense/max_del.npy` (shape `(1,)`, `u32`). Both are always written
/// — a contig with no deletions emits all-zero artifacts so the consumer never
/// special-cases a missing file. `O(total indel calls)`; a single serial pass
/// (measure before parallelizing — this is I/O-bound).
pub fn write_max_del(
    contig_dir: &Path,
    n_samples: usize,
    ploidy: usize,
) -> Result<(), ConversionError> {
    let var_key = var_key_max_del(contig_dir, n_samples, ploidy)?;
    let out = layout::max_del(contig_dir);
    write_npy(&out, &var_key).map_err(|source| ConversionError::Npy {
        path: out.display().to_string(),
        source,
    })?;

    let dense = dense_max_del_scalar(contig_dir)?;
    let dense_out = layout::dense_max_del(contig_dir);
    // dense/max_del.npy lives under {contig_dir}/dense, which may not exist for a
    // contig without any dense stream. Create it so the write (and the contract)
    // always succeeds.
    if let Some(parent) = dense_out.parent() {
        std::fs::create_dir_all(parent).map_err(|source| ConversionError::Io {
            context: format!("creating {}", parent.display()),
            source,
        })?;
    }
    let dense_arr = Array1::from_vec(vec![dense]); // shape (1,)
    write_npy(&dense_out, &dense_arr).map_err(|source| ConversionError::Npy {
        path: dense_out.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Per-column max deletion length over the `var_key/indel` stream, reshaped to
/// `(n_samples, ploidy)`. Absent stream ⇒ all-zero (pure-SNP / no-indel contig).
fn var_key_max_del(
    contig_dir: &Path,
    n_samples: usize,
    ploidy: usize,
) -> Result<Array2<u32>, ConversionError> {
    let total_columns = n_samples * ploidy;
    let indel_dir = layout::var_key_indel_dir(contig_dir);
    let offsets_path = layout::offsets(&indel_dir);

    // No offsets file ⇒ the stream was never written ⇒ zero output.
    if !offsets_path.exists() {
        return Ok(Array2::zeros((n_samples, ploidy)));
    }

    let offsets: Array1<u64> =
        ndarray_npy::read_npy(&offsets_path).map_err(|source| ConversionError::ReadNpy {
            path: offsets_path.display().to_string(),
            source,
        })?;
    debug_assert_eq!(
        offsets.len(),
        total_columns + 1,
        "offsets.npy length must be total_columns + 1"
    );

    let keys = read_keys(&layout::alleles(&indel_dir))?;

    let mut per_col = vec![0u32; total_columns];
    for (c, slot) in per_col.iter_mut().enumerate() {
        let lo = offsets[c] as usize;
        let hi = offsets[c + 1] as usize;
        *slot = keys[lo..hi]
            .iter()
            .copied()
            .map(deletion_len)
            .max()
            .unwrap_or(0);
    }

    // Sample-major columns (c = s*ploidy + p) ⇒ row-major reshape to (n_samples, ploidy).
    Ok(Array2::from_shape_vec((n_samples, ploidy), per_col)
        .expect("per_col length == n_samples * ploidy"))
}

/// Single max deletion length over the shared `dense/indel` key table. Absent
/// stream ⇒ 0.
fn dense_max_del_scalar(contig_dir: &Path) -> Result<u32, ConversionError> {
    let keys = read_keys(&layout::alleles(&layout::dense_indel_dir(contig_dir)))?;
    Ok(keys.iter().copied().map(deletion_len).max().unwrap_or(0))
}

/// Read a raw little-endian `u32` key file into a `Vec<u32>`. A missing file is
/// treated as empty (an absent stream). Alignment-agnostic: `std::fs::read` may
/// return an unaligned buffer, so decode via `chunks_exact(4)` rather than
/// `bytemuck::cast_slice`.
fn read_keys(path: &Path) -> Result<Vec<u32>, ConversionError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(source) => Err(ConversionError::Io {
            context: format!("reading {}", path.display()),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    // `use super::*` already brings `Array1`, `Array2`, `Path`, `layout`,
    // `write_npy`, `deletion_len`, and `write_max_del` into scope.
    use super::*;
    use tempfile::tempdir;

    /// Pure-DEL key for a deletion of `d` reference bases (`ilen = -d`), matching
    /// the pure-DEL lane of `rvk::pack_variant`. `d == 0` yields a non-deletion
    /// inline key (deletion_len 0). The encoder↔decoder faithfulness itself is
    /// proven in `rvk`'s `deletion_len` tests; here we only exercise the scan.
    fn del_key(d: u32) -> u32 {
        if d == 0 {
            0 // inline lane, bit 31 clear → not a deletion
        } else {
            ((-(d as i32)) << 1) as u32
        }
    }

    /// Write a synthetic `var_key/indel` stream: `offsets.npy` (u64 prefix sum,
    /// len total_columns+1) and `alleles.bin` (raw le u32 keys).
    fn write_var_key_indel(contig_dir: &Path, offsets: &[u64], keys: &[u32]) {
        let dir = layout::var_key_indel_dir(contig_dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_npy(layout::offsets(&dir), &Array1::from_vec(offsets.to_vec())).unwrap();
        std::fs::write(layout::alleles(&dir), bytemuck::cast_slice(keys)).unwrap();
    }

    /// Write a synthetic `dense/indel` stream: just `alleles.bin` (raw le u32 keys).
    fn write_dense_indel(contig_dir: &Path, keys: &[u32]) {
        let dir = layout::dense_indel_dir(contig_dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(layout::alleles(&dir), bytemuck::cast_slice(keys)).unwrap();
    }

    fn read_max_del(contig_dir: &Path) -> Array2<u32> {
        ndarray_npy::read_npy(layout::max_del(contig_dir)).unwrap()
    }

    fn read_dense_max_del(contig_dir: &Path) -> Array1<u32> {
        ndarray_npy::read_npy(layout::dense_max_del(contig_dir)).unwrap()
    }

    #[test]
    fn absent_streams_emit_all_zero_artifacts() {
        // No var_key/indel, no dense/indel dirs at all. Contract still holds:
        // an all-zero (n_samples, ploidy) file and a [0] dense scalar.
        let tmp = tempdir().unwrap();
        let c = tmp.path();
        write_max_del(c, 2, 2).unwrap();
        assert_eq!(read_max_del(c), Array2::<u32>::zeros((2, 2)));
        assert_eq!(read_dense_max_del(c), Array1::from_vec(vec![0u32]));
    }

    #[test]
    fn per_column_max_over_var_key_indel() {
        // np = 4 columns (2 samples x ploidy 2), sample-major c = s*ploidy + p.
        //   col 0: dels {3, 1}      → 3
        //   col 1: {} (empty)       → 0
        //   col 2: SNP/INS only     → 0
        //   col 3: dels {2}         → 2
        let tmp = tempdir().unwrap();
        let c = tmp.path();
        let keys = vec![
            del_key(3),
            del_key(1), // col 0
            // col 1 empty
            del_key(0),
            del_key(0), // col 2: non-deletion inline keys
            del_key(2), // col 3
        ];
        let offsets = vec![0u64, 2, 2, 4, 5];
        write_var_key_indel(c, &offsets, &keys);

        write_max_del(c, 2, 2).unwrap();
        let m = read_max_del(c);
        // row-major reshape: [[col0, col1], [col2, col3]]
        assert_eq!(m[[0, 0]], 3);
        assert_eq!(m[[0, 1]], 0);
        assert_eq!(m[[1, 0]], 0);
        assert_eq!(m[[1, 1]], 2);
        // No dense stream → dense scalar is 0.
        assert_eq!(read_dense_max_del(c), Array1::from_vec(vec![0u32]));
    }

    #[test]
    fn dense_scalar_is_single_max_over_shared_keys() {
        let tmp = tempdir().unwrap();
        let c = tmp.path();
        // A pure-SNP var_key side (no deletions) + a dense indel table with dels.
        write_var_key_indel(c, &[0u64, 0, 0, 0, 0], &[]);
        write_dense_indel(c, &[del_key(5), del_key(2), del_key(0)]);

        write_max_del(c, 2, 2).unwrap();
        assert_eq!(read_max_del(c), Array2::<u32>::zeros((2, 2)));
        assert_eq!(read_dense_max_del(c), Array1::from_vec(vec![5u32]));
    }

    use proptest::prelude::*;

    proptest! {
        // Random per-column deletion lengths → the produced (n_samples, ploidy)
        // max_del must equal the brute-force per-column maximum. `0` marks a
        // non-deletion call (SNP/INS). This is the primary correctness gate for
        // the scan + column slicing + reshape.
        #[test]
        fn prop_var_key_max_del_matches_oracle(
            n_samples in 1usize..4,
            ploidy in 1usize..3,
            // per-column deletion lengths, flattened; sized/chunked below.
            col_lens in proptest::collection::vec(0usize..6, 0..12),
            del_seeds in proptest::collection::vec(0u32..1000, 0..64),
        ) {
            let total_columns = n_samples * ploidy;

            // Deterministic per-column call counts (0..=5) derived from col_lens.
            let counts: Vec<usize> =
                (0..total_columns).map(|c| col_lens.get(c).copied().unwrap_or(0)).collect();

            // Assign deletion lengths to calls from del_seeds (cycled), building
            // both the key stream and the per-column oracle max in lockstep.
            let mut keys: Vec<u32> = Vec::new();
            let mut offsets: Vec<u64> = vec![0u64; total_columns + 1];
            let mut oracle = vec![0u32; total_columns];
            let mut seed_i = 0usize;
            for c in 0..total_columns {
                let mut col_max = 0u32;
                for _ in 0..counts[c] {
                    let d = del_seeds.get(seed_i % del_seeds.len().max(1)).copied().unwrap_or(0);
                    seed_i += 1;
                    keys.push(del_key(d));
                    col_max = col_max.max(d);
                }
                offsets[c + 1] = offsets[c] + counts[c] as u64;
                oracle[c] = col_max;
            }

            let tmp = tempdir().unwrap();
            let cdir = tmp.path();
            write_var_key_indel(cdir, &offsets, &keys);
            write_max_del(cdir, n_samples, ploidy).unwrap();

            let m = read_max_del(cdir);
            prop_assert_eq!(m.shape(), &[n_samples, ploidy]);
            for (c, &want) in oracle.iter().enumerate() {
                let s = c / ploidy;
                let p = c % ploidy;
                prop_assert_eq!(m[[s, p]], want, "col {}", c);
            }
        }
    }
}

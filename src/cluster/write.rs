//! Gather a contig's mutations, classify per sample, and write the four
//! `cluster_class` value streams.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use memmap2::MmapMut;
use rayon::prelude::*;

use crate::field::StorageDtype;
use crate::layout::{ContigPaths, FieldSub};
use crate::query::field::{FieldValue, FieldView};
use crate::query::reader::ContigReader;
use crate::query::sidecar::as_bytes;

use super::classify::{Mutation, NONCLUSTERED, NOT_ANNOTATED, cluster_sample};

/// Name of the FORMAT field this module owns.
pub const CLUSTER_CLASS: &str = "cluster_class";

/// Everything that can go wrong while annotating one contig.
#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("{0}")]
    Invalid(String),
}

/// Where a mutation came from, so its label can be written back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    /// Absolute index into `vk_snp`'s call stream.
    VkCall(usize),
    /// Dense SNP column; the label lands at `col * n_samples + sample`.
    DenseCol(usize),
}

struct SampleMutations {
    muts: Vec<Mutation>,
    /// One entry per mutation, in `muts` order: every source element (var_key
    /// call / dense column) that produced it, so a deduped homozygous pair
    /// still gets its label written to both calls.
    origins: Vec<Vec<Origin>>,
}

fn field_f64(v: FieldValue) -> f64 {
    match v {
        FieldValue::Bool(b) => u8::from(b) as f64,
        FieldValue::I8(x) => x as f64,
        FieldValue::U8(x) => x as f64,
        FieldValue::I16(x) => x as f64,
        FieldValue::U16(x) => x as f64,
        FieldValue::I32(x) => x as f64,
        FieldValue::U32(x) => x as f64,
        FieldValue::F16(x) => x.to_f64(),
        FieldValue::F32(x) => x as f64,
    }
}

/// A missing float (NaN) becomes upstream's `-1.5` VAF fill sentinel.
fn map_missing(v: f64) -> f64 {
    if v.is_nan() { -1.5 } else { v }
}

/// All of `sample`'s SNPs on this contig, from both the var_key and dense
/// streams, sorted by `(pos, alt)` and deduped (a dense variant carried by
/// both haplotypes is ONE mutation; a var_key/dense overlap keeps var_key,
/// matching mutcat's precedence).
fn gather_sample(
    reader: &ContigReader,
    snp_positions: &[u32],
    dense_positions: &[u32],
    sample: usize,
    vaf_vk: Option<&FieldView>,
    vaf_dense: Option<&FieldView>,
) -> SampleMutations {
    let mut muts: Vec<Mutation> = Vec::new();
    let mut origins: Vec<Origin> = Vec::new();
    for p in 0..reader.ploidy {
        let col = sample * reader.ploidy + p;
        for call in reader.vk_snp.column(col) {
            let pos = snp_positions[call];
            let key = svar2_codec::unpack_snp_key_at(as_bytes(&reader.vk_snp.keys), call);
            let alt = svar2_codec::decode_snp_2bit(key);
            let vaf = vaf_vk.map(|v| map_missing(field_f64(v.value_at(call))));
            muts.push(Mutation { pos, alt, vaf });
            origins.push(Origin::VkCall(call));
        }
        if let Some(dense) = &reader.dense_snp {
            let keys = as_bytes(&dense.keys);
            dense.for_each_carried(col, |dcol| {
                let pos = dense_positions[dcol];
                let alt = svar2_codec::decode_snp_2bit(svar2_codec::unpack_snp_key_at(keys, dcol));
                let vaf = vaf_dense.map(|v| map_missing(field_f64(v.format_at(dcol, sample))));
                muts.push(Mutation { pos, alt, vaf });
                origins.push(Origin::DenseCol(dcol));
            });
        }
    }
    let mut order: Vec<usize> = (0..muts.len()).collect();
    order.sort_by_key(|&i| {
        let rank = matches!(origins[i], Origin::DenseCol(_)) as u8;
        (muts[i].pos, muts[i].alt, rank)
    });
    let mut out = SampleMutations {
        muts: Vec::with_capacity(muts.len()),
        origins: Vec::with_capacity(muts.len()),
    };
    for i in order {
        let m = muts[i];
        let duplicate = out
            .muts
            .last()
            .is_some_and(|last| last.pos == m.pos && last.alt == m.alt);
        if duplicate {
            out.origins.last_mut().unwrap().push(origins[i]);
        } else {
            out.muts.push(m);
            out.origins.push(vec![origins[i]]);
        }
    }
    out
}

/// Classify every sample of `reader` and write all four `cluster_class`
/// `values.bin` files for `paths`. `cutoffs` has one entry per cohort sample.
/// `vaf` is `(field name, stored dtype)`; `None` runs the no-VAF port.
pub fn annotate_contig(
    reader: &ContigReader,
    paths: &ContigPaths,
    cutoffs: &[f64],
    vaf: Option<(&str, StorageDtype)>,
    vaf_cut: f64,
) -> Result<(), ClusterError> {
    let n_samples = reader.n_samples;
    if cutoffs.len() != n_samples {
        return Err(ClusterError::Invalid(format!(
            "imd_cutoff has {} values for {n_samples} samples",
            cutoffs.len()
        )));
    }
    let snp_positions = reader.vk_snp.positions();
    let dense_positions: &[u32] = reader
        .dense_snp
        .as_ref()
        .map(|d| d.positions())
        .unwrap_or(&[]);
    let vk_calls = reader.vk_snp.offsets.last().copied().unwrap_or(0) as usize;

    let (vaf_vk, vaf_dense) = match vaf {
        Some((name, dtype)) => {
            let vk = FieldView::open(paths, "format", name, FieldSub::VkSnp, dtype, n_samples)?;
            let dense =
                FieldView::open(paths, "format", name, FieldSub::DenseSnp, dtype, n_samples)?;
            let dense_elems = reader
                .dense_snp
                .as_ref()
                .map(|d| d.n_dense_variants * n_samples)
                .unwrap_or(0);
            if vk.len() != vk_calls || dense.len() != dense_elems {
                return Err(ClusterError::Invalid(format!(
                    "VAF field {name:?} has {} var_key and {} dense values, expected \
                     {vk_calls} and {dense_elems}",
                    vk.len(),
                    dense.len()
                )));
            }
            (Some(vk), Some(dense))
        }
        None => (None, None),
    };

    // Gather + classify in parallel; the mmap writes stay single-threaded.
    // If `ContigReader` turns out not to be `Sync` (the long-allele LUT reader
    // is the only suspect), drop `.into_par_iter()` to a plain `(0..n_samples)`
    // map; correctness is identical.
    let classified: Vec<(SampleMutations, Vec<u8>)> = (0..n_samples)
        .into_par_iter()
        .map(|s| {
            let g = gather_sample(
                reader,
                snp_positions,
                dense_positions,
                s,
                vaf_vk.as_ref(),
                vaf_dense.as_ref(),
            );
            let labels = cluster_sample(&g.muts, cutoffs[s], vaf_cut);
            (g, labels)
        })
        .collect();

    let seen_vk: usize = classified
        .iter()
        .map(|(g, _)| {
            g.origins
                .iter()
                .flatten()
                .filter(|o| matches!(o, Origin::VkCall(_)))
                .count()
        })
        .sum();
    if seen_vk != vk_calls {
        return Err(ClusterError::Invalid(format!(
            "gathered {seen_vk} var_key SNP calls, the stream has {vk_calls}"
        )));
    }

    let vk_indel_calls = reader.vk_indel.offsets.last().copied().unwrap_or(0) as usize;
    let dense_snp_elems = reader
        .dense_snp
        .as_ref()
        .map(|d| d.n_dense_variants * n_samples)
        .unwrap_or(0);
    let dense_indel_elems = reader
        .dense_indel
        .as_ref()
        .map(|d| d.n_dense_variants * n_samples)
        .unwrap_or(0);

    write_values(
        &paths.field_values("format", CLUSTER_CLASS, FieldSub::VkSnp),
        vk_calls,
        |buf| {
            buf.fill(NONCLUSTERED);
            for (g, labels) in &classified {
                for (i, origins) in g.origins.iter().enumerate() {
                    for origin in origins {
                        if let Origin::VkCall(call) = *origin {
                            buf[call] = labels[i];
                        }
                    }
                }
            }
        },
    )?;
    write_values(
        &paths.field_values("format", CLUSTER_CLASS, FieldSub::VkIndel),
        vk_indel_calls,
        |buf| buf.fill(NOT_ANNOTATED),
    )?;
    write_values(
        &paths.field_values("format", CLUSTER_CLASS, FieldSub::DenseSnp),
        dense_snp_elems,
        |buf| {
            buf.fill(NOT_ANNOTATED);
            for (sample, (g, labels)) in classified.iter().enumerate() {
                for (i, origins) in g.origins.iter().enumerate() {
                    for origin in origins {
                        if let Origin::DenseCol(col) = *origin {
                            buf[col * n_samples + sample] = labels[i];
                        }
                    }
                }
            }
        },
    )?;
    write_values(
        &paths.field_values("format", CLUSTER_CLASS, FieldSub::DenseIndel),
        dense_indel_elems,
        |buf| buf.fill(NOT_ANNOTATED),
    )?;
    Ok(())
}

/// Write 255 into all four streams (for contigs outside `contigs=` scope), so
/// selecting `cluster_class` and decoding an unannotated contig is coherent.
pub fn fill_contig(reader: &ContigReader, paths: &ContigPaths) -> Result<(), ClusterError> {
    let n_samples = reader.n_samples;
    let streams = [
        (
            FieldSub::VkSnp,
            reader.vk_snp.offsets.last().copied().unwrap_or(0) as usize,
        ),
        (
            FieldSub::VkIndel,
            reader.vk_indel.offsets.last().copied().unwrap_or(0) as usize,
        ),
        (
            FieldSub::DenseSnp,
            reader
                .dense_snp
                .as_ref()
                .map(|d| d.n_dense_variants * n_samples)
                .unwrap_or(0),
        ),
        (
            FieldSub::DenseIndel,
            reader
                .dense_indel
                .as_ref()
                .map(|d| d.n_dense_variants * n_samples)
                .unwrap_or(0),
        ),
    ];
    for (sub, len) in streams {
        write_values(
            &paths.field_values("format", CLUSTER_CLASS, sub),
            len,
            |buf| buf.fill(NOT_ANNOTATED),
        )?;
    }
    Ok(())
}

/// Create `path.tmp` at `len` bytes, hand the mmap to `fill`, then fsync +
/// rename (the `mutcat::sidecar` pattern). `len == 0` writes an empty file.
fn write_values(path: &Path, len: usize, fill: impl FnOnce(&mut [u8])) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = temp_path(path);
    remove_if_exists(&tmp)?;
    if len == 0 {
        File::create(&tmp)?;
        return fs::rename(&tmp, path);
    }
    // `MmapMut` maps PROT_READ|PROT_WRITE, so the staging file must be
    // opened for reading as well as writing (`File::create` is write-only).
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    file.set_len(len as u64)?;
    // SAFETY: `tmp` is private to this call; nothing else maps or writes it
    // between `map_mut` and the rename below.
    let mut mm = unsafe { MmapMut::map_mut(&file)? };
    fill(&mut mm[..]);
    mm.flush()?;
    drop(mm);
    file.sync_all()?;
    fs::rename(&tmp, path)
}

/// `path` with `.tmp` appended to its file name, so staging shares the
/// destination's directory and filesystem.
fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::DOUBLET;

    #[test]
    fn write_values_writes_len_and_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("values.bin");
        write_values(&path, 4, |buf| buf.copy_from_slice(&[1, 2, 3, 4])).unwrap();
        assert_eq!(fs::read(&path).unwrap(), vec![1, 2, 3, 4]);
        // a second call replaces the contents and leaves no temp behind
        write_values(&path, 2, |buf| buf.fill(9)).unwrap();
        assert_eq!(fs::read(&path).unwrap(), vec![9, 9]);
        assert!(!temp_path(&path).exists());
    }

    #[test]
    fn write_values_handles_zero_len() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("values.bin");
        write_values(&path, 0, |_| unreachable!()).unwrap();
        assert!(path.is_file());
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    }

    #[test]
    fn failed_write_leaves_destination_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("values.bin");
        write_values(&path, 1, |buf| buf[0] = 5).unwrap();
        // Block the temp path with a directory: the staged write must fail
        // before the committed destination is replaced.
        fs::create_dir(temp_path(&path)).unwrap();
        assert!(write_values(&path, 1, |buf| buf[0] = 9).is_err());
        assert_eq!(fs::read(&path).unwrap(), vec![5]);
    }

    #[test]
    fn missing_vaf_becomes_the_upstream_sentinel() {
        assert_eq!(map_missing(f64::NAN), -1.5);
        assert_eq!(map_missing(0.4), 0.4);
    }

    /// A homozygous var_key SNP is one mutation for classification, but both
    /// of its calls (one per hap) must get that mutation's label -- and the
    /// call-count invariant must not mistake the dedup for lost calls.
    #[test]
    fn homozygous_vk_calls_share_one_label() {
        use crate::layout;
        use ndarray::Array1;

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().to_str().unwrap();
        let paths = ContigPaths::new(base, "chr1");
        let snp = paths.var_key_snp_dir();
        fs::create_dir_all(&snp).unwrap();
        // 1 sample, ploidy 2: hap 0 carries a doublet at 10/11, hap 1 carries
        // the position-10 variant again (homozygous alt) -> 3 calls, 2 mutations.
        let positions: [u32; 3] = [10, 11, 10];
        fs::write(layout::positions(&snp), bytemuck::cast_slice(&positions)).unwrap();
        // all three calls use alt code 0; one packed byte holds four codes
        fs::write(layout::alleles(&snp), [0u8]).unwrap();
        let offsets = Array1::from_vec(vec![0u64, 2, 3]);
        ndarray_npy::write_npy(layout::offsets(&snp), &offsets).unwrap();

        let reader = ContigReader::open(base, "chr1", 1, 2).unwrap();
        annotate_contig(&reader, &paths, &[25.0], None, 0.1).unwrap();

        let values =
            fs::read(paths.field_values("format", CLUSTER_CLASS, FieldSub::VkSnp)).unwrap();
        assert_eq!(
            values,
            vec![DOUBLET; 3],
            "both hap calls of the doublet must be labelled"
        );
    }
}

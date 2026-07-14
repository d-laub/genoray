//! Single source of truth for the SVAR2 on-disk directory + file layout. Every
//! path the pipeline reads or writes is constructed here, so the finalized
//! on-disk file names are defined in exactly one place.

use std::path::{Path, PathBuf};

/// The four sub-streams a mutcat sidecar mirrors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutcatSub {
    VkSnp,
    VkIndel,
    DenseSnp,
    DenseIndel,
}

impl MutcatSub {
    fn dir_name(self) -> &'static str {
        match self {
            MutcatSub::VkSnp => "var_key_snp",
            MutcatSub::VkIndel => "var_key_indel",
            MutcatSub::DenseSnp => "dense_snp",
            MutcatSub::DenseIndel => "dense_indel",
        }
    }
    /// Whether this sub-stream carries a 2-bit ref-base stream (snp only).
    pub fn has_ref(self) -> bool {
        matches!(self, MutcatSub::VkSnp | MutcatSub::DenseSnp)
    }
    /// SNP sub-streams also carry a 2-bit `strand.bin` when GTF-annotated.
    pub fn has_strand(self) -> bool {
        matches!(self, MutcatSub::VkSnp | MutcatSub::DenseSnp)
    }
}

/// The four sub-streams a field sidecar mirrors. Same four directories as
/// `MutcatSub`, but kept separate: field dirs live under `fields/{category}/`
/// and gain no `has_ref` notion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSub {
    VkSnp,
    VkIndel,
    DenseSnp,
    DenseIndel,
}

impl FieldSub {
    pub fn dir_name(self) -> &'static str {
        match self {
            FieldSub::VkSnp => "var_key_snp",
            FieldSub::VkIndel => "var_key_indel",
            FieldSub::DenseSnp => "dense_snp",
            FieldSub::DenseIndel => "dense_indel",
        }
    }
    /// Every sub-stream, in a fixed order (for iteration at open/finalize time).
    pub fn all() -> [FieldSub; 4] {
        [
            FieldSub::VkSnp,
            FieldSub::VkIndel,
            FieldSub::DenseSnp,
            FieldSub::DenseIndel,
        ]
    }
}

pub struct ContigPaths {
    base_out_dir: String,
    chrom: String,
}

impl ContigPaths {
    pub fn new(base_out_dir: &str, chrom: &str) -> Self {
        Self {
            base_out_dir: base_out_dir.to_string(),
            chrom: chrom.to_string(),
        }
    }

    fn var_key_dir(&self) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("var_key")
    }
    pub fn var_key_snp_dir(&self) -> PathBuf {
        self.var_key_dir().join("snp")
    }
    pub fn var_key_indel_dir(&self) -> PathBuf {
        self.var_key_dir().join("indel")
    }
    pub fn dense_snp_dir(&self) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("dense")
            .join("snp")
    }
    pub fn dense_indel_dir(&self) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("dense")
            .join("indel")
    }
    /// Shared per-contig indel long-allele LUT dir. Both var_key/indel and
    /// dense/indel reference this single table (spilled keys are
    /// representation-portable).
    pub fn shared_indel_dir(&self) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("indel")
    }
    pub fn long_alleles_bin(&self) -> PathBuf {
        self.shared_indel_dir().join("long_alleles.bin")
    }
    pub fn long_allele_offsets(&self) -> PathBuf {
        self.shared_indel_dir().join("long_allele_offsets.npy")
    }

    fn mutcat_dir(&self, sub: MutcatSub) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("mutcat")
            .join(sub.dir_name())
    }
    pub fn mutcat_code(&self, sub: MutcatSub) -> PathBuf {
        self.mutcat_dir(sub).join("code.bin")
    }
    pub fn mutcat_ref(&self, sub: MutcatSub) -> PathBuf {
        self.mutcat_dir(sub).join("ref.bin")
    }
    pub fn mutcat_strand(&self, sub: MutcatSub) -> PathBuf {
        self.mutcat_dir(sub).join("strand.bin")
    }
    /// Directory created before writing a sidecar sub-stream.
    pub fn mutcat_sub_dir(&self, sub: MutcatSub) -> PathBuf {
        self.mutcat_dir(sub)
    }

    /// `{out}/{contig}/fields/{category}/{name}/{sub}/values.bin`.
    /// `category` is `"info"` or `"format"` (see `FieldCategory::as_str`).
    pub fn field_values(&self, category: &str, name: &str, sub: FieldSub) -> PathBuf {
        Path::new(&self.base_out_dir)
            .join(&self.chrom)
            .join("fields")
            .join(category)
            .join(name)
            .join(sub.dir_name())
            .join("values.bin")
    }
}

pub fn chunk_pos(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_pos.bin", chunk_id))
}
pub fn chunk_key(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_key.bin", chunk_id))
}
pub fn positions(dir: &Path) -> PathBuf {
    dir.join("positions.bin")
}
pub fn alleles(dir: &Path) -> PathBuf {
    dir.join("alleles.bin")
}
pub fn offsets(dir: &Path) -> PathBuf {
    dir.join("offsets.npy")
}
pub fn chunk_geno(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_geno.bin", chunk_id))
}
pub fn chunk_field(dir: &Path, chunk_id: usize, field_ix: usize) -> PathBuf {
    dir.join(format!("chunk_{}_field{}.bin", chunk_id, field_ix))
}
pub fn chunk_field_info(dir: &Path, chunk_id: usize, field_ix: usize) -> PathBuf {
    dir.join(format!("chunk_{}_finfo{}.bin", chunk_id, field_ix))
}
pub fn chunk_field_format(dir: &Path, chunk_id: usize, field_ix: usize) -> PathBuf {
    dir.join(format!("chunk_{}_fformat{}.bin", chunk_id, field_ix))
}
pub fn genotypes(dir: &Path) -> PathBuf {
    dir.join("genotypes.bin")
}

/// Contig-dir-relative path helpers for the standalone `max_del` post-pass. These
/// take the contig directory (`{out}/{contig}`) directly, unlike the `ContigPaths`
/// methods which build from `base_out_dir` + `chrom`. Keeping them here preserves
/// layout.rs as the single source of on-disk paths.
pub fn var_key_indel_dir(contig_dir: &Path) -> PathBuf {
    contig_dir.join("var_key").join("indel")
}
pub fn dense_indel_dir(contig_dir: &Path) -> PathBuf {
    contig_dir.join("dense").join("indel")
}
pub fn max_del(contig_dir: &Path) -> PathBuf {
    contig_dir.join("max_del.npy")
}
pub fn dense_max_del(contig_dir: &Path) -> PathBuf {
    contig_dir.join("dense").join("max_del.npy")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_key_stream_dirs() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(p.var_key_snp_dir(), Path::new("/out/chr1/var_key/snp"));
        assert_eq!(p.var_key_indel_dir(), Path::new("/out/chr1/var_key/indel"));
    }

    #[test]
    fn test_long_allele_paths_live_under_shared_indel() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(
            p.long_alleles_bin(),
            Path::new("/out/chr1/indel/long_alleles.bin")
        );
        assert_eq!(
            p.long_allele_offsets(),
            Path::new("/out/chr1/indel/long_allele_offsets.npy")
        );
    }

    #[test]
    fn test_dense_dirs() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(p.dense_snp_dir(), Path::new("/out/chr1/dense/snp"));
        assert_eq!(p.dense_indel_dir(), Path::new("/out/chr1/dense/indel"));
    }

    #[test]
    fn test_max_del_postpass_paths() {
        let c = Path::new("/out/chr1");
        assert_eq!(var_key_indel_dir(c), Path::new("/out/chr1/var_key/indel"));
        assert_eq!(dense_indel_dir(c), Path::new("/out/chr1/dense/indel"));
        assert_eq!(max_del(c), Path::new("/out/chr1/max_del.npy"));
        assert_eq!(dense_max_del(c), Path::new("/out/chr1/dense/max_del.npy"));
    }

    #[test]
    fn test_dense_chunk_and_final_names() {
        let dir = Path::new("/out/chr1/dense/snp");
        assert_eq!(
            chunk_geno(dir, 2),
            Path::new("/out/chr1/dense/snp/chunk_2_geno.bin")
        );
        assert_eq!(
            genotypes(dir),
            Path::new("/out/chr1/dense/snp/genotypes.bin")
        );
    }

    #[test]
    fn test_chunk_field_names() {
        let dir = Path::new("/out/chr1/var_key/snp");
        assert_eq!(
            chunk_field(dir, 3, 0),
            Path::new("/out/chr1/var_key/snp/chunk_3_field0.bin")
        );
        let dense_dir = Path::new("/out/chr1/dense/snp");
        assert_eq!(
            chunk_field_info(dense_dir, 3, 1),
            Path::new("/out/chr1/dense/snp/chunk_3_finfo1.bin")
        );
        assert_eq!(
            chunk_field_format(dense_dir, 3, 2),
            Path::new("/out/chr1/dense/snp/chunk_3_fformat2.bin")
        );
    }

    #[test]
    fn test_chunk_and_final_names() {
        let dir = Path::new("/out/chr1/var_key/snp");
        assert_eq!(
            chunk_pos(dir, 3),
            Path::new("/out/chr1/var_key/snp/chunk_3_pos.bin")
        );
        assert_eq!(
            chunk_key(dir, 3),
            Path::new("/out/chr1/var_key/snp/chunk_3_key.bin")
        );
        assert_eq!(
            positions(dir),
            Path::new("/out/chr1/var_key/snp/positions.bin")
        );
        assert_eq!(alleles(dir), Path::new("/out/chr1/var_key/snp/alleles.bin"));
        assert_eq!(offsets(dir), Path::new("/out/chr1/var_key/snp/offsets.npy"));
    }

    #[test]
    fn test_max_del_paths() {
        let contig = Path::new("/out/chr1");
        assert_eq!(max_del(contig), Path::new("/out/chr1/max_del.npy"));
        assert_eq!(
            dense_max_del(contig),
            Path::new("/out/chr1/dense/max_del.npy")
        );
    }

    #[test]
    fn test_field_values_paths() {
        let paths = ContigPaths::new("/out", "chr1");
        assert_eq!(
            paths.field_values("format", "DS", FieldSub::VkSnp),
            Path::new("/out/chr1/fields/format/DS/var_key_snp/values.bin")
        );
        assert_eq!(
            paths.field_values("info", "AF", FieldSub::DenseIndel),
            Path::new("/out/chr1/fields/info/AF/dense_indel/values.bin")
        );
    }

    #[test]
    fn mutcat_paths() {
        let p = ContigPaths::new("/out", "chr1");
        assert_eq!(
            p.mutcat_code(MutcatSub::VkSnp),
            Path::new("/out/chr1/mutcat/var_key_snp/code.bin")
        );
        assert_eq!(
            p.mutcat_ref(MutcatSub::DenseSnp),
            Path::new("/out/chr1/mutcat/dense_snp/ref.bin")
        );
        assert!(MutcatSub::VkSnp.has_ref());
        assert!(!MutcatSub::VkIndel.has_ref());
    }
}

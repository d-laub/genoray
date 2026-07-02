//! Single source of truth for the SVAR2 on-disk directory + file layout. Every
//! path the pipeline reads or writes is constructed here so the (still
//! provisional) filenames can be changed in exactly one place before M6 decode.

use std::path::{Path, PathBuf};

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
}

pub fn chunk_pos(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_pos.bin", chunk_id))
}
pub fn chunk_key(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_key.bin", chunk_id))
}
pub fn final_positions(dir: &Path) -> PathBuf {
    dir.join("final_positions.bin")
}
pub fn final_keys(dir: &Path) -> PathBuf {
    dir.join("final_keys.bin")
}
pub fn final_offsets(dir: &Path) -> PathBuf {
    dir.join("final_offsets.npy")
}
pub fn chunk_geno(dir: &Path, chunk_id: usize) -> PathBuf {
    dir.join(format!("chunk_{}_geno.bin", chunk_id))
}
pub fn final_genotypes(dir: &Path) -> PathBuf {
    dir.join("final_genotypes.bin")
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
    fn test_dense_chunk_and_final_names() {
        let dir = Path::new("/out/chr1/dense/snp");
        assert_eq!(
            chunk_geno(dir, 2),
            Path::new("/out/chr1/dense/snp/chunk_2_geno.bin")
        );
        assert_eq!(
            final_genotypes(dir),
            Path::new("/out/chr1/dense/snp/final_genotypes.bin")
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
            final_positions(dir),
            Path::new("/out/chr1/var_key/snp/final_positions.bin")
        );
        assert_eq!(
            final_keys(dir),
            Path::new("/out/chr1/var_key/snp/final_keys.bin")
        );
        assert_eq!(
            final_offsets(dir),
            Path::new("/out/chr1/var_key/snp/final_offsets.npy")
        );
    }
}

//! Read/write the per-contig mutcat sidecar. `code.bin` is raw u8 (one per
//! record); `ref.bin` (snp subs only) is 2-bit packed A/C/T/G ref bases.

use std::fs;
use std::io::Write;
use std::path::Path;

use memmap2::Mmap;

use crate::layout::{ContigPaths, MutcatSub};
use crate::query::sidecar::mmap_file;
use svar2_codec::{pack_snp_keys, unpack_snp_key_at};

pub fn write_sidecar(
    paths: &ContigPaths,
    sub: MutcatSub,
    codes: &[u8],
    ref_codes: Option<&[u8]>,
    strand_codes: Option<&[u8]>,
) -> std::io::Result<()> {
    let dir = paths.mutcat_sub_dir(sub);
    fs::create_dir_all(&dir)?;
    write_bytes(&paths.mutcat_code(sub), codes)?;
    if sub.has_ref() {
        let refs = ref_codes.expect("snp sub-stream requires ref_codes");
        debug_assert_eq!(refs.len(), codes.len());
        let packed = pack_snp_keys(refs);
        write_bytes(&paths.mutcat_ref(sub), &packed)?;
    }
    if sub.has_strand()
        && let Some(strands) = strand_codes
    {
        debug_assert_eq!(strands.len(), codes.len());
        let packed = pack_snp_keys(strands);
        write_bytes(&paths.mutcat_strand(sub), &packed)?;
    }
    Ok(())
}

fn write_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(bytes)?;
    f.flush()
}

pub struct MutcatView {
    code: Option<Mmap>,
    ref_packed: Option<Mmap>,
    strand_packed: Option<Mmap>,
    pub n: usize,
    pub has_strand: bool,
}

impl MutcatView {
    /// Class-local mutation code at record `i` (u8; may be a sentinel).
    #[inline]
    pub fn code_at(&self, i: usize) -> u8 {
        match &self.code {
            Some(m) => m[i],
            None => crate::mutcat::NOT_ANNOTATED,
        }
    }
    /// 2-bit reference-base code at snp record `i` (`0–3`). Panics if no ref stream.
    #[inline]
    pub fn ref_at(&self, i: usize) -> u8 {
        let m = self
            .ref_packed
            .as_ref()
            .expect("ref_at on a stream with no ref.bin");
        unpack_snp_key_at(&m[..], i)
    }
    /// 2-bit transcriptional-strand code at snp record `i` (`STRAND_{T,U,N,B}`),
    /// or `STRAND_NA` if this sidecar has no strand stream (annotated without a GTF).
    #[inline]
    pub fn strand_at(&self, i: usize) -> u8 {
        match &self.strand_packed {
            Some(m) => unpack_snp_key_at(&m[..], i),
            None => crate::mutcat::STRAND_NA,
        }
    }
}

pub fn open_sidecar(paths: &ContigPaths, sub: MutcatSub) -> std::io::Result<MutcatView> {
    let code = mmap_file(&paths.mutcat_code(sub))?;
    let n = code.as_ref().map(|m| m.len()).unwrap_or(0);
    let ref_packed = if sub.has_ref() {
        mmap_file(&paths.mutcat_ref(sub))?
    } else {
        None
    };
    let strand_packed = if sub.has_strand() {
        mmap_file(&paths.mutcat_strand(sub))?
    } else {
        None
    };
    let has_strand = strand_packed.is_some();
    Ok(MutcatView {
        code,
        ref_packed,
        strand_packed,
        n,
        has_strand,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snp_sidecar_round_trips_code_and_ref() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let codes = [5u8, 95, 254, 0];
        let refs = [1u8, 3, 0, 2]; // C,G(→ codec 3),A,T
        write_sidecar(&paths, MutcatSub::VkSnp, &codes, Some(&refs), None).unwrap();
        let v = open_sidecar(&paths, MutcatSub::VkSnp).unwrap();
        assert_eq!(v.n, 4);
        assert!(!v.has_strand);
        for i in 0..4 {
            assert_eq!(v.code_at(i), codes[i]);
            assert_eq!(v.ref_at(i), refs[i]);
            assert_eq!(v.strand_at(i), crate::mutcat::STRAND_NA);
        }
    }

    #[test]
    fn snp_sidecar_round_trips_strand() {
        use crate::mutcat::{STRAND_B, STRAND_N, STRAND_T, STRAND_U};
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let codes = [5u8, 95, 254, 0];
        let refs = [1u8, 3, 0, 2];
        let strands = [STRAND_T, STRAND_U, STRAND_N, STRAND_B];
        write_sidecar(
            &paths,
            MutcatSub::VkSnp,
            &codes,
            Some(&refs),
            Some(&strands),
        )
        .unwrap();
        let v = open_sidecar(&paths, MutcatSub::VkSnp).unwrap();
        assert!(v.has_strand);
        for (i, &want) in strands.iter().enumerate() {
            assert_eq!(v.strand_at(i), want);
        }
    }

    #[test]
    fn indel_sidecar_has_no_ref() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let codes = [10u8, 82, 255];
        write_sidecar(&paths, MutcatSub::VkIndel, &codes, None, None).unwrap();
        let v = open_sidecar(&paths, MutcatSub::VkIndel).unwrap();
        assert_eq!(v.n, 3);
        assert_eq!(v.code_at(1), 82);
    }

    #[test]
    fn missing_sidecar_opens_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let v = open_sidecar(&paths, MutcatSub::DenseSnp).unwrap();
        assert_eq!(v.n, 0);
        assert_eq!(v.code_at(0), crate::mutcat::NOT_ANNOTATED);
    }
}

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
    // Auxiliary streams first, `code.bin` last. `code.bin`'s existence is the
    // annotation gate (`SparseVar2._is_annotated`), so a reader must never see
    // it before the streams it indexes are in place.
    if sub.has_ref() {
        let refs = ref_codes.expect("snp sub-stream requires ref_codes");
        debug_assert_eq!(refs.len(), codes.len());
        let packed = pack_snp_keys(refs);
        write_bytes(&paths.mutcat_ref(sub), &packed)?;
    }
    if sub.has_strand() {
        match strand_codes {
            Some(strands) => {
                debug_assert_eq!(strands.len(), codes.len());
                let packed = pack_snp_keys(strands);
                write_bytes(&paths.mutcat_strand(sub), &packed)?;
            }
            // A GTF-less re-annotation must not leave the previous
            // annotation's strand classes behind: `strand.bin` is read as
            // ground truth for SBS192/SBS384.
            None => remove_if_exists(&paths.mutcat_strand(sub))?,
        }
    }
    write_bytes(&paths.mutcat_code(sub), codes)
}

fn write_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = temp_path(path);
    remove_if_exists(&tmp)?;
    let mut f = fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    f.flush()?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)
}

/// `path` with `.tmp` appended to its file name (`code.bin` -> `code.bin.tmp`),
/// so staging shares the destination's directory and filesystem.
fn temp_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

fn remove_if_exists(path: &Path) -> std::io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
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

/// Open a mutcat sub-stream sidecar, verifying each present stream's byte
/// length against `expected_n` records (1 byte per code; 2-bit packed aux
/// streams). A missing stream counts as 0 bytes, which is legitimate exactly
/// when nothing is expected of it - an absent `strand.bin` is a gtf-less
/// annotation, not an error. Any other mismatch means the sidecar is truncated
/// (a writer killed mid-write) or stale (a different variant set) and is
/// rejected rather than indexed.
pub fn open_sidecar(
    paths: &ContigPaths,
    sub: MutcatSub,
    expected_n: usize,
) -> std::io::Result<MutcatView> {
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
    check_stream_len(&paths.mutcat_code(sub), n, expected_n)?;
    let packed_expected = expected_n.div_ceil(4);
    if sub.has_ref() {
        check_stream_len(
            &paths.mutcat_ref(sub),
            ref_packed.as_ref().map(|m| m.len()).unwrap_or(0),
            packed_expected,
        )?;
    }
    if let Some(m) = strand_packed.as_ref() {
        check_stream_len(&paths.mutcat_strand(sub), m.len(), packed_expected)?;
    }
    let has_strand = strand_packed.is_some();
    Ok(MutcatView {
        code,
        ref_packed,
        strand_packed,
        n,
        has_strand,
    })
}

/// Reject a stream whose on-disk length disagrees with the contig's records.
fn check_stream_len(path: &Path, len: usize, expected: usize) -> std::io::Result<()> {
    if len == expected {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "{} is truncated or stale: expected {expected} bytes, found {len}; re-run annotate_mutations",
            path.display()
        ),
    ))
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
        let v = open_sidecar(&paths, MutcatSub::VkSnp, 4).unwrap();
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
        let v = open_sidecar(&paths, MutcatSub::VkSnp, 4).unwrap();
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
        let v = open_sidecar(&paths, MutcatSub::VkIndel, 3).unwrap();
        assert_eq!(v.n, 3);
        assert_eq!(v.code_at(1), 82);
    }

    #[test]
    fn missing_sidecar_opens_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        let v = open_sidecar(&paths, MutcatSub::DenseSnp, 0).unwrap();
        assert_eq!(v.n, 0);
        assert_eq!(v.code_at(0), crate::mutcat::NOT_ANNOTATED);
    }

    #[test]
    fn successful_write_leaves_no_temp_files() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        write_sidecar(&paths, MutcatSub::VkSnp, &[5u8, 95], Some(&[1u8, 2]), None).unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(paths.mutcat_sub_dir(MutcatSub::VkSnp))
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp residue: {leftovers:?}");
    }

    #[test]
    fn failed_write_leaves_destination_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        write_sidecar(&paths, MutcatSub::VkSnp, &[5u8], Some(&[1u8]), None).unwrap();
        // Block the temp path with a directory: the staged write must fail
        // before the already-committed destination is replaced.
        let dir = paths.mutcat_sub_dir(MutcatSub::VkSnp);
        std::fs::create_dir(dir.join("code.bin.tmp")).unwrap();
        let err = write_sidecar(&paths, MutcatSub::VkSnp, &[9u8], Some(&[2u8]), None);
        assert!(
            err.is_err(),
            "write should fail when the staging path is blocked"
        );
        let v = open_sidecar(&paths, MutcatSub::VkSnp, 1).unwrap();
        assert_eq!(v.n, 1);
        assert_eq!(
            v.code_at(0),
            5,
            "destination was modified by a failed write"
        );
    }

    #[test]
    fn reannotate_without_strand_drops_stale_strand_stream() {
        use crate::mutcat::STRAND_T;
        let tmp = tempfile::tempdir().unwrap();
        let paths = ContigPaths::new(tmp.path().to_str().unwrap(), "chr1");
        write_sidecar(
            &paths,
            MutcatSub::VkSnp,
            &[5u8],
            Some(&[1u8]),
            Some(&[STRAND_T]),
        )
        .unwrap();
        assert!(
            open_sidecar(&paths, MutcatSub::VkSnp, 1)
                .unwrap()
                .has_strand
        );

        write_sidecar(&paths, MutcatSub::VkSnp, &[6u8], Some(&[1u8]), None).unwrap();
        let v = open_sidecar(&paths, MutcatSub::VkSnp, 1).unwrap();
        assert!(
            !v.has_strand,
            "stale strand.bin survived a gtf-less re-annotation"
        );
        assert_eq!(v.code_at(0), 6);
    }
}

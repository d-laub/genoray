//! `RecordSource` that builds ONE merged stream out of N single-sample VCFs with
//! DIFFERENT site lists (`SparseVar2.from_vcf_list`'s merge core).
//!
//! Each file is normalized independently to atomic biallelic VCF-form records
//! (`normalize::atomize_to_vcf_biallelic`), then the N per-file atom streams are
//! k-way merged by `(pos, ilen, alt)` identity: atoms that share a key across
//! files collapse into ONE merged `RawRecord`, with every column not carrying
//! that atom filled hom-ref (`0`). This is a *reorder* merge, not a plain
//! synchronized merge — a file can be several records "ahead" of another before
//! it's safe to release the lowest-keyed atom, mirroring the reorder buffer in
//! `chunk_assembler`. Genotypes only in this task; INFO/FORMAT fields are always
//! emitted empty (Task 6 wires them through).

use crate::error::ConversionError;
use crate::normalize::{L_MAX, atomize_to_vcf_biallelic, validate_ref};
use crate::record_source::{RawRecord, RecordSource};
use crate::vcf_reader::VcfRecordSource;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

/// One atomized, biallelic, single-file/single-sample variant, not yet merged
/// with any other file's atoms sharing the same `(pos, ilen, alt)` key.
struct NormAtom {
    pos: u32,
    ilen: i32,
    reference: Vec<u8>,
    alt: Vec<u8>,
    /// Per-haplotype presence code, length `ploidy`: `-1` missing, `1` carries
    /// this atom's source ALT, `0` carries something else (incl. REF).
    ploid_codes: Vec<i32>,
}

/// One per-file cursor: an (optional, if the contig is absent) VCF reader, a
/// small buffer of not-yet-emitted atoms decomposed from the last record it
/// pulled, and the position of the last record it *read* (its "frontier" —
/// distinct from the position of any individual atom still buffered).
struct FileCursor {
    vcf: Option<VcfRecordSource>,
    buf: VecDeque<NormAtom>,
    frontier: Option<u32>,
    eof: bool,
    col: usize,
    dropped: u64,
}

impl FileCursor {
    /// If the buffer is empty and the file isn't exhausted, pull one more raw
    /// record, atomize it (validating REF against `ref_seq` first, matching
    /// `chunk_assembler`'s fail-fast contract), and refill the buffer. Then pop
    /// and return the front atom, if any. May return `None` while `!eof` when
    /// the pulled record atomized to zero atoms (e.g. all-`*`/dropped ALTs) —
    /// the caller just calls `advance` again to keep making progress.
    fn advance(
        &mut self,
        ref_seq: Option<&[u8]>,
        ploidy: usize,
        skip_out_of_scope: bool,
    ) -> Result<Option<NormAtom>, ConversionError> {
        if self.buf.is_empty() && !self.eof {
            let vcf = self
                .vcf
                .as_mut()
                .expect("FileCursor with vcf=None must be eof");
            match vcf.next_record()? {
                None => {
                    self.eof = true;
                }
                Some(rec) => {
                    self.frontier = Some(rec.pos);
                    if let Some(rs) = ref_seq {
                        validate_ref(rec.pos, &rec.reference, rs)?;
                    }
                    let alt_refs: Vec<&[u8]> = rec.alts.iter().map(|a| a.as_slice()).collect();
                    let (records, dropped) = atomize_to_vcf_biallelic(
                        rec.pos,
                        &rec.reference,
                        &alt_refs,
                        ref_seq,
                        skip_out_of_scope,
                    )?;
                    self.dropped += dropped as u64;
                    for br in records {
                        // Uppercase defensively: a soft-masked reference gives back
                        // lowercase REF/anchor bytes from `atomize_to_vcf_biallelic`,
                        // but `RawRecord.reference` is contractually uppercase ASCII,
                        // and the merge key below compares ALT bytes across files —
                        // both must be normalized to the same case to collide.
                        let mut reference = br.reference;
                        reference.make_ascii_uppercase();
                        let mut alt = br.alt;
                        alt.make_ascii_uppercase();
                        let ilen = alt.len() as i32 - reference.len() as i32;

                        let k = br.source_alt_index as i32;
                        let mut ploid_codes = vec![0i32; ploidy];
                        for (p, code) in ploid_codes.iter_mut().enumerate() {
                            let g = rec.gt[p];
                            *code = if g == -1 {
                                -1
                            } else if g == k {
                                1
                            } else {
                                0
                            };
                        }

                        self.buf.push_back(NormAtom {
                            pos: br.pos,
                            ilen,
                            reference,
                            alt,
                            ploid_codes,
                        });
                    }
                }
            }
        }
        Ok(self.buf.pop_front())
    }
}

/// Min-heap entry: ordered by the merge key `(pos, ilen, alt)` first, then by
/// column — the column tiebreak only affects which of several same-key atoms
/// is "on top" (irrelevant to correctness, but makes iteration order
/// deterministic for tests).
struct HeapEntry {
    key: (u32, i32, Vec<u8>),
    col: usize,
    atom: NormAtom,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.col == other.col
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key).then(self.col.cmp(&other.col))
    }
}

/// K-way merge of N single-sample VCFs (with possibly disjoint site lists) into
/// one biallelic, hom-ref-filled `RawRecord` stream.
pub struct VcfListRecordSource {
    cursors: Vec<FileCursor>,
    heap: BinaryHeap<Reverse<HeapEntry>>,
    ref_seq: Option<Vec<u8>>,
    ploidy: usize,
    skip_out_of_scope: bool,
    num_samples: usize,
}

impl VcfListRecordSource {
    /// `vcf_paths[i]` is a single-sample file whose sample is `samples[i]` and
    /// whose merged column is `i`. `ref_seq` is the contig sequence (`None` ⇒
    /// no-reference mode). Files without `chrom` in their header are skipped
    /// (their column is filled hom-ref for every merged record).
    pub fn new(
        vcf_paths: &[String],
        samples: &[&str],
        chrom: &str,
        ref_seq: Option<&[u8]>,
        ploidy: usize,
        htslib_threads: usize,
        skip_out_of_scope: bool,
    ) -> Result<Self, ConversionError> {
        assert_eq!(
            vcf_paths.len(),
            samples.len(),
            "vcf_paths and samples must be parallel (one sample per file)"
        );
        let mut cursors = Vec::with_capacity(vcf_paths.len());
        for (col, (path, &sample)) in vcf_paths.iter().zip(samples.iter()).enumerate() {
            let single_sample = [sample];
            match VcfRecordSource::new(path, chrom, &single_sample, htslib_threads, ploidy, &[]) {
                Ok(vcf) => cursors.push(FileCursor {
                    vcf: Some(vcf),
                    buf: VecDeque::new(),
                    frontier: None,
                    eof: false,
                    col,
                    dropped: 0,
                }),
                Err(ConversionError::Input(msg)) if msg.contains("not found in VCF header") => {
                    cursors.push(FileCursor {
                        vcf: None,
                        buf: VecDeque::new(),
                        frontier: None,
                        eof: true,
                        col,
                        dropped: 0,
                    });
                }
                Err(e) => return Err(e),
            }
        }
        Ok(Self {
            cursors,
            heap: BinaryHeap::new(),
            ref_seq: ref_seq.map(|s| s.to_vec()),
            ploidy,
            skip_out_of_scope,
            num_samples: vcf_paths.len(),
        })
    }

    /// Total out-of-scope (symbolic/breakend) ALTs dropped across every file so
    /// far. Valid after the read loop drains.
    pub fn dropped_out_of_scope(&self) -> u64 {
        self.cursors.iter().map(|c| c.dropped).sum()
    }

    // Build the merged `RawRecord` for a group of same-key `(col, NormAtom)`
    // pairs: one biallelic record whose REF/ALT come from the first atom (they
    // agree by construction — same key), with every column not present in the
    // group left hom-ref (`0`, the `gt` vec's fill value).
    fn build_record(&self, key: &(u32, i32, Vec<u8>), group: Vec<(usize, NormAtom)>) -> RawRecord {
        let reference = group[0].1.reference.clone();
        let alt = group[0].1.alt.clone();
        let mut gt = vec![0i32; self.num_samples * self.ploidy];
        for (col, atom) in &group {
            let base = col * self.ploidy;
            gt[base..base + self.ploidy].copy_from_slice(&atom.ploid_codes);
        }
        RawRecord {
            pos: key.0,
            reference,
            alts: vec![alt],
            gt,
            info_raw: vec![],
            format_raw: vec![],
        }
    }
}

impl RecordSource for VcfListRecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        loop {
            let live_frontiers: Vec<Option<u32>> = self
                .cursors
                .iter()
                .filter(|c| !c.eof)
                .map(|c| c.frontier)
                .collect();
            let all_eof = live_frontiers.is_empty();
            let all_started = live_frontiers.iter().all(|f| f.is_some());
            let min_started = live_frontiers.iter().filter_map(|&f| f).min();

            // Unstarted cursors (`frontier == None`) must be read before anything
            // can release — otherwise a not-yet-opened file's first record could
            // sort below an already-released key, corrupting the merge order.
            let releasable = all_eof
                || (all_started
                    && self.heap.peek().is_some_and(|Reverse(e)| {
                        e.key.0 < min_started.unwrap().saturating_sub(L_MAX)
                    }));

            if releasable {
                return Ok(match self.heap.pop() {
                    None => None,
                    Some(Reverse(top)) => {
                        let key = top.key.clone();
                        let mut group = vec![(top.col, top.atom)];
                        while let Some(Reverse(next)) = self.heap.peek() {
                            if next.key != key {
                                break;
                            }
                            let Reverse(next) = self.heap.pop().unwrap();
                            group.push((next.col, next.atom));
                        }
                        Some(self.build_record(&key, group))
                    }
                });
            }

            // Advance the live cursor with the smallest frontier; `None` (an
            // unstarted cursor) sorts before every `Some(_)`.
            let advance_idx = self
                .cursors
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.eof)
                .min_by_key(|(_, c)| c.frontier)
                .map(|(i, _)| i);
            match advance_idx {
                Some(i) => {
                    let ref_seq = self.ref_seq.as_deref();
                    if let Some(atom) =
                        self.cursors[i].advance(ref_seq, self.ploidy, self.skip_out_of_scope)?
                    {
                        let key = (atom.pos, atom.ilen, atom.alt.clone());
                        let col = self.cursors[i].col;
                        self.heap.push(Reverse(HeapEntry { key, col, atom }));
                    }
                }
                // No live cursors: `all_eof` was already true above, which forces
                // `releasable`, so this arm is unreachable — kept for safety.
                None => return Ok(None),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_htslib::bcf::record::GenotypeAllele;
    use rust_htslib::bcf::{Format, Header, Writer};
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    // One synthetic single-sample record: 0-based pos, REF, ALTs, and a flat
    // per-haplotype genotype (allele index, or -1 for missing), length ploidy.
    struct SsRec {
        pos: i64,
        r: &'static [u8],
        alts: Vec<&'static [u8]>,
        gt: Vec<i32>,
    }

    // Write a single-sample, CSI-indexed BCF (a binary VCF — same on-disk
    // contract `VcfRecordSource` expects) containing `records`, all on `chrom`.
    fn write_ss_vcf(
        dir: &Path,
        name: &str,
        chrom: &str,
        chrom_len: u64,
        sample: &str,
        records: &[SsRec],
    ) -> PathBuf {
        let path = dir.join(format!("{name}.bcf"));
        let mut header = Header::new();
        header.push_record(format!("##contig=<ID={chrom},length={chrom_len}>").as_bytes());
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_sample(sample.as_bytes());
        {
            let mut writer =
                Writer::from_path(&path, &header, false, Format::Bcf).expect("open BCF writer");
            for rec in records {
                let mut record = writer.empty_record();
                record.set_rid(Some(0));
                record.set_pos(rec.pos);
                let mut alleles: Vec<&[u8]> = Vec::with_capacity(1 + rec.alts.len());
                alleles.push(rec.r);
                alleles.extend(rec.alts.iter().copied());
                record.set_alleles(&alleles).expect("set alleles");
                let gt_alleles: Vec<GenotypeAllele> = rec
                    .gt
                    .iter()
                    .map(|&g| {
                        if g < 0 {
                            GenotypeAllele::PhasedMissing
                        } else {
                            GenotypeAllele::Phased(g)
                        }
                    })
                    .collect();
                record.push_genotypes(&gt_alleles).expect("push genotypes");
                writer.write(&record).expect("write record");
            }
        }
        rust_htslib::bcf::index::build(&path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
            .expect("build BCF index");
        path
    }

    // Build a FASTA (+ .fai) whose single contig is `chrom_len` bases of 'N'
    // filler with `stamps` (0-based pos, bases) written in — 'N' never
    // satisfies left-align's repeat-roll condition, so records that shouldn't
    // move don't (mirrors `tests/common::build_fasta_with_index`).
    fn make_ref(dir: &Path, chrom: &str, chrom_len: usize, stamps: &[(usize, &[u8])]) -> Vec<u8> {
        let mut seq = vec![b'N'; chrom_len];
        for &(pos, bases) in stamps {
            seq[pos..pos + bases.len()].copy_from_slice(bases);
        }
        let path = dir.join(format!("{chrom}.fa"));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).expect("create fasta");
            writeln!(f, ">{chrom}").expect("write header");
            f.write_all(&seq).expect("write seq");
            writeln!(f).expect("write newline");
        }
        rust_htslib::faidx::build(&path).expect("build .fai");
        seq
    }

    #[test]
    fn two_files_disjoint_sites_hom_ref_fill() {
        let tmp = tempdir().unwrap();
        // file A: chr1 POS3 (0-based pos 2) A>G, GT 1|0.
        let a = write_ss_vcf(
            tmp.path(),
            "a",
            "chr1",
            100,
            "SA",
            &[SsRec {
                pos: 2,
                r: b"A",
                alts: vec![b"G"],
                gt: vec![1, 0],
            }],
        );
        // file B: chr1 POS7 (0-based pos 6) C>CAT, GT 0|1.
        let b = write_ss_vcf(
            tmp.path(),
            "b",
            "chr1",
            100,
            "SB",
            &[SsRec {
                pos: 6,
                r: b"C",
                alts: vec![b"CAT"],
                gt: vec![0, 1],
            }],
        );
        let ref_seq = make_ref(tmp.path(), "chr1", 100, &[(2, b"A"), (6, b"C")]);

        let paths = vec![
            a.to_str().unwrap().to_string(),
            b.to_str().unwrap().to_string(),
        ];
        let samples = vec!["SA", "SB"];
        let mut src =
            VcfListRecordSource::new(&paths, &samples, "chr1", Some(&ref_seq), 2, 1, false)
                .unwrap();

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(r1.alts, vec![b"G".to_vec()]);
        assert_eq!(r1.gt, vec![1, 0, /* SB */ 0, 0]);

        let r2 = src.next_record().unwrap().unwrap();
        assert_eq!(r2.pos, 6);
        assert_eq!(r2.gt, vec![/* SA */ 0, 0, /* SB */ 0, 1]);

        assert!(src.next_record().unwrap().is_none());
        assert_eq!(src.dropped_out_of_scope(), 0);
    }

    #[test]
    fn shared_site_merges_into_one_record() {
        let tmp = tempdir().unwrap();
        let a = write_ss_vcf(
            tmp.path(),
            "a",
            "chr1",
            100,
            "SA",
            &[SsRec {
                pos: 2,
                r: b"A",
                alts: vec![b"G"],
                gt: vec![1, 0],
            }],
        );
        let b = write_ss_vcf(
            tmp.path(),
            "b",
            "chr1",
            100,
            "SB",
            &[SsRec {
                pos: 2,
                r: b"A",
                alts: vec![b"G"],
                gt: vec![0, 1],
            }],
        );
        let ref_seq = make_ref(tmp.path(), "chr1", 100, &[(2, b"A")]);

        let paths = vec![
            a.to_str().unwrap().to_string(),
            b.to_str().unwrap().to_string(),
        ];
        let samples = vec!["SA", "SB"];
        let mut src =
            VcfListRecordSource::new(&paths, &samples, "chr1", Some(&ref_seq), 2, 1, false)
                .unwrap();

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(r1.alts, vec![b"G".to_vec()]);
        assert_eq!(r1.gt, vec![1, 0, 0, 1]);
        assert!(src.next_record().unwrap().is_none());
    }

    #[test]
    fn multiallelic_in_one_file_splits_across_records() {
        let tmp = tempdir().unwrap();
        // file A: chr1 POS3 (0-based pos 2) A>G,T, GT 1|2.
        let a = write_ss_vcf(
            tmp.path(),
            "a",
            "chr1",
            100,
            "SA",
            &[SsRec {
                pos: 2,
                r: b"A",
                alts: vec![b"G", b"T"],
                gt: vec![1, 2],
            }],
        );
        let ref_seq = make_ref(tmp.path(), "chr1", 100, &[(2, b"A")]);

        let paths = vec![a.to_str().unwrap().to_string()];
        let samples = vec!["SA"];
        let mut src =
            VcfListRecordSource::new(&paths, &samples, "chr1", Some(&ref_seq), 2, 1, false)
                .unwrap();

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(r1.alts, vec![b"G".to_vec()]);
        assert_eq!(r1.gt, vec![1, 0]);

        let r2 = src.next_record().unwrap().unwrap();
        assert_eq!(r2.pos, 2);
        assert_eq!(r2.alts, vec![b"T".to_vec()]);
        assert_eq!(r2.gt, vec![0, 1]);

        assert!(src.next_record().unwrap().is_none());
    }

    #[test]
    fn within_file_missing_stays_minus_one() {
        let tmp = tempdir().unwrap();
        // file A: chr1 POS3 (0-based pos 2) A>G, GT .|1.
        let a = write_ss_vcf(
            tmp.path(),
            "a",
            "chr1",
            100,
            "SA",
            &[SsRec {
                pos: 2,
                r: b"A",
                alts: vec![b"G"],
                gt: vec![-1, 1],
            }],
        );
        let ref_seq = make_ref(tmp.path(), "chr1", 100, &[(2, b"A")]);

        let paths = vec![a.to_str().unwrap().to_string()];
        let samples = vec!["SA"];
        let mut src =
            VcfListRecordSource::new(&paths, &samples, "chr1", Some(&ref_seq), 2, 1, false)
                .unwrap();

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(r1.gt, vec![-1, 1]);
        assert!(src.next_record().unwrap().is_none());
    }

    #[test]
    fn file_without_contig_is_skipped() {
        let tmp = tempdir().unwrap();
        // file A: chr1 POS3 (0-based pos 2) A>G, GT 1|0.
        let a = write_ss_vcf(
            tmp.path(),
            "a",
            "chr1",
            100,
            "SA",
            &[SsRec {
                pos: 2,
                r: b"A",
                alts: vec![b"G"],
                gt: vec![1, 0],
            }],
        );
        // file B: only chr2 records — has no "chr1" contig at all.
        let b_path = tmp.path().join("b.bcf");
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chr2,length=100>");
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_sample(b"SB");
        {
            let mut writer =
                Writer::from_path(&b_path, &header, false, Format::Bcf).expect("open writer");
            let mut record = writer.empty_record();
            record.set_rid(Some(0));
            record.set_pos(10);
            record.set_alleles(&[b"A", b"C"]).unwrap();
            record
                .push_genotypes(&[GenotypeAllele::Phased(1), GenotypeAllele::Phased(1)])
                .unwrap();
            writer.write(&record).unwrap();
        }
        rust_htslib::bcf::index::build(&b_path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
            .expect("build BCF index");

        let ref_seq = make_ref(tmp.path(), "chr1", 100, &[(2, b"A")]);

        let paths = vec![
            a.to_str().unwrap().to_string(),
            b_path.to_str().unwrap().to_string(),
        ];
        let samples = vec!["SA", "SB"];
        let mut src =
            VcfListRecordSource::new(&paths, &samples, "chr1", Some(&ref_seq), 2, 1, false)
                .unwrap();

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(r1.gt, vec![1, 0, /* SB, filled hom-ref */ 0, 0]);
        assert!(src.next_record().unwrap().is_none());
    }
}

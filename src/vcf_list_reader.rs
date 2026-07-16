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
//! `chunk_assembler`. INFO fields resolve first-carrier-wins (the lowest-numbered
//! column carrying the atom supplies the value); FORMAT fields are per-sample,
//! each carrier contributing its own file's value and non-carriers falling back
//! to the field's default downstream.

use crate::chunk_assembler::resolve_scalar;
use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec};
use crate::normalize::{L_MAX, atomize_to_vcf_biallelic};
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
    /// This atom's INFO values, one per requested INFO `FieldSpec` (spec
    /// order), already resolved via `resolve_scalar` against this file's raw
    /// record buffer and this atom's `source_alt_index`.
    info_vals: Vec<f64>,
    /// This atom's FORMAT values for this file's single sample, one per
    /// requested FORMAT `FieldSpec` (spec order), resolved the same way.
    format_vals: Vec<f64>,
}

/// Reject a record whose position precedes the file's own last-read position.
/// Factored out of `FileCursor::advance` so it's directly unit-testable: a
/// real integration-level fixture would need a deliberately stale/mismatched
/// index, since htslib's own CSI/tabix indexer refuses to build a valid index
/// over genuinely unsorted records (`hts_idx_push` rejects them at build
/// time) -- so a whole-file-unsorted BCF can't be constructed through the
/// normal write+index helpers this test module otherwise uses.
fn check_position_sorted(path: &str, prev: Option<u32>, pos: u32) -> Result<(), ConversionError> {
    if let Some(prev) = prev
        && pos < prev
    {
        return Err(ConversionError::Input(format!(
            "{path}: records are not position-sorted (pos {pos} follows pos {prev}) -- \
             from_vcf_list requires every input file's records to already be \
             position-sorted per contig; sort it first (e.g. `bcftools sort`) and retry"
        )));
    }
    Ok(())
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
    /// Records excluded because their REF disagreed with the reference under
    /// `CheckRef::Exclude`.
    ref_excluded: u64,
    /// This file's own path, kept only for error messages (unsorted-input
    /// detection, cross-file REF disagreement) that need to name the
    /// offending file.
    path: String,
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
        check_ref: crate::normalize::CheckRef,
        info_specs: &[FieldSpec],
        format_specs: &[FieldSpec],
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
                    // The merge is correct only if every input is
                    // position-sorted per contig -- one comparison per
                    // record turns an unsorted file (which would otherwise
                    // silently corrupt the k-way merge, poisoning the WHOLE
                    // output store with no way to identify which of N files
                    // was at fault) into a clear, named error instead.
                    check_position_sorted(&self.path, self.frontier, rec.pos)?;
                    self.frontier = Some(rec.pos);
                    if let Some(rs) = ref_seq {
                        match crate::normalize::apply_check_ref(
                            check_ref,
                            rec.pos,
                            &rec.reference,
                            rs,
                        )? {
                            crate::normalize::RefDecision::Keep => {}
                            crate::normalize::RefDecision::Exclude(e) => {
                                self.ref_excluded += 1;
                                if self.ref_excluded == 1 {
                                    println!(
                                        "Notice: check_ref=x excluding record(s) in {} \
                                         whose REF disagrees with the reference (first: {e}).",
                                        self.path
                                    );
                                }
                                // This record contributes no atoms; fall through to
                                // the buffer pop, exactly like an all-`*`/dropped ALT
                                // record — the caller re-advances.
                                return Ok(self.buf.pop_front());
                            }
                        }
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

                        // Resolve this atom's field values against THIS file's raw
                        // record buffer now, while `rec.info_raw`/`rec.format_raw`
                        // (and `br.source_alt_index`) are still in scope — the merge
                        // heap only ever sees the pre-resolved scalars from here on.
                        let info_vals: Vec<f64> = info_specs
                            .iter()
                            .zip(rec.info_raw.iter())
                            .map(|(spec, raw)| {
                                resolve_scalar(raw.as_deref(), br.source_alt_index, spec)
                            })
                            .collect();
                        // Single sample per file ⇒ always inner index 0.
                        let format_vals: Vec<f64> = format_specs
                            .iter()
                            .zip(rec.format_raw.iter())
                            .map(|(spec, raw)| {
                                let sample_vals = raw.as_ref().map(|v| v[0].as_slice());
                                resolve_scalar(sample_vals, br.source_alt_index, spec)
                            })
                            .collect();

                        self.buf.push_back(NormAtom {
                            pos: br.pos,
                            ilen,
                            reference,
                            alt,
                            ploid_codes,
                            info_vals,
                            format_vals,
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
/// deterministic for tests). The key is read straight off the `atom` rather
/// than stored alongside it: the atom already owns `pos`/`ilen`/`alt`, so a
/// separate key would duplicate the (`alt`) allocation on every one of the
/// millions of atoms pushed.
struct HeapEntry {
    col: usize,
    atom: NormAtom,
}

impl HeapEntry {
    /// The merge key `(pos, ilen, alt)` as borrowed fields — no allocation.
    #[inline]
    fn key(&self) -> (u32, i32, &[u8]) {
        (self.atom.pos, self.atom.ilen, &self.atom.alt)
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key() && self.col == other.col
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
        self.key().cmp(&other.key()).then(self.col.cmp(&other.col))
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
    check_ref: crate::normalize::CheckRef,
    num_samples: usize,
    info_specs: Vec<FieldSpec>,
    format_specs: Vec<FieldSpec>,
}

impl VcfListRecordSource {
    /// `vcf_paths[i]` is a single-sample file whose sample is `samples[i]` and
    /// whose merged column is `i`. `ref_seq` is the contig sequence (`None` ⇒
    /// no-reference mode). Files without `chrom` in their header are skipped
    /// (their column is filled hom-ref for every merged record).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vcf_paths: &[String],
        samples: &[&str],
        chrom: &str,
        ref_seq: Option<&[u8]>,
        ploidy: usize,
        htslib_threads: usize,
        skip_out_of_scope: bool,
        check_ref: crate::normalize::CheckRef,
        fields: &[FieldSpec],
    ) -> Result<Self, ConversionError> {
        if vcf_paths.len() != samples.len() {
            return Err(ConversionError::Input(format!(
                "vcf_paths and samples must be parallel (one sample per file): \
                 got {} paths and {} samples",
                vcf_paths.len(),
                samples.len()
            )));
        }
        let mut cursors = Vec::with_capacity(vcf_paths.len());
        for (col, (path, &sample)) in vcf_paths.iter().zip(samples.iter()).enumerate() {
            let single_sample = [sample];
            match VcfRecordSource::new(
                path,
                chrom,
                &single_sample,
                htslib_threads,
                ploidy,
                fields,
                Vec::new(),
            ) {
                Ok(vcf) => cursors.push(FileCursor {
                    vcf: Some(vcf),
                    buf: VecDeque::new(),
                    frontier: None,
                    eof: false,
                    col,
                    dropped: 0,
                    ref_excluded: 0,
                    path: path.clone(),
                }),
                // This file's header simply doesn't declare `chrom` at all --
                // matched on the typed variant (not a message substring, which
                // would silently stop matching if `vcf_reader.rs`'s wording
                // ever changed) -- skip it, filling its column hom-ref for
                // every merged record on this contig.
                Err(ConversionError::ContigNotInHeader { .. }) => {
                    cursors.push(FileCursor {
                        vcf: None,
                        buf: VecDeque::new(),
                        frontier: None,
                        eof: true,
                        col,
                        dropped: 0,
                        ref_excluded: 0,
                        path: path.clone(),
                    });
                }
                Err(e) => return Err(e),
            }
        }
        let info_specs: Vec<FieldSpec> = fields
            .iter()
            .filter(|f| f.category == FieldCategory::Info)
            .cloned()
            .collect();
        let format_specs: Vec<FieldSpec> = fields
            .iter()
            .filter(|f| f.category == FieldCategory::Format)
            .cloned()
            .collect();
        Ok(Self {
            cursors,
            heap: BinaryHeap::new(),
            ref_seq: ref_seq.map(|s| s.to_vec()),
            ploidy,
            skip_out_of_scope,
            check_ref,
            num_samples: vcf_paths.len(),
            info_specs,
            format_specs,
        })
    }

    /// Total out-of-scope (symbolic/breakend) ALTs dropped across every file so
    /// far. Valid after the read loop drains.
    pub fn dropped_out_of_scope(&self) -> u64 {
        self.cursors.iter().map(|c| c.dropped).sum()
    }

    /// Total REF-mismatch records excluded across every file so far
    /// (`CheckRef::Exclude`). Valid after the read loop drains.
    pub fn ref_excluded(&self) -> u64 {
        self.cursors.iter().map(|c| c.ref_excluded).sum()
    }

    /// Number of atoms currently buffered in the merge heap. Test-only: used to
    /// prove the streaming frontier release actually gates on the frontier
    /// rather than buffering the whole contig before ever releasing a record
    /// (see `streaming_release_before_full_drain`).
    #[cfg(test)]
    fn heap_len(&self) -> usize {
        self.heap.len()
    }

    // Build the merged `RawRecord` for a group of same-key `(col, NormAtom)`
    // pairs: one biallelic record whose REF/ALT come from the first atom (they
    // agree by construction — same key), with every column not present in the
    // group left hom-ref (`0`, the `gt` vec's fill value).
    fn build_record(
        &self,
        key: &(u32, i32, Vec<u8>),
        group: Vec<(usize, NormAtom)>,
    ) -> Result<RawRecord, ConversionError> {
        let reference = group[0].1.reference.clone();
        // The merge key is `(pos, ilen, alt)` and excludes REF.
        if self.ref_seq.is_some() {
            // `validate_ref` ran for every atom against the SAME reference,
            // so cross-file REF disagreement at a matching key is provably
            // impossible here -- keep only a debug-build sanity net rather
            // than paying a real check in release builds.
            #[cfg(debug_assertions)]
            for (_, atom) in &group {
                debug_assert_eq!(
                    atom.reference, reference,
                    "cross-file REF disagreement at pos {} despite matching merge key {:?} \
                     (unexpected: validate_ref should make this impossible when a reference \
                     is supplied)",
                    key.0, key
                );
            }
        } else {
            // No reference: `validate_ref` never ran, so two files CAN
            // genuinely disagree about REF at the same `(pos, ilen, alt)`
            // key. Silently taking `group[0]`'s REF (as a debug-only assert
            // would let happen in release builds) would corrupt every other
            // column's REF at this atom -- make it a real, named error in
            // every build, not just a debug assertion.
            for (col, atom) in &group {
                if atom.reference != reference {
                    return Err(ConversionError::Input(format!(
                        "cross-file REF disagreement at pos {} (atom key {:?}): file {:?} \
                         (column {}) has REF {:?} but file {:?} (column {}) has REF {:?} -- \
                         no_reference mode has no reference to arbitrate which is correct; \
                         supply a `reference` FASTA, or ensure every input file agrees on \
                         REF at this position",
                        key.0,
                        key,
                        self.cursors[group[0].0].path,
                        group[0].0,
                        String::from_utf8_lossy(&reference),
                        self.cursors[*col].path,
                        col,
                        String::from_utf8_lossy(&atom.reference),
                    )));
                }
            }
        }
        let alt = group[0].1.alt.clone();
        let mut gt = vec![0i32; self.num_samples * self.ploidy];
        for (col, atom) in &group {
            let base = col * self.ploidy;
            gt[base..base + self.ploidy].copy_from_slice(&atom.ploid_codes);
        }

        // INFO: first-carrier wins. `group` is popped off the heap in
        // ascending `(key, col)` order (see `HeapEntry`'s `Ord`), so
        // `group[0]` is always the lowest-numbered column carrying this atom
        // — the earliest file in list order — regardless of which column's
        // atom happened to be on top of the heap.
        let info_raw: Vec<Option<Vec<f64>>> = (0..self.info_specs.len())
            .map(|i| Some(vec![group[0].1.info_vals[i]]))
            .collect();

        // FORMAT: per-sample. Each carrier column supplies its own file's
        // value; a non-carrier column is left an empty buffer, which
        // `resolve_scalar` downstream (in `ChunkAssembler`) resolves to the
        // field's default — same contract as a record-absent field.
        let format_raw: Vec<Option<Vec<Vec<f64>>>> = (0..self.format_specs.len())
            .map(|j| {
                let mut per_sample: Vec<Vec<f64>> = vec![Vec::new(); self.num_samples];
                for (col, atom) in &group {
                    per_sample[*col] = vec![atom.format_vals[j]];
                }
                Some(per_sample)
            })
            .collect();

        Ok(RawRecord {
            pos: key.0,
            reference,
            alts: vec![alt],
            gt,
            info_raw,
            format_raw,
        })
    }
}

impl RecordSource for VcfListRecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        loop {
            // One pass over the cursors computes every liveness aggregate AND
            // the advance target at once. This inner `loop` runs once per atom
            // read plus once per record emitted, so at N files it was
            // O((atoms + records) * N) with a fresh `Vec<Option<u32>>`
            // allocated every iteration (both the top-of-loop `live_frontiers`
            // collect and the separate `min_by_key` argmin scan showed up as
            // the dominant cost under `perf`). Fusing them keeps the algorithm
            // O(N) per iteration but drops the per-iteration allocation and
            // three of the four scans. (A frontier heap would make selection
            // O(log N) — deferred; the constant-factor win here is the easy
            // one and leaves behaviour identical.)
            let mut any_live = false;
            let mut all_started = true;
            let mut min_started: Option<u32> = None;
            // The live cursor with the smallest frontier, `None` (unstarted)
            // sorting before every `Some(_)` — matches the old
            // `min_by_key(|c| c.frontier)`, including its first-min-wins
            // tiebreak (strict `<` keeps the earliest index on ties).
            let mut advance_idx: Option<usize> = None;
            let mut advance_key: Option<Option<u32>> = None;
            for (i, c) in self.cursors.iter().enumerate() {
                if c.eof {
                    continue;
                }
                any_live = true;
                match c.frontier {
                    None => all_started = false,
                    Some(f) => min_started = Some(min_started.map_or(f, |m| m.min(f))),
                }
                if advance_key.is_none_or(|best| c.frontier < best) {
                    advance_key = Some(c.frontier);
                    advance_idx = Some(i);
                }
            }
            let all_eof = !any_live;

            // Unstarted cursors (`frontier == None`) must be read before anything
            // can release — otherwise a not-yet-opened file's first record could
            // sort below an already-released key, corrupting the merge order.
            let releasable = all_eof
                || (all_started
                    && self.heap.peek().is_some_and(|Reverse(e)| {
                        min_started.is_some_and(|m| e.atom.pos < m.saturating_sub(L_MAX))
                    }));

            if releasable {
                return match self.heap.pop() {
                    None => Ok(None),
                    Some(Reverse(top)) => {
                        let mut group = vec![(top.col, top.atom)];
                        // Group-drain by the front atom's own key (borrowed, no
                        // clone): pop every heap entry sharing `(pos, ilen,
                        // alt)` with `group[0]`.
                        while let Some(Reverse(next)) = self.heap.peek() {
                            let g0 = &group[0].1;
                            if next.atom.pos != g0.pos
                                || next.atom.ilen != g0.ilen
                                || next.atom.alt != g0.alt
                            {
                                break;
                            }
                            let Reverse(next) = self.heap.pop().unwrap();
                            group.push((next.col, next.atom));
                        }
                        // Materialize the key once per released group (not once
                        // per atom) for `build_record`'s pos/error-message use.
                        let g0 = &group[0].1;
                        let key = (g0.pos, g0.ilen, g0.alt.clone());
                        self.build_record(&key, group).map(Some)
                    }
                };
            }

            // Advance the live cursor with the smallest frontier (computed in
            // the single pass above).
            match advance_idx {
                Some(i) => {
                    let ref_seq = self.ref_seq.as_deref();
                    if let Some(atom) = self.cursors[i].advance(
                        ref_seq,
                        self.ploidy,
                        self.skip_out_of_scope,
                        self.check_ref,
                        &self.info_specs,
                        &self.format_specs,
                    )? {
                        // The heap orders by the atom's own `(pos, ilen, alt)`
                        // (see `HeapEntry::key`) — no separate key to allocate.
                        let col = self.cursors[i].col;
                        self.heap.push(Reverse(HeapEntry { col, atom }));
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
    use crate::normalize::CheckRef;
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
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            Some(&ref_seq),
            2,
            1,
            false,
            CheckRef::Error,
            &[],
        )
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
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            Some(&ref_seq),
            2,
            1,
            false,
            CheckRef::Error,
            &[],
        )
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
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            Some(&ref_seq),
            2,
            1,
            false,
            CheckRef::Error,
            &[],
        )
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
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            Some(&ref_seq),
            2,
            1,
            false,
            CheckRef::Error,
            &[],
        )
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
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            Some(&ref_seq),
            2,
            1,
            false,
            CheckRef::Error,
            &[],
        )
        .unwrap();

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(r1.gt, vec![1, 0, /* SB, filled hom-ref */ 0, 0]);
        assert!(src.next_record().unwrap().is_none());
    }

    // Every other test in this file uses `chrom_len = 100` with variants at pos
    // 2/6/10. Since `L_MAX = 1000`, the frontier gate
    // (`min_started.saturating_sub(L_MAX)`) is *always* 0 for those tests, so
    // `releasable`'s streaming branch never actually fires — every record ends
    // up released through the `all_eof` drain instead. This test spreads sites
    // out past `L_MAX` so the frontier gate is exercised for real, with a
    // third file (SC) that hits EOF early while SA/SB are still mid-contig.
    #[test]
    fn streaming_release_before_full_drain() {
        let tmp = tempdir().unwrap();
        let chrom_len: usize = 5000;

        // file A: pos 10, 1500, 4000 — all SNPs A>G.
        let a = write_ss_vcf(
            tmp.path(),
            "a",
            "chr1",
            chrom_len as u64,
            "SA",
            &[
                SsRec {
                    pos: 10,
                    r: b"A",
                    alts: vec![b"G"],
                    gt: vec![1, 0],
                },
                SsRec {
                    pos: 1500,
                    r: b"A",
                    alts: vec![b"G"],
                    gt: vec![1, 0],
                },
                SsRec {
                    pos: 4000,
                    r: b"A",
                    alts: vec![b"G"],
                    gt: vec![1, 0],
                },
            ],
        );
        // file B: pos 1500 (the SAME atom as A's, must merge into one record)
        // and pos 4000 as a DEL — distinct `ilen` from A's pos-4000 SNP, so it
        // must NOT merge with A's atom there.
        let b = write_ss_vcf(
            tmp.path(),
            "b",
            "chr1",
            chrom_len as u64,
            "SB",
            &[
                SsRec {
                    pos: 1500,
                    r: b"A",
                    alts: vec![b"G"],
                    gt: vec![0, 1],
                },
                SsRec {
                    pos: 4000,
                    r: b"AT",
                    alts: vec![b"A"],
                    gt: vec![0, 1],
                },
            ],
        );
        // file C: pos 10 only — hits EOF while A and B are still live, well
        // before their pos-1500/4000 atoms have even been read.
        let c = write_ss_vcf(
            tmp.path(),
            "c",
            "chr1",
            chrom_len as u64,
            "SC",
            &[SsRec {
                pos: 10,
                r: b"A",
                alts: vec![b"G"],
                gt: vec![1, 1],
            }],
        );
        let ref_seq = make_ref(
            tmp.path(),
            "chr1",
            chrom_len,
            &[(10, b"A"), (1500, b"A"), (4000, b"AT")],
        );

        let paths = vec![
            a.to_str().unwrap().to_string(),
            b.to_str().unwrap().to_string(),
            c.to_str().unwrap().to_string(),
        ];
        let samples = vec!["SA", "SB", "SC"];
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            Some(&ref_seq),
            2,
            1,
            false,
            CheckRef::Error,
            &[],
        )
        .unwrap();

        // Record 1: pos 10 — SA + SC carry, SB filled hom-ref.
        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 10);
        assert_eq!(r1.alts, vec![b"G".to_vec()]);
        assert_eq!(r1.gt, vec![/* SA */ 1, 0, /* SB */ 0, 0, /* SC */ 1, 1]);

        // Discriminator: this record can only have been released through the
        // FRONTIER gate, not the `all_eof` drain — SA and SB are still live
        // (only SC has hit EOF). If release instead required draining every
        // cursor to EOF first (i.e. `releasable` degenerated to `all_eof`),
        // every atom in the contig (6 total: SA@10/1500/4000, SB@1500/4000,
        // SC@10) would already have been pulled from disk and sitting in the
        // heap by the time the first record was released, leaving 4 behind
        // after popping the pos-10 group. Under real streaming release, SA
        // and SB haven't been read past pos 1500 yet — their pos-4000 atoms
        // haven't even been fetched — so only the two pos-1500 atoms remain
        // buffered.
        assert_eq!(
            src.heap_len(),
            2,
            "expected only the two pos-1500 atoms buffered right after the first \
             release; a heap_len of 4 here means the whole contig was buffered \
             before any release happened, i.e. the streaming frontier gate isn't \
             actually gating anything"
        );

        // Record 2: pos 1500 — SA + SB merge into ONE record, SC hom-ref.
        let r2 = src.next_record().unwrap().unwrap();
        assert_eq!(r2.pos, 1500);
        assert_eq!(r2.alts, vec![b"G".to_vec()]);
        assert_eq!(r2.gt, vec![/* SA */ 1, 0, /* SB */ 0, 1, /* SC */ 0, 0]);

        // Records 3 & 4: pos 4000 stays as TWO separate records — SB's DEL
        // (ilen -1) sorts before SA's SNP (ilen 0) under the `(pos, ilen,
        // alt)` key even though both are at the same position, exercising the
        // full-key group-drain across columns.
        let r3 = src.next_record().unwrap().unwrap();
        assert_eq!(r3.pos, 4000);
        assert_eq!(r3.reference, b"AT".to_vec());
        assert_eq!(r3.alts, vec![b"A".to_vec()]);
        assert_eq!(r3.gt, vec![/* SA */ 0, 0, /* SB */ 0, 1, /* SC */ 0, 0]);

        let r4 = src.next_record().unwrap().unwrap();
        assert_eq!(r4.pos, 4000);
        assert_eq!(r4.reference, b"A".to_vec());
        assert_eq!(r4.alts, vec![b"G".to_vec()]);
        assert_eq!(r4.gt, vec![/* SA */ 1, 0, /* SB */ 0, 0, /* SC */ 0, 0]);

        // SC's early EOF didn't strand or truncate anything downstream.
        assert!(src.next_record().unwrap().is_none());
        assert_eq!(src.dropped_out_of_scope(), 0);
    }

    // Write a single-sample, CSI-indexed BCF declaring `INFO AF` (Number=A,
    // Float) and `FORMAT GT:DP` (DP Number=1, Integer), with per-record
    // AF/DP values supplied alongside the usual pos/REF/ALT/GT. Only used by
    // the field-carry-through test below — every other test in this module
    // has no fields declared at all.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn write_ss_vcf_with_fields(
        dir: &Path,
        name: &str,
        chrom: &str,
        chrom_len: u64,
        sample: &str,
        // (pos, ref, alts, gt, INFO AF, FORMAT DP)
        records: &[(i64, &'static [u8], Vec<&'static [u8]>, Vec<i32>, f32, i32)],
    ) -> PathBuf {
        let path = dir.join(format!("{name}.bcf"));
        let mut header = Header::new();
        header.push_record(format!("##contig=<ID={chrom},length={chrom_len}>").as_bytes());
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_record(b"##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">");
        header.push_record(b"##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">");
        header.push_sample(sample.as_bytes());
        {
            let mut writer =
                Writer::from_path(&path, &header, false, Format::Bcf).expect("open BCF writer");
            for (pos, r, alts, gt, af, dp) in records {
                let mut record = writer.empty_record();
                record.set_rid(Some(0));
                record.set_pos(*pos);
                let mut alleles: Vec<&[u8]> = Vec::with_capacity(1 + alts.len());
                alleles.push(r);
                alleles.extend(alts.iter().copied());
                record.set_alleles(&alleles).expect("set alleles");
                let gt_alleles: Vec<GenotypeAllele> = gt
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
                record.push_info_float(b"AF", &[*af]).expect("push AF");
                record.push_format_integer(b"DP", &[*dp]).expect("push DP");
                writer.write(&record).expect("write record");
            }
        }
        rust_htslib::bcf::index::build(&path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
            .expect("build BCF index");
        path
    }

    #[test]
    fn info_first_carrier_and_format_per_sample_with_non_carrier_default() {
        // Two files: a shared SNP@pos2 that BOTH carry (proves INFO
        // first-carrier-wins across two DIFFERENT AF values, and FORMAT
        // per-sample carrying each file's own DP), plus a second SNP@pos20
        // that ONLY file A carries (proves the non-carrier column's
        // `format_raw` entry is the empty `vec![]` the design contract
        // promises -- NOT file A's carrier value leaking across columns,
        // and NOT some other placeholder).
        //
        // This is deliberately tested at the `RawRecord` layer rather than
        // through `SparseVar2.decode()`: `decode()`'s Ragged output only
        // ever emits ALT-carrying (sample, ploid) cells (see
        // `test_from_vcf_list_missing_hap_is_unobservable_in_decode` for the
        // analogous genotype-side limitation), so a non-carrier column's
        // resolved value never surfaces through it -- the `RawRecord` this
        // merge produces is the only place the empty-vs-populated buffer
        // contract is directly observable.
        let tmp = tempdir().unwrap();
        let a = write_ss_vcf_with_fields(
            tmp.path(),
            "a",
            "chr1",
            100,
            "SA",
            &[
                (2, b"A", vec![b"G"], vec![1, 0], 0.1, 10),  // shared w/ B
                (20, b"A", vec![b"C"], vec![1, 1], 0.4, 77), // A-only
            ],
        );
        let b = write_ss_vcf_with_fields(
            tmp.path(),
            "b",
            "chr1",
            100,
            "SB",
            &[(2, b"A", vec![b"G"], vec![0, 1], 0.9, 20)], // shared w/ A
        );
        let ref_seq = make_ref(tmp.path(), "chr1", 100, &[(2, b"A"), (20, b"A")]);

        let paths = vec![
            a.to_str().unwrap().to_string(),
            b.to_str().unwrap().to_string(),
        ];
        let samples = vec!["SA", "SB"];
        let fields = crate::field::parse_manifest(vec![
            (
                "AF".to_string(),
                "info".to_string(),
                "float".to_string(),
                None,
                None,
            ),
            (
                "DP".to_string(),
                "format".to_string(),
                "int".to_string(),
                None,
                None,
            ),
        ])
        .unwrap();
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            Some(&ref_seq),
            2,
            1,
            false,
            CheckRef::Error,
            &fields,
        )
        .unwrap();

        // pos2: both files carry it -- INFO first-carrier-wins picks SA's
        // 0.1, NOT SB's 0.9 (nor e.g. their max/last). FORMAT is per-sample:
        // SA's own DP (10) and SB's own DP (20), each in its own column.
        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(
            r1.info_raw,
            // htslib stores AF as f32; the raw buffer is widened to f64
            // verbatim (no rounding), so the expected value must go through
            // the same f32 round-trip rather than a literal f64 `0.1`.
            vec![Some(vec![0.1_f32 as f64])],
            "INFO AF must be SA's (first-carrier/col0) value 0.1, not SB's 0.9"
        );
        assert_eq!(
            r1.format_raw,
            vec![Some(vec![vec![10.0], vec![20.0]])],
            "FORMAT DP must be per-sample: SA's own 10, SB's own 20"
        );

        // pos20: only SA carries it -- SB's column is a non-member (not just
        // a genotype non-carrier), so its FORMAT buffer must be the empty
        // `vec![]` the design contract calls for (-> downstream default),
        // not SA's 77 leaking across columns.
        let r2 = src.next_record().unwrap().unwrap();
        assert_eq!(r2.pos, 20);
        assert_eq!(r2.info_raw, vec![Some(vec![0.4_f32 as f64])]);
        assert_eq!(
            r2.format_raw,
            vec![Some(vec![vec![77.0], vec![]])],
            "SB is not a member at pos20 -- its FORMAT DP buffer must be \
             empty, not SA's carrier value"
        );

        assert!(src.next_record().unwrap().is_none());
    }

    #[test]
    fn check_position_sorted_rejects_a_decrease() {
        // Mirrors what `FileCursor::advance` feeds it: this file's own
        // last-read position (20, its `frontier`) followed by a smaller one
        // (5) -- the exact shape of an unsorted input file, unit-tested
        // directly since htslib's own CSI/tabix indexer refuses to build a
        // valid index over genuinely unsorted on-disk records (see the
        // function's doc comment), so this can't be exercised end-to-end
        // through the write+index test helpers above.
        let err = check_position_sorted("bad.bcf", Some(20), 5).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("bad.bcf"),
            "error must name the offending file path, got: {msg}"
        );
        assert!(
            msg.contains("20") && msg.contains('5'),
            "error must name the offending positions, got: {msg}"
        );
        assert!(
            msg.contains("sort"),
            "error should point at a remedy (sorting), got: {msg}"
        );
    }

    #[test]
    fn check_position_sorted_accepts_ties_and_increases() {
        // No prior frontier (first record of the file): always fine.
        assert!(check_position_sorted("f.bcf", None, 5).is_ok());
        // Equal position (multiple atoms/records can legitimately share a
        // POS, e.g. a multiallelic split): not a decrease, so fine.
        assert!(check_position_sorted("f.bcf", Some(5), 5).is_ok());
        // Strictly increasing: the ordinary case.
        assert!(check_position_sorted("f.bcf", Some(5), 6).is_ok());
    }

    #[test]
    fn no_reference_cross_file_ref_disagreement_errors() {
        // file A and B both claim a variant at the SAME (pos, ilen, alt) key
        // ("A>G" at pos 2) but disagree about REF ("A" vs "C") -- with
        // `ref_seq: None`, `validate_ref` never runs, so nothing upstream of
        // `build_record` catches this. Must be a real `ConversionError` in
        // every build (not merely a `debug_assert`), naming both files, both
        // RE-Fs, and the position -- silently taking file A's REF for file
        // B's column would otherwise corrupt file B's row with no signal.
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
                r: b"C",
                alts: vec![b"G"],
                gt: vec![0, 1],
            }],
        );

        let paths = vec![
            a.to_str().unwrap().to_string(),
            b.to_str().unwrap().to_string(),
        ];
        let samples = vec!["SA", "SB"];
        // ref_seq: None -- no_reference mode, exercising the branch validate_ref
        // never guards.
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            None,
            2,
            1,
            false,
            CheckRef::Error,
            &[],
        )
        .unwrap();

        // `RawRecord` isn't `Debug` (see `record_source.rs`), so `unwrap_err`
        // isn't available on `Result<Option<RawRecord>, ConversionError>` --
        // match it out instead.
        let err = match src.next_record() {
            Err(e) => e,
            Ok(_) => panic!("expected an error, got Ok"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains(a.to_str().unwrap()) && msg.contains(b.to_str().unwrap()),
            "error must name both conflicting files, got: {msg}"
        );
        assert!(
            msg.to_uppercase().contains('A') && msg.contains('C'),
            "error must name both conflicting REF bytes, got: {msg}"
        );
    }

    #[test]
    fn contig_in_header_but_no_records_is_hom_ref_filled_not_errored() {
        // The MOST common real shape for `from_vcf_list`'s inputs: N
        // per-sample VCFs sharing one pipeline header (so every file's header
        // declares every contig), where a given sample simply has no variant
        // calls on this particular contig. This is a DIFFERENT code path
        // than `file_without_contig_is_skipped` above (which omits the
        // contig from the header entirely, failing at `name2rid`): here
        // `name2rid` and `fetch` both succeed, and the FIRST `next_record()`
        // call returns `None` immediately. Must still hom-ref-fill, not
        // error.
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
        // file B: header declares chr1 (shared pipeline header) but has ZERO
        // records on it.
        let b_path = tmp.path().join("b.bcf");
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chr1,length=100>");
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_sample(b"SB");
        {
            let writer =
                Writer::from_path(&b_path, &header, false, Format::Bcf).expect("open writer");
            drop(writer); // no records written at all
        }
        rust_htslib::bcf::index::build(&b_path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
            .expect("build BCF index");

        let ref_seq = make_ref(tmp.path(), "chr1", 100, &[(2, b"A")]);

        let paths = vec![
            a.to_str().unwrap().to_string(),
            b_path.to_str().unwrap().to_string(),
        ];
        let samples = vec!["SA", "SB"];
        let mut src = VcfListRecordSource::new(
            &paths,
            &samples,
            "chr1",
            Some(&ref_seq),
            2,
            1,
            false,
            CheckRef::Error,
            &[],
        )
        .unwrap();

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(r1.gt, vec![1, 0, /* SB, hom-ref filled */ 0, 0]);
        assert!(src.next_record().unwrap().is_none());
        assert_eq!(src.dropped_out_of_scope(), 0);
    }
}

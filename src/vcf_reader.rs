use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec, HtslibType};
use crate::normalize::atomize_record;
use crate::types::{BitGrid3, DenseChunk, StagedColumn};
use rayon::prelude::*;
use rust_htslib::bcf::record::Record;
use rust_htslib::bcf::{IndexedReader, Read};
use rust_htslib::errors::Error as HtslibError;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

// A decomposed atom awaiting emission. Carries a shared handle to its source record's
// per-column allele indices so genotype presence is computed at chunk-build time.
struct PendingAtom {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    source_alt_index: u16,
    gt: Arc<Vec<i32>>, // len = num_samples * ploidy; allele index per column (-1 = missing)
    seq: u64,          // stable tiebreak for equal positions

    // Resolved field values for THIS atom (already indexed by source_alt_index
    // where the underlying VCF field is Number=A). Populated in
    // `decompose_current_record`, gathered into `DenseChunk::{info,format}_staged`
    // in `read_next_chunk`'s sequential metadata pass. Empty when no fields of
    // that category were requested.
    info_vals: Vec<f64>,   // len == VcfChunkReader::info_fields.len()
    format_vals: Vec<f64>, // len == VcfChunkReader::format_fields.len() * num_samples, field-major then sample: field*num_samples + sample
}

impl PartialEq for PendingAtom {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.seq == other.seq
    }
}
impl Eq for PendingAtom {}
impl PartialOrd for PendingAtom {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PendingAtom {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pos.cmp(&other.pos).then(self.seq.cmp(&other.seq))
    }
}

// Pack variant row `vi`'s presence bits into `words`, where `words[0]` corresponds
// to global word index `word_base`. Bit for (row vi, column col) lives at global
// flat index `vi*columns + col`; the local word index subtracts `word_base`.
// Presence is `gt[col] == source_alt_index`. Bits start zeroed and are only OR-set,
// and each word is assembled in a register and written once (identical result to a
// per-bit `or_bit` loop, far fewer stores).
#[inline]
fn pack_row(words: &mut [u64], word_base: usize, vi: usize, a: &PendingAtom, columns: usize) {
    let src = a.source_alt_index as i32;
    let gtc: &[i32] = &a.gt;
    let base = vi * columns;
    let mut col = 0usize;
    while col < columns {
        let flat = base + col;
        let w = (flat >> 6) - word_base;
        let b = flat & 63;
        let n = (64 - b).min(columns - col);
        let mut acc = 0u64;
        for k in 0..n {
            // SAFETY: col + k < columns == gtc.len().
            acc |= ((unsafe { *gtc.get_unchecked(col + k) } == src) as u64) << (b + k);
        }
        // SAFETY: w indexes a word within this row's target slice.
        unsafe {
            *words.get_unchecked_mut(w) |= acc;
        }
        col += n;
    }
}

// Sequential full-grid presence packing: one row at a time into the whole `words`
// slice (global word index == local word index, so `word_base == 0`).
fn pack_presence_seq(words: &mut [u64], atoms: &[PendingAtom], columns: usize) {
    for (vi, a) in atoms.iter().enumerate() {
        pack_row(words, 0, vi, a, columns);
    }
}

// Below this many variants in a chunk, parallel packing's per-task overhead
// outweighs the win — pack sequentially instead. Tunable; measure on gdc/germline.
const PARALLEL_MIN_VARIANTS: usize = 512;

#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

// Parallel presence packing. Variants are partitioned into word-aligned blocks:
// row `vi` occupies bits `[vi*columns, (vi+1)*columns)`, so a block boundary at a
// multiple of `g = 64/gcd(columns,64)` variants lands exactly on a u64 boundary.
// `par_chunks_mut(words_per_block)` hands each rayon task a word-DISJOINT slice, so
// there are no shared boundary words and no atomics — the result is bit-identical to
// `pack_presence_seq`. Block `c` covers variants `[c*g, min((c+1)*g, v))` and words
// `[c*words_per_block, ...)`, whose global base is `word_base = c*words_per_block`.
fn pack_presence_par(
    words: &mut [u64],
    atoms: &[PendingAtom],
    columns: usize,
    pool: &rayon::ThreadPool,
) {
    let d = gcd(columns, 64);
    let g = 64 / d; // variants per word-aligned block
    let words_per_block = columns / d; // == g * columns / 64, always an integer
    let v = atoms.len();

    pool.install(|| {
        words
            .par_chunks_mut(words_per_block)
            .enumerate()
            .for_each(|(c, wchunk)| {
                let vi_start = c * g;
                let vi_end = ((c + 1) * g).min(v);
                let word_base = c * words_per_block;
                // `vi` is dual-purpose here: it's both the `atoms` index and the row
                // index `pack_row` needs to compute the flat bit offset, so it can't
                // be replaced by a plain iterator/enumerate.
                #[allow(clippy::needless_range_loop)]
                for vi in vi_start..vi_end {
                    pack_row(wchunk, word_base, vi, &atoms[vi], columns);
                }
            });
    });
}

// Delegates to `FieldSpec::missing_sentinel` — single source of truth shared
// with `dense2sparse_vk`'s genotype-aligned non-carrier fill (src/rvk.rs).
fn sentinel_default(spec: &FieldSpec) -> f64 {
    spec.missing_sentinel()
}

// htslib missing-value detection on an already-widened f64: floats use NaN
// (covers both htslib's MISSING and VECTOR_END float encodings, which are
// distinct NaN bit patterns); ints use MISSING (i32::MIN) or VECTOR_END
// (i32::MIN + 1) — the round-trip through f64 is exact for both since they're
// in i32 range.
fn is_htslib_missing(raw_val: f64, is_float: bool) -> bool {
    if is_float {
        raw_val.is_nan()
    } else {
        let iv = raw_val as i32;
        iv == i32::MIN || iv == i32::MIN + 1
    }
}

// Resolve one atom's value from a record-level raw buffer (already widened to
// f64): `None` (field absent from the record/sample) or an out-of-range
// Number=A index falls back to the spec's sentinel/default; a length-1 buffer
// is a Number=1 scalar, otherwise indexed by `source_alt_index - 1` (Number=A):
// `source_alt_index` is 1-based (ALT1 → 1, matching BCF GT allele codes; see
// `normalize::atomize_record`), but htslib's Number=A buffer is 0-based
// per-ALT (ALT1's value lives at `vals[0]`).
fn resolve_scalar(vals: Option<&[f64]>, source_alt_index: u16, spec: &FieldSpec) -> f64 {
    let default_val = sentinel_default(spec);
    let Some(vals) = vals else {
        return default_val;
    };
    if vals.is_empty() {
        return default_val;
    }
    let raw_val = if vals.len() == 1 {
        vals[0]
    } else {
        // source_alt_index is 1-based; htslib Number=A buffers are 0-based per-ALT.
        match vals.get(source_alt_index.saturating_sub(1) as usize) {
            Some(&v) => v,
            None => return default_val,
        }
    };
    if is_htslib_missing(raw_val, spec.stage_is_float()) {
        default_val
    } else {
        raw_val
    }
}

// Decode one INFO field for the CURRENT record, once. `Ok(None)` means the
// field is absent from this record (a normal, expected occurrence — NOT an
// error); a genuine htslib read failure (bad type, corrupt buffer) surfaces
// as `ConversionError::Input`, matching the GT-decode error style.
fn decode_info_raw(record: &Record, spec: &FieldSpec) -> Result<Option<Vec<f64>>, ConversionError> {
    let pos = record.pos();
    match spec.htype {
        HtslibType::Flag => {
            // A Flag is never "missing": absent ⇒ false ⇒ 0.0.
            let present = record.info(spec.name.as_bytes()).flag().map_err(|e| {
                ConversionError::Input(format!(
                    "Failed to read INFO/{} flag at pos {pos}: {e}",
                    spec.name
                ))
            })?;
            Ok(Some(vec![if present { 1.0 } else { 0.0 }]))
        }
        HtslibType::Int => match record.info(spec.name.as_bytes()).integer() {
            Ok(Some(buf)) => Ok(Some(buf.iter().map(|&v| v as f64).collect())),
            Ok(None) => Ok(None),
            Err(e) => Err(ConversionError::Input(format!(
                "Failed to read INFO/{} at pos {pos}: {e}",
                spec.name
            ))),
        },
        HtslibType::Float => match record.info(spec.name.as_bytes()).float() {
            Ok(Some(buf)) => Ok(Some(buf.iter().map(|&v| v as f64).collect())),
            Ok(None) => Ok(None),
            Err(e) => Err(ConversionError::Input(format!(
                "Failed to read INFO/{} at pos {pos}: {e}",
                spec.name
            ))),
        },
    }
}

// Decode one FORMAT field for the CURRENT record, once, for every VCF sample
// in the header (not just the selected ones — matches the GT-decode idiom,
// which indexes into the full per-header-sample buffer via
// `self.sample_indices`). `Ok(None)` means the field is absent from this
// record for all samples (htslib reports this as `BcfMissingTag`, a normal,
// expected per-record occurrence, NOT an error). Any other htslib error is a
// genuine read failure and surfaces as `ConversionError::Input`.
//
// FORMAT Flag is not valid VCF (Flag is INFO-only); defensively treated as Int.
fn decode_format_raw(
    record: &Record,
    spec: &FieldSpec,
) -> Result<Option<Vec<Vec<f64>>>, ConversionError> {
    let pos = record.pos();
    let result = match spec.htype {
        HtslibType::Float => record.format(spec.name.as_bytes()).float().map(|bb| {
            bb.iter()
                .map(|s| s.iter().map(|&v| v as f64).collect())
                .collect::<Vec<Vec<f64>>>()
        }),
        HtslibType::Int | HtslibType::Flag => {
            record.format(spec.name.as_bytes()).integer().map(|bb| {
                bb.iter()
                    .map(|s| s.iter().map(|&v| v as f64).collect())
                    .collect::<Vec<Vec<f64>>>()
            })
        }
    };
    match result {
        Ok(v) => Ok(Some(v)),
        Err(HtslibError::BcfMissingTag { .. }) => Ok(None),
        Err(e) => Err(ConversionError::Input(format!(
            "Failed to read FORMAT/{} at pos {pos}: {e}",
            spec.name
        ))),
    }
}

pub struct VcfChunkReader {
    inner_reader: IndexedReader,
    num_samples: usize,
    ploidy: usize,
    sample_indices: Vec<usize>,
    // full 0-based contig sequence, uppercased. Consumed by `validate_ref`/`left_align`
    // in `decompose_current_record`.
    ref_seq: Vec<u8>,
    // Whether a reference FASTA was provided. When false, `validate_ref` and
    // `left_align` are skipped and the input is trusted to be pre-normalized.
    has_reference: bool,
    // Whether to skip (vs. error on) out-of-scope symbolic/breakend ALTs.
    skip_out_of_scope: bool,
    // Running count of out-of-scope ALTs dropped across this contig.
    dropped_out_of_scope: u64,

    // Requested field specs, partitioned by category (order preserved within
    // each partition). Index i into `info_fields`/`format_fields` matches
    // index i into `PendingAtom::info_vals`/`format_vals` and
    // `DenseChunk::info_staged`/`format_staged`.
    info_fields: Vec<FieldSpec>,
    format_fields: Vec<FieldSpec>,

    // Reorder state, persisted across read_next_chunk calls.
    record: Record,
    heap: BinaryHeap<Reverse<PendingAtom>>,
    frontier: u32, // start pos of the most recently read record: all future atoms have pos >= this
    eof: bool,
    next_seq: u64,
}

/// Load `chrom`'s full sequence from the FASTA at `fasta_path`, uppercased to
/// ASCII (matching the classifiers'/normalizer's expectation). Shared by
/// `VcfChunkReader::new` (validate_ref/left_align) and the orchestrator's
/// write-time `signatures` annotation — keep both call sites byte-identical.
pub(crate) fn load_contig_seq(fasta_path: &str, chrom: &str) -> Result<Vec<u8>, ConversionError> {
    // A wrong reference path reaches Rust unchecked (Python does not validate
    // it), so surface it as FileNotFoundError specifically.
    if !std::path::Path::new(fasta_path).exists() {
        return Err(ConversionError::MissingFile {
            path: fasta_path.to_string(),
        });
    }
    let fasta = rust_htslib::faidx::Reader::from_path(fasta_path).map_err(|e| {
        ConversionError::Input(format!(
            "Failed to open reference FASTA '{fasta_path}' (is there a .fai?): {e}"
        ))
    })?;
    // htslib's faidx_seq_len returns -1 for an unknown contig, surfaced via
    // rust-htslib as u64::MAX — check explicitly for a clear message.
    let contig_len_raw = fasta.fetch_seq_len(chrom);
    if contig_len_raw == u64::MAX {
        return Err(ConversionError::Input(format!(
            "Contig '{chrom}' not found in reference FASTA"
        )));
    }
    let contig_len = contig_len_raw as usize;
    let mut ref_seq = if contig_len == 0 {
        Vec::new()
    } else {
        fasta
            .fetch_seq(chrom, 0, contig_len - 1)
            .map_err(|e| ConversionError::Io {
                context: format!("fetching contig '{chrom}' from reference FASTA"),
                source: std::io::Error::other(e.to_string()),
            })?
    };
    ref_seq.make_ascii_uppercase();
    Ok(ref_seq)
}

impl VcfChunkReader {
    // opens the file, applies the sample filter at the C-level, and jumps to the chromosome.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vcf_path: &str,
        fasta_path: Option<&str>,
        chrom: &str,
        samples: &[&str],
        htslib_threads: usize,
        ploidy: usize,
        skip_out_of_scope: bool,
        fields: &[FieldSpec],
    ) -> Result<Self, ConversionError> {
        // A wrong VCF path reaches Rust when the file was removed after Python's
        // upstream indexing/open; surface it as FileNotFoundError, not a ".tbi?" Input.
        if !std::path::Path::new(vcf_path).exists() {
            return Err(ConversionError::MissingFile {
                path: vcf_path.to_string(),
            });
        }

        let mut reader = IndexedReader::from_path(vcf_path).map_err(|e| {
            ConversionError::Input(format!(
                "Failed to open VCF/BCF index for '{vcf_path}' \
                 (is there a .tbi or .csi file?): {e}"
            ))
        })?;

        reader
            .set_threads(htslib_threads)
            .map_err(|e| ConversionError::Io {
                context: format!("allocating {htslib_threads} HTSlib background threads"),
                source: std::io::Error::other(e.to_string()),
            })?;

        let header = reader.header().clone();

        let rid = header.name2rid(chrom.as_bytes()).map_err(|_| {
            ConversionError::Input(format!("Chromosome '{chrom}' not found in VCF header"))
        })?;

        reader
            .fetch(rid, 0, None)
            .map_err(|e| ConversionError::Io {
                context: format!("fetching region for chromosome '{chrom}'"),
                source: std::io::Error::other(e.to_string()),
            })?;

        let sample_indices: Vec<usize> = samples
            .iter()
            .map(|name| {
                header.sample_id(name.as_bytes()).ok_or_else(|| {
                    ConversionError::Input(format!("Sample '{name}' not found in VCF"))
                })
            })
            .collect::<Result<_, _>>()?;

        // Reference is optional. With a FASTA, cache the full uppercased contig for
        // validate_ref/left_align; without one, leave it empty and skip both.
        let (ref_seq, has_reference) = match fasta_path {
            Some(path) => (load_contig_seq(path, chrom)?, true),
            None => (Vec::new(), false),
        };

        let record = reader.empty_record();

        let info_fields: Vec<FieldSpec> = fields
            .iter()
            .filter(|f| f.category == FieldCategory::Info)
            .cloned()
            .collect();
        let format_fields: Vec<FieldSpec> = fields
            .iter()
            .filter(|f| f.category == FieldCategory::Format)
            .cloned()
            .collect();

        Ok(Self {
            inner_reader: reader,
            num_samples: samples.len(),
            ploidy,
            sample_indices,
            ref_seq,
            has_reference,
            skip_out_of_scope,
            dropped_out_of_scope: 0,
            info_fields,
            format_fields,
            record,
            heap: BinaryHeap::new(),
            frontier: 0,
            eof: false,
            next_seq: 0,
        })
    }

    /// Total out-of-scope ALTs dropped so far (valid after the read loop drains).
    pub fn dropped_out_of_scope(&self) -> u64 {
        self.dropped_out_of_scope
    }

    // Decompose `self.record` into atoms and push them onto the reorder heap, sharing
    // one decoded genotype vector across all atoms of the record.
    fn decompose_current_record(&mut self) -> Result<(), ConversionError> {
        let pos = self.record.pos() as u32;

        // Own the alleles so the record borrow is released before we mutate self.
        let ref_allele: Vec<u8>;
        let alts_owned: Vec<Vec<u8>>;
        {
            let alleles = self.record.alleles();
            ref_allele = alleles[0].to_vec();
            alts_owned = alleles[1..].iter().map(|a| a.to_vec()).collect();
        }

        // Decode per-column allele indices (-1 = missing).
        let columns = self.num_samples * self.ploidy;
        let mut gt = vec![-1i32; columns];
        {
            // Decode GT straight from the raw BCF integer buffer instead of
            // `record.genotypes().get(i)`, which allocates a per-sample
            // `Genotype(Vec<GenotypeAllele>)` for every sample of every record —
            // the dominant reader-side allocation churn. BCF GT encoding: an
            // allele is `(idx + 1) << 1 | phased`, so `e >= 2` decodes to
            // `(e >> 1) - 1`; `e` of 0/1 is missing and `i32::MIN` is vector-end
            // padding — both `< 2`, so a single `e >= 2` test reproduces the
            // `GenotypeAllele::index()` semantics exactly.
            let gts = self.record.format(b"GT").integer().map_err(|e| {
                ConversionError::Input(format!("Failed to read GT format at pos {pos}: {e}"))
            })?;
            let ploidy = self.ploidy;
            for (s_idx, &vcf_idx) in self.sample_indices.iter().enumerate() {
                let raw = gts[vcf_idx];
                let base = s_idx * ploidy;
                for p in 0..ploidy {
                    gt[base + p] = match raw.get(p) {
                        Some(&e) if e >= 2 => (e >> 1) - 1,
                        _ => -1,
                    };
                }
            }
        }
        let gt = Arc::new(gt);

        // Fail fast only when a reference is available; without one we trust the
        // input is already normalized/left-aligned.
        if self.has_reference {
            crate::normalize::validate_ref(pos, &ref_allele, &self.ref_seq)?;
        }

        let alt_refs: Vec<&[u8]> = alts_owned.iter().map(|a| a.as_slice()).collect();
        let mut atoms = Vec::new();
        let dropped = atomize_record(
            pos,
            &ref_allele,
            &alt_refs,
            &mut atoms,
            self.skip_out_of_scope,
        )?;
        self.dropped_out_of_scope += dropped as u64;

        // Decode each requested field's raw buffer ONCE for this record (position-
        // independent of atomize_record's per-ALT split); per-atom resolution below
        // only differs in which `source_alt_index` a Number=A buffer is indexed by.
        let info_raw: Vec<Option<Vec<f64>>> = self
            .info_fields
            .iter()
            .map(|spec| decode_info_raw(&self.record, spec))
            .collect::<Result<_, _>>()?;
        let format_raw: Vec<Option<Vec<Vec<f64>>>> = self
            .format_fields
            .iter()
            .map(|spec| decode_format_raw(&self.record, spec))
            .collect::<Result<_, _>>()?;

        for atom in atoms {
            // Left-align only when a reference is available; otherwise store the atom
            // at its as-given (right-trimmed) position.
            let atom = if self.has_reference {
                crate::normalize::left_align(atom, &self.ref_seq, crate::normalize::L_MAX)
            } else {
                atom
            };

            let info_vals: Vec<f64> = self
                .info_fields
                .iter()
                .zip(info_raw.iter())
                .map(|(spec, raw)| resolve_scalar(raw.as_deref(), atom.source_alt_index, spec))
                .collect();

            let mut format_vals = Vec::with_capacity(self.format_fields.len() * self.num_samples);
            for (spec, raw) in self.format_fields.iter().zip(format_raw.iter()) {
                for &vcf_idx in &self.sample_indices {
                    let sample_vals = raw.as_ref().map(|v| v[vcf_idx].as_slice());
                    format_vals.push(resolve_scalar(sample_vals, atom.source_alt_index, spec));
                }
            }

            let seq = self.next_seq;
            self.next_seq += 1;
            self.heap.push(Reverse(PendingAtom {
                pos: atom.pos,
                ilen: atom.ilen,
                alt: atom.alt,
                source_alt_index: atom.source_alt_index,
                gt: Arc::clone(&gt),
                seq,
                info_vals,
                format_vals,
            }));
        }
        Ok(())
    }

    // Yield the next atom in global position order, reading and decomposing more
    // records as needed. Left-alignment (M2b) can move an atom up to `L_MAX` bases below
    // its record's start, so a future record at start `frontier` may still produce an atom
    // as low as `frontier - L_MAX`. An atom is therefore safe to emit only once its
    // position is strictly below `frontier - L_MAX` (saturating), or the input is
    // exhausted — this preserves the position-sorted invariant the Phase-2 merge relies on.
    //
    // Memory note: the heap holds every atom with pos >= frontier - L_MAX until a later
    // record advances the frontier past that window (or EOF). For coordinate-sorted input
    // this is bounded by the atoms within an `L_MAX`-wide window of the frontier, plus
    // atoms sharing a single start position — a run of many records all starting at the
    // same pos never advances the frontier, so the heap can still grow unboundedly in
    // that (pre-existing) pathological case.
    fn next_atom(&mut self) -> Result<Option<PendingAtom>, ConversionError> {
        loop {
            if let Some(Reverse(top)) = self.heap.peek() {
                if self.eof || top.pos < self.frontier.saturating_sub(crate::normalize::L_MAX) {
                    return Ok(Some(self.heap.pop().unwrap().0));
                }
            } else if self.eof {
                return Ok(None);
            }

            match self.inner_reader.read(&mut self.record) {
                Some(Ok(())) => {
                    self.frontier = self.record.pos() as u32;
                    self.decompose_current_record()?;
                }
                Some(Err(e)) => {
                    return Err(ConversionError::Io {
                        context: "reading next VCF record".to_string(),
                        source: std::io::Error::other(e.to_string()),
                    });
                }
                None => self.eof = true,
            }
        }
    }

    // Pull up to `chunk_size` atoms (already globally position-sorted) and pack them
    // into a variant-major DenseChunk. `pool`, when present and the chunk is large
    // enough, hosts parallel presence packing (word-aligned variant blocks);
    // otherwise packing is sequential. Returns None once no atoms remain.
    pub fn read_next_chunk(
        &mut self,
        chunk_size: usize,
        chunk_id: usize,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Option<DenseChunk>, ConversionError> {
        let mut atoms: Vec<PendingAtom> = Vec::with_capacity(chunk_size);
        while atoms.len() < chunk_size {
            match self.next_atom()? {
                Some(a) => atoms.push(a),
                None => break,
            }
        }
        if atoms.is_empty() {
            return Ok(None);
        }

        let v = atoms.len();
        let columns = self.num_samples * self.ploidy;

        let mut pos = Vec::with_capacity(v);
        let mut ilens = Vec::with_capacity(v);
        let mut alt = Vec::with_capacity(v * 2);
        let mut alt_offsets = Vec::with_capacity(v + 1);
        alt_offsets.push(0u32);
        let mut genos = BitGrid3::zeros(v, self.num_samples, self.ploidy);

        let num_samples = self.num_samples;
        let mut info_staged: Vec<StagedColumn> = self
            .info_fields
            .iter()
            .map(|spec| StagedColumn::with_capacity(spec.stage_is_float(), v))
            .collect();
        let mut format_staged: Vec<StagedColumn> = self
            .format_fields
            .iter()
            .map(|spec| StagedColumn::with_capacity(spec.stage_is_float(), v * num_samples))
            .collect();

        // Sequential metadata pass (cheap, ordering-preserving).
        let mut off = 0u32;
        for a in atoms.iter() {
            pos.push(a.pos);
            ilens.push(a.ilen);
            alt.extend_from_slice(&a.alt);
            off += a.alt.len() as u32;
            alt_offsets.push(off);

            for (i, col) in info_staged.iter_mut().enumerate() {
                col.push_f64(a.info_vals[i]);
            }
            for (j, col) in format_staged.iter_mut().enumerate() {
                for s in 0..num_samples {
                    col.push_f64(a.format_vals[j * num_samples + s]);
                }
            }
        }

        // Presence packing: parallel over word-aligned variant blocks when a
        // multi-thread pool is available and the chunk is large enough to amortize
        // the fan-out; identical output to the sequential path either way.
        let parallel =
            matches!(pool, Some(p) if p.current_num_threads() >= 2) && v >= PARALLEL_MIN_VARIANTS;
        if parallel {
            pack_presence_par(&mut genos.words, &atoms, columns, pool.unwrap());
        } else {
            pack_presence_seq(&mut genos.words, &atoms, columns);
        }

        Ok(Some(DenseChunk {
            chunk_id,
            pos,
            ilens,
            alt,
            alt_offsets,
            genos,
            info_staged,
            format_staged,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::sync::OnceLock;

    // One shared 4-thread pool for all proptest cases (building a pool per case is slow).
    fn test_pool() -> &'static rayon::ThreadPool {
        static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
        POOL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
        })
    }

    // Minimal PendingAtom carrying only the fields the packers read.
    fn atom(gt: Vec<i32>, src: u16) -> PendingAtom {
        PendingAtom {
            pos: 0,
            ilen: 0,
            alt: Vec::new(),
            source_alt_index: src,
            gt: std::sync::Arc::new(gt),
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Vec::new(),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        // Parallel packing reproduces sequential packing bit-for-bit, for arbitrary
        // shapes (incl. v not a multiple of the word-aligned block size), allele
        // indices (incl. missing -1 and out-of-range values), and source alts.
        #[test]
        fn test_par_packing_matches_seq(
            num_samples in 1usize..40,
            ploidy in 1usize..4,
            v in 1usize..70,
            seed in any::<u64>(),
        ) {
            let columns = num_samples * ploidy;
            // xorshift64 for deterministic per-case gt/src patterns.
            let mut state = seed | 1;
            let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };

            let mut atoms = Vec::with_capacity(v);
            for _ in 0..v {
                let src = (next() % 4) as u16; // small alt index space
                let gt: Vec<i32> = (0..columns)
                    .map(|_| match next() % 5 {
                        0 => -1,            // missing
                        1 => src as i32,    // present (matches src)
                        2 => 7,             // out-of-range allele
                        _ => (next() % 4) as i32,
                    })
                    .collect();
                atoms.push(atom(gt, src));
            }

            let mut seq = BitGrid3::zeros(v, num_samples, ploidy);
            pack_presence_seq(&mut seq.words, &atoms, columns);

            let mut par = BitGrid3::zeros(v, num_samples, ploidy);
            pack_presence_par(&mut par.words, &atoms, columns, test_pool());

            prop_assert_eq!(seq.words, par.words, "columns={}, v={}", columns, v);
        }
    }
}

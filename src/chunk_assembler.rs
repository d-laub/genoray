use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec};
use crate::normalize::atomize_record;
use crate::record_source::{Calls, FormatVals, RawRecord, RecordSource};
use crate::types::{BitGrid3, DenseChunk, StagedColumn};
use rayon::prelude::*;
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
    calls: Arc<Calls>, // shared across the atoms decomposed from one record
    seq: u64,          // stable tiebreak for equal positions

    // INFO is resolved eagerly (already indexed by source_alt_index where the
    // underlying VCF field is Number=A) since it's already O(1) per atom -- one
    // scalar per requested spec, not F x N. Populated in `decompose_raw_record`,
    // gathered into `DenseChunk::info_staged` in `read_next_chunk`'s sequential
    // metadata pass. Empty when no INFO fields were requested.
    info_vals: Vec<f64>, // len == VcfChunkReader::info_fields.len()

    // FORMAT, by contrast, is a source-record-level buffer shared across every
    // atom decomposed from that record (like `calls` above) rather than
    // resolved per atom: resolving it per atom would materialise F x N per
    // atom even when the record has one carrier out of N, which is churn site
    // #2 this type exists to remove. Resolved lazily, per (sample, field), in
    // `read_next_chunk`'s metadata pass via `resolve_format`.
    format_vals: Arc<FormatVals>,
}

struct DecomposedRecord {
    source_pos: u32,
    atoms: Vec<PendingAtom>,
    dropped_out_of_scope: u64,
    /// `Some(detail)` when this record was dropped by `CheckRef::Exclude`
    /// (its REF disagreed with the reference); `detail` is the mismatch
    /// message, surfaced once for the first exclusion on the contig. The
    /// decomposition runs off-thread, so the owning `ChunkAssembler` tallies
    /// `ref_excluded` from this field rather than mutating a counter directly.
    ref_excluded: Option<String>,
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
    let base = vi * columns;
    match a.calls.as_ref() {
        Calls::Dense(gtc) => {
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
        Calls::Sparse(_) => {
            // Only the carriers can match `src`; every other column is REF and packs 0.
            // This is the O(carriers) path that replaces the O(columns) scan.
            for (col, allele) in a.calls.iter_non_ref() {
                if allele == src {
                    let flat = base + col as usize;
                    let w = (flat >> 6) - word_base;
                    // SAFETY: col < columns by construction (see VcfListRecordSource).
                    unsafe {
                        *words.get_unchecked_mut(w) |= 1u64 << (flat & 63);
                    }
                }
            }
        }
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
pub(crate) fn resolve_scalar(vals: Option<&[f64]>, source_alt_index: u16, spec: &FieldSpec) -> f64 {
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

// Field `j` (`spec`) for sample `s`, from a record-level `FormatVals` -- the
// lazy counterpart to the eager per-atom resolution `decompose_raw_record` used
// to do. `source_alt_index` is THIS ATOM's own index (atoms sharing one source
// record can each carry a different ALT after `atomize_record` decomposes a
// multiallelic record), not the record's.
//
// **The ALT-index asymmetry between the two arms is deliberate, not a bug:**
// - `Dense` buffers are still record-raw (Number=A, one entry per source ALT,
//   0-based) exactly as `decode_format_raw`/`RawRecord::format_vals` produced
//   them -- so this arm must re-apply `source_alt_index` via `resolve_scalar`,
//   the same call `decompose_raw_record` made before this change.
// - `ByCarrier` values are already fully resolved, per carrier, against THAT
//   FILE's OWN `source_alt_index` at merge time (`vcf_list_reader.rs`'s
//   `FileCursor::advance`, before the merge heap ever sees them) -- a k-way
//   merge of single-sample files only ever emits single-ALT `RawRecord`s
//   (`alts: vec![alt]`), so re-applying an index here would double-resolve
//   against the WRONG (always-1) index and silently corrupt a non-carrier's
//   fallback path. `cf.value` already returns `None` for a non-carrier, and
//   `resolve_scalar(None, 0, spec)` resolves that to the spec default
//   regardless of the index passed, so `0` there is inert, not a real choice.
fn resolve_format(
    fv: &FormatVals,
    spec: &FieldSpec,
    source_alt_index: u16,
    s: usize,
    j: usize,
) -> f64 {
    match fv {
        FormatVals::ByCarrier(cf) => cf
            .value(s, j)
            .unwrap_or_else(|| resolve_scalar(None, 0, spec)),
        FormatVals::Dense(raw) => {
            let sample_vals = raw[j].as_ref().map(|v| v[s].as_slice());
            resolve_scalar(sample_vals, source_alt_index, spec)
        }
    }
}

// An atom whose presence bits are already packed into the chunk's BitGrid. `gt`
// is dropped at that point, so per-chunk staging memory no longer scales with
// `chunk_size * num_samples * ploidy`. `source_alt_index` is retained (a `u16`,
// not a per-sample buffer) because FORMAT resolution is now lazy: the metadata
// pass in `read_next_chunk` needs it to resolve `format_vals`'s `Dense` arm
// (see `resolve_format`) the same way `decompose_raw_record` used to, eagerly,
// per atom.
struct AtomMeta {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    source_alt_index: u16,
    info_vals: Vec<f64>,
    format_vals: Arc<FormatVals>,
}

// Atoms buffered before their presence bits are flushed into the chunk's BitGrid.
// Rounded UP to a multiple of the word-aligned block size `g = 64/gcd(columns,64)`
// at call time, so every flush offset lands on a u64 boundary and
// `pack_presence_par` keeps its word-disjoint invariant. 1024 keeps the window
// above `PARALLEL_MIN_VARIANTS` (512) so parallel packing still engages.
const PACK_WINDOW: usize = 1024;
const NORMALIZE_BATCH_RECORDS: usize = 1024;

// Pack `buf`'s presence bits into `genos` starting at variant offset `v0`, then
// move each atom's metadata into `metas`, dropping `gt`.
//
// `v0` MUST be a multiple of the word-aligned block size, so `v0 * columns` is a
// multiple of 64 and the window owns a whole-word-aligned sub-slice of
// `genos.words`. Only the FINAL window may have a length that is not a multiple of
// that block size (its trailing partial word is not shared, because nothing is
// packed after it).
fn flush_window(
    genos: &mut BitGrid3,
    metas: &mut Vec<AtomMeta>,
    buf: &mut Vec<PendingAtom>,
    v0: usize,
    columns: usize,
    pool: Option<&rayon::ThreadPool>,
) {
    if buf.is_empty() {
        return;
    }
    debug_assert_eq!((v0 * columns) % 64, 0, "flush offset must be word-aligned");
    let word_base = (v0 * columns) / 64;
    let n_words = (buf.len() * columns).div_ceil(64);
    let words = &mut genos.words[word_base..word_base + n_words];

    let parallel = matches!(pool, Some(p) if p.current_num_threads() >= 2)
        && buf.len() >= PARALLEL_MIN_VARIANTS;
    if parallel {
        pack_presence_par(words, buf, columns, pool.unwrap());
    } else {
        pack_presence_seq(words, buf, columns);
    }

    metas.reserve(buf.len());
    for a in buf.drain(..) {
        metas.push(AtomMeta {
            pos: a.pos,
            ilen: a.ilen,
            alt: a.alt,
            source_alt_index: a.source_alt_index,
            info_vals: a.info_vals,
            format_vals: a.format_vals,
        });
    }
}

pub struct ChunkAssembler {
    source: Box<dyn RecordSource + Send>,
    num_samples: usize,
    ploidy: usize,
    /// Full 0-based contig sequence, uppercased; empty when no reference was given.
    ref_seq: Arc<Vec<u8>>,
    has_reference: bool,
    owned_range: Option<(u32, u32)>,
    skip_out_of_scope: bool,
    check_ref: crate::normalize::CheckRef,
    ref_excluded: u64,
    dropped_out_of_scope: u64,
    info_fields: Vec<FieldSpec>,
    format_fields: Vec<FieldSpec>,
    heap: BinaryHeap<Reverse<PendingAtom>>,
    frontier: u32,
    eof: bool,
    next_seq: u64,
}

#[allow(clippy::too_many_arguments)]
fn decompose_raw_record(
    rec: RawRecord,
    record_seq: u64,
    ref_seq: &[u8],
    has_reference: bool,
    skip_out_of_scope: bool,
    check_ref: crate::normalize::CheckRef,
    info_fields: &[FieldSpec],
) -> Result<DecomposedRecord, ConversionError> {
    let pos = rec.pos;
    let calls = Arc::new(rec.calls);
    // Shared, not resolved: every atom decomposed from this record gets a cheap
    // `Arc::clone` of the SAME buffer, resolved lazily per (sample, field) at
    // chunk-metadata time (`resolve_format`) rather than widened to F x N here.
    let format_vals = Arc::new(rec.format_vals);

    // Only when a reference is available: fail fast (`CheckRef::Error`) or drop
    // the record (`CheckRef::Exclude`) if its REF disagrees with the reference.
    // Without a reference we trust the input is already normalized/left-aligned.
    if has_reference {
        match crate::normalize::apply_check_ref(check_ref, pos, &rec.reference, ref_seq)? {
            crate::normalize::RefDecision::Keep => {}
            crate::normalize::RefDecision::Exclude(e) => {
                return Ok(DecomposedRecord {
                    source_pos: pos,
                    atoms: Vec::new(),
                    dropped_out_of_scope: 0,
                    ref_excluded: Some(e.to_string()),
                });
            }
        }
    }

    let alt_refs: Vec<&[u8]> = rec.alts.iter().map(|a| a.as_slice()).collect();
    let mut atoms = Vec::new();
    let dropped = atomize_record(
        pos,
        &rec.reference,
        &alt_refs,
        &mut atoms,
        skip_out_of_scope,
    )?;

    let mut pending = Vec::with_capacity(atoms.len());
    for (atom_ix, atom) in atoms.into_iter().enumerate() {
        let atom = if has_reference {
            crate::normalize::left_align(atom, ref_seq, crate::normalize::L_MAX)
        } else {
            atom
        };

        let info_vals: Vec<f64> = info_fields
            .iter()
            .zip(rec.info_raw.iter())
            .map(|(spec, raw)| resolve_scalar(raw.as_deref(), atom.source_alt_index, spec))
            .collect();

        let seq = record_seq
            .saturating_mul(1u64 << 32)
            .saturating_add(atom_ix as u64);
        pending.push(PendingAtom {
            pos: atom.pos,
            ilen: atom.ilen,
            alt: atom.alt,
            source_alt_index: atom.source_alt_index,
            calls: Arc::clone(&calls),
            seq,
            info_vals,
            format_vals: Arc::clone(&format_vals),
        });
    }

    Ok(DecomposedRecord {
        source_pos: pos,
        atoms: pending,
        dropped_out_of_scope: dropped as u64,
        ref_excluded: None,
    })
}

impl ChunkAssembler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: Box<dyn RecordSource + Send>,
        num_samples: usize,
        ploidy: usize,
        fasta_path: Option<&str>,
        chrom: &str,
        skip_out_of_scope: bool,
        check_ref: crate::normalize::CheckRef,
        fields: &[FieldSpec],
    ) -> Result<Self, ConversionError> {
        let (ref_seq, has_reference) = match fasta_path {
            Some(path) => (
                Arc::new(crate::vcf_reader::load_contig_seq(path, chrom)?),
                true,
            ),
            None => (Arc::new(Vec::new()), false),
        };
        Ok(Self::with_reference(
            source,
            num_samples,
            ploidy,
            ref_seq,
            has_reference,
            skip_out_of_scope,
            check_ref,
            fields,
            None,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_reference(
        source: Box<dyn RecordSource + Send>,
        num_samples: usize,
        ploidy: usize,
        ref_seq: Arc<Vec<u8>>,
        has_reference: bool,
        skip_out_of_scope: bool,
        check_ref: crate::normalize::CheckRef,
        fields: &[FieldSpec],
        owned_range: Option<(u32, u32)>,
    ) -> Self {
        Self {
            source,
            num_samples,
            ploidy,
            ref_seq,
            has_reference,
            owned_range,
            skip_out_of_scope,
            check_ref,
            ref_excluded: 0,
            dropped_out_of_scope: 0,
            info_fields: fields
                .iter()
                .filter(|f| f.category == FieldCategory::Info)
                .cloned()
                .collect(),
            format_fields: fields
                .iter()
                .filter(|f| f.category == FieldCategory::Format)
                .cloned()
                .collect(),
            heap: BinaryHeap::new(),
            frontier: 0,
            eof: false,
            next_seq: 0,
        }
    }

    /// Total out-of-scope (symbolic/breakend) ALTs dropped so far. Valid after the
    /// read loop drains.
    pub fn dropped_out_of_scope(&self) -> u64 {
        self.dropped_out_of_scope
    }

    /// Records excluded because their REF disagreed with the reference under
    /// `CheckRef::Exclude`. Valid after the read loop drains.
    pub fn ref_excluded(&self) -> u64 {
        self.ref_excluded
    }

    fn fill_normalize_batch(
        &mut self,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<(), ConversionError> {
        let mut records = Vec::with_capacity(NORMALIZE_BATCH_RECORDS);
        while records.len() < NORMALIZE_BATCH_RECORDS {
            match self.source.next_record()? {
                Some(rec) => {
                    self.frontier = rec.pos;
                    let record_seq = self.next_seq;
                    self.next_seq += 1;
                    records.push((record_seq, rec));
                }
                None => {
                    self.eof = true;
                    break;
                }
            }
        }
        if records.is_empty() {
            return Ok(());
        }

        let parallel =
            matches!(pool, Some(p) if p.current_num_threads() >= 2) && records.len() >= 2;
        let decomposed: Vec<DecomposedRecord> = if parallel {
            pool.unwrap().install(|| {
                records
                    .into_par_iter()
                    .map(|(record_seq, rec)| {
                        decompose_raw_record(
                            rec,
                            record_seq,
                            self.ref_seq.as_slice(),
                            self.has_reference,
                            self.skip_out_of_scope,
                            self.check_ref,
                            &self.info_fields,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
        } else {
            records
                .into_iter()
                .map(|(record_seq, rec)| {
                    decompose_raw_record(
                        rec,
                        record_seq,
                        self.ref_seq.as_slice(),
                        self.has_reference,
                        self.skip_out_of_scope,
                        self.check_ref,
                        &self.info_fields,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        for record in decomposed {
            let source_record_owned = self
                .owned_range
                .is_none_or(|(start, end)| record.source_pos >= start && record.source_pos < end);
            // A CheckRef::Exclude drop produces no atoms; only the owning shard
            // tallies it (a padded boundary record can be seen by two shards).
            if let Some(detail) = record.ref_excluded {
                if source_record_owned {
                    self.ref_excluded += 1;
                    if self.ref_excluded == 1 {
                        println!(
                            "Notice: check_ref=x excluding record(s) whose REF disagrees \
                             with the reference (first: {detail}); further exclusions on \
                             this contig are counted, not printed."
                        );
                    }
                }
                continue;
            }
            if source_record_owned {
                self.dropped_out_of_scope += record.dropped_out_of_scope;
            }
            for atom in record.atoms {
                let atom_owned = self
                    .owned_range
                    .is_none_or(|(start, end)| atom.pos >= start && atom.pos < end);
                if atom_owned {
                    self.heap.push(Reverse(atom));
                }
            }
        }
        Ok(())
    }

    // Yield the next atom in global position order. Left-alignment can move an atom
    // up to `L_MAX` bases below its record's start, so an atom is safe to emit only
    // once its position is strictly below `frontier - L_MAX` (saturating), or the
    // input is exhausted. This preserves the position-sorted invariant the Phase-2
    // merge relies on. Refill happens in bounded record batches so normalization
    // can use the reader-side processing pool without changing the emit rule.
    fn next_atom(
        &mut self,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Option<PendingAtom>, ConversionError> {
        loop {
            if let Some(Reverse(top)) = self.heap.peek() {
                if self.eof || top.pos < self.frontier.saturating_sub(crate::normalize::L_MAX) {
                    return Ok(Some(self.heap.pop().unwrap().0));
                }
            } else if self.eof {
                return Ok(None);
            }

            self.fill_normalize_batch(pool)?;
        }
    }

    // Pull up to `chunk_size` atoms (already globally position-sorted) and pack them
    // into a variant-major DenseChunk. Presence bits are packed in windows of
    // `PACK_WINDOW` atoms so the reader never holds more than one window's worth of
    // per-column genotype vectors. `pool`, when present and the window is large
    // enough, hosts parallel packing; otherwise packing is sequential. Output is
    // bit-identical either way. Returns None once no atoms remain.
    pub fn read_next_chunk(
        &mut self,
        chunk_size: usize,
        chunk_id: usize,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Option<DenseChunk>, ConversionError> {
        let columns = self.num_samples * self.ploidy;
        // Word-aligned block size: `g` variants span exactly `columns/gcd` u64 words.
        let g = 64 / gcd(columns, 64);
        let window = PACK_WINDOW.div_ceil(g) * g;

        // Allocate for the full chunk up front (packed size: chunk_size*columns bits),
        // then shrink to the true variant count after EOF.
        let mut genos = BitGrid3::zeros(chunk_size, self.num_samples, self.ploidy);
        let mut metas: Vec<AtomMeta> = Vec::with_capacity(chunk_size);
        let mut buf: Vec<PendingAtom> = Vec::with_capacity(window);
        let mut v = 0usize;

        while v + buf.len() < chunk_size {
            match self.next_atom(pool)? {
                Some(a) => {
                    buf.push(a);
                    if buf.len() == window {
                        flush_window(&mut genos, &mut metas, &mut buf, v, columns, pool);
                        v += window;
                    }
                }
                None => break,
            }
        }
        if !buf.is_empty() {
            let n = buf.len();
            flush_window(&mut genos, &mut metas, &mut buf, v, columns, pool);
            v += n;
        }

        if v == 0 {
            return Ok(None);
        }
        genos.truncate_v(v);

        let num_samples = self.num_samples;
        let mut pos = Vec::with_capacity(v);
        let mut ilens = Vec::with_capacity(v);
        let mut alt = Vec::with_capacity(v * 2);
        let mut alt_offsets = Vec::with_capacity(v + 1);
        alt_offsets.push(0u32);
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
        for a in metas.iter() {
            pos.push(a.pos);
            ilens.push(a.ilen);
            alt.extend_from_slice(&a.alt);
            off += a.alt.len() as u32;
            alt_offsets.push(off);

            for (i, col) in info_staged.iter_mut().enumerate() {
                col.push_f64(a.info_vals[i]);
            }
            // Only dense-routed variants need every sample's value; resolve per
            // column here rather than materialising F x N per atom upstream.
            for (j, col) in format_staged.iter_mut().enumerate() {
                for s in 0..num_samples {
                    col.push_f64(resolve_format(
                        &a.format_vals,
                        &self.format_fields[j],
                        a.source_alt_index,
                        s,
                        j,
                    ));
                }
            }
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
    use crate::field::{HtslibType, StorageDtype};
    use crate::record_source::CarrierFormat;
    use proptest::prelude::*;
    use std::sync::OnceLock;

    fn format_spec(name: &str) -> FieldSpec {
        FieldSpec {
            name: name.to_string(),
            category: FieldCategory::Format,
            htype: HtslibType::Float,
            dtype: StorageDtype::Auto,
            default: None,
        }
    }

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
            calls: std::sync::Arc::new(Calls::Dense(gt)),
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
        }
    }

    fn atom_at(gt: Vec<i32>, src: u16, pos: u32) -> PendingAtom {
        PendingAtom {
            pos,
            ilen: 0,
            alt: Vec::new(),
            source_alt_index: src,
            calls: std::sync::Arc::new(Calls::Dense(gt)),
            seq: pos as u64,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
        }
    }

    // Task 7's own correctness trap: `resolve_format`'s two arms must NOT treat
    // `source_alt_index` the same way. `Dense` buffers are still record-raw
    // (Number=A, one entry per source ALT, 0-based) exactly as they were before
    // this change, so this arm must keep re-applying `source_alt_index` via
    // `resolve_scalar` -- the same call `decompose_raw_record` made eagerly
    // pre-refactor. `ByCarrier` values are already fully resolved per carrier at
    // merge time (`vcf_list_reader.rs`'s `FileCursor::advance`, against THAT
    // FILE's own `source_alt_index`, before the merge heap ever sees them), so
    // re-applying an index here would double-resolve and is required to be inert.
    #[test]
    fn resolve_format_dense_reapplies_alt_index_but_by_carrier_does_not() {
        let spec = format_spec("DP");
        // One sample, Number=A buffer with a DIFFERENT value per source ALT:
        // ALT1 -> 10.0, ALT2 -> 20.0.
        let dense = FormatVals::Dense(vec![Some(vec![vec![10.0, 20.0]])]);
        assert_eq!(
            resolve_format(&dense, &spec, 1, 0, 0),
            10.0,
            "an ALT1 atom must read vals[0]"
        );
        assert_eq!(
            resolve_format(&dense, &spec, 2, 0, 0),
            20.0,
            "an ALT2 atom must read vals[1], not ALT1's value"
        );

        let mut cf = CarrierFormat::new(1);
        cf.push_sample(0, &[7.0]);
        let by_carrier = FormatVals::ByCarrier(cf);
        assert_eq!(resolve_format(&by_carrier, &spec, 1, 0, 0), 7.0);
        assert_eq!(
            resolve_format(&by_carrier, &spec, 2, 0, 0),
            7.0,
            "ByCarrier must not re-apply source_alt_index -- it's already resolved"
        );
    }

    // End-to-end regression for the same trap, through the REAL pipeline (not just
    // the helper in isolation): a genuinely multiallelic Dense-sourced record (the
    // shape `from_vcf`/`from_pgen`/`from_svar1` produce, as opposed to
    // `from_vcf_list`'s always-single-ALT merged records) must hand each of its
    // decomposed atoms ITS OWN `source_alt_index`, paired with the SAME shared
    // `format_vals` buffer, so `resolve_format` at chunk-metadata time reads back
    // each atom's own ALT1/ALT2 slot rather than a fixed or leaked index.
    #[test]
    fn decompose_raw_record_threads_each_atoms_own_alt_index_to_dense_format() {
        let format_specs = [format_spec("DP")];
        let rec = RawRecord {
            pos: 0,
            reference: b"A".to_vec(),
            alts: vec![b"C".to_vec(), b"G".to_vec()],
            calls: Calls::Dense(vec![1, 2]), // irrelevant to FORMAT resolution
            info_raw: Vec::new(),
            format_vals: FormatVals::Dense(vec![Some(vec![vec![10.0, 20.0]])]),
        };
        let decomposed = decompose_raw_record(
            rec,
            0,
            &[],
            false,
            false,
            crate::normalize::CheckRef::Error,
            &[],
        )
        .unwrap();
        assert_eq!(
            decomposed.atoms.len(),
            2,
            "REF A / ALT C,G must atomize to two SNV atoms"
        );
        for atom in &decomposed.atoms {
            let expect = if atom.source_alt_index == 1 {
                10.0
            } else {
                20.0
            };
            let got = resolve_format(
                &atom.format_vals,
                &format_specs[0],
                atom.source_alt_index,
                0,
                0,
            );
            assert_eq!(
                got, expect,
                "atom with source_alt_index={} must resolve its OWN ALT slot",
                atom.source_alt_index
            );
        }
    }

    #[test]
    fn pack_row_dense_calls_matches_the_raw_gt_loop() {
        // Guards the Task 4 migration: packing from Calls::Dense must reproduce, bit for
        // bit, what the old `&a.gt` loop produced. Any drift here is a store diff.
        let columns = 8usize;
        let gt = vec![0i32, 1, 1, 0, 2, -1, 1, 0];
        let src_alt = 1i32;

        let mut expect = vec![0u64; 1];
        for (col, &g) in gt.iter().enumerate() {
            if g == src_alt {
                expect[0] |= 1u64 << col;
            }
        }

        let atom = PendingAtom {
            pos: 100,
            ilen: 0,
            alt: b"A".to_vec(),
            source_alt_index: src_alt as u16,
            calls: std::sync::Arc::new(crate::record_source::Calls::Dense(gt)),
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
        };

        let mut got = vec![0u64; 1];
        pack_row(&mut got, 0, 0, &atom, columns);
        assert_eq!(got, expect);
    }

    #[test]
    fn pack_row_sparse_and_dense_produce_identical_bits() {
        // The whole design rests on this: a carrier list and a widened vector are two
        // encodings of the same record, so they must pack to the same bits.
        let columns = 8usize;
        let gt = vec![0i32, 1, 1, 0, 2, -1, 1, 0];
        let src_alt = 1u16;

        let mut carriers = crate::record_source::Carriers::new();
        for (col, &g) in gt.iter().enumerate() {
            if g != 0 {
                carriers.push(col as u32, g);
            }
        }

        let mk = |calls: crate::record_source::Calls| PendingAtom {
            pos: 100,
            ilen: 0,
            alt: b"A".to_vec(),
            source_alt_index: src_alt,
            calls: std::sync::Arc::new(calls),
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
        };

        let mut dense_bits = vec![0u64; 1];
        pack_row(
            &mut dense_bits,
            0,
            0,
            &mk(crate::record_source::Calls::Dense(gt)),
            columns,
        );

        let mut sparse_bits = vec![0u64; 1];
        pack_row(
            &mut sparse_bits,
            0,
            0,
            &mk(crate::record_source::Calls::Sparse(carriers)),
            columns,
        );

        assert_eq!(sparse_bits, dense_bits);
    }

    #[test]
    fn pack_row_sparse_matches_dense_across_word_boundaries() {
        // columns = 100 puts variant 1's row across words and exercises `word_base`.
        let columns = 100usize;
        let mut gt = vec![0i32; columns];
        for c in [0usize, 63, 64, 65, 99] {
            gt[c] = 1;
        }
        let mut carriers = crate::record_source::Carriers::new();
        for (col, &g) in gt.iter().enumerate() {
            if g != 0 {
                carriers.push(col as u32, g);
            }
        }
        let mk = |calls: crate::record_source::Calls| PendingAtom {
            pos: 1,
            ilen: 0,
            alt: b"A".to_vec(),
            source_alt_index: 1,
            calls: std::sync::Arc::new(calls),
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
        };
        let words = (columns * 2).div_ceil(64);

        let mut d = vec![0u64; words];
        pack_row(
            &mut d,
            0,
            1,
            &mk(crate::record_source::Calls::Dense(gt)),
            columns,
        );
        let mut s = vec![0u64; words];
        pack_row(
            &mut s,
            0,
            1,
            &mk(crate::record_source::Calls::Sparse(carriers)),
            columns,
        );
        assert_eq!(s, d);
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

    // Windowed packing must be bit-identical to packing the whole chunk at once.
    // `flush_window` is only ever called at word-aligned variant offsets except
    // for the final (partial) window, which nothing follows — mirror that here.
    proptest! {
        #[test]
        fn windowed_pack_matches_full_pack(
            n_samples in 1usize..9,
            ploidy in 1usize..3,
            srcs in prop::collection::vec(0u16..3u16, 1..200),
        ) {
            let columns = n_samples * ploidy;
            let v = srcs.len();

            // Deterministic gt: column c of variant i carries allele (i + c) % 3.
            let atoms: Vec<PendingAtom> = srcs
                .iter()
                .enumerate()
                .map(|(i, &src)| {
                    let gt: Vec<i32> = (0..columns).map(|c| ((i + c) % 3) as i32).collect();
                    atom_at(gt, src, i as u32)
                })
                .collect();

            // Reference: one full-grid sequential pack.
            let mut expect = BitGrid3::zeros(v, n_samples, ploidy);
            pack_presence_seq(&mut expect.words, &atoms, columns);

            // Windowed: flush every `window` atoms, where `window` is a multiple of
            // the word-aligned block size `g`.
            let g = 64 / gcd(columns, 64);
            let window = 4 * g;
            let mut got = BitGrid3::zeros(v, n_samples, ploidy);
            let mut metas: Vec<AtomMeta> = Vec::new();
            let mut buf: Vec<PendingAtom> = Vec::new();
            let mut v0 = 0usize;
            for a in atoms {
                buf.push(a);
                if buf.len() == window {
                    let n = buf.len();
                    flush_window(&mut got, &mut metas, &mut buf, v0, columns, Some(test_pool()));
                    v0 += n;
                }
            }
            if !buf.is_empty() {
                flush_window(&mut got, &mut metas, &mut buf, v0, columns, Some(test_pool()));
            }

            prop_assert_eq!(got.words, expect.words);
            prop_assert_eq!(metas.len(), v);
        }
    }
}

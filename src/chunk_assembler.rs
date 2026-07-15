use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec};
use crate::normalize::atomize_record;
use crate::record_source::{RawRecord, RecordSource};
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

// An atom whose presence bits are already packed into the chunk's BitGrid. `gt`
// and `source_alt_index` are dropped at that point, so per-chunk staging memory
// no longer scales with `chunk_size * num_samples * ploidy`.
struct AtomMeta {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    info_vals: Vec<f64>,
    format_vals: Vec<f64>,
}

// Atoms buffered before their presence bits are flushed into the chunk's BitGrid.
// Rounded UP to a multiple of the word-aligned block size `g = 64/gcd(columns,64)`
// at call time, so every flush offset lands on a u64 boundary and
// `pack_presence_par` keeps its word-disjoint invariant. 1024 keeps the window
// above `PARALLEL_MIN_VARIANTS` (512) so parallel packing still engages.
const PACK_WINDOW: usize = 1024;

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
    ref_seq: Vec<u8>,
    has_reference: bool,
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
            Some(path) => (crate::vcf_reader::load_contig_seq(path, chrom)?, true),
            None => (Vec::new(), false),
        };
        Ok(Self {
            source,
            num_samples,
            ploidy,
            ref_seq,
            has_reference,
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
        })
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

    // Decompose one record into atoms and push them onto the reorder heap, sharing
    // one decoded genotype vector across all atoms of the record.
    fn decompose_record(&mut self, rec: RawRecord) -> Result<(), ConversionError> {
        let pos = rec.pos;
        let gt = Arc::new(rec.gt);

        // Fail fast only when a reference is available; without one we trust the
        // input is already normalized/left-aligned.
        if self.has_reference {
            match crate::normalize::apply_check_ref(
                self.check_ref,
                pos,
                &rec.reference,
                &self.ref_seq,
            )? {
                crate::normalize::RefDecision::Keep => {}
                crate::normalize::RefDecision::Exclude(e) => {
                    self.ref_excluded += 1;
                    if self.ref_excluded == 1 {
                        println!(
                            "Notice: check_ref=x excluding record(s) whose REF disagrees \
                             with the reference (first: {e}); further exclusions on this \
                             contig are counted, not printed."
                        );
                    }
                    return Ok(());
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
            self.skip_out_of_scope,
        )?;
        self.dropped_out_of_scope += dropped as u64;

        for atom in atoms {
            let atom = if self.has_reference {
                crate::normalize::left_align(atom, &self.ref_seq, crate::normalize::L_MAX)
            } else {
                atom
            };

            let info_vals: Vec<f64> = self
                .info_fields
                .iter()
                .zip(rec.info_raw.iter())
                .map(|(spec, raw)| resolve_scalar(raw.as_deref(), atom.source_alt_index, spec))
                .collect();

            let mut format_vals = Vec::with_capacity(self.format_fields.len() * self.num_samples);
            for (spec, raw) in self.format_fields.iter().zip(rec.format_raw.iter()) {
                for s in 0..self.num_samples {
                    let sample_vals = raw.as_ref().map(|v| v[s].as_slice());
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

    // Yield the next atom in global position order. Left-alignment can move an atom
    // up to `L_MAX` bases below its record's start, so an atom is safe to emit only
    // once its position is strictly below `frontier - L_MAX` (saturating), or the
    // input is exhausted. This preserves the position-sorted invariant the Phase-2
    // merge relies on. (Unchanged from the pre-refactor VcfChunkReader.)
    fn next_atom(&mut self) -> Result<Option<PendingAtom>, ConversionError> {
        loop {
            if let Some(Reverse(top)) = self.heap.peek() {
                if self.eof || top.pos < self.frontier.saturating_sub(crate::normalize::L_MAX) {
                    return Ok(Some(self.heap.pop().unwrap().0));
                }
            } else if self.eof {
                return Ok(None);
            }

            match self.source.next_record()? {
                Some(rec) => {
                    self.frontier = rec.pos;
                    self.decompose_record(rec)?;
                }
                None => self.eof = true,
            }
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
            match self.next_atom()? {
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
            for (j, col) in format_staged.iter_mut().enumerate() {
                for s in 0..num_samples {
                    col.push_f64(a.format_vals[j * num_samples + s]);
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

    fn atom_at(gt: Vec<i32>, src: u16, pos: u32) -> PendingAtom {
        PendingAtom {
            pos,
            ilen: 0,
            alt: Vec::new(),
            source_alt_index: src,
            gt: std::sync::Arc::new(gt),
            seq: pos as u64,
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

use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec};
use crate::normalize::atomize_record;
use crate::record_source::{
    Calls, Carriers, FormatVals, RawRecord, RecordSource, resolve_format, resolve_scalar,
};
use crate::types::{BitGrid3, DenseChunk, StagedColumn};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

// What an atom retains in order to pack its presence row later.
//
// Dense sources keep a bitset (`columns/8` bytes, shared across the atoms of one
// record) rather than the record's allele vector (`columns*4`). Sparse sources
// keep their calls verbatim -- already O(carriers), and `flush_window` needs the
// carriers themselves, not just their bits.
enum AtomCalls {
    Masks {
        masks: Arc<PresenceMasks>,
        /// This atom's slot within the record's slab.
        slot: u16,
    },
    Sparse(Arc<Calls>),
}

// A decomposed atom awaiting emission. Carries a shared handle to its source record's
// per-column allele indices so genotype presence is computed at chunk-build time.
struct PendingAtom {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    source_alt_index: u16,
    // Shared across the atoms decomposed from one record. For dense sources this
    // is a presence bitset, NOT the allele vector: retaining the vector is what
    // made the reader cost ~8 KB per sample per contig (issue #155).
    calls: AtomCalls,
    seq: u64,        // stable tiebreak for equal positions
    global_idx: i32, // threaded verbatim from the source record

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
    /// Atoms decomposed from this record whose position moved during
    /// left-alignment.
    normalized: u64,
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
    let base = vi * columns;
    match &a.calls {
        AtomCalls::Masks { masks, slot } => {
            or_mask_into(words, word_base, base, masks.mask(*slot), columns);
        }
        AtomCalls::Sparse(calls) => {
            // Only the carriers can match `src`; every other column is REF and packs 0.
            // This is the O(carriers) path that replaces the O(columns) scan.
            let src = a.source_alt_index as i32;
            for (col, allele) in calls.iter_non_ref() {
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

// OR one row's presence mask into `words` at flat bit offset `base`, where
// `words[0]` is global word `word_base`. `mask` carries `columns` meaningful
// bits; bits at or beyond `columns` are zero by construction
// (`PresenceMasks::from_dense` only ever sets `col < columns`).
//
// Replaces an O(columns) allele comparison with an O(columns/64) shifted word
// OR. `base` is generally not word-aligned -- row `vi` starts at `vi*columns` --
// so each mask word contributes to two target words.
#[inline]
fn or_mask_into(words: &mut [u64], word_base: usize, base: usize, mask: &[u64], columns: usize) {
    if columns == 0 {
        return;
    }
    let w0 = (base >> 6) - word_base;
    let s = base & 63;
    let last = ((base + columns - 1) >> 6) - word_base;

    for (j, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        // A nonzero mask word's lowest set bit is at some `col < columns`, so
        // its low half always lands inside the row's span.
        let lo = w0 + j;
        words[lo] |= m << s;
        if s > 0 {
            let hi = lo + 1;
            if hi <= last {
                words[hi] |= m >> (64 - s);
            } else {
                // Not merely an optimisation: `pack_presence_par` gives each
                // task a word-DISJOINT slice, so writing past `last` would
                // corrupt another task's words. A carry here is always zero,
                // because bits at or beyond `columns` are zero.
                debug_assert_eq!(m >> (64 - s), 0, "carry outside the row's span");
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

/// Per-record presence bitsets: for each in-scope ALT of one source record, the
/// set of haplotype columns whose call is that ALT.
///
/// This is what dense sources retain INSTEAD of `Calls::Dense(Vec<i32>)`. The
/// only things a retained `Calls::Dense` was ever used for are `pack_row`'s
/// `gt[col] == source_alt_index` test and carrier recovery -- and dense sources
/// skip carrier recovery (`flush_window` returns `None`, since recovering
/// carriers from the packed grid is cheaper). So the retained payload can be
/// that test's ANSWER: one bit per column instead of one `i32`, `columns/8`
/// bytes per record instead of `columns*4`. That 32x is what keeps the reader's
/// live set bounded at biobank cohort widths -- see issue #155.
struct PresenceMasks {
    /// Slot-major: slot `s` owns `words[s*words_per_mask .. (s+1)*words_per_mask]`.
    words: Vec<u64>,
    words_per_mask: usize,
}

impl PresenceMasks {
    /// Build one slab from a record's dense calls, in a SINGLE pass over `gt`.
    ///
    /// `wanted` is the ascending, deduplicated list of `source_alt_index` values
    /// this record's atoms actually carry; slot `i` corresponds to `wanted[i]`.
    /// Restricting to those means a record whose other ALTs were dropped as
    /// out-of-scope pays for the ALTs it kept, not for `n_alts`. Alleles outside
    /// `wanted` -- REF `0`, missing `-1`, dropped ALTs -- set no bit, which is
    /// exactly what `gt[col] == src` does for any `src` in `wanted`.
    fn from_dense(gt: &[i32], columns: usize, wanted: &[u16]) -> Self {
        debug_assert!(
            wanted.iter().all(|&a| a > 0),
            "source_alt_index is 1-based; allele 0 is REF and can never be an atom's ALT"
        );
        let words_per_mask = columns.div_ceil(64);
        let mut words = vec![0u64; words_per_mask * wanted.len()];

        // allele -> slot. `u16::MAX` means "no slot"; sized by the largest
        // wanted allele rather than by `n_alts`, which is not passed in.
        let max_allele = wanted.iter().copied().max().unwrap_or(0) as usize;
        let mut slot_of = vec![u16::MAX; max_allele + 1];
        for (slot, &allele) in wanted.iter().enumerate() {
            slot_of[allele as usize] = slot as u16;
        }

        for (col, &a) in gt.iter().take(columns).enumerate() {
            if a < 0 {
                continue; // missing
            }
            let a = a as usize;
            if a >= slot_of.len() {
                continue; // REF, or an ALT no atom kept
            }
            let slot = slot_of[a];
            if slot == u16::MAX {
                continue;
            }
            words[slot as usize * words_per_mask + (col >> 6)] |= 1u64 << (col & 63);
        }
        Self {
            words,
            words_per_mask,
        }
    }

    #[inline]
    fn mask(&self, slot: u16) -> &[u64] {
        let start = slot as usize * self.words_per_mask;
        &self.words[start..start + self.words_per_mask]
    }
}

// Below this much packing work in a window, parallel packing's per-task overhead
// outweighs the win -- pack sequentially instead.
//
// CELLS, not variants. A variant count is a threshold on the wrong quantity: the
// work is `variants * columns`, so a cell-budgeted window at large `S` drops
// below any fixed variant count and silently disengages parallel packing.
//
// MEASURED (Task 7): a wall-time sweep on carter-cn-03 over
// {0, 512*1_024, 8*512*1_024, usize::MAX} at S=2,000/8,000/32,000/128,000
// (3 reps each, full 22-contig conversions) found no value distinguishably
// faster or slower than any other at any width -- the run-to-run spread
// (up to ~8% at S=128,000) exceeds the spread across values (at most ~3%) at
// every measured point. Kept at its seeded value rather than tuned off
// noise. Full tables:
// docs/superpowers/plans/results/2026-08-06-reader-cell-budget-measurement.md
//
// Seeded at the product that reproduces the OLD gate (512 variants) at a
// 1,024-column cohort: the new gate matches the old one exactly at columns ==
// 1,024, is STRICTER below it (e.g. at columns = 512 it now takes 1,024 atoms
// to go parallel where 512 used to suffice), and LOOSER above it -- it does
// NOT reproduce the old gate's behavior generally, only at that one column
// count.
const PARALLEL_MIN_CELLS: usize = 512 * 1_024;

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

// An atom whose presence bits are already packed into the chunk's BitGrid. `gt`
// is dropped at that point, so per-chunk staging memory no longer scales with
// `chunk_size * num_samples * ploidy`. `source_alt_index` is retained (a `u16`,
// not a per-sample buffer) because FORMAT resolution is now lazy: the metadata
// pass in `read_next_chunk` needs it to resolve `format_vals`'s `Dense` arm
// (see `resolve_format`) the same way `decompose_raw_record` used to, eagerly,
// per atom. Carrier-bearing chunks never run that resolution here: their
// `format_vals` is moved wholesale into `DenseChunk::format_by_carrier` and
// resolved in `rvk` per carrier instead.
struct AtomMeta {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    source_alt_index: u16,
    info_vals: Vec<f64>,
    format_vals: Arc<FormatVals>,
    global_idx: i32,
    /// Columns whose allele is this atom's `source_alt_index` -- i.e. exactly the
    /// bits `pack_row` sets for this atom. `None` when the source is natively
    /// dense, where recovering carriers from the grid is correct and cheaper than
    /// retaining them (see `DenseChunk::carriers`, src/types.rs).
    carriers: Option<Carriers>,
}

// *** These two budgets set the reader's peak RAM. ***
//
// They used to be fixed VARIANT counts (`PACK_WINDOW`/`NORMALIZE_BATCH_RECORDS`,
// both 1024) multiplying an O(n_samples) payload, giving a live set of
// `min(V, 2048) * n_samples * ploidy * 4` bytes -- up to 16 KB per sample,
// bounded by neither `chunk_size` nor `max_mem`, and the blocker behind
// `RamLaw::PGEN`'s conservative margin (issue #155).
//
// They are BYTE budgets now, in the units each buffer actually holds:
//
//   * `RAW_STAGE_BYTES` bounds `fill_normalize_batch`'s staged `RawRecord`s,
//     which still carry `Calls::Dense` -- `columns * 4` bytes each.
//   * `MASK_STAGE_BYTES` bounds the atoms a pack window RETAINS, which carry
//     `PresenceMasks` -- `columns/8` bytes each, 32x smaller. That 32x is why
//     the window stays large in the normal regime instead of collapsing to ~65
//     records at S=128,000.
//
// Deliberately constants rather than derived from `max_mem`: threading a budget
// into `ChunkAssembler` would add a regressor to `RamLaw::PGEN`, whereas a
// constant lands in `base_mb` where a constant belongs, and the bound stays
// checkable by arithmetic instead of only by measurement.
const RAW_STAGE_BYTES: usize = 64 << 20;
const MASK_STAGE_BYTES: usize = 64 << 20;

// Caps preserve today's value as the ceiling. They bind -- i.e. nothing changes
// -- only while a buffer's per-record cost keeps 1024 records inside the budget:
// up to S = 8,192 for the batch, and up to S = 262,144 for the window, the
// latter because masks are 32x cheaper per record. Between those, the batch
// shrinks with cohort width. That is the fix, and it is why `RamLaw::PGEN` has
// to be re-fitted rather than carried over.
const MAX_BATCH_RECORDS: usize = 1024;
const MAX_PACK_WINDOW: usize = 1024;

// Floors are small ON PURPOSE. A thread-scaled floor would defeat the budget:
// 48 threads x 4 records is 192 records, which at S=128,000 is 197 MB against a
// 64 MiB budget. At 8, the floor binds only when one record exceeds an eighth of
// the budget -- roughly S = 1,000,000 -- and even then the batch costs exactly
// the budget rather than a multiple of it. Past that width staging resumes
// growing with S and decode has only 8 tasks to spread; each is 8+ MB of work,
// so the pool is coarsely fed rather than starved. That is a stated limit of the
// bound, not a claim that it holds everywhere.
const MIN_BATCH_RECORDS: usize = 8;
const MIN_PACK_WINDOW: usize = 8;

/// Records `fill_normalize_batch` stages before decomposing them.
fn batch_records(columns: usize) -> usize {
    (RAW_STAGE_BYTES / (columns * 4).max(1)).clamp(MIN_BATCH_RECORDS, MAX_BATCH_RECORDS)
}

/// Atoms buffered before their presence bits are flushed into the chunk's grid.
///
/// The CALLER rounds this up to a multiple of the word-aligned block size
/// `g = 64/gcd(columns, 64)`, so every flush offset lands on a u64 boundary and
/// `pack_presence_par` keeps its word-disjoint invariant. That rounding can
/// exceed the budget by at most `g - 1 <= 63` records -- a few percent at the
/// widths where the budget binds, and zero whenever `columns` is a multiple of 64.
fn pack_window(columns: usize) -> usize {
    (MASK_STAGE_BYTES / (columns.div_ceil(64) * 8).max(1)).clamp(MIN_PACK_WINDOW, MAX_PACK_WINDOW)
}

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
        && buf.len().saturating_mul(columns) >= PARALLEL_MIN_CELLS;
    if parallel {
        pack_presence_par(words, buf, columns, pool.unwrap());
    } else {
        pack_presence_seq(words, buf, columns);
    }

    metas.reserve(buf.len());
    for a in buf.drain(..) {
        // Reuses the same `allele == src` filter `pack_row` already applies (see
        // above) -- O(carriers), not a new scan. `Calls::Sparse`'s carrier list is
        // ascending by construction (Task 3-7), and filtering an ascending sequence
        // keeps it ascending, so `Carriers::push`'s ordering invariant holds.
        let carriers = match &a.calls {
            AtomCalls::Masks { .. } => None,
            AtomCalls::Sparse(calls) => {
                let src = a.source_alt_index as i32;
                let mut c = Carriers::new();
                for (col, allele) in calls.iter_non_ref() {
                    if allele == src {
                        c.push(col, allele);
                    }
                }
                Some(c)
            }
        };
        metas.push(AtomMeta {
            pos: a.pos,
            ilen: a.ilen,
            alt: a.alt,
            source_alt_index: a.source_alt_index,
            info_vals: a.info_vals,
            format_vals: a.format_vals,
            global_idx: a.global_idx,
            carriers,
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
    /// Contig label, used only to tag `tracing` events.
    chrom: String,
    ref_excluded: u64,
    dropped_out_of_scope: u64,
    normalized_total: u64,
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
    chrom: &str,
) -> Result<DecomposedRecord, ConversionError> {
    let pos = rec.pos;
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
                tracing::debug!(chrom = %chrom, pos = pos, detail = %e,
                    "excluded record: REF disagrees with reference");
                return Ok(DecomposedRecord {
                    source_pos: pos,
                    atoms: Vec::new(),
                    dropped_out_of_scope: 0,
                    normalized: 0,
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
    // Ends the borrow of `rec.alts` so `rec.calls` can be moved out below.
    drop(alt_refs);

    // Collapse the record's calls into what its atoms will actually retain.
    // For a dense source that is one presence bitset per in-scope ALT; the
    // `Vec<i32>` is dropped at the end of this match, which is the whole point
    // -- 1.024 MB per record at S=128,000 that used to survive into the heap.
    enum RecordCalls {
        Masks {
            masks: Arc<PresenceMasks>,
            /// Indexed by `source_alt_index`; `u16::MAX` for alleles with no slot.
            slot_of: Vec<u16>,
        },
        Sparse(Arc<Calls>),
    }
    let record_calls = match rec.calls {
        Calls::Dense(gt) => {
            let columns = gt.len();
            let mut wanted: Vec<u16> = atoms.iter().map(|a| a.source_alt_index).collect();
            wanted.sort_unstable();
            wanted.dedup();
            let mut slot_of =
                vec![u16::MAX; wanted.iter().copied().max().unwrap_or(0) as usize + 1];
            for (slot, &allele) in wanted.iter().enumerate() {
                slot_of[allele as usize] = slot as u16;
            }
            RecordCalls::Masks {
                masks: Arc::new(PresenceMasks::from_dense(&gt, columns, &wanted)),
                slot_of,
            }
        }
        sparse @ Calls::Sparse(_) => RecordCalls::Sparse(Arc::new(sparse)),
    };

    let mut normalized = 0u64;
    let mut pending = Vec::with_capacity(atoms.len());
    for (atom_ix, atom) in atoms.into_iter().enumerate() {
        let atom = if has_reference {
            let pre_pos = atom.pos;
            let aligned = crate::normalize::left_align(atom, ref_seq, crate::normalize::L_MAX);
            if aligned.pos != pre_pos {
                normalized += 1;
                tracing::debug!(chrom = %chrom, from = pre_pos, to = aligned.pos,
                    "left-aligned indel");
            }
            aligned
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
            calls: match &record_calls {
                RecordCalls::Masks { masks, slot_of } => AtomCalls::Masks {
                    masks: Arc::clone(masks),
                    slot: slot_of[atom.source_alt_index as usize],
                },
                RecordCalls::Sparse(c) => AtomCalls::Sparse(Arc::clone(c)),
            },
            seq,
            info_vals,
            format_vals: Arc::clone(&format_vals),
            global_idx: rec.global_idx,
        });
    }
    // No per-record atom-count invariant here: multiallelic records legitimately
    // atomize to >1 atom (see the ALT C,G test below). `global_idx` is threaded
    // onto every atom a record produces, not asserted to produce exactly one.

    Ok(DecomposedRecord {
        source_pos: pos,
        atoms: pending,
        dropped_out_of_scope: dropped as u64,
        normalized,
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
            chrom,
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
        chrom: &str,
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
            chrom: chrom.to_string(),
            ref_excluded: 0,
            dropped_out_of_scope: 0,
            normalized_total: 0,
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

    /// Total atoms whose position moved during left-alignment so far. Valid
    /// after the read loop drains.
    pub fn normalized_total(&self) -> u64 {
        self.normalized_total
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
        let cap = batch_records(self.num_samples * self.ploidy);
        let mut records = Vec::with_capacity(cap);
        while records.len() < cap {
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
                            &self.chrom,
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
                        &self.chrom,
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
                        // Duplicates `report_ref_excluded`'s per-contig info summary
                        // (chrom + total count) by design: this adds the specific
                        // first-offender detail that the summary doesn't carry, so
                        // it's kept at debug rather than dropped outright.
                        tracing::debug!(
                            chrom = %self.chrom,
                            detail = %detail,
                            "check_ref=x excluding record(s) whose REF disagrees with the \
                             reference; further exclusions on this contig are counted, not \
                             logged individually"
                        );
                    }
                }
                continue;
            }
            if source_record_owned {
                self.dropped_out_of_scope += record.dropped_out_of_scope;
                self.normalized_total += record.normalized;
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
    // `pack_window(columns)` atoms so the reader never holds more than one window's
    // worth of buffered `PresenceMasks` bitsets. `pool`, when present and the window
    // is large enough, hosts parallel packing; otherwise packing is sequential.
    // Output is bit-identical either way. Returns None once no atoms remain.
    pub fn read_next_chunk(
        &mut self,
        chunk_size: usize,
        chunk_id: usize,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Option<DenseChunk>, ConversionError> {
        let columns = self.num_samples * self.ploidy;
        // Word-aligned block size: `g` variants span exactly `columns/gcd` u64 words.
        let g = 64 / gcd(columns, 64);
        let window = pack_window(columns).div_ceil(g) * g;

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
        let mut global_idx: Vec<i32> = Vec::with_capacity(v);
        let mut ilens = Vec::with_capacity(v);
        let mut alt = Vec::with_capacity(v * 2);
        let mut alt_offsets = Vec::with_capacity(v + 1);
        alt_offsets.push(0u32);
        let mut info_staged: Vec<StagedColumn> = self
            .info_fields
            .iter()
            .map(|spec| StagedColumn::with_capacity(spec.stage_is_float(), v))
            .collect();
        // A chunk is carrier-bearing iff its first atom carries (uniformity is
        // asserted below). Decide once, up front, so FORMAT staging can be
        // skipped entirely for carrier-bearing chunks.
        let carrier_bearing = metas.first().is_some_and(|a| a.carriers.is_some());
        let mut format_staged: Vec<StagedColumn> = if carrier_bearing {
            Vec::new()
        } else {
            self.format_fields
                .iter()
                .map(|spec| StagedColumn::with_capacity(spec.stage_is_float(), v * num_samples))
                .collect()
        };

        // Sequential metadata pass (cheap, ordering-preserving).
        let mut off = 0u32;
        let mut carrier_opts: Vec<Option<Carriers>> = Vec::with_capacity(metas.len());
        let mut format_arcs: Vec<Arc<FormatVals>> = Vec::with_capacity(metas.len());
        for a in metas.iter_mut() {
            pos.push(a.pos);
            global_idx.push(a.global_idx);
            ilens.push(a.ilen);
            alt.extend_from_slice(&a.alt);
            off += a.alt.len() as u32;
            alt_offsets.push(off);

            for (i, col) in info_staged.iter_mut().enumerate() {
                col.push_f64(a.info_vals[i]);
            }
            // Dense-source chunks stage every sample's value per atom (F x N):
            // for these, `genos`/`format_staged` IS the representation and the
            // per-sample column is the real work. Carrier-bearing chunks skip
            // this entirely -- their FORMAT rides `format_by_carrier` and `rvk`
            // resolves it per carrier (route-before-densify), so the old
            // unconditional F x N staging (the from_vcf_list O(N^2)) is gone.
            if !carrier_bearing {
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
            format_arcs.push(Arc::clone(&a.format_vals));
            carrier_opts.push(a.carriers.take());
        }

        // A chunk is fed by a single source, so carrier-bearing is all-or-nothing:
        // either every atom carried one (the k-way merge) or none did (a natively
        // dense source). `rvk` treats `Some`/`None` as its routing switch, so a
        // mixed chunk would silently corrupt that decision -- assert the
        // uniformity here rather than downstream.
        let all_some = carrier_opts.iter().all(|c| c.is_some());
        let all_none = carrier_opts.iter().all(|c| c.is_none());
        debug_assert!(
            all_some || all_none,
            "a chunk must not mix carrier-bearing and dense-source atoms"
        );
        let carriers = if all_some {
            Some(
                carrier_opts
                    .into_iter()
                    .map(|c| c.expect("checked all_some above"))
                    .collect(),
            )
        } else {
            None
        };

        // `format_by_carrier` is Some/None in lockstep with `carriers`: both
        // come from a carrier-bearing source or neither does (the `all_some`/
        // `all_none` uniformity asserted above).
        let format_by_carrier = if all_some { Some(format_arcs) } else { None };

        Ok(Some(DenseChunk {
            chunk_id,
            pos,
            global_idx,
            ilens,
            alt,
            alt_offsets,
            genos,
            info_staged,
            format_staged,
            carriers,
            format_by_carrier,
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

    // Test-only: builds a single-slot `PresenceMasks` matching `gt[col] == src`
    // for ANY src, including 0. `PresenceMasks::from_dense` debug-asserts its
    // `wanted` slots are all ALT (> 0) because production `source_alt_index` is
    // always 1-based -- but `atom`/`atom_at` are exercised by proptests that
    // deliberately probe src across REF/ALT/missing/out-of-range, so this
    // bypasses that precondition by constructing the bitset directly rather
    // than routing through `from_dense`.
    fn mask_for(gt: &[i32], columns: usize, src: u16) -> PresenceMasks {
        let words_per_mask = columns.div_ceil(64);
        let mut words = vec![0u64; words_per_mask];
        for (col, &g) in gt.iter().take(columns).enumerate() {
            if g == src as i32 {
                words[col >> 6] |= 1u64 << (col & 63);
            }
        }
        PresenceMasks {
            words,
            words_per_mask,
        }
    }

    // Minimal PendingAtom carrying only the fields the packers read.
    fn atom(gt: Vec<i32>, src: u16) -> PendingAtom {
        let columns = gt.len();
        let masks = std::sync::Arc::new(mask_for(&gt, columns, src));
        PendingAtom {
            pos: 0,
            ilen: 0,
            alt: Vec::new(),
            source_alt_index: src,
            calls: AtomCalls::Masks { masks, slot: 0 },
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
            global_idx: -1,
        }
    }

    fn atom_at(gt: Vec<i32>, src: u16, pos: u32) -> PendingAtom {
        let columns = gt.len();
        let masks = std::sync::Arc::new(mask_for(&gt, columns, src));
        PendingAtom {
            pos,
            ilen: 0,
            alt: Vec::new(),
            source_alt_index: src,
            calls: AtomCalls::Masks { masks, slot: 0 },
            seq: pos as u64,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
            global_idx: -1,
        }
    }

    // `Calls::Sparse` counterpart to `atom`: same shape, carriers derived from
    // `gt`'s non-zero entries in ascending column order (what `Carriers`
    // requires).
    fn sparse_atom(gt: &[i32], src: u16) -> PendingAtom {
        let mut carriers = crate::record_source::Carriers::new();
        for (col, &g) in gt.iter().enumerate() {
            if g != 0 {
                carriers.push(col as u32, g);
            }
        }
        PendingAtom {
            pos: 0,
            ilen: 0,
            alt: Vec::new(),
            source_alt_index: src,
            calls: AtomCalls::Sparse(std::sync::Arc::new(Calls::Sparse(carriers))),
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
            global_idx: -1,
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
            global_idx: -1,
        };
        let decomposed = decompose_raw_record(
            rec,
            0,
            &[],
            false,
            false,
            crate::normalize::CheckRef::Error,
            &[],
            "chr1",
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
    fn decompose_threads_record_global_idx_onto_each_atom() {
        // A single biallelic SNP record tagged global id 7 must yield exactly one
        // atom whose global_idx == 7 (the 1:1 record<->atom contract).
        let rec = RawRecord {
            pos: 0,
            reference: b"A".to_vec(),
            alts: vec![b"C".to_vec()], // biallelic SNP -> 1 atom
            calls: Calls::Dense(vec![1]),
            info_raw: Vec::new(),
            format_vals: FormatVals::Dense(vec![Some(vec![vec![10.0]])]),
            global_idx: 7,
        };
        let decomposed = decompose_raw_record(
            rec,
            0,
            &[],
            false,
            false,
            crate::normalize::CheckRef::Error,
            &[],
            "chr1",
        )
        .unwrap();
        assert_eq!(
            decomposed.atoms.len(),
            1,
            "REF A / ALT C must atomize to exactly one SNV atom"
        );
        assert_eq!(decomposed.atoms[0].global_idx, 7);
    }

    #[test]
    fn decompose_retains_masks_not_the_allele_vector_for_dense_sources() {
        // The memory claim, asserted structurally rather than by measurement: after
        // decomposition a dense record's atoms must not hold anything sized like
        // `columns * 4`. If this ever reverts to `AtomCalls::Sparse` or to a
        // retained `Calls::Dense`, issue #155's ratchet is back.
        let columns = 128usize;
        let mut gt = vec![0i32; columns];
        gt[7] = 1;
        let rec = RawRecord {
            pos: 100,
            reference: b"A".to_vec(),
            alts: vec![b"C".to_vec()],
            calls: crate::record_source::Calls::Dense(gt),
            format_vals: FormatVals::Dense(Vec::new()),
            info_raw: Vec::new(),
            global_idx: -1,
        };
        let d = decompose_raw_record(
            rec,
            0,
            &[],
            false,
            true,
            crate::normalize::CheckRef::Error,
            &[],
            "chrT",
        )
        .expect("decompose");
        assert_eq!(d.atoms.len(), 1);
        match &d.atoms[0].calls {
            AtomCalls::Masks { masks, slot } => {
                assert_eq!(masks.mask(*slot)[0], 1u64 << 7);
            }
            AtomCalls::Sparse(_) => panic!("dense source must retain masks, not calls"),
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
            calls: {
                let m = PresenceMasks::from_dense(&gt, columns, &[src_alt as u16]);
                AtomCalls::Masks {
                    masks: std::sync::Arc::new(m),
                    slot: 0,
                }
            },
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
            global_idx: -1,
        };

        let mut got = vec![0u64; 1];
        pack_row(&mut got, 0, 0, &atom, columns);
        assert_eq!(got, expect);
    }

    #[test]
    fn presence_masks_mark_exactly_the_columns_matching_each_alt() {
        let gt = vec![0i32, 1, 2, -1, 1, 2, 0, 1];
        let m = PresenceMasks::from_dense(&gt, 8, &[1, 2]);
        // slot 0 == ALT 1 -> columns 1, 4, 7
        assert_eq!(m.mask(0)[0], (1u64 << 1) | (1 << 4) | (1 << 7));
        // slot 1 == ALT 2 -> columns 2, 5
        assert_eq!(m.mask(1)[0], (1u64 << 2) | (1 << 5));
    }

    #[test]
    fn presence_masks_ignore_ref_missing_and_out_of_scope_alts() {
        // A record whose ALT 2 was dropped as out-of-scope (symbolic/breakend) gets
        // ONE slot. REF (0), missing (-1) and the dropped ALT must not leak into it.
        let gt = vec![0i32, 1, 2, -1, 3];
        let m = PresenceMasks::from_dense(&gt, 5, &[1]);
        assert_eq!(m.mask(0)[0], 1u64 << 1);
    }

    #[test]
    fn presence_masks_cost_one_bit_per_column_per_slot() {
        // The whole point of the type: 200 columns cost 4 words per slot, not 200
        // i32s per record (issue #155).
        let gt = vec![0i32; 200];
        let m = PresenceMasks::from_dense(&gt, 200, &[1, 2]);
        assert_eq!(m.mask(0).len(), 4);
        assert_eq!(m.mask(1).len(), 4);
    }

    #[test]
    fn presence_masks_set_high_columns_in_the_right_word() {
        let mut gt = vec![0i32; 200];
        for c in [0usize, 63, 64, 65, 199] {
            gt[c] = 1;
        }
        let m = PresenceMasks::from_dense(&gt, 200, &[1]);
        let w = m.mask(0);
        assert_eq!(w[0], (1u64 << 0) | (1u64 << 63));
        assert_eq!(w[1], (1u64 << 0) | (1u64 << 1));
        assert_eq!(w[3], 1u64 << (199 - 192));
    }

    #[test]
    fn or_mask_into_handles_a_word_aligned_row() {
        // vi = 0, columns = 64: s == 0, so the carry branch must not run at all
        // (`>> 64` is UB).
        let gt: Vec<i32> = (0..64).map(|c| if c % 3 == 0 { 1 } else { 0 }).collect();
        let m = PresenceMasks::from_dense(&gt, 64, &[1]);
        let mut got = vec![0u64; 1];
        or_mask_into(&mut got, 0, 0, m.mask(0), 64);
        let mut want = 0u64;
        for c in 0..64 {
            if c % 3 == 0 {
                want |= 1u64 << c;
            }
        }
        assert_eq!(got[0], want);
    }

    #[test]
    fn or_mask_into_places_a_high_column_in_the_rows_last_word() {
        // columns = 100, vi = 1 -> the row spans bits 100..200 (last bit 199), i.e.
        // words 1..=3: word 3 (bits 192..256) is the row's OWN last word (it holds
        // bits 192..199), not a foreign next-row word -- so the check below is the
        // bit landing in exactly word 3, at exactly bit 7, and nowhere else (a wrong
        // `w0`/`s`/`last` computation would misplace it or panic on an out-of-bounds
        // `words[hi]`).
        let mut gt = vec![0i32; 100];
        gt[99] = 1;
        let m = PresenceMasks::from_dense(&gt, 100, &[1]);
        let mut got = vec![0u64; 4];
        or_mask_into(&mut got, 0, 100, m.mask(0), 100);
        assert_eq!(got[(199) >> 6], 1u64 << (199 & 63));
    }

    #[test]
    fn or_mask_into_never_writes_past_the_rows_last_word() {
        // columns = 70, vi = 1 -> base = 70, w0 = 1, s = 6, last = (70+70-1)>>6 = 2.
        // gt[69] = 1 puts the only set bit in mask word 1 (local bit 69&63 = 5), so
        // `lo = w0+1 = 2` and the natural carry destination is `hi = 3`. `3 <= last`
        // is FALSE, so the `hi <= last` guard must suppress that write.
        //
        // `got` is sized to EXACTLY `last + 1 = 3` words -- the row's own last word,
        // nothing beyond it. `pack_presence_par` hands each rayon task a
        // word-disjoint slice sized just like this, so if the guard were missing (or
        // wrong), the carry write `words[3] |= ...` would index a length-3 slice out
        // of bounds and PANIC here, rather than silently corrupting a neighbouring
        // task's words as it would in production.
        let mut gt = vec![0i32; 70];
        gt[69] = 1;
        let m = PresenceMasks::from_dense(&gt, 70, &[1]);
        let mut got = vec![0u64; 3];
        or_mask_into(&mut got, 0, 70, m.mask(0), 70);
        assert_eq!(got[0], 0);
        assert_eq!(got[1], 0);
        assert_eq!(got[2], 1u64 << 11);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        // The migration's safety argument: packing from a mask must reproduce, bit
        // for bit, what the per-allele scan produced. Generates `columns` that are
        // and are not multiples of 64, rows at every shift, missing calls, and
        // multiallelic records.
        #[test]
        fn or_mask_into_matches_the_allele_scan(
            columns in 1usize..300,
            vi in 0usize..40,
            src in 1u16..4,
            seed in any::<u64>(),
        ) {
            // xorshift64, matching `test_par_packing_matches_seq` in this module.
            let mut state = seed | 1;
            let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
            // Alleles in -1..=3: missing, REF, and ALTs 1..3, so `src` both matches and misses.
            let gt: Vec<i32> = (0..columns).map(|_| (next() % 5) as i32 - 1).collect();

            // Reference: the pre-mask element scan.
            let total_words = ((vi + 1) * columns).div_ceil(64);
            let mut want = vec![0u64; total_words];
            for (col, &g) in gt.iter().enumerate() {
                if g == src as i32 {
                    let flat = vi * columns + col;
                    want[flat >> 6] |= 1u64 << (flat & 63);
                }
            }

            let masks = PresenceMasks::from_dense(&gt, columns, &[src]);
            let mut got = vec![0u64; total_words];
            or_mask_into(&mut got, 0, vi * columns, masks.mask(0), columns);
            prop_assert_eq!(got, want);
        }
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

        let mk = |calls: AtomCalls| PendingAtom {
            pos: 100,
            ilen: 0,
            alt: b"A".to_vec(),
            source_alt_index: src_alt,
            calls,
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
            global_idx: -1,
        };

        let mut dense_bits = vec![0u64; 1];
        let masks = std::sync::Arc::new(PresenceMasks::from_dense(&gt, columns, &[src_alt]));
        pack_row(
            &mut dense_bits,
            0,
            0,
            &mk(AtomCalls::Masks { masks, slot: 0 }),
            columns,
        );

        let mut sparse_bits = vec![0u64; 1];
        pack_row(
            &mut sparse_bits,
            0,
            0,
            &mk(AtomCalls::Sparse(std::sync::Arc::new(
                crate::record_source::Calls::Sparse(carriers),
            ))),
            columns,
        );

        assert_eq!(sparse_bits, dense_bits);
    }

    #[test]
    fn pack_row_sparse_matches_dense_across_word_boundaries() {
        // columns = 100 puts variant 1's row across words 1/2/3, exercising the
        // `w = (flat >> 6) - word_base` arithmetic across several word indices.
        // Both calls below pass `word_base = 0`; a nonzero `word_base` is not
        // covered here (see the `Calls::Sparse` proptest coverage in
        // `test_par_packing_matches_seq` for that).
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
        let mk = |calls: AtomCalls| PendingAtom {
            pos: 1,
            ilen: 0,
            alt: b"A".to_vec(),
            source_alt_index: 1,
            calls,
            seq: 0,
            info_vals: Vec::new(),
            format_vals: Arc::new(FormatVals::Dense(Vec::new())),
            global_idx: -1,
        };
        let words = (columns * 2).div_ceil(64);

        let mut d = vec![0u64; words];
        let masks = std::sync::Arc::new(PresenceMasks::from_dense(&gt, columns, &[1]));
        pack_row(
            &mut d,
            0,
            1,
            &mk(AtomCalls::Masks { masks, slot: 0 }),
            columns,
        );
        let mut s = vec![0u64; words];
        pack_row(
            &mut s,
            0,
            1,
            &mk(AtomCalls::Sparse(std::sync::Arc::new(
                crate::record_source::Calls::Sparse(carriers),
            ))),
            columns,
        );
        assert_eq!(s, d);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        // Parallel packing reproduces sequential packing bit-for-bit, for arbitrary
        // shapes (incl. v not a multiple of the word-aligned block size), allele
        // indices (incl. missing -1 and out-of-range values), source alts, and a
        // mix of `Calls::Dense`/`Calls::Sparse` atoms -- production hits
        // `Calls::Sparse` at a nonzero `word_base` via this same parallel path
        // (`pack_presence_par`'s per-block `word_base = c * words_per_block`), so
        // this generator must exercise that combination too, not just `Dense`.
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
                if next() % 2 == 0 {
                    atoms.push(sparse_atom(&gt, src));
                } else {
                    atoms.push(atom(gt, src));
                }
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

    #[test]
    fn batch_records_bounds_staged_bytes_at_every_cohort_width() {
        for &s in &[100usize, 2_000, 32_000, 128_000, 500_000] {
            let columns = s * 2;
            let bytes = batch_records(columns) * columns * 4;
            assert!(
                bytes <= RAW_STAGE_BYTES,
                "S={s} stages {bytes} B against a {RAW_STAGE_BYTES} B budget"
            );
        }
    }

    #[test]
    fn batch_records_holds_todays_value_only_for_narrow_cohorts() {
        // The cap binds up to columns = RAW_STAGE_BYTES/(4*MAX_BATCH_RECORDS) =
        // 16,384, i.e. S = 8,192 at ploidy 2.
        assert_eq!(batch_records(16_384), MAX_BATCH_RECORDS);
        // Wider cohorts DO change: S=32,000 -- inside RamLaw::PGEN's fitted domain --
        // stages 262 records rather than 1,024. That is the fix working, not a
        // regression, and it is precisely why the law must be re-fitted (Task 9)
        // rather than carried over.
        assert_eq!(batch_records(64_000), 262);
    }

    #[test]
    fn the_batch_floor_is_the_documented_limit_of_the_bound() {
        // Past columns == RAW_STAGE_BYTES/(4*MIN_BATCH_RECORDS) -- about S=1,000,000
        // at ploidy 2 -- the floor binds and staging resumes growing with S. Asserted
        // rather than hidden: this is the boundary of what the budget promises.
        assert_eq!(batch_records(4_000_000), MIN_BATCH_RECORDS);
    }

    #[test]
    fn pack_window_bounds_retained_mask_bytes() {
        for &s in &[100usize, 2_000, 128_000, 500_000] {
            let columns = s * 2;
            let bytes = pack_window(columns) * columns.div_ceil(64) * 8;
            assert!(bytes <= MASK_STAGE_BYTES, "S={s} retains {bytes} B");
        }
    }

    #[test]
    fn pack_window_stays_at_todays_value_until_far_past_the_fitted_domain() {
        // Masks are what keep this non-binding in the normal regime: budgeting the
        // same bytes over raw calls would give ~65 records at S=128,000.
        assert_eq!(pack_window(256_000), MAX_PACK_WINDOW); // S=128,000
        assert!(pack_window(1_000_000) < MAX_PACK_WINDOW); // S=500,000
    }
}

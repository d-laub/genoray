//! `Svar2Source`: a [`RecordSource`] over a **finished** SVAR2 contig.
//!
//! This is the seam that lets `SparseVar2.write_view` re-emit a region+sample
//! subset of an existing store by re-running the ordinary conversion pipeline
//! (`process_chromosome` -> cost model -> merge -> writer) rather than by
//! surgically rewriting sidecars. The store is read back through the normal
//! batched query path (`query::read_ranges` + `BatchResult::decode_hap`, one
//! gather for all (region, sample, hap) triples), the decoded calls are
//! re-grouped variant-major, and each group is handed to the pipeline as a
//! synthetic [`RawRecord`].
//!
//! ## Why the synthetic REF bases are safe
//!
//! `write_view` runs with **no FASTA** (`fasta_path = None`), which disables the
//! only two REF-bases-dependent normalization steps (`validate_ref` and
//! `left_align`) — and it does not need them: a finished store is already
//! atomic, biallelic and left-aligned. Everything downstream
//! (`classify_variant` / `pack_variant`) reads only `ilen` + `alt`, never the
//! REF bases. So `to_raw_record` only has to synthesize REF bytes that survive
//! `atomize_record` unchanged (see its rules below), not REF bytes that match
//! any reference genome.
//!
//! ## MVP scope
//!
//! Genotypes only: `info_raw` / `format_raw` are empty. Faithful INFO/FORMAT
//! carry-through would need the store's `FieldView` per variant and is deferred.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::ConversionError;
use crate::query::{ContigReader, read_ranges};
use crate::record_source::{RawRecord, RecordSource};

/// How a variant's overlap with a query region is judged.
///
/// Mirrors genoray's established public `regions_overlap: "pos" | "record" |
/// "variant"` contract (`python/genoray/_svar/_regions.py`). Two of the three
/// modes are **POS-membership** rules — only `variant` is an extent rule:
///
/// | mode      | keep iff                       |
/// |-----------|--------------------------------|
/// | `pos`     | `q_start <= POS < q_end`       |
/// | `record`  | `q_start <= POS < q_end + 1`   |
/// | `variant` | the variant's *extent* overlaps `[q_start, q_end)` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapMode {
    /// Keep a call only if its POS lies in `[q_start, q_end)`. Deletions whose
    /// *extent* reaches into the region but whose POS lies before it are pruned.
    Pos,
    /// Keep a call only if its POS lies in `[q_start, q_end + 1)` — i.e. `Pos`
    /// with the region end widened by exactly one base, so a variant at
    /// `POS == q_end` is kept. Still a POS rule: a deletion whose extent reaches
    /// into the region but whose POS precedes `q_start` is pruned.
    Record,
    /// Keep every call whose *extent* overlaps `[q_start, q_end)` — i.e. every
    /// call the reader returns, with no POS filter (deletions spanning the
    /// region start included).
    Variant,
}

/// Widen `regions` for the reader's windowed extent-overlap search, per
/// `OverlapMode`. `Record` keeps variants at `POS == q_end`, so the *reader*
/// window has to be widened by one base too — an extent-overlap query over
/// `[q_start, q_end)` would never surface them. `Pos`/`Variant` pass `regions`
/// through unchanged. `saturating_add` so a region ending at `u32::MAX` can't
/// wrap; [`keeps`] below is written inclusively for the same reason.
pub fn query_window(regions: &[(u32, u32)], m: OverlapMode) -> Vec<(u32, u32)> {
    match m {
        OverlapMode::Record => regions
            .iter()
            .map(|&(qs, qe)| (qs, qe.saturating_add(1)))
            .collect(),
        OverlapMode::Pos | OverlapMode::Variant => regions.to_vec(),
    }
}

/// Whether a call at `pos` is kept for the *original* (unwidened) query window
/// `[q_start, q_end)` under overlap mode `m`. This is the final POS-precision
/// filter applied to calls a windowed extent-overlap search (over
/// [`query_window`]'s widened bounds) already returned — not a substitute for
/// that search. In particular `Variant` unconditionally returns `true`: its
/// extent-overlap condition is enforced entirely by the windowed search that
/// produced the call, not by this predicate. See [`OverlapMode`]'s doc for the
/// three rules.
pub fn keeps(m: OverlapMode, q_start: u32, q_end: u32, pos: u32) -> bool {
    match m {
        OverlapMode::Pos => q_start <= pos && pos < q_end,
        OverlapMode::Record => q_start <= pos && pos <= q_end,
        OverlapMode::Variant => true,
    }
}

/// A cursor over the variant-major records of one finished contig, restricted to
/// a region set and a sample subset. Built eagerly: `new` decodes every kept hap
/// and groups the calls, `next_record` just drains the result.
pub struct Svar2Source {
    /// Pre-built records in ascending `(pos, ilen, alt)` order. `RawRecord` is
    /// not `Clone`, so records are handed out by move.
    records: std::vec::IntoIter<RawRecord>,
}

impl Svar2Source {
    /// Open `{store_path}/{chrom}` and enumerate the variants carried by
    /// `sample_orig_idx` (original sample column indices, in *output* order)
    /// within `regions`.
    ///
    /// A variant is emitted iff at least one kept haplotype carries it, so
    /// MAC=0 variants (relative to the subset) drop out automatically; carrier
    /// bits from overlapping regions OR together, so a variant seen through two
    /// regions is emitted once.
    pub fn new(
        store_path: &str,
        chrom: &str,
        sample_orig_idx: &[usize],
        ploidy: usize,
        regions: &[(u32, u32)],
        overlap_mode: OverlapMode,
    ) -> Result<Self, ConversionError> {
        let n_samples_orig = read_n_samples(store_path)?;
        for &s in sample_orig_idx {
            if s >= n_samples_orig {
                return Err(ConversionError::Input(format!(
                    "sample index {s} is out of range for a {n_samples_orig}-sample store"
                )));
            }
        }

        let reader =
            ContigReader::open(store_path, chrom, n_samples_orig, ploidy).map_err(|e| {
                ConversionError::Io {
                    context: format!("{store_path}/{chrom}"),
                    source: e,
                }
            })?;

        let n_haps_out = sample_orig_idx.len() * ploidy;
        // Key order is exactly the emission order the pipeline needs: ascending
        // POS, then a stable tiebreak among variants sharing a POS.
        let mut groups: BTreeMap<(u32, i32, Vec<u8>), Vec<bool>> = BTreeMap::new();

        let query_regions: Vec<(u32, u32)> = query_window(regions, overlap_mode);

        // ONE batched gather for every (region, sample, hap): `find_ranges`
        // builds the dense union + search tree once for the whole call, whereas
        // the per-sample `oracle::overlap_sample` would rebuild both per
        // (sample, region) — O(S * R * D log D) on a whole-contig view.
        let batch = read_ranges(&reader, &query_regions, Some(sample_orig_idx));

        // `BatchResult` hap index is `(r * n_samples + s) * ploidy + p` over the
        // *selected* samples: `s` is the position within `sample_orig_idx`, not
        // the original sample column (`find_ranges` resolves `sample_cols[s]` to
        // the original column itself). So `s == s_out` here.
        for (r, &(q_start, q_end)) in regions.iter().enumerate() {
            for s_out in 0..sample_orig_idx.len() {
                for p in 0..ploidy {
                    let hc = batch.decode_hap(&reader, r, s_out, p);
                    let h_out = s_out * ploidy + p;
                    for i in 0..hc.positions.len() {
                        let pos = hc.positions[i];
                        if !keeps(overlap_mode, q_start, q_end, pos) {
                            continue;
                        }
                        groups
                            .entry((pos, hc.ilens[i], hc.alts[i].clone()))
                            .or_insert_with(|| vec![false; n_haps_out])[h_out] = true;
                    }
                }
            }
        }

        let records: Vec<RawRecord> = groups
            .into_iter()
            .map(|((pos, ilen, alt), carriers)| to_raw_record(pos, ilen, &alt, &carriers))
            .collect();

        Ok(Self {
            records: records.into_iter(),
        })
    }
}

impl RecordSource for Svar2Source {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        Ok(self.records.next())
    }
}

/// Cohort size the store was written with, from the top-level `meta.json`. The
/// contig's sidecars are indexed by *original* sample column, so `ContigReader`
/// must be opened with the original cohort size, not the subset's.
pub(crate) fn read_n_samples(store_path: &str) -> Result<usize, ConversionError> {
    let path = Path::new(store_path).join("meta.json");
    if !path.exists() {
        return Err(ConversionError::MissingFile {
            path: path.display().to_string(),
        });
    }
    let text = std::fs::read_to_string(&path).map_err(|e| ConversionError::Io {
        context: path.display().to_string(),
        source: e,
    })?;
    let meta: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ConversionError::Input(format!("malformed {}: {e}", path.display())))?;
    meta["samples"]
        .as_array()
        .map(|a| a.len())
        .ok_or_else(|| ConversionError::Input(format!("{} has no `samples` array", path.display())))
}

/// Rebuild one variant-major `RawRecord` from a decoded `(pos, ilen, alt)` group
/// and its carrier bitset.
///
/// The REF bases are synthetic (the store does not keep them) and are chosen so
/// that `atomize_record` passes the record through unchanged:
///
/// * **SNP** (`ilen == 0`): the store's `alt` is the single ALT base. REF must be
///   a *different* base — an equal REF would make `atomize_biallelic` drop the
///   record as a non-variant.
/// * **INS** (`ilen > 0`): the store's `alt` is `[anchor, inserted...]`, so
///   `REF = [anchor]` reproduces the clean-anchor form (`r[0] == a[0]`) and
///   avoids a spurious SNV+INS split. Long ALTs re-spill to the LUT downstream.
/// * **pure DEL** (`ilen < 0`): the store keeps no bytes at all (`alt` is empty)
///   because `encode_pure_del` discards them; only the length matters. `ilen =
///   alt_len - ref_len = 1 - ref_len`, so `ref_len = 1 - ilen`; emit `ref_len`
///   filler bases with `alt = [REF[0]]` (a clean anchor).
fn to_raw_record(pos: u32, ilen: i32, alt: &[u8], carriers: &[bool]) -> RawRecord {
    let (reference, alts): (Vec<u8>, Vec<Vec<u8>>) = if ilen == 0 {
        let base = alt[0];
        let refb = if base == b'A' { b'C' } else { b'A' };
        (vec![refb], vec![alt.to_vec()])
    } else if ilen > 0 {
        (vec![alt[0]], vec![alt.to_vec()])
    } else {
        let rlen = (1 - ilen) as usize;
        let refv = vec![b'N'; rlen];
        let anchor = vec![refv[0]];
        (refv, vec![anchor])
    };
    let gt = carriers.iter().map(|&c| if c { 1i32 } else { 0 }).collect();
    RawRecord {
        pos,
        reference,
        alts,
        gt,
        info_raw: Vec::new(),
        format_raw: Vec::new(),
    }
}

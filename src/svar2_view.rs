//! Region/sample **selection** for SVAR2 views — the predicate shared by BOTH
//! routings of `svar2_slice`.
//!
//! `query_window` widens a region per overlap mode before the tree search;
//! `keeps` applies the final POS-precision filter. Both routings call these, so
//! `reroute=True` and `reroute=False` provably select the IDENTICAL variant set.
//! Do not re-derive this logic anywhere else.

use std::path::Path;

use crate::error::ConversionError;

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

/// Parse the public `regions_overlap` string into an [`OverlapMode`].
pub fn parse_overlap_mode(s: &str) -> Result<OverlapMode, ConversionError> {
    match s {
        "pos" => Ok(OverlapMode::Pos),
        "record" => Ok(OverlapMode::Record),
        "variant" => Ok(OverlapMode::Variant),
        other => Err(ConversionError::Input(format!(
            "regions_overlap must be 'pos', 'record', or 'variant'; got {other:?}"
        ))),
    }
}

/// True iff ANY allele's anchor-trimmed genomic span overlaps `[q_start, q_end)`.
/// Anchor-trimming removes the shared prefix/suffix of REF vs each ALT so a
/// deletion `ACGT>A` is judged on the deleted `CGT` (offset span `[pos+1,pos+4)`),
/// not the full record span — matching bcftools `--regions-overlap variant`.
/// An allele that fully trims away (a pure insertion) is a zero-width point at
/// the insertion offset and overlaps iff `q_start <= point < q_end`.
pub fn extent_overlaps(
    pos: u32,
    _ref_len: u32,
    alts: &[&[u8]],
    ref_allele: &[u8],
    q_start: u32,
    q_end: u32,
) -> bool {
    for alt in alts {
        // shared prefix
        let mut p = 0usize;
        let max_p = ref_allele.len().min(alt.len());
        while p < max_p && ref_allele[p] == alt[p] {
            p += 1;
        }
        // shared suffix (not crossing the prefix already consumed)
        let mut s = 0usize;
        while s < (ref_allele.len() - p).min(alt.len() - p)
            && ref_allele[ref_allele.len() - 1 - s] == alt[alt.len() - 1 - s]
        {
            s += 1;
        }
        let v_start = pos + p as u32;
        let ref_consumed = ref_allele.len().saturating_sub(p + s) as u32;
        let v_end = v_start + ref_consumed; // may equal v_start for insertions
        let overlaps = if v_end == v_start {
            q_start <= v_start && v_start < q_end // zero-width insertion point
        } else {
            v_start < q_end && q_start < v_end
        };
        if overlaps {
            return true;
        }
    }
    false
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

//! Streaming mutation-catalogue counting with on-the-fly DBS pairing.

use ndarray::Array2;
use rayon::prelude::*;

use crate::dense::DenseClass;
use crate::layout::{ContigPaths, MutcatSub};
use crate::mutcat::classify::dbs78_code;
use crate::mutcat::sidecar::{MutcatView, open_sidecar};
use crate::mutcat::{
    DBS78_OFFSET, ID83_OFFSET, N_CODES, SBS96_OFFSET, SBS384_OFFSET, STRAND_NA, UNCLASSIFIED,
};
use crate::query::ContigReader;
use crate::query::sidecar::as_bytes;
use svar2_codec::{decode_snp_2bit, unpack_snp_key_at};

/// One SNV on a single haplotype (already position-sorted & deduped).
#[derive(Debug, Clone, Copy)]
pub struct SnvCall {
    pub pos: u32,
    pub sbs: u8,    // SBS96 class-local index or sentinel
    pub ref_i: u8,  // 2-bit ref base code
    pub alt_i: u8,  // 2-bit alt base code
    pub strand: u8, // STRAND_{T,U,N,B}, or STRAND_NA when no strand.bin
}

/// Emit one unified mutation code per SNV (SBS, or DBS once for an isolated
/// adjacent pair — the pair contributes a single DBS call, not one per
/// member). Sentinels are skipped. Mirrors v1 isolation logic.
pub fn emit_snv_codes(snvs: &[SnvCall], out: &mut impl FnMut(usize)) {
    let n = snvs.len();
    let mut j = 0;
    while j < n {
        let v = snvs[j];
        // try to pair v with the next SNV
        if j + 1 < n && snvs[j + 1].pos == v.pos + 1 {
            let w = snvs[j + 1];
            // isolation: no adjacent SNV before v, none after w.
            let before = j > 0 && snvs[j - 1].pos + 1 == v.pos;
            let after = j + 2 < n && snvs[j + 2].pos == w.pos + 1;
            if !before && !after {
                let code = dbs78_code(
                    decode_snp_2bit(v.ref_i),
                    decode_snp_2bit(v.alt_i),
                    decode_snp_2bit(w.ref_i),
                    decode_snp_2bit(w.alt_i),
                );
                if code != UNCLASSIFIED {
                    out(DBS78_OFFSET + code as usize);
                    j += 2;
                    continue;
                }
            }
        }
        if (v.sbs as usize) < 96 {
            out(SBS96_OFFSET + v.sbs as usize);
            // Strand-resolved SBS384 (same SNV), only when the sidecar carried a
            // strand stream. DBS-paired SNVs `continue` above, so they never reach
            // here — SBS384 is emitted for exactly the SNVs that emit SBS96.
            if v.strand != STRAND_NA {
                out(SBS384_OFFSET + (v.strand as usize) * 96 + v.sbs as usize);
            }
        }
        j += 1;
    }
    #[allow(clippy::assertions_on_constants)]
    {
        debug_assert!(N_CODES == 641);
    }
}

/// The four mutcat sidecar views for one contig, bundled so `count_contig`
/// takes a single handle instead of four.
pub struct Sidecars {
    pub vk_snp: MutcatView,
    pub vk_indel: MutcatView,
    pub dense_snp: MutcatView,
    pub dense_indel: MutcatView,
}

impl Sidecars {
    /// Open all four mutcat sub-stream sidecars for `paths`'s contig. Any
    /// sub-stream with no sidecar on disk opens as an empty `MutcatView`
    /// (`open_sidecar`'s existing missing-file tolerance).
    pub fn open(paths: &ContigPaths) -> std::io::Result<Sidecars> {
        Ok(Sidecars {
            vk_snp: open_sidecar(paths, MutcatSub::VkSnp)?,
            vk_indel: open_sidecar(paths, MutcatSub::VkIndel)?,
            dense_snp: open_sidecar(paths, MutcatSub::DenseSnp)?,
            dense_indel: open_sidecar(paths, MutcatSub::DenseIndel)?,
        })
    }
}

/// Push `v` unless it duplicates the last-pushed position (the `dedup_by_key`
/// half of the old sort+dedup, applied inline during the merge).
#[inline]
fn push_unique(out: &mut Vec<SnvCall>, v: SnvCall) {
    if out.last().is_none_or(|l| l.pos != v.pos) {
        out.push(v);
    }
}

/// Merge two already-position-sorted SNV runs (`a` = var_key, `b` = dense) into
/// `out`, deduped by position. O(len a + len b) — replaces sorting the
/// concatenation. On an equal position the var_key call wins (matches the old
/// stable-sort + `dedup_by_key`, which kept the first / var_key entry).
fn merge_dedup(a: &[SnvCall], b: &[SnvCall], out: &mut Vec<SnvCall>) {
    out.clear();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i].pos <= b[j].pos {
            push_unique(out, a[i]);
            i += 1;
        } else {
            push_unique(out, b[j]);
            j += 1;
        }
    }
    while i < a.len() {
        push_unique(out, a[i]);
        i += 1;
    }
    while j < b.len() {
        push_unique(out, b[j]);
        j += 1;
    }
}

/// Gather + classify one flat `(sample, ploid)` column's SNVs/indels
/// (`col = sample * ploidy + ploid`), accumulating raw (increment-only) counts
/// into `row` (a `(N_CODES,)` slice, shared across the sample's ploidy columns):
/// 1. Gathers that hap's SNVs from `var_key/snp` (per-call, sidecar-indexed by
///    the absolute call index) and `dense/snp` (per carried dense variant),
///    both already position-sorted; `merge_dedup` unions them in O(n) and
///    `emit_snv_codes` runs on-the-fly DBS pairing over the merged run.
/// 2. Gathers that hap's indels from `var_key/indel` + `dense/indel`; each
///    emits `ID83_OFFSET + code` directly (no pairing). A sentinel code
///    (`UNCLASSIFIED`/`NOT_ANNOTATED`, both `>= N_ID83`) pushes the unified
///    code past `N_CODES`, so the single `unified < N_CODES` bound filters
///    both sentinels without naming them.
///
/// `vk`/`dn`/`merged` are caller-owned scratch buffers reused across columns so
/// this does no per-column heap allocation.
#[allow(clippy::too_many_arguments)]
fn count_column_into(
    reader: &ContigReader,
    sidecars: &Sidecars,
    col: usize,
    row: &mut [i64],
    vk: &mut Vec<SnvCall>,
    dn: &mut Vec<SnvCall>,
    merged: &mut Vec<SnvCall>,
) {
    // --- SNVs: var_key/snp (already position-sorted) ---
    vk.clear();
    {
        let vk_range = reader.vk_snp.column(col);
        let (o0, o1) = (vk_range.start, vk_range.end);
        let positions = &reader.vk_snp.positions()[o0..o1];
        let keys = as_bytes(&reader.vk_snp.keys);
        for (i, &pos) in positions.iter().enumerate() {
            let abs_i = o0 + i;
            vk.push(SnvCall {
                pos,
                sbs: sidecars.vk_snp.code_at(abs_i),
                ref_i: sidecars.vk_snp.ref_at(abs_i),
                alt_i: unpack_snp_key_at(keys, abs_i),
                strand: if sidecars.vk_snp.has_strand {
                    sidecars.vk_snp.strand_at(abs_i)
                } else {
                    STRAND_NA
                },
            });
        }
    }
    // --- dense/snp (position-sorted); iterate only carried variants ---
    dn.clear();
    if let Some(dense) = reader.dense_view(DenseClass::Snp) {
        let positions = dense.positions();
        let keys = as_bytes(&dense.keys);
        dense.for_each_carried(col, |dcol| {
            dn.push(SnvCall {
                pos: positions[dcol],
                sbs: sidecars.dense_snp.code_at(dcol),
                ref_i: sidecars.dense_snp.ref_at(dcol),
                alt_i: unpack_snp_key_at(keys, dcol),
                strand: if sidecars.dense_snp.has_strand {
                    sidecars.dense_snp.strand_at(dcol)
                } else {
                    STRAND_NA
                },
            });
        });
    }
    merge_dedup(vk, dn, merged);
    emit_snv_codes(merged, &mut |code| row[code] += 1);

    // --- indels: var_key/indel + dense/indel, each direct (no pairing) ---
    {
        let vk_range = reader.vk_indel.column(col);
        for abs_i in vk_range {
            let unified = ID83_OFFSET + sidecars.vk_indel.code_at(abs_i) as usize;
            if unified < N_CODES {
                row[unified] += 1;
            }
        }
    }
    if let Some(dense) = reader.dense_view(DenseClass::Indel) {
        dense.for_each_carried(col, |dcol| {
            let unified = ID83_OFFSET + sidecars.dense_indel.code_at(dcol) as usize;
            if unified < N_CODES {
                row[unified] += 1;
            }
        });
    }
}

/// Accumulate one contig's full mutation-count matrix into `acc`
/// (`(n_samples, N_CODES)`).
///
/// Parallelized over **samples** with `rayon`: each sample owns one contiguous
/// `N_CODES` row of `acc` and folds its `ploidy` columns straight into that row
/// (`count_column_into`). Rows are disjoint, so no per-job full-matrix
/// accumulator and no reduce step are needed — the old `fold`/`reduce`
/// allocated and summed an `(n_samples, N_CODES)` matrix per rayon split-job,
/// which dominated the runtime. `acc` is the accumulator, so counting straight
/// into its rows IS `acc += this contig`. The result is still order-independent:
/// each sample's counts come only from its own columns.
///
/// `per_sample` (presence) semantics are applied per row AFTER its columns are
/// summed, by clipping to `{0, 1}`: a code counts once if it occurred on ANY of
/// that sample's ploidy columns — reproducing the single-threaded per-sample
/// `seen` bitmap.
pub fn count_contig(
    reader: &ContigReader,
    sidecars: &Sidecars,
    per_sample: bool,
    acc: &mut Array2<i64>,
) {
    let ploidy = reader.ploidy;

    // Write each sample's counts straight into its own `acc` row. `acc` is the
    // accumulator (`acc += this contig`), so `count_column_into`'s `row[c] += 1`
    // onto the acc row IS that accumulation — no intermediate matrix, no memset,
    // no full-matrix add. Rows are disjoint chunks, so this parallelizes cleanly.
    acc.as_slice_mut()
        .expect("count matrix accumulator must be C-contiguous")
        .par_chunks_mut(N_CODES)
        .enumerate()
        .for_each(|(sample, acc_row)| {
            let mut vk = Vec::new();
            let mut dn = Vec::new();
            let mut merged = Vec::new();
            if per_sample {
                // Presence within THIS contig, then OR (add) into acc: count this
                // sample's ploidy columns into a private row, clip to {0,1}, add.
                let mut local = [0i64; N_CODES];
                for p in 0..ploidy {
                    count_column_into(
                        reader,
                        sidecars,
                        sample * ploidy + p,
                        &mut local,
                        &mut vk,
                        &mut dn,
                        &mut merged,
                    );
                }
                for (a, &l) in acc_row.iter_mut().zip(local.iter()) {
                    *a += l.min(1);
                }
            } else {
                for p in 0..ploidy {
                    count_column_into(
                        reader,
                        sidecars,
                        sample * ploidy + p,
                        acc_row,
                        &mut vk,
                        &mut dn,
                        &mut merged,
                    );
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snv(pos: u32, ref_b: u8, alt_b: u8) -> SnvCall {
        use crate::mutcat::classify::sbs96_code;
        SnvCall {
            pos,
            sbs: sbs96_code(b'A', ref_b, alt_b, b'A'), // dummy flanks for the test
            ref_i: svar2_codec::encode_snp_2bit(ref_b),
            alt_i: svar2_codec::encode_snp_2bit(alt_b),
            strand: crate::mutcat::STRAND_NA,
        }
    }

    fn snv_strand(pos: u32, ref_b: u8, alt_b: u8, strand: u8) -> SnvCall {
        SnvCall {
            strand,
            ..snv(pos, ref_b, alt_b)
        }
    }

    #[test]
    fn isolated_snv_emits_sbs96_and_sbs384_when_strand_annotated() {
        use crate::mutcat::{SBS384_OFFSET, STRAND_U};
        // Isolated C>A, no adjacent SNV -> SBS96. sbs code with dummy A_A flanks.
        let sbs = {
            use crate::mutcat::classify::sbs96_code;
            sbs96_code(b'A', b'C', b'A', b'A') as usize
        };
        let snvs = [snv_strand(10, b'C', b'A', STRAND_U)];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        assert_eq!(
            codes,
            vec![sbs, SBS384_OFFSET + (STRAND_U as usize) * 96 + sbs]
        );
    }

    #[test]
    fn isolated_snv_emits_only_sbs96_when_not_strand_annotated() {
        // strand == STRAND_NA (via `snv`) -> no SBS384 emission.
        let snvs = [snv(10, b'C', b'A')];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        assert_eq!(codes.len(), 1);
        assert!(codes[0] < DBS78_OFFSET);
    }

    #[test]
    fn isolated_pair_emits_one_dbs() {
        let snvs = [snv(10, b'A', b'C'), snv(11, b'C', b'A')];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        // AC>CA is DBS index 0 → unified 96; the doublet contributes exactly once.
        assert_eq!(codes, vec![96]);
    }

    #[test]
    fn run_of_three_stays_sbs() {
        let snvs = [
            snv(10, b'C', b'A'),
            snv(11, b'C', b'A'),
            snv(12, b'C', b'A'),
        ];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        assert_eq!(
            codes.len(),
            3,
            "all three SNVs in the run must emit an SBS code"
        );
        assert!(
            codes.iter().all(|&c| c < DBS78_OFFSET),
            "runs of 3 must stay SBS"
        );
    }

    #[test]
    fn non_adjacent_snvs_are_sbs() {
        let snvs = [snv(10, b'C', b'A'), snv(20, b'C', b'A')];
        let mut codes = vec![];
        emit_snv_codes(&snvs, &mut |c| codes.push(c));
        assert!(codes.iter().all(|&c| c < DBS78_OFFSET));
        assert_eq!(codes.len(), 2);
    }
}

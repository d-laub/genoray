//! Streaming mutation-catalogue counting with on-the-fly DBS pairing.

use ndarray::Array2;
use rayon::prelude::*;

use crate::dense::DenseClass;
use crate::layout::{ContigPaths, MutcatSub};
use crate::mutcat::classify::dbs78_code;
use crate::mutcat::sidecar::{MutcatView, open_sidecar};
use crate::mutcat::{DBS78_OFFSET, ID83_OFFSET, N_CODES, SBS96_OFFSET, UNCLASSIFIED};
use crate::query::ContigReader;
use crate::query::sidecar::as_bytes;
use svar2_codec::{decode_snp_2bit, unpack_snp_key_at};

/// One SNV on a single haplotype (already position-sorted & deduped).
#[derive(Debug, Clone, Copy)]
pub struct SnvCall {
    pub pos: u32,
    pub sbs: u8,   // SBS96 class-local index or sentinel
    pub ref_i: u8, // 2-bit ref base code
    pub alt_i: u8, // 2-bit alt base code
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
        }
        j += 1;
    }
    #[allow(clippy::assertions_on_constants)]
    {
        debug_assert!(N_CODES == 257);
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

/// Gather + classify one flat `(sample, ploid)` column's SNVs/indels
/// (`col = sample * ploidy + ploid`, matching
/// `ContigReader::vk_slice`/`dense_carried`'s convention), returning its raw
/// `(N_CODES,)` count row:
/// 1. Gathers that hap's SNVs from `var_key/snp` (per-call, sidecar-indexed
///    by the absolute call index) and `dense/snp` (per dense variant the hap
///    carries, sidecar-indexed by the dense variant column), merges by
///    position, dedups, and runs `emit_snv_codes` for on-the-fly DBS pairing.
/// 2. Gathers that hap's indels from `var_key/indel` + `dense/indel`; each
///    emits `ID83_OFFSET + code` directly (no pairing). A sentinel code
///    (`UNCLASSIFIED`/`NOT_ANNOTATED`, both `>= N_ID83`) pushes the unified
///    code past `N_CODES`, so the single `unified < N_CODES` bound filters
///    both sentinels without naming them.
///
/// Every emitted code increments its slot by one (raw counts) — this column
/// is independent of every other column, so it can run on its own thread;
/// per-sample presence-clipping happens once, after `count_contig` reduces
/// a sample's columns together.
fn count_column(reader: &ContigReader, sidecars: &Sidecars, col: usize) -> Vec<i64> {
    let mut row = vec![0i64; N_CODES];
    let mut bump = |code: usize| {
        row[code] += 1;
    };

    // --- SNVs: var_key/snp + dense/snp, merged by position ---
    let mut snvs: Vec<SnvCall> = Vec::new();
    {
        let vk_range = reader.vk_snp.column(col);
        let (o0, o1) = (vk_range.start, vk_range.end);
        let positions = &reader.vk_snp.positions()[o0..o1];
        let keys = as_bytes(&reader.vk_snp.keys);
        for (i, &pos) in positions.iter().enumerate() {
            let abs_i = o0 + i;
            snvs.push(SnvCall {
                pos,
                sbs: sidecars.vk_snp.code_at(abs_i),
                ref_i: sidecars.vk_snp.ref_at(abs_i),
                alt_i: unpack_snp_key_at(keys, abs_i),
            });
        }
    }
    if let Some(dense) = reader.dense_view(DenseClass::Snp) {
        let keys = as_bytes(&dense.keys);
        for (dcol, &pos) in dense.positions().iter().enumerate() {
            if dense.carried(col, dcol) {
                snvs.push(SnvCall {
                    pos,
                    sbs: sidecars.dense_snp.code_at(dcol),
                    ref_i: sidecars.dense_snp.ref_at(dcol),
                    alt_i: unpack_snp_key_at(keys, dcol),
                });
            }
        }
    }
    snvs.sort_by_key(|s| s.pos);
    snvs.dedup_by_key(|s| s.pos);
    emit_snv_codes(&snvs, &mut bump);

    // --- indels: var_key/indel + dense/indel, each direct (no pairing) ---
    {
        let vk_range = reader.vk_indel.column(col);
        for abs_i in vk_range {
            let unified = ID83_OFFSET + sidecars.vk_indel.code_at(abs_i) as usize;
            if unified < N_CODES {
                bump(unified);
            }
        }
    }
    if let Some(dense) = reader.dense_view(DenseClass::Indel) {
        for dcol in 0..dense.n_dense_variants {
            if dense.carried(col, dcol) {
                let unified = ID83_OFFSET + sidecars.dense_indel.code_at(dcol) as usize;
                if unified < N_CODES {
                    bump(unified);
                }
            }
        }
    }

    row
}

/// Accumulate one contig's full mutation-count matrix into `acc`
/// (`(n_samples, N_CODES)`).
///
/// Parallelized over flat `(sample, ploid)` columns with `rayon`:
/// `count_column` gathers/classifies each column independently and returns
/// its own raw (increment-only) `(N_CODES,)` row. `fold` sums each thread's
/// columns into a private `(n_samples, N_CODES)` accumulator (a column's row
/// lands in `local[[col / ploidy, ..]]`), and `reduce` sums those private
/// accumulators together with elementwise `+`. Integer addition is
/// associative & commutative, so the resulting per-sample sums are identical
/// no matter how columns are split across threads.
///
/// `per_sample` (presence) semantics are applied AFTER that full
/// column->sample reduce, by clipping each sample's row to `{0, 1}`: a code
/// counts once if it occurred on ANY of that sample's ploidy columns. This
/// exactly reproduces the single-threaded per-sample `seen` bitmap (which
/// marked a code at most once per sample across all its ploids), and doing
/// the clip once — post-reduce, on the fully-summed row — makes it
/// independent of fold/column order too.
pub fn count_contig(
    reader: &ContigReader,
    sidecars: &Sidecars,
    per_sample: bool,
    acc: &mut Array2<i64>,
) {
    let ploidy = reader.ploidy;
    let n_samples = reader.n_samples;
    let n_cols = n_samples * ploidy;

    let mut sample_rows = (0..n_cols)
        .into_par_iter()
        .fold(
            || Array2::<i64>::zeros((n_samples, N_CODES)),
            |mut local, col| {
                let sample = col / ploidy;
                for (code, v) in count_column(reader, sidecars, col).into_iter().enumerate() {
                    local[[sample, code]] += v;
                }
                local
            },
        )
        .reduce(|| Array2::<i64>::zeros((n_samples, N_CODES)), |a, b| a + b);

    if per_sample {
        sample_rows.mapv_inplace(|v| v.min(1));
    }

    *acc += &sample_rows;
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
        }
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

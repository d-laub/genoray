//! Test/gvl-facing query wrappers: `overlap_sample` (per-sample, tree-per-
//! query reference implementation), the split read-bound gather
//! (`gather_ranges_readbound`), and the `decode_keyref` test-oracle wrappers
//! (`decode_keyref_pub`/`decode_keyref_alt_pub`). A later task (M2's Task 6)
//! renames these off the top-level `query::` path; this task moves them here
//! verbatim and keeps the old paths alive via `query/mod.rs` re-exports.

use crate::rvk;
use crate::spine::{self, KeyRef};

use super::decode::{HapCalls, QueryResult, decode_keyref};
use super::gather::{BatchResultSplit, PresenceBitWriter, RangesBundle};
use super::reader::ContigReader;
use super::sidecar::{as_bytes, as_u32};

/// Test-facing public wrapper around `decode_keyref`: decode `(position, key)`
/// against `reader`'s long-allele LUT and return just `ilen`. Lets gvl-side
/// test oracles decode a raw `KeyRef` (as surfaced by `BatchResultSplit`)
/// without duplicating the SNP/indel/long-INS decode logic.
///
/// Intentionally public (not `cfg(test)`-gated): this ships in `genoray_core`'s
/// public API as the gvl-side read-bound parity-oracle surface, alongside
/// `decode_keyref_alt_pub` and `bits_get_bit`.
pub fn decode_keyref_pub(position: u32, key: u32, reader: &ContigReader) -> i32 {
    decode_keyref(KeyRef { position, key }, reader.lut.as_ref()).ilen
}

/// Like `decode_keyref_pub`, but also returns the decoded ALT bytes so test
/// oracles can assert on alleles (not just `ilen`, which is `0` for every SNP
/// regardless of which base it decodes to).
///
/// Intentionally public (not `cfg(test)`-gated): part of the gvl-side
/// read-bound parity-oracle surface alongside `decode_keyref_pub` and
/// `bits_get_bit`.
pub fn decode_keyref_alt_pub(position: u32, key: u32, reader: &ContigReader) -> (i32, Vec<u8>) {
    let call = decode_keyref(KeyRef { position, key }, reader.lut.as_ref());
    (call.ilen, call.alt)
}

/// Return every variant that `sample` carries overlapping `[q_start, q_end)`, per
/// haplotype, position-sorted, unioning the var_key and dense sub-streams. M5's
/// public contract, re-expressed on the M6.1 spine: gather uniform KeyRefs, do
/// the final `var_key ⋈ dense` 2-way merge, then decode.
pub fn overlap_sample(
    reader: &ContigReader,
    sample: usize,
    q_start: u32,
    q_end: u32,
) -> QueryResult {
    let ploidy = reader.ploidy;
    let lut = reader.lut.as_ref();
    let dense = reader.dense_union();
    let (ds, de) = dense.overlap(q_start, q_end);

    let mut per_hap = Vec::with_capacity(ploidy);
    for p in 0..ploidy {
        let col = sample * ploidy + p; // flat column
        let hap = col; // sample-major hap index == flat column
        let vk = reader.vk_slice(col, sample, p, q_start, q_end);
        let dn = reader.dense_carried(&dense, hap, ds, de, q_start);
        let merged = spine::merge_keys(vec![vk, dn]);

        let mut hc = HapCalls::default();
        for kr in merged {
            let c = decode_keyref(kr, lut);
            hc.positions.push(c.position);
            hc.ilens.push(c.ilen);
            hc.alts.push(c.alt);
        }
        per_hap.push(hc);
    }
    QueryResult { per_hap }
}

/// Tree-free, union-free gather: replay a `RangesBundle` into a split-dense
/// `BatchResultSplit`. Builds NO `SearchTree` and never calls `dense_union()` —
/// each region's dense windows come from the per-class `dense_snp_range` /
/// `dense_indel_range` computed in `find_ranges`. The var_key channel is
/// identical to `gather_ranges`; only the dense side is split per class.
#[allow(clippy::needless_range_loop)]
pub fn gather_ranges_readbound(reader: &ContigReader, rb: &RangesBundle) -> BatchResultSplit {
    let ploidy = rb.ploidy;
    let n_samples = rb.n_samples;
    let n_regions = rb.n_regions;
    let hpr = n_samples * ploidy;

    let snp_positions = reader.vk_snp.positions();
    let snp_keys = as_bytes(&reader.vk_snp.keys);
    let indel_positions = reader.vk_indel.positions();
    let indel_keys = as_u32(&reader.vk_indel.keys);

    // Dense class tables (may be absent).
    let d_snp = reader.dense_snp.as_ref();
    let d_indel = reader.dense_indel.as_ref();
    let d_snp_pos: &[u32] = d_snp.map(|d| d.positions()).unwrap_or(&[]);
    let d_indel_pos: &[u32] = d_indel.map(|d| d.positions()).unwrap_or(&[]);

    // --- dense channel windows (per region), decoded to uniform keys once ---
    let mut dense_snp: Vec<KeyRef> = Vec::new();
    let mut dense_snp_range: Vec<(usize, usize)> = Vec::with_capacity(n_regions);
    let mut dense_indel: Vec<KeyRef> = Vec::new();
    let mut dense_indel_range: Vec<(usize, usize)> = Vec::with_capacity(n_regions);
    for r in 0..n_regions {
        let (ss, se) = rb.dense_snp_range[r];
        let base = dense_snp.len();
        if let Some(d) = d_snp {
            let keys = as_bytes(&d.keys);
            for j in ss..se {
                dense_snp.push(KeyRef {
                    position: d_snp_pos[j],
                    key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, j)),
                });
            }
        }
        dense_snp_range.push((base, dense_snp.len()));

        let (is_, ie_) = rb.dense_indel_range[r];
        let base = dense_indel.len();
        if let Some(d) = d_indel {
            let keys = as_u32(&d.keys);
            for j in is_..ie_ {
                dense_indel.push(KeyRef {
                    position: d_indel_pos[j],
                    key: keys[j],
                });
            }
        }
        dense_indel_range.push((base, dense_indel.len()));
    }

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut snp_presence = PresenceBitWriter::new();
    let mut indel_presence = PresenceBitWriter::new();

    for r in 0..n_regions {
        let qs = rb.region_starts[r];
        let (ss, se) = rb.dense_snp_range[r];
        let (is_r, ie_r) = rb.dense_indel_range[r];
        for si in 0..n_samples {
            let orig_s = rb.sample_cols[si];
            for p in 0..ploidy {
                let col = orig_s * ploidy + p;
                let hap = col;
                let row = r * hpr + si * ploidy + p;

                // --- var_key gather (identical to gather_ranges) ---
                let (vs, ve) = rb.vk_snp_range[row];
                let mut snp_run: Vec<KeyRef> = Vec::new();
                for (j, &pos) in snp_positions.iter().enumerate().take(ve).skip(vs) {
                    if qs < pos + 1 {
                        snp_run.push(KeyRef {
                            position: pos,
                            key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                        });
                    }
                }
                let (vis, vie) = rb.vk_indel_range[row];
                let mut indel_run: Vec<KeyRef> = Vec::new();
                for j in vis..vie {
                    let pos = indel_positions[j];
                    let v_end = pos + 1 + rvk::deletion_len(indel_keys[j]);
                    if qs < v_end {
                        indel_run.push(KeyRef {
                            position: pos,
                            key: indel_keys[j],
                        });
                    }
                }
                let merged = spine::merge_keys(vec![snp_run, indel_run]);
                vk.extend_from_slice(&merged);
                vk_off.push(vk.len());

                // --- dense/snp presence bits over [ss..se) ---
                let nbits = se - ss;
                snp_presence.push_hap(nbits, |k| {
                    let j = ss + k;
                    match d_snp {
                        // snp v_end = pos + 1; left-overlap re-check qs < v_end.
                        Some(d) => d.carried(hap, j) && qs < d_snp_pos[j] + 1,
                        None => false,
                    }
                });

                // --- dense/indel presence bits over [is_r..ie_r) ---
                let nbits = ie_r - is_r;
                indel_presence.push_hap(nbits, |k| {
                    let j = is_r + k;
                    match d_indel {
                        Some(d) => {
                            let keys = as_u32(&d.keys);
                            let v_end = d_indel_pos[j] + 1 + rvk::deletion_len(keys[j]);
                            d.carried(hap, j) && qs < v_end
                        }
                        None => false,
                    }
                });
            }
        }
    }

    let (dense_snp_present, dense_snp_present_off) = snp_presence.into_parts();
    let (dense_indel_present, dense_indel_present_off) = indel_presence.into_parts();

    BatchResultSplit {
        n_regions,
        n_samples,
        ploidy,
        vk,
        vk_off,
        dense_snp,
        dense_snp_range,
        dense_snp_present,
        dense_snp_present_off,
        dense_indel,
        dense_indel_range,
        dense_indel_present,
        dense_indel_present_off,
    }
}

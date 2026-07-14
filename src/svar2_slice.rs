//! `slice_contig_genos`: the `reroute=False` direct array-slicer for one
//! finished SVAR2 contig's genotype sidecars.
//!
//! Unlike `svar2_source::Svar2Source` (which re-runs the ordinary conversion
//! pipeline over synthesized `RawRecord`s — `reroute=True`), this module reads
//! a finished store's mmap'd sidecars directly and writes a region/sample
//! subset as a new finished contig's sidecars: `var_key/{snp,indel}/*`,
//! `dense/{snp,indel}/*`, the shared indel long-allele LUT, and `max_del`.
//! O(output) memory, no cost model, no pipeline re-run.
//!
//! Reuses `svar2_source::{query_window, keeps}` verbatim so `reroute=False`
//! and `reroute=True` select the identical variant set — the query-window
//! widening and per-mode keep predicate are the SAME code, not a re-derived
//! copy of the semantics. The tree-narrowed windows come from
//! `ContigReader::{vk_snp_overlap, vk_indel_overlap, dense_snp_overlap,
//! dense_indel_overlap}` — the exact same per-region search `find_ranges`
//! (the batched query path `Svar2Source` runs through) uses, so a call is
//! windowed-in here iff it would be windowed-in there.
//!
//! Fields, mutcat, and the top-level `meta.json` are out of scope here (later
//! tasks in the `write_view(reroute=False)` plan).

use std::collections::HashSet;
use std::fs;
use std::ops::Range;
use std::path::Path;

use ndarray::Array1;
use ndarray_npy::write_npy;

use crate::bits::set_bit;
use crate::error::ConversionError;
use crate::layout::{self, ContigPaths};
use crate::max_del;
use crate::query::ContigReader;
use crate::query::sidecar::{DenseView, as_bytes, as_u32};
use crate::rvk::{deletion_len, pack_snp_keys, unpack_snp_key_at};
use crate::svar2_source::{OverlapMode, keeps, query_window, read_n_samples};

/// Slice `{src_store}/{chrom}`'s genotype sidecars down to `sample_orig_idx`
/// (original sample column indices, in *output* order) and `regions`, writing
/// the result to `{out_store}/{chrom}`. Returns the number of distinct
/// variants written (a var_key call is per-carrier; a variant carried by
/// several haplotypes is counted once).
///
/// `overlap` and `regions` select calls with EXACTLY `Svar2Source`'s
/// semantics (`query_window` widens the search window per mode, `keeps`
/// applies the final POS-precision filter) — see `svar2_source::OverlapMode`.
pub fn slice_contig_genos(
    src_store: &str,
    out_store: &str,
    chrom: &str,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    overlap: OverlapMode,
) -> Result<usize, ConversionError> {
    let n_samples_orig = read_n_samples(src_store)?;
    for &s in sample_orig_idx {
        if s >= n_samples_orig {
            return Err(ConversionError::Input(format!(
                "sample index {s} is out of range for a {n_samples_orig}-sample store"
            )));
        }
    }

    let reader = ContigReader::open(src_store, chrom, n_samples_orig, ploidy).map_err(|e| {
        ConversionError::Io {
            context: format!("{src_store}/{chrom}"),
            source: e,
        }
    })?;

    let query_regions = query_window(regions, overlap);
    let out_paths = ContigPaths::new(out_store, chrom);

    // ---- var_key ----
    let (vk_snp_pos, vk_snp_codes, vk_snp_offsets) = slice_var_key_snp(
        &reader,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
    );
    let vk_snp_dir = out_paths.var_key_snp_dir();
    create_dir(&vk_snp_dir)?;
    write_bytes(
        &layout::positions(&vk_snp_dir),
        bytemuck::cast_slice(&vk_snp_pos),
    )?;
    write_bytes(&layout::alleles(&vk_snp_dir), &pack_snp_keys(&vk_snp_codes))?;
    write_offsets(&layout::offsets(&vk_snp_dir), &vk_snp_offsets)?;

    let (vk_indel_pos, vk_indel_keys, vk_indel_offsets) = slice_var_key_indel(
        &reader,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
    );
    let vk_indel_dir = out_paths.var_key_indel_dir();
    create_dir(&vk_indel_dir)?;
    write_bytes(
        &layout::positions(&vk_indel_dir),
        bytemuck::cast_slice(&vk_indel_pos),
    )?;
    write_bytes(
        &layout::alleles(&vk_indel_dir),
        bytemuck::cast_slice(&vk_indel_keys),
    )?;
    write_offsets(&layout::offsets(&vk_indel_dir), &vk_indel_offsets)?;

    // ---- dense ----
    let (dense_snp_pos, dense_snp_alleles, dense_snp_bits) = slice_dense(
        reader.dense_snp.as_ref(),
        true,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
        |qsw, qew| reader.dense_snp_overlap(qsw, qew),
    );
    let dense_snp_dir = out_paths.dense_snp_dir();
    create_dir(&dense_snp_dir)?;
    write_bytes(
        &layout::positions(&dense_snp_dir),
        bytemuck::cast_slice(&dense_snp_pos),
    )?;
    write_bytes(&layout::alleles(&dense_snp_dir), &dense_snp_alleles)?;
    write_bytes(&layout::genotypes(&dense_snp_dir), &dense_snp_bits)?;

    let (dense_indel_pos, dense_indel_alleles, dense_indel_bits) = slice_dense(
        reader.dense_indel.as_ref(),
        false,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
        |qsw, qew| reader.dense_indel_overlap(qsw, qew),
    );
    let dense_indel_dir = out_paths.dense_indel_dir();
    create_dir(&dense_indel_dir)?;
    write_bytes(
        &layout::positions(&dense_indel_dir),
        bytemuck::cast_slice(&dense_indel_pos),
    )?;
    write_bytes(&layout::alleles(&dense_indel_dir), &dense_indel_alleles)?;
    write_bytes(&layout::genotypes(&dense_indel_dir), &dense_indel_bits)?;

    // ---- shared indel long-allele LUT: copied verbatim (row indices in the
    // sliced indel keys point into it unchanged; a subset can only leave rows
    // unreferenced, never invalidate one). ----
    let src_paths = ContigPaths::new(src_store, chrom);
    create_dir(&out_paths.shared_indel_dir())?;
    if src_paths.long_alleles_bin().exists() {
        copy_file(&src_paths.long_alleles_bin(), &out_paths.long_alleles_bin())?;
        copy_file(
            &src_paths.long_allele_offsets(),
            &out_paths.long_allele_offsets(),
        )?;
    }

    // ---- max_del: recomputed over the OUTPUT's own (already-written) indel
    // key streams, not copied — the subset's per-column/per-contig maxima are
    // generally tighter than the source's. ----
    let out_contig_dir = Path::new(out_store).join(chrom);
    max_del::write_max_del(&out_contig_dir, sample_orig_idx.len(), ploidy)?;

    // ---- distinct variant count: var_key entries are per-carrier-call, so
    // dedupe by (pos, key) before adding the dense side (already one row per
    // distinct variant). ----
    let distinct_vk_snp: HashSet<(u32, u8)> = vk_snp_pos
        .iter()
        .copied()
        .zip(vk_snp_codes.iter().copied())
        .collect();
    let distinct_vk_indel: HashSet<(u32, u32)> = vk_indel_pos
        .iter()
        .copied()
        .zip(vk_indel_keys.iter().copied())
        .collect();

    Ok(distinct_vk_snp.len()
        + distinct_vk_indel.len()
        + dense_snp_pos.len()
        + dense_indel_pos.len())
}

/// Region-overlap hit indices into `positions` (ascending, deduped): for each
/// region, narrow via `overlap_range` (the tree-based windowed search over
/// `query_regions[i]`'s widened bounds — identical to what `find_ranges` uses
/// for this same region/column/class), then keep the calls that pass BOTH the
/// `keeps` POS-precision filter (against the ORIGINAL, unwidened region bounds)
/// AND the per-element left-extent re-check `q_start < v_end`. Multiple regions
/// may re-discover the same call; `dedup` after sorting collapses that (mirrors
/// `Svar2Source`'s carrier-bit OR across overlapping regions).
///
/// The extent re-check is load-bearing and CANNOT be dropped: `overlap_range`
/// returns a contiguous SUPERSET `[s, e)` that trims only the first/last
/// overlaps, so interior indel rows whose extent ends at/before `q_start`
/// (v_ends is non-monotonic because deletion length varies) survive the window.
/// The real query path filters them element-wise — var_key via
/// `gather_vk`/`spine::gather_keys` (`q_start < v_end`, `gather.rs`/`spine.rs`),
/// dense via the presence loop (`dense.v_ends[j] > qs`, `gather.rs`). Omitting
/// it here would write indel calls `reroute=True` never emits — most visibly
/// under `OverlapMode::Variant`, whose `keeps` returns `true` unconditionally,
/// so this check is the ONLY thing excluding a `pos < q_start` non-overlapping
/// deletion pulled into the window by a long-deletion `max_del` bound. SNP
/// channels pass `v_end = pos + 1` (their window is already exact —
/// `max_region_length = 0` — so the check is a no-op but keeps one code path).
fn region_hits(
    positions: &[u32],
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
    overlap_range: impl Fn(u32, u32) -> Range<usize>,
    v_end_of: impl Fn(usize) -> u32,
) -> Vec<usize> {
    let mut hits: Vec<usize> = Vec::new();
    for (&(qs, qe), &(qsw, qew)) in regions.iter().zip(query_regions.iter()) {
        for i in overlap_range(qsw, qew) {
            if keeps(overlap, qs, qe, positions[i]) && qs < v_end_of(i) {
                hits.push(i);
            }
        }
    }
    hits.sort_unstable();
    hits.dedup();
    hits
}

/// Slice `var_key/snp`: per output column, the kept source calls' positions +
/// 2-bit ALT codes (unpacked; packed back to `alleles.bin`'s on-disk format by
/// the caller) + the output CSR `offsets` (len `n_subset*ploidy + 1`).
fn slice_var_key_snp(
    reader: &ContigReader,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
) -> (Vec<u32>, Vec<u8>, Vec<u64>) {
    let positions = reader.vk_snp.positions();
    let keys = as_bytes(&reader.vk_snp.keys);

    let mut out_positions: Vec<u32> = Vec::new();
    let mut out_codes: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = vec![0];

    for &s_orig in sample_orig_idx {
        for p in 0..ploidy {
            let col_src = s_orig * ploidy + p;
            let hits = region_hits(
                positions,
                regions,
                query_regions,
                overlap,
                |qsw, qew| reader.vk_snp_overlap(col_src, qsw, qew),
                // SNP v_end = pos + 1 (no deletion); the exact window makes
                // this a no-op, but keeps `region_hits` uniform.
                |i| positions[i] + 1,
            );
            for i in hits {
                out_positions.push(positions[i]);
                out_codes.push(unpack_snp_key_at(keys, i));
            }
            offsets.push(out_positions.len() as u64);
        }
    }
    (out_positions, out_codes, offsets)
}

/// Slice `var_key/indel`: same shape as [`slice_var_key_snp`], but keys are
/// the raw 32-bit indel keys (inline or LUT-lookup), copied verbatim — never
/// re-encoded, so a `Lookup` key's row index still resolves through the
/// (verbatim-copied) LUT unchanged.
fn slice_var_key_indel(
    reader: &ContigReader,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
) -> (Vec<u32>, Vec<u32>, Vec<u64>) {
    let positions = reader.vk_indel.positions();
    let keys = as_u32(&reader.vk_indel.keys);

    let mut out_positions: Vec<u32> = Vec::new();
    let mut out_keys: Vec<u32> = Vec::new();
    let mut offsets: Vec<u64> = vec![0];

    for &s_orig in sample_orig_idx {
        for p in 0..ploidy {
            let col_src = s_orig * ploidy + p;
            // The indel channel's tree search needs a per-(sample, ploid)
            // max_del bound — `sample` here must be the ORIGINAL column
            // (`vk_indel_max_del` is indexed by the source cohort, not the
            // subset), mirroring `find_ranges`'s `orig_s` usage.
            let hits = region_hits(
                positions,
                regions,
                query_regions,
                overlap,
                |qsw, qew| reader.vk_indel_overlap(col_src, s_orig, p, qsw, qew),
                // Indel v_end = pos + 1 + deletion_len(key) — the SAME formula
                // `gather_vk`/`spine::gather_keys` use for the left-extent
                // re-check.
                |i| positions[i] + 1 + deletion_len(keys[i]),
            );
            for i in hits {
                out_positions.push(positions[i]);
                out_keys.push(keys[i]);
            }
            offsets.push(out_positions.len() as u64);
        }
    }
    (out_positions, out_keys, offsets)
}

/// Slice one dense class table (`is_snp` selects the SNP vs. indel key
/// decoding). Returns `(positions, alleles.bin bytes, genotypes.bin bytes)`,
/// all sized to the OUTPUT (kept rows x subset haps) — or all-empty if the
/// source has no table of this class.
///
/// A row survives iff it BOTH region-overlaps (`region_hits`, over the SAME
/// windowed search `find_ranges` uses for the dense channel) AND is carried
/// by at least one subset haplotype (`carried_by_subset`, marked via
/// `for_each_carried` — O(source dense rows for this class), never O(cohort x
/// variants)). `row_out_of_src` doubles as that AND: a row only gets a `>= 0`
/// entry once it has passed both checks, so the final per-hap bit fill below
/// (which walks `for_each_carried` again, this time over just the subset's
/// OWN carried bits) needs no separate membership test.
#[allow(clippy::too_many_arguments)]
fn slice_dense(
    dense: Option<&DenseView>,
    is_snp: bool,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
    overlap_range: impl Fn(u32, u32) -> Range<usize>,
) -> (Vec<u32>, Vec<u8>, Vec<u8>) {
    let Some(d) = dense else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    let n_dense = d.n_dense_variants;
    let positions = d.positions();

    // `d.keys` is 2-bit-packed bytes for the SNP class but a raw `u32` LE
    // array for indel — `as_u32` on the (generally non-multiple-of-4-byte)
    // packed SNP buffer would trip bytemuck's size check, so each view is
    // only materialized for its own class.
    let keys_bytes: &[u8] = if is_snp { as_bytes(&d.keys) } else { &[] };
    let keys_u32: &[u32] = if is_snp { &[] } else { as_u32(&d.keys) };

    let hits = region_hits(
        positions,
        regions,
        query_regions,
        overlap,
        overlap_range,
        // Dense v_end: snp = pos + 1 (exact window, no-op); indel = pos + 1 +
        // deletion_len(key) — the SAME formula the dense presence loop uses
        // (`dense.v_ends[j] > qs`, `gather.rs`).
        |i| {
            if is_snp {
                positions[i] + 1
            } else {
                positions[i] + 1 + deletion_len(keys_u32[i])
            }
        },
    );

    let mut carried_by_subset = vec![false; n_dense];
    for &s_orig in sample_orig_idx {
        for p in 0..ploidy {
            let hap_src = s_orig * ploidy + p;
            d.for_each_carried(hap_src, |col| carried_by_subset[col] = true);
        }
    }

    let mut row_out_of_src: Vec<i64> = vec![-1; n_dense];
    let mut out_positions: Vec<u32> = Vec::new();
    // snp: one unpacked 2-bit code per kept row (packed by the caller); indel:
    // the raw 4 LE bytes of the key per kept row.
    let mut out_key_bytes: Vec<u8> = Vec::new();
    for &row in &hits {
        if !carried_by_subset[row] {
            continue;
        }
        row_out_of_src[row] = out_positions.len() as i64;
        out_positions.push(positions[row]);
        if is_snp {
            out_key_bytes.push(unpack_snp_key_at(keys_bytes, row));
        } else {
            out_key_bytes.extend_from_slice(&keys_u32[row].to_le_bytes());
        }
    }

    let n_kept = out_positions.len();
    let columns_out = sample_orig_idx.len() * ploidy;
    let mut bits = vec![0u8; (columns_out * n_kept).div_ceil(8)];
    for (c_out, &s_orig) in sample_orig_idx.iter().enumerate() {
        for p in 0..ploidy {
            let hap_src = s_orig * ploidy + p;
            let hap_out = c_out * ploidy + p;
            d.for_each_carried(hap_src, |col| {
                let row_out = row_out_of_src[col];
                if row_out >= 0 {
                    set_bit(&mut bits, hap_out * n_kept + row_out as usize);
                }
            });
        }
    }

    let alleles = if is_snp {
        pack_snp_keys(&out_key_bytes)
    } else {
        out_key_bytes
    };
    (out_positions, alleles, bits)
}

fn create_dir(path: &Path) -> Result<(), ConversionError> {
    fs::create_dir_all(path).map_err(|e| ConversionError::Io {
        context: format!("create_dir_all {}", path.display()),
        source: e,
    })
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), ConversionError> {
    fs::write(path, bytes).map_err(|e| ConversionError::Io {
        context: format!("writing {}", path.display()),
        source: e,
    })
}

fn write_offsets(path: &Path, offsets: &[u64]) -> Result<(), ConversionError> {
    write_npy(path, &Array1::from_vec(offsets.to_vec())).map_err(|source| ConversionError::Npy {
        path: path.display().to_string(),
        source,
    })
}

fn copy_file(src: &Path, dst: &Path) -> Result<(), ConversionError> {
    fs::copy(src, dst).map_err(|e| ConversionError::Io {
        context: format!("copying {} -> {}", src.display(), dst.display()),
        source: e,
    })?;
    Ok(())
}

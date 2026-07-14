//! `slice_contig` / `slice_contig_genos`: the `reroute=False` direct
//! array-slicer for one finished SVAR2 contig's sidecars.
//!
//! Unlike `svar2_source::Svar2Source` (which re-runs the ordinary conversion
//! pipeline over synthesized `RawRecord`s — `reroute=True`), this module reads
//! a finished store's mmap'd sidecars directly and writes a region/sample
//! subset as a new finished contig's sidecars: `var_key/{snp,indel}/*`,
//! `dense/{snp,indel}/*`, the shared indel long-allele LUT, `max_del`, and
//! (via `slice_contig`) each requested INFO/FORMAT field's `values.bin`.
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
//! Fields ride the SAME provenance the genotype gather already computes: the
//! genotype pass records, per kept var_key call, its SOURCE call index, and
//! per kept dense row, its SOURCE row index (both in output order). A field's
//! `values.bin` is laid out parallel to those (one element per var_key call;
//! one per dense INFO row; `n_dense × n_samples` for dense FORMAT), so the
//! field pass re-gathers each field at those same source indices — no second
//! query, still O(output).
//!
//! Mutcat and the top-level `meta.json` are out of scope here (later tasks in
//! the `write_view(reroute=False)` plan).

use std::collections::HashSet;
use std::fs;
use std::ops::Range;
use std::path::Path;

use ndarray::Array1;
use ndarray_npy::write_npy;

use crate::bits::set_bit;
use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec};
use crate::layout::{self, ContigPaths, FieldSub};
use crate::max_del;
use crate::query::ContigReader;
use crate::query::field::FieldView;
use crate::query::sidecar::{DenseView, as_bytes, as_u32};
use crate::rvk::{deletion_len, pack_snp_keys, unpack_snp_key_at};
use crate::svar2_source::{OverlapMode, keeps, query_window, read_n_samples};

/// SOURCE-index provenance from the genotype gather, consumed by the field
/// pass so a field's `values.bin` is re-gathered in the exact output order the
/// genotype sidecars were written in. Each `*_src` vec is parallel to the
/// corresponding output sidecar's rows/calls.
struct GenoProvenance {
    /// Source var_key/snp CALL index per output call (parallel to the output
    /// `var_key/snp` positions/alleles).
    vk_snp_src: Vec<usize>,
    /// Source var_key/indel CALL index per output call.
    vk_indel_src: Vec<usize>,
    /// Source dense/snp ROW index per output dense row (output row order).
    dense_snp_src: Vec<usize>,
    /// Source dense/indel ROW index per output dense row.
    dense_indel_src: Vec<usize>,
    /// Distinct variant count (the value both public entry points return).
    n_variants: usize,
}

/// Slice `{src_store}/{chrom}` down to `sample_orig_idx` (original sample column
/// indices, in *output* order) and `regions`, writing the genotype sidecars AND
/// each field in `fields` to `{out_store}/{chrom}`. Returns the number of
/// distinct variants written.
///
/// `fields` are the store's finalized INFO/FORMAT specs (concrete dtype, never
/// `Auto`). A field/sub whose source `values.bin` is missing or empty is a
/// legal empty sub-stream and is skipped (no output file written for it).
#[allow(clippy::too_many_arguments)]
pub fn slice_contig(
    src_store: &str,
    out_store: &str,
    chrom: &str,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    overlap: OverlapMode,
    fields: &[FieldSpec],
) -> Result<usize, ConversionError> {
    let (prov, n_samples_orig) = slice_genos_inner(
        src_store,
        out_store,
        chrom,
        sample_orig_idx,
        ploidy,
        regions,
        overlap,
    )?;

    if !fields.is_empty() {
        let src_paths = ContigPaths::new(src_store, chrom);
        let out_paths = ContigPaths::new(out_store, chrom);
        slice_fields(
            &src_paths,
            &out_paths,
            fields,
            &prov,
            sample_orig_idx,
            n_samples_orig,
        )?;
    }
    Ok(prov.n_variants)
}

/// Genotypes-only slice: [`slice_contig`] with no fields. Kept as a distinct
/// entry point for genotype-only callers and the byte-parity tests.
pub fn slice_contig_genos(
    src_store: &str,
    out_store: &str,
    chrom: &str,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    overlap: OverlapMode,
) -> Result<usize, ConversionError> {
    let (prov, _) = slice_genos_inner(
        src_store,
        out_store,
        chrom,
        sample_orig_idx,
        ploidy,
        regions,
        overlap,
    )?;
    Ok(prov.n_variants)
}

/// Write `{out_store}/{chrom}`'s genotype sidecars (var_key/dense/LUT/max_del)
/// as a region/sample subset of `{src_store}/{chrom}`, and return the SOURCE
/// provenance the field pass needs plus the source cohort size.
///
/// `overlap` and `regions` select calls with EXACTLY `Svar2Source`'s semantics
/// (`query_window` widens the search window per mode, `keeps` applies the final
/// POS-precision filter) — see `svar2_source::OverlapMode`.
fn slice_genos_inner(
    src_store: &str,
    out_store: &str,
    chrom: &str,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    overlap: OverlapMode,
) -> Result<(GenoProvenance, usize), ConversionError> {
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
    let (vk_snp_pos, vk_snp_codes, vk_snp_offsets, vk_snp_src) = slice_var_key_snp(
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

    let (vk_indel_pos, vk_indel_keys, vk_indel_offsets, vk_indel_src) = slice_var_key_indel(
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
    let (dense_snp_pos, dense_snp_alleles, dense_snp_bits, dense_snp_src) = slice_dense(
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

    let (dense_indel_pos, dense_indel_alleles, dense_indel_bits, dense_indel_src) = slice_dense(
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

    let n_variants = distinct_vk_snp.len()
        + distinct_vk_indel.len()
        + dense_snp_pos.len()
        + dense_indel_pos.len();

    Ok((
        GenoProvenance {
            vk_snp_src,
            vk_indel_src,
            dense_snp_src,
            dense_indel_src,
            n_variants,
        },
        n_samples_orig,
    ))
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
/// the caller) + the output CSR `offsets` (len `n_subset*ploidy + 1`) + the
/// SOURCE call index per output call (parallel to `positions`, for the field
/// gather).
fn slice_var_key_snp(
    reader: &ContigReader,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
) -> (Vec<u32>, Vec<u8>, Vec<u64>, Vec<usize>) {
    let positions = reader.vk_snp.positions();
    let keys = as_bytes(&reader.vk_snp.keys);

    let mut out_positions: Vec<u32> = Vec::new();
    let mut out_codes: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = vec![0];
    let mut src_calls: Vec<usize> = Vec::new();

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
                src_calls.push(i);
            }
            offsets.push(out_positions.len() as u64);
        }
    }
    (out_positions, out_codes, offsets, src_calls)
}

/// Slice `var_key/indel`: same shape as [`slice_var_key_snp`], but keys are
/// the raw 32-bit indel keys (inline or LUT-lookup), copied verbatim — never
/// re-encoded, so a `Lookup` key's row index still resolves through the
/// (verbatim-copied) LUT unchanged. Also returns the SOURCE call index per
/// output call.
fn slice_var_key_indel(
    reader: &ContigReader,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
) -> (Vec<u32>, Vec<u32>, Vec<u64>, Vec<usize>) {
    let positions = reader.vk_indel.positions();
    let keys = as_u32(&reader.vk_indel.keys);

    let mut out_positions: Vec<u32> = Vec::new();
    let mut out_keys: Vec<u32> = Vec::new();
    let mut offsets: Vec<u64> = vec![0];
    let mut src_calls: Vec<usize> = Vec::new();

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
                src_calls.push(i);
            }
            offsets.push(out_positions.len() as u64);
        }
    }
    (out_positions, out_keys, offsets, src_calls)
}

/// Slice one dense class table (`is_snp` selects the SNP vs. indel key
/// decoding). Returns `(positions, alleles.bin bytes, genotypes.bin bytes,
/// src_rows)` — all sized to the OUTPUT (kept rows x subset haps; `src_rows`
/// parallel to `positions`, giving each output row's SOURCE row index for the
/// field gather) — or all-empty if the source has no table of this class.
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
) -> (Vec<u32>, Vec<u8>, Vec<u8>, Vec<usize>) {
    let Some(d) = dense else {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
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
    let mut src_rows: Vec<usize> = Vec::new();
    for &row in &hits {
        if !carried_by_subset[row] {
            continue;
        }
        row_out_of_src[row] = out_positions.len() as i64;
        out_positions.push(positions[row]);
        src_rows.push(row);
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
    (out_positions, alleles, bits, src_rows)
}

/// Slice every field in `fields` across its four sub-streams, re-gathering each
/// field's `values.bin` at the SOURCE indices the genotype pass recorded so the
/// output values stay parallel to the output genotype sidecars. A field/sub
/// whose source `values.bin` is missing/empty is a legal empty sub-stream and
/// is skipped (no output file). O(output): each gather is one element per kept
/// call/row (dense FORMAT: per kept row × subset sample), never the source
/// cohort × variants.
fn slice_fields(
    src_paths: &ContigPaths,
    out_paths: &ContigPaths,
    fields: &[FieldSpec],
    prov: &GenoProvenance,
    sample_orig_idx: &[usize],
    n_samples_orig: usize,
) -> Result<(), ConversionError> {
    for spec in fields {
        let width = spec.dtype.width_bytes().ok_or_else(|| {
            ConversionError::Input(format!(
                "field {:?} has unresolved dtype {}; the store was never finalized",
                spec.name,
                spec.dtype.as_str()
            ))
        })?;

        // var_key subs: one element per kept CALL, in output order — identical
        // handling for INFO and FORMAT (var_key stores per-call).
        slice_field_var_key(
            src_paths,
            out_paths,
            spec,
            width,
            FieldSub::VkSnp,
            &prov.vk_snp_src,
            n_samples_orig,
        )?;
        slice_field_var_key(
            src_paths,
            out_paths,
            spec,
            width,
            FieldSub::VkIndel,
            &prov.vk_indel_src,
            n_samples_orig,
        )?;

        // dense subs: INFO is one element per kept ROW; FORMAT is re-strided to
        // the subset (`row_out*n_subset + s_out`).
        slice_field_dense(
            src_paths,
            out_paths,
            spec,
            width,
            FieldSub::DenseSnp,
            &prov.dense_snp_src,
            sample_orig_idx,
            n_samples_orig,
        )?;
        slice_field_dense(
            src_paths,
            out_paths,
            spec,
            width,
            FieldSub::DenseIndel,
            &prov.dense_indel_src,
            sample_orig_idx,
            n_samples_orig,
        )?;
    }
    Ok(())
}

/// Open a source field sub-stream. Returns `Ok(None)` for a legal empty/missing
/// sub-stream (nothing to slice), `Ok(Some(view))` otherwise.
fn open_source_field(
    src_paths: &ContigPaths,
    spec: &FieldSpec,
    sub: FieldSub,
    n_samples_orig: usize,
) -> Result<Option<FieldView>, ConversionError> {
    let view = FieldView::open(
        src_paths,
        spec.category.as_str(),
        &spec.name,
        sub,
        spec.dtype,
        n_samples_orig,
    )
    .map_err(|source| ConversionError::Io {
        context: format!(
            "opening field {}/{}/{}",
            spec.category.as_str(),
            spec.name,
            sub.dir_name()
        ),
        source,
    })?;
    if view.is_empty() {
        Ok(None)
    } else {
        Ok(Some(view))
    }
}

/// One var_key field sub-stream: gather `view.bytes_at(src_call)` for each kept
/// source call (output order), write the concatenation as the output
/// `values.bin`.
fn slice_field_var_key(
    src_paths: &ContigPaths,
    out_paths: &ContigPaths,
    spec: &FieldSpec,
    width: usize,
    sub: FieldSub,
    src_calls: &[usize],
    n_samples_orig: usize,
) -> Result<(), ConversionError> {
    let Some(view) = open_source_field(src_paths, spec, sub, n_samples_orig)? else {
        return Ok(());
    };
    let mut out = Vec::with_capacity(src_calls.len() * width);
    for &i in src_calls {
        out.extend_from_slice(view.bytes_at(i));
    }
    write_field_values(out_paths, spec, sub, &out)
}

/// One dense field sub-stream. INFO: one element per kept row. FORMAT: for each
/// kept row, for each subset sample `s_out` (original column
/// `sample_orig_idx[s_out]`), copy source element `row*n_samples_orig + orig`
/// into new position `row_out*n_subset + s_out` — the re-stride `FieldView`'s
/// `format_at` reads back on the query side.
#[allow(clippy::too_many_arguments)]
fn slice_field_dense(
    src_paths: &ContigPaths,
    out_paths: &ContigPaths,
    spec: &FieldSpec,
    width: usize,
    sub: FieldSub,
    src_rows: &[usize],
    sample_orig_idx: &[usize],
    n_samples_orig: usize,
) -> Result<(), ConversionError> {
    let Some(view) = open_source_field(src_paths, spec, sub, n_samples_orig)? else {
        return Ok(());
    };
    let mut out: Vec<u8> = Vec::new();
    match spec.category {
        FieldCategory::Info => {
            out.reserve(src_rows.len() * width);
            for &row in src_rows {
                out.extend_from_slice(view.bytes_at(row));
            }
        }
        FieldCategory::Format => {
            out.reserve(src_rows.len() * sample_orig_idx.len() * width);
            for &row in src_rows {
                for &orig in sample_orig_idx {
                    out.extend_from_slice(view.bytes_at(row * n_samples_orig + orig));
                }
            }
        }
    }
    write_field_values(out_paths, spec, sub, &out)
}

/// Create the field's output sub-stream dir and write its `values.bin`.
fn write_field_values(
    out_paths: &ContigPaths,
    spec: &FieldSpec,
    sub: FieldSub,
    bytes: &[u8],
) -> Result<(), ConversionError> {
    let path = out_paths.field_values(spec.category.as_str(), &spec.name, sub);
    if let Some(parent) = path.parent() {
        create_dir(parent)?;
    }
    write_bytes(&path, bytes)
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

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
//! Reuses `svar2_view::{query_window, keeps}` verbatim so `reroute=False`
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

use std::collections::{HashMap, HashSet};
use std::fs;
use std::ops::Range;
use std::path::Path;

use ndarray::Array1;
use ndarray_npy::write_npy;

use crate::bits::set_bit;
use crate::cost_model::{
    Class, Representation, SIDECAR_BITS_INDEL, SIDECAR_BITS_SNP, choose_representation,
};
use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec};
use crate::layout::{self, ContigPaths, FieldSub};
use crate::max_del;
use crate::query::ContigReader;
use crate::query::field::FieldView;
use crate::query::sidecar::{DenseView, as_bytes, as_u32};
use crate::rvk::{deletion_len, pack_snp_keys, unpack_snp_key_at};
use crate::svar2_view::{OverlapMode, keeps, query_window, read_n_samples};
use svar2_codec::{PAYLOAD_TOP_SHIFT, snp_code_to_key};

/// Which on-disk stream each variant lands in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Routing {
    /// `reroute=False`: a variant's output stream is its source stream.
    Preserve,
    /// `reroute=True`: re-run the cost model against the SUBSET's carrier count.
    Recompute,
}

/// Where one OUTPUT var_key call's genotype + field bytes come from.
#[derive(Clone, Copy)]
enum CallSrc {
    /// Unflipped: a source var_key call index.
    VarKey { call: usize },
    /// Flipped dense -> var_key: source dense row + ORIGINAL sample column.
    /// Constructed by `route_class` under `Routing::Recompute` (Task 3); read by
    /// the field pass (`slice_field_var_key`) to re-gather the field value off
    /// the source dense sub-stream.
    Dense { row: usize, s_orig: usize },
}

/// Where one OUTPUT dense row's genotype + field bytes come from.
enum RowSrc {
    /// Unflipped: a source dense row index.
    Dense { row: usize },
    /// Flipped var_key -> dense. `per_sample_call[s_out]` is a representative
    /// source var_key call for that output sample (any carrier call — see the
    /// key invariant), or `None` for a non-carrier (=> field sentinel).
    /// Constructed by `route_class` under `Routing::Recompute` (Task 3); read by
    /// the field pass (`slice_field_dense`) to re-gather each output sample's
    /// value (or the field sentinel for a non-carrier) off the source var_key
    /// sub-stream.
    VarKey {
        per_sample_call: Vec<Option<usize>>,
        /// Any source call of this variant — INFO is per-variant, so any will do.
        info_call: usize,
    },
}

/// One gathered var_key call, before routing.
struct GatheredCall {
    src: usize,     // source call index
    col_out: usize, // OUTPUT hap column (s_out*ploidy + p)
    pos: u32,
    key: u32, // SNP: snp_code_to_key(code); indel: the raw key
}

/// One gathered dense row, before routing.
struct GatheredRow {
    src: usize, // source row index
    pos: u32,
    key: u32,
    carriers_out: Vec<usize>, // OUTPUT hap columns carrying it
}

/// Recover the packed 2-bit SNP ALT code from a uniform-key-space `u32`
/// produced by [`snp_code_to_key`] — the tiny local inverse `emit_var_key` /
/// `emit_dense` need to write the SNP `alleles.bin` back out. Round-trips for
/// every valid code (`snp_code_to_key(c) -> decode_snp_2bit_code -> c`, see
/// the unit test below).
#[inline]
fn decode_snp_2bit_code(key: u32) -> u8 {
    ((key >> PAYLOAD_TOP_SHIFT) & 3) as u8
}

/// Route each class's gathered calls/rows to their OUTPUT stream.
///
/// Under `Routing::Preserve` every variant keeps its source stream — the
/// identity mapping that reproduces today's exact bytes.
///
/// Under `Routing::Recompute` (`reroute=True`) the cost model is re-run against
/// the SUBSET's own carrier count (`route_class`), flipping variants between
/// streams when the subset makes the other representation cheaper. The three
/// cost terms mirror the production converter (`src/rvk.rs:230-270`):
/// `sidecar_bits_enabled` resolves to the per-class mutational-signature cost
/// (`SIDECAR_BITS_SNP`/`SIDECAR_BITS_INDEL`, or 0 when disabled); `info_bits` /
/// `format_bits` are the summed per-record field widths. SNP and indel classes
/// are routed independently — they never exchange variants.
#[allow(clippy::too_many_arguments)]
fn route(
    routing: Routing,
    vk_snp: Vec<GatheredCall>,
    vk_indel: Vec<GatheredCall>,
    d_snp: Vec<GatheredRow>,
    d_indel: Vec<GatheredRow>,
    sample_orig_idx: &[usize],
    n_subset: usize,
    ploidy: usize,
    sidecar_bits_enabled: bool,
    info_bits: u64,
    format_bits: u64,
) -> RoutePlan {
    match routing {
        Routing::Preserve => RoutePlan {
            vk_snp: preserve_calls(vk_snp),
            vk_indel: preserve_calls(vk_indel),
            d_snp: preserve_rows(d_snp),
            d_indel: preserve_rows(d_indel),
        },
        Routing::Recompute => {
            let (snp_sidecar, indel_sidecar) = if sidecar_bits_enabled {
                (SIDECAR_BITS_SNP, SIDECAR_BITS_INDEL)
            } else {
                (0, 0)
            };
            let mut plan = RoutePlan::default();
            route_class(
                &mut plan,
                Class::Snp,
                vk_snp,
                d_snp,
                sample_orig_idx,
                n_subset,
                ploidy,
                snp_sidecar,
                info_bits,
                format_bits,
            );
            route_class(
                &mut plan,
                Class::Indel,
                vk_indel,
                d_indel,
                sample_orig_idx,
                n_subset,
                ploidy,
                indel_sidecar,
                info_bits,
                format_bits,
            );
            plan.sort();
            plan
        }
    }
}

/// Route one class (SNP or indel) under `Routing::Recompute`. The two classes
/// never exchange variants, so each is routed with its own call/row streams and
/// its own per-class `sidecar_bits`.
///
/// var_key variants are grouped by `(pos, key)` — one group is one variant's
/// carrier calls WITHIN the subset, so `group.len()` is the subset carrier
/// count `x_sub`. Dense rows already carry their subset popcount in
/// `carriers_out`. Each variant's `x_sub` is fed to `choose_representation`; if
/// the cheaper representation differs from the source stream the variant flips.
///
/// `x_sub` is always `> 0`: a var_key variant with no subset carriers has no
/// gathered calls (never enters `by_variant`), and a dense row with no subset
/// carriers was already dropped by `gather_dense`'s `carried_by_subset` filter.
/// That preserves the existing MAC=0 drop under both routings.
#[allow(clippy::too_many_arguments)]
fn route_class(
    plan: &mut RoutePlan,
    class: Class,
    calls: Vec<GatheredCall>,
    rows: Vec<GatheredRow>,
    sample_orig_idx: &[usize],
    n_subset: usize,
    ploidy: usize,
    sidecar_bits: u64,
    info_bits: u64,
    format_bits: u64,
) {
    // 1. Group var_key calls by variant: one gathered call == one carrier hap.
    let mut by_variant: HashMap<(u32, u32), Vec<GatheredCall>> = HashMap::new();
    for c in calls {
        by_variant.entry((c.pos, c.key)).or_default().push(c);
    }

    // 2. var_key variants: stay var_key, or flip to dense.
    for ((pos, key), group) in by_variant {
        let x_sub = group.len();
        debug_assert!(x_sub > 0, "a grouped var_key variant has >= 1 carrier call");
        match choose_representation(
            class,
            n_subset,
            ploidy,
            x_sub,
            sidecar_bits,
            info_bits,
            format_bits,
        ) {
            Representation::VarKey => {
                for c in group {
                    plan.push_call(class, CallSrc::VarKey { call: c.src }, c.col_out, pos, key);
                }
            }
            Representation::Dense => {
                // Carriers by OUTPUT hap column; plus a representative source
                // call per OUTPUT SAMPLE for the (Task 4) field pass — all
                // carrier calls of a sample hold identical field bytes.
                let mut carriers_out: Vec<usize> = group.iter().map(|c| c.col_out).collect();
                carriers_out.sort_unstable();
                let mut per_sample_call = vec![None; n_subset];
                for c in &group {
                    per_sample_call[c.col_out / ploidy].get_or_insert(c.src);
                }
                let info_call = group[0].src;
                plan.push_row(
                    class,
                    RowSrc::VarKey {
                        per_sample_call,
                        info_call,
                    },
                    pos,
                    key,
                    carriers_out,
                );
            }
        }
    }

    // 3. dense rows: stay dense, or flip to var_key.
    for r in rows {
        let x_sub = r.carriers_out.len(); // subset popcount of the row
        debug_assert!(x_sub > 0, "a gathered dense row has >= 1 subset carrier");
        match choose_representation(
            class,
            n_subset,
            ploidy,
            x_sub,
            sidecar_bits,
            info_bits,
            format_bits,
        ) {
            Representation::Dense => {
                plan.push_row(
                    class,
                    RowSrc::Dense { row: r.src },
                    r.pos,
                    r.key,
                    r.carriers_out,
                );
            }
            Representation::VarKey => {
                for &col_out in &r.carriers_out {
                    // The ORIGINAL sample column of this output hap, for the
                    // (Task 4) field re-gather off the dense source row.
                    let s_orig = sample_orig_idx[col_out / ploidy];
                    plan.push_call(
                        class,
                        CallSrc::Dense { row: r.src, s_orig },
                        col_out,
                        r.pos,
                        r.key,
                    );
                }
            }
        }
    }
}

/// The routed, emit-ready plan for all four sidecar classes: `(src, col_out,
/// pos, key)` for var_key (must be `(col_out, pos, key)`-ordered for
/// `emit_var_key`) and `(src, pos, key, carriers_out)` for dense (must be
/// `(pos, key)`-ordered for `emit_dense`). Under `Routing::Preserve` the gather
/// order already satisfies both; under `Routing::Recompute` flipped variants
/// arrive out of order, so `sort` restores the required order.
#[derive(Default)]
struct RoutePlan {
    vk_snp: Vec<(CallSrc, usize, u32, u32)>,
    vk_indel: Vec<(CallSrc, usize, u32, u32)>,
    d_snp: Vec<(RowSrc, u32, u32, Vec<usize>)>,
    d_indel: Vec<(RowSrc, u32, u32, Vec<usize>)>,
}

impl RoutePlan {
    /// Append one routed var_key call to its class stream.
    fn push_call(&mut self, class: Class, src: CallSrc, col_out: usize, pos: u32, key: u32) {
        match class {
            Class::Snp => self.vk_snp.push((src, col_out, pos, key)),
            Class::Indel => self.vk_indel.push((src, col_out, pos, key)),
        }
    }

    /// Append one routed dense row to its class stream.
    fn push_row(
        &mut self,
        class: Class,
        src: RowSrc,
        pos: u32,
        key: u32,
        carriers_out: Vec<usize>,
    ) {
        match class {
            Class::Snp => self.d_snp.push((src, pos, key, carriers_out)),
            Class::Indel => self.d_indel.push((src, pos, key, carriers_out)),
        }
    }

    /// Restore the per-stream order `emit_var_key`/`emit_dense` require after a
    /// `Routing::Recompute` flip inserted variants out of order: var_key by
    /// `(col_out, pos, key)`, dense by `(pos, key)`.
    ///
    /// `sort_by_key` is stable, so entries sharing a full sort key keep their
    /// insertion order. Same-position ties across DIFFERENT keys are ordered by
    /// `key` here — the one place the byte layout can differ from a fresh
    /// `from_vcf` conversion, which orders a position's variants by
    /// `(pos, ilen, alt)` rather than by this uniform-key-space `key`. That is
    /// precisely why `reroute=True` is verified by decode-equivalence +
    /// routing-equality + output size (Tasks 3, 8), NOT by byte-parity against
    /// `from_vcf`. This is not new in kind: the shipped `reroute=True` path also
    /// re-orders (its `Svar2Source` emits from a `BTreeMap<(pos, ilen, alt), _>`).
    fn sort(&mut self) {
        for s in [&mut self.vk_snp, &mut self.vk_indel] {
            s.sort_by_key(|&(_, col, pos, key)| (col, pos, key));
        }
        for s in [&mut self.d_snp, &mut self.d_indel] {
            s.sort_by_key(|(_, pos, key, _)| (*pos, *key));
        }
    }
}

fn preserve_calls(gathered: Vec<GatheredCall>) -> Vec<(CallSrc, usize, u32, u32)> {
    gathered
        .into_iter()
        .map(|g| (CallSrc::VarKey { call: g.src }, g.col_out, g.pos, g.key))
        .collect()
}

fn preserve_rows(gathered: Vec<GatheredRow>) -> Vec<(RowSrc, u32, u32, Vec<usize>)> {
    gathered
        .into_iter()
        .map(|g| (RowSrc::Dense { row: g.src }, g.pos, g.key, g.carriers_out))
        .collect()
}

/// SOURCE provenance from the genotype gather, consumed by the field pass so a
/// field's `values.bin` is re-gathered in the exact output order the genotype
/// sidecars were written in. Each `*_src` vec is parallel to the corresponding
/// output sidecar's rows/calls.
struct GenoProvenance {
    /// Per-output-call provenance for `var_key/snp` (parallel to the output
    /// `var_key/snp` positions/alleles).
    vk_snp_src: Vec<CallSrc>,
    /// Per-output-call provenance for `var_key/indel`.
    vk_indel_src: Vec<CallSrc>,
    /// Per-output-row provenance for `dense/snp` (output row order).
    dense_snp_src: Vec<RowSrc>,
    /// Per-output-row provenance for `dense/indel`.
    dense_indel_src: Vec<RowSrc>,
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
///
/// `routing` selects the output stream policy: `Routing::Preserve`
/// (`reroute=False`) keeps every variant on its source stream;
/// `Routing::Recompute` (`reroute=True`) re-runs the cost model against the
/// subset's own carrier count, so it needs the same cost terms the production
/// converter uses: `sidecar_bits_enabled` (mutational-signature sidecar on, i.e.
/// a reference was given), and `info_bits`/`format_bits` (the summed per-record
/// storage widths of the INFO/FORMAT specs in `fields`, in bits). The caller
/// computes them exactly as `src/rvk.rs:230-238` does. Under `Routing::Preserve`
/// these three are ignored (the identity route is independent of cost).
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
    routing: Routing,
    sidecar_bits_enabled: bool,
    info_bits: u64,
    format_bits: u64,
) -> Result<usize, ConversionError> {
    let (prov, n_samples_orig) = slice_genos_inner(
        src_store,
        out_store,
        chrom,
        sample_orig_idx,
        ploidy,
        regions,
        overlap,
        routing,
        sidecar_bits_enabled,
        info_bits,
        format_bits,
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
///
/// Genotype-only implies no fields (`info_bits = format_bits = 0`) and no
/// reference-derived sidecar (`sidecar_bits_enabled = false`) — those are the
/// only cost terms `Routing::Recompute` needs, so they are fixed here rather
/// than threaded (a genotype-only slice cannot carry field/sidecar costs).
#[allow(clippy::too_many_arguments)]
pub fn slice_contig_genos(
    src_store: &str,
    out_store: &str,
    chrom: &str,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    overlap: OverlapMode,
    routing: Routing,
) -> Result<usize, ConversionError> {
    let (prov, _) = slice_genos_inner(
        src_store,
        out_store,
        chrom,
        sample_orig_idx,
        ploidy,
        regions,
        overlap,
        routing,
        false, // sidecar_bits_enabled: genotype-only => no reference sidecar
        0,     // info_bits: no fields
        0,     // format_bits: no fields
    )?;
    Ok(prov.n_variants)
}

/// Write `{out_store}/{chrom}`'s genotype sidecars (var_key/dense/LUT/max_del)
/// as a region/sample subset of `{src_store}/{chrom}`, and return the SOURCE
/// provenance the field pass needs plus the source cohort size.
///
/// `overlap` and `regions` select calls with EXACTLY `Svar2Source`'s semantics
/// (`query_window` widens the search window per mode, `keeps` applies the final
/// POS-precision filter) — see `svar2_view::OverlapMode`.
///
/// Three phases: **gather** each class's kept calls/rows with source
/// provenance (`gather_var_key`/`gather_dense`), **route** them to an output
/// stream per `routing` (`route`), then **emit** the routed plan as final byte
/// buffers (`emit_var_key`/`emit_dense`). Splitting gather from emit is what
/// lets a variant change stream — `Routing::Preserve` is the identity route,
/// so this phase split is a no-op on the bytes written.
#[allow(clippy::too_many_arguments)]
fn slice_genos_inner(
    src_store: &str,
    out_store: &str,
    chrom: &str,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    overlap: OverlapMode,
    routing: Routing,
    sidecar_bits_enabled: bool,
    info_bits: u64,
    format_bits: u64,
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
    let n_cols_out = sample_orig_idx.len() * ploidy;

    // ---- gather ----
    let vk_snp_positions = reader.vk_snp.positions();
    let vk_snp_keys = as_bytes(&reader.vk_snp.keys);
    let vk_snp_g = gather_var_key(
        vk_snp_positions,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
        |col_src, _s_orig, _p, qsw, qew| reader.vk_snp_overlap(col_src, qsw, qew),
        |i| snp_code_to_key(unpack_snp_key_at(vk_snp_keys, i)),
        |i| vk_snp_positions[i] + 1,
    );

    let vk_indel_positions = reader.vk_indel.positions();
    let vk_indel_keys = as_u32(&reader.vk_indel.keys);
    let vk_indel_g = gather_var_key(
        vk_indel_positions,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
        // The indel channel's tree search needs a per-(sample, ploid) max_del
        // bound — `sample` here must be the ORIGINAL column
        // (`vk_indel_max_del` is indexed by the source cohort, not the
        // subset), mirroring `find_ranges`'s `orig_s` usage.
        |col_src, s_orig, p, qsw, qew| reader.vk_indel_overlap(col_src, s_orig, p, qsw, qew),
        |i| vk_indel_keys[i],
        |i| vk_indel_positions[i] + 1 + deletion_len(vk_indel_keys[i]),
    );

    let d_snp_g = gather_dense(
        reader.dense_snp.as_ref(),
        true,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
        |qsw, qew| reader.dense_snp_overlap(qsw, qew),
    );

    let d_indel_g = gather_dense(
        reader.dense_indel.as_ref(),
        false,
        sample_orig_idx,
        ploidy,
        regions,
        &query_regions,
        overlap,
        |qsw, qew| reader.dense_indel_overlap(qsw, qew),
    );

    // ---- route ----
    let plan = route(
        routing,
        vk_snp_g,
        vk_indel_g,
        d_snp_g,
        d_indel_g,
        sample_orig_idx,
        sample_orig_idx.len(),
        ploidy,
        sidecar_bits_enabled,
        info_bits,
        format_bits,
    );

    // ---- distinct variant count: var_key entries are per-carrier-call, so
    // dedupe by (pos, key) before adding the dense side (already one row per
    // distinct variant). Computed from the routed plan, before `emit_*`
    // consumes the dense halves by value. ----
    let distinct_vk_snp: HashSet<(u32, u8)> = plan
        .vk_snp
        .iter()
        .map(|&(_, _, pos, key)| (pos, decode_snp_2bit_code(key)))
        .collect();
    let distinct_vk_indel: HashSet<(u32, u32)> = plan
        .vk_indel
        .iter()
        .map(|&(_, _, pos, key)| (pos, key))
        .collect();
    let n_dense_snp = plan.d_snp.len();
    let n_dense_indel = plan.d_indel.len();
    let n_variants = distinct_vk_snp.len() + distinct_vk_indel.len() + n_dense_snp + n_dense_indel;

    // ---- emit ----
    let vk_snp_src = emit_var_key(&out_paths.var_key_snp_dir(), &plan.vk_snp, true, n_cols_out)?;
    let vk_indel_src = emit_var_key(
        &out_paths.var_key_indel_dir(),
        &plan.vk_indel,
        false,
        n_cols_out,
    )?;
    let dense_snp_src = emit_dense(&out_paths.dense_snp_dir(), plan.d_snp, true, n_cols_out)?;
    let dense_indel_src = emit_dense(
        &out_paths.dense_indel_dir(),
        plan.d_indel,
        false,
        n_cols_out,
    )?;

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

/// Gather the kept var_key calls of one class (SNP or indel). `key_of(i)`
/// reads the source call's key into the uniform 32-bit key space (a SNP code
/// re-expanded via `snp_code_to_key`, or an indel key verbatim); `v_end_of(i)`
/// its right extent. Column order is the OUTPUT order (`sample_orig_idx` x
/// ploidy) so the CSR emit stays a simple scan. `overlap_range` takes `(col_src,
/// s_orig, p, q_start_widened, q_end_widened)` — the SNP caller ignores
/// `s_orig`/`p` (its window doesn't depend on them), the indel caller needs
/// both for the per-(sample, ploid) `max_del` bound.
#[allow(clippy::too_many_arguments)]
fn gather_var_key(
    positions: &[u32],
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
    overlap_range: impl Fn(usize, usize, usize, u32, u32) -> Range<usize>,
    key_of: impl Fn(usize) -> u32,
    v_end_of: impl Fn(usize) -> u32,
) -> Vec<GatheredCall> {
    let mut out = Vec::new();
    for (s_out, &s_orig) in sample_orig_idx.iter().enumerate() {
        for p in 0..ploidy {
            let col_src = s_orig * ploidy + p;
            let col_out = s_out * ploidy + p;
            let hits = region_hits(
                positions,
                regions,
                query_regions,
                overlap,
                |qsw, qew| overlap_range(col_src, s_orig, p, qsw, qew),
                &v_end_of,
            );
            for i in hits {
                out.push(GatheredCall {
                    src: i,
                    col_out,
                    pos: positions[i],
                    key: key_of(i),
                });
            }
        }
    }
    out
}

/// Gather the kept dense rows of one class (`is_snp` selects the SNP vs. indel
/// key decoding), each with its OUTPUT hap-column carriers — or empty if the
/// source has no table of this class.
///
/// A row survives iff it BOTH region-overlaps (`region_hits`, over the SAME
/// windowed search `find_ranges` uses for the dense channel) AND is carried
/// by at least one subset haplotype (`carried_by_subset`, marked via
/// `for_each_carried` — O(source dense rows for this class), never O(cohort x
/// variants)). `row_of_src` doubles as that AND: a row only gets a `>= 0`
/// entry once it has passed both checks, so the final per-hap carrier fill
/// below (which walks `for_each_carried` again, this time over just the
/// subset's OWN carried bits) needs no separate membership test.
#[allow(clippy::too_many_arguments)]
fn gather_dense(
    dense: Option<&DenseView>,
    is_snp: bool,
    sample_orig_idx: &[usize],
    ploidy: usize,
    regions: &[(u32, u32)],
    query_regions: &[(u32, u32)],
    overlap: OverlapMode,
    overlap_range: impl Fn(u32, u32) -> Range<usize>,
) -> Vec<GatheredRow> {
    let Some(d) = dense else {
        return Vec::new();
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

    let mut rows: Vec<GatheredRow> = Vec::new();
    let mut row_of_src: Vec<i64> = vec![-1; n_dense];
    for &row in &hits {
        if !carried_by_subset[row] {
            continue;
        }
        row_of_src[row] = rows.len() as i64;
        let key = if is_snp {
            snp_code_to_key(unpack_snp_key_at(keys_bytes, row))
        } else {
            keys_u32[row]
        };
        rows.push(GatheredRow {
            src: row,
            pos: positions[row],
            key,
            carriers_out: Vec::new(),
        });
    }
    for (s_out, &s_orig) in sample_orig_idx.iter().enumerate() {
        for p in 0..ploidy {
            let hap_src = s_orig * ploidy + p;
            let hap_out = s_out * ploidy + p;
            d.for_each_carried(hap_src, |col| {
                let r = row_of_src[col];
                if r >= 0 {
                    rows[r as usize].carriers_out.push(hap_out);
                }
            });
        }
    }
    rows
}

/// Emit one var_key class. `calls` must already be sorted by (col_out, pos, key)
/// — under `Routing::Preserve`, `gather_var_key`'s column-major gather order
/// already guarantees this. Returns the per-output-call provenance, parallel
/// to the written sidecars.
fn emit_var_key(
    dir: &Path,
    calls: &[(CallSrc, usize, u32, u32)],
    is_snp: bool,
    n_cols_out: usize,
) -> Result<Vec<CallSrc>, ConversionError> {
    let mut positions = Vec::with_capacity(calls.len());
    let mut codes: Vec<u8> = Vec::new();
    let mut keys: Vec<u32> = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(n_cols_out + 1);
    let mut prov = Vec::with_capacity(calls.len());

    offsets.push(0);
    let mut c = 0usize;
    for col in 0..n_cols_out {
        while c < calls.len() && calls[c].1 == col {
            let (src, _, pos, key) = calls[c];
            positions.push(pos);
            if is_snp {
                codes.push(decode_snp_2bit_code(key));
            } else {
                keys.push(key);
            }
            prov.push(src);
            c += 1;
        }
        offsets.push(positions.len() as u64);
    }

    create_dir(dir)?;
    write_bytes(&layout::positions(dir), bytemuck::cast_slice(&positions))?;
    if is_snp {
        write_bytes(&layout::alleles(dir), &pack_snp_keys(&codes))?;
    } else {
        write_bytes(&layout::alleles(dir), bytemuck::cast_slice(&keys))?;
    }
    write_offsets(&layout::offsets(dir), &offsets)?;
    Ok(prov)
}

/// Emit one dense class. `rows` must already be sorted by (pos, key) — under
/// `Routing::Preserve`, `gather_dense`'s ascending-hit-index gather order
/// already guarantees this (dense rows are stored position-ascending).
fn emit_dense(
    dir: &Path,
    rows: Vec<(RowSrc, u32, u32, Vec<usize>)>,
    is_snp: bool,
    n_cols_out: usize,
) -> Result<Vec<RowSrc>, ConversionError> {
    let n_kept = rows.len();
    let mut positions = Vec::with_capacity(n_kept);
    let mut key_bytes: Vec<u8> = Vec::new();
    let mut bits = vec![0u8; (n_cols_out * n_kept).div_ceil(8)];
    let mut prov = Vec::with_capacity(n_kept);

    for (row_out, (src, pos, key, carriers)) in rows.into_iter().enumerate() {
        positions.push(pos);
        if is_snp {
            key_bytes.push(decode_snp_2bit_code(key));
        } else {
            key_bytes.extend_from_slice(&key.to_le_bytes());
        }
        for hap_out in carriers {
            set_bit(&mut bits, hap_out * n_kept + row_out);
        }
        prov.push(src);
    }

    create_dir(dir)?;
    write_bytes(&layout::positions(dir), bytemuck::cast_slice(&positions))?;
    let alleles = if is_snp {
        pack_snp_keys(&key_bytes)
    } else {
        key_bytes
    };
    write_bytes(&layout::alleles(dir), &alleles)?;
    write_bytes(&layout::genotypes(dir), &bits)?;
    Ok(prov)
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

/// One var_key field sub-stream: gather each kept output call's value (output
/// order) and write the concatenation as the output `values.bin`.
///
/// An unflipped call (`CallSrc::VarKey`) reads the source's var_key sub of the
/// same class. A call flipped from dense (`CallSrc::Dense`, `Routing::Recompute`)
/// reads the source's DENSE sub of the same class instead — so both sub-streams
/// may need to be open for one output stream. INFO is one element per source
/// dense row; FORMAT reads the flipped sample's own value at
/// `row * n_samples_orig + s_orig` (the source dense FORMAT `(row, sample)`
/// layout). Per the key invariant, any carrier call of a sample holds that
/// sample's value, so a dense -> var_key flip is lossless for the carrier it
/// keeps.
fn slice_field_var_key(
    src_paths: &ContigPaths,
    out_paths: &ContigPaths,
    spec: &FieldSpec,
    width: usize,
    sub: FieldSub,
    src_calls: &[CallSrc],
    n_samples_orig: usize,
) -> Result<(), ConversionError> {
    let vk_view = open_source_field(src_paths, spec, sub, n_samples_orig)?;
    // A flipped call reads from the DENSE sub of the same class.
    let dense_sub = match sub {
        FieldSub::VkSnp => FieldSub::DenseSnp,
        FieldSub::VkIndel => FieldSub::DenseIndel,
        _ => unreachable!("slice_field_var_key takes only var_key subs"),
    };
    let d_view = open_source_field(src_paths, spec, dense_sub, n_samples_orig)?;
    if vk_view.is_none() && d_view.is_none() {
        return Ok(());
    }

    let mut out = Vec::with_capacity(src_calls.len() * width);
    for src in src_calls {
        match *src {
            CallSrc::VarKey { call } => {
                let v = vk_view
                    .as_ref()
                    .expect("unflipped call needs the source var_key sub");
                out.extend_from_slice(v.bytes_at(call));
            }
            CallSrc::Dense { row, s_orig } => {
                let v = d_view
                    .as_ref()
                    .expect("flipped call needs the source dense sub");
                match spec.category {
                    // INFO is one element per dense ROW.
                    FieldCategory::Info => out.extend_from_slice(v.bytes_at(row)),
                    // FORMAT is (row, sample) -> the flipped sample's own value.
                    FieldCategory::Format => {
                        out.extend_from_slice(v.bytes_at(row * n_samples_orig + s_orig))
                    }
                }
            }
        }
    }
    write_field_values(out_paths, spec, sub, &out)
}

/// One dense field sub-stream. INFO: one element per kept row. FORMAT: for each
/// kept row, a full n_subset-wide column laid out `(row_out, s_out)` (the
/// re-stride `FieldView::format_at` reads back on the query side).
///
/// An unflipped row (`RowSrc::Dense`) reads the source's dense sub of the same
/// class: INFO at `row`, FORMAT at `row*n_samples_orig + orig` for each subset
/// sample's ORIGINAL column. A row flipped from var_key (`RowSrc::VarKey`,
/// `Routing::Recompute`) reads the source's VAR_KEY sub instead — so both
/// sub-streams may need to be open for one output stream. INFO reads any call
/// of the variant (`info_call` — INFO is per-variant). FORMAT emits the full
/// subset column: a carrier sample reads its representative source call's value,
/// a NON-carrier gets the field's missing sentinel — byte-identical to what
/// `rvk.rs`'s dense push writes for a genuine dense non-carrier (via the shared
/// `field_finalize` encoder in `FieldSpec::encode_scalar`).
#[allow(clippy::too_many_arguments)]
fn slice_field_dense(
    src_paths: &ContigPaths,
    out_paths: &ContigPaths,
    spec: &FieldSpec,
    width: usize,
    sub: FieldSub,
    src_rows: &[RowSrc],
    sample_orig_idx: &[usize],
    n_samples_orig: usize,
) -> Result<(), ConversionError> {
    let d_view = open_source_field(src_paths, spec, sub, n_samples_orig)?;
    // A flipped row reads from the VAR_KEY sub of the same class.
    let vk_sub = match sub {
        FieldSub::DenseSnp => FieldSub::VkSnp,
        FieldSub::DenseIndel => FieldSub::VkIndel,
        _ => unreachable!("slice_field_dense takes only dense subs"),
    };
    let vk_view = open_source_field(src_paths, spec, vk_sub, n_samples_orig)?;
    if d_view.is_none() && vk_view.is_none() {
        return Ok(());
    }

    // The field's missing value, encoded at its on-disk dtype — the same
    // sentinel `rvk.rs`'s dense push writes for a dense non-carrier.
    let sentinel = spec.encode_scalar(spec.missing_sentinel());

    let n_out = match spec.category {
        FieldCategory::Info => src_rows.len(),
        FieldCategory::Format => src_rows.len() * sample_orig_idx.len(),
    };
    let mut out: Vec<u8> = Vec::with_capacity(n_out * width);
    for src in src_rows {
        match src {
            RowSrc::Dense { row } => {
                let v = d_view
                    .as_ref()
                    .expect("unflipped row needs the source dense sub");
                match spec.category {
                    FieldCategory::Info => out.extend_from_slice(v.bytes_at(*row)),
                    FieldCategory::Format => {
                        for &orig in sample_orig_idx {
                            out.extend_from_slice(v.bytes_at(row * n_samples_orig + orig));
                        }
                    }
                }
            }
            RowSrc::VarKey {
                per_sample_call,
                info_call,
            } => {
                let v = vk_view
                    .as_ref()
                    .expect("flipped row needs the source var_key sub");
                match spec.category {
                    // INFO: one element per row; any call of the variant carries it.
                    FieldCategory::Info => out.extend_from_slice(v.bytes_at(*info_call)),
                    // FORMAT: a full n_subset-wide column; NON-CARRIERS get the sentinel.
                    FieldCategory::Format => {
                        for call in per_sample_call {
                            match call {
                                Some(c) => out.extend_from_slice(v.bytes_at(*c)),
                                None => out.extend_from_slice(&sentinel),
                            }
                        }
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snp_2bit_code_round_trips_through_the_uniform_key_space() {
        for c in 0u8..4 {
            assert_eq!(decode_snp_2bit_code(snp_code_to_key(c)), c);
        }
    }
}

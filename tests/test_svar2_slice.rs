// Disk-integration tests for `slice_contig_genos`: the `reroute=False`
// direct array-slicer for one finished SVAR2 contig's genotype sidecars
// (`var_key/*`, `dense/*`, the indel LUT, `max_del`). Builds a small store via
// the real conversion pipeline (SNP + indel, var_key + dense routed, one long
// insertion that spills to the LUT), then checks:
//
//   1. a full-region, all-sample slice is byte-identical to the source (an
//      identity slice), across every genotype sidecar file; and
//   2. a 1-sample subset slice decodes (via the production query path) to
//      exactly the same calls as querying that sample directly on the source.

mod common;

use common::{SynthRecord, build_contig, build_fasta_with_index};
use genoray_core::field::{FieldCategory, FieldSpec, HtslibType, StorageDtype};
use genoray_core::layout::{ContigPaths, FieldSub};
use genoray_core::meta::{FORMAT_VERSION, write_meta};
use genoray_core::mutcat::annotate::annotate_contig;
use genoray_core::query::field::{FieldValue, FieldView};
use genoray_core::query::{ContigReader, oracle::overlap_sample};
use genoray_core::run_slice_view;
use genoray_core::svar2_slice::{Routing, slice_contig, slice_contig_genos};
use genoray_core::svar2_source::OverlapMode;
use pyo3::Python;
use std::path::Path;
use tempfile::{TempDir, tempdir};

/// Five records exercising every routing/LUT combination at n=2, ploidy=2
/// (np=4): a rare (var_key) and common (dense, x=2) SNP carried by sample 0,
/// and a rare (var_key) indel, a common (dense, x=2) indel, and a rare long
/// insertion (var_key + LUT spill) all carried by sample 1. Flat gt order is
/// `[s0p0, s0p1, s1p0, s1p1]`.
fn fixture_records() -> Vec<SynthRecord<'static>> {
    vec![
        // sample 0: SNP var_key (x=1) + SNP dense (x=2)
        SynthRecord {
            pos: 10,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0],
        },
        SynthRecord {
            pos: 20,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![1, 1, 0, 0],
        },
        // sample 1: indel var_key (x=1) + indel dense (x=2) + long-INS var_key (LUT, x=1)
        SynthRecord {
            pos: 30,
            ref_allele: b"ATG",
            alts: vec![&b"A"[..]], // pure DEL, ilen = -2
            gt: vec![0, 0, 1, 0],
        },
        SynthRecord {
            pos: 40,
            ref_allele: b"A",
            alts: vec![&b"ACG"[..]], // INS, ilen = 2
            gt: vec![0, 0, 1, 1],
        },
        SynthRecord {
            pos: 50,
            ref_allele: b"A",
            alts: vec![&b"ACGTACGTACGTACGT"[..]], // 16bp INS -> spills to LUT
            gt: vec![0, 0, 1, 0],
        },
    ]
}

fn build_fixture_store(dir: &Path, samples: &[&str]) {
    std::fs::create_dir_all(dir).unwrap();
    let records = fixture_records();
    build_contig(dir, "chr1", samples, 2, &records);
    // `build_contig` overwrites `max_del.npy`/`dense/max_del.npy` with a
    // deliberately conservative test fixture (`write_max_del_fixture`, over
    // ALL records regardless of routing) rather than the real production scan
    // `process_chromosome` already ran. `slice_contig_genos` recomputes the
    // REAL (routing-aware) max_del for its output, so the source must have the
    // real value too for the full-coverage byte-parity check below to be
    // meaningful — recompute it here to undo the fixture's conservative
    // overwrite.
    genoray_core::max_del::write_max_del(&dir.join("chr1"), samples.len(), 2).unwrap();
    write_meta(
        dir,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();
}

fn read_if_exists(base: &Path, rel: &str) -> Option<Vec<u8>> {
    let p = base.join(rel);
    if p.exists() {
        Some(std::fs::read(p).unwrap())
    } else {
        None
    }
}

#[test]
fn slice_full_coverage_is_byte_identical_genos() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let n = slice_contig_genos(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "chr1",
        &(0..samples.len()).collect::<Vec<_>>(),
        2,
        &[(0, u32::MAX)],
        OverlapMode::Variant,
        Routing::Preserve,
    )
    .unwrap();
    assert_eq!(
        n, 5,
        "all 5 fixture variants must survive a full-coverage slice"
    );

    for rel in [
        "var_key/snp/positions.bin",
        "var_key/snp/alleles.bin",
        "var_key/snp/offsets.npy",
        "var_key/indel/positions.bin",
        "var_key/indel/alleles.bin",
        "var_key/indel/offsets.npy",
        "dense/snp/positions.bin",
        "dense/snp/alleles.bin",
        "dense/snp/genotypes.bin",
        "dense/indel/positions.bin",
        "dense/indel/alleles.bin",
        "dense/indel/genotypes.bin",
        "max_del.npy",
        "dense/max_del.npy",
        "indel/long_alleles.bin",
        "indel/long_allele_offsets.npy",
    ] {
        let rel = Path::new("chr1").join(rel);
        let rel = rel.to_str().unwrap();
        assert_eq!(
            read_if_exists(&src, rel),
            read_if_exists(&out, rel),
            "{rel}"
        );
    }
}

#[test]
fn slice_one_sample_subset_decodes_equivalently_to_the_source() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    // Sample 1 (original index 1) carries the indel var_key, indel dense, and
    // long-INS (LUT) records — sample 0's SNPs must drop out entirely.
    let n = slice_contig_genos(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "chr1",
        &[1],
        2,
        &[(0, u32::MAX)],
        OverlapMode::Variant,
        Routing::Preserve,
    )
    .unwrap();
    assert_eq!(
        n, 3,
        "only sample 1's 3 variants should survive the subset slice"
    );

    let src_reader = ContigReader::open(src.to_str().unwrap(), "chr1", samples.len(), 2).unwrap();
    let out_reader = ContigReader::open(out.to_str().unwrap(), "chr1", 1, 2).unwrap();

    let want = overlap_sample(&src_reader, 1, 0, u32::MAX);
    let got = overlap_sample(&out_reader, 0, 0, u32::MAX);
    assert_eq!(got, want);

    // Sanity: the subset genuinely carries something (guards against a
    // vacuously-passing empty-vs-empty comparison).
    assert!(want.per_hap.iter().any(|hc| !hc.positions.is_empty()));
}

#[test]
fn slice_variant_mode_excludes_non_overlapping_indel_in_widened_window() {
    // Regression for the missing per-element left-extent re-check. The three
    // records are all carried by S0/hap0, so they share var_key/indel column 0
    // (positions ascending: 28, 31, 36):
    //   D1 @28 DEL of 6 bases  -> extent [28, 35)  (truly overlaps [34, 44))
    //   A  @31 DEL of 1 base   -> extent [31, 33)  (does NOT reach q_start=34)
    //   D2 @36 DEL of 1 base   -> extent [36, 38)  (truly overlaps [34, 44))
    //
    // `overlap_range` already trims the first/last non-overlaps, so a boundary
    // non-overlapper would never surface — the bug needs A INTERIOR between two
    // true overlaps. Column 0's max_del of 6 (from D1) widens the search's left
    // bound to 28, so `overlap_range` returns the contiguous window [D1, A, D2];
    // its own forward/backward scans keep D1 and D2 (v_end > 34) as the window
    // ends, leaving A interior (v_end 33 <= 34). `keeps(Variant, ...)` returns
    // `true` for A, so the ONLY thing that excludes it is the per-element
    // `q_start < v_end` re-check (34 < 33 is false) that `gather_vk` /
    // `spine::gather_keys` apply on the real query path. Without it the slicer
    // wrote A as an extra var_key indel call `reroute=True` never emits.
    //
    // The reference bases at 28..37 are laid out so the overlapping DEL REF
    // stamps agree and no allele's anchor repeats at its end (no left-align
    // shift): 28=C 29=G 30=T 31=C 32=G 33=T 34=G 35=A 36=A 37=T.
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    let samples = ["S0"];
    let records = vec![
        SynthRecord {
            pos: 28,
            ref_allele: b"CGTCGTG", // 28..34; anchor C, last G -> no roll
            alts: vec![&b"C"[..]],  // DEL, ilen = -6, v_end = 35
            gt: vec![1, 0],
        },
        SynthRecord {
            pos: 31,
            ref_allele: b"CG",     // 31..32; consistent with D1's bases there
            alts: vec![&b"C"[..]], // DEL, ilen = -1, v_end = 33
            gt: vec![1, 0],
        },
        SynthRecord {
            pos: 36,
            ref_allele: b"AT",     // 36..37
            alts: vec![&b"A"[..]], // DEL, ilen = -1, v_end = 38
            gt: vec![1, 0],
        },
    ];
    build_contig(&src, "chr1", &samples, 2, &records);
    // Recompute the REAL (routing-aware) max_del over `build_contig`'s
    // conservative fixture (both give col 0 max_del = 6, which widens the
    // window's left bound to 28 so D1 — and thus interior A — enter it).
    genoray_core::max_del::write_max_del(&src.join("chr1"), samples.len(), 2).unwrap();
    write_meta(
        &src,
        FORMAT_VERSION,
        &samples.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    slice_contig_genos(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "chr1",
        &[0],
        2,
        &[(34, 44)],
        OverlapMode::Variant,
        Routing::Preserve,
    )
    .unwrap();

    let src_reader = ContigReader::open(src.to_str().unwrap(), "chr1", samples.len(), 2).unwrap();
    let out_reader = ContigReader::open(out.to_str().unwrap(), "chr1", 1, 2).unwrap();

    // `overlap_sample` over the query region on the source is the reroute=True
    // reference selection for `Variant` mode (same extent re-check via
    // `spine::gather_keys`); the sliced store decoded over its whole extent
    // must match it exactly.
    let want = overlap_sample(&src_reader, 0, 34, 44);
    let got = overlap_sample(&out_reader, 0, 0, u32::MAX);
    assert_eq!(
        got, want,
        "sliced store must not contain the interior non-overlapping indel"
    );

    // Positive assertions pinning the mechanism: the two true-overlap DELs
    // survive, the interior non-overlapping DEL @31 does not.
    let hap0 = &got.per_hap[0];
    assert_eq!(
        hap0.positions,
        vec![28, 36],
        "only the two truly-overlapping DELs survive"
    );
    assert!(
        !hap0.positions.contains(&31),
        "interior non-overlapping DEL @31 must be excluded"
    );
}

// ---- Field slicing (Task 2) ----

fn write_field_f32(paths: &ContigPaths, name: &str, sub: FieldSub, vals: &[f32]) {
    let p = paths.field_values("info", name, sub);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, bytemuck::cast_slice(vals)).unwrap();
}

fn write_field_i32(paths: &ContigPaths, name: &str, sub: FieldSub, vals: &[i32]) {
    let p = paths.field_values("format", name, sub);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, bytemuck::cast_slice(vals)).unwrap();
}

fn vk_call_count(store: &Path, kind: &str) -> usize {
    // last offset of var_key/{kind}/offsets.npy == total CSR calls.
    let off = store
        .join("chr1")
        .join("var_key")
        .join(kind)
        .join("offsets.npy");
    let arr: ndarray::Array1<u64> = ndarray_npy::read_npy(off).unwrap();
    *arr.last().unwrap() as usize
}

fn dense_row_count(store: &Path, kind: &str) -> usize {
    let pos = store
        .join("chr1")
        .join("dense")
        .join(kind)
        .join("positions.bin");
    std::fs::read(pos).map(|b| b.len() / 4).unwrap_or(0)
}

/// Synthesize the `AF` (INFO, f32) and `DP` (FORMAT, i32) source field sidecars
/// on a genotype store, sized to its ACTUAL var_key call / dense row counts,
/// with distinct recognizable values. Returns the two `FieldSpec`s. Asserts the
/// routing produced the counts this test reasons about (a routing change would
/// trip this loudly, not silently).
fn synth_fields(src: &Path, n_samples: usize) -> Vec<FieldSpec> {
    let paths = ContigPaths::new(src.to_str().unwrap(), "chr1");
    let n_vksnp = vk_call_count(src, "snp");
    let n_vkindel = vk_call_count(src, "indel");
    let n_dsnp = dense_row_count(src, "snp");
    let n_dindel = dense_row_count(src, "indel");
    assert_eq!(
        (n_vksnp, n_vkindel, n_dsnp, n_dindel),
        (1, 2, 1, 1),
        "fixture routing changed; update the field test's expected counts"
    );

    // AF (INFO): one element per var_key CALL and per dense ROW.
    write_field_f32(&paths, "AF", FieldSub::VkSnp, &[10.5]);
    write_field_f32(&paths, "AF", FieldSub::VkIndel, &[30.5, 50.5]);
    write_field_f32(&paths, "AF", FieldSub::DenseSnp, &[20.5]);
    write_field_f32(&paths, "AF", FieldSub::DenseIndel, &[40.5]);

    // DP (FORMAT): per var_key CALL, and per dense ROW * n_samples (variant-major).
    write_field_i32(&paths, "DP", FieldSub::VkSnp, &[1000]);
    write_field_i32(&paths, "DP", FieldSub::VkIndel, &[3000, 5000]);
    assert_eq!(n_dsnp * n_samples, 2);
    write_field_i32(&paths, "DP", FieldSub::DenseSnp, &[2000, 2001]);
    assert_eq!(n_dindel * n_samples, 2);
    write_field_i32(&paths, "DP", FieldSub::DenseIndel, &[4000, 4001]);

    vec![
        FieldSpec {
            name: "AF".into(),
            category: FieldCategory::Info,
            htype: HtslibType::Float,
            dtype: StorageDtype::F32,
            default: None,
        },
        FieldSpec {
            name: "DP".into(),
            category: FieldCategory::Format,
            htype: HtslibType::Int,
            dtype: StorageDtype::I32,
            default: None,
        },
    ]
}

const FIELD_SUBS: [(&str, FieldSub); 4] = [
    ("var_key_snp", FieldSub::VkSnp),
    ("var_key_indel", FieldSub::VkIndel),
    ("dense_snp", FieldSub::DenseSnp),
    ("dense_indel", FieldSub::DenseIndel),
];

#[test]
fn slice_fields_full_coverage_is_byte_identical() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);
    let fields = synth_fields(&src, samples.len());

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    slice_contig(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "chr1",
        &(0..samples.len()).collect::<Vec<_>>(),
        2,
        &[(0, u32::MAX)],
        OverlapMode::Variant,
        &fields,
        Routing::Preserve,
        false, // sidecar_bits_enabled (ignored under Preserve)
        0,     // info_bits (ignored under Preserve)
        0,     // format_bits (ignored under Preserve)
    )
    .unwrap();

    for (cat, name) in [("info", "AF"), ("format", "DP")] {
        for (sub_dir, _) in FIELD_SUBS {
            let rel = format!("chr1/fields/{cat}/{name}/{sub_dir}/values.bin");
            assert_eq!(
                read_if_exists(&src, &rel),
                read_if_exists(&out, &rel),
                "{rel}"
            );
        }
    }
}

#[test]
fn slice_fields_subset_decodes_equivalently() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);
    let fields = synth_fields(&src, samples.len());

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    // Subset to S1 (orig index 1): only its variants (indel var_key x2 + indel
    // dense x1) survive; S0's SNPs drop out entirely.
    slice_contig(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "chr1",
        &[1],
        2,
        &[(0, u32::MAX)],
        OverlapMode::Variant,
        &fields,
        Routing::Preserve,
        false, // sidecar_bits_enabled (ignored under Preserve)
        0,     // info_bits (ignored under Preserve)
        0,     // format_bits (ignored under Preserve)
    )
    .unwrap();

    let src_paths = ContigPaths::new(src.to_str().unwrap(), "chr1");
    let out_paths = ContigPaths::new(out.to_str().unwrap(), "chr1");

    // AF (INFO) var_key/indel: both source calls (pos30, pos50) survive in the
    // same order -> values carried through unchanged.
    let af_vki_src = FieldView::open(
        &src_paths,
        "info",
        "AF",
        FieldSub::VkIndel,
        StorageDtype::F32,
        2,
    )
    .unwrap();
    let af_vki_out = FieldView::open(
        &out_paths,
        "info",
        "AF",
        FieldSub::VkIndel,
        StorageDtype::F32,
        1,
    )
    .unwrap();
    assert_eq!(af_vki_out.len(), 2);
    assert_eq!(af_vki_out.value_at(0), af_vki_src.value_at(0));
    assert_eq!(af_vki_out.value_at(1), af_vki_src.value_at(1));
    assert_eq!(af_vki_out.value_at(0), FieldValue::F32(30.5));
    assert_eq!(af_vki_out.value_at(1), FieldValue::F32(50.5));

    // DP (FORMAT) var_key/indel: per-call, same order.
    let dp_vki_out = FieldView::open(
        &out_paths,
        "format",
        "DP",
        FieldSub::VkIndel,
        StorageDtype::I32,
        1,
    )
    .unwrap();
    assert_eq!(dp_vki_out.len(), 2);
    assert_eq!(dp_vki_out.value_at(0), FieldValue::I32(3000));
    assert_eq!(dp_vki_out.value_at(1), FieldValue::I32(5000));

    // AF (INFO) dense/indel: the one kept row's value carried through.
    let af_di_out = FieldView::open(
        &out_paths,
        "info",
        "AF",
        FieldSub::DenseIndel,
        StorageDtype::F32,
        1,
    )
    .unwrap();
    assert_eq!(af_di_out.len(), 1);
    assert_eq!(af_di_out.value_at(0), FieldValue::F32(40.5));

    // DP (FORMAT) dense/indel: re-strided to the subset. Source row 0 x sample
    // 1 (orig) = element 0*2 + 1 = 4001; new layout is row_out 0 x s_out 0 =
    // element 0*1 + 0. `format_at(0, 0)` on the sliced store must read 4001.
    let dp_di_out = FieldView::open(
        &out_paths,
        "format",
        "DP",
        FieldSub::DenseIndel,
        StorageDtype::I32,
        1, // subset cohort size (n_subset = 1)
    )
    .unwrap();
    assert_eq!(dp_di_out.len(), 1, "1 kept row * 1 subset sample");
    assert_eq!(dp_di_out.format_at(0, 0), FieldValue::I32(4001));

    // S0-only sub-streams collapse to empty in the subset (0 kept calls/rows).
    let af_vks_out = FieldView::open(
        &out_paths,
        "info",
        "AF",
        FieldSub::VkSnp,
        StorageDtype::F32,
        1,
    )
    .unwrap();
    assert!(af_vks_out.is_empty(), "S0's SNP var_key field drops out");
    let af_ds_out = FieldView::open(
        &out_paths,
        "info",
        "AF",
        FieldSub::DenseSnp,
        StorageDtype::F32,
        1,
    )
    .unwrap();
    assert!(af_ds_out.is_empty(), "S0's dense SNP field drops out");
}

// ---- Routing::Preserve (Task 2): gather -> route -> emit must be byte-parity ----

/// Every genotype + field sidecar `slice_contig` writes, relative to
/// `{store}/chr1/` — the same rel list `slice_full_coverage_is_byte_identical_genos`
/// and `slice_fields_full_coverage_is_byte_identical` check separately, combined,
/// since `Routing::Preserve` must reproduce BOTH exactly.
fn geno_and_field_parity_rels() -> Vec<String> {
    let mut rels: Vec<String> = [
        "var_key/snp/positions.bin",
        "var_key/snp/alleles.bin",
        "var_key/snp/offsets.npy",
        "var_key/indel/positions.bin",
        "var_key/indel/alleles.bin",
        "var_key/indel/offsets.npy",
        "dense/snp/positions.bin",
        "dense/snp/alleles.bin",
        "dense/snp/genotypes.bin",
        "dense/indel/positions.bin",
        "dense/indel/alleles.bin",
        "dense/indel/genotypes.bin",
        "max_del.npy",
        "dense/max_del.npy",
        "indel/long_alleles.bin",
        "indel/long_allele_offsets.npy",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for (cat, name) in [("info", "AF"), ("format", "DP")] {
        for sub in ["var_key_snp", "var_key_indel", "dense_snp", "dense_indel"] {
            rels.push(format!("fields/{cat}/{name}/{sub}/values.bin"));
        }
    }
    rels
}

/// The whole point of Task 2: splitting the slicer into gather -> route -> emit
/// must not move a single byte when `routing` is `Routing::Preserve` (every
/// variant keeps its source stream). A full-coverage identity slice through the
/// new two-phase entry point must reproduce the source's genotype AND field
/// sidecars exactly.
#[test]
fn preserve_identity_slice_is_byte_parity() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);
    let fields = synth_fields(&src, samples.len());

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let n = slice_contig(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "chr1",
        &(0..samples.len()).collect::<Vec<_>>(),
        2,
        &[(0, u32::MAX)],
        OverlapMode::Variant,
        &fields,
        Routing::Preserve,
        false, // sidecar_bits_enabled (ignored under Preserve)
        0,     // info_bits (ignored under Preserve)
        0,     // format_bits (ignored under Preserve)
    )
    .unwrap();
    assert_eq!(
        n, 5,
        "all 5 fixture variants must survive a full-coverage Routing::Preserve slice"
    );

    for rel in geno_and_field_parity_rels() {
        let rel = Path::new("chr1").join(rel);
        let rel = rel.to_str().unwrap();
        assert_eq!(
            read_if_exists(&src, rel),
            read_if_exists(&out, rel),
            "{rel}"
        );
    }
}

// ---- run_slice_view pyfunction (Task 3) ----

/// Reconstruct the reference sequence exactly as `common::build_fasta_with_index`
/// stamps it (an 'N' background with each record's REF written at its 0-based
/// pos), so the ref the test annotates the SOURCE mutcat with agrees byte-for-
/// byte with `src/in.fa` — the fasta `run_slice_view` loads for the OUTPUT.
fn reconstruct_ref(records: &[SynthRecord], len: usize) -> Vec<u8> {
    let mut seq = vec![b'N'; len];
    for r in records {
        let s = r.pos as usize;
        seq[s..s + r.ref_allele.len()].copy_from_slice(r.ref_allele);
    }
    seq
}

/// Every full-coverage-identity sidecar `run_slice_view` must reproduce
/// byte-for-byte: genotypes, LUT, max_del, the two fields' four subs each, and
/// the mutcat code/ref sidecars.
fn all_parity_rels() -> Vec<String> {
    let mut rels: Vec<String> = [
        "var_key/snp/positions.bin",
        "var_key/snp/alleles.bin",
        "var_key/snp/offsets.npy",
        "var_key/indel/positions.bin",
        "var_key/indel/alleles.bin",
        "var_key/indel/offsets.npy",
        "dense/snp/positions.bin",
        "dense/snp/alleles.bin",
        "dense/snp/genotypes.bin",
        "dense/indel/positions.bin",
        "dense/indel/alleles.bin",
        "dense/indel/genotypes.bin",
        "max_del.npy",
        "dense/max_del.npy",
        "indel/long_alleles.bin",
        "indel/long_allele_offsets.npy",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for (cat, name) in [("info", "AF"), ("format", "DP")] {
        for sub in ["var_key_snp", "var_key_indel", "dense_snp", "dense_indel"] {
            rels.push(format!("fields/{cat}/{name}/{sub}/values.bin"));
        }
    }
    for sub in ["var_key_snp", "var_key_indel", "dense_snp", "dense_indel"] {
        rels.push(format!("mutcat/{sub}/code.bin"));
    }
    for sub in ["var_key_snp", "dense_snp"] {
        rels.push(format!("mutcat/{sub}/ref.bin"));
    }
    rels
}

#[test]
fn run_slice_view_full_coverage_carries_genos_fields_and_mutcat() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples); // genos + meta + max_del
    let _ = synth_fields(&src, samples.len()); // AF (info/f32) + DP (format/i32)

    // Annotate the SOURCE mutcat with a ref matching the fixture positions —
    // the SAME sequence `src/in.fa` holds, so the OUTPUT (annotated by
    // run_slice_view from that fasta) must produce byte-identical sidecars.
    let ref_seq = reconstruct_ref(&fixture_records(), 1_000_000);
    let src_reader = ContigReader::open(src.to_str().unwrap(), "chr1", samples.len(), 2).unwrap();
    let src_paths = ContigPaths::new(src.to_str().unwrap(), "chr1");
    annotate_contig(&src_reader, &src_paths, &ref_seq).unwrap();

    let out = tmp.path().join("out");
    let fasta = src.join("in.fa");
    let field_tuples = vec![
        (
            "AF".to_string(),
            "info".to_string(),
            "f32".to_string(),
            None,
        ),
        (
            "DP".to_string(),
            "format".to_string(),
            "i32".to_string(),
            None,
        ),
    ];

    Python::attach(|py| {
        run_slice_view(
            py,
            src.to_str().unwrap().to_string(),
            out.to_str().unwrap().to_string(),
            vec!["chr1".to_string()],
            samples.iter().map(|s| s.to_string()).collect(),
            vec![("chr1".to_string(), 0u32, u32::MAX)],
            "variant".to_string(),
            false,
            field_tuples,
            Some(fasta.to_str().unwrap().to_string()),
            false,
        )
    })
    .expect("run_slice_view should succeed");

    // (a) meta.json: subset samples + the requested fields (name/category/dtype).
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("meta.json")).unwrap()).unwrap();
    assert_eq!(meta["samples"], serde_json::json!(["S0", "S1"]));
    assert_eq!(meta["contigs"], serde_json::json!(["chr1"]));
    assert_eq!(meta["ploidy"], 2);
    assert_eq!(
        meta["fields"][0],
        serde_json::json!({"name":"AF","category":"info","dtype":"f32","default":null})
    );
    assert_eq!(
        meta["fields"][1],
        serde_json::json!({"name":"DP","category":"format","dtype":"i32","default":null})
    );

    // (b)+(c) genotype/field/mutcat sidecars byte-identical (full-coverage identity).
    for rel in all_parity_rels() {
        let rel = format!("chr1/{rel}");
        assert_eq!(
            read_if_exists(&src, &rel),
            read_if_exists(&out, &rel),
            "{rel}"
        );
    }

    // Non-vacuity guard: mutcat was actually written (so the parity above is not
    // a None == None pass).
    assert!(
        out.join("chr1/mutcat/var_key_snp/code.bin").exists(),
        "mutcat code.bin must be present in the output"
    );
}

#[test]
fn run_slice_view_without_reference_skips_mutcat() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);

    let out = tmp.path().join("out");
    Python::attach(|py| {
        run_slice_view(
            py,
            src.to_str().unwrap().to_string(),
            out.to_str().unwrap().to_string(),
            vec!["chr1".to_string()],
            samples.iter().map(|s| s.to_string()).collect(),
            vec![("chr1".to_string(), 0u32, u32::MAX)],
            "variant".to_string(),
            false,
            Vec::new(),
            None, // no reference -> no mutcat
            false,
        )
    })
    .expect("run_slice_view should succeed without a reference");

    assert!(out.join("meta.json").exists());
    assert!(
        !out.join("chr1/mutcat").exists(),
        "no reference => mutcat must be skipped entirely"
    );
}

/// A nonexistent `reference` must be rejected in the fail-fast band, BEFORE
/// any output byte is written — not discovered mid-loop (the mutcat-recompute
/// step) after that contig's genotype/field sidecars are already on disk.
/// Before the fix, `load_contig_seq` was only called inside the per-contig
/// loop, AFTER `create_dir_all(out_dir)` and after `slice_contig` had already
/// written chr1's sidecars — leaving a partially-written, unreadable store
/// (sidecars present, `meta.json` absent) instead of a clean raise.
#[test]
fn run_slice_view_bad_reference_fails_before_any_output() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);

    let out = tmp.path().join("out");
    assert!(!out.exists(), "precondition: out_dir must not exist yet");

    let bad_fasta = tmp.path().join("does-not-exist.fa");
    let result = Python::attach(|py| {
        run_slice_view(
            py,
            src.to_str().unwrap().to_string(),
            out.to_str().unwrap().to_string(),
            vec!["chr1".to_string()],
            samples.iter().map(|s| s.to_string()).collect(),
            vec![("chr1".to_string(), 0u32, u32::MAX)],
            "variant".to_string(),
            false,
            Vec::new(),
            Some(bad_fasta.to_str().unwrap().to_string()),
            false,
        )
    });

    assert!(
        result.is_err(),
        "a nonexistent reference FASTA must be rejected"
    );
    assert!(
        !out.exists(),
        "fail-fast: out_dir must NOT be created when the reference is invalid \
         (got a partially-written store instead)"
    );
}

/// A reference FASTA that EXISTS and is openable but is MISSING an out-contig
/// must also be rejected up front, before any output byte — the up-front
/// validation checks every out-contig's presence (via `fetch_seq_len`), not
/// just that the path/faidx opens. Uses a fasta whose only contig is
/// "chrOther", so requesting a view on "chr1" fails the per-contig existence
/// check in the fail-fast band.
#[test]
fn run_slice_view_reference_missing_contig_fails_before_any_output() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);

    // A real, openable FASTA (+ .fai) that simply lacks "chr1".
    let other_fasta = tmp.path().join("other.fa");
    build_fasta_with_index(&other_fasta, "chrOther", 1_000, &[]);

    let out = tmp.path().join("out");
    assert!(!out.exists(), "precondition: out_dir must not exist yet");

    let result = Python::attach(|py| {
        run_slice_view(
            py,
            src.to_str().unwrap().to_string(),
            out.to_str().unwrap().to_string(),
            vec!["chr1".to_string()],
            samples.iter().map(|s| s.to_string()).collect(),
            vec![("chr1".to_string(), 0u32, u32::MAX)],
            "variant".to_string(),
            false,
            Vec::new(),
            Some(other_fasta.to_str().unwrap().to_string()),
            false,
        )
    });

    assert!(
        result.is_err(),
        "a reference FASTA missing an out-contig must be rejected"
    );
    assert!(
        !out.exists(),
        "fail-fast: out_dir must NOT be created when the reference lacks an \
         out-contig (got a partially-written store instead)"
    );
}

// ---- Routing::Recompute (Task 3): the cost model re-runs on the subset ----

/// The single SNP in every flip fixture below lives at this 0-based POS. Stored
/// positions are the `SynthRecord.pos` verbatim (see the other tests), so this
/// is exactly what `var_key/snp/positions.bin` / `dense/snp/positions.bin` hold.
const SNP_POS: u32 = 100;

/// A finished single-contig genotype store plus the metadata a subset slice
/// needs (its own tempdir keeps the store alive for the test's lifetime).
struct GenoFixture {
    _tmp: TempDir,
    path: String,
    n_samples: usize,
}

/// Build a `chr1` store holding ONE biallelic SNP (`A>C` at [`SNP_POS`]) with
/// the given flat genotype vector, routed by the REAL production cost model
/// (`process_chromosome`, signatures off, no fields -> sidecar/info/format bits
/// all 0). `gt` is `[s0p0, s0p1, s1p0, ...]`.
fn build_geno_fixture(n_samples: usize, gt: Vec<i32>) -> GenoFixture {
    assert_eq!(gt.len(), n_samples * 2, "gt must be n_samples * ploidy(2)");
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    let sample_names: Vec<String> = (0..n_samples).map(|i| format!("S{i}")).collect();
    let sample_refs: Vec<&str> = sample_names.iter().map(|s| s.as_str()).collect();

    let records = vec![SynthRecord {
        pos: SNP_POS as i64,
        ref_allele: b"A",
        alts: vec![&b"C"[..]],
        gt,
    }];
    build_contig(&src, "chr1", &sample_refs, 2, &records);
    // Undo `build_contig`'s conservative max_del fixture with the real value
    // (irrelevant for a SNP-only store, but keeps the store production-shaped).
    genoray_core::max_del::write_max_del(&src.join("chr1"), n_samples, 2).unwrap();
    write_meta(
        &src,
        FORMAT_VERSION,
        &sample_names,
        &["chr1".to_string()],
        2,
        &[],
    )
    .unwrap();

    GenoFixture {
        path: src.to_str().unwrap().to_string(),
        n_samples,
        _tmp: tmp,
    }
}

/// A SNP common enough in a large cohort to route DENSE at full size, but
/// carried by sample 0 on exactly ONE hap (so a 1-sample `[0]` subset sees
/// `x_sub = 1`). Additional carrier haps fill from hap index 2 onward.
fn fixture_dense_snp_store(n_samples: usize, carrier_haps: usize) -> GenoFixture {
    assert!(carrier_haps >= 1 && carrier_haps <= n_samples * 2);
    let mut gt = vec![0i32; n_samples * 2];
    gt[0] = 1; // sample 0, hap 0 -> the sole carrier hap of sample 0
    for h in gt.iter_mut().take(carrier_haps + 1).skip(2) {
        *h = 1;
    }
    // total carriers = 1 (hap0) + (carrier_haps + 1 - 2) = carrier_haps.
    build_geno_fixture(n_samples, gt)
}

/// A SNP rare enough to route VAR_KEY at full size (3 carrier haps), all within
/// samples 0 and 1: sample 0 both haps + sample 1 hap 0. A `[0, 1]` subset
/// keeps all 3 carriers (`x_sub = 3`).
fn fixture_var_key_snp_store(n_samples: usize) -> GenoFixture {
    assert!(n_samples >= 2);
    let mut gt = vec![0i32; n_samples * 2];
    gt[0] = 1; // s0 hap0
    gt[1] = 1; // s0 hap1
    gt[2] = 1; // s1 hap0
    build_geno_fixture(n_samples, gt)
}

fn snp_positions(store: &str, chrom: &str, stream: &str) -> Vec<u32> {
    let p = Path::new(store)
        .join(chrom)
        .join(stream)
        .join("snp")
        .join("positions.bin");
    if p.exists() {
        common::read_u32_bin(&p)
    } else {
        Vec::new()
    }
}

fn var_key_snp_positions(store: &str, chrom: &str) -> Vec<u32> {
    snp_positions(store, chrom, "var_key")
}

fn dense_snp_positions(store: &str, chrom: &str) -> Vec<u32> {
    snp_positions(store, chrom, "dense")
}

/// Decode-equivalence: for each output sample `s_out` (original column
/// `orig[s_out]`), the sliced store must decode to exactly the calls the source
/// yields for that original sample. Representation-agnostic (var_key vs dense),
/// so it holds across a flip.
fn assert_decode_matches_source(
    src: &str,
    src_n_samples: usize,
    out: &str,
    chrom: &str,
    orig: &[usize],
) {
    let src_reader = ContigReader::open(src, chrom, src_n_samples, 2).unwrap();
    let out_reader = ContigReader::open(out, chrom, orig.len(), 2).unwrap();
    let mut carried_something = false;
    for (s_out, &s_orig) in orig.iter().enumerate() {
        let want = overlap_sample(&src_reader, s_orig, 0, u32::MAX);
        let got = overlap_sample(&out_reader, s_out, 0, u32::MAX);
        assert_eq!(got, want, "sample {s_out} (orig {s_orig}) decode mismatch");
        carried_something |= want.per_hap.iter().any(|hc| !hc.positions.is_empty());
    }
    assert!(
        carried_something,
        "non-vacuity: the subset must genuinely carry the SNP"
    );
}

/// A SNP carried by MANY haps of a large cohort routes DENSE at full size; a
/// 1-sample subset in which it has a single carrier hap must flip dense ->
/// var_key.
///
/// By-hand cost (SNP: `per_call = 32 + 2 + sidecar`, `sidecar = 0` genos-only;
/// `dense = 32 + 2 + n*ploidy`; route Dense iff `dense < per_call * x`):
///   source, n=200 ploidy=2 (np=400), x=180:
///     dense = 32+2+400 = 434; var_key = 34*180 = 6120 -> 434 < 6120 -> DENSE.
///   subset [0], n_sub=1 (np=2), x_sub=1:
///     dense = 32+2+2 = 36; var_key = 34*1 = 34 -> 36 < 34 is false -> VAR_KEY.
///   => the variant FLIPS dense -> var_key.
#[test]
fn recompute_flips_dense_to_var_key() {
    let src = fixture_dense_snp_store(200, 180);
    let out = tempdir().unwrap();
    let outp = out.path().to_str().unwrap();

    // Precondition: the source really routed this SNP dense.
    assert!(dense_snp_positions(&src.path, "chr1").contains(&SNP_POS));
    assert!(var_key_snp_positions(&src.path, "chr1").is_empty());

    slice_contig_genos(
        &src.path,
        outp,
        "chr1",
        &[0],
        2,
        &[(0, u32::MAX)],
        OverlapMode::Variant,
        Routing::Recompute,
    )
    .unwrap();

    assert!(
        var_key_snp_positions(outp, "chr1").contains(&SNP_POS),
        "flipped SNP must now live in var_key/snp"
    );
    assert!(
        dense_snp_positions(outp, "chr1").is_empty(),
        "dense/snp must be empty after the flip"
    );
    assert_decode_matches_source(&src.path, src.n_samples, outp, "chr1", &[0]);
}

/// A var_key SNP with few carriers at full size, all packed into a small subset,
/// must flip var_key -> dense.
///
/// By-hand cost:
///   source, n=100 ploidy=2 (np=200), x=3:
///     dense = 32+2+200 = 234; var_key = 34*3 = 102 -> 234 < 102 false -> VAR_KEY.
///   subset [0,1], n_sub=2 (np=4), x_sub=3:
///     dense = 32+2+4 = 38; var_key = 34*3 = 102 -> 38 < 102 -> DENSE.
///   => the variant FLIPS var_key -> dense.
#[test]
fn recompute_flips_var_key_to_dense() {
    let src = fixture_var_key_snp_store(100);
    let out = tempdir().unwrap();
    let outp = out.path().to_str().unwrap();

    // Precondition: the source really routed this SNP var_key.
    assert!(var_key_snp_positions(&src.path, "chr1").contains(&SNP_POS));
    assert!(dense_snp_positions(&src.path, "chr1").is_empty());

    slice_contig_genos(
        &src.path,
        outp,
        "chr1",
        &[0, 1],
        2,
        &[(0, u32::MAX)],
        OverlapMode::Variant,
        Routing::Recompute,
    )
    .unwrap();

    assert!(
        dense_snp_positions(outp, "chr1").contains(&SNP_POS),
        "flipped SNP must now live in dense/snp"
    );
    assert!(
        var_key_snp_positions(outp, "chr1").is_empty(),
        "var_key/snp must be empty after the flip"
    );
    assert_decode_matches_source(&src.path, src.n_samples, outp, "chr1", &[0, 1]);
}

/// Full coverage => every variant's `x_sub` equals its full-size `x`, and the
/// cost terms match the source conversion's (genotype fixture built field-free:
/// sidecar/info/format bits all 0), so `Routing::Recompute` produces ZERO flips.
/// Zero flips => Recompute reproduces `Routing::Preserve`, which is byte-parity
/// with the source. The strongest test in the suite: it pins that the Recompute
/// route + `RoutePlan::sort` are a genuine no-op on the bytes when nothing moves.
///
/// Cost params passed as `(false, 0, 0)` to MATCH the source's field-free
/// conversion — the AF/DP fields are synthesized post-hoc onto an already-routed
/// store, so the genotype routing never accounted for their bits; feeding
/// nonzero info/format bits here would flip variants and (correctly) break the
/// byte-parity this test asserts.
#[test]
fn recompute_full_coverage_is_byte_parity() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let samples = ["S0", "S1"];
    build_fixture_store(&src, &samples);
    let fields = synth_fields(&src, samples.len());

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let n = slice_contig(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "chr1",
        &(0..samples.len()).collect::<Vec<_>>(),
        2,
        &[(0, u32::MAX)],
        OverlapMode::Variant,
        &fields,
        Routing::Recompute,
        false, // sidecar_bits_enabled: source was converted field/signature-free
        0,     // info_bits: match the source conversion (fields added post-hoc)
        0,     // format_bits: same
    )
    .unwrap();
    assert_eq!(
        n, 5,
        "all 5 fixture variants must survive a full-coverage Routing::Recompute slice"
    );

    for rel in geno_and_field_parity_rels() {
        let rel = Path::new("chr1").join(rel);
        let rel = rel.to_str().unwrap();
        assert_eq!(
            read_if_exists(&src, rel),
            read_if_exists(&out, rel),
            "{rel}"
        );
    }
}

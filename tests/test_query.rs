//! Disk-integration tests for the `(range, sample)` query. Builds finished SVAR2
//! contigs via the real conversion pipeline and a hand-written `max_del.npy`
//! fixture (until the max_del post-pass producer lands).

mod common;

use common::{SynthRecord, build_bcf_with_index};
use genoray_core::process_chromosome;
use genoray_core::query::ContigReader;
use genoray_core::query::overlap_sample;
use ndarray::{Array1, Array2};
use std::path::Path;
use tempfile::tempdir;

/// Deletion length implied by a record's ref/alt lengths (`max(0, -ilen)`).
fn del_len_of(rec: &SynthRecord) -> u32 {
    let ilen = rec.alts[0].len() as i32 - rec.ref_allele.len() as i32;
    if ilen < 0 { (-ilen) as u32 } else { 0 }
}

/// Write the `max_del` sidecars for a finished contig. Conservative per-column
/// bound: each `(sample, ploid)` column's max over ALL deletions it carries (an
/// over-estimate vs. the var_key-only contract, but `overlap_range`'s overshoot
/// is proven safe — see `search.rs` `overlap_max_region_length_overshoot_is_safe`),
/// and the global max for `dense/max_del`. This exercises per-column indexing in
/// the consumer while remaining independent of the not-yet-landed producer.
fn write_max_del_fixture(
    contig_dir: &Path,
    n_samples: usize,
    ploidy: usize,
    records: &[SynthRecord],
) {
    let columns = n_samples * ploidy;
    let mut per_col = vec![0u32; columns];
    let mut global = 0u32;
    for rec in records {
        let d = del_len_of(rec);
        global = global.max(d);
        for (hap, &g) in rec.gt.iter().enumerate() {
            if g == 1 {
                per_col[hap] = per_col[hap].max(d);
            }
        }
    }
    let arr = Array2::from_shape_vec((n_samples, ploidy), per_col).unwrap();
    ndarray_npy::write_npy(contig_dir.join("max_del.npy"), &arr).unwrap();

    std::fs::create_dir_all(contig_dir.join("dense")).unwrap();
    let dense = Array1::from_vec(vec![global]);
    ndarray_npy::write_npy(contig_dir.join("dense").join("max_del.npy"), &dense).unwrap();
}

/// Convert `records` to a finished SVAR2 contig under `out/{chrom}` and write the
/// `max_del` fixture. `out` must already exist.
fn build_contig(out: &Path, chrom: &str, samples: &[&str], ploidy: usize, records: &[SynthRecord]) {
    let bcf = out.join("in.bcf");
    build_bcf_with_index(&bcf, chrom, 1_000_000, samples, records);
    process_chromosome(
        bcf.to_str().unwrap(),
        chrom,
        out.to_str().unwrap(),
        samples,
        1000, // chunk_size
        ploidy,
        1,    // htslib_threads
        4096, // long_allele_capacity
    )
    .expect("process_chromosome should succeed");
    write_max_del_fixture(&out.join(chrom), samples.len(), ploidy, records);
}

#[test]
fn test_open_ok_and_missing_dirs_tolerated() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 1, 1],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 0, 0],
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    assert!(ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).is_ok());

    // A bare contig dir with no sub-streams at all still opens (everything empty).
    let empty = tmp.path().join("empty_out");
    std::fs::create_dir_all(empty.join("chrX")).unwrap();
    assert!(ContigReader::open(empty.to_str().unwrap(), "chrX", 1, 2).is_ok());
}

// Mirrors the shape of `test_e2e_normalized_bcf_pipeline`: SNP@100 (-> dense/snp,
// x=3), INS@200 (-> var_key/indel, x=1), DEL@300 (-> dense/indel, x=2). The query
// unions across those three sub-streams.
#[test]
fn test_overlap_sample_known_contig() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 1, 1],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![0, 1, 0, 0],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 0, 0],
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);
    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap();

    // Sample 0, whole contig.
    let r = overlap_sample(&reader, 0, 0, 1000);
    assert_eq!(r.per_hap.len(), 2);
    // hap 0 (S0_p0): SNP@100 (gt 1) + DEL@300 (gt 1); INS@200 gt 0.
    assert_eq!(r.per_hap[0].positions, vec![100, 300]);
    assert_eq!(r.per_hap[0].ilens, vec![0, -1]);
    assert_eq!(r.per_hap[0].alts, vec![b"C".to_vec(), Vec::<u8>::new()]);
    // hap 1 (S0_p1): INS@200 (gt 1) + DEL@300 (gt 1); SNP@100 gt 0.
    assert_eq!(r.per_hap[1].positions, vec![200, 300]);
    assert_eq!(r.per_hap[1].ilens, vec![1, -1]);
    assert_eq!(r.per_hap[1].alts, vec![b"AT".to_vec(), Vec::<u8>::new()]);

    // Sample 1: only the SNP@100 (gt for haps 2,3 = 1,1).
    let r1 = overlap_sample(&reader, 1, 0, 1000);
    assert_eq!(r1.per_hap[0].positions, vec![100]);
    assert_eq!(r1.per_hap[1].positions, vec![100]);

    // Deletion spanning the query start: [301, 302) still returns DEL@300
    // (v_end = 300 + 1 + 1 = 302). Exercises the dense/indel max_del path.
    let r2 = overlap_sample(&reader, 0, 301, 302);
    assert_eq!(r2.per_hap[0].positions, vec![300]);
    assert_eq!(r2.per_hap[1].positions, vec![300]);

    // No overlap: gap between variants.
    let r3 = overlap_sample(&reader, 0, 150, 160);
    assert!(r3.per_hap[0].positions.is_empty());
    assert!(r3.per_hap[1].positions.is_empty());
}

// Regression guard for the var_key/snp decode path: `overlap_sample` must unpack
// the 2-bit SNP code at the ABSOLUTE index `o0 + i` into the whole-stream packed
// keys buffer, not the local (per-column) index `i`. A fixture where every
// var_key/snp column happens to start at CSR offset 0 (as in
// `test_overlap_sample_known_contig`) can't catch a regression to local `i` --
// this test forces a nonzero offset for the second column.
//
// Two singleton-carrier (x=1) SNPs always route to var_key/snp regardless of
// cohort size (see `cost_model::choose_representation`: var_key_bits = 34 <=
// dense_bits = 34+np for every np >= 0). SNP_A@100 (ALT C, code 1) is carried
// only by S0's hap p0 (flat col 0); SNP_B@200 (ALT G, code 3) only by S0's hap
// p1 (flat col 1). The sample-major transpose fills var_key/snp column-by-
// column, so col 0's CSR range is [0,1) and col 1's is [1,2): querying hap p1
// unpacks the packed-key buffer at absolute index `o0+i = 1+0 = 1`. Under a
// regression to local `i` it would instead read index 0 -- col 0's code (`C`)
// -- and decode SNP_B's ALT as `C` instead of `G`.
#[test]
fn test_overlap_sample_var_key_snp_nonzero_csr_offset() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let samples = ["S0"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0], // hap p0 only
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![0, 1], // hap p1 only
        },
    ];
    build_contig(&out, "chr1", &samples, 2, &records);

    // Confirm both SNPs actually routed to var_key/snp (not dense/snp): the
    // var_key/snp positions sidecar must hold both calls, and dense/snp must be
    // absent/empty.
    let vk_snp_positions = out
        .join("chr1")
        .join("var_key")
        .join("snp")
        .join("positions.bin");
    let vk_positions = common::read_u32_bin(&vk_snp_positions);
    assert_eq!(
        vk_positions,
        vec![100, 200],
        "expected both SNPs to route to var_key/snp"
    );
    let dense_snp_positions = out
        .join("chr1")
        .join("dense")
        .join("snp")
        .join("positions.bin");
    assert!(
        !dense_snp_positions.exists()
            || std::fs::metadata(&dense_snp_positions).unwrap().len() == 0,
        "expected dense/snp to be empty"
    );

    // Sanity-check the CSR offsets directly: col 0 -> [0,1), col 1 -> [1,2).
    let offsets = common::read_offsets_npy(
        &out.join("chr1")
            .join("var_key")
            .join("snp")
            .join("offsets.npy"),
    );
    assert_eq!(offsets, vec![0, 1, 2]);

    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 1, 2).unwrap();
    let r = overlap_sample(&reader, 0, 0, 1000);
    assert_eq!(r.per_hap.len(), 2);

    // hap p0 (col 0, o0=0): SNP_A@100, ALT C.
    assert_eq!(r.per_hap[0].positions, vec![100]);
    assert_eq!(r.per_hap[0].alts, vec![b"C".to_vec()]);

    // hap p1 (col 1, o0=1): SNP_B@200, ALT G. This is the assertion the
    // local-`i` regression breaks: it would decode col 0's code and return `C`.
    assert_eq!(r.per_hap[1].positions, vec![200]);
    assert_eq!(r.per_hap[1].alts, vec![b"G".to_vec()]);
}

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

use common::{SynthRecord, build_contig};
use genoray_core::meta::{FORMAT_VERSION, write_meta};
use genoray_core::query::{ContigReader, oracle::overlap_sample};
use genoray_core::svar2_slice::slice_contig_genos;
use genoray_core::svar2_source::OverlapMode;
use std::path::Path;
use tempfile::tempdir;

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

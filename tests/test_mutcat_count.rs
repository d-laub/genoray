//! Disk-integration test for Task 9's `count_contig`: a real `ContigReader`
//! over a `build_contig` fixture, with an injected `VkSnp` mutcat sidecar
//! (count_contig reads sidecar codes/refs, never the FASTA, so the sidecar
//! fully controls classification) driving an isolated-adjacent-SNV DBS pair.

mod common;

use common::{SynthRecord, build_contig, read_u32_bin};
use genoray_core::layout::{ContigPaths, MutcatSub};
use genoray_core::mutcat::count::{Sidecars, count_contig};
use genoray_core::mutcat::sidecar::write_sidecar;
use genoray_core::mutcat::{DBS78_OFFSET, N_CODES, UNCLASSIFIED};
use genoray_core::query::ContigReader;
use ndarray::{Array2, s};
use svar2_codec::encode_snp_2bit;
use tempfile::tempdir;

#[test]
fn count_contig_pairs_adjacent_snvs_for_one_sample() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    // 1 sample, ploidy 1: at np=1 the cost model (`cost_model::choose_representation`)
    // strictly prefers var_key over dense for a singly-carried SNP (34 bits vs 35),
    // so both records land in var_key/snp, not dense/snp.
    let samples = ["S0"];
    let records = vec![
        SynthRecord {
            pos: 10,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1],
        },
        SynthRecord {
            pos: 11,
            ref_allele: b"C",
            alts: vec![&b"A"[..]],
            gt: vec![1],
        },
    ];
    build_contig(&out, "chr1", &samples, 1, &records);

    // Confirm routing + absolute index order before trusting the injected
    // sidecar's alignment: both SNVs in var_key/snp, position-sorted, and
    // nothing in dense/snp.
    let vk_positions = read_u32_bin(
        &out.join("chr1")
            .join("var_key")
            .join("snp")
            .join("positions.bin"),
    );
    assert_eq!(
        vk_positions,
        vec![10, 11],
        "expected both SNVs in var_key/snp, in order"
    );
    let dense_snp_positions_path = out
        .join("chr1")
        .join("dense")
        .join("snp")
        .join("positions.bin");
    assert!(
        !dense_snp_positions_path.exists() || read_u32_bin(&dense_snp_positions_path).is_empty(),
        "expected no SNVs routed to dense/snp"
    );

    // Inject a VkSnp sidecar aligned to that absolute index order: record 0
    // (pos 10) REF=A, record 1 (pos 11) REF=C. The real ALT alleles built
    // above (C then A) come from var_key's packed key stream, so the doublet
    // is REF "AC" -> ALT "CA": DBS78 label "AC>CA" is index 0 (see
    // src/mutcat/classify.rs's DBS78_LABELS[0] and its `dbs78_literal_hit`
    // test). The per-record SBS code is irrelevant here since a successfully
    // paired doublet never falls back to it — set to UNCLASSIFIED.
    let paths = ContigPaths::new(out.to_str().unwrap(), "chr1");
    let codes = [UNCLASSIFIED, UNCLASSIFIED];
    let refs = [encode_snp_2bit(b'A'), encode_snp_2bit(b'C')];
    write_sidecar(&paths, MutcatSub::VkSnp, &codes, Some(&refs)).unwrap();

    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", 1, 1).unwrap();
    let sidecars = Sidecars::open(&paths).unwrap();

    let mut acc = Array2::<i64>::zeros((1, N_CODES));
    count_contig(&reader, &sidecars, false, &mut acc);

    // Count-once fix (Task 8, commit 1bd79cd): an isolated adjacent pair
    // contributes exactly ONE DBS call, not one per member.
    assert_eq!(
        acc[[0, DBS78_OFFSET]],
        1,
        "expected exactly one DBS78 index-0 call"
    );
    assert_eq!(
        acc.slice(s![0, 0..96]).sum(),
        0,
        "the two SNVs that formed the doublet must not also show up as SBS96 calls"
    );
}

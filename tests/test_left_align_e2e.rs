// End-to-end left-alignment: feed un-left-aligned indels against a repeat-containing
// reference and assert the reader emits atoms left-aligned to the positions bcftools
// produces (`bcftools norm -a -m- -f`).
mod common;

use common::{SynthRecord, build_bcf_with_index};
use genoray_core::chunk_assembler::ChunkAssembler;
use genoray_core::vcf_reader::VcfRecordSource;
use proptest::prelude::*;
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

// Reference: 0:C 1:A 2:A 3:A 4:A 5:T 6:C 7:A 8:G 9:A 10:G 11:T
const REF_SEQ: &[u8] = b"CAAAATCAGAGT";

fn write_ref(fasta_path: &Path, chrom: &str, seq: &[u8]) {
    {
        let mut f = std::fs::File::create(fasta_path).unwrap();
        writeln!(f, ">{}", chrom).unwrap();
        f.write_all(seq).unwrap();
        writeln!(f).unwrap();
    }
    rust_htslib::faidx::build(fasta_path).unwrap();
}

fn drain(bcf: &Path, fasta: &Path, chrom: &str, samples: &[&str]) -> Vec<(u32, i32)> {
    let source = VcfRecordSource::new(
        bcf.to_str().unwrap(),
        chrom,
        samples,
        1,
        2,
        &[],
        Vec::new(),
        genoray_core::svar2_view::OverlapMode::Pos,
    )
    .unwrap();
    let mut reader = ChunkAssembler::new(
        Box::new(source),
        samples.len(),
        2,
        Some(fasta.to_str().unwrap()),
        chrom,
        false,
        genoray_core::normalize::CheckRef::Error,
        &[],
    )
    .unwrap();
    let mut out = Vec::new();
    let mut cid = 0;
    while let Some(chunk) = reader.read_next_chunk(100, cid, None).unwrap() {
        for i in 0..chunk.pos.len() {
            out.push((chunk.pos[i], chunk.ilens[i]));
        }
        cid += 1;
    }
    out
}

#[test]
fn indels_left_align_to_bcftools_positions() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("la.bcf");
    let fasta = tmp.path().join("la.fa");
    write_ref(&fasta, "chr1", REF_SEQ);

    let samples = vec!["S0"];
    // AA>A @3 (del one A), A>AA @3 (ins one A), GAG>G @8 (del "AG").
    let records = vec![
        SynthRecord {
            pos: 3,
            ref_allele: b"AA",
            alts: vec![&b"A"[..]],
            gt: vec![1, 0],
        },
        SynthRecord {
            pos: 3,
            ref_allele: b"A",
            alts: vec![&b"AA"[..]],
            gt: vec![0, 1],
        },
        SynthRecord {
            pos: 8,
            ref_allele: b"GAG",
            alts: vec![&b"G"[..]],
            gt: vec![1, 1],
        },
    ];
    build_bcf_with_index(&bcf, "chr1", 12, &samples, &records);

    let atoms = drain(&bcf, &fasta, "chr1", &samples);
    // bcftools norm -a -m- -f: DEL→pos0 ilen-1, INS→pos0 ilen+1, DEL→pos6 ilen-2.
    // Emission is globally position-sorted, so pos 0 atoms come first (order between the
    // two pos-0 atoms is the reorder heap's stable seq tiebreak: del then ins).
    let mut sorted = atoms.clone();
    sorted.sort();
    assert_eq!(atoms, sorted, "emitted positions must be globally sorted");
    assert!(atoms.contains(&(0, -1)), "homopolymer DEL → pos 0");
    assert!(atoms.contains(&(0, 1)), "homopolymer INS → pos 0");
    assert!(atoms.contains(&(6, -2)), "AGAG DEL → pos 6");
}

#[test]
fn ref_mismatch_panics() {
    // FASTA says pos 3 is 'A' but the record claims REF "GG" → validate_ref fails,
    // read_next_chunk returns Err, and `drain`'s `.unwrap()` turns that into a panic.
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("mm.bcf");
    let fasta = tmp.path().join("mm.fa");
    write_ref(&fasta, "chr1", REF_SEQ);

    let samples = vec!["S0"];
    let records = vec![SynthRecord {
        pos: 3,
        ref_allele: b"GG",
        alts: vec![&b"G"[..]],
        gt: vec![1, 0],
    }];
    build_bcf_with_index(&bcf, "chr1", 12, &samples, &records);

    let res = std::panic::catch_unwind(|| drain(&bcf, &fasta, "chr1", &samples));
    assert!(res.is_err(), "REF/FASTA mismatch must panic");
}

use std::process::Command;

// Build a plain-text VCF (bgzip-free is fine for `bcftools norm`, which reads text VCF).
fn write_vcf(path: &Path, chrom: &str, chrom_len: usize, records: &[(u32, &[u8], &[u8])]) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "##fileformat=VCFv4.2").unwrap();
    writeln!(f, "##contig=<ID={},length={}>", chrom, chrom_len).unwrap();
    writeln!(
        f,
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
    )
    .unwrap();
    writeln!(
        f,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0"
    )
    .unwrap();
    for &(pos0, r, a) in records {
        // VCF POS is 1-based.
        writeln!(
            f,
            "{}\t{}\t.\t{}\t{}\t.\t.\t.\tGT\t1|0",
            chrom,
            pos0 + 1,
            std::str::from_utf8(r).unwrap(),
            std::str::from_utf8(a).unwrap()
        )
        .unwrap();
    }
}

#[test]
fn reader_matches_bcftools_norm() {
    let tmp = tempdir().unwrap();
    let fasta = tmp.path().join("x.fa");
    write_ref(&fasta, "chr1", REF_SEQ);
    let vcf = tmp.path().join("x.vcf");
    let records: &[(u32, &[u8], &[u8])] = &[(3, b"AA", b"A"), (3, b"A", b"AA"), (8, b"GAG", b"G")];
    write_vcf(&vcf, "chr1", REF_SEQ.len(), records);

    let out = Command::new("bcftools")
        .args(["norm", "-a", "-m-", "-f"])
        .arg(&fasta)
        .arg(&vcf)
        .output()
        .expect("bcftools must be on PATH (run under `pixi run -e default`)");
    assert!(out.status.success(), "bcftools norm failed: {:?}", out);
    // Parse (pos0, ilen) from bcftools' left-aligned biallelic records.
    let mut expected: Vec<(u32, i32)> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            let pos0 = c[1].parse::<u32>().unwrap() - 1;
            let ilen = c[4].len() as i32 - c[3].len() as i32;
            (pos0, ilen)
        })
        .collect();
    expected.sort();

    // Drive the reader over an equivalent BCF.
    let bcf = tmp.path().join("x.bcf");
    let samples = vec!["S0"];
    let synth = vec![
        SynthRecord {
            pos: 3,
            ref_allele: b"AA",
            alts: vec![&b"A"[..]],
            gt: vec![1, 0],
        },
        SynthRecord {
            pos: 3,
            ref_allele: b"A",
            alts: vec![&b"AA"[..]],
            gt: vec![1, 0],
        },
        SynthRecord {
            pos: 8,
            ref_allele: b"GAG",
            alts: vec![&b"G"[..]],
            gt: vec![1, 0],
        },
    ];
    build_bcf_with_index(&bcf, "chr1", REF_SEQ.len() as u64, &samples, &synth);
    let mut got = drain(&bcf, &fasta, "chr1", &samples);
    got.sort();

    assert_eq!(
        got, expected,
        "reader atoms must match `bcftools norm -a -m-`"
    );
}

#[test]
fn left_shifts_stay_sorted_across_chunk_boundaries() {
    // A later record's indel left-aligns to a position *below* an earlier-emitted record's
    // start. With chunk_size = 1, ordering must still hold across chunks.
    // Reference: index 6:C 7:A 8:G 9:A 10:G 11:T — a SNP at 7, then GAG>G @8 which rolls
    // to pos 6. The reorder buffer (widened by L_MAX) must emit pos 6 before pos 7.
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("order.bcf");
    let fasta = tmp.path().join("order.fa");
    write_ref(&fasta, "chr1", REF_SEQ);

    let samples = vec!["S0"];
    let records = vec![
        SynthRecord {
            pos: 7,
            ref_allele: b"A",
            alts: vec![&b"T"[..]],
            gt: vec![1, 0],
        },
        SynthRecord {
            pos: 8,
            ref_allele: b"GAG",
            alts: vec![&b"G"[..]],
            gt: vec![1, 0],
        },
    ];
    build_bcf_with_index(&bcf, "chr1", REF_SEQ.len() as u64, &samples, &records);

    // chunk_size = 1 forces every atom into its own chunk.
    let source = VcfRecordSource::new(
        bcf.to_str().unwrap(),
        "chr1",
        &samples,
        1,
        2,
        &[],
        Vec::new(),
        genoray_core::svar2_view::OverlapMode::Pos,
    )
    .unwrap();
    let mut reader = ChunkAssembler::new(
        Box::new(source),
        samples.len(),
        2,
        Some(fasta.to_str().unwrap()),
        "chr1",
        false,
        genoray_core::normalize::CheckRef::Error,
        &[],
    )
    .unwrap();
    let mut positions = Vec::new();
    let mut cid = 0;
    while let Some(chunk) = reader.read_next_chunk(1, cid, None).unwrap() {
        positions.extend_from_slice(&chunk.pos);
        cid += 1;
    }
    // GAG>G rolls to pos 6, SNP stays at 7 → sorted [6, 7].
    assert_eq!(positions, vec![6, 7]);
    let mut sorted = positions.clone();
    sorted.sort();
    assert_eq!(
        positions, sorted,
        "emitted positions must be non-decreasing"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // Random homopolymer/repeat reference + random indels: whatever the reader emits is
    // globally position-sorted, even with left-shifts and chunk_size = 1.
    #[test]
    fn emitted_positions_are_sorted_under_random_left_shifts(
        seed in proptest::collection::vec(
            prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T')], 20..60),
        anchors in proptest::collection::vec(2usize..18, 1..6),
    ) {
        let tmp = tempdir().unwrap();
        let bcf = tmp.path().join("p.bcf");
        let fasta = tmp.path().join("p.fa");
        write_ref(&fasta, "chr1", &seed);

        // Build clean single-base deletions at sorted, distinct anchor positions (each
        // deletes the base to its right, so REF is [ref, ref+1]).
        let mut sorted_anchors: Vec<usize> = anchors.clone();
        sorted_anchors.sort_unstable();
        sorted_anchors.dedup();
        let refs: Vec<(usize, Vec<u8>, Vec<u8>)> = sorted_anchors
            .iter()
            .filter(|&&p| p + 1 < seed.len())
            .map(|&p| (p, vec![seed[p], seed[p + 1]], vec![seed[p]]))
            .collect();
        prop_assume!(!refs.is_empty());

        let samples = vec!["S0"];
        let synth: Vec<SynthRecord> = refs
            .iter()
            .map(|(p, r, a)| SynthRecord {
                pos: *p as i64,
                ref_allele: r.as_slice(),
                alts: vec![a.as_slice()],
                gt: vec![1, 0],
            })
            .collect();
        build_bcf_with_index(&bcf, "chr1", seed.len() as u64, &samples, &synth);

        let source = VcfRecordSource::new(
            bcf.to_str().unwrap(),
            "chr1",
            &samples,
            1,
            2,
            &[],
            Vec::new(),
            genoray_core::svar2_view::OverlapMode::Pos,
        )
        .unwrap();
        let mut reader = ChunkAssembler::new(
            Box::new(source),
            samples.len(),
            2,
            Some(fasta.to_str().unwrap()),
            "chr1",
            false,
            genoray_core::normalize::CheckRef::Error,
            &[],
        )
        .unwrap();
        let mut positions = Vec::new();
        let mut cid = 0;
        while let Some(chunk) = reader.read_next_chunk(1, cid, None).unwrap() {
            positions.extend_from_slice(&chunk.pos);
            cid += 1;
        }
        let mut sorted = positions.clone();
        sorted.sort();
        prop_assert_eq!(positions, sorted);
    }
}

// End-to-end atomization/normalization: feed un-normalized records and assert the
// reader emits correctly split, atomized, and globally position-sorted DenseChunks.
mod common;

use common::{SynthRecord, build_bcf_with_index, build_fasta_with_index};
use genoray_core::chunk_assembler::ChunkAssembler;
use genoray_core::vcf_reader::VcfRecordSource;
use std::path::Path;
use tempfile::tempdir;

// Collect every atom the reader emits across all chunks, as (pos, ilen) plus the
// per-column presence bits, in emission order.
fn drain_reader(
    bcf_path: &Path,
    chrom: &str,
    samples: &[&str],
    ploidy: usize,
    chunk_size: usize,
) -> Vec<(u32, i32, Vec<bool>)> {
    // Each test builds `bcf_path.with_extension("fa")` from its records *before* calling
    // drain_reader (see the per-test edit below); here we just point the reader at it.
    let fasta_path = bcf_path.with_extension("fa");
    let source =
        VcfRecordSource::new(bcf_path.to_str().unwrap(), chrom, samples, 1, ploidy, &[]).unwrap();
    let mut reader = ChunkAssembler::new(
        Box::new(source),
        samples.len(),
        ploidy,
        Some(fasta_path.to_str().unwrap()),
        chrom,
        false,
        genoray_core::normalize::CheckRef::Error,
        &[],
    )
    .unwrap();
    let columns = samples.len() * ploidy;
    let mut out = Vec::new();
    let mut chunk_id = 0;
    while let Some(chunk) = reader.read_next_chunk(chunk_size, chunk_id, None).unwrap() {
        let v = chunk.pos.len();
        for i in 0..v {
            let mut presence = Vec::with_capacity(columns);
            for col in 0..columns {
                presence.push(chunk.genos.get_bit(i * columns + col));
            }
            out.push((chunk.pos[i], chunk.ilens[i], presence));
        }
        chunk_id += 1;
    }
    out
}

#[test]
fn multiallelic_site_splits_and_remaps_genotypes() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("multi.bcf");
    let samples = vec!["S0"];
    // One diploid sample, genotype 1|2 at a 2-ALT site A>C,G.
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"A",
        alts: vec![&b"C"[..], &b"G"[..]],
        gt: vec![1, 2],
    }];
    build_bcf_with_index(&bcf, "chr1", 10_000, &samples, &records);
    build_fasta_with_index(&bcf.with_extension("fa"), "chr1", 10_000, &records);

    let atoms = drain_reader(&bcf, "chr1", &samples, 2, 100);
    // Two SNP atoms at pos 100: ALT C carried on hap0 only, ALT G on hap1 only.
    assert_eq!(atoms.len(), 2);
    assert_eq!((atoms[0].0, atoms[0].1), (100, 0));
    assert_eq!((atoms[1].0, atoms[1].1), (100, 0));
    assert_eq!(atoms[0].2, vec![true, false]); // source ALT 1 (C) → hap0
    assert_eq!(atoms[1].2, vec![false, true]); // source ALT 2 (G) → hap1
}

#[test]
fn mnp_atomizes_to_snps_shared_presence() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("mnp.bcf");
    let samples = vec!["S0"];
    // AC>GT MNP, sample homozygous for the ALT on both haps.
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"AC",
        alts: vec![&b"GT"[..]],
        gt: vec![1, 1],
    }];
    build_bcf_with_index(&bcf, "chr1", 10_000, &samples, &records);
    build_fasta_with_index(&bcf.with_extension("fa"), "chr1", 10_000, &records);

    let atoms = drain_reader(&bcf, "chr1", &samples, 2, 100);
    // Two SNP atoms (A>G@100, C>T@101), both carried on both haps.
    assert_eq!(atoms.len(), 2);
    assert_eq!((atoms[0].0, atoms[0].1), (100, 0));
    assert_eq!((atoms[1].0, atoms[1].1), (101, 0));
    assert_eq!(atoms[0].2, vec![true, true]);
    assert_eq!(atoms[1].2, vec![true, true]);
}

#[test]
fn atoms_are_globally_position_sorted_across_records() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("sorted.bcf");
    let samples = vec!["S0"];
    // An MNP at 100 spans to 104; a SNP record starts at 102 — its atom would land
    // between the MNP's atoms unless the reader reorders. Also force small chunks so
    // the ordering must hold ACROSS chunk boundaries.
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"ACGTA",
            alts: vec![&b"GCGTG"[..]],
            gt: vec![1, 1],
        },
        SynthRecord {
            pos: 102,
            ref_allele: b"G",
            alts: vec![&b"T"[..]],
            gt: vec![1, 0],
        },
    ];
    build_bcf_with_index(&bcf, "chr1", 10_000, &samples, &records);
    build_fasta_with_index(&bcf.with_extension("fa"), "chr1", 10_000, &records);

    // chunk_size = 1 → every atom lands in its own chunk; emission order must still
    // be globally sorted.
    let atoms = drain_reader(&bcf, "chr1", &samples, 2, 1);
    let positions: Vec<u32> = atoms.iter().map(|a| a.0).collect();
    // MNP → A>G@100, A>G@104; SNP → T@102. Sorted: 100, 102, 104.
    assert_eq!(positions, vec![100, 102, 104]);
    let mut sorted = positions.clone();
    sorted.sort();
    assert_eq!(
        positions, sorted,
        "emitted positions must be globally sorted"
    );
}

#[test]
fn complex_deletion_with_substituted_anchor() {
    let tmp = tempdir().unwrap();
    let bcf = tmp.path().join("complex.bcf");
    let samples = vec!["S0"];
    // ATG>CG (previously rejected as "complex"): → SNV(A>C)@100 + DEL(ilen=-1)@100.
    let records = vec![SynthRecord {
        pos: 100,
        ref_allele: b"ATG",
        alts: vec![&b"CG"[..]],
        gt: vec![1, 1],
    }];
    build_bcf_with_index(&bcf, "chr1", 10_000, &samples, &records);
    build_fasta_with_index(&bcf.with_extension("fa"), "chr1", 10_000, &records);

    let atoms = drain_reader(&bcf, "chr1", &samples, 2, 100);
    assert_eq!(atoms.len(), 2);
    assert_eq!((atoms[0].0, atoms[0].1), (100, 0)); // SNV A>C
    assert_eq!((atoms[1].0, atoms[1].1), (100, -1)); // DEL
    assert_eq!(atoms[0].2, vec![true, true]);
    assert_eq!(atoms[1].2, vec![true, true]);
}

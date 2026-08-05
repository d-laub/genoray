mod common;

use common::{SynthRecord, build_bcf_with_index, build_fasta_with_index};
use genoray_core::logging::{Event, EventSink};
use genoray_core::normalize::CheckRef;
use genoray_core::orchestrator::{SourceSpec, process_chromosome};
use tempfile::TempDir;

// Three records; the middle one's REF ("G") will disagree with the FASTA
// (which carries "T" at that position), reproducing issue #116.
fn bcf_records() -> Vec<SynthRecord<'static>> {
    vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"G",
            alts: vec![&b"GTTT"[..]],
            gt: vec![1, 1, 0, 0],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"T",
            alts: vec![&b"A"[..]],
            gt: vec![0, 0, 1, 0],
        },
    ]
}

// Same loci, but the FASTA truth at pos 200 is "T" (not the record's "G").
fn fasta_records() -> Vec<SynthRecord<'static>> {
    vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![],
            gt: vec![],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"T",
            alts: vec![],
            gt: vec![],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"T",
            alts: vec![],
            gt: vec![],
        },
    ]
}

fn convert(
    out: &std::path::Path,
    check_ref: CheckRef,
) -> Result<u64, genoray_core::error::ConversionError> {
    let tmp = out.parent().unwrap();
    let bcf = tmp.join("in.bcf");
    let fasta = tmp.join("in.fa");
    let samples = ["S0", "S1"];
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &bcf_records());
    build_fasta_with_index(&fasta, "chr1", 1000, &fasta_records());
    let sample_refs: Vec<&str> = samples.to_vec();
    process_chromosome(
        SourceSpec::Vcf {
            vcf_path: bcf.to_str().unwrap().to_string(),
            htslib_threads: 1,
            reader_workers: 1,
            regions: Vec::new(),
            overlap: genoray_core::svar2_view::OverlapMode::Pos,
        },
        Some(fasta.to_str().unwrap()),
        "chr1",
        out.to_str().unwrap(),
        &sample_refs,
        25_000,
        2,
        8 * 1024 * 1024,
        false,     // skip_out_of_scope
        check_ref, // NEW
        1,         // processing_threads
        false,     // signatures
        &[],       // fields
        &genoray_core::logging::EventSink::disabled(),
    )
}

#[test]
fn ref_mismatch_errors_under_e() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    assert!(
        convert(&out, CheckRef::Error).is_err(),
        "check_ref=e must abort on a REF mismatch"
    );
}

#[test]
fn ref_mismatch_excluded_under_x() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    // Succeeds; the two clean SNPs (pos 100, 300) are still written.
    convert(&out, CheckRef::Exclude).unwrap();
    assert!(
        out.join("chr1/var_key/snp/positions.bin").exists(),
        "clean SNPs still converted"
    );
}

use common::build_bcf_with_index as build_single;

// Two single-sample files; file B carries a REF at pos 200 that disagrees with
// the FASTA (which says "T"). Under x, B's bad record is dropped; the merge
// still produces a store.
#[test]
fn vcf_list_ref_mismatch_excluded_under_x() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    let fasta = tmp.path().join("in.fa");
    build_fasta_with_index(&fasta, "chr1", 1000, &fasta_records());

    let a = tmp.path().join("a.bcf");
    let b = tmp.path().join("b.bcf");
    build_single(
        &a,
        "chr1",
        1000,
        &["A"],
        &[SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0],
        }],
    );
    build_single(
        &b,
        "chr1",
        1000,
        &["B"],
        &[SynthRecord {
            pos: 200,
            ref_allele: b"G",
            alts: vec![&b"GTTT"[..]],
            gt: vec![1, 0],
        }],
    );

    let dropped = genoray_core::orchestrator::run_vcf_list(
        &[
            a.to_str().unwrap().to_string(),
            b.to_str().unwrap().to_string(),
        ],
        Some(fasta.to_str().unwrap()),
        &["chr1".to_string()],
        out.to_str().unwrap(),
        &["A".to_string(), "B".to_string()],
        25_000,
        2,
        Some(1),
        8 * 1024 * 1024,
        false,                                      // skip_out_of_scope
        CheckRef::Exclude,                          // NEW
        false,                                      // signatures
        Vec::new(),                                 // info_fields
        Vec::new(),                                 // format_fields
        Vec::new(),                                 // region_ranges
        genoray_core::svar2_view::OverlapMode::Pos, // overlap
        vec![vec![true; 2]; 1],                     // contig_membership: both files carry chr1
        &genoray_core::logging::EventSink::disabled(),
    )
    .unwrap();
    assert_eq!(
        dropped, 0,
        "no out-of-scope ALTs; the REF-mismatch is excluded, not counted here"
    );
    assert!(
        out.join("meta.json").exists(),
        "merge completed despite the bad record"
    );
}

// Restores, at the Rust level, the guarantee a removed Python stdout
// assertion (`test_from_vcf_sharded_check_ref_exclude_counts_owned_record_once`)
// used to check: a ref-excluded record is counted exactly ONCE in the
// per-contig summary even when sub-contig sharding causes multiple shards'
// padded fetch windows to see it. That summary is now `ContigDone.excluded`,
// a real `EventSink` field (Task 5), so drive `process_chromosome` with a
// live sink and assert on the emitted event instead of captured stdout.
//
// `process_chromosome`'s sharded path (`vcf_reader::plan_vcf_shards` inside
// `orchestrator.rs`) only activates when `regions` is non-empty (an empty
// region list disables sharding — see the Python `from_vcf` comment on why
// it always fills `[0, len)` for whole-contig conversion) and
// `overlap == OverlapMode::Pos`. Passing the whole-contig range explicitly,
// a small `chunk_size` (target shard span), and `reader_workers > 1`
// reproduces the Python test's sharded scenario (there: `threads=16,
// chunk_size=1`) directly against the Rust entry point.
#[test]
fn sharded_ref_excluded_counted_once_in_contig_done() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    let bcf = tmp.path().join("in.bcf");
    let fasta = tmp.path().join("in.fa");
    let samples = ["S0", "S1"];
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &bcf_records());
    build_fasta_with_index(&fasta, "chr1", 1000, &fasta_records());
    let sample_refs: Vec<&str> = samples.to_vec();

    let (tx, rx) = crossbeam_channel::unbounded();
    let sink = EventSink::new(tx, 1);

    process_chromosome(
        SourceSpec::Vcf {
            vcf_path: bcf.to_str().unwrap().to_string(),
            htslib_threads: 1,
            reader_workers: 8,
            // Non-empty, whole-contig range: required to enable sub-contig
            // sharding (see comment above).
            regions: vec![(0, 1000)],
            overlap: genoray_core::svar2_view::OverlapMode::Pos,
        },
        Some(fasta.to_str().unwrap()),
        "chr1",
        out.to_str().unwrap(),
        &sample_refs,
        1, // chunk_size (target shard span, bp) -- tiny to force many shards
        2, // ploidy
        8 * 1024 * 1024,
        false,             // skip_out_of_scope
        CheckRef::Exclude, // the pos-200 REF mismatch is dropped, not an error
        8,                 // processing_threads -- >1 so shards.len() > 1
        false,             // signatures
        &[],               // fields
        &sink,
    )
    .expect("process_chromosome should succeed despite the excluded record");

    let events: Vec<Event> = rx.try_iter().collect();
    let contig_done: Vec<&Event> = events
        .iter()
        .filter(|e| matches!(e, Event::ContigDone { .. }))
        .collect();
    assert_eq!(
        contig_done.len(),
        1,
        "expected exactly one ContigDone event, got {:?}",
        contig_done
    );
    match contig_done[0] {
        Event::ContigDone { excluded, .. } => {
            assert_eq!(
                *excluded, 1,
                "the single ref-mismatched record must be counted exactly \
                 once, even though the sharded path's padded fetch windows \
                 let multiple shards see it"
            );
        }
        _ => unreachable!(),
    }
}

/// Four distinct per-sample genotypes at each of three loci, so any mix-up of
/// the sample→header-column mapping changes the output bytes. `gt` is flat
/// (sample-major, ploidy 2).
fn permuted_subset_records() -> Vec<SynthRecord<'static>> {
    vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 1, 0, 1, 0, 0, 1, 0],
        },
        SynthRecord {
            pos: 500,
            ref_allele: b"T",
            alts: vec![&b"G"[..]],
            gt: vec![0, 0, 1, 0, 1, 1, 0, 1],
        },
        SynthRecord {
            pos: 900,
            ref_allele: b"C",
            alts: vec![&b"A"[..]],
            gt: vec![0, 1, 1, 1, 0, 0, 1, 1],
        },
    ]
}

/// Every regular file under `root`, as (relative path, bytes), sorted — enough
/// to compare two output stores byte-for-byte without knowing the layout.
fn store_bytes(root: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    fn walk(dir: &std::path::Path, base: &std::path::Path, out: &mut Vec<(String, Vec<u8>)>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .expect("read_dir")
            .map(|e| e.expect("dir entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, base, out);
            } else {
                let rel = path
                    .strip_prefix(base)
                    .expect("strip_prefix")
                    .to_string_lossy()
                    .into_owned();
                out.push((rel, std::fs::read(&path).expect("read file")));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out
}

/// Sub-contig sharding must not change WHICH header column each requested
/// sample maps to.
///
/// The orchestrator resolves the cohort's header-column indices ONCE per
/// contig and hands the resolved vector to every shard, rather than re-running
/// the name lookup inside the per-shard closure (that cost O(reader_workers x
/// S) header parses, which dominated phase-1 CPU at large cohorts). A hoist
/// like that is only safe if the resolved mapping is genuinely shard-invariant,
/// so pin it: request a REORDERED SUBSET -- `["S2", "S0", "S3"]` out of header
/// order `["S0", "S1", "S2", "S3"]`, skipping S1 -- and require the sharded
/// output to be byte-identical to the unsharded output.
///
/// The existing sharded test passes the full cohort in header order, where the
/// mapping is the identity and any resolution bug is invisible.
#[test]
fn sharded_permuted_sample_subset_matches_unsharded_bytes() {
    let tmp = TempDir::new().unwrap();
    let bcf = tmp.path().join("in.bcf");
    let samples = ["S0", "S1", "S2", "S3"];
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &permuted_subset_records());

    // Reordered and non-contiguous: header columns 2, 0, 3 in that order.
    let requested: Vec<&str> = vec!["S2", "S0", "S3"];

    let run = |out: &std::path::Path, reader_workers: usize, regions: Vec<(u32, u32)>| {
        process_chromosome(
            SourceSpec::Vcf {
                vcf_path: bcf.to_str().unwrap().to_string(),
                htslib_threads: 1,
                reader_workers,
                regions,
                overlap: genoray_core::svar2_view::OverlapMode::Pos,
            },
            None, // no reference: this test is about sample mapping only
            "chr1",
            out.to_str().unwrap(),
            &requested,
            1, // chunk_size (target shard span, bp) -- tiny to force many shards
            2, // ploidy
            8 * 1024 * 1024,
            false,
            // Inert here: with no FASTA there is nothing to check REF against,
            // so neither policy can drop a record. `Error` keeps the test
            // honest -- if a check somehow did run and disagree, it fails loudly
            // instead of silently excluding the record from both runs.
            CheckRef::Error,
            8,
            false,
            &[],
            &EventSink::disabled(),
        )
        .expect("process_chromosome should succeed");
    };

    let sharded = tmp.path().join("sharded");
    let unsharded = tmp.path().join("unsharded");
    // Non-empty regions + OverlapMode::Pos + workers>1 enables sub-contig
    // sharding; an empty region list disables it (see the sharded test above).
    run(&sharded, 8, vec![(0, 1000)]);
    run(&unsharded, 1, Vec::new());

    let a = store_bytes(&sharded);
    let b = store_bytes(&unsharded);
    assert!(!a.is_empty(), "sanity: the sharded run wrote something");
    assert_eq!(
        a.iter().map(|(p, _)| p).collect::<Vec<_>>(),
        b.iter().map(|(p, _)| p).collect::<Vec<_>>(),
        "sharded and unsharded runs must write the same set of files"
    );
    for ((pa, ba), (_, bb)) in a.iter().zip(b.iter()) {
        assert_eq!(
            ba, bb,
            "{pa} differs between the sharded and unsharded runs -- the \
             per-contig sample-index resolution is not shard-invariant"
        );
    }
}

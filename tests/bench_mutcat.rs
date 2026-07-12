//! Profiling harness for the mutcat hot paths (`count_contig`, `annotate_contig`).
//! Not a correctness test — gated behind `#[ignore]` so it only runs when asked:
//!
//!   cargo test --release --no-default-features --features conversion \
//!       --test bench_mutcat -- --ignored --nocapture
//!
//! Knobs via env: BENCH_SAMPLES, BENCH_PLOIDY, BENCH_RECORDS, BENCH_ITERS.

mod common;

use std::time::Instant;

use common::{SynthRecord, build_contig};
use genoray_core::layout::{ContigPaths, MutcatSub};
use genoray_core::mutcat::count::{Sidecars, count_contig};
use genoray_core::mutcat::sidecar::write_sidecar;
use genoray_core::query::ContigReader;
use ndarray::Array2;
use svar2_codec::encode_snp_2bit;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A tiny deterministic LCG so the workload is reproducible without rand.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

/// Owned buffers so `SynthRecord<'a>` can borrow them.
struct Owned {
    ref_a: Vec<u8>,
    alt: Vec<u8>,
    gt: Vec<i32>,
    pos: i64,
}

fn make_workload(n_samples: usize, ploidy: usize, n_records: usize) -> Vec<Owned> {
    const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
    let n_haps = n_samples * ploidy;
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n_records);
    let mut pos: i64 = 100;
    let mut i = 0;
    while i < n_records {
        // ~15% indels, rest SNPs. Among SNPs, ~10% form an adjacent doublet
        // (exercises the DBS pairing branch), ~30% are "common" (many carriers
        // -> dense route), the rest rare (few carriers -> var_key route).
        let roll = rng.below(100);
        if roll < 15 {
            // indel: half INS, half DEL, tail 1..4
            let anchor = BASES[rng.below(4) as usize];
            let tlen = 1 + rng.below(4) as usize;
            let mut tail = Vec::with_capacity(tlen);
            for _ in 0..tlen {
                let mut b = BASES[rng.below(4) as usize];
                if b == anchor {
                    b = BASES[((anchor as usize) + 1) & 3];
                }
                tail.push(b);
            }
            let (ref_a, alt) = if rng.below(2) == 0 {
                let mut a = vec![anchor];
                a.extend_from_slice(&tail);
                (vec![anchor], a) // INS
            } else {
                let mut r = vec![anchor];
                r.extend_from_slice(&tail);
                (r, vec![anchor]) // DEL
            };
            let carriers = 1 + rng.below(5) as usize;
            let gt = random_gt(&mut rng, n_haps, carriers);
            out.push(Owned {
                ref_a,
                alt,
                gt,
                pos,
            });
            pos += 20;
            i += 1;
        } else {
            let anchor = BASES[rng.below(4) as usize];
            let alt_b = BASES[((rng.below(3) as usize) + 1 + anchor as usize) & 3];
            let common = rng.below(100) < 30;
            let carriers = if common {
                n_haps / 3 + rng.below((n_haps / 3).max(1) as u32) as usize
            } else {
                1 + rng.below(3) as usize
            };
            let gt = random_gt(&mut rng, n_haps, carriers);
            out.push(Owned {
                ref_a: vec![anchor],
                alt: vec![alt_b],
                gt,
                pos,
            });
            // ~10% chance of an adjacent partner one base over (rare route).
            if !common && rng.below(100) < 10 && i + 1 < n_records {
                let anchor2 = BASES[rng.below(4) as usize];
                let alt2 = BASES[((rng.below(3) as usize) + 1 + anchor2 as usize) & 3];
                let carriers2 = 1 + rng.below(3) as usize;
                let gt2 = random_gt(&mut rng, n_haps, carriers2);
                out.push(Owned {
                    ref_a: vec![anchor2],
                    alt: vec![alt2],
                    gt: gt2,
                    pos: pos + 1,
                });
                i += 1;
            }
            pos += 20;
            i += 1;
        }
    }
    out
}

fn random_gt(rng: &mut Lcg, n_haps: usize, carriers: usize) -> Vec<i32> {
    let mut gt = vec![0i32; n_haps];
    let carriers = carriers.min(n_haps);
    let mut placed = 0;
    while placed < carriers {
        let idx = rng.below(n_haps as u32) as usize;
        if gt[idx] == 0 {
            gt[idx] = 1;
            placed += 1;
        }
    }
    gt
}

/// Inject valid mutcat sidecars for whichever substreams got records, so
/// `count_contig` sees real codes without needing a reference FASTA pass.
fn inject_sidecars(out: &std::path::Path, chrom: &str) {
    let paths = ContigPaths::new(out.to_str().unwrap(), chrom);
    let cdir = out.join(chrom);
    // var_key/snp + dense/snp: u8 code (SBS class-local, use a valid 0..96) + 2-bit ref.
    for (sub, sub_dir) in [
        (MutcatSub::VkSnp, cdir.join("var_key").join("snp")),
        (MutcatSub::DenseSnp, cdir.join("dense").join("snp")),
    ] {
        let pth = sub_dir.join("positions.bin");
        let n = if pth.exists() {
            std::fs::metadata(&pth)
                .map(|m| m.len() as usize / 4)
                .unwrap_or(0)
        } else {
            0
        };
        if n > 0 {
            let codes: Vec<u8> = (0..n).map(|i| (i % 96) as u8).collect();
            let refs: Vec<u8> = (0..n).map(|_| encode_snp_2bit(b'C')).collect();
            write_sidecar(&paths, sub, &codes, Some(&refs)).unwrap();
        }
    }
    for (sub, sub_dir) in [
        (MutcatSub::VkIndel, cdir.join("var_key").join("indel")),
        (MutcatSub::DenseIndel, cdir.join("dense").join("indel")),
    ] {
        let pth = sub_dir.join("positions.bin");
        let n = if pth.exists() {
            std::fs::metadata(&pth)
                .map(|m| m.len() as usize / 4)
                .unwrap_or(0)
        } else {
            0
        };
        if n > 0 {
            let codes: Vec<u8> = (0..n).map(|i| (i % 83) as u8).collect();
            write_sidecar(&paths, sub, &codes, None).unwrap();
        }
    }
}

#[test]
#[ignore]
fn bench_count_contig() {
    let n_samples = env_usize("BENCH_SAMPLES", 300);
    let ploidy = env_usize("BENCH_PLOIDY", 2);
    let n_records = env_usize("BENCH_RECORDS", 3000);
    let iters = env_usize("BENCH_ITERS", 200);

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();

    let owned = make_workload(n_samples, ploidy, n_records);
    let records: Vec<SynthRecord> = owned
        .iter()
        .map(|o| SynthRecord {
            pos: o.pos,
            ref_allele: &o.ref_a,
            alts: vec![&o.alt[..]],
            gt: o.gt.clone(),
        })
        .collect();
    let sample_names: Vec<String> = (0..n_samples).map(|i| format!("S{i}")).collect();
    let sample_refs: Vec<&str> = sample_names.iter().map(|s| s.as_str()).collect();

    let t = Instant::now();
    build_contig(&out, "chr1", &sample_refs, ploidy, &records);
    inject_sidecars(&out, "chr1");
    eprintln!(
        "[setup] built + annotated {n_records} records in {:?}",
        t.elapsed()
    );

    let reader = ContigReader::open(out.to_str().unwrap(), "chr1", n_samples, ploidy).unwrap();
    let sidecars = Sidecars::open(&ContigPaths::new(out.to_str().unwrap(), "chr1")).unwrap();

    // warm-up (page in mmaps)
    let mut acc = Array2::<i64>::zeros((n_samples, genoray_core::mutcat::N_CODES));
    count_contig(&reader, &sidecars, false, &mut acc);
    let checksum: i64 = acc.sum();

    let t = Instant::now();
    let mut sink = 0i64;
    for _ in 0..iters {
        let mut acc = Array2::<i64>::zeros((n_samples, genoray_core::mutcat::N_CODES));
        count_contig(&reader, &sidecars, false, &mut acc);
        sink = sink.wrapping_add(acc[[0, 0]]);
    }
    let dt = t.elapsed();
    eprintln!(
        "[count_contig] {iters} iters in {dt:?} => {:.3} ms/iter (checksum={checksum}, sink={sink}, samples={n_samples} ploidy={ploidy} records={n_records})",
        dt.as_secs_f64() * 1e3 / iters as f64
    );
}

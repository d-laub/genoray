//! Isolates the SVAR2 executor's emission stage (#176, #177).
//!
//! `dense2sparse_vk` is the whole of what the per-contig `exec-{chrom}` thread
//! does per chunk, so timing it on a synthetic `DenseChunk` measures the
//! executor exactly -- no readers, no htslib, no I/O, no cohort-scale corpus to
//! build. Chunks are constructed with `carriers == None`, which is the dispatch
//! `from_vcf` takes (a multi-sample VCF yields `Calls::Dense`), so this drives
//! `dense2sparse_vk_by_scan`.
//!
//! Reports a content digest alongside the timing: #177 is a pure loop
//! restructure whose bar is byte-identical output, so an A/B against the parent
//! commit must show the digest UNCHANGED while the time moves.
//!
//! Run one point:
//!   cargo run --release --no-default-features --features conversion \
//!     --bin bench_dense2sparse -- --samples 535662 --variants 5000 --af 0.001
//!
//! A/B against a baseline commit MUST use a separate CARGO_TARGET_DIR per tree
//! and confirm the binary hash actually differs -- a second tree at a different
//! path can otherwise be handed the first tree's artifact byte-for-byte.

use std::time::Instant;

use genoray_core::rvk::dense2sparse_vk;
use genoray_core::types::{BitGrid3, DenseChunk};

/// Deterministic LCG. Avoids a dev-dependency and makes a point reproducible
/// from its `--seed` alone, which matters when comparing across builds.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// FNV-1a over the emitted chunk. Only needs to be stable across builds of the
/// same input, not cryptographic.
struct Fnv(u64);
impl Fnv {
    // Not `new`: a `new` with no arguments and no `Default` impl trips
    // clippy::new_without_default, and the gates run with `-D warnings`.
    fn start() -> Self {
        Fnv(0xcbf29ce484222325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_samples: usize = arg(&args, "--samples", "16000").parse().unwrap();
    let v_variants: usize = arg(&args, "--variants", "5000").parse().unwrap();
    let ploidy: usize = arg(&args, "--ploidy", "2").parse().unwrap();
    let af: f64 = arg(&args, "--af", "0.001").parse().unwrap();
    let seed: u64 = arg(&args, "--seed", "0xC0FFEE").parse().unwrap_or(0xC0FFEE);
    let reps: usize = arg(&args, "--reps", "3").parse().unwrap();

    let columns = n_samples * ploidy;
    // Carriers per variant. Floored at 1 so a low `--af` still exercises the
    // emission path rather than degenerating to an empty grid.
    let x = ((af * columns as f64).round() as usize).max(1);

    eprintln!(
        "building chunk: V={v_variants} S={n_samples} P={ploidy} columns={columns} \
         af={af} carriers/variant={x} grid={:.2} GB",
        (v_variants * columns) as f64 / 8.0 / 1e9
    );

    let build = Instant::now();
    let mut genos = BitGrid3::zeros(v_variants, n_samples, ploidy);
    let mut rng = Lcg(seed);
    // Set carrier bits directly rather than walking every slot: building the
    // grid must not cost more than the thing being measured.
    for v in 0..v_variants {
        let row = v * columns;
        for _ in 0..x {
            let col = rng.below(columns as u64) as usize;
            let flat = row + col;
            genos.words[flat >> 6] |= 1u64 << (flat & 63);
        }
    }

    // ALTs cycle SNP / INS / DEL so both var_key streams are exercised; no long
    // insertions, so the allele bank stays out of the measurement.
    let mut alt = Vec::new();
    let mut alt_offsets = vec![0u32];
    let mut ilens = Vec::with_capacity(v_variants);
    for v in 0..v_variants {
        match v % 3 {
            0 => {
                alt.push(b'C');
                ilens.push(0);
            }
            1 => {
                alt.extend_from_slice(b"AT");
                ilens.push(1);
            }
            _ => {
                alt.push(b'A');
                ilens.push(-1);
            }
        }
        alt_offsets.push(alt.len() as u32);
    }

    let chunk = DenseChunk {
        chunk_id: 0,
        pos: (0..v_variants as u32).map(|i| 100 + i * 10).collect(),
        global_idx: vec![-1; v_variants],
        ilens,
        alt,
        alt_offsets,
        genos,
        info_staged: Vec::new(),
        format_staged: Vec::new(),
        carriers: None,
        format_by_carrier: None,
    };
    eprintln!("built in {:.2}s", build.elapsed().as_secs_f64());

    let mut best = f64::INFINITY;
    let mut digest = 0u64;
    let mut calls = 0usize;
    for rep in 0..reps {
        let (tx_long, rx_long) = crossbeam_channel::bounded(1 << 12);
        let mut bank = genoray_core::nrvk::LongAlleleTableWriter::new(tx_long, 1 << 20);

        let t = Instant::now();
        let out = dense2sparse_vk(&chunk, &mut bank, false, &[]);
        let secs = t.elapsed().as_secs_f64();

        let mut h = Fnv::start();
        let mut n = 0usize;
        for (_tag, sub) in out.streams.iter() {
            h.write(bytemuck::cast_slice(&sub.call_positions));
            h.write(&sub.call_keys);
            h.write(bytemuck::cast_slice(&sub.sample_lengths));
            n += sub.call_positions.len();
        }
        for (_class, sub) in out.dense.iter() {
            h.write(bytemuck::cast_slice(&sub.positions));
            h.write(&sub.keys);
            h.write(&sub.geno_bits);
        }
        digest = h.0;
        calls = n;
        best = best.min(secs);
        eprintln!("  rep {rep}: {secs:.3}s");
        drop(bank);
        while rx_long.recv().is_ok() {}
    }

    // One line, machine-parseable, so a sweep can concatenate points.
    println!(
        "samples={n_samples} ploidy={ploidy} columns={columns} variants={v_variants} \
         af={af} carriers_per_variant={x} calls={calls} \
         secs={best:.4} ms_per_variant={:.4} digest={digest:016x}",
        best * 1000.0 / v_variants as f64
    );
}

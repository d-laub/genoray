//! Native driver for `orchestrator::run_vcf_list` — the clean target for dhat,
//! perf, callgrind, and cargo-show-asm (no Python in the loop). Sample names are
//! read from each file's header via rust-htslib.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use genoray_core::normalize::CheckRef;
use genoray_core::orchestrator::run_vcf_list;
use rust_htslib::bcf::{Read, Reader};
use std::fs;
use std::time::Instant;

fn sample_of(path: &str) -> String {
    let r = Reader::from_path(path).expect("open vcf");
    let s = r.header().samples();
    String::from_utf8_lossy(s[0]).into_owned()
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <manifest> <out_dir> <chrom> [reference.fa]",
            args[0]
        );
        std::process::exit(2);
    }
    let manifest = &args[1];
    let out_dir = &args[2];
    let chrom = &args[3];
    let reference = args.get(4).map(|s| s.as_str());

    let vcf_paths: Vec<String> = fs::read_to_string(manifest)
        .expect("read manifest")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .collect();
    let samples: Vec<String> = vcf_paths.iter().map(|p| sample_of(p)).collect();

    let t = Instant::now();
    let dropped = run_vcf_list(
        &vcf_paths,
        reference,
        std::slice::from_ref(chrom),
        out_dir,
        &samples,
        25_000,          // chunk_size (later tasks may sweep this)
        2,               // ploidy
        None,            // max_threads = auto
        8 * 1024 * 1024, // long_allele_capacity
        false,           // skip_out_of_scope
        CheckRef::Error, // check_ref (mirrors the from_vcf_list default "e")
        false,           // signatures
        Vec::new(),      // info_fields
        Vec::new(),      // format_fields
    )
    .expect("run_vcf_list");
    eprintln!(
        "done: dropped={dropped} elapsed={:.1}s files={}",
        t.elapsed().as_secs_f64(),
        vcf_paths.len()
    );
}

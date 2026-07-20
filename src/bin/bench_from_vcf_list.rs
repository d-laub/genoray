//! Native driver for `orchestrator::run_vcf_list` — the clean target for dhat,
//! perf, callgrind, and cargo-show-asm (no Python in the loop). Sample names are
//! read from each file's header via rust-htslib.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use genoray_core::normalize::CheckRef;
use genoray_core::orchestrator::run_vcf_list;
use genoray_core::svar2_view::OverlapMode;
use rust_htslib::bcf::{IndexedReader, Read, Reader};
use std::fs;
use std::time::Instant;

/// One INFO/FORMAT field as `run_vcf_list` takes it:
/// `(name, category, htslib_type, storage_dtype, default)` — the same shape
/// `field::parse_manifest` consumes.
type FieldTuple = (String, String, String, Option<String>, Option<f64>);

fn sample_of(path: &str) -> String {
    let r = Reader::from_path(path).expect("open vcf");
    let s = r.header().samples();
    String::from_utf8_lossy(s[0]).into_owned()
}

/// Does `path` actually carry at least one record on `chrom`? Mirrors the
/// cyvcf2 probe `run_vcf_list_conversion_pipeline` runs Python-side: a shared
/// pipeline header can declare a contig every file lacks records on (e.g. a
/// female sample has no somatic chrY), so a fetch either seek-raises (VCF.gz +
/// tabix/CSI) or returns empty (BCF + CSI) -- both mean "not a member". Real
/// cohorts (issue #120) hit this, so the bench must compute membership rather
/// than assume every file carries every contig (which would reproduce #122).
fn file_has_contig(path: &str, chrom: &str) -> bool {
    let Ok(mut reader) = IndexedReader::from_path(path) else {
        return false;
    };
    let Ok(rid) = reader.header().name2rid(chrom.as_bytes()) else {
        return false; // contig absent from header entirely
    };
    if reader.fetch(rid, 0, None).is_err() {
        return false; // seek raised => no index entry for this contig
    }
    let mut rec = reader.empty_record();
    matches!(reader.read(&mut rec), Some(Ok(())))
}

fn main() {
    genoray_core::logging::install_fmt_fallback();

    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <manifest> <out_dir> <chrom>[,<chrom>...] [reference.fa] \
             [format_field:htype,...]",
            args[0]
        );
        eprintln!(
            "  Multiple chroms exercise the per-contig lifecycle; a non-empty field \
             list exercises the F x N staging path. Both are required to reproduce \
             production memory behaviour -- see issue #120."
        );
        std::process::exit(2);
    }
    let manifest = &args[1];
    let out_dir = &args[2];
    // `run_vcf_list` takes a slice of chroms; passing more than one is what exposes
    // memory that is retained across contig boundaries rather than per contig.
    let chroms: Vec<String> = args[3].split(',').map(|s| s.trim().to_string()).collect();
    let reference = args
        .get(4)
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty() && *s != "-");

    // `name:htype` pairs, e.g. "VAF:float,DP:int". FORMAT values are staged per
    // (variant x sample), so this is the dominant per-chunk cost at large N.
    let format_fields: Vec<FieldTuple> = args
        .get(5)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.split(',')
                .map(|item| {
                    let (name, htype) = item
                        .split_once(':')
                        .unwrap_or_else(|| panic!("field {item:?} must be name:htype"));
                    (
                        name.trim().to_string(),
                        "format".to_string(),
                        htype.trim().to_string(),
                        None,
                        None,
                    )
                })
                .collect()
        })
        .unwrap_or_default();

    let vcf_paths: Vec<String> = fs::read_to_string(manifest)
        .expect("read manifest")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .collect();
    let samples: Vec<String> = vcf_paths.iter().map(|p| sample_of(p)).collect();

    eprintln!(
        "bench: files={} chroms={:?} format_fields={} reference={}",
        vcf_paths.len(),
        chroms,
        format_fields.len(),
        reference.unwrap_or("<none>"),
    );

    // Per-contig membership, computed exactly as the Python entrypoint does
    // (probe each file for records on each contig) so the bench runs on real
    // cohorts whose shared header declares contigs some files lack (issue #122).
    let contig_membership: Vec<Vec<bool>> = chroms
        .iter()
        .map(|c| vcf_paths.iter().map(|p| file_has_contig(p, c)).collect())
        .collect();

    let t = Instant::now();
    let dropped = run_vcf_list(
        &vcf_paths,
        reference,
        &chroms,
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
        format_fields,
        Vec::new(),       // region_ranges: empty => whole contig, as the bench intends
        OverlapMode::Pos, // overlap (inert with no regions; mirrors the default)
        contig_membership,
    )
    .expect("run_vcf_list");
    eprintln!(
        "done: dropped={dropped} elapsed={:.1}s files={}",
        t.elapsed().as_secs_f64(),
        vcf_paths.len()
    );
}

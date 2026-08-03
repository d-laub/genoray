//! Per-contig work estimates used to order contigs longest-first before
//! dispatch. Reads index/header metadata only -- never variant data.
//!
//! Only RATIOS matter: these values are a sort key and nothing else. That is
//! why a coarse tier is an acceptable fallback and why the absolute unit is
//! allowed to differ between tiers.

use rust_htslib::bcf::Read;
use std::collections::HashMap;
use std::ffi::CString;

/// Exact per-contig record counts from the `.csi`/`.tbi`, or `None` if the
/// index is absent or carries no per-reference statistics.
///
/// `hts_idx_get_stat` is documented for BAM; whether CSI/TBI over VCF
/// populates the mapped count is not guaranteed, which is why every failure
/// mode here returns `None` and lets the caller fall back rather than
/// reporting a confident zero.
fn counts_from_index(vcf_path: &str, chroms: &[String]) -> Option<HashMap<String, u64>> {
    let reader = rust_htslib::bcf::Reader::from_path(vcf_path).ok()?;
    let header = reader.header();

    let c_path = CString::new(vcf_path).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the
    // call; `hts_idx_load` returns null on any failure, which is checked.
    let idx = unsafe {
        rust_htslib::htslib::hts_idx_load(c_path.as_ptr(), rust_htslib::htslib::HTS_FMT_CSI as i32)
    };
    if idx.is_null() {
        return None;
    }

    let mut out = HashMap::new();
    let mut any_nonzero = false;
    for chrom in chroms {
        let Some(rid) = header.name2rid(chrom.as_bytes()).ok() else {
            continue;
        };
        let mut mapped: u64 = 0;
        let mut unmapped: u64 = 0;
        // SAFETY: `idx` is non-null and owned here; `rid` came from this
        // file's own header; both out-params are valid for the call.
        let ret = unsafe {
            rust_htslib::htslib::hts_idx_get_stat(idx, rid as i32, &mut mapped, &mut unmapped)
        };
        if ret == 0 && mapped > 0 {
            any_nonzero = true;
            out.insert(chrom.clone(), mapped);
        }
    }
    // SAFETY: `idx` was produced by `hts_idx_load` and is destroyed exactly
    // once, here, on every path that got past the null check.
    unsafe { rust_htslib::htslib::hts_idx_destroy(idx) };

    any_nonzero.then_some(out)
}

/// Header contig lengths. Always available, and a reasonable proxy for a
/// whole-genome VCF; a poor one for exome or targeted data, which is why it
/// is the fallback rather than the primary source.
fn lengths_from_header(vcf_path: &str, chroms: &[String]) -> HashMap<String, u64> {
    use rust_htslib::bcf::header::HeaderRecord;

    let Ok(reader) = rust_htslib::bcf::Reader::from_path(vcf_path) else {
        return HashMap::new();
    };
    // rust-htslib 1.0's HeaderView has no rid2length; contig lengths are only
    // reachable through the structured header records.
    let wanted: std::collections::HashSet<&str> = chroms.iter().map(|s| s.as_str()).collect();
    let mut out = HashMap::new();
    for rec in reader.header().header_records() {
        let HeaderRecord::Contig { values, .. } = rec else {
            continue;
        };
        let (Some(id), Some(len)) = (values.get("ID"), values.get("length")) else {
            continue;
        };
        if wanted.contains(id.as_str())
            && let Ok(n) = len.parse::<u64>()
        {
            out.insert(id.clone(), n);
        }
    }
    out
}

/// Per-contig work estimates, best source available.
pub fn estimate_contig_costs(vcf_path: &str, chroms: &[String]) -> HashMap<String, u64> {
    match counts_from_index(vcf_path, chroms) {
        Some(counts) => {
            tracing::debug!(source = "index", n = counts.len(), "contig cost estimates");
            counts
        }
        None => {
            let lens = lengths_from_header(vcf_path, chroms);
            tracing::debug!(
                source = "header_length",
                n = lens.len(),
                "contig cost estimates (index carried no per-contig counts)"
            );
            lens
        }
    }
}

/// Order contigs most-expensive-first (LPT). Rayon's work stealing does the
/// dynamic balancing; descending order is the whole scheduling contribution.
///
/// A contig with no estimate sorts FIRST: guessing high costs a slightly worse
/// order, guessing low risks starting the longest job last. Ties break by name
/// so dispatch order is deterministic across runs.
pub fn order_longest_first(chroms: &[String], costs: &HashMap<String, u64>) -> Vec<String> {
    let mut out = chroms.to_vec();
    out.sort_by(|a, b| {
        let ca = costs.get(a).copied().unwrap_or(u64::MAX);
        let cb = costs.get(b).copied().unwrap_or(u64::MAX);
        cb.cmp(&ca).then_with(|| a.cmp(b))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_htslib::bcf::record::GenotypeAllele;
    use rust_htslib::bcf::{Format, Header, Writer};
    use std::collections::HashMap;

    /// Build a 3-contig VCF with deliberately unequal record counts
    /// (chrA=5, chrB=40, chrC=15) and index it. Mirrors the fixture pattern
    /// already used in `src/vcf_reader.rs`'s tests.
    fn three_contig_vcf(dir: &std::path::Path) -> String {
        let path = dir.join("cost.vcf.gz");
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chrA,length=100000>");
        header.push_record(b"##contig=<ID=chrB,length=100000>");
        header.push_record(b"##contig=<ID=chrC,length=100000>");
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_sample(b"s1");
        {
            let mut w = Writer::from_path(&path, &header, false, Format::Vcf).unwrap();
            for (chrom, n) in [("chrA", 5u32), ("chrB", 40), ("chrC", 15)] {
                let rid = w.header().name2rid(chrom.as_bytes()).unwrap();
                for i in 0..n {
                    let mut rec = w.empty_record();
                    rec.set_rid(Some(rid));
                    rec.set_pos((i as i64 + 1) * 100);
                    rec.set_alleles(&[b"A", b"C"]).unwrap();
                    rec.push_genotypes(&[GenotypeAllele::Phased(0), GenotypeAllele::Phased(0)])
                        .unwrap();
                    w.write(&rec).unwrap();
                }
            }
        }
        let p = path.to_str().unwrap().to_string();
        rust_htslib::bcf::index::build(&p, None, 1, rust_htslib::bcf::index::Type::Csi(14))
            .unwrap();
        p
    }

    #[test]
    fn estimates_rank_contigs_by_true_record_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = three_contig_vcf(dir.path());
        let chroms: Vec<String> = ["chrA", "chrB", "chrC"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let costs = estimate_contig_costs(&path, &chroms);
        // Only the ORDER is contractual -- the absolute unit differs per tier.
        // If this fails with all-equal costs, tier 1 returned nothing useful
        // for CSI-over-VCF; delete tier 1 and ship the header-length tier.
        assert!(costs["chrB"] > costs["chrC"], "costs = {costs:?}");
        assert!(costs["chrC"] > costs["chrA"], "costs = {costs:?}");
    }

    #[test]
    fn orders_longest_first() {
        let costs: HashMap<String, u64> = [
            ("chrA".to_string(), 5u64),
            ("chrB".to_string(), 40),
            ("chrC".to_string(), 15),
        ]
        .into_iter()
        .collect();
        let chroms: Vec<String> = ["chrA", "chrB", "chrC"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            order_longest_first(&chroms, &costs),
            vec!["chrB", "chrC", "chrA"]
        );
    }

    /// An unestimated contig sorts FIRST. Guessing high costs a slightly worse
    /// order; guessing low risks starting the longest job last, which is the
    /// exact failure the ordering exists to prevent.
    #[test]
    fn unknown_contig_sorts_first() {
        let costs: HashMap<String, u64> = [("chrA".to_string(), 5u64), ("chrB".to_string(), 40)]
            .into_iter()
            .collect();
        let chroms: Vec<String> = ["chrA", "chrB", "chrZ"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            order_longest_first(&chroms, &costs),
            vec!["chrZ", "chrB", "chrA"]
        );
    }

    /// Ordering must be total and deterministic -- rayon dispatch order feeds
    /// the output layout, and a tie broken by HashMap iteration order would
    /// make the run non-reproducible.
    #[test]
    fn ties_break_deterministically_by_name() {
        let costs: HashMap<String, u64> = [
            ("chrA".to_string(), 7u64),
            ("chrB".to_string(), 7),
            ("chrC".to_string(), 7),
        ]
        .into_iter()
        .collect();
        let chroms: Vec<String> = ["chrC", "chrA", "chrB"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for _ in 0..20 {
            assert_eq!(
                order_longest_first(&chroms, &costs),
                vec!["chrA", "chrB", "chrC"]
            );
        }
    }

    #[test]
    fn missing_index_falls_back_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let chroms = vec!["chrA".to_string()];
        let costs =
            estimate_contig_costs(dir.path().join("absent.vcf.gz").to_str().unwrap(), &chroms);
        assert!(costs.is_empty() || costs.contains_key("chrA"));
    }
}

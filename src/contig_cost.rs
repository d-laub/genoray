//! Per-contig work estimates used to order contigs longest-first before
//! dispatch. Reads index/header metadata only -- never variant data.
//!
//! Only RATIOS matter: these values are a sort key and nothing else. That is
//! why a coarse tier is an acceptable fallback and why the absolute unit is
//! allowed to differ between tiers.

use rust_htslib::bcf::Read;
use std::collections::HashMap;
use std::ffi::CString;

/// Whether `c_path`'s content is VCF-flavoured (tabix's territory) or
/// something else (BCF, in this crate's usage), or `None` if the file can't
/// even be opened to check. Determined by `hts_get_format`, i.e. htslib's
/// own inspection of the file's actual magic bytes -- never guessed from the
/// file extension -- mirroring how htslib's own `bcf_index_build3` dispatches
/// (`switch (fp->format.format) { case bcf: ...; case vcf: ...; }`).
///
/// This exists because a BCF's `.csi` carries no tabix meta at all (only
/// `vcf_idx_init` writes it, `bcf_idx_init` never does), so attempting the
/// tabix load on a BCF makes `tbx.c`'s `index_load` hit its `l_meta < 28`
/// check and unconditionally log `"Invalid index header for %s"` to stderr
/// -- `HTS_IDX_SILENT_FAIL` does not gate that particular message. Checking
/// the format first and skipping the tabix attempt for non-VCF input avoids
/// ever triggering it.
fn is_vcf_flavoured(c_path: &CString) -> Option<bool> {
    // SAFETY: `c_path` is a valid NUL-terminated string alive for the call;
    // the mode string is a static NUL-terminated literal; `hts_open` returns
    // null on any failure, checked before use.
    let fp = unsafe { rust_htslib::htslib::hts_open(c_path.as_ptr(), c"r".as_ptr()) };
    if fp.is_null() {
        return None;
    }
    // SAFETY: `fp` is non-null, just checked; `hts_get_format` returns a
    // pointer into `fp`'s own storage, valid until `fp` is closed, and is
    // read before that close.
    let format = unsafe { (*rust_htslib::htslib::hts_get_format(fp)).format };
    // SAFETY: `fp` was produced by `hts_open` and is closed exactly once,
    // here, on every path that got past the null check.
    unsafe { rust_htslib::htslib::hts_close(fp) };
    Some(format == rust_htslib::htslib::htsExactFormat_vcf)
}

/// Exact per-contig record counts via the tabix path (bgzipped VCF only --
/// callers must not invoke this for BCF input; see `is_vcf_flavoured`), or
/// `None` if no tabix-flavoured index is present.
///
/// `tbx_name2id` resolves each contig name in the INDEX's own id space,
/// which is required here and not optional: a VCF-flavoured CSI/TBI stores
/// only *covered* references, compacted in order of first appearance among
/// data records, so the index's per-contig slot number does not equal the
/// header's declared rid whenever any header contig is uncovered (the common
/// case for a single-contig VCF carrying a full reference-genome header) or
/// records arrive out of header order. Resolving by header rid instead reads
/// the wrong slot -- a confidently wrong count attributed to the wrong
/// contig -- and `hts_idx_get_stat` performs no bounds check of its `tid`
/// argument, so a header rid at or past the index's slot count is an
/// out-of-bounds read.
fn counts_from_tabix(c_path: &CString, chroms: &[String]) -> Option<HashMap<String, u64>> {
    // SAFETY: `c_path` is a valid NUL-terminated string alive for the call;
    // `tbx_index_load3` returns null on any failure (no tabix index present,
    // or a `.csi`/`.tbi` that isn't tabix-flavoured), checked before use;
    // `HTS_IDX_SILENT_FAIL` suppresses htslib's stderr log for that routine
    // case rather than logging on every run without an index.
    let tbx = unsafe {
        rust_htslib::htslib::tbx_index_load3(
            c_path.as_ptr(),
            std::ptr::null(),
            rust_htslib::htslib::HTS_IDX_SILENT_FAIL as i32,
        )
    };
    if tbx.is_null() {
        return None;
    }
    // SAFETY: `tbx` is non-null, just checked above.
    let nseq = unsafe { rust_htslib::htslib::hts_idx_nseq((*tbx).idx) };

    let mut out = HashMap::new();
    let mut any_nonzero = false;
    for chrom in chroms {
        let Ok(c_chrom) = CString::new(chrom.as_str()) else {
            continue;
        };
        // SAFETY: `tbx` is non-null and owned for the duration of this loop;
        // `c_chrom` is a valid NUL-terminated string for the call.
        let tid = unsafe { rust_htslib::htslib::tbx_name2id(tbx, c_chrom.as_ptr()) };
        if tid < 0 || tid >= nseq {
            // Not in the index's name table (uncovered/unknown contig), or
            // -- belt and suspenders against a truncated/corrupt index,
            // where the meta name count and on-disk `n_ref` could disagree
            // -- out of `(*tbx).idx`'s slot range. Never pass this to
            // `hts_idx_get_stat`, which does not bounds-check it.
            continue;
        }
        let mut mapped: u64 = 0;
        let mut unmapped: u64 = 0;
        // SAFETY: `tbx` is non-null; `tid` was just bounded against
        // `hts_idx_nseq((*tbx).idx)` above, so it is in range for
        // `(*tbx).idx`'s `bidx` array -- the one thing `hts_idx_get_stat`
        // does not check for itself. Both out-params are valid locals.
        let ret = unsafe {
            rust_htslib::htslib::hts_idx_get_stat((*tbx).idx, tid, &mut mapped, &mut unmapped)
        };
        if ret == 0 && mapped > 0 {
            any_nonzero = true;
            out.insert(chrom.clone(), mapped);
        }
    }
    // SAFETY: `tbx` was produced by `tbx_index_load3` and is destroyed
    // exactly once, here, on every path that got past the null check.
    unsafe { rust_htslib::htslib::tbx_destroy(tbx) };

    any_nonzero.then_some(out)
}

/// Exact per-contig record counts via a CSI loaded directly, or `None` if
/// none is present. This is the non-VCF (BCF) path, reached only through
/// `counts_from_index`'s format dispatch: unlike tabix, a bare CSI has no
/// name table, so contigs are resolved by the VCF header's own rid, which is
/// correct here because BCF's CSI genuinely is keyed by header rid --
/// `bcf_idx_init` sizes the index directly from the header's own contig
/// count (`hts_idx_init(n_contigs, ...)`), and BCF has no separate
/// covered-only compaction the way tabix-over-VCF does. Every id is still
/// bounded against `hts_idx_nseq` before use, unconditionally:
/// `hts_idx_get_stat` never bounds-checks its `tid` argument on any path,
/// and a header can in principle declare more contigs than the index has
/// slots for.
fn counts_from_csi(
    vcf_path: &str,
    c_path: &CString,
    chroms: &[String],
) -> Option<HashMap<String, u64>> {
    let reader = rust_htslib::bcf::Reader::from_path(vcf_path).ok()?;
    let header = reader.header();

    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the
    // call; `hts_idx_load3` returns null on any failure, checked before use.
    // `HTS_FMT_TBI` is a search-order hint, not a format assertion: it makes
    // htslib try `.csi` first, then `.tbi`, regardless of the file's actual
    // on-disk format. `HTS_IDX_SILENT_FAIL` suppresses htslib's stderr log
    // for the routine "no index" case.
    let idx = unsafe {
        rust_htslib::htslib::hts_idx_load3(
            c_path.as_ptr(),
            std::ptr::null(),
            rust_htslib::htslib::HTS_FMT_TBI as i32,
            rust_htslib::htslib::HTS_IDX_SILENT_FAIL as i32,
        )
    };
    if idx.is_null() {
        return None;
    }
    // SAFETY: `idx` is non-null, just checked above.
    let nseq = unsafe { rust_htslib::htslib::hts_idx_nseq(idx) };

    let mut out = HashMap::new();
    let mut any_nonzero = false;
    for chrom in chroms {
        let Some(rid) = header.name2rid(chrom.as_bytes()).ok() else {
            continue;
        };
        let tid = rid as i32;
        if tid < 0 || tid >= nseq {
            // Out of the index's slot range -- never pass this to
            // `hts_idx_get_stat`, which does not bounds-check it.
            continue;
        }
        let mut mapped: u64 = 0;
        let mut unmapped: u64 = 0;
        // SAFETY: `idx` is non-null and owned here; `tid` was just bounded
        // against `hts_idx_nseq(idx)` above, so it is in range for `idx`'s
        // `bidx` array; both out-params are valid locals.
        let ret =
            unsafe { rust_htslib::htslib::hts_idx_get_stat(idx, tid, &mut mapped, &mut unmapped) };
        if ret == 0 && mapped > 0 {
            any_nonzero = true;
            out.insert(chrom.clone(), mapped);
        }
    }
    // SAFETY: `idx` was produced by `hts_idx_load3` and is destroyed exactly
    // once, here, on every path that got past the null check.
    unsafe { rust_htslib::htslib::hts_idx_destroy(idx) };

    any_nonzero.then_some(out)
}

/// Exact per-contig record counts from the index, or `None` if the file
/// can't be opened, no index is present, or the index carries no
/// per-reference statistics for any requested contig. Dispatches on the
/// file's actual content format (`is_vcf_flavoured`) rather than trying
/// tabix and falling back to CSI on any failure: a tabix load against BCF
/// input logs an htslib error unconditionally (see `is_vcf_flavoured`'s
/// doc), and falling back to the header-rid CSI path after a tabix load
/// that merely found zero matching counts would let it run against a
/// tabix-flavoured index using header rids -- the same mixed-id-space
/// mistake this module exists to avoid, just with a `hts_idx_nseq` bound
/// keeping it from going out of range. Format dispatch makes both
/// impossible: exactly one path ever runs, never both.
fn counts_from_index(vcf_path: &str, chroms: &[String]) -> Option<HashMap<String, u64>> {
    // `hts_open` (used by `is_vcf_flavoured` to check the file's actual
    // format) logs an error unconditionally on a failed open -- unlike
    // `hts_idx_load3`/`tbx_index_load3`, it has no silent-fail flag to pass.
    // Checked here first rather than ever handing it a path that can't be
    // opened, mirroring `rust_htslib::bcf::Reader::from_path`'s own
    // existence pre-check (which is why the old `bcf::Reader`-first code
    // never had this problem: a missing file never reached htslib at all).
    if !std::path::Path::new(vcf_path).exists() {
        return None;
    }
    let c_path = CString::new(vcf_path).ok()?;
    match is_vcf_flavoured(&c_path)? {
        true => counts_from_tabix(&c_path, chroms),
        false => counts_from_csi(vcf_path, &c_path, chroms),
    }
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

/// Per-contig work estimates, tagged with which tier produced them.
///
/// `Deref`s to the inner map so existing indexing/`.get`/`.contains_key`/
/// `.is_empty` call sites keep working unchanged; `exact_counts` is the only
/// thing callers need to check before treating a value as anything more
/// specific than "a sort key" (see `order_longest_first`'s doc: only RATIOS
/// are contractual across tiers, and the absolute unit differs between
/// them).
#[derive(Debug, Clone, Default)]
pub struct ContigCosts {
    pub values: HashMap<String, u64>,
    /// True iff `values` are exact per-contig VARIANT RECORD counts from the
    /// index tier (tabix/CSI, `counts_from_index`). False means the
    /// header-length fallback tier (`lengths_from_header`), whose values are
    /// base-pair CONTIG LENGTHS -- a different unit that must never be
    /// treated as a record count (e.g. for a resident-chunk-size memory
    /// estimate: `min(chunk_size, records)` is only valid math when
    /// `records` really are record counts).
    pub exact_counts: bool,
}

impl std::ops::Deref for ContigCosts {
    type Target = HashMap<String, u64>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// Per-contig work estimates, best source available.
pub fn estimate_contig_costs(vcf_path: &str, chroms: &[String]) -> ContigCosts {
    match counts_from_index(vcf_path, chroms) {
        Some(counts) => {
            tracing::debug!(source = "index", n = counts.len(), "contig cost estimates");
            ContigCosts {
                values: counts,
                exact_counts: true,
            }
        }
        None => {
            let lens = lengths_from_header(vcf_path, chroms);
            tracing::debug!(
                source = "header_length",
                n = lens.len(),
                "contig cost estimates (index carried no per-contig counts)"
            );
            ContigCosts {
                values: lens,
                exact_counts: false,
            }
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

    /// Build a VCF over contigs chrA/chrB/chrC, each header-declared with the
    /// same length, with the given per-contig record count (order chrA, chrB,
    /// chrC; a count may be 0, leaving that contig with no records at all),
    /// and index it. Mirrors the fixture pattern already used in
    /// `src/vcf_reader.rs`'s tests.
    fn three_contig_vcf(dir: &std::path::Path, counts: [u32; 3]) -> String {
        let path = dir.join("cost.vcf.gz");
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chrA,length=100000>");
        header.push_record(b"##contig=<ID=chrB,length=100000>");
        header.push_record(b"##contig=<ID=chrC,length=100000>");
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_sample(b"s1");
        {
            let mut w = Writer::from_path(&path, &header, false, Format::Vcf).unwrap();
            for (chrom, n) in [
                ("chrA", counts[0]),
                ("chrB", counts[1]),
                ("chrC", counts[2]),
            ] {
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

    /// BCF counterpart of `three_contig_vcf`: same contigs, header lengths,
    /// and per-contig record counts, written as `Format::Bcf` and
    /// CSI-indexed. A BCF's CSI carries no tabix meta at all, so this
    /// exercises `counts_from_csi`'s header-rid path end to end -- and, via
    /// `bcf_input_uses_csi_path_without_a_tabix_error` below, that the
    /// tabix attempt is never even made against it.
    fn three_contig_bcf(dir: &std::path::Path, counts: [u32; 3]) -> String {
        let path = dir.join("cost.bcf");
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chrA,length=100000>");
        header.push_record(b"##contig=<ID=chrB,length=100000>");
        header.push_record(b"##contig=<ID=chrC,length=100000>");
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_sample(b"s1");
        {
            let mut w = Writer::from_path(&path, &header, false, Format::Bcf).unwrap();
            for (chrom, n) in [
                ("chrA", counts[0]),
                ("chrB", counts[1]),
                ("chrC", counts[2]),
            ] {
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
        let path = three_contig_vcf(dir.path(), [5, 40, 15]);
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
        // This VCF has a tabix index, so the counts came from tier 1 (exact
        // per-contig record counts) -- callers that need to distinguish
        // "these are real record counts" from "these are byte-length
        // proxies" depend on this flag being set correctly per tier.
        assert!(costs.exact_counts);
    }

    /// A VCF-flavoured CSI/TBI stores only COVERED references, compacted in
    /// order of first appearance among data records -- an uncovered header
    /// contig leaves no slot at all. chrB here has zero records, so if
    /// `counts_from_index` ever resolved by raw header rid instead of the
    /// index's own id space, chrC's on-disk slot would be read under chrB's
    /// rid (an off-by-one misattribution) or, more generally, any header rid
    /// at or beyond the index's slot count would index out of bounds.
    #[test]
    fn zero_record_contig_does_not_corrupt_other_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = three_contig_vcf(dir.path(), [5, 0, 15]);
        let chroms: Vec<String> = ["chrA", "chrB", "chrC"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let costs = estimate_contig_costs(&path, &chroms);
        // chrB has no records: it must not silently absorb chrC's count.
        assert_ne!(
            costs.get("chrB").copied(),
            costs.get("chrC").copied(),
            "costs = {costs:?}"
        );
        // The two covered contigs must still rank by their true counts.
        assert!(costs["chrC"] > costs["chrA"], "costs = {costs:?}");
    }

    /// A BCF's `.csi` has no tabix meta at all (only `vcf_idx_init` writes
    /// it; `bcf_idx_init` never does). Before `is_vcf_flavoured` gated the
    /// tabix attempt by actual file format, `counts_from_tabix` was tried
    /// unconditionally, including here, and `tbx.c`'s `index_load` logged
    /// `"Invalid index header for %s"` to stderr on every call -- run this
    /// test with `--nocapture` to confirm that line is gone. Correctness of
    /// the counts themselves is asserted here; the absence of that log line
    /// is confirmed by inspecting the run's captured output, not by this
    /// assertion (there is no portable in-process way to assert on
    /// htslib's own C-level stderr writes).
    #[test]
    fn bcf_input_uses_csi_path_without_a_tabix_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = three_contig_bcf(dir.path(), [5, 40, 15]);
        let chroms: Vec<String> = ["chrA", "chrB", "chrC"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let costs = estimate_contig_costs(&path, &chroms);
        assert_eq!(costs.get("chrA").copied(), Some(5));
        assert_eq!(costs.get("chrB").copied(), Some(40));
        assert_eq!(costs.get("chrC").copied(), Some(15));
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

    /// `missing_index_falls_back_without_panicking` points at a file that
    /// doesn't exist at all, so BOTH tiers return empty and
    /// `lengths_from_header`'s actual body (the `HeaderRecord::Contig` match,
    /// the `"ID"`/`"length"` lookups, the `u64` parse) never runs against
    /// real data -- its failure mode there is silent (an empty map just
    /// makes every contig "unknown"). This test opens a real, existing VCF
    /// with no `.csi`/`.tbi` at all, so `counts_from_index` falls through
    /// cleanly and tier 2 must produce the true header lengths.
    #[test]
    fn tier2_returns_real_header_lengths_when_no_index_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_index.vcf.gz");
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chrA,length=10000>");
        header.push_record(b"##contig=<ID=chrB,length=800000>");
        header.push_record(b"##contig=<ID=chrC,length=300000>");
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        header.push_sample(b"s1");
        {
            let mut w = Writer::from_path(&path, &header, false, Format::Vcf).unwrap();
            let rid = w.header().name2rid(b"chrA").unwrap();
            let mut rec = w.empty_record();
            rec.set_rid(Some(rid));
            rec.set_pos(0);
            rec.set_alleles(&[b"A", b"C"]).unwrap();
            rec.push_genotypes(&[GenotypeAllele::Phased(0), GenotypeAllele::Phased(0)])
                .unwrap();
            w.write(&rec).unwrap();
        }
        // Deliberately no `rust_htslib::bcf::index::build` call.
        let path = path.to_str().unwrap().to_string();

        let chroms: Vec<String> = ["chrA", "chrB", "chrC"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let costs = estimate_contig_costs(&path, &chroms);
        assert_eq!(costs.get("chrA").copied(), Some(10000));
        assert_eq!(costs.get("chrB").copied(), Some(800000));
        assert_eq!(costs.get("chrC").copied(), Some(300000));
        // No index at all -- these values are header CONTIG LENGTHS, not
        // record counts. A caller that mistook this tier for `exact_counts`
        // would badly under-estimate a `min(chunk_size, records)` memory
        // guard against real record counts thousands of times smaller.
        assert!(!costs.exact_counts);
    }
}

use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec, HtslibType};
use crate::record_source::{RawRecord, RecordSource};
use rust_htslib::bcf::record::Record;
use rust_htslib::bcf::{IndexedReader, Read};
use rust_htslib::errors::Error as HtslibError;

// Decode one INFO field for the CURRENT record, once. `Ok(None)` means the
// field is absent from this record (a normal, expected occurrence — NOT an
// error); a genuine htslib read failure (bad type, corrupt buffer) surfaces
// as `ConversionError::Input`, matching the GT-decode error style.
fn decode_info_raw(record: &Record, spec: &FieldSpec) -> Result<Option<Vec<f64>>, ConversionError> {
    let pos = record.pos();
    match spec.htype {
        HtslibType::Flag => {
            // A Flag is never "missing": absent ⇒ false ⇒ 0.0.
            let present = record.info(spec.name.as_bytes()).flag().map_err(|e| {
                ConversionError::Input(format!(
                    "Failed to read INFO/{} flag at pos {pos}: {e}",
                    spec.name
                ))
            })?;
            Ok(Some(vec![if present { 1.0 } else { 0.0 }]))
        }
        HtslibType::Int => match record.info(spec.name.as_bytes()).integer() {
            Ok(Some(buf)) => Ok(Some(buf.iter().map(|&v| v as f64).collect())),
            Ok(None) => Ok(None),
            Err(e) => Err(ConversionError::Input(format!(
                "Failed to read INFO/{} at pos {pos}: {e}",
                spec.name
            ))),
        },
        HtslibType::Float => match record.info(spec.name.as_bytes()).float() {
            Ok(Some(buf)) => Ok(Some(buf.iter().map(|&v| v as f64).collect())),
            Ok(None) => Ok(None),
            Err(e) => Err(ConversionError::Input(format!(
                "Failed to read INFO/{} at pos {pos}: {e}",
                spec.name
            ))),
        },
    }
}

// Decode one FORMAT field for the CURRENT record, once, for every VCF sample
// in the header (not just the selected ones — matches the GT-decode idiom,
// which indexes into the full per-header-sample buffer via
// `self.sample_indices`). `Ok(None)` means the field is absent from this
// record for all samples (htslib reports this as `BcfMissingTag`, a normal,
// expected per-record occurrence, NOT an error). Any other htslib error is a
// genuine read failure and surfaces as `ConversionError::Input`.
//
// FORMAT Flag is not valid VCF (Flag is INFO-only); defensively treated as Int.
fn decode_format_raw(
    record: &Record,
    spec: &FieldSpec,
) -> Result<Option<Vec<Vec<f64>>>, ConversionError> {
    let pos = record.pos();
    let result = match spec.htype {
        HtslibType::Float => record.format(spec.name.as_bytes()).float().map(|bb| {
            bb.iter()
                .map(|s| s.iter().map(|&v| v as f64).collect())
                .collect::<Vec<Vec<f64>>>()
        }),
        HtslibType::Int | HtslibType::Flag => {
            record.format(spec.name.as_bytes()).integer().map(|bb| {
                bb.iter()
                    .map(|s| s.iter().map(|&v| v as f64).collect())
                    .collect::<Vec<Vec<f64>>>()
            })
        }
    };
    match result {
        Ok(v) => Ok(Some(v)),
        Err(HtslibError::BcfMissingTag { .. }) => Ok(None),
        Err(e) => Err(ConversionError::Input(format!(
            "Failed to read FORMAT/{} at pos {pos}: {e}",
            spec.name
        ))),
    }
}

/// Load `chrom`'s full sequence from the FASTA at `fasta_path`, uppercased to
/// ASCII (matching the classifiers'/normalizer's expectation). Shared by
/// `ChunkAssembler::new` (validate_ref/left_align) and the orchestrator's
/// write-time `signatures` annotation — keep both call sites byte-identical.
pub(crate) fn load_contig_seq(fasta_path: &str, chrom: &str) -> Result<Vec<u8>, ConversionError> {
    // A wrong reference path reaches Rust unchecked (Python does not validate
    // it), so surface it as FileNotFoundError specifically.
    if !std::path::Path::new(fasta_path).exists() {
        return Err(ConversionError::MissingFile {
            path: fasta_path.to_string(),
        });
    }
    let fasta = rust_htslib::faidx::Reader::from_path(fasta_path).map_err(|e| {
        ConversionError::Input(format!(
            "Failed to open reference FASTA '{fasta_path}' (is there a .fai?): {e}"
        ))
    })?;
    // htslib's faidx_seq_len returns -1 for an unknown contig, surfaced via
    // rust-htslib as u64::MAX — check explicitly for a clear message.
    let contig_len_raw = fasta.fetch_seq_len(chrom);
    if contig_len_raw == u64::MAX {
        return Err(ConversionError::Input(format!(
            "Contig '{chrom}' not found in reference FASTA"
        )));
    }
    let contig_len = contig_len_raw as usize;
    let mut ref_seq = if contig_len == 0 {
        Vec::new()
    } else {
        fasta
            .fetch_seq(chrom, 0, contig_len - 1)
            .map_err(|e| ConversionError::Io {
                context: format!("fetching contig '{chrom}' from reference FASTA"),
                source: std::io::Error::other(e.to_string()),
            })?
    };
    ref_seq.make_ascii_uppercase();
    Ok(ref_seq)
}

/// Cheaply validate that `fasta_path` is openable and contains every contig in
/// `chroms`, WITHOUT fetching any full sequence — opens the faidx once and
/// checks each contig via `fetch_seq_len` (the same checks `load_contig_seq`
/// runs before it fetches bytes). Used by `run_slice_view`'s fail-fast band so
/// a bad `reference=` raises before any output byte is written, while the
/// per-contig loop still loads each sequence lazily (peak O(1 contig), not
/// O(genome)). Error variants/messages are byte-identical to `load_contig_seq`
/// so the raise is indistinguishable from the loop's.
pub(crate) fn validate_contigs_in_fasta(
    fasta_path: &str,
    chroms: &[String],
) -> Result<(), ConversionError> {
    if !std::path::Path::new(fasta_path).exists() {
        return Err(ConversionError::MissingFile {
            path: fasta_path.to_string(),
        });
    }
    let fasta = rust_htslib::faidx::Reader::from_path(fasta_path).map_err(|e| {
        ConversionError::Input(format!(
            "Failed to open reference FASTA '{fasta_path}' (is there a .fai?): {e}"
        ))
    })?;
    for chrom in chroms {
        // u64::MAX == htslib's -1 for an unknown contig; no sequence is fetched.
        if fasta.fetch_seq_len(chrom.as_str()) == u64::MAX {
            return Err(ConversionError::Input(format!(
                "Contig '{chrom}' not found in reference FASTA"
            )));
        }
    }
    Ok(())
}

pub struct VcfRecordSource {
    inner_reader: IndexedReader,
    chrom: String,
    rid: u32,
    regions: Vec<(u32, u32)>,
    current_region: usize,
    num_samples: usize,
    ploidy: usize,
    sample_indices: Vec<usize>,
    info_fields: Vec<FieldSpec>,
    format_fields: Vec<FieldSpec>,
    record: Record,
    eof: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VcfShard {
    pub fetch_start: u32,
    pub fetch_end: u32,
    pub own_start: u32,
    pub own_end: u32,
    pub ordinal: usize,
    pub record_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VcfRegionPositions {
    pub start: u32,
    pub end: u32,
    pub positions: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VcfShardPlan {
    pub shards: Vec<VcfShard>,
    pub total_records: usize,
    pub target_records: usize,
}

pub(crate) fn coalesce_fetch_regions(
    mut regions: Vec<(u32, u32)>,
    chrom: &str,
) -> Result<Vec<(u32, u32)>, ConversionError> {
    if regions.is_empty() {
        return Ok(regions);
    }
    regions.sort_unstable_by_key(|&(start, end)| (start, end));

    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(regions.len());
    for (start, end) in regions {
        if end <= start {
            return Err(ConversionError::Input(format!(
                "Invalid fetch interval for {chrom}: start {start} must be less than end {end}"
            )));
        }
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
            continue;
        }
        merged.push((start, end));
    }
    Ok(merged)
}

pub(crate) fn plan_vcf_shards_by_variant_count(
    regions: &[VcfRegionPositions],
    worker_budget: usize,
    max_records_per_shard: usize,
) -> Result<VcfShardPlan, ConversionError> {
    for region in regions {
        if region.end <= region.start {
            return Err(ConversionError::Input(format!(
                "Invalid VCF planning interval: start {} must be less than end {}",
                region.start, region.end
            )));
        }
        if region.positions.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(ConversionError::Input(format!(
                "VCF positions are not sorted inside [{}, {})",
                region.start, region.end
            )));
        }
        if let Some(&pos) = region
            .positions
            .iter()
            .find(|&&pos| pos < region.start || pos >= region.end)
        {
            return Err(ConversionError::Input(format!(
                "VCF position {pos} falls outside planning interval [{}, {})",
                region.start, region.end
            )));
        }
    }

    let total_records: usize = regions.iter().map(|region| region.positions.len()).sum();
    let target_records = if total_records == 0 {
        1
    } else {
        total_records
            .div_ceil(worker_budget.max(1))
            .min(max_records_per_shard.max(1))
            .max(1)
    };

    let mut shards = Vec::new();
    for region in regions {
        let mut record_start = 0usize;
        let mut own_start = region.start;
        while record_start < region.positions.len() {
            let mut record_end = (record_start + target_records).min(region.positions.len());
            while record_end < region.positions.len()
                && region.positions[record_end] == region.positions[record_end - 1]
            {
                record_end += 1;
            }

            let own_end = if record_end == region.positions.len() {
                region.end
            } else {
                region.positions[record_end]
            };
            let fetch_start = if own_start == region.start {
                region.start
            } else {
                own_start
                    .saturating_sub(crate::normalize::L_MAX)
                    .max(region.start)
            };
            let fetch_end = if own_end == region.end {
                region.end
            } else {
                own_end
                    .saturating_add(crate::normalize::L_MAX)
                    .min(region.end)
            };
            shards.push(VcfShard {
                fetch_start,
                fetch_end,
                own_start,
                own_end,
                ordinal: shards.len(),
                record_count: record_end - record_start,
            });
            own_start = own_end;
            record_start = record_end;
        }
    }

    Ok(VcfShardPlan {
        shards,
        total_records,
        target_records,
    })
}

fn fetch_region(
    reader: &mut IndexedReader,
    rid: u32,
    chrom: &str,
    region: Option<(u32, u32)>,
) -> Result<(), ConversionError> {
    let (start, end, label) = match region {
        Some((start, end)) => (
            start as u64,
            Some(end as u64),
            format!("{chrom}:{start}-{end}"),
        ),
        None => (0, None, format!("chromosome '{chrom}'")),
    };
    reader
        .fetch(rid, start, end)
        .map_err(|e| ConversionError::Io {
            context: format!("fetching region for {label}"),
            source: std::io::Error::other(e.to_string()),
        })
}

fn configure_htslib_threads(
    reader: &mut IndexedReader,
    thread_count: usize,
) -> Result<(), ConversionError> {
    if thread_count == 0 {
        return Ok(());
    }
    reader
        .set_threads(thread_count)
        .map_err(|e| ConversionError::Io {
            context: format!("allocating {thread_count} HTSlib background threads"),
            source: std::io::Error::other(e.to_string()),
        })
}

pub(crate) fn scan_vcf_region_positions(
    vcf_path: &str,
    chrom: &str,
    htslib_threads: usize,
    regions: Vec<(u32, u32)>,
) -> Result<Vec<VcfRegionPositions>, ConversionError> {
    if !std::path::Path::new(vcf_path).exists() {
        return Err(ConversionError::MissingFile {
            path: vcf_path.to_string(),
        });
    }

    let mut reader = IndexedReader::from_path(vcf_path).map_err(|e| {
        ConversionError::Input(format!(
            "Failed to open VCF/BCF index for '{vcf_path}' while planning variant-count shards: {e}"
        ))
    })?;
    configure_htslib_threads(&mut reader, htslib_threads)?;
    let rid = reader.header().name2rid(chrom.as_bytes()).map_err(|_| {
        ConversionError::ContigNotInHeader {
            chrom: chrom.to_string(),
        }
    })?;

    let requested = coalesce_fetch_regions(regions, chrom)?;
    let scan_regions = if requested.is_empty() {
        vec![(0, u32::MAX)]
    } else {
        requested
    };
    let whole_contig = scan_regions.len() == 1 && scan_regions[0] == (0, u32::MAX);
    let mut planned = Vec::with_capacity(scan_regions.len());
    for (start, end) in scan_regions {
        fetch_region(
            &mut reader,
            rid,
            chrom,
            if whole_contig {
                None
            } else {
                Some((start, end))
            },
        )?;
        let mut record = reader.empty_record();
        let mut positions = Vec::new();
        loop {
            match reader.read(&mut record) {
                None => break,
                Some(Err(e)) => {
                    return Err(ConversionError::Io {
                        context: format!("scanning VCF positions for {chrom}:[{start}, {end})"),
                        source: std::io::Error::other(e.to_string()),
                    });
                }
                Some(Ok(())) => {
                    let pos = record.pos() as u32;
                    if pos >= start && pos < end {
                        positions.push(pos);
                    }
                }
            }
        }
        planned.push(VcfRegionPositions {
            start,
            end,
            positions,
        });
    }
    Ok(planned)
}

impl VcfRecordSource {
    // opens the file, applies the sample filter at the C-level, and jumps to the chromosome.
    pub fn new(
        vcf_path: &str,
        chrom: &str,
        samples: &[&str],
        htslib_threads: usize,
        ploidy: usize,
        fields: &[FieldSpec],
        regions: Vec<(u32, u32)>,
    ) -> Result<Self, ConversionError> {
        // A wrong VCF path reaches Rust when the file was removed after Python's
        // upstream indexing/open; surface it as FileNotFoundError, not a ".tbi?" Input.
        if !std::path::Path::new(vcf_path).exists() {
            return Err(ConversionError::MissingFile {
                path: vcf_path.to_string(),
            });
        }

        let mut reader = IndexedReader::from_path(vcf_path).map_err(|e| {
            ConversionError::Input(format!(
                "Failed to open VCF/BCF index for '{vcf_path}' \
                 (is there a .tbi or .csi file?): {e}"
            ))
        })?;

        configure_htslib_threads(&mut reader, htslib_threads)?;

        let header = reader.header().clone();

        let rid =
            header
                .name2rid(chrom.as_bytes())
                .map_err(|_| ConversionError::ContigNotInHeader {
                    chrom: chrom.to_string(),
                })?;

        let regions = coalesce_fetch_regions(regions, chrom)?;
        fetch_region(&mut reader, rid, chrom, regions.first().copied())?;

        let sample_indices: Vec<usize> = samples
            .iter()
            .map(|name| {
                header.sample_id(name.as_bytes()).ok_or_else(|| {
                    ConversionError::Input(format!("Sample '{name}' not found in VCF"))
                })
            })
            .collect::<Result<_, _>>()?;

        let record = reader.empty_record();

        let info_fields: Vec<FieldSpec> = fields
            .iter()
            .filter(|f| f.category == FieldCategory::Info)
            .cloned()
            .collect();
        let format_fields: Vec<FieldSpec> = fields
            .iter()
            .filter(|f| f.category == FieldCategory::Format)
            .cloned()
            .collect();

        Ok(Self {
            inner_reader: reader,
            chrom: chrom.to_string(),
            rid,
            regions,
            current_region: 0,
            num_samples: samples.len(),
            ploidy,
            sample_indices,
            info_fields,
            format_fields,
            record,
            eof: false,
        })
    }

    fn active_region(&self) -> Option<(u32, u32)> {
        self.regions.get(self.current_region).copied()
    }

    fn advance_region(&mut self) -> Result<bool, ConversionError> {
        if self.regions.is_empty() || self.current_region + 1 >= self.regions.len() {
            return Ok(false);
        }
        self.current_region += 1;
        fetch_region(
            &mut self.inner_reader,
            self.rid,
            &self.chrom,
            self.regions.get(self.current_region).copied(),
        )?;
        Ok(true)
    }
}

impl RecordSource for VcfRecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        if self.eof {
            return Ok(None);
        }
        loop {
            match self.inner_reader.read(&mut self.record) {
                None => {
                    if self.advance_region()? {
                        continue;
                    }
                    self.eof = true;
                    return Ok(None);
                }
                Some(Err(e)) => {
                    return Err(ConversionError::Io {
                        context: "reading next VCF record".to_string(),
                        source: std::io::Error::other(e.to_string()),
                    });
                }
                Some(Ok(())) => {}
            }

            let pos = self.record.pos() as u32;
            if let Some((start, end)) = self.active_region()
                && (pos < start || pos >= end)
            {
                continue;
            }

            return self.record_to_raw(pos);
        }
    }
}

impl VcfRecordSource {
    fn record_to_raw(&self, pos: u32) -> Result<Option<RawRecord>, ConversionError> {
        let reference: Vec<u8>;
        let alts: Vec<Vec<u8>>;
        {
            let alleles = self.record.alleles();
            reference = alleles[0].to_vec();
            alts = alleles[1..].iter().map(|a| a.to_vec()).collect();
        }

        // Decode GT straight from the raw BCF integer buffer instead of
        // `record.genotypes().get(i)`, which allocates a per-sample
        // `Genotype(Vec<GenotypeAllele>)` for every sample of every record -- the
        // dominant reader-side allocation churn. BCF GT encoding: an allele is
        // `(idx + 1) << 1 | phased`, so `e >= 2` decodes to `(e >> 1) - 1`; `e` of
        // 0/1 is missing and `i32::MIN` is vector-end padding -- both `< 2`, so a
        // single `e >= 2` test reproduces `GenotypeAllele::index()` exactly.
        let columns = self.num_samples * self.ploidy;
        let mut gt = vec![-1i32; columns];
        {
            let gts = self.record.format(b"GT").integer().map_err(|e| {
                ConversionError::Input(format!("Failed to read GT format at pos {pos}: {e}"))
            })?;
            let ploidy = self.ploidy;
            for (s_idx, &vcf_idx) in self.sample_indices.iter().enumerate() {
                let raw = gts[vcf_idx];
                let base = s_idx * ploidy;
                for p in 0..ploidy {
                    gt[base + p] = match raw.get(p) {
                        Some(&e) if e >= 2 => (e >> 1) - 1,
                        _ => -1,
                    };
                }
            }
        }

        let info_raw: Vec<Option<Vec<f64>>> = self
            .info_fields
            .iter()
            .map(|spec| decode_info_raw(&self.record, spec))
            .collect::<Result<_, _>>()?;

        // Remap htslib's header-sample-indexed FORMAT buffers into SELECTED-sample
        // order, so the assembler never needs to know about `sample_indices`.
        let format_raw: Vec<Option<Vec<Vec<f64>>>> = self
            .format_fields
            .iter()
            .map(|spec| {
                decode_format_raw(&self.record, spec).map(|opt| {
                    opt.map(|per_header_sample| {
                        self.sample_indices
                            .iter()
                            .map(|&vcf_idx| per_header_sample[vcf_idx].clone())
                            .collect()
                    })
                })
            })
            .collect::<Result<_, _>>()?;

        Ok(Some(RawRecord {
            pos,
            reference,
            alts,
            gt,
            info_raw,
            format_raw,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_count_shards_balance_dense_and_sparse_coordinates() {
        let regions = vec![
            VcfRegionPositions {
                start: 0,
                end: 100,
                positions: vec![1, 2, 3, 4],
            },
            VcfRegionPositions {
                start: 1_000_000,
                end: 2_000_000,
                positions: vec![1_100_000, 1_300_000, 1_600_000, 1_900_000],
            },
        ];

        let plan = plan_vcf_shards_by_variant_count(&regions, 4, 25_000).unwrap();
        assert_eq!(plan.total_records, 8);
        assert_eq!(plan.target_records, 2);
        assert_eq!(
            plan.shards
                .iter()
                .map(|shard| shard.record_count)
                .collect::<Vec<_>>(),
            vec![2, 2, 2, 2]
        );
    }

    #[test]
    fn variant_count_shards_do_not_split_equal_position_records() {
        let regions = vec![VcfRegionPositions {
            start: 0,
            end: 100,
            positions: vec![10, 10, 10, 20, 30, 40],
        }];

        let plan = plan_vcf_shards_by_variant_count(&regions, 3, 2).unwrap();
        assert_eq!(
            plan.shards
                .iter()
                .map(|shard| shard.record_count)
                .collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
        assert_eq!(
            plan.shards
                .iter()
                .map(|shard| (shard.own_start, shard.own_end))
                .collect::<Vec<_>>(),
            vec![(0, 20), (20, 40), (40, 100)]
        );
    }

    #[test]
    fn variant_count_shard_padding_stays_inside_requested_region() {
        let regions = vec![VcfRegionPositions {
            start: 100,
            end: 200,
            positions: vec![110, 120, 130, 140],
        }];

        let plan = plan_vcf_shards_by_variant_count(&regions, 2, 2).unwrap();
        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].fetch_start, 100);
        assert_eq!(plan.shards[1].fetch_end, 200);
        assert!(plan.shards[0].fetch_end >= plan.shards[1].own_start);
        assert!(plan.shards[1].fetch_start <= plan.shards[0].own_end);
    }
}

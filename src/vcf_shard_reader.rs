//! Reader pieces for position-sharded, multi-sample VCF cohorts.
//!
//! `vcf_list_reader` is sample-sharded (one sample per file). This module is
//! the orthogonal shape: every file has the SAME cohort columns and owns
//! disjoint raw-POS intervals. The orchestrator plans padded normalized-POS
//! work units; each unit uses this source to concatenate only the source-owned
//! raw slices intersecting its padded fetch window. A local `ChunkAssembler`
//! then normalizes those records and keeps atoms in the unit's owned range,
//! exactly like single-file VCF sub-contig sharding.

use std::collections::VecDeque;

use crate::error::ConversionError;
use crate::field::FieldSpec;
use crate::record_source::{RawRecord, RecordSource};
use crate::shard::WorkUnit;
use crate::svar2_view::OverlapMode;
use crate::vcf_reader::VcfRecordSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VcfCohortShardUnit {
    pub path_index: usize,
    /// Raw-POS source ownership, 0-based half-open.
    pub start: u32,
    pub end: u32,
    pub ordinal: usize,
}

/// Turn source-owned intervals into normalized-POS work units.
///
/// Consecutive intervals from one physical source are one run. The boundary
/// between runs is the next run's first raw position; gaps therefore belong
/// to the run on their left. Padding lets both neighboring workers see raw
/// records that normalize across that boundary, while `owned_range` keeps
/// each normalized atom in exactly one output unit.
pub fn plan_work_units(
    owned: &[VcfCohortShardUnit],
    pad: u32,
) -> Result<Vec<WorkUnit>, ConversionError> {
    if owned.is_empty() {
        return Err(ConversionError::Input(
            "position-sharded VCF conversion has no owned intervals".to_string(),
        ));
    }
    for (expected, unit) in owned.iter().enumerate() {
        if unit.ordinal != expected {
            return Err(ConversionError::Input(format!(
                "position-sharded VCF intervals must have dense ordinals; expected {expected}, got {}",
                unit.ordinal
            )));
        }
        if unit.start >= unit.end {
            return Err(ConversionError::Input(format!(
                "invalid position-sharded VCF ownership interval {}-{}",
                unit.start, unit.end
            )));
        }
        if let Some(previous) = expected.checked_sub(1).map(|i| owned[i])
            && unit.start < previous.end
        {
            return Err(ConversionError::Input(format!(
                "position-sharded VCF ownership intervals overlap: {}-{} and {}-{}",
                previous.start, previous.end, unit.start, unit.end
            )));
        }
    }

    let mut run_starts = vec![owned[0].start];
    for pair in owned.windows(2) {
        if pair[0].path_index != pair[1].path_index {
            run_starts.push(pair[1].start);
        }
    }

    Ok(run_starts
        .iter()
        .enumerate()
        .map(|(ordinal, &run_start)| {
            let own_start = if ordinal == 0 { 0 } else { run_start };
            let own_end = run_starts.get(ordinal + 1).copied().unwrap_or(u32::MAX);
            WorkUnit {
                own_start,
                own_end,
                fetch_start: own_start.saturating_sub(pad),
                fetch_end: own_end.saturating_add(pad),
                ordinal,
            }
        })
        .collect())
}

/// Consecutive source slices for one padded work-unit fetch window. Adjacent
/// intervals owned by the same physical file are grouped so one indexed reader
/// can seek through all of them without reparsing the large cohort header.
pub fn slices_for_fetch(
    owned: &[VcfCohortShardUnit],
    fetch_start: u32,
    fetch_end: u32,
) -> Vec<(usize, Vec<(u32, u32)>)> {
    let mut groups: Vec<(usize, Vec<(u32, u32)>)> = Vec::new();
    for shard in owned {
        let start = shard.start.max(fetch_start);
        let end = shard.end.min(fetch_end);
        if start >= end {
            continue;
        }
        if let Some((path_index, regions)) = groups.last_mut()
            && *path_index == shard.path_index
        {
            regions.push((start, end));
        } else {
            groups.push((shard.path_index, vec![(start, end)]));
        }
    }
    groups
}

/// Concatenates globally ordered source-owned VCF slices. Readers for the
/// small number of physical sources touched by one padded work unit are opened
/// up front; only one is consumed at a time. Parallelism comes from
/// `shard_exec` running independent padded work units concurrently.
pub struct VcfCohortSliceRecordSource {
    readers: VecDeque<VcfRecordSource>,
    current: Option<VcfRecordSource>,
}

impl VcfCohortSliceRecordSource {
    pub fn new(
        vcf_paths: &[String],
        slices: Vec<(usize, Vec<(u32, u32)>)>,
        chrom: &str,
        samples: &[&str],
        ploidy: usize,
        fields: &[FieldSpec],
    ) -> Result<Self, ConversionError> {
        if slices.is_empty() {
            return Err(ConversionError::Input(format!(
                "position-sharded VCF fetch has no source-owned slices for {chrom}"
            )));
        }
        if let Some((path_index, _)) = slices
            .iter()
            .find(|(path_index, _)| *path_index >= vcf_paths.len())
        {
            return Err(ConversionError::Input(format!(
                "position-sharded VCF slice references missing path index {path_index}"
            )));
        }
        let readers = slices
            .into_iter()
            .map(|(path_index, regions)| {
                VcfRecordSource::new(
                    &vcf_paths[path_index],
                    chrom,
                    samples,
                    1,
                    ploidy,
                    fields,
                    regions,
                    OverlapMode::Pos,
                )
            })
            .collect::<Result<VecDeque<_>, _>>()?;
        Ok(Self {
            readers,
            current: None,
        })
    }
}

impl RecordSource for VcfCohortSliceRecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        loop {
            if self.current.is_none() {
                self.current = self.readers.pop_front();
                if self.current.is_none() {
                    return Ok(None);
                }
            }
            match self
                .current
                .as_mut()
                .expect("current reader opened")
                .next_record()?
            {
                Some(record) => return Ok(Some(record)),
                None => self.current = None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_slices_intersect_and_group_consecutive_paths() {
        let owned = vec![
            VcfCohortShardUnit {
                path_index: 0,
                start: 0,
                end: 10,
                ordinal: 0,
            },
            VcfCohortShardUnit {
                path_index: 0,
                start: 20,
                end: 30,
                ordinal: 1,
            },
            VcfCohortShardUnit {
                path_index: 1,
                start: 40,
                end: 50,
                ordinal: 2,
            },
        ];
        assert_eq!(
            slices_for_fetch(&owned, 5, 45),
            vec![(0, vec![(5, 10), (20, 30)]), (1, vec![(40, 45)])]
        );
    }

    #[test]
    fn work_units_group_source_runs_and_assign_gaps_left() {
        let owned = vec![
            VcfCohortShardUnit {
                path_index: 0,
                start: 100,
                end: 120,
                ordinal: 0,
            },
            VcfCohortShardUnit {
                path_index: 0,
                start: 140,
                end: 160,
                ordinal: 1,
            },
            VcfCohortShardUnit {
                path_index: 1,
                start: 200,
                end: 220,
                ordinal: 2,
            },
        ];
        assert_eq!(
            plan_work_units(&owned, 10).unwrap(),
            vec![
                WorkUnit {
                    own_start: 0,
                    own_end: 200,
                    fetch_start: 0,
                    fetch_end: 210,
                    ordinal: 0,
                },
                WorkUnit {
                    own_start: 200,
                    own_end: u32::MAX,
                    fetch_start: 190,
                    fetch_end: u32::MAX,
                    ordinal: 1,
                },
            ]
        );
    }

    #[test]
    fn work_unit_planner_rejects_overlap() {
        let owned = vec![
            VcfCohortShardUnit {
                path_index: 0,
                start: 10,
                end: 20,
                ordinal: 0,
            },
            VcfCohortShardUnit {
                path_index: 1,
                start: 19,
                end: 30,
                ordinal: 1,
            },
        ];
        assert!(plan_work_units(&owned, 10).is_err());
    }
}

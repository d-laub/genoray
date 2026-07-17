//! SVAR1 record source: reconstruct variant-major `RawRecord`s from SVAR1's
//! sample-major sparse store, so `from_svar1` reuses the shared conversion spine
//! (`chunk_assembler` onward) exactly as VCF/PGEN do. See
//! `docs/superpowers/specs/2026-07-13-svar1-to-svar2-conversion-design.md`.

/// Invert SVAR1's sample-major CSR (`variant_idxs`/`offsets`) into a variant-major
/// carrier list for ONE contig. Returns, per local variant `0..n_local`, the
/// `(haplotype column, flat entry index)` pairs of the haplotypes carrying it.
///
/// `variant_idxs` holds each haplotype's sorted global non-ref variant ids;
/// `offsets` is the CSR over `num_haps = num_samples * ploidy` haplotypes
/// (`offsets.len() == num_haps + 1`). Contigs are contiguous in global-id space,
/// so this contig owns global ids `[contig_start, contig_start + n_local)`; per
/// hap we binary-search that sub-range (ids are sorted) rather than scanning all
/// entries. `flat entry index` indexes both `variant_idxs` and any per-entry
/// field array (they share `offsets`).
pub fn build_variant_major(
    variant_idxs: &[i32],
    offsets: &[i64],
    num_haps: usize,
    contig_start: i32,
    n_local: usize,
) -> Vec<Vec<(u32, u64)>> {
    let contig_end = contig_start + n_local as i32;
    let mut buckets: Vec<Vec<(u32, u64)>> = vec![Vec::new(); n_local];
    for h in 0..num_haps {
        let lo = offsets[h] as usize;
        let hi = offsets[h + 1] as usize;
        let hap = &variant_idxs[lo..hi];
        let s = hap.partition_point(|&g| g < contig_start);
        let e = hap.partition_point(|&g| g < contig_end);
        for (k, &g) in hap.iter().enumerate().take(e).skip(s) {
            let local = (g - contig_start) as usize;
            buckets[local].push((h as u32, (lo + k) as u64));
        }
    }
    buckets
}

use crate::error::ConversionError;
use crate::field::FieldSpec;
use crate::record_source::{RawRecord, RecordSource};
use crate::svar2_view::{OverlapMode, extent_overlaps, keeps};
use memmap2::Mmap;
use std::fs::File;

/// A per-entry field array, mmap'd raw and read as f64 on demand.
enum FieldArray {
    F32(Mmap),
    F16(Mmap),
    I8(Mmap),
    I16(Mmap),
    I32(Mmap),
    U8(Mmap),
    U16(Mmap),
    U32(Mmap),
}

impl FieldArray {
    fn open(path: &std::path::Path, np_dtype: &str) -> Result<Self, ConversionError> {
        let mmap = mmap_ro(path)?;
        Ok(match np_dtype {
            "float32" => FieldArray::F32(mmap),
            "float16" => FieldArray::F16(mmap),
            "int8" => FieldArray::I8(mmap),
            "int16" => FieldArray::I16(mmap),
            "int32" => FieldArray::I32(mmap),
            "uint8" => FieldArray::U8(mmap),
            "uint16" => FieldArray::U16(mmap),
            "uint32" => FieldArray::U32(mmap),
            other => {
                return Err(ConversionError::Input(format!(
                    "SVAR1 field dtype {other:?} is unsupported for conversion"
                )));
            }
        })
    }

    fn value_f64(&self, entry: usize) -> f64 {
        match self {
            FieldArray::F32(m) => bytemuck::cast_slice::<u8, f32>(m)[entry] as f64,
            FieldArray::F16(m) => {
                let bytes = [m[entry * 2], m[entry * 2 + 1]];
                f64::from(half::f16::from_le_bytes(bytes))
            }
            FieldArray::I8(m) => bytemuck::cast_slice::<u8, i8>(m)[entry] as f64,
            FieldArray::I16(m) => bytemuck::cast_slice::<u8, i16>(m)[entry] as f64,
            FieldArray::I32(m) => bytemuck::cast_slice::<u8, i32>(m)[entry] as f64,
            FieldArray::U8(m) => m[entry] as f64,
            FieldArray::U16(m) => bytemuck::cast_slice::<u8, u16>(m)[entry] as f64,
            FieldArray::U32(m) => bytemuck::cast_slice::<u8, u32>(m)[entry] as f64,
        }
    }
}

fn mmap_ro(path: &std::path::Path) -> Result<Mmap, ConversionError> {
    let f = File::open(path).map_err(|e| ConversionError::Io {
        context: format!("open {path:?}"),
        source: e,
    })?;
    // SAFETY: read-only map of a file we do not mutate for the source's lifetime.
    unsafe { Mmap::map(&f) }.map_err(|e| ConversionError::Io {
        context: format!("mmap {path:?}"),
        source: e,
    })
}

pub struct Svar1RecordSource {
    num_haps: usize,
    ploidy: usize,
    num_samples: usize,
    pos: Vec<u32>,
    ref_bytes: Vec<u8>,
    ref_offsets: Vec<i64>,
    alt_bytes: Vec<u8>,
    alt_offsets: Vec<i64>,
    buckets: Vec<Vec<(u32, u64)>>, // per local variant -> (OUTPUT hap col, entry idx)
    fields: Vec<(FieldSpec, FieldArray)>,
    cursor: usize,
    n_local: usize,
    /// Sorted, non-overlapping, 0-based half-open query intervals for this
    /// contig. Empty means "no region filter" -- every record is kept
    /// (byte-identical to full-contig conversion).
    regions: Vec<(u32, u32)>,
    /// How a record's overlap with `regions[cur_region]` is judged.
    overlap: OverlapMode,
    /// Index into `regions` of the earliest region that might still match an
    /// upcoming (higher-POS) record. Advances monotonically as `pos[v]`
    /// passes each region's end -- see `next_record`.
    cur_region: usize,
}

impl Svar1RecordSource {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        svar1_dir: &str,
        contig_start: usize,
        n_local: usize,
        num_samples: usize,
        ploidy: usize,
        pos: Vec<u32>,
        ref_bytes: Vec<u8>,
        ref_offsets: Vec<i64>,
        alt_bytes: Vec<u8>,
        alt_offsets: Vec<i64>,
        format_fields: &[FieldSpec],
        format_src_dtypes: &[String],
        regions: Vec<(u32, u32)>,
        overlap: OverlapMode,
        sample_idx: Vec<usize>,
    ) -> Result<Self, ConversionError> {
        let dir = std::path::Path::new(svar1_dir);

        let vi_mmap = mmap_ro(&dir.join("variant_idxs.npy"))?;
        let off_mmap = mmap_ro(&dir.join("offsets.npy"))?;
        let variant_idxs: &[i32] = bytemuck::cast_slice(&vi_mmap);
        let offsets: &[i64] = bytemuck::cast_slice(&off_mmap);
        // `offsets.npy` is GLOBAL over every original haplotype in the SVAR1
        // store -- NOT `num_samples * ploidy`, since `num_samples` here is the
        // OUTPUT/subset column count once `samples=` narrows the cohort. The
        // original hap count is derived from the CSR itself.
        if offsets.is_empty() || !(offsets.len() - 1).is_multiple_of(ploidy) {
            return Err(ConversionError::Input(format!(
                "SVAR1 offsets.npy has {} entries; expected 1 + a multiple of ploidy ({})",
                offsets.len(),
                ploidy
            )));
        }
        let orig_num_haps = offsets.len() - 1;
        let orig_num_samples = orig_num_haps / ploidy;
        for &s in &sample_idx {
            if s >= orig_num_samples {
                return Err(ConversionError::Input(format!(
                    "sample index {s} out of range for a SVAR1 store with {orig_num_samples} samples"
                )));
            }
        }
        if sample_idx.len() != num_samples {
            return Err(ConversionError::Input(format!(
                "sample_idx has {} entries; expected num_samples = {num_samples}",
                sample_idx.len()
            )));
        }

        let raw_buckets = build_variant_major(
            variant_idxs,
            offsets,
            orig_num_haps,
            contig_start as i32,
            n_local,
        );

        // Remap ORIGINAL hap columns -> OUTPUT hap columns per `sample_idx`
        // (caller/output order): `sample_idx[out_s]` is the original sample
        // index that output sample `out_s` reads from. A carrier on a
        // hap belonging to a sample NOT selected is dropped. Keeping this a
        // post-transpose remap (rather than threading `sample_idx` into
        // `build_variant_major` itself) leaves that function and its
        // existing unit tests over the ORIGINAL hap space untouched.
        let mut orig_hap_to_out: Vec<Option<u32>> = vec![None; orig_num_haps];
        for (out_s, &orig_s) in sample_idx.iter().enumerate() {
            for p in 0..ploidy {
                orig_hap_to_out[orig_s * ploidy + p] = Some((out_s * ploidy + p) as u32);
            }
        }
        let buckets: Vec<Vec<(u32, u64)>> = raw_buckets
            .into_iter()
            .map(|bucket| {
                bucket
                    .into_iter()
                    .filter_map(|(col, e)| orig_hap_to_out[col as usize].map(|oc| (oc, e)))
                    .collect()
            })
            .collect();

        let num_haps = num_samples * ploidy;

        let mut fields = Vec::with_capacity(format_fields.len());
        for (spec, np_dtype) in format_fields.iter().zip(format_src_dtypes) {
            let arr = FieldArray::open(&dir.join(format!("{}.npy", spec.name)), np_dtype)?;
            fields.push((spec.clone(), arr));
        }

        Ok(Self {
            num_haps,
            ploidy,
            num_samples,
            pos,
            ref_bytes,
            ref_offsets,
            alt_bytes,
            alt_offsets,
            buckets,
            fields,
            cursor: 0,
            n_local,
            regions,
            overlap,
            cur_region: 0,
        })
    }
}

impl RecordSource for Svar1RecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        loop {
            let v = self.cursor;
            if v >= self.n_local {
                return Ok(None);
            }
            self.cursor += 1;

            if !self.regions.is_empty() {
                let pos = self.pos[v];
                // Advance past regions `pos` has already moved beyond -- regions
                // are sorted and non-overlapping, and POS is non-decreasing
                // within a contig, so this cursor only ever moves forward.
                loop {
                    let Some(&(_, q_end)) = self.regions.get(self.cur_region) else {
                        // Every region has been exhausted; no later record
                        // (POS only increases) can match anything either.
                        return Ok(None);
                    };
                    let past_end = match self.overlap {
                        OverlapMode::Record => pos > q_end,
                        OverlapMode::Pos | OverlapMode::Variant => pos >= q_end,
                    };
                    if past_end {
                        self.cur_region += 1;
                        continue;
                    }
                    break;
                }
                let (q_start, q_end) = self.regions[self.cur_region];
                let r0 = self.ref_offsets[v] as usize;
                let r1 = self.ref_offsets[v + 1] as usize;
                let a0 = self.alt_offsets[v] as usize;
                let a1 = self.alt_offsets[v + 1] as usize;
                let kept = match self.overlap {
                    OverlapMode::Variant => extent_overlaps(
                        pos,
                        (r1 - r0) as u32,
                        &[&self.alt_bytes[a0..a1]],
                        &self.ref_bytes[r0..r1],
                        q_start,
                        q_end,
                    ),
                    m => keeps(m, q_start, q_end, pos),
                };
                if !kept {
                    continue;
                }
            }

            let reference = self.ref_bytes
                [self.ref_offsets[v] as usize..self.ref_offsets[v + 1] as usize]
                .to_vec();
            let alt = self.alt_bytes
                [self.alt_offsets[v] as usize..self.alt_offsets[v + 1] as usize]
                .to_vec();

            let mut gt = vec![0i32; self.num_haps];
            for &(col, _e) in &self.buckets[v] {
                gt[col as usize] = 1; // biallelic: ALT1
            }

            let format_raw = self
                .fields
                .iter()
                .map(|(spec, arr)| {
                    let sent = spec.missing_sentinel();
                    let mut per_sample: Vec<Vec<f64>> = vec![vec![sent]; self.num_samples];
                    for &(col, e) in &self.buckets[v] {
                        let s = col as usize / self.ploidy;
                        per_sample[s] = vec![arr.value_f64(e as usize)];
                    }
                    Some(per_sample)
                })
                .collect();

            return Ok(Some(RawRecord {
                pos: self.pos[v],
                reference,
                alts: vec![alt],
                calls: crate::record_source::Calls::Dense(gt),
                info_raw: Vec::new(), // SVAR1 has no INFO fields
                format_vals: crate::record_source::FormatVals::Dense(format_raw),
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record_source::FormatVals;

    // 2 samples × ploidy 2 = 4 haplotypes. Global variant ids 0..5.
    // Contig under test starts at global id 2, has 3 local variants (ids 2,3,4).
    // Per-hap sorted global ids (offsets CSR):
    //   hap0: [0, 2, 4]   hap1: [3]   hap2: [2]   hap3: []
    #[test]
    fn transpose_buckets_carriers_by_local_variant() {
        let variant_idxs: Vec<i32> = vec![0, 2, 4, /*h0*/ 3, /*h1*/ 2 /*h2*/];
        let offsets: Vec<i64> = vec![0, 3, 4, 5, 5]; // len num_haps+1 = 5
        let got = build_variant_major(&variant_idxs, &offsets, 4, 2, 3);

        // local 0 (gid 2): hap0 at entry 1, hap2 at entry 4
        assert_eq!(got[0], vec![(0u32, 1u64), (2u32, 4u64)]);
        // local 1 (gid 3): hap1 at entry 3
        assert_eq!(got[1], vec![(1u32, 3u64)]);
        // local 2 (gid 4): hap0 at entry 2
        assert_eq!(got[2], vec![(0u32, 2u64)]);
    }

    #[test]
    fn transpose_empty_contig_is_all_empty() {
        let got = build_variant_major(&[0, 1], &[0, 1, 2, 2, 2], 4, 100, 2);
        assert_eq!(got, vec![Vec::new(), Vec::new()]);
    }

    use std::io::Write;

    fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(bytemuck::cast_slice(data)).unwrap();
    }

    #[test]
    fn record_source_yields_variant_major_records_with_dosage() {
        // 2 samples, ploidy 2 (4 haps). One contig, global ids == local ids 0..2.
        //   var0 (gid0): carried by hap0 (S0 hap0) and hap2 (S1 hap0)
        //   var1 (gid1): carried by hap3 (S1 hap1)
        // Per-hap CSR: hap0:[0] hap1:[] hap2:[0] hap3:[1]
        let tmp = std::env::temp_dir().join(format!("svar1_rs_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        write_raw::<i32>(&tmp, "variant_idxs.npy", &[0, 0, 1]);
        write_raw::<i64>(&tmp, "offsets.npy", &[0, 1, 1, 2, 3]);
        // dosages aligned 1:1 with variant_idxs entries: entry0=hap0(var0)->0.5,
        // entry1=hap2(var0)->2.5, entry2=hap3(var1)->1.5
        write_raw::<f32>(&tmp, "dosages.npy", &[0.5, 2.5, 1.5]);

        let ds = crate::field::FieldSpec {
            name: "dosages".into(),
            category: crate::field::FieldCategory::Format,
            htype: crate::field::HtslibType::Float,
            dtype: crate::field::StorageDtype::F32,
            default: None,
        };
        let mut src = Svar1RecordSource::new(
            tmp.to_str().unwrap(),
            0,
            2,
            2,
            2,
            vec![10, 20], // pos (0-based) for var0, var1
            b"AC".to_vec(),
            vec![0, 1, 2], // REF: "A","C"
            b"GT".to_vec(),
            vec![0, 1, 2], // ALT: "G","T"
            std::slice::from_ref(&ds),
            &["float32".to_string()],
            Vec::new(),       // regions: empty -> no filter (byte-compat)
            OverlapMode::Pos, // irrelevant when regions is empty
            vec![0, 1],       // sample_idx: identity (no subset/reorder)
        )
        .unwrap();

        let r0 = src.next_record().unwrap().unwrap();
        assert_eq!(r0.pos, 10);
        assert_eq!(r0.reference, b"A");
        assert_eq!(r0.alts, vec![b"G".to_vec()]);
        assert_eq!(
            r0.calls,
            crate::record_source::Calls::Dense(vec![1, 0, 1, 0])
        ); // hap0 & hap2 carry ALT1
        // format_vals[0] = Some(per-sample). S0 carried on hap0 -> 0.5; S1 on hap2 -> 2.5
        let FormatVals::Dense(fv0) = &r0.format_vals else {
            panic!("SVAR1 source must produce Dense FormatVals")
        };
        let ds0 = fv0[0].as_ref().unwrap();
        assert_eq!(ds0[0], vec![0.5]);
        assert_eq!(ds0[1], vec![2.5]);

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(
            r1.calls,
            crate::record_source::Calls::Dense(vec![0, 0, 0, 1])
        ); // only S1 hap1
        let FormatVals::Dense(fv1) = &r1.format_vals else {
            panic!("SVAR1 source must produce Dense FormatVals")
        };
        let ds1 = fv1[0].as_ref().unwrap();
        assert!(ds1[0][0].is_nan()); // S0 non-carrier -> missing sentinel (NaN)
        assert_eq!(ds1[1], vec![1.5]); // S1 carrier

        assert!(src.next_record().unwrap().is_none());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn sample_remap_reorders_and_drops_hap_columns() {
        // 3 ORIGINAL samples x ploidy 2 = 6 haps: hap0,1=S0 hap2,3=S1 hap4,5=S2.
        // One contig, global ids == local ids 0..2.
        //   var0 (gid0): carried by hap0 (S0h0), hap3 (S1h1), hap4 (S2h0)
        //   var1 (gid1): carried by hap5 (S2h1) only
        // Per-hap CSR: hap0:[0] hap1:[] hap2:[] hap3:[0] hap4:[0] hap5:[1]
        let tmp = std::env::temp_dir().join(format!("svar1_rs_remap_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        write_raw::<i32>(&tmp, "variant_idxs.npy", &[0, 0, 0, 1]);
        write_raw::<i64>(&tmp, "offsets.npy", &[0, 1, 1, 1, 2, 3, 4]); // len orig_num_haps+1 = 7

        // sample_idx = [1, 0]: output sample 0 = orig S1, output sample 1 = orig
        // S0 -- orig S2 is NOT selected and must vanish entirely.
        let src = Svar1RecordSource::new(
            tmp.to_str().unwrap(),
            0,
            2,
            2, // num_samples: OUTPUT/subset count
            2, // ploidy
            vec![10, 20],
            b"AC".to_vec(),
            vec![0, 1, 2],
            b"GT".to_vec(),
            vec![0, 1, 2],
            &[],
            &[],
            Vec::new(),
            OverlapMode::Pos,
            vec![1, 0],
        )
        .unwrap();

        // var0 (gid0): orig hap0 (S0h0) -> output hap2 (out S1 h0); orig hap3
        // (S1h1) -> output hap1 (out S0 h1); orig hap4 (S2h0) -> dropped.
        assert_eq!(src.buckets[0], vec![(2u32, 0u64), (1u32, 1u64)]);
        // var1 (gid1): orig hap5 (S2h1) -> dropped entirely (S2 not selected).
        assert_eq!(src.buckets[1], Vec::new());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn region_filter_skips_records_outside_query_window() {
        // Same 2-sample/2-variant fixture as
        // `record_source_yields_variant_major_records_with_dosage`, but with a
        // region restricted to [0, 15) (POS mode) -- only var0 (pos 10) should
        // survive; var1 (pos 20) is filtered and the source ends immediately
        // (POS only increases, so no later record can match either).
        let tmp = std::env::temp_dir().join(format!("svar1_rs_region_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        write_raw::<i32>(&tmp, "variant_idxs.npy", &[0, 0, 1]);
        write_raw::<i64>(&tmp, "offsets.npy", &[0, 1, 1, 2, 3]);

        let mut src = Svar1RecordSource::new(
            tmp.to_str().unwrap(),
            0,
            2,
            2,
            2,
            vec![10, 20],
            b"AC".to_vec(),
            vec![0, 1, 2],
            b"GT".to_vec(),
            vec![0, 1, 2],
            &[],
            &[],
            vec![(0, 15)],
            OverlapMode::Pos,
            vec![0, 1],
        )
        .unwrap();

        let r0 = src.next_record().unwrap().unwrap();
        assert_eq!(r0.pos, 10);
        assert!(src.next_record().unwrap().is_none());
        std::fs::remove_dir_all(&tmp).ok();
    }
}

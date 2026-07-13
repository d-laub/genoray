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
    buckets: Vec<Vec<(u32, u64)>>, // per local variant -> (hap col, entry idx)
    fields: Vec<(FieldSpec, FieldArray)>,
    cursor: usize,
    n_local: usize,
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
    ) -> Result<Self, ConversionError> {
        let dir = std::path::Path::new(svar1_dir);
        let num_haps = num_samples * ploidy;

        let vi_mmap = mmap_ro(&dir.join("variant_idxs.npy"))?;
        let off_mmap = mmap_ro(&dir.join("offsets.npy"))?;
        let variant_idxs: &[i32] = bytemuck::cast_slice(&vi_mmap);
        let offsets: &[i64] = bytemuck::cast_slice(&off_mmap);
        if offsets.len() != num_haps + 1 {
            return Err(ConversionError::Input(format!(
                "SVAR1 offsets.npy has {} entries; expected num_samples*ploidy+1 = {}",
                offsets.len(),
                num_haps + 1
            )));
        }
        let buckets = build_variant_major(
            variant_idxs,
            offsets,
            num_haps,
            contig_start as i32,
            n_local,
        );

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
        })
    }
}

impl RecordSource for Svar1RecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        let v = self.cursor;
        if v >= self.n_local {
            return Ok(None);
        }
        self.cursor += 1;

        let reference =
            self.ref_bytes[self.ref_offsets[v] as usize..self.ref_offsets[v + 1] as usize].to_vec();
        let alt =
            self.alt_bytes[self.alt_offsets[v] as usize..self.alt_offsets[v + 1] as usize].to_vec();

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

        Ok(Some(RawRecord {
            pos: self.pos[v],
            reference,
            alts: vec![alt],
            gt,
            info_raw: Vec::new(), // SVAR1 has no INFO fields
            format_raw,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        )
        .unwrap();

        let r0 = src.next_record().unwrap().unwrap();
        assert_eq!(r0.pos, 10);
        assert_eq!(r0.reference, b"A");
        assert_eq!(r0.alts, vec![b"G".to_vec()]);
        assert_eq!(r0.gt, vec![1, 0, 1, 0]); // hap0 & hap2 carry ALT1
        // format_raw[0] = Some(per-sample). S0 carried on hap0 -> 0.5; S1 on hap2 -> 2.5
        let ds0 = r0.format_raw[0].as_ref().unwrap();
        assert_eq!(ds0[0], vec![0.5]);
        assert_eq!(ds0[1], vec![2.5]);

        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.gt, vec![0, 0, 0, 1]); // only S1 hap1
        let ds1 = r1.format_raw[0].as_ref().unwrap();
        assert!(ds1[0][0].is_nan()); // S0 non-carrier -> missing sentinel (NaN)
        assert_eq!(ds1[1], vec![1.5]); // S1 carrier

        assert!(src.next_record().unwrap().is_none());
        std::fs::remove_dir_all(&tmp).ok();
    }
}

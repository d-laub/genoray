//! Decode a uniform `KeyRef` into a `Call` (position + ilen + ALT bytes), and
//! the per-hap / per-query result types built from decoded calls.

use crate::nrvk::LongAlleleReader;
use crate::rvk::{self, DecodedKey};
use crate::spine::KeyRef;

/// One overlapping variant call, decoded. The per-element intermediate before
/// the k-way merge (the columnar `HapCalls` is assembled from these).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub position: u32,
    /// Length delta (ALT − REF): `0` SNP, `> 0` insertion, `< 0` deletion.
    pub ilen: i32,
    /// Decoded ALT bytes for SNP/INS; empty for a pure DEL (the anchor base is
    /// not stored in the key — recovered from the reference downstream).
    pub alt: Vec<u8>,
}

/// Decode one uniform `KeyRef` into a `Call`, resolving long-INS lookups through
/// the LUT. The single place a query result touches alleles: SNP/INS decode
/// inline (`Inline`), a pure DEL yields an empty ALT (`PureDel`), a long INS
/// resolves via the bank (`Lookup`). `ilen = alt.len() - 1` for the inline lanes
/// matches M5's `decode_snp_2bit`/`decode_indel_hit` contract exactly.
pub(crate) fn decode_keyref(kr: KeyRef, lut: Option<&LongAlleleReader>) -> Call {
    match rvk::decode_key(kr.key) {
        DecodedKey::Inline { alt } => Call {
            position: kr.position,
            ilen: alt.len() as i32 - 1,
            alt,
        },
        DecodedKey::PureDel { ilen } => Call {
            position: kr.position,
            ilen,
            alt: Vec::new(),
        },
        DecodedKey::Lookup { row } => {
            let alt = lut
                .expect("indel lookup key requires a long-allele LUT")
                .get_allele(row);
            Call {
                position: kr.position,
                ilen: alt.len() as i32 - 1,
                alt,
            }
        }
    }
}

/// Per-haplotype overlapping calls, position-sorted. Struct-of-arrays for a
/// numpy-friendly M6 hand-off.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HapCalls {
    pub positions: Vec<u32>,
    pub ilens: Vec<i32>,
    pub alts: Vec<Vec<u8>>,
}

/// Result of an `overlap_sample` query: one `HapCalls` per haplotype
/// (`per_hap.len() == ploidy`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryResult {
    pub per_hap: Vec<HapCalls>,
}

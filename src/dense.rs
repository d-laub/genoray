//! Dense-representation class registry. Mirrors `streams.rs` but for the
//! *per-variant* dense matrix (not per-call streams). Two classes: 2-bit SNP
//! (packed post-merge, no LUT) and 32-bit indel (shares the per-contig LUT).

use crate::cost_model::Class;
use crate::enum_map::{EnumKey, EnumMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseClass {
    Snp = 0,
    Indel = 1,
}

impl EnumKey for DenseClass {
    const COUNT: usize = 2;
    const ALL: &'static [DenseClass] = &[DenseClass::Snp, DenseClass::Indel];
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

impl DenseClass {
    #[inline]
    pub fn key_bytes(self) -> usize {
        match self {
            DenseClass::Snp => 1,   // one raw 2-bit code per variant (packed at merge)
            DenseClass::Indel => 4, // u32 key per variant
        }
    }
    #[inline]
    pub fn cost_class(self) -> Class {
        match self {
            DenseClass::Snp => Class::Snp,
            DenseClass::Indel => Class::Indel,
        }
    }
}

pub struct DenseSpec {
    pub class: DenseClass,
    pub subdir: &'static str,
    pub key_bytes: usize,
    /// 2-bit-pack the merged key file (SNP only).
    pub pack_snp: bool,
}

pub const DENSE_REGISTRY: [DenseSpec; DenseClass::COUNT] = [
    DenseSpec {
        class: DenseClass::Snp,
        subdir: "dense/snp",
        key_bytes: 1,
        pack_snp: true,
    },
    DenseSpec {
        class: DenseClass::Indel,
        subdir: "dense/indel",
        key_bytes: 4,
        pack_snp: false,
    },
];

/// Fixed-size map keyed by `DenseClass`, array-backed (O(1), no hashing).
pub type DenseMap<T> = EnumMap<DenseClass, T, { DenseClass::COUNT }>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_indices_match_classes() {
        for spec in &DENSE_REGISTRY {
            assert_eq!(DENSE_REGISTRY[spec.class.index()].class, spec.class);
        }
    }

    #[test]
    fn test_key_bytes() {
        assert_eq!(DenseClass::Snp.key_bytes(), 1);
        assert_eq!(DenseClass::Indel.key_bytes(), 4);
    }

    #[test]
    fn test_densemap_get_set() {
        let mut m: DenseMap<u32> = DenseMap::from_fn(|_| 0);
        *m.get_mut(DenseClass::Indel) = 9;
        assert_eq!(*m.get(DenseClass::Snp), 0);
        assert_eq!(*m.get(DenseClass::Indel), 9);
    }
}

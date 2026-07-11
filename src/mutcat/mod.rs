//! COSMIC mutation-catalogue code space and classifiers for SVAR2.
//! Code-space layout MUST match python/genoray/_mutcat/codebook.py.

use std::ops::Range;

/// Class-local index counts (COSMIC codebook sizes).
pub const N_SBS96: usize = 96;
pub const N_DBS78: usize = 78;
pub const N_ID83: usize = 83;

/// Unified code-space offsets — mirror codebook.py exactly.
pub const SBS96_OFFSET: usize = 0;
pub const DBS78_OFFSET: usize = SBS96_OFFSET + N_SBS96; // 96
pub const ID83_OFFSET: usize = DBS78_OFFSET + N_DBS78; // 174
pub const N_CODES: usize = ID83_OFFSET + N_ID83; // 257

/// Sidecar `u8` sentinels (unsigned; v1's negative int16 sentinels don't survive).
pub const UNCLASSIFIED: u8 = 254;
pub const NOT_ANNOTATED: u8 = 255;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Sbs96,
    Dbs78,
    Id83,
}

impl Kind {
    /// Half-open unified-code range for this kind.
    pub fn code_range(self) -> Range<usize> {
        match self {
            Kind::Sbs96 => SBS96_OFFSET..DBS78_OFFSET,
            Kind::Dbs78 => DBS78_OFFSET..ID83_OFFSET,
            Kind::Id83 => ID83_OFFSET..N_CODES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_space_matches_codebook() {
        assert_eq!(SBS96_OFFSET, 0);
        assert_eq!(DBS78_OFFSET, 96);
        assert_eq!(ID83_OFFSET, 174);
        assert_eq!(N_CODES, 257);
    }

    #[test]
    fn kind_ranges_are_contiguous_and_sized() {
        assert_eq!(Kind::Sbs96.code_range(), 0..96);
        assert_eq!(Kind::Dbs78.code_range(), 96..174);
        assert_eq!(Kind::Id83.code_range(), 174..257);
    }
}

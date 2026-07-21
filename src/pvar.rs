//! Streaming `.pvar` / `.pvar.zst` variant-metadata reader for the PGEN record
//! source. PGEN stores genotypes only; POS/REF/ALT live in the sibling `.pvar`.
//!
//! Written against the PLINK2 `.pvar` text format (a VCF-like TSV). Contains no
//! plink-ng code.
//!
//! `.bim` is intentionally unsupported: it only accompanies a PLINK1 `.bed`,
//! which SVAR2 does not read.

use crate::error::ConversionError;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// One `.pvar` row, reduced to what the conversion spine needs.
pub struct PvarRecord {
    /// 0-based start position (`.pvar` POS is 1-based on disk).
    pub pos: u32,
    pub reference: Vec<u8>,
    /// ALT alleles, comma-split. ALT1 is `alts[0]`. Empty when the `.pvar` ALT
    /// field is the bare `.` sentinel -- plink2's spelling of "no alternate
    /// allele" for a monomorphic site (all samples are REF).
    pub alts: Vec<Vec<u8>>,
}

pub struct PvarReader {
    lines: Box<dyn BufRead + Send>,
    path: String,
    pos_col: usize,
    ref_col: usize,
    alt_col: usize,
    /// Global variant index of the next row to be returned. Used for error
    /// messages and exposed via `current_vidx` for callers that need the
    /// absolute `.pvar` row index of the record about to be returned.
    vidx: usize,
    buf: String,
}

// Manual impl: `lines` is `Box<dyn BufRead + Send>`, which has no `Debug` impl,
// so `#[derive(Debug)]` isn't available. `Result::unwrap_err` (used in tests)
// requires the `Ok` type to be `Debug`.
impl std::fmt::Debug for PvarReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PvarReader")
            .field("path", &self.path)
            .field("pos_col", &self.pos_col)
            .field("ref_col", &self.ref_col)
            .field("alt_col", &self.alt_col)
            .field("vidx", &self.vidx)
            .finish_non_exhaustive()
    }
}

impl PvarReader {
    /// Open `path` (`.pvar` or `.pvar.zst`), consume the header, and skip forward
    /// to variant index `var_start` (0-based, counting data rows only).
    pub fn open(path: &str, var_start: usize) -> Result<Self, ConversionError> {
        if !std::path::Path::new(path).exists() {
            return Err(ConversionError::MissingFile {
                path: path.to_string(),
            });
        }
        let file = File::open(path).map_err(|e| ConversionError::Io {
            context: format!("opening pvar {path}"),
            source: e,
        })?;
        let lines: Box<dyn BufRead + Send> = if path.ends_with(".zst") {
            let dec = zstd::stream::read::Decoder::new(file).map_err(|e| ConversionError::Io {
                context: format!("opening zstd stream for {path}"),
                source: e,
            })?;
            Box::new(BufReader::new(dec))
        } else {
            Box::new(BufReader::new(file))
        };

        let mut me = Self {
            lines,
            path: path.to_string(),
            pos_col: 0,
            ref_col: 0,
            alt_col: 0,
            vidx: 0,
            buf: String::new(),
        };

        // Header: skip `##` meta lines, then require the `#CHROM ...` column line.
        // plink2 --make-pgen always writes one; a headerless .pvar is rejected
        // rather than guessed at.
        loop {
            me.buf.clear();
            let n = me.read_line()?;
            if n == 0 {
                return Err(ConversionError::Input(format!(
                    "{path}: reached EOF without a '#CHROM' header line"
                )));
            }
            if me.buf.starts_with("##") {
                continue;
            }
            if !me.buf.starts_with("#CHROM") {
                return Err(ConversionError::Input(format!(
                    "{path}: expected a '#CHROM' header line, found '{}'. \
                     Headerless .pvar files are not supported.",
                    me.buf.trim_end()
                )));
            }
            let cols: Vec<&str> = me.buf.trim_end().split('\t').collect();
            let find = |name: &str| -> Result<usize, ConversionError> {
                cols.iter().position(|c| *c == name).ok_or_else(|| {
                    ConversionError::Input(format!("{path}: header is missing a '{name}' column"))
                })
            };
            me.pos_col = find("POS")?;
            me.ref_col = find("REF")?;
            me.alt_col = find("ALT")?;
            break;
        }

        for _ in 0..var_start {
            me.buf.clear();
            if me.read_line()? == 0 {
                return Err(ConversionError::Input(format!(
                    "{path}: ran out of variants while skipping to index {var_start}"
                )));
            }
            me.vidx += 1;
        }
        Ok(me)
    }

    /// Absolute `.pvar` row index of the NEXT row `next_variant` will return.
    pub fn current_vidx(&self) -> usize {
        self.vidx
    }

    fn read_line(&mut self) -> Result<usize, ConversionError> {
        self.lines
            .read_line(&mut self.buf)
            .map_err(|e| ConversionError::Io {
                context: format!("reading pvar {}", self.path),
                source: e,
            })
    }

    /// Next variant, or `None` at EOF.
    pub fn next_variant(&mut self) -> Result<Option<PvarRecord>, ConversionError> {
        self.buf.clear();
        if self.read_line()? == 0 {
            return Ok(None);
        }
        let line = self.buf.trim_end_matches(['\n', '\r']);
        let cols: Vec<&str> = line.split('\t').collect();
        let want = self.pos_col.max(self.ref_col).max(self.alt_col);
        if cols.len() <= want {
            return Err(ConversionError::Input(format!(
                "{}: variant {} has {} columns, need at least {}",
                self.path,
                self.vidx,
                cols.len(),
                want + 1
            )));
        }

        let pos_1based: u32 = cols[self.pos_col].parse().map_err(|_| {
            ConversionError::Input(format!(
                "{}: variant {} has a non-integer POS '{}'",
                self.path, self.vidx, cols[self.pos_col]
            ))
        })?;
        if pos_1based == 0 {
            return Err(ConversionError::Input(format!(
                "{}: variant {} has POS 0; .pvar POS is 1-based",
                self.path, self.vidx
            )));
        }

        let mut reference = cols[self.ref_col].as_bytes().to_vec();
        reference.make_ascii_uppercase();
        // A bare `.` ALT means "no alternate allele" (monomorphic site), not a
        // one-character allele literally spelled ".". Only the exact field
        // value `.` is the sentinel -- a real allele can never contain a
        // comma, so this can't misfire on a multiallelic ALT.
        let alt_field = cols[self.alt_col];
        let alts: Vec<Vec<u8>> = if alt_field == "." {
            Vec::new()
        } else {
            alt_field
                .split(',')
                .map(|a| {
                    let mut v = a.as_bytes().to_vec();
                    v.make_ascii_uppercase();
                    v
                })
                .collect()
        };

        self.vidx += 1;
        Ok(Some(PvarRecord {
            pos: pos_1based - 1,
            reference,
            alts,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, body: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join(name)).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        dir
    }

    const BODY: &str = "\
##fileformat=PVARv1.0
#CHROM\tPOS\tID\tREF\tALT
chr1\t3\t.\tA\tG
chr1\t7\t.\tC\tCAT
chr1\t12\t.\tGTA\tG,GT
";

    #[test]
    fn reads_pos_ref_alts_zero_based() {
        let dir = write_tmp("x.pvar", BODY);
        let p = dir.path().join("x.pvar");
        let mut r = PvarReader::open(p.to_str().unwrap(), 0).unwrap();

        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 2); // 1-based 3 -> 0-based 2
        assert_eq!(v.reference, b"A");
        assert_eq!(v.alts, vec![b"G".to_vec()]);

        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 6);
        assert_eq!(v.alts, vec![b"CAT".to_vec()]);

        // Multiallelic ALT is comma-separated.
        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 11);
        assert_eq!(v.reference, b"GTA");
        assert_eq!(v.alts, vec![b"G".to_vec(), b"GT".to_vec()]);

        assert!(r.next_variant().unwrap().is_none());
    }

    const BODY_WITH_MONO: &str = "\
##fileformat=PVARv1.0
#CHROM\tPOS\tID\tREF\tALT
chr1\t3\t.\tA\tG
chr1\t5\t.\tC\t.
chr1\t7\t.\tC\tCAT
";

    #[test]
    fn dot_alt_is_monomorphic_zero_alts() {
        // plink2 emits a bare `.` ALT for a monomorphic site (REF present, no
        // ALT observed). It must decode to zero alts, not the literal string
        // ".", and must not disturb the rows around it.
        let dir = write_tmp("mono.pvar", BODY_WITH_MONO);
        let p = dir.path().join("mono.pvar");
        let mut r = PvarReader::open(p.to_str().unwrap(), 0).unwrap();

        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 2); // 1-based 3 -> 0-based 2
        assert_eq!(v.reference, b"A");
        assert_eq!(v.alts, vec![b"G".to_vec()]);

        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 4); // 1-based 5 -> 0-based 4
        assert_eq!(v.reference, b"C");
        assert!(
            v.alts.is_empty(),
            "ALT '.' must decode to an empty alt list"
        );

        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 6);
        assert_eq!(v.reference, b"C");
        assert_eq!(v.alts, vec![b"CAT".to_vec()]);

        assert!(r.next_variant().unwrap().is_none());
    }

    #[test]
    fn var_start_skips_leading_variants() {
        let dir = write_tmp("x.pvar", BODY);
        let p = dir.path().join("x.pvar");
        let mut r = PvarReader::open(p.to_str().unwrap(), 2).unwrap();
        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 11);
        assert!(r.next_variant().unwrap().is_none());
    }

    #[test]
    fn current_vidx_is_absolute_index_of_next_row() {
        // BODY has 3 data rows (absolute indices 0, 1, 2).
        let dir = write_tmp("x.pvar", BODY);
        let p = dir.path().join("x.pvar");
        let mut r = PvarReader::open(p.to_str().unwrap(), 0).unwrap();

        // Before any `next_variant`, the next row to be returned is row 0.
        assert_eq!(r.current_vidx(), 0);

        // The call that returns row 0's record leaves `current_vidx` at 1 --
        // the absolute index of the *next* row, not the one just returned.
        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 2); // row 0: 1-based POS 3 -> 0-based 2
        assert_eq!(r.current_vidx(), 1);

        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 6); // row 1
        assert_eq!(r.current_vidx(), 2);

        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 11); // row 2
        assert_eq!(r.current_vidx(), 3);

        assert!(r.next_variant().unwrap().is_none());

        // With `var_start` skipping to index 2, the accessor reflects the
        // skip immediately -- i.e. the value captured BEFORE a `next_variant`
        // call equals the absolute row that call is about to return.
        let mut r2 = PvarReader::open(p.to_str().unwrap(), 2).unwrap();
        assert_eq!(r2.current_vidx(), 2);
        let v = r2.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 11); // row 2
        assert_eq!(r2.current_vidx(), 3);
    }

    #[test]
    fn missing_header_line_is_an_error() {
        let dir = write_tmp("x.pvar", "chr1\t3\t.\tA\tG\n");
        let p = dir.path().join("x.pvar");
        let err = PvarReader::open(p.to_str().unwrap(), 0).unwrap_err();
        assert!(format!("{err}").contains("#CHROM"));
    }

    #[test]
    fn reads_zstd_compressed_pvar() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.pvar.zst");
        let f = std::fs::File::create(&p).unwrap();
        let mut enc = zstd::stream::write::Encoder::new(f, 3).unwrap();
        enc.write_all(BODY.as_bytes()).unwrap();
        enc.finish().unwrap();

        let mut r = PvarReader::open(p.to_str().unwrap(), 0).unwrap();
        let v = r.next_variant().unwrap().unwrap();
        assert_eq!(v.pos, 2);
        assert_eq!(v.reference, b"A");
    }
}

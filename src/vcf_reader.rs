use crate::normalize::atomize_record;
use crate::types::{BitGrid3, DenseChunk};
use rust_htslib::bcf::record::Record;
use rust_htslib::bcf::{IndexedReader, Read};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

// A decomposed atom awaiting emission. Carries a shared handle to its source record's
// per-column allele indices so genotype presence is computed at chunk-build time.
struct PendingAtom {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    source_alt_index: u16,
    gt: Arc<Vec<i32>>, // len = num_samples * ploidy; allele index per column (-1 = missing)
    seq: u64,          // stable tiebreak for equal positions
}

impl PartialEq for PendingAtom {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.seq == other.seq
    }
}
impl Eq for PendingAtom {}
impl PartialOrd for PendingAtom {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PendingAtom {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pos.cmp(&other.pos).then(self.seq.cmp(&other.seq))
    }
}

pub struct VcfChunkReader {
    inner_reader: IndexedReader,
    num_samples: usize,
    ploidy: usize,
    sample_indices: Vec<usize>,
    // full 0-based contig sequence, uppercased. Consumed by `validate_ref`/`left_align`
    // in `decompose_current_record`.
    ref_seq: Vec<u8>,
    // Whether a reference FASTA was provided. When false, `validate_ref` and
    // `left_align` are skipped and the input is trusted to be pre-normalized.
    has_reference: bool,
    // Whether to skip (vs. error on) out-of-scope symbolic/breakend ALTs.
    skip_out_of_scope: bool,
    // Running count of out-of-scope ALTs dropped across this contig.
    dropped_out_of_scope: u64,

    // Reorder state, persisted across read_next_chunk calls.
    record: Record,
    heap: BinaryHeap<Reverse<PendingAtom>>,
    frontier: u32, // start pos of the most recently read record: all future atoms have pos >= this
    eof: bool,
    next_seq: u64,
}

impl VcfChunkReader {
    // opens the file, applies the sample filter at the C-level, and jumps to the chromosome.
    pub fn new(
        vcf_path: &str,
        fasta_path: Option<&str>,
        chrom: &str,
        samples: &[&str],
        htslib_threads: usize,
        ploidy: usize,
        skip_out_of_scope: bool,
    ) -> Self {
        let mut reader = IndexedReader::from_path(vcf_path)
            .expect("Failed to open VCF/BCF index. Is there a .tbi or .csi file?");

        reader
            .set_threads(htslib_threads)
            .expect("Failed to allocate HTSlib background threads");

        let header = reader.header().clone();

        let rid = header
            .name2rid(chrom.as_bytes())
            .expect("Chromosome not found in VCF header");

        reader
            .fetch(rid, 0, None)
            .expect("Failed to fetch chromosome region");

        let sample_indices: Vec<usize> = samples
            .iter()
            .map(|name| {
                header
                    .sample_id(name.as_bytes())
                    .unwrap_or_else(|| panic!("Sample {} not found in VCF", name))
            })
            .collect();

        // Reference is optional. With a FASTA, cache the full uppercased contig for
        // validate_ref/left_align; without one, leave it empty and skip both.
        let (ref_seq, has_reference) = match fasta_path {
            Some(path) => {
                let fasta = rust_htslib::faidx::Reader::from_path(path)
                    .expect("Failed to open reference FASTA (is there a .fai?)");
                // htslib's faidx_seq_len returns -1 for an unknown contig, which
                // fetch_seq_len (via rust-htslib) surfaces as u64::MAX rather than 0 —
                // check for it explicitly so a missing contig fails fast with a clear
                // message instead of the generic "Failed to fetch contig" panic below.
                let contig_len_raw = fasta.fetch_seq_len(chrom);
                if contig_len_raw == u64::MAX {
                    panic!("Contig '{chrom}' not found in reference FASTA");
                }
                let contig_len = contig_len_raw as usize;
                let mut ref_seq = if contig_len == 0 {
                    Vec::new()
                } else {
                    fasta
                        .fetch_seq(chrom, 0, contig_len - 1)
                        .expect("Failed to fetch contig from reference FASTA")
                };
                ref_seq.make_ascii_uppercase();
                (ref_seq, true)
            }
            None => (Vec::new(), false),
        };

        let record = reader.empty_record();

        Self {
            inner_reader: reader,
            num_samples: samples.len(),
            ploidy,
            sample_indices,
            ref_seq,
            has_reference,
            skip_out_of_scope,
            dropped_out_of_scope: 0,
            record,
            heap: BinaryHeap::new(),
            frontier: 0,
            eof: false,
            next_seq: 0,
        }
    }

    /// Total out-of-scope ALTs dropped so far (valid after the read loop drains).
    pub fn dropped_out_of_scope(&self) -> u64 {
        self.dropped_out_of_scope
    }

    // Decompose `self.record` into atoms and push them onto the reorder heap, sharing
    // one decoded genotype vector across all atoms of the record.
    fn decompose_current_record(&mut self) {
        let pos = self.record.pos() as u32;

        // Own the alleles so the record borrow is released before we mutate self.
        let ref_allele: Vec<u8>;
        let alts_owned: Vec<Vec<u8>>;
        {
            let alleles = self.record.alleles();
            ref_allele = alleles[0].to_vec();
            alts_owned = alleles[1..].iter().map(|a| a.to_vec()).collect();
        }

        // Decode per-column allele indices (-1 = missing).
        let columns = self.num_samples * self.ploidy;
        let mut gt = vec![-1i32; columns];
        {
            // Decode GT straight from the raw BCF integer buffer instead of
            // `record.genotypes().get(i)`, which allocates a per-sample
            // `Genotype(Vec<GenotypeAllele>)` for every sample of every record —
            // the dominant reader-side allocation churn. BCF GT encoding: an
            // allele is `(idx + 1) << 1 | phased`, so `e >= 2` decodes to
            // `(e >> 1) - 1`; `e` of 0/1 is missing and `i32::MIN` is vector-end
            // padding — both `< 2`, so a single `e >= 2` test reproduces the
            // `GenotypeAllele::index()` semantics exactly.
            let gts = self
                .record
                .format(b"GT")
                .integer()
                .expect("Failed to read GT format");
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
        let gt = Arc::new(gt);

        // Fail fast only when a reference is available; without one we trust the
        // input is already normalized/left-aligned.
        if self.has_reference {
            crate::normalize::validate_ref(pos, &ref_allele, &self.ref_seq)
                .expect("REF disagrees with reference FASTA");
        }

        let alt_refs: Vec<&[u8]> = alts_owned.iter().map(|a| a.as_slice()).collect();
        let mut atoms = Vec::new();
        let dropped = atomize_record(
            pos,
            &ref_allele,
            &alt_refs,
            &mut atoms,
            self.skip_out_of_scope,
        )
        .expect("symbolic/breakend ALT is out of scope for SVAR2 (short-read only)");
        self.dropped_out_of_scope += dropped as u64;

        for atom in atoms {
            // Left-align only when a reference is available; otherwise store the atom
            // at its as-given (right-trimmed) position.
            let atom = if self.has_reference {
                crate::normalize::left_align(atom, &self.ref_seq, crate::normalize::L_MAX)
            } else {
                atom
            };
            let seq = self.next_seq;
            self.next_seq += 1;
            self.heap.push(Reverse(PendingAtom {
                pos: atom.pos,
                ilen: atom.ilen,
                alt: atom.alt,
                source_alt_index: atom.source_alt_index,
                gt: Arc::clone(&gt),
                seq,
            }));
        }
    }

    // Yield the next atom in global position order, reading and decomposing more
    // records as needed. Left-alignment (M2b) can move an atom up to `L_MAX` bases below
    // its record's start, so a future record at start `frontier` may still produce an atom
    // as low as `frontier - L_MAX`. An atom is therefore safe to emit only once its
    // position is strictly below `frontier - L_MAX` (saturating), or the input is
    // exhausted — this preserves the position-sorted invariant the Phase-2 merge relies on.
    //
    // Memory note: the heap holds every atom with pos >= frontier - L_MAX until a later
    // record advances the frontier past that window (or EOF). For coordinate-sorted input
    // this is bounded by the atoms within an `L_MAX`-wide window of the frontier, plus
    // atoms sharing a single start position — a run of many records all starting at the
    // same pos never advances the frontier, so the heap can still grow unboundedly in
    // that (pre-existing) pathological case.
    fn next_atom(&mut self) -> Option<PendingAtom> {
        loop {
            if let Some(Reverse(top)) = self.heap.peek() {
                if self.eof || top.pos < self.frontier.saturating_sub(crate::normalize::L_MAX) {
                    return Some(self.heap.pop().unwrap().0);
                }
            } else if self.eof {
                return None;
            }

            match self.inner_reader.read(&mut self.record) {
                Some(Ok(())) => {
                    self.frontier = self.record.pos() as u32;
                    self.decompose_current_record();
                }
                Some(Err(e)) => panic!("VCF Read Error: {}", e),
                None => self.eof = true,
            }
        }
    }

    // Pull up to `chunk_size` atoms (already globally position-sorted) and pack them
    // into a variant-major DenseChunk. Returns None once no atoms remain.
    pub fn read_next_chunk(&mut self, chunk_size: usize, chunk_id: usize) -> Option<DenseChunk> {
        let mut atoms: Vec<PendingAtom> = Vec::with_capacity(chunk_size);
        while atoms.len() < chunk_size {
            match self.next_atom() {
                Some(a) => atoms.push(a),
                None => break,
            }
        }
        if atoms.is_empty() {
            return None;
        }

        let v = atoms.len();
        let columns = self.num_samples * self.ploidy;

        let mut pos = Vec::with_capacity(v);
        let mut ilens = Vec::with_capacity(v);
        let mut alt = Vec::with_capacity(v * 2);
        let mut alt_offsets = Vec::with_capacity(v + 1);
        alt_offsets.push(0u32);
        let mut genos = BitGrid3::zeros(v, self.num_samples, self.ploidy);

        let mut off = 0u32;
        for (vi, a) in atoms.iter().enumerate() {
            pos.push(a.pos);
            ilens.push(a.ilen);
            alt.extend_from_slice(&a.alt);
            off += a.alt.len() as u32;
            alt_offsets.push(off);

            let src = a.source_alt_index as i32;
            let base = vi * columns;
            // Pack the presence bits one 64-bit word at a time. `or_bit` wrote a
            // whole word back to memory per bit — up to 64 redundant
            // load-modify-stores per word; here each word is assembled in a
            // register and written once. Bits start zeroed, so a plain OR-store
            // is correct. Produces the identical BitGrid3 as the per-bit loop.
            let gtc: &[i32] = &a.gt;
            let words = &mut genos.words;
            let mut col = 0usize;
            while col < columns {
                let flat = base + col;
                let w = flat >> 6;
                let b = flat & 63;
                let n = (64 - b).min(columns - col);
                let mut acc = 0u64;
                for k in 0..n {
                    // SAFETY: col + k < columns == gtc.len().
                    acc |= ((unsafe { *gtc.get_unchecked(col + k) } == src) as u64) << (b + k);
                }
                // SAFETY: w indexes the word holding bit `base + col`, in range.
                unsafe {
                    *words.get_unchecked_mut(w) |= acc;
                }
                col += n;
            }
        }

        Some(DenseChunk {
            chunk_id,
            pos,
            ilens,
            alt,
            alt_offsets,
            genos,
        })
    }
}

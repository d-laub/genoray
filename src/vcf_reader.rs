use crate::normalize::atomize_record;
use crate::types::{BitGrid3, DenseChunk};
use rust_htslib::bcf::record::Record;
use rust_htslib::bcf::{IndexedReader, Read};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::rc::Rc;

// A decomposed atom awaiting emission. Carries a shared handle to its source record's
// per-column allele indices so genotype presence is computed at chunk-build time.
struct PendingAtom {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    source_alt_index: u16,
    gt: Rc<Vec<i32>>, // len = num_samples * ploidy; allele index per column (-1 = missing)
    seq: u64,         // stable tiebreak for equal positions
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
        chrom: &str,
        samples: &[&str],
        htslib_threads: usize,
        ploidy: usize,
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

        let record = reader.empty_record();

        Self {
            inner_reader: reader,
            num_samples: samples.len(),
            ploidy,
            sample_indices,
            record,
            heap: BinaryHeap::new(),
            frontier: 0,
            eof: false,
            next_seq: 0,
        }
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
            let genotypes = self.record.genotypes().expect("Failed to read genotypes");
            for (s_idx, &vcf_idx) in self.sample_indices.iter().enumerate() {
                let sample_gt = genotypes.get(vcf_idx);
                for p in 0..self.ploidy {
                    let idx = if p < sample_gt.len() {
                        sample_gt[p].index().map(|v| v as i32).unwrap_or(-1)
                    } else {
                        -1
                    };
                    gt[s_idx * self.ploidy + p] = idx;
                }
            }
        }
        let gt = Rc::new(gt);

        let alt_refs: Vec<&[u8]> = alts_owned.iter().map(|a| a.as_slice()).collect();
        let mut atoms = Vec::new();
        atomize_record(pos, &ref_allele, &alt_refs, &mut atoms)
            .expect("symbolic/breakend ALT is out of scope for SVAR2 (short-read only)");

        for atom in atoms {
            let seq = self.next_seq;
            self.next_seq += 1;
            self.heap.push(Reverse(PendingAtom {
                pos: atom.pos,
                ilen: atom.ilen,
                alt: atom.alt,
                source_alt_index: atom.source_alt_index,
                gt: Rc::clone(&gt),
                seq,
            }));
        }
    }

    // Yield the next atom in global position order, reading and decomposing more
    // records as needed. An atom is safe to emit once its position is strictly below
    // the read frontier (no future atom can precede the frontier, since there is no
    // left-alignment) or once the input is exhausted.
    //
    // Memory note: the heap holds every atom whose pos == frontier until a record with a
    // strictly greater start pos advances the frontier (or EOF). For coordinate-sorted
    // input this is bounded by the atoms at a single position; a pathological run of many
    // records sharing one start pos would grow it. M2b (left-alignment) weakens the
    // `pos >= record_start` premise and must revisit this bound.
    fn next_atom(&mut self) -> Option<PendingAtom> {
        loop {
            if let Some(Reverse(top)) = self.heap.peek() {
                if self.eof || top.pos < self.frontier {
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
            for col in 0..columns {
                genos.or_bit(base + col, a.gt[col] == src);
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

# PGEN → SVAR2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SparseVar2.from_pgen`, converting a PLINK2 PGEN to an SVAR2 store by reusing the existing VCF conversion pipeline's normalization/atom/merge spine behind a new record source.

**Architecture:** Split `VcfChunkReader` into a source-agnostic `ChunkAssembler` (reorder heap, atom decomposition, presence packing, field staging) plus a `RecordSource` trait with two implementations: `VcfRecordSource` (htslib) and `PgenRecordSource`. `PgenRecordSource` gets per-haplotype allele codes from the **already-depended-on** `pgenlib` Python wheel via `read_alleles_range` (which releases the GIL), and variant metadata from a new Rust `.pvar` streamer. Along the way, presence packing becomes windowed so staging memory stops scaling with `chunk_size × n_samples`.

**Tech Stack:** Rust (pyo3 0.29, numpy, rayon, crossbeam, new: `zstd`), Python 3.10+ (polars, pgenlib), pixi, pytest, proptest.

## Global Constraints

- **Licensing (non-negotiable).** genoray is MIT. **Do not copy any code or comments from plink-ng into this repo.** Do not link, vendor, or build plink-ng C++. The only permitted use of plink code is calling the **already-installed** `pgenlib` PyPI wheel (LGPL-3.0) through its **public Python API**, exactly as `genoray/_pgen.py` and `genoray/_svar/_convert.py` already do. `plink2` the binary may be invoked as a subprocess in tests only (already in `pixi.toml`).
- **Coordinates:** all ranges 0-based, half-open `[start, end)`. `.pvar` POS is 1-based on disk and must be converted to 0-based.
- **Missing sentinels:** pgenlib allele codes use `-9` for missing; the pipeline's `PendingAtom.gt` convention is `-1`. Map `-9 → -1`.
- **Ploidy:** PGEN is diploid-only. `ploidy = 2` everywhere on the PGEN path; do not add a `ploidy` parameter to `from_pgen`.
- **Public API rule (CLAUDE.md):** any change to a name reachable from `import genoray` without an underscore MUST update `skills/genoray-api/SKILL.md` in the same PR.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `perf:`). Do **not** bump the version or regenerate `CHANGELOG.md`'s versioned sections by hand — add entries under `## Unreleased` only.
- **Rust tests must be run as** `pixi run bash -lc 'cargo test --no-default-features <args>'`. With default features the pyo3 test binary fails to link (`undefined symbol: _Py_Dealloc`).
- **Never background a long `cargo`/`maturin` build.** Run them in the foreground and wait.
- Working tree: worktree `.claude/worktrees/pgen-to-svar2`, branch `worktree-pgen-to-svar2`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/vcf_reader.rs` (modify) | **Shrinks.** Keeps only htslib-specific work: open/fetch, GT decode, INFO/FORMAT raw decode. Becomes `VcfRecordSource`. |
| `src/record_source.rs` (create) | `RawRecord` (owned) + `RecordSource` trait. The seam. |
| `src/chunk_assembler.rs` (create) | Source-agnostic: `PendingAtom`, `AtomMeta`, presence packing, reorder heap, REF validation, left-alignment, atomization, field resolution, `DenseChunk` assembly. |
| `src/pvar.rs` (create) | Streaming `.pvar` / `.pvar.zst` reader → `PvarRecord { pos, reference, alts }`. |
| `src/pgen_reader.rs` (create) | `PgenRecordSource`: pgenlib handle + batched `(B, 2S)` int32 buffer + `PvarReader`. |
| `src/orchestrator.rs` (modify) | `process_chromosome` takes a `SourceSpec` and builds the right `RecordSource`. |
| `src/lib.rs` (modify) | Adds the `run_pgen_conversion_pipeline` pyfunction. |
| `src/types.rs` (unchanged) | `DenseChunk`, `BitGrid3` (already has `truncate_v`). |
| `python/genoray/_svar2.py` (modify) | `SparseVar2.from_pgen`. |
| `python/genoray/_cli/__main__.py` (modify) | `genoray write` dispatches on the `.pgen` suffix. |
| `skills/genoray-api/SKILL.md` (modify) | Document `from_pgen`. |
| `CHANGELOG.md` (modify) | `## Unreleased` entry. |
| `tests/test_svar2_from_pgen.py` (create) | Cross-backend equivalence + error cases. |
| `tests/test_pvar.rs` (create) | Rust `.pvar` parser tests. |

---

## Task 1: Windowed presence packing (VCF path, memory fix)

Today `read_next_chunk` buffers `chunk_size` `PendingAtom`s, each holding an `S × P` i32 `gt` vector — 32× the packed bit-grid it produces (~2 GB at `chunk_size=25_000`, 10k samples; not representable at biobank scale). Pack in windows and drop `gt` as soon as its bits are set.

**Files:**
- Modify: `src/vcf_reader.rs` (the `read_next_chunk` function around line 592, plus new helpers)
- Test: `src/vcf_reader.rs` `mod tests` (existing proptest module at the bottom)

**Interfaces:**
- Consumes: existing `PendingAtom`, `pack_presence_seq`, `pack_presence_par`, `gcd`, `BitGrid3::{zeros, truncate_v}`.
- Produces: `struct AtomMeta`, `fn flush_window(...)`, `const PACK_WINDOW: usize`. Task 2 moves all of these verbatim into `chunk_assembler.rs`.

- [ ] **Step 1: Write the failing test**

Add to the existing `mod tests` in `src/vcf_reader.rs` (it already has a `test_pool()` helper and an `atom(gt, src)` builder):

```rust
    // Windowed packing must be bit-identical to packing the whole chunk at once.
    // `flush_window` is only ever called at word-aligned variant offsets except
    // for the final (partial) window, which nothing follows — mirror that here.
    proptest! {
        #[test]
        fn windowed_pack_matches_full_pack(
            n_samples in 1usize..9,
            ploidy in 1usize..3,
            srcs in prop::collection::vec(0u16..3u16, 1..200),
        ) {
            let columns = n_samples * ploidy;
            let v = srcs.len();

            // Deterministic gt: column c of variant i carries allele (i + c) % 3.
            let atoms: Vec<PendingAtom> = srcs
                .iter()
                .enumerate()
                .map(|(i, &src)| {
                    let gt: Vec<i32> = (0..columns).map(|c| ((i + c) % 3) as i32).collect();
                    atom_at(gt, src, i as u32)
                })
                .collect();

            // Reference: one full-grid sequential pack.
            let mut expect = BitGrid3::zeros(v, n_samples, ploidy);
            pack_presence_seq(&mut expect.words, &atoms, columns);

            // Windowed: flush every `window` atoms, where `window` is a multiple of
            // the word-aligned block size `g`.
            let g = 64 / gcd(columns, 64);
            let window = 4 * g;
            let mut got = BitGrid3::zeros(v, n_samples, ploidy);
            let mut metas: Vec<AtomMeta> = Vec::new();
            let mut buf: Vec<PendingAtom> = Vec::new();
            let mut v0 = 0usize;
            for a in atoms {
                buf.push(a);
                if buf.len() == window {
                    let n = buf.len();
                    flush_window(&mut got, &mut metas, &mut buf, v0, columns, Some(test_pool()));
                    v0 += n;
                }
            }
            if !buf.is_empty() {
                flush_window(&mut got, &mut metas, &mut buf, v0, columns, Some(test_pool()));
            }

            prop_assert_eq!(got.words, expect.words);
            prop_assert_eq!(metas.len(), v);
        }
    }
```

The existing `atom(gt, src)` helper hardcodes `pos: 0`. Add a sibling that also sets `pos`/`seq` so `metas` ordering is checkable:

```rust
    fn atom_at(gt: Vec<i32>, src: u16, pos: u32) -> PendingAtom {
        PendingAtom {
            pos,
            ilen: 0,
            alt: Vec::new(),
            source_alt_index: src,
            gt: std::sync::Arc::new(gt),
            seq: pos as u64,
            info_vals: Vec::new(),
            format_vals: Vec::new(),
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features windowed_pack_matches_full_pack'`
Expected: FAIL to compile — `cannot find function 'flush_window'`, `cannot find type 'AtomMeta'`.

- [ ] **Step 3: Write minimal implementation**

In `src/vcf_reader.rs`, above `pub struct VcfChunkReader`, add:

```rust
// An atom whose presence bits are already packed into the chunk's BitGrid. `gt`
// and `source_alt_index` are dropped at that point, so per-chunk staging memory
// no longer scales with `chunk_size * num_samples * ploidy`.
struct AtomMeta {
    pos: u32,
    ilen: i32,
    alt: Vec<u8>,
    info_vals: Vec<f64>,
    format_vals: Vec<f64>,
}

// Atoms buffered before their presence bits are flushed into the chunk's BitGrid.
// Rounded UP to a multiple of the word-aligned block size `g = 64/gcd(columns,64)`
// at call time, so every flush offset lands on a u64 boundary and
// `pack_presence_par` keeps its word-disjoint invariant. 1024 keeps the window
// above `PARALLEL_MIN_VARIANTS` (512) so parallel packing still engages.
const PACK_WINDOW: usize = 1024;

// Pack `buf`'s presence bits into `genos` starting at variant offset `v0`, then
// move each atom's metadata into `metas`, dropping `gt`.
//
// `v0` MUST be a multiple of the word-aligned block size, so `v0 * columns` is a
// multiple of 64 and the window owns a whole-word-aligned sub-slice of
// `genos.words`. Only the FINAL window may have a length that is not a multiple of
// that block size (its trailing partial word is not shared, because nothing is
// packed after it).
fn flush_window(
    genos: &mut BitGrid3,
    metas: &mut Vec<AtomMeta>,
    buf: &mut Vec<PendingAtom>,
    v0: usize,
    columns: usize,
    pool: Option<&rayon::ThreadPool>,
) {
    if buf.is_empty() {
        return;
    }
    debug_assert_eq!((v0 * columns) % 64, 0, "flush offset must be word-aligned");
    let word_base = (v0 * columns) / 64;
    let n_words = (buf.len() * columns).div_ceil(64);
    let words = &mut genos.words[word_base..word_base + n_words];

    let parallel = matches!(pool, Some(p) if p.current_num_threads() >= 2)
        && buf.len() >= PARALLEL_MIN_VARIANTS;
    if parallel {
        pack_presence_par(words, buf, columns, pool.unwrap());
    } else {
        pack_presence_seq(words, buf, columns);
    }

    metas.reserve(buf.len());
    for a in buf.drain(..) {
        metas.push(AtomMeta {
            pos: a.pos,
            ilen: a.ilen,
            alt: a.alt,
            info_vals: a.info_vals,
            format_vals: a.format_vals,
        });
    }
}
```

Then replace the body of `read_next_chunk` (currently lines ~592-670) with:

```rust
    // Pull up to `chunk_size` atoms (already globally position-sorted) and pack them
    // into a variant-major DenseChunk. Presence bits are packed in windows of
    // `PACK_WINDOW` atoms so the reader never holds more than one window's worth of
    // per-column genotype vectors. `pool`, when present and the window is large
    // enough, hosts parallel packing; otherwise packing is sequential. Output is
    // bit-identical either way. Returns None once no atoms remain.
    pub fn read_next_chunk(
        &mut self,
        chunk_size: usize,
        chunk_id: usize,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Option<DenseChunk>, ConversionError> {
        let columns = self.num_samples * self.ploidy;
        // Word-aligned block size: `g` variants span exactly `columns/gcd` u64 words.
        let g = 64 / gcd(columns, 64);
        let window = PACK_WINDOW.div_ceil(g) * g;

        // Allocate for the full chunk up front (packed size: chunk_size*columns bits),
        // then shrink to the true variant count after EOF.
        let mut genos = BitGrid3::zeros(chunk_size, self.num_samples, self.ploidy);
        let mut metas: Vec<AtomMeta> = Vec::with_capacity(chunk_size);
        let mut buf: Vec<PendingAtom> = Vec::with_capacity(window);
        let mut v = 0usize;

        while v + buf.len() < chunk_size {
            match self.next_atom()? {
                Some(a) => {
                    buf.push(a);
                    if buf.len() == window {
                        flush_window(&mut genos, &mut metas, &mut buf, v, columns, pool);
                        v += window;
                    }
                }
                None => break,
            }
        }
        if !buf.is_empty() {
            let n = buf.len();
            flush_window(&mut genos, &mut metas, &mut buf, v, columns, pool);
            v += n;
        }

        if v == 0 {
            return Ok(None);
        }
        genos.truncate_v(v);

        let num_samples = self.num_samples;
        let mut pos = Vec::with_capacity(v);
        let mut ilens = Vec::with_capacity(v);
        let mut alt = Vec::with_capacity(v * 2);
        let mut alt_offsets = Vec::with_capacity(v + 1);
        alt_offsets.push(0u32);
        let mut info_staged: Vec<StagedColumn> = self
            .info_fields
            .iter()
            .map(|spec| StagedColumn::with_capacity(spec.stage_is_float(), v))
            .collect();
        let mut format_staged: Vec<StagedColumn> = self
            .format_fields
            .iter()
            .map(|spec| StagedColumn::with_capacity(spec.stage_is_float(), v * num_samples))
            .collect();

        // Sequential metadata pass (cheap, ordering-preserving).
        let mut off = 0u32;
        for a in metas.iter() {
            pos.push(a.pos);
            ilens.push(a.ilen);
            alt.extend_from_slice(&a.alt);
            off += a.alt.len() as u32;
            alt_offsets.push(off);

            for (i, col) in info_staged.iter_mut().enumerate() {
                col.push_f64(a.info_vals[i]);
            }
            for (j, col) in format_staged.iter_mut().enumerate() {
                for s in 0..num_samples {
                    col.push_f64(a.format_vals[j * num_samples + s]);
                }
            }
        }

        Ok(Some(DenseChunk {
            chunk_id,
            pos,
            ilens,
            alt,
            alt_offsets,
            genos,
            info_staged,
            format_staged,
        }))
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: PASS, including `windowed_pack_matches_full_pack` and the pre-existing `pack_presence_par`-vs-`seq` proptests.

Run: `pixi run pytest tests/test_svar2_from_vcf.py tests/test_svar2.py tests/test_svar2_fields.py -q`
Expected: PASS. **This is the byte-identical-output guard** — the VCF path's stored output must not change.

- [ ] **Step 5: Commit**

```bash
git add src/vcf_reader.rs
git commit -m "perf(svar2): pack presence bits in windows, drop gt eagerly

The reader buffered chunk_size PendingAtoms, each holding an S*P i32 genotype
vector -- 32x the packed bit-grid they produce. Pack in PACK_WINDOW-sized,
word-aligned windows and drop gt as soon as its bits are set, so reader staging
memory is bounded by window*S*P*4 instead of chunk_size*S*P*4. Output is
bit-identical."
```

---

## Task 2: Extract `RecordSource` + `ChunkAssembler`

Pure refactor. `VcfChunkReader` currently fuses htslib iteration, atom decomposition, the reorder heap, chunk packing, and field staging. Only the first is VCF-specific.

**Files:**
- Create: `src/record_source.rs`
- Create: `src/chunk_assembler.rs`
- Modify: `src/vcf_reader.rs` (becomes `VcfRecordSource`; loses everything moved out)
- Modify: `src/lib.rs` (declare the two new modules)
- Modify: `src/orchestrator.rs` (construct `ChunkAssembler::new(Box::new(VcfRecordSource::new(...)), ...)`)

**Interfaces:**
- Produces (consumed by Tasks 4 and 5):
  - `record_source::RawRecord { pos: u32, reference: Vec<u8>, alts: Vec<Vec<u8>>, gt: Vec<i32>, info_raw: Vec<Option<Vec<f64>>>, format_raw: Vec<Option<Vec<Vec<f64>>>> }`
  - `record_source::RecordSource` trait with `fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError>`
  - `chunk_assembler::ChunkAssembler::new(source: Box<dyn RecordSource + Send>, num_samples: usize, ploidy: usize, fasta_path: Option<&str>, chrom: &str, skip_out_of_scope: bool, fields: &[FieldSpec]) -> Result<Self, ConversionError>`
  - `ChunkAssembler::read_next_chunk(&mut self, chunk_size: usize, chunk_id: usize, pool: Option<&rayon::ThreadPool>) -> Result<Option<DenseChunk>, ConversionError>`
  - `ChunkAssembler::dropped_out_of_scope(&self) -> u64`
  - `vcf_reader::VcfRecordSource::new(vcf_path: &str, chrom: &str, samples: &[&str], htslib_threads: usize, ploidy: usize, fields: &[FieldSpec]) -> Result<Self, ConversionError>`
  - `vcf_reader::load_contig_seq` stays where it is (the orchestrator's `signatures` pass already calls it).

**Design note — `RawRecord` is OWNED, not borrowed.** A borrowed `RawRecord<'a>` would fight the borrow checker inside `ChunkAssembler::next_atom` (source borrowed while the heap is mutated). Owning costs nothing: the VCF path *already* allocates `ref_allele.to_vec()`, `alts_owned`, and a fresh `gt` vec per record, and the assembler needs `gt` in an `Arc<Vec<i32>>` regardless.

**Design note — `format_raw` sample indexing.** Today `decode_format_raw` returns buffers indexed by *header* sample index, and `decompose_current_record` remaps via `self.sample_indices`. The assembler cannot know about `sample_indices` (PGEN has no such mapping). So `VcfRecordSource` does the remap when it builds `RawRecord`: `format_raw[j][selected_sample_index]`. `PgenRecordSource` always emits an empty `format_raw`.

- [ ] **Step 1: Create `src/record_source.rs`**

```rust
//! The seam between "where variant records come from" (VCF via htslib, PGEN via
//! pgenlib) and the source-agnostic conversion spine (`chunk_assembler`).
//!
//! `RawRecord` is deliberately OWNED rather than borrowed: the assembler mutates
//! its heap while consuming a record, and the VCF path already allocates every one
//! of these fields per record anyway, so owning costs nothing.

use crate::error::ConversionError;

/// One variant record, decoded to the minimum the conversion spine needs.
pub struct RawRecord {
    /// 0-based start position (VCF/pvar POS minus 1).
    pub pos: u32,
    /// REF allele bases, uppercase ASCII.
    pub reference: Vec<u8>,
    /// ALT alleles in file order. ALT1 is `alts[0]`; note `PendingAtom::
    /// source_alt_index` is 1-based (ALT1 => 1), matching BCF GT allele codes.
    pub alts: Vec<Vec<u8>>,
    /// Allele index per haplotype column, length `num_samples * ploidy`,
    /// sample-major then ploidy-minor. `0` = REF, `k` = ALT k, `-1` = missing.
    pub gt: Vec<i32>,
    /// Raw INFO buffers, widened to f64, one entry per requested INFO `FieldSpec`
    /// in spec order. `None` = the field is absent from this record. Empty when no
    /// INFO fields were requested.
    pub info_raw: Vec<Option<Vec<f64>>>,
    /// Raw FORMAT buffers, widened to f64: outer index = requested FORMAT
    /// `FieldSpec` in spec order, inner index = **selected** sample
    /// (`0..num_samples`, already remapped from the source's own sample order).
    /// Outer `None` = the field is absent from this record for all samples. Empty
    /// when no FORMAT fields were requested.
    pub format_raw: Vec<Option<Vec<Vec<f64>>>>,
}

/// A cursor over one contig's variant records, in file order.
pub trait RecordSource {
    /// Next record, or `None` at end of contig.
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError>;
}
```

- [ ] **Step 2: Create `src/chunk_assembler.rs`**

Move, verbatim from `src/vcf_reader.rs`: `PendingAtom` (+ its `PartialEq`/`Eq`/`PartialOrd`/`Ord` impls), `pack_row`, `pack_presence_seq`, `PARALLEL_MIN_VARIANTS`, `gcd`, `pack_presence_par`, `sentinel_default`, `is_htslib_missing`, `resolve_scalar`, `AtomMeta`, `PACK_WINDOW`, `flush_window`, and the whole `mod tests` block. Then add the assembler:

```rust
pub struct ChunkAssembler {
    source: Box<dyn RecordSource + Send>,
    num_samples: usize,
    ploidy: usize,
    /// Full 0-based contig sequence, uppercased; empty when no reference was given.
    ref_seq: Vec<u8>,
    has_reference: bool,
    skip_out_of_scope: bool,
    dropped_out_of_scope: u64,
    info_fields: Vec<FieldSpec>,
    format_fields: Vec<FieldSpec>,
    heap: BinaryHeap<Reverse<PendingAtom>>,
    frontier: u32,
    eof: bool,
    next_seq: u64,
}

impl ChunkAssembler {
    pub fn new(
        source: Box<dyn RecordSource + Send>,
        num_samples: usize,
        ploidy: usize,
        fasta_path: Option<&str>,
        chrom: &str,
        skip_out_of_scope: bool,
        fields: &[FieldSpec],
    ) -> Result<Self, ConversionError> {
        let (ref_seq, has_reference) = match fasta_path {
            Some(path) => (crate::vcf_reader::load_contig_seq(path, chrom)?, true),
            None => (Vec::new(), false),
        };
        Ok(Self {
            source,
            num_samples,
            ploidy,
            ref_seq,
            has_reference,
            skip_out_of_scope,
            dropped_out_of_scope: 0,
            info_fields: fields
                .iter()
                .filter(|f| f.category == FieldCategory::Info)
                .cloned()
                .collect(),
            format_fields: fields
                .iter()
                .filter(|f| f.category == FieldCategory::Format)
                .cloned()
                .collect(),
            heap: BinaryHeap::new(),
            frontier: 0,
            eof: false,
            next_seq: 0,
        })
    }

    /// Total out-of-scope (symbolic/breakend) ALTs dropped so far. Valid after the
    /// read loop drains.
    pub fn dropped_out_of_scope(&self) -> u64 {
        self.dropped_out_of_scope
    }

    // Decompose one record into atoms and push them onto the reorder heap, sharing
    // one decoded genotype vector across all atoms of the record.
    fn decompose_record(&mut self, rec: RawRecord) -> Result<(), ConversionError> {
        let pos = rec.pos;
        let gt = Arc::new(rec.gt);

        // Fail fast only when a reference is available; without one we trust the
        // input is already normalized/left-aligned.
        if self.has_reference {
            crate::normalize::validate_ref(pos, &rec.reference, &self.ref_seq)?;
        }

        let alt_refs: Vec<&[u8]> = rec.alts.iter().map(|a| a.as_slice()).collect();
        let mut atoms = Vec::new();
        let dropped = atomize_record(
            pos,
            &rec.reference,
            &alt_refs,
            &mut atoms,
            self.skip_out_of_scope,
        )?;
        self.dropped_out_of_scope += dropped as u64;

        for atom in atoms {
            let atom = if self.has_reference {
                crate::normalize::left_align(atom, &self.ref_seq, crate::normalize::L_MAX)
            } else {
                atom
            };

            let info_vals: Vec<f64> = self
                .info_fields
                .iter()
                .zip(rec.info_raw.iter())
                .map(|(spec, raw)| resolve_scalar(raw.as_deref(), atom.source_alt_index, spec))
                .collect();

            let mut format_vals = Vec::with_capacity(self.format_fields.len() * self.num_samples);
            for (spec, raw) in self.format_fields.iter().zip(rec.format_raw.iter()) {
                for s in 0..self.num_samples {
                    let sample_vals = raw.as_ref().map(|v| v[s].as_slice());
                    format_vals.push(resolve_scalar(sample_vals, atom.source_alt_index, spec));
                }
            }

            let seq = self.next_seq;
            self.next_seq += 1;
            self.heap.push(Reverse(PendingAtom {
                pos: atom.pos,
                ilen: atom.ilen,
                alt: atom.alt,
                source_alt_index: atom.source_alt_index,
                gt: Arc::clone(&gt),
                seq,
                info_vals,
                format_vals,
            }));
        }
        Ok(())
    }

    // Yield the next atom in global position order. Left-alignment can move an atom
    // up to `L_MAX` bases below its record's start, so an atom is safe to emit only
    // once its position is strictly below `frontier - L_MAX` (saturating), or the
    // input is exhausted. This preserves the position-sorted invariant the Phase-2
    // merge relies on. (Unchanged from the pre-refactor VcfChunkReader.)
    fn next_atom(&mut self) -> Result<Option<PendingAtom>, ConversionError> {
        loop {
            if let Some(Reverse(top)) = self.heap.peek() {
                if self.eof || top.pos < self.frontier.saturating_sub(crate::normalize::L_MAX) {
                    return Ok(Some(self.heap.pop().unwrap().0));
                }
            } else if self.eof {
                return Ok(None);
            }

            match self.source.next_record()? {
                Some(rec) => {
                    self.frontier = rec.pos;
                    self.decompose_record(rec)?;
                }
                None => self.eof = true,
            }
        }
    }

    // Body is EXACTLY the `read_next_chunk` written in Task 1, with `self.num_samples`
    // / `self.ploidy` / `self.info_fields` / `self.format_fields` reading from the
    // assembler instead of the old VcfChunkReader.
    pub fn read_next_chunk(
        &mut self,
        chunk_size: usize,
        chunk_id: usize,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Option<DenseChunk>, ConversionError> {
        // ... paste Task 1's read_next_chunk body verbatim ...
    }
}
```

with these imports at the top of the file:

```rust
use crate::error::ConversionError;
use crate::field::{FieldCategory, FieldSpec};
use crate::normalize::atomize_record;
use crate::record_source::{RawRecord, RecordSource};
use crate::types::{BitGrid3, DenseChunk, StagedColumn};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
```

- [ ] **Step 3: Reduce `src/vcf_reader.rs` to `VcfRecordSource`**

Delete everything moved to `chunk_assembler.rs`. Keep `load_contig_seq` (make it `pub(crate)`, unchanged), `HtslibType`-driven `decode_info_raw` / `decode_format_raw` (unchanged), and rewrite the struct:

```rust
pub struct VcfRecordSource {
    inner_reader: IndexedReader,
    num_samples: usize,
    ploidy: usize,
    sample_indices: Vec<usize>,
    info_fields: Vec<FieldSpec>,
    format_fields: Vec<FieldSpec>,
    record: Record,
    eof: bool,
}

impl VcfRecordSource {
    pub fn new(
        vcf_path: &str,
        chrom: &str,
        samples: &[&str],
        htslib_threads: usize,
        ploidy: usize,
        fields: &[FieldSpec],
    ) -> Result<Self, ConversionError> {
        // Body is the old VcfChunkReader::new MINUS the fasta/ref_seq handling
        // (that moved to ChunkAssembler::new) and MINUS the heap/frontier/next_seq
        // fields. Keep the MissingFile check, IndexedReader::from_path, set_threads,
        // name2rid, fetch(rid, 0, None), and the sample_indices lookup verbatim.
        // ...
    }
}

impl RecordSource for VcfRecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        if self.eof {
            return Ok(None);
        }
        match self.inner_reader.read(&mut self.record) {
            None => {
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
```

- [ ] **Step 4: Wire up `src/lib.rs` and `src/orchestrator.rs`**

In `src/lib.rs`, next to the existing `mod` declarations, add (keeping the `conversion` feature gating that `vcf_reader` already has):

```rust
#[cfg(feature = "conversion")]
mod chunk_assembler;
#[cfg(feature = "conversion")]
mod record_source;
```

In `src/orchestrator.rs`, replace the reader construction inside the reader thread closure:

```rust
                let s_refs: Vec<&str> = s_owned.iter().map(|s| s.as_str()).collect();
                let source = crate::vcf_reader::VcfRecordSource::new(
                    &vcf,
                    &chr,
                    &s_refs,
                    htslib_threads,
                    ploidy,
                    &fields_owned,
                )?;
                let mut reader = crate::chunk_assembler::ChunkAssembler::new(
                    Box::new(source),
                    s_refs.len(),
                    ploidy,
                    fasta.as_deref(),
                    &chr,
                    skip_out_of_scope,
                    &fields_owned,
                )?;
                let mut chunk_id = 0;
                while let Some(dense_chunk) =
                    reader.read_next_chunk(chunk_size, chunk_id, Some(&pool))?
                {
                    tx_dense.send(dense_chunk).unwrap();
                    chunk_id += 1;
                }
                Ok(reader.dropped_out_of_scope())
```

- [ ] **Step 5: Run the full suite — this is the regression gate**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: PASS (all moved proptests still pass under their new module).

Run: `pixi run test`
Expected: PASS. The VCF path must be **byte-identical**; any diff in `tests/test_svar2*.py` output means the refactor changed behavior.

Run: `pixi run bash -lc 'cargo clippy --all-targets -- -D warnings && cargo fmt --check'`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/record_source.rs src/chunk_assembler.rs src/vcf_reader.rs src/lib.rs src/orchestrator.rs
git commit -m "refactor(svar2): split VcfChunkReader into RecordSource + ChunkAssembler

VcfChunkReader fused htslib iteration, atom decomposition, the reorder heap,
presence packing, and field staging; only the first is VCF-specific. Extract the
source-agnostic spine into ChunkAssembler behind a RecordSource trait yielding an
owned RawRecord, leaving VcfRecordSource as the htslib adapter. No behavior
change -- SVAR2 output is byte-identical."
```

---

## Task 3: Streaming `.pvar` reader

**Files:**
- Create: `src/pvar.rs`
- Modify: `Cargo.toml` (add `zstd`)
- Modify: `src/lib.rs` (declare `mod pvar`)
- Test: `src/pvar.rs` `mod tests`

**Interfaces:**
- Produces (consumed by Task 4):
  - `pvar::PvarRecord { pub pos: u32, pub reference: Vec<u8>, pub alts: Vec<Vec<u8>> }`
  - `pvar::PvarReader::open(path: &str, var_start: usize) -> Result<PvarReader, ConversionError>`
  - `PvarReader::next_variant(&mut self) -> Result<Option<PvarRecord>, ConversionError>`

- [ ] **Step 1: Add the `zstd` dependency**

In `Cargo.toml`, under `[dependencies]` (alphabetical, after `thiserror`):

```toml
# .pvar.zst decompression for the PGEN record source. BSD-3/MIT; no plink-ng code.
zstd = { version = "0.13", optional = true }
```

and extend the `conversion` feature:

```toml
conversion = ["dep:rust-htslib", "dep:zstd"]
```

- [ ] **Step 2: Write the failing tests**

Create `src/pvar.rs` with only the test module first:

```rust
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pixi run bash -lc 'cargo test --no-default-features pvar'`
Expected: FAIL to compile — `cannot find type 'PvarReader'`.

- [ ] **Step 4: Implement the reader**

Prepend to `src/pvar.rs`:

```rust
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
    /// ALT alleles, comma-split. ALT1 is `alts[0]`.
    pub alts: Vec<Vec<u8>>,
}

pub struct PvarReader {
    lines: Box<dyn BufRead + Send>,
    path: String,
    pos_col: usize,
    ref_col: usize,
    alt_col: usize,
    /// Global variant index of the next row to be returned. Only used for error
    /// messages; the caller tracks its own range.
    vidx: usize,
    buf: String,
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
        let alts: Vec<Vec<u8>> = cols[self.alt_col]
            .split(',')
            .map(|a| {
                let mut v = a.as_bytes().to_vec();
                v.make_ascii_uppercase();
                v
            })
            .collect();

        self.vidx += 1;
        Ok(Some(PvarRecord {
            pos: pos_1based - 1,
            reference,
            alts,
        }))
    }
}
```

Declare it in `src/lib.rs` alongside the other conversion-gated modules:

```rust
#[cfg(feature = "conversion")]
mod pvar;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features pvar'`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/pvar.rs src/lib.rs
git commit -m "feat(svar2): streaming .pvar/.pvar.zst variant-metadata reader

PGEN stores genotypes only; POS/REF/ALT come from the sibling .pvar. Streams
rather than materializing, so per-contig metadata memory stays O(1) in variants.
Written against the PLINK2 text format; contains no plink-ng code."
```

---

## Task 4: `PgenRecordSource`

**Files:**
- Create: `src/pgen_reader.rs`
- Modify: `src/lib.rs` (declare `mod pgen_reader`)

**Interfaces:**
- Consumes: `record_source::{RawRecord, RecordSource}` (Task 2), `pvar::{PvarReader, PvarRecord}` (Task 3).
- Produces (consumed by Task 5):
  - `pgen_reader::PGEN_BATCH_BYTES: usize`
  - `pgen_reader::PgenRecordSource::new(pgen_reader: Py<PyAny>, pvar_path: &str, var_start: usize, var_end: usize, num_samples: usize, chunk_size: usize) -> Result<PgenRecordSource, ConversionError>`
  - `impl RecordSource for PgenRecordSource`

**Design notes:**
- `pgen_reader` is a **`pgenlib.PgenReader` constructed in Python** and handed down as `Py<PyAny>` (`Py<T>` is `Send`, so it can cross into the `py.detach` worker threads). One reader per contig — each contig's reader seeks independently, so they must not be shared.
- `read_alleles_range(lo, hi, out, hap_maj=False)` fills `out[(hi-lo), 2*S]` int32 with per-haplotype allele codes (phase-aware, multiallelic-aware, missing `-9`) and runs its decode under `prange(..., nogil=True)` — the GIL is released for the actual work, so `Python::attach` here costs only call dispatch.
- Batch size is a **byte budget**, not a variant count: at 200 samples this is thousands of variants per Python call; at 500k samples it is a handful. That single knob keeps the FFI boundary cheap and memory bounded across the whole cohort-size range.
- `.pvar` and `.pgen` advance in lockstep — one `next_variant()` per served row.

- [ ] **Step 1: Write the implementation**

There is no pure-Rust unit test for this type (it needs a live `pgenlib.PgenReader`); it is covered end-to-end by Task 7's Python tests. Create `src/pgen_reader.rs`:

```rust
//! PGEN record source.
//!
//! Genotypes come from the `pgenlib` PyPI wheel (LGPL-3.0) via its **public Python
//! API** -- the same way `genoray/_pgen.py` already uses it. genoray links no
//! plink-ng code and vendors none; `_core` stays MIT. Do not change this without
//! reading `docs/superpowers/specs/2026-07-12-pgen-to-svar2-design.md`.
//!
//! `PgenReader.read_alleles_range` releases the GIL for its decode loop, so the
//! `Python::attach` below only costs call dispatch, not the decode.

use crate::error::ConversionError;
use crate::pvar::PvarReader;
use crate::record_source::{RawRecord, RecordSource};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Byte budget for the `(batch, 2 * n_samples)` int32 allele-code buffer. Sized so
/// small cohorts get thousands of variants per Python call (amortizing dispatch)
/// while biobank cohorts stay memory-bounded.
pub const PGEN_BATCH_BYTES: usize = 32 * 1024 * 1024;

pub struct PgenRecordSource {
    /// A `pgenlib.PgenReader`, constructed in Python and owned here.
    reader: Py<PyAny>,
    /// `(batch, 2 * num_samples)` int32 allele-code scratch buffer, reused per refill.
    buf: Py<PyArray2<i32>>,
    batch: usize,
    num_samples: usize,
    /// Next global variant index to fetch from the .pgen.
    var_next: usize,
    /// Exclusive end of this contig's variant index range.
    var_end: usize,
    /// Rows currently valid in `buf`, and the next one to serve.
    filled: usize,
    row: usize,
    pvar: PvarReader,
}

impl PgenRecordSource {
    pub fn new(
        reader: Py<PyAny>,
        pvar_path: &str,
        var_start: usize,
        var_end: usize,
        num_samples: usize,
        chunk_size: usize,
    ) -> Result<Self, ConversionError> {
        let batch = (PGEN_BATCH_BYTES / (2 * num_samples * 4)).clamp(1, chunk_size.max(1));
        let pvar = PvarReader::open(pvar_path, var_start)?;
        let buf = Python::attach(|py| {
            PyArray2::<i32>::zeros(py, [batch, 2 * num_samples], false).unbind()
        });
        Ok(Self {
            reader,
            buf,
            batch,
            num_samples,
            var_next: var_start,
            var_end,
            filled: 0,
            row: 0,
            pvar,
        })
    }

    /// Refill `buf` with the next `min(batch, var_end - var_next)` variants.
    /// Returns the number of rows filled (0 => this contig is exhausted).
    fn refill(&mut self) -> Result<usize, ConversionError> {
        let lo = self.var_next;
        if lo >= self.var_end {
            return Ok(0);
        }
        let hi = (lo + self.batch).min(self.var_end);
        Python::attach(|py| -> Result<(), ConversionError> {
            self.reader
                .bind(py)
                .call_method1(
                    "read_alleles_range",
                    (lo as u32, hi as u32, self.buf.bind(py), false),
                )
                .map_err(|e| {
                    ConversionError::Input(format!(
                        "pgenlib read_alleles_range({lo}, {hi}) failed: {e}"
                    ))
                })?;
            Ok(())
        })?;
        self.var_next = hi;
        self.filled = hi - lo;
        self.row = 0;
        Ok(self.filled)
    }
}

impl RecordSource for PgenRecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord>, ConversionError> {
        if self.row == self.filled && self.refill()? == 0 {
            return Ok(None);
        }

        // .pvar and .pgen advance in lockstep -- one metadata row per genotype row.
        let Some(meta) = self.pvar.next_variant()? else {
            return Err(ConversionError::Input(
                "pvar ran out of variants before the .pgen did; \
                 the .pvar and .pgen disagree on variant count"
                    .to_string(),
            ));
        };

        let columns = 2 * self.num_samples;
        let mut gt = vec![-1i32; columns];
        Python::attach(|py| {
            let arr = self.buf.bind(py).readonly();
            let flat = arr.as_slice().expect("pgen buffer is C-contiguous");
            let base = self.row * columns;
            for (c, out) in gt.iter_mut().enumerate() {
                // pgenlib encodes missing as -9; the conversion spine uses -1.
                let code = flat[base + c];
                *out = if code < 0 { -1 } else { code };
            }
        });
        self.row += 1;

        Ok(Some(RawRecord {
            pos: meta.pos,
            reference: meta.reference,
            alts: meta.alts,
            gt,
            // PGEN has no FORMAT, and .pvar INFO extraction is out of scope for v1.
            info_raw: Vec::new(),
            format_raw: Vec::new(),
        }))
    }
}
```

Declare it in `src/lib.rs`:

```rust
#[cfg(feature = "conversion")]
mod pgen_reader;
```

- [ ] **Step 2: Verify it compiles and lints**

Run: `pixi run bash -lc 'cargo check && cargo clippy --all-targets -- -D warnings && cargo fmt --check'`
Expected: clean. (No new tests here — Task 7's end-to-end tests are what exercise this.)

- [ ] **Step 3: Commit**

```bash
git add src/pgen_reader.rs src/lib.rs
git commit -m "feat(svar2): PgenRecordSource backed by pgenlib read_alleles_range

Gets per-haplotype allele codes (phase- and multiallelic-aware) from the already-
depended-on pgenlib wheel through its public Python API, in byte-budgeted batches
so the FFI boundary stays cheap at 200 samples and memory-bounded at 500k. No
plink-ng code is linked or vendored."
```

---

## Task 5: Orchestrator `SourceSpec` + `run_pgen_conversion_pipeline`

**Files:**
- Modify: `src/orchestrator.rs` (`process_chromosome` takes a `SourceSpec`)
- Modify: `src/lib.rs` (add the `run_pgen_conversion_pipeline` pyfunction; register it)

**Interfaces:**
- Consumes: `PgenRecordSource::new` (Task 4), `VcfRecordSource::new` + `ChunkAssembler::new` (Task 2).
- Produces (consumed by Task 6):
  - `_core.run_pgen_conversion_pipeline(pgen_path: str, pvar_path: str, reference_path: str | None, chroms: list[str], contig_ranges: list[tuple[int, int]], output_dir: str, samples: list[str], chunk_size: int, max_threads: int | None, long_allele_capacity: int, skip_out_of_scope: bool, signatures: bool, pgen_readers: list[Any]) -> int`
  - `contig_ranges[i]` is the half-open `[var_start, var_end)` variant index range of `chroms[i]` in the `.pvar`. `pgen_readers[i]` is a distinct `pgenlib.PgenReader` for `chroms[i]`.

- [ ] **Step 1: Add `SourceSpec` and make `process_chromosome` generic over it**

In `src/orchestrator.rs`, above `process_chromosome`:

```rust
/// Which backend a contig's records come from. Everything downstream of
/// `ChunkAssembler` is identical for both.
pub enum SourceSpec {
    Vcf {
        vcf_path: String,
        htslib_threads: usize,
    },
    Pgen {
        pgen_path: String,
        pvar_path: String,
        var_start: usize,
        var_end: usize,
        /// A `pgenlib.PgenReader` for THIS contig. Readers seek independently, so
        /// each concurrent contig needs its own -- never share one.
        reader: pyo3::Py<pyo3::PyAny>,
    },
}
```

Change `process_chromosome`'s signature: drop `vcf_path: &str` and `htslib_threads: usize`, add `source: SourceSpec`. Inside the reader thread closure, replace the `VcfChunkReader::new` call with:

```rust
                let s_refs: Vec<&str> = s_owned.iter().map(|s| s.as_str()).collect();
                let src: Box<dyn crate::record_source::RecordSource + Send> = match source {
                    SourceSpec::Vcf {
                        vcf_path,
                        htslib_threads,
                    } => Box::new(crate::vcf_reader::VcfRecordSource::new(
                        &vcf_path,
                        &chr,
                        &s_refs,
                        htslib_threads,
                        ploidy,
                        &fields_owned,
                    )?),
                    SourceSpec::Pgen {
                        pgen_path: _,
                        pvar_path,
                        var_start,
                        var_end,
                        reader,
                    } => Box::new(crate::pgen_reader::PgenRecordSource::new(
                        reader,
                        &pvar_path,
                        var_start,
                        var_end,
                        s_refs.len(),
                        chunk_size,
                    )?),
                };
                let mut reader = crate::chunk_assembler::ChunkAssembler::new(
                    src,
                    s_refs.len(),
                    ploidy,
                    fasta.as_deref(),
                    &chr,
                    skip_out_of_scope,
                    &fields_owned,
                )?;
```

`source` must be `move`d into the closure (it already captures by `move`). `SourceSpec::Pgen.pgen_path` is unused by the reader (pgenlib already holds the open file) but is kept for error messages and symmetry — silence the unused binding with `pgen_path: _` as above.

**Note on the thread budget:** the PGEN path needs no htslib decompression threads. Task 6 calls `plan_thread_budget` exactly as the VCF path does and simply ignores `htslib_threads`; those cores go idle rather than being reallocated. That is deliberate — reallocating is a separate, measurable optimization, not a guess.

- [ ] **Step 2: Add the pyfunction in `src/lib.rs`**

Model it directly on the existing `run_conversion_pipeline` (same budgeting, same rayon fan-out, same error aggregation). The only differences are the `SourceSpec` it builds and the absent `fields`.

```rust
/// Convert a PLINK2 PGEN to an SVAR2 store.
///
/// `contig_ranges[i]` is the half-open `[var_start, var_end)` variant index range
/// of `chroms[i]` within the `.pvar`. `pgen_readers[i]` is a distinct
/// `pgenlib.PgenReader` for `chroms[i]` -- readers seek independently, so contigs
/// must not share one.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_pgen_conversion_pipeline(
    py: Python,
    pgen_path: String,
    pvar_path: String,
    reference_path: Option<String>,
    chroms: Vec<String>,
    contig_ranges: Vec<(usize, usize)>,
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize,
    max_threads: Option<usize>,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    signatures: bool,
    pgen_readers: Vec<Py<PyAny>>,
) -> PyResult<usize> {
    if chroms.len() != contig_ranges.len() || chroms.len() != pgen_readers.len() {
        return Err(PyValueError::new_err(
            "chroms, contig_ranges, and pgen_readers must be the same length",
        ));
    }
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
    // PGEN is diploid-only.
    let ploidy = 2usize;
    // PGEN carries no FORMAT, and .pvar INFO extraction is out of scope.
    let fields: Vec<crate::field::FieldSpec> = Vec::new();

    // Pair each contig with its own reader BEFORE detaching, so the Py handles move
    // into the worker threads (Py<PyAny> is Send; PyAny is not).
    let jobs: Vec<(String, (usize, usize), Py<PyAny>)> = chroms
        .iter()
        .cloned()
        .zip(contig_ranges.iter().copied())
        .zip(pgen_readers)
        .map(|((c, r), rd)| (c, r, rd))
        .collect();

    let results: Vec<Result<u64, crate::error::ConversionError>> = py.detach(|| {
        let available_cores = match max_threads {
            Some(t) if t > 0 => t,
            _ => std::thread::available_parallelism().unwrap().get(),
        };
        let plan = crate::budget::plan_thread_budget(available_cores, jobs.len());
        let concurrent_chroms = plan.concurrent_chroms;
        let processing_threads = plan.processing_threads;
        println!(
            "Pipeline Config (PGEN): {} concurrent chromosomes | {} processing threads each.",
            concurrent_chroms, processing_threads
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrent_chroms)
            .thread_name(|i| format!("chrom-{}", i))
            .build()
            .expect("build chrom pool");

        pool.install(|| {
            use rayon::prelude::*;
            jobs.into_par_iter()
                .map(|(chrom, (lo, hi), reader)| {
                    crate::orchestrator::process_chromosome(
                        crate::orchestrator::SourceSpec::Pgen {
                            pgen_path: pgen_path.clone(),
                            pvar_path: pvar_path.clone(),
                            var_start: lo,
                            var_end: hi,
                            reader,
                        },
                        reference_path.as_deref(),
                        &chrom,
                        &output_dir,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        long_allele_capacity,
                        skip_out_of_scope,
                        processing_threads,
                        signatures,
                        &fields,
                    )
                })
                .collect()
        })
    });

    let mut dropped = 0u64;
    for r in results {
        dropped += r.map_err(crate::error::to_pyerr)?;
    }
    Ok(dropped as usize)
}
```

Adjust the argument order of the `process_chromosome` call to match whatever the refactored signature actually is, and reuse the **existing** error-conversion helper that `run_conversion_pipeline` uses (read it; do not invent `to_pyerr` if the codebase names it differently). Register the function next to the others:

```rust
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_pgen_conversion_pipeline, m)?)?;
```

- [ ] **Step 3: Build and verify the VCF path still works**

Run: `pixi run bash -lc 'cargo clippy --all-targets -- -D warnings && cargo fmt --check'`
Expected: clean.

Run: `pixi run bash -lc 'maturin develop'` (foreground — this takes minutes; do NOT background it)
Expected: builds.

Run: `pixi run pytest tests/test_svar2_from_vcf.py -q`
Expected: PASS — the VCF path is unaffected by the `SourceSpec` plumbing.

- [ ] **Step 4: Commit**

```bash
git add src/orchestrator.rs src/lib.rs
git commit -m "feat(svar2): SourceSpec + run_pgen_conversion_pipeline

process_chromosome now selects its RecordSource from a SourceSpec, so the PGEN
backend reuses the whole executor/merge/finalize spine. One pgenlib.PgenReader per
contig -- readers seek independently and must not be shared."
```

---

## Task 6: `SparseVar2.from_pgen`

**Files:**
- Modify: `python/genoray/_svar2.py`
- Test: `tests/test_svar2_from_pgen.py` (created here, extended in Task 7)

**Interfaces:**
- Consumes: `_core.run_pgen_conversion_pipeline` (Task 5), and from `genoray._pgen`: `_read_psam`, `_scan_pvar`.
- Produces: `SparseVar2.from_pgen(...) -> int` (see the signature below).

- [ ] **Step 1: Write the failing test**

Create `tests/test_svar2_from_pgen.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genoray import SparseVar2

# 40 bp reference. 1-based POS 3 = 'A', 7 = 'C', 12..14 = 'GTA'.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"

# Phased, no half-calls, no symbolics: plink2's VCF import is lossless here, so
# from_pgen and from_vcf must agree exactly. (A half-call like './1' would NOT
# round-trip: gen_from_vcf.sh passes --vcf-half-call r, which rewrites it.)
_VCF_BODY = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=40>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
    "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
    "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    "chr1\t12\t.\tGTA\tG,GT\t.\t.\t.\tGT\t1|2\t0|1\n"
    "chr1\t20\t.\tT\tA\t.\t.\t.\tGT\t.|.\t1|0\n"
)


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> tuple[Path, Path, Path]:
    """(reference fasta, bgzipped+indexed vcf, pgen) for the same variants."""
    d = tmp_path_factory.mktemp("frompgen")

    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    plain = d / "in.vcf"
    plain.write_text(_VCF_BODY)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)

    subprocess.run(
        ["plink2", "--make-pgen", "--vcf", str(gz), "--out", str(d / "in")],
        check=True,
    )
    return ref, gz, d / "in.pgen"


def test_from_pgen_matches_from_vcf(sources, tmp_path):
    ref, vcf, pgen = sources
    from_vcf = tmp_path / "vcf.svar2"
    from_pgen = tmp_path / "pgen.svar2"

    SparseVar2.from_vcf(from_vcf, vcf, ref)
    SparseVar2.from_pgen(from_pgen, pgen, ref)

    a = SparseVar2(from_vcf)
    b = SparseVar2(from_pgen)
    assert a.n_samples == b.n_samples == 2

    regions = [(0, len(_REF))]
    ragged_vcf = a.decode("chr1", regions)
    ragged_pgen = b.decode("chr1", regions)
    assert ragged_pgen.offsets.tolist() == ragged_vcf.offsets.tolist()
    assert ragged_pgen.data.tolist() == ragged_vcf.data.tolist()


def test_from_pgen_requires_exactly_one_of_reference_or_no_reference(sources, tmp_path):
    _, _, pgen = sources
    with pytest.raises(ValueError, match="exactly one"):
        SparseVar2.from_pgen(tmp_path / "a.svar2", pgen)
    with pytest.raises(ValueError, match="exactly one"):
        SparseVar2.from_pgen(
            tmp_path / "b.svar2", pgen, "ref.fa", no_reference=True
        )


def test_from_pgen_refuses_to_overwrite(sources, tmp_path):
    ref, _, pgen = sources
    out = tmp_path / "exists.svar2"
    SparseVar2.from_pgen(out, pgen, ref)
    with pytest.raises(FileExistsError):
        SparseVar2.from_pgen(out, pgen, ref)
    SparseVar2.from_pgen(out, pgen, ref, overwrite=True)  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar2_from_pgen.py -q`
Expected: FAIL with `AttributeError: type object 'SparseVar2' has no attribute 'from_pgen'`.

- [ ] **Step 3: Implement `from_pgen`**

In `python/genoray/_svar2.py`, add a classmethod after `from_vcf`:

```python
    @classmethod
    def from_pgen(
        cls,
        out: str | Path,
        source: str | Path,
        reference: str | Path | None = None,
        *,
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        chunk_size: int | None = None,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
        signatures: bool = False,
    ) -> int:
        """Convert a PLINK2 PGEN to an SVAR2 store.

        Genotypes are read through the ``pgenlib`` package; variant metadata comes
        from the sibling ``.pvar``/``.pvar.zst`` and sample names from the ``.psam``.

        Exactly one of `reference` or `no_reference=True` is required, with the same
        meaning as :meth:`from_vcf`: with a reference, indels are validated against
        and left-aligned to the FASTA; with `no_reference`, both are skipped and the
        input is trusted to be already normalized. Returns the number of
        out-of-scope (symbolic/breakend) ALTs dropped (0 unless `skip_out_of_scope`).

        PGEN is diploid, so there is no `ploidy` parameter.

        chunk_size: variants per conversion chunk. Defaults to a value derived from
        a memory budget, since a packed dense chunk costs
        ``chunk_size * n_samples * 2 / 8`` bytes.

        Not supported (and silently ignored rather than errored, where noted):

        - **Dosages.** SVAR2 stores no dosages; a ``.pgen`` dosage track is ignored
          and hardcalls are read as usual.
        - **INFO/FORMAT fields.** PGEN has no FORMAT; ``.pvar`` INFO extraction is
          not implemented.
        - **Sample subsetting.** All samples in the ``.psam`` are converted, matching
          :meth:`from_vcf`.

        Haplotype resolution for *unphased* heterozygotes follows the allele-code
        order ``pgenlib`` returns — the same caveat :meth:`from_vcf` carries for
        unphased ``GT``.
        """
        from genoray._pgen import _read_psam, _scan_pvar

        out = Path(out)
        source = Path(source)

        if (reference is None) == (not no_reference):
            raise ValueError(
                "provide exactly one of `reference` (a FASTA path) or "
                "`no_reference=True`"
            )
        if signatures and no_reference:
            raise ValueError("signatures=True requires a reference (not no_reference).")
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        if source.suffix != ".pgen":
            raise ValueError(f"Expected a .pgen file, got {source}")
        if not source.exists():
            raise FileNotFoundError(source)

        pvar = _find_pvar(source)
        psam = source.with_suffix(".psam")
        if not psam.exists():
            raise FileNotFoundError(psam)
        out.parent.mkdir(parents=True, exist_ok=True)

        samples = cast("list[str]", _read_psam(psam).tolist())
        n_samples = len(samples)
        if n_samples == 0:
            raise ValueError(f"No samples found in {psam}.")

        contigs, ranges = _pvar_contig_ranges(pvar)
        if not contigs:
            raise ValueError(f"No variants found in {pvar}.")

        if chunk_size is None:
            chunk_size = _auto_chunk_size(n_samples)

        import pgenlib

        # One reader per contig: readers seek independently, so concurrent contigs
        # must not share one.
        readers = [
            pgenlib.PgenReader(bytes(source), n_samples) for _ in contigs
        ]

        return _core.run_pgen_conversion_pipeline(
            str(source),
            str(pvar),
            None if no_reference else str(reference),
            contigs,
            ranges,
            str(out),
            samples,
            chunk_size,
            threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            readers,
        )
```

and these module-level helpers in the same file:

```python
def _find_pvar(pgen: Path) -> Path:
    """Locate the `.pvar` / `.pvar.zst` sibling of `pgen`."""
    for suffix in (".pvar", ".pvar.zst"):
        cand = pgen.with_suffix(suffix)
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"No .pvar or .pvar.zst found next to {pgen}. "
        f"Looked for {pgen.with_suffix('.pvar')} and {pgen.with_suffix('.pvar.zst')}."
    )


def _pvar_contig_ranges(pvar: Path) -> tuple[list[str], list[tuple[int, int]]]:
    """Contigs in `.pvar` file order, with each one's half-open `[lo, hi)` variant
    index range.

    Raises if a contig's variants are not contiguous — SVAR2 converts one contig at
    a time from a variant index range, which requires the `.pvar` to be grouped by
    contig (as plink2 always writes it).
    """
    import polars as pl

    from genoray._pgen import _scan_pvar

    df = (
        _scan_pvar(pvar)
        .select("#CHROM")
        .with_row_index("vidx")
        .group_by("#CHROM", maintain_order=True)
        .agg(
            pl.col("vidx").min().alias("lo"),
            pl.col("vidx").max().alias("hi"),
            pl.len().alias("n"),
        )
        .collect()
    )
    contigs: list[str] = []
    ranges: list[tuple[int, int]] = []
    for chrom, lo, hi, n in df.iter_rows():
        if hi - lo + 1 != n:
            raise ValueError(
                f"Contig {chrom!r} is not contiguous in {pvar} "
                f"(spans indices {lo}..{hi} but has {n} variants). "
                "SVAR2 requires a .pvar grouped by contig."
            )
        contigs.append(str(chrom))
        ranges.append((int(lo), int(hi) + 1))
    return contigs, ranges


# Target byte size of one packed dense chunk (chunk_size * n_samples * ploidy / 8).
_DENSE_CHUNK_TARGET_BYTES = 256 * 1024 * 1024


def _auto_chunk_size(n_samples: int, ploidy: int = 2) -> int:
    """Variants per chunk, derived from a memory budget rather than a fixed count.

    A packed dense chunk costs `chunk_size * n_samples * ploidy / 8` bytes, so a
    fixed 25k chunk that is fine at 200 samples is not at 500k.
    """
    bits_per_variant = n_samples * ploidy
    by_budget = (_DENSE_CHUNK_TARGET_BYTES * 8) // max(bits_per_variant, 1)
    return max(1024, min(25_000, int(by_budget)))
```

Ensure `cast` is imported from `typing` at the top of the module if it is not already.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run bash -lc 'maturin develop'` (foreground)
Run: `pixi run pytest tests/test_svar2_from_pgen.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_svar2.py tests/test_svar2_from_pgen.py
git commit -m "feat(svar2): SparseVar2.from_pgen

Converts a PLINK2 PGEN to an SVAR2 store through the same normalization, atom,
and merge spine as from_vcf. Diploid-only (no ploidy kwarg); dosages, INFO/FORMAT
fields, and sample subsetting are out of scope and documented as such."
```

---

## Task 7: Coverage tests — multiallelic, missing, indels, `.pvar.zst`, no-reference

**Files:**
- Modify: `tests/test_svar2_from_pgen.py`

**Interfaces:**
- Consumes: `SparseVar2.from_pgen` (Task 6), the `sources` fixture from Task 6.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar2_from_pgen.py`:

```python
def test_from_pgen_no_reference_matches_from_vcf(sources, tmp_path):
    """The no_reference path (no REF validation, no left-alignment) must also agree."""
    _, vcf, pgen = sources
    from_vcf = tmp_path / "vcf.svar2"
    from_pgen = tmp_path / "pgen.svar2"

    SparseVar2.from_vcf(from_vcf, vcf, no_reference=True)
    SparseVar2.from_pgen(from_pgen, pgen, no_reference=True)

    a = SparseVar2(from_vcf)
    b = SparseVar2(from_pgen)
    regions = [(0, len(_REF))]
    assert b.decode("chr1", regions).data.tolist() == a.decode("chr1", regions).data.tolist()


def test_from_pgen_reads_zstd_pvar(sources, tmp_path):
    """plink2 `vzs` writes a .pvar.zst; the Rust streamer must handle it."""
    ref, vcf, _ = sources
    d = tmp_path / "zst"
    d.mkdir()
    subprocess.run(
        ["plink2", "--make-pgen", "vzs", "--vcf", str(vcf), "--out", str(d / "in")],
        check=True,
    )
    assert (d / "in.pvar.zst").exists()

    out_zst = tmp_path / "zst.svar2"
    out_ref = tmp_path / "plain.svar2"
    SparseVar2.from_pgen(out_zst, d / "in.pgen", ref)
    SparseVar2.from_vcf(out_ref, vcf, ref)

    regions = [(0, len(_REF))]
    a = SparseVar2(out_ref).decode("chr1", regions)
    b = SparseVar2(out_zst).decode("chr1", regions)
    assert b.data.tolist() == a.data.tolist()


def test_from_pgen_multi_contig(tmp_path):
    """Contigs are converted from disjoint .pvar index ranges; a two-contig file
    exercises the range computation and the per-contig PgenReader."""
    d = tmp_path / "multi"
    d.mkdir()

    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n>chr2\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        "##contig=<ID=chr2,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr2\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    subprocess.run(
        ["plink2", "--make-pgen", "--vcf", str(gz), "--out", str(d / "in")], check=True
    )

    from_vcf = tmp_path / "mv.svar2"
    from_pgen = tmp_path / "mp.svar2"
    SparseVar2.from_vcf(from_vcf, gz, ref)
    SparseVar2.from_pgen(from_pgen, d / "in.pgen", ref)

    a, b = SparseVar2(from_vcf), SparseVar2(from_pgen)
    regions = [(0, len(_REF))]
    for contig in ("chr1", "chr2"):
        assert (
            b.decode(contig, regions).data.tolist()
            == a.decode(contig, regions).data.tolist()
        )


def test_from_pgen_missing_pvar_is_a_clear_error(sources, tmp_path):
    _, _, pgen = sources
    lonely = tmp_path / "lonely.pgen"
    lonely.write_bytes(pgen.read_bytes())
    with pytest.raises(FileNotFoundError, match="No .pvar or .pvar.zst"):
        SparseVar2.from_pgen(tmp_path / "x.svar2", lonely, no_reference=True)
```

The multiallelic (`1|2`), missing (`.|.`), and indel (`C -> CAT`, `GTA -> G`) cases are already in `_VCF_BODY` from Task 6, so `test_from_pgen_matches_from_vcf` covers them; these tests add the no-reference path, zstd, multi-contig, and the sibling-file error.

- [ ] **Step 2: Run tests**

Run: `pixi run pytest tests/test_svar2_from_pgen.py -q`
Expected: PASS (7 tests).

If `test_from_pgen_matches_from_vcf` fails on the multiallelic or missing row, **do not "fix" the test** — that is a real bug in `PgenRecordSource`'s allele-code mapping. Debug with `pgenlib.PgenReader.read_alleles_range` directly in a scratch script and compare its codes against the VCF GTs.

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar2_from_pgen.py
git commit -m "test(svar2): from_pgen coverage for zstd pvar, multi-contig, no-reference"
```

---

## Task 8: CLI, skill doc, changelog

**Files:**
- Modify: `python/genoray/_cli/__main__.py`
- Modify: `skills/genoray-api/SKILL.md`
- Modify: `CHANGELOG.md`
- Test: `tests/cli/` (follow the existing CLI test layout)

**Interfaces:**
- Consumes: `SparseVar2.from_pgen` (Task 6).

- [ ] **Step 1: Write the failing CLI test**

Add to the existing CLI test module (`tests/cli/`, matching its current style — read it first):

```python
def test_write_dispatches_pgen(tmp_path, capsys):
    """`genoray write` already advertises VCF/PGEN in its help; make it true."""
    # Build ref + vcf + pgen exactly as tests/test_svar2_from_pgen.py's `sources`
    # fixture does, then:
    from genoray._cli.__main__ import app

    out = tmp_path / "cli.svar2"
    app(["write", str(pgen), str(out), "--reference", str(ref)])
    assert (out / "chr1").is_dir()
```

- [ ] **Step 2: Implement the dispatch**

In `python/genoray/_cli/__main__.py`, inside `write_svar2`, replace the direct `SparseVar2.from_vcf(...)` call with a suffix dispatch. Note `ploidy`, `chunk_size`, and `long_allele_capacity` are already CLI parameters; PGEN accepts `chunk_size` but not `ploidy`.

```python
    from genoray import SparseVar2

    skip_out_of_scope = skip_symbolics_and_breakends
    if source.suffix == ".pgen":
        if ploidy != 2:
            raise ValueError(
                "PGEN is diploid; --ploidy is only meaningful for VCF/BCF sources."
            )
        dropped = SparseVar2.from_pgen(
            out,
            source,
            reference,
            no_reference=no_reference,
            skip_out_of_scope=skip_out_of_scope,
            chunk_size=chunk_size,
            threads=threads,
            overwrite=overwrite,
            long_allele_capacity=long_allele_capacity,
        )
    else:
        dropped = SparseVar2.from_vcf(
            out,
            source,
            reference,
            no_reference=no_reference,
            skip_out_of_scope=skip_out_of_scope,
            ploidy=ploidy,
            chunk_size=chunk_size,
            threads=threads,
            overwrite=overwrite,
            long_allele_capacity=long_allele_capacity,
        )
```

The CLI's `chunk_size` currently defaults to `25_000`. Change its default to `None` and update the docstring, so PGEN gets the memory-derived default and VCF keeps 25k:

```python
    chunk_size: int | None = None,
```

and in the VCF branch pass `chunk_size=chunk_size if chunk_size is not None else 25_000`.

Update the `source` and `chunk_size` docstring entries:

```
    source
        Path to a bgzipped VCF (``.vcf.gz``), BCF (``.bcf``), or PLINK2 PGEN
        (``.pgen``, with its ``.pvar``/``.pvar.zst`` and ``.psam`` siblings).
        VCF/BCF inputs are auto-indexed (``.csi``) if no index is present.
    chunk_size
        Variants per conversion chunk. Defaults to 25000 for VCF/BCF, and to a
        memory-derived value for PGEN.
    ploidy
        Ploidy of the samples. Default 2. VCF/BCF only — PGEN is diploid.
```

- [ ] **Step 3: Update `skills/genoray-api/SKILL.md`**

Read the existing `SparseVar2.from_vcf` section and add a sibling `from_pgen` section in the same style, documenting: the signature, that exactly one of `reference`/`no_reference` is required, the absence of `ploidy`, the return value (dropped out-of-scope ALT count), and the four unsupported things (dosages, INFO/FORMAT fields, sample subsetting, PLINK1 `.bed`). Also update the CLI section to note that `genoray write` accepts `.pgen`.

- [ ] **Step 4: Add the CHANGELOG entry**

Under `## Unreleased` → `### Added` in `CHANGELOG.md`:

```markdown
- `SparseVar2.from_pgen` converts a PLINK2 PGEN to an SVAR2 store, reusing the
  VCF pipeline's normalization, atomization, and merge spine behind a new
  `RecordSource` seam. Genotypes come from the existing `pgenlib` dependency via
  its public Python API (no plink-ng code is linked or vendored; genoray stays
  MIT); variant metadata is streamed from the `.pvar`/`.pvar.zst`. `genoray write`
  dispatches on the `.pgen` suffix. Diploid-only; dosages, INFO/FORMAT field
  extraction, and sample subsetting are not supported.
```

and under `### Fix` (or a new `### Perf` heading, matching whatever the file already uses):

```markdown
- Conversion reader staging memory no longer scales with `chunk_size`: presence
  bits are packed in word-aligned windows and each atom's per-column genotype
  vector is dropped as soon as its bits are set, bounding reader memory at
  `window * n_samples * ploidy * 4` bytes instead of
  `chunk_size * n_samples * ploidy * 4`. Output is bit-identical. Benefits both
  `from_vcf` and `from_pgen`.
```

- [ ] **Step 5: Run the full suite**

Run: `pixi run test`
Expected: PASS.

Run: `pixi run bash -lc 'cargo test --no-default-features && cargo clippy --all-targets -- -D warnings && cargo fmt --check'`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_cli/__main__.py skills/genoray-api/SKILL.md CHANGELOG.md tests/cli
git commit -m "feat(cli): genoray write accepts .pgen; document from_pgen

The write command's help already claimed VCF/PGEN; make it true. Updates the
installable API skill and the changelog per CLAUDE.md's public-name rule."
```

---

## Task 9: Benchmark PGEN vs VCF conversion

The spec's performance claim — "the reader stops being the bottleneck" — is a hypothesis. Measure it; do not assert it.

**Files:**
- Create: `benches/` entry or a `scripts/` benchmark, following whatever the repo already does for the conversion profiling harness (there is prior art: see the `[profile.profiling]` section in `Cargo.toml` and the SVAR1-vs-SVAR2 timing design doc at `docs/superpowers/specs/2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md` — read it first and match its methodology).
- Modify: `CHANGELOG.md` (fill in the measured numbers)

- [ ] **Step 1: Build a cohort-scale fixture**

Take an existing multi-sample VCF (or subset a public one), convert it with
`plink2 --make-pgen`, and confirm both conversions produce equivalent stores
before timing anything. A benchmark of a wrong conversion is worthless.

- [ ] **Step 2: Time both paths**

```bash
pixi run bash -lc '
  /usr/bin/time -v python -c "
from genoray import SparseVar2
SparseVar2.from_vcf(\"/tmp/out_vcf.svar2\", \"<cohort>.vcf.gz\", \"<ref>.fa\", overwrite=True)
" 2>&1 | grep -E "Elapsed|Maximum resident"
  /usr/bin/time -v python -c "
from genoray import SparseVar2
SparseVar2.from_pgen(\"/tmp/out_pgen.svar2\", \"<cohort>.pgen\", \"<ref>.fa\", overwrite=True)
" 2>&1 | grep -E "Elapsed|Maximum resident"
'
```

Record wall time and peak RSS for both.

- [ ] **Step 3: Check the two open performance questions from the spec**

1. **Is the `.pvar` skip cost material?** Each contig's reader skips through the
   `.pvar` from the top, so total `.pvar` I/O is `n_contigs × pvar_size`. Time a
   multi-contig conversion; if the skip is a visible fraction of wall time, the
   mitigation is a one-time contig byte-offset sidecar — note it as a follow-up,
   do not build it speculatively.
2. **Do concurrent contig readers serialize on the GIL, or does pgenlib's OpenMP
   `prange` oversubscribe?** Watch thread counts (`top -H`, `pidstat -t`) during a
   multi-contig run. If readers serialize, the batch byte budget
   (`PGEN_BATCH_BYTES`) is the first knob to turn.

- [ ] **Step 4: Record the results honestly**

Update the CHANGELOG entry from Task 8 with the measured numbers. **If PGEN
conversion is not faster than VCF, say so** — the point of the benchmark is to
find out, not to confirm. File the findings as a follow-up note in the design doc.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md docs/superpowers/specs/2026-07-12-pgen-to-svar2-design.md
git commit -m "perf(svar2): benchmark PGEN vs VCF conversion, record results"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
| --- | --- |
| Licensing position (no linking, pgenlib via Python API only) | Global Constraints; Task 4 module doc |
| `RecordSource` refactor, `RawRecord`, `ChunkAssembler` | Task 2 |
| `PgenRecordSource`, byte-budgeted batch, `-9 → -1` | Task 4 |
| `.pvar` / `.pvar.zst` streaming, Python-computed contig ranges | Tasks 3, 6 |
| Bounded-window packing | Task 1 |
| Auto `chunk_size` for `from_pgen` | Task 6 (`_auto_chunk_size`) |
| Public API `from_pgen`, no `ploidy` | Task 6 |
| CLI dispatch, SKILL.md, CHANGELOG | Task 8 |
| Out-of-scope items documented in the docstring | Task 6 |
| Unphased-het / missing semantics documented | Task 6 |
| Tests: cross-backend equivalence, `.pvar` unit tests, refactor guard | Tasks 2, 3, 6, 7 |
| Benchmark + the two open perf questions | Task 9 |

**Deviation from the spec, called out deliberately:** the spec named "pgenlib itself" as the *primary* test oracle. In the plan, the primary oracle is **cross-backend equivalence** (`from_pgen ≡ from_vcf`), because SVAR2's public read surface returns decoded sequences rather than allele codes, so a direct allele-code comparison would require a test-only API. This is not a weakening: the VCF path is already oracle-tested against `vcfixture`'s decoded `GroundTruth`, so cross-backend equality transitively grounds `from_pgen` in the same ground truth — and it additionally catches errors a raw allele-code check would miss (normalization, left-alignment, atomization). Fixtures are chosen to avoid the places plink2's VCF import is genuinely lossy (half-calls under `--vcf-half-call r`, symbolic ALTs).

**Type consistency:** `RawRecord` fields (`pos`, `reference`, `alts`, `gt`, `info_raw`, `format_raw`) are used identically in Tasks 2 and 4. `ChunkAssembler::new`'s argument list matches its call sites in Tasks 2 and 5. `PvarReader::open(path, var_start)` / `next_variant()` match Task 4's usage. `run_pgen_conversion_pipeline`'s parameter order matches the Python call in Task 6.

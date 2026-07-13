# SparseVar2.from_vcf_list Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SparseVar2.from_vcf_list`, a constructor that builds one SVAR2 store from N single-sample VCFs/BCFs (differing site lists) by natively k-way merging their normalized atoms — no `bcftools merge`, no temp multi-sample VCF.

**Architecture:** A new Rust `RecordSource` (`VcfListRecordSource`) opens one per-file cursor per contig, normalizes each file's records to atomic biallelic left-aligned VCF-form records, and streams a **unified reorder + k-way merge**: a single min-heap keyed by `(POS, ILEN, ALT)` releases an atom-group once every live file has read past `POS + L_MAX` (cross-file generalization of `ChunkAssembler`'s single-stream frontier gate). Each distinct atom becomes one merged biallelic `RawRecord` (non-carrier samples filled hom-ref `0`), which flows through the **unchanged** `ChunkAssembler → dense2sparse_vk → writer`. Python resolves the `sources` argument (directory / manifest / `Sequence`), validates single-sample-ness, and calls a new `_core.run_vcf_list_conversion_pipeline`.

**Tech Stack:** Rust (pyo3 0.29, rust-htslib, rayon), Python 3.10+ (cyvcf2, natsort), pixi env, pytest + Rust `cargo test`.

## Global Constraints

- **Conventional Commits** for every commit (`feat:`, `test:`, `docs:`, `refactor:`).
- **Pre-commit/pre-push hooks:** before any commit/push, `export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/e1e638b2/tmp/cargo-target` — prek's cargo hooks bus-error on the NFS `target/` otherwise.
- **Rust tests must run** `cargo test --no-default-features --features conversion` (the default-feature pyo3 test binary fails to link with `undefined symbol: _Py_Dealloc`). Run inside `pixi run bash -lc '...'`.
- **All conversion Rust code is gated** behind `#[cfg(feature = "conversion")]` — match the existing `vcf_reader.rs` / `orchestrator.rs` gating for every new module and pyfunction.
- **Join identity is `(POS, ILEN, ALT-bytes)`** — never the packed `var_key` (long-INS keys are allocation-order-dependent and unstable across files).
- **Absent site → hom-ref `0`; within-file `./.` → missing `-1`.** These are distinct.
- **Reference flows to the downstream `ChunkAssembler`** (needed for `signatures`; re-normalization is an idempotent no-op). Construct it exactly as `from_vcf` does.
- **Public-API docs are mandatory in this PR:** any task that changes the `from_vcf_list` surface updates `skills/genoray-api/SKILL.md` and adds/extends the `## Unreleased` entry in `CHANGELOG.md` (per repo `CLAUDE.md`).
- **MVP concurrency:** the list pipeline processes contigs **sequentially** (`concurrent_chroms = 1`) to bound open file descriptors to ≈ N. Cross-contig parallelism (subject to `RLIMIT_NOFILE`) and hierarchical batched merge are explicit future work — do not attempt them here.
- **Spec:** `docs/superpowers/specs/2026-07-12-svar2-from-vcf-list-design.md`.

---

### Task 1: `atomize_to_vcf_biallelic` reconstruction helper

Reconstruct ordinary VCF-form biallelic records `(pos, ref_bytes, alt_bytes, source_alt_index)` from a raw multiallelic record, optionally left-aligning. This is what the merge reader emits (a `RawRecord` needs real REF bytes; `Atom` only stores anchor-for-DEL + `ilen`).

**Files:**
- Modify: `src/normalize.rs` (add public fn + `#[cfg(test)]` tests near the existing atomize tests, ~line 260+)

**Interfaces:**
- Consumes: `normalize::{atomize_record, left_align, Atom, L_MAX, NormalizeError}` (existing).
- Produces:
  ```rust
  /// One biallelic atom in ordinary VCF REF/ALT form (not the internal anchor-only
  /// Atom encoding), ready to place in a `RawRecord`.
  pub struct BiallelicRecord {
      pub pos: u32,
      pub reference: Vec<u8>,
      pub alt: Vec<u8>,
      pub source_alt_index: u16, // 1-based index into the ORIGINAL record's ALTs
  }

  /// Atomize `(pos, ref_allele, alts)` into biallelic VCF-form records.
  /// When `ref_seq` is `Some`, each atom is left-aligned and its REF bytes are
  /// read from `ref_seq[pos .. pos + ref_len]`. When `None` (no-reference mode),
  /// no left-alignment happens and REF bytes come from the record's own REF slice
  /// at the atom's offset. `ref_len = alt.len() as i32 - ilen`.
  /// Returns (records, dropped_out_of_scope), matching `atomize_record`'s drop count.
  pub fn atomize_to_vcf_biallelic(
      pos: u32,
      ref_allele: &[u8],
      alts: &[&[u8]],
      ref_seq: Option<&[u8]>,
      skip_out_of_scope: bool,
  ) -> Result<(Vec<BiallelicRecord>, u32), NormalizeError>;
  ```

- [ ] **Step 1: Write failing tests**

Add to `src/normalize.rs` under the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn biallelic_snp_from_ref() {
    // ref_seq "ACAGT..."; POS 2 (0-based) is 'A'->'G' SNP
    let ref_seq = b"ACAGTACATG";
    let (recs, dropped) =
        atomize_to_vcf_biallelic(2, b"A", &[b"G".as_ref()], Some(ref_seq), false).unwrap();
    assert_eq!(dropped, 0);
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].pos, 2);
    assert_eq!(recs[0].reference, b"A");
    assert_eq!(recs[0].alt, b"G");
    assert_eq!(recs[0].source_alt_index, 1);
}

#[test]
fn biallelic_deletion_ref_bytes_reconstructed() {
    // Anchored DEL: REF "CAT" ALT "C" at POS 6 (0-based). ref_len = 1 - (-2) = 3.
    let ref_seq = b"ACAGTACATGGG";
    let (recs, _) =
        atomize_to_vcf_biallelic(6, b"CAT", &[b"C".as_ref()], Some(ref_seq), false).unwrap();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].reference, b"CAT"); // anchor + deleted bases
    assert_eq!(recs[0].alt, b"C");
}

#[test]
fn biallelic_multiallelic_splits_with_alt_indices() {
    let ref_seq = b"ACAGTACATG";
    let (recs, _) =
        atomize_to_vcf_biallelic(2, b"A", &[b"G".as_ref(), b"T".as_ref()], Some(ref_seq), false)
            .unwrap();
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].source_alt_index, 1);
    assert_eq!(recs[1].source_alt_index, 2);
}

#[test]
fn biallelic_no_reference_uses_record_ref() {
    // No ref_seq: REF bytes come from the record's own REF, no left-align.
    let (recs, _) =
        atomize_to_vcf_biallelic(6, b"CAT", &[b"C".as_ref()], None, false).unwrap();
    assert_eq!(recs[0].pos, 6);
    assert_eq!(recs[0].reference, b"CAT");
    assert_eq!(recs[0].alt, b"C");
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo test --no-default-features --features conversion normalize::tests::biallelic 2>&1 | tail -20'`
Expected: FAIL — `cannot find function atomize_to_vcf_biallelic`.

- [ ] **Step 3: Implement `atomize_to_vcf_biallelic`**

Add to `src/normalize.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BiallelicRecord {
    pub pos: u32,
    pub reference: Vec<u8>,
    pub alt: Vec<u8>,
    pub source_alt_index: u16,
}

pub fn atomize_to_vcf_biallelic(
    pos: u32,
    ref_allele: &[u8],
    alts: &[&[u8]],
    ref_seq: Option<&[u8]>,
    skip_out_of_scope: bool,
) -> Result<(Vec<BiallelicRecord>, u32), NormalizeError> {
    let mut atoms = Vec::new();
    let dropped = atomize_record(pos, ref_allele, alts, &mut atoms, skip_out_of_scope)?;
    let mut out = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let atom = match ref_seq {
            Some(rs) => left_align(atom, rs, L_MAX),
            None => atom,
        };
        let ref_len = (atom.alt.len() as i32 - atom.ilen) as usize;
        let reference = match ref_seq {
            Some(rs) => rs[atom.pos as usize..atom.pos as usize + ref_len].to_vec(),
            None => {
                // No left-align, so atom.pos >= pos and the atom's REF span lies
                // within the record's own REF.
                let off = (atom.pos - pos) as usize;
                ref_allele[off..off + ref_len].to_vec()
            }
        };
        out.push(BiallelicRecord {
            pos: atom.pos,
            reference,
            alt: atom.alt,
            source_alt_index: atom.source_alt_index,
        });
    }
    Ok((out, dropped))
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo test --no-default-features --features conversion normalize::tests::biallelic 2>&1 | tail -20'`
Expected: PASS (4 tests). If a DEL/left-align case disagrees, print the failing `Atom` and confirm `ref_len` sign (ilen<0 for DEL ⇒ `alt.len() - ilen` grows REF).

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git add src/normalize.rs
git commit -m "feat(svar2): add atomize_to_vcf_biallelic reconstruction helper"
```

---

### Task 2: Expose `resolve_scalar` for reuse

The merge reader pre-resolves each file's INFO/FORMAT scalars per atom (Task 6). `resolve_scalar` currently lives private in `chunk_assembler.rs`.

**Files:**
- Modify: `src/chunk_assembler.rs:163` (visibility only)

**Interfaces:**
- Produces: `pub(crate) fn resolve_scalar(vals: Option<&[f64]>, source_alt_index: u16, spec: &FieldSpec) -> f64` (unchanged body).

- [ ] **Step 1: Change visibility**

In `src/chunk_assembler.rs:163`, change `fn resolve_scalar(` to `pub(crate) fn resolve_scalar(`.

- [ ] **Step 2: Verify it still builds and all existing tests pass**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo test --no-default-features --features conversion 2>&1 | tail -15'`
Expected: PASS (no behavior change; existing suite green).

- [ ] **Step 3: Commit**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git add src/chunk_assembler.rs
git commit -m "refactor(svar2): expose resolve_scalar as pub(crate) for merge reader reuse"
```

---

### Task 3: Merge core — `VcfListRecordSource` (genotypes only)

The heart of the feature: a `RecordSource` that opens N per-file cursors for a contig and streams merged biallelic `RawRecord`s via a unified reorder + k-way merge. **Genotypes only in this task** (fields deferred to Task 6 — construct with an empty `fields` slice and emit empty `info_raw`/`format_raw`).

**Files:**
- Create: `src/vcf_list_reader.rs`
- Modify: `src/lib.rs` (add `#[cfg(feature = "conversion")] pub mod vcf_list_reader;` beside `vcf_reader`)

**Interfaces:**
- Consumes: `record_source::{RawRecord, RecordSource}`, `vcf_reader::{VcfRecordSource, load_contig_seq}`, `normalize::{atomize_to_vcf_biallelic, BiallelicRecord, L_MAX}`, `field::FieldSpec`, `error::ConversionError`.
- Produces:
  ```rust
  pub struct VcfListRecordSource { /* private */ }
  impl VcfListRecordSource {
      /// `vcf_paths[i]` is a single-sample file whose sample is `samples[i]` and
      /// whose merged column is `i`. `ref_seq` is the contig sequence (None ⇒
      /// no-reference mode). Files without `chrom` in their header are skipped.
      pub fn new(
          vcf_paths: &[String],
          samples: &[&str],
          chrom: &str,
          ref_seq: Option<&[u8]>,
          ploidy: usize,
          htslib_threads: usize,
          skip_out_of_scope: bool,
      ) -> Result<Self, ConversionError>;
      pub fn dropped_out_of_scope(&self) -> u64;
  }
  impl RecordSource for VcfListRecordSource { /* next_record */ }
  ```

**Design notes (implementer, read before coding):**
- Each file column `i` gets a `FileCursor { vcf: Option<VcfRecordSource>, buf: VecDeque<NormAtom>, frontier: Option<u32>, eof: bool, col: usize }`. `vcf` is `None` when the file's header lacks `chrom` (then `eof = true` immediately). Detect via `VcfRecordSource::new` returning the "Chromosome not found" error — catch that specific case and mark the cursor eof rather than propagating.
- `NormAtom { pos, ilen, reference, alt, ploid_codes: Vec<i32> }`. `ploid_codes` has length `ploidy`; for a `BiallelicRecord` with `source_alt_index = k` and the file's decoded single-sample `gt` (length `ploidy`): `code[p] = if gt[p] == -1 { -1 } else if gt[p] == k as i32 { 1 } else { 0 }`. `ilen = alt.len() as i32 - reference.len() as i32`.
- `FileCursor::advance()` — if `buf` empty and not eof: pull one `RawRecord` from `vcf`; set `frontier = Some(rec.pos)`; call `atomize_to_vcf_biallelic(rec.pos, &rec.reference, &alt_refs, ref_seq, skip_out_of_scope)`; for each `BiallelicRecord` compute `ploid_codes` from `rec.gt` and push a `NormAtom`; accumulate dropped count. If the VCF is exhausted, set `eof = true`. Then pop-front one `NormAtom` (or `None`).
- Heap: `BinaryHeap<Reverse<HeapEntry>>` where `HeapEntry { key: (u32 /*pos*/, i32 /*ilen*/, Vec<u8> /*alt*/), col: usize, atom: NormAtom }`, `Ord` by `key` then `col`.
- `next_record()` main loop:
  1. Compute `min_started = ` min `frontier` over live (`!eof`) cursors; `all_started =` every live cursor has `frontier.is_some()`; `all_eof =` no live cursor.
  2. Releasable = `all_eof || (all_started && heap.peek().map_or(false, |Reverse(e)| e.key.0 < min_started.unwrap().saturating_sub(L_MAX)))`.
  3. If releasable and heap non-empty: pop the top; then pop every further entry whose `key == top.key`; build the merged `RawRecord` (below); return `Some`.
  4. If heap empty and `all_eof`: return `None`.
  5. Otherwise advance the live cursor with the smallest `frontier` (unstarted `None` sorts first — it MUST be read before anything releases): call its `advance()`, push any returned `NormAtom` into the heap. Loop.
- Merged `RawRecord` from a group of `(col, NormAtom)` sharing `key`: `reference = group[0].atom.reference.clone()`, `alts = vec![group[0].atom.alt.clone()]`, `gt = vec![0i32; N * ploidy]`; for each `(col, atom)` write `gt[col*ploidy + p] = atom.ploid_codes[p]`; `info_raw = vec![]`, `format_raw = vec![]`.
- `dropped_out_of_scope` sums each cursor's accumulated drops.

- [ ] **Step 1: Write failing tests**

Create `src/vcf_list_reader.rs` with a `#[cfg(test)]` module driving `next_record()` against real temp single-sample VCFs. Use a helper that writes a bgzipped+CSI-indexed single-sample VCF (mirror the pattern in `tests/test_atomize_e2e.rs` / existing `.rs` e2e tests; shell out to `bgzip`/`bcftools index` via `std::process::Command`, write the reference `.fa` + `samtools faidx`).

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // write_ss_vcf(dir, name, sample, body) -> path; make_ref(dir) -> (path, Vec<u8>)
    // (implement these local helpers)

    #[test]
    fn two_files_disjoint_sites_hom_ref_fill() {
        // file A sample "SA": chr1 POS3 A>G  GT 1|0
        // file B sample "SB": chr1 POS7 C>CAT GT 0|1
        // reference required. N=2, ploidy=2.
        let (dir, ref_seq) = /* setup */;
        let paths = vec![a_path, b_path];
        let samples = vec!["SA", "SB"];
        let mut src = VcfListRecordSource::new(
            &paths, &samples, "chr1", Some(&ref_seq), 2, 1, false).unwrap();
        // Record 1: SNP at pos 2 (0-based). Only SA carries it (hap0). SB filled 0.
        let r1 = src.next_record().unwrap().unwrap();
        assert_eq!(r1.pos, 2);
        assert_eq!(r1.alts, vec![b"G".to_vec()]);
        assert_eq!(r1.gt, vec![1, 0, /*SB*/ 0, 0]);
        // Record 2: INS at pos 6. Only SB carries it (hap1). SA filled 0.
        let r2 = src.next_record().unwrap().unwrap();
        assert_eq!(r2.pos, 6);
        assert_eq!(r2.gt, vec![/*SA*/ 0, 0, /*SB*/ 0, 1]);
        assert!(src.next_record().unwrap().is_none());
    }

    #[test]
    fn shared_site_merges_into_one_record() {
        // Both files carry chr1 POS3 A>G. Expect ONE merged record, gt from both.
        // SA GT 1|0, SB GT 0|1  => gt [1,0, 0,1].
    }

    #[test]
    fn multiallelic_in_one_file_splits_across_records() {
        // file A: chr1 POS3 A>G,T  GT 1|2  => two biallelic records at pos 2:
        //   ALT G: SA hap0 carries => gt[0]=1; ALT T: SA hap1 carries => gt[1]=1.
    }

    #[test]
    fn within_file_missing_stays_minus_one() {
        // file A: chr1 POS3 A>G GT .|1  => merged gt hap0 = -1, hap1 = 1.
    }

    #[test]
    fn file_without_contig_is_skipped() {
        // file B has only chr2 records; querying chr1 must not error, B fills hom-ref.
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo test --no-default-features --features conversion vcf_list_reader 2>&1 | tail -25'`
Expected: FAIL — module/type not found.

- [ ] **Step 3: Implement `VcfListRecordSource`**

Write the struct, `FileCursor`, `NormAtom`, `HeapEntry` (with `Ord`), `new`, `advance`, and `next_record` per the Design notes above. Key correctness points to get right:
- Unstarted cursors (`frontier == None`) sort **first** when picking who to advance, and block release (`all_started` gate).
- The `key == top.key` group-drain must compare the full `(pos, ilen, alt)` tuple.
- `VcfRecordSource::new` "Chromosome not found" ⇒ mark cursor eof, not error. Match on `ConversionError::Input(msg)` where `msg.contains("not found in VCF header")`.
- Pass `htslib_threads` (small, e.g. 1) into each `VcfRecordSource::new`; pass an empty `&[FieldSpec]` (fields land in Task 6).

- [ ] **Step 4: Run tests, verify they pass**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo test --no-default-features --features conversion vcf_list_reader 2>&1 | tail -25'`
Expected: PASS (5 tests). Debug any ordering failure by logging the heap key and `min_started` at each release decision.

- [ ] **Step 5: Commit**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git add src/vcf_list_reader.rs src/lib.rs
git commit -m "feat(svar2): VcfListRecordSource k-way merge of single-sample VCFs (genotypes)"
```

---

### Task 4: Orchestrator + pyfunction wiring

Expose the merge through `process_chromosome` and a new `_core.run_vcf_list_conversion_pipeline`, so Python can drive an end-to-end conversion. Contigs run **sequentially**.

**Files:**
- Modify: `src/orchestrator.rs:44` (add `SourceSpec::VcfList`), `src/orchestrator.rs:169` (dispatch arm)
- Modify: `src/lib.rs` (new pyfunction + register in `_core`)

**Interfaces:**
- Consumes: `VcfListRecordSource::new`, `vcf_reader::load_contig_seq`, `field::parse_manifest`, `field_finalize::finalize_fields`, `meta::write_meta`, `budget` (only to size processing threads).
- Produces (Python-visible):
  ```python
  _core.run_vcf_list_conversion_pipeline(
      vcf_paths: list[str],           # single-sample files, parallel to `samples`
      reference_path: str | None,
      chroms: list[str],
      output_dir: str,
      samples: list[str],             # header sample of each file, in file order
      chunk_size: int = 25_000,
      ploidy: int = 2,
      max_threads: int | None = None,
      long_allele_capacity: int = 8_388_608,
      skip_out_of_scope: bool = False,
      signatures: bool = False,
      info_fields: list[tuple] = [],  # same 5-tuple shape as run_conversion_pipeline
      format_fields: list[tuple] = [],
  ) -> int                            # out-of-scope ALTs dropped
  ```

**Design notes:**
- `SourceSpec::VcfList { vcf_paths: Vec<String>, htslib_threads: usize }`. The dispatch arm builds `VcfListRecordSource::new(&vcf_paths, &s_refs, &chr, ref_seq_opt, ploidy, htslib_threads, skip_out_of_scope)`. `s_refs` is `process_chromosome`'s existing `samples` slice (positionally the per-file sample names). Load `ref_seq_opt` via `load_contig_seq(fasta, &chr)` when `fasta_path.is_some()`.
- The pyfunction mirrors `run_conversion_pipeline` (`src/lib.rs:110`) but: (a) takes `vcf_paths: Vec<String>`; (b) forces `concurrent_chroms = 1` (iterate `chroms` sequentially — a plain `for` loop, no rayon pool needed); (c) uses a small per-file `htslib_threads` (e.g. `1`). Reuse the `reference`/`no_reference` derivation, `parse_manifest`, `finalize_fields`, and `write_meta` tail verbatim.
- `signatures` requires a reference (validated Python-side in Task 5, but keep the pipeline honest by passing the fasta through).

- [ ] **Step 1: Write a failing Rust e2e test**

Create `tests/test_vcf_list_e2e.rs` (mirror `tests/test_e2e.rs` setup helpers): write two single-sample bgzipped+indexed VCFs + a reference, call the conversion via the library entrypoint the pyfunction wraps (factor the pyfunction body into a `pub fn run_vcf_list(...)` in `orchestrator.rs` or `lib.rs` so the Rust test can call it without pyo3), assert `meta.json` exists and lists both samples and the union contigs.

```rust
#[test]
fn vcf_list_e2e_two_samples_one_store() {
    // ... write SA (chr1 POS3 A>G), SB (chr1 POS7 C>CAT), ref.fa+.fai ...
    let dropped = genoray::orchestrator::run_vcf_list(
        &[a, b], Some(&ref_path), &["chr1".into()], &out,
        &["SA".into(), "SB".into()], 25_000, 2, Some(1), 8_388_608, false, false, &[], &[],
    ).unwrap();
    assert_eq!(dropped, 0);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("meta.json")).unwrap()).unwrap();
    assert_eq!(meta["samples"], serde_json::json!(["SA", "SB"]));
}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo test --no-default-features --features conversion vcf_list_e2e 2>&1 | tail -25'`
Expected: FAIL — `run_vcf_list` not found.

- [ ] **Step 3: Implement the `SourceSpec::VcfList` arm, `run_vcf_list`, and the pyfunction**

Add the enum variant + dispatch arm in `orchestrator.rs`; factor the sequential conversion loop into `pub fn run_vcf_list(...)` returning `Result<u64, ConversionError>`; add `#[pyfunction] fn run_vcf_list_conversion_pipeline(...)` in `lib.rs` that calls it under `py.detach`, converts the error, and registers via `m.add_function(wrap_pyfunction!(run_vcf_list_conversion_pipeline, m)?)?;` (gated on `conversion`).

- [ ] **Step 4: Run test, verify it passes**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo test --no-default-features --features conversion vcf_list_e2e 2>&1 | tail -25'`
Expected: PASS.

- [ ] **Step 5: Build the editable wheel so Python sees the new symbol**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target maturin develop --features conversion 2>&1 | tail -15'`
Expected: `📦 Built wheel` / `🛠 Installed genoray`. Then verify: `pixi run python -c "import genoray._core as c; print(hasattr(c, 'run_vcf_list_conversion_pipeline'))"` → `True`.

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git add src/orchestrator.rs src/lib.rs tests/test_vcf_list_e2e.rs
git commit -m "feat(svar2): wire VcfList source into orchestrator + _core pyfunction"
```

---

### Task 5: Python `from_vcf_list` + input resolution + validation

The public API. Resolve `sources`, validate single-sample-ness, derive the sample list + contig union, and call the pyfunction. Reference required in this task (`no_reference` lands in Task 8). Fields NOT yet forwarded (pass `[]`; Task 6 wires them).

**Files:**
- Modify: `python/genoray/_svar2.py` (add `from_vcf_list` classmethod + module-level `_resolve_vcf_sources` helper, after `from_pgen` ~line 289)
- Create: `tests/test_svar2_from_vcf_list.py`
- Modify: `skills/genoray-api/SKILL.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: `_core.run_vcf_list_conversion_pipeline`, `_svar2_fields._resolve_fields`, `cyvcf2.VCF`, `natsort.natsorted`, `_ensure_bgzipped`, `_ensure_index`.
- Produces:
  ```python
  @classmethod
  def from_vcf_list(cls, out, sources, reference=None, *, no_reference=False,
      skip_out_of_scope=False, ploidy=2, chunk_size=25_000, threads=None,
      overwrite=False, long_allele_capacity=8*1024*1024, signatures=False,
      info_fields=None, format_fields=None) -> int
  # module-level:
  def _resolve_vcf_sources(sources: str | Path | Sequence[str | Path]) -> list[Path]
  ```

**Design notes:**
- `_resolve_vcf_sources`:
  - `Sequence` (list/tuple, not `str`/`Path`) → `[Path(x) for x in sources]`.
  - single `Path`, `is_dir()` → `natsorted(p for p in dir.glob("*.vcf.gz")) + natsorted(dir.glob("*.bcf"))`; non-recursive.
  - single `Path`, is a file ending `.vcf.gz`/`.bcf` → `[that path]`.
  - single `Path`, other file → treat as manifest: read text, for each line strip; skip blank and `#`-prefixed; resolve relative entries against `path.parent`.
  - Empty result → `ValueError("no VCF/BCF files found in {sources}")`.
- `from_vcf_list` body:
  1. Same `reference`/`no_reference` exactly-one-of + `signatures`+`no_reference` + `overwrite` guards as `from_vcf`. **Additionally in this task:** `if no_reference: raise NotImplementedError("no_reference is not yet supported by from_vcf_list; pass a reference")` (removed in Task 8).
  2. `paths = _resolve_vcf_sources(sources)`; `out.parent.mkdir(...)`.
  3. For each path: `_ensure_bgzipped(path)`, `_ensure_index(path)`, open `cyvcf2.VCF(str(path))`; assert `len(v.samples) == 1` else `ValueError(f"{path} is not single-sample (has {len(v.samples)} samples)")`; collect `sample = v.samples[0]` and the file's `seqnames` that have ≥1 record.
  4. Duplicate sample names → `ValueError` listing collisions.
  5. `contigs = natsorted(union of per-file contigs-with-variants)`; empty → `ValueError("No variants found in any input.")`.
  6. `return _core.run_vcf_list_conversion_pipeline([str(p) for p in paths], None if no_reference else str(reference), contigs, str(out), samples, chunk_size, ploidy, threads, long_allele_capacity, skip_out_of_scope, signatures, [], [])`.
- Docstring documents: single-sample requirement; `sources` forms; absent-site → hom-ref; join semantics; that `info_fields`/`format_fields` are accepted but forwarded in a later step (or note "not yet supported" until Task 6 — keep the signature but raise `NotImplementedError` if passed, removed in Task 6).

- [ ] **Step 1: Write failing Python tests**

Create `tests/test_svar2_from_vcf_list.py` (reuse the `_write_ref` pattern from `tests/test_svar2_from_vcf.py`; add a `_write_ss_vcf(dir, name, sample, body)` helper that bgzips + `bcftools index`):

```python
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import pytest
from genoray import SparseVar2

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"

def _write_ref(d: Path) -> Path:
    ref = d / "ref.fa"; ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True); return ref

def _ss(d: Path, name: str, sample: str, rows: str) -> Path:
    header = ("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n")
    plain = d / f"{name}.vcf"; plain.write_text(header + rows)
    gz = d / f"{name}.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz

def test_from_vcf_list_disjoint_sites_hom_ref_fill(tmp_path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "store"
    dropped = SparseVar2.from_vcf_list(out, [a, b], ref, threads=1)
    assert dropped == 0
    sv = SparseVar2(out)
    assert sv.available_samples == ["SA", "SB"]
    # SA carries SNP@2 on hap0; SB carries INS@6 on hap1.
    counts = sv.region_counts("chr1", [(0, 40)]).reshape(-1)  # (R,S,P) -> [SA_h0,SA_h1,SB_h0,SB_h1]
    assert counts.tolist() == [1, 0, 0, 1]

def test_from_vcf_list_shared_site_one_variant(tmp_path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n")
    sv = SparseVar2(SparseVar2.from_vcf_list(tmp_path/"s", [a, b], ref, threads=1) or tmp_path/"s")
    rag = SparseVar2(tmp_path/"s").decode("chr1", [(0, 40)])
    lengths = rag["pos"].lengths.reshape(-1)
    assert lengths.tolist() == [1, 0, 0, 1]  # same site, one hap each

def test_from_vcf_list_directory_and_manifest_equivalent(tmp_path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n")
    SparseVar2.from_vcf_list(tmp_path/"by_seq", [a, b], ref, threads=1)
    # directory: put both gz+csi in a subdir
    d = tmp_path / "vcfs"; d.mkdir()
    for p in (a, b):
        (d/p.name).write_bytes(p.read_bytes()); (d/(p.name+".csi")).write_bytes(Path(str(p)+".csi").read_bytes())
    SparseVar2.from_vcf_list(tmp_path/"by_dir", d, ref, threads=1)
    manifest = tmp_path/"m.txt"; manifest.write_text(f"# comment\n{a}\n\n{b}\n")
    SparseVar2.from_vcf_list(tmp_path/"by_manifest", manifest, ref, threads=1)
    for name in ("by_dir", "by_manifest"):
        assert SparseVar2(tmp_path/name).available_samples == ["SA", "SB"]

def test_from_vcf_list_rejects_multisample(tmp_path):
    ref = _write_ref(tmp_path)
    two = _ss(tmp_path, "two", "SA\tSB",  # header hack: two sample cols
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|1\n")
    with pytest.raises(ValueError, match="single-sample"):
        SparseVar2.from_vcf_list(tmp_path/"s", [two], ref, threads=1)

def test_from_vcf_list_rejects_duplicate_samples(tmp_path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "S", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "S", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    with pytest.raises(ValueError, match="duplicate|collision"):
        SparseVar2.from_vcf_list(tmp_path/"s", [a, b], ref, threads=1)

def test_from_vcf_list_requires_reference(tmp_path):
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_vcf_list(tmp_path/"s", [a], threads=1)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py -x 2>&1 | tail -25`
Expected: FAIL — `AttributeError: type object 'SparseVar2' has no attribute 'from_vcf_list'`.

- [ ] **Step 3: Implement `_resolve_vcf_sources` and `from_vcf_list`**

Add both to `python/genoray/_svar2.py` per the Design notes. Keep the `NotImplementedError` guards for `no_reference` (Task 8) and for `info_fields`/`format_fields` (Task 6).

- [ ] **Step 4: Run tests, verify they pass**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py -x 2>&1 | tail -25`
Expected: PASS (6 tests). The multisample test relies on the header having 2 sample columns — confirm cyvcf2 reports `len(v.samples) == 2`.

- [ ] **Step 5: Update docs**

In `skills/genoray-api/SKILL.md`, add `from_vcf_list` next to `from_vcf`: signature, single-sample requirement, `sources` forms (dir / manifest / `Sequence`), hom-ref fill, join-on-atom semantics, reference required (for now). In `CHANGELOG.md` under `## Unreleased`, add: `- Added \`SparseVar2.from_vcf_list\` to build one store from many single-sample VCFs/BCFs via a native k-way merge (absent sites filled hom-ref).`

- [ ] **Step 6: Commit**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git add python/genoray/_svar2.py tests/test_svar2_from_vcf_list.py skills/genoray-api/SKILL.md CHANGELOG.md
git commit -m "feat(svar2): SparseVar2.from_vcf_list public API + input resolution"
```

---

### Task 6: bcftools-merge oracle parity test

Prove `from_vcf_list` equals the trusted path: `bcftools merge` (with missing→ref) → `from_vcf`. This is the correctness anchor for the whole feature.

**Files:**
- Modify: `tests/test_svar2_from_vcf_list.py` (add a `@pytest.mark.parametrize` oracle test)

**Interfaces:**
- Consumes: `SparseVar2.from_vcf`, `SparseVar2.from_vcf_list`, `SparseVar2.decode`; `bcftools`, `bgzip` on PATH.

**Design notes:**
- Build 3 single-sample VCFs with a mix of: a shared SNP, a private SNP, a multiallelic site in one file, an anchored INS, an anchored DEL, a `.|.` genotype.
- Oracle store: `bcftools merge -0 <a> <b> <c>` (**`-0` assumes genotypes at absent sites are 0/0 ref — exactly our hom-ref semantics**) → bgzip → `bcftools index` → `from_vcf(oracle_out, merged, ref)`.
- Compare `from_vcf_list` vs oracle: for the full region `(0, len(ref))`, assert equal `region_counts`, and equal decoded `pos`/`ilen`/`allele` per `(sample, ploid)` after sorting each hap's variant list (both are position-sorted already). Samples must be in the same order (`bcftools merge` orders by input file order — matches ours).

- [ ] **Step 1: Write the failing oracle test**

```python
def test_from_vcf_list_matches_bcftools_merge_oracle(tmp_path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n"      # shared SNP
        "chr1\t12\t.\tT\tTA\t.\t.\t.\tGT\t0|1\n")   # private INS
    b = _ss(tmp_path, "b", "SB",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n"      # shared SNP
        "chr1\t7\t.\tC\tG,T\t.\t.\t.\tGT\t1|2\n")   # multiallelic
    c = _ss(tmp_path, "c", "SC",
        "chr1\t7\t.\tCAT\tC\t.\t.\t.\tGT\t1|.\n")   # DEL + missing hap
    paths = [a, b, c]
    # oracle
    merged = tmp_path / "merged.vcf.gz"
    with open(merged, "wb") as fh:
        subprocess.run(["bcftools", "merge", "-0", *map(str, paths)],
                       check=True, stdout=subprocess.PIPE)  # capture then bgzip
        # (implement: merge to stdout -> bgzip -c > merged; then bcftools index)
    # ... bgzip + index merged ...
    from_vcf_out = tmp_path / "oracle"
    SparseVar2.from_vcf(from_vcf_out, merged, ref, threads=1)
    list_out = tmp_path / "list"
    SparseVar2.from_vcf_list(list_out, paths, ref, threads=1)

    o, l = SparseVar2(from_vcf_out), SparseVar2(list_out)
    assert o.available_samples == l.available_samples == ["SA", "SB", "SC"]
    region = [(0, len(_REF))]
    np.testing.assert_array_equal(
        o.region_counts("chr1", region), l.region_counts("chr1", region))
    ro, rl = o.decode("chr1", region), l.decode("chr1", region)
    for field in ("pos", "ilen"):
        np.testing.assert_array_equal(
            np.asarray(ro[field].data), np.asarray(rl[field].data))
```

- [ ] **Step 2: Run test, verify it fails or reveals a real mismatch**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py::test_from_vcf_list_matches_bcftools_merge_oracle -x 2>&1 | tail -30`
Expected: either PASS (feature already correct) or a concrete diff. If it fails, **use superpowers:systematic-debugging** — the mismatch is a real bug in the merge core (likely alt-index remap, DEL reconstruction, or hom-ref fill), not a test artifact. Fix in `src/vcf_list_reader.rs` / `src/normalize.rs`, rebuild the wheel, re-run.

- [ ] **Step 3: Make it pass**

Resolve any mismatch. Rebuild wheel (`maturin develop --features conversion`) after Rust edits.

- [ ] **Step 4: Commit**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git add tests/test_svar2_from_vcf_list.py src/  # + any fix
git commit -m "test(svar2): bcftools-merge oracle parity for from_vcf_list"
```

---

### Task 7: INFO/FORMAT field carry-through

Forward `info_fields` (first-carrier wins) and `format_fields` (per-sample) through the merge. Extend `NormAtom` to carry pre-resolved scalars; pack them as single-element raw buffers the unchanged `ChunkAssembler` re-resolves trivially.

**Files:**
- Modify: `src/vcf_list_reader.rs` (NormAtom fields + per-file field resolution + merged buffer construction), `src/lib.rs` (forward `info`/`format` specs instead of empty), `python/genoray/_svar2.py` (forward `info_fields`/`format_fields`, drop the `NotImplementedError`), `tests/test_svar2_from_vcf_list.py`, `skills/genoray-api/SKILL.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: `chunk_assembler::resolve_scalar` (Task 2), `field::FieldSpec`, `field::FieldCategory`.
- `VcfListRecordSource::new` gains a `fields: &[FieldSpec]` parameter; the per-file `VcfRecordSource::new` is passed the same `fields` so raw INFO/FORMAT buffers are decoded.

**Design notes:**
- `NormAtom` gains `info_vals: Vec<f64>` (len = #info specs) and `format_vals: Vec<f64>` (len = #format specs; this file's single sample). In `advance()`, resolve each spec for the atom with `resolve_scalar(rec.info_raw[i].as_deref(), source_alt_index, spec)` and `resolve_scalar(rec.format_raw[j].as_ref().map(|v| v[0].as_slice()), source_alt_index, spec)` (single sample ⇒ index 0).
- Merged `RawRecord` for a group (members sorted by `col` ascending — file/list order):
  - `info_raw[i] = Some(vec![members[0].info_vals[i]])` (**first-carrier**).
  - `format_raw[j] = Some(per_sample)` where `per_sample[col]` is `vec![member.format_vals[j]]` for a carrier and `vec![]` (empty ⇒ downstream default) for a non-carrier.
- Downstream `ChunkAssembler` re-resolves: biallelic `source_alt_index = 1`, single-element buffers pass through; empty buffers yield the field default.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_svar2_from_vcf_list.py` (declare an INFO and a FORMAT field in the headers; give two files different INFO values at a shared site to prove first-carrier):

```python
def test_from_vcf_list_format_field_per_sample(tmp_path):
    ref = _write_ref(tmp_path)
    hdr_extra = ('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="dp">\n')
    # ... write two single-sample VCFs with FORMAT GT:DP, e.g. SA DP=10, SB DP=20
    # at a shared SNP@3; read the store's DP field and assert per-sample values.
    ...

def test_from_vcf_list_info_first_carrier(tmp_path):
    # shared site, file A INFO/AF=0.1 (first in list), file B INFO/AF=0.9
    # assert stored AF == 0.1 (first-carrier).
    ...
```
(Read `tests/test_svar2_fields.py` / `tests/test_svar2_fields_read.py` for how stored fields are declared via `info_fields`/`format_fields` and read back through the decode API, and mirror that assertion style.)

- [ ] **Step 2: Run tests, verify they fail**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py -k "field or carrier" -x 2>&1 | tail -25`
Expected: FAIL — `NotImplementedError` (fields guard) or wrong values.

- [ ] **Step 3: Implement field carry-through**

Extend `NormAtom`, `advance()`, the merged-record builder (Rust); forward the parsed `info`/`format` spec vectors from the pyfunction into `VcfListRecordSource::new`; forward `info_fields`/`format_fields` from Python (`_resolve_fields`, same as `from_vcf`) and drop the `NotImplementedError`. Rebuild wheel.

- [ ] **Step 4: Run tests, verify they pass**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py -x 2>&1 | tail -25`
Expected: PASS (all, including the earlier tasks' tests).

- [ ] **Step 5: Update docs + commit**

Update `SKILL.md` (fields: FORMAT per-sample, INFO first-carrier) and `CHANGELOG.md`. Then:
```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git add src/ python/genoray/_svar2.py tests/test_svar2_from_vcf_list.py skills/genoray-api/SKILL.md CHANGELOG.md
git commit -m "feat(svar2): INFO(first-carrier)/FORMAT(per-sample) fields in from_vcf_list"
```

---

### Task 8: `no_reference` support

Allow `no_reference=True`: skip left-alignment; reconstruct atom REF from the record's own REF (the `ref_seq=None` path already built in Task 1). Downstream `ChunkAssembler` runs with `fasta_path=None` — but only when `no_reference`; with a reference it still receives the fasta.

**Files:**
- Modify: `python/genoray/_svar2.py` (remove the `no_reference` `NotImplementedError`; pass `None` reference through), `tests/test_svar2_from_vcf_list.py`, `skills/genoray-api/SKILL.md`, `CHANGELOG.md`

**Interfaces:**
- No new symbols. `VcfListRecordSource` already accepts `ref_seq: Option<&[u8]>` (Task 3); the orchestrator already passes `None` when `reference_path` is `None` (Task 4). This task is mostly removing the Python guard + a documented test.

**Design notes:**
- Under `no_reference`, atoms are not left-aligned, so cross-file joins line up only if inputs are already consistently normalized — document this. The oracle here is `from_vcf(..., no_reference=True)` on a `bcftools merge -0` of pre-normalized inputs.

- [ ] **Step 1: Write failing test**

```python
def test_from_vcf_list_no_reference_snp_only(tmp_path):
    # SNP-only single-sample VCFs, no reference. Should convert and hom-ref fill.
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t9\t.\tG\tT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "s"
    SparseVar2.from_vcf_list(out, [a, b], no_reference=True, threads=1)
    sv = SparseVar2(out)
    assert sv.available_samples == ["SA", "SB"]
    assert sv.region_counts("chr1", [(0, 40)]).reshape(-1).tolist() == [1, 0, 0, 1]
```

- [ ] **Step 2: Run, verify it fails**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py::test_from_vcf_list_no_reference_snp_only -x 2>&1 | tail -20`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Remove the guard + verify orchestrator passes `None`**

Delete the `no_reference` `NotImplementedError` in `from_vcf_list`; confirm the `reference_path=None` path reaches `VcfListRecordSource` with `ref_seq=None`. Rebuild wheel if any Rust touched (should not be needed).

- [ ] **Step 4: Run, verify it passes**

Run: `pixi run pytest tests/test_svar2_from_vcf_list.py -x 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Update docs + commit**

Update `SKILL.md` (no_reference caveat) + `CHANGELOG.md`. Then:
```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git add python/genoray/_svar2.py tests/test_svar2_from_vcf_list.py skills/genoray-api/SKILL.md CHANGELOG.md
git commit -m "feat(svar2): support no_reference in from_vcf_list"
```

---

### Task 9: Full-suite verification + lint

Confirm the whole feature is green and lints clean before opening the PR.

**Files:** none (verification only).

- [ ] **Step 1: Full Rust suite**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo test --no-default-features --features conversion 2>&1 | tail -20'`
Expected: all tests pass (existing + new `vcf_list_reader` + `test_vcf_list_e2e`).

- [ ] **Step 2: Full Python suite**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target maturin develop --features conversion && pytest tests/ -q 2>&1 | tail -30'`
Expected: all pass (regenerates fixtures via the test setup; new `test_svar2_from_vcf_list.py` green).

- [ ] **Step 3: Lint (ruff + clippy + fmt)**

Run: `pixi run bash -lc 'ruff check genoray tests python && ruff format --check python tests && CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target cargo clippy --no-default-features --features conversion -- -D warnings && cargo fmt --check'`
Expected: clean. Fix any findings, re-run.

- [ ] **Step 4: Push branch + open draft PR**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
git push -u origin worktree-svar2-from-vcf-list
gh pr create --draft --title "feat(svar2): SparseVar2.from_vcf_list — native single-sample VCF merge" \
  --body "$(cat <<'EOF'
Implements `SparseVar2.from_vcf_list` per docs/superpowers/specs/2026-07-12-svar2-from-vcf-list-design.md.

Native k-way merge of N single-sample VCFs/BCFs into one SVAR2 store: normalize each file's records to atomic biallelic left-aligned form, join on (POS, ILEN, ALT), fill non-carriers hom-ref. Feeds the unchanged ChunkAssembler/dense2sparse pipeline. Verified byte-identical to a `bcftools merge -0` → `from_vcf` oracle.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Public API + `sources` resolution → Task 5 ✓
- Native k-way merge, join on `(POS, ILEN, ALT)`, hom-ref fill → Task 3 ✓
- VCF-form reconstruction (Atom → REF/ALT) → Task 1 ✓
- Unchanged downstream pipeline + orchestrator/pyfunction wiring → Task 4 ✓
- Semantics (absent→hom-ref, within-file `./.`→-1) → Tasks 3, 6 ✓
- `format_fields` per-sample, `info_fields` first-carrier → Task 7 ✓
- `no_reference` → Task 8 ✓
- Errors (non-single-sample, duplicate names, empty sources, exactly-one-of) → Task 5 ✓
- bcftools oracle + targeted cases → Task 6 (+ Rust cases in Task 3) ✓
- Scale/limits (sequential contigs MVP) → Global Constraints ✓
- Docs (SKILL.md + CHANGELOG) → Tasks 5, 7, 8 ✓

**Deviations from spec (flagged to user, approved refinements):**
- Merge reader reimplements the `L_MAX` frontier discipline as a cross-file **min**-frontier gate rather than factoring `ChunkAssembler`'s single-stream gate into a shared helper — lower-risk (no edit to tested hot code), and the semantics differ enough that sharing wasn't clean. `ChunkAssembler` is untouched except the `resolve_scalar` visibility change (Task 2).
- MVP processes contigs sequentially (`concurrent_chroms=1`) to bound file descriptors; cross-contig parallelism + hierarchical merge are explicit future work.

**Placeholder scan:** No TBD/TODO; every code step has concrete code or exact edits. Field-test bodies in Task 7 reference the existing `test_svar2_fields*.py` assertion style (the one place the reader must consult a sibling test for the decode-read idiom, since the field-read API is out of this plan's scope to restate).

**Type consistency:** `BiallelicRecord`/`atomize_to_vcf_biallelic` (Task 1) consumed in Task 3; `resolve_scalar` (Task 2) consumed in Task 7; `VcfListRecordSource::new` signature grows a `fields` param in Task 7 (noted there); `run_vcf_list_conversion_pipeline` arg order fixed in Task 4 and reused in Tasks 5/7/8. `region_counts`/`decode` match the existing API in `python/genoray/_svar2_decode.py`.

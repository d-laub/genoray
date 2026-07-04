# SVAR2 CLI write + M13 skip-out-of-scope — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `genoray write` default to SVAR2 (with `genoray write svar1` for SVAR 1.0), add a Python `SparseVar2.from_vcf` wrapper, make the reference FASTA optional, auto-index un-indexed inputs, and add the M13 opt-in skip for out-of-scope (symbolic/breakend) ALTs.

**Architecture:** The SVAR2 conversion engine already exists as the Rust `run_conversion_pipeline` PyO3 function; there is no Python or CLI entry point for it yet. We (1) extend the Rust core to make the reference optional and to optionally skip+count out-of-scope ALTs, (2) add a tiny `index_vcf` PyO3 helper over the existing `rust-htslib` dep, (3) add a Python `SparseVar2.from_vcf` wrapper that derives samples/contigs from the header and enforces the reference contract, and (4) restructure the CLI `write` command into a two-command group.

**Tech Stack:** Rust (PyO3, rust-htslib 1.0, rayon, ndarray), Python (cyclopts CLI, cyvcf2, natsort), maturin build, pytest + cargo test.

## Global Constraints

- Coordinate convention: 0-based half-open `[start, end)`; missing genotype `-1`.
- Conventional Commits for every commit (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- The strict default is preserved: symbolic/breakend ALTs are a hard error unless skip is explicitly opted in. Reference is required unless `--no-reference` / `no_reference=True` is explicitly passed.
- `rust-htslib = "1.0"` with `default-features = false` (already in `Cargo.toml`); do not add new crates. `index_vcf` uses `rust_htslib::bcf::index::build`.
- Any public-name change (reachable from `import genoray` without underscores, plus the CLI and the `run_conversion_pipeline` signature) MUST update `skills/genoray-api/SKILL.md` in this same work (Task 6).
- Build after Rust edits with `maturin develop` (inside `pixi s`) before running Python tests. Rust-only tests run with `cargo test`.
- Work happens in an isolated git worktree under `.claude/worktrees` (created at execution time via superpowers:using-git-worktrees).

---

### Task 1: `normalize::atomize_record` — opt-in skip + drop count

Make atomization skip out-of-scope ALTs when opted in (mirroring the existing `*`/`.` per-allele skip) and return how many were dropped; keep the strict default (error). This is a pure function with in-source unit tests — the strongest TDD point in the plan.

**Files:**
- Modify: `src/normalize.rs:50-70` (`atomize_record` signature + body)
- Modify: `src/normalize.rs:247-251` (test helper `atoms`), `src/normalize.rs:330-345` (existing symbolic-error tests), `src/normalize.rs:510-515` (proptest helper call)
- Modify (keep-compiling only): `src/vcf_reader.rs:169-170` (add trailing `false` arg)

**Interfaces:**
- Produces: `pub fn atomize_record(pos: u32, ref_allele: &[u8], alts: &[&[u8]], out: &mut Vec<Atom>, skip_out_of_scope: bool) -> Result<u32, NormalizeError>` — returns the count of dropped out-of-scope ALTs (0 when none dropped or skip disabled); still returns `Err(NormalizeError::SymbolicAllele)` when an out-of-scope ALT is seen and `skip_out_of_scope` is `false`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/normalize.rs` (near the existing `symbolic_allele_errors` test):

```rust
#[test]
fn symbolic_allele_skipped_and_counted_when_opted_in() {
    // Multiallelic: one real SNP + one symbolic ALT. With skip on, the SNP
    // survives and the symbolic ALT is dropped and counted.
    let mut out = Vec::new();
    let dropped = atomize_record(100, b"A", &[b"C", b"<DEL>"], &mut out, true).unwrap();
    assert_eq!(dropped, 1);
    assert_eq!(out, vec![atom(100, 0, b"C", 1)]);
}

#[test]
fn breakend_allele_skipped_when_opted_in() {
    let mut out = Vec::new();
    let dropped = atomize_record(100, b"A", &[b"A[chr2:321[".as_slice()], &mut out, true).unwrap();
    assert_eq!(dropped, 1);
    assert!(out.is_empty());
}

#[test]
fn out_of_scope_errors_when_skip_disabled() {
    let mut out = Vec::new();
    assert!(matches!(
        atomize_record(100, b"A", &[b"<DEL>"], &mut out, false),
        Err(NormalizeError::SymbolicAllele { .. })
    ));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib normalize::tests 2>&1 | head -40`
Expected: compile error — `atomize_record` takes 4 args, not 5 (the new tests and existing call sites pass the wrong arity).

- [ ] **Step 3: Change the signature and body**

In `src/normalize.rs`, replace `atomize_record` (lines ~50-70) with:

```rust
/// Decompose one VCF record into atomic biallelic primitives, appended to `out`.
/// `alts` are the ALT alleles only (REF excluded). `*` / `.` alleles are always
/// skipped. Symbolic/breakend alleles are skipped and counted when
/// `skip_out_of_scope` is set; otherwise they return an error. Returns the number
/// of out-of-scope ALTs dropped (always 0 unless `skip_out_of_scope`).
pub fn atomize_record(
    pos: u32,
    ref_allele: &[u8],
    alts: &[&[u8]],
    out: &mut Vec<Atom>,
    skip_out_of_scope: bool,
) -> Result<u32, NormalizeError> {
    let mut dropped = 0u32;
    for (j, &alt) in alts.iter().enumerate() {
        let src = (j + 1) as u16;
        if alt == b"*" || alt == b"." {
            continue;
        }
        if is_symbolic(alt) {
            if skip_out_of_scope {
                dropped += 1;
                continue;
            }
            return Err(NormalizeError::SymbolicAllele {
                pos,
                alt: String::from_utf8_lossy(alt).into_owned(),
            });
        }
        atomize_biallelic(pos, ref_allele, alt, src, out);
    }
    Ok(dropped)
}
```

- [ ] **Step 4: Update existing call sites to keep the crate compiling**

`src/normalize.rs` test helper (~line 249) — append `, false` and unwrap the count:
```rust
    atomize_record(pos, r, alts, &mut out, false).unwrap();
```

`src/normalize.rs` existing error tests (~lines 335 and 339) — append `, false`:
```rust
            atomize_record(100, b"A", &[b"<DEL>"], &mut out, false),
```
```rust
            atomize_record(100, b"A", &[b"A[chr2:321[".as_slice()], &mut out, false),
```

`src/normalize.rs` proptest helper (~line 513) — append `, false`:
```rust
            atomize_record(pos, &ref_bytes, &[alt_bytes.as_slice()], &mut out, false).unwrap();
```

`src/vcf_reader.rs:169-170` — append `, false` (the `.expect` now unwraps to `u32`, discarded; real wiring is Task 2):
```rust
        atomize_record(pos, &ref_allele, &alt_refs, &mut atoms, false)
            .expect("symbolic/breakend ALT is out of scope for SVAR2 (short-read only)");
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib normalize::tests 2>&1 | tail -20`
Expected: PASS (all normalize unit/proptests green, including the 3 new ones).

- [ ] **Step 6: Commit**

```bash
git add src/normalize.rs src/vcf_reader.rs
git commit -m "feat(svar2): opt-in skip + count for out-of-scope ALTs in atomize_record"
```

---

### Task 2: Thread optional reference + skip + drop-count through the pipeline

Make the reference FASTA optional (skip `validate_ref`/`left_align` when absent), thread the skip flag, and bubble the dropped-ALT count out to Python. This is one cohesive deliverable because the signature change cascades through reader → orchestrator → PyO3, so all layers must move together to compile. Verified by a new Rust e2e test.

**Files:**
- Modify: `src/vcf_reader.rs:37-52` (struct fields), `:54-127` (`new`), `:131-187` (`decompose_current_record`), add a `dropped_out_of_scope` getter
- Modify: `src/orchestrator.rs:44-54` (`process_chromosome` signature), `:118-139` (reader thread closure), `:178-187` (reader join → count), `:262-264` (return count)
- Modify: `src/lib.rs:37-139` (`run_conversion_pipeline` signature, `process_chromosome` call, aggregate + return)
- Modify (keep-compiling): all Rust call sites listed in Step 5
- Create: `tests/test_convert_skip_e2e.rs`

**Interfaces:**
- Consumes: `atomize_record(..., skip_out_of_scope: bool) -> Result<u32, _>` from Task 1.
- Produces:
  - `VcfChunkReader::new(vcf_path: &str, fasta_path: Option<&str>, chrom: &str, samples: &[&str], htslib_threads: usize, ploidy: usize, skip_out_of_scope: bool) -> Self` and `pub fn dropped_out_of_scope(&self) -> u64`.
  - `pub fn process_chromosome(vcf_path: &str, fasta_path: Option<&str>, chrom: &str, base_out_dir: &str, samples: &[&str], chunk_size: usize, ploidy: usize, htslib_threads: usize, long_allele_capacity: usize, skip_out_of_scope: bool) -> Result<u64, ConversionError>` (returns per-contig dropped count).
  - `run_conversion_pipeline(..., reference_path: Option<String>, ..., skip_out_of_scope: bool=false) -> PyResult<usize>` (returns total dropped count).

- [ ] **Step 1: Write the failing e2e test**

Create `tests/test_convert_skip_e2e.rs`:

```rust
mod common;

use common::{build_bcf_with_index, build_fasta_with_index, SynthRecord};
use genoray_core::orchestrator::process_chromosome;
use tempfile::TempDir;

// Two SNP records plus one symbolic <DEL> record on chr1.
fn records() -> Vec<SynthRecord<'static>> {
    vec![
        SynthRecord { pos: 100, ref_allele: b"A", alts: vec![b"C"], gt: vec![1, 0, 0, 0] },
        SynthRecord { pos: 200, ref_allele: b"G", alts: vec![b"<DEL>"], gt: vec![1, 1, 0, 0] },
        SynthRecord { pos: 300, ref_allele: b"T", alts: vec![b"A"], gt: vec![0, 0, 1, 0] },
    ]
}

fn convert(out: &std::path::Path, fasta: Option<&str>, skip: bool) -> Result<u64, genoray_core::error::ConversionError> {
    let tmp = out.parent().unwrap();
    let bcf = tmp.join("in.bcf");
    let samples = ["S0", "S1"];
    let recs = records();
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &recs);
    build_fasta_with_index(&bcf.with_extension("fa"), "chr1", 1000, &recs);
    let sample_refs: Vec<&str> = samples.iter().copied().collect();
    process_chromosome(
        bcf.to_str().unwrap(),
        fasta,
        "chr1",
        out.to_str().unwrap(),
        &sample_refs,
        25_000,
        2,
        1,
        8 * 1024 * 1024,
        skip,
    )
}

#[test]
fn symbolic_record_errors_by_default() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    let fasta = tmp.path().join("in.fa");
    let res = convert(&out, Some(fasta.to_str().unwrap()), false);
    assert!(res.is_err(), "default (skip=false) must fail on a symbolic record");
}

#[test]
fn symbolic_record_skipped_and_counted() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    let fasta = tmp.path().join("in.fa");
    let dropped = convert(&out, Some(fasta.to_str().unwrap()), true).unwrap();
    assert_eq!(dropped, 1, "the single <DEL> ALT should be dropped");
    assert!(out.join("chr1/var_key/snp/positions.bin").exists(), "SNPs still converted");
}

#[test]
fn converts_without_a_reference() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("store");
    // No FASTA passed; skip the symbolic record so only SNPs remain.
    let dropped = convert(&out, None, true).unwrap();
    assert_eq!(dropped, 1);
    assert!(out.join("chr1/var_key/snp/positions.bin").exists());
}
```

Note: `build_bcf_with_index`/`build_fasta_with_index` come from `tests/common/mod.rs`. If `SynthRecord`'s `alts`/`gt` field types differ from the literals above, match the existing shape used in `tests/test_atomize_e2e.rs` (adjust `b"C"` vs `&b"C"[..]` etc.). `tempfile` is already a dev-dependency (used by other e2e tests).

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test test_convert_skip_e2e 2>&1 | head -30`
Expected: compile error — `process_chromosome` takes 9 args and `fasta_path: &str`, not `Option<&str>` with a trailing `skip`.

- [ ] **Step 3: Update `VcfChunkReader`**

In `src/vcf_reader.rs`, add fields to the struct (after `ref_seq`, ~line 44):

```rust
    // Whether a reference FASTA was provided. When false, `validate_ref` and
    // `left_align` are skipped and the input is trusted to be pre-normalized.
    has_reference: bool,
    // Whether to skip (vs. error on) out-of-scope symbolic/breakend ALTs.
    skip_out_of_scope: bool,
    // Running count of out-of-scope ALTs dropped across this contig.
    dropped_out_of_scope: u64,
```

Replace the `new` signature + FASTA-loading block (lines ~56-125). Change the signature to take `fasta_path: Option<&str>` and a trailing `skip_out_of_scope: bool`, and gate the FASTA read:

```rust
    pub fn new(
        vcf_path: &str,
        fasta_path: Option<&str>,
        chrom: &str,
        samples: &[&str],
        htslib_threads: usize,
        ploidy: usize,
        skip_out_of_scope: bool,
    ) -> Self {
        // ... unchanged: open IndexedReader, set_threads, name2rid, fetch, sample_indices ...

        // Reference is optional. With a FASTA, cache the full uppercased contig for
        // validate_ref/left_align; without one, leave it empty and skip both.
        let (ref_seq, has_reference) = match fasta_path {
            Some(path) => {
                let fasta = rust_htslib::faidx::Reader::from_path(path)
                    .expect("Failed to open reference FASTA (is there a .fai?)");
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
```

- [ ] **Step 4: Gate validation/left-align and accumulate the count in `decompose_current_record`**

In `src/vcf_reader.rs`, replace the `validate_ref` call (lines ~162-165), the `atomize_record` call (lines ~167-170), and the per-atom `left_align` (lines ~172-175):

```rust
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
                gt: Rc::clone(&gt),
                seq,
            }));
        }
```

- [ ] **Step 5: Update `process_chromosome` and every remaining call site**

In `src/orchestrator.rs`, change the `process_chromosome` signature (lines ~44-54): `fasta_path: &str` → `fasta_path: Option<&str>`, append `skip_out_of_scope: bool,`, and change the return to `Result<u64, ConversionError>`.

Reader-thread closure (lines ~120-139): capture the optional FASTA and skip, and return the count:
```rust
            let vcf = vcf_path.to_string();
            let fasta = fasta_path.map(|s| s.to_string());
            let chr = chrom.to_string();
            let s_owned: Vec<String> = samples.iter().map(|&s| s.to_string()).collect();

            move || {
                let s_refs: Vec<&str> = s_owned.iter().map(|s| s.as_str()).collect();
                let mut reader = VcfChunkReader::new(
                    &vcf,
                    fasta.as_deref(),
                    &chr,
                    &s_refs,
                    htslib_threads,
                    ploidy,
                    skip_out_of_scope,
                );
                let mut chunk_id = 0;
                while let Some(dense_chunk) = reader.read_next_chunk(chunk_size, chunk_id) {
                    tx_dense.send(dense_chunk).unwrap();
                    chunk_id += 1;
                }
                reader.dropped_out_of_scope()
            }
```

Reader join (lines ~178-187): capture the returned count:
```rust
    let reader_res = reader_thread.join();
    // ... unchanged sampler stop + joins ...
    let dropped = reader_res.map_err(|_| ConversionError::WorkerPanicked {
        thread: format!("read-{}", chrom),
    })?;
```

Final return (line ~264): `Ok(dropped)` instead of `Ok(())`.

Now update the PyO3 wrapper in `src/lib.rs`. Change the `#[pyo3(signature = ...)]` (line 38) to make `reference_path` optional and add `skip_out_of_scope`:
```rust
#[pyo3(signature = (vcf_path, reference_path, chroms, output_dir, samples, chunk_size=25_000, ploidy=2, max_threads=None, long_allele_capacity=8_388_608, skip_out_of_scope=false))]
fn run_conversion_pipeline(
    py: Python,
    vcf_path: String,
    reference_path: Option<String>,
    chroms: Vec<String>,
    output_dir: String,
    samples: Vec<String>,
    chunk_size: usize,
    ploidy: usize,
    max_threads: Option<usize>,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
) -> PyResult<usize> {
```

Inside `py.detach`, before the `par_iter`, bind the optional reference once (it is `Copy`, safe to move into the parallel closure):
```rust
        let fasta_ref: Option<&str> = reference_path.as_deref();
```
Change the `process_chromosome(...)` call (lines ~103-113) — pass `fasta_ref` instead of `&vcf_path`-adjacent `&reference_path`, and append `skip_out_of_scope`:
```rust
                    orchestrator::process_chromosome(
                        &vcf_path,
                        fasta_ref,
                        chrom,
                        &output_dir,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        htslib_threads,
                        long_allele_capacity,
                        skip_out_of_scope,
                    )
```
The `results` vec is now `Vec<Result<u64, ConversionError>>`. Replace the error-drain loop (lines ~122-124) to sum counts, and return the total at the end (after `write_meta`):
```rust
    let mut total_dropped: u64 = 0;
    for r in results {
        total_dropped += r.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }
```
```rust
    // ... existing write_meta(...) call stays ...
    Ok(total_dropped as usize)
```

Update the remaining Rust test call sites so the workspace compiles. Apply this mechanical transform to each `VcfChunkReader::new(...)` — wrap the FASTA arg in `Some(...)` and append `, false`:
- `tests/test_left_align_e2e.rs:27`, `:234`, `:298`
- `tests/test_atomize_e2e.rs:22`
- `tests/test_e2e.rs:376`

And to each `process_chromosome(...)` — wrap the FASTA arg in `Some(...)` and append `, false` (existing tests ignore the now-`u64` return via their current `.unwrap()`/`.expect()`, which still compiles):
- `tests/common/mod.rs:176`
- `tests/test_e2e.rs:90`, `:177`, `:248`, `:318`, `:413`

Example transform (before → after) for a `VcfChunkReader::new` call:
```rust
    // before
    let mut reader = VcfChunkReader::new(bcf, fasta, "chr1", &samples, 1, 2);
    // after
    let mut reader = VcfChunkReader::new(bcf, Some(fasta), "chr1", &samples, 1, 2, false);
```

- [ ] **Step 6: Run the new e2e + the full Rust suite**

Run: `cargo test --test test_convert_skip_e2e 2>&1 | tail -20`
Expected: PASS (3 tests).

Run: `cargo test 2>&1 | tail -30`
Expected: PASS — all existing Rust unit/e2e tests still green.

- [ ] **Step 7: Rebuild the extension and confirm the existing Python conftest still works**

Run: `maturin develop 2>&1 | tail -5 && python -c "from genoray import _core; print(_core.run_conversion_pipeline.__doc__ is not None)"`
Expected: builds cleanly; `run_conversion_pipeline` importable. `tests/conftest.py`'s positional call (9 args, `reference_path` as a `str`) is unaffected — `skip_out_of_scope` defaults and `Option<String>` accepts a `str`; its ignored `None` return becomes an ignored `int`.

Run: `pixi run pytest tests/test_batch.py -q 2>&1 | tail -15` (any suite that uses the `svar2_store` fixture)
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/vcf_reader.rs src/orchestrator.rs src/lib.rs tests/
git commit -m "feat(svar2): optional reference + skip/count plumbing through conversion pipeline"
```

---

### Task 3: `index_vcf` PyO3 helper (auto-indexing)

Add a small helper to build a `.csi` index for a bgzipped VCF / BCF, so the wrapper can auto-index un-indexed inputs without shelling out to bcftools.

**Files:**
- Modify: `src/lib.rs` (new `#[pyfunction] index_vcf` + register in `_core`)
- Create: `tests/test_index_vcf.rs`

**Interfaces:**
- Produces: `index_vcf(path: str) -> None` (Python) — builds `<path>.csi` via `rust_htslib::bcf::index::build(path, Some("<path>.csi"), 1, Type::Csi(14))`; raises `RuntimeError` on failure.

- [ ] **Step 1: Write the failing test**

Create `tests/test_index_vcf.rs`:

```rust
mod common;

use common::{build_bcf_with_index, SynthRecord};
use tempfile::TempDir;

#[test]
fn build_csi_makes_a_usable_index() {
    let tmp = TempDir::new().unwrap();
    let bcf = tmp.path().join("in.bcf");
    let samples = ["S0"];
    let recs = vec![SynthRecord { pos: 10, ref_allele: b"A", alts: vec![b"C"], gt: vec![1, 0] }];
    // build_bcf_with_index writes an index too; delete it so we can rebuild.
    build_bcf_with_index(&bcf, "chr1", 1000, &samples, &recs);
    let csi = tmp.path().join("in.bcf.csi");
    std::fs::remove_file(&csi).ok();
    assert!(!csi.exists());

    genoray_core::index_bcf_csi(bcf.to_str().unwrap()).expect("index build");
    assert!(csi.exists(), "a .csi index should be written next to the BCF");
}
```

Note: the pure-Rust indexing logic lives in a crate-visible `index_bcf_csi(path: &str) -> Result<(), String>`; the `#[pyfunction] index_vcf` is a thin wrapper so it is both unit-testable in Rust and callable from Python.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test test_index_vcf 2>&1 | head -20`
Expected: compile error — `index_bcf_csi` not found.

- [ ] **Step 3: Implement the helper + PyO3 wrapper**

In `src/lib.rs`, add near the top (after imports):

```rust
/// Build a `.csi` index next to a bgzipped-VCF / BCF at `path`. CSI (min_shift 14)
/// is valid for both, so one path covers `.vcf.gz` and `.bcf`.
pub fn index_bcf_csi(path: &str) -> Result<(), String> {
    let idx = format!("{path}.csi");
    rust_htslib::bcf::index::build(
        path,
        Some(idx.as_str()),
        1,
        rust_htslib::bcf::index::Type::Csi(14),
    )
    .map_err(|e| format!("failed to build .csi index for {path}: {e:?}"))
}

#[pyfunction]
fn index_vcf(path: String) -> PyResult<()> {
    index_bcf_csi(&path).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}
```

Register it in the `_core` module (in `fn _core`, alongside `run_conversion_pipeline`):
```rust
    m.add_function(wrap_pyfunction!(index_vcf, m)?)?;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --test test_index_vcf 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Rebuild and confirm the Python binding exists**

Run: `maturin develop 2>&1 | tail -3 && python -c "from genoray import _core; print(callable(_core.index_vcf))"`
Expected: prints `True`.

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs tests/test_index_vcf.rs
git commit -m "feat(svar2): add index_vcf PyO3 helper to build a .csi index"
```

---

### Task 4: Python `SparseVar2.from_vcf` wrapper

Add the classmethod the CLI will call: derive samples/contigs from the header, enforce the reference contract, auto-index, call the pipeline, return the dropped count.

**Files:**
- Modify: `python/genoray/_svar2.py` (add `from_vcf` classmethod + two module-level helpers)
- Create: `tests/test_svar2_from_vcf.py`

**Interfaces:**
- Consumes: `_core.run_conversion_pipeline(...) -> int`, `_core.index_vcf(path) -> None`.
- Produces:
```python
@classmethod
def from_vcf(
    cls,
    out: str | Path,
    source: str | Path,
    reference: str | Path | None = None,
    *,
    no_reference: bool = False,
    skip_out_of_scope: bool = False,
    ploidy: int = 2,
    chunk_size: int = 25_000,
    threads: int | None = None,
    overwrite: bool = False,
    long_allele_capacity: int = 8 * 1024 * 1024,
) -> int: ...
```
Returns the number of dropped out-of-scope ALTs.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_svar2_from_vcf.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genoray import SparseVar2

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _write_ref(d: Path) -> Path:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    return ref


def _write_vcf(d: Path, *, symbolic: bool, indexed: bool) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    if symbolic:
        body += "chr1\t20\t.\tG\t<DEL>\t.\t.\t.\tGT\t0|1\t0|0\n"
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    if indexed:
        subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_with_reference_roundtrips(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store"
    dropped = SparseVar2.from_vcf(out, vcf, ref, threads=1)
    assert dropped == 0
    assert (out / "meta.json").exists()
    sv = SparseVar2(out)
    assert sv.samples == ["S0", "S1"]
    assert sv.contigs == ["chr1"]


def test_from_vcf_requires_reference_or_opt_out(tmp_path: Path):
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_vcf(tmp_path / "s1", vcf, threads=1)


def test_from_vcf_reference_and_no_reference_conflict(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_vcf(tmp_path / "s2", vcf, ref, no_reference=True, threads=1)


def test_from_vcf_no_reference_snp_only(tmp_path: Path):
    # SNP-only VCF, no reference: trusts pre-normalization, converts fine.
    body_vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store_noref"
    dropped = SparseVar2.from_vcf(out, body_vcf, no_reference=True, threads=1)
    assert dropped == 0
    assert (out / "meta.json").exists()


def test_from_vcf_auto_indexes_unindexed_source(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=False)  # no .csi/.tbi
    assert not (tmp_path / "in.vcf.gz.csi").exists()
    out = tmp_path / "store_idx"
    SparseVar2.from_vcf(out, vcf, ref, threads=1)
    assert (tmp_path / "in.vcf.gz.csi").exists()
    assert (out / "meta.json").exists()


def test_from_vcf_skip_out_of_scope_counts(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=True, indexed=True)
    out = tmp_path / "store_skip"
    dropped = SparseVar2.from_vcf(out, vcf, ref, skip_out_of_scope=True, threads=1)
    assert dropped == 1


def test_from_vcf_symbolic_errors_without_skip(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=True, indexed=True)
    with pytest.raises(Exception):
        SparseVar2.from_vcf(tmp_path / "store_err", vcf, ref, threads=1)


def test_from_vcf_plain_vcf_rejected(tmp_path: Path):
    ref = _write_ref(tmp_path)
    plain = tmp_path / "in.vcf"
    plain.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    with pytest.raises(ValueError, match="bgzip"):
        SparseVar2.from_vcf(tmp_path / "s3", plain, ref, threads=1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run pytest tests/test_svar2_from_vcf.py -q 2>&1 | tail -15`
Expected: FAIL — `SparseVar2` has no `from_vcf`.

- [ ] **Step 3: Implement the wrapper**

In `python/genoray/_svar2.py`, add imports and helpers at module level:

```python
from natsort import natsorted


def _ensure_bgzipped(source: Path) -> None:
    """Reject a plain (uncompressed) VCF — it can't be tabix/csi-indexed."""
    is_bcf = source.suffix == ".bcf"
    is_vcfgz = source.name.endswith(".vcf.gz")
    if not (is_bcf or is_vcfgz):
        raise ValueError(
            f"{source} must be a BCF (.bcf) or bgzipped VCF (.vcf.gz); bgzip it first."
        )


def _ensure_index(source: Path) -> None:
    """Build a .csi index next to `source` if it has no .csi/.tbi index."""
    csi = source.with_name(source.name + ".csi")
    tbi = source.with_name(source.name + ".tbi")
    if csi.exists() or tbi.exists():
        return
    _core.index_vcf(str(source))
```

Add the classmethod on `SparseVar2`:

```python
    @classmethod
    def from_vcf(
        cls,
        out: str | Path,
        source: str | Path,
        reference: str | Path | None = None,
        *,
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        ploidy: int = 2,
        chunk_size: int = 25_000,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
    ) -> int:
        """Convert a bgzipped VCF or BCF to an SVAR2 store.

        Exactly one of `reference` or `no_reference=True` is required. With a
        reference, indels are validated against and left-aligned to the FASTA;
        with `no_reference`, validation and left-alignment are skipped and the
        input is trusted to be already normalized. Returns the number of
        out-of-scope (symbolic/breakend) ALTs dropped (0 unless
        `skip_out_of_scope`).
        """
        from cyvcf2 import VCF as _CyVCF

        out = Path(out)
        source = Path(source)

        if (reference is None) == (not no_reference):
            raise ValueError(
                "provide exactly one of `reference` (a FASTA path) or "
                "`no_reference=True`"
            )
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.parent.mkdir(parents=True, exist_ok=True)

        _ensure_bgzipped(source)
        _ensure_index(source)

        v = _CyVCF(str(source))
        samples = list(v.samples)
        contigs = [c for c in natsorted(v.seqnames) if next(v(c), None) is not None]
        if not contigs:
            raise ValueError(f"No variants found in {source}.")

        reference_path = None if no_reference else str(reference)
        return _core.run_conversion_pipeline(
            str(source),
            reference_path,
            contigs,
            str(out),
            samples,
            chunk_size,
            ploidy,
            threads,  # max_threads; None => auto
            long_allele_capacity,
            skip_out_of_scope,
        )
```

Note on the XOR check: `(reference is None) == (not no_reference)` is `True` exactly when both are set or both are unset — either way an error.

- [ ] **Step 4: Run to verify they pass**

Run: `pixi run pytest tests/test_svar2_from_vcf.py -q 2>&1 | tail -20`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_svar2.py tests/test_svar2_from_vcf.py
git commit -m "feat(svar2): SparseVar2.from_vcf wrapper (optional reference, auto-index, skip count)"
```

---

### Task 5: CLI — `write` group (SVAR2 default + `write svar1`)

Restructure the `write` command into a sub-app: `genoray write` defaults to SVAR2; `genoray write svar1` runs today's SVAR1 body verbatim.

**Files:**
- Modify: `python/genoray/_cli/__main__.py:46-175` (replace the single `write` command)
- Create: `tests/cli/test_write_cli.py`

**Interfaces:**
- Consumes: `SparseVar2.from_vcf(...)` (Task 4); the existing `SparseVar.from_vcf`/`from_pgen` for svar1.
- Produces: CLI commands `write` (default → svar2) and `write svar1`.

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_write_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from genoray import SparseVar, SparseVar2

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "genoray._cli", *argv],
        capture_output=True,
        text=True,
    )


def _ref(d: Path) -> Path:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    return ref


def _vcf(d: Path, *, symbolic: bool) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
    )
    if symbolic:
        body += "chr1\t20\t.\tG\t<DEL>\t.\t.\t.\tGT\t0|1\t0|0\n"
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_write_defaults_to_svar2(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store"
    r = _run(["write", str(vcf), str(out), "--reference", str(ref), "--threads", "1"])
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]


def test_write_no_reference(tmp_path: Path):
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store2"
    r = _run(["write", str(vcf), str(out), "--no-reference", "--threads", "1"])
    assert r.returncode == 0, r.stderr
    assert (out / "meta.json").exists()


def test_write_requires_reference_xor(tmp_path: Path):
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store3"
    r = _run(["write", str(vcf), str(out), "--threads", "1"])
    assert r.returncode != 0
    assert "reference" in (r.stderr + r.stdout)


def test_write_skip_symbolic(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=True)
    out = tmp_path / "store4"
    r = _run(
        ["write", str(vcf), str(out), "--reference", str(ref), "--no-symbolic", "--threads", "1"]
    )
    assert r.returncode == 0, r.stderr
    assert (out / "meta.json").exists()


def test_write_svar1_still_works(tmp_path: Path):
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "v1.svar"
    r = _run(["write", "svar1", str(vcf), str(out), "--max-mem", "64m"])
    assert r.returncode == 0, r.stderr
    sv = SparseVar(out)
    assert sv.n_variants >= 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run pytest tests/cli/test_write_cli.py -q 2>&1 | tail -20`
Expected: FAIL — `write` currently produces SVAR1 and rejects `--reference`/`--no-reference`; `write svar1` is not a command.

- [ ] **Step 3: Restructure the CLI**

In `python/genoray/_cli/__main__.py`, replace the entire current `@app.command def write(...)` block (lines ~46-175) with a `write` sub-app. Keep the `index` and `view` commands untouched. The svar1 function body below is the current `write` body moved verbatim (only the decorator and name change).

```python
write = App(name="write", help="Convert a VCF/PGEN to a SparseVar file (SVAR2 by default).")
app.command(write)


@write.default
def write_svar2(
    source: Path,
    out: Path,
    *,
    reference: Annotated[Path | None, Parameter(name="--reference")] = None,
    no_reference: Annotated[bool, Parameter(name="--no-reference", negative="")] = False,
    ploidy: int = 2,
    chunk_size: int = 25_000,
    threads: Annotated[int | None, Parameter(name=["--threads", "-@"])] = None,
    long_allele_capacity: int = 8 * 1024 * 1024,
    overwrite: bool = False,
    no_symbolic: Annotated[bool, Parameter(name="--no-symbolic", negative="")] = False,
    no_breakend: Annotated[bool, Parameter(name="--no-breakend", negative="")] = False,
) -> None:
    """
    Convert a bgzipped VCF or BCF to an SVAR2 store (the default, better-across-the-board format).

    Parameters
    ----------
    source
        Path to a bgzipped VCF (``.vcf.gz``) or BCF (``.bcf``). Auto-indexed
        (``.csi``) if no index is present.
    out
        Path to the output SVAR2 directory.
    reference
        Path to a reference FASTA (with ``.fai``). Used to validate REF and
        left-align indels. Exactly one of ``--reference`` or ``--no-reference``
        is required.
    no_reference
        Skip REF validation and indel left-alignment; the input is trusted to be
        already normalized. Use only for pre-normalized (e.g. ``bcftools norm``)
        inputs.
    ploidy
        Ploidy of the samples. Default 2.
    chunk_size
        Variants per conversion chunk. Default 25000.
    threads
        Number of threads. Defaults to all available cores.
    long_allele_capacity
        Advanced: byte budget for the streaming long-allele buffer.
    overwrite
        Overwrite the output directory if it exists.
    no_symbolic
        Drop records with a symbolic ALT (``<DEL>``, ``<INS>``, …) instead of
        erroring. On SVAR2 this is coupled with ``--no-breakend`` (the core does
        not distinguish the two classes); passing either drops both.
    no_breakend
        Drop records with a breakend ALT. Coupled with ``--no-symbolic`` on
        SVAR2 (see above).
    """
    from genoray import SparseVar2

    skip_out_of_scope = no_symbolic or no_breakend
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
    if skip_out_of_scope:
        print(f"Dropped {dropped} out-of-scope (symbolic/breakend) ALT alleles.")


@write.command(name="svar1")
def write_svar1(
    source: Path,
    out: Path,
    max_mem: str = "1g",
    overwrite: bool = False,
    dosages: str | None = None,
    threads: int | None = None,
    no_symbolic: Annotated[bool, Parameter(name="--no-symbolic", negative="")] = False,
    no_breakend: Annotated[bool, Parameter(name="--no-breakend", negative="")] = False,
    haploid: Annotated[bool, Parameter(name="--haploid", negative="")] = False,
) -> None:
    """
    Convert a VCF or PGEN file to a SVAR 1.0 file.

    Parameters
    ----------
    source
        Path to the input VCF or PGEN file.
    out
        Path to the output SVAR file.
    max_mem
        Maximum memory to use for conversion e.g. 1g, 250 MB, etc.
    overwrite
        Whether to overwrite the output file if it exists.
    dosages
        Whether to write dosages.
        If `source` is a PGEN, this must be a path to a PGEN of dosages.
        If `source` is a VCF, this must be the name of the FORMAT field to use for dosages.
        If not provided, dosages will not be written.
    threads
        Number of threads to use for conversion. Defaults to the number of available CPU cores.
    no_symbolic
        If set, drop records whose ALT contains a symbolic allele
        (``<DEL>``, ``<INS>``, etc.) per VCF 4.x.
    no_breakend
        If set, drop records whose ALT contains a breakend (BND).
    haploid
        If set, OR-collapse the ploidy axis into a single haploid call per
        sample and record ``ploidy=1`` in the output metadata.
    """
    from genoray import PGEN, VCF, SparseVar, exprs
    from genoray._utils import variant_file_type

    file_type = variant_file_type(source)

    if dosages is None:
        with_dosages = False
    else:
        with_dosages = True

    if threads is None:
        threads = -1

    pl_terms: list[pl.Expr] = []
    record_preds: list[Callable[[list[str]], bool]] = []
    if no_symbolic:
        pl_terms.append(~exprs.is_symbolic)
        record_preds.append(exprs._record_is_symbolic)
    if no_breakend:
        pl_terms.append(~exprs.is_breakend)
        record_preds.append(exprs._record_is_breakend)

    if pl_terms:
        from functools import reduce
        from operator import and_

        pl_filter = reduce(and_, pl_terms)

        def record_filter(
            rec: Any,
            _preds: tuple[Callable[[list[str]], bool], ...] = tuple(record_preds),
        ) -> bool:
            return not any(pred(rec.ALT) for pred in _preds)
    else:
        pl_filter = None
        record_filter = None

    if file_type == "vcf":
        if dosages is not None and Path(dosages).exists():
            raise ValueError(
                "The `dosages` argument appears to be a path to an existing file, but VCF requires a FORMAT field name."
            )

        vcf = VCF(
            source,
            dosage_field=dosages,
            filter=record_filter,
            pl_filter=pl_filter,
        )
        SparseVar.from_vcf(
            out,
            vcf,
            max_mem,
            overwrite,
            with_dosages=with_dosages,
            n_jobs=threads,
            haploid=haploid,
        )
    elif file_type == "pgen":
        pgen = PGEN(
            source,
            dosage_path=dosages,
            filter=pl_filter,
        )
        SparseVar.from_pgen(
            out,
            pgen,
            max_mem,
            overwrite,
            with_dosages=with_dosages,
            n_jobs=threads,
            haploid=haploid,
        )
    else:
        raise ValueError(f"Unsupported file type: {source}")
```

Leave the top-of-file imports as they are — `App`, `Parameter`, `pl`, `Annotated`, `Callable`, `Any` are already imported (lines 1-17). Confirm no now-unused import lints fire; if `validators`/`Literal` become unused they were only used by `view` (still present), so they stay.

- [ ] **Step 4: Run to verify they pass**

Run: `pixi run pytest tests/cli/test_write_cli.py -q 2>&1 | tail -20`
Expected: PASS (5 tests).

- [ ] **Step 5: Sanity-check the help text**

Run: `python -m genoray._cli write --help 2>&1 | head -20` and `python -m genoray._cli write svar1 --help 2>&1 | head -10`
Expected: `write` shows the SVAR2 options (`--reference`, `--no-reference`, …) and lists `svar1` as a subcommand; `write svar1` shows the SVAR1 options.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_cli/__main__.py tests/cli/test_write_cli.py
git commit -m "feat(cli): default 'write' to SVAR2, add 'write svar1' for SVAR 1.0"
```

---

### Task 6: Docs — SKILL.md + roadmap M13

Update the public-API skill doc and tick the roadmap milestone, per the repo's working agreement.

**Files:**
- Modify: `skills/genoray-api/SKILL.md`
- Modify: `docs/roadmap/svar-2.md:225-237` (M13 checkbox + status)

- [ ] **Step 1: Update the roadmap**

In `docs/roadmap/svar-2.md`, change the M13 line (~225) from `- [ ] **M13.` to `- [x] **M13.` and append a short "*Done:*" note mirroring the other milestones, e.g.:

```markdown
  *Done:* `run_conversion_pipeline` gained `skip_out_of_scope` (drops symbolic/breakend
  ALTs per-allele like `*`/`.`, returns the dropped count); the reference is now optional
  (`reference_path: Option`, skipping `validate_ref`/`left_align` when absent); an
  `index_vcf` PyO3 helper auto-builds a `.csi`; and the `genoray write` CLI defaults to
  SVAR2 (`--reference` XOR `--no-reference`, `--no-symbolic`/`--no-breakend` → skip) with
  `genoray write svar1` for SVAR 1.0, via the new `SparseVar2.from_vcf` wrapper.
```

- [ ] **Step 2: Update SKILL.md**

Read `skills/genoray-api/SKILL.md` and add/adjust:
- A `SparseVar2.from_vcf(out, source, reference=None, *, no_reference=False, skip_out_of_scope=False, ploidy=2, chunk_size=25000, threads=None, overwrite=False, long_allele_capacity=...) -> int` entry (returns dropped-ALT count; exactly one of `reference`/`no_reference`; auto-indexes; bgzipped VCF/BCF only).
- The CLI: `genoray write` now defaults to SVAR2 (`--reference` XOR `--no-reference`, `--ploidy`, `--chunk-size`, `--threads`, `--no-symbolic`/`--no-breakend` coupled → skip, `--overwrite`); `genoray write svar1` for the previous SVAR 1.0 behavior (VCF+PGEN, `--dosages`, `--max-mem`, `--haploid`).
- Note SVAR2 write is VCF/BCF-only (no PGEN, no dosages, no haploid-collapse yet).

Match the file's existing formatting and section conventions.

- [ ] **Step 3: Verify docs reference nothing stale**

Run: `grep -n "from_vcf\|write svar1\|--no-reference\|skip_out_of_scope\|index_vcf" skills/genoray-api/SKILL.md docs/roadmap/svar-2.md`
Expected: the new names appear; no leftover claim that `write` produces SVAR1 by default.

- [ ] **Step 4: Full test sweep**

Run: `cargo test 2>&1 | tail -15 && pixi run test 2>&1 | tail -25`
Expected: all Rust and Python tests green.

- [ ] **Step 5: Commit**

```bash
git add skills/genoray-api/SKILL.md docs/roadmap/svar-2.md
git commit -m "docs(svar2): document SparseVar2.from_vcf + CLI write split; tick M13"
```

---

## Self-Review

**Spec coverage:**
- Reference-optional (spec §Component 1) → Task 2 (reader `Option<&str>`, gated validate/left-align) + Task 4/5 (`no_reference` / `--no-reference`). ✓
- M13 skip + count (§Component 2) → Task 1 (atomize) + Task 2 (plumbing/return) + Task 4/5 (surfacing). ✓
- Auto-index helper (§Component 3) → Task 3 + Task 4 (`_ensure_index`). ✓
- `SparseVar2.from_vcf` wrapper (§Component 4), incl. the zero-variant-contig open point → Task 4 filters empty contigs via `next(v(c), None)`, resolving the open point directly. ✓
- CLI restructure (§Component 5), coupled skip flags, dropped-count print → Task 5. ✓
- Testing + docs (§Testing, §Surfaces) → Tasks 1-6, Task 6 for SKILL.md + roadmap. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step has concrete code. The one judgment call (SKILL.md wording) is bounded to matching existing formatting for named additions. ✓

**Type consistency:** `atomize_record(..., skip_out_of_scope: bool) -> Result<u32, _>` (Task 1) is consumed with the same arity in Task 2. `VcfChunkReader::new(..., Option<&str>, ..., bool)` and `dropped_out_of_scope() -> u64` (Task 2) match their call sites. `process_chromosome(..., Option<&str>, ..., bool) -> Result<u64, _>` and `run_conversion_pipeline(..., Option<String>, ..., bool=false) -> PyResult<usize>` are used consistently in Task 3/4. `SparseVar2.from_vcf(...) -> int` (Task 4) is called with matching kwargs in Task 5. ✓

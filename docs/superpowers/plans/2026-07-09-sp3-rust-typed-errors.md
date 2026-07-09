# SP-3 — Rust panics → typed errors: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `panic!`/`.expect()`/`.unwrap()` on I/O and user-input conditions in the Rust conversion and query paths with propagated typed errors, so Python callers get a meaningful exception type and message instead of a context-free `WorkerPanicked` or a process abort.

**Architecture:** Extend `ConversionError` with `Input`/`MissingFile` categories and a `From<NormalizeError>` bridge; add `impl From<ConversionError> for PyErr` mapping each category to a Python builtin. Thread `Result` through the reader → writer → merge worker paths and surface it at the orchestrator's thread joins. Thread `io::Result` through the query sidecar loaders (rides the existing `ContigReader::open -> io::Result -> PyOSError` seam). Make the `bundle_from_dict` FFI parser fallible.

**Tech Stack:** Rust (pyo3, thiserror, rayon, ndarray-npy, rust-htslib), Python (pytest), Pixi.

## Global Constraints

- Panics are retained **only** for genuine invariants (proptested internal contracts). After SP-3, a Rust panic reaching Python means "genoray bug," never "bad VCF/args."
- **Python exception surface is builtins-only** — no new public Python names. Mapping: user-input content → `ValueError`; missing required file → `FileNotFoundError`; corrupt/failed disk I/O → `OSError`; genuine panic → `RuntimeError`.
- Rust tests MUST run with `pixi run bash -lc 'cargo test --no-default-features <args>'` — the default features build a pyo3 test binary that fails to link (`undefined symbol: _Py_Dealloc`).
- Behavior-preserving on the happy path; the only intended change is panics/aborts → typed exceptions.
- `ConversionError` must stay `Send` (rayon `try_for_each` requires it): only `String`/`std::io::Error`/`ndarray_npy` sources, all `Send`.
- The full suite (`pixi run test`) and `cargo test --no-default-features` stay green after every commit.
- Commit messages follow Conventional Commits (`feat:`/`fix:`/`refactor:`). End each with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- Branch: `sp3-rust-typed-errors` (already checked out). prek hooks are installed; let them run.

---

## File Structure

- `src/error.rs` — extend `ConversionError` (new variants + `From<NormalizeError>` + `From<ConversionError> for PyErr`).
- `src/lib.rs` — `run_conversion_pipeline` result loop uses `?` via the new `PyErr` conversion.
- `src/vcf_reader.rs` — `new`, `decompose_current_record`, `next_atom`, `read_next_chunk` return `Result<_, ConversionError>`.
- `src/writer.rs` — `run_io_writer`, `run_long_allele_writer`, `write_bin` return `Result<(), ConversionError>`.
- `src/orchestrator.rs` — reader/writer thread closures return `Result`; three-way join surfacing; `?` at merge/pack call sites.
- `src/merge.rs`, `src/dense_merge.rs` — `merge_mini_sc` / `merge_dense_class` return `Result<(), ConversionError>`.
- `src/rvk.rs` — `pack_snp_key_file` returns `Result<(), ConversionError>`.
- `src/streams.rs` — `post_merge` hook type gains a `Result` return.
- `src/query/sidecar.rs` — `load_offsets`/`load_max_del`/`load_dense_max_del` return `io::Result`.
- `src/nrvk.rs` — `LongAlleleReader::new` returns `io::Result`.
- `src/query/reader.rs`, `src/query/decode.rs` — propagate the loader/`new` `io::Result`.
- `src/py_query_ranges.rs` — `bundle_from_dict` returns `PyResult<RangesBundle>`.
- `tests/test_svar2_errors.py` — NEW pytest asserting exception types/messages.
- `tests/test_svar2_from_vcf.py` — tighten the existing `raises(Exception)` symbolic test to `raises(ValueError)`.
- `skills/genoray-api/SKILL.md` — add an **Errors** subsection.

---

## Commit 1 — Taxonomy foundation

### Task 1: Extend `ConversionError` and add the Python mapping

**Files:**
- Modify: `src/error.rs`
- Modify: `src/lib.rs:182-185`
- Test: `src/error.rs` (`#[cfg(test)]` module)

**Interfaces:**
- Produces: `ConversionError::Input(String)`, `ConversionError::MissingFile { path: String }`, `impl From<NormalizeError> for ConversionError`, `impl From<ConversionError> for pyo3::PyErr`. Later tasks construct `Input`/`MissingFile`/`Io` and rely on the `PyErr` mapping.

- [ ] **Step 1: Write the failing test**

Add to the bottom of `src/error.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::NormalizeError;

    #[test]
    fn normalize_error_maps_to_input_with_message() {
        let ne = NormalizeError::RefMismatch {
            pos: 7,
            expected: "A".into(),
            found: "C".into(),
        };
        let expected_msg = ne.to_string();
        let ce: ConversionError = ne.into();
        match ce {
            ConversionError::Input(msg) => {
                assert_eq!(msg, expected_msg);
                assert!(msg.contains("disagrees"));
            }
            other => panic!("expected Input, got {other:?}"),
        }
    }

    #[test]
    fn missing_file_message_includes_path() {
        let ce = ConversionError::MissingFile {
            path: "/x/in.vcf.gz.tbi".into(),
        };
        assert!(ce.to_string().contains("/x/in.vcf.gz.tbi"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features error::tests'`
Expected: FAIL — `Input`/`MissingFile` variants and `From<NormalizeError>` do not exist (compile error).

- [ ] **Step 3: Write the implementation**

Replace the enum in `src/error.rs` (keep the module docstring) with:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// User-recoverable *content* error: bad contig/sample name, symbolic ALT,
    /// REF/FASTA mismatch, missing index. The message is the whole error.
    #[error("{0}")]
    Input(String),
    /// A required input file (index, FASTA) is absent.
    #[error("required file not found: {path}")]
    MissingFile { path: String },
    #[error("I/O error at {context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("worker thread '{thread}' panicked")]
    WorkerPanicked { thread: String },
    #[error("failed to write npy at {path}: {source}")]
    Npy {
        path: String,
        #[source]
        source: ndarray_npy::WriteNpyError,
    },
    #[error("failed to read npy at {path}: {source}")]
    ReadNpy {
        path: String,
        #[source]
        source: ndarray_npy::ReadNpyError,
    },
}

impl From<crate::normalize::NormalizeError> for ConversionError {
    fn from(e: crate::normalize::NormalizeError) -> Self {
        ConversionError::Input(e.to_string())
    }
}

impl From<ConversionError> for pyo3::PyErr {
    fn from(e: ConversionError) -> Self {
        use pyo3::exceptions::{
            PyFileNotFoundError, PyOSError, PyRuntimeError, PyValueError,
        };
        let msg = e.to_string();
        match e {
            ConversionError::Input(_) => PyValueError::new_err(msg),
            ConversionError::MissingFile { .. } => PyFileNotFoundError::new_err(msg),
            ConversionError::Io { .. }
            | ConversionError::Npy { .. }
            | ConversionError::ReadNpy { .. } => PyOSError::new_err(msg),
            ConversionError::WorkerPanicked { .. } => PyRuntimeError::new_err(msg),
        }
    }
}
```

Also update the module docstring's stale note (lines 1-3) — the "worker-thread hot loops still panic (converting them is a follow-up)" clause is now inaccurate; change it to: `//! Boundary-level typed errors for the conversion pipeline. Categories map to`
`//! Python builtins via \`impl From<ConversionError> for PyErr\`.`

- [ ] **Step 4: Update the `lib.rs` mapping to use the new conversion**

In `src/lib.rs`, replace lines 182-185:

```rust
    let mut total_dropped: u64 = 0;
    for r in results {
        total_dropped += r.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }
```

with:

```rust
    let mut total_dropped: u64 = 0;
    for r in results {
        total_dropped += r?; // ConversionError -> PyErr via From (category-aware)
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features error::tests'`
Expected: PASS (2 tests).

Run: `pixi run bash -lc 'cargo build --no-default-features'`
Expected: builds clean (the `?` in `lib.rs` resolves via the new `From`).

- [ ] **Step 6: Commit**

```bash
git add src/error.rs src/lib.rs
git commit -m "feat(error): add Input/MissingFile categories and PyErr mapping

Adds user-input vs. internal categories to ConversionError, a From<NormalizeError>
bridge, and a category-aware From<ConversionError> for PyErr (Input->ValueError,
MissingFile->FileNotFoundError, Io/Npy->OSError, WorkerPanicked->RuntimeError).
run_conversion_pipeline now propagates via ? instead of flattening to RuntimeError.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Commit 2 — Conversion workers → typed errors

### Task 2: Reader path → typed errors, surfaced at the join

This is the crown-jewel task: it turns swallowed `WorkerPanicked`s (bad contig/sample/REF/symbolic ALT) into real `ValueError`s. It is independently shippable and tested end-to-end from Python.

**Files:**
- Modify: `src/vcf_reader.rs:153-233` (`new`), `:242-323` (`decompose_current_record`), `:338-357` (`next_atom`), `:363-378+` (`read_next_chunk`)
- Modify: `src/orchestrator.rs:137-166` (reader closure), `:208-217` (reader join surfacing)
- Test: `tests/test_svar2_errors.py` (NEW)

**Interfaces:**
- Consumes: `ConversionError::{Input, MissingFile, Io}`, `From<NormalizeError>` (Task 1).
- Produces: `VcfChunkReader::new(...) -> Result<Self, ConversionError>`; `read_next_chunk(...) -> Result<Option<DenseChunk>, ConversionError>`. The reader closure returns `Result<u64, ConversionError>`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_svar2_errors.py`:

```python
"""SP-3: conversion errors surface as typed Python exceptions, not RuntimeError."""

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


def _write_vcf(d: Path, body_rows: str, *, contig_len: int = 40) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        f"##contig=<ID=chr1,length={contig_len}>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        + body_rows
    )
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_symbolic_alt_raises_value_error(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, "chr1\t20\t.\tT\t<DEL>\t.\t.\t.\tGT\t0|1\t0|0\n")
    with pytest.raises(ValueError, match="symbolic"):
        SparseVar2.from_vcf(tmp_path / "store", vcf, ref, threads=1)


def test_ref_mismatch_raises_value_error(tmp_path: Path):
    ref = _write_ref(tmp_path)
    # POS 3 (1-based) in _REF is 'A'; claim REF='G' to force a mismatch.
    vcf = _write_vcf(tmp_path, "chr1\t3\t.\tG\tT\t.\t.\t.\tGT\t0|1\t0|0\n")
    with pytest.raises(ValueError, match="disagrees"):
        SparseVar2.from_vcf(tmp_path / "store", vcf, ref, threads=1)


def test_missing_reference_raises_file_not_found(tmp_path: Path):
    vcf = _write_vcf(tmp_path, "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n")
    with pytest.raises(FileNotFoundError):
        SparseVar2.from_vcf(tmp_path / "store", vcf, tmp_path / "nope.fa", threads=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar2_errors.py -v`
Expected: FAIL — the pipeline currently panics, surfacing as `RuntimeError` (or a `pyo3_runtime.PanicException`), not `ValueError`.

- [ ] **Step 3: Convert `VcfChunkReader::new`**

In `src/vcf_reader.rs`, change the signature at line 161 from `) -> Self {` to `) -> Result<Self, ConversionError> {` and rewrite the body's fallible sites. Add `use crate::error::ConversionError;` to the file's imports if not present. Replacement body:

```rust
        let mut reader = IndexedReader::from_path(vcf_path).map_err(|e| {
            ConversionError::Input(format!(
                "Failed to open VCF/BCF index for '{vcf_path}' \
                 (is there a .tbi or .csi file?): {e}"
            ))
        })?;

        reader.set_threads(htslib_threads).map_err(|e| ConversionError::Io {
            context: format!("allocating {htslib_threads} HTSlib background threads"),
            source: std::io::Error::other(e.to_string()),
        })?;

        let header = reader.header().clone();

        let rid = header.name2rid(chrom.as_bytes()).map_err(|_| {
            ConversionError::Input(format!("Chromosome '{chrom}' not found in VCF header"))
        })?;

        reader.fetch(rid, 0, None).map_err(|e| ConversionError::Io {
            context: format!("fetching region for chromosome '{chrom}'"),
            source: std::io::Error::other(e.to_string()),
        })?;

        let sample_indices: Vec<usize> = samples
            .iter()
            .map(|name| {
                header.sample_id(name.as_bytes()).ok_or_else(|| {
                    ConversionError::Input(format!("Sample '{name}' not found in VCF"))
                })
            })
            .collect::<Result<_, _>>()?;

        // Reference is optional. With a FASTA, cache the full uppercased contig for
        // validate_ref/left_align; without one, leave it empty and skip both.
        let (ref_seq, has_reference) = match fasta_path {
            Some(path) => {
                // A wrong reference path reaches Rust unchecked (Python does not
                // validate it), so surface it as FileNotFoundError specifically.
                if !std::path::Path::new(path).exists() {
                    return Err(ConversionError::MissingFile { path: path.to_string() });
                }
                let fasta = rust_htslib::faidx::Reader::from_path(path).map_err(|e| {
                    ConversionError::Input(format!(
                        "Failed to open reference FASTA '{path}' (is there a .fai?): {e}"
                    ))
                })?;
                // htslib's faidx_seq_len returns -1 for an unknown contig, surfaced
                // via rust-htslib as u64::MAX — check explicitly for a clear message.
                let contig_len_raw = fasta.fetch_seq_len(chrom);
                if contig_len_raw == u64::MAX {
                    return Err(ConversionError::Input(format!(
                        "Contig '{chrom}' not found in reference FASTA"
                    )));
                }
                let contig_len = contig_len_raw as usize;
                let mut ref_seq = if contig_len == 0 {
                    Vec::new()
                } else {
                    fasta.fetch_seq(chrom, 0, contig_len - 1).map_err(|e| {
                        ConversionError::Io {
                            context: format!("fetching contig '{chrom}' from reference FASTA"),
                            source: std::io::Error::other(e.to_string()),
                        }
                    })?
                };
                ref_seq.make_ascii_uppercase();
                (ref_seq, true)
            }
            None => (Vec::new(), false),
        };

        let record = reader.empty_record();

        Ok(Self {
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
        })
```

- [ ] **Step 4: Convert `decompose_current_record` and `next_atom`**

Change `fn decompose_current_record(&mut self) {` (line 242) to `fn decompose_current_record(&mut self) -> Result<(), ConversionError> {`. Inside it:

Replace the GT-read `.expect(...)` (line 266-270) with:
```rust
            let gts = self
                .record
                .format(b"GT")
                .integer()
                .map_err(|e| ConversionError::Input(format!(
                    "Failed to read GT format at pos {pos}: {e}"
                )))?;
```

Replace the `validate_ref(...).expect(...)` (lines 288-289) with:
```rust
            crate::normalize::validate_ref(pos, &ref_allele, &self.ref_seq)?;
```

Replace the `atomize_record(...).expect(...)` (lines 294-301) with:
```rust
        let dropped = atomize_record(pos, &ref_allele, &alt_refs, &mut atoms, self.skip_out_of_scope)?;
```

Add `Ok(())` as the final line of the function (after the `for atom in atoms { ... }` loop).

Change `fn next_atom(&mut self) -> Option<PendingAtom> {` (line 338) to `fn next_atom(&mut self) -> Result<Option<PendingAtom>, ConversionError> {`. In its body: change `return Some(...)` → `return Ok(Some(...))`; change `return None` → `return Ok(None)`; change the `self.decompose_current_record();` call to `self.decompose_current_record()?;`; and replace `Some(Err(e)) => panic!("VCF Read Error: {}", e),` with:
```rust
                Some(Err(e)) => {
                    return Err(ConversionError::Io {
                        context: "reading next VCF record".to_string(),
                        source: std::io::Error::other(e.to_string()),
                    })
                }
```

- [ ] **Step 5: Convert `read_next_chunk`**

Change the signature (line 363-368) return type from `-> Option<DenseChunk> {` to `-> Result<Option<DenseChunk>, ConversionError> {`. Change the gather loop (lines 370-378):

```rust
        let mut atoms: Vec<PendingAtom> = Vec::with_capacity(chunk_size);
        while atoms.len() < chunk_size {
            match self.next_atom()? {
                Some(a) => atoms.push(a),
                None => break,
            }
        }
        if atoms.is_empty() {
            return Ok(None);
        }
```

At the end of the function, wrap the returned `DenseChunk` in `Ok(Some(...))` (find the final `Some(DenseChunk { ... })` or `DenseChunk { ... }` return and make it `Ok(Some(DenseChunk { ... }))`).

- [ ] **Step 6: Update the reader thread closure + join in `orchestrator.rs`**

Replace the closure body (lines 147-166, the `move || { ... }` block) with a `Result`-returning closure:

```rust
            move || -> Result<u64, ConversionError> {
                // passing the thread budget down to HTSLib
                let s_refs: Vec<&str> = s_owned.iter().map(|s| s.as_str()).collect();
                let mut reader = VcfChunkReader::new(
                    &vcf,
                    fasta.as_deref(),
                    &chr,
                    &s_refs,
                    htslib_threads,
                    ploidy,
                    skip_out_of_scope,
                )?;
                let mut chunk_id = 0;
                while let Some(dense_chunk) =
                    reader.read_next_chunk(chunk_size, chunk_id, Some(&pool))?
                {
                    tx_dense.send(dense_chunk).unwrap();
                    chunk_id += 1;
                }
                Ok(reader.dropped_out_of_scope())
            }
```

Replace the reader-join surfacing (lines 215-217):

```rust
    let dropped = match reader_res {
        Ok(r) => r?, // ConversionError propagates with its real message
        Err(_) => {
            return Err(ConversionError::WorkerPanicked {
                thread: format!("read-{}", chrom),
            })
        }
    };
```

(Leave the sampler/executor/writer joins at 218-234 unchanged for now — Task 3/4 update the writer ones.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo build --no-default-features'`
Expected: builds clean.

Run: `pixi run pytest tests/test_svar2_errors.py -v`
Expected: PASS (3 tests) — symbolic + REF-mismatch raise `ValueError`, missing reference raises `FileNotFoundError`.

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: PASS (existing Rust tests unaffected).

- [ ] **Step 8: Commit**

```bash
git add src/vcf_reader.rs src/orchestrator.rs tests/test_svar2_errors.py
git commit -m "fix(vcf): surface reader errors as typed ValueError instead of WorkerPanicked

Threads Result through VcfChunkReader (new/decompose/next_atom/read_next_chunk)
and the orchestrator reader thread, so bad contig, missing sample, REF mismatch,
and symbolic ALT reach Python as ValueError with the real message.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 3: Writer threads → typed errors

**Files:**
- Modify: `src/writer.rs:14-77` (`run_io_writer`, `run_long_allele_writer`, `write_bin`)
- Modify: `src/orchestrator.rs:186-201` (writer closures), `:229-234` (writer joins)
- Test: `src/writer.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `ConversionError::Io` (Task 1).
- Produces: `run_io_writer(...) -> Result<(), ConversionError>`, `run_long_allele_writer(...) -> Result<(), ConversionError>`, `write_bin(...) -> Result<(), ConversionError>`.

- [ ] **Step 1: Write the failing test**

Add to `src/writer.rs`'s test module:

```rust
    #[test]
    fn write_bin_returns_io_error_on_unwritable_path() {
        // A path whose parent dir does not exist cannot be created.
        let bad = std::path::Path::new("/nonexistent-sp3-dir/child/out.bin");
        let err = write_bin(bad, &[1u8, 2, 3]).unwrap_err();
        match err {
            crate::error::ConversionError::Io { context, .. } => {
                assert!(context.contains("out.bin"));
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features writer::tests::write_bin_returns_io_error'`
Expected: FAIL — `write_bin` returns `()` and panics (compile error on `.unwrap_err()`).

- [ ] **Step 3: Convert `write_bin`, `run_io_writer`, `run_long_allele_writer`**

Add `use crate::error::ConversionError;` to `src/writer.rs`. Replace the three functions:

```rust
pub fn run_io_writer(
    rx_sparse: Receiver<SparseChunk>,
    dirs: StreamMap<PathBuf>,
    dense_dirs: DenseMap<PathBuf>,
) -> Result<(), ConversionError> {
    while let Ok(chunk) = rx_sparse.recv() {
        let id = chunk.chunk_id;

        for (tag, sub) in chunk.streams.iter() {
            let dir = dirs.get(tag);
            write_bin(
                &layout::chunk_pos(dir, id),
                bytemuck::cast_slice(&sub.call_positions),
            )?;
            write_bin(&layout::chunk_key(dir, id), &sub.call_keys)?;
        }

        for spec in &DENSE_REGISTRY {
            let sub = chunk.dense.get(spec.class);
            if sub.n_dense_variants == 0 {
                continue;
            }
            let dir = dense_dirs.get(spec.class);
            write_bin(&layout::chunk_pos(dir, id), bytemuck::cast_slice(&sub.positions))?;
            write_bin(&layout::chunk_key(dir, id), &sub.keys)?;
            write_bin(&layout::chunk_geno(dir, id), &sub.geno_bits)?;
        }
    }

    println!("Writer Thread: Channel closed. All chunks safely committed to SSD.");
    Ok(())
}

pub fn run_long_allele_writer(
    rx_long: Receiver<Vec<u8>>,
    out_path: &Path,
    chrom_label: &str,
) -> Result<(), ConversionError> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(out_path)
        .map_err(|e| ConversionError::Io {
            context: format!("creating {}", out_path.display()),
            source: e,
        })?;
    let mut disk_writer = BufWriter::with_capacity(1024 * 1024, file);
    while let Ok(buffer) = rx_long.recv() {
        disk_writer.write_all(&buffer).map_err(|e| ConversionError::Io {
            context: format!("writing long alleles to {}", out_path.display()),
            source: e,
        })?;
    }
    disk_writer.flush().map_err(|e| ConversionError::Io {
        context: format!("flushing {}", out_path.display()),
        source: e,
    })?;
    println!(
        "[{}] Long Allele Writer: All buffer data safely committed.",
        chrom_label
    );
    Ok(())
}

fn write_bin(path: &Path, bytes: &[u8]) -> Result<(), ConversionError> {
    let f = File::create(path).map_err(|e| ConversionError::Io {
        context: format!("creating {}", path.display()),
        source: e,
    })?;
    let mut f = BufWriter::new(f);
    f.write_all(bytes).map_err(|e| ConversionError::Io {
        context: format!("writing {}", path.display()),
        source: e,
    })?;
    f.flush().map_err(|e| ConversionError::Io {
        context: format!("flushing {}", path.display()),
        source: e,
    })?;
    Ok(())
}
```

- [ ] **Step 4: Update the writer closures + joins in `orchestrator.rs`**

The chunk-writer closure (line 200) is `move || writer::run_io_writer(...)` — it now returns `Result`, no change to the closure text. Same for the long-allele writer closure (line 199). Update the two joins (lines 229-234):

```rust
    match chunk_writer_res {
        Ok(r) => r?,
        Err(_) => {
            return Err(ConversionError::WorkerPanicked {
                thread: format!("cw-{}", chrom),
            })
        }
    }
    match long_allele_writer_res {
        Ok(r) => r?,
        Err(_) => {
            return Err(ConversionError::WorkerPanicked {
                thread: format!("lw-{}", chrom),
            })
        }
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features writer::tests'`
Expected: PASS (existing writer test + the new one).

Run: `pixi run bash -lc 'cargo build --no-default-features'`
Expected: builds clean.

- [ ] **Step 6: Commit**

```bash
git add src/writer.rs src/orchestrator.rs
git commit -m "fix(writer): return ConversionError::Io from writer threads

run_io_writer/run_long_allele_writer/write_bin propagate I/O failures as typed
Io errors; the orchestrator surfaces them at the writer joins instead of aborting.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 4: Merge / dense-merge / pack → typed errors

**Files:**
- Modify: `src/merge.rs:24-31` (signature) + all I/O `.expect()`/`panic!` sites in `merge_mini_sc`
- Modify: `src/dense_merge.rs` (`merge_dense_class` signature + I/O sites)
- Modify: `src/rvk.rs:23-56` (`pack_snp_key_file`)
- Modify: `src/streams.rs:31` (hook type)
- Modify: `src/orchestrator.rs:256-266`, `:276-284` (merge/pack call sites use `?`)
- Test: `src/rvk.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `ConversionError::{Io, Npy}` (Task 1).
- Produces: `merge_mini_sc(...) -> Result<(), ConversionError>`, `merge_dense_class(...) -> Result<(), ConversionError>`, `pack_snp_key_file(&Path) -> Result<(), ConversionError>`; `StreamSpec::post_merge: Option<fn(&Path) -> Result<(), ConversionError>>`.

- [ ] **Step 1: Write the failing test**

Add to `src/rvk.rs`'s test module (create a `#[cfg(test)] mod tests` if absent, with `use super::*;`):

```rust
    #[test]
    fn pack_snp_key_file_errors_on_missing_dir() {
        let missing = std::path::Path::new("/nonexistent-sp3-pack/dir");
        let err = pack_snp_key_file(missing).unwrap_err();
        match err {
            crate::error::ConversionError::Io { .. } => {}
            other => panic!("expected Io, got {other:?}"),
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features rvk::tests::pack_snp_key_file_errors'`
Expected: FAIL — `pack_snp_key_file` returns `()` and panics (compile error on `.unwrap_err()`).

- [ ] **Step 3: Convert `pack_snp_key_file`**

In `src/rvk.rs`, add `use crate::error::ConversionError;`. Change the signature (line 23) to `pub fn pack_snp_key_file(dir: &Path) -> Result<(), ConversionError> {` and replace every `.expect(...)` with a mapped `?`:

```rust
    let src = layout::alleles(dir);
    let tmp = dir.join("alleles.packed.tmp");

    let mut reader = BufReader::new(File::open(&src).map_err(|e| ConversionError::Io {
        context: format!("opening {}", src.display()),
        source: e,
    })?);
    let mut writer = BufWriter::new(File::create(&tmp).map_err(|e| ConversionError::Io {
        context: format!("creating {}", tmp.display()),
        source: e,
    })?);

    const BLOCK: usize = 4 * 1024 * 1024; // multiple of 4
    let mut buf = vec![0u8; BLOCK];
    loop {
        let mut filled = 0usize;
        while filled < BLOCK {
            let n = reader.read(&mut buf[filled..]).map_err(|e| ConversionError::Io {
                context: format!("reading {}", src.display()),
                source: e,
            })?;
            match n {
                0 => break,
                n => filled += n,
            }
        }
        if filled == 0 {
            break;
        }
        let packed = pack_snp_keys(&buf[..filled]);
        writer.write_all(&packed).map_err(|e| ConversionError::Io {
            context: format!("writing {}", tmp.display()),
            source: e,
        })?;
        if filled < BLOCK {
            break;
        }
    }
    writer.flush().map_err(|e| ConversionError::Io {
        context: format!("flushing {}", tmp.display()),
        source: e,
    })?;
    drop(writer);
    drop(reader);

    std::fs::rename(&tmp, &src).map_err(|e| ConversionError::Io {
        context: format!("renaming {} -> {}", tmp.display(), src.display()),
        source: e,
    })?;
    Ok(())
```

- [ ] **Step 4: Update the `post_merge` hook type in `streams.rs`**

In `src/streams.rs`, change line 31 from:
```rust
    pub post_merge: Option<fn(&Path)>,
```
to:
```rust
    pub post_merge: Option<fn(&Path) -> Result<(), crate::error::ConversionError>>,
```
The `REGISTRY` entry `Some(pack_snp_key_file)` (line 40) now type-checks against the new signature — no change to that line. The `test_only_snp_has_post_merge` test (lines 94-102) still passes (it only checks `is_some()`/`is_none()`).

- [ ] **Step 5: Convert `merge_mini_sc`**

In `src/merge.rs`, add `use crate::error::ConversionError;`. Change the signature (line 31) `) {` → `) -> Result<(), ConversionError> {`. Convert the sequential I/O sites:

- `write_npy(layout::offsets(...), &offsets_array).expect("Failed to write final offsets");` →
  ```rust
  write_npy(layout::offsets(output_dir_path), &offsets_array).map_err(|source| {
      ConversionError::Npy {
          path: layout::offsets(output_dir_path).to_string_lossy().into_owned(),
          source,
      }
  })?;
  ```
- The `File::create(...).expect("Failed to create positions.bin")` + `.set_len(...).expect("Failed to size positions.bin")` and the `alleles.bin` pair → map each to `ConversionError::Io { context, source }` with a `?`, e.g.:
  ```rust
  let final_pos_file = File::create(layout::positions(output_dir_path)).map_err(|e| ConversionError::Io {
      context: "creating positions.bin".to_string(),
      source: e,
  })?;
  final_pos_file.set_len(pos_total_bytes).map_err(|e| ConversionError::Io {
      context: "sizing positions.bin".to_string(),
      source: e,
  })?;
  let final_key_file = File::create(layout::alleles(output_dir_path)).map_err(|e| ConversionError::Io {
      context: "creating alleles.bin".to_string(),
      source: e,
  })?;
  final_key_file.set_len(key_total_bytes).map_err(|e| ConversionError::Io {
      context: "sizing alleles.bin".to_string(),
      source: e,
  })?;
  ```
- The `chunk_files` builder (the `.map(|c| { ... panic!(...) })` closure) → collect into a `Result`:
  ```rust
  let chunk_files: Vec<(File, File)> = (0..num_chunks)
      .map(|c| -> Result<(File, File), ConversionError> {
          let pf = File::open(layout::chunk_pos(output_dir_path, c)).map_err(|e| ConversionError::Io {
              context: format!("opening chunk_{c}_pos.bin"),
              source: e,
          })?;
          let kf = File::open(layout::chunk_key(output_dir_path, c)).map_err(|e| ConversionError::Io {
              context: format!("opening chunk_{c}_key.bin"),
              source: e,
          })?;
          Ok((pf, kf))
      })
      .collect::<Result<_, _>>()?;
  ```

- Convert the parallel gather from `tile_starts.par_iter().for_each(|&tile_start_col| { ... });` to `try_for_each` returning `Result`, and map the four inner `.expect()` I/O sites. The closure's early `return;` (empty tile) becomes `return Ok(());`, and add `Ok(())` at the closure's end:
  ```rust
  tile_starts.par_iter().try_for_each(|&tile_start_col| -> Result<(), ConversionError> {
      // ... unchanged setup ...
      if tile_total_items == 0 {
          return Ok(());
      }
      // ... unchanged buffer/gather setup ...
      // Inside the `for chunk_id in 0..num_chunks` loop, the two preads:
      chunk_files_ref[chunk_id].0.read_exact_at(&mut chunk_pos_bytes, pos_byte_offset)
          .map_err(|e| ConversionError::Io { context: "pread chunk pos".into(), source: e })?;
      chunk_files_ref[chunk_id].1.read_exact_at(&mut chunk_key_bytes, key_byte_offset)
          .map_err(|e| ConversionError::Io { context: "pread chunk key".into(), source: e })?;
      // ... unchanged stitch loop ...
      // The two final pwrites:
      final_pos_ref.write_all_at(tile_pos_bytes, tile_pos_byte_offset)
          .map_err(|e| ConversionError::Io { context: "pwrite positions.bin".into(), source: e })?;
      final_key_ref.write_all_at(&tile_key_buffer, tile_key_byte_offset)
          .map_err(|e| ConversionError::Io { context: "pwrite alleles.bin".into(), source: e })?;
      Ok(())
  })?;
  ```

- After the parallel block, the temp-file cleanup loop and the `println!` finalize stay as-is. Add `Ok(())` as the function's final expression. If any remaining `.expect()` on cleanup `remove_file` exists, leave it (best-effort cleanup on genoray's own temp files is an acceptable invariant) — but if the reviewer prefers, map to `Io` too.

- [ ] **Step 6: Convert `merge_dense_class`**

`src/dense_merge.rs`'s `merge_dense_class` mirrors `merge_mini_sc`'s I/O shape (create/set_len the final files, open chunk files, parallel pread/pwrite, write offsets npy). Apply the identical mapping rule: add `use crate::error::ConversionError;`, change the signature's `) {` → `) -> Result<(), ConversionError> {`, and convert each I/O `.expect(...)`/`panic!(...)` site to a `.map_err(|e| ConversionError::Io { context: "<describe the op>".into(), source: e })?` (use `ConversionError::Npy` for the `write_npy` site, as in Step 5). Convert any `par_iter().for_each` to `try_for_each` returning `Result<(), ConversionError>` exactly as shown in Step 5. Discover the exact sites with:

Run: `grep -n 'expect\|unwrap()\|panic!' src/dense_merge.rs`

and convert each I/O one (leave any genuine-invariant `unwrap` on in-memory data, e.g. `.last().unwrap()` on a known-nonempty vec). End the function with `Ok(())`.

- [ ] **Step 7: Update the merge/pack call sites in `orchestrator.rs`**

Replace the `merge_mini_sc(...)` call + hook (lines 256-266):
```rust
        merge::merge_mini_sc(
            spec.key_bytes,
            num_chunks,
            samples.len(),
            ploidy,
            dir.to_str().unwrap(),
            ledger,
        )?;
        if let Some(hook) = spec.post_merge {
            hook(&dir)?;
        }
```

Replace the `merge_dense_class(...)` call (lines 276-284) — append `?` after the closing `);`:
```rust
        crate::dense_merge::merge_dense_class(
            num_chunks,
            samples.len(),
            ploidy,
            spec.key_bytes,
            spec.pack_snp,
            dir.to_str().unwrap(),
            ledger,
        )?;
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: PASS (rvk pack test + all existing merge/dense_merge tests green).

Run: `pixi run pytest tests/test_svar2_from_vcf.py tests/test_svar2.py -q`
Expected: PASS (happy-path conversion unchanged).

- [ ] **Step 9: Commit**

```bash
git add src/merge.rs src/dense_merge.rs src/rvk.rs src/streams.rs src/orchestrator.rs
git commit -m "fix(merge): return ConversionError from merge/dense-merge/pack paths

merge_mini_sc, merge_dense_class, and pack_snp_key_file propagate I/O failures as
typed errors (parallel tiles via try_for_each); the post_merge hook signature and
orchestrator call sites thread the Result through.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 5: Tighten the existing symbolic-ALT test

**Files:**
- Modify: `tests/test_svar2_from_vcf.py:96-100`

- [ ] **Step 1: Replace the loose assertion**

The existing `test_from_vcf_symbolic_errors_without_skip` uses `pytest.raises(Exception)`. Now that symbolic ALT raises a typed `ValueError`, tighten it. Replace:

```python
    with pytest.raises(Exception):
        SparseVar2.from_vcf(tmp_path / "store_err", vcf, ref, threads=1)
```

with:

```python
    with pytest.raises(ValueError, match="symbolic"):
        SparseVar2.from_vcf(tmp_path / "store_err", vcf, ref, threads=1)
```

- [ ] **Step 2: Run the test**

Run: `pixi run pytest tests/test_svar2_from_vcf.py::test_from_vcf_symbolic_errors_without_skip -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar2_from_vcf.py
git commit -m "test(svar2): assert symbolic ALT raises ValueError, not bare Exception

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Commit 3 — Query sidecar + FFI

### Task 6: Query sidecar loaders → `io::Result`

**Scope boundary:** convert the setup-time loaders (`load_offsets`/`load_max_del`/`load_dense_max_del`) and `LongAlleleReader::new`, all of which are reached through `ContigReader::open -> io::Result` and surface as `PyOSError`. The per-call `LongAlleleReader::get_allele`/`all_bytes` preads stay `.expect()` — they run in the decode hot path on genoray's own finished files; making them fallible would ripple `Result` through the entire decode/gather chain for a can't-happen I/O fault with no user-facing benefit.

**Files:**
- Modify: `src/query/sidecar.rs:94-121` (three loaders)
- Modify: `src/nrvk.rs:105-119` (`LongAlleleReader::new`)
- Modify: `src/query/reader.rs` (callers of the loaders + `LongAlleleReader::new`)
- Test: `src/query/sidecar.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces: `load_offsets(path, columns) -> std::io::Result<Vec<u64>>`, `load_max_del(path, n, p) -> std::io::Result<Array2<u32>>`, `load_dense_max_del(path) -> std::io::Result<u32>`, `LongAlleleReader::new(dir, chrom) -> std::io::Result<Self>`.

- [ ] **Step 1: Write the failing test**

Add to `src/query/sidecar.rs` (create a `#[cfg(test)] mod tests { use super::*; ... }` if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_offsets_missing_file_is_empty_prefix_sum() {
        let p = std::path::Path::new("/nonexistent-sp3/offsets.npy");
        let v = load_offsets(p, 3).unwrap();
        assert_eq!(v, vec![0u64; 4]);
    }

    #[test]
    fn load_offsets_corrupt_file_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("offsets.npy");
        std::fs::write(&p, b"not a valid npy header").unwrap();
        assert!(load_offsets(&p, 3).is_err());
    }
}
```

(If `tempfile` is not already a dev-dependency, it is — the writer tests use `tempfile::tempdir`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features query::sidecar::tests'`
Expected: FAIL — `load_offsets` returns `Vec<u64>` and `.expect()`s on corrupt input (compile error on `.unwrap()`/`.is_err()`).

- [ ] **Step 3: Convert the three loaders**

In `src/query/sidecar.rs`, replace lines 94-121:

```rust
/// Load a CSR `offsets.npy` (len `columns + 1`); a missing file means an empty
/// stream — return an all-zero prefix-sum so every column is empty.
pub(crate) fn load_offsets(path: &Path, columns: usize) -> std::io::Result<Vec<u64>> {
    if path.exists() {
        let a: Array1<u64> = ndarray_npy::read_npy(path)
            .map_err(|e| std::io::Error::other(format!("read {}: {e}", path.display())))?;
        Ok(a.to_vec())
    } else {
        Ok(vec![0u64; columns + 1])
    }
}

/// Load `max_del.npy` (`u32`, shape `(n_samples, ploidy)`); a missing file
/// (pure-SNP contig, or predating the post-pass) defaults to all-zero.
pub(crate) fn load_max_del(
    path: &Path,
    n_samples: usize,
    ploidy: usize,
) -> std::io::Result<Array2<u32>> {
    if path.exists() {
        ndarray_npy::read_npy(path)
            .map_err(|e| std::io::Error::other(format!("read {}: {e}", path.display())))
    } else {
        Ok(Array2::zeros((n_samples, ploidy)))
    }
}

/// Load `dense/max_del.npy` (`u32`, shape `(1,)`); missing defaults to `0`.
pub(crate) fn load_dense_max_del(path: &Path) -> std::io::Result<u32> {
    if path.exists() {
        let a: Array1<u32> = ndarray_npy::read_npy(path)
            .map_err(|e| std::io::Error::other(format!("read {}: {e}", path.display())))?;
        Ok(a.into_iter().next().unwrap_or(0))
    } else {
        Ok(0)
    }
}
```

- [ ] **Step 4: Convert `LongAlleleReader::new`**

In `src/nrvk.rs`, change `pub fn new(output_dir: &str, chrom: &str) -> Self {` (line 106) to `-> std::io::Result<Self> {` and rewrite:

```rust
    pub fn new(output_dir: &str, chrom: &str) -> std::io::Result<Self> {
        let paths = ContigPaths::new(output_dir, chrom);
        let file = File::open(paths.long_alleles_bin())?;
        let offsets_array: ndarray::Array1<u64> = ndarray_npy::read_npy(paths.long_allele_offsets())
            .map_err(|e| std::io::Error::other(format!("read long_allele_offsets.npy: {e}")))?;
        Ok(Self {
            file,
            offsets: offsets_array.into_raw_vec_and_offset().0,
        })
    }
```

Also remove the stale `// TODO: Decide which will call this ...` comment above it (the caller is now `ContigReader::open`).

- [ ] **Step 5: Propagate at the callers in `query/reader.rs`**

`ContigReader::open` already returns `io::Result`. Update its loader/`new` call sites to `?`:

- The `load_offsets(...)` / `load_max_del(...)` / `load_dense_max_del(...)` calls each gain a `?` (they now return `io::Result`).
- The LUT construction at `src/query/reader.rs:75` — `Some(LongAlleleReader::new(base_out_dir, chrom))` — becomes `Some(LongAlleleReader::new(base_out_dir, chrom)?)`. Confirm the surrounding expression is inside the `-> io::Result<...>` `open` body; if the `Some(...)` is produced by an `if`/`match` arm, keep the arm shape and only add `?` inside `Some(...)`.

Discover any loader call sites you may have missed with:

Run: `grep -rn 'load_offsets\|load_max_del\|load_dense_max_del\|LongAlleleReader::new' src/query/`

and add `?` to each (all are inside `open`, which returns `io::Result`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: PASS (new sidecar tests + all existing query/decode tests green).

Run: `pixi run pytest tests/test_svar2_ranges.py tests/test_svar2_decode.py -q`
Expected: PASS (query happy path unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/query/sidecar.rs src/nrvk.rs src/query/reader.rs
git commit -m "fix(query): thread io::Result through sidecar loaders and LUT open

load_offsets/load_max_del/load_dense_max_del and LongAlleleReader::new return
io::Result; ContigReader::open surfaces a corrupt/truncated sidecar as PyOSError
instead of aborting. Per-call get_allele/all_bytes preads stay panics by design.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 7: `bundle_from_dict` → `PyResult`

**Files:**
- Modify: `src/py_query_ranges.rs:139-201` (`bundle_from_dict`) + its call sites (`read_ranges`/`gather_ranges`)
- Test: `tests/test_svar2_errors.py` (extend)

**Interfaces:**
- Produces: `bundle_from_dict(d) -> PyResult<RangesBundle>`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar2_errors.py`:

```python
def test_gather_ranges_malformed_bundle_raises_keyerror(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(
        tmp_path,
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n",
    )
    out = tmp_path / "store"
    SparseVar2.from_vcf(out, vcf, ref, threads=1)
    sv = SparseVar2(out)

    bundle = sv.find_ranges("chr1", [0], [40])
    del bundle["sample_cols"]  # drop a required key
    with pytest.raises(KeyError, match="sample_cols"):
        sv.gather_ranges("chr1", bundle)
```

(If `SparseVar2(out)` is not the opening idiom, mirror how `tests/test_svar2_ranges.py` opens a store for querying and adapt the two query calls accordingly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar2_errors.py::test_gather_ranges_malformed_bundle_raises_keyerror -v`
Expected: FAIL — the missing key currently triggers a Rust `unwrap` panic (`PanicException`/abort), not a clean `KeyError`.

- [ ] **Step 3: Convert `bundle_from_dict`**

In `src/py_query_ranges.rs`, add the imports (near the existing pyo3 imports): `use pyo3::exceptions::{PyKeyError, PyTypeError};`. Replace the function (lines 139-201):

```rust
/// Inverse of `bundle_to_dict`: read a `find_ranges` dict back into a
/// `RangesBundle` for `gather_ranges`. Fallible: a missing key or wrong
/// dtype/shape becomes a Python KeyError/TypeError rather than a Rust panic.
fn bundle_from_dict(d: &Bound<'_, PyDict>) -> PyResult<RangesBundle> {
    let require = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(k)?
            .ok_or_else(|| PyKeyError::new_err(format!("bundle missing key '{k}'")))
    };
    let get_i32 = |k: &str| -> PyResult<Vec<i32>> {
        let obj = require(k)?;
        let arr = obj
            .cast::<PyArray1<i32>>()
            .map_err(|_| PyTypeError::new_err(format!("bundle key '{k}' must be an int32 1D array")))?;
        Ok(arr.readonly().as_slice()?.to_vec())
    };
    let get_i64 = |k: &str| -> PyResult<Vec<i64>> {
        let obj = require(k)?;
        let arr = obj
            .cast::<PyArray1<i64>>()
            .map_err(|_| PyTypeError::new_err(format!("bundle key '{k}' must be an int64 1D array")))?;
        Ok(arr.readonly().as_slice()?.to_vec())
    };
    let get_i32_pairs = |k: &str| -> PyResult<Vec<Range<usize>>> {
        let obj = require(k)?;
        let arr = obj
            .cast::<PyArray2<i32>>()
            .map_err(|_| PyTypeError::new_err(format!("bundle key '{k}' must be an int32 (N,2) array")))?
            .readonly();
        Ok(arr
            .as_array()
            .rows()
            .into_iter()
            .map(|row| (row[0] as usize)..(row[1] as usize))
            .collect())
    };
    let get_i64_pairs = |k: &str| -> PyResult<Vec<Range<usize>>> {
        let obj = require(k)?;
        let arr = obj
            .cast::<PyArray2<i64>>()
            .map_err(|_| PyTypeError::new_err(format!("bundle key '{k}' must be an int64 (N,2) array")))?
            .readonly();
        Ok(arr
            .as_array()
            .rows()
            .into_iter()
            .map(|row| (row[0] as usize)..(row[1] as usize))
            .collect())
    };
    let get_usize = |k: &str| -> PyResult<usize> { require(k)?.extract() };

    Ok(RangesBundle {
        n_regions: get_usize("n_regions")?,
        n_samples: get_usize("n_samples")?,
        ploidy: get_usize("ploidy")?,
        region_starts: get_i32("region_starts")?
            .into_iter()
            .map(|x| x as u32)
            .collect(),
        dense_range: get_i32_pairs("dense_range")?,
        sample_cols: get_i64("sample_cols")?
            .into_iter()
            .map(|x| x as usize)
            .collect(),
        vk_snp_range: get_i64_pairs("vk_snp_range")?,
        vk_indel_range: get_i64_pairs("vk_indel_range")?,
        dense_snp_range: get_i32_pairs("dense_snp_range")?,
        dense_indel_range: get_i32_pairs("dense_indel_range")?,
    })
}
```

- [ ] **Step 4: Propagate at the call sites**

Find the `bundle_from_dict(...)` call(s) inside the `#[pymethods]` (`gather_ranges`, and any other consumer). Each is inside a `-> PyResult<...>` method, so add `?`:

Run: `grep -n 'bundle_from_dict' src/py_query_ranges.rs`

Change each `let rb = bundle_from_dict(&d);` (or inline use) to `let rb = bundle_from_dict(&d)?;`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run bash -lc 'cargo build --no-default-features'`
Expected: builds clean.

Run: `pixi run pytest tests/test_svar2_errors.py -v`
Expected: PASS (all error tests, including the new KeyError case).

Run: `pixi run pytest tests/test_svar2_ranges.py -q`
Expected: PASS (valid bundles still round-trip).

- [ ] **Step 6: Commit**

```bash
git add src/py_query_ranges.rs tests/test_svar2_errors.py
git commit -m "fix(ffi): make bundle_from_dict fallible (KeyError/TypeError not panic)

A malformed find_ranges bundle (missing key, wrong dtype) now raises a clean
Python KeyError/TypeError at the FFI boundary instead of aborting via unwrap.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 8: Document the exception contract in SKILL.md

**Files:**
- Modify: `skills/genoray-api/SKILL.md`

- [ ] **Step 1: Add an Errors subsection**

Add a short subsection (place it near the SparseVar2 conversion/query docs; match the file's existing heading style):

```markdown
### Errors

genoray raises standard Python builtins, by category:

- `ValueError` — bad input content: contig/sample not found, REF disagrees with
  the reference FASTA, or a symbolic/breakend ALT with `skip_out_of_scope=False`.
- `FileNotFoundError` — a required input file is missing.
- `OSError` — a corrupt/truncated store sidecar or an underlying disk I/O failure.
- `RuntimeError` — an internal genoray bug (a worker thread panicked); please
  report it.
```

- [ ] **Step 2: Verify the suite is fully green**

Run: `pixi run test`
Expected: PASS (full Python suite).

Run: `pixi run bash -lc 'cargo test --no-default-features'`
Expected: PASS (full Rust suite).

- [ ] **Step 3: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(skill): document the genoray exception-type contract

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes (for the executor)

- **Every `cargo test` invocation MUST use** `pixi run bash -lc 'cargo test --no-default-features ...'` — a bare `cargo test` fails to link the pyo3 test binary.
- If `cargo clippy` (a prek hook) flags the new `try_for_each` closure's `ConversionError` not being `Send`, re-check that no variant carries a non-`Send` source — all current variants are `Send`.
- Tasks 2, 6, and 7 each add observable Python behavior with a dedicated pytest; Tasks 3 and 4 are I/O-on-own-files conversions verified by a Rust unit test plus the suite staying green.
- The `dense_merge.rs` (Task 6, step 6) and call-site (`grep`) steps intentionally use discovery commands because the sites are a mechanical repeat of the fully-shown `merge_mini_sc` pattern — apply the identical `.map_err(|e| ConversionError::Io { context, source: e })?` rule to each.

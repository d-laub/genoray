# PGEN → SVAR2 conversion

Date: 2026-07-12
Status: approved (design), ready for implementation planning

## Problem

`SparseVar2` can only be built from a VCF/BCF (`SparseVar2.from_vcf`). PLINK2
PGEN is the format cohort- and biobank-scale genotype data actually ships in, so
SVAR2 needs a `from_pgen` path. Two things make this non-trivial:

1. **Licensing.** The PGEN decoder that matters (plink-ng / plink 2.0) is
   copyleft. genoray is MIT. We must not create an obligation we cannot meet.
2. **Performance and memory.** PGEN's reason for existing is large `n_samples`.
   The current conversion pipeline has a staging buffer whose size scales as
   `chunk_size × n_samples × ploidy × 4 bytes`, which is not representable at
   biobank scale.

## Licensing position

| Component | License | How genoray uses it |
| --- | --- | --- |
| genoray | MIT | — |
| `pgenlib` (PyPI) | LGPL-3.0 | **Already** a hard runtime dependency; unmodified, separately-installed wheel, used through its public Python API |
| plink-ng `2.0/include/pgenlib_*.cc` | LGPL-3.0 | not linked, not vendored |
| plink2 application | GPL-3.0 | invoked as a separate program at *test* time only (already in `pixi.toml`) |

This design **does not change how genoray consumes plink code**. It keeps using
the `pgenlib` Python wheel through its public API, which imposes no copyleft
obligation on genoray's own source or on the compiled `_core` extension. The
`_core` extension continues to contain zero plink-ng code and stays MIT.

Rejected on licensing grounds:

- **Statically linking plink-ng's C++ `pgenlib` into `_core`.** This creates a
  combined work and triggers LGPLv3 §4: prominent notice, license texts, and
  *either* dynamic linking *or* shipping relinkable object code with every
  wheel. That is an ongoing compliance burden on our release CI (which builds
  four portable wheel platforms) for a decoder we already ship via pip.

Rule for implementers: **do not copy code or comments from plink-ng into
genoray.** If a native decoder is ever written, it must be written from
`pgen_spec.pdf`, and that provenance must be documented.

## Rejected alternative: a native Rust PGEN decoder

`genoio` (github.com/mancusolab/genoio, MIT) has a from-spec Rust PGEN decoder.
Its Rust sources contain no reference to plink-ng, pgenlib, Purcell, or Chang,
and read as independent work — the MIT label appears accurate. It is still not
usable here:

- **Multiallelic decode is not implemented** and is an explicit hard error
  (`pgen/header.rs`: "unsupported pgen multiallelic hard-call patch set"); its
  `.pvar` parser silently keeps only the first ALT.
- **No allele-code API.** Everything funnels into an f32 dense matrix. There is
  no equivalent of `read_alleles_range`, and haplotype reads *error* on an
  unphased het rather than reporting a phasepresent mask.
- Storage modes 0x01 and 0x11 are rejected.
- Not published to crates.io (path-only inter-crate deps, no `description`), and
  `genoio-io` pulls in noodles + bundled SQLite with no cargo features to slim
  it.

Writing our own equivalent means ~3k lines of difflist / LD-compression / aux-track
bit-twiddling with real correctness risk, to replace a decoder that is already a
dependency and is faster (genoio self-reports pgenlib at 1.4–1.8× its speed).
Not worth it. Revisit only if the GIL boundary below is measured to be a
bottleneck.

## Why pgenlib is the right decoder

`pgenlib.PgenReader.read_alleles_range(start, end, out, hap_maj=0)`:

- fills an `(n_variants, 2 · n_samples)` **int32** buffer with **per-haplotype
  allele codes**, phase-aware, multiallelic-aware, missing = `-9`;
- runs its decode loop under `prange(..., nogil=True)` — **the GIL is released**
  for the actual work, so a Rust reader thread holds it only for call dispatch.

The pipeline's `PendingAtom.gt` is a `Vec<i32>` of length
`n_samples × ploidy` holding allele index per haplotype column (`-1` = missing).
That is the same layout `read_alleles_range` produces. PGEN therefore needs only
a new *record source*; normalization, left-alignment, atom decomposition, the
executor, merge, and finalize are all reused unchanged.

## Architecture

### 1. `RecordSource` refactor (prerequisite, VCF-neutral)

`VcfChunkReader` currently fuses five responsibilities: htslib record iteration,
record → atom decomposition (REF validation, left-alignment, multiallelic split,
out-of-scope skipping), the `L_MAX` reorder heap, chunk assembly and presence
bit-packing, and INFO/FORMAT field staging. Only the first is VCF-specific.

Introduce:

```rust
pub struct RawRecord<'a> {
    pos: u32,               // 0-based
    reference: &'a [u8],
    alts: &'a [&'a [u8]],
    gt: &'a [i32],          // len = n_samples * ploidy; -1 = missing
    info_vals: &'a [f64],   // empty for PGEN
    format_vals: &'a [f64], // empty for PGEN
}

pub trait RecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord<'_>>, ConversionError>;
}
```

Everything downstream of `next_record` moves into a source-agnostic
`ChunkAssembler` that owns the heap, the decomposition, and chunk packing.
`VcfRecordSource` wraps htslib; `PgenRecordSource` is new.
`process_chromosome` becomes generic over the source (or takes a
`Box<dyn RecordSource>`).

Consequence: **PGEN inherits REF validation, indel left-alignment, and symbolic /
breakend skipping for free, with semantics identical to the VCF path.**

This lands as its own commit, with the existing test suite green and
**byte-identical** SVAR2 output on a VCF fixture, before any PGEN code is added.

### 2. `PgenRecordSource`

Owns:

- a `Py<PyAny>` handle to a `pgenlib.PgenReader` (constructed in Python, passed
  down), and a reusable `(B, 2·S)` int32 numpy buffer;
- a streaming `.pvar` reader for variant metadata.

`next_record()` serves from the buffer; on exhaustion it refills via
`Python::attach(|py| reader.call_method1("read_alleles_range", (lo, hi, buf, false)))`.
The GIL is held only for call dispatch — pgenlib's decode is `nogil`. Allele
codes are mapped `-9 → -1` to match the `PendingAtom.gt` convention.

**Batch size is derived from a byte budget, not a variant count:**

```
B = clamp(PGEN_BATCH_BYTES / (2 * S * 4), 1, chunk_size)   # PGEN_BATCH_BYTES = 32 MiB
```

At 200 samples this is thousands of variants per Python call (overhead
amortized); at 500k samples it is ~8 (memory bounded). This is the single knob
that keeps the Rust↔Python boundary cheap across the whole cohort-size range.

**Variant metadata.** Rust streams `.pvar` / `.pvar.zst` with a small TSV parser
(adds the `zstd` crate — BSD/MIT). `.bim` is not supported, since it only
accompanies a PLINK1 `.bed`, which is out of scope. Python computes the
per-contig `[var_start, var_end)` variant index ranges up front using the
**existing** `_scan_pvar` polars scan (which already handles header sniffing and
`.zst`) and passes only those ranges to Rust; the Rust streamer skips to
`var_start` by line count.

Known cost: with `concurrent_chroms > 1`, each contig's reader skips through the
`.pvar` from the top, so total `.pvar` I/O is `n_contigs × pvar_size`. This is
sequential scan I/O, parallel across contigs, and is expected to be negligible
next to the genotype decode. **Measure it in the benchmark.** If it shows up,
the mitigation is a one-time contig byte-offset sidecar — do not build that
speculatively.

Sample names come from the existing `_read_psam`.

### 3. Bounded-window packing (shared with the VCF path, in this PR)

`read_next_chunk` today collects `chunk_size` `PendingAtom`s into a `Vec` and
packs them all at the end. Each atom holds an `S × P` i32 `gt` vector — **32×
the size of the packed bit-grid it ultimately produces**. At
`chunk_size = 25_000` and 10k samples that staging buffer is ~2 GB; at biobank
scale it is not representable.

Fix: **pack in sub-blocks.** Buffer `W` atoms (`W ≈ 1024`), pack them in parallel
into the growing `BitGrid3` at their variant offset, drop their `gt`, and
continue. Staging memory becomes `W · S · P · 4` instead of
`chunk_size · S · P · 4`, and parallel packing (previously measured worth −6.7%)
is retained. The `L_MAX = 1000` reorder heap already bounds the sort window, so
atom ordering is unaffected and output is byte-identical.

This is a shared improvement: it benefits `from_vcf` identically and is what
makes large-`n_samples` conversion feasible on either backend.

Additionally, `from_pgen` auto-derives `chunk_size` from a memory budget when the
caller does not supply one, since the packed dense chunk is itself
`chunk_size · S · P / 8` bytes. `from_vcf`'s default (`25_000`) is unchanged.

## Public API

```python
SparseVar2.from_pgen(
    out: str | Path,
    source: str | Path,
    reference: str | Path | None = None,
    *,
    no_reference: bool = False,
    skip_out_of_scope: bool = False,
    chunk_size: int | None = None,   # None => auto from a memory budget
    threads: int | None = None,
    overwrite: bool = False,
    long_allele_capacity: int = 8 * 1024 * 1024,
    signatures: bool = False,
) -> int   # number of out-of-scope ALTs dropped
```

- `source` is a `.pgen` path; `.pvar[.zst]` and `.psam` siblings are resolved by
  stem.
- **No `ploidy` parameter.** PGEN is diploid; exposing the knob would only create
  an invalid state to reject.
- `reference` / `no_reference` are mutually exclusive and behave exactly as in
  `from_vcf`. `signatures=True` requires a reference, as in `from_vcf`.
- CLI: `genoray write` dispatches on the `.pgen` suffix (its help text already
  claims VCF/PGEN).
- `skills/genoray-api/SKILL.md` is updated in the same PR (CLAUDE.md requires it
  for any public-name change).
- `CHANGELOG.md` gains an entry under `## Unreleased`.

### Out of scope for v1 (documented in the docstring)

- **Dosages.** SVAR2 has no dosage store. A `.pgen` dosage track is ignored;
  hardcalls are still read correctly.
- **`info_fields` / `format_fields`.** PGEN has no FORMAT; `.pvar` INFO
  extraction is deferred.
- **Sample subsetting / reordering.** Matches `from_vcf`, which also converts all
  samples.
- **PLINK1 `.bed`.**

### Semantics to document

Haplotype resolution for **unphased** heterozygotes follows the allele-code order
pgenlib returns. This is the same caveat the VCF path carries for unphased `GT`,
and is exactly what `SparseVar.from_pgen` (SVAR1) already does. Missing calls are
`-9` in pgenlib and are mapped to `-1`.

## Testing

1. **Primary oracle — pgenlib itself.** Write an SVAR2 store from a PGEN, decode
   it back to `(samples, ploidy, variants)` allele indices via the existing query
   API, and compare against `pgenlib.PgenReader.read_alleles_list` on the same
   variant indices. Exact, and independent of the VCF path.
2. **Cross-backend equivalence.** Generate PGEN twins of the existing VCF
   fixtures with `plink2 --vcf ... --make-pgen` (plink2 is already in
   `pixi.toml`) and assert `from_pgen ≡ from_vcf` on curated fixtures covering
   phased, multiallelic, and indel records. Where plink2's VCF import legitimately
   differs (symbolic ALTs, unphased-het haplotype order), compare genotype
   presence rather than raw store bytes.
3. **Rust unit tests** for the `.pvar` streamer: header sniffing, `.zst`,
   multiallelic ALT splitting, missing codes, `[var_start, var_end)` skipping.
4. **Refactor guard:** the `RecordSource` commit must produce byte-identical
   SVAR2 output for an existing VCF fixture.
5. **Bounded-window packing guard:** byte-identical output before/after, plus a
   peak-RSS assertion (or measurement) showing staging memory no longer scales
   with `chunk_size`.
6. **Benchmark** PGEN vs VCF conversion on the same cohort. Expectation: the
   reader stops being the bottleneck (VCF conversion is currently
   htslib-input-bound). Record the numbers in the CHANGELOG entry.

## Risks

- **Refactor regresses the VCF path.** Mitigated by landing it as an isolated
  commit gated on byte-identical output.
- **GIL contention across concurrent contigs.** Each contig's reader briefly
  attaches to call `read_alleles_range`. The decode is `nogil`, so contention is
  on call dispatch only. If the benchmark shows readers serializing, the batch
  byte budget is the first knob; a native decoder is the last resort.
- **pgenlib's `prange` uses OpenMP if its wheel was built with it**, which could
  oversubscribe alongside `concurrent_chroms`. Check thread counts in the
  benchmark; cap via `OMP_NUM_THREADS` if needed. Note that the PGEN path needs
  no htslib decompression threads, so those cores are available to give back.

# SVAR2 — `SparseVar2.from_vcf_list` (native k-way merge of single-sample VCFs)

> **Status:** design approved, pre-implementation · **Epic:** SVAR 2.0
>
> Adds a constructor that builds one SVAR2 store from a *collection* of
> single-sample VCFs/BCFs whose variant sites differ, merging them natively
> (no `bcftools merge`, no temp multi-sample VCF) by streaming normalization +
> a k-way join on canonical atom identity.

## Context

`SparseVar2.from_vcf` (`python/genoray/_svar2.py:91`) converts **one** multi-sample
VCF: its reader emits, per contig, a stream of records that already carry a
genotype column for every sample, which the pipeline packs into a shared-sites
`BitGrid3` (`(variants, samples, ploidy)`) before splitting into the dense /
`var_key` representations.

A common upstream shape is instead **N single-sample VCFs**, one per sample,
independently called, with **differing site lists**. Building a joint SVAR2 store
from these requires unioning sites across files and filling non-carriers — a
merge the current pipeline does not perform (it reads a single `RecordSource`).

`bcftools merge` is the off-the-shelf tool but (a) is slow enough at cohort scale
that we maintain a separate parallelized Nextflow wrapper for it, and (b) would
materialize a dense multi-sample VCF on disk, defeating SVAR2's whole purpose.
So we merge **natively**.

### Key architectural facts (verified against the code)

- **Ingest is a shared-sites union matrix.** Each atom carries a `gt` vector of
  length `num_samples * ploidy`; `pack_row` (`src/chunk_assembler.rs:54`) sets a
  presence bit iff `gt[col] == source_alt_index`. A site cannot exist without a
  genotype column for every sample. Sparsity (rare variant → single hap's
  `var_key` stream) is *within* that shared column space, not across files.
- **Normalization is already reusable.** `normalize::atomize_record`,
  `normalize::left_align`, `normalize::validate_ref` (`src/normalize.rs:52,111,85`)
  are pure functions producing `Atom { pos, ilen, alt, source_alt_index }`,
  independent of genotype-matrix assembly. `ChunkAssembler` only *calls* them.
- **Ordering after left-align** is maintained by a min-heap reorder buffer + a
  frontier gate (`ChunkAssembler::next_atom`, `src/chunk_assembler.rs:377-395`):
  an atom is only released once `top.pos < frontier - L_MAX` (`L_MAX = 1000`,
  `src/normalize.rs:33`). This is the ~20-line piece we factor into a shared
  helper so each per-file merge stream can reuse it.
- **var_key** (`classify_variant`/`pack_variant`, `src/rvk.rs:126,99`) is pure on
  `(ilen, alt)` and computable in isolation — **but long insertions
  (`alt_len > 13`) spill to an allocation-order-dependent bank**, so their packed
  key is *not* stable across files. The stable cross-file identity is the
  canonical atom tuple `(POS, ILEN, ALT-bytes)`.
- **Idempotent re-normalization.** A biallelic, atomic, left-aligned record in
  ordinary VCF REF/ALT form passes through `atomize_record`/`left_align` as a
  proven fixpoint (`src/normalize.rs` proptest). So the merge reader can safely
  hand already-normalized records to the *unchanged* `ChunkAssembler`: its
  re-atomize/re-left-align/`validate_ref` are harmless no-ops. The reference is
  still passed downstream (not `None`) because `signatures` classification needs
  genomic context; skipping normalization there would be a micro-optimization
  that breaks signatures, so we do not.

Consequence: the merge slots in as a **new `RecordSource`** feeding the
**unchanged** `ChunkAssembler` → `dense2sparse_vk` → writer.

## Public API

```python
@classmethod
def from_vcf_list(
    cls,
    out: str | Path,
    sources: str | Path | Sequence[str | Path],
    reference: str | Path | None = None,
    *,
    no_reference: bool = False,
    skip_out_of_scope: bool = False,
    ploidy: int = 2,
    chunk_size: int = 25_000,
    threads: int | None = None,
    overwrite: bool = False,
    long_allele_capacity: int = 8 * 1024 * 1024,
    signatures: bool = False,
    info_fields: Sequence[str | InfoField] | None = None,
    format_fields: Sequence[str | FormatField] | None = None,
) -> int:
```

Signature mirrors `from_vcf` exactly, replacing `source: str | Path` with
`sources: str | Path | Sequence[str | Path]`. Returns the number of out-of-scope
(symbolic/breakend) ALTs dropped, same as `from_vcf`. All keyword semantics
(`reference`/`no_reference` exactly-one-of, `signatures` requires a reference,
`overwrite`, `ploidy`, `chunk_size`, `threads`, `long_allele_capacity`) carry
over unchanged.

Every input **must be single-sample**. The store's `samples` list is the header
sample name of each file, in resolved input order.

### `sources` resolution

Resolved in Python (`_svar2.py`) into an ordered `list[Path]`:

| `sources` value | Interpretation |
| --- | --- |
| `Sequence[str \| Path]` | Explicit ordered list, used as-is. |
| single path, **directory** | Non-recursive glob `*.vcf.gz` + `*.bcf`, `natsorted`. |
| single path, **file not ending `.vcf.gz`/`.bcf`** | Newline-delimited manifest: strip whitespace, skip blank lines and `#` comments; relative entries resolved against the manifest's parent dir. |
| single path, `.vcf.gz`/`.bcf` file | Degenerate one-element list. |

## Merge architecture

New `SourceSpec::VcfList { vcf_paths, htslib_threads }` (`src/orchestrator.rs`)
dispatching to a new `VcfListRecordSource` (`src/record_source.rs` impl, new
module e.g. `src/vcf_list_reader.rs`). Downstream is **byte-for-byte the existing
path**: `ChunkAssembler` (receiving the reference as usual — re-normalization is
an idempotent no-op, and the reference must stay available for `signatures`),
`dense2sparse_vk`,
writer, field finalize, `meta.json`.

Per contig, `VcfListRecordSource`:

1. Opens one htslib cursor per input file **that lists this contig** (files
   without the contig are skipped for this contig).
2. Wraps each cursor in a per-file **normalize-and-reorder stage**: for each raw
   record, `validate_ref` → `atomize_record` → per-atom `left_align` against the
   reference, buffered through the **shared heap + `L_MAX` frontier gate** (lifted
   from `ChunkAssembler::next_atom`) so each file yields atoms monotonic in
   `(POS, ILEN, ALT)`. With `no_reference`, `left_align`/`validate_ref` are
   skipped exactly as in `from_vcf`.
3. **K-way merges** the N per-file atom iterators with a min-heap keyed by the
   canonical identity **`(POS, ILEN, ALT-bytes)`**.
4. At each distinct atom, builds the merged `gt` vector (length `N·ploidy`,
   sample-major, ploidy-minor):
   - all columns initialized to `0` (**hom-ref fill** for files that don't carry
     the atom);
   - each carrier file writes allele code `1` (the ALT) into its `ploidy`
     columns, or `-1` for a within-file `./.` (missing), preserving per-sample
     phase as the existing pipeline does.
5. Emits one **biallelic, atomic, left-aligned `RawRecord`** in ordinary VCF
   REF/ALT form (`source_alt_index` therefore `1`, matching `pack_row`).

### Join-key rationale

The join identity is `(POS, ILEN, ALT-bytes)` rather than `(POS, var_key)`:
var_key is stable for SNPs, inline indels, and pure DELs, but long insertions
(`alt_len > 13`) encode as `(bank_row_index << 1) | 1` with an allocation-order
row index that differs across files. ALT-bytes is the content var_key encodes and
is universally stable. (var_key may still be used as a cheap same-position
pre-filter; ALT-bytes decides identity.)

### Refactor surface

- **New:** `src/vcf_list_reader.rs` (`VcfListRecordSource`), `SourceSpec::VcfList`
  arm in `orchestrator.rs`, a `run_vcf_list_conversion_pipeline` pyfunction in
  `src/lib.rs`, and `from_vcf_list` in `python/genoray/_svar2.py`.
- **Shared helper:** the heap + `L_MAX` frontier monotonicity gate factored out
  of `ChunkAssembler::next_atom` into a reusable struct both paths call. This is
  the *only* change to existing Rust; `normalize.rs`, `dense2sparse_vk`, writer,
  and `ChunkAssembler`'s core are otherwise untouched.

## Semantics

- **Absent site → hom-ref (`0/0`).** A file not listing an atom is treated as
  homozygous reference there.
- **Within-file `./.` → missing (`-1`).** Distinct from absent-site fill.
- **Reference** drives left-alignment *inside the merge reader* so atoms from
  different files converge to identical POS before the join; it is also passed to
  the downstream `ChunkAssembler` (idempotent re-normalization, and required for
  `signatures` context). `no_reference=True` keeps
  `from_vcf`'s "trust inputs as pre-normalized" contract, documented to require
  consistently pre-normalized inputs for cross-file joins to line up.
- **`format_fields`** (per-sample): merged exactly — each carrier contributes its
  sample's value; non-carriers get the field default/missing.
- **`info_fields`** (per-site): **first-carrier wins** — the value from the first
  input file (in list order) carrying the atom. Documented; re-deriving cohort
  INFO from merged genotypes is out of scope.
- **`signatures`**: unchanged from `from_vcf` (requires a reference; classified on
  the merged atom stream).

## Scale and limits

The merge holds N htslib readers open per contig; with `concurrent_chroms`
contigs in flight, open file descriptors ≈ `N · concurrent_chroms`, which can
exceed `RLIMIT_NOFILE` (~1024) for large cohorts.

**MVP mitigation:** probe the soft `RLIMIT_NOFILE`, reserve headroom, and shrink
`concurrent_chroms` so `N · concurrent_chroms` stays under it (down to 1 contig
at a time for very large N). The k-way merge and packing still parallelize within
a contig.

**Future extension (not in this PR):** a hierarchical batched merge — merge B
files into intermediate SVAR2 stores, then merge stores — for N beyond what one
process can hold open. Noted as the same ceiling the external
`nf-bcftools-par-merge` pipeline works around.

## Errors

Fail fast with clear messages for:

- an input file whose header sample count ≠ 1 (not single-sample);
- duplicate sample names across files (list the collisions);
- empty `sources`, or a directory/manifest resolving to zero files;
- a manifest entry that does not exist;
- the usual exactly-one-of `reference`/`no_reference`; `signatures` +
  `no_reference`; `out` exists without `overwrite`.

## Contig discovery

`contigs` = `natsorted` union of the input files' header `seqnames` that have at
least one variant in at least one file (mirrors `from_vcf`'s
`next(v(c), None) is not None` check). Per contig, only files whose header lists
the contig are opened.

## Verification

- **bcftools oracle.** For a small synthetic cohort of single-sample VCFs,
  `bcftools merge` + `+setGT -- -t . -n 0` (missing→ref) into one multi-sample
  VCF, run through the existing `from_vcf`, produces a reference store; assert
  `from_vcf_list` decodes byte-identically (positions, ILENs, ALT bytes, per-hap
  genotypes, and any `format_fields`).
- **Targeted cases:** multiallelic site split across files; indel that
  left-aligns to the same POS from different files; long-INS join (ALT-bytes
  identity, not packed key); within-file `./.` vs absent-site fill; phased vs
  unphased inputs; a file missing an entire contig; a sample entirely hom-ref on
  a contig.
- **`info_fields` first-carrier** value matches the first list-order carrier.
- **Input resolution:** directory, manifest (with comments/blank lines/relative
  paths), and explicit `Sequence` all resolve to the same store.
- **Error paths:** each error condition above raises the expected exception.

## Docs (mandatory)

Per `CLAUDE.md`, the same PR updates `skills/genoray-api/SKILL.md` with
`from_vcf_list` (signature, `sources` forms, single-sample requirement, hom-ref
fill, first-carrier INFO) and adds a `## Unreleased` entry to `CHANGELOG.md`.

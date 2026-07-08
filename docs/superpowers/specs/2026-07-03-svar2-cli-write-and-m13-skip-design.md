# SVAR2 in the `genoray write` CLI + M13 skip-out-of-scope — design

> **Status:** approved design · **Branch:** work in a new worktree off `svar-2` ·
> **Home:** `genoray` · **Roadmap:** completes [M13](../../roadmap/svar-2.md#beyond-mvp)
> and adds a SVAR2 conversion entry point to the CLI + a Python `SparseVar2.from_vcf`.

## Goal

Two coupled deliverables:

1. **M13 — opt-in skip for out-of-scope alleles during conversion.** Today the Rust
   conversion core treats symbolic (`<DEL>`, `<INS:ME:*>`, …) and breakend ALTs as a
   hard error: `normalize::atomize_record` returns `SymbolicAllele` and
   `vcf_reader::decompose_current_record` `.expect()`s on it, so one out-of-scope record
   aborts the whole conversion. Add an **opt-in** skip mode that drops such records
   (exactly as `*`/`.` alleles are already skipped) and reports how many were dropped.
   The strict default (error) stays.

2. **SVAR2 writing in the `genoray` CLI.** `genoray write` currently only produces
   SVAR 1.0. Make `genoray write` default to **SVAR2** (expected to be better across the
   board) and add `genoray write svar1` for SVAR 1.0. This requires a Python-facing
   `SparseVar2.from_vcf` wrapper around the existing `run_conversion_pipeline` PyO3
   entry point, since none exists yet.

Work is done in a fresh git worktree under `.claude/worktrees`. It spans Rust (rebuilt
via maturin), Python, and docs.

## Non-goals

- PGEN → SVAR2 (roadmap M7), dosages, `--haploid` OR-collapse, and `max_mem`-based
  chunking for SVAR2. The `run_conversion_pipeline` engine has no path for these; they
  remain SVAR1-only.
- Splitting symbolic vs. breakend detection. The Rust `is_symbolic` check covers both
  classes; M13 treats them as one out-of-scope class.
- Verifying left-alignment without a reference. Not possible reference-free; see the
  reference-optional contract below.

## Background: what the reference is actually used for

`normalize::atomize_biallelic` is **entirely reference-free** — it right-trims the shared
suffix and derives every atom (including the pure-DEL anchor base) from the record's own
REF/ALT. The reference FASTA is consumed only by:

- `normalize::validate_ref` — fail-fast REF-vs-FASTA consistency check, and
- `normalize::left_align` — repeat-shifting anchored indels leftward.

Therefore a reference-optional conversion is clean: with no reference we still atomize +
right-trim, and we simply skip validation and left-alignment.

## Component 1 — Reference-optional conversion (Rust)

- `run_conversion_pipeline`'s `reference_path: String` becomes `Option<String>`;
  threaded as `Option<&str>` through `orchestrator::process_chromosome` into
  `vcf_reader::VcfChunkReader`.
- When `Some(path)`: unchanged — open the FASTA, run `validate_ref` and `left_align` per
  record (today's behavior).
- When `None`: do **not** open the FASTA; in `decompose_current_record` skip both
  `validate_ref` and `left_align`, pushing each atomized primitive at its as-given
  (right-trimmed) position. **No per-variant panic** — the input is trusted to be
  pre-normalized.
- The "you must have a reference unless you opt out" contract lives at the CLI /
  wrapper (see Component 4), not in the core.

## Component 2 — M13 opt-in skip (Rust)

- `run_conversion_pipeline` gains `skip_out_of_scope: bool` (default `false`), threaded
  through `process_chromosome` → `VcfChunkReader`.
- `normalize::atomize_record` gains the flag plus a `&mut u64` (or returned) drop counter.
  On an out-of-scope ALT:
  - if `skip_out_of_scope` — increment the counter and `continue` (per-ALT skip, mirroring
    the existing `*`/`.` handling), rather than returning `Err`;
  - else — return the existing `NormalizeError::SymbolicAllele` (strict default preserved).
- `vcf_reader::decompose_current_record` stops `.expect()`ing the symbolic/breakend error;
  it accumulates the per-contig drop count on the reader struct.
- `process_chromosome` returns the per-contig drop count (extend its `Ok` payload from
  `()` to `usize`); `run_conversion_pipeline` sums across contigs.
- `run_conversion_pipeline`'s return type changes from `PyResult<()>` to
  `PyResult<usize>` — the total number of dropped out-of-scope ALT alleles.
- REF-mismatch and genotype-decode `.expect()`s stay hard errors — out of M13 scope.

## Component 3 — Auto-index helper (Rust + wrapper)

- The `vcf_reader` fetches per-contig via a `.tbi`/`.csi` index, so an un-indexed source
  must be indexed first.
- Add a small PyO3 helper (e.g. `index_vcf(path: str)`) built on the existing
  `rust-htslib` dependency (`bcf::index::build`), so indexing needs **no external
  binary**. Registered in `_core` alongside `run_conversion_pipeline`.
- The Python wrapper calls it when the source lacks an index. A plain **uncompressed**
  `.vcf` cannot be tabix/csi-indexed → the wrapper raises a clear error telling the user
  to bgzip first (auto-bgzip is out of scope).

## Component 4 — Python wrapper: `SparseVar2.from_vcf`

New classmethod in `python/genoray/_svar2.py`, mirroring `SparseVar.from_vcf`, that the
CLI calls. Responsibilities:

- Derive `samples` from the VCF header and `contigs` from header `seqnames`
  (natural-sorted), reusing `genoray.VCF`.
- Enforce the reference contract: exactly one of `reference` / `no_reference=True`. Raise
  a clear `ValueError` if neither is provided (or both).
- Auto-index the source if it has no `.tbi`/`.csi` (Component 3); raise on an
  uncompressed `.vcf`.
- Validate `out` vs. `overwrite`.
- Call `_core.run_conversion_pipeline(...)` with `reference_path=None` when
  `no_reference`, and `skip_out_of_scope` from the coupled skip flags.
- Return the dropped-allele count so the CLI can report it.

Signature (final names TBD in the plan, shape fixed here):

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
) -> int:  # number of dropped out-of-scope ALT alleles
    ...
```

**Open point (resolve in the plan via TDD):** a reference-style header (e.g. full hg38)
lists contigs with zero records. The plan must confirm the orchestrator tolerates a
zero-variant contig (htslib `fetch` on a header contig with no index entries yields an
empty iterator, so no panic is expected); if it does not, filter empty contigs in the
wrapper before calling the pipeline.

## Component 5 — CLI restructure

`write` becomes a two-command group (no existing CLI test covers `write`, so this is a
clean move):

- `genoray write <source> <out> …` → **SVAR2** via `@write.default`.
- `genoray write svar1 <source> <out> …` → today's `write` body, moved verbatim.

**SVAR2 command surface:**

- `source` (bgzipped VCF or BCF, indexed or not — auto-indexed), `out` (directory).
- `--reference PATH` **XOR** `--no-reference` — exactly one required; error if neither.
- `--no-symbolic` / `--no-breakend` — same names as svar1 for parity. On svar2 they are
  **coupled** (the core does not split the classes): passing either enables the single
  `skip_out_of_scope`. Documented explicitly in the help text.
- `--ploidy` (default 2), `--threads`, `--chunk-size` (default 25000), `--overwrite`,
  and an advanced `--long-allele-capacity`.
- On success, print the dropped-allele count when skipping was enabled, e.g.
  `Dropped 1368 out-of-scope (symbolic/breakend) ALT alleles.`

**SVAR1 command** = today's `write` body unchanged: VCF+PGEN, `dosages`, `max_mem`,
`--haploid`, and its existing `--no-symbolic`/`--no-breakend` cyvcf2/polars filter.

## Testing

- **Rust:** unit/proptests for `atomize_record` skip + count (strict default still errors;
  skip drops per-ALT and counts); the `reference=None` path (atomize + right-trim, no
  validate/left-align); the `index_vcf` helper builds a usable index. An e2e conversion
  of a symbolic-bearing VCF across the matrix {skip on/off} × {reference / no-reference},
  asserting the dropped count and that non-skip + symbolic still errors.
- **Python:** CLI tests for both `write` (svar2) and `write svar1`, including the
  reference/`--no-reference` XOR error, auto-index of an un-indexed input, the coupled
  skip flags, and a round-trip readable by `SparseVar2`.

## Surfaces to update (repo rules)

- `skills/genoray-api/SKILL.md` — new `SparseVar2.from_vcf`, the CLI `write` / `write
  svar1` split, the changed `run_conversion_pipeline` signature/return, and the new
  `index_vcf` helper.
- `docs/roadmap/svar-2.md` — tick M13.
- This spec + the forthcoming implementation plan.

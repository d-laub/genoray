# Design: `check_ref` for all SVAR2 conversion methods (issue #116)

Date: 2026-07-15

## Problem

Every SVAR2 conversion validates each record's REF against the reference FASTA
and **hard-aborts the entire build** on the first disagreement:

```
ValueError: REF 'G' at pos 46405660 disagrees with reference FASTA ('T')
```

For a large k-way merge (e.g. ~7,000 single-sample somatic VCFs), a single
malformed record — SAGE/PURPLE occasionally emits a REF that disagrees with the
genome in low-complexity/repeat regions — kills the whole build and leaves a
partial, unusable store. The cost of one bad record is the entire cohort merge.

The check lives in `src/normalize.rs::validate_ref` and is invoked at two sites:

- `chunk_assembler.rs:320` — shared by `from_vcf`, `from_pgen`, `from_svar1`
  (single-source spine: atomize with the record's REF, then left-align).
- `vcf_list_reader.rs:115` — `from_vcf_list` (k-way merge; `atomize_to_vcf_biallelic`
  reconstructs REF bytes from the FASTA).

Both propagate the error → hard abort. Validation runs only when a reference is
supplied (`no_reference` skips it entirely).

## Goal

Add a `check_ref` policy mirroring a subset of `bcftools norm --check-ref` to
**all four** conversion entry points:

- `"e"` (error) — current behavior; abort on disagreement. **Default** (non-breaking).
- `"x"` (exclude) — drop the offending record and continue.

`"w"` (warn/keep) and `"s"` (set REF from FASTA) are **out of scope** (see below).

REF comparison is already ASCII-case-insensitive in `validate_ref`, so
soft-masked (lowercase) reference bases match uppercase VCF REF — no change
needed there.

## Architecture — one shared decision point (DRY)

Both call sites currently do `validate_ref(...)?`. Replace both with a single
shared helper in `src/normalize.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckRef {
    /// Abort on REF/FASTA disagreement (current behavior).
    Error,
    /// Drop the offending record and continue.
    Exclude,
}

/// Per-record outcome of applying a `CheckRef` policy.
pub enum RefDecision {
    /// REF matches — process the record normally.
    Keep,
    /// REF disagrees under `Exclude` — skip this whole record.
    Exclude,
}

/// Apply the `CheckRef` policy to one record. Returns `Err` only under
/// `CheckRef::Error` (propagating `RefMismatch` / `RefOutOfContig`); under
/// `CheckRef::Exclude` a disagreement becomes `Ok(RefDecision::Exclude)`.
pub fn apply_check_ref(
    mode: CheckRef,
    pos: u32,
    ref_allele: &[u8],
    ref_seq: &[u8],
) -> Result<RefDecision, NormalizeError> {
    match validate_ref(pos, ref_allele, ref_seq) {
        Ok(()) => Ok(RefDecision::Keep),
        Err(e) => match mode {
            CheckRef::Error => Err(e),
            CheckRef::Exclude => Ok(RefDecision::Exclude),
        },
    }
}
```

Behavior:

- Under `Error`, the `?` propagates exactly today's `RefMismatch` /
  `RefOutOfContig` → byte-identical to current behavior.
- Under `Exclude`, both a base mismatch **and** `RefOutOfContig` (REF runs past
  the contig end) drop the record and continue — both are cases of "this record
  disagrees with this reference," which is precisely what a robust cohort merge
  wants to skip.

Each call site becomes:

```rust
match apply_check_ref(mode, pos, &rec.reference, ref_seq)? {
    RefDecision::Keep => { /* proceed: atomize + left-align as today */ }
    RefDecision::Exclude => { self.ref_excluded += 1; /* skip record */ }
}
```

- `chunk_assembler::decompose_record`: on `Exclude`, increment the counter and
  `return Ok(())` before atomizing — the record contributes no atoms.
- `vcf_list_reader::FileCursor::advance`: on `Exclude`, increment the counter and
  skip the atomize/buffer-fill block, falling through to the existing
  `Ok(self.buf.pop_front())`. That returns `None` while `!eof`, and the caller
  re-advances — the exact path already used when a record atomizes to zero atoms
  (all-`*`/dropped ALTs). No new control flow in the merge loop.

## Threading

`check_ref` flows top-down:

1. 4 pyo3 functions in `src/lib.rs`
   (`run_conversion_pipeline`, `run_pgen_conversion_pipeline`,
   `run_vcf_list_conversion_pipeline`, `run_svar1_conversion_pipeline`)
   gain a `check_ref: &str` parameter in their `#[pyo3(signature = ...)]`.
2. → the orchestrator / pipeline bodies → `process_chromosome`.
3. → `ChunkAssembler::new` and `VcfListRecordSource::new` (stored as a
   `CheckRef` field), and to `FileCursor::advance`.

Python passes the string `"e"` / `"x"`; Rust parses it into `CheckRef` (via
`TryFrom<&str>` / `FromStr`), erroring on an unknown value. The enum is the
single source of truth for the mapping; the Python `Literal` is the typed
surface. Extending to `"w"`/`"s"` later requires no signature change.

`check_ref` is only consulted when a reference is present; `no_reference` builds
are unaffected (validation never runs).

## Reporting

The excluded-record count is surfaced via the existing `println!("Notice: ...")`
pattern used throughout the pipeline:

- a per-contig count, logged as the pipeline finishes each contig,
- the **first** offending locus (`pos REF vs FASTA`, from the carried
  `NormalizeError`) logged once per source when the first exclusion occurs.

Bounded output — never one line per excluded record. A cross-contig grand total
is intentionally **not** emitted: it would require changing `process_chromosome`'s
internal return type (rippling through all four pipeline callers and the
thread-join plumbing) for marginal value over the per-contig lines, which the
user can trivially sum.

**Return type is unchanged:** each `from_*` still returns `int` = out-of-scope
(symbolic/breakend) ALTs dropped. Rationale: a second count cannot be added
without a breaking change to the return type, and stdout logging matches how the
pipeline already reports progress and the `skip_out_of_scope` behavior. If
programmatic access to the exclusion count is wanted later, a small stats object
is a future (breaking) release.

## Python / CLI / docs surface

- Add `check_ref: Literal["e", "x"] = "e"` (keyword-only) to `from_vcf`,
  `from_pgen`, `from_vcf_list`, `from_svar1`. Validate the literal, pass through.
  Docstrings updated to explain that `"x"` drops REF/FASTA disagreements
  (including out-of-contig) and continues.
- CLI `genoray write` (the svar2 default command, which covers VCF + PGEN): add
  `--check-ref {e,x}` (default `e`) and a closing "Notice" line, mirroring the
  existing `--skip-symbolics-and-breakends` treatment. (`from_vcf_list` and
  `from_svar1` have no CLI entry point today; no CLI change for them.)
- **`skills/genoray-api/SKILL.md` updated** — required by CLAUDE.md, since a
  public keyword is added to four public methods.

## Testing

- **Rust unit** (`src/normalize.rs`): `apply_check_ref` — `Error` propagates both
  `RefMismatch` and `RefOutOfContig`; `Exclude` returns `RefDecision::Exclude`
  for mismatch and out-of-contig, `RefDecision::Keep` on match.
- **Rust e2e** (new `tests/test_check_ref_e2e.rs`, mirroring
  `tests/test_convert_skip_e2e.rs`): a tiny VCF with one bad-REF record plus a
  tiny FASTA → `check_ref="e"` errors; `check_ref="x"` produces a store missing
  exactly that variant and retaining the rest. Covers the shared
  chunk_assembler path; a `from_vcf_list` variant covers the merge path.
- **Python** (extend `tests/test_svar2_from_vcf.py`, `_from_pgen`,
  `_from_vcf_list`, `_from_svar1`): reproduce issue #116's `REF=G`-where-FASTA-is-
  `T` record (a T-homopolymer insertion); assert `check_ref="e"` raises and
  `check_ref="x"` succeeds, excluding only the offending record while keeping a
  legitimate adjacent variant.

## Out of scope

- `"w"` (warn/keep): in genoray a kept record can't be trusted for left-alignment
  against a FASTA it disagrees with, so "keep as-is" would mean processing it in
  no-reference mode — muddy semantics for little benefit over `"x"` in the
  cohort-merge use case.
- `"s"` (set REF from FASTA): a footgun for indels (rewriting REF changes the
  variant; bcftools itself only reliably fixes SNVs and may silently drop
  indels), and the REF-reconstruction differs between the two call sites.

Both can be added later without changing the signature (the `check_ref` string
surface and the `CheckRef` enum are already extensible).

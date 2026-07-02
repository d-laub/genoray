# SVAR 2.0 — M2b: left-alignment during conversion (design)

> Spec for roadmap milestone **M2b** in
> [`docs/roadmap/svar-2.md`](../../roadmap/svar-2.md).
> Supplements: [`data-model.md`](../../roadmap/data-model.md#variant-normalization).
> Branch off `svar-2`, own worktree. **Parallel-safe** with both M5 specs (the `max_del`
> post-pass and the `(range, sample)` query) — this spec edits the **read/normalize**
> side (`vcf_reader.rs`, `normalize.rs`) while those edit the write/query side. It changes
> variant *start positions* but not deletion *lengths* or the on-disk schema, so it does
> not invalidate M5's inputs.

## Context

M2 shipped atomization + biallelic splitting (`src/normalize.rs`), leaving indels
**anchored but not left-aligned**. M2b shifts each indel to its leftmost equivalent
position, matching bcftools `norm`. It was deferred from M2 because it is the only
normalization step that needs a **reference genome** (a new required conversion argument)
and it **weakens the reorder buffer's ordering premise** — `src/vcf_reader.rs::next_atom`
already flags this:

> "M2b (left-alignment) weakens the `pos >= record_start` premise and must revisit this
> bound."

Today an atom is safe to emit once `top.pos < frontier` (the read frontier = current
record's start), because without left-alignment an atom's position is always `>=` its
record's start. Left-alignment can move an indel's start **below** its record's start, so
the emit-safety condition must widen by the maximum possible leftward shift.

## Scope

**In:**

- A **reference FASTA (faidx)** input, threaded as a new **required** argument through the
  conversion entrypoint (`src/orchestrator.rs` and the PyO3 binding). Random-access reads
  via `rust_htslib::faidx` (htslib is already a dependency).
- **Left-alignment of anchored indel atoms** in the normalize path: after `atomize_record`
  produces anchored INS/DEL/SNP atoms, roll each indel leftward while the shift is
  reference-consistent (classic VCF left-align / trimming roll). SNP atoms never move.
- **Reorder-buffer bound widening** in `vcf_reader.rs`: introduce a bounded left-shift
  window `L_MAX` and change the heap emit-safety test to
  `top.pos < frontier.saturating_sub(L_MAX)`, keeping the position-sorted invariant the
  Phase-2 merge relies on. Shifts are capped at `L_MAX` (matching bcftools' windowed
  `--buffer-size` behavior) so memory stays bounded.
- **Contig-boundary handling**: stop rolling at the contig's first base; never read
  reference outside `[0, contig_len)`.
- **Genotype remapping unchanged**: left-alignment mutates an atom's `pos` and its anchor
  base(s) only; `source_alt_index` and the shared `gt` vector are untouched, so the
  existing haplotype remap still holds.

**Out:**

- Any change to the encoding, cost model, dense path, on-disk layout, or `max_del`
  (deletion lengths are invariant under left-alignment — see below).
- Multi-threaded reference sharing beyond what per-contig conversion already needs (one
  faidx handle / cached contig sequence per contig worker).
- Right-alignment, trimming beyond the existing atomize suffix-trim, or reference-mismatch
  *correction* (a REF that disagrees with the FASTA is an error, not silently fixed).

## Why this stays orthogonal to M5

`max_del` / `overlap_range` bound how far **left of the query start** a spanning deletion
can begin, via the maximum **deletion length**. Left-alignment changes a deletion's
*start position* but never its *length* (the number of deleted reference bases is
invariant). The bound is applied relative to each deletion's own (post-alignment) start,
so overlap correctness is preserved with no coordination. The only shared concern is that
M5's `max_del` post-pass should run over the **final left-aligned** output — which it does
by construction (it is a post-pass over finished streams).

## Design

- **Reference access.** Open the FASTA with `rust_htslib::faidx::Reader` once; per contig
  worker, fetch the needed reference bases (either a cached full contig sequence when it
  fits, or windowed `fetch_seq` calls around each indel — **measure**; cache-per-contig is
  simplest and usually fine for one contig at a time).
- **Left-align routine** (`normalize.rs`, pure given a reference-byte accessor):
  `left_align(atom, ref_accessor) -> Atom`. For an anchored indel at `pos` with indel
  sequence `s` (inserted bases for INS, deleted bases for DEL):
  roll left while `pos > contig_start`, the shift count `< L_MAX`, and
  `ref[pos-1] == s[last]` — decrement `pos`, rotate `s`, and update the anchor base from
  `ref[pos]`. This is the standard "shared-suffix / repeat" roll; validate against a
  couple of bcftools-normalized fixtures.
- **Ordering.** Left-align each atom **after** atomization, **before** pushing to the heap
  in `decompose_current_record`. The heap key stays `atom.pos` (now the aligned pos).
- **Buffer bound.** Add `L_MAX` (const or config). Update `next_atom`'s emit guard to
  `top.pos < frontier.saturating_sub(L_MAX)` and update the memory-note comment. Choose
  `L_MAX` from the empirical short-read indel/STR distribution — **measure**, don't pick by
  convention (per the project's "measure, don't guess" principle); document the default and
  the truncation semantics for shifts that would exceed it (atom left partially aligned,
  as bcftools does at its buffer limit).
- **Errors.** REF disagreeing with the FASTA at `pos` ⇒ a new `NormalizeError` variant
  (fail fast). Out-of-contig fetch ⇒ same. Symbolic/breakend/`*`/`.` handling unchanged.

## Files

- **Touch:** `src/normalize.rs` (`left_align`, new error variant), `src/vcf_reader.rs`
  (faidx handle, call `left_align`, widen `next_atom` bound + comment),
  `src/orchestrator.rs` (thread the FASTA path), the PyO3 binding (new required arg),
  `Cargo.toml` (enable `rust-htslib` faidx feature if not already on).
- **Docs:** update
  [`data-model.md`](../../roadmap/data-model.md#variant-normalization) and the M2b
  checkbox per the roadmap working agreement.

## Testing

- **Unit (`normalize.rs`):** `left_align` on hand-built cases — a deletion in a homopolymer
  run (rolls fully), a deletion at contig start (stops at boundary), an insertion in a
  tandem repeat, a SNP (no move), and a shift that would exceed `L_MAX` (partial align).
- **Proptest:** random reference + random indel; assert the aligned variant is
  reference-equivalent to the original (re-expanding both against the reference yields the
  same alt sequence) and is genuinely leftmost within `L_MAX`.
- **bcftools cross-check:** for a small fixture VCF+FASTA, assert atoms match
  `bcftools norm -a -m- -f ref.fa` positions/alleles (modulo the documented
  substituted-deletion-anchor deviation already noted in `normalize.rs`).
- **Reorder invariant:** proptest that emitted atom positions are non-decreasing across
  chunk boundaries even with left-shifts up to `L_MAX` (the property the widened bound
  must preserve).
- **e2e:** convert an un-left-aligned VCF with and without the reference; assert indels
  move to expected positions and a round-trip still reconstructs the correct haplotypes.

## Open questions

- **`L_MAX` value and units** — a fixed base window vs. tied to the max observed repeat.
  Measure on representative short-read VCFs; default conservative, document truncation.
- **Reference caching strategy** — full-contig cache vs. windowed `fetch_seq`. Start with
  full-contig-per-worker; revisit if peak RAM at cohort scale is a problem.
- **Making the FASTA optional** — should conversion still allow skipping left-alignment
  (already-normalized input)? Recommendation: keep the arg **required** for M2b to avoid a
  silent "not actually normalized" footgun; a future flag can opt out explicitly.

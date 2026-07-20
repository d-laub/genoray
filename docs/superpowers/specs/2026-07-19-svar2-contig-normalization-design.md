# SVAR2 contig-name normalization

Make every place `SparseVar2` matches a caller-supplied contig name robust to
alternative naming schemes (`10` ↔ `chr10`, and the `{M, MT, chrM, chrMT}` mito
aliases), consistent with `genoray._contigs.ContigNormalizer`.

## Problem

`ContigNormalizer` bridges `chr`-prefix and mito-alias differences, and genoray
already uses it in most places a contig name crosses a boundary. Two surfaces
still match contig names by exact string and therefore break on a naming-scheme
mismatch:

### 1. Reader methods on a finished store (Python)

The query/subset methods on `SparseVar2` look a contig up directly in
`self._readers` / `self.contigs`:

| Method | File | Current behavior on `10` vs a `chr10` store |
| --- | --- | --- |
| `decode(contig, …)` | `_svar2_decode.py` | `self._readers[contig]` → bare `KeyError` |
| `region_counts(contig, …)` | `_svar2_decode.py` | bare `KeyError` |
| `read_ranges(contig, …)` | `_svar2_batch.py` | bare `KeyError` |
| `_overlap_batch` / `_find_ranges` / `_gather_ranges` | `_svar2_batch.py` | bare `KeyError` |
| `subset_contigs(contigs=…)` | `_svar2.py` | `ValueError: contigs not in store` |
| `annotate_mutations(contigs=…)` | `_svar2_mutcat.py` | **silently** empty scope — annotates nothing, no error |

The conversion/`write_view` *region* inputs are already robust: they funnel
through `_normalize_svar2_regions`, which builds a `ContigNormalizer` over the
source contigs. This part of the design does **not** touch them.

### 2. Reference-FASTA lookup during conversion (Rust)

`reference=` is passed to Rust as a path. The single choke point that reads
per-contig reference bytes, `vcf_reader::load_contig_seq(fasta_path, chrom)`
(and its fail-fast sibling `validate_contigs_in_fasta`), calls htslib's
`fetch_seq_len(chrom)` / `fetch_seq(chrom, …)` with the **raw** variant/store
contig string. htslib returns `u64::MAX` for an unknown contig, so an
unprefixed variant source against a `chr`-prefixed FASTA raises:

```
Contig '10' not found in reference FASTA
```

This is the observed failure: `SparseVar2.from_svar1(reference=…)` on a `10`-named
source against a `chr10` FASTA, during left-alignment/variant normalization. The
same path backs `from_vcf` / `from_pgen` with `reference=` and conversion-time
`from_vcf(signatures=True)` annotation.

Python's `Reference` class **already** normalizes (it builds a `ContigNormalizer`
over the FASTA's own contigs in `__init__`), so post-hoc
`annotate_mutations(reference=…)` works. The Rust conversion path bypasses
`Reference` entirely, so this is an existing inconsistency between the two
annotate paths, not merely a missing feature.

## Design

Two independent parts under one theme: *anywhere a caller-supplied contig name
is matched, resolve it with the same equivalence rule `ContigNormalizer` uses.*

### Part A — Python reader-side normalization

Build one `ContigNormalizer` over the store's contigs in `SparseVar2.__init__`
and route every reader-side contig argument through it.

**`SparseVar2.__init__`** (`_svar2.py`): after `self.contigs` is set, add

```python
from genoray._contigs import ContigNormalizer
self._cnorm = ContigNormalizer(self.contigs)
```

**Two helper methods on `SparseVar2`:**

```python
def _resolve_contig(self, contig: str) -> str:
    """Resolve a caller contig name to the store's spelling (chr-prefix and
    mito-alias insensitive). Raise a clear ValueError if it matches no contig."""
    norm = self._cnorm.norm(contig)
    if norm is None:
        raise ValueError(
            f"Contig {contig!r} not found in store; available: {self.contigs}"
        )
    return norm

def _reader(self, contig: str):
    """The PyContigReader for `contig`, resolving alternative naming schemes."""
    return self._readers[self._resolve_contig(contig)]
```

`self._cnorm.norm` already returns the store's canonical spelling (or `None`),
and the store contigs are the `ContigNormalizer` targets, so `norm(store_spelling)`
is the identity and `norm(other_spelling)` maps to the store spelling.

**Call-site changes** — replace `self._readers[contig]` with `self._reader(contig)` in:

- `_svar2_decode.py`: `decode`, `region_counts`
- `_svar2_batch.py`: `_overlap_batch`, `read_ranges`, `_find_ranges`, `_gather_ranges`

The mixins declare the attributes/methods they borrow from the host for
isolated type-checking (they already declare `_readers`); add a `_reader`
method stub / `_resolve_contig` declaration to each mixin's host-provided
section so `self._reader(...)` type-checks there.

**Plural contig arguments** (a small `_resolve_contigs` helper, or inline
`_resolve_contig` in a comprehension):

- `subset_contigs(output, contigs, …)` (`_svar2.py`): resolve each requested
  contig via `_resolve_contig` before the membership check, so
  `wanted`/`missing`/`kept` all compare store spellings. A genuinely-absent
  contig still raises `ValueError`, now with the same clear message.
- `annotate_mutations(…, contigs=…)` (`_svar2_mutcat.py`): resolve each
  requested contig, then intersect with `self.contigs`. **Behavior change:** if
  a caller passes `contigs=` and none resolve to a store contig, raise a
  `ValueError` instead of silently annotating nothing.

`split_by_contig` iterates `self.contigs` (no caller input) — unchanged.
`concat` matches contigs across independently-produced stores by exact string;
that is a distinct "are these the same store's contigs" concern, out of scope.

**Behavior changes (all strictly more permissive):**

1. Reader methods accept any equivalent spelling. An unresolvable contig now
   raises a clear `ValueError` listing available contigs, where `decode` /
   `region_counts` / `read_ranges` / the batch helpers previously raised a bare
   `KeyError`. This error type was never documented.
2. `subset_contigs(contigs=…)` accepts equivalent spellings.
3. `annotate_mutations(contigs=…)` accepts equivalent spellings and raises on an
   all-miss request (previously a silent no-op).

### Part B — Rust reference-FASTA normalization

Resolve the query contig against the FASTA's **own** naming scheme at the single
Rust choke point, mirroring `ContigNormalizer`.

**New helper** in `vcf_reader.rs` (or a small dedicated module):

```rust
/// Resolve `query` to the FASTA's own spelling of the same contig, mirroring
/// Python's `ContigNormalizer`: exact match, then `chr`-prefix add/strip, then
/// the {M, MT, chrM, chrMT} mito-alias group. Returns None if no FASTA contig
/// is equivalent.
fn resolve_fasta_contig(fasta: &faidx::Reader, query: &str) -> Option<String>
```

Implementation:

- Enumerate the FASTA's contigs via `faidx::Reader::seq_names()` (verified
  present in rust-htslib 1.0; falls back to `n_seqs()` + `seq_name(i)` if needed).
- Build the resolution the same way `ContigNormalizer.__init__` builds
  `contig_map`: for each FASTA name `c`, register `c → c`, plus `c[3:] → c` when
  `c` starts with `chr`, plus `chr{c} → c` otherwise; then, if any FASTA contig
  is in `{M, MT, chrM, chrMT}`, map all four aliases to it. Exact matches win
  over derived ones (same precedence as the Python dict-merge order).
- Look up `query` in that map.

**Route both fetchers through it** (`vcf_reader.rs`):

- `load_contig_seq`: resolve `chrom` → FASTA spelling before `fetch_seq_len` /
  `fetch_seq`. On `None`, keep the existing `Contig '{chrom}' not found in
  reference FASTA` error (report the caller's spelling).
- `validate_contigs_in_fasta`: resolve each `chrom` the same way; error message
  stays byte-identical so the fail-fast raise remains indistinguishable from the
  per-contig loop's (as the current doc-comment promises).

This fixes `from_svar1` / `from_vcf` / `from_pgen` with `reference=` **and**
conversion-time `from_vcf(signatures=True)`, and brings the Rust path in line
with Python's `Reference`.

**Tradeoff — accepted:** the normalization rule is reimplemented in Rust because
the conversion hot path cannot call Python's `ContigNormalizer`. This is a
drift risk against `genoray._contigs`. Mitigation: a parity test (below) pins the
Rust resolver to `ContigNormalizer`'s behavior on a shared case table. The
alternative — threading Python-resolved names through the FFI — is more invasive
for no functional gain and is rejected.

## Testing

### Part A (Python, `tests/test_svar2_*`)

New reader-alias tests, mirroring the existing mito-alias case pattern:

- Build a `chr`-prefixed store; query with unprefixed contigs (and a
  `1`-prefixed store queried with `chr1`) across `decode`, `region_counts`,
  `read_ranges`, `subset_contigs`, and `annotate_mutations(contigs=…)`; assert
  results byte-identical to the native spelling.
- A mito-alias query (`MT` against a `chrM` store) resolves.
- A genuinely-absent contig raises `ValueError` with the available-contigs
  message (covers the `KeyError → ValueError` change).
- `annotate_mutations(contigs=<all-miss>)` raises (covers the silent-no-op fix).

### Part B (Rust + Python)

- Rust unit test for `resolve_fasta_contig`: `chr10 ↔ 10`, mito aliases, exact
  match, and a miss (`None`).
- **Parity test** asserting the Rust resolver agrees with `ContigNormalizer` on a
  shared table of `(fasta_contigs, query, expected)` cases — the anti-drift
  guard for the reimplemented rule.
- Python e2e: a `chr`-named FASTA + an unprefixed-contig svar1/VCF source;
  `from_svar1(reference=…)` (and `from_vcf(…, reference=…, signatures=True)`)
  succeeds and produces a result byte-identical to the matched-naming run.

Run: `cargo test --no-default-features` for the Rust tests (per the repo's test
convention) and `pixi run test` for the Python suite. Rebuild the extension
(`maturin develop --release`) before any Python-level verification of the Rust
change — `pixi run test` does not rebuild the `.so`.

## Non-goals

- `_check_consistent_contig_naming` in `from_vcf_list` (mixed naming *across*
  input files) stays strict — a deliberate, separate guard.
- `SparseVar2.concat`'s exact-string contig matching across independent stores.
- Conversion/`write_view` region inputs (already normalized via
  `_normalize_svar2_regions`).

## Docs

Update `skills/genoray-api/SKILL.md`:

- Reader `contig` arguments (`decode`, `region_counts`, `read_ranges`,
  `subset_contigs`, `annotate_mutations`) accept alternate naming schemes
  (`chr`-prefix and mito aliases), resolved via `ContigNormalizer`.
- A `reference=` FASTA may use a different contig naming scheme than the variant
  source; genoray resolves it (mito aliases included).
